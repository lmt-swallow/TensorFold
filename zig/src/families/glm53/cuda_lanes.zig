//! GLM-5.3-Flash streams behind the lane core (core/lanes) on two ranks: rank 0 runs the round loop and sends each
//! backend call to rank 1 as a step record before both run it (a dsv41-style follower), so the ranks make the same
//! calls in the same order and their all-gathers pair. Every rank agrees on each record first (its sha256 pieces
//! all-gathered), so ranks that would run different steps stop before their gathers cross.
//!
//! Each stream owns a sequence (cuda_engine.zig `Seq`: the backbone's states and caches, the MTP head's cache and
//! the rows it owes) in one of `max_streams` slots; the window and prompt buffers are shared. A round's windows run
//! one stream after another, each as that stream alone would run it, so a stream's tokens never depend on the
//! others'. Greedy decoding only: a row's token is the best of both ranks' best (value, then the lower id), drawn on
//! the host. With drafts, a stream's drafts are its MTP head's greedy chain (up to 3, cut where the product of their
//! confidences falls under 0.35, decode.py `draft`); the head absorbs the kept rows only when it drafts (or when it
//! owes 256), as drafter_choice.py's backlog does. Drafts never change a token, only how many a round lands.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const engine = @import("cuda_engine.zig");
const copy_drafts = @import("copy_drafts.zig");

const be = lanes.backend;
const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

/// Tokens a handle names (a drawn token waits on the host once its forward returns).
const ring = 1024;

/// The rows a verify window takes at most (the reference's widest decode graph).
pub const max_rows = engine.rows;

/// The streams a round runs at most.
pub const max_streams = 4;

/// The MTP chain's drafts at most (decode.py DepthPolicy `fixed`).
pub const depth = copy_drafts.depth_most;

/// Step opcodes; each record starts with one.
pub const Op = enum(u8) { open = 1, fill = 2, verify = 3, keep = 4, draft = 5, release = 6, stop = 7 };

/// A record's fixed header before a prompt's ids (u32 each).
const open_header = 7;

/// The bytes a record takes at most for sequences of `capacity` positions (an `open` of a whole prompt).
pub fn maxOpBytes(capacity: usize) usize {
    return @max(open_header + 4 * capacity, 2 + max_streams * (3 + 4 * max_rows));
}

const Slot = struct {
    seq: ?engine.Seq = null,
    prompt: ?[]u32 = null, // the prompt pass under way: its ids
    start: usize = 0, // the next chunk's first row
    first: u32 = 0, // the reply's first token, once the pass ended
    pending: ?usize = null, // the last window's rows, uncommitted
    drafts: bool = false, // the stream drafts: its head absorbs the kept rows
    sampled: [max_rows]u32 = undefined,
    held: [depth]u32 = undefined,
    probs: [depth]f64 = undefined,
    nheld: usize = 0,
};

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lay: config.Layout,
    r: *engine.Runner,
    comm: *const cuda.comm.Comm,
    capacity: usize,
    drafts: bool,
    slots: [max_streams]Slot = @splat(.{}),
    // rank 0: the stream each slot serves
    owners: [max_streams]?*const lanes.Stream = @splat(null),
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    op_buf: []u8,
    agree_send: Buf,
    agree_recv: Buf,
    stepped: u64 = 0,
    broken: bool = false,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, r: *engine.Runner, comm: *const cuda.comm.Comm, capacity: usize, drafts: bool) !*Lanes {
        const self = try gpa.create(Lanes);
        errdefer gpa.destroy(self);
        const op_buf = try gpa.alloc(u8, maxOpBytes(capacity));
        errdefer gpa.free(op_buf);
        var send = try Buf.alloc(d, 16);
        errdefer send.free();
        self.* = .{ .gpa = gpa, .d = d, .lay = lay, .r = r, .comm = comm, .capacity = capacity, .drafts = drafts, .op_buf = op_buf, .agree_send = send, .agree_recv = try Buf.alloc(d, 16 * lay.world) };
        return self;
    }

    pub fn deinit(self: *Lanes) void {
        for (&self.slots) |*s| self.clear(s);
        self.agree_send.free();
        self.agree_recv.free();
        self.gpa.free(self.op_buf);
        self.gpa.destroy(self);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .prefill_part = prefillPartFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .probabilities = probabilitiesFn,
            .tree = heldFn,
            .release = releaseFn,
        } };
    }

    /// The facts the round loop reads: up to `max_streams` streams a round, each window as wide as `max_rows` (every
    /// width keeps a one-token step's bits), greedy draws on the host; with drafts, the head's chain of up to
    /// `depth` with each draft's confidence (no cost table: the round loop keeps every draft it holds).
    pub fn facts(self: *const Lanes) lanes.Model {
        const rows = max_streams * max_rows;
        if (!self.drafts) return .{ .exact_width = max_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = false, .max_streams = max_streams, .batch_rows = rows };
        return .{ .exact_width = max_rows, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = depth, .draft_probabilities = true, .draft_streams = true, .hidden_rows = false, .max_streams = max_streams, .batch_rows = rows };
    }

    pub fn windowRows(_: *const Lanes) u32 {
        return max_rows;
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn clear(self: *Lanes, s: *Slot) void {
        if (s.prompt) |p| self.gpa.free(p);
        if (s.seq) |*q| q.deinit();
        s.* = .{};
    }

    // -- the steps every rank runs ------------------------------------------------------------------------------

    /// Every rank's digest of `op` must match (four 16-bit pieces of its sha256 as f32, one all-gather). A rank that
    /// cannot run `op` offers a digest no rank's matches, so every rank stops at the agreement.
    fn agree(self: *Lanes, op: []const u8, can_run: bool) !void {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(op, &digest, .{});
        var mine: [4]f32 = undefined;
        for (&mine, 0..) |*m, i| m.* = @floatFromInt(std.mem.readInt(u16, digest[2 * i ..][0..2], .little));
        if (!can_run) mine[0] = -1;
        const stream = self.r.stream;
        try self.agree_send.uploadAsync(0, std.mem.sliceAsBytes(&mine), stream.handle);
        try self.comm.allGather(self.agree_send.ptr, self.agree_recv.ptr, 4, .f32, stream.handle);
        try stream.synchronize();
        var every: [4 * cuda.comm.Comm.max_world]f32 = undefined;
        const world = self.lay.world;
        try self.agree_recv.download(0, std.mem.sliceAsBytes(every[0 .. 4 * world]));
        for (1..world) |rk| for (0..4) |i| if (every[rk * 4 + i] != every[i]) {
            const code: []const u8 = if (op.len > 0) (if (std.enums.fromInt(Op, op[0])) |c| @tagName(c) else "?") else "empty";
            std.log.err("glm53 rank {d}: step {d} ({s}, {d} bytes, runnable here {}) differs between the ranks: {any}", .{ self.lay.rank, self.stepped, code, op.len, can_run, every[0 .. 4 * world] });
            return error.OutOfStep;
        };
        self.stepped += 1;
    }

    /// Rank 0 sends the step to rank 1, both agree on it, both run it. A failure breaks the lane (the follower
    /// would otherwise pair its next gather wrongly).
    fn both(self: *Lanes, op: []const u8) !void {
        if (self.broken) return error.OutOfStep;
        errdefer self.broken = true;
        for (1..self.lay.world) |peer| try self.comm.sendTo(@intCast(peer), op);
        try self.agree(op, true);
        self.run(op) catch |e| {
            std.log.err("glm53 rank {d}: step {d} failed ({s}); restart both ranks", .{ self.lay.rank, self.stepped, @errorName(e) });
            return e;
        };
    }

    /// Rank 1: run rank 0's steps until it says stop or the link closes.
    pub fn follow(self: *Lanes) !void {
        if (self.lay.rank == 0) return;
        std.log.info("glm53 rank {d}: following rank 0", .{self.lay.rank});
        while (true) {
            const op = self.comm.recvFromSpin(0, self.op_buf, 200_000) catch |err| switch (err) {
                error.LinkClosed => return,
                else => return err,
            };
            if (op.len == 0) return error.OutOfStep;
            const code = std.enums.fromInt(Op, op[0]) orelse return error.OutOfStep;
            if (code == .stop) return;
            const ok = self.runnable(op);
            try self.agree(op, ok);
            self.run(op) catch |e| {
                std.log.err("glm53 rank {d}: step {d} ({s}) failed ({s}); restart both ranks", .{ self.lay.rank, self.stepped, @tagName(code), @errorName(e) });
                return e;
            };
        }
    }

    /// Rank 0 tells the followers to stop (no agreement: it is tearing its link down).
    pub fn stop(self: *Lanes) void {
        if (self.lay.rank != 0) return;
        const op = [_]u8{@intFromEnum(Op.stop)};
        for (1..self.lay.world) |peer| self.comm.sendTo(@intCast(peer), &op) catch {};
    }

    /// Whether this rank can run the record (it parses, and names slots in the state it needs).
    fn runnable(self: *const Lanes, op: []const u8) bool {
        var rd: Reader = .{ .b = op[1..] };
        const code = std.enums.fromInt(Op, op[0]) orelse return false;
        switch (code) {
            .open => {
                const slot = rd.byte() orelse return false;
                const flags = rd.byte() orelse return false;
                const n = rd.int() orelse return false;
                return slot < max_streams and self.slots[slot].seq == null and flags <= 1 and n > 0 and rd.b.len == 4 * @as(usize, n);
            },
            .fill => {
                const slot = rd.byte() orelse return false;
                _ = rd.int() orelse return false;
                return slot < max_streams and self.slots[slot].prompt != null and rd.b.len == 0;
            },
            .release => {
                const slot = rd.byte() orelse return false;
                return slot < max_streams and rd.b.len == 0;
            },
            .verify, .keep, .draft => {
                const n = rd.byte() orelse return false;
                if (n > max_streams) return false;
                for (0..n) |_| {
                    const slot = rd.byte() orelse return false;
                    if (slot >= max_streams or self.slots[slot].seq == null or self.slots[slot].prompt != null) return false;
                    const rows = rd.byte() orelse return false;
                    if (rows > max_rows) return false;
                    switch (code) {
                        .verify => if (rows == 0 or rd.skip(4 * @as(usize, rows)) == null) return false,
                        .keep => {},
                        .draft => {
                            _ = rd.byte() orelse return false;
                            if (rd.skip(4 * @as(usize, rows)) == null) return false;
                        },
                        else => unreachable,
                    }
                }
                return rd.b.len == 0;
            },
            .stop => return true,
        }
    }

    /// The step a record names, on this rank.
    fn run(self: *Lanes, op: []const u8) !void {
        var rd: Reader = .{ .b = op[1..] };
        const code = std.enums.fromInt(Op, op[0]) orelse return error.BadRecord;
        switch (code) {
            .open => {
                const s = &self.slots[rd.byte().?];
                const flags = rd.byte().?;
                const n = rd.int().?;
                const ids = try self.gpa.alloc(u32, n);
                errdefer self.gpa.free(ids);
                for (ids) |*t| t.* = rd.int().?;
                s.seq = try engine.Seq.init(self.gpa, self.d, self.lay, self.r.model.layers.len, self.capacity);
                s.prompt = ids;
                s.start = 0;
                s.drafts = flags & 1 != 0;
            },
            .fill => {
                const s = &self.slots[rd.byte().?];
                const chunks = rd.int().?;
                const q = &s.seq.?;
                const ids = s.prompt.?;
                var n: u32 = 0;
                while (n < chunks and s.start < ids.len) : (n += 1) s.start = try self.r.fill(q, ids, s.start);
                if (s.start == ids.len) {
                    s.first = try self.r.first(q);
                    self.gpa.free(ids);
                    s.prompt = null;
                }
            },
            .verify => {
                const n = rd.byte().?;
                for (0..n) |_| {
                    const s = &self.slots[rd.byte().?];
                    const rows = rd.byte().?;
                    var ids: [max_rows]u32 = undefined;
                    for (ids[0..rows]) |*t| t.* = rd.int().?;
                    try self.bind(s, null);
                    try self.r.window(&s.seq.?, ids[0..rows], &s.sampled);
                    s.pending = rows;
                    s.nheld = 0;
                }
            },
            .keep => {
                const n = rd.byte().?;
                for (0..n) |_| {
                    const s = &self.slots[rd.byte().?];
                    try self.bind(s, rd.byte().?);
                }
            },
            .draft => {
                const n = rd.byte().?;
                for (0..n) |_| {
                    const s = &self.slots[rd.byte().?];
                    const kept = rd.byte().?;
                    const want = rd.byte().?;
                    var after: [max_rows]u32 = undefined;
                    for (after[0..kept]) |*t| t.* = rd.int().?;
                    try self.bind(s, if (kept > 0) kept else null);
                    const q = &s.seq.?;
                    // the kept rows' hidden rows (the last window's) and the token after each; after a prompt the
                    // head already owes its last row (`Runner.first`)
                    if (kept > 0 and s.drafts) {
                        if (engine.Runner.owesTooMany(q, kept)) try self.r.absorb(q);
                        try self.r.owe(q, q.last.ptr, after[0..kept]);
                    }
                    s.nheld = 0;
                    if (want > 0 and s.drafts) {
                        try self.r.absorb(q);
                        s.nheld = try self.r.chain(q, @min(want, depth), &s.held, &s.probs);
                    }
                }
            },
            .release => self.clear(&self.slots[rd.byte().?]),
            .stop => {},
        }
    }

    /// The slot's last window committed: its first `kept` rows, else all of them.
    fn bind(self: *Lanes, s: *Slot, kept: ?usize) !void {
        const rows = s.pending orelse return;
        const k = kept orelse rows;
        if (k == 0 or k > rows) return error.BadKeep;
        try self.r.commit(&s.seq.?, rows, k);
        s.pending = null;
    }

    // -- rank 0: the vtable ---------------------------------------------------------------------------------------

    fn slotOf(self: *const Lanes, s: *const lanes.Stream) ?u8 {
        for (self.owners, 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try prefillPartFn(ptr, s, std.math.maxInt(u32))) {}
    }

    /// The next `chunks` 2,048-row chunks of the stream's prompt pass, the first call opening a slot's sequence;
    /// true once the pass ended (its first token drawn). Each chunk runs whole inside the call.
    fn prefillPartFn(ptr: *anyopaque, s: *lanes.Stream, chunks: u32) anyerror!bool {
        const self = of(ptr);
        var slot = self.slotOf(s);
        if (slot) |i| if (self.slots[i].prompt == null) { // a stream prefilled again: a new sequence
            releaseFn(ptr, s);
            slot = null;
        };
        if (slot == null) slot = try self.open(s);
        if (s.isCancelled()) return error.Cancelled;
        var buf: [7]u8 = undefined;
        buf[0] = @intFromEnum(Op.fill);
        buf[1] = slot.?;
        std.mem.writeInt(u32, buf[2..6], chunks, .little);
        try self.both(buf[0..6]);
        return self.slots[slot.?].prompt == null;
    }

    fn open(self: *Lanes, s: *lanes.Stream) !u8 {
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotSupported;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + max_rows >= self.capacity) return error.PromptTooLong;
        const slot: u8 = for (self.owners, 0..) |o, i| {
            if (o == null) break @intCast(i);
        } else return error.NoRoom;
        const op = self.op_buf[0 .. open_header + 4 * ids.len];
        op[0] = @intFromEnum(Op.open);
        op[1] = slot;
        op[2] = @intFromBool(self.drafts);
        std.mem.writeInt(u32, op[3..7], @intCast(ids.len), .little);
        for (ids, 0..) |t, i| std.mem.writeInt(u32, op[open_header + 4 * i ..][0..4], t, .little);
        try self.both(op);
        self.owners[slot] = s;
        return slot;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const i = self.slotOf(s) orelse return error.UnknownStream;
        if (self.slots[i].prompt != null) return error.PromptNotDone;
        return self.take(self.slots[i].first);
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Every window (the pending row, then host drafts or the held drafts) at its stream's length, one stream after
    /// another; committed lazily (`keep`, else the next step that binds the slot).
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > max_streams) return error.TooManyStreams;
        var w: Writer = .{ .b = self.op_buf };
        w.byte(@intFromEnum(Op.verify));
        w.byte(@intCast(windows.len));
        var slots: [max_streams]u8 = undefined;
        var echo: [max_streams][depth]u32 = undefined; // the held drafts each window carries (`run` drops them)
        for (windows, 0..) |win, i| {
            if (win.parents != null) return error.TreesNotBuilt;
            const slot = self.slotOf(win.stream) orelse return error.UnknownStream;
            for (slots[0..i]) |o| if (o == slot) return error.StreamTwice;
            slots[i] = slot;
            const sl = &self.slots[slot];
            if (win.held > sl.nheld) return error.NoHeldDrafts;
            const rows = win.rows();
            if (rows > max_rows) return error.WindowTooWide;
            // a window left uncommitted keeps all its rows before this one
            const len = sl.seq.?.len() + (sl.pending orelse 0);
            for (win.positions, 0..) |p, r| if (p != len + 1 + r) return error.PositionMismatch;
            @memcpy(echo[i][0..win.held], sl.held[0..win.held]);
            w.byte(slot);
            w.byte(@intCast(rows));
            w.int(win.pending);
            for (sl.held[0..win.held]) |t| w.int(t);
            for (win.tokens) |t| w.int(t);
        }
        try self.both(w.done());
        for (windows, out, slots[0..windows.len], 0..) |win, o, slot, i| {
            @memcpy(o.sampled[0..win.rows()], self.slots[slot].sampled[0..win.rows()]);
            @memcpy(o.drafts[0..win.held], echo[i][0..win.held]);
            @memcpy(o.drafts[win.held..][0..win.tokens.len], win.tokens);
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        var w: Writer = .{ .b = self.op_buf };
        w.byte(@intFromEnum(Op.keep));
        w.byte(@intCast(windows.len));
        for (windows, paths) |win, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            w.byte(self.slotOf(win.stream) orelse return error.UnknownStream);
            w.byte(@intCast(path.len));
        }
        try self.both(w.done());
    }

    /// Every request's kept rows owed to its head (after a prompt the head already owes its last row), then, for a
    /// request that wants drafts, the owed rows absorbed and the chain drafted; the drafts held for its next window.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (requests.len > max_streams) return error.TooManyStreams;
        if (!self.drafts) {
            for (requests) |r| if (r.depth > 0) return error.NoDraftHead;
        }
        var w: Writer = .{ .b = self.op_buf };
        w.byte(@intFromEnum(Op.draft));
        w.byte(@intCast(requests.len));
        for (requests) |r| {
            if (r.lanes != null) return error.TreesNotBuilt;
            const slot = self.slotOf(r.stream) orelse return error.UnknownStream;
            const sl = &self.slots[slot];
            const kept: usize = if (r.rows) |rows| rows.len else 0;
            if (r.rows) |rows| for (rows, 0..) |row, k| if (row != k) return error.TreesNotBuilt;
            if (kept > max_rows or r.follow.len < kept) return error.BadRequest;
            const len = sl.seq.?.len() + if (sl.pending) |p| (if (kept > 0) kept else p) else 0;
            if (r.position != len + 1) return error.PositionMismatch;
            w.byte(slot);
            w.byte(@intCast(kept));
            w.byte(@intCast(@min(r.depth, depth)));
            for (r.follow[0..kept]) |t| w.int(t);
        }
        try self.both(w.done());
    }

    /// The held drafts as host tokens (a chain: no parents).
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const self = of(ptr);
        const i = self.slotOf(s) orelse return null;
        const sl = &self.slots[i];
        return .{ .count = @intCast(sl.nheld), .tokens = try gpa.dupe(u32, sl.held[0..sl.nheld]) };
    }

    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const i = self.slotOf(s) orelse return false;
        const sl = &self.slots[i];
        const n = @min(out.len, sl.nheld);
        @memcpy(out[0..n], sl.probs[0..n]);
        return true;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const i = self.slotOf(s) orelse return;
        self.owners[i] = null;
        const op = [_]u8{ @intFromEnum(Op.release), i };
        self.both(&op) catch |e| std.log.err("glm53: releasing slot {d} failed ({s})", .{ i, @errorName(e) });
    }
};

/// A record's little-endian fields.
const Writer = struct {
    b: []u8,
    n: usize = 0,

    fn byte(w: *Writer, x: u8) void {
        w.b[w.n] = x;
        w.n += 1;
    }
    fn int(w: *Writer, x: u32) void {
        std.mem.writeInt(u32, w.b[w.n..][0..4], x, .little);
        w.n += 4;
    }
    fn done(w: *const Writer) []const u8 {
        return w.b[0..w.n];
    }
};

const Reader = struct {
    b: []const u8,

    fn byte(r: *Reader) ?u8 {
        if (r.b.len < 1) return null;
        defer r.b = r.b[1..];
        return r.b[0];
    }
    fn int(r: *Reader) ?u32 {
        if (r.b.len < 4) return null;
        defer r.b = r.b[4..];
        return std.mem.readInt(u32, r.b[0..4], .little);
    }
    fn skip(r: *Reader, n: usize) ?void {
        if (r.b.len < n) return null;
        r.b = r.b[n..];
    }
};

test "records the follower parses: a verify of two windows, a short one refused" {
    var buf: [64]u8 = undefined;
    var w: Writer = .{ .b = &buf };
    w.byte(@intFromEnum(Op.verify));
    w.byte(2);
    for ([_]u8{ 0, 1 }) |slot| {
        w.byte(slot);
        w.byte(2);
        w.int(7);
        w.int(8);
    }
    var rd: Reader = .{ .b = w.done()[1..] };
    try std.testing.expectEqual(@as(?u8, 2), rd.byte());
    try std.testing.expectEqual(@as(?u8, 0), rd.byte());
    try std.testing.expectEqual(@as(?u8, 2), rd.byte());
    try std.testing.expectEqual(@as(?u32, 7), rd.int());
    try std.testing.expectEqual(@as(?u32, 8), rd.int());
    var short: Reader = .{ .b = buf[0..3] };
    try std.testing.expect(short.skip(4) == null);
}
