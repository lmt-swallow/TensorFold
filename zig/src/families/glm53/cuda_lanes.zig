//! GLM-5.3-Flash streams behind the lane core (core/lanes) on two ranks: rank 0 runs the round loop and sends each
//! backend call to rank 1 as a step record before both run it (a dsv41-style follower), so the ranks make the same
//! calls in the same order and their all-gathers pair. Every rank agrees on each record first (its sha256 pieces
//! all-gathered), so ranks that would run different steps stop before their gathers cross.
//!
//! Each stream owns a sequence (cuda_engine.zig `Seq`: the backbone's states and caches, the MTP head's cache and
//! the rows it owes) in one of `max_streams` slots; the window and prompt buffers are shared. A round's windows run
//! as one forward of up to `engine.batch_rows` rows (cuda_model.zig `forwardSegs`): the row-wise blocks over every
//! row, each stream's KDA chains and DSA attention over its own rows, so a stream's tokens are its own alone. Greedy decoding only: a row's token is the best of both ranks' best (value, then the lower id), drawn on
//! the host. With drafts, a stream's drafts are its MTP head's greedy chain (up to 3, cut where the product of their
//! confidences falls under 0.35, decode.py `draft`); the head absorbs the kept rows only when it drafts (or when it
//! owes 256), as drafter_choice.py's backlog does. Drafts never change a token, only how many a round lands.
//!
//! The streams' DSA caches are extents of one pool (cuda_pool.zig); a finished prompt stays there as a kept prompt
//! (cuda_keep.zig), and a later prompt that starts the same continues it from their longest shared boundary. Rank 0
//! alone chooses (the book's match and placement, the kept prompts a finishing one covers) and sends its choice in
//! the records; every rank's book follows the same records, so the ranks keep the same book.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const engine = @import("cuda_engine.zig");
const P = @import("cuda_pool.zig");
const copy_drafts = @import("copy_drafts.zig");
const dsa = @import("cuda_dsa.zig");
const keep = @import("cuda_keep.zig");
const kb = keep.book;
const Tr = @import("cuda_tier.zig");

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

/// A record's fixed header before a prompt's ids (u32 each): the opcode, the slot, the flags (bit 0: drafts; bit 1:
/// continues a kept prompt), the extent's base and size (positions), the kept prompt and its boundary (book points),
/// the prompt's length (u32 each).
const open_header = 23;

/// The bytes a record takes at most for sequences of `capacity` positions (an `open` of a whole prompt).
pub fn maxOpBytes(capacity: usize) usize {
    return @max(open_header + 4 * capacity, @max(2 + max_streams * (3 + 4 * max_rows), 3 + 4 * kb.max_kept));
}

const Slot = struct {
    seq: ?engine.Seq = null, // made once, reused by one stream after another
    busy: bool = false,
    base: usize = 0, // the stream's extent of the pool
    size: usize = 0,
    prompt: ?[]u32 = null, // the stream's prompt (held until it leaves: a kept prompt keeps its ids)
    filling: bool = false, // the prompt pass is under way
    start: usize = 0, // the next chunk's first row
    marks: kb.Marks = .{}, // the snapshots at its boundaries (book points), inherited ones first
    points: kb.Points = .{}, // the boundaries its pass snapshots at
    first: u32 = 0, // the reply's first token, once the pass ended
    pending: ?usize = null, // the last window's rows, uncommitted
    drafts: bool = false, // the stream drafts: its head absorbs the kept rows
    sampled: [max_rows]u32 = undefined,
    held: [depth]u32 = undefined,
    probs: [depth]f64 = undefined,
    nheld: usize = 0,
};

const KeptTier = struct { eid: u32, h: usize, hm: usize };

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lay: config.Layout,
    r: *engine.Runner,
    comm: *const cuda.comm.Comm,
    capacity: usize, // a stream's positions at most (the context's)
    pool: *P.Pool,
    ring: usize, // each slot's index ring rows
    drafts: bool,
    book: kb.Book,
    released: kb.Released = .{},
    snaps: ?keep.Snaps = null,
    /// The DSA latent rows on a tier (TF_GLM_TIER): each stream's caches tier streams of their own, a kept prompt's
    /// kept with it (`kept_tier`) and read through by the prompts that continue it (cuda_tier `alias`).
    tier: ?*Tr.Tier = null,
    kept_tier: [kb.max_kept + 1]?KeptTier = @splat(null),
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

    /// Lanes over `pool`, each stream at most `capacity` positions; every slot's sequence made now (its KDA states,
    /// rings and head buffers), so a stream waits for none.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, r: *engine.Runner, comm: *const cuda.comm.Comm, pool: *P.Pool, capacity: usize, drafts: bool, tier: ?*Tr.Tier) !*Lanes {
        const self = try gpa.create(Lanes);
        errdefer gpa.destroy(self);
        const op_buf = try gpa.alloc(u8, maxOpBytes(capacity));
        errdefer gpa.free(op_buf);
        var send = try Buf.alloc(d, 16);
        errdefer send.free();
        self.* = .{ .gpa = gpa, .d = d, .lay = lay, .r = r, .comm = comm, .capacity = capacity, .pool = pool, .ring = dsa.indexRing(capacity, engine.prompt_rows), .drafts = drafts, .book = .{ .set = keep.settings(), .chunk = keep.chunk, .window = 0 }, .op_buf = op_buf, .agree_send = send, .agree_recv = try Buf.alloc(d, 16 * lay.world) };
        errdefer self.deinit();
        self.book.released = &self.released;
        self.tier = tier;
        for (&self.slots) |*sl| sl.seq = try engine.Seq.initSlot(gpa, d, lay, r.model.layers.len, self.ring, pool.kv);
        if (self.book.set.on) self.snaps = try keep.Snaps.init(gpa, d, self.book.set.snaps, keep.snapBytes(&self.slots[0].seq.?));
        return self;
    }

    pub fn deinit(self: *Lanes) void {
        for (&self.slots) |*s| {
            self.clear(s);
            if (s.seq) |*q| q.deinit();
        }
        for (self.book.slice()) |k| self.gpa.free(k.ids);
        if (self.snaps) |*x| x.deinit();
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

    /// The facts the round loop reads: up to `max_streams` streams a round in one forward of up to
    /// `engine.batch_rows` rows, each window as wide as `max_rows` (every width keeps a one-token step's bits),
    /// greedy draws on the host; with drafts, the head's chain of up to `depth` with each draft's confidence (no cost
    /// table: the round loop keeps every draft it holds).
    pub fn facts(self: *const Lanes) lanes.Model {
        const rows = engine.batch_rows;
        if (!self.drafts) return .{ .exact_width = max_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = true, .max_streams = max_streams, .batch_rows = rows };
        return .{ .exact_width = max_rows, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = depth, .draft_probabilities = true, .draft_streams = true, .hidden_rows = true, .max_streams = max_streams, .batch_rows = rows };
    }

    pub fn windowRows(_: *const Lanes) u32 {
        return max_rows;
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    /// The slot free again: its extent back to the pool, its snapshots let go, its prompt freed; its sequence kept
    /// for the next stream.
    fn clear(self: *Lanes, s: *Slot) void {
        if (s.prompt) |p| self.gpa.free(p);
        if (s.seq) |*q| if (q.tier) |x| {
            x.t.close(x.h);
            x.t.close(x.hm);
            q.tier = null;
        };
        if (s.busy) {
            self.pool.extents.give(s.base, s.size);
            if (self.snaps) |*x| for (s.marks.slice()) |m| {
                x.refs[m.snap] -|= 1;
            };
        }
        const seq = s.seq;
        s.* = .{ .seq = seq };
    }

    /// What the book let go, freed: its snapshots' holds dropped, its kept ids' storage.
    fn settle(self: *Lanes) void {
        const r = &self.released;
        if (self.snaps) |*x| for (r.snaps[0..r.n_snaps]) |id| {
            x.refs[id] -|= 1;
        };
        for (r.ids[0..r.n_ids]) |ids| self.gpa.free(ids);
        r.* = .{};
    }

    /// A finishing stream's prompt kept (`Book.keep`) when its pass ended with a boundary snapshotted, else its
    /// extent freed; then the kept prompts `drops` (rank 0's `Book.covered`) go, and the oldest past the entries.
    fn finish(self: *Lanes, s: *Slot, drops: []const u32) void {
        if (!s.busy) return;
        const ids = s.prompt.?;
        const keepable = self.book.set.on and !s.filling and s.marks.n > 0;
        const owned: ?[]i32 = if (keepable) self.gpa.alloc(i32, s.marks.last().?) catch null else null;
        if (owned) |o| {
            if (s.seq.?.tier) |x| {
                // the kept prompt keeps the stream's tier streams (its entry's id is the book's next)
                for (&self.kept_tier) |*k| if (k.* == null) {
                    k.* = .{ .eid = self.book.next_eid, .h = x.h, .hm = x.hm };
                    break;
                };
                s.seq.?.tier = null;
            }
            self.book.keep(&self.pool.extents, s.base, s.size, asI32(ids), &s.marks, false, o);
            s.marks = .{};
            s.busy = false;
        }
        self.clear(s);
        self.book.finish(&self.pool.extents, drops);
        self.settle();
        self.sweepTier();
    }

    /// The tier streams of kept prompts the book no longer holds, closed (a stream that a continuing prompt reads
    /// through stays until that one closes).
    fn sweepTier(self: *Lanes) void {
        const t = self.tier orelse return;
        for (&self.kept_tier) |*k| if (k.*) |x| if (self.book.find(x.eid) == null) {
            t.close(x.h);
            t.close(x.hm);
            k.* = null;
        };
    }

    fn keptTier(self: *const Lanes, eid: u32) ?KeptTier {
        for (self.kept_tier) |k| if (k) |x| if (x.eid == eid) return x;
        return null;
    }

    /// Positions a stream of `prompt` tokens and up to `max_new` more takes (its windows' rows included), aligned;
    /// null past a stream's capacity.
    fn extentFor(self: *const Lanes, prompt: usize, max_new: usize) ?usize {
        const need = prompt + max_new + max_rows + 1;
        if (need > self.capacity) return null;
        return P.Extents.size(need);
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
                const base = rd.int() orelse return false;
                const size = rd.int() orelse return false;
                const eid = rd.int() orelse return false;
                const cut = rd.int() orelse return false;
                const n = rd.int() orelse return false;
                if (slot >= max_streams or self.slots[slot].busy or flags > 3 or n == 0 or rd.b.len != 4 * @as(usize, n)) return false;
                if (size > self.capacity or size % P.extent_align != 0 or n + max_rows >= size) return false;
                const reuse: ?kb.Reuse = if (flags & 2 != 0) .{ .eid = eid, .cut = cut } else null;
                if (reuse) |ru| {
                    if (!self.book.set.on or ru.cut >= n) return false;
                    const k = &self.book.kept[self.book.find(ru.eid) orelse return false];
                    if (k.marks.find(ru.cut) == null or ru.cut > k.top) return false;
                    for (k.keys()[0..ru.cut], 0..) |a, i| if (@as(u32, @bitCast(a)) != std.mem.readInt(u32, rd.b[4 * i ..][0..4], .little)) return false;
                }
                const placed = self.trialPlace(size, reuse) orelse return false;
                return placed.base == base;
            },
            .fill => {
                const slot = rd.byte() orelse return false;
                _ = rd.int() orelse return false;
                return slot < max_streams and self.slots[slot].busy and self.slots[slot].filling and rd.b.len == 0;
            },
            .release => {
                const slot = rd.byte() orelse return false;
                const n = rd.byte() orelse return false;
                return slot < max_streams and n <= kb.max_kept and rd.b.len == 4 * @as(usize, n);
            },
            .verify, .keep, .draft => {
                const n = rd.byte() orelse return false;
                if (n > max_streams) return false;
                for (0..n) |_| {
                    const slot = rd.byte() orelse return false;
                    if (slot >= max_streams or !self.slots[slot].busy or self.slots[slot].filling) return false;
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
                const base: usize = rd.int().?;
                const size: usize = rd.int().?;
                const eid = rd.int().?;
                const cut: usize = rd.int().?;
                const n = rd.int().?;
                const ids = try self.gpa.alloc(u32, n);
                errdefer self.gpa.free(ids);
                for (ids) |*t| t.* = rd.int().?;
                try self.admit(s, base, size, ids, if (flags & 2 != 0) .{ .eid = eid, .cut = cut } else null);
                s.drafts = flags & 1 != 0;
            },
            .fill => {
                const s = &self.slots[rd.byte().?];
                const chunks = rd.int().?;
                const q = &s.seq.?;
                const ids = s.prompt.?;
                var n: u32 = 0;
                while (n < chunks and s.start < ids.len) : (n += 1) {
                    s.start = try self.r.fill(q, ids, s.start);
                    if (s.points.has(s.start + 1)) if (self.snapshot(s)) |id| s.marks.add(.{ .at = s.start + 1, .snap = id });
                }
                if (s.start == ids.len) {
                    s.first = try self.r.first(q);
                    s.filling = false;
                    if (q.tier) |x| x.t.favor(x.h, false);
                }
            },
            .verify => {
                const n = rd.byte().?;
                var ids: [max_streams][max_rows]u32 = undefined;
                var wins: [max_streams]engine.Runner.Window = undefined;
                var which: [max_streams]*Slot = undefined;
                for (0..n) |i| {
                    const s = &self.slots[rd.byte().?];
                    const rows = rd.byte().?;
                    for (ids[i][0..rows]) |*t| t.* = rd.int().?;
                    try self.bind(s, null);
                    wins[i] = .{ .q = &s.seq.?, .ids = ids[i][0..rows], .out = &s.sampled };
                    which[i] = s;
                }
                try self.r.windows(wins[0..n]);
                for (wins[0..n], which[0..n]) |w, s| {
                    s.pending = w.ids.len;
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
            .release => {
                const s = &self.slots[rd.byte().?];
                var drops: [kb.max_kept]u32 = undefined;
                const n = rd.byte().?;
                for (drops[0..n]) |*x| x.* = rd.int().?;
                self.finish(s, drops[0..n]);
            },
            .stop => {},
        }
    }

    /// Where an admission of `size` positions continuing `reuse` would go (the book and extents untouched); null:
    /// no room even with every kept prompt gone.
    fn trialPlace(self: *const Lanes, size: usize, reuse: ?kb.Reuse) ?kb.Placed {
        var b = self.book.trial();
        var ex = self.pool.extents;
        return b.place(&ex, size, if (reuse) |r| r.eid else null);
    }

    /// The stream's extent placed by the book at `base` (continuing `reuse`: the kept prompt's rows before its
    /// boundary copied in unless the extent grew in place, and the slot's state restored from the snapshot there, so
    /// the prompt's chunks from the boundary are a fresh prefill's own).
    fn admit(self: *Lanes, s: *Slot, base: usize, size: usize, ids: []u32, reuse: ?kb.Reuse) !void {
        var inherited: kb.Marks = .{};
        var src_base: usize = 0;
        var cut_snap: u16 = 0;
        if (reuse) |r| {
            const k = &self.book.kept[self.book.find(r.eid).?];
            src_base = k.base;
            cut_snap = k.marks.find(r.cut).?;
            // the kept prompt's snapshots to the boundary become the stream's too (placing may forget it)
            for (k.marks.slice()) |m| if (m.at <= r.cut) {
                inherited.add(m);
                self.snaps.?.refs[m.snap] += 1;
            };
        }
        self.book.ticks += 1;
        const placed = self.book.place(&self.pool.extents, size, if (reuse) |r| r.eid else null) orelse return error.NoRoom;
        if (placed.base != base) return error.PositionMismatch;
        if (placed.how == .copy) {
            const k = &self.book.kept[self.book.find(reuse.?.eid).?];
            k.tick = self.book.ticks;
            k.hits += 1;
        }
        self.settle();
        s.* = .{ .seq = s.seq, .busy = true, .base = base, .size = size, .prompt = ids, .filling = true, .marks = inherited };
        const q = &s.seq.?;
        if (self.tier) |t| {
            const nd = q.st.caches.len;
            const h = try t.acquire(nd);
            errdefer t.close(h);
            q.tier = .{ .t = t, .h = h, .hm = try t.acquire(1) };
            t.favor(h, true);
        }
        try q.bind(self.pool, base, size);
        try q.reset(self.r.stream);
        if (reuse) |r| {
            const at = r.cut - 1; // the book's point is one past the boundary
            // the latent rows before the boundary: read through the kept prompt's tier streams, else copied
            if (q.tier) |x| {
                const src = self.keptTier(r.eid) orelse return error.NoKeptTier;
                try x.t.alias(src.h, x.h, at);
                try x.t.alias(src.hm, x.hm, at);
            }
            if (placed.how == .copy or placed.how == .move) try keep.copyRows(self.pool, src_base, base, at, self.r.stream);
            try self.snaps.?.restore(cut_snap, q, at, self.r.stream);
            s.start = at;
            if (self.lay.rank == 0) std.log.info("glm53: a prompt of {d} continues kept prompt {d} from {d} ({t})", .{ ids.len, r.eid, at, placed.how });
        }
        if (self.book.set.on) s.points = keep.pointsFor(&self.book, ids.len);
        self.sweepTier();
    }

    /// The slot's sequence into a free snapshot (the least recently used kept prompts give theirs back first); null
    /// when none frees (the boundary is not kept).
    fn snapshot(self: *Lanes, s: *Slot) ?u16 {
        const x = &(self.snaps orelse return null);
        var id = x.open();
        while (id == null) {
            if (!self.book.forgetOldest(&self.pool.extents)) return null;
            self.settle();
            id = x.open();
        }
        x.save(id.?, &s.seq.?, self.r.stream) catch {
            x.refs[id.?] = 0;
            return null;
        };
        return id;
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
        if (slot) |i| if (!self.slots[i].filling) { // a stream prefilled again: a new sequence
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
        return !self.slots[slot.?].filling;
    }

    fn open(self: *Lanes, s: *lanes.Stream) !u8 {
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotSupported;
        const ids = s.prompt();
        if (ids.len == 0) return error.PromptTooLong;
        const size = self.extentFor(ids.len, s.max_new) orelse return error.PromptTooLong;
        const slot: u8 = for (self.owners, 0..) |o, i| {
            if (o == null) break @intCast(i);
        } else return error.NoRoom;
        var reuse: ?kb.Reuse = if (self.book.set.on) self.book.match(ids, false) else null;
        // on a tier a kept prompt is continued by reading through its tier streams: one whose chain is already
        // max_chain deep (a continuation of continuations) is prefilled instead (the record carries the choice)
        if (reuse) |r| if (self.tier) |t| {
            const src = self.keptTier(r.eid);
            if (src == null or t.host.aliasDepth(src.?.h) > Tr.kv.max_chain or t.host.aliasDepth(src.?.hm) > Tr.kv.max_chain) reuse = null;
        };
        const base = (self.trialPlace(size, reuse) orelse return error.NoRoom).base;
        const op = self.op_buf[0 .. open_header + 4 * ids.len];
        op[0] = @intFromEnum(Op.open);
        op[1] = slot;
        op[2] = @as(u8, @intFromBool(self.drafts)) | @as(u8, @intFromBool(reuse != null)) << 1;
        std.mem.writeInt(u32, op[3..7], @intCast(base), .little);
        std.mem.writeInt(u32, op[7..11], @intCast(size), .little);
        std.mem.writeInt(u32, op[11..15], if (reuse) |r| r.eid else 0, .little);
        std.mem.writeInt(u32, op[15..19], if (reuse) |r| @intCast(r.cut) else 0, .little);
        std.mem.writeInt(u32, op[19..23], @intCast(ids.len), .little);
        for (ids, 0..) |t, i| std.mem.writeInt(u32, op[open_header + 4 * i ..][0..4], t, .little);
        try self.both(op);
        self.owners[slot] = s;
        return slot;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const i = self.slotOf(s) orelse return error.UnknownStream;
        if (self.slots[i].filling) return error.PromptNotDone;
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
        var total: usize = 0;
        for (windows, 0..) |win, i| {
            if (win.parents != null) return error.TreesNotBuilt;
            const slot = self.slotOf(win.stream) orelse return error.UnknownStream;
            for (slots[0..i]) |o| if (o == slot) return error.StreamTwice;
            slots[i] = slot;
            const sl = &self.slots[slot];
            if (win.held > sl.nheld) return error.NoHeldDrafts;
            const rows = win.rows();
            if (rows > max_rows) return error.WindowTooWide;
            total += rows;
            if (total > engine.batch_rows) return error.WindowTooWide;
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
        // the kept prompts this one covers when it is kept (the same ids to their top, their boundaries all kept)
        const sl = &self.slots[i];
        var drops: [kb.max_kept]u32 = undefined;
        var nd: usize = 0;
        if (self.book.set.on and !sl.filling and sl.marks.n > 0) {
            var have: [kb.max_marks]usize = undefined;
            for (sl.marks.slice(), have[0..sl.marks.n]) |m, *at| at.* = m.at;
            nd = self.book.covered(asI32(sl.prompt.?), have[0..sl.marks.n], false, &drops);
        }
        var w: Writer = .{ .b = self.op_buf };
        w.byte(@intFromEnum(Op.release));
        w.byte(i);
        w.byte(@intCast(nd));
        for (drops[0..nd]) |e| w.int(e);
        self.both(w.done()) catch |e| std.log.err("glm53: releasing slot {d} failed ({s})", .{ i, @errorName(e) });
    }
};

/// Token ids as the book holds them (ids are below 2^31).
fn asI32(ids: []const u32) []const i32 {
    return @ptrCast(ids);
}

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

test "a follower runs only records whose slots hold what the step needs" {
    var cfg: config.Config = undefined;
    var pool: P.Pool = .{ .gpa = std.testing.allocator, .latent = &.{}, .pk = &.{}, .rows = 2 * P.extent_align, .kv = .bf16, .extents = P.Extents.init(2 * P.extent_align) };
    var l: Lanes = .{ .gpa = std.testing.allocator, .d = undefined, .lay = .{ .rank = 1, .world = 2, .cfg = &cfg }, .r = undefined, .comm = undefined, .capacity = 2 * P.extent_align, .pool = &pool, .ring = 64, .drafts = true, .book = .{ .set = .{ .on = false }, .chunk = keep.chunk, .window = 0 }, .op_buf = &.{}, .agree_send = undefined, .agree_recv = undefined };
    var buf: [64]u8 = undefined;
    // open: slot 0, drafts, the extent [2048, 4096) (the book places the first free one: [0, 2048) is taken), two ids
    try std.testing.expect(pool.extents.takeAt(0, P.extent_align));
    var w: Writer = .{ .b = &buf };
    w.byte(@intFromEnum(Op.open));
    w.byte(0);
    w.byte(1);
    w.int(P.extent_align);
    w.int(P.extent_align);
    w.int(0);
    w.int(0);
    w.int(2);
    w.int(5);
    w.int(6);
    try std.testing.expect(l.runnable(w.done()));
    // one id short; a slot past the last; an extent that is taken
    try std.testing.expect(!l.runnable(w.done()[0 .. w.n - 1]));
    buf[1] = max_streams;
    try std.testing.expect(!l.runnable(w.done()));
    buf[1] = 0;
    try std.testing.expect(pool.extents.takeAt(P.extent_align, P.extent_align));
    try std.testing.expect(!l.runnable(w.done()));
    // continuing a kept prompt with kept prompts off
    pool.extents.give(P.extent_align, P.extent_align);
    buf[2] = 3;
    try std.testing.expect(!l.runnable(w.done()));
    // a verify names a slot no stream holds: refused
    var v: Writer = .{ .b = &buf };
    v.byte(@intFromEnum(Op.verify));
    v.byte(1);
    v.byte(0);
    v.byte(1);
    v.int(9);
    try std.testing.expect(!l.runnable(v.done()));
    try std.testing.expect(!l.runnable(&.{99}));
}
