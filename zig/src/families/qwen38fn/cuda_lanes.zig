//! Qwen3.8-Flash-Next streams behind the lane core (core/lanes): each stream owns its sequence (caches, DeltaNet
//! states, n-gram tail) and shares the model and the window buffers; a round's windows of up to `max_streams`
//! streams run as one forward (multi.py `round`), each row with the bits of its stream's own window.
//! Greedy decoding only for now: a row's token is the first index of its largest logit (torch.argmax), read on the
//! host. With the MTP head, a stream's drafts come from its greedy confidence chain (decode.py `draft`: up to 6, the
//! chain ending below 0.70) and are verified in its next window; the tokens kept are always the main model's.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Config = @import("config.zig").Config;
const fwd = @import("cuda_forward.zig");
const prompt_mod = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");

const be = lanes.backend;

/// Tokens a handle names (a drawn token waits on the host once its forward returns).
const ring = 1024;

/// The rows a decode window takes at most (forward.py windows keep each row's bits up to this many).
pub const max_rows = 16;

/// The streams a round runs at most (multi.py `MultiDecoder` slots): the shared buffers hold this many windows.
pub const max_streams = 4;

/// A shared round's rows at most.
const shared_rows = max_streams * max_rows;

const Lane = struct {
    st: *fwd.Stream,
    pending_rows: ?usize = null,
    a0: usize = 0, // the first row of its window in the last round's buffers
    round: u64 = 0, // that round (Lanes.rounds): its final streams stay in b.streams until the next one
    last: ?cuda.DeviceBuffer = null, // the prompt's last row's final streams (the head's first absorb)
    held: [mtp.depth]u32 = undefined, // the drafts the head holds for the next window
    probs: [mtp.depth]f64 = undefined,
    nheld: usize = 0,
};

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *fwd.Model,
    set: *cuda.aot.Set,
    f: fwd.Forward,
    pr: prompt_mod.Prompt,
    b: fwd.Buffers,
    pb: prompt_mod.Buffers,
    stream: cuda.Stream,
    capacity: usize,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    host_logits: []u16,
    rounds: u64 = 0,
    head: ?*mtp.Head = null,
    mb: ?fwd.Buffers = null, // the head's decode buffers
    sc: ?mtp.Scratch = null,
    scp: ?mtp.Scratch = null, // the head's prompt-chunk scratch

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, set: *cuda.aot.Set, capacity: usize, sms: u32) !*Lanes {
        const self = try gpa.create(Lanes);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .d = d, .m = m, .set = set, .f = fwd.Forward.init(d, m, set), .pr = undefined, .b = undefined, .pb = undefined, .stream = undefined, .capacity = capacity, .host_logits = undefined };
        self.pr = try prompt_mod.Prompt.init(&self.f, sms);
        errdefer self.pr.deinit();
        self.b = try fwd.Buffers.init(gpa, d, m, shared_rows, capacity);
        errdefer self.b.deinit();
        self.pb = try prompt_mod.Buffers.init(gpa, d, m, prompt_mod.chunk_rows, capacity);
        errdefer self.pb.deinit();
        self.stream = try cuda.Stream.init(d, true);
        errdefer self.stream.deinit();
        self.host_logits = try gpa.alloc(u16, shared_rows * m.cfg.vocab);
        return self;
    }

    /// Load the pack's MTP head and draft with it (before any stream exists: each stream then keeps a head cache).
    pub fn withHead(self: *Lanes) !void {
        self.head = try mtp.Head.load(self.gpa, self.d, self.m, max_streams, self.stream);
        self.mb = try fwd.Buffers.init(self.gpa, self.d, self.m, shared_rows, self.capacity);
        self.sc = try mtp.Scratch.init(self.d, &self.m.cfg, shared_rows);
        self.scp = try mtp.Scratch.init(self.d, &self.m.cfg, prompt_mod.chunk_rows);
    }

    pub fn deinit(self: *Lanes) void {
        var it = self.lanes.valueIterator();
        while (it.next()) |l| self.drop(l);
        if (self.mb) |*x| x.deinit();
        if (self.sc) |*x| x.deinit();
        if (self.scp) |*x| x.deinit();
        if (self.head) |h| h.deinit();
        self.lanes.deinit(self.gpa);
        self.gpa.free(self.host_logits);
        self.stream.deinit();
        self.pb.deinit();
        self.b.deinit();
        self.pr.deinit();
        self.gpa.destroy(self);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
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

    /// The facts the round loop reads: up to `max_streams` streams a round, greedy draws on the host. Without the
    /// head, one-row windows; with it, windows of the pending token and up to 6 drafts, the head's own chain deciding
    /// how many (its probabilities given, no cost table: the round loop keeps every draft it holds).
    pub fn facts(self: *Lanes) lanes.Model {
        const rows = max_streams * self.windowRows();
        if (self.head == null) return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = false, .max_streams = max_streams, .batch_rows = rows };
        return .{ .exact_width = self.windowRows(), .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = mtp.depth, .draft_probabilities = true, .draft_streams = true, .hidden_rows = false, .max_streams = max_streams, .batch_rows = rows };
    }

    /// A window's rows at most: the pending token and, with the head, its drafts.
    pub fn windowRows(self: *const Lanes) u32 {
        return if (self.head != null) 1 + mtp.depth else 1;
    }

    fn drop(self: *Lanes, l: *Lane) void {
        if (l.last) |*x| x.free();
        l.st.deinit();
        self.gpa.destroy(l.st);
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    /// The lane's stream, its pending window committed first (`kept` rows, else all of them).
    fn bindLane(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        if (l.pending_rows) |rows| try self.f.commit(l.st, rows, kept orelse rows, self.stream);
        l.pending_rows = null;
        return l;
    }

    /// Greedy draws of `rows` logit rows (bf16, row stride vocab): the first index of each row's largest value.
    fn argmax(self: *Lanes, logits: cuda.DeviceBuffer, rows: usize, out: []u32) !void {
        const v = self.m.cfg.vocab;
        const host = self.host_logits[0 .. rows * v];
        try logits.download(0, std.mem.sliceAsBytes(host));
        for (0..rows) |r| {
            const row = host[r * v ..][0..v];
            var best: usize = 0;
            var top = bf16(row[0]);
            for (row[1..], 1..) |x, i| {
                const f = bf16(x);
                if (f > top) {
                    top = f;
                    best = i;
                }
            }
            out[r] = @intCast(best);
        }
    }

    fn bf16(x: u16) f32 {
        return @bitCast(@as(u32, x) << 16);
    }

    fn cancelledCtx(ctx: *anyopaque) bool {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        return s.isCancelled();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// A new sequence for the stream, then its prompt in 2,048-row chunks; the first token drawn from the last row.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotPorted;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + max_rows > self.capacity) return error.PromptTooLong;
        const gop = try self.lanes.getOrPut(self.gpa, s);
        if (gop.found_existing) self.drop(gop.value_ptr);
        const st = self.gpa.create(fwd.Stream) catch |e| {
            self.lanes.removeByPtr(gop.key_ptr);
            return e;
        };
        st.* = fwd.Stream.init(self.gpa, self.d, self.m, self.capacity, max_rows) catch |e| {
            self.gpa.destroy(st);
            self.lanes.removeByPtr(gop.key_ptr);
            return e;
        };
        gop.value_ptr.* = .{ .st = st };
        const l = gop.value_ptr;
        l.last = try cuda.DeviceBuffer.alloc(self.d, self.m.cfg.wide() * 2);
        const toks = try self.gpa.alloc(i32, ids.len);
        defer self.gpa.free(toks);
        for (ids, toks) |t, *x| x.* = @intCast(t);
        try mtp.prefill(self.head, &self.pr, st, &self.pb, if (self.scp) |*x| x else null, toks, l.last.?, .{ .ctx = s, .check = cancelledCtx }, self.stream);
        try self.stream.synchronize();
        var first: [1]u32 = undefined;
        try self.argmax(self.pb.logits, 1, &first);
        _ = self.take(first[0]);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return self.next - 1;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// multi.py `round` without a prompt pass: every window (pending row, then host drafts) in one forward, each
    /// stream's rows after the last's, drawn greedily; committed lazily.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > max_streams) return error.TooManyStreams;
        var segs: [max_streams]fwd.Seg = undefined;
        var ids: [shared_rows]i32 = undefined;
        var R: usize = 0;
        for (windows, 0..) |w, i| {
            if (w.parents != null) return error.TreesNotBuilt;
            if (w.held > 0) return error.NoDraftHead;
            const l = try self.bindLane(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            for (w.positions, 0..) |p, r| if (p != l.st.pos + 1 + r) return error.PositionMismatch;
            for (windows[0..i]) |o| if (o.stream == w.stream) return error.StreamTwice;
            ids[R] = @intCast(w.pending);
            for (w.tokens, 1..) |t, k| ids[R + k] = @intCast(t);
            segs[i] = .{ .st = l.st, .a0 = R, .n = rows };
            R += rows;
        }
        if (R == 0) return;
        try self.f.stageSegs(segs[0..windows.len], &self.b, ids[0..R], self.stream);
        try self.f.computeSegs(segs[0..windows.len], &self.b, .{}, self.stream);
        try self.stream.synchronize();
        var drawn: [shared_rows]u32 = undefined;
        try self.argmax(self.b.logits, R, drawn[0..R]);
        self.rounds += 1;
        for (windows, out, segs[0..windows.len]) |w, o, sg| {
            @memcpy(o.sampled[0..sg.n], drawn[sg.a0..][0..sg.n]);
            @memcpy(o.drafts, w.tokens);
            const l = self.lanes.getPtr(w.stream).?;
            l.pending_rows = sg.n;
            l.a0 = sg.a0;
            l.round = self.rounds;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            _ = try self.bindLane(w.stream, path.len);
        }
    }

    /// multi.py `_draft_all`: every request's kept rows (or, after the prompt, its last row) absorbed with the token
    /// after each in one head call, then the chains, one head call a depth; each stream's drafts held for its next
    /// window.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const h = self.head orelse {
            for (requests) |r| if (r.depth > 0) return error.NoDraftHead;
            return;
        };
        if (requests.len > max_streams) return error.TooManyStreams;
        const wide = self.m.cfg.wide();
        var next: [max_streams][max_rows]i32 = undefined;
        var reqs: [max_streams]mtp.Head.Request = undefined;
        var held: [max_streams]*Lane = undefined;
        for (requests, 0..) |r, i| {
            const l = try self.bindLane(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != l.st.pos + 1) return error.PositionMismatch;
            var n: usize = 0;
            var streams: cuda.abi.DevicePtr = undefined;
            if (r.rows) |rows| {
                if (l.round != self.rounds) return error.StaleRows; // another round overwrote its final streams
                for (rows, 0..) |row, k| if (row != k) return error.TreesNotBuilt;
                for (r.follow[0..rows.len], 0..) |t, k| next[i][k] = @intCast(t);
                n = rows.len;
                streams = self.b.streams.ptr + l.a0 * wide * 2;
            } else {
                next[i][0] = @intCast(if (r.first) |first| switch (first) {
                    .handle => |hd| self.drawn[hd % ring],
                    .value => |v| v,
                } else if (r.follow.len == 1) r.follow[0] else return error.NoFirstToken);
                n = 1;
                streams = l.last.?.ptr;
            }
            l.nheld = 0;
            held[i] = l;
            reqs[i] = .{ .st = l.st, .streams = streams, .next = next[i][0..n], .count = @min(r.depth, mtp.depth), .out = &l.held, .probs = &l.probs };
        }
        try h.draftAll(&self.f, reqs[0..requests.len], &self.mb.?, &self.sc.?, self.stream);
        for (reqs[0..requests.len], held[0..requests.len]) |q, l| l.nheld = q.n;
    }

    /// The held drafts as host tokens (a chain: no parents), read where the round loop plans the next window.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const self = of(ptr);
        const l = self.lanes.getPtr(s) orelse return null;
        return .{ .count = @intCast(l.nheld), .tokens = try gpa.dupe(u32, l.held[0..l.nheld]) };
    }

    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const l = self.lanes.getPtr(s) orelse return false;
        const n = @min(out.len, l.nheld);
        @memcpy(out[0..n], l.probs[0..n]);
        return true;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        var kv = self.lanes.fetchRemove(s) orelse return;
        self.stream.synchronize() catch {};
        self.drop(&kv.value);
    }
};
