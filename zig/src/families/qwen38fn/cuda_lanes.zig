//! Qwen3.8-Flash-Next streams behind the lane core (core/lanes): each stream owns its sequence (caches, DeltaNet
//! states, n-gram tail) and shares the model and the window buffers; a round's windows of up to `max_streams`
//! streams run as one forward (multi.py `round`), each row with the bits of its stream's own window.
//! Greedy decoding only for now: a row's token is the first index of its largest logit (torch.argmax, a NaN
//! first), drawn on the device and read back as one int32 a row. With the MTP head, a stream's drafts come from its greedy confidence chain (decode.py `draft`: up to 6, the
//! chain ending below 0.70) and are verified in its next window; the tokens kept are always the main model's.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Config = @import("config.zig").Config;
const fwd = @import("cuda_forward.zig");
const prompt_mod = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const round_stats = @import("round_stats.zig");

const be = lanes.backend;

/// Tokens a handle names (a drawn token waits on the host once its forward returns).
const ring = 1024;

/// The rows a decode window takes at most (forward.py windows keep each row's bits up to this many).
pub const max_rows = 16;

/// The streams a round runs at most (multi.py `MultiDecoder` slots): the shared buffers hold this many windows.
pub const max_streams = 4;

/// Captured lone-stream rounds kept at most (each key: a slot, a row count, a DeltaNet parity, a context bucket).
const max_graphs = 256;

/// A shared round's rows at most.
const shared_rows = max_streams * max_rows;

/// A sequence's buffers, made once and reused (state.py `reset`) by one stream after another.
const Slot = struct {
    st: fwd.Stream,
    last: cuda.DeviceBuffer, // the prompt's last row's final streams (the head's first absorb)
    busy: bool = false,
};

const Lane = struct {
    st: *fwd.Stream,
    slot: usize,
    last: cuda.DeviceBuffer,
    // the prompt pass under way (prefill_part): the prompt's ids and the next chunk's first row
    fill: ?struct { ids: []i32, start: usize } = null,
    first: u64 = 0, // the handle of the token drawn from the prompt's last row
    pending_rows: ?usize = null,
    a0: usize = 0, // the first row of its window in the last round's buffers
    round: u64 = 0, // that round (Lanes.rounds): its final streams stay in b.streams until the next one
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
    argmax_mod: cuda.Module,
    argmax_fn: cuda.Function, // tf_argmax_rows_i32_kernel (torch_ops/argmax.cu)
    drawn_dev: cuda.DeviceBuffer, // int32 [shared_rows]
    rounds: u64 = 0,
    stats: ?round_stats.Stats = null, // TF_QFN_ROUND_STATS=1
    // a lone stream's rounds from captured graphs (TF_QFN_GRAPHS=0: every round eager)
    graphs: ?fwd.Graphs = null,
    slots: [max_streams]?Slot = @splat(null),
    head: ?*mtp.Head = null,
    mb: ?fwd.Buffers = null, // the head's decode buffers
    sc: ?mtp.Scratch = null,
    scp: ?mtp.Scratch = null, // the head's prompt-chunk scratch

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, set: *cuda.aot.Set, capacity: usize, sms: u32) !*Lanes {
        const self = try gpa.create(Lanes);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .d = d, .m = m, .set = set, .f = fwd.Forward.init(d, m, set), .pr = undefined, .b = undefined, .pb = undefined, .stream = undefined, .capacity = capacity, .argmax_mod = undefined, .argmax_fn = undefined, .drawn_dev = undefined };
        self.pr = try prompt_mod.Prompt.init(&self.f, sms);
        errdefer self.pr.deinit();
        self.b = try fwd.Buffers.init(gpa, d, m, shared_rows, capacity);
        errdefer self.b.deinit();
        self.pb = try prompt_mod.Buffers.init(gpa, d, m, prompt_mod.chunk_rows, capacity);
        errdefer self.pb.deinit();
        self.stream = try cuda.Stream.init(d, true);
        errdefer self.stream.deinit();
        self.argmax_mod = try cuda.Module.load(d, cuda.kernels.torch_argmax);
        errdefer self.argmax_mod.unload();
        self.argmax_fn = try self.argmax_mod.function("tf_argmax_rows_i32_kernel");
        self.drawn_dev = try cuda.DeviceBuffer.alloc(d, shared_rows * 4);
        const off = if (std.c.getenv("TF_QFN_GRAPHS")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        if (!off) self.graphs = .{ .gpa = gpa, .most = max_graphs };
        self.stats = round_stats.Stats.fromEnv();
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
        for (&self.slots) |*x| if (x.*) |*sl| {
            sl.st.deinit();
            sl.last.free();
        };
        if (self.mb) |*x| x.deinit();
        if (self.sc) |*x| x.deinit();
        if (self.scp) |*x| x.deinit();
        if (self.head) |h| h.deinit();
        self.lanes.deinit(self.gpa);
        if (self.graphs) |*g| g.deinit();
        self.drawn_dev.free();
        self.argmax_mod.unload();
        self.stream.deinit();
        self.pb.deinit();
        self.b.deinit();
        self.pr.deinit();
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

    /// The facts the round loop reads: up to `max_streams` streams a round (hidden_rows: their windows in one
    /// forward), greedy draws on the device. Without the
    /// head, one-row windows; with it, windows of the pending token and up to 6 drafts, the head's own chain deciding
    /// how many (its probabilities given, no cost table: the round loop keeps every draft it holds); in a shared
    /// round too every stream may draft the full depth (multi.py `_draft_all`: its room, not its last grant).
    pub fn facts(self: *Lanes) lanes.Model {
        const rows = max_streams * self.windowRows();
        if (self.head == null) return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = false, .max_streams = max_streams, .batch_rows = rows };
        return .{ .exact_width = self.windowRows(), .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = mtp.depth, .draft_probabilities = true, .draft_streams = true, .draft_slack = mtp.depth, .hidden_rows = true, .max_streams = max_streams, .batch_rows = rows };
    }

    /// A window's rows at most: the pending token and, with the head, its drafts.
    pub fn windowRows(self: *const Lanes) u32 {
        return if (self.head != null) 1 + mtp.depth else 1;
    }

    /// The lane's slot free again (its buffers kept for the next stream), its prompt pass's ids freed.
    fn drop(self: *Lanes, l: *Lane) void {
        if (l.fill) |f| self.gpa.free(f.ids);
        l.fill = null;
        self.slots[l.slot].?.busy = false;
    }

    /// The first `n` slots' sequences made now (caches allocated and zeroed), so no request waits for them.
    pub fn makeSlots(self: *Lanes, n: u32) !void {
        for (self.slots[0..@min(n, max_streams)]) |*x| if (x.* == null) {
            var st = try fwd.Stream.init(self.gpa, self.d, self.m, self.capacity, max_rows);
            errdefer st.deinit();
            x.* = .{ .st = st, .last = try cuda.DeviceBuffer.alloc(self.d, self.m.cfg.wide() * 2) };
        };
        try self.stream.synchronize();
    }

    /// A free slot reset for a new sequence; its buffers are made on its first use.
    fn takeSlot(self: *Lanes) !usize {
        for (&self.slots, 0..) |*x, i| {
            if (x.*) |*sl| {
                if (sl.busy) continue;
                try sl.st.reset(self.m, self.stream);
                sl.busy = true;
                return i;
            }
        }
        for (&self.slots, 0..) |*x, i| if (x.* == null) {
            var st = try fwd.Stream.init(self.gpa, self.d, self.m, self.capacity, max_rows);
            errdefer st.deinit();
            x.* = .{ .st = st, .last = try cuda.DeviceBuffer.alloc(self.d, self.m.cfg.wide() * 2), .busy = true };
            return i;
        };
        return error.TooManyStreams;
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

    /// The n-gram table rows a window of `tokens` after the stream's committed history reads, asked of the page cache
    /// ahead of the round's `stage` (the mapped table is read from disk on a row's first use).
    fn prefetchRows(self: *Lanes, st: *const fwd.Stream, tokens: []const i64) void {
        const g = &(self.m.ng orelse return);
        const t = &(self.m.table orelse return);
        var ids: [max_rows * 64]i64 = undefined;
        const n = tokens.len * g.heads();
        if (n > ids.len) return;
        g.ids(st.history[0 .. g.n - 1], tokens, ids[0..n]) catch return;
        t.prefetch(ids[0..n]);
    }

    /// The n-gram table rows of the prompt chunk starting at row `start` (its history: the prompt's n - 1 tokens
    /// before it), asked of the page cache ahead of its stage. The first chunk's history is the initial one: not asked.
    fn prefetchPrompt(self: *Lanes, ids: []const i32, start: usize) void {
        const g = &(self.m.ng orelse return);
        const t = &(self.m.table orelse return);
        const ctx = g.n - 1;
        if (start < ctx or start >= ids.len) return;
        const end = @min(start + self.pb.rows, ids.len);
        const seq = self.gpa.alloc(i64, ctx + end - start) catch return;
        defer self.gpa.free(seq);
        for (ids[start - ctx .. end], seq) |x, *y| y.* = x;
        const rows = self.gpa.alloc(i64, (end - start) * g.heads()) catch return;
        defer self.gpa.free(rows);
        g.idsAny(self.gpa, seq[0..ctx], seq[ctx..], rows) catch return;
        t.prefetch(rows);
    }

    /// The lane's stream, its pending window committed first (`kept` rows, else all of them).
    fn bindLane(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        if (l.pending_rows) |rows| try self.f.commit(l.st, rows, kept orelse rows, self.stream);
        l.pending_rows = null;
        return l;
    }

    /// Greedy draws of `rows` logit rows (bf16, row stride vocab): torch.argmax of each row (the first index of its
    /// largest value, a NaN first) on the device, read back once the stream reaches it.
    fn argmax(self: *Lanes, logits: cuda.DeviceBuffer, rows: usize, out: []u32) !void {
        const v = self.m.cfg.vocab;
        if (rows > shared_rows) return error.WindowTooLarge;
        var a: cuda.Args = .{};
        a.add(logits.ptr);
        a.add(self.drawn_dev.ptr);
        for ([_]usize{ rows, v, v }) |x| a.add(@as(i64, @intCast(x)));
        a.add(@as(i32, 0)); // bf16
        try cuda.launch.launch(self.argmax_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, self.stream, &a);
        var ids: [shared_rows]i32 = undefined;
        try self.drawn_dev.downloadAsync(0, std.mem.sliceAsBytes(ids[0..rows]), self.stream.handle);
        try self.stream.synchronize();
        for (ids[0..rows], out[0..rows]) |x, *o| o.* = @intCast(x);
    }

    fn cancelledCtx(ctx: *anyopaque) bool {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        return s.isCancelled();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// A new sequence for the stream, then its whole prompt in 2,048-row chunks (prefillPartFn's pass in one call).
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try prefillPartFn(ptr, s, std.math.maxInt(u32))) {}
    }

    /// The next `chunks` 2,048-row chunks of the stream's prompt pass (multi_fill.py's passes between rounds), the
    /// first call taking a slot for its sequence; true once the pass ended, its first token drawn from the last row.
    /// Each chunk runs whole inside the call, so other streams' rounds may run between two calls.
    fn prefillPartFn(ptr: *anyopaque, s: *lanes.Stream, chunks: u32) anyerror!bool {
        const self = of(ptr);
        var l = self.lanes.getPtr(s) orelse try self.startFill(s);
        if (l.fill == null) { // a stream prefilled again: a new sequence
            self.drop(l);
            _ = self.lanes.remove(s);
            l = try self.startFill(s);
        }
        const f = &l.fill.?;
        const t0 = if (self.stats) |*st| st.begin() else 0;
        var n: u32 = 0;
        defer if (self.stats) |*st| st.prompt(t0, n);
        while (n < chunks and f.start < f.ids.len) : (n += 1) {
            if (s.isCancelled()) return error.Cancelled;
            f.start = try mtp.prefillChunk(self.head, &self.pr, l.st, &self.pb, if (self.scp) |*x| x else null, f.ids, f.start, l.last, self.stream);
            // the chunk computes on the device: meanwhile the next chunk's n-gram rows are asked of the page cache
            self.prefetchPrompt(f.ids, f.start);
        }
        if (f.start < f.ids.len) return false;
        var first: [1]u32 = undefined;
        try self.argmax(self.pb.logits, 1, &first);
        l.first = self.take(first[0]);
        self.gpa.free(f.ids);
        l.fill = null;
        return true;
    }

    /// The stream's lane on a reset slot, its prompt's ids kept for the pass.
    fn startFill(self: *Lanes, s: *lanes.Stream) !*Lane {
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotPorted;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + max_rows > self.capacity) return error.PromptTooLong;
        const toks = try self.gpa.alloc(i32, ids.len);
        errdefer self.gpa.free(toks);
        for (ids, toks) |t, *x| x.* = @intCast(t);
        const slot = try self.takeSlot();
        errdefer self.slots[slot].?.busy = false;
        const sl = &self.slots[slot].?;
        try self.lanes.put(self.gpa, s, .{ .st = &sl.st, .slot = slot, .last = sl.last, .fill = .{ .ids = toks, .start = 0 } });
        return self.lanes.getPtr(s).?;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        if (l.fill != null) return error.PromptNotDone;
        return l.first;
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
        const t0 = if (self.stats) |*st| st.begin() else 0;
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
        if (self.graphs != null and windows.len == 1) {
            try self.graphs.?.compute(&self.f, segs[0].st, &self.b, R, self.stream);
        } else try self.f.computeSegs(segs[0..windows.len], &self.b, .{}, self.stream);
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
        if (self.stats) |*st| st.verify(t0, windows.len, R);
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        const t0 = if (self.stats) |*st| st.begin() else 0;
        var kept: usize = 0;
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            _ = try self.bindLane(w.stream, path.len);
            kept += path.len;
        }
        if (self.stats) |*st| st.keep(t0, kept);
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
        const t0 = if (self.stats) |*st| st.begin() else 0;
        defer if (self.stats) |*st| st.draft(t0);
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
                streams = l.last.ptr;
            }
            l.nheld = 0;
            held[i] = l;
            reqs[i] = .{ .st = l.st, .streams = streams, .next = next[i][0..n], .count = @min(r.depth, mtp.depth), .out = &l.held, .probs = &l.probs };
        }
        try h.draftAll(&self.f, reqs[0..requests.len], &self.mb.?, &self.sc.?, self.stream);
        for (reqs[0..requests.len], held[0..requests.len]) |q, l| {
            l.nheld = q.n;
            // the next window: the pending token (the last kept row's follower) and the drafts
            var win: [max_rows]i64 = undefined;
            win[0] = q.next[q.next.len - 1];
            for (l.held[0..l.nheld], 1..) |t, k| win[k] = t;
            self.prefetchRows(l.st, win[0 .. 1 + l.nheld]);
        }
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
        if (self.stats) |*st| if (self.lanes.count() == 0) st.flush();
    }
};
