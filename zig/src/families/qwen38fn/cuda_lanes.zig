//! Qwen3.8-Flash-Next streams behind the lane core (core/lanes): each stream owns its sequence (caches, DeltaNet
//! states, n-gram tail), shares the model and the window buffers, and runs its windows one forward at a time.
//! Greedy decoding only for now: a row's token is the first index of its largest logit (torch.argmax), read on the
//! host. No draft head yet (rounds of one row a stream), so every token is the serial decoder's.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Config = @import("config.zig").Config;
const fwd = @import("cuda_forward.zig");
const prompt_mod = @import("cuda_prompt.zig");

const be = lanes.backend;

/// Tokens a handle names (a drawn token waits on the host once its forward returns).
const ring = 1024;

/// The rows a decode window takes at most (forward.py windows keep each row's bits up to this many).
pub const max_rows = 16;

const Lane = struct { st: *fwd.Stream, pending_rows: ?usize = null };

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
    costs: [1]lanes.config.Cost = undefined,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, set: *cuda.aot.Set, capacity: usize, sms: u32) !*Lanes {
        const self = try gpa.create(Lanes);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .d = d, .m = m, .set = set, .f = fwd.Forward.init(d, m, set), .pr = undefined, .b = undefined, .pb = undefined, .stream = undefined, .capacity = capacity, .host_logits = undefined };
        self.pr = try prompt_mod.Prompt.init(&self.f, sms);
        errdefer self.pr.deinit();
        self.b = try fwd.Buffers.init(gpa, d, m, max_rows, capacity);
        errdefer self.b.deinit();
        self.pb = try prompt_mod.Buffers.init(gpa, d, m, prompt_mod.chunk_rows, capacity);
        errdefer self.pb.deinit();
        self.stream = try cuda.Stream.init(d, true);
        errdefer self.stream.deinit();
        self.host_logits = try gpa.alloc(u16, max_rows * m.cfg.vocab);
        return self;
    }

    pub fn deinit(self: *Lanes) void {
        var it = self.lanes.valueIterator();
        while (it.next()) |l| {
            l.st.deinit();
            self.gpa.destroy(l.st);
        }
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
            .release = releaseFn,
        } };
    }

    /// The facts the round loop reads: one-row rounds, one stream a forward, greedy draws on the host.
    pub fn facts(self: *Lanes) lanes.Model {
        _ = self;
        return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = false, .max_streams = 1, .batch_rows = 1 };
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

    fn cancelled(s: *lanes.Stream) bool {
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
        if (gop.found_existing) {
            gop.value_ptr.st.deinit();
            self.gpa.destroy(gop.value_ptr.st);
        }
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
        const toks = try self.gpa.alloc(i32, ids.len);
        defer self.gpa.free(toks);
        for (ids, toks) |t, *x| x.* = @intCast(t);
        var start: usize = 0;
        while (start < ids.len) {
            if (cancelled(s)) return error.Cancelled;
            const end = @min(start + prompt_mod.chunk_rows, ids.len);
            try self.pr.stage(st, &self.pb, toks[start..end], self.stream);
            try self.pr.compute(st, &self.pb, end - start, end == ids.len, .{}, self.stream);
            try self.pr.commit(st, &self.pb, end - start, self.stream);
            start = end;
        }
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

    /// Each window on its own forward (pending row, then host drafts), drawn greedily; committed lazily.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        for (windows, out) |w, o| {
            if (w.parents != null) return error.TreesNotBuilt;
            if (w.held > 0) return error.NoDraftHead;
            const l = try self.bindLane(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            for (w.positions, 0..) |p, r| if (p != l.st.pos + 1 + r) return error.PositionMismatch;
            var ids: [max_rows]i32 = undefined;
            ids[0] = @intCast(w.pending);
            for (w.tokens, 1..) |t, i| ids[i] = @intCast(t);
            try self.f.stage(l.st, &self.b, ids[0..rows], self.stream);
            try self.f.compute(l.st, &self.b, rows, .{}, self.stream);
            try self.stream.synchronize();
            try self.argmax(self.b.logits, rows, o.sampled);
            @memcpy(o.drafts, w.tokens);
            l.pending_rows = rows;
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

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth > 0) return error.NoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.stream.synchronize() catch {};
        kv.value.st.deinit();
        self.gpa.destroy(kv.value.st);
    }
};
