//! DeepSeek-V4.1 behind the lane core on two ranks, one stream at a time, greedy, no drafts: rank 0 serves and
//! sends every step (the prompt, each one-token round, the release) to rank 1 over the communicator's control link
//! before running it; rank 1 runs the same steps in `follow`, so both ranks issue the same all-gathers in the same
//! order (model.py's lockstep). A sha256 digest of each step is all-gathered first (multi.py `_agree`): ranks that
//! planned different steps stop instead of pairing collectives wrongly.
//!
//! Interim: the prompt runs through the decode layer loop in windows of up to 16 rows (`Forward.rows`), not the
//! fork's prompt kernels, so a prompt's cache bits may differ from the Python prompt path until cuda_prompt lands;
//! sampling above temperature 0 is refused (the keyed draw is not ported).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const exl3 = @import("cuda_exl3.zig");
const experts = @import("cuda_experts.zig");
const attention = @import("cuda_attention.zig");
const forward = @import("cuda_forward.zig");
const RankCache = @import("rank_cache.zig").RankCache;

const be = lanes.backend;
const Ptr = cuda.abi.DevicePtr;

pub const max_rows: usize = 16;
/// Positions past the context a sequence's RoPE tables and caches hold (engine.py: limit + max_rows + 8).
pub const extra: usize = max_rows + 8;
pub const rope_align: usize = 4096;

pub const Error = error{ WorldNotTwo, BadMaster, NoRankCache, NoRopeTables, RopeRowsShort, OutOfStep, LinkClosed, SamplingNotSupported, WindowTooWide, PositionMismatch, UnknownStream, StreamBusy, PromptTooLong, Cancelled, NoDraftHead, NotPipelined, TreesNotBuilt, EmptyPath };

pub const Op = enum(u8) { prefill = 1, round = 2, release = 3, stop = 4 };

/// A step's record on the wire (little-endian): the opcode, then its fields.
const max_op_bytes = 4 + 8 + 8 * 16384 + 16;

pub const Settings = struct {
    rank: u32,
    world: u32,
    master: [4]u8,
    port: u16,
    context: usize,
    bucket: usize,
    rope_dir: []const u8,
    cache_dir: []const u8,
    kernels_dir: []const u8,
};

/// RoPE rows the tables hold for `context` positions (ops.py RopeTables.rows of the engine's rope_cap).
pub fn ropeRows(context: usize) usize {
    const need = context + extra;
    return @max(rope_align, (need + rope_align - 1) / rope_align * rope_align);
}

pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    cfg: Config,
    settings: Settings,
    set: cuda.aot.Set,
    glue: attention.Glue,
    kernels: exl3.Kernels,
    ek: experts.Kernels,
    w: weights.Weights,
    rope_c: attention.Rope,
    rope_w: attention.Rope,
    fwd: forward.Forward,
    comm: cuda.comm.Comm,
    stream: cuda.Stream,
    agree_send: cuda.DeviceBuffer,
    agree_recv: cuda.DeviceBuffer,
    logits_h: []f32,
    ids_h: []i64,
    op_buf: []u8,
    pos: usize = 0,
    stepped: u64 = 0,
    /// A step failed on this rank past its agreement: the other rank may sit in a collective, so no further
    /// collective is issued (it would pair with the wrong one or hang); the ranks must be restarted.
    broken: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, s: Settings) !*Engine {
        if (s.world != 2) return error.WorldNotTwo;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.d = ctx.d;
        e.settings = s;
        e.pos = 0;
        e.stepped = 0;
        e.broken = false;
        e.cfg = try Config.read(gpa, io, dir);
        // the communicator first: rank 1 waits here for rank 0's store, as the Python ranks do
        e.comm = try cuda.comm.Comm.init(s.rank, s.world, s.master, s.port, 1_800_000);
        errdefer e.comm.deinit();
        e.set = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, s.kernels_dir);
        errdefer e.set.deinit();
        e.glue = try attention.Glue.load(ctx.d);
        errdefer e.glue.deinit();
        e.kernels = try exl3.Kernels.load(ctx.d);
        errdefer e.kernels.deinit();
        e.ek = try experts.Kernels.load(ctx.d);
        errdefer e.ek.deinit();
        const rc_path = (try RankCache.locate(gpa, io, s.cache_dir, s.rank, s.world)) orelse return error.NoRankCache;
        defer gpa.free(rc_path);
        var rc = try RankCache.open(gpa, rc_path);
        defer rc.close();
        std.log.info("deepseek_v41 rank {d} of {d}: weights from {s}", .{ s.rank, s.world, rc_path });
        e.w = try weights.load(gpa, ctx.d, &e.cfg, &rc, &e.kernels, s.rank, s.world, .mul1, null);
        errdefer e.w.deinit();
        e.stream = try cuda.Stream.init(ctx.d, true);
        errdefer e.stream.deinit();
        const rows = ropeRows(s.context);
        e.rope_c = try loadRope(gpa, io, ctx.d, s.rope_dir, "rope_c", rows);
        errdefer freeRope(&e.rope_c);
        e.rope_w = try loadRope(gpa, io, ctx.d, s.rope_dir, "rope_w", rows);
        errdefer freeRope(&e.rope_w);
        e.fwd = try forward.Forward.init(gpa, ctx.d, &e.cfg, &e.w, &e.set, &e.glue, &e.ek, s.world, max_rows, s.context + extra, s.bucket, e.rope_c, e.rope_w, e.stream);
        errdefer e.fwd.deinit();
        e.agree_send = try cuda.DeviceBuffer.alloc(ctx.d, 16);
        errdefer e.agree_send.free();
        e.agree_recv = try cuda.DeviceBuffer.alloc(ctx.d, 16 * s.world);
        errdefer e.agree_recv.free();
        e.logits_h = try gpa.alloc(f32, e.cfg.vocab);
        errdefer gpa.free(e.logits_h);
        e.ids_h = try gpa.alloc(i64, max_rows);
        errdefer gpa.free(e.ids_h);
        e.op_buf = try gpa.alloc(u8, max_op_bytes);
        errdefer gpa.free(e.op_buf);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.gpa.free(e.op_buf);
        e.gpa.free(e.ids_h);
        e.gpa.free(e.logits_h);
        e.agree_recv.free();
        e.agree_send.free();
        e.fwd.deinit();
        freeRope(&e.rope_w);
        freeRope(&e.rope_c);
        e.stream.deinit();
        e.w.deinit();
        e.ek.deinit();
        e.kernels.deinit();
        e.glue.deinit();
        e.set.deinit();
        e.comm.deinit();
        e.gpa.destroy(e);
    }

    fn gather(self: *Engine) forward.Gather {
        return .{ .ctx = self, .call = gatherFn };
    }

    fn gatherFn(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        try e.comm.allGather(partial, out, words, .f32, stream.handle);
    }

    /// Every rank's digest of `op` must match (four 16-bit pieces of its sha256 as f32, one all-gather).
    fn agree(e: *Engine, op: []const u8) !void {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(op, &digest, .{});
        var mine: [4]f32 = undefined;
        for (&mine, 0..) |*m, i| m.* = @floatFromInt(std.mem.readInt(u16, digest[2 * i ..][0..2], .little));
        try e.agree_send.upload(0, std.mem.sliceAsBytes(&mine));
        try e.comm.allGather(e.agree_send.ptr, e.agree_recv.ptr, 4, .f32, e.stream.handle);
        try e.stream.synchronize();
        var every: [4 * cuda.comm.Comm.max_world]f32 = undefined;
        try e.agree_recv.download(0, std.mem.sliceAsBytes(every[0 .. 4 * e.settings.world]));
        for (1..e.settings.world) |r| for (0..4) |i| if (every[r * 4 + i] != every[i]) {
            std.log.err("rank {d}: step {d} differs between the ranks (digest piece {d})", .{ e.settings.rank, e.stepped, i });
            return error.OutOfStep;
        };
        e.stepped += 1;
    }

    /// Rank 0 sends the step to every other rank, then every rank agrees on it. A failure breaks the lane: a
    /// follower that got the step but not the agreement would otherwise pair its next collective wrongly.
    fn announce(e: *Engine, op: []const u8) !void {
        if (e.broken) return error.OutOfStep;
        errdefer e.broken = true;
        if (e.settings.rank == 0) for (1..e.settings.world) |peer| try e.comm.sendTo(@intCast(peer), op);
        try e.agree(op);
    }

    /// Runs a step; a failure breaks the lane on this rank (the other rank fails the same step on its own).
    fn step(e: *Engine, comptime f: anytype, args: anytype) !void {
        if (e.broken) return error.OutOfStep;
        @call(.auto, f, args) catch |err| {
            e.broken = true;
            std.log.err("rank {d}: step {d} failed ({s}); the lane is broken, restart both ranks", .{ e.settings.rank, e.stepped, @errorName(err) });
            return err;
        };
    }

    // -- the steps (the same on every rank) -------------------------------------------------------------------

    /// The prompt's ids into the caches from position 0, in windows of `max_rows`; the last row's logits stay in
    /// `fwd.logits` (the first token is drawn from them on rank 0). Runs every window once started: the follower
    /// cannot see a cancel, so stopping early on one rank would leave the other in an unpaired all-gather.
    fn runPrefill(e: *Engine, ids: []const i64) !void {
        if (ids.len + 1 + extra > e.settings.context + extra) return error.PromptTooLong;
        e.pos = 0;
        var i: usize = 0;
        while (i < ids.len) {
            const n = @min(max_rows, ids.len - i);
            _ = try e.fwd.rows(e.gather(), ids[i .. i + n], i, n, false, e.stream);
            i += n;
        }
        e.pos = ids.len;
    }

    /// One token at the cache's end: its logits (the next token's) in `fwd.logits`.
    fn runRound(e: *Engine, token: i64) !void {
        if (e.pos + 1 + extra > e.settings.context + extra) return error.PromptTooLong;
        e.ids_h[0] = token;
        _ = try e.fwd.rows(e.gather(), e.ids_h[0..1], e.pos, 1, false, e.stream);
        e.pos += 1;
    }

    /// The last logits' argmax (the first index of the maximum, torch's rule), read on the host.
    fn argmax(e: *Engine) !u32 {
        try e.stream.synchronize();
        try e.fwd.logits.download(0, std.mem.sliceAsBytes(e.logits_h));
        var best: usize = 0;
        for (e.logits_h, 0..) |v, i| if (v > e.logits_h[best]) {
            best = i;
        };
        return @intCast(best);
    }

    fn encodePrefill(buf: []u8, ids: []const i64) []const u8 {
        buf[0] = @intFromEnum(Op.prefill);
        std.mem.writeInt(u32, buf[1..5], @intCast(ids.len), .little);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, buf[5 + 8 * i ..][0..8], id, .little);
        return buf[0 .. 5 + 8 * ids.len];
    }

    fn encodeRound(buf: []u8, token: i64, pos: usize) []const u8 {
        buf[0] = @intFromEnum(Op.round);
        std.mem.writeInt(i64, buf[1..9], token, .little);
        std.mem.writeInt(u64, buf[9..17], @intCast(pos), .little);
        return buf[0..17];
    }

    /// Rank 1 (and any other follower): run rank 0's steps until it says stop or the link closes.
    pub fn follow(e: *Engine) !void {
        if (e.settings.rank == 0) return;
        std.log.info("deepseek_v41 rank {d}: following rank 0", .{e.settings.rank});
        while (true) {
            const op = e.comm.recvFrom(0, e.op_buf) catch |err| switch (err) {
                error.LinkClosed => return,
                else => return err,
            };
            if (op.len == 0) return error.OutOfStep;
            const code = std.enums.fromInt(Op, op[0]) orelse return error.OutOfStep;
            // rank 0 sends stop without an agreement (it is tearing its communicator down)
            if (code == .stop) return;
            try e.agree(op);
            switch (code) {
                .prefill => {
                    if (op.len < 5) return error.OutOfStep;
                    const n = std.mem.readInt(u32, op[1..5], .little);
                    if (op.len != 5 + 8 * @as(usize, n)) return error.OutOfStep;
                    const ids = try e.gpa.alloc(i64, n);
                    defer e.gpa.free(ids);
                    for (ids, 0..) |*id, i| id.* = std.mem.readInt(i64, op[5 + 8 * i ..][0..8], .little);
                    try e.step(Engine.runPrefill, .{ e, ids });
                },
                .round => {
                    if (op.len != 17) return error.OutOfStep;
                    const token = std.mem.readInt(i64, op[1..9], .little);
                    const pos: usize = @intCast(std.mem.readInt(u64, op[9..17], .little));
                    if (pos != e.pos) return error.PositionMismatch;
                    try e.step(Engine.runRound, .{ e, token });
                },
                .release => e.pos = 0,
                .stop => unreachable,
            }
        }
    }

    pub fn stop(e: *Engine) void {
        if (e.settings.rank != 0) return;
        const op = [_]u8{@intFromEnum(Op.stop)};
        for (1..e.settings.world) |peer| e.comm.sendTo(@intCast(peer), &op) catch {};
    }
};

fn freeRope(r: *attention.Rope) void {
    r.cos.free();
    r.sin.free();
}

/// `<dir>/<name>_<rows>.bin`: cos then sin, f32 [rows, rope_dim / 2] each (tools/zig/dsv41/dump_rope.py).
fn loadRope(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, name: []const u8, rows: usize) !attention.Rope {
    const half = 32;
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}_{d}.bin", .{ dir, name, rows });
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31)) catch {
        std.log.err("no RoPE table at {s}: run tools/zig/dsv41/dump_rope.py --rows {d}", .{ path, rows });
        return error.NoRopeTables;
    };
    defer gpa.free(bytes);
    if (bytes.len != 2 * rows * half * 4) return error.RopeRowsShort;
    var cos = try cuda.DeviceBuffer.fromHost(d, bytes[0 .. rows * half * 4]);
    errdefer cos.free();
    const sin = try cuda.DeviceBuffer.fromHost(d, bytes[rows * half * 4 ..]);
    return .{ .cos = cos, .sin = sin, .rows = rows };
}

/// The lane backend on rank 0: one stream, plain one-token rounds.
pub const Lanes = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    active: ?*const lanes.Stream = null,
    drawn: [1024]u32 = undefined,
    next: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) Lanes {
        return .{ .gpa = gpa, .e = e };
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

    pub fn facts(_: *const Lanes) lanes.Model {
        return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .hidden_rows = false, .batch_rows = 1, .max_streams = 1 };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % self.drawn.len] = token;
        self.next += 1;
        return h;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        if (self.active != null and self.active != s) return error.StreamBusy;
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotSupported;
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + max_rows > e.settings.context) return error.PromptTooLong;
        if (prompt.len > 16384) return error.PromptTooLong; // the op record's bound for this interim prompt path
        // a cancel is honoured only before the step is announced (runPrefill)
        if (s.isCancelled()) return error.Cancelled;
        const ids = try self.gpa.alloc(i64, prompt.len);
        defer self.gpa.free(ids);
        for (ids, prompt) |*id, t| id.* = t;
        self.active = s;
        try e.announce(Engine.encodePrefill(e.op_buf, ids));
        try e.step(Engine.runPrefill, .{ e, ids });
        _ = self.take(try e.argmax());
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
        if (handle >= self.next or self.next - handle > self.drawn.len) return error.NoSuchToken;
        return self.drawn[handle % self.drawn.len];
    }

    /// One window of one row: the pending token at the cache's end, its draw greedy.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        if (windows.len != 1) return error.WindowTooWide;
        const w = windows[0];
        if (self.active != w.stream) return error.UnknownStream;
        if (w.parents != null) return error.TreesNotBuilt;
        if (w.rows() != 1) return error.WindowTooWide;
        if (w.positions[0] != e.pos + 1) return error.PositionMismatch;
        try e.announce(Engine.encodeRound(e.op_buf, w.pending, e.pos));
        try e.step(Engine.runRound, .{ e, @as(i64, w.pending) });
        out[0].sampled[0] = try e.argmax();
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |_, path| {
            if (path.len == 0) return error.EmptyPath;
            if (path.len != 1 or path[0] != 0) return error.TreesNotBuilt;
        }
    }

    fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
        return error.NoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        if (self.active != s) return;
        self.active = null;
        const e = self.e;
        const op = [_]u8{@intFromEnum(Op.release)};
        if (!e.broken) e.announce(&op) catch |err| std.log.err("release: {s}", .{@errorName(err)});
        e.pos = 0;
    }
};

test "rope rows align up to 4096 past the context and the window's extra rows" {
    try std.testing.expectEqual(@as(usize, 4096), ropeRows(1024));
    try std.testing.expectEqual(@as(usize, 20480), ropeRows(16384));
    try std.testing.expectEqual(@as(usize, 69632), ropeRows(65536));
}
