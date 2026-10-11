//! GLM-5.3-Flash (the EXL3 pack, two DGX Sparks over NCCL) behind the native family interface: rank 0 serves the
//! lane core, rank 1 follows its steps (cuda_lanes.zig). The ranks' settings come from the environment (`--tp/--rank/
//! --master/--master-port` export them: TF_TP_WORLD, TF_TP_RANK, TF_TP_MASTER, TF_TP_PORT); the kernel set
//! (TENSORFOLD_CUDA_KERNELS) must hold the reference's captured Triton kernels (tools/zig/glm53/capture.sh). Up to 4
//! streams decode together, their DSA caches in one pool of TF_GLM_POOL positions (default: one stream's context;
//! at least that, a multiple of 2,048), each stream an extent of it; every slot's other state (KDA states, rings, the
//! head's rows) is made at load. Both ranks must run with the same --context, TF_GLM_POOL and drafts setting.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const plan = @import("plan.zig");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const weights = @import("cuda_weights.zig");
const moe = @import("cuda_moe.zig");
const dsa = @import("cuda_dsa.zig");
const mtp = @import("cuda_mtp.zig");
const M = @import("cuda_model.zig");
const engine = @import("cuda_engine.zig");
const lanesmod = @import("cuda_lanes.zig");
const P = @import("cuda_pool.zig");

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 16384;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
/// Rounds of the decoding streams after each 2,048-row chunk of another stream's prompt pass.
pub const prefill_interleave: u32 = 1;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    pool_positions: usize = 0,
    pool_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
};

/// Everything a rank's engine holds, at fixed addresses (the runner and the lanes point into it).
const Owned = struct {
    gpa: std.mem.Allocator,
    cfg: config.Config,
    set: cuda.aot.Set,
    hc: glue.Hc,
    kk: kda.Kernels,
    mk: moe.Kernels,
    tk: mtp.Kernels,
    model: M.Model,
    head: mtp.Mtp,
    comm: cuda.comm.Comm,
    stream: cuda.Stream,
    runner: engine.Runner,
    pool: P.Pool,
    lanes: *lanesmod.Lanes,
};

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

fn envInt(name: [:0]const u8, default: u32) !u32 {
    const v = getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch error.BadEnvironment;
}

/// Whether this process is a follower (rank > 0), from the same environment `open` reads.
pub fn isFollower() bool {
    return (envInt("TF_TP_RANK", 0) catch 0) != 0;
}

/// A stream's positions at most for a context of `context` tokens: the context and a verify window, a whole extent.
pub fn capacityFor(context: usize) usize {
    return P.Extents.size(context + lanesmod.max_rows + 1);
}

/// TF_GLM_POOL's positions for streams of `capacity` positions: unset, `capacity`; else a whole number at least it
/// and a multiple of the extent alignment (error.BadPool otherwise).
pub fn poolSetting(text: ?[]const u8, capacity: usize) error{BadPool}!usize {
    const v = text orelse return capacity;
    const pool = std.fmt.parseInt(usize, std.mem.trim(u8, v, " "), 10) catch return error.BadPool;
    if (pool < capacity or pool % P.extent_align != 0) return error.BadPool;
    return pool;
}


/// This rank's device bytes of weights (its share of the pack's text tensors and its head rows).
pub fn weightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    var why: config.Why = .{};
    const cfg = config.read(a, io, dir, &why) catch return 0;
    const rank = envInt("TF_TP_RANK", 0) catch return 0;
    var b: plan.Bytes = .{};
    const Ctx = struct { b: *plan.Bytes, lay: config.Layout };
    plan.visit(&cfg, Ctx{ .b = &b, .lay = .{ .rank = rank, .world = config.world, .cfg = &cfg } }, struct {
        fn f(x: Ctx, w: plan.Want) !void {
            try x.b.add(x.lay, w);
        }
    }.f) catch return 0;
    return b.total;
}

fn gatherFn(p: *anyopaque, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, floats: usize, stream: cuda.Stream) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(p));
    try own.comm.allGather(send, recv, floats, .f32, stream.handle);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const world = try envInt("TF_TP_WORLD", 1);
    const rank = try envInt("TF_TP_RANK", 0);
    if (world != config.world or rank >= world) {
        std.log.err("glm53 serves on two ranks: start rank 0 and rank 1 with --tp 2 (TF_TP_WORLD=2)", .{});
        return error.WorldNotTwo;
    }
    const port: u16 = @intCast(try envInt("TF_TP_PORT", 29551));
    var master: [4]u8 = .{ 127, 0, 0, 1 };
    if (getenv("TF_TP_MASTER")) |m| {
        var it = std.mem.splitScalar(u8, m, '.');
        for (&master) |*b| b.* = std.fmt.parseInt(u8, it.next() orelse return error.BadMaster, 10) catch return error.BadMaster;
    }
    const d = ctx.d;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    var why: config.Why = .{};
    own.cfg = config.read(gpa, io, dir, &why) catch |e| {
        std.log.err("glm53: {s}", .{why.text()});
        return e;
    };
    const lay: config.Layout = .{ .rank = rank, .world = world, .cfg = &own.cfg };
    own.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, kernels);
    errdefer own.set.deinit();
    own.hc = try glue.Hc.load(d);
    errdefer own.hc.deinit();
    own.kk = try kda.Kernels.load(d);
    errdefer own.kk.deinit();
    own.mk = try moe.Kernels.load(d);
    errdefer own.mk.deinit();
    own.tk = try mtp.Kernels.load(d);
    errdefer own.tk.deinit();
    {
        var pack = try weights.Pack.open(gpa, io, dir);
        defer pack.close();
        const l: weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
        own.model = try M.Model.load(l, own.cfg.layers);
        errdefer own.model.deinit();
        own.head = try mtp.Mtp.load(l);
        pack.dropCache();
    }
    errdefer own.model.deinit();
    errdefer own.head.deinit();
    // each rank loads on its own; the ranks meet here
    own.comm = try cuda.comm.Comm.init(rank, world, master, port, 1_800_000);
    errdefer own.comm.deinit();
    own.stream = try cuda.Stream.init(d, true);
    errdefer own.stream.deinit();
    const capacity = capacityFor(o.context);
    const k: M.Kernels = .{ .g = .{ .set = &own.set, .hc = &own.hc, .hidden = own.cfg.hidden, .streams = own.cfg.streams }, .kk = &own.kk, .mk = &own.mk };
    own.runner = try engine.Runner.init(gpa, d, lay, k, &own.tk, &own.model, &own.head, .{ .ctx = own, .gather = gatherFn }, own.stream, capacity);
    errdefer own.runner.deinit();
    const positions = poolSetting(getenv("TF_GLM_POOL"), capacity) catch {
        std.log.err("TF_GLM_POOL is {s}: it takes a number of positions, at least a stream's {d}, a multiple of {d}", .{ getenv("TF_GLM_POOL").?, capacity, P.extent_align });
        return error.BadEnvironment;
    };
    const planes = own.cfg.count(.dsa) + 1;
    const kv = glue.Kv.parse(getenv("TF_GLM_KV") orelse "") catch {
        std.log.err("TF_GLM_KV is {s}: bf16 or fp8", .{getenv("TF_GLM_KV").?});
        return error.BadEnvironment;
    };
    own.pool = try P.Pool.init(gpa, d, &own.cfg, planes, positions, kv);
    errdefer own.pool.deinit();
    own.lanes = try lanesmod.Lanes.init(gpa, d, lay, &own.runner, &own.comm, &own.pool, capacity, o.drafts);
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = own.lanes.windowRows(),
        // the pool and every slot are allocated by the load: a stream needs no more
        .stream_bytes = 0,
        .pool_positions = positions,
        .pool_bytes = P.Pool.bytes(&own.cfg, planes, positions, kv),
        .ctx = own,
        .deinit = release,
    };
}

/// Rank 1's loop: runs rank 0's steps until rank 0 stops; returns when the link closes.
pub fn follow(p: *anyopaque) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    try own.lanes.follow();
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.SamplingNotSupported => "this engine decodes greedily for now: send temperature 0",
        error.NoRoom => "every stream slot is busy, or the pool shared by the running requests has no room for this one: retry when one finishes, or lower max_tokens",
        error.OutOfStep => "the two ranks planned different steps (check both ranks' --context and --no-drafts); restart both ranks",
        error.UnportedPromptSplitSelection => "this prompt's length needs the prompt path's split block selection, which is not ported yet",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lanes.stop();
    own.lanes.deinit();
    own.pool.deinit();
    own.runner.deinit();
    own.stream.deinit();
    own.comm.deinit();
    own.head.deinit();
    own.model.deinit();
    own.tk.deinit();
    own.mk.deinit();
    own.kk.deinit();
    own.hc.deinit();
    own.set.deinit();
    own.gpa.destroy(own);
}

test "a stream's capacity holds the context and a window, a whole extent" {
    for ([_]usize{ 4096, 16384, 1 << 20, 100000 }) |c| {
        const cap = capacityFor(c);
        try std.testing.expect(cap > c + lanesmod.max_rows);
        try std.testing.expectEqual(@as(usize, 0), cap % P.extent_align);
    }
}

test "TF_GLM_POOL: a stream's capacity unless set; at least it, a multiple of the extent alignment" {
    const m: usize = 1 << 20;
    try std.testing.expectEqual(m, try poolSetting(null, m));
    try std.testing.expectEqual(4 * m, try poolSetting("4194304", m));
    try std.testing.expectError(error.BadPool, poolSetting("1046528", m));
    try std.testing.expectError(error.BadPool, poolSetting("2097153", m));
    try std.testing.expectError(error.BadPool, poolSetting("4M", m));
}
