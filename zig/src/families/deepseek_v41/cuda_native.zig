//! DeepSeek-V4.1-Flash (EXL3, two DGX Sparks over NCCL) behind the native family interface: rank 0 serves the lane
//! core, rank 1 follows its steps. The ranks' settings come from the environment (`--tp/--rank/--master/
//! --master-port` export them: TF_TP_WORLD, TF_TP_RANK, TF_TP_MASTER, TF_TP_PORT), the weights from the Python
//! family's rank cache (TF_DS_RANK_CACHE), the RoPE tables from TF_DS_ROPE_DIR (default `<kernels>/rope`).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const lanesmod = @import("cuda_lanes.zig");
const Engine = lanesmod.Engine;
const Lanes = lanesmod.Lanes;
const RankCache = @import("rank_cache.zig").RankCache;

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 16384;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = @intCast(lanesmod.max_rows);

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
};

const Owned = struct { gpa: std.mem.Allocator, e: *Engine, lanes: Lanes };

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

fn envInt(name: [:0]const u8, default: u32) !u32 {
    const v = getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch error.BadEnvironment;
}

/// The rank settings the environment names; `rank` and `world` default to one rank, which this family refuses.
pub fn settings(gpa: std.mem.Allocator, kernels_dir: []const u8, context: usize) !lanesmod.Settings {
    const world = try envInt("TF_TP_WORLD", 1);
    const rank = try envInt("TF_TP_RANK", 0);
    const port: u16 = @intCast(try envInt("TF_TP_PORT", 29551));
    var master: [4]u8 = .{ 127, 0, 0, 1 };
    if (getenv("TF_TP_MASTER")) |m| {
        var it = std.mem.splitScalar(u8, m, '.');
        for (&master) |*b| b.* = std.fmt.parseInt(u8, it.next() orelse return error.BadMaster, 10) catch return error.BadMaster;
    }
    const bucket_default: usize = @max(1024, std.math.ceilPowerOfTwo(usize, context) catch context);
    const bucket: usize = if (getenv("TF_DS_BUCKET")) |b| (std.fmt.parseInt(usize, b, 10) catch return error.BadEnvironment) else bucket_default;
    const rope_dir = if (getenv("TF_DS_ROPE_DIR")) |r| try gpa.dupe(u8, r) else try std.fs.path.join(gpa, &.{ kernels_dir, "rope" });
    const cache_dir = try gpa.dupe(u8, getenv("TF_DS_RANK_CACHE") orelse return error.NoRankCache);
    return .{ .rank = rank, .world = world, .master = master, .port = port, .context = context, .bucket = bucket, .rope_dir = rope_dir, .cache_dir = cache_dir, .kernels_dir = kernels_dir };
}

/// This rank's device bytes: the rank cache's backbone entries (the DSpark blocks are not loaded yet).
pub fn weightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    _ = dir;
    const cache_dir = getenv("TF_DS_RANK_CACHE") orelse return 0;
    const world = envInt("TF_TP_WORLD", 1) catch return 0;
    const rank = envInt("TF_TP_RANK", 0) catch return 0;
    const path = (RankCache.locate(a, io, cache_dir, rank, world) catch return 0) orelse return 0;
    var rc = RankCache.open(a, path) catch return 0;
    defer rc.close();
    var total: u64 = 0;
    for (rc.entries) |e| if (!std.mem.startsWith(u8, e.key, "mtp.")) {
        total += e.bytes;
    };
    return total;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const s = try settings(gpa, kernels, o.context);
    if (s.world != 2) {
        std.log.err("deepseek_v41 serves on two ranks: start rank 0 and rank 1 with --tp 2 (TF_TP_WORLD=2)", .{});
        return error.WorldNotTwo;
    }
    const e = try Engine.init(gpa, io, ctx, dir, s);
    errdefer e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .lanes = Lanes.init(gpa, e) };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = 1,
        .stream_bytes = (o.context + lanesmod.extra) * 1024,
        .ctx = own,
        .deinit = release,
        .lone = null,
    };
}

/// Rank 1's loop: runs rank 0's steps until rank 0 stops; returns when the link closes.
pub fn follow(p: *anyopaque) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    try own.e.follow();
}

/// Whether this process is a follower (rank > 0), from the same environment `open` reads.
pub fn isFollower() bool {
    return (envInt("TF_TP_RANK", 0) catch 0) != 0;
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.SamplingNotSupported => "this engine decodes greedily for now: send temperature 0",
        error.WindowTooWide, error.NoDraftHead => "this engine serves one plain stream for now: no drafts, no concurrent rounds",
        error.OutOfStep => "the two ranks planned different steps; restart both ranks",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.e.stop();
    own.e.deinit();
    own.gpa.destroy(own);
}

test "settings read the environment's rank and world" {
    _ = settings;
    try std.testing.expectEqual(@as(u32, 16), prompt_rows);
}
