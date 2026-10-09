//! `dsv41-prompt-time`: the served prompt path's speed on one rank, without the other rank or a fixture. The rank's
//! first `layers` layers from its rank cache, the RoPE tables of the Triton set's `rope/` dir, Engram when
//! TF_DS_ENGRAM names the tables, a synthetic prompt of `tokens` ids (a fixed pseudo-random draw a run), and a
//! gather that stands the rank's own partial in for the other rank's. Every chunk is profiled (`Prompt.Profile`);
//! the output is timing only, its bits are not checked. TF_DS_REPLAY=1 runs CED bounded replay (the upper layers
//! over the prompt's last window rows); TF_DS_PROFILE=0 times the chunks without the per-chunk profile.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");

const Gpu = check.Gpu;
const Ptr = cuda.abi.DevicePtr;

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

/// The other rank's partial: this rank's own (the bits differ from a real pair; the shapes and launches do not).
fn selfGather(_: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
    const d = bound.?;
    const dst: cuda.DeviceBuffer = .{ .d = d, .ptr = out, .len = 2 * words * 4 };
    try dst.copyFrom(0, partial, words * 4, stream.handle);
    try dst.copyFrom(words * 4, partial, words * 4, stream.handle);
}

var bound: ?*const cuda.Driver = null;

fn loadRope(gpu: Gpu, dir: []const u8, name: []const u8, rows: usize) !dsv41.attention.Rope {
    const half = 32;
    const path = try std.fmt.allocPrint(gpu.gpa, "{s}/rope/{s}_{d}.bin", .{ dir, name, rows });
    defer gpu.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 31));
    defer gpu.gpa.free(bytes);
    if (bytes.len != 2 * rows * half * 4) return error.RopeRowsShort;
    var cos = try cuda.DeviceBuffer.fromHost(gpu.d, bytes[0 .. rows * half * 4]);
    errdefer cos.free();
    const sin = try cuda.DeviceBuffer.fromHost(gpu.d, bytes[rows * half * 4 ..]);
    return .{ .cos = cos, .sin = sin, .rows = rows };
}

pub fn run(gpu: Gpu, cache_dir: []const u8, model_dir: []const u8, aot_dir: []const u8, rank: usize, layers: usize, tokens: usize, runs: usize) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const L = dsv41.serve_lanes;
    bound = gpu.d;
    var set = try cuda.aot.Set.load(gpa, io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var glue = try dsv41.attention.Glue.load(gpu.d);
    defer glue.deinit();
    var kernels = try dsv41.exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var ek = try dsv41.experts.Kernels.load(gpu.d);
    defer ek.deinit();
    var cfg = try dsv41.config.Config.read(gpa, io, model_dir);
    const world: u32 = 2;
    const rc_path = (try dsv41.rank_cache.RankCache.locate(gpa, io, cache_dir, rank, world)) orelse return error.NoRankCache;
    defer gpa.free(rc_path);
    var rc = try dsv41.rank_cache.RankCache.open(gpa, rc_path);
    defer rc.close();
    var w = try dsv41.weights.load(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1, layers);
    defer w.deinit();
    std.debug.print("loaded {d} layers of rank {d}\n", .{ w.layers.len, rank });
    const cap = std.mem.alignForward(usize, tokens + L.extra + 64, 4096);
    // the 1M-context tables (any table at least `cap` rows serves; the set's dir holds that one)
    const rows: usize = L.ropeRows(1 << 20);
    var rope_c = try loadRope(gpu, aot_dir, "rope_c", rows);
    defer rope_c.cos.free();
    defer rope_c.sin.free();
    var rope_w = try loadRope(gpu, aot_dir, "rope_w", rows);
    defer rope_w.cos.free();
    defer rope_w.sin.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var fwd = try dsv41.forward.Forward.init(gpa, gpu.d, &cfg, &w, &set, &glue, &ek, world, L.max_rows, cap, cap, rope_c, rope_w, stream);
    defer fwd.deinit();
    var eng: ?*dsv41.engram_apply.Engram = null;
    if (getenv("TF_DS_ENGRAM")) |dir| {
        const tm = if (getenv("TF_DS_TOKEN_MAP")) |p| try gpa.dupe(u8, p) else try std.fs.path.join(gpa, &.{ aot_dir, "token_map.json" });
        defer gpa.free(tm);
        eng = try dsv41.engram_apply.Engram.init(gpa, io, gpu.d, &cfg, &w, &set, dir, tm, L.default_chunk, 32);
    }
    defer if (eng) |e| e.deinit();
    var seq = try L.Sequence.init(gpa, gpu.d, &fwd, &kernels, eng, L.default_chunk, stream, null);
    defer seq.deinit(gpa);
    if (!std.mem.eql(u8, getenv("TF_DS_PROFILE") orelse "1", "0")) try seq.pr.enableProfile();
    const replay = std.mem.eql(u8, getenv("TF_DS_REPLAY") orelse "0", "1");
    const gather: dsv41.forward.Gather = .{ .ctx = &seq, .call = selfGather };
    const ids = try gpa.alloc(i64, tokens);
    defer gpa.free(ids);
    for (0..runs) |r| {
        // a fresh draw a run: the Engram rows of one run are not in the page cache for the next
        var prng = std.Random.DefaultPrng.init(0x5eed + r);
        for (ids) |*id| id.* = prng.random().intRangeLessThan(i64, 3, @intCast(cfg.vocab - 1024));
        try stream.synchronize();
        const t0 = check.now(io);
        try seq.prefill(gather, ids, L.default_chunk, replay);
        try stream.synchronize();
        const ms = @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6;
        std.debug.print("RESULT dsv41-prompt-time run {d}: {d} tokens, {d} layers, Engram {s}, replay {s}: {d:.0} ms, {d:.0} tok/s\n", .{ r, tokens, w.layers.len, if (eng != null) "on" else "off", if (replay) "on" else "off", ms, @as(f64, @floatFromInt(tokens)) * 1000 / ms });
    }
}
