//! `dsv41-prompt-time`: the served prompt path's speed on one rank, without the other rank or a fixture. The rank's
//! first `layers` layers from its rank cache, the RoPE tables of the Triton set's `rope/` dir, Engram when
//! TF_DS_ENGRAM names the tables, a synthetic prompt of `tokens` ids (a fixed pseudo-random draw a run), and a
//! gather that stands the rank's own partial in for the other rank's. Every chunk is profiled (`Prompt.Profile`);
//! the output is timing only, its bits are not checked. TF_DS_REPLAY=1 runs CED bounded replay (the upper layers
//! over the prompt's last window rows); TF_DS_PROFILE=0 times the chunks without the per-chunk profile.
//! TF_DS_PREFILL_CHUNK sets the chunk rows (default 2048). TF_DS_COMPARE_CHUNK=<rows> then prefills run 0's ids
//! from cleared caches at both chunk sizes and checks that the last row's logits and every layer's ring and planes
//! are byte-equal between them (PASS/FAIL): the prompt path's rows do not depend on how the prompt is chunked.
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
        const most: usize = if (getenv("TF_DS_PREFILL_CHUNK")) |v| try std.fmt.parseInt(usize, v, 10) else L.default_chunk;
        const cmp: usize = if (getenv("TF_DS_COMPARE_CHUNK")) |v| try std.fmt.parseInt(usize, v, 10) else 0;
        eng = try dsv41.engram_apply.Engram.init(gpa, io, gpu.d, &cfg, &w, &set, dir, tm, @max(L.max_rows, most, cmp), 32);
    }
    defer if (eng) |e| e.deinit();
    const chunk: usize = if (getenv("TF_DS_PREFILL_CHUNK")) |v| try std.fmt.parseInt(usize, v, 10) else L.default_chunk;
    const other: ?usize = if (getenv("TF_DS_COMPARE_CHUNK")) |v| try std.fmt.parseInt(usize, v, 10) else null;
    var seq = try L.Sequence.init(gpa, gpu.d, &fwd, &kernels, eng, @max(chunk, other orelse 0), stream, null);
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
        try seq.prefill(gather, ids, chunk, replay);
        try stream.synchronize();
        const ms = @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6;
        std.debug.print("RESULT dsv41-prompt-time run {d}: {d} tokens, {d} layers, Engram {s}, replay {s}, chunk {d}: {d:.0} ms, {d:.0} tok/s\n", .{ r, tokens, w.layers.len, if (eng != null) "on" else "off", if (replay) "on" else "off", chunk, ms, @as(f64, @floatFromInt(tokens)) * 1000 / ms });
    }
    const c2 = other orelse return;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    for (ids) |*id| id.* = prng.random().intRangeLessThan(i64, 3, @intCast(cfg.vocab - 1024));
    if (seq.pr.profile) |*p| {
        p.deinit();
        seq.pr.profile = null;
    }
    var want = try digests(gpu, &fwd, &seq, gather, ids, chunk, replay, stream);
    defer want.deinit(gpa);
    var got = try digests(gpu, &fwd, &seq, gather, ids, c2, replay, stream);
    defer got.deinit(gpa);
    var bad: usize = 0;
    for (want.items, got.items, 0..) |a, b, i| if (a != b) {
        if (bad < 8) std.debug.print("  differs: {s}\n", .{digestName(&fwd, i)});
        bad += 1;
    };
    try check.expect(bad == 0, "dsv41-prompt-time chunk {d} vs {d}: {d} tokens, replay {s}: {d} of {d} buffers differ", .{ chunk, c2, tokens, if (replay) "on" else "off", bad, want.items.len });
    check.pass("dsv41-prompt-time chunk {d} vs {d}: {d} tokens, {d} layers, replay {s}: logits, rings and planes byte-equal ({d} buffers)", .{ chunk, c2, tokens, w.layers.len, if (replay) "on" else "off", want.items.len });
}

const plane_fields = .{ "comp_codes", "comp_scales", "idxk_codes", "idxk_scales", "raw_kv", "raw_score" };

/// Which buffer digest `i` of `digests` is.
fn digestName(fwd: *const dsv41.forward.Forward, i: usize) []const u8 {
    if (i == 0) return "logits";
    const k = i - 1;
    if (k < fwd.rings.len) return "a ring";
    return "a plane";
}

/// From cleared caches (as Forward.init leaves them): the prompt at `chunk` rows a chunk, then the digests of the
/// last row's logits, every layer's ring and every kv source's planes.
fn digests(gpu: Gpu, fwd: *dsv41.forward.Forward, seq: *dsv41.serve_lanes.Sequence, gather: dsv41.forward.Gather, ids: []const i64, chunk: usize, replay: bool, stream: cuda.Stream) !std.ArrayList(u64) {
    const gpa = gpu.gpa;
    for (fwd.rings) |r| try r.fill8(0, stream.handle);
    for (fwd.planes) |pl| if (pl) |x| {
        inline for (plane_fields) |f| try @field(x, f).fill8(if (comptime std.mem.eql(u8, f, "idxk_scales")) 127 else 0, stream.handle);
    };
    try seq.prefill(gather, ids, chunk, replay);
    try stream.synchronize();
    var out: std.ArrayList(u64) = .empty;
    errdefer out.deinit(gpa);
    const v = fwd.cfg.vocab;
    try out.append(gpa, try digest(gpu, .{ .d = gpu.d, .ptr = seq.logits, .len = v * 4 }));
    for (fwd.rings) |r| try out.append(gpa, try digest(gpu, r));
    for (fwd.planes) |pl| if (pl) |x| {
        inline for (plane_fields) |f| try out.append(gpa, try digest(gpu, @field(x, f)));
    };
    return out;
}

fn digest(gpu: Gpu, b: cuda.DeviceBuffer) !u64 {
    if (b.len == 0) return 0;
    const bytes = try check.download(gpu, b);
    defer gpu.gpa.free(bytes);
    return std.hash.Wyhash.hash(0, bytes);
}

/// `dsv41-gemm-rows`: the prompt path's fp32 cuBLAS GEMMs (the MoE gate's [m, 384] and the indexer weights' [m, 64]
/// over k = 5120) on m rows of a 4096-row draw: for each m, the rows equal to the same rows of the 4096-row product,
/// both with the heuristic's algorithm for m (`run`) and with the 2048-row one's (`runAs`). RESULT lines say which
/// row counts differ; PASS when every pinned run matches.
pub fn gemmRows(gpu: Gpu) !void {
    const gpa = gpu.gpa;
    const K: usize = 5120;
    const M: usize = 4096;
    var g = try dsv41.prompt.Gemm32.init(gpu.d);
    defer g.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(0x6e3);
    const xh = try gpa.alloc(f32, M * K);
    defer gpa.free(xh);
    // bf16-valued inputs, as the prompt path's x.float() is
    for (xh) |*v| v.* = @as(f32, @bitCast(@as(u32, @bitCast(prng.random().floatNorm(f32))) & 0xffff0000));
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    var all_ok = true;
    for ([_]usize{ 384, 64 }) |N| {
        const wh = try gpa.alloc(f32, N * K);
        defer gpa.free(wh);
        for (wh) |*v| v.* = prng.random().floatNorm(f32) * 0.02;
        var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wh));
        defer w.free();
        var full = try cuda.DeviceBuffer.alloc(gpu.d, M * N * 4);
        defer full.free();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, M * N * 4);
        defer part.free();
        try g.run(x.ptr, w.ptr, full.ptr, M, N, K, stream);
        try stream.synchronize();
        const want = try check.download(gpu, full);
        defer gpa.free(want);
        for ([_]usize{ 2048, 1024, 1280, 640, 3000, 904, 2952, 129, 64, 17 }) |m| {
            for ([_]bool{ false, true }) |pinned| {
                for ([_]usize{ 0, M - m }) |r0| {
                    const xp = x.ptr + r0 * K * 4;
                    if (pinned) try g.runAs(xp, w.ptr, part.ptr, m, 2048, N, K, stream) else try g.run(xp, w.ptr, part.ptr, m, N, K, stream);
                    try stream.synchronize();
                    const got = try check.download(gpu, part);
                    defer gpa.free(got);
                    const same = std.mem.eql(u8, got[0 .. m * N * 4], want[r0 * N * 4 .. (r0 + m) * N * 4]);
                    if (!same and pinned) all_ok = false;
                    std.debug.print("RESULT dsv41-gemm-rows N {d} m {d} rows {d}.. {s}: {s}\n", .{ N, m, r0, if (pinned) "algorithm of 2048 rows" else "algorithm of m rows", if (same) "equal to the 4096-row product" else "DIFFERS" });
                }
            }
        }
    }
    try check.expect(all_ok, "dsv41-gemm-rows: a pinned run differs from the 4096-row product", .{});
    check.pass("dsv41-gemm-rows: every pinned run equals the 4096-row product's rows", .{});
}
