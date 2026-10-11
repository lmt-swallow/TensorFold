//! `glm53-prompt-time <aot-dir> <model-dir> [pos ...]`: the prompt path's speed on one rank (rank 0 of two), without
//! the other rank or a fixture. The rank's whole model (45 layers, the MTP layer, the head), zero caches over
//! `capacity` tokens, a 2,048-row chunk of fixed token ids at each position given (default 0 and 30,720; each chunk
//! starts there, the caches before it zero) with its MTP absorb, and a gather that stands the rank's own partial in
//! for the other rank's. Each chunk runs twice (the first warms up); the second's segments are timed one by one
//! (the stream synchronized after each) and summed by kind. The bits are not checked. `aot-dir` holds Triton
//! variants of 2,048-row chunks at those positions (glm53/capture.sh prompt's covers 0 .. 34,816).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");

const Gpu = check.Gpu;
const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;
const P = glm.prompt;
const F = glm.forward;
const A = glm.dsa;
const M = glm.moe;
const T = glm.mtp;

const capacity = 40_960;

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

var bound: ?*const cuda.Driver = null;

/// The other rank's partial: this rank's own (the bits differ from a real pair; the shapes and launches do not).
fn selfGather(_: *anyopaque, x: P.Exchange, stream: cuda.Stream) anyerror!void {
    const dst: Buf = .{ .d = bound.?, .ptr = x.gath, .len = 2 * x.bytes };
    try dst.copyFrom(0, x.part, x.bytes, stream.handle);
    try dst.copyFrom(x.bytes, x.part, x.bytes, stream.handle);
}

pub fn run(gpu: Gpu, aot_dir: []const u8, model: []const u8, positions: []const usize) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    bound = d;
    var why: glm.config.Why = .{};
    const cfg = glm.config.read(gpa, gpu.io, model, &why) catch |e| {
        std.debug.print("FAIL glm53 config: {t}: {s}\n", .{ e, why.text() });
        return e;
    };
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var kk = try glm.kda.Kernels.load(d);
    defer kk.deinit();
    var mk = try M.Kernels.load(d);
    defer mk.deinit();
    var tk = try T.Kernels.load(d);
    defer tk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    const lay: glm.config.Layout = .{ .rank = 0, .world = 2, .cfg = &cfg };
    const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
    const t0 = nowNs();
    var whole = try glm.model.Model.load(l, cfg.layers);
    defer whole.deinit();
    var mtp = try T.Mtp.load(l);
    defer mtp.deinit();
    pack.dropCache();
    std.debug.print("glm53-prompt-time: rank 0's {d} layers, the MTP layer and the head loaded in {d:.1} s\n", .{ cfg.layers, @as(f64, @floatFromInt(nowNs() - t0)) / 1e9 });
    const players = try gpa.alloc(P.Layer, cfg.layers);
    defer gpa.free(players);
    const m = P.of(&whole, &cfg, &mtp, players);
    var nk: usize = 0;
    var nd: usize = 0;
    for (0..cfg.layers) |i| switch (cfg.kind(i)) {
        .kda => nk += 1,
        .dsa => nd += 1,
    };
    var po = try P.Owned.init(d, lay, P.chunk_rows, capacity);
    defer po.deinit();
    var st = try F.KdaState.init(d, lay, nk, 16);
    defer st.deinit();
    const caches = try gpa.alloc(A.Cache, nd);
    defer gpa.free(caches);
    for (caches) |*c| c.* = try A.Cache.init(d, &cfg, capacity, A.indexRing(capacity, P.chunk_rows));
    defer for (caches) |*c| c.deinit();
    var mcache = try A.Cache.init(d, &cfg, capacity, A.indexRing(capacity, P.chunk_rows));
    defer mcache.deinit();
    var pos_dev = try Buf.alloc(d, 16);
    defer pos_dev.free();
    var mpos_dev = try Buf.alloc(d, 16);
    defer mpos_dev.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams };
    const k: P.Kernels = .{ .g = g, .kda = &kk, .moe = &mk, .mtp = &tk };

    const ids = try gpa.alloc(u32, P.chunk_rows + 1);
    defer gpa.free(ids);
    var prng = std.Random.DefaultPrng.init(20261012);
    for (ids) |*x| x.* = prng.random().intRangeLessThan(u32, 1000, 150_000);
    var dummy: u8 = 0;
    const gather: P.Gather = .{ .ctx = &dummy, .run = selfGather };

    const kinds = [_][]const u8{ "kda mixer", "dsa mixer", "dense mlp", "moe", "tail + mtp attention", "mtp moe", "mtp head + commit" };
    for (positions) |pos| {
        var ms: [kinds.len]f64 = @splat(0);
        var total: f64 = 0;
        for (0..2) |rep| {
            var ps: P.Stream = .{ .kda = &st, .caches = caches, .mtp_cache = &mcache, .pos = pos, .mtp_len = pos, .capacity = capacity, .pos_dev = pos_dev.ptr, .mtp_pos_dev = mpos_dev.ptr };
            const c: P.Chunk = .{ .p = po.view(), .m = &m, .k = k, .lay = lay, .st = &ps, .ids = ids[0..P.chunk_rows], .next = ids[1 .. P.chunk_rows + 1] };
            try stream.synchronize();
            const a = nowNs();
            const n = c.sites();
            for (0..n + 1) |i| {
                const s0 = nowNs();
                try c.run(i, stream);
                if (i < n) try gather.run(gather.ctx, c.exchange(i), stream);
                try stream.synchronize();
                const dt = @as(f64, @floatFromInt(nowNs() - s0)) / 1e6;
                if (rep == 1) {
                    const slot: usize = switch (c.kind(i)) {
                        .mixer => |li| if (m.layers[li].mixer == .kda) 0 else 1,
                        .ffn => |li| if (m.layers[li].ffn == .mlp) 2 else 3,
                        .tail => 4,
                        .mtp_ffn => 5,
                        .mtp_end => 6,
                    };
                    ms[slot] += dt;
                }
            }
            if (rep == 1) total = @as(f64, @floatFromInt(nowNs() - a)) / 1e6;
        }
        std.debug.print("glm53-prompt-time: 2048 rows at {d}: {d:.1} ms ({d:.0} tok/s on one rank, its gathers local)\n", .{ pos, total, 2048.0 / (total / 1e3) });
        for (kinds, ms) |name, v| std.debug.print("  {s:<22} {d:>8.1} ms  {d:>5.1}%\n", .{ name, v, 100 * v / total });
    }
    check.pass("glm53-prompt-time: {d} positions timed", .{positions.len});
}
