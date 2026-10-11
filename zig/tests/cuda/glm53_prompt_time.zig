//! `glm53-prompt-time <aot-dir> <model-dir> [pos ...]`: the prompt path's speed on one rank (rank 0 of two), without
//! the other rank or a fixture. The rank's whole model (45 layers, the MTP layer, the head), zero caches over
//! `capacity` tokens, a 2,048-row chunk of fixed token ids at each position given (default 0 and 30,720; each chunk
//! starts there, the caches before it zero) with its MTP absorb, and gathers that stand the rank's own bytes in for
//! the other rank's. Each chunk runs twice (the first warms up); the second's ops are timed one by one (the stream
//! synchronized after each) and summed by kind. TF_GLM_HC_SPLIT=1 plans the chunk row-split. The bits are not checked. `aot-dir` holds Triton
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

pub fn run(gpu: Gpu, aot_dir: []const u8, model: []const u8, positions: []const usize) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
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
    // TF_GLM_HC_SPLIT=1: the chunk row-split as served (two pieces, the fronts during the swaps), its swaps local
    const split: ?P.Split = if (std.c.getenv("TF_GLM_HC_SPLIT")) |v| (if (std.mem.eql(u8, std.mem.span(v), "1")) P.Split{} else null) else null;
    const kinds = [_][]const u8{ "hc glue", "kda front", "kda chain", "kda o_proj", "dsa front", "dsa rows", "dsa o_proj", "mlp", "moe route + experts", "moe shared", "moe sum", "gathers (local)", "tail", "mtp" };
    for (positions) |pos| {
        var ms: [kinds.len]f64 = @splat(0);
        var total: f64 = 0;
        for (0..2) |rep| {
            var ps: P.Stream = .{ .kda = &st, .caches = caches, .mtp_cache = &mcache, .pos = pos, .mtp_len = pos, .capacity = capacity, .pos_dev = pos_dev.ptr, .mtp_pos_dev = mpos_dev.ptr };
            const c: P.Chunk = .{ .p = po.view(), .m = &m, .k = k, .lay = lay, .st = &ps, .ids = ids[0..P.chunk_rows], .next = ids[1 .. P.chunk_rows + 1], .split = split };
            const ops = try c.plan(gpa);
            defer gpa.free(ops);
            try stream.synchronize();
            const a = nowNs();
            for (ops) |op| {
                const s0 = nowNs();
                switch (op) {
                    .gather => |x| {
                        // the other rank's bytes: this rank's own (the bits differ from a pair; the launches do not)
                        const dst: Buf = .{ .d = d, .ptr = x.recv, .len = 2 * x.bytes };
                        try dst.copyFrom(0, x.send, x.bytes, stream.handle);
                        try dst.copyFrom(x.bytes, x.send, x.bytes, stream.handle);
                        if (x.copy) |cp| {
                            const to: Buf = .{ .d = d, .ptr = cp.dst, .len = cp.bytes };
                            try to.copyFrom(0, cp.src, cp.bytes, stream.handle);
                        }
                    },
                    .wait => {},
                    else => try c.exec(op, stream),
                }
                try stream.synchronize();
                const dt = @as(f64, @floatFromInt(nowNs() - s0)) / 1e6;
                if (rep == 0) continue;
                const slot: usize = switch (op) {
                    .embed, .zero_pad, .hc_pre, .hc_post, .post_pair, .mean => 0,
                    .front => |x| if (x.site == .mixer) (if (m.layers[x.l].mixer == .kda) @as(usize, 1) else 4) else (if (m.layers[x.l].ffn == .mlp) @as(usize, 7) else 9),
                    .rest => |x| if (x.site == .mixer) (if (m.layers[x.l].mixer == .kda) @as(usize, 2) else 5) else 8,
                    .fill => |x| if (x.site == .mixer) (if (m.layers[x.l].mixer == .kda) @as(usize, 3) else 6) else (if (m.layers[x.l].ffn == .mlp) @as(usize, 7) else 10),
                    .gather, .wait => 11,
                    .tail => 12,
                    .mtp_front, .mtp_mid, .mtp_end, .commit => 13,
                };
                ms[slot] += dt;
            }
            if (rep == 1) total = @as(f64, @floatFromInt(nowNs() - a)) / 1e6;
        }
        std.debug.print("glm53-prompt-time: 2048 rows at {d}{s}: {d:.1} ms ({d:.0} tok/s on one rank, its gathers local, synchronized after every op)\n", .{ pos, if (split != null) " (row-split)" else "", total, 2048.0 / (total / 1e3) });
        for (kinds, ms) |name, v| std.debug.print("  {s:<22} {d:>8.1} ms  {d:>5.1}%\n", .{ name, v, 100 * v / total });
    }
    check.pass("glm53-prompt-time: {d} positions timed", .{positions.len});
}
