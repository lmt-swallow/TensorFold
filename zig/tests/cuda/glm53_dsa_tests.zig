//! `glm53-dsa <work-dir> <model-dir> [isolated]`: GLM-5.3-Flash's first DSA layer (layer 3: the indexer, the pooled
//! keys, the top-512 pool selection, the latent attention dense and sparse, then its MoE) on both ranks against the
//! Mia fork's call-by-call fixture (tools/zig/glm53/dump_dsa.py, packed by capture.sh): run a (one stream from
//! position 0 across the dense limit, prompt chunks and decode windows) and run l (decode windows at 40,000 and
//! 600,000 positions over the fixture's drawn caches), every loaded tensor's sha256 against the reference loader's.
//! The MTP layer's run m is not checked here yet. Both ranks run on this one GPU; see glm53_harness.zig.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const hs = @import("glm53_harness.zig");
const Gpu = check.Gpu;
const Harness = hs.Harness;

const F = glm.forward;
const M = glm.moe;
const A = glm.dsa;
const Buf = cuda.DeviceBuffer;
const Ptr = cuda.abi.DevicePtr;

const max_rows = 16;
const prompt_rows = 2048;

const Rank = struct {
    frame: glm.weights.Frame,
    dsa: A.Dsa,
    moe: M.Moe,
    norm: Buf,
    embed: Buf,
    b: F.Buffers,
    pb: F.Buffers,
    ms: M.Scratch,
    pms: M.Scratch,
    fnormed: Buf,
    fxs: Buf,
    pos_dev: Buf,

    fn deinit(r: *Rank) void {
        r.frame.deinit();
        r.dsa.deinit();
        r.moe.deinit();
        r.norm.free();
        r.embed.free();
        r.b.deinit();
        r.pb.deinit();
        r.ms.deinit();
        r.pms.deinit();
        r.fnormed.free();
        r.fxs.free();
        r.pos_dev.free();
    }
};

/// One run's per-rank caches and DSA scratch over `capacity` tokens.
const Run = struct {
    cache: [2]A.Cache,
    ds: [2]A.Scratch,
    pds: [2]A.Scratch,

    fn init(d: *const cuda.Driver, cfg: *const glm.config.Config, capacity: usize) !Run {
        var r: Run = undefined;
        for (0..2) |i| {
            const lay: glm.config.Layout = .{ .rank = i, .world = 2, .cfg = cfg };
            r.cache[i] = try A.Cache.init(d, cfg, capacity, A.indexRing(capacity, @max(max_rows, prompt_rows)));
            r.ds[i] = try A.Scratch.init(d, lay, max_rows, capacity, false);
            r.pds[i] = try A.Scratch.init(d, lay, prompt_rows, capacity, true);
        }
        return r;
    }

    fn deinit(r: *Run) void {
        for (&r.cache) |*c| c.deinit();
        for (&r.ds) |*s| s.deinit();
        for (&r.pds) |*s| s.deinit();
    }
};

fn weightsOk(gpa: std.mem.Allocator, io: std.Io, work: []const u8, ranks: []Rank) !void {
    const path = try std.fs.path.join(gpa, &.{ work, "out", "weights.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var bad: usize = 0;
    var n: usize = 0;
    for (ranks, 0..) |*r, ri| {
        var kb: [16]u8 = undefined;
        const want = parsed.value.object.get(try std.fmt.bufPrint(&kb, "rank{d}", .{ri})).?.object;
        const a = &r.dsa;
        const pairs = [_]struct { []const u8, Buf }{
            .{ "layers.3.dsa.proj.weight", a.proj.buf }, .{ "layers.3.dsa.q_norm", a.q_norm },           .{ "layers.3.dsa.kv_norm", a.kv_norm },
            .{ "layers.3.dsa.q_b.weight", a.q_b.buf },   .{ "layers.3.dsa.absorb.wk", a.wk },            .{ "layers.3.dsa.absorb.wv", a.wv },
            .{ "layers.3.dsa.o.weight", a.o.buf },       .{ "layers.3.dsa.index.kw.weight", a.kw.buf },  .{ "layers.3.dsa.index.qb.weight", a.qb.buf },
            .{ "layers.3.dsa.index.ln_w", a.ln_w },      .{ "layers.3.dsa.index.ln_b", a.ln_b },         .{ "layers.3.dsa.index.gate", a.gate },
            .{ "layers.3.dsa.index.ape", a.ape },        .{ "layers.3.attn_hc.fn", r.frame.attn_hc.fn_ }, .{ "layers.3.in_norm", r.frame.in_norm },
            .{ "norm", r.norm },
        };
        for (pairs) |p| {
            const got = try hs.sha(gpa, p[1]);
            n += 1;
            if (!std.mem.eql(u8, &got, want.get(p[0]).?.string)) {
                std.debug.print("FAIL glm53 rank {d} {s}: sha256 {s}, the reference's {s}\n", .{ ri, p[0], got[0..16], want.get(p[0]).?.string[0..16] });
                bad += 1;
            }
        }
    }
    try check.expect(bad == 0, "glm53-dsa: {d} of {d} loaded tensors differ from the reference loader's", .{ bad, n });
    check.pass("glm53-dsa: every rank's layer-3 DSA tensors equal the reference loader's ({d} sha256)", .{n});
}

pub fn run(gpu: Gpu, work: []const u8, model: []const u8, isolated: bool) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: glm.config.Why = .{};
    const cfg = glm.config.read(gpa, gpu.io, model, &why) catch |e| {
        std.debug.print("FAIL glm53 config: {t}: {s}\n", .{ e, why.text() });
        return e;
    };
    const aot_dir = try std.fs.path.join(gpa, &.{ work, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_dir = try std.fs.path.join(gpa, &.{ work, "out", "fixture" });
    defer gpa.free(fx_dir);
    var h: Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir), .stream = try cuda.Stream.init(d, true), .isolated = isolated };
    defer h.fx.deinit();
    defer h.stream.deinit();
    defer h.deinit();
    const layer: usize = @intCast(try h.fx.int("layer"));
    const capacity: usize = @intCast(try h.fx.int("capacity"));
    const long_capacity: usize = @intCast(try h.fx.int("long_capacity"));

    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var mk = try M.Kernels.load(d);
    defer mk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    var ranks: [2]Rank = undefined;
    var loaded: usize = 0;
    defer for (ranks[0..loaded]) |*r| r.deinit();
    for (&ranks, 0..) |*r, ri| {
        const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
        const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
        r.frame = try glm.weights.Frame.load(l, layer);
        r.dsa = try A.Dsa.load(l, layer);
        r.moe = try M.Moe.load(l, layer);
        r.norm = try l.share(glm.weights_map.norm);
        r.embed = try l.share(glm.weights_map.embed);
        r.b = try F.Buffers.init(d, lay, max_rows, false);
        r.pb = try F.Buffers.init(d, lay, prompt_rows, true);
        r.ms = try M.Scratch.init(d, lay, max_rows, false);
        r.pms = try M.Scratch.init(d, lay, prompt_rows, true);
        r.fnormed = try Buf.alloc(d, prompt_rows * cfg.hidden * 2);
        r.fxs = try Buf.alloc(d, prompt_rows * cfg.hidden / 64 * 4);
        r.pos_dev = try Buf.alloc(d, 16);
        loaded += 1;
    }
    try weightsOk(gpa, gpu.io, work, &ranks);
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams };
    const steps = h.fx.parsed.value.object.get("params").?.object.get("steps").?.object;

    {
        var run_a = try Run.init(d, &cfg, capacity);
        defer run_a.deinit();
        try runSteps(&h, g, &mk, &cfg, &ranks, &run_a, steps.get("a").?.array.items, "a", false);
    }

    var run_l = try Run.init(d, &cfg, long_capacity);
    defer run_l.deinit();
    for (&run_l.cache) |*c| {
        const lat = try h.fx.bytes("l_latent0");
        defer gpa.free(lat);
        try c.latent.upload(0, lat);
        const pk = try h.fx.bytes("l_pk0");
        defer gpa.free(pk);
        try c.pk.upload(0, pk);
    }
    try runSteps(&h, g, &mk, &cfg, &ranks, &run_l, steps.get("l").?.array.items, "l", true);
    try check.expect(h.failures == 0, "glm53-dsa: {d} comparisons differ ({d} equal)", .{ h.failures, h.passes });
    check.pass("glm53-dsa: runs a and l, {d} comparisons bit-exact{s}", .{ h.passes, if (isolated) " (isolated)" else "" });
}

fn runSteps(h: *Harness, g: glm.glue.Glue, mk: *const M.Kernels, cfg: *const glm.config.Config, ranks: *[2]Rank, run_: *Run, steps: []const std.json.Value, comptime tag: []const u8, long: bool) !void {
    const stream = h.stream;
    const D = cfg.hidden;
    const d = h.gpu.d;
    for (steps, 0..) |sv, s| {
        const kind = sv.object.get("kind").?.string;
        const R: usize = @intCast(sv.object.get("rows").?.integer);
        const keep: usize = @intCast(sv.object.get("keep").?.integer);
        const pos: usize = @intCast(sv.object.get("pos").?.integer);
        const prefill = kind[0] == 'p';
        const before = h.failures;
        var nb: [16]u8 = undefined;
        const step = try std.fmt.bufPrint(&nb, tag ++ "{d}", .{s});
        // the attention site: embed, hc_pre, the DSA block
        for (ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
            const cx: F.Ctx = .{ .g = g, .kk = undefined, .temp = undefined, .prompt_scratch = undefined, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            const ds = if (prefill) &run_.pds[ri] else &run_.ds[ri];
            const cache = &run_.cache[ri];
            try h.beginStep(ri, step);
            const p32: i32 = @intCast(pos);
            try r.pos_dev.upload(0, std.mem.asBytes(&p32));
            try h.feed("ids", b.ids);
            try g.embed(b.ids.ptr, r.embed.ptr, b.x.ptr, R, cfg.streams, stream);
            try h.call("embed", "out_out", b.x.ptr);
            try F.hcPre(cx, &r.frame, false, b, R, stream);
            try h.call("hc_pre", "out_out", b.normed.ptr);
            try A.dsaBlock(g, lay, &r.dsa, cache, ds, .{ .normed = b.normed.ptr, .part = b.part.ptr, .sk = b.sk.ptr, .pos_dev = r.pos_dev.ptr, .host_pos = pos, .nch = (pos + R + 511) / 512, .R = R }, stream);
            try h.call("mm", "out_out", ds.dp.ptr);
            try h.call("rmsnorm", "out_out", ds.qr.ptr);
            try h.call("rmsnorm", "out_out", ds.lat.ptr);
            try h.call("mm", "out_out", ds.q.ptr);
            if (!long) try h.call("latent_write", "cache_out", cache.latent.ptr) else _ = try h.next("latent_write");
            try h.call("mm", "out_out", ds.ikr.ptr);
            try h.call("router", "out_out", ds.igr.ptr);
            _ = try h.next("index_update");
            try h.alsoIf("index_update", "ik_out", cache.ik.ptr);
            try h.alsoIf("index_update", "ig_out", cache.ig.ptr);
            if (!long) try h.alsoIf("index_update", "pk_out", cache.pk.ptr);
            try h.call("absorb_q", "out_out", ds.qa.ptr);
            const limit = cfg.denseLimit();
            if (pos < limit) try h.call("dense_attention", "out_out", ds.ol.ptr);
            if (pos + R - 1 >= limit) {
                try h.call("mm", "out_out", ds.qi.ptr);
                try h.call("select_tokens", "ret0", ds.tokens.ptr);
                try h.also("select_tokens", "ret1", ds.counts.ptr);
                try h.call(if (prefill) "sparse_onepass" else "sparse_attention", "out_out", ds.ol.ptr);
            }
            try h.call("expand_v", "out_out", ds.vn.ptr);
            try h.call("mm", "out_out", b.part.ptr);
        }
        try gatherRanks(ranks, prefill, R * D * 4, d, stream);
        // the FFN site: hc_post, hc_pre, the MoE
        for (ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
            const cx: F.Ctx = .{ .g = g, .kk = undefined, .temp = undefined, .prompt_scratch = undefined, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            const sc = if (prefill) &r.pms else &r.ms;
            try h.beginStep(ri, step);
            try h.call("gather", "ret", b.gath.ptr);
            try F.hcPost(cx, b, R, stream);
            try h.call("hc_post", "xout_out", b.x.ptr);
            try F.hcPre(cx, &r.frame, true, b, R, stream);
            try h.call("hc_pre", "out_out", b.normed.ptr);
            try M.moeBlock(g, mk, lay, &r.moe, sc, b.normed.ptr, b.part.ptr, b.sk.ptr, R, stream);
            try h.call("combine_shared", "out_out", b.part.ptr);
        }
        try gatherRanks(ranks, prefill, R * D * 4, d, stream);
        // the layer's end and the final norm (the MTP head's input)
        for (ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
            const cx: F.Ctx = .{ .g = g, .kk = undefined, .temp = undefined, .prompt_scratch = undefined, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.beginStep(ri, step);
            try F.hcPost(cx, b, R, stream);
            try h.stepArray("x_out", b.x.ptr);
            try g.streamMean(b.x.ptr, b.hidden.ptr, R, stream);
            try g.rmsnorm(b.hidden.ptr, D, r.norm.ptr, r.fnormed.ptr, D, r.fxs.ptr, cfg.eps, R, D, stream);
            try h.stepArray("fnormed", r.fnormed.ptr);
        }
        _ = keep; // the DSA caches commit by position: the next step's pos (the fixture's) is where it writes
        if (h.failures == before) check.pass("BITEXACT glm53 layer 3 step {s} ({s} {d} rows at {d}): both ranks equal to the reference", .{ step, kind, R, pos });
    }
}

fn gatherRanks(ranks: *[2]Rank, prefill: bool, n: usize, d: *const cuda.Driver, stream: cuda.Stream) !void {
    var parts: [2]Ptr = undefined;
    var gaths: [2]Ptr = undefined;
    for (ranks, 0..) |*r, i| {
        const b = if (prefill) &r.pb else &r.b;
        parts[i] = b.part.ptr;
        gaths[i] = b.gath.ptr;
    }
    try hs.gather(&parts, &gaths, n, d, stream);
}
