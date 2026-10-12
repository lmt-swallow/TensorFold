//! `glm53-kda <work-dir> <model-dir> [isolated]`: GLM-5.3-Flash's embeddings and decoder layer 0 (KDA, the dense
//! MLP, both hyper-connections) on both ranks of the two-rank split against the Mia fork's call-by-call fixture
//! (tools/zig/glm53/dump_kda.py, packed by capture.sh): `work-dir` holds out/fixture (manifest.json + arrays, the
//! call trace), out/weights.json and aot/ (the captured Triton variants); the weights come from the EXL3 pack in
//! `model-dir` through the family's loader, each rank's share checked against the reference's sha256.
//!
//! Both ranks run on this one GPU, stage by stage; their all-gathers are copies of the two partials into each rank's
//! gather buffer (the bytes NCCL would move). Every traced call's outputs are compared with the reference's in trace
//! order, so a first mismatch names the call. By default each stage runs on this engine's own outputs; `isolated`
//! puts the reference's bytes back after every comparison (a mismatch then cannot hide the stages after it).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const hs = @import("glm53_harness.zig");
const Harness = hs.Harness;
const sha = hs.sha;
const Gpu = check.Gpu;

const F = glm.forward;
const Buf = cuda.DeviceBuffer;

/// One rank's layer 0 and its buffers.
const Rank = struct {
    frame: glm.weights.Frame,
    kda: glm.weights.Kda,
    mlp: glm.weights.Mlp,
    embed: Buf,
    b: F.Buffers,
    pb: F.Buffers,
    st: F.KdaState,

    fn deinit(r: *Rank) void {
        r.frame.deinit();
        r.kda.deinit();
        r.mlp.deinit();
        r.embed.free();
        r.b.deinit();
        r.pb.deinit();
        r.st.deinit();
    }
};

/// Each loaded tensor's sha256 against weights.json's (the reference loader's bytes).
fn checkWeights(gpa: std.mem.Allocator, io: std.Io, work: []const u8, ranks: []Rank) !void {
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
        const pairs = [_]struct { []const u8, Buf }{
            .{ "layers.0.attn_hc.fn", r.frame.attn_hc.fn_ },   .{ "layers.0.attn_hc.base", r.frame.attn_hc.base }, .{ "layers.0.attn_hc.scale", r.frame.attn_hc.scale },
            .{ "layers.0.ffn_hc.fn", r.frame.ffn_hc.fn_ },     .{ "layers.0.ffn_hc.base", r.frame.ffn_hc.base },   .{ "layers.0.ffn_hc.scale", r.frame.ffn_hc.scale },
            .{ "layers.0.in_norm", r.frame.in_norm },          .{ "layers.0.post_norm", r.frame.post_norm },       .{ "layers.0.kda.proj.weight", r.kda.proj.buf },
            .{ "layers.0.kda.fb.weight", r.kda.fb.buf },       .{ "layers.0.kda.gb.weight", r.kda.gb.buf },        .{ "layers.0.kda.conv", r.kda.conv },
            .{ "layers.0.kda.a_log", r.kda.a_log },            .{ "layers.0.kda.dt_bias", r.kda.dt_bias },         .{ "layers.0.kda.norm", r.kda.norm },
            .{ "layers.0.kda.o.weight", r.kda.o.buf },         .{ "layers.0.mlp.gu.weight", r.mlp.gu.buf },        .{ "layers.0.mlp.down.weight", r.mlp.down.buf },
            .{ "embed", r.embed },
        };
        for (pairs) |p| {
            const got = try sha(gpa, p[1]);
            n += 1;
            if (!std.mem.eql(u8, &got, want.get(p[0]).?.string)) {
                std.debug.print("FAIL glm53 rank {d} {s}: sha256 {s}, the reference's {s}\n", .{ ri, p[0], got[0..16], want.get(p[0]).?.string[0..16] });
                bad += 1;
            }
        }
    }
    try check.expect(bad == 0, "glm53-kda: {d} of {d} loaded tensors differ from the reference loader's", .{ bad, n });
    check.pass("glm53-kda: every rank's layer-0 tensors equal the reference loader's ({d} sha256)", .{n});
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
    // a blocking stream: the harness's synchronous uploads from pageable memory land before its next kernels
    var h: Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir), .stream = try cuda.Stream.init(d, false), .isolated = isolated };
    defer h.fx.deinit();
    defer h.stream.deinit();
    defer h.deinit();
    const stream = h.stream;
    const rows: usize = @intCast(try h.fx.int("rows"));
    const prompt_rows: usize = @intCast(try h.fx.int("prompt_rows"));

    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var kk = try glm.kda.Kernels.load(d);
    defer kk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    var ranks: [2]Rank = undefined;
    var loaded: usize = 0;
    defer for (ranks[0..loaded]) |*r| r.deinit();
    for (&ranks, 0..) |*r, ri| {
        const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
        const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
        r.frame = try glm.weights.Frame.load(l, 0);
        r.kda = try glm.weights.Kda.load(l, 0);
        r.mlp = try glm.weights.Mlp.load(l, "mlp.", 0);
        r.embed = try l.share(glm.weights_map.embed);
        r.b = try F.Buffers.init(d, lay, rows, false);
        r.pb = try F.Buffers.init(d, lay, prompt_rows, true);
        r.st = try F.KdaState.init(d, lay, 1, rows);
        try r.st.rec.fill8(0, null);
        try r.st.conv.fill8(0, null);
        loaded += 1;
    }
    try checkWeights(gpa, gpu.io, work, &ranks);
    const lay0: glm.config.Layout = .{ .rank = 0, .world = 2, .cfg = &cfg };
    const H = lay0.size(.lin);
    var temp = try glm.kda.Temp.init(d, @max(rows, prompt_rows), H);
    defer temp.deinit();
    var pscratch = try glm.kda.Scratch.init(d, 1, prompt_rows, H);
    defer pscratch.deinit();
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams };

    const steps = h.fx.parsed.value.object.get("params").?.object.get("steps").?.array.items;
    const D = cfg.hidden;
    for (steps, 0..) |sv, s| {
        const kind = sv.object.get("kind").?.string;
        const R: usize = @intCast(sv.object.get("rows").?.integer);
        const keep: usize = @intCast(sv.object.get("keep").?.integer);
        const prefill = kind[0] == 'p';
        const before = h.failures;
        // the attention site: embed, hc_pre, the KDA block up to its partial
        for (&ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
            const cx: F.Ctx = .{ .g = g, .kk = &kk, .temp = &temp, .prompt_scratch = &pscratch, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.begin(ri, s);
            try h.stepArray("rec_in", r.st.state(r.st.cur, 0));
            try h.stepArray("conv_in", r.st.convOf(0));
            try h.feed("ids", b.ids);
            try g.embed(b.ids.ptr, r.embed.ptr, b.x.ptr, R, cfg.streams, stream);
            try h.call("embed", "out_out", b.x.ptr);
            try F.hcPre(cx, &r.frame, false, b, R, stream);
            try h.call("hc_pre", "out_out", b.normed.ptr);
            try h.also("hc_pre", "post_out", b.post.ptr);
            try h.also("hc_pre", "comb_out", b.comb.ptr);
            try F.kdaBlock(cx, &r.kda, &r.st, 0, b, R, stream);
            const p = if (prefill) b.kproj.ptr else r.st.proj.ptr;
            try h.call("mm", "out_out", p);
            try h.call("mm", "out_out", b.ka.ptr);
            try h.call("mm", "out_out", b.kg.ptr);
            const sc = if (prefill) pscratch.layer(0) else r.st.scratch.layer(0);
            try h.call("chain", "ret", sc.out);
            try h.also("chain", "state_out_out", r.st.state(1 - r.st.cur, 0));
            if (prefill) try h.call("_shift_conv", "conv_out", r.st.convOf(0));
            try h.call("mm", "out_out", b.part.ptr);
        }
        try gatherRanks(&ranks, prefill, R * D * 4, gpu.d, stream);
        // the FFN site: hc_post, hc_pre, the dense MLP up to its partial
        for (&ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
            const cx: F.Ctx = .{ .g = g, .kk = &kk, .temp = &temp, .prompt_scratch = &pscratch, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.begin(ri, s);
            try h.call("gather", "ret", b.gath.ptr);
            try F.hcPost(cx, b, R, stream);
            try h.call("hc_post", "xout_out", b.x.ptr);
            try F.hcPre(cx, &r.frame, true, b, R, stream);
            try h.call("hc_pre", "out_out", b.normed.ptr);
            try F.mlpBlock(cx, &r.mlp, b, R, stream);
            try h.call("mm", "out_out", b.gu.ptr);
            try h.call("swiglu", "out_out", b.act.ptr);
            try h.also("swiglu", "xs_out", b.xs_act.ptr);
            try h.call("mm", "out_out", b.part.ptr);
        }
        try gatherRanks(&ranks, prefill, R * D * 4, gpu.d, stream);
        // the layer's end: hc_post, the streams' mean, the commit
        for (&ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
            const cx: F.Ctx = .{ .g = g, .kk = &kk, .temp = &temp, .prompt_scratch = &pscratch, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.begin(ri, s);
            try F.hcPost(cx, b, R, stream);
            try h.stepArray("x_out", b.x.ptr);
            try g.streamMean(b.x.ptr, b.hidden.ptr, R, stream);
            try h.stepArray("hidden", b.hidden.ptr);
            try F.kdaCommit(cx, &r.st, R, keep, prefill, stream);
            try h.stepArray("rec_commit", r.st.state(r.st.cur, 0));
            try h.stepArray("conv_commit", r.st.convOf(0));
        }
        if (h.failures == before) check.pass("BITEXACT glm53 layer 0 step {d} ({s} {d} rows, {d} kept): both ranks equal to the reference", .{ s, kind, R, keep });
    }
    try check.expect(h.failures == 0, "glm53-kda: {d} comparisons differ ({d} equal)", .{ h.failures, h.passes });
    check.pass("glm53-kda: {d} steps, {d} comparisons bit-exact{s}", .{ steps.len, h.passes, if (isolated) " (isolated)" else "" });
}

fn gatherRanks(ranks: []Rank, prefill: bool, n: usize, d: *const cuda.Driver, stream: cuda.Stream) !void {
    var parts: [2]cuda.abi.DevicePtr = undefined;
    var gaths: [2]cuda.abi.DevicePtr = undefined;
    for (ranks, 0..) |*r, i| {
        const b = if (prefill) &r.pb else &r.b;
        parts[i] = b.part.ptr;
        gaths[i] = b.gath.ptr;
    }
    try hs.gather(&parts, &gaths, n, d, stream);
}
