//! `qwen38fn-layer0 <work-dir> <model-dir> [isolated]`: Qwen3.8-Flash-Next's embeddings and decoder layer 0 (a
//! DeltaNet layer) against TensorFold 0.6.6's stage-by-stage fixture (tools/zig/qwen38fn/dump_layer0.py, packed by
//! capture_layer0.sh): `work-dir` holds out/fixture (manifest.json + arrays) and aot/ (the captured Triton variants);
//! the weights come from the EXL3 pack in `model-dir` through the family's loader.
//!
//! One stream decodes the fixture's windows in order, carrying its own conv window and recurrent state from window to
//! window. Every stage's bytes are compared with the reference's, so a first mismatch names the kernel. By default each
//! stage runs on this engine's own outputs of the stage before; `isolated` feeds every stage the reference's inputs
//! instead (a mismatch then cannot hide the stages after it).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const qfn = @import("qwen38fn");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;
const Glue = qfn.glue.Glue;

const Harness = struct {
    gpu: Gpu,
    fx: fixture.Fixture,
    stream: cuda.Stream,
    isolated: bool,
    failures: usize = 0,
    passes: usize = 0,
    buf: [96]u8 = undefined,

    fn name(self: *Harness, i: usize, what: []const u8) ![]const u8 {
        return std.fmt.bufPrint(&self.buf, "w{d}_{s}", .{ i, what });
    }

    fn want(self: *Harness, i: usize, what: []const u8) ![]u8 {
        return self.fx.bytes(try self.name(i, what));
    }

    /// The device bytes at `buf` + `offset` against the fixture's `w{i}_{what}`: a PASS, or a FAIL counted (with
    /// the first differing byte). Isolated runs then put the reference's bytes there for the next stage.
    fn stage(self: *Harness, i: usize, what: []const u8, buf: cuda.DeviceBuffer, offset: usize) !void {
        try self.stream.synchronize();
        const w = try self.want(i, what);
        defer self.gpu.gpa.free(w);
        const got = try self.gpu.gpa.alloc(u8, w.len);
        defer self.gpu.gpa.free(got);
        try buf.download(offset, got);
        if (std.mem.indexOfDiff(u8, got, w)) |at| {
            var diff: usize = 0;
            for (got, w) |a, b| diff += @intFromBool(a != b);
            std.debug.print("FAIL qwen38fn w{d} {s}: {d} of {d} bytes differ, the first at byte {d}\n", .{ i, what, diff, w.len, at });
            self.failures += 1;
        } else self.passes += 1;
        if (self.isolated) try buf.upload(offset, w);
    }

    /// The fixture array `w{i}_{what}` into `buf` at `offset`.
    fn feed(self: *Harness, i: usize, what: []const u8, buf: cuda.DeviceBuffer, offset: usize) !void {
        const w = try self.want(i, what);
        defer self.gpu.gpa.free(w);
        try buf.upload(offset, w);
    }
};

/// One EXL3 group as the DSV4.1 port's EXL3 linear (the same kernels as 0.6.6's `cuda/exl3/linear.cu`; this test
/// is what checks that its bits are 0.6.6's).
fn linear(gpa: std.mem.Allocator, k: *const dsv41.exl3.Kernels, g: qfn.weights.Group) !dsv41.exl3.Linear {
    // the mapped trellis may sit at any even offset of its shard; Linear.init reads it as 32-bit words
    const words = try gpa.alignedAlloc(u8, .@"4", g.trellis.len);
    defer gpa.free(words);
    @memcpy(words, g.trellis);
    return dsv41.exl3.Linear.init(gpa, k, words, g.kt, g.nt, g.w16, g.suh, g.svh, @enumFromInt(@intFromEnum(g.codebook)));
}

pub fn layer0(gpu: Gpu, work: []const u8, model: []const u8, isolated: bool) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: qfn.config.Why = .{};
    const cfg = qfn.config.read(gpa, gpu.io, model, &why) catch |e| {
        std.debug.print("FAIL qwen38fn config: {t}: {s}\n", .{ e, why.text() });
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
    const fx = h.fx;
    const stream = h.stream;

    // the fixture's geometry is the config's
    try check.expect(@as(usize, @intCast(try fx.int("hidden"))) == cfg.hidden and @as(usize, @intCast(try fx.int("proj_width"))) == cfg.projWidth() and
        @as(usize, @intCast(try fx.int("slots"))) == cfg.slots(), "qwen38fn: the fixture's shapes are not the config's", .{});

    // weights (layer 0 of the pack, prepared as 0.6.6 prepares them)
    var pack = try qfn.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    const offset = try qfn.weights.centredOffset(gpa, &pack, &cfg);
    try check.expect(offset == @as(f32, @floatCast(try fx.float("centred_offset"))), "qwen38fn: centred norm offset {d}, the reference's {d}", .{ offset, try fx.float("centred_offset") });
    var embed = try qfn.weights.embedding(d, &pack, &cfg);
    defer embed.free();
    var name: qfn.weights_map.Name = .{};
    var hc_a = try qfn.weights.Hc.load(gpa, d, &pack, &cfg, name.of(qfn.weights_map.lm ++ "layers.0.{s}", .{"attn_hyper_connection"}), true, offset);
    defer hc_a.deinit();
    var hc_m = try qfn.weights.Hc.load(gpa, d, &pack, &cfg, name.of(qfn.weights_map.lm ++ "layers.0.{s}", .{"mlp_hyper_connection"}), true, offset);
    defer hc_m.deinit();
    try check.expect(hc_a.down.sk == @as(usize, @intCast(try fx.int("f16_sk_down"))) and hc_a.up.sk == @as(usize, @intCast(try fx.int("f16_sk_up"))), "qwen38fn: hyper-connection K slices {d}/{d}, the reference's {d}/{d}", .{ hc_a.down.sk, hc_a.up.sk, try fx.int("f16_sk_down"), try fx.int("f16_sk_up") });
    var gw = try qfn.weights.Gdn.load(gpa, d, &pack, &cfg, 0);
    defer gw.deinit();
    try check.expect(gw.ba.sk == @as(usize, @intCast(try fx.int("f16_sk_ba"))), "qwen38fn: (b ; a) K slices {d}, the reference's {d}", .{ gw.ba.sk, try fx.int("f16_sk_ba") });
    var router = try qfn.weights.router(gpa, d, &pack, &cfg, 0);
    defer router.free();
    var xk = try dsv41.exl3.Kernels.load(d);
    defer xk.deinit();
    var qkv = try linear(gpa, &xk, gw.qkv);
    defer qkv.deinit();
    var zl = try linear(gpa, &xk, gw.z);
    defer zl.deinit();
    var out = try linear(gpa, &xk, gw.out);
    defer out.deinit();
    var gk = try qfn.gdn.Kernels.load(d, cfg.nv);
    defer gk.deinit();
    check.pass("qwen38fn: layer 0 loaded (centred offset {d}, K slices down {d} up {d} (b;a) {d}, EXL3 plans qkv {d}x{d} z {d}x{d} out {d}x{d})", .{ offset, hc_a.down.sk, hc_a.up.sk, gw.ba.sk, qkv.sk, qkv.wk, zl.sk, zl.wk, out.sk, out.wk });

    // buffers
    const rows: usize = @intCast(try fx.int("rows"));
    const pw = cfg.projWidth();
    const conv_dim = cfg.convDim();
    const vw = cfg.nv * cfg.dv;
    const slots = cfg.slots();
    const part_floats = @max(hc_a.down.sk * rows * hc_a.down.n, gw.ba.sk * rows * gw.ba.n);
    var w = try qfn.hc.Window.init(d, &cfg, rows, part_floats);
    defer w.deinit();
    var ids = try cuda.DeviceBuffer.alloc(d, rows * 4);
    defer ids.free();
    var xh = try cuda.DeviceBuffer.alloc(d, rows * @max(cfg.hidden, vw) * 2);
    defer xh.free();
    var z = try cuda.DeviceBuffer.alloc(d, @max(16, 4 * @max(qkv.scratchWords(rows), @max(zl.scratchWords(rows), out.scratchWords(rows)))));
    defer z.free();
    const state_bytes = cfg.nv * cfg.dv * cfg.dk * 4;
    var rec = try cuda.DeviceBuffer.alloc(d, 2 * state_bytes);
    defer rec.free();
    try rec.fill8(0, null);
    var conv = try cuda.DeviceBuffer.alloc(d, (cfg.conv_kernel - 1) * conv_dim * 2);
    defer conv.free();
    try conv.fill8(0, null);
    var scratch = try qfn.gdn.Scratch.init(d, rows, cfg.nk, cfg.nv);
    defer scratch.deinit();
    var logits = try cuda.DeviceBuffer.alloc(d, rows * (cfg.experts + 1) * 4);
    defer logits.free();
    var pick = try cuda.DeviceBuffer.alloc(d, rows * slots * 4);
    defer pick.free();
    var wts = try cuda.DeviceBuffer.alloc(d, rows * slots * 4);
    defer wts.free();
    var xk_moe = try qfn.experts.Kernels.load(d);
    defer xk_moe.deinit();
    var experts = try qfn.experts.Experts.load(gpa, d, &pack, &cfg, 0);
    defer experts.deinit();
    var moe_scratch = try qfn.experts.Scratch.init(d, &experts, rows, slots);
    defer moe_scratch.deinit();
    var cur: usize = 0;

    const g: Glue = .{ .set = &set, .hidden = cfg.hidden, .streams = cfg.streams, .low = cfg.low };
    const windows: usize = @intCast(try fx.int("windows"));
    var nb: [32]u8 = undefined;
    for (0..windows) |i| {
        const R: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_rows", .{i})));
        const keep: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_keep", .{i})));
        const before = h.failures;
        try h.feed(i, "ids", ids, 0);
        // the carried state is the reference's state before this window
        try h.stage(i, "conv_in", conv, 0);
        try h.stage(i, "rec_in", rec, cur * state_bytes);

        try g.embed(ids.ptr, embed.ptr, w.h.ptr, R, stream);
        try h.stage(i, "h0", w.h, 0);

        // the attention hyper-connection (mode 0: nothing pending at layer 0)
        try g.hcWriteback(w.h.ptr, w.pss.ptr, R, .none, stream);
        try h.stage(i, "a_pss", w.pss, 0);
        try readoutStaged(&h, g, &w, &hc_a, &cfg, i, R, w.inj_a.ptr, "a_", stream);
        try h.stage(i, "inj_a", w.inj_a, 0);

        // the DeltaNet projection [qkv | z | b ; a] (exl3_mm.py Stack: one row straight into its columns, more rows
        // through the scratch rows and a copy)
        {
            const parts = [_]struct { ?*const dsv41.exl3.Linear, usize, usize }{ .{ &qkv, 0, conv_dim }, .{ &zl, conv_dim, vw }, .{ null, conv_dim + vw, gw.ba.n } };
            for (parts) |p| {
                const lin, const at, const n = p;
                const direct = R == 1;
                const dst = if (direct) w.proj.ptr + at * 2 else w.tmp.ptr;
                if (lin) |l| {
                    try l.run(w.mixed.ptr, .bf16, R, xh.ptr, z.ptr, dst, .bf16, stream);
                } else {
                    try g.f16Call(&gw.ba, w.mixed.ptr, cfg.hidden, R, dst, if (direct) pw else n, w.part.ptr, stream);
                }
                if (!direct) for (0..R) |r| try w.proj.copyFrom((r * pw + at) * 2, w.tmp.ptr + r * n * 2, n * 2, stream.handle);
            }
        }
        try h.stage(i, "proj", w.proj, 0);

        // the chain: rec[cur] -> rec[1 - cur]
        try qfn.gdn.chain(&gk, .{
            .p = w.proj.ptr,       .conv_state = conv.ptr,     .conv_w = gw.conv.ptr, .state_in = rec.ptr + cur * state_bytes,
            .a_log = gw.a_log.ptr, .dt_bias = gw.dt_bias.ptr,  .norm_w = gw.norm.ptr, .eps = cfg.eps,
            .rows = R,             .out = w.gout.ptr,          .xs = w.gxs.ptr,       .state_out = rec.ptr + (1 - cur) * state_bytes,
            .scratch = &scratch,
        }, stream);
        try h.stage(i, "gout", w.gout, 0);
        try h.stage(i, "gxs", w.gxs, 0);
        try h.stage(i, "rec_out", rec, (1 - cur) * state_bytes);
        try h.stage(i, "sk", scratch.k, 0);
        try h.stage(i, "sv", scratch.v, 0);
        try h.stage(i, "sg", scratch.g, 0);
        try h.stage(i, "sb", scratch.b, 0);
        try out.run(w.gout.ptr, .bf16, R, xh.ptr, z.ptr, w.branch.ptr, .bf16, stream);
        try h.stage(i, "branch", w.branch, 0);

        // the MLP hyper-connection: the branch written back (mode 1), then its read-out
        try g.hcWriteback(w.h.ptr, w.pss.ptr, R, .{ .branch = .{ .branch = w.branch.ptr, .inject = w.inj_a.ptr } }, stream);
        try h.stage(i, "h1", w.h, 0);
        try h.stage(i, "m_pss", w.pss, 0);
        try readoutStaged(&h, g, &w, &hc_m, &cfg, i, R, w.inj_m.ptr, "m_", stream);
        try h.stage(i, "inj_m", w.inj_m, 0);

        // routing
        try g.router(w.mixed.ptr, cfg.hidden, router.ptr, cfg.experts + 1, logits.ptr, R, stream);
        try h.stage(i, "logits", logits, 0);
        try g.selectRows(logits.ptr, cfg.experts, cfg.topk, pick.ptr, wts.ptr, R, stream);
        try h.stage(i, "pick", pick, 0);
        try h.stage(i, "wts", wts, 0);

        // the routed experts and the shared one, then the MoE write-back (mode 2): layer 0's output streams
        _ = try qfn.experts.routed(&xk_moe, &experts, &moe_scratch, w.mixed.ptr, cfg.hidden, pick.ptr, R, stream);
        try h.stage(i, "y", moe_scratch.y, 0);
        try g.hcWriteback(w.h.ptr, w.pss.ptr, R, .{ .moe = .{ .y = moe_scratch.y.ptr, .wts = wts.ptr, .inject = w.inj_m.ptr, .slots = slots } }, stream);
        try h.stage(i, "h2", w.h, 0);
        try h.stage(i, "pss2", w.pss, 0);

        // commit keep rows (forward.py `commit`): a rejected tail replays the state; the conv window shifts
        if (keep < R) try qfn.gdn.replay(&gk, rec.ptr + cur * state_bytes, &scratch, keep, rec.ptr + (1 - cur) * state_bytes, stream);
        cur = 1 - cur;
        try g.shiftWindows(conv.ptr, (cfg.conv_kernel - 1) * conv_dim, w.proj.ptr, rows * pw, pw, keep, 1, conv_dim, cfg.conv_kernel - 1, stream);
        try h.stage(i, "conv_out", conv, 0);
        try h.stage(i, "rec_commit", rec, cur * state_bytes);
        if (h.failures == before) check.pass("BITEXACT qwen38fn layer 0 window {d} ({d} rows, {d} kept): every stage equal to 0.6.6's", .{ i, R, keep });
    }
    try check.expect(h.failures == 0, "qwen38fn-layer0: {d} stages differ ({d} equal)", .{ h.failures, h.passes });
    check.pass("qwen38fn-layer0: {d} windows, {d} stages bit-exact{s}", .{ windows, h.passes, if (isolated) " (isolated)" else "" });
}

/// cuda_hc.zig `readout`'s steps one by one, each compared (the fixture's `{tag}normed` ... `{tag}xs_mixed`).
fn readoutStaged(h: *Harness, g: Glue, w: *qfn.hc.Window, hc: *const qfn.weights.Hc, cfg: *const qfn.config.Config, i: usize, R: usize, inject: cuda.abi.DevicePtr, comptime tag: []const u8, stream: cuda.Stream) !void {
    const wide = cfg.wide();
    try g.hcNormed(w.h.ptr, w.pss.ptr, hc.scale.ptr, w.normed.ptr, w.xs_normed.ptr, cfg.eps, R, stream);
    try h.stage(i, tag ++ "normed", w.normed, 0);
    try h.stage(i, tag ++ "xs_normed", w.xs_normed, 0);
    try g.f16mm(&hc.down, w.normed.ptr, wide, R, w.part.ptr, 0, true, stream);
    try h.stage(i, tag ++ "part", w.part, 0);
    try g.hcReduceAct(w.part.ptr, hc.down.sk, hc.down.n, w.act.ptr, w.xs_act.ptr, inject, R, stream);
    try h.stage(i, tag ++ "act", w.act, 0);
    try h.stage(i, tag ++ "xs_act", w.xs_act, 0);
    try g.f16Call(&hc.up, w.act.ptr, cfg.low, R, w.up.ptr, wide, w.part.ptr, stream);
    try h.stage(i, tag ++ "up", w.up, 0);
    try g.hcMix(w.up.ptr, w.normed.ptr, w.mixed.ptr, w.xs_mixed.ptr, R, stream);
    try h.stage(i, tag ++ "mixed", w.mixed, 0);
    try h.stage(i, tag ++ "xs_mixed", w.xs_mixed, 0);
}
