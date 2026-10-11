//! `glm53-moe <work-dir> <model-dir> [isolated]`: GLM-5.3-Flash's MoE block of one layer (router, top-8 of 288, the
//! plan, the EXL3 routed experts in their decode and prompt forms, the BF16 shared expert, the weighted sum, the
//! all-gather and hc_post) on both ranks against the Mia fork's call-by-call fixture (tools/zig/glm53/dump_moe.py,
//! packed by capture.sh), every loaded tensor's sha256 against the reference loader's. Both ranks run on this one
//! GPU; see glm53_harness.zig.
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
const Buf = cuda.DeviceBuffer;

const Rank = struct {
    frame: glm.weights.Frame,
    moe: M.Moe,
    b: F.Buffers,
    pb: F.Buffers,
    s: M.Scratch,
    ps: M.Scratch,

    fn deinit(r: *Rank) void {
        r.frame.deinit();
        r.moe.deinit();
        r.b.deinit();
        r.pb.deinit();
        r.s.deinit();
        r.ps.deinit();
    }
};

fn checkWeights(gpa: std.mem.Allocator, io: std.Io, work: []const u8, layer: usize, ranks: []Rank) !void {
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
        const x = &r.moe.experts;
        const pairs = [_]struct { []const u8, Buf }{
            .{ "ffn_hc.fn", r.frame.ffn_hc.fn_ },  .{ "ffn_hc.base", r.frame.ffn_hc.base },      .{ "ffn_hc.scale", r.frame.ffn_hc.scale },
            .{ "post_norm", r.frame.post_norm },   .{ "moe.router", r.moe.router },              .{ "moe.bias", r.moe.bias },
            .{ "moe.experts.gt", x.gt },           .{ "moe.experts.ut", x.ut },                  .{ "moe.experts.dt", x.dt },
            .{ "moe.experts.suh_g", x.suh_g },     .{ "moe.experts.suh_u", x.suh_u },            .{ "moe.experts.svh_g", x.svh_g },
            .{ "moe.experts.svh_u", x.svh_u },     .{ "moe.experts.suh_d", x.suh_d },            .{ "moe.experts.svh_d", x.svh_d },
            .{ "moe.experts.shared_suh", x.shared_suh }, .{ "moe.shared.gu.weight", r.moe.shared.gu.buf }, .{ "moe.shared.down.weight", r.moe.shared.down.buf },
        };
        for (pairs) |p| {
            var key: [64]u8 = undefined;
            const k = try std.fmt.bufPrint(&key, "layers.{d}.{s}", .{ layer, p[0] });
            const got = try hs.sha(gpa, p[1]);
            n += 1;
            if (!std.mem.eql(u8, &got, want.get(k).?.string)) {
                std.debug.print("FAIL glm53 rank {d} {s}: sha256 {s}, the reference's {s}\n", .{ ri, k, got[0..16], want.get(k).?.string[0..16] });
                bad += 1;
            }
        }
    }
    try check.expect(bad == 0, "glm53-moe: {d} of {d} loaded tensors differ from the reference loader's", .{ bad, n });
    check.pass("glm53-moe: every rank's layer-{d} MoE tensors equal the reference loader's ({d} sha256)", .{ layer, n });
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
    const stream = h.stream;
    const layer: usize = @intCast(try h.fx.int("layer"));
    const rows: usize = @intCast(try h.fx.int("rows"));
    const prompt_rows: usize = @intCast(try h.fx.int("prompt_rows"));

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
        r.moe = try M.Moe.load(l, layer);
        r.b = try F.Buffers.init(d, lay, rows, false);
        r.pb = try F.Buffers.init(d, lay, prompt_rows, true);
        r.s = try M.Scratch.init(d, lay, rows, false);
        r.ps = try M.Scratch.init(d, lay, prompt_rows, true);
        loaded += 1;
    }
    try checkWeights(gpa, gpu.io, work, layer, &ranks);
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams };

    const steps = h.fx.parsed.value.object.get("params").?.object.get("steps").?.array.items;
    const D = cfg.hidden;
    for (steps, 0..) |sv, s| {
        const kind = sv.object.get("kind").?.string;
        const R: usize = @intCast(sv.object.get("rows").?.integer);
        const prefill = kind[0] == 'p';
        const before = h.failures;
        for (&ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
            const cx: F.Ctx = .{ .g = g, .kk = undefined, .temp = undefined, .prompt_scratch = undefined, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            const sc = if (prefill) &r.ps else &r.s;
            try h.begin(ri, s);
            try h.feed("x_in", b.x);
            try F.hcPre(cx, &r.frame, true, b, R, stream);
            try h.stepArray("normed", b.normed.ptr);
            try h.stepArray("xs", b.xs.ptr);
            try M.moeBlock(g, &mk, lay, &r.moe, sc, b.normed.ptr, b.part.ptr, b.sk.ptr, R, stream);
            try h.call("router", "out_out", sc.logits.ptr);
            try h.call("select", "pick_out", sc.pick.ptr);
            try h.also("select", "wts_out", sc.wts.ptr);
            try h.stepArray("plan_members", sc.members.ptr);
            try h.stepArray("plan_counts", sc.counts.ptr);
            try h.stepArray("xg", sc.xg.ptr); // zero-initialised as exl3_mm.Scratch: the same untouched rows
            try h.stepArray("xd", sc.xd.ptr);
            if (!prefill) {
                // the routed slots' rows of y (the shared slot's rows are never written: exl3.cu skips id E)
                var starts: [16 * 9]usize = undefined;
                var n: usize = 0;
                for (0..R) |row| for (0..cfg.top_k) |k| {
                    starts[n] = ((row * cfg.slots()) + k) * D * 4;
                    n += 1;
                };
                const seq = try h.next("routed");
                try h.sameRanges(try h.arrayName(seq, "routed", "y_out"), sc.y.ptr, starts[0..n], D * 4);
            }
            try h.call("mm", "out_out", sc.sgu.ptr);
            try h.call("swiglu", "out_out", sc.sact.ptr);
            try h.call("mm", "out_out", sc.sy.ptr);
            try h.call("combine_shared", "out_out", b.part.ptr);
        }
        var parts: [2]cuda.abi.DevicePtr = undefined;
        var gaths: [2]cuda.abi.DevicePtr = undefined;
        for (&ranks, 0..) |*r, i| {
            const b = if (prefill) &r.pb else &r.b;
            parts[i] = b.part.ptr;
            gaths[i] = b.gath.ptr;
        }
        try hs.gather(&parts, &gaths, R * D * 4, d, stream);
        for (&ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
            const cx: F.Ctx = .{ .g = g, .kk = undefined, .temp = undefined, .prompt_scratch = undefined, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.begin(ri, s);
            try h.call("gather", "ret", b.gath.ptr);
            try F.hcPost(cx, b, R, stream);
            try h.call("hc_post", "xout_out", b.x.ptr);
            try h.stepArray("x_out", b.x.ptr);
        }
        if (h.failures == before) check.pass("BITEXACT glm53 layer {d} MoE step {d} ({s} {d} rows): both ranks equal to the reference", .{ layer, s, kind, R });
    }
    try check.expect(h.failures == 0, "glm53-moe: {d} comparisons differ ({d} equal)", .{ h.failures, h.passes });
    check.pass("glm53-moe: {d} steps, {d} comparisons bit-exact{s}", .{ steps.len, h.passes, if (isolated) " (isolated)" else "" });
}
