//! DeepSeek-V4.1's concurrent paths against the fork's fixtures: the batched DSpark drafts over a DraftPool
//! (tools/zig/dsv41/dump_draft_pool.py, packed by capture_draft_pool.sh): N streams absorbed to their own lengths in
//! their own slots, then one draft pass, every stage's bytes the Python fork's.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const base = @import("dsv41_dspark_tests.zig");
const Gpu = check.Gpu;

const dsp = dsv41.dspark;
const exl3 = dsv41.exl3;

const Prefixed = struct {
    h: *base.Harness,
    prefix: []const u8,
};

fn hook(ctx: *anyopaque, what: []const u8, buf: cuda.abi.DevicePtr) anyerror!void {
    const p: *Prefixed = @ptrCast(@alignCast(ctx));
    var nb: [96]u8 = undefined;
    try p.h.stage(try std.fmt.bufPrint(&nb, "{s}_{s}", .{ p.prefix, what }), buf);
}

/// `dsv41-draft-pool <dir> [aot-dir]`.
pub fn draftPool(gpu: Gpu, dir: []const u8, aot_override: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const aot_dir = if (aot_override) |a| try gpa.dupe(u8, a) else try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_path = try std.fs.path.join(gpa, &.{ dir, "out", "fixture" });
    defer gpa.free(fx_path);
    var h: base.Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(gpa, gpu.io, fx_path), .stream = try cuda.Stream.init(gpu.d, true) };
    defer h.fx.deinit();
    defer h.stream.deinit();
    const fx = h.fx;
    const dims: dsp.Dims = .{
        .d = @intCast(try fx.int("D")),
        .q_rank = @intCast(try fx.int("q_rank")),
        .o_rank = @intCast(try fx.int("o_rank")),
        .heads = @intCast(try fx.int("heads")),
        .hd = @intCast(try fx.int("hd")),
        .rd = @intCast(try fx.int("rd")),
        .groups = @intCast(try fx.int("groups")),
        .window = @intCast(try fx.int("window")),
        .ring = @intCast(try fx.int("ring")),
        .block = @intCast(try fx.int("block")),
        .noise = @intCast(try fx.int("noise")),
        .vocab = @intCast(try fx.int("vocab")),
        .vocab_local = @intCast(@divExact(try fx.int("vocab"), try fx.int("world"))),
        .rank = @intCast(try fx.int("rank")),
        .world = @intCast(try fx.int("world")),
        .markov_rank = @intCast(try fx.int("markov_rank")),
        .n_routed = @intCast(try fx.int("n_routed")),
        .topk = @intCast(try fx.int("topk")),
        .eps = @floatCast(try fx.float("eps")),
        .hc_eps = @floatCast(try fx.float("hc_eps")),
        .hc_iters = @intCast(try fx.int("iters")),
        .swiglu_limit = @floatCast(try fx.float("swiglu_limit")),
        .route_scale = @floatCast(try fx.float("route_scale")),
        .markov_split = (try fx.int("markov_split")) != 0,
    };
    const stages_n: usize = @intCast(try fx.int("stages"));
    const slots_n: usize = @intCast(try fx.int("slots"));
    var kernels = try exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var ex_kernels = try dsv41.experts.Kernels.load(gpu.d);
    defer ex_kernels.deinit();
    var glue = try dsv41.attention.Glue.load(gpu.d);
    defer glue.deinit();
    var main_proj = try base.linearOf(&h, &kernels, "main_proj", .{});
    defer main_proj.deinit();
    var head = try base.linearOf(&h, &kernels, "head", .{});
    defer head.deinit();
    var main_norm = try h.upload("main_norm");
    defer main_norm.free();
    var norm = try h.upload("norm");
    defer norm.free();
    var markov_embed = try h.upload("markov_embed");
    defer markov_embed.free();
    var markov_head = try h.upload("markov_head");
    defer markov_head.free();
    const me_host = try fx.bytes("markov_embed");
    defer gpa.free(me_host);
    const conf_host = try fx.bytes("conf_w");
    defer gpa.free(conf_host);
    const dhead: dsp.Head = .{
        .main_proj = &main_proj, .main_norm = main_norm.ptr, .norm = norm.ptr, .head = &head,
        .markov_embed = markov_embed.ptr, .markov_head = markov_head.ptr,
        .markov_embed_host = @alignCast(std.mem.bytesAsSlice(u16, me_host)), .conf = @alignCast(std.mem.bytesAsSlice(u16, conf_host)),
    };
    const owned = try gpa.alloc(base.StageOwned, stages_n);
    defer gpa.free(owned);
    var loaded: usize = 0;
    defer for (owned[0..loaded]) |*s| s.deinit();
    const views = try gpa.alloc(dsp.Stage, stages_n);
    defer gpa.free(views);
    for (0..stages_n) |j| {
        owned[j] = try base.loadStage(&h, &kernels, j, dims.groups, dims.n_routed);
        loaded += 1;
        views[j] = owned[j].view();
        check.pass("dsv41 draft-pool stage {d} loaded ({d} experts)", .{ j, dims.n_routed + 1 });
    }
    var cos = try h.upload("cos");
    defer cos.free();
    var sin = try h.upload("sin");
    defer sin.free();
    const rope: dsp.Rope = .{ .cos = cos, .sin = sin, .rows = @intCast(try fx.int("rope_rows")) };
    const drafter = try dsp.Drafter.init(gpa, gpu.d, dims, &set, &glue, &ex_kernels, views, dhead, null);
    defer drafter.deinit();

    const n_cases: usize = @intCast(try fx.int("n_cases"));
    var nb: [64]u8 = undefined;
    for (0..n_cases) |ci| {
        const N: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "c{d}_N", .{ci})));
        var pool = try dsp.DraftPool.init(gpa, gpu.d, stages_n, slots_n, dims);
        defer pool.deinit(gpa);
        var tokens: [16]u32 = undefined;
        var q0: [16]usize = undefined;
        var slots: [16]usize = undefined;
        for (0..N) |s_| {
            const len: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "c{d}_len{d}", .{ ci, s_ })));
            tokens[s_] = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "c{d}_token{d}", .{ ci, s_ })));
            slots[s_] = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "c{d}_slot{d}", .{ ci, s_ })));
            q0[s_] = len;
            var taps = try h.upload(try std.fmt.bufPrint(&nb, "c{d}_taps{d}", .{ ci, s_ }));
            defer taps.free();
            try drafter.absorbSlot(&pool, slots[s_], taps.ptr, 0, len, rope, h.stream);
        }
        for (0..stages_n) |j| try h.stage(try std.fmt.bufPrint(&nb, "c{d}_ring{d}_absorbed", .{ ci, j }), pool.rings[j].ptr);
        var emb = try h.upload(try std.fmt.bufPrint(&nb, "c{d}_emb", .{ci}));
        defer emb.free();
        var pbuf: [16]u8 = undefined;
        var pref: Prefixed = .{ .h = &h, .prefix = try std.fmt.bufPrint(&pbuf, "c{d}", .{ci}) };
        var out: [16]dsp.Drafts = undefined;
        try drafter.batchDraftWithTable(&pool, tokens[0..N], q0[0..N], slots[0..N], emb.ptr, rope, h.stream, &pref, hook, out[0..N]);
        try h.stage(try std.fmt.bufPrint(&nb, "c{d}_xc", .{ci}), drafter.xc.ptr);
        try h.stage(try std.fmt.bufPrint(&nb, "c{d}_xn", .{ci}), drafter.xn.ptr);
        try h.stage(try std.fmt.bufPrint(&nb, "c{d}_local", .{ci}), drafter.local.ptr);
        // the drafts: out[s, 1..] of the Markov buffer against the fixture's [N, steps]
        const want = try fx.bytes(try std.fmt.bufPrint(&nb, "c{d}_drafts", .{ci}));
        defer gpa.free(want);
        const wd = std.mem.bytesAsSlice(i64, want);
        const steps: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "c{d}_steps", .{ci})));
        var bad: usize = 0;
        for (0..N) |s_| for (0..steps) |i| {
            if (out[s_].tokens[i] != wd[s_ * steps + i]) bad += 1;
        };
        if (bad == 0) check.pass("BITEXACT dsv41 draft-pool c{d} drafts: {d} streams x {d} equal to the Python fork's", .{ ci, N, steps }) else {
            h.failures += 1;
            std.debug.print("FAIL dsv41 draft-pool c{d} drafts: {d} differ\n", .{ ci, bad });
        }
        const want_conf = try fx.bytes(try std.fmt.bufPrint(&nb, "c{d}_conf", .{ci}));
        defer gpa.free(want_conf);
        const wc = std.mem.bytesAsSlice(f32, want_conf);
        var worst: f32 = 0;
        for (0..N) |s_| for (0..steps) |i| {
            worst = @max(worst, @abs(out[s_].conf[i] - wc[s_ * steps + i]) / @max(1.0, @abs(wc[s_ * steps + i])));
        };
        if (worst > 1e-4) {
            h.failures += 1;
            std.debug.print("FAIL dsv41 draft-pool c{d} conf: worst relative difference {e}\n", .{ ci, worst });
        } else check.pass("dsv41 draft-pool c{d} conf: within {e} of the fork's", .{ ci, worst });
    }
    try check.expect(h.failures == 0, "dsv41-draft-pool: {d} stage(s) differ from the Python fork", .{h.failures});
}
