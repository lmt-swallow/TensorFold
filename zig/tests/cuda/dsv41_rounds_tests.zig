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

// -- the backbone round ----------------------------------------------------------------------------------------

const F = dsv41.forward;
const Rounds = dsv41.rounds;

/// The other rank's partials, in gather order, from the fixture; this rank's own copied in beside them (the forward
/// test's Supply), and every probed buffer kept for the comparison after the run.
const Supply = struct {
    gpu: Gpu,
    rank: usize = 0,
    rank1: std.ArrayList(cuda.DeviceBuffer),
    next: usize = 0,
    names: std.ArrayList([]const u8),
    probes: std.ArrayList(cuda.DeviceBuffer),

    fn gather(ctx: *anyopaque, partial: cuda.abi.DevicePtr, words: usize, out: cuda.abi.DevicePtr, stream: cuda.Stream) anyerror!void {
        const self: *Supply = @ptrCast(@alignCast(ctx));
        if (self.next >= self.rank1.items.len) return error.GatherOverrun;
        const other = self.rank1.items[self.next];
        self.next += 1;
        if (other.len != words * 4) return error.GatherShape;
        const dst: cuda.DeviceBuffer = .{ .d = self.gpu.d, .ptr = out, .len = 2 * words * 4 };
        try dst.copyFrom(0, partial, words * 4, stream.handle);
        try dst.copyFrom(words * 4, other.ptr, words * 4, stream.handle);
    }

    fn probe(ctx: *anyopaque, name: []const u8, ptr: cuda.abi.DevicePtr, bytes: usize, stream: cuda.Stream) anyerror!void {
        const self: *Supply = @ptrCast(@alignCast(ctx));
        var buf = try cuda.DeviceBuffer.alloc(self.gpu.d, bytes);
        try buf.copyFrom(0, ptr, bytes, stream.handle);
        // the round names its own partials `.r0`; the fixture's are by rank
        const key = try self.gpu.gpa.dupe(u8, name);
        if (self.rank == 1 and std.mem.endsWith(u8, key, ".r0")) key[key.len - 1] = '1';
        try self.names.append(self.gpu.gpa, key);
        try self.probes.append(self.gpu.gpa, buf);
    }

    fn deinit(self: *Supply) void {
        for (self.rank1.items) |*b| b.free();
        self.rank1.deinit(self.gpu.gpa);
        for (self.names.items) |n| self.gpu.gpa.free(n);
        self.names.deinit(self.gpu.gpa);
        for (self.probes.items) |*b| b.free();
        self.probes.deinit(self.gpu.gpa);
    }
};

/// `dsv41-rounds <dir> <rank-cache-dir> <model-dir> [aot-dir]`: one round over two streams' windows in a pool,
/// rank 0's weights from the Python rank cache, the other rank's partials from the fixture.
pub fn rounds(gpu: Gpu, dir: []const u8, cache_dir: []const u8, model_dir: []const u8, aot_override: ?[]const u8, rank_arg: ?[]const u8) !void {
    const rank: usize = if (rank_arg) |r| try std.fmt.parseInt(usize, r, 10) else 0;
    const other: usize = 1 - rank;
    const gpa = gpu.gpa;
    const io = gpu.io;
    const aot_dir = if (aot_override) |a| try gpa.dupe(u8, a) else try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_path = try std.fs.path.join(gpa, &.{ dir, "out", "fixture" });
    defer gpa.free(fx_path);
    var h: base.Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(gpa, io, fx_path), .stream = try cuda.Stream.init(gpu.d, true) };
    defer h.fx.deinit();
    defer h.stream.deinit();
    const fx = h.fx;
    const layers: usize = @intCast(try fx.int("layers"));
    const world: u32 = @intCast(try fx.int("world"));
    try check.expect(world == 2, "the fixture's world is {d}; this test supplies one other rank", .{world});
    const slots: usize = @intCast(try fx.int("slots"));
    const cap: usize = @intCast(try fx.int("cap"));
    const bucket: usize = @intCast(try fx.int("bucket"));
    var kernels = try exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var ek = try dsv41.experts.Kernels.load(gpu.d);
    defer ek.deinit();
    var glue = try dsv41.attention.Glue.load(gpu.d);
    defer glue.deinit();
    var cfg = try dsv41.config.Config.read(gpa, io, model_dir);
    const rc_path = (try dsv41.rank_cache.RankCache.locate(gpa, io, cache_dir, rank, world)) orelse return error.NoRankCache;
    defer gpa.free(rc_path);
    var rc = try dsv41.rank_cache.RankCache.open(gpa, rc_path);
    defer rc.close();
    var w = try dsv41.weights.load(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1, layers);
    defer w.deinit();
    check.pass("dsv41 rounds: {d} layers of rank {d} loaded from {s}", .{ layers, rank, rc_path });
    var rope_c: dsv41.attention.Rope = .{ .cos = try h.upload("cos_c"), .sin = try h.upload("sin_c"), .rows = @intCast(try fx.int("rope_rows")) };
    defer rope_c.cos.free();
    defer rope_c.sin.free();
    var rope_w: dsv41.attention.Rope = .{ .cos = try h.upload("cos_w"), .sin = try h.upload("sin_w"), .rows = @intCast(try fx.int("rope_rows")) };
    defer rope_w.cos.free();
    defer rope_w.sin.free();
    var pool = try Rounds.Pool.init(gpa, gpu.d, &cfg, layers, slots, cap);
    defer pool.deinit();
    var nb: [96]u8 = undefined;
    const plane_names = [_][]const u8{ "comp_codes", "comp_scales", "idxk_codes", "idxk_scales", "raw_kv", "raw_score" };
    for (0..layers) |li| {
        const ring = try fx.bytes(try std.fmt.bufPrint(&nb, "before_L{d}.ring", .{li}));
        defer gpa.free(ring);
        try pool.rings[li].upload(0, ring);
        if (pool.planes[li]) |*p| {
            inline for (plane_names) |pn| {
                const bytes = try fx.bytes(try std.fmt.bufPrint(&nb, "before_L{d}.{s}", .{ li, pn }));
                defer gpa.free(bytes);
                if (bytes.len > 0) try @field(p, pn).upload(0, bytes);
            }
        }
    }
    var runner = try Rounds.RoundRunner.init(gpa, gpu.d, &cfg, &w, &set, &glue, &ek, world, bucket, rope_c, rope_w, &pool);
    defer runner.deinit();
    var supply: Supply = .{ .gpu = gpu, .rank = rank, .rank1 = .empty, .names = .empty, .probes = .empty };
    defer supply.deinit();
    runner.probe = .{ .ctx = &supply, .call = Supply.probe };
    for (0..layers) |li| {
        try supply.rank1.append(gpa, try h.upload(try std.fmt.bufPrint(&nb, "L{d}.p_attn.r{d}", .{ li, other })));
        try supply.rank1.append(gpa, try h.upload(try std.fmt.bufPrint(&nb, "L{d}.p_moe.r{d}", .{ li, other })));
    }
    try supply.rank1.append(gpa, try h.upload(try std.fmt.bufPrint(&nb, "head.local.r{d}", .{other})));
    // the windows
    const ids_bytes = try fx.bytes("ids");
    defer gpa.free(ids_bytes);
    const ids = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(ids_bytes)));
    const n_windows: usize = @intCast(try fx.int("n_windows"));
    var windows: [8]Rounds.Window = undefined;
    var at: usize = 0;
    for (0..n_windows) |i| {
        const rows: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_rows", .{i})));
        windows[i] = .{
            .slot = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_slot", .{i}))),
            .base = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_base", .{i}))),
            .size = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_size", .{i}))),
            .position = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_pos", .{i}))),
            .ids = ids[at .. at + rows],
        };
        at += rows;
    }
    const gather: F.Gather = .{ .ctx = &supply, .call = Supply.gather };
    const logits = try runner.forward(windows[0..n_windows], gather, h.stream);
    try h.stream.synchronize();
    for (supply.names.items, supply.probes.items) |name, buf| try h.stage(name, buf.ptr);
    try h.stage("logits", logits);
    for (0..layers) |li| {
        try h.stage(try std.fmt.bufPrint(&nb, "after_L{d}.ring", .{li}), pool.rings[li].ptr);
        if (pool.planes[li]) |*p| {
            const ratio = cfg.ratio(li);
            // every extent's scratch row is written by every row of its window whose group is not complete, in no
            // fixed order (the fork's own bytes there depend on the launch's scheduling): left out, as the forward test
            var skip: [8]usize = undefined;
            for (windows[0..n_windows], 0..) |wd, i| skip[i] = wd.base / ratio + wd.size / ratio - 1;
            try stageExcept(&h, try std.fmt.bufPrint(&nb, "after_L{d}.comp_codes", .{li}), p.comp_codes.ptr, cfg.head_dim / 2, skip[0..n_windows]);
            try stageExcept(&h, try std.fmt.bufPrint(&nb, "after_L{d}.comp_scales", .{li}), p.comp_scales.ptr, cfg.head_dim / 16, skip[0..n_windows]);
            if (p.idxk_codes.len > 0) {
                try stageExcept(&h, try std.fmt.bufPrint(&nb, "after_L{d}.idxk_codes", .{li}), p.idxk_codes.ptr, cfg.idx_dim / 2, skip[0..n_windows]);
                try stageExcept(&h, try std.fmt.bufPrint(&nb, "after_L{d}.idxk_scales", .{li}), p.idxk_scales.ptr, cfg.idx_dim / 32, skip[0..n_windows]);
            }
            if (p.raw_kv.len > 0) {
                try h.stage(try std.fmt.bufPrint(&nb, "after_L{d}.raw_kv", .{li}), p.raw_kv.ptr);
                try h.stage(try std.fmt.bufPrint(&nb, "after_L{d}.raw_score", .{li}), p.raw_score.ptr);
            }
        }
    }
    try check.expect(h.failures == 0, "dsv41-rounds: {d} stage(s) differ from the Python fork", .{h.failures});
}

/// `stage` with the rows `skip` (of `row_bytes` each) zeroed on both sides before the comparison.
fn stageExcept(h: *base.Harness, what: []const u8, got: cuda.abi.DevicePtr, row_bytes: usize, skip: []const usize) !void {
    try h.stream.synchronize();
    const gpa = h.gpu.gpa;
    const want = try h.fx.bytes(what);
    defer gpa.free(want);
    const bytes = try gpa.alloc(u8, want.len);
    defer gpa.free(bytes);
    const view: cuda.DeviceBuffer = .{ .d = h.gpu.d, .ptr = got, .len = want.len };
    try view.download(0, bytes);
    for (skip) |row| if ((row + 1) * row_bytes <= want.len) {
        @memset(want[row * row_bytes .. (row + 1) * row_bytes], 0);
        @memset(bytes[row * row_bytes .. (row + 1) * row_bytes], 0);
    };
    var label: [96]u8 = undefined;
    const text = try std.fmt.bufPrint(&label, "dsv41 rounds {s} (scratch rows left out)", .{what});
    check.sameBytes(text, bytes, want) catch {
        h.failures += 1;
        return;
    };
    check.pass("BITEXACT {s}: {d} bytes equal to the Python fork's", .{ text, want.len });
}
