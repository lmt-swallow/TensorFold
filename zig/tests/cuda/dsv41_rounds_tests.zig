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
        // rank order: this rank's partial sits at its own index of the gathered [world, words]
        const dst: cuda.DeviceBuffer = .{ .d = self.gpu.d, .ptr = out, .len = 2 * words * 4 };
        const mine: usize = self.rank * words * 4;
        const theirs: usize = (1 - self.rank) * words * 4;
        try dst.copyFrom(mine, partial, words * 4, stream.handle);
        try dst.copyFrom(theirs, other.ptr, words * 4, stream.handle);
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

/// `dsv41-rounds <dir> <rank-cache-dir> <model-dir> [aot-dir] [rank]`: one round over two streams' windows in a pool,
/// this rank's weights from the Python rank cache, the other rank's partials from the fixture. A fixture dumped with
/// `--engram` runs Engram on its layers (TF_DS_ENGRAM, TF_DS_TOKEN_MAP as dsv41-forward), each window's host ids its
/// stream's prompt and then its rows. TF_DS_ROUNDS_CAP sizes the pool for that many positions instead of the
/// fixture's cap (the windows stay where the fixture put them), with the RoPE tables of `ropeRows(cap)` rows from
/// TF_DS_ROPE_DIR (`rope_{c,w}_<rows>.bin`, dump_rope.py) in place of the fixture's; their first rows are compared
/// with the fixture's tables, and the runner may scan up to the cap's bucket: the round must pick the fixture's
/// bucket from its windows (cuda_rounds.zig `bucketOf`). TF_DS_ROUNDS_TIME=<reps> then times rounds of 1 and 16
/// noise rows on slot 0 (no Engram, the gathers local) at position 0 and at the cap's bucket, each also with every
/// round scanning the cap's bucket whole.
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
    const cap: usize = if (getenv("TF_DS_ROUNDS_CAP")) |v| try std.fmt.parseInt(usize, v, 10) else @intCast(try fx.int("cap"));
    const fx_bucket: usize = @intCast(try fx.int("bucket"));
    const bucket: usize = if (getenv("TF_DS_ROUNDS_CAP") != null) @max(Rounds.bucket_min, try std.math.ceilPowerOfTwo(usize, cap)) else fx_bucket;
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
    var rope_c: dsv41.attention.Rope = undefined;
    var rope_w: dsv41.attention.Rope = undefined;
    if (getenv("TF_DS_ROUNDS_CAP") != null) {
        const dir_r = getenv("TF_DS_ROPE_DIR") orelse return error.NoRopeDir;
        const rows = dsv41.serve_lanes.ropeRows(cap);
        rope_c = try loadRope(gpu, dir_r, "rope_c", rows);
        rope_w = try loadRope(gpu, dir_r, "rope_w", rows);
        // the fixture's tables are the first rows of these (no row's bits depend on the table's length)
        try h.stage("cos_c", rope_c.cos.ptr);
        try h.stage("sin_c", rope_c.sin.ptr);
        try h.stage("cos_w", rope_w.cos.ptr);
        try h.stage("sin_w", rope_w.sin.ptr);
    } else {
        rope_c = .{ .cos = try h.upload("cos_c"), .sin = try h.upload("sin_c"), .rows = @intCast(try fx.int("rope_rows")) };
        rope_w = .{ .cos = try h.upload("cos_w"), .sin = try h.upload("sin_w"), .rows = @intCast(try fx.int("rope_rows")) };
    }
    defer rope_c.cos.free();
    defer rope_c.sin.free();
    defer rope_w.cos.free();
    defer rope_w.sin.free();
    const before_alloc = cuda.usage(false);
    var pool = try Rounds.Pool.init(gpa, gpu.d, &cfg, layers, slots, cap);
    defer pool.deinit();
    {
        const got = cuda.usage(false).device - before_alloc.device;
        const want = Rounds.Pool.bytesFor(&cfg, layers, cap, slots);
        try check.expect(got == want, "dsv41 rounds: the pool holds {d} bytes, bytesFor says {d}", .{ got, want });
        const full = Rounds.Pool.bytesFor(&cfg, cfg.n_layers, cap, slots);
        check.pass("dsv41 rounds: pool of {d} positions x {d} slots over {d} layers: {d} bytes (= bytesFor); all {d} layers: {d} bytes ({d:.1} MiB)", .{ cap, slots, layers, got, cfg.n_layers, full, @as(f64, @floatFromInt(full)) / (1 << 20) });
    }
    var nb: [96]u8 = undefined;
    const plane_names = [_][]const u8{ "comp_codes", "comp_scales", "idxk_codes", "idxk_scales", "raw_kv", "raw_score" };
    for (0..layers) |li| {
        const ring = try fx.bytes(try std.fmt.bufPrint(&nb, "before_L{d}.ring", .{li}));
        defer gpa.free(ring);
        try pool.rings[li].upload(0, ring);
        if (pool.planes[li]) |*p| {
            inline for (plane_names) |pn| if (@field(p, pn).len > 0) {
                // (a ratio-1 kv source keeps no compressor inputs: the fork has no such plane)
                const bytes = try fx.bytes(try std.fmt.bufPrint(&nb, "before_L{d}.{s}", .{ li, pn }));
                defer gpa.free(bytes);
                if (bytes.len > 0) try @field(p, pn).upload(0, bytes);
            };
        }
    }
    var runner = try Rounds.RoundRunner.init(gpa, gpu.d, &cfg, &w, &set, &glue, &ek, world, bucket, rope_c, rope_w, &pool);
    defer runner.deinit();
    var supply: Supply = .{ .gpu = gpu, .rank = rank, .rank1 = .empty, .names = .empty, .probes = .empty };
    defer supply.deinit();
    runner.probe = .{ .ctx = &supply, .call = Supply.probe };
    const eng = try @import("dsv41_forward_tests.zig").engramFor(gpu, fx, &cfg, &w, &set, aot_dir, Rounds.max_rows);
    defer if (eng) |e| e.deinit();
    if (eng) |e| {
        runner.engram = e.roundHook();
        e.probe = runner.probe;
    }
    for (0..layers) |li| {
        if (eng != null and w.layers[li].engram_wkv != null)
            try supply.rank1.append(gpa, try h.upload(try std.fmt.bufPrint(&nb, "L{d}.p_engram.r{d}", .{ li, other })));
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
    var hosts: [8]std.ArrayList(i32) = @splat(.empty);
    defer for (&hosts) |*hl| hl.deinit(gpa);
    var at: usize = 0;
    for (0..n_windows) |i| {
        const rows: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_rows", .{i})));
        // the stream's host ids: its prompt, then the window's rows
        if (eng != null) {
            const pb = try fx.bytes(try std.fmt.bufPrint(&nb, "prompt{d}", .{i}));
            defer gpa.free(pb);
            for (std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(pb)))) |t| try hosts[i].append(gpa, @intCast(t));
            for (ids[at .. at + rows]) |t| try hosts[i].append(gpa, @intCast(t));
        }
        windows[i] = .{
            .host = hosts[i].items,
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
    try check.expect(runner.round_bucket == fx_bucket, "dsv41 rounds: the round scanned a bucket of {d} positions, the fork's {d}", .{ runner.round_bucket, fx_bucket });
    check.pass("dsv41 rounds: the round scans the fork's bucket of {d} positions (the runner's widest {d})", .{ runner.round_bucket, runner.bucket });
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
    if (getenv("TF_DS_ROUNDS_TIME")) |v| try timeRounds(gpu, &runner, cap, try std.fmt.parseInt(usize, v, 10), h.stream);
}

/// A gather of one rank's partial into both ranks' places (timing only: the bits are not compared).
const Echo = struct {
    d: *const cuda.Driver,

    fn gather(ctx: *anyopaque, partial: cuda.abi.DevicePtr, words: usize, out: cuda.abi.DevicePtr, stream: cuda.Stream) anyerror!void {
        const self: *Echo = @ptrCast(@alignCast(ctx));
        const dst: cuda.DeviceBuffer = .{ .d = self.d, .ptr = out, .len = 2 * words * 4 };
        try dst.copyFrom(0, partial, words * 4, stream.handle);
        try dst.copyFrom(words * 4, partial, words * 4, stream.handle);
    }
};

/// Rounds of 1 and 16 noise rows on slot 0 over the whole pool, at position 0 and at the widest bucket's end, each
/// with the least bucket graph.py's and the widest (every round scanning the pool whole): the best of `reps`
/// after a warm-up, in ms.
fn timeRounds(gpu: Gpu, runner: *Rounds.RoundRunner, cap: usize, reps: usize, stream: cuda.Stream) !void {
    runner.probe = null;
    runner.engram = null;
    var echo: Echo = .{ .d = gpu.d };
    const gather: F.Gather = .{ .ctx = &echo, .call = Echo.gather };
    const ids: [Rounds.max_rows]i64 = @splat(1);
    const rope_rows = @min(runner.rope_c.rows, runner.rope_w.rows);
    const least = runner.least;
    defer runner.least = least;
    for ([_]usize{ least, runner.bucket }) |lb| {
        runner.least = lb;
        for ([_]usize{ 1, Rounds.max_rows }) |rows| {
            const deep = @min(@min(cap, rope_rows), runner.bucket) - rows;
            for ([_]usize{ 0, deep }) |p| {
                const w = [1]Rounds.Window{.{ .slot = 0, .base = 0, .size = cap, .position = p, .ids = ids[0..rows] }};
                var best: f64 = std.math.inf(f64);
                for (0..reps + 1) |rep| {
                    try stream.synchronize();
                    var t0: std.c.timespec = undefined;
                    _ = std.c.clock_gettime(.MONOTONIC, &t0);
                    _ = try runner.forward(&w, gather, stream);
                    try stream.synchronize();
                    var t1: std.c.timespec = undefined;
                    _ = std.c.clock_gettime(.MONOTONIC, &t1);
                    const ms = @as(f64, @floatFromInt((t1.sec - t0.sec) * 1_000_000_000 + (t1.nsec - t0.nsec))) / 1e6;
                    if (rep > 0) best = @min(best, ms);
                }
                std.debug.print("dsv41-rounds time: cap {d}, least bucket {d}: {d} row(s) at position {d} scan bucket {d}: {d:.3} ms\n", .{ cap, lb, rows, p, runner.round_bucket, best });
            }
        }
    }
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

/// `<dir>/<name>_<rows>.bin`: cos then sin, f32 [rows, rope_dim / 2] each (tools/zig/dsv41/dump_rope.py).
fn loadRope(gpu: Gpu, dir: []const u8, name: []const u8, rows: usize) !dsv41.attention.Rope {
    const half = 32;
    const path = try std.fmt.allocPrint(gpu.gpa, "{s}/{s}_{d}.bin", .{ dir, name, rows });
    defer gpu.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 31));
    defer gpu.gpa.free(bytes);
    if (bytes.len != 2 * rows * half * 4) return error.RopeRowsShort;
    var cos = try cuda.DeviceBuffer.fromHost(gpu.d, bytes[0 .. rows * half * 4]);
    errdefer cos.free();
    const sin = try cuda.DeviceBuffer.fromHost(gpu.d, bytes[rows * half * 4 ..]);
    return .{ .cos = cos, .sin = sin, .rows = rows };
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
