//! DeepSeek-V4.1's DSpark drafter against the fork's fixture (tools/zig/dsv41/dump_dspark.py, packed by
//! capture_dspark.sh): absorb (main_x, every stage's ring), then the draft stage by stage (x after hc_pre, the
//! attention partial, h after hc_post, x after the FFN hc_pre, the MoE partial, h), the collapse, the final norm, the
//! head's logits and the Markov drafts, every stage's bytes the Python fork's; the confidence within a tolerance
//! (the fork's is a cuBLAS fp32 matmul, ours a host dot).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;

const dsp = dsv41.dspark;
const exl3 = dsv41.exl3;

pub const Harness = struct {
    gpu: Gpu,
    fx: fixture.Fixture,
    stream: cuda.Stream,
    failures: usize = 0,
    nb: [96]u8 = undefined,

    pub fn upload(self: *Harness, what: []const u8) !cuda.DeviceBuffer {
        const bytes = try self.fx.bytes(what);
        defer self.gpu.gpa.free(bytes);
        return cuda.DeviceBuffer.fromHost(self.gpu.d, bytes);
    }

    pub fn stage(self: *Harness, what: []const u8, got: cuda.abi.DevicePtr) !void {
        try self.stream.synchronize();
        const want = try self.fx.bytes(what);
        defer self.gpu.gpa.free(want);
        const bytes = try self.gpu.gpa.alloc(u8, want.len);
        defer self.gpu.gpa.free(bytes);
        const view: cuda.DeviceBuffer = .{ .d = self.gpu.d, .ptr = got, .len = want.len };
        try view.download(0, bytes);
        var label: [96]u8 = undefined;
        const text = try std.fmt.bufPrint(&label, "dsv41 dspark {s}", .{what});
        check.sameBytes(text, bytes, want) catch {
            self.failures += 1;
            return;
        };
        check.pass("BITEXACT {s}: {d} bytes equal to the Python fork's", .{ text, want.len });
    }

    pub fn name(self: *Harness, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&self.nb, fmt, args);
    }
};

pub fn codebook(s: []const u8) exl3.Codebook {
    return if (std.mem.eql(u8, s, "mul1")) .mul1 else if (std.mem.eql(u8, s, "mcg")) .mcg else .inst3;
}

/// An EXL3 group of the fixture by `name` (`<name>_trellis/_suh/_svh`, params `<name>_kt/_nt/_w16/_cb`).
pub fn linearOf(h: *Harness, kernels: *const exl3.Kernels, comptime fmt: []const u8, args: anytype) !exl3.Linear {
    const gpa = h.gpu.gpa;
    var nb: [96]u8 = undefined;
    const base = try std.fmt.bufPrint(&nb, fmt, args);
    var kb: [112]u8 = undefined;
    const tr = try h.fx.bytes(try std.fmt.bufPrint(&kb, "{s}_trellis", .{base}));
    defer gpa.free(tr);
    const suh = try h.fx.bytes(try std.fmt.bufPrint(&kb, "{s}_suh", .{base}));
    defer gpa.free(suh);
    const svh = try h.fx.bytes(try std.fmt.bufPrint(&kb, "{s}_svh", .{base}));
    defer gpa.free(svh);
    const kt: usize = @intCast(try h.fx.int(try std.fmt.bufPrint(&kb, "{s}_kt", .{base})));
    const nt: usize = @intCast(try h.fx.int(try std.fmt.bufPrint(&kb, "{s}_nt", .{base})));
    const w16: usize = @intCast(try h.fx.int(try std.fmt.bufPrint(&kb, "{s}_w16", .{base})));
    const cb = codebook(try h.fx.string(try std.fmt.bufPrint(&kb, "{s}_cb", .{base})));
    return exl3.Linear.init(gpa, kernels, tr, kt, nt, w16, suh, svh, cb);
}

pub const StageOwned = struct {
    plains: [13]cuda.DeviceBuffer,
    wq_a: exl3.Linear,
    wq_b: exl3.Linear,
    wkv: exl3.Linear,
    wo_a: [8]exl3.Linear,
    groups: usize,
    wo_b: exl3.Linear,
    experts: dsv41.experts.Experts,

    pub fn view(self: *const StageOwned) dsp.Stage {
        const p = &self.plains;
        return .{
            .hc_attn_fn = p[0].ptr, .hc_attn_scale = p[1].ptr, .hc_attn_base = p[2].ptr, .hc_ffn_fn = p[3].ptr,
            .hc_ffn_scale = p[4].ptr, .hc_ffn_base = p[5].ptr, .attn_norm = p[6].ptr, .ffn_norm = p[7].ptr,
            .q_norm = p[8].ptr, .kv_norm = p[9].ptr, .sink = p[10].ptr, .gate_w = p[11].ptr, .gate_b = p[12].ptr,
            .wq_a = &self.wq_a, .wq_b = &self.wq_b, .wkv = &self.wkv, .wo_a = self.wo_a[0..self.groups], .wo_b = &self.wo_b,
            .experts = &self.experts,
        };
    }

    pub fn deinit(self: *StageOwned) void {
        for (&self.plains) |*b| b.free();
        self.wq_a.deinit();
        self.wq_b.deinit();
        self.wkv.deinit();
        for (self.wo_a[0..self.groups]) |*w| w.deinit();
        self.wo_b.deinit();
        self.experts.deinit();
    }
};

const plain_names = [_][]const u8{ "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "attn_norm", "ffn_norm", "q_norm", "kv_norm", "sink", "gate_w", "gate_b" };

pub fn loadStage(h: *Harness, kernels: *const exl3.Kernels, j: usize, groups: usize, n_routed: usize) !StageOwned {
    const gpa = h.gpu.gpa;
    var s: StageOwned = undefined;
    s.groups = groups;
    var nb: [64]u8 = undefined;
    for (plain_names, 0..) |pn, i| s.plains[i] = try h.upload(try std.fmt.bufPrint(&nb, "s{d}_{s}", .{ j, pn }));
    s.wq_a = try linearOf(h, kernels, "s{d}_wq_a", .{j});
    s.wq_b = try linearOf(h, kernels, "s{d}_wq_b", .{j});
    s.wkv = try linearOf(h, kernels, "s{d}_wkv", .{j});
    for (0..groups) |g| s.wo_a[g] = try linearOf(h, kernels, "s{d}_wo_a{d}", .{ j, g });
    s.wo_b = try linearOf(h, kernels, "s{d}_wo_b", .{j});
    // the experts: routed then the shared one, this (whole) column range
    const n = n_routed + 1;
    const mats = try gpa.alloc(dsv41.experts.ExpertMats, n);
    defer gpa.free(mats);
    var owned = std.ArrayList([]u8).empty;
    defer {
        for (owned.items) |b| gpa.free(b);
        owned.deinit(gpa);
    }
    for (0..n) |e| {
        var m: dsv41.experts.ExpertMats = undefined;
        inline for (.{ "gate", "up", "down" }) |part| {
            var kb: [96]u8 = undefined;
            const base = try std.fmt.bufPrint(&kb, "s{d}_e{d}_{s}", .{ j, e, part });
            var kb2: [112]u8 = undefined;
            const tr = try h.fx.bytes(try std.fmt.bufPrint(&kb2, "{s}_trellis", .{base}));
            try owned.append(gpa, tr);
            const suh = try h.fx.bytes(try std.fmt.bufPrint(&kb2, "{s}_suh", .{base}));
            try owned.append(gpa, suh);
            const svh = try h.fx.bytes(try std.fmt.bufPrint(&kb2, "{s}_svh", .{base}));
            try owned.append(gpa, svh);
            @field(m, part) = .{
                .trellis = tr, .suh = suh, .svh = svh,
                .kt = @intCast(try h.fx.int(try std.fmt.bufPrint(&kb2, "{s}_kt", .{base}))),
                .nt = @intCast(try h.fx.int(try std.fmt.bufPrint(&kb2, "{s}_nt", .{base}))),
                .w16 = @intCast(try h.fx.int(try std.fmt.bufPrint(&kb2, "{s}_w16", .{base}))),
            };
        }
        mats[e] = m;
    }
    const cb = codebook(try h.fx.string("s0_e0_gate_cb"));
    s.experts = try dsv41.experts.Experts.init(gpa, h.gpu.d, mats, cb);
    return s;
}

/// `dsv41-dspark <dir> [aot-dir]`: the fixture's own captured set, or any set holding the drafter's variants (the
/// serving run's, tools/zig/dsv41/serve_capture.md).
pub fn dspark(gpu: Gpu, dir: []const u8, aot_override: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const aot_dir = if (aot_override) |a| try gpa.dupe(u8, a) else try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_path = try std.fs.path.join(gpa, &.{ dir, "out", "fixture" });
    defer gpa.free(fx_path);
    var h: Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(gpa, gpu.io, fx_path), .stream = try cuda.Stream.init(gpu.d, true) };
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
    var kernels = try exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var ex_kernels = try dsv41.experts.Kernels.load(gpu.d);
    defer ex_kernels.deinit();
    var glue = try dsv41.attention.Glue.load(gpu.d);
    defer glue.deinit();

    // the heads
    var main_proj = try linearOf(&h, &kernels, "main_proj", .{});
    defer main_proj.deinit();
    var head = try linearOf(&h, &kernels, "head", .{});
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
    // the stages
    const owned = try gpa.alloc(StageOwned, stages_n);
    defer gpa.free(owned);
    var loaded: usize = 0;
    defer for (owned[0..loaded]) |*s| s.deinit();
    const views = try gpa.alloc(dsp.Stage, stages_n);
    defer gpa.free(views);
    for (0..stages_n) |j| {
        owned[j] = try loadStage(&h, &kernels, j, dims.groups, dims.n_routed);
        loaded += 1;
        views[j] = owned[j].view();
        check.pass("dsv41 dspark stage {d} loaded ({d} experts)", .{ j, dims.n_routed + 1 });
    }
    var cos = try h.upload("cos");
    defer cos.free();
    var sin = try h.upload("sin");
    defer sin.free();
    const rope: dsp.Rope = .{ .cos = cos, .sin = sin, .rows = @intCast(try fx.int("rope_rows")) };
    var emb2 = try h.upload("emb2");
    defer emb2.free();

    const drafter = try dsp.Drafter.init(gpa, gpu.d, dims, &set, &glue, &ex_kernels, views, dhead, null);
    defer drafter.deinit();
    var cache = try drafter.newCache();
    defer cache.deinit(gpa);

    // -- absorb -----------------------------------------------------------------------------------------------
    const n_abs: usize = @intCast(try fx.int("n_abs"));
    var taps = try h.upload("taps");
    defer taps.free();
    try drafter.absorb(&cache, taps.ptr, 0, n_abs, rope, h.stream);
    try h.stage("main_x", drafter.main_x.ptr);
    for (0..stages_n) |j| try h.stage(try h.name("ring{d}_absorbed", .{j}), cache.rings[j].ptr);

    // -- draft, stage by stage (the drafter's own buffers read between launches) -------------------------------
    // the drafter runs whole; the fixture's intermediates are compared by re-running the stages one at a time
    const token: u32 = @intCast(try fx.int("token"));
    const q0: usize = @intCast(try fx.int("q0"));
    // emb2 stands in for the embedding table: rows 0 (token) and 1 (noise), so copies address it by those ids
    const drafts = try drafter.draftWithTable(&cache, token, q0, emb2.ptr, rope, h.stream, &h, stageHook);
    try h.stage("xc", drafter.xc.ptr);
    try h.stage("xn", drafter.xn.ptr);
    try h.stage("local", drafter.local.ptr);
    try h.stage("drafts", drafter.markov.out.ptr + 8);
    const want_conf = try fx.bytes("conf");
    defer gpa.free(want_conf);
    const wc = std.mem.bytesAsSlice(f32, want_conf);
    var worst: f32 = 0;
    for (0..drafts.len) |i| worst = @max(worst, @abs(drafts.conf[i] - wc[i]) / @max(1.0, @abs(wc[i])));
    if (worst > 1e-4) {
        h.failures += 1;
        std.debug.print("FAIL dsv41 dspark conf: worst relative difference {e}\n", .{worst});
    } else check.pass("dsv41 dspark conf: within {e} of the fork's cuBLAS dot ({d} drafts)", .{ worst, drafts.len });
    try check.expect(h.failures == 0, "dsv41-dspark: {d} stage(s) differ from the Python fork", .{h.failures});
}

/// The drafter calls this after each named step; the test compares the buffer with the fixture's array of that name.
fn stageHook(ctx: *anyopaque, what: []const u8, buf: cuda.abi.DevicePtr) anyerror!void {
    const h: *Harness = @ptrCast(@alignCast(ctx));
    try h.stage(what, buf);
}
