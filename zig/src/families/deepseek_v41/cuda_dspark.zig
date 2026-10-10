//! DSpark drafting (DeepSeek-V4.1's in-checkpoint draft blocks, `mtp.*`) as the Python family's `dspark.py` runs it
//! for one stream (`Drafter.absorb`, `DraftGraph._body`): the drafter only proposes, the target verifies.
//!
//! absorb: the target's taps [n, 3 d] (main_x of layers 37-39's streams) -> main_proj -> main_norm -> every stage's
//! wkv -> kv_norm_rope into that stage's ring [ring, 512] at position % ring (ring = window + 16).
//! draft: the block [token, noise x 4] -> embed -> for each stage: hc_pre -> window attention (the ring's newest
//! 128 absorbed positions and every block row, through `comp` = the block's own keys) -> gather -> hc_post -> hc_pre
//! -> MoE (dspark_routed experts, top-3, the shared expert) -> gather -> hc_post; then collapse -> rmsnorm(norm) ->
//! head (this rank's vocabulary columns) -> the Markov loop (markov.py: `_bias`, `_score`, `_finish`, `_pick`) ->
//! the drafts, and the confidence head on [xc, markov_embed[drafts]] (the depth policies' input, not a token's).
//!
//! Every kernel is the fork's Triton variant replayed from the captured AOT set, or an EXL3 kernel already ported
//! (cuda_exl3, cuda_experts); the host loop keeps the fork's order of launches.
//!
//! A batched draft over the DraftPool replays a CUDA graph per stream count (dspark.py `BatchDraftGraph`,
//! TF_DS_GRAPHS=0 keeps it eager): its row tables, embedding rows and the Markov loop's first tokens go to fixed
//! device buffers before the launch, so the pass's launches (the stages, the head, the Markov steps and their
//! gathers) depend on the host only through the stream count, the pool's rings and the RoPE tables. Its stages take
//! the fork's fused kernels (dspark.py `_stages_fused`, TF_DS_DRAFT_FUSED; here TF_DS_FUSED, default on): each
//! sublayer's hc_post fused into the next sublayer's mixes (`_hc_mix_part` posted, `_hc_finish_s`), wq_a and wkv as
//! one grouped EXL3 launch, q's RMSNorm writing wq_b's rotated rows (`_rmsnorm_rot`) and q's RoPE in wq_b's epilogue,
//! wo_a's epilogue writing wo_b's rotated rows, and the paced L2 prefetch forks; the same bits. Absorbs run every
//! stage's wkv as one grouped launch (dspark.py `stage_kv`).
const std = @import("std");
const cuda = @import("cuda");
const exl3 = @import("cuda_exl3.zig");
const experts_mod = @import("cuda_experts.zig");
const hc_mod = @import("cuda_hc.zig");
const route_mod = @import("cuda_route.zig");
const attention = @import("cuda_attention.zig");
const weights = @import("cuda_weights.zig");
const rounds = @import("cuda_rounds.zig");
const Config = @import("config.zig").Config;

const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;

pub const Error = error{ BadShape, TooManyRows, OutOfMemory, NoComm } || cuda.Error || exl3.Error || experts_mod.Error;

pub const Dims = struct {
    d: usize, // 5120
    q_rank: usize, // 1280
    o_rank: usize, // 1024
    heads: usize, // this rank's heads (64 / world)
    hd: usize = 512,
    rd: usize = 64,
    groups: usize, // this rank's wo_a slices (8 / world)
    window: usize = 128,
    ring: usize = 144, // window + 16
    block: usize = 5, // dspark_block_size
    noise: u32, // dspark_noise_token_id
    vocab: usize, // 129280
    vocab_local: usize, // vocab / world: the head's columns this rank holds
    rank: usize,
    world: usize,
    markov_rank: usize = 256,
    n_routed: usize = 128, // dspark_n_routed_experts
    topk: usize = 3,
    eps: f32,
    hc_eps: f32,
    hc_iters: u32,
    swiglu_limit: f32,
    route_scale: f32,
    /// markov.py SPLIT (TP > 1): each rank scores its vocabulary half, the bests go through one gather a step.
    markov_split: bool = true,
};

/// One draft block's device weights, as the drafter reads them (a view of cuda_weights.Layer or a fixture's).
pub const Stage = struct {
    hc_attn_fn: Ptr,
    hc_attn_scale: Ptr,
    hc_attn_base: Ptr,
    hc_ffn_fn: Ptr,
    hc_ffn_scale: Ptr,
    hc_ffn_base: Ptr,
    attn_norm: Ptr,
    ffn_norm: Ptr,
    q_norm: Ptr,
    kv_norm: Ptr,
    sink: Ptr,
    gate_w: Ptr,
    gate_b: Ptr,
    wq_a: *const exl3.Linear,
    wq_b: *const exl3.Linear,
    wkv: *const exl3.Linear,
    wo_a: []const exl3.Linear,
    wo_b: *const exl3.Linear,
    experts: *const experts_mod.Experts,

    pub fn of(l: *const weights.Layer) Stage {
        return .{
            .hc_attn_fn = l.hc_attn_fn.ptr(), .hc_attn_scale = l.hc_attn_scale.ptr(), .hc_attn_base = l.hc_attn_base.ptr(),
            .hc_ffn_fn = l.hc_ffn_fn.ptr(),   .hc_ffn_scale = l.hc_ffn_scale.ptr(),   .hc_ffn_base = l.hc_ffn_base.ptr(),
            .attn_norm = l.attn_norm.ptr(),   .ffn_norm = l.ffn_norm.ptr(),           .q_norm = l.q_norm.ptr(),
            .kv_norm = l.kv_norm.ptr(),       .sink = l.sink.ptr(),                   .gate_w = l.gate_w.ptr(),
            .gate_b = l.gate_b.ptr(),         .wq_a = &l.wq_a,                        .wq_b = &l.wq_b,
            .wkv = &l.wkv,                    .wo_a = l.wo_a[0..l.wo_a_count],        .wo_b = &l.wo_b,
            .experts = &l.experts,
        };
    }
};

/// The drafter's own weights beyond the blocks (weights.py DSparkWeights).
pub const Head = struct {
    main_proj: *const exl3.Linear, // [3 d -> d]
    main_norm: Ptr,
    norm: Ptr,
    head: *const exl3.Linear, // the target's head, this rank's vocabulary columns
    markov_embed: Ptr, // bf16 [V, rank]
    markov_head: Ptr, // f16 [V, rank]
    /// markov_embed on the host (bf16 bits), for the confidence head's rows.
    markov_embed_host: []const u16,
    conf: []const u16, // f16 [d + rank]: the confidence head's weight, on the host
};

/// A stream's drafter state: one ring a stage, and how many positions it absorbed.
pub const Cache = struct {
    rings: []cuda.DeviceBuffer,
    absorbed: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, stages: usize, dims: Dims) !Cache {
        const rings = try gpa.alloc(cuda.DeviceBuffer, stages);
        errdefer gpa.free(rings);
        var made: usize = 0;
        errdefer for (rings[0..made]) |*r| r.free();
        for (rings) |*r| {
            r.* = try cuda.DeviceBuffer.alloc(d, dims.ring * dims.hd * 2);
            try r.fill8(0, null);
            made += 1;
        }
        return .{ .rings = rings };
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        for (self.rings) |*r| r.free();
        gpa.free(self.rings);
    }
};

pub const Rope = attention.Rope;

/// model.py `mm` past 128 rows (exl3_prefill.matmul, cuda_prompt.zig's `mm`): the rows rotated by suh to fp16, the
/// weight unpacked to fp16 [K, N], then the Hadamard-folded `_gemm` (the fork's prompt GEMM, replayed from the AOT
/// set) into y [m, N].
pub const Wide = struct {
    kernels: *const exl3.Kernels,
    set: *const aot.Set,
    rows: usize,
    xh: cuda.DeviceBuffer, // fp16 [rows, K]
    wq: cuda.DeviceBuffer, // fp16 [K, N]
    had: cuda.DeviceBuffer, // the 128 x 128 Hadamard, bf16

    const scale: f32 = 0.08838834764831845; // prefill.py HAD_SCALE

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, kernels: *const exl3.Kernels, set: *const aot.Set, rows: usize, kmax: usize, kn: usize) !*Wide {
        const self = try gpa.create(Wide);
        errdefer gpa.destroy(self);
        self.kernels = kernels;
        self.set = set;
        self.rows = rows;
        // the Hadamard: 1 - 2 * parity(i & j), bf16 (exact)
        var hh: [128 * 128]u16 = undefined;
        for (0..128) |i| for (0..128) |j| {
            const p: u32 = @popCount(@as(u32, @intCast(i & j))) & 1;
            hh[i * 128 + j] = if (p == 0) 0x3F80 else 0xBF80;
        };
        self.had = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&hh));
        errdefer self.had.free();
        self.xh = try cuda.DeviceBuffer.alloc(d, rows * kmax * 2);
        errdefer self.xh.free();
        self.wq = try cuda.DeviceBuffer.alloc(d, kn * 2);
        return self;
    }

    pub fn deinit(self: *Wide, gpa: std.mem.Allocator) void {
        self.wq.free();
        self.xh.free();
        self.had.free();
        gpa.destroy(self);
    }

    pub fn run(self: *const Wide, lin: *const exl3.Linear, x: Ptr, m: usize, y: Ptr, y_dtype: exl3.DType, stream: cuda.Stream) !void {
        const K = lin.k;
        const N = lin.n;
        if (m > self.rows or m * K * 2 > self.xh.len or K * N * 2 > self.wq.len) return error.TooManyRows;
        {
            var a: cuda.Args = .{};
            a.add(x);
            a.add(@intFromEnum(exl3.DType.bf16));
            a.add(lin.suh.ptr);
            a.add(self.xh.ptr);
            a.add(@as(c_int, @intCast(K)));
            try cuda.launch.launch(lin.rot_in, .{ .grid = .{ .x = @intCast((K / 128 + 3) / 4), .y = @intCast(m) }, .block = .{ .x = 128 } }, stream, &a);
        }
        {
            const s = lin.strides();
            const f = try self.kernels.unpack(lin.k2, lin.cb);
            var a: cuda.Args = .{};
            a.add(lin.words.ptr);
            a.add(self.wq.ptr);
            a.add(@as(c_int, @intCast(N)));
            a.add(s.k);
            a.add(s.nb);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(N / 16), .y = @intCast(K / 16) }, .block = .{ .x = 32 } }, stream, &a);
        }
        const bm: usize = 128;
        const grid: u32 = @intCast(((m + bm - 1) / bm) * (N / 128));
        try self.set.run(stream, "_gemm", .{ grid, 1, 1 }, &.{
            aot.ptr("X", "*fp16", self.xh.ptr),   aot.ptr("W", "*fp16", self.wq.ptr),    aot.ptr("H", "*bf16", self.had.ptr),
            aot.ptr("SVH", "*fp16", lin.svh.ptr), aot.ptr("BIAS", "*fp16", lin.svh.ptr), aot.ptr("OUT", if (y_dtype == .f32) "*fp32" else "*bf16", y),
            aot.int("M", @intCast(m)),            aot.int("o_stride", @intCast(N)),
        }, &.{ aot.ci("K", @intCast(K)), aot.ci("N", @intCast(N)), aot.ci("BM", 128), aot.ci("BK", 32), aot.ci("GROUP", 8), aot.ci("HAS_BIAS", 0), aot.cf("SCALE", scale) });
    }
};

/// What a draft returns: the block's drafts and their confidences (the head's logits, one a draft).
pub const Drafts = struct { tokens: [16]u32, conf: [16]f32, len: usize };

/// gathered [world, words] f32 <- every rank's `partial` [words] in rank order, on `stream` (cuda_forward.Gather's
/// shape): a test's stand-in for the communicator.
pub const GatherHook = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void,
};

/// The drafter's dimensions on `rank` of `world` from the checkpoint's config (dspark.py Drafter over model.py's
/// split: this rank's heads, wo_a groups and vocabulary columns).
pub fn dimsOf(cfg: *const Config, rank: usize, world: usize) Dims {
    return .{
        .d = cfg.dim,
        .q_rank = cfg.q_rank,
        .o_rank = cfg.o_rank,
        .heads = cfg.n_heads / world,
        .hd = cfg.head_dim,
        .rd = cfg.rope_dim,
        .groups = cfg.o_groups / world,
        .window = cfg.window,
        .ring = cfg.window + 16,
        .block = cfg.dspark_block,
        .noise = cfg.dspark_noise,
        .vocab = cfg.vocab,
        .vocab_local = cfg.vocab / world,
        .rank = rank,
        .world = world,
        .markov_rank = cfg.dspark_rank,
        .n_routed = cfg.dspark_routed,
        .topk = cfg.dspark_topk,
        .eps = cfg.eps,
        .hc_eps = cfg.hc_eps,
        .hc_iters = @intCast(cfg.hc_iters),
        .swiglu_limit = cfg.swiglu_limit,
        .route_scale = cfg.route_scale,
    };
}

/// The drafter's head from its loaded weights (cuda_weights.loadDSpark) and the target's head (this rank's
/// vocabulary columns).
pub fn headOf(ds: *const weights.DSpark, target_head: *const exl3.Linear) Head {
    return .{
        .main_proj = &ds.main_proj,
        .main_norm = ds.main_norm.ptr(),
        .norm = ds.norm.ptr(),
        .head = target_head,
        .markov_embed = ds.markov_embed.ptr(),
        .markov_head = ds.markov_head.ptr(),
        .markov_embed_host = ds.markov_embed_host,
        .conf = ds.conf_host,
    };
}

/// A batched pass's graph: its gathers, and the rings and RoPE tables it was recorded for.
pub const BatchGraph = struct {
    exec: cuda.graph.Exec,
    gathers: usize,
    key: Key,

    pub const max_rings = 4;
    pub const Key = struct { rings: [max_rings]Ptr = @splat(0), cos: Ptr = 0, sin: Ptr = 0 };
};

/// A draft stage's grouped EXL3 launches (dspark.py's fused stage: attn_in's group, q_proj's, wo_ab's).
pub const StageGroups = struct {
    in: exl3.Group, // wq_a, wkv
    q: exl3.Group, // wq_b
    wo_a: exl3.Group,
    wo_b: exl3.Group,

    pub fn init(gk: *const exl3.GroupKernels, s: Stage, d: *const cuda.Driver) !StageGroups {
        _ = d;
        var wo: [exl3.gmax]*const exl3.Linear = undefined;
        if (s.wo_a.len > exl3.gmax) return error.BadShape;
        for (s.wo_a, 0..) |*x, i| wo[i] = x;
        return .{
            .in = try exl3.Group.init(gk, &.{ s.wq_a, s.wkv }),
            .q = try exl3.Group.init(gk, &.{s.wq_b}),
            .wo_a = try exl3.Group.init(gk, wo[0..s.wo_a.len]),
            .wo_b = try exl3.Group.init(gk, &.{s.wo_b}),
        };
    }

    pub fn deinit(self: *StageGroups) void {
        _ = self;
    }
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    dims: Dims,
    set: *const aot.Set,
    glue: *const attention.Glue,
    hc: hc_mod.Hc,
    route: route_mod.Route,
    ex_kernels: *const experts_mod.Kernels,
    stages: []const Stage,
    head: Head,
    comm: ?*const cuda.comm.Comm,
    /// When set, every gather goes here instead of `comm` (a test's supply of the other ranks' partials).
    gather_hook: ?GatherHook = null,
    /// The prompt GEMM for absorbs past 128 rows (`enableWide`); null: 128-row pieces of the row kernel.
    wide: ?*Wide = null,
    /// Batched drafts replay graphs (TF_DS_GRAPHS, default on): one a stream count, recorded on first use or by
    /// `warmBatch`, again when the pool's rings or the RoPE tables differ from the recorded ones.
    graphs: bool = true,
    batch_graphs: [max_rows + 1]?BatchGraph = @splat(null),
    captures: usize = 0,
    /// Called before each graph launch (tests' gather supplies; the communicator needs none).
    launch_hook: ?rounds.GraphLaunch = null,
    counting: ?*usize = null, // a capture's gathers
    /// The passes' and absorbs' host tables, copied on the caller's stream (in order with the launches that read
    /// them, which may not have run yet: a plain upload runs outside that order).
    uploads: attention.Uploads,
    ex_scratch: experts_mod.Scratch,
    markov: Markov,
    // the block's row tables (int64)
    ids: cuda.DeviceBuffer,
    pos: cuda.DeviceBuffer,
    wpos: cuda.DeviceBuffer,
    bidx: cuda.DeviceBuffer,
    rbase: cuda.DeviceBuffer, // [rows] i64: each row's stream's ring rows start (slot x ring), batched drafts
    neg: cuda.DeviceBuffer,
    zero: cuda.DeviceBuffer,
    emb_ids: cuda.DeviceBuffer, // [block] i64: rows of `emb2` (0 = token, 1 = noise)
    emb2: cuda.DeviceBuffer, // bf16 [2, d]: the token's and the noise's embedding rows
    // activations (block rows)
    h: cuda.DeviceBuffer,
    pre: cuda.DeviceBuffer,
    pre_a: cuda.DeviceBuffer,
    pre_f: cuda.DeviceBuffer,
    post: cuda.DeviceBuffer,
    comb: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer,
    xh: cuda.DeviceBuffer,
    z: cuda.DeviceBuffer,
    qa: cuda.DeviceBuffer,
    qr: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    ykv: cuda.DeviceBuffer,
    kvb: cuda.DeviceBuffer,
    pm: cuda.DeviceBuffer,
    pl: cuda.DeviceBuffer,
    po: cuda.DeviceBuffer,
    o: cuda.DeviceBuffer,
    og: cuda.DeviceBuffer,
    ug: cuda.DeviceBuffer,
    u: cuda.DeviceBuffer,
    attn_out: cuda.DeviceBuffer, // f32 [n, d]
    gathered: cuda.DeviceBuffer, // f32 [world, n, d]
    parts: cuda.DeviceBuffer,
    pick: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    moe_out: cuda.DeviceBuffer, // f32 [n, d]
    xc: cuda.DeviceBuffer, // bf16 [n, d]
    xn: cuda.DeviceBuffer,
    local: cuda.DeviceBuffer, // f32 [n, vocab_local]
    lg: cuda.DeviceBuffer, // f32 [world, n, vocab_local] (the unsplit Markov loop's gather)
    // absorb
    abs_pos: cuda.DeviceBuffer,
    rr: attention.RopeRows, // the rope rows of a pass's positions (slot 0) and of an absorb's (slot 1)
    abs_slots: cuda.DeviceBuffer,
    main_x: cuda.DeviceBuffer,
    main_y: cuda.DeviceBuffer,
    abs_kv: cuda.DeviceBuffer,
    abs_rows: usize,
    // host
    host_out: []i64,
    host_xc: []u16,
    emb_pair: []u16,
    // the fused stages (batched passes) and the grouped absorb
    fused: bool,
    gk: exl3.GroupKernels,
    sg: []StageGroups,
    kv_group: exl3.Group, // every stage's wkv (absorbs)
    kvg: cuda.DeviceBuffer, // bf16 [stages, 128, hd]: the grouped absorb's outputs
    h2: cuda.DeviceBuffer,
    xh_in: cuda.DeviceBuffer,
    xh_q: cuda.DeviceBuffer,
    xh_woa: cuda.DeviceBuffer,
    xh_wob: cuda.DeviceBuffer,
    zg: cuda.DeviceBuffer,
    l2: rounds.L2,

    pub const max_absorb = 2048;
    /// Rows a draft pass holds at most (kernels.DECODE_ROWS): N streams x the block.
    pub const max_rows = 16;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, dims: Dims, set: *const aot.Set, glue: *const attention.Glue, ex_kernels: *const experts_mod.Kernels, stages: []const Stage, head: Head, comm: ?*const cuda.comm.Comm) !*Drafter {
        return initWith(gpa, d, dims, set, glue, ex_kernels, stages, head, comm, null);
    }

    /// `init` with the gathers going to `hook` when given (then `comm` may be null at any world).
    pub fn initWith(gpa: std.mem.Allocator, d: *const cuda.Driver, dims: Dims, set: *const aot.Set, glue: *const attention.Glue, ex_kernels: *const experts_mod.Kernels, stages: []const Stage, head: Head, comm: ?*const cuda.comm.Comm, hook: ?GatherHook) !*Drafter {
        const n = max_rows; // every per-row buffer holds a batched draft's rows (N streams x the block)
        if (dims.block > max_rows or dims.block < 1) return error.TooManyRows;
        if (dims.world > 1 and comm == null and hook == null) return error.NoComm;
        const self = try gpa.create(Drafter);
        errdefer gpa.destroy(self);
        self.* = undefined;
        self.gpa = gpa;
        self.d = d;
        self.dims = dims;
        self.set = set;
        self.glue = glue;
        self.ex_kernels = ex_kernels;
        self.stages = stages;
        self.head = head;
        self.comm = comm;
        self.gather_hook = hook;
        self.wide = null;
        self.graphs = rounds.graphsFromEnv();
        self.batch_graphs = @splat(null);
        self.captures = 0;
        self.launch_hook = null;
        self.counting = null;
        self.uploads = try attention.Uploads.init(d, attention.Uploads.room(2, max_absorb * 8) + attention.Uploads.room(5, max_rows * max_rows * 8) + attention.Uploads.room(Markov.np, 8));
        errdefer self.uploads.deinit();
        self.hc = hc_mod.Hc.init(set, @intCast(dims.d), dims.hc_iters, dims.eps, dims.hc_eps);
        self.route = route_mod.Route.init(set, @intCast(dims.d), @intCast(dims.n_routed), @intCast(dims.topk), dims.route_scale, @intCast(dims.n_routed));
        self.ex_scratch = try experts_mod.Scratch.init(d, stages[0].experts, n, dims.topk + 1);
        errdefer self.ex_scratch.deinit();
        self.markov = try Markov.init(gpa, d, set, dims);
        errdefer self.markov.deinit();
        const D = dims.d;
        const nd = n * D;
        // the row tables
        var tab: [16]i64 = @splat(0);
        self.ids = try cuda.DeviceBuffer.alloc(d, n * 8);
        self.pos = try cuda.DeviceBuffer.alloc(d, n * 8);
        self.wpos = try cuda.DeviceBuffer.alloc(d, n * 8);
        self.bidx = try cuda.DeviceBuffer.alloc(d, n * n * 8);
        self.rbase = try cuda.DeviceBuffer.alloc(d, n * 8);
        self.neg = try cuda.DeviceBuffer.alloc(d, n * 8);
        for (0..n) |i| tab[i] = -1;
        try self.neg.upload(0, std.mem.sliceAsBytes(tab[0..n]));
        self.zero = try cuda.DeviceBuffer.alloc(d, n * 8);
        try self.zero.fill8(0, null);
        self.emb_ids = try cuda.DeviceBuffer.alloc(d, n * 8);
        self.emb2 = try cuda.DeviceBuffer.alloc(d, (max_rows + 1) * D * 2);
        // activations
        self.h = try cuda.DeviceBuffer.alloc(d, nd * hc_mod.hc_streams * 2);
        self.pre = try cuda.DeviceBuffer.alloc(d, n * 4 * 4);
        self.pre_a = try cuda.DeviceBuffer.alloc(d, n * 4 * 4);
        self.pre_f = try cuda.DeviceBuffer.alloc(d, n * 4 * 4);
        self.post = try cuda.DeviceBuffer.alloc(d, n * 4 * 4);
        self.comb = try cuda.DeviceBuffer.alloc(d, n * 16 * 4);
        self.part = try cuda.DeviceBuffer.alloc(d, hc_mod.Hc.partWords(n) * 4);
        self.x = try cuda.DeviceBuffer.alloc(d, nd * 2);
        self.xh = try cuda.DeviceBuffer.alloc(d, @max(n, max_absorb) * 3 * D * 2);
        var zw: usize = head.head.scratchWords(n);
        zw = @max(zw, head.main_proj.scratchWords(max_absorb));
        for (stages) |s| {
            zw = @max(zw, s.wq_a.scratchWords(n));
            zw = @max(zw, s.wq_b.scratchWords(n));
            zw = @max(zw, s.wkv.scratchWords(max_absorb));
            zw = @max(zw, s.wo_b.scratchWords(n));
            for (s.wo_a) |w| zw = @max(zw, w.scratchWords(n));
        }
        self.z = try cuda.DeviceBuffer.alloc(d, @max(16, zw * 4));
        self.qa = try cuda.DeviceBuffer.alloc(d, n * dims.q_rank * 2);
        self.qr = try cuda.DeviceBuffer.alloc(d, n * dims.q_rank * 2);
        self.q = try cuda.DeviceBuffer.alloc(d, n * dims.heads * dims.hd * 2);
        self.ykv = try cuda.DeviceBuffer.alloc(d, n * dims.hd * 2);
        self.kvb = try cuda.DeviceBuffer.alloc(d, n * dims.hd * 2);
        const groups_h = dims.heads / 16;
        self.pm = try cuda.DeviceBuffer.alloc(d, n * groups_h * 8 * 16 * 4);
        self.pl = try cuda.DeviceBuffer.alloc(d, n * groups_h * 8 * 16 * 4);
        self.po = try cuda.DeviceBuffer.alloc(d, n * groups_h * 8 * 16 * dims.hd * 4);
        self.o = try cuda.DeviceBuffer.alloc(d, n * dims.heads * dims.hd * 2);
        self.og = try cuda.DeviceBuffer.alloc(d, n * (dims.heads * dims.hd / dims.groups) * 2);
        self.ug = try cuda.DeviceBuffer.alloc(d, n * dims.o_rank * 2);
        self.u = try cuda.DeviceBuffer.alloc(d, n * dims.groups * dims.o_rank * 2);
        self.attn_out = try cuda.DeviceBuffer.alloc(d, nd * 4);
        self.gathered = try cuda.DeviceBuffer.alloc(d, dims.world * nd * 4);
        self.parts = try cuda.DeviceBuffer.alloc(d, self.route.partsWords(n) * 4);
        self.pick = try cuda.DeviceBuffer.alloc(d, n * (dims.topk + 1) * 4);
        self.wts = try cuda.DeviceBuffer.alloc(d, n * (dims.topk + 1) * 4);
        self.moe_out = try cuda.DeviceBuffer.alloc(d, nd * 4);
        self.xc = try cuda.DeviceBuffer.alloc(d, nd * 2);
        self.xn = try cuda.DeviceBuffer.alloc(d, nd * 2);
        self.local = try cuda.DeviceBuffer.alloc(d, n * dims.vocab_local * 4);
        self.lg = try cuda.DeviceBuffer.alloc(d, dims.world * n * dims.vocab_local * 4);
        self.abs_pos = try cuda.DeviceBuffer.alloc(d, max_absorb * 8);
        self.rr = try attention.RopeRows.init(gpa, d, glue, max_absorb, 2, dims.rd / 2);
        self.abs_slots = try cuda.DeviceBuffer.alloc(d, max_absorb * 8);
        self.main_x = try cuda.DeviceBuffer.alloc(d, max_absorb * D * 2);
        self.main_y = try cuda.DeviceBuffer.alloc(d, max_absorb * D * 2);
        self.abs_kv = try cuda.DeviceBuffer.alloc(d, max_absorb * dims.hd * 2);
        self.abs_rows = max_absorb;
        self.host_out = try gpa.alloc(i64, Markov.np * (dims.block + 1));
        self.host_xc = try gpa.alloc(u16, nd);
        self.emb_pair = try gpa.alloc(u16, 2 * D);
        self.fused = rounds.fusedFromEnv();
        self.gk = try exl3.GroupKernels.load(d);
        self.sg = try gpa.alloc(StageGroups, stages.len);
        var in_b: usize = 16;
        var q_b: usize = 16;
        var woa_b: usize = 16;
        var wob_b: usize = 16;
        var zgw: usize = 4;
        var kv_l: [exl3.gmax]*const exl3.Linear = undefined;
        if (stages.len > exl3.gmax) return error.BadShape;
        for (self.sg, stages, 0..) |*g, st, j| {
            g.* = try StageGroups.init(&self.gk, st, d);
            in_b = @max(in_b, g.in.bufferBytes(n));
            q_b = @max(q_b, g.q.bufferBytes(n));
            woa_b = @max(woa_b, g.wo_a.bufferBytes(n));
            wob_b = @max(wob_b, g.wo_b.bufferBytes(n));
            inline for (.{ "in", "q", "wo_a", "wo_b" }) |f| zgw = @max(zgw, @field(g, f).scratchWords(n));
            kv_l[j] = st.wkv;
        }
        self.kv_group = try exl3.Group.init(&self.gk, kv_l[0..stages.len]);
        zgw = @max(zgw, self.kv_group.scratchWords(128));
        self.kvg = try cuda.DeviceBuffer.alloc(d, stages.len * 128 * dims.hd * 2);
        self.h2 = try cuda.DeviceBuffer.alloc(d, nd * hc_mod.hc_streams * 2);
        self.xh_in = try cuda.DeviceBuffer.alloc(d, @max(in_b, self.kv_group.bufferBytes(128)));
        self.xh_q = try cuda.DeviceBuffer.alloc(d, q_b);
        self.xh_woa = try cuda.DeviceBuffer.alloc(d, woa_b);
        self.xh_wob = try cuda.DeviceBuffer.alloc(d, wob_b);
        self.zg = try cuda.DeviceBuffer.alloc(d, zgw * 4);
        self.l2 = try rounds.L2.create(gpa, d, stages.len);
        {
            // dspark.py's forks: wo_a, wo_b after wq_b; the FFN's mix, gate and shared expert after wo_b; the next
            // stage's mix, input projections and wq_b (the head after the last) after the experts
            const fn_bytes = 24 * 4 * D * 4;
            var bufs: [16]cuda.DeviceBuffer = undefined;
            for (stages, 0..) |st, j| {
                var k: usize = 0;
                for (st.wo_a) |x| {
                    bufs[k] = x.words;
                    k += 1;
                }
                bufs[k] = st.wo_b.words;
                self.l2.setAttn(j, bufs[0 .. k + 1]);
                bufs[0] = .{ .d = d, .ptr = st.hc_ffn_fn, .len = fn_bytes };
                bufs[1] = .{ .d = d, .ptr = st.gate_w, .len = dims.n_routed * D * 2 };
                const shared = try rounds.L2.sharedTrellises(st.experts);
                @memcpy(bufs[2..5], &shared);
                self.l2.setMoe(j, bufs[0..5]);
                if (j + 1 < stages.len) {
                    const x = stages[j + 1];
                    bufs[0] = .{ .d = d, .ptr = x.hc_attn_fn, .len = fn_bytes };
                    bufs[1] = x.wq_a.words;
                    bufs[2] = x.wkv.words;
                    bufs[3] = x.wq_b.words;
                    self.l2.setNext(j, false, false, bufs[0..4]);
                } else self.l2.setNext(j, false, false, &.{head.head.words});
            }
        }
        return self;
    }

    pub fn deinit(self: *Drafter) void {
        inline for (.{ "ids", "pos", "wpos", "bidx", "rbase", "neg", "zero", "emb_ids", "emb2", "h", "pre", "pre_a", "pre_f", "post", "comb", "part", "x", "xh", "z", "qa", "qr", "q", "ykv", "kvb", "pm", "pl", "po", "o", "og", "ug", "u", "attn_out", "gathered", "parts", "pick", "wts", "moe_out", "xc", "xn", "local", "lg", "abs_pos", "abs_slots", "main_x", "main_y", "abs_kv" }) |f| @field(self, f).free();
        if (self.wide) |wd| wd.deinit(self.gpa);
        self.rr.deinit();
        self.dropGraphs();
        self.uploads.deinit();
        self.ex_scratch.deinit();
        self.markov.deinit();
        self.gpa.free(self.host_out);
        self.gpa.free(self.host_xc);
        self.gpa.free(self.emb_pair);
        for (self.sg) |*g| g.deinit();
        self.gpa.free(self.sg);
        inline for (.{ "kvg", "h2", "xh_in", "xh_q", "xh_woa", "xh_wob", "zg" }) |f| @field(self, f).free();
        self.gk.deinit();
        self.l2.deinit();
        self.gpa.destroy(self);
    }

    /// Absorbs past 128 rows (a prompt chunk's last `ring` rows) through the prompt GEMM, as the fork's `mm` runs
    /// them; `kernels`: the EXL3 kernels the weights were loaded with (their unpack instantiations).
    pub fn enableWide(self: *Drafter, kernels: *const exl3.Kernels) !void {
        if (self.wide != null) return;
        var kn: usize = self.head.main_proj.k * self.head.main_proj.n;
        var kmax: usize = self.head.main_proj.k;
        for (self.stages) |s| {
            kn = @max(kn, s.wkv.k * s.wkv.n);
            kmax = @max(kmax, s.wkv.k);
        }
        self.wide = try Wide.init(self.gpa, self.d, kernels, self.set, self.dims.ring, kmax, kn);
    }

    pub fn newCache(self: *Drafter) !Cache {
        return Cache.init(self.gpa, self.d, self.stages.len, self.dims);
    }

    // -- absorb --------------------------------------------------------------------------------------------------

    /// The target's taps [n, 3 d] bf16 (device) of positions start .. start + n - 1 into every stage's ring; `embed`:
    /// the target's embedding table (unused here; the draft reads it). At most `ring` rows matter: older ones are
    /// overwritten in the ring anyway, as in the fork (`keep = min(n, ring_size)`).
    pub fn absorb(self: *Drafter, cache: *Cache, taps: Ptr, start: usize, n_in: usize, rope: Rope, stream: cuda.Stream) !void {
        const dm = self.dims;
        var n = n_in;
        var first = start;
        var tap_ptr = taps;
        if (n > dm.ring) {
            first = start + n - dm.ring;
            tap_ptr = taps + (n - dm.ring) * 3 * dm.d * 2;
            n = dm.ring;
        }
        if (n == 0) {
            cache.absorbed = start + n_in;
            return;
        }
        if (n > self.abs_rows) return error.TooManyRows;
        var tab: [max_absorb]i64 = undefined;
        try self.uploads.begin(); // (waits for the last tables' copies: their staging is reused)
        for (0..n) |i| tab[i] = @intCast(first + i);
        try self.uploads.put(self.abs_pos, 0, std.mem.sliceAsBytes(tab[0..n]), stream);
        for (0..n) |i| tab[i] = @intCast((first + i) % dm.ring);
        try self.uploads.put(self.abs_slots, 0, std.mem.sliceAsBytes(tab[0..n]), stream);
        const ra = try self.rr.at(rope, 1, self.abs_pos.ptr, n, stream);
        // main_x = rmsnorm(main_proj(taps), main_norm), then every stage's keys into its ring; the EXL3 row kernel
        // takes 128 rows a launch (the fork's prompt GEMM beyond that is a cuBLAS path: pieces here)
        // model.py `mm`: past 128 rows the prompt GEMM (`wide`), whose bits differ from the row kernel's
        const wide = if (n > 128) self.wide else null;
        if (wide) |wd| try wd.run(self.head.main_proj, tap_ptr, n, self.main_y.ptr, .bf16, stream);
        var at: usize = 0;
        while (at < n) : (at += 128) {
            const m = @min(128, n - at);
            if (wide == null) try self.head.main_proj.run(tap_ptr + at * 3 * dm.d * 2, .bf16, m, self.xh.ptr, self.z.ptr, self.main_y.ptr + at * dm.d * 2, .bf16, stream);
            try self.rmsnorm(self.main_y.ptr + at * dm.d * 2, self.head.main_norm, self.main_x.ptr + at * dm.d * 2, m, dm.d, stream);
        }
        for (self.stages, cache.rings) |s, ring| {
            if (wide) |wd| try wd.run(s.wkv, self.main_x.ptr, n, self.abs_kv.ptr, .bf16, stream);
            at = 0;
            while (at < n) : (at += 128) {
                const m = @min(128, n - at);
                if (wide == null) try s.wkv.run(self.main_x.ptr + at * dm.d * 2, .bf16, m, self.xh.ptr, self.z.ptr, self.abs_kv.ptr + at * dm.hd * 2, .bf16, stream);
                try self.kvNormRope(self.abs_kv.ptr + at * dm.hd * 2, s.kv_norm, ra.from(at), self.abs_slots.ptr + at * 8, ring.ptr, self.abs_kv.ptr + at * dm.hd * 2, m, stream);
            }
        }
        cache.absorbed = start + n_in;
    }

    // -- draft -----------------------------------------------------------------------------------------------------

    /// A test's look at a named buffer after a step ("h0", "s{j}_x1", "s{j}_attn", "s{j}_h1", "s{j}_x2",
    /// "s{j}_moe", "s{j}_h2").
    pub const Hook = *const fn (ctx: *anyopaque, name: []const u8, buf: Ptr) anyerror!void;

    /// Greedy drafts after `token` (which sits at position q0, every position before it absorbed); `embed` is the
    /// target's embedding table bf16 [V, d] on the device, read for the token and the noise.
    pub fn draft(self: *Drafter, cache: *const Cache, token: u32, q0: usize, embed: Ptr, rope: Rope, stream: cuda.Stream) !Drafts {
        if (cache.absorbed != q0) return error.BadShape;
        const D = self.dims.d;
        try self.emb2.copyFrom(0, embed + @as(u64, token) * D * 2, D * 2, stream.handle);
        try self.emb2.copyFrom(D * 2, embed + @as(u64, self.dims.noise) * D * 2, D * 2, stream.handle);
        var out: [1]Drafts = undefined;
        try self.runRows(1, &.{token}, &.{q0}, null, cache.rings, rope, stream, null, null, &out);
        return out[0];
    }

    /// `draft` with `table2` a two-row table (row 0 the token's embedding, row 1 the noise's) and a hook after each
    /// step: what a fixture test needs.
    pub fn draftWithTable(self: *Drafter, cache: *const Cache, token: u32, q0: usize, table2: Ptr, rope: Rope, stream: cuda.Stream, ctx: *anyopaque, hook: Hook) !Drafts {
        if (cache.absorbed != q0) return error.BadShape;
        try self.emb2.copyFrom(0, table2, 2 * self.dims.d * 2, stream.handle);
        var out: [1]Drafts = undefined;
        try self.runRows(1, &.{token}, &.{q0}, null, cache.rings, rope, stream, ctx, hook, &out);
        return out[0];
    }

    /// dspark.py `BatchDraftGraph`: drafts for N streams in one pass, each stream's block at its own positions in its
    /// own slot's rings (`pool`), the Markov loop batched. `embed`: the target's table; N x block rows <= max_rows.
    pub fn batchDraft(self: *Drafter, pool: *const DraftPool, tokens: []const u32, q0: []const usize, slots: []const usize, embed: Ptr, rope: Rope, stream: cuda.Stream, out: []Drafts) !void {
        const N = tokens.len;
        if (N == 0 or N * self.dims.block > max_rows or q0.len != N or slots.len != N or out.len < N) return error.TooManyRows;
        const D = self.dims.d;
        for (tokens, 0..) |t, s| try self.emb2.copyFrom(s * D * 2, embed + @as(u64, t) * D * 2, D * 2, stream.handle);
        try self.emb2.copyFrom(N * D * 2, embed + @as(u64, self.dims.noise) * D * 2, D * 2, stream.handle);
        try self.runRows(N, tokens, q0, slots, pool.rings, rope, stream, null, null, out);
    }

    /// `batchDraft` with the embedding rows given as a table [N + 1, d] (the tokens' rows, then the noise's) and a
    /// hook (tests; with a hook the pass runs eager, without one it replays its graph as `batchDraft`).
    pub fn batchDraftWithTable(self: *Drafter, pool: *const DraftPool, tokens: []const u32, q0: []const usize, slots: []const usize, table: Ptr, rope: Rope, stream: cuda.Stream, ctx: ?*anyopaque, hook: ?Hook, out: []Drafts) !void {
        const N = tokens.len;
        if (N == 0 or N * self.dims.block > max_rows or q0.len != N or slots.len != N or out.len < N) return error.TooManyRows;
        try self.emb2.copyFrom(0, table, (N + 1) * self.dims.d * 2, stream.handle);
        try self.runRows(N, tokens, q0, slots, pool.rings, rope, stream, ctx, hook, out);
    }

    /// The pass over N streams' blocks (R = N x block rows): `slots` null = one stream on its own rings (`rings` a
    /// cache's, every row's ring base 0, HAS_BASE off), else each row addresses its stream's rows of the pool's ring
    /// planes through per-row bases. `self.emb2` holds the embedding rows: stream s's token at row s, the noise at
    /// row N.
    fn runRows(self: *Drafter, N: usize, tokens: []const u32, q0: []const usize, slots: ?[]const usize, rings: []const cuda.DeviceBuffer, rope: Rope, stream: cuda.Stream, ctx: ?*anyopaque, hook: ?Hook, out: []Drafts) !void {
        const dm = self.dims;
        const n = dm.block;
        const R = N * n;
        const D = dm.d;
        const batched = slots != null;
        // the row tables: embedding rows, positions, the window position every row sees (q0 - 1), the ring bases
        var tab: [max_rows]i64 = undefined;
        try self.uploads.begin(); // (waits for the last tables' copies: their staging is reused)
        for (0..R) |r| tab[r] = if (r % n == 0) @intCast(r / n) else @intCast(N);
        try self.uploads.put(self.emb_ids, 0, std.mem.sliceAsBytes(tab[0..R]), stream);
        for (0..R) |r| tab[r] = @intCast(q0[r / n] + r % n);
        try self.uploads.put(self.pos, 0, std.mem.sliceAsBytes(tab[0..R]), stream);
        for (0..R) |r| tab[r] = @as(i64, @intCast(q0[r / n])) - 1;
        try self.uploads.put(self.wpos, 0, std.mem.sliceAsBytes(tab[0..R]), stream);
        if (batched) {
            for (0..R) |r| tab[r] = @intCast(slots.?[r / n] * dm.ring);
            try self.uploads.put(self.rbase, 0, std.mem.sliceAsBytes(tab[0..R]), stream);
        }
        // every row of a stream's block sees that stream's block rows (the keys as `comp`, no mask inside the block)
        var bi: [max_rows * max_rows]i64 = undefined;
        for (0..R) |r| for (0..n) |c| {
            bi[r * n + c] = @intCast((r / n) * n + c);
        };
        try self.uploads.put(self.bidx, 0, std.mem.sliceAsBytes(bi[0 .. R * n]), stream);
        try self.markov.seed(tokens, N, &self.uploads, stream);
        if (batched and hook == null and self.graphs and rings.len <= BatchGraph.max_rings) {
            const g = try self.batchGraph(N, rings, rope, stream);
            if (self.launch_hook) |hk| try hk.call(hk.ctx, g.gathers, stream);
            try g.exec.launchOn(stream);
        } else try self.pass(N, rings, batched, rope, stream, ctx, hook);
        try stream.synchronize();
        try self.markov.out.download(0, std.mem.sliceAsBytes(self.host_out[0 .. N * (n + 1)]));
        try self.xc.download(0, std.mem.sliceAsBytes(self.host_xc[0 .. R * D]));
        for (0..N) |s_| {
            var o: Drafts = .{ .tokens = undefined, .conf = undefined, .len = n };
            for (0..n) |i| o.tokens[i] = @intCast(self.host_out[s_ * (n + 1) + i + 1]);
            self.confidence(&o, s_, n, N);
            out[s_] = o;
        }
    }

    /// The pass's launches after its tables (`runRows`): the embedding rows, every stage, the head, the Markov loop
    /// into `markov.out` (its first tokens already there).
    fn pass(self: *Drafter, N: usize, rings: []const cuda.DeviceBuffer, batched: bool, rope: Rope, stream: cuda.Stream, ctx: ?*anyopaque, hook: ?Hook) !void {
        const dm = self.dims;
        const n = dm.block;
        const R = N * n;
        const D = dm.d;
        const ra = try self.rr.at(rope, 0, self.pos.ptr, R, stream);
        try self.hc.embedInit(stream, self.emb2.ptr, self.emb_ids.ptr, self.h.ptr, self.pre.ptr, @intCast(R));
        if (hook) |f| try f(ctx.?, "h0", self.h.ptr);
        var pre = self.pre.ptr;
        var pre_f = self.pre_f.ptr;
        var nb: [32]u8 = undefined;
        var h = self.h.ptr;
        if (batched and self.fused) {
            const st = try self.fusedStages(R, rings, ra, stream, ctx, hook);
            h = st.h;
            pre = st.pre;
        } else for (self.stages, rings, 0..) |s, ring, j| {
            try self.hc.pre(stream, self.h.ptr, s.hc_attn_fn, s.hc_attn_scale, s.hc_attn_base, pre, s.attn_norm, self.x.ptr, self.pre_a.ptr, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(R));
            if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_x1", .{j}), self.x.ptr);
            const pa = try self.stageAttention(s, ring.ptr, R, batched, ra, stream);
            if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_attn", .{j}), pa);
            const ga = try self.gather(pa, R * D, stream);
            try self.hc.post(stream, ga, @intCast(dm.world), self.h.ptr, self.post.ptr, self.comb.ptr, self.h.ptr, @intCast(R));
            if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_h1", .{j}), self.h.ptr);
            try self.hc.pre(stream, self.h.ptr, s.hc_ffn_fn, s.hc_ffn_scale, s.hc_ffn_base, self.pre_a.ptr, s.ffn_norm, self.x.ptr, pre_f, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(R));
            if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_x2", .{j}), self.x.ptr);
            const pmo = try self.stageMoe(s, R, stream);
            if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_moe", .{j}), pmo);
            const gm = try self.gather(pmo, R * D, stream);
            try self.hc.post(stream, gm, @intCast(dm.world), self.h.ptr, self.post.ptr, self.comb.ptr, self.h.ptr, @intCast(R));
            if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_h2", .{j}), self.h.ptr);
            const t = pre;
            pre = pre_f;
            pre_f = t;
        }
        // xc = collapse(h, pre); local = head(rmsnorm(xc, norm))
        try self.set.run(stream, "_collapse", .{ @intCast(R), @intCast(D / 1024), 1 }, &.{
            aot.ptr("X", "*bf16", h), aot.ptr("PRE", "*fp32", pre), aot.ptr("OUT", "*bf16", self.xc.ptr),
        }, &.{ aot.ci("D", @intCast(D)), aot.ci("BLOCK", 1024) });
        try self.rmsnorm(self.xc.ptr, self.head.norm, self.xn.ptr, R, D, stream);
        try self.head.head.run(self.xn.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.local.ptr, .f32, stream);
        // the Markov loop: out[s, 0] = token, out[s, 1..] the drafts
        const lg: Ptr = if (self.markov.split) self.local.ptr else blk: {
            if (dm.world == 1) break :blk self.local.ptr;
            try self.allGather(self.local.ptr, self.lg.ptr, R * dm.vocab_local, stream);
            break :blk self.lg.ptr;
        };
        try self.markov.loop(lg, self.head, N, n, n, self, stream);
    }

    /// dspark.py `BatchDraftGraph._stages_fused`: every stage with each sublayer's post fused into the next
    /// sublayer's mixes (kernels.hc_pre2 without a side stream: `_hc_mix_part` posted, then `_hc_finish_s` with both
    /// halves), the last post alone, the L2 forks joined at the end. Returns the streams' buffer and the last pre-mix.
    fn fusedStages(self: *Drafter, R: usize, rings: []const cuda.DeviceBuffer, ra: attention.RopeAt, stream: cuda.Stream, ctx: ?*anyopaque, hook: ?Hook) !struct { h: Ptr, pre: Ptr } {
        const dm = self.dims;
        const D = dm.d;
        const rows: u32 = @intCast(R);
        var h = self.h.ptr;
        var spare = self.h2.ptr;
        var pre = self.pre.ptr;
        var pre_f = self.pre_f.ptr;
        var pending: ?Ptr = null; // the last sublayer's gathered partials, not yet posted
        var nb: [32]u8 = undefined;
        for (self.stages, rings, self.sg, 0..) |s, ring, *g, j| {
            inline for (0..2) |sub| {
                const fn_ = if (sub == 0) s.hc_attn_fn else s.hc_ffn_fn;
                const scale = if (sub == 0) s.hc_attn_scale else s.hc_ffn_scale;
                const base = if (sub == 0) s.hc_attn_base else s.hc_ffn_base;
                const norm = if (sub == 0) s.attn_norm else s.ffn_norm;
                const pre_in = if (sub == 0) pre else self.pre_a.ptr;
                const pre_out = if (sub == 0) self.pre_a.ptr else pre_f;
                if (pending) |ga| {
                    try self.hc.mixPart(stream, h, spare, ga, @intCast(dm.world), self.post.ptr, self.comb.ptr, fn_, self.part.ptr, rows, true, true);
                    const t = h;
                    h = spare;
                    spare = t;
                    pending = null;
                    if (hook) |f| try f(ctx.?, if (sub == 0) try std.fmt.bufPrint(&nb, "s{d}_h2", .{j - 1}) else try std.fmt.bufPrint(&nb, "s{d}_h1", .{j}), h);
                } else try self.hc.mixPart(stream, h, h, null, 1, self.post.ptr, self.comb.ptr, fn_, self.part.ptr, rows, true, false);
                try self.hc.finishS(stream, h, self.part.ptr, base, scale, pre_in, norm, self.x.ptr, pre_out, self.post.ptr, self.comb.ptr, rows, true);
                if (sub == 0) {
                    if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_x1", .{j}), self.x.ptr);
                    const pa = try self.fusedAttention(s, g, ring.ptr, R, ra, stream, j);
                    if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_attn", .{j}), pa);
                    pending = try self.gather(pa, R * D, stream);
                } else {
                    if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_x2", .{j}), self.x.ptr);
                    const pmo = try self.stageMoe(s, R, stream);
                    try self.l2.next(j, self.stages.len, stream);
                    if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_moe", .{j}), pmo);
                    pending = try self.gather(pmo, R * D, stream);
                }
            }
            const t = pre;
            pre = pre_f;
            pre_f = t;
        }
        if (pending) |gm| {
            try self.hc.post(stream, gm, @intCast(dm.world), h, self.post.ptr, self.comb.ptr, h, rows);
            if (hook) |f| try f(ctx.?, try std.fmt.bufPrint(&nb, "s{d}_h2", .{self.stages.len - 1}), h);
        }
        try self.l2.join(stream);
        return .{ .h = h, .pre = pre };
    }

    /// dspark.py `BatchDraftGraph._attention(fused=True)`: wq_a and wkv in one grouped launch, q's RMSNorm writing
    /// wq_b's rotated rows (`_rmsnorm_rot`) and q's RoPE in wq_b's epilogue, the block's keys and the attention as
    /// the plain stage, wo_a's inputs rotated from o's column blocks in place and wo_a's epilogue writing wo_b's
    /// rotated rows (model.py `wo_ab`). This rank's fp32 partial.
    fn fusedAttention(self: *Drafter, s: Stage, g: *const StageGroups, ring: Ptr, R: usize, ra: attention.RopeAt, stream: cuda.Stream, j: usize) !Ptr {
        const dm = self.dims;
        const n = dm.block;
        {
            const xr: exl3.Rows = .{ .ptr = self.x.ptr, .dtype = .bf16, .ld = dm.d };
            const outs = [_]exl3.Rows{ .{ .ptr = self.qa.ptr, .dtype = .bf16, .ld = dm.q_rank }, .{ .ptr = self.ykv.ptr, .dtype = .bf16, .ld = dm.hd } };
            try g.in.rotate(&.{ xr, xr }, R, self.xh_in.ptr, stream);
            try g.in.rotated(R, self.xh_in.ptr, &outs, self.zg.ptr, null, null, stream);
        }
        {
            const block = std.math.ceilPowerOfTwo(usize, dm.q_rank) catch return error.BadShape;
            try self.set.run(stream, "_rmsnorm_rot", .{ @intCast(R), 1, 1 }, &.{
                aot.ptr("X", "*bf16", self.qa.ptr),     aot.int("xs", @intCast(dm.q_rank)), aot.ptr("W", "*bf16", s.q_norm),
                aot.ptr("OUT", "*bf16", self.qr.ptr),   aot.int("os_", @intCast(dm.q_rank)), aot.float("eps", dm.eps),
                aot.ptr("S0", "*fp16", s.wq_b.suh.ptr), aot.ptr("H0", "*fp16", self.xh_q.ptr),
                aot.ptr("S1", "*bf16", self.qr.ptr),    aot.ptr("H1", "*bf16", self.qr.ptr),
            }, &.{ aot.ci("D", @intCast(dm.q_rank)), aot.ci("BLOCK", @intCast(block)), aot.ci("NROT", 1), aot.ci("PDL", 1) });
            var flags: [exl3.gmax]bool = @splat(false);
            flags[0] = true;
            const outs = [_]exl3.Rows{.{ .ptr = self.q.ptr, .dtype = .bf16, .ld = dm.heads * dm.hd }};
            try g.q.rotated(R, self.xh_q.ptr, &outs, self.zg.ptr, null, .{ .cos = ra.cos, .sin = ra.sin, .pos = ra.pos, .hd = dm.hd, .rd = dm.rd, .layers = flags }, stream);
        }
        try self.l2.attn(j, stream);
        try self.kvNormRope(self.ykv.ptr, s.kv_norm, ra, self.neg.ptr, ring, self.kvb.ptr, R, stream);
        const groups_h = dm.heads / 16;
        const picks: usize = (n + 31) / 32;
        const nblk: usize = dm.window / 32 + picks;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dm.hd))));
        try self.set.run(stream, "_sparse_attn_part", .{ @intCast(R), @intCast(groups_h), 8 }, &.{
            aot.ptr("Q", "*bf16", self.q.ptr),      aot.ptr("WSRC", "*bf16", ring),
            aot.ptr("WLO", "*i64", self.zero.ptr),  aot.ptr("COMP", "*bf16", self.kvb.ptr),
            aot.ptr("CSC", "*bf16", ring),          aot.ptr("IDX", "*i64", self.bidx.ptr),
            aot.ptr("POS", "*i64", self.wpos.ptr),  aot.ptr("PM", "*fp32", self.pm.ptr),
            aot.ptr("PL", "*fp32", self.pl.ptr),    aot.ptr("PO", "*fp32", self.po.ptr),
            aot.ptr("SINK", "*fp32", s.sink),       aot.ptr("OUT", "*bf16", self.o.ptr),
            aot.float("scale", scale),              aot.int("ring_size", @intCast(dm.ring)),
            aot.int("n_idx", @intCast(n)),          aot.ptr("RBASE", "*i64", self.rbase.ptr),
            aot.ptr("CBASE", "*i64", self.zero.ptr),
        }, &.{
            aot.ci("H", @intCast(dm.heads)), aot.ci("HD", @intCast(dm.hd)), aot.ci("HB", 16),      aot.ci("WIN", @intCast(dm.window)),
            aot.ci("BN", 32),                aot.ci("RING", 1),               aot.ci("HAS_COMP", 1), aot.ci("PACKED", 0),
            aot.ci("SPLITS", 8),             aot.ci("NBLK", @intCast(nblk)),  aot.ci("FINAL", 0),    aot.ci("HAS_BASE", 1),
            aot.ci("PDL", 1),                aot.ci("WPF", 1),
        });
        try self.set.run(stream, "_sparse_attn_merge", .{ @intCast(R), @intCast(groups_h), 1 }, &.{
            aot.ptr("PM", "*fp32", self.pm.ptr), aot.ptr("PL", "*fp32", self.pl.ptr), aot.ptr("PO", "*fp32", self.po.ptr),
            aot.ptr("SINK", "*fp32", s.sink),   aot.ptr("OUT", "*bf16", self.o.ptr),
        }, &.{ aot.ci("H", @intCast(dm.heads)), aot.ci("HD", @intCast(dm.hd)), aot.ci("HB", 16), aot.ci("SPLITS", 8), aot.ci("PDL", 1) });
        try self.ropeHeads(self.o.ptr, ra, R, dm.heads, dm.hd, true, stream);
        {
            const groups = g.wo_a.n;
            const o_cols = dm.heads * dm.hd;
            const g_cols = o_cols / groups;
            var xs: [exl3.gmax]exl3.Rows = undefined;
            var outs: [exl3.gmax]exl3.Rows = undefined;
            var rot: [exl3.gmax]?exl3.RotOut = @splat(null);
            var col: usize = 0;
            for (0..groups) |q| {
                xs[q] = .{ .ptr = self.o.ptr + q * g_cols * 2, .dtype = .bf16, .ld = o_cols };
                outs[q] = .{ .ptr = self.u.ptr + col * 2, .dtype = .bf16, .ld = groups * dm.o_rank };
                rot[q] = .{ .suh = s.wo_b.suh.ptr, .xh = self.xh_wob.ptr, .ld = s.wo_b.k, .off = col };
                col += s.wo_a[q].n;
            }
            try g.wo_a.rotate(xs[0..groups], R, self.xh_woa.ptr, stream);
            try g.wo_a.rotated(R, self.xh_woa.ptr, outs[0..groups], self.zg.ptr, rot[0..groups], null, stream);
            const ob = [_]exl3.Rows{.{ .ptr = self.attn_out.ptr, .dtype = .f32, .ld = dm.d }};
            try g.wo_b.rotated(R, self.xh_wob.ptr, &ob, self.zg.ptr, null, null, stream);
        }
        try self.l2.moe(j, stream);
        return self.attn_out.ptr;
    }

    /// The graph of a batched pass of N streams over `rings` with `rope`, recorded (nothing runs) when missing or
    /// recorded for other rings or tables.
    fn batchGraph(self: *Drafter, N: usize, rings: []const cuda.DeviceBuffer, rope: Rope, stream: cuda.Stream) !*BatchGraph {
        var key: BatchGraph.Key = .{ .cos = if (rope.freqs) |f| f.ptr else rope.cos.ptr, .sin = rope.sin.ptr };
        for (rings, key.rings[0..rings.len]) |r, *k| k.* = r.ptr;
        if (self.batch_graphs[N]) |*g| {
            if (std.meta.eql(g.key, key)) return g;
            g.exec.deinit();
            self.batch_graphs[N] = null;
        }
        var gathers: usize = 0;
        self.counting = &gathers;
        defer self.counting = null;
        try cuda.graph.beginCapture(stream, .thread_local);
        self.pass(N, rings, true, rope, stream, null, null) catch |err| {
            if (cuda.graph.endCapture(stream)) |gr| {
                var x = gr;
                x.deinit();
            } else |_| {}
            return err;
        };
        var gr = try cuda.graph.endCapture(stream);
        defer gr.deinit();
        var exec = try gr.instantiate();
        errdefer exec.deinit();
        try exec.upload(stream);
        self.batch_graphs[N] = .{ .exec = exec, .gathers = gathers, .key = key };
        self.captures += 1;
        return &self.batch_graphs[N].?;
    }

    /// Records the batched passes of 1 .. `most` streams over `pool` ahead of the drafts that replay them; nothing
    /// runs. Every rank calls it with the same arguments (the gathers are recorded, not issued).
    pub fn warmBatch(self: *Drafter, pool: *const DraftPool, most: usize, rope: Rope, stream: cuda.Stream) !void {
        if (!self.graphs or pool.rings.len > BatchGraph.max_rings) return;
        for (1..most + 1) |N| {
            if (N * self.dims.block > max_rows) break;
            _ = try self.batchGraph(N, pool.rings, rope, stream);
        }
        try stream.synchronize();
    }

    /// Drops every recorded graph.
    pub fn dropGraphs(self: *Drafter) void {
        for (&self.batch_graphs) |*g| if (g.*) |*x| {
            x.exec.deinit();
            g.* = null;
        };
    }

    /// conf[i] = conf_w . [xc[row] (f32), markov_embed[out[s, i]] (f32)] (the fork: a cuBLAS fp32 matmul; here a
    /// host dot in the row's order, so the last bits may differ: the confidence feeds depth policies, never a token).
    fn confidence(self: *Drafter, o: *Drafts, s_: usize, n: usize, N: usize) void {
        _ = N;
        const dm = self.dims;
        for (0..n) |i| {
            const row = s_ * n + i;
            var acc: f64 = 0;
            for (0..dm.d) |j| acc += @as(f64, bf16ToF32(self.host_xc[row * dm.d + j])) * @as(f64, f16ToF32(self.head.conf[j]));
            const tok: usize = @intCast(self.host_out[s_ * (n + 1) + i]); // the token before draft i, as the fork's embs
            for (0..dm.markov_rank) |k| acc += @as(f64, bf16ToF32(self.head.markov_embed_host[tok * dm.markov_rank + k])) * @as(f64, f16ToF32(self.head.conf[dm.d + k]));
            o.conf[i] = @floatCast(acc);
        }
    }

    fn gather(self: *Drafter, partial: Ptr, count: usize, stream: cuda.Stream) !Ptr {
        if (self.dims.world == 1) return partial;
        try self.allGather(partial, self.gathered.ptr, count, stream);
        return self.gathered.ptr;
    }

    /// out [world, words] f32 <- every rank's `partial` [words] in rank order: through `gather_hook` when set, else
    /// the communicator. Every gather a draft issues goes through here, in the fork's order.
    pub fn allGather(self: *const Drafter, partial: Ptr, out: Ptr, words: usize, stream: cuda.Stream) !void {
        if (self.counting) |c| c.* += 1;
        if (self.gather_hook) |g| return g.call(g.ctx, partial, words, out, stream);
        const comm = self.comm orelse return error.NoComm;
        try comm.allGather(partial, out, words, .f32, stream.handle);
    }

    /// dspark.py `Drafter._attention` / `BatchDraftGraph._attention` (unfused): the rows attend their stream's ring's
    /// newest 128 absorbed positions (ending at q0 - 1) and their block's rows (its keys as `comp`, no mask inside
    /// the block); this rank's fp32 partial. `batched`: per-row ring bases into the pool's plane (HAS_BASE).
    fn stageAttention(self: *Drafter, s: Stage, ring: Ptr, R: usize, batched: bool, ra: attention.RopeAt, stream: cuda.Stream) !Ptr {
        const dm = self.dims;
        const n = dm.block;
        try s.wq_a.run(self.x.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.qa.ptr, .bf16, stream);
        try s.wkv.run(self.x.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.ykv.ptr, .bf16, stream);
        try self.rmsnorm(self.qa.ptr, s.q_norm, self.qr.ptr, R, dm.q_rank, stream);
        try s.wq_b.run(self.qr.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.q.ptr, .bf16, stream);
        try self.ropeHeads(self.q.ptr, ra, R, dm.heads, dm.hd, false, stream);
        // the block's keys, normalized, roped and quantized like the ring's, not stored (slots -1)
        try self.kvNormRope(self.ykv.ptr, s.kv_norm, ra, self.neg.ptr, ring, self.kvb.ptr, R, stream);
        if (R > max_rows) return error.TooManyRows;
        const groups_h = dm.heads / 16;
        const picks: usize = (n + 31) / 32;
        const nblk: usize = dm.window / 32 + picks;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dm.hd))));
        try self.set.run(stream, "_sparse_attn_part", .{ @intCast(R), @intCast(groups_h), 8 }, &.{
            aot.ptr("Q", "*bf16", self.q.ptr),      aot.ptr("WSRC", "*bf16", ring),
            aot.ptr("WLO", "*i64", self.zero.ptr),  aot.ptr("COMP", "*bf16", self.kvb.ptr),
            aot.ptr("CSC", "*bf16", ring),          aot.ptr("IDX", "*i64", self.bidx.ptr),
            aot.ptr("POS", "*i64", self.wpos.ptr),  aot.ptr("PM", "*fp32", self.pm.ptr),
            aot.ptr("PL", "*fp32", self.pl.ptr),    aot.ptr("PO", "*fp32", self.po.ptr),
            aot.ptr("SINK", "*fp32", s.sink),       aot.ptr("OUT", "*bf16", self.o.ptr),
            aot.float("scale", scale),              aot.int("ring_size", @intCast(dm.ring)),
            aot.int("n_idx", @intCast(n)),          aot.ptr("RBASE", "*i64", if (batched) self.rbase.ptr else self.wpos.ptr),
            aot.ptr("CBASE", "*i64", if (batched) self.zero.ptr else self.wpos.ptr),
        }, &.{
            aot.ci("H", @intCast(dm.heads)), aot.ci("HD", @intCast(dm.hd)), aot.ci("HB", 16),      aot.ci("WIN", @intCast(dm.window)),
            aot.ci("BN", 32),                aot.ci("RING", 1),               aot.ci("HAS_COMP", 1), aot.ci("PACKED", 0),
            aot.ci("SPLITS", 8),             aot.ci("NBLK", @intCast(nblk)),  aot.ci("FINAL", 0),    aot.ci("HAS_BASE", @intFromBool(batched)),
            aot.ci("PDL", 1),                aot.ci("WPF", 1),
        });
        try self.set.run(stream, "_sparse_attn_merge", .{ @intCast(R), @intCast(groups_h), 1 }, &.{
            aot.ptr("PM", "*fp32", self.pm.ptr), aot.ptr("PL", "*fp32", self.pl.ptr), aot.ptr("PO", "*fp32", self.po.ptr),
            aot.ptr("SINK", "*fp32", s.sink),   aot.ptr("OUT", "*bf16", self.o.ptr),
        }, &.{ aot.ci("H", @intCast(dm.heads)), aot.ci("HD", @intCast(dm.hd)), aot.ci("HB", 16), aot.ci("SPLITS", 8), aot.ci("PDL", 1) });
        try self.ropeHeads(self.o.ptr, ra, R, dm.heads, dm.hd, true, stream);
        const o_cols = dm.heads * dm.hd;
        const g_cols = o_cols / dm.groups;
        for (0..dm.groups) |g| {
            try self.copyCols(self.o.ptr + g * g_cols * 2, o_cols, self.og.ptr, g_cols, R, g_cols, stream);
            try s.wo_a[g].run(self.og.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.ug.ptr, .bf16, stream);
            try self.copyCols(self.ug.ptr, dm.o_rank, self.u.ptr + g * dm.o_rank * 2, dm.groups * dm.o_rank, R, dm.o_rank, stream);
        }
        try s.wo_b.run(self.u.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.attn_out.ptr, .f32, stream);
        return self.attn_out.ptr;
    }

    /// model.py `Model.moe` at the drafter's top-k: the gate's chunk sums, the picks, the routed experts' fp32 partial.
    fn stageMoe(self: *Drafter, s: Stage, R: usize, stream: cuda.Stream) !Ptr {
        try self.route.gate(stream, self.x.ptr, s.gate_w, self.parts.ptr, @intCast(R));
        try self.route.pick(stream, self.parts.ptr, s.gate_b, self.pick.ptr, self.wts.ptr, @intCast(R));
        try experts_mod.routed(self.ex_kernels, s.experts, &self.ex_scratch, self.x.ptr, self.pick.ptr, self.wts.ptr, R, self.dims.swiglu_limit, self.moe_out.ptr, stream);
        return self.moe_out.ptr;
    }

    fn rmsnorm(self: *Drafter, x: Ptr, w: Ptr, out: Ptr, n: usize, d: usize, stream: cuda.Stream) !void {
        const block = std.math.ceilPowerOfTwo(usize, d) catch return error.BadShape;
        try self.set.run(stream, "_rmsnorm", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("W", "*bf16", w),
            aot.ptr("OUT", "*bf16", out), aot.int("os_", @intCast(d)), aot.float("eps", self.dims.eps),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("PDL", 1) });
    }

    fn ropeHeads(self: *Drafter, x: Ptr, at: attention.RopeAt, n: usize, h: usize, hd: usize, inverse: bool, stream: cuda.Stream) !void {
        const hb = @min(h, 16);
        try self.set.run(stream, "_rope_heads", .{ @intCast(n), @intCast(h / hb), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("COS", "*fp32", at.cos), aot.ptr("SIN", "*fp32", at.sin), aot.ptr("POS", "*i64", at.pos),
        }, &.{ aot.ci("H", @intCast(h)), aot.ci("HD", @intCast(hd)), aot.ci("RD", @intCast(self.dims.rd)), aot.ci("INV", @intFromBool(inverse)), aot.ci("HB", @intCast(hb)), aot.ci("PDL", 1) });
    }

    /// kernels.kv_norm_rope: y [n, hd] -> out (normalized, roped, FP8 quantize-dequantized), and into ring[slots]
    /// where a slot is >= 0 (`ring_rows`: the plane's rows, a pool's slots x ring).
    fn kvNormRope(self: *Drafter, y: Ptr, w: Ptr, at: attention.RopeAt, slots: Ptr, ring: Ptr, out: Ptr, n: usize, stream: cuda.Stream) !void {
        const dm = self.dims;
        try self.set.run(stream, "_kv_norm_rope", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("Y", "*bf16", y),            aot.ptr("W", "*bf16", w),         aot.ptr("COS", "*fp32", at.cos),
            aot.ptr("SIN", "*fp32", at.sin),     aot.ptr("POS", "*i64", at.pos),   aot.ptr("OUT", "*bf16", out),
            aot.ptr("RING", "*bf16", ring),      aot.ptr("SLOT_OF", "*i64", slots), aot.int("ring_size", @intCast(dm.ring)),
            aot.float("eps", dm.eps),
        }, &.{ aot.ci("QUANT", 1), aot.ci("D", @intCast(dm.hd)), aot.ci("RD", @intCast(dm.rd)), aot.ci("PDL", 1) });
    }

    fn copyCols(self: *Drafter, src: Ptr, sld: usize, dst: Ptr, dld: usize, rows: usize, cols: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_int, @intCast(sld)));
        a.add(dst);
        a.add(@as(c_int, @intCast(dld)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(cols)));
        try cuda.launch.launch(self.glue.copy_cols, .{ .grid = .{ .x = @intCast((cols + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    // -- the pool's rings (concurrent streams) -------------------------------------------------------------------

    /// dspark.py `absorb_many` over several streams at once: the taps [n, 3 d] (row i at position `pos[i]` of slot
    /// `slots[i]`) into every stage's ring plane (row slots[i] x ring + pos[i] % ring), the GEMMs over all n rows in one
    /// pass as the fork concatenates them (a 1-row GEMM's bits are not a 4-row one's); n <= 128, each stream's rows
    /// fewer than a ring. The slots' `absorbed` are the caller's.
    pub fn absorbRows(self: *Drafter, pool: *DraftPool, taps: Ptr, pos: []const usize, slots: []const usize, rope: Rope, stream: cuda.Stream) !void {
        const dm = self.dims;
        const n = pos.len;
        if (n == 0) return;
        if (n > 128 or slots.len != n) return error.TooManyRows;
        var tab: [128]i64 = undefined;
        try self.uploads.begin(); // (waits for the last tables' copies: their staging is reused)
        for (0..n) |i| tab[i] = @intCast(pos[i]);
        try self.uploads.put(self.abs_pos, 0, std.mem.sliceAsBytes(tab[0..n]), stream);
        for (0..n) |i| tab[i] = @intCast(slots[i] * dm.ring + pos[i] % dm.ring);
        try self.uploads.put(self.abs_slots, 0, std.mem.sliceAsBytes(tab[0..n]), stream);
        const ra = try self.rr.at(rope, 1, self.abs_pos.ptr, n, stream);
        try self.head.main_proj.run(taps, .bf16, n, self.xh.ptr, self.z.ptr, self.main_y.ptr, .bf16, stream);
        try self.rmsnorm(self.main_y.ptr, self.head.main_norm, self.main_x.ptr, n, dm.d, stream);
        if (self.fused) {
            try self.stageKv(self.main_x.ptr, n, stream);
            for (self.stages, pool.rings, 0..) |s, ring, j| {
                const y = self.kvg.ptr + j * 128 * dm.hd * 2;
                try self.kvNormRope(y, s.kv_norm, ra, self.abs_slots.ptr, ring.ptr, y, n, stream);
            }
            return;
        }
        for (self.stages, pool.rings) |s, ring| {
            try s.wkv.run(self.main_x.ptr, .bf16, n, self.xh.ptr, self.z.ptr, self.abs_kv.ptr, .bf16, stream);
            try self.kvNormRope(self.abs_kv.ptr, s.kv_norm, ra, self.abs_slots.ptr, ring.ptr, self.abs_kv.ptr, n, stream);
        }
    }

    /// dspark.py `stage_kv`: every stage's wkv of main_x [m, d] (m <= 128) in one grouped launch, stage j's rows at
    /// `kvg` + j x 128 rows; each the bits of the stage's own launch.
    fn stageKv(self: *Drafter, main_x: Ptr, m: usize, stream: cuda.Stream) !void {
        const dm = self.dims;
        var xs: [exl3.gmax]exl3.Rows = undefined;
        var outs: [exl3.gmax]exl3.Rows = undefined;
        for (0..self.stages.len) |j| {
            xs[j] = .{ .ptr = main_x, .dtype = .bf16, .ld = dm.d };
            outs[j] = .{ .ptr = self.kvg.ptr + j * 128 * dm.hd * 2, .dtype = .bf16, .ld = dm.hd };
        }
        const k = self.stages.len;
        try self.kv_group.rotate(xs[0..k], m, self.xh_in.ptr, stream);
        try self.kv_group.rotated(m, self.xh_in.ptr, outs[0..k], self.zg.ptr, null, null, stream);
    }

    /// dspark.py `absorb_many` for one slot: the taps of positions start .. start + n - 1 into slot `slot`'s rows of
    /// every stage's ring plane (row = slot x ring + position % ring); the slot's `absorbed` is the caller's.
    pub fn absorbSlot(self: *Drafter, pool: *DraftPool, slot: usize, taps: Ptr, start: usize, n_in: usize, rope: Rope, stream: cuda.Stream) !void {
        const dm = self.dims;
        var n = n_in;
        var first = start;
        var tap_ptr = taps;
        if (n > dm.ring) {
            first = start + n - dm.ring;
            tap_ptr = taps + (n - dm.ring) * 3 * dm.d * 2;
            n = dm.ring;
        }
        if (n > 0) {
            if (n > self.abs_rows) return error.TooManyRows;
            var tab: [max_absorb]i64 = undefined;
            try self.uploads.begin(); // (waits for the last tables' copies: their staging is reused)
            for (0..n) |i| tab[i] = @intCast(first + i);
            try self.uploads.put(self.abs_pos, 0, std.mem.sliceAsBytes(tab[0..n]), stream);
            for (0..n) |i| tab[i] = @intCast(slot * dm.ring + (first + i) % dm.ring);
            try self.uploads.put(self.abs_slots, 0, std.mem.sliceAsBytes(tab[0..n]), stream);
            const ra = try self.rr.at(rope, 1, self.abs_pos.ptr, n, stream);
            var at: usize = 0;
            while (at < n) : (at += 128) {
                const m = @min(128, n - at);
                try self.head.main_proj.run(tap_ptr + at * 3 * dm.d * 2, .bf16, m, self.xh.ptr, self.z.ptr, self.main_y.ptr + at * dm.d * 2, .bf16, stream);
                try self.rmsnorm(self.main_y.ptr + at * dm.d * 2, self.head.main_norm, self.main_x.ptr + at * dm.d * 2, m, dm.d, stream);
            }
            if (self.fused) {
                at = 0;
                while (at < n) : (at += 128) {
                    const m = @min(128, n - at);
                    try self.stageKv(self.main_x.ptr + at * dm.d * 2, m, stream);
                    for (self.stages, pool.rings, 0..) |s, ring, j| {
                        const y = self.kvg.ptr + j * 128 * dm.hd * 2;
                        try self.kvNormRope(y, s.kv_norm, ra.from(at), self.abs_slots.ptr + at * 8, ring.ptr, y, m, stream);
                    }
                }
            } else for (self.stages, pool.rings) |s, ring| {
                at = 0;
                while (at < n) : (at += 128) {
                    const m = @min(128, n - at);
                    try s.wkv.run(self.main_x.ptr + at * dm.d * 2, .bf16, m, self.xh.ptr, self.z.ptr, self.abs_kv.ptr + at * dm.hd * 2, .bf16, stream);
                    try self.kvNormRope(self.abs_kv.ptr + at * dm.hd * 2, s.kv_norm, ra.from(at), self.abs_slots.ptr + at * 8, ring.ptr, self.abs_kv.ptr + at * dm.hd * 2, m, stream);
                }
            }
        }
        pool.absorbed[slot] = start + n_in;
    }
};

/// dspark.py `DraftPool`: `slots` drafter caches in one ring plane a stage (slot s: rows [s x ring, (s + 1) x ring)).
pub const DraftPool = struct {
    rings: []cuda.DeviceBuffer, // per stage [slots x ring, hd] bf16
    absorbed: []usize,
    slots: usize,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, stages: usize, slots: usize, dims: Dims) !DraftPool {
        const rings = try gpa.alloc(cuda.DeviceBuffer, stages);
        errdefer gpa.free(rings);
        var made: usize = 0;
        errdefer for (rings[0..made]) |*r| r.free();
        for (rings) |*r| {
            r.* = try cuda.DeviceBuffer.alloc(d, slots * dims.ring * dims.hd * 2);
            try r.fill8(0, null);
            made += 1;
        }
        const absorbed = try gpa.alloc(usize, slots);
        @memset(absorbed, 0);
        return .{ .rings = rings, .absorbed = absorbed, .slots = slots };
    }

    pub fn deinit(self: *DraftPool, gpa: std.mem.Allocator) void {
        for (self.rings) |*r| r.free();
        gpa.free(self.rings);
        gpa.free(self.absorbed);
    }
};

/// markov.py `Markov` for up to NP streams: the head rows this rank scores, no cached rows yet (every token's bias
/// row is computed into its staging row: the same bits as a cached one, markov.py's own words).
pub const Markov = struct {
    gpa: std.mem.Allocator,
    set: *const aot.Set,
    split: bool,
    world: usize,
    rank: usize,
    vocab: usize,
    n_cols: usize, // V / world: a segment
    segs: usize,
    seg0: usize,
    cols: usize, // segs * n_cols
    rank_dim: usize,
    block: usize,
    b_tiles: usize,
    s_tiles: usize,
    n_part: usize,
    bp: usize,
    out: cuda.DeviceBuffer, // i64 [NP, block + 1]
    st: cuda.DeviceBuffer, // f16 [NP, cols]: the staging rows
    pv: cuda.DeviceBuffer, // f32 [NP, n_part]
    pi: cuda.DeviceBuffer, // i32 [NP, n_part]
    send: cuda.DeviceBuffer, // f32 [NP, 4]
    recv: cuda.DeviceBuffer, // f32 [world, NP, 4]
    slot: cuda.DeviceBuffer, // i32 [V] = -1
    cache: cuda.DeviceBuffer, // f16 [1]: no cached rows

    pub const bn = 64;
    pub const sub = 4;
    pub const bc = 1024;
    pub const np = 16;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, set: *const aot.Set, dims: Dims) !Markov {
        const split = dims.markov_split and dims.world > 1;
        const n_cols = dims.vocab / dims.world;
        if (n_cols * dims.world != dims.vocab or n_cols % bn != 0) return error.BadShape;
        const segs: usize = if (split) 1 else dims.world;
        const seg0: usize = if (split) dims.rank else 0;
        const cols = segs * n_cols;
        const s_tiles = (n_cols + bc - 1) / bc;
        const n_part = s_tiles * segs;
        var self: Markov = .{
            .gpa = gpa, .set = set, .split = split, .world = dims.world, .rank = dims.rank, .vocab = dims.vocab,
            .n_cols = n_cols, .segs = segs, .seg0 = seg0, .cols = cols, .rank_dim = dims.markov_rank, .block = dims.block,
            .b_tiles = (n_cols + bn * sub - 1) / (bn * sub), .s_tiles = s_tiles, .n_part = n_part,
            .bp = std.math.ceilPowerOfTwo(usize, n_part) catch return error.BadShape,
            .out = undefined, .st = undefined, .pv = undefined, .pi = undefined, .send = undefined, .recv = undefined, .slot = undefined, .cache = undefined,
        };
        self.out = try cuda.DeviceBuffer.alloc(d, np * (dims.block + 1) * 8);
        self.st = try cuda.DeviceBuffer.alloc(d, np * cols * 2);
        self.pv = try cuda.DeviceBuffer.alloc(d, np * n_part * 4);
        self.pi = try cuda.DeviceBuffer.alloc(d, np * n_part * 4);
        self.send = try cuda.DeviceBuffer.alloc(d, np * 16);
        self.recv = try cuda.DeviceBuffer.alloc(d, dims.world * np * 16);
        self.slot = try cuda.DeviceBuffer.alloc(d, dims.vocab * 4);
        try self.slot.fill32(0xFFFFFFFF, null);
        self.cache = try cuda.DeviceBuffer.alloc(d, 16);
        try self.cache.fill8(0, null);
        return self;
    }

    pub fn deinit(self: *Markov) void {
        inline for (.{ "out", "st", "pv", "pi", "send", "recv", "slot", "cache" }) |f| @field(self, f).free();
    }

    /// `lg`: the drafter head's logits this loop reads for N streams of `block` rows (stream s's step i at row
    /// s x block + i): this rank's [R, n_cols] with split, else [world, R, n_cols]. out[s, 0] = tokens[s], out[s,
    /// 1 .. steps] the drafts (device, `self.out`, row stride block + 1).
    /// out[s, 0] = tokens[s] for the N streams, copied on `stream` ahead of `loop`'s launches.
    pub fn seed(self: *Markov, tokens: []const u32, N: usize, up: *attention.Uploads, stream: cuda.Stream) !void {
        if (N > np or tokens.len < N) return error.TooManyRows;
        const ts: usize = self.block + 1; // out.stride(0)
        for (0..N) |s_| {
            const first: i64 = tokens[s_];
            try up.put(self.out, s_ * ts * 8, std.mem.asBytes(&first), stream);
        }
    }

    /// The Markov steps' launches after `seed`: out[s, 1 .. n_steps] (the drafts).
    pub fn loop(self: *Markov, lg: Ptr, head: Head, N: usize, block: usize, n_steps: usize, gatherer: *const Drafter, stream: cuda.Stream) !void {
        if (N > np or block != self.block or n_steps > block) return error.TooManyRows;
        const ts: usize = block + 1; // out.stride(0)
        const nc = self.n_cols;
        for (0..n_steps) |i| {
            const tok = self.out.ptr + i * 8;
            try self.set.run(stream, "_bias", .{ @intCast(self.b_tiles), @intCast(self.segs), 1 }, &.{
                aot.ptr("TOK", "*i64", tok),           aot.int("t_stride", @intCast(ts)),     aot.ptr("SLOT", "*i32", self.slot.ptr),
                aot.ptr("EMB", "*bf16", head.markov_embed), aot.ptr("HEAD", "*fp16", head.markov_head), aot.ptr("OUT", "*fp16", self.st.ptr),
                aot.int("o_row", @intCast(self.cols)), aot.int("n_rows", @intCast(N)),         aot.int("n_cols", @intCast(nc)),
                aot.int("seg0", @intCast(self.seg0)),
            }, &.{ aot.ci("BN", bn), aot.ci("SUB", sub), aot.ci("RANK", @intCast(self.rank_dim)), aot.ci("NP", np), aot.ci("PDL", 1) });
            try self.set.run(stream, "_score", .{ @intCast(self.s_tiles), @intCast(self.segs), 1 }, &.{
                aot.ptr("LG", "*fp32", lg + i * nc * 4), aot.int("l_seg", @intCast(N * block * nc)), aot.int("l_row", @intCast(block * nc)),
                aot.ptr("TOK", "*i64", tok),             aot.int("t_stride", @intCast(ts)),      aot.ptr("SLOT", "*i32", self.slot.ptr),
                aot.ptr("CACHE", "*fp16", self.cache.ptr), aot.int("c_row", @intCast(self.cols)), aot.ptr("STAGE", "*fp16", self.st.ptr),
                aot.int("s_row", @intCast(self.cols)),   aot.ptr("PV", "*fp32", self.pv.ptr),    aot.ptr("PI", "*i32", self.pi.ptr),
                aot.int("n_rows", @intCast(N)),          aot.int("n_cols", @intCast(nc)),        aot.int("seg0", @intCast(self.seg0)),
                aot.int("n_part", @intCast(self.n_part)),
            }, &.{ aot.ci("BC", bc), aot.ci("PDL", 1) });
            try self.set.run(stream, "_finish", .{ @intCast(N), 1, 1 }, &.{
                aot.ptr("PV", "*fp32", self.pv.ptr), aot.ptr("PI", "*i32", self.pi.ptr), aot.int("n_part", @intCast(self.n_part)),
                aot.ptr("OUT", "*i64", self.out.ptr + (i + 1) * 8), aot.int("o_stride", @intCast(ts)), aot.ptr("SEND", "*fp32", self.send.ptr),
                aot.int("send", @intFromBool(self.split)),
            }, &.{ aot.ci("BP", @intCast(self.bp)), aot.ci("PDL", 1) });
            if (self.split) {
                try gatherer.allGather(self.send.ptr, self.recv.ptr, N * 4, stream);
                try self.set.run(stream, "_pick", .{ @intCast(N), 1, 1 }, &.{
                    aot.ptr("G", "*fp32", self.recv.ptr), aot.int("n_rows", @intCast(N)), aot.ptr("OUT", "*i64", self.out.ptr + (i + 1) * 8), aot.int("o_stride", @intCast(ts)),
                }, &.{ aot.ci("WORLD", @intCast(self.world)), aot.ci("PDL", 1) });
            }
        }
    }
};

fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn f16ToF32(h: u16) f32 {
    return @floatCast(@as(f16, @bitCast(h)));
}

test "the Markov loop's tiling follows markov.py" {
    const v: usize = 129280;
    try std.testing.expectEqual(@as(usize, 505), (v + 256 - 1) / 256); // b_tiles
    try std.testing.expectEqual(@as(usize, 127), (v + 1023) / 1024); // s_tiles
    try std.testing.expectEqual(@as(usize, 128), std.math.ceilPowerOfTwo(usize, 127) catch unreachable);
    try std.testing.expectEqual(@as(usize, 64), (v / 2 + 1023) / 1024);
}
