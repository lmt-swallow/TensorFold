//! The attention sublayer of one DeepSeek-V4.1 backbone layer at decode sizes (1 to 16 rows of one stream, this
//! rank's heads): the Python fork's graph.py `StaticDecoder._attention`, launch for launch. The EXL3 projections are
//! cuda_exl3.zig's (each `mm`), the Triton kernels come from the captured AOT set (cuda.aot: `_rmsnorm`,
//! `_rope_heads`, `_kv_norm_rope`, `_fp4qd_p2`, `_fp4_store`, `_rowmm`, `_index_score`, `_sparse_attn_part`,
//! `_sparse_attn_merge`) and the torch glue (the compressor's softmax pool, the indexer weights' scaling, column
//! block copies) from attn_glue.cu. The indexer's top-k selection (torch's total-order keys) runs on the host for
//! now. Every row's bits are its own, as in the fork: the output partial [n, d] fp32 is what rank 0 would gather.

const std = @import("std");
const cuda = @import("cuda");
const exl3 = @import("cuda_exl3.zig");

const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;

pub const Error = error{ BadShape, NoCandidateSource, OutOfMemory, TooManyRows } || cuda.Error || exl3.Error;

/// The sizes the launches read: the pack's (defaults), and the layer's ratio, sources and context bucket.
pub const Dims = struct {
    d: usize = 5120,
    heads: usize = 32, // this rank's heads (64 over 2 ranks)
    hd: usize = 512,
    rd: usize = 64,
    q_rank: usize = 1280,
    o_rank: usize = 1024,
    groups: usize = 4, // this rank's wo_a slices (8 over 2 ranks)
    window: usize = 128,
    ring: usize = 144, // window + RING_EXTRA slots
    raw: usize = 64, // compressor inputs kept by position (RAW)
    idx_heads: usize = 32,
    idx_dim: usize = 128,
    idx_topk: usize = 512,
    ratio: usize, // compress ratio of this layer (0: window attention only)
    kv_source: bool,
    index_source: bool,
    has_idx_wk: bool,
    eps: f32,
    bucket: usize, // the context bucket: the indexer scans bucket / ratio compressed entries
    max_rows: usize = 16,

    pub fn nb(self: Dims) usize {
        return if (self.ratio == 0) 0 else self.bucket / self.ratio;
    }

    pub fn kk(self: Dims) usize {
        return @min(self.idx_topk, self.nb());
    }
};

/// attn_glue.cu, loaded from a cubin (built by tools/zig/dsv41/attn_capture.sh with the fork image's nvcc).
pub const Glue = struct {
    module: cuda.Module,
    comp_pool2: cuda.Function,
    bf16_scale: cuda.Function,
    copy_cols: cuda.Function,
    f32_to_f16: cuda.Function,

    /// The fatbin built into the binary (zig/build/cuda.zig: dsv41_attn_glue).
    pub fn load(d: *const cuda.Driver) !Glue {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        return loadBytes(d, cuda.kernels.dsv41_attn_glue);
    }

    /// A cubin of attn_glue.cu read from disk (tools/zig/dsv41/attn_capture.sh writes one).
    pub fn loadBytes(d: *const cuda.Driver, cubin: []const u8) !Glue {
        var module = try cuda.Module.load(d, cubin);
        errdefer module.unload();
        return .{
            .module = module,
            .comp_pool2 = try module.function("comp_pool2"),
            .bf16_scale = try module.function("bf16_scale"),
            .copy_cols = try module.function("copy_cols"),
            .f32_to_f16 = try module.function("f32_to_f16"),
        };
    }

    pub fn deinit(self: *Glue) void {
        self.module.unload();
    }
};

/// This rank's weights of one layer's attention (weights.py `load_block`).
pub const Layer = struct {
    wq_a: exl3.Linear,
    wq_b: exl3.Linear, // this rank's head columns
    wkv: exl3.Linear,
    wo_a: [4]exl3.Linear, // this rank's slices
    wo_b: exl3.Linear, // this rank's input rows
    comp_wkv: ?exl3.Linear = null,
    comp_wgate: ?exl3.Linear = null,
    idx_wq_b: ?exl3.Linear = null,
    idx_wk: ?exl3.Linear = null,
    q_norm: cuda.DeviceBuffer, // bf16 [q_rank]
    kv_norm: cuda.DeviceBuffer, // bf16 [hd]
    sink: cuda.DeviceBuffer, // fp32 [heads]
    comp_norm: ?cuda.DeviceBuffer = null, // bf16 [hd]
    idx_k_norm: ?cuda.DeviceBuffer = null, // bf16 [idx_dim]
    idx_proj_h: ?cuda.DeviceBuffer = null, // fp16 [idx_heads, d]

    pub fn deinit(self: *Layer) void {
        self.wq_a.deinit();
        self.wq_b.deinit();
        self.wkv.deinit();
        for (&self.wo_a) |*w| w.deinit();
        self.wo_b.deinit();
        inline for (.{ "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk" }) |f| if (@field(self, f)) |*l| l.deinit();
        self.q_norm.free();
        self.kv_norm.free();
        self.sink.free();
        inline for (.{ "comp_norm", "idx_k_norm", "idx_proj_h" }) |f| if (@field(self, f)) |*b| b.free();
    }

    pub fn scratchWords(self: *const Layer, rows: usize) usize {
        return self.params().scratchWords(rows);
    }

    pub fn params(self: *const Layer) Params {
        return .{
            .wq_a = &self.wq_a,
            .wq_b = &self.wq_b,
            .wkv = &self.wkv,
            .wo_b = &self.wo_b,
            .wo_a = .{ &self.wo_a[0], &self.wo_a[1], &self.wo_a[2], &self.wo_a[3] },
            .comp_wkv = if (self.comp_wkv) |*l| l else null,
            .comp_wgate = if (self.comp_wgate) |*l| l else null,
            .idx_wq_b = if (self.idx_wq_b) |*l| l else null,
            .idx_wk = if (self.idx_wk) |*l| l else null,
            .q_norm = self.q_norm.ptr,
            .kv_norm = self.kv_norm.ptr,
            .sink = self.sink.ptr,
            .comp_norm = if (self.comp_norm) |b| b.ptr else 0,
            .idx_k_norm = if (self.idx_k_norm) |b| b.ptr else 0,
            .idx_proj_h = if (self.idx_proj_h) |b| b.ptr else 0,
        };
    }
};

/// What a forward reads of a layer: a view over an owned `Layer` or over cuda_weights' layer (the loader's), the
/// EXL3 layers by pointer and the plain tensors by address.
pub const Params = struct {
    wq_a: *const exl3.Linear,
    wq_b: *const exl3.Linear,
    wkv: *const exl3.Linear,
    wo_b: *const exl3.Linear,
    wo_a: [4]*const exl3.Linear,
    comp_wkv: ?*const exl3.Linear = null,
    comp_wgate: ?*const exl3.Linear = null,
    idx_wq_b: ?*const exl3.Linear = null,
    idx_wk: ?*const exl3.Linear = null,
    q_norm: Ptr,
    kv_norm: Ptr,
    sink: Ptr,
    comp_norm: Ptr = 0,
    idx_k_norm: Ptr = 0,
    idx_proj_h: Ptr = 0, // fp16 [idx_heads, d]

    /// fp32 words of split-K scratch the widest of these layers needs for `rows` rows.
    pub fn scratchWords(self: Params, rows: usize) usize {
        var most: usize = 0;
        most = @max(most, self.wq_a.scratchWords(rows));
        most = @max(most, self.wq_b.scratchWords(rows));
        most = @max(most, self.wkv.scratchWords(rows));
        most = @max(most, self.wo_b.scratchWords(rows));
        for (self.wo_a) |w| most = @max(most, w.scratchWords(rows));
        inline for (.{ "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk" }) |f| if (@field(self, f)) |l| {
            most = @max(most, l.scratchWords(rows));
        };
        return most;
    }
};

/// The cache planes the sublayer reads and writes: this layer's window ring and compressor inputs, and the
/// compressed and indexer planes of the layer's kv source (its own when it is one). Packed FP4 (KV_QUANT).
pub const Caches = struct {
    ring: cuda.DeviceBuffer, // bf16 [ring, hd]
    comp_codes: cuda.DeviceBuffer, // u8 [rows, hd / 2]
    comp_scales: cuda.DeviceBuffer, // u8 [rows, hd / 16] (E4M3)
    idxk_codes: cuda.DeviceBuffer, // u8 [rows, idx_dim / 2]
    idxk_scales: cuda.DeviceBuffer, // u8 [rows, idx_dim / 32] (E8M0)
    raw_kv: cuda.DeviceBuffer, // fp32 [raw, hd] (ratio > 1)
    raw_score: cuda.DeviceBuffer, // fp32 [raw, hd]
    rows: usize, // compressed rows held: cap / ratio + 2, the last one the scratch row

    pub fn deinit(self: *Caches) void {
        self.ring.free();
        self.comp_codes.free();
        self.comp_scales.free();
        self.idxk_codes.free();
        self.idxk_scales.free();
        self.raw_kv.free();
        self.raw_score.free();
    }
};

/// (cos, sin) fp32 [rows, rd / 2] of this layer's rope kind (ops.py `rope_cs`; here loaded, not computed).
pub const Rope = struct { cos: cuda.DeviceBuffer, sin: cuda.DeviceBuffer, rows: usize };

/// The picks a compressed layer attends through: its own (an index source) or the latest index layer's before it
/// (the fork's shared["topk"]): cidx [n, kk] int64 (-1: none).
pub const Picks = struct { cidx: Ptr, kk: usize };

/// The sublayer's launcher and its scratch for up to `dims.max_rows` rows.
pub const Attention = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    dims: Dims,
    set: *const aot.Set,
    glue: *const Glue,
    // per-row tables (int64), uploaded a forward
    pos: cuda.DeviceBuffer,
    slots: cuda.DeviceBuffer,
    vis: cuda.DeviceBuffer,
    first: cuda.DeviceBuffer,
    target: cuda.DeviceBuffer,
    zero: cuda.DeviceBuffer,
    // activations
    xh: cuda.DeviceBuffer, // fp16 [rows, d]: every linear's rotated input
    z: cuda.DeviceBuffer, // fp32 split-K partials
    qa: cuda.DeviceBuffer,
    qr: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    ykv: cuda.DeviceBuffer,
    kv: cuda.DeviceBuffer,
    ckv: cuda.DeviceBuffer, // fp32 [rows, hd] (bf16 at ratio 1)
    cgate: cuda.DeviceBuffer,
    pooled: cuda.DeviceBuffer,
    lat: cuda.DeviceBuffer,
    lat2: cuda.DeviceBuffer,
    kraw: cuda.DeviceBuffer,
    k: cuda.DeviceBuffer,
    iq: cuda.DeviceBuffer,
    iqq: cuda.DeviceBuffer,
    wl: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    score: cuda.DeviceBuffer,
    cidx: cuda.DeviceBuffer,
    pm: cuda.DeviceBuffer,
    pl: cuda.DeviceBuffer,
    po: cuda.DeviceBuffer,
    o: cuda.DeviceBuffer,
    og: cuda.DeviceBuffer,
    ug: cuda.DeviceBuffer,
    u: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    // host side
    tab: []i64, // 5 x max_rows
    score_h: []f32,
    keys_h: []i64,
    cidx_h: []i64,

    /// `z_words`: fp32 words of split-K scratch the widest EXL3 layer of this kind needs at `dims.max_rows` rows
    /// (`Params.scratchWords`).
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, dims: Dims, set: *const aot.Set, glue: *const Glue, z_words: usize) !Attention {
        const r = dims.max_rows;
        if (r > 16) return error.TooManyRows;
        const groups_h = dims.heads / 16;
        const nb = dims.nb();
        const kk = dims.kk();
        var self: Attention = undefined;
        self.gpa = gpa;
        self.d = d;
        self.dims = dims;
        self.set = set;
        self.glue = glue;
        self.pos = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.slots = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.vis = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.first = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.target = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.zero = try cuda.DeviceBuffer.alloc(d, 8);
        try self.zero.fill8(0, null);
        self.xh = try cuda.DeviceBuffer.alloc(d, r * dims.d * 2);
        self.z = try cuda.DeviceBuffer.alloc(d, @max(16, z_words * 4));
        self.qa = try cuda.DeviceBuffer.alloc(d, r * dims.q_rank * 2);
        self.qr = try cuda.DeviceBuffer.alloc(d, r * dims.q_rank * 2);
        self.q = try cuda.DeviceBuffer.alloc(d, r * dims.heads * dims.hd * 2);
        self.ykv = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.kv = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.ckv = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 4);
        self.cgate = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 4);
        self.pooled = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.lat = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.lat2 = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.kraw = try cuda.DeviceBuffer.alloc(d, r * dims.idx_dim * 2);
        self.k = try cuda.DeviceBuffer.alloc(d, r * dims.idx_dim * 2);
        self.iq = try cuda.DeviceBuffer.alloc(d, r * dims.idx_heads * dims.idx_dim * 2);
        self.iqq = try cuda.DeviceBuffer.alloc(d, r * dims.idx_heads * dims.idx_dim * 2);
        self.wl = try cuda.DeviceBuffer.alloc(d, r * dims.idx_heads * 4);
        self.wts = try cuda.DeviceBuffer.alloc(d, r * dims.idx_heads * 2);
        self.score = try cuda.DeviceBuffer.alloc(d, @max(16, r * nb * 4));
        self.cidx = try cuda.DeviceBuffer.alloc(d, @max(16, r * kk * 8));
        self.pm = try cuda.DeviceBuffer.alloc(d, r * groups_h * 8 * 16 * 4);
        self.pl = try cuda.DeviceBuffer.alloc(d, r * groups_h * 8 * 16 * 4);
        self.po = try cuda.DeviceBuffer.alloc(d, r * groups_h * 8 * 16 * dims.hd * 4);
        self.o = try cuda.DeviceBuffer.alloc(d, r * dims.heads * dims.hd * 2);
        self.og = try cuda.DeviceBuffer.alloc(d, r * (dims.heads * dims.hd / dims.groups) * 2);
        self.ug = try cuda.DeviceBuffer.alloc(d, r * dims.o_rank * 2);
        self.u = try cuda.DeviceBuffer.alloc(d, r * dims.groups * dims.o_rank * 2);
        self.out = try cuda.DeviceBuffer.alloc(d, r * dims.d * 4);
        self.tab = try gpa.alloc(i64, 5 * r);
        self.score_h = try gpa.alloc(f32, @max(1, r * nb));
        self.keys_h = try gpa.alloc(i64, @max(1, nb));
        self.cidx_h = try gpa.alloc(i64, @max(1, r * kk));
        return self;
    }

    pub fn deinit(self: *Attention) void {
        inline for (.{ "pos", "slots", "vis", "first", "target", "zero", "xh", "z", "qa", "qr", "q", "ykv", "kv", "ckv", "cgate", "pooled", "lat", "lat2", "kraw", "k", "iq", "iqq", "wl", "wts", "score", "cidx", "pm", "pl", "po", "o", "og", "ug", "u", "out" }) |f| @field(self, f).free();
        self.gpa.free(self.tab);
        self.gpa.free(self.score_h);
        self.gpa.free(self.keys_h);
        self.gpa.free(self.cidx_h);
    }

    /// Rows x bf16 [n, d] at positions start .. start + n - 1 -> this rank's fp32 partial [n, d] (`self.out`),
    /// the caches updated as the fork's; `self.cidx` holds the picks. Synchronous at the top-k (a host read).
    /// `picks`: the index layer's picks a layer without an indexer reads (required when the layer is compressed and
    /// not an index source); an index source makes its own (`self.picks()`).
    pub fn forward(self: *Attention, p: Params, c: *const Caches, rope: Rope, x: Ptr, n: usize, start: usize, given: ?Picks, stream: cuda.Stream) !Ptr {
        const dm = self.dims;
        if (n < 1 or n > dm.max_rows) return error.TooManyRows;
        const ratio = dm.ratio;
        // -- the row tables ---------------------------------------------------------------------------------------
        const r = dm.max_rows;
        const pos = self.tab[0..r];
        const slots = self.tab[r .. 2 * r];
        const vis = self.tab[2 * r .. 3 * r];
        const first = self.tab[3 * r .. 4 * r];
        const target = self.tab[4 * r .. 5 * r];
        const scratch_row: i64 = @intCast(c.rows - 1);
        for (0..n) |i| {
            const pv: i64 = @intCast(start + i);
            pos[i] = pv;
            slots[i] = @mod(pv, @as(i64, @intCast(dm.ring)));
            if (ratio > 0) {
                const rt: i64 = @intCast(ratio);
                vis[i] = @divFloor(pv + 1, rt);
                const g = @divFloor(pv, rt);
                first[i] = g * rt;
                target[i] = if (@mod(pv + 1, rt) == 0) g else scratch_row;
            }
        }
        try self.pos.upload(0, std.mem.sliceAsBytes(pos[0..n]));
        try self.slots.upload(0, std.mem.sliceAsBytes(slots[0..n]));
        if (ratio > 0) {
            try self.vis.upload(0, std.mem.sliceAsBytes(vis[0..n]));
            try self.first.upload(0, std.mem.sliceAsBytes(first[0..n]));
            try self.target.upload(0, std.mem.sliceAsBytes(target[0..n]));
        }
        // -- attn_in: wq_a, wkv (and the compressor's) on x ------------------------------------------------------
        try p.wq_a.run(x, .bf16, n, self.xh.ptr, self.z.ptr, self.qa.ptr, .bf16, stream);
        try p.wkv.run(x, .bf16, n, self.xh.ptr, self.z.ptr, self.ykv.ptr, .bf16, stream);
        if (ratio > 0 and dm.kv_source) {
            try p.comp_wkv.?.run(x, .bf16, n, self.xh.ptr, self.z.ptr, self.ckv.ptr, if (ratio == 1) .bf16 else .f32, stream);
            if (ratio > 1) try p.comp_wgate.?.run(x, .bf16, n, self.xh.ptr, self.z.ptr, self.cgate.ptr, .f32, stream);
        }
        // -- q: RMSNorm, wq_b, RoPE on this rank's heads ----------------------------------------------------------
        try self.rmsnorm(self.qa.ptr, p.q_norm, self.qr.ptr, n, dm.q_rank, stream);
        try p.wq_b.run(self.qr.ptr, .bf16, n, self.xh.ptr, self.z.ptr, self.q.ptr, .bf16, stream);
        try self.ropeHeads(self.q.ptr, rope, self.pos.ptr, n, dm.heads, dm.hd, false, stream);
        // -- the window KV into the ring ------------------------------------------------------------------------
        try self.set.run(stream, "_kv_norm_rope", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("Y", "*bf16", self.ykv.ptr),  aot.ptr("W", "*bf16", p.kv_norm), aot.ptr("COS", "*fp32", rope.cos.ptr),
            aot.ptr("SIN", "*fp32", rope.sin.ptr), aot.ptr("POS", "*i64", self.pos.ptr),     aot.ptr("OUT", "*bf16", self.kv.ptr),
            aot.ptr("RING", "*bf16", c.ring.ptr),  aot.ptr("SLOT_OF", "*i64", self.slots.ptr), aot.int("ring_size", @intCast(dm.ring)),
            aot.float("eps", dm.eps),
        }, &.{ aot.ci("QUANT", 1), aot.ci("D", @intCast(dm.hd)), aot.ci("RD", @intCast(dm.rd)), aot.ci("PDL", 1) });
        if (ratio > 0) {
            // -- the kv source: compressed latents (and index keys) of the rows' groups --------------------------
            if (dm.kv_source) {
                if (ratio == 1) {
                    try self.rmsnorm(self.ckv.ptr, p.comp_norm, self.lat.ptr, n, dm.hd, stream);
                } else {
                    // the rows' compressor inputs by position, then each group's softmax pool (attn_glue comp_pool2)
                    for (0..n) |i| {
                        const slot: usize = @intCast(@mod(pos[i], @as(i64, @intCast(dm.raw))));
                        try c.raw_kv.copyFrom(slot * dm.hd * 4, try self.ckv.at(i * dm.hd * 4), dm.hd * 4, stream.handle);
                        try c.raw_score.copyFrom(slot * dm.hd * 4, try self.cgate.at(i * dm.hd * 4), dm.hd * 4, stream.handle);
                    }
                    if (ratio != 2) return error.BadShape;
                    var a: cuda.Args = .{};
                    a.add(c.raw_kv.ptr);
                    a.add(c.raw_score.ptr);
                    a.add(self.first.ptr);
                    a.add(@as(c_int, @intCast(dm.raw)));
                    a.add(@as(c_int, @intCast(dm.hd)));
                    a.add(self.pooled.ptr);
                    try cuda.launch.launch(self.glue.comp_pool2, .{ .grid = .{ .x = @intCast((dm.hd + 255) / 256), .y = @intCast(n) }, .block = .{ .x = 256 } }, stream, &a);
                    try self.rmsnorm(self.pooled.ptr, p.comp_norm, self.lat.ptr, n, dm.hd, stream);
                }
                if (dm.has_idx_wk) {
                    try p.idx_wk.?.run(self.lat.ptr, .bf16, n, self.xh.ptr, self.z.ptr, self.kraw.ptr, .bf16, stream);
                    try self.rmsnorm(self.kraw.ptr, p.idx_k_norm, self.k.ptr, n, dm.idx_dim, stream);
                    try self.ropeHeads(self.k.ptr, rope, self.first.ptr, n, 1, dm.idx_dim, false, stream);
                    try self.fp4Store(self.k.ptr, dm.idx_dim, c.idxk_codes.ptr, c.idxk_scales.ptr, self.target.ptr, n, 32, false, stream);
                }
                try self.lat2.copyFrom(0, self.lat.ptr, n * dm.hd * 2, stream.handle);
                try self.ropeHeads(self.lat2.ptr, rope, self.first.ptr, n, 1, dm.hd, false, stream);
                try self.fp4Store(self.lat2.ptr, dm.hd, c.comp_codes.ptr, c.comp_scales.ptr, self.target.ptr, n, 16, true, stream);
            }
            // -- the indexer: scores over the visible compressed entries, the top-k picks -------------------------
            if (dm.index_source) {
                const nb = dm.nb();
                const kk = dm.kk();
                try p.idx_wq_b.?.run(self.qr.ptr, .bf16, n, self.xh.ptr, self.z.ptr, self.iq.ptr, .bf16, stream);
                try self.ropeHeads(self.iq.ptr, rope, self.pos.ptr, n, dm.idx_heads, dm.idx_dim, false, stream);
                const nblk: usize = n * dm.idx_heads * dm.idx_dim / 32;
                try self.set.run(stream, "_fp4qd_p2", .{ @intCast((nblk + 31) / 32), 1, 1 }, &.{
                    aot.ptr("X", "*bf16", self.iq.ptr), aot.ptr("OUT", "*bf16", self.iqq.ptr), aot.int("nblk", @intCast(nblk)),
                }, &.{ aot.ci("BLKS", 32), aot.ci("PDL", 1) });
                try self.set.run(stream, "_rowmm", .{ @intCast(n), @intCast((dm.idx_heads + 7) / 8), 1 }, &.{
                    aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(dm.d)), aot.ptr("W", "*fp16", p.idx_proj_h), aot.ptr("OUT", "*fp32", self.wl.ptr),
                }, &.{ aot.ci("K", @intCast(dm.d)), aot.ci("N", @intCast(dm.idx_heads)), aot.ci("BN", 8), aot.ci("BK", 256), aot.ci("PDL", 1) });
                {
                    const scale: f32 = @floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(dm.idx_dim))) * @sqrt(@as(f64, @floatFromInt(dm.idx_heads)))));
                    const count: c_int = @intCast(n * dm.idx_heads);
                    var a: cuda.Args = .{};
                    a.add(self.wl.ptr);
                    a.add(self.wts.ptr);
                    a.add(scale);
                    a.add(count);
                    try cuda.launch.launch(self.glue.bf16_scale, .{ .grid = .{ .x = @intCast((n * dm.idx_heads + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
                }
                const nt: usize = (nb + 63) / 64;
                try self.set.run(stream, "_index_score", .{ @intCast(n), @intCast(nt), 1 }, &.{
                    aot.ptr("Q", "*bf16", self.iqq.ptr),     aot.ptr("K", "*u8", c.idxk_codes.ptr),  aot.ptr("KS", "*u8", c.idxk_scales.ptr),
                    aot.ptr("Wt", "*bf16", self.wts.ptr),    aot.ptr("VIS", "*i64", self.vis.ptr),   aot.ptr("OUT", "*fp32", self.score.ptr),
                    aot.int("n", @intCast(nb)),              aot.ptr("KB", "*i64", self.vis.ptr),    aot.ptr("CAND", "*i64", self.vis.ptr),
                    aot.int("cs", 0),                        aot.ptr("TMAX", "*fp32", self.score.ptr), aot.int("nt", @intCast(nt)),
                }, &.{
                    aot.ci("IH", @intCast(dm.idx_heads)), aot.ci("ID", @intCast(dm.idx_dim)), aot.ci("BN", 64),      aot.ci("PACKED", 1),
                    aot.ci("HAS_BASE", 0),                aot.ci("PDL", 1),                    aot.ci("HAS_CAND", 0), aot.ci("CB", 8),
                    aot.ci("KEYS", 0),                    aot.ci("HAS_TMAX", 0),               aot.ci("SKIP_DEAD", 1), aot.ci("DEAD_PAD", 0),
                });
                try self.topk(n, nb, kk, vis[0..n], stream);
            }
        }
        // -- sparse attention over the window ring and the picks, 8 key splits a head group, then the merge --------
        {
            const groups_h = dm.heads / 16;
            const has = ratio > 0; // no compressed source: the wrapper passes the window source (bf16) for COMP and CSC
            const pk: Picks = if (dm.index_source) .{ .cidx = self.cidx.ptr, .kk = dm.kk() } else if (has) (given orelse return error.BadShape) else .{ .cidx = self.pos.ptr, .kk = 0 };
            const kk = pk.kk;
            const pick_blocks: usize = if (has) (kk + 31) / 32 else 0;
            const nblk: usize = dm.window / 32 + pick_blocks;
            const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dm.hd))));
            try self.set.run(stream, "_sparse_attn_part", .{ @intCast(n), @intCast(groups_h), 8 }, &.{
                aot.ptr("Q", "*bf16", self.q.ptr),       aot.ptr("WSRC", "*bf16", c.ring.ptr),
                aot.ptr("WLO", "*i64", self.zero.ptr),   aot.ptr("COMP", if (has) "*u8" else "*bf16", if (has) c.comp_codes.ptr else c.ring.ptr),
                aot.ptr("CSC", if (has) "*u8" else "*bf16", if (has) c.comp_scales.ptr else c.ring.ptr), aot.ptr("IDX", "*i64", if (has) pk.cidx else self.pos.ptr),
                aot.ptr("POS", "*i64", self.pos.ptr),    aot.ptr("PM", "*fp32", self.pm.ptr),
                aot.ptr("PL", "*fp32", self.pl.ptr),     aot.ptr("PO", "*fp32", self.po.ptr),
                aot.ptr("SINK", "*fp32", p.sink),  aot.ptr("OUT", "*bf16", self.o.ptr),
                aot.float("scale", scale),               aot.int("ring_size", @intCast(dm.ring)),
                aot.int("n_idx", @intCast(kk)),          aot.ptr("RBASE", "*i64", self.pos.ptr),
                aot.ptr("CBASE", "*i64", self.pos.ptr),
            }, &.{
                aot.ci("H", @intCast(dm.heads)), aot.ci("HD", @intCast(dm.hd)),     aot.ci("HB", 16),    aot.ci("WIN", @intCast(dm.window)),
                aot.ci("BN", 32),                aot.ci("RING", 1),                   aot.ci("HAS_COMP", @intFromBool(has)), aot.ci("PACKED", @intFromBool(has)),
                aot.ci("SPLITS", 8),             aot.ci("NBLK", @intCast(nblk)),      aot.ci("FINAL", 0),  aot.ci("HAS_BASE", 0),
                aot.ci("PDL", 1),                aot.ci("WPF", 1),
            });
            try self.set.run(stream, "_sparse_attn_merge", .{ @intCast(n), @intCast(groups_h), 1 }, &.{
                aot.ptr("PM", "*fp32", self.pm.ptr), aot.ptr("PL", "*fp32", self.pl.ptr), aot.ptr("PO", "*fp32", self.po.ptr),
                aot.ptr("SINK", "*fp32", p.sink), aot.ptr("OUT", "*bf16", self.o.ptr),
            }, &.{ aot.ci("H", @intCast(dm.heads)), aot.ci("HD", @intCast(dm.hd)), aot.ci("HB", 16), aot.ci("SPLITS", 8), aot.ci("PDL", 1) });
        }
        try self.ropeHeads(self.o.ptr, rope, self.pos.ptr, n, dm.heads, dm.hd, true, stream);
        // -- wo_a's slices on o's column blocks, then wo_b: this rank's fp32 partial ------------------------------
        const o_cols = dm.heads * dm.hd;
        const g_cols = o_cols / dm.groups;
        for (0..dm.groups) |g| {
            try self.copyCols(self.o.ptr + g * g_cols * 2, o_cols, self.og.ptr, g_cols, n, g_cols, stream);
            try p.wo_a[g].run(self.og.ptr, .bf16, n, self.xh.ptr, self.z.ptr, self.ug.ptr, .bf16, stream);
            try self.copyCols(self.ug.ptr, dm.o_rank, self.u.ptr + g * dm.o_rank * 2, dm.groups * dm.o_rank, n, dm.o_rank, stream);
        }
        try p.wo_b.run(self.u.ptr, .bf16, n, self.xh.ptr, self.z.ptr, self.out.ptr, .f32, stream);
        return self.out.ptr;
    }

    /// The picks this index layer made in its last forward, for the layers after it.
    pub fn picks(self: *const Attention) Picks {
        return .{ .cidx = self.cidx.ptr, .kk = self.dims.kk() };
    }

    fn rmsnorm(self: *Attention, x: Ptr, w: Ptr, out: Ptr, n: usize, d: usize, stream: cuda.Stream) !void {
        const block = std.math.ceilPowerOfTwo(usize, d) catch return error.BadShape;
        try self.set.run(stream, "_rmsnorm", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("W", "*bf16", w),
            aot.ptr("OUT", "*bf16", out), aot.int("os_", @intCast(d)), aot.float("eps", self.dims.eps),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("PDL", 1) });
    }

    /// RoPE on the last rd dims of every head of x [n, h, hd] bf16 in place, at the positions in `pos`.
    fn ropeHeads(self: *Attention, x: Ptr, rope: Rope, pos: Ptr, n: usize, h: usize, hd: usize, inverse: bool, stream: cuda.Stream) !void {
        const hb = @min(h, 16);
        try self.set.run(stream, "_rope_heads", .{ @intCast(n), @intCast(h / hb), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("COS", "*fp32", rope.cos.ptr), aot.ptr("SIN", "*fp32", rope.sin.ptr), aot.ptr("POS", "*i64", pos),
        }, &.{ aot.ci("H", @intCast(h)), aot.ci("HD", @intCast(hd)), aot.ci("RD", @intCast(self.dims.rd)), aot.ci("INV", @intFromBool(inverse)), aot.ci("HB", @intCast(hb)), aot.ci("PDL", 1) });
    }

    /// Rows x bf16 [n, d] packed as FP4 (E2M1 codes, E4M3 or E8M0 scales per `block`) into cache rows `rows`.
    fn fp4Store(self: *Attention, x: Ptr, d: usize, codes: Ptr, scales: Ptr, rows: Ptr, n: usize, block: usize, e4m3: bool, stream: cuda.Stream) !void {
        try self.set.run(stream, "_fp4_store", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("CODES", "*u8", codes), aot.ptr("SCALES", "*u8", scales), aot.ptr("ROWS", "*i64", rows),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("E4M3", @intFromBool(e4m3)), aot.ci("PDL", 1) });
    }

    fn copyCols(self: *Attention, src: Ptr, sld: usize, dst: Ptr, dld: usize, rows: usize, cols: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_int, @intCast(sld)));
        a.add(dst);
        a.add(@as(c_int, @intCast(dld)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(cols)));
        try cuda.launch.launch(self.glue.copy_cols, .{ .grid = .{ .x = @intCast((cols + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    /// kernels.topk_indices on the host: each row's kk highest scores' indices ascending (ties to the lower index,
    /// one int64 key a score: its ordered bits above, the inverted index below), -1 at or past the row's visible
    /// count (`torch.where(top < vis, top, -1)`). Reads the scores back: a stream sync.
    fn topk(self: *Attention, n: usize, nb: usize, kk: usize, vis: []const i64, stream: cuda.Stream) !void {
        try stream.synchronize();
        try self.score.download(0, std.mem.sliceAsBytes(self.score_h[0 .. n * nb]));
        for (0..n) |r| {
            const row = self.score_h[r * nb .. (r + 1) * nb];
            for (row, 0..) |s, j| self.keys_h[j] = key(s, j);
            std.sort.pdq(i64, self.keys_h[0..nb], {}, std.sort.desc(i64));
            const picked = self.cidx_h[r * kk .. (r + 1) * kk];
            for (self.keys_h[0..kk], 0..) |kv, i| picked[i] = 0xFFFFFFFF - (kv & 0xFFFFFFFF);
            std.sort.pdq(i64, picked, {}, std.sort.asc(i64));
            for (picked) |*p| if (p.* >= vis[r]) {
                p.* = -1;
            };
        }
        try self.cidx.upload(0, std.mem.sliceAsBytes(self.cidx_h[0 .. n * kk]));
    }
};

/// torch.topk's total-order key of a score at index j: the score's bits (negatives inverted, -0 as +0) above the
/// inverted index, as kernels.topk_indices builds it.
pub fn key(score: f32, j: usize) i64 {
    const s = score + 0.0;
    const bits: i32 = @bitCast(s);
    const ordered: i32 = if (bits < 0) bits ^ 0x7FFFFFFF else bits;
    return (@as(i64, ordered) << 32) | @as(i64, @intCast(0xFFFFFFFF - j));
}

test "the top-k key orders scores and breaks ties by the lower index" {
    try std.testing.expect(key(1.0, 3) > key(0.5, 0));
    try std.testing.expect(key(-0.5, 0) > key(-1.0, 0));
    try std.testing.expect(key(0.0, 1) > key(-0.0, 2)); // the same score: the lower index wins
    try std.testing.expect(key(2.0, 7) > key(2.0, 8));
    try std.testing.expect(key(-3.0, 1) < key(0.0, 500));
}
