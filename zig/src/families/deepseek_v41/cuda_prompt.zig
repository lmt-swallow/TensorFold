//! The prompt (prefill) path of one DeepSeek-V4.1 sequence: the Python fork's `Model.forward` / `_forward_k` over a
//! chunk of TF_DS_PREFILL_CHUNK rows (engine.py `prefill_steps`), on this rank, sharing a `Forward`'s weights, caches
//! and kernels (cuda_forward.zig: the decode windows). What differs from a decode window, launch for launch:
//!
//! - EXL3 projections past 128 rows take the prompt GEMM (exl3/prefill.py `matmul`: rot_in, the trellis unpacked
//!   to fp16 once a chunk, the `_gemm` Triton kernel rotating each 128-column block in its epilogue); 128 rows or
//!   fewer keep the row-invariant linear kernel (`mm` in model.py).
//! - Attention (model.py `attention_k` with `ring_mode` off): the window keys assembled from the ring and the
//!   chunk's rows (`wsrc`), the kv-source update over whole groups (`_compress`, `kv_source_update`), the indexer
//!   in row blocks of at most 64 MB of fp32 scores (model.py `rb`) with the top-k on the device (`topk_indices`,
//!   the candidate pool at the candidate source and its mask after it: cuda_attention.zig's selection; the host
//!   oracle under TF_DS_TOPK=host), one attention launch a row (no key splits), the ring written last.
//! - The MoE gate's logits and the indexer's weights through cuBLAS fp32 GEMMs (torch's `x.float() @ w.t()`), the
//!   route kernel on them, and the experts' prompt path (experts.py `routed` for 64 rows or more: member lists on the
//!   device, the gate/up `grouped_mma` with its splits folded, the down `grouped_rows`, the epilogue and the combine).
//! - DSpark taps (mean over the streams) on the tap layers, and CED bounded replay (TF_DS_REPLAY=1) as an option.
//!
//! Chunks of 16 rows or fewer are decode windows in every kernel: they go through `Forward.rows`.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const exl3 = @import("cuda_exl3.zig");
const experts = @import("cuda_experts.zig");
const hcmod = @import("cuda_hc.zig");
const attention = @import("cuda_attention.zig");
const forward = @import("cuda_forward.zig");

const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;
const Forward = forward.Forward;
const Gather = forward.Gather;

pub const Error = error{ TooManyRows, BadShape, OutOfMemory, NoCandidateSource } || cuda.Error || exl3.Error || experts.Error || cuda.cublaslt.Error;

pub const max_chunk: usize = 2048;
const decode_rows: usize = 16; // kernels.DECODE_ROWS: rows at most of a window; more is a prompt chunk
const linear_rows: usize = 128; // model.py mm: the row-invariant linear up to here, the prompt GEMM beyond
const exact_rows: usize = 64; // experts.EXACT_ROWS: the experts' prompt path from here
const prompt_bias_scale: f32 = 0.08838834764831845; // prefill.py HAD_SCALE
const score_bytes_limit: usize = 1 << 26; // the indexer's score matrix a row block (model.py rb)
const topk_prune_bn: usize = 64;

/// prompt_glue.cu.
pub const Glue = struct {
    module: cuda.Module,
    copy_rows: cuda.Function,
    pool_pairs: cuda.Function,
    bf16_to_f32: cuda.Function,
    f16_to_f32: cuda.Function,
    fill_i64: cuda.Function,
    rms_norm_torch: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Glue {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_prompt_glue);
        errdefer module.unload();
        return .{
            .module = module,
            .copy_rows = try module.function("tf_dsv41_copy_rows"),
            .pool_pairs = try module.function("tf_dsv41_pool_pairs"),
            .bf16_to_f32 = try module.function("tf_dsv41_bf16_to_f32"),
            .f16_to_f32 = try module.function("tf_dsv41_f16_to_f32"),
            .fill_i64 = try module.function("tf_dsv41_fill_i64"),
            .rms_norm_torch = try module.function("tf_dsv41_rms_norm_torch"),
        };
    }

    pub fn deinit(self: *Glue) void {
        self.module.unload();
    }

    /// ops.rms_norm (torch's expression and summation order) on bf16 rows [n, d] -> out bf16 [n, d]; d a multiple
    /// of 128.
    pub fn rmsNormTorch(self: *const Glue, x: Ptr, w: Ptr, eps: f32, n: usize, d: usize, out: Ptr, stream: cuda.Stream) !void {
        if (d % 128 != 0) return error.BadShape;
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(eps);
        a.add(@as(c_int, @intCast(d)));
        a.add(out);
        try cuda.launch.launch(self.rms_norm_torch, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 32 } }, stream, &a);
    }
};

/// experts_prompt.cu: grouped_mma_kernel<2, LO, HI, true, 4> and grouped_rows_kernel<2, 8, 4, 1, LO, HI, 2, false>
/// by K2 range, with the grouping kernels of the experts fatbin.
pub const ExpertKernels = struct {
    module: cuda.Module,
    mma: [3]cuda.Function,
    rows: [3]cuda.Function,
    group_count: cuda.Function,
    group_place: cuda.Function,
    work_list: cuda.Function,
    rot_in: cuda.Function,
    gateup_epilogue: cuda.Function,
    down_combine: cuda.Function,

    pub fn load(d: *const cuda.Driver, ek: *const experts.Kernels) !ExpertKernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_exl3_experts_prompt);
        errdefer module.unload();
        var k: ExpertKernels = undefined;
        k.module = module;
        const ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 16 } };
        for (ranges, 0..) |r, i| {
            var buf: [200]u8 = undefined;
            k.mma[i] = try module.function(try symbol(&buf, "_ZN8tf_exl3x18grouped_mma_kernelILi2ELi{d}ELi{d}ELb1ELi4EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiiiS7_", .{ r[0], r[1] }));
            k.rows[i] = try module.function(try symbol(&buf, "_ZN8tf_exl3x19grouped_rows_kernelILi2ELi8ELi4ELi1ELi{d}ELi{d}ELi2ELb0EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiiS7_", .{ r[0], r[1] }));
        }
        k.group_count = try ek.prep_mod.function("_ZN16tf_exl3x_experts18group_count_kernelEPKiPii");
        k.group_place = try ek.prep_mod.function("_ZN16tf_exl3x_experts18group_place_kernelEPKiS1_PiS2_S2_iiii");
        k.work_list = try ek.prep_mod.function("_ZN16tf_exl3x_experts16work_list_kernelEPKiS1_S1_iPiiiS2_ii");
        k.rot_in = try ek.prep_mod.function("_ZN16tf_exl3x_experts13rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iii");
        k.gateup_epilogue = try ek.prep_mod.function("_ZN16tf_exl3x_experts22gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiifi");
        k.down_combine = try ek.prep_mod.function("_ZN16tf_exl3x_experts19down_combine_kernelEPKfPKiPK6__halfPfS1_S1_S7_iiiiii");
        return k;
    }

    pub fn deinit(self: *ExpertKernels) void {
        self.module.unload();
    }
};

fn symbol(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]const u8 {
    const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args);
    buf[text.len] = 0;
    return buf[0..text.len :0];
}

/// experts.py Scratch(prompt=True) for up to `rows` rows: the member lists, the rotated inputs, the folded sums.
pub const PromptScratch = struct {
    rows: usize,
    slots: usize,
    maxu: usize,
    maxm: usize, // the member lists' stride: max(32, rows rounded up to 32)
    xg: cuda.DeviceBuffer, // fp16 [P, D]
    xu: cuda.DeviceBuffer,
    xd: cuda.DeviceBuffer, // fp16 [P, I]
    z: cuda.DeviceBuffer, // fp32 [prompt_z * P]
    no_y: cuda.DeviceBuffer, // fp32 [1]
    counts: cuda.DeviceBuffer, // int32 [E]
    ids: cuda.DeviceBuffer, // int32 [maxu]
    count: cuda.DeviceBuffer, // int32 [1]
    members: cuda.DeviceBuffer, // int32 [maxu, maxm]
    work_gu: cuda.DeviceBuffer, // int32 [len_gu, 2]
    work_d: cuda.DeviceBuffer, // int32 [len_d, 2]
    len_gu: usize,
    len_d: usize,

    pub fn init(d: *const cuda.Driver, ex: *const experts.Experts, rows: usize, slots: usize) !PromptScratch {
        const D = ex.dims;
        const I = ex.width;
        const P = rows * slots;
        const maxu = @min(P, ex.count);
        const maxm = @max(32, (rows + 31) / 32 * 32);
        const zrow = @max(2 * I + D / 2, D);
        var s: PromptScratch = undefined;
        s.rows = rows;
        s.slots = slots;
        s.maxu = maxu;
        s.maxm = maxm;
        s.len_gu = listLen(P, maxu, 64);
        s.len_d = listLen(P, maxu, 32);
        const fields = .{ "xg", "xu", "xd", "z", "no_y", "counts", "ids", "count", "members", "work_gu", "work_d" };
        const sizes = [_]usize{ P * D * 2, P * D * 2, P * I * 2, zrow * P * 4, 4, ex.count * 4, maxu * 4, 4, maxu * maxm * 4, s.len_gu * 8, s.len_d * 8 };
        var made: usize = 0;
        errdefer {
            inline for (fields, 0..) |f, i| if (i < made) @field(s, f).free();
        }
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try cuda.DeviceBuffer.alloc(d, @max(4, sizes[i]));
            made += 1;
            try @field(s, f).fill32(0, null);
        }
        return s;
    }

    pub fn deinit(self: *PromptScratch) void {
        inline for (.{ "xg", "xu", "xd", "z", "no_y", "counts", "ids", "count", "members", "work_gu", "work_d" }) |f| @field(self, f).free();
    }
};

/// experts.py _list_len: a work list's length for P member slots over at most maxu places in groups of `rows`.
pub fn listLen(P: usize, maxu: usize, rows: usize) usize {
    return (P + maxu * (rows - 1)) / rows;
}

/// cuBLAS fp32 GEMM as torch runs `x.float() @ w.float().t()`: out [m, n] = x [m, k] . w [n, k]^T, fp32 in and out,
/// fp32 compute (no TF32), the heuristic's first algorithm for the shape.
pub const Gemm32 = struct {
    lt: cuda.cublaslt.Library,
    handle: cuda.cublaslt.Handle,
    workspace: cuda.DeviceBuffer,

    pub const workspace_bytes: usize = 32 << 20;

    pub fn init(d: *const cuda.Driver) !Gemm32 {
        var lt = try cuda.cublaslt.Library.open();
        errdefer lt.close();
        var handle: cuda.cublaslt.Handle = null;
        try lt.check(lt.api.cublasLtCreate(&handle), "cublasLtCreate");
        errdefer _ = lt.api.cublasLtDestroy(handle);
        const workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes);
        return .{ .lt = lt, .handle = handle, .workspace = workspace };
    }

    pub fn deinit(self: *Gemm32) void {
        self.workspace.free();
        _ = self.lt.api.cublasLtDestroy(self.handle);
        self.lt.close();
    }

    pub fn run(self: *Gemm32, x: Ptr, w: Ptr, out: Ptr, m: usize, n: usize, k: usize, stream: cuda.Stream) !void {
        const lt = &self.lt;
        const api = lt.api;
        var desc: cuda.cublaslt.MatmulDesc = null;
        try lt.check(api.cublasLtMatmulDescCreate(&desc, cuda.cublaslt.compute_32f, .f32), "cublasLtMatmulDescCreate");
        defer _ = api.cublasLtMatmulDescDestroy(desc);
        const ta: c_int = @intFromEnum(cuda.cublaslt.Op.t);
        const tb: c_int = @intFromEnum(cuda.cublaslt.Op.n);
        try lt.check(api.cublasLtMatmulDescSetAttribute(desc, cuda.cublaslt.desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(api.cublasLtMatmulDescSetAttribute(desc, cuda.cublaslt.desc_transb, &tb, @sizeOf(c_int)), "set transb");
        // column-major view (torch's call): out^T [n, m] = w [n, k] (stored k x n, transposed) . x^T [k, m]
        var la: cuda.cublaslt.Layout = null;
        var lb: cuda.cublaslt.Layout = null;
        var lc: cuda.cublaslt.Layout = null;
        try lt.check(api.cublasLtMatrixLayoutCreate(&la, .f32, k, n, @intCast(k)), "layout W");
        defer _ = api.cublasLtMatrixLayoutDestroy(la);
        try lt.check(api.cublasLtMatrixLayoutCreate(&lb, .f32, k, m, @intCast(k)), "layout X");
        defer _ = api.cublasLtMatrixLayoutDestroy(lb);
        try lt.check(api.cublasLtMatrixLayoutCreate(&lc, .f32, n, m, @intCast(n)), "layout OUT");
        defer _ = api.cublasLtMatrixLayoutDestroy(lc);
        var pref: cuda.cublaslt.Preference = null;
        try lt.check(api.cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        defer _ = api.cublasLtMatmulPreferenceDestroy(pref);
        const limit: u64 = workspace_bytes;
        try lt.check(api.cublasLtMatmulPreferenceSetAttribute(pref, cuda.cublaslt.pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        var found: [1]cuda.cublaslt.Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(api.cublasLtMatmulAlgoGetHeuristic(self.handle, desc, la, lb, lc, lc, pref, 1, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        const one: f32 = 1;
        const zero: f32 = 0;
        try lt.check(api.cublasLtMatmul(self.handle, desc, &one, w, la, x, lb, &zero, out, lc, out, lc, &found[0].algo, self.workspace.ptr, workspace_bytes, stream.handle), "cublasLtMatmul");
    }
};

/// The taps of a chunk's rows for the drafter: [rows, taps * d] bf16, a column block a tap layer.
pub const Taps = struct { buf: Ptr, stride: usize };

pub const Options = struct {
    /// CED bounded replay (engine.py prefill_steps with TF_DS_REPLAY=1): layers from n_layers / 2 run only rows at
    /// positions >= replay (max(0, prompt length - window)); null: every row through every layer.
    replay: ?usize = null,
    taps: ?Taps = null,
    /// The head on the chunk's last row (the fork runs it every chunk; only the prompt's last chunk's is read).
    logits: bool = true,
};

/// A chunk's timing (`Prompt.enableProfile`, TF_DS_PROFILE=1 in the served engine): CUDA events between every
/// layer's sublayer groups, read back after the chunk (a host sync a chunk, so chunks no longer overlap), and the
/// host's time inside the chunk call. `chunk` logs one `info:` line a profiled chunk.
pub const Profile = struct {
    pub const groups = [_][]const u8{ "engram", "attention", "attention gather", "moe", "moe gather" };
    /// [1 + layers * groups.len + 1]: before the embedding, after each layer's groups, after the head.
    marks: []cuda.Event,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, layers: usize) !Profile {
        const marks = try gpa.alloc(cuda.Event, 2 + layers * groups.len);
        errdefer gpa.free(marks);
        var made: usize = 0;
        errdefer for (marks[0..made]) |*e| e.deinit();
        for (marks) |*e| {
            e.* = try cuda.Event.init(d, true);
            made += 1;
        }
        return .{ .marks = marks, .gpa = gpa };
    }

    pub fn deinit(self: *Profile) void {
        for (self.marks) |*e| e.deinit();
        self.gpa.free(self.marks);
    }
};

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

pub const Prompt = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    fwd: *Forward,
    /// Per-chunk timing when set (`enableProfile`).
    profile: ?Profile = null,
    kernels: *const exl3.Kernels,
    glue: Glue,
    pk: ExpertKernels,
    gemm: Gemm32,
    rows: usize, // the chunk rows this holds buffers for (at most max_chunk)
    had: cuda.DeviceBuffer, // the 128 x 128 Hadamard bf16 (prefill.py Workspace.hadamard)
    // the chunk's streams and mixes
    ids: cuda.DeviceBuffer,
    h: cuda.DeviceBuffer,
    pre: cuda.DeviceBuffer,
    pre_a: cuda.DeviceBuffer,
    pre_f: cuda.DeviceBuffer,
    post: cuda.DeviceBuffer,
    comb: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer,
    gathered: cuda.DeviceBuffer,
    // the prompt GEMM's workspace
    xh: cuda.DeviceBuffer, // fp16 [rows, d]: every linear's rotated input
    wq: cuda.DeviceBuffer, // fp16 [K, N] of the widest unpacked projection
    z: cuda.DeviceBuffer, // fp32 split-K scratch of the row-invariant linears up to 128 rows
    // attention
    pos: cuda.DeviceBuffer, // int64 [rows]
    vis: cuda.DeviceBuffer,
    vis_blk: cuda.DeviceBuffer, // a row block's vis rows, 16-aligned as the fork's `vis[r0:r1].contiguous()`
    gpos: cuda.DeviceBuffer, // int64 [rows + 1]: the groups' positions (group * ratio) of a kv source
    grows: cuda.DeviceBuffer, // int64 [rows + 1]: the groups' cache rows
    src_idx: cuda.DeviceBuffer, // int64 [ring]: ring rows gathered into wsrc
    dst_idx: cuda.DeviceBuffer, // int64 [ring]: ring rows written from the chunk
    wlo: cuda.DeviceBuffer, // int64 [1]
    qa: cuda.DeviceBuffer,
    qr: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    ykv: cuda.DeviceBuffer,
    wsrc: cuda.DeviceBuffer, // bf16 [window - 1 + rows, hd]
    ckv: cuda.DeviceBuffer, // fp32 [rows + 1, hd] (a pending row before the chunk's)
    cgate: cuda.DeviceBuffer,
    pooled: cuda.DeviceBuffer, // bf16 [rows + 1, hd]
    lat: cuda.DeviceBuffer,
    lat2: cuda.DeviceBuffer,
    kraw: cuda.DeviceBuffer,
    k: cuda.DeviceBuffer,
    iq: cuda.DeviceBuffer,
    iqq: cuda.DeviceBuffer,
    x32: cuda.DeviceBuffer, // fp32 [rows, d]: x.float() for the cuBLAS matmuls
    wl: cuda.DeviceBuffer, // fp32 [rows, idx_heads]
    wts: cuda.DeviceBuffer, // bf16 [rows, idx_heads]
    score: cuda.DeviceBuffer, // fp32 scores or int64 keys [rb, n_comp]
    tmax: cuda.DeviceBuffer, // int64 [rb, nt]: the keyed path's tile maxima (unread)
    cidx: cuda.DeviceBuffer, // int64 [rows, idx_topk]
    cand: cuda.DeviceBuffer, // u8 [rows, cand cols]: the candidate pool of the chunk's rows (device selection)
    smax: cuda.DeviceBuffer, // fp32 [rb, cand cols]: a row block's block maxima
    o: cuda.DeviceBuffer,
    og: cuda.DeviceBuffer,
    ug: cuda.DeviceBuffer,
    u: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer, // fp32 [rows, d]: the attention partial
    // MoE
    gate_w32: []cuda.DeviceBuffer, // per layer: the gate weight as fp32 (gate_w.float())
    logits: cuda.DeviceBuffer, // fp32 [rows, E]
    pick: cuda.DeviceBuffer,
    wts_moe: cuda.DeviceBuffer,
    moe: cuda.DeviceBuffer, // fp32 [rows, d]
    scratch: PromptScratch,
    // the head's
    xc: cuda.DeviceBuffer,
    local: cuda.DeviceBuffer,
    glog: cuda.DeviceBuffer,
    head_logits: cuda.DeviceBuffer,
    // host
    ids_h: []i64,
    tab: []i64,
    score_h: []f32,
    keys_h: []i64,
    cidx_h: []i64,
    cand_h: []u8, // the candidate pool's block mask [rows, cand_blocks_of(n_comp)] from the candidate source
    cand_cols: usize,
    cand_rows: usize,
    // the shared pick list (model.py shared["topk"]): its columns as the last index layer made them, and the row
    // stride `cidx` holds them at now (the columns, or padded to a multiple of 16 for sparse_attn)
    cidx_cols: usize,
    cidx_stride: usize,
    kv_layer: usize, // shared["kv_layer"]
    /// The rows the last chunk ran through every layer: positions top_start .. top_start + top_rows - 1 (the
    /// chunk's, or under bounded replay those at or past `replay`; none for an encoder-only chunk). The taps hold
    /// these rows (model.py `taps_start`).
    top_start: usize = 0,
    top_rows: usize = 0,

    /// Buffers for chunks of up to `rows` rows (at most max_chunk), sharing `fwd`'s weights and caches.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, fwd: *Forward, kernels: *const exl3.Kernels, rows: usize) !Prompt {
        if (rows < 1 or rows > max_chunk) return error.TooManyRows;
        const c = fwd.cfg;
        const dm = c.dim;
        const w = fwd.w;
        const heads = c.n_heads / fwd.world;
        const slots = c.topk + 1;
        var self: Prompt = undefined;
        self.gpa = gpa;
        self.d = d;
        self.fwd = fwd;
        self.profile = null;
        self.kernels = kernels;
        self.rows = rows;
        self.glue = try Glue.load(d);
        errdefer self.glue.deinit();
        self.pk = try ExpertKernels.load(d, fwd.ek);
        errdefer self.pk.deinit();
        self.gemm = try Gemm32.init(d);
        errdefer self.gemm.deinit();
        // the Hadamard: 1 - 2 * parity(i & j), bf16 (exact)
        var hh: [128 * 128]u16 = undefined;
        for (0..128) |i| for (0..128) |j| {
            const p: u32 = @popCount(@as(u32, @intCast(i & j))) & 1;
            hh[i * 128 + j] = if (p == 0) 0x3F80 else 0xBF80;
        };
        self.had = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&hh));
        const r = rows;
        self.ids = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.h = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * dm * 2);
        self.pre = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * 4);
        self.pre_a = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * 4);
        self.pre_f = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * 4);
        self.post = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * 4);
        self.comb = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * hcmod.hc_streams * 4);
        self.part = try cuda.DeviceBuffer.alloc(d, hcmod.Hc.partWords(r) * 4);
        self.x = try cuda.DeviceBuffer.alloc(d, r * dm * 2);
        self.gathered = try cuda.DeviceBuffer.alloc(d, fwd.world * r * dm * 4);
        // the widest projection's K x N and the linears' split-K scratch at 128 rows
        var kn: usize = 0;
        var zw: usize = 0;
        for (w.layers) |*l| {
            inline for (.{ "wq_a", "wq_b", "wkv", "wo_b" }) |f| {
                kn = @max(kn, @field(l, f).k * @field(l, f).n);
                zw = @max(zw, @field(l, f).scratchWords(linear_rows));
            }
            for (l.wo_a[0..l.wo_a_count]) |*x| {
                kn = @max(kn, x.k * x.n);
                zw = @max(zw, x.scratchWords(linear_rows));
            }
            inline for (.{ "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk", "engram_wkv" }) |f| if (@field(l, f)) |*x| {
                kn = @max(kn, x.k * x.n);
                zw = @max(zw, x.scratchWords(linear_rows));
            };
        }
        zw = @max(zw, w.head.scratchWords(1));
        const kmax: usize = @max(dm, heads * c.head_dim); // the widest input (wo_b's u)
        self.xh = try cuda.DeviceBuffer.alloc(d, r * kmax * 2);
        self.wq = try cuda.DeviceBuffer.alloc(d, @max(16, kn * 2));
        self.z = try cuda.DeviceBuffer.alloc(d, @max(16, zw * 4));
        self.pos = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.vis = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.vis_blk = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.gpos = try cuda.DeviceBuffer.alloc(d, (r + 1) * 8);
        self.grows = try cuda.DeviceBuffer.alloc(d, (r + 1) * 8);
        self.src_idx = try cuda.DeviceBuffer.alloc(d, (c.window + 16) * 8);
        self.dst_idx = try cuda.DeviceBuffer.alloc(d, (c.window + 16) * 8);
        self.wlo = try cuda.DeviceBuffer.alloc(d, 8);
        self.qa = try cuda.DeviceBuffer.alloc(d, r * c.q_rank * 2);
        self.qr = try cuda.DeviceBuffer.alloc(d, r * c.q_rank * 2);
        self.q = try cuda.DeviceBuffer.alloc(d, r * heads * c.head_dim * 2);
        self.ykv = try cuda.DeviceBuffer.alloc(d, r * c.head_dim * 2);
        self.wsrc = try cuda.DeviceBuffer.alloc(d, (c.window - 1 + r) * c.head_dim * 2);
        self.ckv = try cuda.DeviceBuffer.alloc(d, (r + 1) * c.head_dim * 4);
        self.cgate = try cuda.DeviceBuffer.alloc(d, (r + 1) * c.head_dim * 4);
        self.pooled = try cuda.DeviceBuffer.alloc(d, (r + 1) * c.head_dim * 2);
        self.lat = try cuda.DeviceBuffer.alloc(d, (r + 1) * c.head_dim * 2);
        self.lat2 = try cuda.DeviceBuffer.alloc(d, (r + 1) * c.head_dim * 2);
        self.kraw = try cuda.DeviceBuffer.alloc(d, (r + 1) * c.idx_dim * 2);
        self.k = try cuda.DeviceBuffer.alloc(d, (r + 1) * c.idx_dim * 2);
        self.iq = try cuda.DeviceBuffer.alloc(d, r * c.idx_heads * c.idx_dim * 2);
        self.iqq = try cuda.DeviceBuffer.alloc(d, r * c.idx_heads * c.idx_dim * 2);
        self.x32 = try cuda.DeviceBuffer.alloc(d, r * dm * 4);
        self.wl = try cuda.DeviceBuffer.alloc(d, r * c.idx_heads * 4);
        self.wts = try cuda.DeviceBuffer.alloc(d, r * c.idx_heads * 2);
        const nb_max: usize = fwd.cap + 2;
        const score_words: usize = @max(score_bytes_limit / 4, 16 * nb_max);
        self.score = try cuda.DeviceBuffer.alloc(d, score_words * 8);
        self.tmax = try cuda.DeviceBuffer.alloc(d, @max(16, r * ((nb_max + 63) / 64) * 8));
        self.cidx = try cuda.DeviceBuffer.alloc(d, @max(16, r * (c.idx_topk + 16) * 8));
        const cb = @max(1, c.cand_block);
        const cand_cols_max = (nb_max + cb - 1) / cb;
        self.cand = try cuda.DeviceBuffer.alloc(d, @max(16, r * cand_cols_max));
        self.smax = try cuda.DeviceBuffer.alloc(d, @max(16, (score_words / cb + r) * 4));
        self.o = try cuda.DeviceBuffer.alloc(d, r * heads * c.head_dim * 2);
        const groups = c.o_groups / fwd.world;
        self.og = try cuda.DeviceBuffer.alloc(d, r * (heads * c.head_dim / groups) * 2);
        self.ug = try cuda.DeviceBuffer.alloc(d, r * c.o_rank * 2);
        self.u = try cuda.DeviceBuffer.alloc(d, r * groups * c.o_rank * 2);
        self.out = try cuda.DeviceBuffer.alloc(d, r * dm * 4);
        self.gate_w32 = try gpa.alloc(cuda.DeviceBuffer, w.layers.len);
        self.logits = try cuda.DeviceBuffer.alloc(d, r * c.n_routed * 4);
        self.pick = try cuda.DeviceBuffer.alloc(d, r * slots * 4);
        self.wts_moe = try cuda.DeviceBuffer.alloc(d, r * slots * 4);
        self.moe = try cuda.DeviceBuffer.alloc(d, r * dm * 4);
        self.scratch = try PromptScratch.init(d, &w.layers[0].experts, r, slots);
        self.xc = try cuda.DeviceBuffer.alloc(d, dm * 2);
        const vh = w.vocab[1] - w.vocab[0];
        self.local = try cuda.DeviceBuffer.alloc(d, vh * 4);
        self.glog = try cuda.DeviceBuffer.alloc(d, fwd.world * vh * 4);
        self.head_logits = try cuda.DeviceBuffer.alloc(d, c.vocab * 4);
        self.ids_h = try gpa.alloc(i64, r);
        self.tab = try gpa.alloc(i64, 6 * (r + 1));
        self.score_h = try gpa.alloc(f32, score_words * 2);
        self.keys_h = try gpa.alloc(i64, @max(1, nb_max));
        self.cidx_h = try gpa.alloc(i64, @max(1, r * (c.idx_topk + 16)));
        self.cand_cols = (nb_max + c.cand_block - 1) / @max(1, c.cand_block);
        self.cand_h = try gpa.alloc(u8, @max(1, r * self.cand_cols));
        self.cand_rows = 0;
        self.cidx_cols = 0;
        self.cidx_stride = 0;
        self.kv_layer = 0;
        self.top_start = 0;
        self.top_rows = 0;
        // the gate weights as fp32 (torch's gate_w.float(), once)
        var stream = try cuda.Stream.init(d, true);
        defer stream.deinit();
        for (w.layers, 0..) |*l, li| {
            self.gate_w32[li] = try cuda.DeviceBuffer.alloc(d, l.gate_w.count * 4);
            try self.castF16(l.gate_w.ptr(), self.gate_w32[li].ptr, l.gate_w.count, stream);
        }
        // every setup copy and fill so far (this one's, and the weights' and caches' before it) ran on the legacy
        // stream, which the chunks' non-blocking stream does not wait for: they complete before the first chunk
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        return self;
    }

    /// Times every chunk from now on (see `Profile`).
    pub fn enableProfile(self: *Prompt) !void {
        if (self.profile == null) self.profile = try Profile.init(self.gpa, self.d, self.fwd.w.layers.len);
    }

    pub fn deinit(self: *Prompt) void {
        if (self.profile) |*p| p.deinit();
        for (self.gate_w32) |*b| b.free();
        self.gpa.free(self.gate_w32);
        inline for (.{ "had", "ids", "h", "pre", "pre_a", "pre_f", "post", "comb", "part", "x", "gathered", "xh", "wq", "z", "pos", "vis", "vis_blk", "gpos", "grows", "src_idx", "dst_idx", "wlo", "qa", "qr", "q", "ykv", "wsrc", "ckv", "cgate", "pooled", "lat", "lat2", "kraw", "k", "iq", "iqq", "x32", "wl", "wts", "score", "tmax", "cidx", "cand", "smax", "o", "og", "ug", "u", "out", "logits", "pick", "wts_moe", "moe", "xc", "local", "glog", "head_logits" }) |f| @field(self, f).free();
        self.scratch.deinit();
        self.gemm.deinit();
        self.pk.deinit();
        self.glue.deinit();
        self.gpa.free(self.ids_h);
        self.gpa.free(self.tab);
        self.gpa.free(self.score_h);
        self.gpa.free(self.keys_h);
        self.gpa.free(self.cidx_h);
        self.gpa.free(self.cand_h);
    }

    // -- the chunk -----------------------------------------------------------------------------------------------

    /// The prompt's ids at positions start .. start + n - 1 into the caches; returns the last row's logits [1, V]
    /// f32 (null for a replay chunk that holds no decoder rows, or without `o.logits`). Chunks of 16 rows or fewer go
    /// through `Forward.rows` (the window kernels), as the fork's: under replay too once the chunk starts at or past
    /// `replay` (the window kernels read no floor, and the kv source's update runs on the same rows either way).
    pub fn chunk(self: *Prompt, gather: Gather, ids: []const i64, start: usize, o: Options, stream: cuda.Stream) !?Ptr {
        const n = ids.len;
        if (n < 1 or n > self.rows) return error.TooManyRows;
        const whole = if (o.replay) |r| r <= start else true;
        if (n <= decode_rows and whole and o.taps == null and o.logits) {
            self.top_start = start;
            self.top_rows = n;
            return try self.fwd.rows(gather, ids, start, n, false, stream);
        }
        const t0 = nowNs();
        var engram_ns: i128 = 0;
        try self.mark(0, stream);
        const out = try self.layers(gather, ids, start, o, stream, &engram_ns);
        try self.mark(1 + self.fwd.w.layers.len * Profile.groups.len, stream);
        if (self.profile != null and o.replay == null) try self.report(start, n, nowNs() - t0, engram_ns, stream);
        return out;
    }

    fn mark(self: *const Prompt, i: usize, stream: cuda.Stream) !void {
        if (self.profile) |p| try p.marks[i].record(stream);
    }

    /// The profiled chunk's line: the GPU's ms by sublayer group (summed over the layers), the head's, and the host's
    /// ms in the call (and in the Engram hook: hashing, row reads, uploads).
    fn report(self: *const Prompt, start: usize, n: usize, host_ns: i128, engram_ns: i128, stream: cuda.Stream) !void {
        const p = self.profile.?;
        const L = self.fwd.w.layers.len;
        const ng = Profile.groups.len;
        const t_sync = nowNs();
        try stream.synchronize();
        const wait_ms = @as(f64, @floatFromInt(nowNs() - t_sync)) / 1e6;
        var sum: [Profile.groups.len]f64 = @splat(0);
        for (0..L * ng) |i| sum[i % ng] += try cuda.Event.elapsedMs(p.marks[i], p.marks[i + 1]);
        const head = try cuda.Event.elapsedMs(p.marks[L * ng], p.marks[L * ng + 1]);
        const total = try cuda.Event.elapsedMs(p.marks[0], p.marks[L * ng + 1]);
        std.log.info("deepseek_v41 rank {d}: chunk at {d} ({d} rows): GPU {d:.0} ms (engram {d:.0}, attention {d:.0}, attention gather {d:.0}, moe {d:.0}, moe gather {d:.0}, head {d:.1}); host {d:.0} ms in the call ({d:.0} in Engram), then {d:.0} ms waiting for the GPU", .{
            self.fwd.w.rank, start, n, total, sum[0], sum[1], sum[2], sum[3], sum[4], head,
            @as(f64, @floatFromInt(host_ns)) / 1e6, @as(f64, @floatFromInt(engram_ns)) / 1e6, wait_ms,
        });
    }

    fn layers(self: *Prompt, gather: Gather, ids: []const i64, start: usize, o: Options, stream: cuda.Stream, engram_ns: *i128) !?Ptr {
        const n = ids.len;
        const c = self.fwd.cfg;
        const dm = c.dim;
        const w = self.fwd.w;
        const fwd = self.fwd;
        @memcpy(self.ids_h[0..n], ids);
        // every host table goes up in stream order: a layer rewrites the tables (positions, gather and scatter rows,
        // picks) that launches still queued from the layer before read, and a plain upload (the legacy stream) is not
        // ordered after a non-blocking stream's work
        try self.ids.uploadAsync(0, std.mem.sliceAsBytes(self.ids_h[0..n]), stream.handle);
        try fwd.hc.embedInit(stream, w.embed.ptr(), self.ids.ptr, self.h.ptr, self.pre.ptr, @intCast(n));
        var h = self.h.ptr;
        var pre = self.pre.ptr;
        var pre_f = self.pre_f.ptr;
        var rows = n;
        var at = start;
        var floor: usize = 0;
        var kv_done = false;
        var nb: [32]u8 = undefined;
        self.cidx_cols = 0;
        self.cidx_stride = 0;
        self.cand_rows = 0;
        self.top_start = start;
        self.top_rows = n;
        // the window layers' picks (shared["topk"]) once replay leaves 16 rows or fewer: `attendWindow`
        var window_picks: ?attention.Picks = null;
        for (w.layers, 0..) |*l, li| {
            if (o.replay) |replay| if (li == c.n_layers / 2) {
                // CED bounded replay: this layer's kv source over every row, then only the rows at or past `replay`
                try fwd.hc.pre(stream, h, l.hc_attn_fn.ptr(), l.hc_attn_scale.ptr(), l.hc_attn_base.ptr(), pre, l.attn_norm.ptr(), self.x.ptr, self.pre_a.ptr, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(rows));
                try self.kvSourceUpdate(li, self.x.ptr, rows, at, stream);
                const first = @max(at, replay) - at;
                if (first >= rows) {
                    self.top_rows = 0;
                    return null;
                }
                h += first * hcmod.hc_streams * dm * 2;
                pre += first * hcmod.hc_streams * 4;
                at += first;
                rows -= first;
                floor = replay;
                kv_done = true;
                self.top_start = at;
                self.top_rows = rows;
            };
            const m0 = 1 + li * Profile.groups.len;
            if (l.engram_wkv != null) if (fwd.engram) |e| {
                const te = nowNs();
                try e.call(e.ctx, li, h, at, rows, .{ .ctx = self, .call = mmF32 }, gather, stream);
                engram_ns.* += nowNs() - te;
                try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_engram", .{li}), h, rows * hcmod.hc_streams * dm * 2, stream);
            };
            if (o.taps) |t| if (c.dspark_taps.indexOf(@intCast(li))) |ti| {
                try fwd.hc.tap(stream, h, t.buf + ti * dm * 2, @intCast(t.stride), @intCast(rows));
            };
            try self.mark(m0, stream);
            try fwd.hc.pre(stream, h, l.hc_attn_fn.ptr(), l.hc_attn_scale.ptr(), l.hc_attn_base.ptr(), pre, l.attn_norm.ptr(), self.x.ptr, self.pre_a.ptr, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(rows));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_attn", .{li}), self.x.ptr, rows * dm * 2, stream);
            // 16 rows or fewer past a replay cut run model.py attention_k's ring mode: the window kernels
            const pa = if (kv_done and rows <= decode_rows)
                try self.attendWindow(li, self.x.ptr, rows, at, li == c.n_layers / 2, &window_picks, stream)
            else
                try self.attend(li, self.x.ptr, rows, at, floor, kv_done and li == c.n_layers / 2, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_attn.r0", .{li}), pa, rows * dm * 4, stream);
            try self.mark(m0 + 1, stream);
            try gather.run(pa, rows * dm, self.gathered.ptr, stream);
            try fwd.hc.post(stream, self.gathered.ptr, fwd.world, h, self.post.ptr, self.comb.ptr, h, @intCast(rows));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_attn", .{li}), h, rows * hcmod.hc_streams * dm * 2, stream);
            try self.mark(m0 + 2, stream);
            try fwd.hc.pre(stream, h, l.hc_ffn_fn.ptr(), l.hc_ffn_scale.ptr(), l.hc_ffn_base.ptr(), self.pre_a.ptr, l.ffn_norm.ptr(), self.x.ptr, pre_f, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(rows));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_ffn", .{li}), self.x.ptr, rows * dm * 2, stream);
            const pm = try self.moeRows(li, self.x.ptr, rows, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_moe.r0", .{li}), pm, rows * dm * 4, stream);
            try self.mark(m0 + 3, stream);
            try gather.run(pm, rows * dm, self.gathered.ptr, stream);
            try fwd.hc.post(stream, self.gathered.ptr, fwd.world, h, self.post.ptr, self.comb.ptr, h, @intCast(rows));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{li}), h, rows * hcmod.hc_streams * dm * 2, stream);
            try self.mark(m0 + 4, stream);
            const t = pre;
            pre = pre_f;
            pre_f = t;
        }
        // the head on the last row
        if (!o.logits) return null;
        const h0 = h + (rows - 1) * hcmod.hc_streams * dm * 2;
        const pre0 = pre + (rows - 1) * hcmod.hc_streams * 4;
        try fwd.hc.collapseNorm(stream, h0, pre0, w.norm.ptr(), self.xc.ptr, 1);
        try self.probeOut("head.xc", self.xc.ptr, dm * 2, stream);
        const vh = w.vocab[1] - w.vocab[0];
        try w.head.run(self.xc.ptr, .bf16, 1, self.xh.ptr, self.z.ptr, self.local.ptr, .f32, stream);
        try self.probeOut("head.local.r0", self.local.ptr, vh * 4, stream);
        try gather.run(self.local.ptr, vh, self.glog.ptr, stream);
        for (0..fwd.world) |rank| try self.head_logits.copyFrom(rank * vh * 4, self.glog.ptr + rank * vh * 4, vh * 4, stream.handle);
        return self.head_logits.ptr;
    }

    fn probeOut(self: *const Prompt, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) !void {
        if (self.fwd.probe) |p| try p.call(p.ctx, name, ptr, bytes, stream);
    }

    // -- EXL3 projections at chunk rows ---------------------------------------------------------------------------

    /// y [m, N] (y_dtype) = x [m, K] bf16 @ W: the row-invariant linear up to 128 rows, the prompt GEMM beyond
    /// (model.py `mm`).
    fn mm(self: *Prompt, lin: *const exl3.Linear, x: Ptr, m: usize, y: Ptr, y_dtype: exl3.DType, stream: cuda.Stream) !void {
        if (m <= linear_rows) return lin.run(x, .bf16, m, self.xh.ptr, self.z.ptr, y, y_dtype, stream);
        const K = lin.k;
        const N = lin.n;
        {
            // rot_in: xh [m, K] fp16 = the rows rotated by suh (linear.cu rot_in_kernel)
            var a: cuda.Args = .{};
            a.add(x);
            a.add(@intFromEnum(exl3.DType.bf16));
            a.add(lin.suh.ptr);
            a.add(self.xh.ptr);
            a.add(@as(c_int, @intCast(K)));
            try cuda.launch.launch(lin.rot_in, .{ .grid = .{ .x = @intCast((K / 128 + 3) / 4), .y = @intCast(m) }, .block = .{ .x = 128 } }, stream, &a);
        }
        {
            // unpack: W_q [K, N] fp16 from the strips
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
        try self.fwd.set.run(stream, "_gemm", .{ grid, 1, 1 }, &.{
            aot.ptr("X", "*fp16", self.xh.ptr),     aot.ptr("W", "*fp16", self.wq.ptr),   aot.ptr("H", "*bf16", self.had.ptr),
            aot.ptr("SVH", "*fp16", lin.svh.ptr),   aot.ptr("BIAS", "*fp16", lin.svh.ptr), aot.ptr("OUT", if (y_dtype == .f32) "*fp32" else "*bf16", y),
            aot.int("M", @intCast(m)),              aot.int("o_stride", @intCast(N)),
        }, &.{ aot.ci("K", @intCast(K)), aot.ci("N", @intCast(N)), aot.ci("BM", 128), aot.ci("BK", 32), aot.ci("GROUP", 8), aot.ci("HAS_BIAS", 0), aot.cf("SCALE", prompt_bias_scale) });
    }

    /// `mm` to fp32 for a hook (Engram's projection at the chunk's rows).
    fn mmF32(ctx: *anyopaque, lin: *const exl3.Linear, x: Ptr, m: usize, y: Ptr, stream: cuda.Stream) anyerror!void {
        const self: *Prompt = @ptrCast(@alignCast(ctx));
        try self.mm(lin, x, m, y, .f32, stream);
    }

    // -- attention at chunk rows ----------------------------------------------------------------------------------

    fn attend(self: *Prompt, li: usize, x: Ptr, n: usize, start: usize, floor: usize, kv_done: bool, stream: cuda.Stream) !Ptr {
        var nb: [32]u8 = undefined;
        const fwd = self.fwd;
        const c = fwd.cfg;
        const p = self.paramsOf(li);
        const caches = fwd.cachesOf(li);
        const dims = self.dimsOf(li);
        const rope = if (c.ratio(li) > 0) fwd.rope_c else fwd.rope_w;
        const hd = c.head_dim;
        const rd = c.rope_dim;
        const heads = dims.heads;
        const ratio = c.ratio(li);
        const R = dims.ring;
        // the row tables
        const pos = self.tab[0..n];
        const vis = self.tab[self.rows + 1 .. self.rows + 1 + n];
        for (0..n) |i| {
            pos[i] = @intCast(start + i);
            vis[i] = if (ratio > 0) @divFloor(pos[i] + 1, @as(i64, @intCast(ratio))) else 0;
        }
        try self.pos.uploadAsync(0, std.mem.sliceAsBytes(pos), stream.handle);
        if (ratio > 0) try self.vis.uploadAsync(0, std.mem.sliceAsBytes(vis), stream.handle);
        // attn_in: wq_a, wkv (the compressor's in kvSourceUpdate)
        try self.mm(p.wq_a, x, n, self.qa.ptr, .bf16, stream);
        try self.mm(p.wkv, x, n, self.ykv.ptr, .bf16, stream);
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.qa.r0", .{li}), self.qa.ptr, n * c.q_rank * 2, stream);
        try self.rmsnorm(self.qa.ptr, p.q_norm, self.qr.ptr, n, c.q_rank, stream);
        try self.mm(p.wq_b, self.qr.ptr, n, self.q.ptr, .bf16, stream);
        try self.ropeHeads(self.q.ptr, rope, self.pos.ptr, n, heads, hd, false, stream);
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.q.r0", .{li}), self.q.ptr, n * heads * hd * 2, stream);
        // the window keys: the ring's rows before start (from lo), then the chunk's normed, roped, quantized rows
        const lo: usize = @max(floor, if (start + 1 > c.window) start + 1 - c.window else 0);
        const before = start - lo;
        if (before > 0) {
            const idx = self.tab[2 * (self.rows + 1) .. 2 * (self.rows + 1) + before];
            for (0..before) |i| idx[i] = @intCast((lo + i) % R);
            try self.src_idx.uploadAsync(0, std.mem.sliceAsBytes(idx), stream.handle);
            try self.copyRows(caches.ring.ptr, self.src_idx.ptr, self.wsrc.ptr, 0, before, hd * 2 / 4, stream);
        }
        const kv = self.wsrc.ptr + before * hd * 2;
        try self.fillI64(self.grows.ptr, -1, n, stream); // no ring slot: the chunk's keys land in wsrc, the ring last
        try fwd.set.run(stream, "_kv_norm_rope", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("Y", "*bf16", self.ykv.ptr),  aot.ptr("W", "*bf16", p.kv_norm), aot.ptr("COS", "*fp32", rope.cos.ptr),
            aot.ptr("SIN", "*fp32", rope.sin.ptr), aot.ptr("POS", "*i64", self.pos.ptr), aot.ptr("OUT", "*bf16", kv),
            aot.ptr("RING", "*bf16", caches.ring.ptr), aot.ptr("SLOT_OF", "*i64", self.grows.ptr), aot.int("ring_size", @intCast(R)),
            aot.float("eps", c.eps),
        }, &.{ aot.ci("QUANT", 1), aot.ci("D", @intCast(hd)), aot.ci("RD", @intCast(rd)), aot.ci("PDL", 1) });
        var has = false;
        var n_idx: usize = 0;
        if (ratio > 0) {
            if (p.comp_wkv != null and !kv_done) try self.kvSourceUpdate(li, x, n, start, stream);
            const n_comp_end = (start + n) / ratio;
            if (p.idx_wq_b != null) {
                if (n_comp_end == 0) {
                    self.cidx_cols = 0;
                    self.cidx_stride = 0;
                } else {
                    try self.index(li, x, n, n_comp_end, stream);
                }
            }
            n_idx = self.cidx_cols;
            has = n_idx > 0;
        }
        // the picks padded with -1 to a multiple of 16 entries (sparse_attn at prompt rows)
        var n_pad = n_idx;
        if (has and n_idx % 16 != 0) n_pad = n_idx + 16 - n_idx % 16;
        if (has and self.cidx_stride != n_pad) try self.padCidx(n, n_idx, n_pad, stream);
        var picks: usize = 0;
        if (has) {
            picks = (n_pad + 31) / 32;
            picks = std.math.ceilPowerOfTwo(usize, picks) catch return error.BadShape;
        }
        const nblk: usize = c.window / 32 + picks;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
        const wlo_v: [1]i64 = .{@intCast(lo)};
        try self.wlo.uploadAsync(0, std.mem.sliceAsBytes(&wlo_v), stream.handle);
        const srcp = if (ratio > 0) fwd.planes[fwd.source[li]] else null;
        try fwd.set.run(stream, "_sparse_attn_part", .{ @intCast(n), @intCast(heads / 16), 1 }, &.{
            aot.ptr("Q", "*bf16", self.q.ptr),       aot.ptr("WSRC", "*bf16", self.wsrc.ptr),
            aot.ptr("WLO", "*i64", self.wlo.ptr),    aot.ptr("COMP", if (has) "*u8" else "*bf16", if (has) srcp.?.comp_codes.ptr else self.wsrc.ptr),
            aot.ptr("CSC", if (has) "*u8" else "*bf16", if (has) srcp.?.comp_scales.ptr else self.wsrc.ptr), aot.ptr("IDX", "*i64", if (has) self.cidx.ptr else self.pos.ptr),
            aot.ptr("POS", "*i64", self.pos.ptr),    aot.ptr("PM", "*bf16", self.o.ptr),
            aot.ptr("PL", "*bf16", self.o.ptr),      aot.ptr("PO", "*bf16", self.o.ptr),
            aot.ptr("SINK", "*fp32", p.sink),        aot.ptr("OUT", "*bf16", self.o.ptr),
            aot.float("scale", scale),               aot.int("ring_size", 16),
            aot.int("n_idx", @intCast(n_pad)),       aot.ptr("RBASE", "*i64", self.pos.ptr),
            aot.ptr("CBASE", "*i64", self.pos.ptr),
        }, &.{
            aot.ci("H", @intCast(heads)), aot.ci("HD", @intCast(hd)), aot.ci("HB", 16), aot.ci("WIN", @intCast(c.window)),
            aot.ci("BN", 32), aot.ci("RING", 0), aot.ci("HAS_COMP", @intFromBool(has)), aot.ci("PACKED", @intFromBool(has)),
            aot.ci("SPLITS", 1), aot.ci("NBLK", @intCast(nblk)), aot.ci("FINAL", 1), aot.ci("HAS_BASE", 0),
            aot.ci("PDL", 1), aot.ci("WPF", 0),
        });
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.wsrc.r0", .{li}), self.wsrc.ptr, (before + n) * hd * 2, stream);
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.o_raw.r0", .{li}), self.o.ptr, n * heads * hd * 2, stream);
        try self.ropeHeads(self.o.ptr, rope, self.pos.ptr, n, heads, hd, true, stream);
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.o.r0", .{li}), self.o.ptr, n * heads * hd * 2, stream);
        // the ring gets the chunk's last rows (positions start + n - keep ..)
        {
            const keep = @min(n, R);
            const idx = self.tab[3 * (self.rows + 1) .. 3 * (self.rows + 1) + keep];
            for (0..keep) |i| idx[i] = @intCast((start + n - keep + i) % R);
            try self.dst_idx.uploadAsync(0, std.mem.sliceAsBytes(idx), stream.handle);
            try self.copyRows(kv + (n - keep) * hd * 2, 0, caches.ring.ptr, self.dst_idx.ptr, keep, hd * 2 / 4, stream);
        }
        // wo_a's slices on o's column blocks, then wo_b
        const o_cols = heads * hd;
        const g_cols = o_cols / dims.groups;
        for (0..dims.groups) |g| {
            try self.copyCols(self.o.ptr + g * g_cols * 2, o_cols, self.og.ptr, g_cols, n, g_cols, stream);
            try self.mm(p.wo_a[g], self.og.ptr, n, self.ug.ptr, .bf16, stream);
            try self.copyCols(self.ug.ptr, c.o_rank, self.u.ptr + g * c.o_rank * 2, dims.groups * c.o_rank, n, c.o_rank, stream);
        }
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.u.r0", .{li}), self.u.ptr, n * dims.groups * c.o_rank * 2, stream);
        try self.mm(p.wo_b, self.u.ptr, n, self.out.ptr, .f32, stream);
        return self.out.ptr;
    }

    /// model.py attention_k at 16 rows or fewer (ring mode, as a decode window: the window keys from the ring, no
    /// floor) through the decode layer's `Attention`; `kv_done`: the kv source's update already ran for these rows
    /// (the replay layer), so the window's own is skipped. `picks`: the latest index layer's, carried between layers.
    fn attendWindow(self: *Prompt, li: usize, x: Ptr, n: usize, start: usize, kv_done: bool, picks: *?attention.Picks, stream: cuda.Stream) !Ptr {
        const fwd = self.fwd;
        const c = fwd.cfg;
        const caches = fwd.cachesOf(li);
        const rope = if (c.ratio(li) > 0) fwd.rope_c else fwd.rope_w;
        const a = &fwd.attn[li];
        const source = a.dims.kv_source;
        if (kv_done) a.dims.kv_source = false;
        defer a.dims.kv_source = source;
        const pa = try a.forward(self.paramsOf(li), &caches, rope, x, n, start, picks.*, stream);
        if (c.isIndexSource(li)) picks.* = a.picks();
        return pa;
    }

    /// model.py kv_source_update over whole groups: the compressed latents and index keys of the groups the chunk
    /// completes (ratio 2: pairs, the pending row from the compressor's positional store), into the planes.
    fn kvSourceUpdate(self: *Prompt, li: usize, x: Ptr, n: usize, start: usize, stream: cuda.Stream) !void {
        const fwd = self.fwd;
        const c = fwd.cfg;
        const p = self.paramsOf(li);
        const caches = fwd.cachesOf(li);
        const dims = self.dimsOf(li);
        const rope = fwd.rope_c;
        const hd = c.head_dim;
        const ratio = c.ratio(li);
        self.kv_layer = li;
        var full: usize = 0;
        var first_group: usize = 0;
        if (ratio == 1) {
            try self.mm(p.comp_wkv.?, x, n, self.pooled.ptr, .bf16, stream);
            full = n;
            first_group = start;
        } else {
            if (ratio != 2) return error.BadShape;
            const pend = start % ratio;
            // the chunk's compressor inputs after the pending row (from the positional store)
            try self.mm(p.comp_wkv.?, x, n, self.ckv.ptr + pend * hd * 4, .f32, stream);
            try self.mm(p.comp_wgate.?, x, n, self.cgate.ptr + pend * hd * 4, .f32, stream);
            if (pend > 0) {
                const prev: [1]i64 = .{@intCast((start - 1) % dims.raw)};
                try self.src_idx.uploadAsync(0, std.mem.sliceAsBytes(&prev), stream.handle);
                try self.copyRows(caches.raw_kv.ptr, self.src_idx.ptr, self.ckv.ptr, 0, 1, hd, stream);
                try self.copyRows(caches.raw_score.ptr, self.src_idx.ptr, self.cgate.ptr, 0, 1, hd, stream);
            }
            // the store keeps the chunk's last rows by position
            const keep = @min(n, dims.raw);
            const idx = self.tab[3 * (self.rows + 1) .. 3 * (self.rows + 1) + keep];
            for (0..keep) |i| idx[i] = @intCast((start + n - keep + i) % dims.raw);
            try self.dst_idx.uploadAsync(0, std.mem.sliceAsBytes(idx), stream.handle);
            try self.copyRows(self.ckv.ptr + (pend + n - keep) * hd * 4, 0, caches.raw_kv.ptr, self.dst_idx.ptr, keep, hd, stream);
            try self.copyRows(self.cgate.ptr + (pend + n - keep) * hd * 4, 0, caches.raw_score.ptr, self.dst_idx.ptr, keep, hd, stream);
            full = (pend + n) / ratio;
            first_group = (start - pend) / ratio;
            if (full == 0) return;
            var a: cuda.Args = .{};
            a.add(self.ckv.ptr);
            a.add(self.cgate.ptr);
            a.add(@as(c_int, @intCast(full)));
            a.add(@as(c_int, @intCast(hd)));
            a.add(self.pooled.ptr);
            try cuda.launch.launch(self.glue.pool_pairs, .{ .grid = .{ .x = @intCast((hd + 255) / 256), .y = @intCast(full) }, .block = .{ .x = 256 } }, stream, &a);
        }
        // model.py _compress and kv_source_update normalize with ops.rms_norm (torch), not the Triton kernel
        try self.glue.rmsNormTorch(self.pooled.ptr, p.comp_norm, c.eps, full, hd, self.lat.ptr, stream);
        // the groups' positions (group * ratio) and cache rows
        const gpos = self.tab[4 * (self.rows + 1) .. 5 * (self.rows + 1)];
        const grows = self.tab[5 * (self.rows + 1) .. 6 * (self.rows + 1)];
        for (0..full) |g| {
            gpos[g] = @intCast((first_group + g) * ratio);
            grows[g] = @intCast(first_group + g);
        }
        try self.gpos.uploadAsync(0, std.mem.sliceAsBytes(gpos[0..full]), stream.handle);
        try self.grows.uploadAsync(0, std.mem.sliceAsBytes(grows[0..full]), stream.handle);
        if (dims.has_idx_wk) {
            try self.mm(p.idx_wk.?, self.lat.ptr, full, self.kraw.ptr, .bf16, stream);
            try self.glue.rmsNormTorch(self.kraw.ptr, p.idx_k_norm, c.eps, full, c.idx_dim, self.k.ptr, stream);
            try self.ropeHeads(self.k.ptr, rope, self.gpos.ptr, full, 1, c.idx_dim, false, stream);
            try self.fp4Store(self.k.ptr, c.idx_dim, caches.idxk_codes.ptr, caches.idxk_scales.ptr, self.grows.ptr, full, 32, false, stream);
        }
        try self.lat2.copyFrom(0, self.lat.ptr, full * hd * 2, stream.handle);
        try self.ropeHeads(self.lat2.ptr, rope, self.gpos.ptr, full, 1, hd, false, stream);
        try self.fp4Store(self.lat2.ptr, hd, caches.comp_codes.ptr, caches.comp_scales.ptr, self.grows.ptr, full, 16, true, stream);
    }

    fn dimsOf(self: *const Prompt, li: usize) attention.Dims {
        const c = self.fwd.cfg;
        return .{
            .d = c.dim,
            .heads = c.n_heads / self.fwd.world,
            .hd = c.head_dim,
            .rd = c.rope_dim,
            .q_rank = c.q_rank,
            .o_rank = c.o_rank,
            .groups = c.o_groups / self.fwd.world,
            .window = c.window,
            .ring = c.window + 16,
            .raw = 64,
            .idx_heads = c.idx_heads,
            .idx_dim = c.idx_dim,
            .idx_topk = c.idx_topk,
            .ratio = c.ratio(li),
            .kv_source = c.isKvSource(li),
            .index_source = c.isIndexSource(li),
            .has_idx_wk = c.isKvSource(li) and c.isIndexSource(li),
            .eps = c.eps,
            .bucket = self.fwd.bucket,
            .max_rows = self.fwd.max_rows,
            .cand = attention.CandRole.of(c.cand_source, li, c.isIndexSource(li) and c.ratio(li) > 0),
            .cand_blocks = c.cand_blocks,
            .cand_block = @max(1, c.cand_block),
        };
    }

    fn paramsOf(self: *const Prompt, li: usize) attention.Params {
        const l = &self.fwd.w.layers[li];
        return .{
            .wq_a = &l.wq_a,
            .wq_b = &l.wq_b,
            .wkv = &l.wkv,
            .wo_b = &l.wo_b,
            .wo_a = .{ &l.wo_a[0], &l.wo_a[1], &l.wo_a[2], &l.wo_a[3] },
            .comp_wkv = if (l.comp_wkv) |*x| x else null,
            .comp_wgate = if (l.comp_wgate) |*x| x else null,
            .idx_wq_b = if (l.idx_wq_b) |*x| x else null,
            .idx_wk = if (l.idx_wk) |*x| x else null,
            .q_norm = l.q_norm.ptr(),
            .kv_norm = l.kv_norm.ptr(),
            .sink = l.sink.ptr(),
            .comp_norm = if (l.comp_norm) |pl| pl.ptr() else 0,
            .idx_k_norm = if (l.idx_k_norm) |pl| pl.ptr() else 0,
            .idx_proj_h = self.fwd.idx_proj_h[li].ptr,
        };
    }

    /// The indexer at chunk rows: the picks `cidx` [n, kk] (-1 past the row's visible count) from fp32 scores over
    /// the compressed entries of the kv source, the candidate pool kept at the candidate source and applied after it.
    fn index(self: *Prompt, li: usize, x: Ptr, n: usize, n_comp_end: usize, stream: cuda.Stream) !void {
        const fwd = self.fwd;
        const c = fwd.cfg;
        const p = self.paramsOf(li);
        const caches = fwd.cachesOf(li);
        const rope = fwd.rope_c;
        const kk = @min(c.idx_topk, n_comp_end);
        try self.mm(p.idx_wq_b.?, self.qr.ptr, n, self.iq.ptr, .bf16, stream);
        try self.ropeHeads(self.iq.ptr, rope, self.pos.ptr, n, c.idx_heads, c.idx_dim, false, stream);
        const nblk: usize = n * c.idx_heads * c.idx_dim / 32;
        try fwd.set.run(stream, "_fp4qd_p2", .{ @intCast((nblk + 31) / 32), 1, 1 }, &.{
            aot.ptr("X", "*bf16", self.iq.ptr), aot.ptr("OUT", "*bf16", self.iqq.ptr), aot.int("nblk", @intCast(nblk)),
        }, &.{ aot.ci("BLKS", 32), aot.ci("PDL", 1) });
        // the weights: x.float() @ idx_proj.t() (cuBLAS), to bf16, scaled
        try self.castBf16(x, self.x32.ptr, n * c.dim, stream);
        try self.gemm.run(self.x32.ptr, fwd.w.layers[li].idx_proj.?.ptr(), self.wl.ptr, n, c.idx_heads, c.dim, stream);
        {
            const scale: f32 = @floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(c.idx_dim))) * @sqrt(@as(f64, @floatFromInt(c.idx_heads)))));
            var a: cuda.Args = .{};
            a.add(self.wl.ptr);
            a.add(self.wts.ptr);
            a.add(scale);
            a.add(@as(c_int, @intCast(n * c.idx_heads)));
            try cuda.launch.launch(fwd.glue.bf16_scale, .{ .grid = .{ .x = @intCast((n * c.idx_heads + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
        }
        // scores in row blocks (the [rb, n_comp] matrix bounded), the top-k of each on the device (or the host)
        const rb = @max(16, @min(n, score_bytes_limit / (4 * @max(n_comp_end, 1))));
        const nt: usize = (n_comp_end + 63) / 64;
        const vis = self.tab[self.rows + 1 .. self.rows + 1 + n];
        const is_cand_source = c.cand_source >= 0 and li == @as(usize, @intCast(c.cand_source));
        const after_cand = c.cand_source >= 0 and @as(usize, @intCast(c.cand_source)) < li;
        if (is_cand_source) {
            self.cand_cols = (n_comp_end + c.cand_block - 1) / c.cand_block;
            self.cand_rows = n;
        } else if (after_cand and self.cand_rows != n) return error.NoCandidateSource;
        const host = fwd.glue.hostSelect();
        // (switch "prompt_keys") layers before the candidate source with a power-of-two k: the score kernel writes
        // topk_indices' int64 keys (and tile maxima) instead of scores, and the selection reads them. It must be
        // that variant: the KEYS and the score variants are compiled apart and need not sum the heads in one order
        const keyed = !is_cand_source and !after_cand and (kk & (kk - 1)) == 0;
        const cb = @max(1, c.cand_block);
        const ccols = (n_comp_end + cb - 1) / cb;
        var r0: usize = 0;
        while (r0 < n) : (r0 += rb) {
            const r1 = @min(n, r0 + rb);
            const rows = r1 - r0;
            const q = self.iqq.ptr + r0 * c.idx_heads * c.idx_dim * 2;
            const wt = self.wts.ptr + r0 * c.idx_heads * 2;
            // the block's vis rows copied to an aligned buffer: the captured kernels were specialized on a 16-aligned VIS
            try self.vis_blk.copyFrom(0, self.vis.ptr + r0 * 8, rows * 8, stream.handle);
            const vr = self.vis_blk.ptr;
            if (rows <= decode_rows) {
                // kernels.index_score: a block of decode rows takes one row a program (the same kernel as a window)
                try fwd.set.run(stream, "_index_score", .{ @intCast(rows), @intCast(nt), 1 }, &.{
                    aot.ptr("Q", "*bf16", q),                     aot.ptr("K", "*u8", caches.idxk_codes.ptr),  aot.ptr("KS", "*u8", caches.idxk_scales.ptr),
                    aot.ptr("Wt", "*bf16", wt),                   aot.ptr("VIS", "*i64", vr),                  aot.ptr("OUT", if (keyed) "*i64" else "*fp32", self.score.ptr),
                    aot.int("n", @intCast(n_comp_end)),           aot.ptr("KB", "*i64", vr),                   aot.ptr("CAND", "*i64", vr),
                    aot.int("cs", 0),                             aot.ptr("TMAX", if (keyed) "*i64" else "*fp32", if (keyed) self.tmax.ptr else self.score.ptr), aot.int("nt", @intCast(nt)),
                }, &.{
                    aot.ci("IH", @intCast(c.idx_heads)), aot.ci("ID", @intCast(c.idx_dim)), aot.ci("BN", 64),      aot.ci("PACKED", 1),
                    aot.ci("HAS_BASE", 0),                aot.ci("PDL", 1),                  aot.ci("HAS_CAND", 0), aot.ci("CB", 8),
                    aot.ci("KEYS", @intFromBool(keyed)),  aot.ci("HAS_TMAX", @intFromBool(keyed)), aot.ci("SKIP_DEAD", 1), aot.ci("DEAD_PAD", 0),
                });
            } else {
                try fwd.set.run(stream, "_index_score_rows", .{ @intCast((rows + 7) / 8), @intCast(nt), 1 }, &.{
                    aot.ptr("Q", "*bf16", q),                     aot.ptr("K", "*u8", caches.idxk_codes.ptr),
                    aot.ptr("KS", "*u8", caches.idxk_scales.ptr), aot.ptr("Wt", "*bf16", wt),
                    aot.ptr("VIS", "*i64", vr),                   aot.ptr("OUT", if (keyed) "*i64" else "*fp32", self.score.ptr),
                    aot.int("n", @intCast(n_comp_end)),           aot.int("rows", @intCast(rows)),
                    aot.ptr("TMAX", if (keyed) "*i64" else "*fp32", if (keyed) self.tmax.ptr else self.score.ptr), aot.int("nt", @intCast(nt)),
                }, &.{ aot.ci("IH", @intCast(c.idx_heads)), aot.ci("ID", @intCast(c.idx_dim)), aot.ci("BN", 64), aot.ci("RB", 8), aot.ci("PACKED", 1), aot.ci("KEYS", @intFromBool(keyed)), aot.ci("HAS_TMAX", @intFromBool(keyed)) });
            }
            if (!host) {
                const tg = &fwd.glue.topk.?;
                const mask = self.cand.ptr + r0 * ccols;
                if (is_cand_source) try tg.candidates(self.score.ptr, n_comp_end, rows, n_comp_end, vr, c.cand_blocks, cb, self.smax.ptr, mask, ccols, stream);
                if (after_cand) try tg.apply(self.score.ptr, n_comp_end, rows, n_comp_end, mask, ccols, cb, stream);
                if (keyed) {
                    try tg.topkKeys(self.score.ptr, n_comp_end, rows, n_comp_end, kk, vr, self.cidx.ptr + r0 * kk * 8, kk, stream);
                } else {
                    try tg.topk(self.score.ptr, n_comp_end, rows, n_comp_end, kk, vr, self.cidx.ptr + r0 * kk * 8, kk, stream);
                }
                continue;
            }
            try stream.synchronize();
            if (keyed) {
                const keys = std.mem.bytesAsSlice(i64, @as([]align(8) u8, @alignCast(std.mem.sliceAsBytes(self.score_h[0 .. rows * n_comp_end * 2]))));
                try self.score.download(0, std.mem.sliceAsBytes(keys));
                for (0..rows) |i| {
                    const r = r0 + i;
                    attention.hostTopkKeys(keys[i * n_comp_end .. (i + 1) * n_comp_end], kk, vis[r], self.cidx_h[r * kk .. (r + 1) * kk]);
                }
            } else {
                try self.score.download(0, std.mem.sliceAsBytes(self.score_h[0 .. rows * n_comp_end]));
                for (0..rows) |i| {
                    const r = r0 + i;
                    const row = self.score_h[i * n_comp_end .. (i + 1) * n_comp_end];
                    const mask = self.cand_h[r * self.cand_cols .. (r + 1) * self.cand_cols];
                    if (is_cand_source) attention.hostCandidates(row, vis[r], c.cand_blocks, cb, self.keys_h, mask);
                    if (after_cand) attention.hostApplyCandidates(row, mask, cb);
                    attention.hostTopk(row, kk, vis[r], self.keys_h, self.cidx_h[r * kk .. (r + 1) * kk]);
                }
            }
        }
        if (host) try self.cidx.uploadAsync(0, std.mem.sliceAsBytes(self.cidx_h[0 .. n * kk]), stream.handle);
        self.cidx_cols = kk;
        self.cidx_stride = kk;
        var nbuf: [32]u8 = undefined;
        try self.probeOut(try std.fmt.bufPrint(&nbuf, "L{d}.cidx", .{li}), self.cidx.ptr, n * kk * 8, stream);
    }

    /// cidx [n, n_idx] -> [n, n_pad] with -1 past n_idx (torch.nn.functional.pad): rows moved from the back, in place
    /// on the host copy too, so once a chunk (the layers after the index layer read the padded list as it is).
    fn padCidx(self: *Prompt, n: usize, n_idx: usize, n_pad: usize, stream: cuda.Stream) !void {
        try stream.synchronize();
        if (!self.fwd.glue.hostSelect() and self.cidx_stride == n_idx) try self.cidx.download(0, std.mem.sliceAsBytes(self.cidx_h[0 .. n * n_idx]));
        var r: usize = n;
        while (r > 0) {
            r -= 1;
            const src = self.cidx_h[r * n_idx .. (r + 1) * n_idx];
            const dst = self.cidx_h[r * n_pad .. (r + 1) * n_pad];
            std.mem.copyBackwards(i64, dst[0..n_idx], src);
            @memset(dst[n_idx..], -1);
        }
        try self.cidx.uploadAsync(0, std.mem.sliceAsBytes(self.cidx_h[0 .. n * n_pad]), stream.handle);
        self.cidx_stride = n_pad;
    }

    // -- the MoE at chunk rows -------------------------------------------------------------------------------------

    /// model.py moe at chunk rows: the gate's logits through cuBLAS (x.float() @ gate_w.float().t()), the route
    /// kernel, then experts.py routed() for 64 rows or more (the decode path below that).
    fn moeRows(self: *Prompt, li: usize, x: Ptr, n: usize, stream: cuda.Stream) !Ptr {
        const fwd = self.fwd;
        const c = fwd.cfg;
        const l = &fwd.w.layers[li];
        const E = c.n_routed;
        const slots = c.topk + 1;
        if (n <= decode_rows) {
            try fwd.route.gate(stream, x, l.gate_w.ptr(), fwd.parts.ptr, @intCast(n));
            try fwd.route.pick(stream, fwd.parts.ptr, l.gate_b.ptr(), self.pick.ptr, self.wts_moe.ptr, @intCast(n));
        } else {
            try self.castBf16(x, self.x32.ptr, n * c.dim, stream);
            try self.gemm.run(self.x32.ptr, self.gate_w32[li].ptr, self.logits.ptr, n, E, c.dim, stream);
            const nb = std.math.ceilPowerOfTwo(usize, E) catch return error.BadShape;
            const sp = std.math.ceilPowerOfTwo(usize, slots) catch return error.BadShape;
            try fwd.set.run(stream, "_route", .{ @intCast(n), 1, 1 }, &.{
                aot.ptr("L", "*fp32", self.logits.ptr), aot.ptr("BIAS", "*fp32", l.gate_b.ptr()), aot.ptr("PICK", "*i32", self.pick.ptr),
                aot.ptr("WTS", "*fp32", self.wts_moe.ptr), aot.float("scale", c.route_scale),    aot.int("shared_id", @intCast(l.experts.count - 1)),
            }, &.{ aot.ci("NE", @intCast(E)), aot.ci("NB", @intCast(nb)), aot.ci("TOPK", @intCast(c.topk)), aot.ci("SLOTS", @intCast(slots)), aot.ci("SP", @intCast(sp)), aot.ci("PDL", 1) });
        }
        if (n < exact_rows) {
            try experts.routed(fwd.ek, &l.experts, &fwd.scratch, x, self.pick.ptr, self.wts_moe.ptr, n, c.swiglu_limit, self.moe.ptr, stream);
            return self.moe.ptr;
        }
        try self.routedPrompt(&l.experts, x, n, c.swiglu_limit, stream);
        return self.moe.ptr;
    }

    /// experts.py routed() for a prompt chunk (the fork's defaults): member lists and work lists on the device, the
    /// rotated inputs, gate/up on grouped_mma (splits folded), the epilogue, down on grouped_rows, the combine.
    fn routedPrompt(self: *Prompt, ex: *const experts.Experts, x: Ptr, R: usize, limit: f32, stream: cuda.Stream) !void {
        const s = &self.scratch;
        const D = ex.dims;
        const I = ex.width;
        const E = ex.count;
        const slots = s.slots;
        const P = R * slots;
        const maxu = s.maxu;
        const maxm = @max(32, (R + 31) / 32 * 32);
        if (R > s.rows or maxm > s.maxm) return error.TooManyRows;
        const ci = struct {
            fn f(v: usize) c_int {
                return @intCast(v);
            }
        }.f;
        {
            var a: cuda.Args = .{};
            a.add(self.pick.ptr);
            a.add(s.counts.ptr);
            a.add(ci(P));
            try cuda.launch.launch(self.pk.group_count, .{ .grid = .{ .x = @intCast(E) }, .block = .{ .x = 256 } }, stream, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(self.pick.ptr);
            a.add(s.counts.ptr);
            a.add(s.ids.ptr);
            a.add(s.count.ptr);
            a.add(s.members.ptr);
            a.add(ci(P));
            a.add(ci(slots));
            a.add(ci(E));
            a.add(ci(maxm));
            try cuda.launch.launch(self.pk.group_place, .{ .grid = .{ .x = @intCast(E) }, .block = .{ .x = 256 } }, stream, &a);
        }
        // the work lists: gate/up in 64-row member blocks (grouped_mma), down in 32-row member groups (grouped_rows)
        const len_gu = listLen(P, maxu, 64);
        const len_d = listLen(P, maxu, 32);
        if (len_gu > s.len_gu or len_d > s.len_d) return error.TooManyRows;
        {
            var a: cuda.Args = .{};
            a.add(s.counts.ptr);
            a.add(s.ids.ptr);
            a.add(s.count.ptr);
            a.add(ci(maxu));
            a.add(s.work_gu.ptr);
            a.add(ci(64));
            a.add(ci(len_gu));
            a.add(s.work_d.ptr);
            a.add(ci(32));
            a.add(ci(len_d));
            try cuda.launch.launch(self.pk.work_list, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &a);
        }
        {
            // rot_in: every (row, slot)'s input rotated by its expert's suh (gate and up)
            var a: cuda.Args = .{};
            a.add(x);
            a.add(ci(D));
            a.add(self.pick.ptr);
            a.add(ex.suh_g.ptr);
            a.add(ex.suh_u.ptr);
            a.add(s.xg.ptr);
            a.add(s.xu.ptr);
            a.add(ci(D));
            a.add(ci(slots));
            a.add(ci(E));
            try cuda.launch.launch(self.pk.rot_in, .{ .grid = .{ .x = @intCast(P), .y = @intCast(D / 128), .z = 2 }, .block = .{ .x = 32 } }, stream, &a);
        }
        const cfg_gu = s_cfg(D, I, true);
        const cfg_d = s_cfg(I, D, false);
        {
            // gate/up: grouped_mma over the work list, the splits folded (Z [2, P, I])
            const range = experts.rangeOf(ex.k2_gu[0], ex.k2_gu[1]);
            var a: cuda.Args = .{};
            a.add(s.xg.ptr);
            a.add(s.xu.ptr);
            a.add(ex.gate_ptr.ptr);
            a.add(ex.up_ptr.ptr);
            a.add(ex.gate_k2.ptr);
            a.add(ex.up_k2.ptr);
            a.add(s.ids.ptr);
            a.add(s.count.ptr);
            a.add(s.members.ptr);
            a.add(s.z.ptr);
            a.add(ci(D)); // K
            a.add(ci(I)); // N
            a.add(ci(P));
            a.add(ci(cfg_gu.sk)); // SK
            a.add(ci(cfg_gu.warps)); // WK
            a.add(ci(maxm));
            a.add(ci(slots));
            a.add(ci(len_gu)); // nexp: the list's pairs
            a.add(s.work_gu.ptr);
            try cuda.launch.launch(self.pk.mma[@intFromEnum(range)], .{ .grid = .{ .x = 1, .y = @intCast(I / 64), .z = @intCast(len_gu * 2) }, .block = .{ .x = 256 } }, stream, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(s.z.ptr);
            a.add(self.pick.ptr);
            a.add(ex.svh_g.ptr);
            a.add(ex.svh_u.ptr);
            a.add(ex.suh_d.ptr);
            a.add(s.xd.ptr);
            a.add(ci(P));
            a.add(ci(I));
            a.add(ci(1)); // SK: folded
            a.add(ci(E));
            a.add(limit);
            a.add(experts.act_f32);
            try cuda.launch.launch(self.pk.gateup_epilogue, .{ .grid = .{ .x = @intCast(P), .y = @intCast(I / 128) }, .block = .{ .x = 32 } }, stream, &a);
        }
        {
            // down: grouped_rows over its work list (Z [1, 1, P, D])
            const range = experts.rangeOf(ex.k2_d[0], ex.k2_d[1]);
            var a: cuda.Args = .{};
            a.add(s.xd.ptr);
            a.add(s.xd.ptr);
            a.add(ex.down_ptr.ptr);
            a.add(ex.down_ptr.ptr);
            a.add(ex.down_k2.ptr);
            a.add(ex.down_k2.ptr);
            a.add(s.ids.ptr);
            a.add(s.count.ptr);
            a.add(s.members.ptr);
            a.add(s.z.ptr);
            a.add(ci(I)); // K
            a.add(ci(D)); // N
            a.add(ci(P));
            a.add(ci(cfg_d.sk));
            a.add(ci(maxm));
            a.add(ci(slots));
            a.add(ci(len_d));
            a.add(s.work_d.ptr);
            try cuda.launch.launch(self.pk.rows[@intFromEnum(range)], .{ .grid = .{ .x = 1, .y = @intCast(D / 128), .z = @intCast(len_d * cfg_d.sk) }, .block = .{ .x = 128 } }, stream, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(s.z.ptr);
            a.add(self.pick.ptr);
            a.add(ex.svh_d.ptr);
            a.add(s.no_y.ptr);
            a.add(self.wts_moe.ptr);
            a.add(@as(u64, 0)); // add
            a.add(self.moe.ptr);
            a.add(ci(P));
            a.add(ci(D));
            a.add(ci(cfg_d.sk));
            a.add(ci(E));
            a.add(ci(slots));
            a.add(ci(0)); // store_y
            try cuda.launch.launch(self.pk.down_combine, .{ .grid = .{ .x = @intCast(R), .y = @intCast(D / 128) }, .block = .{ .x = @intCast(32 * slots) } }, stream, &a);
        }
    }

    // -- small launches ------------------------------------------------------------------------------------------

    fn rmsnorm(self: *Prompt, x: Ptr, w: Ptr, out: Ptr, n: usize, d: usize, stream: cuda.Stream) !void {
        const block = std.math.ceilPowerOfTwo(usize, d) catch return error.BadShape;
        try self.fwd.set.run(stream, "_rmsnorm", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("W", "*bf16", w),
            aot.ptr("OUT", "*bf16", out), aot.int("os_", @intCast(d)), aot.float("eps", self.fwd.cfg.eps),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("PDL", 1) });
    }

    fn ropeHeads(self: *Prompt, x: Ptr, rope: attention.Rope, pos: Ptr, n: usize, h: usize, hd: usize, inverse: bool, stream: cuda.Stream) !void {
        const hb = @min(h, 16);
        try self.fwd.set.run(stream, "_rope_heads", .{ @intCast(n), @intCast(h / hb), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("COS", "*fp32", rope.cos.ptr), aot.ptr("SIN", "*fp32", rope.sin.ptr), aot.ptr("POS", "*i64", pos),
        }, &.{ aot.ci("H", @intCast(h)), aot.ci("HD", @intCast(hd)), aot.ci("RD", @intCast(self.fwd.cfg.rope_dim)), aot.ci("INV", @intFromBool(inverse)), aot.ci("HB", @intCast(hb)), aot.ci("PDL", 1) });
    }

    fn fp4Store(self: *Prompt, x: Ptr, d: usize, codes: Ptr, scales: Ptr, rows: Ptr, n: usize, block: usize, e4m3: bool, stream: cuda.Stream) !void {
        try self.fwd.set.run(stream, "_fp4_store", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("CODES", "*u8", codes), aot.ptr("SCALES", "*u8", scales), aot.ptr("ROWS", "*i64", rows),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("E4M3", @intFromBool(e4m3)), aot.ci("PDL", 1) });
    }

    fn copyCols(self: *Prompt, src: Ptr, sld: usize, dst: Ptr, dld: usize, rows: usize, cols: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_int, @intCast(sld)));
        a.add(dst);
        a.add(@as(c_int, @intCast(dld)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(cols)));
        try cuda.launch.launch(self.fwd.glue.copy_cols, .{ .grid = .{ .x = @intCast((cols + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    /// dst[dst_idx[r] or r] = src[src_idx[r] or r] (0: linear), rows of `words` 4-byte words.
    fn copyRows(self: *Prompt, src: Ptr, src_idx: Ptr, dst: Ptr, dst_idx: Ptr, rows: usize, words: usize, stream: cuda.Stream) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(src_idx);
        a.add(dst);
        a.add(dst_idx);
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(words)));
        try cuda.launch.launch(self.glue.copy_rows, .{ .grid = .{ .x = @intCast((words + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    fn castBf16(self: *Prompt, src: Ptr, dst: Ptr, count: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i64, @intCast(count)));
        try cuda.launch.launch(self.glue.bf16_to_f32, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }

    fn castF16(self: *Prompt, src: Ptr, dst: Ptr, count: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i64, @intCast(count)));
        try cuda.launch.launch(self.glue.f16_to_f32, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }

    fn fillI64(self: *Prompt, dst: Ptr, value: i64, count: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(dst);
        a.add(value);
        a.add(@as(i64, @intCast(count)));
        try cuda.launch.launch(self.glue.fill_i64, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }
};

fn s_cfg(K: usize, N: usize, gateup: bool) experts.Tiles {
    return experts.defaultConfig(K, N, gateup) catch unreachable;
}

test "work list lengths follow experts._list_len" {
    try std.testing.expectEqual(@as(usize, (600 * 7 + 385 * 63) / 64), listLen(600 * 7, 385, 64));
    try std.testing.expectEqual(@as(usize, (64 * 7 + 385 * 31) / 32), listLen(64 * 7, 385, 32));
}
