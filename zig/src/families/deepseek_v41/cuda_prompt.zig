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

pub const Error = error{ TooManyRows, BadShape, OutOfMemory, NoCandidateSource, PromptKernelMismatch } || cuda.Error || exl3.Error || experts.Error || cuda.cublaslt.Error;

/// The widest chunk a `Prompt` holds buffers for (TF_DS_PREFILL_CHUNK at most this).
pub const max_chunk: usize = 8192;
/// The fork's production chunk (TF_DS_PREFILL_CHUNK's default). A wider chunk holds whole chunks of this width
/// (lanes.chunkEnd), and the work whose bits depend on the row count (the fp32 GEMMs: cuBLAS picks its algorithm by
/// the rows) runs on them one at a time, so a wider pass makes the bytes of this one.
pub const base_chunk: usize = 2048;
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

/// experts_prompt.cu: grouped_mma_kernel<2, LO, HI, true, 4>, grouped_mmb_kernel<2, LO, HI, true, 4> and
/// grouped_rows_kernel<2, 8, 4, 1, LO, HI, 2, false> by K2 range, with the grouping kernels of the experts fatbin.
pub const ExpertKernels = struct {
    module: cuda.Module,
    mma: [3]cuda.Function,
    rows: [3]cuda.Function,
    /// grouped_mmb_kernel<2, LO, HI, true, 4> (gate/up: grouped_mma's bits, the mma spread over every warp)
    mmb: [3]cuda.Function,
    /// grouped_rows_cp_kernel<2, LO, HI> (down with its loads ahead, TF_DS_PROMPT_DOWN=rowscp)
    rowscp: [3]cuda.Function,
    down: Down = .rowscp,
    /// The kernel a prompt chunk's gate/up takes (TF_DS_PROMPT_GU), and TF_DS_PROMPT_CHECK=1: every layer also runs
    /// gate/up on grouped_mma (the reference) and fails unless the outputs are byte-equal.
    gu: Gu = .mmb,
    check: bool = false,
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
            k.rowscp[i] = try module.function(try symbol(&buf, "_ZN8tf_exl3x22grouped_rows_cp_kernelILi2ELi{d}ELi{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiiS7_", .{ r[0], r[1] }));
            k.mmb[i] = try module.function(try symbol(&buf, "_ZN8tf_exl3x18grouped_mmb_kernelILi2ELi{d}ELi{d}ELb1ELi4EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiiiS7_", .{ r[0], r[1] }));
        }
        k.gu = std.meta.stringToEnum(Gu, envOr("TF_DS_PROMPT_GU", "mmb")) orelse return error.BadShape;
        k.check = std.mem.eql(u8, envOr("TF_DS_PROMPT_CHECK", "0"), "1");
        k.down = std.meta.stringToEnum(Down, envOr("TF_DS_PROMPT_DOWN", "rowscp")) orelse return error.BadShape;
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

/// index_score.cu: the prompt indexer's scores with `_index_score_rows`' bits at the mma rate (TF_DS_INDEX_SCORE=mma,
/// the default; `triton` takes the captured kernel), the packed keys dequantized as each tile is staged. TF_DS_INDEX_CHECK=1
/// runs the captured kernel beside it on every row block and fails unless scores, keys and tile maxima are
/// byte-equal.
pub const IndexKernels = struct {
    module: cuda.Module,
    score: cuda.Function,
    keys: cuda.Function,
    /// rows a program, one warp a row (TF_DS_INDEX_RB: 8 or 16)
    rb: usize,
    use: bool,
    check: bool,
    /// 64-key tiles a program (TF_DS_INDEX_TILES)
    tiles: usize,

    pub fn load(d: *const cuda.Driver) !IndexKernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_index_score);
        errdefer module.unload();
        const mode = envOr("TF_DS_INDEX_SCORE", "mma");
        const rb = std.fmt.parseInt(usize, envOr("TF_DS_INDEX_RB", "16"), 10) catch return error.BadShape;
        if (rb != 8 and rb != 16) return error.BadShape;
        if (!std.mem.eql(u8, mode, "mma") and !std.mem.eql(u8, mode, "triton")) return error.BadShape;
        return .{
            .module = module,
            .score = try module.function(if (rb == 8) "tf_dsv41_index_score_rows8" else "tf_dsv41_index_score_rows16"),
            .keys = try module.function(if (rb == 8) "tf_dsv41_index_keys_rows8" else "tf_dsv41_index_keys_rows16"),
            .rb = rb,
            .use = std.mem.eql(u8, mode, "mma"),
            .check = std.mem.eql(u8, envOr("TF_DS_INDEX_CHECK", "0"), "1"),
            .tiles = std.fmt.parseInt(usize, envOr("TF_DS_INDEX_TILES", "16"), 10) catch return error.BadShape,
        };
    }

    pub fn deinit(self: *IndexKernels) void {
        self.module.unload();
    }
};

/// down's prompt kernel (TF_DS_PROMPT_DOWN): grouped_rows_cp (the default: grouped_rows' programs, the loads several k
/// tiles ahead through shared memory; GB10 down 228 -> 211 ms a 2,048-row chunk) or grouped_rows (the fork's); the same
/// bits in both: every output the same 4 chains of 18 k tiles from zero, added in order.
pub const Down = enum { rows, rowscp };

/// sparse_attn.cu: the prompt path's sparse attention with `_sparse_attn_part`'s bits, each key block staged once for
/// both head blocks of a row (TF_DS_SPARSE_ATTN=mma; the default `triton` keeps the captured kernel, which GB10 still
/// runs faster: 3.1-3.3 against 3.4-3.7 ms a 2,048-row call with picks).
/// TF_DS_ATTN_CHECK=1 runs the captured kernel beside it on every call and fails unless the outputs are byte-equal.
/// TF_DS_ATTN_ORDER (0-2) picks the order of the four warps' block sums (sparse_attn.cu sum4; the captured kernel's
/// is 0, the default).
pub const AttnKernels = struct {
    module: cuda.Module,
    attn: cuda.Function,
    use: bool,
    check: bool,

    /// TF_DS_SPARSE_ATTN=ws: the warp-specialized kernel (384 threads, smem_ws)
    ws: bool = false,

    /// sparse_attn.cu's dynamic shared memory: the row's queries, a key block, p, the reductions
    pub const smem: u32 = 2 * 32 * 520 * 2 + 2 * 16 * 40 * 2 + 2 * 4 * 16 * 4 + 32 * 4 + 32 * 8;
    /// the warp-specialized kernel's: the queries, two key blocks, p, the reductions, the keys' validity
    pub const smem_ws: u32 = 32 * 512 * 2 + 2 * 32 * 512 * 2 + 2 * 16 * 32 * 2 + 2 * 4 * 16 * 4 + 2 * 32 * 4;

    pub fn load(d: *const cuda.Driver) !AttnKernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_sparse_attn);
        errdefer module.unload();
        const mode = envOr("TF_DS_SPARSE_ATTN", "triton");
        if (!std.mem.eql(u8, mode, "mma") and !std.mem.eql(u8, mode, "ws") and !std.mem.eql(u8, mode, "triton")) return error.BadShape;
        const ws = std.mem.eql(u8, mode, "ws");
        const order = envOr("TF_DS_ATTN_ORDER", "0");
        var buf: [40]u8 = undefined;
        const name = try symbol(&buf, "tf_dsv41_sparse_attn{s}", .{order});
        const f = try module.function(if (ws) "tf_dsv41_sparse_attn_ws" else name);
        try f.allowDynamicShared(if (ws) smem_ws else smem);
        return .{ .module = module, .attn = f, .ws = ws, .use = ws or std.mem.eql(u8, mode, "mma"), .check = std.mem.eql(u8, envOr("TF_DS_ATTN_CHECK", "0"), "1") };
    }

    pub fn deinit(self: *AttnKernels) void {
        self.module.unload();
    }
};

/// The prompt mHC between two sublayers: a sublayer's hc_post fused into the next sublayer's mixes, the posted
/// streams written into the other stream buffer. TF_DS_HC=cuda: hc_fused.cu (`_hc_post` and `_hc_partial` in one
/// pass, their bits), then the captured `_hc_finish`; `mix`: the fork's kernels.py hc_pre2 with `gathered`
/// (`_hc_mix_part` posted, then `_hc_finish_s`; the same bits, slower at prompt rows: its programs are the decode
/// path's); the default `split` keeps `_hc_post`, `_hc_partial`, `_hc_finish`. Fused where nothing reads the posted
/// streams between the two sublayers: within a layer always, across layers unless the next one starts with Engram, a
/// DSpark tap, the bounded-replay cut, or the walk's end (a piece's boundary).
/// TF_DS_HC_CHECK=1 runs the split kernels beside every fused call and fails unless x, the new streams, pre, post and
/// comb are byte-equal.
pub const HcFuse = struct {
    mode: Mode,
    check: bool,
    module: ?cuda.Module = null,
    post_partial: ?cuda.Function = null,

    pub const Mode = enum { split, mix, cuda };
    /// hc_fused.cu's rows a program and dynamic shared memory (a step's fn columns of the 4 streams)
    pub const rows_per_program = 8;
    pub const smem: u32 = 24 * 4 * 128 * 4;

    pub fn load(d: *const cuda.Driver) !HcFuse {
        const mode = std.meta.stringToEnum(Mode, envOr("TF_DS_HC", "split")) orelse return error.BadShape;
        var self: HcFuse = .{ .mode = mode, .check = std.mem.eql(u8, envOr("TF_DS_HC_CHECK", "0"), "1") };
        if (mode == .cuda) {
            if (!cuda.kernels.available) return error.BuiltWithoutKernels;
            var module = try cuda.Module.load(d, cuda.kernels.dsv41_hc_fused);
            errdefer module.unload();
            const f = try module.function("tf_dsv41_hc_post_partial");
            try f.allowDynamicShared(smem);
            self.module = module;
            self.post_partial = f;
        }
        return self;
    }

    pub fn deinit(self: *HcFuse) void {
        if (self.module) |*m| m.unload();
    }

    pub fn on(self: HcFuse) bool {
        return self.mode != .split;
    }

    /// hc_fused.cu's kernel `f` (rows_per_program rows a program, smem bytes): h_out [rows, 4, d] = _hc_post of h with
    /// gathered [2, rows, d] (rank partials rs elements apart), post [rows, 4], comb [rows, 4, 4]; part [rows, 40, 32]
    /// = _hc_partial of h_out with fn [24, 4 d] (columns 0..24). h_out must not overlap h.
    pub fn postPartial(f: cuda.Function, rows_a_program: u32, stream: cuda.Stream, h: Ptr, h_out: Ptr, gathered: Ptr, rs: usize, post_: Ptr, comb: Ptr, fn_: Ptr, part: Ptr, rows: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(h);
        a.add(h_out);
        a.add(gathered);
        a.add(@as(i64, @intCast(rs)));
        a.add(post_);
        a.add(comb);
        a.add(fn_);
        a.add(part);
        a.add(ci(rows));
        a.add(ci(d));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows + rows_a_program - 1) / rows_a_program), .y = @intCast(d / 512) }, .block = .{ .x = 256 }, .shared = smem }, stream, &a);
    }
};

/// The prompt GEMM's preparation: TF_DS_GEMM_PREP=fused runs linear.cu `prep` (rot_in's rows and unpack's W_q in one
/// launch, the weights stored 16 bytes a lane) and lets wo_a read o's column blocks and write u's in place (row
/// strides) instead of copy_cols; the default `split` keeps rot_in, unpack and copy_cols. Only where the values are
/// placed changes. TF_DS_GEMM_PREP_CHECK=1 compares xh, W_q and every strided output with the split path's.
pub const GemmPrep = struct {
    fused: bool,
    check: bool,

    pub fn load() !GemmPrep {
        const mode = envOr("TF_DS_GEMM_PREP", "split");
        if (!std.mem.eql(u8, mode, "fused") and !std.mem.eql(u8, mode, "split")) return error.BadShape;
        return .{ .fused = std.mem.eql(u8, mode, "fused"), .check = std.mem.eql(u8, envOr("TF_DS_GEMM_PREP_CHECK", "0"), "1") };
    }
};

/// Byte equality for a TF_DS_*_CHECK comparison: error.PromptKernelMismatch (logged with the count) when they differ.
fn checkEqual(what: []const u8, want: []const u8, got: []const u8) !void {
    var bad: usize = 0;
    for (want, got) |a, b| bad += @intFromBool(a != b);
    if (bad != 0) {
        std.log.err("deepseek_v41: prompt GEMM prep: {s}: {d} of {d} bytes differ from the split path", .{ what, bad, want.len });
        return error.PromptKernelMismatch;
    }
}

/// gate/up's prompt kernel: grouped_mma (the fork's) or grouped_mmb (the same bits, the mma over every warp).
pub const Gu = enum { mma, mmb };

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn envOr(name: [:0]const u8, default: []const u8) []const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return default);
    return if (v.len == 0) default else v;
}

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
    zc: ?cuda.DeviceBuffer = null,

    pub fn init(d: *const cuda.Driver, ex: *const experts.Experts, rows: usize, slots: usize) !PromptScratch {
        const D = ex.dims;
        const I = ex.width;
        const P = rows * slots;
        const maxu = @min(P, ex.count);
        const maxm = @max(32, (rows + 31) / 32 * 32);
        const zrow = @max(2 * I + D / 2, D);
        var s: PromptScratch = undefined;
        s.zc = null;
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
        if (self.zc) |*b| b.free();
    }

    /// TF_DS_PROMPT_CHECK: the reference kernels' output, z's size (made on first use).
    fn zcheck(self: *PromptScratch) !cuda.DeviceBuffer {
        if (self.zc == null) self.zc = try cuda.DeviceBuffer.alloc(self.z.d, self.z.len);
        return self.zc.?;
    }

    /// TF_DS_PROMPT_CHECK: the first `words` 32-bit words of z and of the reference output byte-equal, else
    /// error.PromptKernelMismatch (logged with the count of differing words).
    fn compare(self: *PromptScratch, li: usize, what: []const u8, words: usize, stream: cuda.Stream) !void {
        try stream.synchronize();
        const gpa = std.heap.page_allocator;
        const a = try gpa.alloc(u32, words);
        defer gpa.free(a);
        const b = try gpa.alloc(u32, words);
        defer gpa.free(b);
        try self.z.download(0, std.mem.sliceAsBytes(a));
        try self.zc.?.download(0, std.mem.sliceAsBytes(b));
        var bad: usize = 0;
        for (a, b) |x, y| bad += @intFromBool(x != y);
        if (bad != 0) {
            std.log.err("deepseek_v41: layer {d} {s}: {d} of {d} words differ from the reference kernel", .{ li, what, bad, words });
            return error.PromptKernelMismatch;
        }
        std.log.info("deepseek_v41: layer {d} {s}: {d} words byte-equal to the reference kernel", .{ li, what, words });
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
        return self.runAs(x, w, out, m, m, n, k, stream);
    }

    /// `run` on m rows in pieces of `base_chunk` rows (the last one shorter): the rows of each piece get the bits of
    /// a `base_chunk`-row chunk's run, as the chunks of a wider pass hold whole such chunks.
    pub fn runChunked(self: *Gemm32, x: Ptr, w: Ptr, out: Ptr, m: usize, n: usize, k: usize, stream: cuda.Stream) !void {
        var r: usize = 0;
        while (r < m) : (r += base_chunk) {
            const p = @min(base_chunk, m - r);
            try self.run(x + r * k * 4, w, out + r * n * 4, p, n, k, stream);
        }
    }

    /// `run` on m rows with the algorithm the heuristic picks for `as_rows` rows (the rows of a whole chunk when it
    /// runs in pieces).
    pub fn runAs(self: *Gemm32, x: Ptr, w: Ptr, out: Ptr, m: usize, as_rows: usize, n: usize, k: usize, stream: cuda.Stream) !void {
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
        try lt.check(api.cublasLtMatrixLayoutCreate(&lb, .f32, k, as_rows, @intCast(k)), "layout X");
        defer _ = api.cublasLtMatrixLayoutDestroy(lb);
        try lt.check(api.cublasLtMatrixLayoutCreate(&lc, .f32, n, as_rows, @intCast(n)), "layout OUT");
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
        if (as_rows != m) {
            var lb2: cuda.cublaslt.Layout = null;
            var lc2: cuda.cublaslt.Layout = null;
            try lt.check(api.cublasLtMatrixLayoutCreate(&lb2, .f32, k, m, @intCast(k)), "layout X");
            defer _ = api.cublasLtMatrixLayoutDestroy(lb2);
            try lt.check(api.cublasLtMatrixLayoutCreate(&lc2, .f32, n, m, @intCast(n)), "layout OUT");
            defer _ = api.cublasLtMatrixLayoutDestroy(lc2);
            const one: f32 = 1;
            const zero: f32 = 0;
            try lt.check(api.cublasLtMatmul(self.handle, desc, &one, w, la, x, lb2, &zero, out, lc2, out, lc2, &found[0].algo, self.workspace.ptr, workspace_bytes, stream.handle), "cublasLtMatmul");
            return;
        }
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
/// completed layer's sublayer groups, read back after the chunk (a host sync a chunk, so chunks no longer overlap), and the
/// host's time inside the chunk call. A replay cut's unfinished source update is charged to the head/tail interval.
/// `chunk` logs one `info:` line a profiled chunk.
pub const Profile = struct {
    pub const groups = [_][]const u8{ "engram/replay source", "attention", "attention gather", "moe", "moe gather" };
    /// The MoE group's parts: the FFN mix, the gate's logits and the route, the member and work lists, the experts'
    /// input rotation, gate/up, their epilogue, down, the combine (the decode path's experts count as the combine).
    pub const moe_parts = [_][]const u8{ "mix", "gate", "lists", "rot_in", "gate/up", "epilogue", "down", "combine" };
    /// Prompt attention stages; a replay tail using window attention is charged to input projections.
    pub const attn_parts = [_][]const u8{ "mix", "input projections", "kv/index", "sparse", "output projections" };
    /// [1 + layers * groups.len + 1]: before the embedding, after each layer's groups, after the head.
    marks: []cuda.Event,
    /// [layers * (moe_parts.len - 1)]: between the MoE group's parts.
    moe: []cuda.Event,
    /// [layers * (attn_parts.len - 1)]: between the attention group's parts.
    attn: []cuda.Event,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, layers: usize) !Profile {
        const marks = try gpa.alloc(cuda.Event, 2 + layers * groups.len + layers * (moe_parts.len - 1) + layers * (attn_parts.len - 1));
        errdefer gpa.free(marks);
        var made: usize = 0;
        errdefer for (marks[0..made]) |*e| e.deinit();
        for (marks) |*e| {
            e.* = try cuda.Event.init(d, true);
            made += 1;
        }
        const n = 2 + layers * groups.len;
        const na = layers * (attn_parts.len - 1);
        return .{ .marks = marks[0..n], .moe = marks[n .. marks.len - na], .attn = marks[marks.len - na ..], .gpa = gpa };
    }

    pub fn deinit(self: *Profile) void {
        const all = self.marks.ptr[0 .. self.marks.len + self.moe.len + self.attn.len];
        for (all) |*e| e.deinit();
        self.gpa.free(all);
    }
};

/// A prompt's chunk timeline without a host sync a chunk (`Prompt.enableTrace`, TF_DS_PREFILL_TRACE=1 in the served
/// engine): CUDA events at each chunk's ends and around each of its gathers, in two sets that alternate between
/// chunks. A set is read when the chunk after next starts (a wait on its end event: the host runs at most about a
/// chunk ahead, so the GPU stays busy) and the rest by `Prompt.traceReport` after the prompt's sync. Sums the chunks'
/// GPU time, the time in gathers (waiting for the other rank included), and the GPU's idle time between one chunk's
/// end and the next one's start.
pub const Trace = struct {
    pub const max_gathers = 128;
    const Set = struct {
        begin: cuda.Event,
        end: cuda.Event,
        gathers: [max_gathers][2]cuda.Event,
        n: usize = 0,
        live: bool = false,
    };
    sets: [2]Set,
    k: usize = 0,
    chunks: usize = 0,
    gpu_ms: f64 = 0,
    gather_ms: f64 = 0,
    gap_ms: f64 = 0,
    gathers: usize = 0,

    pub fn init(d: *const cuda.Driver) !Trace {
        var t: Trace = .{ .sets = undefined };
        for (&t.sets) |*s| {
            s.* = .{ .begin = try cuda.Event.init(d, true), .end = try cuda.Event.init(d, true), .gathers = undefined };
            for (&s.gathers) |*g| g.* = .{ try cuda.Event.init(d, true), try cuda.Event.init(d, true) };
        }
        return t;
    }

    pub fn deinit(self: *Trace) void {
        for (&self.sets) |*s| {
            s.begin.deinit();
            s.end.deinit();
            for (&s.gathers) |*g| {
                g[0].deinit();
                g[1].deinit();
            }
        }
    }

    /// Folds set `i` (its chunk waited for) into the sums, with the gap to the next chunk when that one has begun.
    fn drain(self: *Trace, i: usize) !void {
        const s = &self.sets[i];
        if (!s.live) return;
        try s.end.synchronize();
        self.chunks += 1;
        self.gpu_ms += try cuda.Event.elapsedMs(s.begin, s.end);
        for (s.gathers[0..s.n]) |g| self.gather_ms += try cuda.Event.elapsedMs(g[0], g[1]);
        self.gathers += s.n;
        const next = &self.sets[1 - i];
        if (next.live) {
            try next.begin.synchronize();
            self.gap_ms += @max(0, try cuda.Event.elapsedMs(s.end, next.begin));
        }
        s.live = false;
    }

    /// A chunk begins on `stream`: the set it takes (the one of the chunk before last, read first).
    fn begin(self: *Trace, stream: cuda.Stream) !void {
        const i = self.k % 2;
        try self.drain(i);
        const s = &self.sets[i];
        s.n = 0;
        try s.begin.record(stream);
        s.live = true;
    }

    fn end(self: *Trace, stream: cuda.Stream) !void {
        const s = &self.sets[self.k % 2];
        try s.end.record(stream);
        self.k += 1;
    }

    /// The gather's two events (null past max_gathers: that gather is not timed).
    fn gatherEvents(self: *Trace) ?*[2]cuda.Event {
        const s = &self.sets[self.k % 2];
        if (s.n == max_gathers) return null;
        s.n += 1;
        return &s.gathers[s.n - 1];
    }

    /// Every set read; the sums reset after the caller took them.
    fn finish(self: *Trace) !void {
        // the older chunk first: its gap reaches the newer one's begin
        try self.drain(self.k % 2);
        try self.drain((self.k + 1) % 2);
    }
};

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// TF_DS_HC_CHECK: the split kernels' outputs for a fused mHC call (`Prompt.mixes`), and the byte comparison.
const HcRef = struct {
    h: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer,
    pre: cuda.DeviceBuffer,
    post: cuda.DeviceBuffer,
    comb: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    got: cuda.DeviceBuffer, // a fused output, copied out of the walk's buffers for the download

    fn init(d: *const cuda.Driver, rows: usize, dm: usize) !HcRef {
        const hs = hcmod.hc_streams;
        return .{
            .h = try cuda.DeviceBuffer.alloc(d, rows * hs * dm * 2),
            .x = try cuda.DeviceBuffer.alloc(d, rows * dm * 2),
            .pre = try cuda.DeviceBuffer.alloc(d, rows * hs * 4),
            .post = try cuda.DeviceBuffer.alloc(d, rows * hs * 4),
            .comb = try cuda.DeviceBuffer.alloc(d, rows * hs * hs * 4),
            .part = try cuda.DeviceBuffer.alloc(d, hcmod.Hc.partWords(rows) * 4),
            .got = try cuda.DeviceBuffer.alloc(d, rows * hs * dm * 2),
        };
    }

    fn deinit(self: *HcRef) void {
        inline for (.{ "h", "x", "pre", "post", "comb", "part", "got" }) |f| @field(self, f).free();
    }

    /// The fused call's new streams, x, pre_out, post, comb (`got`, in that order) byte-equal to the split kernels',
    /// else error.PromptKernelMismatch (logged with the first differing output and its count of differing bytes).
    fn compare(self: *HcRef, li: usize, rows: usize, dm: usize, got: [5]Ptr, stream: cuda.Stream) !void {
        const hs = hcmod.hc_streams;
        const names = [5][]const u8{ "streams", "x", "pre", "post", "comb" };
        const want = [5]cuda.DeviceBuffer{ self.h, self.x, self.pre, self.post, self.comb };
        const bytes = [5]usize{ rows * hs * dm * 2, rows * dm * 2, rows * hs * 4, rows * hs * 4, rows * hs * hs * 4 };
        const gpa = std.heap.page_allocator;
        const a = try gpa.alloc(u8, bytes[0]);
        defer gpa.free(a);
        const b = try gpa.alloc(u8, bytes[0]);
        defer gpa.free(b);
        for (names, want, got, bytes) |name, wb, gp, n| {
            try self.got.copyFrom(0, gp, n, stream.handle);
            try stream.synchronize();
            try wb.download(0, a[0..n]);
            try self.got.download(0, b[0..n]);
            var bad: usize = 0;
            for (a[0..n], b[0..n]) |x, y| bad += @intFromBool(x != y);
            if (bad != 0) {
                std.log.err("deepseek_v41: layer {d} fused mHC ({d} rows): {s}: {d} of {d} bytes differ from _hc_post + _hc_partial + _hc_finish", .{ li, rows, name, bad, n });
                return error.PromptKernelMismatch;
            }
        }
        std.log.info("deepseek_v41: layer {d} fused mHC ({d} rows): byte-equal to _hc_post + _hc_partial + _hc_finish", .{ li, rows });
    }
};

pub const Prompt = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    fwd: *Forward,
    /// Per-chunk timing when set (`enableProfile`).
    profile: ?Profile = null,
    /// The chunks' timeline when set (`enableTrace`), and the gather it wraps during a chunk.
    trace: ?Trace = null,
    traced: Gather = undefined,
    kernels: *const exl3.Kernels,
    glue: Glue,
    pk: ExpertKernels,
    ix: IndexKernels,
    sa: AttnKernels,
    hcf: HcFuse,
    gp: GemmPrep,
    gemm: Gemm32,
    rows: usize, // the chunk rows this holds buffers for (at most max_chunk)
    rr: attention.RopeRows, // a chunk's rope rows (the device path's: its positions, its groups', the indexer's)
    had: cuda.DeviceBuffer, // the 128 x 128 Hadamard bf16 (prefill.py Workspace.hadamard)
    // the chunk's streams and mixes
    ids: cuda.DeviceBuffer,
    h: cuda.DeviceBuffer,
    h2: ?cuda.DeviceBuffer, // TF_DS_HC=fused: the streams' other buffer (the posted streams go there)
    hc_ref: ?HcRef, // TF_DS_HC_CHECK: the split kernels' outputs (made on first use)
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
    tmax: cuda.DeviceBuffer, // int64 [rb, nt]: the keyed path's tile maxima of a row block (unread)
    cidx: cuda.DeviceBuffer, // int64 [rows, idx_topk]
    cand: cuda.DeviceBuffer, // uint32 [rows, bitWords(cand cols)]: the chunk's rows' candidate pool as bits (device selection)
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
    /// A chunk issued in pieces (`piece`): the layer loop's state between them.
    at_piece: Walk = undefined,
    /// `piece`: bounded replay left the current chunk no row past its cut, so its later pieces issue nothing
    piece_cut: bool = false,

    /// The layer loop's state: the residual streams and mixes it reads next, the rows and positions it runs, and
    /// the next layer.
    const Walk = struct {
        h: Ptr,
        h_alt: Ptr, // TF_DS_HC=fused: h's rows in the other stream buffer
        pre: Ptr,
        pre_f: Ptr,
        rows: usize,
        at: usize,
        floor: usize,
        kv_done: bool,
        window_picks: ?attention.Picks,
        li: usize,
    };

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
        self.trace = null;
        self.kernels = kernels;
        self.rows = rows;
        self.glue = try Glue.load(d);
        errdefer self.glue.deinit();
        self.pk = try ExpertKernels.load(d, fwd.ek);
        errdefer self.pk.deinit();
        self.ix = try IndexKernels.load(d);
        errdefer self.ix.deinit();
        self.sa = try AttnKernels.load(d);
        errdefer self.sa.deinit();
        std.log.info("deepseek_v41 rank {d}: prompt sparse attention: {s}", .{ fwd.w.rank, if (self.sa.ws) "sparse_attn.cu, warp-specialized (TF_DS_SPARSE_ATTN=ws)" else if (self.sa.use) "sparse_attn.cu (TF_DS_SPARSE_ATTN=mma)" else "the captured _sparse_attn_part (TF_DS_SPARSE_ATTN=triton)" });
        self.hcf = try HcFuse.load(d);
        errdefer self.hcf.deinit();
        // hc_fused.cu: two rank partials, 512-column K blocks (4 D = 40 x 512)
        if (self.hcf.mode == .cuda and (fwd.world != 2 or 4 * dm != hcmod.hc_blocks * 512)) return error.BadShape;
        std.log.info("deepseek_v41 rank {d}: prompt mHC: {s}{s}", .{ fwd.w.rank, switch (self.hcf.mode) {
            .split => "_hc_post, _hc_partial, _hc_finish (TF_DS_HC=split)",
            .mix => "_hc_mix_part posted + _hc_finish_s (TF_DS_HC=mix)",
            .cuda => "hc_fused.cu + _hc_finish (TF_DS_HC=cuda)",
        }, if (self.hcf.check) ", checked against the split kernels (TF_DS_HC_CHECK=1)" else "" });
        self.hc_ref = null;
        self.gp = try GemmPrep.load();
        std.log.info("deepseek_v41 rank {d}: prompt GEMM prep: {s}{s}", .{ fwd.w.rank, if (self.gp.fused) "linear.cu prep, strided wo_a rows (TF_DS_GEMM_PREP=fused)" else "rot_in, unpack, copy_cols (TF_DS_GEMM_PREP=split)", if (self.gp.check) ", checked against the split path (TF_DS_GEMM_PREP_CHECK=1)" else "" });
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
        self.h2 = if (self.hcf.on()) try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * dm * 2) else null;
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
        // the indexer's [rows, n_comp] scores (or keys) of a row block: model.py's bound of 2^24 entries, one row's
        // n_comp at least (`index` scores a block of decode rows a few rows at a time past it)
        const score_words: usize = @max(score_bytes_limit / 4, nb_max);
        self.score = try cuda.DeviceBuffer.alloc(d, score_words * 8);
        // a row block's tile maxima: its rows x ceil(n_comp / 64), the block's rows x n_comp within score_words
        self.tmax = try cuda.DeviceBuffer.alloc(d, (score_words / 64 + r) * 8);
        self.cidx = try cuda.DeviceBuffer.alloc(d, @max(16, r * (c.idx_topk + 16) * 8));
        const cb = @max(1, c.cand_block);
        const cand_cols_max = (nb_max + cb - 1) / cb;
        self.cand = try cuda.DeviceBuffer.alloc(d, @max(16, r * attention.TopkGlue.bitWords(cand_cols_max) * 4));
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
        self.rr = try attention.RopeRows.init(gpa, d, fwd.glue, r + 1, 3, c.rope_dim / 2);
        const host = fwd.glue.hostSelect(); // the host selection's copies of the scores and the pool
        self.score_h = try gpa.alloc(f32, if (host) score_words * 2 else 0);
        self.keys_h = try gpa.alloc(i64, if (host) @max(1, nb_max) else 0);
        self.cidx_h = try gpa.alloc(i64, @max(1, r * (c.idx_topk + 16)));
        self.cand_cols = (nb_max + c.cand_block - 1) / @max(1, c.cand_block);
        self.cand_h = try gpa.alloc(u8, if (host) @max(1, r * self.cand_cols) else 0);
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
        // the fp32 GEMMs once at full chunk rows (the gate's, the indexer weights'): cuBLAS loads their kernels on
        // first use, which would otherwise land in the first prompt's first chunk (~170 ms); the outputs are scratch
        try self.gemm.runChunked(self.x32.ptr, self.gate_w32[0].ptr, self.logits.ptr, r, c.n_routed, dm, stream);
        for (w.layers) |*l| if (l.idx_proj) |ip| {
            try self.gemm.runChunked(self.x32.ptr, ip.ptr(), self.wl.ptr, r, c.idx_heads, dm, stream);
            break;
        };
        // every setup copy and fill so far (this one's, and the weights' and caches' before it) ran on the legacy
        // stream, which the chunks' non-blocking stream does not wait for: they complete before the first chunk
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        return self;
    }

    /// Times every chunk from now on (see `Profile`).
    pub fn enableProfile(self: *Prompt) !void {
        if (self.profile == null) self.profile = try Profile.init(self.gpa, self.d, self.fwd.w.layers.len);
    }

    /// Traces every chunk from now on (see `Trace`; `traceReport` logs a prompt's sums).
    pub fn enableTrace(self: *Prompt) !void {
        if (self.trace == null) self.trace = try Trace.init(self.d);
    }

    /// After the prompt's last chunk has run (the caller synchronized): its chunks' sums as one `info:` line
    /// (nothing without a trace or a traced chunk), and the sums reset.
    pub fn traceReport(self: *Prompt, rows: usize, wall_ms: f64) !void {
        const t = &(self.trace orelse return);
        try t.finish();
        defer {
            t.chunks = 0;
            t.gpu_ms = 0;
            t.gather_ms = 0;
            t.gap_ms = 0;
            t.gathers = 0;
        }
        if (t.chunks == 0) return;
        std.log.info("deepseek_v41 rank {d}: prefill trace: {d} rows in {d} chunks, wall {d:.0} ms: GPU {d:.0} ms in the chunks ({d:.0} ms in {d} gathers), {d:.0} ms idle between chunks", .{
            self.fwd.w.rank, rows, t.chunks, wall_ms, t.gpu_ms, t.gather_ms, t.gathers, t.gap_ms,
        });
    }

    fn tracedGather(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
        const self: *Prompt = @ptrCast(@alignCast(ctx));
        const ev = self.trace.?.gatherEvents();
        if (ev) |e| try e[0].record(stream);
        try self.traced.run(partial, words, out, stream);
        if (ev) |e| try e[1].record(stream);
    }

    pub fn deinit(self: *Prompt) void {
        if (self.trace) |*t| t.deinit();
        if (self.profile) |*p| p.deinit();
        for (self.gate_w32) |*b| b.free();
        self.gpa.free(self.gate_w32);
        if (self.h2) |*b| b.free();
        if (self.hc_ref) |*b| b.deinit();
        inline for (.{ "had", "ids", "h", "pre", "pre_a", "pre_f", "post", "comb", "part", "x", "gathered", "xh", "wq", "z", "pos", "vis", "vis_blk", "gpos", "grows", "src_idx", "dst_idx", "wlo", "qa", "qr", "q", "ykv", "wsrc", "ckv", "cgate", "pooled", "lat", "lat2", "kraw", "k", "iq", "iqq", "x32", "wl", "wts", "score", "tmax", "cidx", "cand", "smax", "o", "og", "ug", "u", "out", "logits", "pick", "wts_moe", "moe", "xc", "local", "glog", "head_logits" }) |f| @field(self, f).free();
        self.scratch.deinit();
        self.gemm.deinit();
        self.pk.deinit();
        self.ix.deinit();
        self.sa.deinit();
        self.hcf.deinit();
        self.glue.deinit();
        self.rr.deinit();
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
        var g = gather;
        if (self.trace) |*t| {
            try t.begin(stream);
            self.traced = gather;
            g = .{ .ctx = self, .call = tracedGather };
        }
        const out = try self.layers(g, ids, start, o, stream, &engram_ns);
        if (self.trace) |*t| try t.end(stream);
        // A replay cut stops after its source update. Only completed layers have all profile marks;
        // the remaining source update is timed with the head/tail interval.
        const completed = self.at_piece.li;
        try self.mark(1 + completed * Profile.groups.len, stream);
        if (self.profile != null) try self.report(start, n, completed, nowNs() - t0, engram_ns, stream);
        return out;
    }

    /// How many pieces `piece` issues a chunk of `n` rows with options `o` in when asked for `want`: `want` (at
    /// most the layers) for a chunk through the layer loop without profile or trace (bounded replay included: the
    /// cut happens in the piece holding layer n_layers / 2); else 1.
    pub fn piecesFor(self: *const Prompt, n: usize, o: Options, want: usize) usize {
        if (want <= 1 or self.profile != null or self.trace != null) return 1;
        if (n <= decode_rows and o.taps == null and o.logits) return 1;
        return @min(want, self.fwd.w.layers.len);
    }

    /// Piece `k` of `pieces` (`piecesFor`) of `chunk(gather, ids, start, o, stream)`: the embedding with the first,
    /// layers [L * k / pieces, L * (k + 1) / pieces), the head with the last, which returns what `chunk` returns
    /// (the others null). The pieces in order, on one stream, issue exactly `chunk`'s work in `chunk`'s order;
    /// between them the caller may issue other work that writes none of this path's buffers.
    pub fn piece(self: *Prompt, gather: Gather, ids: []const i64, start: usize, o: Options, stream: cuda.Stream, k: usize, pieces: usize) !?Ptr {
        if (pieces <= 1) return self.chunk(gather, ids, start, o, stream);
        if (k >= pieces or pieces != self.piecesFor(ids.len, o, pieces)) return error.BadShape;
        const L = self.fwd.w.layers.len;
        var engram_ns: i128 = 0;
        if (k == 0) {
            try self.walkBegin(ids, start, stream);
            self.piece_cut = false;
        }
        // bounded replay left the chunk no row past its cut (in an earlier piece): the rest is empty, as in `layers`
        if (self.piece_cut) return null;
        if (self.at_piece.li != L * k / pieces) return error.BadShape;
        if (!try self.walk(gather, o, stream, &engram_ns, L * (k + 1) / pieces)) {
            self.piece_cut = true;
            return null;
        }
        if (k + 1 < pieces) return null;
        return try self.walkHead(gather, o, stream);
    }

    fn mark(self: *const Prompt, i: usize, stream: cuda.Stream) !void {
        if (self.profile) |p| try p.marks[i].record(stream);
    }

    /// The end of the MoE part `k` (Profile.moe_parts, the last excluded) of layer `li`.
    fn moeMark(self: *const Prompt, li: usize, k: usize, stream: cuda.Stream) !void {
        if (self.profile) |p| try p.moe[li * (Profile.moe_parts.len - 1) + k].record(stream);
    }

    fn attnMark(self: *const Prompt, li: usize, k: usize, stream: cuda.Stream) !void {
        if (self.profile) |p| try p.attn[li * (Profile.attn_parts.len - 1) + k].record(stream);
    }

    /// The profiled chunk's line: the GPU's ms by sublayer group (summed over completed layers), the head/replay tail's, and the host's
    /// ms in the call (and in the Engram hook: hashing, row reads, uploads).
    fn report(self: *const Prompt, start: usize, n: usize, completed: usize, host_ns: i128, engram_ns: i128, stream: cuda.Stream) !void {
        const p = self.profile.?;
        const L = completed;
        const ng = Profile.groups.len;
        const t_sync = nowNs();
        try stream.synchronize();
        const wait_ms = @as(f64, @floatFromInt(nowNs() - t_sync)) / 1e6;
        var sum: [Profile.groups.len]f64 = @splat(0);
        for (0..L * ng) |i| sum[i % ng] += try cuda.Event.elapsedMs(p.marks[i], p.marks[i + 1]);
        const head = try cuda.Event.elapsedMs(p.marks[L * ng], p.marks[L * ng + 1]);
        const total = try cuda.Event.elapsedMs(p.marks[0], p.marks[L * ng + 1]);
        const nm = Profile.moe_parts.len;
        var moe: [nm]f64 = @splat(0);
        for (0..L) |li| {
            const ms = p.moe[li * (nm - 1) ..][0 .. nm - 1];
            moe[0] += try cuda.Event.elapsedMs(p.marks[1 + li * ng + 2], ms[0]);
            for (1..nm - 1) |k| moe[k] += try cuda.Event.elapsedMs(ms[k - 1], ms[k]);
            moe[nm - 1] += try cuda.Event.elapsedMs(ms[nm - 2], p.marks[1 + li * ng + 3]);
        }
        const na = Profile.attn_parts.len;
        var attn: [na]f64 = @splat(0);
        for (0..L) |li| {
            const ms = p.attn[li * (na - 1) ..][0 .. na - 1];
            attn[0] += try cuda.Event.elapsedMs(p.marks[1 + li * ng], ms[0]);
            for (1..na - 1) |k| attn[k] += try cuda.Event.elapsedMs(ms[k - 1], ms[k]);
            attn[na - 1] += try cuda.Event.elapsedMs(ms[na - 2], p.marks[1 + li * ng + 1]);
        }
        std.log.info("deepseek_v41 rank {d}: chunk at {d}: attention ms by part: mix {d:.1}, input projections {d:.1}, kv/index {d:.1}, sparse {d:.1}, output projections {d:.1}", .{
            self.fwd.w.rank, start, attn[0], attn[1], attn[2], attn[3], attn[4],
        });
        std.log.info("deepseek_v41 rank {d}: chunk at {d} ({d} rows, {d} completed layers): GPU {d:.0} ms (engram/replay source {d:.0}, attention {d:.0}, attention gather {d:.0}, moe {d:.0}, moe gather {d:.0}, head/replay tail {d:.1}); host {d:.0} ms in the call ({d:.0} in Engram), then {d:.0} ms waiting for the GPU", .{
            self.fwd.w.rank, start, n, completed, total, sum[0], sum[1], sum[2], sum[3], sum[4], head,
            @as(f64, @floatFromInt(host_ns)) / 1e6, @as(f64, @floatFromInt(engram_ns)) / 1e6, wait_ms,
        });
        std.log.info("deepseek_v41 rank {d}: chunk at {d}: moe ms by part: mix {d:.1}, gate {d:.1}, lists {d:.1}, rot_in {d:.1}, gate/up {d:.1}, epilogue {d:.1}, down {d:.1}, combine {d:.1}", .{
            self.fwd.w.rank, start, moe[0], moe[1], moe[2], moe[3], moe[4], moe[5], moe[6], moe[7],
        });
    }

    fn layers(self: *Prompt, gather: Gather, ids: []const i64, start: usize, o: Options, stream: cuda.Stream, engram_ns: *i128) !?Ptr {
        try self.walkBegin(ids, start, stream);
        if (!try self.walk(gather, o, stream, engram_ns, self.fwd.w.layers.len)) return null;
        return try self.walkHead(gather, o, stream);
    }

    /// The chunk's ids up and embedded, the layer loop at its first layer (`at_piece`).
    fn walkBegin(self: *Prompt, ids: []const i64, start: usize, stream: cuda.Stream) !void {
        const n = ids.len;
        const w = self.fwd.w;
        @memcpy(self.ids_h[0..n], ids);
        // every host table goes up in stream order: a layer rewrites the tables (positions, gather and scatter rows,
        // picks) that launches still queued from the layer before read, and a plain upload (the legacy stream) is not
        // ordered after a non-blocking stream's work
        try self.ids.uploadAsync(0, std.mem.sliceAsBytes(self.ids_h[0..n]), stream.handle);
        try self.fwd.hc.embedInit(stream, w.embed.ptr(), self.ids.ptr, self.h.ptr, self.pre.ptr, @intCast(n));
        self.cidx_cols = 0;
        self.cidx_stride = 0;
        self.cand_rows = 0;
        self.top_start = start;
        self.top_rows = n;
        // the window layers' picks (shared["topk"]) once replay leaves 16 rows or fewer: `attendWindow`
        self.at_piece = .{ .h = self.h.ptr, .h_alt = if (self.h2) |b| b.ptr else 0, .pre = self.pre.ptr, .pre_f = self.pre_f.ptr, .rows = n, .at = start, .floor = 0, .kv_done = false, .window_picks = null, .li = 0 };
    }

    /// The layer loop from `at_piece.li` to layer `end`; false when bounded replay leaves no row past the cut (the
    /// chunk then holds no decoder rows: no head).
    fn walk(self: *Prompt, gather: Gather, o: Options, stream: cuda.Stream, engram_ns: *i128, end: usize) !bool {
        const c = self.fwd.cfg;
        const dm = c.dim;
        const w = self.fwd.w;
        const fwd = self.fwd;
        const st = &self.at_piece;
        var nb: [32]u8 = undefined;
        // TF_DS_HC=fused: self.gathered holds the last sublayer's partials, not yet posted into st.h (the next
        // sublayer's mixes post them)
        var pending = false;
        while (st.li < end) : (st.li += 1) {
            const li = st.li;
            const l = &w.layers[li];
            if (o.replay) |replay| if (li == c.n_layers / 2) {
                // CED bounded replay: this layer's kv source over every row, then only the rows at or past `replay`
                try fwd.hc.pre(stream, st.h, l.hc_attn_fn.ptr(), l.hc_attn_scale.ptr(), l.hc_attn_base.ptr(), st.pre, l.attn_norm.ptr(), self.x.ptr, self.pre_a.ptr, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(st.rows));
                try self.kvSourceUpdate(li, self.x.ptr, st.rows, st.at, stream);
                const first = @max(st.at, replay) - st.at;
                if (first >= st.rows) {
                    self.top_rows = 0;
                    return false;
                }
                st.h += first * hcmod.hc_streams * dm * 2;
                st.h_alt += first * hcmod.hc_streams * dm * 2;
                st.pre += first * hcmod.hc_streams * 4;
                st.at += first;
                st.rows -= first;
                st.floor = replay;
                st.kv_done = true;
                self.top_start = st.at;
                self.top_rows = st.rows;
            };
            const rows = st.rows;
            const at = st.at;
            const hbytes = rows * hcmod.hc_streams * dm * 2;
            const m0 = 1 + li * Profile.groups.len;
            if (l.engram_wkv != null) if (fwd.engram) |e| {
                std.debug.assert(!pending);
                const te = nowNs();
                try e.call(e.ctx, li, st.h, at, rows, .{ .ctx = self, .call = mmF32 }, gather, stream);
                engram_ns.* += nowNs() - te;
                try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_engram", .{li}), st.h, hbytes, stream);
            };
            if (o.taps) |t| if (c.dspark_taps.indexOf(@intCast(li))) |ti| {
                std.debug.assert(!pending);
                try fwd.hc.tap(stream, st.h, t.buf + ti * dm * 2, @intCast(t.stride), @intCast(rows));
            };
            try self.mark(m0, stream);
            try self.mixes(st, pending, l.hc_attn_fn.ptr(), l.hc_attn_scale.ptr(), l.hc_attn_base.ptr(), st.pre, l.attn_norm.ptr(), self.pre_a.ptr, rows, stream);
            if (pending) try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{li - 1}), st.h, hbytes, stream);
            pending = false;
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_attn", .{li}), self.x.ptr, rows * dm * 2, stream);
            try self.attnMark(li, 0, stream);
            // 16 rows or fewer past a replay cut run model.py attention_k's ring mode: the window kernels
            const pa = if (st.kv_done and rows <= decode_rows)
                try self.attendWindow(li, self.x.ptr, rows, at, li == c.n_layers / 2, &st.window_picks, stream)
            else
                try self.attend(li, self.x.ptr, rows, at, st.floor, st.kv_done and li == c.n_layers / 2, stream);
            // Window attention has no prompt stages; attribute it to the first interval.
            if (st.kv_done and rows <= decode_rows) for (1..Profile.attn_parts.len - 1) |k| try self.attnMark(li, k, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_attn.r0", .{li}), pa, rows * dm * 4, stream);
            try self.mark(m0 + 1, stream);
            try gather.run(pa, rows * dm, self.gathered.ptr, stream);
            if (self.hcf.on()) pending = true else {
                try fwd.hc.post(stream, self.gathered.ptr, fwd.world, st.h, self.post.ptr, self.comb.ptr, st.h, @intCast(rows));
                try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_attn", .{li}), st.h, hbytes, stream);
            }
            try self.mark(m0 + 2, stream);
            try self.mixes(st, pending, l.hc_ffn_fn.ptr(), l.hc_ffn_scale.ptr(), l.hc_ffn_base.ptr(), self.pre_a.ptr, l.ffn_norm.ptr(), st.pre_f, rows, stream);
            if (pending) try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_attn", .{li}), st.h, hbytes, stream);
            pending = false;
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_ffn", .{li}), self.x.ptr, rows * dm * 2, stream);
            const pm = try self.moeRows(li, self.x.ptr, rows, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_moe.r0", .{li}), pm, rows * dm * 4, stream);
            try self.mark(m0 + 3, stream);
            try gather.run(pm, rows * dm, self.gathered.ptr, stream);
            if (self.hcf.on() and self.fusesInto(li + 1, end, o)) pending = true else {
                try fwd.hc.post(stream, self.gathered.ptr, fwd.world, st.h, self.post.ptr, self.comb.ptr, st.h, @intCast(rows));
                try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{li}), st.h, hbytes, stream);
            }
            try self.mark(m0 + 4, stream);
            const t = st.pre;
            st.pre = st.pre_f;
            st.pre_f = t;
        }
        std.debug.assert(!pending);
        return true;
    }

    /// TF_DS_HC=fused: whether layer `li`'s attention mixes may post the previous layer's MoE partials, i.e. nothing
    /// reads the posted streams before them: `li` is in this walk, and starts with no Engram, no DSpark tap and no
    /// bounded-replay cut.
    fn fusesInto(self: *const Prompt, li: usize, end: usize, o: Options) bool {
        const c = self.fwd.cfg;
        if (li >= end) return false;
        if (self.fwd.w.layers[li].engram_wkv != null and self.fwd.engram != null) return false;
        if (o.taps != null and c.dspark_taps.indexOf(@intCast(li)) != null) return false;
        if (o.replay != null and li == c.n_layers / 2) return false;
        return true;
    }

    /// hc_pre of a sublayer on st.h: x, pre_out and the sublayer's post / comb from its mixes (fn, scale, base) and
    /// pre_in. `posted`: self.gathered holds the previous sublayer's partials, not yet posted into st.h; they are posted
    /// into the other stream buffer (by the mixes' own programs), which becomes st.h.
    fn mixes(self: *Prompt, st: *Walk, posted: bool, fn_: Ptr, scale: Ptr, base: Ptr, pre_in: Ptr, norm: Ptr, pre_out: Ptr, rows: usize, stream: cuda.Stream) !void {
        const hc = &self.fwd.hc;
        const r: u32 = @intCast(rows);
        if (!posted) return hc.pre(stream, st.h, fn_, scale, base, pre_in, norm, self.x.ptr, pre_out, self.post.ptr, self.comb.ptr, self.part.ptr, r);
        var ref: ?*HcRef = null;
        if (self.hcf.check) {
            if (self.hc_ref == null) self.hc_ref = try HcRef.init(self.d, self.rows, self.fwd.cfg.dim);
            ref = &self.hc_ref.?;
            const f = ref.?;
            try hc.post(stream, self.gathered.ptr, self.fwd.world, st.h, self.post.ptr, self.comb.ptr, f.h.ptr, r);
            try hc.pre(stream, f.h.ptr, fn_, scale, base, pre_in, norm, f.x.ptr, f.pre.ptr, f.post.ptr, f.comb.ptr, f.part.ptr, r);
        }
        const fused = self.hcf.post_partial != null;
        if (self.hcf.post_partial) |f| {
            try HcFuse.postPartial(f, HcFuse.rows_per_program, stream, st.h, st.h_alt, self.gathered.ptr, rows * self.fwd.cfg.dim, self.post.ptr, self.comb.ptr, fn_, self.part.ptr, rows, self.fwd.cfg.dim);
        } else {
            try hc.mixPart(stream, st.h, st.h_alt, self.gathered.ptr, @intCast(self.fwd.world), self.post.ptr, self.comb.ptr, fn_, self.part.ptr, r, hc.pdl, true);
        }
        const t = st.h;
        st.h = st.h_alt;
        st.h_alt = t;
        if (fused) {
            try hc.finish(stream, st.h, self.part.ptr, base, scale, pre_in, norm, self.x.ptr, pre_out, self.post.ptr, self.comb.ptr, r);
        } else {
            try hc.finishS(stream, st.h, self.part.ptr, base, scale, pre_in, norm, self.x.ptr, pre_out, self.post.ptr, self.comb.ptr, r, true);
        }
        if (ref) |f| try f.compare(st.li, rows, self.fwd.cfg.dim, .{ st.h, self.x.ptr, pre_out, self.post.ptr, self.comb.ptr }, stream);
    }

    /// The head on the walked chunk's last row (null without `o.logits`).
    fn walkHead(self: *Prompt, gather: Gather, o: Options, stream: cuda.Stream) !?Ptr {
        if (!o.logits) return null;
        const fwd = self.fwd;
        const w = fwd.w;
        const dm = fwd.cfg.dim;
        const st = &self.at_piece;
        const h0 = st.h + (st.rows - 1) * hcmod.hc_streams * dm * 2;
        const pre0 = st.pre + (st.rows - 1) * hcmod.hc_streams * 4;
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
        return self.mmStrided(lin, x, lin.k, m, y, lin.n, y_dtype, stream);
    }

    /// `mm` on rows x_stride elements apart (a column block of a wider matrix) into rows y_stride elements apart;
    /// strides other than (K, N) only at prompt rows with TF_DS_GEMM_PREP=fused.
    fn mmStrided(self: *Prompt, lin: *const exl3.Linear, x: Ptr, x_stride: usize, m: usize, y: Ptr, y_stride: usize, y_dtype: exl3.DType, stream: cuda.Stream) !void {
        const K = lin.k;
        const N = lin.n;
        if (m <= linear_rows) {
            std.debug.assert(x_stride == K and y_stride == N);
            return lin.run(x, .bf16, m, self.xh.ptr, self.z.ptr, y, y_dtype, stream);
        }
        if (self.gp.fused) return self.mmFused(lin, x, x_stride, m, y, y_stride, y_dtype, stream);
        std.debug.assert(x_stride == K and y_stride == N);
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

    /// TF_DS_GEMM_PREP=fused: linear.cu `prep` (rot_in's rows read x_stride apart and the unpacked W_q in one launch),
    /// then `_gemm` writing rows y_stride apart. TF_DS_GEMM_PREP_CHECK=1: xh, W_q and the output against the split
    /// path's (copy_cols, rot_in, unpack, `_gemm` into contiguous rows), byte for byte.
    fn mmFused(self: *Prompt, lin: *const exl3.Linear, x: Ptr, x_stride: usize, m: usize, y: Ptr, y_stride: usize, y_dtype: exl3.DType, stream: cuda.Stream) !void {
        const K = lin.k;
        const N = lin.n;
        {
            const s = lin.strides();
            const f = try self.kernels.prep(lin.k2, lin.cb);
            const rot_blocks: usize = ((K / 128 + 3) / 4) * m;
            const unpack_blocks: usize = ((K / 16) * (N / 16) + 3) / 4;
            var a: cuda.Args = .{};
            a.add(x);
            a.add(@intFromEnum(exl3.DType.bf16));
            a.add(ci(x_stride));
            a.add(lin.suh.ptr);
            a.add(self.xh.ptr);
            a.add(ci(K));
            a.add(ci(rot_blocks));
            a.add(lin.words.ptr);
            a.add(self.wq.ptr);
            a.add(ci(N));
            a.add(s.k);
            a.add(s.nb);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rot_blocks + unpack_blocks) }, .block = .{ .x = 128 } }, stream, &a);
        }
        if (self.gp.check) try self.prepCheck(lin, x, x_stride, m, stream);
        try self.promptGemm(lin, m, y, y_stride, y_dtype, stream);
        if (self.gp.check and y_stride != N) {
            // the split path's output: contiguous rows (x32 as scratch), compared with y's column block
            try self.promptGemm(lin, m, self.x32.ptr, N, y_dtype, stream);
            const el: usize = if (y_dtype == .f32) 4 else 2;
            try stream.synchronize();
            const gpa = std.heap.page_allocator;
            const want = try gpa.alloc(u8, m * N * el);
            defer gpa.free(want);
            const got = try gpa.alloc(u8, m * N * el);
            defer gpa.free(got);
            try self.x32.download(0, want);
            for (0..m) |r| try (cuda.DeviceBuffer{ .d = self.d, .ptr = y + r * y_stride * el, .len = N * el }).download(0, got[r * N * el ..][0 .. N * el]);
            try checkEqual("the strided _gemm output", want, got);
        }
    }

    fn promptGemm(self: *Prompt, lin: *const exl3.Linear, m: usize, y: Ptr, y_stride: usize, y_dtype: exl3.DType, stream: cuda.Stream) !void {
        const N = lin.n;
        const bm: usize = 128;
        const grid: u32 = @intCast(((m + bm - 1) / bm) * (N / 128));
        try self.fwd.set.run(stream, "_gemm", .{ grid, 1, 1 }, &.{
            aot.ptr("X", "*fp16", self.xh.ptr),     aot.ptr("W", "*fp16", self.wq.ptr),   aot.ptr("H", "*bf16", self.had.ptr),
            aot.ptr("SVH", "*fp16", lin.svh.ptr),   aot.ptr("BIAS", "*fp16", lin.svh.ptr), aot.ptr("OUT", if (y_dtype == .f32) "*fp32" else "*bf16", y),
            aot.int("M", @intCast(m)),              aot.int("o_stride", @intCast(y_stride)),
        }, &.{ aot.ci("K", @intCast(lin.k)), aot.ci("N", @intCast(N)), aot.ci("BM", 128), aot.ci("BK", 32), aot.ci("GROUP", 8), aot.ci("HAS_BIAS", 0), aot.cf("SCALE", prompt_bias_scale) });
    }

    /// TF_DS_GEMM_PREP_CHECK: the fused prep's xh and W_q (just written) against copy_cols + rot_in + unpack's; the
    /// split path's values are left in xh and W_q.
    fn prepCheck(self: *Prompt, lin: *const exl3.Linear, x: Ptr, x_stride: usize, m: usize, stream: cuda.Stream) !void {
        const K = lin.k;
        const N = lin.n;
        try stream.synchronize();
        const gpa = std.heap.page_allocator;
        const xh_got = try gpa.alloc(u8, m * K * 2);
        defer gpa.free(xh_got);
        const wq_got = try gpa.alloc(u8, K * N * 2);
        defer gpa.free(wq_got);
        try self.xh.download(0, xh_got);
        try self.wq.download(0, wq_got);
        var src = x;
        if (x_stride != K) {
            try self.copyCols(x, x_stride, self.x32.ptr, K, m, K, stream);
            src = self.x32.ptr;
        }
        {
            var a: cuda.Args = .{};
            a.add(src);
            a.add(@intFromEnum(exl3.DType.bf16));
            a.add(lin.suh.ptr);
            a.add(self.xh.ptr);
            a.add(ci(K));
            try cuda.launch.launch(lin.rot_in, .{ .grid = .{ .x = @intCast((K / 128 + 3) / 4), .y = @intCast(m) }, .block = .{ .x = 128 } }, stream, &a);
        }
        {
            const s = lin.strides();
            const f = try self.kernels.unpack(lin.k2, lin.cb);
            var a: cuda.Args = .{};
            a.add(lin.words.ptr);
            a.add(self.wq.ptr);
            a.add(ci(N));
            a.add(s.k);
            a.add(s.nb);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(N / 16), .y = @intCast(K / 16) }, .block = .{ .x = 32 } }, stream, &a);
        }
        try stream.synchronize();
        const want = try gpa.alloc(u8, @max(m * K, K * N) * 2);
        defer gpa.free(want);
        try self.xh.download(0, want[0 .. m * K * 2]);
        try checkEqual("xh", want[0 .. m * K * 2], xh_got);
        try self.wq.download(0, want[0 .. K * N * 2]);
        try checkEqual("W_q", want[0 .. K * N * 2], wq_got);
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
        const at_pos = try self.rr.at(rope, 0, self.pos.ptr, n, stream);
        // attn_in: wq_a, wkv (the compressor's in kvSourceUpdate)
        try self.mm(p.wq_a, x, n, self.qa.ptr, .bf16, stream);
        try self.mm(p.wkv, x, n, self.ykv.ptr, .bf16, stream);
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.qa.r0", .{li}), self.qa.ptr, n * c.q_rank * 2, stream);
        try self.rmsnorm(self.qa.ptr, p.q_norm, self.qr.ptr, n, c.q_rank, stream);
        try self.mm(p.wq_b, self.qr.ptr, n, self.q.ptr, .bf16, stream);
        try self.ropeHeads(self.q.ptr, at_pos, n, heads, hd, false, stream);
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.q.r0", .{li}), self.q.ptr, n * heads * hd * 2, stream);
        try self.attnMark(li, 1, stream);
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
            aot.ptr("Y", "*bf16", self.ykv.ptr),  aot.ptr("W", "*bf16", p.kv_norm), aot.ptr("COS", "*fp32", at_pos.cos),
            aot.ptr("SIN", "*fp32", at_pos.sin), aot.ptr("POS", "*i64", at_pos.pos), aot.ptr("OUT", "*bf16", kv),
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
        const comp_codes: Ptr = if (has) srcp.?.comp_codes.ptr else self.wsrc.ptr;
        const comp_scales: Ptr = if (has) srcp.?.comp_scales.ptr else self.wsrc.ptr;
        try self.attnMark(li, 2, stream);
        // window-only calls (no picks) stay on the captured kernel: GB10 runs them faster there
        const fast = self.sa.use and has and hd == 512 and heads == 32 and c.window == 128;
        if (fast and !self.sa.check) {
            try self.sparseAttnMma(n, comp_codes, comp_scales, has, p.sink, scale, n_pad, nblk, stream);
        } else try fwd.set.run(stream, "_sparse_attn_part", .{ @intCast(n), @intCast(heads / 16), 1 }, &.{
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
        if (fast and self.sa.check) try self.attnCheck(li, n, comp_codes, comp_scales, has, p.sink, scale, n_pad, nblk, stream);
        try self.attnMark(li, 3, stream);
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.wsrc.r0", .{li}), self.wsrc.ptr, (before + n) * hd * 2, stream);
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.o_raw.r0", .{li}), self.o.ptr, n * heads * hd * 2, stream);
        try self.ropeHeads(self.o.ptr, at_pos, n, heads, hd, true, stream);
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
            if (self.gp.fused and n > linear_rows) {
                // o's column block read and u's written in place (the GEMM prep's row stride, _gemm's o_stride)
                try self.mmStrided(p.wo_a[g], self.o.ptr + g * g_cols * 2, o_cols, n, self.u.ptr + g * c.o_rank * 2, dims.groups * c.o_rank, .bf16, stream);
                continue;
            }
            try self.copyCols(self.o.ptr + g * g_cols * 2, o_cols, self.og.ptr, g_cols, n, g_cols, stream);
            try self.mm(p.wo_a[g], self.og.ptr, n, self.ug.ptr, .bf16, stream);
            try self.copyCols(self.ug.ptr, c.o_rank, self.u.ptr + g * c.o_rank * 2, dims.groups * c.o_rank, n, c.o_rank, stream);
        }
        try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.dbg.u.r0", .{li}), self.u.ptr, n * dims.groups * c.o_rank * 2, stream);
        try self.mm(p.wo_b, self.u.ptr, n, self.out.ptr, .f32, stream);
        return self.out.ptr;
    }

    /// `_sparse_attn_part` (prompt rows, one split) through sparse_attn.cu into `o`.
    fn sparseAttnMma(self: *Prompt, n: usize, comp: Ptr, csc: Ptr, has: bool, sink: Ptr, scale: f32, n_idx: usize, nblk: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(self.q.ptr);
        a.add(self.wsrc.ptr);
        a.add(self.wlo.ptr);
        a.add(comp);
        a.add(csc);
        a.add(if (has) self.cidx.ptr else self.pos.ptr);
        a.add(self.pos.ptr);
        a.add(sink);
        a.add(self.o.ptr);
        a.add(scale);
        a.add(@as(c_int, @intCast(n_idx)));
        a.add(@as(c_int, @intCast(nblk)));
        a.add(@as(c_int, @intFromBool(has)));
        const block: u32 = if (self.sa.ws) 384 else 256;
        const shared: u32 = if (self.sa.ws) AttnKernels.smem_ws else AttnKernels.smem;
        try cuda.launch.launch(self.sa.attn, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = block }, .shared = shared }, stream, &a);
    }

    /// TF_DS_ATTN_CHECK: the captured kernel's `o` (just written) against sparse_attn.cu's, byte for byte; sparse_attn.cu's
    /// stays.
    fn attnCheck(self: *Prompt, li: usize, n: usize, comp: Ptr, csc: Ptr, has: bool, sink: Ptr, scale: f32, n_idx: usize, nblk: usize, stream: cuda.Stream) !void {
        const c = self.fwd.cfg;
        const bytes = n * self.dimsOf(li).heads * c.head_dim * 2;
        try stream.synchronize();
        const gpa = std.heap.page_allocator;
        const want = try gpa.alloc(u8, bytes);
        defer gpa.free(want);
        const got = try gpa.alloc(u8, bytes);
        defer gpa.free(got);
        try self.o.download(0, want);
        try self.sparseAttnMma(n, comp, csc, has, sink, scale, n_idx, nblk, stream);
        try stream.synchronize();
        try self.o.download(0, got);
        var bad: usize = 0;
        var first: usize = bytes;
        for (want, got, 0..) |x, y, i| if (x != y) {
            bad += 1;
            if (first == bytes) first = i;
        };
        if (bad != 0) {
            std.log.err("deepseek_v41: layer {d} sparse attention ({d} rows, {d} blocks, picks {}): {d} of {d} bytes differ from _sparse_attn_part, the first at {d}", .{ li, n, nblk, has, bad, bytes, first });
            return error.PromptKernelMismatch;
        }
        std.log.info("deepseek_v41: layer {d} sparse attention ({d} rows, {d} blocks, picks {}): byte-equal to _sparse_attn_part", .{ li, n, nblk, has });
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
        const pa = try a.forward(self.paramsOf(li), &caches, rope, &fwd.rr, x, n, start, picks.*, stream);
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
        const at_g = try self.rr.at(rope, 1, self.gpos.ptr, full, stream);
        if (dims.has_idx_wk) {
            try self.mm(p.idx_wk.?, self.lat.ptr, full, self.kraw.ptr, .bf16, stream);
            try self.glue.rmsNormTorch(self.kraw.ptr, p.idx_k_norm, c.eps, full, c.idx_dim, self.k.ptr, stream);
            try self.ropeHeads(self.k.ptr, at_g, full, 1, c.idx_dim, false, stream);
            try self.fp4Store(self.k.ptr, c.idx_dim, caches.idxk_codes.ptr, caches.idxk_scales.ptr, self.grows.ptr, full, 32, false, stream);
        }
        try self.lat2.copyFrom(0, self.lat.ptr, full * hd * 2, stream.handle);
        try self.ropeHeads(self.lat2.ptr, at_g, full, 1, hd, false, stream);
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
        try self.ropeHeads(self.iq.ptr, try self.rr.at(rope, 2, self.pos.ptr, n, stream), n, c.idx_heads, c.idx_dim, false, stream);
        const nblk: usize = n * c.idx_heads * c.idx_dim / 32;
        try fwd.set.run(stream, "_fp4qd_p2", .{ @intCast((nblk + 31) / 32), 1, 1 }, &.{
            aot.ptr("X", "*bf16", self.iq.ptr), aot.ptr("OUT", "*bf16", self.iqq.ptr), aot.int("nblk", @intCast(nblk)),
        }, &.{ aot.ci("BLKS", 32), aot.ci("PDL", 1) });
        // the weights: x.float() @ idx_proj.t() (cuBLAS), to bf16, scaled
        try self.castBf16(x, self.x32.ptr, n * c.dim, stream);
        try self.gemm.runChunked(self.x32.ptr, fwd.w.layers[li].idx_proj.?.ptr(), self.wl.ptr, n, c.idx_heads, c.dim, stream);
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
        // a block of decode rows (rb at its least, 16) past the score buffer runs as steps of the rows it holds: the
        // decode rows' kernel scores and selects each row on its own, so the steps write what one block would
        const step = if (rb * n_comp_end <= self.score.len / 8) rb else @max(1, (self.score.len / 8) / @max(n_comp_end, 1));
        if (step < rb and rb > decode_rows) return error.BadShape;
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
        const fast = self.ix.use and n > decode_rows and c.idx_heads == 32 and c.idx_dim == 128;
        var r0: usize = 0;
        while (r0 < n) : (r0 += step) {
            const r1 = @min(n, r0 + step);
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
            } else if (fast and !self.ix.check) {
                try self.indexScoreMma(li, q, wt, vr, rows, n_comp_end, nt, keyed, stream);
            } else {
                try fwd.set.run(stream, "_index_score_rows", .{ @intCast((rows + 7) / 8), @intCast(nt), 1 }, &.{
                    aot.ptr("Q", "*bf16", q),                     aot.ptr("K", "*u8", caches.idxk_codes.ptr),
                    aot.ptr("KS", "*u8", caches.idxk_scales.ptr), aot.ptr("Wt", "*bf16", wt),
                    aot.ptr("VIS", "*i64", vr),                   aot.ptr("OUT", if (keyed) "*i64" else "*fp32", self.score.ptr),
                    aot.int("n", @intCast(n_comp_end)),           aot.int("rows", @intCast(rows)),
                    aot.ptr("TMAX", if (keyed) "*i64" else "*fp32", if (keyed) self.tmax.ptr else self.score.ptr), aot.int("nt", @intCast(nt)),
                }, &.{ aot.ci("IH", @intCast(c.idx_heads)), aot.ci("ID", @intCast(c.idx_dim)), aot.ci("BN", 64), aot.ci("RB", 8), aot.ci("PACKED", 1), aot.ci("KEYS", @intFromBool(keyed)), aot.ci("HAS_TMAX", @intFromBool(keyed)) });
                if (fast) try self.indexCheck(li, q, wt, vr, rows, n_comp_end, nt, keyed, stream);
            }
            if (!host) {
                const tg = &fwd.glue.topk.?;
                const words = attention.TopkGlue.bitWords(ccols);
                const bits = self.cand.ptr + r0 * words * 4;
                if (is_cand_source) try tg.candidatesBits(self.score.ptr, n_comp_end, rows, n_comp_end, vr, c.cand_blocks, cb, self.smax.ptr, bits, words, stream);
                if (after_cand) try tg.applyBits(self.score.ptr, n_comp_end, rows, n_comp_end, bits, words, cb, stream);
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

    /// `_index_score_rows` on the row block at `q` through index_score.cu: scores (or keyed, the keys and tile maxima)
    /// into `score` / `tmax`, from the layer's packed index keys.
    fn indexScoreMma(self: *Prompt, li: usize, q: Ptr, wt: Ptr, vr: Ptr, rows: usize, n: usize, nt: usize, keyed: bool, stream: cuda.Stream) !void {
        const caches = self.fwd.cachesOf(li);
        const tiles = @max(1, self.ix.tiles);
        const rb = self.ix.rb;
        var a: cuda.Args = .{};
        a.add(q);
        a.add(caches.idxk_codes.ptr);
        a.add(caches.idxk_scales.ptr);
        a.add(wt);
        a.add(vr);
        a.add(self.score.ptr);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(rows)));
        if (keyed) a.add(self.tmax.ptr);
        a.add(@as(c_int, @intCast(nt)));
        a.add(@as(c_int, @intCast(tiles)));
        if (keyed) a.add(@as(c_int, 1));
        const grid: cuda.launch.Dim3 = .{ .x = @intCast((rows + rb - 1) / rb), .y = @intCast((nt + tiles - 1) / tiles) };
        try cuda.launch.launch(if (keyed) self.ix.keys else self.ix.score, .{ .grid = grid, .block = .{ .x = @intCast(32 * rb) } }, stream, &a);
    }

    /// TF_DS_INDEX_CHECK: the captured kernel's output (just written) against index_score.cu's on the same block,
    /// byte for byte (scores or keys, and the keyed path's tile maxima); index_score.cu's stays.
    fn indexCheck(self: *Prompt, li: usize, q: Ptr, wt: Ptr, vr: Ptr, rows: usize, n: usize, nt: usize, keyed: bool, stream: cuda.Stream) !void {
        try stream.synchronize();
        const gpa = std.heap.page_allocator;
        const sb = rows * n * @as(usize, if (keyed) 8 else 4);
        const tb = if (keyed) rows * nt * 8 else 0;
        const want = try gpa.alloc(u8, sb + tb);
        defer gpa.free(want);
        const got = try gpa.alloc(u8, sb + tb);
        defer gpa.free(got);
        try self.score.download(0, want[0..sb]);
        if (keyed) try self.tmax.download(0, want[sb..]);
        try self.indexScoreMma(li, q, wt, vr, rows, n, nt, keyed, stream);
        try stream.synchronize();
        try self.score.download(0, got[0..sb]);
        if (keyed) try self.tmax.download(0, got[sb..]);
        var bad: usize = 0;
        for (want, got) |x, y| bad += @intFromBool(x != y);
        if (bad != 0) {
            std.log.err("deepseek_v41: layer {d} indexer ({d} rows, {d} keys, keyed {}): {d} of {d} bytes differ from _index_score_rows", .{ li, rows, n, keyed, bad, want.len });
            return error.PromptKernelMismatch;
        }
        std.log.info("deepseek_v41: layer {d} indexer ({d} rows, {d} keys, keyed {}): byte-equal to _index_score_rows", .{ li, rows, n, keyed });
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
        try self.moeMark(li, 0, stream);
        if (n <= decode_rows) {
            try fwd.route.gate(stream, x, l.gate_w.ptr(), fwd.parts.ptr, @intCast(n));
            try fwd.route.pick(stream, fwd.parts.ptr, l.gate_b.ptr(), self.pick.ptr, self.wts_moe.ptr, @intCast(n));
        } else {
            try self.castBf16(x, self.x32.ptr, n * c.dim, stream);
            try self.gemm.runChunked(self.x32.ptr, self.gate_w32[li].ptr, self.logits.ptr, n, E, c.dim, stream);
            const nb = std.math.ceilPowerOfTwo(usize, E) catch return error.BadShape;
            const sp = std.math.ceilPowerOfTwo(usize, slots) catch return error.BadShape;
            try fwd.set.run(stream, "_route", .{ @intCast(n), 1, 1 }, &.{
                aot.ptr("L", "*fp32", self.logits.ptr), aot.ptr("BIAS", "*fp32", l.gate_b.ptr()), aot.ptr("PICK", "*i32", self.pick.ptr),
                aot.ptr("WTS", "*fp32", self.wts_moe.ptr), aot.float("scale", c.route_scale),    aot.int("shared_id", @intCast(l.experts.count - 1)),
            }, &.{ aot.ci("NE", @intCast(E)), aot.ci("NB", @intCast(nb)), aot.ci("TOPK", @intCast(c.topk)), aot.ci("SLOTS", @intCast(slots)), aot.ci("SP", @intCast(sp)), aot.ci("PDL", 1) });
        }
        try self.moeMark(li, 1, stream);
        if (n < exact_rows) {
            for (2..Profile.moe_parts.len - 1) |k| try self.moeMark(li, k, stream);
            try experts.routed(fwd.ek, &l.experts, &fwd.scratch, x, self.pick.ptr, self.wts_moe.ptr, n, c.swiglu_limit, self.moe.ptr, stream);
            return self.moe.ptr;
        }
        try self.routedPrompt(li, &l.experts, x, n, c.swiglu_limit, stream);
        return self.moe.ptr;
    }

    /// experts.py routed() for a prompt chunk (the fork's defaults): member lists and work lists on the device, the
    /// rotated inputs, gate/up on grouped_mmb (grouped_mma's bits; splits folded), the epilogue, down on grouped_rows,
    /// the combine.
    fn routedPrompt(self: *Prompt, li: usize, ex: *const experts.Experts, x: Ptr, R: usize, limit: f32, stream: cuda.Stream) !void {
        const s = &self.scratch;
        const D = ex.dims;
        const I = ex.width;
        const E = ex.count;
        const slots = s.slots;
        const P = R * slots;
        const maxu = s.maxu;
        const maxm = @max(32, (R + 31) / 32 * 32);
        if (R > s.rows or maxm > s.maxm) return error.TooManyRows;
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
        try self.moeMark(li, 2, stream);
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
        try self.moeMark(li, 3, stream);
        const cfg_gu = s_cfg(D, I, true);
        const cfg_d = s_cfg(I, D, false);
        {
            // gate/up: grouped_mma (or grouped_mmb) over the work list, the splits folded (Z [2, P, I])
            const range = @intFromEnum(experts.rangeOf(ex.k2_gu[0], ex.k2_gu[1]));
            const f = switch (self.pk.gu) {
                .mma => self.pk.mma[range],
                .mmb => self.pk.mmb[range],
            };
            if (self.pk.check) try self.gateUp(self.pk.mma[range], ex, (try s.zcheck()).ptr, P, maxm, len_gu, cfg_gu, stream);
            try self.gateUp(f, ex, s.z.ptr, P, maxm, len_gu, cfg_gu, stream);
            if (self.pk.check) try s.compare(li, "gate/up", 2 * P * I, stream);
        }
        try self.moeMark(li, 4, stream);
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
        try self.moeMark(li, 5, stream);
        {
            // down: grouped_rows (or grouped_rows_cp) over its work list (Z [1, 1, P, D])
            const range = @intFromEnum(experts.rangeOf(ex.k2_d[0], ex.k2_d[1]));
            const alt = self.pk.down == .rowscp;
            if (alt and self.pk.check) try self.downRows(self.pk.rows[range], ex, (try s.zcheck()).ptr, P, maxm, len_d, cfg_d, stream);
            try self.downRows(if (alt) self.pk.rowscp[range] else self.pk.rows[range], ex, s.z.ptr, P, maxm, len_d, cfg_d, stream);
            if (alt and self.pk.check) try s.compare(li, "down", P * D, stream);
        }
        try self.moeMark(li, 6, stream);
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

    /// gate/up on `f` (grouped_mma_kernel's or grouped_mmb_kernel's arguments) into z [2, P, I].
    fn gateUp(self: *Prompt, f: cuda.Function, ex: *const experts.Experts, z: Ptr, P: usize, maxm: usize, len_gu: usize, cfg: experts.Tiles, stream: cuda.Stream) !void {
        const s = &self.scratch;
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
        a.add(z);
        a.add(ci(ex.dims)); // K
        a.add(ci(ex.width)); // N
        a.add(ci(P));
        a.add(ci(cfg.sk)); // SK
        a.add(ci(cfg.warps)); // WK
        a.add(ci(maxm));
        a.add(ci(s.slots));
        a.add(ci(len_gu)); // nexp: the list's pairs
        a.add(s.work_gu.ptr);
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = @intCast(ex.width / 64), .z = @intCast(len_gu * 2) }, .block = .{ .x = 256 } }, stream, &a);
    }

    /// down on `f` (grouped_rows_kernel's arguments, the fork's tile setting) into z [P, D].
    fn downRows(self: *Prompt, f: cuda.Function, ex: *const experts.Experts, z: Ptr, P: usize, maxm: usize, len_d: usize, cfg: experts.Tiles, stream: cuda.Stream) !void {
        const s = &self.scratch;
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
        a.add(z);
        a.add(ci(ex.width)); // K
        a.add(ci(ex.dims)); // N
        a.add(ci(P));
        a.add(ci(cfg.sk));
        a.add(ci(maxm));
        a.add(ci(s.slots));
        a.add(ci(len_d));
        a.add(s.work_d.ptr);
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = @intCast(ex.dims / 128), .z = @intCast(len_d * cfg.sk) }, .block = .{ .x = 128 } }, stream, &a);
    }

    // -- small launches ------------------------------------------------------------------------------------------

    fn rmsnorm(self: *Prompt, x: Ptr, w: Ptr, out: Ptr, n: usize, d: usize, stream: cuda.Stream) !void {
        const block = std.math.ceilPowerOfTwo(usize, d) catch return error.BadShape;
        try self.fwd.set.run(stream, "_rmsnorm", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("W", "*bf16", w),
            aot.ptr("OUT", "*bf16", out), aot.int("os_", @intCast(d)), aot.float("eps", self.fwd.cfg.eps),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("PDL", 1) });
    }

    fn ropeHeads(self: *Prompt, x: Ptr, at: attention.RopeAt, n: usize, h: usize, hd: usize, inverse: bool, stream: cuda.Stream) !void {
        const hb = @min(h, 16);
        try self.fwd.set.run(stream, "_rope_heads", .{ @intCast(n), @intCast(h / hb), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("COS", "*fp32", at.cos), aot.ptr("SIN", "*fp32", at.sin), aot.ptr("POS", "*i64", at.pos),
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
