//! Flash Next's prompt path for one stream (TensorFold 0.6.6 `forward.forward` on the prompt buffers, `decode.py:
//! prefill_chunk` without the MTP head): a chunk of up to `rows` prompt rows through every layer with the prompt
//! arithmetic (EXL3 matrices decoded once a chunk into a fixed-tile GEMM, the DeltaNet chunk chain, attention in
//! 256-row blocks, routed experts in 1,024-row windows whose slots are kept as bf16; the switches TF_QFN_PROMPT_*
//! put three parts on our kernels with the same bytes), the head on the chunk's last
//! row only, and the commit (the DeltaNet states and conv windows are committed during the chunk). A row's bits do not
//! depend on where a chunk starts or ends.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const glue_mod = @import("cuda_glue.zig");
const hc_mod = @import("cuda_hc.zig");
const experts = @import("cuda_experts.zig");
const attn = @import("cuda_attn.zig");
const fwd = @import("cuda_forward.zig");

const Glue = glue_mod.Glue;
const Ptr = cuda.abi.DevicePtr;
const Linear = dsv41.exl3.Linear;

/// decode.py PREFILL_ROWS: rows of a prompt chunk.
pub const chunk_rows = 2048;

/// The rows of a served prompt chunk: TF_QFN_PROMPT_ROWS (a multiple of att_rows up to 16,384), else chunk_rows. A
/// row's bits do not depend on where a chunk starts or ends, so any chunk size gives the same bytes.
pub fn chunkRows() !usize {
    const v = std.mem.span(std.c.getenv("TF_QFN_PROMPT_ROWS") orelse return chunk_rows);
    if (v.len == 0) return chunk_rows;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadPromptRows;
    if (n == 0 or n % att_rows != 0 or n > 16384) return error.BadPromptRows;
    return n;
}
/// state.py ATT_ROWS: a prompt's attention runs in blocks of this many rows.
pub const att_rows = 256;
/// exl3_pack.py MOE_WINDOW: most rows a routed-expert call takes.
pub const moe_window = 1024;

pub const Kernels = struct {
    io: cuda.Module,
    front: cuda.Function,
    back: cuda.Function,
    pre: cuda.Module,
    chain128: cuda.Function,
    chain64: cuda.Function,
    pglue: cuda.Module,
    to_bf16: cuda.Function,
    f16_mod: cuda.Module, // f16_rows.cu: F16 matrices over a chunk's rows in one launch
    f16_bf16: cuda.Function,
    f16_fp16: cuda.Function,
    f16_diff: cuda.Function,
    f16_reduce: cuda.Function,
    sc: cuda.Module, // scores.cu: the indexer's block scores for many rows a program
    sc_rows: cuda.Function,
    sc_diff: cuda.Function,
    sel: cuda.Module, // select.cu: the indexer's block lists, one program a row
    sel_rows: cuda.Function,
    sel_diff: cuda.Function,
    sms: u32,

    pub fn load(d: *const cuda.Driver, sms: u32) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var io = try cuda.Module.load(d, cuda.kernels.qwen38fn_gdn_io);
        errdefer io.unload();
        var pre = try cuda.Module.load(d, cuda.kernels.qwen38fn_gdn_prefill);
        errdefer pre.unload();
        var pglue = try cuda.Module.load(d, cuda.kernels.qwen38fn_prompt_glue);
        errdefer pglue.unload();
        var f16_mod = try cuda.Module.load(d, cuda.kernels.qwen38fn_f16_rows);
        errdefer f16_mod.unload();
        var sc = try cuda.Module.load(d, cuda.kernels.qwen38fn_scores);
        errdefer sc.unload();
        var sel = try cuda.Module.load(d, cuda.kernels.qwen38fn_select);
        errdefer sel.unload();
        return .{
            .io = io,
            .front = try io.function("_ZN13tf_qfn_gdn_io12front_kernelILi16ELi48EEEvPK13__nv_bfloat16PKxPKiS7_S3_PKfS9_PfSA_PS1_SA_SA_"),
            .back = try io.function("_ZN13tf_qfn_gdn_io11back_kernelILi16ELi48EEEvPK13__nv_bfloat16S3_S3_fPS1_Pf"),
            .pre = pre,
            .chain128 = try pre.function("_ZN18tf_qfn_gdn_prefill12chain_kernelIfLi128ELi32EEEvPKT_S3_PK13__nv_bfloat16PKfS8_S8_PfPS4_iii"),
            .chain64 = try pre.function("_ZN18tf_qfn_gdn_prefill12chain_kernelIfLi64ELi32EEEvPKT_S3_PK13__nv_bfloat16PKfS8_S8_PfPS4_iii"),
            .pglue = pglue,
            .to_bf16 = try pglue.function("tf_qfn_f32_to_bf16"),
            .f16_mod = f16_mod,
            .f16_bf16 = try f16_mod.function("tf_qfn_f16_rows_bf16"),
            .f16_fp16 = try f16_mod.function("tf_qfn_f16_rows_fp16"),
            .f16_diff = try f16_mod.function("tf_qfn_f16_count_diff"),
            .f16_reduce = try f16_mod.function("tf_qfn_f16_reduce"),
            .sc = sc,
            .sc_rows = try sc.function("tf_qfn_scores_rows"),
            .sc_diff = try sc.function("tf_qfn_scores_diff"),
            .sel = sel,
            .sel_rows = try sel.function("tf_qfn_select_rows"),
            .sel_diff = try sel.function("tf_qfn_select_diff"),
            .sms = sms,
        };
    }

    pub fn deinit(self: *Kernels) void {
        self.io.unload();
        self.pre.unload();
        self.pglue.unload();
        self.f16_mod.unload();
        self.sc.unload();
        self.sel.unload();
    }
};

/// The kernels of a prompt chunk's F16 matrices (exl3_mm.py `F16.__call__`): `captured` 0.6.6's `_f16_mm` in
/// 128-row calls and `_reduce`, `fused` f16_rows.cu (the same bytes, one launch), `check` both, compared after
/// every call (error.F16Differ).
pub const F16Mode = enum { captured, fused, check };

/// TF_QFN_PROMPT_F16: `fused` (default), `captured` or `check` (F16Mode).
fn f16Mode() !F16Mode {
    return modeOf(F16Mode, "TF_QFN_PROMPT_F16");
}

/// The most blocks (81,920 keys) select.cu lists; its time grows with the blocks, `_select`'s with the power-of-two
/// width, so it is faster at about 17k blocks (0.13 vs 0.51 ms a call) and slower at about 32k (0.78 vs 0.62).
const fused_select_blocks = 20480;

/// scores.cu SMEM: a scores program's dynamic shared bytes (32 rows' queries as fp32, 64 keys).
const scores_shared: u32 = 32 * 4 * 128 * 4 + 64 * (128 + 8) * 2;

/// TF_QFN_PROMPT_SCORES: the indexer's block scores of a prompt block: `fused` (default) scores.cu (the keys read
/// once for 32 rows), `captured` 0.6.6's `_scores` (the same bytes), `check` both, compared after every call.
fn scoresMode() !F16Mode {
    return modeOf(F16Mode, "TF_QFN_PROMPT_SCORES");
}

fn modeOf(comptime M: type, comptime name: [:0]const u8) !M {
    return modeOr(M, name, .fused);
}

fn modeOr(comptime M: type, comptime name: [:0]const u8, default: M) !M {
    const v = std.mem.span(std.c.getenv(name) orelse return default);
    if (v.len == 0) return default;
    return std.meta.stringToEnum(M, v) orelse error.BadPromptMode;
}

/// TF_QFN_PROMPT_EXPERTS: the kernels of a prompt window's routed experts (cuda_experts.zig `Mode`): `fused`
/// (default) experts_prompt.cu, `captured` 0.6.6's (the same bytes), `check` both, compared after every window.
fn expertsMode() !experts.Mode {
    return modeOf(experts.Mode, "TF_QFN_PROMPT_EXPERTS");
}

/// The prompt buffers (state.py `Buffers(prefill=True)`) and the EXL3 prompt workspace (exl3/prefill.py).
pub const Buffers = struct {
    rows: usize,
    w: hc_mod.Window,
    dn: cuda.DeviceBuffer, // bf16 [rows, low + S]: a hyper-connection's down projection (prompt: summed)
    ids: cuda.DeviceBuffer,
    proj: cuda.DeviceBuffer, // bf16 [rows, PW]: every DeltaNet layer's in turn
    windows: cuda.DeviceBuffer, // int32 [rows, taps]: row r's conv taps r .. r + taps - 1 of [conv state | rows]
    sid: cuda.DeviceBuffer, // int32 [rows] zeros: one stream
    conv_ptr: cuda.DeviceBuffer, // int64 [1]: the layer's conv window
    gq: cuda.DeviceBuffer, // fp32 [rows, nk, dk]
    gk: cuda.DeviceBuffer,
    gv: cuda.DeviceBuffer, // bf16 [rows, nv, dv]
    gg: cuda.DeviceBuffer, // fp32 [rows, nv]
    gb: cuda.DeviceBuffer,
    gy: cuda.DeviceBuffer, // bf16 [rows, nv, dv]: the chain's read-out
    pa: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    iq: cuda.DeviceBuffer,
    attn_o: cuda.DeviceBuffer, // bf16 [rows, heads, head_dim]
    gated: cuda.DeviceBuffer,
    xs_gated: cuda.DeviceBuffer,
    attn: attn.Scratch, // 256 rows
    pos_blk: cuda.DeviceBuffer, // int32 [1]: an attention block's first position
    xh: cuda.DeviceBuffer, // fp16 [rows, K]: the rotated input of a prompt matmul (or the head's)
    wq: cuda.DeviceBuffer, // fp16 [K, N]: a matrix decoded once a chunk
    hadamard: cuda.DeviceBuffer, // bf16 [128, 128]
    z: cuda.DeviceBuffer, // the head's split-K scratch
    logits_router: cuda.DeviceBuffer,
    pick: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer, // bf16 [rows, slots, D]: the experts' slots (MoEBuffers(prefill).y)
    moe: experts.Scratch, // 1,024-row windows
    ple_dev: cuda.DeviceBuffer,
    ple_host: []u8,
    ple_emb: cuda.DeviceBuffer,
    ple_keys: cuda.DeviceBuffer,
    ple_vals: cuda.DeviceBuffer,
    ple_gated: cuda.DeviceBuffer,
    ple_pss: cuda.DeviceBuffer,
    ple_nrow: cuda.DeviceBuffer,
    streams: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer, // bf16 [1, vocab]: the chunk's last row
    f16_mode: F16Mode = .fused,
    scores_mode: F16Mode = .fused,
    select_mode: F16Mode = .fused,
    select_check: ?cuda.DeviceBuffer = null, // check: the fused lists (int32 ids [att_rows, idw], nk, sparse)
    scores_check: ?cuda.DeviceBuffer = null, // check: the fused scores (fp32 [att_rows, nb])
    f16_check: ?cuda.DeviceBuffer = null, // check: the fused call's rows (bf16 [rows, widest N])
    f16_part: ?cuda.DeviceBuffer = null, // fused: K slices' sums (fp32 [sk, rows, n], the largest matrix's)
    f16_diffs: ?cuda.DeviceBuffer = null,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *const fwd.Model, rows: usize, capacity: usize) !Buffers {
        const cfg = &m.cfg;
        var b: Buffers = undefined;
        b.gpa = gpa;
        b.rows = rows;
        const wide = cfg.wide();
        var part: usize = 0;
        for ([_][2]usize{ .{ cfg.low + cfg.streams, wide }, .{ cfg.low, wide }, .{ 2 * cfg.nv, cfg.hidden }, .{ cfg.hidden, cfg.hidden }, .{ wide, cfg.hidden } }) |nk| {
            part = @max(part, weights.f16Split(nk[0], nk[1]) * glue_mod.call_rows * nk[0]);
        }
        b.w = try hc_mod.Window.init(d, cfg, rows, part);
        var kmax: usize = cfg.hidden;
        var kn: usize = 0;
        var zmax: usize = 16;
        for (m.layers) |*l| switch (l.mixer) {
            .gdn => |*g| for ([_]*const Linear{ &g.qkv, &g.z, &g.out }) |x| {
                kmax = @max(kmax, x.k);
                kn = @max(kn, x.k * x.n);
            },
            .attn => |*a| for ([_]*const Linear{ &a.parts[0], &a.parts[1], &a.parts[2], &a.parts[3], &a.o }) |x| {
                kmax = @max(kmax, x.k);
                kn = @max(kn, x.k * x.n);
            },
        };
        if (m.head) |*h| zmax = @max(zmax, h.scratchWords(1) * 4);
        const aw = cfg.heads * 2 * cfg.head_dim + 2 * cfg.kv_heads * cfg.head_dim + (cfg.index_heads + 1) * cfg.index_dim;
        const slots = cfg.slots();
        const ple_words: usize = if (m.table) |t| t.words else 1;
        const ple_heads: usize = if (m.ng) |g| g.heads() else 1;
        const sizes = [_]struct { *cuda.DeviceBuffer, usize }{
            .{ &b.dn, rows * (cfg.low + cfg.streams) * 2 },       .{ &b.ids, rows * 4 },
            .{ &b.proj, rows * cfg.projWidth() * 2 },              .{ &b.windows, rows * cfg.conv_kernel * 4 },
            .{ &b.sid, rows * 4 },                                 .{ &b.conv_ptr, 16 },
            .{ &b.gq, rows * cfg.nk * cfg.dk * 4 },                .{ &b.gk, rows * cfg.nk * cfg.dk * 4 },
            .{ &b.gv, rows * cfg.nv * cfg.dv * 2 },                .{ &b.gg, rows * cfg.nv * 4 },
            .{ &b.gb, rows * cfg.nv * 4 },                         .{ &b.gy, rows * cfg.nv * cfg.dv * 2 },
            .{ &b.pa, rows * aw * 2 },                             .{ &b.q, rows * cfg.heads * cfg.head_dim * 2 },
            .{ &b.iq, rows * cfg.index_heads * cfg.index_dim * 2 }, .{ &b.attn_o, rows * cfg.heads * cfg.head_dim * 2 },
            .{ &b.gated, rows * cfg.heads * cfg.head_dim * 2 },    .{ &b.xs_gated, rows * cfg.heads * cfg.head_dim / 32 * 4 },
            .{ &b.pos_blk, 16 },                                   .{ &b.xh, rows * @max(kmax, cfg.wide()) * 2 },
            .{ &b.wq, kn * 2 },                                    .{ &b.hadamard, 128 * 128 * 2 },
            .{ &b.z, zmax },                                       .{ &b.logits_router, rows * (cfg.experts + 1) * 4 },
            .{ &b.pick, rows * slots * 4 },                        .{ &b.wts, rows * slots * 4 },
            .{ &b.y, rows * slots * cfg.hidden * 2 },              .{ &b.ple_dev, rows * ple_heads * ple_words * 2 },
            .{ &b.ple_emb, rows * cfg.hidden * 2 },                .{ &b.ple_keys, rows * wide * 2 },
            .{ &b.ple_vals, rows * cfg.hidden * 2 },               .{ &b.ple_gated, rows * wide * 2 },
            .{ &b.ple_pss, rows * cfg.streams * 4 },               .{ &b.ple_nrow, rows * wide * 2 },
            .{ &b.streams, rows * wide * 2 },                      .{ &b.logits, cfg.vocab * 2 },
        };
        for (sizes) |s| {
            s[0].* = try cuda.DeviceBuffer.alloc(d, s[1]);
            try s[0].fill8(0, null);
        }
        // state.py: windows = arange(rows)[:, None] + arange(taps); the Hadamard matrix of prefill.py `hadamard`
        const win = try gpa.alloc(i32, rows * cfg.conv_kernel);
        defer gpa.free(win);
        for (0..rows) |r| for (0..cfg.conv_kernel) |t| {
            win[r * cfg.conv_kernel + t] = @intCast(r + t);
        };
        try b.windows.upload(0, std.mem.sliceAsBytes(win));
        var h: [128 * 128]u16 = undefined;
        for (0..128) |i| for (0..128) |j| {
            h[i * 128 + j] = if (@popCount(i & j) & 1 == 1) 0xBF80 else 0x3F80; // bf16 -1.0 / 1.0
        };
        try b.hadamard.upload(0, std.mem.sliceAsBytes(&h));
        b.attn = try attn.Scratch.init(d, cfg, @min(rows, att_rows), capacity);
        b.moe = try experts.Scratch.init(d, &m.layers[0].experts, @min(rows, moe_window), slots);
        try b.moe.setMode(d, &m.layers[0].experts, try expertsMode(), rows);
        b.f16_mode = try f16Mode();
        b.scores_mode = try scoresMode();
        // TF_QFN_PROMPT_SELECT: the indexer's block lists of a prompt block of up to fused_select_blocks blocks:
        // `fused` (default) select.cu (the same lists, one program a row), `captured` 0.6.6's `_select`, `check`
        // both, compared. Past them 0.6.6's kernels run in every mode.
        b.select_mode = try modeOr(F16Mode, "TF_QFN_PROMPT_SELECT", .fused);
        b.select_check = null;
        b.f16_check = null;
        b.f16_part = null;
        b.f16_diffs = null;
        b.scores_check = null;
        if (b.f16_mode == .check or b.scores_mode == .check or b.select_mode == .check) b.f16_diffs = try cuda.DeviceBuffer.alloc(d, 8);
        if (b.select_mode == .check) b.select_check = try cuda.DeviceBuffer.alloc(d, (att_rows * b.attn.idw + 2 * att_rows) * 4);
        if (b.f16_mode == .check) b.f16_check = try cuda.DeviceBuffer.alloc(d, rows * wide * 2);
        if (b.f16_mode != .captured) {
            var most: usize = 0;
            for (m.layers) |*l| {
                for ([_]*const weights.F16{ &l.attn_hc.down, &l.attn_hc.up, &l.mlp_hc.down, &l.mlp_hc.up }) |x| most = @max(most, x.sk * x.n);
                if (l.ple) |*pl| {
                    for ([_]*const weights.F16{ &pl.key, &pl.value }) |x| most = @max(most, x.sk * x.n);
                }
                switch (l.mixer) {
                    .gdn => |*gl| {
                        most = @max(most, gl.w.ba.sk * gl.w.ba.n);
                    },
                    .attn => {},
                }
            }
            most = @max(most, @max(m.mixer.down.sk * m.mixer.down.n, m.mixer.up.sk * m.mixer.up.n));
            b.f16_part = try cuda.DeviceBuffer.alloc(d, rows * most * 4);
        }
        if (b.scores_mode == .check) b.scores_check = try cuda.DeviceBuffer.alloc(d, b.attn.scores.len);
        b.ple_host = try gpa.alloc(u8, rows * ple_heads * ple_words * 2);
        return b;
    }

    pub fn deinit(self: *Buffers) void {
        self.w.deinit();
        for ([_]*cuda.DeviceBuffer{ &self.dn, &self.ids, &self.proj, &self.windows, &self.sid, &self.conv_ptr, &self.gq, &self.gk, &self.gv, &self.gg, &self.gb, &self.gy, &self.pa, &self.q, &self.iq, &self.attn_o, &self.gated, &self.xs_gated, &self.pos_blk, &self.xh, &self.wq, &self.hadamard, &self.z, &self.logits_router, &self.pick, &self.wts, &self.y, &self.ple_dev, &self.ple_emb, &self.ple_keys, &self.ple_vals, &self.ple_gated, &self.ple_pss, &self.ple_nrow, &self.streams, &self.logits }) |x| x.free();
        if (self.f16_check) |*x| x.free();
        if (self.f16_part) |*x| x.free();
        if (self.scores_check) |*x| x.free();
        if (self.select_check) |*x| x.free();
        if (self.f16_diffs) |*x| x.free();
        self.attn.deinit();
        self.moe.deinit();
        self.gpa.free(self.ple_host);
    }
};

/// Hooks a test uses to compare a chunk's layer boundaries (all optional).
pub const Probe = struct {
    ctx: ?*anyopaque = null,
    /// before layer `l`: the streams `h` (its pending write-back not yet applied)
    layer_in: ?*const fn (ctx: ?*anyopaque, l: usize, h: cuda.DeviceBuffer, rows: usize) anyerror!void = null,
    /// after layer `l`: its pending expert slots (bf16), weights and gates
    layer_out: ?*const fn (ctx: ?*anyopaque, l: usize, y: cuda.DeviceBuffer, wts: cuda.DeviceBuffer, inj: cuda.DeviceBuffer, rows: usize) anyerror!void = null,
};

pub const Prompt = struct {
    f: *fwd.Forward,
    // the prompt path's glue: the forward's, without the decode's f16_rows routing (prompt chunks pick their own
    // F16 kernels, b.f16_mode)
    g: Glue,
    k: Kernels,

    /// The prompt path's kernels; the decode forward `f` borrows f16_rows.cu for its wider windows (Glue.f16_rows)
    /// unless TF_QFN_DECODE_F16 is "captured".
    pub fn init(f: *fwd.Forward, sms: u32) !Prompt {
        var p: Prompt = .{ .f = f, .g = f.g, .k = try Kernels.load(f.d, sms) };
        p.g.f16_rows = null;
        const captured = if (std.c.getenv("TF_QFN_DECODE_F16")) |v| std.mem.eql(u8, std.mem.span(v), "captured") else false;
        if (!captured) f.g.f16_rows = .{ .bf16 = p.k.f16_bf16, .fp16 = p.k.f16_fp16 };
        return p;
    }

    pub fn deinit(self: *Prompt) void {
        self.k.deinit();
    }

    /// exl3/prefill.py `matmul`: out [m, n] (row stride o_stride) = x [m, k] bf16 @ W for any m.
    pub fn matmul(p: *Prompt, l: *const Linear, x: Ptr, m: usize, out: Ptr, o_stride: usize, b: *Buffers, stream: cuda.Stream) !void {
        const xk = &p.f.m.xk;
        {
            var a: cuda.Args = .{};
            a.add(x);
            a.add(@as(c_int, @intFromEnum(dsv41.exl3.DType.bf16)));
            a.add(l.suh.ptr);
            a.add(b.xh.ptr);
            a.add(@as(c_int, @intCast(l.k)));
            try cuda.launch.launch(xk.rot_in, .{ .grid = .{ .x = @intCast((l.k / 128 + 3) / 4), .y = @intCast(m) }, .block = .{ .x = 128 } }, stream, &a);
        }
        {
            const s = l.strides();
            var a: cuda.Args = .{};
            a.add(l.words.ptr);
            a.add(b.wq.ptr);
            a.add(@as(c_int, @intCast(l.n)));
            a.add(s.k);
            a.add(s.nb);
            const f = try xk.unpack(l.k2, l.cb);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(l.n / 16), .y = @intCast(l.k / 16) }, .block = .{ .x = 32 } }, stream, &a);
        }
        try p.g.gemm(b.xh.ptr, b.wq.ptr, b.hadamard.ptr, l.svh.ptr, out, o_stride, m, l.k, l.n, stream);
    }

    /// exl3_mm.py `F16.__call__` on a chunk's rows (cuda_glue.zig `f16Rows`): out [m, w.n] bf16 (row stride o_stride)
    /// = x [m, w.k] (bf16 or fp16, row stride x_stride) @ w^T, on the kernels b.f16_mode names.
    pub fn f16Rows(p: *Prompt, w: *const weights.F16, x: Ptr, x_ty: []const u8, x_stride: usize, m: usize, out: Ptr, o_stride: usize, b: *Buffers, stream: cuda.Stream) !void {
        const fp16 = std.mem.eql(u8, x_ty, "*fp16");
        if (b.f16_mode == .captured or (!fp16 and !std.mem.eql(u8, x_ty, "*bf16")) or (w.k / w.sk) % 64 != 0) return p.g.f16Rows(w, x, x_ty, x_stride, m, out, o_stride, b.w.part.ptr, stream);
        const fused_out = if (b.f16_mode == .check) b.f16_check.?.ptr else out;
        const fused_stride = if (b.f16_mode == .check) w.n else o_stride;
        const part = b.f16_part.?;
        if (w.sk > 1 and w.sk * m * w.n * 4 > part.len) return error.F16PartTooSmall;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intCast(x_stride)));
        a.add(w.w.ptr);
        a.add(fused_out);
        a.add(@as(c_int, @intCast(fused_stride)));
        a.add(part.ptr);
        for ([_]usize{ m, w.n, w.k, w.k / w.sk }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(if (fp16) p.k.f16_fp16 else p.k.f16_bf16, .{ .grid = .{ .x = @intCast((m + 127) / 128), .y = @intCast((w.n + 127) / 128), .z = @intCast(w.sk) }, .block = .{ .x = 256 } }, stream, &a);
        if (w.sk > 1) {
            var r: cuda.Args = .{};
            r.add(part.ptr);
            r.add(@as(c_int, @intCast(w.sk)));
            r.add(@as(c_int, @intCast(m)));
            r.add(@as(c_int, @intCast(w.n)));
            r.add(fused_out);
            r.add(@as(c_int, @intCast(fused_stride)));
            try cuda.launch.launch(p.k.f16_reduce, .{ .grid = .{ .x = @intCast((m * w.n + 255) / 256) }, .block = .{ .x = 256 } }, stream, &r);
        }
        if (b.f16_mode != .check) return;
        try p.g.f16Rows(w, x, x_ty, x_stride, m, out, o_stride, b.w.part.ptr, stream);
        const diffs = b.f16_diffs.?;
        try diffs.fill8(0, stream.handle);
        var c: cuda.Args = .{};
        c.add(out);
        c.add(@as(c_int, @intCast(o_stride)));
        c.add(fused_out);
        c.add(@as(c_int, @intCast(fused_stride)));
        c.add(@as(c_int, @intCast(m)));
        c.add(@as(c_int, @intCast(w.n)));
        c.add(diffs.ptr);
        try cuda.launch.launch(p.k.f16_diff, .{ .grid = .{ .x = 1024 }, .block = .{ .x = 256 } }, stream, &c);
        var n: u64 = 0;
        try diffs.downloadAsync(0, std.mem.asBytes(&n), stream.handle);
        try stream.synchronize();
        if (n != 0) {
            std.debug.print("qwen38fn F16 check: {d} of {d} values differ (k {d}, n {d}, {d} slices, {d} rows)\n", .{ n, m * w.n, w.k, w.n, w.sk, m });
            return error.F16Differ;
        }
    }

    /// forward.py `_readout_plain` on the prompt path: the down projection summed (F16 call), `hc_act`, up, mix.
    fn readout(p: *Prompt, b: *Buffers, h: Ptr, hc: *const weights.Hc, R: usize, inject: Ptr, stream: cuda.Stream) !void {
        const g = p.g;
        const cfg = &p.f.m.cfg;
        const w = &b.w;
        const wide = cfg.wide();
        try g.hcNormed(h, w.pss.ptr, hc.scale.ptr, w.normed.ptr, w.xs_normed.ptr, cfg.eps, R, stream);
        try p.f16Rows(&hc.down, w.normed.ptr, "*bf16", wide, R, b.dn.ptr, hc.down.n, b, stream);
        try g.hcAct(b.dn.ptr, hc.down.n, w.act.ptr, w.xs_act.ptr, inject, R, stream);
        try p.f16Rows(&hc.up, w.act.ptr, "*bf16", cfg.low, R, w.up.ptr, wide, b, stream);
        try g.hcMix(w.up.ptr, w.normed.ptr, w.mixed.ptr, w.xs_mixed.ptr, R, stream);
    }

    fn block(p: *Prompt, b: *Buffers, hc: *const weights.Hc, R: usize, pending: Glue.Pending, inject: Ptr, stream: cuda.Stream) !void {
        try p.g.hcWriteback(b.w.h.ptr, b.w.pss.ptr, R, pending, stream);
        try p.readout(b, b.w.h.ptr, hc, R, if (hc.inject) inject else 0, stream);
    }

    /// forward.py `stage` for a prompt chunk: its ids and n-gram rows.
    pub fn stage(p: *Prompt, st: *fwd.Stream, b: *Buffers, tokens: []const i32, stream: cuda.Stream) !void {
        const R = tokens.len;
        if (R > b.rows) return error.WindowTooLarge;
        try b.ids.uploadAsync(0, std.mem.sliceAsBytes(tokens), stream.handle);
        const m = p.f.m;
        if (m.ng) |*g| {
            const heads = g.heads();
            const gpa = b.gpa;
            const toks = try gpa.alloc(i64, R);
            defer gpa.free(toks);
            for (tokens, toks) |t, *x| x.* = t;
            const row_ids = try gpa.alloc(i64, R * heads);
            defer gpa.free(row_ids);
            try g.idsAny(gpa, st.history[0 .. g.n - 1], toks, row_ids);
            const t = &m.table.?;
            const bytes = b.ple_host[0 .. R * heads * t.words * 2];
            try t.gather(row_ids, bytes);
            try b.ple_dev.uploadAsync(0, bytes, stream.handle);
            // the history after the chunk (prompt chunks keep every row)
            var hist = st.history;
            g.advance(hist[0 .. g.n - 1], toks);
            st.prompt_history = hist;
        }
    }

    /// The chunk's forward on staged rows: every layer, b.streams (the streams with the last write-back, for the MTP
    /// head), and with `head` the last row's logits into b.logits.
    pub fn compute(p: *Prompt, st: *fwd.Stream, b: *Buffers, R: usize, head: bool, probe: Probe, stream: cuda.Stream) !void {
        const f = p.f;
        const m = f.m;
        const g = f.g;
        const w = &b.w;
        try g.embed(b.ids.ptr, m.embed.ptr, w.h.ptr, R, stream);
        var pending: Glue.Pending = .none;
        for (m.layers) |*layer| {
            if (probe.layer_in) |cb| {
                try stream.synchronize();
                try cb(probe.ctx, layer.index, w.h, R);
            }
            const at: ?fwd.AttnAt = switch (layer.mixer) {
                .attn => |*al| st.mainAt(al.ai),
                .gdn => null,
            };
            try p.layerStep(layer, st, b, R, pending, at, stream);
            if (probe.layer_out) |cb| {
                try stream.synchronize();
                try cb(probe.ctx, layer.index, b.y, b.wts, w.inj_m, R);
            }
            pending = p.moePending(b);
        }
        try p.finish(&m.mixer, b, R, pending, stream);
        if (!head) return;
        const hl = if (m.head) |*x| x else return error.NoHead;
        try hl.run(w.mixed.ptr, .bf16, 1, b.xh.ptr, b.z.ptr, b.logits.ptr, .bf16, stream);
    }

    pub fn moePending(p: *Prompt, b: *Buffers) Glue.Pending {
        return .{ .moe_bf16 = .{ .y = b.y.ptr, .wts = b.wts.ptr, .inject = b.w.inj_m.ptr, .slots = p.f.m.cfg.slots() } };
    }

    /// forward.py `finish(ends=())` on the prompt path: b.streams = every row's streams with the last write-back, and
    /// the read-out of `mixer` on the last row only into b.w.mixed (row 0).
    pub fn finish(p: *Prompt, mixer: *const weights.Hc, b: *Buffers, R: usize, pending: Glue.Pending, stream: cuda.Stream) !void {
        const cfg = &p.f.m.cfg;
        const w = &b.w;
        try b.streams.copyFrom(0, w.h.ptr, R * cfg.wide() * 2, stream.handle);
        try p.g.hcWriteback(b.streams.ptr, w.pss.ptr, R, pending, stream);
        const last = R - 1;
        const pss_row = (cfg.hidden / 256) * cfg.streams * 4;
        try w.pss.copyFrom(0, w.pss.ptr + last * pss_row, pss_row, stream.handle);
        try p.readout(b, b.streams.ptr + last * cfg.wide() * 2, mixer, 1, 0, stream);
    }

    /// forward.py `layer_forward` on the prompt path for one stream (attention at `at`); the write-back stays pending.
    pub fn layerStep(p: *Prompt, layer: *const fwd.Layer, st: *fwd.Stream, b: *Buffers, R: usize, pending_in: Glue.Pending, at: ?fwd.AttnAt, stream: cuda.Stream) !void {
        const f = p.f;
        const m = f.m;
        const cfg = &m.cfg;
        const g = f.g;
        const w = &b.w;
        const lin = m.linearLayers();
        var pending = pending_in;
        if (layer.ple) |*pl| {
            if (pending != .none) {
                try g.hcWriteback(w.h.ptr, w.pss.ptr, R, pending, stream);
                pending = .none;
            }
            try p.pleBlock(pl, st, b, R, stream);
        }
        try p.block(b, &layer.attn_hc, R, pending, w.inj_a.ptr, stream);
        switch (layer.mixer) {
            .gdn => |*gl| {
                const pw = cfg.projWidth();
                // Stack.prefill: each part straight into its columns
                try p.matmul(&gl.qkv, w.mixed.ptr, R, b.proj.ptr, pw, b, stream);
                try p.matmul(&gl.z, w.mixed.ptr, R, b.proj.ptr + gl.qkv.n * 2, pw, b, stream);
                try p.f16Rows(&gl.w.ba, w.mixed.ptr, "*bf16", cfg.hidden, R, b.proj.ptr + (gl.qkv.n + gl.z.n) * 2, pw, b, stream);
                try p.chain(st, b, gl, lin, R, stream);
                try p.matmul(&gl.out, w.gout.ptr, R, w.branch.ptr, cfg.hidden, b, stream);
            },
            .attn => |*al| {
                const a = at orelse return error.NoAttentionCache;
                const pw = al.w.projWidth();
                var col: usize = 0;
                for (&al.parts) |*part| {
                    try p.matmul(part, w.mixed.ptr, R, b.pa.ptr + col * 2, pw, b, stream);
                    col += part.n;
                }
                try p.attention(a, b, al, pw, R, stream);
                try p.matmul(&al.o, b.gated.ptr, R, w.branch.ptr, cfg.hidden, b, stream);
            },
        }
        try p.block(b, &layer.mlp_hc, R, .{ .branch = .{ .branch = w.branch.ptr, .inject = w.inj_a.ptr } }, w.inj_m.ptr, stream);
        try p.moe(layer, b, R, stream);
    }

    /// forward.py `_prefill_chain` (no kept point): front, the chunk chain from the committed state, back; the layer
    /// commits at once (its state slot flips, its conv window shifts past the chunk).
    fn chain(p: *Prompt, st: *fwd.Stream, b: *Buffers, gl: *const fwd.GdnLayer, lin: usize, R: usize, stream: cuda.Stream) !void {
        const m = p.f.m;
        const cfg = &m.cfg;
        const cur = st.cur[gl.li];
        const conv = st.convPtr(cfg, gl.li);
        try b.conv_ptr.uploadAsync(0, std.mem.asBytes(&conv), stream.handle);
        {
            var a: cuda.Args = .{};
            for ([_]Ptr{ b.proj.ptr, b.conv_ptr.ptr, b.sid.ptr, b.windows.ptr, gl.w.conv.ptr, gl.w.a_log.ptr, gl.w.dt_bias.ptr, b.gq.ptr, b.gk.ptr, b.gv.ptr, b.gg.ptr, b.gb.ptr }) |x| a.add(x);
            try cuda.launch.launch(p.k.front, .{ .grid = .{ .x = @intCast(R), .y = @intCast(cfg.nk + cfg.nv) }, .block = .{ .x = 128 } }, stream, &a);
        }
        {
            const wide = cfg.nv >= p.k.sms;
            const rows_blk: usize = if (wide) 128 else 64;
            var a: cuda.Args = .{};
            for ([_]Ptr{ b.gq.ptr, b.gk.ptr, b.gv.ptr, b.gg.ptr, b.gb.ptr, st.recPtr(cfg, lin, gl.li, cur), st.recPtr(cfg, lin, gl.li, 1 - cur), b.gy.ptr }) |x| a.add(x);
            for ([_]usize{ R, cfg.nk, cfg.nv }) |v| a.add(@as(c_int, @intCast(v)));
            try cuda.launch.launch(if (wide) p.k.chain128 else p.k.chain64, .{ .grid = .{ .x = @intCast(cfg.nv), .y = @intCast(cfg.dv / rows_blk) }, .block = .{ .x = @intCast(2 * rows_blk) } }, stream, &a);
        }
        {
            var a: cuda.Args = .{};
            for ([_]Ptr{ b.gy.ptr, b.proj.ptr, gl.w.norm.ptr }) |x| a.add(x);
            a.add(cfg.eps);
            a.add(b.w.gout.ptr);
            a.add(b.w.gxs.ptr);
            try cuda.launch.launch(p.k.back, .{ .grid = .{ .x = @intCast(R), .y = @intCast(cfg.nv) }, .block = .{ .x = 128 } }, stream, &a);
        }
        st.cur[gl.li] = 1 - cur;
        const taps = cfg.conv_kernel - 1;
        try p.g.shiftWindows(conv, taps * cfg.convDim(), b.proj.ptr, b.rows * cfg.projWidth(), cfg.projWidth(), R, 1, cfg.convDim(), taps, stream);
    }

    /// attention.py `qsa_rows` for a prompt block of n rows at b.pos_blk (keys up to `context`): the scores on the
    /// kernels b.scores_mode names, then the selection on b.select_mode's.
    fn rowsSelect(p: *Prompt, iq: Ptr, c: *const attn.Caches, b: *Buffers, n: usize, context: usize, stream: cuda.Stream) !void {
        const ak = p.f.ak;
        const s = &b.attn;
        const blocks = s.blocks(context);
        if (b.scores_mode == .captured) {
            try ak.scores(iq, c, b.pos_blk.ptr, s, s.scores.ptr, n, blocks, stream);
        } else {
            const check = b.scores_mode == .check;
            const sc = if (check) b.scores_check.? else s.scores;
            if (check) {
                try sc.copyFrom(0, s.scores.ptr, s.scores.len, stream.handle);
                try ak.scores(iq, c, b.pos_blk.ptr, s, s.scores.ptr, n, blocks, stream);
            }
            {
                var a: cuda.Args = .{};
                a.add(iq);
                a.add(c.pooled.ptr);
                a.add(b.pos_blk.ptr);
                a.add(sc.ptr);
                a.add(@as(c_int, @intCast(s.nb)));
                a.add(@as(c_int, @intCast(n)));
                try p.k.sc_rows.allowDynamicShared(scores_shared);
                try cuda.launch.launch(p.k.sc_rows, .{ .grid = .{ .x = @intCast((n + 31) / 32), .y = @intCast((blocks + 63) / 64) }, .block = .{ .x = 256 }, .shared = scores_shared }, stream, &a);
            }
            if (check) {
                var a: cuda.Args = .{};
                a.add(s.scores.ptr);
                a.add(sc.ptr);
                a.add(@as(c_int, @intCast(s.nb)));
                a.add(@as(c_int, @intCast(n)));
                a.add(@as(c_int, @intCast(blocks)));
                try countDiffs(p.k.sc_diff, b, &a, .{ .x = 1024 }, "scores", n * blocks, stream);
            }
        }
        // select.cu up to fused_select_blocks blocks; past them 0.6.6's `_select` and `_select_tiles` are faster
        const tiles = blocks > fused_select_blocks;
        if (b.select_mode == .captured or tiles) return ak.select(b.pos_blk.ptr, s, n, blocks, false, stream);
        const check = b.select_mode == .check;
        if (check) try ak.select(b.pos_blk.ptr, s, n, blocks, false, stream);
        const out = if (check) b.select_check.? else s.ids;
        const nk = if (check) out.ptr + att_rows * s.idw * 4 else s.nk.ptr;
        const sparse = if (check) nk + att_rows * 4 else s.sparse.ptr;
        {
            var a: cuda.Args = .{};
            a.add(s.scores.ptr);
            a.add(b.pos_blk.ptr);
            a.add(out.ptr);
            a.add(nk);
            a.add(sparse);
            for ([_]usize{ s.nb, s.ratio, s.budget / s.ratio, s.idw }) |v| a.add(@as(c_int, @intCast(v)));
            try cuda.launch.launch(p.k.sel_rows, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, stream, &a);
        }
        if (check) {
            var a: cuda.Args = .{};
            for ([_]Ptr{ s.ids.ptr, s.nk.ptr, s.sparse.ptr, out.ptr, nk, sparse }) |v| a.add(v);
            a.add(@as(c_int, @intCast(n)));
            a.add(@as(c_int, @intCast(s.idw)));
            try countDiffs(p.k.sel_diff, b, &a, .{ .x = @intCast(n) }, "selection rows", n, stream);
        }
    }

    /// Check modes: `f` (a diff kernel whose last argument is the count) over `a`, then error.PromptCheckDiffers
    /// unless the count is 0.
    fn countDiffs(f: cuda.Function, b: *Buffers, a: *cuda.Args, grid: cuda.launch.Dim3, what: []const u8, of: usize, stream: cuda.Stream) !void {
        const diffs = b.f16_diffs.?;
        try diffs.fill8(0, stream.handle);
        a.add(diffs.ptr);
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, stream, a);
        var nd: u64 = 0;
        try diffs.downloadAsync(0, std.mem.asBytes(&nd), stream.handle);
        try stream.synchronize();
        if (nd != 0) {
            std.debug.print("qwen38fn prompt check: {d} of {d} {s} differ\n", .{ nd, of, what });
            return error.PromptCheckDiffers;
        }
    }

    /// forward.py `attn_block` on the prompt path for one stream.
    fn attention(p: *Prompt, a: fwd.AttnAt, b: *Buffers, al: *const fwd.AttnLayer, pw: usize, R: usize, stream: cuda.Stream) !void {
        const m = p.f.m;
        const ak = p.f.ak;
        const c = a.caches;
        try ak.prep(b.pa.ptr, pw, a.pos_dev, al.w.q_scale.ptr, al.w.k_scale.ptr, al.w.iq_scale.ptr, m.inv.ptr, b.q.ptr, c, b.iq.ptr, R, stream);
        if (b.attn.qsa) try ak.pool(c, a.pos_dev, al.w.ik_scale.ptr, m.inv.ptr, &b.attn, R, stream);
        const cfg = &m.cfg;
        var r0: usize = 0;
        while (r0 < R) : (r0 += att_rows) {
            const n = @min(att_rows, R - r0);
            const p0: i32 = @intCast(a.pos + r0);
            try b.pos_blk.uploadAsync(0, std.mem.asBytes(&p0), stream.handle);
            const ends = a.pos + r0 + n;
            const iq = b.iq.ptr + r0 * cfg.index_heads * cfg.index_dim * 2;
            const q = b.q.ptr + r0 * cfg.heads * cfg.head_dim * 2;
            if (b.attn.qsa) try p.rowsSelect(iq, c, b, n, ends, stream);
            try ak.attendInto(q, c, b.pos_blk.ptr, &b.attn, n, ends, b.attn_o.ptr + r0 * cfg.heads * cfg.head_dim * 2, stream);
        }
        try ak.gate(b.attn_o.ptr, b.pa.ptr, pw, b.gated.ptr, b.xs_gated.ptr, R, stream);
    }

    /// forward.py `_exl3_moe` on the prompt path: router, picks, then the experts in 1,024-row windows whose fp32 slots
    /// are kept as bf16 (`buf.y[r0:r0 + n].copy_(y)`).
    fn moe(p: *Prompt, layer: *const fwd.Layer, b: *Buffers, R: usize, stream: cuda.Stream) !void {
        const m = p.f.m;
        const cfg = &m.cfg;
        const g = p.g;
        const w = &b.w;
        try g.router(w.mixed.ptr, cfg.hidden, layer.router.ptr, cfg.experts + 1, b.logits_router.ptr, R, stream);
        try g.selectRows(b.logits_router.ptr, cfg.experts, cfg.topk, b.pick.ptr, b.wts.ptr, R, stream);
        const slots = cfg.slots();
        // fused: the chunk in one window; check: that, then the captured windows, compared
        switch (b.moe.mode) {
            .captured => {},
            .fused => return experts.promptWindow(&m.ek, &layer.experts, &b.moe, w.mixed.ptr, cfg.hidden, b.pick.ptr, R, b.y.ptr, stream),
            .check => try experts.promptWindow(&m.ek, &layer.experts, &b.moe, w.mixed.ptr, cfg.hidden, b.pick.ptr, R, b.moe.ycheck.?.ptr, stream),
        }
        var r0: usize = 0;
        while (r0 < R) : (r0 += moe_window) {
            const n = @min(moe_window, R - r0);
            const y = try experts.routed(&m.ek, &layer.experts, &b.moe, w.mixed.ptr + r0 * cfg.hidden * 2, cfg.hidden, b.pick.ptr + r0 * slots * 4, n, stream);
            if (b.moe.mode == .check) try experts.checkSwiglu(&m.ek, &b.moe, r0 * slots, n * slots, stream);
            const count: i64 = @intCast(n * slots * cfg.hidden);
            var a: cuda.Args = .{};
            a.add(y);
            a.add(b.y.ptr + r0 * slots * cfg.hidden * 2);
            a.add(@as(c_longlong, count));
            try cuda.launch.launch(p.k.to_bf16, .{ .grid = .{ .x = @intCast((@as(usize, @intCast(count)) + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
        }
        if (b.moe.mode == .check) try experts.checkWindow(&m.ek, &b.moe, b.y.ptr, R * slots * cfg.hidden * 2, stream);
    }

    fn pleBlock(p: *Prompt, pl: *const weights.Ple, st: *fwd.Stream, b: *Buffers, R: usize, stream: cuda.Stream) !void {
        const m = p.f.m;
        const cfg = &m.cfg;
        const g = p.g;
        const t = &m.table.?;
        const ng = m.ng.?;
        try g.pleRows(b.ple_dev.ptr, t.words, pl.head_bias.ptr, ng.heads(), t.dh, t.bits, b.ple_emb.ptr, R, stream);
        try p.f16Rows(&pl.key, b.ple_emb.ptr, "*fp16", cfg.hidden, R, b.ple_keys.ptr, cfg.wide(), b, stream);
        try p.f16Rows(&pl.value, b.ple_emb.ptr, "*fp16", cfg.hidden, R, b.ple_vals.ptr, cfg.hidden, b, stream);
        try g.pleGate(b.ple_keys.ptr, b.ple_vals.ptr, b.w.h.ptr, pl.norm_key.ptr, pl.norm_query.ptr, b.ple_gated.ptr, b.ple_pss.ptr, cfg.eps, R, stream);
        try g.pleConv(b.ple_gated.ptr, b.ple_pss.ptr, pl.norm_conv.ptr, st.ple_tail.ptr, pl.conv.ptr, pl.taps, cfg.ngram, b.w.h.ptr, b.ple_nrow.ptr, cfg.eps, R, stream);
    }

    /// forward.py `commit` for a prompt chunk (every row kept; the DeltaNet layers committed during the chunk): the
    /// n-gram history and tail, the position.
    pub fn commit(p: *Prompt, st: *fwd.Stream, b: *Buffers, R: usize, stream: cuda.Stream) !void {
        const m = p.f.m;
        const cfg = &m.cfg;
        if (m.ng != null) {
            st.history = st.prompt_history;
            const t_rows = (cfg.ple_kernel - 1) * cfg.ngram;
            try p.g.shiftWindows(st.ple_tail.ptr, t_rows * cfg.wide(), b.ple_nrow.ptr, b.rows * cfg.wide(), cfg.wide(), R, 1, cfg.wide(), t_rows, stream);
        }
        st.pos += R;
        const pos: i32 = @intCast(st.pos);
        try st.pos_dev.uploadAsync(0, std.mem.asBytes(&pos), stream.handle);
    }
};
