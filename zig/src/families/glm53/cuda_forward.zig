//! GLM-5.3-Flash's layer blocks on one rank, in the Mia fork's `forward.py` call order (strict setting): each block
//! ends in this rank's fp32 partial [R, D] (`Buffers.part`), which the ranks all-gather and `hcPost` adds to the
//! streams rank 0 first. Rows never mix: row r of a window gets the bits it gets alone, so a decode window, a
//! drafted window and a prompt chunk differ only in which kernels the reference picks for them (`prefill`).
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const glue_mod = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const weights = @import("cuda_weights.zig");

const Glue = glue_mod.Glue;
const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

fn alloc(d: *const cuda.Driver, n: usize) !Buf {
    return Buf.alloc(d, @max(n, 16));
}

/// forward.py `Buffers` for windows of up to `rows` rows (`prefill`: a prompt chunk's buffers).
pub const Buffers = struct {
    rows: usize,
    prefill: bool,
    ids: Buf, // [rows] i32
    x: Buf, // [rows, S D] bf16: the residual streams
    normed: Buf, // [rows, D] bf16
    xs: Buf, // [rows, D / 64] fp32
    post: Buf, // [rows, S] fp32
    comb: Buf, // [rows, S S] fp32
    hcpart: Buf, // [rows, 16, 32] fp32
    ka: Buf, // [rows, L 128] bf16: f_b's output (the decay rows)
    kg: Buf, // [rows, L 128] bf16: g_b's output (the gate rows)
    kproj: Buf, // [rows, KDA projection width] bf16 (a decode window's: several streams' rows, `kdaSegs`)
    kout: Buf, // decode: [rows, L 128] bf16, several streams' chain outputs (`kdaSegs`)
    gu: Buf, // [rows, 2 W_dense] bf16
    act: Buf, // [rows, W_dense] bf16
    xs_act: Buf, // [rows, W_dense / 64] fp32
    part: Buf, // [rows, D] fp32: this rank's partial
    gath: Buf, // [world, rows, D] fp32: every rank's partials
    sk: Buf, // split-K partials of the BF16 matmuls
    hidden: Buf, // [rows, D] bf16: the streams' mean

    pub fn init(d: *const cuda.Driver, lay: config.Layout, rows: usize, prefill: bool) !Buffers {
        const c = lay.cfg;
        const D = c.hidden;
        const S = c.streams;
        const LW = lay.size(.lin) * c.lin_dim;
        const W = lay.size(.dense);
        var b: Buffers = undefined;
        b.rows = rows;
        b.prefill = prefill;
        b.ids = try alloc(d, rows * 4);
        b.x = try alloc(d, rows * S * D * 2);
        b.normed = try alloc(d, rows * D * 2);
        b.xs = try alloc(d, rows * D / 64 * 4);
        b.post = try alloc(d, rows * S * 4);
        b.comb = try alloc(d, rows * S * S * 4);
        b.hcpart = try alloc(d, rows * glue_mod.hc_blocks * 32 * 4);
        b.ka = try alloc(d, rows * LW * 2);
        b.kg = try alloc(d, rows * LW * 2);
        b.kproj = try alloc(d, rows * lay.kdaProj() * 2);
        b.kout = try alloc(d, if (prefill) 0 else rows * LW * 2);
        b.gu = try alloc(d, rows * 2 * W * 2);
        b.act = try alloc(d, rows * W * 2);
        b.xs_act = try alloc(d, rows * W / 64 * 4);
        b.part = try alloc(d, rows * D * 4);
        b.gath = try alloc(d, lay.world * rows * D * 4);
        // the largest split-K product of a BF16 matmul a window runs: 8 slices of a D-wide output
        b.sk = try alloc(d, 8 * rows * D * 4);
        b.hidden = try alloc(d, rows * D * 2);
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        inline for (.{ "ids", "x", "normed", "xs", "post", "comb", "hcpart", "ka", "kg", "kproj", "kout", "gu", "act", "xs_act", "part", "gath", "sk", "hidden" }) |f| @field(b, f).free();
    }
};

/// One stream's KDA state over `layers` KDA layers (forward.py `Slots` for one slot): recurrent states of both
/// parities [2, layers, H, 128, 128] fp32, conv windows [layers, 3, C] bf16, the decode window's projection rows
/// [layers, rows, width] bf16 (the commit's conv shift reads them) and its replay scratch.
pub const KdaState = struct {
    rec: Buf,
    conv: Buf,
    proj: Buf,
    scratch: kda.Scratch,
    layers: usize,
    heads: usize,
    rows: usize,
    width: usize,
    cur: u1 = 0,
    pos: usize = 0,

    pub fn init(d: *const cuda.Driver, lay: config.Layout, layers: usize, rows: usize) !KdaState {
        const H = lay.size(.lin);
        const C = 3 * H * kda.dk;
        var rec = try Buf.alloc(d, 2 * layers * H * kda.dv * kda.dk * 4);
        errdefer rec.free();
        var conv = try Buf.alloc(d, layers * 3 * C * 2);
        errdefer conv.free();
        var proj = try Buf.alloc(d, layers * rows * lay.kdaProj() * 2);
        errdefer proj.free();
        return .{ .rec = rec, .conv = conv, .proj = proj, .scratch = try kda.Scratch.init(d, layers, rows, H), .layers = layers, .heads = H, .rows = rows, .width = lay.kdaProj() };
    }

    pub fn deinit(s: *KdaState) void {
        s.rec.free();
        s.conv.free();
        s.proj.free();
        s.scratch.deinit();
    }

    pub fn stateBytes(s: *const KdaState) usize {
        return s.heads * kda.dv * kda.dk * 4;
    }

    /// Layer `li`'s state of parity `p`.
    pub fn state(s: *const KdaState, p: u1, li: usize) Ptr {
        return s.rec.ptr + (@as(usize, p) * s.layers + li) * s.stateBytes();
    }

    pub fn convOf(s: *const KdaState, li: usize) Ptr {
        return s.conv.ptr + li * 3 * 3 * s.heads * kda.dk * 2;
    }
};

/// What a block needs besides its weights: the glue, the KDA kernels and their temporaries.
pub const Ctx = struct {
    g: Glue,
    kk: *const kda.Kernels,
    temp: *const kda.Temp,
    prompt_scratch: *const kda.Scratch, // a prompt chunk's KDA scratch (one layer of `prompt rows`)
    lay: config.Layout,
};

/// glue.hc_pre of the layer's attention (`ffn` false) or FFN site over b.x into b.normed, b.xs, b.post, b.comb.
pub fn hcPre(cx: Ctx, f: *const weights.Frame, ffn: bool, b: *const Buffers, R: usize, stream: cuda.Stream) !void {
    const c = cx.lay.cfg;
    const h = if (ffn) &f.ffn_hc else &f.attn_hc;
    try cx.g.hcPre(.{
        .x = b.x.ptr,               .fn_ = h.fn_.ptr, .base = h.base.ptr, .scale = h.scale.ptr,
        .norm = (if (ffn) f.post_norm else f.in_norm).ptr,
        .out = b.normed.ptr,        .xs = b.xs.ptr,   .post = b.post.ptr, .comb = b.comb.ptr,
        .part = b.hcpart.ptr,       .eps = c.eps,     .hc_eps = c.hc_eps, .iters = c.hc_iters,
        .rows = R,                  .prompt = b.prefill,
    }, stream);
}

/// glue.hc_post of b.gath (every rank's partials) into the streams b.x.
pub fn hcPost(cx: Ctx, b: *const Buffers, R: usize, stream: cuda.Stream) !void {
    try cx.g.hcPost(b.x.ptr, b.x.ptr, b.gath.ptr, cx.lay.world, b.post.ptr, b.comb.ptr, R, stream);
}

/// forward.py `kda_block` up to the all-gather: the projections, the chain on layer `li`'s state, out_proj into
/// b.part. A prompt chunk (b.prefill) commits its layer here (forward.py `kda_rows`): the state flips parity for
/// this layer and its conv window takes the chunk's last rows.
pub fn kdaBlock(cx: Ctx, w: *const weights.Kda, st: *KdaState, li: usize, b: *const Buffers, R: usize, stream: cuda.Stream) !void {
    const c = cx.lay.cfg;
    const width = st.width;
    const offs = cx.lay.kdaOffsets();
    const p: Ptr = if (b.prefill) b.kproj.ptr else st.proj.ptr + li * st.rows * width * 2;
    try cx.g.bmm(b.normed.ptr, c.hidden, w.proj.b16(), R, p, false, b.sk.ptr, stream);
    try cx.g.bmm(p + offs[0] * 2, width, w.fb.b16(), R, b.ka.ptr, false, b.sk.ptr, stream);
    try cx.g.bmm(p + offs[1] * 2, width, w.gb.b16(), R, b.kg.ptr, false, b.sk.ptr, stream);
    const LW = w.heads * kda.dk;
    const sc = if (b.prefill) cx.prompt_scratch.layer(0) else st.scratch.layer(li);
    const cur = st.cur;
    try kda.chainWide(cx.kk, cx.temp, .{
        .heads = w.heads,         .p = p,                         .p_stride = width,              .b_off = offs[2],
        .a = b.ka.ptr,            .a_stride = LW,                 .g = b.kg.ptr,                  .g_stride = LW,
        .conv_state = st.convOf(li), .conv_w = w.conv.ptr,        .state_in = st.state(cur, li),    .a_log = w.a_log.ptr,
        .dt_bias = w.dt_bias.ptr, .norm_w = w.norm.ptr,           .eps = c.eps,                   .lower = c.lower,
        .rows = R,                .out = sc.out,                  .state_out = st.state(1 - cur, li),
        .k = sc.k,                .v = sc.v,                      .gs = sc.g,                     .b = sc.b,
    }, stream);
    if (b.prefill) {
        // kda_rows: this layer commits now (its parity flips at the step's end, `kdaCommit`), the window shifted;
        // the projection's layer stride is the prompt buffer's (forward.py passes b.kproj's stride)
        const C = 3 * LW;
        try cx.g.convShift(st.convOf(li), p, R, 1, C, 3 * C, b.rows * width, width, stream);
    }
    try cx.g.bmm(sc.out, LW, w.o.b16(), R, b.part.ptr, true, b.sk.ptr, stream);
}

/// One stream's rows of a decode window several streams share: rows [a0, a0 + n) of the buffers, the stream's KDA
/// state and its index among the KDA layers.
pub const KdaSeg = struct { st: *KdaState, li: usize, a0: usize, n: usize };

/// `kdaBlock` for a decode window of several streams' rows (`segs`, in row order, R rows in all): the projections and
/// out_proj over every row at once, each stream's chain over its own rows and state, so every row gets the bits of
/// its stream's window alone. Each stream's projection rows land in its state as `kdaBlock`'s do (the commit's conv
/// shift reads them).
pub fn kdaSegs(cx: Ctx, w: *const weights.Kda, segs: []const KdaSeg, b: *const Buffers, R: usize, stream: cuda.Stream) !void {
    if (b.prefill) return error.PromptSegments;
    const c = cx.lay.cfg;
    const width = segs[0].st.width;
    const offs = cx.lay.kdaOffsets();
    try cx.g.bmm(b.normed.ptr, c.hidden, w.proj.b16(), R, b.kproj.ptr, false, b.sk.ptr, stream);
    try cx.g.bmm(b.kproj.ptr + offs[0] * 2, width, w.fb.b16(), R, b.ka.ptr, false, b.sk.ptr, stream);
    try cx.g.bmm(b.kproj.ptr + offs[1] * 2, width, w.gb.b16(), R, b.kg.ptr, false, b.sk.ptr, stream);
    const LW = w.heads * kda.dk;
    for (segs) |sg| {
        const st = sg.st;
        const p: Ptr = st.proj.ptr + sg.li * st.rows * width * 2;
        try st.proj.copyFrom(sg.li * st.rows * width * 2, b.kproj.ptr + sg.a0 * width * 2, sg.n * width * 2, stream.handle);
        const sc = st.scratch.layer(sg.li);
        const cur = st.cur;
        try kda.chainWide(cx.kk, cx.temp, .{
            .heads = w.heads,           .p = p,                          .p_stride = width,                   .b_off = offs[2],
            .a = b.ka.ptr + sg.a0 * LW * 2, .a_stride = LW,              .g = b.kg.ptr + sg.a0 * LW * 2,      .g_stride = LW,
            .conv_state = st.convOf(sg.li), .conv_w = w.conv.ptr,        .state_in = st.state(cur, sg.li),    .a_log = w.a_log.ptr,
            .dt_bias = w.dt_bias.ptr,   .norm_w = w.norm.ptr,            .eps = c.eps,                        .lower = c.lower,
            .rows = sg.n,               .out = b.kout.ptr + sg.a0 * LW * 2, .state_out = st.state(1 - cur, sg.li),
            .k = sc.k,                  .v = sc.v,                       .gs = sc.g,                          .b = sc.b,
        }, stream);
    }
    try cx.g.bmm(b.kout.ptr, LW, w.o.b16(), R, b.part.ptr, true, b.sk.ptr, stream);
}

/// forward.py `commit` for the KDA layers: a decode window keeping `keep` of its R rows replays every layer's state
/// when keep < R, flips the parity and shifts every conv window; a prompt chunk (all rows kept) flips the parity its
/// layers' chains already wrote.
pub fn kdaCommit(cx: Ctx, st: *KdaState, R: usize, keep: usize, prefill: bool, stream: cuda.Stream) !void {
    if (keep == 0 or keep > R or (prefill and keep != R)) return error.BadKeep;
    if (!prefill) {
        if (keep < R) try kda.replayLayers(cx.kk, st.state(st.cur, 0), &st.scratch, keep, st.state(1 - st.cur, 0), stream);
        const C = 3 * st.heads * kda.dk;
        try cx.g.convShift(st.conv.ptr, st.proj.ptr, keep, st.layers, C, 3 * C, st.rows * st.width, st.width, stream);
    }
    st.cur = 1 - st.cur;
    st.pos += keep;
}

/// forward.py `mlp_block` up to the all-gather: [gate | up], the limited SwiGLU, down into b.part.
pub fn mlpBlock(cx: Ctx, w: *const weights.Mlp, b: *const Buffers, R: usize, stream: cuda.Stream) !void {
    const c = cx.lay.cfg;
    try cx.g.bmm(b.normed.ptr, c.hidden, w.gu.b16(), R, b.gu.ptr, false, b.sk.ptr, stream);
    try cx.g.swiglu(b.gu.ptr, b.act.ptr, b.xs_act.ptr, c.limit, R, w.width, stream);
    try cx.g.bmm(b.act.ptr, w.width, w.down.b16(), R, b.part.ptr, true, b.sk.ptr, stream);
}
