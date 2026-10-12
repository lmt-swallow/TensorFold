//! GLM-5.3-Flash's MTP head on one rank, as the Mia fork's `mtp.py: mtp_compute` runs it: the next tokens'
//! embedding (the stream's first position zeroed) and the main model's final-normed rows, each RMS-normed, side by
//! side through eh_proj; then a DSA layer on the head's own caches and a MoE, each added to the row plainly (no
//! hyper-connections, the partials summed rank 0 first: `residual_add`); then shared_head.norm of the last row and
//! the 4-bit draft head (draft_head.zig) over the rank's vocabulary share: decode rows on qmm.cu's kernel, a prompt
//! chunk's last row on qmm_prefill.cu's (other bits; drafts only propose).
//!
//! The steps stop where the ranks all-gather: `front` (to the attention's input), the DSA block (cuda_dsa.zig),
//! `afterAttention` (the residual, the FFN's input), the MoE block (cuda_moe.zig), `head`.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const map = @import("weights_map.zig");
const glue_mod = @import("cuda_glue.zig");
const weights = @import("cuda_weights.zig");
const dsa = @import("cuda_dsa.zig");
const moe = @import("cuda_moe.zig");
const draft_head = @import("draft_head.zig");
const Tr = @import("cuda_tier.zig");

const Glue = glue_mod.Glue;
const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

pub const Kernels = struct {
    dec_mod: cuda.Module,
    pre_mod: cuda.Module,
    dec: cuda.Function, // qmm_kernel<64, 16, 64, 1, 4, 4, bf16 out, no cluster>
    pre: cuda.Function, // prefill_kernel<64, 128, 128, 2, 4, 3, bf16 out>
    dec_smem: u32,
    pre_smem: u32,

    fn smem(d: *const cuda.Driver, m: cuda.Module, name: [:0]const u8) !u32 {
        const g = try m.global(name);
        var v: u64 = 0;
        const view: Buf = .{ .d = d, .ptr = g.ptr, .len = g.len };
        try view.download(0, std.mem.asBytes(&v));
        return @intCast(v);
    }

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var a = try cuda.Module.load(d, cuda.kernels.glm53_qmm);
        errdefer a.unload();
        var b = try cuda.Module.load(d, cuda.kernels.glm53_qmm_prefill);
        errdefer b.unload();
        var k: Kernels = .{
            .dec_mod = a,
            .pre_mod = b,
            .dec = try a.function("_ZN12tf_glm53_qmm10qmm_kernelILi64ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii"),
            .pre = try b.function("_ZN20tf_glm53_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii"),
            .dec_smem = try smem(d, a, "tf_glm53_qmm_smem"),
            .pre_smem = try smem(d, b, "tf_glm53_qmm_prefill_smem"),
        };
        try k.dec.allowDynamicShared(k.dec_smem);
        try k.pre.allowDynamicShared(k.pre_smem);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.dec_mod.unload();
        k.pre_mod.unload();
    }
};

/// The draft head on the device (qmm.py `Q4`): packed words, scales and biases [k / 64, npad] bf16.
pub const DraftHead = struct {
    words: Buf,
    scales: Buf,
    biases: Buf,
    n: usize,
    k: usize,
    npad: usize,

    pub fn load(l: weights.Loader) !DraftHead {
        const rows = try l.headRows();
        defer l.gpa.free(rows);
        const k = l.lay.cfg.hidden;
        var p = try draft_head.quantizePack(l.gpa, std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(rows))), rows.len / 2 / k, k);
        defer p.deinit(l.gpa);
        var w = try Buf.fromHost(l.d, std.mem.sliceAsBytes(p.words));
        errdefer w.free();
        var s = try Buf.fromHost(l.d, std.mem.sliceAsBytes(p.scales));
        errdefer s.free();
        return .{ .words = w, .scales = s, .biases = try Buf.fromHost(l.d, std.mem.sliceAsBytes(p.biases)), .n = p.n, .k = k, .npad = p.npad };
    }

    pub fn deinit(h: *DraftHead) void {
        h.words.free();
        h.scales.free();
        h.biases.free();
    }
};

/// The MTP layer's rank share (weights.py `MTPW`).
pub const Mtp = struct {
    enorm: Buf,
    hnorm: Buf,
    eh: weights.Matrix, // [D, 2 D]
    norm: Buf, // shared_head.norm
    in_norm: Buf,
    post_norm: Buf,
    dsa: dsa.Dsa,
    moe: moe.Moe,
    head: DraftHead,

    pub fn load(l: weights.Loader) !Mtp {
        const i = l.lay.cfg.layers;
        var n: map.Name = .{};
        var m: Mtp = undefined;
        m.enorm = try l.share(map.layer(&n, i, "enorm.weight"));
        errdefer m.enorm.free();
        m.hnorm = try l.share(map.layer(&n, i, "hnorm.weight"));
        errdefer m.hnorm.free();
        m.eh = try l.matrix(&.{map.layer(&n, i, "eh_proj.weight")}, 2 * l.lay.cfg.hidden);
        errdefer m.eh.deinit();
        m.norm = try l.share(map.layer(&n, i, "shared_head.norm.weight"));
        errdefer m.norm.free();
        m.in_norm = try l.share(map.layer(&n, i, "input_layernorm.weight"));
        errdefer m.in_norm.free();
        m.post_norm = try l.share(map.layer(&n, i, "post_attention_layernorm.weight"));
        errdefer m.post_norm.free();
        m.dsa = try dsa.Dsa.load(l, i);
        errdefer m.dsa.deinit();
        m.moe = try moe.Moe.load(l, i);
        errdefer m.moe.deinit();
        m.head = try DraftHead.load(l);
        return m;
    }

    pub fn deinit(m: *Mtp) void {
        m.enorm.free();
        m.hnorm.free();
        m.eh.deinit();
        m.norm.free();
        m.in_norm.free();
        m.post_norm.free();
        m.dsa.deinit();
        m.moe.deinit();
        m.head.deinit();
    }
};

/// The head's own rows for up to `rows` rows (forward.py `Buffers`' MTP fields and those it shares with a layer).
pub const Buffers = struct {
    rows: usize,
    prefill: bool,
    ids: Buf,
    hin: Buf, // [rows, D] bf16: the main model's final-normed rows (or a draft's chained row)
    me: Buf, // [rows, D]
    mcat: Buf, // [rows, 2 D]
    mx: Buf, // [rows, D]: the head's residual row
    normed: Buf,
    xs: Buf,
    part: Buf,
    gath: Buf,
    sk: Buf,
    fnormed: Buf, // [1, D]
    fxs: Buf,
    logits: Buf, // [1, n] bf16

    borrowed: bool = false, // part, gath and sk are another set of buffers' (`initSharing`), not freed here

    pub fn init(d: *const cuda.Driver, lay: config.Layout, rows: usize, prefill: bool) !Buffers {
        return initSharing(d, lay, rows, prefill, null);
    }

    /// The buffers of the backbone's chunks a prompt chunk's MTP absorb reuses: its own partials, their all-gather and
    /// the split-K slices, which the backbone has finished with when the absorb starts (cuda_prompt.zig runs the
    /// absorb after the chunk's tail, each gather waited for, on the chunk's stream).
    pub const Shared = struct { part: Buf, gath: Buf, sk: Buf };

    /// `init`, with part, gath and sk the `shared` buffers when given (each at least as large as this set's own).
    pub fn initSharing(d: *const cuda.Driver, lay: config.Layout, rows: usize, prefill: bool, shared: ?Shared) !Buffers {
        const D = lay.cfg.hidden;
        const sizes = .{ .part = rows * D * 4, .gath = lay.world * rows * D * 4, .sk = 8 * rows * D * 4 };
        if (shared) |x| if (x.part.len < sizes.part or x.gath.len < sizes.gath or x.sk.len < sizes.sk) return error.SharedBuffersTooSmall;
        var b: Buffers = undefined;
        b.rows = rows;
        b.prefill = prefill;
        b.borrowed = shared != null;
        b.ids = try Buf.alloc(d, rows * 4);
        b.hin = try Buf.alloc(d, rows * D * 2);
        b.me = try Buf.alloc(d, rows * D * 2);
        b.mcat = try Buf.alloc(d, rows * 2 * D * 2);
        b.mx = try Buf.alloc(d, rows * D * 2);
        b.normed = try Buf.alloc(d, rows * D * 2);
        b.xs = try Buf.alloc(d, rows * D / 64 * 4);
        if (shared) |x| {
            b.part = x.part;
            b.gath = x.gath;
            b.sk = x.sk;
        } else {
            b.part = try Buf.alloc(d, sizes.part);
            b.gath = try Buf.alloc(d, sizes.gath);
            b.sk = try Buf.alloc(d, sizes.sk);
        }
        // a decode set holds the head rows of up to `dsa.max_segs` streams' steps at once (`runSegs`), row i each's
        const heads: usize = if (prefill) 1 else dsa.max_segs;
        b.fnormed = try Buf.alloc(d, heads * D * 2);
        b.fxs = try Buf.alloc(d, heads * D / 64 * 4);
        b.logits = try Buf.alloc(d, heads * lay.size(.vocab) * 2);
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        inline for (.{ "ids", "hin", "me", "mcat", "mx", "normed", "xs", "fnormed", "fxs", "logits" }) |f| @field(b, f).free();
        if (!b.borrowed) inline for (.{ "part", "gath", "sk" }) |f| @field(b, f).free();
    }
};

/// mtp_compute up to the attention: the embedding (`zero_first`: the stream's first position, its row zeroed), the
/// two norms side by side, eh_proj into b.mx and its RMSNorm into b.normed / b.xs.
pub fn front(g: Glue, c: *const config.Config, m: *const Mtp, b: *const Buffers, embed: Ptr, n: usize, zero_first: bool, stream: cuda.Stream) !void {
    const D = c.hidden;
    try g.embed(b.ids.ptr, embed, b.me.ptr, n, 1, stream);
    if (zero_first) {
        const row: Buf = .{ .d = b.me.d, .ptr = b.me.ptr, .len = D * 2 };
        try row.fill8(0, stream.handle);
    }
    try g.rmsnorm(b.me.ptr, D, m.enorm.ptr, b.mcat.ptr, 2 * D, 0, c.eps, n, D, stream);
    try g.rmsnorm(b.hin.ptr, D, m.hnorm.ptr, b.mcat.ptr + D * 2, 2 * D, 0, c.eps, n, D, stream);
    try g.bmm(b.mcat.ptr, 2 * D, m.eh.b16(), n, b.mx.ptr, false, b.sk.ptr, stream);
    try g.rmsnorm(b.mx.ptr, D, m.in_norm.ptr, b.normed.ptr, D, b.xs.ptr, c.eps, n, D, stream);
}

/// After the attention's all-gather: the plain residual, then the FFN's input (post_attention_layernorm).
pub fn afterAttention(g: Glue, c: *const config.Config, m: *const Mtp, b: *const Buffers, world: usize, n: usize, stream: cuda.Stream) !void {
    const D = c.hidden;
    try g.residualAdd(b.mx.ptr, b.mx.ptr, b.gath.ptr, world, n, stream);
    try g.rmsnorm(b.mx.ptr, D, m.post_norm.ptr, b.normed.ptr, D, b.xs.ptr, c.eps, n, D, stream);
}

/// After the MoE's all-gather: the plain residual, shared_head.norm of the last row, the draft head's logits of it.
pub fn head(g: Glue, k: *const Kernels, c: *const config.Config, m: *const Mtp, b: *const Buffers, world: usize, n: usize, stream: cuda.Stream) !void {
    const D = c.hidden;
    try g.residualAdd(b.mx.ptr, b.mx.ptr, b.gath.ptr, world, n, stream);
    try g.rmsnorm(b.mx.ptr + (n - 1) * D * 2, D, m.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, c.eps, 1, D, stream);
    try headLogits(k, &m.head, b.fnormed.ptr, b.fxs.ptr, b.logits.ptr, 1, b.prefill, stream);
}

/// qmm.py `matmul` (decode rows: a 16-row tile, one K slice) or `prefill_matmul` (tile 0) of the draft head over x
/// [M, K] bf16 (contiguous) with its group sums xs: out [M, n] bf16.
pub fn headLogits(k: *const Kernels, h: *const DraftHead, x: Ptr, xs: Ptr, out: Ptr, M: usize, prefill: bool, stream: cuda.Stream) !void {
    const K = h.k;
    const N = h.n;
    const bm: usize = if (prefill) 128 else 16;
    const bn: usize = if (prefill) 128 else 64;
    const rows_t = (M + bm - 1) / bm;
    // qmm.cu `launch`: a group of row tiles keeps its inputs near 12 MB of L2 while its blocks sweep the columns
    const group: usize = @max(1, @min(rows_t, (12 << 20) / (bm * K * 2)));
    const ldx: c_int = @intCast(K);
    var a: cuda.Args = .{};
    a.add(x);
    if (!prefill) a.add(xs);
    a.add(h.words.ptr);
    a.add(h.scales.ptr);
    a.add(h.biases.ptr);
    a.add(out);
    if (!prefill) a.add(@as(Ptr, 0));
    a.add(@as(c_int, @intCast(M)));
    a.add(@as(c_int, @intCast(N)));
    a.add(@as(c_int, @intCast(K)));
    if (!prefill) a.add(@as(c_int, 1));
    a.add(@as(c_int, @intCast(h.npad)));
    a.add(ldx);
    a.add(@as(c_int, @intCast(group)));
    const blocks: u32 = @intCast(rows_t * ((N + bn - 1) / bn));
    if (prefill) {
        try cuda.launch.launch(k.pre, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 }, .shared = k.pre_smem }, stream, &a);
    } else {
        try cuda.launch.launch(k.dec, .{ .grid = .{ .x = blocks, .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = k.dec_smem }, stream, &a);
    }
}

/// The all-gather the head's two blocks end in (cuda_model.zig `Gather`'s shape).
pub const Gather = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, part: Ptr, gath: Ptr, floats: usize, stream: cuda.Stream) anyerror!void,
};

/// What `run` needs besides the head: its caches and the blocks' scratch (a prompt chunk's or a decode window's).
pub const Step = struct {
    cache: *const dsa.Cache,
    ds: *const dsa.Scratch,
    ms: *const moe.Scratch,
    pos_dev: Buf, // an int32 for the head's position
};

/// mtp.py `mtp_forward`: `next` tokens (host) and their input rows (`hidden`, n rows of D bf16 on the device, row
/// stride D) at head position `mtp_len` (the first position's embedding zeroed at 0); the draft head's logits of the
/// last row in b.logits and its shared_head.norm output in b.fnormed (the next chained draft's input).
pub fn run(g: Glue, k: *const Kernels, mk: *const moe.Kernels, lay: config.Layout, m: *const Mtp, b: *const Buffers, step: Step, embed: Ptr, next: []const u32, hidden: Ptr, mtp_len: usize, gather: Gather, stream: cuda.Stream) !void {
    const c = lay.cfg;
    const D = c.hidden;
    const n = next.len;
    try b.ids.uploadAsync(0, std.mem.sliceAsBytes(next), stream.handle);
    if (hidden != b.hin.ptr) try b.hin.copyFrom(0, hidden, n * D * 2, stream.handle);
    const p32: i32 = @intCast(mtp_len);
    try step.pos_dev.uploadAsync(0, std.mem.asBytes(&p32), stream.handle);
    try front(g, c, m, b, embed, n, mtp_len == 0, stream);
    try dsa.dsaBlock(g, lay, &m.dsa, step.cache, step.ds, .{
        .normed = b.normed.ptr, .part = b.part.ptr, .sk = b.sk.ptr, .pos_dev = step.pos_dev.ptr,
        .host_pos = mtp_len,    .nch = (mtp_len + n + 511) / 512, .R = n,
    }, stream);
    try gather.run(gather.ctx, b.part.ptr, b.gath.ptr, n * D, stream);
    try afterAttention(g, c, m, b, lay.world, n, stream);
    try moe.moeBlock(g, mk, lay, &m.moe, step.ms, b.normed.ptr, b.part.ptr, b.sk.ptr, n, stream);
    try gather.run(gather.ctx, b.part.ptr, b.gath.ptr, n * D, stream);
    try head(g, k, c, m, b, lay.world, n, stream);
}

/// One stream's rows in a head step several streams share (`runSegs`): its next tokens and their input rows, its head
/// position and cache.
pub const Seg = struct {
    next: []const u32,
    hidden: Ptr, // next.len rows of D bf16 (row stride D)
    mtp_len: usize,
    cache: *const dsa.Cache,
    pos_dev: Buf,
};

/// `run` for several streams at once (decode rows, `segs` in row order, at most b.rows rows in all): the row-wise
/// blocks (the embedding, the norms, eh_proj, the projections, the MoE, the draft head) over every row, each
/// stream's DSA attention over its own rows and cache (cuda_model.zig `forwardSegs`' split). Stream i's last row's
/// shared_head.norm lands in b.fnormed row i and its logits in b.logits row i, the bits `run` gives it alone when
/// the row-wise kernels give a row the same bits whatever the rows beside it (glm53-shared checks).
pub fn runSegs(g: Glue, k: *const Kernels, mk: *const moe.Kernels, lay: config.Layout, m: *const Mtp, b: *const Buffers, ds: *const dsa.Scratch, ms: *const moe.Scratch, segs: []const Seg, embed: Ptr, gather: Gather, stream: cuda.Stream) !void {
    const c = lay.cfg;
    const D = c.hidden;
    if (b.prefill or segs.len == 0 or segs.len > dsa.max_segs) return error.BadSegments;
    var a0s: [dsa.max_segs]usize = undefined;
    var R: usize = 0;
    var sparse = false;
    for (segs, 0..) |sg, i| {
        const n = sg.next.len;
        if (n == 0 or n > dsa.seg_rows) return error.BadSegments;
        a0s[i] = R;
        try b.ids.uploadAsync(R * 4, std.mem.sliceAsBytes(sg.next), stream.handle);
        try b.hin.copyFrom(R * D * 2, sg.hidden, n * D * 2, stream.handle);
        const p32: i32 = @intCast(sg.mtp_len);
        try sg.pos_dev.uploadAsync(0, std.mem.asBytes(&p32), stream.handle);
        sparse = sparse or dsa.sparseRows(c, sg.mtp_len, n);
        R += n;
    }
    if (R > b.rows) return error.BadSegments;
    // front over every row, the first position's embedding zeroed for a stream at position 0
    try g.embed(b.ids.ptr, embed, b.me.ptr, R, 1, stream);
    for (segs, a0s[0..segs.len]) |sg, a0| if (sg.mtp_len == 0) {
        const row: Buf = .{ .d = b.me.d, .ptr = b.me.ptr + a0 * D * 2, .len = D * 2 };
        try row.fill8(0, stream.handle);
    };
    try g.rmsnorm(b.me.ptr, D, m.enorm.ptr, b.mcat.ptr, 2 * D, 0, c.eps, R, D, stream);
    try g.rmsnorm(b.hin.ptr, D, m.hnorm.ptr, b.mcat.ptr + D * 2, 2 * D, 0, c.eps, R, D, stream);
    try g.bmm(b.mcat.ptr, 2 * D, m.eh.b16(), R, b.mx.ptr, false, b.sk.ptr, stream);
    try g.rmsnorm(b.mx.ptr, D, m.in_norm.ptr, b.normed.ptr, D, b.xs.ptr, c.eps, R, D, stream);
    // the DSA layer: every stream's selection, then (on a tier, one batch of reads) every stream's attention
    try dsa.dsaFront(g, lay, &m.dsa, ds, b.normed.ptr, b.sk.ptr, 0, R, stream);
    try dsa.dsaIndex(g, lay, &m.dsa, ds, b.normed.ptr, b.sk.ptr, R, sparse, stream);
    var calls: [dsa.max_segs]dsa.Call = undefined;
    for (segs, calls[0..segs.len]) |sg, *cl| cl.* = .{
        .normed = b.normed.ptr,
        .part = b.part.ptr,
        .sk = b.sk.ptr,
        .pos_dev = sg.pos_dev.ptr,
        .host_pos = sg.mtp_len,
        .nch = (sg.mtp_len + sg.next.len + 511) / 512,
        .R = sg.next.len,
    };
    for (segs, calls[0..segs.len], a0s[0..segs.len], 0..) |sg, cl, a0, i| try dsa.dsaSelect(g, lay, &m.dsa, sg.cache, ds, cl, a0, ds.selection(i), stream);
    var resolved: [dsa.max_segs]?Tr.Tier.Resolved = @splat(null);
    if (segs[0].cache.tier) |tr| {
        var sels: [dsa.max_segs]Tr.Tier.Sel = undefined;
        var which: [dsa.max_segs]usize = undefined;
        var n: usize = 0;
        for (segs, 0..) |sg, i| if (dsa.sparseRows(c, sg.mtp_len, sg.next.len)) {
            const x = sg.cache.tier.?;
            const sel = ds.selection(i);
            sels[n] = .{ .h = x.h, .layer = x.layer, .tokens = sel.tokens, .counts = sel.counts, .R = sg.next.len };
            which[n] = i;
            n += 1;
        };
        var out: [dsa.max_segs]Tr.Tier.Resolved = undefined;
        if (n > 0) try tr.t.resolveMany(sels[0..n], glue_mod.sparse_tokens, stream, out[0..n]);
        for (which[0..n], out[0..n]) |i, o| resolved[i] = o;
    }
    for (segs, calls[0..segs.len], a0s[0..segs.len], 0..) |sg, cl, a0, i| try dsa.dsaAttendRows(g, lay, &m.dsa, sg.cache, ds, cl, a0, ds.selection(i), resolved[i], stream);
    try dsa.dsaExpand(g, lay, &m.dsa, ds, R, stream);
    try dsa.dsaOut(g, lay, &m.dsa, ds, b.part.ptr, b.sk.ptr, 0, R, stream);
    try gather.run(gather.ctx, b.part.ptr, b.gath.ptr, R * D, stream);
    try afterAttention(g, c, m, b, lay.world, R, stream);
    try moe.moeBlock(g, mk, lay, &m.moe, ms, b.normed.ptr, b.part.ptr, b.sk.ptr, R, stream);
    try gather.run(gather.ctx, b.part.ptr, b.gath.ptr, R * D, stream);
    // the residual over every row; each stream's last row normed into its head row, then the heads' logits at once
    try g.residualAdd(b.mx.ptr, b.mx.ptr, b.gath.ptr, lay.world, R, stream);
    for (segs, a0s[0..segs.len], 0..) |sg, a0, i| {
        try g.rmsnorm(b.mx.ptr + (a0 + sg.next.len - 1) * D * 2, D, m.norm.ptr, b.fnormed.ptr + i * D * 2, D, b.fxs.ptr + i * D / 64 * 4, c.eps, 1, D, stream);
    }
    try headLogits(k, &m.head, b.fnormed.ptr, b.fxs.ptr, b.logits.ptr, segs.len, false, stream);
}
