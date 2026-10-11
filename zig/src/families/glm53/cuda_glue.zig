//! The Mia fork's GLM-5.3-Flash row kernels replayed from captured cubins (`cuda.aot.Set`): each call launches the
//! Triton kernel with the grid, runtime arguments and constexprs of the Python wrapper it names, so the set picks the
//! variant Triton compiled for that call and the bits are the reference's; a launch the capture never saw fails with
//! error.MissingTritonVariant. The decode rows' hyper-connection mixing dots run hc.cu's kernel (glm53_hc fatbin).
//!
//! Wrappers (glm5_next/cuda module: function): glue: embed (_embed_b16), rmsnorm, hc_pre (hc.cu or
//! _hc_partial_mma, then _hc_finish), hc_post, residual_add, stream_mean, swiglu, router (_router_part,
//! _router_sum), select (_topk), combine_shared (_combine_sy); qmm: group_sums, matmul on BF16 weights (_bmm,
//! _reduce); forward: _shift_conv (_conv_shift).
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const Ptr = cuda.abi.DevicePtr;

/// glue.py HC_BLOCKS: the mixing dots' fixed K split.
pub const hc_blocks = 16;
/// qmm.py B16_BN, B16_BK, SPLIT_TARGET, SEQ_ROWS (TF_GLM_SEQ_ROWS unset).
const b16_bn = 64;
const b16_bk = 64;
const split_target = 192;
const seq_rows = 256;
/// glue.py ROUTER_KS.
const router_ks = 8;

fn n32(v: usize) i32 {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// qmm.py `b16_split_k`: K slices of a BF16 matmul, fixed by the shape.
pub fn b16SplitK(n: usize, k: usize) usize {
    const tiles = cdiv(n, b16_bn);
    const steps = k / b16_bk;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// qmm.py `bucket`.
fn bucket(m: usize) usize {
    for ([_]usize{ 16, 32, 64, 128 }) |b| if (m <= b) return b;
    unreachable;
}

pub const Hc = struct {
    module: cuda.Module,
    partial: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Hc {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.glm53_hc);
        errdefer module.unload();
        return .{ .module = module, .partial = try module.function("_ZN11tf_glm53_hc17hc_partial_kernelEPK13__nv_bfloat16S2_Pfi") };
    }

    pub fn deinit(self: *Hc) void {
        self.module.unload();
    }
};

/// A BF16 matrix [n, k] on the device (qmm.py `B16`), row-major.
pub const B16 = struct { ptr: Ptr, n: usize, k: usize };

pub const Glue = struct {
    set: *const aot.Set,
    hc: *const Hc,
    hidden: usize, // D
    streams: usize, // S

    /// glue.py `embed` of a BF16 table: out [rows, copies D] bf16, token ids[r]'s row in every copy.
    pub fn embed(g: Glue, ids: Ptr, table: Ptr, out: Ptr, rows: usize, copies: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_embed_b16", .{ @intCast(rows), @intCast(g.hidden / 64), 1 }, &.{
            aot.ptr("IDS", "*i32", ids), aot.ptr("W", "*bf16", table), aot.ptr("OUT", "*bf16", out),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("COPIES", @intCast(copies)) });
    }

    /// glue.py `rmsnorm`: out [rows, d] bf16 (row stride o_stride) of x (row stride x_stride), and its 64-group
    /// sums xs when not 0.
    pub fn rmsnorm(g: Glue, x: Ptr, x_stride: usize, w: Ptr, out: Ptr, o_stride: usize, xs: Ptr, eps: f32, rows: usize, d: usize, stream: cuda.Stream) !void {
        const block = std.math.ceilPowerOfTwoAssert(usize, d);
        try g.set.run(stream, "_rmsnorm", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x),     aot.int("x_stride", n32(x_stride)),            aot.ptr("W", "*bf16", w),
            aot.ptr("OUT", "*bf16", out), aot.int("o_stride", n32(o_stride)),            aot.ptr("XS", if (xs != 0) "*fp32" else "*bf16", if (xs != 0) xs else out),
            aot.float("eps", eps),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("SUMS", @intFromBool(xs != 0)) });
    }

    pub const HcPre = struct {
        x: Ptr, // [rows, S D] bf16, contiguous
        fn_: Ptr, // [24, S D] bf16
        base: Ptr, // [24] fp32
        scale: Ptr, // [3] fp32
        norm: Ptr, // [D] bf16
        out: Ptr, // [rows, D] bf16
        xs: Ptr, // [rows, D / 64] fp32
        post: Ptr, // [rows, S] fp32
        comb: Ptr, // [rows, S S] fp32
        part: Ptr, // [rows, 16, 32] fp32
        eps: f32,
        hc_eps: f32,
        iters: usize,
        rows: usize,
        prompt: bool,
    };

    /// glue.py `hc_pre`: the mixing dots (a prompt chunk's on the tensor cores, decode rows' in hc.cu, each with
    /// its own bits), then _hc_finish: the normed collapsed row, its sums, post and the Sinkhorn-balanced comb.
    pub fn hcPre(g: Glue, a: HcPre, stream: cuda.Stream) !void {
        const wide = g.streams * g.hidden;
        if (a.prompt) {
            try g.set.run(stream, "_hc_partial_mma", .{ @intCast(cdiv(a.rows, 64)), hc_blocks, 1 }, &.{
                aot.ptr("X", "*bf16", a.x), aot.ptr("FN", "*bf16", a.fn_), aot.ptr("PART", "*fp32", a.part), aot.int("R", n32(a.rows)),
            }, &.{ aot.ci("WIDE", @intCast(wide)), aot.ci("NB", hc_blocks), aot.ci("BM", 64), aot.ci("BK", 64) });
        } else {
            if (wide != 16384) return error.UnsupportedShape;
            var args: cuda.Args = .{};
            args.add(a.x);
            args.add(a.fn_);
            args.add(a.part);
            args.add(@as(c_int, @intCast(a.rows)));
            try cuda.launch.launch(g.hc.partial, .{ .grid = .{ .x = hc_blocks, .y = 3, .z = @intCast(cdiv(a.rows, 4)) }, .block = .{ .x = 288 } }, stream, &args);
        }
        try g.set.run(stream, "_hc_finish", .{ @intCast(a.rows), 1, 1 }, &.{
            aot.ptr("X", "*bf16", a.x),          aot.ptr("PART", "*fp32", a.part), aot.ptr("BASE", "*fp32", a.base),
            aot.ptr("SCALE", "*fp32", a.scale),  aot.ptr("NW", "*bf16", a.norm),   aot.ptr("OUT", "*bf16", a.out),
            aot.ptr("XS", "*fp32", a.xs),        aot.ptr("POST", "*fp32", a.post), aot.ptr("COMB", "*fp32", a.comb),
            aot.float("eps_norm", a.eps),        aot.float("hc_eps", a.hc_eps),
        }, &.{
            aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("NB", hc_blocks),
            aot.ci("ITERS", @intCast(a.iters)), aot.ci("BLOCK", @intCast(g.hidden)),
        });
    }

    /// glue.py `hc_post`: xout = the streams of x mixed by comb plus post times bf16(the ranks' fp32 partials
    /// `gathered` [world, rows, D], rank 0 first).
    pub fn hcPost(g: Glue, x: Ptr, xout: Ptr, gathered: Ptr, world: usize, post: Ptr, comb: Ptr, rows: usize, stream: cuda.Stream) !void {
        const block = @min(1024, g.hidden);
        try g.set.run(stream, "_hc_post", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("X", "*bf16", x),       aot.ptr("XOUT", "*bf16", xout), aot.ptr("G", "*fp32", gathered),
            aot.ptr("POST", "*fp32", post), aot.ptr("COMB", "*fp32", comb), aot.int("RS", n32(rows * g.hidden)),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("WORLD", @intCast(world)), aot.ci("BLOCK", @intCast(block)) });
    }

    /// glue.py `residual_add` (the MTP layer): xout = bf16(x + bf16(the partials summed rank 0 first)).
    pub fn residualAdd(g: Glue, x: Ptr, xout: Ptr, gathered: Ptr, world: usize, rows: usize, stream: cuda.Stream) !void {
        const block = @min(1024, g.hidden);
        try g.set.run(stream, "_residual_add", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("XOUT", "*bf16", xout), aot.ptr("G", "*fp32", gathered), aot.int("RS", n32(rows * g.hidden)),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("WORLD", @intCast(world)), aot.ci("BLOCK", @intCast(block)) });
    }

    /// glue.py `stream_mean`: out [rows, D] bf16 = the streams' mean.
    pub fn streamMean(g: Glue, x: Ptr, out: Ptr, rows: usize, stream: cuda.Stream) !void {
        const block = @min(1024, g.hidden);
        try g.set.run(stream, "_stream_mean", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("OUT", "*bf16", out),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("BLOCK", @intCast(block)) });
    }

    /// glue.py `swiglu`: out [rows, w] bf16 = the limited SwiGLU of [gate | up] rows, and its 64-group sums.
    pub fn swiglu(g: Glue, gu: Ptr, out: Ptr, xs: Ptr, limit: f32, rows: usize, w: usize, stream: cuda.Stream) !void {
        const block = std.math.gcd(512, w);
        try g.set.run(stream, "_swiglu", .{ @intCast(rows), @intCast(w / block), 1 }, &.{
            aot.ptr("GU", "*bf16", gu), aot.ptr("OUT", "*bf16", out), aot.ptr("XS", "*fp32", xs), aot.float("LIMIT", limit),
        }, &.{ aot.ci("W", @intCast(w)), aot.ci("BLOCK", @intCast(block)) });
    }

    /// qmm.py `group_sums`: xs [m, k / 64] fp32, each 64-input group's sum of bf16 x (row stride x_stride).
    pub fn groupSums(g: Glue, x: Ptr, x_stride: usize, xs: Ptr, m: usize, k: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_group_sums", .{ @intCast(m), @intCast(cdiv(k / 64, 16)), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("XS", "*fp32", xs), aot.int("x_stride", n32(x_stride)),
        }, &.{ aot.ci("K", @intCast(k)), aot.ci("GB", 16) });
    }

    /// qmm.py `matmul` of BF16 weights (`_matmul_b16`): out [m, n] (bf16, or fp32 with f32) = x [m, k] (row stride
    /// x_stride) times w^T, K in the shape's fixed slices summed in slice order (one program for prompt rows, else
    /// fp32 partials in `part`, at least sk m n floats, and _reduce).
    pub fn bmm(g: Glue, x: Ptr, x_stride: usize, w: B16, m: usize, out: Ptr, fp32: bool, part: Ptr, stream: cuda.Stream) !void {
        const bm = bucket(@min(m, 128));
        const sk = b16SplitK(w.n, w.k);
        const seq = sk > 1 and m >= seq_rows and w.n >= 512 and w.k <= 4096;
        const out_ty = if (fp32) "*fp32" else "*bf16";
        const split = sk > 1 and !seq;
        try g.set.run(stream, "_bmm", .{ @intCast(cdiv(m, bm)), @intCast(cdiv(w.n, b16_bn)), @intCast(if (seq) 1 else sk) }, &.{
            aot.ptr("X", "*bf16", x),  aot.ptr("W", "*bf16", w.ptr), aot.ptr("OUT", out_ty, out),
            aot.ptr("PART", if (split) "*fp32" else out_ty, if (split) part else out),
            aot.int("M", n32(m)),      aot.int("x_stride", n32(x_stride)),
        }, &.{
            aot.ci("N", @intCast(w.n)), aot.ci("K", @intCast(w.k)), aot.ci("SK", @intCast(sk)), aot.ci("BM", @intCast(bm)),
            aot.ci("BLOCK_N", b16_bn),  aot.ci("BK", b16_bk),         aot.ci("F32", @intFromBool(fp32)), aot.ci("SEQ", @intFromBool(seq)),
        });
        if (!split) return;
        const total = m * w.n;
        try g.set.run(stream, "_reduce", .{ @intCast(cdiv(total, 1024)), 1, 1 }, &.{
            aot.ptr("PART", "*fp32", part), aot.ptr("OUT", out_ty, out), aot.int("total", n32(total)),
        }, &.{ aot.ci("SK", @intCast(sk)), aot.ci("BLOCK", 1024), aot.ci("F32", @intFromBool(fp32)) });
    }

    /// forward.py `_shift_conv`: each of `layers` conv windows [3, C] (layer stride conv_layer) takes rows keep ..
    /// keep + 2 of [window; the layer's projection rows] (proj layer stride proj_layer, row stride proj_row).
    pub fn convShift(g: Glue, conv: Ptr, proj: Ptr, keep: usize, layers: usize, c: usize, conv_layer: usize, proj_layer: usize, proj_row: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_conv_shift", .{ @intCast(layers), @intCast(cdiv(c, 1024)), 1 }, &.{
            aot.ptr("CONV", "*bf16", conv), aot.ptr("PROJ", "*bf16", proj), aot.int("keep", n32(keep)),
            aot.int("conv_layer", n32(conv_layer)), aot.int("proj_layer", n32(proj_layer)), aot.int("proj_row", n32(proj_row)),
        }, &.{ aot.ci("C", @intCast(c)), aot.ci("TAPS", 3), aot.ci("BLOCK", 1024) });
    }

    /// glue.py `router` (ROUTER_KS slices): out [m, ne] fp32 = x [m, D] times the router's bf16 rows, the slices'
    /// partials (`part`, ks m ne floats) added in slice order.
    pub fn router(g: Glue, x: Ptr, x_stride: usize, w: Ptr, ne: usize, m: usize, out: Ptr, part: Ptr, stream: cuda.Stream) !void {
        const bm: usize = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
        try g.set.run(stream, "_router_part", .{ @intCast(cdiv(m, bm)), @intCast(cdiv(ne, 32)), router_ks }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("W", "*bf16", w), aot.ptr("PART", "*fp32", part), aot.int("M", n32(m)), aot.int("x_stride", n32(x_stride)),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("NE", @intCast(ne)), aot.ci("BM", @intCast(bm)), aot.ci("BLOCK_E", 32), aot.ci("BK", 64), aot.ci("KS", router_ks) });
        const total = m * ne;
        try g.set.run(stream, "_router_sum", .{ @intCast(cdiv(total, 1024)), 1, 1 }, &.{
            aot.ptr("PART", "*fp32", part), aot.ptr("OUT", "*fp32", out), aot.int("total", n32(total)),
        }, &.{ aot.ci("KS", router_ks), aot.ci("BLOCK", 1024) });
    }

    /// glue.py `select`: each row's top-k experts by sigmoid score plus bias (lower id on ties), their weights
    /// normalized and scaled, then the shared expert (id `experts`, weight 1): pick int32 / wts fp32 [rows, k + 1].
    pub fn select(g: Glue, logits: Ptr, bias: Ptr, pick: Ptr, wts: Ptr, top_k: usize, experts: usize, scale: f32, norm: bool, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_topk", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("L", "*fp32", logits), aot.ptr("BIAS", "*fp32", bias), aot.ptr("PICK", "*i32", pick), aot.ptr("WTS", "*fp32", wts), aot.float("scale", scale),
        }, &.{
            aot.ci("NE", @intCast(experts)),                                                      aot.ci("TOPK", @intCast(top_k)),
            aot.ci("SLOTS", @intCast(top_k + 1)),                                                 aot.ci("BLOCK", @intCast(std.math.ceilPowerOfTwoAssert(usize, experts + 1))),
            aot.ci("SLOTP", @intCast(std.math.ceilPowerOfTwoAssert(usize, top_k + 1))), aot.ci("NORM", @intFromBool(norm)),
        });
    }

    /// glue.py `combine_shared`: out [rows, D] fp32 = the routed slots of y [rows, slots, D] fp32 weighted, then the
    /// shared expert's row sy [rows, D] times its weight, added in slot order.
    pub fn combineShared(g: Glue, y: Ptr, sy: Ptr, wts: Ptr, out: Ptr, slots: usize, rows: usize, stream: cuda.Stream) !void {
        const block = @min(1024, g.hidden);
        try g.set.run(stream, "_combine_sy", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("Y", "*fp32", y), aot.ptr("SY", "*fp32", sy), aot.ptr("WTS", "*fp32", wts), aot.ptr("OUT", "*fp32", out),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("SLOTS", @intCast(slots)), aot.ci("BLOCK", @intCast(block)) });
    }
};

test "qmm.py's BF16 K slices for the rank's shapes" {
    // KDA projection [12576, 4096] at two ranks: 197 tiles, one slice; o_proj [4096, 4096]: 64 tiles, 4 slices
    try std.testing.expectEqual(@as(usize, 1), b16SplitK(12576, 4096));
    try std.testing.expectEqual(@as(usize, 4), b16SplitK(4096, 4096));
    // f_b / g_b [4096, 128]: two K steps, no split
    try std.testing.expectEqual(@as(usize, 1), b16SplitK(4096, 128));
    // the dense MLP's down [4096, 6144]: 4 slices; its gate | up [12288, 4096]: one
    try std.testing.expectEqual(@as(usize, 4), b16SplitK(4096, 6144));
    try std.testing.expectEqual(@as(usize, 1), b16SplitK(12288, 4096));
}
