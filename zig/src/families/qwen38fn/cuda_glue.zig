//! TensorFold 0.6.6's Flash Next Triton glue replayed from captured cubins (`cuda.aot.Set`): each call launches the
//! kernel with the grid, runtime arguments, argument dtypes and constexprs of the Python wrapper it names, so the set
//! picks the variant Triton compiled for that call and the bits are the reference's. A launch the capture never saw
//! fails with error.MissingTritonVariant instead of running something else.
//!
//! Kernels (module: Python wrapper): exl3_mm: `embed` (_embed), F16 matrices (_f16_mm, _reduce); glue: hc_writeback,
//! hc_normed, hc_reduce_act, hc_act, hc_mix; forward: shift_windows; tensorfold.cuda.moe: router, select_rows
//! (_topk_rows).
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const F16 = @import("cuda_weights.zig").F16;

const Ptr = cuda.abi.DevicePtr;

/// exl3_mm.py ROWS: most rows one F16 or EXL3 call takes.
pub const call_rows = 128;

pub const Glue = struct {
    set: *const aot.Set,
    hidden: usize, // D
    streams: usize, // S
    low: usize,

    fn n32(v: usize) i32 {
        return @intCast(v);
    }

    /// exl3_mm.py `embed`: out [rows, S * D] bf16 = the bf16 table row of ids[r] (int32), repeated per stream.
    pub fn embed(g: Glue, ids: Ptr, table: Ptr, out: Ptr, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_embed", .{ @intCast(rows), @intCast(g.hidden / 256), 1 }, &.{
            aot.ptr("IDS", "*i32", ids), aot.ptr("T", "*bf16", table), aot.ptr("OUT", "*bf16", out),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("BLOCK", 256) });
    }

    /// What a write-back adds to the streams (glue.py `hc_writeback` modes 0, 1, 2).
    pub const Pending = union(enum) {
        none,
        branch: struct { branch: Ptr, inject: Ptr }, // bf16 [rows, D], bf16 [rows, S]
        moe: struct { y: Ptr, wts: Ptr, inject: Ptr, slots: usize }, // fp32 [rows, slots, D], fp32 [rows, slots], bf16 [rows, S]
        moe_bf16: struct { y: Ptr, wts: Ptr, inject: Ptr, slots: usize }, // a prompt chunk's slots, kept as bf16
    };

    /// glue.py `hc_writeback(h, h, pss, S, mode, ...)`: the pending branch added into the streams h (in place), and
    /// each stream's per-256 sums of squares into pss fp32 [rows, D / 256, S].
    pub fn hcWriteback(g: Glue, h: Ptr, pss: Ptr, rows: usize, pending: Pending, stream: cuda.Stream) !void {
        const mode: i64, const br: aot.Arg, const inj: aot.Arg, const y: aot.Arg, const wts: aot.Arg, const top: i64, const slots: i64 = switch (pending) {
            .none => .{ 0, aot.ptr("BR", "*bf16", h), aot.ptr("INJ", "*bf16", h), aot.ptr("Y", "*bf16", h), aot.ptr("WTS", "*bf16", h), 1, 1 },
            .branch => |b| .{ 1, aot.ptr("BR", "*bf16", b.branch), aot.ptr("INJ", "*bf16", b.inject), aot.ptr("Y", "*bf16", h), aot.ptr("WTS", "*bf16", h), 1, 1 },
            .moe => |m| .{ 2, aot.ptr("BR", "*bf16", h), aot.ptr("INJ", "*bf16", m.inject), aot.ptr("Y", "*fp32", m.y), aot.ptr("WTS", "*fp32", m.wts), @intCast(m.slots - 1), @intCast(m.slots) },
            .moe_bf16 => |m| .{ 2, aot.ptr("BR", "*bf16", h), aot.ptr("INJ", "*bf16", m.inject), aot.ptr("Y", "*bf16", m.y), aot.ptr("WTS", "*fp32", m.wts), @intCast(m.slots - 1), @intCast(m.slots) },
        };
        try g.set.run(stream, "_hc_writeback", .{ @intCast(rows), @intCast(g.hidden / 256), 1 }, &.{
            aot.ptr("H", "*bf16", h), aot.ptr("HOUT", "*bf16", h), aot.ptr("PSS", "*fp32", pss), br, inj, y, wts,
            aot.int("RS", n32(rows * g.hidden)),
        }, &.{
            aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("MODE", mode), aot.ci("TOPK", top),
            aot.ci("SLOTS", slots),          aot.ci("BLOCK", 256),               aot.ci("WORLD", 1),
        });
    }

    /// glue.py `hc_normed`: normed [rows, S * D] bf16 = h * rinv_stream * scale, and its 32-group sums xs.
    pub fn hcNormed(g: Glue, h: Ptr, pss: Ptr, scale: Ptr, normed: Ptr, xs: Ptr, eps: f32, rows: usize, stream: cuda.Stream) !void {
        const wide = g.streams * g.hidden;
        try g.set.run(stream, "_hc_normed", .{ @intCast(rows), @intCast(wide / 512), 1 }, &.{
            aot.ptr("H", "*bf16", h),           aot.ptr("PSS", "*fp32", pss), aot.ptr("SCALE", "*fp32", scale),
            aot.ptr("NORMED", "*bf16", normed), aot.ptr("XS", "*fp32", xs),   aot.float("eps", eps),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("NC", @intCast(g.hidden / 256)), aot.ci("BLOCK", 512) });
    }

    /// glue.py `hc_reduce_act`: the down projection's slices part fp32 [sk, rows, ndn] summed in order, then act
    /// [rows, low] bf16, its group sums, and (when `inject` is not 0) the inject gates bf16 [rows, S].
    pub fn hcReduceAct(g: Glue, part: Ptr, sk: usize, ndn: usize, act: Ptr, xs: Ptr, inject: Ptr, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_hc_reduce_act", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("PART", "*fp32", part), aot.ptr("ACT", "*bf16", act), aot.ptr("XS", "*fp32", xs),
            aot.ptr("INJ", "*bf16", if (inject != 0) inject else act), aot.int("M", n32(rows)),
        }, &.{
            aot.ci("SK", @intCast(sk)),                                 aot.ci("S", @intCast(g.streams)), aot.ci("LOW", @intCast(g.low)),
            aot.ci("LOWP", @intCast(std.math.ceilPowerOfTwoAssert(usize, g.low))), aot.ci("HAS_INJ", @intFromBool(inject != 0)), aot.ci("NDN", @intCast(ndn)),
        });
    }

    /// glue.py `hc_mix`: mixed [rows, D] bf16 = the streams' sigmoid(up)-weighted normed rows averaged, and its sums.
    pub fn hcMix(g: Glue, up: Ptr, normed: Ptr, mixed: Ptr, xs: Ptr, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_hc_mix", .{ @intCast(rows), @intCast(g.hidden / 256), 1 }, &.{
            aot.ptr("UP", "*bf16", up), aot.ptr("NORMED", "*bf16", normed), aot.ptr("MIXED", "*bf16", mixed), aot.ptr("XS", "*fp32", xs),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("BLOCK", 256) });
    }

    /// exl3_mm.py `F16._launch`: x [m, K] bf16 (row stride x_stride) times w^T into out: fp32 slices [sk, m, n] when
    /// `slices`, else bf16 rows of stride o_stride (only when sk == 1).
    pub fn f16mm(g: Glue, w: *const F16, x: Ptr, x_stride: usize, m: usize, out: Ptr, o_stride: usize, slices: bool, stream: cuda.Stream) !void {
        return g.f16mmOf(w, x, "*bf16", x_stride, m, out, o_stride, slices, stream);
    }

    /// `f16mm` for x of dtype `x_ty` ("*bf16", or "*fp16" for the n-gram rows).
    pub fn f16mmOf(g: Glue, w: *const F16, x: Ptr, x_ty: []const u8, x_stride: usize, m: usize, out: Ptr, o_stride: usize, slices: bool, stream: cuda.Stream) !void {
        if (m > call_rows or (!slices and w.sk != 1)) return error.BadShape;
        try g.set.run(stream, "_f16_mm", .{ @intCast((m + 15) / 16), @intCast((w.n + 63) / 64), @intCast(w.sk) }, &.{
            aot.ptr("X", x_ty, x),                aot.ptr("W", "*fp16", w.w.ptr), aot.ptr("OUT", if (slices) "*fp32" else "*bf16", out),
            aot.int("M", n32(m)),                    aot.int("N", n32(w.n)),         aot.int("x_stride", n32(x_stride)),
            aot.int("o_stride", n32(if (slices) w.n else o_stride)),
        }, &.{
            aot.ci("K", @intCast(w.k)), aot.ci("KS", @intCast(w.k / w.sk)), aot.ci("BM", 16), aot.ci("BN", 64), aot.ci("BK", 64),
            aot.ci("F32", @intFromBool(slices)),
        });
    }

    /// exl3_mm.py `_reduce`: out [m, n] bf16 (row stride o_stride) = the sum of the sk slices, in order.
    pub fn reduce(g: Glue, part: Ptr, sk: usize, m: usize, n: usize, out: Ptr, o_stride: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_reduce", .{ @intCast(m), @intCast((n + 255) / 256), 1 }, &.{
            aot.ptr("P", "*fp32", part), aot.ptr("OUT", "*bf16", out), aot.int("M", n32(m)), aot.int("N", n32(n)), aot.int("o_stride", n32(o_stride)),
        }, &.{ aot.ci("SK", @intCast(sk)), aot.ci("BLOCK", 256) });
    }

    /// exl3_mm.py `F16.__call__` for m <= 128 rows: out [m, n] bf16 (row stride o_stride); `part` holds sk * m * n fp32.
    pub fn f16Call(g: Glue, w: *const F16, x: Ptr, x_stride: usize, m: usize, out: Ptr, o_stride: usize, part: Ptr, stream: cuda.Stream) !void {
        return g.f16CallOf(w, x, "*bf16", x_stride, m, out, o_stride, part, stream);
    }

    pub fn f16CallOf(g: Glue, w: *const F16, x: Ptr, x_ty: []const u8, x_stride: usize, m: usize, out: Ptr, o_stride: usize, part: Ptr, stream: cuda.Stream) !void {
        if (w.sk == 1) return g.f16mmOf(w, x, x_ty, x_stride, m, out, o_stride, false, stream);
        try g.f16mmOf(w, x, x_ty, x_stride, m, part, 0, true, stream);
        try g.reduce(part, w.sk, m, w.n, out, o_stride, stream);
    }

    /// forward.py `shift_windows(old, new, keep, channels)` for `layers` layers: rows j < T of each layer's window
    /// become rows keep + j of [old (T rows); new rows].
    pub fn shiftWindows(g: Glue, old: Ptr, old_l: usize, new: Ptr, new_l: usize, new_row: usize, keep: usize, layers: usize, channels: usize, t: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_shift_windows", .{ @intCast(layers), @intCast((channels + 255) / 256), 1 }, &.{
            aot.ptr("OLD", "*bf16", old), aot.ptr("NEW", "*bf16", new), aot.int("keep", n32(keep)),
            aot.int("OLD_L", n32(old_l)), aot.int("NEW_L", n32(new_l)),  aot.int("NEW_ROW", n32(new_row)),
        }, &.{ aot.ci("C", @intCast(channels)), aot.ci("T", @intCast(t)), aot.ci("TP", @intCast(std.math.ceilPowerOfTwoAssert(usize, t))), aot.ci("BLOCK", 256) });
    }

    /// tensorfold.cuda.moe `router`: out [m, ne] fp32 = x [m, D] bf16 (row stride x_stride) . rows [ne, D] bf16.
    pub fn router(g: Glue, x: Ptr, x_stride: usize, rows: Ptr, ne: usize, out: Ptr, m: usize, stream: cuda.Stream) !void {
        const bm: usize = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
        const be: usize, const bk: usize = if (bm == 16) .{ 32, 256 } else .{ 64, 64 };
        try g.set.run(stream, "_router", .{ @intCast((m + bm - 1) / bm), @intCast((ne + be - 1) / be), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("W", "*bf16", rows), aot.ptr("OUT", "*fp32", out), aot.int("M", n32(m)), aot.int("x_stride", n32(x_stride)),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("NE", @intCast(ne)), aot.ci("BM", @intCast(bm)), aot.ci("BLOCK_E", @intCast(be)), aot.ci("BK", @intCast(bk)) });
    }

    /// tensorfold.cuda.moe `select_rows`: each row's top-k experts (pick int32, weights fp32), the shared expert last.
    pub fn selectRows(g: Glue, logits: Ptr, experts: usize, topk: usize, pick: Ptr, wts: Ptr, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_topk_rows", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("L", "*fp32", logits), aot.ptr("PICK", "*i32", pick), aot.ptr("WTS", "*fp32", wts),
        }, &.{
            aot.ci("NE", @intCast(experts)),                                                       aot.ci("NL", @intCast(experts + 1)), aot.ci("TOPK", @intCast(topk)),
            aot.ci("SLOTS", @intCast(topk + 1)),                                                   aot.ci("BLOCK", @intCast(std.math.ceilPowerOfTwoAssert(usize, experts + 1))),
            aot.ci("SLOTP", @intCast(std.math.ceilPowerOfTwoAssert(usize, topk + 1))),
        });
    }

    /// glue.py `hc_act`: the down projection's rows dn [rows, ndn] bf16 (already summed) -> act [rows, low] bf16 =
    /// bf16(silu(bf16(v / S))), its group sums, and (when `inject` is not 0) the inject gates bf16 [rows, S].
    pub fn hcAct(g: Glue, dn: Ptr, ndn: usize, act: Ptr, xs: Ptr, inject: Ptr, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_hc_act", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("DN", "*bf16", dn), aot.ptr("ACT", "*bf16", act), aot.ptr("XS", "*fp32", xs),
            aot.ptr("INJ", "*bf16", if (inject != 0) inject else act),
        }, &.{
            aot.ci("S", @intCast(g.streams)), aot.ci("LOW", @intCast(g.low)), aot.ci("LOWP", @intCast(std.math.ceilPowerOfTwoAssert(usize, g.low))),
            aot.ci("HAS_INJ", @intFromBool(inject != 0)), aot.ci("NDN", @intCast(ndn)),
        });
    }

    /// exl3/prefill.py `_gemm`: out [m, n] (row stride o_stride, bf16) = ((xh @ W_q) @ H) * scale * svh for the rotated
    /// inputs xh [m, k] fp16 and the decoded W_q [k, n] fp16, a 128-column Hadamard block a program, K in 32 steps.
    pub fn gemm(g: Glue, xh: Ptr, wq: Ptr, hadamard: Ptr, svh: Ptr, out: Ptr, o_stride: usize, m: usize, k: usize, n: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_gemm", .{ @intCast(((m + 127) / 128) * (n / 128)), 1, 1 }, &.{
            aot.ptr("X", "*fp16", xh), aot.ptr("W", "*fp16", wq), aot.ptr("H", "*bf16", hadamard), aot.ptr("SVH", "*fp16", svh),
            aot.ptr("BIAS", "*fp16", svh), aot.ptr("OUT", "*bf16", out), aot.int("M", n32(m)), aot.int("o_stride", n32(o_stride)),
        }, &.{
            aot.ci("K", @intCast(k)), aot.ci("N", @intCast(n)), aot.ci("BM", 128), aot.ci("BK", 32), aot.ci("GROUP", 8),
            aot.ci("HAS_BIAS", 0),    aot.cf("SCALE", had_scale),
        });
    }

    /// exl3_mm.py `F16.__call__` for any row count: one launch when the matrix has no K slices, else 128-row calls of
    /// the slices and their in-order sum (rows are independent, so the calls keep every row's bits).
    pub fn f16Rows(g: Glue, w: *const F16, x: Ptr, x_ty: []const u8, x_stride: usize, m: usize, out: Ptr, o_stride: usize, part: Ptr, stream: cuda.Stream) !void {
        if (w.sk == 1) {
            try g.set.run(stream, "_f16_mm", .{ @intCast((m + 15) / 16), @intCast((w.n + 63) / 64), 1 }, &.{
                aot.ptr("X", x_ty, x),        aot.ptr("W", "*fp16", w.w.ptr),   aot.ptr("OUT", "*bf16", out),
                aot.int("M", n32(m)),         aot.int("N", n32(w.n)),           aot.int("x_stride", n32(x_stride)),
                aot.int("o_stride", n32(o_stride)),
            }, &.{ aot.ci("K", @intCast(w.k)), aot.ci("KS", @intCast(w.k)), aot.ci("BM", 16), aot.ci("BN", 64), aot.ci("BK", 64), aot.ci("F32", 0) });
            return;
        }
        const esize: usize = if (std.mem.eql(u8, x_ty, "*fp32")) 4 else 2;
        var r0: usize = 0;
        while (r0 < m) : (r0 += call_rows) {
            const rows = @min(call_rows, m - r0);
            try g.f16mmOf(w, x + r0 * x_stride * esize, x_ty, x_stride, rows, part, 0, true, stream);
            try g.reduce(part, w.sk, rows, w.n, out + r0 * o_stride * 2, o_stride, stream);
        }
    }

    /// exl3_mm.py `ple_rows`: staged table rows (row r * heads + h: an fp16 scale word, then dh `bits`-bit values)
    /// -> out [rows, heads * dh] fp16, decoded as ExLlamaV3's `ngram_dequant` plus each head's bias.
    pub fn pleRows(g: Glue, staged: Ptr, words: usize, head_bias: Ptr, heads: usize, dh: usize, bits: usize, out: Ptr, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_ple_rows", .{ @intCast(rows), @intCast(heads), 1 }, &.{
            aot.ptr("PK", "*i16", staged), aot.ptr("HB", "*fp16", head_bias), aot.ptr("OUT", "*fp16", out),
        }, &.{
            aot.ci("WORDS", @intCast(words)), aot.ci("KB", @intCast(bits)), aot.ci("HEADS", @intCast(heads)), aot.ci("DH", @intCast(dh)),
            aot.ci("BLOCK", @intCast(std.math.ceilPowerOfTwoAssert(usize, dh))), aot.cf("K_INV", ple_k_inv), aot.cf("K_BIAS", ple_k_bias),
        });
    }

    /// glue.py `ple_gate`: the n-gram branch's key streams against the residual streams h, gating the values into
    /// gated [rows, S * D] bf16 and their per-stream sums of squares pss fp32 [rows, S].
    pub fn pleGate(g: Glue, keys: Ptr, vals: Ptr, h: Ptr, norm_key: Ptr, norm_query: Ptr, gated: Ptr, pss: Ptr, eps: f32, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_ple_gate", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("KEYS", "*bf16", keys),     aot.ptr("VALS", "*bf16", vals),  aot.ptr("H", "*bf16", h),
            aot.ptr("NK", "*fp32", norm_key),   aot.ptr("NQ", "*fp32", norm_query), aot.ptr("GATED", "*bf16", gated),
            aot.ptr("PSS", "*fp32", pss),       aot.float("eps", eps),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("BLOCK", 512) });
    }

    /// glue.py `ple_conv(gated, pss, norm_conv, tail, conv_w, h, h, nrow, eps, S, dilation)`: the dilated conv over
    /// [tail; rows] added into the streams h in place, each row's normed values kept in nrow for the next tail.
    pub fn pleConv(g: Glue, gated: Ptr, pss: Ptr, norm_conv: Ptr, tail: Ptr, conv_w: Ptr, taps: usize, dilation: usize, h: Ptr, nrow: Ptr, eps: f32, rows: usize, stream: cuda.Stream) !void {
        const wide = g.streams * g.hidden;
        try g.set.run(stream, "_ple_conv", .{ @intCast(rows), @intCast(wide / 512), 1 }, &.{
            aot.ptr("GATED", "*bf16", gated), aot.ptr("PSS", "*fp32", pss),  aot.ptr("NC", "*fp32", norm_conv),
            aot.ptr("TAIL", "*bf16", tail),   aot.ptr("CW", "*bf16", conv_w), aot.ptr("H", "*bf16", h),
            aot.ptr("HOUT", "*bf16", h),      aot.ptr("NROW", "*bf16", nrow), aot.float("eps", eps),
            aot.int("R", n32(rows)),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("TAPS", @intCast(taps)), aot.ci("DIL", @intCast(dilation)), aot.ci("BLOCK", 512) });
    }
};

/// exl3/prefill.py HAD_SCALE: 1 / sqrt(128), as Python passes it (a float).
pub const had_scale: f32 = @floatCast(@as(f64, 0.08838834764831845));

/// exl3_mm.py K_INV, K_BIAS: the mul1 codebook's fp16 constants 0x1EEE and 0xC931, as Python passes them (floats).
pub const ple_k_inv: f32 = @floatCast(@as(f16, @bitCast(@as(u16, 0x1EEE))));
pub const ple_k_bias: f32 = @floatCast(@as(f16, @bitCast(@as(u16, 0xC931))));

