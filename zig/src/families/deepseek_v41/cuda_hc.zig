//! The hyper-connection (mHC) glue around DeepSeek-V4.1's sublayers and the head's collapse, as the Python family's
//! `kernels.py` launches them (`embed_init`, `hc_pre`, `hc_post`, `collapse_norm`, `tap`, `engram_gate`): the fork's
//! Triton kernels replayed through the captured AOT set, each launch's grid, arguments and constexprs the wrapper's.
//!
//! Streams: h [rows, 4, D] bf16 (hc = 4 residual streams); pre/post [rows, 4] f32 and comb [rows, 4, 4] f32 are a
//! sublayer's mixes; `part` [rows * 40 * 32] f32 holds hc_pre's partial dots (HC_BLOCKS = 40 is the fixed K split).
//! A sublayer reads the mixes the previous one computed (delayed collapse): hc_pre's `pre_in` is the previous
//! sublayer's `pre_out`.
const std = @import("std");
const cuda = @import("cuda");

const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;

pub const hc_streams = 4;
pub const hc_blocks = 40;
pub const block = 1024;

pub const Hc = struct {
    set: *const aot.Set,
    d: u32,
    iters: u32,
    eps: f32,
    hc_eps: f32,
    /// Whether the fork launched its decode kernels as programmatic dependent launches (TF_DS_TRITON_PDL, default 1):
    /// the captured variants carry the constexpr, so the replay passes the same.
    pdl: bool = true,

    pub fn init(set: *const aot.Set, d: u32, iters: u32, eps: f32, hc_eps: f32) Hc {
        return .{ .set = set, .d = d, .iters = iters, .eps = eps, .hc_eps = hc_eps };
    }

    /// f32 words of `part` for `rows`.
    pub fn partWords(rows: usize) usize {
        return rows * hc_blocks * 32;
    }

    fn pdlConst(self: *const Hc) aot.Const {
        return aot.ci("PDL", @intFromBool(self.pdl));
    }

    /// h [rows, 4, D] = embed[ids] on every stream; pre = [1, 0, 0, 0] a row.
    pub fn embedInit(self: *const Hc, stream: cuda.Stream, embed: Ptr, ids: Ptr, h: Ptr, pre_: Ptr, rows: u32) !void {
        try self.set.run(stream, "_embed_init", .{ rows, self.d / block, 1 }, &.{
            aot.ptr("EMB", "*bf16", embed), aot.ptr("IDS", "*i64", ids), aot.ptr("H", "*bf16", h), aot.ptr("PRE", "*fp32", pre_),
        }, &.{ aot.ci("D", self.d), aot.ci("HC", hc_streams), aot.ci("BLOCK", block), self.pdlConst() });
    }

    /// out [rows, D] bf16 = RMSNorm(sum_j pre_in[j] h_j) * norm_w; pre_out, post, comb from h's own mixes
    /// (fn [24, 4D] f32, scale [3] f32, base [24] f32). Two launches: the partial dots, then the finish.
    pub fn pre(self: *const Hc, stream: cuda.Stream, h: Ptr, fn_: Ptr, scale: Ptr, base: Ptr, pre_in: Ptr, norm_w: Ptr, out: Ptr, pre_out: Ptr, post_: Ptr, comb: Ptr, part: Ptr, rows: u32) !void {
        const wide: i64 = 4 * @as(i64, self.d);
        try self.set.run(stream, "_hc_partial", .{ rows, hc_blocks, 1 }, &.{
            aot.ptr("X", "*bf16", h), aot.ptr("FN", "*fp32", fn_), aot.ptr("PART", "*fp32", part),
        }, &.{ aot.ci("WIDE", wide), aot.ci("NB", hc_blocks), aot.ci("SUB", 128) });
        try self.set.run(stream, "_hc_finish", .{ rows, 1, 1 }, &.{
            aot.ptr("X", "*bf16", h),           aot.ptr("PART", "*fp32", part),   aot.ptr("BASE", "*fp32", base),
            aot.ptr("SCALE", "*fp32", scale),   aot.ptr("PRE_IN", "*fp32", pre_in), aot.ptr("NW", "*bf16", norm_w),
            aot.ptr("OUT", "*bf16", out),       aot.ptr("PRE_OUT", "*fp32", pre_out), aot.ptr("POST", "*fp32", post_),
            aot.ptr("COMB", "*fp32", comb),     aot.float("eps", self.eps),       aot.float("hc_eps", self.hc_eps),
        }, &.{ aot.ci("D", self.d), aot.ci("NB", hc_blocks), aot.ci("ITERS", self.iters), aot.ci("BLOCK", block) });
    }

    /// gathered [world, rows, D] f32 partials (rank order) -> y = bf16(sum); out_k = post_k y + sum_j comb[j, k] h_j
    /// -> out [rows, 4, D] bf16 (`out` may be `h`).
    pub fn post(self: *const Hc, stream: cuda.Stream, gathered: Ptr, world: u32, h: Ptr, post_: Ptr, comb: Ptr, out: Ptr, rows: u32) !void {
        const rs: i32 = @intCast(rows * self.d);
        try self.set.run(stream, "_hc_post", .{ rows, self.d / block, 1 }, &.{
            aot.ptr("G", "*fp32", gathered), aot.int("RS", rs),           aot.ptr("X", "*bf16", h),
            aot.ptr("XOUT", "*bf16", out),   aot.ptr("POST", "*fp32", post_), aot.ptr("COMB", "*fp32", comb),
        }, &.{ aot.ci("D", self.d), aot.ci("WORLD", world), aot.ci("BLOCK", block), self.pdlConst() });
    }

    /// out [rows, D] bf16 = RMSNorm(sum_j pre[j] h_j) * w: the head's input.
    pub fn collapseNorm(self: *const Hc, stream: cuda.Stream, h: Ptr, pre_: Ptr, w: Ptr, out: Ptr, rows: u32) !void {
        try self.set.run(stream, "_collapse_norm", .{ rows, 1, 1 }, &.{
            aot.ptr("X", "*bf16", h), aot.ptr("PRE", "*fp32", pre_), aot.ptr("NW", "*bf16", w), aot.ptr("OUT", "*bf16", out), aot.float("eps", self.eps),
        }, &.{ aot.ci("D", self.d), aot.ci("BLOCK", block), self.pdlConst() });
    }

    /// out [rows, D] (a column block of the taps buffer, `out_stride` elements between rows) = bf16(mean over the 4
    /// streams of h in f32): a DSpark tap.
    pub fn tap(self: *const Hc, stream: cuda.Stream, h: Ptr, out: Ptr, out_stride: i32, rows: u32) !void {
        try self.set.run(stream, "_tap", .{ rows, self.d / block, 1 }, &.{
            aot.ptr("H", "*bf16", h), aot.ptr("OUT", "*bf16", out), aot.int("os_", out_stride),
        }, &.{ aot.ci("D", self.d), aot.ci("BLOCK", block), self.pdlConst() });
    }

    /// Engram's gate: kv [rows, 5D] bf16 (4 keys, then the value), qk [4, D] f32 (q_weight * k_weight) -> out
    /// [rows, 4, D] = h_s + sigmoid(sign(dot) sqrt(|dot|)) * value.
    pub fn engramGate(self: *const Hc, stream: cuda.Stream, h: Ptr, kv: Ptr, qk: Ptr, out: Ptr, rows: u32) !void {
        try self.set.run(stream, "_engram_gate", .{ rows, hc_streams, 1 }, &.{
            aot.ptr("H", "*bf16", h), aot.ptr("KV", "*bf16", kv), aot.ptr("QK", "*fp32", qk), aot.ptr("OUT", "*bf16", out), aot.float("eps", self.eps),
        }, &.{ aot.ci("D", self.d), aot.ci("BLOCK", block), self.pdlConst() });
    }
};
