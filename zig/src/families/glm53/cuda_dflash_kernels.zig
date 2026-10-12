//! Reference DFlash2 Triton launches. Missing captured variants fail closed.
const cuda = @import("cuda");
const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;

pub const Kernels = struct {
    set: *aot.Set,

    pub fn conv(k: Kernels, x: Ptr, dyn: Ptr, base: Ptr, residual: ?Ptr, out: Ptr, branch: u1, rows: usize, stream: cuda.Stream) !void {
        // One noncausal block only: adjacent streams must never share the previous-row convolution.
        if (rows != 8) return error.BadDflashRows;
        try k.set.run(stream, "_dconv_kernel", .{ @intCast(rows), 4, 1 }, &.{
            aot.ptr("X", "*bf16", x),       aot.ptr("DYN", "*bf16", dyn),
            aot.ptr("BASE", "*bf16", base), aot.ptr("RES", "*bf16", residual orelse x),
            aot.ptr("OUT", "*bf16", out),
        }, &.{ aot.ci("D", 4096), aot.ci("G", 256), aot.ci("GS", 16), aot.ci("BRANCH", branch), aot.ci("HAS_RES", @intFromBool(residual != null)), aot.ci("BLOCK", 1024) });
    }

    /// qkv is contiguous [rows, (heads + 8) * 128]; outputs are [heads, rows, 128].
    /// heads=0 updates committed tap KV; heads=16 prepares a speculative block.
    pub fn prep(k: Kernels, qkv: Ptr, qnorm: Ptr, knorm: Ptr, cos: Ptr, sin: Ptr, q: Ptr, key: Ptr, value: Ptr, rows: usize, heads: usize, eps: f32, stream: cuda.Stream) !void {
        if (rows == 0 or rows > 64 or (heads != 0 and heads != 16)) return error.BadDflashRows;
        try k.set.run(stream, "_prep_kernel", .{ @intCast(rows), @intCast(heads + 8), 1 }, &.{
            aot.ptr("QKV", "*bf16", qkv),                   aot.ptr("QN", "*bf16", qnorm), aot.ptr("KN", "*bf16", knorm),
            aot.ptr("COS", "*fp32", cos),                   aot.ptr("SIN", "*fp32", sin),  aot.ptr("QO", "*bf16", if (heads == 0) qkv else q),
            aot.ptr("KO", "*bf16", key),                    aot.ptr("VO", "*bf16", value), aot.int("L", @intCast(rows)),
            aot.int("stride", @intCast((heads + 8) * 128)), aot.float("eps", eps),
        }, &.{ aot.ci("H", @intCast(heads)), aot.ci("HKV", 4), aot.ci("HALF", 64) });
    }

    /// pos points to the committed absolute length (i64); this launch does not advance it.
    /// Each ring is [4, capacity, 128] BF16; shared/target caches are not inputs.
    pub fn attention(k: Kernels, q: Ptr, key: Ptr, value: Ptr, out: Ptr, pos: Ptr, capacity: usize, stream: cuda.Stream) !void {
        if (capacity != 2176) return error.BadDflashRing;
        try k.set.run(stream, "_dattn_kernel", .{ 4, 1, 1 }, &.{
            aot.ptr("Q", "*bf16", q),                aot.ptr("K", "*bf16", key),  aot.ptr("V", "*bf16", value),
            aot.ptr("OUT", "*bf16", out),            aot.ptr("POS", "*i64", pos), aot.int("window", 2047),
            aot.float("scale", 0.08838834764831845),
        }, &.{ aot.ci("N", 8), aot.ci("G", 4), aot.ci("NH", 16), aot.ci("HD", 128), aot.ci("CAP", @intCast(capacity)), aot.ci("BK", 64), aot.ci("CAUSAL", 0), aot.ci("RING", 1), aot.ci("SKIP", 1) });
    }
};
