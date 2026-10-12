//! Proposer operations. Target verification remains responsible for emitted tokens.
const cuda = @import("cuda");
const B = @import("cuda_dflash_buffers.zig");
const W = @import("cuda_dflash_weights.zig");
const K = @import("cuda_dflash_kernels.zig").Kernels;
const Q = @import("cuda_dflash_qmm.zig").Kernels;
const G = @import("cuda_glue.zig").Glue;
const ring = @import("cuda_dflash_ring.zig");

/// Append one committed tap tile [rows,20480] BF16, rows<=64.
/// cos/sin are reference FP32 rotary tables [rows,64] at context.logical.end.
/// Their arithmetic generation is an independent gate; this interface also accepts fixture tables.
/// Only a complete five-layer update advances context. A launch failure invalidates it until reset.
pub fn addTaps(context: *B.Context, model: *const W.Model, scratch: B.Scratch, kernels: K, projections: Q, glue: G, taps: cuda.DeviceBuffer, cos: cuda.DeviceBuffer, sin: cuda.DeviceBuffer, rows: usize, eps: f32, stream: cuda.Stream) !void {
    if (!context.valid) return error.InvalidDflashContext;
    if (rows == 0 or rows > 64 or taps.len < rows * 20480 * 2 or cos.len < rows * 64 * 4 or sin.len < rows * 64 * 4) return error.BadDflashTapTile;
    if (rows > context.logical.capacity - context.logical.end) return error.DflashContextOverflow;
    const sums = scratch.get("sums").ptr;
    const ctx = scratch.get("context").ptr;
    const normed = scratch.get("normed").ptr;
    const qkv = scratch.get("qkv").ptr;
    const key = scratch.get("key");
    const value = scratch.get("value");
    const partials = scratch.get("matrix_partials").ptr;
    errdefer context.valid = false;
    try glue.groupSums(taps.ptr, 20480, sums, rows, 20480, stream);
    try projections.matmul(&model.fc, taps.ptr, sums, ctx, rows, false, partials, stream);
    try glue.rmsnorm(ctx, 4096, model.hidden_norm.ptr, normed, 4096, sums, eps, rows, 4096, stream);
    for (model.layers, 0..) |layer, i| {
        try projections.matmul(&layer.kv, normed, sums, qkv, rows, false, partials, stream);
        try kernels.prep(qkv, layer.q_norm.ptr, layer.k_norm.ptr, cos.ptr, sin.ptr, 0, key.ptr, value.ptr, rows, 0, eps, stream);
        try ring.write(context.keys[i], key, context.logical.end, rows, stream);
        try ring.write(context.values[i], value, context.logical.end, rows, stream);
    }
    try context.commitRows(rows, stream);
}
