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

const T = @import("cuda_mtp.zig");
const Ptr = cuda.abi.DevicePtr;

fn rowProjection(projections: Q, glue: G, weight: *const T.DraftHead, input: Ptr, scratch: B.Scratch, sums_ready: bool, gather: T.Gather, stream: cuda.Stream) !Ptr {
    const sums = scratch.get("sums").ptr;
    const partial = scratch.get("partial").ptr;
    const gathered = scratch.get("gathered").ptr;
    const out = scratch.get("context").ptr;
    if (!sums_ready) try glue.groupSums(input, weight.k, sums, 8, weight.k, stream);
    try projections.matmul(weight, input, sums, partial, 8, true, scratch.get("matrix_partials").ptr, stream);
    try gather.run(gather.ctx, partial, gathered, 8 * 4096, stream);
    try projections.sumRanks(gathered, out, 8 * 4096, stream);
    return out;
}

/// Execute one block [pending,mask*7]; ids and rotary tables must already be uploaded.
/// Leaves committed position unchanged. The final normalized rows/logits/projection are borrowed scratch views.
/// Context KV beyond its committed end is speculative and overwritten by the next addTaps.
pub fn block(context: *B.Context, model: *const W.Model, head: *const T.DraftHead, scratch: B.Scratch, kernels: K, projections: Q, glue: G, embedding: Ptr, cos: Ptr, sin: Ptr, eps: f32, gather: T.Gather, hidden_projection: *const cuda.cublaslt.Linear, stream: cuda.Stream) !void {
    if (!context.valid or context.logical.end == 0) return error.InvalidDflashContext;
    if (head.n != 77440 or head.k != 4096) return error.BadDflashProjection;
    const x = scratch.get("x").ptr;
    const normed = scratch.get("normed").ptr;
    const sums = scratch.get("sums").ptr;
    const temporary = scratch.get("context").ptr;
    const dyn = scratch.get("dyn").ptr;
    const qkv = scratch.get("qkv").ptr;
    const q = scratch.get("q").ptr;
    const key = scratch.get("key");
    const value = scratch.get("value");
    const attention = scratch.get("attention").ptr;
    const gu = scratch.get("gu").ptr;
    const activation = scratch.get("activation").ptr;
    const partials = scratch.get("matrix_partials").ptr;
    errdefer context.valid = false;
    try glue.embed(scratch.get("ids").ptr, embedding, x, 8, 1, stream);
    for (model.layers, 0..) |layer, i| {
        try glue.rmsnorm(x, 4096, layer.in_norm.ptr, normed, 4096, sums, eps, 8, 4096, stream);
        try projections.matmul(&layer.a_kp, normed, sums, dyn, 8, false, partials, stream);
        try kernels.conv(normed, dyn, layer.a_base.ptr, null, temporary, 0, 8, stream);
        try glue.groupSums(temporary, 4096, sums, 8, 4096, stream);
        try projections.matmul(&layer.qkv, temporary, sums, qkv, 8, false, partials, stream);
        try kernels.prep(qkv, layer.q_norm.ptr, layer.k_norm.ptr, cos, sin, q, key.ptr, value.ptr, 8, 16, eps, stream);
        try ring.write(context.keys[i], key, context.logical.end, 8, stream);
        try ring.write(context.values[i], value, context.logical.end, 8, stream);
        try kernels.attention(q, context.keys[i].ptr, context.values[i].ptr, attention, context.position.ptr, 2176, stream);
        const attn_row = try rowProjection(projections, glue, &layer.o, attention, scratch, false, gather, stream);
        try kernels.conv(attn_row, dyn, layer.a_base.ptr, x, x, 1, 8, stream);
        try glue.rmsnorm(x, 4096, layer.post_norm.ptr, normed, 4096, sums, eps, 8, 4096, stream);
        try projections.matmul(&layer.m_kp, normed, sums, dyn, 8, false, partials, stream);
        try kernels.conv(normed, dyn, layer.m_base.ptr, null, temporary, 0, 8, stream);
        try glue.groupSums(temporary, 4096, sums, 8, 4096, stream);
        try projections.matmul(&layer.gu, temporary, sums, gu, 8, false, partials, stream);
        try glue.swiglu(gu, activation, sums, 1e30, 8, 6144, stream);
        const mlp_row = try rowProjection(projections, glue, &layer.down, activation, scratch, true, gather, stream);
        try kernels.conv(mlp_row, dyn, layer.m_base.ptr, x, x, 1, 8, stream);
    }
    try glue.rmsnorm(x + 4096 * 2, 4096, model.norm.ptr, normed, 4096, sums, eps, 7, 4096, stream);
    try projections.matmul(head, normed, sums, scratch.get("logits").ptr, 7, false, partials, stream);
    // Reference F.linear has BF16 output; the CPU selector expands these bits to f32.
    // This library projection has an independent full-byte fixture gate.
    const workspace = scratch.get("blas_workspace");
    try hidden_projection.run(normed, model.hproj.ptr, scratch.get("projected").ptr, workspace.ptr, workspace.len, stream.handle);
}
