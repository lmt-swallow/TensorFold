//! Device copies for the proposer KV ring; this never changes committed length.
const cuda = @import("cuda");
const D = @import("dflash2.zig");

/// Copy contiguous [4,rows,128] BF16 into [4,2176,128] at absolute position.
/// Source and destination allocations must be disjoint and owned by this sequence.
pub fn write(destination: cuda.DeviceBuffer, source: cuda.DeviceBuffer, position: usize, rows: usize, stream: cuda.Stream) !void {
    const cap = 2176;
    const width = 128 * 2;
    if (rows > 64 or source.len < 4 * rows * width or destination.len != 4 * cap * width) return error.BadDflashRingWrite;
    const plan = try D.ringWrite(position, rows, cap);
    const src_end = source.ptr + source.len;
    const dst_end = destination.ptr + destination.len;
    if (source.ptr < dst_end and destination.ptr < src_end) return error.AliasedDflashRingWrite;
    for (0..4) |head| {
        for (plan.spans[0..plan.count]) |span| {
            try destination.copyFrom((head * cap + span.ring_row) * width, source.ptr + (head * rows + span.source_row) * width, span.rows * width, stream.handle);
        }
    }
}
