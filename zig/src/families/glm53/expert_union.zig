//! Counts routed experts from row-major router picks. Shared slots never contribute to routed weight bytes.
const std = @import("std");
pub const max_experts = 288;
pub const max_rows = 16;
pub const max_slots = 9;
pub const max_layers = 42;
pub const max_words = max_layers * max_rows * max_slots;
pub const Span = struct { start: usize, rows: usize };

/// Counts the union across a contiguous set of rows. IDs equal to `experts` denote the shared slot;
/// negative and greater IDs are invalid rather than silently interpreted as padding.
pub fn count(picks: []const i32, rows: usize, slots: usize, experts: usize, span: Span) !usize {
    if (rows > max_rows or slots == 0 or slots > max_slots or experts == 0 or experts > max_experts or picks.len != rows * slots) return error.InvalidShape;
    if (span.start > rows or span.rows > rows - span.start) return error.InvalidSpan;
    var seen: [max_experts]bool = @splat(false);
    var n: usize = 0;
    for (picks[span.start * slots .. (span.start + span.rows) * slots]) |id| {
        if (id < 0 or id > experts) return error.InvalidExpert;
        if (id == experts) continue;
        const e: usize = @intCast(id);
        if (!seen[e]) {
            seen[e] = true;
            n += 1;
        }
    }
    return n;
}

/// The storage bound covers every main MoE layer and a full shared decode window.
pub fn words(layers: usize, rows: usize, slots: usize) !usize {
    if (layers == 0 or layers > max_layers or rows == 0 or rows > max_rows or slots == 0 or slots > max_slots) return error.InvalidShape;
    return layers * rows * slots;
}

test "union excludes shared, deduplicates rows, and distinguishes combined streams" {
    const ids = [_]i32{ 0, 1, 4, 1, 2, 4, 2, 3, 4 };
    try std.testing.expectEqual(4, try count(&ids, 3, 3, 4, .{ .start = 0, .rows = 3 }));
    try std.testing.expectEqual(3, try count(&ids, 3, 3, 4, .{ .start = 0, .rows = 2 }));
    try std.testing.expectEqual(2, try count(&ids, 3, 3, 4, .{ .start = 2, .rows = 1 }));
    try std.testing.expectError(error.InvalidExpert, count(&.{-1}, 1, 1, 4, .{ .start = 0, .rows = 1 }));
    try std.testing.expectError(error.InvalidExpert, count(&.{5}, 1, 1, 4, .{ .start = 0, .rows = 1 }));
    try std.testing.expectError(error.InvalidSpan, count(&ids, 3, 3, 4, .{ .start = 3, .rows = 1 }));
}

test "bounded storage and repeated union calls do not retain prior phase IDs" {
    try std.testing.expectEqual(max_words, try words(42, 16, 9));
    try std.testing.expectError(error.InvalidShape, words(43, 16, 9));
    try std.testing.expectError(error.InvalidShape, words(42, 17, 9));
    try std.testing.expectError(error.InvalidShape, words(42, 16, 10));
    for (1..max_experts + 1) |experts| {
        const a = [_]i32{ 0, @intCast(experts - 1), @intCast(experts) };
        try std.testing.expectEqual(@min(experts, 2), try count(&a, 1, 3, experts, .{ .start = 0, .rows = 1 }));
        try std.testing.expectEqual(0, try count(&.{@intCast(experts)}, 1, 1, experts, .{ .start = 0, .rows = 1 }));
    }
}
