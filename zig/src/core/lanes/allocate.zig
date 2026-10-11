//! Shared rows by landing probability and forward costs between timed widths (Python allocate.py, extend_costs).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;

pub const Cost = struct { width: u32, ms: f64 };

/// Draft prefixes per stream with the most expected tokens a ms; fixed rows always run. A round's ms is `costs` at
/// its total rows, plus each stream's `row_ms[s][r]` at its own rows r (the share its position adds; `row_ms`
/// empty: none, else one slice a stream reaching at least `fixed[s] + probs[s].len`), plus `overhead_ms`. Nodes join
/// in order of chance, so the counts returned are the best prefix of that order. Caller frees.
pub fn allocate(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, row_ms: []const []const f64, overhead_ms: f64, max_rows: u64) ![]u32 {
    const counts = try gpa.alloc(u32, probs.len);
    defer gpa.free(counts);
    const best_counts = try gpa.alloc(u32, probs.len);
    @memset(counts, 0);
    @memset(best_counts, 0);
    var rows: u64 = 0;
    for (fixed) |f| rows += f;
    var expected: f64 = @floatFromInt(rows);
    const timed = !costs.empty();
    var extra: f64 = 0.0;
    if (row_ms.len > 0) for (row_ms, fixed) |m, f| {
        extra += m[f];
    };
    var best = rate(costs, timed, rows, expected, overhead_ms + extra);
    while (rows < max_rows) {
        // the likeliest next node, ties to the lowest stream (Python's heapq of (-p, stream))
        var pick: ?usize = null;
        var pick_p: f64 = 0.0;
        for (probs, 0..) |p, s| {
            if (counts[s] >= p.len) continue;
            const v = p[counts[s]];
            if (pick == null or v > pick_p) {
                pick = s;
                pick_p = v;
            }
        }
        const s = pick orelse break;
        if (row_ms.len > 0) extra += row_ms[s][fixed[s] + counts[s] + 1] - row_ms[s][fixed[s] + counts[s]];
        counts[s] += 1;
        rows += 1;
        expected += pick_p;
        const now = rate(costs, timed, rows, expected, overhead_ms + extra);
        if (now > best) {
            best = now;
            @memcpy(best_counts, counts);
        }
    }
    return best_counts;
}

fn rate(costs: *const Table, timed: bool, total: u64, expected: f64, overhead_ms: f64) f64 {
    var cost: f64 = 1.0;
    if (timed) cost = costs.get(@intCast(total)) orelse return -1.0;
    return expected / (cost + overhead_ms);
}

/// A window's forward ms by rows at several stream positions (timed at load), as the ms a stream's rows add at its
/// position over the shallowest table's: `extraMs(p, r)` is 0 at or before the shallowest position, the difference
/// of the two tables around p interpolated linearly in p between their positions, and past the deepest that of the
/// two deepest extended along the same line (a row's attention reads grow with its position). Rows past a table's
/// widest follow `extendCosts`. Fewer than two positions: always 0.
pub const PositionCosts = struct {
    positions: []u64 = &.{},
    tables: []Table = &.{},

    pub const Timed = struct { position: u64, costs: []const Cost };

    /// The timed tables (any order; a position given twice keeps the last), each extended to `rows`.
    pub fn init(gpa: Allocator, timed: []const Timed, rows: u32) !PositionCosts {
        var pc: PositionCosts = .{};
        errdefer pc.deinit(gpa);
        var order: std.ArrayList(Timed) = .empty;
        defer order.deinit(gpa);
        for (timed) |t| {
            if (t.costs.len == 0) continue;
            for (order.items) |*o| {
                if (o.position == t.position) {
                    o.* = t;
                    break;
                }
            } else try order.append(gpa, t);
        }
        std.mem.sort(Timed, order.items, {}, struct {
            fn less(_: void, a: Timed, b: Timed) bool {
                return a.position < b.position;
            }
        }.less);
        if (order.items.len < 2) return pc;
        pc.positions = try gpa.alloc(u64, order.items.len);
        pc.tables = try gpa.alloc(Table, order.items.len);
        @memset(pc.tables, .{});
        for (order.items, pc.positions, pc.tables) |t, *p, *table| {
            p.* = t.position;
            table.* = try extendCosts(gpa, t.costs, rows);
        }
        return pc;
    }

    pub fn deinit(pc: *PositionCosts, gpa: Allocator) void {
        for (pc.tables) |*t| t.deinit(gpa);
        gpa.free(pc.tables);
        gpa.free(pc.positions);
        pc.* = .{};
    }

    /// The ms `rows` rows of a stream at `position` add over the shallowest table's (0 for 0 rows).
    pub fn extraMs(pc: PositionCosts, position: u64, rows: u32) f64 {
        const n = pc.positions.len;
        if (n < 2 or rows == 0 or position <= pc.positions[0]) return 0.0;
        var hi: usize = 1;
        while (hi + 1 < n and pc.positions[hi] < position) hi += 1;
        const lo = hi - 1;
        const base = pc.tables[0].get(rows) orelse return 0.0;
        const a = (pc.tables[lo].get(rows) orelse return 0.0) - base;
        const b = (pc.tables[hi].get(rows) orelse return 0.0) - base;
        const p0: f64 = @floatFromInt(pc.positions[lo]);
        const p1: f64 = @floatFromInt(pc.positions[hi]);
        return a + (b - a) * (@as(f64, @floatFromInt(position)) - p0) / (p1 - p0);
    }

    /// `out[r]` = `extraMs(position, r)` for r in 0 .. out.len - 1 (the row prices `allocate` reads).
    pub fn rowsAt(pc: PositionCosts, position: u64, out: []f64) void {
        for (out, 0..) |*o, r| o.* = pc.extraMs(position, @intCast(r));
    }

    pub fn empty(pc: PositionCosts) bool {
        return pc.positions.len < 2;
    }
};

/// A chain's nodes' chances from per-depth acceptance (the j-th lands if every earlier one did). Caller frees.
pub fn chainProbabilities(gpa: Allocator, rates: []const f64, count: usize) ![]f64 {
    const out = try gpa.alloc(f64, count);
    var reach: f64 = 1.0;
    for (out, 0..) |*o, j| {
        reach *= if (j < rates.len) rates[j] else rates[rates.len - 1];
        o.* = reach;
    }
    return out;
}

/// Costs for 1..rows: timed widths as measured, interpolated between them, the widest's ms a row past it.
pub fn extendCosts(gpa: Allocator, costs: []const Cost, rows: u32) !Table {
    var out: Table = .{};
    if (costs.len == 0) return out;
    const measured = try gpa.dupe(Cost, costs);
    defer gpa.free(measured);
    std.mem.sort(Cost, measured, {}, struct {
        fn less(_: void, a: Cost, b: Cost) bool {
            return a.width < b.width;
        }
    }.less);
    const widest = measured[measured.len - 1];
    var total: u32 = 1;
    while (total <= rows) : (total += 1) {
        var value: f64 = undefined;
        if (find(measured, total)) |ms| {
            value = ms;
        } else if (total > widest.width) {
            value = widest.ms * @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(widest.width));
        } else {
            var above: Cost = undefined;
            for (measured) |c| {
                if (c.width > total) {
                    above = c;
                    break;
                }
            }
            var below: ?Cost = null;
            for (measured) |c| {
                if (c.width < total) below = c;
            }
            if (below) |b| {
                const span: f64 = @floatFromInt(above.width - b.width);
                value = b.ms + (above.ms - b.ms) * @as(f64, @floatFromInt(total - b.width)) / span;
            } else value = above.ms;
        }
        try out.put(gpa, total, value);
    }
    return out;
}

fn find(costs: []const Cost, width: u32) ?f64 {
    for (costs) |c| if (c.width == width) return c.ms;
    return null;
}

test "allocate takes the widest round when no cost is timed" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    const probs = [_][]const f64{ &.{ 0.9, 0.5 }, &.{0.8} };
    const counts = try allocate(gpa, &.{ 1, 1 }, &probs, &empty, &.{}, 0.0, 8);
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1 }, counts);
}

test "extendCosts interpolates and extrapolates" {
    const gpa = std.testing.allocator;
    var t = try extendCosts(gpa, &.{ .{ .width = 4, .ms = 8.0 }, .{ .width = 2, .ms = 4.0 } }, 6);
    defer t.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 4.0), t.get(1));
    try std.testing.expectEqual(@as(?f64, 6.0), t.get(3));
    try std.testing.expectEqual(@as(?f64, 12.0), t.get(6));
}

test "allocate prices a stream's rows at its position" {
    const gpa = std.testing.allocator;
    // 10 ms a round plus 1 ms a row; stream 1's rows cost 5 ms more each at its position
    var costs = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 17.0 } }, 8);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.6, 0.36 }, &.{ 0.6, 0.36 } };
    const flat = [_]f64{ 0, 0, 0, 0 };
    const deep = [_]f64{ 0, 5, 10, 15 };
    const level = try allocate(gpa, &.{ 1, 1 }, &probs, &costs, &.{ &flat, &flat }, 0.0, 8);
    defer gpa.free(level);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2 }, level);
    const priced = try allocate(gpa, &.{ 1, 1 }, &probs, &costs, &.{ &flat, &deep }, 0.0, 8);
    defer gpa.free(priced);
    // stream 1's pending row is paid whatever it drafts; its drafts no longer pay, and the round is then dear
    // enough that stream 0's second draft does not either
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, priced);
}

test "PositionCosts interpolates by position and extends past the deepest" {
    const gpa = std.testing.allocator;
    var pc = try PositionCosts.init(gpa, &.{
        .{ .position = 65536, .costs = &.{ .{ .width = 1, .ms = 12.0 }, .{ .width = 4, .ms = 18.0 } } },
        .{ .position = 1024, .costs = &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 4, .ms = 13.0 } } },
        .{ .position = 263168, .costs = &.{ .{ .width = 1, .ms = 14.0 }, .{ .width = 4, .ms = 25.0 } } },
    }, 4);
    defer pc.deinit(gpa);
    try std.testing.expect(!pc.empty());
    try std.testing.expectEqual(@as(f64, 0.0), pc.extraMs(0, 4));
    try std.testing.expectEqual(@as(f64, 0.0), pc.extraMs(1024, 4));
    try std.testing.expectEqual(@as(f64, 0.0), pc.extraMs(100000, 0));
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), pc.extraMs(65536, 4), 1e-9);
    // halfway between 65536 and 263168: 5 and 12 ms at 4 rows
    try std.testing.expectApproxEqAbs(@as(f64, 8.5), pc.extraMs(164352, 4), 1e-9);
    // a row count between timed widths: the extended tables' (3 rows: 16 ms at 65536, 12 ms at 1024)
    try std.testing.expectApproxEqAbs(@as(f64, 16.0 - 12.0), pc.extraMs(65536, 3), 1e-9);
    // past the deepest: the line through the two deepest
    try std.testing.expectApproxEqAbs(@as(f64, 19.0), pc.extraMs(263168 + 197632, 4), 1e-9);
    var row: [5]f64 = undefined;
    pc.rowsAt(65536, &row);
    try std.testing.expectEqual(@as(f64, 0.0), row[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), row[1], 1e-9);
    // one table: no position prices
    var one = try PositionCosts.init(gpa, &.{.{ .position = 1024, .costs = &.{.{ .width = 1, .ms = 10.0 }} }}, 4);
    defer one.deinit(gpa);
    try std.testing.expect(one.empty());
    try std.testing.expectEqual(@as(f64, 0.0), one.extraMs(1 << 20, 4));
}
