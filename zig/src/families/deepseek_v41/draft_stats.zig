//! Draft statistics a stream gathers over its reply (TF_DS_DRAFT_STATS=1, rank 0 logs them as the stream leaves):
//! how many drafts each round offered against how many the drafter held, how many landed, and, depth by depth,
//! the confidence head's predicted chance of landing against the observed share. They read the round loop's
//! choices; they never change one.
const std = @import("std");

/// Drafts a window holds at most (a window of 16 rows: the pending token and 15 drafts).
pub const max_drafts = 15;

pub fn enabled() bool {
    const v = std.mem.span(std.c.getenv("TF_DS_DRAFT_STATS") orelse return false);
    return std.mem.eql(u8, v, "1");
}

/// One stream's tallies. A round is `offer`ed when its window is verified and `land`ed once its kept rows are
/// known; a round never landed (the stream left) is not counted.
pub const Stats = struct {
    rounds: u64 = 0,
    /// rounds whose window carried host tokens (forced or copied), not the drafter's: not in the tallies below
    host_rounds: u64 = 0,
    /// rounds by drafts offered (index: drafts in the window)
    by_offered: [max_drafts + 1]u64 = @splat(0),
    /// drafts landed in the rounds that offered that many (index: drafts offered)
    kept_by_offered: [max_drafts + 1]u64 = @splat(0),
    /// rounds by drafts the drafter held (index: drafts held; offered <= held, the rest trimmed by the round loop)
    by_held: [max_drafts + 1]u64 = @splat(0),
    /// rounds that offered draft j (index j, depth j + 1) and those where it landed
    tried: [max_drafts]u64 = @splat(0),
    landed: [max_drafts]u64 = @splat(0),
    /// the sums of the confidence head's predicted chance that drafts 0 .. j all land, over the rounds that
    /// offered draft j
    predicted: [max_drafts]f64 = @splat(0),
    /// the rounds `predicted` sums over (those whose drafts carried a prediction)
    predicted_n: [max_drafts]u64 = @splat(0),
    /// the round waiting for its kept rows: drafts offered and their predicted survivals
    pending: ?struct { offered: usize, survival: [max_drafts]f64 } = null,

    /// A verified window: `offered` drafts out of `held`, `survival[j]` the predicted chance that drafts 0 .. j
    /// land (survival.len >= offered, or empty when the drafts carry no prediction).
    pub fn offer(self: *Stats, offered: usize, held: usize, survival: []const f64) void {
        const n = @min(offered, max_drafts);
        self.by_held[@min(held, max_drafts)] += 1;
        var p: [max_drafts]f64 = @splat(-1);
        if (survival.len >= n) @memcpy(p[0..n], survival[0..n]);
        self.pending = .{ .offered = n, .survival = p };
    }

    /// The pending round kept `kept` drafts (its path's rows less the pending token's); a no-op without one.
    pub fn land(self: *Stats, kept: usize) void {
        const p = self.pending orelse return;
        self.pending = null;
        const k = @min(kept, p.offered);
        self.rounds += 1;
        self.by_offered[p.offered] += 1;
        self.kept_by_offered[p.offered] += k;
        for (0..p.offered) |j| {
            self.tried[j] += 1;
            if (j < k) self.landed[j] += 1;
            if (p.survival[j] >= 0) {
                self.predicted[j] += p.survival[j];
                self.predicted_n[j] += 1;
            }
        }
    }

    pub fn add(self: *Stats, o: Stats) void {
        self.rounds += o.rounds;
        self.host_rounds += o.host_rounds;
        for (&self.by_offered, o.by_offered) |*a, b| a.* += b;
        for (&self.kept_by_offered, o.kept_by_offered) |*a, b| a.* += b;
        for (&self.by_held, o.by_held) |*a, b| a.* += b;
        for (&self.tried, o.tried) |*a, b| a.* += b;
        for (&self.landed, o.landed) |*a, b| a.* += b;
        for (&self.predicted, o.predicted) |*a, b| a.* += b;
        for (&self.predicted_n, o.predicted_n) |*a, b| a.* += b;
    }

    /// Tokens a drafted round brought on average: the kept drafts and the target's own token.
    pub fn tokensPerRound(self: *const Stats) f64 {
        if (self.rounds == 0) return 0;
        var kept: u64 = 0;
        for (self.kept_by_offered) |k| kept += k;
        return 1.0 + @as(f64, @floatFromInt(kept)) / @as(f64, @floatFromInt(self.rounds));
    }

    /// One line: rounds, tokens a drafted round, `offered=` rounds by drafts offered, `held=` by drafts held, and per depth
    /// `depth=` landed/tried (the observed share | the mean predicted survival, -1 without predictions).
    pub fn write(self: *const Stats, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("{d} rounds ({d} of host tokens), {d:.3} tokens a drafted round, offered=", .{ self.rounds, self.host_rounds, self.tokensPerRound() });
        try hist(w, &self.by_offered);
        try w.writeAll(" held=");
        try hist(w, &self.by_held);
        try w.writeAll(" depth=");
        var first = true;
        for (self.tried, self.landed, self.predicted, self.predicted_n, 1..) |t, l, p, pn, d| {
            if (t == 0) continue;
            if (!first) try w.writeAll(",");
            first = false;
            const tf: f64 = @floatFromInt(t);
            const pm: f64 = if (pn == 0) -1 else p / @as(f64, @floatFromInt(pn));
            try w.print("{d}:{d}/{d}({d:.3}|{d:.3})", .{ d, l, t, @as(f64, @floatFromInt(l)) / tf, pm });
        }
    }

    fn hist(w: *std.Io.Writer, h: []const u64) !void {
        var first = true;
        for (h, 0..) |c, k| {
            if (c == 0) continue;
            if (!first) try w.writeAll(",");
            first = false;
            try w.print("{d}:{d}", .{ k, c });
        }
        if (first) try w.writeAll("-");
    }
};

test "a round counts once its kept rows land, its drafts tried in order and landed as a prefix" {
    var s: Stats = .{};
    s.land(3); // nothing pending
    try std.testing.expectEqual(@as(u64, 0), s.rounds);
    s.offer(4, 5, &.{ 0.9, 0.7, 0.5, 0.3, 0.1 });
    s.land(2);
    s.offer(2, 5, &.{});
    s.land(9); // more than offered: capped
    try std.testing.expectEqual(@as(u64, 2), s.rounds);
    try std.testing.expectEqual(@as(u64, 1), s.by_offered[4]);
    try std.testing.expectEqual(@as(u64, 2), s.by_held[5]);
    try std.testing.expectEqualSlices(u64, &.{ 2, 2, 1, 1 }, s.tried[0..4]);
    try std.testing.expectEqualSlices(u64, &.{ 2, 2, 0, 0 }, s.landed[0..4]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), s.predicted[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 + 4.0 / 2.0), s.tokensPerRound(), 1e-12);
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try s.write(&w);
    try std.testing.expectEqualStrings("2 rounds (0 of host tokens), 3.000 tokens a drafted round, offered=2:1,4:1 held=5:2 depth=1:2/2(1.000|0.900),2:2/2(1.000|0.700),3:0/1(0.000|0.500),4:0/1(0.000|0.300)", w.buffered());
}

test "landed drafts never exceed the tried ones, and tried ones never increase with depth" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    var s: Stats = .{};
    for (0..2000) |_| {
        const held = r.intRangeAtMost(usize, 0, max_drafts);
        s.offer(r.intRangeAtMost(usize, 0, held), held, &.{});
        if (r.boolean()) s.land(r.intRangeAtMost(usize, 0, max_drafts));
    }
    var total: u64 = 0;
    for (s.by_offered) |c| total += c;
    try std.testing.expectEqual(s.rounds, total);
    for (0..max_drafts) |j| {
        try std.testing.expect(s.landed[j] <= s.tried[j]);
        if (j > 0) try std.testing.expect(s.tried[j] <= s.tried[j - 1] and s.landed[j] <= s.landed[j - 1]);
    }
}
