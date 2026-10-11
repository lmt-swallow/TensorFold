//! Prompt-lookup ("copy") drafts, as the Mia fork's `copy_drafts.py` proposes them: when the context's last `match`
//! tokens (the pending token included) occurred before, the tokens that followed the latest earlier occurrence with
//! enough tokens after it become the drafts. They only propose; verification keeps the serial sample, so they never
//! change a reply, only the tokens a round lands. Both ranks hold the same context and propose the same drafts.
//!
//! The served setting (the recipe's exports): match 8, at most 15 drafts (TF_GLM_COPY_MAX), an occurrence starting
//! inside the reply must match the last 16 tokens (TF_GLM_COPY_REPLY_MATCH), no miss limit, and a window of more than
//! 8 rows short of the 16-row graph width padded to it by repeating the last draft (TF_GLM_WIDE_GRAPHS=16).
const std = @import("std");

/// engine.GRAPH_ROWS: windows up to this many rows always replay CUDA graphs.
pub const graphed = 8;

pub const Settings = struct {
    match: usize = 8,
    most: usize = 15,
    reply_match: usize = 16, // 0: occurrences in the reply need `match` tokens like the prompt's
    pad: []const usize = &.{16},
    miss_most: usize = 0,
};

pub const CopyDrafts = struct {
    gpa: std.mem.Allocator,
    s: Settings,
    buf: std.ArrayList(u32) = .empty,
    prompt: usize, // leading context tokens that are the prompt
    proposed: usize = 0,
    missed: bool = false,

    /// `context`: the prompt and the pending token (the reply's first).
    pub fn init(gpa: std.mem.Allocator, s: Settings, context: []const u32) !CopyDrafts {
        var c: CopyDrafts = .{ .gpa = gpa, .s = s, .prompt = context.len -| 1 };
        if (c.s.reply_match <= c.s.match) c.s.reply_match = 0;
        try c.buf.appendSlice(gpa, context);
        return c;
    }

    pub fn deinit(c: *CopyDrafts) void {
        c.buf.deinit(c.gpa);
    }

    /// Committed tokens join the context (the last is the next round's pending token); after a proposal they
    /// settle whether its round kept all of its copied drafts.
    pub fn extend(c: *CopyDrafts, tokens: []const u32) !void {
        if (tokens.len == 0) return;
        if (c.proposed != 0) {
            c.missed = tokens.len <= c.proposed;
            c.proposed = 0;
        }
        try c.buf.appendSlice(c.gpa, tokens);
    }

    /// Whether an earlier occurrence starting at `h` is one `starts` keeps.
    fn occurs(c: *const CopyDrafts, h: usize) bool {
        const ctx = c.buf.items;
        const L = ctx.len;
        const n = c.s.match;
        if (!std.mem.eql(u32, ctx[h .. h + n], ctx[L - n .. L])) return false;
        const extra = if (c.s.reply_match != 0) c.s.reply_match - n else 0;
        if (extra == 0 or h < c.prompt) return true;
        // an occurrence inside the reply: the `extra` tokens before it must match too
        if (h < extra or L < n + extra) return false;
        return std.mem.eql(u32, ctx[h - extra .. h], ctx[L - n - extra .. L - n]);
    }

    /// Up to min(most, room) drafts: what followed the latest earlier occurrence with that many tokens after it, else
    /// the most after the earliest; none when the suffix never occurred; padded as `padded` says. `out` holds the
    /// drafts (at most most + the pad); returns how many.
    pub fn propose(c: *CopyDrafts, room: usize, out: []u32) usize {
        const most = if (c.missed and c.s.miss_most != 0) @min(c.s.most, c.s.miss_most) else c.s.most;
        const k = @min(most, room);
        c.proposed = 0;
        if (k < 1) return 0;
        const ctx = c.buf.items;
        const L = ctx.len;
        const n = c.s.match;
        if (L <= n) return 0;
        // starts 0 .. L - n - 1, ascending: the latest with k tokens after its match, else the earliest
        var first: ?usize = null;
        var full: ?usize = null;
        var h: usize = 0;
        while (h + n < L) : (h += 1) {
            if (!c.occurs(h)) continue;
            if (first == null) first = h;
            if (h + n + k <= L) full = h;
        }
        const s = full orelse first orelse return 0;
        const end = @min(s + n + k, L);
        const m = end - (s + n);
        @memcpy(out[0..m], ctx[s + n .. end]);
        c.proposed = m;
        return c.padded(out, m, room);
    }

    /// The drafts filled up to the next pad width's window by repeating the last one, when their window is wider
    /// than `graphed` rows, narrower than that width, and `room` leaves space for it.
    fn padded(c: *const CopyDrafts, out: []u32, m: usize, room: usize) usize {
        const rows = 1 + m;
        if (c.s.pad.len == 0 or m == 0 or rows <= graphed) return m;
        const width = for (c.s.pad) |w| {
            if (w >= rows) break w;
        } else 0;
        if (width <= rows or width - 1 > room) return m;
        for (out[m .. width - 1]) |*x| x.* = out[m - 1];
        return width - 1;
    }
};

/// decode.py `DepthPolicy` with `fixed` (greedy MTP rounds: 3 drafts, the chain cut where the product of the
/// drafts' confidences falls under 0.35).
pub const depth_most = 3;
pub const confidence = 0.35;

test "a repeated passage drafts what followed it, its latest full occurrence first" {
    const gpa = std.testing.allocator;
    // prompt: 1..10 then 1..8 again; pending 8: the suffix 1..8 occurred at 0, followed by 9 10 1 2 ...
    var ctx: [18]u32 = undefined;
    for (0..10) |i| ctx[i] = @intCast(i + 1);
    for (0..8) |i| ctx[10 + i] = @intCast(i + 1);
    var c = try CopyDrafts.init(gpa, .{ .most = 3, .pad = &.{} }, &ctx);
    defer c.deinit();
    var out: [32]u32 = undefined;
    const m = c.propose(15, &out);
    try std.testing.expectEqualSlices(u32, &.{ 9, 10, 1 }, out[0..m]);
    // the round kept only 1 token of the 3: missed (no miss limit set, so the next proposal is not narrowed)
    try c.extend(&.{9});
    try std.testing.expect(c.missed);
}

test "a suffix never seen proposes nothing; a window short of the graph width is padded" {
    const gpa = std.testing.allocator;
    var ctx: [40]u32 = undefined;
    for (&ctx, 0..) |*x, i| x.* = @intCast(i);
    var c = try CopyDrafts.init(gpa, .{}, &ctx);
    defer c.deinit();
    var out: [32]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 0), c.propose(15, &out));
    // 0..19 then 0..9: the 8-token suffix 2..9 occurred in the prompt; 10 drafts (11 rows) pad to 15 (16 rows)
    var ctx2: [30]u32 = undefined;
    for (0..20) |i| ctx2[i] = @intCast(i);
    for (0..10) |i| ctx2[20 + i] = @intCast(i);
    var d = try CopyDrafts.init(gpa, .{ .most = 10, .pad = &.{16} }, &ctx2);
    defer d.deinit();
    const m = d.propose(15, &out);
    try std.testing.expectEqual(@as(usize, 15), m);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 19, 19, 19, 19 }, out[0..m]);
}
