//! TF_QFN_ROUND_STATS=1: where the serving time goes, logged every ten seconds the backend is busy. Rounds by
//! stream count; verify forwards by total rows (count and wall ms, the device's work and the draws' read-back
//! included); the head's drafting (wall ms); commits; prompt chunks; and the time
//! between backend calls (the round loop, the HTTP side, an idle server). A window also ends when the last stream
//! leaves. It only observes: no choice reads it.
const std = @import("std");

pub const max_streams = 4;
pub const max_rows = 64;

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

fn ms(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// Confidence bins' lower edges (a draft's head probability).
pub const bins = [_]f64{ 0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 };
pub const depth = 6;

/// The head's drafts as their rounds verified them: offered and accepted by depth and by confidence bin, chains by
/// length and by how they ended, and the probabilities of the drafts that ended chains (never verified).
pub const Drafts = struct {
    offered: [depth]u64 = @splat(0),
    accepted: [depth]u64 = @splat(0),
    bin_offered: [bins.len]u64 = @splat(0),
    bin_accepted: [bins.len]u64 = @splat(0),
    chains: [depth + 1]u64 = @splat(0), // by draft count
    ended_low: u64 = 0, // a later draft below the confidence ended it
    first_low: u64 = 0, // its first draft was below the confidence (kept, the chain ended)
    cut_bins: [bins.len]u64 = @splat(0), // the ending drafts' probabilities

    fn bin(p: f64) usize {
        var b: usize = 0;
        while (b + 1 < bins.len and p >= bins[b + 1]) b += 1;
        return b;
    }

    /// A window's drafts `probs` of which the first `accepted` were kept.
    pub fn verified(d: *Drafts, probs: []const f64, accepted: usize) void {
        for (probs, 0..) |p, i| {
            const ok = i < accepted;
            if (i < depth) {
                d.offered[i] += 1;
                d.accepted[i] += @intFromBool(ok);
            }
            d.bin_offered[bin(p)] += 1;
            d.bin_accepted[bin(p)] += @intFromBool(ok);
        }
    }

    /// A chain the head drafted: its probabilities and, when a later draft ended it, that draft's.
    pub fn chain(d: *Drafts, probs: []const f64, cut: ?f64, confidence: f64) void {
        d.chains[@min(probs.len, depth)] += 1;
        if (cut) |c| {
            d.ended_low += 1;
            d.cut_bins[bin(c)] += 1;
        } else if (probs.len > 0 and probs[0] < confidence) d.first_low += 1;
    }

    fn print(d: *const Drafts, w: *std.Io.Writer, name: []const u8) void {
        w.print(" {s}: by depth", .{name}) catch {};
        for (d.offered, d.accepted, 0..) |o, a, i| if (o > 0) w.print(" {d}:{d}/{d}", .{ i + 1, a, o }) catch {};
        w.print("; by confidence", .{}) catch {};
        for (d.bin_offered, d.bin_accepted, bins) |o, a, lo| if (o > 0) w.print(" >={d:.2}:{d}/{d}", .{ lo, a, o }) catch {};
        w.print("; chains by length", .{}) catch {};
        for (d.chains, 0..) |n, k| if (n > 0) w.print(" {d}:{d}", .{ k, n }) catch {};
        w.print("; ended by a low draft {d} (their confidence", .{d.ended_low}) catch {};
        for (d.cut_bins, bins) |n, lo| if (n > 0) w.print(" >={d:.2}:{d}", .{ lo, n }) catch {};
        w.print("), first draft low {d};", .{d.first_low}) catch {};
    }
};

pub const Stats = struct {
    since: i128,
    last_end: i128 = 0,
    rounds: [max_streams + 1]u64 = @splat(0),
    rows_n: [max_rows + 1]u64 = @splat(0),
    rows_ms: [max_rows + 1]f64 = @splat(0),
    verify_ms: f64 = 0,
    draft_ms: f64 = 0,
    draft_calls: u64 = 0,
    keep_ms: f64 = 0,
    kept: u64 = 0,
    prompt_ms: f64 = 0,
    prompt_chunks: u64 = 0,
    between_ms: f64 = 0,
    idle: bool = false, // the last stream left; the next call starts a new window
    // drafts by the round that verified them (0: a lone stream's, 1: a shared round's)
    drafts: [2]Drafts = .{ .{}, .{} },

    /// Stats when TF_QFN_ROUND_STATS is "1", else null.
    pub fn fromEnv() ?Stats {
        const v = std.c.getenv("TF_QFN_ROUND_STATS") orelse return null;
        if (!std.mem.eql(u8, std.mem.span(v), "1")) return null;
        return .{ .since = nowNs() };
    }

    /// A backend call starts: the time since the last one ended counts as between calls.
    pub fn begin(s: *Stats) i128 {
        const t = nowNs();
        if (s.idle) { // the first call after the server went idle opens the window: the idle time is not counted
            s.* = .{ .since = t };
            return t;
        }
        if (s.last_end != 0) s.between_ms += ms(t - s.last_end);
        return t;
    }

    fn end(s: *Stats, t0: i128) f64 {
        const t = nowNs();
        s.last_end = t;
        return ms(t - t0);
    }

    pub fn verify(s: *Stats, t0: i128, streams: usize, rows: usize) void {
        const d = s.end(t0);
        s.verify_ms += d;
        s.rounds[@min(streams, max_streams)] += 1;
        const r = @min(rows, max_rows);
        s.rows_n[r] += 1;
        s.rows_ms[r] += d;
        s.maybeLog();
    }

    pub fn draft(s: *Stats, t0: i128) void {
        s.draft_ms += s.end(t0);
        s.draft_calls += 1;
    }

    pub fn keep(s: *Stats, t0: i128, tokens: usize) void {
        s.keep_ms += s.end(t0);
        s.kept += tokens;
    }

    pub fn prompt(s: *Stats, t0: i128, chunks: usize) void {
        s.prompt_ms += s.end(t0);
        s.prompt_chunks += chunks;
        s.maybeLog();
    }

    /// The window so far, logged now (the last stream left: the window's rounds are over).
    pub fn flush(s: *Stats) void {
        s.log(true);
        s.idle = true;
    }

    fn maybeLog(s: *Stats) void {
        s.log(false);
    }

    fn log(s: *Stats, now: bool) void {
        const t = nowNs();
        const wall = ms(t - s.since);
        if (wall < 10_000 and !now) return;
        if (s.verify_ms == 0 and s.prompt_ms == 0) return;
        var buf: [4096]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        w.print("qwen38fn round stats over {d:.1} s: rounds by streams", .{wall / 1000}) catch {};
        for (s.rounds[1..], 1..) |n, k| if (n > 0) w.print(" {d}:{d}", .{ k, n }) catch {};
        w.print("; verify by rows", .{}) catch {};
        for (s.rows_n, s.rows_ms, 0..) |n, t_ms, r| if (n > 0) w.print(" {d}:{d}x{d:.1}ms", .{ r, n, t_ms / @as(f64, @floatFromInt(n)) }) catch {};
        w.print("; verify {d:.0} ms ({d:.0}%), draft {d:.0} ms in {d} calls ({d:.0}%), keep {d:.0} ms, prompt {d:.0} ms in {d} chunks ({d:.0}%), between calls {d:.0} ms ({d:.0}%); {d} tokens kept", .{
            s.verify_ms, 100 * s.verify_ms / wall, s.draft_ms, s.draft_calls, 100 * s.draft_ms / wall, s.keep_ms, s.prompt_ms, s.prompt_chunks, 100 * s.prompt_ms / wall, s.between_ms, 100 * s.between_ms / wall, s.kept,
        }) catch {};
        std.log.info("{s}", .{w.buffered()});
        var w2: std.Io.Writer = .fixed(&buf);
        w2.print("qwen38fn draft stats (accepted/offered):", .{}) catch {};
        s.drafts[0].print(&w2, "lone rounds");
        s.drafts[1].print(&w2, "shared rounds");
        std.log.info("{s}", .{w2.buffered()});
        s.* = .{ .since = t, .last_end = s.last_end };
    }
};
