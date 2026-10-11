//! TF_QFN_ROUND_STATS=1: where the serving time goes, logged every ten seconds the backend is busy. Rounds by
//! stream count; verify forwards by total rows (count and wall ms, the device's work and the draws' read-back
//! included); the head's drafting (wall ms, of it the host's draft sampling); commits; prompt chunks; and the time
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

pub const Stats = struct {
    since: i128,
    last_end: i128 = 0,
    rounds: [max_streams + 1]u64 = @splat(0),
    rows_n: [max_rows + 1]u64 = @splat(0),
    rows_ms: [max_rows + 1]f64 = @splat(0),
    verify_ms: f64 = 0,
    draft_ms: f64 = 0,
    draft_calls: u64 = 0,
    sample_ms: f64 = 0,
    keep_ms: f64 = 0,
    kept: u64 = 0,
    prompt_ms: f64 = 0,
    prompt_chunks: u64 = 0,
    between_ms: f64 = 0,
    idle: bool = false, // the last stream left; the next call starts a new window

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

    pub fn draft(s: *Stats, t0: i128, sample_ns: u64) void {
        s.draft_ms += s.end(t0);
        s.draft_calls += 1;
        s.sample_ms += @as(f64, @floatFromInt(sample_ns)) / 1e6;
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
        var buf: [2048]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        w.print("qwen38fn round stats over {d:.1} s: rounds by streams", .{wall / 1000}) catch {};
        for (s.rounds[1..], 1..) |n, k| if (n > 0) w.print(" {d}:{d}", .{ k, n }) catch {};
        w.print("; verify by rows", .{}) catch {};
        for (s.rows_n, s.rows_ms, 0..) |n, t_ms, r| if (n > 0) w.print(" {d}:{d}x{d:.1}ms", .{ r, n, t_ms / @as(f64, @floatFromInt(n)) }) catch {};
        w.print("; verify {d:.0} ms ({d:.0}%), draft {d:.0} ms in {d} calls ({d:.0}%, host sampling {d:.0} ms), keep {d:.0} ms, prompt {d:.0} ms in {d} chunks ({d:.0}%), between calls {d:.0} ms ({d:.0}%); {d} tokens kept", .{
            s.verify_ms, 100 * s.verify_ms / wall, s.draft_ms, s.draft_calls, 100 * s.draft_ms / wall, s.sample_ms, s.keep_ms, s.prompt_ms, s.prompt_chunks, 100 * s.prompt_ms / wall, s.between_ms, 100 * s.between_ms / wall, s.kept,
        }) catch {};
        std.log.info("{s}", .{w.buffered()});
        s.* = .{ .since = t, .last_end = s.last_end };
    }
};
