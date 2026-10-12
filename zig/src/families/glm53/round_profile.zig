//! TF_GLM_ROUND_PROFILE=1: where a decode round's time goes, on rank 0, for the first `rounds` rounds a stream runs
//! after its prompt: the verify forward's GPU time (events around it), the greedy draw, the ranks' agreement, the
//! commit, the draft head's absorb (GPU time) and each chain step (host time: a step ends in its draw), and the
//! round's period from one verify to the next (the rest is the lane core's and the records'). One log line a round.
//! Off, every call is a branch on a null; on, it adds one event record pair and a host clock read a phase.
const std = @import("std");
const cuda = @import("cuda");

pub const rounds = 20;
pub const max_steps = 4;

pub const Profile = struct {
    a: cuda.Event,
    b: cuda.Event,
    c: cuda.Event,
    d: cuda.Event,
    logged: usize = 0,
    started: i128 = 0, // the current round's verify start (0: none yet)
    rows: usize = 0,
    forward_ms: f64 = 0,
    draw_ms: f64 = 0,
    agree_ms: f64 = 0,
    agrees: usize = 0,
    commit_ms: f64 = 0,
    absorb_ms: f64 = 0,
    absorbed: usize = 0,
    steps: [max_steps]f64 = @splat(0),
    nsteps: usize = 0,
    forward_open: bool = false,
    absorb_open: bool = false,

    /// A profile when TF_GLM_ROUND_PROFILE=1, else null.
    pub fn fromEnv(d: *const cuda.Driver) !?Profile {
        const v = std.c.getenv("TF_GLM_ROUND_PROFILE") orelse return null;
        if (!std.mem.eql(u8, std.mem.span(v), "1")) return null;
        return .{ .a = try cuda.Event.init(d, true), .b = try cuda.Event.init(d, true), .c = try cuda.Event.init(d, true), .d = try cuda.Event.init(d, true) };
    }

    pub fn deinit(p: *Profile) void {
        p.a.deinit();
        p.b.deinit();
        p.c.deinit();
        p.d.deinit();
    }

    pub fn done(p: *const Profile) bool {
        return p.logged >= rounds;
    }

    /// A verify window of `rows` rows begins: the last round (if any) is logged.
    pub fn verifyBegin(p: *Profile, rows: usize, stream: cuda.Stream) !void {
        if (p.done()) return;
        const now = nowNs();
        if (p.started != 0) try p.log(now);
        p.* = .{ .a = p.a, .b = p.b, .c = p.c, .d = p.d, .logged = p.logged, .started = now, .rows = rows };
        try p.a.record(stream);
        p.forward_open = true;
    }

    /// The verify forward ended (before its draw).
    pub fn forwardEnd(p: *Profile, stream: cuda.Stream) !void {
        if (!p.forward_open) return;
        try p.b.record(stream);
    }

    /// The forward's GPU time, read once the draw synced the stream.
    pub fn drawEnd(p: *Profile, t0: i128) !void {
        if (!p.forward_open) return;
        p.forward_open = false;
        p.forward_ms = try cuda.Event.elapsedMs(p.a, p.b);
        p.draw_ms = msSince(t0);
    }

    pub fn agreed(p: *Profile, t0: i128) void {
        if (p.started == 0 or p.done()) return;
        p.agree_ms += msSince(t0);
        p.agrees += 1;
    }

    pub fn committed(p: *Profile, t0: i128) void {
        if (p.started == 0 or p.done()) return;
        p.commit_ms += msSince(t0);
    }

    pub fn absorbBegin(p: *Profile, stream: cuda.Stream) !void {
        if (p.started == 0 or p.done()) return;
        try p.c.record(stream);
        p.absorb_open = true;
    }

    pub fn absorbEnd(p: *Profile, rows: usize, stream: cuda.Stream) !void {
        if (!p.absorb_open) return;
        p.absorb_open = false;
        try p.d.record(stream);
        try p.d.synchronize();
        p.absorb_ms += try cuda.Event.elapsedMs(p.c, p.d);
        p.absorbed += rows;
    }

    pub fn step(p: *Profile, t0: i128) void {
        if (p.started == 0 or p.done() or p.nsteps == max_steps) return;
        p.steps[p.nsteps] = msSince(t0);
        p.nsteps += 1;
    }

    fn log(p: *Profile, now: i128) !void {
        const period = @as(f64, @floatFromInt(now - p.started)) / 1e6;
        var chain: f64 = 0;
        for (p.steps[0..p.nsteps]) |s| chain += s;
        const known = p.forward_ms + p.draw_ms + p.agree_ms + p.commit_ms + p.absorb_ms + chain;
        std.log.info("glm53 round {d}: period {d:.2} ms, {d} rows: verify forward {d:.2} (GPU), draw {d:.2}, agree {d:.2} ({d}x), commit {d:.2}, absorb {d:.2} (GPU, {d} rows), chain {d:.2} ({d} steps: {any}), rest {d:.2}", .{
            p.logged, period, p.rows, p.forward_ms, p.draw_ms, p.agree_ms, p.agrees, p.commit_ms, p.absorb_ms, p.absorbed, chain, p.nsteps, p.steps[0..p.nsteps], period - known,
        });
        p.logged += 1;
    }
};

pub fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

fn msSince(t0: i128) f64 {
    return @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
}
