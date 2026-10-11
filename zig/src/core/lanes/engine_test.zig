//! The round loop on the fake target: drafted == one-token rounds, shared rounds == solo, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
    think_budget: u32 = 0,
};

fn model() !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: []const Case) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const GuardCase = struct {
    prompt: []const u32,
    max_new: u32 = 400,
    drafts: bool = true,
    think_budget: u32 = 0,
    loop_guard: bool = true,
    think_open: ?bool = null,
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
};

const GuardOut = struct {
    emitted: []u32,
    reason: sm.Reason,
    period: ?u32,
    finished: bool,
};

fn runGuard(cases: []const GuardCase, step_limit: usize) ![]GuardOut {
    var cfg = try model();
    defer cfg.deinit(gpa);
    const c = cases[0];
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = c.cycle_after, .probe_prompt = c.prompt.len, .pattern = c.pattern, .answer_cycles = c.answer_cycles };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |item, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "guard", .prompt = item.prompt, .max_new = item.max_new, .eos = &.{96}, .drafts = item.drafts, .proposer = p.proposer(), .think_budget = item.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = item.think_open, .loop_guard = item.loop_guard });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < step_limit) : (steps += 1) try engine.step();
    const out = try gpa.alloc(GuardOut, cases.len);
    for (out, streams) |*result, *s| result.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .finished = s.finished };
    return out;
}

fn freeGuard(runs: []GuardOut) void {
    for (runs) |run_result| gpa.free(run_result.emitted);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "drafted rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        // and the fake target's own decode
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &p1);
        for (drafted[0]) |t| {
            try std.testing.expectEqual(fake.next(history.items, s, history.items.len), t);
            try history.append(gpa, t);
        }
    }
}

test "shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } });
    defer free(together);
    const one = try run(&.{.{ .prompt = &p1 }});
    defer free(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30 }});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], together[0]);
    try std.testing.expectEqualSlices(u32, two[0], together[1]);
}

test "the thinking budget's forced close is the same drafted, plain and shared" {
    const drafted = try run(&.{.{ .prompt = &p2, .think_budget = 9 }});
    defer free(drafted);
    const plain = try run(&.{.{ .prompt = &p2, .think_budget = 9, .drafts = false }});
    defer free(plain);
    const shared = try run(&.{ .{ .prompt = &p2, .think_budget = 9 }, .{ .prompt = &p1, .drafts = false } });
    defer free(shared);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, plain[0], shared[0]);
    try std.testing.expectEqual(@as(u32, 90), drafted[0][8]);
}

test "the loop guard matches drafted and plain, reports its period, and answers after close" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2 }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), drafted[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[329..]);

    const off = try runGuard(&.{.{ .prompt = &p2, .loop_guard = false, .think_open = true }}, 5000);
    defer freeGuard(off);
    try std.testing.expect(off[0].finished and off[0].reason == .length and off[0].emitted.len == 400);
}

test "a cycle in the answer does not refire or reclose" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), plain[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .length);
    var close_count: usize = 0;
    for (plain[0].emitted) |token| close_count += @intFromBool(token == 90);
    try std.testing.expectEqual(@as(usize, 1), close_count);
}

const Interleaved = struct { emitted: [][]u32, mid_pass: u64, released: bool };

/// `run`'s cases on a fake whose prompt passes take `chunks` parts: the first stream added whole, each later one
/// filled a chunk at a time with `between` rounds of the live streams after each part; `cancel` (an index) is
/// cancelled after its first part.
fn runInterleaved(cases: []const Case, chunks: usize, between: usize, cancel: ?usize) !Interleaved {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .parts = true, .prefill_chunks = chunks };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    var flags = try gpa.alloc(bool, cases.len);
    defer gpa.free(flags);
    @memset(flags, false);
    const Flag = struct {
        fn check(ptr: *anyopaque) bool {
            return @as(*bool, @ptrCast(ptr)).*;
        }
    };
    for (cases, streams, proposers, flags) |c, *s, *p, *flag| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .cancel_check = .{ .ptr = flag, .check = Flag.check } });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    try engine.addStream(&streams[0]);
    for (streams[1..], 1..) |*s, i| {
        var parts: usize = 0;
        while (true) {
            if (cancel == i and parts == 1) flags[i] = true;
            const done = engine.fillStream(s, 1) catch |err| {
                if (err != error.Cancelled) return err;
                break;
            };
            parts += 1;
            if (done) break;
            for (0..between) |_| if (engine.activeCount() > 0) try engine.step();
        }
    }
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return .{ .emitted = out, .mid_pass = target.rounds_mid_pass, .released = target.lanes.count() == 0 };
}

test "a prompt pass in parts between other streams' rounds commits what a whole pass commits, for every stream" {
    const sampled: Sampling = .{ .seed = 4, .temperature = 0.8, .top_k = 0, .top_p = 0.9 };
    const cases = [_]Case{ .{ .prompt = &p1, .max_new = 60 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 }, .{ .prompt = &p1, .drafts = false, .max_new = 20 } };
    const whole = try run(&cases);
    defer free(whole);
    for ([_]usize{ 1, 4 }) |chunks| for ([_]usize{ 0, 1, 3 }) |between| {
        const parts = try runInterleaved(&cases, chunks, between, null);
        defer free(parts.emitted);
        for (whole, parts.emitted) |w, p| try std.testing.expectEqualSlices(u32, w, p);
        // rounds ran between parts exactly when the policy asked for them and a pass had more than one part
        try std.testing.expectEqual(chunks > 1 and between > 0, parts.mid_pass > 0);
        try std.testing.expect(parts.released);
    };
}

test "a stream cancelled between the parts of its pass is released and leaves the others' tokens unchanged" {
    const cases = [_]Case{ .{ .prompt = &p1, .max_new = 60 }, .{ .prompt = &p2, .max_new = 30 }, .{ .prompt = &p2, .max_new = 25 } };
    const whole = try run(&.{ cases[0], cases[2] });
    defer free(whole);
    const parts = try runInterleaved(&cases, 4, 2, 1);
    defer free(parts.emitted);
    try std.testing.expectEqual(@as(usize, 0), parts.emitted[1].len);
    try std.testing.expectEqualSlices(u32, whole[0], parts.emitted[0]);
    try std.testing.expectEqualSlices(u32, whole[1], parts.emitted[2]);
    try std.testing.expect(parts.released);
}

test "without prefill_part a stream's pass runs whole at its first part" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &p1, .max_new = 8, .proposer = p.proposer() });
    defer s.deinit(gpa);
    try std.testing.expect(try engine.fillStream(&s, 1));
    try std.testing.expectEqual(@as(usize, 3), target.prefill_count);
    try std.testing.expectEqual(@as(usize, 1), engine.activeCount());
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqual(@as(usize, 8), s.emitted().len);
}

/// One stream on a target whose reply cycles `pattern` after `cycle_after` tokens: its emitted tokens and its copy
/// windows' drafts (from the round events).
fn runCopies(copy_pricing: bool, drafts: bool, pattern: []const u32, copies: *std.ArrayList(usize)) ![]u32 {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 20.0 + 4.0 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .first_copy_rows = 5, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .copy_pricing = copy_pricing }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 12, .probe_prompt = p1.len, .pattern = pattern };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var log = @import("events.zig").Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &p1, .max_new = 240, .eos = &.{}, .drafts = drafts, .proposer = p.proposer() });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    for (log.events.items) |fields| {
        var round = false;
        var copy = false;
        var rows: usize = 0;
        for (fields) |x| {
            if (std.mem.eql(u8, x.k, "ev")) round = std.mem.eql(u8, x.v.str, "round");
            if (std.mem.eql(u8, x.k, "kind")) copy = x.v == .str and std.mem.eql(u8, x.v.str, "copy");
            if (std.mem.eql(u8, x.k, "window") and x.v == .u32s) rows = x.v.u32s.len;
        }
        if (round and copy) try copies.append(gpa, rows - 1);
    }
    return gpa.dupe(u32, s.emitted());
}

test "priced copies keep the one-token decode and widen past the ramp's first width on a repeating reply" {
    const pattern = [_]u32{ 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
    var none: std.ArrayList(usize) = .empty;
    defer none.deinit(gpa);
    const plain = try runCopies(false, false, &pattern, &none);
    defer gpa.free(plain);
    try std.testing.expectEqual(@as(usize, 0), none.items.len);
    var python: std.ArrayList(usize) = .empty;
    defer python.deinit(gpa);
    const ramp = try runCopies(false, true, &pattern, &python);
    defer gpa.free(ramp);
    var priced: std.ArrayList(usize) = .empty;
    defer priced.deinit(gpa);
    const got = try runCopies(true, true, &pattern, &priced);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u32, plain, ramp);
    try std.testing.expectEqualSlices(u32, plain, got);
    // the ramp's copies start at its first width; priced ones take the widest window the costs pay for at once
    // the ramp's copies start at its first width; priced ones as wide as the match's continuation at once
    try std.testing.expect(python.items.len > 0 and python.items[0] <= 4);
    try std.testing.expect(priced.items.len > 0 and priced.items[0] > python.items[0]);
    try std.testing.expectEqual(@as(usize, 15), std.mem.max(usize, priced.items));
}

test "a priced copy gives way to the head's drafts once its copies stop landing" {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 20.0 + 4.0 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .first_copy_rows = 5, .mtp = true, .speculate = true, .drafts = 4, .window_costs = &costs, .hidden_rows = true, .copy_pricing = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4, .silence_rounds = 0 });
    defer p.deinit();
    const text = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 } ++ [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &text, .max_new = 64, .proposer = p.proposer() });
    defer s.deinit(gpa);
    const win = @import("windows.zig");
    // the prior (copy_rate 0.94 a token) pays for a window wider than the head's (4 drafts at its prior); at 20 ms
    // a window and 4 ms a row, 10 copied tokens land the most a ms
    const continuation = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5 };
    try std.testing.expectEqualSlices(u32, continuation[0..10], try win.copyProposal(&engine, &s));
    // copies that land half their tokens: a shorter window still pays
    for (0..8) |_| try win.observeCopy(&engine, &s, 15, 3);
    const half = try win.copyProposal(&engine, &s);
    try std.testing.expect(half.len >= 2 and half.len < 15);
    // copies whose first token never lands: the head drafts instead
    for (0..40) |_| try win.observeCopy(&engine, &s, 15, 0);
    try std.testing.expectEqual(@as(usize, 0), (try win.copyProposal(&engine, &s)).len);
}

test "a priced copy's window shrinks where its stream's rows cost more" {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 20.0 + 4.0 * @as(f64, @floatFromInt(w)) };
    var deep: [16]@import("config.zig").Cost = undefined;
    for (&deep, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 20.0 + 12.0 * @as(f64, @floatFromInt(w)) };
    const at = [_]@import("allocate.zig").PositionCosts.Timed{ .{ .position = 1024, .costs = &costs }, .{ .position = 262144, .costs = &deep } };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .first_copy_rows = 5, .mtp = true, .speculate = true, .drafts = 4, .window_costs = &costs, .hidden_rows = true, .copy_pricing = true, .copy_position_costs = &at }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    const text = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 } ++ [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &text, .max_new = 64, .proposer = p.proposer() });
    defer s.deinit(gpa);
    const win = @import("windows.zig");
    s.cache_len = 1024; // at the shallow table: the window costs alone (10 tokens, as without position costs)
    const near = (try win.copyProposal(&engine, &s)).len;
    try std.testing.expectEqual(@as(usize, 10), near);
    s.cache_len = 262144; // rows three times dearer: a shorter window pays
    const far = (try win.copyProposal(&engine, &s)).len;
    try std.testing.expect(far < near);
}

test "shared copies take only the rows the other streams would not get" {
    var costs: [32]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 20.0 + 4.0 * @as(f64, @floatFromInt(w)) };
    const win = @import("windows.zig");
    var counts: [2][3]usize = undefined; // [share off, on][the copy's rows, the head stream's rows, head-only]
    for ([_]bool{ false, true }, 0..) |share, k| {
        var cfg = try Config.init(gpa, .{ .exact_width = 16, .first_copy_rows = 6, .mtp = true, .speculate = true, .drafts = 5, .window_costs = costs[0..16], .shared_costs = &costs, .hidden_rows = true, .batch_rows = 16, .max_streams = 4, .draft_streams = true, .copy_pricing = true, .copy_share = share }, 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        engine.alone = false;
        var a = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &p1, .max_new = 64 });
        defer a.deinit(gpa);
        var b = try sm.Stream.init(gpa, .{ .id = "b", .prompt = &p2, .max_new = 64 });
        defer b.deinit(gpa);
        a.copy_p = try gpa.alloc(f64, 15);
        @memset(a.copy_p.?, 0.99);
        const copied = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        var plans = [_]win.Plan{ .{ .stream = &a, .position = 64, .kind = .copy, .tokens = &copied }, .{ .stream = &b, .position = 64, .kind = .head, .held = 5 } };
        try win.allocate(&engine, &plans);
        counts[k][0] = plans[0].count();
        counts[k][1] = plans[1].count();
        // the head stream's rows were the other one drafting with its head too
        var heads = [_]win.Plan{ .{ .stream = &a, .position = 64, .kind = .head, .held = 5 }, .{ .stream = &b, .position = 64, .kind = .head, .held = 5 } };
        try win.allocate(&engine, &heads);
        counts[k][2] = heads[1].count();
        _ = engine.arena.reset(.retain_capacity);
    }
    // off: the sure copy crowds the head stream out of rows it would have had; on: it keeps them
    try std.testing.expect(counts[0][1] < counts[0][2]);
    try std.testing.expectEqual(counts[1][2], counts[1][1]);
    try std.testing.expect(counts[1][0] >= 2 and counts[1][0] + counts[1][1] <= 15);
}
