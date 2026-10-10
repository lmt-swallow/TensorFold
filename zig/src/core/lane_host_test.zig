//! Host memory reporting and isolated request refusal through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Memory = api.Memory;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "a lane host reports its backend's memory counts, and none without them" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().memory(false) == null);
    const Counts = struct {
        resets: u32 = 0,
        fn read(ctx: ?*anyopaque, reset_peak: bool) ?Memory {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            if (reset_peak) c.resets += 1;
            return .{ .active = 5, .peak = if (reset_peak) 5 else 9 };
        }
    };
    var counts: Counts = .{};
    host.memory = .{ .ctx = &counts, .read = Counts.read };
    try std.testing.expectEqual(@as(u64, 9), host.engine().memory(false).?.peak);
    try std.testing.expectEqual(@as(u64, 5), host.engine().memory(true).?.peak);
    try std.testing.expectEqual(@as(u32, 1), counts.resets);
}

test "a request the backend refuses fails alone, in the backend's words" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_sampled = true };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Words = struct {
        fn text(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return if (err == error.SamplingRefused) "send temperature 0" else null;
        }
    };
    host.explain = .{ .text = Words.text };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        message: []const u8 = "",
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                },
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var plain: Box = .{};
    var sampled: Box = .{};
    const greedy: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    const keyed: Request = .{ .prompt = &prompt, .max_tokens = 64, .sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 5 } };
    const e = host.engine();
    try e.submit(1, &greedy, .{ .ctx = &plain, .event = Box.event });
    try e.submit(2, &keyed, .{ .ctx = &sampled, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, sampled.wait());
    try std.testing.expectEqualStrings("send temperature 0", sampled.message);
    try std.testing.expectEqual(Reason.length, plain.wait());
    try std.testing.expectEqual(@as(usize, 64), plain.tokens);
}

test "interleaving, a stream decodes between the chunks of a later prompt's pass, and every stream keeps its tokens" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .parts = true, .prefill_chunks = 6 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    host.interleave = 2;
    // the core's step count at each chunk of each pass
    const Steps = struct {
        core: *lanes.Engine,
        at: [3][6]u64 = undefined,
        pass: usize = 0,
        fn chunk(ctx: *anyopaque, _: *lanes.Stream, i: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            x.at[x.pass][i] = x.core.steps;
            if (i == 5) x.pass += 1;
        }
    };
    var steps: Steps = .{ .core = &core };
    target.prefill_hook = Steps.chunk;
    target.prefill_hook_ctx = &steps;
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompts = [_][]const u32{ &.{ 3, 1, 4, 1, 5 }, &.{ 2, 7, 1, 8, 2, 8 }, &.{ 1, 6, 1, 8 } };
    const budgets = [_]u32{ 120, 30, 30 };
    var requests: [3]Request = undefined;
    var boxes: [3]Box = .{ .{}, .{}, .{} };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    const e = host.engine();
    // queued before the host thread starts, so the later passes find the first stream decoding
    for (&requests, prompts, budgets, &boxes, 0..) |*r, p, n, *b, i| {
        r.* = .{ .prompt = p, .max_tokens = n };
        try e.submit(@intCast(i + 1), r, .{ .ctx = b, .event = Box.event });
    }
    try host.start();
    defer host.stop();
    for (&boxes, prompts, budgets) |*b, p, n| {
        try std.testing.expectEqual(Reason.length, b.wait());
        try std.testing.expectEqual(@as(usize, n), b.tokens.items.len);
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, p);
        for (b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
    }
    // the first pass ran whole with no stream decoding; each later pass's gaps held 2 rounds of the decoding streams
    try std.testing.expectEqual(@as(usize, 3), steps.pass);
    for (steps.at[0][1..], steps.at[0][0..5]) |b, a| try std.testing.expectEqual(a, b);
    for (steps.at[1..]) |at| for (at[1..], at[0..5]) |b, a| try std.testing.expectEqual(a + 2, b);
    try std.testing.expect(target.rounds_mid_pass > 0);
}

test "overlapping, a part computes while the decoding streams take rounds, the next starts once it is done, and every stream keeps its tokens" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    // each part goes on computing for 3 rounds after its call
    var target: lanes.fake.Fake = .{ .gpa = gpa, .parts = true, .prefill_chunks = 6, .part_rounds = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    host.interleave = 1;
    host.overlap = true;
    const Steps = struct {
        core: *lanes.Engine,
        at: [3][6]u64 = undefined,
        pass: usize = 0,
        fn chunk(ctx: *anyopaque, _: *lanes.Stream, i: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            x.at[x.pass][i] = x.core.steps;
            if (i == 5) x.pass += 1;
        }
    };
    var steps: Steps = .{ .core = &core };
    target.prefill_hook = Steps.chunk;
    target.prefill_hook_ctx = &steps;
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompts = [_][]const u32{ &.{ 3, 1, 4, 1, 5 }, &.{ 2, 7, 1, 8, 2, 8 }, &.{ 1, 6, 1, 8 } };
    const budgets = [_]u32{ 600, 300, 30 };
    var requests: [3]Request = undefined;
    var boxes: [3]Box = .{ .{}, .{}, .{} };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    const e = host.engine();
    for (&requests, prompts, budgets, &boxes, 0..) |*r, p, n, *b, i| {
        r.* = .{ .prompt = p, .max_tokens = n };
        try e.submit(@intCast(i + 1), r, .{ .ctx = b, .event = Box.event });
    }
    try host.start();
    defer host.stop();
    for (&boxes, prompts, budgets) |*b, p, n| {
        try std.testing.expectEqual(Reason.length, b.wait());
        try std.testing.expectEqual(@as(usize, n), b.tokens.items.len);
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, p);
        for (b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
    }
    try std.testing.expectEqual(@as(usize, 3), steps.pass);
    // with no stream decoding, the first pass's parts came one after another (each waiting for the last); with
    // streams decoding, each later part came once its predecessor's 3 rounds had run, never earlier
    for (steps.at[0][1..], steps.at[0][0..5]) |b, a| try std.testing.expectEqual(a, b);
    for (steps.at[1..]) |at| for (at[1..], at[0..5]) |b, a| try std.testing.expectEqual(a + 3, b);
    try std.testing.expect(target.waited >= 5);
    try std.testing.expect(target.ready_parts >= 10);
    try std.testing.expect(target.rounds_mid_pass > 0);
}
