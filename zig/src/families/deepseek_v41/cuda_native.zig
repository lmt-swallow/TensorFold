//! DeepSeek-V4.1-Flash (EXL3, two DGX Sparks over NCCL) behind the native family interface: rank 0 serves the lane
//! core, rank 1 follows its steps. The ranks' settings come from the environment (`--tp/--rank/--master/
//! --master-port` export them: TF_TP_WORLD, TF_TP_RANK, TF_TP_MASTER, TF_TP_PORT), the weights from the Python
//! family's rank cache (TF_DS_RANK_CACHE), the RoPE frequencies (TF_DS_ROPE=device, the default: each pass's rows
//! computed on the device) or whole tables (TF_DS_ROPE=table) from TF_DS_ROPE_DIR (default `<kernels>/rope`), the prompt
//! chunk's rows from TF_DS_PREFILL_CHUNK (default 2048, at most 2048). Up to four streams decode together in shared
//! rounds (16 rows a round at most), their caches in one pool of TF_DS_POOL positions (cuda_lanes.zig `Streams`;
//! default the context, else at least the context and a multiple of 2048, the same on both ranks), each stream within
//! the context (`--context`: at most the model's window); TF_DS_SLOTS (default 4, 1 to 4, the same on both ranks)
//! sets the pool's slots. With drafts (the default;
//! `--no-drafts` turns them off on both ranks) the rank cache's DSpark blocks load and every stream verifies drafted
//! windows of up to 6 rows (TF_DS_DRAFTS: up to that many drafts and the pending token, 1 to the checkpoint's block
//! of 5, the same on both ranks), the rounds' and drafts' costs timed on both ranks at load. The load logs each phase's
//! time (`info:` lines); each rank loads on its own and the ranks meet after their loads, so both may start at once.
//! TF_DS_PROFILE=1 logs every prompt chunk's GPU time by sublayer group and its host time (a host sync a chunk).
//! TF_DS_DRAFT_STATS=1 has rank 0 log each stream's draft tallies as it leaves (draft_stats.zig).
//! Kept prompts (cuda_keep.zig) are on unless TF_DS_KEEP=0: a prompt that starts with a finished one prefills from
//! their longest shared kept chunk boundary (TF_DS_KEEP_ENTRIES, TF_DS_KEEP_MARKS, TF_DS_KEEP_SHRINK,
//! TF_DS_KEEP_SHRINK_MIN as in the fork; TF_DS_KEEP_SNAPS ring snapshots, default 32; TF_DS_KEEP_CHECK=1 compares
//! every continuation with a prefill from 0). Both ranks must set the same values.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const lanesmod = @import("cuda_lanes.zig");
const Engine = lanesmod.Engine;
const Lanes = lanesmod.Lanes;
const RankCache = @import("rank_cache.zig").RankCache;
const mem_note = @import("mem_note.zig");

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 16384;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = @intCast(lanesmod.default_chunk);

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    /// The streams' pool: its positions (each stream reaches at most the context's) and its device bytes, allocated
    /// by the load (the startup line names them).
    pool_positions: usize = 0,
    pool_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
};

const Owned = struct { gpa: std.mem.Allocator, e: *Engine, lanes: Lanes };

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

fn envInt(name: [:0]const u8, default: u32) !u32 {
    const v = getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch error.BadEnvironment;
}

/// The rank settings the environment names; `rank` and `world` default to one rank, which this family refuses.
pub fn settings(gpa: std.mem.Allocator, kernels_dir: []const u8, context: usize) !lanesmod.Settings {
    const world = try envInt("TF_TP_WORLD", 1);
    const rank = try envInt("TF_TP_RANK", 0);
    const port: u16 = @intCast(try envInt("TF_TP_PORT", 29551));
    var master: [4]u8 = .{ 127, 0, 0, 1 };
    if (getenv("TF_TP_MASTER")) |m| {
        var it = std.mem.splitScalar(u8, m, '.');
        for (&master) |*b| b.* = std.fmt.parseInt(u8, it.next() orelse return error.BadMaster, 10) catch return error.BadMaster;
    }
    const bucket_default: usize = @max(1024, std.math.ceilPowerOfTwo(usize, context) catch context);
    const bucket: usize = if (getenv("TF_DS_BUCKET")) |b| (std.fmt.parseInt(usize, b, 10) catch return error.BadEnvironment) else bucket_default;
    const rope_dir = if (getenv("TF_DS_ROPE_DIR")) |r| try gpa.dupe(u8, r) else try std.fs.path.join(gpa, &.{ kernels_dir, "rope" });
    const cache_dir = try gpa.dupe(u8, getenv("TF_DS_RANK_CACHE") orelse return error.NoRankCache);
    const chunk = try envInt("TF_DS_PREFILL_CHUNK", @intCast(lanesmod.default_chunk));
    if (chunk < 1 or chunk > lanesmod.default_chunk) {
        std.log.err("TF_DS_PREFILL_CHUNK is {d}: it takes 1 to {d} rows", .{ chunk, lanesmod.default_chunk });
        return error.BadEnvironment;
    }
    const slots = try envInt("TF_DS_SLOTS", @intCast(lanesmod.max_slots));
    if (slots < 1 or slots > lanesmod.max_slots) {
        std.log.err("TF_DS_SLOTS is {d}: it takes 1 to {d} streams", .{ slots, lanesmod.max_slots });
        return error.BadEnvironment;
    }
    const pool = poolSetting(getenv("TF_DS_POOL"), context) catch {
        std.log.err("TF_DS_POOL is {s}: it takes a number of positions, at least the context's {d}, a multiple of {d}", .{ getenv("TF_DS_POOL").?, context, lanesmod.extent_align });
        return error.BadEnvironment;
    };
    const most_drafts = try envInt("TF_DS_DRAFTS", 0);
    if (most_drafts > lanesmod.most_drafts_max) {
        std.log.err("TF_DS_DRAFTS is {d}: it takes 1 to {d} drafts a window, at most the checkpoint's DSpark block (unset: the block)", .{ most_drafts, lanesmod.most_drafts_max });
        return error.BadEnvironment;
    }
    return .{ .rank = rank, .world = world, .master = master, .port = port, .context = context, .bucket = bucket, .rope_dir = rope_dir, .cache_dir = cache_dir, .kernels_dir = kernels_dir, .chunk = chunk, .slots = slots, .pool = pool, .replay = if (getenv("TF_DS_REPLAY")) |r| std.mem.eql(u8, r, "1") else false, .most_drafts = most_drafts };
}

/// TF_DS_POOL's positions for a context of `context`: unset, the context; else a whole number at least the
/// context and a multiple of the extent alignment (error.BadPool otherwise).
pub fn poolSetting(text: ?[]const u8, context: usize) error{BadPool}!usize {
    const v = text orelse return context;
    const pool = std.fmt.parseInt(usize, std.mem.trim(u8, v, " "), 10) catch return error.BadPool;
    if (pool < context or pool % lanesmod.extent_align != 0) return error.BadPool;
    return pool;
}

/// This rank's device bytes: every entry of the rank cache (the backbone, and the DSpark blocks drafts load).
pub fn weightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    _ = dir;
    const cache_dir = getenv("TF_DS_RANK_CACHE") orelse return 0;
    const world = envInt("TF_TP_WORLD", 1) catch return 0;
    const rank = envInt("TF_TP_RANK", 0) catch return 0;
    const path = (RankCache.locate(a, io, cache_dir, rank, world) catch return 0) orelse return 0;
    var rc = RankCache.open(a, path) catch return 0;
    defer rc.close();
    var total: u64 = 0;
    for (rc.entries) |e| total += e.bytes;
    return total;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    var s = try settings(gpa, kernels, o.context);
    s.drafts = o.drafts;
    if (processAge()) |age| std.log.info("deepseek_v41 rank {d}: the engine's load begins {d:.1} s after the process started", .{ s.rank, age });
    if (s.world != 2) {
        std.log.err("deepseek_v41 serves on two ranks: start rank 0 and rank 1 with --tp 2 (TF_TP_WORLD=2)", .{});
        return error.WorldNotTwo;
    }
    const e = try Engine.init(gpa, io, ctx, dir, s);
    errdefer e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .lanes = Lanes.init(gpa, e) };
    // rank 0 times the rounds and drafts with rank 1 (its follow loop runs the same step)
    if (s.rank == 0) {
        var clock = lanesmod.Clock.start(s.rank);
        try own.lanes.measure(2);
        clock.mark("the measure step", .{});
        mem_note.note(ctx.d, "the measure step", .{});
    }
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = own.lanes.windowRows(),
        // every slot's caches (the pool, the drafter's rings) and the round graphs are allocated by the load: the
        // budget's room is what is left after them, so a stream needs no more (counting the pool again refused
        // streams the pool already holds)
        .stream_bytes = 0,
        .pool_positions = s.poolPositions(),
        .pool_bytes = e.streamBytes() * s.slots,
        .ctx = own,
        .deinit = release,
        .lone = null,
    };
}

/// Rank 1's loop: runs rank 0's steps until rank 0 stops; returns when the link closes.
pub fn follow(p: *anyopaque) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    try own.e.follow();
}

/// Whether this process is a follower (rank > 0), from the same environment `open` reads.
pub fn isFollower() bool {
    return (envInt("TF_TP_RANK", 0) catch 0) != 0;
}

/// Seconds since this process started (Linux: /proc/self/stat's start time against /proc/uptime), null elsewhere.
fn processAge() ?f64 {
    var stat_buf: [1024]u8 = undefined;
    var up_buf: [128]u8 = undefined;
    const stat = readSmall("/proc/self/stat", &stat_buf) orelse return null;
    const up = readSmall("/proc/uptime", &up_buf) orelse return null;
    // the fields after the command's closing parenthesis start at field 3; the start time (clock ticks after boot,
    // USER_HZ = 100 on Linux) is field 22
    const close = std.mem.lastIndexOfScalar(u8, stat, ')') orelse return null;
    var it = std.mem.tokenizeScalar(u8, stat[close + 1 ..], ' ');
    var field: usize = 3;
    const ticks = while (it.next()) |f| : (field += 1) {
        if (field == 22) break std.fmt.parseInt(u64, f, 10) catch return null;
    } else return null;
    var ut = std.mem.tokenizeAny(u8, up, " \n");
    const uptime = std.fmt.parseFloat(f64, ut.next() orelse return null) catch return null;
    return uptime - @as(f64, @floatFromInt(ticks)) / 100.0;
}

fn readSmall(path: [:0]const u8, buf: []u8) ?[]const u8 {
    const fd = std.c.open(path, .{});
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, buf.ptr, buf.len);
    return if (n > 0) buf[0..@intCast(n)] else null;
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.SamplingNotSupported => "this engine decodes greedily for now: send temperature 0",
        error.WindowTooWide, error.NoDraftHead, error.TreesNotBuilt => "this engine verifies drafted windows of up to TF_DS_DRAFTS + 1 rows (6 by default), 16 rows a round, no trees",
        error.NoRoom => "every stream slot is busy, or the context window shared by the running requests has no room for this one: retry when one finishes, or lower max_tokens",
        error.OutOfStep => "the two ranks planned different steps (or one cannot run a step: check both ranks' TF_DS_PREFILL_CHUNK, TF_DS_SLOTS, TF_DS_ENGRAM, TF_DS_DRAFTS and --no-drafts); restart both ranks",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.e.stop();
    own.e.deinit();
    own.gpa.destroy(own);
}

test "settings read the environment's rank and world" {
    _ = settings;
    try std.testing.expectEqual(@as(u32, 2048), prompt_rows);
}

test "TF_DS_POOL: the context unless set; at least the context, a multiple of the extent alignment" {
    const m: usize = 1 << 20;
    try std.testing.expectEqual(m, try poolSetting(null, m));
    try std.testing.expectEqual(@as(usize, 100000), try poolSetting(null, 100000));
    try std.testing.expectEqual(4 * m, try poolSetting("4194304", m));
    try std.testing.expectEqual(m, try poolSetting(" 1048576", m));
    try std.testing.expectError(error.BadPool, poolSetting("1046528", m)); // below the context
    try std.testing.expectError(error.BadPool, poolSetting("2097153", m)); // not aligned
    try std.testing.expectError(error.BadPool, poolSetting("4M", m));
}

/// A host-only stand-in for `Lanes` (the same facts and rows): records the core's calls, holds 5 drafts a draft
/// call, and draws each verify row as the held draft (so every draft lands) then a fixed token.
const Mock = struct {
    draft_depths: std.ArrayList(u32) = .empty,
    verify_held: std.ArrayList(u32) = .empty,
    held: u32 = 0,
    drafts: [5]u32 = .{ 11, 12, 13, 14, 15 },

    fn backend(m: *Mock) lanes.backend.Backend {
        return .{ .ptr = m, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = releaseFn } };
    }
    fn of(p: *anyopaque) *Mock {
        return @ptrCast(@alignCast(p));
    }
    fn prefill(_: *anyopaque, _: *lanes.Stream) anyerror!void {}
    fn first(_: *anyopaque, _: *lanes.Stream, _: u64) anyerror!u64 {
        return 0;
    }
    fn queue(_: *anyopaque, _: *lanes.Stream, _: lanes.backend.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }
    fn read(_: *anyopaque, _: u64) anyerror!u32 {
        return 10;
    }
    fn verify(p: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const m = of(p);
        const w = windows[0];
        try m.verify_held.append(std.testing.allocator, w.held);
        for (out[0].sampled, 0..) |*t, r| t.* = if (r < w.held) m.drafts[r] else 20;
        @memcpy(out[0].drafts[0..w.held], m.drafts[0..w.held]);
        @memcpy(out[0].drafts[w.held..][0..w.tokens.len], w.tokens);
        m.held = 0;
    }
    fn keep(_: *anyopaque, _: []const lanes.backend.Window, _: []const []const u32) anyerror!void {}
    fn draft(p: *anyopaque, requests: []const lanes.backend.DraftRequest) anyerror!void {
        const m = of(p);
        for (requests) |r| {
            try m.draft_depths.append(std.testing.allocator, r.depth);
            m.held = r.depth;
        }
    }
    fn releaseFn(_: *anyopaque, _: *lanes.Stream) void {}
};

test "a drafted stream drafts after its first token and verifies the held drafts (the facts and rows the host gets)" {
    const gpa = std.testing.allocator;
    // the window costs as measured on the two Sparks (31.3 .. 43.2 ms for 1 .. 6 rows, drafts included)
    const ms = [_]f64{ 31.3, 34.8 + 4.49, 38.2 + 4.49, 41.4 + 4.49, 41.5 + 4.49, 43.2 + 4.49 };
    var costs: [lanesmod.draft_width]lanes.config.Cost = undefined;
    for (&costs, ms, 1..) |*c, v, w| c.* = .{ .width = @intCast(w), .ms = v };
    const rows = lanesmod.rowsOf(true);
    var cfg = try lanes.Config.init(gpa, lanesmod.factsOf(true, 1, &costs, &.{}), rows, rows - 1);
    defer cfg.deinit(gpa);
    try std.testing.expect(cfg.family_mtp);
    try std.testing.expectEqual(@as(u32, lanesmod.draft_width - 1), cfg.most_drafts);
    var mock: Mock = .{};
    defer mock.draft_depths.deinit(gpa);
    defer mock.verify_held.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, mock.backend(), clock.clock());
    defer engine.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &prompt, .max_new = 24, .drafts = true });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    try std.testing.expect(mock.draft_depths.items.len == 1 and mock.draft_depths.items[0] >= 1);
    for (0..3) |_| if (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(mock.verify_held.items.len >= 1);
    try std.testing.expectEqual(mock.draft_depths.items[0], mock.verify_held.items[0]);
    for (mock.verify_held.items) |h| try std.testing.expect(h >= 1);

    // the old rows (one) turned drafting off whatever the facts said
    var one = try lanes.Config.init(gpa, lanesmod.factsOf(true, 1, &costs, &.{}), 1, 0);
    defer one.deinit(gpa);
    try std.testing.expect(!one.family_mtp and one.most_drafts == 0);
}
