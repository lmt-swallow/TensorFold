//! `qwen38fn-select <kernel-set> <model-dir>`: the indexer's block selection (cuda_attn.zig `select`, the captured
//! `_select` and `_select_tiles`) lists, for every sparse row, the TOP blocks of largest score (ties: lower block id
//! first) in block order, then the unfinished tail; a dense row keeps only its length. Rows ending anywhere up to
//! the caches' capacity (past 131,072 keys `_select_tiles` runs), prompt row blocks (256 rows, 4,096-block tiles) and
//! decode windows (16 rows, 8,192-block tiles), scores with many ties and both zeros, each against a host oracle; where
//! both kernels fit, `_select_tiles` must list what `_select` lists.

const std = @import("std");
const cuda = @import("cuda");
const qfn = @import("qwen38fn");
const check = @import("check.zig");
const Gpu = check.Gpu;
const A = qfn.attn;

/// `_select`'s order-preserving key of a score.
fn key(v: f32) u32 {
    const bits: u32 = @bitCast(v);
    return if (bits & 0x80000000 != 0) ~bits else bits | 0x80000000;
}

const Lists = struct { ids: []i32, nk: []i32, sparse: []i32 };

fn download(gpa: std.mem.Allocator, s: *const A.Scratch, rows: usize) !Lists {
    const l: Lists = .{ .ids = try gpa.alloc(i32, rows * s.idw), .nk = try gpa.alloc(i32, rows), .sparse = try gpa.alloc(i32, rows) };
    try s.ids.download(0, std.mem.sliceAsBytes(l.ids));
    try s.nk.download(0, std.mem.sliceAsBytes(l.nk));
    try s.sparse.download(0, std.mem.sliceAsBytes(l.sparse));
    return l;
}

fn free(gpa: std.mem.Allocator, l: Lists) void {
    gpa.free(l.ids);
    gpa.free(l.nk);
    gpa.free(l.sparse);
}

/// Row r's mismatches against the oracle (0 when equal).
fn oracle(gpa: std.mem.Allocator, s: *const A.Scratch, scores: []const f32, pos0: usize, r: usize, got: Lists) !usize {
    const ratio = s.ratio;
    const top = s.budget / ratio;
    const end = pos0 + r + 1;
    const complete = end / ratio;
    if (complete <= top) return @intFromBool(got.nk[r] != end or got.sparse[r] != 0);
    const row = scores[r * s.nb ..][0..complete];
    const order = try gpa.alloc(u32, complete);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    const Ctx = struct {
        row: []const f32,
        fn less(c: @This(), a: u32, b: u32) bool {
            const ka = key(c.row[a]);
            const kb = key(c.row[b]);
            return if (ka != kb) ka > kb else a < b;
        }
    };
    std.mem.sort(u32, order, Ctx{ .row = row }, Ctx.less);
    const chosen = order[0..top];
    std.mem.sort(u32, chosen, {}, std.sort.asc(u32));
    var bad: usize = 0;
    const nk = top * ratio + end - ratio * complete;
    if (got.nk[r] != nk or got.sparse[r] != 1) return 1;
    const ids = got.ids[r * s.idw ..][0..nk];
    for (chosen, 0..) |b, i| for (0..ratio) |k| {
        bad += @intFromBool(ids[i * ratio + k] != @as(i32, @intCast(b * ratio + k)));
    };
    for (0..end - ratio * complete) |t| bad += @intFromBool(ids[top * ratio + t] != @as(i32, @intCast(ratio * complete + t)));
    return bad;
}

pub fn select(gpu: Gpu, kernels: []const u8, model: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: qfn.config.Why = .{};
    const cfg = try qfn.config.read(gpa, gpu.io, model, &why);
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, kernels);
    defer set.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const k: A.Kernels = .{ .set = &set, .cfg = &cfg };
    const capacity = cfg.max_positions;
    var pos_dev = try cuda.DeviceBuffer.alloc(d, 16);
    defer pos_dev.free();
    var rng = std.Random.DefaultPrng.init(0x5e1ec7);
    const rand = rng.random();
    var failures: usize = 0;
    var checked: usize = 0;
    var tiled: usize = 0;
    // (rows, the window's first position): across the 32,768-block width, deep, and ending at the capacity
    const cases = [_][2]usize{
        .{ 256, 0 },            .{ 256, 1792 },           .{ 256, 131072 - 256 }, .{ 256, 131072 - 128 },
        .{ 256, 200000 },       .{ 256, capacity - 256 }, .{ 16, 2040 },          .{ 16, 131072 - 8 },
        .{ 16, 163840 + 3 },    .{ 16, capacity - 16 },   .{ 1, capacity - 1 },
    };
    for (cases) |c| {
        const rows, const pos0 = c;
        var s = try A.Scratch.init(d, &cfg, rows, capacity);
        defer s.deinit();
        const scores = try gpa.alloc(f32, rows * s.nb);
        defer gpa.free(scores);
        for (0..2) |kind| {
            // kind 0: few distinct values (ties everywhere, +0 and -0 among them); kind 1: spread values
            for (scores) |*v| v.* = if (kind == 0) switch (rand.uintLessThan(u32, 8)) {
                0 => 0.0,
                1 => -0.0,
                else => @as(f32, @floatFromInt(rand.uintLessThan(u32, 16))) * 0.5 - 2.0,
            } else (rand.float(f32) - 0.25) * 8.0;
            try s.scores.upload(0, std.mem.sliceAsBytes(scores));
            const p: i32 = @intCast(pos0);
            try pos_dev.upload(0, std.mem.asBytes(&p));
            const blocks = @min(s.nb, @max(1, (pos0 + rows + s.ratio - 1) / s.ratio));
            const width = std.math.ceilPowerOfTwoAssert(usize, blocks);
            const runs: []const bool = if (width <= A.select_regs) &.{ false, true } else &.{false};
            var first: ?Lists = null;
            defer if (first) |l| free(gpa, l);
            for (runs) |tiles| {
                try s.ids.fill8(0xff, null);
                try k.select(pos_dev.ptr, &s, rows, blocks, tiles, stream);
                try stream.synchronize();
                const got = try download(gpa, &s, rows);
                var bad: usize = 0;
                for (0..rows) |r| bad += try oracle(gpa, &s, scores, pos0, r, got);
                if (tiles or width > A.select_regs) tiled += 1;
                checked += 1;
                if (bad != 0) {
                    std.debug.print("FAIL qwen38fn-select rows {d} at {d} ({s}, {s} scores): {d} entries differ from the oracle\n", .{ rows, pos0, if (tiles or width > A.select_regs) "_select_tiles" else "_select", if (kind == 0) "tied" else "spread", bad });
                    failures += 1;
                }
                if (first) |f| {
                    if (!std.mem.eql(i32, f.nk, got.nk) or !std.mem.eql(i32, f.sparse, got.sparse)) failures += 1;
                    free(gpa, got);
                } else first = got;
            }
        }
    }
    try check.expect(failures == 0, "qwen38fn-select: {d} of {d} launches differ", .{ failures, checked });
    check.pass("qwen38fn-select: {d} launches ({d} of _select_tiles, up to {d} positions) list the oracle's blocks", .{ checked, tiled, capacity });
}
