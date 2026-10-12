//! `glm53-topk`: the draft top-k on the device (glm53/topk.cu: the one-block `tf_glm53_topk_row` and the two-stage
//! `tf_glm53_topk_part` + `tf_glm53_topk_merge`, TF_GLM_DRAFT_TOPK=1 / 2) against cuda_engine.zig's host scan of the
//! same bf16 row (Runner.hostBest: the first k taken, then a later value only when strictly better, so of equal values
//! the lowest ids stay), on rows of a rank's vocabulary share: distinct values, one value repeated at more than k ids
//! around the cut, signed zeros, -inf columns and every column equal.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const Gpu = check.Gpu;
const Buf = cuda.DeviceBuffer;

const nv = 77_440; // a rank's vocabulary share
const k = 28;
const blocks = 64;

const Cand = struct { v: f32, id: u32 };

fn better(_: void, a: Cand, b: Cand) bool {
    return a.v > b.v or (a.v == b.v and a.id < b.id);
}

/// cuda_engine.zig `Runner.hostBest`, as written there.
fn hostBest(row: []const u16, base: u32) [2 * k]u32 {
    var best: [k]Cand = undefined;
    var n: usize = 0;
    for (row, 0..) |x, j| {
        const cand: Cand = .{ .v = @bitCast(@as(u32, x) << 16), .id = base + @as(u32, @intCast(j)) };
        if (n < k) {
            best[n] = cand;
            n += 1;
            std.mem.sort(Cand, best[0..n], {}, better);
        } else if (better({}, cand, best[k - 1])) {
            best[k - 1] = cand;
            std.mem.sort(Cand, best[0..k], {}, better);
        }
    }
    var out: [2 * k]u32 = undefined;
    for (best, 0..) |b, i| {
        out[i] = @bitCast(b.v);
        out[k + i] = b.id;
    }
    return out;
}

const Pattern = enum { distinct, ties_at_cut, signed_zero, neg_inf, all_equal };

fn bf16(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

pub fn run(gpu: Gpu) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var module = try cuda.Module.load(d, cuda.kernels.glm53_topk);
    defer module.unload();
    const row_fn = try module.function("tf_glm53_topk_row");
    const part_fn = try module.function("tf_glm53_topk_part");
    const merge_fn = try module.function("tf_glm53_topk_merge");
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var keys = try Buf.alloc(d, blocks * k * 8);
    defer keys.free();
    var out1 = try Buf.alloc(d, 2 * k * 4);
    defer out1.free();
    var out2 = try Buf.alloc(d, 2 * k * 4);
    defer out2.free();
    var prng = std.Random.DefaultPrng.init(20261015);
    const rng = prng.random();
    const row = try gpa.alloc(u16, nv);
    defer gpa.free(row);
    var failures: usize = 0;
    var cases: usize = 0;
    for (std.enums.values(Pattern)) |p| for (0..8) |rep| for ([_]u32{ 0, nv }) |base| {
        for (row) |*x| x.* = switch (p) {
            .distinct => bf16(rng.floatNorm(f32) * 4.0),
            // a value above the rest at 40 scattered ids (more than k), the rest below
            .ties_at_cut => bf16(rng.floatNorm(f32) - 8.0),
            .signed_zero => if (rng.boolean()) 0x8000 else 0x0000,
            .neg_inf => if (rng.intRangeLessThan(u32, 0, 4) == 0) bf16(rng.floatNorm(f32)) else 0xFF80,
            .all_equal => bf16(2.5),
        };
        if (p == .ties_at_cut) {
            const top = bf16(3.0);
            for (0..10) |_| row[rng.intRangeLessThan(usize, 0, nv)] = bf16(5.0 + rng.float(f32)); // a few above
            for (0..40) |_| row[rng.intRangeLessThan(usize, 0, nv)] = top;
        }
        var src = try Buf.fromHost(d, std.mem.sliceAsBytes(row));
        defer src.free();
        const want = hostBest(row, base);
        // one block
        var a: cuda.Args = .{};
        a.add(src.ptr);
        a.add(@as(i32, nv));
        a.add(@as(i32, k));
        a.add(@as(i32, @intCast(base)));
        a.add(out1.ptr);
        try cuda.launch.launch(row_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, stream, &a);
        // two stages
        const slice: i32 = (nv + blocks - 1) / blocks;
        var b: cuda.Args = .{};
        b.add(src.ptr);
        b.add(@as(i32, nv));
        b.add(slice);
        b.add(@as(i32, k));
        b.add(@as(i32, @intCast(base)));
        b.add(keys.ptr);
        try cuda.launch.launch(part_fn, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, stream, &b);
        var c: cuda.Args = .{};
        c.add(src.ptr);
        c.add(keys.ptr);
        c.add(@as(i32, blocks));
        c.add(@as(i32, k));
        c.add(@as(i32, @intCast(base)));
        c.add(out2.ptr);
        try cuda.launch.launch(merge_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &c);
        try stream.synchronize();
        var got1: [2 * k]u32 = undefined;
        var got2: [2 * k]u32 = undefined;
        try out1.download(0, std.mem.sliceAsBytes(&got1));
        try out2.download(0, std.mem.sliceAsBytes(&got2));
        cases += 1;
        const ok1 = std.mem.eql(u32, &got1, &want);
        const ok2 = std.mem.eql(u32, &got2, &want);
        if (!ok1 or !ok2) {
            failures += 1;
            std.debug.print("FAIL glm53-topk {t} #{d} base {d}: one block {s}, two stages {s}\n", .{ p, rep, base, if (ok1) "equal" else "differs", if (ok2) "equal" else "differs" });
        }
    };
    // the two stages' time a row (the host scan is about 1.4 ms on the node)
    var t0 = try cuda.Event.init(d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(d, true);
    defer t1.deinit();
    var src = try Buf.fromHost(d, std.mem.sliceAsBytes(row));
    defer src.free();
    try t0.record(stream);
    for (0..100) |_| {
        const slice: i32 = (nv + blocks - 1) / blocks;
        var b: cuda.Args = .{};
        b.add(src.ptr);
        b.add(@as(i32, nv));
        b.add(slice);
        b.add(@as(i32, k));
        b.add(@as(i32, 0));
        b.add(keys.ptr);
        try cuda.launch.launch(part_fn, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, stream, &b);
        var c: cuda.Args = .{};
        c.add(src.ptr);
        c.add(keys.ptr);
        c.add(@as(i32, blocks));
        c.add(@as(i32, k));
        c.add(@as(i32, 0));
        c.add(out2.ptr);
        try cuda.launch.launch(merge_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &c);
    }
    try t1.record(stream);
    try t1.synchronize();
    const ms = try t0.elapsedMs(t1);
    std.debug.print("glm53-topk: two stages {d:.1} us a row (100 rows)\n", .{ms * 10.0});
    try check.expect(failures == 0, "glm53-topk: {d} of {d} rows differ from the host scan", .{ failures, cases });
    check.pass("glm53-topk: {d} rows equal to the host scan (one block and two stages)", .{cases});
}
