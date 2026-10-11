//! `dsv41-index-probe <aot-dir>`: the order in which the captured prompt-chunk indexer kernel (`_index_score_rows`)
//! adds a score's 32 heads. Every row of a launch is one probe: three heads contribute 2^24, 1 and -2^24 (each head's
//! dot product is one exact product, its weight +-1), the others 0, so the score is 0 when the 1 joins the 2^24 head's
//! partial sum before the -2^24 head does (2^24 + 1 rounds to 2^24) and 1 otherwise. Prints `PROBE a b c score` for every
//! (a, b, c) at two row offsets, then `FUSE a b score` lines saying whether head a's product is fused (fma) into the
//! addition where it meets head b's partial sum (FAILs when a row's 64 key columns disagree);
//! tools/zig/dsv41/index_tree.py checks index_score.cu's order against them.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const Gpu = check.Gpu;
const aot = cuda.aot;

const IH = 32;
const ID = 128;
const N = 64; // keys (one tile, every key the same)
const ROWS = 2048;

fn bf16(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16); // exact for the probe's values (powers of two, 0)
}

pub fn probe(gpu: Gpu, aot_dir: []const u8) !void {
    const gpa = gpu.gpa;
    var set = try aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    // keys: dims 0, 1 and 2 = 1.0 (e2m1 code 2 in the low nibbles of bytes 0..2), every other dim 0, every E8M0
    // scale 2^0
    var kc: [N * ID / 2]u8 = @splat(0);
    for (0..N) |t| for (0..3) |d| {
        kc[t * ID / 2 + d] = 0x02;
    };
    const ks: [N * ID / 32]u8 = @splat(127);
    var k_codes = try cuda.DeviceBuffer.fromHost(gpu.d, &kc);
    defer k_codes.free();
    var k_scales = try cuda.DeviceBuffer.fromHost(gpu.d, &ks);
    defer k_scales.free();
    const vis_h: [ROWS]i64 = @splat(N);
    var vis = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&vis_h));
    defer vis.free();
    var q = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * IH * ID * 2);
    defer q.free();
    var w = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * IH * 2);
    defer w.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * N * 4);
    defer out.free();

    const q_h = try gpa.alloc(u16, ROWS * IH * ID);
    defer gpa.free(q_h);
    const w_h = try gpa.alloc(u16, ROWS * IH);
    defer gpa.free(w_h);
    const o_h = try gpa.alloc(f32, ROWS * N);
    defer gpa.free(o_h);

    // every (a, b, c): a < c, b neither
    var probes: std.ArrayList([3]u8) = .empty;
    defer probes.deinit(gpa);
    for (0..IH) |a| for (a + 1..IH) |c| for (0..IH) |b| if (b != a and b != c) try probes.append(gpa, .{ @intCast(a), @intCast(b), @intCast(c) });

    var bad_cols: usize = 0;
    for ([_]usize{ 0, 3 }) |shift| {
        var p0: usize = 0;
        while (p0 < probes.items.len) {
            @memset(q_h, 0);
            @memset(w_h, 0);
            const take = @min(ROWS - shift, probes.items.len - p0);
            for (0..take) |i| {
                const r = shift + i;
                const p = probes.items[p0 + i];
                const vals = [3]f32{ 16777216.0, 1.0, 16777216.0 };
                const wts = [3]f32{ 1.0, 1.0, -1.0 };
                for (p, vals, wts) |h, v, wt| {
                    q_h[(r * IH + h) * ID] = bf16(v);
                    w_h[r * IH + h] = bf16(wt);
                }
            }
            try launch(&set, stream, q, w, k_codes, k_scales, vis, out, q_h, w_h, o_h);
            for (0..take) |i| {
                const r = shift + i;
                const p = probes.items[p0 + i];
                const row = o_h[r * N .. (r + 1) * N];
                for (row) |v| bad_cols += @intFromBool(@as(u32, @bitCast(v)) != @as(u32, @bitCast(row[0])));
                std.debug.print("PROBE {d} {d} {d} {d} {d}\n", .{ shift, p[0], p[1], p[2], row[0] });
            }
            p0 += take;
        }
    }
    // which products are fused (fma) into the addition where their head joins: head a's product (1 + 2^-23)(1 + 2^-7)
    // rounds off 2^-30 unless fused; head b contributes -(1 + 2^-7 + 2^-23) exactly. The score is 2^-30 when a's
    // product meets a partial sum holding b inside an fma, else 0.
    var pairs: std.ArrayList([2]u8) = .empty;
    defer pairs.deinit(gpa);
    for (0..IH) |a| for (0..IH) |b| if (a != b) try pairs.append(gpa, .{ @intCast(a), @intCast(b) });
    var p0: usize = 0;
    while (p0 < pairs.items.len) {
        @memset(q_h, 0);
        @memset(w_h, 0);
        const take = @min(ROWS, pairs.items.len - p0);
        for (0..take) |r| {
            const p = pairs.items[p0 + r];
            const qa = q_h[(r * IH + p[0]) * ID ..];
            qa[0] = bf16(1.0);
            qa[1] = bf16(std.math.pow(f32, 2, -23));
            w_h[r * IH + p[0]] = bf16(1.0 + std.math.pow(f32, 2, -7));
            const qb = q_h[(r * IH + p[1]) * ID ..];
            qb[0] = bf16(1.0);
            qb[1] = bf16(std.math.pow(f32, 2, -7));
            qb[2] = bf16(std.math.pow(f32, 2, -23));
            w_h[r * IH + p[1]] = bf16(-1.0);
        }
        try launch(&set, stream, q, w, k_codes, k_scales, vis, out, q_h, w_h, o_h);
        for (0..take) |r| {
            const p = pairs.items[p0 + r];
            const row = o_h[r * N .. (r + 1) * N];
            for (row) |v| bad_cols += @intFromBool(@as(u32, @bitCast(v)) != @as(u32, @bitCast(row[0])));
            std.debug.print("FUSE {d} {d} {e}\n", .{ p[0], p[1], row[0] });
        }
        p0 += take;
    }
    try check.expect(bad_cols == 0, "dsv41-index-probe: {d} key columns disagree with their row's first", .{bad_cols});
    check.pass("dsv41-index-probe: {d} probes at 2 row offsets, every key column equal", .{probes.items.len});
}

fn launch(set: *aot.Set, stream: cuda.Stream, q: cuda.DeviceBuffer, w: cuda.DeviceBuffer, k_codes: cuda.DeviceBuffer, k_scales: cuda.DeviceBuffer, vis: cuda.DeviceBuffer, out: cuda.DeviceBuffer, q_h: []const u16, w_h: []const u16, o_h: []f32) !void {
    try q.upload(0, std.mem.sliceAsBytes(q_h));
    try w.upload(0, std.mem.sliceAsBytes(w_h));
    try set.run(stream, "_index_score_rows", .{ ROWS / 8, 1, 1 }, &.{
        aot.ptr("Q", "*bf16", q.ptr),       aot.ptr("K", "*u8", k_codes.ptr),
        aot.ptr("KS", "*u8", k_scales.ptr), aot.ptr("Wt", "*bf16", w.ptr),
        aot.ptr("VIS", "*i64", vis.ptr),    aot.ptr("OUT", "*fp32", out.ptr),
        aot.int("n", N),                    aot.int("rows", ROWS),
        aot.ptr("TMAX", "*fp32", out.ptr),  aot.int("nt", 1),
    }, &.{ aot.ci("IH", IH), aot.ci("ID", ID), aot.ci("BN", 64), aot.ci("RB", 8), aot.ci("PACKED", 1), aot.ci("KEYS", 0), aot.ci("HAS_TMAX", 0) });
    try stream.synchronize();
    try out.download(0, std.mem.sliceAsBytes(o_h));
}
