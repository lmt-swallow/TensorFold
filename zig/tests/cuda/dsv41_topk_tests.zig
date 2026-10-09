//! `dsv41-topk [aot-dir]`: the DSV4.1 indexer's device selection (cuda_attention.zig: topk_glue.cu's row select,
//! the candidate pool and its mask, and with an AOT set the captured `_score_keys` + `_topk_sel`) against the host
//! oracle (kernels.topk_indices / model.py `_candidates` / `apply_candidates` on the host), on seeded scores built to
//! stress the total order: many ties, -0 beside +0, -inf past each row's visible count, visible counts of 0 and of
//! the whole row, and candidate pools smaller than the visible blocks (the binding case of long contexts).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const Gpu = check.Gpu;
const A = dsv41.attention;

const Case = struct { rows: usize, n: usize, kk: usize };

/// Scores [rows, n] with ties (a few distinct values), signed zeros, and -inf at or past vis[r].
fn scores(rnd: std.Random, rows: usize, n: usize, vis: []i64, out: []f32) void {
    for (0..rows) |r| {
        vis[r] = switch (r % 5) {
            0 => @intCast(n),
            1 => 0,
            else => @intCast(rnd.intRangeAtMost(usize, 0, n)),
        };
        const distinct = 1 + rnd.uintLessThan(usize, 40);
        for (0..n) |t| {
            const v: f32 = switch (rnd.uintLessThan(u8, 8)) {
                0 => 0.0,
                1 => -0.0,
                else => @as(f32, @floatFromInt(@as(i64, @intCast(rnd.uintLessThan(usize, distinct))) - 10)) * 0.37,
            };
            out[r * n + t] = if (@as(i64, @intCast(t)) < vis[r]) v else -std.math.inf(f32);
        }
    }
}

fn upload(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(d, bytes);
}

pub fn topk(gpu: Gpu, aot_dir: ?[]const u8) !void {
    const gpa = gpu.gpa;
    var glue = try A.Glue.load(gpu.d);
    defer glue.deinit();
    try check.expect(glue.topk != null, "no selection kernels in this build", .{});
    const tg = &glue.topk.?;
    var set: ?cuda.aot.Set = if (aot_dir) |dir| try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, dir) else null;
    defer if (set) |*s| s.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(20261009);
    const rnd = prng.random();
    var failures: usize = 0;

    // -- topk_indices: the glue's row select (and the fused selection where it applies) against the host --------
    const cases = [_]Case{
        .{ .rows = 5, .n = 37, .kk = 37 },     .{ .rows = 7, .n = 512, .kk = 512 },    .{ .rows = 16, .n = 600, .kk = 512 },
        .{ .rows = 16, .n = 1024, .kk = 512 }, .{ .rows = 9, .n = 2048, .kk = 512 },   .{ .rows = 16, .n = 4096, .kk = 512 },
        .{ .rows = 6, .n = 9001, .kk = 512 },  .{ .rows = 16, .n = 16384, .kk = 512 }, .{ .rows = 3, .n = 70000, .kk = 512 },
        .{ .rows = 40, .n = 4500, .kk = 512 }, .{ .rows = 4, .n = 2048, .kk = 2048 },  .{ .rows = 5, .n = 16, .kk = 8 },
    };
    for (cases) |cs| {
        const sc = try gpa.alloc(f32, cs.rows * cs.n);
        defer gpa.free(sc);
        const vis = try gpa.alloc(i64, cs.rows);
        defer gpa.free(vis);
        scores(rnd, cs.rows, cs.n, vis, sc);
        const want = try gpa.alloc(i64, cs.rows * cs.kk);
        defer gpa.free(want);
        const keys = try gpa.alloc(i64, cs.n);
        defer gpa.free(keys);
        for (0..cs.rows) |r| A.hostTopk(sc[r * cs.n .. (r + 1) * cs.n], cs.kk, vis[r], keys, want[r * cs.kk .. (r + 1) * cs.kk]);
        var d_sc = try upload(gpu.d, std.mem.sliceAsBytes(sc));
        defer d_sc.free();
        var d_vis = try upload(gpu.d, std.mem.sliceAsBytes(vis));
        defer d_vis.free();
        var d_out = try cuda.DeviceBuffer.alloc(gpu.d, cs.rows * cs.kk * 8);
        defer d_out.free();
        const got = try gpa.alloc(i64, cs.rows * cs.kk);
        defer gpa.free(got);
        try d_out.fill8(0xAB, null);
        try tg.topk(d_sc.ptr, cs.n, cs.rows, cs.n, cs.kk, d_vis.ptr, d_out.ptr, cs.kk, stream);
        try stream.synchronize();
        try d_out.download(0, std.mem.sliceAsBytes(got));
        var what: [96]u8 = undefined;
        if (check.sameBytes(try std.fmt.bufPrint(&what, "topk glue rows {d} n {d} k {d}", .{ cs.rows, cs.n, cs.kk }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) |_| {
            check.pass("dsv41-topk glue: {d} rows of {d} scores, k {d}: the host oracle's picks", .{ cs.rows, cs.n, cs.kk });
        } else |_| failures += 1;
        {
            // the same picks from the score kernels' int64 keys (KEYS)
            const hk = try gpa.alloc(i64, cs.rows * cs.n);
            defer gpa.free(hk);
            for (0..cs.rows) |r| for (0..cs.n) |t| {
                hk[r * cs.n + t] = A.key(sc[r * cs.n + t], t);
            };
            var d_k = try upload(gpu.d, std.mem.sliceAsBytes(hk));
            defer d_k.free();
            try d_out.fill8(0xAB, null);
            try tg.topkKeys(d_k.ptr, cs.n, cs.rows, cs.n, cs.kk, d_vis.ptr, d_out.ptr, cs.kk, stream);
            try stream.synchronize();
            try d_out.download(0, std.mem.sliceAsBytes(got));
            if (check.sameBytes(try std.fmt.bufPrint(&what, "topk keys rows {d} n {d} k {d}", .{ cs.rows, cs.n, cs.kk }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) |_| {
                check.pass("dsv41-topk glue from keys: {d} rows of {d} keys, k {d}: the host oracle's picks", .{ cs.rows, cs.n, cs.kk });
            } else |_| failures += 1;
        }
        if (set) |*s| {
            var d_keys = try cuda.DeviceBuffer.alloc(gpu.d, cs.rows * cs.n * 8);
            defer d_keys.free();
            try d_out.fill8(0xAB, null);
            if (try A.fusedTopk(s, d_sc.ptr, cs.rows, cs.n, cs.kk, d_vis.ptr, d_keys.ptr, d_out.ptr, stream)) {
                try stream.synchronize();
                try d_out.download(0, std.mem.sliceAsBytes(got));
                if (check.sameBytes(try std.fmt.bufPrint(&what, "topk fused rows {d} n {d} k {d}", .{ cs.rows, cs.n, cs.kk }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) |_| {
                    check.pass("dsv41-topk fused (_score_keys + _topk_sel N={d} KK={d}): {d} rows: the host oracle's picks", .{ cs.n, cs.kk, cs.rows });
                } else |_| failures += 1;
            }
        }
    }

    // -- _candidates and apply_candidates, pools smaller than the visible blocks ---------------------------------
    const cand_cases = [_]struct { rows: usize, n: usize, nblocks: usize, bsize: usize }{
        .{ .rows = 16, .n = 1000, .nblocks = 16, .bsize = 8 },
        .{ .rows = 16, .n = 17000, .nblocks = 2048, .bsize = 8 },
        .{ .rows = 9, .n = 32768, .nblocks = 2048, .bsize = 8 },
        .{ .rows = 5, .n = 9001, .nblocks = 2048, .bsize = 8 },
    };
    for (cand_cases) |cs| {
        const kk: usize = 512;
        const nblk = (cs.n + cs.bsize - 1) / cs.bsize;
        const sc = try gpa.alloc(f32, cs.rows * cs.n);
        defer gpa.free(sc);
        const vis = try gpa.alloc(i64, cs.rows);
        defer gpa.free(vis);
        scores(rnd, cs.rows, cs.n, vis, sc);
        // the binding case wants most rows' visible blocks past the pool: widen the visible counts
        for (vis, 0..) |*v, r| if (r % 5 >= 2) {
            v.* = @intCast(cs.n - rnd.uintLessThan(usize, @max(1, cs.n / 50)));
            for (0..cs.n) |t| {
                if (@as(i64, @intCast(t)) < v.*) {
                    if (sc[r * cs.n + t] == -std.math.inf(f32)) sc[r * cs.n + t] = @as(f32, @floatFromInt(rnd.uintLessThan(u32, 30))) * 0.11 - 1.0;
                } else sc[r * cs.n + t] = -std.math.inf(f32);
            }
        };
        const mask_want = try gpa.alloc(u8, cs.rows * nblk);
        defer gpa.free(mask_want);
        const keys = try gpa.alloc(i64, cs.n);
        defer gpa.free(keys);
        const masked = try gpa.dupe(f32, sc);
        defer gpa.free(masked);
        const want = try gpa.alloc(i64, cs.rows * kk);
        defer gpa.free(want);
        var bound: usize = 0;
        for (0..cs.rows) |r| {
            const row = sc[r * cs.n .. (r + 1) * cs.n];
            const m = mask_want[r * nblk .. (r + 1) * nblk];
            A.hostCandidates(row, vis[r], cs.nblocks, cs.bsize, keys, m);
            const vis_blocks: usize = @intCast(@divFloor(vis[r] + @as(i64, @intCast(cs.bsize)) - 1, @as(i64, @intCast(cs.bsize))));
            if (vis_blocks > cs.nblocks) bound += 1;
            A.hostApplyCandidates(masked[r * cs.n .. (r + 1) * cs.n], m, cs.bsize);
            A.hostTopk(masked[r * cs.n .. (r + 1) * cs.n], kk, vis[r], keys, want[r * kk .. (r + 1) * kk]);
        }
        var d_sc = try upload(gpu.d, std.mem.sliceAsBytes(sc));
        defer d_sc.free();
        var d_vis = try upload(gpu.d, std.mem.sliceAsBytes(vis));
        defer d_vis.free();
        var d_smax = try cuda.DeviceBuffer.alloc(gpu.d, cs.rows * nblk * 4);
        defer d_smax.free();
        var d_mask = try cuda.DeviceBuffer.alloc(gpu.d, cs.rows * nblk);
        defer d_mask.free();
        var d_out = try cuda.DeviceBuffer.alloc(gpu.d, cs.rows * kk * 8);
        defer d_out.free();
        try d_mask.fill8(0xEE, null);
        try tg.candidates(d_sc.ptr, cs.n, cs.rows, cs.n, d_vis.ptr, cs.nblocks, cs.bsize, d_smax.ptr, d_mask.ptr, nblk, stream);
        try tg.apply(d_sc.ptr, cs.n, cs.rows, cs.n, d_mask.ptr, nblk, cs.bsize, stream);
        try tg.topk(d_sc.ptr, cs.n, cs.rows, cs.n, kk, d_vis.ptr, d_out.ptr, kk, stream);
        try stream.synchronize();
        const mask_got = try gpa.alloc(u8, cs.rows * nblk);
        defer gpa.free(mask_got);
        try d_mask.download(0, mask_got);
        const masked_got = try gpa.alloc(f32, cs.rows * cs.n);
        defer gpa.free(masked_got);
        try d_sc.download(0, std.mem.sliceAsBytes(masked_got));
        const got = try gpa.alloc(i64, cs.rows * kk);
        defer gpa.free(got);
        try d_out.download(0, std.mem.sliceAsBytes(got));
        var what: [96]u8 = undefined;
        var ok = true;
        check.sameBytes(try std.fmt.bufPrint(&what, "candidates mask n {d} pool {d}", .{ cs.n, cs.nblocks }), mask_got, mask_want) catch {
            ok = false;
        };
        check.sameBytes(try std.fmt.bufPrint(&what, "apply_candidates n {d} pool {d}", .{ cs.n, cs.nblocks }), std.mem.sliceAsBytes(masked_got), std.mem.sliceAsBytes(masked)) catch {
            ok = false;
        };
        check.sameBytes(try std.fmt.bufPrint(&what, "picks after the pool n {d} pool {d}", .{ cs.n, cs.nblocks }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want)) catch {
            ok = false;
        };
        if (ok) {
            check.pass("dsv41-topk candidates: {d} rows of {d} scores, a pool of {d} blocks of {d} ({d} rows with more visible blocks than the pool): mask, masked scores and picks equal the host oracle's", .{ cs.rows, cs.n, cs.nblocks, cs.bsize, bound });
        } else failures += 1;
    }
    try check.expect(failures == 0, "{d} selection case(s) differ from the host oracle", .{failures});
}

/// `dsv41-rmsnorm-torch <dir>`: cuda_prompt.zig's torch-order rms_norm (prompt_glue.cu) against the fork's
/// ops.rms_norm on tools/zig/dsv41/dump_rmsnorm.py's rows (128 and 512 columns), byte for byte.
pub fn rmsnormTorch(gpu: Gpu, dir: []const u8) !void {
    const gpa = gpu.gpa;
    const fixture = @import("fixture.zig");
    var glue = try dsv41.prompt.Glue.load(gpu.d);
    defer glue.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var failures: usize = 0;
    for ([_]usize{ 128, 512 }) |d| {
        var nb: [32]u8 = undefined;
        const path = try std.fs.path.join(gpa, &.{ dir, try std.fmt.bufPrint(&nb, "rms_{d}", .{d}) });
        defer gpa.free(path);
        var fx = try fixture.Fixture.open(gpa, gpu.io, path);
        defer fx.deinit();
        const rows: usize = @intCast(try fx.int("rows"));
        const eps: f32 = @floatCast(try fx.float("eps"));
        const xb = try fx.bytes("x");
        defer gpa.free(xb);
        const wb = try fx.bytes("w");
        defer gpa.free(wb);
        const want = try fx.bytes("expected_y");
        defer gpa.free(want);
        var dx = try upload(gpu.d, xb);
        defer dx.free();
        var dw = try upload(gpu.d, wb);
        defer dw.free();
        var dy = try cuda.DeviceBuffer.alloc(gpu.d, want.len);
        defer dy.free();
        try glue.rmsNormTorch(dx.ptr, dw.ptr, eps, rows, d, dy.ptr, stream);
        try stream.synchronize();
        const got = try gpa.alloc(u8, want.len);
        defer gpa.free(got);
        try dy.download(0, got);
        var what: [64]u8 = undefined;
        if (check.sameBytes(try std.fmt.bufPrint(&what, "rms_norm torch order d {d}", .{d}), got, want)) |_| {
            check.pass("BITEXACT dsv41 rms_norm (torch order): {d} rows of {d} equal to ops.rms_norm's", .{ rows, d });
        } else |_| failures += 1;
    }
    try check.expect(failures == 0, "{d} rms_norm case(s) differ", .{failures});
}
