//! DeepSeek-V4.1 family pieces against fixtures the Python fork dumped (zig/tests/cuda/dsv41/*.py).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;

/// `exl3-linear <dir>`: every layer and row count of dump_linear.py's fixture, bf16 and fp32 outputs byte-equal.
pub fn exl3Linear(gpu: Gpu, dir: []const u8) !void {
    var fx = try fixture.Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var kernels = try dsv41.exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const layers: usize = @intCast(try fx.int("layers"));
    const rows_text = try fx.string("rows");
    var failures: usize = 0;
    for (0..layers) |li| {
        var nb: [64]u8 = undefined;
        const kt: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "L{d}.kt", .{li})));
        const nt: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "L{d}.nt", .{li})));
        const w16: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "L{d}.w16", .{li})));
        const name = try gpu.gpa.dupe(u8, try fx.string(try std.fmt.bufPrint(&nb, "L{d}.name", .{li})));
        defer gpu.gpa.free(name);
        const trellis = try fx.bytes(try std.fmt.bufPrint(&nb, "L{d}.trellis", .{li}));
        defer gpu.gpa.free(trellis);
        const suh = try fx.bytes(try std.fmt.bufPrint(&nb, "L{d}.suh", .{li}));
        defer gpu.gpa.free(suh);
        const svh = try fx.bytes(try std.fmt.bufPrint(&nb, "L{d}.svh", .{li}));
        defer gpu.gpa.free(svh);
        var layer = try dsv41.exl3.Linear.init(gpu.gpa, &kernels, trellis, kt, nt, w16, suh, svh, .mul1);
        defer layer.deinit();
        var it = std.mem.splitScalar(u8, rows_text, ',');
        while (it.next()) |r| {
            const m = try std.fmt.parseInt(usize, r, 10);
            const x = try fx.bytes(try std.fmt.bufPrint(&nb, "L{d}.x{d}", .{ li, m }));
            defer gpu.gpa.free(x);
            var xd = try cuda.DeviceBuffer.fromHost(gpu.d, x);
            defer xd.free();
            var xh = try cuda.DeviceBuffer.alloc(gpu.d, m * layer.k * 2);
            defer xh.free();
            var z = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, layer.scratchWords(m) * 4));
            defer z.free();
            const Case = struct { name: []const u8, dtype: dsv41.exl3.DType, size: usize };
            const cases = [_]Case{ .{ .name = "bf16", .dtype = .bf16, .size = 2 }, .{ .name = "f32", .dtype = .f32, .size = 4 } };
            for (cases) |case| {
                const want = try fx.bytes(try std.fmt.bufPrint(&nb, "L{d}.y_{s}{d}", .{ li, case.name, m }));
                defer gpu.gpa.free(want);
                var y = try cuda.DeviceBuffer.alloc(gpu.d, m * layer.n * case.size);
                defer y.free();
                try y.fill8(0, null);
                try layer.run(xd.ptr, .bf16, m, xh.ptr, z.ptr, y.ptr, case.dtype, stream);
                try stream.synchronize();
                const got = try check.download(gpu, y);
                defer gpu.gpa.free(got);
                var what: [128]u8 = undefined;
                const label = try std.fmt.bufPrint(&what, "exl3-linear {s} K={d} N={d} bits={d}/2 sk={d} wk={d} rows={d} {s}", .{ name, layer.k, layer.n, layer.k2, layer.sk, layer.wk, m, case.name });
                if (check.sameBytes(label, got, want)) {
                    check.pass("{s}", .{label});
                } else |_| {
                    failures += 1;
                }
            }
        }
    }
    try check.expect(failures == 0, "exl3-linear: {d} cases differ from the Python fork", .{failures});
}

/// `exl3-experts <dir>`: every rank's halves and row count of dump_experts.py's fixture, the fused decode window's
/// fp32 partial byte-equal to the fork's.
pub fn exl3Experts(gpu: Gpu, dir: []const u8) !void {
    var fx = try fixture.Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const params = fx.parsed.value.object.get("params").?.object;
    var kernels = try dsv41.experts.Kernels.load(gpu.d);
    defer kernels.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const ranks: usize = @intCast(try fx.int("ranks"));
    const slots: usize = @intCast(try fx.int("slots"));
    const limit: f32 = @floatCast(try fx.float("limit"));
    const rows_text = try fx.string("rows");
    var failures: usize = 0;
    for (0..ranks) |rank| {
        var nb: [64]u8 = undefined;
        const E: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "R{d}.E", .{rank})));
        const D: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "R{d}.D", .{rank})));
        const I: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "R{d}.I", .{rank})));
        const names = [_][]const u8{ "gate", "up", "down" };
        var blobs: [3][]u8 = undefined;
        var suhs: [3][]u8 = undefined;
        var svhs: [3][]u8 = undefined;
        var shapes: [3]std.json.Array = undefined;
        for (names, 0..) |name, j| {
            blobs[j] = try fx.bytes(try std.fmt.bufPrint(&nb, "R{d}.{s}.trellis", .{ rank, name }));
            suhs[j] = try fx.bytes(try std.fmt.bufPrint(&nb, "R{d}.{s}.suh", .{ rank, name }));
            svhs[j] = try fx.bytes(try std.fmt.bufPrint(&nb, "R{d}.{s}.svh", .{ rank, name }));
            shapes[j] = params.get(try std.fmt.bufPrint(&nb, "R{d}.{s}.shapes", .{ rank, name })).?.array;
        }
        defer for (0..3) |j| {
            gpu.gpa.free(blobs[j]);
            gpu.gpa.free(suhs[j]);
            gpu.gpa.free(svhs[j]);
        };
        const mats = try gpu.gpa.alloc(dsv41.experts.ExpertMats, E);
        defer gpu.gpa.free(mats);
        var offs: [3]usize = .{ 0, 0, 0 };
        for (0..E) |e| {
            var parts: [3]dsv41.experts.Mat = undefined;
            for (0..3) |j| {
                const sh = shapes[j].items[e].array.items;
                const kt: usize = @intCast(sh[0].integer);
                const nt: usize = @intCast(sh[1].integer);
                const w16: usize = @intCast(sh[2].integer);
                const bytes = kt * nt * w16 * 2;
                parts[j] = .{ .trellis = blobs[j][offs[j]..][0..bytes], .kt = kt, .nt = nt, .w16 = w16, .suh = suhs[j][e * kt * 16 * 2 ..][0 .. kt * 16 * 2], .svh = svhs[j][e * nt * 16 * 2 ..][0 .. nt * 16 * 2] };
                offs[j] += bytes;
            }
            mats[e] = .{ .gate = parts[0], .up = parts[1], .down = parts[2] };
        }
        var ex = try dsv41.experts.Experts.init(gpu.gpa, gpu.d, mats, .mul1);
        defer ex.deinit();
        try check.expect(ex.count == E and ex.dims == D and ex.width == I, "rank {d}: E={d} D={d} I={d}, expected {d} {d} {d}", .{ rank, ex.count, ex.dims, ex.width, E, D, I });
        var scratch = try dsv41.experts.Scratch.init(gpu.d, &ex, 64, slots);
        defer scratch.deinit();
        var it = std.mem.splitScalar(u8, rows_text, ',');
        while (it.next()) |r| {
            const R = try std.fmt.parseInt(usize, r, 10);
            const x = try fx.bytes(try std.fmt.bufPrint(&nb, "R{d}.x{d}", .{ rank, R }));
            defer gpu.gpa.free(x);
            const pick = try fx.bytes(try std.fmt.bufPrint(&nb, "R{d}.pick{d}", .{ rank, R }));
            defer gpu.gpa.free(pick);
            const wts = try fx.bytes(try std.fmt.bufPrint(&nb, "R{d}.wts{d}", .{ rank, R }));
            defer gpu.gpa.free(wts);
            const want = try fx.bytes(try std.fmt.bufPrint(&nb, "R{d}.y{d}", .{ rank, R }));
            defer gpu.gpa.free(want);
            var xd = try cuda.DeviceBuffer.fromHost(gpu.d, x);
            defer xd.free();
            var pd = try cuda.DeviceBuffer.fromHost(gpu.d, pick);
            defer pd.free();
            var wd = try cuda.DeviceBuffer.fromHost(gpu.d, wts);
            defer wd.free();
            var out = try cuda.DeviceBuffer.alloc(gpu.d, R * D * 4);
            defer out.free();
            try out.fill8(0, null);
            var what: [128]u8 = undefined;
            const label = try std.fmt.bufPrint(&what, "exl3-experts rank {d} E={d} D={d} I={d} k2_gu={d}..{d} k2_d={d}..{d} rows={d}", .{ rank, E, D, I, ex.k2_gu[0], ex.k2_gu[1], ex.k2_d[0], ex.k2_d[1], R });
            // twice: the second run reads the counters and flags the first left behind, as the serving loop does
            for (0..2) |run| {
                try dsv41.experts.routed(&kernels, &ex, &scratch, xd.ptr, pd.ptr, wd.ptr, R, limit, out.ptr, stream);
                try stream.synchronize();
                const got = try check.download(gpu, out);
                defer gpu.gpa.free(got);
                var lbl: [160]u8 = undefined;
                const run_label = try std.fmt.bufPrint(&lbl, "{s} run {d}", .{ label, run });
                if (check.sameBytes(run_label, got, want)) {
                    check.pass("{s}", .{run_label});
                } else |_| {
                    failures += 1;
                }
            }
        }
    }
    try check.expect(failures == 0, "exl3-experts: {d} cases differ from the Python fork", .{failures});
}
