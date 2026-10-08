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
