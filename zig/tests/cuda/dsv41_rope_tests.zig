//! `dsv41-rope <rope-dir> [rows ...]`: the DSV4.1 RoPE rows computed on the device (rope.cu, cuda_attention.zig
//! RopeGen) against the fork's tables (tools/zig/dsv41/dump_rope.py: `rope_c_<rows>.bin`, `rope_w_<rows>.bin`, the
//! frequencies in `rope_c_freqs.bin`, `rope_w_freqs.bin`): every row of each table byte for byte, and rows gathered
//! at scattered positions through a position array (the device path's per-pass tables). Rows default to the tables
//! of 20480, 790528 and 1052672 rows the directory holds.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const Gpu = check.Gpu;
const A = dsv41.attention;

const half: usize = 32; // rope_dim / 2

fn readFile(gpu: Gpu, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 31));
}

fn ulps(a: u32, b: u32) u64 {
    const key = struct {
        fn f(u: u32) i64 {
            const s: i32 = @bitCast(u);
            return if (s < 0) @as(i64, std.math.minInt(i32)) - s else s;
        }
    }.f;
    return @abs(key(a) - key(b));
}

/// The first differing f32 of got against want as (row, column, ulps), or null.
fn firstDiff(got: []const u32, want: []const u32) ?struct { row: usize, col: usize, ulp: u64, n: usize } {
    var first: ?usize = null;
    var n: usize = 0;
    for (got, want, 0..) |g, w, i| if (g != w) {
        if (first == null) first = i;
        n += 1;
    };
    const i = first orelse return null;
    return .{ .row = i / half, .col = i % half, .ulp = ulps(got[i], want[i]), .n = n };
}

pub fn rope(gpu: Gpu, dir: []const u8, rows_arg: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    var gen = try A.RopeGen.load(gpu.d);
    defer gen.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var start_ev = try cuda.Event.init(gpu.d, true);
    defer start_ev.deinit();
    var end_ev = try cuda.Event.init(gpu.d, true);
    defer end_ev.deinit();

    var row_counts: std.ArrayList(usize) = .empty;
    defer row_counts.deinit(gpa);
    if (rows_arg.len > 0) {
        for (rows_arg) |r| try row_counts.append(gpa, try std.fmt.parseInt(usize, r, 10));
    } else for ([_]usize{ 20480, 790528, 1052672 }) |r| {
        const p = try std.fmt.allocPrint(gpa, "{s}/rope_c_{d}.bin", .{ dir, r });
        defer gpa.free(p);
        std.Io.Dir.cwd().access(gpu.io, p, .{}) catch continue;
        try row_counts.append(gpa, r);
    }
    try check.expect(row_counts.items.len > 0, "no rope_c_<rows>.bin in {s}", .{dir});

    var failures: usize = 0;
    var tables: usize = 0;
    for ([_][]const u8{ "rope_c", "rope_w" }) |kind| {
        const fpath = try std.fmt.allocPrint(gpa, "{s}/{s}_freqs.bin", .{ dir, kind });
        defer gpa.free(fpath);
        const fbytes = try readFile(gpu, fpath);
        defer gpa.free(fbytes);
        try check.expect(fbytes.len == half * 4, "{s}: {d} bytes, want {d}", .{ fpath, fbytes.len, half * 4 });
        var freqs = try cuda.DeviceBuffer.fromHost(gpu.d, fbytes);
        defer freqs.free();

        for (row_counts.items) |rows| {
            const tpath = try std.fmt.allocPrint(gpa, "{s}/{s}_{d}.bin", .{ dir, kind, rows });
            defer gpa.free(tpath);
            const want = try readFile(gpu, tpath);
            defer gpa.free(want);
            try check.expect(want.len == 2 * rows * half * 4, "{s}: {d} bytes, want {d}", .{ tpath, want.len, 2 * rows * half * 4 });
            const plane = rows * half * 4;
            var out = try cuda.DeviceBuffer.alloc(gpu.d, 2 * plane);
            defer out.free();

            // every row, positions 0 .. rows - 1
            try start_ev.record(stream);
            try gen.rows(0, 0, rows, freqs.ptr, half, out.ptr, out.ptr + plane, stream);
            try end_ev.record(stream);
            try stream.synchronize();
            const ms = try cuda.Event.elapsedMs(start_ev, end_ev);
            const got = try check.download(gpu, out);
            defer gpa.free(got);
            tables += 1;
            const before = failures;
            for ([_][]const u8{ "cos", "sin" }, 0..) |part, pi| {
                const g = std.mem.bytesAsSlice(u32, got[pi * plane ..][0..plane]);
                const w = std.mem.bytesAsSlice(u32, want[pi * plane ..][0..plane]);
                if (firstDiff(@alignCast(g), @alignCast(w))) |dd| {
                    std.debug.print("FAIL {s}_{d} {s}: {d} values differ, first at row {d} column {d} ({d} ulp)\n", .{ kind, rows, part, dd.n, dd.row, dd.col, dd.ulp });
                    failures += 1;
                }
            }
            if (failures == before) check.pass("{s}_{d}: {d} rows x {d} (cos, sin) bit-exact, {d:.2} ms on the device", .{ kind, rows, rows, half, ms });

            // rows at scattered positions through a position array (repeats, both ends, descending runs)
            const n: usize = 4099;
            const pos = try gpa.alloc(i64, n);
            defer gpa.free(pos);
            var prng = std.Random.DefaultPrng.init(20261011 + rows);
            const rnd = prng.random();
            for (pos, 0..) |*p, i| p.* = switch (i % 7) {
                0 => 0,
                1 => @intCast(rows - 1),
                2 => @intCast(rows - 1 - (i % rows)),
                else => @intCast(rnd.uintLessThan(usize, rows)),
            };
            var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos));
            defer dpos.free();
            var small = try cuda.DeviceBuffer.alloc(gpu.d, 2 * n * half * 4);
            defer small.free();
            try gen.rows(dpos.ptr, 0, n, freqs.ptr, half, small.ptr, small.ptr + n * half * 4, stream);
            try stream.synchronize();
            const sg = try check.download(gpu, small);
            defer gpa.free(sg);
            var bad: usize = 0;
            for ([_]usize{ 0, 1 }) |pi| for (pos, 0..) |p, r| {
                const a = sg[pi * n * half * 4 + r * half * 4 ..][0 .. half * 4];
                const b = want[pi * plane + @as(usize, @intCast(p)) * half * 4 ..][0 .. half * 4];
                if (!std.mem.eql(u8, a, b)) bad += 1;
            };
            if (bad > 0) {
                std.debug.print("FAIL {s}_{d}: {d} of {d} gathered rows differ\n", .{ kind, rows, bad, 2 * n });
                failures += 1;
            } else check.pass("{s}_{d}: {d} rows at scattered positions bit-exact", .{ kind, rows, n });
        }
    }
    try check.expect(failures == 0, "dsv41-rope: {d} failures", .{failures});
    check.pass("dsv41-rope: {d} tables", .{tables});
}
