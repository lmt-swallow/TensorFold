//! DeepSeek-V4.1 Triton kernels captured from the Python fork (tools/zig/dsv41/capture.sh) replayed through
//! cuda.aot.Set, the engine's own launcher: the variant picked by constexprs and argument specialization, bytes
//! compared with the fork's outputs. `dir` is capture.sh's work dir (out/fixtures/<kernel>/manifest.json, aot/).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../check.zig");
const Fixture = @import("../fixture.zig").Fixture;
const Gpu = check.Gpu;

fn upload(gpu: Gpu, fx: Fixture, name: []const u8) !cuda.DeviceBuffer {
    const bytes = try fx.bytes(name);
    defer gpu.gpa.free(bytes);
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}

fn expectSame(gpu: Gpu, fx: Fixture, got: cuda.DeviceBuffer, name: []const u8) !void {
    const want_name = try std.fmt.allocPrint(gpu.gpa, "expected_{s}", .{name});
    defer gpu.gpa.free(want_name);
    const want = try fx.bytes(want_name);
    defer gpu.gpa.free(want);
    const bytes = try check.download(gpu, got);
    defer gpu.gpa.free(bytes);
    try check.sameBytes(name, bytes, want);
}

fn openSet(gpu: Gpu, dir: []const u8) !cuda.aot.Set {
    const aot_dir = try std.fs.path.join(gpu.gpa, &.{ dir, "aot" });
    defer gpu.gpa.free(aot_dir);
    return cuda.aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
}

/// The fixture dir string must outlive the Fixture (it keeps the slice); the caller frees it after deinit.
fn fixtureDir(gpu: Gpu, dir: []const u8, kernel: []const u8) ![]const u8 {
    return std.fs.path.join(gpu.gpa, &.{ dir, "out", "fixtures", kernel });
}

/// `_rmsnorm` as kernels.py's `rmsnorm` launches it: grid (rows,), D and BLOCK constexprs, PDL on.
pub fn rmsnorm(gpu: Gpu, dir: []const u8) !void {
    var set = try openSet(gpu, dir);
    defer set.deinit();
    const fx_dir = try fixtureDir(gpu, dir, "rmsnorm");
    defer gpu.gpa.free(fx_dir);
    var fx = try Fixture.open(gpu.gpa, gpu.io, fx_dir);
    defer fx.deinit();
    const rows: u32 = @intCast(try fx.int("rows"));
    const d = try fx.int("D");
    const block = try fx.int("BLOCK");
    const eps: f32 = @floatCast(try fx.float("eps"));
    var x = try upload(gpu, fx, "x");
    defer x.free();
    var w = try upload(gpu, fx, "w");
    defer w.free();
    var out = try upload(gpu, fx, "out");
    defer out.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const a = cuda.aot;
    try set.run(stream, "_rmsnorm", .{ rows, 1, 1 }, &.{
        a.ptr("X", "*bf16", x.ptr), a.int("xs", @intCast(d)), a.ptr("W", "*bf16", w.ptr),
        a.ptr("OUT", "*bf16", out.ptr), a.int("os_", @intCast(d)), a.float("eps", eps),
    }, &.{ a.ci("D", d), a.ci("BLOCK", block), a.ci("PDL", 1) });
    try stream.synchronize();
    try expectSame(gpu, fx, out, "out");
    check.pass("BITEXACT aot _rmsnorm: {d} rows x {d}, variant picked by aot.Set, output equal to the Python fork's", .{ rows, d });
}

/// `_kv_norm_rope` (RMSNorm, RoPE on the last RD dims, FP8 QDQ per 32, the window ring row): the fork's launch.
pub fn kvNormRope(gpu: Gpu, dir: []const u8) !void {
    var set = try openSet(gpu, dir);
    defer set.deinit();
    const fx_dir = try fixtureDir(gpu, dir, "kv_norm_rope");
    defer gpu.gpa.free(fx_dir);
    var fx = try Fixture.open(gpu.gpa, gpu.io, fx_dir);
    defer fx.deinit();
    const rows: u32 = @intCast(try fx.int("rows"));
    const d = try fx.int("D");
    const rd = try fx.int("RD");
    const ring_size: i32 = @intCast(try fx.int("ring_size"));
    const eps: f32 = @floatCast(try fx.float("eps"));
    const names = [_][]const u8{ "y", "w", "cos", "sin", "pos", "out", "ring", "slots" };
    var bufs: [names.len]cuda.DeviceBuffer = undefined;
    var n: usize = 0;
    defer for (bufs[0..n]) |*b| b.free();
    for (names, &bufs) |name, *b| {
        b.* = try upload(gpu, fx, name);
        n += 1;
    }
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const a = cuda.aot;
    try set.run(stream, "_kv_norm_rope", .{ rows, 1, 1 }, &.{
        a.ptr("Y", "*bf16", bufs[0].ptr),  a.ptr("W", "*bf16", bufs[1].ptr),    a.ptr("COS", "*fp32", bufs[2].ptr),
        a.ptr("SIN", "*fp32", bufs[3].ptr), a.ptr("POS", "*i64", bufs[4].ptr),   a.ptr("OUT", "*bf16", bufs[5].ptr),
        a.ptr("RING", "*bf16", bufs[6].ptr), a.ptr("SLOT_OF", "*i64", bufs[7].ptr), a.int("ring_size", ring_size),
        a.float("eps", eps),
    }, &.{ a.ci("QUANT", 1), a.ci("D", d), a.ci("RD", rd), a.ci("PDL", 1) });
    try stream.synchronize();
    try expectSame(gpu, fx, bufs[5], "out");
    try expectSame(gpu, fx, bufs[6], "ring");
    check.pass("BITEXACT aot _kv_norm_rope: {d} rows x {d} (rope {d}, fp8 qdq), out and ring equal to the Python fork's", .{ rows, d, rd });
}
