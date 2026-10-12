//! `dsv41-hc-fused-probe <aot-dir> <fixture-dir>`: hc_fused.cu (the prompt mHC's `_hc_post` and the next
//! `_hc_partial` in one pass) against the fork's unfused pair, byte for byte, at 2048, 777, 64 and 1 rows:
//!  - on the fixture tools/zig/dsv41/dump_hc_mix.py writes (a real layer's ffn mixing weights, seeded streams and
//!    rank partials; the fork's `_hc_post` -> h1 and `_hc_partial` -> part), the same comparison that holds the fork's
//!    fused `_hc_mix_part` to the pair;
//!  - on rows whose values span 2^-12 .. 2^12 (streams, rank partials) with random weights spanning 2^-10 .. 1,
//!    against the captured `_hc_post` + `_hc_partial` run here.
//! The posted streams, the 24 dots and the sum of squares of every (row, block) must be byte-equal. The kernel's
//! VARIANT instances (the post without contraction, the dots' butterfly in the other order, the squares added in
//! column order) run on the same rows and print how many outputs they change: what the comparison resolves.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");

const Gpu = check.Gpu;
const aot = cuda.aot;
const HcFuse = dsv41.prompt.HcFuse;
const Hc = dsv41.hc.Hc;

const S: usize = 4;
const NB: usize = 40;
const synthetic_rows = [_]usize{ 2048, 777, 64, 1 };

const names = [_][:0]const u8{ "tf_dsv41_hc_post_partial", "tf_dsv41_hc_post_partial_r16", "tf_dsv41_hc_post_partial_v1", "tf_dsv41_hc_post_partial_v2", "tf_dsv41_hc_post_partial_v3" };
const per_program = [_]u32{ 8, 16, 8, 8, 8 };
const exact = 2; // names[0..exact] are the kernels that must match; the rest are the sensitivity variants

/// A float of random sign, random 23-bit mantissa and exponent uniform in [lo, hi].
fn spread(r: std.Random, lo: i32, hi: i32) f32 {
    const e: i32 = r.intRangeAtMost(i32, lo, hi);
    const bits: u32 = (@as(u32, @intCast(e + 127)) << 23) | (r.int(u32) & 0x7FFFFF) | (@as(u32, r.int(u1)) << 31);
    return @bitCast(bits);
}

fn bf16(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16); // truncation: the inputs are made as bf16 values
}

/// One row count's inputs on the device.
const Inputs = struct {
    x: cuda.DeviceBuffer, // [rows, 4 D] bf16
    g: cuda.DeviceBuffer, // [2, rows, D] f32
    post: cuda.DeviceBuffer, // [rows, 4] f32
    comb: cuda.DeviceBuffer, // [rows, 4, 4] f32
    fn_: cuda.DeviceBuffer, // [24, 4 D] f32

    fn deinit(self: *Inputs) void {
        inline for (.{ "x", "g", "post", "comb", "fn_" }) |f| @field(self, f).free();
    }
};

/// Every kernel of `names` on `in`, its streams and partial sums against `want_h` / `want_p` (columns 0..24 of each
/// (row, block)); true when the exact kernels match.
fn compare(gpu: Gpu, funs: []const cuda.Function, stream: cuda.Stream, in: Inputs, rows: usize, d: usize, want_h: []const u16, want_p: []const u32, label: []const u8) !bool {
    const gpa = gpu.gpa;
    const wide = S * d;
    var got_h = try cuda.DeviceBuffer.alloc(gpu.d, rows * wide * 2);
    defer got_h.free();
    var got_p = try cuda.DeviceBuffer.alloc(gpu.d, rows * NB * 32 * 4);
    defer got_p.free();
    const oh = try gpa.alloc(u16, rows * wide);
    defer gpa.free(oh);
    const op = try gpa.alloc(u32, rows * NB * 32);
    defer gpa.free(op);
    var ok = true;
    for (names, funs, per_program, 0..) |name, fun, rp, k| {
        try got_p.fill32(0, stream.handle);
        try got_h.fill32(0, stream.handle);
        try HcFuse.postPartial(fun, rp, stream, in.x.ptr, got_h.ptr, in.g.ptr, rows * d, in.post.ptr, in.comb.ptr, in.fn_.ptr, got_p.ptr, rows, d);
        try stream.synchronize();
        try got_h.download(0, std.mem.sliceAsBytes(oh));
        try got_p.download(0, std.mem.sliceAsBytes(op));
        var bad_h: usize = 0;
        var bad_m: usize = 0;
        var bad_s: usize = 0;
        for (want_h, oh) |a, b| bad_h += @intFromBool(a != b);
        for (0..rows * NB) |i| {
            for (0..24) |m| bad_m += @intFromBool(want_p[i * 32 + m] != op[i * 32 + m]);
            bad_s += @intFromBool(want_p[i * 32 + 24] != op[i * 32 + 24]);
        }
        std.debug.print("RESULT dsv41-hc-fused-probe {s} rows {d} {s}: streams {d} of {d} differ, dots {d} of {d}, squares {d} of {d}\n", .{ label, rows, name, bad_h, rows * wide, bad_m, rows * NB * 24, bad_s, rows * NB });
        if (k < exact and bad_h + bad_m + bad_s != 0) ok = false;
    }
    return ok;
}

pub fn probe(gpu: Gpu, aot_dir: []const u8, fixture_dir: []const u8) !void {
    const gpa = gpu.gpa;
    var set = try aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var module = try cuda.Module.load(gpu.d, cuda.kernels.dsv41_hc_fused);
    defer module.unload();
    var funs: [names.len]cuda.Function = undefined;
    for (names, &funs) |n, *f| {
        f.* = try module.function(n);
        try f.allowDynamicShared(HcFuse.smem);
    }
    var ok = true;

    // the fork's fixture
    var fx = try fixture.Fixture.open(gpa, gpu.io, fixture_dir);
    defer fx.deinit();
    const d: usize = @intCast(try fx.int("D"));
    const fn_bytes = try fx.bytes("fn_f");
    defer gpa.free(fn_bytes);
    var nb: [32]u8 = undefined;
    for (0..@intCast(try fx.int("n_count"))) |i| {
        const rows: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "n{d}", .{i})));
        var in: Inputs = undefined;
        inline for (.{ .{ "x", "h0" }, .{ "g", "g" }, .{ "post", "post" }, .{ "comb", "comb" } }) |pair| {
            const b = try fx.bytes(try std.fmt.bufPrint(&nb, "n{d}_" ++ pair[1], .{rows}));
            defer gpa.free(b);
            @field(in, pair[0]) = try cuda.DeviceBuffer.fromHost(gpu.d, b);
        }
        in.fn_ = try cuda.DeviceBuffer.fromHost(gpu.d, fn_bytes);
        defer in.deinit();
        const wh = try fx.bytes(try std.fmt.bufPrint(&nb, "n{d}_h1", .{rows}));
        defer gpa.free(wh);
        const wp = try fx.bytes(try std.fmt.bufPrint(&nb, "n{d}_part", .{rows}));
        defer gpa.free(wp);
        if (!try compare(gpu, &funs, stream, in, rows, d, @alignCast(std.mem.bytesAsSlice(u16, wh)), @alignCast(std.mem.bytesAsSlice(u32, wp)), "fixture")) ok = false;
    }

    // wide-range rows against the captured pair run here
    const hc = Hc.init(&set, @intCast(d), 20, 1e-6, 1e-6);
    var prng = std.Random.DefaultPrng.init(0x4d43);
    const r = prng.random();
    for (synthetic_rows) |rows| {
        const wide = S * d;
        const xh = try gpa.alloc(u16, rows * wide);
        defer gpa.free(xh);
        for (xh) |*v| v.* = bf16(spread(r, -12, 12));
        const gh = try gpa.alloc(f32, 2 * rows * d);
        defer gpa.free(gh);
        for (gh) |*v| v.* = spread(r, -12, 12);
        const ph = try gpa.alloc(f32, rows * S);
        defer gpa.free(ph);
        for (ph) |*v| v.* = @abs(spread(r, -3, 0));
        const ch = try gpa.alloc(f32, rows * S * S);
        defer gpa.free(ch);
        for (ch) |*v| v.* = @abs(spread(r, -6, -1));
        const fh = try gpa.alloc(f32, 24 * wide);
        defer gpa.free(fh);
        for (fh) |*v| v.* = spread(r, -10, 0);
        var in: Inputs = .{
            .x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh)),
            .g = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(gh)),
            .post = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ph)),
            .comb = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ch)),
            .fn_ = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(fh)),
        };
        defer in.deinit();
        var want_h = try cuda.DeviceBuffer.alloc(gpu.d, rows * wide * 2);
        defer want_h.free();
        var want_p = try cuda.DeviceBuffer.alloc(gpu.d, rows * NB * 32 * 4);
        defer want_p.free();
        try want_p.fill32(0, stream.handle);
        try hc.post(stream, in.g.ptr, 2, in.x.ptr, in.post.ptr, in.comb.ptr, want_h.ptr, @intCast(rows));
        try set.run(stream, "_hc_partial", .{ @intCast(rows), @intCast(NB), 1 }, &.{
            aot.ptr("X", "*bf16", want_h.ptr), aot.ptr("FN", "*fp32", in.fn_.ptr), aot.ptr("PART", "*fp32", want_p.ptr),
        }, &.{ aot.ci("WIDE", @intCast(wide)), aot.ci("NB", @intCast(NB)), aot.ci("SUB", 128) });
        try stream.synchronize();
        const wh = try gpa.alloc(u16, rows * wide);
        defer gpa.free(wh);
        try want_h.download(0, std.mem.sliceAsBytes(wh));
        const wp = try gpa.alloc(u32, rows * NB * 32);
        defer gpa.free(wp);
        try want_p.download(0, std.mem.sliceAsBytes(wp));
        if (!try compare(gpu, &funs, stream, in, rows, d, wh, wp, "wide-range")) ok = false;
    }
    try check.expect(ok, "dsv41-hc-fused-probe: hc_fused.cu differs from _hc_post + _hc_partial", .{});
    std.debug.print("PASS BITEXACT dsv41-hc-fused-probe: hc_fused.cu equal to _hc_post + _hc_partial at every row count\n", .{});
}
