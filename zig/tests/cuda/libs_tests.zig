//! cuBLASLt and NCCL through dlopen: one bf16 GEMM against a host reference, one-rank collectives.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

/// Round-to-nearest-even fp32 to bf16 bits (inputs here are finite).
fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn cublaslt(gpu: Gpu) !void {
    const d = gpu.d;
    var lt = try cuda.cublaslt.Library.open();
    defer lt.close();
    const m: usize = 16;
    const n: usize = 256;
    const k: usize = 512;
    var rng = std.Random.DefaultPrng.init(1234);
    const r = rng.random();
    const x = try gpu.gpa.alloc(u16, m * k);
    defer gpu.gpa.free(x);
    const w = try gpu.gpa.alloc(u16, n * k);
    defer gpu.gpa.free(w);
    for (x) |*v| v.* = bf16(r.floatNorm(f32));
    for (w) |*v| v.* = bf16(r.floatNorm(f32) * 0.05);

    var xd = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x));
    defer xd.free();
    var wd = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(w));
    defer wd.free();
    var out = try cuda.DeviceBuffer.alloc(d, m * n * 4);
    defer out.free();
    const workspace_len: usize = 32 << 20;
    var ws = try cuda.DeviceBuffer.alloc(d, workspace_len);
    defer ws.free();
    var linear = try cuda.cublaslt.Linear.init(&lt, m, n, k, .f32, workspace_len);
    defer linear.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try linear.run(xd.ptr, wd.ptr, out.ptr, ws.ptr, workspace_len, stream.handle);
    try stream.synchronize();
    const first = try check.download(gpu, out);
    defer gpu.gpa.free(first);
    try out.fill8(0, null);
    try linear.run(xd.ptr, wd.ptr, out.ptr, ws.ptr, workspace_len, stream.handle);
    try stream.synchronize();
    const second = try check.download(gpu, out);
    defer gpu.gpa.free(second);
    try check.sameBytes("cuBLASLt rerun", second, first);

    var worst: f64 = 0;
    const got = std.mem.bytesAsSlice(f32, first);
    for (0..m) |i| for (0..n) |j| {
        var acc: f64 = 0;
        for (0..k) |t| acc += @as(f64, widen(x[i * k + t])) * @as(f64, widen(w[j * k + t]));
        worst = @max(worst, @abs(@as(f64, got[i * n + j]) - acc));
    };
    try expect(worst < 1e-3, "cuBLASLt max abs error {e} vs fp64 reference", .{worst});
    check.pass("cuBLASLt {d}: bf16 x bf16 -> fp32 ({d}x{d}x{d}), reruns bit-equal, max abs error {e} vs fp64, workspace {d} B", .{ lt.api.cublasLtGetVersion(), m, n, k, worst, linear.workspace_need });
}

pub fn nccl(gpu: Gpu) !void {
    const d = gpu.d;
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const version = try lib.version();
    var comms: [1]cuda.nccl.Comm = .{null};
    const devices = [1]c_int{gpu.ctx.device};
    try lib.check(lib.api.ncclCommInitAll(&comms, 1, &devices), "ncclCommInitAll");
    defer _ = lib.api.ncclCommDestroy(comms[0]);
    const count: usize = 4096;
    const src = try gpu.gpa.alloc(f32, count);
    defer gpu.gpa.free(src);
    for (src, 0..) |*v, i| v.* = @floatFromInt(i);
    var send = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(src));
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, count * 4);
    defer recv.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try lib.check(lib.api.ncclAllReduce(send.ptr, recv.ptr, count, .f32, .sum, comms[0], stream.handle), "ncclAllReduce");
    try stream.synchronize();
    const got = try check.download(gpu, recv);
    defer gpu.gpa.free(got);
    try check.sameBytes("one-rank all-reduce", got, std.mem.sliceAsBytes(src));
    try recv.fill8(0, null);
    try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, count, .f32, comms[0], stream.handle), "ncclAllGather");
    try stream.synchronize();
    try recv.download(0, got);
    try check.sameBytes("one-rank all-gather", got, std.mem.sliceAsBytes(src));
    check.pass("NCCL {d}: one-rank communicator, all-reduce and all-gather exact", .{version});
}

/// Two ranks on two machines: the unique id over TCP, one f32 all-gather in rank order, then a control-link
/// round trip. `master` is rank 0's IPv4; both ranks print the gathered values.
pub fn comm(gpu: Gpu, rank: u32, world: u32, master: []const u8, port: u16) !void {
    const d = gpu.d;
    var ip: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, master, '.');
    for (&ip) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    var c = try cuda.comm.Comm.init(rank, world, ip, port, 120_000);
    defer c.deinit();
    const count: usize = 4;
    var host: [4]f32 = undefined;
    for (&host, 0..) |*v, i| v.* = @as(f32, @floatFromInt(rank * 100 + i));
    var send = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&host));
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, count * 4 * world);
    defer recv.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try c.allGather(send.ptr, recv.ptr, count, .f32, stream.handle);
    try stream.synchronize();
    const got = try check.download(gpu, recv);
    defer gpu.gpa.free(got);
    const vals = std.mem.bytesAsSlice(f32, got);
    for (vals, 0..) |v, i| {
        const want: f32 = @floatFromInt((i / count) * 100 + i % count);
        if (v != want) {
            std.debug.print("rank {d}: gathered[{d}] = {d}, want {d}\n", .{ rank, i, v, want });
            return error.Mismatch;
        }
    }
    var buf: [64]u8 = undefined;
    if (rank == 0) {
        for (1..world) |peer| {
            try c.sendTo(@intCast(peer), "ping");
            const reply = try c.recvFrom(@intCast(peer), &buf);
            if (!std.mem.eql(u8, reply, "pong")) return error.Mismatch;
        }
    } else {
        const msg = try c.recvFrom(0, &buf);
        if (!std.mem.eql(u8, msg, "ping")) return error.Mismatch;
        try c.sendTo(0, "pong");
    }
    std.debug.print("RESULT comm rank {d} of {d}: all-gather of {d} f32 in rank order and the control link round trip\n", .{ rank, world, count * world });
}
