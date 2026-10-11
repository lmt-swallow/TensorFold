//! The rounds' RDMA ring (deepseek_v41/rdma.zig) across real ranks: every gather's output bytes against NCCL's
//! all-gather of the same slices and against what each rank wrote (random bit patterns, NaNs and all), eagerly and
//! replayed from a CUDA graph, then each path's time a gather.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");

const rdma = dsv41.rdma;
const Gpu = check.Gpu;

/// Words a rank: one row of the hidden size up to the ring's slot (4 MiB of a 4.25 MiB slot).
const sizes = [_]usize{ 4, 4096, 4 * 4096, 16 * 4096, 32 * 4096, 256 * 1024, 1024 * 1024 };
const slot_bytes: usize = 4456448;
const checks = 64; // gathers compared a size, each with new bits
const reps = 200; // timed gathers a size and path
const graph_gathers = 50; // gathers a recorded graph

pub fn rdmaRing(gpu: Gpu, rank: u32, world: u32, master: []const u8, port: u16, list: []const u8, gid: u8) !void {
    const d = gpu.d;
    const gpa = gpu.gpa;
    var ip: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, master, '.');
    for (&ip) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    var comm = try cuda.comm.Comm.init(rank, world, ip, port, 120_000);
    defer comm.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    var module = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
    defer module.unload();
    var names: [rdma.max_devices][]const u8 = undefined;
    const ring = try rdma.openOver(gpa, d, try rdma.Kernels.load(module), try rdma.deviceList(list, &names), &comm, stream, .{ .max_bytes = slot_bytes, .gid_index = gid });
    defer ring.destroy(gpa);

    const most = sizes[sizes.len - 1];
    var send = try cuda.DeviceBuffer.alloc(d, most * 4);
    defer send.free();
    var by_ring = try cuda.DeviceBuffer.alloc(d, most * 4 * world);
    defer by_ring.free();
    var by_nccl = try cuda.DeviceBuffer.alloc(d, most * 4 * world);
    defer by_nccl.free();
    const host_send = try gpa.alloc(u32, most);
    defer gpa.free(host_send);
    const want = try gpa.alloc(u32, most * world);
    defer gpa.free(want);
    const got_ring = try gpa.alloc(u32, most * world);
    defer gpa.free(got_ring);
    const got_nccl = try gpa.alloc(u32, most * world);
    defer gpa.free(got_nccl);

    var failures: usize = 0;
    for (sizes) |words| {
        const n = words * world;
        // eager: each gather with new bits, both paths, against each other and every rank's bits
        var differ: usize = 0;
        for (0..checks) |c| {
            for (0..world) |p| fill(want[p * words ..][0..words], words, p, c);
            @memcpy(host_send[0..words], want[rank * words ..][0..words]);
            try send.upload(0, std.mem.sliceAsBytes(host_send[0..words]));
            try ring.gather(stream, send.ptr, @intCast(words / 4), by_ring.ptr);
            try comm.allGather(send.ptr, by_nccl.ptr, words, .f32, stream.handle);
            try stream.synchronize();
            try by_ring.download(0, std.mem.sliceAsBytes(got_ring[0..n]));
            try by_nccl.download(0, std.mem.sliceAsBytes(got_nccl[0..n]));
            if (!std.mem.eql(u32, got_ring[0..n], got_nccl[0..n]) or !std.mem.eql(u32, got_ring[0..n], want[0..n])) differ += 1;
        }
        // a graph of back-to-back ring gathers, replayed: every replay's last output is this round's bits
        var g_differ: usize = 0;
        var exec = try record(stream, ring, &comm, send.ptr, by_ring.ptr, words, true);
        defer exec.deinit();
        for (0..4) |c| {
            for (0..world) |p| fill(want[p * words ..][0..words], words, p, checks + c);
            @memcpy(host_send[0..words], want[rank * words ..][0..words]);
            try send.upload(0, std.mem.sliceAsBytes(host_send[0..words]));
            try by_ring.fill8(0, stream.handle);
            try exec.launchOn(stream);
            try stream.synchronize();
            try by_ring.download(0, std.mem.sliceAsBytes(got_ring[0..n]));
            if (!std.mem.eql(u32, got_ring[0..n], want[0..n])) g_differ += 1;
        }
        failures += differ + g_differ;

        // times: one gather and a host sync (p50, p90), and a graph's gathers back to back (mean)
        const ring_us = try timeEager(stream, ring, &comm, send.ptr, by_ring.ptr, words, true);
        const nccl_us = try timeEager(stream, ring, &comm, send.ptr, by_nccl.ptr, words, false);
        var nccl_exec = try record(stream, ring, &comm, send.ptr, by_nccl.ptr, words, false);
        defer nccl_exec.deinit();
        const ring_graph = try timeGraph(stream, exec);
        const nccl_graph = try timeGraph(stream, nccl_exec);
        std.debug.print("RESULT dsv41-rdma rank {d}: {d} words a rank ({d} KiB): ring {s} ({d} eager, {d} graph replays differ); eager p50 ring {d:.1} us vs nccl {d:.1} us (p90 {d:.1} vs {d:.1}); in a graph ring {d:.1} us vs nccl {d:.1} us a gather\n", .{ rank, words, words * 4 / 1024, if (differ + g_differ == 0) "equal to nccl and to every rank's bits" else "DIFFERS", differ, g_differ, ring_us[0], nccl_us[0], ring_us[1], nccl_us[1], ring_graph, nccl_graph });
    }
    if (ring.failure()) |why| {
        std.debug.print("RESULT dsv41-rdma rank {d}: the ring failed: {s}\n", .{ rank, why });
        return error.RingFailed;
    }
    if (failures > 0) return error.Differs;
    std.debug.print("RESULT dsv41-rdma rank {d}: PASS ({s})\n", .{ rank, list });
}

/// Rank `p`'s words for check `c`: every u32 bit pattern may appear (NaNs, infinities, subnormals).
fn fill(out: []u32, words: usize, p: usize, c: usize) void {
    var prng = std.Random.DefaultPrng.init(@as(u64, words) *% 0x9E37_79B9 +% @as(u64, p) *% 0x1000_0001 +% c);
    prng.random().bytes(std.mem.sliceAsBytes(out));
}

fn gatherOnce(stream: cuda.Stream, ring: *rdma.Ring, comm: *const cuda.comm.Comm, send: cuda.abi.DevicePtr, out: cuda.abi.DevicePtr, words: usize, over_ring: bool) !void {
    if (over_ring) return ring.gather(stream, send, @intCast(words / 4), out);
    try comm.allGather(send, out, words, .f32, stream.handle);
}

fn record(stream: cuda.Stream, ring: *rdma.Ring, comm: *const cuda.comm.Comm, send: cuda.abi.DevicePtr, out: cuda.abi.DevicePtr, words: usize, over_ring: bool) !cuda.graph.Exec {
    try cuda.graph.beginCapture(stream, .thread_local);
    for (0..graph_gathers) |_| gatherOnce(stream, ring, comm, send, out, words, over_ring) catch |err| {
        if (cuda.graph.endCapture(stream)) |gr| {
            var x = gr;
            x.deinit();
        } else |_| {}
        return err;
    };
    var gr = try cuda.graph.endCapture(stream);
    defer gr.deinit();
    var exec = try gr.instantiate();
    errdefer exec.deinit();
    try exec.upload(stream);
    return exec;
}

fn timeEager(stream: cuda.Stream, ring: *rdma.Ring, comm: *const cuda.comm.Comm, send: cuda.abi.DevicePtr, out: cuda.abi.DevicePtr, words: usize, over_ring: bool) ![2]f64 {
    for (0..10) |_| try gatherOnce(stream, ring, comm, send, out, words, over_ring);
    try stream.synchronize();
    var us: [reps]f64 = undefined;
    for (&us) |*u| {
        const t0 = nowNs();
        try gatherOnce(stream, ring, comm, send, out, words, over_ring);
        try stream.synchronize();
        u.* = @as(f64, @floatFromInt(nowNs() - t0)) / 1e3;
    }
    std.mem.sort(f64, &us, {}, std.sort.asc(f64));
    return .{ us[reps / 2], us[reps * 9 / 10] };
}

fn timeGraph(stream: cuda.Stream, exec: cuda.graph.Exec) !f64 {
    try exec.launchOn(stream);
    try stream.synchronize();
    const runs = 10;
    const t0 = nowNs();
    for (0..runs) |_| try exec.launchOn(stream);
    try stream.synchronize();
    return @as(f64, @floatFromInt(nowNs() - t0)) / 1e3 / (runs * graph_gathers);
}

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}
