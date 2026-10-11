//! `qwen38fn-prompt-time <kernel-set> <model-dir> [depths] [chunks]`: the served prompt path's speed (cuda_mtp.zig
//! `prefillChunk` with the MTP head, 2,048-row chunks) at chosen depths, without a fixture. Each depth (comma-separated
//! positions, default 0,65536,204800) starts a reset stream whose position and head length are set to the depth over
//! zeroed caches, runs one warm chunk, then `chunks` (default 3) timed chunks of a synthetic prompt. The bits are not
//! checked (the zeroed caches stand in for a real prefix: same launches and shapes, different values). A 2 s pause
//! separates the depths so a profiler's timeline splits at them (nsys: tools/zig/qwen38fn/prompt_profile.py).
//! TF_QFN_PROMPT_ROWS sets the chunk rows (default 2048). TF_QFN_COMPARE_CHUNK=<rows> instead prefills one
//! 25,093-token prompt from an empty stream in 2,048-row chunks and in chunks of <rows>, and checks that the last
//! row's logits and streams and every state the prompt leaves (DeltaNet states and conv windows, attention and
//! indexer caches, the MTP head's cache, the n-gram tail and history) are byte-equal (PASS/FAIL).
const std = @import("std");
const cuda = @import("cuda");
const qfn = @import("qwen38fn");
const check = @import("check.zig");

const Gpu = check.Gpu;
const F = qfn.forward;
const P = qfn.prompt;
const M = qfn.mtp;

pub fn run(gpu: Gpu, kernels: []const u8, model: []const u8, depths_arg: ?[]const u8, chunks: usize) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: qfn.config.Why = .{};
    const cfg = try qfn.config.read(gpa, gpu.io, model, &why);
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, kernels);
    defer set.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const t0 = check.now(gpu.io);
    const m = try F.Model.load(gpa, gpu.io, d, model, cfg, cfg.layers, true);
    defer m.deinit();
    const head = try M.Head.load(gpa, d, m, 1, stream);
    defer head.deinit();
    std.debug.print("qwen38fn-prompt-time: {d} layers and the MTP head loaded in {d:.1} s\n", .{ cfg.layers, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9 });

    const capacity = cfg.max_positions;
    if (std.c.getenv("TF_QFN_COMPARE_CHUNK")) |v| {
        var fw0 = F.Forward.init(d, m, &set);
        var pr0 = try P.Prompt.init(&fw0, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
        defer pr0.deinit();
        return compareChunks(gpu, m, head, &pr0, try std.fmt.parseInt(usize, std.mem.span(v), 10), stream);
    }
    const rows = try P.chunkRows();
    var pb = try P.Buffers.init(gpa, d, m, rows, capacity);
    defer pb.deinit();
    var scp = try M.Scratch.init(d, &m.cfg, rows);
    defer scp.deinit();
    var fw = F.Forward.init(d, m, &set);
    var pr = try P.Prompt.init(&fw, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer pr.deinit();
    var st = try F.Stream.init(gpa, d, m, capacity, 16);
    defer st.deinit();
    var last = try cuda.DeviceBuffer.alloc(d, cfg.wide() * 2);
    defer last.free();

    // a fixed pseudo-random prompt (+1 id: the head reads each chunk's next ids)
    const ids = try gpa.alloc(i32, (chunks + 1) * rows + 1);
    defer gpa.free(ids);
    var rng = std.Random.DefaultPrng.init(0x51f15e);
    for (ids) |*x| x.* = rng.random().intRangeLessThan(i32, 1000, 150000);

    var it = std.mem.tokenizeScalar(u8, depths_arg orelse "0,65536,204800", ',');
    while (it.next()) |text| {
        const depth = try std.fmt.parseInt(usize, text, 10);
        if (depth + (chunks + 1) * rows > capacity) return error.DepthPastCapacity;
        try st.reset(m, stream);
        st.pos = depth;
        const p: i32 = @intCast(depth);
        try st.pos_dev.uploadAsync(0, std.mem.asBytes(&p), stream.handle);
        try st.setMtpLen(depth, stream);
        try stream.synchronize();
        std.Io.sleep(gpu.io, .fromMilliseconds(2000), .awake) catch {};
        var start: usize = 0;
        var total_ns: i96 = 0;
        for (0..chunks + 1) |c| {
            const tc = check.now(gpu.io);
            start = try M.prefillChunk(head, &pr, &st, &pb, &scp, ids, start, last, stream);
            try stream.synchronize();
            const ns = check.now(gpu.io) - tc;
            if (c > 0) total_ns += ns;
            std.debug.print("depth {d} chunk {d}{s}: {d} rows from {d} in {d:.1} ms ({d:.0} tok/s)\n", .{ depth, c, if (c == 0) " (warm)" else "", rows, st.pos - rows, @as(f64, @floatFromInt(ns)) / 1e6, @as(f64, @floatFromInt(rows)) * 1e9 / @as(f64, @floatFromInt(ns)) });
        }
        const ms = @as(f64, @floatFromInt(total_ns)) / 1e6 / @as(f64, @floatFromInt(chunks));
        check.pass("qwen38fn-prompt-time depth {d}: {d:.1} ms a {d}-row chunk, {d:.0} tok/s ({d} chunks)", .{ depth, ms, rows, @as(f64, @floatFromInt(rows)) * 1e3 / ms, chunks });
    }
}

/// sha256 of `len` bytes at `ptr`.
fn digest(gpu: Gpu, ptr: cuda.abi.DevicePtr, len: usize) ![32]u8 {
    const host = try gpu.gpa.alloc(u8, len);
    defer gpu.gpa.free(host);
    const buf: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = len };
    try buf.download(0, host);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(host, &out, .{});
    return out;
}

const Part = struct { name: []const u8, ptr: cuda.abi.DevicePtr, len: usize };

/// The state a prompt leaves, as (name, bytes) parts.
fn stateParts(gpa: std.mem.Allocator, m: *const F.Model, st: *const F.Stream, pb: *const P.Buffers, last: cuda.DeviceBuffer) ![]Part {
    const cfg = &m.cfg;
    var parts: std.ArrayList(Part) = .empty;
    try parts.append(gpa, .{ .name = "logits", .ptr = pb.logits.ptr, .len = cfg.vocab * 2 });
    try parts.append(gpa, .{ .name = "last streams", .ptr = last.ptr, .len = cfg.wide() * 2 });
    const lin = m.linearLayers();
    const sb = cfg.nv * cfg.dv * cfg.dk * 4;
    for (0..lin) |li| try parts.append(gpa, .{ .name = "DeltaNet state", .ptr = st.recPtr(cfg, lin, li, st.cur[li]), .len = sb });
    try parts.append(gpa, .{ .name = "conv windows", .ptr = st.conv.ptr, .len = st.conv.len });
    const kv = st.pos * cfg.kv_heads * cfg.head_dim * 2;
    for (st.caches) |*c| {
        try parts.append(gpa, .{ .name = "keys", .ptr = c.k.ptr, .len = kv });
        try parts.append(gpa, .{ .name = "values", .ptr = c.v.ptr, .len = kv });
        try parts.append(gpa, .{ .name = "indexer keys", .ptr = c.ik.ptr, .len = st.pos * cfg.index_dim * 2 });
        try parts.append(gpa, .{ .name = "pooled", .ptr = c.pooled.ptr, .len = st.pos / cfg.index_ratio * cfg.index_dim * 2 });
    }
    if (st.mtp_caches) |*c| {
        const n = st.mtp_len * cfg.kv_heads * cfg.head_dim * 2;
        try parts.append(gpa, .{ .name = "head keys", .ptr = c.k.ptr, .len = n });
        try parts.append(gpa, .{ .name = "head values", .ptr = c.v.ptr, .len = n });
    }
    try parts.append(gpa, .{ .name = "n-gram tail", .ptr = st.ple_tail.ptr, .len = st.ple_tail.len });
    return parts.toOwnedSlice(gpa);
}

fn compareChunks(gpu: Gpu, m: *F.Model, head: *M.Head, pr: *P.Prompt, rows_b: usize, stream: cuda.Stream) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const cfg = &m.cfg;
    const capacity = cfg.max_positions;
    const n = 3 * 8192 + 517;
    const ids = try gpa.alloc(i32, n);
    defer gpa.free(ids);
    var rng = std.Random.DefaultPrng.init(0xc0ffee);
    for (ids) |*x| x.* = rng.random().intRangeLessThan(i32, 1000, 150000);
    var st = try F.Stream.init(gpa, d, m, capacity, 16);
    defer st.deinit();
    var last = try cuda.DeviceBuffer.alloc(d, cfg.wide() * 2);
    defer last.free();
    var sums: [2][][32]u8 = undefined;
    var names: []Part = &.{};
    defer gpa.free(names);
    var histories: [2][16]i64 = undefined;
    for ([_]usize{ P.chunk_rows, rows_b }, 0..) |rows, pass| {
        var pb = try P.Buffers.init(gpa, d, m, rows, capacity);
        defer pb.deinit();
        var scp = try M.Scratch.init(d, cfg, rows);
        defer scp.deinit();
        try st.reset(m, stream);
        const tc = check.now(gpu.io);
        var start: usize = 0;
        var chunks: usize = 0;
        while (start < n) : (chunks += 1) start = try M.prefillChunk(head, pr, &st, &pb, &scp, ids, start, last, stream);
        try stream.synchronize();
        std.debug.print("compare: {d} tokens in {d} chunks of {d} rows: {d:.0} ms\n", .{ n, chunks, rows, @as(f64, @floatFromInt(check.now(gpu.io) - tc)) / 1e6 });
        const parts = try stateParts(gpa, m, &st, &pb, last);
        sums[pass] = try gpa.alloc([32]u8, parts.len);
        for (parts, sums[pass]) |pt, *h| h.* = try digest(gpu, pt.ptr, pt.len);
        if (pass == 0) names = parts else gpa.free(parts);
        histories[pass] = @splat(0);
        @memcpy(histories[pass][0..st.history.len], st.history[0..]);
    }
    defer for (sums) |x| gpa.free(x);
    var bad: usize = 0;
    for (names, sums[0], sums[1]) |pt, a, b| if (!std.mem.eql(u8, &a, &b)) {
        std.debug.print("FAIL qwen38fn chunk compare: {s} differs\n", .{pt.name});
        bad += 1;
    };
    if (!std.mem.eql(i64, &histories[0], &histories[1])) bad += 1;
    try check.expect(bad == 0, "qwen38fn chunk compare: {d} of {d} parts differ between {d}- and {d}-row chunks", .{ bad, names.len + 1, P.chunk_rows, rows_b });
    check.pass("BITEXACT qwen38fn chunk compare: {d} tokens in {d}-row and {d}-row chunks leave the same logits, streams and states ({d} parts)", .{ n, P.chunk_rows, rows_b, names.len + 1 });
}
