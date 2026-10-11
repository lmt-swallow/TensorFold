//! `qwen38fn-prompt-time <kernel-set> <model-dir> [depths] [chunks]`: the served prompt path's speed (cuda_mtp.zig
//! `prefillChunk` with the MTP head, 2,048-row chunks) at chosen depths, without a fixture. Each depth (comma-separated
//! positions, default 0,65536,204800) starts a reset stream whose position and head length are set to the depth over
//! zeroed caches, runs one warm chunk, then `chunks` (default 3) timed chunks of a synthetic prompt. The bits are not
//! checked (the zeroed caches stand in for a real prefix: same launches and shapes, different values). A 2 s pause
//! separates the depths so a profiler's timeline splits at them (nsys: tools/zig/qwen38fn/prompt_profile.py).
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

    const rows = P.chunk_rows;
    const capacity = cfg.max_positions;
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
        check.pass("qwen38fn-prompt-time depth {d}: {d:.1} ms a 2,048-row chunk, {d:.0} tok/s ({d} chunks)", .{ depth, ms, @as(f64, rows) * 1e3 / ms, chunks });
    }
}
