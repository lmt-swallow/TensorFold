//! `glm53-tier <aot-dir> <model-dir> <file-dir> [positions]`: the tiered latent cache (glm53 kv_tier.zig and
//! cuda_tier.zig) against a whole cache in device memory, on one GPU. One stream-layer of `positions` latent rows
//! (default 262,144) is written through the reference's latent write (`_lat_write`) into both, a prompt's chunks
//! and then decode windows with drafts, into a slab that holds about half of them (the rest on a file in
//! `file-dir`, which should be on the NVMe). Then selections like the indexer's (512 pools of 4 and the incomplete
//! pool's rows, recent and older pools mixed) run the reference's sparse attention twice: over the whole cache with
//! the positions, and over the slab with the positions the tier remapped. Every output byte must be equal: decode
//! windows of 1, 3 and 16 rows (`_sparse_chunks` and `_merge`) and a 64-row prompt block (`_sparse_onepass`).
//! It also times the sparse kernels reading the slab (mapped pinned memory) against the whole cache, and a
//! decode window's resolve (read back, fetch from the file, remap).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Buf = cuda.DeviceBuffer;
const Ptr = cuda.abi.DevicePtr;
const kv = glm.tier.kv;
const W = glm.glue.sparse_tokens;

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn ms(t0: i96, io: std.Io) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e6;
}

/// Random finite bf16 values of magnitude 2^-7 .. 2^2.
fn fillBf16(rnd: std.Random, out: []u8) void {
    var i: usize = 0;
    while (i + 1 < out.len) : (i += 2) {
        // sign, exponent 120 .. 128 (magnitudes 2^-7 .. 2^1), 7 mantissa bits
        const v: u16 = (@as(u16, rnd.uintLessThan(u16, 2)) << 15) | (@as(u16, 120 + rnd.uintLessThan(u16, 9)) << 7) | rnd.uintLessThan(u16, 128);
        std.mem.writeInt(u16, out[i..][0..2], v, .little);
    }
}

/// An indexer-like selection for R rows at pos .. pos + R - 1: per row 512 distinct pools below the row's
/// complete ones (a third recent, the rest anywhere), then the incomplete pool's rows.
fn selection(rnd: std.Random, pos: usize, R: usize, tokens: []i32, counts: []i32, seen: []u32, stamp: *u32) void {
    for (0..R) |r| {
        const end = pos + r + 1; // visible positions
        const full = end / 4;
        stamp.* += 1;
        var n: usize = 0;
        const want = @min(512, full);
        while (n / 4 < want) {
            const p = if (rnd.uintLessThan(u8, 3) == 0) full - 1 - rnd.uintLessThan(usize, @min(full, 4096)) else rnd.uintLessThan(usize, full);
            if (seen[p] == stamp.*) continue;
            seen[p] = stamp.*;
            for (0..4) |j| tokens[r * W + n + j] = @intCast(p * 4 + j);
            n += 4;
        }
        for (full * 4..end) |q| {
            tokens[r * W + n] = @intCast(q);
            n += 1;
        }
        counts[r] = @intCast(n);
    }
}

pub fn run(gpu: Gpu, aot_dir: []const u8, model: []const u8, file_dir: []const u8, positions_arg: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const io = gpu.io;
    var why: glm.config.Why = .{};
    const cfg = try glm.config.read(gpa, io, model, &why);
    const lay: glm.config.Layout = .{ .rank = 0, .world = 2, .cfg = &cfg };
    const H = lay.size(.heads);
    const lw = cfg.kv_lora;
    const rb = lw * 2;
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(cfg.qk_dim), -0.5));
    const N = if (positions_arg) |a| try std.fmt.parseInt(usize, a, 10) else 262_144;
    if (N % kv.seg_positions != 0) return error.BadShape;
    var set = try cuda.aot.Set.load(gpa, io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const g: glm.glue.Glue = .{ .set = &set, .hc = undefined, .hidden = cfg.hidden, .streams = cfg.streams };
    const stream = try cuda.Stream.init(d, true);
    var stream_ = stream;
    defer stream_.deinit();

    // the slab: the home block, the recent window and about half of the rest
    const recent = 8;
    const runs = kv.home_runs + recent + 2 + (N / kv.seg_positions) / 2;
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/glm53_tier_test.bin", .{file_dir}, 0);
    defer gpa.free(path);
    const o: kv.Options = .{ .streams = 1, .layers = 1, .capacity = N, .row_bytes = rb, .recent_runs = recent };
    const tier = try glm.tier.Tier.init(gpa, d, o, runs * kv.run_pools * kv.block, 64, W, path);
    defer {
        tier.deinit(gpa);
        _ = std.c.unlink(path.ptr);
    }
    try tier.open(0, 1);
    var whole = try Buf.alloc(d, N * rb);
    defer whole.free();
    try whole.fill8(0, null);

    var prng = std.Random.DefaultPrng.init(0x61a53);
    const rnd = prng.random();
    const chunk = kv.seg_positions;
    const host_lat = try gpa.alloc(u8, chunk * rb);
    defer gpa.free(host_lat);
    var lat = try Buf.alloc(d, chunk * rb);
    defer lat.free();
    var pos_dev = try Buf.alloc(d, 16);
    defer pos_dev.free();

    // the prompt: chunks of 2,048 rows up to N - 4,096, each committed
    const t_fill = now(io);
    var pos: usize = 0;
    while (pos + 2 * chunk < N) : (pos += chunk) {
        fillBf16(rnd, host_lat);
        try lat.upload(0, host_lat);
        const v: i32 = @intCast(pos);
        try pos_dev.upload(0, std.mem.asBytes(&v));
        try g.latWrite(lat.ptr, lw, whole.ptr, .bf16, pos_dev.ptr, chunk, lw, stream);
        try tier.beginWindow(0, pos, chunk);
        for (try tier.writes(0, 0, pos, chunk, pos_dev.ptr)) |pc| if (pc) |p| {
            try g.latWrite(lat.ptr + (p.pos - pos) * rb, lw, p.base, .bf16, p.pos_dev, p.rows, lw, stream);
        };
        try stream.synchronize();
        try tier.commit(0, pos + chunk);
    }
    std.debug.print("glm53-tier: {d} positions written in {d:.0} ms ({d} pools written, {d} read back, slab {d} of {d} runs)\n", .{ pos, ms(t_fill, io), tier.host.writes, tier.host.reads, runs, N / kv.seg_positions });

    const tokens = try gpa.alloc(i32, 64 * W);
    defer gpa.free(tokens);
    const counts = try gpa.alloc(i32, 64);
    defer gpa.free(counts);
    const seen = try gpa.alloc(u32, N / 4);
    defer gpa.free(seen);
    @memset(seen, 0);
    var stamp: u32 = 0;
    var qa = try Buf.alloc(d, 64 * H * rb);
    defer qa.free();
    const host_qa = try gpa.alloc(u8, 64 * H * rb);
    defer gpa.free(host_qa);
    var tok_a = try Buf.alloc(d, 64 * W * 4);
    defer tok_a.free();
    var tok_b = try Buf.alloc(d, 64 * W * 4);
    defer tok_b.free();
    var cnt = try Buf.alloc(d, 64 * 4);
    defer cnt.free();
    var out_a = try Buf.alloc(d, 64 * H * rb);
    defer out_a.free();
    var out_b = try Buf.alloc(d, 64 * H * rb);
    defer out_b.free();
    const ns = 5 * 16 * H; // chunks x rows x heads (decode windows of up to 16 rows)
    var po = try Buf.alloc(d, ns * lw * 4);
    defer po.free();
    var pm = try Buf.alloc(d, ns * 4);
    defer pm.free();
    var pl = try Buf.alloc(d, ns * 4);
    defer pl.free();
    const parts: glm.glue.Glue.Partials = .{ .po = po.ptr, .pm = pm.ptr, .pl = pl.ptr };
    const ha = try gpa.alloc(u8, 64 * H * rb);
    defer gpa.free(ha);
    const hb = try gpa.alloc(u8, 64 * H * rb);
    defer gpa.free(hb);
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();

    // decode windows with drafts: write R rows, select, attend both ways, keep a prefix
    var checked: usize = 0;
    var resolve_ms: f64 = 0;
    var resolves: usize = 0;
    var t_whole: f64 = 0;
    var t_slab: f64 = 0;
    const sizes = [_]usize{ 1, 3, 16 };
    var step: usize = 0;
    while (step < 48 and pos + 16 < N) : (step += 1) {
        const R = sizes[step % sizes.len];
        fillBf16(rnd, host_lat[0 .. R * rb]);
        try lat.upload(0, host_lat[0 .. R * rb]);
        const v: i32 = @intCast(pos);
        try pos_dev.upload(0, std.mem.asBytes(&v));
        try g.latWrite(lat.ptr, lw, whole.ptr, .bf16, pos_dev.ptr, R, lw, stream);
        try tier.beginWindow(0, pos, R);
        for (try tier.writes(0, 0, pos, R, pos_dev.ptr)) |pc| if (pc) |p| {
            try g.latWrite(lat.ptr + (p.pos - pos) * rb, lw, p.base, .bf16, p.pos_dev, p.rows, lw, stream);
        };
        selection(rnd, pos, R, tokens, counts, seen, &stamp);
        fillBf16(rnd, host_qa[0 .. R * H * rb]);
        try qa.upload(0, host_qa[0 .. R * H * rb]);
        try tok_a.upload(0, std.mem.sliceAsBytes(tokens[0 .. R * W]));
        try tok_b.upload(0, std.mem.sliceAsBytes(tokens[0 .. R * W]));
        try cnt.upload(0, std.mem.sliceAsBytes(counts[0..R]));
        try e0.record(stream);
        try g.sparseAttention(qa.ptr, whole.ptr, .bf16, tok_a.ptr, cnt.ptr, parts, out_a.ptr, R, H, lw, W, glm.dsa.headBlock(R), scale, stream);
        try e1.record(stream);
        try e1.synchronize();
        t_whole += try e0.elapsedMs(e1);
        const t_r = now(io);
        const base = try tier.resolve(0, 0, tok_b.ptr, cnt.ptr, R, W, stream);
        resolve_ms += ms(t_r, io);
        resolves += 1;
        try e0.record(stream);
        try g.sparseAttention(qa.ptr, base, .bf16, tok_b.ptr, cnt.ptr, parts, out_b.ptr, R, H, lw, W, glm.dsa.headBlock(R), scale, stream);
        try e1.record(stream);
        try e1.synchronize();
        t_slab += try e0.elapsedMs(e1);
        try out_a.download(0, ha[0 .. R * H * rb]);
        try out_b.download(0, hb[0 .. R * H * rb]);
        try check.expect(std.mem.eql(u8, ha[0 .. R * H * rb], hb[0 .. R * H * rb]), "glm53-tier: decode window of {d} rows at {d}: the slab's output differs from the whole cache's", .{ R, pos });
        checked += 1;
        const keep = 1 + rnd.uintLessThan(usize, R);
        pos += keep;
        tier.rewind(0, pos);
        try tier.commit(0, pos);
    }
    check.pass("glm53-tier: {d} decode windows (1, 3, 16 rows) over {d} positions: the slab's sparse attention equals the whole cache's byte for byte; resolve {d:.2} ms a window, kernel {d:.3} ms (slab) vs {d:.3} ms (device)", .{ checked, pos, resolve_ms / @as(f64, @floatFromInt(resolves)), t_slab / @as(f64, @floatFromInt(checked)), t_whole / @as(f64, @floatFromInt(checked)) });

    // a prompt block: 64 rows, one-pass kernel
    {
        const R = 64;
        try tier.beginWindow(0, pos, R);
        fillBf16(rnd, host_lat[0 .. R * rb]);
        try lat.upload(0, host_lat[0 .. R * rb]);
        const v: i32 = @intCast(pos);
        try pos_dev.upload(0, std.mem.asBytes(&v));
        try g.latWrite(lat.ptr, lw, whole.ptr, .bf16, pos_dev.ptr, R, lw, stream);
        for (try tier.writes(0, 0, pos, R, pos_dev.ptr)) |pc| if (pc) |p| {
            try g.latWrite(lat.ptr + (p.pos - pos) * rb, lw, p.base, .bf16, p.pos_dev, p.rows, lw, stream);
        };
        selection(rnd, pos, R, tokens, counts, seen, &stamp);
        fillBf16(rnd, host_qa[0 .. R * H * rb]);
        try qa.upload(0, host_qa[0 .. R * H * rb]);
        try tok_a.upload(0, std.mem.sliceAsBytes(tokens[0 .. R * W]));
        try tok_b.upload(0, std.mem.sliceAsBytes(tokens[0 .. R * W]));
        try cnt.upload(0, std.mem.sliceAsBytes(counts[0..R]));
        try e0.record(stream);
        try g.sparseOnepass(qa.ptr, whole.ptr, .bf16, tok_a.ptr, cnt.ptr, out_a.ptr, R, H, lw, W, scale, stream);
        try e1.record(stream);
        try e1.synchronize();
        const tw = try e0.elapsedMs(e1);
        const t_r = now(io);
        const reads = tier.host.reads;
        const base = try tier.resolve(0, 0, tok_b.ptr, cnt.ptr, R, W, stream);
        const rms = ms(t_r, io);
        try e0.record(stream);
        try g.sparseOnepass(qa.ptr, base, .bf16, tok_b.ptr, cnt.ptr, out_b.ptr, R, H, lw, W, scale, stream);
        try e1.record(stream);
        try e1.synchronize();
        const ts = try e0.elapsedMs(e1);
        try out_a.download(0, ha[0 .. R * H * rb]);
        try out_b.download(0, hb[0 .. R * H * rb]);
        try check.expect(std.mem.eql(u8, ha[0 .. R * H * rb], hb[0 .. R * H * rb]), "glm53-tier: prompt block of {d} rows: the slab's output differs from the whole cache's", .{R});
        check.pass("glm53-tier: a 64-row prompt block (one-pass kernel) equals the whole cache's byte for byte; resolve {d:.2} ms ({d} pools read), kernel {d:.3} ms (slab) vs {d:.3} ms (device)", .{ rms, tier.host.reads - reads, ts, tw });
    }
}
