//! `glm53-select <aot-dir>`: select.cu's prompt pool selection (glue.zig `Glue.promptPools`) against the definition
//! (each row's 512 best scores in total order, ties to the lower pool, ascending, computed on the host) and, where
//! the Triton set holds the variant, against the captured kernels (`_select_rows`, `select_split`'s pool form) on
//! the same scores. Score rows are drawn to stress the selection: distinct values, a few distinct values (many
//! ties at the threshold), mostly zeros, -inf columns past the visible pools, one value everywhere, signed zeros,
//! and threshold bins fuller than select.cu's shared memory (its fallback).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");

const Gpu = check.Gpu;
const Buf = cuda.DeviceBuffer;

const K = 512;

const Pattern = enum { distinct, few, zeros, neg_inf_tail, one_value, signed_zero, full_bin, cap_exact, cap_over };

/// select.cu's shared-memory candidates (select.cu kCand): cap_exact puts exactly this many keys in the threshold
/// bin, cap_over one more (the fallback).
const cand_cap = 6144;

fn orderKey(s: f32) u32 {
    var b: u32 = @bitCast(s);
    if (b == 0x80000000) b = 0;
    const sign: u32 = @bitCast(@as(i32, @bitCast(b)) >> 31);
    return b ^ (sign | 0x80000000);
}

fn draw(rng: std.Random, p: Pattern, np: usize, row: []f32) void {
    for (row, 0..) |*x, i| x.* = switch (p) {
        .distinct => rng.float(f32) * 8.0 - 4.0,
        .few => @as(f32, @floatFromInt(rng.intRangeLessThan(i32, -3, 4))) * 0.25,
        .zeros => if (rng.intRangeLessThan(u32, 0, 64) == 0) rng.float(f32) else 0.0,
        .neg_inf_tail => if (i + 1023 >= np) -std.math.inf(f32) else rng.float(f32) - 0.5,
        .one_value => 1.5,
        .signed_zero => if (rng.boolean()) -0.0 else 0.0,
        // values spread inside one 2^-11-wide band of keys: the threshold bin holds most of the row
        .full_bin => 1.0 + @as(f32, @floatFromInt(rng.intRangeLessThan(u32, 0, 1 << 12))) * (1.0 / 8388608.0),
        // the top bin's keys at scattered places, every other key below it
        .cap_exact, .cap_over => 0.0,
    };
    if (p == .cap_exact or p == .cap_over) {
        const n: usize = if (p == .cap_exact) cand_cap else cand_cap + 1;
        var placed: usize = 0;
        for (row) |*x| x.* = rng.float(f32) * 0.5; // below 1.0: lower bins
        while (placed < n) {
            const i = rng.intRangeLessThan(usize, 0, np);
            if (row[i] >= 1.0) continue;
            row[i] = 1.0 + @as(f32, @floatFromInt(rng.intRangeLessThan(u32, 0, 1 << 10))) * (1.0 / 8388608.0);
            placed += 1;
        }
    }
}

/// The definition on the host: the k best by (key, then the lower index), ascending.
fn reference(gpa: std.mem.Allocator, row: []const f32, out: []i64) !void {
    const idx = try gpa.alloc(u32, row.len);
    defer gpa.free(idx);
    for (idx, 0..) |*x, i| x.* = @intCast(i);
    const Ctx = struct {
        row: []const f32,
        fn better(c: @This(), a: u32, b: u32) bool {
            const ka = orderKey(c.row[a]);
            const kb = orderKey(c.row[b]);
            return ka > kb or (ka == kb and a < b);
        }
    };
    std.mem.sort(u32, idx, Ctx{ .row = row }, Ctx.better);
    const top = idx[0..K];
    std.mem.sort(u32, top, {}, std.sort.asc(u32));
    for (out, top) |*o, t| o.* = t;
}

pub fn run(gpu: Gpu, aot_dir: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var hc = try glm.glue.Hc.load(d);
    defer hc.deinit();
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hc, .hidden = 4096, .streams = 4 };
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(20261012);
    const rng = prng.random();

    const nps = [_]usize{ 1024, 2048, 4096, 8192, 9216, 20480, 20481, 65536, 65537, 150016 };
    const rowss = [_]usize{ 1, 37, 63, 64, 512 };
    var failures: usize = 0;
    var cases: usize = 0;
    var captured: usize = 0;
    for (nps) |np| for (rowss) |n| for (std.enums.values(Pattern)) |p| {
        if ((p == .cap_exact or p == .cap_over) and np < 2 * cand_cap) continue;
        if (n == 512 and np > 20481 and p != .distinct and p != .cap_over) continue; // keep the host sort's time bounded
        const scores = try gpa.alloc(f32, n * np);
        defer gpa.free(scores);
        for (0..n) |r| draw(rng, p, np, scores[r * np ..][0..np]);
        var sbuf = try Buf.fromHost(d, std.mem.sliceAsBytes(scores));
        defer sbuf.free();
        var obuf = try Buf.alloc(d, n * K * 8);
        defer obuf.free();
        var split = try Buf.alloc(d, glm.glue.splitScratchBytes(@max(n, 1), np));
        defer split.free();
        hc.sel.use = true;
        hc.sel.check = false;
        try g.promptPools(sbuf.ptr, np, obuf.ptr, n, split, stream);
        try stream.synchronize();
        const got = try gpa.alloc(i64, n * K);
        defer gpa.free(got);
        try obuf.download(0, std.mem.sliceAsBytes(got));
        const want = try gpa.alloc(i64, K);
        defer gpa.free(want);
        var bad_rows: usize = 0;
        for (0..n) |r| {
            try reference(gpa, scores[r * np ..][0..np], want);
            if (!std.mem.eql(i64, want, got[r * K ..][0..K])) bad_rows += 1;
        }
        // the captured kernels, where the set holds their variant
        var cap_bad: ?usize = null;
        hc.sel.use = false;
        var cbuf = try Buf.alloc(d, n * K * 8);
        defer cbuf.free();
        if (g.promptPools(sbuf.ptr, np, cbuf.ptr, n, split, stream)) |_| {
            try stream.synchronize();
            const cap = try gpa.alloc(i64, n * K);
            defer gpa.free(cap);
            try cbuf.download(0, std.mem.sliceAsBytes(cap));
            var b: usize = 0;
            for (0..n) |r| b += @intFromBool(!std.mem.eql(i64, cap[r * K ..][0..K], got[r * K ..][0..K]));
            cap_bad = b;
            captured += 1;
        } else |e| if (e != error.MissingTritonVariant) return e;
        cases += 1;
        if (bad_rows != 0 or (cap_bad orelse 0) != 0) {
            failures += 1;
            std.debug.print("FAIL glm53-select np {d} rows {d} {t}: {d} rows differ from the definition, {?d} from the captured kernels\n", .{ np, n, p, bad_rows, cap_bad });
        }
    };
    try check.expect(failures == 0, "glm53-select: {d} of {d} cases differ", .{ failures, cases });
    check.pass("glm53-select: {d} cases equal to the definition, {d} of them also to the captured kernels", .{ cases, captured });
}

/// `glm53-scores <aot-dir>`: scores.cu (glue.zig `Glue.scores`) against the captured `_scores` (TF_GLM_SCORE_RB's
/// variant, which the set must hold) byte for byte, on drawn queries, head weights and pooled keys, over block
/// positions and row counts that put the visible-pool edge inside, before and after the 128-pool tiles.
pub fn runScores(gpu: Gpu, aot_dir: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var hc = try glm.glue.Hc.load(d);
    defer hc.deinit();
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hc, .hidden = 4096, .streams = 4 };
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(20261013);
    const rng = prng.random();
    const H = 32;
    const D = 128;
    const cases = [_]struct { pos: usize, rows: usize, np: usize }{
        .{ .pos = 2051, .rows = 512, .np = 2048 },   .{ .pos = 4096, .rows = 37, .np = 2048 },
        .{ .pos = 30720, .rows = 512, .np = 8192 },  .{ .pos = 32767, .rows = 512, .np = 9216 },
        .{ .pos = 81919, .rows = 64, .np = 21504 },  .{ .pos = 204800, .rows = 512, .np = 52224 },
        .{ .pos = 262141, .rows = 100, .np = 66560 }, .{ .pos = 600000, .rows = 512, .np = 150016 },
    };
    var failures: usize = 0;
    for (cases) |cs| {
        const pools = cs.np + 2;
        const qi = try gpa.alloc(u16, cs.rows * H * D);
        defer gpa.free(qi);
        const wts = try gpa.alloc(u16, cs.rows * (D + H)); // the indexer's [keys | weights] rows: weights at D
        defer gpa.free(wts);
        const pk = try gpa.alloc(u16, pools * D);
        defer gpa.free(pk);
        for ([_][]u16{ qi, wts, pk }) |arr| for (arr) |*x| {
            const v: f32 = rng.floatNorm(f32);
            x.* = @intCast(@as(u32, @bitCast(v)) >> 16);
        };
        var bq = try Buf.fromHost(d, std.mem.sliceAsBytes(qi));
        defer bq.free();
        var bw = try Buf.fromHost(d, std.mem.sliceAsBytes(wts));
        defer bw.free();
        var bk = try Buf.fromHost(d, std.mem.sliceAsBytes(pk));
        defer bk.free();
        const p32: i32 = @intCast(cs.pos);
        var bp = try Buf.fromHost(d, std.mem.asBytes(&p32));
        defer bp.free();
        var o1 = try Buf.alloc(d, cs.rows * cs.np * 4);
        defer o1.free();
        var o2 = try Buf.alloc(d, cs.rows * cs.np * 4);
        defer o2.free();
        hc.sco.check = false;
        hc.sco.use = true;
        try g.scores(bq.ptr, bw.ptr + D * 2, D + H, bk.ptr, .bf16, o1.ptr, bp.ptr, cs.rows, cs.np, stream);
        hc.sco.use = false;
        try g.scores(bq.ptr, bw.ptr + D * 2, D + H, bk.ptr, .bf16, o2.ptr, bp.ptr, cs.rows, cs.np, stream);
        try stream.synchronize();
        const a = try gpa.alloc(u8, cs.rows * cs.np * 4);
        defer gpa.free(a);
        const b = try gpa.alloc(u8, cs.rows * cs.np * 4);
        defer gpa.free(b);
        try o1.download(0, a);
        try o2.download(0, b);
        if (std.mem.indexOfDiff(u8, a, b)) |at| {
            failures += 1;
            std.debug.print("FAIL glm53-scores pos {d} rows {d} np {d}: first difference at row {d} pool {d}\n", .{ cs.pos, cs.rows, cs.np, at / 4 / cs.np, at / 4 % cs.np });
        }
    }
    try check.expect(failures == 0, "glm53-scores: {d} of {d} cases differ from _scores", .{ failures, cases.len });
    check.pass("glm53-scores: {d} cases byte-equal to the captured _scores", .{cases.len});
}

/// `glm53-onepass <aot-dir>`: onepass.cu (glue.zig `Glue.sparseOnepass`) against the captured `_sparse_onepass` byte
/// for byte on drawn queries, latents and token lists: counts 0 (rows left alone), a few tokens, a partial last tile,
/// and the full 2,051, over caches whose scores span many tiles' maxima (the online softmax's rescaling).
pub fn runOnepass(gpu: Gpu, aot_dir: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var hc = try glm.glue.Hc.load(d);
    defer hc.deinit();
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hc, .hidden = 4096, .streams = 4 };
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(20261014);
    const rng = prng.random();
    const H = 32;
    const LW = 512;
    const W = glm.glue.sparse_tokens;
    const R = 300;
    const tokens = 40_000;
    const cache = try gpa.alloc(u16, tokens * LW);
    defer gpa.free(cache);
    const qa = try gpa.alloc(u16, R * H * LW);
    defer gpa.free(qa);
    for (cache, 0..) |*x, i| {
        // rows of different magnitudes: the tiles' maxima move, so the softmax rescales
        const v: f32 = rng.floatNorm(f32) * (1.0 + @as(f32, @floatFromInt((i / LW) % 7)));
        x.* = @intCast(@as(u32, @bitCast(v)) >> 16);
    }
    for (qa) |*x| x.* = @intCast(@as(u32, @bitCast(rng.floatNorm(f32) * 0.5)) >> 16);
    const tok = try gpa.alloc(i32, R * W);
    defer gpa.free(tok);
    const cnt = try gpa.alloc(i32, R);
    defer gpa.free(cnt);
    for (0..R) |r| {
        const n: usize = switch (r % 6) {
            0 => 0,
            1 => 1 + r % 31,
            2 => 33,
            3 => 2051,
            4 => 2048 + r % 3,
            else => 1000 + r,
        };
        cnt[r] = @intCast(n);
        var last: usize = rng.intRangeLessThan(usize, 0, 100);
        for (0..W) |j| {
            if (j < n) {
                tok[r * W + j] = @intCast(last % tokens);
                last += 1 + rng.intRangeLessThan(usize, 0, 15);
            } else tok[r * W + j] = -1;
        }
    }
    var bc = try Buf.fromHost(d, std.mem.sliceAsBytes(cache));
    defer bc.free();
    var bq = try Buf.fromHost(d, std.mem.sliceAsBytes(qa));
    defer bq.free();
    var bt = try Buf.fromHost(d, std.mem.sliceAsBytes(tok));
    defer bt.free();
    var bn = try Buf.fromHost(d, std.mem.sliceAsBytes(cnt));
    defer bn.free();
    var o1 = try Buf.alloc(d, R * H * LW * 2);
    defer o1.free();
    var o2 = try Buf.alloc(d, R * H * LW * 2);
    defer o2.free();
    try o1.fill8(0x5A, null); // rows with count 0 keep this in both
    try o2.fill8(0x5A, null);
    const scale: f32 = @floatCast(std.math.pow(f64, 256.0, -0.5));
    hc.one.check = false;
    hc.one.use = true;
    try g.sparseOnepass(bq.ptr, bc.ptr, .bf16, bt.ptr, bn.ptr, o1.ptr, R, H, LW, W, scale, stream);
    hc.one.use = false;
    try g.sparseOnepass(bq.ptr, bc.ptr, .bf16, bt.ptr, bn.ptr, o2.ptr, R, H, LW, W, scale, stream);
    try stream.synchronize();
    const a = try gpa.alloc(u8, R * H * LW * 2);
    defer gpa.free(a);
    const b = try gpa.alloc(u8, R * H * LW * 2);
    defer gpa.free(b);
    try o1.download(0, a);
    try o2.download(0, b);
    var bad: usize = 0;
    for (0..R) |r| {
        const n = H * LW * 2;
        if (!std.mem.eql(u8, a[r * n ..][0..n], b[r * n ..][0..n])) {
            if (bad < 5) std.debug.print("FAIL glm53-onepass row {d} (count {d}) differs from _sparse_onepass\n", .{ r, cnt[r] });
            bad += 1;
        }
    }
    try check.expect(bad == 0, "glm53-onepass: {d} of {d} rows differ from _sparse_onepass", .{ bad, R });
    check.pass("glm53-onepass: {d} rows byte-equal to the captured _sparse_onepass", .{R});
}
