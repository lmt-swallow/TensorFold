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
