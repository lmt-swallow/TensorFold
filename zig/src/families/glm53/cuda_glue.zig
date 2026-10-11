//! The Mia fork's GLM-5.3-Flash row kernels replayed from captured cubins (`cuda.aot.Set`): each call launches the
//! Triton kernel with the grid, runtime arguments and constexprs of the Python wrapper it names, so the set picks the
//! variant Triton compiled for that call and the bits are the reference's; a launch the capture never saw fails with
//! error.MissingTritonVariant. The decode rows' hyper-connection mixing dots run hc.cu's kernel (glm53_hc fatbin).
//!
//! Wrappers (glm5_next/cuda module: function): glue: embed (_embed_b16), rmsnorm, hc_pre (hc.cu or
//! _hc_partial_mma, then _hc_finish), hc_post, residual_add, stream_mean, swiglu, router (_router_part,
//! _router_sum), select (_topk), combine_shared (_combine_sy); qmm: group_sums, matmul on BF16 weights (_bmm,
//! _reduce); forward: _shift_conv (_conv_shift).
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const Ptr = cuda.abi.DevicePtr;

/// glue.py HC_BLOCKS: the mixing dots' fixed K split.
pub const hc_blocks = 16;
/// qmm.py B16_BN, B16_BK, SPLIT_TARGET, SEQ_ROWS (TF_GLM_SEQ_ROWS unset).
const b16_bn = 64;
const b16_bk = 64;
const split_target = 192;
const seq_rows = 256;
/// glue.py ROUTER_KS.
const router_ks = 8;
/// sparse.py's score scales: index_dim ** -0.5 and the 32 heads' 1 / 5.656854249492381 (32 ** -0.5 as written).
const index_scale: f32 = @floatCast(std.math.pow(f64, 128.0, -0.5));
const index_wscale: f32 = @floatCast(1.0 / 5.656854249492381);
/// sparse.py TOKENS: a sparse row's attended tokens (512 pools of 4, then its incomplete pool's 3).
pub const sparse_tokens = 512 * 4 + 3;

/// The DSA caches' format (the Mia fork's TF_GLM_KV, glm5_next/cuda/kv8.py): the latent rows and the indexer's
/// pooled keys, bf16 (the default), or fp8: a row of `width` values is `width` e4m3 codes, its fp32 power-of-two
/// scale and 12 zero bytes (`rowBytes`), as `_lat_write` and `_pool_keys` store it. The indexer's per-token key and
/// gate ring is bf16 in both. Every kernel that reads or writes these caches takes the format as its FP8 constexpr,
/// its row stride RS (elements of the value pointer) and the value and scale pointers' types.
pub const Kv = enum {
    bf16,
    fp8,

    /// kv8.py PAD: the bytes after an FP8 row's codes (its scale, then zeros).
    pub const pad = 16;

    /// kv8.py `kv_kind`: TF_GLM_KV's value ("" is bf16).
    pub fn parse(text: []const u8) !Kv {
        if (text.len == 0) return .bf16;
        return std.meta.stringToEnum(Kv, text) orelse error.BadKvFormat;
    }

    /// kv8.py `row_bytes`: a cache row of `width` values.
    pub fn rowBytes(k: Kv, width: usize) usize {
        return switch (k) {
            .bf16 => 2 * width,
            .fp8 => width + pad,
        };
    }

    /// kv8.py `parts`: a row's stride in elements of the kernel's value pointer.
    fn stride(k: Kv, width: usize) usize {
        return switch (k) {
            .bf16 => width,
            .fp8 => width + pad,
        };
    }

    /// The value pointer's Triton type: bf16 values, or the e4m3 codes (a torch.float8_e4m3fn view).
    fn values(k: Kv) []const u8 {
        return switch (k) {
            .bf16 => "*bf16",
            .fp8 => "*fp8e4nv",
        };
    }

    /// The scale pointer's Triton type: an FP8 cache's fp32 view; a bf16 cache passes itself, unread.
    fn scales(k: Kv) []const u8 {
        return switch (k) {
            .bf16 => "*bf16",
            .fp8 => "*fp32",
        };
    }

    fn flag(k: Kv) i64 {
        return @intFromBool(k == .fp8);
    }

    /// kv8.py `quantize_rows` on the host: one bf16 row (`row`, bf16 bits) as an FP8 cache row (`out`, `rowBytes`
    /// long): its scale 2^(ceil(log2 amax) - 8), the biased exponent held within 1 .. 254 (a zero row: 2^-126), each
    /// value times the scale's exact reciprocal to e4m3 rounding to nearest even (|x| stays within 256: nothing
    /// saturates), the fp32 scale little-endian, 12 zero bytes; the bytes `_lat_write` and `_pool_keys` store.
    pub fn quantizeRow(row: []const u16, out: []u8) void {
        std.debug.assert(out.len == row.len + pad);
        var amax: f32 = 0;
        for (row) |x| amax = @max(amax, @abs(fromBf16(x)));
        const bits: u32 = @bitCast(amax);
        const e0: i32 = @intCast(((bits >> 23) & 0xFF) + @intFromBool(bits & 0x7FFFFF != 0));
        const e = std.math.clamp(e0 - 8, 1, 254);
        const scale: f32 = @bitCast(@as(u32, @intCast(e)) << 23);
        const inv: f32 = @bitCast(@as(u32, @intCast(254 - e)) << 23);
        for (row, out[0..row.len]) |x, *o| o.* = e4m3(fromBf16(x) * inv);
        @memcpy(out[row.len..][0..4], std.mem.asBytes(&scale));
        @memset(out[row.len + 4 ..], 0);
    }

    fn fromBf16(x: u16) f32 {
        return @bitCast(@as(u32, x) << 16);
    }

    /// fp32 -> e4m3fn (bias 7, 3 mantissa bits, subnormals from 2^-9), rounding to nearest even, saturating at 448.
    fn e4m3(v: f32) u8 {
        const sign: u8 = if (std.math.signbit(v)) 0x80 else 0;
        const a: f64 = @abs(@as(f64, v));
        if (a < 0x1p-6) return sign | @as(u8, @intCast(rne(a * 0x1p9)));    // subnormal (8: the smallest normal)
        var exp: i32 = std.math.ilogb(a);
        var m = rne((a / std.math.ldexp(@as(f64, 1), exp) - 1) * 8);
        if (m == 8) {
            m = 0;
            exp += 1;
        }
        const code: u32 = @as(u32, @intCast(exp + 7)) << 3 | m;
        return sign | @as(u8, @intCast(@min(code, 0x7E)));
    }

    fn rne(x: f64) u32 {
        const f = @floor(x);
        const d = x - f;
        const n: u32 = @intFromFloat(f);
        return if (d > 0.5 or (d == 0.5 and n % 2 == 1)) n + 1 else n;
    }
};

/// sparse.py `split_scratch`: select_split's int32s for R rows of np scores.
pub fn splitScratchBytes(R: usize, np: usize) usize {
    const chs: usize = if (np <= 16384) 2048 else 4096;
    return R * (1024 + std.math.ceilPowerOfTwoAssert(usize, cdiv(np, chs)) * 257) * 4;
}

fn n32(v: usize) i32 {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// qmm.py `b16_split_k`: K slices of a BF16 matmul, fixed by the shape.
pub fn b16SplitK(n: usize, k: usize) usize {
    const tiles = cdiv(n, b16_bn);
    const steps = k / b16_bk;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// qmm.py `bucket`.
fn bucket(m: usize) usize {
    for ([_]usize{ 16, 32, 64, 128 }) |b| if (m <= b) return b;
    unreachable;
}

/// The glue's own CUDA kernels: hc.cu's mixing dots (decode rows) and bmm.cu's BF16 matmul (`Bmm`).
pub const Hc = struct {
    module: cuda.Module,
    partial: cuda.Function,
    bmm: Bmm,

    pub fn load(d: *const cuda.Driver) !Hc {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.glm53_hc);
        errdefer module.unload();
        const partial = try module.function("_ZN11tf_glm53_hc17hc_partial_kernelEPK13__nv_bfloat16S2_Pfi");
        return .{ .module = module, .partial = partial, .bmm = try Bmm.load(d) };
    }

    pub fn deinit(self: *Hc) void {
        self.bmm.deinit();
        self.module.unload();
    }
};

/// bmm.cu: `_bmm` (and `_reduce` after the split grid) with their bits at the mma rate, for prompt rows (seq_rows or
/// more) of the shapes in `table`; other calls take the captured kernels. TF_GLM_BMM=0 keeps the captured kernels
/// for every call; TF_GLM_PROMPT_CHECK=1 runs the captured kernels beside bmm.cu on every call it takes and fails
/// unless the outputs are byte-equal.
pub const Bmm = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    fns: [kernels.len]cuda.Function,
    use: bool,
    check: bool,

    const Kernel = struct { name: [:0]const u8, bm: u32, bn: u32, bk: u32, threads: u32, stages: u32, swizzled: bool, split: bool, f32: bool };
    const kernels = [_]Kernel{
        .{ .name = "tf_glm53_bmm_128x256_1b", .bm = 128, .bn = 256, .bk = 32, .threads = 256, .stages = 3, .swizzled = false, .split = false, .f32 = false },
        .{ .name = "tf_glm53_bmm_128x256k64z_1b", .bm = 128, .bn = 256, .bk = 64, .threads = 256, .stages = 2, .swizzled = true, .split = false, .f32 = false },
        .{ .name = "tf_glm53_bmm_128x128k64_1b", .bm = 128, .bn = 128, .bk = 64, .threads = 256, .stages = 2, .swizzled = false, .split = false, .f32 = false },
        .{ .name = "tf_glm53_bmm_128x128k64z_1b", .bm = 128, .bn = 128, .bk = 64, .threads = 256, .stages = 3, .swizzled = true, .split = false, .f32 = false },
        .{ .name = "tf_glm53_bmm_128x128k64_sb", .bm = 128, .bn = 128, .bk = 64, .threads = 256, .stages = 2, .swizzled = false, .split = true, .f32 = false },
        .{ .name = "tf_glm53_bmm_128x128k64_sf", .bm = 128, .bn = 128, .bk = 64, .threads = 256, .stages = 2, .swizzled = false, .split = true, .f32 = true },
        .{ .name = "tf_glm53_bmm_128x128k64z_sb", .bm = 128, .bn = 128, .bk = 64, .threads = 256, .stages = 3, .swizzled = true, .split = true, .f32 = false },
        .{ .name = "tf_glm53_bmm_128x128k64z_sf", .bm = 128, .bn = 128, .bk = 64, .threads = 256, .stages = 3, .swizzled = true, .split = true, .f32 = true },
        .{ .name = "tf_glm53_bmm_128x64_sb", .bm = 128, .bn = 64, .bk = 32, .threads = 256, .stages = 4, .swizzled = false, .split = true, .f32 = false },
    };
    /// Per-rank shapes (n, k, fp32 out) bmm.cu takes, and the kernel each runs (measured on GB10 at 2,048 rows); the
    /// rest stay on the captured kernels (bmm.cu was slower there).
    const Shape = struct { n: usize, k: usize, f32: bool, kernel: []const u8 };
    const table = [_]Shape{
        .{ .n = 12576, .k = 4096, .f32 = false, .kernel = "tf_glm53_bmm_128x256k64z_1b" },
        .{ .n = 12288, .k = 4096, .f32 = false, .kernel = "tf_glm53_bmm_128x256k64z_1b" },
        .{ .n = 2048, .k = 4096, .f32 = false, .kernel = "tf_glm53_bmm_128x128k64z_sb" },
        .{ .n = 4096, .k = 1024, .f32 = true, .kernel = "tf_glm53_bmm_128x128k64z_sf" },
        .{ .n = 4096, .k = 1536, .f32 = false, .kernel = "tf_glm53_bmm_128x128k64z_sb" },
        .{ .n = 4096, .k = 4096, .f32 = true, .kernel = "tf_glm53_bmm_128x128k64z_sf" },
        .{ .n = 4096, .k = 6144, .f32 = true, .kernel = "tf_glm53_bmm_128x128k64z_sf" },
        .{ .n = 4096, .k = 8192, .f32 = false, .kernel = "tf_glm53_bmm_128x128k64z_sb" },
    };

    pub fn load(d: *const cuda.Driver) !Bmm {
        var module = try cuda.Module.load(d, cuda.kernels.glm53_bmm);
        errdefer module.unload();
        var fns: [kernels.len]cuda.Function = undefined;
        for (kernels, &fns) |k, *f| {
            f.* = try module.function(k.name);
            try f.allowDynamicShared(smem(k));
        }
        return .{
            .d = d,
            .module = module,
            .fns = fns,
            .use = !std.mem.eql(u8, envOr("TF_GLM_BMM", "1"), "0"),
            .check = std.mem.eql(u8, envOr("TF_GLM_PROMPT_CHECK", "0"), "1"),
        };
    }

    pub fn deinit(self: *Bmm) void {
        self.module.unload();
    }

    fn smem(k: Kernel) u32 {
        return k.stages * (k.bm + k.bn) * (if (k.swizzled) k.bk else k.bk + 8) * 2;
    }

    /// The kernel for a call, or null (the captured kernels): prompt rows, a shape in `table`, and bmm.cu's operand
    /// requirements (16-byte aligned rows, k a multiple of the kernel's K step times sk, n even).
    fn pick(self: *const Bmm, x: Ptr, x_stride: usize, w: B16, m: usize, fp32: bool, sk: usize) ?usize {
        if (!self.use or m < seq_rows) return null;
        if (x % 16 != 0 or w.ptr % 16 != 0 or x_stride % 8 != 0 or w.n % 2 != 0) return null;
        const name = for (table) |e| {
            if (e.n == w.n and e.k == w.k and e.f32 == fp32) break e.kernel;
        } else return null;
        for (kernels, 0..) |k, i| if (std.mem.eql(u8, k.name, name)) {
            if (k.split != (sk > 1) or k.f32 != fp32 or w.k % (k.bk * sk) != 0) return null;
            return i;
        };
        return null;
    }

    fn run(self: *const Bmm, i: usize, x: Ptr, x_stride: usize, w: B16, m: usize, out: Ptr, sk: usize, stream: cuda.Stream) !void {
        const k = kernels[i];
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intCast(x_stride)));
        a.add(w.ptr);
        a.add(out);
        a.add(@as(c_int, @intCast(m)));
        a.add(@as(c_int, @intCast(w.n)));
        a.add(@as(c_int, @intCast(w.k)));
        a.add(@as(c_int, @intCast(sk)));
        try cuda.launch.launch(self.fns[i], .{
            .grid = .{ .x = @intCast(cdiv(m, k.bm)), .y = @intCast(cdiv(w.n, k.bn)) },
            .block = .{ .x = k.threads },
            .shared = smem(k),
        }, stream, &a);
    }
};

fn envOr(name: [:0]const u8, default: []const u8) []const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return default);
    return if (v.len == 0) default else v;
}

/// A BF16 matrix [n, k] on the device (qmm.py `B16`), row-major.
pub const B16 = struct { ptr: Ptr, n: usize, k: usize };

pub const Glue = struct {
    set: *const aot.Set,
    hc: *const Hc,
    hidden: usize, // D
    streams: usize, // S

    /// glue.py `embed` of a BF16 table: out [rows, copies D] bf16, token ids[r]'s row in every copy.
    pub fn embed(g: Glue, ids: Ptr, table: Ptr, out: Ptr, rows: usize, copies: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_embed_b16", .{ @intCast(rows), @intCast(g.hidden / 64), 1 }, &.{
            aot.ptr("IDS", "*i32", ids), aot.ptr("W", "*bf16", table), aot.ptr("OUT", "*bf16", out),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("COPIES", @intCast(copies)) });
    }

    /// glue.py `rmsnorm`: out [rows, d] bf16 (row stride o_stride) of x (row stride x_stride), and its 64-group
    /// sums xs when not 0.
    pub fn rmsnorm(g: Glue, x: Ptr, x_stride: usize, w: Ptr, out: Ptr, o_stride: usize, xs: Ptr, eps: f32, rows: usize, d: usize, stream: cuda.Stream) !void {
        const block = std.math.ceilPowerOfTwoAssert(usize, d);
        try g.set.run(stream, "_rmsnorm", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x),     aot.int("x_stride", n32(x_stride)),            aot.ptr("W", "*bf16", w),
            aot.ptr("OUT", "*bf16", out), aot.int("o_stride", n32(o_stride)),            aot.ptr("XS", if (xs != 0) "*fp32" else "*bf16", if (xs != 0) xs else out),
            aot.float("eps", eps),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("SUMS", @intFromBool(xs != 0)) });
    }

    pub const HcPre = struct {
        x: Ptr, // [rows, S D] bf16, contiguous
        fn_: Ptr, // [24, S D] bf16
        base: Ptr, // [24] fp32
        scale: Ptr, // [3] fp32
        norm: Ptr, // [D] bf16
        out: Ptr, // [rows, D] bf16
        xs: Ptr, // [rows, D / 64] fp32
        post: Ptr, // [rows, S] fp32
        comb: Ptr, // [rows, S S] fp32
        part: Ptr, // [rows, 16, 32] fp32
        eps: f32,
        hc_eps: f32,
        iters: usize,
        rows: usize,
        prompt: bool,
    };

    /// glue.py `hc_pre`: the mixing dots (a prompt chunk's on the tensor cores, decode rows' in hc.cu, each with
    /// its own bits), then _hc_finish: the normed collapsed row, its sums, post and the Sinkhorn-balanced comb.
    pub fn hcPre(g: Glue, a: HcPre, stream: cuda.Stream) !void {
        const wide = g.streams * g.hidden;
        if (a.prompt) {
            try g.set.run(stream, "_hc_partial_mma", .{ @intCast(cdiv(a.rows, 64)), hc_blocks, 1 }, &.{
                aot.ptr("X", "*bf16", a.x), aot.ptr("FN", "*bf16", a.fn_), aot.ptr("PART", "*fp32", a.part), aot.int("R", n32(a.rows)),
            }, &.{ aot.ci("WIDE", @intCast(wide)), aot.ci("NB", hc_blocks), aot.ci("BM", 64), aot.ci("BK", 64) });
        } else {
            if (wide != 16384) return error.UnsupportedShape;
            var args: cuda.Args = .{};
            args.add(a.x);
            args.add(a.fn_);
            args.add(a.part);
            args.add(@as(c_int, @intCast(a.rows)));
            try cuda.launch.launch(g.hc.partial, .{ .grid = .{ .x = hc_blocks, .y = 3, .z = @intCast(cdiv(a.rows, 4)) }, .block = .{ .x = 288 } }, stream, &args);
        }
        try g.set.run(stream, "_hc_finish", .{ @intCast(a.rows), 1, 1 }, &.{
            aot.ptr("X", "*bf16", a.x),          aot.ptr("PART", "*fp32", a.part), aot.ptr("BASE", "*fp32", a.base),
            aot.ptr("SCALE", "*fp32", a.scale),  aot.ptr("NW", "*bf16", a.norm),   aot.ptr("OUT", "*bf16", a.out),
            aot.ptr("XS", "*fp32", a.xs),        aot.ptr("POST", "*fp32", a.post), aot.ptr("COMB", "*fp32", a.comb),
            aot.float("eps_norm", a.eps),        aot.float("hc_eps", a.hc_eps),
        }, &.{
            aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("NB", hc_blocks),
            aot.ci("ITERS", @intCast(a.iters)), aot.ci("BLOCK", @intCast(g.hidden)),
        });
    }

    /// glue.py `hc_post`: xout = the streams of x mixed by comb plus post times bf16(the ranks' fp32 partials
    /// `gathered` [world, rows, D], rank 0 first).
    pub fn hcPost(g: Glue, x: Ptr, xout: Ptr, gathered: Ptr, world: usize, post: Ptr, comb: Ptr, rows: usize, stream: cuda.Stream) !void {
        const block = @min(1024, g.hidden);
        try g.set.run(stream, "_hc_post", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("X", "*bf16", x),       aot.ptr("XOUT", "*bf16", xout), aot.ptr("G", "*fp32", gathered),
            aot.ptr("POST", "*fp32", post), aot.ptr("COMB", "*fp32", comb), aot.int("RS", n32(rows * g.hidden)),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("WORLD", @intCast(world)), aot.ci("BLOCK", @intCast(block)) });
    }

    /// glue.py `hc_post_pair` (TF_GLM_HC_SPLIT): hc_post of two ranks' fp32 partials [rows, D] held apart (rank 0's at
    /// g0, rank 1's at g1): the same kernel reading rank 1's rows at their element distance from rank 0's, so the
    /// sum (rank 0 first) and every bit are hc_post's.
    pub fn hcPostPair(g: Glue, x: Ptr, xout: Ptr, g0: Ptr, g1: Ptr, post: Ptr, comb: Ptr, rows: usize, stream: cuda.Stream) !void {
        if (rows == 0) return;
        const block = @min(1024, g.hidden);
        const dist = @as(i64, @intCast(g1)) - @as(i64, @intCast(g0));
        if (@mod(dist, 4) != 0) return error.UnalignedPartials;
        const rs: i32 = std.math.cast(i32, @divExact(dist, 4)) orelse return error.PartialsTooFarApart;
        try g.set.run(stream, "_hc_post", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("X", "*bf16", x),       aot.ptr("XOUT", "*bf16", xout), aot.ptr("G", "*fp32", g0),
            aot.ptr("POST", "*fp32", post), aot.ptr("COMB", "*fp32", comb), aot.int("RS", rs),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("WORLD", 2), aot.ci("BLOCK", @intCast(block)) });
    }

    /// glue.py `residual_add` (the MTP layer): xout = bf16(x + bf16(the partials summed rank 0 first)).
    pub fn residualAdd(g: Glue, x: Ptr, xout: Ptr, gathered: Ptr, world: usize, rows: usize, stream: cuda.Stream) !void {
        const block = @min(1024, g.hidden);
        try g.set.run(stream, "_residual_add", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("XOUT", "*bf16", xout), aot.ptr("G", "*fp32", gathered), aot.int("RS", n32(rows * g.hidden)),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("WORLD", @intCast(world)), aot.ci("BLOCK", @intCast(block)) });
    }

    /// glue.py `stream_mean`: out [rows, D] bf16 = the streams' mean.
    pub fn streamMean(g: Glue, x: Ptr, out: Ptr, rows: usize, stream: cuda.Stream) !void {
        const block = @min(1024, g.hidden);
        try g.set.run(stream, "_stream_mean", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("OUT", "*bf16", out),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("S", @intCast(g.streams)), aot.ci("BLOCK", @intCast(block)) });
    }

    /// glue.py `swiglu`: out [rows, w] bf16 = the limited SwiGLU of [gate | up] rows, and its 64-group sums.
    pub fn swiglu(g: Glue, gu: Ptr, out: Ptr, xs: Ptr, limit: f32, rows: usize, w: usize, stream: cuda.Stream) !void {
        const block = std.math.gcd(512, w);
        try g.set.run(stream, "_swiglu", .{ @intCast(rows), @intCast(w / block), 1 }, &.{
            aot.ptr("GU", "*bf16", gu), aot.ptr("OUT", "*bf16", out), aot.ptr("XS", "*fp32", xs), aot.float("LIMIT", limit),
        }, &.{ aot.ci("W", @intCast(w)), aot.ci("BLOCK", @intCast(block)) });
    }

    /// qmm.py `group_sums`: xs [m, k / 64] fp32, each 64-input group's sum of bf16 x (row stride x_stride).
    pub fn groupSums(g: Glue, x: Ptr, x_stride: usize, xs: Ptr, m: usize, k: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_group_sums", .{ @intCast(m), @intCast(cdiv(k / 64, 16)), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("XS", "*fp32", xs), aot.int("x_stride", n32(x_stride)),
        }, &.{ aot.ci("K", @intCast(k)), aot.ci("GB", 16) });
    }

    /// qmm.py `matmul` of BF16 weights (`_matmul_b16`): out [m, n] (bf16, or fp32 with f32) = x [m, k] (row stride
    /// x_stride) times w^T, K in the shape's fixed slices summed in slice order (one program for prompt rows, else
    /// fp32 partials in `part`, at least sk m n floats, and _reduce).
    pub fn bmm(g: Glue, x: Ptr, x_stride: usize, w: B16, m: usize, out: Ptr, fp32: bool, part: Ptr, stream: cuda.Stream) !void {
        const b = &g.hc.bmm;
        const sk = b16SplitK(w.n, w.k);
        const i = b.pick(x, x_stride, w, m, fp32, sk) orelse return g.bmmCaptured(x, x_stride, w, m, out, fp32, part, stream);
        try b.run(i, x, x_stride, w, m, out, sk, stream);
        if (!b.check) return;
        // TF_GLM_PROMPT_CHECK: the captured kernels' output on the same call, byte for byte
        const bytes = m * w.n * @as(usize, if (fp32) 4 else 2);
        var ref = try cuda.DeviceBuffer.alloc(b.d, bytes);
        defer ref.free();
        try g.bmmCaptured(x, x_stride, w, m, ref.ptr, fp32, part, stream);
        try stream.synchronize();
        const gpa = std.heap.page_allocator;
        const want = try gpa.alloc(u8, bytes);
        defer gpa.free(want);
        const got = try gpa.alloc(u8, bytes);
        defer gpa.free(got);
        try ref.download(0, want);
        const view: cuda.DeviceBuffer = .{ .d = b.d, .ptr = out, .len = bytes };
        try view.download(0, got);
        var bad: usize = 0;
        for (want, got) |p, q| bad += @intFromBool(p != q);
        if (bad != 0) {
            std.log.err("glm53: bmm.cu {s} ({d} rows, n {d}, k {d}, sk {d}): {d} of {d} bytes differ from _bmm", .{ Bmm.kernels[i].name, m, w.n, w.k, sk, bad, bytes });
            return error.PromptKernelMismatch;
        }
        std.log.info("glm53: bmm.cu {s} ({d} rows, n {d}, k {d}, sk {d}): byte-equal to _bmm", .{ Bmm.kernels[i].name, m, w.n, w.k, sk });
    }

    /// `bmm` through the captured `_bmm` (and `_reduce`).
    fn bmmCaptured(g: Glue, x: Ptr, x_stride: usize, w: B16, m: usize, out: Ptr, fp32: bool, part: Ptr, stream: cuda.Stream) !void {
        const bm = bucket(@min(m, 128));
        const sk = b16SplitK(w.n, w.k);
        const seq = sk > 1 and m >= seq_rows and w.n >= 512 and w.k <= 4096;
        const out_ty = if (fp32) "*fp32" else "*bf16";
        const split = sk > 1 and !seq;
        try g.set.run(stream, "_bmm", .{ @intCast(cdiv(m, bm)), @intCast(cdiv(w.n, b16_bn)), @intCast(if (seq) 1 else sk) }, &.{
            aot.ptr("X", "*bf16", x),  aot.ptr("W", "*bf16", w.ptr), aot.ptr("OUT", out_ty, out),
            aot.ptr("PART", if (split) "*fp32" else out_ty, if (split) part else out),
            aot.int("M", n32(m)),      aot.int("x_stride", n32(x_stride)),
        }, &.{
            aot.ci("N", @intCast(w.n)), aot.ci("K", @intCast(w.k)), aot.ci("SK", @intCast(sk)), aot.ci("BM", @intCast(bm)),
            aot.ci("BLOCK_N", b16_bn),  aot.ci("BK", b16_bk),         aot.ci("F32", @intFromBool(fp32)), aot.ci("SEQ", @intFromBool(seq)),
        });
        if (!split) return;
        const total = m * w.n;
        try g.set.run(stream, "_reduce", .{ @intCast(cdiv(total, 1024)), 1, 1 }, &.{
            aot.ptr("PART", "*fp32", part), aot.ptr("OUT", out_ty, out), aot.int("total", n32(total)),
        }, &.{ aot.ci("SK", @intCast(sk)), aot.ci("BLOCK", 1024), aot.ci("F32", @intFromBool(fp32)) });
    }

    /// forward.py `_shift_conv`: each of `layers` conv windows [3, C] (layer stride conv_layer) takes rows keep ..
    /// keep + 2 of [window; the layer's projection rows] (proj layer stride proj_layer, row stride proj_row).
    pub fn convShift(g: Glue, conv: Ptr, proj: Ptr, keep: usize, layers: usize, c: usize, conv_layer: usize, proj_layer: usize, proj_row: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_conv_shift", .{ @intCast(layers), @intCast(cdiv(c, 1024)), 1 }, &.{
            aot.ptr("CONV", "*bf16", conv), aot.ptr("PROJ", "*bf16", proj), aot.int("keep", n32(keep)),
            aot.int("conv_layer", n32(conv_layer)), aot.int("proj_layer", n32(proj_layer)), aot.int("proj_row", n32(proj_row)),
        }, &.{ aot.ci("C", @intCast(c)), aot.ci("TAPS", 3), aot.ci("BLOCK", 1024) });
    }

    /// glue.py `router` (ROUTER_KS slices): out [m, ne] fp32 = x [m, D] times the router's bf16 rows, the slices'
    /// partials (`part`, ks m ne floats) added in slice order.
    pub fn router(g: Glue, x: Ptr, x_stride: usize, w: Ptr, ne: usize, m: usize, out: Ptr, part: Ptr, stream: cuda.Stream) !void {
        const bm: usize = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
        try g.set.run(stream, "_router_part", .{ @intCast(cdiv(m, bm)), @intCast(cdiv(ne, 32)), router_ks }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("W", "*bf16", w), aot.ptr("PART", "*fp32", part), aot.int("M", n32(m)), aot.int("x_stride", n32(x_stride)),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("NE", @intCast(ne)), aot.ci("BM", @intCast(bm)), aot.ci("BLOCK_E", 32), aot.ci("BK", 64), aot.ci("KS", router_ks) });
        const total = m * ne;
        try g.set.run(stream, "_router_sum", .{ @intCast(cdiv(total, 1024)), 1, 1 }, &.{
            aot.ptr("PART", "*fp32", part), aot.ptr("OUT", "*fp32", out), aot.int("total", n32(total)),
        }, &.{ aot.ci("KS", router_ks), aot.ci("BLOCK", 1024) });
    }

    /// glue.py `select`: each row's top-k experts by sigmoid score plus bias (lower id on ties), their weights
    /// normalized and scaled, then the shared expert (id `experts`, weight 1): pick int32 / wts fp32 [rows, k + 1].
    pub fn select(g: Glue, logits: Ptr, bias: Ptr, pick: Ptr, wts: Ptr, top_k: usize, experts: usize, scale: f32, norm: bool, rows: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_topk", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("L", "*fp32", logits), aot.ptr("BIAS", "*fp32", bias), aot.ptr("PICK", "*i32", pick), aot.ptr("WTS", "*fp32", wts), aot.float("scale", scale),
        }, &.{
            aot.ci("NE", @intCast(experts)),                                                      aot.ci("TOPK", @intCast(top_k)),
            aot.ci("SLOTS", @intCast(top_k + 1)),                                                 aot.ci("BLOCK", @intCast(std.math.ceilPowerOfTwoAssert(usize, experts + 1))),
            aot.ci("SLOTP", @intCast(std.math.ceilPowerOfTwoAssert(usize, top_k + 1))), aot.ci("NORM", @intFromBool(norm)),
        });
    }

    /// glue.py `combine_shared`: out [rows, D] fp32 = the routed slots of y [rows, slots, D] fp32 weighted, then the
    /// shared expert's row sy [rows, D] times its weight, added in slot order.
    pub fn combineShared(g: Glue, y: Ptr, sy: Ptr, wts: Ptr, out: Ptr, slots: usize, rows: usize, stream: cuda.Stream) !void {
        const block = @min(1024, g.hidden);
        try g.set.run(stream, "_combine_sy", .{ @intCast(rows), @intCast(g.hidden / block), 1 }, &.{
            aot.ptr("Y", "*fp32", y), aot.ptr("SY", "*fp32", sy), aot.ptr("WTS", "*fp32", wts), aot.ptr("OUT", "*fp32", out),
        }, &.{ aot.ci("D", @intCast(g.hidden)), aot.ci("SLOTS", @intCast(slots)), aot.ci("BLOCK", @intCast(block)) });
    }

    // -- DSA: the latent cache, the indexer and the attention (latent.py, sparse.py) ------------------------------

    /// latent.py `latent_write`: lat rows [R, 512] (row stride lat_stride) into cache rows pos .. pos + R - 1 (an FP8
    /// cache: each row's codes and scale, a function of the row alone).
    pub fn latWrite(g: Glue, lat: Ptr, lat_stride: usize, cache: Ptr, kv: Kv, pos: Ptr, R: usize, lw: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_lat_write", .{ @intCast(R), 1, 1 }, &.{
            aot.ptr("LAT", "*bf16", lat), aot.int("lat_stride", n32(lat_stride)), aot.ptr("LC", kv.values(), cache),
            aot.ptr("LS", kv.scales(), cache), aot.ptr("POS", "*i32", pos),
        }, &.{ aot.ci("LW", @intCast(lw)), aot.ci("RS", @intCast(kv.stride(lw))), aot.ci("FP8", kv.flag()) });
    }

    /// sparse.py `index_update`: the rows' LayerNormed index keys and their gates into the ring (row t at t % ring),
    /// then every pool of 4 the window completes into the pooled keys.
    /// `kv`: the pooled keys' format (the ring is bf16 either way).
    pub fn indexUpdate(g: Glue, kr: Ptr, k_stride: usize, gr: Ptr, ln_w: Ptr, ln_b: Ptr, ape: Ptr, ik: Ptr, ig: Ptr, ring: usize, pk: Ptr, kv: Kv, pos: Ptr, R: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_index_write", .{ @intCast(R), 1, 1 }, &.{
            aot.ptr("KR", "*bf16", kr), aot.int("k_stride", n32(k_stride)), aot.ptr("GR", "*fp32", gr), aot.ptr("LNW", "*bf16", ln_w),
            aot.ptr("LNB", "*bf16", ln_b), aot.ptr("IK", "*bf16", ik), aot.ptr("IG", "*bf16", ig), aot.ptr("POS", "*i32", pos),
            aot.int("RING", n32(ring)), aot.float("eps", 1e-6),
        }, &.{aot.ci("D", 128)});
        try g.set.run(stream, "_pool_keys", .{ @intCast(R / 4 + 2), 1, 1 }, &.{
            aot.ptr("IK", "*bf16", ik),     aot.ptr("IG", "*bf16", ig),   aot.ptr("APE", "*bf16", ape), aot.ptr("PK", kv.values(), pk),
            aot.ptr("PKS", kv.scales(), pk), aot.ptr("POS", "*i32", pos), aot.int("R", n32(R)),       aot.int("RING", n32(ring)),
        }, &.{ aot.ci("D", 128), aot.ci("RS", @intCast(kv.stride(128))), aot.ci("FP8", kv.flag()) });
    }

    /// latent.py `absorb_q` (bf16 kv_b): qa [R, H, LW] = q [R, H, D] @ wk [H, D, LW]; prompt rows on the tensor cores.
    pub fn absorbQ(g: Glue, q: Ptr, wk: Ptr, qa: Ptr, R: usize, H: usize, D: usize, lw: usize, prompt: bool, stream: cuda.Stream) !void {
        if (prompt) return g.set.run(stream, "_absorb_mma", .{ @intCast(H), @intCast(cdiv(R, 64)), @intCast(lw / 128) }, &.{
            aot.ptr("Q", "*bf16", q), aot.ptr("WK", "*bf16", wk), aot.ptr("QA", "*bf16", qa), aot.int("R", n32(R)),
        }, &.{ aot.ci("H", @intCast(H)), aot.ci("D", @intCast(D)), aot.ci("LW", @intCast(lw)), aot.ci("BM", 64), aot.ci("BN", 128), aot.ci("BK", 64) });
        try g.set.run(stream, "_absorb_q", .{ @intCast(H), @intCast(lw / 32), 1 }, &.{
            aot.ptr("Q", "*bf16", q), aot.ptr("WK", "*bf16", wk), aot.ptr("QA", "*bf16", qa), aot.int("R", n32(R)),
        }, &.{ aot.ci("H", @intCast(H)), aot.ci("D", @intCast(D)), aot.ci("LW", @intCast(lw)), aot.ci("BN", 32) });
    }

    /// latent.py `expand_v` (bf16 kv_b): out [R, H, DV] = ol [R, H, LW] @ wv [H, DV, LW]^T.
    pub fn expandV(g: Glue, ol: Ptr, wv: Ptr, out: Ptr, R: usize, H: usize, dv: usize, lw: usize, prompt: bool, stream: cuda.Stream) !void {
        if (prompt) return g.set.run(stream, "_expand_mma", .{ @intCast(H), @intCast(cdiv(R, 64)), @intCast(dv / 128) }, &.{
            aot.ptr("OL", "*bf16", ol), aot.ptr("WV", "*bf16", wv), aot.ptr("OUT", "*bf16", out), aot.int("R", n32(R)),
        }, &.{ aot.ci("H", @intCast(H)), aot.ci("DV", @intCast(dv)), aot.ci("LW", @intCast(lw)), aot.ci("BM", 64), aot.ci("BN", 128), aot.ci("BK", 64) });
        try g.set.run(stream, "_expand_v", .{ @intCast(H), @intCast(dv / 16), 1 }, &.{
            aot.ptr("OL", "*bf16", ol), aot.ptr("WV", "*bf16", wv), aot.ptr("OUT", "*bf16", out), aot.int("R", n32(R)),
        }, &.{ aot.ci("H", @intCast(H)), aot.ci("DV", @intCast(dv)), aot.ci("LW", @intCast(lw)), aot.ci("BN", 16) });
    }

    pub const Partials = struct { po: Ptr, pm: Ptr, pl: Ptr };

    /// latent.py `attention`: rows 0 .. R at device position pos (keys 0 .. pos + r visible to row r) over `nch`
    /// 512-key chunks into fp32 partials, then their merge in chunk order into out [R, H, LW].
    pub fn denseAttention(g: Glue, qa: Ptr, cache: Ptr, kv: Kv, pos: Ptr, p: Partials, out: Ptr, R: usize, H: usize, lw: usize, nch: usize, hb: usize, scale: f32, stream: cuda.Stream) !void {
        try g.set.run(stream, "_dense_chunks", .{ @intCast(R), @intCast(cdiv(H, hb)), @intCast(nch) }, &.{
            aot.ptr("QA", "*bf16", qa), aot.ptr("LC", kv.values(), cache), aot.ptr("LS", kv.scales(), cache), aot.ptr("POS", "*i32", pos),
            aot.ptr("PO", "*fp32", p.po), aot.ptr("PM", "*fp32", p.pm), aot.ptr("PL", "*fp32", p.pl), aot.int("R", n32(R)),
        }, &.{
            aot.ci("H", @intCast(H)),   aot.ci("LW", @intCast(lw)), aot.ci("CH", 512), aot.cf("SCALE", scale),
            aot.ci("HBT", @intCast(hb)), aot.ci("KTT", 32),          aot.ci("RS", @intCast(kv.stride(lw))), aot.ci("FP8", kv.flag()),
        });
        try g.merge(p, out, 0, R, H, lw, nch, false, stream);
    }

    /// latent.py `_merge`: each (row, head)'s chunk partials in chunk order -> out bf16; `sparse`: rows whose
    /// count is 0 left alone.
    pub fn merge(g: Glue, p: Partials, out: Ptr, counts: Ptr, R: usize, H: usize, lw: usize, nch: usize, sparse: bool, stream: cuda.Stream) !void {
        try g.set.run(stream, "_merge", .{ @intCast(R), @intCast(H), 1 }, &.{
            aot.ptr("PO", "*fp32", p.po),  aot.ptr("PM", "*fp32", p.pm), aot.ptr("PL", "*fp32", p.pl), aot.ptr("OUT", "*bf16", out),
            aot.ptr("CNT", "*i32", if (sparse) counts else p.pm), aot.int("R", n32(R)),
        }, &.{ aot.ci("H", @intCast(H)), aot.ci("LW", @intCast(lw)), aot.ci("NCH", @intCast(nch)), aot.ci("SPARSE", @intFromBool(sparse)) });
    }

    /// latent.py `sparse_attention`: rows with a count over their selected tokens (W a row), chunks of 512 tokens
    /// in list order, then the merge; other rows left alone.
    pub fn sparseAttention(g: Glue, qa: Ptr, cache: Ptr, kv: Kv, tok: Ptr, counts: Ptr, p: Partials, out: Ptr, R: usize, H: usize, lw: usize, w: usize, hb: usize, scale: f32, stream: cuda.Stream) !void {
        const nch = cdiv(w, 512);
        try g.set.run(stream, "_sparse_chunks", .{ @intCast(R), @intCast(cdiv(H, hb)), @intCast(nch) }, &.{
            aot.ptr("QA", "*bf16", qa), aot.ptr("LC", kv.values(), cache), aot.ptr("LS", kv.scales(), cache), aot.ptr("TOK", "*i32", tok),
            aot.ptr("CNT", "*i32", counts), aot.ptr("PO", "*fp32", p.po), aot.ptr("PM", "*fp32", p.pm), aot.ptr("PL", "*fp32", p.pl),
            aot.int("R", n32(R)),
        }, &.{
            aot.ci("W", @intCast(w)),   aot.ci("H", @intCast(H)), aot.ci("LW", @intCast(lw)), aot.ci("CH", 512), aot.cf("SCALE", scale),
            aot.ci("HBT", @intCast(hb)), aot.ci("KTT", 32),        aot.ci("RS", @intCast(kv.stride(lw))), aot.ci("FP8", kv.flag()),
        });
        try g.merge(p, out, counts, R, H, lw, nch, true, stream);
    }

    /// latent.py `sparse_onepass` (prompt rows): one online softmax a row over its selected tokens, straight to out.
    pub fn sparseOnepass(g: Glue, qa: Ptr, cache: Ptr, kv: Kv, tok: Ptr, counts: Ptr, out: Ptr, R: usize, H: usize, lw: usize, w: usize, scale: f32, stream: cuda.Stream) !void {
        try g.set.run(stream, "_sparse_onepass", .{ @intCast(R), @intCast(cdiv(H, 32)), 1 }, &.{
            aot.ptr("QA", "*bf16", qa),      aot.ptr("LC", kv.values(), cache), aot.ptr("LS", kv.scales(), cache), aot.ptr("TOK", "*i32", tok),
            aot.ptr("CNT", "*i32", counts), aot.ptr("OUT", "*bf16", out),
        }, &.{
            aot.ci("W", @intCast(w)), aot.ci("H", @intCast(H)), aot.ci("LW", @intCast(lw)), aot.cf("SCALE", scale), aot.ci("HBT", 32),
            aot.ci("KTT", 32),        aot.ci("SPLIT", 1),        aot.ci("STAGES", 1),          aot.ci("PREFETCH", 1),   aot.ci("RS", @intCast(kv.stride(lw))),
            aot.ci("FP8", kv.flag()),
        });
    }

    /// sparse.py `_scores_rows` (decode rows from device position pos): row r's scores over its visible pools,
    /// `grid` programs a row, into out [R, np] (row stride np).
    pub fn scoresRows(g: Glue, qi: Ptr, w: Ptr, w_stride: usize, pk: Ptr, kv: Kv, out: Ptr, pos: Ptr, R: usize, np: usize, grid: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_scores_rows", .{ @intCast(R), @intCast(grid), 1 }, &.{
            aot.ptr("QI", "*bf16", qi), aot.ptr("W", "*bf16", w),       aot.int("w_stride", n32(w_stride)), aot.ptr("PK", kv.values(), pk),
            aot.ptr("PKS", kv.scales(), pk), aot.ptr("OUT", "*fp32", out),  aot.int("o_stride", n32(np)),       aot.ptr("POS", "*i32", pos),
            aot.ptr("SPR", "*i32", pos), aot.ptr("BASE", "*i32", pos), aot.int("R", n32(R)),              aot.int("NP", n32(np)),
            aot.float("scale", index_scale), aot.float("wscale", index_wscale),
        }, &.{
            aot.ci("H", 32), aot.ci("HP", 32), aot.ci("D", 128), aot.ci("BP", 64), aot.ci("G", @intCast(grid)), aot.ci("SEG", 0),
            aot.ci("RS", @intCast(kv.stride(128))), aot.ci("FP8", kv.flag()), aot.ci("K", 512),
        });
    }

    /// sparse.py `_scores` (a prompt chunk's rows from device position pos, RB rows a program) into out [R, np].
    pub fn scores(g: Glue, qi: Ptr, w: Ptr, w_stride: usize, pk: Ptr, kv: Kv, out: Ptr, pos: Ptr, R: usize, np: usize, stream: cuda.Stream) !void {
        const rb = 4;
        try g.set.run(stream, "_scores", .{ @intCast(cdiv(R, rb)), @intCast(cdiv(np, 64)), 1 }, &.{
            aot.ptr("QI", "*bf16", qi),  aot.ptr("W", "*bf16", w),     aot.int("w_stride", n32(w_stride)), aot.ptr("PK", kv.values(), pk),
            aot.ptr("PKS", kv.scales(), pk), aot.ptr("OUT", "*fp32", out), aot.ptr("POS", "*i32", pos),        aot.int("R", n32(R)),
            aot.int("NP", n32(np)),      aot.float("scale", index_scale), aot.float("wscale", index_wscale),
        }, &.{ aot.ci("H", 32), aot.ci("HP", 32), aot.ci("D", 128), aot.ci("BP", 64), aot.ci("RB", rb), aot.ci("RS", @intCast(kv.stride(128))), aot.ci("FP8", kv.flag()) });
    }

    /// sparse.py `_select_rows`: each row's 512 best pools ascending (ties to the lower pool) into out int64 [R, 512];
    /// `pos` 0: every one of the row's np scores, else decode rows at pos + r over their visible bound.
    pub fn selectRows(g: Glue, s: Ptr, out: Ptr, np: usize, pos: Ptr, R: usize, stream: cuda.Stream) !void {
        const wide = R < 64 and np > 1024;
        try g.set.run(stream, "_select_rows", .{ @intCast(R), 1, 1 }, &.{
            aot.ptr("S", "*fp32", s), aot.ptr("OUT", "*i64", out), aot.int("NPS", n32(np)),
            if (pos != 0) aot.ptr("POS", "*i32", pos) else aot.ptr("POS", "*fp32", s),
        }, &.{ aot.ci("K", 512), aot.ci("BLOCK", if (wide) 4096 else 1024), aot.ci("VIS", @intFromBool(pos != 0)) });
    }

    /// sparse.py `_tokens`: each row's pools' tokens ascending, then its incomplete pool's visible tokens; its count
    /// past the dense limit (0 within it).
    pub fn tokens(g: Glue, pools: Ptr, pos: Ptr, tok: Ptr, cnt: Ptr, R: usize, stream: cuda.Stream) !void {
        try g.set.run(stream, "_tokens", .{ @intCast(R), 1, 1 }, &.{
            aot.ptr("POOLS", "*i64", pools), aot.ptr("POS", "*i32", pos), aot.ptr("TOK", "*i32", tok), aot.ptr("CNT", "*i32", cnt),
        }, &.{ aot.ci("W", sparse_tokens), aot.ci("K", 512), aot.ci("PL", 4), aot.ci("BLOCK", 1024) });
    }

    /// sparse.py `select_split` with tokens (decode rows of at least 8,192 scores): five passes of chunk programs,
    /// the selection written as select_tokens' tokens and counts. `scratch`: split_scratch's int32s.
    pub fn selectSplit(g: Glue, s: Ptr, np: usize, pos: Ptr, scratch_buf: cuda.DeviceBuffer, tok: Ptr, cnt: Ptr, R: usize, stream: cuda.Stream) !void {
        const scratch = scratch_buf.ptr;
        const chs: usize = if (np <= 16384) 2048 else 4096;
        const cp = std.math.ceilPowerOfTwoAssert(usize, cdiv(np, chs));
        const tot = scratch;
        const hist = scratch + R * 1024 * 4;
        const gt = scratch + R * (1024 + cp * 256) * 4;
        if (scratch_buf.len < splitScratchBytes(R, np)) return error.ScratchTooSmall;
        const zero: cuda.DeviceBuffer = .{ .d = scratch_buf.d, .ptr = tot, .len = R * 1024 * 4 };
        try zero.fill8(0, stream.handle);
        for (0..5) |step| try g.set.run(stream, "_select_split", .{ @intCast(R), @intCast(cp), 1 }, &.{
            aot.ptr("S", "*fp32", s),  aot.int("s_stride", n32(np)), aot.int("NPC", n32(np)),  aot.ptr("POS", "*i32", pos),
            aot.ptr("SPR", "*i32", scratch), aot.ptr("TOT", "*i32", tot), aot.ptr("HIST", "*i32", hist), aot.ptr("GT", "*i32", gt),
            aot.ptr("OUT", "*i32", scratch), aot.ptr("TOK", "*i32", tok), aot.ptr("CNT", "*i32", cnt),
        }, &.{
            aot.ci("W", sparse_tokens), aot.ci("K", 512), aot.ci("PL", 4), aot.ci("CHS", @intCast(chs)), aot.ci("CP", @intCast(cp)),
            aot.ci("STEP", @intCast(step)), aot.ci("SEG", 0), aot.ci("VIS", 1), aot.ci("TOKS", 1),
        });
    }

    /// sparse.py `select_split` without positions (top_pools of a prompt chunk's selection block of fewer than 64
    /// rows over 8,192 scores or more): each of the R rows' 512 best of its np scores (row stride np), ascending,
    /// ties to the lower pool, into out int64 [R, 512]: _select_rows' pools, across chunk programs.
    pub fn selectSplitPools(g: Glue, s: Ptr, np: usize, scratch_buf: cuda.DeviceBuffer, out: Ptr, R: usize, stream: cuda.Stream) !void {
        const scratch = scratch_buf.ptr;
        const chs: usize = if (np <= 16384) 2048 else 4096;
        const cp = std.math.ceilPowerOfTwoAssert(usize, cdiv(np, chs));
        const tot = scratch;
        const hist = scratch + R * 1024 * 4;
        const gt = scratch + R * (1024 + cp * 256) * 4;
        if (scratch_buf.len < splitScratchBytes(R, np)) return error.ScratchTooSmall;
        const zero: cuda.DeviceBuffer = .{ .d = scratch_buf.d, .ptr = tot, .len = R * 1024 * 4 };
        try zero.fill8(0, stream.handle);
        // the positions, the segment table, the tokens and the counts are unread here: the scratch stands in
        for (0..5) |step| try g.set.run(stream, "_select_split", .{ @intCast(R), @intCast(cp), 1 }, &.{
            aot.ptr("S", "*fp32", s),  aot.int("s_stride", n32(np)), aot.int("NPC", n32(np)),  aot.ptr("POS", "*i32", scratch),
            aot.ptr("SPR", "*i32", scratch), aot.ptr("TOT", "*i32", tot), aot.ptr("HIST", "*i32", hist), aot.ptr("GT", "*i32", gt),
            aot.ptr("OUT", "*i64", out), aot.ptr("TOK", "*i32", scratch), aot.ptr("CNT", "*i32", scratch),
        }, &.{
            aot.ci("W", sparse_tokens), aot.ci("K", 512), aot.ci("PL", 4), aot.ci("CHS", @intCast(chs)), aot.ci("CP", @intCast(cp)),
            aot.ci("STEP", @intCast(step)), aot.ci("SEG", 0), aot.ci("VIS", 0), aot.ci("TOKS", 0),
        });
    }
};

test "qmm.py's BF16 K slices for the rank's shapes" {
    // KDA projection [12576, 4096] at two ranks: 197 tiles, one slice; o_proj [4096, 4096]: 64 tiles, 4 slices
    try std.testing.expectEqual(@as(usize, 1), b16SplitK(12576, 4096));
    try std.testing.expectEqual(@as(usize, 4), b16SplitK(4096, 4096));
    // f_b / g_b [4096, 128]: two K steps, no split
    try std.testing.expectEqual(@as(usize, 1), b16SplitK(4096, 128));
    // the dense MLP's down [4096, 6144]: 4 slices; its gate | up [12288, 4096]: one
    try std.testing.expectEqual(@as(usize, 4), b16SplitK(4096, 6144));
    try std.testing.expectEqual(@as(usize, 1), b16SplitK(12288, 4096));
}

test "kv8.py's cache rows" {
    try std.testing.expectEqual(Kv.bf16, try Kv.parse(""));
    try std.testing.expectEqual(Kv.fp8, try Kv.parse("fp8"));
    try std.testing.expectError(error.BadKvFormat, Kv.parse("fp4"));
    // a latent row of 512 and a pooled key of 128 (a pool of 4 tokens): 12 indexed layers hold 13,056 bytes a
    // position in bf16, 6,768 in fp8
    try std.testing.expectEqual(@as(usize, 1024), Kv.bf16.rowBytes(512));
    try std.testing.expectEqual(@as(usize, 528), Kv.fp8.rowBytes(512));
    try std.testing.expectEqual(@as(usize, 144), Kv.fp8.rowBytes(128));
    try std.testing.expectEqual(@as(usize, 13_056), 12 * (Kv.bf16.rowBytes(512) + Kv.bf16.rowBytes(128) / 4));
    try std.testing.expectEqual(@as(usize, 6_768), 12 * (Kv.fp8.rowBytes(512) + Kv.fp8.rowBytes(128) / 4));
    // rows stay 16-byte aligned (kv8.parts)
    try std.testing.expectEqual(@as(usize, 0), Kv.fp8.rowBytes(512) % 16);
    try std.testing.expectEqual(@as(usize, 0), Kv.fp8.rowBytes(128) % 16);
}

fn bf16Bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

test "kv8.py's quantize_rows on the host" {
    // amax 3: scale 2^-6, codes of 64x: 1 -> 64 (0x68), -0.5 -> -32 (0xe0), 0, 3 -> 192 (0x74); 136 / 128 = 1.0625
    // ties to even (128: 0x70), 152 / 128 = 1.1875 ties up (160: 0x72); 2^-12 x 64 = 2^-6, the smallest normal (0x08);
    // 2^-15 x 64 = 2^-9, the smallest subnormal (0x01)
    const vals = [_]f32{ 1.0, -0.5, 0.0, 3.0, 136.0 / 64.0, 152.0 / 64.0, 0x1p-12, 0x1p-15 };
    var row: [8]u16 = undefined;
    for (vals, &row) |v, *b| b.* = bf16Bits(v);
    var out: [8 + Kv.pad]u8 = undefined;
    Kv.quantizeRow(&row, &out);
    try std.testing.expectEqualSlices(u8, &.{ 0x68, 0xe0, 0x00, 0x74, 0x70, 0x72, 0x08, 0x01 }, out[0..8]);
    try std.testing.expectEqual(@as(f32, 0x1p-6), std.mem.bytesToValue(f32, out[8..12]));
    try std.testing.expectEqualSlices(u8, &@as([12]u8, @splat(0)), out[12..]);
    // every code times the scale gives back a value within half a step of the row's (relative 2^-4 at most)
    for (vals, out[0..8]) |v, c| {
        const mag: f32 = blk: {
            const e: i32 = (c >> 3) & 0xF;
            const m: f32 = @floatFromInt(c & 7);
            break :blk if (e == 0) m * 0x1p-9 else (1 + m / 8) * std.math.ldexp(@as(f32, 1), e - 7);
        };
        const back = (if (c & 0x80 != 0) -mag else mag) * 0x1p-6;
        try std.testing.expect(@abs(back - v) <= @abs(v) / 16);
    }
    // a zero row: scale 2^-126, zero codes
    const zeros: [4]u16 = @splat(0);
    var zout: [4 + Kv.pad]u8 = undefined;
    Kv.quantizeRow(&zeros, &zout);
    try std.testing.expectEqual(@as(f32, 0x1p-126), std.mem.bytesToValue(f32, zout[4..8]));
    try std.testing.expectEqualSlices(u8, &@as([4]u8, @splat(0)), zout[0..4]);
}
