//! EXL3 linear layers on CUDA: the DSV4.1 fork's `cuda/exl3/linear.py` and `linear.cpp` in Zig over the fork's
//! kernels (zig/kernels/cuda/dsv41/linear.cu). y [M, N] = x [M, K] @ W + bias for 1 to 128 rows, each row's bits
//! its own: the K split and warps a program come from the layer's shape alone (`plan`), so a layer keeps one
//! reduction order whatever the row count.

const std = @import("std");
const cuda = @import("cuda");

pub const Codebook = enum(c_int) { inst3 = 0, mcg = 1, mul1 = 2 };
pub const DType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

pub const Error = error{ BadShape, BadWidth, OutOfMemory } || cuda.Error;

/// (K splits, warps a program) for a K x N layer (linear.py `plan`): the shape's alone.
pub const Plan = struct { sk: u32, wk: u32 };

pub fn plan(k: usize, n: usize) Plan {
    const blocks: usize = 192;
    const min_tiles: usize = 8;
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) { // a narrow layer needs K splits to fill the SMs at all
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ .sk = @intCast(sk), .wk = @intCast(wk) };
}

/// Trellis words [K/16, N/16, W] (W = 8 * bits 32-bit words a tile) -> strips [N/128, K/16, 8, W]: each 128-column
/// block's tiles in k order (linear.py `strips`). The caller frees the result.
pub fn packStrips(gpa: std.mem.Allocator, words: []const u32, kt: usize, nt: usize, w: usize) ![]u32 {
    if (words.len != kt * nt * w or nt % 8 != 0) return error.BadShape;
    const out = try gpa.alloc(u32, words.len);
    const nb = nt / 8;
    for (0..nb) |b| for (0..kt) |k| for (0..8) |t| {
        const src = ((k * nt) + (b * 8 + t)) * w;
        const dst = (((b * kt) + k) * 8 + t) * w;
        @memcpy(out[dst..][0..w], words[src..][0..w]);
    };
    return out;
}

/// K2 (= 2 * bits) from a trellis' last dimension in int16 words (16 * bits).
pub fn k2Of(w16: usize) Error!u32 {
    if (w16 % 8 != 0) return error.BadWidth;
    const k2 = w16 / 8;
    const known = [_]usize{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 };
    for (known) |v| if (v == k2) return @intCast(k2);
    return error.BadWidth;
}

/// The fatbin's kernels: one module, rot_in and the instantiation a layer's (K2, codebook, warps) picks.
pub const Kernels = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    rot_in: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_exl3_linear);
        errdefer module.unload();
        const rot_in = try module.function("_ZN14tf_exl3_linear13rot_in_kernelEPKviPK6__halfPS2_i");
        return .{ .d = d, .module = module, .rot_in = rot_in };
    }

    pub fn deinit(self: *Kernels) void {
        self.module.unload();
    }

    /// linear_kernel<K2, CB, WK> as cuobjdump lists it for the built fatbin.
    pub fn linear(self: *const Kernels, k2: u32, cb: Codebook, wk: u32) !cuda.Function {
        var buf: [160]u8 = undefined;
        const name = try symbol(&buf, "_ZN14tf_exl3_linear13linear_kernelILi{d}ELi{d}ELi{d}EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii", .{ k2, @intFromEnum(cb), wk });
        return self.module.function(name);
    }

    /// unpack_kernel<K2, CB>: W_q [K, N] fp16 from the words.
    pub fn unpack(self: *const Kernels, k2: u32, cb: Codebook) !cuda.Function {
        var buf: [160]u8 = undefined;
        const name = try symbol(&buf, "_ZN14tf_exl3_linear13unpack_kernelILi{d}ELi{d}EEEvPKjP6__halfill", .{ k2, @intFromEnum(cb) });
        return self.module.function(name);
    }
};

/// A mangled name formatted into `buf` with its terminator (cuModuleGetFunction wants a C string).
fn symbol(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]const u8 {
    const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args);
    buf[text.len] = 0;
    return buf[0..text.len :0];
}

/// One EXL3 layer on the GPU: strips of trellis words, fp16 suh [K] and svh [N], no bias (the DSV4.1 pack has none).
pub const Linear = struct {
    words: cuda.DeviceBuffer,
    suh: cuda.DeviceBuffer,
    svh: cuda.DeviceBuffer,
    counters: cuda.DeviceBuffer,
    k: usize,
    n: usize,
    k2: u32,
    cb: Codebook,
    sk: u32,
    wk: u32,
    kernel: cuda.Function,
    rot_in: cuda.Function,

    /// From a group's checkpoint tensors: trellis int16 [K/16, N/16, w16] raw bytes, suh / svh fp16 raw bytes.
    pub fn init(gpa: std.mem.Allocator, kernels: *const Kernels, trellis: []const u8, kt: usize, nt: usize, w16: usize, suh: []const u8, svh: []const u8, cb: Codebook) !Linear {
        const d = kernels.d;
        const k = 16 * kt;
        const n = 16 * nt;
        if (k % 128 != 0 or n % 128 != 0) return error.BadShape;
        if (trellis.len != kt * nt * w16 * 2 or suh.len != k * 2 or svh.len != n * 2) return error.BadShape;
        const k2 = try k2Of(w16);
        const w = w16 / 2;
        const stored = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(trellis)));
        const strips = try packStrips(gpa, stored, kt, nt, w);
        defer gpa.free(strips);
        var words = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(strips));
        errdefer words.free();
        var suh_d = try cuda.DeviceBuffer.fromHost(d, suh);
        errdefer suh_d.free();
        var svh_d = try cuda.DeviceBuffer.fromHost(d, svh);
        errdefer svh_d.free();
        var counters = try cuda.DeviceBuffer.alloc(d, 8 * (n / 128) * 4);
        errdefer counters.free();
        try counters.fill32(0, null);
        const p = plan(k, n);
        return .{
            .words = words,
            .suh = suh_d,
            .svh = svh_d,
            .counters = counters,
            .k = k,
            .n = n,
            .k2 = k2,
            .cb = cb,
            .sk = p.sk,
            .wk = p.wk,
            .kernel = try kernels.linear(k2, cb, p.wk),
            .rot_in = kernels.rot_in,
        };
    }

    pub fn deinit(self: *Linear) void {
        self.words.free();
        self.suh.free();
        self.svh.free();
        self.counters.free();
    }

    /// (words between k tiles, words between 128-column blocks) in `words` (strips).
    pub fn strides(self: *const Linear) struct { k: i64, nb: i64 } {
        const tw: i64 = 4 * @as(i64, self.k2);
        return .{ .k = 8 * tw, .nb = @as(i64, @intCast(self.k / 16)) * 8 * tw };
    }

    /// fp32 words of split-K scratch `z` a run of m rows needs (0 when the plan has one split).
    pub fn scratchWords(self: *const Linear, m: usize) usize {
        return if (self.sk > 1) self.sk * m * self.n else 0;
    }

    /// y [m, N] (y_dtype) = x [m, K] (x_dtype: fp16, bf16 or fp32) @ W; xh: fp16 scratch [m, K]; z: fp32 scratch of
    /// `scratchWords(m)` (any pointer when none is needed). Two launches on `stream`, as the Python `__call__`.
    pub fn run(self: *const Linear, x: cuda.abi.DevicePtr, x_dtype: DType, m: usize, xh: cuda.abi.DevicePtr, z: cuda.abi.DevicePtr, y: cuda.abi.DevicePtr, y_dtype: DType, stream: cuda.Stream) !void {
        if (m < 1 or m > 128) return error.BadShape;
        const K: c_int = @intCast(self.k);
        const N: c_int = @intCast(self.n);
        const M: c_int = @intCast(m);
        {
            var a: cuda.Args = .{};
            a.add(x);
            a.add(@intFromEnum(x_dtype));
            a.add(self.suh.ptr);
            a.add(xh);
            a.add(K);
            try cuda.launch.launch(self.rot_in, .{ .grid = .{ .x = @intCast((self.k / 128 + 3) / 4), .y = @intCast(m) }, .block = .{ .x = 128 } }, stream, &a);
        }
        {
            const s = self.strides();
            var a: cuda.Args = .{};
            a.add(xh);
            a.add(self.words.ptr);
            a.add(s.k);
            a.add(s.nb);
            a.add(self.svh.ptr);
            a.add(@as(u64, 0)); // bias
            a.add(y);
            a.add(@intFromEnum(y_dtype));
            a.add(if (self.sk > 1) z else @as(u64, 0));
            a.add(self.counters.ptr);
            a.add(M);
            a.add(K);
            a.add(N);
            a.add(@as(c_int, @intCast(self.sk)));
            const smem: u32 = @intCast(self.wk * @min(m, 8) * 128 * @sizeOf(f32));
            try cuda.launch.launch(self.kernel, .{ .grid = .{ .x = @intCast(self.n / 128), .y = self.sk }, .block = .{ .x = self.wk * 32 }, .shared = smem }, stream, &a);
        }
    }
};

test "plan follows linear.py for the DSV4.1 shapes" {
    try std.testing.expectEqual(Plan{ .sk = 8, .wk = 4 }, plan(5120, 1280)); // wq_a
    try std.testing.expectEqual(Plan{ .sk = 8, .wk = 4 }, plan(5120, 512)); // wkv
    try std.testing.expectEqual(Plan{ .sk = 1, .wk = 8 }, plan(1280, 32768)); // wq_b
    try std.testing.expectEqual(Plan{ .sk = 1, .wk = 8 }, plan(5120, 129280)); // head
}

test "strips put each 128-column block's tiles in k order" {
    const kt = 2;
    const nt = 16;
    const w = 1;
    var words: [kt * nt * w]u32 = undefined;
    for (&words, 0..) |*v, i| v.* = @intCast(i);
    const out = try packStrips(std.testing.allocator, &words, kt, nt, w);
    defer std.testing.allocator.free(out);
    // strip 1 (columns 128..255), k tile 1, tile 3 = stored (k=1, nt=11) = 1 * 16 + 11
    try std.testing.expectEqual(@as(u32, 27), out[((1 * kt + 1) * 8 + 3) * w]);
}
