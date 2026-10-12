//! DFlash2 Q4 projections. Row-parallel outputs stay FP32 until the TP reduction.
const cuda = @import("cuda");
const T = @import("cuda_mtp.zig");
const Ptr = cuda.abi.DevicePtr;

pub const Kernels = struct {
    module: cuda.Module,
    bf16: cuda.Function,
    fp32: cuda.Function,
    shared: u32,
    reduce_bf16: cuda.Function,
    reduce_fp32: cuda.Function,
    sum: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var m = try cuda.Module.load(d, cuda.kernels.glm53_qmm);
        errdefer m.unload();
        const global = try m.global("tf_glm53_qmm_smem");
        const view: cuda.DeviceBuffer = .{ .d = d, .ptr = global.ptr, .len = global.len };
        var shared: u64 = 0;
        try view.download(0, @import("std").mem.asBytes(&shared));
        var k: Kernels = .{
            .module = m,
            .sum = try m.function("tf_glm53_dflash_sum"),
            .bf16 = try m.function("_ZN12tf_glm53_qmm10qmm_kernelILi64ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii"),
            .fp32 = try m.function("_ZN12tf_glm53_qmm10qmm_kernelILi64ELi16ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii"),
            .shared = @intCast(shared),
            .reduce_bf16 = try m.function("_ZN12tf_glm53_qmm13reduce_kernelILb0EEEvPKfPvxi"),
            .reduce_fp32 = try m.function("_ZN12tf_glm53_qmm13reduce_kernelILb1EEEvPKfPvxi"),
        };
        try k.bf16.allowDynamicShared(k.shared);
        try k.fp32.allowDynamicShared(k.shared);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }

    /// gathered [2,count] FP32, rank zero first; out [count] BF16.
    pub fn sumRanks(k: Kernels, gathered: Ptr, out: Ptr, count: usize, stream: cuda.Stream) !void {
        if (count == 0 or count > 8 * 4096) return error.BadDflashProjection;
        var args: cuda.Args = .{};
        args.add(gathered);
        args.add(out);
        args.add(@as(c_int, @intCast(count)));
        try cuda.launch.launch(k.sum, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
    }

    /// x [rows,K] BF16, sums [rows,K/64] FP32; out is BF16 or FP32 as requested.
    pub fn matmul(k: Kernels, h: *const T.DraftHead, x: Ptr, sums: Ptr, out: Ptr, rows: usize, fp32: bool, partial: Ptr, stream: cuda.Stream) !void {
        if (rows == 0 or rows > 64 or h.k == 0 or h.k % 64 != 0 or h.n == 0) return error.BadDflashProjection;
        const sk = @import("dflash2.zig").splitK(h.n, h.k);
        if (sk > 1 and partial == 0) return error.MissingDflashPartials;
        const tiles = (rows + 15) / 16;
        const group = @max(1, @min(tiles, (12 << 20) / (16 * h.k * 2)));
        var args: cuda.Args = .{};
        args.add(x);
        args.add(sums);
        args.add(h.words.ptr);
        args.add(h.scales.ptr);
        args.add(h.biases.ptr);
        args.add(out);
        args.add(partial);
        args.add(@as(c_int, @intCast(rows)));
        args.add(@as(c_int, @intCast(h.n)));
        args.add(@as(c_int, @intCast(h.k)));
        args.add(@as(c_int, @intCast(sk)));
        args.add(@as(c_int, @intCast(h.npad)));
        args.add(@as(c_int, @intCast(h.k)));
        args.add(@as(c_int, @intCast(group)));
        try cuda.launch.launch(if (fp32) k.fp32 else k.bf16, .{
            .grid = .{ .x = @intCast(tiles * ((h.n + 63) / 64)), .z = @intCast(sk) },
            .block = .{ .x = 128 },
            .shared = k.shared,
        }, stream, &args);
        if (sk > 1) {
            var reduction: cuda.Args = .{};
            reduction.add(partial);
            reduction.add(out);
            reduction.add(@as(i64, @intCast(rows * h.n)));
            reduction.add(@as(c_int, @intCast(sk)));
            try cuda.launch.launch(if (fp32) k.reduce_fp32 else k.reduce_bf16, .{
                .grid = .{ .x = @intCast((rows * h.n + 255) / 256) },
                .block = .{ .x = 256 },
            }, stream, &reduction);
        }
    }
};
