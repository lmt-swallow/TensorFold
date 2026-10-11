//! Flash Next's Gated DeltaNet on CUDA: TensorFold 0.6.6's `gdn.cu` chain and replay kernels (built into the
//! qwen38fn_gdn fatbin with the extension's flags), launched as `gdn.py` / `gdn.cpp` launch them.
//!
//! `chain` runs a window of `rows` rows of one stream through one layer: rows r < rows of the projection P
//! [rows, PW] bf16 (row stride PW), over the conv window cs [taps - 1, C] bf16 and the state `state_in`
//! [nv, dv, dk] fp32, into out [rows, nv * dv] bf16, its 32-channel group sums xs fp32, the state after the last row
//! (`state_out`, may alias nothing else) and, when the scratch pointers are not null, each row's normalized k, v, g
//! and beta, from which `replay` rebuilds the state after any prefix of the window with the same update routine:
//! replay(state_in, scratch, keep) has the bits of chain's state after row keep - 1.
const std = @import("std");
const cuda = @import("cuda");

const Ptr = cuda.abi.DevicePtr;

pub const dk = 128;
pub const dv = 128;
pub const taps = 4;

pub const Kernels = struct {
    module: cuda.Module,
    chain_one: cuda.Function, // chain_kernel<16, 48, false>: one row, values read where used
    chain_ahead: cuda.Function, // chain_kernel<16, 48, true>: 2+ rows, a row's inputs loaded a row ahead
    replay: cuda.Function, // replay_kernel<16, 48>
    nk: usize,
    nv: usize,

    /// The kernels for nv value heads (48: one GPU; 24: a tensor-parallel rank's half).
    pub fn load(d: *const cuda.Driver, nv: usize) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.qwen38fn_gdn);
        errdefer module.unload();
        return switch (nv) {
            48 => .{
                .module = module,
                .chain_one = try module.function("_ZN10tf_qfn_gdn12chain_kernelILi16ELi48ELb0EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S3_fiPS1_PfS7_S7_S6_S7_S7_"),
                .chain_ahead = try module.function("_ZN10tf_qfn_gdn12chain_kernelILi16ELi48ELb1EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S3_fiPS1_PfS7_S7_S6_S7_S7_"),
                .replay = try module.function("_ZN10tf_qfn_gdn13replay_kernelILi16ELi48EEEvPKfS2_PK13__nv_bfloat16S2_S2_iPf"),
                .nk = 16,
                .nv = 48,
            },
            24 => .{
                .module = module,
                .chain_one = try module.function("_ZN10tf_qfn_gdn12chain_kernelILi8ELi24ELb0EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S3_fiPS1_PfS7_S7_S6_S7_S7_"),
                .chain_ahead = try module.function("_ZN10tf_qfn_gdn12chain_kernelILi8ELi24ELb1EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S3_fiPS1_PfS7_S7_S6_S7_S7_"),
                .replay = try module.function("_ZN10tf_qfn_gdn13replay_kernelILi8ELi24EEEvPKfS2_PK13__nv_bfloat16S2_S2_iPf"),
                .nk = 8,
                .nv = 24,
            },
            else => error.UnsupportedHeads,
        };
    }

    pub fn deinit(self: *Kernels) void {
        self.module.unload();
    }

    /// Conv channels (q | k | v) and the projection row width (qkv | z | b | a).
    pub fn widths(self: *const Kernels) struct { conv: usize, proj: usize } {
        const conv = 2 * self.nk * dk + self.nv * dv;
        return .{ .conv = conv, .proj = conv + self.nv * dv + 2 * self.nv };
    }
};

/// One window's per-row replay inputs (gdn.py `GDNScratch`): k fp32 [rows, nk, dk], v bf16 [rows, nv, dv],
/// g and beta fp32 [rows, nv].
pub const Scratch = struct {
    k: cuda.DeviceBuffer,
    v: cuda.DeviceBuffer,
    g: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    rows: usize,

    pub fn init(d: *const cuda.Driver, rows: usize, nk: usize, nv: usize) !Scratch {
        var k = try cuda.DeviceBuffer.alloc(d, rows * nk * dk * 4);
        errdefer k.free();
        var v = try cuda.DeviceBuffer.alloc(d, rows * nv * dv * 2);
        errdefer v.free();
        var g = try cuda.DeviceBuffer.alloc(d, rows * nv * 4);
        errdefer g.free();
        return .{ .k = k, .v = v, .g = g, .b = try cuda.DeviceBuffer.alloc(d, rows * nv * 4), .rows = rows };
    }

    pub fn deinit(self: *Scratch) void {
        self.k.free();
        self.v.free();
        self.g.free();
        self.b.free();
    }
};

pub const ChainArgs = struct {
    p: Ptr,
    conv_state: Ptr,
    conv_w: Ptr,
    state_in: Ptr,
    a_log: Ptr,
    dt_bias: Ptr,
    norm_w: Ptr,
    eps: f32,
    rows: usize,
    out: Ptr,
    xs: Ptr,
    state_out: Ptr, // 0: the state is not written
    scratch: ?*const Scratch, // null: no replay inputs kept
};

/// gdn.cpp `chain` (gdn_chain_cuda): one block of 1024 threads per value head.
pub fn chain(k: *const Kernels, a: ChainArgs, stream: cuda.Stream) !void {
    if (a.scratch) |s| if (s.rows < a.rows) return error.ScratchTooSmall;
    var args: cuda.Args = .{};
    args.add(a.p);
    args.add(a.conv_state);
    args.add(a.conv_w);
    args.add(a.state_in);
    args.add(a.a_log);
    args.add(a.dt_bias);
    args.add(a.norm_w);
    args.add(a.eps);
    args.add(@as(c_int, @intCast(a.rows)));
    args.add(a.out);
    args.add(a.xs);
    args.add(a.state_out);
    if (a.scratch) |s| {
        args.add(s.k.ptr);
        args.add(s.v.ptr);
        args.add(s.g.ptr);
        args.add(s.b.ptr);
    } else for (0..4) |_| args.add(@as(Ptr, 0));
    const f = if (a.rows > 1) k.chain_ahead else k.chain_one;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(k.nv) }, .block = .{ .x = 1024 } }, stream, &args);
}

/// gdn.cpp `replay` (gdn_replay_cuda): the state after the first `rows` rows of the window `s` recorded.
pub fn replay(k: *const Kernels, state_in: Ptr, s: *const Scratch, rows: usize, state_out: Ptr, stream: cuda.Stream) !void {
    if (rows > s.rows) return error.ScratchTooSmall;
    var args: cuda.Args = .{};
    args.add(state_in);
    args.add(s.k.ptr);
    args.add(s.v.ptr);
    args.add(s.g.ptr);
    args.add(s.b.ptr);
    args.add(@as(c_int, @intCast(rows)));
    args.add(state_out);
    try cuda.launch.launch(k.replay, .{ .grid = .{ .x = @intCast(k.nv) }, .block = .{ .x = 1024 } }, stream, &args);
}
