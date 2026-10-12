//! GLM-5.3-Flash's KDA (Kimi delta attention) on CUDA: the Mia fork's `kda.cu` kernels (zig/kernels/cuda/glm53/kda.cu,
//! built into the glm53_kda fatbin with the extension's flags), launched as `kda.py` / `kda.cpp` launch them in the
//! strict setting: every window and prompt chunk takes the three-kernel path (`chain_wide`: TF_GLM_KDA_WIDE_ROWS 1,
//! TF_GLM_KDA_CHUNKED 0), and a commit that keeps fewer rows than ran replays every layer's state at once
//! (`replay_layers`).
//!
//! `chainWide` runs `rows` rows of one stream through one layer from the projection rows P (row stride p_stride:
//! [q | k | v | f_a | g_a | b] of this rank's H heads, b at b_off), the decay rows A and gate rows G (f_b and g_b
//! outputs, bf16), the conv window cs [3, 3 H 128] bf16 and the state `state_in` [H, 128, 128] fp32: out [rows,
//! H 128] bf16, the state after the last row in `state_out`, and each row's replay inputs (normalized k, v, decay,
//! beta) in the scratch, from which `replayLayers` rebuilds the state after any prefix with the same update routine.
const std = @import("std");
const cuda = @import("cuda");

const Ptr = cuda.abi.DevicePtr;

pub const dk = 128;
pub const dv = 128;

/// kda.cu kda_chain_wide_cuda's step kernel tile: 4 warps a block, 16 rows staged at a time.
const step_warps = 4;
/// kda_step.cu's step kernel: value rows a block (128 threads).
const step_rows_vr = 32;

pub const Kernels = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    prep: cuda.Function,
    step: cuda.Function,
    out: cuda.Function,
    replay_layers: cuda.Function,
    /// kda_step.cu (glm53_kda_step fatbin): step_kernel's read-outs and state, byte for byte, for any row count.
    /// TF_GLM_KDA_STEP=0 keeps step_kernel; TF_GLM_PROMPT_CHECK=1 runs step_kernel beside it on every call and fails
    /// unless the read-outs and the state are byte-equal.
    rows_mod: cuda.Module,
    step_rows: cuda.Function,
    use_rows: bool,
    check: bool,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.glm53_kda);
        errdefer module.unload();
        var rows_mod = try cuda.Module.load(d, cuda.kernels.glm53_kda_step);
        errdefer rows_mod.unload();
        return .{
            .d = d,
            .rows_mod = rows_mod,
            .step_rows = try rows_mod.function("tf_glm53_kda_step_rows"),
            .use_rows = !std.mem.eql(u8, envOr("TF_GLM_KDA_STEP", "1"), "0"),
            .check = std.mem.eql(u8, envOr("TF_GLM_PROMPT_CHECK", "0"), "1"),
            .module = module,
            .prep = try module.function("_ZN12tf_glm53_kda11prep_kernelEiPK13__nv_bfloat16iiS2_iS2_S2_PKfS4_fPfS5_PS0_S5_S5_"),
            .step = try module.function("_ZN12tf_glm53_kda11step_kernelILi4ELi16EEEviPKfS2_S2_PK13__nv_bfloat16S2_S2_iPS3_Pf"),
            .out = try module.function("_ZN12tf_glm53_kda10out_kernelEiPK13__nv_bfloat16S2_iS2_fPS0_"),
            .replay_layers = try module.function("_ZN12tf_glm53_kda20replay_layers_kernelEiPKfmS1_PK13__nv_bfloat16S1_S1_mmiPf"),
        };
    }

    pub fn deinit(self: *Kernels) void {
        self.rows_mod.unload();
        self.module.unload();
    }
};

fn envOr(name: [:0]const u8, default: []const u8) []const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return default);
    return if (v.len == 0) default else v;
}

/// One window's replay inputs for `layers` layers (kda.py `KDAScratchSet`), each layer's at a fixed stride:
/// out bf16 [layers, rows, H 128], k fp32 [layers, rows, H, 128], v bf16 [same], g fp32 [same], b fp32 [layers, rows, H].
pub const Scratch = struct {
    out: cuda.DeviceBuffer,
    k: cuda.DeviceBuffer,
    v: cuda.DeviceBuffer,
    g: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    layers: usize,
    rows: usize,
    heads: usize,

    pub fn init(d: *const cuda.Driver, layers: usize, rows: usize, heads: usize) !Scratch {
        const n = layers * rows * heads * dk;
        var out = try cuda.DeviceBuffer.alloc(d, n * 2);
        errdefer out.free();
        var k = try cuda.DeviceBuffer.alloc(d, n * 4);
        errdefer k.free();
        var v = try cuda.DeviceBuffer.alloc(d, n * 2);
        errdefer v.free();
        var g = try cuda.DeviceBuffer.alloc(d, n * 4);
        errdefer g.free();
        return .{ .out = out, .k = k, .v = v, .g = g, .b = try cuda.DeviceBuffer.alloc(d, layers * rows * heads * 4), .layers = layers, .rows = rows, .heads = heads };
    }

    pub fn deinit(self: *Scratch) void {
        self.out.free();
        self.k.free();
        self.v.free();
        self.g.free();
        self.b.free();
    }

    /// Layer `l`'s views: out, k, v, g, b.
    pub fn layer(self: *const Scratch, l: usize) struct { out: Ptr, k: Ptr, v: Ptr, g: Ptr, b: Ptr } {
        const n = self.rows * self.heads * dk;
        return .{
            .out = self.out.ptr + l * n * 2,
            .k = self.k.ptr + l * n * 4,
            .v = self.v.ptr + l * n * 2,
            .g = self.g.ptr + l * n * 4,
            .b = self.b.ptr + l * self.rows * self.heads * 4,
        };
    }
};

/// The wide path's shared temporaries (kda.py `_wide_scratch`): the normalized q fp32 and the read-out bf16 of up
/// to `rows` rows; every layer uses them in turn.
pub const Temp = struct {
    q: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    rows: usize,

    pub fn init(d: *const cuda.Driver, rows: usize, heads: usize) !Temp {
        var q = try cuda.DeviceBuffer.alloc(d, rows * heads * dk * 4);
        errdefer q.free();
        return .{ .q = q, .y = try cuda.DeviceBuffer.alloc(d, rows * heads * dv * 2), .rows = rows };
    }

    pub fn deinit(self: *Temp) void {
        self.q.free();
        self.y.free();
    }
};

pub const ChainArgs = struct {
    heads: usize,
    p: Ptr,
    p_stride: usize, // elements
    b_off: usize,
    a: Ptr,
    a_stride: usize,
    g: Ptr,
    g_stride: usize,
    conv_state: Ptr, // [3, 3 H 128] bf16
    conv_w: Ptr, // [3 H 128, 4] bf16
    state_in: Ptr,
    a_log: Ptr, // [H] fp32
    dt_bias: Ptr, // [H 128] fp32
    norm_w: Ptr, // [128] bf16
    eps: f32,
    lower: f32,
    rows: usize,
    out: Ptr, // [rows, H 128] bf16
    state_out: Ptr,
    k: Ptr,
    v: Ptr,
    gs: Ptr,
    b: Ptr,
};

fn int(v: usize) c_int {
    return @intCast(v);
}

/// kda.cpp `chain_wide` (kda_chain_wide_cuda): prep (row, head) blocks of 512, step (head, value part) blocks of 4
/// warps, out (row, head) blocks of 128.
pub fn chainWide(k: *const Kernels, t: *const Temp, a: ChainArgs, stream: cuda.Stream) !void {
    if (a.rows > t.rows) return error.ScratchTooSmall;
    const grid: cuda.launch.Dim3 = .{ .x = @intCast(a.rows), .y = @intCast(a.heads) };
    var prep: cuda.Args = .{};
    prep.add(int(a.heads));
    prep.add(a.p);
    prep.add(int(a.p_stride));
    prep.add(int(a.b_off));
    prep.add(a.a);
    prep.add(int(a.a_stride));
    prep.add(a.conv_state);
    prep.add(a.conv_w);
    prep.add(a.a_log);
    prep.add(a.dt_bias);
    prep.add(a.lower);
    prep.add(t.q.ptr);
    prep.add(a.k);
    prep.add(a.v);
    prep.add(a.gs);
    prep.add(a.b);
    try cuda.launch.launch(k.prep, .{ .grid = grid, .block = .{ .x = 512 } }, stream, &prep);
    var step: cuda.Args = .{};
    step.add(int(a.heads));
    step.add(a.state_in);
    step.add(t.q.ptr);
    step.add(a.k);
    step.add(a.v);
    step.add(a.gs);
    step.add(a.b);
    step.add(int(a.rows));
    step.add(t.y.ptr);
    step.add(a.state_out);
    if (k.use_rows) {
        try cuda.launch.launch(k.step_rows, .{ .grid = .{ .x = @intCast(a.heads), .y = dv / step_rows_vr }, .block = .{ .x = 128 } }, stream, &step);
        if (k.check) try checkStep(k, t, a, stream);
    } else {
        try cuda.launch.launch(k.step, .{ .grid = .{ .x = @intCast(a.heads), .y = dv / 4 / step_warps }, .block = .{ .x = step_warps * 32 } }, stream, &step);
    }
    var out: cuda.Args = .{};
    out.add(int(a.heads));
    out.add(t.y.ptr);
    out.add(a.g);
    out.add(int(a.g_stride));
    out.add(a.norm_w);
    out.add(a.eps);
    out.add(a.out);
    try cuda.launch.launch(k.out, .{ .grid = grid, .block = .{ .x = dv } }, stream, &out);
}

/// TF_GLM_PROMPT_CHECK: step_kernel on the same call into scratch, its read-outs and state against step_rows' (just
/// written to t.y and a.state_out), byte for byte.
fn checkStep(k: *const Kernels, t: *const Temp, a: ChainArgs, stream: cuda.Stream) !void {
    const yb = a.rows * a.heads * dv * 2;
    const sb = if (a.state_out != 0) a.heads * dv * dk * 4 else 0;
    var y = try cuda.DeviceBuffer.alloc(k.d, yb);
    defer y.free();
    var st = try cuda.DeviceBuffer.alloc(k.d, @max(sb, 4));
    defer st.free();
    var step: cuda.Args = .{};
    step.add(int(a.heads));
    step.add(a.state_in);
    step.add(t.q.ptr);
    step.add(a.k);
    step.add(a.v);
    step.add(a.gs);
    step.add(a.b);
    step.add(int(a.rows));
    step.add(y.ptr);
    step.add(if (sb != 0) st.ptr else @as(Ptr, 0));
    try cuda.launch.launch(k.step, .{ .grid = .{ .x = @intCast(a.heads), .y = dv / 4 / step_warps }, .block = .{ .x = step_warps * 32 } }, stream, &step);
    try stream.synchronize();
    const gpa = std.heap.page_allocator;
    const want = try gpa.alloc(u8, yb + sb);
    defer gpa.free(want);
    const got = try gpa.alloc(u8, yb + sb);
    defer gpa.free(got);
    try y.download(0, want[0..yb]);
    try (cuda.DeviceBuffer{ .d = k.d, .ptr = t.y.ptr, .len = yb }).download(0, got[0..yb]);
    if (sb != 0) {
        try st.download(0, want[yb..]);
        try (cuda.DeviceBuffer{ .d = k.d, .ptr = a.state_out, .len = sb }).download(0, got[yb..]);
    }
    var bad: usize = 0;
    for (want, got) |p, q| bad += @intFromBool(p != q);
    if (bad != 0) {
        std.log.err("glm53: kda step_rows ({d} rows, {d} heads): {d} of {d} bytes differ from step_kernel", .{ a.rows, a.heads, bad, want.len });
        return error.PromptKernelMismatch;
    }
    std.log.info("glm53: kda step_rows ({d} rows, {d} heads): byte-equal to step_kernel", .{ a.rows, a.heads });
}

/// kda.cpp `replay_layers`: every layer's state after the scratch's first `rows` rows, state_in / state_out
/// [layers, H, 128, 128] (one layer after another).
pub fn replayLayers(k: *const Kernels, state_in: Ptr, s: *const Scratch, rows: usize, state_out: Ptr, stream: cuda.Stream) !void {
    if (rows > s.rows) return error.ScratchTooSmall;
    var args: cuda.Args = .{};
    args.add(int(s.heads));
    args.add(state_in);
    args.add(@as(usize, s.heads * dv * dk));
    args.add(s.k.ptr);
    args.add(s.v.ptr);
    args.add(s.g.ptr);
    args.add(s.b.ptr);
    args.add(@as(usize, s.rows * s.heads * dk));
    args.add(@as(usize, s.rows * s.heads));
    args.add(int(rows));
    args.add(state_out);
    try cuda.launch.launch(k.replay_layers, .{ .grid = .{ .x = @intCast(s.layers * s.heads) }, .block = .{ .x = 1024 } }, stream, &args);
}
