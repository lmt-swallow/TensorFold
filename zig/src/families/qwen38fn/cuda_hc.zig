//! A decode window's buffers (TensorFold 0.6.6 `state.py: Buffers`, the part a DeltaNet layer touches) and the
//! hyper-connection read-out over them (forward.py `_readout_plain` with an EXL3 pack's fp16 matrices): the
//! streams' norm, the down projection's K slices summed into SiLU and the inject gates, the up projection, the mix.
//! Every kernel treats each row alone, so a row of a window has the bits of the same row run alone.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const Glue = @import("cuda_glue.zig").Glue;

const Ptr = cuda.abi.DevicePtr;

pub const Window = struct {
    rows: usize,
    h: cuda.DeviceBuffer, // bf16 [rows, S * D]: the residual streams
    pss: cuda.DeviceBuffer, // fp32 [rows, D / 256, S]
    normed: cuda.DeviceBuffer, // bf16 [rows, S * D]
    xs_normed: cuda.DeviceBuffer, // fp32 [rows, S * D / 32]
    part: cuda.DeviceBuffer, // fp32 K slices of the F16 matrices (exl3_mm Scratch.part)
    act: cuda.DeviceBuffer, // bf16 [rows, low]
    xs_act: cuda.DeviceBuffer,
    inj_a: cuda.DeviceBuffer, // bf16 [rows, S]
    inj_m: cuda.DeviceBuffer,
    up: cuda.DeviceBuffer, // bf16 [rows, S * D]
    mixed: cuda.DeviceBuffer, // bf16 [rows, D]
    xs_mixed: cuda.DeviceBuffer,
    branch: cuda.DeviceBuffer, // bf16 [rows, D]
    proj: cuda.DeviceBuffer, // bf16 [rows, PW]: one DeltaNet layer's projection rows
    gout: cuda.DeviceBuffer, // bf16 [rows, nv * dv]
    gxs: cuda.DeviceBuffer,
    tmp: cuda.DeviceBuffer, // bf16 [rows, widest projection part: DeltaNet qkv, attention q with its gate] (exl3_mm Scratch.tmp)

    pub fn init(d: *const cuda.Driver, cfg: *const Config, rows: usize, part_floats: usize) !Window {
        var w: Window = undefined;
        w.rows = rows;
        const wide = cfg.wide();
        const sizes = [_]struct { *cuda.DeviceBuffer, usize }{
            .{ &w.h, rows * wide * 2 },                         .{ &w.pss, rows * (cfg.hidden / 256) * cfg.streams * 4 },
            .{ &w.normed, rows * wide * 2 },                    .{ &w.xs_normed, rows * wide / 32 * 4 },
            .{ &w.part, part_floats * 4 },                      .{ &w.act, rows * cfg.low * 2 },
            .{ &w.xs_act, rows * cfg.low / 32 * 4 },            .{ &w.inj_a, rows * cfg.streams * 2 },
            .{ &w.inj_m, rows * cfg.streams * 2 },              .{ &w.up, rows * wide * 2 },
            .{ &w.mixed, rows * cfg.hidden * 2 },               .{ &w.xs_mixed, rows * cfg.hidden / 32 * 4 },
            .{ &w.branch, rows * cfg.hidden * 2 },              .{ &w.proj, rows * cfg.projWidth() * 2 },
            .{ &w.gout, rows * cfg.nv * cfg.dv * 2 },           .{ &w.gxs, rows * cfg.nv * cfg.dv / 32 * 4 },
            .{ &w.tmp, rows * @max(@max(cfg.convDim(), cfg.nv * cfg.dv), cfg.heads * 2 * cfg.head_dim) * 2 },
        };
        var made: usize = 0;
        errdefer for (sizes[0..made]) |s| s[0].free();
        for (sizes) |s| {
            s[0].* = try cuda.DeviceBuffer.alloc(d, s[1]);
            made += 1;
        }
        return w;
    }

    pub fn deinit(self: *Window) void {
        for ([_]*cuda.DeviceBuffer{ &self.h, &self.pss, &self.normed, &self.xs_normed, &self.part, &self.act, &self.xs_act, &self.inj_a, &self.inj_m, &self.up, &self.mixed, &self.xs_mixed, &self.branch, &self.proj, &self.gout, &self.gxs, &self.tmp }) |b| b.free();
    }
};

/// forward.py `hc_block`: the pending branch written back into the streams, then the read-out of `hc` into
/// w.mixed / w.xs_mixed and (an inject hyper-connection) its gates into `inject_out` bf16 [rows, S].
pub fn block(g: Glue, w: *Window, hc: *const weights.Hc, cfg: *const Config, rows: usize, pending: Glue.Pending, inject_out: Ptr, stream: cuda.Stream) !void {
    try g.hcWriteback(w.h.ptr, w.pss.ptr, rows, pending, stream);
    try readout(g, w, hc, cfg, rows, if (hc.inject) inject_out else 0, stream);
}

/// forward.py `_readout_plain` for an EXL3 pack (fp16 down and up, a decode window: the down projection's slices
/// summed inside the activation).
pub fn readout(g: Glue, w: *Window, hc: *const weights.Hc, cfg: *const Config, rows: usize, inject_out: Ptr, stream: cuda.Stream) !void {
    return readoutOf(g, w, w.h.ptr, hc, cfg, rows, inject_out, stream);
}

/// `readout` of the streams at `h` (the head reads a copy of the window's streams).
pub fn readoutOf(g: Glue, w: *Window, h: Ptr, hc: *const weights.Hc, cfg: *const Config, rows: usize, inject_out: Ptr, stream: cuda.Stream) !void {
    const wide = cfg.wide();
    try g.hcNormed(h, w.pss.ptr, hc.scale.ptr, w.normed.ptr, w.xs_normed.ptr, cfg.eps, rows, stream);
    try g.f16mm(&hc.down, w.normed.ptr, wide, rows, w.part.ptr, 0, true, stream);
    try g.hcReduceAct(w.part.ptr, hc.down.sk, hc.down.n, w.act.ptr, w.xs_act.ptr, inject_out, rows, stream);
    try g.f16Call(&hc.up, w.act.ptr, cfg.low, rows, w.up.ptr, wide, w.part.ptr, stream);
    try g.hcMix(w.up.ptr, w.normed.ptr, w.mixed.ptr, w.xs_mixed.ptr, rows, stream);
}
