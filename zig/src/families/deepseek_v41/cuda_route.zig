//! The MoE gate of a decode or verify window (rows <= 16), as the Python family routes it at its defaults
//! (`kernels.py: rowmm_gate` + `route`, switches "rowmm" and "rowmm_parts" on): the gate's logits as chunk sums
//! S [rows, K / 256, N] f32 (`_rowmm_parts`: a program a block of outputs and a 256-column chunk, every row of the
//! window in it), then `_route_parts` adds a row's chunks in K order (rowmm2's accumulation, so rowmm's bits),
//! scores sqrt(softplus), picks the top-k by score + bias (the lowest id among ties), normalizes the picked scores
//! (x route_scale) and puts the shared expert in the last slot with weight 1.
const std = @import("std");
const cuda = @import("cuda");

const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;

pub const chunk = 256; // BK
pub const max_rows = 16; // DECODE_ROWS

pub const Route = struct {
    set: *const aot.Set,
    k: u32, // the model width
    n: u32, // routed experts
    topk: u32,
    scale: f32, // routed_scaling_factor
    shared_id: i32, // the shared expert's id (the experts' count minus one)
    pdl: bool = true,

    pub fn init(set: *const aot.Set, k: u32, n: u32, topk: u32, scale: f32, shared_id: i32) Route {
        return .{ .set = set, .k = k, .n = n, .topk = topk, .scale = scale, .shared_id = shared_id };
    }

    pub fn slots(self: *const Route) u32 {
        return self.topk + 1;
    }

    pub fn chunks(self: *const Route) u32 {
        return self.k / chunk;
    }

    /// f32 words of the chunk sums `parts` for `rows`.
    pub fn partsWords(self: *const Route, rows: usize) usize {
        return rows * self.chunks() * self.n;
    }

    /// (BN, CPG, num_warps) a program by window rows (`_rp_tile`).
    fn tile(rows: u32) struct { bn: u32, cpg: u32 } {
        return if (rows <= 6) .{ .bn = 16, .cpg = 1 } else .{ .bn = 8, .cpg = 1 };
    }

    /// parts [rows, K / 256, N] = chunk sums of x [rows, K] bf16 . gate_w [N, K] fp16.
    pub fn gate(self: *const Route, stream: cuda.Stream, x: Ptr, gate_w: Ptr, parts: Ptr, rows: u32) !void {
        if (rows < 1 or rows > max_rows) return error.BadShape;
        const t = tile(rows);
        const kc = self.chunks();
        try self.set.run(stream, "_rowmm_parts", .{ (self.n + t.bn - 1) / t.bn, kc / t.cpg, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(self.k)), aot.ptr("W", "*fp16", gate_w), aot.ptr("S", "*fp32", parts), aot.int("rows", @intCast(rows)),
        }, &.{ aot.ci("K", self.k), aot.ci("N", self.n), aot.ci("BN", t.bn), aot.ci("BK", chunk), aot.ci("CPG", t.cpg), aot.ci("RB", rows), aot.ci("PDL", @intFromBool(self.pdl)) });
    }

    /// pick [rows, slots] i32 and wts [rows, slots] f32 from the chunk sums and the gate's bias [N] f32.
    pub fn pick(self: *const Route, stream: cuda.Stream, parts: Ptr, bias: Ptr, pick_: Ptr, wts: Ptr, rows: u32) !void {
        const nb = std.math.ceilPowerOfTwo(u32, self.n) catch unreachable;
        const sp = std.math.ceilPowerOfTwo(u32, self.slots()) catch unreachable;
        try self.set.run(stream, "_route_parts", .{ rows, 1, 1 }, &.{
            aot.ptr("S", "*fp32", parts), aot.ptr("BIAS", "*fp32", bias),   aot.ptr("PICK", "*i32", pick_),
            aot.ptr("WTS", "*fp32", wts), aot.float("scale", self.scale), aot.int("shared_id", self.shared_id),
        }, &.{ aot.ci("NE", self.n), aot.ci("NB", nb), aot.ci("TOPK", self.topk), aot.ci("SLOTS", self.slots()), aot.ci("SP", sp), aot.ci("KC", self.chunks()), aot.ci("PDL", @intFromBool(self.pdl)) });
    }
};
