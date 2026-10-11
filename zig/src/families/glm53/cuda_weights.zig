//! One rank's GLM-5.3-Flash tensors on the device, prepared as the Mia fork's `weights.py: load` prepares them in
//! the strict setting (TF_GLM_DENSE=bf16), byte for byte: the rank's share of each split tensor (weights_map.zig),
//! the stacked matrices' rows in weights.py's order, the KDA convs' [n, 1, 4] rows as [n, 4]. Every tensor of the
//! pack is BF16 / F32 already in the dtype the engine holds, so a share is copied as stored.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const map = @import("weights_map.zig");
const B16 = @import("cuda_glue.zig").B16;

const Checkpoint = core.Checkpoint;
const Tensor = core.checkpoint.Tensor;
const Ptr = cuda.abi.DevicePtr;

/// The pack's text tensors and its head, every shard mapped read-only.
pub const Pack = struct {
    text: Checkpoint,
    head: Checkpoint,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Pack {
        var text = try Checkpoint.openModelPrefix(gpa, io, dir, map.lm);
        errdefer text.close();
        return .{ .text = text, .head = try Checkpoint.openModelPrefix(gpa, io, dir, "lm_head.") };
    }

    pub fn close(self: *Pack) void {
        self.text.close();
        self.head.close();
    }

    pub fn get(self: *Pack, name: []const u8) !Tensor {
        return if (std.mem.startsWith(u8, name, "lm_head.")) self.head.get(name) else self.text.get(name);
    }
};

/// Builds one rank's device tensors from the pack: `share` copies the rank's part of a tensor (its split axis cut
/// by weights_map.cut), `stack` the rows of several into one buffer.
pub const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    pack: *Pack,
    lay: config.Layout,

    /// The rank's bytes of tensor `name` in host memory (the caller frees them): a row share is a contiguous range,
    /// a column share one strided range a row.
    pub fn hostShare(self: Loader, name: []const u8) ![]u8 {
        const t = try self.pack.get(name);
        const shape = t.shape[0..t.rank];
        const c = try map.cut(self.lay, name, shape) orelse return self.gpa.dupe(u8, t.bytes);
        const elem = t.dtype.size();
        var inner: usize = elem;
        for (shape[c.axis + 1 ..]) |dim| inner *= dim;
        const run = inner * (c.stop - c.start);
        var outer: usize = 1;
        for (shape[0..c.axis]) |dim| outer *= dim;
        const row = inner * shape[c.axis];
        const out = try self.gpa.alloc(u8, outer * run);
        for (0..outer) |o| @memcpy(out[o * run ..][0..run], t.bytes[o * row + c.start * inner ..][0..run]);
        return out;
    }

    /// The rank's share of `name` on the device.
    pub fn share(self: Loader, name: []const u8) !cuda.DeviceBuffer {
        const bytes = try self.hostShare(name);
        defer self.gpa.free(bytes);
        return cuda.DeviceBuffer.fromHost(self.d, bytes);
    }

    /// The rank's shares of `names` (each [rows_i, ...] with the same row width) one after another on the device.
    pub fn stack(self: Loader, names: []const []const u8) !cuda.DeviceBuffer {
        var parts: [8][]u8 = undefined;
        var n: usize = 0;
        defer for (parts[0..n]) |p| self.gpa.free(p);
        var total: usize = 0;
        for (names) |name| {
            parts[n] = try self.hostShare(name);
            total += parts[n].len;
            n += 1;
        }
        var buf = try cuda.DeviceBuffer.alloc(self.d, total);
        errdefer buf.free();
        var at: usize = 0;
        for (parts[0..n]) |p| {
            try buf.upload(at, p);
            at += p.len;
        }
        return buf;
    }

    /// A BF16 matrix: the rank's share of `name` (or the stack of `names`) with `k` columns.
    pub fn matrix(self: Loader, names: []const []const u8, k: usize) !Matrix {
        const buf = try self.stack(names);
        return .{ .buf = buf, .n = buf.len / 2 / k, .k = k };
    }
};

/// A BF16 matrix [n, k] the engine owns.
pub const Matrix = struct {
    buf: cuda.DeviceBuffer,
    n: usize,
    k: usize,

    pub fn b16(self: *const Matrix) B16 {
        return .{ .ptr = self.buf.ptr, .n = self.n, .k = self.k };
    }

    pub fn deinit(self: *Matrix) void {
        self.buf.free();
    }
};

/// A hyper-connection site (weights.py `HCW`): fn [24, S D] bf16, base [24] fp32, scale [3] fp32.
pub const Hc = struct {
    fn_: cuda.DeviceBuffer,
    base: cuda.DeviceBuffer,
    scale: cuda.DeviceBuffer,

    pub fn load(l: Loader, i: usize, comptime site: []const u8) !Hc {
        var n: map.Name = .{};
        var f = try l.share(n.of(map.lm ++ "layers.{d}.hc_" ++ site ++ "_fn", .{i}));
        errdefer f.free();
        var b = try l.share(n.of(map.lm ++ "layers.{d}.hc_" ++ site ++ "_base", .{i}));
        errdefer b.free();
        return .{ .fn_ = f, .base = b, .scale = try l.share(n.of(map.lm ++ "layers.{d}.hc_" ++ site ++ "_scale", .{i})) };
    }

    pub fn deinit(self: *Hc) void {
        self.fn_.free();
        self.base.free();
        self.scale.free();
    }
};

/// A KDA layer's rank share (weights.py `KDAW`).
pub const Kda = struct {
    proj: Matrix, // [q | k | v | f_a | g_a | b] [3 L 128 + 256 + L, D]
    fb: Matrix, // [L 128, 128]
    gb: Matrix,
    conv: cuda.DeviceBuffer, // [3 L 128, 4] bf16
    a_log: cuda.DeviceBuffer, // [L] fp32
    dt_bias: cuda.DeviceBuffer, // [L 128] fp32
    norm: cuda.DeviceBuffer, // [128] bf16
    o: Matrix, // [D, L 128]
    heads: usize,

    pub fn load(l: Loader, i: usize) !Kda {
        const c = l.lay.cfg;
        var nb: [8]map.Name = @splat(.{});
        const sa = map.lm ++ "layers.{d}.self_attn.";
        var proj = try l.matrix(&.{
            nb[0].of(sa ++ "q_proj.weight", .{i}),   nb[1].of(sa ++ "k_proj.weight", .{i}), nb[2].of(sa ++ "v_proj.weight", .{i}),
            nb[3].of(sa ++ "f_a_proj.weight", .{i}), nb[4].of(sa ++ "g_a_proj.weight", .{i}), nb[5].of(sa ++ "b_proj.weight", .{i}),
        }, c.hidden);
        errdefer proj.deinit();
        if (proj.n != l.lay.kdaProj()) return error.UnexpectedTensor;
        var fb = try l.matrix(&.{nb[0].of(sa ++ "f_b_proj.weight", .{i})}, c.lin_dim);
        errdefer fb.deinit();
        var gb = try l.matrix(&.{nb[0].of(sa ++ "g_b_proj.weight", .{i})}, c.lin_dim);
        errdefer gb.deinit();
        var conv = try l.stack(&.{ nb[0].of(sa ++ "q_conv1d.weight", .{i}), nb[1].of(sa ++ "k_conv1d.weight", .{i}), nb[2].of(sa ++ "v_conv1d.weight", .{i}) });
        errdefer conv.free();
        var a_log = try l.share(nb[0].of(sa ++ "A_log", .{i}));
        errdefer a_log.free();
        var dt = try l.share(nb[0].of(sa ++ "dt_bias", .{i}));
        errdefer dt.free();
        var norm = try l.share(nb[0].of(sa ++ "o_norm.weight", .{i}));
        errdefer norm.free();
        const heads = l.lay.size(.lin);
        const o = try l.matrix(&.{nb[0].of(sa ++ "o_proj.weight", .{i})}, heads * c.lin_dim);
        return .{ .proj = proj, .fb = fb, .gb = gb, .conv = conv, .a_log = a_log, .dt_bias = dt, .norm = norm, .o = o, .heads = heads };
    }

    pub fn deinit(self: *Kda) void {
        self.proj.deinit();
        self.fb.deinit();
        self.gb.deinit();
        self.conv.free();
        self.a_log.free();
        self.dt_bias.free();
        self.norm.free();
        self.o.deinit();
    }
};

/// A dense MLP's rank share (weights.py `MLPW`): [gate | up] rows and down's columns.
pub const Mlp = struct {
    gu: Matrix, // [2 W, D]
    down: Matrix, // [D, W]
    width: usize,

    pub fn load(l: Loader, comptime prefix: []const u8, i: usize) !Mlp {
        const c = l.lay.cfg;
        var nb: [2]map.Name = @splat(.{});
        var gu = try l.matrix(&.{ nb[0].of(map.lm ++ "layers.{d}." ++ prefix ++ "gate_proj.weight", .{i}), nb[1].of(map.lm ++ "layers.{d}." ++ prefix ++ "up_proj.weight", .{i}) }, c.hidden);
        errdefer gu.deinit();
        const width = gu.n / 2;
        return .{ .gu = gu, .down = try l.matrix(&.{nb[0].of(map.lm ++ "layers.{d}." ++ prefix ++ "down_proj.weight", .{i})}, width), .width = width };
    }

    pub fn deinit(self: *Mlp) void {
        self.gu.deinit();
        self.down.deinit();
    }
};

/// What every layer holds besides its mixer and FFN: the two hyper-connection sites and two norms.
pub const Frame = struct {
    attn_hc: Hc,
    ffn_hc: Hc,
    in_norm: cuda.DeviceBuffer,
    post_norm: cuda.DeviceBuffer,

    pub fn load(l: Loader, i: usize) !Frame {
        var a = try Hc.load(l, i, "attn");
        errdefer a.deinit();
        var f = try Hc.load(l, i, "ffn");
        errdefer f.deinit();
        var n: map.Name = .{};
        var in_norm = try l.share(map.layer(&n, i, "input_layernorm.weight"));
        errdefer in_norm.free();
        return .{ .attn_hc = a, .ffn_hc = f, .in_norm = in_norm, .post_norm = try l.share(map.layer(&n, i, "post_attention_layernorm.weight")) };
    }

    pub fn deinit(self: *Frame) void {
        self.attn_hc.deinit();
        self.ffn_hc.deinit();
        self.in_norm.free();
        self.post_norm.free();
    }
};
