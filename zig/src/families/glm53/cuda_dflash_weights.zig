//! DFlash2 rank weights: affine Q4 projections, BF16 normalization and CPU selector codebooks.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const D = @import("dflash2.zig");
const H = @import("dflash_weights.zig");
const Q = @import("draft_head.zig");
const T = @import("cuda_mtp.zig");
const Buf = cuda.DeviceBuffer;

fn upload(d: *const cuda.Driver, p: *Q.Packed) !T.DraftHead {
    var words = try Buf.fromHost(d, std.mem.sliceAsBytes(p.words));
    errdefer words.free();
    var scales = try Buf.fromHost(d, std.mem.sliceAsBytes(p.scales));
    errdefer scales.free();
    return .{ .words = words, .scales = scales, .biases = try Buf.fromHost(d, std.mem.sliceAsBytes(p.biases)), .n = p.n, .k = p.k, .npad = p.npad };
}

pub const Layer = struct {
    in_norm: Buf,
    post_norm: Buf,
    a_base: Buf,
    m_base: Buf,
    q_norm: Buf,
    k_norm: Buf,
    a_kp: T.DraftHead,
    m_kp: T.DraftHead,
    qkv: T.DraftHead,
    kv: T.DraftHead,
    o: T.DraftHead,
    gu: T.DraftHead,
    down: T.DraftHead,

    pub fn deinit(l: *Layer) void {
        inline for (.{ "in_norm", "post_norm", "a_base", "m_base", "q_norm", "k_norm" }) |field| @field(l, field).free();
        inline for (.{ "a_kp", "m_kp", "qkv", "kv", "o", "gu", "down" }) |field| @field(l, field).deinit();
    }
};

pub const Model = struct {
    a: std.mem.Allocator,
    fc: T.DraftHead,
    hidden_norm: Buf,
    norm: Buf,
    hproj: Buf,
    predecessor: []u16,
    successor: []u16,
    layers: [5]Layer,

    pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, checkpoint: *core.Checkpoint, c: D.Config, rank: usize) !Model {
        const l: H.Loader = .{ .a = a, .checkpoint = checkpoint, .config = c, .rank = rank };
        try l.validate();
        var m: Model = undefined;
        m.a = a;
        var p = try l.matrix("fc.weight", c.hidden_size, 5 * c.hidden_size, .whole);
        defer p.deinit(a);
        m.fc = try upload(d, &p);
        errdefer m.fc.deinit();
        m.hidden_norm = try Buf.fromHost(d, try l.tensor("hidden_norm.weight", &.{c.hidden_size}));
        errdefer m.hidden_norm.free();
        m.norm = try Buf.fromHost(d, try l.tensor("norm.weight", &.{c.hidden_size}));
        errdefer m.norm.free();
        m.hproj = try Buf.fromHost(d, try l.tensor("candidate_selector.hidden_projection.weight", &.{ c.dflash_config.selector_rank, c.hidden_size }));
        errdefer m.hproj.free();
        const code_shape = [_]usize{ c.vocab_size, c.dflash_config.selector_rank };
        m.predecessor = try H.share(a, try l.tensor("candidate_selector.predecessor_codebook", &code_shape), code_shape[0], code_shape[1], .whole, rank);
        errdefer a.free(m.predecessor);
        m.successor = try H.share(a, try l.tensor("candidate_selector.successor_codebook", &code_shape), code_shape[0], code_shape[1], .whole, rank);
        errdefer a.free(m.successor);
        var loaded: usize = 0;
        errdefer for (m.layers[0..loaded]) |*layer| layer.deinit();
        for (&m.layers, 0..) |*layer, i| {
            layer.* = try loadLayer(l, d, i);
            loaded += 1;
        }
        return m;
    }

    pub fn deinit(m: *Model) void {
        for (&m.layers) |*l| l.deinit();
        m.fc.deinit();
        m.hidden_norm.free();
        m.norm.free();
        m.hproj.free();
        m.a.free(m.predecessor);
        m.a.free(m.successor);
    }
};

fn loadLayer(l: H.Loader, d: *const cuda.Driver, i: usize) !Layer {
    var layer: Layer = undefined;
    var names: [3][128]u8 = undefined;
    const c = l.config;
    const h = c.hidden_size;
    const buffer_fields = .{ "in_norm", "post_norm", "a_base", "m_base", "q_norm", "k_norm" };
    const suffixes = .{ "input_layernorm.weight", "post_attention_layernorm.weight", "attention_conv.base_kernel", "mlp_conv.base_kernel", "self_attn.q_norm.weight", "self_attn.k_norm.weight" };
    var buffers: usize = 0;
    errdefer inline for (buffer_fields, 0..) |field, j| {
        if (j < buffers) @field(layer, field).free();
    };
    inline for (buffer_fields, suffixes, 0..) |field, suffix, j| {
        const shape: []const usize = if (j == 2 or j == 3) &.{ 2, 2, h } else if (j >= 4) &.{c.head_dim} else &.{h};
        @field(layer, field) = try Buf.fromHost(d, try l.tensor(try std.fmt.bufPrint(&names[0], "layers.{d}." ++ suffix, .{i}), shape));
        buffers += 1;
    }
    const matrices = .{ "a_kp", "m_kp", "o", "down" };
    const matrix_suffixes = .{ "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight" };
    var projections: usize = 0;
    errdefer inline for (matrices, 0..) |field, j| {
        if (j < projections) @field(layer, field).deinit();
    };
    inline for (matrices, matrix_suffixes, 0..) |field, suffix, j| {
        const n = if (j < 2) 4 * h / c.dflash_config.conv_group_size else h;
        const k = if (j < 2) h else if (j == 2) c.num_attention_heads * c.head_dim else c.intermediate_size;
        var p = try l.matrix(try std.fmt.bufPrint(&names[0], "layers.{d}." ++ suffix, .{i}), n, k, if (j < 2) .whole else .columns);
        defer p.deinit(l.a);
        @field(layer, field) = try upload(d, &p);
        projections += 1;
    }
    var qkv = try l.stack(&.{ try std.fmt.bufPrint(&names[0], "layers.{d}.self_attn.q_proj.weight", .{i}), try std.fmt.bufPrint(&names[1], "layers.{d}.self_attn.k_proj.weight", .{i}), try std.fmt.bufPrint(&names[2], "layers.{d}.self_attn.v_proj.weight", .{i}) }, &.{ c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim, c.num_key_value_heads * c.head_dim }, h);
    defer qkv.deinit(l.a);
    layer.qkv = try upload(d, &qkv);
    errdefer layer.qkv.deinit();
    var kv = try l.stack(&.{ try std.fmt.bufPrint(&names[0], "layers.{d}.self_attn.k_proj.weight", .{i}), try std.fmt.bufPrint(&names[1], "layers.{d}.self_attn.v_proj.weight", .{i}) }, &.{ c.num_key_value_heads * c.head_dim, c.num_key_value_heads * c.head_dim }, h);
    defer kv.deinit(l.a);
    layer.kv = try upload(d, &kv);
    errdefer layer.kv.deinit();
    var gu = try l.stack(&.{ try std.fmt.bufPrint(&names[0], "layers.{d}.mlp.gate_proj.weight", .{i}), try std.fmt.bufPrint(&names[1], "layers.{d}.mlp.up_proj.weight", .{i}) }, &.{ c.intermediate_size, c.intermediate_size }, h);
    defer gu.deinit(l.a);
    layer.gu = try upload(d, &gu);
    return layer;
}
