//! Every tensor the engine reads from the pack, with the dtype and shape weights.py expects of it, and what each rank
//! reads of it (weights_map.zig's split). `visit` lists them in load order (layers, then the MTP layer, then the
//! globals); a pack holds exactly these plus the vision tower, which the text engine never reads.
const std = @import("std");
const config = @import("config.zig");
const map = @import("weights_map.zig");
const Config = config.Config;

pub const DType = enum { bf16, f16, f32, i16, i32 };

pub const Want = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
};

/// Calls `f(ctx, want)` for every tensor of the pack the engine reads; `want.name` lives until `f` returns.
pub fn visit(c: *const Config, ctx: anytype, comptime f: fn (@TypeOf(ctx), Want) anyerror!void) !void {
    var n: map.Name = .{};
    const D = c.hidden;
    const S = c.streams;
    const mix = 2 * S + S * S; // hc mixing rows: pre, post, comb
    for (0..c.layers + @min(c.mtp_layers, 1)) |i| {
        const mtp = i == c.layers;
        if (!mtp) for ([_][]const u8{ "attn", "ffn" }) |site| {
            try f(ctx, .{ .name = n.of(map.lm ++ "layers.{d}.hc_{s}_fn", .{ i, site }), .dtype = .bf16, .shape = &.{ mix, S * D } });
            try f(ctx, .{ .name = n.of(map.lm ++ "layers.{d}.hc_{s}_base", .{ i, site }), .dtype = .f32, .shape = &.{mix} });
            try f(ctx, .{ .name = n.of(map.lm ++ "layers.{d}.hc_{s}_scale", .{ i, site }), .dtype = .f32, .shape = &.{3} });
        };
        try f(ctx, .{ .name = map.layer(&n, i, "input_layernorm.weight"), .dtype = .bf16, .shape = &.{D} });
        try f(ctx, .{ .name = map.layer(&n, i, "post_attention_layernorm.weight"), .dtype = .bf16, .shape = &.{D} });
        if (mtp) {
            try f(ctx, .{ .name = map.layer(&n, i, "enorm.weight"), .dtype = .bf16, .shape = &.{D} });
            try f(ctx, .{ .name = map.layer(&n, i, "hnorm.weight"), .dtype = .bf16, .shape = &.{D} });
            try f(ctx, .{ .name = map.layer(&n, i, "eh_proj.weight"), .dtype = .bf16, .shape = &.{ D, 2 * D } });
            try f(ctx, .{ .name = map.layer(&n, i, "shared_head.norm.weight"), .dtype = .bf16, .shape = &.{D} });
        }
        switch (c.kind(i)) {
            .kda => try kda(c, i, ctx, f, &n),
            .dsa => try dsa(c, i, ctx, f, &n),
        }
        switch (c.mlpKind(i)) {
            .dense => {
                try f(ctx, .{ .name = map.layer(&n, i, "mlp.gate_proj.weight"), .dtype = .bf16, .shape = &.{ c.dense_width, D } });
                try f(ctx, .{ .name = map.layer(&n, i, "mlp.up_proj.weight"), .dtype = .bf16, .shape = &.{ c.dense_width, D } });
                try f(ctx, .{ .name = map.layer(&n, i, "mlp.down_proj.weight"), .dtype = .bf16, .shape = &.{ D, c.dense_width } });
            },
            .moe => try moe(c, i, ctx, f, &n),
        }
    }
    try f(ctx, .{ .name = map.embed, .dtype = .bf16, .shape = &.{ c.vocab, D } });
    try f(ctx, .{ .name = map.norm, .dtype = .bf16, .shape = &.{D} });
    try f(ctx, .{ .name = map.head, .dtype = .bf16, .shape = &.{ c.vocab, D } });
}

fn kda(c: *const Config, i: usize, ctx: anytype, comptime f: anytype, n: *map.Name) !void {
    const D = c.hidden;
    const W = c.lin_heads * c.lin_dim;
    for ([_][]const u8{ "q", "k", "v" }) |x| {
        try f(ctx, .{ .name = n.of(map.lm ++ "layers.{d}.self_attn.{s}_proj.weight", .{ i, x }), .dtype = .bf16, .shape = &.{ W, D } });
        try f(ctx, .{ .name = n.of(map.lm ++ "layers.{d}.self_attn.{s}_conv1d.weight", .{ i, x }), .dtype = .bf16, .shape = &.{ W, 1, c.conv } });
    }
    for ([_][]const u8{ "f", "g" }) |x| {
        try f(ctx, .{ .name = n.of(map.lm ++ "layers.{d}.self_attn.{s}_a_proj.weight", .{ i, x }), .dtype = .bf16, .shape = &.{ c.lin_dim, D } });
        try f(ctx, .{ .name = n.of(map.lm ++ "layers.{d}.self_attn.{s}_b_proj.weight", .{ i, x }), .dtype = .bf16, .shape = &.{ W, c.lin_dim } });
    }
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.b_proj.weight"), .dtype = .bf16, .shape = &.{ c.lin_heads, D } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.A_log"), .dtype = .f32, .shape = &.{c.lin_heads} });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.dt_bias"), .dtype = .f32, .shape = &.{W} });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.o_norm.weight"), .dtype = .bf16, .shape = &.{c.lin_dim} });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.o_proj.weight"), .dtype = .bf16, .shape = &.{ D, W } });
}

fn dsa(c: *const Config, i: usize, ctx: anytype, comptime f: anytype, n: *map.Name) !void {
    const D = c.hidden;
    const H = c.heads;
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.q_a_proj.weight"), .dtype = .bf16, .shape = &.{ c.q_lora, D } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.q_a_layernorm.weight"), .dtype = .bf16, .shape = &.{c.q_lora} });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.kv_a_proj_with_mqa.weight"), .dtype = .bf16, .shape = &.{ c.kv_lora, D } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.kv_a_layernorm.weight"), .dtype = .bf16, .shape = &.{c.kv_lora} });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.q_b_proj.weight"), .dtype = .bf16, .shape = &.{ H * c.qk_dim, c.q_lora } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.kv_b_proj.weight"), .dtype = .bf16, .shape = &.{ H * (c.qk_dim + c.v_dim), c.kv_lora } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.o_proj.weight"), .dtype = .bf16, .shape = &.{ D, H * c.v_dim } });
    const I = c.index_heads;
    const d = c.index_dim;
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.indexer.wk.weight"), .dtype = .bf16, .shape = &.{ d, D } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.indexer.weights_proj.weight"), .dtype = .bf16, .shape = &.{ I, D } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.indexer.wq_b.weight"), .dtype = .bf16, .shape = &.{ I * d, c.q_lora } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.indexer.k_norm.weight"), .dtype = .bf16, .shape = &.{d} });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.indexer.k_norm.bias"), .dtype = .bf16, .shape = &.{d} });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.indexer.index_kpool_compress_gate"), .dtype = .bf16, .shape = &.{ d, D } });
    try f(ctx, .{ .name = map.layer(n, i, "self_attn.indexer.index_kpool_compress_ape"), .dtype = .bf16, .shape = &.{ c.kpool, d } });
}

fn moe(c: *const Config, i: usize, ctx: anytype, comptime f: anytype, n: *map.Name) !void {
    const D = c.hidden;
    try f(ctx, .{ .name = map.layer(n, i, "mlp.gate.weight"), .dtype = .bf16, .shape = &.{ c.experts, D } });
    try f(ctx, .{ .name = map.layer(n, i, "mlp.gate.e_score_correction_bias"), .dtype = .f32, .shape = &.{c.experts} });
    try f(ctx, .{ .name = map.layer(n, i, "mlp.shared_experts.gate_proj.weight"), .dtype = .bf16, .shape = &.{ c.shared_width, D } });
    try f(ctx, .{ .name = map.layer(n, i, "mlp.shared_experts.up_proj.weight"), .dtype = .bf16, .shape = &.{ c.shared_width, D } });
    try f(ctx, .{ .name = map.layer(n, i, "mlp.shared_experts.down_proj.weight"), .dtype = .bf16, .shape = &.{ D, c.shared_width } });
    const W = c.moe_width;
    for (0..c.experts) |e| {
        // trellis int16 [K/16, N/16, 16 * bits] (4-bit tiles), suh f16 [K], svh f16 [N], the codebook marker
        try f(ctx, .{ .name = map.expert(n, i, e, "gate_proj", "trellis"), .dtype = .i16, .shape = &.{ D / 16, W / 16, 64 } });
        try f(ctx, .{ .name = map.expert(n, i, e, "gate_proj", "suh"), .dtype = .f16, .shape = &.{D} });
        try f(ctx, .{ .name = map.expert(n, i, e, "gate_proj", "svh"), .dtype = .f16, .shape = &.{W} });
        try f(ctx, .{ .name = map.expert(n, i, e, "gate_proj", "mcg"), .dtype = .i32, .shape = &.{1} });
        try f(ctx, .{ .name = map.expert(n, i, e, "up_proj", "trellis"), .dtype = .i16, .shape = &.{ D / 16, W / 16, 64 } });
        try f(ctx, .{ .name = map.expert(n, i, e, "up_proj", "suh"), .dtype = .f16, .shape = &.{D} });
        try f(ctx, .{ .name = map.expert(n, i, e, "up_proj", "svh"), .dtype = .f16, .shape = &.{W} });
        try f(ctx, .{ .name = map.expert(n, i, e, "up_proj", "mcg"), .dtype = .i32, .shape = &.{1} });
        try f(ctx, .{ .name = map.expert(n, i, e, "down_proj", "trellis"), .dtype = .i16, .shape = &.{ W / 16, D / 16, 64 } });
        try f(ctx, .{ .name = map.expert(n, i, e, "down_proj", "suh"), .dtype = .f16, .shape = &.{W} });
        try f(ctx, .{ .name = map.expert(n, i, e, "down_proj", "svh"), .dtype = .f16, .shape = &.{D} });
        try f(ctx, .{ .name = map.expert(n, i, e, "down_proj", "mcg"), .dtype = .i32, .shape = &.{1} });
    }
}

pub fn size(d: DType) usize {
    return switch (d) {
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
    };
}

/// Bytes rank `l` reads by part of the model, for the startup estimate and the validator's report.
pub const Bytes = struct {
    total: usize = 0,
    experts: usize = 0, // EXL3 routed experts
    dense: usize = 0, // BF16 matrices of the layers (projections, MLPs, shared experts, routers, hc, norms)
    globals: usize = 0, // embedding, final norm, the rank's head rows

    pub fn add(b: *Bytes, l: config.Layout, w: Want) !void {
        const n = try map.rankBytes(l, w.name, w.shape, size(w.dtype));
        b.total += n;
        if (std.mem.indexOf(u8, w.name, ".mlp.experts.") != null) {
            b.experts += n;
        } else if (map.layerOf(w.name) != null) {
            b.dense += n;
        } else b.globals += n;
    }
};

test "every wanted tensor has a split rule, and rank bytes add up to the pack's text tensors" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("testdata/config_exl3_4bpw.json"), &why);
    var count: usize = 0;
    var ranks: [2]Bytes = .{ .{}, .{} };
    const Ctx = struct { count: *usize, ranks: *[2]Bytes, c: *const Config };
    try visit(&c, Ctx{ .count = &count, .ranks = &ranks, .c = &c }, struct {
        fn f(x: Ctx, w: Want) !void {
            try std.testing.expect(map.rule(w.name) != null);
            x.count.* += 1;
            for (x.ranks, 0..) |*b, r| try b.add(.{ .rank = r, .world = 2, .cfg = x.c }, w);
        }
    }.f);
    // the pack's headers: 150,226 tensors, 347 of them the vision tower's
    try std.testing.expectEqual(@as(usize, 150226 - 347), count);
    // two ranks read the split tensors once between them: the replicated ones twice, the head once
    try std.testing.expectEqual(ranks[0].total, ranks[1].total);
    // a routed expert on one rank: half of each trellis, both suh of gate / up and down's svh, half the rest,
    // the three codebook markers; 42 MoE layers and the MTP layer of 288 experts
    const expert = 3 * (256 * 64 * 64 * 2) + 2 * 4096 * 2 + 4096 * 2 + 3 * 1024 * 2 + 3 * 4;
    try std.testing.expectEqual(@as(usize, 43 * 288 * expert), ranks[0].experts);
}
