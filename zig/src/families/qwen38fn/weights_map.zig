//! Tensor names of a Qwen3.8-Flash-Next EXL3 pack and how TensorFold 0.6.6 (`qwen4_exp/cuda/exl3.py: load`) groups
//! them. An EXL3 group `{prefix}` is `.trellis int16 [K/16, N/16, 16 * bits]`, `.suh f16 [K]`, `.svh f16 [N]` and a
//! codebook marker (`.mul1`, `.mcg`, or neither for 3INST). Everything else is a plain tensor.
//!
//! Per DeltaNet layer: the projection is [in_proj_qkv | in_proj_z | (in_proj_b ; in_proj_a)] side by side (two EXL3
//! groups and one fp16 matrix of 2 * nv rows, b's rows first); each hyper-connection's down projection is the fp16
//! rows [input_mix_weight_down ; block_inject_weight] (the mixer has no inject rows); the router is the bf16 rows
//! [mlp.gate ; mlp.shared_expert_gate] (the shared expert's gate is row `experts`).
const std = @import("std");

pub const lm = "model.language_model.";

/// One name formatted into a caller's buffer.
pub const Name = struct {
    buf: [192]u8 = undefined,

    pub fn of(self: *Name, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&self.buf, fmt, args) catch unreachable; // every name fits 192 bytes
    }
};

pub const embed = lm ++ "embed_tokens.weight";
pub const head = "lm_head";
pub const mixer = lm ++ "hyper_connection_mixer";

pub const Hc = enum {
    attn,
    mlp,

    pub fn prefix(self: Hc) []const u8 {
        return switch (self) {
            .attn => "attn_hyper_connection",
            .mlp => "mlp_hyper_connection",
        };
    }
};

/// `model.language_model.layers.{layer}.{rest}`.
pub fn layer(n: *Name, i: usize, comptime rest: []const u8) []const u8 {
    return n.of(lm ++ "layers.{d}." ++ rest, .{i});
}

/// A hyper-connection's tensor `part` (input_mix_weight_down, input_mix_weight_up, block_inject_weight, hc_norm).
pub fn hc(n: *Name, i: usize, which: Hc, comptime piece: []const u8) []const u8 {
    return n.of(lm ++ "layers.{d}.{s}." ++ piece ++ ".weight", .{ i, which.prefix() });
}

/// The pieces of a DeltaNet layer's projection row, in row order.
pub const gdn_proj_groups = [_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z" };
pub const gdn_proj_f16 = [_][]const u8{ "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight" };

/// `prefix.part` of an EXL3 group.
pub fn part(n: *Name, prefix: []const u8, comptime p: []const u8) []const u8 {
    return n.of("{s}." ++ p, .{prefix});
}

/// The routed expert `e`'s projection (`gate_proj`, `up_proj`, `down_proj`) group prefix; e == experts is the shared one.
pub fn expert(n: *Name, i: usize, e: usize, experts: usize, comptime proj: []const u8) []const u8 {
    if (e == experts) return n.of(lm ++ "layers.{d}.mlp.shared_expert." ++ proj, .{i});
    return n.of(lm ++ "layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e });
}

test "names" {
    var n: Name = .{};
    try std.testing.expectEqualStrings("model.language_model.layers.0.linear_attn.A_log", layer(&n, 0, "linear_attn.A_log"));
    try std.testing.expectEqualStrings("model.language_model.layers.7.mlp_hyper_connection.hc_norm.weight", hc(&n, 7, .mlp, "hc_norm"));
    try std.testing.expectEqualStrings("model.language_model.layers.3.mlp.shared_expert.up_proj", expert(&n, 3, 512, 512, "up_proj"));
    try std.testing.expectEqualStrings("model.language_model.layers.3.mlp.experts.12.down_proj", expert(&n, 3, 12, 512, "down_proj"));
}
