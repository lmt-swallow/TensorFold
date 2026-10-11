//! Tensor names of GLM-5.3-Flash's EXL3 pack (Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) and how the Mia fork
//! splits them over two ranks (`glm5_next/cuda/split.py: rule`, `tp.py`) and stacks them into the engine's matrices
//! (`weights.py: load`). The split decides which rank sums which terms, so it is the reference's or the partials'
//! sums (rank 0's first) change.
//!
//! Rules: `rep` every rank holds the tensor; `row` rank r holds its share of axis 0; `col` of axis 1 of a matrix;
//! `dim1` of axis 1 of an EXL3 gate/up trellis [K/16, N/16, 64] (its column tiles); `drop` the vision tower.
//! The head is `rep` on disk; weights.py then keeps the rank's vocabulary rows (`head_rows`).
//!
//! Stacked matrices (rows concatenated in this order, each part already the rank's share):
//!   KDA proj   q_proj | k_proj | v_proj | f_a_proj | g_a_proj | b_proj        [3 L 128 + 256 + L, 4096]
//!   KDA conv   q_conv1d | k_conv1d | v_conv1d                               [3 L 128, 4] (the [n, 1, 4] reshaped)
//!   DSA proj   q_a_proj | kv_a_proj_with_mqa                                [2048, 4096]
//!   indexer    indexer.wk | indexer.weights_proj                           [160, 4096]
//!   dense MLP  gate_proj | up_proj                                         [2 W, 4096]
//!   shared     shared_experts.gate_proj | shared_experts.up_proj           [2 W, 4096]
//! (L: the rank's KDA heads, W: the rank's share of the width.) kv_b_proj's rows are per head [256 key rows; 256
//! value rows] and go to the latent path's absorb (keys) and expand (values) matrices (latent.py `from_bf16`).
const std = @import("std");
const config = @import("config.zig");
const Config = config.Config;
const Family = config.Family;
const Layout = config.Layout;

pub const lm = "model.language_model.";
pub const head = "lm_head.weight";
pub const embed = lm ++ "embed_tokens.weight";
pub const norm = lm ++ "norm.weight";

/// One name formatted into a caller's buffer.
pub const Name = struct {
    buf: [192]u8 = undefined,

    pub fn of(self: *Name, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&self.buf, fmt, args) catch unreachable; // every name fits 192 bytes
    }
};

/// `model.language_model.layers.{i}.{rest}` (the MTP layer is `layers.{layers}`).
pub fn layer(n: *Name, i: usize, comptime rest: []const u8) []const u8 {
    return n.of(lm ++ "layers.{d}." ++ rest, .{i});
}

/// Routed expert `e`'s `proj` (gate_proj, up_proj, down_proj) tensor `part` (trellis, suh, svh, mcg).
pub fn expert(n: *Name, i: usize, e: usize, comptime proj: []const u8, comptime part: []const u8) []const u8 {
    return n.of(lm ++ "layers.{d}.mlp.experts.{d}." ++ proj ++ "." ++ part, .{ i, e });
}

pub const Rule = enum { rep, row, col, dim1, drop };

/// split.py `rule`: how two ranks share tensor `name` (null: no rule, the loader refuses the pack).
pub fn rule(name: []const u8) ?Rule {
    if (std.mem.startsWith(u8, name, "model.visual.")) return .drop;
    if (std.mem.indexOf(u8, name, ".mlp.experts.")) |_| {
        const tail = lastTwo(name);
        if (std.mem.eql(u8, tail[1], "mcg")) return .rep;
        const down = std.mem.eql(u8, tail[0], "down_proj");
        if (!down and !std.mem.eql(u8, tail[0], "gate_proj") and !std.mem.eql(u8, tail[0], "up_proj")) return null;
        if (std.mem.eql(u8, tail[1], "trellis")) return if (down) .row else .dim1;
        if (std.mem.eql(u8, tail[1], "suh")) return if (down) .row else .rep;
        if (std.mem.eql(u8, tail[1], "svh")) return if (down) .rep else .row;
        return null;
    }
    const sa = ".self_attn.";
    const rows = [_][]const u8{
        ".mlp.shared_experts.gate_proj.", ".mlp.shared_experts.up_proj.", ".mlp.gate_proj.", ".mlp.up_proj.",
        sa ++ "q_proj.",                  sa ++ "k_proj.",                sa ++ "v_proj.", sa ++ "q_conv1d.",
        sa ++ "k_conv1d.",                sa ++ "v_conv1d.",              sa ++ "f_b_proj.", sa ++ "g_b_proj.",
        sa ++ "b_proj.",                  sa ++ "q_b_proj.",              sa ++ "kv_b_proj.",
    };
    const cols = [_][]const u8{ ".mlp.shared_experts.down_proj.", ".mlp.down_proj.", sa ++ "o_proj." };
    for (rows) |p| if (std.mem.indexOf(u8, name, p) != null) return .row;
    for (cols) |p| if (std.mem.indexOf(u8, name, p) != null) return .col;
    if (std.mem.endsWith(u8, name, sa ++ "A_log") or std.mem.endsWith(u8, name, sa ++ "dt_bias")) return .row;
    const reps = [_][]const u8{
        "embed_tokens.",       "_layernorm.weight",   ".hc_attn_fn",    ".hc_attn_base",  ".hc_attn_scale",
        ".hc_ffn_fn",          ".hc_ffn_base",        ".hc_ffn_scale",  ".mlp.gate.weight", ".mlp.gate.e_score_correction_bias",
        sa ++ "indexer.",      sa ++ "q_a_proj.",     sa ++ "kv_a_proj_with_mqa.", sa ++ "f_a_proj.", sa ++ "g_a_proj.",
        sa ++ "o_norm.weight", ".eh_proj.",           ".enorm.weight",  ".hnorm.weight",  ".shared_head.norm.weight",
    };
    if (std.mem.startsWith(u8, name, "lm_head.") or std.mem.eql(u8, name, norm)) return .rep;
    for (reps) |p| if (std.mem.indexOf(u8, name, p) != null) return .rep;
    return null;
}

fn lastTwo(name: []const u8) [2][]const u8 {
    const b = std.mem.lastIndexOfScalar(u8, name, '.') orelse return .{ "", name };
    const a = std.mem.lastIndexOfScalar(u8, name[0..b], '.') orelse return .{ name[0..b], name[b + 1 ..] };
    return .{ name[a + 1 .. b], name[b + 1 ..] };
}

/// The layer index of `model.language_model.layers.{i}.…`, else null.
pub fn layerOf(name: []const u8) ?usize {
    const p = lm ++ "layers.";
    if (!std.mem.startsWith(u8, name, p)) return null;
    const rest = name[p.len..];
    const end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(usize, rest[0..end], 10) catch null;
}

/// tp.py `Layout.family`: the family a split tensor's axis holds (o_proj: KDA heads in a KDA layer, else MLA heads).
pub fn family(c: *const Config, name: []const u8) ?Family {
    if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) return .moe;
    if (std.mem.indexOf(u8, name, ".mlp.shared_experts.") != null) return .shared;
    if (std.mem.indexOf(u8, name, ".mlp.gate_proj.") != null or std.mem.indexOf(u8, name, ".mlp.up_proj.") != null or
        std.mem.indexOf(u8, name, ".mlp.down_proj.") != null) return .dense;
    if (std.mem.indexOf(u8, name, ".self_attn.q_b_proj.") != null or std.mem.indexOf(u8, name, ".self_attn.kv_b_proj.") != null) return .heads;
    if (std.mem.indexOf(u8, name, ".self_attn.o_proj.") != null) {
        const i = layerOf(name) orelse return null;
        return if (c.kind(i) == .kda) .lin else .heads;
    }
    if (std.mem.indexOf(u8, name, ".self_attn.") != null) return .lin;
    if (std.mem.startsWith(u8, name, "lm_head.")) return .vocab;
    return null;
}

/// The axis `r` splits and the rank's [start, stop) of it for a tensor of `shape` (null: replicated or dropped).
pub const Cut = struct { axis: usize, start: usize, stop: usize };

pub fn cut(l: Layout, name: []const u8, shape: []const usize) !?Cut {
    const r = rule(name) orelse return error.NoSplitRule;
    const axis: usize = switch (r) {
        .rep, .drop => return null,
        .row => 0,
        .col, .dim1 => 1,
    };
    if (shape.len <= axis) return error.BadSplitShape;
    const f = family(l.cfg, name) orelse return error.NoSplitFamily;
    const s = l.cut(f, shape[axis]);
    if (s[0] * l.total(f) != l.share(f)[0] * shape[axis] or s[1] * l.total(f) != l.share(f)[1] * shape[axis]) return error.BadSplitShape;
    return .{ .axis = axis, .start = s[0], .stop = s[1] };
}

/// The bytes rank `l` reads of a tensor (its share of the split axis, all of a replicated one, none of the vision
/// tower's; the head's vocabulary rows only, as weights.py keeps them).
pub fn rankBytes(l: Layout, name: []const u8, shape: []const usize, elem: usize) !usize {
    var n: usize = elem;
    for (shape) |d| n *= d;
    if (rule(name) == .drop) return 0;
    if (std.mem.eql(u8, name, head)) return n / l.world;
    const c = try cut(l, name, shape) orelse return n;
    return n / shape[c.axis] * (c.stop - c.start);
}

/// KDA projection parts in row order (weights.py `kda`), each `layers.{i}.self_attn.{part}.weight`.
pub const kda_proj = [_][]const u8{ "q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj" };
pub const kda_conv = [_][]const u8{ "q_conv1d", "k_conv1d", "v_conv1d" };
/// DSA projection parts (weights.py `dsa`: `stack([q_a_proj, kv_a_proj_with_mqa])`).
pub const dsa_proj = [_][]const u8{ "q_a_proj", "kv_a_proj_with_mqa" };
/// The indexer's stacked key and head-weight projections (`IndexW.kw`).
pub const index_proj = [_][]const u8{ "indexer.wk", "indexer.weights_proj" };

test "split.py's rules on every family of the pack" {
    const cases = [_]struct { []const u8, ?Rule }{
        .{ "model.language_model.layers.0.self_attn.q_proj.weight", .row },
        .{ "model.language_model.layers.0.self_attn.q_conv1d.weight", .row },
        .{ "model.language_model.layers.0.self_attn.f_a_proj.weight", .rep },
        .{ "model.language_model.layers.0.self_attn.f_b_proj.weight", .row },
        .{ "model.language_model.layers.0.self_attn.b_proj.weight", .row },
        .{ "model.language_model.layers.0.self_attn.A_log", .row },
        .{ "model.language_model.layers.0.self_attn.dt_bias", .row },
        .{ "model.language_model.layers.0.self_attn.o_norm.weight", .rep },
        .{ "model.language_model.layers.0.self_attn.o_proj.weight", .col },
        .{ "model.language_model.layers.3.self_attn.q_a_proj.weight", .rep },
        .{ "model.language_model.layers.3.self_attn.q_a_layernorm.weight", .rep },
        .{ "model.language_model.layers.3.self_attn.q_b_proj.weight", .row },
        .{ "model.language_model.layers.3.self_attn.kv_b_proj.weight", .row },
        .{ "model.language_model.layers.3.self_attn.indexer.wq_b.weight", .rep },
        .{ "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape", .rep },
        .{ "model.language_model.layers.0.mlp.gate_proj.weight", .row },
        .{ "model.language_model.layers.0.mlp.down_proj.weight", .col },
        .{ "model.language_model.layers.3.mlp.gate.weight", .rep },
        .{ "model.language_model.layers.3.mlp.gate.e_score_correction_bias", .rep },
        .{ "model.language_model.layers.3.mlp.shared_experts.up_proj.weight", .row },
        .{ "model.language_model.layers.3.mlp.shared_experts.down_proj.weight", .col },
        .{ "model.language_model.layers.3.mlp.experts.7.gate_proj.trellis", .dim1 },
        .{ "model.language_model.layers.3.mlp.experts.7.up_proj.suh", .rep },
        .{ "model.language_model.layers.3.mlp.experts.7.up_proj.svh", .row },
        .{ "model.language_model.layers.3.mlp.experts.7.down_proj.trellis", .row },
        .{ "model.language_model.layers.3.mlp.experts.7.down_proj.suh", .row },
        .{ "model.language_model.layers.3.mlp.experts.7.down_proj.svh", .rep },
        .{ "model.language_model.layers.3.mlp.experts.7.down_proj.mcg", .rep },
        .{ "model.language_model.layers.45.eh_proj.weight", .rep },
        .{ "model.language_model.layers.45.shared_head.norm.weight", .rep },
        .{ "model.language_model.layers.0.hc_attn_fn", .rep },
        .{ "model.language_model.embed_tokens.weight", .rep },
        .{ "model.language_model.norm.weight", .rep },
        .{ "lm_head.weight", .rep },
        .{ "model.visual.blocks.0.attn.qkv.weight", .drop },
        .{ "model.language_model.layers.0.self_attn.mystery", null },
    };
    for (cases) |c| try std.testing.expectEqual(c[1], rule(c[0]));
}

test "rank 1's cuts and bytes" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("testdata/config_exl3_4bpw.json"), &why);
    const r1: Layout = .{ .rank = 1, .world = 2, .cfg = &c };
    // a KDA layer's o_proj [4096, 8192]: KDA heads' columns
    try std.testing.expectEqual(Cut{ .axis = 1, .start = 4096, .stop = 8192 }, (try cut(r1, "model.language_model.layers.0.self_attn.o_proj.weight", &.{ 4096, 8192 })).?);
    // a DSA layer's o_proj [4096, 16384]: MLA heads' columns
    try std.testing.expectEqual(Cut{ .axis = 1, .start = 8192, .stop = 16384 }, (try cut(r1, "model.language_model.layers.3.self_attn.o_proj.weight", &.{ 4096, 16384 })).?);
    // b_proj [64, 4096]: one row a KDA head
    try std.testing.expectEqual(Cut{ .axis = 0, .start = 32, .stop = 64 }, (try cut(r1, "model.language_model.layers.0.self_attn.b_proj.weight", &.{ 64, 4096 })).?);
    // the gate trellis [256, 128, 64]: the expert width's column tiles
    try std.testing.expectEqual(Cut{ .axis = 1, .start = 64, .stop = 128 }, (try cut(r1, "model.language_model.layers.3.mlp.experts.0.gate_proj.trellis", &.{ 256, 128, 64 })).?);
    // the down trellis [128, 256, 64]: its K tiles
    try std.testing.expectEqual(Cut{ .axis = 0, .start = 64, .stop = 128 }, (try cut(r1, "model.language_model.layers.3.mlp.experts.0.down_proj.trellis", &.{ 128, 256, 64 })).?);
    try std.testing.expectEqual(@as(?Cut, null), try cut(r1, embed, &.{ 154880, 4096 }));
    try std.testing.expectEqual(@as(usize, 77440 * 4096 * 2), try rankBytes(r1, head, &.{ 154880, 4096 }, 2));
    try std.testing.expectEqual(@as(usize, 0), try rankBytes(r1, "model.visual.patch_embed.proj.weight", &.{ 1024, 1176 }, 2));
    try std.testing.expectEqual(@as(?usize, 45), layerOf("model.language_model.layers.45.enorm.weight"));
}
