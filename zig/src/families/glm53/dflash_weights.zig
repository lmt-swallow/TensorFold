//! Shape-checked BF16 drafter checkpoint shares, independent of the target's quantization.
const std = @import("std");
const core = @import("core");
const D = @import("dflash2.zig");
const Q = @import("draft_head.zig");

pub const Cut = enum { whole, rows, columns };

/// Copy one TP share, preserving BF16 bits. Row and column cuts use contiguous halves.
pub fn share(a: std.mem.Allocator, bytes: []const u8, n: usize, k: usize, cut: Cut, rank: usize) ![]u16 {
    if (rank >= 2 or bytes.len != n * k * 2 or (cut == .rows and n % 2 != 0) or (cut == .columns and k % 2 != 0)) return error.BadDraftTensor;
    const rn = if (cut == .rows) n / 2 else n;
    const rk = if (cut == .columns) k / 2 else k;
    const out = try a.alloc(u16, rn * rk);
    for (0..rn) |row| for (0..rk) |col| {
        const src = (row + if (cut == .rows) rank * rn else 0) * k + col + if (cut == .columns) rank * rk else 0;
        out[row * rk + col] = std.mem.readInt(u16, bytes[2 * src ..][0..2], .little);
    };
    return out;
}

pub const Loader = struct {
    a: std.mem.Allocator,
    checkpoint: *core.Checkpoint,
    config: D.Config,
    rank: usize,

    pub fn tensor(l: Loader, name: []const u8, shape: []const usize) ![]const u8 {
        const t = try l.checkpoint.get(name);
        if (!t.is(.bf16, shape)) return error.BadDraftTensor;
        return t.bytes;
    }

    pub fn matrix(l: Loader, name: []const u8, n: usize, k: usize, cut: Cut) !Q.Packed {
        const b = try l.tensor(name, &.{ n, k });
        const rows = try share(l.a, b, n, k, cut, l.rank);
        defer l.a.free(rows);
        return Q.quantizePack(l.a, rows, if (cut == .rows) n / 2 else n, if (cut == .columns) k / 2 else k);
    }

    /// Stack separately sharded matrices in reference order, then quantize with group64 rounding.
    pub fn stack(l: Loader, names: []const []const u8, ns: []const usize, k: usize) !Q.Packed {
        if (names.len != ns.len or names.len > 3) return error.BadDraftTensor;
        var parts: [3][]u16 = undefined;
        var count: usize = 0;
        defer for (parts[0..count]) |part| l.a.free(part);
        var rows: usize = 0;
        for (names, ns) |name, n| {
            const b = try l.tensor(name, &.{ n, k });
            parts[count] = try share(l.a, b, n, k, .rows, l.rank);
            count += 1;
            rows += n / 2;
        }
        const all = try l.a.alloc(u16, rows * k);
        defer l.a.free(all);
        var at: usize = 0;
        for (parts[0..count]) |part| {
            @memcpy(all[at..][0..part.len], part);
            at += part.len;
        }
        return Q.quantizePack(l.a, all, rows, k);
    }

    /// Check every expected tensor before device allocations; no checkpoint payload is copied here.
    pub fn validate(l: Loader) !void {
        try l.config.validate(2);
        const c = l.config;
        const h = c.hidden_size;
        _ = try l.tensor("fc.weight", &.{ h, 5 * h });
        _ = try l.tensor("hidden_norm.weight", &.{h});
        _ = try l.tensor("norm.weight", &.{h});
        _ = try l.tensor("candidate_selector.hidden_projection.weight", &.{ c.dflash_config.selector_rank, h });
        inline for (.{ "predecessor", "successor" }) |side| _ = try l.tensor("candidate_selector." ++ side ++ "_codebook", &.{ c.vocab_size, c.dflash_config.selector_rank });
        for (0..c.num_hidden_layers) |i| {
            var buf: [128]u8 = undefined;
            inline for (.{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |suffix| _ = try l.tensor(try std.fmt.bufPrint(&buf, "layers.{d}." ++ suffix, .{i}), &.{h});
            inline for (.{ "attention_conv", "mlp_conv" }) |conv| {
                _ = try l.tensor(try std.fmt.bufPrint(&buf, "layers.{d}." ++ conv ++ ".base_kernel", .{i}), &.{ 2, 2, h });
                _ = try l.tensor(try std.fmt.bufPrint(&buf, "layers.{d}." ++ conv ++ ".kernel_projection.weight", .{i}), &.{ 4 * h / c.dflash_config.conv_group_size, h });
            }
            inline for (.{ "q", "k" }) |side| _ = try l.tensor(try std.fmt.bufPrint(&buf, "layers.{d}.self_attn." ++ side ++ "_norm.weight", .{i}), &.{c.head_dim});
            inline for (.{ "q", "k", "v" }) |side| _ = try l.tensor(try std.fmt.bufPrint(&buf, "layers.{d}.self_attn." ++ side ++ "_proj.weight", .{i}), &.{ (if (comptime std.mem.eql(u8, side, "q")) c.num_attention_heads else c.num_key_value_heads) * c.head_dim, h });
            _ = try l.tensor(try std.fmt.bufPrint(&buf, "layers.{d}.self_attn.o_proj.weight", .{i}), &.{ h, c.num_attention_heads * c.head_dim });
            inline for (.{ "gate", "up" }) |side| _ = try l.tensor(try std.fmt.bufPrint(&buf, "layers.{d}.mlp." ++ side ++ "_proj.weight", .{i}), &.{ c.intermediate_size, h });
            _ = try l.tensor(try std.fmt.bufPrint(&buf, "layers.{d}.mlp.down_proj.weight", .{i}), &.{ h, c.intermediate_size });
        }
    }
};

test "TP shares preserve bits and disjoint row and column coverage" {
    const a = std.testing.allocator;
    var raw: [32]u8 = undefined;
    for (0..16) |i| std.mem.writeInt(u16, raw[2 * i ..][0..2], @intCast(i), .little);
    for ([_]Cut{ .rows, .columns }) |cut| {
        const left = try share(a, &raw, 4, 4, cut, 0);
        defer a.free(left);
        const right = try share(a, &raw, 4, 4, cut, 1);
        defer a.free(right);
        var seen: [16]bool = @splat(false);
        for (left) |id| {
            try std.testing.expect(!seen[id]);
            seen[id] = true;
        }
        for (right) |id| {
            try std.testing.expect(!seen[id]);
            seen[id] = true;
        }
        for (seen) |yes| try std.testing.expect(yes);
    }
    try std.testing.expectError(error.BadDraftTensor, share(a, &raw, 4, 4, .rows, 2));
    try std.testing.expectError(error.BadDraftTensor, share(a, raw[0..30], 4, 4, .rows, 0));
}
