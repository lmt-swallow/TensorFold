//! Which tensors of the EXL3 pack a rank loads and how each is split across the ranks (the Python family's
//! `weights.py: load_block`). EXL3 groups are `{prefix}.trellis int16 [K/16, N/16, 16*bits]`, `.suh f16 [K]`,
//! `.svh f16 [N]`; a split takes output columns [lo, hi) (N tiles hi/16, svh rows) or input rows [lo, hi) (K tiles,
//! suh rows), always on 128-wide Hadamard blocks so every part's bits are the full matrix's.
const std = @import("std");
const Config = @import("config.zig").Config;

pub const Split = union(enum) {
    whole,
    cols: [2]usize, // output columns [lo, hi)
    rows: [2]usize, // input rows [lo, hi)
};

pub const Kind = enum { exl3, bf16, f16, f32 };

/// One tensor (or EXL3 group) of a block: its name under the block prefix, how it is stored, this rank's part.
pub const Entry = struct {
    name: []const u8,
    kind: Kind,
    split: Split = .whole,
    /// For plain tensors: this rank's element range [lo, hi) of the first dimension when split (sink, experts).
    optional: bool = false,
};

pub const max_entries = 48;

pub const Block = struct {
    entries: [max_entries]Entry = undefined,
    len: usize = 0,

    fn add(self: *Block, e: Entry) void {
        self.entries[self.len] = e;
        self.len += 1;
    }

    pub fn slice(self: *const Block) []const Entry {
        return self.entries[0..self.len];
    }
};

/// Block `i` (backbone layers.i, or a DSpark stage mtp.j at i = n_layers + j) for `rank` of `world`, without the
/// experts, which `expertSplit` describes (there are n_routed + 1 of them, the shared expert last).
pub fn block(cfg: *const Config, i: usize, rank: usize, world: usize) Block {
    var b: Block = .{};
    const hl = cfg.n_heads / world * cfg.head_dim; // this rank's head columns of wq_b
    const gl = cfg.o_groups / world; // this rank's wo_a slices
    b.add(.{ .name = "hc_attn_fn", .kind = .f32 });
    b.add(.{ .name = "hc_attn_scale", .kind = .f32 });
    b.add(.{ .name = "hc_attn_base", .kind = .f32 });
    b.add(.{ .name = "hc_ffn_fn", .kind = .f32 });
    b.add(.{ .name = "hc_ffn_scale", .kind = .f32 });
    b.add(.{ .name = "hc_ffn_base", .kind = .f32 });
    b.add(.{ .name = "attn_norm.weight", .kind = .bf16 });
    b.add(.{ .name = "ffn_norm.weight", .kind = .bf16 });
    b.add(.{ .name = "attn.wq_a", .kind = .exl3 });
    b.add(.{ .name = "attn.q_norm.weight", .kind = .bf16 });
    b.add(.{ .name = "attn.wq_b", .kind = .exl3, .split = .{ .cols = .{ rank * hl, (rank + 1) * hl } } });
    b.add(.{ .name = "attn.wkv", .kind = .exl3 });
    b.add(.{ .name = "attn.kv_norm.weight", .kind = .bf16 });
    b.add(.{ .name = "attn.attn_sink", .kind = .f32, .split = .{ .rows = .{ rank * cfg.n_heads / world, (rank + 1) * cfg.n_heads / world } } });
    b.add(.{ .name = "attn.wo_b", .kind = .exl3, .split = .{ .rows = .{ rank * gl * cfg.o_rank, (rank + 1) * gl * cfg.o_rank } } });
    if (cfg.isKvSource(i)) {
        b.add(.{ .name = "attn.compressor.wkv", .kind = .exl3 });
        b.add(.{ .name = "attn.compressor.norm.weight", .kind = .bf16 });
        if (cfg.ratio(i) > 1) b.add(.{ .name = "attn.compressor.wgate", .kind = .exl3 });
    }
    if (cfg.isIndexSource(i)) {
        b.add(.{ .name = "attn.indexer.wq_b", .kind = .exl3 });
        b.add(.{ .name = "attn.indexer.weights_proj.weight", .kind = .f32 });
        if (cfg.isKvSource(i)) {
            b.add(.{ .name = "attn.indexer.wk", .kind = .exl3 });
            b.add(.{ .name = "attn.indexer.k_norm.weight", .kind = .bf16 });
        }
    }
    b.add(.{ .name = "ffn.gate.weight", .kind = .f16 });
    b.add(.{ .name = "ffn.gate.bias", .kind = .f32 });
    if (cfg.isEngram(i)) {
        const per = (cfg.engram_ngram - 1) * cfg.engram_heads / world * cfg.engram_dim;
        b.add(.{ .name = "engram.wkv", .kind = .exl3, .split = .{ .rows = .{ rank * per, (rank + 1) * per } } });
        b.add(.{ .name = "engram.q_weight", .kind = .f32 });
        b.add(.{ .name = "engram.k_weight", .kind = .f32 });
    }
    return b;
}

/// wo_a's slices this rank holds: `attn.wo_a.slice.{g}` for g in [lo, hi).
pub fn woASlices(cfg: *const Config, rank: usize, world: usize) [2]usize {
    const gl = cfg.o_groups / world;
    return .{ rank * gl, (rank + 1) * gl };
}

/// An expert's three groups: w1 (gate) and w3 (up) take this rank's output columns, w2 (down) the same input rows,
/// so the ranks' partial sums add up to the full expert.
pub const ExpertSplit = struct { gate_up: Split, down: Split };

pub fn expertSplit(inter: usize, rank: usize, world: usize) ExpertSplit {
    const il = inter / world;
    const r: [2]usize = .{ rank * il, (rank + 1) * il };
    return .{ .gate_up = .{ .cols = r }, .down = .{ .rows = r } };
}

/// The expert prefixes of block `p`: `p.ffn.experts.{e}` for e < n, then `p.ffn.shared_experts`.
pub fn expertPrefix(buf: []u8, p: []const u8, e: usize, n_routed: usize) []const u8 {
    if (e < n_routed) return std.fmt.bufPrint(buf, "{s}.ffn.experts.{d}", .{ p, e }) catch unreachable;
    return std.fmt.bufPrint(buf, "{s}.ffn.shared_experts", .{p}) catch unreachable;
}

/// Block `i`'s prefix: `layers.i` for the backbone, `mtp.j` for DSpark stage j = i - n_layers.
pub fn prefix(buf: []u8, cfg: *const Config, i: usize) []const u8 {
    if (i < cfg.n_layers) return std.fmt.bufPrint(buf, "layers.{d}", .{i}) catch unreachable;
    return std.fmt.bufPrint(buf, "mtp.{d}", .{i - cfg.n_layers}) catch unreachable;
}

/// The vocabulary columns of the head this rank holds (and the logits columns it contributes to the gather).
pub fn vocabRange(cfg: *const Config, rank: usize, world: usize) [2]usize {
    const per = cfg.vocab / world;
    return .{ rank * per, (rank + 1) * per };
}

test "layer 2 at rank 1 of 2 names the kv-source compressor and splits heads, groups and experts" {
    const cfgmod = @import("config.zig");
    var why: cfgmod.Why = .{};
    const text = @embedFile("testdata/config_mia.json");
    const cfg = try cfgmod.parse(std.testing.allocator, text, &why);
    const b = block(&cfg, 2, 1, 2);
    var saw_comp = false;
    var saw_gate = false;
    for (b.slice()) |e| {
        if (std.mem.eql(u8, e.name, "attn.compressor.wgate")) saw_comp = true;
        if (std.mem.eql(u8, e.name, "attn.wq_b")) try std.testing.expectEqual([2]usize{ 32 * 512, 64 * 512 }, e.split.cols);
        if (std.mem.eql(u8, e.name, "attn.wo_b")) try std.testing.expectEqual([2]usize{ 4 * 1024, 8 * 1024 }, e.split.rows);
        if (std.mem.eql(u8, e.name, "ffn.gate.weight")) saw_gate = true;
    }
    try std.testing.expect(saw_comp and saw_gate);
    try std.testing.expectEqual([2]usize{ 4, 8 }, woASlices(&cfg, 1, 2));
    const es = expertSplit(cfg.inter, 1, 2);
    try std.testing.expectEqual([2]usize{ 1152, 2304 }, es.gate_up.cols);
    try std.testing.expectEqual([2]usize{ 1152, 2304 }, es.down.rows);
    try std.testing.expectEqual([2]usize{ 64640, 129280 }, vocabRange(&cfg, 1, 2));
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("mtp.1", prefix(&buf, &cfg, 41));
    try std.testing.expectEqualStrings("layers.3.ffn.shared_experts", expertPrefix(&buf, "layers.3", 384, 384));
    // layer 1 holds Engram: this rank's 12 hash columns x 256 = 3072 input rows
    const b1 = block(&cfg, 1, 1, 2);
    var saw_engram = false;
    for (b1.slice()) |e| if (std.mem.eql(u8, e.name, "engram.wkv")) {
        saw_engram = true;
        try std.testing.expectEqual([2]usize{ 3072, 6144 }, e.split.rows);
    };
    try std.testing.expect(saw_engram);
}
