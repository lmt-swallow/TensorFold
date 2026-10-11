//! Qwen3.8-Flash-Next text dimensions from an EXL3 pack's config.json (`text_config`), the fields TensorFold
//! 0.6.6's `qwen4_exp/cuda/weight_types.py: Config.read` takes, checked against what the CUDA kernels serve.
//! Refused before any allocation: a pack that is not EXL3, a rope that is not the default (YaRN changes every
//! output and no bit-exact reference serves it), and shapes the kernels are not built for.
const std = @import("std");

pub const max_layers = 64;

pub const Kind = enum { linear, attention };

/// The reason a config was refused, for the caller's message.
pub const Why = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Config = struct {
    hidden: usize,
    layers: usize,
    kinds: [max_layers]Kind = undefined,
    vocab: usize,
    eps: f32,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    nk: usize, // DeltaNet key heads
    nv: usize, // DeltaNet value heads
    dk: usize,
    dv: usize,
    conv_kernel: usize,
    experts: usize,
    topk: usize,
    moe_width: usize,
    shared_width: usize,
    streams: usize, // hyper-connection streams (hc_count)
    low: usize, // hyper-connection low rank (hc_lowrank)
    index_heads: usize,
    index_dim: usize,
    index_budget: usize,
    index_ratio: usize,
    ple_layer: ?usize, // the zero-indexed decoder layer carrying the n-gram embedding
    max_positions: usize,
    bits_x100: usize, // the pack's average bits per weight, x100 (305 for 3.05bpw)
    head_bits: usize,
    mtp_layers: usize,
    ngram: usize, // ngram_size
    heads_per_ngram: usize,
    ngram_base: u64, // ngram_vocab_size_base
    ngram_shards: usize, // split_ngram_parts
    seed: u64,
    ple_kernel: usize,
    ple_eos: i64, // the text config's eos_token_id (the n-gram's segment break)

    /// Channels of the DeltaNet depthwise conv: q | k | v.
    pub fn convDim(c: *const Config) usize {
        return 2 * c.nk * c.dk + c.nv * c.dv;
    }

    /// Width of one DeltaNet projection row: qkv | z | b | a (gdn.cu's PW).
    pub fn projWidth(c: *const Config) usize {
        return c.convDim() + c.nv * c.dv + 2 * c.nv;
    }

    /// Width of the residual streams a row carries.
    pub fn wide(c: *const Config) usize {
        return c.streams * c.hidden;
    }

    /// Slots a row's MoE output holds: the routed experts, then the shared expert.
    pub fn slots(c: *const Config) usize {
        return c.topk + 1;
    }

    /// The index of decoder layer `i` among the DeltaNet layers (its slot in the recurrent state), or null.
    pub fn linearIndex(c: *const Config, i: usize) ?usize {
        if (i >= c.layers or c.kinds[i] != .linear) return null;
        var n: usize = 0;
        for (c.kinds[0..i]) |k| n += @intFromBool(k == .linear);
        return n;
    }

    pub fn linearLayers(c: *const Config) usize {
        var n: usize = 0;
        for (c.kinds[0..c.layers]) |k| n += @intFromBool(k == .linear);
        return n;
    }
};

fn int(o: std.json.ObjectMap, name: []const u8) !usize {
    const v = o.get(name) orelse return error.MissingField;
    if (v != .integer or v.integer < 0) return error.BadConfig;
    return @intCast(v.integer);
}

fn intOr(o: std.json.ObjectMap, name: []const u8, default: usize) !usize {
    return if (o.get(name) != null) int(o, name) else default;
}

fn float(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |x| @floatFromInt(x),
        .float => |x| x,
        else => error.BadConfig,
    };
}

pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, why: *Why) !Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    return parse(gpa, bytes, why);
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, why: *Why) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const top = parsed.value.object;
    const t = if (top.get("text_config")) |v| (if (v == .object) v.object else top) else top;
    const q = top.get("quantization_config") orelse t.get("quantization_config") orelse {
        why.set("config.json has no quantization_config; this family reads EXL3 packs", .{});
        return error.UnsupportedQuantization;
    };
    const method = if (q == .object) q.object.get("quant_method") else null;
    if (method == null or method.? != .string or !std.mem.eql(u8, method.?.string, "exl3")) {
        why.set("config.json's quant_method is not exl3; this family reads EXL3 packs", .{});
        return error.UnsupportedQuantization;
    }
    const rope = if (t.get("rope_parameters")) |v| (if (v == .object) v.object else std.json.ObjectMap.empty) else std.json.ObjectMap.empty;
    for ([_][]const u8{ "rope_type", "type" }) |key| if (rope.get(key)) |v| {
        if (v != .string or !std.mem.eql(u8, v.string, "default")) {
            why.set("rope {s} is not the default rope: the outputs would change and no bit-exact reference serves it", .{key});
            return error.UnsupportedFlashRope;
        }
    };
    if (t.get("rope_scaling")) |v| if (v != .null) {
        why.set("rope_scaling is set: the outputs would change and no bit-exact reference serves it", .{});
        return error.UnsupportedFlashRope;
    };
    const hidden = try int(t, "hidden_size");
    const heads = try int(t, "num_attention_heads");
    const head_dim = try intOr(t, "head_dim", hidden / @max(heads, 1));
    const partial = try float(rope.get("partial_rotary_factor") orelse t.get("partial_rotary_factor") orelse .{ .float = 0.25 });
    const ple_ids = t.get("ple_layer_ids");
    var c: Config = .{
        .hidden = hidden,
        .layers = try int(t, "num_hidden_layers"),
        .vocab = try int(t, "vocab_size"),
        .eps = @floatCast(try float(t.get("rms_norm_eps") orelse return error.MissingField)),
        .heads = heads,
        .kv_heads = try int(t, "num_key_value_heads"),
        .head_dim = head_dim,
        .rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * partial),
        .rope_theta = try float(rope.get("rope_theta") orelse .{ .float = 10_000_000 }),
        .nk = try int(t, "linear_num_key_heads"),
        .nv = try int(t, "linear_num_value_heads"),
        .dk = try int(t, "linear_key_head_dim"),
        .dv = try int(t, "linear_value_head_dim"),
        .conv_kernel = try int(t, "linear_conv_kernel_dim"),
        .experts = try int(t, "num_experts"),
        .topk = try int(t, "num_experts_per_tok"),
        .moe_width = try int(t, "moe_intermediate_size"),
        .shared_width = try int(t, "shared_expert_intermediate_size"),
        .streams = try intOr(t, "hc_count", 4),
        .low = try intOr(t, "hc_lowrank", 320),
        .index_heads = try intOr(t, "indexer_n_heads", 4),
        .index_dim = try intOr(t, "indexer_head_dim", 128),
        .index_budget = try intOr(t, "indexer_budget", 2048),
        .index_ratio = try intOr(t, "indexer_compress_ratio", 4),
        .ple_layer = null,
        .max_positions = try intOr(t, "max_position_embeddings", 262144),
        .bits_x100 = @intFromFloat(@round(try float(q.object.get("bits") orelse .{ .integer = 0 }) * 100)),
        .head_bits = try intOr(q.object, "head_bits", 0),
        .mtp_layers = try intOr(t, "mtp_num_hidden_layers", 0),
        .ngram = try intOr(t, "ngram_size", 3),
        .heads_per_ngram = try intOr(t, "heads_per_ngram", 8),
        .ngram_base = try intOr(t, "ngram_vocab_size_base", 20_000_000),
        .ngram_shards = try intOr(t, "split_ngram_parts", 128),
        .seed = try intOr(t, "seed", 1234),
        .ple_kernel = try intOr(t, "ple_conv_kernel_size", 4),
        .ple_eos = 0,
    };
    if (t.get("eos_token_id")) |v| switch (v) {
        .integer => |x| c.ple_eos = x,
        .array => |a| if (a.items.len > 0 and a.items[0] == .integer) {
            c.ple_eos = a.items[0].integer;
        },
        else => {},
    };
    if (ple_ids) |v| if (v == .array) {
        if (v.array.items.len > 1) {
            why.set("{d} n-gram embedding layers; the CUDA path serves one", .{v.array.items.len});
            return error.UnsupportedShape;
        }
        for (v.array.items) |item| {
            if (item != .integer or item.integer < 1) return error.BadConfig;
            c.ple_layer = @intCast(item.integer - 1); // ple_layer_ids count from 1
        }
    };
    if (c.layers > max_layers) return error.BadConfig;
    const kinds = t.get("layer_types") orelse return error.MissingField;
    if (kinds != .array or kinds.array.items.len != c.layers) return error.BadConfig;
    for (kinds.array.items, 0..) |v, i| {
        if (v != .string) return error.BadConfig;
        c.kinds[i] = if (std.mem.eql(u8, v.string, "linear_attention")) .linear else if (std.mem.eql(u8, v.string, "full_attention") or std.mem.eql(u8, v.string, "sparse_attention")) .attention else return error.BadConfig;
    }
    // gdn.cu: DK = DV = 128, 4 taps, 48 value heads over 16 key heads (or a rank's 24 over 8)
    if (c.dk != 128 or c.dv != 128 or c.conv_kernel != 4 or c.nv != 3 * c.nk or (c.nv != 48 and c.nv != 24)) {
        why.set("DeltaNet heads {d}/{d} of {d}x{d}, {d} taps: gdn.cu serves 16/48 of 128x128 and 4 taps", .{ c.nk, c.nv, c.dk, c.dv, c.conv_kernel });
        return error.UnsupportedShape;
    }
    // the hyper-connection glue: 4 streams, 256-wide blocks of the hidden size, a low rank under 512
    if (c.streams != 4 or c.hidden % 512 != 0 or c.low > 512 or c.low % 32 != 0 or c.topk + 1 > 16) {
        why.set("streams {d}, hidden {d}, low rank {d}, top-k {d}: the glue serves 4 streams of a multiple of 512, a low rank to 512, top-k to 15", .{ c.streams, c.hidden, c.low, c.topk });
        return error.UnsupportedShape;
    }
    return c;
}

test "parse the 3.05bpw pack's config" {
    var why: Why = .{};
    const c = try parse(std.testing.allocator, @embedFile("testdata/config_exl3_305.json"), &why);
    try std.testing.expectEqual(@as(usize, 48), c.layers);
    try std.testing.expectEqual(@as(usize, 36), c.linearLayers());
    try std.testing.expectEqual(Kind.linear, c.kinds[0]);
    try std.testing.expectEqual(Kind.attention, c.kinds[3]);
    try std.testing.expectEqual(@as(?usize, 0), c.linearIndex(0));
    try std.testing.expectEqual(@as(?usize, null), c.linearIndex(3));
    try std.testing.expectEqual(@as(?usize, 3), c.linearIndex(4));
    try std.testing.expectEqual(@as(usize, 10240), c.convDim());
    try std.testing.expectEqual(@as(usize, 16480), c.projWidth());
    try std.testing.expectEqual(@as(usize, 10240), c.wide());
    try std.testing.expectEqual(@as(usize, 11), c.slots());
    try std.testing.expectEqual(@as(?usize, 1), c.ple_layer);
    try std.testing.expectEqual(@as(usize, 305), c.bits_x100);
    try std.testing.expectEqual(@as(usize, 5), c.head_bits);
    try std.testing.expectEqual(@as(usize, 64), c.rotary_dim);
    try std.testing.expectEqual(@as(usize, 262144), c.max_positions);
    try std.testing.expectEqual(@as(i64, 248044), c.ple_eos);
    try std.testing.expectEqual(@as(usize, 128), c.ngram_shards);
}

test "refuse a non-EXL3 pack and a YaRN rope before any allocation" {
    var why: Why = .{};
    try std.testing.expectError(error.UnsupportedQuantization, parse(std.testing.allocator,
        \\{"text_config":{"hidden_size":2560},"quantization_config":{"quant_method":"mlx","bits":4}}
    , &why));
    try std.testing.expectError(error.UnsupportedFlashRope, parse(std.testing.allocator,
        \\{"quantization_config":{"quant_method":"exl3","bits":3.05},
        \\"text_config":{"hidden_size":2560,"rope_parameters":{"rope_type":"yarn","factor":4}}}
    , &why));
}
