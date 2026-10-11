//! GLM-5.3-Flash's text dimensions from an EXL3 pack's config.json (`text_config`), the fields the Mia fork's
//! `glm5_next/cuda/weights.py: Config.read` takes (TensorFold v0.6.0 with the recipe's patches, image
//! v0.6.0-9f73cca659a1), and the tensor-parallel layout of its `tp.py` at two ranks. Refused before any allocation:
//! a pack that is not the 4-bit "mcg" EXL3 of the routed experts with BF16 elsewhere, and shapes the kernels are not
//! built for. Layer kinds follow `layer_types`; the MTP layer (index `layers`) is a DSA layer with a MoE and no
//! hyper-connections.
const std = @import("std");

/// Backbone layers plus the MTP layer.
pub const max_layers = 48;

pub const Kind = enum { kda, dsa };
pub const MlpKind = enum { dense, moe };

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
    layers: usize, // backbone layers; the MTP layer is `layers`
    kinds: [max_layers]Kind = undefined,
    mlp_kinds: [max_layers]MlpKind = undefined,
    vocab: usize,
    eps: f32,
    heads: usize, // DSA (MLA) attention heads
    q_lora: usize,
    kv_lora: usize, // the latent width a DSA cache row holds
    qk_dim: usize, // nope only: qk_rope_head_dim is 0
    v_dim: usize,
    lin_heads: usize, // KDA heads
    lin_dim: usize,
    conv: usize, // KDA short-conv taps
    lower: f32, // KDA gate lower bound
    experts: usize,
    top_k: usize,
    moe_width: usize,
    shared_width: usize,
    dense_width: usize,
    routed_scale: f32,
    norm_topk: bool,
    streams: usize, // hyper-connection streams (hc_mult)
    hc_iters: usize,
    hc_eps: f32,
    index_heads: usize,
    index_dim: usize,
    index_topk: usize, // tokens a sparse row keeps: index_topk / kpool whole pools
    kpool: usize,
    limit: f32, // SwiGLU clamp
    mtp_layers: usize,
    eos: [4]u32 = @splat(0),
    eos_n: usize = 0,

    /// Width of the residual streams a row carries.
    pub fn wide(c: *const Config) usize {
        return c.streams * c.hidden;
    }

    /// Slots a row's MoE output holds: the routed experts, then the shared expert.
    pub fn slots(c: *const Config) usize {
        return c.top_k + 1;
    }

    /// The largest context (tokens) where DSA's top-k keeps every visible key, so attention stays dense
    /// (weights.py `Config.dense_limit`; sparse.py's SPARSE_FROM when index_topk is 2048 and kpool 4).
    pub fn denseLimit(c: *const Config) usize {
        return c.index_topk + c.kpool - 1;
    }

    /// Pools a sparse row selects (sparse.py TOPK_POOLS).
    pub fn topkPools(c: *const Config) usize {
        return c.index_topk / c.kpool;
    }

    /// A sparse row's attended tokens: its selected pools' tokens, then its incomplete pool's (sparse.py TOKENS).
    pub fn sparseTokens(c: *const Config) usize {
        return c.index_topk + c.kpool - 1;
    }

    pub fn kind(c: *const Config, i: usize) Kind {
        return if (i == c.layers) .dsa else c.kinds[i];
    }

    pub fn mlpKind(c: *const Config, i: usize) MlpKind {
        return if (i == c.layers) .moe else c.mlp_kinds[i];
    }

    pub fn count(c: *const Config, k: Kind) usize {
        var n: usize = 0;
        for (c.kinds[0..c.layers]) |x| n += @intFromBool(x == k);
        return n;
    }

    /// The index of layer `i` among the backbone layers of its kind (forward.py `State.kda_index` / `dsa_index`);
    /// the MTP layer's DSA index is `count(.dsa)` (its caches follow the backbone's).
    pub fn kindIndex(c: *const Config, i: usize) usize {
        const k = c.kind(i);
        var n: usize = 0;
        for (c.kinds[0..@min(i, c.layers)]) |x| n += @intFromBool(x == k);
        return n;
    }
};

/// Tensor-parallel families (tp.py UNITS): what a rank holds a share of.
pub const Family = enum { heads, lin, dense, moe, shared, vocab };

/// One rank of `world` (tp.py `Layout` at world 2: exact halves of every family, the remainder never arises).
pub const Layout = struct {
    rank: usize,
    world: usize,
    cfg: *const Config,

    pub fn total(l: Layout, f: Family) usize {
        const c = l.cfg;
        return switch (f) {
            .heads => c.heads,
            .lin => c.lin_heads,
            .dense => c.dense_width,
            .moe => c.moe_width,
            .shared => c.shared_width,
            .vocab => c.vocab,
        };
    }

    /// This rank's [start, stop) of family `f`.
    pub fn share(l: Layout, f: Family) [2]usize {
        const n = l.total(f) / l.world;
        return .{ l.rank * n, (l.rank + 1) * n };
    }

    pub fn size(l: Layout, f: Family) usize {
        const s = l.share(f);
        return s[1] - s[0];
    }

    /// This rank's [start, stop) of an axis of `length` elements holding family `f`'s total (tp.py `Layout.cut`).
    pub fn cut(l: Layout, f: Family, length: usize) [2]usize {
        const s = l.share(f);
        const t = l.total(f);
        return .{ s[0] * length / t, s[1] * length / t };
    }

    /// The KDA input projection's rows on this rank: [q | k | v | f_a | g_a | b] (weights.py `kda`), f_a and g_a whole.
    pub fn kdaProj(l: Layout) usize {
        return 3 * l.size(.lin) * l.cfg.lin_dim + 2 * l.cfg.lin_dim + l.size(.lin);
    }

    /// Offsets of f_a, g_a and b in a KDA projection row (weights.py `KDAW.fa_off`, `ga_off`, `b_off`).
    pub fn kdaOffsets(l: Layout) [3]usize {
        const fa = 3 * l.size(.lin) * l.cfg.lin_dim;
        return .{ fa, fa + l.cfg.lin_dim, fa + 2 * l.cfg.lin_dim };
    }
};

/// Every world the family serves: the two Sparks.
pub const world = 2;

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

fn floatOr(o: std.json.ObjectMap, name: []const u8, default: f64) !f64 {
    return if (o.get(name)) |v| float(v) else default;
}

fn str(o: std.json.ObjectMap, name: []const u8) ?[]const u8 {
    const v = o.get(name) orelse return null;
    return if (v == .string) v.string else null;
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
    // glm5_next/__init__.py EXL3_VARIANT: 4-bit trellis, the "mcg" codebook, routed experts only (BF16 elsewhere)
    const qv = top.get("quantization_config") orelse top.get("quantization") orelse {
        why.set("config.json has no quantization_config; this family reads the EXL3 pack", .{});
        return error.UnsupportedQuantization;
    };
    if (qv != .object) return error.BadConfig;
    const q = qv.object;
    const method = str(q, "quant_method") orelse "";
    const codebook = str(q, "codebook") orelse "";
    const scope = str(q, "scope") orelse "";
    const bits = try floatOr(q, "bits", 0);
    if (!std.mem.eql(u8, method, "exl3") or bits != 4 or !std.mem.eql(u8, codebook, "mcg") or
        !std.mem.eql(u8, scope, "glm53_routed_experts_only"))
    {
        why.set("quantization {s} {d} bits, codebook {s}, scope {s}: the CUDA kernels read 4-bit mcg EXL3 routed experts with BF16 elsewhere", .{ method, bits, codebook, scope });
        return error.UnsupportedQuantization;
    }
    const lin = if (t.get("linear_attn_config")) |v| (if (v == .object) v.object else std.json.ObjectMap.empty) else std.json.ObjectMap.empty;
    if ((try intOr(t, "n_group", 1)) != 1 or (try intOr(t, "topk_group", 1)) != 1) {
        why.set("grouped expert selection (n_group > 1) is not implemented", .{});
        return error.UnsupportedShape;
    }
    if ((try intOr(t, "qk_rope_head_dim", 0)) != 0) {
        why.set("only NoPE MLA (qk_rope_head_dim 0) is implemented", .{});
        return error.UnsupportedShape;
    }
    const moe_width = try int(t, "moe_intermediate_size");
    var c: Config = .{
        .hidden = try int(t, "hidden_size"),
        .layers = try int(t, "num_hidden_layers"),
        .vocab = try int(t, "vocab_size"),
        .eps = @floatCast(try float(t.get("rms_norm_eps") orelse return error.MissingField)),
        .heads = try int(t, "num_attention_heads"),
        .q_lora = try int(t, "q_lora_rank"),
        .kv_lora = try int(t, "kv_lora_rank"),
        .qk_dim = try int(t, "qk_nope_head_dim"),
        .v_dim = try int(t, "v_head_dim"),
        .lin_heads = try intOr(lin, "num_heads", 64),
        .lin_dim = try intOr(lin, "head_dim", 128),
        .conv = try intOr(lin, "short_conv_kernel_size", 4),
        .lower = @floatCast(try floatOr(lin, "gate_lower_bound", -5.0)),
        .experts = try int(t, "n_routed_experts"),
        .top_k = try int(t, "num_experts_per_tok"),
        .moe_width = moe_width,
        .shared_width = moe_width * try intOr(t, "n_shared_experts", 1),
        .dense_width = try int(t, "intermediate_size"),
        .routed_scale = @floatCast(try float(t.get("routed_scaling_factor") orelse return error.MissingField)),
        .norm_topk = if (t.get("norm_topk_prob")) |v| (v == .bool and v.bool) else true,
        .streams = try intOr(t, "hc_mult", 4),
        .hc_iters = try intOr(t, "hc_sinkhorn_iters", 20),
        .hc_eps = @floatCast(try floatOr(t, "hc_eps", 1e-6)),
        .index_heads = try intOr(t, "index_n_heads", 32),
        .index_dim = try intOr(t, "index_head_dim", 128),
        .index_topk = try intOr(t, "index_topk", 2048),
        .kpool = try intOr(t, "index_kpool", 4),
        .limit = @floatCast(try floatOr(t, "swiglu_limit", 10.0)),
        .mtp_layers = try intOr(t, "num_nextn_predict_layers", 0),
    };
    if (c.layers + 1 > max_layers) return error.BadConfig;
    const kinds = t.get("layer_types") orelse return error.MissingField;
    if (kinds != .array or kinds.array.items.len != c.layers) return error.BadConfig;
    for (kinds.array.items, 0..) |v, i| {
        if (v != .string) return error.BadConfig;
        c.kinds[i] = if (std.mem.eql(u8, v.string, "linear_attention")) .kda else .dsa;
    }
    const dense_first = try intOr(t, "first_k_dense_replace", 3);
    if (t.get("mlp_layer_types")) |v| {
        if (v != .array or v.array.items.len != c.layers) return error.BadConfig;
        for (v.array.items, 0..) |m, i| {
            if (m != .string) return error.BadConfig;
            c.mlp_kinds[i] = if (std.mem.eql(u8, m.string, "sparse")) .moe else .dense;
        }
    } else for (0..c.layers) |i| {
        c.mlp_kinds[i] = if (i < dense_first) .dense else .moe;
    }
    if (t.get("eos_token_id") orelse top.get("eos_token_id")) |v| switch (v) {
        .integer => |x| {
            c.eos[0] = @intCast(x);
            c.eos_n = 1;
        },
        .array => |a| for (a.items) |e| {
            if (e != .integer or c.eos_n == c.eos.len) return error.BadConfig;
            c.eos[c.eos_n] = @intCast(e.integer);
            c.eos_n += 1;
        },
        else => return error.BadConfig,
    };
    // kda.cu: DK = DV = 128, 4 taps; the glue: 4 streams of 4,096 (hc.cu's 16,384-wide mixing dots, 24 of them);
    // latent.py: a 512-wide latent; sparse.py: 128-wide index keys, pools of 4, 512 of them a sparse row;
    // exl3.cu's decode kernel: a rank's expert width in DEC_WIDTHS (1,024 at two ranks); glue._topk: 16 slots at most
    if (c.lin_dim != 128 or c.conv != 4 or c.streams != 4 or c.hidden != 4096 or c.kv_lora != 512 or
        c.index_dim != 128 or c.kpool != 4 or c.index_topk != 2048 or c.moe_width != 2048 or c.top_k + 1 > 16 or
        c.lin_heads % world != 0 or c.heads % world != 0 or c.vocab % (64 * world) != 0 or
        c.dense_width % (128 * world) != 0 or c.shared_width % (128 * world) != 0)
    {
        why.set("KDA {d}x{d} heads, {d} taps, {d} streams of {d}, latent {d}, index {d}/{d}/{d}, expert width {d}, top-{d}: the kernels serve GLM-5.3-Flash's shape", .{ c.lin_heads, c.lin_dim, c.conv, c.streams, c.hidden, c.kv_lora, c.index_dim, c.kpool, c.index_topk, c.moe_width, c.top_k });
        return error.UnsupportedShape;
    }
    return c;
}

test "parse the 4bpw pack's config" {
    var why: Why = .{};
    const c = try parse(std.testing.allocator, @embedFile("testdata/config_exl3_4bpw.json"), &why);
    try std.testing.expectEqual(@as(usize, 45), c.layers);
    try std.testing.expectEqual(@as(usize, 34), c.count(.kda));
    try std.testing.expectEqual(@as(usize, 11), c.count(.dsa));
    try std.testing.expectEqual(Kind.kda, c.kind(0));
    try std.testing.expectEqual(Kind.dsa, c.kind(3));
    try std.testing.expectEqual(Kind.kda, c.kind(44));
    try std.testing.expectEqual(Kind.dsa, c.kind(45)); // the MTP layer
    try std.testing.expectEqual(MlpKind.dense, c.mlpKind(2));
    try std.testing.expectEqual(MlpKind.moe, c.mlpKind(3));
    try std.testing.expectEqual(MlpKind.moe, c.mlpKind(45));
    try std.testing.expectEqual(@as(usize, 10), c.kindIndex(43));
    try std.testing.expectEqual(@as(usize, 11), c.kindIndex(45));
    try std.testing.expectEqual(@as(usize, 33), c.kindIndex(44));
    try std.testing.expectEqual(@as(usize, 2051), c.denseLimit());
    try std.testing.expectEqual(@as(usize, 512), c.topkPools());
    try std.testing.expectEqual(@as(usize, 2051), c.sparseTokens());
    try std.testing.expectEqual(@as(usize, 16384), c.wide());
    try std.testing.expectEqual(@as(usize, 9), c.slots());
    try std.testing.expectEqual(@as(usize, 256), c.qk_dim);
    try std.testing.expectEqual(@as(usize, 2048), c.shared_width);
    try std.testing.expectEqual(@as(f32, 2.5), c.routed_scale);
    try std.testing.expectEqual(@as(f32, 1e-5), c.eps);
    try std.testing.expectEqual(@as(usize, 3), c.eos_n);
    try std.testing.expectEqual(@as(u32, 154827), c.eos[1]);
}

test "two ranks: tp.py's halves and the stacked KDA projection" {
    var why: Why = .{};
    const c = try parse(std.testing.allocator, @embedFile("testdata/config_exl3_4bpw.json"), &why);
    const r1: Layout = .{ .rank = 1, .world = 2, .cfg = &c };
    try std.testing.expectEqual([2]usize{ 32, 64 }, r1.share(.heads));
    try std.testing.expectEqual([2]usize{ 32, 64 }, r1.share(.lin));
    try std.testing.expectEqual([2]usize{ 6144, 12288 }, r1.share(.dense));
    try std.testing.expectEqual([2]usize{ 1024, 2048 }, r1.share(.moe));
    try std.testing.expectEqual([2]usize{ 77440, 154880 }, r1.share(.vocab));
    // q_b's 16,384 rows hold the 64 heads' 256-wide queries
    try std.testing.expectEqual([2]usize{ 8192, 16384 }, r1.cut(.heads, 16384));
    // an expert's gate trellis holds 2,048 / 16 column tiles
    try std.testing.expectEqual([2]usize{ 64, 128 }, r1.cut(.moe, 128));
    // forward.py's split-K table entry "12576x4096" is this projection
    try std.testing.expectEqual(@as(usize, 12576), r1.kdaProj());
    try std.testing.expectEqual([3]usize{ 12288, 12416, 12544 }, r1.kdaOffsets());
}

test "refuse another quantization and another shape before any allocation" {
    var why: Why = .{};
    try std.testing.expectError(error.UnsupportedQuantization, parse(std.testing.allocator,
        \\{"text_config":{"hidden_size":4096},"quantization_config":{"quant_method":"mlx","bits":4}}
    , &why));
    try std.testing.expectError(error.UnsupportedQuantization, parse(std.testing.allocator,
        \\{"text_config":{"hidden_size":4096},"quantization_config":{"quant_method":"exl3","bits":3,"codebook":"mcg","scope":"glm53_routed_experts_only"}}
    , &why));
    try std.testing.expectError(error.UnsupportedShape, parse(std.testing.allocator,
        \\{"text_config":{"qk_rope_head_dim":64},"quantization_config":{"quant_method":"exl3","bits":4,"codebook":"mcg","scope":"glm53_routed_experts_only"}}
    , &why));
}
