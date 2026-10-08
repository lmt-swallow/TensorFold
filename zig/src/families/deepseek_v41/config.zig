//! DeepSeek-V4.1-Flash dimensions from config.json (`text_config`, else the top level), the fields the Python family's
//! `config.py` reads, checked against what the CUDA kernels serve.
const std = @import("std");

pub const max_layers = 64;
pub const max_list = 16;

pub const Why = struct {
    buf: [320]u8 = undefined,
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

/// A short list from config.json (layer ids, ratios): `items[0..len]`.
pub const List = struct {
    items: [max_layers]i64 = undefined,
    len: usize = 0,

    pub fn slice(self: *const List) []const i64 {
        return self.items[0..self.len];
    }

    pub fn has(self: *const List, v: i64) bool {
        for (self.slice()) |x| if (x == v) return true;
        return false;
    }

    pub fn indexOf(self: *const List, v: i64) ?usize {
        for (self.slice(), 0..) |x, i| if (x == v) return i;
        return null;
    }
};

pub const Config = struct {
    vocab: usize,
    dim: usize,
    inter: usize,
    n_layers: usize, // backbone layers (40); the checkpoint also carries `nextn` DSpark blocks (mtp.0..)
    nextn: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    q_rank: usize,
    o_rank: usize,
    o_groups: usize,
    window: usize,
    eps: f32,
    n_routed: usize,
    n_shared: usize,
    topk: usize,
    route_scale: f32,
    swiglu_limit: f32,
    norm_topk: bool,
    compress_ratios: List, // one per layer (backbone + nextn); 0 = window attention only
    kv_sources: List,
    index_sources: List,
    idx_heads: usize,
    idx_dim: usize,
    idx_topk: usize,
    cand_source: i64,
    cand_blocks: usize,
    cand_block: usize,
    hc: usize,
    hc_iters: usize,
    hc_eps: f32,
    rope_theta: f64,
    compress_theta: f64,
    rope_factor: f64,
    orig_len: usize,
    beta_fast: f64,
    beta_slow: f64,
    engram_layers: List,
    engram_rows: List,
    engram_ngram: usize,
    engram_vocab: usize,
    engram_heads: usize,
    engram_dim: usize,
    engram_pad: u32,
    engram_cvocab: usize,
    dspark_block: usize,
    dspark_noise: u32,
    dspark_taps: List,
    dspark_rank: usize,
    dspark_routed: usize,
    dspark_topk: usize,
    max_positions: usize,
    /// quantization_config: exl3's average bits x 10 (29 for the 2.9 bpw pack) and the head's bits.
    avg_bits_x10: usize = 0,
    head_bits: usize = 0,

    /// Compression ratio of layer `i` (0 for window-only layers and the DSpark blocks).
    pub fn ratio(self: *const Config, i: usize) usize {
        if (i >= self.compress_ratios.len) return 0;
        return @intCast(self.compress_ratios.items[i]);
    }

    pub fn isKvSource(self: *const Config, i: usize) bool {
        return self.kv_sources.has(@intCast(i));
    }

    pub fn isIndexSource(self: *const Config, i: usize) bool {
        return self.index_sources.has(@intCast(i));
    }

    pub fn isEngram(self: *const Config, i: usize) bool {
        return self.engram_layers.has(@intCast(i));
    }

    /// This rank's attention heads under TP: heads split evenly (64 -> 32 a rank).
    pub fn localHeads(self: *const Config, world: usize) usize {
        return self.n_heads / world;
    }

    /// The compressed KV entries a layer at `r` holds for `positions` (ratio 0: none).
    pub fn compressedEntries(positions: usize, r: usize) usize {
        return if (r == 0) 0 else (positions + r - 1) / r;
    }

    /// `dir`/config.json.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        var why: Why = .{};
        return parse(gpa, text, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
    }
};

fn intOr(obj: std.json.ObjectMap, key: []const u8, default: i64) i64 {
    const v = obj.get(key) orelse return default;
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => default,
    };
}

fn int(obj: std.json.ObjectMap, key: []const u8) !usize {
    const v = obj.get(key) orelse {
        std.log.err("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    return switch (v) {
        .integer => |i| @intCast(i),
        else => error.BadConfig,
    };
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn floatOr(obj: std.json.ObjectMap, key: []const u8, default: f64) f64 {
    const v = obj.get(key) orelse return default;
    return number(v) catch default;
}

fn float(obj: std.json.ObjectMap, key: []const u8) !f32 {
    return @floatCast(try number(obj.get(key) orelse return error.BadConfig));
}

fn list(obj: std.json.ObjectMap, key: []const u8, required: bool) !List {
    var out: List = .{};
    const v = obj.get(key) orelse {
        if (required) {
            std.log.err("config.json has no {s}", .{key});
            return error.BadConfig;
        }
        return out;
    };
    if (v != .array) return error.BadConfig;
    if (v.array.items.len > max_layers) return error.BadConfig;
    for (v.array.items) |item| {
        out.items[out.len] = switch (item) {
            .integer => |i| i,
            else => return error.BadConfig,
        };
        out.len += 1;
    }
    return out;
}

/// config.json's text; a refusal's reason goes to `why`.
pub fn parse(allocator: std.mem.Allocator, text: []const u8, why: *Why) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    const top = parsed.value.object;
    const mt = (top.get("model_type") orelse return error.BadConfig);
    if (mt != .string or !std.mem.eql(u8, mt.string, "deepseek_v41")) {
        why.set("config.json's model_type is {f}, not deepseek_v41", .{std.json.fmt(mt, .{})});
        return error.NotDeepseekV41;
    }
    const t = if (top.get("text_config")) |tc| (if (tc == .object) tc.object else top) else top;
    const rs: ?std.json.ObjectMap = if (t.get("rope_scaling")) |r| (if (r == .object) r.object else null) else null;
    var c: Config = .{
        .vocab = try int(t, "vocab_size"),
        .dim = try int(t, "hidden_size"),
        .inter = try int(t, "moe_intermediate_size"),
        .n_layers = try int(t, "num_hidden_layers"),
        .nextn = @intCast(intOr(t, "num_nextn_predict_layers", 0)),
        .n_heads = try int(t, "num_attention_heads"),
        .head_dim = try int(t, "head_dim"),
        .rope_dim = try int(t, "qk_rope_head_dim"),
        .q_rank = try int(t, "q_lora_rank"),
        .o_rank = try int(t, "o_lora_rank"),
        .o_groups = try int(t, "o_groups"),
        .window = try int(t, "sliding_window"),
        .eps = try float(t, "rms_norm_eps"),
        .n_routed = try int(t, "n_routed_experts"),
        .n_shared = @intCast(intOr(t, "n_shared_experts", 1)),
        .topk = try int(t, "num_experts_per_tok"),
        .route_scale = try float(t, "routed_scaling_factor"),
        .swiglu_limit = try float(t, "swiglu_limit"),
        .norm_topk = if (t.get("norm_topk_prob")) |v| (v == .bool and v.bool) else true,
        .compress_ratios = try list(t, "compress_ratios", true),
        .kv_sources = try list(t, "kv_source_layer_ids", true),
        .index_sources = try list(t, "index_source_layer_ids", true),
        .idx_heads = try int(t, "index_n_heads"),
        .idx_dim = try int(t, "index_head_dim"),
        .idx_topk = try int(t, "index_topk"),
        .cand_source = intOr(t, "candidate_source_layer_id", -1),
        .cand_blocks = @intCast(intOr(t, "candidate_topk_blocks", 0)),
        .cand_block = @intCast(intOr(t, "candidate_block_size", 0)),
        .hc = try int(t, "hc_mult"),
        .hc_iters = try int(t, "hc_sinkhorn_iters"),
        .hc_eps = try float(t, "hc_eps"),
        .rope_theta = floatOr(t, "rope_theta", 10000),
        .compress_theta = floatOr(t, "compress_rope_theta", 10000),
        .rope_factor = if (rs) |r| floatOr(r, "factor", 1.0) else 1.0,
        .orig_len = if (rs) |r| @intCast(intOr(r, "original_max_position_embeddings", 0)) else 0,
        .beta_fast = if (rs) |r| floatOr(r, "beta_fast", 32) else 32,
        .beta_slow = if (rs) |r| floatOr(r, "beta_slow", 1) else 1,
        .engram_layers = try list(t, "engram_layer_ids", false),
        .engram_rows = try list(t, "engram_num_embeddings", false),
        .engram_ngram = @intCast(intOr(t, "engram_max_ngram_size", 1)),
        .engram_vocab = @intCast(intOr(t, "engram_vocab_size", 0)),
        .engram_heads = @intCast(intOr(t, "engram_n_heads", 0)),
        .engram_dim = @intCast(intOr(t, "engram_head_dim", 0)),
        .engram_pad = @intCast(intOr(t, "engram_pad_token_id", 2)),
        .engram_cvocab = @intCast(intOr(t, "engram_compressed_vocab_size", 0)),
        .dspark_block = @intCast(intOr(t, "dspark_block_size", 0)),
        .dspark_noise = @intCast(intOr(t, "dspark_noise_token_id", 0)),
        .dspark_taps = try list(t, "dspark_target_layer_ids", false),
        .dspark_rank = @intCast(intOr(t, "dspark_markov_rank", 0)),
        .dspark_routed = @intCast(intOr(t, "dspark_n_routed_experts", 0)),
        .dspark_topk = @intCast(intOr(t, "dspark_num_experts_per_tok", 0)),
        .max_positions = @intCast(intOr(t, "max_position_embeddings", 65536)),
    };
    if (top.get("quantization_config")) |q| if (q == .object) {
        const qm = q.object.get("quant_method") orelse std.json.Value{ .null = {} };
        if (!(qm == .string and std.mem.eql(u8, qm.string, "exl3"))) {
            why.set("config.json's quant_method is {f}; this family reads EXL3 packs", .{std.json.fmt(qm, .{})});
            return error.UnsupportedQuantization;
        }
        c.avg_bits_x10 = @intFromFloat(@round(floatOr(q.object, "avg_bits", 0) * 10));
        c.head_bits = @intCast(intOr(q.object, "head_bits", 0));
    };
    if (c.compress_ratios.len != c.n_layers + c.nextn) {
        why.set("compress_ratios has {d} entries for {d} backbone + {d} DSpark layers", .{ c.compress_ratios.len, c.n_layers, c.nextn });
        return error.BadConfig;
    }
    if (c.head_dim != 512 or c.rope_dim != 64 or c.hc != 4) {
        why.set("head_dim {d}, rope {d}, hc {d}: the kernels serve 512/64/4", .{ c.head_dim, c.rope_dim, c.hc });
        return error.UnsupportedShape;
    }
    return c;
}

test "parse the Mia pack's config" {
    const text =
        \\{"model_type":"deepseek_v41","quantization_config":{"quant_method":"exl3","avg_bits":2.9,"head_bits":6},
        \\"text_config":{"vocab_size":129280,"hidden_size":5120,"moe_intermediate_size":2304,"num_hidden_layers":40,
        \\"num_nextn_predict_layers":3,"num_attention_heads":64,"head_dim":512,"qk_rope_head_dim":64,"q_lora_rank":1280,
        \\"o_lora_rank":1024,"o_groups":8,"sliding_window":128,"rms_norm_eps":1e-20,"n_routed_experts":384,
        \\"num_experts_per_tok":6,"routed_scaling_factor":1.5,"swiglu_limit":10.0,
        \\"compress_ratios":[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        \\"kv_source_layer_ids":[2,8,14,20],"index_source_layer_ids":[2,8,14,20,24,28,32,36],"index_n_heads":32,
        \\"index_head_dim":128,"index_topk":512,"candidate_source_layer_id":20,"candidate_topk_blocks":2048,
        \\"candidate_block_size":8,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,"rope_theta":10000,
        \\"compress_rope_theta":160000,"rope_scaling":{"rope_type":"yarn","factor":16,"beta_fast":32,"beta_slow":1,
        \\"original_max_position_embeddings":65536},"engram_layer_ids":[1,14],"engram_num_embeddings":[384006168,384016682],
        \\"engram_max_ngram_size":4,"engram_vocab_size":16000000,"engram_n_heads":8,"engram_head_dim":256,
        \\"engram_pad_token_id":2,"engram_compressed_vocab_size":99092,"dspark_block_size":5,"dspark_noise_token_id":128799,
        \\"dspark_target_layer_ids":[37,38,39],"dspark_markov_rank":256,"dspark_n_routed_experts":128,
        \\"dspark_num_experts_per_tok":3,"max_position_embeddings":1048576}}
    ;
    var why: Why = .{};
    const c = try parse(std.testing.allocator, text, &why);
    try std.testing.expectEqual(@as(usize, 40), c.n_layers);
    try std.testing.expectEqual(@as(usize, 2), c.ratio(2));
    try std.testing.expectEqual(@as(usize, 1), c.ratio(20));
    try std.testing.expectEqual(@as(usize, 0), c.ratio(40));
    try std.testing.expect(c.isEngram(14) and !c.isEngram(13));
    try std.testing.expectEqual(@as(usize, 29), c.avg_bits_x10);
    try std.testing.expectEqual(@as(usize, 32), c.localHeads(2));
}
