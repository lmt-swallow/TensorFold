//! DFlash2's independent proposer contract. Target verification still chooses every emitted token.
const std = @import("std");

pub const Config = struct {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    sliding_window: usize,
    rms_norm_eps: f32,
    is_causal: bool,
    rope_parameters: struct { rope_theta: f32 },
    dflash_config: struct {
        block_size: usize,
        conv_group_size: usize,
        conv_kernel_size: usize,
        mask_token_id: u32,
        selector_rank: usize,
        selector_top_k: usize,
        target_layer_ids: []const usize,
    },

    pub fn parse(a: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Config) {
        var p = try std.json.parseFromSlice(Config, a, bytes, .{ .ignore_unknown_fields = true });
        errdefer p.deinit();
        try p.value.validate(2);
        return p;
    }

    pub fn validate(c: Config, world: usize) !void {
        const dc = c.dflash_config;
        if (world != 2 or c.hidden_size != 4096 or c.num_hidden_layers != 5 or c.num_attention_heads != 32 or c.num_key_value_heads != 8 or c.head_dim != 128 or c.intermediate_size != 12288 or c.vocab_size != 154880 or c.sliding_window != 2048 or c.is_causal or dc.block_size != 8 or dc.conv_group_size != 16 or dc.conv_kernel_size != 2 or dc.selector_rank != 256 or dc.selector_top_k != 16 or dc.mask_token_id >= c.vocab_size or dc.target_layer_ids.len != 5 or !(c.rms_norm_eps > 0) or !std.math.isFinite(c.rms_norm_eps) or !(c.rope_parameters.rope_theta > 0) or !std.math.isFinite(c.rope_parameters.rope_theta)) return error.UnsupportedDflashConfig;
        for (dc.target_layer_ids, 0..) |layer, i| {
            if (layer >= 45 or (i > 0 and layer <= dc.target_layer_ids[i - 1])) return error.UnsupportedDflashConfig;
        }
    }

    pub fn ringRows(c: Config) usize {
        // Python draft_ring_rows reserves a tile of slack for kept-prefix snapshots.
        return (c.sliding_window - 1 + c.dflash_config.block_size + 126) / 64 * 64;
    }
};

/// Logical committed context; speculative block writes never advance its end.
/// A ring slot may be reused only after its old position left the sliding window.
pub const Context = struct {
    end: usize = 0,
    capacity: usize,
    ring: usize,

    pub fn init(capacity: usize, ring: usize, window: usize, block: usize) !Context {
        if (capacity == 0 or ring < window + block or block == 0) return error.BadDflashContext;
        return .{ .capacity = capacity, .ring = ring };
    }

    pub fn append(c: *Context, rows: usize) !void {
        if (rows > c.capacity -| c.end) return error.DflashContextOverflow;
        c.end += rows;
    }

    pub fn slot(c: Context, position: usize) usize {
        return position % c.ring;
    }

    pub fn reset(c: *Context) void {
        c.end = 0;
    }
};

pub const Candidate = struct { token: u32, logit: f64 };

/// Rank-independent candidate order: descending score, then ascending global token ID.
pub fn candidateBefore(_: void, a: Candidate, b: Candidate) bool {
    return a.logit > b.logit or (a.logit == b.logit and a.token < b.token);
}

/// Greedy selector score. BF16 codebooks are expanded exactly to f32 before f64 arithmetic.
/// The projected row is reference F.linear's f32 output; no target sampling is performed here.
pub fn edgeScore(logit: f64, predecessor: []const f32, successor: []const f32, projected: []const f32) !f64 {
    if (predecessor.len != successor.len or predecessor.len != projected.len) return error.BadSelectorShape;
    var edge: f64 = 0;
    for (predecessor, successor, projected) |p, s, h| edge += @as(f64, s) * (@as(f64, p) * @as(f64, h));
    return logit + 0.6 * edge;
}

test "speculative ring writes do not commit context and rejected appends preserve state" {
    var c = try Context.init(10000, 2112, 2047, 8);
    for (0..9999) |p| {
        try std.testing.expectEqual(p % 2112, c.slot(p));
        const before = c.end;
        _ = c.slot(c.end + 7);
        try std.testing.expectEqual(before, c.end);
        try c.append(1);
    }
    try std.testing.expectError(error.DflashContextOverflow, c.append(2));
    try std.testing.expectEqual(@as(usize, 9999), c.end);
    c.reset();
    try std.testing.expectEqual(@as(usize, 0), c.end);
}

test "candidate merge ties use global token IDs regardless of rank input order" {
    var items = [_]Candidate{ .{ .token = 77442, .logit = 2 }, .{ .token = 7, .logit = 2 }, .{ .token = 1, .logit = 3 } };
    std.mem.sort(Candidate, &items, {}, candidateBefore);
    try std.testing.expectEqual(@as(u32, 1), items[0].token);
    try std.testing.expectEqual(@as(u32, 7), items[1].token);
    try std.testing.expectEqual(@as(u32, 77442), items[2].token);
    try std.testing.expectError(error.BadSelectorShape, edgeScore(0, &.{1}, &.{}, &.{1}));
    try std.testing.expectApproxEqAbs(@as(f64, 4.8), try edgeScore(0, &.{ 2, 1 }, &.{ 1, 2 }, &.{ 1, 3 }), 1e-12);
}

test "supported draft geometry rejects incompatible layers and vocabulary before allocation" {
    const raw =
        \\{"hidden_size":4096,"num_hidden_layers":5,"num_attention_heads":32,"num_key_value_heads":8,
        \\"head_dim":128,"intermediate_size":12288,"vocab_size":154880,"sliding_window":2048,
        \\"rms_norm_eps":0.00001,"is_causal":false,"rope_parameters":{"rope_theta":10000},
        \\"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":154856,
        \\"selector_rank":256,"selector_top_k":16,"target_layer_ids":[5,14,24,33,42]}}
    ;
    var parsed = try Config.parse(std.testing.allocator, raw);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 2176), parsed.value.ringRows());
    var wrong = parsed.value;
    wrong.vocab_size -= 1;
    try std.testing.expectError(error.UnsupportedDflashConfig, wrong.validate(2));
    wrong = parsed.value;
    wrong.dflash_config.target_layer_ids = &.{ 5, 14, 24, 24, 42 };
    try std.testing.expectError(error.UnsupportedDflashConfig, wrong.validate(2));
    wrong.dflash_config.target_layer_ids = &.{ 5, 14, 24, 33, 45 };
    try std.testing.expectError(error.UnsupportedDflashConfig, wrong.validate(2));
    try std.testing.expectError(error.UnsupportedDflashConfig, parsed.value.validate(4));
}

/// glm5_next/qmm.py's fixed shape rule. Changing K splits changes proposal bits.
pub fn splitK(n: usize, k: usize) usize {
    if (n == 12576 and k == 4096) return 4;
    if (n == 4096 and k == 4096) return 2;
    if (n == 2048 and k == 4096) return 4;
    if (n == 8192 and (k == 1536 or k == 512)) return 4;
    if (n == 4096 and k == 8192) return 8;
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

test "projection K slices cover real DFlash shapes without changing head arithmetic" {
    const cases = [_][3]usize{
        .{ 4096, 20480, 4 }, // tap fc
        .{ 1024, 4096, 8 }, // dynamic conv / committed KV
        .{ 3072, 4096, 4 }, // QKV
        .{ 4096, 2048, 4 }, // attention output
        .{ 12288, 4096, 1 }, // gate/up
        .{ 4096, 6144, 4 }, // MLP output
        .{ 77440, 4096, 1 }, // shared target draft head
    };
    for (cases) |c| {
        try std.testing.expectEqual(c[2], splitK(c[0], c[1]));
        try std.testing.expectEqual(@as(usize, 0), (c[1] / 64) % c[2]);
    }
}

pub const RingSpan = struct { source_row: usize, ring_row: usize, rows: usize };
pub const RingWrite = struct { spans: [2]RingSpan, count: usize };

/// Split one contiguous update at the physical ring boundary. Does not commit logical rows.
pub fn ringWrite(position: usize, rows: usize, capacity: usize) !RingWrite {
    if (capacity == 0 or rows == 0 or rows > capacity) return error.BadDflashRingWrite;
    const slot = position % capacity;
    const first = @min(rows, capacity - slot);
    return .{ .spans = .{
        .{ .source_row = 0, .ring_row = slot, .rows = first },
        .{ .source_row = first, .ring_row = 0, .rows = rows - first },
    }, .count = if (first == rows) 1 else 2 };
}

test "ring copies preserve absolute row mapping across wrap without aliasing live update rows" {
    for (0..3 * 2176) |position| {
        for (1..65) |rows| {
            const plan = try ringWrite(position, rows, 2176);
            var copied: usize = 0;
            for (plan.spans[0..plan.count]) |span| {
                try std.testing.expect(span.ring_row + span.rows <= 2176);
                try std.testing.expectEqual(copied, span.source_row);
                for (0..span.rows) |i| try std.testing.expectEqual((position + copied + i) % 2176, span.ring_row + i);
                copied += span.rows;
            }
            try std.testing.expectEqual(rows, copied);
        }
    }
    try std.testing.expectError(error.BadDflashRingWrite, ringWrite(0, 0, 2176));
    try std.testing.expectError(error.BadDflashRingWrite, ringWrite(0, 2177, 2176));
}
