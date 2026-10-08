//! Engram's n-gram row ids from a token history, DeepSeek's rule as the Python family hashes it on the host
//! (`ops.EngramHasher`, `cuda/model.py Engram.hashes`): bit for bit, numpy int64 arithmetic.
//!
//! Per position p and Engram layer l, column c = (n - 2) * heads + head for n-gram sizes n = 2 .. max_ngram holds
//!   row = (xor_{i < n} class(tok[p - i]) *% mult[l][i]) mod prime[l][c] + offset[l][c]
//! where class() is the compressed token map, a token before position 0 or at/behind an image span (negative id)
//! is the pad class (and so is every longer n-gram from there on), primes are the next unused primes above
//! engram_vocab - 1 drawn in (layer, n, head) order, and offset is the sum of the primes before column c (every
//! layer's primes sum to its table's rows). `hashes` writes `out[(p * layers + l) * cols + c]`.
//!
//! The token map itself comes from tokenizer.json through the `tokenizers` normalizers (NFKC, NFD, strip accents,
//! lowercase, whitespace); the Python engine caches it as a JSON int array at ~/.cache/dsv41_token_map.json
//! (TF_DS_TOKEN_MAP), and `loadTokenMap` reads that file: the normalization is not reproduced here.

const std = @import("std");

pub const max_layers = 4;
pub const max_ngram = 8;
pub const max_heads = 16;
pub const max_cols = (max_ngram - 1) * max_heads;

pub const Params = struct {
    layers: []const i64, // engram_layer_ids, e.g. {1, 14}
    rows: []const i64, // engram_num_embeddings per layer (the primes' sums), checked
    ngram: usize, // engram_max_ngram_size (4)
    heads: usize, // engram_n_heads (8)
    vocab: usize, // engram_vocab_size (16,000,000): primes start above vocab - 1
    cvocab: usize, // engram_compressed_vocab_size (99,092): the token map's class count
    pad_token: u32, // engram_pad_token_id (2)
};

/// numpy default_rng(10007 * layer).integers(0, (2^63 - 1) // cvocab // 2, size=ngram) * 2 + 1 for the
/// checkpoint's (cvocab, ngram); numpy's generator stream is not reproduced, the values are kept.
const KnownMult = struct { cvocab: usize, ngram: usize, layer: i64, mult: [4]i64 };
const known_mults = [_]KnownMult{
    .{ .cvocab = 99092, .ngram = 4, .layer = 1, .mult = .{ 76632096046245, 4839876093313, 35959672319349, 73987337458391 } },
    .{ .cvocab = 99092, .ngram = 4, .layer = 14, .mult = .{ 67716810739261, 51510806800915, 30921347202721, 82619226485591 } },
};

pub const Error = error{ TooMany, BadParams, UnknownMultipliers, RowsMismatch, TokenOutOfMap };

pub const Hasher = struct {
    n_layers: usize,
    ngram: usize,
    heads: usize,
    cols: usize,
    primes: [max_layers][max_cols]i64,
    offsets: [max_layers][max_cols]i64,
    mult: [max_layers][max_ngram]i64,
    map: []const i32,
    pad: i64,

    /// `token_map[token] = class`; `mult` overrides the known multipliers (one row per layer) when given.
    pub fn init(p: Params, token_map: []const i32, mult: ?[]const [max_ngram]i64) Error!Hasher {
        if (p.layers.len == 0 or p.layers.len > max_layers or p.ngram < 2 or p.ngram > max_ngram or p.heads == 0 or p.heads > max_heads) return error.BadParams;
        if (p.rows.len != p.layers.len) return error.BadParams;
        if (p.pad_token >= token_map.len) return error.TokenOutOfMap;
        var self: Hasher = .{
            .n_layers = p.layers.len,
            .ngram = p.ngram,
            .heads = p.heads,
            .cols = (p.ngram - 1) * p.heads,
            .primes = undefined,
            .offsets = undefined,
            .mult = undefined,
            .map = token_map,
            .pad = token_map[p.pad_token],
        };
        try drawPrimes(p, &self.primes);
        for (0..self.n_layers) |l| {
            var sum: i64 = 0;
            for (0..self.cols) |c| {
                self.offsets[l][c] = sum;
                sum += self.primes[l][c];
            }
            if (sum != p.rows[l]) return error.RowsMismatch;
            if (mult) |m| {
                self.mult[l] = m[l];
            } else {
                self.mult[l] = knownMult(p, p.layers[l]) orelse return error.UnknownMultipliers;
            }
        }
        return self;
    }

    /// Row ids of positions start .. start + n - 1 given `ids`, the whole sequence so far (an image span's
    /// positions negative); `out` holds n * layers * cols values.
    pub fn hashes(self: *const Hasher, ids: []const i32, start: usize, n: usize, out: []i64) Error!void {
        std.debug.assert(out.len >= n * self.n_layers * self.cols);
        std.debug.assert(start + n <= ids.len);
        var toks: [max_ngram]i64 = undefined;
        for (0..n) |i| {
            const pos = start + i;
            var blocked = false;
            for (0..self.ngram) |shift| {
                var cls: i64 = self.pad;
                if (pos < shift) {
                    blocked = true;
                } else {
                    const raw = ids[pos - shift];
                    if (raw < 0) {
                        blocked = true;
                    } else {
                        if (@as(usize, @intCast(raw)) >= self.map.len) return error.TokenOutOfMap;
                        cls = self.map[@intCast(raw)];
                    }
                }
                toks[shift] = if (blocked) self.pad else cls;
            }
            for (0..self.n_layers) |l| {
                var rolling: i64 = toks[0] *% self.mult[l][0];
                const base = (i * self.n_layers + l) * self.cols;
                for (1..self.ngram) |k| {
                    rolling ^= toks[k] *% self.mult[l][k];
                    for (0..self.heads) |h| {
                        const c = (k - 1) * self.heads + h;
                        out[base + c] = @mod(rolling, self.primes[l][c]) + self.offsets[l][c];
                    }
                }
            }
        }
    }

    /// The columns rank r of `world` reads: [cols * r / world, cols * (r + 1) / world).
    pub fn rankColumns(self: *const Hasher, rank: usize, world: usize) [2]usize {
        return .{ self.cols * rank / world, self.cols * (rank + 1) / world };
    }
};

fn knownMult(p: Params, layer: i64) ?[max_ngram]i64 {
    for (known_mults) |k| if (k.cvocab == p.cvocab and k.ngram == p.ngram and k.layer == layer) {
        var m: [max_ngram]i64 = @splat(0);
        for (k.mult, 0..) |v, i| m[i] = v;
        return m;
    };
    return null;
}

fn isPrime(n: i64) bool {
    if (n < 2) return false;
    if (@rem(n, 2) == 0) return n == 2;
    var d: i64 = 3;
    while (d * d <= n) : (d += 2) if (@rem(n, d) == 0) return false;
    return true;
}

/// ops.engram_primes: for every (layer, n-gram size, head) in order, the next prime above vocab - 1 not drawn before.
fn drawPrimes(p: Params, primes: *[max_layers][max_cols]i64) Error!void {
    var seen: [max_layers * max_cols]i64 = undefined;
    var n_seen: usize = 0;
    for (0..p.layers.len) |l| {
        for (0..p.ngram - 1) |k| {
            var cur: i64 = @intCast(p.vocab - 1);
            for (0..p.heads) |h| {
                cur += 1;
                while (!isPrime(cur) or contains(seen[0..n_seen], cur)) cur += 1;
                seen[n_seen] = cur;
                n_seen += 1;
                primes[l][k * p.heads + h] = cur;
            }
        }
    }
}

fn contains(xs: []const i64, v: i64) bool {
    for (xs) |x| if (x == v) return true;
    return false;
}

/// The Python engine's cached token map (a JSON array of ints, token id -> class) as i32.
pub fn loadTokenMap(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]i32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const items = parsed.value.array.items;
    const out = try gpa.alloc(i32, items.len);
    errdefer gpa.free(out);
    for (items, 0..) |v, i| out[i] = @intCast(v.integer);
    return out;
}

// -- test against the Python fork's fixture (tools/zig/dsv41/engram_fixture.py) ---------------------------------------

fn ints(gpa: std.mem.Allocator, v: std.json.Value) ![]i64 {
    const out = try gpa.alloc(i64, v.array.items.len);
    for (v.array.items, 0..) |x, i| out[i] = x.integer;
    return out;
}

test "row ids equal the Python fork's" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const text = std.Io.Dir.cwd().readFileAlloc(io, "zig/tests/cuda/dsv41_fixtures/engram_hash.json", gpa, .limited(1 << 24)) catch return error.SkipZigTest;
    defer gpa.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    const layers = try ints(gpa, o.get("engram_layers").?);
    defer gpa.free(layers);
    const rows = try ints(gpa, o.get("engram_rows").?);
    defer gpa.free(rows);
    const p: Params = .{
        .layers = layers,
        .rows = rows,
        .ngram = @intCast(o.get("ngram").?.integer),
        .heads = @intCast(o.get("heads").?.integer),
        .vocab = @intCast(o.get("vocab").?.integer),
        .cvocab = @intCast(o.get("cvocab").?.integer),
        .pad_token = @intCast(o.get("pad_token").?.integer),
    };
    const tokens = try ints(gpa, o.get("tokens").?);
    defer gpa.free(tokens);
    var map_len: usize = p.pad_token + 1;
    for (tokens) |t| if (t >= 0) {
        map_len = @max(map_len, @as(usize, @intCast(t)) + 1);
    };
    const map = try gpa.alloc(i32, map_len);
    defer gpa.free(map);
    @memset(map, -1);
    for (o.get("token_classes").?.array.items) |pair| map[@intCast(pair.array.items[0].integer)] = @intCast(pair.array.items[1].integer);
    const h = try Hasher.init(p, map, null);
    try std.testing.expectEqual(o.get("pad_class").?.integer, h.pad);
    for (o.get("primes").?.array.items, 0..) |row, l| for (row.array.items, 0..) |v, c| try std.testing.expectEqual(v.integer, h.primes[l][c]);
    for (o.get("offsets").?.array.items, 0..) |row, l| for (row.array.items, 0..) |v, c| try std.testing.expectEqual(v.integer, h.offsets[l][c]);
    for (o.get("mult").?.array.items, 0..) |row, l| for (row.array.items, 0..) |v, k| try std.testing.expectEqual(v.integer, h.mult[l][k]);
    const ids = try gpa.alloc(i32, tokens.len);
    defer gpa.free(ids);
    for (tokens, 0..) |t, i| ids[i] = @intCast(t);
    var checked: usize = 0;
    for (o.get("cases").?.array.items) |case| {
        const start: usize = @intCast(case.object.get("start").?.integer);
        const n: usize = @intCast(case.object.get("n").?.integer);
        const out = try gpa.alloc(i64, n * h.n_layers * h.cols);
        defer gpa.free(out);
        try h.hashes(ids, start, n, out);
        for (case.object.get("rows").?.array.items, 0..) |per_pos, i| for (per_pos.array.items, 0..) |per_layer, l| for (per_layer.array.items, 0..) |v, c| {
            try std.testing.expectEqual(v.integer, out[(i * h.n_layers + l) * h.cols + c]);
            checked += 1;
        };
    }
    try std.testing.expect(checked > 0);
}
