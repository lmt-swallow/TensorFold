//! Flash Next's n-gram embedding rows on the host (TensorFold 0.6.6 `qwen4_exp/cuda/ngram.py`, `exl3_pack.py:
//! NgramTable`): each token's row ids in the n-gram table, one per hash head, from the token and the n - 1 before it
//! (an EOS starts a new segment: earlier tokens read as EOS), and the rows themselves read from the table's shards.
//!
//! Row id of head h for a k-gram (k = 2..n, heads k-2 .. a block of `per_ngram`): ((t0 * m0) ^ (t1 * m1) ^ ...
//! ^ (t(k-1) * m(k-1))) mod size_h + offset_h, the products in wrapping int64 and the remainder non-negative (numpy's
//! `%` with a positive divisor), t_j the token j places back.
const std = @import("std");

pub const max_n = 4;
pub const max_heads = 32;

pub const NGram = struct {
    n: usize, // ngram_size
    per_ngram: usize, // heads_per_ngram
    eos: i64,
    multipliers: [max_n]i64 = @splat(0),
    sizes: [max_heads]i64 = @splat(0),
    offsets: [max_heads]i64 = @splat(0),

    pub fn heads(self: *const NGram) usize {
        return (self.n - 1) * self.per_ngram;
    }

    /// The `n - 1` tokens before a sequence's first: all EOS.
    pub fn initialHistory(self: *const NGram, out: []i64) void {
        @memset(out[0 .. self.n - 1], self.eos);
    }

    /// Row ids [tokens.len, heads] (row-major into `out`) for `tokens` after `history` (n - 1 tokens).
    pub fn ids(self: *const NGram, history: []const i64, tokens: []const i64, out: []i64) !void {
        const ctx = self.n - 1;
        if (history.len != ctx or out.len != tokens.len * self.heads()) return error.BadShape;
        var buf: [256]i64 = undefined;
        if (ctx + tokens.len > buf.len) return error.WindowTooLarge;
        const seq = buf[0 .. ctx + tokens.len];
        @memcpy(seq[0..ctx], history);
        @memcpy(seq[ctx..], tokens);
        self.idsOf(seq, ctx, out);
    }

    /// `ids` for any number of tokens (a prompt chunk), through a scratch allocation.
    pub fn idsAny(self: *const NGram, gpa: std.mem.Allocator, history: []const i64, tokens: []const i64, out: []i64) !void {
        const ctx = self.n - 1;
        if (history.len != ctx or out.len != tokens.len * self.heads()) return error.BadShape;
        const seq = try gpa.alloc(i64, ctx + tokens.len);
        defer gpa.free(seq);
        @memcpy(seq[0..ctx], history);
        @memcpy(seq[ctx..], tokens);
        self.idsOf(seq, ctx, out);
    }

    fn idsOf(self: *const NGram, seq: []const i64, ctx: usize, out: []i64) void {
        var last_eos: i64 = -1;
        for (0..seq.len) |p| {
            const in_segment: i64 = @as(i64, @intCast(p)) - (last_eos + 1);
            if (p >= ctx) {
                var shifted: [max_n]i64 = undefined;
                for (0..self.n) |s| {
                    const ok = in_segment >= @as(i64, @intCast(s)) and p >= s;
                    shifted[s] = if (ok) seq[p - s] else self.eos;
                }
                const row = out[(p - ctx) * self.heads() ..][0..self.heads()];
                for (2..self.n + 1) |k| {
                    var mixed: i64 = shifted[0] *% self.multipliers[0];
                    for (1..k) |j| mixed ^= shifted[j] *% self.multipliers[j];
                    const first = (k - 2) * self.per_ngram;
                    for (0..self.per_ngram) |hh| {
                        const head = first + hh;
                        row[head] = @mod(mixed, self.sizes[head]) + self.offsets[head];
                    }
                }
            }
            if (seq[p] == self.eos) last_eos = @intCast(p);
        }
    }

    /// The history after `tokens` were kept: the last n - 1 of [history; tokens].
    pub fn advance(self: *const NGram, history: []i64, tokens: []const i64) void {
        const ctx = self.n - 1;
        for (tokens) |t| {
            std.mem.copyForwards(i64, history[0 .. ctx - 1], history[1..ctx]);
            history[ctx - 1] = t;
        }
    }
};

fn isPrime(v: u64) bool {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    var d: u64 = 3;
    while (d * d <= v) : (d += 2) if (v % d == 0) return false;
    return true;
}

fn nthPrimeAfter(start: u64, count: u64) u64 {
    var p = start;
    var left = count;
    while (left > 0) : (left -= 1) {
        p += 1;
        while (!isPrime(p)) p += 1;
    }
    return p;
}

fn splitmix64(value: u64) u64 {
    var v = value +% 0x9E3779B97F4A7C15;
    v = (v ^ (v >> 30)) *% 0xBF58476D1CE4E5B9;
    v = (v ^ (v >> 27)) *% 0x94D049BB133111EB;
    return v ^ (v >> 31);
}

/// ngram.py `NGram(...)`: the table geometry and hash multipliers derived from the config (the loader checks them
/// against the pack's head_offsets, head_vocab_sizes and layer_multipliers).
pub fn derive(vocab: usize, n: usize, per_ngram: usize, vocab_base: u64, seed: u64, eos: i64, ple_index: usize) !NGram {
    if (n < 2 or n > max_n or (n - 1) * per_ngram > max_heads) return error.UnsupportedNgram;
    var g: NGram = .{ .n = n, .per_ngram = per_ngram, .eos = eos };
    var total: i64 = 0;
    for (0..g.heads()) |h| {
        const size = nthPrimeAfter(vocab_base - 1, ple_index * g.heads() + h + 1);
        g.sizes[h] = @intCast(size);
        g.offsets[h] = total;
        total += @intCast(size);
    }
    const half: u64 = @max(1, ((@as(u64, 1) << 63) - 1) / @max(vocab, 1) / 2);
    const base = seed +% 10007 *% ple_index;
    for (0..n) |i| g.multipliers[i] = @intCast(2 * (splitmix64(base +% 0x9E3779B97F4A7C15 *% (i + 1)) % half) + 1);
    return g;
}

test "row ids are 0.6.6's (EOS resets the n-grams)" {
    const g = try derive(248320, 3, 8, 20_000_000, 1234, 248044, 0);
    try std.testing.expectEqual([3]i64{ 23703573157769, 20109073645365, 8052911324071 }, g.multipliers[0..3].*);
    try std.testing.expectEqual(@as(i64, 20000003), g.sizes[0]);
    try std.testing.expectEqual(@as(i64, 20000171), g.sizes[15]);
    try std.testing.expectEqual(@as(i64, 300001275), g.offsets[15]);
    var hist: [2]i64 = undefined;
    g.initialHistory(&hist);
    const toks = [_]i64{ 9707, 11, 1879, 248044, 77, 248319, 5 };
    var out: [7 * 16]i64 = undefined;
    try g.ids(&hist, &toks, &out);
    // ngram.py NGram.ids on the same tokens (rows 0, 3 and 6, every head)
    try std.testing.expectEqualSlices(i64, &.{ 16410909, 39682429, 55103279, 60931720, 87006904, 116506179, 131512017, 152932897, 169641436, 182022480, 209277433, 237891023, 256841529, 277007269, 290665954, 300984276 }, out[0..16]);
    try std.testing.expectEqualSlices(i64, &.{ 18827606, 27700912, 56521518, 65780303, 81132822, 100518937, 120474864, 142052741, 163793416, 184135950, 219854666, 227756662, 248975183, 271245822, 285569892, 318943450 }, out[48..64]);
    try std.testing.expectEqualSlices(i64, &.{ 16757868, 22422241, 58999532, 78402164, 81784946, 103711543, 127350423, 153600400, 177205656, 184373949, 207281446, 231126602, 254149269, 278070838, 292911170, 308610507 }, out[96..112]);
    // a history ending in EOS
    var out2: [2 * 16]i64 = undefined;
    try g.ids(&.{ 1879, 248044 }, &.{ 42, 43 }, &out2);
    try std.testing.expectEqual(@as(i64, 4064167), out2[0]);
    try std.testing.expectEqual(@as(i64, 314347972), out2[15]);
    try std.testing.expectEqual(@as(i64, 18529343), out2[16]);
    try std.testing.expectEqual(@as(i64, 317050322), out2[31]);
    g.advance(&hist, &toks);
    try std.testing.expectEqual([2]i64{ 248319, 5 }, hist);
}
