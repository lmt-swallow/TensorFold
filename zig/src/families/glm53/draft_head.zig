//! The MTP drafts' 4-bit copy of a rank's head rows (the Mia fork's `qmm.draft_quantize4` with TF_GLM_DRAFT_QUANT
//! unset: `quantize4`, each group's full min / max range), packed as the shared lane matmul reads it
//! (tensorfold/cuda/kernels/qmm.py `pack`). Drafts only propose, so this copy never changes a reply, but the drafts
//! (and so the tokens a round lands) follow its bits, which these functions reproduce on the host:
//!
//!   per row and group of 64: lo, hi = min, max of the bf16 values (fp32); scale = bf16(max((hi - lo) / 15, 1e-8));
//!   bias = bf16(lo); q = clamp(round_half_even((x - bias) / scale), 0, 15) in fp32 arithmetic.
//!
//! The packed layout: words int32 [npad / 64][k / 64][8][32][2] with npad the rows rounded up to 128; word
//! [t][g][j][4 r + c][v] holds nibble p (bits 4p ..) of row 64 t + 8 j + r, input 64 g + 32 v + 2 c + OFFSETS[p];
//! scales and biases bf16 [k / 64][npad], zero past the rows.
const std = @import("std");

pub const gs = 64;
/// qmm.py OFFSETS: nibble slot p of a lane's word holds input 32 v + 2 (lane % 4) + OFFSETS[p] of the group.
const offsets = [8]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };

fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// fp32 -> bf16, round to nearest even (torch's `.to(torch.bfloat16)`; the values here are finite).
fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    const lsb = (u >> 16) & 1;
    return @intCast((u + 0x7FFF + lsb) >> 16);
}

/// torch.round: to nearest, ties to even.
fn roundEven(x: f32) f32 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2.0) != 0) return r - std.math.sign(x);
    return r;
}

pub const Packed = struct {
    words: []u32, // [npad / 64][k / 64][8][32][2]
    scales: []u16, // [k / 64][npad] bf16
    biases: []u16,
    n: usize,
    k: usize,
    npad: usize,

    pub fn deinit(p: *Packed, gpa: std.mem.Allocator) void {
        gpa.free(p.words);
        gpa.free(p.scales);
        gpa.free(p.biases);
    }
};

/// `quantize4` then `pack` of bf16 rows `w` [n, k] (row-major).
pub fn quantizePack(gpa: std.mem.Allocator, w: []const u16, n: usize, k: usize) !Packed {
    if (k % gs != 0 or w.len != n * k) return error.BadShape;
    const kg = k / gs;
    const npad = (n + 127) / 128 * 128;
    var p: Packed = .{
        .words = try gpa.alloc(u32, npad / 64 * kg * 8 * 32 * 2),
        .scales = try gpa.alloc(u16, kg * npad),
        .biases = try gpa.alloc(u16, kg * npad),
        .n = n,
        .k = k,
        .npad = npad,
    };
    errdefer p.deinit(gpa);
    @memset(p.words, 0);
    @memset(p.scales, 0);
    @memset(p.biases, 0);
    var q: [gs]u8 = undefined;
    for (0..n) |row| {
        const t = row / 64;
        const j = (row % 64) / 8;
        const r = row % 8;
        for (0..kg) |g| {
            const vals = w[row * k + g * gs ..][0..gs];
            var lo: f32 = std.math.inf(f32);
            var hi: f32 = -std.math.inf(f32);
            for (vals) |v| {
                const f = bf16ToF32(v);
                lo = @min(lo, f);
                hi = @max(hi, f);
            }
            const scale_b = f32ToBf16(@max((hi - lo) / 15.0, 1e-8));
            const bias_b = f32ToBf16(lo);
            const scale = bf16ToF32(scale_b);
            const bias = bf16ToF32(bias_b);
            for (vals, &q) |v, *o| {
                const x = roundEven((bf16ToF32(v) - bias) / scale);
                o.* = @intFromFloat(std.math.clamp(x, 0.0, 15.0));
            }
            p.scales[g * npad + row] = scale_b;
            p.biases[g * npad + row] = bias_b;
            // [t][g][j][4 r + c][v]
            for (0..2) |v| for (0..4) |c| {
                var word: u32 = 0;
                for (offsets, 0..) |off, slot| word |= @as(u32, q[32 * v + 2 * c + off]) << @intCast(4 * slot);
                p.words[((((t * kg + g) * 8 + j) * 32 + 4 * r + c) * 2) + v] = word;
            };
        }
    }
    return p;
}

test "quantize4's rounding: ties to even, the range's ends exact" {
    try std.testing.expectEqual(@as(f32, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f32, 4), roundEven(3.5));
    try std.testing.expectEqual(@as(f32, -2), roundEven(-2.5));
    try std.testing.expectEqual(@as(f32, 3), roundEven(2.6));
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(@bitCast(@as(u32, 0x3F808000)))); // a tie to even
    try std.testing.expectEqual(@as(u16, 0x3F82), f32ToBf16(@bitCast(@as(u32, 0x3F818000))));
    // one group spanning 0 .. 15: q equals the value, scale 1, bias 0
    var w: [64]u16 = undefined;
    for (&w, 0..) |*x, i| x.* = f32ToBf16(@floatFromInt(i % 16));
    var p = try quantizePack(std.testing.allocator, &w, 1, 64);
    defer p.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u16, 0x3F80), p.scales[0]);
    try std.testing.expectEqual(@as(u16, 0), p.biases[0]);
    // word (r 0, c 0, v 0) holds inputs 0, 8, 16, 24, 1, 9, 17, 25 -> values 0, 8, 0, 8, 1, 9, 1, 9
    try std.testing.expectEqual(@as(u32, 0x91918080), p.words[0]);
}
