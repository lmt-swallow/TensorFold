//! A Qwen3.8-Flash-Next EXL3 pack's tensors for one GPU, prepared as TensorFold 0.6.6 prepares them
//! (`qwen4_exp/cuda/exl3.py: load`), byte for byte: the fp16 matrices concatenated in its row order, the centred norms
//! as fp32 `stored + offset`, A_log and dt_bias widened to fp32, the router's fp16 rows rounded to bf16 (nearest even).
//! EXL3 groups stay host views (`Group`): the EXL3 linear that consumes them owns their device layout.
const std = @import("std");
const upload = @import("cuda_upload.zig");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const names = @import("weights_map.zig");

const Checkpoint = core.Checkpoint;
const Tensor = core.checkpoint.Tensor;

/// The pack's language-model tensors (and its head), every shard mapped read-only.
pub const Pack = struct {
    lm: Checkpoint,
    head: Checkpoint,
    mtp: Checkpoint, // the MTP head's tensors, and its mixer from the patch file beside the shards

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Pack {
        var lm = try Checkpoint.openModelPrefix(gpa, io, dir, names.lm);
        errdefer lm.close();
        var head = try Checkpoint.openModelPrefix(gpa, io, dir, names.head ++ ".");
        errdefer head.close();
        var mtp = try Checkpoint.openModelPrefix(gpa, io, dir, names.mtp);
        errdefer mtp.close();
        const patch = try std.fs.path.join(gpa, &.{ dir, mtp_patch });
        defer gpa.free(patch);
        if (std.Io.Dir.cwd().access(io, patch, .{})) |_| {
            try mtp.add(dir, mtp_patch);
        } else |_| {}
        return .{ .lm = lm, .head = head, .mtp = mtp };
    }

    pub fn close(self: *Pack) void {
        self.lm.close();
        self.head.close();
        self.mtp.close();
    }

    fn of(self: *Pack, name: []const u8) *Checkpoint {
        if (std.mem.startsWith(u8, name, names.lm)) return &self.lm;
        if (std.mem.startsWith(u8, name, names.mtp)) return &self.mtp;
        return &self.head;
    }

    pub fn has(self: *Pack, name: []const u8) bool {
        for (self.of(name).files.items) |*f| if (f.get(name) != null) return true;
        return false;
    }

    pub fn get(self: *Pack, name: []const u8) !Tensor {
        return self.of(name).get(name);
    }

    pub fn expect(self: *Pack, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !Tensor {
        return self.of(name).expect(name, dtype, shape);
    }

    /// EXL3 group `prefix`: its trellis, scales and codebook (exl3_pack.py `Pack.codebook`, `scales`).
    pub fn group(self: *Pack, prefix: []const u8) !Group {
        var n: names.Name = .{};
        const tr = try self.get(names.part(&n, prefix, "trellis"));
        if (tr.dtype != .i16 or tr.rank != 3) return error.UnexpectedTensor;
        const kt = tr.shape[0];
        const nt = tr.shape[1];
        const suh = try self.expect(names.part(&n, prefix, "suh"), .f16, &.{16 * kt});
        const svh = try self.expect(names.part(&n, prefix, "svh"), .f16, &.{16 * nt});
        const cb: Codebook = if (self.has(names.part(&n, prefix, "mul1"))) .mul1 else if (self.has(names.part(&n, prefix, "mcg"))) .mcg else .inst3;
        return .{ .trellis = tr.bytes, .kt = kt, .nt = nt, .w16 = tr.shape[2], .suh = suh.bytes, .svh = svh.bytes, .codebook = cb };
    }
};

/// exl3_pack.py EXTRA_FILES: the MTP head's hyper-connection mixer, outside the index.
pub const mtp_patch = "mtp_hyper_connection_mixer_patch.safetensors";

/// exl3/linear.py CODEBOOK_IDS.
pub const Codebook = enum(c_int) { inst3 = 0, mcg = 1, mul1 = 2 };

/// One EXL3 group as stored: trellis int16 [kt, nt, w16], suh f16 [16 kt], svh f16 [16 nt].
pub const Group = struct {
    trellis: []const u8,
    kt: usize,
    nt: usize,
    w16: usize,
    suh: []const u8,
    svh: []const u8,
    codebook: Codebook,

    pub fn k(self: Group) usize {
        return 16 * self.kt;
    }
    pub fn n(self: Group) usize {
        return 16 * self.nt;
    }
};

/// exl3_mm.py `f16_split`: K slices of an (n, k) fp16 matrix, a function of the shape only (so a row's sum order is
/// the same at every row count).
pub fn f16Split(n: usize, k: usize) usize {
    const target: usize = 96;
    const tiles = (n + 63) / 64;
    var sk: usize = 1;
    while (sk < 32 and tiles * sk < target and k % (sk * 2 * 64) == 0 and k / (sk * 2) >= 256) sk *= 2;
    return sk;
}

/// exl3.py `centred_offset`: 1.0 when the pack stores the centred norms as gamma - 1, 0.0 when as gamma; decided from
/// the means of every layer's attention hyper-connection norm.
pub fn centredOffset(gpa: std.mem.Allocator, pack: *Pack, cfg: *const Config) !f32 {
    var means: std.ArrayList(f64) = .empty;
    defer means.deinit(gpa);
    var n: names.Name = .{};
    for (0..cfg.layers) |i| {
        const name = names.hc(&n, i, .attn, "hc_norm");
        if (!pack.has(name)) continue;
        const t = try pack.get(name);
        var sum: f64 = 0;
        for (0..t.numel()) |j| sum += try element(t, j);
        try means.append(gpa, sum / @as(f64, @floatFromInt(t.numel())));
    }
    if (means.items.len == 0) return 1.0;
    var above: usize = 0;
    for (means.items) |m| above += @intFromBool(m > 0.5);
    std.mem.sort(f64, means.items, {}, std.sort.asc(f64));
    const len = means.items.len;
    const median = if (len % 2 == 1) means.items[len / 2] else (means.items[len / 2 - 1] + means.items[len / 2]) / 2;
    const frac = @as(f64, @floatFromInt(above)) / @as(f64, @floatFromInt(len));
    const around_zero = frac <= 0.1 and median >= -0.5 and median <= 0.25;
    const around_one = frac >= 0.9 and median >= 0.75 and median <= 1.5;
    if (around_zero == around_one) return error.AmbiguousNormStorage;
    return if (around_zero) 1.0 else 0.0;
}

pub fn element(t: Tensor, j: usize) !f64 {
    return switch (t.dtype) {
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * j ..][0..2], .little)))),
        .bf16 => @floatCast(bf16ToF32(std.mem.readInt(u16, t.bytes[2 * j ..][0..2], .little))),
        .f32 => @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, t.bytes[4 * j ..][0..4], .little)))),
        else => error.UnexpectedTensor,
    };
}

pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// torch's float -> bfloat16 (c10 round_to_nearest_even): nearest, ties to even, NaN to the quiet 0x7FC0.
pub fn f32ToBf16(x: f32) u16 {
    if (std.math.isNan(x)) return 0x7FC0;
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// A plain tensor as fp32 values (torch's `.float()`: exact from f16, bf16 and f32).
fn widen(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    const out = try gpa.alloc(f32, t.numel());
    errdefer gpa.free(out);
    for (out, 0..) |*v, j| v.* = @floatCast(try element(t, j));
    return out;
}

/// Rows of fp16 matrices stacked (exl3_mm.py `f16`: torch.cat of `.to(float16)` rows); every part must be fp16 with
/// `k` columns (the pack stores them so; a conversion here would be a rounding the reference may not make).
fn stackF16(gpa: std.mem.Allocator, parts: []const Tensor, k: usize) ![]u8 {
    var len: usize = 0;
    for (parts) |p| {
        if (p.dtype != .f16 or p.rank != 2 or p.shape[1] != k) return error.UnexpectedTensor;
        len += p.bytes.len;
    }
    const out = try gpa.alloc(u8, len);
    var at: usize = 0;
    for (parts) |p| {
        @memcpy(out[at..][0..p.bytes.len], p.bytes);
        at += p.bytes.len;
    }
    return out;
}

/// An fp16 matrix [n, k] on the device and its fixed K slices (exl3_mm.py `F16`).
pub const F16 = struct {
    w: cuda.DeviceBuffer,
    n: usize,
    k: usize,
    sk: usize,

    fn init(d: *const cuda.Driver, bytes: []const u8, n: usize, k: usize) !F16 {
        if (bytes.len != n * k * 2) return error.BadShape;
        return .{ .w = try cuda.DeviceBuffer.fromHost(d, bytes), .n = n, .k = k, .sk = f16Split(n, k) };
    }

    pub fn deinit(self: *F16) void {
        self.w.free();
    }
};

/// One hyper-connection (exl3.py `hc`): down [low (+ streams inject rows)] x wide, up wide x low, scale fp32 [wide].
pub const Hc = struct {
    down: F16,
    up: F16,
    scale: cuda.DeviceBuffer,
    inject: bool,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, pack: *Pack, cfg: *const Config, prefix: []const u8, inject: bool, offset: f32) !Hc {
        var n: names.Name = .{};
        const wide = cfg.wide();
        var parts: [2]Tensor = undefined;
        parts[0] = try pack.expect(n.of("{s}.input_mix_weight_down.weight", .{prefix}), .f16, &.{ cfg.low, wide });
        if (inject) parts[1] = try pack.expect(n.of("{s}.block_inject_weight.weight", .{prefix}), .f16, &.{ cfg.streams, wide });
        const down_bytes = try stackF16(gpa, parts[0 .. @as(usize, 1) + @intFromBool(inject)], wide);
        defer gpa.free(down_bytes);
        var down = try F16.init(d, down_bytes, cfg.low + if (inject) cfg.streams else 0, wide);
        errdefer down.deinit();
        const up_t = try pack.expect(n.of("{s}.input_mix_weight_up.weight", .{prefix}), .f16, &.{ wide, cfg.low });
        var up = try F16.init(d, up_t.bytes, wide, cfg.low);
        errdefer up.deinit();
        const norm = try pack.get(n.of("{s}.hc_norm.weight", .{prefix}));
        if (norm.numel() != wide) return error.UnexpectedTensor;
        const scale = try widen(gpa, norm);
        defer gpa.free(scale);
        for (scale) |*v| v.* += offset;
        return .{ .down = down, .up = up, .scale = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(scale)), .inject = inject };
    }

    pub fn deinit(self: *Hc) void {
        self.down.deinit();
        self.up.deinit();
        self.scale.free();
    }
};

/// A DeltaNet layer's own tensors (exl3.py `gdn`): the projection's fp16 part (b ; a), the conv taps, gates, norm,
/// and the three EXL3 groups (qkv, z, out) as host views.
pub const Gdn = struct {
    ba: F16,
    conv: cuda.DeviceBuffer, // bf16 [conv_dim, taps]
    a_log: cuda.DeviceBuffer, // fp32 [nv]
    dt_bias: cuda.DeviceBuffer, // fp32 [nv]
    norm: cuda.DeviceBuffer, // bf16 [dv]
    qkv: Group,
    z: Group,
    out: Group,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, pack: *Pack, cfg: *const Config, i: usize) !Gdn {
        const w_at: names.Where = .{ .lm = i };
        var n: names.Name = .{};
        var n2: names.Name = .{};
        const qkv = try pack.group(names.layer(&n, w_at, "linear_attn.in_proj_qkv"));
        const z = try pack.group(names.layer(&n, w_at, "linear_attn.in_proj_z"));
        const out = try pack.group(names.layer(&n, w_at, "linear_attn.out_proj"));
        if (qkv.n() != cfg.convDim() or z.n() != cfg.nv * cfg.dv or out.k() != cfg.nv * cfg.dv or out.n() != cfg.hidden) return error.UnexpectedTensor;
        const parts = [_]Tensor{
            try pack.expect(names.layer(&n, w_at, "linear_attn.in_proj_b.weight"), .f16, &.{ cfg.nv, cfg.hidden }),
            try pack.expect(names.layer(&n2, w_at, "linear_attn.in_proj_a.weight"), .f16, &.{ cfg.nv, cfg.hidden }),
        };
        const ba_bytes = try stackF16(gpa, &parts, cfg.hidden);
        defer gpa.free(ba_bytes);
        var ba = try F16.init(d, ba_bytes, 2 * cfg.nv, cfg.hidden);
        errdefer ba.deinit();
        const conv_t = try pack.expect(names.layer(&n, w_at, "linear_attn.conv1d.weight"), .bf16, &.{ cfg.convDim(), 1, cfg.conv_kernel });
        var conv = try cuda.DeviceBuffer.fromHost(d, conv_t.bytes);
        errdefer conv.free();
        var gates: [2]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (gates[0..made]) |*g| g.free();
        for ([_][]const u8{ names.layer(&n, w_at, "linear_attn.A_log"), names.layer(&n2, w_at, "linear_attn.dt_bias") }, &gates) |name, *g| {
            const t = try pack.get(name);
            if (t.numel() != cfg.nv) return error.UnexpectedTensor;
            const v = try widen(gpa, t);
            defer gpa.free(v);
            g.* = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(v));
            made += 1;
        }
        const norm_t = try pack.expect(names.layer(&n, w_at, "linear_attn.norm.weight"), .bf16, &.{cfg.dv});
        return .{ .ba = ba, .conv = conv, .a_log = gates[0], .dt_bias = gates[1], .norm = try cuda.DeviceBuffer.fromHost(d, norm_t.bytes), .qkv = qkv, .z = z, .out = out };
    }

    pub fn deinit(self: *Gdn) void {
        self.ba.deinit();
        self.conv.free();
        self.a_log.free();
        self.dt_bias.free();
        self.norm.free();
    }
};

/// The router rows (exl3.py `moe`): [mlp.gate ; mlp.shared_expert_gate] as bf16 [experts + 1, hidden].
pub fn router(gpa: std.mem.Allocator, d: *const cuda.Driver, pack: *Pack, cfg: *const Config, w_at: names.Where) !cuda.DeviceBuffer {
    var n: names.Name = .{};
    var n2: names.Name = .{};
    const parts = [_]Tensor{
        try pack.get(names.layer(&n, w_at, "mlp.gate.weight")),
        try pack.get(names.layer(&n2, w_at, "mlp.shared_expert_gate.weight")),
    };
    const rows = cfg.experts + 1;
    const out = try gpa.alloc(u16, rows * cfg.hidden);
    defer gpa.free(out);
    var at: usize = 0;
    for (parts) |t| {
        if (t.rank != 2 or t.shape[1] != cfg.hidden) return error.UnexpectedTensor;
        for (0..t.numel()) |j| {
            out[at] = switch (t.dtype) {
                .bf16 => std.mem.readInt(u16, t.bytes[2 * j ..][0..2], .little),
                else => f32ToBf16(@floatCast(try element(t, j))),
            };
            at += 1;
        }
    }
    if (at != out.len) return error.UnexpectedTensor;
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(out));
}

/// The bf16 embedding table [vocab, hidden] (an EXL3 pack leaves it unquantized).
pub fn embedding(d: *const cuda.Driver, pack: *Pack, cfg: *const Config) !cuda.DeviceBuffer {
    const t = try pack.expect(names.embed, .bf16, &.{ cfg.vocab, cfg.hidden });
    return cuda.DeviceBuffer.fromHost(d, t.bytes);
}

test "f16 split follows the shape alone" {
    try std.testing.expectEqual(@as(usize, 16), f16Split(324, 10240)); // a hyper-connection's down projection
    try std.testing.expectEqual(@as(usize, 1), f16Split(10240, 320)); // its up projection
    try std.testing.expectEqual(@as(usize, 8), f16Split(96, 2560)); // a DeltaNet layer's (b ; a) rows
    try std.testing.expectEqual(@as(usize, 32), f16Split(320, 10240)); // the mixer's down (no inject rows)
}

test "bf16 rounding is torch's" {
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(@bitCast(@as(u32, 0x3F808000)))); // tie to even (down)
    try std.testing.expectEqual(@as(u16, 0x3F82), f32ToBf16(@bitCast(@as(u32, 0x3F818000)))); // tie to even (up)
    try std.testing.expectEqual(@as(u16, 0x7FC0), f32ToBf16(std.math.nan(f32)));
    try std.testing.expectEqual(@as(u16, 0x7F80), f32ToBf16(std.math.inf(f32)));
}

/// The n-gram table of an EXL3 pack (exl3_pack.py `NgramTable`): int16 rows of one fp16 scale word and `dh` values
/// of `bits` bits, split over shards, mapped read-only from ngram_embedding.safetensors.
pub const NgramTable = struct {
    file: core.safetensors.File,
    shards: [][]const u8, // each shard's rows, row-major
    starts: []usize, // first global row of each shard, then the total
    words: usize, // int16 words a row
    dh: usize,
    bits: usize,
    head_bias: []const u8, // fp16 [heads, dh]
    head_offsets: []const u8, // int64 [heads]
    head_sizes: []const u8,
    multipliers: []const u8, // int64 [n]
    gpa: std.mem.Allocator,

    /// Layer `layer`'s table from `dir`/ngram_embedding.safetensors.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, layer: usize, shards: usize) !NgramTable {
        var n: names.Name = .{};
        const path = try std.fs.path.join(gpa, &.{ dir, "ngram_embedding.safetensors" });
        defer gpa.free(path);
        const base = n.of(names.lm ++ "layers.{d}.ple.ple_embedding.ngram_embedding.", .{layer});
        var file = try core.safetensors.File.openPrefix(gpa, io, path, base);
        errdefer file.close(io);
        var t: NgramTable = undefined;
        t.gpa = gpa;
        t.shards = try gpa.alloc([]const u8, shards);
        errdefer gpa.free(t.shards);
        t.starts = try gpa.alloc(usize, shards + 1);
        errdefer gpa.free(t.starts);
        t.starts[0] = 0;
        var nn: names.Name = .{};
        var words: ?usize = null;
        for (0..shards) |i| {
            const s = file.get(nn.of("{s}shard_{d}.trellis", .{ base, i })) orelse return error.MissingTensor;
            if (s.dtype != .i16 or s.rank != 2) return error.UnexpectedTensor;
            if (words != null and words.? != s.shape[1]) return error.UnexpectedTensor;
            words = s.shape[1];
            t.shards[i] = s.bytes;
            t.starts[i + 1] = t.starts[i] + s.shape[0];
        }
        t.words = words orelse return error.MissingTensor;
        t.dh = 160;
        t.bits = (t.words - 1) * 16 / t.dh;
        if (t.bits < 2 or t.bits > 8 or 1 + t.dh * t.bits / 16 != t.words) return error.UnexpectedTensor;
        const get = struct {
            fn f(fl: *const core.safetensors.File, nm: *names.Name, b: []const u8, what: []const u8) ![]const u8 {
                return (fl.get(nm.of("{s}{s}", .{ b, what })) orelse return error.MissingTensor).bytes;
            }
        }.f;
        t.head_bias = try get(&file, &nn, base, "head_bias");
        t.head_offsets = try get(&file, &nn, base, "head_offsets");
        t.head_sizes = try get(&file, &nn, base, "head_vocab_sizes");
        t.multipliers = try get(&file, &nn, base, "layer_multipliers");
        t.file = file;
        return t;
    }

    pub fn close(self: *NgramTable, io: std.Io) void {
        self.gpa.free(self.shards);
        self.gpa.free(self.starts);
        self.file.close(io);
    }

    pub fn rows(self: *const NgramTable) usize {
        return self.starts[self.starts.len - 1];
    }

    /// Rows `ids` (global) into `out` (ids.len * words int16, as bytes).
    pub fn gather(self: *const NgramTable, ids: []const i64, out: []u8) !void {
        const rb = 2 * self.words;
        if (out.len != ids.len * rb) return error.BadShape;
        for (ids, 0..) |id, i| {
            if (id < 0 or @as(usize, @intCast(id)) >= self.rows()) return error.RowOutOfRange;
            const r: usize = @intCast(id);
            const shard = std.sort.upperBound(usize, self.starts, r, struct {
                fn order(a: usize, b: usize) std.math.Order {
                    return std.math.order(a, b);
                }
            }.order) - 1;
            const at = (r - self.starts[shard]) * rb;
            @memcpy(out[i * rb ..][0..rb], self.shards[shard][at..][0..rb]);
        }
    }

    /// Ask the kernel to read rows `ids` of the mapped table ahead of a `gather` (asynchronous readahead; a hint, so
    /// ids out of range and failures are ignored).
    pub fn prefetch(self: *const NgramTable, ids: []const i64) void {
        const rb = 2 * self.words;
        for (ids) |id| {
            if (id < 0 or @as(usize, @intCast(id)) >= self.rows()) continue;
            const r: usize = @intCast(id);
            const shard = std.sort.upperBound(usize, self.starts, r, struct {
                fn order(a: usize, b: usize) std.math.Order {
                    return std.math.order(a, b);
                }
            }.order) - 1;
            upload.prefetch(self.shards[shard][(r - self.starts[shard]) * rb ..][0..rb]);
        }
    }

    /// The pack's head offsets, sizes and multipliers equal the ones derived from the config (ngram.py `check`).
    pub fn check(self: *const NgramTable, g: *const @import("ngram.zig").NGram) !void {
        const heads = g.heads();
        if (self.head_offsets.len != heads * 8 or self.head_sizes.len != heads * 8 or self.multipliers.len != g.n * 8) return error.NgramMismatch;
        for (0..heads) |h| {
            if (std.mem.readInt(i64, self.head_offsets[8 * h ..][0..8], .little) != g.offsets[h]) return error.NgramMismatch;
            if (std.mem.readInt(i64, self.head_sizes[8 * h ..][0..8], .little) != g.sizes[h]) return error.NgramMismatch;
        }
        for (0..g.n) |i| if (std.mem.readInt(i64, self.multipliers[8 * i ..][0..8], .little) != g.multipliers[i]) return error.NgramMismatch;
    }
};

/// A PLE layer's own tensors (exl3.py `ple_layer`): the key and value projections (fp16), the centred norms (fp32),
/// the conv taps (bf16 [S * D, taps]) and the table's head biases (fp16).
pub const Ple = struct {
    key: F16,
    value: F16,
    norm_key: cuda.DeviceBuffer,
    norm_query: cuda.DeviceBuffer,
    norm_conv: cuda.DeviceBuffer,
    conv: cuda.DeviceBuffer,
    head_bias: cuda.DeviceBuffer,
    taps: usize,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, pack: *Pack, cfg: *const Config, table: *const NgramTable, i: usize, offset: f32) !Ple {
        const w_at: names.Where = .{ .lm = i };
        var n: names.Name = .{};
        const wide = cfg.wide();
        const key_t = try pack.expect(names.layer(&n, w_at, "ple.key_proj.weight"), .f16, &.{ wide, cfg.hidden });
        var key = try F16.init(d, key_t.bytes, wide, cfg.hidden);
        errdefer key.deinit();
        const val_t = try pack.expect(names.layer(&n, w_at, "ple.value_proj.weight"), .f16, &.{ cfg.hidden, cfg.hidden });
        var value = try F16.init(d, val_t.bytes, cfg.hidden, cfg.hidden);
        errdefer value.deinit();
        var norms: [3]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (norms[0..made]) |*b| b.free();
        for ([_][]const u8{ "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight" }, &norms) |nm, *b| {
            const t = try pack.get(n.of(names.lm ++ "layers.{d}.{s}", .{ i, nm }));
            if (t.numel() != wide) return error.UnexpectedTensor;
            const v = try widen(gpa, t);
            defer gpa.free(v);
            for (v) |*x| x.* += offset;
            b.* = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(v));
            made += 1;
        }
        const conv_t = try pack.get(names.layer(&n, w_at, "ple.conv1d.weight"));
        if (conv_t.rank != 3 or conv_t.shape[0] != wide or conv_t.shape[1] != 1) return error.UnexpectedTensor;
        const conv = try gpa.alloc(u16, conv_t.numel());
        defer gpa.free(conv);
        for (conv, 0..) |*c, j| c.* = switch (conv_t.dtype) {
            .bf16 => std.mem.readInt(u16, conv_t.bytes[2 * j ..][0..2], .little),
            else => f32ToBf16(@floatCast(try element(conv_t, j))), // `.to(torch.bfloat16)`
        };
        var conv_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(conv));
        errdefer conv_d.free();
        return .{
            .key = key, .value = value, .norm_key = norms[0], .norm_query = norms[1], .norm_conv = norms[2], .conv = conv_d,
            .head_bias = try cuda.DeviceBuffer.fromHost(d, table.head_bias), .taps = conv_t.shape[2],
        };
    }

    pub fn deinit(self: *Ple) void {
        self.key.deinit();
        self.value.deinit();
        self.norm_key.free();
        self.norm_query.free();
        self.norm_conv.free();
        self.conv.free();
        self.head_bias.free();
    }
};

/// A sparse-attention layer's own tensors (exl3.py `attention`): the projection [q (with its gate) | k | v | index qk]
/// as four EXL3 groups, the centred q, k, index-q and index-k norms (fp32), and o_proj.
pub const Attn = struct {
    q: Group,
    k: Group,
    v: Group,
    iqk: Group,
    o: Group,
    q_scale: cuda.DeviceBuffer,
    k_scale: cuda.DeviceBuffer,
    iq_scale: cuda.DeviceBuffer,
    ik_scale: cuda.DeviceBuffer,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, pack: *Pack, w_at: names.Where, offset: f32) !Attn {
        var n: names.Name = .{};
        var a: Attn = undefined;
        a.q = try pack.group(names.layer(&n, w_at, "self_attn.q_proj"));
        a.k = try pack.group(names.layer(&n, w_at, "self_attn.k_proj"));
        a.v = try pack.group(names.layer(&n, w_at, "self_attn.v_proj"));
        a.iqk = try pack.group(names.layer(&n, w_at, "self_attn.indexer.index_qk_proj"));
        a.o = try pack.group(names.layer(&n, w_at, "self_attn.o_proj"));
        const parts = [_][]const u8{ "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.indexer.q_layernorm.weight", "self_attn.indexer.k_layernorm.weight" };
        var bufs: [4]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (parts, &bufs) |p, *b| {
            const t = try pack.get(names.layerAny(&n, w_at, p));
            const v = try widen(gpa, t);
            defer gpa.free(v);
            for (v) |*x| x.* += offset;
            b.* = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(v));
            made += 1;
        }
        a.q_scale, a.k_scale, a.iq_scale, a.ik_scale = bufs;
        return a;
    }

    pub fn deinit(self: *Attn) void {
        self.q_scale.free();
        self.k_scale.free();
        self.iq_scale.free();
        self.ik_scale.free();
    }

    /// The projection row's width: q (heads * head_dim * 2, the gate interleaved as 0.6.6 stores it) + k + v + index.
    pub fn projWidth(self: *const Attn) usize {
        return self.q.n() + self.k.n() + self.v.n() + self.iqk.n();
    }
};

/// The rotary inverse frequencies (exl3.py `load`): theta ** (-j / half) in fp64, j < half, then rounded to fp32.
pub fn invFreq(out: []f32, theta: f64) void {
    const half: f64 = @floatFromInt(out.len);
    for (out, 0..) |*v, j| v.* = @floatCast(std.math.pow(f64, theta, -@as(f64, @floatFromInt(j)) / half));
}

test "inverse frequencies round as fp64 pow then fp32" {
    var got: [32]f32 = undefined;
    invFreq(&got, 10_000_000);
    // numpy float64 ** then astype(float32), the same arithmetic as the loader's torch float64 pow
    const want = [_]u32{ 0x3f800000, 0x3f1ab32b, 0x3ebaf81a, 0x3e61f836, 0x3e088d77, 0x3da50957, 0x3d47763f, 0x3cf11176, 0x3c91ad39, 0x3c301052, 0x3bd4ca14, 0x3b80967d, 0x3b1b690d, 0x3abbd3ec, 0x3a6301e2, 0x3a092e02, 0x39a5cb5f, 0x394860c1, 0x38f22ce3, 0x3892587f, 0x3830df51, 0x37d5c442, 0x37812dac, 0x371c1fc4, 0x36bcb0c1, 0x36640cc6, 0x3609cf4b, 0x35a68e4c, 0x35494c56, 0x34f3499c, 0x3493048e, 0x3431af44 };
    for (got, want) |g, w| try std.testing.expectEqual(w, @as(u32, @bitCast(g)));
}
