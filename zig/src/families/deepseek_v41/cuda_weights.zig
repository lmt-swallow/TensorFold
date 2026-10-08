//! This rank's weights on the device, read from the Python family's rank cache (rank_cache.zig: the TP split already
//! applied, tensors in the loader's order) and packed into the kernels' layouts (cuda_exl3.Linear strips, cuda_experts
//! trellis pack). The keys are weights.py's: `{prefix}|{cols}|{rows}|tr|suh|svh` for EXL3 parts (Python tuple text,
//! `None` when unsliced) and `{name}|{dtype}` for plain tensors (`torch.float32` or `None` for the stored dtype).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const map = @import("weights_map.zig");
const RankCache = @import("rank_cache.zig").RankCache;
const exl3 = @import("cuda_exl3.zig");
const experts_mod = @import("cuda_experts.zig");

pub const Error = error{ MissingTensor, BadShape, OutOfMemory, NoSpaceLeft } || RankCache.Error || cuda.Error || exl3.Error || experts_mod.Error;

/// A plain tensor on the device (norm weights, mixes, gates, sinks).
pub const Plain = struct {
    buf: cuda.DeviceBuffer,
    bytes: usize,
    count: usize,

    pub fn ptr(self: *const Plain) cuda.abi.DevicePtr {
        return self.buf.ptr;
    }
};

pub const Layer = struct {
    idx: usize,
    ratio: usize,
    hc_attn_fn: Plain,
    hc_attn_scale: Plain,
    hc_attn_base: Plain,
    hc_ffn_fn: Plain,
    hc_ffn_scale: Plain,
    hc_ffn_base: Plain,
    attn_norm: Plain,
    ffn_norm: Plain,
    wq_a: exl3.Linear,
    q_norm: Plain,
    wq_b: exl3.Linear,
    wkv: exl3.Linear,
    kv_norm: Plain,
    sink: Plain,
    wo_a: [8]exl3.Linear, // this rank's o_groups / world slices, in order
    wo_a_count: usize,
    wo_b: exl3.Linear,
    comp_wkv: ?exl3.Linear = null,
    comp_wgate: ?exl3.Linear = null,
    comp_norm: ?Plain = null,
    idx_wq_b: ?exl3.Linear = null,
    idx_proj: ?Plain = null,
    idx_wk: ?exl3.Linear = null,
    idx_k_norm: ?Plain = null,
    gate_w: Plain, // f16 [E, d]
    gate_b: Plain, // f32 [E]
    experts: experts_mod.Experts,
    engram_wkv: ?exl3.Linear = null,
    engram_qk: ?Plain = null, // q_weight * k_weight [hc, d] f32 (made here from the two cached tensors)
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    cfg: *const Config,
    rank: usize,
    world: usize,
    embed: Plain, // bf16 [V, d]
    norm: Plain,
    head: exl3.Linear, // this rank's vocabulary columns
    vocab: [2]usize,
    layers: []Layer,

    pub fn deinit(self: *Weights) void {
        for (self.layers) |*l| freeLayer(l);
        self.gpa.free(self.layers);
        self.embed.buf.free();
        self.norm.buf.free();
        self.head.deinit();
    }
};

/// Loads the backbone layers [0, n_layers_wanted) (all of them when null) for `rank` of `world`.
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const Config, rc: *const RankCache, kernels: *const exl3.Kernels, rank: usize, world: usize, cb: exl3.Codebook, n_layers_wanted: ?usize) Error!Weights {
    var w: Weights = .{ .gpa = gpa, .cfg = cfg, .rank = rank, .world = world, .embed = undefined, .norm = undefined, .head = undefined, .vocab = map.vocabRange(cfg, rank, world), .layers = &.{} };
    w.embed = try plain(d, rc, "embed.weight", null);
    errdefer w.embed.buf.free();
    w.norm = try plain(d, rc, "norm.weight", null);
    errdefer w.norm.buf.free();
    w.head = try linear(gpa, d, rc, kernels, "head", .{ .cols = w.vocab }, cb);
    errdefer w.head.deinit();
    const n = n_layers_wanted orelse cfg.n_layers;
    const layers = try gpa.alloc(Layer, n);
    errdefer gpa.free(layers);
    var done: usize = 0;
    errdefer for (layers[0..done]) |*l| freeLayer(l);
    for (layers, 0..) |*l, i| {
        l.* = try loadBlock(gpa, d, cfg, rc, kernels, i, rank, world, cb);
        done += 1;
    }
    w.layers = layers;
    return w;
}

fn freeLayer(l: *Layer) void {
    inline for (.{ "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "attn_norm", "ffn_norm", "q_norm", "kv_norm", "sink", "gate_w", "gate_b" }) |f| @field(l, f).buf.free();
    inline for (.{ "wq_a", "wq_b", "wkv", "wo_b" }) |f| @field(l, f).deinit();
    for (l.wo_a[0..l.wo_a_count]) |*x| x.deinit();
    inline for (.{ "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk", "engram_wkv" }) |f| if (@field(l, f)) |*x| x.deinit();
    inline for (.{ "comp_norm", "idx_proj", "idx_k_norm", "engram_qk" }) |f| if (@field(l, f)) |*x| x.buf.free();
    l.experts.deinit();
}

/// Block `i` as weights.py's `load_block`, through the rank cache.
pub fn loadBlock(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const Config, rc: *const RankCache, kernels: *const exl3.Kernels, i: usize, rank: usize, world: usize, cb: exl3.Codebook) Error!Layer {
    var pbuf: [32]u8 = undefined;
    const p = map.prefix(&pbuf, cfg, i);
    const hl = cfg.n_heads / world * cfg.head_dim;
    const gl = cfg.o_groups / world;
    var l: Layer = undefined;
    l.idx = i;
    l.ratio = cfg.ratio(i);
    l.comp_wkv = null;
    l.comp_wgate = null;
    l.comp_norm = null;
    l.idx_wq_b = null;
    l.idx_proj = null;
    l.idx_wk = null;
    l.idx_k_norm = null;
    l.engram_wkv = null;
    l.engram_qk = null;
    l.wo_a_count = 0;
    l.hc_attn_fn = try plainOf(d, rc, p, "hc_attn_fn", "torch.float32");
    l.hc_attn_scale = try plainOf(d, rc, p, "hc_attn_scale", "torch.float32");
    l.hc_attn_base = try plainOf(d, rc, p, "hc_attn_base", "torch.float32");
    l.hc_ffn_fn = try plainOf(d, rc, p, "hc_ffn_fn", "torch.float32");
    l.hc_ffn_scale = try plainOf(d, rc, p, "hc_ffn_scale", "torch.float32");
    l.hc_ffn_base = try plainOf(d, rc, p, "hc_ffn_base", "torch.float32");
    l.attn_norm = try plainOf(d, rc, p, "attn_norm.weight", null);
    l.ffn_norm = try plainOf(d, rc, p, "ffn_norm.weight", null);
    l.wq_a = try linearOf(gpa, d, rc, kernels, p, "attn.wq_a", .whole, cb);
    l.q_norm = try plainOf(d, rc, p, "attn.q_norm.weight", null);
    l.wq_b = try linearOf(gpa, d, rc, kernels, p, "attn.wq_b", .{ .cols = .{ rank * hl, (rank + 1) * hl } }, cb);
    l.wkv = try linearOf(gpa, d, rc, kernels, p, "attn.wkv", .whole, cb);
    l.kv_norm = try plainOf(d, rc, p, "attn.kv_norm.weight", null);
    // the cache holds every head's sink (weights.py slices after `_kept`): this rank's heads' slice
    l.sink = try plainOf(d, rc, p, "attn.attn_sink", "torch.float32");
    if (l.sink.count == cfg.n_heads and world > 1) {
        const hl_heads = cfg.n_heads / world;
        var part = try cuda.DeviceBuffer.alloc(d, hl_heads * 4);
        errdefer part.free();
        try part.copyFrom(0, l.sink.buf.ptr + rank * hl_heads * 4, hl_heads * 4, null);
        l.sink.buf.free();
        l.sink = .{ .buf = part, .bytes = hl_heads * 4, .count = hl_heads };
    }
    for (0..gl) |g| {
        var nb: [48]u8 = undefined;
        const name = std.fmt.bufPrint(&nb, "attn.wo_a.slice.{d}", .{rank * gl + g}) catch unreachable;
        l.wo_a[g] = try linearOf(gpa, d, rc, kernels, p, name, .whole, cb);
        l.wo_a_count = g + 1;
    }
    l.wo_b = try linearOf(gpa, d, rc, kernels, p, "attn.wo_b", .{ .rows = .{ rank * gl * cfg.o_rank, (rank + 1) * gl * cfg.o_rank } }, cb);
    if (cfg.isKvSource(i)) {
        l.comp_wkv = try linearOf(gpa, d, rc, kernels, p, "attn.compressor.wkv", .whole, cb);
        l.comp_norm = try plainOf(d, rc, p, "attn.compressor.norm.weight", null);
        if (l.ratio > 1) l.comp_wgate = try linearOf(gpa, d, rc, kernels, p, "attn.compressor.wgate", .whole, cb);
    }
    if (cfg.isIndexSource(i)) {
        l.idx_wq_b = try linearOf(gpa, d, rc, kernels, p, "attn.indexer.wq_b", .whole, cb);
        l.idx_proj = try plainOf(d, rc, p, "attn.indexer.weights_proj.weight", "torch.float32");
        if (cfg.isKvSource(i)) {
            l.idx_wk = try linearOf(gpa, d, rc, kernels, p, "attn.indexer.wk", .whole, cb);
            l.idx_k_norm = try plainOf(d, rc, p, "attn.indexer.k_norm.weight", null);
        }
    }
    l.gate_w = try plainOf(d, rc, p, "ffn.gate.weight", "torch.float16");
    l.gate_b = try plainOf(d, rc, p, "ffn.gate.bias", "torch.float32");
    l.experts = try expertsOf(gpa, d, cfg, rc, p, rank, world, cb, if (i < cfg.n_layers) cfg.n_routed else cfg.dspark_routed);
    if (cfg.isEngram(i)) {
        const per = (cfg.engram_ngram - 1) * cfg.engram_heads / world * cfg.engram_dim;
        l.engram_wkv = try linearOf(gpa, d, rc, kernels, p, "engram.wkv", .{ .rows = .{ rank * per, (rank + 1) * per } }, cb);
        l.engram_qk = try engramQk(gpa, d, rc, p);
    }
    return l;
}

/// The drafter's own weights (weights.py DSparkWeights): the stage blocks (mtp.j, dspark_routed experts each), the
/// taps' projection and norm, the last stage's norm, the Markov bigram head and the confidence head.
pub const DSpark = struct {
    blocks: []Layer,
    main_proj: exl3.Linear,
    main_norm: Plain,
    norm: Plain,
    markov_embed: Plain, // bf16 [V, rank]
    markov_head: Plain, // f16 [V, rank]
    conf: Plain, // f16 [1, d + rank]
    markov_embed_host: []u16, // the same bf16 bits on the host (the confidence head's rows)
    conf_host: []u16,

    pub fn deinit(self: *DSpark, gpa: std.mem.Allocator) void {
        for (self.blocks) |*l| freeLayer(l);
        gpa.free(self.blocks);
        self.main_proj.deinit();
        inline for (.{ "main_norm", "norm", "markov_embed", "markov_head", "conf" }) |f| @field(self, f).buf.free();
        gpa.free(self.markov_embed_host);
        gpa.free(self.conf_host);
    }
};

/// The DSpark stages and heads for `rank` of `world` (the rank cache holds them when the Python server loaded with
/// drafts).
pub fn loadDSpark(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const Config, rc: *const RankCache, kernels: *const exl3.Kernels, rank: usize, world: usize, cb: exl3.Codebook) Error!DSpark {
    const n = cfg.nextn;
    if (n == 0) return error.MissingTensor;
    const blocks = try gpa.alloc(Layer, n);
    errdefer gpa.free(blocks);
    var done: usize = 0;
    errdefer for (blocks[0..done]) |*l| freeLayer(l);
    for (blocks, 0..) |*l, j| {
        l.* = try loadBlock(gpa, d, cfg, rc, kernels, cfg.n_layers + j, rank, world, cb);
        done += 1;
    }
    var lb: [32]u8 = undefined;
    const last = std.fmt.bufPrint(&lb, "mtp.{d}", .{n - 1}) catch unreachable;
    var main_proj = try linear(gpa, d, rc, kernels, "mtp.0.main_proj", .whole, cb);
    errdefer main_proj.deinit();
    var main_norm = try plain(d, rc, "mtp.0.main_norm.weight", null);
    errdefer main_norm.buf.free();
    var norm = try plainOf(d, rc, last, "norm.weight", null);
    errdefer norm.buf.free();
    var markov_embed = try plainOf(d, rc, last, "markov_head.embed.weight", null);
    errdefer markov_embed.buf.free();
    var markov_head = try plainOf(d, rc, last, "markov_head.head.weight", null);
    errdefer markov_head.buf.free();
    var conf = try plainOf(d, rc, last, "confidence_head.proj.weight", null);
    errdefer conf.buf.free();
    var nb: [96]u8 = undefined;
    var kb: [160]u8 = undefined;
    const me = try readAlloc(gpa, rc, plainKey(&kb, std.fmt.bufPrint(&nb, "{s}.markov_head.embed.weight", .{last}) catch unreachable, null));
    errdefer gpa.free(me);
    const cf = try readAlloc(gpa, rc, plainKey(&kb, std.fmt.bufPrint(&nb, "{s}.confidence_head.proj.weight", .{last}) catch unreachable, null));
    errdefer gpa.free(cf);
    return .{
        .blocks = blocks, .main_proj = main_proj, .main_norm = main_norm, .norm = norm, .markov_embed = markov_embed,
        .markov_head = markov_head, .conf = conf,
        .markov_embed_host = @alignCast(std.mem.bytesAsSlice(u16, me)), .conf_host = @alignCast(std.mem.bytesAsSlice(u16, cf)),
    };
}

// -- the rank cache's keys ----------------------------------------------------------------------------------------

/// `{prefix}|{cols}|{rows}|{part}` with Python's tuple text.
pub fn exl3Key(buf: []u8, prefix: []const u8, split: map.Split, part: []const u8) []const u8 {
    return switch (split) {
        .whole => std.fmt.bufPrint(buf, "{s}|None|None|{s}", .{ prefix, part }),
        .cols => |c| std.fmt.bufPrint(buf, "{s}|({d}, {d})|None|{s}", .{ prefix, c[0], c[1], part }),
        .rows => |r| std.fmt.bufPrint(buf, "{s}|None|({d}, {d})|{s}", .{ prefix, r[0], r[1], part }),
    } catch unreachable;
}

/// `{name}|{dtype}` (`None` when the loader keeps the stored dtype).
pub fn plainKey(buf: []u8, name: []const u8, dtype: ?[]const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}|{s}", .{ name, dtype orelse "None" }) catch unreachable;
}

fn readAlloc(gpa: std.mem.Allocator, rc: *const RankCache, key: []const u8) Error![]u8 {
    const e = rc.find(key) orelse {
        std.log.err("rank cache has no {s}", .{key});
        return error.MissingTensor;
    };
    const bytes = try gpa.alloc(u8, @intCast(e.bytes));
    errdefer gpa.free(bytes);
    try rc.read(e, bytes);
    return bytes;
}

fn plain(d: *const cuda.Driver, rc: *const RankCache, name: []const u8, dtype: ?[]const u8) Error!Plain {
    var kb: [160]u8 = undefined;
    const key = plainKey(&kb, name, dtype);
    const e = rc.find(key) orelse {
        std.log.err("rank cache has no {s}", .{key});
        return error.MissingTensor;
    };
    var buf = try cuda.DeviceBuffer.alloc(d, @intCast(@max(e.bytes, 1)));
    errdefer buf.free();
    if (e.bytes > 0) {
        // through a host buffer in 256 MiB pieces (the embedding is 1.3 GB)
        const piece: usize = 256 << 20;
        var tmp = try std.heap.page_allocator.alloc(u8, @min(piece, @as(usize, @intCast(e.bytes))));
        defer std.heap.page_allocator.free(tmp);
        var off: usize = 0;
        while (off < e.bytes) {
            const n = @min(tmp.len, @as(usize, @intCast(e.bytes)) - off);
            try preadInto(rc, e.offset + off, tmp[0..n]);
            try buf.upload(off, tmp[0..n]);
            off += n;
        }
    }
    return .{ .buf = buf, .bytes = @intCast(e.bytes), .count = e.count() };
}

fn preadInto(rc: *const RankCache, offset: u64, out: []u8) Error!void {
    const fake: @import("rank_cache.zig").Entry = .{ .key = "", .dtype = .other, .shape = .{ 1, 1, 1, 1 }, .rank = 1, .offset = offset, .bytes = out.len };
    try rc.read(&fake, out);
}

fn plainOf(d: *const cuda.Driver, rc: *const RankCache, prefix: []const u8, name: []const u8, dtype: ?[]const u8) Error!Plain {
    var nb: [128]u8 = undefined;
    const full = std.fmt.bufPrint(&nb, "{s}.{s}", .{ prefix, name }) catch unreachable;
    return plain(d, rc, full, dtype);
}

/// An EXL3 group from the cache: trellis `[kt, nt, w16] int16`, suh, svh; packed by cuda_exl3.Linear.
fn linear(gpa: std.mem.Allocator, d: *const cuda.Driver, rc: *const RankCache, kernels: *const exl3.Kernels, prefix: []const u8, split: map.Split, cb: exl3.Codebook) Error!exl3.Linear {
    _ = d;
    var kb: [200]u8 = undefined;
    const tr_e = rc.find(exl3Key(&kb, prefix, split, "tr")) orelse {
        std.log.err("rank cache has no {s}", .{exl3Key(&kb, prefix, split, "tr")});
        return error.MissingTensor;
    };
    if (tr_e.rank != 3) return error.BadShape;
    const tr = try gpa.alloc(u8, @intCast(tr_e.bytes));
    defer gpa.free(tr);
    try rc.read(tr_e, tr);
    const suh = try readAlloc(gpa, rc, exl3Key(&kb, prefix, split, "suh"));
    defer gpa.free(suh);
    const svh = try readAlloc(gpa, rc, exl3Key(&kb, prefix, split, "svh"));
    defer gpa.free(svh);
    return exl3.Linear.init(gpa, kernels, tr, tr_e.shape[0], tr_e.shape[1], tr_e.shape[2], suh, svh, cb);
}

fn linearOf(gpa: std.mem.Allocator, d: *const cuda.Driver, rc: *const RankCache, kernels: *const exl3.Kernels, prefix: []const u8, name: []const u8, split: map.Split, cb: exl3.Codebook) Error!exl3.Linear {
    var nb: [128]u8 = undefined;
    const full = std.fmt.bufPrint(&nb, "{s}.{s}", .{ prefix, name }) catch unreachable;
    return linear(gpa, d, rc, kernels, full, split, cb);
}

/// The block's routed experts and the shared expert (last), this rank's halves, into one device pack.
fn expertsOf(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const Config, rc: *const RankCache, p: []const u8, rank: usize, world: usize, cb: exl3.Codebook, n_routed: usize) Error!experts_mod.Experts {
    const es = map.expertSplit(cfg.inter, rank, world);
    const n = n_routed + 1;
    const mats = try gpa.alloc(experts_mod.ExpertMats, n);
    defer gpa.free(mats);
    var owned = std.ArrayList([]u8).empty;
    defer {
        for (owned.items) |b| gpa.free(b);
        owned.deinit(gpa);
    }
    for (0..n) |e| {
        var eb: [96]u8 = undefined;
        const ep = map.expertPrefix(&eb, p, e, n_routed);
        var m: experts_mod.ExpertMats = undefined;
        inline for (.{ "w1", "w3", "w2" }, .{ "gate", "up", "down" }) |src, dst| {
            var nb: [128]u8 = undefined;
            const full = std.fmt.bufPrint(&nb, "{s}.{s}", .{ ep, src }) catch unreachable;
            const split: map.Split = if (std.mem.eql(u8, src, "w2")) es.down else es.gate_up;
            var kb: [220]u8 = undefined;
            const tr_e = rc.find(exl3Key(&kb, full, split, "tr")) orelse {
                std.log.err("rank cache has no {s}", .{exl3Key(&kb, full, split, "tr")});
                return error.MissingTensor;
            };
            if (tr_e.rank != 3) return error.BadShape;
            const tr = try readAlloc(gpa, rc, exl3Key(&kb, full, split, "tr"));
            try owned.append(gpa, tr);
            const suh = try readAlloc(gpa, rc, exl3Key(&kb, full, split, "suh"));
            try owned.append(gpa, suh);
            const svh = try readAlloc(gpa, rc, exl3Key(&kb, full, split, "svh"));
            try owned.append(gpa, svh);
            @field(m, dst) = .{ .trellis = tr, .kt = tr_e.shape[0], .nt = tr_e.shape[1], .w16 = tr_e.shape[2], .suh = suh, .svh = svh };
        }
        mats[e] = m;
    }
    return experts_mod.Experts.init(gpa, d, mats, cb);
}

/// q_weight * k_weight [hc, d] f32 (weights.py makes the product on load; the cache holds the two factors).
fn engramQk(gpa: std.mem.Allocator, d: *const cuda.Driver, rc: *const RankCache, p: []const u8) Error!Plain {
    var nb: [128]u8 = undefined;
    const qk = std.fmt.bufPrint(&nb, "{s}.engram.q_weight", .{p}) catch unreachable;
    var kb: [160]u8 = undefined;
    const q = try readAlloc(gpa, rc, plainKey(&kb, qk, "torch.float32"));
    defer gpa.free(q);
    const kk = std.fmt.bufPrint(&nb, "{s}.engram.k_weight", .{p}) catch unreachable;
    const k = try readAlloc(gpa, rc, plainKey(&kb, kk, "torch.float32"));
    defer gpa.free(k);
    if (q.len != k.len) return error.BadShape;
    const qf = std.mem.bytesAsSlice(f32, q);
    const kf = std.mem.bytesAsSlice(f32, k);
    const prod = try gpa.alloc(f32, qf.len);
    defer gpa.free(prod);
    for (prod, qf, kf) |*o, a, b| o.* = a * b;
    const buf = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(prod));
    return .{ .buf = buf, .bytes = q.len, .count = qf.len };
}

test "rank cache keys follow weights.py" {
    var b: [200]u8 = undefined;
    try std.testing.expectEqualStrings("layers.0.attn.wq_a|None|None|tr", exl3Key(&b, "layers.0.attn.wq_a", .whole, "tr"));
    try std.testing.expectEqualStrings("layers.0.attn.wq_b|(16384, 32768)|None|svh", exl3Key(&b, "layers.0.attn.wq_b", .{ .cols = .{ 16384, 32768 } }, "svh"));
    try std.testing.expectEqualStrings("layers.3.ffn.experts.7.w2|None|(0, 1152)|suh", exl3Key(&b, "layers.3.ffn.experts.7.w2", .{ .rows = .{ 0, 1152 } }, "suh"));
    try std.testing.expectEqualStrings("layers.0.hc_attn_fn|torch.float32", plainKey(&b, "layers.0.hc_attn_fn", "torch.float32"));
    try std.testing.expectEqualStrings("layers.0.attn_norm.weight|None", plainKey(&b, "layers.0.attn_norm.weight", null));
}
