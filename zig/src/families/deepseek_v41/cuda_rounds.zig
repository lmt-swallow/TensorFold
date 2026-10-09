//! Concurrent decode rounds (`--parallel`, the Python family's rounds.py `RoundDecoder` and model.py `PoolCache`):
//! one forward over every live stream's verify window, every row at its own stream's position and in its own
//! stream's cache rows. A pool is one window of `cap` positions whose compressed planes the streams share by
//! extents (a stream owns positions [base, base + size) of the planes, its last compressed row its scratch row), and
//! per-slot window rings and compressor inputs. A row's kernels address its stream's rows through per-row tables
//! (ids, positions, slot, extent base and end), the fork's `_ix` arithmetic computed here on the host.
//!
//! The fused path (default; the fork's default switches, rounds.py `_layers_fused` + `_attention2`): each sublayer's
//! hc_post fused into the next sublayer's mixes with the Sinkhorn half and the mixes' dots on a side stream
//! (`_hc_post_only`, `_hc_finish_s`, `_hc_mix_part`, `_hc_sinkhorn`), attn_in, wq_b and wo_a / wo_b as grouped EXL3
//! launches (`exl3.Group`: rot_many + glinear) with q's RMSNorm writing wq_b's rotated rows beside the window KV's
//! norm (`_q_kv_norm`), q's RoPE in wq_b's epilogue, the attention merge applying the inverse RoPE and rotating wo_a's
//! input (`_sparse_attn_merge_rot2`), wo_a's epilogue rotating wo_b's input, and the paced L2 prefetch forks (`L2`,
//! TF_DS_L2_PREFETCH). TF_DS_FUSED=0 keeps the plain path (the fork's TF_DS_HC_FUSED=0, TF_DS_ROT_*=0: plain hc_pre /
//! hc_post, the q path in three launches, the merge apart from wo_a); both write the same bits. The indexer's
//! selection: at short buckets (bucket / ratio <= idx_topk) every compressed entry below the row's visible count
//! (`alltop`, no scores); past them the scores of every row against its own stream's extent (`_index_score` with
//! per-row bases), the candidate pool at the candidate source and its mask after it, and the top-k on the device
//! (cuda_attention.zig's selection: the same set as the fork's `topk_indices`, a total order). Engram runs on its
//! layers when a hook is set (`engram`), each row's n-grams from its own stream's ids (`Window.host`).
//!
//! A round scans a context bucket sized by its windows, not the pool (graph.py `bucket_for`): the smallest power of
//! two, `bucket_min` (BUCKET_MIN) at least, holding the deepest row's position + 1, so its cost follows the live
//! streams' lengths. Every row's bits are the same at any bucket holding it (the scans mask past a row's visible
//! count); the fork's round runs at this bucket, so its launches do too. The bucket depends only on the windows,
//! so ranks running the same windows run the same shapes.
//!
//! Graphs (rounds.py `RoundDecoder.capture`, TF_DS_GRAPHS=0 keeps every round eager): a round's launches depend on the
//! host only through its row count R and its bucket; every per-row value (ids, positions, ring and raw rows, extents,
//! each ratio's compressed rows and visible counts, the short buckets' picks) is a device table filled by one copy
//! before the launches (`setTables`, the fork's `inputs.copy_`). So the launches of a key (R, bucket) are recorded
//! once as CUDA graphs, one a stretch of layers (cut before each Engram layer when Engram runs: its host read and
//! upload stay eager between the stretches, as the fork's ENGRAM_SPLIT), and replayed; a replay runs the eager path's
//! launches with the same arguments, so it writes the same bytes. The gathers are recorded with the rest (the
//! communicator's all-gather runs on the round's stream). The host selection (TF_DS_TOPK=host), a probe and buckets
//! past `graph_bucket_max` stay eager. Every rank records the same graphs in the same order: the keys follow from
//! the windows alone.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const exl3 = @import("cuda_exl3.zig");
const experts = @import("cuda_experts.zig");
const hcmod = @import("cuda_hc.zig");
const routemod = @import("cuda_route.zig");
const attention = @import("cuda_attention.zig");
const fwd = @import("cuda_forward.zig");

const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;

pub const Error = error{ TooManyRows, BadShape, Unsupported, OutOfMemory } || cuda.Error || exl3.Error || experts.Error;

pub const max_rows = 16;
pub const raw_rows = 64; // compressor inputs kept by position a slot (model.py RAW)
/// The smallest context bucket (graph.py BUCKET_MIN).
pub const bucket_min = 1024;

/// graph.py `bucket_for(deepest, cap)`: the smallest power of two >= `deepest`, `least` at least, then at most
/// max(cap, least). Monotone in `deepest`; >= `deepest` whenever `deepest` <= max(cap, least).
pub fn bucketFor(deepest: usize, cap: usize, least: usize) usize {
    var b = least;
    while (b < deepest) b *= 2;
    return @min(b, @max(cap, least));
}

/// One stream's verify window in a round (multi.py `_round`'s tuple): its slot, its extent, the first row's
/// position (relative to the extent) and the rows' ids (the pending token, then the drafts). `host`: the stream's
/// ids from position 0 through the window's last row (`position + ids.len` of them at least), the n-gram history
/// Engram hashes; read only when an Engram hook is set.
pub const Window = struct {
    slot: usize,
    base: usize,
    size: usize,
    position: usize,
    ids: []const i64,
    host: []const i32 = &.{},
};

/// One stream's rows of a round for Engram: positions start .. start + n - 1 of the sequence `ids`.
pub const EngramSpan = struct { ids: []const i32, start: usize, n: usize };

/// Engram on an Engram layer of a round (rounds.py `_stretch`: `engram_gate(h, Comm.sum(mm(engram_wkv, e_in)))`):
/// h [R, 4, d] bf16, the rows the spans' in order, gated in place; one gather over the ranks (every rank calls it at
/// the same layers). null hook: no tables (the fork skips the gate).
pub const Engram = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, layer: usize, h: Ptr, spans: []const EngramSpan, gather: fwd.Gather, stream: cuda.Stream) anyerror!void,
};

/// A kv-source layer's compressed planes of the whole window (shared by position) and every slot's compressor inputs.
pub const PoolPlanes = struct {
    comp_codes: cuda.DeviceBuffer, // u8 [rows, hd / 2]
    comp_scales: cuda.DeviceBuffer, // u8 [rows, hd / 16]
    idxk_codes: cuda.DeviceBuffer, // u8 [rows, idx_dim / 2] (index sources)
    idxk_scales: cuda.DeviceBuffer, // u8 [rows, idx_dim / 32]
    raw_kv: cuda.DeviceBuffer, // f32 [slots * raw, hd] (ratio > 1)
    raw_score: cuda.DeviceBuffer,
    rows: usize,

    /// The bytes of each plane of layer `li` for a window of `cap` positions and `slots` slots.
    pub const Sizes = struct {
        rows: usize,
        comp_codes: usize,
        comp_scales: usize,
        idxk_codes: usize,
        idxk_scales: usize,
        raw: usize, // raw_kv's, and raw_score's

        pub fn total(self: Sizes) usize {
            return self.comp_codes + self.comp_scales + self.idxk_codes + self.idxk_scales + 2 * self.raw;
        }
    };

    pub fn sizes(cfg: *const Config, li: usize, cap: usize, slots: usize) Sizes {
        const ratio = cfg.ratio(li);
        const rows = cap / ratio + 2;
        const hd = cfg.head_dim;
        const indexed = cfg.isIndexSource(li);
        return .{
            .rows = rows,
            .comp_codes = rows * hd / 2,
            .comp_scales = rows * hd / 16,
            .idxk_codes = if (indexed) rows * cfg.idx_dim / 2 else 0,
            .idxk_scales = if (indexed) rows * cfg.idx_dim / 32 else 0,
            .raw = if (ratio > 1) slots * raw_rows * hd * 4 else 0,
        };
    }

    fn init(d: *const cuda.Driver, cfg: *const Config, li: usize, cap: usize, slots: usize) !PoolPlanes {
        const sz = sizes(cfg, li, cap, slots);
        var p: PoolPlanes = undefined;
        p.rows = sz.rows;
        p.comp_codes = try cuda.DeviceBuffer.alloc(d, sz.comp_codes);
        errdefer p.comp_codes.free();
        p.comp_scales = try cuda.DeviceBuffer.alloc(d, sz.comp_scales);
        errdefer p.comp_scales.free();
        p.idxk_codes = try cuda.DeviceBuffer.alloc(d, sz.idxk_codes);
        errdefer p.idxk_codes.free();
        p.idxk_scales = try cuda.DeviceBuffer.alloc(d, sz.idxk_scales);
        errdefer p.idxk_scales.free();
        p.raw_kv = try cuda.DeviceBuffer.alloc(d, sz.raw);
        errdefer p.raw_kv.free();
        p.raw_score = try cuda.DeviceBuffer.alloc(d, sz.raw);
        errdefer p.raw_score.free();
        try p.comp_codes.fill8(0, null);
        try p.comp_scales.fill8(0, null);
        try p.idxk_codes.fill8(0, null);
        try p.idxk_scales.fill8(127, null);
        try p.raw_kv.fill8(0, null);
        try p.raw_score.fill8(0, null);
        return p;
    }

    fn deinit(self: *PoolPlanes) void {
        inline for (.{ "comp_codes", "comp_scales", "idxk_codes", "idxk_scales", "raw_kv", "raw_score" }) |f| @field(self, f).free();
    }
};

/// One window of `cap` positions shared by up to `slots` streams (model.py `new_pool`). Sizes are usize and the
/// rows' positions, extents and compressed rows int64, so a window of 2^20 positions and more is addressed whole.
pub const Pool = struct {
    gpa: std.mem.Allocator,
    slots: usize,
    cap: usize,
    ring: usize, // rows a slot's window ring
    rings: []cuda.DeviceBuffer, // per layer bf16 [slots * ring, hd]
    planes: []?PoolPlanes, // per layer: its own when a kv source
    source: []usize, // per layer: the kv source whose planes it reads
    tokens: cuda.DeviceBuffer, // i64 [cap]

    /// Rows a slot's window ring holds (the window and RING_EXTRA rows).
    pub fn ringRows(cfg: *const Config) usize {
        return cfg.window + 16;
    }

    /// The device bytes `init(.., layers, slots, cap)` allocates, exactly: the rings, the kv sources' planes and the
    /// token row; what a served engine reserves for its streams' caches at a context of `cap` positions.
    pub fn bytesFor(cfg: *const Config, layers: usize, cap: usize, slots: usize) usize {
        var total: usize = cap * 8;
        for (0..layers) |li| {
            total += slots * ringRows(cfg) * cfg.head_dim * 2;
            if (cfg.ratio(li) > 0 and cfg.isKvSource(li)) total += PoolPlanes.sizes(cfg, li, cap, slots).total();
        }
        return total;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const Config, layers: usize, slots: usize, cap: usize) !Pool {
        const ring = ringRows(cfg);
        const rings = try gpa.alloc(cuda.DeviceBuffer, layers);
        errdefer gpa.free(rings);
        const planes = try gpa.alloc(?PoolPlanes, layers);
        errdefer gpa.free(planes);
        const source = try gpa.alloc(usize, layers);
        errdefer gpa.free(source);
        var made: usize = 0;
        errdefer for (rings[0..made]) |*r| r.free();
        var made_p: usize = 0;
        errdefer for (planes[0..made_p]) |*p| if (p.*) |*pl| pl.deinit();
        var src: usize = 0;
        for (0..layers) |li| {
            rings[li] = try cuda.DeviceBuffer.alloc(d, slots * ring * cfg.head_dim * 2);
            try rings[li].fill8(0, null);
            made += 1;
            if (cfg.ratio(li) > 0 and cfg.isKvSource(li)) {
                planes[li] = try PoolPlanes.init(d, cfg, li, cap, slots);
                src = li;
            } else planes[li] = null;
            made_p += 1;
            source[li] = src;
        }
        var tokens = try cuda.DeviceBuffer.alloc(d, cap * 8);
        errdefer tokens.free();
        try tokens.fill8(0, null);
        return .{ .gpa = gpa, .slots = slots, .cap = cap, .ring = ring, .rings = rings, .planes = planes, .source = source, .tokens = tokens };
    }

    pub fn deinit(self: *Pool) void {
        for (self.rings) |*r| r.free();
        for (self.planes) |*p| if (p.*) |*pl| pl.deinit();
        self.gpa.free(self.rings);
        self.gpa.free(self.planes);
        self.gpa.free(self.source);
        self.tokens.free();
    }
};

/// Called before each graph launch with the gathers the graph replays, on the stream it launches on: what a test's
/// supply of the other rank's partials needs to put the next partials where the recorded gathers read them (the
/// communicator's all-gathers need nothing).
pub const GraphLaunch = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, gathers: usize, stream: cuda.Stream) anyerror!void,
};

/// Whether rounds and drafts replay CUDA graphs: TF_DS_GRAPHS unset or anything but "0".
pub fn graphsFromEnv() bool {
    const v = std.mem.span(std.c.getenv("TF_DS_GRAPHS") orelse return true);
    return !std.mem.eql(u8, v, "0");
}

/// The distinct compress ratios a runner keeps tables for.
pub const max_ratios = 4;
/// The stretches a round is cut into at most (one, plus one an Engram layer).
pub const max_segments = 9;

/// The round's device tables, int64 words: `rows` tables of max_rows words, then `per_ratio` a ratio, then each
/// ratio's picks table [max_rows, kmax].
pub const Tab = struct {
    pub const ids = 0;
    pub const pos = 1;
    pub const wslot = 2; // slot x ring + pos % ring: the row's ring row
    pub const wbase = 3; // slot x ring
    pub const rslot = 4; // slot x RAW + pos % RAW: the row's raw compressor row
    pub const rows = 5;
    // a ratio's
    pub const cbase = 0; // base / ratio: the extent's first compressed row
    pub const ctarget = 1; // the compressed row this row writes
    pub const gpos = 2; // the group's first position (its RoPE position)
    pub const gfirst = 3; // slot x RAW + gpos % RAW: the group's first raw row
    pub const vis = 4; // (pos + 1) / ratio: the compressed rows the row sees
    pub const per_ratio = 5;

    /// The word offset of table `t` of the rows.
    pub fn row(t: usize) usize {
        return t * max_rows;
    }

    /// The word offset of table `t` of ratio slot `j`.
    pub fn ratio(j: usize, t: usize) usize {
        return (rows + j * per_ratio + t) * max_rows;
    }

    /// The word offset of ratio slot `j`'s picks table.
    pub fn picks(j: usize, kmax: usize) usize {
        return (rows + max_ratios * per_ratio) * max_rows + j * max_rows * kmax;
    }

    pub fn words(kmax: usize) usize {
        return picks(max_ratios, kmax);
    }
};

/// Whether rounds (and batched drafts) take the fork's fused decode kernels: TF_DS_FUSED unset or anything but "0".
pub fn fusedFromEnv() bool {
    const v = std.mem.span(std.c.getenv("TF_DS_FUSED") orelse return true);
    return !std.mem.eql(u8, v, "0");
}

/// A layer's grouped EXL3 launches (model.py `_attn_in_group`, `q_proj`'s groups, `wo_ab`'s): the projections of
/// one input, each with its own plan, so the bits of each layer's own launches.
pub const LayerGroups = struct {
    in: exl3.Group, // wq_a, wkv (and the compressor's wkv, wgate)
    q1: exl3.Group, // wq_b
    q2: ?exl3.Group, // wq_b, idx_wq_b (an index layer)
    wo_a: exl3.Group, // the wo_a slices
    wo_b: exl3.Group,
    wo_a_suh: cuda.DeviceBuffer, // the slices' suh concatenated (the attention merge's rotation)

    pub fn init(gk: *const exl3.GroupKernels, cfg: *const Config, l: *const weights.Layer, li: usize, d: *const cuda.Driver) !LayerGroups {
        var in_l: [4]*const exl3.Linear = .{ &l.wq_a, &l.wkv, undefined, undefined };
        var n_in: usize = 2;
        if (cfg.ratio(li) > 0) if (l.comp_wkv) |*ck| {
            in_l[2] = ck;
            n_in = 3;
            if (l.comp_wgate) |*cg| {
                in_l[3] = cg;
                n_in = 4;
            }
        };
        var wo: [exl3.gmax]*const exl3.Linear = undefined;
        var suh_bytes: usize = 0;
        for (l.wo_a[0..l.wo_a_count], 0..) |*x, i| {
            wo[i] = x;
            suh_bytes += x.suh.len;
        }
        var suh = try cuda.DeviceBuffer.alloc(d, suh_bytes);
        errdefer suh.free();
        var off: usize = 0;
        for (l.wo_a[0..l.wo_a_count]) |*x| {
            try suh.copyFrom(off, x.suh.ptr, x.suh.len, null);
            off += x.suh.len;
        }
        return .{
            .in = try exl3.Group.init(gk, in_l[0..n_in]),
            .q1 = try exl3.Group.init(gk, &.{&l.wq_b}),
            .q2 = if (l.idx_wq_b) |*iq| try exl3.Group.init(gk, &.{ &l.wq_b, iq }) else null,
            .wo_a = try exl3.Group.init(gk, wo[0..l.wo_a_count]),
            .wo_b = try exl3.Group.init(gk, &.{&l.wo_b}),
            .wo_a_suh = suh,
        };
    }

    pub fn deinit(self: *LayerGroups) void {
        self.wo_a_suh.free();
    }
};

/// The decode bodies' paced L2 prefetch forks (model.py `l2_fork` / `l2_join`, TF_DS_L2_PREFETCH unset or not "0"):
/// on a side stream, once the round's stream has reached the fork, the weights the next kernels read: "attn" after
/// wq_b (wo_a's slices, wo_b; 16 MiB), "moe" after wo_b (the FFN's mHC mix, the gate, the shared expert; 8 MiB),
/// "next" after the experts (the next layer's Engram projection when Engram runs, mHC mix, input projections and
/// wq_b, or the head after the last layer; 16 MiB, 4 MiB when it ends a stretch before an Engram layer). A stretch
/// joins its forks at its end. Nothing is written: no bit changes.
pub const L2 = struct {
    gpa: std.mem.Allocator,
    on: bool,
    pf: exl3.Prefetch,
    side: cuda.Stream,
    ev: cuda.Event,
    ev_join: cuda.Event,
    live: bool = false,
    /// Whether the "next" forks include the Engram projections (the runner's hook is set).
    engram: bool = false,
    attn_r: []exl3.PArgs,
    moe_r: []exl3.PArgs,
    next_r: [][4]exl3.PArgs, // [engram x 2 + stretch end]

    pub const attn_mb = 16;
    pub const moe_mb = 8;
    pub const next_mb = 16;
    pub const stretch_end_mb = 4;

    pub fn fromEnv() bool {
        const v = std.mem.span(std.c.getenv("TF_DS_L2_PREFETCH") orelse return true);
        return !std.mem.eql(u8, v, "0");
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const Config, w: *const weights.Weights) !L2 {
        _ = cfg;
        const n = w.layers.len;
        var self: L2 = .{
            .gpa = gpa,
            .on = fromEnv(),
            .pf = try exl3.Prefetch.load(d),
            .side = undefined,
            .ev = undefined,
            .ev_join = undefined,
            .attn_r = try gpa.alloc(exl3.PArgs, n),
            .moe_r = try gpa.alloc(exl3.PArgs, n),
            .next_r = try gpa.alloc([4]exl3.PArgs, n),
        };
        self.side = try cuda.Stream.init(d, true);
        self.ev = try cuda.Event.init(d, false);
        self.ev_join = try cuda.Event.init(d, false);
        const chunk = self.pf.chunk;
        var bufs: [16]cuda.DeviceBuffer = undefined;
        for (w.layers, 0..) |*l, li| {
            var k: usize = 0;
            for (l.wo_a[0..l.wo_a_count]) |x| {
                bufs[k] = x.words;
                k += 1;
            }
            bufs[k] = l.wo_b.words;
            k += 1;
            exl3.Prefetch.ranges(bufs[0..k], attn_mb << 20, &self.attn_r[li], chunk);
            // the FFN's mix and gate, then the shared expert's trellises (the last expert's gate, up, down)
            k = 0;
            bufs[k] = l.hc_ffn_fn.buf;
            bufs[k + 1] = l.gate_w.buf;
            k += 2;
            const shared = try sharedTrellises(&l.experts);
            for (shared) |b| {
                bufs[k] = b;
                k += 1;
            }
            exl3.Prefetch.ranges(bufs[0..k], moe_mb << 20, &self.moe_r[li], chunk);
            for (0..4) |v| {
                const with_engram = v / 2 == 1;
                const mb: usize = if (v % 2 == 1) stretch_end_mb else next_mb;
                const budget = mb << 20;
                k = 0;
                if (li + 1 < n) {
                    const x = &w.layers[li + 1];
                    if (with_engram) if (x.engram_wkv) |e| {
                        bufs[k] = e.words;
                        k += 1;
                    };
                    bufs[k] = x.hc_attn_fn.buf;
                    bufs[k + 1] = x.wq_a.words;
                    bufs[k + 2] = x.wkv.words;
                    k += 3;
                    if (x.ratio > 0) {
                        if (x.comp_wkv) |cw| {
                            bufs[k] = cw.words;
                            k += 1;
                        }
                        if (x.comp_wgate) |cg| {
                            bufs[k] = cg.words;
                            k += 1;
                        }
                    }
                    bufs[k] = x.wq_b.words;
                    k += 1;
                } else {
                    bufs[k] = w.head.words;
                    k += 1;
                }
                exl3.Prefetch.ranges(bufs[0..k], budget, &self.next_r[li][v], chunk);
            }
        }
        return self;
    }

    /// The shared expert's (the last expert's) gate, up and down trellises, as views of the layer's trellis buffer.
    fn sharedTrellises(ex: *const experts.Experts) ![3]cuda.DeviceBuffer {
        const E = ex.count;
        var out: [3]cuda.DeviceBuffer = undefined;
        const tables = [3]cuda.DeviceBuffer{ ex.gate_ptr, ex.up_ptr, ex.down_ptr };
        const k2t = [3]cuda.DeviceBuffer{ ex.gate_k2, ex.up_k2, ex.down_k2 };
        for (0..3) |j| {
            var p: [1]i64 = undefined;
            var k2: [1]i32 = undefined;
            try tables[j].download((E - 1) * 8, std.mem.sliceAsBytes(&p));
            try k2t[j].download((E - 1) * 4, std.mem.sliceAsBytes(&k2));
            // D x I (or I x D) weights at k2 / 2 bits each: D * I * k2 / 16 bytes
            out[j] = .{ .d = ex.trellises.d, .ptr = @intCast(p[0]), .len = ex.dims * ex.width * @as(usize, @intCast(k2[0])) / 16 };
        }
        return out;
    }

    pub fn deinit(self: *L2) void {
        self.gpa.free(self.attn_r);
        self.gpa.free(self.moe_r);
        self.gpa.free(self.next_r);
        self.pf.deinit();
        self.side.deinit();
        self.ev.deinit();
        self.ev_join.deinit();
    }

    fn fork(self: *L2, ranges: *const exl3.PArgs, stream: cuda.Stream) !void {
        if (!self.on or ranges.count == 0) return;
        try self.ev.record(stream);
        try self.side.wait(self.ev);
        var a = ranges.*;
        try self.pf.run(&a, self.side);
        self.live = true;
    }

    pub fn attn(self: *L2, li: usize, stream: cuda.Stream) !void {
        try self.fork(&self.attn_r[li], stream);
    }

    pub fn moe(self: *L2, li: usize, stream: cuda.Stream) !void {
        try self.fork(&self.moe_r[li], stream);
    }

    /// After layer li's experts, in a stretch ending before layer `end`.
    pub fn next(self: *L2, li: usize, end: usize, stream: cuda.Stream) !void {
        const stretch_end = li + 1 == end and end < self.next_r.len;
        try self.fork(&self.next_r[li][@as(usize, @intFromBool(self.engram)) * 2 + @intFromBool(stretch_end)], stream);
    }

    /// `stream` waits for every prefetch forked so far.
    pub fn join(self: *L2, stream: cuda.Stream) !void {
        if (!self.live) return;
        try self.ev_join.record(self.side);
        try stream.wait(self.ev_join);
        self.live = false;
    }
};

/// Layers [first, end) of a round: the embedding before layer 0 in the first, the head after the last layer in the last.
const Seg = struct { first: usize, end: usize };

/// The layer loop's host state at a stretch's end (the next stretch starts from it).
const SegState = struct {
    h: Ptr = 0, // the streams [R, 4, D] bf16 (the fused path posts into the other buffer: `spare`)
    spare: Ptr = 0,
    pre: Ptr = 0,
    pre_f: Ptr = 0,
    tap: usize = 0,
    cidx: Ptr = 0, // the picks the compressed layers read (the last index layer's)
    cand_made: bool = false,
};

/// One key's graphs: a stretch's executable (null until recorded), the gathers it replays, the state at its end.
const Graphs = struct {
    n: usize,
    execs: [max_segments]?cuda.graph.Exec = @splat(null),
    gathers: [max_segments]usize = @splat(0),
    after: [max_segments]SegState = @splat(.{}),
};

/// A gather that counts its calls (a capture's).
const Counted = struct {
    inner: fwd.Gather,
    n: usize = 0,

    fn call(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
        const self: *Counted = @ptrCast(@alignCast(ctx));
        self.n += 1;
        return self.inner.run(partial, words, out, stream);
    }
};

/// A round's forward over one pool: R <= 16 rows of several streams, the fork's `RoundDecoder`.
pub const RoundRunner = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    cfg: *const Config,
    w: *const weights.Weights,
    set: *const aot.Set,
    glue: *const attention.Glue,
    ek: *const experts.Kernels,
    hc: hcmod.Hc,
    route: routemod.Route,
    world: u32,
    /// The widest bucket a round may scan (the per-round buffers' size): the context's.
    bucket: usize,
    /// The least bucket a round scans (graph.py BUCKET_MIN; the pool's widest makes every round scan it whole).
    least: usize = bucket_min,
    /// The last round's bucket (`bucketOf` its windows).
    round_bucket: usize = 0,
    rope_c: attention.Rope,
    rope_w: attention.Rope,
    pool: *Pool,
    probe: ?fwd.Probe = null,
    engram: ?Engram = null,
    idx_proj_h: []cuda.DeviceBuffer, // per layer: fp16 [idx_heads, d] on index layers
    // the round's tables (int64), host and device, one layout (`Tab`): the rows' (ids, positions, ring and raw rows),
    // each compress ratio's (compressed base, target row, group position, first raw row, visible count) and its picks
    // at short buckets; filled by one copy a round
    tab_h: []i64,
    end_h: []i64, // the rows' extent ends (host only)
    base_h: []i64, // the rows' extent bases (host only)
    slot_h: []i64, // the rows' slots (host only)
    t_all: cuda.DeviceBuffer,
    ratios: [max_ratios]usize, // the compress ratios of the layers (0 not among them)
    n_ratios: usize,
    kmax: usize, // the widest selection (columns of a ratio's picks table)
    cidx_h: []i64,
    uploads: attention.Uploads, // the tables' copy a round (and the picks of the host selection)
    t_cidx: cuda.DeviceBuffer, // the device selection's picks [R, kk]
    t_zero: cuda.DeviceBuffer,
    rows_glue: cuda.Module, // prompt_glue.cu: tf_dsv41_copy_rows (the compressor inputs into their raw rows)
    copy_rows: cuda.Function,
    // graphs (TF_DS_GRAPHS, default on): the stretches of each (rows, bucket, Engram split), captured on first use or
    // by `warm`
    graphs: bool,
    execs: std.AutoHashMapUnmanaged(u64, *Graphs),
    captures: usize,
    /// Called before each graph launch (tests' gather supplies; the communicator needs none).
    launch_hook: ?GraphLaunch = null,
    /// Rounds at wider buckets run eager: each bucket's graphs hold their recorded all-gathers' resources (about
    /// 0.65 GiB a bucket for 16 row counts at 43 layers), so recording every bucket of a long context would not fit.
    graph_bucket_max: usize = std.math.maxInt(usize),
    // the layer loop's state between stretches (`segment`)
    st: SegState,
    // the scored indexer (buckets past idx_topk compressed entries)
    iq: cuda.DeviceBuffer, // bf16 [R, idx_heads, idx_dim]
    iqq: cuda.DeviceBuffer,
    wl: cuda.DeviceBuffer, // fp32 [R, idx_heads]
    wts_i: cuda.DeviceBuffer, // bf16 [R, idx_heads]
    score: cuda.DeviceBuffer, // fp32 [R, bucket]
    keys: cuda.DeviceBuffer, // int64 [R, bucket / ratio] for the fused selection (where it applies)
    smax: cuda.DeviceBuffer, // fp32 [R, cand cols]
    cand_mask: cuda.DeviceBuffer, // u8 [R, cand cols]: the candidate source's pool this round
    score_h: []f32, // the host selection's
    keys_h: []i64,
    cand_h: []u8,
    // activations
    h: cuda.DeviceBuffer,
    pre: cuda.DeviceBuffer,
    pre_a: cuda.DeviceBuffer,
    pre_f: cuda.DeviceBuffer,
    post: cuda.DeviceBuffer,
    comb: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer,
    xh: cuda.DeviceBuffer,
    z: cuda.DeviceBuffer,
    qa: cuda.DeviceBuffer,
    qr: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    ykv: cuda.DeviceBuffer,
    kv: cuda.DeviceBuffer,
    ckv: cuda.DeviceBuffer,
    cgate: cuda.DeviceBuffer,
    pooled: cuda.DeviceBuffer,
    lat: cuda.DeviceBuffer,
    lat2: cuda.DeviceBuffer,
    kraw: cuda.DeviceBuffer,
    k: cuda.DeviceBuffer,
    pm: cuda.DeviceBuffer,
    pl: cuda.DeviceBuffer,
    po: cuda.DeviceBuffer,
    o: cuda.DeviceBuffer,
    og: cuda.DeviceBuffer,
    ug: cuda.DeviceBuffer,
    u: cuda.DeviceBuffer,
    attn_out: cuda.DeviceBuffer,
    gathered: cuda.DeviceBuffer,
    parts: cuda.DeviceBuffer,
    pick: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    moe: cuda.DeviceBuffer,
    scratch: experts.Scratch,
    xc: cuda.DeviceBuffer,
    local: cuda.DeviceBuffer,
    glog: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer,
    taps: cuda.DeviceBuffer, // bf16 [R, ntap x d]
    ntap: usize,
    // the fused decode path (TF_DS_FUSED, default on): rounds.py `_layers_fused` + `_attention2` at the fork's default
    // switches
    fused: bool,
    gk: exl3.GroupKernels,
    lg: []LayerGroups,
    h2: cuda.DeviceBuffer, // the posted streams' other buffer
    xh_in: cuda.DeviceBuffer, // attn_in's rotated rows (rot_many)
    xh_q: cuda.DeviceBuffer, // wq_b's (and idx_wq_b's) rotated rows (q_kv_norm)
    xh_woa: cuda.DeviceBuffer, // wo_a's rotated rows (the attention merge)
    xh_wob: cuda.DeviceBuffer, // wo_b's rotated rows (wo_a's epilogue)
    zg: cuda.DeviceBuffer, // the grouped launches' split-K partials
    side: cuda.Stream, // the mHC Sinkhorn half (switch "hc_defer")
    ev_fork: cuda.Event,
    ev_join: cuda.Event,
    side_live: bool,
    l2: L2,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const Config, w: *const weights.Weights, set: *const aot.Set, glue: *const attention.Glue, ek: *const experts.Kernels, world: u32, bucket: usize, rope_c: attention.Rope, rope_w: attention.Rope, pool: *Pool) !RoundRunner {
        const D = cfg.dim;
        const R = max_rows;
        const heads = cfg.n_heads / world;
        const groups = cfg.o_groups / world;
        const hd = cfg.head_dim;
        var self: RoundRunner = undefined;
        self.gpa = gpa;
        self.d = d;
        self.cfg = cfg;
        self.w = w;
        self.set = set;
        self.glue = glue;
        self.ek = ek;
        self.world = world;
        self.bucket = bucket;
        self.least = bucket_min;
        self.round_bucket = 0;
        self.rope_c = rope_c;
        self.rope_w = rope_w;
        self.pool = pool;
        self.probe = null;
        self.engram = null;
        self.hc = hcmod.Hc.init(set, @intCast(D), @intCast(cfg.hc_iters), cfg.eps, cfg.hc_eps);
        self.route = routemod.Route.init(set, @intCast(D), @intCast(cfg.n_routed), @intCast(cfg.topk), cfg.route_scale, @intCast(cfg.n_routed));
        // the index layers' weights_proj as fp16 rows (model.py idx_proj_h), converted on the host
        self.idx_proj_h = try gpa.alloc(cuda.DeviceBuffer, w.layers.len);
        for (self.idx_proj_h, w.layers) |*b, *l| {
            if (l.idx_proj) |p| {
                const f32s = try gpa.alloc(f32, p.count);
                defer gpa.free(f32s);
                try p.buf.download(0, std.mem.sliceAsBytes(f32s));
                const f16s = try gpa.alloc(u16, p.count);
                defer gpa.free(f16s);
                for (f16s, f32s) |*o, v| o.* = @bitCast(@as(f16, @floatCast(v)));
                b.* = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(f16s));
            } else b.* = .{ .d = d, .ptr = 0, .len = 0 };
        }
        const kk = @min(cfg.idx_topk, bucket); // the widest selection (ratio 1)
        self.kmax = kk;
        self.n_ratios = 0;
        self.ratios = @splat(0);
        for (0..w.layers.len) |li| {
            const r = cfg.ratio(li);
            if (r == 0 or std.mem.indexOfScalar(usize, self.ratios[0..self.n_ratios], r) != null) continue;
            if (self.n_ratios == max_ratios) return error.Unsupported;
            self.ratios[self.n_ratios] = r;
            self.n_ratios += 1;
        }
        self.tab_h = try gpa.alloc(i64, Tab.words(kk));
        @memset(self.tab_h, 0);
        self.end_h = try gpa.alloc(i64, R);
        self.base_h = try gpa.alloc(i64, R);
        self.slot_h = try gpa.alloc(i64, R);
        self.cidx_h = try gpa.alloc(i64, R * kk);
        self.uploads = try attention.Uploads.init(d, attention.Uploads.room(1, Tab.words(kk) * 8) + attention.Uploads.room(w.layers.len, R * kk * 8));
        self.t_all = try cuda.DeviceBuffer.alloc(d, Tab.words(kk) * 8);
        try self.t_all.fill8(0, null);
        self.t_cidx = try cuda.DeviceBuffer.alloc(d, R * kk * 8);
        self.t_zero = try cuda.DeviceBuffer.alloc(d, R * 8);
        try self.t_zero.fill8(0, null);
        if (!cuda.kernels.available) return error.Unsupported;
        self.rows_glue = try cuda.Module.load(d, cuda.kernels.dsv41_prompt_glue);
        self.copy_rows = try self.rows_glue.function("tf_dsv41_copy_rows");
        self.graphs = graphsFromEnv();
        self.execs = .empty;
        self.captures = 0;
        self.launch_hook = null;
        self.graph_bucket_max = std.math.maxInt(usize);
        self.st = .{};
        self.iq = try cuda.DeviceBuffer.alloc(d, R * cfg.idx_heads * cfg.idx_dim * 2);
        self.iqq = try cuda.DeviceBuffer.alloc(d, R * cfg.idx_heads * cfg.idx_dim * 2);
        self.wl = try cuda.DeviceBuffer.alloc(d, R * cfg.idx_heads * 4);
        self.wts_i = try cuda.DeviceBuffer.alloc(d, R * cfg.idx_heads * 2);
        // the widest scan is the ratio-1 layers' (bucket entries); the smallest is the fused selection's candidate
        const scored = bucket > cfg.idx_topk;
        self.score = try cuda.DeviceBuffer.alloc(d, if (scored) R * bucket * 4 else 16);
        // the fused selection's keys: scans of up to 4096 entries (fusedTopk's widest), at any round's bucket
        self.keys = try cuda.DeviceBuffer.alloc(d, if (scored) R * @as(usize, @min(bucket, 4096)) * 8 else 16);
        const ccols = (bucket + @max(1, cfg.cand_block) - 1) / @max(1, cfg.cand_block);
        self.smax = try cuda.DeviceBuffer.alloc(d, if (scored) R * ccols * 4 else 16);
        self.cand_mask = try cuda.DeviceBuffer.alloc(d, if (scored) R * ccols else 16);
        self.score_h = try gpa.alloc(f32, if (scored) R * bucket else 1);
        self.keys_h = try gpa.alloc(i64, if (scored) bucket else 1);
        self.cand_h = try gpa.alloc(u8, if (scored) R * ccols else 1);
        self.h = try cuda.DeviceBuffer.alloc(d, R * 4 * D * 2);
        self.pre = try cuda.DeviceBuffer.alloc(d, R * 16);
        self.pre_a = try cuda.DeviceBuffer.alloc(d, R * 16);
        self.pre_f = try cuda.DeviceBuffer.alloc(d, R * 16);
        self.post = try cuda.DeviceBuffer.alloc(d, R * 16);
        self.comb = try cuda.DeviceBuffer.alloc(d, R * 64);
        self.part = try cuda.DeviceBuffer.alloc(d, hcmod.Hc.partWords(R) * 4);
        self.x = try cuda.DeviceBuffer.alloc(d, R * D * 2);
        self.xh = try cuda.DeviceBuffer.alloc(d, R * D * 2);
        var zw: usize = w.head.scratchWords(R);
        for (w.layers) |*l| {
            zw = @max(zw, l.wq_a.scratchWords(R));
            zw = @max(zw, l.wq_b.scratchWords(R));
            zw = @max(zw, l.wkv.scratchWords(R));
            zw = @max(zw, l.wo_b.scratchWords(R));
            for (l.wo_a[0..l.wo_a_count]) |x| zw = @max(zw, x.scratchWords(R));
            inline for (.{ "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk" }) |f| if (@field(l, f)) |x| {
                zw = @max(zw, x.scratchWords(R));
            };
        }
        self.z = try cuda.DeviceBuffer.alloc(d, @max(16, zw * 4));
        self.qa = try cuda.DeviceBuffer.alloc(d, R * cfg.q_rank * 2);
        self.qr = try cuda.DeviceBuffer.alloc(d, R * cfg.q_rank * 2);
        self.q = try cuda.DeviceBuffer.alloc(d, R * heads * hd * 2);
        self.ykv = try cuda.DeviceBuffer.alloc(d, R * hd * 2);
        self.kv = try cuda.DeviceBuffer.alloc(d, R * hd * 2);
        self.ckv = try cuda.DeviceBuffer.alloc(d, R * hd * 4);
        self.cgate = try cuda.DeviceBuffer.alloc(d, R * hd * 4);
        self.pooled = try cuda.DeviceBuffer.alloc(d, R * hd * 2);
        self.lat = try cuda.DeviceBuffer.alloc(d, R * hd * 2);
        self.lat2 = try cuda.DeviceBuffer.alloc(d, R * hd * 2);
        self.kraw = try cuda.DeviceBuffer.alloc(d, R * cfg.idx_dim * 2);
        self.k = try cuda.DeviceBuffer.alloc(d, R * cfg.idx_dim * 2);
        const gh = heads / 16;
        self.pm = try cuda.DeviceBuffer.alloc(d, R * gh * 8 * 16 * 4);
        self.pl = try cuda.DeviceBuffer.alloc(d, R * gh * 8 * 16 * 4);
        self.po = try cuda.DeviceBuffer.alloc(d, R * gh * 8 * 16 * hd * 4);
        self.o = try cuda.DeviceBuffer.alloc(d, R * heads * hd * 2);
        self.og = try cuda.DeviceBuffer.alloc(d, R * (heads * hd / groups) * 2);
        self.ug = try cuda.DeviceBuffer.alloc(d, R * cfg.o_rank * 2);
        self.u = try cuda.DeviceBuffer.alloc(d, R * groups * cfg.o_rank * 2);
        self.attn_out = try cuda.DeviceBuffer.alloc(d, R * D * 4);
        self.gathered = try cuda.DeviceBuffer.alloc(d, world * R * D * 4);
        self.parts = try cuda.DeviceBuffer.alloc(d, self.route.partsWords(R) * 4);
        self.pick = try cuda.DeviceBuffer.alloc(d, R * (cfg.topk + 1) * 4);
        self.wts = try cuda.DeviceBuffer.alloc(d, R * (cfg.topk + 1) * 4);
        self.moe = try cuda.DeviceBuffer.alloc(d, R * D * 4);
        self.scratch = try experts.Scratch.init(d, &w.layers[0].experts, R, cfg.topk + 1);
        self.xc = try cuda.DeviceBuffer.alloc(d, R * D * 2);
        const vl = cfg.vocab / world;
        self.local = try cuda.DeviceBuffer.alloc(d, R * vl * 4);
        self.glog = try cuda.DeviceBuffer.alloc(d, world * R * vl * 4);
        self.logits = try cuda.DeviceBuffer.alloc(d, R * cfg.vocab * 4);
        var ntap: usize = 0;
        for (cfg.dspark_taps.slice()) |t| if (t >= 0 and @as(usize, @intCast(t)) < w.layers.len) {
            ntap += 1;
        };
        self.ntap = ntap;
        self.taps = try cuda.DeviceBuffer.alloc(d, R * @max(ntap, 1) * D * 2);
        self.fused = fusedFromEnv();
        self.side_live = false;
        self.gk = try exl3.GroupKernels.load(d);
        self.lg = try gpa.alloc(LayerGroups, w.layers.len);
        var in_b: usize = 16;
        var q_b: usize = 16;
        var woa_b: usize = 16;
        var wob_b: usize = 16;
        var zgw: usize = 4;
        for (self.lg, w.layers, 0..) |*g, *l, li| {
            g.* = try LayerGroups.init(&self.gk, cfg, l, li, d);
            in_b = @max(in_b, g.in.bufferBytes(R));
            q_b = @max(q_b, (g.q2 orelse g.q1).bufferBytes(R));
            woa_b = @max(woa_b, g.wo_a.bufferBytes(R));
            wob_b = @max(wob_b, g.wo_b.bufferBytes(R));
            inline for (.{ "in", "q1", "wo_a", "wo_b" }) |f| zgw = @max(zgw, @field(g, f).scratchWords(R));
            if (g.q2) |q2| zgw = @max(zgw, q2.scratchWords(R));
        }
        self.h2 = try cuda.DeviceBuffer.alloc(d, R * 4 * D * 2);
        self.xh_in = try cuda.DeviceBuffer.alloc(d, in_b);
        self.xh_q = try cuda.DeviceBuffer.alloc(d, q_b);
        self.xh_woa = try cuda.DeviceBuffer.alloc(d, woa_b);
        self.xh_wob = try cuda.DeviceBuffer.alloc(d, wob_b);
        self.zg = try cuda.DeviceBuffer.alloc(d, zgw * 4);
        self.side = try cuda.Stream.init(d, true);
        self.ev_fork = try cuda.Event.init(d, false);
        self.ev_join = try cuda.Event.init(d, false);
        self.l2 = try L2.init(gpa, d, cfg, w);
        return self;
    }

    pub fn deinit(self: *RoundRunner) void {
        for (self.idx_proj_h) |*b| if (b.len > 0) b.free();
        self.gpa.free(self.idx_proj_h);
        self.dropGraphs();
        self.execs.deinit(self.gpa);
        self.rows_glue.unload();
        self.gpa.free(self.tab_h);
        self.gpa.free(self.end_h);
        self.gpa.free(self.base_h);
        self.gpa.free(self.slot_h);
        self.gpa.free(self.cidx_h);
        self.gpa.free(self.score_h);
        self.gpa.free(self.keys_h);
        self.gpa.free(self.cand_h);
        self.uploads.deinit();
        inline for (.{ "t_all", "t_cidx", "t_zero", "iq", "iqq", "wl", "wts_i", "score", "keys", "smax", "cand_mask", "h", "pre", "pre_a", "pre_f", "post", "comb", "part", "x", "xh", "z", "qa", "qr", "q", "ykv", "kv", "ckv", "cgate", "pooled", "lat", "lat2", "kraw", "k", "pm", "pl", "po", "o", "og", "ug", "u", "attn_out", "gathered", "parts", "pick", "wts", "moe", "xc", "local", "glog", "logits", "taps" }) |f| @field(self, f).free();
        self.scratch.deinit();
        for (self.lg) |*g| g.deinit();
        self.gpa.free(self.lg);
        inline for (.{ "h2", "xh_in", "xh_q", "xh_woa", "xh_wob", "zg" }) |f| @field(self, f).free();
        self.gk.deinit();
        self.side.deinit();
        self.ev_fork.deinit();
        self.ev_join.deinit();
        self.l2.deinit();
    }

    fn probeOut(self: *const RoundRunner, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) !void {
        if (self.probe) |p| try p.call(p.ctx, name, ptr, bytes, stream);
    }

    /// The bucket a round of `windows` scans: `bucketFor` its deepest row (position + 1) over the pool, the
    /// runner's `least` at least; error.BadShape past the runner's widest (its buffers).
    pub fn bucketOf(self: *const RoundRunner, windows: []const Window) error{BadShape}!usize {
        var deepest: usize = 1;
        for (windows) |wd| deepest = @max(deepest, wd.position + wd.ids.len);
        const b = bucketFor(deepest, self.pool.cap, @min(self.least, self.bucket));
        if (b < deepest or b > self.bucket) return error.BadShape;
        return b;
    }

    /// Every window's rows in one forward: logits [R, V] f32 (`self.logits`, row order = the windows'), the taps
    /// in `self.taps` [R, ntap x d]; the pool holds the rows' keys after. With graphs, the stretches of the round's
    /// key are replayed (recorded first when missing); without, launched one by one; the same launches either way.
    pub fn forward(self: *RoundRunner, windows: []const Window, gather: fwd.Gather, stream: cuda.Stream) !Ptr {
        const D = self.cfg.dim;
        var R: usize = 0;
        for (windows) |wd| R += wd.ids.len;
        if (R < 1 or R > max_rows) return error.TooManyRows;
        // every row inside its extent, the extent inside the window, its positions inside the RoPE tables
        const rope_rows = @min(self.rope_c.rows, self.rope_w.rows);
        for (windows) |wd| {
            if (wd.ids.len == 0 or wd.slot >= self.pool.slots or wd.base + wd.size > self.pool.cap) return error.BadShape;
            if (wd.position + wd.ids.len > wd.size or wd.position + wd.ids.len > rope_rows) return error.BadShape;
            if (self.engram != null and wd.host.len < wd.position + wd.ids.len) return error.BadShape;
        }
        self.round_bucket = try self.bucketOf(windows);
        self.l2.engram = self.engram != null;
        try self.setTables(windows, R, stream);
        var segs: [max_segments]Seg = undefined;
        const ns = try self.segments(&segs);
        var spans: [max_rows]EngramSpan = undefined;
        if (self.engram != null) {
            for (windows, 0..) |wd, i| spans[i] = .{ .ids = wd.host, .start = wd.position, .n = wd.ids.len };
        }
        const entry: ?*Graphs = if (self.graphed() and self.round_bucket <= self.graph_bucket_max) try self.graphsFor(R, self.round_bucket, ns) else null;
        var nb: [48]u8 = undefined;
        for (segs[0..ns], 0..) |sg, k| {
            if (k > 0) {
                // the stretch starts at an Engram layer: its rows read and gated here, between the graphs
                const e = self.engram.?;
                try e.call(e.ctx, sg.first, self.st.h, spans[0..windows.len], gather, stream);
                try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_engram", .{sg.first}), self.st.h, R * 4 * D * 2, stream);
            }
            if (entry) |g| {
                try self.record(g, segs[0..ns], k, R, gather, stream);
                if (self.launch_hook) |hk| try hk.call(hk.ctx, g.gathers[k], stream);
                try g.execs[k].?.launchOn(stream);
                self.st = g.after[k];
            } else try self.segment(segs[0..ns], k, R, gather, stream);
        }
        return self.logits.ptr;
    }

    /// Whether rounds replay graphs now: on, no probe (its callbacks run per launch), the device selection.
    fn graphed(self: *const RoundRunner) bool {
        return self.graphs and self.probe == null and !self.glue.hostSelect();
    }

    /// Records the graphs of rounds of each row count in `rows` (1 .. max_rows) at each bucket of `buckets` the runner
    /// scans, ahead of the rounds that replay them; nothing runs. Every rank calls it with the same arguments, so
    /// the ranks record the same gathers in the same order (none is issued).
    pub fn warm(self: *RoundRunner, rows: []const usize, buckets: []const usize, gather: fwd.Gather, stream: cuda.Stream) !void {
        if (!self.graphed()) return;
        self.l2.engram = self.engram != null;
        var segs: [max_segments]Seg = undefined;
        const ns = try self.segments(&segs);
        for (buckets) |b| {
            if (b > self.bucket or b < @min(self.least, self.bucket) or b > self.graph_bucket_max) continue;
            for (rows) |R| {
                if (R < 1 or R > max_rows) return error.TooManyRows;
                self.round_bucket = b;
                const g = try self.graphsFor(R, b, ns);
                for (0..ns) |k| {
                    try self.record(g, segs[0..ns], k, R, gather, stream);
                    self.st = g.after[k];
                }
            }
        }
        try stream.synchronize();
    }

    /// The graphs this runner holds (each stretch of each key recorded).
    pub fn graphCount(self: *const RoundRunner) usize {
        var n: usize = 0;
        var it = self.execs.valueIterator();
        while (it.next()) |g| for (g.*.execs[0..g.*.n]) |e| {
            if (e != null) n += 1;
        };
        return n;
    }

    /// Drops every recorded graph (later rounds record their own again).
    pub fn dropGraphs(self: *RoundRunner) void {
        var it = self.execs.valueIterator();
        while (it.next()) |g| {
            for (&g.*.execs) |*e| if (e.*) |*x| {
                x.deinit();
                e.* = null;
            };
            self.gpa.destroy(g.*);
        }
        self.execs.clearRetainingCapacity();
    }

    fn graphsFor(self: *RoundRunner, R: usize, bucket: usize, ns: usize) !*Graphs {
        const key: u64 = @as(u64, R) | @as(u64, ns) << 5 | @as(u64, bucket) << 9;
        const got = try self.execs.getOrPut(self.gpa, key);
        if (!got.found_existing) {
            got.value_ptr.* = self.gpa.create(Graphs) catch |err| {
                _ = self.execs.remove(key);
                return err;
            };
            got.value_ptr.*.* = .{ .n = ns };
        }
        return got.value_ptr.*;
    }

    /// Stretch `k` of `g` recorded (from the state the stretch before it left) unless it already is.
    fn record(self: *RoundRunner, g: *Graphs, segs: []const Seg, k: usize, R: usize, gather: fwd.Gather, stream: cuda.Stream) !void {
        if (g.execs[k] != null) return;
        if (k > 0) self.st = g.after[k - 1];
        var counted: Counted = .{ .inner = gather };
        const cg: fwd.Gather = .{ .ctx = &counted, .call = Counted.call };
        try cuda.graph.beginCapture(stream, .thread_local);
        self.segment(segs, k, R, cg, stream) catch |err| {
            if (cuda.graph.endCapture(stream)) |gr| {
                var x = gr;
                x.deinit();
            } else |_| {}
            return err;
        };
        var gr = try cuda.graph.endCapture(stream);
        defer gr.deinit();
        var exec = try gr.instantiate();
        errdefer exec.deinit();
        try exec.upload(stream);
        g.execs[k] = exec;
        g.gathers[k] = counted.n;
        g.after[k] = self.st;
        self.captures += 1;
    }

    /// The round's stretches: one, or (an Engram hook set) cut before each Engram layer (layer 0's: after the
    /// embedding alone). Returns their count.
    fn segments(self: *const RoundRunner, out: *[max_segments]Seg) error{Unsupported}!usize {
        var n: usize = 0;
        var first: usize = 0;
        if (self.engram != null) for (self.w.layers, 0..) |*l, li| {
            if (l.engram_wkv == null) continue;
            if (n + 2 > max_segments) return error.Unsupported;
            out[n] = .{ .first = first, .end = li };
            n += 1;
            first = li;
        };
        out[n] = .{ .first = first, .end = self.w.layers.len };
        return n + 1;
    }

    /// The round's tables (rounds.py RoundDecoder.set and `_ix`) on the host, then to the device in one copy on
    /// `stream`, ahead of the round's launches (which read them there).
    fn setTables(self: *RoundRunner, windows: []const Window, R: usize, stream: cuda.Stream) !void {
        const t = self.tab_h;
        const ids = t[Tab.row(Tab.ids)..][0..max_rows];
        const pos = t[Tab.row(Tab.pos)..][0..max_rows];
        const wslot = t[Tab.row(Tab.wslot)..][0..max_rows];
        const wbase = t[Tab.row(Tab.wbase)..][0..max_rows];
        const rslot = t[Tab.row(Tab.rslot)..][0..max_rows];
        const RAW: i64 = raw_rows;
        var r: usize = 0;
        for (windows) |wd| {
            for (wd.ids, 0..) |id, i| {
                ids[r] = id;
                pos[r] = @intCast(wd.position + i);
                self.slot_h[r] = @intCast(wd.slot);
                self.base_h[r] = @intCast(wd.base);
                self.end_h[r] = @intCast(wd.base + wd.size);
                r += 1;
            }
        }
        const RS: i64 = @intCast(self.pool.ring);
        for (0..R) |i| {
            wbase[i] = self.slot_h[i] * RS;
            wslot[i] = wbase[i] + @mod(pos[i], RS);
            rslot[i] = self.slot_h[i] * RAW + @mod(pos[i], RAW);
        }
        var used: usize = Tab.ratio(self.n_ratios, 0);
        for (self.ratios[0..self.n_ratios], 0..) |ratio, j| {
            const rt: i64 = @intCast(ratio);
            const cbase = t[Tab.ratio(j, Tab.cbase)..][0..max_rows];
            const ctarget = t[Tab.ratio(j, Tab.ctarget)..][0..max_rows];
            const gpos = t[Tab.ratio(j, Tab.gpos)..][0..max_rows];
            const gfirst = t[Tab.ratio(j, Tab.gfirst)..][0..max_rows];
            const vis = t[Tab.ratio(j, Tab.vis)..][0..max_rows];
            for (0..R) |i| {
                cbase[i] = @divFloor(self.base_h[i], rt);
                const scratch = @divFloor(self.end_h[i], rt) - 1 - cbase[i];
                const groups_i = @divFloor(pos[i], rt);
                const target: i64 = if (@mod(pos[i] + 1, rt) == 0) groups_i else scratch;
                ctarget[i] = cbase[i] + (if (ratio == 1) pos[i] else target);
                gpos[i] = (if (ratio == 1) pos[i] else groups_i) * rt;
                gfirst[i] = self.slot_h[i] * RAW + @mod(gpos[i], RAW);
                vis[i] = @divFloor(pos[i] + 1, rt);
            }
            // short buckets: every scanned entry is picked (the top-k of the scores is every index below vis)
            const nb_ = self.round_bucket / ratio;
            if (nb_ <= self.cfg.idx_topk) {
                const kk = @min(self.cfg.idx_topk, nb_);
                const picks = t[Tab.picks(j, self.kmax)..][0 .. R * kk];
                for (0..R) |i| for (0..kk) |cc| {
                    picks[i * kk + cc] = if (@as(i64, @intCast(cc)) < vis[i]) @intCast(cc) else -1;
                };
                used = Tab.picks(j, self.kmax) + R * kk;
            }
        }
        try self.uploads.begin();
        try self.uploads.put(self.t_all, 0, std.mem.sliceAsBytes(t[0..used]), stream);
    }

    /// The device address of word `word` of the tables.
    fn tab(self: *const RoundRunner, word: usize) Ptr {
        return self.t_all.ptr + word * 8;
    }

    fn ratioSlot(self: *const RoundRunner, ratio: usize) usize {
        return std.mem.indexOfScalar(usize, self.ratios[0..self.n_ratios], ratio).?;
    }

    /// Stretch `k` of the round's launches (`segs[k]`): the embedding first in stretch 0, the head last in the last;
    /// the layer loop's state in `self.st` between stretches. Reads the host only for R and the bucket (the host
    /// selection aside), so its launches can be recorded.
    fn segment(self: *RoundRunner, segs: []const Seg, k: usize, R: usize, gather: fwd.Gather, stream: cuda.Stream) !void {
        const c = self.cfg;
        const D = c.dim;
        const w = self.w;
        const sg = segs[k];
        if (k == 0) {
            try self.hc.embedInit(stream, w.embed.ptr(), self.tab(Tab.row(Tab.ids)), self.h.ptr, self.pre.ptr, @intCast(R));
            self.st = .{ .h = self.h.ptr, .spare = self.h2.ptr, .pre = self.pre.ptr, .pre_f = self.pre_f.ptr, .tap = 0, .cidx = self.t_cidx.ptr, .cand_made = false };
        }
        var nb: [48]u8 = undefined;
        if (self.fused) try self.fusedLayers(sg, R, gather, stream) else for (w.layers[sg.first..sg.end], sg.first..) |*l, li| {
            if (c.dspark_taps.has(@intCast(li))) {
                try self.hc.tap(stream, self.h.ptr, self.taps.ptr + self.st.tap * D * 2, @intCast(@max(self.ntap, 1) * D), @intCast(R));
                self.st.tap += 1;
            }
            try self.hc.pre(stream, self.h.ptr, l.hc_attn_fn.ptr(), l.hc_attn_scale.ptr(), l.hc_attn_base.ptr(), self.st.pre, l.attn_norm.ptr(), self.x.ptr, self.pre_a.ptr, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_attn", .{li}), self.x.ptr, R * D * 2, stream);
            const pa = try self.roundAttention(li, R, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_attn.r0", .{li}), pa, R * D * 4, stream);
            try gather.run(pa, R * D, self.gathered.ptr, stream);
            try self.hc.post(stream, self.gathered.ptr, self.world, self.h.ptr, self.post.ptr, self.comb.ptr, self.h.ptr, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_attn", .{li}), self.h.ptr, R * 4 * D * 2, stream);
            try self.hc.pre(stream, self.h.ptr, l.hc_ffn_fn.ptr(), l.hc_ffn_scale.ptr(), l.hc_ffn_base.ptr(), self.pre_a.ptr, l.ffn_norm.ptr(), self.x.ptr, self.st.pre_f, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_ffn", .{li}), self.x.ptr, R * D * 2, stream);
            try self.route.gate(stream, self.x.ptr, l.gate_w.ptr(), self.parts.ptr, @intCast(R));
            try self.route.pick(stream, self.parts.ptr, l.gate_b.ptr(), self.pick.ptr, self.wts.ptr, @intCast(R));
            try experts.routed(self.ek, &l.experts, &self.scratch, self.x.ptr, self.pick.ptr, self.wts.ptr, R, c.swiglu_limit, self.moe.ptr, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_moe.r0", .{li}), self.moe.ptr, R * D * 4, stream);
            try gather.run(self.moe.ptr, R * D, self.gathered.ptr, stream);
            try self.hc.post(stream, self.gathered.ptr, self.world, self.h.ptr, self.post.ptr, self.comb.ptr, self.h.ptr, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{li}), self.h.ptr, R * 4 * D * 2, stream);
            const t = self.st.pre;
            self.st.pre = self.st.pre_f;
            self.st.pre_f = t;
        }
        if (sg.end < w.layers.len) return;
        // -- the head: every rank's vocabulary columns, rank order within a row ----------------------------------
        try self.hc.collapseNorm(stream, self.st.h, self.st.pre, w.norm.ptr(), self.xc.ptr, @intCast(R));
        try self.probeOut("head.xc", self.xc.ptr, R * D * 2, stream);
        try w.head.run(self.xc.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.local.ptr, .f32, stream);
        const vl = c.vocab / self.world;
        try self.probeOut("head.local.r0", self.local.ptr, R * vl * 4, stream);
        try gather.run(self.local.ptr, R * vl, self.glog.ptr, stream);
        for (0..self.world) |rk| {
            // [world, R, vl] -> [R, world x vl]: rank rk's columns of every row (f32 moved as pairs of 16-bit words)
            try self.copyCols(self.glog.ptr + rk * R * vl * 4, vl * 2, self.logits.ptr + rk * vl * 4, c.vocab * 2, R, vl * 2, stream);
        }
    }

    /// rounds.py `RoundDecoder._attention` (unfused) for layer `li` over the round's rows: this rank's fp32 partial.
    fn roundAttention(self: *RoundRunner, li: usize, R: usize, stream: cuda.Stream) !Ptr {
        const c = self.cfg;
        const l = &self.w.layers[li];
        const hd = c.head_dim;
        const heads = c.n_heads / self.world;
        const groups = c.o_groups / self.world;
        const ratio = c.ratio(li);
        const rope = if (ratio > 0) self.rope_c else self.rope_w;
        const t_pos = self.tab(Tab.row(Tab.pos));
        const t_wbase = self.tab(Tab.row(Tab.wbase));
        const ring = self.pool.rings[li];
        // -- attn_in: wq_a, wkv (and the compressor's) on x ------------------------------------------------------
        try l.wq_a.run(self.x.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.qa.ptr, .bf16, stream);
        try l.wkv.run(self.x.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.ykv.ptr, .bf16, stream);
        const kv_source = ratio > 0 and c.isKvSource(li);
        if (kv_source) {
            try l.comp_wkv.?.run(self.x.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.ckv.ptr, if (ratio == 1) .bf16 else .f32, stream);
            if (ratio > 1) try l.comp_wgate.?.run(self.x.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.cgate.ptr, .f32, stream);
        }
        // -- q: RMSNorm, wq_b, RoPE on this rank's heads ----------------------------------------------------------
        try self.rmsnorm(self.qa.ptr, l.q_norm.ptr(), self.qr.ptr, R, c.q_rank, stream);
        try l.wq_b.run(self.qr.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.q.ptr, .bf16, stream);
        try self.ropeHeads(self.q.ptr, rope, t_pos, R, heads, hd, false, stream);
        // -- the window KV into the slot's ring rows ------------------------------------------------------------
        try self.set.run(stream, "_kv_norm_rope", .{ @intCast(R), 1, 1 }, &.{
            aot.ptr("Y", "*bf16", self.ykv.ptr),  aot.ptr("W", "*bf16", l.kv_norm.ptr()), aot.ptr("COS", "*fp32", rope.cos.ptr),
            aot.ptr("SIN", "*fp32", rope.sin.ptr), aot.ptr("POS", "*i64", t_pos),            aot.ptr("OUT", "*bf16", self.kv.ptr),
            aot.ptr("RING", "*bf16", ring.ptr),   aot.ptr("SLOT_OF", "*i64", self.tab(Tab.row(Tab.wslot))), aot.int("ring_size", @intCast(self.pool.slots * self.pool.ring)),
            aot.float("eps", c.eps),
        }, &.{ aot.ci("QUANT", 1), aot.ci("D", @intCast(hd)), aot.ci("RD", @intCast(c.rope_dim)), aot.ci("PDL", 1) });
        var comp_codes: Ptr = ring.ptr;
        var comp_scales: Ptr = ring.ptr;
        var t_cbase: Ptr = t_wbase;
        var kk: usize = 0;
        if (ratio > 0) {
            const j = self.ratioSlot(ratio);
            t_cbase = self.tab(Tab.ratio(j, Tab.cbase));
            const t_ctarget = self.tab(Tab.ratio(j, Tab.ctarget));
            const t_gpos = self.tab(Tab.ratio(j, Tab.gpos));
            const src = &self.pool.planes[self.pool.source[li]].?;
            if (kv_source) try self.compressor(li, R, j, src, rope, t_ctarget, t_gpos, stream);
            // -- the indexer: every visible compressed entry (short buckets: no scores, the picks table) ---------
            const nb_ = self.round_bucket / ratio;
            kk = @min(c.idx_topk, nb_);
            if (c.isIndexSource(li)) {
                if (nb_ > c.idx_topk) {
                    try self.roundIndex(li, R, nb_, kk, j, false, stream);
                    self.st.cidx = self.t_cidx.ptr;
                } else self.st.cidx = self.tab(Tab.picks(j, self.kmax));
            }
            comp_codes = src.comp_codes.ptr;
            comp_scales = src.comp_scales.ptr;
        }
        // -- sparse attention over the slot's ring and the extent's picks, with per-row bases ------------------
        {
            const gh = heads / 16;
            const has = ratio > 0;
            const picks: usize = if (has) (kk + 31) / 32 else 0;
            const nblk: usize = c.window / 32 + picks;
            const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
            try self.set.run(stream, "_sparse_attn_part", .{ @intCast(R), @intCast(gh), 8 }, &.{
                aot.ptr("Q", "*bf16", self.q.ptr),       aot.ptr("WSRC", "*bf16", ring.ptr),
                aot.ptr("WLO", "*i64", self.t_zero.ptr), if (has) aot.ptr("COMP", "*u8", comp_codes) else aot.ptr("COMP", "*bf16", ring.ptr),
                if (has) aot.ptr("CSC", "*u8", comp_scales) else aot.ptr("CSC", "*bf16", ring.ptr), aot.ptr("IDX", "*i64", if (has) self.st.cidx else t_pos),
                aot.ptr("POS", "*i64", t_pos),           aot.ptr("PM", "*fp32", self.pm.ptr),
                aot.ptr("PL", "*fp32", self.pl.ptr),     aot.ptr("PO", "*fp32", self.po.ptr),
                aot.ptr("SINK", "*fp32", l.sink.ptr()),  aot.ptr("OUT", "*bf16", self.o.ptr),
                aot.float("scale", scale),               aot.int("ring_size", @intCast(self.pool.ring)),
                aot.int("n_idx", @intCast(kk)),          aot.ptr("RBASE", "*i64", t_wbase),
                aot.ptr("CBASE", "*i64", t_cbase),
            }, &.{
                aot.ci("H", @intCast(heads)), aot.ci("HD", @intCast(hd)),       aot.ci("HB", 16),    aot.ci("WIN", @intCast(c.window)),
                aot.ci("BN", 32),             aot.ci("RING", 1),                 aot.ci("HAS_COMP", @intFromBool(has)), aot.ci("PACKED", @intFromBool(has)),
                aot.ci("SPLITS", 8),          aot.ci("NBLK", @intCast(nblk)),    aot.ci("FINAL", 0),  aot.ci("HAS_BASE", 1),
                aot.ci("PDL", 1),             aot.ci("WPF", 1),
            });
            try self.set.run(stream, "_sparse_attn_merge", .{ @intCast(R), @intCast(gh), 1 }, &.{
                aot.ptr("PM", "*fp32", self.pm.ptr), aot.ptr("PL", "*fp32", self.pl.ptr), aot.ptr("PO", "*fp32", self.po.ptr),
                aot.ptr("SINK", "*fp32", l.sink.ptr()), aot.ptr("OUT", "*bf16", self.o.ptr),
            }, &.{ aot.ci("H", @intCast(heads)), aot.ci("HD", @intCast(hd)), aot.ci("HB", 16), aot.ci("SPLITS", 8), aot.ci("PDL", 1) });
        }
        try self.ropeHeads(self.o.ptr, rope, t_pos, R, heads, hd, true, stream);
        const o_cols = heads * hd;
        const g_cols = o_cols / groups;
        for (0..groups) |g| {
            try self.copyCols(self.o.ptr + g * g_cols * 2, o_cols, self.og.ptr, g_cols, R, g_cols, stream);
            try l.wo_a[g].run(self.og.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.ug.ptr, .bf16, stream);
            try self.copyCols(self.ug.ptr, c.o_rank, self.u.ptr + g * c.o_rank * 2, groups * c.o_rank, R, c.o_rank, stream);
        }
        try l.wo_b.run(self.u.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.attn_out.ptr, .f32, stream);
        return self.attn_out.ptr;
    }

    /// A kv-source layer's compressor for the round's rows (rounds.py `_kv_idx`): the compressed latents (ratio 1:
    /// RMSNorm of the compressor's kv; ratio 2: the rows' inputs into their raw rows, each group's softmax pool) and,
    /// on an index layer, the index keys, FP4 into the planes at each row's target row.
    fn compressor(self: *RoundRunner, li: usize, R: usize, j: usize, src: *const PoolPlanes, rope: attention.Rope, t_ctarget: Ptr, t_gpos: Ptr, stream: cuda.Stream) !void {
        const c = self.cfg;
        const l = &self.w.layers[li];
        const hd = c.head_dim;
        const ratio = c.ratio(li);
        // -- the kv source: compressed latents (and index keys) of the rows' groups ----------------------
        if (ratio == 1) {
            try self.rmsnorm(self.ckv.ptr, l.comp_norm.?.ptr(), self.lat.ptr, R, hd, stream);
        } else {
            if (ratio != 2 or raw_rows % 2 != 0) return error.Unsupported;
            // the rows' compressor inputs into their slots' raw rows by position (rows of different streams
            // write different slots, a stream's rows different positions), then each row's group's softmax
            // pool from its slot's raw rows: comp_pool2 over the whole plane (raw = slots x RAW), each row's
            // first raw row its slot's (a group starts at an even position, so its pair stays in the slot)
            try self.copyRows(self.ckv.ptr, src.raw_kv.ptr, self.tab(Tab.row(Tab.rslot)), R, hd, stream);
            try self.copyRows(self.cgate.ptr, src.raw_score.ptr, self.tab(Tab.row(Tab.rslot)), R, hd, stream);
            var a: cuda.Args = .{};
            a.add(src.raw_kv.ptr);
            a.add(src.raw_score.ptr);
            a.add(self.tab(Tab.ratio(j, Tab.gfirst)));
            a.add(@as(c_int, @intCast(self.pool.slots * raw_rows)));
            a.add(@as(c_int, @intCast(hd)));
            a.add(self.pooled.ptr);
            try cuda.launch.launch(self.glue.comp_pool2, .{ .grid = .{ .x = @intCast((hd + 255) / 256), .y = @intCast(R) }, .block = .{ .x = 256 } }, stream, &a);
            try self.rmsnorm(self.pooled.ptr, l.comp_norm.?.ptr(), self.lat.ptr, R, hd, stream);
        }
        if (l.idx_wk) |*wk| {
            try wk.run(self.lat.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.kraw.ptr, .bf16, stream);
            try self.rmsnorm(self.kraw.ptr, l.idx_k_norm.?.ptr(), self.k.ptr, R, c.idx_dim, stream);
            try self.ropeHeads(self.k.ptr, rope, t_gpos, R, 1, c.idx_dim, false, stream);
            try self.fp4Store(self.k.ptr, c.idx_dim, src.idxk_codes.ptr, src.idxk_scales.ptr, t_ctarget, R, 32, false, stream);
        }
        try self.lat2.copyFrom(0, self.lat.ptr, R * hd * 2, stream.handle);
        try self.ropeHeads(self.lat2.ptr, rope, t_gpos, R, 1, hd, false, stream);
        try self.fp4Store(self.lat2.ptr, hd, src.comp_codes.ptr, src.comp_scales.ptr, t_ctarget, R, 16, true, stream);
    }

    /// rounds.py `_attention`'s scored indexer on layer `li` (an index layer, nb > idx_topk): the rows' index
    /// queries and weights, `_index_score` against each row's stream's extent (KB = the extent's first compressed
    /// row), the candidate pool made (the candidate source) or applied (after it), then the picks into `t_cidx`.
    fn roundIndex(self: *RoundRunner, li: usize, R: usize, nb: usize, kk: usize, j: usize, iq_ready: bool, stream: cuda.Stream) !void {
        const c = self.cfg;
        const l = &self.w.layers[li];
        const src = &self.pool.planes[self.pool.source[li]].?;
        const role = attention.CandRole.of(c.cand_source, li, true);
        const bsize = @max(1, c.cand_block);
        const ccols = (nb + bsize - 1) / bsize;
        const t_vis = self.tab(Tab.ratio(j, Tab.vis));
        const t_cbase = self.tab(Tab.ratio(j, Tab.cbase));
        const vis = self.tab_h[Tab.ratio(j, Tab.vis)..][0..R];
        if (!iq_ready) try l.idx_wq_b.?.run(self.qr.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.iq.ptr, .bf16, stream);
        try self.ropeHeads(self.iq.ptr, self.rope_c, self.tab(Tab.row(Tab.pos)), R, c.idx_heads, c.idx_dim, false, stream);
        const nblk: usize = R * c.idx_heads * c.idx_dim / 32;
        try self.set.run(stream, "_fp4qd_p2", .{ @intCast((nblk + 31) / 32), 1, 1 }, &.{
            aot.ptr("X", "*bf16", self.iq.ptr), aot.ptr("OUT", "*bf16", self.iqq.ptr), aot.int("nblk", @intCast(nblk)),
        }, &.{ aot.ci("BLKS", 32), aot.ci("PDL", 1) });
        try self.set.run(stream, "_rowmm", .{ @intCast(R), @intCast((c.idx_heads + 7) / 8), 1 }, &.{
            aot.ptr("X", "*bf16", self.x.ptr), aot.int("xs", @intCast(c.dim)), aot.ptr("W", "*fp16", self.idx_proj_h[li].ptr), aot.ptr("OUT", "*fp32", self.wl.ptr),
        }, &.{ aot.ci("K", @intCast(c.dim)), aot.ci("N", @intCast(c.idx_heads)), aot.ci("BN", 8), aot.ci("BK", 256), aot.ci("PDL", 1) });
        {
            const scale: f32 = @floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(c.idx_dim))) * @sqrt(@as(f64, @floatFromInt(c.idx_heads)))));
            var a: cuda.Args = .{};
            a.add(self.wl.ptr);
            a.add(self.wts_i.ptr);
            a.add(scale);
            a.add(@as(c_int, @intCast(R * c.idx_heads)));
            try cuda.launch.launch(self.glue.bf16_scale, .{ .grid = .{ .x = @intCast((R * c.idx_heads + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
        }
        const nt: usize = (nb + 63) / 64;
        try self.set.run(stream, "_index_score", .{ @intCast(R), @intCast(nt), 1 }, &.{
            aot.ptr("Q", "*bf16", self.iqq.ptr),     aot.ptr("K", "*u8", src.idxk_codes.ptr),  aot.ptr("KS", "*u8", src.idxk_scales.ptr),
            aot.ptr("Wt", "*bf16", self.wts_i.ptr),  aot.ptr("VIS", "*i64", t_vis),            aot.ptr("OUT", "*fp32", self.score.ptr),
            aot.int("n", @intCast(nb)),              aot.ptr("KB", "*i64", t_cbase),          aot.ptr("CAND", "*i64", t_vis),
            aot.int("cs", 0),                        aot.ptr("TMAX", "*fp32", self.score.ptr), aot.int("nt", @intCast(nt)),
        }, &.{
            aot.ci("IH", @intCast(c.idx_heads)), aot.ci("ID", @intCast(c.idx_dim)), aot.ci("BN", 64),      aot.ci("PACKED", 1),
            aot.ci("HAS_BASE", 1),                aot.ci("PDL", 1),                  aot.ci("HAS_CAND", 0), aot.ci("CB", 8),
            aot.ci("KEYS", 0),                    aot.ci("HAS_TMAX", 0),             aot.ci("SKIP_DEAD", 1), aot.ci("DEAD_PAD", 0),
        });
        if (role == .after and !self.st.cand_made) return error.Unsupported; // no candidate source before this layer
        if (self.glue.hostSelect()) {
            try stream.synchronize();
            try self.score.download(0, std.mem.sliceAsBytes(self.score_h[0 .. R * nb]));
            for (0..R) |r| {
                const row = self.score_h[r * nb .. (r + 1) * nb];
                const mrow = self.cand_h[r * ccols .. (r + 1) * ccols];
                switch (role) {
                    .source => attention.hostCandidates(row, vis[r], c.cand_blocks, bsize, self.keys_h, mrow),
                    .after => attention.hostApplyCandidates(row, mrow, bsize),
                    .none => {},
                }
                attention.hostTopk(row, kk, vis[r], self.keys_h, self.cidx_h[r * kk .. (r + 1) * kk]);
            }
            try self.uploads.put(self.t_cidx, 0, std.mem.sliceAsBytes(self.cidx_h[0 .. R * kk]), stream);
        } else {
            const tg = &self.glue.topk.?;
            switch (role) {
                .source => try tg.candidates(self.score.ptr, nb, R, nb, t_vis, c.cand_blocks, bsize, self.smax.ptr, self.cand_mask.ptr, ccols, stream),
                .after => try tg.apply(self.score.ptr, nb, R, nb, self.cand_mask.ptr, ccols, bsize, stream),
                .none => {},
            }
            const fused = self.glue.select == .auto and self.keys.len >= R * nb * 8 and
                try attention.fusedTopk(self.set, self.score.ptr, R, nb, kk, t_vis, self.keys.ptr, self.t_cidx.ptr, stream);
            if (!fused) try tg.topk(self.score.ptr, nb, R, nb, kk, t_vis, self.t_cidx.ptr, kk, stream);
        }
        if (role == .source) self.st.cand_made = true;
    }

    // -- the fused decode path (rounds.py `_layers_fused`, `_attention2`; TF_DS_FUSED=0: the plain loop above) ----

    /// The side stream starts after everything enqueued on `stream` so far.
    fn fork(self: *RoundRunner, stream: cuda.Stream) !void {
        try self.ev_fork.record(stream);
        try self.side.wait(self.ev_fork);
        self.side_live = true;
    }

    /// `stream` waits for everything enqueued on the side stream so far (rounds.py `join`).
    fn join(self: *RoundRunner, stream: cuda.Stream) !void {
        if (!self.side_live) return;
        try self.ev_join.record(self.side);
        try stream.wait(self.ev_join);
        self.side_live = false;
    }

    /// kernels.hc_pre2 with `sink` (switches "hc_defer", "hc_dots"): the mixes of `st.h`, or (`posted`: the previous
    /// sublayer's gathered partials in `self.gathered`) its post written into `st.spare` alone, which then holds the
    /// streams; the collapse + RMSNorm into x on `stream`, the mixes' dots and the Sinkhorn half (pre_out, post, comb) on
    /// the side stream. The caller joins the side stream before anything reads pre_out, post or comb or writes part.
    fn pre2(self: *RoundRunner, fn_: Ptr, scale: Ptr, base: Ptr, pre_in: Ptr, norm: Ptr, pre_out: Ptr, posted: bool, R: usize, stream: cuda.Stream) !void {
        const rows: u32 = @intCast(R);
        if (!posted) {
            const h = self.st.h;
            try self.hc.mixPart(stream, h, h, null, 1, self.post.ptr, self.comb.ptr, fn_, self.part.ptr, rows, true, false);
            try self.fork(stream);
            try self.hc.sinkhorn(self.side, self.part.ptr, base, scale, pre_out, self.post.ptr, self.comb.ptr, rows);
            try self.hc.finishS(stream, h, self.part.ptr, base, scale, pre_in, norm, self.x.ptr, pre_out, self.post.ptr, self.comb.ptr, rows, false);
            return;
        }
        const h = self.st.h;
        const ho = self.st.spare;
        try self.hc.postOnly(stream, h, ho, self.gathered.ptr, self.world, self.post.ptr, self.comb.ptr, rows);
        try self.hc.finishS(stream, ho, self.part.ptr, base, scale, pre_in, norm, self.x.ptr, pre_out, self.post.ptr, self.comb.ptr, rows, false);
        try self.fork(stream);
        try self.hc.mixPart(self.side, ho, ho, null, 1, self.post.ptr, self.comb.ptr, fn_, self.part.ptr, rows, false, false);
        try self.hc.sinkhorn(self.side, self.part.ptr, base, scale, pre_out, self.post.ptr, self.comb.ptr, rows);
        self.st.h = ho;
        self.st.spare = h;
    }

    /// Layers [sg.first, sg.end) of the fused round (rounds.py `_layers_fused`): each sublayer's post fused into the
    /// next sublayer's mixes, the stretch's last post alone at its end; a tap taken of the streams the attention
    /// sublayer's mixes read. The same bits as the plain loop.
    fn fusedLayers(self: *RoundRunner, sg: Seg, R: usize, gather: fwd.Gather, stream: cuda.Stream) !void {
        const c = self.cfg;
        const D = c.dim;
        var nb: [48]u8 = undefined;
        var posted = false;
        for (self.w.layers[sg.first..sg.end], sg.first..) |*l, li| {
            try self.pre2(l.hc_attn_fn.ptr(), l.hc_attn_scale.ptr(), l.hc_attn_base.ptr(), self.st.pre, l.attn_norm.ptr(), self.pre_a.ptr, posted, R, stream);
            if (posted) try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{li - 1}), self.st.h, R * 4 * D * 2, stream);
            if (c.dspark_taps.has(@intCast(li))) {
                try self.hc.tap(stream, self.st.h, self.taps.ptr + self.st.tap * D * 2, @intCast(@max(self.ntap, 1) * D), @intCast(R));
                self.st.tap += 1;
            }
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_attn", .{li}), self.x.ptr, R * D * 2, stream);
            const pa = try self.fusedAttention(li, R, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_attn.r0", .{li}), pa, R * D * 4, stream);
            try self.join(stream);
            try gather.run(pa, R * D, self.gathered.ptr, stream);
            try self.pre2(l.hc_ffn_fn.ptr(), l.hc_ffn_scale.ptr(), l.hc_ffn_base.ptr(), self.pre_a.ptr, l.ffn_norm.ptr(), self.st.pre_f, true, R, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_attn", .{li}), self.st.h, R * 4 * D * 2, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_ffn", .{li}), self.x.ptr, R * D * 2, stream);
            try self.route.gate(stream, self.x.ptr, l.gate_w.ptr(), self.parts.ptr, @intCast(R));
            try self.route.pick(stream, self.parts.ptr, l.gate_b.ptr(), self.pick.ptr, self.wts.ptr, @intCast(R));
            try experts.routed(self.ek, &l.experts, &self.scratch, self.x.ptr, self.pick.ptr, self.wts.ptr, R, c.swiglu_limit, self.moe.ptr, stream);
            try self.l2.next(li, sg.end, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_moe.r0", .{li}), self.moe.ptr, R * D * 4, stream);
            try self.join(stream);
            try gather.run(self.moe.ptr, R * D, self.gathered.ptr, stream);
            posted = true;
            const t = self.st.pre;
            self.st.pre = self.st.pre_f;
            self.st.pre_f = t;
        }
        if (posted) {
            try self.hc.post(stream, self.gathered.ptr, self.world, self.st.h, self.post.ptr, self.comb.ptr, self.st.h, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{sg.end - 1}), self.st.h, R * 4 * D * 2, stream);
        }
        try self.l2.join(stream);
    }

    /// rounds.py `_attention2` at the fork's default switches for layer `li`: attn_in as one grouped launch, q's
    /// RMSNorm writing wq_b's (and the indexer wq_b's) rotated rows beside the window KV's norm + RoPE into the ring
    /// (`_q_kv_norm`), q's RoPE in wq_b's epilogue, the compressor and the indexer as the plain path, the attention
    /// merge applying the inverse RoPE and writing wo_a's rotated rows (`_sparse_attn_merge_rot2`), wo_a's epilogue
    /// writing wo_b's rotated rows. This rank's fp32 partial.
    fn fusedAttention(self: *RoundRunner, li: usize, R: usize, stream: cuda.Stream) !Ptr {
        const c = self.cfg;
        const l = &self.w.layers[li];
        const g = &self.lg[li];
        const hd = c.head_dim;
        const heads = c.n_heads / self.world;
        const ratio = c.ratio(li);
        const rope = if (ratio > 0) self.rope_c else self.rope_w;
        const t_pos = self.tab(Tab.row(Tab.pos));
        const t_wbase = self.tab(Tab.row(Tab.wbase));
        const ring = self.pool.rings[li];
        // -- attn_in: wq_a, wkv (and the compressor's) in one rot_many and the glinear launches ------------------
        {
            const xr: exl3.Rows = .{ .ptr = self.x.ptr, .dtype = .bf16, .ld = c.dim };
            const xs = [_]exl3.Rows{ xr, xr, xr, xr };
            var outs: [4]exl3.Rows = .{
                .{ .ptr = self.qa.ptr, .dtype = .bf16, .ld = c.q_rank },
                .{ .ptr = self.ykv.ptr, .dtype = .bf16, .ld = hd },
                .{ .ptr = self.ckv.ptr, .dtype = if (ratio == 1) .bf16 else .f32, .ld = hd },
                .{ .ptr = self.cgate.ptr, .dtype = .f32, .ld = hd },
            };
            if (g.in.n > 2) outs[2].ld = l.comp_wkv.?.n;
            if (g.in.n > 3) outs[3].ld = l.comp_wgate.?.n;
            try g.in.rotate(xs[0..g.in.n], R, self.xh_in.ptr, stream);
            try g.in.rotated(R, self.xh_in.ptr, outs[0..g.in.n], self.zg.ptr, null, null, stream);
        }
        // -- q: _q_kv_norm (qr, the q group's rotated rows, the window KV into the ring), wq_b with q's RoPE -------
        const has_idx = ratio > 0 and l.idx_wq_b != null;
        const nb_ = if (ratio > 0) self.round_bucket / ratio else 0;
        const idx_all = has_idx and @as(i64, @intCast(li)) != c.cand_source and nb_ <= c.idx_topk;
        const need_iq = has_idx and !idx_all;
        const qg = if (need_iq) &g.q2.? else &g.q1;
        {
            const nrot = qg.n;
            const h0 = self.xh_q.ptr;
            const h1 = self.xh_q.ptr + qg.offset(R, 1) * 2;
            const block = std.math.ceilPowerOfTwo(usize, c.q_rank) catch return error.BadShape;
            try self.set.run(stream, "_q_kv_norm", .{ @intCast(R), @intCast(2 + nrot), 1 }, &.{
                aot.ptr("X", "*bf16", self.qa.ptr),       aot.int("xs", @intCast(c.q_rank)),    aot.ptr("W", "*bf16", l.q_norm.ptr()),
                aot.ptr("OUT", "*bf16", self.qr.ptr),     aot.int("os_", @intCast(c.q_rank)),   aot.float("eps", c.eps),
                aot.ptr("S0", "*fp16", l.wq_b.suh.ptr),   aot.ptr("H0", "*fp16", h0),
                if (nrot > 1) aot.ptr("S1", "*fp16", l.idx_wq_b.?.suh.ptr) else aot.ptr("S1", "*bf16", self.qr.ptr),
                if (nrot > 1) aot.ptr("H1", "*fp16", h1) else aot.ptr("H1", "*bf16", self.qr.ptr),
                aot.ptr("Y", "*bf16", self.ykv.ptr),      aot.ptr("WK", "*bf16", l.kv_norm.ptr()), aot.ptr("COS", "*fp32", rope.cos.ptr),
                aot.ptr("SIN", "*fp32", rope.sin.ptr),    aot.ptr("POS", "*i64", t_pos),          aot.ptr("KOUT", "*bf16", self.kv.ptr),
                aot.ptr("RING", "*bf16", ring.ptr),       aot.ptr("SLOT_OF", "*i64", self.tab(Tab.row(Tab.wslot))),
                aot.int("ring_size", @intCast(self.pool.slots * self.pool.ring)),
            }, &.{
                aot.ci("QUANT", 1), aot.ci("D", @intCast(c.q_rank)), aot.ci("BLOCK", @intCast(block)), aot.ci("NROT", @intCast(nrot)),
                aot.ci("DK", @intCast(hd)), aot.ci("RD", @intCast(c.rope_dim)), aot.ci("PDL", 1), aot.ci("SPLIT", 1),
            });
            var flags: [exl3.gmax]bool = @splat(false);
            flags[0] = true;
            const outs = [_]exl3.Rows{
                .{ .ptr = self.q.ptr, .dtype = .bf16, .ld = heads * hd },
                .{ .ptr = self.iq.ptr, .dtype = .bf16, .ld = c.idx_heads * c.idx_dim },
            };
            try qg.rotated(R, self.xh_q.ptr, outs[0..nrot], self.zg.ptr, null, .{ .cos = rope.cos.ptr, .sin = rope.sin.ptr, .pos = t_pos, .hd = hd, .rd = c.rope_dim, .layers = flags }, stream);
        }
        try self.l2.attn(li, stream);
        // -- the compressor's caches and the indexer (as the plain path; the window KV is in the ring already) ----
        var comp_codes: Ptr = ring.ptr;
        var comp_scales: Ptr = ring.ptr;
        var t_cbase: Ptr = t_wbase;
        var kk: usize = 0;
        if (ratio > 0) {
            const j = self.ratioSlot(ratio);
            t_cbase = self.tab(Tab.ratio(j, Tab.cbase));
            const t_ctarget = self.tab(Tab.ratio(j, Tab.ctarget));
            const t_gpos = self.tab(Tab.ratio(j, Tab.gpos));
            const src = &self.pool.planes[self.pool.source[li]].?;
            if (c.isKvSource(li)) try self.compressor(li, R, j, src, rope, t_ctarget, t_gpos, stream);
            kk = @min(c.idx_topk, nb_);
            if (c.isIndexSource(li)) {
                if (nb_ > c.idx_topk) {
                    try self.roundIndex(li, R, nb_, kk, j, need_iq, stream);
                    self.st.cidx = self.t_cidx.ptr;
                } else self.st.cidx = self.tab(Tab.picks(j, self.kmax));
            }
            comp_codes = src.comp_codes.ptr;
            comp_scales = src.comp_scales.ptr;
        }
        // -- sparse attention; the merge with the inverse RoPE and wo_a's input rotation ------------------------
        {
            const gh = heads / 16;
            const has = ratio > 0;
            const picks: usize = if (has) (kk + 31) / 32 else 0;
            const nblk: usize = c.window / 32 + picks;
            const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
            try self.set.run(stream, "_sparse_attn_part", .{ @intCast(R), @intCast(gh), 8 }, &.{
                aot.ptr("Q", "*bf16", self.q.ptr),       aot.ptr("WSRC", "*bf16", ring.ptr),
                aot.ptr("WLO", "*i64", self.t_zero.ptr), if (has) aot.ptr("COMP", "*u8", comp_codes) else aot.ptr("COMP", "*bf16", ring.ptr),
                if (has) aot.ptr("CSC", "*u8", comp_scales) else aot.ptr("CSC", "*bf16", ring.ptr), aot.ptr("IDX", "*i64", if (has) self.st.cidx else t_pos),
                aot.ptr("POS", "*i64", t_pos),           aot.ptr("PM", "*fp32", self.pm.ptr),
                aot.ptr("PL", "*fp32", self.pl.ptr),     aot.ptr("PO", "*fp32", self.po.ptr),
                aot.ptr("SINK", "*fp32", l.sink.ptr()),  aot.ptr("OUT", "*bf16", self.o.ptr),
                aot.float("scale", scale),               aot.int("ring_size", @intCast(self.pool.ring)),
                aot.int("n_idx", @intCast(kk)),          aot.ptr("RBASE", "*i64", t_wbase),
                aot.ptr("CBASE", "*i64", t_cbase),
            }, &.{
                aot.ci("H", @intCast(heads)), aot.ci("HD", @intCast(hd)),       aot.ci("HB", 16),    aot.ci("WIN", @intCast(c.window)),
                aot.ci("BN", 32),             aot.ci("RING", 1),                 aot.ci("HAS_COMP", @intFromBool(has)), aot.ci("PACKED", @intFromBool(has)),
                aot.ci("SPLITS", 8),          aot.ci("NBLK", @intCast(nblk)),    aot.ci("FINAL", 0),  aot.ci("HAS_BASE", 1),
                aot.ci("PDL", 1),             aot.ci("WPF", 1),
            });
            const ghd = l.wo_a[0].k / hd; // heads a wo_a slice
            try self.set.run(stream, "_sparse_attn_merge_rot2", .{ @intCast(R), @intCast(heads), 1 }, &.{
                aot.ptr("PM", "*fp32", self.pm.ptr),   aot.ptr("PL", "*fp32", self.pl.ptr),    aot.ptr("PO", "*fp32", self.po.ptr),
                aot.ptr("SINK", "*fp32", l.sink.ptr()), aot.ptr("OUT", "*bf16", self.o.ptr),   aot.ptr("COS", "*fp32", rope.cos.ptr),
                aot.ptr("SIN", "*fp32", rope.sin.ptr), aot.ptr("POS", "*i64", t_pos),          aot.ptr("SUH", "*fp16", g.wo_a_suh.ptr),
                aot.ptr("XH", "*fp16", self.xh_woa.ptr), aot.int("rows", @intCast(R)),
            }, &.{
                aot.ci("H", @intCast(heads)), aot.ci("HD", @intCast(hd)), aot.ci("HB", 16), aot.ci("SPLITS", 8), aot.ci("RD", @intCast(c.rope_dim)),
                aot.ci("GH", @intCast(ghd)),  aot.ci("PDL", 1),           aot.ci("DISCARD", 1),
            });
        }
        // -- wo_a (its epilogue writing wo_b's rotated rows), then wo_b ------------------------------------------
        {
            const groups = g.wo_a.n;
            var outs: [exl3.gmax]exl3.Rows = undefined;
            var rot: [exl3.gmax]?exl3.RotOut = @splat(null);
            var col: usize = 0;
            for (0..groups) |q| {
                const n = l.wo_a[q].n;
                outs[q] = .{ .ptr = self.u.ptr + col * 2, .dtype = .bf16, .ld = groups * c.o_rank };
                rot[q] = .{ .suh = l.wo_b.suh.ptr, .xh = self.xh_wob.ptr, .ld = l.wo_b.k, .off = col };
                col += n;
            }
            try g.wo_a.rotated(R, self.xh_woa.ptr, outs[0..groups], self.zg.ptr, rot[0..groups], null, stream);
            const ob = [_]exl3.Rows{.{ .ptr = self.attn_out.ptr, .dtype = .f32, .ld = c.dim }};
            try g.wo_b.rotated(R, self.xh_wob.ptr, &ob, self.zg.ptr, null, null, stream);
        }
        try self.l2.moe(li, stream);
        return self.attn_out.ptr;
    }

    fn rmsnorm(self: *RoundRunner, x: Ptr, w: Ptr, out: Ptr, n: usize, d: usize, stream: cuda.Stream) !void {
        const block = std.math.ceilPowerOfTwo(usize, d) catch return error.BadShape;
        try self.set.run(stream, "_rmsnorm", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("W", "*bf16", w),
            aot.ptr("OUT", "*bf16", out), aot.int("os_", @intCast(d)), aot.float("eps", self.cfg.eps),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("PDL", 1) });
    }

    fn ropeHeads(self: *RoundRunner, x: Ptr, rope: attention.Rope, pos: Ptr, n: usize, h: usize, hd: usize, inverse: bool, stream: cuda.Stream) !void {
        const hb = @min(h, 16);
        try self.set.run(stream, "_rope_heads", .{ @intCast(n), @intCast(h / hb), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("COS", "*fp32", rope.cos.ptr), aot.ptr("SIN", "*fp32", rope.sin.ptr), aot.ptr("POS", "*i64", pos),
        }, &.{ aot.ci("H", @intCast(h)), aot.ci("HD", @intCast(hd)), aot.ci("RD", @intCast(self.cfg.rope_dim)), aot.ci("INV", @intFromBool(inverse)), aot.ci("HB", @intCast(hb)), aot.ci("PDL", 1) });
    }

    fn fp4Store(self: *RoundRunner, x: Ptr, d: usize, codes: Ptr, scales: Ptr, rows: Ptr, n: usize, block: usize, e4m3: bool, stream: cuda.Stream) !void {
        try self.set.run(stream, "_fp4_store", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("CODES", "*u8", codes), aot.ptr("SCALES", "*u8", scales), aot.ptr("ROWS", "*i64", rows),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("E4M3", @intFromBool(e4m3)), aot.ci("PDL", 1) });
    }

    /// dst[idx[r]] = src[r] for r < rows, rows of `words` 4-byte words (prompt_glue.cu tf_dsv41_copy_rows).
    fn copyRows(self: *RoundRunner, src: Ptr, dst: Ptr, idx: Ptr, rows: usize, words: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(Ptr, 0));
        a.add(dst);
        a.add(idx);
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(words)));
        try cuda.launch.launch(self.copy_rows, .{ .grid = .{ .x = @intCast((words + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    fn copyCols(self: *RoundRunner, src: Ptr, sld: usize, dst: Ptr, dld: usize, rows: usize, cols: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_int, @intCast(sld)));
        a.add(dst);
        a.add(@as(c_int, @intCast(dld)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(cols)));
        try cuda.launch.launch(self.glue.copy_cols, .{ .grid = .{ .x = @intCast((cols + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }
};

test "a round's bucket is graph.py bucket_for: a power of two holding the deepest row, at least the least, at most the pool's" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 1024), bucketFor(1, 1 << 20, bucket_min));
    try t.expectEqual(@as(usize, 1024), bucketFor(1024, 1 << 20, bucket_min));
    try t.expectEqual(@as(usize, 2048), bucketFor(1025, 1 << 20, bucket_min));
    try t.expectEqual(@as(usize, 1 << 20), bucketFor((1 << 19) + 1, 1 << 20, bucket_min));
    // a pool narrower than the power of two: its capacity (the fork's min(b, max(cap, BUCKET_MIN)))
    try t.expectEqual(@as(usize, 4120), bucketFor(4100, 4120, bucket_min));
    try t.expectEqual(@as(usize, 1024), bucketFor(100, 600, bucket_min));
    // the least bucket the pool's widest: every round scans the whole pool
    try t.expectEqual(@as(usize, 1 << 20), bucketFor(7, 1 << 20, 1 << 20));
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    for (0..2000) |_| {
        const cap = rnd.intRangeAtMost(usize, 1, 1 << 21);
        const deep = rnd.intRangeAtMost(usize, 1, cap);
        const b = bucketFor(deep, cap, bucket_min);
        try t.expect(b >= deep and b >= bucket_min and b <= @max(cap, bucket_min));
        try t.expect(std.math.isPowerOfTwo(b) or b == cap);
        // the smallest such: half of it does not hold the row
        if (b > bucket_min) try t.expect(b / 2 < deep);
    }
    // monotone in the deepest row over one pool
    var last: usize = 0;
    for (1..70000) |deep| {
        const b = bucketFor(deep, 65536 + 24, bucket_min);
        try t.expect(b >= last);
        last = b;
    }
}

test "the round's tables: every table its own 16-aligned words inside the buffer" {
    const t = std.testing;
    for ([_]usize{ 1, 64, 512, 1000 }) |kmax| {
        var spans: [Tab.rows + max_ratios * Tab.per_ratio + max_ratios][2]usize = undefined;
        var n: usize = 0;
        for (0..Tab.rows) |k| {
            spans[n] = .{ Tab.row(k), Tab.row(k) + max_rows };
            n += 1;
        }
        for (0..max_ratios) |j| for (0..Tab.per_ratio) |k| {
            spans[n] = .{ Tab.ratio(j, k), Tab.ratio(j, k) + max_rows };
            n += 1;
        };
        for (0..max_ratios) |j| {
            spans[n] = .{ Tab.picks(j, kmax), Tab.picks(j, kmax) + max_rows * kmax };
            n += 1;
        }
        for (spans[0..n], 0..) |a, i| {
            try t.expect(a[1] <= Tab.words(kmax));
            try t.expect(a[0] * 8 % 16 == 0);
            for (spans[0..n][i + 1 ..]) |b| try t.expect(a[1] <= b[0] or b[1] <= a[0]);
        }
    }
}

test "a ratio-2 group's raw rows stay in its slot: comp_pool2 over the whole plane reads the slot's pair" {
    // gfirst = slot x RAW + gpos % RAW with gpos even: gfirst and gfirst + 1 (comp_pool2's pair, modulo the plane's
    // slots x RAW rows) are rows of that slot, the pair the per-slot launch read at (gpos % RAW, (gpos + 1) % RAW)
    const t = std.testing;
    const RAW: i64 = raw_rows;
    for (1..9) |slots| for (0..slots) |slot| for (0..300) |p| {
        const gpos: i64 = @divFloor(@as(i64, @intCast(p)), 2) * 2;
        const gfirst = @as(i64, @intCast(slot)) * RAW + @mod(gpos, RAW);
        const plane = @as(i64, @intCast(slots)) * RAW;
        const a = @mod(gfirst, plane);
        const b = @mod(gfirst + 1, plane);
        try t.expectEqual(@as(i64, @intCast(slot)) * RAW + @mod(gpos, RAW), a);
        try t.expectEqual(@as(i64, @intCast(slot)) * RAW + @mod(gpos + 1, RAW), b);
    };
}
