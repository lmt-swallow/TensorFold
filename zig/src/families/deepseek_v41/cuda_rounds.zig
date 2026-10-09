//! Concurrent decode rounds (`--parallel`, the Python family's rounds.py `RoundDecoder` and model.py `PoolCache`):
//! one forward over every live stream's verify window, every row at its own stream's position and in its own
//! stream's cache rows. A pool is one window of `cap` positions whose compressed planes the streams share by
//! extents (a stream owns positions [base, base + size) of the planes, its last compressed row its scratch row), and
//! per-slot window rings and compressor inputs. A row's kernels address its stream's rows through per-row tables
//! (ids, positions, slot, extent base and end), the fork's `_ix` arithmetic computed here on the host.
//!
//! The unfused path (TF_DS_HC_FUSED=0, TF_DS_ROT_*=0): plain hc_pre / hc_post, the attention's q path in three
//! launches, the merge apart from wo_a; the fork documents the fused launches as the same bits. The indexer's
//! selection is the short-context one (`alltop`: every compressed entry below the row's visible count, bucket /
//! ratio <= idx_topk); longer buckets' scored top-k is not here yet. Engram runs on its layers when a hook is set
//! (`engram`), each row's n-grams from its own stream's ids (`Window.host`).
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

/// A round's forward over one pool: R <= 16 rows of several streams, the fork's `RoundDecoder` without graphs.
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
    bucket: usize,
    rope_c: attention.Rope,
    rope_w: attention.Rope,
    pool: *Pool,
    probe: ?fwd.Probe = null,
    engram: ?Engram = null,
    idx_proj_h: []cuda.DeviceBuffer, // per layer: fp16 [idx_heads, d] on index layers
    // the round's tables (int64 [R]), host and device
    tab: []i64, // 12 x max_rows: ids pos slot base end wslot cbase ctarget gpos vis rslot first
    cidx_h: []i64,
    uploads: attention.Uploads, // the tables' copies: 4 a round, then 3 and the picks a compressed layer
    t_ids: cuda.DeviceBuffer,
    t_pos: cuda.DeviceBuffer,
    t_wslot: cuda.DeviceBuffer,
    t_wbase: cuda.DeviceBuffer,
    t_cbase: cuda.DeviceBuffer,
    t_ctarget: cuda.DeviceBuffer,
    t_gpos: cuda.DeviceBuffer,
    t_cidx: cuda.DeviceBuffer,
    t_zero: cuda.DeviceBuffer,
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
        self.tab = try gpa.alloc(i64, 12 * R);
        const kk = @min(cfg.idx_topk, bucket); // the widest selection (ratio 1)
        self.cidx_h = try gpa.alloc(i64, R * kk);
        self.uploads = try attention.Uploads.init(d, attention.Uploads.room(4 + 3 * w.layers.len, R * 8) + attention.Uploads.room(w.layers.len, R * kk * 8));
        self.t_ids = try cuda.DeviceBuffer.alloc(d, R * 8);
        self.t_pos = try cuda.DeviceBuffer.alloc(d, R * 8);
        self.t_wslot = try cuda.DeviceBuffer.alloc(d, R * 8);
        self.t_wbase = try cuda.DeviceBuffer.alloc(d, R * 8);
        self.t_cbase = try cuda.DeviceBuffer.alloc(d, R * 8);
        self.t_ctarget = try cuda.DeviceBuffer.alloc(d, R * 8);
        self.t_gpos = try cuda.DeviceBuffer.alloc(d, R * 8);
        self.t_cidx = try cuda.DeviceBuffer.alloc(d, R * kk * 8);
        self.t_zero = try cuda.DeviceBuffer.alloc(d, R * 8);
        try self.t_zero.fill8(0, null);
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
        return self;
    }

    pub fn deinit(self: *RoundRunner) void {
        for (self.idx_proj_h) |*b| if (b.len > 0) b.free();
        self.gpa.free(self.idx_proj_h);
        self.gpa.free(self.tab);
        self.gpa.free(self.cidx_h);
        self.uploads.deinit();
        inline for (.{ "t_ids", "t_pos", "t_wslot", "t_wbase", "t_cbase", "t_ctarget", "t_gpos", "t_cidx", "t_zero", "h", "pre", "pre_a", "pre_f", "post", "comb", "part", "x", "xh", "z", "qa", "qr", "q", "ykv", "kv", "ckv", "cgate", "pooled", "lat", "lat2", "kraw", "k", "pm", "pl", "po", "o", "og", "ug", "u", "attn_out", "gathered", "parts", "pick", "wts", "moe", "xc", "local", "glog", "logits", "taps" }) |f| @field(self, f).free();
        self.scratch.deinit();
    }

    fn probeOut(self: *const RoundRunner, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) !void {
        if (self.probe) |p| try p.call(p.ctx, name, ptr, bytes, stream);
    }

    /// Every window's rows in one forward: logits [R, V] f32 (`self.logits`, row order = the windows'), the taps
    /// in `self.taps` [R, ntap x d]; the pool holds the rows' keys after.
    pub fn forward(self: *RoundRunner, windows: []const Window, gather: fwd.Gather, stream: cuda.Stream) !Ptr {
        const c = self.cfg;
        const D = c.dim;
        var R: usize = 0;
        for (windows) |wd| R += wd.ids.len;
        if (R < 1 or R > max_rows) return error.TooManyRows;
        // -- the per-row tables (rounds.py RoundDecoder.set and _ix) ----------------------------------------------
        const ids = self.tab[0..max_rows];
        const pos = self.tab[max_rows .. 2 * max_rows];
        const slot = self.tab[2 * max_rows .. 3 * max_rows];
        const base = self.tab[3 * max_rows .. 4 * max_rows];
        const end = self.tab[4 * max_rows .. 5 * max_rows];
        // every row inside its extent, the extent inside the window, its positions inside the RoPE tables
        const rope_rows = @min(self.rope_c.rows, self.rope_w.rows);
        for (windows) |wd| {
            if (wd.ids.len == 0 or wd.slot >= self.pool.slots or wd.base + wd.size > self.pool.cap) return error.BadShape;
            if (wd.position + wd.ids.len > wd.size or wd.position + wd.ids.len > rope_rows) return error.BadShape;
            if (self.engram != null and wd.host.len < wd.position + wd.ids.len) return error.BadShape;
        }
        var r: usize = 0;
        for (windows) |wd| {
            for (wd.ids, 0..) |id, i| {
                ids[r] = id;
                pos[r] = @intCast(wd.position + i);
                slot[r] = @intCast(wd.slot);
                base[r] = @intCast(wd.base);
                end[r] = @intCast(wd.base + wd.size);
                r += 1;
            }
        }
        const wslot = self.tab[5 * max_rows .. 6 * max_rows];
        const wbase = self.tab[6 * max_rows .. 7 * max_rows];
        const RS: i64 = @intCast(self.pool.ring);
        for (0..R) |i| {
            wbase[i] = slot[i] * RS;
            wslot[i] = wbase[i] + @mod(pos[i], RS);
        }
        try self.uploads.begin();
        try self.uploads.put(self.t_ids, 0, std.mem.sliceAsBytes(ids[0..R]), stream);
        try self.uploads.put(self.t_pos, 0, std.mem.sliceAsBytes(pos[0..R]), stream);
        try self.uploads.put(self.t_wslot, 0, std.mem.sliceAsBytes(wslot[0..R]), stream);
        try self.uploads.put(self.t_wbase, 0, std.mem.sliceAsBytes(wbase[0..R]), stream);
        // -- the layers ------------------------------------------------------------------------------------------
        const w = self.w;
        try self.hc.embedInit(stream, w.embed.ptr(), self.t_ids.ptr, self.h.ptr, self.pre.ptr, @intCast(R));
        var pre = self.pre.ptr;
        var pre_f = self.pre_f.ptr;
        var nb: [48]u8 = undefined;
        var tap: usize = 0;
        var spans: [max_rows]EngramSpan = undefined;
        if (self.engram != null) {
            for (windows, 0..) |wd, i| spans[i] = .{ .ids = wd.host, .start = wd.position, .n = wd.ids.len };
        }
        for (w.layers, 0..) |*l, li| {
            if (l.engram_wkv != null) if (self.engram) |e| {
                try e.call(e.ctx, li, self.h.ptr, spans[0..windows.len], gather, stream);
                try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_engram", .{li}), self.h.ptr, R * 4 * D * 2, stream);
            };
            if (c.dspark_taps.has(@intCast(li))) {
                try self.hc.tap(stream, self.h.ptr, self.taps.ptr + tap * D * 2, @intCast(@max(self.ntap, 1) * D), @intCast(R));
                tap += 1;
            }
            try self.hc.pre(stream, self.h.ptr, l.hc_attn_fn.ptr(), l.hc_attn_scale.ptr(), l.hc_attn_base.ptr(), pre, l.attn_norm.ptr(), self.x.ptr, self.pre_a.ptr, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_attn", .{li}), self.x.ptr, R * D * 2, stream);
            const pa = try self.roundAttention(li, windows, R, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_attn.r0", .{li}), pa, R * D * 4, stream);
            try gather.run(pa, R * D, self.gathered.ptr, stream);
            try self.hc.post(stream, self.gathered.ptr, self.world, self.h.ptr, self.post.ptr, self.comb.ptr, self.h.ptr, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_attn", .{li}), self.h.ptr, R * 4 * D * 2, stream);
            try self.hc.pre(stream, self.h.ptr, l.hc_ffn_fn.ptr(), l.hc_ffn_scale.ptr(), l.hc_ffn_base.ptr(), self.pre_a.ptr, l.ffn_norm.ptr(), self.x.ptr, pre_f, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_ffn", .{li}), self.x.ptr, R * D * 2, stream);
            try self.route.gate(stream, self.x.ptr, l.gate_w.ptr(), self.parts.ptr, @intCast(R));
            try self.route.pick(stream, self.parts.ptr, l.gate_b.ptr(), self.pick.ptr, self.wts.ptr, @intCast(R));
            try experts.routed(self.ek, &l.experts, &self.scratch, self.x.ptr, self.pick.ptr, self.wts.ptr, R, c.swiglu_limit, self.moe.ptr, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_moe.r0", .{li}), self.moe.ptr, R * D * 4, stream);
            try gather.run(self.moe.ptr, R * D, self.gathered.ptr, stream);
            try self.hc.post(stream, self.gathered.ptr, self.world, self.h.ptr, self.post.ptr, self.comb.ptr, self.h.ptr, @intCast(R));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{li}), self.h.ptr, R * 4 * D * 2, stream);
            const t = pre;
            pre = pre_f;
            pre_f = t;
        }
        // -- the head: every rank's vocabulary columns, rank order within a row ----------------------------------
        try self.hc.collapseNorm(stream, self.h.ptr, pre, w.norm.ptr(), self.xc.ptr, @intCast(R));
        try self.probeOut("head.xc", self.xc.ptr, R * D * 2, stream);
        try w.head.run(self.xc.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.local.ptr, .f32, stream);
        const vl = c.vocab / self.world;
        try self.probeOut("head.local.r0", self.local.ptr, R * vl * 4, stream);
        try gather.run(self.local.ptr, R * vl, self.glog.ptr, stream);
        for (0..self.world) |rk| {
            // [world, R, vl] -> [R, world x vl]: rank rk's columns of every row (f32 moved as pairs of 16-bit words)
            try self.copyCols(self.glog.ptr + rk * R * vl * 4, vl * 2, self.logits.ptr + rk * vl * 4, c.vocab * 2, R, vl * 2, stream);
        }
        return self.logits.ptr;
    }

    /// rounds.py `RoundDecoder._attention` (unfused) for layer `li` over the window rows: this rank's fp32 partial.
    fn roundAttention(self: *RoundRunner, li: usize, windows: []const Window, R: usize, stream: cuda.Stream) !Ptr {
        const c = self.cfg;
        const l = &self.w.layers[li];
        const hd = c.head_dim;
        const heads = c.n_heads / self.world;
        const groups = c.o_groups / self.world;
        const ratio = c.ratio(li);
        const rope = if (ratio > 0) self.rope_c else self.rope_w;
        const pos = self.tab[max_rows .. 2 * max_rows];
        const base = self.tab[3 * max_rows .. 4 * max_rows];
        const end = self.tab[4 * max_rows .. 5 * max_rows];
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
        try self.ropeHeads(self.q.ptr, rope, self.t_pos.ptr, R, heads, hd, false, stream);
        // -- the window KV into the slot's ring rows ------------------------------------------------------------
        try self.set.run(stream, "_kv_norm_rope", .{ @intCast(R), 1, 1 }, &.{
            aot.ptr("Y", "*bf16", self.ykv.ptr),  aot.ptr("W", "*bf16", l.kv_norm.ptr()), aot.ptr("COS", "*fp32", rope.cos.ptr),
            aot.ptr("SIN", "*fp32", rope.sin.ptr), aot.ptr("POS", "*i64", self.t_pos.ptr),     aot.ptr("OUT", "*bf16", self.kv.ptr),
            aot.ptr("RING", "*bf16", ring.ptr),   aot.ptr("SLOT_OF", "*i64", self.t_wslot.ptr), aot.int("ring_size", @intCast(self.pool.slots * self.pool.ring)),
            aot.float("eps", c.eps),
        }, &.{ aot.ci("QUANT", 1), aot.ci("D", @intCast(hd)), aot.ci("RD", @intCast(c.rope_dim)), aot.ci("PDL", 1) });
        var comp_codes: Ptr = ring.ptr;
        var comp_scales: Ptr = ring.ptr;
        var kk: usize = 0;
        if (ratio > 0) {
            const rt: i64 = @intCast(ratio);
            const cbase = self.tab[7 * max_rows .. 8 * max_rows];
            const ctarget = self.tab[8 * max_rows .. 9 * max_rows];
            const gpos = self.tab[9 * max_rows .. 10 * max_rows];
            const vis = self.tab[10 * max_rows .. 11 * max_rows];
            for (0..R) |i| {
                cbase[i] = @divFloor(base[i], rt);
                const scratch = @divFloor(end[i], rt) - 1 - cbase[i];
                const groups_i = @divFloor(pos[i], rt);
                const target: i64 = if (@mod(pos[i] + 1, rt) == 0) groups_i else scratch;
                ctarget[i] = cbase[i] + (if (ratio == 1) pos[i] else target);
                gpos[i] = (if (ratio == 1) pos[i] else groups_i) * rt;
                vis[i] = @divFloor(pos[i] + 1, rt);
            }
            try self.uploads.put(self.t_cbase, 0, std.mem.sliceAsBytes(cbase[0..R]), stream);
            try self.uploads.put(self.t_ctarget, 0, std.mem.sliceAsBytes(ctarget[0..R]), stream);
            try self.uploads.put(self.t_gpos, 0, std.mem.sliceAsBytes(gpos[0..R]), stream);
            const src = &self.pool.planes[self.pool.source[li]].?;
            if (kv_source) {
                // -- the kv source: compressed latents (and index keys) of the rows' groups ----------------------
                if (ratio == 1) {
                    try self.rmsnorm(self.ckv.ptr, l.comp_norm.?.ptr(), self.lat.ptr, R, hd, stream);
                } else {
                    if (ratio != 2) return error.Unsupported;
                    // the rows' compressor inputs by position into the slot's raw rows, then each stream's groups'
                    // softmax pool (the raw rows of one slot a launch)
                    var r0: usize = 0;
                    for (windows) |wd| {
                        const rb: usize = wd.slot * raw_rows;
                        for (0..wd.ids.len) |i| {
                            const rs: usize = rb + @as(usize, @intCast(@mod(pos[r0 + i], @as(i64, raw_rows))));
                            try src.raw_kv.copyFrom(rs * hd * 4, try self.ckv.at((r0 + i) * hd * 4), hd * 4, stream.handle);
                            try src.raw_score.copyFrom(rs * hd * 4, try self.cgate.at((r0 + i) * hd * 4), hd * 4, stream.handle);
                        }
                        var a: cuda.Args = .{};
                        a.add(src.raw_kv.ptr + rb * hd * 4);
                        a.add(src.raw_score.ptr + rb * hd * 4);
                        a.add(self.t_gpos.ptr + r0 * 8);
                        a.add(@as(c_int, raw_rows));
                        a.add(@as(c_int, @intCast(hd)));
                        a.add(self.pooled.ptr + r0 * hd * 2);
                        try cuda.launch.launch(self.glue.comp_pool2, .{ .grid = .{ .x = @intCast((hd + 255) / 256), .y = @intCast(wd.ids.len) }, .block = .{ .x = 256 } }, stream, &a);
                        r0 += wd.ids.len;
                    }
                    try self.rmsnorm(self.pooled.ptr, l.comp_norm.?.ptr(), self.lat.ptr, R, hd, stream);
                }
                if (l.idx_wk) |*wk| {
                    try wk.run(self.lat.ptr, .bf16, R, self.xh.ptr, self.z.ptr, self.kraw.ptr, .bf16, stream);
                    try self.rmsnorm(self.kraw.ptr, l.idx_k_norm.?.ptr(), self.k.ptr, R, c.idx_dim, stream);
                    try self.ropeHeads(self.k.ptr, rope, self.t_gpos.ptr, R, 1, c.idx_dim, false, stream);
                    try self.fp4Store(self.k.ptr, c.idx_dim, src.idxk_codes.ptr, src.idxk_scales.ptr, self.t_ctarget.ptr, R, 32, false, stream);
                }
                try self.lat2.copyFrom(0, self.lat.ptr, R * hd * 2, stream.handle);
                try self.ropeHeads(self.lat2.ptr, rope, self.t_gpos.ptr, R, 1, hd, false, stream);
                try self.fp4Store(self.lat2.ptr, hd, src.comp_codes.ptr, src.comp_scales.ptr, self.t_ctarget.ptr, R, 16, true, stream);
            }
            // -- the indexer: every visible compressed entry (short buckets: no scores) -------------------------
            const nb_ = self.bucket / ratio;
            kk = @min(c.idx_topk, nb_);
            if (c.isIndexSource(li)) {
                if (nb_ > c.idx_topk) return error.Unsupported; // the scored top-k with bases: not here yet
                for (0..R) |i| for (0..kk) |j| {
                    self.cidx_h[i * kk + j] = if (@as(i64, @intCast(j)) < vis[i]) @intCast(j) else -1;
                };
                try self.uploads.put(self.t_cidx, 0, std.mem.sliceAsBytes(self.cidx_h[0 .. R * kk]), stream);
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
                if (has) aot.ptr("CSC", "*u8", comp_scales) else aot.ptr("CSC", "*bf16", ring.ptr), aot.ptr("IDX", "*i64", if (has) self.t_cidx.ptr else self.t_pos.ptr),
                aot.ptr("POS", "*i64", self.t_pos.ptr),  aot.ptr("PM", "*fp32", self.pm.ptr),
                aot.ptr("PL", "*fp32", self.pl.ptr),     aot.ptr("PO", "*fp32", self.po.ptr),
                aot.ptr("SINK", "*fp32", l.sink.ptr()),  aot.ptr("OUT", "*bf16", self.o.ptr),
                aot.float("scale", scale),               aot.int("ring_size", @intCast(self.pool.ring)),
                aot.int("n_idx", @intCast(kk)),          aot.ptr("RBASE", "*i64", self.t_wbase.ptr),
                aot.ptr("CBASE", "*i64", if (has) self.t_cbase.ptr else self.t_wbase.ptr),
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
        try self.ropeHeads(self.o.ptr, rope, self.t_pos.ptr, R, heads, hd, true, stream);
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
