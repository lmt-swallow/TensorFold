//! Flash Next's decode forward for one stream on one GPU (TensorFold 0.6.6 `forward.py: stage, compute, commit` on an
//! EXL3 pack): the n-gram rows staged on the host, the embedding, every decoder layer (the n-gram branch where the
//! layer has one, the attention hyper-connection, DeltaNet or sparse attention, the MLP hyper-connection, the routed
//! experts, whose write-back the next layer applies), the mixer and the head; `commit` keeps a window's first rows.
//! Every kernel treats each row alone, so a window row has the bits of the same token decoded alone.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const core = @import("core");
const Config = @import("config.zig").Config;
const names = @import("weights_map.zig");
const weights = @import("cuda_weights.zig");
const glue_mod = @import("cuda_glue.zig");
const hc_mod = @import("cuda_hc.zig");
const gdn = @import("cuda_gdn.zig");
const experts = @import("cuda_experts.zig");
const attn = @import("cuda_attn.zig");
const ngram = @import("ngram.zig");

const Glue = glue_mod.Glue;
const Ptr = cuda.abi.DevicePtr;
const Linear = dsv41.exl3.Linear;

fn linear(gpa: std.mem.Allocator, k: *const dsv41.exl3.Kernels, g: weights.Group) !Linear {
    // a trellis mapped from its shard may sit at any even offset; the linear reads 32-bit words
    const words = try gpa.alignedAlloc(u8, .@"4", g.trellis.len);
    defer gpa.free(words);
    @memcpy(words, g.trellis);
    return Linear.init(gpa, k, words, g.kt, g.nt, g.w16, g.suh, g.svh, @enumFromInt(@intFromEnum(g.codebook)));
}

pub const GdnLayer = struct {
    w: weights.Gdn,
    qkv: Linear,
    z: Linear,
    out: Linear,
    li: usize, // index among the DeltaNet layers

    fn deinit(self: *GdnLayer) void {
        self.w.deinit();
        self.qkv.deinit();
        self.z.deinit();
        self.out.deinit();
    }
};

pub const AttnLayer = struct {
    w: weights.Attn,
    parts: [4]Linear, // q (with its gate), k, v, index qk: side by side in the projection row
    o: Linear,
    ai: usize, // index among the attention layers

    fn deinit(self: *AttnLayer) void {
        self.w.deinit();
        for (&self.parts) |*p| p.deinit();
        self.o.deinit();
    }
};

pub const Layer = struct {
    index: usize,
    attn_hc: weights.Hc,
    mlp_hc: weights.Hc,
    mixer: union(enum) { gdn: GdnLayer, attn: AttnLayer },
    router: cuda.DeviceBuffer,
    experts: experts.Experts,
    ple: ?weights.Ple = null,

    pub fn deinit(self: *Layer) void {
        self.attn_hc.deinit();
        self.mlp_hc.deinit();
        switch (self.mixer) {
            .gdn => |*g| g.deinit(),
            .attn => |*a| a.deinit(),
        }
        self.router.free();
        self.experts.deinit();
        if (self.ple) |*p| p.deinit();
    }
};

/// The model's weights on the GPU: the first `layers` decoder layers (all of them for logits), the mixer and head.
pub const Model = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    cfg: Config,
    pack: weights.Pack,
    table: ?weights.NgramTable,
    ng: ?ngram.NGram,
    xk: dsv41.exl3.Kernels,
    gk: gdn.Kernels,
    ek: experts.Kernels,
    embed: cuda.DeviceBuffer,
    inv: cuda.DeviceBuffer, // fp32 rotary inverse frequencies
    layers: []Layer,
    mixer: weights.Hc,
    head: ?Linear,
    offset: f32,
    has_mtp: bool = false, // the streams keep an MTP cache (cuda_mtp.zig loads the head)

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, cfg: Config, layers: usize, with_head: bool) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.io = io;
        m.cfg = cfg;
        m.pack = try weights.Pack.open(gpa, io, dir);
        errdefer m.pack.close();
        m.offset = try weights.centredOffset(gpa, &m.pack, &m.cfg);
        m.xk = try dsv41.exl3.Kernels.load(d);
        errdefer m.xk.deinit();
        m.gk = try gdn.Kernels.load(d, cfg.nv);
        errdefer m.gk.deinit();
        m.ek = try experts.Kernels.load(d);
        errdefer m.ek.deinit();
        m.embed = try weights.embedding(d, &m.pack, &m.cfg);
        errdefer m.embed.free();
        var inv: [64]f32 = undefined;
        weights.invFreq(inv[0 .. cfg.rotary_dim / 2], cfg.rope_theta);
        m.inv = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(inv[0 .. cfg.rotary_dim / 2]));
        errdefer m.inv.free();
        m.table = null;
        m.ng = null;
        if (cfg.ple_layer) |pl| if (pl < layers) {
            m.table = try weights.NgramTable.open(gpa, io, dir, pl, cfg.ngram_shards);
            const g = try ngram.derive(cfg.vocab, cfg.ngram, cfg.heads_per_ngram, cfg.ngram_base, cfg.seed, cfg.ple_eos, 0);
            try m.table.?.check(&g);
            m.ng = g;
        };
        errdefer if (m.table) |*t| t.close(io);
        m.layers = try gpa.alloc(Layer, layers);
        var made: usize = 0;
        errdefer {
            for (m.layers[0..made]) |*l| l.deinit();
            gpa.free(m.layers);
        }
        var lin: usize = 0;
        var att: usize = 0;
        var n: names.Name = .{};
        for (0..layers) |i| {
            var l: Layer = undefined;
            l.index = i;
            l.attn_hc = try weights.Hc.load(gpa, d, &m.pack, &m.cfg, n.of(names.lm ++ "layers.{d}.attn_hyper_connection", .{i}), true, m.offset);
            l.mlp_hc = try weights.Hc.load(gpa, d, &m.pack, &m.cfg, n.of(names.lm ++ "layers.{d}.mlp_hyper_connection", .{i}), true, m.offset);
            if (cfg.kinds[i] == .linear) {
                const w = try weights.Gdn.load(gpa, d, &m.pack, &m.cfg, i);
                l.mixer = .{ .gdn = .{ .w = w, .qkv = try linear(gpa, &m.xk, w.qkv), .z = try linear(gpa, &m.xk, w.z), .out = try linear(gpa, &m.xk, w.out), .li = lin } };
                lin += 1;
            } else {
                const w = try weights.Attn.load(gpa, d, &m.pack, .{ .lm = i }, m.offset);
                l.mixer = .{ .attn = .{
                    .w = w,
                    .parts = .{ try linear(gpa, &m.xk, w.q), try linear(gpa, &m.xk, w.k), try linear(gpa, &m.xk, w.v), try linear(gpa, &m.xk, w.iqk) },
                    .o = try linear(gpa, &m.xk, w.o),
                    .ai = att,
                } };
                att += 1;
            }
            l.router = try weights.router(gpa, d, &m.pack, &m.cfg, .{ .lm = i });
            l.experts = try experts.Experts.load(gpa, d, &m.pack, &m.cfg, .{ .lm = i });
            l.ple = if (cfg.ple_layer != null and cfg.ple_layer.? == i) try weights.Ple.load(gpa, d, &m.pack, &m.cfg, &m.table.?, i, m.offset) else null;
            m.layers[i] = l;
            made += 1;
        }
        m.mixer = try weights.Hc.load(gpa, d, &m.pack, &m.cfg, names.mixer, false, m.offset);
        errdefer m.mixer.deinit();
        m.head = if (with_head) try linear(gpa, &m.xk, try m.pack.group(names.head)) else null;
        return m;
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*l| l.deinit();
        self.gpa.free(self.layers);
        self.mixer.deinit();
        if (self.head) |*h| h.deinit();
        if (self.table) |*t| t.close(self.io);
        self.embed.free();
        self.inv.free();
        self.ek.deinit();
        self.gk.deinit();
        self.xk.deinit();
        self.pack.close();
        self.gpa.destroy(self);
    }

    pub fn linearLayers(self: *const Model) usize {
        var n: usize = 0;
        for (self.layers) |l| n += @intFromBool(l.mixer == .gdn);
        return n;
    }

    pub fn attnLayers(self: *const Model) usize {
        return self.layers.len - self.linearLayers();
    }
};

/// One stream's committed state (state.py `State`): position, DeltaNet states and conv windows, attention caches, the
/// n-gram history and conv tail.
pub const Stream = struct {
    pos: usize = 0,
    pos_dev: cuda.DeviceBuffer, // int32 [1]
    conv: cuda.DeviceBuffer, // bf16 [lin, taps - 1, conv_dim]
    rec: cuda.DeviceBuffer, // fp32 [2, lin, nv, dv, dk]
    cur: []u1,
    scratch: []gdn.Scratch,
    caches: []attn.Caches,
    ple_tail: cuda.DeviceBuffer, // bf16 [(ple_kernel - 1) * ngram, S * D]
    // the last decode window's rows a commit reads (the stream's own, so another stream's window may run between)
    rows: usize,
    proj: cuda.DeviceBuffer, // bf16 [lin, rows, PW]: each DeltaNet layer's projection rows
    ple_nrow: cuda.DeviceBuffer, // bf16 [rows, S * D]: the n-gram branch's normed rows
    // the MTP head's own attention cache and its length (mtp.py: st.mtp_kc, mtp_pos, mtp_len, mtp_drafted)
    mtp_caches: ?attn.Caches = null,
    mtp_pos_dev: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0 },
    mtp_len: usize = 0,
    mtp_drafted: usize = 0,
    history: [ngram.max_n]i64 = @splat(0),
    prompt_history: [ngram.max_n]i64 = @splat(0), // the history after the prompt chunk being run
    last: ?struct { history: [ngram.max_n]i64, tokens: [64]i64, n: usize } = null,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *const Model, capacity: usize, rows: usize) !Stream {
        const cfg = &m.cfg;
        const lin = m.linearLayers();
        var s: Stream = .{ .gpa = gpa, .pos_dev = undefined, .conv = undefined, .rec = undefined, .cur = undefined, .scratch = undefined, .caches = undefined, .ple_tail = undefined, .rows = rows, .proj = undefined, .ple_nrow = undefined };
        s.proj = try cuda.DeviceBuffer.alloc(d, @max(16, lin * rows * cfg.projWidth() * 2));
        try s.proj.fill8(0, null);
        s.ple_nrow = try cuda.DeviceBuffer.alloc(d, rows * cfg.wide() * 2);
        try s.ple_nrow.fill8(0, null);
        s.pos_dev = try cuda.DeviceBuffer.alloc(d, 16);
        try s.pos_dev.fill8(0, null);
        s.conv = try cuda.DeviceBuffer.alloc(d, @max(16, lin * (cfg.conv_kernel - 1) * cfg.convDim() * 2));
        try s.conv.fill8(0, null);
        s.rec = try cuda.DeviceBuffer.alloc(d, @max(16, 2 * lin * cfg.nv * cfg.dv * cfg.dk * 4));
        try s.rec.fill8(0, null);
        s.cur = try gpa.alloc(u1, lin);
        @memset(s.cur, 0);
        s.scratch = try gpa.alloc(gdn.Scratch, lin);
        for (s.scratch) |*x| x.* = try gdn.Scratch.init(d, rows, cfg.nk, cfg.nv);
        s.caches = try gpa.alloc(attn.Caches, m.attnLayers());
        for (s.caches) |*c| c.* = try attn.Caches.init(d, cfg, capacity);
        s.ple_tail = try cuda.DeviceBuffer.alloc(d, (cfg.ple_kernel - 1) * cfg.ngram * cfg.wide() * 2);
        try s.ple_tail.fill8(0, null);
        if (m.ng) |*g| g.initialHistory(s.history[0 .. g.n - 1]);
        if (m.has_mtp) {
            s.mtp_caches = try attn.Caches.init(d, cfg, capacity);
            s.mtp_pos_dev = try cuda.DeviceBuffer.alloc(d, 16);
            try s.mtp_pos_dev.fill8(0, null);
        }
        return s;
    }

    pub fn deinit(self: *Stream) void {
        if (self.mtp_caches) |*c| {
            c.deinit();
            self.mtp_pos_dev.free();
        }
        self.proj.free();
        self.ple_nrow.free();
        self.pos_dev.free();
        self.conv.free();
        self.rec.free();
        for (self.scratch) |*x| x.deinit();
        for (self.caches) |*c| c.deinit();
        self.ple_tail.free();
        self.gpa.free(self.cur);
        self.gpa.free(self.scratch);
        self.gpa.free(self.caches);
    }

    fn stateBytes(cfg: *const Config) usize {
        return cfg.nv * cfg.dv * cfg.dk * 4;
    }

    /// The device address of DeltaNet layer li's state slot `which` (0 or 1).
    pub fn recPtr(self: *const Stream, cfg: *const Config, lin: usize, li: usize, which: u1) Ptr {
        return self.rec.ptr + (@as(usize, which) * lin + li) * stateBytes(cfg);
    }

    /// state.py `set_mtp_len`: the MTP head's cache length, on the host and the device.
    pub fn setMtpLen(self: *Stream, n: usize, stream: cuda.Stream) !void {
        self.mtp_len = n;
        const v: i32 = @intCast(n);
        try self.mtp_pos_dev.uploadAsync(0, std.mem.asBytes(&v), stream.handle);
    }

    /// The main layers' attention: the stream's caches at its committed position.
    pub fn mainAt(self: *const Stream, ai: usize) AttnAt {
        return .{ .caches = &self.caches[ai], .pos_dev = self.pos_dev.ptr, .pos = self.pos };
    }

    /// The MTP head's attention: its own cache at its own length.
    pub fn mtpAt(self: *const Stream) AttnAt {
        return .{ .caches = &self.mtp_caches.?, .pos_dev = self.mtp_pos_dev.ptr, .pos = self.mtp_len };
    }

    pub fn convPtr(self: *const Stream, cfg: *const Config, li: usize) Ptr {
        return self.conv.ptr + li * (cfg.conv_kernel - 1) * cfg.convDim() * 2;
    }
};

/// Where an attention layer reads and writes: its caches and the stream's position for it (forward.py `_caches`).
pub const AttnAt = struct { caches: *const attn.Caches, pos_dev: Ptr, pos: usize };

/// One stream's rows in a shared window (forward.py `stage`'s segments): buffer rows a0 .. a0 + n. A shared window's
/// segments follow each other from row 0; every per-stream step (n-gram conv, DeltaNet, attention, commit) runs on
/// its segment alone and every other step on all rows, so each row has the bits of its stream's own window.
pub const Seg = struct { st: *Stream, a0: usize, n: usize };

/// Which caches a layer's attention uses: the main model's at the stream's position, or the MTP head's own.
pub const Cache = enum {
    main,
    mtp,

    fn at(c: Cache, st: *const Stream, ai: usize) AttnAt {
        return switch (c) {
            .main => st.mainAt(ai),
            .mtp => st.mtpAt(),
        };
    }
};

/// The rows the segments cover, checked to follow each other from row 0.
pub fn segRows(segs: []const Seg) !usize {
    var r: usize = 0;
    for (segs) |sg| {
        if (sg.a0 != r or sg.n == 0 or sg.n > sg.st.rows) return error.BadSegments;
        r += sg.n;
    }
    return r;
}

/// A decode window's buffers (state.py `Buffers`, decode).
pub const Buffers = struct {
    rows: usize,
    w: hc_mod.Window,
    ids: cuda.DeviceBuffer, // int32 [rows]
    pa: cuda.DeviceBuffer, // bf16 [rows, attention projection width]
    q: cuda.DeviceBuffer, // bf16 [rows, heads, head_dim]
    iq: cuda.DeviceBuffer, // bf16 [rows, index_heads, index_dim]
    gated: cuda.DeviceBuffer, // bf16 [rows, heads * head_dim]
    xs_gated: cuda.DeviceBuffer,
    attn: attn.Scratch,
    xh: cuda.DeviceBuffer, // fp16 scratch of the EXL3 linear
    z: cuda.DeviceBuffer, // fp32 split-K scratch of the EXL3 linear
    logits_router: cuda.DeviceBuffer, // fp32 [rows, experts + 1]
    pick: cuda.DeviceBuffer, // int32 [rows, slots]
    wts: cuda.DeviceBuffer, // fp32 [rows, slots]
    moe: experts.Scratch,
    ple_dev: cuda.DeviceBuffer, // int16 [rows * heads, words]
    ple_host: []u8,
    ple_emb: cuda.DeviceBuffer, // fp16 [rows, ple_dim]
    ple_keys: cuda.DeviceBuffer, // bf16 [rows, S * D]
    ple_vals: cuda.DeviceBuffer, // bf16 [rows, D]
    ple_gated: cuda.DeviceBuffer,
    ple_pss: cuda.DeviceBuffer, // fp32 [rows, S]
    streams: cuda.DeviceBuffer, // bf16 [rows, S * D]: the head's copy of the streams
    logits: cuda.DeviceBuffer, // bf16 [rows, vocab]
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *const Model, rows: usize, capacity: usize) !Buffers {
        const cfg = &m.cfg;
        var b: Buffers = undefined;
        b.gpa = gpa;
        b.rows = rows;
        const wide = cfg.wide();
        // the widest fp16 K-slice scratch any F16 matrix of the model needs
        var part: usize = 0;
        for ([_]usize{ cfg.low + cfg.streams, cfg.low }, [_]usize{ wide, wide }) |nn, k| part = @max(part, weights.f16Split(nn, k) * rows * nn);
        part = @max(part, weights.f16Split(2 * cfg.nv, cfg.hidden) * rows * 2 * cfg.nv);
        part = @max(part, weights.f16Split(cfg.hidden, cfg.hidden) * rows * cfg.hidden);
        part = @max(part, weights.f16Split(wide, cfg.hidden) * rows * wide);
        b.w = try hc_mod.Window.init(d, cfg, rows, part);
        const aw = cfg.heads * 2 * cfg.head_dim + 2 * cfg.kv_heads * cfg.head_dim + (cfg.index_heads + 1) * cfg.index_dim;
        var zmax: usize = 16;
        var kmax: usize = cfg.hidden;
        for (m.layers) |*l| switch (l.mixer) {
            .gdn => |*g| for ([_]*const Linear{ &g.qkv, &g.z, &g.out }) |x| {
                zmax = @max(zmax, x.scratchWords(rows) * 4);
                kmax = @max(kmax, x.k);
            },
            .attn => |*a| for ([_]*const Linear{ &a.parts[0], &a.parts[1], &a.parts[2], &a.parts[3], &a.o }) |x| {
                zmax = @max(zmax, x.scratchWords(rows) * 4);
                kmax = @max(kmax, x.k);
            },
        };
        if (m.head) |*h| zmax = @max(zmax, h.scratchWords(rows) * 4);
        const slots = cfg.slots();
        const ple_words: usize = if (m.table) |t| t.words else 1;
        const ple_heads: usize = if (m.ng) |g| g.heads() else 1;
        const sizes = [_]struct { *cuda.DeviceBuffer, usize }{
            .{ &b.ids, rows * 4 },
            .{ &b.pa, rows * aw * 2 },                     .{ &b.q, rows * cfg.heads * cfg.head_dim * 2 },
            .{ &b.iq, rows * cfg.index_heads * cfg.index_dim * 2 }, .{ &b.gated, rows * cfg.heads * cfg.head_dim * 2 },
            .{ &b.xs_gated, rows * cfg.heads * cfg.head_dim / 32 * 4 }, .{ &b.xh, rows * @max(kmax, cfg.wide()) * 2 },
            .{ &b.z, @max(zmax, 8 * rows * cfg.wide() * 4) }, // the MTP head's fc_hidden: S rows a row
            .{ &b.logits_router, rows * (cfg.experts + 1) * 4 },
            .{ &b.pick, rows * slots * 4 },                .{ &b.wts, rows * slots * 4 },
            .{ &b.ple_dev, rows * ple_heads * ple_words * 2 }, .{ &b.ple_emb, rows * cfg.hidden * 2 },
            .{ &b.ple_keys, rows * wide * 2 },             .{ &b.ple_vals, rows * cfg.hidden * 2 },
            .{ &b.ple_gated, rows * wide * 2 },            .{ &b.ple_pss, rows * cfg.streams * 4 },
            .{ &b.streams, rows * wide * 2 },
            .{ &b.logits, rows * cfg.vocab * 2 },
        };
        for (sizes) |s| {
            s[0].* = try cuda.DeviceBuffer.alloc(d, s[1]);
            try s[0].fill8(0, null);
        }
        b.attn = try attn.Scratch.init(d, cfg, rows, capacity);
        b.moe = try experts.Scratch.init(d, &m.layers[0].experts, rows, slots);
        b.ple_host = try gpa.alloc(u8, rows * ple_heads * ple_words * 2);
        return b;
    }

    pub fn deinit(self: *Buffers) void {
        self.w.deinit();
        for ([_]*cuda.DeviceBuffer{ &self.ids, &self.pa, &self.q, &self.iq, &self.gated, &self.xs_gated, &self.xh, &self.z, &self.logits_router, &self.pick, &self.wts, &self.ple_dev, &self.ple_emb, &self.ple_keys, &self.ple_vals, &self.ple_gated, &self.ple_pss, &self.streams, &self.logits }) |x| x.free();
        self.attn.deinit();
        self.moe.deinit();
        self.gpa.free(self.ple_host);
    }
};

/// Hooks a test uses to compare intermediate values (all optional).
pub const Probe = struct {
    ctx: ?*anyopaque = null,
    /// before layer `l` (its pending write-back not yet applied): the streams at b.w.h
    layer_in: ?*const fn (ctx: ?*anyopaque, l: usize, b: *Buffers, rows: usize) anyerror!void = null,
    /// after layer `l`: its pending expert slots (b.moe.y), weights (b.wts) and gates (b.w.inj_m)
    layer_out: ?*const fn (ctx: ?*anyopaque, l: usize, b: *Buffers, rows: usize) anyerror!void = null,
};

pub const Forward = struct {
    m: *Model,
    g: Glue,
    ak: attn.Kernels,
    d: *const cuda.Driver,

    pub fn init(d: *const cuda.Driver, m: *Model, set: *const cuda.aot.Set) Forward {
        return .{ .m = m, .d = d, .g = .{ .set = set, .hidden = m.cfg.hidden, .streams = m.cfg.streams, .low = m.cfg.low }, .ak = .{ .set = set, .cfg = &m.cfg } };
    }

    /// forward.py `stage` for one stream.
    pub fn stage(f: *Forward, st: *Stream, b: *Buffers, tokens: []const i32, stream: cuda.Stream) !void {
        try f.stageSegs(&.{.{ .st = st, .a0 = 0, .n = tokens.len }}, b, tokens, stream);
    }

    /// forward.py `stage`: the token ids of every segment's rows (`tokens`, in row order) and each stream's n-gram
    /// rows from its own history, copied to the device.
    pub fn stageSegs(f: *Forward, segs: []const Seg, b: *Buffers, tokens: []const i32, stream: cuda.Stream) !void {
        const R = try segRows(segs);
        if (R != tokens.len) return error.BadSegments;
        if (R > b.rows) return error.WindowTooLarge;
        try b.ids.upload(0, std.mem.sliceAsBytes(tokens));
        if (f.m.ng) |*g| {
            const heads = g.heads();
            const t = &f.m.table.?;
            const row_bytes = heads * t.words * 2;
            for (segs) |sg| {
                if (sg.n > 64) return error.WindowTooLarge;
                var toks: [64]i64 = undefined;
                for (tokens[sg.a0..][0..sg.n], 0..) |x, i| toks[i] = x;
                var row_ids: [64 * ngram.max_heads]i64 = undefined;
                try g.ids(sg.st.history[0 .. g.n - 1], toks[0..sg.n], row_ids[0 .. sg.n * heads]);
                try t.gather(row_ids[0 .. sg.n * heads], b.ple_host[sg.a0 * row_bytes ..][0 .. sg.n * row_bytes]);
                sg.st.last = .{ .history = sg.st.history, .tokens = toks, .n = sg.n };
            }
            try b.ple_dev.uploadAsync(0, b.ple_host[0 .. R * row_bytes], stream.handle);
        }
    }

    pub fn mm(f: *Forward, l: *const Linear, x: Ptr, R: usize, y: Ptr, b: *Buffers, stream: cuda.Stream) !void {
        _ = f;
        try l.run(x, .bf16, R, b.xh.ptr, b.z.ptr, y, .bf16, stream);
    }

    /// exl3_mm.py `Stack`: matrices side by side into rows of width `width` (one row straight into its columns, more
    /// rows through the scratch rows and a copy each).
    fn stack(f: *Forward, parts: []const *const Linear, f16part: ?*const weights.F16, x: Ptr, R: usize, out: Ptr, width: usize, b: *Buffers, stream: cuda.Stream) !void {
        var at: usize = 0;
        const direct = R == 1;
        for (parts) |p| {
            const dst = if (direct) out + at * 2 else b.w.tmp.ptr;
            try f.mm(p, x, R, dst, b, stream);
            if (!direct) try copyRows(b, out, width, at, p.n, R, stream);
            at += p.n;
        }
        if (f16part) |w| {
            const dst = if (direct) out + at * 2 else b.w.tmp.ptr;
            try f.g.f16Call(w, x, f.m.cfg.hidden, R, dst, if (direct) width else w.n, b.w.part.ptr, stream);
            if (!direct) try copyRows(b, out, width, at, w.n, R, stream);
            at += w.n;
        }
        if (at != width) return error.BadShape;
    }

    fn copyRows(b: *Buffers, out: Ptr, width: usize, at: usize, n: usize, R: usize, stream: cuda.Stream) !void {
        for (0..R) |r| {
            const dst: cuda.DeviceBuffer = .{ .d = b.w.tmp.d, .ptr = out + (r * width + at) * 2, .len = n * 2 };
            try dst.copyFrom(0, b.w.tmp.ptr + r * n * 2, n * 2, stream.handle);
        }
    }

    /// forward.py `compute` for one stream: logits of the window's rows into b.logits (with the head loaded).
    pub fn compute(f: *Forward, st: *Stream, b: *Buffers, R: usize, probe: Probe, stream: cuda.Stream) !void {
        try f.computeSegs(&.{.{ .st = st, .a0 = 0, .n = R }}, b, probe, stream);
    }

    /// forward.py `compute` over a shared window: logits of every segment's rows into b.logits.
    pub fn computeSegs(f: *Forward, segs: []const Seg, b: *Buffers, probe: Probe, stream: cuda.Stream) !void {
        const m = f.m;
        const g = f.g;
        const w = &b.w;
        const R = try segRows(segs);
        try g.embed(b.ids.ptr, m.embed.ptr, w.h.ptr, R, stream);
        var pending: Glue.Pending = .none;
        for (m.layers) |*layer| {
            if (probe.layer_in) |cb| {
                try stream.synchronize();
                try cb(probe.ctx, layer.index, b, R);
            }
            try f.layerStep(layer, segs, b, pending, .main, stream);
            if (probe.layer_out) |cb| {
                try stream.synchronize();
                try cb(probe.ctx, layer.index, b, R);
            }
            pending = f.moePending(b);
        }
        // finish: the last write-back on a copy of the streams, the mixer, the head
        const head = if (m.head) |*h| h else return;
        try f.finish(&m.mixer, b, R, pending, stream);
        try f.mm(head, w.mixed.ptr, R, b.logits.ptr, b, stream);
    }

    /// The pending MoE write-back a layer leaves in `b`.
    pub fn moePending(f: *Forward, b: *Buffers) Glue.Pending {
        return .{ .moe = .{ .y = b.moe.y.ptr, .wts = b.wts.ptr, .inject = b.w.inj_m.ptr, .slots = f.m.cfg.slots() } };
    }

    /// forward.py `finish` without the head: b.streams = the streams with the last write-back, and the read-out of
    /// `mixer` into b.w.mixed / xs_mixed.
    pub fn finish(f: *Forward, mixer: *const weights.Hc, b: *Buffers, R: usize, pending: Glue.Pending, stream: cuda.Stream) !void {
        const cfg = &f.m.cfg;
        try b.streams.copyFrom(0, b.w.h.ptr, R * cfg.wide() * 2, stream.handle);
        try f.g.hcWriteback(b.streams.ptr, b.w.pss.ptr, R, pending, stream);
        try hc_mod.readoutOf(f.g, &b.w, b.streams.ptr, mixer, cfg, R, 0, stream);
    }

    /// forward.py `layer_forward` up to its experts over a shared window: the n-gram branch, both hyper-connections,
    /// DeltaNet or attention (each segment's at its `cache`), router and experts; the layer's write-back is left
    /// pending in `b`.
    pub fn layerStep(f: *Forward, layer: *const Layer, segs: []const Seg, b: *Buffers, pending_in: Glue.Pending, cache: Cache, stream: cuda.Stream) !void {
        const m = f.m;
        const cfg = &m.cfg;
        const g = f.g;
        const w = &b.w;
        const lin = m.linearLayers();
        const R = try segRows(segs);
        var pending = pending_in;
        if (layer.ple) |*p| {
            if (pending != .none) {
                try g.hcWriteback(w.h.ptr, w.pss.ptr, R, pending, stream);
                pending = .none;
            }
            try f.pleBlock(p, segs, b, R, stream);
        }
        // the attention hyper-connection
        try hc_mod.block(g, w, &layer.attn_hc, cfg, R, pending, w.inj_a.ptr, stream);
        switch (layer.mixer) {
            .gdn => |*gl| {
                const pw = cfg.projWidth();
                const vw = cfg.nv * cfg.dv;
                // one stream: its projection rows straight into its own; more: all rows, then each stream's copied
                const one = segs.len == 1;
                const at = if (one) segs[0].st.proj.ptr + gl.li * segs[0].st.rows * pw * 2 else w.proj.ptr;
                try f.stack(&.{ &gl.qkv, &gl.z }, &gl.w.ba, w.mixed.ptr, R, at, pw, b, stream);
                for (segs) |sg| {
                    const st = sg.st;
                    const proj: cuda.DeviceBuffer = .{ .d = st.proj.d, .ptr = st.proj.ptr + gl.li * st.rows * pw * 2, .len = st.rows * pw * 2 };
                    if (!one) try proj.copyFrom(0, w.proj.ptr + sg.a0 * pw * 2, sg.n * pw * 2, stream.handle);
                    const cur = st.cur[gl.li];
                    try gdn.chain(&m.gk, .{
                        .p = proj.ptr,                     .conv_state = st.convPtr(cfg, gl.li), .conv_w = gl.w.conv.ptr,
                        .state_in = st.recPtr(cfg, lin, gl.li, cur), .a_log = gl.w.a_log.ptr, .dt_bias = gl.w.dt_bias.ptr,
                        .norm_w = gl.w.norm.ptr,           .eps = cfg.eps,                         .rows = sg.n,
                        .out = w.gout.ptr + sg.a0 * vw * 2, .xs = w.gxs.ptr + sg.a0 * vw / 32 * 4, .state_out = st.recPtr(cfg, lin, gl.li, 1 - cur),
                        .scratch = &st.scratch[gl.li],
                    }, stream);
                }
                try f.mm(&gl.out, w.gout.ptr, R, w.branch.ptr, b, stream);
            },
            .attn => |*al| {
                const pw = al.w.projWidth();
                const qw = cfg.heads * cfg.head_dim;
                const iw = cfg.index_heads * cfg.index_dim;
                try f.stack(&.{ &al.parts[0], &al.parts[1], &al.parts[2], &al.parts[3] }, null, w.mixed.ptr, R, b.pa.ptr, pw, b, stream);
                for (segs) |sg| {
                    const a = cache.at(sg.st, al.ai);
                    try f.ak.block(b.pa.ptr + sg.a0 * pw * 2, pw, a.pos_dev, a.pos, .{ al.w.q_scale.ptr, al.w.k_scale.ptr, al.w.iq_scale.ptr, al.w.ik_scale.ptr }, m.inv.ptr, b.q.ptr + sg.a0 * qw * 2, b.iq.ptr + sg.a0 * iw * 2, a.caches, &b.attn, b.gated.ptr + sg.a0 * qw * 2, b.xs_gated.ptr + sg.a0 * qw / 32 * 4, sg.n, stream);
                }
                try f.mm(&al.o, b.gated.ptr, R, w.branch.ptr, b, stream);
            },
        }
        // the MLP hyper-connection, then the experts
        try hc_mod.block(g, w, &layer.mlp_hc, cfg, R, .{ .branch = .{ .branch = w.branch.ptr, .inject = w.inj_a.ptr } }, w.inj_m.ptr, stream);
        try g.router(w.mixed.ptr, cfg.hidden, layer.router.ptr, cfg.experts + 1, b.logits_router.ptr, R, stream);
        try g.selectRows(b.logits_router.ptr, cfg.experts, cfg.topk, b.pick.ptr, b.wts.ptr, R, stream);
        _ = try experts.routed(&m.ek, &layer.experts, &b.moe, w.mixed.ptr, cfg.hidden, b.pick.ptr, R, stream);
    }

    /// forward.py `ple_block`: the table rows, keys, values and gate on all rows, each stream's conv on its own.
    fn pleBlock(f: *Forward, p: *const weights.Ple, segs: []const Seg, b: *Buffers, R: usize, stream: cuda.Stream) !void {
        const m = f.m;
        const cfg = &m.cfg;
        const g = f.g;
        const t = &m.table.?;
        const ng = m.ng.?;
        const wide = cfg.wide();
        try g.pleRows(b.ple_dev.ptr, t.words, p.head_bias.ptr, ng.heads(), t.dh, t.bits, b.ple_emb.ptr, R, stream);
        try g.f16CallOf(&p.key, b.ple_emb.ptr, "*fp16", cfg.hidden, R, b.ple_keys.ptr, wide, b.w.part.ptr, stream);
        try g.f16CallOf(&p.value, b.ple_emb.ptr, "*fp16", cfg.hidden, R, b.ple_vals.ptr, cfg.hidden, b.w.part.ptr, stream);
        try g.pleGate(b.ple_keys.ptr, b.ple_vals.ptr, b.w.h.ptr, p.norm_key.ptr, p.norm_query.ptr, b.ple_gated.ptr, b.ple_pss.ptr, cfg.eps, R, stream);
        for (segs) |sg| try g.pleConv(b.ple_gated.ptr + sg.a0 * wide * 2, b.ple_pss.ptr + sg.a0 * cfg.streams * 4, p.norm_conv.ptr, sg.st.ple_tail.ptr, p.conv.ptr, p.taps, cfg.ngram, b.w.h.ptr + sg.a0 * wide * 2, sg.st.ple_nrow.ptr, cfg.eps, sg.n, stream);
    }

    /// forward.py `commit(keep)`: the first `keep` of the window's R rows become the stream's (a rejected tail
    /// replays each DeltaNet state; conv windows and the n-gram tail shift; the position advances).
    pub fn commit(f: *Forward, st: *Stream, R: usize, keep: usize, stream: cuda.Stream) !void {
        const m = f.m;
        const cfg = &m.cfg;
        if (keep < 1 or keep > R) return error.BadKeep;
        const lin = m.linearLayers();
        for (0..lin) |li| {
            const cur = st.cur[li];
            if (keep < R) try gdn.replay(&m.gk, st.recPtr(cfg, lin, li, cur), &st.scratch[li], keep, st.recPtr(cfg, lin, li, 1 - cur), stream);
            st.cur[li] = 1 - cur;
        }
        const taps = cfg.conv_kernel - 1;
        if (lin > 0) try f.g.shiftWindows(st.conv.ptr, taps * cfg.convDim(), st.proj.ptr, st.rows * cfg.projWidth(), cfg.projWidth(), keep, lin, cfg.convDim(), taps, stream);
        if (st.last) |last| {
            const ng = m.ng.?;
            var hist: [ngram.max_n]i64 = last.history;
            ng.advance(hist[0 .. ng.n - 1], last.tokens[0..keep]);
            st.history = hist;
            st.last = null;
            const t_rows = (cfg.ple_kernel - 1) * cfg.ngram;
            try f.g.shiftWindows(st.ple_tail.ptr, t_rows * cfg.wide(), st.ple_nrow.ptr, st.rows * cfg.wide(), cfg.wide(), keep, 1, cfg.wide(), t_rows, stream);
        }
        st.pos += keep;
        const p: i32 = @intCast(st.pos);
        try st.pos_dev.uploadAsync(0, std.mem.asBytes(&p), stream.handle);
    }
};
