//! The layer loop of one DeepSeek-V4.1 sequence at decode sizes (1 to 16 rows): the Python fork's decode body
//! (graph.py `StaticDecoder._body`, model.py `_forward_k` for a window) on this rank: embed, then per layer mHC pre ->
//! attention -> gather -> mHC post, mHC pre -> MoE (gate, pick, routed experts) -> gather -> mHC post, then the head
//! (collapse_norm -> this rank's vocabulary columns -> gather -> the rows' logits). The gathers go through `Gather`:
//! the communicator's all-gather over the ranks, or a test's supply of the other rank's partials. No replay, no
//! vision, no DSpark taps; Engram runs on its layers when a hook is set (`engram`, cuda_engram_apply.zig), as the fork
//! skips it without tables.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const exl3 = @import("cuda_exl3.zig");
const experts = @import("cuda_experts.zig");
const hcmod = @import("cuda_hc.zig");
const routemod = @import("cuda_route.zig");
const attention = @import("cuda_attention.zig");
const mem_note = @import("mem_note.zig");

const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;

pub const Error = error{ TooManyRows, BadShape, OutOfMemory } || cuda.Error || exl3.Error || experts.Error;

/// gathered [world, words] f32 <- every rank's `partial` [words], in rank order, on `stream` (the fork's comm.gather).
pub const Gather = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void,

    pub fn run(self: Gather, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) !void {
        return self.call(self.ctx, partial, words, out, stream);
    }
};

/// A linear at the caller's row count: y [m, N] (fp32) = x [m, K] bf16 @ W (model.py `mm`; the prompt path's GEMM
/// past 128 rows).
pub const Mm = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, lin: *const exl3.Linear, x: Ptr, m: usize, y: Ptr, stream: cuda.Stream) anyerror!void,
};

/// Engram on an Engram layer (model.py `engram_apply`): h [rows, 4, d] bf16 at positions start .. start + rows - 1,
/// gated in place by the rows the tables hold for the sequence's n-grams; one gather over the ranks (every rank calls
/// it at the same layers). `mm` null: the row-invariant decode linear. null hook: no tables (the fork skips the gate).
pub const Engram = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, layer: usize, h: Ptr, start: usize, rows: usize, mm: ?Mm, gather: Gather, stream: cuda.Stream) anyerror!void,
};

/// A probe for tests: called after each sublayer with the buffer's name, its device address and bytes.
pub const Probe = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) anyerror!void,
};

/// The compressed planes of one kv-source layer (one sequence): packed FP4, plus the compressor's raw inputs.
pub const Planes = struct {
    comp_codes: cuda.DeviceBuffer,
    comp_scales: cuda.DeviceBuffer,
    idxk_codes: cuda.DeviceBuffer,
    idxk_scales: cuda.DeviceBuffer,
    raw_kv: cuda.DeviceBuffer,
    raw_score: cuda.DeviceBuffer,
    rows: usize,

    fn init(d: *const cuda.Driver, cfg: *const Config, li: usize, cap: usize) !Planes {
        const ratio = cfg.ratio(li);
        const rows = cap / ratio + 2;
        const hd = cfg.head_dim;
        var p: Planes = undefined;
        p.rows = rows;
        p.comp_codes = try cuda.DeviceBuffer.alloc(d, rows * hd / 2);
        errdefer p.comp_codes.free();
        p.comp_scales = try cuda.DeviceBuffer.alloc(d, rows * hd / 16);
        errdefer p.comp_scales.free();
        const indexed = cfg.isIndexSource(li);
        p.idxk_codes = try cuda.DeviceBuffer.alloc(d, if (indexed) rows * cfg.idx_dim / 2 else 0);
        errdefer p.idxk_codes.free();
        p.idxk_scales = try cuda.DeviceBuffer.alloc(d, if (indexed) rows * cfg.idx_dim / 32 else 0);
        errdefer p.idxk_scales.free();
        const raw: usize = if (ratio > 1) 64 * hd * 4 else 0;
        p.raw_kv = try cuda.DeviceBuffer.alloc(d, raw);
        errdefer p.raw_kv.free();
        p.raw_score = try cuda.DeviceBuffer.alloc(d, raw);
        errdefer p.raw_score.free();
        try p.comp_codes.fill8(0, null);
        try p.comp_scales.fill8(0, null);
        try p.idxk_codes.fill8(0, null);
        try p.idxk_scales.fill8(127, null); // E8M0 scales start at 2^0 (model.py new_cache)
        try p.raw_kv.fill8(0, null);
        try p.raw_score.fill8(0, null);
        return p;
    }

    fn deinit(self: *Planes) void {
        inline for (.{ "comp_codes", "comp_scales", "idxk_codes", "idxk_scales", "raw_kv", "raw_score" }) |f| @field(self, f).free();
    }
};

pub const Forward = struct {
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
    max_rows: usize,
    cap: usize,
    bucket: usize,
    rope_c: attention.Rope, // the compressed layers' (YaRN)
    rope_w: attention.Rope, // the window layers'
    rr: attention.RopeRows, // a window's rope rows (the device path's: its positions and its groups' first ones)
    probe: ?Probe = null,
    engram: ?Engram = null,
    // caches
    rings: []cuda.DeviceBuffer, // per layer [ring, hd] bf16
    planes: []?Planes, // per layer: its own when a kv source
    source: []usize, // per layer: the kv source whose planes it reads
    attn: []attention.Attention,
    attn_scratch: *attention.Scratch, // what every layer's attention shares (the layers run one after another)
    idx_proj_h: []cuda.DeviceBuffer, // per layer: fp16 [idx_heads, d] on index layers (empty otherwise)
    // the window's buffers
    ids: cuda.DeviceBuffer,
    h: cuda.DeviceBuffer,
    pre: cuda.DeviceBuffer,
    pre_a: cuda.DeviceBuffer,
    pre_f: cuda.DeviceBuffer,
    post: cuda.DeviceBuffer,
    comb: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer,
    gathered: cuda.DeviceBuffer,
    parts: cuda.DeviceBuffer,
    pick: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    moe: cuda.DeviceBuffer,
    scratch: experts.Scratch,
    xc: cuda.DeviceBuffer,
    xh: cuda.DeviceBuffer, // the head's rotated input, fp16
    z: cuda.DeviceBuffer, // the head's split-K scratch
    local: cuda.DeviceBuffer,
    glog: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer,
    ids_h: []i64,
    uploads: attention.Uploads,

    /// `stream`: where the one-time setup launches go (the index layers' fp16 weights); synchronized before returning.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const Config, w: *const weights.Weights, set: *const aot.Set, glue: *const attention.Glue, ek: *const experts.Kernels, world: u32, max_rows: usize, cap: usize, bucket: usize, rope_c: attention.Rope, rope_w: attention.Rope, stream: cuda.Stream) !Forward {
        if (max_rows > 16) return error.TooManyRows;
        const L = w.layers.len;
        const dm = cfg.dim;
        const slots = cfg.topk + 1;
        var self: Forward = undefined;
        self.gpa = gpa;
        self.d = d;
        self.cfg = cfg;
        self.w = w;
        self.set = set;
        self.glue = glue;
        self.ek = ek;
        self.hc = hcmod.Hc.init(set, @intCast(dm), @intCast(cfg.hc_iters), cfg.eps, cfg.hc_eps);
        self.route = routemod.Route.init(set, @intCast(dm), @intCast(cfg.n_routed), @intCast(cfg.topk), cfg.route_scale, @intCast(w.layers[0].experts.count - 1));
        self.world = world;
        self.max_rows = max_rows;
        self.cap = cap;
        self.bucket = bucket;
        self.rope_c = rope_c;
        self.rope_w = rope_w;
        self.rr = try attention.RopeRows.init(gpa, d, glue, max_rows, 2, cfg.rope_dim / 2);
        errdefer self.rr.deinit();
        self.probe = null;
        self.engram = null;
        self.rings = try gpa.alloc(cuda.DeviceBuffer, L);
        self.planes = try gpa.alloc(?Planes, L);
        self.source = try gpa.alloc(usize, L);
        self.attn = try gpa.alloc(attention.Attention, L);
        self.idx_proj_h = try gpa.alloc(cuda.DeviceBuffer, L);
        var src: usize = 0;
        var attn_bytes: u64 = 0;
        self.attn_scratch = try gpa.create(attention.Scratch); // made after the layers, for the widest of them
        for (0..L) |li| {
            self.rings[li] = try cuda.DeviceBuffer.alloc(d, (cfg.window + 16) * cfg.head_dim * 2);
            try self.rings[li].fill8(0, null);
            self.planes[li] = if (cfg.isKvSource(li)) try Planes.init(d, cfg, li, cap) else null;
            if (cfg.isKvSource(li)) src = li;
            self.source[li] = src;
            const l = &w.layers[li];
            self.idx_proj_h[li] = try cuda.DeviceBuffer.alloc(d, if (l.idx_proj) |p| p.count * 2 else 0);
            if (l.idx_proj) |p| {
                var a: cuda.Args = .{};
                a.add(p.ptr());
                a.add(self.idx_proj_h[li].ptr);
                a.add(@as(c_int, @intCast(p.count)));
                try cuda.launch.launch(glue.f32_to_f16, .{ .grid = .{ .x = @intCast((p.count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
            }
            const before = cuda.usage(false).device;
            self.attn[li] = try attention.Attention.initShared(gpa, d, self.dimsOf(li), set, glue, self.attn_scratch);
            attn_bytes += cuda.usage(false).device - before;
        }
        {
            const before = cuda.usage(false).device;
            var need: attention.Scratch.Need = .{};
            for (0..L) |li| need = need.merge(attention.Scratch.Need.of(self.dimsOf(li), self.paramsOf(li).scratchWords(max_rows), glue.hostSelect()));
            self.attn_scratch.* = try attention.Scratch.init(gpa, d, self.dimsOf(0), need);
            attn_bytes += cuda.usage(false).device - before;
        }
        try stream.synchronize();
        if (mem_note.on()) std.log.info("mem: the decode loop's attention {d:.1} MiB for its {d} layers (one shared scratch; bucket {d})", .{ @as(f64, @floatFromInt(attn_bytes)) / (1 << 20), L, bucket });
        mem_note.note(d, "the decode loop's layers (rings, planes of {d} positions, attention scratch)", .{cap});
        const r = max_rows;
        self.ids = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.h = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * dm * 2);
        self.pre = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * 4);
        self.pre_a = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * 4);
        self.pre_f = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * 4);
        self.post = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * 4);
        self.comb = try cuda.DeviceBuffer.alloc(d, r * hcmod.hc_streams * hcmod.hc_streams * 4);
        self.part = try cuda.DeviceBuffer.alloc(d, hcmod.Hc.partWords(r) * 4);
        self.x = try cuda.DeviceBuffer.alloc(d, r * dm * 2);
        self.gathered = try cuda.DeviceBuffer.alloc(d, world * r * dm * 4);
        self.parts = try cuda.DeviceBuffer.alloc(d, self.route.partsWords(r) * 4);
        self.pick = try cuda.DeviceBuffer.alloc(d, r * slots * 4);
        self.wts = try cuda.DeviceBuffer.alloc(d, r * slots * 4);
        self.moe = try cuda.DeviceBuffer.alloc(d, r * dm * 4);
        self.scratch = try experts.Scratch.init(d, &w.layers[0].experts, 64, slots);
        self.xc = try cuda.DeviceBuffer.alloc(d, r * dm * 2);
        self.xh = try cuda.DeviceBuffer.alloc(d, r * dm * 2);
        self.z = try cuda.DeviceBuffer.alloc(d, @max(16, w.head.scratchWords(r) * 4));
        const vh = w.vocab[1] - w.vocab[0];
        self.local = try cuda.DeviceBuffer.alloc(d, r * vh * 4);
        self.glog = try cuda.DeviceBuffer.alloc(d, world * r * vh * 4);
        self.logits = try cuda.DeviceBuffer.alloc(d, r * cfg.vocab * 4);
        self.ids_h = try gpa.alloc(i64, r);
        self.uploads = try attention.Uploads.init(d, attention.Uploads.room(1, r * 8));
        mem_note.note(d, "the decode loop's row buffers", .{});
        return self;
    }

    pub fn deinit(self: *Forward) void {
        for (self.rings) |*b| b.free();
        for (self.planes) |*p| if (p.*) |*pl| pl.deinit();
        for (self.attn) |*a| a.deinit();
        self.attn_scratch.deinit();
        self.gpa.destroy(self.attn_scratch);
        for (self.idx_proj_h) |*b| b.free();
        self.gpa.free(self.rings);
        self.gpa.free(self.planes);
        self.gpa.free(self.source);
        self.gpa.free(self.attn);
        self.gpa.free(self.idx_proj_h);
        inline for (.{ "ids", "h", "pre", "pre_a", "pre_f", "post", "comb", "part", "x", "gathered", "parts", "pick", "wts", "moe", "xc", "xh", "z", "local", "glog", "logits" }) |f| @field(self, f).free();
        self.scratch.deinit();
        self.gpa.free(self.ids_h);
        self.uploads.deinit();
        self.rr.deinit();
    }

    fn dimsOf(self: *const Forward, li: usize) attention.Dims {
        const c = self.cfg;
        return .{
            .d = c.dim,
            .heads = c.n_heads / self.world,
            .hd = c.head_dim,
            .rd = c.rope_dim,
            .q_rank = c.q_rank,
            .o_rank = c.o_rank,
            .groups = c.o_groups / self.world,
            .window = c.window,
            .ring = c.window + 16,
            .raw = 64,
            .idx_heads = c.idx_heads,
            .idx_dim = c.idx_dim,
            .idx_topk = c.idx_topk,
            .ratio = c.ratio(li),
            .kv_source = c.isKvSource(li),
            .index_source = c.isIndexSource(li),
            .has_idx_wk = c.isKvSource(li) and c.isIndexSource(li),
            .eps = c.eps,
            .bucket = self.bucket,
            .max_rows = self.max_rows,
            .cand = attention.CandRole.of(c.cand_source, li, c.isIndexSource(li) and c.ratio(li) > 0),
            .cand_blocks = c.cand_blocks,
            .cand_block = @max(1, c.cand_block),
        };
    }

    fn paramsOf(self: *const Forward, li: usize) attention.Params {
        const l = &self.w.layers[li];
        return .{
            .wq_a = &l.wq_a,
            .wq_b = &l.wq_b,
            .wkv = &l.wkv,
            .wo_b = &l.wo_b,
            .wo_a = .{ &l.wo_a[0], &l.wo_a[1], &l.wo_a[2], &l.wo_a[3] },
            .comp_wkv = if (l.comp_wkv) |*x| x else null,
            .comp_wgate = if (l.comp_wgate) |*x| x else null,
            .idx_wq_b = if (l.idx_wq_b) |*x| x else null,
            .idx_wk = if (l.idx_wk) |*x| x else null,
            .q_norm = l.q_norm.ptr(),
            .kv_norm = l.kv_norm.ptr(),
            .sink = l.sink.ptr(),
            .comp_norm = if (l.comp_norm) |p| p.ptr() else 0,
            .idx_k_norm = if (l.idx_k_norm) |p| p.ptr() else 0,
            .idx_proj_h = self.idx_proj_h[li].ptr,
        };
    }

    /// The planes layer `li` reads: its ring, and its kv source's compressed planes (empty for window layers).
    pub fn cachesOf(self: *const Forward, li: usize) attention.Caches {
        const none: cuda.DeviceBuffer = .{ .d = self.d, .ptr = 0, .len = 0 };
        const p: ?Planes = if (self.cfg.ratio(li) > 0) self.planes[self.source[li]] else null;
        return .{
            .ring = self.rings[li],
            .comp_codes = if (p) |pl| pl.comp_codes else none,
            .comp_scales = if (p) |pl| pl.comp_scales else none,
            .idxk_codes = if (p) |pl| pl.idxk_codes else none,
            .idxk_scales = if (p) |pl| pl.idxk_scales else none,
            .raw_kv = if (p) |pl| pl.raw_kv else none,
            .raw_score = if (p) |pl| pl.raw_score else none,
            .rows = if (p) |pl| pl.rows else 1,
        };
    }

    fn probeOut(self: *const Forward, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) !void {
        if (self.probe) |p| try p.call(p.ctx, name, ptr, bytes, stream);
    }

    /// The window's ids at positions start .. start + n - 1 -> logits [n (all_logits) or 1, V] f32 (`self.logits`);
    /// the caches hold the window's rows after.
    pub fn rows(self: *Forward, gather: Gather, ids: []const i64, start: usize, n: usize, all_logits: bool, stream: cuda.Stream) !Ptr {
        if (n < 1 or n > self.max_rows) return error.TooManyRows;
        const c = self.cfg;
        const dm = c.dim;
        const w = self.w;
        @memcpy(self.ids_h[0..n], ids);
        try self.uploads.begin();
        try self.uploads.put(self.ids, 0, std.mem.sliceAsBytes(self.ids_h[0..n]), stream);
        try self.hc.embedInit(stream, w.embed.ptr(), self.ids.ptr, self.h.ptr, self.pre.ptr, @intCast(n));
        var pre = self.pre.ptr;
        var pre_f = self.pre_f.ptr;
        var picks: ?attention.Picks = null; // the latest index layer's (shared["topk"])
        var nb: [32]u8 = undefined;
        for (w.layers, 0..) |*l, li| {
            if (l.engram_wkv != null) if (self.engram) |e| {
                try e.call(e.ctx, li, self.h.ptr, start, n, null, gather, stream);
                try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_engram", .{li}), self.h.ptr, n * hcmod.hc_streams * dm * 2, stream);
            };
            try self.hc.pre(stream, self.h.ptr, l.hc_attn_fn.ptr(), l.hc_attn_scale.ptr(), l.hc_attn_base.ptr(), pre, l.attn_norm.ptr(), self.x.ptr, self.pre_a.ptr, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(n));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_attn", .{li}), self.x.ptr, n * dm * 2, stream);
            const caches = self.cachesOf(li);
            const rope = if (c.ratio(li) > 0) self.rope_c else self.rope_w;
            const pa = try self.attn[li].forward(self.paramsOf(li), &caches, rope, &self.rr, self.x.ptr, n, start, picks, stream);
            if (c.isIndexSource(li)) picks = self.attn[li].picks();
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_attn.r0", .{li}), pa, n * dm * 4, stream);
            try gather.run(pa, n * dm, self.gathered.ptr, stream);
            try self.hc.post(stream, self.gathered.ptr, self.world, self.h.ptr, self.post.ptr, self.comb.ptr, self.h.ptr, @intCast(n));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_attn", .{li}), self.h.ptr, n * hcmod.hc_streams * dm * 2, stream);
            try self.hc.pre(stream, self.h.ptr, l.hc_ffn_fn.ptr(), l.hc_ffn_scale.ptr(), l.hc_ffn_base.ptr(), self.pre_a.ptr, l.ffn_norm.ptr(), self.x.ptr, pre_f, self.post.ptr, self.comb.ptr, self.part.ptr, @intCast(n));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.x_ffn", .{li}), self.x.ptr, n * dm * 2, stream);
            try self.route.gate(stream, self.x.ptr, l.gate_w.ptr(), self.parts.ptr, @intCast(n));
            try self.route.pick(stream, self.parts.ptr, l.gate_b.ptr(), self.pick.ptr, self.wts.ptr, @intCast(n));
            try experts.routed(self.ek, &l.experts, &self.scratch, self.x.ptr, self.pick.ptr, self.wts.ptr, n, c.swiglu_limit, self.moe.ptr, stream);
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.p_moe.r0", .{li}), self.moe.ptr, n * dm * 4, stream);
            try gather.run(self.moe.ptr, n * dm, self.gathered.ptr, stream);
            try self.hc.post(stream, self.gathered.ptr, self.world, self.h.ptr, self.post.ptr, self.comb.ptr, self.h.ptr, @intCast(n));
            try self.probeOut(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{li}), self.h.ptr, n * hcmod.hc_streams * dm * 2, stream);
            const t = pre;
            pre = pre_f;
            pre_f = t;
        }
        // the head: every row's logits, or the last row's
        const hr: usize = if (all_logits) n else 1;
        const h0: Ptr = if (all_logits) self.h.ptr else self.h.ptr + (n - 1) * hcmod.hc_streams * dm * 2;
        const pre0: Ptr = if (all_logits) pre else pre + (n - 1) * hcmod.hc_streams * 4;
        try self.hc.collapseNorm(stream, h0, pre0, w.norm.ptr(), self.xc.ptr, @intCast(hr));
        try self.probeOut("head.xc", self.xc.ptr, hr * dm * 2, stream);
        const vh = w.vocab[1] - w.vocab[0];
        try w.head.run(self.xc.ptr, .bf16, hr, self.xh.ptr, self.z.ptr, self.local.ptr, .f32, stream);
        try self.probeOut("head.local.r0", self.local.ptr, hr * vh * 4, stream);
        try gather.run(self.local.ptr, hr * vh, self.glog.ptr, stream);
        for (0..hr) |r| for (0..self.world) |rank| {
            try self.logits.copyFrom((r * c.vocab + rank * vh) * 4, self.glog.ptr + ((rank * hr + r) * vh) * 4, vh * 4, stream.handle);
        };
        return self.logits.ptr;
    }
};
