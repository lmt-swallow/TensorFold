//! GLM-5.3-Flash's prompt path on one rank: the Mia fork's `decode.prefill` / `prefill_chunk` in the strict setting.
//! A prompt is committed in chunks of up to `Prompt.rows` rows (2,048 as served; a resumed prompt's first chunk
//! starts anywhere): the chunk's forward on the prompt buffers (every block's prompt-form kernels: `prefill` true),
//! the head's logits of its last row, the MTP layer's absorb of its rows (the next token of each, so a prompt's last
//! row waits for its first sampled token), then the commit (the KDA layers committed inside the forward).
//!
//! A chunk is planned as a list of ops (`Chunk.plan`) in the reference's call order: compute on this rank, and
//! all-gathers between the ranks (the partials, rank 0's first; with TF_GLM_HC_SPLIT the row pieces each rank swaps
//! with the other). `prefillChunk` runs them on a `Link`, its gathers on a second stream with `Overlap`; a test runs
//! two ranks' ops in turn on one GPU and moves their bytes itself. Rows never mix, so a row's bits are those it
//! gets in any chunk of the same kind; which kernels run depends on the chunk's row count and position alone.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const glue_mod = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const weights = @import("cuda_weights.zig");
const forward = @import("cuda_forward.zig");
const dsa = @import("cuda_dsa.zig");
const moe = @import("cuda_moe.zig");
const mtp = @import("cuda_mtp.zig");
const model = @import("cuda_model.zig");

const Glue = glue_mod.Glue;
const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

/// decode.py PREFILL_ROWS: the served prompt chunk.
pub const chunk_rows = 2048;

/// A layer's token mixer on this rank: a KDA layer (its index `li` among the KDA layers of the stream's
/// `forward.KdaState`) or a DSA layer (its index `di` among the stream's DSA caches, `Stream.caches`).
pub const Mixer = union(enum) {
    kda: struct { w: *const weights.Kda, li: usize },
    dsa: struct { w: *const dsa.Dsa, di: usize },
};

pub const Ffn = union(enum) {
    mlp: *const weights.Mlp,
    moe: *const moe.Moe,
};

pub const Layer = struct {
    frame: *const weights.Frame,
    mixer: Mixer,
    ffn: Ffn,
};

/// What the prompt path reads of one rank's model: the layers in order, the embedding, the final norm, the head's
/// BF16 vocabulary rows and the MTP layer.
pub const Model = struct {
    layers: []const Layer,
    embed: Ptr,
    norm: Ptr,
    head: glue_mod.B16,
    mtp: *const mtp.Mtp,
};

/// The prompt path's view of one rank's whole model (cuda_model.zig): `out` [m.layers.len] takes its layers, each
/// mixer's index among the stream's KDA states or DSA caches (config.zig `kindIndex`, as model.State keeps them).
pub fn of(m: *const model.Model, cfg: *const config.Config, mtp_w: *const mtp.Mtp, out: []Layer) Model {
    for (m.layers, out) |*x, *y| y.* = .{
        .frame = &x.frame,
        .mixer = if (x.kda) |*w| .{ .kda = .{ .w = w, .li = cfg.kindIndex(x.index) } } else .{ .dsa = .{ .w = &x.dsa.?, .di = cfg.kindIndex(x.index) } },
        .ffn = if (x.mlp) |*w| .{ .mlp = w } else .{ .moe = &x.moe.? },
    };
    return .{ .layers = out, .embed = m.embed.ptr, .norm = m.norm.ptr, .head = m.head.b16(), .mtp = mtp_w };
}

pub const Kernels = struct {
    g: Glue,
    kda: *const kda.Kernels,
    moe: *const moe.Kernels,
    mtp: *const mtp.Kernels,
};

/// The prompt buffers a chunk runs on (decode.Engine's `pbuf` and the scratch its blocks use), for chunks of up to
/// `rows` rows; every one built for prompt chunks (`prefill` true). A caller that keeps its own (an engine's prompt
/// work buffers) points at them; `Owned` allocates a set.
pub const Prompt = struct {
    rows: usize,
    b: *const forward.Buffers,
    temp: *const kda.Temp,
    kscratch: *const kda.Scratch, // one layer's chain scratch: each KDA layer of a chunk commits before the next runs
    ds: *const dsa.Scratch,
    ms: *const moe.Scratch,
    mb: *const mtp.Buffers,
    fnormed: Ptr, // [rows, D] bf16: the final norm of the streams' mean (the MTP head's input)
    fxs: Ptr, // [rows, D / 64] fp32
    logits: Ptr, // [1, vocab share] bf16: the head of the chunk's last row
    // the split gathers the peer's partial pieces into b.gath ([2, n, D] a piece, in piece order: hcsplit.py's
    // `got`), which hc_post_pair reads at its element distance from b.part (an int32: the two stay close)
    stage_rows: Ptr, // [rows + 2, D] bf16: the split's swapped rows' pieces as gathered
};

/// A set of prompt buffers of its own (`view`).
pub const Owned = struct {
    rows: usize,
    b: forward.Buffers,
    temp: kda.Temp,
    kscratch: kda.Scratch,
    ds: dsa.Scratch,
    ms: moe.Scratch,
    mb: mtp.Buffers,
    fnormed: Buf,
    fxs: Buf,
    logits: Buf,
    stage_rows: Buf,

    pub fn init(d: *const cuda.Driver, lay: config.Layout, rows: usize, capacity: usize) !Owned {
        const c = lay.cfg;
        var p: Owned = undefined;
        p.rows = rows;
        p.b = try forward.Buffers.init(d, lay, rows, true);
        errdefer p.b.deinit();
        p.temp = try kda.Temp.init(d, rows, lay.size(.lin));
        errdefer p.temp.deinit();
        p.kscratch = try kda.Scratch.init(d, 1, rows, lay.size(.lin));
        errdefer p.kscratch.deinit();
        p.ds = try dsa.Scratch.init(d, lay, rows, capacity, true);
        errdefer p.ds.deinit();
        p.ms = try moe.Scratch.init(d, lay, rows, true);
        errdefer p.ms.deinit();
        p.mb = try mtp.Buffers.init(d, lay, rows, true);
        errdefer p.mb.deinit();
        p.fnormed = try Buf.alloc(d, rows * c.hidden * 2);
        errdefer p.fnormed.free();
        p.fxs = try Buf.alloc(d, rows * c.hidden / 64 * 4);
        errdefer p.fxs.free();
        p.logits = try Buf.alloc(d, lay.size(.vocab) * 2);
        errdefer p.logits.free();
        p.stage_rows = try Buf.alloc(d, (rows + 2) * c.hidden * 2);
        return p;
    }

    pub fn deinit(p: *Owned) void {
        p.b.deinit();
        p.temp.deinit();
        p.kscratch.deinit();
        p.ds.deinit();
        p.ms.deinit();
        p.mb.deinit();
        p.fnormed.free();
        p.fxs.free();
        p.logits.free();
        p.stage_rows.free();
    }

    pub fn view(p: *const Owned) Prompt {
        return .{
            .rows = p.rows, .b = &p.b, .temp = &p.temp, .kscratch = &p.kscratch, .ds = &p.ds, .ms = &p.ms, .mb = &p.mb,
            .fnormed = p.fnormed.ptr, .fxs = p.fxs.ptr, .logits = p.logits.ptr, .stage_rows = p.stage_rows.ptr,
        };
    }
};

/// One stream on this rank: its KDA state, its DSA layers' caches and the MTP layer's (each a stream's view of the
/// pool), its next position and the MTP cache's length, and two device int32s of its own that the chunk sets to
/// them (the position the kernels read; 16-byte aligned, as the reference's tensors are).
pub const Stream = struct {
    kda: *forward.KdaState,
    caches: []const dsa.Cache,
    mtp_cache: *const dsa.Cache,
    pos: usize,
    mtp_len: usize,
    capacity: usize,
    pos_dev: Ptr,
    mtp_pos_dev: Ptr,
};

/// TF_GLM_HC_SPLIT=1 (hcsplit.py `SplitSettings` of two ranks, p2p): a prompt chunk of `min_rows` rows or more has
/// its hyper-connection glue split by rows between the ranks, in `pieces` row pieces a site; `fronts`
/// (TF_GLM_PREFILL_OVERLAP=2): the next block's row-independent front runs on this rank's rows of each piece after
/// its glue. Every bit is the unsplit path's: the glue kernels are row-independent and the partials are summed rank 0
/// first as hc_post sums them.
pub const Split = struct {
    pieces: usize = 2,
    min_rows: usize = 64,
    fronts: bool = true,
};

/// Which block of layer `l` an op belongs to: its token mixer or its FFN.
pub const Site = enum { mixer, ffn };

/// One step of a chunk on this rank (`Chunk.plan`). `gather` is the only exchange: every rank's `bytes` at `send`
/// into `recv`, rank 0's first; with `copy`, the other rank's part then copied to `dst` (a row swap). `wait` makes
/// what follows wait for gather `ev` (a no-op when the gathers run on the chunk's own stream).
pub const Op = union(enum) {
    embed,
    zero_pad: struct { lo: usize, hi: usize }, // the split's pad rows of the streams and the partial
    hc_pre: struct { l: usize, site: Site, lo: usize, hi: usize },
    hc_post, // unsplit: every row from the gathered partials
    post_pair: struct { lo: usize, hi: usize, got: Ptr }, // split: this rank's rows from its partial and the peer's
    // hc_post (got null) or post_pair (got the peer's piece) with the next hc_pre's prompt dots in one pass
    // (Glue.hcPostPartial), then that hc_pre's finish: the same outputs as the two ops in turn
    post_pre: struct { l: usize, site: Site, lo: usize, hi: usize, got: ?Ptr },
    mean: struct { lo: usize, hi: usize }, // the streams' mean into the hidden rows
    front: struct { l: usize, site: Site, lo: usize, hi: usize },
    rest: struct { l: usize, site: Site },
    fill: struct { l: usize, site: Site, lo: usize, hi: usize },
    gather: struct { send: Ptr, recv: Ptr, bytes: usize, ev: u8, copy: ?struct { src: Ptr, dst: Ptr, bytes: usize } = null },
    wait: u8,
    tail, // the final norm of the hidden rows and the head of the last row
    mtp_front, // the MTP absorb: its front and attention
    mtp_mid, // its residual, post-norm and MoE
    mtp_end, // its residual and draft head
    commit,
};

/// The all-gather between the ranks (NCCL on the caller's stream), `bytes` a rank, rank 0's first.
pub const Link = struct {
    ctx: *anyopaque,
    gather: *const fn (ctx: *anyopaque, send: Ptr, recv: Ptr, bytes: usize, stream: cuda.Stream) anyerror!void,
};

/// TF_GLM_PREFILL_OVERLAP: the gathers on a second stream (`side`), the chunk's stream waiting only where it reads
/// their result. `link` must issue on `side`: the chunk's stream issues none while a chunk runs.
pub const Overlap = struct {
    side: cuda.Stream,
    ready: cuda.Event, // the chunk's stream has made what the next gather sends
    ev: [16]cuda.Event,

    pub fn init(d: *const cuda.Driver) !Overlap {
        var o: Overlap = .{ .side = try cuda.Stream.init(d, true), .ready = try cuda.Event.init(d, false), .ev = undefined };
        for (&o.ev) |*e| e.* = try cuda.Event.init(d, false);
        return o;
    }

    pub fn deinit(o: *Overlap) void {
        o.side.deinit();
        o.ready.deinit();
        for (&o.ev) |*e| e.deinit();
    }
};

/// Rows `ids` at stream position st.pos (`next`: the token after each of the first `next.len` rows, which the MTP
/// layer absorbs: R of them, or R - 1 at a prompt's end).
pub const Chunk = struct {
    p: Prompt,
    m: *const Model,
    k: Kernels,
    lay: config.Layout,
    st: *Stream,
    ids: []const u32,
    next: []const u32,
    head: bool = true,
    split: ?Split = null,

    pub fn rows(c: *const Chunk) usize {
        return c.ids.len;
    }

    /// forward.py `check_room` and the buffers' bound.
    pub fn check(c: *const Chunk) !void {
        const R = c.rows();
        if (R == 0 or R > c.p.rows) return error.TooManyRows;
        if (c.next.len != R and c.next.len + 1 != R) return error.BadNextTokens;
        if (c.st.pos + R > c.st.capacity or c.st.mtp_len + c.next.len > c.st.capacity) return error.PastCapacity;
    }

    /// Whether hc_post and the next hc_pre run as one op (`post_pre`): hc_fused.cu takes two ranks' partials.
    fn fusesHc(c: *const Chunk) bool {
        return c.k.g.hc.fused and c.lay.world == 2;
    }

    /// Whether this chunk runs row-split (hcsplit.py `applies`).
    pub fn splits(c: *const Chunk) bool {
        const sp = c.split orelse return false;
        return c.lay.world == 2 and c.rows() >= sp.min_rows and 2 * cdiv(c.rows(), 2) <= c.p.rows;
    }

    /// The chunk's steps in the reference's order (decode.prefill_chunk with forward.py `_compute`, or with
    /// `split_layer_steps` when it splits, then the MTP absorb and the commit).
    pub fn plan(c: *const Chunk, gpa: std.mem.Allocator) ![]Op {
        try c.check();
        var ops: std.ArrayList(Op) = .empty;
        errdefer ops.deinit(gpa);
        const R = c.rows();
        const D = c.lay.cfg.hidden;
        const L = c.m.layers.len;
        const b = c.p.b;
        try ops.append(gpa, .embed);
        if (c.splits()) {
            try c.planSplit(gpa, &ops);
        } else {
            const fuse = c.fusesHc();
            for (0..L) |l| {
                if (l > 0 and fuse) {
                    try ops.append(gpa, .{ .post_pre = .{ .l = l, .site = .mixer, .lo = 0, .hi = R, .got = null } });
                } else {
                    if (l > 0) try ops.append(gpa, .hc_post);
                    try ops.append(gpa, .{ .hc_pre = .{ .l = l, .site = .mixer, .lo = 0, .hi = R } });
                }
                try c.block(gpa, &ops, l, .mixer, null, null);
                try ops.append(gpa, .{ .gather = .{ .send = b.part.ptr, .recv = b.gath.ptr, .bytes = R * D * 4, .ev = 0 } });
                try ops.append(gpa, .{ .wait = 0 });
                if (fuse) {
                    try ops.append(gpa, .{ .post_pre = .{ .l = l, .site = .ffn, .lo = 0, .hi = R, .got = null } });
                } else {
                    try ops.append(gpa, .hc_post);
                    try ops.append(gpa, .{ .hc_pre = .{ .l = l, .site = .ffn, .lo = 0, .hi = R } });
                }
                try c.block(gpa, &ops, l, .ffn, null, null);
                try ops.append(gpa, .{ .gather = .{ .send = b.part.ptr, .recv = b.gath.ptr, .bytes = R * D * 4, .ev = 0 } });
                try ops.append(gpa, .{ .wait = 0 });
            }
            try ops.append(gpa, .hc_post);
            try ops.append(gpa, .{ .mean = .{ .lo = 0, .hi = R } });
        }
        try ops.append(gpa, .tail);
        const n = c.next.len;
        if (n > 0) {
            const mb = c.p.mb;
            try ops.append(gpa, .mtp_front);
            try ops.append(gpa, .{ .gather = .{ .send = mb.part.ptr, .recv = mb.gath.ptr, .bytes = n * D * 4, .ev = 0 } });
            try ops.append(gpa, .{ .wait = 0 });
            try ops.append(gpa, .mtp_mid);
            try ops.append(gpa, .{ .gather = .{ .send = mb.part.ptr, .recv = mb.gath.ptr, .bytes = n * D * 4, .ev = 0 } });
            try ops.append(gpa, .{ .wait = 0 });
            try ops.append(gpa, .mtp_end);
        }
        try ops.append(gpa, .commit);
        return ops.toOwnedSlice(gpa);
    }

    const Range = struct { lo: usize, hi: usize };

    /// forward.py `undone`: the ranges of [0, R) outside `done`, in row order.
    fn undone(R: usize, done: ?Range, out: *[2]Range) []const Range {
        const d = done orelse {
            out[0] = .{ .lo = 0, .hi = R };
            return out[0..1];
        };
        var k: usize = 0;
        if (@min(d.lo, R) > 0) {
            out[k] = .{ .lo = 0, .hi = @min(d.lo, R) };
            k += 1;
        }
        if (R > d.hi) {
            out[k] = .{ .lo = d.hi, .hi = R };
            k += 1;
        }
        return out[0..k];
    }

    /// A block of layer l in the reference's call order: its fronts on the rows `done` leaves, the rest, then its
    /// output projection: on every row (`sp` null), or as the split's `partial` (the peer's pieces, each gathered, then
    /// this rank's rows).
    fn block(c: *const Chunk, gpa: std.mem.Allocator, ops: *std.ArrayList(Op), l: usize, site: Site, done: ?Range, sp: ?*const SplitPlan) !void {
        const R = c.rows();
        const layer = &c.m.layers[l];
        const moe_ffn = site == .ffn and layer.ffn == .moe;
        var buf: [2]Range = undefined;
        if (moe_ffn) {
            try ops.append(gpa, .{ .rest = .{ .l = l, .site = site } });
        } else {
            for (undone(R, done, &buf)) |r| try ops.append(gpa, .{ .front = .{ .l = l, .site = site, .lo = r.lo, .hi = r.hi } });
            if (site == .mixer) try ops.append(gpa, .{ .rest = .{ .l = l, .site = site } });
        }
        const fill = struct {
            fn f(cc: *const Chunk, g2: std.mem.Allocator, o: *std.ArrayList(Op), ll: usize, s2: Site, moe2: bool, dn: ?Range, lo: usize, hi_: usize) !void {
                const hi = @min(hi_, cc.rows());
                if (hi <= lo) return;
                if (moe2) {
                    // glue.py's MoE fill: the shared expert on the rows of lo .. hi no front ran on, then the sum
                    var b2: [2]Range = undefined;
                    for (undone(hi, dn, &b2)) |r| if (@max(r.lo, lo) < r.hi)
                        try o.append(g2, .{ .front = .{ .l = ll, .site = s2, .lo = @max(r.lo, lo), .hi = r.hi } });
                }
                try o.append(g2, .{ .fill = .{ .l = ll, .site = s2, .lo = lo, .hi = hi } });
            }
        }.f;
        const s = sp orelse return fill(c, gpa, ops, l, site, moe_ffn, done, 0, R);
        const D = c.lay.cfg.hidden;
        for (s.pieces[0..s.n], 0..) |pc, j| {
            try fill(c, gpa, ops, l, site, moe_ffn, done, s.theirs + pc.lo, s.theirs + pc.hi);
            const n = pc.hi - pc.lo;
            try ops.append(gpa, .{ .gather = .{ .send = c.p.b.part.ptr + (s.theirs + pc.lo) * D * 4, .recv = c.p.b.gath.ptr + 2 * pc.lo * D * 4, .bytes = n * D * 4, .ev = @intCast(j) } });
        }
        try fill(c, gpa, ops, l, site, moe_ffn, done, s.mine, s.mine + s.H);
    }

    const SplitPlan = struct {
        H: usize,
        mine: usize,
        theirs: usize,
        pieces: [8]Range,
        n: usize,
    };

    fn splitPlan(c: *const Chunk) SplitPlan {
        const sp = c.split.?;
        const H = cdiv(c.rows(), 2);
        var s: SplitPlan = .{ .H = H, .mine = c.lay.rank * H, .theirs = (1 - c.lay.rank) * H, .pieces = undefined, .n = 0 };
        var size = cdiv(H, @max(1, @min(sp.pieces, 8)));
        if (size > 64) size = @min(H, cdiv(size, 64) * 64); // whole 64-row tiles for the glue kernels
        var off: usize = 0;
        while (off < H) : (off += size) {
            s.pieces[s.n] = .{ .lo = off, .hi = @min(H, off + size) };
            s.n += 1;
        }
        return s;
    }

    /// forward.py `split_layer_steps` (hcsplit.py `HcSplit`): rank r owns rows [r H, r H + H), H = ceil(R / 2).
    fn planSplit(c: *const Chunk, gpa: std.mem.Allocator, ops: *std.ArrayList(Op)) !void {
        const R = c.rows();
        const L = c.m.layers.len;
        const D = c.lay.cfg.hidden;
        const s = c.splitPlan();
        const fronts = c.split.?.fronts;
        if (2 * s.H > R) try ops.append(gpa, .{ .zero_pad = .{ .lo = R, .hi = 2 * s.H } });
        try ops.append(gpa, .{ .hc_pre = .{ .l = 0, .site = .mixer, .lo = 0, .hi = 2 * s.H } });
        var done: ?Range = null;
        const own: Range = .{ .lo = s.mine, .hi = @min(s.mine + s.H, R) };
        for (0..L) |l| {
            try c.block(gpa, ops, l, .mixer, done, &s);
            // the FFN's glue: this rank's rows of each piece, the normed rows swapped, its front meanwhile
            try c.glue(gpa, ops, &s, .{ .l = l, .site = .ffn }, false, if (fronts) .{ .l = l, .site = .ffn } else null);
            try c.block(gpa, ops, l, .ffn, if (fronts) own else null, &s);
            const last = l + 1 == L;
            const front: ?LayerSite = if (fronts and !last) .{ .l = l + 1, .site = .mixer } else null;
            try c.glue(gpa, ops, &s, if (last) null else .{ .l = l + 1, .site = .mixer }, last, front);
            done = if (front != null) own else null;
        }
        _ = D;
    }

    const LayerSite = struct { l: usize, site: Site };

    /// glue.hc_pre of rows lo .. hi of layer l's `site` (partial_done: hcPostPartial wrote their prompt dots).
    fn hcPreRows(c: *const Chunk, l: usize, site: Site, lo: usize, hi: usize, partial_done: bool, stream: cuda.Stream) !void {
        const cfg = c.lay.cfg;
        const D = cfg.hidden;
        const S = cfg.streams;
        const b = c.p.b;
        const f = c.m.layers[l].frame;
        const hc = if (site == .ffn) &f.ffn_hc else &f.attn_hc;
        try c.k.g.hcPre(.{
            .x = b.x.ptr + lo * S * D * 2, .fn_ = hc.fn_.ptr, .base = hc.base.ptr, .scale = hc.scale.ptr,
            .norm = (if (site == .ffn) f.post_norm else f.in_norm).ptr,
            .out = b.normed.ptr + lo * D * 2, .xs = b.xs.ptr + lo * D / 64 * 4,
            .post = b.post.ptr + lo * S * 4, .comb = b.comb.ptr + lo * S * S * 4,
            .part = b.hcpart.ptr + lo * glue_mod.hc_blocks * 32 * 4,
            .eps = cfg.eps, .hc_eps = cfg.hc_eps, .iters = cfg.hc_iters, .rows = hi - lo, .prompt = true,
            .partial_done = partial_done,
        }, stream);
    }

    /// hcsplit.py `HcSplit.glue`: per piece, this rank's rows' hc_post (after the piece's partial arrived), the final
    /// mean (`final`), the next hc_pre (`next`), the rows swapped, then `front` on them; the swaps all done after.
    fn glue(c: *const Chunk, gpa: std.mem.Allocator, ops: *std.ArrayList(Op), s: *const SplitPlan, next: ?LayerSite, final: bool, front: ?LayerSite) !void {
        const D = c.lay.cfg.hidden;
        const b = c.p.b;
        const peer = 1 - c.lay.rank;
        for (s.pieces[0..s.n], 0..) |pc, j| {
            const lo = s.mine + pc.lo;
            const hi = s.mine + pc.hi;
            const n = pc.hi - pc.lo;
            try ops.append(gpa, .{ .wait = @intCast(j) });
            const got = c.p.b.gath.ptr + (2 * pc.lo + peer * n) * D * 4;
            if (next != null and !final and c.fusesHc()) {
                try ops.append(gpa, .{ .post_pre = .{ .l = next.?.l, .site = next.?.site, .lo = lo, .hi = hi, .got = got } });
            } else {
                try ops.append(gpa, .{ .post_pair = .{ .lo = lo, .hi = hi, .got = got } });
                if (final) try ops.append(gpa, .{ .mean = .{ .lo = lo, .hi = hi } });
                if (next) |x| try ops.append(gpa, .{ .hc_pre = .{ .l = x.l, .site = x.site, .lo = lo, .hi = hi } });
            }
            // the rows the next step reads whole: the normed rows, or the hidden rows after the last layer
            const t: Ptr = if (final) b.hidden.ptr else b.normed.ptr;
            const stage = c.p.stage_rows + 2 * pc.lo * D * 2;
            try ops.append(gpa, .{ .gather = .{ .send = t + lo * D * 2, .recv = stage, .bytes = n * D * 2, .ev = @intCast(8 + j), .copy = .{ .src = stage + peer * n * D * 2, .dst = t + (s.theirs + pc.lo) * D * 2, .bytes = n * D * 2 } } });
            if (front) |f| {
                const fhi = @min(hi, c.rows());
                if (fhi > lo) try ops.append(gpa, .{ .front = .{ .l = f.l, .site = f.site, .lo = lo, .hi = fhi } });
            }
        }
        try ops.append(gpa, .{ .wait = @intCast(8 + s.n - 1) });
    }

    fn ctx(c: *const Chunk) forward.Ctx {
        return .{ .g = c.k.g, .kk = c.k.kda, .temp = c.p.temp, .prompt_scratch = c.p.kscratch, .lay = c.lay };
    }

    fn dsaCall(c: *const Chunk) dsa.Call {
        const p = c.p;
        const R = c.rows();
        return .{ .normed = p.b.normed.ptr, .part = p.b.part.ptr, .sk = p.b.sk.ptr, .pos_dev = c.st.pos_dev, .host_pos = c.st.pos, .nch = cdiv(c.st.pos + R, 512), .R = R };
    }

    /// Runs one op of `plan` that is not a `gather` or `wait` (those are the caller's: `prefillChunk`, or a test that
    /// moves the ranks' bytes itself).
    pub fn exec(c: *const Chunk, op: Op, stream: cuda.Stream) !void {
        const p = c.p;
        const b = p.b;
        const cfg = c.lay.cfg;
        const D = cfg.hidden;
        const S = cfg.streams;
        const R = c.rows();
        const n = c.next.len;
        const g = c.k.g;
        switch (op) {
            .embed => {
                const pos: i32 = @intCast(c.st.pos);
                const mpos: i32 = @intCast(c.st.mtp_len);
                try devInt(c.st.pos_dev, b.ids.d).uploadAsync(0, std.mem.asBytes(&pos), stream.handle);
                try devInt(c.st.mtp_pos_dev, b.ids.d).uploadAsync(0, std.mem.asBytes(&mpos), stream.handle);
                try b.ids.uploadAsync(0, std.mem.sliceAsBytes(c.ids), stream.handle);
                if (n > 0) try p.mb.ids.uploadAsync(0, std.mem.sliceAsBytes(c.next), stream.handle);
                try stream.synchronize(); // the host slices need not outlive the call
                try g.embed(b.ids.ptr, c.m.embed, b.x.ptr, R, S, stream);
            },
            .zero_pad => |z| {
                const xs: Buf = .{ .d = b.x.d, .ptr = b.x.ptr + z.lo * S * D * 2, .len = (z.hi - z.lo) * S * D * 2 };
                try xs.fill8(0, stream.handle);
                const ps: Buf = .{ .d = b.x.d, .ptr = b.part.ptr + z.lo * D * 4, .len = (z.hi - z.lo) * D * 4 };
                try ps.fill8(0, stream.handle);
            },
            .hc_pre => |h| try c.hcPreRows(h.l, h.site, h.lo, h.hi, false, stream),
            .post_pre => |x| {
                const f = c.m.layers[x.l].frame;
                const hc = if (x.site == .ffn) &f.ffn_hc else &f.attn_hc;
                const xp = b.x.ptr + x.lo * S * D * 2;
                // rank 0's partial first: unsplit, the gathered [2, R, D]; split, this rank's piece and the peer's
                const g0: Ptr, const rs: i64 = if (x.got) |got| blk: {
                    const own = b.part.ptr + x.lo * D * 4;
                    const a0, const a1 = if (c.lay.rank == 0) .{ own, got } else .{ got, own };
                    break :blk .{ a0, @divExact(@as(i64, @intCast(a1)) - @as(i64, @intCast(a0)), 4) };
                } else .{ b.gath.ptr, @intCast(R * D) };
                try g.hcPostPartial(xp, xp, g0, rs, b.post.ptr + x.lo * S * 4, b.comb.ptr + x.lo * S * S * 4, hc.fn_.ptr, b.hcpart.ptr + x.lo * glue_mod.hc_blocks * 32 * 4, x.hi - x.lo, stream);
                try c.hcPreRows(x.l, x.site, x.lo, x.hi, true, stream);
            },
            .hc_post => try forward.hcPost(c.ctx(), b, R, stream),
            .post_pair => |x| {
                const own = b.part.ptr + x.lo * D * 4;
                const g0, const g1 = if (c.lay.rank == 0) .{ own, x.got } else .{ x.got, own };
                const xp = b.x.ptr + x.lo * S * D * 2;
                try g.hcPostPair(xp, xp, g0, g1, b.post.ptr + x.lo * S * 4, b.comb.ptr + x.lo * S * S * 4, x.hi - x.lo, stream);
            },
            .mean => |x| try g.streamMean(b.x.ptr + x.lo * S * D * 2, b.hidden.ptr + x.lo * D * 2, x.hi - x.lo, stream),
            .front => |x| try c.runFront(x.l, x.site, x.lo, x.hi, stream),
            .rest => |x| try c.runRest(x.l, x.site, stream),
            .fill => |x| try c.fillRows(x.l, x.site, x.lo, x.hi, stream),
            .tail => {
                try g.rmsnorm(b.hidden.ptr, D, c.m.norm, p.fnormed, D, p.fxs, cfg.eps, R, D, stream);
                // the head reads the last row only (forward.py `_compute` of a prompt buffer)
                if (c.head) try g.bmm(p.fnormed + (R - 1) * D * 2, D, c.m.head, 1, p.logits, false, b.sk.ptr, stream);
            },
            .mtp_front => {
                // decode._absorb_rows: the chunk's final-normed rows into the MTP head (mtp.py `mtp_stage`)
                const hin: Buf = .{ .d = p.mb.hin.d, .ptr = p.mb.hin.ptr, .len = n * D * 2 };
                try hin.copyFrom(0, p.fnormed, n * D * 2, stream.handle);
                const m = c.m.mtp;
                try mtp.front(g, cfg, m, p.mb, c.m.embed, n, c.st.mtp_len == 0, stream);
                try dsa.dsaBlock(g, c.lay, &m.dsa, c.st.mtp_cache, p.ds, .{
                    .normed = p.mb.normed.ptr, .part = p.mb.part.ptr, .sk = p.mb.sk.ptr, .pos_dev = c.st.mtp_pos_dev,
                    .host_pos = c.st.mtp_len, .nch = cdiv(c.st.mtp_len + n, 512), .R = n,
                }, stream);
            },
            .mtp_mid => {
                const m = c.m.mtp;
                try mtp.afterAttention(g, cfg, m, p.mb, c.lay.world, n, stream);
                try moe.moeBlock(g, c.k.moe, c.lay, &m.moe, p.ms, p.mb.normed.ptr, p.mb.part.ptr, p.mb.sk.ptr, n, stream);
            },
            .mtp_end => try mtp.head(g, c.k.mtp, cfg, c.m.mtp, p.mb, c.lay.world, n, stream),
            .commit => {
                // forward.py `commit` of a prompt chunk: every row kept, the KDA layers' states committed by their chains
                try forward.kdaCommit(c.ctx(), c.st.kda, R, R, true, stream);
                c.st.pos += R;
                c.st.mtp_len += n;
            },
            .gather, .wait => unreachable,
        }
    }

    /// A block's row-independent start on rows lo .. hi: KDA's projections, DSA's projection / norms / query, the
    /// dense MLP's gate/up and SwiGLU, the MoE's shared expert (forward.py `kda_front`, `dsa_front`, `mlp_front`,
    /// `shared_front`).
    fn runFront(c: *const Chunk, l: usize, site: Site, lo: usize, hi: usize, stream: cuda.Stream) !void {
        const p = c.p;
        const b = p.b;
        const cfg = c.lay.cfg;
        const D = cfg.hidden;
        const g = c.k.g;
        const n = hi - lo;
        const layer = &c.m.layers[l];
        const x = b.normed.ptr + lo * D * 2;
        switch (site) {
            .mixer => switch (layer.mixer) {
                .kda => |k| {
                    const width = c.lay.kdaProj();
                    const offs = c.lay.kdaOffsets();
                    const LW = k.w.heads * kda.dk;
                    const pr = b.kproj.ptr + lo * width * 2;
                    try g.bmm(x, D, k.w.proj.b16(), n, pr, false, b.sk.ptr, stream);
                    try g.bmm(pr + offs[0] * 2, width, k.w.fb.b16(), n, b.ka.ptr + lo * LW * 2, false, b.sk.ptr, stream);
                    try g.bmm(pr + offs[1] * 2, width, k.w.gb.b16(), n, b.kg.ptr + lo * LW * 2, false, b.sk.ptr, stream);
                },
                .dsa => |a| try dsa.dsaFront(g, c.lay, a.w, p.ds, b.normed.ptr, b.sk.ptr, lo, hi, stream),
            },
            .ffn => switch (layer.ffn) {
                .mlp => |w| {
                    try g.bmm(x, D, w.gu.b16(), n, b.gu.ptr + lo * 2 * w.width * 2, false, b.sk.ptr, stream);
                    try g.swiglu(b.gu.ptr + lo * 2 * w.width * 2, b.act.ptr + lo * w.width * 2, b.xs_act.ptr + lo * w.width / 64 * 4, cfg.limit, n, w.width, stream);
                },
                .moe => |w| {
                    const sh = &w.shared;
                    const ms = p.ms;
                    try g.bmm(x, D, sh.gu.b16(), n, ms.sgu.ptr + lo * 2 * sh.width * 2, false, b.sk.ptr, stream);
                    try g.swiglu(ms.sgu.ptr + lo * 2 * sh.width * 2, ms.sact.ptr + lo * sh.width * 2, ms.sxs.ptr + lo * sh.width / 64 * 4, cfg.limit, n, sh.width, stream);
                    try g.bmm(ms.sact.ptr + lo * sh.width * 2, sh.width, sh.down.b16(), n, ms.sy.ptr + lo * D * 4, true, b.sk.ptr, stream);
                },
            },
        }
    }

    /// What a block runs on every row between its fronts and its output: KDA's chain (the layer committed and its conv
    /// window shifted, forward.py `kda_rows`), DSA's `dsa_rows`, the MoE's routing and routed experts.
    fn runRest(c: *const Chunk, l: usize, site: Site, stream: cuda.Stream) !void {
        const p = c.p;
        const b = p.b;
        const cfg = c.lay.cfg;
        const R = c.rows();
        const g = c.k.g;
        const layer = &c.m.layers[l];
        switch (site) {
            .mixer => switch (layer.mixer) {
                .kda => |k| {
                    const st = c.st.kda;
                    const width = c.lay.kdaProj();
                    const offs = c.lay.kdaOffsets();
                    const LW = k.w.heads * kda.dk;
                    const sc = p.kscratch.layer(0);
                    try kda.chainWide(c.k.kda, p.temp, .{
                        .heads = k.w.heads, .p = b.kproj.ptr, .p_stride = width, .b_off = offs[2],
                        .a = b.ka.ptr, .a_stride = LW, .g = b.kg.ptr, .g_stride = LW,
                        .conv_state = st.convOf(k.li), .conv_w = k.w.conv.ptr, .state_in = st.state(st.cur, k.li), .a_log = k.w.a_log.ptr,
                        .dt_bias = k.w.dt_bias.ptr, .norm_w = k.w.norm.ptr, .eps = cfg.eps, .lower = cfg.lower,
                        .rows = R, .out = sc.out, .state_out = st.state(1 - st.cur, k.li),
                        .k = sc.k, .v = sc.v, .gs = sc.g, .b = sc.b,
                    }, stream);
                    const C = 3 * LW;
                    try g.convShift(st.convOf(k.li), b.kproj.ptr, R, 1, C, 3 * C, b.rows * width, width, stream);
                },
                .dsa => |a| try dsa.dsaRows(g, c.lay, a.w, &c.st.caches[a.di], p.ds, c.dsaCall(), stream),
            },
            .ffn => switch (layer.ffn) {
                .mlp => {},
                .moe => |w| {
                    const ms = p.ms;
                    try g.router(b.normed.ptr, cfg.hidden, w.router.ptr, cfg.experts, R, ms.logits.ptr, ms.rpart.ptr, stream);
                    try g.select(ms.logits.ptr, w.bias.ptr, ms.pick.ptr, ms.wts.ptr, cfg.top_k, cfg.experts, cfg.routed_scale, cfg.norm_topk, R, stream);
                    try moe.route(c.k.moe, ms, R, cfg.experts, stream);
                    try moe.routed(c.k.moe, &w.experts, ms, b.normed.ptr, cfg.hidden, R, cfg.limit, stream);
                },
            },
        }
    }

    /// A block's output on rows lo .. hi into this rank's fp32 partial: the output projection (o_proj, down), or
    /// the MoE's weighted sum of its routed slots and shared expert.
    fn fillRows(c: *const Chunk, l: usize, site: Site, lo: usize, hi: usize, stream: cuda.Stream) !void {
        const p = c.p;
        const b = p.b;
        const D = c.lay.cfg.hidden;
        const g = c.k.g;
        const n = hi - lo;
        const part = b.part.ptr + lo * D * 4;
        const layer = &c.m.layers[l];
        switch (site) {
            .mixer => switch (layer.mixer) {
                .kda => |k| {
                    const LW = k.w.heads * kda.dk;
                    try g.bmm(p.kscratch.layer(0).out + lo * LW * 2, LW, k.w.o.b16(), n, part, true, b.sk.ptr, stream);
                },
                .dsa => |a| try dsa.dsaOut(g, c.lay, a.w, p.ds, b.part.ptr, b.sk.ptr, lo, hi, stream),
            },
            .ffn => switch (layer.ffn) {
                .mlp => |w| try g.bmm(b.act.ptr + lo * w.width * 2, w.width, w.down.b16(), n, part, true, b.sk.ptr, stream),
                .moe => {
                    const ms = p.ms;
                    try g.combineShared(ms.y.ptr + lo * ms.slots * D * 4, ms.sy.ptr + lo * D * 4, ms.wts.ptr + lo * ms.slots * 4, part, ms.slots, n, stream);
                },
            },
        }
    }
};

/// decode.prefill_chunk on one rank: `c.plan`'s ops, the gathers through `link` (on `overlap`'s side stream when
/// given, the chunk's stream waiting where it reads them).
pub fn prefillChunk(c: *const Chunk, gpa: std.mem.Allocator, link: Link, overlap: ?*Overlap, stream: cuda.Stream) !void {
    const ops = try c.plan(gpa);
    defer gpa.free(ops);
    for (ops) |op| switch (op) {
        .gather => |x| {
            const on = if (overlap) |o| blk: {
                try o.ready.record(stream);
                try o.side.wait(o.ready);
                break :blk o.side;
            } else stream;
            try link.gather(link.ctx, x.send, x.recv, x.bytes, on);
            if (x.copy) |cp| {
                const dst: Buf = .{ .d = c.p.b.x.d, .ptr = cp.dst, .len = cp.bytes };
                try dst.copyFrom(0, cp.src, cp.bytes, on.handle);
            }
            if (overlap) |o| try o.ev[x.ev].record(o.side);
        },
        .wait => |e| if (overlap) |o| try stream.wait(o.ev[e]),
        else => try c.exec(op, stream),
    };
}

fn devInt(ptr: Ptr, d: *const cuda.Driver) Buf {
    return .{ .d = d, .ptr = ptr, .len = 4 };
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

test "a chunk's undone rows" {
    var buf: [2]Chunk.Range = undefined;
    try std.testing.expectEqual(@as(usize, 1), Chunk.undone(2048, .{ .lo = 0, .hi = 1024 }, &buf).len);
    try std.testing.expectEqual(@as(usize, 1024), Chunk.undone(2048, .{ .lo = 0, .hi = 1024 }, &buf)[0].lo);
    try std.testing.expectEqual(@as(usize, 0), Chunk.undone(2047, .{ .lo = 1024, .hi = 2048 }, &buf)[0].lo);
    try std.testing.expectEqual(@as(usize, 1), Chunk.undone(2047, .{ .lo = 1024, .hi = 2048 }, &buf).len);
}
