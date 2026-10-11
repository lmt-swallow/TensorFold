//! GLM-5.3-Flash's prompt path on one rank: the Mia fork's `decode.prefill` / `prefill_chunk` in the strict setting.
//! A prompt is committed in chunks of up to `Prompt.rows` rows (2,048 as served; a resumed prompt's first chunk
//! starts anywhere): the chunk's forward on the prompt buffers (every block's prompt-form kernels: `prefill` true),
//! the head's logits of its last row, the MTP layer's absorb of its rows (the next token of each, so a prompt's last
//! row waits for its first sampled token), then the commit (the KDA layers committed inside the forward).
//!
//! A chunk runs as `Chunk.sites() + 1` segments: segment i ends in this rank's fp32 partial `exchange(i)`, which the
//! ranks all-gather (rank 0 first) before segment i + 1; the last segment ends the chunk. `prefillChunk` runs them
//! with a `Gather`; a test runs two ranks' segments in turn on one GPU. Rows never mix, so a row's bits are those it
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
    }

    pub fn view(p: *const Owned) Prompt {
        return .{
            .rows = p.rows, .b = &p.b, .temp = &p.temp, .kscratch = &p.kscratch, .ds = &p.ds, .ms = &p.ms, .mb = &p.mb,
            .fnormed = p.fnormed.ptr, .fxs = p.fxs.ptr, .logits = p.logits.ptr,
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

/// What a segment leaves to all-gather: this rank's partial [rows, D] fp32 into every rank's `gath`.
pub const Exchange = struct { part: Ptr, gath: Ptr, bytes: usize };

pub const Gather = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, x: Exchange, stream: cuda.Stream) anyerror!void,
};

/// What segment i of a chunk runs (`Chunk.kind`).
pub const Kind = union(enum) {
    mixer: usize, // layer l - 1's end (layer 0: the embedding), layer l's hc_pre and token mixer
    ffn: usize, // layer l's mixer end, its FFN hc_pre and FFN
    tail, // the last layer's end, the streams' mean, the final norm, the head; then the MTP front and attention
    mtp_ffn, // the MTP layer's residual, post-norm and MoE
    mtp_end, // its residual and draft head
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

    pub fn rows(c: *const Chunk) usize {
        return c.ids.len;
    }

    /// The all-gathers the chunk takes: two a layer, and two for the MTP absorb when it has rows. Segments 0 ..
    /// sites() run in order, each but the last followed by its all-gather; the last commits.
    pub fn sites(c: *const Chunk) usize {
        return 2 * c.m.layers.len + @as(usize, if (c.next.len > 0) 2 else 0);
    }

    pub fn kind(c: *const Chunk, i: usize) Kind {
        const L = c.m.layers.len;
        if (i < 2 * L) return if (i % 2 == 0) .{ .mixer = i / 2 } else .{ .ffn = i / 2 };
        return switch (i - 2 * L) {
            0 => .tail,
            1 => .mtp_ffn,
            else => .mtp_end,
        };
    }

    /// The partial segment i (< sites()) leaves to all-gather.
    pub fn exchange(c: *const Chunk, i: usize) Exchange {
        const D = c.lay.cfg.hidden;
        return switch (c.kind(i)) {
            .mixer, .ffn => .{ .part = c.p.b.part.ptr, .gath = c.p.b.gath.ptr, .bytes = c.rows() * D * 4 },
            .tail, .mtp_ffn => .{ .part = c.p.mb.part.ptr, .gath = c.p.mb.gath.ptr, .bytes = c.next.len * D * 4 },
            .mtp_end => unreachable,
        };
    }

    /// forward.py `check_room` and the buffers' bound.
    pub fn check(c: *const Chunk) !void {
        const R = c.rows();
        if (R == 0 or R > c.p.rows) return error.TooManyRows;
        if (c.next.len != R and c.next.len + 1 != R) return error.BadNextTokens;
        if (c.st.pos + R > c.st.capacity or c.st.mtp_len + c.next.len > c.st.capacity) return error.PastCapacity;
    }

    fn ctx(c: *const Chunk) forward.Ctx {
        return .{ .g = c.k.g, .kk = c.k.kda, .temp = c.p.temp, .prompt_scratch = c.p.kscratch, .lay = c.lay };
    }

    fn mixer(c: *const Chunk, l: *const Layer, stream: cuda.Stream) !void {
        const p = c.p;
        const R = c.rows();
        try forward.hcPre(c.ctx(), l.frame, false, p.b, R, stream);
        switch (l.mixer) {
            .kda => |x| try forward.kdaBlock(c.ctx(), x.w, c.st.kda, x.li, p.b, R, stream),
            .dsa => |x| try dsa.dsaBlock(c.k.g, c.lay, x.w, &c.st.caches[x.di], p.ds, .{
                .normed = p.b.normed.ptr, .part = p.b.part.ptr, .sk = p.b.sk.ptr, .pos_dev = c.st.pos_dev,
                .host_pos = c.st.pos, .nch = cdiv(c.st.pos + R, 512), .R = R,
            }, stream),
        }
    }

    /// Segment i: the work up to exchange(i) (i < sites()), or the chunk's end and its commit (i == sites()).
    pub fn run(c: *const Chunk, i: usize, stream: cuda.Stream) !void {
        const p = c.p;
        const cfg = c.lay.cfg;
        const D = cfg.hidden;
        const R = c.rows();
        const layers = c.m.layers;
        const n = c.next.len;
        switch (c.kind(i)) {
            .mixer => |li| {
                if (li == 0) {
                    try c.check();
                    const pos: i32 = @intCast(c.st.pos);
                    const mpos: i32 = @intCast(c.st.mtp_len);
                    try devInt(c.st.pos_dev, p.b.ids.d).uploadAsync(0, std.mem.asBytes(&pos), stream.handle);
                    try devInt(c.st.mtp_pos_dev, p.b.ids.d).uploadAsync(0, std.mem.asBytes(&mpos), stream.handle);
                    try p.b.ids.uploadAsync(0, std.mem.sliceAsBytes(c.ids), stream.handle);
                    if (n > 0) try p.mb.ids.uploadAsync(0, std.mem.sliceAsBytes(c.next), stream.handle);
                    try stream.synchronize(); // the host slices need not outlive the call
                    try c.k.g.embed(p.b.ids.ptr, c.m.embed, p.b.x.ptr, R, cfg.streams, stream);
                } else {
                    try forward.hcPost(c.ctx(), p.b, R, stream);
                }
                try c.mixer(&layers[li], stream);
            },
            .ffn => |li| {
                const l = &layers[li];
                try forward.hcPost(c.ctx(), p.b, R, stream);
                try forward.hcPre(c.ctx(), l.frame, true, p.b, R, stream);
                switch (l.ffn) {
                    .mlp => |w| try forward.mlpBlock(c.ctx(), w, p.b, R, stream),
                    .moe => |w| try moe.moeBlock(c.k.g, c.k.moe, c.lay, w, p.ms, p.b.normed.ptr, p.b.part.ptr, p.b.sk.ptr, R, stream),
                }
            },
            .tail => {
                try forward.hcPost(c.ctx(), p.b, R, stream);
                try c.k.g.streamMean(p.b.x.ptr, p.b.hidden.ptr, R, stream);
                try c.k.g.rmsnorm(p.b.hidden.ptr, D, c.m.norm, p.fnormed, D, p.fxs, cfg.eps, R, D, stream);
                // the head reads the last row only (forward.py `_compute` of a prompt buffer)
                if (c.head) try c.k.g.bmm(p.fnormed + (R - 1) * D * 2, D, c.m.head, 1, p.logits, false, p.b.sk.ptr, stream);
                if (n == 0) return c.commit(stream);
                // decode._absorb_rows: the chunk's final-normed rows into the MTP head (mtp.py `mtp_stage`)
                const hin: Buf = .{ .d = p.mb.hin.d, .ptr = p.mb.hin.ptr, .len = n * D * 2 };
                try hin.copyFrom(0, p.fnormed, n * D * 2, stream.handle);
                const m = c.m.mtp;
                try mtp.front(c.k.g, cfg, m, p.mb, c.m.embed, n, c.st.mtp_len == 0, stream);
                try dsa.dsaBlock(c.k.g, c.lay, &m.dsa, c.st.mtp_cache, p.ds, .{
                    .normed = p.mb.normed.ptr, .part = p.mb.part.ptr, .sk = p.mb.sk.ptr, .pos_dev = c.st.mtp_pos_dev,
                    .host_pos = c.st.mtp_len, .nch = cdiv(c.st.mtp_len + n, 512), .R = n,
                }, stream);
            },
            .mtp_ffn => {
                const m = c.m.mtp;
                try mtp.afterAttention(c.k.g, cfg, m, p.mb, c.lay.world, n, stream);
                try moe.moeBlock(c.k.g, c.k.moe, c.lay, &m.moe, p.ms, p.mb.normed.ptr, p.mb.part.ptr, p.mb.sk.ptr, n, stream);
            },
            .mtp_end => {
                try mtp.head(c.k.g, c.k.mtp, cfg, c.m.mtp, p.mb, c.lay.world, n, stream);
                try c.commit(stream);
            },
        }
    }

    /// forward.py `commit` of a prompt chunk: every row kept, the KDA layers' states committed by their chains.
    fn commit(c: *const Chunk, stream: cuda.Stream) !void {
        const R = c.rows();
        try forward.kdaCommit(c.ctx(), c.st.kda, R, R, true, stream);
        c.st.pos += R;
        c.st.mtp_len += c.next.len;
    }
};

/// decode.prefill_chunk on one rank: every segment, each partial all-gathered with `gather`.
pub fn prefillChunk(c: *const Chunk, gather: Gather, stream: cuda.Stream) !void {
    const n = c.sites();
    for (0..n + 1) |i| {
        try c.run(i, stream);
        if (i < n) try gather.run(gather.ctx, c.exchange(i), stream);
    }
}

fn devInt(ptr: Ptr, d: *const cuda.Driver) Buf {
    return .{ .d = d, .ptr = ptr, .len = 4 };
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

test "a chunk's segments" {
    var layers: [2]Layer = undefined;
    var m: Model = undefined;
    m.layers = &layers;
    var c: Chunk = undefined;
    c.m = &m;
    c.ids = &[_]u32{ 1, 2, 3 };
    c.next = &[_]u32{ 2, 3 };
    try std.testing.expectEqual(@as(usize, 6), c.sites());
    try std.testing.expectEqual(Kind{ .mixer = 1 }, c.kind(2));
    try std.testing.expectEqual(Kind.tail, c.kind(4));
    try std.testing.expectEqual(Kind.mtp_ffn, c.kind(5));
    try std.testing.expectEqual(Kind.mtp_end, c.kind(6));
    c.next = &.{};
    try std.testing.expectEqual(@as(usize, 4), c.sites());
}
