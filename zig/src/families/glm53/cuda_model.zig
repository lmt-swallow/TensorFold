//! GLM-5.3-Flash's whole forward on one rank (the Mia fork's `forward.py: _compute` and `commit`, strict setting):
//! the token rows embedded into every hyper-connection stream, the 45 layers (KDA or DSA, then the dense MLP or the
//! MoE, each ending in this rank's fp32 partial that the ranks all-gather), the streams' mean, the final norm and the
//! rank's head rows. Prompt chunks run unsplit (TF_GLM_HC_SPLIT=0): the row split moves the same rows' partials
//! between the ranks and gives the same bits (B1's forward fixtures, split and unsplit, agree byte for byte).
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const map = @import("weights_map.zig");
const glue_mod = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const weights = @import("cuda_weights.zig");
const F = @import("cuda_forward.zig");
const A = @import("cuda_dsa.zig");
const M = @import("cuda_moe.zig");

const Glue = glue_mod.Glue;
const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

pub const Layer = struct {
    index: usize,
    frame: weights.Frame,
    kda: ?weights.Kda = null,
    dsa: ?A.Dsa = null,
    mlp: ?weights.Mlp = null,
    moe: ?M.Moe = null,

    pub fn deinit(l: *Layer) void {
        l.frame.deinit();
        if (l.kda) |*x| x.deinit();
        if (l.dsa) |*x| x.deinit();
        if (l.mlp) |*x| x.deinit();
        if (l.moe) |*x| x.deinit();
    }
};

/// One rank's model: the backbone layers, the embedding, the final norm, the rank's head rows.
pub const Model = struct {
    gpa: std.mem.Allocator,
    layers: []Layer,
    embed: Buf,
    norm: Buf,
    head: weights.Matrix, // [vocab / world, D]

    /// Layers 0 .. `count` (all: the config's) of rank `l.lay`; the page cache dropped after each layer.
    pub fn load(l: weights.Loader, count: usize) !Model {
        const c = l.lay.cfg;
        const layers = try l.gpa.alloc(Layer, count);
        var made: usize = 0;
        errdefer {
            for (layers[0..made]) |*x| x.deinit();
            l.gpa.free(layers);
        }
        for (layers, 0..) |*x, i| {
            x.* = .{ .index = i, .frame = try weights.Frame.load(l, i) };
            switch (c.kind(i)) {
                .kda => x.kda = try weights.Kda.load(l, i),
                .dsa => x.dsa = try A.Dsa.load(l, i),
            }
            switch (c.mlpKind(i)) {
                .dense => x.mlp = try weights.Mlp.load(l, "mlp.", i),
                .moe => x.moe = try M.Moe.load(l, i),
            }
            made += 1;
            l.pack.dropCache();
        }
        var embed = try l.share(map.embed);
        errdefer embed.free();
        var norm = try l.share(map.norm);
        errdefer norm.free();
        const rows = try l.headRows();
        defer l.gpa.free(rows);
        const head: weights.Matrix = .{ .buf = try Buf.fromHost(l.d, rows), .n = rows.len / 2 / c.hidden, .k = c.hidden };
        l.pack.dropCache();
        return .{ .gpa = l.gpa, .layers = layers, .embed = embed, .norm = norm, .head = head };
    }

    pub fn deinit(m: *Model) void {
        for (m.layers) |*x| x.deinit();
        m.gpa.free(m.layers);
        m.embed.free();
        m.norm.free();
        m.head.deinit();
    }
};

/// One stream's state: the KDA layers' recurrent states and conv windows, the DSA layers' caches, the position.
pub const State = struct {
    gpa: std.mem.Allocator,
    kda: F.KdaState,
    caches: []A.Cache,
    kda_index: []usize, // layer -> its index among the KDA layers (or among the DSA layers)
    pos: usize = 0,
    pos_dev: Buf,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, layers: usize, capacity: usize, rows: usize, prompt_rows: usize) !State {
        const c = lay.cfg;
        var nk: usize = 0;
        var nd: usize = 0;
        const index = try gpa.alloc(usize, layers);
        errdefer gpa.free(index);
        for (0..layers) |i| switch (c.kind(i)) {
            .kda => {
                index[i] = nk;
                nk += 1;
            },
            .dsa => {
                index[i] = nd;
                nd += 1;
            },
        };
        var ks = try F.KdaState.init(d, lay, @max(nk, 1), rows);
        errdefer ks.deinit();
        try ks.rec.fill8(0, null);
        try ks.conv.fill8(0, null);
        const caches = try gpa.alloc(A.Cache, nd);
        var made: usize = 0;
        errdefer {
            for (caches[0..made]) |*x| x.deinit();
            gpa.free(caches);
        }
        for (caches) |*x| {
            x.* = try A.Cache.init(d, c, capacity, A.indexRing(capacity, @max(rows, prompt_rows)));
            made += 1;
        }
        return .{ .gpa = gpa, .kda = ks, .caches = caches, .kda_index = index, .pos_dev = try Buf.alloc(d, 16) };
    }

    pub fn deinit(s: *State) void {
        s.kda.deinit();
        for (s.caches) |*x| x.deinit();
        s.gpa.free(s.caches);
        s.gpa.free(s.kda_index);
        s.pos_dev.free();
    }
};

/// The buffers of windows of up to `rows` rows (decode) or a prompt chunk's `prompt_rows`.
pub const Work = struct {
    b: F.Buffers,
    pb: F.Buffers,
    ds: A.Scratch,
    pds: A.Scratch,
    ms: M.Scratch,
    pms: M.Scratch,
    temp: kda.Temp,
    pscratch: kda.Scratch,
    fnormed: Buf,
    fxs: Buf,
    logits: Buf, // [rows, vocab / world] bf16

    pub fn init(d: *const cuda.Driver, lay: config.Layout, rows: usize, prompt_rows: usize, capacity: usize) !Work {
        const c = lay.cfg;
        var w: Work = undefined;
        w.b = try F.Buffers.init(d, lay, rows, false);
        w.pb = try F.Buffers.init(d, lay, prompt_rows, true);
        w.ds = try A.Scratch.init(d, lay, rows, capacity, false);
        w.pds = try A.Scratch.init(d, lay, prompt_rows, capacity, true);
        w.ms = try M.Scratch.init(d, lay, rows, false);
        w.pms = try M.Scratch.init(d, lay, prompt_rows, true);
        w.temp = try kda.Temp.init(d, @max(rows, prompt_rows), lay.size(.lin));
        w.pscratch = try kda.Scratch.init(d, 1, prompt_rows, lay.size(.lin));
        w.fnormed = try Buf.alloc(d, prompt_rows * c.hidden * 2);
        w.fxs = try Buf.alloc(d, prompt_rows * c.hidden / 64 * 4);
        w.logits = try Buf.alloc(d, rows * lay.size(.vocab) * 2);
        return w;
    }

    pub fn deinit(w: *Work) void {
        w.b.deinit();
        w.pb.deinit();
        w.ds.deinit();
        w.pds.deinit();
        w.ms.deinit();
        w.pms.deinit();
        w.temp.deinit();
        w.pscratch.deinit();
        w.fnormed.free();
        w.fxs.free();
        w.logits.free();
    }
};

/// The all-gather of every rank's fp32 partial [rows, D] into [world, rows, D], rank 0 first (NCCL between the
/// Sparks; a test emulates it).
pub const Gather = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, part: Ptr, gath: Ptr, floats: usize, stream: cuda.Stream) anyerror!void,
};

/// A layer's end, for a caller that checks the streams (b.x after layer `index`).
pub const Hook = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, index: usize, x: Ptr, R: usize) anyerror!void,
};

pub const Kernels = struct { g: Glue, kk: *const kda.Kernels, mk: *const M.Kernels };

/// forward.py `_compute` of the window whose token ids are in (work's) b.ids[:R] at state position st.pos: every
/// layer, the streams' mean, the final norm and the head's logits (a decode window's every row into work.logits; a
/// prompt chunk's last row). The DSA caches take the rows at their positions; the KDA states the chain's (a prompt
/// chunk's layers commit as they go); `commit` keeps rows.
pub fn forward(k: Kernels, lay: config.Layout, m: *const Model, st: *State, w: *Work, R: usize, prefill: bool, gather: Gather, hook: ?Hook, stream: cuda.Stream) !void {
    const c = lay.cfg;
    const D = c.hidden;
    const b = if (prefill) &w.pb else &w.b;
    const cx: F.Ctx = .{ .g = k.g, .kk = k.kk, .temp = &w.temp, .prompt_scratch = &w.pscratch, .lay = lay };
    const p32: i32 = @intCast(st.pos);
    try st.pos_dev.uploadAsync(0, std.mem.asBytes(&p32), stream.handle);
    try k.g.embed(b.ids.ptr, m.embed.ptr, b.x.ptr, R, c.streams, stream);
    for (m.layers) |*layer| {
        try F.hcPre(cx, &layer.frame, false, b, R, stream);
        if (layer.kda) |*x| {
            try F.kdaBlock(cx, x, &st.kda, st.kda_index[layer.index], b, R, stream);
        } else {
            const di = st.kda_index[layer.index];
            try A.dsaBlock(k.g, lay, &layer.dsa.?, &st.caches[di], if (prefill) &w.pds else &w.ds, .{
                .normed = b.normed.ptr, .part = b.part.ptr, .sk = b.sk.ptr, .pos_dev = st.pos_dev.ptr,
                .host_pos = st.pos,     .nch = (st.pos + R + 511) / 512, .R = R,
            }, stream);
        }
        try gather.run(gather.ctx, b.part.ptr, b.gath.ptr, R * D, stream);
        try F.hcPost(cx, b, R, stream);
        try F.hcPre(cx, &layer.frame, true, b, R, stream);
        if (layer.mlp) |*x| {
            try F.mlpBlock(cx, x, b, R, stream);
        } else {
            try M.moeBlock(k.g, k.mk, lay, &layer.moe.?, if (prefill) &w.pms else &w.ms, b.normed.ptr, b.part.ptr, b.sk.ptr, R, stream);
        }
        try gather.run(gather.ctx, b.part.ptr, b.gath.ptr, R * D, stream);
        try F.hcPost(cx, b, R, stream);
        if (hook) |h| try h.run(h.ctx, layer.index, b.x.ptr, R);
    }
    try k.g.streamMean(b.x.ptr, b.hidden.ptr, R, stream);
    try k.g.rmsnorm(b.hidden.ptr, D, m.norm.ptr, w.fnormed.ptr, D, w.fxs.ptr, c.eps, R, D, stream);
    const first = if (prefill) R - 1 else 0;
    try k.g.bmm(w.fnormed.ptr + first * D * 2, D, m.head.b16(), R - first, w.logits.ptr, false, b.sk.ptr, stream);
}

/// forward.py `commit`: keep the window's first `keep` rows (a prompt chunk keeps all).
pub fn commit(k: Kernels, lay: config.Layout, st: *State, R: usize, keep: usize, prefill: bool, w: *Work, stream: cuda.Stream) !void {
    const cx: F.Ctx = .{ .g = k.g, .kk = k.kk, .temp = &w.temp, .prompt_scratch = &w.pscratch, .lay = lay };
    try F.kdaCommit(cx, &st.kda, R, keep, prefill, stream);
    st.pos = st.kda.pos;
}
