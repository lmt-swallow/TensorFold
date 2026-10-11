//! Flash Next's sparse attention for a decode window (TensorFold 0.6.6 `forward.py: attn_block` with one stream,
//! `attention.py`, `glue.py: attn_prep / attn_gate`), on the captured Triton kernels: the projection row's q, k and
//! indexer heads normalized and rotated into the caches at the stream's positions, the indexer's pooled blocks and,
//! past the budget, each row's selected blocks, chunked attention over a row's keys (all of them below the budget,
//! its selected blocks and tail past it) merged in chunk order, and the gate.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const Config = @import("config.zig").Config;

const Ptr = cuda.abi.DevicePtr;

/// attention.py CHUNK: keys a chunk program reads.
pub const chunk = 512;
/// attention.py SELECT_REGS: block counts `_select` holds in registers (past it `_select_tiles`).
pub const select_regs = 32768;

/// One attention layer's caches for a stream (state.py: KVCache bf16, ikc, pooled).
pub const Caches = struct {
    k: cuda.DeviceBuffer, // bf16 [capacity, kv_heads, head_dim]
    v: cuda.DeviceBuffer,
    scale: cuda.DeviceBuffer, // fp16 [1]: a bf16 cache's one-element scales (the kernels take one argument list)
    ik: cuda.DeviceBuffer, // bf16 [capacity, index_dim]: the indexer's keys
    pooled: cuda.DeviceBuffer, // bf16 [capacity / ratio, index_dim]: the indexer's pooled blocks
    capacity: usize,

    pub fn init(d: *const cuda.Driver, cfg: *const Config, capacity: usize) !Caches {
        var c: Caches = undefined;
        c.capacity = capacity;
        const nb = (capacity + cfg.index_ratio - 1) / cfg.index_ratio;
        const sizes = [_]struct { *cuda.DeviceBuffer, usize }{
            .{ &c.k, capacity * cfg.kv_heads * cfg.head_dim * 2 }, .{ &c.v, capacity * cfg.kv_heads * cfg.head_dim * 2 },
            .{ &c.scale, 16 },                                       .{ &c.ik, capacity * cfg.index_dim * 2 },
            .{ &c.pooled, nb * cfg.index_dim * 2 },
        };
        var made: usize = 0;
        errdefer for (sizes[0..made]) |s| s[0].free();
        for (sizes) |s| {
            s[0].* = try cuda.DeviceBuffer.alloc(d, s[1]);
            try s[0].fill8(0, null);
            made += 1;
        }
        return c;
    }

    pub fn deinit(self: *Caches) void {
        for ([_]*cuda.DeviceBuffer{ &self.k, &self.v, &self.scale, &self.ik, &self.pooled }) |b| b.free();
    }
};

/// attention.py `AttnScratch` for windows of `rows` rows over caches of `capacity` positions.
pub const Scratch = struct {
    nch: usize,
    po: cuda.DeviceBuffer, // fp32 [rows, nch, heads, head_dim]
    pm: cuda.DeviceBuffer, // fp32 [rows, nch, heads]
    pl: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer, // bf16 [rows, heads, head_dim]
    budget: usize,
    ratio: usize,
    idw: usize,
    qsa: bool,
    nb: usize,
    ids: cuda.DeviceBuffer, // int32 [rows, idw]
    nk: cuda.DeviceBuffer, // int32 [rows]
    sparse: cuda.DeviceBuffer, // int32 [rows]
    scores: cuda.DeviceBuffer, // fp32 [rows, nb] (qsa)

    pub fn init(d: *const cuda.Driver, cfg: *const Config, rows: usize, capacity: usize) !Scratch {
        var s: Scratch = undefined;
        s.budget = cfg.index_budget;
        s.ratio = cfg.index_ratio;
        s.nch = (@min(capacity, s.budget + s.ratio - 1) + chunk - 1) / chunk;
        s.idw = s.budget + s.ratio;
        s.qsa = capacity > s.budget;
        s.nb = (capacity + s.ratio - 1) / s.ratio;
        const h = cfg.heads;
        const hd = cfg.head_dim;
        const sizes = [_]struct { *cuda.DeviceBuffer, usize }{
            .{ &s.po, rows * s.nch * h * hd * 4 }, .{ &s.pm, rows * s.nch * h * 4 }, .{ &s.pl, rows * s.nch * h * 4 },
            .{ &s.out, rows * h * hd * 2 },         .{ &s.ids, rows * s.idw * 4 },    .{ &s.nk, rows * 4 },
            .{ &s.sparse, rows * 4 },               .{ &s.scores, @max(16, rows * s.nb * 4) },
        };
        var made: usize = 0;
        errdefer for (sizes[0..made]) |x| x[0].free();
        for (sizes) |x| {
            x[0].* = try cuda.DeviceBuffer.alloc(d, x[1]);
            try x[0].fill8(0, null);
            made += 1;
        }
        return s;
    }

    pub fn deinit(self: *Scratch) void {
        for ([_]*cuda.DeviceBuffer{ &self.po, &self.pm, &self.pl, &self.out, &self.ids, &self.nk, &self.sparse, &self.scores }) |b| b.free();
    }
};

pub const Kernels = struct {
    set: *const aot.Set,
    cfg: *const Config,

    fn n32(v: usize) i32 {
        return @intCast(v);
    }

    /// glue.py `attn_prep` (text positions, bf16 caches): q [rows, heads, head_dim] and iq [rows, index_heads,
    /// index_dim] normalized and rotated; keys, values and raw indexer keys stored at positions pos0 + r.
    pub fn prep(k: Kernels, p: Ptr, pw: usize, pos0: Ptr, q_scale: Ptr, k_scale: Ptr, iq_scale: Ptr, inv: Ptr, q: Ptr, c: *const Caches, iq: Ptr, rows: usize, stream: cuda.Stream) !void {
        const cfg = k.cfg;
        try k.set.run(stream, "_attn_prep", .{ @intCast(rows), @intCast(cfg.heads + cfg.kv_heads + cfg.index_heads + 1), 1 }, &.{
            aot.ptr("P", "*bf16", p),        aot.ptr("POS0", "*i32", pos0),    aot.ptr("QW", "*fp32", q_scale),
            aot.ptr("KW", "*fp32", k_scale), aot.ptr("IW", "*fp32", iq_scale), aot.ptr("INV", "*fp32", inv),
            aot.ptr("Q", "*bf16", q),        aot.ptr("KC", "*bf16", c.k.ptr),  aot.ptr("VC", "*bf16", c.v.ptr),
            aot.ptr("KS", "*fp16", c.scale.ptr), aot.ptr("VS", "*fp16", c.scale.ptr), aot.ptr("IQ", "*bf16", iq),
            aot.ptr("IKC", "*bf16", c.ik.ptr), aot.ptr("ROPE", "*i32", pos0),  aot.ptr("DELTA", "*i32", pos0),
            aot.int("length", 0),            aot.float("eps", cfg.eps),
        }, &.{
            aot.ci("PW", @intCast(pw)),         aot.ci("NQ", @intCast(cfg.heads)),       aot.ci("NKV", @intCast(cfg.kv_heads)),
            aot.ci("HD", @intCast(cfg.head_dim)), aot.ci("NI", @intCast(cfg.index_heads)), aot.ci("IHD", @intCast(cfg.index_dim)),
            aot.ci("HALF", @intCast(cfg.rotary_dim / 2)), aot.ci("BITS", 0),          aot.ci("MODE", 0),
            aot.ci("S1", 11),                   aot.ci("S2", 10),
        });
    }

    /// attention.py `qsa_pool` (text positions): every block of `ratio` indexer keys the window completes, pooled.
    pub fn pool(k: Kernels, c: *const Caches, pos0: Ptr, ik_scale: Ptr, inv: Ptr, s: *const Scratch, rows: usize, stream: cuda.Stream) !void {
        const cfg = k.cfg;
        try k.set.run(stream, "_pool", .{ @intCast(rows / s.ratio + 2), 1, 1 }, &.{
            aot.ptr("IKC", "*bf16", c.ik.ptr), aot.ptr("POOLED", "*bf16", c.pooled.ptr), aot.ptr("POS0", "*i32", pos0),
            aot.ptr("W", "*fp32", ik_scale),   aot.ptr("INV", "*fp32", inv),             aot.float("eps", cfg.eps),
            aot.int("R", n32(rows)),           aot.int("length", 0),
        }, &.{
            aot.ci("DI", @intCast(cfg.index_dim)), aot.ci("HALF", @intCast(cfg.rotary_dim / 2)), aot.ci("RATIO", @intCast(s.ratio)),
            aot.cnone("ROPE"),                     aot.cnone("DELTA"),                            aot.ci("MODE", 0),
            aot.ci("S1", 11),                      aot.ci("S2", 10),
        });
    }

    /// attention.py `qsa_rows`: score each sparse row's complete blocks and list its selected blocks (scratch.ids,
    /// nk, sparse); `context` bounds the launches to the keys in use.
    pub fn rowsSelect(k: Kernels, iq: Ptr, c: *const Caches, pos0: Ptr, s: *const Scratch, rows: usize, context: usize, stream: cuda.Stream) !void {
        const cfg = k.cfg;
        const top = s.budget / s.ratio;
        const bb: usize = 64;
        const blocks = @min(s.nb, @max(1, (context + s.ratio - 1) / s.ratio));
        try k.set.run(stream, "_scores", .{ @intCast(rows), @intCast((blocks + bb - 1) / bb), 1 }, &.{
            aot.ptr("IQ", "*bf16", iq), aot.ptr("POOLED", "*bf16", c.pooled.ptr), aot.ptr("POS0", "*i32", pos0),
            aot.ptr("SC", "*fp32", s.scores.ptr), aot.int("NB", n32(s.nb)),
        }, &.{ aot.ci("HI", @intCast(cfg.index_heads)), aot.ci("DI", @intCast(cfg.index_dim)), aot.ci("RATIO", @intCast(s.ratio)), aot.ci("TOP", @intCast(top)), aot.ci("BB", @intCast(bb)) });
        const width = std.math.ceilPowerOfTwoAssert(usize, blocks);
        if (width > select_regs) return error.SelectTilesNotPorted; // _select_tiles: past 131,072 keys
        try k.set.run(stream, "_select", .{ @intCast(rows), 1, 1 }, &.{
            aot.ptr("SC", "*fp32", s.scores.ptr), aot.ptr("POS0", "*i32", pos0), aot.ptr("IDS", "*i32", s.ids.ptr),
            aot.ptr("NKR", "*i32", s.nk.ptr),     aot.ptr("SPR", "*i32", s.sparse.ptr), aot.int("NB", n32(s.nb)),
        }, &.{ aot.ci("RATIO", @intCast(s.ratio)), aot.ci("TOP", @intCast(top)), aot.ci("IDW", @intCast(s.idw)), aot.ci("BLOCK", @intCast(width)) });
    }

    /// attention.py `attention` (bf16 caches): q [rows, heads, head_dim] against each row's keys -> s.out bf16.
    pub fn attend(k: Kernels, q: Ptr, c: *const Caches, pos0: Ptr, s: *const Scratch, rows: usize, context: usize, stream: cuda.Stream) !void {
        return k.attendInto(q, c, pos0, s, rows, context, s.out.ptr, stream);
    }

    /// `attend` merging into `out` (a prompt block's rows of the chunk's output).
    pub fn attendInto(k: Kernels, q: Ptr, c: *const Caches, pos0: Ptr, s: *const Scratch, rows: usize, context: usize, out: Ptr, stream: cuda.Stream) !void {
        const cfg = k.cfg;
        const g = cfg.heads / cfg.kv_heads;
        var keys = context;
        if (s.qsa) keys = @min(keys, (s.budget / s.ratio + 1) * s.ratio - 1);
        const chunks = @min(s.nch, (keys + chunk - 1) / chunk);
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(cfg.head_dim))));
        const consts = [_]aot.Const{
            aot.ci("H", @intCast(cfg.heads)), aot.ci("HK", @intCast(cfg.kv_heads)), aot.ci("D", @intCast(cfg.head_dim)), aot.ci("G", @intCast(g)),
            aot.ci("CH", chunk),              aot.ci("NCH", @intCast(s.nch)),       aot.ci("QSA", @intFromBool(s.qsa)),  aot.ci("BITS", 0),
        };
        try k.set.run(stream, "_chunks", .{ @intCast(rows), @intCast(cfg.kv_heads), @intCast(chunks) }, &.{
            aot.ptr("Q", "*bf16", q),          aot.ptr("KC", "*bf16", c.k.ptr),       aot.ptr("VC", "*bf16", c.v.ptr),
            aot.ptr("KSC", "*fp16", c.scale.ptr), aot.ptr("VSC", "*fp16", c.scale.ptr), aot.ptr("POS0", "*i32", pos0),
            aot.ptr("PO", "*fp32", s.po.ptr),  aot.ptr("PM", "*fp32", s.pm.ptr),      aot.ptr("PL", "*fp32", s.pl.ptr),
            aot.ptr("IDS", "*i32", s.ids.ptr), aot.ptr("NKR", "*i32", s.nk.ptr),      aot.ptr("SPR", "*i32", s.sparse.ptr),
        }, &(consts ++ [_]aot.Const{ aot.cf("SCALE", scale), aot.ci("IDW", @intCast(s.idw)) }));
        try k.set.run(stream, "_merge", .{ @intCast(rows), @intCast(cfg.kv_heads), 1 }, &.{
            aot.ptr("PO", "*fp32", s.po.ptr), aot.ptr("PM", "*fp32", s.pm.ptr), aot.ptr("PL", "*fp32", s.pl.ptr),
            aot.ptr("POS0", "*i32", pos0),    aot.ptr("OUT", "*bf16", out),       aot.ptr("NKR", "*i32", s.nk.ptr),
            aot.ptr("SPR", "*i32", s.sparse.ptr),
        }, &consts);
    }

    /// glue.py `attn_gate`: out [rows, heads * head_dim] bf16 = o * sigmoid(gate) (the gate from the projection's
    /// [q | gate] pairs), and its 32-group sums.
    pub fn gate(k: Kernels, o: Ptr, p: Ptr, pw: usize, out: Ptr, xs: Ptr, rows: usize, stream: cuda.Stream) !void {
        const cfg = k.cfg;
        try k.set.run(stream, "_attn_gate", .{ @intCast(rows), @intCast(cfg.heads), 1 }, &.{
            aot.ptr("O", "*bf16", o), aot.ptr("P", "*bf16", p), aot.ptr("OUT", "*bf16", out), aot.ptr("XS", "*fp32", xs),
        }, &.{ aot.ci("PW", @intCast(pw)), aot.ci("NQ", @intCast(cfg.heads)), aot.ci("HD", @intCast(cfg.head_dim)) });
    }

    /// forward.py `attn_block`'s attention for one stream's window of `rows` rows at host position `pos` (pos0 its
    /// device copy): prep, the indexer's selection (qsa), chunked attention, the gate into `gated` / `xs_gated`.
    pub fn block(k: Kernels, p: Ptr, pw: usize, pos0: Ptr, pos: usize, scales: [4]Ptr, inv: Ptr, q: Ptr, iq: Ptr, c: *const Caches, s: *const Scratch, gated: Ptr, xs_gated: Ptr, rows: usize, stream: cuda.Stream) !void {
        const keys = pos + rows;
        try k.prep(p, pw, pos0, scales[0], scales[1], scales[2], inv, q, c, iq, rows, stream);
        if (s.qsa) {
            try k.pool(c, pos0, scales[3], inv, s, rows, stream);
            try k.rowsSelect(iq, c, pos0, s, rows, keys, stream);
        }
        try k.attend(q, c, pos0, s, rows, keys, stream);
        try k.gate(s.out.ptr, p, pw, gated, xs_gated, rows, stream);
    }
};
