//! GLM-5.3-Flash's DSA layers (and the MTP layer's attention) on one rank, as the Mia fork's `forward.py: dsa_rows`
//! runs them on the latent cache in the strict setting (TF_GLM_KV=bf16 or fp8, TF_GLM_KVB=bf16): the projection and norms,
//! the latent row written at its position, the indexer's keys, gates and pooled keys, the query absorbed into the
//! 512-wide latent, attention (dense up to the dense limit; past it over each row's 512 best pools of 4 and its
//! incomplete pool), the read-out expanded through kv_b's value rows, o_proj into the rank's fp32 partial.
//!
//! Prompt chunks (prefill buffers) take other kernels than decode windows, each with its own bits and each
//! independent of the other rows: absorb / expand on the tensor cores, dense attention in blocks of 512 rows with
//! 32-head tiles, the pool selection in blocks of 512 rows scoring a multiple of 1,024 pools, one-pass sparse
//! attention. Decode windows score a power-of-two bucket of pools (`pool_bucket`) and select per row, split across
//! chunk programs from 8,192 pools on.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const map = @import("weights_map.zig");
const glue_mod = @import("cuda_glue.zig");
const weights = @import("cuda_weights.zig");
const tier_mod = @import("cuda_tier.zig");

const Glue = glue_mod.Glue;
pub const Kv = glue_mod.Kv;
const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

/// tensorfold/cuda/geometry.py MLA_PROMPT_ATT_ROWS and MLA_SELECT_ROWS.
pub const prompt_att_rows = 512;
pub const select_rows = 512;
/// Rows a prompt chunk's pool selection scores at a time. sparse.py's `_select_prompt` takes blocks of select_rows; a
/// row's scores, pools and tokens depend on that row alone (`_scores` keeps rows apart, the selection is per row), so
/// smaller blocks give the same bytes and shrink the scores scratch: [block, pools] fp32 is 512 MiB at 512 rows and a
/// 1M context (262,144 pools), 128 MiB at 128 rows.
pub const select_block = 128;
/// sparse.py PROMPT_ROWS (windows from this many rows take the prompt selection), SPLIT_FROM, SPLIT_ROWS,
/// ROW_SCORE_G; latent.py CHUNK.
const prompt_select_from = 64;
const split_from = 8192;
const split_rows = 64;
const row_score_g = 128;
const chunk = 512;

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// latent.py `head_block`: a rank's 32 heads a tile for windows of 64 rows or more, else 16.
pub fn headBlock(R: usize) usize {
    return if (R >= 64) 32 else 16;
}

/// sparse.py `pool_bucket`: pools to score for rows pos .. pos + R - 1 (decode): the visible ones rounded up to a
/// power of two, at least 1,024, at most `np_max`.
pub fn poolBucket(pos: usize, R: usize, np_max: usize) usize {
    const visible = (pos + R) / 4 + 1;
    return @min(np_max, @max(1024, std.math.ceilPowerOfTwoAssert(usize, visible)));
}

/// sparse.py `pool_count`: pools a prompt chunk scores, the visible ones rounded up to a multiple of 1,024.
pub fn poolCount(pos: usize, R: usize, np_max: usize) usize {
    const visible = (pos + R) / 4 + 1;
    return @min(np_max, cdiv(visible, 1024) * 1024);
}

/// A DSA layer's rank share (weights.py `DSAW` with `IndexW`, the latent path's `AbsorbW`).
pub const Dsa = struct {
    proj: weights.Matrix, // [q_a | kv_a] [q_lora + kv_lora, D]
    q_norm: Buf,
    kv_norm: Buf,
    q_b: weights.Matrix, // [H qk_dim, q_lora]
    wk: Buf, // [H, qk_dim, kv_lora] bf16: kv_b's key rows a head
    wv: Buf, // [H, v_dim, kv_lora]: its value rows
    o: weights.Matrix, // [D, H v_dim]
    kw: weights.Matrix, // [wk | weights_proj] [index_dim + index_heads, D]
    qb: weights.Matrix, // wq_b [index_heads index_dim, q_lora]
    ln_w: Buf,
    ln_b: Buf,
    gate: Buf, // [index_dim, D] bf16: the pool compression's gate rows (glue.router's weight)
    ape: Buf, // [kpool, index_dim] bf16
    heads: usize,

    pub fn load(l: weights.Loader, i: usize) !Dsa {
        const c = l.lay.cfg;
        var nb: [2]map.Name = @splat(.{});
        const sa = map.lm ++ "layers.{d}.self_attn.";
        var a: Dsa = undefined;
        a.heads = l.lay.size(.heads);
        a.proj = try l.matrix(&.{ nb[0].of(sa ++ "q_a_proj.weight", .{i}), nb[1].of(sa ++ "kv_a_proj_with_mqa.weight", .{i}) }, c.hidden);
        errdefer a.proj.deinit();
        a.q_norm = try l.share(nb[0].of(sa ++ "q_a_layernorm.weight", .{i}));
        errdefer a.q_norm.free();
        a.kv_norm = try l.share(nb[0].of(sa ++ "kv_a_layernorm.weight", .{i}));
        errdefer a.kv_norm.free();
        a.q_b = try l.matrix(&.{nb[0].of(sa ++ "q_b_proj.weight", .{i})}, c.q_lora);
        errdefer a.q_b.deinit();
        {
            // kv_b's rank rows [H (qk_dim + v_dim), kv_lora]: a head's key rows first, then its value rows
            const kvb = try l.hostShare(nb[0].of(sa ++ "kv_b_proj.weight", .{i}));
            defer l.gpa.free(kvb);
            const row = c.kv_lora * 2;
            const per = (c.qk_dim + c.v_dim) * row;
            if (kvb.len != a.heads * per) return error.UnexpectedTensor;
            a.wk = try Buf.alloc(l.d, a.heads * c.qk_dim * row);
            errdefer a.wk.free();
            a.wv = try Buf.alloc(l.d, a.heads * c.v_dim * row);
            errdefer a.wv.free();
            for (0..a.heads) |h| {
                try a.wk.upload(h * c.qk_dim * row, kvb[h * per ..][0 .. c.qk_dim * row]);
                try a.wv.upload(h * c.v_dim * row, kvb[h * per + c.qk_dim * row ..][0 .. c.v_dim * row]);
            }
        }
        errdefer a.wk.free();
        errdefer a.wv.free();
        a.o = try l.matrix(&.{nb[0].of(sa ++ "o_proj.weight", .{i})}, a.heads * c.v_dim);
        errdefer a.o.deinit();
        a.kw = try l.matrix(&.{ nb[0].of(sa ++ "indexer.wk.weight", .{i}), nb[1].of(sa ++ "indexer.weights_proj.weight", .{i}) }, c.hidden);
        errdefer a.kw.deinit();
        a.qb = try l.matrix(&.{nb[0].of(sa ++ "indexer.wq_b.weight", .{i})}, c.q_lora);
        errdefer a.qb.deinit();
        a.ln_w = try l.share(nb[0].of(sa ++ "indexer.k_norm.weight", .{i}));
        errdefer a.ln_w.free();
        a.ln_b = try l.share(nb[0].of(sa ++ "indexer.k_norm.bias", .{i}));
        errdefer a.ln_b.free();
        a.gate = try l.share(nb[0].of(sa ++ "indexer.index_kpool_compress_gate", .{i}));
        errdefer a.gate.free();
        a.ape = try l.share(nb[0].of(sa ++ "indexer.index_kpool_compress_ape", .{i}));
        return a;
    }

    pub fn deinit(a: *Dsa) void {
        a.proj.deinit();
        a.q_norm.free();
        a.kv_norm.free();
        a.q_b.deinit();
        a.wk.free();
        a.wv.free();
        a.o.deinit();
        a.kw.deinit();
        a.qb.deinit();
        a.ln_w.free();
        a.ln_b.free();
        a.gate.free();
        a.ape.free();
    }
};

/// A cache's latent rows on the tier: stream `h`'s layer `layer`.
pub const TierRef = struct { t: *tier_mod.Tier, h: usize, layer: usize };

/// One stream's caches of one DSA layer (forward.py `Caches` bound to a stream): the latent rows [cap, kv_lora], the
/// indexer's key and gate ring [ring, index_dim] bf16 (token t at t % ring) and the pooled keys [cap / 4 + 2,
/// index_dim]; the latent rows and pooled keys in the format `kv` (TF_GLM_KV: bf16, or fp8's rows of
/// `glue.Kv.rowBytes`). The latent and pooled-key rows are its own (`init`, zeroed: a zero FP8 row reads as zeros) or
/// a pool's extent (`rebind`: Mia's `Arena.view`, the pooled keys' 2 pad rows overlapping the next extent's first);
/// the rings are always its own.
pub const Cache = struct {
    latent: Buf,
    ik: Buf,
    ig: Buf,
    pk: Buf,
    capacity: usize,
    ring: usize,
    kv: Kv,
    owned: bool = true, // latent and pk are this cache's allocations, not a pool's
    /// The positions a stream may reach (a served stream's context, not its extent: 0 means `capacity`). sparse.py
    /// sizes its selection by the cache's own capacity, so the pool counts and their kernels depend on it; the scores
    /// read no pooled key past a row's visible ones, so a view smaller than this reads nothing outside itself.
    most: usize = 0,
    /// The latent rows on the tier (cuda_tier.zig: a slab of resident pools, the rest on NVMe) instead of `latent`:
    /// the latent write, dense and sparse attention take their addresses from it, the kernels and bits unchanged.
    tier: ?TierRef = null,

    pub fn init(d: *const cuda.Driver, c: *const config.Config, capacity: usize, ring: usize, kv: Kv) !Cache {
        var s = try rings(d, c, ring, kv);
        errdefer s.deinit();
        s.owned = true;
        s.latent = try Buf.alloc(d, capacity * kv.rowBytes(c.kv_lora));
        s.pk = try Buf.alloc(d, (capacity / 4 + 2) * kv.rowBytes(c.index_dim));
        for ([_]*Buf{ &s.latent, &s.pk }) |x| try x.fill8(0, null);
        s.capacity = capacity;
        return s;
    }

    /// The rings alone (zeroed); `rebind` points the latent and pooled-key rows at a pool's extent.
    pub fn rings(d: *const cuda.Driver, c: *const config.Config, ring: usize, kv: Kv) !Cache {
        var ik = try Buf.alloc(d, ring * c.index_dim * 2);
        errdefer ik.free();
        var ig = try Buf.alloc(d, ring * c.index_dim * 2);
        errdefer ig.free();
        for ([_]*Buf{ &ik, &ig }) |x| try x.fill8(0, null);
        const none: Buf = .{ .d = d, .ptr = 0, .len = 0 };
        return .{ .latent = none, .ik = ik, .ig = ig, .pk = none, .capacity = 0, .ring = ring, .kv = kv, .owned = false };
    }

    /// The device bytes `init` allocates for one layer's caches.
    pub fn bytes(c: *const config.Config, capacity: usize, ring: usize, kv: Kv) usize {
        return capacity * kv.rowBytes(c.kv_lora) + 2 * ring * c.index_dim * 2 + (capacity / 4 + 2) * kv.rowBytes(c.index_dim);
    }

    /// Latent row `pos`'s offset in `latent`.
    pub fn latentAt(s: *const Cache, c: *const config.Config, pos: usize) usize {
        return pos * s.kv.rowBytes(c.kv_lora);
    }

    /// The latent and pooled-key rows of a pool's extent of `capacity` positions (views; the pool owns them).
    pub fn rebind(s: *Cache, latent: Buf, pk: Buf, capacity: usize) void {
        s.dropOwned();
        s.latent = latent;
        s.pk = pk;
        s.capacity = capacity;
    }

    fn dropOwned(s: *Cache) void {
        if (!s.owned) return;
        s.latent.free();
        s.pk.free();
        s.owned = false;
    }

    pub fn deinit(s: *Cache) void {
        s.dropOwned();
        s.ik.free();
        s.ig.free();
    }

    /// sparse.py's np_max: the pools of the positions the stream may reach (`most`, else the cache's capacity).
    pub fn poolRows(s: *const Cache) usize {
        return (if (s.most != 0) s.most else s.capacity) / 4;
    }
};

/// forward.py `index_ring`: ring rows for windows of up to `rows` rows (and the 3 before them, rounded up to 64).
pub fn indexRing(capacity: usize, rows: usize) usize {
    return @min(capacity, cdiv(rows + 3, 64) * 64);
}

/// Streams a shared decode forward takes at most, and the rows of each one's window at most (TF_GLM_MAX_ROWS).
pub const max_segs = 4;
pub const seg_rows = 16;

/// What a DSA block writes besides the shared buffers, for windows of up to `rows` rows (`prefill`: a prompt
/// chunk's) over caches of up to `capacity` tokens.
pub const Scratch = struct {
    rows: usize,
    prefill: bool,
    dp: Buf, // [rows, q_lora + kv_lora] bf16
    qr: Buf, // [rows, q_lora]
    xs_qr: Buf,
    lat: Buf, // [rows, kv_lora]
    xs_lat: Buf,
    q: Buf, // [rows, H, qk_dim]
    ikr: Buf, // [rows, index_dim + index_heads] bf16
    igr: Buf, // [rows, index_dim] fp32
    gpart: Buf, // the gate router's K-slice partials [8, rows, index_dim] fp32
    qi: Buf, // [rows, index_heads index_dim] bf16
    qa: Buf, // [rows, H, kv_lora] bf16
    ol: Buf, // [rows, H, kv_lora]
    vn: Buf, // [rows, H v_dim] bf16
    dense: Glue.Partials, // [chunks, part_rows, H, kv_lora + 2] fp32
    dense_bufs: [3]Buf,
    nch: usize, // dense chunks the partials hold
    part_rows: usize,
    sparse: Glue.Partials, // [5, rows, H, kv_lora + 2] fp32
    sparse_bufs: [3]Buf,
    scores: Buf, // [select rows, np] fp32
    pools: Buf, // [rows, 512] i64
    tokens: Buf, // [rows, 2051] i32
    counts: Buf, // [rows] i32
    split: Buf, // select_split's scratch
    at: Buf, // a prompt chunk's blocks' first positions (i32): the dense blocks' packed, the selection's 16 bytes apart

    pub fn init(d: *const cuda.Driver, lay: config.Layout, rows: usize, capacity: usize, prefill: bool) !Scratch {
        const c = lay.cfg;
        const H = lay.size(.heads);
        const lw = c.kv_lora;
        var s: Scratch = undefined;
        s.rows = rows;
        s.prefill = prefill;
        const a = struct {
            fn f(dd: *const cuda.Driver, n: usize) !Buf {
                return Buf.alloc(dd, @max(n, 16));
            }
        }.f;
        s.dp = try a(d, rows * (c.q_lora + lw) * 2);
        s.qr = try a(d, rows * c.q_lora * 2);
        s.xs_qr = try a(d, rows * c.q_lora / 64 * 4);
        s.lat = try a(d, rows * lw * 2);
        s.xs_lat = try a(d, rows * lw / 64 * 4);
        s.q = try a(d, rows * H * c.qk_dim * 2);
        s.ikr = try a(d, rows * (c.index_dim + c.index_heads) * 2);
        s.igr = try a(d, rows * c.index_dim * 4);
        s.gpart = try a(d, 8 * rows * c.index_dim * 4);
        s.qi = try a(d, rows * c.index_heads * c.index_dim * 2);
        s.qa = try a(d, rows * H * lw * 2);
        s.ol = try a(d, rows * H * lw * 2);
        s.vn = try a(d, rows * H * c.v_dim * 2);
        // forward.py Buffers: LatentScratch(rows, H, chunks_for(min(capacity, 2560) + rows), part_rows)
        s.part_rows = if (prefill) @min(rows, prompt_att_rows) else rows;
        s.nch = cdiv(@min(capacity, 2560) + rows, chunk);
        const n = s.nch * s.part_rows * H;
        s.dense_bufs = .{ try a(d, n * lw * 4), try a(d, n * 4), try a(d, n * 4) };
        s.dense = .{ .po = s.dense_bufs[0].ptr, .pm = s.dense_bufs[1].ptr, .pl = s.dense_bufs[2].ptr };
        // decode windows' sparse chunks and merge; a prompt chunk's sparse rows take the one-pass kernel instead
        const ns = if (prefill) 1 else cdiv(glue_mod.sparse_tokens, chunk) * rows * H;
        s.sparse_bufs = .{ try a(d, ns * lw * 4), try a(d, ns * 4), try a(d, ns * 4) };
        s.sparse = .{ .po = s.sparse_bufs[0].ptr, .pm = s.sparse_bufs[1].ptr, .pl = s.sparse_bufs[2].ptr };
        const np_max = capacity / 4;
        s.scores = try a(d, @min(rows, select_block) * @max(np_max, 1024) * 4);
        s.pools = try a(d, rows * 512 * 8);
        // a decode window's streams' selections each in a block of `seg_rows` rows (16-byte aligned starts)
        const sel_rows = if (prefill) rows else @max(rows, max_segs * seg_rows);
        s.tokens = try a(d, sel_rows * glue_mod.sparse_tokens * 4);
        s.counts = try a(d, sel_rows * 4);
        // decode windows split rows over 8,192 scores or more; a prompt chunk its last selection block of fewer
        // than 64 rows (sizes grow with the scores, but the chunk width steps at 16,384: both sides of it)
        s.split = try a(d, if (prefill)
            @max(glue_mod.splitScratchBytes(split_rows - 1, @max(np_max, 1024)), glue_mod.splitScratchBytes(split_rows - 1, std.math.clamp(np_max, 1024, 16384)))
        else
            glue_mod.splitScratchBytes(rows, @max(np_max, 1024)));
        s.at = try a(d, at_select + (rows / select_block + 2) * 16);
        return s;
    }

    /// Stream `i`'s selection block of a shared decode window (block 0: a lone window's).
    pub fn selection(s: *const Scratch, i: usize) Selection {
        return .{ .tokens = s.tokens.ptr + i * seg_rows * glue_mod.sparse_tokens * 4, .counts = s.counts.ptr + i * seg_rows * 4 };
    }

    pub fn deinit(s: *Scratch) void {
        inline for (.{ "dp", "qr", "xs_qr", "lat", "xs_lat", "q", "ikr", "igr", "gpart", "qi", "qa", "ol", "vn", "scores", "pools", "tokens", "counts", "split", "at" }) |f| @field(s, f).free();
        for (&s.dense_bufs) |*b| b.free();
        for (&s.sparse_bufs) |*b| b.free();
    }
};

/// One DSA block's call: the rows' inputs and where its output goes.
pub const Call = struct {
    normed: Ptr, // [R, D] bf16
    part: Ptr, // [R, D] fp32 out
    sk: Ptr, // split-K partials of the BF16 matmuls
    pos_dev: Ptr, // the first row's position, int32 on the device
    host_pos: usize, // the same on the host
    nch: usize, // forward.py chunks_for(st, R): ceil((pos + R) / 512)
    R: usize,
};

/// Where Scratch.at's selection positions start: past the dense blocks' packed ones.
const at_select = 256;

/// A device int32 holding host position `pos` (a prompt chunk's block start; the host knows the chunk's
/// position). The pointer's alignment is the reference's, which Triton specializes on: forward.py's dense blocks
/// read element i of one tensor `pos_dev + starts` (4 bytes apart); sparse.py's selection blocks a tensor of their own
/// (`pos_dev + a`, an allocation: 16-byte aligned).
fn devicePos(s: *const Scratch, comptime kind: enum { dense, select }, slot: usize, pos: usize, stream: cuda.Stream) !Ptr {
    const off = switch (kind) {
        .dense => slot * 4,
        .select => at_select + slot * 16,
    };
    const v: i32 = @intCast(pos);
    try s.at.uploadAsync(off, std.mem.asBytes(&v), stream.handle);
    try stream.synchronize();
    return s.at.ptr + off;
}

/// forward.py `dsa_front` and `dsa_rows` up to o_proj, then o_proj into call.part (forward.py `_dsa_latent`).
pub fn dsaBlock(g: Glue, lay: config.Layout, a: *const Dsa, cache: *const Cache, s: *const Scratch, call: Call, stream: cuda.Stream) !void {
    try dsaFront(g, lay, a, s, call.normed, call.sk, 0, call.R, stream);
    try dsaRows(g, lay, a, cache, s, call, stream);
    try dsaOut(g, lay, a, s, call.part, call.sk, 0, call.R, stream);
}

/// forward.py `dsa_front` of rows lo .. hi of the normed rows [R, D] (`normed`: row 0's): the projection, the query and
/// latent norms, the query expansion into the scratch's rows lo .. hi. Each row's bits are its own.
pub fn dsaFront(g: Glue, lay: config.Layout, a: *const Dsa, s: *const Scratch, normed: Ptr, sk: Ptr, lo: usize, hi: usize, stream: cuda.Stream) !void {
    const c = lay.cfg;
    const lw = c.kv_lora;
    const n = hi - lo;
    const dp = s.dp.ptr + lo * (c.q_lora + lw) * 2;
    const qr = s.qr.ptr + lo * c.q_lora * 2;
    try g.bmm(normed + lo * c.hidden * 2, c.hidden, a.proj.b16(), n, dp, false, sk, stream);
    try g.rmsnorm(dp, c.q_lora + lw, a.q_norm.ptr, qr, c.q_lora, s.xs_qr.ptr + lo * c.q_lora / 64 * 4, c.eps, n, c.q_lora, stream);
    try g.rmsnorm(dp + c.q_lora * 2, c.q_lora + lw, a.kv_norm.ptr, s.lat.ptr + lo * lw * 2, lw, s.xs_lat.ptr + lo * lw / 64 * 4, c.eps, n, lw, stream);
    try g.bmm(qr, c.q_lora, a.q_b.b16(), n, s.q.ptr + lo * a.heads * c.qk_dim * 2, false, sk, stream);
}

/// forward.py o_proj of a DSA block's rows lo .. hi (its expanded read-out) into `part`'s rows (fp32 [R, D]).
pub fn dsaOut(g: Glue, lay: config.Layout, a: *const Dsa, s: *const Scratch, part: Ptr, sk: Ptr, lo: usize, hi: usize, stream: cuda.Stream) !void {
    const c = lay.cfg;
    const w = a.heads * c.v_dim;
    try g.bmm(s.vn.ptr + lo * w * 2, w, a.o.b16(), hi - lo, part + lo * c.hidden * 4, true, sk, stream);
}

/// Whether a window of R rows at `pos` has rows past the dense limit (they attend sparsely).
pub fn sparseRows(c: *const config.Config, pos: usize, R: usize) bool {
    return pos + R - 1 >= c.denseLimit();
}

/// forward.py `dsa_rows` of a window whose fronts ran (`dsaFront` on every row): the latent rows written at their
/// positions, the indexer's keys and pools, the absorbed query, the attention and the expanded read-out (s.vn).
pub fn dsaRows(g: Glue, lay: config.Layout, a: *const Dsa, cache: *const Cache, s: *const Scratch, call: Call, stream: cuda.Stream) !void {
    try dsaIndex(g, lay, a, s, call.normed, call.sk, call.R, sparseRows(lay.cfg, call.host_pos, call.R), stream);
    try dsaAttend(g, lay, a, cache, s, call, 0, stream);
    try dsaExpand(g, lay, a, s, call.R, stream);
}

/// The row-wise part of `dsa_rows` over R rows (rows never mix): the indexer's keys and gates, the absorbed queries,
/// and when some row attends sparsely (`sparse`) the indexer's queries.
pub fn dsaIndex(g: Glue, lay: config.Layout, a: *const Dsa, s: *const Scratch, normed: Ptr, sk: Ptr, R: usize, sparse: bool, stream: cuda.Stream) !void {
    const c = lay.cfg;
    try g.bmm(normed, c.hidden, a.kw.b16(), R, s.ikr.ptr, false, sk, stream);
    try g.router(normed, c.hidden, a.gate.ptr, c.index_dim, R, s.igr.ptr, s.gpart.ptr, stream);
    try g.absorbQ(s.q.ptr, a.wk.ptr, s.qa.ptr, R, a.heads, c.qk_dim, c.kv_lora, s.prefill, stream);
    if (sparse) try g.bmm(s.qr.ptr, c.q_lora, a.qb.b16(), R, s.qi.ptr, false, sk, stream);
}

/// One stream's part of `dsa_rows`: `dsaSelect`, then `dsaAttendRows` on its selection in s.tokens / s.counts.
pub fn dsaAttend(g: Glue, lay: config.Layout, a: *const Dsa, cache: *const Cache, s: *const Scratch, call: Call, r0: usize, stream: cuda.Stream) !void {
    const sel = s.selection(0);
    try dsaSelect(g, lay, a, cache, s, call, r0, sel, stream);
    try dsaAttendRows(g, lay, a, cache, s, call, r0, sel, null, stream);
}

/// Where one stream's selected tokens [R, 2051] i32 and their counts [R] go.
pub const Selection = struct { tokens: Ptr, counts: Ptr };

/// One stream's rows [r0, r0 + call.R) of the scratch (the latent rows, the indexer's keys, gates and queries) into
/// its caches at its position, and, when a row attends sparsely, each such row's selected tokens into `sel`.
pub fn dsaSelect(g: Glue, lay: config.Layout, a: *const Dsa, cache: *const Cache, s: *const Scratch, call: Call, r0: usize, sel: Selection, stream: cuda.Stream) !void {
    const c = lay.cfg;
    const R = call.R;
    const lw = c.kv_lora;
    const ikw = c.index_dim + c.index_heads;
    const ikr = s.ikr.ptr + r0 * ikw * 2;
    const lat = s.lat.ptr + r0 * lw * 2;
    if (cache.tier) |tr| {
        // one piece, or two across a tier segment's end (a copy of rows: the same bytes)
        for (try tr.t.writes(tr.h, tr.layer, call.host_pos, R, call.pos_dev)) |pc| if (pc) |p| {
            try g.latWrite(lat + (p.pos - call.host_pos) * lw * 2, lw, p.base, cache.kv, p.pos_dev, p.rows, lw, stream);
        };
    } else try g.latWrite(lat, lw, cache.latent.ptr, cache.kv, call.pos_dev, R, lw, stream);
    try g.indexUpdate(ikr, ikw, s.igr.ptr + r0 * c.index_dim * 4, a.ln_w.ptr, a.ln_b.ptr, a.ape.ptr, cache.ik.ptr, cache.ig.ptr, cache.ring, cache.pk.ptr, cache.kv, call.pos_dev, R, stream);
    if (sparseRows(c, call.host_pos, R)) try selectTokens(g, s, cache, call, s.qi.ptr + r0 * c.index_heads * c.index_dim * 2, ikr + c.index_dim * 2, ikw, sel.tokens, sel.counts, stream);
}

/// One stream's attention over its rows [r0, r0 + call.R) after `dsaSelect`: dense up to the dense limit, past it
/// over the selected tokens in `sel`, into the read-out rows from r0.
pub fn dsaAttendRows(g: Glue, lay: config.Layout, a: *const Dsa, cache: *const Cache, s: *const Scratch, call: Call, r0: usize, sel: Selection, resolved: ?tier_mod.Tier.Resolved, stream: cuda.Stream) !void {
    const c = lay.cfg;
    const R = call.R;
    const H = a.heads;
    const lw = c.kv_lora;
    const qa = s.qa.ptr + r0 * H * lw * 2;
    const ol = s.ol.ptr + r0 * H * lw * 2;
    const limit = c.denseLimit();
    const all_sparse = call.host_pos >= limit;
    const sparse_rows = sparseRows(c, call.host_pos, R);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.qk_dim), -0.5));
    // dense attention reads positions [0, pos + R) at one base: the tier's home block holds [0, 6,144)
    const dense_base = if (cache.tier) |tr| tr.t.denseBase(tr.h, tr.layer) else cache.latent.ptr;
    if (!all_sparse) {
        // rows past the dense limit are recomputed sparsely below: the dense pass takes the rows before it
        const dense = if (sparse_rows) @min(R, limit -| call.host_pos) else R;
        const nch = @min(call.nch, s.nch);
        const hb = headBlock(R);
        if (dense <= s.part_rows) {
            try g.denseAttention(qa, dense_base, cache.kv, call.pos_dev, s.dense, ol, dense, H, lw, nch, hb, scale, stream);
        } else {
            var q0: usize = 0;
            var i: usize = 0;
            while (q0 < dense) : ({
                q0 += s.part_rows;
                i += 1;
            }) {
                const q1 = @min(dense, q0 + s.part_rows);
                const at = try devicePos(s, .dense, i, call.host_pos + q0, stream);
                try g.denseAttention(qa + q0 * H * lw * 2, dense_base, cache.kv, at, s.dense, ol + q0 * H * lw * 2, q1 - q0, H, lw, nch, hb, scale, stream);
            }
        }
    }
    if (sparse_rows) {
        // on the tier: the selected pools made resident, the positions remapped to slab rows (sel kept); a round's
        // streams may resolve together first (cuda_tier.resolveMany) and pass `resolved`
        const sp: tier_mod.Tier.Resolved = if (resolved) |x| x else if (cache.tier) |tr| try tr.t.resolve(tr.h, tr.layer, sel.tokens, sel.counts, R, glue_mod.sparse_tokens, stream) else .{ .base = cache.latent.ptr, .tokens = sel.tokens };
        if (s.prefill) {
            try g.sparseOnepass(qa, sp.base, cache.kv, sp.tokens, sel.counts, ol, R, H, lw, glue_mod.sparse_tokens, scale, stream);
        } else {
            try g.sparseAttention(qa, sp.base, cache.kv, sp.tokens, sel.counts, s.sparse, ol, R, H, lw, glue_mod.sparse_tokens, headBlock(R), scale, stream);
        }
    }
}

/// The read-out rows expanded through kv_b's value rows (s.vn), over R rows.
pub fn dsaExpand(g: Glue, lay: config.Layout, a: *const Dsa, s: *const Scratch, R: usize, stream: cuda.Stream) !void {
    const c = lay.cfg;
    try g.expandV(s.ol.ptr, a.wv.ptr, s.vn.ptr, R, a.heads, c.v_dim, c.kv_lora, s.prefill, stream);
}

/// sparse.py `select_tokens` (no graph bucket): each row's attended tokens and their count into s.tokens / s.counts.
fn selectTokens(g: Glue, s: *const Scratch, cache: *const Cache, call: Call, qi: Ptr, wts: Ptr, w_stride: usize, tokens: Ptr, counts: Ptr, stream: cuda.Stream) !void {
    const R = call.R;
    const np_max = cache.poolRows();
    const qrow = 32 * 128 * 2;
    if (R >= prompt_select_from) {
        // _select_prompt's selection, in blocks of select_block rows scoring the chunk's pool_count pools
        const NP = poolCount(call.host_pos, R, np_max);
        const B = @min(R, select_block);
        var a: usize = 0;
        var i: usize = 0;
        while (a < R) : ({
            a += B;
            i += 1;
        }) {
            const n = @min(B, R - a);
            const at = if (a == 0) call.pos_dev else try devicePos(s, .select, i, call.host_pos + a, stream);
            try g.scores(qi + a * qrow, wts + a * w_stride * 2, w_stride, cache.pk.ptr, cache.kv, s.scores.ptr, at, n, NP, stream);
            try g.promptPools(s.scores.ptr, NP, s.pools.ptr, n, s.split, stream);
            try g.tokens(s.pools.ptr, at, tokens + a * glue_mod.sparse_tokens * 4, counts + a * 4, n, stream);
        }
        return;
    }
    const np = poolBucket(call.host_pos, R, np_max);
    const grid = @min(row_score_g, cdiv(np, 64));
    try g.scoresRows(qi, wts, w_stride, cache.pk.ptr, cache.kv, s.scores.ptr, call.pos_dev, R, np, grid, stream);
    if (R < split_rows and np >= split_from) {
        try g.selectSplit(s.scores.ptr, np, call.pos_dev, s.split, tokens, counts, R, stream);
    } else {
        try g.selectRows(s.scores.ptr, s.pools.ptr, np, call.pos_dev, R, stream);
        try g.tokens(s.pools.ptr, call.pos_dev, tokens, counts, R, stream);
    }
}

test "sparse.py's pool counts" {
    // decode rows at 4,159: 1,040 visible pools score a bucket of 2,048
    try std.testing.expectEqual(@as(usize, 2048), poolBucket(4159, 3, 2048));
    try std.testing.expectEqual(@as(usize, 16384), poolBucket(40_000, 16, 150_016));
    try std.testing.expectEqual(@as(usize, 150_016), poolBucket(600_000, 1, 150_016));
    // a prompt chunk from 2,070: 1,030 visible pools score 2,048
    try std.testing.expectEqual(@as(usize, 2048), poolCount(2070, 2048, 2048));
    try std.testing.expectEqual(@as(usize, 2112), indexRing(8192, 2048));
}
