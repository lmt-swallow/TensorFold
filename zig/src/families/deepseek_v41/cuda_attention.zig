//! The attention sublayer of one DeepSeek-V4.1 backbone layer at decode sizes (1 to 16 rows of one stream, this
//! rank's heads): the Python fork's graph.py `StaticDecoder._attention`, launch for launch. The EXL3 projections are
//! cuda_exl3.zig's (each `mm`), the Triton kernels come from the captured AOT set (cuda.aot: `_rmsnorm`,
//! `_rope_heads`, `_kv_norm_rope`, `_fp4qd_p2`, `_fp4_store`, `_rowmm`, `_index_score`, `_sparse_attn_part`,
//! `_sparse_attn_merge`) and the torch glue (the compressor's softmax pool, the indexer weights' scaling, column
//! block copies) from attn_glue.cu. The indexer's selection (kernels.topk_indices: torch's total-order keys, so one
//! set whatever the algorithm) runs on the device: the captured `_score_keys` + `_topk_sel` where the fork's fused
//! selection has a variant for the shape, else topk_glue.cu's row select; the candidate pool of the candidate source
//! (model.py `_candidates`) and its mask on the index layers after it (`apply_candidates`) as topk_glue.cu kernels.
//! TF_DS_TOPK=host keeps the host selection (the oracle). Every row's bits are its own, as in the fork: the output
//! partial [n, d] fp32 is what rank 0 would gather.

const std = @import("std");
const cuda = @import("cuda");
const exl3 = @import("cuda_exl3.zig");

const aot = cuda.aot;
const Ptr = cuda.abi.DevicePtr;

pub const Error = error{ BadShape, NoCandidateSource, OutOfMemory, TooManyRows } || cuda.Error || exl3.Error;

/// Small host tables copied to the device in a stream's order: each `put` copies its bytes into the next free part
/// of a page-locked buffer and from there asynchronously on the caller's stream (a plain upload runs on the legacy
/// stream, outside the order of the non-blocking streams the kernels run on). A put's host bytes stay untouched until
/// its copy has run: `begin` (once a forward, before its first put) waits for the previous forward's copies.
pub const Uploads = struct {
    host: cuda.HostBuffer,
    used: usize,
    done: cuda.Event,
    pending: bool,

    pub const alignment: usize = 16;

    /// Room for puts of `bytes` in total a forward, each rounded up to `alignment`.
    pub fn init(d: *const cuda.Driver, bytes: usize) !Uploads {
        var host = try cuda.HostBuffer.alloc(d, @max(alignment, bytes));
        errdefer host.free();
        return .{ .host = host, .used = 0, .done = try cuda.Event.init(d, false), .pending = false };
    }

    /// The room `puts` copies of up to `bytes` each need.
    pub fn room(puts: usize, bytes: usize) usize {
        return puts * std.mem.alignForward(usize, bytes, alignment);
    }

    pub fn deinit(self: *Uploads) void {
        if (self.pending) self.done.synchronize() catch {};
        self.done.deinit();
        self.host.free();
    }

    /// Starts a forward's puts: the previous forward's copies have run once this returns, and their room is free.
    pub fn begin(self: *Uploads) !void {
        if (self.pending) try self.done.synchronize();
        self.pending = false;
        self.used = 0;
    }

    /// dst[offset ..] <- bytes, on `stream` after the work already queued there.
    pub fn put(self: *Uploads, dst: cuda.DeviceBuffer, offset: usize, bytes: []const u8, stream: cuda.Stream) !void {
        const need = std.mem.alignForward(usize, bytes.len, alignment);
        if (self.used + need > self.host.bytes.len) return error.OutOfMemory;
        const staged = self.host.bytes[self.used .. self.used + bytes.len];
        @memcpy(staged, bytes);
        self.used += need;
        try dst.uploadAsync(offset, staged, stream.handle);
        try self.done.record(stream);
        self.pending = true;
    }
};

/// The sizes the launches read: the pack's (defaults), and the layer's ratio, sources and context bucket.
pub const Dims = struct {
    d: usize = 5120,
    heads: usize = 32, // this rank's heads (64 over 2 ranks)
    hd: usize = 512,
    rd: usize = 64,
    q_rank: usize = 1280,
    o_rank: usize = 1024,
    groups: usize = 4, // this rank's wo_a slices (8 over 2 ranks)
    window: usize = 128,
    ring: usize = 144, // window + RING_EXTRA slots
    raw: usize = 64, // compressor inputs kept by position (RAW)
    idx_heads: usize = 32,
    idx_dim: usize = 128,
    idx_topk: usize = 512,
    ratio: usize, // compress ratio of this layer (0: window attention only)
    kv_source: bool,
    index_source: bool,
    has_idx_wk: bool,
    eps: f32,
    bucket: usize, // the context bucket: the indexer scans bucket / ratio compressed entries
    max_rows: usize = 16,
    cand: CandRole = .none, // the candidate pool: made here (the candidate source), applied here (an index layer after it)
    cand_blocks: usize = 0, // the pool's blocks (candidate_topk_blocks)
    cand_block: usize = 8, // entries a block (candidate_block_size)

    pub fn nb(self: Dims) usize {
        return if (self.ratio == 0) 0 else self.bucket / self.ratio;
    }

    pub fn kk(self: Dims) usize {
        return @min(self.idx_topk, self.nb());
    }

    /// The pool's mask columns over the scanned entries (ceil(nb / cand_block)).
    pub fn candCols(self: Dims) usize {
        return (self.nb() + self.cand_block - 1) / @max(1, self.cand_block);
    }
};

/// An index layer's part in the candidate pool (config candidate_source_layer_id).
pub const CandRole = enum {
    none,
    source, // the candidate source: its scores make the pool
    after, // an index layer after the source: its scores masked to the pool

    /// Layer `li`'s role given the config's candidate source (-1: none) and whether `li` is an index layer.
    pub fn of(cand_source: i64, li: usize, index_layer: bool) CandRole {
        if (!index_layer or cand_source < 0) return .none;
        const src: usize = @intCast(cand_source);
        if (li == src) return .source;
        return if (li > src) .after else .none;
    }
};

/// How the indexer selects its picks: `auto` the device (the captured fused selection where it has the shape's
/// variant, else topk_glue.cu), `glue` topk_glue.cu always, `host` the host oracle (a stream sync a layer). From
/// TF_DS_TOPK (auto, glue, host); auto when unset.
pub const Select = enum {
    auto,
    glue,
    host,

    pub fn fromEnv() Select {
        const v = std.mem.span(std.c.getenv("TF_DS_TOPK") orelse return .auto);
        if (std.mem.eql(u8, v, "host")) return .host;
        if (std.mem.eql(u8, v, "glue")) return .glue;
        return .auto;
    }
};

/// topk_glue.cu, the indexer's selection kernels.
pub const TopkGlue = struct {
    module: cuda.Module,
    topk_rows: cuda.Function,
    topk_keys: cuda.Function,
    cand_blockmax: cuda.Function,
    cand_mask: cuda.Function,
    apply_cand: cuda.Function,
    topk_keyset: cuda.Function,
    cand_blocks: cuda.Function,
    cand_bits: cuda.Function,
    apply_cand_bits: cuda.Function,

    pub const threads: u32 = 512; // topk_glue.cu kThreads
    pub const max_picks: usize = 1024; // topk_glue.cu kMaxPicks

    pub fn load(d: *const cuda.Driver) !TopkGlue {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_topk_glue);
        errdefer module.unload();
        return .{
            .module = module,
            .topk_rows = try module.function("tf_dsv41_topk_rows"),
            .topk_keys = try module.function("tf_dsv41_topk_keys"),
            .cand_blockmax = try module.function("tf_dsv41_cand_blockmax"),
            .cand_mask = try module.function("tf_dsv41_cand_mask"),
            .apply_cand = try module.function("tf_dsv41_apply_cand"),
            .topk_keyset = try module.function("tf_dsv41_topk_keyset"),
            .cand_blocks = try module.function("tf_dsv41_cand_blocks"),
            .cand_bits = try module.function("tf_dsv41_cand_bits"),
            .apply_cand_bits = try module.function("tf_dsv41_apply_cand_bits"),
        };
    }

    pub fn deinit(self: *TopkGlue) void {
        self.module.unload();
    }

    /// kernels.topk_indices then `where(top < vis, top, -1)` on rows of fp32 scores [rows, n] (row stride `ld`
    /// floats): out [rows, kk] int64 (row stride `out_ld`), each row's kk highest scores' indices ascending (ties to
    /// the lower index), -1 at or past vis[r] (`vis` 0: no mask). kk <= n.
    pub fn topk(self: *const TopkGlue, score: Ptr, ld: usize, rows: usize, n: usize, kk: usize, vis: Ptr, out: Ptr, out_ld: usize, stream: cuda.Stream) !void {
        return self.select(self.topk_rows, score, ld, rows, n, kk, vis, out, out_ld, stream);
    }

    /// `topk` from the int64 keys a score kernel wrote with KEYS (kernels.topk_select's input; row stride `ld`
    /// keys): the picks of the scores that kernel computed.
    pub fn topkKeys(self: *const TopkGlue, keys: Ptr, ld: usize, rows: usize, n: usize, kk: usize, vis: Ptr, out: Ptr, out_ld: usize, stream: cuda.Stream) !void {
        return self.select(self.topk_keys, keys, ld, rows, n, kk, vis, out, out_ld, stream);
    }

    /// kernels.topk_select's torch.topk + `_topk_finish` on int64 keys [rows, n] (row stride `ld`) in any order (the
    /// pruned selection's gathered tiles, the candidate pool's keys): each row's kk largest keys' indices (decoded
    /// from the keys) ascending, -1 at or past vis[r] (`vis` 0: no mask). kk <= n, kk <= max_picks.
    pub fn topkKeyset(self: *const TopkGlue, keys: Ptr, ld: usize, rows: usize, n: usize, kk: usize, vis: Ptr, out: Ptr, out_ld: usize, stream: cuda.Stream) !void {
        if (kk > max_picks) return error.BadShape;
        return self.select(self.topk_keyset, keys, ld, rows, n, kk, vis, out, out_ld, stream);
    }

    /// rounds.py `_candidate_blocks` on rows of fp32 scores [rows, n] (row stride `ld`): cblk [rows, min(nblocks,
    /// ceil(n / bsize))] int32 (row stride `cblk_ld`), the blocks `candidates` marks as ascending block ids, -1 for
    /// a block whose maximum is not above -inf. `smax`: fp32 scratch [rows, ceil(n / bsize)].
    pub fn candidateBlocks(self: *const TopkGlue, score: Ptr, ld: usize, rows: usize, n: usize, vis: Ptr, nblocks: usize, bsize: usize, smax: Ptr, cblk: Ptr, cblk_ld: usize, stream: cuda.Stream) !void {
        if (rows == 0 or n == 0) return;
        const nblk = (n + bsize - 1) / bsize;
        try self.blockMax(score, ld, rows, n, vis, bsize, smax, nblk, stream);
        var a: cuda.Args = .{};
        a.add(smax);
        a.add(@as(c_longlong, @intCast(nblk)));
        a.add(@as(c_int, @intCast(nblk)));
        a.add(@as(c_int, @intCast(@min(nblocks, nblk))));
        a.add(cblk);
        a.add(@as(c_longlong, @intCast(cblk_ld)));
        try cuda.launch.launch(self.cand_blocks, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = threads } }, stream, &a);
    }

    fn blockMax(self: *const TopkGlue, score: Ptr, ld: usize, rows: usize, n: usize, vis: Ptr, bsize: usize, smax: Ptr, nblk: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(@as(c_int, @intCast(n)));
        a.add(vis);
        a.add(@as(c_int, @intCast(bsize)));
        a.add(smax);
        a.add(@as(c_longlong, @intCast(nblk)));
        a.add(@as(c_int, @intCast(nblk)));
        try cuda.launch.launch(self.cand_blockmax, .{ .grid = .{ .x = @intCast((nblk + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    fn select(self: *const TopkGlue, f: cuda.Function, src: Ptr, ld: usize, rows: usize, n: usize, kk: usize, vis: Ptr, out: Ptr, out_ld: usize, stream: cuda.Stream) !void {
        _ = self;
        if (kk == 0 or rows == 0) return;
        if (kk > n) return error.BadShape;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(kk)));
        a.add(vis);
        a.add(out);
        a.add(@as(c_longlong, @intCast(out_ld)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = threads } }, stream, &a);
    }

    /// model.py _candidates on rows of fp32 scores [rows, n] (row stride `ld`): mask [rows, ceil(n / bsize)] u8 (row
    /// stride `mask_ld`), 1 on the `nblocks` blocks with the largest maxima (the block of entry vis[r] - 1 pinned in)
    /// whose maximum is above -inf. `smax`: fp32 scratch [rows, ceil(n / bsize)].
    pub fn candidates(self: *const TopkGlue, score: Ptr, ld: usize, rows: usize, n: usize, vis: Ptr, nblocks: usize, bsize: usize, smax: Ptr, mask: Ptr, mask_ld: usize, stream: cuda.Stream) !void {
        if (rows == 0 or n == 0) return;
        const nblk = (n + bsize - 1) / bsize;
        try self.blockMax(score, ld, rows, n, vis, bsize, smax, nblk, stream);
        var a: cuda.Args = .{};
        a.add(smax);
        a.add(@as(c_longlong, @intCast(nblk)));
        a.add(@as(c_int, @intCast(nblk)));
        a.add(@as(c_int, @intCast(@min(nblocks, nblk))));
        a.add(mask);
        a.add(@as(c_longlong, @intCast(mask_ld)));
        try cuda.launch.launch(self.cand_mask, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = threads } }, stream, &a);
    }

    /// `candidates` with the pool as bits: block b of row r is bit b % 32 of word b / 32 of bits [rows,
    /// bitWords(ceil(n / bsize))] uint32 (row stride `bits_ld` words), 1 exactly where `candidates`' mask is 1.
    pub fn candidatesBits(self: *const TopkGlue, score: Ptr, ld: usize, rows: usize, n: usize, vis: Ptr, nblocks: usize, bsize: usize, smax: Ptr, bits: Ptr, bits_ld: usize, stream: cuda.Stream) !void {
        if (rows == 0 or n == 0) return;
        const nblk = (n + bsize - 1) / bsize;
        try self.blockMax(score, ld, rows, n, vis, bsize, smax, nblk, stream);
        var a: cuda.Args = .{};
        a.add(smax);
        a.add(@as(c_longlong, @intCast(nblk)));
        a.add(@as(c_int, @intCast(nblk)));
        a.add(@as(c_int, @intCast(@min(nblocks, nblk))));
        a.add(bits);
        a.add(@as(c_longlong, @intCast(bits_ld)));
        try cuda.launch.launch(self.cand_bits, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = threads } }, stream, &a);
    }

    /// The uint32 words a row of the pool as bits takes for `blocks` blocks.
    pub fn bitWords(blocks: usize) usize {
        return (blocks + 31) / 32;
    }

    /// `apply` with the pool as bits (`candidatesBits`): score[r, t] = -inf where block t / bsize's bit is 0 (t < n).
    pub fn applyBits(self: *const TopkGlue, score: Ptr, ld: usize, rows: usize, n: usize, bits: Ptr, bits_ld: usize, bsize: usize, stream: cuda.Stream) !void {
        if (rows == 0 or n == 0) return;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(@as(c_int, @intCast(n)));
        a.add(bits);
        a.add(@as(c_longlong, @intCast(bits_ld)));
        a.add(@as(c_int, @intCast(bsize)));
        try cuda.launch.launch(self.apply_cand_bits, .{ .grid = .{ .x = @intCast((n + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    /// model.py apply_candidates in place: score[r, t] = -inf where mask[r, t / bsize] is 0 (t < n).
    pub fn apply(self: *const TopkGlue, score: Ptr, ld: usize, rows: usize, n: usize, mask: Ptr, mask_ld: usize, bsize: usize, stream: cuda.Stream) !void {
        if (rows == 0 or n == 0) return;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(@as(c_int, @intCast(n)));
        a.add(mask);
        a.add(@as(c_longlong, @intCast(mask_ld)));
        a.add(@as(c_int, @intCast(bsize)));
        try cuda.launch.launch(self.apply_cand, .{ .grid = .{ .x = @intCast((n + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }
};

/// The fork's fused selection (kernels.topk_select: `_score_keys` then `_topk_sel`, one launch a row) where the
/// captured set has its shape: n and kk powers of two, kk <= n <= TOPK_FUSED_MAX (4096), contiguous rows. Returns
/// false (nothing launched) where it does not apply; the picks are the same set either way (the keys' total order).
/// `keys`: int64 scratch [rows, n].
pub fn fusedTopk(set: *const aot.Set, score: Ptr, rows: usize, n: usize, kk: usize, vis: Ptr, keys: Ptr, out: Ptr, stream: cuda.Stream) !bool {
    const pow2 = struct {
        fn f(v: usize) bool {
            return v > 0 and (v & (v - 1)) == 0;
        }
    }.f;
    if (!pow2(n) or !pow2(kk) or kk > n or n > 4096 or keys == 0) return false;
    const key_args = [_]aot.Arg{
        aot.ptr("S", "*fp32", score), aot.ptr("OUT", "*i64", keys), aot.int("n", @intCast(n)),
        aot.ptr("TMAX", "*i64", keys), aot.int("nt", @intCast((n + 63) / 64)),
    };
    const key_consts = [_]aot.Const{ aot.ci("BN", 1024), aot.ci("PDL", 1), aot.ci("HAS_TMAX", 0), aot.ci("TB", 64) };
    const sel_args = [_]aot.Arg{ aot.ptr("KEYS", "*i64", keys), aot.ptr("VIS", "*i64", vis), aot.ptr("OUT", "*i64", out) };
    const sel_consts = [_]aot.Const{ aot.ci("N", @intCast(n)), aot.ci("KK", @intCast(kk)), aot.ci("PDL", 1) };
    if (!captured(set, "_score_keys", &key_args, &key_consts) or !captured(set, "_topk_sel", &sel_args, &sel_consts)) return false;
    try set.run(stream, "_score_keys", .{ @intCast(rows), @intCast((n + 1023) / 1024), 1 }, &key_args, &key_consts);
    try set.run(stream, "_topk_sel", .{ @intCast(rows), 1, 1 }, &sel_args, &sel_consts);
    return true;
}

/// Whether `set` holds a variant `Set.run` would launch for these arguments (aot.zig's matching, strict or with a
/// less specialized capture), without its missing-variant report: an optional fast path probes with it.
pub fn captured(set: *const aot.Set, function: []const u8, args: []const aot.Arg, consts: []const aot.Const) bool {
    for ([_]bool{ false, true }) |loose| for (set.variants) |v| {
        if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
        if (variantMatches(v.spec, args, consts, loose)) return true;
    };
    return false;
}

fn variantMatches(k: anytype, args: []const aot.Arg, consts: []const aot.Const, loose: bool) bool {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return false;
        if (c.int) |x| if (got.int == null or got.int.? != x) return false;
    }
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    const got = k.consts.map.get(a.name) orelse return false;
                    if (x != 1 or got.int == null or got.int.? != 1) return false;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return false;
                if (!p.nospec and x == 1 and !loose) return false;
                const div = !p.nospec and @mod(x, 16) == 0;
                if (p.div16 != div and !(loose and !p.div16)) return false;
            },
            .ptr => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, x.ty)) return false;
                const div = x.addr % 16 == 0;
                if (p.div16 != div and !(loose and !p.div16)) return false;
            },
            .f32 => {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "fp32")) return false;
            },
            .u64 => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "u64")) return false;
                const div = x % 16 == 0;
                if (p.div16 != div and !(loose and !p.div16)) return false;
            },
        }
        runtime += 1;
    }
    return runtime == k.params.len;
}

/// kernels.topk_indices on one row on the host: the kk highest scores' indices ascending (ties to the lower index),
/// -1 at or past `vis`. `keys`: scratch of row.len.
pub fn hostTopk(row: []const f32, kk: usize, vis: i64, keys: []i64, out: []i64) void {
    for (row, 0..) |v, j| keys[j] = key(v, j);
    hostTopkKeys(keys[0..row.len], kk, vis, out);
}

/// kernels.topk_select on one row of int64 keys on the host (sorts `keys` in place).
pub fn hostTopkKeys(keys: []i64, kk: usize, vis: i64, out: []i64) void {
    std.sort.pdq(i64, keys, {}, std.sort.desc(i64));
    for (keys[0..kk], 0..) |kv, i| out[i] = 0xFFFFFFFF - (kv & 0xFFFFFFFF);
    std.sort.pdq(i64, out[0..kk], {}, std.sort.asc(i64));
    for (out[0..kk]) |*p| if (p.* >= vis) {
        p.* = -1;
    };
}

/// model.py _candidates on one row on the host: mask[b] = 1 on the `nblocks` blocks of `bsize` entries with the
/// largest maxima (the block of entry vis - 1 pinned in) whose maximum is above -inf. `keys`: scratch of the blocks.
pub fn hostCandidates(row: []const f32, vis: i64, nblocks: usize, bsize: usize, keys: []i64, mask: []u8) void {
    const n = row.len;
    const nb = (n + bsize - 1) / bsize;
    const last: i64 = if (vis - 1 >= 0) @divFloor(vis - 1, @as(i64, @intCast(bsize))) else -1;
    @memset(mask[0..nb], 0);
    for (0..nb) |b| {
        var m: f32 = -std.math.inf(f32);
        var nan = false;
        for (row[b * bsize .. @min(n, (b + 1) * bsize)]) |v| {
            if (std.math.isNan(v)) nan = true else if (v > m) m = v;
        }
        if (nan) m = std.math.nan(f32);
        if (@as(i64, @intCast(b)) == last) m = std.math.inf(f32);
        keys[b] = key(m, b);
    }
    const ks = keys[0..nb];
    std.sort.pdq(i64, ks, {}, std.sort.desc(i64));
    // `s.gather(idx) > -inf`: true exactly for the keys strictly above -inf's and at most +inf's (NaN maxima order
    // outside that range)
    const neg_inf_hi = key(-std.math.inf(f32), 0) >> 32;
    const inf_hi = key(std.math.inf(f32), 0) >> 32;
    for (ks[0..@min(nblocks, nb)]) |kv| {
        const b: usize = @intCast(0xFFFFFFFF - (kv & 0xFFFFFFFF));
        const hi = kv >> 32;
        if (hi > neg_inf_hi and hi <= inf_hi) mask[b] = 1;
    }
}

/// model.py apply_candidates on one row on the host: -inf outside the pool's blocks.
pub fn hostApplyCandidates(row: []f32, mask: []const u8, bsize: usize) void {
    for (row, 0..) |*v, t| if (mask[t / bsize] == 0) {
        v.* = -std.math.inf(f32);
    };
}

/// attn_glue.cu, loaded from a cubin (built by tools/zig/dsv41/attn_capture.sh with the fork image's nvcc).
pub const Glue = struct {
    module: cuda.Module,
    comp_pool2: cuda.Function,
    bf16_scale: cuda.Function,
    copy_cols: cuda.Function,
    f32_to_f16: cuda.Function,
    // the round's tail (null in a cubin built before them: the callers keep copy_cols and the host argmax)
    copy_rows16: ?cuda.Function = null,
    argmax_rows: ?cuda.Function = null,
    topk: ?TopkGlue = null, // the selection kernels (built-in fatbin; null in host-only builds: the host selection)
    select: Select = .auto,
    rope: ?RopeGen = null, // rope.cu (built-in fatbin; null in host-only builds: the table path only)

    /// The fatbin built into the binary (zig/build/cuda.zig: dsv41_attn_glue).
    pub fn load(d: *const cuda.Driver) !Glue {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        return loadBytes(d, cuda.kernels.dsv41_attn_glue);
    }

    /// A cubin of attn_glue.cu read from disk (tools/zig/dsv41/attn_capture.sh writes one).
    pub fn loadBytes(d: *const cuda.Driver, cubin: []const u8) !Glue {
        var module = try cuda.Module.load(d, cubin);
        errdefer module.unload();
        const sel = Select.fromEnv();
        return .{
            .module = module,
            .comp_pool2 = try module.function("comp_pool2"),
            .bf16_scale = try module.function("bf16_scale"),
            .copy_cols = try module.function("copy_cols"),
            .f32_to_f16 = try module.function("f32_to_f16"),
            .copy_rows16 = module.function("copy_rows16") catch null,
            .argmax_rows = module.function("argmax_rows") catch null,
            .topk = if (cuda.kernels.available) try TopkGlue.load(d) else null,
            .select = if (cuda.kernels.available) sel else .host,
            .rope = if (cuda.kernels.available) try RopeGen.load(d) else null,
        };
    }

    pub fn deinit(self: *Glue) void {
        if (self.rope) |*r| r.deinit();
        if (self.topk) |*t| t.deinit();
        self.module.unload();
    }

    /// Whether the indexer selects on the host (TF_DS_TOPK=host, or no selection kernels).
    pub fn hostSelect(self: *const Glue) bool {
        return self.select == .host or self.topk == null;
    }

    /// dst[r * dld + c] = src[r * sld + c] for c < bytes, r < rows, in 16-byte words; false (nothing launched) when
    /// the kernel is missing or a pointer, stride or length is not a multiple of 16 bytes: copy_cols's bytes then.
    pub fn copyRows16(self: *const Glue, src: Ptr, sld: usize, dst: Ptr, dld: usize, rows: usize, bytes: usize, stream: cuda.Stream) !bool {
        const f = self.copy_rows16 orelse return false;
        if ((src | dst | sld | dld | bytes) % 16 != 0 or rows == 0 or rows > 65535) return false;
        const n16 = bytes / 16;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(u64, sld / 16));
        a.add(dst);
        a.add(@as(u64, dld / 16));
        a.add(@as(u64, n16));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((n16 + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
        return true;
    }

    /// out[r] = row r's greedy token for r < rows (logits fp32, row stride `ld` floats, `n` of them a row): the first
    /// index of the largest value, a NaN never larger, 0 when the row's first value is NaN (the host loop's
    /// `x > row[best]` rule). False (nothing launched) without the kernel.
    pub fn argmaxRows(self: *const Glue, logits: Ptr, ld: usize, n: usize, rows: usize, out: Ptr, stream: cuda.Stream) !bool {
        const f = self.argmax_rows orelse return false;
        if (rows == 0) return true;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, ld));
        a.add(@as(c_int, @intCast(n)));
        a.add(out);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &a);
        return true;
    }
};

/// This rank's weights of one layer's attention (weights.py `load_block`).
pub const Layer = struct {
    wq_a: exl3.Linear,
    wq_b: exl3.Linear, // this rank's head columns
    wkv: exl3.Linear,
    wo_a: [4]exl3.Linear, // this rank's slices
    wo_b: exl3.Linear, // this rank's input rows
    comp_wkv: ?exl3.Linear = null,
    comp_wgate: ?exl3.Linear = null,
    idx_wq_b: ?exl3.Linear = null,
    idx_wk: ?exl3.Linear = null,
    q_norm: cuda.DeviceBuffer, // bf16 [q_rank]
    kv_norm: cuda.DeviceBuffer, // bf16 [hd]
    sink: cuda.DeviceBuffer, // fp32 [heads]
    comp_norm: ?cuda.DeviceBuffer = null, // bf16 [hd]
    idx_k_norm: ?cuda.DeviceBuffer = null, // bf16 [idx_dim]
    idx_proj_h: ?cuda.DeviceBuffer = null, // fp16 [idx_heads, d]

    pub fn deinit(self: *Layer) void {
        self.wq_a.deinit();
        self.wq_b.deinit();
        self.wkv.deinit();
        for (&self.wo_a) |*w| w.deinit();
        self.wo_b.deinit();
        inline for (.{ "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk" }) |f| if (@field(self, f)) |*l| l.deinit();
        self.q_norm.free();
        self.kv_norm.free();
        self.sink.free();
        inline for (.{ "comp_norm", "idx_k_norm", "idx_proj_h" }) |f| if (@field(self, f)) |*b| b.free();
    }

    pub fn scratchWords(self: *const Layer, rows: usize) usize {
        return self.params().scratchWords(rows);
    }

    pub fn params(self: *const Layer) Params {
        return .{
            .wq_a = &self.wq_a,
            .wq_b = &self.wq_b,
            .wkv = &self.wkv,
            .wo_b = &self.wo_b,
            .wo_a = .{ &self.wo_a[0], &self.wo_a[1], &self.wo_a[2], &self.wo_a[3] },
            .comp_wkv = if (self.comp_wkv) |*l| l else null,
            .comp_wgate = if (self.comp_wgate) |*l| l else null,
            .idx_wq_b = if (self.idx_wq_b) |*l| l else null,
            .idx_wk = if (self.idx_wk) |*l| l else null,
            .q_norm = self.q_norm.ptr,
            .kv_norm = self.kv_norm.ptr,
            .sink = self.sink.ptr,
            .comp_norm = if (self.comp_norm) |b| b.ptr else 0,
            .idx_k_norm = if (self.idx_k_norm) |b| b.ptr else 0,
            .idx_proj_h = if (self.idx_proj_h) |b| b.ptr else 0,
        };
    }
};

/// What a forward reads of a layer: a view over an owned `Layer` or over cuda_weights' layer (the loader's), the
/// EXL3 layers by pointer and the plain tensors by address.
pub const Params = struct {
    wq_a: *const exl3.Linear,
    wq_b: *const exl3.Linear,
    wkv: *const exl3.Linear,
    wo_b: *const exl3.Linear,
    wo_a: [4]*const exl3.Linear,
    comp_wkv: ?*const exl3.Linear = null,
    comp_wgate: ?*const exl3.Linear = null,
    idx_wq_b: ?*const exl3.Linear = null,
    idx_wk: ?*const exl3.Linear = null,
    q_norm: Ptr,
    kv_norm: Ptr,
    sink: Ptr,
    comp_norm: Ptr = 0,
    idx_k_norm: Ptr = 0,
    idx_proj_h: Ptr = 0, // fp16 [idx_heads, d]

    /// fp32 words of split-K scratch the widest of these layers needs for `rows` rows.
    pub fn scratchWords(self: Params, rows: usize) usize {
        var most: usize = 0;
        most = @max(most, self.wq_a.scratchWords(rows));
        most = @max(most, self.wq_b.scratchWords(rows));
        most = @max(most, self.wkv.scratchWords(rows));
        most = @max(most, self.wo_b.scratchWords(rows));
        for (self.wo_a) |w| most = @max(most, w.scratchWords(rows));
        inline for (.{ "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk" }) |f| if (@field(self, f)) |l| {
            most = @max(most, l.scratchWords(rows));
        };
        return most;
    }
};

/// The cache planes the sublayer reads and writes: this layer's window ring and compressor inputs, and the
/// compressed and indexer planes of the layer's kv source (its own when it is one). Packed FP4 (KV_QUANT).
pub const Caches = struct {
    ring: cuda.DeviceBuffer, // bf16 [ring, hd]
    comp_codes: cuda.DeviceBuffer, // u8 [rows, hd / 2]
    comp_scales: cuda.DeviceBuffer, // u8 [rows, hd / 16] (E4M3)
    idxk_codes: cuda.DeviceBuffer, // u8 [rows, idx_dim / 2]
    idxk_scales: cuda.DeviceBuffer, // u8 [rows, idx_dim / 32] (E8M0)
    raw_kv: cuda.DeviceBuffer, // fp32 [raw, hd] (ratio > 1)
    raw_score: cuda.DeviceBuffer, // fp32 [raw, hd]
    rows: usize, // compressed rows held: cap / ratio + 2, the last one the scratch row

    pub fn deinit(self: *Caches) void {
        self.ring.free();
        self.comp_codes.free();
        self.comp_scales.free();
        self.idxk_codes.free();
        self.idxk_scales.free();
        self.raw_kv.free();
        self.raw_score.free();
    }
};

/// A rope kind (ops.py `rope_cs`) for positions below `rows`: on the table path its (cos, sin) fp32 [rows, rd / 2]
/// loaded from dump_rope.py's tables; on the device path (`freqs` set: fp32 [rd / 2]) no table at all (cos and sin
/// empty, null pointers), the rows a pass reads computed from `freqs` by RopeRows.
pub const Rope = struct {
    cos: cuda.DeviceBuffer,
    sin: cuda.DeviceBuffer,
    rows: usize,
    freqs: ?cuda.DeviceBuffer = null,
};

/// What the rope kernels take: COS, SIN fp32 [*, rd / 2] and POS int64 [n], row r rotated by row POS[r].
pub const RopeAt = struct {
    cos: Ptr,
    sin: Ptr,
    pos: Ptr,

    /// The rows from row `r` on.
    pub fn from(self: RopeAt, r: usize) RopeAt {
        return .{ .cos = self.cos, .sin = self.sin, .pos = self.pos + r * 8 };
    }
};

/// A pass's rope rows: `at` gives a kind's rows at a position array, on the table path the table itself indexed
/// by the positions, on the device path rows computed into slot `slot` of this scratch (at most `cap` rows) and
/// indexed by 0 .. n - 1. A slot holds one position array's rows until it is computed again.
pub const RopeRows = struct {
    gen: ?*const RopeGen = null,
    iota: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0 }, // int64 0 .. cap - 1
    buf: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0 }, // fp32 [slots][cos, sin][cap, half]
    cap: usize = 0,
    slots: usize = 0,
    half: usize = 0,

    /// Scratch for `slots` position arrays of up to `cap` rows (none without rope.cu: the table path only).
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, glue: *const Glue, cap: usize, slots: usize, half: usize) !RopeRows {
        const gen = if (glue.rope) |*g| g else return .{};
        const ids = try gpa.alloc(i64, cap);
        defer gpa.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(i);
        var iota = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ids));
        errdefer iota.free();
        const buf = try cuda.DeviceBuffer.alloc(d, slots * 2 * cap * half * 4);
        return .{ .gen = gen, .iota = iota, .buf = buf, .cap = cap, .slots = slots, .half = half };
    }

    pub fn deinit(self: *RopeRows) void {
        if (self.gen == null) return;
        self.buf.free();
        self.iota.free();
    }

    /// `rope`'s rows at the positions pos[0 .. n] (int64, on the device, read in stream order: after their upload).
    pub fn at(self: *const RopeRows, rope: Rope, slot: usize, pos: Ptr, n: usize, stream: cuda.Stream) !RopeAt {
        try self.fill(rope, slot, pos, n, stream);
        return self.view(rope, slot, pos);
    }

    /// The device path: slot `slot` gets `rope`'s rows at pos[0 .. n] (a launch on `stream`); the table path: nothing.
    pub fn fill(self: *const RopeRows, rope: Rope, slot: usize, pos: Ptr, n: usize, stream: cuda.Stream) !void {
        const fr = rope.freqs orelse return;
        const gen = self.gen orelse return error.NoRopeRows;
        if (slot >= self.slots or n > self.cap or fr.len != self.half * 4) return error.BadShape;
        const v = self.view(rope, slot, pos);
        try gen.rows(pos, 0, n, fr.ptr, self.half, v.cos, v.sin, stream);
    }

    /// What `at` returns, without computing anything: the table and `pos`, or slot `slot` and 0 .. cap - 1 (its
    /// rows as the last `fill` of the slot left them).
    pub fn view(self: *const RopeRows, rope: Rope, slot: usize, pos: Ptr) RopeAt {
        if (rope.freqs == null) return .{ .cos = rope.cos.ptr, .sin = rope.sin.ptr, .pos = pos };
        const plane = self.cap * self.half * 4;
        const cos = self.buf.ptr + 2 * slot * plane;
        return .{ .cos = cos, .sin = cos + plane, .pos = self.iota.ptr };
    }
};

/// rope.cu: rows of a rope kind's (cos, sin) table computed on the device, the same bits as ops.py `rope_cs`'s.
pub const RopeGen = struct {
    module: cuda.Module,
    rows_fn: cuda.Function,

    pub const block: u32 = 256; // rope.cu's launch bounds

    pub fn load(d: *const cuda.Driver) !RopeGen {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_rope);
        errdefer module.unload();
        return .{ .module = module, .rows_fn = try module.function("tf_dsv41_rope_rows") };
    }

    pub fn deinit(self: *RopeGen) void {
        self.module.unload();
    }

    /// cos, sin fp32 [n, half]: row r is the table's row at position pos[r] (`pos` int64 [n]; 0: position start + r),
    /// from `freqs` fp32 [half] (dump_rope.py's rope_<kind>_freqs.bin).
    pub fn rows(self: *const RopeGen, pos: Ptr, start: u64, n: usize, freqs: Ptr, half: usize, cos: Ptr, sin: Ptr, stream: cuda.Stream) !void {
        if (n == 0) return;
        const total = n * half;
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(@as(c_longlong, @intCast(start)));
        a.add(@as(c_int, @intCast(n)));
        a.add(freqs);
        a.add(@as(c_int, @intCast(half)));
        a.add(cos);
        a.add(sin);
        try cuda.launch.launch(self.rows_fn, .{ .grid = .{ .x = @intCast((total + block - 1) / block) }, .block = .{ .x = block } }, stream, &a);
    }
};

/// The picks a compressed layer attends through: its own (an index source) or the latest index layer's before it
/// (the fork's shared["topk"]): cidx [n, kk] int64 (-1: none). Past the candidate source they also carry its pool
/// (shared["cand"]): the block mask [n, cand_ld] u8 on the device, or on the host under the host selection.
pub const Picks = struct {
    cidx: Ptr,
    kk: usize,
    cand: Ptr = 0,
    cand_ld: usize = 0,
    cand_h: []const u8 = &.{},

    fn hasCand(self: Picks) bool {
        return self.cand != 0 or self.cand_h.len > 0;
    }
};

/// What a layer's forward writes and reads back within the call: the activations, the split-K partials and the
/// indexer's scores (with their host copies under the host selection). The layers of one decode loop run one after
/// another on one stream and read none of it after their forward returns, so they share one `Scratch`; what outlives
/// a forward (the row tables, the picks, the candidate pool) stays in each layer's `Attention`.
pub const Scratch = struct {
    gpa: std.mem.Allocator,
    xh: cuda.DeviceBuffer, // fp16 [rows, d]: every linear's rotated input
    z: cuda.DeviceBuffer, // fp32 split-K partials
    qa: cuda.DeviceBuffer,
    qr: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    ykv: cuda.DeviceBuffer,
    kv: cuda.DeviceBuffer,
    ckv: cuda.DeviceBuffer, // fp32 [rows, hd] (bf16 at ratio 1)
    cgate: cuda.DeviceBuffer,
    pooled: cuda.DeviceBuffer,
    lat: cuda.DeviceBuffer,
    lat2: cuda.DeviceBuffer,
    kraw: cuda.DeviceBuffer,
    k: cuda.DeviceBuffer,
    iq: cuda.DeviceBuffer,
    iqq: cuda.DeviceBuffer,
    wl: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    score: cuda.DeviceBuffer, // fp32 [rows, nb] of the widest index layer
    keys: cuda.DeviceBuffer, // int64 [rows, nb]: the fused selection's keys (where it applies)
    pm: cuda.DeviceBuffer,
    pl: cuda.DeviceBuffer,
    po: cuda.DeviceBuffer,
    o: cuda.DeviceBuffer,
    og: cuda.DeviceBuffer,
    ug: cuda.DeviceBuffer,
    u: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    score_h: []f32, // the host selection's: [rows, nb]
    keys_h: []i64, // [nb]

    /// What the layers sharing a scratch need beyond their common sizes (`Dims` but ratio, sources and pool).
    pub const Need = struct {
        z_words: usize = 0, // the widest EXL3 layer's split-K words at max_rows (`Params.scratchWords`)
        nb: usize = 0, // the widest index scan (compressed entries a row); 0: no index layer
        keys_nb: usize = 0, // the widest scan the fused selection keys (a power of two up to 4096); 0: none
        host: bool = false, // the host selection (`Glue.hostSelect`): host copies of the scores

        /// One layer's: its index scan only where it is an index layer.
        pub fn of(dims: Dims, z_words: usize, host: bool) Need {
            const nb = if (dims.index_source) dims.nb() else 0;
            const fused = nb > 0 and nb <= 4096 and (nb & (nb - 1)) == 0;
            return .{ .z_words = z_words, .nb = nb, .keys_nb = if (fused) nb else 0, .host = host };
        }

        pub fn merge(a: Need, b: Need) Need {
            return .{ .z_words = @max(a.z_words, b.z_words), .nb = @max(a.nb, b.nb), .keys_nb = @max(a.keys_nb, b.keys_nb), .host = a.host or b.host };
        }
    };

    /// Buffers for `dims.max_rows` rows of layers of `dims`' common sizes, as `need` asks.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, dims: Dims, need: Need) !Scratch {
        const r = dims.max_rows;
        if (r > 16) return error.TooManyRows;
        const groups_h = dims.heads / 16;
        var self: Scratch = undefined;
        self.gpa = gpa;
        self.xh = try cuda.DeviceBuffer.alloc(d, r * dims.d * 2);
        self.z = try cuda.DeviceBuffer.alloc(d, @max(16, need.z_words * 4));
        self.qa = try cuda.DeviceBuffer.alloc(d, r * dims.q_rank * 2);
        self.qr = try cuda.DeviceBuffer.alloc(d, r * dims.q_rank * 2);
        self.q = try cuda.DeviceBuffer.alloc(d, r * dims.heads * dims.hd * 2);
        self.ykv = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.kv = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.ckv = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 4);
        self.cgate = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 4);
        self.pooled = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.lat = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.lat2 = try cuda.DeviceBuffer.alloc(d, r * dims.hd * 2);
        self.kraw = try cuda.DeviceBuffer.alloc(d, r * dims.idx_dim * 2);
        self.k = try cuda.DeviceBuffer.alloc(d, r * dims.idx_dim * 2);
        self.iq = try cuda.DeviceBuffer.alloc(d, r * dims.idx_heads * dims.idx_dim * 2);
        self.iqq = try cuda.DeviceBuffer.alloc(d, r * dims.idx_heads * dims.idx_dim * 2);
        self.wl = try cuda.DeviceBuffer.alloc(d, r * dims.idx_heads * 4);
        self.wts = try cuda.DeviceBuffer.alloc(d, r * dims.idx_heads * 2);
        self.score = try cuda.DeviceBuffer.alloc(d, @max(16, r * need.nb * 4));
        self.keys = try cuda.DeviceBuffer.alloc(d, @max(16, r * need.keys_nb * 8));
        self.pm = try cuda.DeviceBuffer.alloc(d, r * groups_h * 8 * 16 * 4);
        self.pl = try cuda.DeviceBuffer.alloc(d, r * groups_h * 8 * 16 * 4);
        self.po = try cuda.DeviceBuffer.alloc(d, r * groups_h * 8 * 16 * dims.hd * 4);
        self.o = try cuda.DeviceBuffer.alloc(d, r * dims.heads * dims.hd * 2);
        self.og = try cuda.DeviceBuffer.alloc(d, r * (dims.heads * dims.hd / dims.groups) * 2);
        self.ug = try cuda.DeviceBuffer.alloc(d, r * dims.o_rank * 2);
        self.u = try cuda.DeviceBuffer.alloc(d, r * dims.groups * dims.o_rank * 2);
        self.out = try cuda.DeviceBuffer.alloc(d, r * dims.d * 4);
        self.score_h = try gpa.alloc(f32, if (need.host) @max(1, r * need.nb) else 0);
        self.keys_h = try gpa.alloc(i64, if (need.host) @max(1, need.nb) else 0);
        return self;
    }

    pub fn deinit(self: *Scratch) void {
        inline for (.{ "xh", "z", "qa", "qr", "q", "ykv", "kv", "ckv", "cgate", "pooled", "lat", "lat2", "kraw", "k", "iq", "iqq", "wl", "wts", "score", "keys", "pm", "pl", "po", "o", "og", "ug", "u", "out" }) |f| @field(self, f).free();
        self.gpa.free(self.score_h);
        self.gpa.free(self.keys_h);
    }
};

/// The sublayer's launcher for up to `dims.max_rows` rows: its row tables, picks and candidate pool, and the
/// `Scratch` it shares with the other layers of its loop (or owns, from `init`).
pub const Attention = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    dims: Dims,
    set: *const aot.Set,
    glue: *const Glue,
    s: *Scratch,
    own: ?*Scratch, // `s` when this layer owns it
    // per-row tables (int64), uploaded a forward
    pos: cuda.DeviceBuffer,
    slots: cuda.DeviceBuffer,
    vis: cuda.DeviceBuffer,
    first: cuda.DeviceBuffer,
    target: cuda.DeviceBuffer,
    zero: cuda.DeviceBuffer,
    cidx: cuda.DeviceBuffer, // int64 [rows, kk]: the picks (an index layer's are read by the layers after it)
    smax: cuda.DeviceBuffer, // fp32 [rows, cand cols]: the candidate source's block maxima
    cand_mask: cuda.DeviceBuffer, // u8 [rows, cand cols]: the candidate source's pool
    // host side
    tab: []i64, // 5 x max_rows
    uploads: Uploads, // the row tables' and the picks' copies
    cidx_h: []i64,
    cand_h: []u8, // the pool on the host (the candidate source under the host selection)
    cand_from: Picks, // the pool this layer's scores were masked to (an index layer after the source)

    /// A layer with a scratch of its own. `z_words`: fp32 words of split-K scratch the widest EXL3 layer of this kind
    /// needs at `dims.max_rows` rows (`Params.scratchWords`).
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, dims: Dims, set: *const aot.Set, glue: *const Glue, z_words: usize) !Attention {
        const s = try gpa.create(Scratch);
        errdefer gpa.destroy(s);
        s.* = try Scratch.init(gpa, d, dims, Scratch.Need.of(dims, z_words, glue.hostSelect()));
        errdefer s.deinit();
        var self = try initShared(gpa, d, dims, set, glue, s);
        self.own = s;
        return self;
    }

    /// A layer on `s`, which outlives it and was made for a `Need` covering this layer's (`Scratch.Need.of`).
    pub fn initShared(gpa: std.mem.Allocator, d: *const cuda.Driver, dims: Dims, set: *const aot.Set, glue: *const Glue, s: *Scratch) !Attention {
        const r = dims.max_rows;
        if (r > 16) return error.TooManyRows;
        const kk = dims.kk();
        const host = glue.hostSelect();
        var self: Attention = undefined;
        self.gpa = gpa;
        self.d = d;
        self.dims = dims;
        self.set = set;
        self.glue = glue;
        self.s = s;
        self.own = null;
        self.pos = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.slots = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.vis = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.first = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.target = try cuda.DeviceBuffer.alloc(d, r * 8);
        self.zero = try cuda.DeviceBuffer.alloc(d, 8);
        try self.zero.fill8(0, null);
        self.cidx = try cuda.DeviceBuffer.alloc(d, @max(16, if (dims.index_source) r * kk * 8 else 0));
        const ccols = if (dims.cand == .source) dims.candCols() else 0;
        self.smax = try cuda.DeviceBuffer.alloc(d, @max(16, r * ccols * 4));
        self.cand_mask = try cuda.DeviceBuffer.alloc(d, @max(16, r * ccols));
        self.tab = try gpa.alloc(i64, 5 * r);
        self.cidx_h = try gpa.alloc(i64, if (host) @max(1, r * kk) else 0);
        self.cand_h = try gpa.alloc(u8, if (host) r * ccols else 0);
        self.cand_from = .{ .cidx = 0, .kk = 0 };
        self.uploads = try Uploads.init(d, Uploads.room(5, r * 8) + Uploads.room(1, r * kk * 8));
        return self;
    }

    pub fn deinit(self: *Attention) void {
        inline for (.{ "pos", "slots", "vis", "first", "target", "zero", "cidx", "smax", "cand_mask" }) |f| @field(self, f).free();
        self.gpa.free(self.tab);
        self.gpa.free(self.cand_h);
        self.gpa.free(self.cidx_h);
        self.uploads.deinit();
        if (self.own) |s| {
            s.deinit();
            self.gpa.destroy(s);
        }
    }

    /// Rows x bf16 [n, d] at positions start .. start + n - 1 -> this rank's fp32 partial [n, d] (`self.s.out`),
    /// the caches updated as the fork's; `self.cidx` holds the picks (synchronous only under the host selection).
    /// `given`: the latest index layer's picks, which a layer without an indexer attends through (required when
    /// the layer is compressed and not an index source) and whose candidate pool an index layer after the candidate
    /// source masks its scores to (required there); an index source makes its own picks (`self.picks()`).
    pub fn forward(self: *Attention, p: Params, c: *const Caches, rope: Rope, rr: *const RopeRows, x: Ptr, n: usize, start: usize, given: ?Picks, stream: cuda.Stream) !Ptr {
        const dm = self.dims;
        if (n < 1 or n > dm.max_rows) return error.TooManyRows;
        const ratio = dm.ratio;
        // -- the row tables ---------------------------------------------------------------------------------------
        const r = dm.max_rows;
        const pos = self.tab[0..r];
        const slots = self.tab[r .. 2 * r];
        const vis = self.tab[2 * r .. 3 * r];
        const first = self.tab[3 * r .. 4 * r];
        const target = self.tab[4 * r .. 5 * r];
        const scratch_row: i64 = @intCast(c.rows - 1);
        for (0..n) |i| {
            const pv: i64 = @intCast(start + i);
            pos[i] = pv;
            slots[i] = @mod(pv, @as(i64, @intCast(dm.ring)));
            if (ratio > 0) {
                const rt: i64 = @intCast(ratio);
                vis[i] = @divFloor(pv + 1, rt);
                const g = @divFloor(pv, rt);
                first[i] = g * rt;
                target[i] = if (@mod(pv + 1, rt) == 0) g else scratch_row;
            }
        }
        try self.uploads.begin();
        try self.uploads.put(self.pos, 0, std.mem.sliceAsBytes(pos[0..n]), stream);
        try self.uploads.put(self.slots, 0, std.mem.sliceAsBytes(slots[0..n]), stream);
        if (ratio > 0) {
            try self.uploads.put(self.vis, 0, std.mem.sliceAsBytes(vis[0..n]), stream);
            try self.uploads.put(self.first, 0, std.mem.sliceAsBytes(first[0..n]), stream);
            try self.uploads.put(self.target, 0, std.mem.sliceAsBytes(target[0..n]), stream);
        }
        const at_pos = try rr.at(rope, 0, self.pos.ptr, n, stream);
        const at_first: RopeAt = if (ratio > 0 and dm.kv_source) try rr.at(rope, 1, self.first.ptr, n, stream) else undefined;
        // -- attn_in: wq_a, wkv (and the compressor's) on x ------------------------------------------------------
        try p.wq_a.run(x, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.qa.ptr, .bf16, stream);
        try p.wkv.run(x, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.ykv.ptr, .bf16, stream);
        if (ratio > 0 and dm.kv_source) {
            try p.comp_wkv.?.run(x, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.ckv.ptr, if (ratio == 1) .bf16 else .f32, stream);
            if (ratio > 1) try p.comp_wgate.?.run(x, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.cgate.ptr, .f32, stream);
        }
        // -- q: RMSNorm, wq_b, RoPE on this rank's heads ----------------------------------------------------------
        try self.rmsnorm(self.s.qa.ptr, p.q_norm, self.s.qr.ptr, n, dm.q_rank, stream);
        try p.wq_b.run(self.s.qr.ptr, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.q.ptr, .bf16, stream);
        try self.ropeHeads(self.s.q.ptr, at_pos, n, dm.heads, dm.hd, false, stream);
        // -- the window KV into the ring ------------------------------------------------------------------------
        try self.set.run(stream, "_kv_norm_rope", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("Y", "*bf16", self.s.ykv.ptr),  aot.ptr("W", "*bf16", p.kv_norm), aot.ptr("COS", "*fp32", at_pos.cos),
            aot.ptr("SIN", "*fp32", at_pos.sin), aot.ptr("POS", "*i64", at_pos.pos),     aot.ptr("OUT", "*bf16", self.s.kv.ptr),
            aot.ptr("RING", "*bf16", c.ring.ptr),  aot.ptr("SLOT_OF", "*i64", self.slots.ptr), aot.int("ring_size", @intCast(dm.ring)),
            aot.float("eps", dm.eps),
        }, &.{ aot.ci("QUANT", 1), aot.ci("D", @intCast(dm.hd)), aot.ci("RD", @intCast(dm.rd)), aot.ci("PDL", 1) });
        if (ratio > 0) {
            // -- the kv source: compressed latents (and index keys) of the rows' groups --------------------------
            if (dm.kv_source) {
                if (ratio == 1) {
                    try self.rmsnorm(self.s.ckv.ptr, p.comp_norm, self.s.lat.ptr, n, dm.hd, stream);
                } else {
                    // the rows' compressor inputs by position, then each group's softmax pool (attn_glue comp_pool2)
                    for (0..n) |i| {
                        const slot: usize = @intCast(@mod(pos[i], @as(i64, @intCast(dm.raw))));
                        try c.raw_kv.copyFrom(slot * dm.hd * 4, try self.s.ckv.at(i * dm.hd * 4), dm.hd * 4, stream.handle);
                        try c.raw_score.copyFrom(slot * dm.hd * 4, try self.s.cgate.at(i * dm.hd * 4), dm.hd * 4, stream.handle);
                    }
                    if (ratio != 2) return error.BadShape;
                    var a: cuda.Args = .{};
                    a.add(c.raw_kv.ptr);
                    a.add(c.raw_score.ptr);
                    a.add(self.first.ptr);
                    a.add(@as(c_int, @intCast(dm.raw)));
                    a.add(@as(c_int, @intCast(dm.hd)));
                    a.add(self.s.pooled.ptr);
                    try cuda.launch.launch(self.glue.comp_pool2, .{ .grid = .{ .x = @intCast((dm.hd + 255) / 256), .y = @intCast(n) }, .block = .{ .x = 256 } }, stream, &a);
                    try self.rmsnorm(self.s.pooled.ptr, p.comp_norm, self.s.lat.ptr, n, dm.hd, stream);
                }
                if (dm.has_idx_wk) {
                    try p.idx_wk.?.run(self.s.lat.ptr, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.kraw.ptr, .bf16, stream);
                    try self.rmsnorm(self.s.kraw.ptr, p.idx_k_norm, self.s.k.ptr, n, dm.idx_dim, stream);
                    try self.ropeHeads(self.s.k.ptr, at_first, n, 1, dm.idx_dim, false, stream);
                    try self.fp4Store(self.s.k.ptr, dm.idx_dim, c.idxk_codes.ptr, c.idxk_scales.ptr, self.target.ptr, n, 32, false, stream);
                }
                try self.s.lat2.copyFrom(0, self.s.lat.ptr, n * dm.hd * 2, stream.handle);
                try self.ropeHeads(self.s.lat2.ptr, at_first, n, 1, dm.hd, false, stream);
                try self.fp4Store(self.s.lat2.ptr, dm.hd, c.comp_codes.ptr, c.comp_scales.ptr, self.target.ptr, n, 16, true, stream);
            }
            // -- the indexer: scores over the visible compressed entries, the top-k picks -------------------------
            if (dm.index_source) {
                const nb = dm.nb();
                const kk = dm.kk();
                try p.idx_wq_b.?.run(self.s.qr.ptr, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.iq.ptr, .bf16, stream);
                try self.ropeHeads(self.s.iq.ptr, at_pos, n, dm.idx_heads, dm.idx_dim, false, stream);
                const nblk: usize = n * dm.idx_heads * dm.idx_dim / 32;
                try self.set.run(stream, "_fp4qd_p2", .{ @intCast((nblk + 31) / 32), 1, 1 }, &.{
                    aot.ptr("X", "*bf16", self.s.iq.ptr), aot.ptr("OUT", "*bf16", self.s.iqq.ptr), aot.int("nblk", @intCast(nblk)),
                }, &.{ aot.ci("BLKS", 32), aot.ci("PDL", 1) });
                try self.set.run(stream, "_rowmm", .{ @intCast(n), @intCast((dm.idx_heads + 7) / 8), 1 }, &.{
                    aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(dm.d)), aot.ptr("W", "*fp16", p.idx_proj_h), aot.ptr("OUT", "*fp32", self.s.wl.ptr),
                }, &.{ aot.ci("K", @intCast(dm.d)), aot.ci("N", @intCast(dm.idx_heads)), aot.ci("BN", 8), aot.ci("BK", 256), aot.ci("PDL", 1) });
                {
                    const scale: f32 = @floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(dm.idx_dim))) * @sqrt(@as(f64, @floatFromInt(dm.idx_heads)))));
                    const count: c_int = @intCast(n * dm.idx_heads);
                    var a: cuda.Args = .{};
                    a.add(self.s.wl.ptr);
                    a.add(self.s.wts.ptr);
                    a.add(scale);
                    a.add(count);
                    try cuda.launch.launch(self.glue.bf16_scale, .{ .grid = .{ .x = @intCast((n * dm.idx_heads + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
                }
                const nt: usize = (nb + 63) / 64;
                try self.set.run(stream, "_index_score", .{ @intCast(n), @intCast(nt), 1 }, &.{
                    aot.ptr("Q", "*bf16", self.s.iqq.ptr),     aot.ptr("K", "*u8", c.idxk_codes.ptr),  aot.ptr("KS", "*u8", c.idxk_scales.ptr),
                    aot.ptr("Wt", "*bf16", self.s.wts.ptr),    aot.ptr("VIS", "*i64", self.vis.ptr),   aot.ptr("OUT", "*fp32", self.s.score.ptr),
                    aot.int("n", @intCast(nb)),              aot.ptr("KB", "*i64", self.vis.ptr),    aot.ptr("CAND", "*i64", self.vis.ptr),
                    aot.int("cs", 0),                        aot.ptr("TMAX", "*fp32", self.s.score.ptr), aot.int("nt", @intCast(nt)),
                }, &.{
                    aot.ci("IH", @intCast(dm.idx_heads)), aot.ci("ID", @intCast(dm.idx_dim)), aot.ci("BN", 64),      aot.ci("PACKED", 1),
                    aot.ci("HAS_BASE", 0),                aot.ci("PDL", 1),                    aot.ci("HAS_CAND", 0), aot.ci("CB", 8),
                    aot.ci("KEYS", 0),                    aot.ci("HAS_TMAX", 0),               aot.ci("SKIP_DEAD", 1), aot.ci("DEAD_PAD", 0),
                });
                try self.select(n, nb, kk, vis[0..n], given, stream);
            }
        }
        // -- sparse attention over the window ring and the picks, 8 key splits a head group, then the merge --------
        {
            const groups_h = dm.heads / 16;
            const has = ratio > 0; // no compressed source: the wrapper passes the window source (bf16) for COMP and CSC
            const pk: Picks = if (dm.index_source) .{ .cidx = self.cidx.ptr, .kk = dm.kk() } else if (has) (given orelse return error.BadShape) else .{ .cidx = self.pos.ptr, .kk = 0 };
            const kk = pk.kk;
            const pick_blocks: usize = if (has) (kk + 31) / 32 else 0;
            const nblk: usize = dm.window / 32 + pick_blocks;
            const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dm.hd))));
            try self.set.run(stream, "_sparse_attn_part", .{ @intCast(n), @intCast(groups_h), 8 }, &.{
                aot.ptr("Q", "*bf16", self.s.q.ptr),       aot.ptr("WSRC", "*bf16", c.ring.ptr),
                aot.ptr("WLO", "*i64", self.zero.ptr),   aot.ptr("COMP", if (has) "*u8" else "*bf16", if (has) c.comp_codes.ptr else c.ring.ptr),
                aot.ptr("CSC", if (has) "*u8" else "*bf16", if (has) c.comp_scales.ptr else c.ring.ptr), aot.ptr("IDX", "*i64", if (has) pk.cidx else self.pos.ptr),
                aot.ptr("POS", "*i64", self.pos.ptr),    aot.ptr("PM", "*fp32", self.s.pm.ptr),
                aot.ptr("PL", "*fp32", self.s.pl.ptr),     aot.ptr("PO", "*fp32", self.s.po.ptr),
                aot.ptr("SINK", "*fp32", p.sink),  aot.ptr("OUT", "*bf16", self.s.o.ptr),
                aot.float("scale", scale),               aot.int("ring_size", @intCast(dm.ring)),
                aot.int("n_idx", @intCast(kk)),          aot.ptr("RBASE", "*i64", self.pos.ptr),
                aot.ptr("CBASE", "*i64", self.pos.ptr),
            }, &.{
                aot.ci("H", @intCast(dm.heads)), aot.ci("HD", @intCast(dm.hd)),     aot.ci("HB", 16),    aot.ci("WIN", @intCast(dm.window)),
                aot.ci("BN", 32),                aot.ci("RING", 1),                   aot.ci("HAS_COMP", @intFromBool(has)), aot.ci("PACKED", @intFromBool(has)),
                aot.ci("SPLITS", 8),             aot.ci("NBLK", @intCast(nblk)),      aot.ci("FINAL", 0),  aot.ci("HAS_BASE", 0),
                aot.ci("PDL", 1),                aot.ci("WPF", 1),
            });
            try self.set.run(stream, "_sparse_attn_merge", .{ @intCast(n), @intCast(groups_h), 1 }, &.{
                aot.ptr("PM", "*fp32", self.s.pm.ptr), aot.ptr("PL", "*fp32", self.s.pl.ptr), aot.ptr("PO", "*fp32", self.s.po.ptr),
                aot.ptr("SINK", "*fp32", p.sink), aot.ptr("OUT", "*bf16", self.s.o.ptr),
            }, &.{ aot.ci("H", @intCast(dm.heads)), aot.ci("HD", @intCast(dm.hd)), aot.ci("HB", 16), aot.ci("SPLITS", 8), aot.ci("PDL", 1) });
        }
        try self.ropeHeads(self.s.o.ptr, at_pos, n, dm.heads, dm.hd, true, stream);
        // -- wo_a's slices on o's column blocks, then wo_b: this rank's fp32 partial ------------------------------
        const o_cols = dm.heads * dm.hd;
        const g_cols = o_cols / dm.groups;
        for (0..dm.groups) |g| {
            try self.copyCols(self.s.o.ptr + g * g_cols * 2, o_cols, self.s.og.ptr, g_cols, n, g_cols, stream);
            try p.wo_a[g].run(self.s.og.ptr, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.ug.ptr, .bf16, stream);
            try self.copyCols(self.s.ug.ptr, dm.o_rank, self.s.u.ptr + g * dm.o_rank * 2, dm.groups * dm.o_rank, n, dm.o_rank, stream);
        }
        try p.wo_b.run(self.s.u.ptr, .bf16, n, self.s.xh.ptr, self.s.z.ptr, self.s.out.ptr, .f32, stream);
        return self.s.out.ptr;
    }

    /// The picks this index layer made in its last forward, for the layers after it, with the candidate pool it
    /// made (the source) or masked to (after the source).
    pub fn picks(self: *const Attention) Picks {
        var p: Picks = .{ .cidx = self.cidx.ptr, .kk = self.dims.kk() };
        switch (self.dims.cand) {
            .source => {
                p.cand_ld = self.dims.candCols();
                if (self.glue.hostSelect()) p.cand_h = self.cand_h else p.cand = self.cand_mask.ptr;
            },
            .after => {
                p.cand = self.cand_from.cand;
                p.cand_ld = self.cand_from.cand_ld;
                p.cand_h = self.cand_from.cand_h;
            },
            .none => {},
        }
        return p;
    }

    /// The indexer's picks from `self.s.score` [n, nb]: the candidate pool made or applied first, then each row's kk
    /// highest scores' indices ascending, -1 at or past its visible count, into `self.cidx` [n, kk].
    fn select(self: *Attention, n: usize, nb: usize, kk: usize, vis: []const i64, given: ?Picks, stream: cuda.Stream) !void {
        const dm = self.dims;
        self.cand_from = .{ .cidx = 0, .kk = 0 };
        if (dm.cand == .after) {
            const pk = given orelse return error.NoCandidateSource;
            if (!pk.hasCand()) return error.NoCandidateSource;
            self.cand_from = pk;
        }
        if (self.glue.hostSelect()) return self.topk(n, nb, kk, vis, stream);
        const tg = &self.glue.topk.?;
        switch (dm.cand) {
            .source => try tg.candidates(self.s.score.ptr, nb, n, nb, self.vis.ptr, dm.cand_blocks, dm.cand_block, self.smax.ptr, self.cand_mask.ptr, dm.candCols(), stream),
            .after => {
                if (self.cand_from.cand == 0) return error.NoCandidateSource;
                try tg.apply(self.s.score.ptr, nb, n, nb, self.cand_from.cand, self.cand_from.cand_ld, dm.cand_block, stream);
            },
            .none => {},
        }
        if (self.glue.select == .auto and try fusedTopk(self.set, self.s.score.ptr, n, nb, kk, self.vis.ptr, self.s.keys.ptr, self.cidx.ptr, stream)) return;
        try tg.topk(self.s.score.ptr, nb, n, nb, kk, self.vis.ptr, self.cidx.ptr, kk, stream);
    }

    fn rmsnorm(self: *Attention, x: Ptr, w: Ptr, out: Ptr, n: usize, d: usize, stream: cuda.Stream) !void {
        const block = std.math.ceilPowerOfTwo(usize, d) catch return error.BadShape;
        try self.set.run(stream, "_rmsnorm", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("W", "*bf16", w),
            aot.ptr("OUT", "*bf16", out), aot.int("os_", @intCast(d)), aot.float("eps", self.dims.eps),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("PDL", 1) });
    }

    /// RoPE on the last rd dims of every head of x [n, h, hd] bf16 in place, by the rows `at`.
    fn ropeHeads(self: *Attention, x: Ptr, at: RopeAt, n: usize, h: usize, hd: usize, inverse: bool, stream: cuda.Stream) !void {
        const hb = @min(h, 16);
        try self.set.run(stream, "_rope_heads", .{ @intCast(n), @intCast(h / hb), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("COS", "*fp32", at.cos), aot.ptr("SIN", "*fp32", at.sin), aot.ptr("POS", "*i64", at.pos),
        }, &.{ aot.ci("H", @intCast(h)), aot.ci("HD", @intCast(hd)), aot.ci("RD", @intCast(self.dims.rd)), aot.ci("INV", @intFromBool(inverse)), aot.ci("HB", @intCast(hb)), aot.ci("PDL", 1) });
    }

    /// Rows x bf16 [n, d] packed as FP4 (E2M1 codes, E4M3 or E8M0 scales per `block`) into cache rows `rows`.
    fn fp4Store(self: *Attention, x: Ptr, d: usize, codes: Ptr, scales: Ptr, rows: Ptr, n: usize, block: usize, e4m3: bool, stream: cuda.Stream) !void {
        try self.set.run(stream, "_fp4_store", .{ @intCast(n), 1, 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.int("xs", @intCast(d)), aot.ptr("CODES", "*u8", codes), aot.ptr("SCALES", "*u8", scales), aot.ptr("ROWS", "*i64", rows),
        }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(block)), aot.ci("E4M3", @intFromBool(e4m3)), aot.ci("PDL", 1) });
    }

    fn copyCols(self: *Attention, src: Ptr, sld: usize, dst: Ptr, dld: usize, rows: usize, cols: usize, stream: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_int, @intCast(sld)));
        a.add(dst);
        a.add(@as(c_int, @intCast(dld)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(cols)));
        try cuda.launch.launch(self.glue.copy_cols, .{ .grid = .{ .x = @intCast((cols + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    /// The host selection (the oracle): kernels.topk_indices on each row (ties to the lower index, one int64 key a
    /// score: its ordered bits above, the inverted index below), -1 at or past the row's visible count
    /// (`torch.where(top < vis, top, -1)`), the candidate pool made or applied first. Reads the scores back: a
    /// stream sync.
    fn topk(self: *Attention, n: usize, nb: usize, kk: usize, vis: []const i64, stream: cuda.Stream) !void {
        const dm = self.dims;
        try stream.synchronize();
        try self.s.score.download(0, std.mem.sliceAsBytes(self.s.score_h[0 .. n * nb]));
        const ccols = dm.candCols();
        for (0..n) |r| {
            const row = self.s.score_h[r * nb .. (r + 1) * nb];
            switch (dm.cand) {
                .source => hostCandidates(row, vis[r], dm.cand_blocks, dm.cand_block, self.s.keys_h, self.cand_h[r * ccols .. (r + 1) * ccols]),
                .after => {
                    const ld = self.cand_from.cand_ld;
                    if (self.cand_from.cand_h.len < (r + 1) * ld) return error.NoCandidateSource;
                    hostApplyCandidates(row, self.cand_from.cand_h[r * ld .. (r + 1) * ld], dm.cand_block);
                },
                .none => {},
            }
            hostTopk(row, kk, vis[r], self.s.keys_h, self.cidx_h[r * kk .. (r + 1) * kk]);
        }
        try self.uploads.put(self.cidx, 0, std.mem.sliceAsBytes(self.cidx_h[0 .. n * kk]), stream);
    }
};

/// torch.topk's total-order key of a score at index j: the score's bits (negatives inverted, -0 as +0) above the
/// inverted index, as kernels.topk_indices builds it.
pub fn key(score: f32, j: usize) i64 {
    const s = score + 0.0;
    const bits: i32 = @bitCast(s);
    const ordered: i32 = if (bits < 0) bits ^ 0x7FFFFFFF else bits;
    return (@as(i64, ordered) << 32) | @as(i64, @intCast(0xFFFFFFFF - j));
}

test "the top-k key orders scores and breaks ties by the lower index" {
    try std.testing.expect(key(1.0, 3) > key(0.5, 0));
    try std.testing.expect(key(-0.5, 0) > key(-1.0, 0));
    try std.testing.expect(key(0.0, 1) > key(-0.0, 2)); // the same score: the lower index wins
    try std.testing.expect(key(2.0, 7) > key(2.0, 8));
    try std.testing.expect(key(-3.0, 1) < key(0.0, 500));
}
