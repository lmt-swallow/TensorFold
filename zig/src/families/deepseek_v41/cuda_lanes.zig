//! DeepSeek-V4.1 behind the lane core on two ranks, up to `max_slots` streams at once, greedy: rank 0 serves and
//! sends every step (an admission, a round, its drafts and keeps, a release) to rank 1 over the communicator's
//! control link before running it; rank 1 runs the same steps in `follow`, so both ranks issue the same
//! all-gathers in the same order (model.py's lockstep). A sha256 digest of each step is all-gathered first
//! (multi.py `_agree`): ranks that planned different steps, or a rank that cannot run one, stop instead of pairing
//! collectives wrongly.
//!
//! The streams share one window of `pool + extra` positions (multi.py MultiDecoder over model.py's PoolCache,
//! `Streams`; `pool` is TF_DS_POOL, the context unless set wider): a stream takes the lowest free slot (its window
//! rings, compressor inputs and drafter rings) and the first free extent of the window aligned to 2048 that holds its
//! prompt, its reply and a round's rows (`Extents`), an extent of at most the context's positions (`Streams.most`):
//! a stream's positions, RoPE rows and round buckets are a lone stream's of that context whatever the pool.
//! Its admission fills the extent from position 0 through the prompt path (cuda_prompt.zig) on a view of the pool,
//! in chunks of TF_DS_PREFILL_CHUNK rows (default 2048; engine.py `prefill_steps`), each chunk's taps absorbed into
//! its slot's drafter rings, and draws its first token from the last row. A round runs every stream's window (its
//! pending token, then its held DSpark drafts or host tokens) through one forward (rounds.py RoundRunner: every row
//! at its own stream's position and cache rows, 16 rows at most); every row's bits are the ones the row gets alone,
//! so a stream's reply does not depend on the others. A drafts step absorbs each stream's kept rows' taps into its
//! rings (which keeps those rows) and drafts the drafting streams' blocks in batched passes over the DraftPool
//! (cuda_dspark.zig); a keeps step sets the other streams' lengths to their kept rows (the rejected rows' cache
//! writes stay and are overwritten before any later row reads them, as in the fork). Drafts are deterministic given
//! the same state, so the records carry ids and positions only; every rank drafts the same tokens. Sampling above
//! temperature 0 is refused (the keyed draw is not ported).
//!
//! Kept prompts (cuda_keep.zig, multi.py `Kept`, TF_DS_KEEP=1 by default): a finished stream's prompt rows stay in
//! the window with ring snapshots at its chunk boundaries, and a later prompt that starts the same is admitted at the
//! longest boundary they share (its rows copied or its extent grown in place, its slot's rings restored), prefilling
//! only the rest; usage's cached tokens are that boundary. Rank 0 matches and sends the kept prompt and boundary in
//! the admission record and the kept prompts a finished prompt covers in the free record; every rank places, keeps
//! and evicts by the same book, whose fingerprint each record carries.
//!
//! Engram (cuda_engram_apply.zig) runs on its layers when TF_DS_ENGRAM names the tables' dir (the token map from
//! TF_DS_TOKEN_MAP, default `<kernels>/token_map.json`; TF_DS_ENGRAM_THREADS reader threads, default 32): every rank
//! keeps each stream's ids on the host (`Lane.hist`) and hashes them the same way, so the ranks' Engram gathers
//! pair. Without the tables the gate is skipped, as the fork does, and the output is degraded.
//!
//! `Sequence` is one sequence on caches of its own (the prompt path and the decode layer loop); the fixture tests
//! of those paths drive it with the `Prefill`, `Verify`, `Keep` and `Draft` records.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const exl3 = @import("cuda_exl3.zig");
const experts = @import("cuda_experts.zig");
const attention = @import("cuda_attention.zig");
const forward = @import("cuda_forward.zig");
const prompt = @import("cuda_prompt.zig");
const engram_apply = @import("cuda_engram_apply.zig");
const dspark = @import("cuda_dspark.zig");
const rounds = @import("cuda_rounds.zig");
pub const kept_prompts = @import("cuda_keep.zig");
const RankCache = @import("rank_cache.zig").RankCache;
const mem_note = @import("mem_note.zig");
const draft_stats = @import("draft_stats.zig");

const be = lanes.backend;
const Ptr = cuda.abi.DevicePtr;

/// A lone sequence's window (`Sequence`, the decode loop's `Forward`) takes at most this many rows; so does each
/// stream's window in a round, which keeps the extents' and RoPE tables' extra rows.
pub const max_rows: usize = 16;
/// The rows a round's records and buffers take at most; a served round takes at most `Settings.rows`
/// (TF_DS_MAX_ROWS) of them.
pub const round_rows_most: usize = rounds.max_rows;
/// Positions past the context a sequence's RoPE tables and caches hold (engine.py: limit + max_rows + 8).
pub const extra: usize = max_rows + 8;
pub const rope_align: usize = 4096;

pub const Error = error{ WorldNotTwo, BadMaster, NoRankCache, NoRopeTables, RopeRowsShort, OutOfStep, LinkClosed, SamplingNotSupported, WindowTooWide, PositionMismatch, UnknownStream, StreamBusy, PromptTooLong, Cancelled, NoDraftHead, NotPipelined, TreesNotBuilt, EmptyPath, BadChunk, BadRecord, NoHeldDrafts, NoFirstToken, BadTapLayers, BadSlots, NoRoom };

/// The widest bucket whose round graphs are recorded at load (`Streams.warm`) and kept; the keys of wider buckets
/// are recorded on first use under TF_DS_GRAPH_MIB (`RoundRunner.graph_budget`: a bucket's 16 row counts take about
/// 0.65 GiB).
pub const warm_bucket_max: usize = 8192;

/// The most rows of the round graphs recorded at load (`Streams.warm`) and kept; wider rounds' graphs are recorded on
/// first use under TF_DS_GRAPH_MIB, as the wider buckets' are (their row counts are few in a served run).
pub const warm_rows_max: usize = rounds.rows_default;

/// A drafting round's widest window by default: the pending token and DSpark's block of drafts (engine.py: drafts
/// + 1 rows); TF_DS_DRAFTS sets the drafts (`Settings.most_drafts`).
pub const draft_width: usize = 6;

/// The drafts a window holds at most (TF_DS_DRAFTS): a window is at most a round's rows.
pub const most_drafts_max: usize = max_rows - 1;

/// The DSpark head's acceptance at depth 1, 2, ... given the ones before it, until a stream measures its own (the
/// fork's served acceptance is about 0.75 a depth), to the deepest window TF_DS_DRAFTS allows.
pub const draft_prior: [most_drafts_max]f64 = @splat(0.75);

/// Prompt rows a chunk unless TF_DS_PREFILL_CHUNK says otherwise (the fork's production setting).
pub const default_chunk: usize = 2048;
/// The widest prompt chunk TF_DS_PREFILL_CHUNK may name.
pub const max_chunk: usize = prompt.max_chunk;

/// Step opcodes. The served engine sends `measure` and `admit` .. `fill`; the others are `Sequence`'s records.
pub const Op = enum(u8) { prefill = 1, round = 2, release = 3, stop = 4, verify = 5, keep = 6, draft = 7, measure = 8, admit = 9, rounds = 10, drafts = 11, keeps = 12, free = 13, fill = 14 };

/// The bytes a step's record takes at most for a context of `context` positions (a prompt of the whole context).
pub fn maxOpBytes(context: usize) usize {
    return @max(Prefill.header, Admit.header) + 8 * (context + extra);
}

/// The prefill step's record (little-endian): the opcode, the chunk rows (u32), the flags (u8: bit 0, Engram on;
/// bit 1, DSpark taps absorbed; bit 2, CED bounded replay), the prompt's length (u32), then its ids (i64 each). The record is what the ranks'
/// digests agree on, so ranks that would chunk differently or disagree on Engram or the drafter (and so pair
/// different gathers) stop before the first gather.
pub const Prefill = struct {
    chunk: usize,
    engram: bool,
    drafts: bool,
    replay: bool,
    ids: []const u8, // the ids' little-endian bytes, 8 a token

    pub const header: usize = 10;

    pub fn encode(buf: []u8, chunk: usize, engram: bool, drafts: bool, replay: bool, ids: []const i64) []const u8 {
        buf[0] = @intFromEnum(Op.prefill);
        std.mem.writeInt(u32, buf[1..5], @intCast(chunk), .little);
        buf[5] = @as(u8, @intFromBool(engram)) | @as(u8, @intFromBool(drafts)) << 1 | @as(u8, @intFromBool(replay)) << 2;
        std.mem.writeInt(u32, buf[6..10], @intCast(ids.len), .little);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, buf[header + 8 * i ..][0..8], id, .little);
        return buf[0 .. header + 8 * ids.len];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Prefill {
        if (op.len < header or op[0] != @intFromEnum(Op.prefill) or op[5] > 7) return error.BadRecord;
        const n = std.mem.readInt(u32, op[6..10], .little);
        if (op.len != header + 8 * @as(usize, n)) return error.BadRecord;
        return .{ .chunk = std.mem.readInt(u32, op[1..5], .little), .engram = op[5] & 1 != 0, .drafts = op[5] & 2 != 0, .replay = op[5] & 4 != 0, .ids = op[header..] };
    }

    pub fn len(self: Prefill) usize {
        return self.ids.len / 8;
    }

    /// The ids into `out` (at least `len()` long); returns them.
    pub fn idsInto(self: Prefill, out: []i64) []i64 {
        const n = self.len();
        for (out[0..n], 0..) |*id, i| id.* = std.mem.readInt(i64, self.ids[8 * i ..][0..8], .little);
        return out[0..n];
    }

    /// The prefill step every rank runs for the record `op`: its ids (into `ids_buf`) through `seq.prefill` in its
    /// chunks, the gathers through `gather`.
    pub fn run(op: []const u8, seq: *Sequence, gather: forward.Gather, ids_buf: []i64) !void {
        const p = try parse(op);
        if (p.len() > ids_buf.len) return error.PromptTooLong;
        try seq.prefill(gather, p.idsInto(ids_buf), p.chunk, p.replay);
    }
};

fn encodeRound(buf: []u8, token: i64, pos: usize) []const u8 {
    buf[0] = @intFromEnum(Op.round);
    std.mem.writeInt(i64, buf[1..9], token, .little);
    std.mem.writeInt(u64, buf[9..17], @intCast(pos), .little);
    return buf[0..17];
}

/// The verify step's record: the opcode, the window's rows (u8), its start (u64: the cache length before it), then
/// its ids (i64 each).
pub const Verify = struct {
    start: usize,
    ids: []const u8,

    pub const header: usize = 10;

    pub fn encode(buf: []u8, start: usize, ids: []const i64) []const u8 {
        buf[0] = @intFromEnum(Op.verify);
        buf[1] = @intCast(ids.len);
        std.mem.writeInt(u64, buf[2..10], @intCast(start), .little);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, buf[header + 8 * i ..][0..8], id, .little);
        return buf[0 .. header + 8 * ids.len];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Verify {
        if (op.len < header or op[0] != @intFromEnum(Op.verify) or op[1] < 1 or op[1] > max_rows) return error.BadRecord;
        if (op.len != header + 8 * @as(usize, op[1])) return error.BadRecord;
        return .{ .start = @intCast(std.mem.readInt(u64, op[2..10], .little)), .ids = op[header..] };
    }

    pub fn len(self: Verify) usize {
        return self.ids.len / 8;
    }

    pub fn idsInto(self: Verify, out: []i64) []i64 {
        const n = self.len();
        for (out[0..n], 0..) |*id, i| id.* = std.mem.readInt(i64, self.ids[8 * i ..][0..8], .little);
        return out[0..n];
    }
};

/// The keep step's record: the opcode, the verify's start (u64) and the rows kept from it (u32).
pub const Keep = struct {
    start: usize,
    kept: usize,

    pub const bytes: usize = 13;

    pub fn encode(buf: []u8, k: Keep) []const u8 {
        buf[0] = @intFromEnum(Op.keep);
        std.mem.writeInt(u64, buf[1..9], @intCast(k.start), .little);
        std.mem.writeInt(u32, buf[9..13], @intCast(k.kept), .little);
        return buf[0..bytes];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Keep {
        if (op.len != bytes or op[0] != @intFromEnum(Op.keep)) return error.BadRecord;
        return .{ .start = @intCast(std.mem.readInt(u64, op[1..9], .little)), .kept = std.mem.readInt(u32, op[9..13], .little) };
    }
};

/// The draft step's record: the opcode, the last verify's start (u64) and the rows of it to absorb (u32; 0 after a
/// prompt, whose chunks absorbed theirs), the pending token (i64) and its position q0 (u64), the drafts to make
/// (u32; 0: absorb only).
pub const Draft = struct {
    start: usize,
    absorb: usize,
    token: i64,
    q0: usize,
    depth: usize,

    pub const bytes: usize = 33;

    pub fn encode(buf: []u8, r: Draft) []const u8 {
        buf[0] = @intFromEnum(Op.draft);
        std.mem.writeInt(u64, buf[1..9], @intCast(r.start), .little);
        std.mem.writeInt(u32, buf[9..13], @intCast(r.absorb), .little);
        std.mem.writeInt(i64, buf[13..21], r.token, .little);
        std.mem.writeInt(u64, buf[21..29], @intCast(r.q0), .little);
        std.mem.writeInt(u32, buf[29..33], @intCast(r.depth), .little);
        return buf[0..bytes];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Draft {
        if (op.len != bytes or op[0] != @intFromEnum(Op.draft)) return error.BadRecord;
        return .{
            .start = @intCast(std.mem.readInt(u64, op[1..9], .little)),
            .absorb = std.mem.readInt(u32, op[9..13], .little),
            .token = std.mem.readInt(i64, op[13..21], .little),
            .q0 = @intCast(std.mem.readInt(u64, op[21..29], .little)),
            .depth = std.mem.readInt(u32, op[29..33], .little),
        };
    }
};

pub const Settings = struct {
    rank: u32,
    world: u32,
    master: [4]u8,
    port: u16,
    context: usize,
    bucket: usize,
    rope_dir: []const u8,
    cache_dir: []const u8,
    kernels_dir: []const u8,
    /// Prompt rows a chunk (TF_DS_PREFILL_CHUNK), 1 to prompt.max_chunk.
    chunk: usize = default_chunk,
    /// Load the DSpark drafter (the rank cache's `mtp.*` blocks) and serve drafted rounds.
    drafts: bool = false,
    /// Streams at once (the pool's slots), 1 to `max_slots`.
    slots: usize = max_slots,
    /// Positions the streams' pool holds (TF_DS_POOL): at least `context`; 0 means `context`. Each stream still
    /// reaches at most `context` positions.
    pool: usize = 0,
    /// CED bounded replay (TF_DS_REPLAY=1): a prompt's layers from n_layers / 2 run only its last `window` rows.
    replay: bool = false,
    /// The drafts a stream's window holds at most (TF_DS_DRAFTS), 1 to the checkpoint's DSpark block; 0: the block.
    /// A deeper window would need a deeper drafter block, whose kernels' row-count variants (the MoE gate's
    /// `_rowmm_parts` RB) the captured kernel sets do not hold.
    most_drafts: usize = 0,
    /// Rows a round takes at most (TF_DS_MAX_ROWS), 1 to `round_rows_most`; the same on every rank.
    rows: usize = rounds.rows_default,

    /// The pool's positions: `pool`, or the context when it is 0.
    pub fn poolPositions(self: Settings) usize {
        return if (self.pool == 0) self.context else self.pool;
    }
};

/// The drafts a window holds at most for a checkpoint block of `block` and a TF_DS_DRAFTS of `most` (0: unset).
pub fn draftsFor(block: usize, most: usize) usize {
    return if (most == 0) block else most;
}

/// engine.py prefill_steps' `replay`: with bounded replay, the first position the upper layers run (the prompt's
/// last `window` rows); null without.
pub fn replayFrom(replay: bool, prompt_len: usize, window: usize) ?usize {
    return if (replay) prompt_len -| window else null;
}

/// One sequence on caches of its own (the fixture tests' path): the prompt path (`pr`) and the decode layer loop (`fwd`) over the same caches
/// (`fwd`'s rings and compressed planes), the host ids Engram hashes, the drafter's cache and the taps, and the
/// position the next row takes. Every rank holding the same sequence runs the same calls with the same arguments,
/// so their gathers pair.
pub const Sequence = struct {
    fwd: *forward.Forward,
    pr: prompt.Prompt,
    engram: ?*engram_apply.Engram,
    hist: []i32, // the sequence's ids from position 0 (Engram's hashes)
    logits_h: []f32,
    /// The last row's logits [1, V] f32 after `prefill` or `round`; every row's [n, V] after `verify` (the prompt
    /// path's or the decode loop's buffer).
    logits: Ptr = 0,
    pos: usize = 0,
    stream: cuda.Stream,
    // DSpark (null: no drafter)
    spark: ?*dspark.Drafter = null,
    dc: ?dspark.Cache = null,
    /// The taps of the last prompt chunk or verify window: bf16 [rows, taps x d], a column block a tap layer.
    taps: cuda.DeviceBuffer,
    tap_rows: usize = 0, // rows the probe taps (the window running)
    verify_start: usize = 0, // the last verify's start and rows (the taps `taps` holds)
    verify_rows: usize = 0,
    /// The last draft's tokens; the first `held` of them are the stream's held drafts.
    drafts: dspark.Drafts = undefined,
    held: usize = 0,
    prev_probe: ?forward.Probe = null,

    /// `rows`: the widest chunk `prefill` takes (at most prompt.max_chunk). `engram`, when given, holds at least
    /// `rows` rows a call and becomes `fwd`'s hook. `spark`, when given, drafts for this sequence: every tap layer
    /// is one of `fwd`'s layers past the first (a tap is a layer's input).
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, fwd: *forward.Forward, kernels: *const exl3.Kernels, engram: ?*engram_apply.Engram, rows: usize, stream: cuda.Stream, spark: ?*dspark.Drafter) !Sequence {
        var self: Sequence = .{ .fwd = fwd, .pr = undefined, .engram = engram, .hist = undefined, .logits_h = undefined, .stream = stream, .taps = .{ .d = d, .ptr = 0, .len = 0 } };
        if (spark != null) for (fwd.cfg.dspark_taps.slice()) |t| {
            if (t < 1 or t >= @as(i64, @intCast(fwd.w.layers.len))) return error.BadTapLayers;
        };
        self.pr = try prompt.Prompt.init(gpa, d, fwd, kernels, rows);
        errdefer self.pr.deinit();
        self.hist = try gpa.alloc(i32, fwd.cap);
        errdefer gpa.free(self.hist);
        self.logits_h = try gpa.alloc(f32, max_rows * fwd.cfg.vocab);
        errdefer gpa.free(self.logits_h);
        if (spark) |sp| {
            self.taps = try cuda.DeviceBuffer.alloc(d, @max(rows, max_rows) * self.tapCols() * 2);
            errdefer self.taps.free();
            self.dc = try sp.newCache();
            self.spark = sp;
        }
        if (engram) |g| fwd.engram = g.hook();
        return self;
    }

    pub fn deinit(self: *Sequence, gpa: std.mem.Allocator) void {
        if (self.dc) |*c| c.deinit(gpa);
        if (self.taps.len > 0) self.taps.free();
        gpa.free(self.logits_h);
        gpa.free(self.hist);
        self.pr.deinit();
    }

    /// Elements a row of `taps` holds: every tap layer's d.
    fn tapCols(self: *const Sequence) usize {
        return self.fwd.cfg.dspark_taps.len * self.fwd.cfg.dim;
    }

    /// The positions a sequence may reach (the caches' capacity less the window's extra rows).
    pub fn limit(self: *const Sequence) usize {
        return self.fwd.cap - extra;
    }

    /// The RoPE table the drafter's stages read (their layers' kind: window attention).
    fn draftRope(self: *const Sequence) attention.Rope {
        return if (self.fwd.cfg.ratio(self.fwd.cfg.n_layers) > 0) self.fwd.rope_c else self.fwd.rope_w;
    }

    /// The prompt's ids into the caches from position 0 (engine.py prefill_steps from start 0): chunks of `chunk`
    /// rows at positions 0, chunk, 2 chunk, ..., the last one the remainder; the last row's logits in `logits`.
    /// With a drafter, each chunk's taps are absorbed after it (`drafter.absorb(dc, sc, torch.cat(taps, -1),
    /// m.taps_start)`): under bounded replay (`replay`), the rows that reached the upper layers, none from an
    /// encoder-only chunk. Runs every chunk once started: the other ranks cannot see a cancel, so stopping early on
    /// one rank would leave the others in an unpaired gather.
    pub fn prefill(self: *Sequence, gather: forward.Gather, ids: []const i64, chunk: usize, replay: bool) !void {
        if (chunk < 1 or chunk > self.pr.rows) return error.BadChunk;
        if (ids.len == 0 or ids.len + 1 > self.limit()) return error.PromptTooLong;
        self.pos = 0;
        self.logits = 0;
        self.held = 0;
        self.verify_rows = 0;
        for (self.hist[0..ids.len], ids) |*hv, id| hv.* = @intCast(id);
        if (self.engram) |g| g.history = self.hist[0..ids.len];
        if (self.dc) |*c| c.absorbed = 0;
        if (self.engram) |g| {
            try g.resetAhead();
            try g.prefetch(self.hist[0..ids.len], 0, @min(ids.len, chunk));
        }
        var s: usize = 0;
        while (s < ids.len) {
            const e = @min(ids.len, s + chunk);
            const last = e == ids.len;
            // the next chunk's Engram rows read while this one runs
            if (self.engram) |g| if (!last) try g.prefetch(self.hist[0..ids.len], e, @min(ids.len, e + chunk) - e);
            // A chunk of max_rows or fewer is a decode window (it runs the head): Prompt.chunk sends it through
            // `Forward.rows` only when it asks for logits; with taps it goes there through `rowsTapped`.
            const from = replayFrom(replay, ids.len, self.fwd.cfg.window);
            var top: [2]usize = .{ s, e - s };
            const out = if (self.spark != null and e - s <= max_rows and (from orelse 0) <= s)
                try self.rowsTapped(gather, ids[s..e], s, false)
            else blk: {
                const o = try self.pr.chunk(gather, ids[s..e], s, .{
                    .replay = from,
                    .logits = last or e - s <= max_rows,
                    .taps = if (self.spark != null) .{ .buf = self.taps.ptr, .stride = self.tapCols() } else null,
                }, self.stream);
                top = .{ self.pr.top_start, self.pr.top_rows };
                break :blk o;
            };
            if (self.spark) |sp| if (top[1] > 0) try sp.absorb(&self.dc.?, self.taps.ptr, top[0], top[1], self.draftRope(), self.stream);
            if (last) self.logits = out orelse return error.BadChunk;
            s = e;
        }
        self.pos = ids.len;
    }

    /// One token at the cache's end through the decode layer loop: its logits (the next token's) in `logits`.
    pub fn round(self: *Sequence, gather: forward.Gather, token: i64) !void {
        if (self.pos + 1 > self.limit()) return error.PromptTooLong;
        self.hist[self.pos] = @intCast(token);
        if (self.engram) |g| g.history = self.hist[0 .. self.pos + 1];
        const ids = [1]i64{token};
        self.logits = try self.fwd.rows(gather, &ids, self.pos, 1, false, self.stream);
        self.pos += 1;
    }

    /// A window at the cache's end (the pending token, then drafts) through the decode layer loop (graph.py
    /// StaticDecoder with taps): every row's logits [n, V] in `logits`, the rows' taps in `taps` with a drafter.
    pub fn verify(self: *Sequence, gather: forward.Gather, ids: []const i64) !void {
        const n = ids.len;
        if (n < 1 or n > max_rows) return error.WindowTooWide;
        if (self.pos + n > self.limit()) return error.PromptTooLong;
        for (self.hist[self.pos..][0..n], ids) |*hv, id| hv.* = @intCast(id);
        if (self.engram) |g| g.history = self.hist[0 .. self.pos + n];
        self.logits = try self.rowsTapped(gather, ids, self.pos, true);
        self.verify_start = self.pos;
        self.verify_rows = n;
        self.pos += n;
    }

    /// The last verify's first `kept` rows stay (engine.py `sc.length = P + a + 1`): later rows overwrite the
    /// rejected rows' cache writes before any row reads them.
    pub fn keep(self: *Sequence, start: usize, kept: usize) !void {
        if (start != self.verify_start or kept < 1 or kept > self.verify_rows or self.pos != start + self.verify_rows) return error.PositionMismatch;
        self.pos = start + kept;
    }

    /// The drafter absorbs the last verify's first `absorb` rows' taps (at `start`, the kept rows: engine.py
    /// `d.absorb(dc, sc, taps[:a + 1], P)`), then, with `depth` > 0, drafts after `token` at `q0` (the cache's end,
    /// every position before it absorbed); the drafts in `drafts`, the first `depth` held.
    pub fn draft(self: *Sequence, start: usize, absorb: usize, token: i64, q0: usize, depth: usize) !void {
        const sp = self.spark orelse return error.NoDraftHead;
        const c = &self.dc.?;
        self.held = 0;
        if (absorb > 0) {
            if (start != self.verify_start or absorb > self.verify_rows or self.pos != start + absorb) return error.PositionMismatch;
            try sp.absorb(c, self.taps.ptr, start, absorb, self.draftRope(), self.stream);
        }
        if (depth == 0) return;
        if (q0 != self.pos or c.absorbed != q0 or token < 0) return error.PositionMismatch;
        self.drafts = try sp.draft(c, @intCast(token), q0, self.fwd.w.embed.ptr(), self.draftRope(), self.stream);
        self.held = @min(depth, self.drafts.len);
    }

    /// `Forward.rows` with the window's taps written into `taps` (when a drafter is attached): the probe after each
    /// sublayer taps the streams entering a tap layer (after the layer before's MoE, or after its own Engram gate).
    fn rowsTapped(self: *Sequence, gather: forward.Gather, ids: []const i64, start: usize, all_logits: bool) !Ptr {
        if (self.spark == null) return self.fwd.rows(gather, ids, start, ids.len, all_logits, self.stream);
        self.prev_probe = self.fwd.probe;
        self.tap_rows = ids.len;
        self.fwd.probe = .{ .ctx = self, .call = tapProbe };
        defer self.fwd.probe = self.prev_probe;
        return self.fwd.rows(gather, ids, start, ids.len, all_logits, self.stream);
    }

    fn tapProbe(ctx: *anyopaque, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) anyerror!void {
        const self: *Sequence = @ptrCast(@alignCast(ctx));
        if (self.prev_probe) |p| try p.call(p.ctx, name, ptr, bytes, stream);
        const fwd = self.fwd;
        var gated: u64 = 0;
        if (fwd.engram != null) for (fwd.w.layers, 0..) |*l, i| {
            if (l.engram_wkv != null and i < 64) gated |= @as(u64, 1) << @intCast(i);
        };
        const li = tapLayer(name, fwd.w.layers.len, gated) orelse return;
        const cfg = self.fwd.cfg;
        const col = cfg.dspark_taps.indexOf(@intCast(li)) orelse return;
        try self.fwd.hc.tap(stream, ptr, self.taps.ptr + col * cfg.dim * 2, @intCast(self.tapCols()), @intCast(self.tap_rows));
    }

    /// The layer of `layers` whose input the probe `name` shows: "L<i>.h_moe" is layer i + 1's (unless that
    /// layer's Engram gate runs first: bit i + 1 of `gated`), "L<i>.h_engram" is layer i's; null for every other
    /// probe.
    fn tapLayer(name: []const u8, layers: usize, gated: u64) ?usize {
        if (name.len < 2 or name[0] != 'L') return null;
        const dot = std.mem.indexOfScalar(u8, name, '.') orelse return null;
        const i = std.fmt.parseInt(usize, name[1..dot], 10) catch return null;
        const what = name[dot + 1 ..];
        if (std.mem.eql(u8, what, "h_engram")) return i;
        if (!std.mem.eql(u8, what, "h_moe") or i + 1 >= layers) return null;
        if (i + 1 < 64 and gated & (@as(u64, 1) << @intCast(i + 1)) != 0) return null;
        return i + 1;
    }

    /// The last logits' argmax (the first index of the maximum, torch's rule), read on the host.
    pub fn argmax(self: *Sequence) !u32 {
        var one: [1]u32 = undefined;
        try self.argmaxRows(&one);
        return one[0];
    }

    /// Each of the last logits' first `out.len` rows' argmax (torch's rule: the first index of the maximum).
    pub fn argmaxRows(self: *Sequence, out: []u32) !void {
        const v = self.fwd.cfg.vocab;
        if (out.len > max_rows) return error.WindowTooWide;
        try self.stream.synchronize();
        const view: cuda.DeviceBuffer = .{ .d = self.fwd.d, .ptr = self.logits, .len = out.len * v * 4 };
        try view.download(0, std.mem.sliceAsBytes(self.logits_h[0 .. out.len * v]));
        for (out, 0..) |*o, r| {
            const row = self.logits_h[r * v ..][0..v];
            var best: usize = 0;
            for (row, 0..) |x, i| if (x > row[best]) {
                best = i;
            };
            o.* = @intCast(best);
        }
    }
};

/// RoPE rows the tables hold for `context` positions (ops.py RopeTables.rows of the engine's rope_cap).
pub fn ropeRows(context: usize) usize {
    const need = context + extra;
    return @max(rope_align, (need + rope_align - 1) / rope_align * rope_align);
}

// -- concurrent streams over one pool (multi.py MultiDecoder) ------------------------------------------------------

/// Slots the served engine holds: a round's 16 rows hold four streams' windows of up to four rows (the fork's
/// DEPTH_BY "5,5,3,3" at three and four streams).
pub const max_slots: usize = kept_prompts.max_slots;
/// Extents start and end on multiples of this many positions (multi.py ALIGN).
pub const extent_align: usize = kept_prompts.extent_align;
/// Positions an extent holds past its stream's prompt and reply (multi.py `_need`: a round's rows, 8, and the
/// scratch row's 4).
pub const extent_tail: usize = max_rows + 8 + 4;
/// multi.py `Extents` (cuda_keep.zig, beside the kept prompts' book that shares the window).
pub const Extents = kept_prompts.Extents;

/// The admission step's record: the opcode, the slot (u8), the flags (u8: bit 0 Engram on, bit 1 DSpark taps
/// absorbed, bit 2 CED bounded replay, bit 3 the prompt continues a kept prompt, bit 4 that continuation checked
/// against a prefill from 0 (TF_DS_KEEP_CHECK), bit 5 kept prompts on, bit 6 the prompt filled in parts), the chunk
/// rows (u32), the extent's base and size (u64 each), the kept prompt's id (u32) and boundary (u64), the window
/// book's fingerprint before the step (u64), the prompt's length (u32), its ids (i64 each). Every rank places the
/// extent the same way (cuda_keep.zig `Book.place`: the record's base is rank 0's) and fills it through the prompt
/// path in the record's chunks, from position 0 or, continuing a kept prompt, from its boundary: all of them in this
/// step, or with `parts` none, the chunks then run by `Fill` steps (rounds of the other streams may run between).
pub const Admit = struct {
    slot: usize,
    engram: bool,
    drafts: bool,
    replay: bool = false,
    chunk: usize,
    base: usize,
    size: usize,
    reuse: ?kept_prompts.Reuse = null,
    check: bool = false,
    keep: bool = false,
    parts: bool = false,
    book: u64 = 0,
    ids: []const u8,

    pub const header: usize = 47;

    pub fn encode(buf: []u8, a: Admit, ids: []const i64) []const u8 {
        buf[0] = @intFromEnum(Op.admit);
        buf[1] = @intCast(a.slot);
        buf[2] = @as(u8, @intFromBool(a.engram)) | @as(u8, @intFromBool(a.drafts)) << 1 | @as(u8, @intFromBool(a.replay)) << 2 |
            @as(u8, @intFromBool(a.reuse != null)) << 3 | @as(u8, @intFromBool(a.check)) << 4 | @as(u8, @intFromBool(a.keep)) << 5 |
            @as(u8, @intFromBool(a.parts)) << 6;
        std.mem.writeInt(u32, buf[3..7], @intCast(a.chunk), .little);
        std.mem.writeInt(u64, buf[7..15], @intCast(a.base), .little);
        std.mem.writeInt(u64, buf[15..23], @intCast(a.size), .little);
        std.mem.writeInt(u32, buf[23..27], if (a.reuse) |r| r.eid else 0, .little);
        std.mem.writeInt(u64, buf[27..35], if (a.reuse) |r| @intCast(r.cut) else 0, .little);
        std.mem.writeInt(u64, buf[35..43], a.book, .little);
        std.mem.writeInt(u32, buf[43..47], @intCast(ids.len), .little);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, buf[header + 8 * i ..][0..8], id, .little);
        return buf[0 .. header + 8 * ids.len];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Admit {
        if (op.len < header or op[0] != @intFromEnum(Op.admit) or op[2] > 127) return error.BadRecord;
        const n = std.mem.readInt(u32, op[43..47], .little);
        if (op.len != header + 8 * @as(usize, n)) return error.BadRecord;
        return .{
            .slot = op[1],
            .engram = op[2] & 1 != 0,
            .drafts = op[2] & 2 != 0,
            .replay = op[2] & 4 != 0,
            .reuse = if (op[2] & 8 != 0) .{ .eid = std.mem.readInt(u32, op[23..27], .little), .cut = @intCast(std.mem.readInt(u64, op[27..35], .little)) } else null,
            .check = op[2] & 16 != 0,
            .keep = op[2] & 32 != 0,
            .parts = op[2] & 64 != 0,
            .chunk = std.mem.readInt(u32, op[3..7], .little),
            .base = @intCast(std.mem.readInt(u64, op[7..15], .little)),
            .size = @intCast(std.mem.readInt(u64, op[15..23], .little)),
            .book = std.mem.readInt(u64, op[35..43], .little),
            .ids = op[header..],
        };
    }

    pub fn len(self: Admit) usize {
        return self.ids.len / 8;
    }

    pub fn idsInto(self: Admit, out: []i64) []i64 {
        const n = self.len();
        for (out[0..n], 0..) |*id, i| id.* = std.mem.readInt(i64, self.ids[8 * i ..][0..8], .little);
        return out[0..n];
    }
};

/// The free step's record: the opcode, the slot (u8), the kept prompts the stream's prompt covers (u8, then their
/// ids, u32 each: rank 0's `Streams.covered`), the window book's fingerprint before the step (u64). With kept
/// prompts on, the stream's prompt stays as a kept prompt (`Streams.finish`).
pub const Free = struct {
    slot: usize,
    book: u64,
    drops: [kept_prompts.max_kept]u32 = undefined,
    n: usize = 0,

    pub const header: usize = 11;

    pub fn encode(buf: []u8, slot: usize, book: u64, drops: []const u32) []const u8 {
        buf[0] = @intFromEnum(Op.free);
        buf[1] = @intCast(slot);
        buf[2] = @intCast(drops.len);
        std.mem.writeInt(u64, buf[3..11], book, .little);
        for (drops, 0..) |d, i| std.mem.writeInt(u32, buf[header + 4 * i ..][0..4], d, .little);
        return buf[0 .. header + 4 * drops.len];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Free {
        if (op.len < header or op[0] != @intFromEnum(Op.free) or op[2] > kept_prompts.max_kept) return error.BadRecord;
        if (op.len != header + 4 * @as(usize, op[2])) return error.BadRecord;
        var f: Free = .{ .slot = op[1], .book = std.mem.readInt(u64, op[3..11], .little), .n = op[2] };
        for (f.drops[0..f.n], 0..) |*d, i| d.* = std.mem.readInt(u32, op[header + 4 * i ..][0..4], .little);
        return f;
    }
};

/// The fill step's record: the opcode, the slot (u8), the chunks to run (u32, at least 1) and the position the
/// slot's prompt pass stands at before them (u64). Every rank runs the next `chunks` chunks of the pass an
/// admission in parts began (`Streams.fillPart`), the pass's last chunk drawing its first token's logits.
pub const Fill = struct {
    slot: usize,
    chunks: u32,
    at: usize,

    pub const bytes: usize = 14;

    pub fn encode(buf: []u8, f: Fill) []const u8 {
        buf[0] = @intFromEnum(Op.fill);
        buf[1] = @intCast(f.slot);
        std.mem.writeInt(u32, buf[2..6], f.chunks, .little);
        std.mem.writeInt(u64, buf[6..14], @intCast(f.at), .little);
        return buf[0..bytes];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Fill {
        if (op.len != bytes or op[0] != @intFromEnum(Op.fill)) return error.BadRecord;
        const chunks = std.mem.readInt(u32, op[2..6], .little);
        if (chunks < 1) return error.BadRecord;
        return .{ .slot = op[1], .chunks = chunks, .at = @intCast(std.mem.readInt(u64, op[6..14], .little)) };
    }
};

/// One stream's window in a round record: its slot, its extent, the first row's position (in the extent) and its
/// ids (the pending token, then the drafts).
pub const RoundWindow = struct { slot: usize, base: usize, size: usize, position: usize, ids: []const i64 };

/// The round step's record: the opcode, the windows (u8), per window its slot (u8), rows (u8), position, base and
/// size (u64 each), then every window's ids in window order (i64 each). One forward over all of them.
pub const Round = struct {
    windows: [max_slots]RoundWindow,
    count: usize,
    ids: [round_rows_most]i64,

    const per: usize = 26;

    pub fn encode(buf: []u8, windows: []const RoundWindow) []const u8 {
        buf[0] = @intFromEnum(Op.rounds);
        buf[1] = @intCast(windows.len);
        var at: usize = 2;
        for (windows) |w| {
            buf[at] = @intCast(w.slot);
            buf[at + 1] = @intCast(w.ids.len);
            std.mem.writeInt(u64, buf[at + 2 ..][0..8], @intCast(w.position), .little);
            std.mem.writeInt(u64, buf[at + 10 ..][0..8], @intCast(w.base), .little);
            std.mem.writeInt(u64, buf[at + 18 ..][0..8], @intCast(w.size), .little);
            at += per;
        }
        for (windows) |w| for (w.ids) |id| {
            std.mem.writeInt(i64, buf[at..][0..8], id, .little);
            at += 8;
        };
        return buf[0..at];
    }

    /// The record's windows (their ids in `ids`); every window at least one row, at most `max_slots` windows and
    /// `round_rows_most` rows.
    pub fn parse(op: []const u8, out: *Round) error{BadRecord}!void {
        if (op.len < 2 or op[0] != @intFromEnum(Op.rounds) or op[1] < 1 or op[1] > max_slots) return error.BadRecord;
        const count: usize = op[1];
        if (op.len < 2 + per * count) return error.BadRecord;
        var rows: usize = 0;
        for (0..count) |i| {
            const r: usize = op[2 + per * i + 1];
            if (r < 1) return error.BadRecord;
            rows += r;
        }
        if (rows > round_rows_most or op.len != 2 + per * count + 8 * rows) return error.BadRecord;
        out.count = count;
        var at: usize = 2 + per * count;
        var r0: usize = 0;
        for (out.windows[0..count], 0..) |*w, i| {
            const h = op[2 + per * i ..][0..per];
            const n: usize = h[1];
            for (out.ids[r0..][0..n]) |*id| {
                id.* = std.mem.readInt(i64, op[at..][0..8], .little);
                at += 8;
            }
            w.* = .{
                .slot = h[0],
                .position = @intCast(std.mem.readInt(u64, h[2..10], .little)),
                .base = @intCast(std.mem.readInt(u64, h[10..18], .little)),
                .size = @intCast(std.mem.readInt(u64, h[18..26], .little)),
                .ids = out.ids[r0..][0..n],
            };
            r0 += n;
        }
    }
};

/// One stream's draft request in a drafts record: absorb the last round's first `absorb` rows' taps (at `start`:
/// its kept rows, which sets the stream's length to start + absorb), then, with `depth` > 0, draft after `token`
/// at q0 (the stream's length).
pub const DraftAsk = struct { slot: usize, start: usize, absorb: usize, token: i64, q0: usize, depth: usize };

/// The drafts step's record: the opcode, the requests (u8), each its slot (u8), start (u64), absorb (u32), token
/// (i64), q0 (u64) and depth (u32). Every drafting stream's drafts come from batched passes over the DraftPool.
pub const DraftsRec = struct {
    asks: [max_slots]DraftAsk,
    count: usize,

    const per: usize = 33;

    pub fn encode(buf: []u8, asks: []const DraftAsk) []const u8 {
        buf[0] = @intFromEnum(Op.drafts);
        buf[1] = @intCast(asks.len);
        for (asks, 0..) |a, i| {
            const b = buf[2 + per * i ..][0..per];
            b[0] = @intCast(a.slot);
            std.mem.writeInt(u64, b[1..9], @intCast(a.start), .little);
            std.mem.writeInt(u32, b[9..13], @intCast(a.absorb), .little);
            std.mem.writeInt(i64, b[13..21], a.token, .little);
            std.mem.writeInt(u64, b[21..29], @intCast(a.q0), .little);
            std.mem.writeInt(u32, b[29..33], @intCast(a.depth), .little);
        }
        return buf[0 .. 2 + per * asks.len];
    }

    pub fn parse(op: []const u8, out: *DraftsRec) error{BadRecord}!void {
        if (op.len < 2 or op[0] != @intFromEnum(Op.drafts) or op[1] < 1 or op[1] > max_slots) return error.BadRecord;
        const count: usize = op[1];
        if (op.len != 2 + per * count) return error.BadRecord;
        out.count = count;
        for (out.asks[0..count], 0..) |*a, i| {
            const b = op[2 + per * i ..][0..per];
            a.* = .{
                .slot = b[0],
                .start = @intCast(std.mem.readInt(u64, b[1..9], .little)),
                .absorb = std.mem.readInt(u32, b[9..13], .little),
                .token = std.mem.readInt(i64, b[13..21], .little),
                .q0 = @intCast(std.mem.readInt(u64, b[21..29], .little)),
                .depth = std.mem.readInt(u32, b[29..33], .little),
            };
        }
    }
};

/// One stream's keep: the last round's window at `start` keeps its first `kept` rows.
pub const KeepAsk = struct { slot: usize, start: usize, kept: usize };

/// The keeps step's record: the opcode, the streams (u8), each its slot (u8), start (u64) and kept rows (u32).
pub const Keeps = struct {
    asks: [max_slots]KeepAsk,
    count: usize,

    const per: usize = 13;

    pub fn encode(buf: []u8, asks: []const KeepAsk) []const u8 {
        buf[0] = @intFromEnum(Op.keeps);
        buf[1] = @intCast(asks.len);
        for (asks, 0..) |a, i| {
            const b = buf[2 + per * i ..][0..per];
            b[0] = @intCast(a.slot);
            std.mem.writeInt(u64, b[1..9], @intCast(a.start), .little);
            std.mem.writeInt(u32, b[9..13], @intCast(a.kept), .little);
        }
        return buf[0 .. 2 + per * asks.len];
    }

    pub fn parse(op: []const u8, out: *Keeps) error{BadRecord}!void {
        if (op.len < 2 or op[0] != @intFromEnum(Op.keeps) or op[1] < 1 or op[1] > max_slots) return error.BadRecord;
        const count: usize = op[1];
        if (op.len != 2 + per * count) return error.BadRecord;
        out.count = count;
        for (out.asks[0..count], 0..) |*a, i| {
            const b = op[2 + per * i ..][0..per];
            a.* = .{ .slot = b[0], .start = @intCast(std.mem.readInt(u64, b[1..9], .little)), .kept = std.mem.readInt(u32, b[9..13], .little) };
        }
    }
};

/// The measure step's record: the opcode, the repetitions (u8), the pool's slots (u8), the flags (u8: bit 0 Engram,
/// bit 1 the drafter, bits 2 .. 5 a window's drafts at most), the window's positions (u64), a stream's most (u64,
/// `Streams.most`) and a position to also time rounds of every row count at (u64, 0: none; TF_DS_MEASURE_AT): ranks
/// whose pools, per-stream contexts or drafts differ stop at load.
pub const Measure = struct {
    reps: u8,
    slots: usize,
    engram: bool,
    drafts: bool,
    total: usize,
    most: usize,
    /// a window's drafts at most (0 .. most_drafts_max)
    width: usize = 0,
    at: usize = 0,
    /// The rows a round takes at most (`Streams.rows`): ranks timing different row counts would pair different
    /// collectives.
    rows: usize = rounds.rows_default,

    pub const bytes: usize = 29;

    pub fn encode(buf: []u8, m: Measure) []const u8 {
        buf[0] = @intFromEnum(Op.measure);
        buf[1] = m.reps;
        buf[2] = @intCast(m.slots);
        buf[3] = @as(u8, @intFromBool(m.engram)) | @as(u8, @intFromBool(m.drafts)) << 1 | @as(u8, @intCast(m.width)) << 2;
        std.mem.writeInt(u64, buf[4..12], @intCast(m.total), .little);
        std.mem.writeInt(u64, buf[12..20], @intCast(m.most), .little);
        std.mem.writeInt(u64, buf[20..28], @intCast(m.at), .little);
        buf[28] = @intCast(m.rows);
        return buf[0..bytes];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Measure {
        if (op.len != bytes or op[0] != @intFromEnum(Op.measure) or op[3] >> 2 > most_drafts_max) return error.BadRecord;
        return .{ .reps = op[1], .slots = op[2], .engram = op[3] & 1 != 0, .drafts = op[3] & 2 != 0, .total = @intCast(std.mem.readInt(u64, op[4..12], .little)), .most = @intCast(std.mem.readInt(u64, op[12..20], .little)), .width = op[3] >> 2, .at = @intCast(std.mem.readInt(u64, op[20..28], .little)), .rows = op[28] };
    }
};

/// A stream's state in its slot: its extent of the window, the position its next row takes (relative to the
/// extent), its host ids from position 0 (Engram's hashes; the prompt's first `prompt_len`), its prompt's ring
/// snapshots at its chunk boundaries (`marks`: kept prompts), the last round's window (its start, rows and first row
/// among the round's), the positions its drafter rings absorbed, its last drafts (the first `held` of them held).
/// A prompt pass in parts: the position it stands at (a chunk boundary, or the prompt's end once done) and the
/// admission's settings each part runs with.
pub const Pass = struct {
    at: usize,
    cut: usize,
    chunk: usize,
    replay: bool,
    check: bool,
    marks: ?kept_prompts.Points,
    t0: i128,
};

pub const Lane = struct {
    busy: bool = false,
    base: usize = 0,
    size: usize = 0,
    pos: usize = 0,
    hist: []i32 = &.{},
    prompt_len: usize = 0,
    marks: kept_prompts.Marks = .{},
    verify_start: usize = 0,
    verify_rows: usize = 0,
    row0: usize = 0,
    absorbed: usize = 0,
    drafts: dspark.Drafts = undefined,
    held: usize = 0,
    pass: ?Pass = null, // the prompt pass under way in parts (an admission with `parts`), until its last chunk

    /// The positions a window may reach: the extent less its tail (the round's rows, 8, and the scratch row's).
    pub fn limit(self: *const Lane) usize {
        return self.size -| (8 + 4);
    }
};

/// model.py `pool_view`'s single-stream paths and rounds.py's concurrent round over one pool (multi.py
/// MultiDecoder without its scheduler): `slots` streams share one window of `cap` positions by extents of at most
/// `most` positions each (a stream's context and extra rows, aligned down to `extent_align`). An
/// admission fills a stream's slot and extent through the prompt path (cuda_prompt.zig) on a view of the pool
/// (`view`: a `Forward` whose rings, planes and capacity are the slot's and the extent's), its chunks' taps into its
/// slot of the DraftPool; a round runs every window through one `RoundRunner.forward`; drafts run batched over the
/// DraftPool. Every rank holding the same streams runs the same calls with the same arguments, so their gathers
/// pair. Holds pointers into itself: `init` in place, never moved.
pub const Streams = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    cfg: *const Config,
    base: *forward.Forward,
    view: forward.Forward,
    view_rings: []cuda.DeviceBuffer,
    view_planes: []?forward.Planes,
    pool: rounds.Pool,
    runner: rounds.RoundRunner,
    pr: prompt.Prompt,
    extents: Extents,
    /// The positions an extent holds at most: `span` (a stream's context and extra rows) aligned down.
    most: usize,
    engram: ?*engram_apply.Engram,
    spark: ?*dspark.Drafter,
    dpool: ?dspark.DraftPool,
    dc_rings: []cuda.DeviceBuffer, // the admitted slot's drafter rings (views of the DraftPool's planes)
    taps: cuda.DeviceBuffer, // a prompt chunk's taps, bf16 [rows, taps x d]
    kept_taps: cuda.DeviceBuffer, // a drafts step's kept rows' taps, every stream's in request order
    tap_rows: usize = 0,
    prev_probe: ?forward.Probe = null,
    lanes: []Lane,
    logits_h: []f32,
    /// The last admission's last row's logits [1, V], or the last round's [R, V] (window order).
    logits: Ptr = 0,
    /// u32 [rows]: the device argmax's tokens (`argmaxRows` with the runner's tail on).
    amax: cuda.DeviceBuffer,
    /// Rows a round takes at most (the runner's, `Settings.rows`).
    rows: usize,
    stream: cuda.Stream,
    /// Kept prompts (cuda_keep.zig): the book of them, what its operations let go (`settle` frees it), and the ring
    /// snapshots (`snap_count` of `snap_bytes` each: the window rings of the first `snap_layers` layers, then every
    /// compressor's inputs, of one slot), each held by the streams and kept prompts counted in `snap_refs`.
    book: kept_prompts.Book,
    released: kept_prompts.Released = .{},
    snaps: cuda.DeviceBuffer,
    snap_bytes: usize = 0,
    snap_layers: usize = 0,
    snap_refs: []u8 = &.{},
    /// The bounded replay the snapshots were laid out for: only admissions with the same take or continue kept
    /// prompts.
    keep_replay: bool = false,
    /// Admissions that continued a kept prompt, by placement (`How`), and the checked ones that differed.
    keep_stats: [4]u64 = @splat(0),
    keep_differed: u64 = 0,

    /// `base`: the decode loop whose weights, kernels, scratch and RoPE tables the view and the round share (its
    /// own caches go unused). `rows`: the widest prompt chunk. `cap`: the window's positions (the pool and the
    /// window's extra rows). `span`: a stream's positions at most (its context and the extra rows), `cap` at most;
    /// the prompt path's buffers, the round runner's buckets and the host ids are sized by it. `round_rows`: the rows
    /// a round takes at most (1 .. round_rows_most).
    pub fn init(self: *Streams, gpa: std.mem.Allocator, d: *const cuda.Driver, base: *forward.Forward, kernels: *const exl3.Kernels, engram: ?*engram_apply.Engram, rows: usize, stream: cuda.Stream, spark: ?*dspark.Drafter, slots: usize, cap: usize, span: usize, round_rows: usize) !void {
        const cfg = base.cfg;
        const L = base.w.layers.len;
        if (slots < 1 or slots > max_slots) return error.BadSlots;
        if (span > cap or span < extent_align) return error.BadPool;
        if (round_rows < 1 or round_rows > round_rows_most) return error.WindowTooWide;
        if (spark != null) for (cfg.dspark_taps.slice()) |t| {
            if (t < 1 or t >= @as(i64, @intCast(L))) return error.BadTapLayers;
        };
        self.* = .{
            .gpa = gpa, .d = d, .cfg = cfg, .base = base, .view = base.*, .view_rings = &.{}, .view_planes = &.{}, .pool = undefined,
            .runner = undefined, .pr = undefined, .extents = Extents.init(cap), .most = span / extent_align * extent_align, .engram = engram, .spark = spark, .dpool = null,
            .dc_rings = &.{}, .taps = .{ .d = d, .ptr = 0, .len = 0 }, .kept_taps = .{ .d = d, .ptr = 0, .len = 0 }, .lanes = &.{}, .logits_h = &.{}, .amax = .{ .d = d, .ptr = 0, .len = 0 }, .rows = round_rows, .stream = stream,
            .book = .{ .set = .{ .on = false }, .chunk = rows, .window = cfg.window }, .snaps = .{ .d = d, .ptr = 0, .len = 0 },
        };
        self.book.released = &self.released;
        mem_note.note(d, "streams: before the pool", .{});
        self.pool = try rounds.Pool.init(gpa, d, cfg, L, slots, cap);
        errdefer self.pool.deinit();
        self.pool.span = span;
        mem_note.note(d, "streams: the pool ({d} slots, {d} positions, {d} a stream)", .{ slots, cap, span });
        self.view_rings = try gpa.alloc(cuda.DeviceBuffer, L);
        errdefer gpa.free(self.view_rings);
        self.view_planes = try gpa.alloc(?forward.Planes, L);
        errdefer gpa.free(self.view_planes);
        // the view first spans a stream's widest extent (slot 0): the prompt buffers are sized for its capacity
        self.point(0, 0, self.most);
        self.view.cap = span;
        self.view.rings = self.view_rings;
        self.view.planes = self.view_planes;
        self.view.source = self.pool.source;
        self.view.probe = null;
        self.view.engram = if (engram) |g| g.hook() else null;
        self.pr = try prompt.Prompt.init(gpa, d, &self.view, kernels, rows);
        errdefer self.pr.deinit();
        mem_note.note(d, "streams: the prompt path ({d} rows)", .{rows});
        self.runner = try rounds.RoundRunner.init(gpa, d, cfg, base.w, base.set, base.glue, base.ek, base.world, round_rows, base.bucket, base.rope_c, base.rope_w, &self.pool);
        errdefer self.runner.deinit();
        mem_note.note(d, "streams: the round runner (bucket {d})", .{base.bucket});
        if (engram) |g| self.runner.engram = g.roundHook();
        self.lanes = try gpa.alloc(Lane, slots);
        errdefer gpa.free(self.lanes);
        var made: usize = 0;
        errdefer for (self.lanes[0..made]) |l| gpa.free(l.hist);
        for (self.lanes) |*l| {
            l.* = .{ .hist = try gpa.alloc(i32, span) };
            made += 1;
        }
        self.logits_h = try gpa.alloc(f32, round_rows * cfg.vocab);
        errdefer gpa.free(self.logits_h);
        self.amax = try cuda.DeviceBuffer.alloc(d, round_rows * 4);
        errdefer self.amax.free();
        if (spark) |sp| {
            self.taps = try cuda.DeviceBuffer.alloc(d, @max(rows, round_rows) * self.tapCols() * 2);
            errdefer self.taps.free();
            self.kept_taps = try cuda.DeviceBuffer.alloc(d, round_rows * self.tapCols() * 2);
            errdefer self.kept_taps.free();
            self.dpool = try dspark.DraftPool.init(gpa, d, sp.stages.len, slots, sp.dims);
            errdefer self.dpool.?.deinit(gpa);
            self.dc_rings = try gpa.alloc(cuda.DeviceBuffer, sp.stages.len);
            mem_note.note(d, "streams: the taps and the draft pool", .{});
        }
    }

    pub fn deinit(self: *Streams) void {
        const gpa = self.gpa;
        for (self.book.slice()) |k| gpa.free(k.ids);
        if (self.snap_refs.len > 0) gpa.free(self.snap_refs);
        if (self.snaps.len > 0) self.snaps.free();
        if (self.dc_rings.len > 0) gpa.free(self.dc_rings);
        if (self.dpool) |*p| p.deinit(gpa);
        if (self.taps.len > 0) self.taps.free();
        if (self.kept_taps.len > 0) self.kept_taps.free();
        gpa.free(self.logits_h);
        self.amax.free();
        for (self.lanes) |l| gpa.free(l.hist);
        gpa.free(self.lanes);
        self.runner.deinit();
        self.pr.deinit();
        gpa.free(self.view_planes);
        gpa.free(self.view_rings);
        self.pool.deinit();
    }

    /// The device bytes a served engine's streams take a slot at a window of `cap` positions: the pool's
    /// (`Pool.bytesFor`: the rings and compressor inputs of every slot, the window's compressed planes) and the
    /// drafter's rings of `stages` stages, over `slots` slots (rounded up).
    pub fn bytesPerSlot(cfg: *const Config, layers: usize, cap: usize, slots: usize, stages: usize) usize {
        const rings = slots * stages * (cfg.window + 16) * cfg.head_dim * 2;
        return (rounds.Pool.bytesFor(cfg, layers, cap, slots) + rings + slots - 1) / slots;
    }

    fn tapCols(self: *const Streams) usize {
        return self.cfg.dspark_taps.len * self.cfg.dim;
    }

    fn draftRope(self: *const Streams) attention.Rope {
        return if (self.cfg.ratio(self.cfg.n_layers) > 0) self.base.rope_c else self.base.rope_w;
    }

    /// The view's caches over slot `slot` and positions [base, base + size) of the window (model.py `pool_view`):
    /// the slot's window rings and compressor inputs, the extent's compressed rows [base / r, (base + size) / r),
    /// its last one its scratch row.
    fn point(self: *Streams, slot: usize, base: usize, size: usize) void {
        const cfg = self.cfg;
        const p = &self.pool;
        const ring_bytes = p.ring * cfg.head_dim * 2;
        for (self.view_rings, p.rings) |*v, r| v.* = sub(r, slot * ring_bytes, ring_bytes);
        for (self.view_planes, p.planes, 0..) |*v, pp, li| {
            const pl = pp orelse {
                v.* = null;
                continue;
            };
            const r = cfg.ratio(li);
            const lo = base / r;
            const rows = size / r;
            const raw = rounds.raw_rows * cfg.head_dim * 4;
            v.* = .{
                .comp_codes = sub(pl.comp_codes, lo * cfg.head_dim / 2, rows * cfg.head_dim / 2),
                .comp_scales = sub(pl.comp_scales, lo * cfg.head_dim / 16, rows * cfg.head_dim / 16),
                .idxk_codes = if (pl.idxk_codes.len > 0) sub(pl.idxk_codes, lo * cfg.idx_dim / 2, rows * cfg.idx_dim / 2) else pl.idxk_codes,
                .idxk_scales = if (pl.idxk_scales.len > 0) sub(pl.idxk_scales, lo * cfg.idx_dim / 32, rows * cfg.idx_dim / 32) else pl.idxk_scales,
                .raw_kv = if (pl.raw_kv.len > 0) sub(pl.raw_kv, slot * raw, raw) else pl.raw_kv,
                .raw_score = if (pl.raw_score.len > 0) sub(pl.raw_score, slot * raw, raw) else pl.raw_score,
                .rows = rows,
            };
        }
        self.view.cap = size;
    }

    /// Slot `slot`'s drafter cache: its rows of every stage's ring plane (dspark.py `DraftPool.views`).
    fn slotCache(self: *Streams, slot: usize) dspark.Cache {
        const dp = &self.dpool.?;
        const sp = self.spark.?;
        const ring_bytes = sp.dims.ring * sp.dims.hd * 2;
        for (self.dc_rings, dp.rings) |*v, r| v.* = sub(r, slot * ring_bytes, ring_bytes);
        return .{ .rings = self.dc_rings, .absorbed = self.lanes[slot].absorbed };
    }

    /// Whether an admission of `n` prompt ids into `slot` at [base, base + size) can run here: the slot free, the
    /// extent free, aligned and at most `most` positions, the prompt and a round's rows inside it.
    pub fn admits(self: *const Streams, slot: usize, base: usize, size: usize, n: usize, chunk: usize) bool {
        if (slot >= self.lanes.len or self.lanes[slot].busy or chunk < 1 or chunk > self.pr.rows) return false;
        if (n < 1 or size % extent_align != 0 or size > self.most or n + extent_tail > size) return false;
        var probe = self.extents;
        return probe.takeAt(base, size);
    }

    /// multi.py `_admit` + `_fill` to the end (engine.py prefill_steps from 0 on the slot's pool view): the slot and
    /// extent taken, the prompt's chunks of `chunk` rows through the prompt path, each chunk's taps absorbed into the
    /// slot's drafter rings; the last row's logits in `logits`. Runs every chunk once started (the other ranks
    /// cannot see a cancel).
    pub fn admit(self: *Streams, gather: forward.Gather, slot: usize, base: usize, size: usize, ids: []const i64, chunk: usize, replay: bool) !void {
        if (!self.admits(slot, base, size, ids.len, chunk)) return error.PositionMismatch;
        _ = try self.admitKept(gather, slot, base, size, ids, chunk, replay, null, false);
    }

    /// Kept prompts are on for admissions with bounded replay `replay` (the snapshots' layout).
    pub fn keeping(self: *const Streams, replay: bool) bool {
        return self.book.set.on and replay == self.keep_replay;
    }

    /// Kept prompts on (cuda_keep.zig `Settings`): `set.snaps` ring snapshots allocated, boundaries every `chunk`
    /// rows. Off (nothing allocated) when the settings say so or a compressor's ratio does not divide `chunk` (a
    /// boundary must end every compressor's group). Every rank calls it with the same arguments.
    pub fn enableKeep(self: *Streams, set: kept_prompts.Settings, chunk: usize, replay: bool) !void {
        if (!set.on or chunk == 0) return;
        const cfg = self.cfg;
        const p = &self.pool;
        for (p.planes, 0..) |pl, li| if (pl != null and chunk % @max(cfg.ratio(li), 1) != 0) return;
        const L = p.rings.len;
        self.snap_layers = if (replay) @min(cfg.n_layers / 2, L) else L;
        const ring_bytes = p.ring * cfg.head_dim * 2;
        const raw_bytes = rounds.raw_rows * cfg.head_dim * 4;
        var bytes = self.snap_layers * ring_bytes;
        for (p.planes) |pl| if (pl) |x| if (x.raw_kv.len > 0) {
            bytes += 2 * raw_bytes;
        };
        self.snap_bytes = bytes;
        self.snaps = try cuda.DeviceBuffer.alloc(self.d, set.snaps * bytes);
        errdefer self.snaps.free();
        self.snap_refs = try self.gpa.alloc(u8, set.snaps);
        @memset(self.snap_refs, 0);
        self.book.set = set;
        self.book.chunk = chunk;
        self.keep_replay = replay;
    }

    /// The window book's and extents' fingerprint (the records carry rank 0's).
    pub fn fingerprint(self: *const Streams) u64 {
        return self.book.fingerprint(&self.extents);
    }

    /// Where an admission of `size` positions continuing `reuse` would go (the book and extents untouched); null:
    /// no room even with every kept prompt gone.
    pub fn trialPlace(self: *const Streams, size: usize, reuse: ?kept_prompts.Reuse) ?kept_prompts.Placed {
        var b = self.book.trial();
        var ex = self.extents;
        return b.place(&ex, size, if (reuse) |r| r.eid else null);
    }

    /// Whether an admission of the prompt `ids` into `slot` at [base, base + size), continuing `reuse`, can run
    /// here: the slot free, the prompt and a round's rows inside the extent, the kept prompt held with a snapshot at
    /// the boundary and the same ids before it, and the book placing the extent at `base`.
    pub fn admitsKept(self: *const Streams, slot: usize, base: usize, size: usize, ids: []const i64, chunk: usize, replay: bool, reuse: ?kept_prompts.Reuse) bool {
        if (slot >= self.lanes.len or self.lanes[slot].busy or chunk < 1 or chunk > self.pr.rows) return false;
        const n = ids.len;
        if (n < 1 or size % extent_align != 0 or size > self.most or n + extent_tail > size) return false;
        if (reuse) |r| {
            if (!self.keeping(replay) or chunk != self.book.chunk) return false;
            const i = self.book.find(r.eid) orelse return false;
            const k = &self.book.kept[i];
            if (k.replay != replay or k.marks.find(r.cut) == null or r.cut % chunk != 0 or r.cut >= n or r.cut > k.top) return false;
            for (k.keys()[0..r.cut], ids[0..r.cut]) |a, b| if (a != b) return false;
        }
        const placed = self.trialPlace(size, reuse) orelse return false;
        return placed.base == base;
    }

    /// multi.py `_admit` with kept prompts + `_fill` to the end: the extent placed by the book (`Book.place`, at
    /// `base`), and continuing `reuse` the kept prompt's rows to its boundary copied in (unless the extent grew in
    /// place) and its slot's rings and compressor inputs restored from the snapshot there, so the prompt's chunks
    /// from the boundary are a fresh prefill's own; the ring snapshots taken at the prompt's boundaries
    /// (`Book.marksFor`). With `check`, the continued prompt is then prefilled again from 0 and the two compared
    /// (logged: the last row's logits and the caches the prompt wrote). Returns the placement.
    pub fn admitKept(self: *Streams, gather: forward.Gather, slot: usize, base: usize, size: usize, ids: []const i64, chunk: usize, replay: bool, reuse: ?kept_prompts.Reuse, check: bool) !kept_prompts.How {
        const how = try self.begin(slot, base, size, ids, chunk, replay, reuse, check);
        _ = try self.fillPart(gather, slot, ids, std.math.maxInt(u32));
        return how;
    }

    /// `admitKept` up to its first chunk: the extent placed and the kept prompt's rows and slot state restored, the
    /// pass standing at the boundary (`Lane.pass`); `fillPart` runs its chunks. Returns the placement.
    pub fn begin(self: *Streams, slot: usize, base: usize, size: usize, ids: []const i64, chunk: usize, replay: bool, reuse: ?kept_prompts.Reuse, check: bool) !kept_prompts.How {
        if (!self.admitsKept(slot, base, size, ids, chunk, replay, reuse)) return error.PositionMismatch;
        const lane = &self.lanes[slot];
        // the kept prompt's snapshots to the boundary become the stream's too (it may be forgotten placing)
        var inherited: kept_prompts.Marks = .{};
        var src_base: usize = 0;
        var cut: usize = 0;
        var cut_snap: u16 = 0;
        if (reuse) |r| {
            const k = &self.book.kept[self.book.find(r.eid).?];
            cut = r.cut;
            src_base = k.base;
            cut_snap = k.marks.find(cut).?;
            for (k.marks.slice()) |m| if (m.at <= cut) {
                inherited.add(m);
                self.snap_refs[m.snap] += 1;
            };
        }
        self.book.ticks += 1;
        const placed = self.book.place(&self.extents, size, if (reuse) |r| r.eid else null) orelse return error.NoRoom;
        if (placed.base != base) return error.PositionMismatch;
        if (placed.how == .copy) {
            const k = &self.book.kept[self.book.find(reuse.?.eid).?];
            k.tick = self.book.ticks;
            k.hits += 1;
        }
        lane.* = .{ .busy = true, .base = base, .size = size, .hist = lane.hist, .prompt_len = ids.len, .marks = inherited };
        errdefer {
            self.extents.give(base, size);
            lane.busy = false;
        }
        if (self.base.w.rank == 0) {
            var busy: usize = 0;
            for (self.lanes) |l| busy += @intFromBool(l.busy);
            std.log.info("deepseek_v41: slot {d} takes extent [{d}, +{d}) ({t}) for a prompt of {d}; {d} streams and {d} kept prompts resident, {d} of {d} positions free", .{ slot, base, size, placed.how, ids.len, busy, self.book.n, self.extents.freePositions(), self.extents.total });
        }
        for (lane.hist[0..ids.len], ids) |*hv, id| hv.* = @intCast(id);
        self.point(slot, base, size);
        if (placed.how == .copy or placed.how == .move) try self.copyRows(src_base, base, cut);
        if (reuse != null) try self.restore(slot, cut_snap);
        self.settle();
        if (reuse != null) self.keep_stats[@intFromEnum(placed.how)] += 1;
        const marks: ?kept_prompts.Points = if (self.keeping(replay) and chunk == self.book.chunk) self.book.marksFor(ids.len) else null;
        lane.pass = .{ .at = cut, .cut = cut, .chunk = chunk, .replay = replay, .check = check and reuse != null, .marks = marks, .t0 = nowNs() };
        try self.fillStart(slot, ids, cut, chunk);
        return placed.how;
    }

    /// Whether `fillPart(slot, …, chunks)` can run here: a pass under way in the slot, standing at `at`.
    pub fn fills(self: *const Streams, slot: usize, at: usize, chunks: u32) bool {
        if (slot >= self.lanes.len or chunks < 1) return false;
        const l = &self.lanes[slot];
        const p = l.pass orelse return false;
        return l.busy and p.at == at;
    }

    /// The next at most `chunks` chunks of the pass `begin` started in `slot` (its prompt `ids`); true once its last
    /// chunk ran: the stream then stands at its prompt's end, the last row's logits in `logits`, exactly as after a
    /// whole `admitKept` (the rounds that ran between parts read and wrote other slots' state only).
    pub fn fillPart(self: *Streams, gather: forward.Gather, slot: usize, ids: []const i64, chunks: u32) !bool {
        const lane = &self.lanes[slot];
        var p = lane.pass orelse return error.PositionMismatch;
        if (ids.len != lane.prompt_len) return error.PositionMismatch;
        // another slot's rounds and drafts may have moved the view and the drafter's ring views since the last part
        self.point(slot, lane.base, lane.size);
        if (self.engram) |g| g.history = lane.hist[0..ids.len];
        p.at = try self.fillChunks(gather, slot, ids, p.at, chunks, p.chunk, p.replay, if (p.marks) |*m| m else null);
        lane.pass = p;
        if (p.at < ids.len) return false;
        lane.pass = null;
        if (ids.len - p.cut > max_rows or self.pr.trace != null) {
            try self.stream.synchronize();
            const ms = msSince(p.t0);
            std.log.info("deepseek_v41 rank {d}: a prompt of {d} rows from {d} in {d} chunks: {d:.0} ms, {d:.0} tok/s", .{ self.base.w.rank, ids.len, p.cut, (ids.len - p.cut + p.chunk - 1) / p.chunk, ms, @as(f64, @floatFromInt(ids.len - p.cut)) * 1000 / ms });
            try self.pr.traceReport(ids.len - p.cut, ms);
        }
        if (p.check) try self.checkKept(gather, slot, ids, p.cut, p.chunk, p.replay);
        return true;
    }

    /// The prompt's chunks from `from` (a chunk boundary whose rows and slot state are in place) to its end:
    /// engine.py prefill_steps(start=from) on the slot's view; a ring snapshot after each chunk that ends on one of
    /// `marks`.
    fn fill(self: *Streams, gather: forward.Gather, slot: usize, ids: []const i64, from: usize, chunk: usize, replay: bool, marks: ?*const kept_prompts.Points) !void {
        try self.fillStart(slot, ids, from, chunk);
        _ = try self.fillChunks(gather, slot, ids, from, std.math.maxInt(u32), chunk, replay, marks);
    }

    /// A pass from `from`: no logits yet, nothing absorbed, the first chunk's Engram rows read ahead.
    fn fillStart(self: *Streams, slot: usize, ids: []const i64, from: usize, chunk: usize) !void {
        const lane = &self.lanes[slot];
        self.logits = 0;
        lane.absorbed = 0;
        if (self.engram) |g| {
            g.history = lane.hist[0..ids.len];
            try g.resetAhead();
            try g.prefetch(lane.hist[0..ids.len], from, @min(ids.len, from + chunk) - from);
        }
    }

    /// At most `chunks` of the pass's chunks from `from` (`fill`'s loop); the position after them. The pass's last
    /// chunk leaves its last row's logits in `logits` and the stream at the prompt's end; every chunk but the last
    /// reads the next one's Engram rows ahead, across a part's end too (rounds leave those batches pending).
    fn fillChunks(self: *Streams, gather: forward.Gather, slot: usize, ids: []const i64, from: usize, chunks: u32, chunk: usize, replay: bool, marks: ?*const kept_prompts.Points) !usize {
        const lane = &self.lanes[slot];
        var dc: ?dspark.Cache = if (self.spark != null) self.slotCache(slot) else null;
        var s: usize = from;
        var ran: u32 = 0;
        while (s < ids.len and ran < chunks) : (ran += 1) {
            const e = @min(ids.len, s + chunk);
            const last = e == ids.len;
            // the next chunk's Engram rows read while this one runs (engine.py prefill_steps' `ahead`)
            if (self.engram) |g| if (!last) try g.prefetch(lane.hist[0..ids.len], e, @min(ids.len, e + chunk) - e);
            const from_row = replayFrom(replay, ids.len, self.cfg.window);
            var top: [2]usize = .{ s, e - s };
            const out = if (self.spark != null and e - s <= max_rows and (from_row orelse 0) <= s)
                try self.rowsTapped(gather, ids[s..e], s)
            else blk: {
                const o = try self.pr.chunk(gather, ids[s..e], s, .{
                    .replay = from_row,
                    .logits = last or e - s <= max_rows,
                    .taps = if (self.spark != null) .{ .buf = self.taps.ptr, .stride = self.tapCols() } else null,
                }, self.stream);
                top = .{ self.pr.top_start, self.pr.top_rows };
                break :blk o;
            };
            if (self.spark) |sp| if (top[1] > 0) try sp.absorb(&dc.?, self.taps.ptr, top[0], top[1], self.draftRope(), self.stream);
            if (last) self.logits = out orelse return error.BadChunk;
            if (marks) |m| if (e % chunk == 0 and m.has(e) and lane.marks.find(e) == null and lane.marks.n < kept_prompts.max_marks) {
                if (try self.snapshot(slot)) |id| lane.marks.add(.{ .at = e, .snap = id });
            };
            s = e;
        }
        lane.absorbed = if (dc) |c| c.absorbed else 0;
        if (s == ids.len) lane.pos = ids.len;
        return s;
    }

    /// What the book let go, freed: its snapshots' holds dropped, its kept ids' storage.
    fn settle(self: *Streams) void {
        const r = &self.released;
        for (r.snaps[0..r.n_snaps]) |id| self.snap_refs[id] -|= 1;
        for (r.ids[0..r.n_ids]) |ids| self.gpa.free(ids);
        r.* = .{};
    }

    /// Slot `slot`'s window rings (the first `snap_layers` layers') and compressor inputs into a free snapshot (multi.py
    /// `_snapshot`); with none free, the least recently used kept prompts give theirs back first; null when no
    /// snapshot frees (the boundary is not kept).
    fn snapshot(self: *Streams, slot: usize) !?u16 {
        var open = std.mem.indexOfScalar(u8, self.snap_refs, 0);
        while (open == null) {
            if (!self.book.forgetOldest(&self.extents)) return null;
            self.settle();
            open = std.mem.indexOfScalar(u8, self.snap_refs, 0);
        }
        const id = open.?;
        self.snap_refs[id] = 1;
        var off = id * self.snap_bytes;
        const ring_bytes = self.pool.ring * self.cfg.head_dim * 2;
        for (self.pool.rings[0..self.snap_layers]) |r| {
            try self.snaps.copyFrom(off, r.ptr + slot * ring_bytes, ring_bytes, self.stream.handle);
            off += ring_bytes;
        }
        const raw = rounds.raw_rows * self.cfg.head_dim * 4;
        for (self.pool.planes) |pl| if (pl) |x| if (x.raw_kv.len > 0) {
            try self.snaps.copyFrom(off, x.raw_kv.ptr + slot * raw, raw, self.stream.handle);
            try self.snaps.copyFrom(off + raw, x.raw_score.ptr + slot * raw, raw, self.stream.handle);
            off += 2 * raw;
        };
        return @intCast(id);
    }

    /// Snapshot `id` back into slot `slot`'s rings and compressor inputs (multi.py `_restore`).
    fn restore(self: *Streams, slot: usize, id: u16) !void {
        var off = @as(usize, id) * self.snap_bytes;
        const ring_bytes = self.pool.ring * self.cfg.head_dim * 2;
        for (self.pool.rings[0..self.snap_layers]) |r| {
            try sub(r, slot * ring_bytes, ring_bytes).copyFrom(0, self.snaps.ptr + off, ring_bytes, self.stream.handle);
            off += ring_bytes;
        }
        const raw = rounds.raw_rows * self.cfg.head_dim * 4;
        for (self.pool.planes) |pl| if (pl) |x| if (x.raw_kv.len > 0) {
            try sub(x.raw_kv, slot * raw, raw).copyFrom(0, self.snaps.ptr + off, raw, self.stream.handle);
            try sub(x.raw_score, slot * raw, raw).copyFrom(0, self.snaps.ptr + off + raw, raw, self.stream.handle);
            off += 2 * raw;
        };
    }

    /// The compressed and indexer rows of positions [src, src + n) of the window to [dst, dst + n) (multi.py
    /// `_copy_rows`; `n` a multiple of every ratio). Overlapping ranges copy in pieces no longer than their distance,
    /// in the direction that reads every row before it is overwritten.
    fn copyRows(self: *Streams, src: usize, dst: usize, n: usize) !void {
        if (src == dst or n == 0) return;
        const cfg = self.cfg;
        for (self.pool.planes, 0..) |pl, li| if (pl) |x| {
            const r = cfg.ratio(li);
            const a = src / r;
            const b = dst / r;
            const k = n / r;
            const parts = [_]struct { buf: cuda.DeviceBuffer, row: usize }{
                .{ .buf = x.comp_codes, .row = cfg.head_dim / 2 },
                .{ .buf = x.comp_scales, .row = cfg.head_dim / 16 },
                .{ .buf = x.idxk_codes, .row = cfg.idx_dim / 2 },
                .{ .buf = x.idxk_scales, .row = cfg.idx_dim / 32 },
            };
            for (parts) |part| {
                if (part.buf.len == 0) continue;
                const gap = if (a > b) a - b else b - a;
                const piece = @max(1, @min(k, gap));
                var done: usize = 0;
                while (done < k) {
                    const m = @min(piece, k - done);
                    // forward when moving down, backward when moving up
                    const at = if (b < a) done else k - done - m;
                    try part.buf.copyFrom((b + at) * part.row, part.buf.ptr + (a + at) * part.row, m * part.row, self.stream.handle);
                    done += m;
                }
            }
        };
    }

    /// TF_DS_KEEP_CHECK: the continued prompt's last row's logits and the caches it wrote (its extent's compressed
    /// and indexer rows to its end, its slot's rings and compressor inputs) against the same prompt prefilled from 0
    /// in the same slot and extent, which the stream then decodes from. Logged on every rank (each holds its own
    /// shard).
    fn checkKept(self: *Streams, gather: forward.Gather, slot: usize, ids: []const i64, cut: usize, chunk: usize, replay: bool) !void {
        const v = self.cfg.vocab;
        try self.stream.synchronize();
        const cont = try self.gpa.alloc(f32, v);
        defer self.gpa.free(cont);
        try (cuda.DeviceBuffer{ .d = self.d, .ptr = self.logits, .len = v * 4 }).download(0, std.mem.sliceAsBytes(cont));
        const a = try self.cacheDigest(ids.len);
        try self.fill(gather, slot, ids, 0, chunk, replay, null);
        try self.stream.synchronize();
        const fresh = try self.gpa.alloc(f32, v);
        defer self.gpa.free(fresh);
        try (cuda.DeviceBuffer{ .d = self.d, .ptr = self.logits, .len = v * 4 }).download(0, std.mem.sliceAsBytes(fresh));
        const b = try self.cacheDigest(ids.len);
        const same_logits = std.mem.eql(u8, std.mem.sliceAsBytes(cont), std.mem.sliceAsBytes(fresh));
        const same_caches = std.mem.eql(u8, &a, &b);
        if (!same_logits or !same_caches) self.keep_differed += 1;
        std.log.info("deepseek_v41 rank {d}: keep check: a prompt of {d} rows continued at {d}: last row's logits {s}, caches {s} (against a prefill from 0)", .{ self.base.w.rank, ids.len, cut, if (same_logits) "equal" else "DIFFER", if (same_caches) "equal" else "DIFFER" });
    }

    /// sha256 of the view's caches (its slot's and extent's) after a prompt of `n` rows: its extent's compressed and indexer rows of the
    /// prompt's positions (each layer's up to its last whole group), every layer's window ring, every compressor's
    /// inputs.
    fn cacheDigest(self: *Streams, n: usize) ![32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(self.gpa);
        for (self.view_planes, 0..) |pl, li| if (pl) |x| {
            const rows = n / self.cfg.ratio(li);
            for ([_]struct { buf: cuda.DeviceBuffer, row: usize }{
                .{ .buf = x.comp_codes, .row = self.cfg.head_dim / 2 },
                .{ .buf = x.comp_scales, .row = self.cfg.head_dim / 16 },
                .{ .buf = x.idxk_codes, .row = self.cfg.idx_dim / 2 },
                .{ .buf = x.idxk_scales, .row = self.cfg.idx_dim / 32 },
            }) |part| {
                if (part.buf.len == 0) continue;
                try host.resize(self.gpa, rows * part.row);
                try part.buf.download(0, host.items);
                h.update(host.items);
            }
            if (x.raw_kv.len > 0) for ([_]cuda.DeviceBuffer{ x.raw_kv, x.raw_score }) |raw| {
                try host.resize(self.gpa, raw.len);
                try raw.download(0, host.items);
                h.update(host.items);
            };
        };
        for (self.view_rings) |r| {
            try host.resize(self.gpa, r.len);
            try r.download(0, host.items);
            h.update(host.items);
        }
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }

    /// Rank 0: the kept prompts stream `slot`'s prompt will cover when it finishes (`Book.covered`), into `out`.
    pub fn covered(self: *const Streams, slot: usize, out: []u32) usize {
        const l = &self.lanes[slot];
        if (!self.book.set.on or l.marks.n == 0) return 0;
        var have: [kept_prompts.max_marks]usize = undefined;
        for (l.marks.slice(), have[0..l.marks.n]) |m, *at| at.* = m.at;
        return self.book.covered(l.hist[0..l.prompt_len], have[0..l.marks.n], self.keep_replay, out);
    }

    /// The view's decode loop with the window's taps written into `taps` (dspark's tap layers' inputs).
    fn rowsTapped(self: *Streams, gather: forward.Gather, ids: []const i64, start: usize) !Ptr {
        self.prev_probe = self.view.probe;
        self.tap_rows = ids.len;
        self.view.probe = .{ .ctx = self, .call = tapProbe };
        defer self.view.probe = self.prev_probe;
        return self.view.rows(gather, ids, start, ids.len, false, self.stream);
    }

    fn tapProbe(ctx: *anyopaque, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) anyerror!void {
        const self: *Streams = @ptrCast(@alignCast(ctx));
        if (self.prev_probe) |p| try p.call(p.ctx, name, ptr, bytes, stream);
        const v = &self.view;
        var gated: u64 = 0;
        if (v.engram != null) for (v.w.layers, 0..) |*l, i| {
            if (l.engram_wkv != null and i < 64) gated |= @as(u64, 1) << @intCast(i);
        };
        const li = Sequence.tapLayer(name, v.w.layers.len, gated) orelse return;
        const col = self.cfg.dspark_taps.indexOf(@intCast(li)) orelse return;
        try v.hc.tap(stream, ptr, self.taps.ptr + col * self.cfg.dim * 2, @intCast(self.tapCols()), @intCast(self.tap_rows));
    }

    /// Whether `windows` can run as a round here: busy slots, each once, each window at its stream's length and
    /// extent, inside its limit, each window `max_rows` rows at most, the round `rows` at most.
    pub fn roundable(self: *const Streams, windows: []const RoundWindow) bool {
        var rows: usize = 0;
        var seen: u64 = 0;
        for (windows) |w| {
            if (w.slot >= self.lanes.len or seen & (@as(u64, 1) << @intCast(w.slot)) != 0) return false;
            seen |= @as(u64, 1) << @intCast(w.slot);
            const l = &self.lanes[w.slot];
            if (!l.busy or w.base != l.base or w.size != l.size or w.position != l.pos or w.ids.len < 1) return false;
            if (w.position + w.ids.len > l.limit() or w.ids.len > max_rows) return false;
            rows += w.ids.len;
        }
        return windows.len >= 1 and rows <= self.rows;
    }

    /// multi.py `_round`'s forward (rounds.py RoundRunner): every window's rows at its stream's length through one
    /// forward; every row's logits [R, V] in `logits` (window order), their taps in `runner.taps`. Each stream's
    /// length moves past its window (`keep` or a draft's absorb rolls it back to the kept rows).
    pub fn round(self: *Streams, gather: forward.Gather, windows: []const RoundWindow) !void {
        if (!self.roundable(windows)) return error.PositionMismatch;
        var ws: [max_slots]rounds.Window = undefined;
        for (windows, ws[0..windows.len]) |w, *o| {
            const l = &self.lanes[w.slot];
            for (l.hist[l.pos..][0..w.ids.len], w.ids) |*hv, id| hv.* = @intCast(id);
            o.* = .{ .slot = w.slot, .base = l.base, .size = l.size, .position = l.pos, .ids = w.ids, .host = l.hist[0 .. l.pos + w.ids.len] };
        }
        self.logits = try self.runner.forward(ws[0..windows.len], gather, self.stream);
        var r: usize = 0;
        for (windows) |w| {
            const l = &self.lanes[w.slot];
            l.verify_start = l.pos;
            l.verify_rows = w.ids.len;
            l.row0 = r;
            l.held = 0;
            l.pos += w.ids.len;
            r += w.ids.len;
        }
    }

    /// Whether a stream may keep `kept` rows of its last window at `start` (already kept: its length there).
    pub fn keepable(self: *const Streams, k: KeepAsk) bool {
        if (k.slot >= self.lanes.len) return false;
        const l = &self.lanes[k.slot];
        return l.busy and k.start == l.verify_start and k.kept >= 1 and k.kept <= l.verify_rows and (l.pos == k.start + l.verify_rows or l.pos == k.start + k.kept);
    }

    /// multi.py `sc.length = P + a + 1`: the stream's length at its kept rows; the rejected rows' cache writes stay
    /// and are overwritten before any later row reads them.
    pub fn keep(self: *Streams, k: KeepAsk) !void {
        if (!self.keepable(k)) return error.PositionMismatch;
        self.lanes[k.slot].pos = k.start + k.kept;
    }

    /// Whether the draft requests can run here: busy slots, each once; absorbs of the last window's kept rows; a
    /// draft at the stream's length once absorbed.
    pub fn draftable(self: *const Streams, asks: []const DraftAsk) bool {
        if (self.spark == null) return false;
        var seen: u64 = 0;
        for (asks) |a| {
            if (a.slot >= self.lanes.len or seen & (@as(u64, 1) << @intCast(a.slot)) != 0) return false;
            seen |= @as(u64, 1) << @intCast(a.slot);
            const l = &self.lanes[a.slot];
            if (!l.busy) return false;
            var pos = l.pos;
            var absorbed = l.absorbed;
            if (a.absorb > 0) {
                if (a.start != l.verify_start or a.absorb > l.verify_rows) return false;
                if (l.pos != a.start + l.verify_rows and l.pos != a.start + a.absorb) return false;
                pos = a.start + a.absorb;
                absorbed = pos;
            }
            if (a.depth > 0 and (a.q0 != pos or absorbed != pos or a.token < 0 or a.token >= @as(i64, @intCast(self.cfg.vocab)))) return false;
        }
        return true;
    }

    /// The drafts step (multi.py `_round`'s absorb of the kept rows, then `_drafts`): every stream's kept rows' taps
    /// into its slot's rings in one pass (dspark.py `absorb_many`, the rows in request order), each stream's length
    /// at them; then the drafting streams' blocks in batched passes over the DraftPool (dspark.py
    /// `BatchDraftGraph`, at most the drafter's `passStreams` a pass: every slot's), each stream holding its first `depth` drafts.
    pub fn draft(self: *Streams, asks: []const DraftAsk) !void {
        if (!self.draftable(asks)) return error.PositionMismatch;
        const sp = self.spark.?;
        const dp = &self.dpool.?;
        const rope = self.draftRope();
        const row_bytes = self.tapCols() * 2;
        var pos: [round_rows_most]usize = undefined;
        var slot_of: [round_rows_most]usize = undefined;
        var k: usize = 0;
        for (asks) |a| {
            const l = &self.lanes[a.slot];
            l.held = 0;
            if (a.absorb == 0) continue;
            try self.kept_taps.copyFrom(k * row_bytes, self.runner.taps.ptr + l.row0 * row_bytes, a.absorb * row_bytes, self.stream.handle);
            for (0..a.absorb) |i| {
                pos[k + i] = a.start + i;
                slot_of[k + i] = a.slot;
            }
            k += a.absorb;
            l.pos = a.start + a.absorb;
            l.absorbed = l.pos;
        }
        try sp.absorbRows(dp, self.kept_taps.ptr, pos[0..k], slot_of[0..k], rope, self.stream);
        var tokens: [max_slots]u32 = undefined;
        var q0: [max_slots]usize = undefined;
        var slots: [max_slots]usize = undefined;
        var depth: [max_slots]usize = undefined;
        var n: usize = 0;
        for (asks) |a| if (a.depth > 0) {
            tokens[n] = @intCast(a.token);
            q0[n] = a.q0;
            slots[n] = a.slot;
            depth[n] = a.depth;
            n += 1;
        };
        const per = sp.passStreams();
        var out: [max_slots]dspark.Drafts = undefined;
        var at: usize = 0;
        while (at < n) {
            const m = @min(per, n - at);
            try sp.batchDraft(dp, tokens[at..][0..m], q0[at..][0..m], slots[at..][0..m], self.base.w.embed.ptr(), rope, self.stream, out[at..][0..m]);
            at += m;
        }
        for (0..n) |i| {
            const l = &self.lanes[slots[i]];
            l.drafts = out[i];
            l.held = @min(depth[i], out[i].len);
        }
    }

    /// The stream in `slot` leaves: its extent free again (multi.py `_release`), its snapshots let go.
    pub fn free(self: *Streams, slot: usize) void {
        if (slot >= self.lanes.len or !self.lanes[slot].busy) return;
        const l = &self.lanes[slot];
        self.extents.give(l.base, l.size);
        for (l.marks.slice()) |m| self.snap_refs[m.snap] -|= 1;
        l.marks = .{};
        l.busy = false;
        l.held = 0;
        l.pass = null;
    }

    /// multi.py `_finish` for the stream in `slot`: with kept prompts on and a snapshot taken, its prompt stays as a
    /// kept prompt (`Book.keep`: its rows to its last kept boundary, the rest of its extent free again), else it
    /// leaves (`free`); then the kept prompts `drops` (rank 0's `covered`) go, and the oldest past the entries.
    pub fn finish(self: *Streams, slot: usize, drops: []const u32) void {
        if (slot >= self.lanes.len or !self.lanes[slot].busy) return;
        const l = &self.lanes[slot];
        const owned: ?[]i32 = if (self.book.set.on and l.marks.n > 0) self.gpa.alloc(i32, l.marks.last().?) catch null else null;
        if (owned) |o| {
            self.book.keep(&self.extents, l.base, l.size, l.hist[0..l.prompt_len], &l.marks, self.keep_replay, o);
            l.marks = .{};
            l.busy = false;
            l.held = 0;
            l.pass = null;
        } else self.free(slot);
        self.book.finish(&self.extents, drops);
        self.settle();
    }

    /// Whether no stream holds a slot.
    pub fn idle(self: *const Streams) bool {
        for (self.lanes) |l| if (l.busy) return false;
        return true;
    }

    /// The first `out.len` rows of `logits` (from row `row0`): each row's argmax (torch's rule: the first index of
    /// the maximum; a NaN never larger, 0 when a row's first value is NaN). With the runner's tail on (TF_DS_TAIL) the
    /// device draws them and only the tokens come back; else the rows' logits come back and the host draws them.
    pub fn argmaxRows(self: *Streams, row0: usize, out: []u32) !void {
        const v = self.cfg.vocab;
        if (out.len > self.rows) return error.WindowTooWide;
        if (out.len == 0) return;
        if (self.runner.tail and try self.runner.glue.argmaxRows(self.logits + row0 * v * 4, v, v, out.len, self.amax.ptr, self.stream)) {
            try self.stream.synchronize();
            try self.amax.download(0, std.mem.sliceAsBytes(out));
            return;
        }
        try self.stream.synchronize();
        const view: cuda.DeviceBuffer = .{ .d = self.d, .ptr = self.logits + row0 * v * 4, .len = out.len * v * 4 };
        try view.download(0, std.mem.sliceAsBytes(self.logits_h[0 .. out.len * v]));
        for (out, 0..) |*o, r| {
            const row = self.logits_h[r * v ..][0..v];
            var best: usize = 0;
            for (row, 0..) |x, i| if (x > row[best]) {
                best = i;
            };
            o.* = @intCast(best);
        }
    }

    /// The timing step's round: `rows` noise ids at position 0 of slot 0 over a stream's widest extent (no stream's state
    /// changes; every slot must be free: a later admission rewrites every row a later row reads). It scans the
    /// least bucket.
    pub fn probeRound(self: *Streams, gather: forward.Gather, rows: usize) !void {
        try self.probeRoundAt(gather, rows, 0);
    }

    /// The deepest position a probe round of `rows` rows can take whose bucket is the runner's widest (or a
    /// stream's widest extent's end, for one narrower than it).
    pub fn probeDeep(self: *const Streams, rows: usize) usize {
        const rope_rows = @min(self.runner.rope_c.rows, self.runner.rope_w.rows);
        return @min(@min(self.most, rope_rows), self.runner.bucket) -| rows;
    }

    /// `probeRound` with its rows at `position` .. `position + rows - 1`: the round scans their bucket.
    pub fn probeRoundAt(self: *Streams, gather: forward.Gather, rows: usize, position: usize) !void {
        const noise: i32 = @intCast(self.cfg.dspark_noise);
        if (rows < 1 or rows > self.rows) return error.WindowTooWide;
        var ids: [round_rows_most]i64 = @splat(noise);
        const w = [1]rounds.Window{.{ .slot = 0, .base = 0, .size = self.most, .position = position, .ids = ids[0..rows] }};
        if (self.engram != null) {
            // the rows' n-grams reach before them: noise there too (the history is otherwise unset)
            @memset(self.lanes[0].hist[position -| 64 .. position + rows], noise);
            var hw = w;
            hw[0].host = self.lanes[0].hist[0 .. position + rows];
            self.logits = try self.runner.forward(&hw, gather, self.stream);
        } else self.logits = try self.runner.forward(&w, gather, self.stream);
    }

    /// Records the round graphs of 1 .. min(`rows`, `warm_rows_max`) rows at the buckets bucket_min ..
    /// `warm_bucket_max` the runner scans (the others are recorded on first use) and the batched drafts' of 1 .. `passStreams` streams, ahead of the rounds and drafts
    /// that replay them; nothing runs (TF_DS_GRAPHS=0: nothing to record). Every rank calls it at the same point with
    /// the same streams.
    pub fn warm(self: *Streams, gather: forward.Gather) !void {
        self.runner.graph_bucket_max = warm_bucket_max;
        self.runner.graph_rows_max = warm_rows_max;
        self.runner.graph_budget = try rounds.graphBudgetFromEnv();
        var rows: [round_rows_most]usize = undefined;
        for (&rows, 1..) |*r, i| r.* = i;
        var buckets: [8]usize = undefined;
        var nb: usize = 0;
        var b: usize = rounds.bucket_min;
        while (b <= warm_bucket_max and nb < buckets.len) : (b *= 2) {
            buckets[nb] = b;
            nb += 1;
        }
        try self.runner.warm(rows[0..self.rows], buckets[0..nb], gather, self.stream);
        if (self.spark) |sp| try sp.warmBatch(&self.dpool.?, @min(sp.passStreams(), self.lanes.len), self.draftRope(), self.stream);
    }

    /// The timing step's drafts: `n` slots (0 ..) absorb the last probe round's first row at position 0, then one
    /// batched draft over them at position 1.
    pub fn probeDrafts(self: *Streams, n: usize) !void {
        const sp = self.spark orelse return;
        const dp = &self.dpool.?;
        var tokens: [max_slots]u32 = undefined;
        var q0: [max_slots]usize = undefined;
        var slots: [max_slots]usize = undefined;
        for (0..n) |i| {
            try sp.absorbSlot(dp, i, self.runner.taps.ptr, 0, 1, self.draftRope(), self.stream);
            tokens[i] = @intCast(self.cfg.dspark_noise);
            q0[i] = 1;
            slots[i] = i;
        }
        var out: [max_slots]dspark.Drafts = undefined;
        try sp.batchDraft(dp, tokens[0..n], q0[0..n], slots[0..n], self.base.w.embed.ptr(), self.draftRope(), self.stream, out[0..n]);
    }
};

fn sub(b: cuda.DeviceBuffer, off: usize, len: usize) cuda.DeviceBuffer {
    return .{ .d = b.d, .ptr = b.ptr + off, .len = len };
}

/// Positions the base decode loop's own caches hold: they go unused (the streams' caches are the pool's), so the
/// least a `Forward` takes.
const base_cap: usize = extent_align;

pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    cfg: Config,
    settings: Settings,
    set: cuda.aot.Set,
    glue: attention.Glue,
    kernels: exl3.Kernels,
    ek: experts.Kernels,
    w: weights.Weights,
    rope_c: attention.Rope,
    rope_w: attention.Rope,
    fwd: forward.Forward,
    engram: ?*engram_apply.Engram,
    /// The DSpark drafter (settings.drafts): its weights, its stages' views and the drafter over them.
    ds: ?weights.DSpark,
    stages: []dspark.Stage,
    drafter: ?*dspark.Drafter,
    /// A round's ms by rows (index rows - 1) at the least bucket and a batched draft's by streams (index streams -
    /// 1), timed at load by the `measure` step (rank 0's clock).
    round_ms: [round_rows_most]f64,
    /// A round's ms of 1 and `streams.rows` rows at the widest bucket (the context's), the same step's.
    round_deep_ms: [2]f64,
    draft_ms: [max_slots]f64,
    streams: Streams,
    /// Prompt rows a chunk (the settings' chunk, at most the context).
    chunk: usize,
    comm: cuda.comm.Comm,
    stream: cuda.Stream,
    agree_send: cuda.DeviceBuffer,
    agree_recv: cuda.DeviceBuffer,
    /// The digest's page-locked staging: copied to `agree_send` on the stream, in order with the all-gather that
    /// reads it (a plain upload runs outside that order and may land after the gather has read the last step's).
    agree_host: cuda.HostBuffer,
    ids_buf: []i64, // a received prompt's ids
    op_buf: []u8,
    stepped: u64 = 0,
    /// How rank 1 waits for rank 0's records and the link's timings (`LinkWait`).
    link: LinkWait = .{},
    /// A step failed on this rank past its agreement: the other rank may sit in a collective, so no further
    /// collective is issued (it would pair with the wrong one or hang); the ranks must be restarted.
    broken: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, s: Settings) !*Engine {
        if (s.world != 2) return error.WorldNotTwo;
        if (s.slots < 1 or s.slots > max_slots) return error.BadSlots;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.d = ctx.d;
        e.settings = s;
        e.stepped = 0;
        if (s.chunk < 1 or s.chunk > prompt.max_chunk) return error.BadChunk;
        e.chunk = @min(s.chunk, s.context);
        e.broken = false;
        var clock = Clock.start(s.rank);
        clock.d = ctx.d;
        mem_note.note(ctx.d, "before the load", .{});
        e.cfg = try Config.read(gpa, io, dir);
        if (s.drafts and s.most_drafts > e.cfg.dspark_block) {
            std.log.err("TF_DS_DRAFTS is {d}: the drafter's block holds {d} drafts", .{ s.most_drafts, e.cfg.dspark_block });
            return error.WindowTooWide;
        }
        e.set = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, s.kernels_dir);
        errdefer e.set.deinit();
        clock.mark("the Triton set", .{});
        e.glue = try attention.Glue.load(ctx.d);
        errdefer e.glue.deinit();
        e.kernels = try exl3.Kernels.load(ctx.d);
        errdefer e.kernels.deinit();
        e.ek = try experts.Kernels.load(ctx.d);
        errdefer e.ek.deinit();
        clock.mark("the CUDA modules", .{});
        const rc_path = (try RankCache.locate(gpa, io, s.cache_dir, s.rank, s.world)) orelse return error.NoRankCache;
        defer gpa.free(rc_path);
        var rc = try RankCache.open(gpa, rc_path);
        defer rc.close();
        std.log.info("deepseek_v41 rank {d} of {d}: weights from {s}", .{ s.rank, s.world, rc_path });
        e.w = try weights.load(gpa, ctx.d, &e.cfg, &rc, &e.kernels, s.rank, s.world, .mul1, null);
        errdefer e.w.deinit();
        clock.mark("the weights", .{});
        e.stream = try cuda.Stream.init(ctx.d, true);
        errdefer e.stream.deinit();
        const rows = ropeRows(s.context);
        const rope_mode = try ropeMode();
        e.rope_c = try loadRope(gpa, io, ctx.d, s.rope_dir, "rope_c", rows, rope_mode);
        errdefer freeRope(&e.rope_c);
        e.rope_w = try loadRope(gpa, io, ctx.d, s.rope_dir, "rope_w", rows, rope_mode);
        errdefer freeRope(&e.rope_w);
        clock.mark("the RoPE {t} ({d} positions)", .{ rope_mode, rows });
        e.fwd = try forward.Forward.init(gpa, ctx.d, &e.cfg, &e.w, &e.set, &e.glue, &e.ek, s.world, max_rows, base_cap, s.bucket, e.rope_c, e.rope_w, e.stream);
        errdefer e.fwd.deinit();
        clock.mark("the decode loop", .{});
        e.engram = null;
        if (getenv("TF_DS_ENGRAM")) |engram_dir| {
            const tm = if (getenv("TF_DS_TOKEN_MAP")) |p| try gpa.dupe(u8, p) else try std.fs.path.join(gpa, &.{ s.kernels_dir, "token_map.json" });
            defer gpa.free(tm);
            const threads: usize = if (getenv("TF_DS_ENGRAM_THREADS")) |t| std.fmt.parseInt(usize, t, 10) catch 32 else 32;
            e.engram = try engram_apply.Engram.init(gpa, io, ctx.d, &e.cfg, &e.w, &e.set, engram_dir, tm, @max(max_rows, e.chunk), threads);
            std.log.info("deepseek_v41 rank {d}: Engram tables {s}, token map {s}, {d} reader threads", .{ s.rank, engram_dir, tm, threads });
            clock.mark("Engram", .{});
        } else if (s.rank == 0) {
            std.log.warn("deepseek_v41: no Engram tables (TF_DS_ENGRAM): output will be degraded", .{});
        }
        errdefer if (e.engram) |g| g.deinit();
        e.ds = null;
        e.stages = &.{};
        e.drafter = null;
        e.round_ms = @splat(0);
        e.round_deep_ms = @splat(0);
        e.draft_ms = @splat(0);
        errdefer e.freeDrafter();
        if (s.drafts) {
            if (e.cfg.nextn == 0) {
                std.log.err("deepseek_v41: drafts asked for, but the checkpoint has no DSpark blocks", .{});
                return error.NoDraftHead;
            }
            e.ds = try weights.loadDSpark(gpa, ctx.d, &e.cfg, &rc, &e.kernels, s.rank, s.world, .mul1);
            mem_note.note(ctx.d, "the DSpark weights", .{});
            e.stages = try gpa.alloc(dspark.Stage, e.ds.?.blocks.len);
            for (e.stages, e.ds.?.blocks) |*st, *b| st.* = dspark.Stage.of(b);
            var dims = dspark.dimsOf(&e.cfg, s.rank, s.world);
            dims.streams = s.slots; // every slot's block in one pass
            e.drafter = try dspark.Drafter.init(gpa, ctx.d, dims, &e.set, &e.glue, &e.ek, e.stages, dspark.headOf(&e.ds.?, &e.w.head), &e.comm);
            try e.drafter.?.enableWide(&e.kernels);
            std.log.info("deepseek_v41 rank {d}: DSpark drafter ({d} stages, taps of layers {any})", .{ s.rank, e.stages.len, e.cfg.dspark_taps.slice() });
            clock.mark("the DSpark drafter", .{});
        }
        const before = cuda.usage(false).device;
        try e.streams.init(gpa, ctx.d, &e.fwd, &e.kernels, e.engram, e.chunk, e.stream, e.drafter, s.slots, s.poolPositions() + extra, s.context + extra, s.rows);
        errdefer e.streams.deinit();
        if (s.rank == 0) std.log.info("deepseek_v41: {d} stream slots over a window of {d} positions, each stream at most {d} ({d:.2} GiB), prompts in chunks of {d} rows{s}", .{ s.slots, e.streams.extents.total, e.streams.most, @as(f64, @floatFromInt(cuda.usage(false).device - before)) / (1 << 30), e.chunk, if (s.replay) ", bounded replay (TF_DS_REPLAY=1)" else "" });
        {
            const ks = kept_prompts.Settings.fromEnv();
            try e.streams.enableKeep(ks, e.chunk, s.replay);
            const st = &e.streams;
            if (s.rank == 0) {
                if (st.book.set.on)
                    std.log.info("deepseek_v41: kept prompts (TF_DS_KEEP): up to {d}, {d} boundaries each, {d} ring snapshots of {d:.1} MiB ({d:.0} MiB){s}", .{ ks.entries, ks.marks, ks.snaps, @as(f64, @floatFromInt(st.snap_bytes)) / (1 << 20), @as(f64, @floatFromInt(st.snaps.len)) / (1 << 20), if (ks.check) "; every continuation checked against a prefill from 0 (TF_DS_KEEP_CHECK=1)" else "" })
                else
                    std.log.info("deepseek_v41: kept prompts off{s}", .{if (ks.on) " (a compressor's ratio does not divide the chunk)" else " (TF_DS_KEEP=0)"});
            }
        }
        clock.mark("the stream pool and the kept prompts' snapshots", .{});
        if (getenv("TF_DS_PROFILE") != null) try e.streams.pr.enableProfile();
        if (getenv("TF_DS_PREFILL_TRACE") != null) try e.streams.pr.enableTrace();
        // the communicator after the loads: each rank loads on its own, then they meet (rank 1 retries its
        // connection for up to 30 minutes while rank 0 still loads, so the ranks may start in either order)
        e.comm = try cuda.comm.Comm.init(s.rank, s.world, s.master, s.port, 1_800_000);
        errdefer e.comm.deinit();
        e.link = LinkWait.fromEnv();
        if (!e.comm.busyPoll(e.link.busy_poll_us)) std.log.warn("deepseek_v41 rank {d}: SO_BUSY_POLL refused on the control link", .{s.rank});
        std.log.info("deepseek_v41 rank {d}: control link busy poll {d} us, follower spin {d} us{s}", .{ s.rank, e.link.busy_poll_us, e.link.spin_ns / 1000, if (e.link.profile) ", profiled (TF_DS_LINK_PROFILE)" else "" });
        clock.mark("the ranks' rendezvous", .{});
        {
            // every rank records the same graphs in the same order here (the gathers are recorded, not issued)
            const t0 = nowNs();
            const free0 = (try ctx.memInfo()).free;
            try e.streams.warm(e.gather());
            mem_note.note(ctx.d, "the round and draft graphs", .{});
            const free1 = (try ctx.memInfo()).free;
            if (s.rank == 0 and e.streams.runner.graphs) std.log.info("deepseek_v41: {d} round graphs and {d} draft graphs recorded in {d:.1} s, {d:.0} MiB (buckets {d} .. {d}; wider ones {s} {d} MiB)", .{ e.streams.runner.graphCount(), if (e.drafter) |dr| dr.captures else 0, msSince(t0) / 1000, @as(f64, @floatFromInt(@as(i64, @intCast(free0)) - @as(i64, @intCast(free1)))) / (1 << 20), rounds.bucket_min, @min(warm_bucket_max, e.streams.runner.bucket), if (e.streams.runner.graph_budget > 0) "recorded on first use, held under" else "eager (TF_DS_GRAPH_ALL=0), budget", e.streams.runner.graph_budget >> 20 });
        }
        e.agree_send = try cuda.DeviceBuffer.alloc(ctx.d, 16);
        errdefer e.agree_send.free();
        e.agree_recv = try cuda.DeviceBuffer.alloc(ctx.d, 16 * s.world);
        errdefer e.agree_recv.free();
        e.agree_host = try cuda.HostBuffer.alloc(ctx.d, 16);
        errdefer e.agree_host.free();
        e.ids_buf = try gpa.alloc(i64, s.context + extra);
        errdefer gpa.free(e.ids_buf);
        e.op_buf = try gpa.alloc(u8, maxOpBytes(s.context));
        errdefer gpa.free(e.op_buf);
        mem_note.note(ctx.d, "the agreement buffers", .{});
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.gpa.free(e.op_buf);
        e.gpa.free(e.ids_buf);
        e.agree_host.free();
        e.agree_recv.free();
        e.agree_send.free();
        e.streams.deinit();
        e.freeDrafter();
        if (e.engram) |g| g.deinit();
        e.fwd.deinit();
        freeRope(&e.rope_w);
        freeRope(&e.rope_c);
        e.stream.deinit();
        e.w.deinit();
        e.ek.deinit();
        e.kernels.deinit();
        e.glue.deinit();
        e.set.deinit();
        e.comm.deinit();
        e.gpa.destroy(e);
    }

    fn freeDrafter(e: *Engine) void {
        if (e.drafter) |dr| dr.deinit();
        e.drafter = null;
        if (e.stages.len > 0) e.gpa.free(e.stages);
        e.stages = &.{};
        if (e.ds) |*ds| ds.deinit(e.gpa);
        e.ds = null;
    }

    /// The bytes a stream takes for admission (`Streams.bytesPerSlot` of this engine's pool and drafter).
    pub fn streamBytes(e: *const Engine) usize {
        return Streams.bytesPerSlot(&e.cfg, e.w.layers.len, e.settings.poolPositions() + extra, e.settings.slots, e.stages.len);
    }

    fn gather(self: *Engine) forward.Gather {
        return .{ .ctx = self, .call = gatherFn };
    }

    fn gatherFn(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        try e.comm.allGather(partial, out, words, .f32, stream.handle);
    }

    /// Every rank's digest of `op` must match (four 16-bit pieces of its sha256 as f32, one all-gather). A rank that
    /// cannot run `op` (`can_run` false) offers a digest no rank's matches, so every rank stops at the agreement.
    fn agree(e: *Engine, op: []const u8, can_run: bool) !void {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(op, &digest, .{});
        var mine: [4]f32 = undefined;
        for (&mine, 0..) |*m, i| m.* = @floatFromInt(std.mem.readInt(u16, digest[2 * i ..][0..2], .little));
        if (!can_run) mine[0] = -1;
        @memcpy(e.agree_host.bytes[0..16], std.mem.sliceAsBytes(&mine));
        try e.agree_send.uploadAsync(0, e.agree_host.bytes[0..16], e.stream.handle);
        try e.comm.allGather(e.agree_send.ptr, e.agree_recv.ptr, 4, .f32, e.stream.handle);
        try e.stream.synchronize();
        var every: [4 * cuda.comm.Comm.max_world]f32 = undefined;
        try e.agree_recv.download(0, std.mem.sliceAsBytes(every[0 .. 4 * e.settings.world]));
        for (1..e.settings.world) |r| for (0..4) |i| if (every[r * 4 + i] != every[i]) {
            std.log.err("rank {d}: step {d} differs between the ranks (digest piece {d})", .{ e.settings.rank, e.stepped, i });
            e.logDivergence(op, can_run, every[0 .. 4 * e.settings.world]);
            return error.OutOfStep;
        };
        e.stepped += 1;
    }

    /// What a rank saw at a failed agreement: the record's opcode, length and first bytes, whether it could run it,
    /// every rank's digest pieces (-1 first: that rank could not run the record) and its streams' state.
    fn logDivergence(e: *Engine, op: []const u8, can_run: bool, every: []const f32) void {
        const code: []const u8 = if (op.len > 0) (if (std.enums.fromInt(Op, op[0])) |c| @tagName(c) else "?") else "empty";
        const head = op[0..@min(op.len, 48)];
        std.log.err("rank {d}: step {d} record {s}, {d} bytes, runnable here {}, first bytes {x}; digests {any}", .{ e.settings.rank, e.stepped, code, op.len, can_run, head, every });
        for (e.streams.lanes, 0..) |l, i| if (l.busy) {
            std.log.err("rank {d}:   slot {d}: extent [{d}, +{d}), pos {d}, window {d}+{d} (row {d}), absorbed {d}, held {d}, prompt {d}, {d} snapshots, pass at {?d}", .{ e.settings.rank, i, l.base, l.size, l.pos, l.verify_start, l.verify_rows, l.row0, l.absorbed, l.held, l.prompt_len, l.marks.n, if (l.pass) |p| p.at else null });
        };
        for (e.streams.book.slice()) |k| std.log.err("rank {d}:   kept prompt {d}: extent [{d}, +{d}), top {d}, tick {d}, {d} snapshots", .{ e.settings.rank, k.eid, k.base, k.size, k.top, k.tick, k.marks.n });
        std.log.err("rank {d}:   window book fingerprint {x}", .{ e.settings.rank, e.streams.fingerprint() });
    }

    /// Rank 0 sends the step to every other rank, then every rank agrees on it. A failure breaks the lane: a
    /// follower that got the step but not the agreement would otherwise pair its next collective wrongly.
    fn announce(e: *Engine, op: []const u8) !void {
        if (e.broken) return error.OutOfStep;
        errdefer e.broken = true;
        const t0 = nowNs();
        if (e.settings.rank == 0) for (1..e.settings.world) |peer| try e.comm.sendTo(@intCast(peer), op);
        const t1 = nowNs();
        try e.agree(op, true);
        if (e.link.profile) e.link.note(e.settings.rank, .{ t1 - t0, nowNs() - t1 });
    }

    /// Runs a step; a failure breaks the lane on this rank (the other rank fails the same step on its own).
    fn step(e: *Engine, comptime f: anytype, args: anytype) !void {
        if (e.broken) return error.OutOfStep;
        @call(.auto, f, args) catch |err| {
            e.broken = true;
            std.log.err("rank {d}: step {d} failed ({s}); the lane is broken, restart both ranks", .{ e.settings.rank, e.stepped, @errorName(err) });
            return err;
        };
    }

    /// Rank 0 announces `op` and every rank runs it (`run`, the same on every rank).
    fn both(e: *Engine, op: []const u8) !void {
        try e.announce(op);
        try e.step(Engine.run, .{ e, op });
    }

    // -- the steps (the same on every rank) -------------------------------------------------------------------

    fn run(e: *Engine, op: []const u8) !void {
        const code = std.enums.fromInt(Op, op[0]) orelse return error.BadRecord;
        switch (code) {
            .admit => {
                const a = try Admit.parse(op);
                if (a.len() > e.ids_buf.len) return error.PromptTooLong;
                const ids = a.idsInto(e.ids_buf);
                const how = if (a.parts)
                    try e.streams.begin(a.slot, a.base, a.size, ids, a.chunk, a.replay, a.reuse, a.check)
                else
                    try e.streams.admitKept(e.gather(), a.slot, a.base, a.size, ids, a.chunk, a.replay, a.reuse, a.check);
                if (a.reuse) |r| if (e.settings.rank == 0) std.log.info("deepseek_v41: kept prompt {d}: continued at {d} of {d} ({s}); {d} kept prompts", .{ r.eid, r.cut, a.len(), @tagName(how), e.streams.book.n });
            },
            .rounds => {
                var r: Round = undefined;
                try Round.parse(op, &r);
                try e.streams.round(e.gather(), r.windows[0..r.count]);
            },
            .drafts => {
                var d: DraftsRec = undefined;
                try DraftsRec.parse(op, &d);
                try e.streams.draft(d.asks[0..d.count]);
            },
            .keeps => {
                var k: Keeps = undefined;
                try Keeps.parse(op, &k);
                for (k.asks[0..k.count]) |a| try e.streams.keep(a);
            },
            .free => {
                const f = try Free.parse(op);
                e.streams.finish(f.slot, f.drops[0..f.n]);
            },
            .fill => {
                const f = try Fill.parse(op);
                if (!e.streams.fills(f.slot, f.at, f.chunks)) return error.PositionMismatch;
                const l = &e.streams.lanes[f.slot];
                const ids = e.ids_buf[0..l.prompt_len];
                for (ids, l.hist[0..l.prompt_len]) |*id, h| id.* = h;
                _ = try e.streams.fillPart(e.gather(), f.slot, ids, f.chunks);
            },
            .measure => try e.runMeasure(try Measure.parse(op)),
            else => return error.BadRecord,
        }
    }

    /// The measure step (every rank): rounds of 1 .. `streams.rows` rows at position 0 of slot 0 (the least bucket),
    /// of 1 and `streams.rows` rows at the widest bucket (`probeDeep`), then batched drafts of 1 .. passStreams streams, each
    /// timed after a warm-up run. Rank 0 keeps the times. The positions follow from the record's window alone, so
    /// every rank runs the same buckets. With the record's `at`, rank 0 also logs rounds of 1 .. `streams.rows` rows at
    /// that position (at most `probeDeep`'s): the costs the facts leave out, for the record.
    fn runMeasure(e: *Engine, m: Measure) !void {
        const reps: usize = @max(m.reps, 1);
        const st = &e.streams;
        for (1..st.rows + 1) |n| e.round_ms[n - 1] = try e.timeRound(reps, n, 0);
        for ([_]usize{ 1, st.rows }, &e.round_deep_ms) |n, *ms| ms.* = try e.timeRound(reps, n, st.probeDeep(n));
        if (m.at > 0) {
            var at_ms: [round_rows_most]f64 = undefined;
            for (at_ms[0..st.rows], 1..) |*ms, n| ms.* = try e.timeRound(reps, n, @min(m.at, st.probeDeep(n)));
            if (e.settings.rank == 0) std.log.info("deepseek_v41: rounds of 1..{d} rows at position {d}: {any} ms", .{ st.rows, @min(m.at, st.probeDeep(st.rows)), at_ms[0..st.rows] });
        }
        if (st.spark) |sp| {
            const most = @min(sp.passStreams(), st.lanes.len);
            for (1..most + 1) |n| {
                var best: f64 = std.math.inf(f64);
                for (0..reps + 1) |rep| {
                    try e.stream.synchronize();
                    const t0 = nowNs();
                    try st.probeDrafts(n);
                    try e.stream.synchronize();
                    if (rep > 0) best = @min(best, msSince(t0));
                }
                e.draft_ms[n - 1] = best;
            }
        }
    }

    /// The best of `reps` probe rounds of `rows` rows at `position` (after a warm-up one), in ms.
    fn timeRound(e: *Engine, reps: usize, rows: usize, position: usize) !f64 {
        var best: f64 = std.math.inf(f64);
        for (0..reps + 1) |rep| {
            try e.stream.synchronize();
            const t0 = nowNs();
            try e.streams.probeRoundAt(e.gather(), rows, position);
            try e.stream.synchronize();
            if (rep > 0) best = @min(best, msSince(t0));
        }
        return best;
    }

    /// Whether this rank can run `op` as recorded (the checks each step's run makes before its first collective).
    fn runnable(e: *Engine, op: []const u8) bool {
        const code = std.enums.fromInt(Op, op[0]) orelse return false;
        const st = &e.streams;
        switch (code) {
            .admit => {
                const a = Admit.parse(op) catch return false;
                if (a.engram != (e.engram != null) or a.drafts != (e.drafter != null) or a.len() > e.ids_buf.len) return false;
                if (a.keep != st.book.set.on or a.check != (st.book.set.check and a.reuse != null) or a.book != st.fingerprint()) return false;
                return st.admitsKept(a.slot, a.base, a.size, a.idsInto(e.ids_buf), a.chunk, a.replay, a.reuse);
            },
            .rounds => {
                var r: Round = undefined;
                Round.parse(op, &r) catch return false;
                return st.roundable(r.windows[0..r.count]);
            },
            .drafts => {
                var d: DraftsRec = undefined;
                DraftsRec.parse(op, &d) catch return false;
                return st.draftable(d.asks[0..d.count]);
            },
            .keeps => {
                var k: Keeps = undefined;
                Keeps.parse(op, &k) catch return false;
                for (k.asks[0..k.count]) |a| if (!st.keepable(a)) return false;
                return true;
            },
            .free => {
                const f = Free.parse(op) catch return false;
                return f.slot < st.lanes.len and st.lanes[f.slot].busy and f.book == st.fingerprint();
            },
            .fill => {
                const f = Fill.parse(op) catch return false;
                return st.fills(f.slot, f.at, f.chunks);
            },
            .measure => {
                const m = Measure.parse(op) catch return false;
                return m.slots == st.lanes.len and m.total == st.extents.total and m.most == st.most and m.engram == (e.engram != null) and m.drafts == (e.drafter != null) and m.width == e.drafts() and m.rows == st.rows and st.idle();
            },
            else => return false,
        }
    }

    /// Rank 1 (and any other follower): run rank 0's steps until it says stop or the link closes.
    pub fn follow(e: *Engine) !void {
        if (e.settings.rank == 0) return;
        std.log.info("deepseek_v41 rank {d}: following rank 0", .{e.settings.rank});
        while (true) {
            const t0 = nowNs();
            // the next record, polled before the read blocks (rank 0 sends it a fraction of a millisecond after this
            // rank's GPU goes idle; a sleeping thread would add its core's wake-up to every record)
            const op = e.comm.recvFromSpin(0, e.op_buf, e.link.spin_ns) catch |err| switch (err) {
                error.LinkClosed => return,
                else => return err,
            };
            const t1 = nowNs();
            if (op.len == 0) return error.OutOfStep;
            const code = std.enums.fromInt(Op, op[0]) orelse return error.OutOfStep;
            // rank 0 sends stop without an agreement (it is tearing its communicator down)
            if (code == .stop) return;
            const ok = e.runnable(op);
            if (!ok) std.log.err("rank {d}: step {d} ({s}, {d} bytes, first {x}) cannot run here", .{ e.settings.rank, e.stepped, @tagName(code), op.len, op[0..@min(op.len, 48)] });
            try e.agree(op, ok);
            if (e.link.profile) e.link.note(e.settings.rank, .{ t1 - t0, nowNs() - t1 });
            try e.step(Engine.run, .{ e, op });
        }
    }

    /// Rank 0: time the rounds and drafts on every rank (the `measure` step); `round_ms` and `draft_ms` hold rank
    /// 0's.
    pub fn measure(e: *Engine, reps: u8) !void {
        if (e.settings.rank != 0) return;
        var buf: [Measure.bytes]u8 = undefined;
        const at: usize = if (getenv("TF_DS_MEASURE_AT")) |v| std.fmt.parseInt(usize, v, 10) catch 0 else 0;
        try e.both(Measure.encode(&buf, .{ .reps = reps, .slots = e.streams.lanes.len, .engram = e.engram != null, .drafts = e.drafter != null, .total = e.streams.extents.total, .most = e.streams.most, .width = e.drafts(), .at = at, .rows = e.streams.rows }));
    }

    /// The drafts a window holds at most: TF_DS_DRAFTS, else the drafter's block (0 without the drafter).
    pub fn drafts(e: *const Engine) usize {
        if (e.drafter == null) return 0;
        return draftsFor(e.cfg.dspark_block, e.settings.most_drafts);
    }

    pub fn stop(e: *Engine) void {
        if (e.settings.rank != 0) return;
        const op = [_]u8{@intFromEnum(Op.stop)};
        for (1..e.settings.world) |peer| e.comm.sendTo(@intCast(peer), &op) catch {};
    }
};

/// The control link's waits: SO_BUSY_POLL on its sockets, rank 1 polling for the next record before its read
/// blocks, and (TF_DS_LINK_PROFILE=1) every step's link times, logged every `LinkWait.window` steps as p50 / p90 /
/// max in microseconds. Rank 0's pair: the record's send, and from the send to the agreement's gather done on this
/// rank (the record -> gather latency: rank 1's receipt, check and gather launch). Rank 1's pair: the wait for the
/// record (idle time), and from its receipt to the agreement done. Switches: TF_DS_LINK_POLL=0 (neither busy poll nor
/// spin: a blocking read), TF_DS_LINK_BUSY_POLL_US (50), TF_DS_LINK_SPIN_US (20000). None changes what a step
/// computes.
pub const LinkWait = struct {
    busy_poll_us: u32 = 50,
    spin_ns: u64 = 20 * std.time.ns_per_ms,
    profile: bool = false,
    n: usize = 0,
    us: [2][window]f32 = undefined,

    pub const window = 256;

    pub fn fromEnv() LinkWait {
        var w: LinkWait = .{};
        if (getenv("TF_DS_LINK_BUSY_POLL_US")) |v| w.busy_poll_us = std.fmt.parseInt(u32, v, 10) catch w.busy_poll_us;
        if (getenv("TF_DS_LINK_SPIN_US")) |v| w.spin_ns = (std.fmt.parseInt(u64, v, 10) catch w.spin_ns / 1000) * 1000;
        if (getenv("TF_DS_LINK_POLL")) |v| if (std.mem.eql(u8, v, "0")) {
            w.busy_poll_us = 0;
            w.spin_ns = 0;
        };
        if (getenv("TF_DS_LINK_PROFILE")) |v| w.profile = !std.mem.eql(u8, v, "0");
        return w;
    }

    /// One step's two times (ns); a full window logs its summary and starts the next.
    pub fn note(w: *LinkWait, rank: u32, ns: [2]i128) void {
        for (ns, 0..) |x, k| w.us[k][w.n] = @floatCast(@as(f64, @floatFromInt(x)) / 1e3);
        w.n += 1;
        if (w.n < window) return;
        w.n = 0;
        const names: [2][2][]const u8 = .{ .{ "send", "send->agreed" }, .{ "wait", "received->agreed" } };
        var q: [2][3]f32 = undefined;
        for (&w.us, &q) |*xs, *qq| qq.* = quantiles(xs);
        const nm = names[@min(rank, 1)];
        std.log.info("deepseek_v41 rank {d} link ({d} steps, us p50/p90/max): {s} {d:.0}/{d:.0}/{d:.0}, {s} {d:.0}/{d:.0}/{d:.0}", .{ rank, window, nm[0], q[0][0], q[0][1], q[0][2], nm[1], q[1][0], q[1][1], q[1][2] });
    }

    fn quantiles(xs: *[window]f32) [3]f32 {
        std.mem.sort(f32, xs, {}, std.sort.asc(f32));
        return .{ xs[window / 2], xs[window * 9 / 10], xs[window - 1] };
    }
};

test "the link window's quantiles" {
    var w: LinkWait = .{ .profile = true };
    for (0..LinkWait.window) |i| w.note(0, .{ @intCast((LinkWait.window - i) * 1000), @intCast(i * 2000) });
    // the window logged and restarted; its last values were sorted in place
    try std.testing.expectEqual(@as(usize, 0), w.n);
    try std.testing.expectEqual(@as(f32, 1), w.us[0][0]);
    try std.testing.expectEqual(@as(f32, @floatFromInt(LinkWait.window)), w.us[0][LinkWait.window - 1]);
}

/// `weights.DSpark.deinit` (kept for the tests that call it).
pub fn freeDSpark(gpa: std.mem.Allocator, ds: *weights.DSpark) void {
    ds.deinit(gpa);
}
fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

fn msSince(t0: i128) f64 {
    return @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
}

/// `info:` lines of a load's phases: each mark logs the time since the previous one and since the start.
pub const Clock = struct {
    rank: u32,
    t0: i128,
    last: i128,
    /// Set: each mark also notes the phase's device memory (mem_note.zig, TF_DS_MEM_REPORT=1).
    d: ?*const cuda.Driver = null,

    pub fn start(rank: u32) Clock {
        const t = nowNs();
        return .{ .rank = rank, .t0 = t, .last = t };
    }

    pub fn mark(self: *Clock, comptime what: []const u8, args: anytype) void {
        const t = nowNs();
        var buf: [160]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, what, args) catch what;
        std.log.info("deepseek_v41 rank {d}: {s} in {d:.0} ms ({d:.1} s since the load began)", .{ self.rank, name, @as(f64, @floatFromInt(t - self.last)) / 1e6, @as(f64, @floatFromInt(t - self.t0)) / 1e9 });
        if (self.d) |d| mem_note.note(d, what, args);
        self.last = t;
    }
};

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

fn freeRope(r: *attention.Rope) void {
    r.cos.free();
    r.sin.free();
    if (r.freqs) |*f| f.free();
}

/// Where the rounds read RoPE rows from: TF_DS_ROPE=device (the default) computes each pass's rows on the device
/// from the kinds' frequencies (`<dir>/<name>_freqs.bin`, rope.cu), the same bits as the tables, no table held;
/// TF_DS_ROPE=table loads the whole tables (`<dir>/<name>_<rows>.bin`, 2 x 4 x 32 bytes a position a kind).
pub const RopeMode = enum { device, table };

pub fn ropeMode() !RopeMode {
    const v = std.mem.span(std.c.getenv("TF_DS_ROPE") orelse return .device);
    return std.meta.stringToEnum(RopeMode, v) orelse {
        std.log.err("TF_DS_ROPE is {s}: it takes device or table", .{v});
        return error.BadEnvironment;
    };
}

/// A rope kind for positions below `rows` (`mode`: its frequencies, or its table).
fn loadRope(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, name: []const u8, rows: usize, mode: RopeMode) !attention.Rope {
    if (mode == .table) return loadRopeTable(gpa, io, d, dir, name, rows);
    const half = 32;
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}_freqs.bin", .{ dir, name });
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16)) catch {
        std.log.err("no RoPE frequencies at {s}: run tools/zig/dsv41/dump_rope.py --freqs-only (or TF_DS_ROPE=table)", .{path});
        return error.NoRopeTables;
    };
    defer gpa.free(bytes);
    if (bytes.len != half * 4) return error.RopeRowsShort;
    const none: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
    return .{ .cos = none, .sin = none, .rows = rows, .freqs = try cuda.DeviceBuffer.fromHost(d, bytes) };
}

/// `<dir>/<name>_<rows>.bin`: cos then sin, f32 [rows, rope_dim / 2] each (tools/zig/dsv41/dump_rope.py).
fn loadRopeTable(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, name: []const u8, rows: usize) !attention.Rope {
    const half = 32;
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}_{d}.bin", .{ dir, name, rows });
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31)) catch {
        std.log.err("no RoPE table at {s}: run tools/zig/dsv41/dump_rope.py --rows {d}", .{ path, rows });
        return error.NoRopeTables;
    };
    defer gpa.free(bytes);
    if (bytes.len != 2 * rows * half * 4) return error.RopeRowsShort;
    var cos = try cuda.DeviceBuffer.fromHost(d, bytes[0 .. rows * half * 4]);
    errdefer cos.free();
    const sin = try cuda.DeviceBuffer.fromHost(d, bytes[rows * half * 4 ..]);
    return .{ .cos = cos, .sin = sin, .rows = rows };
}

/// The round loop's facts: shared rounds of up to `slots` streams and `rounds.rows_default` rows (`hidden_rows`;
/// `factsWith`: `rows`), windows of
/// up to `draft_width` rows (`factsWith`: `width`); with the drafter, every drafting stream's drafts in one batch (`draft_streams`) and
/// the depths priced by `window_costs` (a lone stream's window, its draft included) and `shared_costs` (a round's
/// rows); without it, host-token windows only.
pub fn factsOf(drafting: bool, slots: usize, window_costs: []const lanes.config.Cost, shared_costs: []const lanes.config.Cost) lanes.Model {
    return factsWith(drafting, draft_width, slots, rounds.rows_default, window_costs, shared_costs);
}

/// `factsOf` with windows of up to `width` rows (a window's drafts at most, plus the pending token) and rounds of up
/// to `rows` rows.
pub fn factsWith(drafting: bool, width: usize, slots: usize, rows: usize, window_costs: []const lanes.config.Cost, shared_costs: []const lanes.config.Cost) lanes.Model {
    return .{
        .exact_width = @intCast(width),
        .gpu_tokens = false,
        .mtp = drafting,
        .speculate = drafting,
        .speculate_early = false,
        .draft_prior = &draft_prior,
        .drafts = @intCast(width - 1),
        .window_costs = window_costs,
        .mtp_step_ms = 0,
        .streams_exact = true,
        .hidden_rows = true,
        .batch_rows = @intCast(rows),
        .max_streams = @intCast(slots),
        .shared_costs = shared_costs,
        .draft_streams = drafting,
        .draft_probabilities = drafting and rounds.tailFromEnv(),
    };
}

/// Prefix survivals of a block's confidences: out[j] = the product of sigmoid(conf[i]) for i <= j, in f64. A
/// confidence's sigmoid is its draft's chance of landing once every earlier draft landed (multi.py's conf rule,
/// dspark.py `choose_k` sums these products as the expected kept drafts), so out[j] is the chance that drafts 0 .. j
/// all land: non-increasing in j, each in [0, 1], out.len <= conf.len.
pub fn survivals(conf: []const f32, out: []f64) void {
    var surv: f64 = 1.0;
    for (out, conf[0..out.len]) |*o, cf| {
        surv *= 1.0 / (1.0 + @exp(-@as(f64, cf)));
        o.* = surv;
    }
}

/// `Loaded.rows` at the default drafts (lanes.Config's max_rows: it caps a window's width and its drafts;
/// `Lanes.windowRows` follows TF_DS_DRAFTS).
pub fn rowsOf(drafting: bool) u32 {
    _ = drafting;
    return draft_width;
}

/// The lane backend on rank 0: up to `slots` streams, each in a slot of the engine's pool; a verify runs every
/// window of a round in one forward (the pending token, then held DSpark drafts or host tokens), each row's draw
/// greedy.
pub const Lanes = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    owner: [max_slots]?*const lanes.Stream = @splat(null),
    drawn: [1024]u32 = undefined,
    next: u64 = 0,
    window_costs: [max_rows]lanes.config.Cost = @splat(.{ .width = 0, .ms = 0 }),
    shared_costs: [round_rows_most]lanes.config.Cost = @splat(.{ .width = 0, .ms = 0 }),
    /// Keeps not yet sent (TF_DS_TAIL on): the next step carries them, and a drafts step whose absorb keeps the same
    /// rows of a stream makes its keep moot (`Streams.draft` sets the length there itself). Every other step sends
    /// them first, so each rank sees the same lengths at every step as without the deferral.
    defer_keeps: bool = false,
    keeps: [max_slots]KeepAsk = undefined,
    n_keeps: usize = 0,
    /// TF_DS_DRAFT_STATS=1: each slot's stream's draft tallies, logged as it leaves, the served streams' summed,
    /// and the rounds by their rows (index: rows)
    stats_on: bool = false,
    stats: [max_slots]draft_stats.Stats = @splat(.{}),
    stats_all: draft_stats.Stats = .{},
    round_rows: [round_rows_most + 1]u64 = @splat(0),

    pub fn init(gpa: std.mem.Allocator, e: *Engine) Lanes {
        return .{ .gpa = gpa, .e = e, .defer_keeps = rounds.tailFromEnv(), .stats_on = draft_stats.enabled() };
    }

    /// The window verified for `slot`: host tokens, or `w.held` of the drafter's held drafts with their predicted
    /// survivals.
    fn offerStats(self: *Lanes, slot: usize, w: be.Window) void {
        const st = &self.stats[slot];
        if (w.tokens.len > 0) {
            st.host_rounds += 1;
            st.pending = null;
            return;
        }
        const l = &self.e.streams.lanes[slot];
        var p: [draft_stats.max_drafts]f64 = undefined;
        const n = @min(w.held, l.drafts.len, p.len);
        survivals(l.drafts.conf[0..n], p[0..n]);
        st.offer(w.held, l.held, p[0..n]);
    }

    fn logStats(self: *Lanes, slot: usize) void {
        const st = &self.stats[slot];
        st.pending = null;
        self.stats_all.add(st.*);
        var buf: [1024]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        st.write(&w) catch {};
        std.log.info("deepseek_v41: draft stats, slot {d}: {s}", .{ slot, w.buffered() });
        var ab: [1024]u8 = undefined;
        var aw: std.Io.Writer = .fixed(&ab);
        self.stats_all.write(&aw) catch {};
        aw.writeAll(" rows=") catch {};
        var first = true;
        for (self.round_rows, 0..) |c, r| if (c > 0) {
            aw.print("{s}{d}:{d}", .{ if (first) "" else ",", r, c }) catch {};
            first = false;
        };
        std.log.info("deepseek_v41: draft stats, every stream so far: {s}", .{aw.buffered()});
        st.* = .{};
    }

    /// The deferred keeps as one keeps step (none: nothing sent).
    fn sendKeeps(self: *Lanes) !void {
        if (self.n_keeps == 0) return;
        const n = self.n_keeps;
        self.n_keeps = 0;
        try self.e.both(Keeps.encode(self.e.op_buf, self.keeps[0..n]));
    }

    /// Whether a drafts step of `requests` makes every deferred keep moot: each kept stream absorbs exactly its kept
    /// rows of the same window (same start), which leaves its length where the keep would (`Streams.draft`).
    fn keepsMoot(self: *const Lanes, requests: []const be.DraftRequest) bool {
        for (self.keeps[0..self.n_keeps]) |k| {
            const moot = for (requests) |r| {
                const slot = self.slotOf(r.stream) catch continue;
                const rows = r.rows orelse continue;
                if (slot == k.slot and rows.len == k.kept and r.start == k.start) break true;
            } else false;
            if (!moot) return false;
        }
        return true;
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .prefill_part = prefillPartFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .probabilities = probabilitiesFn,
            .release = releaseFn,
        } };
    }

    /// Times the rounds and drafts on both ranks (the `measure` step) for the facts' costs: a round of R rows costs
    /// its forward plus, with the drafter, a batched draft of the streams that many rows hold (a drafted round runs
    /// both); a lone window of w rows is a round of w rows.
    pub fn measure(self: *Lanes, reps: u8) !void {
        const e = self.e;
        try self.sendKeeps();
        try e.measure(reps);
        const per = if (e.drafter) |dr| dr.passStreams() else 1;
        for (self.shared_costs[0..e.streams.rows], 0..) |*c, i| {
            const rows = i + 1;
            var ms = e.round_ms[i];
            if (e.drafter != null and rows > 1) {
                var streams = @min((rows + self.width() - 1) / self.width(), e.streams.lanes.len);
                while (streams > 0) : (streams -|= per) ms += e.draft_ms[@min(streams, per) - 1];
            }
            c.* = .{ .width = @intCast(rows), .ms = ms };
        }
        for (self.window_costs[0..self.width()], 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = e.round_ms[i] + if (e.drafter != null and i > 0) e.draft_ms[0] else 0 };
        std.log.info("deepseek_v41: rounds of 1..{d} rows {any} ms at the {d}-position bucket ({d} and {d} rows {any} ms at the {d}-position bucket), batched drafts of 1..{d} streams {any} ms", .{ e.streams.rows, e.round_ms[0..e.streams.rows], rounds.bucket_min, 1, e.streams.rows, e.round_deep_ms, e.streams.runner.bucket, max_slots, e.draft_ms });
    }

    pub fn facts(self: *const Lanes) lanes.Model {
        const timed = self.shared_costs[0].ms > 0;
        const rows = self.e.streams.rows;
        return factsWith(self.e.drafter != null, self.width(), self.e.streams.lanes.len, rows, if (timed) self.window_costs[0..self.width()] else &.{}, if (timed) self.shared_costs[0..rows] else &.{});
    }

    /// The window rows the host sizes the round loop for (`Loaded.rows`): a window's drafts at most and the pending
    /// token (without the drafter, `rowsOf`'s).
    pub fn windowRows(self: *const Lanes) u32 {
        return @intCast(self.width());
    }

    /// A window's rows at most: the drafts (TF_DS_DRAFTS, else the drafter's block) and the pending token.
    pub fn width(self: *const Lanes) usize {
        return if (self.e.drafter != null) self.e.drafts() + 1 else draft_width;
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % self.drawn.len] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) !u32 {
        return switch (feed) {
            .handle => |h| if (h >= self.next or self.next - h > self.drawn.len) error.NoSuchToken else self.drawn[h % self.drawn.len],
            .value => |v| v,
        };
    }

    fn slotOf(self: *const Lanes, s: *const lanes.Stream) !usize {
        for (self.owner, 0..) |o, i| if (o == s) return i;
        return error.UnknownStream;
    }

    /// The stream into the lowest free slot and the extent the window book places for its prompt, its reply and a
    /// round's rows (multi.py `admit`): its prompt fills the slot now, on every rank, from the longest kept boundary
    /// it shares with a kept prompt (`s.cached`, usage's cached tokens) or from 0; its first token is drawn.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        try self.admit(s, false);
        try self.drawFirst();
    }

    /// `prefillFn` in parts: the first call announces the admission (`Admit.parts`), each call then runs the next
    /// `chunks` chunks on every rank (the fill step); the last draws the first token. A cancel is honoured between
    /// parts (the engine releases the stream, freeing its slot on every rank).
    fn prefillPartFn(ptr: *anyopaque, s: *lanes.Stream, chunks: u32) anyerror!bool {
        const self = of(ptr);
        const slot = self.slotOf(s) catch blk: {
            try self.admit(s, true);
            break :blk try self.slotOf(s);
        };
        try self.sendKeeps();
        const e = self.e;
        const pass = e.streams.lanes[slot].pass orelse return error.PositionMismatch;
        try e.both(Fill.encode(e.op_buf, .{ .slot = slot, .chunks = @max(chunks, 1), .at = pass.at }));
        if (e.streams.lanes[slot].pass != null) return false;
        try self.drawFirst();
        return true;
    }

    /// The first token from the prompt's last row (the logits its last chunk left), held for `firstFn`.
    fn drawFirst(self: *Lanes) !void {
        var first: [1]u32 = undefined;
        try self.e.streams.argmaxRows(0, &first);
        _ = self.take(first[0]);
    }

    /// The admission step for `s` (all its chunks, or with `parts` none of them) and its slot owned.
    fn admit(self: *Lanes, s: *lanes.Stream, parts: bool) !void {
        try self.sendKeeps();
        const e = self.e;
        const st = &e.streams;
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotSupported;
        const ptoks = s.prompt();
        const need = ptoks.len + s.max_new + extent_tail;
        if (ptoks.len == 0 or need > st.most) return error.PromptTooLong;
        var slot: ?usize = null;
        for (st.lanes, 0..) |l, i| if (!l.busy and self.owner[i] == null) {
            slot = i;
            break;
        };
        const free = slot orelse return error.NoRoom;
        // the kept prompt this one continues the furthest (multi.py `_match`), and where the book will place it
        const reuse = if (st.keeping(e.settings.replay)) st.book.match(ptoks, e.settings.replay) else null;
        const size = Extents.size(need);
        const placed = st.trialPlace(size, reuse) orelse return error.NoRoom;
        // a cancel is honoured only before the step is announced
        if (s.isCancelled()) return error.Cancelled;
        const ids = e.ids_buf[0..ptoks.len];
        for (ids, ptoks) |*id, t| id.* = t;
        const op = Admit.encode(e.op_buf, .{
            .slot = free, .engram = e.engram != null, .drafts = e.drafter != null, .replay = e.settings.replay, .chunk = e.chunk, .base = placed.base, .size = size,
            .reuse = reuse, .check = st.book.set.check and reuse != null, .keep = st.book.set.on, .parts = parts, .book = st.fingerprint(), .ids = &.{},
        }, ids);
        try e.both(op);
        s.cached = if (reuse) |r| @intCast(r.cut) else 0;
        self.owner[free] = s;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        _ = try self.slotOf(s);
        return self.next - 1;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return of(ptr).value(.{ .handle = handle });
    }

    /// Every window of the round in one forward (the round step): each stream's pending token, then its held drafts
    /// or host tokens, at its length; each row's draw greedy.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        try self.sendKeeps();
        const e = self.e;
        const st = &e.streams;
        if (windows.len > st.lanes.len) return error.WindowTooWide;
        var ids: [round_rows_most]i64 = undefined;
        var rws: [max_slots]RoundWindow = undefined;
        var r: usize = 0;
        for (windows, out, rws[0..windows.len]) |w, o, *rw| {
            const slot = try self.slotOf(w.stream);
            const l = &st.lanes[slot];
            if (w.parents != null) return error.TreesNotBuilt;
            const rows = w.rows();
            if (rows > self.width() or r + rows > st.rows) return error.WindowTooWide;
            for (w.positions, 0..) |p, i| if (p != l.pos + 1 + i) return error.PositionMismatch;
            if (w.held > l.held) return error.NoHeldDrafts;
            const win = ids[r..][0..rows];
            win[0] = w.pending;
            for (win[1..][0..w.held], l.drafts.tokens[0..w.held]) |*id, t| id.* = t;
            for (win[1 + w.held ..][0..w.tokens.len], w.tokens) |*id, t| id.* = t;
            for (o.drafts[0..w.held], l.drafts.tokens[0..w.held]) |*d, t| d.* = t;
            @memcpy(o.drafts[w.held..][0..w.tokens.len], w.tokens);
            rw.* = .{ .slot = slot, .base = l.base, .size = l.size, .position = l.pos, .ids = win };
            r += rows;
            if (self.stats_on) self.offerStats(slot, w);
        }
        if (self.stats_on) self.round_rows[r] += 1;
        try e.both(Round.encode(e.op_buf, rws[0..windows.len]));
        // every window's rows in one draw (one read back), then each window's share
        var sampled: [round_rows_most]u32 = undefined;
        try st.argmaxRows(0, sampled[0..r]);
        r = 0;
        for (windows, out) |w, o| {
            @memcpy(o.sampled[0..w.rows()], sampled[r..][0..w.rows()]);
            r += w.rows();
        }
    }

    /// Each window's kept prefix: the streams not already at it (a draft's absorb keeps too) roll back on every
    /// rank in one keeps step.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        const st = &e.streams;
        var asks: [max_slots]KeepAsk = undefined;
        var n: usize = 0;
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
            const slot = try self.slotOf(w.stream);
            if (self.stats_on) self.stats[slot].land(path.len - 1);
            const l = &st.lanes[slot];
            if (l.pos == l.verify_start + path.len) continue;
            asks[n] = .{ .slot = slot, .start = l.verify_start, .kept = path.len };
            n += 1;
        }
        if (n == 0) return;
        if (!self.defer_keeps) return e.both(Keeps.encode(e.op_buf, asks[0..n]));
        // held for the next step (a stream already holding one, which takes another round between, sends first)
        for (asks[0..n]) |a| for (self.keeps[0..self.n_keeps]) |k| {
            if (k.slot == a.slot) try self.sendKeeps();
        };
        for (asks[0..n]) |a| {
            self.keeps[self.n_keeps] = a;
            self.n_keeps += 1;
        }
    }

    /// Every request in one drafts step: absorb its kept rows (none after the prompt: its chunks absorbed theirs),
    /// then hold `depth` drafts after its pending token, the drafting streams batched.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        const st = &e.streams;
        if (e.drafter == null) return error.NoDraftHead;
        if (requests.len > max_slots) return error.WindowTooWide;
        // the deferred keeps: moot when every kept stream absorbs its kept rows here, else sent before the asks
        // read the streams' lengths
        if (self.keepsMoot(requests)) self.n_keeps = 0 else try self.sendKeeps();
        var asks: [max_slots]DraftAsk = undefined;
        for (requests, asks[0..requests.len]) |r, *a| {
            if (r.lanes != null) return error.TreesNotBuilt;
            const slot = try self.slotOf(r.stream);
            const l = &st.lanes[slot];
            a.* = .{ .slot = slot, .start = l.pos, .absorb = 0, .token = 0, .q0 = l.pos, .depth = @min(r.depth, e.drafts()) };
            if (r.rows) |rows| {
                if (rows.len == 0 or r.follow.len < rows.len) return error.EmptyPath;
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                if (self.stats_on) self.stats[slot].land(rows.len - 1);
                if (r.start != l.verify_start) return error.PositionMismatch;
                a.start = l.verify_start;
                a.absorb = rows.len;
                a.q0 = l.verify_start + rows.len;
                a.token = r.follow[rows.len - 1];
            } else a.token = try self.value(r.first orelse return error.NoFirstToken);
            if (r.position != a.q0 + 1) return error.PositionMismatch;
        }
        try e.both(DraftsRec.encode(e.op_buf, asks[0..requests.len]));
    }

    /// The backend's `probabilities` hook (the round loop's allocation trims each window's held drafts by them, and
    /// with them the depth rule asks for the most drafts and leaves the cut to the allocation): out[j] is the chance
    /// that the stream's held drafts 0 .. j all land, the prefix survivals of the confidence head's logits
    /// (`survivals`). False, so the loop falls back to its per-depth acceptance rates, when the stream holds fewer
    /// than out.len drafts or is not this backend's. The chances choose which rows a round verifies, never a row's
    /// token: every verified row is drawn greedy from the target's logits.
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const slot = self.slotOf(s) catch return false;
        const l = &self.e.streams.lanes[slot];
        if (out.len > l.held or out.len > l.drafts.len) return false;
        survivals(l.drafts.conf[0..out.len], out);
        return true;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const slot = self.slotOf(s) catch return;
        self.owner[slot] = null;
        if (self.stats_on) self.logStats(slot);
        const e = self.e;
        self.sendKeeps() catch |err| std.log.err("release: keeps: {s}", .{@errorName(err)});
        if (!e.streams.lanes[slot].busy or e.broken) return;
        // the kept prompts this stream's prompt covers go when it is kept (multi.py `_covered`)
        var drops: [kept_prompts.max_kept]u32 = undefined;
        const n = e.streams.covered(slot, &drops);
        const op = Free.encode(e.op_buf, slot, e.streams.fingerprint(), drops[0..n]);
        e.both(op) catch |err| std.log.err("release: {s}", .{@errorName(err)});
    }
};

test "TF_DS_DRAFTS: unset, the checkpoint's block; set, that many drafts; the facts follow" {
    try std.testing.expectEqual(@as(usize, 5), draftsFor(5, 0));
    try std.testing.expectEqual(@as(usize, 3), draftsFor(5, 3));
    for (1..most_drafts_max + 1) |m| {
        // a window fits a round
        try std.testing.expect(draftsFor(5, m) + 1 <= max_rows);
        const f = factsWith(true, draftsFor(5, m) + 1, 4, rounds.rows_default, &.{}, &.{});
        try std.testing.expectEqual(@as(u32, @intCast(m)), f.drafts);
        try std.testing.expectEqual(f.drafts + 1, f.exact_width);
        try std.testing.expect(draft_prior.len >= f.drafts);
    }
}

test "draft chances are non-increasing prefix survivals in [0, 1], the products of the confidences' sigmoids" {
    var prng = std.Random.DefaultPrng.init(0xc0f1);
    const r = prng.random();
    var out: [3]f64 = undefined;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.25 / (1.0 + @exp(@as(f64, -2.0))), out[2], 1e-12);
    for (0..500) |_| {
        var conf: [16]f32 = undefined;
        for (&conf) |*c| c.* = (r.float(f32) - 0.5) * 40;
        const n = r.intRangeAtMost(usize, 0, conf.len);
        var p: [16]f64 = undefined;
        survivals(&conf, p[0..n]);
        var prev: f64 = 1.0;
        for (p[0..n], conf[0..n]) |x, c| {
            try std.testing.expect(x >= 0 and x <= prev);
            try std.testing.expectApproxEqRel(prev / (1.0 + @exp(-@as(f64, c))), x, 1e-12);
            prev = x;
        }
    }
}

test "deferred keeps are moot only when every kept stream absorbs exactly its kept rows of the same window" {
    var ss: [3]lanes.Stream = undefined;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .e = undefined, .defer_keeps = true };
    for (&ss, 0..) |*x, i| ln.owner[i] = x;
    const two = [_]u32{ 0, 1 };
    const three = [_]u32{ 0, 1, 2 };
    const absorb2: be.DraftRequest = .{ .stream = &ss[0], .follow = &.{ 7, 8 }, .rows = &two, .start = 40, .position = 43, .depth = 3 };
    const fresh1: be.DraftRequest = .{ .stream = &ss[1], .follow = &.{}, .rows = null, .start = 0, .position = 0, .depth = 3 };
    try std.testing.expect(ln.keepsMoot(&.{absorb2}));
    ln.keeps[0] = .{ .slot = 0, .start = 40, .kept = 2 };
    ln.n_keeps = 1;
    try std.testing.expect(ln.keepsMoot(&.{absorb2}));
    try std.testing.expect(ln.keepsMoot(&.{ fresh1, absorb2 }));
    var other = absorb2;
    other.rows = &three;
    try std.testing.expect(!ln.keepsMoot(&.{other}));
    other = absorb2;
    other.start = 41;
    try std.testing.expect(!ln.keepsMoot(&.{other}));
    try std.testing.expect(!ln.keepsMoot(&.{fresh1}));
    try std.testing.expect(!ln.keepsMoot(&.{}));
    ln.keeps[1] = .{ .slot = 2, .start = 9, .kept = 1 };
    ln.n_keeps = 2;
    try std.testing.expect(!ln.keepsMoot(&.{absorb2}));
}

test "bounded replay starts the upper layers at the prompt's last window rows, at 0 for a prompt within the window" {
    try std.testing.expectEqual(@as(?usize, null), replayFrom(false, 600, 128));
    try std.testing.expectEqual(@as(?usize, 472), replayFrom(true, 600, 128));
    try std.testing.expectEqual(@as(?usize, 0), replayFrom(true, 128, 128));
    try std.testing.expectEqual(@as(?usize, 0), replayFrom(true, 5, 128));
}

test "rope rows align up to 4096 past the context and the window's extra rows" {
    try std.testing.expectEqual(@as(usize, 4096), ropeRows(1024));
    try std.testing.expectEqual(@as(usize, 20480), ropeRows(16384));
    try std.testing.expectEqual(@as(usize, 69632), ropeRows(65536));
}

test "a prefill record parses back to its chunk, Engram, drafter and replay flags and ids; a cut or foreign record is refused" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    var buf: [Prefill.header + 8 * 64]u8 = undefined;
    var ids: [64]i64 = undefined;
    var back: [64]i64 = undefined;
    for (0..200) |_| {
        const n = r.intRangeAtMost(usize, 0, ids.len);
        for (ids[0..n]) |*id| id.* = r.int(i64);
        const chunk = r.intRangeAtMost(usize, 1, prompt.max_chunk);
        const engram = r.boolean();
        const drafts = r.boolean();
        const replay = r.boolean();
        const op = Prefill.encode(&buf, chunk, engram, drafts, replay, ids[0..n]);
        try std.testing.expectEqual(Prefill.header + 8 * n, op.len);
        const p = try Prefill.parse(op);
        try std.testing.expectEqual(chunk, p.chunk);
        try std.testing.expectEqual(replay, p.replay);
        try std.testing.expectEqual(engram, p.engram);
        try std.testing.expectEqual(drafts, p.drafts);
        try std.testing.expectEqualSlices(i64, ids[0..n], p.idsInto(&back));
        if (op.len > 0) try std.testing.expectError(error.BadRecord, Prefill.parse(op[0 .. op.len - 1]));
    }
    var round_op: [Prefill.header]u8 = @splat(0);
    round_op[0] = @intFromEnum(Op.round);
    try std.testing.expectError(error.BadRecord, Prefill.parse(&round_op));
    try std.testing.expect(maxOpBytes(16384) >= Prefill.header + 8 * 16384);
}

test "verify, keep and draft records parse back to what was encoded; cut, widened or foreign records are refused" {
    var prng = std.Random.DefaultPrng.init(0xd5a7);
    const r = prng.random();
    var buf: [Verify.header + 8 * max_rows]u8 = undefined;
    var ids: [max_rows]i64 = undefined;
    var back: [max_rows]i64 = undefined;
    for (0..300) |_| {
        const n = r.intRangeAtMost(usize, 1, max_rows);
        for (ids[0..n]) |*id| id.* = r.int(i64);
        const start = r.int(u32);
        const op = Verify.encode(&buf, start, ids[0..n]);
        const v = try Verify.parse(op);
        try std.testing.expectEqual(@as(usize, start), v.start);
        try std.testing.expectEqualSlices(i64, ids[0..n], v.idsInto(&back));
        try std.testing.expectError(error.BadRecord, Verify.parse(op[0 .. op.len - 1]));
        try std.testing.expectError(error.BadRecord, Keep.parse(op));

        const k: Keep = .{ .start = r.int(u32), .kept = r.intRangeAtMost(usize, 1, draft_width) };
        const kop = Keep.encode(&buf, k);
        try std.testing.expectEqual(k, try Keep.parse(kop));
        try std.testing.expectError(error.BadRecord, Keep.parse(kop[0 .. kop.len - 1]));
        try std.testing.expectError(error.BadRecord, Draft.parse(kop));

        const d: Draft = .{ .start = r.int(u32), .absorb = r.intRangeAtMost(usize, 0, draft_width), .token = r.intRangeAtMost(i64, 0, 129279), .q0 = r.int(u32), .depth = r.intRangeAtMost(usize, 0, draft_width - 1) };
        const dop = Draft.encode(&buf, d);
        try std.testing.expectEqual(d, try Draft.parse(dop));
        try std.testing.expectError(error.BadRecord, Draft.parse(dop[0 .. dop.len - 1]));
        try std.testing.expectError(error.BadRecord, Verify.parse(dop));
    }
    // a window wider than the decode loop's rows never parses
    var wide: [Verify.header + 8 * (max_rows + 1)]u8 = @splat(0);
    wide[0] = @intFromEnum(Op.verify);
    wide[1] = max_rows + 1;
    try std.testing.expectError(error.BadRecord, Verify.parse(&wide));
}

test "every layer past the first is tapped once a window, after its Engram gate when it has one" {
    var nb: [32]u8 = undefined;
    const layers = 40;
    for ([_]u64{ 0, 1 << 2 | 1 << 38, ~@as(u64, 0) }) |gated| {
        var seen: [layers]u32 = @splat(0);
        for (0..layers) |li| {
            if (gated & (@as(u64, 1) << @intCast(li)) != 0) if (Sequence.tapLayer(try std.fmt.bufPrint(&nb, "L{d}.h_engram", .{li}), layers, gated)) |t| {
                try std.testing.expectEqual(li, t);
                seen[t] += 1;
            };
            for ([_][]const u8{ "x_attn", "p_attn.r0", "h_attn", "x_ffn", "p_moe.r0" }) |other|
                try std.testing.expectEqual(@as(?usize, null), Sequence.tapLayer(try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ li, other }), layers, gated));
            if (Sequence.tapLayer(try std.fmt.bufPrint(&nb, "L{d}.h_moe", .{li}), layers, gated)) |t| seen[t] += 1;
        }
        // layer 0's input is the embedding: only its own Engram gate shows it
        try std.testing.expectEqual(@as(u32, @intCast(gated & 1)), seen[0]);
        for (seen[1..]) |c| try std.testing.expectEqual(@as(u32, 1), c);
        try std.testing.expectEqual(@as(?usize, null), Sequence.tapLayer("head.xc", layers, gated));
    }
}

test "extents: first fit, taken extents never overlap, freed extents merge back into one window" {
    var prng = std.Random.DefaultPrng.init(0xe57e);
    const r = prng.random();
    const total = 16 * extent_align;
    for (0..200) |_| {
        var ex = Extents.init(total + r.intRangeLessThan(usize, 0, extent_align)); // the tail below an alignment is unused
        try std.testing.expectEqual(total, ex.total);
        var used: [16]bool = @splat(false); // the model: each aligned block taken or free
        var held: [max_slots][2]usize = undefined;
        var n: usize = 0;
        for (0..60) |_| {
            if (n < max_slots and r.boolean()) {
                const rows = r.intRangeAtMost(usize, 1, 6 * extent_align);
                const blocks = Extents.size(rows) / extent_align;
                // the model's first fit
                var want: ?usize = null;
                var run: usize = 0;
                for (used, 0..) |u, i| {
                    run = if (u) 0 else run + 1;
                    if (run == blocks) {
                        want = (i + 1 - blocks) * extent_align;
                        break;
                    }
                }
                const got = ex.first(rows);
                try std.testing.expectEqual(want, got);
                if (got) |start| {
                    try std.testing.expect(ex.takeAt(start, rows));
                    try std.testing.expect(!ex.takeAt(start, rows)); // taken: never twice
                    for (used[start / extent_align ..][0..blocks]) |*u| u.* = true;
                    held[n] = .{ start, rows };
                    n += 1;
                }
            } else if (n > 0) {
                const i = r.intRangeLessThan(usize, 0, n);
                ex.give(held[i][0], held[i][1]);
                for (used[held[i][0] / extent_align ..][0 .. Extents.size(held[i][1]) / extent_align]) |*u| u.* = false;
                held[i] = held[n - 1];
                n -= 1;
            }
            // the gaps are exactly the model's free runs: sorted, disjoint, never touching
            var free_blocks: usize = 0;
            for (ex.gaps[0..ex.n], 0..) |g, gi| {
                try std.testing.expect(g[0] < g[1] and g[0] % extent_align == 0 and g[1] % extent_align == 0);
                if (gi > 0) try std.testing.expect(ex.gaps[gi - 1][1] < g[0]);
                for (used[g[0] / extent_align .. g[1] / extent_align]) |u| try std.testing.expect(!u);
                free_blocks += (g[1] - g[0]) / extent_align;
            }
            var model_free: usize = 0;
            for (used) |u| model_free += @intFromBool(!u);
            try std.testing.expectEqual(model_free, free_blocks);
        }
        for (held[0..n]) |h| ex.give(h[0], h[1]);
        try std.testing.expectEqual(@as(usize, 1), ex.n);
        try std.testing.expectEqual([2]usize{ 0, total }, ex.gaps[0]);
    }
    // an unaligned start is refused
    var ex = Extents.init(total);
    try std.testing.expect(!ex.takeAt(1, 10));
}

test "admission, free, fill, round, drafts, keeps and measure records parse back to what was encoded; cut or foreign ones are refused" {
    var prng = std.Random.DefaultPrng.init(0x9a7a);
    const r = prng.random();
    var buf: [Admit.header + 8 * 64]u8 = undefined;
    var ids: [64]i64 = undefined;
    var back: [64]i64 = undefined;
    for (0..300) |_| {
        const n = r.intRangeAtMost(usize, 0, ids.len);
        for (ids[0..n]) |*id| id.* = r.int(i64);
        const reuse: ?kept_prompts.Reuse = if (r.boolean()) .{ .eid = r.int(u32), .cut = r.int(u40) } else null;
        const a: Admit = .{ .slot = r.intRangeLessThan(usize, 0, max_slots), .engram = r.boolean(), .drafts = r.boolean(), .replay = r.boolean(), .chunk = r.intRangeAtMost(usize, 1, prompt.max_chunk), .base = r.int(u40), .size = r.int(u40), .reuse = reuse, .check = r.boolean(), .keep = r.boolean(), .parts = r.boolean(), .book = r.int(u64), .ids = &.{} };
        const op = Admit.encode(&buf, a, ids[0..n]);
        const p = try Admit.parse(op);
        try std.testing.expect(p.slot == a.slot and p.engram == a.engram and p.drafts == a.drafts and p.replay == a.replay and p.chunk == a.chunk and p.base == a.base and p.size == a.size);
        try std.testing.expect(std.meta.eql(p.reuse, a.reuse) and p.check == a.check and p.keep == a.keep and p.parts == a.parts and p.book == a.book);

        const fill: Fill = .{ .slot = a.slot, .chunks = r.intRangeAtMost(u32, 1, std.math.maxInt(u32)), .at = r.int(u40) };
        var lbuf: [Fill.bytes]u8 = undefined;
        const lop = Fill.encode(&lbuf, fill);
        try std.testing.expect(std.meta.eql(fill, try Fill.parse(lop)));
        try std.testing.expectError(error.BadRecord, Fill.parse(lop[0 .. lop.len - 1]));
        try std.testing.expectError(error.BadRecord, Admit.parse(lop));
        var none = lbuf;
        std.mem.writeInt(u32, none[2..6], 0, .little);
        try std.testing.expectError(error.BadRecord, Fill.parse(&none)); // a fill of no chunks

        var drops: [kept_prompts.max_kept]u32 = undefined;
        const nd = r.intRangeAtMost(usize, 0, drops.len);
        for (drops[0..nd]) |*d| d.* = r.int(u32);
        var fbuf: [Free.header + 4 * kept_prompts.max_kept]u8 = undefined;
        const fop = Free.encode(&fbuf, a.slot, a.book, drops[0..nd]);
        const f = try Free.parse(fop);
        try std.testing.expect(f.slot == a.slot and f.book == a.book);
        try std.testing.expectEqualSlices(u32, drops[0..nd], f.drops[0..f.n]);
        try std.testing.expectError(error.BadRecord, Free.parse(fop[0 .. fop.len - 1]));
        try std.testing.expectError(error.BadRecord, Admit.parse(fop));
        try std.testing.expectEqualSlices(i64, ids[0..n], p.idsInto(&back));
        try std.testing.expectError(error.BadRecord, Admit.parse(op[0 .. op.len - 1]));
        try std.testing.expectError(error.BadRecord, Prefill.parse(op));

        // a round: 1 .. max_slots windows of 1 .. rows, round_rows_most rows in all
        var windows: [max_slots]RoundWindow = undefined;
        const count = r.intRangeAtMost(usize, 1, max_slots);
        var left: usize = round_rows_most;
        var at: usize = 0;
        for (windows[0..count], 0..) |*w, i| {
            const rows = r.intRangeAtMost(usize, 1, left - (count - 1 - i));
            left -= rows;
            w.* = .{ .slot = r.intRangeLessThan(usize, 0, max_slots), .base = r.int(u40), .size = r.int(u40), .position = r.int(u40), .ids = ids[at..][0..rows] };
            at += rows;
        }
        const rop = Round.encode(&buf, windows[0..count]);
        var rr: Round = undefined;
        try Round.parse(rop, &rr);
        try std.testing.expectEqual(count, rr.count);
        for (windows[0..count], rr.windows[0..count]) |w, g| {
            try std.testing.expect(w.slot == g.slot and w.base == g.base and w.size == g.size and w.position == g.position);
            try std.testing.expectEqualSlices(i64, w.ids, g.ids);
        }
        try std.testing.expectError(error.BadRecord, Round.parse(rop[0 .. rop.len - 1], &rr));

        var asks: [max_slots]DraftAsk = undefined;
        for (asks[0..count]) |*d| d.* = .{ .slot = r.intRangeLessThan(usize, 0, max_slots), .start = r.int(u40), .absorb = r.intRangeAtMost(usize, 0, draft_width), .token = r.intRangeAtMost(i64, 0, 129279), .q0 = r.int(u40), .depth = r.intRangeAtMost(usize, 0, draft_width - 1) };
        const dop = DraftsRec.encode(&buf, asks[0..count]);
        var dr: DraftsRec = undefined;
        try DraftsRec.parse(dop, &dr);
        try std.testing.expectEqualSlices(DraftAsk, asks[0..count], dr.asks[0..dr.count]);
        try std.testing.expectError(error.BadRecord, DraftsRec.parse(dop[0 .. dop.len - 1], &dr));
        try std.testing.expectError(error.BadRecord, Round.parse(dop, &rr));

        var keeps: [max_slots]KeepAsk = undefined;
        for (keeps[0..count]) |*k| k.* = .{ .slot = r.intRangeLessThan(usize, 0, max_slots), .start = r.int(u40), .kept = r.intRangeAtMost(usize, 1, draft_width) };
        const kop = Keeps.encode(&buf, keeps[0..count]);
        var kr: Keeps = undefined;
        try Keeps.parse(kop, &kr);
        try std.testing.expectEqualSlices(KeepAsk, keeps[0..count], kr.asks[0..kr.count]);
        try std.testing.expectError(error.BadRecord, Keeps.parse(kop[0 .. kop.len - 1], &kr));
        try std.testing.expectError(error.BadRecord, DraftsRec.parse(kop, &dr));

        const m: Measure = .{ .reps = r.int(u8), .slots = r.intRangeAtMost(usize, 1, max_slots), .engram = r.boolean(), .drafts = r.boolean(), .total = r.int(u40), .most = r.int(u40), .width = r.intRangeAtMost(usize, 0, most_drafts_max), .at = r.int(u40), .rows = r.intRangeAtMost(usize, 1, round_rows_most) };
        const mop = Measure.encode(&buf, m);
        try std.testing.expectEqual(m, try Measure.parse(mop));
        try std.testing.expectError(error.BadRecord, Measure.parse(mop[0 .. mop.len - 1]));
    }
    // a round wider than round_rows_most rows never parses
    var wide: [2 + 26 + 8 * (round_rows_most + 1)]u8 = @splat(0);
    wide[0] = @intFromEnum(Op.rounds);
    wide[1] = 1;
    wide[3] = round_rows_most + 1;
    var rr: Round = undefined;
    try std.testing.expectError(error.BadRecord, Round.parse(&wide, &rr));
}
