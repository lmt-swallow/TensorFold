//! DeepSeek-V4.1 behind the lane core on two ranks, up to `max_slots` streams at once, greedy: rank 0 serves and
//! sends every step (an admission, a round, its drafts and keeps, a release) to rank 1 over the communicator's
//! control link before running it; rank 1 runs the same steps in `follow`, so both ranks issue the same
//! all-gathers in the same order (model.py's lockstep). A sha256 digest of each step is all-gathered first
//! (multi.py `_agree`): ranks that planned different steps, or a rank that cannot run one, stop instead of pairing
//! collectives wrongly.
//!
//! The streams share one window of `context + extra` positions (multi.py MultiDecoder over model.py's PoolCache,
//! `Streams`): a stream takes the lowest free slot (its window rings, compressor inputs and drafter rings) and the
//! first free extent of the window aligned to 2048 that holds its prompt, its reply and a round's rows (`Extents`).
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
const RankCache = @import("rank_cache.zig").RankCache;

const be = lanes.backend;
const Ptr = cuda.abi.DevicePtr;

pub const max_rows: usize = 16;
/// Positions past the context a sequence's RoPE tables and caches hold (engine.py: limit + max_rows + 8).
pub const extra: usize = max_rows + 8;
pub const rope_align: usize = 4096;

pub const Error = error{ WorldNotTwo, BadMaster, NoRankCache, NoRopeTables, RopeRowsShort, OutOfStep, LinkClosed, SamplingNotSupported, WindowTooWide, PositionMismatch, UnknownStream, StreamBusy, PromptTooLong, Cancelled, NoDraftHead, NotPipelined, TreesNotBuilt, EmptyPath, BadChunk, BadRecord, NoHeldDrafts, NoFirstToken, BadTapLayers, BadSlots, NoRoom };

/// A drafting round's widest window: the pending token and DSpark's block of drafts (engine.py: drafts + 1 rows).
pub const draft_width: usize = 6;

/// The DSpark head's acceptance at depth 1, 2, ... given the ones before it, until a stream measures its own (the
/// fork's served acceptance is about 0.75 a depth).
pub const draft_prior = [_]f64{ 0.75, 0.75, 0.75, 0.75, 0.75 };

/// Prompt rows a chunk unless TF_DS_PREFILL_CHUNK says otherwise (the fork's production setting).
pub const default_chunk: usize = prompt.max_chunk;

/// Step opcodes. The served engine sends `measure` and `admit` .. `free`; the others are `Sequence`'s records.
pub const Op = enum(u8) { prefill = 1, round = 2, release = 3, stop = 4, verify = 5, keep = 6, draft = 7, measure = 8, admit = 9, rounds = 10, drafts = 11, keeps = 12, free = 13 };

/// The bytes a step's record takes at most for a context of `context` positions (a prompt of the whole context).
pub fn maxOpBytes(context: usize) usize {
    return @max(Prefill.header, Admit.header) + 8 * (context + extra);
}

/// The prefill step's record (little-endian): the opcode, the chunk rows (u32), the flags (u8: bit 0, Engram on;
/// bit 1, DSpark taps absorbed), the prompt's length (u32), then its ids (i64 each). The record is what the ranks'
/// digests agree on, so ranks that would chunk differently or disagree on Engram or the drafter (and so pair
/// different gathers) stop before the first gather.
pub const Prefill = struct {
    chunk: usize,
    engram: bool,
    drafts: bool,
    ids: []const u8, // the ids' little-endian bytes, 8 a token

    pub const header: usize = 10;

    pub fn encode(buf: []u8, chunk: usize, engram: bool, drafts: bool, ids: []const i64) []const u8 {
        buf[0] = @intFromEnum(Op.prefill);
        std.mem.writeInt(u32, buf[1..5], @intCast(chunk), .little);
        buf[5] = @as(u8, @intFromBool(engram)) | @as(u8, @intFromBool(drafts)) << 1;
        std.mem.writeInt(u32, buf[6..10], @intCast(ids.len), .little);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, buf[header + 8 * i ..][0..8], id, .little);
        return buf[0 .. header + 8 * ids.len];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Prefill {
        if (op.len < header or op[0] != @intFromEnum(Op.prefill) or op[5] > 3) return error.BadRecord;
        const n = std.mem.readInt(u32, op[6..10], .little);
        if (op.len != header + 8 * @as(usize, n)) return error.BadRecord;
        return .{ .chunk = std.mem.readInt(u32, op[1..5], .little), .engram = op[5] & 1 != 0, .drafts = op[5] & 2 != 0, .ids = op[header..] };
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
        try seq.prefill(gather, p.idsInto(ids_buf), p.chunk);
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
};

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
    /// m.taps_start)`). Runs every chunk once started: the other ranks cannot see a cancel, so stopping early on one
    /// rank would leave the others in an unpaired gather.
    pub fn prefill(self: *Sequence, gather: forward.Gather, ids: []const i64, chunk: usize) !void {
        if (chunk < 1 or chunk > self.pr.rows) return error.BadChunk;
        if (ids.len == 0 or ids.len + 1 > self.limit()) return error.PromptTooLong;
        self.pos = 0;
        self.logits = 0;
        self.held = 0;
        self.verify_rows = 0;
        for (self.hist[0..ids.len], ids) |*hv, id| hv.* = @intCast(id);
        if (self.engram) |g| g.history = self.hist[0..ids.len];
        if (self.dc) |*c| c.absorbed = 0;
        var s: usize = 0;
        while (s < ids.len) {
            const e = @min(ids.len, s + chunk);
            const last = e == ids.len;
            // A chunk of max_rows or fewer is a decode window (it runs the head): Prompt.chunk sends it through
            // `Forward.rows` only when it asks for logits; with taps it goes there through `rowsTapped`.
            const out = if (self.spark != null and e - s <= max_rows)
                try self.rowsTapped(gather, ids[s..e], s, false)
            else
                try self.pr.chunk(gather, ids[s..e], s, .{
                    .logits = last or e - s <= max_rows,
                    .taps = if (self.spark != null) .{ .buf = self.taps.ptr, .stride = self.tapCols() } else null,
                }, self.stream);
            if (self.spark) |sp| try sp.absorb(&self.dc.?, self.taps.ptr, s, e - s, self.draftRope(), self.stream);
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
pub const max_slots: usize = 4;
/// Extents start and end on multiples of this many positions (multi.py ALIGN).
pub const extent_align: usize = 2048;
/// Positions an extent holds past its stream's prompt and reply (multi.py `_need`: a round's rows, 8, and the
/// scratch row's 4).
pub const extent_tail: usize = max_rows + 8 + 4;
/// Streams a batched draft takes at most: their blocks' rows fit the drafter's 16 (dspark.Drafter.max_rows).
pub fn draftBatch(block: usize) usize {
    return @max(1, dspark.Drafter.max_rows / @max(1, block));
}

/// multi.py `Extents`: first-fit extents of a window of `total` positions in `extent_align` steps; a freed extent
/// merges with its neighbours. The gaps are free, disjoint, sorted, and never touch.
pub const Extents = struct {
    total: usize,
    gaps: [max_slots + 2][2]usize = undefined,
    n: usize = 0,

    pub fn init(positions: usize) Extents {
        var e: Extents = .{ .total = positions / extent_align * extent_align };
        if (e.total > 0) {
            e.gaps[0] = .{ 0, e.total };
            e.n = 1;
        }
        return e;
    }

    /// `rows` aligned up to `extent_align`.
    pub fn size(rows: usize) usize {
        return (rows + extent_align - 1) / extent_align * extent_align;
    }

    /// The start of the first gap that holds `rows` (aligned up), nothing taken; null when none does.
    pub fn first(self: *const Extents, rows: usize) ?usize {
        const n = size(rows);
        for (self.gaps[0..self.n]) |g| if (g[1] - g[0] >= n) return g[0];
        return null;
    }

    /// `rows` (aligned up) from `start` if they are free (and `start` is aligned); false leaves the extents as
    /// they were.
    pub fn takeAt(self: *Extents, start: usize, rows: usize) bool {
        const n = size(rows);
        if (n == 0 or start % extent_align != 0) return false;
        for (self.gaps[0..self.n], 0..) |g, i| {
            if (g[0] > start or start + n > g[1]) continue;
            const left = start > g[0];
            const right = start + n < g[1];
            if (left and right) {
                if (self.n == self.gaps.len) return false;
                std.mem.copyBackwards([2]usize, self.gaps[i + 2 .. self.n + 1], self.gaps[i + 1 .. self.n]);
                self.gaps[i] = .{ g[0], start };
                self.gaps[i + 1] = .{ start + n, g[1] };
                self.n += 1;
            } else if (left) {
                self.gaps[i] = .{ g[0], start };
            } else if (right) {
                self.gaps[i] = .{ start + n, g[1] };
            } else {
                std.mem.copyForwards([2]usize, self.gaps[i .. self.n - 1], self.gaps[i + 1 .. self.n]);
                self.n -= 1;
            }
            return true;
        }
        return false;
    }

    /// Positions [start, start + size(rows)) are free again (they must have been taken).
    pub fn give(self: *Extents, start: usize, rows: usize) void {
        const end = start + size(rows);
        var i: usize = 0;
        while (i < self.n and self.gaps[i][0] < start) i += 1;
        const joins_left = i > 0 and self.gaps[i - 1][1] == start;
        const joins_right = i < self.n and self.gaps[i][0] == end;
        if (joins_left and joins_right) {
            self.gaps[i - 1][1] = self.gaps[i][1];
            std.mem.copyForwards([2]usize, self.gaps[i .. self.n - 1], self.gaps[i + 1 .. self.n]);
            self.n -= 1;
        } else if (joins_left) {
            self.gaps[i - 1][1] = end;
        } else if (joins_right) {
            self.gaps[i][0] = start;
        } else {
            std.mem.copyBackwards([2]usize, self.gaps[i + 1 .. self.n + 1], self.gaps[i..self.n]);
            self.gaps[i] = .{ start, end };
            self.n += 1;
        }
    }
};

/// The admission step's record: the opcode, the slot (u8), the flags (u8: bit 0 Engram on, bit 1 DSpark taps
/// absorbed), the chunk rows (u32), the extent's base and size (u64 each), the prompt's length (u32), its ids (i64
/// each). Every rank fills the slot's extent from position 0 through the prompt path in the record's chunks.
pub const Admit = struct {
    slot: usize,
    engram: bool,
    drafts: bool,
    chunk: usize,
    base: usize,
    size: usize,
    ids: []const u8,

    pub const header: usize = 27;

    pub fn encode(buf: []u8, a: Admit, ids: []const i64) []const u8 {
        buf[0] = @intFromEnum(Op.admit);
        buf[1] = @intCast(a.slot);
        buf[2] = @as(u8, @intFromBool(a.engram)) | @as(u8, @intFromBool(a.drafts)) << 1;
        std.mem.writeInt(u32, buf[3..7], @intCast(a.chunk), .little);
        std.mem.writeInt(u64, buf[7..15], @intCast(a.base), .little);
        std.mem.writeInt(u64, buf[15..23], @intCast(a.size), .little);
        std.mem.writeInt(u32, buf[23..27], @intCast(ids.len), .little);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, buf[header + 8 * i ..][0..8], id, .little);
        return buf[0 .. header + 8 * ids.len];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Admit {
        if (op.len < header or op[0] != @intFromEnum(Op.admit) or op[2] > 3) return error.BadRecord;
        const n = std.mem.readInt(u32, op[23..27], .little);
        if (op.len != header + 8 * @as(usize, n)) return error.BadRecord;
        return .{
            .slot = op[1],
            .engram = op[2] & 1 != 0,
            .drafts = op[2] & 2 != 0,
            .chunk = std.mem.readInt(u32, op[3..7], .little),
            .base = @intCast(std.mem.readInt(u64, op[7..15], .little)),
            .size = @intCast(std.mem.readInt(u64, op[15..23], .little)),
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

/// One stream's window in a round record: its slot, its extent, the first row's position (in the extent) and its
/// ids (the pending token, then the drafts).
pub const RoundWindow = struct { slot: usize, base: usize, size: usize, position: usize, ids: []const i64 };

/// The round step's record: the opcode, the windows (u8), per window its slot (u8), rows (u8), position, base and
/// size (u64 each), then every window's ids in window order (i64 each). One forward over all of them.
pub const Round = struct {
    windows: [max_slots]RoundWindow,
    count: usize,
    ids: [max_rows]i64,

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
    /// `max_rows` rows.
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
        if (rows > max_rows or op.len != 2 + per * count + 8 * rows) return error.BadRecord;
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
/// bit 1 the drafter) and the window's positions (u64): ranks whose pools differ stop at load.
pub const Measure = struct {
    reps: u8,
    slots: usize,
    engram: bool,
    drafts: bool,
    total: usize,

    pub const bytes: usize = 12;

    pub fn encode(buf: []u8, m: Measure) []const u8 {
        buf[0] = @intFromEnum(Op.measure);
        buf[1] = m.reps;
        buf[2] = @intCast(m.slots);
        buf[3] = @as(u8, @intFromBool(m.engram)) | @as(u8, @intFromBool(m.drafts)) << 1;
        std.mem.writeInt(u64, buf[4..12], @intCast(m.total), .little);
        return buf[0..bytes];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Measure {
        if (op.len != bytes or op[0] != @intFromEnum(Op.measure) or op[3] > 3) return error.BadRecord;
        return .{ .reps = op[1], .slots = op[2], .engram = op[3] & 1 != 0, .drafts = op[3] & 2 != 0, .total = @intCast(std.mem.readInt(u64, op[4..12], .little)) };
    }
};

/// A stream's state in its slot: its extent of the window, the position its next row takes (relative to the
/// extent), its host ids from position 0 (Engram's hashes), the last round's window (its start, rows and first row
/// among the round's), the positions its drafter rings absorbed, its last drafts (the first `held` of them held).
pub const Lane = struct {
    busy: bool = false,
    base: usize = 0,
    size: usize = 0,
    pos: usize = 0,
    hist: []i32 = &.{},
    verify_start: usize = 0,
    verify_rows: usize = 0,
    row0: usize = 0,
    absorbed: usize = 0,
    drafts: dspark.Drafts = undefined,
    held: usize = 0,

    /// The positions a window may reach: the extent less its tail (the round's rows, 8, and the scratch row's).
    pub fn limit(self: *const Lane) usize {
        return self.size -| (8 + 4);
    }
};

/// model.py `pool_view`'s single-stream paths and rounds.py's concurrent round over one pool (multi.py
/// MultiDecoder without its scheduler): `slots` streams share one window of `cap` positions by extents. An
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
    stream: cuda.Stream,

    /// `base`: the decode loop whose weights, kernels, scratch and RoPE tables the view and the round share (its
    /// own caches go unused). `rows`: the widest prompt chunk. `cap`: the window's positions (the context and the
    /// window's extra rows).
    pub fn init(self: *Streams, gpa: std.mem.Allocator, d: *const cuda.Driver, base: *forward.Forward, kernels: *const exl3.Kernels, engram: ?*engram_apply.Engram, rows: usize, stream: cuda.Stream, spark: ?*dspark.Drafter, slots: usize, cap: usize) !void {
        const cfg = base.cfg;
        const L = base.w.layers.len;
        if (slots < 1 or slots > max_slots) return error.BadSlots;
        if (spark != null) for (cfg.dspark_taps.slice()) |t| {
            if (t < 1 or t >= @as(i64, @intCast(L))) return error.BadTapLayers;
        };
        self.* = .{
            .gpa = gpa, .d = d, .cfg = cfg, .base = base, .view = base.*, .view_rings = &.{}, .view_planes = &.{}, .pool = undefined,
            .runner = undefined, .pr = undefined, .extents = Extents.init(cap), .engram = engram, .spark = spark, .dpool = null,
            .dc_rings = &.{}, .taps = .{ .d = d, .ptr = 0, .len = 0 }, .kept_taps = .{ .d = d, .ptr = 0, .len = 0 }, .lanes = &.{}, .logits_h = &.{}, .stream = stream,
        };
        self.pool = try rounds.Pool.init(gpa, d, cfg, L, slots, cap);
        errdefer self.pool.deinit();
        self.view_rings = try gpa.alloc(cuda.DeviceBuffer, L);
        errdefer gpa.free(self.view_rings);
        self.view_planes = try gpa.alloc(?forward.Planes, L);
        errdefer gpa.free(self.view_planes);
        // the view first spans the whole window (slot 0): the prompt buffers are sized for its capacity
        self.point(0, 0, self.extents.total);
        self.view.cap = cap;
        self.view.rings = self.view_rings;
        self.view.planes = self.view_planes;
        self.view.source = self.pool.source;
        self.view.probe = null;
        self.view.engram = if (engram) |g| g.hook() else null;
        self.pr = try prompt.Prompt.init(gpa, d, &self.view, kernels, rows);
        errdefer self.pr.deinit();
        self.runner = try rounds.RoundRunner.init(gpa, d, cfg, base.w, base.set, base.glue, base.ek, base.world, base.bucket, base.rope_c, base.rope_w, &self.pool);
        errdefer self.runner.deinit();
        if (engram) |g| self.runner.engram = g.roundHook();
        self.lanes = try gpa.alloc(Lane, slots);
        errdefer gpa.free(self.lanes);
        var made: usize = 0;
        errdefer for (self.lanes[0..made]) |l| gpa.free(l.hist);
        for (self.lanes) |*l| {
            l.* = .{ .hist = try gpa.alloc(i32, cap) };
            made += 1;
        }
        self.logits_h = try gpa.alloc(f32, max_rows * cfg.vocab);
        errdefer gpa.free(self.logits_h);
        if (spark) |sp| {
            self.taps = try cuda.DeviceBuffer.alloc(d, @max(rows, max_rows) * self.tapCols() * 2);
            errdefer self.taps.free();
            self.kept_taps = try cuda.DeviceBuffer.alloc(d, max_rows * self.tapCols() * 2);
            errdefer self.kept_taps.free();
            self.dpool = try dspark.DraftPool.init(gpa, d, sp.stages.len, slots, sp.dims);
            errdefer self.dpool.?.deinit(gpa);
            self.dc_rings = try gpa.alloc(cuda.DeviceBuffer, sp.stages.len);
        }
    }

    pub fn deinit(self: *Streams) void {
        const gpa = self.gpa;
        if (self.dc_rings.len > 0) gpa.free(self.dc_rings);
        if (self.dpool) |*p| p.deinit(gpa);
        if (self.taps.len > 0) self.taps.free();
        if (self.kept_taps.len > 0) self.kept_taps.free();
        gpa.free(self.logits_h);
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
    /// extent free and aligned, the prompt and a round's rows inside it.
    pub fn admits(self: *const Streams, slot: usize, base: usize, size: usize, n: usize, chunk: usize) bool {
        if (slot >= self.lanes.len or self.lanes[slot].busy or chunk < 1 or chunk > self.pr.rows) return false;
        if (n < 1 or size % extent_align != 0 or n + extent_tail > size) return false;
        var probe = self.extents;
        return probe.takeAt(base, size);
    }

    /// multi.py `_admit` + `_fill` to the end (engine.py prefill_steps from 0 on the slot's pool view): the slot and
    /// extent taken, the prompt's chunks of `chunk` rows through the prompt path, each chunk's taps absorbed into the
    /// slot's drafter rings; the last row's logits in `logits`. Runs every chunk once started (the other ranks
    /// cannot see a cancel).
    pub fn admit(self: *Streams, gather: forward.Gather, slot: usize, base: usize, size: usize, ids: []const i64, chunk: usize) !void {
        if (!self.admits(slot, base, size, ids.len, chunk)) return error.PositionMismatch;
        if (!self.extents.takeAt(base, size)) return error.PositionMismatch;
        const lane = &self.lanes[slot];
        lane.* = .{ .busy = true, .base = base, .size = size, .hist = lane.hist };
        errdefer {
            self.extents.give(base, size);
            lane.busy = false;
        }
        for (lane.hist[0..ids.len], ids) |*hv, id| hv.* = @intCast(id);
        if (self.engram) |g| g.history = lane.hist[0..ids.len];
        self.point(slot, base, size);
        self.logits = 0;
        var dc: ?dspark.Cache = if (self.spark != null) self.slotCache(slot) else null;
        var s: usize = 0;
        while (s < ids.len) {
            const e = @min(ids.len, s + chunk);
            const last = e == ids.len;
            const out = if (self.spark != null and e - s <= max_rows)
                try self.rowsTapped(gather, ids[s..e], s)
            else
                try self.pr.chunk(gather, ids[s..e], s, .{
                    .logits = last or e - s <= max_rows,
                    .taps = if (self.spark != null) .{ .buf = self.taps.ptr, .stride = self.tapCols() } else null,
                }, self.stream);
            if (self.spark) |sp| try sp.absorb(&dc.?, self.taps.ptr, s, e - s, self.draftRope(), self.stream);
            if (last) self.logits = out orelse return error.BadChunk;
            s = e;
        }
        lane.pos = ids.len;
        lane.absorbed = if (dc) |c| c.absorbed else 0;
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
    /// extent, inside its limit, `max_rows` rows at most.
    pub fn roundable(self: *const Streams, windows: []const RoundWindow) bool {
        var rows: usize = 0;
        var seen: u64 = 0;
        for (windows) |w| {
            if (w.slot >= self.lanes.len or seen & (@as(u64, 1) << @intCast(w.slot)) != 0) return false;
            seen |= @as(u64, 1) << @intCast(w.slot);
            const l = &self.lanes[w.slot];
            if (!l.busy or w.base != l.base or w.size != l.size or w.position != l.pos or w.ids.len < 1) return false;
            if (w.position + w.ids.len > l.limit()) return false;
            rows += w.ids.len;
        }
        return windows.len >= 1 and rows <= max_rows;
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
    /// `BatchDraftGraph`, at most `draftBatch` streams a pass), each stream holding its first `depth` drafts.
    pub fn draft(self: *Streams, asks: []const DraftAsk) !void {
        if (!self.draftable(asks)) return error.PositionMismatch;
        const sp = self.spark.?;
        const dp = &self.dpool.?;
        const rope = self.draftRope();
        const row_bytes = self.tapCols() * 2;
        var pos: [max_rows]usize = undefined;
        var slot_of: [max_rows]usize = undefined;
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
        const per = draftBatch(sp.dims.block);
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

    /// The stream in `slot` leaves: its extent free again (multi.py `_release`).
    pub fn free(self: *Streams, slot: usize) void {
        if (slot >= self.lanes.len or !self.lanes[slot].busy) return;
        const l = &self.lanes[slot];
        self.extents.give(l.base, l.size);
        l.busy = false;
        l.held = 0;
    }

    /// Whether no stream holds a slot.
    pub fn idle(self: *const Streams) bool {
        for (self.lanes) |l| if (l.busy) return false;
        return true;
    }

    /// The first `out.len` rows of `logits` (from row `row0`): each row's argmax (torch's rule: the first index of
    /// the maximum).
    pub fn argmaxRows(self: *Streams, row0: usize, out: []u32) !void {
        const v = self.cfg.vocab;
        if (out.len > max_rows) return error.WindowTooWide;
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

    /// The timing step's round: `rows` noise ids at position 0 of slot 0 over the whole window (no stream's state
    /// changes; every slot must be free: a later admission rewrites every row a later row reads). It scans the
    /// least bucket.
    pub fn probeRound(self: *Streams, gather: forward.Gather, rows: usize) !void {
        try self.probeRoundAt(gather, rows, 0);
    }

    /// The deepest position a probe round of `rows` rows can take whose bucket is the runner's widest (or the
    /// window's end, for a window narrower than it).
    pub fn probeDeep(self: *const Streams, rows: usize) usize {
        const rope_rows = @min(self.runner.rope_c.rows, self.runner.rope_w.rows);
        return @min(@min(self.extents.total, rope_rows), self.runner.bucket) -| rows;
    }

    /// `probeRound` with its rows at `position` .. `position + rows - 1`: the round scans their bucket.
    pub fn probeRoundAt(self: *Streams, gather: forward.Gather, rows: usize, position: usize) !void {
        const noise: i32 = @intCast(self.cfg.dspark_noise);
        var ids: [max_rows]i64 = @splat(noise);
        const w = [1]rounds.Window{.{ .slot = 0, .base = 0, .size = self.extents.total, .position = position, .ids = ids[0..rows] }};
        if (self.engram != null) {
            // the rows' n-grams reach before them: noise there too (the history is otherwise unset)
            @memset(self.lanes[0].hist[position -| 64 .. position + rows], noise);
            var hw = w;
            hw[0].host = self.lanes[0].hist[0 .. position + rows];
            self.logits = try self.runner.forward(&hw, gather, self.stream);
        } else self.logits = try self.runner.forward(&w, gather, self.stream);
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
    round_ms: [max_rows]f64,
    /// A round's ms of 1 and `max_rows` rows at the widest bucket (the context's), the same step's.
    round_deep_ms: [2]f64,
    draft_ms: [max_slots]f64,
    streams: Streams,
    /// Prompt rows a chunk (the settings' chunk, at most the context).
    chunk: usize,
    comm: cuda.comm.Comm,
    stream: cuda.Stream,
    agree_send: cuda.DeviceBuffer,
    agree_recv: cuda.DeviceBuffer,
    ids_buf: []i64, // a received prompt's ids
    op_buf: []u8,
    stepped: u64 = 0,
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
        e.cfg = try Config.read(gpa, io, dir);
        // the communicator first: rank 1 waits here for rank 0's store, as the Python ranks do
        e.comm = try cuda.comm.Comm.init(s.rank, s.world, s.master, s.port, 1_800_000);
        errdefer e.comm.deinit();
        e.set = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, s.kernels_dir);
        errdefer e.set.deinit();
        e.glue = try attention.Glue.load(ctx.d);
        errdefer e.glue.deinit();
        e.kernels = try exl3.Kernels.load(ctx.d);
        errdefer e.kernels.deinit();
        e.ek = try experts.Kernels.load(ctx.d);
        errdefer e.ek.deinit();
        const rc_path = (try RankCache.locate(gpa, io, s.cache_dir, s.rank, s.world)) orelse return error.NoRankCache;
        defer gpa.free(rc_path);
        var rc = try RankCache.open(gpa, rc_path);
        defer rc.close();
        std.log.info("deepseek_v41 rank {d} of {d}: weights from {s}", .{ s.rank, s.world, rc_path });
        e.w = try weights.load(gpa, ctx.d, &e.cfg, &rc, &e.kernels, s.rank, s.world, .mul1, null);
        errdefer e.w.deinit();
        e.stream = try cuda.Stream.init(ctx.d, true);
        errdefer e.stream.deinit();
        const rows = ropeRows(s.context);
        e.rope_c = try loadRope(gpa, io, ctx.d, s.rope_dir, "rope_c", rows);
        errdefer freeRope(&e.rope_c);
        e.rope_w = try loadRope(gpa, io, ctx.d, s.rope_dir, "rope_w", rows);
        errdefer freeRope(&e.rope_w);
        e.fwd = try forward.Forward.init(gpa, ctx.d, &e.cfg, &e.w, &e.set, &e.glue, &e.ek, s.world, max_rows, base_cap, s.bucket, e.rope_c, e.rope_w, e.stream);
        errdefer e.fwd.deinit();
        e.engram = null;
        if (getenv("TF_DS_ENGRAM")) |engram_dir| {
            const tm = if (getenv("TF_DS_TOKEN_MAP")) |p| try gpa.dupe(u8, p) else try std.fs.path.join(gpa, &.{ s.kernels_dir, "token_map.json" });
            defer gpa.free(tm);
            const threads: usize = if (getenv("TF_DS_ENGRAM_THREADS")) |t| std.fmt.parseInt(usize, t, 10) catch 32 else 32;
            e.engram = try engram_apply.Engram.init(gpa, io, ctx.d, &e.cfg, &e.w, &e.set, engram_dir, tm, @max(max_rows, e.chunk), threads);
            std.log.info("deepseek_v41 rank {d}: Engram tables {s}, token map {s}, {d} reader threads", .{ s.rank, engram_dir, tm, threads });
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
            e.stages = try gpa.alloc(dspark.Stage, e.ds.?.blocks.len);
            for (e.stages, e.ds.?.blocks) |*st, *b| st.* = dspark.Stage.of(b);
            e.drafter = try dspark.Drafter.init(gpa, ctx.d, dspark.dimsOf(&e.cfg, s.rank, s.world), &e.set, &e.glue, &e.ek, e.stages, dspark.headOf(&e.ds.?, &e.w.head), &e.comm);
            try e.drafter.?.enableWide(&e.kernels);
            std.log.info("deepseek_v41 rank {d}: DSpark drafter ({d} stages, taps of layers {any})", .{ s.rank, e.stages.len, e.cfg.dspark_taps.slice() });
        }
        const before = cuda.usage(false).device;
        try e.streams.init(gpa, ctx.d, &e.fwd, &e.kernels, e.engram, e.chunk, e.stream, e.drafter, s.slots, s.context + extra);
        errdefer e.streams.deinit();
        if (s.rank == 0) std.log.info("deepseek_v41: {d} stream slots over a window of {d} positions ({d:.2} GiB), prompts in chunks of {d} rows", .{ s.slots, e.streams.extents.total, @as(f64, @floatFromInt(cuda.usage(false).device - before)) / (1 << 30), e.chunk });
        e.agree_send = try cuda.DeviceBuffer.alloc(ctx.d, 16);
        errdefer e.agree_send.free();
        e.agree_recv = try cuda.DeviceBuffer.alloc(ctx.d, 16 * s.world);
        errdefer e.agree_recv.free();
        e.ids_buf = try gpa.alloc(i64, s.context + extra);
        errdefer gpa.free(e.ids_buf);
        e.op_buf = try gpa.alloc(u8, maxOpBytes(s.context));
        errdefer gpa.free(e.op_buf);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.gpa.free(e.op_buf);
        e.gpa.free(e.ids_buf);
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
        return Streams.bytesPerSlot(&e.cfg, e.w.layers.len, e.settings.context + extra, e.settings.slots, e.stages.len);
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
        try e.agree_send.upload(0, std.mem.sliceAsBytes(&mine));
        try e.comm.allGather(e.agree_send.ptr, e.agree_recv.ptr, 4, .f32, e.stream.handle);
        try e.stream.synchronize();
        var every: [4 * cuda.comm.Comm.max_world]f32 = undefined;
        try e.agree_recv.download(0, std.mem.sliceAsBytes(every[0 .. 4 * e.settings.world]));
        for (1..e.settings.world) |r| for (0..4) |i| if (every[r * 4 + i] != every[i]) {
            std.log.err("rank {d}: step {d} differs between the ranks (digest piece {d})", .{ e.settings.rank, e.stepped, i });
            return error.OutOfStep;
        };
        e.stepped += 1;
    }

    /// Rank 0 sends the step to every other rank, then every rank agrees on it. A failure breaks the lane: a
    /// follower that got the step but not the agreement would otherwise pair its next collective wrongly.
    fn announce(e: *Engine, op: []const u8) !void {
        if (e.broken) return error.OutOfStep;
        errdefer e.broken = true;
        if (e.settings.rank == 0) for (1..e.settings.world) |peer| try e.comm.sendTo(@intCast(peer), op);
        try e.agree(op, true);
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
                try e.streams.admit(e.gather(), a.slot, a.base, a.size, a.idsInto(e.ids_buf), a.chunk);
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
            .free => e.streams.free(op[1]),
            .measure => try e.runMeasure(try Measure.parse(op)),
            else => return error.BadRecord,
        }
    }

    /// The measure step (every rank): rounds of 1 .. max_rows rows at position 0 of slot 0 (the least bucket), of
    /// 1 and max_rows rows at the widest bucket (`probeDeep`), then batched drafts of 1 .. draftBatch streams, each
    /// timed after a warm-up run. Rank 0 keeps the times. The positions follow from the record's window alone, so
    /// every rank runs the same buckets.
    fn runMeasure(e: *Engine, m: Measure) !void {
        const reps: usize = @max(m.reps, 1);
        const st = &e.streams;
        for (1..max_rows + 1) |n| e.round_ms[n - 1] = try e.timeRound(reps, n, 0);
        for ([_]usize{ 1, max_rows }, &e.round_deep_ms) |n, *ms| ms.* = try e.timeRound(reps, n, st.probeDeep(n));
        if (st.spark) |sp| {
            const most = @min(draftBatch(sp.dims.block), st.lanes.len);
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
                return a.engram == (e.engram != null) and a.drafts == (e.drafter != null) and a.len() <= e.ids_buf.len and st.admits(a.slot, a.base, a.size, a.len(), a.chunk);
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
            .free => return op.len == 2 and op[1] < st.lanes.len and st.lanes[op[1]].busy,
            .measure => {
                const m = Measure.parse(op) catch return false;
                return m.slots == st.lanes.len and m.total == st.extents.total and m.engram == (e.engram != null) and m.drafts == (e.drafter != null) and st.idle();
            },
            else => return false,
        }
    }

    /// Rank 1 (and any other follower): run rank 0's steps until it says stop or the link closes.
    pub fn follow(e: *Engine) !void {
        if (e.settings.rank == 0) return;
        std.log.info("deepseek_v41 rank {d}: following rank 0", .{e.settings.rank});
        while (true) {
            const op = e.comm.recvFrom(0, e.op_buf) catch |err| switch (err) {
                error.LinkClosed => return,
                else => return err,
            };
            if (op.len == 0) return error.OutOfStep;
            const code = std.enums.fromInt(Op, op[0]) orelse return error.OutOfStep;
            // rank 0 sends stop without an agreement (it is tearing its communicator down)
            if (code == .stop) return;
            const ok = e.runnable(op);
            if (!ok) std.log.err("rank {d}: step {d} ({s}) cannot run here", .{ e.settings.rank, e.stepped, @tagName(code) });
            try e.agree(op, ok);
            try e.step(Engine.run, .{ e, op });
        }
    }

    /// Rank 0: time the rounds and drafts on every rank (the `measure` step); `round_ms` and `draft_ms` hold rank
    /// 0's.
    pub fn measure(e: *Engine, reps: u8) !void {
        if (e.settings.rank != 0) return;
        var buf: [Measure.bytes]u8 = undefined;
        try e.both(Measure.encode(&buf, .{ .reps = reps, .slots = e.streams.lanes.len, .engram = e.engram != null, .drafts = e.drafter != null, .total = e.streams.extents.total }));
    }

    pub fn stop(e: *Engine) void {
        if (e.settings.rank != 0) return;
        const op = [_]u8{@intFromEnum(Op.stop)};
        for (1..e.settings.world) |peer| e.comm.sendTo(@intCast(peer), &op) catch {};
    }
};

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

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

fn freeRope(r: *attention.Rope) void {
    r.cos.free();
    r.sin.free();
}

/// `<dir>/<name>_<rows>.bin`: cos then sin, f32 [rows, rope_dim / 2] each (tools/zig/dsv41/dump_rope.py).
fn loadRope(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, name: []const u8, rows: usize) !attention.Rope {
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

/// The round loop's facts: shared rounds of up to `slots` streams and `max_rows` rows (`hidden_rows`), windows of
/// up to `draft_width` rows; with the drafter, every drafting stream's drafts in one batch (`draft_streams`) and
/// the depths priced by `window_costs` (a lone stream's window, its draft included) and `shared_costs` (a round's
/// rows); without it, host-token windows only.
pub fn factsOf(drafting: bool, slots: usize, window_costs: []const lanes.config.Cost, shared_costs: []const lanes.config.Cost) lanes.Model {
    return .{
        .exact_width = draft_width,
        .gpu_tokens = false,
        .mtp = drafting,
        .speculate = drafting,
        .speculate_early = false,
        .draft_prior = &draft_prior,
        .drafts = @intCast(draft_width - 1),
        .window_costs = window_costs,
        .mtp_step_ms = 0,
        .streams_exact = true,
        .hidden_rows = true,
        .batch_rows = max_rows,
        .max_streams = @intCast(slots),
        .shared_costs = shared_costs,
        .draft_streams = drafting,
    };
}

/// `Loaded.rows` (lanes.Config's max_rows: it caps a window's width and its drafts).
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
    window_costs: [draft_width]lanes.config.Cost = @splat(.{ .width = 0, .ms = 0 }),
    shared_costs: [max_rows]lanes.config.Cost = @splat(.{ .width = 0, .ms = 0 }),

    pub fn init(gpa: std.mem.Allocator, e: *Engine) Lanes {
        return .{ .gpa = gpa, .e = e };
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .release = releaseFn,
        } };
    }

    /// Times the rounds and drafts on both ranks (the `measure` step) for the facts' costs: a round of R rows costs
    /// its forward plus, with the drafter, a batched draft of the streams that many rows hold (a drafted round runs
    /// both); a lone window of w rows is a round of w rows.
    pub fn measure(self: *Lanes, reps: u8) !void {
        const e = self.e;
        try e.measure(reps);
        const per = if (e.drafter) |dr| draftBatch(dr.dims.block) else 1;
        for (&self.shared_costs, 0..) |*c, i| {
            const rows = i + 1;
            var ms = e.round_ms[i];
            if (e.drafter != null and rows > 1) {
                var streams = @min((rows + draft_width - 1) / draft_width, e.streams.lanes.len);
                while (streams > 0) : (streams -|= per) ms += e.draft_ms[@min(streams, per) - 1];
            }
            c.* = .{ .width = @intCast(rows), .ms = ms };
        }
        for (&self.window_costs, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = e.round_ms[i] + if (e.drafter != null and i > 0) e.draft_ms[0] else 0 };
        std.log.info("deepseek_v41: rounds of 1..{d} rows {any} ms at the {d}-position bucket ({d} and {d} rows {any} ms at the {d}-position bucket), batched drafts of 1..{d} streams {any} ms", .{ max_rows, e.round_ms, rounds.bucket_min, 1, max_rows, e.round_deep_ms, e.streams.runner.bucket, max_slots, e.draft_ms });
    }

    pub fn facts(self: *const Lanes) lanes.Model {
        const timed = self.shared_costs[0].ms > 0;
        return factsOf(self.e.drafter != null, self.e.streams.lanes.len, if (timed) &self.window_costs else &.{}, if (timed) &self.shared_costs else &.{});
    }

    /// The window rows the host sizes the round loop for (`Loaded.rows`).
    pub fn windowRows(self: *const Lanes) u32 {
        return rowsOf(self.e.drafter != null);
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

    /// The stream into the lowest free slot and the first extent that holds its prompt, its reply and a round's
    /// rows (multi.py `admit`): its prompt fills the slot now, on every rank; its first token is drawn.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        const st = &e.streams;
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotSupported;
        const ptoks = s.prompt();
        const need = ptoks.len + s.max_new + extent_tail;
        if (ptoks.len == 0 or need > st.extents.total) return error.PromptTooLong;
        var slot: ?usize = null;
        for (st.lanes, 0..) |l, i| if (!l.busy and self.owner[i] == null) {
            slot = i;
            break;
        };
        const free = slot orelse return error.NoRoom;
        const base = st.extents.first(need) orelse return error.NoRoom;
        // a cancel is honoured only before the step is announced
        if (s.isCancelled()) return error.Cancelled;
        const ids = e.ids_buf[0..ptoks.len];
        for (ids, ptoks) |*id, t| id.* = t;
        const op = Admit.encode(e.op_buf, .{ .slot = free, .engram = e.engram != null, .drafts = e.drafter != null, .chunk = e.chunk, .base = base, .size = Extents.size(need), .ids = &.{} }, ids);
        try e.both(op);
        self.owner[free] = s;
        var first: [1]u32 = undefined;
        try st.argmaxRows(0, &first);
        _ = self.take(first[0]);
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
        const e = self.e;
        const st = &e.streams;
        if (windows.len > st.lanes.len) return error.WindowTooWide;
        var ids: [max_rows]i64 = undefined;
        var rws: [max_slots]RoundWindow = undefined;
        var r: usize = 0;
        for (windows, out, rws[0..windows.len]) |w, o, *rw| {
            const slot = try self.slotOf(w.stream);
            const l = &st.lanes[slot];
            if (w.parents != null) return error.TreesNotBuilt;
            const rows = w.rows();
            if (rows > draft_width or r + rows > max_rows) return error.WindowTooWide;
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
        }
        try e.both(Round.encode(e.op_buf, rws[0..windows.len]));
        r = 0;
        for (windows, out) |w, o| {
            try st.argmaxRows(r, o.sampled[0..w.rows()]);
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
            const l = &st.lanes[slot];
            if (l.pos == l.verify_start + path.len) continue;
            asks[n] = .{ .slot = slot, .start = l.verify_start, .kept = path.len };
            n += 1;
        }
        if (n > 0) try e.both(Keeps.encode(e.op_buf, asks[0..n]));
    }

    /// Every request in one drafts step: absorb its kept rows (none after the prompt: its chunks absorbed theirs),
    /// then hold `depth` drafts after its pending token, the drafting streams batched.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        const st = &e.streams;
        if (e.drafter == null) return error.NoDraftHead;
        if (requests.len > max_slots) return error.WindowTooWide;
        var asks: [max_slots]DraftAsk = undefined;
        for (requests, asks[0..requests.len]) |r, *a| {
            if (r.lanes != null) return error.TreesNotBuilt;
            const slot = try self.slotOf(r.stream);
            const l = &st.lanes[slot];
            a.* = .{ .slot = slot, .start = l.pos, .absorb = 0, .token = 0, .q0 = l.pos, .depth = @min(r.depth, e.cfg.dspark_block) };
            if (r.rows) |rows| {
                if (rows.len == 0 or r.follow.len < rows.len) return error.EmptyPath;
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
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

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const slot = self.slotOf(s) catch return;
        self.owner[slot] = null;
        const e = self.e;
        if (!e.streams.lanes[slot].busy or e.broken) return;
        const op = [_]u8{ @intFromEnum(Op.free), @intCast(slot) };
        e.both(&op) catch |err| std.log.err("release: {s}", .{@errorName(err)});
    }
};

test "rope rows align up to 4096 past the context and the window's extra rows" {
    try std.testing.expectEqual(@as(usize, 4096), ropeRows(1024));
    try std.testing.expectEqual(@as(usize, 20480), ropeRows(16384));
    try std.testing.expectEqual(@as(usize, 69632), ropeRows(65536));
}

test "a prefill record parses back to its chunk, Engram and drafter flags and ids; a cut or foreign record is refused" {
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
        const op = Prefill.encode(&buf, chunk, engram, drafts, ids[0..n]);
        try std.testing.expectEqual(Prefill.header + 8 * n, op.len);
        const p = try Prefill.parse(op);
        try std.testing.expectEqual(chunk, p.chunk);
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

test "admission, round, drafts, keeps and measure records parse back to what was encoded; cut or foreign ones are refused" {
    var prng = std.Random.DefaultPrng.init(0x9a7a);
    const r = prng.random();
    var buf: [Admit.header + 8 * 64]u8 = undefined;
    var ids: [64]i64 = undefined;
    var back: [64]i64 = undefined;
    for (0..300) |_| {
        const n = r.intRangeAtMost(usize, 0, ids.len);
        for (ids[0..n]) |*id| id.* = r.int(i64);
        const a: Admit = .{ .slot = r.intRangeLessThan(usize, 0, max_slots), .engram = r.boolean(), .drafts = r.boolean(), .chunk = r.intRangeAtMost(usize, 1, prompt.max_chunk), .base = r.int(u40), .size = r.int(u40), .ids = &.{} };
        const op = Admit.encode(&buf, a, ids[0..n]);
        const p = try Admit.parse(op);
        try std.testing.expect(p.slot == a.slot and p.engram == a.engram and p.drafts == a.drafts and p.chunk == a.chunk and p.base == a.base and p.size == a.size);
        try std.testing.expectEqualSlices(i64, ids[0..n], p.idsInto(&back));
        try std.testing.expectError(error.BadRecord, Admit.parse(op[0 .. op.len - 1]));
        try std.testing.expectError(error.BadRecord, Prefill.parse(op));

        // a round: 1 .. max_slots windows of 1 .. rows, max_rows rows in all
        var windows: [max_slots]RoundWindow = undefined;
        const count = r.intRangeAtMost(usize, 1, max_slots);
        var left: usize = max_rows;
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

        const m: Measure = .{ .reps = r.int(u8), .slots = r.intRangeAtMost(usize, 1, max_slots), .engram = r.boolean(), .drafts = r.boolean(), .total = r.int(u40) };
        const mop = Measure.encode(&buf, m);
        try std.testing.expectEqual(m, try Measure.parse(mop));
        try std.testing.expectError(error.BadRecord, Measure.parse(mop[0 .. mop.len - 1]));
    }
    // a round wider than max_rows rows never parses
    var wide: [2 + 26 + 8 * (max_rows + 1)]u8 = @splat(0);
    wide[0] = @intFromEnum(Op.rounds);
    wide[1] = 1;
    wide[3] = max_rows + 1;
    var rr: Round = undefined;
    try std.testing.expectError(error.BadRecord, Round.parse(&wide, &rr));
}

