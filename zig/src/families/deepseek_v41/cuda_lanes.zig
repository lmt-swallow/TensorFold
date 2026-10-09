//! DeepSeek-V4.1 behind the lane core on two ranks, one stream at a time, greedy: rank 0 serves and sends every step
//! (the prompt, each round, the release) to rank 1 over the communicator's control link before running it; rank 1
//! runs the same steps in `follow`, so both ranks issue the same all-gathers in the same order (model.py's lockstep).
//! A sha256 digest of each step is all-gathered first (multi.py `_agree`): ranks that planned different steps stop
//! instead of pairing collectives wrongly.
//!
//! With DSpark drafts (the checkpoint's `mtp.*` blocks, cuda_dspark.zig) the rounds are engine.py `_spec`'s: the
//! prompt's chunks hand their taps (the tap layers' inputs, mean over the streams) to the drafter; a `draft` step
//! absorbs the kept rows' taps and drafts 5 tokens after the pending one; a `verify` step runs the pending token and
//! the held drafts as one window through the decode layer loop (every row's logits, the window's taps); a `keep` step
//! sets the sequence's length to the kept rows (`sc.length = P + a + 1`: the rejected rows' cache writes stay and are
//! overwritten before any later row reads them, as in the fork). Drafts are deterministic given the same state, so
//! the records carry ids and positions only; every rank drafts the same tokens.
//!
//! The prompt runs through the prompt path (cuda_prompt.zig) in chunks of TF_DS_PREFILL_CHUNK rows (default 2048,
//! the fork's production setting; engine.py `prefill_steps`: chunks start at multiples of it from position 0, the
//! last chunk's last row gives the first token), then each round runs through the decode layer loop
//! (cuda_forward.zig) on the same caches. Both ranks run a chunk's gathers in the same order: per layer Engram's
//! (on its layers, when the tables are loaded), attention's and the MoE's, then the head's on the prompt's last chunk
//! (and on any chunk of 16 rows or fewer, which runs as a decode window). Sampling above temperature 0 is refused
//! (the keyed draw is not ported).
//!
//! Engram (cuda_engram_apply.zig) runs on its layers when TF_DS_ENGRAM names the tables' dir (the token map from
//! TF_DS_TOKEN_MAP, default `<kernels>/token_map.json`; TF_DS_ENGRAM_THREADS reader threads, default 32): every rank
//! keeps the sequence's ids on the host (`hist`) and hashes them the same way, so the ranks' Engram gathers pair.
//! Without the tables the gate is skipped, as the fork does, and the output is degraded.
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
const RankCache = @import("rank_cache.zig").RankCache;

const be = lanes.backend;
const Ptr = cuda.abi.DevicePtr;

pub const max_rows: usize = 16;
/// Positions past the context a sequence's RoPE tables and caches hold (engine.py: limit + max_rows + 8).
pub const extra: usize = max_rows + 8;
pub const rope_align: usize = 4096;

pub const Error = error{ WorldNotTwo, BadMaster, NoRankCache, NoRopeTables, RopeRowsShort, OutOfStep, LinkClosed, SamplingNotSupported, WindowTooWide, PositionMismatch, UnknownStream, StreamBusy, PromptTooLong, Cancelled, NoDraftHead, NotPipelined, TreesNotBuilt, EmptyPath, BadChunk, BadRecord, NoHeldDrafts, NoFirstToken, BadTapLayers };

/// A drafting round's widest window: the pending token and DSpark's block of drafts (engine.py: drafts + 1 rows).
pub const draft_width: usize = 6;

/// The DSpark head's acceptance at depth 1, 2, ... given the ones before it, until a stream measures its own (the
/// fork's served acceptance is about 0.75 a depth).
pub const draft_prior = [_]f64{ 0.75, 0.75, 0.75, 0.75, 0.75 };

/// Prompt rows a chunk unless TF_DS_PREFILL_CHUNK says otherwise (the fork's production setting).
pub const default_chunk: usize = prompt.max_chunk;

pub const Op = enum(u8) { prefill = 1, round = 2, release = 3, stop = 4, verify = 5, keep = 6, draft = 7, measure = 8 };

/// The bytes a step's record takes at most for a context of `context` positions (a prefill of the whole context).
pub fn maxOpBytes(context: usize) usize {
    return Prefill.header + 8 * (context + extra);
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
};

/// One sequence's state on this rank: the prompt path (`pr`) and the decode layer loop (`fwd`) over the same caches
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
    /// Window ms by rows (index rows - 1) and one draft's ms, timed at load by the `measure` step (rank 0's clock).
    costs: [draft_width]f64,
    draft_ms: f64,
    seq: Sequence,
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
        e.fwd = try forward.Forward.init(gpa, ctx.d, &e.cfg, &e.w, &e.set, &e.glue, &e.ek, s.world, max_rows, s.context + extra, s.bucket, e.rope_c, e.rope_w, e.stream);
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
        e.costs = @splat(0);
        e.draft_ms = 0;
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
        e.seq = try Sequence.init(gpa, ctx.d, &e.fwd, &e.kernels, e.engram, e.chunk, e.stream, e.drafter);
        errdefer e.seq.deinit(gpa);
        if (s.rank == 0) std.log.info("deepseek_v41: prompts in chunks of {d} rows", .{e.chunk});
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
        e.seq.deinit(e.gpa);
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
        if (e.ds) |*ds| freeDSpark(e.gpa, ds);
        e.ds = null;
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

    // -- the steps (the same on every rank) -------------------------------------------------------------------

    /// The prefill step from its record: the prompt into the caches in the record's chunks (`Sequence.prefill`).
    fn runPrefill(e: *Engine, op: []const u8) !void {
        try Prefill.run(op, &e.seq, e.gather(), e.ids_buf);
    }

    /// The verify step from its record: the window at the cache's end (`Sequence.verify`).
    fn runVerify(e: *Engine, op: []const u8) !void {
        const v = try Verify.parse(op);
        var ids: [max_rows]i64 = undefined;
        try e.seq.verify(e.gather(), v.idsInto(&ids));
    }

    /// The draft step from its record: absorb the kept rows, then draft (`Sequence.draft`).
    fn runDraft(e: *Engine, op: []const u8) !void {
        const r = try Draft.parse(op);
        try e.seq.draft(r.start, r.absorb, r.token, r.q0, r.depth);
    }

    /// The measure step (every rank): windows of 1 .. draft_width rows of the noise token at position 0, then one
    /// absorb and draft after them, each timed after a warm-up run; the sequence is empty after (the next prefill
    /// rewrites every cache row a later row reads). Rank 0 keeps the times.
    fn runMeasure(e: *Engine, op: []const u8) !void {
        const reps: usize = if (op.len == 2) @max(op[1], 1) else 1;
        const seq = &e.seq;
        var ids: [draft_width]i64 = @splat(@intCast(e.cfg.dspark_noise));
        for (1..draft_width + 1) |n| {
            var best: f64 = std.math.inf(f64);
            for (0..reps + 1) |rep| {
                seq.pos = 0;
                try e.stream.synchronize();
                const t0 = nowNs();
                try seq.verify(e.gather(), ids[0..n]);
                try e.stream.synchronize();
                if (rep > 0) best = @min(best, msSince(t0));
            }
            e.costs[n - 1] = best;
        }
        if (seq.spark != null) {
            var best: f64 = std.math.inf(f64);
            for (0..reps + 1) |rep| {
                seq.pos = 0;
                try seq.verify(e.gather(), ids[0..1]);
                if (seq.dc) |*c| c.absorbed = 0;
                try e.stream.synchronize();
                const t0 = nowNs();
                try seq.draft(0, 1, ids[0], 1, e.cfg.dspark_block);
                if (rep > 0) best = @min(best, msSince(t0));
            }
            e.draft_ms = best;
            if (seq.dc) |*c| c.absorbed = 0;
        }
        seq.pos = 0;
        seq.held = 0;
        seq.verify_rows = 0;
    }

    /// Whether this rank can run `op` as recorded: a prefill whose chunk fits this rank's prompt buffers, whose
    /// Engram and drafter flags match this rank's and whose prompt fits the context; a round, verify, keep or draft
    /// at this rank's position.
    fn runnable(e: *Engine, op: []const u8) bool {
        const code = std.enums.fromInt(Op, op[0]) orelse return false;
        switch (code) {
            .prefill => {
                const p = Prefill.parse(op) catch return false;
                return p.chunk >= 1 and p.chunk <= e.seq.pr.rows and p.engram == (e.engram != null) and p.drafts == (e.drafter != null) and p.len() >= 1 and p.len() + 1 <= e.seq.limit();
            },
            .round => {
                if (op.len != 17) return false;
                const pos: usize = @intCast(std.mem.readInt(u64, op[9..17], .little));
                return pos == e.seq.pos and pos + 1 <= e.seq.limit();
            },
            .verify => {
                const v = Verify.parse(op) catch return false;
                return v.start == e.seq.pos and v.start + v.len() <= e.seq.limit();
            },
            .keep => {
                const k = Keep.parse(op) catch return false;
                return k.start == e.seq.verify_start and k.kept >= 1 and k.kept <= e.seq.verify_rows and e.seq.pos == k.start + e.seq.verify_rows;
            },
            .draft => {
                const r = Draft.parse(op) catch return false;
                if (e.drafter == null) return false;
                if (r.absorb > 0 and (r.start != e.seq.verify_start or r.absorb > e.seq.verify_rows or e.seq.pos != r.start + r.absorb)) return false;
                return r.depth == 0 or r.q0 == e.seq.pos;
            },
            .measure => return op.len == 2 and e.seq.pos == 0,
            .release => return op.len == 1,
            .stop => return false,
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
            switch (code) {
                .prefill => try e.step(Engine.runPrefill, .{ e, op }),
                .round => try e.step(Sequence.round, .{ &e.seq, e.gather(), std.mem.readInt(i64, op[1..9], .little) }),
                .verify => try e.step(Engine.runVerify, .{ e, op }),
                .keep => {
                    const k = try Keep.parse(op);
                    try e.step(Sequence.keep, .{ &e.seq, k.start, k.kept });
                },
                .draft => try e.step(Engine.runDraft, .{ e, op }),
                .measure => try e.step(Engine.runMeasure, .{ e, op }),
                .release => {
                    e.seq.pos = 0;
                    e.seq.held = 0;
                },
                .stop => unreachable,
            }
        }
    }

    /// Rank 0: time the windows and a draft on every rank (the `measure` step); `costs` and `draft_ms` hold rank 0's.
    pub fn measure(e: *Engine, reps: u8) !void {
        if (e.settings.rank != 0) return;
        const op = [_]u8{ @intFromEnum(Op.measure), reps };
        try e.announce(&op);
        try e.step(Engine.runMeasure, .{ e, @as([]const u8, &op) });
    }

    pub fn stop(e: *Engine) void {
        if (e.settings.rank != 0) return;
        const op = [_]u8{@intFromEnum(Op.stop)};
        for (1..e.settings.world) |peer| e.comm.sendTo(@intCast(peer), &op) catch {};
    }
};

/// `weights.DSpark.deinit`, its host rows freed as the bytes they were allocated as (the loader dupes them as u8
/// and keeps them as u16 slices; freeing a u16 slice would claim an alignment the allocation never had).
pub fn freeDSpark(gpa: std.mem.Allocator, ds: *weights.DSpark) void {
    for ([_][]u16{ ds.markov_embed_host, ds.conf_host }) |h| {
        const bytes: [*]u8 = @ptrCast(h.ptr);
        gpa.free(bytes[0 .. 2 * h.len]);
    }
    ds.markov_embed_host = &.{};
    ds.conf_host = &.{};
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

/// The lane backend on rank 0: one stream; with the drafter, drafted windows of up to `draft_width` rows (the
/// pending token, then held DSpark drafts or host tokens), else plain one-token rounds.
pub const Lanes = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    active: ?*const lanes.Stream = null,
    drawn: [1024]u32 = undefined,
    next: u64 = 0,
    costs: [draft_width]lanes.config.Cost = @splat(.{ .width = 0, .ms = 0 }),

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

    /// Times the windows and a draft on both ranks (the `measure` step) for `facts`' window costs: a window of n
    /// rows costs its forward, plus one draft when it carries drafts (a drafted round runs both).
    pub fn measure(self: *Lanes, reps: u8) !void {
        const e = self.e;
        if (e.drafter == null) return;
        try e.measure(reps);
        for (&self.costs, e.costs, 0..) |*c, ms, i| c.* = .{ .width = @intCast(i + 1), .ms = if (i == 0) ms else ms + e.draft_ms };
        std.log.info("deepseek_v41: windows of 1..{d} rows {any} ms, a draft {d:.2} ms", .{ draft_width, e.costs, e.draft_ms });
    }

    pub fn facts(self: *const Lanes) lanes.Model {
        if (self.e.drafter == null) return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .hidden_rows = false, .batch_rows = 1, .max_streams = 1 };
        return .{
            .exact_width = draft_width,
            .gpu_tokens = false,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .draft_prior = &draft_prior,
            .drafts = @intCast(draft_width - 1),
            .window_costs = if (self.costs[0].ms > 0) &self.costs else &.{},
            .mtp_step_ms = 0,
            .hidden_rows = false,
            .batch_rows = draft_width,
            .max_streams = 1,
        };
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

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        if (self.active != null and self.active != s) return error.StreamBusy;
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotSupported;
        const ptoks = s.prompt();
        if (ptoks.len == 0 or ptoks.len + s.max_new + max_rows > e.settings.context) return error.PromptTooLong;
        // a cancel is honoured only before the step is announced (runPrefill)
        if (s.isCancelled()) return error.Cancelled;
        const ids = e.ids_buf[0..ptoks.len];
        for (ids, ptoks) |*id, t| id.* = t;
        self.active = s;
        const op = Prefill.encode(e.op_buf, e.chunk, e.engram != null, e.drafter != null, ids);
        try e.announce(op);
        try e.step(Engine.runPrefill, .{ e, op });
        _ = self.take(try e.seq.argmax());
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return self.next - 1;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return of(ptr).value(.{ .handle = handle });
    }

    /// One stream's window at the cache's end: the pending token, then the held drafts (or host tokens), each row's
    /// draw greedy. Without a drafter, one row through the plain round step.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        if (windows.len != 1) return error.WindowTooWide;
        const w = windows[0];
        if (self.active != w.stream) return error.UnknownStream;
        if (w.parents != null) return error.TreesNotBuilt;
        const rows = w.rows();
        if (rows > (if (e.drafter != null) draft_width else 1)) return error.WindowTooWide;
        for (w.positions, 0..) |p, r| if (p != e.seq.pos + 1 + r) return error.PositionMismatch;
        if (e.drafter == null) {
            try e.announce(encodeRound(e.op_buf, w.pending, e.seq.pos));
            try e.step(Sequence.round, .{ &e.seq, e.gather(), @as(i64, w.pending) });
            out[0].sampled[0] = try e.seq.argmax();
            return;
        }
        if (w.held > e.seq.held) return error.NoHeldDrafts;
        var ids: [draft_width]i64 = undefined;
        ids[0] = w.pending;
        for (ids[1..][0..w.held], e.seq.drafts.tokens[0..w.held]) |*id, t| id.* = t;
        for (ids[1 + w.held ..][0..w.tokens.len], w.tokens) |*id, t| id.* = t;
        for (out[0].drafts[0..w.held], e.seq.drafts.tokens[0..w.held]) |*o, t| o.* = t;
        @memcpy(out[0].drafts[w.held..][0..w.tokens.len], w.tokens);
        e.seq.held = 0;
        const op = Verify.encode(e.op_buf, e.seq.pos, ids[0..rows]);
        try e.announce(op);
        try e.step(Engine.runVerify, .{ e, op });
        try e.seq.argmaxRows(out[0].sampled[0..rows]);
    }

    /// The window's kept prefix: the sequence's length rolls back to it on both ranks.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            if (self.active != w.stream) return error.UnknownStream;
            if (e.drafter == null) {
                if (path.len != 1) return error.TreesNotBuilt;
                continue;
            }
            if (path.len == e.seq.verify_rows) continue;
            const op = Keep.encode(e.op_buf, .{ .start = e.seq.verify_start, .kept = path.len });
            try e.announce(op);
            try e.step(Sequence.keep, .{ &e.seq, e.seq.verify_start, path.len });
        }
    }

    /// Absorb the kept rows (none after the prompt: its chunks absorbed theirs), then hold `depth` drafts after the
    /// pending token.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        if (e.drafter == null) return error.NoDraftHead;
        for (requests) |r| {
            if (self.active != r.stream) return error.UnknownStream;
            if (r.lanes != null) return error.TreesNotBuilt;
            if (r.position != e.seq.pos + 1) return error.PositionMismatch;
            var d: Draft = .{ .start = e.seq.pos, .absorb = 0, .token = 0, .q0 = e.seq.pos, .depth = @min(r.depth, e.cfg.dspark_block) };
            if (r.rows) |rows| {
                if (rows.len == 0 or r.follow.len < rows.len) return error.EmptyPath;
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                d.start = @intCast(r.start);
                d.absorb = rows.len;
                d.token = r.follow[rows.len - 1];
            } else d.token = try self.value(r.first orelse return error.NoFirstToken);
            const op = Draft.encode(e.op_buf, d);
            try e.announce(op);
            try e.step(Engine.runDraft, .{ e, op });
        }
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        if (self.active != s) return;
        self.active = null;
        const e = self.e;
        const op = [_]u8{@intFromEnum(Op.release)};
        if (!e.broken) e.announce(&op) catch |err| std.log.err("release: {s}", .{@errorName(err)});
        e.seq.pos = 0;
        e.seq.held = 0;
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
