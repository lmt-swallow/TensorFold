//! DeepSeek-V4.1 behind the lane core on two ranks, one stream at a time, greedy, no drafts: rank 0 serves and
//! sends every step (the prompt, each one-token round, the release) to rank 1 over the communicator's control link
//! before running it; rank 1 runs the same steps in `follow`, so both ranks issue the same all-gathers in the same
//! order (model.py's lockstep). A sha256 digest of each step is all-gathered first (multi.py `_agree`): ranks that
//! planned different steps stop instead of pairing collectives wrongly.
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
const RankCache = @import("rank_cache.zig").RankCache;

const be = lanes.backend;
const Ptr = cuda.abi.DevicePtr;

pub const max_rows: usize = 16;
/// Positions past the context a sequence's RoPE tables and caches hold (engine.py: limit + max_rows + 8).
pub const extra: usize = max_rows + 8;
pub const rope_align: usize = 4096;

pub const Error = error{ WorldNotTwo, BadMaster, NoRankCache, NoRopeTables, RopeRowsShort, OutOfStep, LinkClosed, SamplingNotSupported, WindowTooWide, PositionMismatch, UnknownStream, StreamBusy, PromptTooLong, Cancelled, NoDraftHead, NotPipelined, TreesNotBuilt, EmptyPath, BadChunk, BadRecord };

/// Prompt rows a chunk unless TF_DS_PREFILL_CHUNK says otherwise (the fork's production setting).
pub const default_chunk: usize = prompt.max_chunk;

pub const Op = enum(u8) { prefill = 1, round = 2, release = 3, stop = 4 };

/// The bytes a step's record takes at most for a context of `context` positions (a prefill of the whole context).
pub fn maxOpBytes(context: usize) usize {
    return Prefill.header + 8 * (context + extra);
}

/// The prefill step's record (little-endian): the opcode, the chunk rows (u32), the flags (u8: bit 0, Engram on),
/// the prompt's length (u32), then its ids (i64 each). The record is what the ranks' digests agree on, so ranks that
/// would chunk differently or disagree on Engram (and so pair different gathers) stop before the first gather.
pub const Prefill = struct {
    chunk: usize,
    engram: bool,
    ids: []const u8, // the ids' little-endian bytes, 8 a token

    pub const header: usize = 10;

    pub fn encode(buf: []u8, chunk: usize, engram: bool, ids: []const i64) []const u8 {
        buf[0] = @intFromEnum(Op.prefill);
        std.mem.writeInt(u32, buf[1..5], @intCast(chunk), .little);
        buf[5] = @intFromBool(engram);
        std.mem.writeInt(u32, buf[6..10], @intCast(ids.len), .little);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, buf[header + 8 * i ..][0..8], id, .little);
        return buf[0 .. header + 8 * ids.len];
    }

    pub fn parse(op: []const u8) error{BadRecord}!Prefill {
        if (op.len < header or op[0] != @intFromEnum(Op.prefill) or op[5] > 1) return error.BadRecord;
        const n = std.mem.readInt(u32, op[6..10], .little);
        if (op.len != header + 8 * @as(usize, n)) return error.BadRecord;
        return .{ .chunk = std.mem.readInt(u32, op[1..5], .little), .engram = op[5] == 1, .ids = op[header..] };
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
};

/// One sequence's state on this rank: the prompt path (`pr`) and the decode layer loop (`fwd`) over the same caches
/// (`fwd`'s rings and compressed planes), the host ids Engram hashes, and the position the next row takes. Every
/// rank holding the same sequence runs the same calls with the same arguments, so their gathers pair.
pub const Sequence = struct {
    fwd: *forward.Forward,
    pr: prompt.Prompt,
    engram: ?*engram_apply.Engram,
    hist: []i32, // the sequence's ids from position 0 (Engram's hashes)
    logits_h: []f32,
    /// The last row's logits [1, V] f32 after `prefill` or `round` (the prompt path's or the decode loop's buffer).
    logits: Ptr = 0,
    pos: usize = 0,
    stream: cuda.Stream,

    /// `rows`: the widest chunk `prefill` takes (at most prompt.max_chunk). `engram`, when given, holds at least
    /// `rows` rows a call and becomes `fwd`'s hook.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, fwd: *forward.Forward, kernels: *const exl3.Kernels, engram: ?*engram_apply.Engram, rows: usize, stream: cuda.Stream) !Sequence {
        var self: Sequence = .{ .fwd = fwd, .pr = undefined, .engram = engram, .hist = undefined, .logits_h = undefined, .stream = stream };
        self.pr = try prompt.Prompt.init(gpa, d, fwd, kernels, rows);
        errdefer self.pr.deinit();
        self.hist = try gpa.alloc(i32, fwd.cap);
        errdefer gpa.free(self.hist);
        self.logits_h = try gpa.alloc(f32, fwd.cfg.vocab);
        if (engram) |g| fwd.engram = g.hook();
        return self;
    }

    pub fn deinit(self: *Sequence, gpa: std.mem.Allocator) void {
        gpa.free(self.logits_h);
        gpa.free(self.hist);
        self.pr.deinit();
    }

    /// The positions a sequence may reach (the caches' capacity less the window's extra rows).
    pub fn limit(self: *const Sequence) usize {
        return self.fwd.cap - extra;
    }

    /// The prompt's ids into the caches from position 0 (engine.py prefill_steps from start 0): chunks of `chunk`
    /// rows at positions 0, chunk, 2 chunk, ..., the last one the remainder; the last row's logits in `logits`.
    /// Runs every chunk once started: the other ranks cannot see a cancel, so stopping early on one rank would leave
    /// the others in an unpaired gather.
    pub fn prefill(self: *Sequence, gather: forward.Gather, ids: []const i64, chunk: usize) !void {
        if (chunk < 1 or chunk > self.pr.rows) return error.BadChunk;
        if (ids.len == 0 or ids.len + 1 > self.limit()) return error.PromptTooLong;
        self.pos = 0;
        self.logits = 0;
        for (self.hist[0..ids.len], ids) |*hv, id| hv.* = @intCast(id);
        if (self.engram) |g| g.history = self.hist[0..ids.len];
        var s: usize = 0;
        while (s < ids.len) {
            const e = @min(ids.len, s + chunk);
            const last = e == ids.len;
            // DSpark hook: with a drafter attached, pass `.taps` (prompt.Taps: the chunk's rows of the tap layers)
            // to this chunk and, after it, have the drafter absorb them (engine.py prefill_steps: `drafter.absorb(dc,
            // sc, torch.cat(taps, -1), m.taps_start)` per chunk). Taps make every chunk take the prompt path.
            // A chunk of max_rows or fewer is a decode window (it runs the head): Prompt.chunk sends it through
            // `Forward.rows` only when it asks for logits.
            const out = try self.pr.chunk(gather, ids[s..e], s, .{ .logits = last or e - s <= max_rows }, self.stream);
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

    /// The last logits' argmax (the first index of the maximum, torch's rule), read on the host.
    pub fn argmax(self: *Sequence) !u32 {
        try self.stream.synchronize();
        const view: cuda.DeviceBuffer = .{ .d = self.fwd.d, .ptr = self.logits, .len = self.logits_h.len * 4 };
        try view.download(0, std.mem.sliceAsBytes(self.logits_h));
        var best: usize = 0;
        for (self.logits_h, 0..) |v, i| if (v > self.logits_h[best]) {
            best = i;
        };
        return @intCast(best);
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
        e.seq = try Sequence.init(gpa, ctx.d, &e.fwd, &e.kernels, e.engram, e.chunk, e.stream);
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

    /// Whether this rank can run `op` as recorded: a prefill whose chunk fits this rank's prompt buffers, whose
    /// Engram flag matches this rank's tables and whose prompt fits the context; a round at this rank's position.
    fn runnable(e: *Engine, op: []const u8) bool {
        const code = std.enums.fromInt(Op, op[0]) orelse return false;
        switch (code) {
            .prefill => {
                const p = Prefill.parse(op) catch return false;
                return p.chunk >= 1 and p.chunk <= e.seq.pr.rows and p.engram == (e.engram != null) and p.len() >= 1 and p.len() + 1 <= e.seq.limit();
            },
            .round => {
                if (op.len != 17) return false;
                const pos: usize = @intCast(std.mem.readInt(u64, op[9..17], .little));
                return pos == e.seq.pos and pos + 1 <= e.seq.limit();
            },
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
                .release => e.seq.pos = 0,
                .stop => unreachable,
            }
        }
    }

    pub fn stop(e: *Engine) void {
        if (e.settings.rank != 0) return;
        const op = [_]u8{@intFromEnum(Op.stop)};
        for (1..e.settings.world) |peer| e.comm.sendTo(@intCast(peer), &op) catch {};
    }
};

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

/// The lane backend on rank 0: one stream, plain one-token rounds.
pub const Lanes = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    active: ?*const lanes.Stream = null,
    drawn: [1024]u32 = undefined,
    next: u64 = 0,

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

    pub fn facts(_: *const Lanes) lanes.Model {
        return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .hidden_rows = false, .batch_rows = 1, .max_streams = 1 };
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
        const op = Prefill.encode(e.op_buf, e.chunk, e.engram != null, ids);
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
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > self.drawn.len) return error.NoSuchToken;
        return self.drawn[handle % self.drawn.len];
    }

    /// One window of one row: the pending token at the cache's end, its draw greedy.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        if (windows.len != 1) return error.WindowTooWide;
        const w = windows[0];
        if (self.active != w.stream) return error.UnknownStream;
        if (w.parents != null) return error.TreesNotBuilt;
        if (w.rows() != 1) return error.WindowTooWide;
        if (w.positions[0] != e.seq.pos + 1) return error.PositionMismatch;
        try e.announce(encodeRound(e.op_buf, w.pending, e.seq.pos));
        try e.step(Sequence.round, .{ &e.seq, e.gather(), @as(i64, w.pending) });
        out[0].sampled[0] = try e.seq.argmax();
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |_, path| {
            if (path.len == 0) return error.EmptyPath;
            if (path.len != 1 or path[0] != 0) return error.TreesNotBuilt;
        }
    }

    fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
        return error.NoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        if (self.active != s) return;
        self.active = null;
        const e = self.e;
        const op = [_]u8{@intFromEnum(Op.release)};
        if (!e.broken) e.announce(&op) catch |err| std.log.err("release: {s}", .{@errorName(err)});
        e.seq.pos = 0;
    }
};

test "rope rows align up to 4096 past the context and the window's extra rows" {
    try std.testing.expectEqual(@as(usize, 4096), ropeRows(1024));
    try std.testing.expectEqual(@as(usize, 20480), ropeRows(16384));
    try std.testing.expectEqual(@as(usize, 69632), ropeRows(65536));
}

test "a prefill record parses back to its chunk, Engram flag and ids; a cut or foreign record is refused" {
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
        const op = Prefill.encode(&buf, chunk, engram, ids[0..n]);
        try std.testing.expectEqual(Prefill.header + 8 * n, op.len);
        const p = try Prefill.parse(op);
        try std.testing.expectEqual(chunk, p.chunk);
        try std.testing.expectEqual(engram, p.engram);
        try std.testing.expectEqualSlices(i64, ids[0..n], p.idsInto(&back));
        if (op.len > 0) try std.testing.expectError(error.BadRecord, Prefill.parse(op[0 .. op.len - 1]));
    }
    var round_op: [Prefill.header]u8 = @splat(0);
    round_op[0] = @intFromEnum(Op.round);
    try std.testing.expectError(error.BadRecord, Prefill.parse(&round_op));
    try std.testing.expect(maxOpBytes(16384) >= Prefill.header + 8 * 16384);
}
