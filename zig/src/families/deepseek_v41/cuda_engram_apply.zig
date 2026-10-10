//! Engram on one rank (the Python fork's `Model.engram_apply` with `model.py Engram`): for an Engram layer's rows at
//! positions start .. start + n - 1 of a sequence, the n-gram row ids from the host token history (engram_hash.zig),
//! this rank's columns of them read from the FP8 shards and decoded to bf16 `e [n, cols_r * 256]`
//! (engram_rows.zig), the rank's `engram_wkv` partial `e @ W_r` in fp32 [n, 5 d], the ranks' partials gathered and
//! summed in rank order to bf16 kv [n, 5 d] (Comm.sum(...).to(BF16)), then `engram_gate` on h [n, 4, d] in place.
//!
//! Every rank runs the same steps on the same host ids (the hashes are deterministic), so the gather pairs. The hook
//! (`hook()`) plugs into `Forward.engram`; `history` must hold the sequence's ids from position 0 through the rows
//! being run (negative ids mark an image span: no n-gram reaches into one).
//!
//! A prompt's chunks read their rows ahead (engine.py prefill_steps' `ahead`): `prefetch` hashes a chunk's rows and
//! a background thread reads every Engram layer's rows into pinned buffers of their own while the chunk before runs;
//! the chunk's layers then upload those rows instead of reading. A prefetched batch serves a call only when its
//! positions and row ids equal the call's, so the bits are the ones a direct read gives.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const exl3 = @import("cuda_exl3.zig");
const hcmod = @import("cuda_hc.zig");
const forward = @import("cuda_forward.zig");
const rounds = @import("cuda_rounds.zig");
const hash = @import("engram_hash.zig");
const rows_mod = @import("engram_rows.zig");

const Ptr = cuda.abi.DevicePtr;

pub const Error = error{ NotEngramLayer, NoTable, TooManyRows, HistoryShort, BadTokenMap };

/// Rows at most a call without an `mm` (the row-invariant decode linear's scratch is sized for these): a round's
/// most (cuda_rounds.max_rows).
pub const decode_rows: usize = rounds.max_rows;

pub const Engram = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    cfg: *const Config,
    w: *const weights.Weights,
    hc: hcmod.Hc,
    tables: rows_mod.Tables,
    reader: *rows_mod.Reader,
    kernels: rows_mod.Kernels,
    token_map: []i32,
    hasher: hash.Hasher,
    cols: [2]usize, // this rank's hash columns [lo, hi)
    max_rows: usize,
    /// One pinned staging pair an Engram layer; `uploaded[i]` marks when its last host-to-device copy ran, so the
    /// host rows are refilled only after it.
    staging: []rows_mod.Staging,
    uploaded: []cuda.Event,
    pending: []bool,
    hashes_h: []i64, // [max_rows, layers, cols]
    idx_h: []i64, // [max_rows, cols_r]
    e: cuda.DeviceBuffer, // bf16 [max_rows, cols_r * 256]
    xh: cuda.DeviceBuffer, // the decode linear's rotated input, fp16 [decode_rows, K]
    z: cuda.DeviceBuffer, // its split-K scratch
    partial: cuda.DeviceBuffer, // fp32 [max_rows, 5 d]
    gathered: cuda.DeviceBuffer, // fp32 [world, max_rows * 5 d]
    kv: cuda.DeviceBuffer, // bf16 [max_rows, 5 d]
    /// The sequence's ids from position 0 (set by the caller before each run; covers every row it runs).
    history: []const i32 = &.{},
    probe: ?forward.Probe = null,
    /// Read-ahead batches (`prefetch`): two buffer sets, the pending batches in order in `fifo` (at most two), the
    /// background reader's queue (`queue[i % 2]` is the set of the i-th posted batch) and counters.
    sets: [2]Batch,
    fifo: [2]u8 = .{ 0, 0 },
    fifo_len: usize = 0,
    queue: [2]u8 = .{ 0, 0 },
    posted: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    finished: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    thread: ?std.Thread = null,
    /// The Engram layers this rank's weights hold (their slots in cfg.engram_layers).
    present: []usize,

    /// `dir`: the Engram shards (`layers.L.engram.embed.{weight,scale}`); `token_map_path`: the Python engine's
    /// compressed token map JSON. `max_rows`: the most rows a call runs. `threads`: the reader pool's workers.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, cfg: *const Config, w: *const weights.Weights, set: *const cuda.aot.Set, dir: []const u8, token_map_path: []const u8, max_rows: usize, threads: usize) !*Engram {
        const self = try gpa.create(Engram);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.d = d;
        self.cfg = cfg;
        self.w = w;
        self.max_rows = max_rows;
        self.history = &.{};
        self.probe = null;
        self.hc = hcmod.Hc.init(set, @intCast(cfg.dim), @intCast(cfg.hc_iters), cfg.eps, cfg.hc_eps);
        self.token_map = try hash.loadTokenMap(gpa, io, token_map_path);
        errdefer gpa.free(self.token_map);
        if (self.token_map.len != cfg.vocab) return error.BadTokenMap;
        self.hasher = try hash.Hasher.init(.{
            .layers = cfg.engram_layers.slice(),
            .rows = cfg.engram_rows.slice(),
            .ngram = cfg.engram_ngram,
            .heads = cfg.engram_heads,
            .vocab = cfg.engram_vocab,
            .cvocab = cfg.engram_cvocab,
            .pad_token = cfg.engram_pad,
        }, self.token_map, null);
        self.cols = self.hasher.rankColumns(w.rank, w.world);
        self.tables = try rows_mod.Tables.open(gpa, io, dir);
        errdefer self.tables.close();
        for (cfg.engram_layers.slice()) |li| {
            if (li < @as(i64, @intCast(w.layers.len)) and self.tables.get(li) == null) return error.NoTable;
        }
        self.reader = try rows_mod.Reader.init(gpa, threads);
        errdefer self.reader.deinit();
        self.kernels = try rows_mod.Kernels.load(d);
        errdefer self.kernels.deinit();
        const nl = cfg.engram_layers.len;
        const per = self.cols[1] - self.cols[0];
        const kvw = (hcmod.hc_streams + 1) * cfg.dim;
        self.staging = try gpa.alloc(rows_mod.Staging, nl);
        errdefer gpa.free(self.staging);
        self.uploaded = try gpa.alloc(cuda.Event, nl);
        errdefer gpa.free(self.uploaded);
        self.pending = try gpa.alloc(bool, nl);
        errdefer gpa.free(self.pending);
        @memset(self.pending, false);
        var made: usize = 0;
        errdefer for (0..made) |i| {
            self.staging[i].deinit();
            self.uploaded[i].deinit();
        };
        while (made < nl) : (made += 1) {
            self.staging[made] = try rows_mod.Staging.init(d, max_rows * per);
            self.uploaded[made] = cuda.Event.init(d, false) catch |err| {
                self.staging[made].deinit();
                return err;
            };
        }
        self.hashes_h = try gpa.alloc(i64, max_rows * nl * self.hasher.cols);
        errdefer gpa.free(self.hashes_h);
        self.idx_h = try gpa.alloc(i64, max_rows * per);
        errdefer gpa.free(self.idx_h);
        // the decode linear's scratch: the widest Engram projection at decode_rows
        var kmax: usize = 0;
        var zw: usize = 16;
        for (w.layers) |*l| if (l.engram_wkv) |*lin| {
            kmax = @max(kmax, lin.k);
            zw = @max(zw, lin.scratchWords(decode_rows));
        };
        self.e = try cuda.DeviceBuffer.alloc(d, max_rows * per * rows_mod.row_w * 2);
        errdefer self.e.free();
        self.xh = try cuda.DeviceBuffer.alloc(d, @max(16, decode_rows * kmax * 2));
        errdefer self.xh.free();
        self.z = try cuda.DeviceBuffer.alloc(d, zw * 4);
        errdefer self.z.free();
        self.partial = try cuda.DeviceBuffer.alloc(d, max_rows * kvw * 4);
        errdefer self.partial.free();
        self.gathered = try cuda.DeviceBuffer.alloc(d, w.world * max_rows * kvw * 4);
        errdefer self.gathered.free();
        self.kv = try cuda.DeviceBuffer.alloc(d, max_rows * kvw * 2);
        errdefer self.kv.free();
        var present: std.ArrayList(usize) = .empty;
        defer present.deinit(gpa);
        for (cfg.engram_layers.slice(), 0..) |li, slot| if (li < @as(i64, @intCast(w.layers.len))) try present.append(gpa, slot);
        self.present = try present.toOwnedSlice(gpa);
        errdefer gpa.free(self.present);
        self.fifo = .{ 0, 0 };
        self.fifo_len = 0;
        self.queue = .{ 0, 0 };
        self.posted = std.atomic.Value(u32).init(0);
        self.finished = std.atomic.Value(u32).init(0);
        self.stop = std.atomic.Value(bool).init(false);
        self.thread = null;
        self.sets[0] = try Batch.init(gpa, d, nl, max_rows * per);
        errdefer self.sets[0].deinit(gpa);
        self.sets[1] = try Batch.init(gpa, d, nl, max_rows * per);
        return self;
    }

    pub fn deinit(self: *Engram) void {
        if (self.thread) |t| {
            self.stop.store(true, .release);
            t.join();
        }
        for (&self.sets) |*b| b.deinit(self.gpa);
        self.gpa.free(self.present);
        for (self.uploaded, self.pending) |ev, p| if (p) ev.synchronize() catch {};
        inline for (.{ "e", "xh", "z", "partial", "gathered", "kv" }) |f| @field(self, f).free();
        for (self.staging) |*s| s.deinit();
        for (self.uploaded) |*ev| ev.deinit();
        self.gpa.free(self.staging);
        self.gpa.free(self.uploaded);
        self.gpa.free(self.pending);
        self.gpa.free(self.hashes_h);
        self.gpa.free(self.idx_h);
        self.kernels.deinit();
        self.reader.deinit();
        self.tables.close();
        self.gpa.free(self.token_map);
        self.gpa.destroy(self);
    }

    /// The `Forward.engram` hook running `apply` on `history`.
    pub fn hook(self: *Engram) forward.Engram {
        return .{ .ctx = self, .call = hookFn };
    }

    fn hookFn(ctx: *anyopaque, layer: usize, h: Ptr, start: usize, n: usize, mm: ?forward.Mm, gather: forward.Gather, stream: cuda.Stream) anyerror!void {
        const self: *Engram = @ptrCast(@alignCast(ctx));
        try self.apply(layer, h, self.history, start, n, mm, gather, stream);
    }

    /// The `RoundRunner.engram` hook: every stream's rows of a round with their own histories (`applySpans`), read
    /// now; a prompt's read-ahead batches stay pending for its next chunk (a round may run between two chunks).
    pub fn roundHook(self: *Engram) rounds.Engram {
        return .{ .ctx = self, .call = roundFn };
    }

    fn roundFn(ctx: *anyopaque, layer: usize, h: Ptr, spans: []const rounds.EngramSpan, gather: forward.Gather, stream: cuda.Stream) anyerror!void {
        const self: *Engram = @ptrCast(@alignCast(ctx));
        try self.spansRead(layer, h, spans, null, gather, stream, false);
    }

    /// h [n, 4, d] bf16 (positions start .. start + n - 1 of the sequence `ids`, which holds at least start + n ids)
    /// <- engram_gate(h, kv, qk) of layer `layer`, in place, on `stream`. `mm` null: the decode linear (n <= 16).
    pub fn apply(self: *Engram, layer: usize, h: Ptr, ids: []const i32, start: usize, n: usize, mm: ?forward.Mm, gather: forward.Gather, stream: cuda.Stream) !void {
        try self.applySpans(layer, h, &.{.{ .ids = ids, .start = start, .n = n }}, mm, gather, stream);
    }

    /// `apply` over rows of several sequences (a round, rounds.py `forward`): h's rows are the spans' rows in span
    /// order, each span's n-gram ids hashed from its own sequence at its own positions; one projection, one gather and
    /// one gate over all of them. The bits of a row are those `apply` gives it alone (the decode linear and the gate
    /// are row-invariant).
    pub fn applySpans(self: *Engram, layer: usize, h: Ptr, spans: []const rounds.EngramSpan, mm: ?forward.Mm, gather: forward.Gather, stream: cuda.Stream) !void {
        try self.spansRead(layer, h, spans, mm, gather, stream, true);
    }

    /// `applySpans`, its rows taken from a read-ahead batch only with `ahead` (else read now, the batches kept).
    fn spansRead(self: *Engram, layer: usize, h: Ptr, spans: []const rounds.EngramSpan, mm: ?forward.Mm, gather: forward.Gather, stream: cuda.Stream, ahead: bool) !void {
        var n: usize = 0;
        for (spans) |sp| {
            if (sp.ids.len < sp.start + sp.n) return error.HistoryShort;
            n += sp.n;
        }
        if (n < 1 or n > self.max_rows or (mm == null and n > decode_rows)) return error.TooManyRows;
        if (layer >= self.w.layers.len) return error.NotEngramLayer;
        const l = &self.w.layers[layer];
        const wkv: *const exl3.Linear = if (l.engram_wkv) |*x| x else return error.NotEngramLayer;
        const qk = l.engram_qk orelse return error.NotEngramLayer;
        const slot = self.cfg.engram_layers.indexOf(@intCast(layer)) orelse return error.NotEngramLayer;
        const table = self.tables.get(@intCast(layer)) orelse return error.NoTable;
        const nl = self.hasher.n_layers;
        const cols = self.hasher.cols;
        const lo = self.cols[0];
        const per = self.cols[1] - lo;
        const kvw = (hcmod.hc_streams + 1) * self.cfg.dim;

        // the row ids of this rank's columns (Engram.hashes(...)[:, layer, lo:hi], the spans' concatenated)
        var at: usize = 0;
        for (spans) |sp| {
            try self.hasher.hashes(sp.ids, sp.start, sp.n, self.hashes_h[at * nl * cols .. (at + sp.n) * nl * cols]);
            at += sp.n;
        }
        for (0..n) |p| for (0..per) |c| {
            self.idx_h[p * per + c] = self.hashes_h[(p * nl + slot) * cols + lo + c];
        };
        // the rows: a prefetched batch's (read ahead into its own pinned buffers), else read now into this layer's
        // pinned staging (its previous copy to the device has run); then to the device
        const taken = if (ahead) try self.takeAhead(slot, spans, n, per, stream) else null;
        const st = if (taken) |batch| batch else blk: {
            try self.idleAhead();
            const st = &self.staging[slot];
            if (self.pending[slot]) try self.uploaded[slot].synchronize();
            self.pending[slot] = false;
            try self.reader.gather(table, self.idx_h[0 .. n * per], st.codes(n * per), st.scales(n * per));
            try st.upload(n * per, stream);
            try self.uploaded[slot].record(stream);
            self.pending[slot] = true;
            break :blk st;
        };
        try self.kernels.decode(stream, st.dev_w.ptr, st.dev_s.ptr, self.e.ptr, n * per);
        // this rank's partial, every rank's in rank order, their sum to bf16, the gate
        if (mm) |f| {
            try f.call(f.ctx, wkv, self.e.ptr, n, self.partial.ptr, stream);
        } else {
            try wkv.run(self.e.ptr, .bf16, n, self.xh.ptr, self.z.ptr, self.partial.ptr, .f32, stream);
        }
        if (self.probe) |p| {
            var nb: [32]u8 = undefined;
            try p.call(p.ctx, try std.fmt.bufPrint(&nb, "L{d}.p_engram.r0", .{layer}), self.partial.ptr, n * kvw * 4, stream);
        }
        try gather.run(self.partial.ptr, n * kvw, self.gathered.ptr, stream);
        try self.kernels.sumToBf16(stream, self.gathered.ptr, @intCast(self.w.world), n * kvw, self.kv.ptr);
        try self.hc.engramGate(stream, h, self.kv.ptr, qk.ptr(), h, @intCast(n));
    }

    // -- read-ahead ------------------------------------------------------------------------------------------------

    /// Starts reading the rows of positions start .. start + n - 1 of `ids` (the sequence's ids from position 0) for
    /// every Engram layer, on the background reader; the chunk's calls at those positions take them (`applySpans`).
    /// Does nothing when two batches are already pending or n exceeds `max_rows`.
    pub fn prefetch(self: *Engram, ids: []const i32, start: usize, n: usize) !void {
        if (n < 1 or n > self.max_rows or self.fifo_len == 2 or self.present.len == 0) return;
        if (ids.len < start + n) return error.HistoryShort;
        const k: u8 = if (self.fifo_len == 1) 1 - self.fifo[0] else 0;
        const b = &self.sets[k];
        // the set's last uploads have run (its host rows are about to be overwritten)
        for (b.pending, b.uploaded) |*p, ev| if (p.*) {
            try ev.synchronize();
            p.* = false;
        };
        const nl = self.hasher.n_layers;
        const cols = self.hasher.cols;
        const lo = self.cols[0];
        const per = self.cols[1] - lo;
        try self.hasher.hashes(ids, start, n, self.hashes_h[0 .. n * nl * cols]);
        for (self.present) |slot| {
            const idx = b.idx[slot * b.rows ..][0 .. n * per];
            for (0..n) |p| for (0..per) |c| {
                idx[p * per + c] = self.hashes_h[(p * nl + slot) * cols + lo + c];
            };
        }
        b.start = start;
        b.n = n;
        @memset(b.consumed, true);
        for (self.present) |slot| {
            b.consumed[slot] = false;
            b.ready[slot].store(0, .release);
        }
        self.fifo[self.fifo_len] = k;
        self.fifo_len += 1;
        const at = self.posted.load(.acquire);
        self.queue[at % 2] = k;
        self.posted.store(at + 1, .release);
        if (self.thread == null) self.thread = try std.Thread.spawn(.{}, aheadLoop, .{self});
    }

    /// The oldest pending batch's rows of layer `slot`, uploaded on `stream`, when that batch is for this call (one
    /// span at its positions with its row ids `idx_h`); pending batches that are not are dropped. Null: read now.
    fn takeAhead(self: *Engram, slot: usize, spans: []const rounds.EngramSpan, n: usize, per: usize, stream: cuda.Stream) !?*rows_mod.Staging {
        if (self.fifo_len == 0) return null;
        const b = &self.sets[self.fifo[0]];
        const fits = spans.len == 1 and spans[0].start == b.start and n == b.n and !b.consumed[slot];
        if (fits) {
            var spins: u32 = 0;
            while (b.ready[slot].load(.acquire) == 0) {
                spins += 1;
                if (spins < 1000) std.atomic.spinLoopHint() else sleepUs(20);
            }
            if (b.ready[slot].load(.acquire) == 2) return error.ReadFailed;
        }
        if (!fits or !std.mem.eql(i64, b.idx[slot * b.rows ..][0 .. n * per], self.idx_h[0 .. n * per])) {
            try self.idleAhead();
            self.fifo_len = 0;
            return null;
        }
        const st = &b.staging[slot];
        try st.upload(n * per, stream);
        try b.uploaded[slot].record(stream);
        b.pending[slot] = true;
        b.consumed[slot] = true;
        if (std.mem.indexOfScalar(bool, b.consumed, false) == null) {
            self.fifo[0] = self.fifo[1];
            self.fifo_len -= 1;
        }
        return st;
    }

    /// Drops every pending batch (a prompt's admission starts from none: an earlier one may have stopped part way).
    pub fn resetAhead(self: *Engram) !void {
        try self.idleAhead();
        self.fifo_len = 0;
    }

    /// Waits until the background reader has finished every posted batch (the reader pool is then this thread's).
    fn idleAhead(self: *Engram) !void {
        var spins: u32 = 0;
        while (self.finished.load(.acquire) != self.posted.load(.acquire)) {
            spins += 1;
            if (spins < 1000) std.atomic.spinLoopHint() else sleepUs(20);
        }
    }

    fn aheadLoop(self: *Engram) void {
        var done: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (self.posted.load(.acquire) == done) {
                if (self.stop.load(.acquire)) return;
                spins += 1;
                if (spins < 1000) std.atomic.spinLoopHint() else sleepUs(50);
            }
            const b = &self.sets[self.queue[done % 2]];
            const per = self.cols[1] - self.cols[0];
            for (self.present) |slot| {
                const table = self.tables.get(self.cfg.engram_layers.slice()[slot]).?;
                const st = &b.staging[slot];
                const ok = if (self.reader.gather(table, b.idx[slot * b.rows ..][0 .. b.n * per], st.codes(b.n * per), st.scales(b.n * per))) |_| true else |_| false;
                b.ready[slot].store(if (ok) 1 else 2, .release);
            }
            done += 1;
            self.finished.store(done, .release);
        }
    }

};

/// A read-ahead batch's buffers: a pinned staging pair and its upload's event a layer, the row ids of every layer
/// (`rows` a layer), and the batch's positions.
const Batch = struct {
    rows: usize,
    start: usize = 0,
    n: usize = 0,
    idx: []i64,
    staging: []rows_mod.Staging,
    uploaded: []cuda.Event,
    pending: []bool, // an upload from the layer's pinned rows is queued
    ready: []std.atomic.Value(u8), // the layer's rows: 0 reading, 1 read, 2 the read failed
    consumed: []bool,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, layers: usize, rows: usize) !Batch {
        var b: Batch = .{ .rows = rows, .idx = &.{}, .staging = &.{}, .uploaded = &.{}, .pending = &.{}, .ready = &.{}, .consumed = &.{} };
        errdefer b.deinit(gpa);
        b.idx = try gpa.alloc(i64, layers * rows);
        b.pending = try gpa.alloc(bool, layers);
        @memset(b.pending, false);
        b.consumed = try gpa.alloc(bool, layers);
        @memset(b.consumed, true);
        b.ready = try gpa.alloc(std.atomic.Value(u8), layers);
        for (b.ready) |*r| r.* = std.atomic.Value(u8).init(0);
        const staging = try gpa.alloc(rows_mod.Staging, layers);
        const events = gpa.alloc(cuda.Event, layers) catch |err| {
            gpa.free(staging);
            return err;
        };
        var made: usize = 0;
        errdefer {
            for (staging[0..made], events[0..made]) |*st, *ev| {
                st.deinit();
                ev.deinit();
            }
            gpa.free(staging);
            gpa.free(events);
        }
        while (made < layers) : (made += 1) {
            staging[made] = try rows_mod.Staging.init(d, rows);
            events[made] = cuda.Event.init(d, false) catch |err| {
                staging[made].deinit();
                return err;
            };
        }
        b.staging = staging;
        b.uploaded = events;
        return b;
    }

    fn deinit(self: *Batch, gpa: std.mem.Allocator) void {
        for (self.uploaded, self.pending) |ev, p| if (p) ev.synchronize() catch {};
        for (self.staging) |*st| st.deinit();
        for (self.uploaded) |*ev| ev.deinit();
        if (self.staging.len > 0) gpa.free(self.staging);
        if (self.uploaded.len > 0) gpa.free(self.uploaded);
        if (self.idx.len > 0) gpa.free(self.idx);
        if (self.pending.len > 0) gpa.free(self.pending);
        if (self.consumed.len > 0) gpa.free(self.consumed);
        if (self.ready.len > 0) gpa.free(self.ready);
        self.* = .{ .rows = 0, .idx = &.{}, .staging = &.{}, .uploaded = &.{}, .pending = &.{}, .ready = &.{}, .consumed = &.{} };
    }
};

fn sleepUs(us: u32) void {
    var req: std.c.timespec = .{ .sec = 0, .nsec = @intCast(us * 1000) };
    _ = std.c.nanosleep(&req, null);
}
