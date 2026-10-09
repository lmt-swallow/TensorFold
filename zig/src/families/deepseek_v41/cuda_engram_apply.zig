//! Engram on one rank (the Python fork's `Model.engram_apply` with `model.py Engram`): for an Engram layer's rows at
//! positions start .. start + n - 1 of a sequence, the n-gram row ids from the host token history (engram_hash.zig),
//! this rank's columns of them read from the FP8 shards and decoded to bf16 `e [n, cols_r * 256]`
//! (engram_rows.zig), the rank's `engram_wkv` partial `e @ W_r` in fp32 [n, 5 d], the ranks' partials gathered and
//! summed in rank order to bf16 kv [n, 5 d] (Comm.sum(...).to(BF16)), then `engram_gate` on h [n, 4, d] in place.
//!
//! Every rank runs the same steps on the same host ids (the hashes are deterministic), so the gather pairs. The hook
//! (`hook()`) plugs into `Forward.engram`; `history` must hold the sequence's ids from position 0 through the rows
//! being run (negative ids mark an image span: no n-gram reaches into one).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const exl3 = @import("cuda_exl3.zig");
const hcmod = @import("cuda_hc.zig");
const forward = @import("cuda_forward.zig");
const hash = @import("engram_hash.zig");
const rows_mod = @import("engram_rows.zig");

const Ptr = cuda.abi.DevicePtr;

pub const Error = error{ NotEngramLayer, NoTable, TooManyRows, HistoryShort, BadTokenMap };

/// Rows at most a call without an `mm` (the row-invariant decode linear's scratch is sized for these).
pub const decode_rows: usize = 16;

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
        return self;
    }

    pub fn deinit(self: *Engram) void {
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

    /// h [n, 4, d] bf16 (positions start .. start + n - 1 of the sequence `ids`, which holds at least start + n ids)
    /// <- engram_gate(h, kv, qk) of layer `layer`, in place, on `stream`. `mm` null: the decode linear (n <= 16).
    pub fn apply(self: *Engram, layer: usize, h: Ptr, ids: []const i32, start: usize, n: usize, mm: ?forward.Mm, gather: forward.Gather, stream: cuda.Stream) !void {
        if (n < 1 or n > self.max_rows or (mm == null and n > decode_rows)) return error.TooManyRows;
        if (ids.len < start + n) return error.HistoryShort;
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

        // the row ids of this rank's columns (Engram.hashes(...)[:, layer, lo:hi])
        try self.hasher.hashes(ids, start, n, self.hashes_h[0 .. n * nl * cols]);
        for (0..n) |p| for (0..per) |c| {
            self.idx_h[p * per + c] = self.hashes_h[(p * nl + slot) * cols + lo + c];
        };
        // the rows into this layer's pinned staging (its previous copy to the device has run), then to the device
        const st = &self.staging[slot];
        if (self.pending[slot]) try self.uploaded[slot].synchronize();
        self.pending[slot] = false;
        try self.reader.gather(table, self.idx_h[0 .. n * per], st.codes(n * per), st.scales(n * per));
        try st.upload(n * per, stream);
        try self.uploaded[slot].record(stream);
        self.pending[slot] = true;
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
};
