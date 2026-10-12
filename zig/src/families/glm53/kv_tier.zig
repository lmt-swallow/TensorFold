//! The DSA latent cache tiered to local NVMe (GLM-5.3-Flash-1: four streams of 1,048,576 positions in bf16 do not
//! fit beside the weights). The host half: which pools are resident, where, and the reads and writes that move them.
//! No CUDA here: the slab is behind `Slab` (cuda_tier keeps it in device memory, the tests in a host slice); the
//! map is the caller's (cuda_tier: mapped pinned memory the device's remap reads).
//!
//! A pool is 4 consecutive positions of one stream's layer (sparse.py's selection unit): 4 rows of `row_bytes`, one
//! 4 KiB block on the file and one slot of the slab. Every position a call can see is resident by the time the call
//! runs, at an address the unchanged kernels reach:
//! - `writeRuns` gives the latent write (and dense attention) a base at which the window's positions are contiguous:
//!   the stream-layer's home run (positions [0, home_positions), one block of runs) or the window's segment run (2,048
//!   positions); segments of the recent window stay such linear runs.
//! - `fetch` makes every pool a selection names resident; `remap` turns each selected position into
//!   `slot * 4 + position % 4`, the row index a kernel reads at the slab's base. The kernels take positions only as
//!   addresses (latent.py `_sparse_chunks`, `_sparse_onepass`: `LC + tok * RS`), so they read the same bytes in the
//!   same order as from a whole cache: the bits do not change.
//! - `commit` writes the pools wholly below the committed position to the file (one 4 KiB block a pool, contiguous
//!   pools as one write); only written pools are evicted, so an eviction never waits for I/O and never loses a row.
//!
//! Properties (tests below): after any sequence of windows, writes, commits, rewinds and fetches, the slab bytes at a
//! remapped position equal the bytes last written at that position; a call's selected pools are never evicted by its
//! own fetch; the map and the slots' owners name each other.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

pub const pool_rows = 4;
pub const block = 4096;
/// Pools a run (a segment) holds: 2,048 positions, the prompt's chunk.
pub const run_pools = 512;
pub const seg_positions = run_pools * pool_rows;
/// Segments of the home block: dense attention (rows before cuda_dsa's dense limit 2,051, windows of up to 2,048
/// rows) reads positions below 4,099 at one base.
pub const home_runs = 3;
pub const home_positions = home_runs * seg_positions;
pub const home_pools = home_positions / pool_rows;
/// Streams a chain of aliases reads through at most (a stream and its ancestors; cuda_tier's remap takes as many).
pub const max_chain = 8;

pub const Error = error{ SlabFailed, FileBusy, TierFull, BadToken, NotPersisted, NotOpen, PastCapacity, OpenFailed, ReadFailed, WriteFailed, BadShape, OutOfMemory, NoStream, ChainTooDeep };

pub const Options = struct {
    pub fn mapBlocks(o: Options) usize {
        return if (o.blocks != 0) o.blocks else o.streams * o.layers;
    }

    streams: usize,
    layers: usize,
    /// Positions a stream holds at most (a multiple of seg_positions).
    capacity: usize,
    /// A latent row's bytes (bf16 kv_lora: 1,024); a pool's (4 rows) must be one block.
    row_bytes: usize = 1024,
    /// Segments past the home block kept linear behind the newest one written.
    recent_runs: usize = 32,
    /// Map blocks (one an open stream-layer: capacity / 4 i32 and their file region) held at once; 0: every
    /// stream-layer at once (streams * layers). The map and file grow with this, not with `streams`.
    blocks: usize = 0,
    /// Keep each slot's last `trace_depth` events (what put bytes in it, for which pool, in which window): a
    /// mismatch then names its history (`history`). Off in serving.
    trace: bool = false,
    /// Committed pools a stream lets accumulate before writing them out together (one slab gather and one wait
    /// for the drive instead of one a stretch); 0 writes at every commit. Pools waiting are resident and only
    /// written ones are evicted, so a slab that runs short writes every stream's out first.
    commit_batch: usize = 512,
    /// Windows a selected pool stays out of the first eviction pass.
    hot_windows: u32 = 2,
    /// Reads and writes in flight at most a lane (Linux AIO), and lanes (threads) a large batch spreads over.
    depth: usize = 64,
    lanes: usize = 4,
};

const none: u32 = std.math.maxInt(u32);
const Kind = enum(u8) { free, linear, loose, taken }; // taken: handed out by takeRuns, linear next
const Run = struct { kind: Kind = .free, key0: u32 = 0, held: u32 = 0 }; // held: slots owning a pool

pub const Piece = struct {
    /// Byte offset from the slab's start of position 0's row (position p's row is at base + p * row_bytes); may be
    /// negative, as a pointer it wraps.
    base: i64,
    pos: usize,
    rows: usize,
};

/// The slab's slots, wherever they live. kv_tier never touches a slot's bytes but through these three calls:
/// cuda_tier keeps the slab in device memory (never visible to the CPU), the host tests in a host slice
/// (`HostSlab`). Each call returns once its bytes landed.
pub const Slab = struct {
    ctx: *anyopaque,
    slots: usize,
    /// bytes[i * block ..][0..block] into slot slots[i]
    put: *const fn (ctx: *anyopaque, slots: []const u32, bytes: []const u8) anyerror!void,
    /// slot slots[i]'s bytes into out[i * block ..][0..block]
    get: *const fn (ctx: *anyopaque, slots: []const u32, out: []u8) anyerror!void,
    /// slot dst[i] := slot src[i]
    copy: *const fn (ctx: *anyopaque, dst: []const u32, src: []const u32) anyerror!void,
};

/// A slab in a host slice (the host tests; a kernel never reads it).
pub const HostSlab = struct {
    bytes: []align(block) u8,

    pub fn slab(h: *HostSlab) Slab {
        return .{ .ctx = h, .slots = h.bytes.len / block, .put = put, .get = get, .copy = copy };
    }

    fn put(ctx: *anyopaque, slots: []const u32, bytes: []const u8) anyerror!void {
        const h: *HostSlab = @ptrCast(@alignCast(ctx));
        for (slots, 0..) |x, i| @memcpy(h.bytes[@as(usize, x) * block ..][0..block], bytes[i * block ..][0..block]);
    }

    fn get(ctx: *anyopaque, slots: []const u32, out: []u8) anyerror!void {
        const h: *HostSlab = @ptrCast(@alignCast(ctx));
        for (slots, 0..) |x, i| @memcpy(out[i * block ..][0..block], h.bytes[@as(usize, x) * block ..][0..block]);
    }

    fn copy(ctx: *anyopaque, dst: []const u32, src: []const u32) anyerror!void {
        const h: *HostSlab = @ptrCast(@alignCast(ctx));
        for (dst, src) |d, x| @memcpy(h.bytes[@as(usize, d) * block ..][0..block], h.bytes[@as(usize, x) * block ..][0..block]);
    }
};

/// What happened to a slot (trace).
pub const EvKind = enum(u8) { none, file_read, copy_in, move_in, move_out, gpu_write, file_write, evict, drop, load, alias_copy, place_empty };
pub const Ev = struct { kind: EvKind = .none, key: u32 = 0, window: u32 = 0, gen: u32 = 0 };
pub const trace_depth = 8;

pub const Tier = struct {
    gpa: std.mem.Allocator,
    o: Options,
    slab: Slab,
    map: []i32,
    // reads pending their slab put: rd_slots[i]'s bytes land at rd_stage[i * block ..] when the file's do
    rd_slots: std.ArrayList(u32) = .empty,
    rd_stage: []align(block) u8 = &.{},
    wr_stage: []align(block) u8 = &.{}, // a commit's pools on their way to the file // [streams, layers, pools]: the slot holding a pool, -1 where none does
    pools: usize, // a stream-layer's
    segs: usize,
    runs: []Run,
    seg_run: []u32, // [streams, layers, segs]: the linear run holding a segment
    home: []u32, // [streams, layers]: the home block's first run
    block_of: []u32, // [streams, layers]: the map block an open stream-layer holds
    block_user: []u32, // [blocks]: the stream-layer (s * layers + l) holding a block, none if free
    away: []u32, // [streams, layers]: written pools not resident (0: every selection is resident, fetch reads nothing)
    owner: []u32, // [slots]: the key (map index) a slot holds
    pin: []u32, // [slots]: the fetch call that last named it
    used: []u32, // [slots]: the window that last named it
    ref: std.DynamicBitSetUnmanaged, // [slots]: the clock's second chance
    persisted: std.DynamicBitSetUnmanaged, // [keys]: the file holds the pool's bytes
    free_slots: std.ArrayList(u32) = .empty, // loose slots holding nothing
    streams: []Stream,
    hand: usize = 0,
    window: u32 = 1,
    gen: u32 = 1,
    io: Io,
    trace_ev: []Ev = &.{}, // [slots * trace_depth], a ring a slot (Options.trace)
    reads: u64 = 0, // pools read from the file so far (statistics)
    flushing: bool = false, // allocLoose is writing waiting pools out
    write_calls: u64 = 0, // slab gets and file write waits a commit made (statistics: each is a device round trip)
    writes: u64 = 0,

    const Stream = struct {
        open: bool = false,
        layers: usize = 0, // layers 0 .. layers - 1 in use (the MTP head's cache: 1)
        hi: usize = 0, // newest segment a window touched
        end: usize = 0, // rows [0, end) hold the stream's rows (rewind lowers it); pools wholly past it are stale
        committed: usize = 0,
        dirty_lo: usize = 0, // pools below this are persisted wherever full and committed
        // an alias's: pools [home_pools, fork_pool) are the parent's (read through it, never written here)
        parent: ?u32 = null,
        fork_pool: usize = 0,
        refs: u32 = 0, // aliases reading through this stream
        pending: bool = false, // closed by its owner, kept for its aliases
        favored: bool = false, // its pools leave only after every other stream's could not (a prompt being filled)
    };

    /// `slab` (whole runs of run_pools slots) and `map` (Options.mapBlocks() * capacity / 4 entries, which the
    /// device's remap reads) are the caller's. `path` is the backing file (created, emptied, locked).
    pub fn init(gpa: std.mem.Allocator, slab: Slab, map: []i32, path: [:0]const u8, o: Options) Error!Tier {
        if (o.row_bytes * pool_rows != block or o.capacity % seg_positions != 0 or o.capacity < home_positions) return error.BadShape;
        if (slab.slots % run_pools != 0 or slab.slots == 0) return error.BadShape;
        const pools = o.capacity / pool_rows;
        const keys = o.mapBlocks() * pools;
        if (map.len != keys or keys >= none) return error.BadShape;
        const nslots = slab.slots;
        if (nslots * pool_rows >= std.math.maxInt(i32)) return error.BadShape;
        var t: Tier = undefined;
        t.gpa = gpa;
        t.o = o;
        t.slab = slab;
        t.map = map;
        t.pools = pools;
        t.segs = o.capacity / seg_positions;
        t.free_slots = .empty;
        t.rd_slots = .empty;
        t.rd_stage = try gpa.alignedAlloc(u8, .fromByteUnits(block), stage_blocks * block);
        errdefer gpa.free(t.rd_stage);
        t.wr_stage = try gpa.alignedAlloc(u8, .fromByteUnits(block), stage_blocks * block);
        errdefer gpa.free(t.wr_stage);
        t.hand = 0;
        t.window = 1;
        t.gen = 1;
        t.reads = 0;
        t.writes = 0;
        t.write_calls = 0;
        t.flushing = false;
        @memset(map, -1);
        t.runs = try gpa.alloc(Run, nslots / run_pools);
        errdefer gpa.free(t.runs);
        @memset(t.runs, .{});
        t.seg_run = try gpa.alloc(u32, o.streams * o.layers * t.segs);
        errdefer gpa.free(t.seg_run);
        @memset(t.seg_run, none);
        t.home = try gpa.alloc(u32, o.streams * o.layers);
        errdefer gpa.free(t.home);
        @memset(t.home, none);
        t.away = try gpa.alloc(u32, o.mapBlocks());
        errdefer gpa.free(t.away);
        @memset(t.away, 0);
        t.block_of = try gpa.alloc(u32, o.streams * o.layers);
        errdefer gpa.free(t.block_of);
        @memset(t.block_of, none);
        t.block_user = try gpa.alloc(u32, o.mapBlocks());
        errdefer gpa.free(t.block_user);
        @memset(t.block_user, none);
        t.owner = try gpa.alloc(u32, nslots);
        errdefer gpa.free(t.owner);
        @memset(t.owner, none);
        t.pin = try gpa.alloc(u32, nslots);
        errdefer gpa.free(t.pin);
        @memset(t.pin, 0);
        t.used = try gpa.alloc(u32, nslots);
        errdefer gpa.free(t.used);
        @memset(t.used, 0);
        t.ref = try .initEmpty(gpa, nslots);
        errdefer t.ref.deinit(gpa);
        t.persisted = try .initEmpty(gpa, keys);
        errdefer t.persisted.deinit(gpa);
        t.streams = try gpa.alloc(Stream, o.streams);
        errdefer gpa.free(t.streams);
        @memset(t.streams, .{});
        t.trace_ev = &.{};
        if (o.trace) {
            t.trace_ev = try gpa.alloc(Ev, nslots * trace_depth);
            @memset(t.trace_ev, .{});
        }
        t.io = try Io.open(gpa, path, o.depth, o.lanes);
        return t;
    }

    pub fn deinit(t: *Tier) void {
        t.io.close();
        t.rd_slots.deinit(t.gpa);
        t.gpa.free(t.rd_stage);
        t.gpa.free(t.wr_stage);
        if (t.trace_ev.len > 0) t.gpa.free(t.trace_ev);
        t.gpa.free(t.streams);
        t.persisted.deinit(t.gpa);
        t.ref.deinit(t.gpa);
        t.gpa.free(t.used);
        t.gpa.free(t.pin);
        t.gpa.free(t.owner);
        t.gpa.free(t.home);
        t.gpa.free(t.away);
        t.gpa.free(t.block_of);
        t.gpa.free(t.block_user);
        t.gpa.free(t.seg_run);
        t.gpa.free(t.runs);
        t.free_slots.deinit(t.gpa);
    }

    /// The map index of stream `s`'s layer `l`'s pool 0 (`map[mapBase(s, l) + position / 4]`).
    pub fn mapBase(t: *const Tier, s: usize, l: usize) usize {
        return @as(usize, t.block_of[s * t.o.layers + l]) * t.pools;
    }

    /// The map block (and away count) of stream `s`'s layer `l`.
    fn blockOf(t: *const Tier, s: usize, l: usize) usize {
        return t.block_of[s * t.o.layers + l];
    }

    fn key(t: *const Tier, s: usize, l: usize, pool: usize) u32 {
        return @intCast(t.mapBase(s, l) + pool);
    }

    /// The key holding stream `s`'s layer `l`'s pool `pool`: its own, or (an alias's pool below the fork, past the
    /// home block) an ancestor's.
    fn keyOf(t: *const Tier, s: usize, l: usize, pool: usize) u32 {
        var h = s;
        while (t.streams[h].parent) |par| {
            if (pool < home_pools or pool >= t.streams[h].fork_pool) break;
            h = par;
        }
        return t.key(h, l, pool);
    }

    /// A stream of a chain and its fork: below it (past the home block) the next stream's pools serve.
    pub const Link = struct { s: usize, fork_pool: usize };

    /// Stream `s`'s chain for the device remap: the streams whose pools it reads (itself first) and each one's fork
    /// (0 for the last); returns its length.
    pub fn chain(t: *const Tier, s: usize, out: *[max_chain]Link) usize {
        var h = s;
        var n: usize = 0;
        while (true) {
            out[n] = .{ .s = h, .fork_pool = if (t.streams[h].parent != null) t.streams[h].fork_pool else 0 };
            n += 1;
            h = t.streams[h].parent orelse break;
        }
        return n;
    }

    fn fileOffset(k: u32) u64 {
        return @as(u64, k) * block;
    }

    /// Blocks a read or write stage holds (a flush past it).
    const stage_blocks = 4096;

    /// The file's block `off` into slot `slot`: read into the stage now, put into the slab at the next flush.
    fn readSlot(t: *Tier, slot: u32, off: u64) Error!void {
        if (t.rd_slots.items.len == stage_blocks) try t.flushReads();
        const i = t.rd_slots.items.len;
        try t.rd_slots.append(t.gpa, slot);
        try t.io.add(.read, t.rd_stage[i * block ..][0..block], off);
    }

    /// Every staged read landed in its slot.
    fn flushReads(t: *Tier) Error!void {
        try t.io.finish();
        if (t.rd_slots.items.len == 0) return;
        t.slab.put(t.slab.ctx, t.rd_slots.items, t.rd_stage[0 .. t.rd_slots.items.len * block]) catch return error.SlabFailed;
        t.rd_slots.clearRetainingCapacity();
    }

    /// slot dst := slot src (staged reads landed first: src may be one of them).
    fn copySlot(t: *Tier, dst: u32, src: u32) Error!void {
        try t.flushReads();
        t.slab.copy(t.slab.ctx, &.{dst}, &.{src}) catch return error.SlabFailed;
    }

    /// Slot `slot`'s bytes into `out` (block long).
    pub fn getSlot(t: *Tier, slot: u32, out: []u8) Error!void {
        try t.flushReads();
        t.slab.get(t.slab.ctx, &.{slot}, out) catch return error.SlabFailed;
    }

    /// A stream of `layers` layers (at most Options.layers; the MTP head's cache is a stream of its own, 1 layer):
    /// its home blocks, one a layer; its previous contents are dropped.
    pub fn open(t: *Tier, s: usize, layers: usize) Error!void {
        if (layers == 0 or layers > t.o.layers) return error.BadShape;
        if (t.streams[s].pending) return error.NoStream;
        if (t.streams[s].open) t.close(s);
        // map blocks first (every key of the stream names one), then the home runs
        var free_n: usize = 0;
        for (t.block_user) |u| free_n += @intFromBool(u == none);
        if (free_n < layers) return error.NoStream;
        for (0..layers) |l| {
            const b = std.mem.indexOfScalar(u32, t.block_user, none).?;
            t.block_user[b] = @intCast(s * t.o.layers + l);
            t.block_of[s * t.o.layers + l] = @intCast(b);
        }
        t.streams[s] = .{ .open = true, .layers = layers };
        errdefer t.free(s);
        for (0..layers) |l| {
            const r = try t.takeRuns(home_runs);
            t.home[s * t.o.layers + l] = r;
            for (0..home_runs) |i| try t.placeLinear(s, l, i, r + @as(u32, @intCast(i)));
        }
    }

    /// A free stream (neither open nor kept for aliases) opened with `layers` layers.
    pub fn acquire(t: *Tier, layers: usize) Error!usize {
        for (t.streams, 0..) |st, s| if (!st.open and !st.pending) {
            try t.open(s, layers);
            return s;
        };
        return error.NoStream;
    }

    /// Drops a stream. One that aliases read through stays (pending) until the last of them closes: only the pools
    /// they read are kept, its home block and recent window let go.
    pub fn close(t: *Tier, s: usize) void {
        t.flushReads() catch {};
        const st = &t.streams[s];
        if (!st.open) return;
        if (st.refs > 0) {
            if (!st.pending) {
                st.pending = true;
                // the aliases copied the home block; the rest turns loose (written pools evictable)
                for (0..st.layers) |l| {
                    const k0 = t.mapBase(s, l);
                    for (0..home_pools) |p| t.drop(k0 + p);
                    const sr = t.seg_run[(s * t.o.layers + l) * t.segs ..][0..t.segs];
                    for (0..home_runs) |g| {
                        t.runs[sr[g]] = .{};
                        sr[g] = none;
                    }
                    for (home_runs..t.segs) |g| if (sr[g] != none) t.demote(s, sr, g);
                    t.home[s * t.o.layers + l] = none;
                }
            }
            return;
        }
        const parent = st.parent;
        t.free(s);
        if (parent) |par| {
            const ps = &t.streams[par];
            ps.refs -= 1;
            if (ps.refs == 0 and ps.pending) t.close(par);
        }
    }

    /// Key `k` forgotten: its slot (if any) free, its file copy no longer counted.
    fn drop(t: *Tier, k: usize) void {
        const slot = t.map[k];
        if (slot >= 0) {
            const u: usize = @intCast(slot);
            t.note(u, .drop, @intCast(k));
            t.disown(u);
            if (t.runs[u / run_pools].kind == .loose) t.free_slots.append(t.gpa, @intCast(u)) catch {};
        } else if (t.persisted.isSet(k)) t.away[k / t.pools] -= 1;
        t.map[k] = -1;
        t.persisted.unset(k);
    }

    fn free(t: *Tier, s: usize) void {
        for (0..t.streams[s].layers) |l| {
            const k0 = t.mapBase(s, l);
            for (0..t.pools) |p| t.drop(k0 + p);
            for (t.seg_run[(s * t.o.layers + l) * t.segs ..][0..t.segs]) |*r| {
                if (r.* != none) t.runs[r.*] = .{};
                r.* = none;
            }
            t.home[s * t.o.layers + l] = none;
            const b = t.blockOf(s, l);
            t.away[b] = 0;
            t.block_user[b] = none;
            t.block_of[s * t.o.layers + l] = none;
        }
        t.streams[s] = .{};
    }

    /// Before a window of R rows at `pos` runs on stream `s` (at a point where no call that reads the tier is in
    /// flight on the device): its segments are linear runs on every layer (their pools copied or read in), the
    /// segments behind the recent window turn loose, and the window's pools count as rewritten (not persisted).
    pub fn beginWindow(t: *Tier, s: usize, pos: usize, R: usize) Error!void {
        const st = &t.streams[s];
        if (!st.open) return error.NotOpen;
        if (R == 0 or R > seg_positions or pos + R > t.o.capacity) return error.PastCapacity;
        if (st.parent != null and pos < st.fork_pool * pool_rows) return error.BadShape; // the parent's rows
        try t.flushReads();
        t.window +%= 1;
        const first = pos / seg_positions;
        const last = (pos + R - 1) / seg_positions;
        st.hi = @max(st.hi, last);
        st.end = @max(st.end, pos + R);
        for (0..st.layers) |l| {
            const sr = t.seg_run[(s * t.o.layers + l) * t.segs ..][0..t.segs];
            for (home_runs..t.segs) |g| if (sr[g] != none and g < first and g + t.o.recent_runs <= st.hi) t.demote(s, sr, g);
        }
        for (0..st.layers) |l| {
            for (first..last + 1) |g| if (t.seg_run[(s * t.o.layers + l) * t.segs + g] == none) {
                const r = try t.takeRuns(1);
                try t.placeLinear(s, l, g, r);
            };
            const k0 = t.mapBase(s, l);
            for (pos / pool_rows..(pos + R - 1) / pool_rows + 1) |p| t.persisted.unset(k0 + p);
        }
        try t.flushReads();
        st.dirty_lo = @min(st.dirty_lo, pos / pool_rows);
        st.committed = @min(st.committed, pos);
    }

    /// Where the latent write puts rows pos .. pos + R - 1 of layer `l` (one piece, or two across a segment's end);
    /// the window's beginWindow ran.
    pub fn writeRuns(t: *const Tier, s: usize, l: usize, pos: usize, R: usize) Error![2]?Piece {
        var out: [2]?Piece = .{ null, null };
        var p = pos;
        var i: usize = 0;
        while (p < pos + R) : (i += 1) {
            if (i == 2) return error.BadShape;
            const g = p / seg_positions;
            if (g < home_runs) {
                const end = @min(pos + R, home_positions);
                out[i] = .{ .base = @intCast(@as(u64, t.home[s * t.o.layers + l]) * run_pools * block), .pos = p, .rows = end - p };
                p = end;
            } else {
                const r = t.seg_run[(s * t.o.layers + l) * t.segs + g];
                if (r == none) return error.NotOpen;
                const end = @min(pos + R, (g + 1) * seg_positions);
                const run_at: i64 = @intCast(@as(u64, r) * run_pools * block);
                out[i] = .{ .base = run_at - @as(i64, @intCast(g * seg_positions * t.o.row_bytes)), .pos = p, .rows = end - p };
                p = end;
            }
        }
        return out;
    }

    /// The base dense attention reads layer `l`'s positions [0, home_positions) at.
    pub fn denseBase(t: *const Tier, s: usize, l: usize) i64 {
        return @intCast(@as(u64, t.home[s * t.o.layers + l]) * run_pools * block);
    }

    /// One stream-layer's selection: rows [counts.len], row r's first counts[r] positions at tokens[r * W ..].
    pub const Selection = struct { s: usize, l: usize, tokens: []const i32, counts: []const i32, W: usize };

    /// Makes resident every pool the rows' selected positions name (`tokens [R, W]`, row r's first counts[r]);
    /// returns once their bytes are in the slab and the map names their slots. None of them is evicted before the
    /// next fetch.
    pub fn fetch(t: *Tier, s: usize, l: usize, tokens: []const i32, counts: []const i32, W: usize) Error!void {
        return t.fetchMany(&.{.{ .s = s, .l = l, .tokens = tokens, .counts = counts, .W = W }});
    }

    /// `fetch` of several selections (streams' or layers') as one batch of reads: the drive's latency is paid
    /// once. None of their pools is evicted before the next fetch.
    pub fn fetchMany(t: *Tier, sels: []const Selection) Error!void {
        try t.flushReads();
        t.gen +%= 1;
        var need: std.ArrayList(u32) = .empty;
        defer need.deinit(t.gpa);
        errdefer for (need.items) |k| if (t.map[k] == -2) {
            t.map[k] = -1;
        };
        for (sels) |x| {
            if (!t.streams[x.s].open) return error.NotOpen;
            if (x.l >= t.streams[x.s].layers) return error.BadShape;
            for (x.counts, 0..) |c, r| {
                if (c < 0 or @as(usize, @intCast(c)) > x.W) return error.BadToken;
                for (x.tokens[r * x.W ..][0..@intCast(c)]) |tok| {
                    if (tok < 0 or @as(usize, @intCast(tok)) >= t.o.capacity) return error.BadToken;
                    const k = t.keyOf(x.s, x.l, @as(usize, @intCast(tok)) / pool_rows);
                    const slot = t.map[k];
                    if (slot >= 0) {
                        t.touch(@intCast(slot));
                    } else if (slot == -1) {
                        if (!t.persisted.isSet(k)) return error.NotPersisted;
                        t.map[k] = -2; // named once
                        try need.append(t.gpa, k);
                    }
                }
            }
        }
        try t.readIn(need.items);
        try t.flushReads();
    }

    /// Whether every written pool of stream `s`'s layer `l` is resident: a fetch would read nothing (the device
    /// remap alone serves a selection).
    pub fn resident(t: *const Tier, s: usize, l: usize) bool {
        var h = s;
        while (true) {
            if (t.away[t.blockOf(h, l)] != 0) return false;
            h = t.streams[h].parent orelse return true;
        }
    }

    /// Starts reading layer `l`'s pools below position `end` that are not resident (prefill's next layer); the
    /// next call on the tier waits for them. `ensureRange` waits at once.
    pub fn prefetchRange(t: *Tier, s: usize, l: usize, end: usize) Error!void {
        if (!t.streams[s].open) return error.NotOpen;
        if (l >= t.streams[s].layers) return error.BadShape;
        if (end > t.o.capacity) return error.PastCapacity;
        try t.flushReads();
        t.gen +%= 1;
        var need: std.ArrayList(u32) = .empty;
        defer need.deinit(t.gpa);
        errdefer for (need.items) |k| if (t.map[k] == -2) {
            t.map[k] = -1;
        };
        for (0..(end + pool_rows - 1) / pool_rows) |p| {
            const k = t.keyOf(s, l, p);
            const slot = t.map[k];
            if (slot >= 0) {
                t.touch(@intCast(slot));
            } else if (t.persisted.isSet(k)) {
                t.map[k] = -2;
                try need.append(t.gpa, k);
            }
        }
        try t.readIn(need.items);
    }

    pub fn ensureRange(t: *Tier, s: usize, l: usize, end: usize) Error!void {
        try t.prefetchRange(s, l, end);
        try t.flushReads();
    }

    /// Selected positions to row indices at the slab's base (the device's remap, on the host): fetch ran.
    pub fn remap(t: *const Tier, s: usize, l: usize, tokens: []i32, counts: []const i32, W: usize) void {
        for (counts, 0..) |c, r| for (tokens[r * W ..][0..@intCast(c)]) |*tok| {
            const p: usize = @intCast(tok.*);
            tok.* = t.map[t.keyOf(s, l, p / pool_rows)] * pool_rows + @as(i32, @intCast(p % pool_rows));
        };
    }

    /// Positions below `pos` of stream `s` are final: every whole pool below it reaches the file.
    pub fn commit(t: *Tier, s: usize, pos: usize) Error!void {
        const st = &t.streams[s];
        if (!st.open) return error.NotOpen;
        try t.flushReads();
        const writes = t.commitWrites(s, pos);
        st.committed = @max(st.committed, pos);
        if (writes) try t.writeOut(s);
        try t.flushReads();
    }

    /// Whether commit(s, pos) would write pools out now (cuda_tier then waits for the stream first: the pools'
    /// latent writes must have landed).
    pub fn commitWrites(t: *const Tier, s: usize, pos: usize) bool {
        const st = &t.streams[s];
        const hi = @max(st.committed, pos) / pool_rows;
        return hi > st.dirty_lo and (hi - st.dirty_lo) * st.layers >= t.o.commit_batch;
    }

    /// Every whole committed pool of stream `s` not on the file yet, written: gathered from the slab a stage at a
    /// time, contiguous file blocks one write each, one wait for the drive a stage.
    fn writeOut(t: *Tier, s: usize) Error!void {
        const st = &t.streams[s];
        const hi = st.committed / pool_rows;
        if (hi <= st.dirty_lo) return;
        var ids: [stage_blocks]u32 = undefined;
        var keys: [stage_blocks]u32 = undefined;
        var n: usize = 0;
        for (0..st.layers) |l| {
            const k0 = t.mapBase(s, l);
            var p = st.dirty_lo;
            while (p < hi) : (p += 1) {
                if (st.parent != null and p >= home_pools and p < st.fork_pool) {
                    p = st.fork_pool - 1; // the parent's pools
                    continue;
                }
                const k: u32 = @intCast(k0 + p);
                if (t.persisted.isSet(k)) continue;
                const slot = t.map[k];
                if (slot < 0) return error.NotPersisted; // a pool never written: nothing to commit
                ids[n] = @intCast(slot);
                keys[n] = k;
                n += 1;
                if (n == stage_blocks) {
                    try t.writeStage(ids[0..n], keys[0..n]);
                    n = 0;
                }
            }
        }
        if (n > 0) try t.writeStage(ids[0..n], keys[0..n]);
        st.dirty_lo = hi;
    }

    /// Slots `ids` (pools `keys`) to the file: one gather into the write stage, a write a run of contiguous file
    /// blocks, one wait; then the pools count as written (evictable).
    fn writeStage(t: *Tier, ids: []const u32, keys: []const u32) Error!void {
        try t.io.finish(); // the stage's previous writes left
        t.slab.get(t.slab.ctx, ids, t.wr_stage[0 .. ids.len * block]) catch return error.SlabFailed;
        var i: usize = 0;
        while (i < ids.len) {
            var m: usize = 1;
            while (i + m < ids.len and keys[i + m] == keys[i] + m) m += 1;
            try t.io.add(.write, t.wr_stage[i * block ..][0 .. m * block], fileOffset(keys[i]));
            i += m;
        }
        try t.io.finish();
        for (ids, keys) |slot, k| {
            t.note(slot, .file_write, k);
            t.persisted.set(k);
        }
        t.writes += ids.len;
        t.write_calls += 1;
    }

    /// Every committed pool written to the file now (batched commits' waiting ones included).
    pub fn flush(t: *Tier) Error!void {
        try t.flushReads();
        for (t.streams, 0..) |st, s| if (st.open) try t.writeOut(s);
        try t.flushReads();
    }

    /// Stream `s` (just opened, nothing written) starts holding `rows` rows of layer `l` given whole (`bytes`, row
    /// p at p * row_bytes; a fixture's drawn cache, a synthetic context): the home block's and the last partial
    /// pool's rows resident, every other full pool written to the file and left there (fetched when selected).
    /// Load every layer before the stream's first window.
    pub fn load(t: *Tier, s: usize, l: usize, bytes: []const u8, rows: usize) Error!void {
        const st = &t.streams[s];
        if (!st.open or l >= st.layers or st.parent != null) return error.BadShape;
        if (rows == 0 or rows > t.o.capacity or bytes.len < rows * t.o.row_bytes) return error.PastCapacity;
        try t.flushReads();
        const full = rows / pool_rows;
        const pb = pool_rows * t.o.row_bytes;
        var one: [block]u8 align(block) = undefined;
        for (0..@min((rows + pool_rows - 1) / pool_rows, home_pools)) |p| {
            const slot: u32 = @intCast(t.map[t.key(s, l, p)]);
            const n = @min(pb, rows * t.o.row_bytes - p * pb);
            @memset(&one, 0);
            @memcpy(one[0..n], bytes[p * pb ..][0..n]);
            t.slab.put(t.slab.ctx, &.{slot}, &one) catch return error.SlabFailed;
        }
        if (full > home_pools) {
            // through an aligned bounce, a run's pools a write
            const bounce = try t.gpa.alignedAlloc(u8, .fromByteUnits(block), run_pools * block);
            defer t.gpa.free(bounce);
            var p: usize = home_pools;
            while (p < full) {
                const n: usize = @min(@as(usize, run_pools), full - p);
                @memcpy(bounce[0 .. n * block], bytes[p * pb ..][0 .. n * block]);
                try t.io.add(.write, bounce[0 .. n * block], fileOffset(t.key(s, l, p)));
                try t.flushReads();
                for (p..p + n) |q| {
                    const k = t.key(s, l, q);
                    if (!t.persisted.isSet(k) and t.map[k] < 0) t.away[k / t.pools] += 1;
                    t.persisted.set(k);
                }
                t.writes += n;
                p += n;
            }
        }
        if (rows % pool_rows != 0 and full >= home_pools) {
            const g = full / run_pools;
            if (t.seg_run[(s * t.o.layers + l) * t.segs + g] == none) try t.placeLinear(s, l, g, try t.takeRuns(1));
            const slot: u32 = @intCast(t.map[t.key(s, l, full)]);
            try t.getSlot(slot, &one);
            @memcpy(one[0 .. (rows % pool_rows) * t.o.row_bytes], bytes[full * pb ..][0 .. (rows % pool_rows) * t.o.row_bytes]);
            t.slab.put(t.slab.ctx, &.{slot}, &one) catch return error.SlabFailed;
        }
        try t.flushReads();
        st.hi = @max(st.hi, (rows - 1) / seg_positions);
        st.committed = @max(st.committed, rows);
        st.end = @max(st.end, rows);
        st.dirty_lo = @max(st.dirty_lo, full);
    }

    /// Streams an alias of `s` would read through (itself and `s`'s chain): alias refuses more than max_chain, so a
    /// caller checks before it commits to continuing a kept prompt and prefills instead.
    pub fn aliasDepth(t: *const Tier, s: usize) usize {
        var depth: usize = 1;
        var h: ?u32 = @intCast(s);
        while (h) |x| : (h = t.streams[x].parent) depth += 1;
        return depth;
    }

    /// Stream `dst` (just opened, nothing written, `src`'s layers) starts as `src`'s first `rows` positions (a kept
    /// prompt continued): its home block and the pool at `rows` (when partial) are copied, the pools between are
    /// read through `src`, which stays (pending once closed) while `dst` lives. `dst` writes from `rows` on. A chain
    /// longer than max_chain is refused (error.ChainTooDeep): the caller prefills instead.
    pub fn alias(t: *Tier, src: usize, dst: usize, rows: usize) Error!void {
        const ss = &t.streams[src];
        const ds = &t.streams[dst];
        if (!ss.open or ss.pending or !ds.open or ds.hi != 0 or ds.committed != 0 or ds.layers != ss.layers) return error.BadShape;
        if (rows == 0 or rows > t.o.capacity - 1) return error.PastCapacity;
        try t.flushReads();
        const full = rows / pool_rows;
        if (full > home_pools) {
            if (t.aliasDepth(src) > max_chain) return error.ChainTooDeep;
            ds.parent = @intCast(src);
            ds.fork_pool = full;
            ss.refs += 1;
        }
        // the pools dst holds itself below `rows`: the home block's, and the partial pool at `rows`
        const own_hi = @min((rows + pool_rows - 1) / pool_rows, home_pools);
        for (0..ds.layers) |l| {
            for (0..own_hi) |p| try t.copyPool(src, dst, l, p);
            if (rows % pool_rows != 0 and full >= home_pools) {
                const g = full / run_pools;
                if (t.seg_run[(dst * t.o.layers + l) * t.segs + g] == none) try t.placeLinear(dst, l, g, try t.takeRuns(1));
                try t.copyPool(src, dst, l, full);
            }
        }
        try t.flushReads();
        ds.hi = (rows - 1) / seg_positions;
        ds.committed = rows;
        ds.end = rows;
        ds.dirty_lo = 0;
    }

    /// Pool `p` of `src`'s layer `l` (through its chain) into `dst`'s resident slot for it (a linear run's).
    fn copyPool(t: *Tier, src: usize, dst: usize, l: usize, p: usize) Error!void {
        const kd = t.key(dst, l, p);
        const sd = t.map[kd];
        if (sd < 0) return error.BadShape;
        const ks = t.keyOf(src, l, p);
        const ss = t.map[ks];
        t.note(@intCast(sd), .alias_copy, kd);
        if (ss >= 0) {
            try t.copySlot(@intCast(sd), @intCast(ss));
        } else if (t.persisted.isSet(ks)) {
            try t.readSlot(@intCast(sd), fileOffset(ks));
            t.reads += 1;
        } else return error.NotPersisted;
    }

    /// Positions from `pos` on are dropped (rejected drafts, a shortened prompt): the next window rewrites them.
    pub fn rewind(t: *Tier, s: usize, pos: usize) void {
        const st = &t.streams[s];
        st.committed = @min(st.committed, pos);
        if (pos < st.end) t.dropPast(s, pos);
    }

    /// Rows from `pos` on are stale: the pools wholly past it that sit in loose slots or only on the file are
    /// forgotten (a linear run's stay for the window that rewrites them), so stale pools never pin slab slots.
    fn dropPast(t: *Tier, s: usize, pos: usize) void {
        const st = &t.streams[s];
        const lo = (pos + pool_rows - 1) / pool_rows;
        const hi = (st.end + pool_rows - 1) / pool_rows;
        for (0..st.layers) |l| {
            const k0 = t.mapBase(s, l);
            for (lo..hi) |p| {
                if (st.parent != null and p >= home_pools and p < st.fork_pool) continue;
                const slot = t.map[k0 + p];
                if (slot >= 0 and t.runs[@as(usize, @intCast(slot)) / run_pools].kind != .loose) continue;
                t.drop(k0 + p);
            }
            // segments wholly past `pos` (beyond the home block) let their linear runs go: a stream sent far back
            // would otherwise keep every segment it ever wrote linear
            const sr = t.seg_run[(s * t.o.layers + l) * t.segs ..][0..t.segs];
            for (@max(home_runs, (pos + seg_positions - 1) / seg_positions)..t.segs) |g| if (sr[g] != none) {
                const r = sr[g];
                for (r * run_pools..(r + 1) * run_pools) |slot| if (t.owner[slot] != none) t.drop(t.owner[slot]);
                t.runs[r] = .{};
                sr[g] = none;
            };
        }
        st.end = pos;
        st.hi = @min(st.hi, if (pos == 0) 0 else (pos - 1) / seg_positions);
    }

    // -- slots and runs --------------------------------------------------------------------------------------------

    fn note(t: *Tier, slot: usize, kind: EvKind, k: u32) void {
        if (t.trace_ev.len == 0) return;
        const ring = t.trace_ev[slot * trace_depth ..][0..trace_depth];
        std.mem.copyBackwards(Ev, ring[1..], ring[0 .. trace_depth - 1]);
        ring[0] = .{ .kind = kind, .key = k, .window = t.window, .gen = t.gen };
    }

    /// A slot's last events, newest first (Options.trace).
    pub fn history(t: *const Tier, slot: usize) []const Ev {
        if (t.trace_ev.len == 0) return &.{};
        return t.trace_ev[slot * trace_depth ..][0..trace_depth];
    }

    /// The slot holding stream `s`'s layer `l`'s pool `pool` (through an alias's chain), or null.
    pub fn slotOf(t: *const Tier, s: usize, l: usize, pool: usize) ?usize {
        const m = t.map[t.keyOf(s, l, pool)];
        return if (m >= 0) @intCast(m) else null;
    }

    /// Records the device's latent write of rows pos .. pos + R - 1 into their slots (trace).
    pub fn noteWindow(t: *Tier, s: usize, l: usize, pos: usize, R: usize) void {
        if (t.trace_ev.len == 0) return;
        for (pos / pool_rows..(pos + R - 1) / pool_rows + 1) |p| {
            const k = t.key(s, l, p);
            if (t.map[k] >= 0) t.note(@intCast(t.map[k]), .gpu_write, k);
        }
    }

    fn own(t: *Tier, slot: usize, k: u32) void {
        if (t.owner[slot] == none) t.runs[slot / run_pools].held += 1;
        t.owner[slot] = k;
    }

    fn disown(t: *Tier, slot: usize) void {
        if (t.owner[slot] != none) t.runs[slot / run_pools].held -= 1;
        t.owner[slot] = none;
    }

    fn touch(t: *Tier, slot: u32) void {
        t.pin[slot] = t.gen;
        t.used[slot] = t.window;
        t.ref.set(slot);
    }

    /// Pools `keys` (map -2, persisted) into loose slots, their reads queued.
    fn readIn(t: *Tier, keys: []const u32) Error!void {
        for (keys) |k| {
            const slot = try t.allocLoose();
            t.away[k / t.pools] -= 1;
            t.own(slot, k);
            t.map[k] = @intCast(slot);
            t.touch(slot);
            t.note(slot, .file_read, k);
            try t.readSlot(slot, fileOffset(k));
        }
        t.reads += keys.len;
    }

    /// Segment `g` of stream `s`'s layer `l` into run `r`: pools resident elsewhere are copied over, persisted ones
    /// read (queued), unwritten ones left as they are.
    fn placeLinear(t: *Tier, s: usize, l: usize, g: usize, r: u32) Error!void {
        t.runs[r] = .{ .kind = .linear, .key0 = t.key(s, l, g * run_pools) };
        t.seg_run[(s * t.o.layers + l) * t.segs + g] = r;
        for (0..run_pools) |i| {
            const k = t.key(s, l, g * run_pools + i);
            const slot: u32 = r * run_pools + @as(u32, @intCast(i));
            const old = t.map[k];
            if (old >= 0) {
                const u: u32 = @intCast(old);
                try t.copySlot(slot, u);
                t.note(slot, .copy_in, k);
                t.note(u, .move_out, k);
                t.disown(u);
                try t.free_slots.append(t.gpa, u);
            } else if (t.persisted.isSet(k)) {
                t.note(slot, .file_read, k);
                try t.readSlot(slot, fileOffset(k));
                t.away[k / t.pools] -= 1;
                t.reads += 1;
            }
            t.own(slot, k);
            t.map[k] = @intCast(slot);
            t.used[slot] = t.window;
        }
    }

    fn demote(t: *Tier, s: usize, sr: []u32, g: usize) void {
        const r = sr[g];
        sr[g] = none;
        t.runs[r].kind = .loose;
        for (r * run_pools..(r + 1) * run_pools) |slot| {
            t.ref.unset(slot);
            // a pool wholly past the stream's rows is stale: it would pin the slot (never written out) for nothing
            const k = t.owner[slot];
            if (k != none and (k % t.pools) * pool_rows >= t.streams[s].end) t.drop(k);
            if (t.owner[slot] == none) t.free_slots.append(t.gpa, @intCast(slot)) catch {};
        }
    }

    /// Whether a slot's pool may leave: written and named by no current fetch; in the first pass (`hot`) also
    /// neither named in the last hot_windows windows nor a favored stream's.
    fn evictable(t: *const Tier, slot: usize, hot: bool) bool {
        const k = t.owner[slot];
        if (k == none or !t.persisted.isSet(k) or t.pin[slot] == t.gen) return false;
        if (!hot) return true;
        return t.used[slot] +% t.o.hot_windows <= t.window and !t.streams[t.block_user[k / t.pools] / t.o.layers].favored;
    }

    /// Stream `s`'s pools stay ahead of every other stream's (a prompt being filled keeps its whole context, so its
    /// chunks' selections read nothing), or no longer.
    pub fn favor(t: *Tier, s: usize, on: bool) void {
        t.streams[s].favored = on;
    }

    fn evict(t: *Tier, slot: usize) void {
        t.note(slot, .evict, t.owner[slot]);
        t.away[t.owner[slot] / t.pools] += 1;
        t.map[t.owner[slot]] = -1;
        t.disown(slot);
        t.ref.unset(slot);
    }

    /// A loose slot holding nothing: a free one (a free run turned loose if need be), else the clock's choice among written pools no current call names
    /// (first sparing the pools of the last hot_windows windows).
    fn allocLoose(t: *Tier) Error!u32 {
        while (t.free_slots.pop()) |slot| {
            if (t.runs[slot / run_pools].kind == .loose and t.owner[slot] == none) return slot;
        }
        // a free run turns loose before anything is evicted
        for (t.runs, 0..) |*r, i| if (r.kind == .free) {
            r.kind = .loose;
            var k: usize = run_pools;
            while (k > 1) {
                k -= 1;
                try t.free_slots.append(t.gpa, @intCast(i * run_pools + k));
            }
            return @intCast(i * run_pools);
        };
        const n = t.owner.len;
        for ([_]bool{ true, false }) |hot| {
            var steps: usize = 0;
            while (steps < 2 * n) : (steps += 1) {
                const slot = t.hand;
                t.hand = (t.hand + 1) % n;
                if (t.runs[slot / run_pools].kind != .loose) continue;
                if (t.owner[slot] == none) return @intCast(slot);
                if (!t.evictable(slot, hot)) continue;
                if (t.ref.isSet(slot)) {
                    t.ref.unset(slot);
                    continue;
                }
                t.evict(slot);
                return @intCast(slot);
            }
        }
        // pools waiting for a batched commit are not evictable: write every stream's out and sweep once more
        if (!t.flushing) {
            var waiting = false;
            for (t.streams) |st| waiting = waiting or (st.open and st.committed / pool_rows > st.dirty_lo);
            if (waiting) {
                t.flushing = true;
                defer t.flushing = false;
                for (t.streams, 0..) |st, x| if (st.open) try t.writeOut(x);
                return t.allocLoose();
            }
        }
        return error.TierFull;
    }

    /// `n` adjacent runs holding nothing: free ones, else loose ones drained (written pools evicted, the others moved
    /// to loose slots outside them).
    fn takeRuns(t: *Tier, n: usize) Error!u32 {
        const nr = t.runs.len;
        if (n > nr) return error.TierFull;
        var run_len: usize = 0;
        for (t.runs, 0..) |r, i| {
            run_len = if (r.kind == .free) run_len + 1 else 0;
            if (run_len == n) {
                const b = i + 1 - n;
                for (b * run_pools..(b + n) * run_pools) |slot| if (t.owner[slot] != none) return error.TierFull; // a free run holds nothing
                for (b..b + n) |x| t.runs[x].kind = .taken;
                return @intCast(b);
            }
        }
        // the block of loose runs holding the fewest pools (each is evicted or moved)
        var best: ?usize = null;
        var best_cost: usize = std.math.maxInt(usize);
        var a: usize = 0;
        while (a + n <= nr) : (a += 1) {
            var cost: usize = 0;
            var ok = true;
            for (a..a + n) |r| switch (t.runs[r].kind) {
                .free => {},
                .linear, .taken => {
                    ok = false;
                    break;
                },
                .loose => cost += t.runs[r].held,
            };
            if (ok and cost < best_cost) {
                best = a;
                best_cost = cost;
                if (cost == 0) break;
            }
        }
        const b = best orelse return error.TierFull;
        // out of the loose set first, so no move lands inside
        for (b..b + n) |r| t.runs[r].kind = .taken;
        for (b * run_pools..(b + n) * run_pools) |slot| {
            const k = t.owner[slot];
            if (k == none) continue;
            if (t.evictable(slot, false)) {
                t.evict(slot);
                continue;
            }
            const dst = t.allocLoose() catch |e| {
                for (b..b + n) |r| t.runs[r].kind = .loose;
                return e;
            };
            t.copySlot(dst, @intCast(slot)) catch |e| {
                for (b..b + n) |r| t.runs[r].kind = .loose;
                return e;
            };
            t.note(dst, .move_in, k);
            t.note(slot, .move_out, k);
            t.own(dst, k);
            t.map[k] = @intCast(dst);
            t.pin[dst] = t.pin[slot];
            t.used[dst] = t.used[slot];
            t.disown(slot);
        }
        for (b * run_pools..(b + n) * run_pools) |slot| t.ref.unset(slot);
        return @intCast(b);
    }

    /// Where a verify found a row that differs from the reference.
    pub const Mismatch = struct { layer: usize, pool: usize, slot: i32, in_file: bool, run_kind: Kind };

    /// Every pool of stream `s` below `upto` positions (whole pools) that the tier holds, checked against `ref`
    /// (layer l's rows at ref[l], row p at p * row_bytes): a resident pool's slab bytes and a written pool's file
    /// block (an alias's pools below its fork are its parent's: checked there). Returns the first that differs.
    pub fn verify(t: *Tier, s: usize, ref: []const []const u8, upto: usize) Error!?Mismatch {
        try t.flushReads();
        const st = t.streams[s];
        const pb = pool_rows * t.o.row_bytes;
        // the file read in spans of `span` blocks (a layer's keys are contiguous on the file)
        const span: usize = 256;
        const buf = try t.gpa.alignedAlloc(u8, .fromByteUnits(block), span * block);
        defer t.gpa.free(buf);
        for (0..st.layers) |l| {
            const np = upto / pool_rows;
            var p0: usize = 0;
            while (p0 < np) : (p0 += span) {
                const n: usize = @min(span, np - p0);
                const got = std.c.pread(t.io.fd, buf.ptr, n * block, @intCast(fileOffset(t.key(s, l, p0))));
                for (p0..p0 + n) |p| {
                    if (st.parent != null and p >= home_pools and p < st.fork_pool) continue;
                    const k = t.key(s, l, p);
                    const want = ref[l][p * pb ..][0..pb];
                    const slot = t.map[k];
                    const kind: Kind = if (slot >= 0) t.runs[@as(usize, @intCast(slot)) / run_pools].kind else .free;
                    if (slot >= 0) {
                        var in_slab: [block]u8 align(block) = undefined;
                        t.slab.get(t.slab.ctx, &.{@intCast(slot)}, &in_slab) catch return error.SlabFailed;
                        if (!std.mem.eql(u8, want, &in_slab)) return .{ .layer = l, .pool = p, .slot = slot, .in_file = false, .run_kind = kind };
                    }
                    if (t.persisted.isSet(k)) {
                        const in = (p - p0) * block;
                        if (got < @as(isize, @intCast(in + block)) or !std.mem.eql(u8, want, buf[in..][0..block]))
                            return .{ .layer = l, .pool = p, .slot = slot, .in_file = true, .run_kind = kind };
                    }
                }
            }
        }
        return null;
    }

    /// Checks that the map and the slots' owners name each other (tests).
    pub fn check(t: *const Tier) !void {
        for (t.map, 0..) |slot, k| if (slot >= 0) {
            if (t.owner[@intCast(slot)] != k) return error.Inconsistent;
        } else if (slot != -1) return error.Inconsistent;
        for (t.owner, 0..) |k, slot| if (k != none) {
            if (t.map[k] != @as(i32, @intCast(slot))) return error.Inconsistent;
            if (t.runs[slot / run_pools].kind == .free) return error.Inconsistent;
        };
        for (t.away, 0..) |a, sl| {
            var n: u32 = 0;
            for (t.map[sl * t.pools ..][0..t.pools], sl * t.pools..) |slot, k| n += @intFromBool(slot == -1 and t.persisted.isSet(k));
            if (n != a) return error.Inconsistent;
        }
        for (t.runs, 0..) |r, i| {
            var held: u32 = 0;
            for (t.owner[i * run_pools ..][0..run_pools]) |k| held += @intFromBool(k != none);
            if (held != r.held) return error.Inconsistent;
        }
    }
};

// -- I/O -----------------------------------------------------------------------------------------------------------

/// Block reads and writes on the backing file: Linux AIO on O_DIRECT where the kernel and file system allow it (the
/// engine runs in Docker, whose default seccomp profile refuses io_uring but not AIO), else positional calls.
/// `add` queues (submitting when the queue is full), `finish` returns once every queued transfer landed.
///
/// Ordering the tier relies on:
/// - The drive's DMA reads and writes only the tier's host stages (rd_stage, wr_stage), never the slab: the slab
///   is device memory reached through Slab.put / get / copy (cuda_tier: cuMemcpy and its kernels).
/// - A latent write lands before the get that writes its pool out: cuda_tier.commit synchronizes the stream
///   that queued the latent writes before kv_tier.commit runs (engine: tierSettle; tests: the same call).
/// - A slot is reused (evicted, moved, read into) only by a host call that runs after every device call reading
///   it finished: resolve synchronizes its stream before fetch, and beginWindow is called after the stream's
///   earlier calls finished (the engine synchronizes in tierBegin). Every public call ends with its reads put in
///   the slab (flushReads), and each Slab call returns once its bytes landed.
extern "c" fn flock(fd: c_int, op: c_int) c_int;
const lock_ex = 2;
const lock_nb = 4;

pub const Io = struct {
    gpa: std.mem.Allocator,
    fd: c_int,
    aio: bool = false,
    direct: bool = false,
    depth: usize,
    queue: std.ArrayList(Req) = .empty,
    lanes: []Lane = &.{},

    const Op = enum { read, write };
    const Req = struct { op: Op, buf: []u8, off: u64 };

    /// Transfers a lane takes before a second lane (thread) starts: below it a thread costs more than it saves.
    const lane_min = 128;

    const Iocb = extern struct {
        data: u64 = 0,
        key: u32 = 0,
        rw_flags: u32 = 0,
        lio_opcode: u16 = 0,
        reqprio: i16 = 0,
        fildes: u32 = 0,
        buf: u64 = 0,
        nbytes: u64 = 0,
        offset: i64 = 0,
        reserved2: u64 = 0,
        flags: u32 = 0,
        resfd: u32 = 0,
    };
    const IoEvent = extern struct { data: u64, obj: u64, res: i64, res2: i64 };

    /// One AIO context and its control blocks: a thread's share of a finish.
    const Lane = struct {
        ctx: usize = 0,
        cbs: []Iocb,
        ptrs: []*Iocb,
        free: []u32,
        bad: bool = false,
    };


    /// `lanes` AIO contexts of `depth` transfers each (the measured NVMe: QD 64 on one thread reads 1.0M 4 KiB
    /// blocks a second, four threads 2.2M).
    pub fn open(gpa: std.mem.Allocator, path: [:0]const u8, depth: usize, lanes: usize) Error!Io {
        var io: Io = .{ .gpa = gpa, .fd = -1, .depth = @max(depth, 1) };
        if (builtin.os.tag == .linux) {
            io.fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true, .DIRECT = true }, @as(c_uint, 0o600));
            io.direct = io.fd >= 0;
        }
        if (io.fd < 0) io.fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true }, @as(c_uint, 0o600));
        if (io.fd < 0) return error.OpenFailed;
        errdefer _ = std.c.close(io.fd);
        // the file is this tier's alone and starts empty: another process on the same path (a second server, a
        // test left running) is refused, and nothing a previous run left is ever there to be read
        if (flock(io.fd, lock_ex | lock_nb) != 0) return error.FileBusy;
        if (std.c.ftruncate(io.fd, 0) != 0) return error.OpenFailed;
        if (builtin.os.tag == .linux) {
            io.lanes = try gpa.alloc(Lane, @max(lanes, 1));
            var made: usize = 0;
            errdefer io.freeLanes(made);
            for (io.lanes) |*l| {
                l.* = .{ .cbs = try gpa.alloc(Iocb, io.depth), .ptrs = undefined, .free = undefined };
                l.ptrs = gpa.alloc(*Iocb, io.depth) catch |e| {
                    gpa.free(l.cbs);
                    return e;
                };
                l.free = gpa.alloc(u32, io.depth) catch |e| {
                    gpa.free(l.cbs);
                    gpa.free(l.ptrs);
                    return e;
                };
                made += 1;
                if (linux.errno(linux.syscall2(.io_setup, io.depth, @intFromPtr(&l.ctx))) != .SUCCESS) l.ctx = 0;
            }
            io.aio = true;
            for (io.lanes) |l| io.aio = io.aio and l.ctx != 0;
        }
        return io;
    }

    fn freeLanes(io: *Io, n: usize) void {
        for (io.lanes[0..n]) |l| {
            if (l.ctx != 0) _ = linux.syscall1(.io_destroy, l.ctx);
            io.gpa.free(l.cbs);
            io.gpa.free(l.ptrs);
            io.gpa.free(l.free);
        }
        io.gpa.free(io.lanes);
    }

    pub fn close(io: *Io) void {
        io.finish() catch {};
        if (builtin.os.tag == .linux) io.freeLanes(io.lanes.len);
        io.queue.deinit(io.gpa);
        _ = std.c.close(io.fd);
    }

    pub fn add(io: *Io, op: Op, buf: []u8, off: u64) Error!void {
        try io.queue.append(io.gpa, .{ .op = op, .buf = buf, .off = off });
        if (io.queue.items.len >= io.depth * 64) try io.finish();
    }

    pub fn finish(io: *Io) Error!void {
        defer io.queue.clearRetainingCapacity();
        const items = io.queue.items;
        if (items.len == 0) return;
        if (builtin.os.tag == .linux and io.aio) {
            // contiguous shares, one a lane; the caller runs the first
            const n = std.math.clamp(items.len / lane_min, 1, io.lanes.len);
            var threads: [16]?std.Thread = @splat(null);
            for (1..@min(n, threads.len)) |i| {
                threads[i] = std.Thread.spawn(.{}, runLane, .{ io, &io.lanes[i], items[i * items.len / n .. (i + 1) * items.len / n] }) catch null;
                if (threads[i] == null) runLane(io, &io.lanes[i], items[i * items.len / n .. (i + 1) * items.len / n]);
            }
            runLane(io, &io.lanes[0], items[0 .. items.len / n]);
            var bad = false;
            for (0..n) |i| {
                if (threads[i]) |t| t.join();
                bad = bad or io.lanes[i].bad;
            }
            if (bad) return error.ReadFailed;
            return;
        }
        for (items) |q| {
            var done: usize = 0;
            while (done < q.buf.len) {
                const n = switch (q.op) {
                    .read => std.c.pread(io.fd, q.buf.ptr + done, q.buf.len - done, @intCast(q.off + done)),
                    .write => std.c.pwrite(io.fd, q.buf.ptr + done, q.buf.len - done, @intCast(q.off + done)),
                };
                if (n <= 0) return if (q.op == .read) error.ReadFailed else error.WriteFailed;
                done += @intCast(n);
            }
        }
    }

    /// `items` through one lane's AIO context, at most `depth` in flight, to completion; a short or failed
    /// transfer marks the lane bad.
    fn runLane(io: *const Io, l: *Lane, items: []const Req) void {
        l.bad = false;
        var nfree: usize = io.depth;
        for (l.free, 0..) |*f, i| f.* = @intCast(i);
        var next: usize = 0;
        var inflight: usize = 0;
        while (next < items.len or inflight > 0) {
            var n: usize = 0;
            while (next < items.len and nfree > 0) : (next += 1) {
                const q = items[next];
                nfree -= 1;
                const c = l.free[nfree];
                l.cbs[c] = .{
                    .data = (@as(u64, c) << 32) | q.buf.len,
                    .lio_opcode = if (q.op == .read) 0 else 1,
                    .fildes = @intCast(io.fd),
                    .buf = @intFromPtr(q.buf.ptr),
                    .nbytes = q.buf.len,
                    .offset = @intCast(q.off),
                };
                l.ptrs[n] = &l.cbs[c];
                n += 1;
            }
            var sent: usize = 0;
            while (sent < n) {
                const r: isize = @bitCast(linux.syscall3(.io_submit, l.ctx, n - sent, @intFromPtr(l.ptrs[sent..].ptr)));
                if (r == -@as(isize, @intFromEnum(linux.E.AGAIN)) or r == -@as(isize, @intFromEnum(linux.E.INTR))) continue;
                if (r <= 0) {
                    // never submitted: count them done, the lane failed
                    l.bad = true;
                    for (l.ptrs[sent..n]) |cb| {
                        l.free[nfree] = @intCast(cb.data >> 32);
                        nfree += 1;
                    }
                    n = sent;
                    next = items.len;
                    break;
                }
                sent += @intCast(r);
            }
            inflight += n;
            if (inflight == 0) break;
            var ev: [256]IoEvent = undefined;
            const got: isize = @bitCast(linux.syscall5(.io_getevents, l.ctx, 1, ev.len, @intFromPtr(&ev), 0));
            if (got < 0) {
                if (got == -@as(isize, @intFromEnum(linux.E.INTR))) continue;
                l.bad = true;
                return;
            }
            for (ev[0..@intCast(got)]) |e| {
                if (e.res != @as(i64, @intCast(e.data & 0xFFFF_FFFF))) l.bad = true;
                l.free[nfree] = @intCast(e.data >> 32);
                nfree += 1;
                inflight -= 1;
            }
        }
    }
};

// -- tests ---------------------------------------------------------------------------------------------------------

const testing = std.testing;

const Fixture = struct {
    host: HostSlab,
    map: []i32,
    path_buf: [256]u8,
    path: [:0]u8,
    tier: Tier,
    ref: [][]u8, // [streams * layers][capacity * row_bytes]: the whole cache the tier stands for
    end: []usize, // a stream's positions written

    fn init(o: Options, runs: usize, seed: u64) !*Fixture {
        return initBlocks(o, runs, seed, o.mapBlocks());
    }

    fn initBlocks(o: Options, runs: usize, seed: u64, blocks: usize) !*Fixture {
        const gpa = testing.allocator;
        const f = try gpa.create(Fixture);
        errdefer gpa.destroy(f);
        f.host = .{ .bytes = try gpa.alignedAlloc(u8, .fromByteUnits(block), runs * run_pools * block) };
        @memset(f.host.bytes, 0xAA);
        f.map = try gpa.alloc(i32, blocks * o.capacity / pool_rows);
        // TF_TIER_TEST_DIR: a directory on the NVMe's file system (O_DIRECT and AIO; /tmp may be tmpfs)
        const dir = if (std.c.getenv("TF_TIER_TEST_DIR")) |e| std.mem.span(e) else "/tmp";
        const name = try std.fmt.bufPrint(&f.path_buf, "{s}/kv_tier_test_{d}_{x}.bin\x00", .{ dir, std.c.getpid(), seed });
        f.path = name[0 .. name.len - 1 :0];
        f.tier = try Tier.init(gpa, f.host.slab(), f.map, f.path, o);
        f.ref = try gpa.alloc([]u8, o.streams * o.layers);
        for (f.ref) |*r| {
            r.* = try gpa.alloc(u8, o.capacity * o.row_bytes);
            @memset(r.*, 0);
        }
        f.end = try gpa.alloc(usize, o.streams);
        @memset(f.end, 0);
        return f;
    }

    fn deinit(f: *Fixture) void {
        const gpa = testing.allocator;
        f.tier.deinit();
        _ = std.c.unlink(f.path.ptr);
        for (f.ref) |r| gpa.free(r);
        gpa.free(f.ref);
        gpa.free(f.end);
        gpa.free(f.map);
        gpa.free(f.host.bytes);
        gpa.destroy(f);
    }

    /// A window's rows written on every layer through writeRuns, as the latent write does.
    fn write(f: *Fixture, rnd: std.Random, s: usize, pos: usize, R: usize) !void {
        const t = &f.tier;
        const rb = t.o.row_bytes;
        try t.beginWindow(s, pos, R);
        for (0..t.streams[s].layers) |l| {
            const pieces = try t.writeRuns(s, l, pos, R);
            var covered: usize = 0;
            for (pieces) |pc| if (pc) |p| {
                for (p.pos..p.pos + p.rows) |q| {
                    const at: usize = @intCast(p.base + @as(i64, @intCast(q * rb)));
                    const dst = f.host.bytes[at..][0..rb];
                    rnd.bytes(dst);
                    @memcpy(f.ref[s * t.o.layers + l][q * rb ..][0..rb], dst);
                }
                covered += p.rows;
            };
            try testing.expectEqual(R, covered);
        }
        f.end[s] = pos + R;
    }

    /// A selection of R rows over the stream's visible positions, fetched and remapped, compared byte for byte.
    /// Every committed row of stream `s` checked where the tier keeps it: a resident pool's slab rows, a written
    /// pool's file block (an alias's rows below its fork are its parent's, checked there).
    fn verifyAll(f: *Fixture, s: usize) !void {
        const t = &f.tier;
        var refs: [16][]const u8 = undefined;
        for (0..t.streams[s].layers) |l| refs[l] = f.ref[s * t.o.layers + l];
        if (try t.verify(s, refs[0..t.streams[s].layers], t.streams[s].committed)) |m| {
            std.debug.print("verify: stream {d} {any}\n", .{ s, m });
            return error.TierDiffers;
        }
    }

    fn select(f: *Fixture, rnd: std.Random, s: usize, R: usize, W: usize) !void {
        const t = &f.tier;
        const rb = t.o.row_bytes;
        const gpa = testing.allocator;
        const tokens = try gpa.alloc(i32, R * W);
        defer gpa.free(tokens);
        const orig = try gpa.alloc(i32, R * W);
        defer gpa.free(orig);
        const counts = try gpa.alloc(i32, R);
        defer gpa.free(counts);
        const l = rnd.uintLessThan(usize, t.streams[s].layers);
        const vis = f.end[s];
        for (0..R) |r| {
            const c = rnd.uintAtMost(usize, W);
            counts[r] = @intCast(c);
            for (0..c) |i| {
                // recent positions, the head and the rest, as the selection mixes them
                const tok = switch (rnd.uintLessThan(u8, 4)) {
                    0 => vis - 1 - rnd.uintLessThan(usize, @min(vis, 64)),
                    1 => rnd.uintLessThan(usize, @min(vis, 64)),
                    else => rnd.uintLessThan(usize, vis),
                };
                tokens[r * W + i] = @intCast(tok);
            }
        }
        @memcpy(orig, tokens);
        try t.fetch(s, l, tokens, counts, W);
        t.remap(s, l, tokens, counts, W);
        for (0..R) |r| for (0..@intCast(counts[r])) |i| {
            const want = f.ref[s * t.o.layers + l][@as(usize, @intCast(orig[r * W + i])) * rb ..][0..rb];
            const got = f.host.bytes[@as(usize, @intCast(tokens[r * W + i])) * rb ..][0..rb];
            try testing.expectEqualSlices(u8, want, got);
        };
    }
};

test "a selection reads the bytes last written, through windows, drafts, commits and evictions" {
    var prng = std.Random.DefaultPrng.init(0x7131);
    const rnd = prng.random();
    // 2 streams x 2 layers x 32,768 positions = 64 runs of latents in a 26-run slab: the home blocks take 12 runs,
    // the recent windows up to 12, the loose runs hold what else fits and the rest comes back from the file
    const o: Options = .{ .streams = 2, .layers = 2, .capacity = 16 * seg_positions, .recent_runs = 2, .depth = 32 };
    const f = try Fixture.init(o, 26, 1);
    defer f.deinit();
    const t = &f.tier;
    for (0..o.streams) |s| try t.open(s, o.layers);
    var committed: [2]usize = .{ 0, 0 };
    var steps: usize = 0;
    const iters: usize = if (std.c.getenv("TF_TIER_ITERS")) |e| std.fmt.parseInt(usize, std.mem.span(e), 10) catch 400 else 400;
    while (steps < iters) : (steps += 1) {
        const s = rnd.uintLessThan(usize, o.streams);
        const pos = committed[s];
        if (pos + seg_positions > o.capacity) continue;
        const prefill = rnd.uintLessThan(u8, 4) == 0;
        const R = if (prefill) 1 + rnd.uintLessThan(usize, seg_positions) else 1 + rnd.uintLessThan(usize, 16);
        try f.write(rnd, s, pos, R);
        try f.select(rnd, s, @min(R, 4), 512);
        // drafts: a round keeps a prefix of its rows; a prompt chunk all of them
        const keep = if (prefill) R else 1 + rnd.uintLessThan(usize, R);
        committed[s] = pos + keep;
        // now and then a stream goes back (a kept prompt shortened): its next window rewrites an older segment
        if (rnd.uintLessThan(u8, 16) == 0) committed[s] = rnd.uintLessThan(usize, committed[s]);
        t.rewind(s, committed[s]);
        try t.commit(s, committed[s]);
        f.end[s] = committed[s];
        try t.check();
        for (0..o.streams) |x| try f.verifyAll(x);
    }
    try testing.expect(t.reads > 0); // the slab is too small to hold everything: some pools came back from the file
    try testing.expect(t.writes > 0);
    if (builtin.os.tag == .linux and std.c.getenv("TF_TIER_TEST_DIR") != null) {
        // on the NVMe the transfers take the engine's path
        try testing.expect(t.io.aio and t.io.direct);
    }
}

test "a stream sent back again and again keeps its slab use bounded (rejected and shortened rows let go)" {
    var prng = std.Random.DefaultPrng.init(0xbac4);
    const rnd = prng.random();
    const o: Options = .{ .streams = 1, .layers = 2, .capacity = 16 * seg_positions, .recent_runs = 2, .depth = 16 };
    // home 6 runs, the recent window and window 2 x 4: a stream that kept every segment it wrote needs 32
    const f = try Fixture.init(o, 16, 13);
    defer f.deinit();
    const t = &f.tier;
    try t.open(0, 2);
    for (0..12) |_| {
        // forward to near the capacity in prompt chunks and drafted windows, then back to an early position
        var pos = t.streams[0].committed;
        while (pos + seg_positions < o.capacity) {
            const R = if (rnd.boolean()) seg_positions else 1 + rnd.uintLessThan(usize, 16);
            try f.write(rnd, 0, pos, R);
            pos += if (R == seg_positions) R else 1 + rnd.uintLessThan(usize, R);
            t.rewind(0, pos);
            try t.commit(0, pos);
            f.end[0] = pos;
        }
        try f.select(rnd, 0, 4, 512);
        const back = rnd.uintLessThan(usize, 4 * seg_positions);
        t.rewind(0, back);
        f.end[0] = back;
        try t.check();
        try f.verifyAll(0);
    }
}

test "fetchMany serves several streams' selections from one batch of reads" {
    var prng = std.Random.DefaultPrng.init(0x99);
    const rnd = prng.random();
    const o: Options = .{ .streams = 3, .layers = 2, .capacity = 8 * seg_positions, .recent_runs = 1, .depth = 16 };
    const f = try Fixture.init(o, 36, 5);
    defer f.deinit();
    const t = &f.tier;
    for (0..3) |s| try t.open(s, 2);
    for (0..3) |s| {
        var pos: usize = 0;
        while (pos < 7 * seg_positions) : (pos += seg_positions) {
            try f.write(rnd, s, pos, seg_positions);
            try t.commit(s, pos + seg_positions);
        }
    }
    const W = 256;
    var toks: [6][2 * W]i32 = undefined;
    var orig: [6][2 * W]i32 = undefined;
    var cnts: [6][2]i32 = undefined;
    var sels: [6]Tier.Selection = undefined;
    for (0..6) |i| {
        for (0..2) |r| {
            cnts[i][r] = W;
            for (0..W) |j| toks[i][r * W + j] = @intCast(rnd.uintLessThan(usize, f.end[i / 2]));
        }
        orig[i] = toks[i];
        sels[i] = .{ .s = i / 2, .l = i % 2, .tokens = &toks[i], .counts = &cnts[i], .W = W };
    }
    const reads = t.reads;
    try t.fetchMany(&sels);
    try testing.expect(t.reads > reads);
    for (0..6) |i| {
        t.remap(i / 2, i % 2, &toks[i], &cnts[i], W);
        for (0..2 * W) |j| {
            const want = f.ref[(i / 2) * 2 + i % 2][@as(usize, @intCast(orig[i][j])) * o.row_bytes ..][0..o.row_bytes];
            try testing.expectEqualSlices(u8, want, f.host.bytes[@as(usize, @intCast(toks[i][j])) * o.row_bytes ..][0..o.row_bytes]);
        }
    }
    try t.check();
}

test "aliases read their parents' pools: kept prompts continued, parents closed first, chains" {
    var prng = std.Random.DefaultPrng.init(0xa11a5);
    const rnd = prng.random();
    const o: Options = .{ .streams = 4, .layers = 2, .capacity = 16 * seg_positions, .recent_runs = 1, .depth = 16 };
    const f = try Fixture.init(o, 60, 6);
    defer f.deinit();
    const t = &f.tier;
    var trial: usize = 0;
    while (trial < 12) : (trial += 1) {
        // a: a prompt of random length, committed
        const a = try t.acquire(2);
        const la = home_positions / 2 + rnd.uintLessThan(usize, 10 * seg_positions);
        var pos: usize = 0;
        while (pos < la) {
            const R = @min(la - pos, 1 + rnd.uintLessThan(usize, seg_positions));
            try f.write(rnd, a, pos, R);
            pos += R;
            try t.commit(a, pos);
        }
        // b continues a's first `rows`, c continues b's
        const b = try t.acquire(2);
        const rows = 1 + rnd.uintLessThan(usize, la);
        try t.alias(a, b, rows);
        for (0..2) |l| @memcpy(f.ref[b * 2 + l][0 .. rows * o.row_bytes], f.ref[a * 2 + l][0 .. rows * o.row_bytes]);
        f.end[b] = rows;
        if (rnd.boolean()) t.close(a); // the kept prompt let go while b reads through it
        pos = rows;
        for (0..6) |_| {
            const R = 1 + rnd.uintLessThan(usize, 16);
            try f.write(rnd, b, pos, R);
            try f.select(rnd, b, R, 256);
            pos += 1 + rnd.uintLessThan(usize, R);
            t.rewind(b, pos);
            try t.commit(b, pos);
            f.end[b] = pos;
        }
        const c = try t.acquire(2);
        const rows_c = 1 + rnd.uintLessThan(usize, pos);
        try t.alias(b, c, rows_c);
        for (0..2) |l| @memcpy(f.ref[c * 2 + l][0 .. rows_c * o.row_bytes], f.ref[b * 2 + l][0 .. rows_c * o.row_bytes]);
        f.end[c] = rows_c;
        t.close(b);
        try f.write(rnd, c, rows_c, 8);
        for (0..4) |_| try f.select(rnd, c, 4, 512);
        try t.check();
        t.close(c);
        t.close(a); // a no-op when closed before (then freed with its last alias)
        try t.check();
        for (t.streams) |st| try testing.expect(!st.open and !st.pending);
    }
}

test "load: a whole drawn cache, then windows and selections over it" {
    var prng = std.Random.DefaultPrng.init(0x10ad);
    const rnd = prng.random();
    const o: Options = .{ .streams = 1, .layers = 2, .capacity = 32 * seg_positions, .recent_runs = 2, .depth = 16 };
    const f = try Fixture.init(o, 30, 7);
    defer f.deinit();
    const t = &f.tier;
    try t.open(0, 2);
    const rows = 30 * seg_positions + 2;
    for (0..2) |l| {
        rnd.bytes(f.ref[l][0 .. rows * o.row_bytes]);
        try t.load(0, l, f.ref[l][0 .. rows * o.row_bytes], rows);
    }
    f.end[0] = rows;
    try t.check();
    // a window inside the drawn rows (the fixture's run l writes at 40,000 then 600,000), then one past them
    for ([_]usize{ 20_000, 20_003, rows }) |pos| {
        try f.write(rnd, 0, pos, 3);
        f.end[0] = @max(rows, pos + 3);
        for (0..3) |_| try f.select(rnd, 0, 3, 512);
        try t.commit(0, pos + 3);
        try t.check();
    }
}

test "a favored stream's pools stay while another stream's selections fill the slab" {
    var prng = std.Random.DefaultPrng.init(0xfa7);
    const rnd = prng.random();
    const o: Options = .{ .streams = 2, .layers = 1, .capacity = 16 * seg_positions, .recent_runs = 1, .hot_windows = 0, .depth = 16 };
    const f = try Fixture.init(o, 17, 8);
    defer f.deinit();
    const t = &f.tier;
    for (0..2) |s| try t.open(s, 1);
    for (0..2) |s| {
        var pos: usize = 0;
        while (pos < 10 * seg_positions) : (pos += seg_positions) {
            try f.write(rnd, s, pos, seg_positions);
            try t.commit(s, pos + seg_positions);
        }
    }
    // stream 0 favored and wholly resident; stream 1's selections over its whole context churn the rest
    t.favor(0, true);
    try t.ensureRange(0, 0, f.end[0]);
    for (0..200) |_| try f.select(rnd, 1, 1, 256);
    const reads = t.reads;
    try t.ensureRange(0, 0, f.end[0]);
    try testing.expectEqual(reads, t.reads);
    try t.check();
}

test "map blocks: streams beyond the blocks wait, a closed stream's block (and file region) serves the next" {
    var prng = std.Random.DefaultPrng.init(0xb10c);
    const rnd = prng.random();
    // 6 streams of 2 layers share 5 blocks: two backbone-like streams (2 layers) and a head-like one (1 layer)
    const o: Options = .{ .streams = 6, .layers = 2, .capacity = 8 * seg_positions, .recent_runs = 1, .depth = 16, .blocks = 5 };
    const gpa = testing.allocator;
    const f = try Fixture.initBlocks(o, 30, 9, o.mapBlocks());
    defer f.deinit();
    const t = &f.tier;
    var round: usize = 0;
    while (round < 6) : (round += 1) {
        const a = try t.acquire(2);
        const b = try t.acquire(2);
        const m = try t.acquire(1);
        try testing.expectError(error.NoStream, t.acquire(1));
        for ([_]usize{ a, b, m }) |s| {
            var pos: usize = 0;
            const n = 1 + rnd.uintLessThan(usize, 5 * seg_positions);
            while (pos < n) {
                const R = @min(n - pos, 1 + rnd.uintLessThan(usize, seg_positions));
                try f.write(rnd, s, pos, R);
                pos += R;
                try t.commit(s, pos);
            }
            for (0..3) |_| try f.select(rnd, s, 4, 512);
        }
        try t.check();
        for ([_]usize{ a, b, m }) |s| t.close(s);
        // the fixture's whole-cache copy of a stream starts over with it
        for (f.ref) |r| @memset(r, 0);
        _ = gpa;
    }
}

test "a chain of max_chain streams reads through every level; one more is refused" {
    var prng = std.Random.DefaultPrng.init(0xc4a1);
    const rnd = prng.random();
    const o: Options = .{ .streams = max_chain + 2, .layers = 1, .capacity = 8 * seg_positions, .recent_runs = 1, .depth = 16 };
    const f = try Fixture.init(o, 60, 10);
    defer f.deinit();
    const t = &f.tier;
    var prev = try t.acquire(1);
    var pos: usize = 0;
    while (pos < home_positions + 64) : (pos += 64) {
        try f.write(rnd, prev, pos, 64);
        try t.commit(prev, pos + 64);
    }
    // each next stream continues the last at its end and writes 300 more rows
    var level: usize = 1;
    while (true) : (level += 1) {
        const next = try t.acquire(1);
        if (t.aliasDepth(prev) > max_chain) {
            try testing.expectError(error.ChainTooDeep, t.alias(prev, next, pos));
            break;
        }
        try t.alias(prev, next, pos);
        @memcpy(f.ref[next][0 .. pos * o.row_bytes], f.ref[prev][0 .. pos * o.row_bytes]);
        const end = pos + 300;
        while (pos < end) : (pos += 60) {
            try f.write(rnd, next, pos, 60);
            try t.commit(next, pos + 60);
        }
        for (0..4) |_| try f.select(rnd, next, 4, 512);
        try t.check();
        prev = next;
    }
    try testing.expectEqual(@as(usize, max_chain), level);
}

test "the backing file is the tier's alone and starts empty" {
    const o: Options = .{ .streams = 1, .layers = 1, .capacity = 4 * seg_positions, .depth = 8 };
    const f = try Fixture.init(o, 8, 12);
    defer f.deinit();
    // a second tier on the same path (another process, or a test still running) is refused
    const slab = try testing.allocator.alignedAlloc(u8, .fromByteUnits(block), 8 * run_pools * block);
    defer testing.allocator.free(slab);
    const map = try testing.allocator.alloc(i32, o.capacity / pool_rows);
    defer testing.allocator.free(map);
    var hs: HostSlab = .{ .bytes = slab };
    try testing.expectError(error.FileBusy, Tier.init(testing.allocator, hs.slab(), map, f.path, o));
    // a file a previous run left behind is emptied at open
    f.tier.deinit();
    const fd = std.c.open(f.path, .{ .ACCMODE = .RDWR }, @as(c_uint, 0));
    try testing.expect(fd >= 0);
    _ = std.c.pwrite(fd, slab.ptr, block, 0);
    _ = std.c.close(fd);
    f.tier = try Tier.init(testing.allocator, f.host.slab(), f.map, f.path, o);
    try testing.expectEqual(@as(std.c.off_t, 0), std.c.lseek(f.tier.io.fd, 0, std.c.SEEK.END));
}

test "decode rounds' commits: pools written and device round trips a round" {
    var prng = std.Random.DefaultPrng.init(0xc0de);
    const rnd = prng.random();
    const o: Options = .{ .streams = 2, .layers = 11, .capacity = 64 * seg_positions, .recent_runs = 4, .depth = 16 };
    const f = try Fixture.init(o, 11 * 12 + 11 * 6, 14);
    defer f.deinit();
    const t = &f.tier;
    try t.open(0, 11);
    try t.open(1, 1); // the MTP head's stream
    var pos: usize = 0;
    while (pos < 8 * seg_positions) : (pos += seg_positions) {
        try f.write(rnd, 0, pos, seg_positions);
        try t.commit(0, pos + seg_positions);
        try f.write(rnd, 1, pos, seg_positions);
        try t.commit(1, pos + seg_positions);
    }
    const w0 = t.writes;
    const c0 = t.write_calls;
    const rounds = 400;
    var mpos = pos;
    for (0..rounds) |_| {
        // a round: a window of 4 rows keeping 1 to 4 (2.84 on average in the reference), the head absorbing them
        const R = 4;
        const keep = 1 + rnd.uintLessThan(usize, R);
        try f.write(rnd, 0, pos, R);
        pos += keep;
        t.rewind(0, pos);
        try t.commit(0, pos);
        try f.write(rnd, 1, mpos, keep);
        mpos += keep;
        try t.commit(1, mpos);
    }
    try t.flush();
    for (0..2) |x| try f.verifyAll(x);
    std.debug.print("\ncommit per decode round: {d:.1} pools written, {d:.2} device round trips (slab get + file write wait)\n", .{ @as(f64, @floatFromInt(t.writes - w0)) / rounds, @as(f64, @floatFromInt(t.write_calls - c0)) / rounds });
}

test "prefetchRange brings a layer's whole context back" {
    var prng = std.Random.DefaultPrng.init(0x55);
    const rnd = prng.random();
    const o: Options = .{ .streams = 2, .layers = 1, .capacity = 16 * seg_positions, .recent_runs = 1, .depth = 8 };
    const f = try Fixture.init(o, 24, 2);
    defer f.deinit();
    const t = &f.tier;
    for (0..2) |s| try t.open(s, 1);
    // both streams to 30,720 positions: 15 segments each, 30 runs of rows in a 24-run slab
    for (0..2) |s| {
        var pos: usize = 0;
        while (pos < 15 * seg_positions) : (pos += seg_positions) {
            try f.write(rnd, s, pos, seg_positions);
            try t.commit(s, pos + seg_positions);
        }
    }
    const reads = t.reads;
    try t.ensureRange(0, 0, f.end[0]);
    try testing.expect(t.reads > reads);
    for (0..f.end[0]) |q| {
        const slot = t.map[t.mapBase(0, 0) + q / pool_rows];
        try testing.expect(slot >= 0);
        const at = @as(usize, @intCast(slot)) * block + (q % pool_rows) * o.row_bytes;
        try testing.expectEqualSlices(u8, f.ref[0][q * o.row_bytes ..][0..o.row_bytes], f.host.bytes[at..][0..o.row_bytes]);
    }
    try t.check();
}

test "a fetch that cannot hold its own pools fails instead of evicting them" {
    const o: Options = .{ .streams = 1, .layers = 1, .capacity = 8 * seg_positions, .recent_runs = 1, .depth = 8 };
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const f = try Fixture.init(o, 5, 3);
    defer f.deinit();
    const t = &f.tier;
    try t.open(0, 1);
    var pos: usize = 0;
    while (pos < 8 * seg_positions) : (pos += seg_positions) {
        try f.write(rnd, 0, pos, seg_positions);
        try t.commit(0, pos + seg_positions);
    }
    // 5 runs: home 3, the newest segment 1, one loose run of 512 slots; a selection of 2,048 old pools cannot fit
    var tokens: [2048]i32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(home_positions + i * pool_rows);
    const counts = [_]i32{2048};
    try testing.expectError(error.TierFull, t.fetch(0, 0, &tokens, &counts, 2048));
}

test "writeRuns splits a window at a segment's end and keeps the home block whole" {
    const o: Options = .{ .streams = 1, .layers = 1, .capacity = 8 * seg_positions, .depth = 8 };
    var prng = std.Random.DefaultPrng.init(4);
    const f = try Fixture.init(o, 8, 4);
    defer f.deinit();
    const t = &f.tier;
    try t.open(0, 1);
    try f.write(prng.random(), 0, 2040, 2048); // inside the home block: one piece
    try testing.expect((try t.writeRuns(0, 0, 2040, 2048))[1] == null);
    try f.write(prng.random(), 0, home_positions + seg_positions - 5, 16); // across segments 4 and 5
    const p = try t.writeRuns(0, 0, home_positions + seg_positions - 5, 16);
    try testing.expectEqual(@as(usize, 5), p[0].?.rows);
    try testing.expectEqual(@as(usize, 11), p[1].?.rows);
    try testing.expectEqual(t.denseBase(0, 0), (try t.writeRuns(0, 0, 0, 1))[0].?.base);
}

test "bench (TF_TIER_BENCH=positions): 4 stream-layers' prompt and decode selections, one by one and batched" {
    const arg = std.c.getenv("TF_TIER_BENCH") orelse return error.SkipZigTest;
    const N = try std.fmt.parseInt(usize, std.mem.span(arg), 10);
    const gpa = std.heap.page_allocator;
    const env = struct {
        fn get(name: [*:0]const u8, default: usize) usize {
            const e = std.c.getenv(name) orelse return default;
            return std.fmt.parseInt(usize, std.mem.span(e), 10) catch default;
        }
    }.get;
    const L = 4;
    const o: Options = .{ .streams = 1, .layers = L, .capacity = N, .recent_runs = 32, .depth = env("TF_TIER_DEPTH", 64), .lanes = env("TF_TIER_LANES", 4) };
    const runs = L * (home_runs + o.recent_runs + 2 + (N / seg_positions) / 2);
    const slab = try gpa.alignedAlloc(u8, .fromByteUnits(block), runs * run_pools * block);
    defer gpa.free(slab);
    const map = try gpa.alloc(i32, L * N / pool_rows);
    defer gpa.free(map);
    var buf: [256]u8 = undefined;
    const dir = if (std.c.getenv("TF_TIER_TEST_DIR")) |e| std.mem.span(e) else "/tmp";
    const name = try std.fmt.bufPrint(&buf, "{s}/kv_tier_bench_{d}.bin\x00", .{ dir, std.c.getpid() });
    const path = name[0 .. name.len - 1 :0];
    defer _ = std.c.unlink(path.ptr);
    var hs: HostSlab = .{ .bytes = slab };
    var t = try Tier.init(gpa, hs.slab(), map, path, o);
    defer t.deinit();
    try t.open(0, L);
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    var ts: std.c.timespec = undefined;
    const clock = struct {
        fn ns(x: *std.c.timespec) i128 {
            _ = std.c.clock_gettime(.MONOTONIC, x);
            return @as(i128, x.sec) * 1_000_000_000 + x.nsec;
        }
    }.ns;
    var pos: usize = 0;
    const t0 = clock(&ts);
    while (pos + 2 * seg_positions <= N) : (pos += seg_positions) {
        try t.beginWindow(0, pos, seg_positions);
        try t.commit(0, pos + seg_positions);
    }
    const fill_ms = @as(f64, @floatFromInt(clock(&ts) - t0)) / 1e6;
    const W = 2051;
    var tokens: [L][4 * W]i32 = undefined;
    var counts: [L][4]i32 = undefined;
    var sels: [L]Tier.Selection = undefined;
    var one_ns: i128 = 0;
    var many_ns: i128 = 0;
    var reads_one: u64 = 0;
    var reads_many: u64 = 0;
    const windows = 100;
    for (0..2 * windows) |w| {
        try t.beginWindow(0, pos, 4);
        for (0..L) |l| {
            for (0..4) |r| {
                for (0..512) |i| {
                    const full = (pos + r + 1) / 4;
                    const p = if (rnd.uintLessThan(u8, 3) == 0) full - 1 - rnd.uintLessThan(usize, 4096) else rnd.uintLessThan(usize, full);
                    for (0..4) |j| tokens[l][r * W + i * 4 + j] = @intCast(p * 4 + j);
                }
                counts[l][r] = 2048;
            }
            sels[l] = .{ .s = 0, .l = l, .tokens = &tokens[l], .counts = &counts[l], .W = W };
        }
        const r0 = t.reads;
        const a = clock(&ts);
        if (w % 2 == 0) {
            for (sels) |x| try t.fetch(x.s, x.l, x.tokens, x.counts, x.W);
            one_ns += clock(&ts) - a;
            reads_one += t.reads - r0;
        } else {
            try t.fetchMany(&sels);
            many_ns += clock(&ts) - a;
            reads_many += t.reads - r0;
        }
        pos += 2;
        try t.commit(0, pos);
    }
    const f = @as(f64, windows);
    std.debug.print("\nkv_tier bench: {d} positions x {d} layers, slab {d} runs (about half); prompt bookkeeping and writes {d:.2} ms a 2,048-row chunk; a window of 4 rows x 512 pools on each layer: one fetch a layer {d:.3} ms ({d:.0} pools read), one fetchMany {d:.3} ms ({d:.0} pools read) (aio {}, direct {}, {d} lanes of {d})\n", .{
        N, L, runs, fill_ms / @as(f64, @floatFromInt(N / seg_positions)), @as(f64, @floatFromInt(one_ns)) / 1e6 / f, @as(f64, @floatFromInt(reads_one)) / f, @as(f64, @floatFromInt(many_ns)) / 1e6 / f, @as(f64, @floatFromInt(reads_many)) / f, t.io.aio, t.io.direct, o.lanes, o.depth,
    });
}
