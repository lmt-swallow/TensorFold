//! The DSA latent cache tiered to local NVMe (GLM-5.3-Flash-1: four streams of 1,048,576 positions in bf16 do not
//! fit beside the weights). The host half: which pools are resident, where, and the reads and writes that move them.
//! No CUDA here; `cuda_tier.zig` puts the slab and the map in mapped pinned memory and runs the remap on the device.
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

pub const Error = error{ TierFull, BadToken, NotPersisted, NotOpen, PastCapacity, OpenFailed, ReadFailed, WriteFailed, BadShape, OutOfMemory };

pub const Options = struct {
    streams: usize,
    layers: usize,
    /// Positions a stream holds at most (a multiple of seg_positions).
    capacity: usize,
    /// A latent row's bytes (bf16 kv_lora: 1,024); a pool's (4 rows) must be one block.
    row_bytes: usize = 1024,
    /// Segments past the home block kept linear behind the newest one written.
    recent_runs: usize = 32,
    /// Windows a selected pool stays out of the first eviction pass.
    hot_windows: u32 = 2,
    /// Reads and writes in flight at most (Linux AIO).
    depth: usize = 256,
};

const none: u32 = std.math.maxInt(u32);
const Kind = enum(u8) { free, linear, loose };
const Run = struct { kind: Kind = .free, key0: u32 = 0 };

pub const Piece = struct {
    /// Byte offset from the slab's start of position 0's row (position p's row is at base + p * row_bytes); may be
    /// negative, as a pointer it wraps.
    base: i64,
    pos: usize,
    rows: usize,
};

pub const Tier = struct {
    gpa: std.mem.Allocator,
    o: Options,
    slab: []align(block) u8,
    map: []i32, // [streams, layers, pools]: the slot holding a pool, -1 where none does
    pools: usize, // a stream-layer's
    segs: usize,
    runs: []Run,
    seg_run: []u32, // [streams, layers, segs]: the linear run holding a segment
    home: []u32, // [streams, layers]: the home block's first run
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
    reads: u64 = 0, // pools read from the file so far (statistics)
    writes: u64 = 0,

    const Stream = struct {
        open: bool = false,
        hi: usize = 0, // newest segment a window touched
        committed: usize = 0,
        dirty_lo: usize = 0, // pools below this are persisted wherever full and committed
    };

    /// `slab` (run_pools * block bytes a run, whole runs) and `map` (streams * layers * capacity / 4 entries) are the
    /// caller's: cuda_tier maps both into the device's address space. `path` is the backing file (created, sized).
    pub fn init(gpa: std.mem.Allocator, slab: []align(block) u8, map: []i32, path: [:0]const u8, o: Options) Error!Tier {
        if (o.row_bytes * pool_rows != block or o.capacity % seg_positions != 0 or o.capacity < home_positions) return error.BadShape;
        const run_bytes = run_pools * block;
        if (slab.len % run_bytes != 0 or slab.len == 0) return error.BadShape;
        const pools = o.capacity / pool_rows;
        const keys = o.streams * o.layers * pools;
        if (map.len != keys or keys >= none) return error.BadShape;
        const nslots = slab.len / block;
        if (nslots * pool_rows >= std.math.maxInt(i32)) return error.BadShape;
        var t: Tier = undefined;
        t.gpa = gpa;
        t.o = o;
        t.slab = slab;
        t.map = map;
        t.pools = pools;
        t.segs = o.capacity / seg_positions;
        t.free_slots = .empty;
        t.hand = 0;
        t.window = 1;
        t.gen = 1;
        t.reads = 0;
        t.writes = 0;
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
        t.io = try Io.open(gpa, path, o.depth);
        return t;
    }

    pub fn deinit(t: *Tier) void {
        t.io.close();
        t.gpa.free(t.streams);
        t.persisted.deinit(t.gpa);
        t.ref.deinit(t.gpa);
        t.gpa.free(t.used);
        t.gpa.free(t.pin);
        t.gpa.free(t.owner);
        t.gpa.free(t.home);
        t.gpa.free(t.seg_run);
        t.gpa.free(t.runs);
        t.free_slots.deinit(t.gpa);
    }

    /// The map index of stream `s`'s layer `l`'s pool 0 (`map[mapBase(s, l) + position / 4]`).
    pub fn mapBase(t: *const Tier, s: usize, l: usize) usize {
        return (s * t.o.layers + l) * t.pools;
    }

    fn key(t: *const Tier, s: usize, l: usize, pool: usize) u32 {
        return @intCast(t.mapBase(s, l) + pool);
    }

    fn fileOffset(k: u32) u64 {
        return @as(u64, k) * block;
    }

    fn slotBytes(t: *const Tier, slot: usize) []align(block) u8 {
        return @alignCast(t.slab[slot * block ..][0..block]);
    }

    /// A stream's home blocks (one a layer); its previous contents are dropped.
    pub fn open(t: *Tier, s: usize) Error!void {
        if (t.streams[s].open) t.close(s);
        for (0..t.o.layers) |l| {
            const r = try t.takeRuns(home_runs);
            t.home[s * t.o.layers + l] = r;
            for (0..home_runs) |i| try t.placeLinear(s, l, i, r + @as(u32, @intCast(i)));
        }
        t.streams[s] = .{ .open = true };
    }

    /// Drops a stream: its runs and slots are free, its pools forgotten.
    pub fn close(t: *Tier, s: usize) void {
        t.io.finish() catch {};
        for (0..t.o.layers) |l| {
            const k0 = t.mapBase(s, l);
            for (0..t.pools) |p| {
                const slot = t.map[k0 + p];
                if (slot >= 0) {
                    const u: usize = @intCast(slot);
                    t.owner[u] = none;
                    if (t.runs[u / run_pools].kind == .loose) t.free_slots.append(t.gpa, @intCast(u)) catch {};
                }
                t.map[k0 + p] = -1;
                t.persisted.unset(k0 + p);
            }
            for (t.seg_run[(s * t.o.layers + l) * t.segs ..][0..t.segs]) |*r| {
                if (r.* != none) t.runs[r.*] = .{};
                r.* = none;
            }
            t.home[s * t.o.layers + l] = none;
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
        try t.io.finish();
        t.window +%= 1;
        const first = pos / seg_positions;
        const last = (pos + R - 1) / seg_positions;
        st.hi = @max(st.hi, last);
        for (0..t.o.layers) |l| {
            const sr = t.seg_run[(s * t.o.layers + l) * t.segs ..][0..t.segs];
            for (home_runs..t.segs) |g| if (sr[g] != none and g < first and g + t.o.recent_runs <= st.hi) t.demote(sr, g);
        }
        for (0..t.o.layers) |l| {
            for (first..last + 1) |g| if (t.seg_run[(s * t.o.layers + l) * t.segs + g] == none) {
                const r = try t.takeRuns(1);
                try t.placeLinear(s, l, g, r);
            };
            const k0 = t.mapBase(s, l);
            for (pos / pool_rows..(pos + R - 1) / pool_rows + 1) |p| t.persisted.unset(k0 + p);
        }
        try t.io.finish();
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

    /// Makes resident every pool the rows' selected positions name (`tokens [R, W]`, row r's first counts[r]);
    /// returns once their bytes are in the slab and the map names their slots. None of them is evicted before the
    /// next fetch.
    pub fn fetch(t: *Tier, s: usize, l: usize, tokens: []const i32, counts: []const i32, W: usize) Error!void {
        if (!t.streams[s].open) return error.NotOpen;
        try t.io.finish();
        t.gen +%= 1;
        const k0 = t.mapBase(s, l);
        var need: std.ArrayList(u32) = .empty;
        defer need.deinit(t.gpa);
        for (counts, 0..) |c, r| {
            if (c < 0 or @as(usize, @intCast(c)) > W) return error.BadToken;
            for (tokens[r * W ..][0..@intCast(c)]) |tok| {
                if (tok < 0 or @as(usize, @intCast(tok)) >= t.o.capacity) return error.BadToken;
                const k: u32 = @intCast(k0 + @as(usize, @intCast(tok)) / pool_rows);
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
        errdefer for (need.items) |k| if (t.map[k] == -2) {
            t.map[k] = -1;
        };
        try t.readIn(need.items);
        try t.io.finish();
    }

    /// Starts reading layer `l`'s pools below position `end` that are not resident (prefill's next layer); the
    /// next call on the tier waits for them. `ensureRange` waits at once.
    pub fn prefetchRange(t: *Tier, s: usize, l: usize, end: usize) Error!void {
        if (!t.streams[s].open) return error.NotOpen;
        if (end > t.o.capacity) return error.PastCapacity;
        try t.io.finish();
        t.gen +%= 1;
        const k0 = t.mapBase(s, l);
        var need: std.ArrayList(u32) = .empty;
        defer need.deinit(t.gpa);
        for (0..(end + pool_rows - 1) / pool_rows) |p| {
            const k: u32 = @intCast(k0 + p);
            const slot = t.map[k];
            if (slot >= 0) {
                t.touch(@intCast(slot));
            } else if (t.persisted.isSet(k)) {
                t.map[k] = -2;
                try need.append(t.gpa, k);
            }
        }
        errdefer for (need.items) |k| if (t.map[k] == -2) {
            t.map[k] = -1;
        };
        try t.readIn(need.items);
    }

    pub fn ensureRange(t: *Tier, s: usize, l: usize, end: usize) Error!void {
        try t.prefetchRange(s, l, end);
        try t.io.finish();
    }

    /// Selected positions to row indices at the slab's base (the device's remap, on the host): fetch ran.
    pub fn remap(t: *const Tier, s: usize, l: usize, tokens: []i32, counts: []const i32, W: usize) void {
        const k0 = t.mapBase(s, l);
        for (counts, 0..) |c, r| for (tokens[r * W ..][0..@intCast(c)]) |*tok| {
            const p: usize = @intCast(tok.*);
            tok.* = t.map[k0 + p / pool_rows] * pool_rows + @as(i32, @intCast(p % pool_rows));
        };
    }

    /// Positions below `pos` of stream `s` are final: every whole pool below it reaches the file.
    pub fn commit(t: *Tier, s: usize, pos: usize) Error!void {
        const st = &t.streams[s];
        if (!st.open) return error.NotOpen;
        try t.io.finish();
        const hi = pos / pool_rows;
        if (hi > st.dirty_lo) {
            for (0..t.o.layers) |l| {
                const k0 = t.mapBase(s, l);
                var p = st.dirty_lo;
                while (p < hi) {
                    const k: u32 = @intCast(k0 + p);
                    if (t.persisted.isSet(k)) {
                        p += 1;
                        continue;
                    }
                    const slot = t.map[k];
                    if (slot < 0) return error.NotPersisted; // a pool never written: nothing to commit
                    // a stretch contiguous in the slab and the file
                    var n: usize = 1;
                    while (p + n < hi and !t.persisted.isSet(k + n) and t.map[k + n] == slot + @as(i32, @intCast(n))) n += 1;
                    try t.io.add(.write, t.slab[@as(usize, @intCast(slot)) * block ..][0 .. n * block], fileOffset(k));
                    for (0..n) |i| t.persisted.set(k + i);
                    t.writes += n;
                    p += n;
                }
            }
            st.dirty_lo = hi;
        }
        st.committed = @max(st.committed, pos);
        try t.io.finish();
    }

    /// Positions from `pos` on are dropped (rejected drafts, a shortened prompt): the next window rewrites them.
    pub fn rewind(t: *Tier, s: usize, pos: usize) void {
        const st = &t.streams[s];
        st.committed = @min(st.committed, pos);
    }

    // -- slots and runs --------------------------------------------------------------------------------------------

    fn touch(t: *Tier, slot: u32) void {
        t.pin[slot] = t.gen;
        t.used[slot] = t.window;
        t.ref.set(slot);
    }

    /// Pools `keys` (map -2, persisted) into loose slots, their reads queued.
    fn readIn(t: *Tier, keys: []const u32) Error!void {
        for (keys) |k| {
            const slot = try t.allocLoose();
            t.owner[slot] = k;
            t.map[k] = @intCast(slot);
            t.touch(slot);
            try t.io.add(.read, t.slotBytes(slot), fileOffset(k));
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
                @memcpy(t.slotBytes(slot), t.slotBytes(u));
                t.owner[u] = none;
                try t.free_slots.append(t.gpa, u);
            } else if (t.persisted.isSet(k)) {
                try t.io.add(.read, t.slotBytes(slot), fileOffset(k));
                t.reads += 1;
            }
            t.owner[slot] = k;
            t.map[k] = @intCast(slot);
            t.used[slot] = t.window;
        }
    }

    fn demote(t: *Tier, sr: []u32, g: usize) void {
        const r = sr[g];
        sr[g] = none;
        t.runs[r].kind = .loose;
        for (r * run_pools..(r + 1) * run_pools) |slot| {
            t.ref.unset(slot);
            if (t.owner[slot] == none) t.free_slots.append(t.gpa, @intCast(slot)) catch {};
        }
    }

    fn evictable(t: *const Tier, slot: usize, hot: bool) bool {
        const k = t.owner[slot];
        if (k == none or !t.persisted.isSet(k) or t.pin[slot] == t.gen) return false;
        return !hot or t.used[slot] +% t.o.hot_windows <= t.window;
    }

    fn evict(t: *Tier, slot: usize) void {
        t.map[t.owner[slot]] = -1;
        t.owner[slot] = none;
        t.ref.unset(slot);
    }

    /// A loose slot holding nothing: a free one, else the clock's choice among written pools no current call names
    /// (first sparing the pools of the last hot_windows windows).
    fn allocLoose(t: *Tier) Error!u32 {
        while (t.free_slots.pop()) |slot| {
            if (t.runs[slot / run_pools].kind == .loose and t.owner[slot] == none) return slot;
        }
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
                return @intCast(b);
            }
        }
        var best: ?usize = null;
        var best_cost: usize = std.math.maxInt(usize);
        var a: usize = 0;
        while (a + n <= nr) : (a += 1) {
            var cost: usize = 0;
            var ok = true;
            for (a..a + n) |r| switch (t.runs[r].kind) {
                .free => {},
                .linear => {
                    ok = false;
                    break;
                },
                .loose => for (r * run_pools..(r + 1) * run_pools) |slot| {
                    if (t.owner[slot] != none) cost += if (t.evictable(slot, false)) 1 else run_pools;
                },
            };
            if (ok and cost < best_cost) {
                best = a;
                best_cost = cost;
                if (cost == 0) break;
            }
        }
        const b = best orelse return error.TierFull;
        // out of the loose set first, so no move lands inside
        for (b..b + n) |r| t.runs[r].kind = .free;
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
            @memcpy(t.slotBytes(dst), t.slotBytes(slot));
            t.owner[dst] = k;
            t.map[k] = @intCast(dst);
            t.pin[dst] = t.pin[slot];
            t.used[dst] = t.used[slot];
            t.owner[slot] = none;
        }
        for (b * run_pools..(b + n) * run_pools) |slot| t.ref.unset(slot);
        return @intCast(b);
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
    }
};

// -- I/O -----------------------------------------------------------------------------------------------------------

/// Block reads and writes on the backing file: Linux AIO on O_DIRECT where the kernel and file system allow it (the
/// engine runs in Docker, whose default seccomp profile refuses io_uring but not AIO), else positional calls.
/// `add` queues (submitting when the queue is full), `finish` returns once every queued transfer landed.
pub const Io = struct {
    gpa: std.mem.Allocator,
    fd: c_int,
    ctx: usize = 0,
    aio: bool = false,
    direct: bool = false,
    depth: usize,
    queue: std.ArrayList(Req) = .empty,
    cbs: []Iocb = &.{},
    ptrs: []*Iocb = &.{},
    failed: bool = false,

    const Op = enum { read, write };
    const Req = struct { op: Op, buf: []u8, off: u64 };

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

    pub fn open(gpa: std.mem.Allocator, path: [:0]const u8, depth: usize) Error!Io {
        var io: Io = .{ .gpa = gpa, .fd = -1, .depth = @max(depth, 1) };
        if (builtin.os.tag == .linux) {
            io.fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true, .DIRECT = true }, @as(c_uint, 0o600));
            io.direct = io.fd >= 0;
        }
        if (io.fd < 0) io.fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true }, @as(c_uint, 0o600));
        if (io.fd < 0) return error.OpenFailed;
        errdefer _ = std.c.close(io.fd);
        if (builtin.os.tag == .linux) {
            if (linux.errno(linux.syscall2(.io_setup, io.depth, @intFromPtr(&io.ctx))) == .SUCCESS) io.aio = true;
            io.cbs = try gpa.alloc(Iocb, io.depth);
            io.ptrs = try gpa.alloc(*Iocb, io.depth);
        }
        return io;
    }

    pub fn close(io: *Io) void {
        io.finish() catch {};
        if (builtin.os.tag == .linux) {
            if (io.aio) _ = linux.syscall1(.io_destroy, io.ctx);
            io.gpa.free(io.cbs);
            io.gpa.free(io.ptrs);
        }
        io.queue.deinit(io.gpa);
        _ = std.c.close(io.fd);
    }

    pub fn add(io: *Io, op: Op, buf: []u8, off: u64) Error!void {
        try io.queue.append(io.gpa, .{ .op = op, .buf = buf, .off = off });
        if (io.queue.items.len >= io.depth * 16) try io.finish();
    }

    pub fn finish(io: *Io) Error!void {
        defer io.queue.clearRetainingCapacity();
        if (io.queue.items.len == 0) return;
        if (builtin.os.tag == .linux and io.aio) return io.finishAio();
        for (io.queue.items) |q| {
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

    /// Every queued transfer through the AIO context, at most `depth` in flight, polled to completion.
    fn finishAio(io: *Io) Error!void {
        const items = io.queue.items;
        var next: usize = 0;
        var inflight: usize = 0;
        var free: std.ArrayList(u32) = .empty;
        defer free.deinit(io.gpa);
        for (0..io.depth) |i| try free.append(io.gpa, @intCast(io.depth - 1 - i));
        var bad = false;
        while (next < items.len or inflight > 0) {
            var n: usize = 0;
            while (next < items.len and free.items.len > 0) : (next += 1) {
                const q = items[next];
                const c = free.pop().?;
                io.cbs[c] = .{
                    .data = (@as(u64, c) << 32) | q.buf.len,
                    .lio_opcode = if (q.op == .read) 0 else 1,
                    .fildes = @intCast(io.fd),
                    .buf = @intFromPtr(q.buf.ptr),
                    .nbytes = q.buf.len,
                    .offset = @intCast(q.off),
                };
                io.ptrs[n] = &io.cbs[c];
                n += 1;
            }
            var sent: usize = 0;
            while (sent < n) {
                const r: isize = @bitCast(linux.syscall3(.io_submit, io.ctx, n - sent, @intFromPtr(io.ptrs[sent..].ptr)));
                if (r == -@as(isize, @intFromEnum(linux.E.AGAIN)) or r == -@as(isize, @intFromEnum(linux.E.INTR))) continue;
                if (r <= 0) return error.ReadFailed;
                sent += @intCast(r);
            }
            inflight += n;
            if (inflight == 0) break;
            var ev: [256]IoEvent = undefined;
            const got: isize = @bitCast(linux.syscall5(.io_getevents, io.ctx, 1, ev.len, @intFromPtr(&ev), 0));
            if (got < 0) {
                if (got == -@as(isize, @intFromEnum(linux.E.INTR))) continue;
                return error.ReadFailed;
            }
            for (ev[0..@intCast(got)]) |e| {
                if (e.res != @as(i64, @intCast(e.data & 0xFFFF_FFFF))) bad = true;
                try free.append(io.gpa, @intCast(e.data >> 32));
                inflight -= 1;
            }
        }
        if (bad) return error.ReadFailed;
    }
};

// -- tests ---------------------------------------------------------------------------------------------------------

const testing = std.testing;

const Fixture = struct {
    slab: []align(block) u8,
    map: []i32,
    path_buf: [256]u8,
    path: [:0]u8,
    tier: Tier,
    ref: [][]u8, // [streams * layers][capacity * row_bytes]: the whole cache the tier stands for
    end: []usize, // a stream's positions written

    fn init(o: Options, runs: usize, seed: u64) !*Fixture {
        const gpa = testing.allocator;
        const f = try gpa.create(Fixture);
        errdefer gpa.destroy(f);
        f.slab = try gpa.alignedAlloc(u8, .fromByteUnits(block), runs * run_pools * block);
        @memset(f.slab, 0xAA);
        f.map = try gpa.alloc(i32, o.streams * o.layers * o.capacity / pool_rows);
        // TF_TIER_TEST_DIR: a directory on the NVMe's file system (O_DIRECT and AIO; /tmp may be tmpfs)
        const dir = if (std.c.getenv("TF_TIER_TEST_DIR")) |e| std.mem.span(e) else "/tmp";
        const name = try std.fmt.bufPrint(&f.path_buf, "{s}/kv_tier_test_{d}_{x}.bin\x00", .{ dir, std.c.getpid(), seed });
        f.path = name[0 .. name.len - 1 :0];
        f.tier = try Tier.init(gpa, f.slab, f.map, f.path, o);
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
        gpa.free(f.slab);
        gpa.destroy(f);
    }

    /// A window's rows written on every layer through writeRuns, as the latent write does.
    fn write(f: *Fixture, rnd: std.Random, s: usize, pos: usize, R: usize) !void {
        const t = &f.tier;
        const rb = t.o.row_bytes;
        try t.beginWindow(s, pos, R);
        for (0..t.o.layers) |l| {
            const pieces = try t.writeRuns(s, l, pos, R);
            var covered: usize = 0;
            for (pieces) |pc| if (pc) |p| {
                for (p.pos..p.pos + p.rows) |q| {
                    const at: usize = @intCast(p.base + @as(i64, @intCast(q * rb)));
                    const dst = t.slab[at..][0..rb];
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
        const l = rnd.uintLessThan(usize, t.o.layers);
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
            const got = t.slab[@as(usize, @intCast(tokens[r * W + i])) * rb ..][0..rb];
            try testing.expectEqualSlices(u8, want, got);
        };
    }
};

test "a selection reads the bytes last written, through windows, drafts, commits and evictions" {
    var prng = std.Random.DefaultPrng.init(0x7131);
    const rnd = prng.random();
    // 2 streams x 2 layers x 32,768 positions = 64 runs of latents in a 32-run slab: the home blocks take 12 runs,
    // the recent windows up to 12, the 8 loose runs hold what else fits and the rest comes back from the file
    const o: Options = .{ .streams = 2, .layers = 2, .capacity = 16 * seg_positions, .recent_runs = 2, .depth = 32 };
    const f = try Fixture.init(o, 32, 1);
    defer f.deinit();
    const t = &f.tier;
    for (0..o.streams) |s| try t.open(s);
    var committed: [2]usize = .{ 0, 0 };
    var steps: usize = 0;
    while (steps < 400) : (steps += 1) {
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
    }
    try testing.expect(t.reads > 0); // the slab is too small to hold everything: some pools came back from the file
    try testing.expect(t.writes > 0);
    if (builtin.os.tag == .linux and std.c.getenv("TF_TIER_TEST_DIR") != null) {
        // on the NVMe the transfers take the engine's path
        try testing.expect(t.io.aio and t.io.direct);
    }
}

test "prefetchRange brings a layer's whole context back" {
    var prng = std.Random.DefaultPrng.init(0x55);
    const rnd = prng.random();
    const o: Options = .{ .streams = 2, .layers = 1, .capacity = 16 * seg_positions, .recent_runs = 1, .depth = 8 };
    const f = try Fixture.init(o, 24, 2);
    defer f.deinit();
    const t = &f.tier;
    for (0..2) |s| try t.open(s);
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
        try testing.expectEqualSlices(u8, f.ref[0][q * o.row_bytes ..][0..o.row_bytes], t.slab[at..][0..o.row_bytes]);
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
    try t.open(0);
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
    try t.open(0);
    try f.write(prng.random(), 0, 2040, 2048); // inside the home block: one piece
    try testing.expect((try t.writeRuns(0, 0, 2040, 2048))[1] == null);
    try f.write(prng.random(), 0, home_positions + seg_positions - 5, 16); // across segments 4 and 5
    const p = try t.writeRuns(0, 0, home_positions + seg_positions - 5, 16);
    try testing.expectEqual(@as(usize, 5), p[0].?.rows);
    try testing.expectEqual(@as(usize, 11), p[1].?.rows);
    try testing.expectEqual(t.denseBase(0, 0), (try t.writeRuns(0, 0, 0, 1))[0].?.base);
}
