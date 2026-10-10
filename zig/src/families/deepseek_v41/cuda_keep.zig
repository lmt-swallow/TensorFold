//! Kept prompts (multi.py `Kept`, TF_DS_KEEP): the host-side book of the window's extents and of the finished
//! prompts that stay in it, so a later prompt that starts the same prefills only what follows. Every function here is
//! a pure function of state every rank holds (the extents, the kept entries, the record), so the ranks keep the same
//! book; rank 0 alone chooses (`match`, `covered`) and sends its choice in the admission and free records.
//!
//! - **What is kept.** A finished stream's prompt rows up to its last kept chunk boundary (`top`): its extent's
//!   positions [base, base + size(top)) (the compressed and indexer rows, which are an extent's) and, at each kept
//!   boundary, a snapshot of its slot's window rings and compressor inputs (which are a slot's, not an extent's; with
//!   bounded replay the encoder layers' rings only). Only prompt rows are kept: a verify round's rows are not a
//!   prompt chunk's bits, and a reply's rows go back with the rest of the extent.
//! - **Boundaries.** A prompt of n tokens snapshots at the chunk boundaries `marksFor(n)`: the doubling ones from the
//!   first chunk, its last two, and for n >= shrink_min the ones at 3/4 and 7/8 of n. When it is kept it keeps at most
//!   `marks` of them (`keptMarks`): the doubling ones, its last two, with shrink every one in its upper half, then the
//!   earliest. A prompt that continued a kept one inherits that one's boundaries up to where it continued.
//! - **Matching.** A new prompt continues the kept prompt it shares the longest kept boundary with (the most recently
//!   used of equals, then the oldest entry): the boundary is at most the prompts' common start and below the new
//!   prompt's last row; with bounded replay a window below it (the decoder layers' rows there are the prompt's own).
//!   The continued prompt's chunks from the boundary are a fresh prefill's own chunks, so its reply is the fresh one.
//! - **Eviction.** An admission takes the first free extent that holds it; failing that, the kept prompt it continues
//!   grows in place, then the other kept prompts give their room back oldest first (with shrink, cut back to the
//!   largest kept boundary whose freed tail makes the room before being forgotten whole), and last the continued one
//!   itself moves. A finished prompt drops the kept prompts it covers (the same ids to their top, every boundary of
//!   theirs kept again), and the oldest entries go past `entries`.
const std = @import("std");

/// Streams a pool holds at most (cuda_lanes.zig max_slots).
pub const max_slots: usize = 4;
/// Kept prompts at most (TF_DS_KEEP_ENTRIES is capped one below: a finishing stream is kept before the trim).
pub const max_kept: usize = 16;
/// Boundaries a stream or kept prompt holds at most (a 2^30-token prompt's doubling ones, its last two, its 3/4 and
/// 7/8, and the ones it inherited).
pub const max_marks: usize = 48;
/// Extents start and end on multiples of this many positions (multi.py ALIGN).
pub const extent_align: usize = 2048;

/// multi.py `Extents`: first-fit extents of a window of `total` positions in `extent_align` steps; a freed extent
/// merges with its neighbours. The gaps are free, disjoint, sorted, and never touch.
pub const Extents = struct {
    total: usize,
    gaps: [max_slots + max_kept + 2][2]usize = undefined,
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

    /// multi.py `take`: the first gap that holds `rows` (aligned up), taken; null when none does.
    pub fn take(self: *Extents, rows: usize) ?usize {
        const start = self.first(rows) orelse return null;
        if (!self.takeAt(start, rows)) return null;
        return start;
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

/// The knobs (TF_DS_KEEP, TF_DS_KEEP_ENTRIES, TF_DS_KEEP_MARKS, TF_DS_KEEP_SHRINK, TF_DS_KEEP_SHRINK_MIN as in the
/// fork; TF_DS_KEEP_SNAPS, the ring snapshots the pool holds; TF_DS_KEEP_CHECK=1, every continued prompt also
/// prefilled from 0 and compared). Both ranks must read the same values.
pub const Settings = struct {
    on: bool = true,
    entries: usize = 8,
    marks: usize = 10,
    shrink: bool = true,
    shrink_min: usize = 131072,
    snaps: usize = 32,
    check: bool = false,

    pub fn fromEnv() Settings {
        var s: Settings = .{};
        s.on = flag("TF_DS_KEEP", true);
        s.entries = @min(int("TF_DS_KEEP_ENTRIES", s.entries), max_kept - 1);
        s.marks = @max(@min(int("TF_DS_KEEP_MARKS", s.marks), max_marks), 2);
        s.shrink = flag("TF_DS_KEEP_SHRINK", true);
        s.shrink_min = int("TF_DS_KEEP_SHRINK_MIN", s.shrink_min);
        s.snaps = @min(int("TF_DS_KEEP_SNAPS", s.snaps), 255);
        s.check = flag("TF_DS_KEEP_CHECK", false);
        if (s.entries == 0 or s.snaps == 0) s.on = false;
        return s;
    }

    fn env(name: [:0]const u8) ?[]const u8 {
        const v = std.mem.span(std.c.getenv(name) orelse return null);
        return if (v.len == 0) null else v;
    }

    fn flag(name: [:0]const u8, default: bool) bool {
        const v = env(name) orelse return default;
        return std.mem.eql(u8, v, "1");
    }

    fn int(name: [:0]const u8, default: usize) usize {
        const v = env(name) orelse return default;
        return std.fmt.parseInt(usize, std.mem.trim(u8, v, " "), 10) catch default;
    }
};

/// A kept boundary: the prompt position and the ring snapshot taken there (an index into the pool's snapshots).
pub const Mark = struct { at: usize, snap: u16 };

/// Boundaries in ascending order.
pub const Marks = struct {
    items: [max_marks]Mark = undefined,
    n: usize = 0,

    pub fn slice(self: *const Marks) []const Mark {
        return self.items[0..self.n];
    }

    /// The snapshot at boundary `at`, null when it is not one of these.
    pub fn find(self: *const Marks, at: usize) ?u16 {
        for (self.slice()) |m| if (m.at == at) return m.snap;
        return null;
    }

    /// `m` in its place (its boundary must be new and there must be room).
    pub fn add(self: *Marks, m: Mark) void {
        var i = self.n;
        while (i > 0 and self.items[i - 1].at > m.at) : (i -= 1) self.items[i] = self.items[i - 1];
        self.items[i] = m;
        self.n += 1;
    }

    pub fn last(self: *const Marks) ?usize {
        return if (self.n == 0) null else self.items[self.n - 1].at;
    }
};

/// Boundaries as positions only (`marksFor`, `keptMarks`), ascending.
pub const Points = struct {
    items: [max_marks]usize = undefined,
    n: usize = 0,

    pub fn slice(self: *const Points) []const usize {
        return self.items[0..self.n];
    }

    pub fn has(self: *const Points, at: usize) bool {
        return std.mem.indexOfScalar(usize, self.slice(), at) != null;
    }

    fn put(self: *Points, at: usize) void {
        if (self.has(at) or self.n == max_marks) return;
        var i = self.n;
        while (i > 0 and self.items[i - 1] > at) : (i -= 1) self.items[i] = self.items[i - 1];
        self.items[i] = at;
        self.n += 1;
    }
};

/// A kept prompt (multi.py `Kept`): positions [base, base + size) hold the compressed and indexer rows of its first
/// `top` prompt tokens; `marks` its slot's ring snapshots at its kept boundaries (the last one `top`); `keys` its ids
/// to `top` (a prefix of `ids`, which it owns); `tick` the admission count when it was kept or last continued.
pub const Kept = struct {
    eid: u32,
    base: usize,
    size: usize,
    top: usize,
    tick: u64,
    hits: u32 = 0,
    replay: bool,
    marks: Marks,
    ids: []i32,

    pub fn keys(self: *const Kept) []const i32 {
        return self.ids[0..self.top];
    }
};

/// The admission record's reuse: the prompt continues kept prompt `eid` from boundary `cut`.
pub const Reuse = struct { eid: u32, cut: usize };

/// Where an admission's extent came from: a free one ("fresh" without a kept prompt, "copy" with one: its rows are
/// copied in), the continued kept prompt's own grown in place ("here"), or anywhere once that one gave its own back
/// too ("move": its rows are copied from where they were).
pub const How = enum { fresh, copy, here, move };

pub const Placed = struct { base: usize, how: How };

/// What a book operation let go: the snapshots to release and the kept ids to free (the caller's to free).
pub const Released = struct {
    snaps: [max_kept * max_marks + max_marks]u16 = undefined,
    n_snaps: usize = 0,
    ids: [max_kept + 1][]i32 = undefined,
    n_ids: usize = 0,
    shrunk: usize = 0,
    forgotten: usize = 0,

    fn snap(self: *Released, s: u16) void {
        self.snaps[self.n_snaps] = s;
        self.n_snaps += 1;
    }
};

/// The kept prompts of one pool. `released` null: a trial (a copy whose operations free nothing, for a rank to see
/// what an admission will do before it runs it).
pub const Book = struct {
    set: Settings,
    /// Prompt rows a chunk: every boundary is a multiple of it.
    chunk: usize,
    /// The attention window (with bounded replay a continued prompt's boundary stays this far below its end).
    window: usize,
    kept: [max_kept]Kept = undefined,
    n: usize = 0,
    next_eid: u32 = 0,
    ticks: u64 = 0,
    released: ?*Released = null,

    pub fn slice(self: *const Book) []const Kept {
        return self.kept[0..self.n];
    }

    pub fn find(self: *const Book, eid: u32) ?usize {
        for (self.kept[0..self.n], 0..) |k, i| if (k.eid == eid) return i;
        return null;
    }

    /// A copy whose operations free nothing.
    pub fn trial(self: *const Book) Book {
        var t = self.*;
        t.released = null;
        return t;
    }

    /// multi.py `_match`: the kept prompt this prompt continues the furthest and the boundary, else null.
    pub fn match(self: *const Book, ids: []const u32, replay: bool) ?Reuse {
        const limit = if (replay) ids.len -| self.window else ids.len -| 1;
        var best: ?struct { i: usize, cut: usize } = null;
        for (self.kept[0..self.n], 0..) |*k, i| {
            if (k.replay != replay) continue;
            const keys = k.keys();
            const n = @min(keys.len, ids.len);
            var common: usize = 0;
            while (common < n and keys[common] >= 0 and @as(u32, @intCast(keys[common])) == ids[common]) common += 1;
            const bound = @min(common, limit);
            var cut: usize = 0;
            for (k.marks.slice()) |m| if (m.at <= bound) {
                cut = @max(cut, m.at);
            };
            if (cut == 0) continue;
            if (best) |b| {
                if (cut < b.cut or (cut == b.cut and k.tick <= self.kept[b.i].tick)) continue;
            }
            best = .{ .i = i, .cut = cut };
        }
        const b = best orelse return null;
        return .{ .eid = self.kept[b.i].eid, .cut = b.cut };
    }

    /// multi.py `_marks`: the chunk boundaries a prompt of `n` tokens snapshots at.
    pub fn marksFor(self: *const Book, n: usize) Points {
        const C = self.chunk;
        var p: Points = .{};
        if (C == 0 or n < C) return p;
        const top = n / C * C;
        p.put(top);
        if (top >= 2 * C) p.put(top - C);
        var b = C;
        while (b <= n) : (b *= 2) p.put(b);
        if (self.set.shrink and n >= self.set.shrink_min) for ([_][2]usize{ .{ 3, 4 }, .{ 7, 8 } }) |f| {
            const at = n * f[0] / f[1] / C * C;
            if (at >= C) p.put(at);
        };
        return p;
    }

    /// multi.py `_kept_marks`: the boundaries a finished stream keeps, `marks` at most: the doubling ones, its last
    /// two, with shrink every one in its upper half, then the earliest.
    pub fn keptMarks(self: *const Book, have: []const usize) Points {
        const C = self.chunk;
        var p: Points = .{};
        if (have.len <= self.set.marks) {
            for (have) |b| p.put(b);
            return p;
        }
        const top = have[have.len - 1];
        for (have, 0..) |b, i| {
            const q = b / C;
            const doubling = q & (q - 1) == 0;
            if (doubling or i + 2 >= have.len or (self.set.shrink and 2 * b > top)) p.put(b);
        }
        var left = self.set.marks -| p.n;
        for (have) |b| {
            if (left == 0) break;
            if (p.has(b)) continue;
            p.put(b);
            left -= 1;
        }
        return p;
    }

    /// multi.py `_room`: whether `need` positions fit once every kept prompt gave its room back.
    pub fn room(self: *const Book, ex: *const Extents, need: usize) bool {
        var t = ex.*;
        for (self.kept[0..self.n]) |k| t.give(k.base, k.size);
        return t.first(need) != null;
    }

    fn forgetAt(self: *Book, ex: *Extents, i: usize, give: bool) void {
        const k = self.kept[i];
        if (give) ex.give(k.base, k.size);
        if (self.released) |r| {
            for (k.marks.slice()) |m| r.snap(m.snap);
            r.ids[r.n_ids] = k.ids;
            r.n_ids += 1;
            r.forgotten += 1;
        }
        std.mem.copyForwards(Kept, self.kept[i .. self.n - 1], self.kept[i + 1 .. self.n]);
        self.n -= 1;
    }

    /// Kept prompt `eid` forgotten, its extent free again (nothing when it is not kept).
    pub fn forget(self: *Book, ex: *Extents, eid: u32) void {
        if (self.find(eid)) |i| self.forgetAt(ex, i, true);
    }

    /// The least recently used kept prompt (the oldest entry of equals), other than `except`.
    fn oldest(self: *const Book, except: ?u32) ?usize {
        var best: ?usize = null;
        for (self.kept[0..self.n], 0..) |k, i| {
            if (except != null and k.eid == except.?) continue;
            if (best == null or k.tick < self.kept[best.?].tick) best = i;
        }
        return best;
    }

    /// The least recently used kept prompt forgotten; false when none is kept.
    pub fn forgetOldest(self: *Book, ex: *Extents) bool {
        const i = self.oldest(null) orelse return false;
        self.forgetAt(ex, i, true);
        return true;
    }

    /// multi.py `_shrink`: kept prompt `i` cut back to its largest kept boundary below its top whose freed tail
    /// makes room for `need` positions; false when none does (nothing changes then).
    fn shrinkAt(self: *Book, ex: *Extents, i: usize, need: usize) bool {
        const k = &self.kept[i];
        var j = k.marks.n;
        while (j > 0) {
            j -= 1;
            const m = k.marks.items[j].at;
            if (m >= k.top) continue;
            const sz = Extents.size(m);
            if (sz >= k.size) continue;
            var t = ex.*;
            t.give(k.base + sz, k.size - sz);
            if (t.first(need) == null) continue;
            ex.give(k.base + sz, k.size - sz);
            k.size = sz;
            k.top = m;
            var kept: Marks = .{};
            for (k.marks.slice()) |mk| {
                if (mk.at <= m) kept.add(mk) else if (self.released) |r| r.snap(mk.snap);
            }
            k.marks = kept;
            if (self.released) |r| r.shrunk += 1;
            return true;
        }
        return false;
    }

    /// multi.py `_place`: an extent for `need` positions, continuing kept prompt `src` (null: none), evicting kept
    /// prompts as it must; null when even an empty window of kept prompts has no room.
    pub fn place(self: *Book, ex: *Extents, need: usize, src: ?u32) ?Placed {
        while (true) {
            if (ex.take(need)) |base| return .{ .base = base, .how = if (src != null) .copy else .fresh };
            if (src) |eid| {
                const si = self.find(eid) orelse return null;
                const s = self.kept[si];
                ex.give(s.base, s.size);
                if (ex.takeAt(s.base, need)) {
                    self.forgetAt(ex, si, false);
                    return .{ .base = s.base, .how = .here };
                }
                if (!ex.takeAt(s.base, s.size)) unreachable; // just given back
            }
            if (self.oldest(src)) |o| {
                if (!(self.set.shrink and self.shrinkAt(ex, o, need))) self.forgetAt(ex, o, true);
                continue;
            }
            const eid = src orelse return null;
            self.forgetAt(ex, self.find(eid).?, true);
            const base = ex.take(need) orelse return null;
            return .{ .base = base, .how = .move };
        }
    }

    /// multi.py `_covered`: the kept prompts a finishing prompt (`ids`, its boundaries `have`) will cover: the same
    /// ids to their top and every boundary of theirs kept again. Into `out`; returns how many.
    pub fn covered(self: *const Book, ids: []const i32, have: []const usize, replay: bool, out: []u32) usize {
        if (have.len == 0) return 0;
        const marks = self.keptMarks(have);
        const top = marks.items[marks.n - 1];
        var n: usize = 0;
        for (self.kept[0..self.n]) |*k| {
            if (n == out.len) break;
            if (k.top > top or k.replay != replay or k.top > ids.len) continue;
            var all = true;
            for (k.marks.slice()) |m| if (!marks.has(m.at)) {
                all = false;
                break;
            };
            if (!all or !std.mem.eql(i32, k.keys(), ids[0..k.top])) continue;
            out[n] = k.eid;
            n += 1;
        }
        return n;
    }

    /// multi.py `_keep`: a finished stream's prompt (`ids` from position 0, its extent [base, base + size), its
    /// boundaries `marks`) kept: its rows to its last kept boundary stay, the rest of its extent goes back, the
    /// boundaries it does not keep are released. `owned`: the kept ids' storage (at least the kept top long), the
    /// book's from now on.
    pub fn keep(self: *Book, ex: *Extents, base: usize, size: usize, ids: []const i32, marks: *const Marks, replay: bool, owned: []i32) void {
        var have: Points = .{};
        for (marks.slice()) |m| have.put(m.at);
        const km = self.keptMarks(have.slice());
        const top = km.items[km.n - 1];
        const sz = Extents.size(top);
        if (size > sz) ex.give(base + sz, size - sz);
        var kept: Marks = .{};
        for (marks.slice()) |m| {
            if (km.has(m.at)) kept.add(m) else if (self.released) |r| r.snap(m.snap);
        }
        @memcpy(owned[0..top], ids[0..top]);
        self.kept[self.n] = .{ .eid = self.next_eid, .base = base, .size = sz, .top = top, .tick = self.ticks, .replay = replay, .marks = kept, .ids = owned };
        self.n += 1;
        self.next_eid +%= 1;
    }

    /// multi.py `_finish`'s last part: the covered kept prompts `drops` go, then the oldest past `entries`.
    pub fn finish(self: *Book, ex: *Extents, drops: []const u32) void {
        for (drops) |eid| self.forget(ex, eid);
        while (self.n > self.set.entries) _ = self.forgetOldest(ex);
    }

    /// The book's and the extents' state as one number (the records carry rank 0's; a rank whose differs stops).
    pub fn fingerprint(self: *const Book, ex: *const Extents) u64 {
        var h = std.hash.Wyhash.init(0x6b65_6570);
        const w = struct {
            fn int(hh: *std.hash.Wyhash, v: u64) void {
                hh.update(std.mem.asBytes(&v));
            }
        };
        w.int(&h, ex.total);
        w.int(&h, ex.n);
        for (ex.gaps[0..ex.n]) |g| {
            w.int(&h, g[0]);
            w.int(&h, g[1]);
        }
        w.int(&h, self.n);
        w.int(&h, self.next_eid);
        w.int(&h, self.ticks);
        for (self.kept[0..self.n]) |k| {
            w.int(&h, k.eid);
            w.int(&h, k.base);
            w.int(&h, k.size);
            w.int(&h, k.top);
            w.int(&h, k.tick);
            w.int(&h, @intFromBool(k.replay));
            for (k.marks.slice()) |m| {
                w.int(&h, m.at);
                w.int(&h, m.snap);
            }
        }
        return h.final();
    }
};

// -- tests -------------------------------------------------------------------------------------------------------

const testing = std.testing;

fn testBook(chunk: usize) Book {
    return .{ .set = .{}, .chunk = chunk, .window = 128 };
}

/// A kept entry of `top` ids (id i = seed + i) at [base, base + size(top)) with snapshots at `marks` (ids from 0).
fn putKept(b: *Book, ex: *Extents, ids: []i32, base: usize, top: usize, marks: []const usize, tick: u64) void {
    const sz = Extents.size(top);
    std.debug.assert(ex.takeAt(base, sz));
    var m: Marks = .{};
    for (marks, 0..) |at, i| m.add(.{ .at = at, .snap = @intCast(b.n * 16 + i) });
    b.kept[b.n] = .{ .eid = b.next_eid, .base = base, .size = sz, .top = top, .tick = tick, .replay = true, .marks = m, .ids = ids };
    b.n += 1;
    b.next_eid += 1;
}

test "a prompt snapshots at the doubling boundaries, its last two, and near its end when long" {
    var b = testBook(2048);
    try testing.expectEqualSlices(usize, &.{}, b.marksFor(2047).slice());
    try testing.expectEqualSlices(usize, &.{2048}, b.marksFor(2048).slice());
    try testing.expectEqualSlices(usize, &.{ 2048, 4096, 6144, 8192 }, b.marksFor(8589).slice());
    try testing.expectEqualSlices(usize, &.{ 2048, 4096, 8192, 16384, 18432, 20480 }, b.marksFor(21000).slice());
    // a long prompt also gets the 3/4 and 7/8 cut points (multiples of the chunk)
    const long = b.marksFor(260000);
    try testing.expect(long.has(260000 * 3 / 4 / 2048 * 2048) and long.has(260000 * 7 / 8 / 2048 * 2048));
    try testing.expect(long.has(131072) and long.has(258048) and long.has(256000));
    b.set.shrink = false;
    try testing.expect(!b.marksFor(260000).has(260000 * 3 / 4 / 2048 * 2048));
    // every boundary is a chunk multiple within the prompt, ascending
    var prng = std.Random.DefaultPrng.init(0x6d61);
    for (0..500) |_| {
        const n = prng.random().intRangeAtMost(usize, 1, 1 << 21);
        const p = b.marksFor(n).slice();
        for (p, 0..) |at, i| {
            try testing.expect(at % 2048 == 0 and at >= 2048 and at <= n);
            if (i > 0) try testing.expect(p[i - 1] < at);
        }
    }
}

test "a finished prompt keeps at most `marks` boundaries: the doubling ones, its last two, its upper half first" {
    var b = testBook(2048);
    b.set.marks = 4;
    var have: [12]usize = undefined;
    for (&have, 1..) |*h, i| h.* = 2048 * i;
    // doubling: 1, 2, 4, 8 chunks; last two: 11, 12; upper half (2b > 12 chunks): 7 .. 12
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 4, 7, 8, 9, 10, 11, 12 }, divided(b.keptMarks(&have).slice()));
    b.set.shrink = false;
    // then the earliest of the rest while there is room (none here: six must-keeps exceed four)
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 4, 8, 11, 12 }, divided(b.keptMarks(&have).slice()));
    b.set.marks = 8;
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 3, 4, 5, 8, 11, 12 }, divided(b.keptMarks(&have).slice()));
    // within the limit every boundary stays
    try testing.expectEqualSlices(usize, have[0..8], b.keptMarks(have[0..8]).slice());
}

var divided_buf: [max_marks]usize = undefined;
fn divided(p: []const usize) []const usize {
    for (p, 0..) |at, i| divided_buf[i] = at / 2048;
    return divided_buf[0..p.len];
}

test "a prompt continues the kept prompt with the longest shared kept boundary, below its end (a window with replay)" {
    var b = testBook(2048);
    var ex = Extents.init(64 * 2048);
    var ids_a: [10000]i32 = undefined;
    for (&ids_a, 0..) |*v, i| v.* = @intCast(i);
    var ids_b = ids_a;
    ids_b[5000] = -7; // b diverges from a at 5000
    putKept(&b, &ex, &ids_a, 0, 8192, &.{ 2048, 4096, 6144, 8192 }, 1);
    putKept(&b, &ex, &ids_b, 16384, 8192, &.{ 2048, 4096, 8192 }, 2);
    var p: [12000]u32 = undefined;
    for (&p, 0..) |*v, i| v.* = @intCast(i);
    // a shares 10000 ids: its 8192 (below 12000 - 128)
    try testing.expectEqual(Reuse{ .eid = 0, .cut = 8192 }, b.match(&p, true).?);
    // a prompt of 8250 ids: with replay the boundary stays at most 8250 - 128 = 8122 (6144); without, 8192
    try testing.expectEqual(Reuse{ .eid = 0, .cut = 6144 }, b.match(p[0..8250], true).?);
    b.kept[0].replay = false;
    b.kept[1].replay = false;
    try testing.expectEqual(Reuse{ .eid = 0, .cut = 8192 }, b.match(p[0..8250], false).?);
    try testing.expectEqual(@as(?Reuse, null), b.match(p[0..8250], true)); // replay differs: none
    b.kept[0].replay = true;
    b.kept[1].replay = true;
    // diverging at 3000: both share 2048; the more recently used (b) wins
    p[3000] = 99999;
    try testing.expectEqual(Reuse{ .eid = 1, .cut = 2048 }, b.match(&p, true).?);
    b.kept[0].tick = 2; // equal ticks: the older entry
    try testing.expectEqual(Reuse{ .eid = 0, .cut = 2048 }, b.match(&p, true).?);
    // no kept boundary below the shared start: none
    p[100] = 99999;
    try testing.expectEqual(@as(?Reuse, null), b.match(&p, true));
    try testing.expectEqual(@as(?Reuse, null), b.match(p[0..50], false));
}

test "placement: a free extent, then the continued prompt's own in place, then the oldest others shrink or go, last a move" {
    const C = 2048;
    const shared = try testing.allocator.alloc(i32, 40 * C);
    defer testing.allocator.free(shared);
    for (shared, 0..) |*v, i| v.* = @intCast(i);
    const ids = [2][]i32{ shared, shared };
    var rel: Released = .{};
    {
        // a free extent holds it: fresh, or copy when it continues a kept prompt
        var b = testBook(C);
        var ex = Extents.init(16 * C);
        b.released = &rel;
        putKept(&b, &ex, ids[0], 0, 4 * C, &.{ C, 2 * C, 4 * C }, 1);
        try testing.expectEqual(Placed{ .base = 4 * C, .how = .copy }, b.place(&ex, 3 * C, 0).?);
        try testing.expectEqual(Placed{ .base = 7 * C, .how = .fresh }, b.place(&ex, 3 * C, null).?);
        try testing.expectEqual(@as(usize, 1), b.n);
    }
    {
        // the window full past the kept prompt: it grows in place
        var b = testBook(C);
        var ex = Extents.init(16 * C);
        b.released = &rel;
        putKept(&b, &ex, ids[0], 0, 4 * C, &.{ C, 2 * C, 4 * C }, 1);
        try testing.expect(ex.takeAt(10 * C, 6 * C));
        try testing.expectEqual(Placed{ .base = 0, .how = .here }, b.place(&ex, 10 * C, 0).?);
        try testing.expectEqual(@as(usize, 0), b.n);
        try testing.expectEqual(@as(usize, 0), ex.n);
    }
    {
        // another kept prompt in the way: the oldest shrinks to the boundary whose tail makes the room
        rel = .{};
        var b = testBook(C);
        var ex = Extents.init(16 * C);
        b.released = &rel;
        putKept(&b, &ex, ids[0], 0, 8 * C, &.{ C, 2 * C, 4 * C, 6 * C, 8 * C }, 1);
        putKept(&b, &ex, ids[1], 8 * C, 6 * C, &.{ C, 2 * C, 4 * C, 6 * C }, 2);
        try testing.expectEqual(Placed{ .base = 14 * C, .how = .fresh }, b.place(&ex, 2 * C, null).?);
        try testing.expectEqual(@as(usize, 2), b.n);
        // the oldest (eid 0, top 8C) cut back to 4C: 6C's tail [6C, 8C) is too short for 3 blocks, 4C's is not
        const got = b.place(&ex, 3 * C, null).?;
        try testing.expectEqual(Placed{ .base = 4 * C, .how = .fresh }, got);
        try testing.expectEqual(@as(usize, 2), b.n);
        try testing.expectEqual(@as(usize, 4 * C), b.kept[0].top);
        try testing.expectEqual(@as(usize, 4 * C), b.kept[0].size);
        try testing.expectEqual(@as(usize, 3), b.kept[0].marks.n);
        try testing.expectEqual(@as(usize, 1), rel.shrunk);
        try testing.expectEqual(@as(usize, 2), rel.n_snaps); // its snapshots at 6C and 8C let go
        try testing.expectEqual(@as(usize, 0), rel.forgotten);
    }
    {
        // no boundary makes the room: the oldest is forgotten whole; then the continued prompt moves
        rel = .{};
        var b = testBook(C);
        var ex = Extents.init(8 * C);
        b.released = &rel;
        putKept(&b, &ex, ids[0], 0, 4 * C, &.{ 2 * C, 4 * C }, 1);
        putKept(&b, &ex, ids[1], 4 * C, 4 * C, &.{ 4 * C }, 2);
        const got = b.place(&ex, 8 * C, 1).?;
        try testing.expectEqual(Placed{ .base = 0, .how = .move }, got);
        try testing.expectEqual(@as(usize, 0), b.n);
        try testing.expectEqual(@as(usize, 2), rel.forgotten);
        // and nothing places past the window
        var b2 = testBook(C);
        var ex2 = Extents.init(8 * C);
        try testing.expectEqual(@as(?Placed, null), b2.place(&ex2, 9 * C, null));
    }
}

test "placement keeps the book and the extents consistent: kept and taken extents disjoint, a trial changes nothing" {
    const C = 2048;
    var prng = std.Random.DefaultPrng.init(0x6b70);
    const r = prng.random();
    const ids = try testing.allocator.alloc(i32, 24 * C);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var hows: [4]usize = @splat(0); // placements by How, over every run
    var shrunk: usize = 0;
    for (0..300) |_| {
        var rel: Released = .{};
        var b = testBook(C);
        b.set.entries = 6;
        b.released = &rel;
        b.set.shrink = r.boolean();
        var ex = Extents.init(24 * C);
        var streams: [max_slots][2]usize = undefined;
        var ns: usize = 0;
        for (0..40) |_| {
            const before = b.fingerprint(&ex);
            const need = r.intRangeAtMost(usize, C, 10 * C);
            const src: ?u32 = if (b.n > 0 and r.boolean()) b.kept[r.uintLessThan(usize, b.n)].eid else null;
            // a trial leaves the book as it was and says what the real placement does
            var t = b.trial();
            var tex = ex;
            const want = t.place(&tex, need, src);
            try testing.expectEqual(before, b.fingerprint(&ex));
            if (ns == max_slots) {
                // a stream finishes: kept (its prompt's boundaries) or released
                const i = r.uintLessThan(usize, ns);
                const s = streams[i];
                if (r.boolean() and b.n < b.set.entries + 1) {
                    var m: Marks = .{};
                    var at: usize = C;
                    while (at <= s[1] -| C) : (at += C) m.add(.{ .at = at, .snap = 0 });
                    if (m.n > 0) {
                        b.ticks += 1;
                        b.keep(&ex, s[0], s[1], ids, &m, true, try arena.allocator().alloc(i32, s[1]));
                        b.finish(&ex, &.{});
                    } else ex.give(s[0], s[1]);
                } else ex.give(s[0], s[1]);
                streams[i] = streams[ns - 1];
                ns -= 1;
                continue;
            }
            const got = b.place(&ex, need, src);
            try testing.expectEqual(want, got);
            if (got) |p| {
                hows[@intFromEnum(p.how)] += 1;
                streams[ns] = .{ p.base, Extents.size(need) };
                ns += 1;
                if (src == null) try testing.expect(p.how == .fresh) else try testing.expect(p.how != .fresh);
            } else {
                // nothing placed: every kept prompt gave its room back and the window still has none
                try testing.expect(b.n == 0 and ex.first(need) == null);
            }
            // every taken position is exactly one stream's or one kept prompt's
            var owner: [24]u8 = @splat(0);
            for (streams[0..ns]) |s| for (s[0] / C..(s[0] + s[1]) / C) |blk| {
                owner[blk] += 1;
            };
            for (b.slice()) |k| {
                try testing.expect(k.top <= k.size and k.size == Extents.size(k.top) and k.marks.last().? == k.top);
                for (k.base / C..(k.base + k.size) / C) |blk| owner[blk] += 1;
            }
            for (ex.gaps[0..ex.n]) |g| for (g[0] / C..g[1] / C) |blk| {
                owner[blk] += 1;
            };
            for (owner) |o| try testing.expectEqual(@as(u8, 1), o);
        }
        shrunk += rel.shrunk;
    }
    // the runs reach every placement and the shrink
    for (hows) |h| try testing.expect(h > 0);
    try testing.expect(shrunk > 0);
}

test "a finished prompt covers the kept prompts it repeats; entries past the limit go oldest first" {
    const C = 2048;
    var ids: [4][12 * C]i32 = undefined;
    for (&ids) |*a| for (a, 0..) |*v, i| {
        v.* = @intCast(i);
    };
    var rel: Released = .{};
    var b = testBook(C);
    b.released = &rel;
    var ex = Extents.init(64 * C);
    putKept(&b, &ex, &ids[0], 0, 4 * C, &.{ C, 2 * C, 4 * C }, 1);
    putKept(&b, &ex, &ids[1], 8 * C, 4 * C, &.{ C, 2 * C, 3 * C, 4 * C }, 2);
    ids[2][100] = -5;
    putKept(&b, &ex, &ids[2], 16 * C, 2 * C, &.{ C, 2 * C }, 3);
    // a prompt of 6C with boundaries C, 2C, 4C, 5C, 6C: covers eid 0 (its boundaries kept again), not eid 1 (3C is
    // not one of them), not eid 2 (its ids differ)
    var out: [8]u32 = undefined;
    const have = [_]usize{ C, 2 * C, 4 * C, 5 * C, 6 * C };
    try testing.expectEqualSlices(u32, &.{0}, out[0..b.covered(&ids[3], &have, true, &out)]);
    try testing.expectEqual(@as(usize, 0), b.covered(&ids[3], &have, false, &out));
    try testing.expectEqual(@as(usize, 0), b.covered(&ids[3], &.{}, true, &out));
    b.set.entries = 1;
    b.finish(&ex, &.{0});
    try testing.expectEqual(@as(usize, 1), b.n);
    try testing.expectEqual(@as(u32, 2), b.kept[0].eid); // eid 1 was the oldest left
    try testing.expectEqual(@as(usize, 2), rel.forgotten);
}

test "keeping a finished stream gives its extent's tail back and keeps its rows to its last kept boundary" {
    const C = 2048;
    var ids: [10 * C]i32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    var owned: [10 * C]i32 = undefined;
    var rel: Released = .{};
    var b = testBook(C);
    b.released = &rel;
    b.set.marks = 3;
    var ex = Extents.init(32 * C);
    try testing.expect(ex.takeAt(4 * C, 12 * C));
    var m: Marks = .{};
    for ([_]usize{ 1, 2, 3, 4, 5, 6 }, 0..) |q, i| m.add(.{ .at = q * C, .snap = @intCast(i) });
    b.keep(&ex, 4 * C, 12 * C, &ids, &m, true, &owned);
    const k = b.kept[0];
    // kept: 1, 2, 4 (doubling), 5, 6 (last two), 4..6 (upper half)
    try testing.expectEqual(@as(usize, 6 * C), k.top);
    try testing.expectEqual(@as(usize, 6 * C), k.size);
    try testing.expectEqual(@as(usize, 5), k.marks.n);
    try testing.expectEqual(@as(usize, 1), rel.n_snaps); // 3C let go
    try testing.expectEqual(@as(u16, 2), rel.snaps[0]);
    try testing.expectEqualSlices(i32, ids[0 .. 6 * C], k.keys());
    try testing.expectEqual(@as(?usize, 4 * C + 6 * C), ex.first(5 * C)); // [10C, 16C) free again, beside [16C, 32C)
}
