//! Kept prompts (TF_GLM_KEEP): a finished stream's prompt stays in the pool, so a later prompt that starts the same
//! prefills only what follows. The book of extents and kept entries is the DeepSeek-V4.1 one (dsv41 cuda_keep.zig
//! `Book`: matching, placement, eviction, shrink); what a boundary holds is GLM's:
//!
//! - **A boundary b** is a multiple of the 2,048-row prompt chunk. There the slot's KDA states and conv windows, and
//!   every DSA layer's (and the head's) index rings, are snapshotted; the latent and pooled-key rows before b are the
//!   extent's. The MTP head has absorbed the b rows before it, the last of them with the prompt's token b as its next
//!   token, so a prompt continues from b only when it shares the first b + 1 tokens: the book's boundaries are b + 1
//!   (`at` = b + 1), its matching then asks for those tokens. A continued prompt's chunks from b are a fresh prefill's
//!   own chunks (the same rows from the same states), so its reply is the fresh one's.
//! - **Snapshots** live in `snaps` slots of `snapBytes` each, counted by the streams and kept entries that hold them.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const engine = @import("cuda_engine.zig");
const P = @import("cuda_pool.zig");
const kda = @import("cuda_kda.zig");
pub const book = @import("dsv41").kept_prompts;

const Buf = cuda.DeviceBuffer;

/// The prompt chunk's rows (engine.prompt_rows): every boundary is a multiple of it.
pub const chunk = engine.prompt_rows;

/// TF_GLM_KEEP (default on), TF_GLM_KEEP_ENTRIES, TF_GLM_KEEP_MARKS, TF_GLM_KEEP_SHRINK, TF_GLM_KEEP_SHRINK_MIN and
/// TF_GLM_KEEP_SNAPS (default 16 snapshots, ~90 MB each on a rank), as the DeepSeek-V4.1 book reads its own. Both
/// ranks must read the same values.
pub fn settings() book.Settings {
    var s: book.Settings = .{ .snaps = 16 };
    s.on = flag("TF_GLM_KEEP", true);
    s.entries = @min(int("TF_GLM_KEEP_ENTRIES", s.entries), book.max_kept - 1);
    s.marks = @max(@min(int("TF_GLM_KEEP_MARKS", s.marks), book.max_marks), 2);
    s.shrink = flag("TF_GLM_KEEP_SHRINK", true);
    s.shrink_min = int("TF_GLM_KEEP_SHRINK_MIN", s.shrink_min);
    s.snaps = @min(int("TF_GLM_KEEP_SNAPS", s.snaps), 255);
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

/// The boundaries a prompt of `n` tokens snapshots at, as the book's points (b + 1 for boundary b): the book's
/// boundaries of the first n - 1 tokens that leave at least 2 rows to prefill after them.
pub fn pointsFor(b: *const book.Book, n: usize) book.Points {
    var out: book.Points = .{};
    if (n < 2) return out;
    const p = b.marksFor(n - 1);
    for (p.slice()) |at| if (at + 2 <= n and out.n < out.items.len) {
        out.items[out.n] = at + 1;
        out.n += 1;
    };
    return out;
}

/// The device bytes one snapshot takes for a slot's sequence.
pub fn snapBytes(q: *const engine.Seq) usize {
    var n = q.st.kda.layers * q.st.kda.stateBytes() + q.st.kda.conv.len;
    for (q.st.caches) |c| n += c.ik.len + c.ig.len;
    return n + q.mcache.ik.len + q.mcache.ig.len;
}

/// The snapshots' storage and their holds.
pub const Snaps = struct {
    gpa: std.mem.Allocator,
    buf: Buf,
    bytes: usize,
    refs: []u8,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, count: usize, bytes: usize) !Snaps {
        const refs = try gpa.alloc(u8, count);
        errdefer gpa.free(refs);
        @memset(refs, 0);
        return .{ .gpa = gpa, .buf = try Buf.alloc(d, count * bytes), .bytes = bytes, .refs = refs };
    }

    pub fn deinit(s: *Snaps) void {
        s.buf.free();
        s.gpa.free(s.refs);
    }

    /// A free snapshot's index, held once; null when every one is held.
    pub fn open(s: *Snaps) ?u16 {
        const i = std.mem.indexOfScalar(u8, s.refs, 0) orelse return null;
        s.refs[i] = 1;
        return @intCast(i);
    }

    /// The sequence's boundary state into snapshot `id`.
    pub fn save(s: *const Snaps, id: u16, q: *const engine.Seq, stream: cuda.Stream) !void {
        var off: usize = @as(usize, id) * s.bytes;
        const k = &q.st.kda;
        const rec = k.layers * k.stateBytes();
        try s.buf.copyFrom(off, k.state(k.cur, 0), rec, stream.handle);
        off += rec;
        try s.buf.copyFrom(off, k.conv.ptr, k.conv.len, stream.handle);
        off += k.conv.len;
        for (q.st.caches) |c| off = try ringsTo(s.buf, off, c, stream);
        _ = try ringsTo(s.buf, off, q.mcache, stream);
    }

    fn ringsTo(buf: Buf, off: usize, c: anytype, stream: cuda.Stream) !usize {
        try buf.copyFrom(off, c.ik.ptr, c.ik.len, stream.handle);
        try buf.copyFrom(off + c.ik.len, c.ig.ptr, c.ig.len, stream.handle);
        return off + c.ik.len + c.ig.len;
    }

    /// Snapshot `id` into the sequence, which then stands at boundary `at` (its head too, owing nothing).
    pub fn restore(s: *const Snaps, id: u16, q: *engine.Seq, at: usize, stream: cuda.Stream) !void {
        var off: usize = @as(usize, id) * s.bytes;
        const k = &q.st.kda;
        k.cur = 0;
        const rec = k.layers * k.stateBytes();
        try sub(k.rec, 0, rec).copyFrom(0, s.buf.ptr + off, rec, stream.handle);
        off += rec;
        try k.conv.copyFrom(0, s.buf.ptr + off, k.conv.len, stream.handle);
        off += k.conv.len;
        for (q.st.caches) |c| off = try ringsFrom(s.buf, off, c, stream);
        _ = try ringsFrom(s.buf, off, q.mcache, stream);
        k.pos = at;
        q.st.pos = at;
        q.m_next.clearRetainingCapacity();
        q.mtp_len = at;
        q.mtp_drafted = 0;
        q.fresh = false;
    }

    fn ringsFrom(buf: Buf, off: usize, c: anytype, stream: cuda.Stream) !usize {
        try c.ik.copyFrom(0, buf.ptr + off, c.ik.len, stream.handle);
        try c.ig.copyFrom(0, buf.ptr + off + c.ik.len, c.ig.len, stream.handle);
        return off + c.ik.len + c.ig.len;
    }
};

fn sub(b: Buf, off: usize, n: usize) Buf {
    return .{ .d = b.d, .ptr = b.ptr + off, .len = n };
}

/// Positions [src, src + n) of every pool plane to [dst, dst + n) (n and both starts multiples of the extent
/// alignment): the latent rows and the pooled keys. Overlapping ranges copy in pieces that never read a row already
/// written.
pub fn copyRows(pool: *const P.Pool, src: usize, dst: usize, n: usize, stream: cuda.Stream) !void {
    if (n == 0 or src == dst) return;
    const step = @min(n, if (src > dst) src - dst else dst - src);
    var done: usize = 0;
    while (done < n) {
        const m = @min(step, n - done);
        // forward when moving down, backward when moving up
        const at = if (dst < src) done else n - done - m;
        for (pool.latent, pool.pk) |lat, pk| {
            const lr = lat.len / pool.rows;
            const pr = pk.len / (pool.rows / 4 + 2);
            try lat.copyFrom((dst + at) * lr, lat.ptr + (src + at) * lr, m * lr, stream.handle);
            try pk.copyFrom((dst + at) / 4 * pr, pk.ptr + (src + at) / 4 * pr, m / 4 * pr, stream.handle);
        }
        done += m;
    }
}

test "a prompt's boundaries are the book's, one past each, with 2 rows after" {
    var b: book.Book = .{ .set = .{}, .chunk = chunk, .window = 0 };
    // 5,000 tokens: boundaries 2,048 and 4,096
    const p = pointsFor(&b, 5000);
    try std.testing.expectEqualSlices(usize, &.{ 2049, 4097 }, p.slice());
    // 4,097 tokens: 4,096 leaves 1 row after it (the last needs a row to prefill and the token after the boundary)
    try std.testing.expectEqualSlices(usize, &.{2049}, pointsFor(&b, 4097).slice());
    try std.testing.expectEqual(@as(usize, 0), pointsFor(&b, 2049).n);
    // a prompt that shares only the first 2,048 tokens of a kept one does not continue it (the head read token 2,048)
    var ids: [2049]i32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i);
    var ex = book.Extents.init(1 << 16);
    var m: book.Marks = .{};
    m.add(.{ .at = 2049, .snap = 0 });
    const base = ex.take(4096).?;
    b.kept[0] = .{ .eid = 0, .base = base, .size = 4096, .top = 2049, .tick = 0, .replay = false, .marks = m, .ids = &ids };
    b.n = 1;
    var other: [3000]u32 = undefined;
    for (&other, 0..) |*x, i| x.* = @intCast(i);
    try std.testing.expectEqual(@as(usize, 2049), b.match(&other, false).?.cut);
    other[2048] = 99999;
    try std.testing.expect(b.match(&other, false) == null);
}
