//! The Python family's rank cache (weights.py `RankCache`, TF_DS_RANK_CACHE): one file a rank holding this rank's
//! bytes of the checkpoint in the loader's order (the TP split's column and row slices, the dtypes the loader
//! converts to), with a JSON index footer: `[[key, dtype, shape, offset, bytes], ...]` then the index's offset as
//! little-endian u64 and the magic `TFDSRK01`. Keys are `{prefix}|{cols}|{rows}|tr|suh|svh` for EXL3 parts and
//! `{name}|{dtype}` for plain tensors (`None` where a slice is absent). The Zig loader reads this file when it
//! exists (a Python start wrote it), so a rank's load is sequential reads of ~100 GB instead of sliced shard reads.
const std = @import("std");
const linux = std.os.linux;

pub const magic = "TFDSRK01";

pub const DType = enum { bfloat16, float16, float32, int16, int32, int64, uint8, other };

pub const Entry = struct {
    key: []const u8,
    dtype: DType,
    shape: [4]usize,
    rank: usize, // dims used in `shape`
    offset: u64,
    bytes: u64,

    pub fn count(self: Entry) usize {
        var n: usize = 1;
        for (self.shape[0..self.rank]) |d| n *= d;
        return n;
    }
};

pub const RankCacheError = error{ OpenFailed, ReadFailed, BadFooter, BadIndex, NotFound, OutOfMemory, ShortRead, BufferTooSmall };

/// Offset, length and buffer alignment a direct (O_DIRECT) read needs.
pub const direct_alignment = 4096;

/// A byte range of the file, both ends at `direct_alignment` (the end may pass the file's end).
pub const Span = struct {
    lo: u64,
    hi: u64,

    pub fn len(self: Span) usize {
        return @intCast(self.hi - self.lo);
    }

    /// The smallest span holding both.
    pub fn merge(a: ?Span, b: ?Span) ?Span {
        const x = a orelse return b;
        const y = b orelse return a;
        return .{ .lo = @min(x.lo, y.lo), .hi = @max(x.hi, y.hi) };
    }
};

/// A span's bytes in memory: `bytes[i]` is the file's byte `span.lo + i`.
pub const Window = struct {
    span: Span,
    bytes: []const u8,

    /// `e`'s bytes when the window holds them all, else null.
    pub fn slice(self: Window, e: *const Entry) ?[]const u8 {
        if (e.offset < self.span.lo or e.offset + e.bytes > self.span.lo + self.bytes.len) return null;
        return self.bytes[@intCast(e.offset - self.span.lo)..][0..@intCast(e.bytes)];
    }
};

pub const RankCache = struct {
    pub const Error = RankCacheError;

    gpa: std.mem.Allocator,
    fd: i32,
    direct_fd: i32, // the same file opened O_DIRECT (-1 where the file system refuses it): bulk reads skip the page cache
    entries: []Entry,
    index: std.StringHashMapUnmanaged(u32), // key -> its first entry
    json: []u8, // the index text the entries' keys point into
    data_end: u64,

    /// Opens `path` and parses its footer.
    pub fn open(gpa: std.mem.Allocator, path: []const u8) Error!RankCache {
        const pz = gpa.allocSentinel(u8, path.len, 0) catch return error.OutOfMemory;
        defer gpa.free(pz);
        @memcpy(pz, path);
        const rc = linux.open(pz, .{ .ACCMODE = .RDONLY }, 0);
        if (linux.errno(rc) != .SUCCESS) return error.OpenFailed;
        const fd: i32 = @intCast(rc);
        errdefer _ = linux.close(fd);
        const end = linux.lseek(fd, 0, linux.SEEK.END);
        if (linux.errno(end) != .SUCCESS) return error.OpenFailed;
        const size: u64 = @intCast(end);
        if (size < 16) return error.BadFooter;
        var tail: [16]u8 = undefined;
        try preadAll(fd, &tail, size - 16);
        if (!std.mem.eql(u8, tail[8..], magic)) return error.BadFooter;
        const at = std.mem.readInt(u64, tail[0..8], .little);
        if (at + 16 > size) return error.BadFooter;
        const json = gpa.alloc(u8, @intCast(size - 16 - at)) catch return error.OutOfMemory;
        errdefer gpa.free(json);
        try preadAll(fd, json, at);
        const entries = try parseIndex(gpa, json);
        errdefer gpa.free(entries);
        var index: std.StringHashMapUnmanaged(u32) = .empty;
        errdefer index.deinit(gpa);
        index.ensureTotalCapacity(gpa, @intCast(entries.len)) catch return error.OutOfMemory;
        for (entries, 0..) |e, i| {
            const slot = index.getOrPutAssumeCapacity(e.key);
            if (!slot.found_existing) slot.value_ptr.* = @intCast(i);
        }
        const drc = linux.open(pz, .{ .ACCMODE = .RDONLY, .DIRECT = true }, 0);
        const direct_fd: i32 = if (linux.errno(drc) == .SUCCESS) @intCast(drc) else -1;
        return .{ .gpa = gpa, .fd = fd, .direct_fd = direct_fd, .entries = entries, .index = index, .json = json, .data_end = at };
    }

    pub fn close(self: *RankCache) void {
        _ = linux.close(self.fd);
        if (self.direct_fd >= 0) _ = linux.close(self.direct_fd);
        self.index.deinit(self.gpa);
        self.gpa.free(self.entries);
        self.gpa.free(self.json);
    }

    /// The entry named `key` (keys are unique in a cache the loader wrote).
    pub fn find(self: *const RankCache, key: []const u8) ?*const Entry {
        const i = self.index.get(key) orelse return null;
        return &self.entries[i];
    }

    /// The aligned span holding every entry whose key starts with `prefix` (null when none does).
    pub fn spanOf(self: *const RankCache, prefix: []const u8) ?Span {
        var lo: u64 = std.math.maxInt(u64);
        var hi: u64 = 0;
        for (self.entries) |e| {
            if (!std.mem.startsWith(u8, e.key, prefix)) continue;
            lo = @min(lo, e.offset);
            hi = @max(hi, e.offset + e.bytes);
        }
        if (hi == 0) return null;
        return .{ .lo = std.mem.alignBackward(u64, lo, direct_alignment), .hi = std.mem.alignForward(u64, hi, direct_alignment) };
    }

    /// Reads `span` into `buf` (at least `span.len()` bytes, `direct_alignment`-aligned) on `threads` threads, each
    /// taking the next `chunk`-byte piece in file order; direct reads when the file allows them. Bytes past the
    /// file's end are left as they were.
    pub fn readSpan(self: *const RankCache, span: Span, buf: []u8, threads: usize, chunk: usize) Error!void {
        var r: SpanRead = undefined;
        try self.startSpan(&r, span, buf, threads, chunk);
        return r.wait();
    }

    /// `readSpan` in the background: the reads are in flight on return; `r.wait()` joins them (`r` must stay put
    /// until then: the readers hold its address).
    pub fn startSpan(self: *const RankCache, r: *SpanRead, span: Span, buf: []u8, threads: usize, chunk: usize) Error!void {
        if (buf.len < span.len()) return error.BufferTooSmall;
        const direct = self.direct_fd >= 0 and @intFromPtr(buf.ptr) % direct_alignment == 0 and span.lo % direct_alignment == 0 and chunk % direct_alignment == 0;
        r.* = .{ .fd = if (direct) self.direct_fd else self.fd, .direct = direct, .span = span, .buf = buf, .chunk = @max(chunk, direct_alignment), .end = self.data_end };
        const n = @max(1, @min(threads, r.threads.len));
        for (r.threads[0..n]) |*t| {
            t.* = std.Thread.spawn(.{}, SpanRead.run, .{r}) catch {
                r.failed.store(true, .release);
                break;
            };
            r.spawned += 1;
        }
    }

    /// `e`'s bytes into `out` (`out.len == e.bytes`).
    pub fn read(self: *const RankCache, e: *const Entry, out: []u8) Error!void {
        if (out.len != e.bytes) return error.ShortRead;
        try preadAll(self.fd, out, e.offset);
    }

    /// The cache file for `rank` of `world` in directory `dir` (`rank{r}of{w}-<key>.bin`), the newest when several.
    pub fn locate(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, rank: usize, world: usize) Error!?[]u8 {
        var prefix_buf: [64]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buf, "rank{d}of{d}-", .{ rank, world }) catch unreachable;
        var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return null;
        defer d.close(io);
        var it = d.iterate();
        var best: ?[]u8 = null;
        while (it.next(io) catch return error.ReadFailed) |ent| {
            if (ent.kind != .file) continue;
            if (!std.mem.startsWith(u8, ent.name, prefix) or !std.mem.endsWith(u8, ent.name, ".bin")) continue;
            if (best) |b| gpa.free(b);
            best = std.fs.path.join(gpa, &.{ dir, ent.name }) catch return error.OutOfMemory;
        }
        return best;
    }
};

/// A span read in flight (RankCache.startSpan), shared with its reader threads until `wait`.
pub const SpanRead = struct {
    fd: i32,
    direct: bool,
    span: Span,
    buf: []u8,
    chunk: usize,
    end: u64, // the data's end: a piece past it is not read
    threads: [32]std.Thread = undefined,
    spawned: usize = 0,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn run(r: *SpanRead) void {
        while (!r.failed.load(.acquire)) {
            const k = r.next.fetchAdd(1, .monotonic);
            const at = r.span.lo + @as(u64, k) * r.chunk;
            if (at >= r.span.hi or at >= r.end) return;
            const n: usize = @intCast(@min(r.span.hi, at + r.chunk) - at);
            const out = r.buf[@intCast(at - r.span.lo)..][0..n];
            readUpTo(r.fd, r.direct, out, at) catch {
                r.failed.store(true, .release);
                return;
            };
        }
    }

    /// Joins the readers; the window's bytes are in `buf` once this returns without an error.
    pub fn wait(self: *SpanRead) RankCacheError!void {
        for (self.threads[0..self.spawned]) |t| t.join();
        self.spawned = 0;
        if (self.failed.load(.acquire)) return error.ReadFailed;
    }
};

/// `buf.len` bytes at `offset`, stopping early only at the file's end (a direct read ends short only there, and its
/// next offset would be unaligned).
fn readUpTo(fd: i32, direct: bool, buf: []u8, offset: u64) RankCacheError!void {
    var done: usize = 0;
    while (done < buf.len) {
        const rc = linux.pread(fd, buf[done..].ptr, buf.len - done, @intCast(offset + done));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.ReadFailed,
        }
        if (rc == 0) return;
        done += @intCast(rc);
        if (direct and done % direct_alignment != 0) return;
    }
}

fn preadAll(fd: i32, buf: []u8, offset: u64) RankCacheError!void {
    var done: usize = 0;
    while (done < buf.len) {
        const rc = linux.pread(fd, buf[done..].ptr, buf.len - done, @intCast(offset + done));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.ReadFailed,
        }
        if (rc == 0) return error.ShortRead;
        done += @intCast(rc);
    }
}

fn dtypeOf(s: []const u8) DType {
    const names = [_][]const u8{ "bfloat16", "float16", "float32", "int16", "int32", "int64", "uint8" };
    inline for (names, 0..) |n, i| if (std.mem.eql(u8, s, n)) return @enumFromInt(i);
    return .other;
}

/// `[[key, dtype, [shape...], offset, bytes], ...]`; keys stay slices of `json`.
fn parseIndex(gpa: std.mem.Allocator, json: []const u8) RankCacheError![]Entry {
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch return error.BadIndex;
    defer parsed.deinit();
    if (parsed.value != .array) return error.BadIndex;
    const items = parsed.value.array.items;
    const entries = gpa.alloc(Entry, items.len) catch return error.OutOfMemory;
    errdefer gpa.free(entries);
    // the parser keeps the entries in the text's order: each key is searched from the previous one's end, so the
    // index parses in time linear in its length (a search from the start each time took minutes on a 13 MB index)
    var cursor: usize = 0;
    for (items, entries) |item, *e| {
        if (item != .array or item.array.items.len != 5) return error.BadIndex;
        const f = item.array.items;
        if (f[0] != .string or f[1] != .string or f[2] != .array or f[3] != .integer or f[4] != .integer) return error.BadIndex;
        // the key must point into `json`, not the parser's copy: find it by value (keys are short and unique)
        const key = findKey(json, f[0].string, cursor) orelse findKey(json, f[0].string, 0) orelse return error.BadIndex;
        cursor = @intFromPtr(key.ptr) - @intFromPtr(json.ptr) + key.len;
        e.* = .{ .key = key, .dtype = dtypeOf(f[1].string), .shape = .{ 1, 1, 1, 1 }, .rank = f[2].array.items.len, .offset = @intCast(f[3].integer), .bytes = @intCast(f[4].integer) };
        if (e.rank > 4) return error.BadIndex;
        for (f[2].array.items, 0..) |d, i| e.shape[i] = if (d == .integer) @intCast(d.integer) else return error.BadIndex;
    }
    return entries;
}

/// The quoted string `s` as it appears in `json` at or after `start` (JSON keys here hold no escapes).
fn findKey(json: []const u8, s: []const u8, start: usize) ?[]const u8 {
    var from: usize = start;
    while (std.mem.indexOfPos(u8, json, from, s)) |i| {
        if (i > 0 and json[i - 1] == '"' and i + s.len < json.len and json[i + s.len] == '"') return json[i .. i + s.len];
        from = i + 1;
    }
    return null;
}

test "a cache file's footer, index and reads" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = ".tf-rank-cache-test";
    std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    const path = dir ++ "/rank1of2-abc.bin";
    // body: two tensors, then the index, its offset and the magic
    const a = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]u8{ 9, 10 };
    const index = "[[\"layers.0.attn.wq_a|None|None|tr\", \"int16\", [2, 2], 0, 8], [\"layers.0.attn_norm.weight|None\", \"bfloat16\", [1], 8, 2]]";
    var body: std.Io.Writer.Allocating = .init(gpa);
    defer body.deinit();
    try body.writer.writeAll(&a);
    try body.writer.writeAll(&b);
    try body.writer.writeAll(index);
    var off: [8]u8 = undefined;
    std.mem.writeInt(u64, &off, 10, .little);
    try body.writer.writeAll(&off);
    try body.writer.writeAll(magic);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = body.written() });

    var rc = try RankCache.open(gpa, path);
    defer rc.close();
    try std.testing.expectEqual(@as(usize, 2), rc.entries.len);
    const e = rc.find("layers.0.attn.wq_a|None|None|tr").?;
    try std.testing.expectEqual(DType.int16, e.dtype);
    try std.testing.expectEqual(@as(usize, 4), e.count());
    var buf: [8]u8 = undefined;
    try rc.read(e, &buf);
    try std.testing.expectEqualSlices(u8, &a, &buf);
    const n = rc.find("layers.0.attn_norm.weight|None").?;
    var buf2: [2]u8 = undefined;
    try rc.read(n, &buf2);
    try std.testing.expectEqualSlices(u8, &b, &buf2);
    try std.testing.expect(rc.find("missing") == null);
    const found = try RankCache.locate(gpa, io, dir, 1, 2);
    defer if (found) |f| gpa.free(f);
    try std.testing.expect(found != null);
    try std.testing.expect(std.mem.endsWith(u8, found.?, "rank1of2-abc.bin"));
    try std.testing.expect((try RankCache.locate(gpa, io, dir, 0, 2)) == null);
}

test "a span read on several threads lands every entry's bytes where its window finds them" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = ".tf-rank-cache-span-test";
    std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    const path = dir ++ "/rank0of2-abc.bin";
    // three entries over several 4 KiB pieces: two of block 0 (one unaligned), one of block 1
    const sizes = [_]usize{ 10_000, 7_003, 9_000 };
    const keys = [_][]const u8{ "layers.0.a|None", "layers.0.b|None", "layers.1.a|None" };
    var body: std.Io.Writer.Allocating = .init(gpa);
    defer body.deinit();
    var offs: [3]usize = undefined;
    var at: usize = 0;
    for (sizes, 0..) |n, i| {
        offs[i] = at;
        for (0..n) |k| try body.writer.writeByte(@intCast((at + k) % 251));
        at += n;
    }
    var index: std.Io.Writer.Allocating = .init(gpa);
    defer index.deinit();
    try index.writer.writeAll("[");
    for (keys, sizes, offs, 0..) |k, n, o, i| try index.writer.print("{s}[\"{s}\", \"uint8\", [{d}], {d}, {d}]", .{ if (i > 0) ", " else "", k, n, o, n });
    try index.writer.writeAll("]");
    try body.writer.writeAll(index.written());
    var off: [8]u8 = undefined;
    std.mem.writeInt(u64, &off, at, .little);
    try body.writer.writeAll(&off);
    try body.writer.writeAll(magic);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = body.written() });

    var rc = try RankCache.open(gpa, path);
    defer rc.close();
    const s0 = rc.spanOf("layers.0.").?;
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 20480 }, s0);
    const s1 = rc.spanOf("layers.1.").?;
    try std.testing.expectEqual(Span{ .lo = 16384, .hi = 28672 }, s1);
    try std.testing.expect(rc.spanOf("layers.2.") == null);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 28672 }, Span.merge(s0, s1).?);
    for ([_]Span{ s0, s1 }) |sp| for ([_]usize{ 1, 3, 8 }) |threads| {
        const buf = try gpa.alignedAlloc(u8, .fromByteUnits(direct_alignment), sp.len());
        defer gpa.free(buf);
        try rc.readSpan(sp, buf, threads, direct_alignment);
        const w: Window = .{ .span = sp, .bytes = buf };
        for (keys) |k| {
            const e = rc.find(k).?;
            const got = w.slice(e) orelse {
                try std.testing.expect(e.offset < sp.lo or e.offset + e.bytes > sp.hi);
                continue;
            };
            const want = try gpa.alloc(u8, @intCast(e.bytes));
            defer gpa.free(want);
            try rc.read(e, want);
            try std.testing.expectEqualSlices(u8, want, got);
        }
    };
    var small: [4096]u8 align(direct_alignment) = undefined;
    try std.testing.expectError(error.BufferTooSmall, rc.readSpan(s0, &small, 2, direct_alignment));
}
