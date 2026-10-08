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

pub const Error = error{ OpenFailed, ReadFailed, BadFooter, BadIndex, NotFound, OutOfMemory, ShortRead };

pub const RankCache = struct {
    gpa: std.mem.Allocator,
    fd: i32,
    entries: []Entry,
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
        return .{ .gpa = gpa, .fd = fd, .entries = entries, .json = json, .data_end = at };
    }

    pub fn close(self: *RankCache) void {
        _ = linux.close(self.fd);
        self.gpa.free(self.entries);
        self.gpa.free(self.json);
    }

    /// The entry named `key` (keys are unique in a cache the loader wrote).
    pub fn find(self: *const RankCache, key: []const u8) ?*const Entry {
        for (self.entries) |*e| if (std.mem.eql(u8, e.key, key)) return e;
        return null;
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

fn preadAll(fd: i32, buf: []u8, offset: u64) Error!void {
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
fn parseIndex(gpa: std.mem.Allocator, json: []const u8) Error![]Entry {
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch return error.BadIndex;
    defer parsed.deinit();
    if (parsed.value != .array) return error.BadIndex;
    const items = parsed.value.array.items;
    const entries = gpa.alloc(Entry, items.len) catch return error.OutOfMemory;
    errdefer gpa.free(entries);
    for (items, entries) |item, *e| {
        if (item != .array or item.array.items.len != 5) return error.BadIndex;
        const f = item.array.items;
        if (f[0] != .string or f[1] != .string or f[2] != .array or f[3] != .integer or f[4] != .integer) return error.BadIndex;
        // the key must point into `json`, not the parser's copy: find it by value (keys are short and unique)
        const key = findKey(json, f[0].string) orelse return error.BadIndex;
        e.* = .{ .key = key, .dtype = dtypeOf(f[1].string), .shape = .{ 1, 1, 1, 1 }, .rank = f[2].array.items.len, .offset = @intCast(f[3].integer), .bytes = @intCast(f[4].integer) };
        if (e.rank > 4) return error.BadIndex;
        for (f[2].array.items, 0..) |d, i| e.shape[i] = if (d == .integer) @intCast(d.integer) else return error.BadIndex;
    }
    return entries;
}

/// The quoted string `s` as it appears in `json` (JSON keys here hold no escapes).
fn findKey(json: []const u8, s: []const u8) ?[]const u8 {
    var from: usize = 0;
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
