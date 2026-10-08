//! Engram rows for DeepSeek-V4.1 (the Python family's `model.py: Engram`): the FP8 tables stay in DeepSeek's original
//! shards (`layers.L.engram.embed.weight` F8_E4M3 [rows, 256] and `.scale` F8_E8M0 [rows, 8]), each row read by
//! offset from local NVMe when a position's hashes name it. A reader pool gathers the code and scale rows of a batch
//! into pinned memory, the staging copies them to the device without a sync, and `decode` turns them into the bf16
//! rows `engram_wkv` reads: row-major over the batch, so a rank's columns [lo, hi) of n positions give
//! `e [n, (hi - lo) * 256]`, the input of its `engram.wkv` rows [(hi - lo) * 256 * rank, ...).
const std = @import("std");
const cuda = @import("cuda");
const linux = std.os.linux;

pub const row_w = 256; // code bytes a row (engram_head_dim)
pub const row_s = 8; // scale bytes a row (one e8m0 a group of 32)
pub const max_layers = 4;

pub const Error = error{ OpenFailed, ReadFailed, BadHeader, NoTable, OutOfMemory, Busy, BadShape };

/// One layer's table: its shard's descriptor and the byte offsets of the two tensors.
pub const Table = struct {
    fd: i32,
    base_w: u64,
    base_s: u64,
    rows: u64,
};

pub const Tables = struct {
    gpa: std.mem.Allocator,
    layers: [max_layers]i64 = @splat(-1),
    tables: [max_layers]Table = undefined,
    len: usize = 0,
    fds: [max_layers]i32 = @splat(-1),
    n_fds: usize = 0,

    /// Opens every `*.safetensors` under `dir` and takes the Engram tables it holds (any layer).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) Error!Tables {
        var self: Tables = .{ .gpa = gpa };
        errdefer self.close();
        var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return error.OpenFailed;
        defer d.close(io);
        var it = d.iterate();
        while (it.next(io) catch return error.OpenFailed) |ent| {
            if (ent.kind != .file or !std.mem.endsWith(u8, ent.name, ".safetensors")) continue;
            const path = std.fs.path.join(gpa, &.{ dir, ent.name }) catch return error.OutOfMemory;
            defer gpa.free(path);
            try self.openShard(path);
        }
        if (self.len == 0) return error.NoTable;
        return self;
    }

    pub fn close(self: *Tables) void {
        for (self.fds[0..self.n_fds]) |fd| _ = linux.close(fd);
        self.n_fds = 0;
        self.len = 0;
    }

    pub fn get(self: *const Tables, layer: i64) ?*const Table {
        for (self.layers[0..self.len], 0..) |l, i| if (l == layer) return &self.tables[i];
        return null;
    }

    fn openShard(self: *Tables, path: []const u8) Error!void {
        const pz = self.gpa.allocSentinel(u8, path.len, 0) catch return error.OutOfMemory;
        defer self.gpa.free(pz);
        @memcpy(pz, path);
        const rc = linux.open(pz, .{ .ACCMODE = .RDONLY }, 0);
        if (linux.errno(rc) != .SUCCESS) return error.OpenFailed;
        const fd: i32 = @intCast(rc);
        var keep = false;
        defer if (!keep) {
            _ = linux.close(fd);
        };
        var head: [8]u8 = undefined;
        try preadAll(fd, &head, 0);
        const size = std.mem.readInt(u64, &head, .little);
        if (size > (16 << 20)) return error.BadHeader;
        const text = self.gpa.alloc(u8, @intCast(size)) catch return error.OutOfMemory;
        defer self.gpa.free(text);
        try preadAll(fd, text, 8);
        var parsed = std.json.parseFromSlice(std.json.Value, self.gpa, text, .{}) catch return error.BadHeader;
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadHeader;
        var jt = parsed.value.object.iterator();
        while (jt.next()) |e| {
            const name = e.key_ptr.*;
            const is_w = std.mem.endsWith(u8, name, "engram.embed.weight");
            const is_s = std.mem.endsWith(u8, name, "engram.embed.scale");
            if (!is_w and !is_s) continue;
            if (!std.mem.startsWith(u8, name, "layers.")) return error.BadHeader;
            const dot = std.mem.indexOfScalarPos(u8, name, 7, '.') orelse return error.BadHeader;
            const layer = std.fmt.parseInt(i64, name[7..dot], 10) catch return error.BadHeader;
            const v = e.value_ptr.*;
            if (v != .object) return error.BadHeader;
            const offs = (v.object.get("data_offsets") orelse return error.BadHeader).array.items;
            const shape = (v.object.get("shape") orelse return error.BadHeader).array.items;
            if (offs.len != 2 or shape.len != 2) return error.BadHeader;
            const rows: u64 = @intCast(shape[0].integer);
            const width: u64 = @intCast(shape[1].integer);
            if (width != (if (is_w) @as(u64, row_w) else @as(u64, row_s))) return error.BadShape;
            const base: u64 = 8 + size + @as(u64, @intCast(offs[0].integer));
            const idx = self.slot(layer) orelse return error.OutOfMemory;
            const t = &self.tables[idx];
            if (is_w) t.base_w = base else t.base_s = base;
            t.fd = fd;
            t.rows = rows;
            keep = true;
        }
        if (keep) {
            if (self.n_fds == max_layers) return error.OutOfMemory;
            self.fds[self.n_fds] = fd;
            self.n_fds += 1;
        }
    }

    fn slot(self: *Tables, layer: i64) ?usize {
        for (self.layers[0..self.len], 0..) |l, i| if (l == layer) return i;
        if (self.len == max_layers) return null;
        self.layers[self.len] = layer;
        self.tables[self.len] = .{ .fd = -1, .base_w = 0, .base_s = 0, .rows = 0 };
        self.len += 1;
        return self.len - 1;
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
        if (rc == 0) return error.ReadFailed;
        done += @intCast(rc);
    }
}

/// A batch of row reads: `idx[i]` names a row; its codes land at out_w[i * 256 ..], its scales at out_s[i * 8 ..].
const Job = struct {
    table: Table = .{ .fd = -1, .base_w = 0, .base_s = 0, .rows = 0 },
    idx: [*]const i64 = undefined,
    n: usize = 0,
    out_w: [*]u8 = undefined,
    out_s: [*]u8 = undefined,
};

/// Persistent reader threads (engram_io.cpp's pool): a batch is 2n preads (codes and scales a row) spread over
/// `threads` workers and the caller; a worker idles on a short sleep between batches.
pub const Reader = struct {
    gpa: std.mem.Allocator,
    threads: []std.Thread,
    job: Job = .{},
    gen: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    next: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    done: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    failed: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn init(gpa: std.mem.Allocator, threads: usize) Error!*Reader {
        const self = gpa.create(Reader) catch return error.OutOfMemory;
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .threads = &.{} };
        const list = gpa.alloc(std.Thread, threads) catch return error.OutOfMemory;
        errdefer gpa.free(list);
        var spawned: usize = 0;
        errdefer {
            self.stop.store(true, .release);
            _ = self.gen.fetchAdd(1, .seq_cst);
            for (list[0..spawned]) |t| t.join();
        }
        for (list) |*t| {
            t.* = std.Thread.spawn(.{}, worker, .{self}) catch return error.OutOfMemory;
            spawned += 1;
        }
        self.threads = list;
        return self;
    }

    pub fn deinit(self: *Reader) void {
        self.stop.store(true, .release);
        _ = self.gen.fetchAdd(1, .seq_cst);
        for (self.threads) |t| t.join();
        self.gpa.free(self.threads);
        self.gpa.destroy(self);
    }

    /// Reads the rows `idx` of `table` into `out_w` (n * 256 bytes) and `out_s` (n * 8 bytes); returns when every
    /// byte landed.
    pub fn gather(self: *Reader, table: *const Table, idx: []const i64, out_w: []u8, out_s: []u8) Error!void {
        if (out_w.len < idx.len * row_w or out_s.len < idx.len * row_s) return error.BadShape;
        if (idx.len == 0) return;
        self.job = .{ .table = table.*, .idx = idx.ptr, .n = idx.len, .out_w = out_w.ptr, .out_s = out_s.ptr };
        self.failed.store(0, .seq_cst);
        self.done.store(0, .seq_cst);
        self.next.store(0, .seq_cst);
        _ = self.gen.fetchAdd(1, .seq_cst);
        self.work(); // the caller takes its share
        const total: u32 = @intCast(2 * idx.len);
        var spins: u32 = 0;
        while (self.done.load(.acquire) < total) {
            spins += 1;
            if (spins < 2000) std.atomic.spinLoopHint() else sleepUs(20);
        }
        if (self.failed.load(.acquire) != 0) return error.ReadFailed;
    }

    fn work(self: *Reader) void {
        while (true) {
            const j = self.next.fetchAdd(1, .seq_cst);
            const jb = self.job;
            if (j >= 2 * jb.n) return;
            const i = j >> 1;
            const row: u64 = @intCast(jb.idx[i]);
            const ok = if (j & 1 == 0)
                preadAll(jb.table.fd, jb.out_w[i * row_w .. (i + 1) * row_w], jb.table.base_w + row * row_w)
            else
                preadAll(jb.table.fd, jb.out_s[i * row_s .. (i + 1) * row_s], jb.table.base_s + row * row_s);
            ok catch self.failed.store(1, .seq_cst);
            _ = self.done.fetchAdd(1, .seq_cst);
        }
    }

    fn worker(self: *Reader) void {
        var seen: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (self.gen.load(.acquire) == seen) {
                spins += 1;
                if (spins < 500) std.atomic.spinLoopHint() else sleepUs(50);
            }
            seen = self.gen.load(.acquire);
            if (self.stop.load(.acquire)) return;
            self.work();
        }
    }
};

fn sleepUs(us: u32) void {
    var req: linux.timespec = .{ .sec = 0, .nsec = @intCast(us * 1000) };
    var rem: linux.timespec = undefined;
    _ = linux.nanosleep(&req, &rem);
}

/// Pinned host rows of a batch and their device copies: `rows` at most, grown by the caller when a batch is larger.
pub const Staging = struct {
    host_w: cuda.HostBuffer,
    host_s: cuda.HostBuffer,
    dev_w: cuda.DeviceBuffer,
    dev_s: cuda.DeviceBuffer,
    rows: usize,

    pub fn init(d: *const cuda.Driver, rows: usize) !Staging {
        var hw = try cuda.HostBuffer.alloc(d, rows * row_w);
        errdefer hw.free();
        var hs = try cuda.HostBuffer.alloc(d, rows * row_s);
        errdefer hs.free();
        var dw = try cuda.DeviceBuffer.alloc(d, rows * row_w);
        errdefer dw.free();
        var ds = try cuda.DeviceBuffer.alloc(d, rows * row_s);
        errdefer ds.free();
        return .{ .host_w = hw, .host_s = hs, .dev_w = dw, .dev_s = ds, .rows = rows };
    }

    pub fn deinit(self: *Staging) void {
        self.host_w.free();
        self.host_s.free();
        self.dev_w.free();
        self.dev_s.free();
    }

    pub fn codes(self: *const Staging, n: usize) []u8 {
        return self.host_w.slice(u8)[0 .. n * row_w];
    }

    pub fn scales(self: *const Staging, n: usize) []u8 {
        return self.host_s.slice(u8)[0 .. n * row_s];
    }

    /// The first `n` rows' host bytes to the device on `stream` (no sync; the host rows must stay until it ran).
    pub fn upload(self: *const Staging, n: usize, stream: cuda.Stream) !void {
        try self.dev_w.uploadAsync(0, self.codes(n), stream.handle);
        try self.dev_s.uploadAsync(0, self.scales(n), stream.handle);
    }
};

/// The device ops: rows to bf16, and the ranks' gathered partials to bf16.
pub const Kernels = struct {
    module: cuda.Module,
    decode_fn: cuda.Function,
    sum_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_engram);
        errdefer module.unload();
        return .{ .module = module, .decode_fn = try module.function("tf_dsv41_engram_decode"), .sum_fn = try module.function("tf_dsv41_engram_sum") };
    }

    pub fn deinit(self: *Kernels) void {
        self.module.unload();
    }

    /// codes [rows, 256] u8, scales [rows, 8] u8 -> out [rows, 256] bf16.
    pub fn decode(self: *const Kernels, stream: cuda.Stream, codes: cuda.abi.DevicePtr, scales: cuda.abi.DevicePtr, out: cuda.abi.DevicePtr, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(codes);
        a.add(scales);
        a.add(out);
        a.add(@as(c_int, @intCast(rows)));
        try cuda.launch.launch(self.decode_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = row_w } }, stream, &a);
    }

    /// g [world, n] f32 (every rank's partial, rank order) -> out [n] bf16, the sum in rank order.
    pub fn sumToBf16(self: *const Kernels, stream: cuda.Stream, g: cuda.abi.DevicePtr, world: u32, n: usize, out: cuda.abi.DevicePtr) !void {
        var a: cuda.Args = .{};
        a.add(g);
        a.add(@as(c_int, @intCast(world)));
        a.add(@as(i64, @intCast(n)));
        a.add(out);
        const block: u32 = 256;
        try cuda.launch.launch(self.sum_fn, .{ .grid = .{ .x = @intCast((n + block - 1) / block) }, .block = .{ .x = block } }, stream, &a);
    }
};

test "the shard header's offsets" {
    // 8-byte little-endian size, then the JSON: base = 8 + size + data_offsets[0]
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = ".tf-engram-test";
    std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    const json = "{\"layers.1.engram.embed.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[3,256],\"data_offsets\":[0,768]},\"layers.1.engram.embed.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[3,8],\"data_offsets\":[768,792]}}";
    var body: std.Io.Writer.Allocating = .init(gpa);
    defer body.deinit();
    var sz: [8]u8 = undefined;
    std.mem.writeInt(u64, &sz, json.len, .little);
    try body.writer.writeAll(&sz);
    try body.writer.writeAll(json);
    var data: [792]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i * 7 + 1);
    try body.writer.writeAll(&data);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/model-00047-of-00048.safetensors", .data = body.written() });
    var t = try Tables.open(gpa, io, dir);
    defer t.close();
    const tab = t.get(1).?;
    try std.testing.expectEqual(@as(u64, 8 + json.len), tab.base_w);
    try std.testing.expectEqual(@as(u64, 8 + json.len + 768), tab.base_s);
    try std.testing.expectEqual(@as(u64, 3), tab.rows);
    try std.testing.expect(t.get(14) == null);
    var r = try Reader.init(gpa, 2);
    defer r.deinit();
    const idx = [_]i64{ 2, 0 };
    var ow: [2 * row_w]u8 = undefined;
    var os: [2 * row_s]u8 = undefined;
    try r.gather(tab, &idx, &ow, &os);
    try std.testing.expectEqualSlices(u8, data[2 * row_w .. 3 * row_w], ow[0..row_w]);
    try std.testing.expectEqualSlices(u8, data[0..row_w], ow[row_w .. 2 * row_w]);
    try std.testing.expectEqualSlices(u8, data[768 + 2 * row_s .. 768 + 3 * row_s], os[0..row_s]);
    try std.testing.expectEqualSlices(u8, data[768 .. 768 + row_s], os[row_s .. 2 * row_s]);
}
