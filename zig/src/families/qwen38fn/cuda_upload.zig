//! The load's bulk copies: host bytes (the pack's mapped tensors) to device buffers through two pinned staging
//! halves on one stream, so the copy of one half runs while the next is filled. A pageable source makes each
//! synchronous upload stage through the driver in small pieces; pinned halves let the copies run at memory speed.
//! The bytes a buffer holds after `finish` are those `put` was given, in any order of puts.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const core = @import("core");
const dio = core.direct_io;

pub const Uploader = struct {
    span: ?SpanReader = null, // whole file ranges read directly (withSpans)
    stream: cuda.Stream,
    half: [2]cuda.HostBuffer,
    ev: [2]cuda.Event,
    pending: [2]bool = .{ false, false },
    cur: u1 = 0,
    fill: usize = 0,

    /// Two staging halves of `half_bytes` each.
    pub fn init(d: *const cuda.Driver, half_bytes: usize) !Uploader {
        var stream = try cuda.Stream.init(d, true);
        errdefer stream.deinit();
        var a = try cuda.HostBuffer.alloc(d, half_bytes);
        errdefer a.free();
        var b = try cuda.HostBuffer.alloc(d, half_bytes);
        errdefer b.free();
        var e0 = try cuda.Event.init(d, false);
        errdefer e0.deinit();
        return .{ .stream = stream, .half = .{ a, b }, .ev = .{ e0, try cuda.Event.init(d, false) } };
    }

    /// Read file ranges of up to `bytes` directly (SpanReader) where a caller knows its tensors' files.
    pub fn withSpans(u: *Uploader, d: *const cuda.Driver, bytes: usize) !void {
        u.span = try SpanReader.init(d, bytes);
    }

    pub fn deinit(u: *Uploader) void {
        u.stream.synchronize() catch {};
        if (u.span) |*r| r.deinit();
        for (&u.half) |*h| h.free();
        for (&u.ev) |*e| e.deinit();
        u.stream.deinit();
    }

    /// `src` into `dst` at `offset`, queued; `src` may be reused once this returns.
    pub fn put(u: *Uploader, dst: cuda.DeviceBuffer, offset: usize, src: []const u8) !void {
        const size = u.half[0].bytes.len;
        var done: usize = 0;
        while (done < src.len) {
            if (u.fill == size) try u.flip();
            const n = @min(src.len - done, size - u.fill);
            const stage = u.half[u.cur].bytes[u.fill..][0..n];
            @memcpy(stage, src[done..][0..n]);
            try dst.uploadAsync(offset + done, stage, u.stream.handle);
            u.fill += n;
            done += n;
        }
    }

    /// `src` (pinned memory that outlives the copy) into `dst` at `offset`, queued without staging.
    pub fn putPinned(u: *Uploader, dst: cuda.DeviceBuffer, offset: usize, src: []const u8) !void {
        try dst.uploadAsync(offset, src, u.stream.handle);
    }

    /// A device buffer holding `src` (freed by the caller), its copy queued.
    pub fn fromHost(u: *Uploader, d: *const cuda.Driver, src: []const u8) !cuda.DeviceBuffer {
        var b = try cuda.DeviceBuffer.alloc(d, src.len);
        errdefer b.free();
        try u.put(b, 0, src);
        return b;
    }

    /// The other half, once its copies ran.
    fn flip(u: *Uploader) !void {
        try u.ev[u.cur].record(u.stream);
        u.pending[u.cur] = true;
        u.cur ^= 1;
        if (u.pending[u.cur]) try u.ev[u.cur].synchronize();
        u.pending[u.cur] = false;
        u.fill = 0;
    }

    /// Every queued copy landed.
    pub fn finish(u: *Uploader) !void {
        try u.stream.synchronize();
        u.pending = .{ false, false };
        u.fill = 0;
    }
};

/// A file range read once into pinned memory with direct reads on several threads (no page cache, no faults), then
/// uploaded from there: two span buffers, the next read while the last span's copies run.
pub const SpanReader = struct {
    buf: [2]cuda.HostBuffer,
    ev: [2]cuda.Event,
    pending: [2]bool = .{ false, false },
    cur: u1 = 0,
    file: ?struct { path: [:0]const u8, f: dio.File } = null,

    pub const threads = 8;

    /// Spans of up to `bytes` (plus the alignment of their ends).
    pub fn init(d: *const cuda.Driver, bytes: usize) !SpanReader {
        const cap = bytes + 2 * dio.alignment;
        var a = try cuda.HostBuffer.alloc(d, cap);
        errdefer a.free();
        var b = try cuda.HostBuffer.alloc(d, cap);
        errdefer b.free();
        if (@intFromPtr(a.bytes.ptr) % dio.alignment != 0 or @intFromPtr(b.bytes.ptr) % dio.alignment != 0) return error.Unaligned;
        var e0 = try cuda.Event.init(d, false);
        errdefer e0.deinit();
        return .{ .buf = .{ a, b }, .ev = .{ e0, try cuda.Event.init(d, false) } };
    }

    pub fn deinit(r: *SpanReader) void {
        for (&r.ev, r.pending) |*e, p| if (p) e.synchronize() catch {};
        if (r.file) |*x| x.f.close();
        for (&r.buf) |*b| b.free();
        for (&r.ev) |*e| e.deinit();
    }

    /// `len` bytes at `offset` of `path` in the next span buffer (once that buffer's last copies ran); valid until
    /// the span after next, its copies queued on `u` before `done`.
    pub fn read(r: *SpanReader, path: [:0]const u8, offset: u64, len: usize) ![]const u8 {
        r.cur ^= 1;
        if (r.pending[r.cur]) try r.ev[r.cur].synchronize();
        r.pending[r.cur] = false;
        if (r.file == null or !std.mem.eql(u8, r.file.?.path, path)) {
            if (r.file) |*x| x.f.close();
            r.file = .{ .path = path, .f = try dio.File.open(path) };
        }
        const f = r.file.?.f;
        const buf: dio.Buffer = @alignCast(r.buf[r.cur].bytes);
        const lo = std.mem.alignBackward(u64, offset, dio.alignment);
        const whole = dio.span(offset, len);
        if (whole > buf.len) return error.BufferTooSmall;
        // equal aligned parts, one a thread
        const part = std.mem.alignForward(usize, (whole + threads - 1) / threads, dio.alignment);
        var ts: [threads]?std.Thread = @splat(null);
        var errs: [threads]?anyerror = @splat(null);
        for (0..threads) |i| {
            const at = i * part;
            if (at >= whole) break;
            const n = @min(part, whole - at);
            const into: dio.Buffer = @alignCast(buf[at..][0..n]); // `part` is a multiple of the alignment
            ts[i] = std.Thread.spawn(.{}, readPart, .{ f, into, lo + at, &errs[i] }) catch blk: {
                readPart(f, into, lo + at, &errs[i]); // no thread to spare: this part here
                break :blk null;
            };
        }
        for (ts) |t| if (t) |x| x.join();
        for (errs) |e| if (e) |x| return x;
        return buf[@intCast(offset - lo)..][0..len];
    }

    fn readPart(f: dio.File, into: []align(dio.alignment) u8, at: u64, err: *?anyerror) void {
        _ = f.read(into, at, into.len) catch |e| {
            err.* = e;
        };
    }

    /// The copies of the current span are queued on `u`'s stream: its buffer is reused once they ran.
    pub fn done(r: *SpanReader, u: *Uploader) !void {
        try r.ev[r.cur].record(u.stream);
        r.pending[r.cur] = true;
    }
};

/// Ask the kernel to read `bytes` (a mapped file's range) ahead of their first touch; a hint, so failures are ignored.
pub fn prefetch(bytes: []const u8) void {
    if (builtin.os.tag != .linux or bytes.len == 0) return;
    const page = std.heap.pageSize();
    const lo = std.mem.alignBackward(usize, @intFromPtr(bytes.ptr), page);
    const hi = @intFromPtr(bytes.ptr) + bytes.len;
    _ = std.os.linux.madvise(@ptrFromInt(lo), hi - lo, std.os.linux.MADV.WILLNEED);
}
