//! The load's bulk copies: host bytes (the pack's mapped tensors) to device buffers through two pinned staging
//! halves on one stream, so the copy of one half runs while the next is filled. A pageable source makes each
//! synchronous upload stage through the driver in small pieces; pinned halves let the copies run at memory speed.
//! The bytes a buffer holds after `finish` are those `put` was given, in any order of puts.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");

pub const Uploader = struct {
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

    pub fn deinit(u: *Uploader) void {
        u.stream.synchronize() catch {};
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

/// Ask the kernel to read `bytes` (a mapped file's range) ahead of their first touch; a hint, so failures are ignored.
pub fn prefetch(bytes: []const u8) void {
    if (builtin.os.tag != .linux or bytes.len == 0) return;
    const page = std.heap.pageSize();
    const lo = std.mem.alignBackward(usize, @intFromPtr(bytes.ptr), page);
    const hi = @intFromPtr(bytes.ptr) + bytes.len;
    _ = std.os.linux.madvise(@ptrFromInt(lo), hi - lo, std.os.linux.MADV.WILLNEED);
}
