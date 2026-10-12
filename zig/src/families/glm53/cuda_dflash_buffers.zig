//! Bounded single-sequence DFlash2 state. Target model caches have separate ownership.
const std = @import("std");
const cuda = @import("cuda");
const D = @import("dflash2.zig");
const Buf = cuda.DeviceBuffer;

pub const Context = struct {
    logical: D.Context,
    valid: bool = true,
    position: Buf,
    keys: [5]Buf,
    values: [5]Buf,

    pub fn init(d: *const cuda.Driver, capacity: usize) !Context {
        if (capacity > std.math.maxInt(i64)) return error.BadDflashContext;
        var c: Context = undefined;
        c.logical = try D.Context.init(capacity, 2176, 2047, 8);
        c.valid = true;
        c.position = try Buf.alloc(d, 8);
        errdefer c.position.free();
        try c.position.fill8(0, null);
        var loaded: usize = 0;
        errdefer for (0..loaded) |i| {
            c.keys[i].free();
            c.values[i].free();
        };
        for (0..5) |i| {
            var key = try Buf.alloc(d, 4 * 2176 * 128 * 2);
            errdefer key.free();
            var value = try Buf.alloc(d, key.len);
            errdefer value.free();
            try key.fill8(0, null);
            try value.fill8(0, null);
            c.keys[i] = key;
            c.values[i] = value;
            loaded += 1;
        }
        return c;
    }

    pub fn deinit(c: *Context) void {
        for (&c.keys) |*b| b.free();
        for (&c.values) |*b| b.free();
        c.position.free();
    }

    /// Reset preserves physical ring bytes, as the reference does; logical masks exclude stale rows.
    pub fn reset(c: *Context, stream: cuda.Stream) !void {
        try c.position.fill8(0, stream.handle);
        c.logical.reset();
        c.valid = true;
    }

    /// Call after all five layers' committed KV writes are queued on this stream.
    /// Speculative block writes must never call this function.
    pub fn commitRows(c: *Context, rows: usize, stream: cuda.Stream) !void {
        if (!c.valid) return error.InvalidDflashContext;
        var next = c.logical;
        try next.append(rows);
        const position: i64 = @intCast(next.end);
        try c.position.uploadAsync(0, std.mem.asBytes(&position), stream.handle);
        c.logical = next;
    }
};

/// All intermediate shapes fit this arena. Views borrow its allocation and must not be freed.
pub const Scratch = struct {
    arena: Buf,
    pub const fields = .{
        .{ "taps", 64 * 20480 * 2 },       .{ "context", 64 * 4096 * 2 },
        .{ "x", 64 * 4096 * 2 },           .{ "normed", 64 * 4096 * 2 },
        .{ "sums", 64 * 320 * 4 },         .{ "dyn", 8 * 1024 * 2 },
        .{ "qkv", 64 * 3072 * 2 },         .{ "q", 8 * 16 * 128 * 2 },
        .{ "key", 64 * 4 * 128 * 2 },      .{ "value", 64 * 4 * 128 * 2 },
        .{ "attention", 8 * 2048 * 2 },    .{ "cos", 64 * 64 * 4 },
        .{ "sin", 64 * 64 * 4 },           .{ "partial", 8 * 4096 * 4 },
        .{ "gathered", 2 * 8 * 4096 * 4 }, .{ "gu", 8 * 12288 * 2 },
        .{ "activation", 8 * 6144 * 2 },   .{ "logits", 7 * 77440 * 2 },
        .{ "projected", 7 * 256 * 4 },     .{ "matrix_partials", 8 * 64 * 4096 * 4 },
        .{ "ids", 8 * 4 },
    };
    pub const bytes: usize = blk: {
        var end: usize = 0;
        for (fields) |field| end += std.mem.alignForward(usize, field[1], 256);
        break :blk end;
    };

    pub fn init(d: *const cuda.Driver) !Scratch {
        return .{ .arena = try Buf.alloc(d, bytes) };
    }

    pub fn deinit(s: *Scratch) void {
        s.arena.free();
    }

    pub fn get(s: Scratch, comptime name: []const u8) Buf {
        comptime var offset: usize = 0;
        inline for (fields) |field| {
            if (comptime std.mem.eql(u8, field[0], name)) return .{ .d = s.arena.d, .ptr = s.arena.ptr + offset, .len = field[1] };
            offset += comptime std.mem.alignForward(usize, field[1], 256);
        }
        @compileError("unknown DFlash scratch buffer: " ++ name);
    }
};
