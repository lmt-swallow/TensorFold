//! Target layer taps, kept separate from the drafter ring until verification accepts their rows.
const cuda = @import("cuda");
const M = @import("cuda_model.zig");
const G = @import("cuda_glue.zig").Glue;
const Buf = cuda.DeviceBuffer;

pub const Collector = struct {
    layers: [5]usize = .{ 5, 14, 24, 33, 42 },
    taps: [5]Buf,
    capacity: usize,
    rows: usize = 0,
    captured: u5 = 0,
    glue: G,
    stream: cuda.Stream,

    pub fn init(d: *const cuda.Driver, capacity: usize, glue: G, stream: cuda.Stream) !Collector {
        if (capacity == 0 or capacity > 2048) return error.BadDflashTapRows;
        var out: Collector = .{ .taps = undefined, .capacity = capacity, .glue = glue, .stream = stream };
        var loaded: usize = 0;
        errdefer for (out.taps[0..loaded]) |*b| b.free();
        for (&out.taps) |*tap| {
            tap.* = try Buf.alloc(d, capacity * 4096 * 2);
            loaded += 1;
        }
        return out;
    }

    pub fn deinit(c: *Collector) void {
        for (&c.taps) |*b| b.free();
    }

    /// Begin one target forward. Rejected rows may be captured but never enter the committed ring here.
    pub fn begin(c: *Collector, rows: usize) !M.Hook {
        if (rows == 0 or rows > c.capacity) return error.BadDflashTapRows;
        c.rows = rows;
        c.captured = 0;
        return .{ .ctx = c, .run = capture };
    }

    fn capture(raw: *anyopaque, index: usize, x: cuda.abi.DevicePtr, rows: usize) !void {
        const c: *Collector = @ptrCast(@alignCast(raw));
        if (rows != c.rows) return error.BadDflashTapRows;
        for (c.layers, 0..) |layer, slot| {
            if (index != layer) continue;
            const bit: u5 = @as(u5, 1) << @as(u3, @intCast(slot));
            if (c.captured & bit != 0) return error.DuplicateDflashTap;
            try c.glue.streamMean(x, c.taps[slot].ptr, rows, c.stream);
            c.captured |= bit;
        }
    }

    /// Pack only caller-selected committed rows into [rows,5*4096] BF16, rows<=64.
    /// This performs device copies and does not synchronize or advance either model's position.
    pub fn pack(c: *const Collector, start: usize, rows: usize, out: Buf) !void {
        if (c.captured != 31 or rows == 0 or rows > 64 or start > c.rows or rows > c.rows - start or out.len < rows * 5 * 4096 * 2) return error.BadDflashTapRows;
        for (c.taps) |tap| {
            if (tap.ptr < out.ptr + out.len and out.ptr < tap.ptr + tap.len) return error.AliasedDflashTaps;
        }
        for (0..rows) |row| {
            for (c.taps, 0..) |tap, slot| {
                try out.copyFrom((row * 5 + slot) * 4096 * 2, tap.ptr + (start + row) * 4096 * 2, 4096 * 2, c.stream.handle);
            }
        }
    }
};
