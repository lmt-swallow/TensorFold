//! Opt-in decode routing capture. D2D snapshots are downloaded only after the caller's existing sample sync.
const std = @import("std");
const cuda = @import("cuda");
const U = @import("expert_union.zig");
const Buf = cuda.DeviceBuffer;
pub const Span = struct { stream: usize, pos: usize, start: usize, rows: usize };
pub const Trace = struct {
    const Record = struct { layer: usize, phase: []const u8, step: usize, serial: usize, rows: usize, slots: usize, experts: usize, spans: [16]Span, nspan: usize };
    gpu: Buf,
    host: [U.max_words]i32 = undefined,
    records: [U.max_layers]Record = undefined,
    spans: [16]Span = undefined,
    nspan: usize = 0,
    nlayer: usize = 0,
    rows: usize = 0,
    slots: usize = 0,
    experts: usize = 0,
    phase: []const u8 = "",
    step: usize = 0,
    serial: usize = 0,

    pub fn init(d: *const cuda.Driver) !Trace {
        return .{ .gpu = try Buf.alloc(d, U.max_words * 4) };
    }
    pub fn deinit(t: *Trace) void {
        t.gpu.free();
    }
    pub fn begin(t: *Trace, phase: []const u8, step: usize, rows: usize, slots: usize, experts: usize, spans: []const Span) !void {
        _ = try U.words(1, rows, slots);
        if (experts == 0 or experts > U.max_experts) return error.InvalidShape;
        if (spans.len == 0 or spans.len > t.spans.len) return error.InvalidSpan;
        var at: usize = 0;
        for (spans) |s| {
            if (s.start != at or s.rows == 0 or s.rows > rows -| at) return error.InvalidSpan;
            at += s.rows;
        }
        if (at != rows) return error.InvalidSpan;
        t.rows = rows;
        t.slots = slots;
        t.experts = experts;
        t.phase = phase;
        t.step = step;
        t.serial += 1;
        t.nspan = spans.len;
        @memcpy(t.spans[0..spans.len], spans);
    }
    pub fn capture(t: *Trace, layer: usize, pick: Buf, rows: usize, slots: usize, stream: cuda.Stream) !void {
        if (rows != t.rows or slots != t.slots or t.nlayer >= U.max_layers) return error.InvalidShape;
        const n = rows * slots;
        try t.gpu.copyFrom(t.nlayer * U.max_rows * U.max_slots * 4, pick.ptr, n * 4, stream.handle);
        t.records[t.nlayer] = .{ .layer = layer, .phase = t.phase, .step = t.step, .serial = t.serial, .rows = rows, .slots = slots, .experts = t.experts, .spans = t.spans, .nspan = t.nspan };
        t.nlayer += 1;
    }
    /// The stream must already have completed every capture in this phase. No synchronization is added here.
    pub fn flush(t: *Trace, rank: usize) !void {
        if (t.nlayer == 0) return;
        const stride = U.max_rows * U.max_slots;
        try t.gpu.download(0, std.mem.sliceAsBytes(t.host[0 .. t.nlayer * stride]));
        for (t.records[0..t.nlayer], 0..) |rec, i| {
            const ids = t.host[i * stride ..][0 .. rec.rows * rec.slots];
            const all = try U.count(ids, rec.rows, rec.slots, rec.experts, .{ .start = 0, .rows = rec.rows });
            std.log.info("glm_union rank={d} phase_id={d} phase={s} step={d} layer={d} rows={d} union={d}", .{ rank, rec.serial, rec.phase, rec.step, rec.layer, rec.rows, all });
            for (rec.spans[0..rec.nspan]) |span| {
                const count = try U.count(ids, rec.rows, rec.slots, rec.experts, .{ .start = span.start, .rows = span.rows });
                std.log.info("glm_union_stream rank={d} phase_id={d} stream={d} pos={d} layer={d} rows={d} union={d}", .{ rank, rec.serial, span.stream, span.pos, rec.layer, span.rows, count });
            }
        }
        t.nlayer = 0;
    }
};

test "phase metadata validates bounded contiguous stream rows and reuses empty capture" {
    var t: Trace = .{ .gpu = undefined };
    try t.begin("main", 0, 16, 9, 288, &.{ .{ .stream = 7, .pos = 52, .start = 0, .rows = 8 }, .{ .stream = 8, .pos = 79, .start = 8, .rows = 8 } });
    try std.testing.expectEqual(2, t.nspan);
    try std.testing.expectEqual(8, t.spans[1].rows);
    try t.flush(0); // No allocation or transfer occurs without captured rows.
    try t.begin("mtp_step", 2, 1, 9, 288, &.{.{ .stream = 8, .pos = 80, .start = 0, .rows = 1 }});
    try std.testing.expectEqualStrings("mtp_step", t.phase);
    try std.testing.expectEqual(1, t.nspan);
    try std.testing.expectEqual(2, t.serial);
    try std.testing.expectError(error.InvalidShape, t.begin("main", 0, 17, 9, 288, &.{}));
    try std.testing.expectError(error.InvalidSpan, t.begin("main", 0, 2, 9, 288, &.{.{ .stream = 7, .pos = 0, .start = 1, .rows = 1 }}));
    try std.testing.expectError(error.InvalidSpan, t.begin("main", 0, 2, 9, 288, &.{.{ .stream = 7, .pos = 0, .start = 0, .rows = 1 }}));
}

test "capture overflow and mismatched rows fail before accessing device storage" {
    var t: Trace = .{ .gpu = undefined, .rows = 1, .slots = 9, .nlayer = U.max_layers };
    try std.testing.expectError(error.InvalidShape, t.capture(3, undefined, 1, 9, undefined));
    t.nlayer = 0;
    try std.testing.expectError(error.InvalidShape, t.capture(3, undefined, 2, 9, undefined));
    try std.testing.expectEqual(0, t.nlayer);
}
