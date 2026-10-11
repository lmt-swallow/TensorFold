//! What the glm53-* GPU tests share: the call-trace harness over a dump's fixture (tools/zig/glm53/ref.py's Tracer
//! names: r{rank}_{step}_{seq}_{function}_{argument}_in / _out / ret), sha256 of device bytes, and the two-rank
//! all-gather emulated on one GPU (each rank's partial copied into every rank's gather buffer, rank 0 first).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;
const Buf = cuda.DeviceBuffer;

/// The trace's calls of one rank's step in order, and the comparison bookkeeping.
pub const Harness = struct {
    gpu: Gpu,
    fx: fixture.Fixture,
    stream: cuda.Stream,
    isolated: bool,
    failures: usize = 0,
    passes: usize = 0,
    // the current (rank, step)'s calls: their seq numbers and function names; `at` the next unread
    seqs: std.ArrayList(i64) = .empty,
    fns: std.ArrayList([]const u8) = .empty,
    at: usize = 0,
    rank: usize = 0,
    step: usize = 0,
    nbuf: [128]u8 = undefined,

    pub fn deinit(h: *Harness) void {
        h.seqs.deinit(h.gpu.gpa);
        h.fns.deinit(h.gpu.gpa);
    }

    /// Points the harness at rank r's calls of step s.
    pub fn begin(h: *Harness, r: usize, s: usize) !void {
        h.seqs.clearRetainingCapacity();
        h.fns.clearRetainingCapacity();
        h.at = 0;
        h.rank = r;
        h.step = s;
        var sb: [16]u8 = undefined;
        const step = try std.fmt.bufPrint(&sb, "s{d}", .{s});
        const trace = h.fx.parsed.value.object.get("trace").?.array.items;
        for (trace) |c| {
            const o = c.object;
            if (o.get("rank").?.integer != r or !std.mem.eql(u8, o.get("step").?.string, step)) continue;
            const f = o.get("fn").?.string;
            try h.seqs.append(h.gpu.gpa, o.get("seq").?.integer);
            try h.fns.append(h.gpu.gpa, f[std.mem.lastIndexOfScalar(u8, f, '.').? + 1 ..]);
        }
    }

    /// The next traced call of `function` (calls of others before it are skipped: group sums no BF16 matmul reads).
    pub fn next(h: *Harness, function: []const u8) !i64 {
        while (h.at < h.fns.items.len) : (h.at += 1) {
            if (std.mem.eql(u8, h.fns.items[h.at], function)) {
                h.at += 1;
                return h.seqs.items[h.at - 1];
            }
        }
        std.debug.print("FAIL glm53 r{d} s{d}: the trace has no further {s} call\n", .{ h.rank, h.step, function });
        return error.TestFailed;
    }

    pub fn arrayName(h: *Harness, seq: i64, function: []const u8, what: []const u8) ![]const u8 {
        return std.fmt.bufPrint(&h.nbuf, "r{d}_s{d}_{d:0>3}_{s}_{s}", .{ h.rank, h.step, @as(u64, @intCast(seq)), function, what });
    }

    /// The device bytes at `ptr` against the fixture array `name` (its length): a PASS, or a FAIL counted with the
    /// first differing byte; isolated runs then put the reference's bytes there.
    pub fn same(h: *Harness, name: []const u8, ptr: cuda.abi.DevicePtr) !void {
        try h.stream.synchronize();
        const want = h.fx.bytes(name) catch |e| {
            std.debug.print("FAIL glm53 {s}: not in the fixture ({t})\n", .{ name, e });
            h.failures += 1;
            return;
        };
        defer h.gpu.gpa.free(want);
        const got = try h.gpu.gpa.alloc(u8, want.len);
        defer h.gpu.gpa.free(got);
        const view: Buf = .{ .d = h.gpu.d, .ptr = ptr, .len = want.len };
        try view.download(0, got);
        if (std.mem.indexOfDiff(u8, got, want)) |at| {
            var diff: usize = 0;
            for (got, want) |a, b| diff += @intFromBool(a != b);
            std.debug.print("FAIL glm53 {s}: {d} of {d} bytes differ, the first at byte {d}\n", .{ name, diff, want.len, at });
            h.failures += 1;
        } else h.passes += 1;
        if (h.isolated) try view.upload(0, want);
    }

    /// `same` over the byte ranges [start, start + len) of `starts` only (buffers whose other bytes nothing writes:
    /// the reference left whatever its allocator held there).
    pub fn sameRanges(h: *Harness, name: []const u8, ptr: cuda.abi.DevicePtr, starts: []const usize, len: usize) !void {
        try h.stream.synchronize();
        const want = try h.fx.bytes(name);
        defer h.gpu.gpa.free(want);
        const got = try h.gpu.gpa.alloc(u8, want.len);
        defer h.gpu.gpa.free(got);
        const view: Buf = .{ .d = h.gpu.d, .ptr = ptr, .len = want.len };
        try view.download(0, got);
        var diff: usize = 0;
        for (starts) |a| {
            if (!std.mem.eql(u8, got[a..][0..len], want[a..][0..len])) diff += 1;
        }
        if (diff != 0) {
            std.debug.print("FAIL glm53 {s}: {d} of {d} ranges differ\n", .{ name, diff, starts.len });
            h.failures += 1;
        } else h.passes += 1;
    }

    /// The next `function` call's output `what` (an argument's "<arg>_out", or "ret") against `ptr`.
    pub fn call(h: *Harness, function: []const u8, what: []const u8, ptr: cuda.abi.DevicePtr) !void {
        const seq = try h.next(function);
        try h.same(try h.arrayName(seq, function, what), ptr);
    }

    /// The same call's further outputs (after `call`).
    pub fn also(h: *Harness, function: []const u8, what: []const u8, ptr: cuda.abi.DevicePtr) !void {
        try h.same(try h.arrayName(h.seqs.items[h.at - 1], function, what), ptr);
    }

    /// A per-step array `r{r}_s{s}_{what}` against `ptr`.
    pub fn stepArray(h: *Harness, what: []const u8, ptr: cuda.abi.DevicePtr) !void {
        var b: [96]u8 = undefined;
        try h.same(try std.fmt.bufPrint(&b, "r{d}_s{d}_{s}", .{ h.rank, h.step, what }), ptr);
    }

    pub fn feed(h: *Harness, what: []const u8, buf: Buf) !void {
        var b: [96]u8 = undefined;
        const w = try h.fx.bytes(try std.fmt.bufPrint(&b, "r{d}_s{d}_{s}", .{ h.rank, h.step, what }));
        defer h.gpu.gpa.free(w);
        try buf.upload(0, w);
    }
};

pub fn sha(gpa: std.mem.Allocator, buf: Buf) ![64]u8 {
    const host = try gpa.alloc(u8, buf.len);
    defer gpa.free(host);
    try buf.download(0, host);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(host, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

/// The all-gather of `n` bytes a rank: every rank's `part` into each rank's `gath`, rank 0 first.
pub fn gather(parts: []const cuda.abi.DevicePtr, gaths: []const cuda.abi.DevicePtr, n: usize, d: *const cuda.Driver, stream: cuda.Stream) !void {
    for (gaths) |g| {
        const view: Buf = .{ .d = d, .ptr = g, .len = parts.len * n };
        for (parts, 0..) |p, k| try view.copyFrom(k * n, p, n, stream.handle);
    }
}
