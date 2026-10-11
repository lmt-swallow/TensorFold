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
    step_buf: [16]u8 = undefined,
    step_len: usize = 0,
    nbuf: [128]u8 = undefined,

    fn stepName(h: *const Harness) []const u8 {
        return h.step_buf[0..h.step_len];
    }

    pub fn deinit(h: *Harness) void {
        h.seqs.deinit(h.gpu.gpa);
        h.fns.deinit(h.gpu.gpa);
    }

    /// Points the harness at rank r's calls of step `s{s}`.
    pub fn begin(h: *Harness, r: usize, s: usize) !void {
        var sb: [16]u8 = undefined;
        return h.beginStep(r, try std.fmt.bufPrint(&sb, "s{d}", .{s}));
    }

    /// Points the harness at rank r's calls of the step named `step` (a dump's step name: s0, a3, l1, ...).
    pub fn beginStep(h: *Harness, r: usize, step: []const u8) !void {
        h.seqs.clearRetainingCapacity();
        h.fns.clearRetainingCapacity();
        h.at = 0;
        h.rank = r;
        @memcpy(h.step_buf[0..step.len], step);
        h.step_len = step.len;
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
        std.debug.print("FAIL glm53 r{d} {s}: the trace has no further {s} call\n", .{ h.rank, h.stepName(), function });
        return error.TestFailed;
    }

    pub fn arrayName(h: *Harness, seq: i64, function: []const u8, what: []const u8) ![]const u8 {
        return std.fmt.bufPrint(&h.nbuf, "r{d}_{s}_{d:0>3}_{s}_{s}", .{ h.rank, h.stepName(), @as(u64, @intCast(seq)), function, what });
    }

    /// The device bytes at `ptr` against the fixture array `name` (its length): a PASS, or a FAIL counted with the
    /// first differing byte; isolated runs then put the reference's bytes there.
    pub fn same(h: *Harness, name_: []const u8, ptr: cuda.abi.DevicePtr) !void {
        try h.stream.synchronize();
        // an argument the call left as it was has no "<arg>_out" (ref.Tracer saves only what changed): its "_in"
        // holds what the call left there
        var alt: [160]u8 = undefined;
        const arrays = h.fx.parsed.value.object.get("arrays").?.object;
        const name = if (arrays.get(name_) == null and std.mem.endsWith(u8, name_, "_out"))
            try std.fmt.bufPrint(&alt, "{s}_in", .{name_[0 .. name_.len - 4]})
        else
            name_;
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

    /// The same call's output `what` when the fixture has it (an argument the call left unchanged has no _out).
    pub fn alsoIf(h: *Harness, function: []const u8, what: []const u8, ptr: cuda.abi.DevicePtr) !void {
        const name = try h.arrayName(h.seqs.items[h.at - 1], function, what);
        const arrays = h.fx.parsed.value.object.get("arrays").?.object;
        if (arrays.get(name) == null) return;
        try h.same(name, ptr);
    }

    /// A per-step array `r{r}_{step}_{what}` against `ptr`.
    pub fn stepArray(h: *Harness, what: []const u8, ptr: cuda.abi.DevicePtr) !void {
        var b: [96]u8 = undefined;
        try h.same(try std.fmt.bufPrint(&b, "r{d}_{s}_{s}", .{ h.rank, h.stepName(), what }), ptr);
    }

    pub fn feed(h: *Harness, what: []const u8, buf: Buf) !void {
        var b: [96]u8 = undefined;
        const w = try h.fx.bytes(try std.fmt.bufPrint(&b, "r{d}_{s}_{s}", .{ h.rank, h.stepName(), what }));
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
