//! `qwen38fn-prompt <work-dir> <model-dir>`: Qwen3.8-Flash-Next's prompt path (cuda_prompt.zig) against TensorFold
//! 0.6.6's prompt fixture (tools/zig/qwen38fn/dump_prompt.py, packed by capture_prompt.sh): each prompt committed in
//! chunks of 2,048 rows from an empty stream, every layer's boundary values of every chunk (by sha256, the fixture's
//! "sha" table) and the last row's logits, then decode windows from the state the prompt left (boundaries and
//! logits). All 48 layers and the head are loaded.

const std = @import("std");
const cuda = @import("cuda");
const qfn = @import("qwen38fn");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;
const F = qfn.forward;
const P = qfn.prompt;

const Ctx = struct {
    gpu: Gpu,
    fx: *fixture.Fixture,
    sha: std.json.ObjectMap,
    prefix: [32]u8 = undefined,
    prefix_len: usize = 0,
    wide: usize,
    slots: usize,
    hidden: usize,
    streams: usize,
    y_bytes: usize, // 2 (a prompt chunk's bf16 slots) or 4 (a decode window's fp32 slots)
    failures: usize = 0,
    passes: usize = 0,
    first_bad: ?[]const u8 = null,
    name: [96]u8 = undefined,

    fn setPrefix(self: *Ctx, comptime fmt: []const u8, args: anytype) !void {
        const s = try std.fmt.bufPrint(&self.prefix, fmt, args);
        self.prefix_len = s.len;
    }

    fn hashCheck(self: *Ctx, l: usize, what: []const u8, buf: cuda.DeviceBuffer, len: usize) !void {
        const n = try std.fmt.bufPrint(&self.name, "{s}L{d}_{s}", .{ self.prefix[0..self.prefix_len], l, what });
        const want = (self.sha.get(n) orelse {
            std.debug.print("FAIL qwen38fn: the fixture has no hash {s}\n", .{n});
            self.failures += 1;
            return;
        }).string;
        const bytes = try self.gpu.gpa.alloc(u8, len);
        defer self.gpu.gpa.free(bytes);
        try buf.download(0, bytes);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        if (!std.mem.eql(u8, &hex, want)) {
            std.debug.print("FAIL qwen38fn {s}: sha256 {s}, 0.6.6's {s}\n", .{ n, hex[0..12], want[0..12] });
            self.failures += 1;
            if (self.first_bad == null) self.first_bad = try self.gpu.gpa.dupe(u8, n);
        } else self.passes += 1;
    }

    fn promptIn(ctx: ?*anyopaque, l: usize, h: cuda.DeviceBuffer, rows: usize) anyerror!void {
        const self: *Ctx = @ptrCast(@alignCast(ctx.?));
        try self.hashCheck(l, "h", h, rows * self.wide * 2);
    }

    fn promptOut(ctx: ?*anyopaque, l: usize, y: cuda.DeviceBuffer, wts: cuda.DeviceBuffer, inj: cuda.DeviceBuffer, rows: usize) anyerror!void {
        const self: *Ctx = @ptrCast(@alignCast(ctx.?));
        try self.hashCheck(l, "y", y, rows * self.slots * self.hidden * self.y_bytes);
        try self.hashCheck(l, "wts", wts, rows * self.slots * 4);
        try self.hashCheck(l, "inj", inj, rows * self.streams * 2);
    }

    fn decodeIn(ctx: ?*anyopaque, l: usize, b: *F.Buffers, rows: usize) anyerror!void {
        try promptIn(ctx, l, b.w.h, rows);
    }

    fn decodeOut(ctx: ?*anyopaque, l: usize, b: *F.Buffers, rows: usize) anyerror!void {
        try promptOut(ctx, l, b.moe.y, b.wts, b.w.inj_m, rows);
    }

    fn logits(self: *Ctx, name: []const u8, buf: cuda.DeviceBuffer) !void {
        const want = try self.fx.bytes(name);
        defer self.gpu.gpa.free(want);
        const got = try self.gpu.gpa.alloc(u8, want.len);
        defer self.gpu.gpa.free(got);
        try buf.download(0, got);
        if (std.mem.indexOfDiff(u8, got, want)) |at| {
            std.debug.print("FAIL qwen38fn {s}: differs from byte {d}\n", .{ name, at });
            self.failures += 1;
        } else self.passes += 1;
    }
};

pub fn prompt(gpu: Gpu, work: []const u8, model: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: qfn.config.Why = .{};
    const cfg = qfn.config.read(gpa, gpu.io, model, &why) catch |e| {
        std.debug.print("FAIL qwen38fn config: {t}: {s}\n", .{ e, why.text() });
        return e;
    };
    const aot_dir = try std.fs.path.join(gpa, &.{ work, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_dir = try std.fs.path.join(gpa, &.{ work, "out", "fixture" });
    defer gpa.free(fx_dir);
    var fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir);
    defer fx.deinit();
    const sha = (fx.parsed.value.object.get("sha") orelse return error.MissingField).object;

    const t0 = check.now(gpu.io);
    const m = try F.Model.load(gpa, gpu.io, d, model, cfg, cfg.layers, true);
    defer m.deinit();
    check.pass("qwen38fn: {d} layers loaded in {d:.1} s", .{ cfg.layers, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9 });
    const capacity: usize = @intCast(try fx.int("capacity"));
    const rows: usize = @intCast(try fx.int("rows"));
    const prows: usize = @intCast(try fx.int("prefill_rows"));
    var b = try F.Buffers.init(gpa, d, m, rows, capacity);
    defer b.deinit();
    var pb = try P.Buffers.init(gpa, d, m, prows, capacity);
    defer pb.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var fw = F.Forward.init(d, m, &set);
    var pr = try P.Prompt.init(&fw, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer pr.deinit();
    var ctx: Ctx = .{ .gpu = gpu, .fx = &fx, .sha = sha, .wide = cfg.wide(), .slots = cfg.slots(), .hidden = cfg.hidden, .streams = cfg.streams, .y_bytes = 2 };
    const pprobe: P.Probe = .{ .ctx = &ctx, .layer_in = Ctx.promptIn, .layer_out = Ctx.promptOut };
    const dprobe: F.Probe = .{ .ctx = &ctx, .layer_in = Ctx.decodeIn, .layer_out = Ctx.decodeOut };

    const prompts: usize = @intCast(try fx.int("prompts"));
    var nb: [48]u8 = undefined;
    for (0..prompts) |pi| {
        const n: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "p{d}_len", .{pi})));
        const ids_b = try fx.bytes(try std.fmt.bufPrint(&nb, "p{d}_ids", .{pi}));
        defer gpa.free(ids_b);
        const ids = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(ids_b)));
        var st = try F.Stream.init(gpa, d, m, capacity, rows);
        defer st.deinit();
        const before = ctx.failures;
        const tp = check.now(gpu.io);
        var start: usize = 0;
        var chunk: usize = 0;
        ctx.y_bytes = 2;
        while (start < n) : (chunk += 1) {
            const end = @min(start + prows, n);
            try ctx.setPrefix("p{d}_c{d}_", .{ pi, chunk });
            try pr.stage(&st, &pb, ids[start..end], stream);
            try pr.compute(&st, &pb, end - start, end == n, pprobe, stream);
            try pr.commit(&st, &pb, end - start, stream);
            start = end;
        }
        try stream.synchronize();
        const ms = @as(f64, @floatFromInt(check.now(gpu.io) - tp)) / 1e6;
        try ctx.logits(try std.fmt.bufPrint(&nb, "p{d}_logits", .{pi}), pb.logits);
        // decode windows from the prompt's state
        ctx.y_bytes = 4;
        const decodes: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "p{d}_decodes", .{pi})));
        for (0..decodes) |j| {
            const R: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "p{d}_d{d}_rows", .{ pi, j })));
            const keep: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "p{d}_d{d}_keep", .{ pi, j })));
            const db = try fx.bytes(try std.fmt.bufPrint(&nb, "p{d}_d{d}_ids", .{ pi, j }));
            defer gpa.free(db);
            const dids = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(db)));
            try ctx.setPrefix("p{d}_d{d}_", .{ pi, j });
            try fw.stage(&st, &b, dids, stream);
            try fw.compute(&st, &b, R, dprobe, stream);
            try stream.synchronize();
            try ctx.logits(try std.fmt.bufPrint(&nb, "p{d}_d{d}_logits", .{ pi, j }), b.logits);
            try fw.commit(&st, R, keep, stream);
        }
        try stream.synchronize();
        if (ctx.failures == before) check.pass("BITEXACT qwen38fn prompt {d}: {d} tokens in {d} chunks ({d:.0} ms, {d:.0} tok/s with the checks), every boundary, the logits and {d} decode windows after it equal to 0.6.6's", .{ pi, n, chunk, ms, @as(f64, @floatFromInt(n)) / ms * 1000, decodes });
    }
    if (ctx.first_bad) |fb| std.debug.print("first difference: {s}\n", .{fb});
    try check.expect(ctx.failures == 0, "qwen38fn-prompt: {d} values differ ({d} equal)", .{ ctx.failures, ctx.passes });
    check.pass("qwen38fn-prompt: {d} prompts, {d} values bit-exact", .{ prompts, ctx.passes });
}
