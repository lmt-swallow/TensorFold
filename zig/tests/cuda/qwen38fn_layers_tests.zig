//! `qwen38fn-layers <work-dir> <model-dir> [layers] [isolated]`: Qwen3.8-Flash-Next's decode forward
//! (cuda_forward.zig) against TensorFold 0.6.6's whole-forward fixture (tools/zig/qwen38fn/dump_layers.py, packed by
//! capture_layers.sh): every layer's entering streams and pending MoE write-back, the n-gram rows of the PLE layer,
//! and with all layers loaded the head's logits, window after window with this engine's own state carried across.
//! `layers` loads the first N layers only (their boundaries are still the full model's); `isolated` puts the
//! reference's boundary values in place after each comparison, so one wrong layer cannot hide the ones after it.

const std = @import("std");
const cuda = @import("cuda");
const qfn = @import("qwen38fn");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;
const F = qfn.forward;

const Ctx = struct {
    gpu: Gpu,
    fx: *fixture.Fixture,
    window: usize = 0,
    isolated: bool,
    failures: usize = 0,
    passes: usize = 0,
    first_bad: ?[2]usize = null, // (window, layer) of the first difference
    name: [96]u8 = undefined,

    fn cmp(self: *Ctx, what: []const u8, l: ?usize, buf: cuda.DeviceBuffer, offset: usize) !bool {
        const n = if (l) |x| try std.fmt.bufPrint(&self.name, "w{d}_L{d}_{s}", .{ self.window, x, what }) else try std.fmt.bufPrint(&self.name, "w{d}_{s}", .{ self.window, what });
        const want = try self.fx.bytes(n);
        defer self.gpu.gpa.free(want);
        const got = try self.gpu.gpa.alloc(u8, want.len);
        defer self.gpu.gpa.free(got);
        try buf.download(offset, got);
        var ok = true;
        if (std.mem.indexOfDiff(u8, got, want)) |at| {
            var diff: usize = 0;
            for (got, want) |a, b| diff += @intFromBool(a != b);
            std.debug.print("FAIL qwen38fn {s}: {d} of {d} bytes differ, the first at byte {d}\n", .{ n, diff, want.len, at });
            self.failures += 1;
            if (self.first_bad == null) self.first_bad = .{ self.window, l orelse 9999 };
            ok = false;
        } else self.passes += 1;
        if (self.isolated) try buf.upload(offset, want);
        return ok;
    }

    fn layerIn(ctx: ?*anyopaque, l: usize, b: *F.Buffers, rows: usize) anyerror!void {
        _ = rows;
        const self: *Ctx = @ptrCast(@alignCast(ctx.?));
        _ = try self.cmp("h", l, b.w.h, 0);
    }

    fn layerOut(ctx: ?*anyopaque, l: usize, b: *F.Buffers, rows: usize) anyerror!void {
        _ = rows;
        const self: *Ctx = @ptrCast(@alignCast(ctx.?));
        _ = try self.cmp("y", l, b.moe.y, 0);
        _ = try self.cmp("wts", l, b.wts, 0);
        _ = try self.cmp("inj", l, b.w.inj_m, 0);
    }
};

pub fn layers(gpu: Gpu, work: []const u8, model: []const u8, n_layers: ?usize, isolated: bool) !void {
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
    const total: usize = @intCast(try fx.int("layers"));
    const L = n_layers orelse total;
    try check.expect(L >= 1 and L <= total, "qwen38fn-layers: {d} layers asked, the fixture has {d}", .{ L, total });
    const with_head = L == cfg.layers;

    var t0 = check.now(gpu.io);
    const m = try F.Model.load(gpa, gpu.io, d, model, cfg, L, with_head);
    defer m.deinit();
    const t1 = check.now(gpu.io);
    check.pass("qwen38fn: {d} layers loaded in {d:.1} s (centred offset {d}, head {})", .{ L, @as(f64, @floatFromInt(t1 - t0)) / 1e9, m.offset, with_head });
    t0 = t1;
    const rows: usize = @intCast(try fx.int("rows"));
    const capacity: usize = @intCast(try fx.int("capacity"));
    var st = try F.Stream.init(gpa, d, m, capacity, rows);
    defer st.deinit();
    var b = try F.Buffers.init(gpa, d, m, rows, capacity);
    defer b.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var fw = F.Forward.init(d, m, &set);
    var ctx: Ctx = .{ .gpu = gpu, .fx = &fx, .isolated = isolated };
    const probe: F.Probe = .{ .ctx = &ctx, .layer_in = Ctx.layerIn, .layer_out = Ctx.layerOut };

    // a warm-up the fixture ran unrecorded (dump_layers.py --warm): the same windows of 16 rows
    const warm_n: usize = @intCast(fx.int("warm") catch 0); // fixtures from before --warm have neither
    if (warm_n > 0) {
        const warm_bytes = try fx.bytes("warm_ids");
        defer gpa.free(warm_bytes);
        const warm = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(warm_bytes)));
        var w0: usize = 0;
        while (w0 < warm_n) : (w0 += rows) {
            const n = @min(rows, warm_n - w0);
            try fw.stage(&st, &b, warm[w0..][0..n], stream);
            try fw.compute(&st, &b, n, .{}, stream);
            try fw.commit(&st, n, n, stream);
        }
        try stream.synchronize();
        check.pass("qwen38fn: warmed up to position {d}", .{st.pos});
    }

    const windows: usize = @intCast(try fx.int("windows"));
    var nb: [32]u8 = undefined;
    for (0..windows) |i| {
        ctx.window = i;
        const R: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_rows", .{i})));
        const keep: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_keep", .{i})));
        const pos: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "w{d}_pos", .{i})));
        try check.expect(pos == st.pos, "qwen38fn: window {d} at {d}, the stream is at {d}", .{ i, pos, st.pos });
        const ids_bytes = try fx.bytes(try std.fmt.bufPrint(&nb, "w{d}_ids", .{i}));
        defer gpa.free(ids_bytes);
        const ids = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(ids_bytes)));
        const before = ctx.failures;
        try fw.stage(&st, &b, ids, stream);
        if (m.ng != null) {
            try stream.synchronize();
            const want = try fx.bytes(try std.fmt.bufPrint(&nb, "w{d}_ple_rows", .{i}));
            defer gpa.free(want);
            const got = b.ple_host[0..want.len];
            if (!std.mem.eql(u8, got, want)) {
                std.debug.print("FAIL qwen38fn w{d}: the staged n-gram rows differ from 0.6.6's\n", .{i});
                ctx.failures += 1;
            } else ctx.passes += 1;
        }
        try fw.compute(&st, &b, R, probe, stream);
        try stream.synchronize();
        if (with_head) _ = try ctx.cmp("logits", null, b.logits, 0);
        try fw.commit(&st, R, keep, stream);
        try stream.synchronize();
        if (ctx.failures == before) check.pass("BITEXACT qwen38fn {d} layers window {d} ({d} rows, {d} kept, at {d}): every boundary{s} equal to 0.6.6's", .{ L, i, R, keep, pos, if (with_head) " and the logits" else "" });
    }
    const t2 = check.now(gpu.io);
    if (ctx.first_bad) |fb| std.debug.print("first difference: window {d}, layer {d}\n", .{ fb[0], fb[1] });
    try check.expect(ctx.failures == 0, "qwen38fn-layers: {d} values differ ({d} equal)", .{ ctx.failures, ctx.passes });
    check.pass("qwen38fn-layers: {d} layers, {d} windows, {d} values bit-exact{s} in {d:.1} s", .{ L, windows, ctx.passes, if (isolated) " (isolated)" else "", @as(f64, @floatFromInt(t2 - t0)) / 1e9 });
}
