//! Isolated DFlash2 projection replay: real rank weights and every recorded Q4 input/output.
//! Reference inputs isolate arithmetic errors; this does not prove a connected proposer or target verification.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const hs = @import("glm53_harness.zig");
const Head = glm.mtp.DraftHead;

fn projection(m: *const glm.dflash_weights.Model, head: *const Head, call: usize) !*const Head {
    if (call == 0) return &m.fc;
    if (call <= 5) return &m.layers[call - 1].kv;
    if (call == 36) return head;
    if (call > 36) return error.BadDflashTrace;
    const layer = &m.layers[(call - 6) / 6];
    return switch ((call - 6) % 6) {
        0 => &layer.a_kp,
        1 => &layer.qkv,
        2 => &layer.o,
        3 => &layer.m_kp,
        4 => &layer.gu,
        5 => &layer.down,
        else => unreachable,
    };
}

pub fn run(gpu: check.Gpu, work: []const u8, target: []const u8, draft: []const u8) !void {
    const a = gpu.gpa;
    const aot_dir = try std.fs.path.join(a, &.{ work, "aot" });
    defer a.free(aot_dir);
    var set = try cuda.aot.Set.load(a, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_dir = try std.fs.path.join(a, &.{ work, "out", "fixture" });
    defer a.free(fx_dir);
    var h: hs.Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(a, gpu.io, fx_dir), .stream = try cuda.Stream.init(gpu.d, false), .isolated = false };
    defer h.fx.deinit();
    defer h.stream.deinit();
    defer h.deinit();
    var hc = try glm.glue.Hc.load(gpu.d);
    defer hc.deinit();
    const glue: glm.glue.Glue = .{ .set = &set, .hc = &hc, .hidden = 4096, .streams = 1 };
    var kernels = try glm.dflash_projections.Kernels.load(gpu.d);
    defer kernels.deinit();
    var why: glm.config.Why = .{};
    const cfg = try glm.config.read(a, gpu.io, target, &why);
    var pack = try glm.weights.Pack.open(a, gpu.io, target);
    defer pack.close();
    var scratch = try glm.dflash_buffers.Scratch.init(gpu.d);
    defer scratch.deinit();
    const trace = h.fx.parsed.value.object.get("trace").?.array.items;
    for (0..2) |rank| {
        var model = try glm.dflash_weights.Model.open(a, gpu.io, gpu.d, draft, rank);
        defer model.deinit();
        const lay: glm.config.Layout = .{ .rank = rank, .world = 2, .cfg = &cfg };
        var head = try Head.load(.{ .gpa = a, .d = gpu.d, .pack = &pack, .lay = lay });
        defer head.deinit();
        for (0..9) |step| {
            try h.begin(rank, step);
            var sb: [16]u8 = undefined;
            const name = try std.fmt.bufPrint(&sb, "s{d}", .{step});
            var count: usize = 0;
            for (trace) |call| {
                const o = call.object;
                if (o.get("rank").?.integer != rank or !std.mem.eql(u8, o.get("step").?.string, name) or !std.mem.endsWith(u8, o.get("fn").?.string, "._mm")) continue;
                const seq = o.get("seq").?.integer;
                const weight = try projection(&model, &head, count);
                const shape = o.get("args").?.object.get("x").?.object.get("shape").?.array.items;
                const rows: usize = @intCast(shape[0].integer);
                try check.expect(shape[1].integer == weight.k, "DFlash2 projection input width", .{});
                const input = try h.fx.bytes(try h.arrayName(seq, "_mm", "x_in"));
                defer a.free(input);
                var x = try cuda.DeviceBuffer.fromHost(gpu.d, input);
                defer x.free();
                const fp32 = std.mem.eql(u8, o.get("ret").?.object.get("dtype").?.string, "torch.float32");
                var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * weight.n * @as(usize, if (fp32) 4 else 2));
                defer out.free();
                const sums = scratch.get("sums");
                try glue.groupSums(x.ptr, weight.k, sums.ptr, rows, weight.k, h.stream);
                try kernels.matmul(weight, x.ptr, sums.ptr, out.ptr, rows, fp32, scratch.get("matrix_partials").ptr, h.stream);
                try h.same(try h.arrayName(seq, "_mm", "ret"), out.ptr);
                count += 1;
            }
            try check.expect(count == 37, "DFlash2 rank {d} step {d}: {d} projections, expected 37", .{ rank, step, count });
        }
    }
    try check.expect(h.failures == 0, "DFlash2 projection replay: {d} failures", .{h.failures});
    check.pass("DFlash2 all 666 rank/step projections equal the captured BF16/FP32 outputs", .{});
}
