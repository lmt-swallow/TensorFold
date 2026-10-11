//! `tf-glm53-validate <pack-dir>`: GLM-5.3-Flash's EXL3 pack against the family's plan, host only (headers and maps,
//! no tensor bytes read, no GPU): every tensor the engine reads is present with the dtype and shape weights.py
//! expects and a split rule, nothing of the text model is left over, and each rank's bytes by part. Exit 1 on a
//! mismatch.
const std = @import("std");
const core = @import("core");
const glm53 = @import("glm53");

const config = glm53.config;
const plan = glm53.plan;
const map = glm53.weights_map;

fn dtypeOf(d: plan.DType) core.checkpoint.DType {
    return switch (d) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        .i16 => .i16,
        .i32 => .i32,
    };
}

const Ctx = struct {
    text: *core.Checkpoint,
    head: *core.Checkpoint,
    ranks: *[config.world]plan.Bytes,
    cfg: *const config.Config,
    bad: *usize,
};

fn check(x: Ctx, w: plan.Want) !void {
    const ck = if (std.mem.startsWith(u8, w.name, "lm_head.")) x.head else x.text;
    _ = ck.expect(w.name, dtypeOf(w.dtype), w.shape) catch |e| {
        x.bad.* += 1;
        if (x.bad.* <= 10) std.debug.print("BAD {s}: {t}\n", .{ w.name, e });
        return;
    };
    if (map.rule(w.name) == null) {
        x.bad.* += 1;
        std.debug.print("BAD {s}: no split rule\n", .{w.name});
        return;
    }
    for (x.ranks, 0..) |*b, r| try b.add(.{ .rank = r, .world = config.world, .cfg = x.cfg }, w);
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) {
        std.debug.print("usage: tf-glm53-validate <pack-dir>\n", .{});
        return 2;
    }
    const gpa = init.gpa;
    const io = init.io;
    var why: config.Why = .{};
    const cfg = config.read(gpa, io, args[1], &why) catch |e| {
        std.debug.print("FAIL config: {t} {s}\n", .{ e, why.text() });
        return 1;
    };
    var text = try core.Checkpoint.openModelPrefix(gpa, io, args[1], map.lm);
    defer text.close();
    var head = try core.Checkpoint.openModelPrefix(gpa, io, args[1], "lm_head.");
    defer head.close();
    var ranks: [config.world]plan.Bytes = @splat(.{});
    var bad: usize = 0;
    try plan.visit(&cfg, Ctx{ .text = &text, .head = &head, .ranks = &ranks, .cfg = &cfg, .bad = &bad }, check);
    const left = text.unused() + head.unused();
    const gib = 1024.0 * 1024.0 * 1024.0;
    for (ranks, 0..) |b, r| std.debug.print("RESULT rank {d}: {d:.3} GiB = experts {d:.3} + layers' BF16 {d:.3} + globals {d:.3}\n", .{
        r,                                         @as(f64, @floatFromInt(b.total)) / gib, @as(f64, @floatFromInt(b.experts)) / gib,
        @as(f64, @floatFromInt(b.dense)) / gib, @as(f64, @floatFromInt(b.globals)) / gib,
    });
    if (bad != 0 or left != 0) {
        std.debug.print("FAIL glm53-validate: {d} mismatched, {d} left over\n", .{ bad, left });
        return 1;
    }
    std.debug.print("PASS glm53-validate: {d} layers + MTP, every text tensor planned\n", .{cfg.layers});
    return 0;
}
