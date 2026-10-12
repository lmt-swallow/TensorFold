//! `glm53-drafted <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir>`: the forward fixture's request decoded
//! as the reference served it, greedy with drafts (drafter_choice.py `auto_decode` without DFlash2: each round copy
//! drafts when the context repeats itself, else an MTP chain of up to 3 drafts cut where the product of their
//! confidences falls under 0.35), on both Sparks. The reply must equal the reference's ids and the rounds must be
//! its rounds: their count, drafted and accepted tokens, each round's drafter ("m" or "c") and kept rows (the
//! fixture's stats).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;

const Buf = cuda.DeviceBuffer;
const Ptr = cuda.abi.DevicePtr;
const M = glm.model;
const T = glm.mtp;

const Link = struct {
    comm: *const cuda.comm.Comm,

    fn gather(p: *anyopaque, send: Ptr, recv: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(p));
        try l.comm.allGather(send, recv, floats, .f32, stream.handle);
    }
};

pub fn run(gpu: Gpu, dir: []const u8, model: []const u8, rank: usize, master: []const u8, port: u16, aot_dir: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: glm.config.Why = .{};
    const cfg = glm.config.read(gpa, gpu.io, model, &why) catch |e| {
        std.debug.print("FAIL glm53 config: {t}: {s}\n", .{ e, why.text() });
        return e;
    };
    const lay: glm.config.Layout = .{ .rank = rank, .world = 2, .cfg = &cfg };
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_dir = try std.fs.path.join(gpa, &.{ dir, "out", "fixture" });
    defer gpa.free(fx_dir);
    var fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir);
    defer fx.deinit();
    var stream = try cuda.Stream.init(d, false); // blocking: a synchronous upload from pageable memory lands before its next kernels
    defer stream.deinit();
    var ip: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, master, '.');
    for (&ip) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    var comm = try cuda.comm.Comm.init(@intCast(rank), 2, ip, port, 600_000);
    defer comm.deinit();
    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var kk = try glm.kda.Kernels.load(d);
    defer kk.deinit();
    var mk = try glm.moe.Kernels.load(d);
    defer mk.deinit();
    var tk = try T.Kernels.load(d);
    defer tk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
    var mdl = try M.Model.load(l, cfg.layers);
    defer mdl.deinit();
    var mtp = try T.Mtp.load(l);
    defer mtp.deinit();
    pack.dropCache();
    var link: Link = .{ .comm = &comm };
    const k: M.Kernels = .{ .g = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams }, .kk = &kk, .mk = &mk };
    var e = try glm.engine.Engine.init(gpa, d, lay, k, &tk, &mdl, &mtp, .{ .ctx = &link, .gather = Link.gather }, stream, 8192, .bf16);
    defer e.deinit();
    var nb: [64]u8 = undefined;
    const prompt_b = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_prompt_ids", .{rank}));
    defer gpa.free(prompt_b);
    const want_b = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_reply_ids", .{rank}));
    defer gpa.free(want_b);
    const prompt = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(prompt_b)));
    const want = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(want_b)));
    const t0 = check.now(gpu.io);
    _ = try e.prefill(prompt);
    const t1 = check.now(gpu.io);
    while (e.out.items.len < want.len) _ = try e.round(want.len);
    const t2 = check.now(gpu.io);
    const got = e.out.items[0..want.len];
    const prefill_s = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const decode_s = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
    std.debug.print("RESULT glm53-drafted rank {d}: prompt {d} tokens in {d:.2} s, {d} tokens in {d} rounds ({s}, keeps {any}), drafted {d} accepted {d}, {d:.2} s ({d:.1} tok/s, host sampling)\n", .{ rank, prompt.len, prefill_s, got.len, e.rounds, e.arms.items, e.keeps.items, e.drafted, e.accepted, decode_s, @as(f64, @floatFromInt(got.len - 1)) / decode_s });
    var bad: usize = 0;
    if (!std.mem.eql(u32, got, want)) {
        const at = std.mem.indexOfDiff(u32, got, want).?;
        std.debug.print("FAIL glm53-drafted rank {d}: token {d} is {d}, the reference's {d}\n", .{ rank, at, got[at], want[at] });
        bad += 1;
    }
    const stats = fx.parsed.value.object.get("params").?.object.get("stats").?.object;
    const want_arms = stats.get("drafters").?.string;
    if (stats.get("rounds").?.integer != e.rounds or stats.get("drafted").?.integer != e.drafted or stats.get("accepted").?.integer != e.accepted or !std.mem.eql(u8, want_arms, e.arms.items)) {
        std.debug.print("FAIL glm53-drafted rank {d}: rounds {d} drafted {d} accepted {d} drafters {s}; the reference's {d} {d} {d} {s}\n", .{ rank, e.rounds, e.drafted, e.accepted, e.arms.items, stats.get("rounds").?.integer, stats.get("drafted").?.integer, stats.get("accepted").?.integer, want_arms });
        bad += 1;
    }
    try check.expect(bad == 0, "glm53-drafted rank {d}: the reply or its rounds differ", .{rank});
    check.pass("BITEXACT glm53-drafted rank {d}: {d} tokens and the reference's {d} rounds ({s})", .{ rank, got.len, e.rounds, e.arms.items });
}
