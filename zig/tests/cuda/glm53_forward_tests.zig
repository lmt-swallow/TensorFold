//! `glm53-forward <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir> [layers]`: GLM-5.3-Flash's whole forward
//! on one rank of the pair (run it on both Sparks at once, rank 0 on the master), the ranks' partials all-gathered
//! over NCCL, against the Mia fork's forward fixture of one served request (tools/zig/glm53/b1/forward_capture.sh with
//! TF_GLM_HC_SPLIT=0: its prompt chunks' layers recorded too). `fwd-dir` is that rank's capture (out/fixture);
//! `aot-dir` a kernel set covering its launches (the capture's own, merged with the layer fixtures').
//!
//! The request's forwards run in order from the fixture's ids at its positions: prompt chunks, then decode windows,
//! each window committing the rows the next one starts after. After every layer the streams are compared (a prompt
//! chunk's last 16 rows, a decode window's every row), then a decode window's head logits (this rank's share).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;

const Buf = cuda.DeviceBuffer;
const Ptr = cuda.abi.DevicePtr;
const M = glm.model;

const Ctx = struct {
    gpu: Gpu,
    fx: *const fixture.Fixture,
    stream: cuda.Stream,
    comm: *const cuda.comm.Comm,
    rank: usize,
    f: usize = 0,
    prefill: bool = false,
    tail: usize,
    D: usize,
    S: usize,
    failures: usize = 0,
    passes: usize = 0,
    first_bad: ?[2]usize = null,

    fn gather(p: *anyopaque, part: Ptr, gath: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const c: *Ctx = @ptrCast(@alignCast(p));
        try c.comm.allGather(part, gath, floats, .f32, stream.handle);
    }

    /// `name`'s fixture bytes against the device rows at `ptr` (the array's length).
    fn same(c: *Ctx, name: []const u8, ptr: Ptr) !bool {
        try c.stream.synchronize();
        const want = c.fx.bytes(name) catch return error.MissingArray;
        defer c.gpu.gpa.free(want);
        const got = try c.gpu.gpa.alloc(u8, want.len);
        defer c.gpu.gpa.free(got);
        const view: Buf = .{ .d = c.gpu.d, .ptr = ptr, .len = want.len };
        try view.download(0, got);
        if (std.mem.eql(u8, got, want)) {
            c.passes += 1;
            return true;
        }
        var diff: usize = 0;
        for (got, want) |a, b| diff += @intFromBool(a != b);
        std.debug.print("FAIL glm53 {s}: {d} of {d} bytes differ, the first at byte {d}\n", .{ name, diff, want.len, std.mem.indexOfDiff(u8, got, want).? });
        c.failures += 1;
        return false;
    }

    fn layer(p: *anyopaque, index: usize, x: Ptr, R: usize) anyerror!void {
        const c: *Ctx = @ptrCast(@alignCast(p));
        var nb: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "r{d}_f{d}_L{d}", .{ c.rank, c.f, index });
        const lo = if (c.prefill) R -| c.tail else 0;
        if (!try c.same(name, x + lo * c.S * c.D * 2) and c.first_bad == null) c.first_bad = .{ c.f, index };
    }
};

pub fn run(gpu: Gpu, dir: []const u8, model: []const u8, rank: usize, master: []const u8, port: u16, aot_dir: []const u8, only_layers: ?usize) !void {
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
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var ip: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, master, '.');
    for (&ip) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    var comm = try cuda.comm.Comm.init(@intCast(rank), 2, ip, port, 600_000);
    defer comm.deinit();
    check.pass("glm53-forward rank {d}: NCCL up with the peer", .{rank});

    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var kk = try glm.kda.Kernels.load(d);
    defer kk.deinit();
    var mk = try glm.moe.Kernels.load(d);
    defer mk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    const n_layers = only_layers orelse cfg.layers;
    const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
    const t0 = check.now(gpu.io);
    var mdl = try M.Model.load(l, n_layers);
    defer mdl.deinit();
    check.pass("glm53-forward rank {d}: {d} layers loaded in {d:.1} s", .{ rank, n_layers, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9 });
    const rows = 16;
    const prompt_rows = 2048;
    const capacity = 8192;
    var st = try M.State.init(gpa, d, lay, n_layers, capacity, rows, prompt_rows);
    defer st.deinit();
    var w = try M.Work.init(d, lay, rows, prompt_rows, capacity);
    defer w.deinit();
    const k: M.Kernels = .{ .g = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams }, .kk = &kk, .mk = &mk };
    var cx: Ctx = .{ .gpu = gpu, .fx = &fx, .stream = stream, .comm = &comm, .rank = rank, .tail = @intCast(try fx.int("tail_rows")), .D = cfg.hidden, .S = cfg.streams };
    const g: M.Gather = .{ .ctx = &cx, .run = Ctx.gather };
    const hook: M.Hook = .{ .ctx = &cx, .run = Ctx.layer };

    const recorded: usize = @intCast(try fx.int("recorded"));
    var nb: [64]u8 = undefined;
    for (0..recorded) |f| {
        const ids = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_f{d}_ids", .{ rank, f }));
        defer gpa.free(ids);
        const posb = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_f{d}_pos", .{ rank, f }));
        defer gpa.free(posb);
        const pos: usize = @intCast(std.mem.bytesToValue(i64, posb[0..8]));
        const R = ids.len / 4;
        const prefill = R > rows;
        try check.expect(pos == st.pos, "glm53-forward f{d}: the fixture's window starts at {d}, the state is at {d}", .{ f, pos, st.pos });
        const b = if (prefill) &w.pb else &w.b;
        try b.ids.upload(0, ids);
        cx.f = f;
        cx.prefill = prefill;
        const before = cx.failures;
        try M.forward(k, lay, &mdl, &st, &w, R, prefill, g, hook, stream);
        if (!prefill and n_layers == cfg.layers) _ = try cx.same(try std.fmt.bufPrint(&nb, "r{d}_f{d}_logits", .{ rank, f }), w.logits.ptr);
        // the rows kept: up to where the next recorded window starts (the last one keeps all)
        var keep = R;
        if (f + 1 < recorded) {
            const nxt = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_f{d}_pos", .{ rank, f + 1 }));
            defer gpa.free(nxt);
            keep = @as(usize, @intCast(std.mem.bytesToValue(i64, nxt[0..8]))) - pos;
        }
        try M.commit(k, lay, &st, R, keep, prefill, &w, stream);
        try stream.synchronize();
        if (cx.failures == before) check.pass("BITEXACT glm53 rank {d} forward f{d} ({s} {d} rows at {d}, {d} kept): every layer's streams{s} equal to the reference", .{ rank, f, if (prefill) "prompt" else "decode", R, pos, keep, if (prefill or n_layers != cfg.layers) "" else " and the logits" });
    }
    if (cx.first_bad) |fb| std.debug.print("first difference: forward f{d}, layer {d}\n", .{ fb[0], fb[1] });
    try check.expect(cx.failures == 0, "glm53-forward rank {d}: {d} comparisons differ ({d} equal)", .{ rank, cx.failures, cx.passes });
    check.pass("glm53-forward rank {d}: {d} forwards, {d} comparisons bit-exact", .{ rank, recorded, cx.passes });
}

/// `glm53-generate <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir>`: the fixture's request decoded greedily
/// and serially (one token a round, no drafts: decode.py `serial_decode`, the reference drafted replies must equal):
/// its prompt in 2,048-row chunks, the first token from the last row, then one row a round; every token drawn as
/// decode.py `sample_rows` draws a greedy one (each rank's best of its vocabulary share, the two gathered, the larger
/// value, the lower id on a tie). The reply must equal the reference's ids.
pub fn generate(gpu: Gpu, dir: []const u8, model: []const u8, rank: usize, master: []const u8, port: u16, aot_dir: []const u8) !void {
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
    var stream = try cuda.Stream.init(d, true);
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
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
    var mdl = try M.Model.load(l, cfg.layers);
    defer mdl.deinit();
    const rows = 16;
    const prompt_rows = 2048;
    const capacity = 8192;
    var st = try M.State.init(gpa, d, lay, cfg.layers, capacity, rows, prompt_rows);
    defer st.deinit();
    var w = try M.Work.init(d, lay, rows, prompt_rows, capacity);
    defer w.deinit();
    const k: M.Kernels = .{ .g = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams }, .kk = &kk, .mk = &mk };
    var cx: Ctx = .{ .gpu = gpu, .fx = &fx, .stream = stream, .comm = &comm, .rank = rank, .tail = 16, .D = cfg.hidden, .S = cfg.streams };
    const g: M.Gather = .{ .ctx = &cx, .run = Ctx.gather };
    var nb: [64]u8 = undefined;
    const prompt_b = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_prompt_ids", .{rank}));
    defer gpa.free(prompt_b);
    const want_b = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_reply_ids", .{rank}));
    defer gpa.free(want_b);
    const prompt = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(prompt_b)));
    const want = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(want_b)));
    var cand = try Buf.alloc(d, 32); // [mine: value, id][gathered: rank 0's, rank 1's]
    defer cand.free();
    var got = try gpa.alloc(i32, want.len);
    defer gpa.free(got);
    const nv = lay.size(.vocab);
    const host = try gpa.alloc(u16, nv);
    defer gpa.free(host);
    const t0 = check.now(gpu.io);
    var start: usize = 0;
    while (start < prompt.len) : (start += prompt_rows) {
        const R = @min(prompt_rows, prompt.len - start);
        try w.pb.ids.upload(0, std.mem.sliceAsBytes(prompt[start .. start + R]));
        try M.forward(k, lay, &mdl, &st, &w, R, true, g, null, stream);
        try M.commit(k, lay, &st, R, R, true, &w, stream);
    }
    const t1 = check.now(gpu.io);
    var tok: i32 = 0;
    for (0..want.len) |i| {
        if (i > 0) {
            try w.b.ids.upload(0, std.mem.asBytes(&tok));
            try M.forward(k, lay, &mdl, &st, &w, 1, false, g, null, stream);
            try M.commit(k, lay, &st, 1, 1, false, &w, stream);
        }
        // sample_rows, greedy: this rank's best (value, id) of its share, gathered, the larger value, the lower id
        try stream.synchronize();
        try w.logits.download(0, std.mem.sliceAsBytes(host));
        var bv: f32 = -std.math.inf(f32);
        var bi: usize = 0;
        for (host, 0..) |x, j| {
            const v: f32 = @bitCast(@as(u32, x) << 16);
            if (v > bv) {
                bv = v;
                bi = j;
            }
        }
        const mine = [2]f32{ bv, @bitCast(@as(i32, @intCast(rank * nv + bi))) };
        try cand.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(cand.ptr, cand.ptr + 16, 2, .f32, stream.handle);
        try stream.synchronize();
        var both: [4]f32 = undefined;
        try cand.download(16, std.mem.sliceAsBytes(&both));
        const id0: i32 = @bitCast(both[1]);
        const id1: i32 = @bitCast(both[3]);
        tok = if (both[2] > both[0] or (both[2] == both[0] and id1 < id0)) id1 else id0;
        got[i] = tok;
    }
    const t2 = check.now(gpu.io);
    const prefill_s = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const decode_s = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
    std.debug.print("RESULT glm53-generate rank {d}: prompt {d} tokens in {d:.2} s ({d:.0} tok/s), {d} tokens serially in {d:.2} s ({d:.1} tok/s)\n", .{ rank, prompt.len, prefill_s, @as(f64, @floatFromInt(prompt.len)) / prefill_s, want.len, decode_s, @as(f64, @floatFromInt(want.len - 1)) / decode_s });
    if (!std.mem.eql(i32, got, want)) {
        const at = std.mem.indexOfDiff(i32, got, want).?;
        std.debug.print("FAIL glm53-generate rank {d}: token {d} is {d}, the reference's {d}\n", .{ rank, at, got[at], want[at] });
        return error.TestFailed;
    }
    check.pass("BITEXACT glm53-generate rank {d}: {d} reply tokens equal to the reference's (served with drafts)", .{ rank, want.len });
}
