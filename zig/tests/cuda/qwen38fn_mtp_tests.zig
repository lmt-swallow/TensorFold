//! `qwen38fn-mtp <work-dir> <model-dir>`: Qwen3.8-Flash-Next's MTP drafting (cuda_mtp.zig) against TensorFold 0.6.6's
//! serial drafted decoder (tools/zig/qwen38fn/dump_mtp.py, packed by capture_mtp.sh): the requantized draft head
//! byte for byte, then the same greedy run (the prompt in chunks with the head absorbing each, then rounds of the
//! pending token and up to 6 drafts, kept to the first mismatch) with every head call's streams and logits, every
//! draft and every round compared in order, and the decoded tokens.

const std = @import("std");
const cuda = @import("cuda");
const qfn = @import("qwen38fn");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;
const F = qfn.forward;
const P = qfn.prompt;
const M = qfn.mtp;

const Ctx = struct {
    gpu: Gpu,
    fx: *fixture.Fixture,
    head: *M.Head,
    calls: []std.json.Value,
    call: usize = 0,
    wide: usize,
    failures: usize = 0,
    passes: usize = 0,
    name: [64]u8 = undefined,

    fn same(self: *Ctx, what: []const u8, buf: cuda.DeviceBuffer, len: ?usize) !void {
        const want = try self.fx.bytes(what);
        defer self.gpu.gpa.free(want);
        const n = len orelse want.len;
        const got = try self.gpu.gpa.alloc(u8, n);
        defer self.gpu.gpa.free(got);
        try buf.download(0, got);
        if (n != want.len or !std.mem.eql(u8, got, want)) {
            const at = std.mem.indexOfDiff(u8, got, want[0..@min(n, want.len)]) orelse n;
            std.debug.print("FAIL qwen38fn-mtp {s}: differs from byte {d} of {d}\n", .{ what, at, want.len });
            self.failures += 1;
        } else self.passes += 1;
    }

    fn onCall(ctx: *anyopaque, prompt: bool, rows: usize, streams: cuda.DeviceBuffer) anyerror!void {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        const j = self.call;
        self.call += 1;
        if (j >= self.calls.len) {
            std.debug.print("FAIL qwen38fn-mtp: head call {d} past the reference's {d}\n", .{ j, self.calls.len });
            self.failures += 1;
            return;
        }
        const c = self.calls[j].object;
        const kind = c.get("kind").?.string;
        const want_rows: usize = @intCast(c.get("rows").?.integer);
        if (std.mem.eql(u8, kind, "prompt") != prompt or want_rows != rows) {
            std.debug.print("FAIL qwen38fn-mtp call {d}: {s} of {d} rows, ours {s} of {d}\n", .{ j, kind, want_rows, if (prompt) "prompt" else "decode", rows });
            self.failures += 1;
            return;
        }
        try self.same(try std.fmt.bufPrint(&self.name, "m{d}_streams", .{j}), streams, rows * self.wide * 2);
        if (!prompt) try self.same(try std.fmt.bufPrint(&self.name, "m{d}_logits", .{j}), self.head.logits, null);
    }
};

fn argmax(host: []const u16) u32 {
    var best: usize = 0;
    var top = qfn.weights.bf16ToF32(host[0]);
    for (host[1..], 1..) |x, i| {
        const v = qfn.weights.bf16ToF32(x);
        if (v > top) {
            top = v;
            best = i;
        }
    }
    return @intCast(best);
}

pub fn mtp(gpu: Gpu, work: []const u8, model: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: qfn.config.Why = .{};
    const cfg = try qfn.config.read(gpa, gpu.io, model, &why);
    const aot_dir = try std.fs.path.join(gpa, &.{ work, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_dir = try std.fs.path.join(gpa, &.{ work, "out", "fixture" });
    defer gpa.free(fx_dir);
    var fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir);
    defer fx.deinit();
    const root = fx.parsed.value.object;
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();

    const t0 = check.now(gpu.io);
    const m = try F.Model.load(gpa, gpu.io, d, model, cfg, cfg.layers, true);
    defer m.deinit();
    const head = try M.Head.load(gpa, d, m, 1, stream);
    defer head.deinit();
    check.pass("qwen38fn: {d} layers and the MTP head loaded in {d:.1} s ({d} draft ids)", .{ cfg.layers, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9, head.ids.len });
    var ctx: Ctx = .{ .gpu = gpu, .fx = &fx, .head = head, .calls = root.get("calls").?.array.items, .wide = cfg.wide() };
    try ctx.same("draft_words", head.draft_head.words, null);
    try ctx.same("draft_scales", head.draft_head.scales, null);
    try ctx.same("draft_biases", head.draft_head.biases, null);
    head.probe = .{ .ctx = &ctx, .call = Ctx.onCall };

    const capacity: usize = @intCast(try fx.int("capacity"));
    const rows: usize = @intCast(try fx.int("rows"));
    var st = try F.Stream.init(gpa, d, m, capacity, rows);
    defer st.deinit();
    var b = try F.Buffers.init(gpa, d, m, rows, capacity);
    defer b.deinit();
    var mb = try F.Buffers.init(gpa, d, m, rows, capacity);
    defer mb.deinit();
    var pb = try P.Buffers.init(gpa, d, m, P.chunk_rows, capacity);
    defer pb.deinit();
    var sc = try M.Scratch.init(d, &m.cfg, rows);
    defer sc.deinit();
    var scp = try M.Scratch.init(d, &m.cfg, P.chunk_rows);
    defer scp.deinit();
    var last = try cuda.DeviceBuffer.alloc(d, cfg.wide() * 2);
    defer last.free();
    var fw = F.Forward.init(d, m, &set);
    var pr = try P.Prompt.init(&fw, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer pr.deinit();

    const pb_ids = try fx.bytes("prompt_ids");
    defer gpa.free(pb_ids);
    const prompt = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(pb_ids)));
    try M.prefill(head, &pr, &st, &pb, &scp, prompt, last, null, stream);
    try stream.synchronize();
    const vh = try gpa.alloc(u16, cfg.vocab * rows);
    defer gpa.free(vh);
    try pb.logits.download(0, std.mem.sliceAsBytes(vh[0..cfg.vocab]));
    const first = argmax(vh[0..cfg.vocab]);
    try check.expect(first == @as(u32, @intCast(try fx.int("first"))), "qwen38fn-mtp: first token {d}, 0.6.6's {d}", .{ first, try fx.int("first") });

    // mtp_decode, greedy
    const count: usize = @intCast(try fx.int("tokens"));
    const want_rounds = root.get("rounds").?.array.items;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try out.append(gpa, first);
    var drafts: [M.depth]u32 = undefined;
    var probs: [M.depth]f64 = undefined;
    var nd: usize = 0;
    const pending0 = [_]i32{@intCast(first)};
    nd = try head.draft(&fw, &st, &mb, &sc, last.ptr, &pending0, @min(M.depth, count - out.items.len), &drafts, &probs, stream);
    var round: usize = 0;
    var unabsorbed: ?usize = null;
    var kept_tokens: [M.depth + 1]i32 = undefined;
    while (out.items.len < count) : (round += 1) {
        var ids: [M.depth + 1]i32 = undefined;
        ids[0] = @intCast(out.items[out.items.len - 1]);
        for (drafts[0..nd], 1..) |t, i| ids[i] = @intCast(t);
        const R = 1 + nd;
        if (round < want_rounds.len) {
            const wt = want_rounds[round].object.get("tokens").?.array.items;
            var ok = wt.len == R;
            if (ok) for (wt, ids[0..R]) |x, y| {
                ok = ok and x.integer == y;
            };
            if (!ok) {
                std.debug.print("FAIL qwen38fn-mtp round {d}: window differs from 0.6.6's\n", .{round});
                ctx.failures += 1;
            } else ctx.passes += 1;
        }
        try fw.stage(&st, &b, ids[0..R], stream);
        try fw.compute(&st, &b, R, .{}, stream);
        try stream.synchronize();
        try b.logits.download(0, std.mem.sliceAsBytes(vh[0 .. R * cfg.vocab]));
        var sampled: [M.depth + 1]u32 = undefined;
        for (0..R) |r| sampled[r] = argmax(vh[r * cfg.vocab ..][0..cfg.vocab]);
        var keep: usize = 1;
        for (drafts[0..nd], 0..) |dd, i| {
            if (sampled[i] != dd) break;
            keep += 1;
        }
        try fw.commit(&st, R, keep, stream);
        try out.appendSlice(gpa, sampled[0..keep]);
        for (sampled[0..keep], 0..) |t, i| kept_tokens[i] = @intCast(t);
        unabsorbed = keep;
        if (out.items.len >= count) break;
        const n = @min(M.depth, count - out.items.len);
        nd = 0;
        if (n > 0) {
            nd = try head.draft(&fw, &st, &mb, &sc, b.streams.ptr, kept_tokens[0..keep], n, &drafts, &probs, stream);
            unabsorbed = null;
        }
    }
    if (unabsorbed) |k| try head.absorb(&fw, &st, &mb, &sc, b.streams.ptr, kept_tokens[0..k], stream);
    // the decoded tokens
    const tb = try fx.bytes("tokens");
    defer gpa.free(tb);
    const want_tokens = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(tb)));
    var same_tokens = want_tokens.len == @min(out.items.len, count);
    if (same_tokens) for (want_tokens, out.items[0..want_tokens.len]) |x, y| {
        same_tokens = same_tokens and @as(u32, @intCast(x)) == y;
    };
    try check.expect(same_tokens, "qwen38fn-mtp: decoded tokens differ from 0.6.6's", .{});
    try check.expect(ctx.call == ctx.calls.len, "qwen38fn-mtp: {d} head calls, 0.6.6 made {d}", .{ ctx.call, ctx.calls.len });
    try check.expect(round + 1 >= want_rounds.len, "qwen38fn-mtp: {d} rounds, 0.6.6 ran {d}", .{ round + 1, want_rounds.len });
    try check.expect(ctx.failures == 0, "qwen38fn-mtp: {d} values differ ({d} equal)", .{ ctx.failures, ctx.passes });
    check.pass("BITEXACT qwen38fn-mtp: draft head, {d} head calls, {d} rounds and {d} tokens equal to 0.6.6's ({d} values)", .{ ctx.call, round + 1, want_tokens.len, ctx.passes });
}
