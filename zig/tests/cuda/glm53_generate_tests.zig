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

/// exact_sampling.py MARGIN: candidates past top-k a rank reads, so ties resolve by id.
const margin = 8;
/// decode.py `sample_rows` with probs: a draft's confidence takes the 20 best and the margin from each rank.
const draft_k = 20 + margin;
const cap = 256; // drafter_choice.py: backlog rows the MTP head may owe

const Rank = struct {
    gpu: Gpu,
    comm: *const cuda.comm.Comm,
    stream: cuda.Stream,
    rank: usize,
    nv: usize,
    host: []u16,
    cand: Buf, // [mine: draft_k (value, id)][gathered: 2 x that]

    fn gather(p: *anyopaque, part: Ptr, gath: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const c: *Rank = @ptrCast(@alignCast(p));
        try c.comm.allGather(part, gath, floats, .f32, stream.handle);
    }

    const Cand = struct { v: f32, id: u32 };

    fn better(_: void, a: Cand, b: Cand) bool {
        return a.v > b.v or (a.v == b.v and a.id < b.id);
    }

    /// decode.py `sample_rows`, greedy, of the logits row at `row` (this rank's share): the token every rank draws
    /// and, with `prob`, its probability among the gathered candidates at temperature 1 (`_probability`).
    fn sample(c: *Rank, logits: Buf, row: usize, k: usize, prob: ?*f64) !u32 {
        try c.stream.synchronize();
        try logits.download(row * c.nv * 2, std.mem.sliceAsBytes(c.host));
        var best: [draft_k]Cand = undefined;
        var n: usize = 0;
        for (c.host, 0..) |x, j| {
            const v: f32 = @bitCast(@as(u32, x) << 16);
            const cand: Cand = .{ .v = v, .id = @intCast(c.rank * c.nv + j) };
            if (n < k) {
                best[n] = cand;
                n += 1;
                std.mem.sort(Cand, best[0..n], {}, better);
            } else if (better({}, cand, best[k - 1])) {
                best[k - 1] = cand;
                std.mem.sort(Cand, best[0..k], {}, better);
            }
        }
        var mine: [2 * draft_k]f32 = undefined;
        for (best[0..k], 0..) |b, i| {
            mine[i] = b.v;
            mine[k + i] = @bitCast(b.id);
        }
        try c.cand.upload(0, std.mem.sliceAsBytes(mine[0 .. 2 * k]));
        try c.comm.allGather(c.cand.ptr, c.cand.ptr + 2 * draft_k * 4, 2 * k, .f32, c.stream.handle);
        try c.stream.synchronize();
        var both: [4 * draft_k]f32 = undefined;
        try c.cand.download(2 * draft_k * 4, std.mem.sliceAsBytes(both[0 .. 4 * k]));
        var all: [2 * draft_k]Cand = undefined;
        for (0..2) |r| for (0..k) |i| {
            all[r * k + i] = .{ .v = both[r * 2 * k + i], .id = @bitCast(both[r * 2 * k + k + i]) };
        };
        std.mem.sort(Cand, all[0 .. 2 * k], {}, better);
        if (prob) |p| {
            var z: f64 = 0;
            const top: f64 = all[0].v;
            for (all[0 .. 2 * k]) |a| z += @exp(@as(f64, a.v) - top);
            p.* = 1.0 / z; // the chosen token is the first: exp(0) / z
        }
        return all[0].id;
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
    const D = cfg.hidden;
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
    const rows = 16;
    const prompt_rows = 2048;
    const capacity = 8192;
    var st = try M.State.init(gpa, d, lay, cfg.layers, capacity, rows, prompt_rows);
    defer st.deinit();
    var w = try M.Work.init(d, lay, rows, prompt_rows, capacity);
    defer w.deinit();
    var mcache = try glm.dsa.Cache.init(d, &cfg, capacity, glm.dsa.indexRing(capacity, prompt_rows));
    defer mcache.deinit();
    var mb = try T.Buffers.init(d, lay, rows, false);
    defer mb.deinit();
    var pmb = try T.Buffers.init(d, lay, prompt_rows, true);
    defer pmb.deinit();
    var mpos = try Buf.alloc(d, 16);
    defer mpos.free();
    var backlog = try Buf.alloc(d, cap * D * 2); // the MTP head's owed input rows
    defer backlog.free();
    var last_hidden = try Buf.alloc(d, D * 2);
    defer last_hidden.free();
    const k: M.Kernels = .{ .g = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams }, .kk = &kk, .mk = &mk };
    var rk: Rank = .{ .gpu = gpu, .comm = &comm, .stream = stream, .rank = rank, .nv = lay.size(.vocab), .host = try gpa.alloc(u16, lay.size(.vocab)), .cand = try Buf.alloc(d, 6 * draft_k * 4) };
    defer gpa.free(rk.host);
    defer rk.cand.free();
    const g: M.Gather = .{ .ctx = &rk, .run = Rank.gather };
    const tg: T.Gather = .{ .ctx = &rk, .run = Rank.gather };
    var nb: [64]u8 = undefined;
    const prompt_b = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_prompt_ids", .{rank}));
    defer gpa.free(prompt_b);
    const want_b = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_reply_ids", .{rank}));
    defer gpa.free(want_b);
    const prompt = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(prompt_b)));
    const want = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(want_b)));
    const count = want.len;
    var mtp_len: usize = 0;
    var mtp_drafted: usize = 0;

    // the prompt: each chunk, then its rows absorbed into the MTP head (the prompt's next tokens; the last row waits)
    const t0 = check.now(gpu.io);
    var start: usize = 0;
    while (start < prompt.len) : (start += prompt_rows) {
        const R = @min(prompt_rows, prompt.len - start);
        try w.pb.ids.upload(0, std.mem.sliceAsBytes(prompt[start .. start + R]));
        try M.forward(k, lay, &mdl, &st, &w, R, true, g, null, stream);
        try last_hidden.copyFrom(0, w.fnormed.ptr + (R - 1) * D * 2, D * 2, stream.handle);
        const nxt = prompt[start + 1 .. @min(start + R + 1, prompt.len)];
        if (nxt.len > 0) {
            try T.run(k.g, &tk, &mk, lay, &mtp, &pmb, .{ .cache = &mcache, .ds = &w.pds, .ms = &w.pms, .pos_dev = mpos }, mdl.embed.ptr, nxt, w.fnormed.ptr, mtp_len, tg, stream);
            mtp_len += nxt.len;
        }
        try M.commit(k, lay, &st, R, R, true, &w, stream);
    }
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try out.append(gpa, try rk.sample(w.logits, 0, 1, null));
    const t1 = check.now(gpu.io);

    // the rounds (drafter_choice.py auto_decode, greedy, no DFlash2)
    const ctx = try gpa.alloc(u32, prompt.len + 1);
    @memcpy(ctx[0..prompt.len], prompt);
    ctx[prompt.len] = out.items[0];
    var copies = try glm.copy_drafts.CopyDrafts.init(gpa, .{}, ctx);
    gpa.free(ctx);
    defer copies.deinit();
    try backlog.copyFrom(0, last_hidden.ptr, D * 2, stream.handle);
    var m_next: std.ArrayList(u32) = .empty;
    defer m_next.deinit(gpa);
    try m_next.append(gpa, out.items[0]);
    var arms: std.ArrayList(u8) = .empty;
    defer arms.deinit(gpa);
    var keeps: std.ArrayList(usize) = .empty;
    defer keeps.deinit(gpa);
    var rounds: usize = 0;
    var drafted: usize = 0;
    var accepted: usize = 0;
    var drafts_buf: [32]u32 = undefined;
    var window: [32]u32 = undefined;
    while (out.items.len < count) {
        const room = count - out.items.len;
        var nd = copies.propose(count - out.items.len - 1, &drafts_buf);
        const arm: u8 = if (nd > 0) 'c' else 'm';
        if (arm == 'm') {
            // decode.py `draft`: absorb the backlog (after rewinding the last chain), then a chain of up to 3
            if (mtp_drafted > 0) {
                mtp_len -= mtp_drafted;
                mtp_drafted = 0;
            }
            var s0: usize = 0;
            while (s0 < m_next.items.len) : (s0 += rows) {
                const part = m_next.items[s0..@min(s0 + rows, m_next.items.len)];
                try T.run(k.g, &tk, &mk, lay, &mtp, &mb, .{ .cache = &mcache, .ds = &w.ds, .ms = &w.ms, .pos_dev = mpos }, mdl.embed.ptr, part, backlog.ptr + s0 * D * 2, mtp_len, tg, stream);
                mtp_len += part.len;
            }
            m_next.clearRetainingCapacity();
            const depth = @max(1, @min(glm.copy_drafts.depth_most, room));
            var chain: f64 = 1.0;
            nd = 0;
            for (0..depth) |j| {
                var p: f64 = 0;
                const dtok = try rk.sample(mb.logits, 0, draft_k, &p);
                if (j > 0 and chain * p < glm.copy_drafts.confidence) break;
                drafts_buf[nd] = dtok;
                nd += 1;
                chain *= p;
                if (chain < glm.copy_drafts.confidence) break;
                if (j + 1 < depth) {
                    try T.run(k.g, &tk, &mk, lay, &mtp, &mb, .{ .cache = &mcache, .ds = &w.ds, .ms = &w.ms, .pos_dev = mpos }, mdl.embed.ptr, &.{dtok}, mb.fnormed.ptr, mtp_len, tg, stream);
                    mtp_len += 1;
                    mtp_drafted += 1;
                }
            }
        }
        // verify [pending | drafts]
        window[0] = out.items[out.items.len - 1];
        @memcpy(window[1 .. 1 + nd], drafts_buf[0..nd]);
        const R = 1 + nd;
        try w.b.ids.upload(0, std.mem.sliceAsBytes(window[0..R]));
        try M.forward(k, lay, &mdl, &st, &w, R, false, g, null, stream);
        var sampled: [32]u32 = undefined;
        for (0..R) |r| sampled[r] = try rk.sample(w.logits, r, 1, null);
        var keep: usize = 1;
        for (drafts_buf[0..nd], 0..) |dt, i| {
            if (sampled[i] != dt or std.mem.indexOfScalar(u32, cfg.eos[0..cfg.eos_n], sampled[i]) != null) break;
            keep += 1;
        }
        try M.commit(k, lay, &st, R, keep, false, &w, stream);
        // the kept rows join the MTP head's backlog (a full one absorbed first)
        if (m_next.items.len + keep > cap) return error.BacklogFull;
        try backlog.copyFrom(m_next.items.len * D * 2, w.fnormed.ptr, keep * D * 2, stream.handle);
        try m_next.appendSlice(gpa, sampled[0..keep]);
        rounds += 1;
        drafted += nd;
        accepted += keep - 1;
        try arms.append(gpa, arm);
        try keeps.append(gpa, keep);
        try copies.extend(sampled[0..keep]);
        try out.appendSlice(gpa, sampled[0..keep]);
    }
    const t2 = check.now(gpu.io);
    const got = out.items[0..@min(out.items.len, count)];
    const prefill_s = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const decode_s = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
    std.debug.print("RESULT glm53-drafted rank {d}: prompt {d} tokens in {d:.2} s, {d} tokens in {d} rounds ({s}, keeps {any}), drafted {d} accepted {d}, {d:.2} s ({d:.1} tok/s, host sampling)\n", .{ rank, prompt.len, prefill_s, got.len, rounds, arms.items, keeps.items, drafted, accepted, decode_s, @as(f64, @floatFromInt(got.len - 1)) / decode_s });
    var bad: usize = 0;
    if (!std.mem.eql(u32, got, want)) {
        const at = std.mem.indexOfDiff(u32, got, want).?;
        std.debug.print("FAIL glm53-drafted rank {d}: token {d} is {d}, the reference's {d}\n", .{ rank, at, got[at], want[at] });
        bad += 1;
    }
    const stats = fx.parsed.value.object.get("params").?.object.get("stats").?.object;
    const want_arms = stats.get("drafters").?.string;
    if (stats.get("rounds").?.integer != rounds or stats.get("drafted").?.integer != drafted or stats.get("accepted").?.integer != accepted or !std.mem.eql(u8, want_arms, arms.items)) {
        std.debug.print("FAIL glm53-drafted rank {d}: rounds {d} drafted {d} accepted {d} drafters {s}; the reference's {d} {d} {d} {s}\n", .{ rank, rounds, drafted, accepted, arms.items, stats.get("rounds").?.integer, stats.get("drafted").?.integer, stats.get("accepted").?.integer, want_arms });
        bad += 1;
    }
    try check.expect(bad == 0, "glm53-drafted rank {d}: the reply or its rounds differ", .{rank});
    check.pass("BITEXACT glm53-drafted rank {d}: {d} tokens and the reference's {d} rounds ({s})", .{ rank, got.len, rounds, arms.items });
}
