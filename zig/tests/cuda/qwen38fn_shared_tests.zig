//! `qwen38fn-shared <kernel-set> <model-dir>`: Qwen3.8-Flash-Next's shared rounds (multi.py `round` and
//! `_draft_all`) keep every stream's bits. Four streams with different prompts decode greedily with MTP drafts, first
//! each alone on a new sequence (the path qwen38fn-mtp holds equal to 0.6.6's), then all four in shared rounds on
//! the same sequences reset (state.py `reset`: old attention rows left in place): one forward a round over every
//! window, one head call a depth for every stream's drafts, and the last stream's prompt chunks run between the
//! others' rounds. Each round's logits rows, each stream's drafts and their probabilities, and the decoded tokens must
//! be the same bytes both ways.

const std = @import("std");
const cuda = @import("cuda");
const qfn = @import("qwen38fn");
const check = @import("check.zig");
const Gpu = check.Gpu;
const F = qfn.forward;
const P = qfn.prompt;
const M = qfn.mtp;

const streams = 4;
const tokens = 40;
const capacity = 4096;
const rows = 16;
const prompt_lens = [streams]usize{ 37, 58, 5, 133 };
/// Prompt chunk rows: small, so the last prompt takes five chunks.
const chunk = 32;

/// One stream's prompt: token ids spread over the vocabulary, different for each stream.
fn promptOf(s: usize, out: []i32) void {
    for (out, 0..) |*x, i| x.* = @intCast((7 * i + 1013 * s + 11) % 150000 + 1000);
}

/// What one stream's decode produced: each round's logits rows, each draft call's drafts and probabilities, tokens.
const Record = struct {
    logits: std.ArrayList([]u8) = .empty,
    drafts: std.ArrayList([]u32) = .empty,
    probs: std.ArrayList([]f64) = .empty,
    out: std.ArrayList(u32) = .empty,

    fn deinit(r: *Record, gpa: std.mem.Allocator) void {
        for (r.logits.items) |x| gpa.free(x);
        for (r.drafts.items) |x| gpa.free(x);
        for (r.probs.items) |x| gpa.free(x);
        r.logits.deinit(gpa);
        r.drafts.deinit(gpa);
        r.probs.deinit(gpa);
        r.out.deinit(gpa);
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

/// A stream mid-decode: its sequence, the prompt's last row, its pending drafts.
const Live = struct {
    st: F.Stream,
    last: cuda.DeviceBuffer,
    drafts: [M.depth]u32 = undefined,
    probs: [M.depth]f64 = undefined,
    nd: usize = 0,
    kept: [M.depth + 1]i32 = undefined,
    keep: usize = 0,
    a0: usize = 0,
    rec: Record = .{},
    ids: [prompt_lens[streams - 1]]i32 = undefined,
    len: usize = 0,
    start: usize = 0, // the next prompt chunk's first row

    fn done(l: *const Live) bool {
        return l.rec.out.items.len >= tokens;
    }

    /// The drafts multi.py `_draft_all` allows a stream: its room left after the kept tokens.
    fn room(l: *const Live) usize {
        return @min(M.depth, tokens - l.rec.out.items.len);
    }
};

const Env = struct {
    gpu: Gpu,
    m: *F.Model,
    head: *M.Head,
    fw: *F.Forward,
    pr: *P.Prompt,
    b: *F.Buffers,
    mb: *F.Buffers,
    pb: *P.Buffers,
    sc: *M.Scratch,
    scp: *M.Scratch,
    vh: []u16,
    stream: cuda.Stream,

    /// Stream `s`'s sequence: new, or `l`'s reset; its prompt ready for its chunks.
    fn begin(e: *Env, l: *Live, s: usize, fresh: bool) !void {
        if (fresh) {
            l.* = .{ .st = try F.Stream.init(e.gpu.gpa, e.gpu.d, e.m, capacity, rows), .last = try cuda.DeviceBuffer.alloc(e.gpu.d, e.m.cfg.wide() * 2) };
        } else {
            try l.st.reset(e.m, e.stream);
            l.nd = 0;
            l.keep = 0;
        }
        l.len = prompt_lens[s];
        l.start = 0;
        promptOf(s, l.ids[0..l.len]);
    }

    /// The prompt's next chunk; true once the prompt ended, its first token taken.
    fn chunkStep(e: *Env, l: *Live) !bool {
        l.start = try M.prefillChunk(e.head, e.pr, &l.st, e.pb, e.scp, l.ids[0..l.len], l.start, l.last, e.stream);
        if (l.start < l.len) return false;
        try e.stream.synchronize();
        const v = e.m.cfg.vocab;
        try e.pb.logits.download(0, std.mem.sliceAsBytes(e.vh[0..v]));
        try l.rec.out.append(e.gpu.gpa, argmax(e.vh[0..v]));
        return true;
    }

    fn free(l: *Live, gpa: std.mem.Allocator) void {
        l.st.deinit();
        l.last.free();
        l.rec.deinit(gpa);
    }

    fn prefillAll(e: *Env, l: *Live) !void {
        while (!try e.chunkStep(l)) {}
    }

    fn noteDrafts(e: *Env, l: *Live) !void {
        const gpa = e.gpu.gpa;
        try l.rec.drafts.append(gpa, try gpa.dupe(u32, l.drafts[0..l.nd]));
        try l.rec.probs.append(gpa, try gpa.dupe(f64, l.probs[0..l.nd]));
    }

    /// One round over `ls` (shared when more than one): their windows in one forward, each stream's logits rows
    /// recorded, the kept rows committed.
    fn round(e: *Env, ls: []const *Live) !void {
        const gpa = e.gpu.gpa;
        const v = e.m.cfg.vocab;
        var segs: [streams]F.Seg = undefined;
        var ids: [streams * (M.depth + 1)]i32 = undefined;
        var R: usize = 0;
        for (ls, 0..) |l, i| {
            const n = 1 + l.nd;
            ids[R] = @intCast(l.rec.out.items[l.rec.out.items.len - 1]);
            for (l.drafts[0..l.nd], 1..) |t, k| ids[R + k] = @intCast(t);
            segs[i] = .{ .st = &l.st, .a0 = R, .n = n };
            l.a0 = R;
            R += n;
        }
        try e.fw.stageSegs(segs[0..ls.len], e.b, ids[0..R], e.stream);
        try e.fw.computeSegs(segs[0..ls.len], e.b, .{}, e.stream);
        try e.stream.synchronize();
        try e.b.logits.download(0, std.mem.sliceAsBytes(e.vh[0 .. R * v]));
        for (ls, segs[0..ls.len]) |l, sg| {
            try l.rec.logits.append(gpa, try gpa.dupe(u8, std.mem.sliceAsBytes(e.vh[sg.a0 * v ..][0 .. sg.n * v])));
            var keep: usize = 1;
            var sampled: [M.depth + 1]u32 = undefined;
            for (0..sg.n) |r| sampled[r] = argmax(e.vh[(sg.a0 + r) * v ..][0..v]);
            for (l.drafts[0..l.nd], 0..) |dd, i| {
                if (sampled[i] != dd) break;
                keep += 1;
            }
            keep = @min(keep, tokens - l.rec.out.items.len);
            try e.fw.commit(&l.st, sg.n, keep, e.stream);
            try l.rec.out.appendSlice(gpa, sampled[0..keep]);
            for (sampled[0..keep], 0..) |t, i| l.kept[i] = @intCast(t);
            l.keep = keep;
            l.nd = 0;
        }
    }

    /// `_draft_all` over `ls`: each absorbs its kept rows (or, with `first`, its prompt's last row and first token).
    fn draft(e: *Env, ls: []const *Live, first: bool) !void {
        const wide = e.m.cfg.wide();
        var reqs: [streams]M.Head.Request = undefined;
        var firsts: [streams][1]i32 = undefined;
        for (ls, 0..) |l, i| {
            if (first) firsts[i] = .{@intCast(l.rec.out.items[0])};
            reqs[i] = .{
                .st = &l.st,
                .streams = if (first) l.last.ptr else e.b.streams.ptr + l.a0 * wide * 2,
                .next = if (first) &firsts[i] else l.kept[0..l.keep],
                .count = l.room(),
                .out = &l.drafts,
                .probs = &l.probs,
            };
        }
        try e.head.draftAll(e.fw, reqs[0..ls.len], e.mb, e.sc, e.stream);
        for (ls, reqs[0..ls.len]) |l, q| {
            l.nd = q.n;
            try e.noteDrafts(l);
        }
    }

    /// One round over the streams of `ls` still decoding, then their drafts; false when none is.
    fn step(e: *Env, ls: []const *Live) !bool {
        var live: [streams]*Live = undefined;
        var n: usize = 0;
        for (ls) |l| if (!l.done()) {
            live[n] = l;
            n += 1;
        };
        if (n == 0) return false;
        try e.round(live[0..n]);
        var more: [streams]*Live = undefined;
        var k: usize = 0;
        for (live[0..n]) |l| if (!l.done()) {
            more[k] = l;
            k += 1;
        };
        if (k > 0) try e.draft(more[0..k], false);
        return true;
    }
};

fn compare(what: []const u8, s: usize, alone: *const Record, shared_: *const Record, failures: *usize, equal: *usize) void {
    const pairs = .{ .{ "logits", alone.logits.items, shared_.logits.items }, .{ "drafts", alone.drafts.items, shared_.drafts.items }, .{ "probs", alone.probs.items, shared_.probs.items } };
    inline for (pairs) |p| {
        if (p[1].len != p[2].len) {
            std.debug.print("FAIL qwen38fn-shared {s} stream {d}: {d} {s} records alone, {d} shared\n", .{ what, s, p[1].len, p[0], p[2].len });
            failures.* += 1;
        }
        var first_diff: ?usize = null;
        for (p[1][0..@min(p[1].len, p[2].len)], p[2][0..@min(p[1].len, p[2].len)], 0..) |x, y, i| {
            if (std.mem.eql(u8, std.mem.sliceAsBytes(x), std.mem.sliceAsBytes(y))) {
                equal.* += 1;
            } else {
                if (first_diff == null) first_diff = i;
                failures.* += 1;
            }
        }
        if (first_diff) |i| std.debug.print("FAIL qwen38fn-shared {s} stream {d}: {s} differ from record {d}\n", .{ what, s, p[0], i });
    }
    if (!std.mem.eql(u32, alone.out.items, shared_.out.items)) {
        std.debug.print("FAIL qwen38fn-shared {s} stream {d}: tokens differ\n", .{ what, s });
        failures.* += 1;
    } else equal.* += 1;
}

pub fn shared(gpu: Gpu, kernels: []const u8, model: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: qfn.config.Why = .{};
    const cfg = try qfn.config.read(gpa, gpu.io, model, &why);
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, kernels);
    defer set.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const t0 = check.now(gpu.io);
    const m = try F.Model.load(gpa, gpu.io, d, model, cfg, cfg.layers, true);
    defer m.deinit();
    const head = try M.Head.load(gpa, d, m, streams, stream);
    defer head.deinit();
    check.pass("qwen38fn: {d} layers and the MTP head loaded in {d:.1} s", .{ cfg.layers, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9 });

    const shared_rows = streams * rows;
    var b = try F.Buffers.init(gpa, d, m, shared_rows, capacity);
    defer b.deinit();
    var mb = try F.Buffers.init(gpa, d, m, shared_rows, capacity);
    defer mb.deinit();
    var pb = try P.Buffers.init(gpa, d, m, chunk, capacity);
    defer pb.deinit();
    var sc = try M.Scratch.init(d, &m.cfg, shared_rows);
    defer sc.deinit();
    var scp = try M.Scratch.init(d, &m.cfg, chunk);
    defer scp.deinit();
    var fw = F.Forward.init(d, m, &set);
    var pr = try P.Prompt.init(&fw, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer pr.deinit();
    const vh = try gpa.alloc(u16, cfg.vocab * shared_rows);
    defer gpa.free(vh);
    var e: Env = .{ .gpu = gpu, .m = m, .head = head, .fw = &fw, .pr = &pr, .b = &b, .mb = &mb, .pb = &pb, .sc = &sc, .scp = &scp, .vh = vh, .stream = stream };

    // each stream alone, on a new sequence
    var ls: [streams]Live = undefined;
    var made: usize = 0;
    defer for (ls[0..made]) |*l| Env.free(l, gpa);
    var alone: [streams]Record = @splat(.{});
    defer for (&alone) |*r| r.deinit(gpa);
    for (&ls, 0..) |*l, s| {
        try e.begin(l, s, true);
        made += 1;
        try e.prefillAll(l);
        try e.draft(&.{l}, true);
        while (try e.step(&.{l})) {}
        alone[s] = l.rec;
        l.rec = .{};
    }
    var failures: usize = 0;
    var equal: usize = 0;
    var rounds: usize = 0;
    for (&alone) |*x| rounds += x.logits.items.len;
    var ptrs: [streams]*Live = undefined;
    // each alone again on new sequences: the same bytes whatever ran before
    for (0..streams) |s| {
        var l: Live = undefined;
        try e.begin(&l, s, true);
        defer Env.free(&l, gpa);
        try e.prefillAll(&l);
        try e.draft(&.{&l}, true);
        while (try e.step(&.{&l})) {}
        compare("alone again", s, &alone[s], &l.rec, &failures, &equal);
    }
    // new sequences, the last prompt's chunks between the others' rounds
    {
        var fresh: [streams]Live = undefined;
        var fm: usize = 0;
        defer for (fresh[0..fm]) |*l| Env.free(l, gpa);
        for (&fresh, &ptrs, 0..) |*l, *p, s| {
            try e.begin(l, s, true);
            fm += 1;
            p.* = l;
        }
        for (ptrs[0 .. streams - 1]) |l| try e.prefillAll(l);
        try e.draft(ptrs[0 .. streams - 1], true);
        while (!try e.chunkStep(ptrs[streams - 1])) _ = try e.step(ptrs[0 .. streams - 1]);
        try e.draft(ptrs[streams - 1 ..], true);
        while (try e.step(&ptrs)) {}
        for (&alone, &fresh, 0..) |*x, *y, s| compare("new sequences, prompt chunks between rounds", s, x, &y.rec, &failures, &equal);
    }
    for (&ls, &ptrs) |*l, *p| p.* = l;
    // all four together on the same sequences reset, every prompt first
    for (&ls, 0..) |*l, s| {
        try e.begin(l, s, false);
        try e.prefillAll(l);
    }
    try e.draft(&ptrs, true);
    while (try e.step(&ptrs)) {}
    for (&alone, &ls, 0..) |*x, *y, s| {
        compare("reset, shared", s, x, &y.rec, &failures, &equal);
        y.rec.deinit(gpa);
        y.rec = .{};
    }
    // again, the last prompt's chunks between the others' rounds
    for (&ls, 0..) |*l, s| try e.begin(l, s, false);
    for (ptrs[0 .. streams - 1]) |l| try e.prefillAll(l);
    try e.draft(ptrs[0 .. streams - 1], true);
    var interleaved: usize = 0;
    while (!try e.chunkStep(ptrs[streams - 1])) interleaved += @intFromBool(try e.step(ptrs[0 .. streams - 1]));
    try e.draft(ptrs[streams - 1 ..], true);
    while (try e.step(&ptrs)) {}
    for (&alone, &ls, 0..) |*x, *y, s| compare("reset, shared, prompt chunks between rounds", s, x, &y.rec, &failures, &equal);

    try check.expect(failures == 0, "qwen38fn-shared: {d} records differ ({d} equal)", .{ failures, equal });
    check.pass("BITEXACT qwen38fn-shared: {d} streams x {d} tokens, {d} rounds' logits, drafts and tokens equal alone and shared on reset sequences, {d} rounds between prompt chunks ({d} records)", .{ streams, tokens, rounds, interleaved, equal });
}
