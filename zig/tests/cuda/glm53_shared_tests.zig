//! `glm53-shared <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir> [layers]`: four streams' windows in one
//! forward (cuda_model.zig `forwardSegs`) against each stream's windows alone, on both Sparks (master "local": both
//! ranks on this GPU, the first `layers` layers). The streams' prompts are prefixes of the forward fixture's (some
//! past the dense limit, some within it); every round each stream verifies a window of its pending token and a few
//! forced tokens (widths changing round to round, up to `E.batch_rows` rows in all) and keeps `keeps` of them. Every
//! row's logits and sample must be the same in the shared forward as alone, and alone as on the single-stream path
//! the fixtures check (cuda_model.zig `forward`).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const pair = @import("glm53_pair.zig");
const Gpu = check.Gpu;

const Ptr = cuda.abi.DevicePtr;
const M = glm.model;
const T = glm.mtp;
const E = glm.engine;
const P = glm.pool;

const streams = 4;
const rounds = 6;
/// Each stream's window rows by round (the pending token and forced tokens), at most `E.batch_rows` (16) rows a round in all.
const widths = [rounds][streams]usize{ .{ 1, 1, 1, 1 }, .{ 1, 3, 8, 4 }, .{ 8, 4, 2, 2 }, .{ 2, 11, 1, 2 }, .{ 4, 4, 4, 4 }, .{ 1, 13, 1, 1 } };
/// Each window's kept rows by round (fewer than its rows: the KDA states replay, the rest is rewritten next round).
const keeps = [rounds][streams]usize{ .{ 1, 1, 1, 1 }, .{ 1, 2, 5, 4 }, .{ 3, 1, 2, 1 }, .{ 2, 6, 1, 1 }, .{ 4, 1, 3, 2 }, .{ 1, 5, 1, 1 } };
/// The prompts' lengths: past the dense limit (2,051) and within it.
const lengths = [streams]usize{ 3000, 2100, 1200, 333 };

const Run = struct {
    logits: std.ArrayList(u8) = .empty,
    tokens: std.ArrayList(u32) = .empty,
    drafts: std.ArrayList(u32) = .empty, // each round's head drafts, then their count
    probs: std.ArrayList(u64) = .empty, // the drafts' confidences' bits

    fn deinit(x: *Run, gpa: std.mem.Allocator) void {
        x.logits.deinit(gpa);
        x.tokens.deinit(gpa);
        x.drafts.deinit(gpa);
        x.probs.deinit(gpa);
    }
};

const Mode = enum { single, alone, shared };

/// Every stream prefilled from 0 in its pool extent; then `rounds` rounds: `shared`, the streams' windows in one
/// forward a round; `alone`, one stream's window a forward (`forwardSegs` of one); `single`, `forward`'s own path. Each
/// stream's rows' logits and samples into its run.
fn decode(rk: *pair.Rank, r: *E.Runner, pool: *const P.Pool, seqs: *[streams]E.Seq, prompts: [streams][]const u32, forced: []const u32, mode: Mode, runs: *[streams]Run) !void {
    const gpa = r.gpa;
    var pending: [streams]u32 = undefined;
    for (seqs, prompts, 0..) |*q, p, i| {
        try rk.tierOpen(q, i);
        try q.bind(pool, i * 8192, 8192);
        try q.reset(r.stream);
        var at: usize = 0;
        while (at < p.len) at = try r.fill(q, p, at);
        pending[i] = try r.first(q);
    }
    var ids: [streams][E.rows]u32 = undefined;
    var outs: [streams][E.rows]u32 = undefined;
    for (widths, 0..) |ws, round| {
        var wins: [streams]E.Runner.Window = undefined;
        for (0..streams) |i| {
            ids[i][0] = pending[i];
            for (ids[i][1..ws[i]], 0..) |*t, j| t.* = forced[(round * 17 + i * 5 + j) % forced.len];
            wins[i] = .{ .q = &seqs[i], .ids = ids[i][0..ws[i]], .out = &outs[i] };
        }
        switch (mode) {
            .shared => {
                try r.windows(&wins);
                for (0..streams) |i| try record(r, &runs[i], &wins, i, outs[i][0..ws[i]]);
            },
            .alone => for (0..streams) |i| {
                try r.windows(wins[i .. i + 1]);
                try record(r, &runs[i], wins[i .. i + 1], 0, outs[i][0..ws[i]]);
            },
            .single => for (0..streams) |i| {
                try r.window(&seqs[i], wins[i].ids, &outs[i]);
                try record(r, &runs[i], wins[i .. i + 1], 0, outs[i][0..ws[i]]);
            },
        }
        for (0..streams) |i| {
            const k = keeps[round][i];
            try r.commit(&seqs[i], ws[i], k);
            pending[i] = outs[i][k - 1];
            // the head owes the kept rows (the token after each: the samples)
            try r.owe(&seqs[i], seqs[i].last.ptr, outs[i][0..k]);
        }
        // the head's drafts for the next round: each stream alone (`absorb`, `chain`), one a call (`draftMany` of
        // one), or every stream's in one batch
        var held: [streams][E.rows]u32 = undefined;
        var probs: [streams][E.rows]f64 = undefined;
        var reqs: [streams]E.Runner.DraftReq = undefined;
        for (&reqs, seqs, 0..) |*q, *sq, i| q.* = .{ .q = sq, .depth = 3, .drafts = &held[i], .probs = &probs[i] };
        switch (mode) {
            .single => for (&reqs) |*q| {
                try r.absorb(q.q);
                q.n = try r.chain(q.q, q.depth, q.drafts, q.probs);
            },
            .alone => for (0..streams) |i| try r.draftMany(reqs[i .. i + 1]),
            .shared => try r.draftMany(&reqs),
        }
        for (reqs, 0..) |q, i| {
            try runs[i].drafts.appendSlice(gpa, q.drafts[0..q.n]);
            try runs[i].drafts.append(gpa, @intCast(q.n));
            for (q.probs[0..q.n]) |pb| try runs[i].probs.append(gpa, @bitCast(pb));
        }
    }
}

/// Window `i` of the last shared forward: its rows' logits and samples.
fn record(r: *E.Runner, run_: *Run, wins: []const E.Runner.Window, i: usize, samples: []const u32) !void {
    var a0: usize = 0;
    for (wins[0..i]) |w| a0 += w.ids.len;
    const row = r.host.len * 2;
    const n = wins[i].ids.len;
    try r.stream.synchronize();
    const at = run_.logits.items.len;
    try run_.logits.resize(r.gpa, at + n * row);
    try r.w.logits.download(a0 * row, run_.logits.items[at..][0 .. n * row]);
    try run_.tokens.appendSlice(r.gpa, samples);
}

pub const Args = pair.Args;

pub fn run(gpu: Gpu, args: Args, rank: usize, master: []const u8, port: u16) !void {
    try pair.launch(gpu, rank, master, port, (E.prompt_rows + 2) * 4096, args, body);
}

fn body(_: Args, gpu: Gpu, rk: *pair.Rank, link: E.Link) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const rank = rk.lay.rank;
    const cfg = &rk.cfg;
    const lay = rk.lay;
    var r = try rk.runner(gpa, d, link, 8192);
    defer r.deinit();
    var pool = try P.Pool.init(gpa, d, cfg, cfg.count(.dsa) + 1, streams * 8192, .bf16, rk.tier == null);
    defer pool.deinit();
    const ring = glm.dsa.indexRing(8192, E.prompt_rows);
    var seqs: [streams]E.Seq = undefined;
    for (&seqs) |*q| q.* = try E.Seq.initSlot(gpa, d, lay, rk.mdl.layers.len, ring, .bf16);
    defer for (&seqs) |*q| q.deinit();

    const ids = rk.prompt();
    var prompts: [streams][]const u32 = undefined;
    for (&prompts, lengths) |*p, n| p.* = ids[0..@min(n, ids.len)];
    const forced = ids[100..600];

    var runs: [3][streams]Run = @splat(@splat(.{}));
    defer for (&runs) |*m| for (m) |*x| x.deinit(gpa);
    for (std.enums.values(Mode), 0..) |mode, mi| try decode(rk, &r, &pool, &seqs, prompts, forced, mode, &runs[mi]);
    var bad: usize = 0;
    const row = r.host.len * 2;
    const single = &runs[@intFromEnum(Mode.single)];
    for ([_]Mode{ .alone, .shared }) |mode| for (0..streams) |i| {
        const x = &single[i];
        const y = &runs[@intFromEnum(mode)][i];
        if (!std.mem.eql(u32, x.drafts.items, y.drafts.items) or !std.mem.eql(u64, x.probs.items, y.probs.items)) {
            std.debug.print("FAIL glm53-shared rank {d}: stream {d} head drafts or confidences differ ({t} against the single path)\n", .{ rank, i, mode });
            bad += 1;
        }
        if (!std.mem.eql(u32, x.tokens.items, y.tokens.items)) {
            std.debug.print("FAIL glm53-shared rank {d}: stream {d} samples differ ({t} against the single path)\n", .{ rank, i, mode });
            bad += 1;
        }
        const n = x.logits.items.len / row;
        for (0..n) |j| if (!std.mem.eql(u8, x.logits.items[j * row ..][0..row], y.logits.items[j * row ..][0..row])) {
            std.debug.print("FAIL glm53-shared rank {d}: stream {d} logits row {d} differ ({t} against the single path)\n", .{ rank, i, j, mode });
            bad += 1;
            break;
        };
    };
    try check.expect(bad == 0, "glm53-shared rank {d}: a stream's rows differ in the shared forward", .{rank});
    var total: usize = 0;
    for (single) |x| total += x.tokens.items.len;
    var nd: usize = 0;
    for (single) |x| nd += x.probs.items.len;
    check.pass("BITEXACT glm53-shared rank {d}: {d} streams, {d} rounds, {d} rows: every row's logits and sample, and {d} head drafts with their confidences, the same on the single path, alone and shared", .{ rank, streams, rounds, total, nd });
}
