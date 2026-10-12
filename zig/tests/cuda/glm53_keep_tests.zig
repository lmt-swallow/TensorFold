//! `glm53-keep <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir> [layers]`: a kept prompt continued, on both
//! Sparks (master "local": both ranks on this GPU, the first `layers` layers).
//! Prompt A (the forward fixture's) is prefilled in a pool extent with a snapshot at its 2,048-row boundary; prompt
//! B shares A's first 2,049 tokens and then differs. B continued from A's boundary (A's rows copied to another
//! extent, the slot's state restored from the snapshot) must equal B prefilled from 0 byte for byte: the last row's
//! logits, the first token, and 16 serial decode steps' tokens and logits.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const pair = @import("glm53_pair.zig");
const Gpu = check.Gpu;

const Buf = cuda.DeviceBuffer;
const Ptr = cuda.abi.DevicePtr;
const M = glm.model;
const T = glm.mtp;
const E = glm.engine;
const P = glm.pool;
const K = glm.kept;

const steps = 16;

/// What a run of prompt B leaves to compare: the logits after its prompt and after each step, and the tokens.
const Trace = struct {
    logits: std.ArrayList(u8) = .empty,
    tokens: std.ArrayList(u32) = .empty,

    fn deinit(t: *Trace, gpa: std.mem.Allocator) void {
        t.logits.deinit(gpa);
        t.tokens.deinit(gpa);
    }
};

fn logitsRow(r: *E.Runner, t: *Trace) !void {
    const n = r.host.len * 2;
    try r.stream.synchronize();
    const at = t.logits.items.len;
    try t.logits.resize(r.gpa, at + n);
    try r.w.logits.download(0, t.logits.items[at..][0..n]);
}

/// B from `start` (the sequence holds the rows before it) to its end, then `steps` one-row windows.
fn finishB(r: *E.Runner, q: *E.Seq, b: []const u32, start: usize, t: *Trace) !void {
    var at = start;
    while (at < b.len) at = try r.fill(q, b, at);
    try logitsRow(r, t);
    var tok = try r.first(q);
    try t.tokens.append(r.gpa, tok);
    var out: [1]u32 = undefined;
    for (0..steps) |_| {
        try r.window(q, &.{tok}, &out);
        try logitsRow(r, t);
        try r.commit(q, 1, 1);
        tok = out[0];
        try t.tokens.append(r.gpa, tok);
    }
}

fn diffRows(r: *const E.Runner, rank: usize, what: []const u8, x: *const Trace, y: *const Trace) usize {
    const row = r.host.len * 2;
    var bad: usize = 0;
    for (0..steps + 1) |i| {
        const a = x.logits.items[i * row ..][0..row];
        const b = y.logits.items[i * row ..][0..row];
        if (!std.mem.eql(u8, a, b)) {
            std.debug.print("FAIL glm53-keep rank {d}: {s} logits row {d} differ (first byte {d})\n", .{ rank, what, i, std.mem.indexOfDiff(u8, a, b).? });
            bad += 1;
        }
    }
    return bad;
}

/// A sequence's state as named digests: the KDA states (the current parity) and conv windows, every DSA cache's
/// index rings, latent rows and pooled keys before `len` (and the head's), so a mismatch names its part.
const Parts = struct {
    names: [64][]const u8 = undefined,
    sums: [64][32]u8 = undefined,
    n: usize = 0,

    fn add(p: *Parts, gpa: std.mem.Allocator, name: []const u8, ptr: Ptr, len: usize, d: *const cuda.Driver) !void {
        const host = try gpa.alloc(u8, len);
        defer gpa.free(host);
        const buf: cuda.DeviceBuffer = .{ .d = d, .ptr = ptr, .len = len };
        try buf.download(0, host);
        std.crypto.hash.sha2.Sha256.hash(host, &p.sums[p.n], .{});
        p.names[p.n] = name;
        p.n += 1;
    }
};

fn parts(r: *E.Runner, q: *const E.Seq, len: usize) !Parts {
    try r.stream.synchronize();
    var p: Parts = .{};
    const k = &q.st.kda;
    const g = r.gpa;
    const d = r.d;
    try p.add(g, "kda states", k.state(k.cur, 0), k.layers * k.stateBytes(), d);
    try p.add(g, "conv windows", k.conv.ptr, k.conv.len, d);
    const names = [_][]const u8{ "dsa 0", "dsa 1", "dsa 2", "dsa 3", "dsa 4", "dsa 5", "dsa 6", "dsa 7", "dsa 8", "dsa 9", "dsa 10", "dsa 11" };
    for (q.st.caches, 0..) |c, i| {
        try p.add(g, names[i], c.ik.ptr, c.ik.len, d);
        try p.add(g, names[i], c.ig.ptr, c.ig.len, d);
        try p.add(g, names[i], c.latent.ptr, len * (c.latent.len / c.capacity), d);
        try p.add(g, names[i], c.pk.ptr, len / 4 * (c.pk.len / (c.capacity / 4 + 2)), d);
    }
    return p;
}

fn compare(rank: usize, step: usize, x: *const Parts, y: *const Parts) usize {
    for (0..x.n) |i| if (!std.mem.eql(u8, &x.sums[i], &y.sums[i])) {
        std.debug.print("FAIL glm53-keep rank {d}: step {d}: {s} (part {d}) differs\n", .{ rank, step, x.names[i], i });
        return 1;
    };
    return 0;
}

/// Both B runs from their starts to B's end, then `steps` one-row windows each, in lockstep: their logits and
/// tokens into the traces, their states compared after the prompt and after every step (the first differing part
/// named). Returns the mismatches.
fn lockstep(r: *E.Runner, qs: [2]*E.Seq, b: []const u32, starts: [2]usize, ts: [2]*Trace, rank: usize) !usize {
    var toks: [2]u32 = undefined;
    for (qs, starts, ts, &toks) |q, start, t, *tok| {
        var at = start;
        while (at < b.len) at = try r.fill(q, b, at);
        try logitsRow(r, t);
        tok.* = try r.first(q);
        try t.tokens.append(r.gpa, tok.*);
    }
    var bad: usize = 0;
    var p0 = try parts(r, qs[0], qs[0].len());
    var p1 = try parts(r, qs[1], qs[1].len());
    bad += compare(rank, 0, &p0, &p1);
    var out: [1]u32 = undefined;
    for (0..steps) |step| {
        for (qs, ts, &toks) |q, t, *tok| {
            try r.window(q, &.{tok.*}, &out);
            try logitsRow(r, t);
            try r.commit(q, 1, 1);
            tok.* = out[0];
            try t.tokens.append(r.gpa, tok.*);
        }
        if (bad == 0) {
            p0 = try parts(r, qs[0], qs[0].len());
            p1 = try parts(r, qs[1], qs[1].len());
            bad += compare(rank, step + 1, &p0, &p1);
        }
    }
    return bad;
}

pub const Args = pair.Args;

/// On a tier, sequence `q` reads A's latent rows before `rows` through A's tier streams (1's: backbone 2, head 3).
fn aliasA(q: *E.Seq, rows: usize) !void {
    const x = q.tier orelse return;
    try x.t.alias(2, x.h, rows);
    try x.t.alias(3, x.hm, rows);
}

pub fn run(gpu: Gpu, args: Args, rank: usize, master: []const u8, port: u16) !void {
    try pair.launch(gpu, rank, master, port, (E.prompt_rows + 2) * 4096, args, body);
}

fn body(_: Args, gpu: Gpu, rk: *pair.Rank, link: E.Link) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const rank = rk.lay.rank;
    const cfg = &rk.cfg;
    const lay = rk.lay;
    const stream = rk.stream;
    const capacity: usize = 8192;
    var r = try rk.runner(gpa, d, link, capacity);
    defer r.deinit();
    var pool = try P.Pool.init(gpa, d, cfg, cfg.count(.dsa) + 1, 4 * capacity, .bf16, rk.tier == null);
    defer pool.deinit();
    const ring = glm.dsa.indexRing(capacity, E.prompt_rows);
    var seqs: [3]E.Seq = undefined;
    for (&seqs) |*q| q.* = try E.Seq.initSlot(gpa, d, lay, rk.mdl.layers.len, ring, .bf16);
    defer for (&seqs) |*q| q.deinit();
    var snaps = try K.Snaps.init(gpa, d, 1, K.snapBytes(&seqs[0]));
    defer snaps.deinit();

    const a = rk.prompt();
    const cut = K.chunk;
    if (a.len < cut + 64) return error.PromptTooShort;
    // B: A's first 2,049 tokens, then A's tail reversed
    const b = try gpa.alloc(u32, a.len);
    defer gpa.free(b);
    @memcpy(b[0 .. cut + 1], a[0 .. cut + 1]);
    for (b[cut + 1 ..], 0..) |*x, i| x.* = a[a.len - 1 - i];

    // A in extent 1 to its boundary, snapshotted there (as a served prompt's pass does), then on to its end
    try rk.tierOpen(&seqs[1], 1);
    try seqs[1].bind(&pool, capacity, capacity);
    try seqs[1].reset(stream);
    var at = try r.fill(&seqs[1], a, 0);
    if (at != cut) return error.BadChunk;
    try snaps.save(0, &seqs[1], stream);
    while (at < a.len) at = try r.fill(&seqs[1], a, at);

    // B fresh in extent 0 (slot 0); B continued in extent 2 (slot 2): A's rows before the boundary copied in, the
    // slot's state restored from the snapshot
    try rk.tierOpen(&seqs[0], 0);
    try seqs[0].bind(&pool, 0, capacity);
    try seqs[0].reset(stream);
    // slot 2 first serves a whole other stream (B from 0, 16 steps) in extent 2, as a served slot does before it is
    // reused: nothing it leaves may reach the continued prompt
    {
        try rk.tierOpen(&seqs[2], 2);
        try seqs[2].bind(&pool, 2 * capacity, capacity);
        try seqs[2].reset(stream);
        var at2: usize = 0;
        while (at2 < b.len) at2 = try r.fill(&seqs[2], b, at2);
        var tok = try r.first(&seqs[2]);
        var o: [1]u32 = undefined;
        for (0..steps) |_| {
            try r.window(&seqs[2], &.{tok}, &o);
            try r.commit(&seqs[2], 1, 1);
            tok = o[0];
        }
    }
    try rk.tierOpen(&seqs[2], 2);
    try seqs[2].bind(&pool, 2 * capacity, capacity);
    try seqs[2].reset(stream);
    try K.copyRows(&pool, capacity, 2 * capacity, cut, stream);
    try aliasA(&seqs[2], cut);
    try snaps.restore(0, &seqs[2], cut, stream);
    var fresh: Trace = .{};
    defer fresh.deinit(gpa);
    var cont: Trace = .{};
    defer cont.deinit(gpa);
    var bad = try lockstep(&r, .{ &seqs[0], &seqs[2] }, b, .{ 0, cut }, .{ &fresh, &cont }, rank);
    if (!std.mem.eql(u32, fresh.tokens.items, cont.tokens.items)) {
        std.debug.print("FAIL glm53-keep rank {d}: tokens {any}, fresh {any}\n", .{ rank, cont.tokens.items, fresh.tokens.items });
        bad += 1;
    }
    bad += diffRows(&r, rank, "continued", &fresh, &cont);
    // slot 0, which served fresh B to its end in extent 0, moves to extent 3 and continues B there (a served slot
    // taking a kept prompt's continuation after another stream): the same rows again
    var moved: Trace = .{};
    defer moved.deinit(gpa);
    try rk.tierOpen(&seqs[0], 3);
    try seqs[0].bind(&pool, 3 * capacity, capacity);
    try seqs[0].reset(stream);
    try K.copyRows(&pool, capacity, 3 * capacity, cut, stream);
    try aliasA(&seqs[0], cut);
    try snaps.restore(0, &seqs[0], cut, stream);
    try finishB(&r, &seqs[0], b, cut, &moved);
    if (!std.mem.eql(u32, fresh.tokens.items, moved.tokens.items)) {
        std.debug.print("FAIL glm53-keep rank {d}: moved slot's tokens {any}, fresh {any}\n", .{ rank, moved.tokens.items, fresh.tokens.items });
        bad += 1;
    }
    bad += diffRows(&r, rank, "moved slot's", &fresh, &moved);
    try check.expect(bad == 0, "glm53-keep rank {d}: the continued prompt differs from the fresh one", .{rank});
    check.pass("BITEXACT glm53-keep rank {d}: a {d}-token prompt continued from a kept boundary at {d}: {d} tokens and {d} logits rows equal to its fresh prefill", .{ rank, b.len, cut, fresh.tokens.items.len, steps + 1 });
}
