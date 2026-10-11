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

pub const Args = struct { dir: []const u8, model: []const u8, aot: []const u8, layers: ?usize };

pub fn run(gpu: Gpu, args: Args, rank: usize, master: []const u8, port: u16) !void {
    try pair.launch(gpu, rank, master, port, (E.prompt_rows + 2) * 4096, args, body);
}

fn body(args: Args, gpu: Gpu, rank: usize, link: E.Link) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var rk: pair.Rank = undefined;
    try rk.load(gpu, args.dir, args.model, rank, args.aot, args.layers);
    defer rk.deinit();
    const cfg = &rk.cfg;
    const lay = rk.lay;
    const stream = rk.stream;
    const capacity: usize = 8192;
    var r = try rk.runner(gpa, d, link, capacity);
    defer r.deinit();
    var pool = try P.Pool.init(gpa, d, cfg, cfg.count(.dsa) + 1, 3 * capacity, .bf16);
    defer pool.deinit();
    const ring = glm.dsa.indexRing(capacity, E.prompt_rows);
    var seqs: [2]E.Seq = undefined;
    for (&seqs) |*q| q.* = try E.Seq.initSlot(gpa, d, lay, rk.mdl.layers.len, ring, .bf16);
    defer for (&seqs) |*q| q.deinit();
    var snaps = try K.Snaps.init(gpa, d, 1, K.snapBytes(&seqs[0]));
    defer snaps.deinit();

    const pr = try rk.prompt();
    defer gpa.free(pr.bytes);
    const a = pr.ids;
    const cut = K.chunk;
    if (a.len < cut + 64) return error.PromptTooShort;
    // B: A's first 2,049 tokens, then A's tail reversed
    const b = try gpa.alloc(u32, a.len);
    defer gpa.free(b);
    @memcpy(b[0 .. cut + 1], a[0 .. cut + 1]);
    for (b[cut + 1 ..], 0..) |*x, i| x.* = a[a.len - 1 - i];

    // B fresh, in extent 0
    var fresh: Trace = .{};
    defer fresh.deinit(gpa);
    try seqs[0].bind(&pool, 0, capacity);
    try seqs[0].reset(stream);
    try finishB(&r, &seqs[0], b, 0, &fresh);

    // A in extent 1 to its boundary, snapshotted there (as a served prompt's pass does), then on to its end
    try seqs[1].bind(&pool, capacity, capacity);
    try seqs[1].reset(stream);
    var at = try r.fill(&seqs[1], a, 0);
    if (at != cut) return error.BadChunk;
    try snaps.save(0, &seqs[1], stream);
    while (at < a.len) at = try r.fill(&seqs[1], a, at);

    // B continued: A's rows before the boundary copied to extent 2, slot 0's state restored from the snapshot
    var cont: Trace = .{};
    defer cont.deinit(gpa);
    try seqs[0].bind(&pool, 2 * capacity, capacity);
    try seqs[0].reset(stream);
    try K.copyRows(&pool, capacity, 2 * capacity, cut, stream);
    try snaps.restore(0, &seqs[0], cut, stream);
    try finishB(&r, &seqs[0], b, cut, &cont);

    var bad: usize = 0;
    if (!std.mem.eql(u32, fresh.tokens.items, cont.tokens.items)) {
        std.debug.print("FAIL glm53-keep rank {d}: tokens {any}, fresh {any}\n", .{ rank, cont.tokens.items, fresh.tokens.items });
        bad += 1;
    }
    const row = r.host.len * 2;
    for (0..steps + 1) |i| {
        const x = fresh.logits.items[i * row ..][0..row];
        const y = cont.logits.items[i * row ..][0..row];
        if (!std.mem.eql(u8, x, y)) {
            std.debug.print("FAIL glm53-keep rank {d}: logits row {d} differ (first byte {d})\n", .{ rank, i, std.mem.indexOfDiff(u8, x, y).? });
            bad += 1;
        }
    }
    try check.expect(bad == 0, "glm53-keep rank {d}: the continued prompt differs from the fresh one", .{rank});
    check.pass("BITEXACT glm53-keep rank {d}: a {d}-token prompt continued from a kept boundary at {d}: {d} tokens and {d} logits rows equal to its fresh prefill", .{ rank, b.len, cut, fresh.tokens.items.len, steps + 1 });
}
