//! `glm53-keep <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir>`: a kept prompt continued, on both Sparks.
//! Prompt A (the forward fixture's) is prefilled in a pool extent with a snapshot at its 2,048-row boundary; prompt
//! B shares A's first 2,049 tokens and then differs. B continued from A's boundary (A's rows copied to another
//! extent, the slot's state restored from the snapshot) must equal B prefilled from 0 byte for byte: the last row's
//! logits, the first token, and 16 serial decode steps' tokens and logits.
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
const E = glm.engine;
const P = glm.pool;
const K = glm.kept;

const Link = struct {
    comm: *const cuda.comm.Comm,

    fn gather(p: *anyopaque, send: Ptr, recv: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(p));
        try l.comm.allGather(send, recv, floats, .f32, stream.handle);
    }
};

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
    var link: Link = .{ .comm = &comm };
    const k: M.Kernels = .{ .g = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams }, .kk = &kk, .mk = &mk };
    const capacity: usize = 8192;
    var r = try E.Runner.init(gpa, d, lay, k, &tk, &mdl, &mtp, .{ .ctx = &link, .gather = Link.gather }, stream, capacity);
    defer r.deinit();
    var pool = try P.Pool.init(gpa, d, &cfg, cfg.count(.dsa) + 1, 3 * capacity, .bf16);
    defer pool.deinit();
    const ring = glm.dsa.indexRing(capacity, E.prompt_rows);
    var seqs: [2]E.Seq = undefined;
    for (&seqs) |*q| q.* = try E.Seq.initSlot(gpa, d, lay, cfg.layers, ring, .bf16);
    defer for (&seqs) |*q| q.deinit();
    var snaps = try K.Snaps.init(gpa, d, 1, K.snapBytes(&seqs[0]));
    defer snaps.deinit();

    var nb: [64]u8 = undefined;
    const prompt_b = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_prompt_ids", .{rank}));
    defer gpa.free(prompt_b);
    const a = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(prompt_b)));
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
