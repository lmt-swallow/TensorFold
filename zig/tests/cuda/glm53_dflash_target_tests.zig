//! Single-stream DFlash2 target gate: native optimized/direct prompt cache parity, then full greedy output IDs.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
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

pub fn run(gpu: Gpu, reference: []const u8, model: []const u8, draft: []const u8, rank: usize, master: []const u8, port: u16, aot_dir: []const u8) !void {
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
    var draft_model = try glm.dflash_weights.Model.open(gpa, gpu.io, d, draft, rank);
    defer draft_model.deinit();
    var projections = try glm.dflash_projections.Kernels.load(d);
    defer projections.deinit();
    const dk: glm.dflash_kernels.Kernels = .{ .set = &set };
    var lt = try cuda.cublaslt.Library.open();
    defer lt.close();
    var hp = try cuda.cublaslt.Linear.init(&lt, 7, 256, 4096, .bf16, 32 << 20);
    defer hp.deinit();
    var session = try glm.dflash_session.Session.init(&e.r, &e.q, &draft_model, &mtp.head, dk, projections, &hp);
    defer session.deinit();
    if (std.c.getenv("TF_GLM_DFLASH_ROUND_TRACE")) |v| session.trace_rounds = std.mem.eql(u8, std.mem.span(v), "1");
    if (session.trace_rounds) std.debug.print("DFLASH_POLICY block=8 topk=16 confidence=0.3 mask={d} target_dense=bf16 target_kv=bf16 copy=off\n", .{draft_model.mask});
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, reference, gpa, .limited(1 << 22));
    defer gpa.free(raw);
    var oracle = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer oracle.deinit();
    try check.expect(std.mem.eql(u8, oracle.value.object.get("mode").?.string, "reference"), "DFlash2 pinned reference mode", .{});
    const case = oracle.value.object.get("cases").?.object.get("code").?.object;
    const prompt = try ids(gpa, case.get("prompt_ids").?);
    defer gpa.free(prompt);
    const want = try ids(gpa, case.get("output_ids").?);
    defer gpa.free(want);
    try check.expect(prompt.len == 52 and want.len == 512, "DFlash2 initial gate uses exactly the pinned 52/512 case", .{});

    // Compare durable target state and final logits before using the direct prompt path for drafting.
    var baseline = try glm.engine.Seq.init(gpa, d, lay, cfg.layers, 8192, .bf16);
    defer baseline.deinit();
    var start: usize = 0;
    while (start < prompt.len) start = try e.r.fill(&baseline, prompt, start);
    try stream.synchronize();
    const baseline_logits = try gpa.alloc(u8, cfg.vocab / 2 * 2);
    defer gpa.free(baseline_logits);
    try e.r.w.logits.download(0, baseline_logits);
    start = 0;
    while (start < prompt.len) start = try session.fill(prompt, start);
    try stream.synchronize();
    try equalState(gpu, &baseline, &e.q, cfg.kv_lora, cfg.index_dim);
    const direct_logits = try gpa.alloc(u8, baseline_logits.len);
    defer gpa.free(direct_logits);
    try e.r.w.logits.download(0, direct_logits);
    try check.expect(std.mem.eql(u8, baseline_logits, direct_logits), "DFlash2 direct and optimized prompt target logits match", .{});

    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(gpa);
    try got.append(gpa, try session.first());
    var rounds: usize = 0;
    while (got.items.len < want.len) {
        const before = e.q.st.pos;
        const draft_start = check.now(gpu.io);
        const drafts = try session.draft(got.items[got.items.len - 1]);
        defer gpa.free(drafts);
        const draft_end = check.now(gpu.io);
        var emitted: [16]u32 = undefined;
        const count = try session.verify(got.items[got.items.len - 1], drafts, want.len - got.items.len, &emitted);
        const verify_end = check.now(gpu.io);
        if (session.trace_rounds) std.debug.print("DFLASH_ROUND round={d} source=learned width={d} proposed={any} verified={any} accepted={d} bonus={d} committed_before={d} committed_after={d} draft_ns={d} verify_commit_taps_ns={d}\n", .{ rounds, 1 + drafts.len, drafts, session.verified_samples[0..session.verified_rows], count - 1, emitted[count - 1], before, e.q.st.pos, draft_end - draft_start, verify_end - draft_end });
        try got.appendSlice(gpa, emitted[0..count]);
        rounds += 1;
        try check.expect(std.mem.eql(u32, got.items, want[0..got.items.len]), "DFlash2 verified output prefix equals full-ID oracle at round {d}", .{rounds});
        if (std.mem.indexOfScalar(u32, cfg.eos[0..cfg.eos_n], got.items[got.items.len - 1]) != null) break;
    }
    try check.expect(std.mem.eql(u32, got.items, want), "DFlash2 entire 512-token output equals pinned full-ID oracle", .{});
    std.debug.print("RESULT glm53-dflash-target rank {d}: prompt {d}, output {d}, rounds {d}, ids {any}\n", .{ rank, prompt.len, got.items.len, rounds, got.items });
    check.pass("DFlash2 target full-ID parity; timings are not a serving performance measurement", .{});
}

fn ids(a: std.mem.Allocator, value: std.json.Value) ![]u32 {
    if (value != .array) return error.BadDflashOracle;
    const out = try a.alloc(u32, value.array.items.len);
    errdefer a.free(out);
    for (value.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer >= 154880) return error.BadDflashOracle;
        out[i] = @intCast(v.integer);
    }
    return out;
}

fn sameRange(gpu: Gpu, left: Ptr, right: Ptr, length: usize) !void {
    const a = try gpu.gpa.alloc(u8, length);
    defer gpu.gpa.free(a);
    const b = try gpu.gpa.alloc(u8, length);
    defer gpu.gpa.free(b);
    try (Buf{ .d = gpu.d, .ptr = left, .len = length }).download(0, a);
    try (Buf{ .d = gpu.d, .ptr = right, .len = length }).download(0, b);
    try check.expect(std.mem.eql(u8, a, b), "DFlash2 direct/optimized target durable state bytes match", .{});
}

fn equalState(gpu: Gpu, left: *const glm.engine.Seq, right: *const glm.engine.Seq, latent_width: usize, index_width: usize) !void {
    try check.expect(left.st.pos == right.st.pos and left.st.pos == 52, "DFlash2 target committed prompt positions match", .{});
    for (0..left.st.kda.layers) |i| try sameRange(gpu, left.st.kda.state(left.st.kda.cur, i), right.st.kda.state(right.st.kda.cur, i), left.st.kda.stateBytes());
    try sameRange(gpu, left.st.kda.conv.ptr, right.st.kda.conv.ptr, left.st.kda.conv.len);
    for (left.st.caches, right.st.caches) |l, r| {
        try sameRange(gpu, l.latent.ptr, r.latent.ptr, 52 * latent_width * 2);
        try sameRange(gpu, l.ik.ptr, r.ik.ptr, 52 * index_width * 2);
        try sameRange(gpu, l.ig.ptr, r.ig.ptr, 52 * index_width * 2);
        try sameRange(gpu, l.pk.ptr, r.pk.ptr, 13 * index_width * 2);
    }
}
