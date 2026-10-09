//! `dsv41-lanes-prefill <dir> <rank-cache-dir> <model-dir> <rank> [aot-dir]`: the served engine's steps
//! (cuda_lanes.zig) at one rank on dump_prompt.py's fixture (prompt_capture.sh's work dir): for each chunk size the
//! fixture holds, the prefill record (`Prefill.encode`, as rank 0 announces it) run by `Prefill.run` on a
//! `Sequence` (as every rank runs it), the other rank's partials supplied from the fixture at every gather in the
//! fork's order; the caches after the prompt and the last row's logits byte-equal to the fork's. A fixture dumped
//! with `--decode` then has the first token (the logits' argmax) equal to the fork's and one round (`Sequence.round`,
//! the decode layer loop on the same caches) byte-equal in its logits and caches. A fixture dumped with `--engram`
//! runs Engram (TF_DS_ENGRAM, TF_DS_TOKEN_MAP); one dumped with `--replay` sends the record's replay flag.
//!
//! `dsv41-lanes-spec <dir> <rank-cache-dir> <model-dir> <rank> [aot-dir]`: the served engine's drafted rounds at one
//! rank on dump_spec.py's fixture (spec_capture.sh's work dir): the prefill record with the drafter (each chunk's
//! taps absorbed), then each round's records as rank 0 announces them and every rank runs them (`Draft`: absorb
//! the kept rows and draft after the pending token; `Verify`: the window through the decode layer loop with taps;
//! `Keep`: the kept prefix), the other rank's partials supplied from the fixture at every gather (the layers', the
//! drafter stages' and the Markov steps') in the fork's order. Per round: the drafts, the window's logits, taps and
//! greedy picks, the accepted count, the caches after the keep and the drafter's rings after the absorb, byte-equal
//! to the fork's. The window costs (1 .. 6 rows) and a draft are timed after, at this rank's layers.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;
const Fixture = fixture.Fixture;
const Ptr = cuda.abi.DevicePtr;

fn upload(gpu: Gpu, fx: Fixture, name: []const u8) !cuda.DeviceBuffer {
    const bytes = try fx.bytes(name);
    defer gpu.gpa.free(bytes);
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}

/// The other rank's partials in gather order from the fixture, this rank's copied in beside them.
const Supply = struct {
    gpu: Gpu,
    rank: usize,
    other: std.ArrayList(cuda.DeviceBuffer),
    next: usize = 0,

    fn gather(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
        const self: *Supply = @ptrCast(@alignCast(ctx));
        if (self.next >= self.other.items.len) return error.GatherOverrun;
        const o = self.other.items[self.next];
        self.next += 1;
        if (o.len != words * 4) return error.GatherShape;
        const dst: cuda.DeviceBuffer = .{ .d = self.gpu.d, .ptr = out, .len = 2 * words * 4 };
        try dst.copyFrom(self.rank * words * 4, partial, words * 4, stream.handle);
        try dst.copyFrom((1 - self.rank) * words * 4, o.ptr, words * 4, stream.handle);
    }

    fn add(self: *Supply, fx: Fixture, name: []const u8) !void {
        try self.other.append(self.gpu.gpa, try upload(self.gpu, fx, name));
    }

    fn clear(self: *Supply) void {
        for (self.other.items) |*b| b.free();
        self.other.clearRetainingCapacity();
        self.next = 0;
    }
};

fn same(gpu: Gpu, fx: Fixture, got: cuda.DeviceBuffer, name: []const u8, label: []const u8, limit: usize) bool {
    var nb: [128]u8 = undefined;
    const want = fx.bytes(std.fmt.bufPrint(&nb, "expected_{s}", .{name}) catch return false) catch {
        std.debug.print("FAIL {s} {s}: no expected array\n", .{ label, name });
        return false;
    };
    defer gpu.gpa.free(want);
    const bytes = check.download(gpu, got) catch return false;
    defer gpu.gpa.free(bytes);
    const n = if (limit == 0) want.len else @min(limit, want.len);
    if (n > bytes.len) {
        std.debug.print("FAIL {s} {s}: {d} bytes expected, {d} held\n", .{ label, name, n, bytes.len });
        return false;
    }
    var what: [192]u8 = undefined;
    check.sameBytes(std.fmt.bufPrint(&what, "{s} {s}", .{ label, name }) catch return false, bytes[0..n], want[0..n]) catch return false;
    return true;
}

/// Whether the fixture holds the array `name` (a replay chunk's encoder-only layers made no gathers).
fn holds(fx: Fixture, name: []const u8) bool {
    const arrays = (fx.parsed.value.object.get("arrays") orelse return false).object;
    return arrays.contains(name);
}

/// Every layer's cache planes against the fixture's `<prefix>L<i>.<plane>` (all but the compressed planes' scratch row).
fn planesSame(gpu: Gpu, fx: Fixture, fwd: *const dsv41.forward.Forward, prefix: []const u8, label: []const u8) !bool {
    const cfg = fwd.cfg;
    var ok = true;
    var nb: [96]u8 = undefined;
    for (0..fwd.w.layers.len) |li| {
        ok = same(gpu, fx, fwd.rings[li], try std.fmt.bufPrint(&nb, "{s}L{d}.ring", .{ prefix, li }), label, 0) and ok;
        if (fwd.planes[li]) |*p| {
            const kept = p.rows - 1;
            const hd = cfg.head_dim;
            ok = same(gpu, fx, p.comp_codes, try std.fmt.bufPrint(&nb, "{s}L{d}.comp_codes", .{ prefix, li }), label, kept * hd / 2) and ok;
            ok = same(gpu, fx, p.comp_scales, try std.fmt.bufPrint(&nb, "{s}L{d}.comp_scales", .{ prefix, li }), label, kept * hd / 16) and ok;
            if (cfg.isIndexSource(li)) {
                ok = same(gpu, fx, p.idxk_codes, try std.fmt.bufPrint(&nb, "{s}L{d}.idxk_codes", .{ prefix, li }), label, kept * cfg.idx_dim / 2) and ok;
                ok = same(gpu, fx, p.idxk_scales, try std.fmt.bufPrint(&nb, "{s}L{d}.idxk_scales", .{ prefix, li }), label, kept * cfg.idx_dim / 32) and ok;
            }
            if (cfg.ratio(li) > 1) {
                ok = same(gpu, fx, p.raw_kv, try std.fmt.bufPrint(&nb, "{s}L{d}.raw_kv", .{ prefix, li }), label, 0) and ok;
                ok = same(gpu, fx, p.raw_score, try std.fmt.bufPrint(&nb, "{s}L{d}.raw_score", .{ prefix, li }), label, 0) and ok;
            }
        }
    }
    return ok;
}

fn resetCaches(fwd: *dsv41.forward.Forward) !void {
    for (fwd.rings) |*r| try r.fill8(0, null);
    for (fwd.planes) |*p| if (p.*) |*pl| {
        try pl.comp_codes.fill8(0, null);
        try pl.comp_scales.fill8(0, null);
        try pl.idxk_codes.fill8(0, null);
        try pl.idxk_scales.fill8(127, null);
        try pl.raw_kv.fill8(0, null);
        try pl.raw_score.fill8(0, null);
    };
}

pub fn prefill(gpu: Gpu, dir: []const u8, cache_dir: []const u8, model_dir: []const u8, rank: usize, aot_override: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const F = dsv41.forward;
    const L = dsv41.serve_lanes;
    try check.expect(rank < 2, "rank {d}: the fixture holds two ranks", .{rank});
    const aot_dir = if (aot_override) |o| try gpa.dupe(u8, o) else try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    std.debug.print("Triton set: {s} ({d} variants)\n", .{ aot_dir, set.variants.len });
    var glue = try dsv41.attention.Glue.load(gpu.d);
    defer glue.deinit();
    var kernels = try dsv41.exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var ek = try dsv41.experts.Kernels.load(gpu.d);
    defer ek.deinit();
    var cfg = try dsv41.config.Config.read(gpa, io, model_dir);
    const cdir = try std.fs.path.join(gpa, &.{ dir, "out", "fixtures", "prompt_common" });
    defer gpa.free(cdir);
    var cf = try Fixture.open(gpa, io, cdir);
    defer cf.deinit();
    const layers: usize = @intCast(try cf.int("layers"));
    const prompt_len: usize = @intCast(try cf.int("prompt"));
    const bucket: usize = @intCast(try cf.int("bucket"));
    const cap: usize = @intCast(try cf.int("cap"));
    const world: u32 = @intCast(try cf.int("world"));
    try check.expect(world == 2, "the fixture's world is {d}; this test supplies one other rank", .{world});
    const replay = (cf.int("replay") catch 0) != 0;

    const rc_path = (try dsv41.rank_cache.RankCache.locate(gpa, io, cache_dir, rank, world)) orelse return error.NoRankCache;
    defer gpa.free(rc_path);
    var rc = try dsv41.rank_cache.RankCache.open(gpa, rc_path);
    defer rc.close();
    var w = try dsv41.weights.load(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1, layers);
    defer w.deinit();
    std.debug.print("loaded {d} layers of rank {d} from {s}\n", .{ layers, rank, rc_path });

    var rope_c: dsv41.attention.Rope = .{ .cos = try upload(gpu, cf, "cos_c"), .sin = try upload(gpu, cf, "sin_c"), .rows = @intCast(try cf.int("rope_rows")) };
    defer rope_c.cos.free();
    defer rope_c.sin.free();
    var rope_w: dsv41.attention.Rope = .{ .cos = try upload(gpu, cf, "cos_w"), .sin = try upload(gpu, cf, "sin_w"), .rows = @intCast(try cf.int("rope_rows")) };
    defer rope_w.cos.free();
    defer rope_w.sin.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    // the engine's shapes: decode windows of max_rows, prompt buffers for the production chunk
    var fwd = try F.Forward.init(gpa, gpu.d, &cfg, &w, &set, &glue, &ek, world, L.max_rows, cap, bucket, rope_c, rope_w, stream);
    defer fwd.deinit();
    const rows = L.default_chunk;
    const eng = try @import("dsv41_forward_tests.zig").engramFor(gpu, cf, &cfg, &w, &set, aot_dir, rows);
    defer if (eng) |e| e.deinit();
    var seq = try L.Sequence.init(gpa, gpu.d, &fwd, &kernels, eng, rows, stream, null);
    defer seq.deinit(gpa);
    try check.expect(seq.limit() == cap - L.extra, "the sequence's limit", .{});

    var supply: Supply = .{ .gpu = gpu, .rank = rank, .other = .empty };
    defer {
        supply.clear();
        supply.other.deinit(gpa);
    }
    const gather: F.Gather = .{ .ctx = &supply, .call = Supply.gather };
    const ids_bytes = try cf.bytes("prompt_ids");
    defer gpa.free(ids_bytes);
    const prompt_ids = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(ids_bytes)));
    try check.expect(prompt_ids.len == prompt_len, "the fixture's prompt has {d} ids for {d}", .{ prompt_ids.len, prompt_len });
    const op_buf = try gpa.alloc(u8, L.maxOpBytes(cap));
    defer gpa.free(op_buf);
    const ids_buf = try gpa.alloc(i64, cap);
    defer gpa.free(ids_buf);

    const chunks_text = try gpa.dupe(u8, try cf.string("chunks"));
    defer gpa.free(chunks_text);
    var it = std.mem.splitScalar(u8, chunks_text, ',');
    var failures: usize = 0;
    const other: usize = 1 - rank;
    while (it.next()) |text| {
        const chunk = try std.fmt.parseInt(usize, text, 10);
        var nb: [128]u8 = undefined;
        const fdir = try std.fs.path.join(gpa, &.{ dir, "out", "fixtures", try std.fmt.bufPrint(&nb, "prompt_c{d}", .{chunk}) });
        defer gpa.free(fdir);
        var fx = try Fixture.open(gpa, io, fdir);
        defer fx.deinit();
        var label_buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "dsv41-lanes-prefill rank {d} chunk {d}", .{ rank, chunk });
        const n_chunks: usize = @intCast(try fx.int("n_chunks"));
        try check.expect(n_chunks == (prompt_len + chunk - 1) / chunk, "{s}: the fixture's chunks", .{label});
        const decode_token: ?i64 = fx.int("decode_token") catch null;
        supply.clear();
        try resetCaches(&fwd);
        // the fork's gathers in order: every chunk's layers (Engram's on its layers, attention's, the MoE's), the
        // head's on the last chunk; then the decode row's
        for (0..n_chunks) |ci| for (0..layers) |li| {
            if (replay and !holds(fx, try std.fmt.bufPrint(&nb, "expected_C{d}.L{d}.p_attn.r{d}", .{ ci, li, other }))) continue;
            if (eng != null and w.layers[li].engram_wkv != null) try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_C{d}.L{d}.p_engram.r{d}", .{ ci, li, other }));
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_C{d}.L{d}.p_attn.r{d}", .{ ci, li, other }));
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_C{d}.L{d}.p_moe.r{d}", .{ ci, li, other }));
        };
        try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_head.local.r{d}", .{other}));
        const prompt_gathers = supply.other.items.len;
        if (decode_token != null) {
            for (0..layers) |li| {
                if (eng != null and w.layers[li].engram_wkv != null) try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_D.L{d}.p_engram.r{d}", .{ li, other }));
                try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_D.L{d}.p_attn.r{d}", .{ li, other }));
                try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_D.L{d}.p_moe.r{d}", .{ li, other }));
            }
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_D.head.local.r{d}", .{other}));
        }

        // the step as announced and as run on every rank
        const op = L.Prefill.encode(op_buf, chunk, eng != null, false, replay, prompt_ids);
        const parsed = try L.Prefill.parse(op);
        try check.expect(parsed.chunk == chunk and parsed.engram == (eng != null) and parsed.replay == replay and parsed.len() == prompt_len, "{s}: the record parses back", .{label});
        // the caches' fills and the partials' uploads above ran on the legacy stream, outside the step's order
        try gpu.ctx.synchronize();
        const t0 = check.now(io);
        try L.Prefill.run(op, &seq, gather, ids_buf);
        try stream.synchronize();
        const ns = check.now(io) - t0;
        var ok = true;
        if (supply.next != prompt_gathers) {
            std.debug.print("FAIL {s}: {d} gathers in the prompt, the fork's {d}\n", .{ label, supply.next, prompt_gathers });
            ok = false;
        }
        ok = (try planesSame(gpu, fx, &fwd, try std.fmt.bufPrint(&nb, "C{d}.", .{n_chunks - 1}), label)) and ok;
        const lview: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = seq.logits, .len = cfg.vocab * 4 };
        ok = same(gpu, fx, lview, "logits", label, 0) and ok;
        ok = seq.pos == prompt_len and ok;
        std.debug.print("{s}: prefill of {d} tokens through {d} layers in {d} ms ({d:.0} tokens/s at this rank, the other rank's partials supplied)\n", .{ label, prompt_len, layers, @divTrunc(ns, 1_000_000), @as(f64, @floatFromInt(prompt_len)) * 1e9 / @as(f64, @floatFromInt(ns)) });
        if (ok) check.pass("BITEXACT {s}: {d} chunk(s) over {d} tokens through {d} layers: the gathers' order, the caches and the last row's logits equal to the Python fork's", .{ label, n_chunks, prompt_len, layers });

        if (decode_token) |want| {
            const token = try seq.argmax();
            var dok = token == want;
            if (!dok) std.debug.print("FAIL {s}: the first token {d}, the fork's {d}\n", .{ label, token, want });
            try gpu.ctx.synchronize();
            try seq.round(gather, token);
            try stream.synchronize();
            if (supply.next != supply.other.items.len) {
                std.debug.print("FAIL {s}: {d} gathers through the round, the fork's {d}\n", .{ label, supply.next, supply.other.items.len });
                dok = false;
            }
            dok = (try planesSame(gpu, fx, &fwd, "D.", label)) and dok;
            const dview: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = seq.logits, .len = cfg.vocab * 4 };
            dok = same(gpu, fx, dview, "D.logits", label, 0) and dok;
            dok = seq.pos == prompt_len + 1 and dok;
            if (dok) check.pass("BITEXACT {s}: the first token {d} and one round after the prompt (its caches and logits) equal to the Python fork's", .{ label, token });
            ok = ok and dok;
        } else std.debug.print("{s}: the fixture holds no decode row (dump_prompt.py --decode)\n", .{label});
        if (!ok) failures += 1;
    }
    try check.expect(failures == 0, "{d} prefill case(s) differ from the fork", .{failures});
}

// -- drafted rounds ------------------------------------------------------------------------------------------------

/// `Supply` that, once the fixture's partials run out, gathers this rank's partial twice (timing runs only).
const Loose = struct {
    supply: *Supply,
    loose: bool = false,

    fn gather(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
        const self: *Loose = @ptrCast(@alignCast(ctx));
        if (!self.loose) return Supply.gather(self.supply, partial, words, out, stream);
        const dst: cuda.DeviceBuffer = .{ .d = self.supply.gpu.d, .ptr = out, .len = 2 * words * 4 };
        try dst.copyFrom(0, partial, words * 4, stream.handle);
        try dst.copyFrom(words * 4, partial, words * 4, stream.handle);
    }
};

fn sameAt(gpu: Gpu, fx: Fixture, ptr: Ptr, len: usize, name: []const u8, label: []const u8) bool {
    return same(gpu, fx, .{ .d = gpu.d, .ptr = ptr, .len = len }, name, label, 0);
}

pub fn spec(gpu: Gpu, dir: []const u8, cache_dir: []const u8, model_dir: []const u8, rank: usize, aot_override: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const F = dsv41.forward;
    const L = dsv41.serve_lanes;
    const dsp = dsv41.dspark;
    try check.expect(rank < 2, "rank {d}: the fixture holds two ranks", .{rank});
    const aot_dir = if (aot_override) |o| try gpa.dupe(u8, o) else try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    std.debug.print("Triton set: {s} ({d} variants)\n", .{ aot_dir, set.variants.len });
    var glue = try dsv41.attention.Glue.load(gpu.d);
    defer glue.deinit();
    var kernels = try dsv41.exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var ek = try dsv41.experts.Kernels.load(gpu.d);
    defer ek.deinit();
    var cfg = try dsv41.config.Config.read(gpa, io, model_dir);
    const fdir = try std.fs.path.join(gpa, &.{ dir, "out", "fixtures", "spec" });
    defer gpa.free(fdir);
    var fx = try Fixture.open(gpa, io, fdir);
    defer fx.deinit();
    const layers: usize = @intCast(try fx.int("layers"));
    const prompt_len: usize = @intCast(try fx.int("prompt"));
    const chunk: usize = @intCast(try fx.int("chunk"));
    const n_chunks: usize = @intCast(try fx.int("n_chunks"));
    const bucket: usize = @intCast(try fx.int("bucket"));
    const cap: usize = @intCast(try fx.int("cap"));
    const world: u32 = @intCast(try fx.int("world"));
    const rounds: usize = @intCast(try fx.int("rounds"));
    const stages: usize = @intCast(try fx.int("stages"));
    const block: usize = @intCast(try fx.int("block"));
    try check.expect(world == 2, "the fixture's world is {d}; this test supplies one other rank", .{world});
    try check.expect(stages == cfg.nextn and block == cfg.dspark_block, "the fixture's drafter: {d} stages of {d} rows", .{ stages, block });
    // the fixture's layers stand in for the tap layers
    cfg.dspark_taps = .{};
    var it = std.mem.splitScalar(u8, try fx.string("taps"), ',');
    while (it.next()) |t| {
        cfg.dspark_taps.items[cfg.dspark_taps.len] = try std.fmt.parseInt(i64, t, 10);
        cfg.dspark_taps.len += 1;
    }

    const rc_path = (try dsv41.rank_cache.RankCache.locate(gpa, io, cache_dir, rank, world)) orelse return error.NoRankCache;
    defer gpa.free(rc_path);
    var rc = try dsv41.rank_cache.RankCache.open(gpa, rc_path);
    defer rc.close();
    var w = try dsv41.weights.load(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1, layers);
    defer w.deinit();
    var ds = try dsv41.weights.loadDSpark(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1);
    defer L.freeDSpark(gpa, &ds);
    std.debug.print("loaded {d} layers and {d} DSpark stages of rank {d} from {s}\n", .{ layers, ds.blocks.len, rank, rc_path });
    const views = try gpa.alloc(dsp.Stage, ds.blocks.len);
    defer gpa.free(views);
    for (views, ds.blocks) |*v, *b| v.* = dsp.Stage.of(b);

    var rope_c: dsv41.attention.Rope = .{ .cos = try upload(gpu, fx, "cos_c"), .sin = try upload(gpu, fx, "sin_c"), .rows = @intCast(try fx.int("rope_rows")) };
    defer rope_c.cos.free();
    defer rope_c.sin.free();
    var rope_w: dsv41.attention.Rope = .{ .cos = try upload(gpu, fx, "cos_w"), .sin = try upload(gpu, fx, "sin_w"), .rows = @intCast(try fx.int("rope_rows")) };
    defer rope_w.cos.free();
    defer rope_w.sin.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    var supply: Supply = .{ .gpu = gpu, .rank = rank, .other = .empty };
    defer {
        supply.clear();
        supply.other.deinit(gpa);
    }
    var loose: Loose = .{ .supply = &supply };
    const gather: F.Gather = .{ .ctx = &loose, .call = Loose.gather };
    const drafter = try dsp.Drafter.initWith(gpa, gpu.d, dsp.dimsOf(&cfg, rank, world), &set, &glue, &ek, views, dsp.headOf(&ds, &w.head), null, .{ .ctx = &loose, .call = Loose.gather });
    defer drafter.deinit();
    try drafter.enableWide(&kernels);
    var fwd = try F.Forward.init(gpa, gpu.d, &cfg, &w, &set, &glue, &ek, world, L.max_rows, cap, bucket, rope_c, rope_w, stream);
    defer fwd.deinit();
    var seq = try L.Sequence.init(gpa, gpu.d, &fwd, &kernels, null, chunk, stream, drafter);
    defer seq.deinit(gpa);

    // the fork's gathers in order: the prompt's chunks (attention's and the MoE's a layer), its head; then each
    // draft (each stage's attention and MoE, each Markov step's bests) and each round's window (its layers, its head)
    const other: usize = 1 - rank;
    var nb: [128]u8 = undefined;
    for (0..n_chunks) |ci| for (0..layers) |li| {
        try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_C{d}.L{d}.p_attn.r{d}", .{ ci, li, other }));
        try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_C{d}.L{d}.p_moe.r{d}", .{ ci, li, other }));
    };
    try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_head.local.r{d}", .{other}));
    const prompt_gathers = supply.other.items.len;
    var marks: [32]usize = undefined; // the supply's count after each draft and each round
    for (0..rounds + 1) |k| {
        for (0..stages) |j| {
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_S{d}.s{d}.p_attn.r{d}", .{ k, j, other }));
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_S{d}.s{d}.p_moe.r{d}", .{ k, j, other }));
        }
        for (0..block) |i| try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_S{d}.mk{d}.send.r{d}", .{ k, i, other }));
        marks[2 * k] = supply.other.items.len;
        if (k == rounds) break;
        for (0..layers) |li| {
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_R{d}.L{d}.p_attn.r{d}", .{ k, li, other }));
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_R{d}.L{d}.p_moe.r{d}", .{ k, li, other }));
        }
        try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_R{d}.head.local.r{d}", .{ k, other }));
        marks[2 * k + 1] = supply.other.items.len;
    }

    const ids_bytes = try fx.bytes("prompt_ids");
    defer gpa.free(ids_bytes);
    const prompt_ids = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(ids_bytes)));
    try check.expect(prompt_ids.len == prompt_len, "the fixture's prompt has {d} ids for {d}", .{ prompt_ids.len, prompt_len });
    const op_buf = try gpa.alloc(u8, L.maxOpBytes(cap));
    defer gpa.free(op_buf);
    const ids_buf = try gpa.alloc(i64, cap);
    defer gpa.free(ids_buf);
    var failures: usize = 0;
    const D = cfg.dim;
    const V = cfg.vocab;
    const tap_bytes = cfg.dspark_taps.len * D * 2;
    const ring_bytes = (cfg.window + 16) * cfg.head_dim * 2;

    // the prompt, as announced and as every rank runs it
    {
        const label = "dsv41-lanes-spec prompt";
        const op = L.Prefill.encode(op_buf, chunk, false, true, false, prompt_ids);
        const parsed = try L.Prefill.parse(op);
        try check.expect(parsed.drafts and !parsed.engram and parsed.chunk == chunk, "{s}: the record parses back", .{label});
        const t0 = check.now(io);
        try L.Prefill.run(op, &seq, gather, ids_buf);
        try stream.synchronize();
        const ns = check.now(io) - t0;
        var ok = supply.next == prompt_gathers;
        if (!ok) std.debug.print("FAIL {s}: {d} gathers, the fork's {d}\n", .{ label, supply.next, prompt_gathers });
        ok = sameAt(gpu, fx, seq.logits, V * 4, "logits", label) and ok;
        const last_rows = prompt_len - (n_chunks - 1) * chunk;
        ok = sameAt(gpu, fx, seq.taps.ptr, last_rows * tap_bytes, try std.fmt.bufPrint(&nb, "C{d}.taps", .{n_chunks - 1}), label) and ok;
        for (seq.dc.?.rings, 0..) |r, j| ok = sameAt(gpu, fx, r.ptr, ring_bytes, try std.fmt.bufPrint(&nb, "P.ring{d}", .{j}), label) and ok;
        const first = try seq.argmax();
        const want_first: u32 = @intCast(try fx.int("first"));
        if (first != want_first) {
            std.debug.print("FAIL {s}: the first token {d}, the fork's {d}\n", .{ label, first, want_first });
            ok = false;
        }
        std.debug.print("{s}: {d} tokens in {d} chunk(s) through {d} layers in {d} ms, the taps absorbed\n", .{ label, prompt_len, n_chunks, layers, @divTrunc(ns, 1_000_000) });
        if (ok) check.pass("BITEXACT {s}: the gathers' order, the last row's logits, the last chunk's taps, the drafter's rings and the first token equal to the Python fork's", .{label}) else failures += 1;
    }

    // the rounds: draft (absorbing the last round's kept rows), verify, keep
    var prev_start: usize = 0;
    var prev_kept: usize = 0;
    var k: usize = 0;
    while (k <= rounds) : (k += 1) {
        var lb: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&lb, "dsv41-lanes-spec rank {d} round {d}", .{ rank, k });
        var ok = true;
        const token = try fx.int(try std.fmt.bufPrint(&nb, "S{d}_token", .{k}));
        const q0: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "S{d}_q0", .{k})));
        const dop = L.Draft.encode(op_buf, .{ .start = prev_start, .absorb = prev_kept, .token = token, .q0 = q0, .depth = block });
        const d = try L.Draft.parse(dop);
        try seq.draft(d.start, d.absorb, d.token, d.q0, d.depth);
        if (supply.next != marks[2 * k]) {
            std.debug.print("FAIL {s}: {d} gathers through the draft, the fork's {d}\n", .{ label, supply.next, marks[2 * k] });
            ok = false;
        }
        if (k > 0) for (seq.dc.?.rings, 0..) |r, j| {
            ok = sameAt(gpu, fx, r.ptr, ring_bytes, try std.fmt.bufPrint(&nb, "R{d}.ring{d}", .{ k - 1, j }), label) and ok;
        };
        const want_d = try fx.bytes(try std.fmt.bufPrint(&nb, "expected_S{d}.drafts", .{k}));
        defer gpa.free(want_d);
        const wd = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(want_d)));
        var dok = seq.held == block and wd.len == block;
        for (0..@min(block, wd.len)) |i| dok = dok and seq.drafts.tokens[i] == wd[i];
        if (!dok) {
            std.debug.print("FAIL {s}: drafts {any}, the fork's {any}\n", .{ label, seq.drafts.tokens[0..seq.drafts.len], wd });
            ok = false;
        }
        const want_c = try fx.bytes(try std.fmt.bufPrint(&nb, "expected_S{d}.conf", .{k}));
        defer gpa.free(want_c);
        const wc = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(want_c)));
        var worst: f32 = 0;
        for (0..@min(block, wc.len)) |i| worst = @max(worst, @abs(seq.drafts.conf[i] - wc[i]) / @max(1.0, @abs(wc[i])));
        if (worst > 1e-4) {
            std.debug.print("FAIL {s}: the confidences differ by {e} at worst\n", .{ label, worst });
            ok = false;
        }
        if (k == rounds) {
            if (ok) check.pass("BITEXACT {s}: the last round's absorb (the drafter's rings) and the drafts after it {any} equal to the Python fork's (confidences within {e})", .{ label, seq.drafts.tokens[0..block], worst }) else failures += 1;
            break;
        }

        // the window as rank 0 announces it: the pending token and the drafts (the fixture's: an oracle round's
        // drafts are the target's serial continuation, as a copy window's)
        const start: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "R{d}_start", .{k})));
        const rows: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "R{d}_rows", .{k})));
        const want_acc: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "R{d}_accepted", .{k})));
        const win_bytes = try fx.bytes(try std.fmt.bufPrint(&nb, "expected_R{d}.window", .{k}));
        defer gpa.free(win_bytes);
        const window = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(win_bytes)));
        if (start != seq.pos or start != q0 or window.len != rows or window[0] != token) {
            std.debug.print("FAIL {s}: the window at {d} (the sequence at {d})\n", .{ label, start, seq.pos });
            failures += 1;
            break;
        }
        const vop = L.Verify.encode(op_buf, seq.pos, window);
        const v = try L.Verify.parse(vop);
        var vids: [L.max_rows]i64 = undefined;
        try stream.synchronize();
        const t0 = check.now(io);
        try seq.verify(gather, v.idsInto(&vids));
        var picks: [L.max_rows]u32 = undefined;
        try seq.argmaxRows(picks[0..rows]);
        const vns = check.now(io) - t0;
        if (supply.next != marks[2 * k + 1]) {
            std.debug.print("FAIL {s}: {d} gathers through the window, the fork's {d}\n", .{ label, supply.next, marks[2 * k + 1] });
            ok = false;
        }
        ok = sameAt(gpu, fx, seq.logits, rows * V * 4, try std.fmt.bufPrint(&nb, "R{d}.logits", .{k}), label) and ok;
        ok = sameAt(gpu, fx, seq.taps.ptr, rows * tap_bytes, try std.fmt.bufPrint(&nb, "R{d}.taps", .{k}), label) and ok;
        const want_t = try fx.bytes(try std.fmt.bufPrint(&nb, "expected_R{d}.target", .{k}));
        defer gpa.free(want_t);
        const wt = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(want_t)));
        var acc: usize = 0;
        while (acc + 1 < rows and window[acc + 1] == picks[acc]) acc += 1;
        for (picks[0..rows], wt) |p, t| if (p != t) {
            std.debug.print("FAIL {s}: picks {any}, the fork's {any}\n", .{ label, picks[0..rows], wt });
            ok = false;
            break;
        };
        if (acc != want_acc) {
            std.debug.print("FAIL {s}: {d} drafts accepted, the fork's {d}\n", .{ label, acc, want_acc });
            ok = false;
        }
        // the keep, as announced (a fully accepted window needs none)
        if (acc + 1 < rows) {
            const kop = L.Keep.encode(op_buf, .{ .start = start, .kept = acc + 1 });
            const kp = try L.Keep.parse(kop);
            try seq.keep(kp.start, kp.kept);
        }
        ok = seq.pos == start + acc + 1 and ok;
        ok = (try planesSame(gpu, fx, &fwd, try std.fmt.bufPrint(&nb, "R{d}.", .{k}), label)) and ok;
        prev_start = start;
        prev_kept = acc + 1;
        if (ok) check.pass("BITEXACT {s}: drafts {any}, a {d}-row window at {d} in {d} ms: its logits, taps and picks, {d} accepted, the caches after the keep equal to the Python fork's", .{ label, seq.drafts.tokens[0..block], rows, start, @divTrunc(vns, 1_000_000), acc }) else failures += 1;
    }
    try check.expect(supply.next == supply.other.items.len, "{d} of the fork's {d} gathers ran", .{ supply.next, supply.other.items.len });

    // the window costs at this rank's layers (the gathers copy this rank's partial: timing only)
    loose.loose = true;
    var costs: [L.draft_width]f64 = undefined;
    var ids: [L.draft_width]i64 = @splat(@intCast(cfg.dspark_noise));
    const base = seq.pos;
    for (1..L.draft_width + 1) |n| {
        var best: f64 = std.math.inf(f64);
        for (0..4) |rep| {
            seq.pos = base;
            try stream.synchronize();
            const t0 = check.now(io);
            try seq.verify(gather, ids[0..n]);
            try stream.synchronize();
            if (rep > 0) best = @min(best, @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6);
        }
        costs[n - 1] = best;
    }
    var draft_ms: f64 = std.math.inf(f64);
    for (0..4) |rep| {
        seq.pos = base;
        try seq.verify(gather, ids[0..1]);
        seq.dc.?.absorbed = base;
        try stream.synchronize();
        const t0 = check.now(io);
        try seq.draft(base, 1, ids[0], base + 1, block);
        if (rep > 0) draft_ms = @min(draft_ms, @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6);
    }
    std.debug.print("dsv41-lanes-spec rank {d}: windows of 1..{d} rows {any} ms through {d} layers, a draft (absorb + {d} stages + Markov) {d:.2} ms (gathers local)\n", .{ rank, L.draft_width, costs, layers, stages, draft_ms });
    try check.expect(failures == 0, "{d} drafted step(s) differ from the fork", .{failures});
}
