//! `dsv41-lanes-prefill <dir> <rank-cache-dir> <model-dir> <rank> [aot-dir]`: the served engine's steps
//! (cuda_lanes.zig) at one rank on dump_prompt.py's fixture (prompt_capture.sh's work dir): for each chunk size the
//! fixture holds, the prefill record (`Prefill.encode`, as rank 0 announces it) run by `Prefill.run` on a
//! `Sequence` (as every rank runs it), the other rank's partials supplied from the fixture at every gather in the
//! fork's order; the caches after the prompt and the last row's logits byte-equal to the fork's. A fixture dumped
//! with `--decode` then has the first token (the logits' argmax) equal to the fork's and one round (`Sequence.round`,
//! the decode layer loop on the same caches) byte-equal in its logits and caches. A fixture dumped with `--engram`
//! runs Engram (TF_DS_ENGRAM, TF_DS_TOKEN_MAP).

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
    var seq = try L.Sequence.init(gpa, gpu.d, &fwd, &kernels, eng, rows, stream);
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
        const op = L.Prefill.encode(op_buf, chunk, eng != null, prompt_ids);
        const parsed = try L.Prefill.parse(op);
        try check.expect(parsed.chunk == chunk and parsed.engram == (eng != null) and parsed.len() == prompt_len, "{s}: the record parses back", .{label});
        try stream.synchronize();
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
