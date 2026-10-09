//! `dsv41-prompt <dir> <rank-cache-dir> <model-dir> [rank] [layers]`: the DeepSeek-V4.1 prompt path (cuda_prompt.zig) on
//! tools/zig/dsv41/prompt_capture.sh's work dir: this rank's weights of the fixture's layers from the Python rank
//! cache, the fork's Triton kernels from the captured AOT set, the other rank's partials supplied from the fixture at
//! every gather; for each chunk size the fixture holds (2048: one chunk, 512: a chunk and a tail), every chunk's
//! mixes' inputs, both sublayers' streams, this rank's partials and the cache planes after it, then the last row's
//! logits, byte-equal to the fork's.

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

/// The other rank's partials in gather order from the fixture, this rank's copied in beside them; every probed
/// buffer kept under the chunk's prefix.
const Supply = struct {
    gpu: Gpu,
    rank: usize,
    rank1: std.ArrayList(cuda.DeviceBuffer),
    next: usize = 0,
    probes: std.StringArrayHashMapUnmanaged(cuda.DeviceBuffer),
    prefix: []const u8 = "",

    fn gather(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
        const self: *Supply = @ptrCast(@alignCast(ctx));
        if (self.next >= self.rank1.items.len) return error.GatherOverrun;
        const other = self.rank1.items[self.next];
        self.next += 1;
        if (other.len != words * 4) return error.GatherShape;
        const dst: cuda.DeviceBuffer = .{ .d = self.gpu.d, .ptr = out, .len = 2 * words * 4 };
        try dst.copyFrom(self.rank * words * 4, partial, words * 4, stream.handle);
        try dst.copyFrom((1 - self.rank) * words * 4, other.ptr, words * 4, stream.handle);
    }

    fn probe(ctx: *anyopaque, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) anyerror!void {
        const self: *Supply = @ptrCast(@alignCast(ctx));
        const key = try std.fmt.allocPrint(self.gpu.gpa, "{s}{s}", .{ if (std.mem.startsWith(u8, name, "head.")) "" else self.prefix, name });
        if (self.rank == 1 and std.mem.endsWith(u8, key, ".r0")) key[key.len - 1] = '1';
        var buf = try cuda.DeviceBuffer.alloc(self.gpu.d, bytes);
        try buf.copyFrom(0, ptr, bytes, stream.handle);
        try self.probes.put(self.gpu.gpa, key, buf);
    }

    fn clear(self: *Supply) void {
        for (self.rank1.items) |*b| b.free();
        self.rank1.clearRetainingCapacity();
        self.next = 0;
        var it = self.probes.iterator();
        while (it.next()) |e| {
            e.value_ptr.free();
            self.gpu.gpa.free(e.key_ptr.*);
        }
        self.probes.clearRetainingCapacity();
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

/// Whether the fixture holds `expected_<name>` (the debug probes exist only where the fixture recorded them).
fn has(gpu: Gpu, fx: Fixture, name: []const u8) bool {
    var nb: [128]u8 = undefined;
    const b = fx.bytes(std.fmt.bufPrint(&nb, "expected_{s}", .{name}) catch return false) catch return false;
    gpu.gpa.free(b);
    return true;
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

pub fn prompt(gpu: Gpu, dir: []const u8, cache_dir: []const u8, model_dir: []const u8, rank: usize, layers_cut: ?usize) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const F = dsv41.forward;
    const aot_dir = try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
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
    const fixture_layers: usize = @intCast(try cf.int("layers"));
    const layers = @min(layers_cut orelse fixture_layers, fixture_layers);
    const whole = layers == fixture_layers;
    const prompt_len: usize = @intCast(try cf.int("prompt"));
    const bucket: usize = @intCast(try cf.int("bucket"));
    const cap: usize = @intCast(try cf.int("cap"));
    const world: u32 = @intCast(try cf.int("world"));
    try check.expect(world == 2, "the fixture's world is {d}; this test supplies one other rank", .{world});

    const rc_path = (try dsv41.rank_cache.RankCache.locate(gpa, io, cache_dir, rank, world)) orelse return error.NoRankCache;
    defer gpa.free(rc_path);
    var rc = try dsv41.rank_cache.RankCache.open(gpa, rc_path);
    defer rc.close();
    const t0 = check.now(io);
    var w = try dsv41.weights.load(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1, layers);
    defer w.deinit();
    std.debug.print("loaded {d} layers of rank {d} from {s} in {d} ms\n", .{ layers, rank, rc_path, @divTrunc(check.now(io) - t0, 1_000_000) });

    var rope_c: dsv41.attention.Rope = .{ .cos = try upload(gpu, cf, "cos_c"), .sin = try upload(gpu, cf, "sin_c"), .rows = @intCast(try cf.int("rope_rows")) };
    defer rope_c.cos.free();
    defer rope_c.sin.free();
    var rope_w: dsv41.attention.Rope = .{ .cos = try upload(gpu, cf, "cos_w"), .sin = try upload(gpu, cf, "sin_w"), .rows = @intCast(try cf.int("rope_rows")) };
    defer rope_w.cos.free();
    defer rope_w.sin.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var fwd = try F.Forward.init(gpa, gpu.d, &cfg, &w, &set, &glue, &ek, world, 16, cap, bucket, rope_c, rope_w, stream);
    defer fwd.deinit();
    var pr = try dsv41.prompt.Prompt.init(gpa, gpu.d, &fwd, &kernels, @min(dsv41.prompt.max_chunk, prompt_len));
    defer pr.deinit();
    var supply: Supply = .{ .gpu = gpu, .rank = rank, .rank1 = .empty, .probes = .empty };
    defer {
        supply.clear();
        supply.rank1.deinit(gpa);
        supply.probes.deinit(gpa);
    }
    fwd.probe = .{ .ctx = &supply, .call = Supply.probe };
    const gather: F.Gather = .{ .ctx = &supply, .call = Supply.gather };
    const ids_bytes = try cf.bytes("prompt_ids");
    defer gpa.free(ids_bytes);
    const prompt_ids = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(ids_bytes)));
    try check.expect(prompt_ids.len == prompt_len, "the fixture's prompt has {d} ids for {d}", .{ prompt_ids.len, prompt_len });

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
        const label = try std.fmt.bufPrint(&label_buf, "dsv41-prompt rank {d} chunk {d}", .{ rank, chunk });
        const n_chunks: usize = @intCast(try fx.int("n_chunks"));
        supply.clear();
        try resetCaches(&fwd);
        // every chunk's other-rank partials in gather order, then the head's
        for (0..n_chunks) |ci| for (0..layers) |li| {
            try supply.rank1.append(gpa, try upload(gpu, fx, try std.fmt.bufPrint(&nb, "expected_C{d}.L{d}.p_attn.r{d}", .{ ci, li, other })));
            try supply.rank1.append(gpa, try upload(gpu, fx, try std.fmt.bufPrint(&nb, "expected_C{d}.L{d}.p_moe.r{d}", .{ ci, li, other })));
        };
        if (whole) try supply.rank1.append(gpa, try upload(gpu, fx, try std.fmt.bufPrint(&nb, "expected_head.local.r{d}", .{other})));
        var ok = true;
        var logits: ?Ptr = null;
        var prefix_buf: [16]u8 = undefined;
        var s: usize = 0;
        var ci: usize = 0;
        while (s < prompt_len) : (ci += 1) {
            const e = @min(prompt_len, s + chunk);
            supply.prefix = try std.fmt.bufPrint(&prefix_buf, "C{d}.", .{ci});
            logits = try pr.chunk(gather, prompt_ids[s..e], s, .{ .logits = whole and e == prompt_len }, stream);
            try stream.synchronize();
            // the caches after this chunk: every row but the compressed planes' scratch row
            for (0..layers) |li| {
                ok = same(gpu, fx, fwd.rings[li], try std.fmt.bufPrint(&nb, "C{d}.L{d}.ring", .{ ci, li }), label, 0) and ok;
                if (fwd.planes[li]) |*p| {
                    const kept = p.rows - 1;
                    const hd = cfg.head_dim;
                    ok = same(gpu, fx, p.comp_codes, try std.fmt.bufPrint(&nb, "C{d}.L{d}.comp_codes", .{ ci, li }), label, kept * hd / 2) and ok;
                    ok = same(gpu, fx, p.comp_scales, try std.fmt.bufPrint(&nb, "C{d}.L{d}.comp_scales", .{ ci, li }), label, kept * hd / 16) and ok;
                    if (cfg.isIndexSource(li)) {
                        ok = same(gpu, fx, p.idxk_codes, try std.fmt.bufPrint(&nb, "C{d}.L{d}.idxk_codes", .{ ci, li }), label, kept * cfg.idx_dim / 2) and ok;
                        ok = same(gpu, fx, p.idxk_scales, try std.fmt.bufPrint(&nb, "C{d}.L{d}.idxk_scales", .{ ci, li }), label, kept * cfg.idx_dim / 32) and ok;
                    }
                    if (cfg.ratio(li) > 1) {
                        ok = same(gpu, fx, p.raw_kv, try std.fmt.bufPrint(&nb, "C{d}.L{d}.raw_kv", .{ ci, li }), label, 0) and ok;
                        ok = same(gpu, fx, p.raw_score, try std.fmt.bufPrint(&nb, "C{d}.L{d}.raw_score", .{ ci, li }), label, 0) and ok;
                    }
                }
            }
            s = e;
        }
        supply.prefix = "";
        // every probed buffer (the mixes' inputs, both ranks' partials, the streams, the head's)
        var pit = supply.probes.iterator();
        while (pit.next()) |pe| {
            if (!whole and std.mem.startsWith(u8, pe.key_ptr.*, "head.")) continue;
            if (std.mem.indexOf(u8, pe.key_ptr.*, ".dbg.") != null and !has(gpu, fx, pe.key_ptr.*)) continue;
            ok = same(gpu, fx, pe.value_ptr.*, pe.key_ptr.*, label, 0) and ok;
        }
        if (whole) {
            if (logits) |lg| {
                const view: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = lg, .len = cfg.vocab * 4 };
                ok = same(gpu, fx, view, "logits", label, 0) and ok;
            } else {
                std.debug.print("FAIL {s}: no logits\n", .{label});
                ok = false;
            }
        }
        if (ok) check.pass("BITEXACT {s}: {d} chunk(s) over {d} tokens through {d} of {d} layers: the mixes, partials, streams and caches{s} equal to the Python fork's", .{ label, n_chunks, prompt_len, layers, fixture_layers, if (whole) " and the last row's logits" else "" });
        if (!ok) failures += 1;
    }
    try check.expect(failures == 0, "{d} prompt case(s) differ from the fork", .{failures});
}
