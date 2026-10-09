//! `dsv41-forward <dir> <rank-cache-dir> <model-dir> [rank] [layers] [aot-dir]`: the DeepSeek-V4.1 layer loop
//! (cuda_forward.zig) on tools/zig/dsv41/forward_capture.sh's work dir: this rank's weights of the fixture's layers
//! from the Python rank cache (`rank`, default 0), the cache planes from the fixture, the fork's Triton kernels from
//! the captured AOT set, the other rank's partials supplied from the fixture at every gather; the mixes' inputs, both
//! sublayers' h, this rank's partials, the logits (every row, then the last row alone) and the cache planes after,
//! byte-equal to the fork's. `layers` (default the fixture's) runs the loop over the first layers only, comparing
//! their sublayers and skipping the logits: for a machine without room for every layer's weights.
//!
//! A fixture dumped with `--engram` runs Engram on its layers (cuda_engram_apply.zig): the tables from TF_DS_ENGRAM,
//! the token map from TF_DS_TOKEN_MAP (default `<aot-dir>/token_map.json`), the host ids the prompt's then the
//! window's; the other rank's Engram partial is supplied at its gather like the sublayers' and the gated streams
//! (`L<i>.h_engram`) and this rank's partial are compared too.

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

/// The other rank's partials, in gather order, from the fixture; rank 0's own copied in beside them. Also keeps
/// every probed buffer for the comparison after the run.
const Supply = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    rank: usize, // this rank: its partial goes to its slot of the gather, the other's from the fixture
    rank1: std.ArrayList(cuda.DeviceBuffer), // the other rank's partials, gather order
    next: usize = 0,
    probes: std.StringArrayHashMapUnmanaged(cuda.DeviceBuffer),

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
        // the Forward names its own partials `.r0`; the fixture's are by rank
        const key = try self.gpu.gpa.dupe(u8, name);
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

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

/// The Engram a fixture dumped with `--engram` needs (its common params' `engram`), from TF_DS_ENGRAM and
/// TF_DS_TOKEN_MAP (default `<aot_dir>/token_map.json`); null for a fixture without.
pub fn engramFor(gpu: Gpu, common: Fixture, cfg: *const dsv41.config.Config, w: *const dsv41.weights.Weights, set: *const cuda.aot.Set, aot_dir: []const u8, max_rows: usize) !?*dsv41.engram_apply.Engram {
    if ((common.int("engram") catch 0) == 0) return null;
    const dir = getenv("TF_DS_ENGRAM") orelse {
        std.debug.print("FAIL the fixture runs Engram: set TF_DS_ENGRAM to the tables' dir\n", .{});
        return error.NoEngramTables;
    };
    const tm = if (getenv("TF_DS_TOKEN_MAP")) |p| try gpu.gpa.dupe(u8, p) else try std.fs.path.join(gpu.gpa, &.{ aot_dir, "token_map.json" });
    defer gpu.gpa.free(tm);
    std.debug.print("Engram: tables {s}, token map {s}\n", .{ dir, tm });
    return try dsv41.engram_apply.Engram.init(gpu.gpa, gpu.io, gpu.d, cfg, w, set, dir, tm, max_rows, 32);
}

fn same(gpu: Gpu, fx: Fixture, got: cuda.DeviceBuffer, name: []const u8, label: []const u8, limit: usize) bool {
    var nb: [96]u8 = undefined;
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
    var what: [160]u8 = undefined;
    check.sameBytes(std.fmt.bufPrint(&what, "{s} {s}", .{ label, name }) catch return false, bytes[0..n], want[0..n]) catch return false;
    return true;
}

/// `aot_override`: a captured Triton set to use instead of the fixture's own (a serving run's full set).
pub fn forward(gpu: Gpu, dir: []const u8, cache_dir: []const u8, model_dir: []const u8, rank: usize, layers_cut: ?usize, aot_override: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const F = dsv41.forward;
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
    const cdir = try std.fs.path.join(gpa, &.{ dir, "out", "fixtures", "fwd_common" });
    defer gpa.free(cdir);
    var cf = try Fixture.open(gpa, io, cdir);
    defer cf.deinit();
    const fixture_layers: usize = @intCast(try cf.int("layers"));
    const layers = @min(layers_cut orelse fixture_layers, fixture_layers);
    const whole = layers == fixture_layers;
    const start: usize = @intCast(try cf.int("start"));
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
    var supply: Supply = .{ .gpu = gpu, .stream = stream, .rank = rank, .rank1 = .empty, .probes = .empty };
    defer {
        supply.clear();
        supply.rank1.deinit(gpa);
        supply.probes.deinit(gpa);
    }
    fwd.probe = .{ .ctx = &supply, .call = Supply.probe };
    const gather: F.Gather = .{ .ctx = &supply, .call = Supply.gather };
    const eng = try engramFor(gpu, cf, &cfg, &w, &set, aot_dir, 16);
    defer if (eng) |e| e.deinit();
    if (eng) |e| {
        fwd.engram = e.hook();
        e.probe = fwd.probe;
    }
    // the sequence's host ids: the prompt's, then the window's
    const prompt_bytes = try cf.bytes("prompt_ids");
    defer gpa.free(prompt_bytes);
    const prompt_ids = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(prompt_bytes)));
    var history: std.ArrayList(i32) = .empty;
    defer history.deinit(gpa);

    const rows_text = try gpa.dupe(u8, try cf.string("rows"));
    defer gpa.free(rows_text);
    var it = std.mem.splitScalar(u8, rows_text, ',');
    var failures: usize = 0;
    while (it.next()) |text| {
        const n = try std.fmt.parseInt(usize, text, 10);
        var nb: [96]u8 = undefined;
        const fdir = try std.fs.path.join(gpa, &.{ dir, "out", "fixtures", try std.fmt.bufPrint(&nb, "fwd_n{d}", .{n}) });
        defer gpa.free(fdir);
        var fx = try Fixture.open(gpa, io, fdir);
        defer fx.deinit();
        var label_buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "dsv41-forward rank {d} n={d}", .{ rank, n });
        const other: usize = 1 - rank;
        for ([_]bool{ true, false }) |all| {
            supply.clear();
            // the caches before the window, every layer
            for (0..layers) |li| {
                const ring = try fx.bytes(try std.fmt.bufPrint(&nb, "L{d}.ring", .{li}));
                defer gpa.free(ring);
                try fwd.rings[li].upload(0, ring);
                if (fwd.planes[li]) |*p| {
                    inline for (.{ "comp_codes", "comp_scales", "idxk_codes", "idxk_scales", "raw_kv", "raw_score" }) |plane| {
                        // a ratio-1 source keeps no raw rows and a non-index source no index keys: the fixture omits them
                        const wanted = if (std.mem.startsWith(u8, plane, "raw_")) cfg.ratio(li) > 1 else if (std.mem.startsWith(u8, plane, "idxk_")) cfg.isIndexSource(li) else true;
                        if (wanted) {
                            const bytes = try fx.bytes(try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ li, plane }));
                            defer gpa.free(bytes);
                            try @field(p, plane).upload(0, bytes);
                        }
                    }
                }
            }
            // the other rank's partials in gather order
            for (0..layers) |li| {
                if (eng != null and w.layers[li].engram_wkv != null)
                    try supply.rank1.append(gpa, try upload(gpu, fx, try std.fmt.bufPrint(&nb, "expected_L{d}.p_engram.r{d}", .{ li, other })));
                try supply.rank1.append(gpa, try upload(gpu, fx, try std.fmt.bufPrint(&nb, "expected_L{d}.p_attn.r{d}", .{ li, other })));
                try supply.rank1.append(gpa, try upload(gpu, fx, try std.fmt.bufPrint(&nb, "expected_L{d}.p_moe.r{d}", .{ li, other })));
            }
            {
                // the head's: every row's, or the last row's
                const bytes = try fx.bytes(try std.fmt.bufPrint(&nb, "expected_head.local.r{d}", .{other}));
                defer gpa.free(bytes);
                const vh = bytes.len / n / 4;
                const tail = if (all) bytes else bytes[(n - 1) * vh * 4 ..];
                try supply.rank1.append(gpa, try cuda.DeviceBuffer.fromHost(gpu.d, tail));
            }
            const ids_bytes = try fx.bytes("ids");
            defer gpa.free(ids_bytes);
            const ids = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(ids_bytes)));
            history.clearRetainingCapacity();
            for (prompt_ids) |t| try history.append(gpa, @intCast(t));
            for (ids) |t| try history.append(gpa, @intCast(t));
            if (eng) |e| e.history = history.items;
            const logits = try fwd.rows(gather, ids, start, n, all, stream);
            try stream.synchronize();
            var ok = true;
            const vocab = cfg.vocab;
            if (all) {
                if (whole) {
                    const view: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = logits, .len = n * vocab * 4 };
                    ok = same(gpu, fx, view, "logits", label, 0) and ok;
                }
                // every probed buffer against the fixture's (the head's only when every layer ran)
                var pit = supply.probes.iterator();
                while (pit.next()) |e| {
                    if (!whole and std.mem.startsWith(u8, e.key_ptr.*, "head.")) continue;
                    ok = same(gpu, fx, e.value_ptr.*, e.key_ptr.*, label, 0) and ok;
                }
                // the caches after: every row but the compressed planes' scratch row
                for (0..layers) |li| {
                    ok = same(gpu, fx, fwd.rings[li], try std.fmt.bufPrint(&nb, "L{d}.ring", .{li}), label, 0) and ok;
                    if (fwd.planes[li]) |*p| {
                        const kept = p.rows - 1;
                        const hd = cfg.head_dim;
                        ok = same(gpu, fx, p.comp_codes, try std.fmt.bufPrint(&nb, "L{d}.comp_codes", .{li}), label, kept * hd / 2) and ok;
                        ok = same(gpu, fx, p.comp_scales, try std.fmt.bufPrint(&nb, "L{d}.comp_scales", .{li}), label, kept * hd / 16) and ok;
                        if (cfg.isIndexSource(li)) {
                            ok = same(gpu, fx, p.idxk_codes, try std.fmt.bufPrint(&nb, "L{d}.idxk_codes", .{li}), label, kept * cfg.idx_dim / 2) and ok;
                            ok = same(gpu, fx, p.idxk_scales, try std.fmt.bufPrint(&nb, "L{d}.idxk_scales", .{li}), label, kept * cfg.idx_dim / 32) and ok;
                        }
                        if (cfg.ratio(li) > 1) {
                            ok = same(gpu, fx, p.raw_kv, try std.fmt.bufPrint(&nb, "L{d}.raw_kv", .{li}), label, 0) and ok;
                            ok = same(gpu, fx, p.raw_score, try std.fmt.bufPrint(&nb, "L{d}.raw_score", .{li}), label, 0) and ok;
                        }
                    }
                }
                if (ok) check.pass("BITEXACT {s} all_logits: {d} of {d} layers' mixes, partials and streams{s} and the caches equal to the Python fork's (positions {d}..)", .{ label, layers, fixture_layers, if (whole) ", the logits" else "", start });
            } else if (whole) {
                // the last row's logits alone equal the last row of every row's
                const want_all = try fx.bytes("expected_logits");
                defer gpa.free(want_all);
                const want = want_all[(n - 1) * vocab * 4 ..];
                const got = try check.download(gpu, .{ .d = gpu.d, .ptr = logits, .len = vocab * 4 });
                defer gpa.free(got);
                check.sameBytes(try std.fmt.bufPrint(&nb, "{s} last-row logits", .{label}), got, want) catch {
                    ok = false;
                };
                if (ok) check.pass("BITEXACT {s} last row: the head on the last row alone equals its row of every row's logits", .{label});
            }
            if (!ok) failures += 1;
        }
    }
    try check.expect(failures == 0, "{d} forward case(s) differ from the fork", .{failures});
}
