//! `dsv41-prompt-time`: the served prompt path's speed on one rank, without the other rank or a fixture. The rank's
//! first `layers` layers from its rank cache, the RoPE tables of the Triton set's `rope/` dir, Engram when
//! TF_DS_ENGRAM names the tables, a synthetic prompt of `tokens` ids (a fixed pseudo-random draw a run), and a
//! gather that stands the rank's own partial in for the other rank's. Every chunk is profiled (`Prompt.Profile`);
//! the output is timing only, its bits are not checked. TF_DS_REPLAY=1 runs CED bounded replay (the upper layers
//! over the prompt's last window rows); TF_DS_PROFILE=0 times the chunks without the per-chunk profile.
//! TF_DS_PREFILL_CHUNK sets the chunk rows (default 2048). TF_DS_COMPARE_CHUNK=<rows> then prefills run 0's ids
//! from cleared caches at both chunk sizes and checks that the last row's logits and every layer's ring and planes
//! are byte-equal between them (PASS/FAIL): the prompt path's rows do not depend on how the prompt is chunked.
//! TF_DS_FIXTURE_CONE=1 compares cold full/cone state instead of timing. TF_DS_FIXTURE_TP2=1 uses real NCCL
//! gathers and requires explicit TF_TP_MASTER/TF_TP_PORT; launch both ranks with identical arguments except rank.
//! TF_DS_FIXTURE_DSPARK=1 additionally checks the original per-chunk absorption and final draft rings.
//! TF_DS_FIXTURE_FUTURE=1 checks 512 greedy outputs and draft/accept traces after both passes;
//! TF_DS_FIXTURE_ORACLE_OUT names the rank-specific JSON output and its .input.json input export.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");

const Gpu = check.Gpu;
const Ptr = cuda.abi.DevicePtr;

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

/// The other rank's partial: this rank's own (the bits differ from a real pair; the shapes and launches do not).
fn selfGather(_: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
    const d = bound.?;
    const dst: cuda.DeviceBuffer = .{ .d = d, .ptr = out, .len = 2 * words * 4 };
    try dst.copyFrom(0, partial, words * 4, stream.handle);
    try dst.copyFrom(words * 4, partial, words * 4, stream.handle);
}

fn tpGather(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
    const comm: *cuda.comm.Comm = @ptrCast(@alignCast(ctx));
    try comm.allGather(partial, out, words, .f32, stream.handle);
}

var bound: ?*const cuda.Driver = null;

fn loadRope(gpu: Gpu, dir: []const u8, name: []const u8, rows: usize) !dsv41.attention.Rope {
    const half = 32;
    const path = try std.fmt.allocPrint(gpu.gpa, "{s}/rope/{s}_{d}.bin", .{ dir, name, rows });
    defer gpu.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 31));
    defer gpu.gpa.free(bytes);
    if (bytes.len != 2 * rows * half * 4) return error.RopeRowsShort;
    var cos = try cuda.DeviceBuffer.fromHost(gpu.d, bytes[0 .. rows * half * 4]);
    errdefer cos.free();
    const sin = try cuda.DeviceBuffer.fromHost(gpu.d, bytes[rows * half * 4 ..]);
    return .{ .cos = cos, .sin = sin, .rows = rows };
}

pub fn run(gpu: Gpu, cache_dir: []const u8, model_dir: []const u8, aot_dir: []const u8, rank: usize, layers: usize, tokens: usize, runs: usize) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const L = dsv41.serve_lanes;
    bound = gpu.d;
    var set = try cuda.aot.Set.load(gpa, io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var glue = try dsv41.attention.Glue.load(gpu.d);
    defer glue.deinit();
    var kernels = try dsv41.exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var ek = try dsv41.experts.Kernels.load(gpu.d);
    defer ek.deinit();
    var cfg = try dsv41.config.Config.read(gpa, io, model_dir);
    const world: u32 = 2;
    const rc_path = (try dsv41.rank_cache.RankCache.locate(gpa, io, cache_dir, rank, world)) orelse return error.NoRankCache;
    defer gpa.free(rc_path);
    var rc = try dsv41.rank_cache.RankCache.open(gpa, rc_path);
    defer rc.close();
    var w = try dsv41.weights.load(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1, layers);
    defer w.deinit();
    std.debug.print("loaded {d} layers of rank {d}\n", .{ w.layers.len, rank });
    const cap = std.mem.alignForward(usize, tokens + L.extra + 64, 4096);
    // the 1M-context tables (any table at least `cap` rows serves; the set's dir holds that one)
    const rows: usize = L.ropeRows(1 << 20);
    var rope_c = try loadRope(gpu, aot_dir, "rope_c", rows);
    defer rope_c.cos.free();
    defer rope_c.sin.free();
    var rope_w = try loadRope(gpu, aot_dir, "rope_w", rows);
    defer rope_w.cos.free();
    defer rope_w.sin.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var fwd = try dsv41.forward.Forward.init(gpa, gpu.d, &cfg, &w, &set, &glue, &ek, world, L.max_rows, cap, cap, rope_c, rope_w, stream);
    defer fwd.deinit();
    var eng: ?*dsv41.engram_apply.Engram = null;
    if (getenv("TF_DS_ENGRAM")) |dir| {
        const tm = if (getenv("TF_DS_TOKEN_MAP")) |p| try gpa.dupe(u8, p) else try std.fs.path.join(gpa, &.{ aot_dir, "token_map.json" });
        defer gpa.free(tm);
        const most: usize = if (getenv("TF_DS_PREFILL_CHUNK")) |v| try std.fmt.parseInt(usize, v, 10) else L.default_chunk;
        const cmp: usize = if (getenv("TF_DS_COMPARE_CHUNK")) |v| try std.fmt.parseInt(usize, v, 10) else 0;
        eng = try dsv41.engram_apply.Engram.init(gpa, io, gpu.d, &cfg, &w, &set, dir, tm, @max(L.max_rows, most, cmp), 32);
    }
    defer if (eng) |e| e.deinit();
    const chunk: usize = if (getenv("TF_DS_PREFILL_CHUNK")) |v| try std.fmt.parseInt(usize, v, 10) else L.default_chunk;
    const other: ?usize = if (getenv("TF_DS_COMPARE_CHUNK")) |v| try std.fmt.parseInt(usize, v, 10) else null;
    const replay = std.mem.eql(u8, getenv("TF_DS_REPLAY") orelse "0", "1");
    const tp2 = std.mem.eql(u8, getenv("TF_DS_FIXTURE_TP2") orelse "0", "1");
    var comm: ?cuda.comm.Comm = null;
    defer if (comm) |*c| c.deinit();
    if (tp2) {
        if (!std.mem.eql(u8, getenv("TF_DS_FIXTURE_CONE") orelse "0", "1")) return error.BadConeFixture;
        var master: [4]u8 = undefined;
        var parts = std.mem.splitScalar(u8, getenv("TF_TP_MASTER") orelse return error.BadConeFixture, '.');
        for (&master) |*part| part.* = try std.fmt.parseInt(u8, parts.next() orelse return error.BadConeFixture, 10);
        if (parts.next() != null) return error.BadConeFixture;
        const port = try std.fmt.parseInt(u16, getenv("TF_TP_PORT") orelse return error.BadConeFixture, 10);
        comm = try cuda.comm.Comm.init(@intCast(rank), world, master, port, 60000);
    }
    const with_spark = std.mem.eql(u8, getenv("TF_DS_FIXTURE_DSPARK") orelse "0", "1");
    if (with_spark) {
        if (!tp2 or cfg.nextn != 3 or cfg.dspark_taps.len != 3) return error.BadConeFixture;
        for (cfg.dspark_taps.slice(), 0..) |tap, i| if (tap != 37 + @as(i64, @intCast(i))) return error.BadConeFixture;
    }
    var ds: ?dsv41.weights.DSpark = null;
    defer if (ds) |*v| v.deinit(gpa);
    var stages: []dsv41.dspark.Stage = &.{};
    defer if (stages.len > 0) gpa.free(stages);
    var spark: ?*dsv41.dspark.Drafter = null;
    defer if (spark) |v| v.deinit();
    if (with_spark) {
        ds = try dsv41.weights.loadDSpark(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1);
        stages = try gpa.alloc(dsv41.dspark.Stage, ds.?.blocks.len);
        for (stages, ds.?.blocks) |*stage, *block| stage.* = dsv41.dspark.Stage.of(block);
        const dims = dsv41.dspark.dimsOf(&cfg, rank, world);
        spark = try dsv41.dspark.Drafter.init(gpa, gpu.d, dims, &set, &glue, &ek, stages, dsv41.dspark.headOf(&ds.?, &w.head), &comm.?);
        try spark.?.enableWide(&kernels);
    }
    var seq = try L.Sequence.init(gpa, gpu.d, &fwd, &kernels, eng, @max(chunk, other orelse 0), stream, spark);
    defer seq.deinit(gpa);
    if (!std.mem.eql(u8, getenv("TF_DS_PROFILE") orelse "1", "0")) try seq.pr.enableProfile();
    const gather: dsv41.forward.Gather = if (comm) |*c| .{ .ctx = c, .call = tpGather } else .{ .ctx = &seq, .call = selfGather };
    const ids = try gpa.alloc(i64, tokens);
    defer gpa.free(ids);
    // Fixture-only dependency-cone experiment; no serving option or production dispatch changes.
    if (std.mem.eql(u8, getenv("TF_DS_FIXTURE_CONE") orelse "0", "1")) {
        if (seq.pr.profile) |*p| {
            p.deinit();
            seq.pr.profile = null;
        }
        if (replay or chunk != L.default_chunk or other != null or runs != 1 or
            cfg.n_layers != 40 or w.layers.len != 40 or cfg.window != 128 or
            seq.pr.pk.gu != .mmb or seq.pr.pk.down != .rowscp or
            !std.mem.eql(u8, getenv("TF_DS_KEEP") orelse "0", "0")) return error.BadConeFixture;
        for (20..40) |li| {
            if (cfg.ratio(li) != 1 or (li > 20 and cfg.isKvSource(li)) or cfg.isEngram(li)) return error.BadConeFixture;
        }
        if (!cfg.isKvSource(20)) return error.BadConeFixture;
        var prng = std.Random.DefaultPrng.init(0x5eed);
        for (ids) |*id| id.* = prng.random().intRangeLessThan(i64, 3, @intCast(cfg.vocab - 1024));
        return coneFixture(gpu, &fwd, &seq, gather, ids, chunk, stream, tp2);
    }
    for (0..runs) |r| {
        // a fresh draw a run: the Engram rows of one run are not in the page cache for the next
        var prng = std.Random.DefaultPrng.init(0x5eed + r);
        for (ids) |*id| id.* = prng.random().intRangeLessThan(i64, 3, @intCast(cfg.vocab - 1024));
        try stream.synchronize();
        const t0 = check.now(io);
        try seq.prefill(gather, ids, chunk, replay);
        try stream.synchronize();
        const ms = @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6;
        std.debug.print("RESULT dsv41-prompt-time run {d}: {d} tokens, {d} layers, Engram {s}, replay {s}, chunk {d}: {d:.0} ms, {d:.0} tok/s\n", .{ r, tokens, w.layers.len, if (eng != null) "on" else "off", if (replay) "on" else "off", chunk, ms, @as(f64, @floatFromInt(tokens)) * 1000 / ms });
    }
    const c2 = other orelse return;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    for (ids) |*id| id.* = prng.random().intRangeLessThan(i64, 3, @intCast(cfg.vocab - 1024));
    if (seq.pr.profile) |*p| {
        p.deinit();
        seq.pr.profile = null;
    }
    // TF_DS_COMPARE_ROWS=1: every sublayer's rows hashed in both passes, the first sublayer whose rows differ named
    var rows_a: RowLog = .{ .gpa = gpa, .seq = &seq };
    defer rows_a.deinit();
    var rows_b: RowLog = .{ .gpa = gpa, .seq = &seq };
    defer rows_b.deinit();
    const by_rows = std.mem.eql(u8, getenv("TF_DS_COMPARE_ROWS") orelse "0", "1");
    if (by_rows) fwd.probe = .{ .ctx = &rows_a, .call = RowLog.call };
    var want = try digests(gpu, &fwd, &seq, gather, ids, chunk, replay, stream);
    defer want.deinit(gpa);
    if (by_rows) fwd.probe = .{ .ctx = &rows_b, .call = RowLog.call };
    var got = try digests(gpu, &fwd, &seq, gather, ids, c2, replay, stream);
    defer got.deinit(gpa);
    fwd.probe = null;
    if (by_rows) rows_a.report(&rows_b);
    var bad: usize = 0;
    for (want.items, got.items, 0..) |a, b, i| if (a != b) {
        var nb: [64]u8 = undefined;
        if (bad < 8) std.debug.print("  differs: {s}\n", .{digestName(&fwd, i, &nb)});
        bad += 1;
    };
    try check.expect(bad == 0, "dsv41-prompt-time chunk {d} vs {d}: {d} tokens, replay {s}: {d} of {d} buffers differ", .{ chunk, c2, tokens, if (replay) "on" else "off", bad, want.items.len });
    check.pass("dsv41-prompt-time chunk {d} vs {d}: {d} tokens, {d} layers, replay {s}: logits, rings and planes byte-equal ({d} buffers)", .{ chunk, c2, tokens, w.layers.len, if (replay) "on" else "off", want.items.len });
}

const plane_fields = .{ "comp_codes", "comp_scales", "idxk_codes", "idxk_scales", "raw_kv", "raw_score" };

/// Which buffer digest `i` of `digests` is.
fn digestName(fwd: *const dsv41.forward.Forward, i: usize, buf: []u8) []const u8 {
    if (i == 0) return "logits";
    const k = i - 1;
    if (k < fwd.rings.len) return std.fmt.bufPrint(buf, "layer {d}'s ring", .{k}) catch "a ring";
    var j = k - fwd.rings.len;
    for (fwd.planes, 0..) |pl, li| if (pl != null) {
        if (j < plane_fields.len) {
            inline for (plane_fields, 0..) |f, fi| if (fi == j) return std.fmt.bufPrint(buf, "layer {d}'s {s}", .{ li, f }) catch "a plane";
        }
        j -= plane_fields.len;
    };
    return "a plane";
}

/// From cleared caches (as Forward.init leaves them): the prompt at `chunk` rows a chunk, then the digests of the
/// last row's logits, every layer's ring and every kv source's planes.
fn digests(gpu: Gpu, fwd: *dsv41.forward.Forward, seq: *dsv41.serve_lanes.Sequence, gather: dsv41.forward.Gather, ids: []const i64, chunk: usize, replay: bool, stream: cuda.Stream) !std.ArrayList(u64) {
    const gpa = gpu.gpa;
    for (fwd.rings) |r| try r.fill8(0, stream.handle);
    for (fwd.planes) |pl| if (pl) |x| {
        inline for (plane_fields) |f| try @field(x, f).fill8(if (comptime std.mem.eql(u8, f, "idxk_scales")) 127 else 0, stream.handle);
    };
    try seq.prefill(gather, ids, chunk, replay);
    try stream.synchronize();
    var out: std.ArrayList(u64) = .empty;
    errdefer out.deinit(gpa);
    const v = fwd.cfg.vocab;
    try out.append(gpa, try digest(gpu, .{ .d = gpu.d, .ptr = seq.logits, .len = v * 4 }));
    for (fwd.rings) |r| try out.append(gpa, try digest(gpu, r));
    for (fwd.planes) |pl| if (pl) |x| {
        inline for (plane_fields) |f| try out.append(gpa, try digest(gpu, @field(x, f)));
    };
    return out;
}

/// Compare actual cache/logit bytes under selfGather or opt-in real TP2 NCCL. This tests the suffix property,
/// Optional DSpark checks absorbed rings and valid last-absorb inputs; future checks extend this through 512 outputs.
fn coneFixture(gpu: Gpu, fwd: *dsv41.forward.Forward, seq: *dsv41.serve_lanes.Sequence, gather: dsv41.forward.Gather, ids: []const i64, chunk: usize, stream: cuda.Stream, tp2: bool) !void {
    const future = std.mem.eql(u8, getenv("TF_DS_FIXTURE_FUTURE") orelse "0", "1");
    if (future and (seq.spark == null or getenv("TF_DS_FIXTURE_ORACLE_OUT") == null)) return error.BadConeFixture;
    const input_sha = if (future) try exportOracleInput(gpu, ids) else @as([64]u8, @splat(0));
    const halo: usize = 144 + 20 * 127;
    const cut = (ids.len -| halo) / chunk * chunk;
    if (cut == 0) return error.ConeFixtureNeedsLongerPrompt;
    var buffers: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer buffers.deinit(gpu.gpa);
    var want: std.ArrayList([]u8) = .empty;
    defer {
        for (want.items) |bytes| gpu.gpa.free(bytes);
        want.deinit(gpu.gpa);
    }
    try clearConeCaches(fwd, stream);
    try clearDraftCaches(seq, stream);
    try seq.prefill(gather, ids, chunk, false);
    try stream.synchronize();
    const baseline_meta = .{ seq.pos, seq.held, seq.verify_rows, seq.verify_start, seq.tap_rows, if (seq.dc) |c| c.absorbed else 0 };
    try buffers.append(gpu.gpa, .{ .d = gpu.d, .ptr = seq.logits, .len = fwd.cfg.vocab * 4 });
    for (fwd.rings) |r| try buffers.append(gpu.gpa, r);
    for (fwd.planes) |pl| if (pl) |x| {
        inline for (plane_fields) |f| try buffers.append(gpu.gpa, @field(x, f));
    };
    if (seq.dc) |c| {
        for (c.rings) |ring| try buffers.append(gpu.gpa, ring);
        const tail = ids.len % chunk;
        const valid: usize = @min(if (tail == 0) chunk else tail, 144);
        const cols = fwd.cfg.dspark_taps.len * fwd.cfg.dim;
        try buffers.append(gpu.gpa, .{ .d = gpu.d, .ptr = seq.taps.ptr + ((if (tail == 0) chunk else tail) - valid) * cols * 2, .len = valid * cols * 2 });
        const sp = seq.spark.?;
        inline for (.{ "main_x", "main_y" }) |field| try buffers.append(gpu.gpa, .{ .d = gpu.d, .ptr = @field(sp, field).ptr, .len = valid * fwd.cfg.dim * 2 });
        inline for (.{ "abs_pos", "abs_slots" }) |field| try buffers.append(gpu.gpa, .{ .d = gpu.d, .ptr = @field(sp, field).ptr, .len = valid * 8 });
    }
    for (buffers.items) |buffer| {
        const bytes = try check.download(gpu, buffer);
        errdefer gpu.gpa.free(bytes);
        try want.append(gpu.gpa, bytes);
    }
    var full_future: ?Future = if (future) try future512(gpu, seq, gather) else null;
    defer if (full_future) |*value| value.trace.deinit(gpu.gpa);
    try clearConeCaches(fwd, stream);
    try clearDraftCaches(seq, stream);
    seq.verify_start = 0;
    seq.tap_rows = 0;
    try prefillCone(seq, gather, ids, chunk, cut);
    try stream.synchronize();
    // The head may use a different scratch address after the second pass.
    buffers.items[0].ptr = seq.logits;
    for (buffers.items, want.items, 0..) |buffer, expected, i| {
        const got = try check.download(gpu, buffer);
        defer gpu.gpa.free(got);
        var label: [96]u8 = undefined;
        var name: [64]u8 = undefined;
        try check.sameBytes(try std.fmt.bufPrint(&label, "cone buffer {d} ({s})", .{ i, digestName(fwd, i, &name) }), got, expected);
    }
    const cone_meta = .{ seq.pos, seq.held, seq.verify_rows, seq.verify_start, seq.tap_rows, if (seq.dc) |c| c.absorbed else 0 };
    inline for (baseline_meta, cone_meta) |expected, actual| try check.expect(expected == actual, "cone sequence metadata", .{});
    try check.expect(seq.pos == ids.len, "cone position", .{});
    if (seq.dc) |c| try check.expect(c.absorbed == ids.len, "cone absorbed position", .{});
    for (seq.hist[0..ids.len], ids) |got, expected| try check.expect(got == expected, "cone history", .{});
    if (future) {
        var cone_future = try future512(gpu, seq, gather);
        defer cone_future.trace.deinit(gpu.gpa);
        const full = &full_future.?;
        try check.expect(std.mem.eql(u32, &full.output, &cone_future.output), "cone all512 greedy output IDs", .{});
        try check.expect(full.trace.items.len == cone_future.trace.items.len, "cone round count", .{});
        for (full.trace.items, cone_future.trace.items, 0..) |expected, actual, round| {
            try check.expect(std.meta.eql(expected, actual), "cone proposal/accept trace round {d}", .{round});
        }
        const out = try std.json.Stringify.valueAlloc(gpu.gpa, .{
            .rank = seq.spark.?.dims.rank,
            .input_ids = ids,
            .input_ids_sha256 = input_sha[0..],
            .settings = .{ .replay = 0, .chunk = chunk, .drafts = 5, .adaptive = 0, .stop_eos = false },
            .full = .{ .output_ids = full.output[0..], .rounds = full.trace.items },
            .cone = .{ .output_ids = cone_future.output[0..], .rounds = cone_future.trace.items },
        }, .{});
        defer gpu.gpa.free(out);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = getenv("TF_DS_FIXTURE_ORACLE_OUT").?, .data = out });
        check.pass("cone all512 output IDs and full proposal/accept traces equal; raw input/output IDs saved; Python parity pending", .{});
    }
    check.pass("dsv41 fixture cone: {d} tokens, cut {d}, halo {d}, original chunk {d}; logits/rings/planes all bytes equal; {s}, Engram={s}, DSpark={s}; no KEEP/Python parity", .{ ids.len, cut, halo, chunk, if (tp2) "real TP2 NCCL" else "selfGather", if (seq.engram != null) "on" else "off", if (seq.spark != null) "on" else "off" });
}

fn clearConeCaches(fwd: *dsv41.forward.Forward, stream: cuda.Stream) !void {
    for (fwd.rings) |r| try r.fill8(0, stream.handle);
    for (fwd.planes) |pl| if (pl) |x| {
        inline for (plane_fields) |f| try @field(x, f).fill8(if (comptime std.mem.eql(u8, f, "idxk_scales")) 127 else 0, stream.handle);
    };
}

/// The fixture's cold pass retains Sequence.prefill's original chunks and Engram history,
/// with an explicit cone cut passed only to Prompt.chunk. It never publishes prefix snapshots.
fn prefillCone(seq: *dsv41.serve_lanes.Sequence, gather: dsv41.forward.Gather, ids: []const i64, chunk: usize, cut: usize) !void {
    seq.pos = 0;
    seq.logits = 0;
    seq.held = 0;
    seq.verify_rows = 0;
    for (seq.hist[0..ids.len], ids) |*hv, id| hv.* = @intCast(id);
    if (seq.dc) |*c| c.absorbed = 0;
    if (seq.engram) |g| {
        g.history = seq.hist[0..ids.len];
        try g.resetAhead();
        try g.prefetch(seq.hist[0..ids.len], 0, @min(chunk, ids.len));
    }
    var start: usize = 0;
    while (start < ids.len) {
        const end = @min(ids.len, start + chunk);
        const last = end == ids.len;
        if (seq.engram) |g| if (!last) try g.prefetch(seq.hist[0..ids.len], end, @min(chunk, ids.len - end));
        var top: [2]usize = .{ start, end - start };
        const out = if (seq.spark != null and end - start <= dsv41.serve_lanes.max_rows and cut <= start)
            try coneRowsTapped(seq, gather, ids[start..end], start)
        else blk: {
            const out = try seq.pr.chunk(gather, ids[start..end], start, .{
                .replay = cut,
                .logits = last or end - start <= dsv41.serve_lanes.max_rows,
                .taps = if (seq.spark != null) .{ .buf = seq.taps.ptr, .stride = seq.fwd.cfg.dspark_taps.len * seq.fwd.cfg.dim } else null,
            }, seq.stream);
            top = .{ seq.pr.top_start, seq.pr.top_rows };
            break :blk out;
        };
        if (seq.spark) |sp| if (top[1] > 0) {
            const rope = if (seq.fwd.cfg.ratio(seq.fwd.cfg.n_layers) > 0) seq.fwd.rope_c else seq.fwd.rope_w;
            try sp.absorb(&seq.dc.?, seq.taps.ptr, top[0], top[1], rope, seq.stream);
        };
        if (last) seq.logits = out orelse return error.BadConeFixture;
        start = end;
    }
    seq.pos = ids.len;
}

/// Export exact synthetic IDs before either prefill so failures still retain the oracle input.
fn exportOracleInput(gpu: Gpu, ids: []const i64) ![64]u8 {
    const raw = try std.json.Stringify.valueAlloc(gpu.gpa, ids, .{});
    defer gpu.gpa.free(raw);
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &sum, .{});
    const hex = std.fmt.bytesToHex(sum, .lower);
    const doc = try std.json.Stringify.valueAlloc(gpu.gpa, .{ .input_ids = ids, .input_ids_sha256 = hex[0..] }, .{});
    defer gpu.gpa.free(doc);
    const path = try std.fmt.allocPrint(gpu.gpa, "{s}.input.json", .{getenv("TF_DS_FIXTURE_ORACLE_OUT").?});
    defer gpu.gpa.free(path);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = doc });
    return hex;
}

const FutureRound = struct {
    position: usize,
    proposed: [5]u32 = @splat(0),
    confidence_bits: [5]u32 = @splat(0),
    targets: [6]u32 = @splat(0),
    width: usize,
    accepted: usize,
};
const Future = struct { output: [512]u32, trace: std.ArrayList(FutureRound) = .empty };

/// Fixed-five, greedy, no-EOS execution matching reference90b engine.py's non-adaptive _spec loop.
fn future512(gpu: Gpu, seq: *dsv41.serve_lanes.Sequence, gather: dsv41.forward.Gather) !Future {
    var out: Future = .{ .output = undefined };
    errdefer out.trace.deinit(gpu.gpa);
    out.output[0] = try seq.argmax();
    var count: usize = 1;
    while (count < out.output.len) {
        const at = seq.pos;
        const token = out.output[count - 1];
        try seq.draft(0, 0, token, at, 5);
        const proposed = seq.drafts;
        if (proposed.len != 5) return error.BadConeFixture;
        const width = @min(@as(usize, 5), out.output.len - count);
        var window: [6]i64 = undefined;
        window[0] = token;
        for (proposed.tokens[0..width], 0..) |id, i| window[i + 1] = id;
        try seq.verify(gather, window[0 .. width + 1]);
        var row: FutureRound = .{ .position = at, .width = width, .accepted = 0 };
        @memcpy(&row.proposed, proposed.tokens[0..5]);
        for (proposed.conf[0..5], &row.confidence_bits) |value, *bits| bits.* = @bitCast(value);
        try seq.argmaxRows(row.targets[0 .. width + 1]);
        while (row.accepted < width and proposed.tokens[row.accepted] == row.targets[row.accepted]) row.accepted += 1;
        try seq.keep(at, row.accepted + 1);
        // Absorb only verified retained rows, including the pending target token, before the next draft.
        try seq.draft(at, row.accepted + 1, row.targets[row.accepted], seq.pos, 0);
        try out.trace.append(gpu.gpa, row);
        for (proposed.tokens[0..row.accepted]) |id| {
            if (count == out.output.len) break;
            out.output[count] = id;
            count += 1;
        }
        if (count < out.output.len) {
            out.output[count] = row.targets[row.accepted];
            count += 1;
        }
    }
    try seq.stream.synchronize();
    return out;
}

fn clearDraftCaches(seq: *dsv41.serve_lanes.Sequence, stream: cuda.Stream) !void {
    if (seq.dc) |*c| {
        for (c.rings) |ring| try ring.fill8(0, stream.handle);
        c.absorbed = 0;
        try seq.taps.fill8(0, stream.handle);
    }
}

// The fixture's small tail uses the same Forward.rows/tap ordering as Sequence.prefill.
fn coneRowsTapped(seq: *dsv41.serve_lanes.Sequence, gather: dsv41.forward.Gather, ids: []const i64, start: usize) !Ptr {
    const previous = seq.fwd.probe;
    if (previous != null) return error.BadConeFixture;
    seq.tap_rows = ids.len;
    seq.fwd.probe = .{ .ctx = seq, .call = coneTapProbe };
    defer seq.fwd.probe = previous;
    return seq.fwd.rows(gather, ids, start, ids.len, false, seq.stream);
}

fn coneTapProbe(ctx: *anyopaque, name: []const u8, ptr: Ptr, _: usize, stream: cuda.Stream) anyerror!void {
    const seq: *dsv41.serve_lanes.Sequence = @ptrCast(@alignCast(ctx));
    if (name.len < 2 or name[0] != 'L') return;
    const dot = std.mem.indexOfScalar(u8, name, '.') orelse return;
    if (!std.mem.eql(u8, name[dot + 1 ..], "h_moe")) return;
    const li = (std.fmt.parseInt(usize, name[1..dot], 10) catch return) + 1;
    // The cone guard excludes upper Engram gates, so h_moe is the tap-layer input.
    const cfg = seq.fwd.cfg;
    const col = cfg.dspark_taps.indexOf(@intCast(li)) orelse return;
    try seq.fwd.hc.tap(stream, ptr, seq.taps.ptr + col * cfg.dim * 2, @intCast(cfg.dspark_taps.len * cfg.dim), @intCast(seq.tap_rows));
}

fn digest(gpu: Gpu, b: cuda.DeviceBuffer) !u64 {
    if (b.len == 0) return 0;
    const bytes = try check.download(gpu, b);
    defer gpu.gpa.free(bytes);
    return std.hash.Wyhash.hash(0, bytes);
}

/// `dsv41-gemm-rows`: the prompt path's fp32 cuBLAS GEMMs (the MoE gate's [m, 384] and the indexer weights' [m, 64]
/// over k = 5120) on m rows of a 4096-row draw: for each m, the rows equal to the same rows of the 4096-row product,
/// both with the heuristic's algorithm for m (`run`) and with the 2048-row one's (`runAs`). RESULT lines say which
/// row counts differ; PASS when every pinned run matches.
pub fn gemmRows(gpu: Gpu) !void {
    const gpa = gpu.gpa;
    const K: usize = 5120;
    const M: usize = 4096;
    var g = try dsv41.prompt.Gemm32.init(gpu.d);
    defer g.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(0x6e3);
    const xh = try gpa.alloc(f32, M * K);
    defer gpa.free(xh);
    // bf16-valued inputs, as the prompt path's x.float() is
    for (xh) |*v| v.* = @as(f32, @bitCast(@as(u32, @bitCast(prng.random().floatNorm(f32))) & 0xffff0000));
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    var all_ok = true;
    for ([_]usize{ 384, 64 }) |N| {
        const wh = try gpa.alloc(f32, N * K);
        defer gpa.free(wh);
        for (wh) |*v| v.* = prng.random().floatNorm(f32) * 0.02;
        var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wh));
        defer w.free();
        var full = try cuda.DeviceBuffer.alloc(gpu.d, M * N * 4);
        defer full.free();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, M * N * 4);
        defer part.free();
        try g.run(x.ptr, w.ptr, full.ptr, M, N, K, stream);
        try stream.synchronize();
        const want = try check.download(gpu, full);
        defer gpa.free(want);
        for ([_]usize{ 2048, 1024, 1280, 640, 3000, 904, 2952, 129, 64, 17 }) |m| {
            for ([_]bool{ false, true }) |pinned| {
                for ([_]usize{ 0, M - m }) |r0| {
                    const xp = x.ptr + r0 * K * 4;
                    if (pinned) try g.runAs(xp, w.ptr, part.ptr, m, 2048, N, K, stream) else try g.run(xp, w.ptr, part.ptr, m, N, K, stream);
                    try stream.synchronize();
                    const got = try check.download(gpu, part);
                    defer gpa.free(got);
                    const same = std.mem.eql(u8, got[0 .. m * N * 4], want[r0 * N * 4 .. (r0 + m) * N * 4]);
                    if (!same and pinned) all_ok = false;
                    std.debug.print("RESULT dsv41-gemm-rows N {d} m {d} rows {d}.. {s}: {s}\n", .{ N, m, r0, if (pinned) "algorithm of 2048 rows" else "algorithm of m rows", if (same) "equal to the 4096-row product" else "DIFFERS" });
                }
            }
        }
    }
    try check.expect(all_ok, "dsv41-gemm-rows: a pinned run differs from the 4096-row product", .{});
    check.pass("dsv41-gemm-rows: every pinned run equals the 4096-row product's rows", .{});
}

/// Per-row hashes of every probed sublayer output of a pass (`TF_DS_COMPARE_ROWS`), by name and absolute row: the
/// probed buffer holds the rows of the prompt path's current walk (`Prompt.at_piece`).
const RowLog = struct {
    gpa: std.mem.Allocator,
    seq: *dsv41.serve_lanes.Sequence,
    names: std.ArrayList([]u8) = .empty,
    rows: std.ArrayList(std.ArrayList(u64)) = .empty,

    fn deinit(self: *RowLog) void {
        for (self.names.items) |n| self.gpa.free(n);
        self.names.deinit(self.gpa);
        for (self.rows.items) |*r| r.deinit(self.gpa);
        self.rows.deinit(self.gpa);
    }

    fn call(ctx: *anyopaque, name: []const u8, ptr: Ptr, bytes: usize, stream: cuda.Stream) anyerror!void {
        const self: *RowLog = @ptrCast(@alignCast(ctx));
        const wk = self.seq.pr.at_piece;
        if (wk.rows == 0 or bytes % wk.rows != 0) return;
        try stream.synchronize();
        const host = try self.gpa.alloc(u8, bytes);
        defer self.gpa.free(host);
        const b: cuda.DeviceBuffer = .{ .d = bound.?, .ptr = ptr, .len = bytes };
        try b.download(0, host);
        var k: usize = 0;
        while (k < self.names.items.len and !std.mem.eql(u8, self.names.items[k], name)) k += 1;
        if (k == self.names.items.len) {
            try self.names.append(self.gpa, try self.gpa.dupe(u8, name));
            try self.rows.append(self.gpa, .empty);
        }
        const list = &self.rows.items[k];
        const per = bytes / wk.rows;
        if (list.items.len < wk.at + wk.rows) try list.appendNTimes(self.gpa, 0, wk.at + wk.rows - list.items.len);
        for (0..wk.rows) |i| list.items[wk.at + i] = std.hash.Wyhash.hash(1, host[i * per ..][0..per]);
    }

    /// Prints the first sublayer (in probe order) whose rows differ from `other`'s, and how many sublayers differ.
    fn report(self: *const RowLog, other: *const RowLog) void {
        var bad: usize = 0;
        for (self.names.items, self.rows.items) |name, a| {
            var k: usize = 0;
            while (k < other.names.items.len and !std.mem.eql(u8, other.names.items[k], name)) k += 1;
            if (k == other.names.items.len) continue;
            const b = other.rows.items[k].items;
            var first: ?usize = null;
            var n: usize = 0;
            for (a.items, 0..) |h, i| if (i < b.len and h != b[i]) {
                if (first == null) first = i;
                n += 1;
            };
            if (n > 0) {
                if (bad < 6) std.debug.print("  rows differ: {s}: {d} rows, the first at {d}\n", .{ name, n, first.? });
                bad += 1;
            }
        }
        std.debug.print("RESULT dsv41-prompt-time rows: {d} of {d} probed sublayers differ\n", .{ bad, self.names.items.len });
    }
};
