//! `glm53-seg <work-dir> <model-dir>`: several streams' decode windows in one forward (cuda_model.zig
//! `forwardSegs`) against the Mia fork's multi-stream verify windows (tools/zig/glm53/dump_seg.py, packed by
//! capture.sh seg: BatchedVerify over layers 0 and 3 of the served pool, bf16 or TF_GLM_KV=fp8 caches), both ranks as
//! threads on this GPU. The fixture's state before its first window (layer 3's latent and pooled-key planes, every
//! slot's index rings, layer 0's KDA states and conv windows) goes into a pool and four stream slots bound to the
//! fixture's extents; then each window runs with its segments' rows, commits each segment's keep, and must leave the
//! bytes the fork's window left: the window's logits (this rank's vocabulary half), each segment's latent rows, the
//! pooled-key plane, the rings, each segment's stream's KDA state and conv window; and before the commits layer 0's
//! KDA projections, low-rank products and chain output against the fork's traced calls. The fork's batched window equals its
//! segments run alone (the dump checks it), so this also holds the engine's per-stream attention to the fork's
//! segmented kernels.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const hs = @import("glm53_harness.zig");
const pair = @import("glm53_pair.zig");
const Gpu = check.Gpu;

const M = glm.model;
const A = glm.dsa;
const P = glm.pool;
const Buf = cuda.DeviceBuffer;
const Json = std.json.Value;

const streams = 4;
const slot_rows = 16;

/// One rank: its model cut to the fixture's layers, kernels, pool, stream slots and window buffers.
const Rank = struct {
    cfg: *const glm.config.Config,
    lay: glm.config.Layout,
    set: *const cuda.aot.Set,
    hck: glm.glue.Hc,
    kk: glm.kda.Kernels,
    mk: glm.moe.Kernels,
    layers: [2]M.Layer,
    mdl: M.Model,
    pool: P.Pool,
    sts: [streams]M.State,
    work: M.Work,
    stream: cuda.Stream,
    failures: usize = 0,
    passes: usize = 0,

    fn kernels(r: *const Rank) M.Kernels {
        return .{ .g = .{ .set = r.set, .hc = &r.hck, .hidden = r.cfg.hidden, .streams = r.cfg.streams }, .kk = &r.kk, .mk = &r.mk };
    }
};

fn param(fx: *const fixture.Fixture, name: []const u8) Json {
    return fx.parsed.value.object.get("params").?.object.get(name).?;
}

fn int(v: Json) usize {
    return @intCast(v.integer);
}

/// dump_seg.py's histories: each stream's prompt chunks of 2,048 flip its KDA parity once a chunk (forward.py
/// kda_rows), so a stream's current parity is its chunk count's.
fn parity(history: usize) u1 {
    return @intCast(((history + 2047) / 2048) % 2);
}

fn load(r: *Rank, gpu: Gpu, cfg: *const glm.config.Config, set: *const cuda.aot.Set, fx: *const fixture.Fixture, model: []const u8, rank: usize, kv: A.Kv) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    r.cfg = cfg;
    r.set = set;
    r.lay = .{ .rank = rank, .world = 2, .cfg = cfg };
    r.stream = try cuda.Stream.init(d, true);
    r.hck = try glm.glue.Hc.load(d);
    r.kk = try glm.kda.Kernels.load(d);
    r.mk = try glm.moe.Kernels.load(d);
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = r.lay };
    // layers 0 (KDA, dense MLP) and 3 (DSA, MoE), as ref.load_rank cuts the model
    const kinds = [2]usize{ 0, 3 };
    for (&r.layers, kinds) |*x, i| {
        x.* = .{ .index = i, .frame = try glm.weights.Frame.load(l, i) };
        switch (cfg.kind(i)) {
            .kda => x.kda = try glm.weights.Kda.load(l, i),
            .dsa => x.dsa = try A.Dsa.load(l, i),
        }
        switch (cfg.mlpKind(i)) {
            .dense => x.mlp = try glm.weights.Mlp.load(l, "mlp.", i),
            .moe => x.moe = try glm.moe.Moe.load(l, i),
        }
    }
    const rows = try l.headRows();
    defer gpa.free(rows);
    r.mdl = .{
        .gpa = gpa,
        .layers = &r.layers,
        .embed = try l.share(glm.weights_map.embed),
        .norm = try l.share(glm.weights_map.norm),
        .head = .{ .buf = try Buf.fromHost(d, rows), .n = rows.len / 2 / cfg.hidden, .k = cfg.hidden },
    };
    pack.dropCache();

    // the fixture's pool and slots
    const pool_rows = int(param(fx, "pool_rows"));
    const ring = int(param(fx, "ring"));
    r.pool = try P.Pool.init(gpa, d, cfg, 1, pool_rows, kv, true);
    var nb: [64]u8 = undefined;
    for ([_][]const u8{ "lat", "pk" }, [_]Buf{ r.pool.latent[0], r.pool.pk[0] }) |name, b| {
        const bytes = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_init_{s}", .{ rank, name }));
        defer gpa.free(bytes);
        if (bytes.len != b.len) return error.FixtureShape;
        try b.upload(0, bytes);
    }
    const ik = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_init_ik", .{rank}));
    defer gpa.free(ik);
    const ig = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_init_ig", .{rank}));
    defer gpa.free(ig);
    const rec = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_init_rec", .{rank}));
    defer gpa.free(rec);
    const conv = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_init_conv", .{rank}));
    defer gpa.free(conv);
    const bases = param(fx, "bases").array.items;
    const extents = param(fx, "extents").array.items;
    const positions = param(fx, "positions").array.items;
    const history = param(fx, "history").array.items;
    for (&r.sts, 0..) |*st, s| {
        st.* = try M.State.initSlot(gpa, d, r.lay, 4, ring, slot_rows, kv);
        try st.bind(&r.pool, int(bases[s]), int(extents[s]));
        const c = &st.caches[0];
        const rb = ring * cfg.index_dim * 2;
        if (ik.len != streams * rb) return error.FixtureShape;
        try c.ik.upload(0, ik[s * rb ..][0..rb]);
        try c.ig.upload(0, ig[s * rb ..][0..rb]);
        // the fork's slots.rec [slot, parity, layer 0, heads, 128, 128] and conv [slot, layer 0, 3, C]
        const sb = st.kda.stateBytes();
        if (rec.len != streams * 2 * sb) return error.FixtureShape;
        for (0..2) |p| try (Buf{ .d = d, .ptr = st.kda.state(@intCast(p), 0), .len = sb }).upload(0, rec[(s * 2 + p) * sb ..][0..sb]);
        const cb = conv.len / streams;
        try (Buf{ .d = d, .ptr = st.kda.convOf(0), .len = cb }).upload(0, conv[s * cb ..][0..cb]);
        st.kda.cur = if (s < history.len) parity(int(history[s])) else 0;
        st.kda.pos = int(positions[s]);
        st.pos = st.kda.pos;
    }
    r.work = try M.Work.init(d, r.lay, int(param(fx, "window_rows")), slot_rows, pool_rows);
}

fn unload(r: *Rank) void {
    r.work.deinit();
    for (&r.sts) |*st| st.deinit();
    r.pool.deinit();
    r.mdl.embed.free();
    r.mdl.norm.free();
    r.mdl.head.deinit();
    for (&r.layers) |*x| x.deinit();
    r.mk.deinit();
    r.kk.deinit();
    r.hck.deinit();
    r.stream.deinit();
}

/// One digest of the fixture's params "digests" against `got` (bytes on the host).
fn same(r: *Rank, fx: *const fixture.Fixture, rank: usize, k: usize, what: []const u8, got: []const u8) !void {
    var nb: [64]u8 = undefined;
    const key = try std.fmt.bufPrint(&nb, "r{d}_w{d}_{s}", .{ rank, k, what });
    const want = param(fx, "digests").object.get(key) orelse return error.MissingDigest;
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(got, &out, .{});
    const hex = std.fmt.bytesToHex(out, .lower);
    const w = want.array.items;
    if (got.len != @as(usize, @intCast(w[1].integer)) or !std.mem.eql(u8, &hex, w[0].string)) {
        std.debug.print("FAIL glm53-seg rank {d} window {d} {s}: {d} bytes sha256 {s}, the fork's {d} bytes {s}\n", .{ rank, k, what, got.len, hex[0..16], w[1].integer, w[0].string[0..16] });
        r.failures += 1;
    } else r.passes += 1;
}

/// The fork's digest of argument `arg` of the `nth` traced call to a function named `name` (the module path's last
/// part) in rank `rank`'s window `k` (dump_prompt.py's DigestTracer entries in the manifest's "trace").
fn traced(fx: *const fixture.Fixture, rank: usize, k: usize, name: []const u8, nth: usize, arg: []const u8) ?Json {
    var nb: [16]u8 = undefined;
    const step = std.fmt.bufPrint(&nb, "w{d}", .{k}) catch return null;
    var seen: usize = 0;
    for (fx.parsed.value.object.get("trace").?.array.items) |call| {
        const o = call.object;
        if (int(o.get("rank").?) != rank or !std.mem.eql(u8, o.get("step").?.string, step)) continue;
        const f = o.get("fn").?.string;
        if (!std.mem.endsWith(f, name) or f[f.len - name.len - 1] != '.') continue;
        if (seen == nth) return o.get("digests").?.object.get(arg);
        seen += 1;
    }
    return null;
}

/// Layer 0's KDA buffers after a window (no later layer writes them) against the fork's traced calls: the
/// projections, the two low-rank products and the chains' output, to place a difference before or after the chain.
fn sameKda(r: *Rank, gpa: std.mem.Allocator, fx: *const fixture.Fixture, rank: usize, k: usize, R: usize, d: *const cuda.Driver) !void {
    const b = &r.work.b;
    const lw = r.layers[0].kda.?.heads * glm.kda.dk;
    const width = r.sts[0].kda.width;
    const checks = [_]struct { []const u8, []const u8, usize, []const u8, cuda.abi.DevicePtr, usize }{
        .{ "kda_proj", "mm", 0, "out", b.kproj.ptr, R * width * 2 },
        .{ "kda_a", "mm", 1, "out", b.ka.ptr, R * lw * 2 },
        .{ "kda_g", "mm", 2, "out", b.kg.ptr, R * lw * 2 },
        .{ "kda_out", "chain_segments", 0, "ret", b.kout.ptr, R * lw * 2 },
    };
    for (checks) |c| {
        const want = traced(fx, rank, k, c[1], c[2], c[3]) orelse return error.MissingTrace;
        const x = try host(gpa, .{ .d = d, .ptr = c[4], .len = c[5] });
        defer gpa.free(x);
        var out: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(x, &out, .{});
        const hex = std.fmt.bytesToHex(out, .lower);
        const w = want.array.items;
        if (x.len != @as(usize, @intCast(w[1].integer)) or !std.mem.eql(u8, &hex, w[0].string)) {
            std.debug.print("FAIL glm53-seg rank {d} window {d} {s}: {d} bytes sha256 {s}, the fork's {d} bytes {s}\n", .{ rank, k, c[0], x.len, hex[0..16], w[1].integer, w[0].string[0..16] });
            r.failures += 1;
        } else r.passes += 1;
    }
}

fn host(gpa: std.mem.Allocator, b: Buf) ![]u8 {
    const x = try gpa.alloc(u8, b.len);
    try b.download(0, x);
    return x;
}

fn sameBuf(r: *Rank, gpa: std.mem.Allocator, fx: *const fixture.Fixture, rank: usize, k: usize, what: []const u8, b: Buf) !void {
    const x = try host(gpa, b);
    defer gpa.free(x);
    try same(r, fx, rank, k, what, x);
}

const Job = struct {
    gpu: Gpu,
    r: *Rank,
    fx: *const fixture.Fixture,
    rank: usize,
    link: *pair.HostLink,
    ids: [][]u8, // each window's token ids and the fork's logits, read before the ranks' threads start
    logits: [][]u8,
    err: ?anyerror = null,
};

/// Rank `j.rank`'s windows, in the fixture's order.
fn windows(j: *Job) !void {
    const r = j.r;
    const fx = j.fx;
    const gpa = j.gpu.gpa;
    const d = j.gpu.d;
    const c = r.cfg;
    const k = r.kernels();
    const gather: M.Gather = .{ .ctx = j.link, .run = pair.HostLink.gather };
    const vocab = r.mdl.head.n;
    for (param(fx, "windows").array.items, 0..) |wv, wi| {
        const segs_json = wv.object.get("segments").?.array.items;
        var segs: [streams]M.Seg = undefined;
        var keeps: [streams]usize = undefined;
        var starts: [streams]usize = undefined;
        var R: usize = 0;
        for (segs_json, 0..) |sv, i| {
            const o = sv.object;
            const s = int(o.get("stream").?);
            if (r.sts[s].pos != int(o.get("pos").?)) return error.PositionMismatch;
            segs[i] = .{ .st = &r.sts[s], .a0 = R, .n = int(o.get("rows").?) };
            keeps[i] = int(o.get("keep").?);
            starts[i] = r.sts[s].pos;
            R += segs[i].n;
        }
        const n = segs_json.len;
        if (j.ids[wi].len != R * 4) return error.FixtureShape;
        try r.work.b.ids.upload(0, j.ids[wi]);
        try M.forwardSegs(k, r.lay, &r.mdl, segs[0..n], &r.work, R, gather, r.stream);
        try r.stream.synchronize();
        try sameKda(r, gpa, fx, j.rank, wi, R, d);
        for (segs[0..n], keeps[0..n]) |sg, keep| try M.commit(k, r.lay, sg.st, sg.n, keep, false, &r.work, r.stream);
        try r.stream.synchronize();
        // the window's logits (the fork's array, and each segment's rows by digest), then what each segment and the
        // pool hold after the commits
        const logits = try host(gpa, .{ .d = d, .ptr = r.work.logits.ptr, .len = R * vocab * 2 });
        defer gpa.free(logits);
        if (!std.mem.eql(u8, logits, j.logits[wi])) {
            std.debug.print("FAIL glm53-seg rank {d} window {d}: logits differ from the fork's\n", .{ j.rank, wi });
            r.failures += 1;
        } else r.passes += 1;
        for (segs[0..n], 0..) |sg, i| {
            var wb: [16]u8 = undefined;
            try same(r, fx, j.rank, wi, try std.fmt.bufPrint(&wb, "s{d}_logits", .{i}), logits[sg.a0 * vocab * 2 ..][0 .. sg.n * vocab * 2]);
        }
        const lat_row = r.pool.kv.rowBytes(c.kv_lora);
        for (segs_json, segs[0..n], starts[0..n], 0..) |sv, sg, p0, i| {
            const s = int(sv.object.get("stream").?);
            const base = int(sv.object.get("base").?);
            var wb: [16]u8 = undefined;
            try sameBuf(r, gpa, fx, j.rank, wi, try std.fmt.bufPrint(&wb, "s{d}_latent", .{i}), .{ .d = d, .ptr = r.pool.latent[0].ptr + (base + p0) * lat_row, .len = sg.n * lat_row });
            const kst = &r.sts[s].kda;
            try sameBuf(r, gpa, fx, j.rank, wi, try std.fmt.bufPrint(&wb, "s{d}_rec", .{i}), .{ .d = d, .ptr = kst.state(kst.cur, 0), .len = kst.stateBytes() });
            const cb = 3 * 3 * kst.heads * glm.kda.dk * 2;
            try sameBuf(r, gpa, fx, j.rank, wi, try std.fmt.bufPrint(&wb, "s{d}_conv", .{i}), .{ .d = d, .ptr = kst.convOf(0), .len = cb });
        }
        try sameBuf(r, gpa, fx, j.rank, wi, "pk", r.pool.pk[0]);
        for ([_][]const u8{ "ik", "ig" }, 0..) |what, which| {
            var all: std.ArrayList(u8) = .empty;
            defer all.deinit(gpa);
            for (&r.sts) |*st| {
                const b = if (which == 0) st.caches[0].ik else st.caches[0].ig;
                const x = try host(gpa, b);
                defer gpa.free(x);
                try all.appendSlice(gpa, x);
            }
            try same(r, fx, j.rank, wi, what, all.items);
        }
        if (j.rank == 0) check.pass("glm53-seg window {d}: {d} segments, {d} rows", .{ wi, n, R });
    }
}

fn runRank(j: *Job) void {
    j.gpu.ctx.makeCurrent() catch |e| {
        j.err = e;
        j.link.h.failed.store(true, .release);
        return;
    };
    windows(j) catch |e| {
        std.debug.print("glm53-seg rank {d}: {t}\n", .{ j.rank, e });
        j.err = e;
        j.link.h.failed.store(true, .release);
    };
}

/// `want`: the cache format the fixture must hold (glm53-seg-fp8), or any (glm53-seg).
pub fn run(gpu: Gpu, work: []const u8, model: []const u8, want: ?A.Kv) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: glm.config.Why = .{};
    const cfg = glm.config.read(gpa, gpu.io, model, &why) catch |e| {
        std.debug.print("FAIL glm53 config: {t}: {s}\n", .{ e, why.text() });
        return e;
    };
    const aot_dir = try std.fs.path.join(gpa, &.{ work, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_dir = try std.fs.path.join(gpa, &.{ work, "out", "fixture" });
    defer gpa.free(fx_dir);
    var fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir);
    defer fx.deinit();
    const kv = try A.Kv.parse(param(&fx, "kv").string);
    if (want) |w| try check.expect(kv == w, "glm53-seg: the fixture's DSA caches are {t}, not {t} (capture it with KV={t})", .{ kv, w, w });

    var ranks: [2]Rank = undefined;
    var loaded: usize = 0;
    defer for (ranks[0..loaded]) |*r| unload(r);
    for (&ranks, 0..) |*r, ri| {
        try load(r, gpu, &cfg, &set, &fx, model, ri, kv);
        r.failures = 0;
        r.passes = 0;
        loaded += 1;
    }
    const most = int(param(&fx, "window_rows")) * cfg.hidden;
    var h: pair.Host = .{ .slots = .{ try gpa.alloc(u8, most * 4), try gpa.alloc(u8, most * 4) } };
    defer for (h.slots) |s| gpa.free(s);
    var links: [2]pair.HostLink = .{ .{ .h = &h, .rank = 0, .d = d }, .{ .h = &h, .rank = 1, .d = d } };
    const nw = param(&fx, "windows").array.items.len;
    var jobs: [2]Job = undefined;
    var arrays: [2][2][][]u8 = .{ .{ &.{}, &.{} }, .{ &.{}, &.{} } };
    var read: [2][2]usize = .{ .{ 0, 0 }, .{ 0, 0 } };
    defer for (&arrays, &read) |*per, *nr| for (per, nr) |a, m| {
        for (a[0..m]) |x| gpa.free(x);
        gpa.free(a);
    };
    for (&jobs, 0..) |*j, ri| {
        for (0..2) |which| {
            arrays[ri][which] = try gpa.alloc([]u8, nw);
            for (0..nw) |wi| {
                var nb: [48]u8 = undefined;
                arrays[ri][which][wi] = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_w{d}_{s}", .{ ri, wi, if (which == 0) "ids" else "logits" }));
                read[ri][which] += 1;
            }
        }
        j.* = .{ .gpu = gpu, .r = &ranks[ri], .fx = &fx, .rank = ri, .link = &links[ri], .ids = arrays[ri][0], .logits = arrays[ri][1] };
    }
    var threads: [2]std.Thread = undefined;
    for (&threads, &jobs) |*t, *j| t.* = try std.Thread.spawn(.{}, runRank, .{j});
    for (threads) |t| t.join();
    for (jobs) |j| if (j.err) |e| return e;
    const bad = ranks[0].failures + ranks[1].failures;
    const good = ranks[0].passes + ranks[1].passes;
    try check.expect(bad == 0, "glm53-seg: {d} comparisons differ ({d} equal)", .{ bad, good });
    check.pass("glm53-seg: {d} windows on {t} caches, both ranks, {d} comparisons bit-exact against the fork's batched windows", .{ param(&fx, "windows").array.items.len, kv, good });
}
