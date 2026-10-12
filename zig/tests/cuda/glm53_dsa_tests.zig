//! `glm53-dsa <work-dir> <model-dir> [isolated]`: GLM-5.3-Flash's first DSA layer (layer 3: the indexer, the pooled
//! keys, the top-512 pool selection, the latent attention dense and sparse, then its MoE) on both ranks against the
//! Mia fork's call-by-call fixture (tools/zig/glm53/dump_dsa.py, packed by capture.sh): run a (one stream from
//! position 0 across the dense limit, prompt chunks and decode windows) and run l (decode windows at 40,000 and
//! 600,000 positions over the fixture's drawn caches), every loaded tensor's sha256 against the reference loader's.
//! The MTP layer's run m is not checked here yet. Both ranks run on this one GPU; see glm53_harness.zig.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const hs = @import("glm53_harness.zig");
const Gpu = check.Gpu;
const Harness = hs.Harness;

const F = glm.forward;
const M = glm.moe;
const A = glm.dsa;
const T = glm.mtp;
const Buf = cuda.DeviceBuffer;
const Ptr = cuda.abi.DevicePtr;

const max_rows = 16;
const prompt_rows = 2048;

const Rank = struct {
    frame: glm.weights.Frame,
    dsa: A.Dsa,
    moe: M.Moe,
    norm: Buf,
    embed: Buf,
    b: F.Buffers,
    pb: F.Buffers,
    ms: M.Scratch,
    pms: M.Scratch,
    fnormed: Buf,
    fxs: Buf,
    pos_dev: Buf,
    mtp: T.Mtp,
    mb: T.Buffers,
    pmb: T.Buffers,

    fn deinit(r: *Rank) void {
        r.mtp.deinit();
        r.mb.deinit();
        r.pmb.deinit();
        r.frame.deinit();
        r.dsa.deinit();
        r.moe.deinit();
        r.norm.free();
        r.embed.free();
        r.b.deinit();
        r.pb.deinit();
        r.ms.deinit();
        r.pms.deinit();
        r.fnormed.free();
        r.fxs.free();
        r.pos_dev.free();
    }
};

/// One run's per-rank caches and DSA scratch over `capacity` tokens. With TF_GLM_TIER=<dir> (bf16 or FP8 rows) each rank's
/// latent rows live on a tier (cuda_tier.zig) whose slab holds about half of them, the rest on a file in <dir>:
/// every comparison must still hold (the latent write's cache is then not compared: the rows are the slab's).
const Run = struct {
    cache: [2]A.Cache,
    ds: [2]A.Scratch,
    pds: [2]A.Scratch,
    tier: [2]?*glm.tier.Tier = .{ null, null },
    gpa: std.mem.Allocator = undefined,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: *const glm.config.Config, capacity: usize, kv: A.Kv, tag: []const u8) !Run {
        var r: Run = .{ .cache = undefined, .ds = undefined, .pds = undefined, .gpa = gpa };
        const dir: ?[]const u8 = if (std.c.getenv("TF_GLM_TIER")) |e| std.mem.span(e) else null;
        for (0..2) |i| {
            const lay: glm.config.Layout = .{ .rank = i, .world = 2, .cfg = cfg };
            r.cache[i] = try A.Cache.init(d, cfg, capacity, A.indexRing(capacity, @max(max_rows, prompt_rows)), kv);
            r.ds[i] = try A.Scratch.init(d, lay, max_rows, capacity, false);
            r.pds[i] = try A.Scratch.init(d, lay, prompt_rows, capacity, true);
            if (dir) |dd| {
                const kvt = glm.tier.kv;
                const cap = @max(std.mem.alignForward(usize, capacity, kvt.seg_positions), kvt.home_positions + kvt.seg_positions);
                const runs = kvt.home_runs + 4 + cap / kvt.seg_positions / 2;
                const path = try std.fmt.allocPrintSentinel(gpa, "{s}/glm53-dsa-{s}-r{d}-{d}.bin", .{ dd, tag, i, std.c.getpid() }, 0);
                defer gpa.free(path);
                const t = try glm.tier.Tier.init(gpa, d, .{ .streams = 1, .layers = 1, .capacity = cap, .row_bytes = kv.rowBytes(cfg.kv_lora), .recent_runs = 2 }, runs * kvt.run_pools * 4 * kv.rowBytes(cfg.kv_lora), prompt_rows, glm.glue.sparse_tokens, path);
                try t.open(0, 1);
                r.tier[i] = t;
                r.cache[i].tier = .{ .t = t, .h = 0, .layer = 0 };
            }
        }
        return r;
    }

    /// Before a window of R rows at `pos` (the fixture's: rows before it are final).
    fn window(r: *Run, pos: usize, R: usize, stream: cuda.Stream) !void {
        for (r.tier) |x| if (x) |t| {
            t.rewind(0, pos);
            try t.commit(0, pos, stream);
            try t.beginWindow(0, pos, R);
        };
    }

    fn deinit(r: *Run) void {
        for (&r.cache) |*c| c.deinit();
        for (&r.ds) |*s| s.deinit();
        for (&r.pds) |*s| s.deinit();
        for (r.tier) |x| if (x) |t| t.deinit(r.gpa);
    }
};

fn weightsOk(gpa: std.mem.Allocator, io: std.Io, work: []const u8, ranks: []Rank, cfg: *const glm.config.Config, pack: *glm.weights.Pack) !void {
    const path = try std.fs.path.join(gpa, &.{ work, "out", "weights.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var bad: usize = 0;
    var n: usize = 0;
    for (ranks, 0..) |*r, ri| {
        var kb: [16]u8 = undefined;
        const want = parsed.value.object.get(try std.fmt.bufPrint(&kb, "rank{d}", .{ri})).?.object;
        const a = &r.dsa;
        const pairs = [_]struct { []const u8, Buf }{
            .{ "layers.3.dsa.proj.weight", a.proj.buf }, .{ "layers.3.dsa.q_norm", a.q_norm },           .{ "layers.3.dsa.kv_norm", a.kv_norm },
            .{ "layers.3.dsa.q_b.weight", a.q_b.buf },   .{ "layers.3.dsa.absorb.wk", a.wk },            .{ "layers.3.dsa.absorb.wv", a.wv },
            .{ "layers.3.dsa.o.weight", a.o.buf },       .{ "layers.3.dsa.index.kw.weight", a.kw.buf },  .{ "layers.3.dsa.index.qb.weight", a.qb.buf },
            .{ "layers.3.dsa.index.ln_w", a.ln_w },      .{ "layers.3.dsa.index.ln_b", a.ln_b },         .{ "layers.3.dsa.index.gate", a.gate },
            .{ "layers.3.dsa.index.ape", a.ape },        .{ "layers.3.attn_hc.fn", r.frame.attn_hc.fn_ }, .{ "layers.3.in_norm", r.frame.in_norm },
            .{ "norm", r.norm },
        };
        for (pairs) |p| {
            const got = try hs.sha(gpa, p[1]);
            n += 1;
            if (!std.mem.eql(u8, &got, want.get(p[0]).?.string)) {
                std.debug.print("FAIL glm53 rank {d} {s}: sha256 {s}, the reference's {s}\n", .{ ri, p[0], got[0..16], want.get(p[0]).?.string[0..16] });
                bad += 1;
            }
        }
    }
    // the MTP drafts' 4-bit head, quantized and packed on the host from each rank's head rows
    for (0..ranks.len) |ri| {
        var kb: [16]u8 = undefined;
        const want = parsed.value.object.get(try std.fmt.bufPrint(&kb, "rank{d}", .{ri})).?.object;
        const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
        const l: glm.weights.Loader = .{ .gpa = gpa, .d = undefined, .pack = pack, .lay = lay };
        const rows = try l.headRows();
        defer gpa.free(rows);
        var p = try glm.draft_head.quantizePack(gpa, std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(rows))), rows.len / 2 / cfg.hidden, cfg.hidden);
        defer p.deinit(gpa);
        const parts = [_]struct { []const u8, []const u8 }{
            .{ "head.weight", rows }, .{ "draft_head.weight", std.mem.sliceAsBytes(p.words) },
            .{ "draft_head.scales", std.mem.sliceAsBytes(p.scales) }, .{ "draft_head.biases", std.mem.sliceAsBytes(p.biases) },
        };
        for (parts) |q| {
            var out: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(q[1], &out, .{});
            const got = std.fmt.bytesToHex(out, .lower);
            n += 1;
            if (!std.mem.eql(u8, &got, want.get(q[0]).?.string)) {
                std.debug.print("FAIL glm53 rank {d} {s}: sha256 {s}, the reference's {s}\n", .{ ri, q[0], got[0..16], want.get(q[0]).?.string[0..16] });
                bad += 1;
            }
        }
    }
    try check.expect(bad == 0, "glm53-dsa: {d} of {d} loaded tensors differ from the reference loader's", .{ bad, n });
    check.pass("glm53-dsa: every rank's layer-3 DSA tensors, head rows and 4-bit draft head equal the reference loader's ({d} sha256)", .{n});
}

/// The fixture's DSA cache format (params "kv": dump_*.py with TF_GLM53_REF_KV; absent: bf16), required to be `want`
/// when given (the -fp8 subcommands).
fn fixtureKv(params: std.json.ObjectMap, want: ?A.Kv) !A.Kv {
    const kv = try A.Kv.parse(if (params.get("kv")) |v| v.string else "");
    if (want) |w| try check.expect(kv == w, "glm53: the fixture's DSA caches are {t}, not {t} (capture it with KV={t})", .{ kv, w, w });
    return kv;
}

/// `want`: the cache format the fixture must hold (glm53-dsa-fp8), or any (glm53-dsa).
pub fn run(gpu: Gpu, work: []const u8, model: []const u8, isolated: bool, want: ?A.Kv) !void {
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
    // a blocking stream: the harness's synchronous uploads from pageable memory land before its next kernels
    var h: Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir), .stream = try cuda.Stream.init(d, false), .isolated = isolated };
    defer h.fx.deinit();
    defer h.stream.deinit();
    defer h.deinit();
    const layer: usize = @intCast(try h.fx.int("layer"));
    const capacity: usize = @intCast(try h.fx.int("capacity"));
    const long_capacity: usize = @intCast(try h.fx.int("long_capacity"));
    const kv = try fixtureKv(h.fx.parsed.value.object.get("params").?.object, want);

    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var mk = try M.Kernels.load(d);
    defer mk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    var ranks: [2]Rank = undefined;
    var loaded: usize = 0;
    defer for (ranks[0..loaded]) |*r| r.deinit();
    for (&ranks, 0..) |*r, ri| {
        const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
        const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
        r.frame = try glm.weights.Frame.load(l, layer);
        r.dsa = try A.Dsa.load(l, layer);
        r.moe = try M.Moe.load(l, layer);
        r.norm = try l.share(glm.weights_map.norm);
        r.embed = try l.share(glm.weights_map.embed);
        r.b = try F.Buffers.init(d, lay, max_rows, false);
        r.pb = try F.Buffers.init(d, lay, prompt_rows, true);
        r.ms = try M.Scratch.init(d, lay, max_rows, false);
        r.pms = try M.Scratch.init(d, lay, prompt_rows, true);
        r.fnormed = try Buf.alloc(d, prompt_rows * cfg.hidden * 2);
        r.fxs = try Buf.alloc(d, prompt_rows * cfg.hidden / 64 * 4);
        r.pos_dev = try Buf.alloc(d, 16);
        r.mtp = try T.Mtp.load(l);
        r.mb = try T.Buffers.init(d, lay, max_rows, false);
        r.pmb = try T.Buffers.init(d, lay, prompt_rows, true);
        loaded += 1;
    }
    try weightsOk(gpa, gpu.io, work, &ranks, &cfg, &pack);
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams };
    const steps = h.fx.parsed.value.object.get("params").?.object.get("steps").?.object;

    {
        var run_a = try Run.init(gpa, d, &cfg, capacity, kv, "a");
        defer run_a.deinit();
        try runSteps(&h, g, &mk, &cfg, &ranks, &run_a, steps.get("a").?.array.items, "a", false);
    }

    {
        var tk = try T.Kernels.load(d);
        defer tk.deinit();
        var run_m = try Run.init(gpa, d, &cfg, capacity, kv, "m");
        defer run_m.deinit();
        try runMtp(&h, g, &mk, &tk, &cfg, &ranks, &run_m, steps.get("m").?.array.items);
    }

    var run_l = try Run.init(gpa, d, &cfg, long_capacity, kv, "l");
    defer run_l.deinit();
    for (&run_l.cache) |*c| {
        const lat = try h.fx.bytes("l_latent0");
        defer gpa.free(lat);
        try c.latent.upload(0, lat);
        if (c.tier) |tr| try tr.t.host.load(0, 0, lat, @min(long_capacity, lat.len / kv.rowBytes(cfg.kv_lora)));
        const pk = try h.fx.bytes("l_pk0");
        defer gpa.free(pk);
        try c.pk.upload(0, pk);
    }
    try runSteps(&h, g, &mk, &cfg, &ranks, &run_l, steps.get("l").?.array.items, "l", true);
    try check.expect(h.failures == 0, "glm53-dsa: {d} comparisons differ ({d} equal)", .{ h.failures, h.passes });
    check.pass("glm53-dsa: runs a, m and l on {t} caches, {d} comparisons bit-exact{s}", .{ kv, h.passes, if (isolated) " (isolated)" else "" });
}

fn runSteps(h: *Harness, g: glm.glue.Glue, mk: *const M.Kernels, cfg: *const glm.config.Config, ranks: *[2]Rank, run_: *Run, steps: []const std.json.Value, comptime tag: []const u8, long: bool) !void {
    const stream = h.stream;
    const D = cfg.hidden;
    const d = h.gpu.d;
    for (steps, 0..) |sv, s| {
        const kind = sv.object.get("kind").?.string;
        const R: usize = @intCast(sv.object.get("rows").?.integer);
        const keep: usize = @intCast(sv.object.get("keep").?.integer);
        const pos: usize = @intCast(sv.object.get("pos").?.integer);
        const prefill = kind[0] == 'p';
        const before = h.failures;
        var nb: [16]u8 = undefined;
        const step = try std.fmt.bufPrint(&nb, tag ++ "{d}", .{s});
        try run_.window(pos, R, h.stream);
        // the attention site: embed, hc_pre, the DSA block
        for (ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
            const cx: F.Ctx = .{ .g = g, .kk = undefined, .temp = undefined, .prompt_scratch = undefined, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            const ds = if (prefill) &run_.pds[ri] else &run_.ds[ri];
            const cache = &run_.cache[ri];
            try h.beginStep(ri, step);
            const p32: i32 = @intCast(pos);
            try r.pos_dev.upload(0, std.mem.asBytes(&p32));
            try h.feed("ids", b.ids);
            try g.embed(b.ids.ptr, r.embed.ptr, b.x.ptr, R, cfg.streams, stream);
            try h.call("embed", "out_out", b.x.ptr);
            try F.hcPre(cx, &r.frame, false, b, R, stream);
            try h.call("hc_pre", "out_out", b.normed.ptr);
            try A.dsaBlock(g, lay, &r.dsa, cache, ds, .{ .normed = b.normed.ptr, .part = b.part.ptr, .sk = b.sk.ptr, .pos_dev = r.pos_dev.ptr, .host_pos = pos, .nch = (pos + R + 511) / 512, .R = R }, stream);
            try compareDsa(h, ds, cache, long, prefill, pos, R, cfg.denseLimit(), b.part.ptr);
        }
        try gatherRanks(ranks, prefill, R * D * 4, d, stream);
        // the FFN site: hc_post, hc_pre, the MoE
        for (ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
            const cx: F.Ctx = .{ .g = g, .kk = undefined, .temp = undefined, .prompt_scratch = undefined, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            const sc = if (prefill) &r.pms else &r.ms;
            try h.beginStep(ri, step);
            try h.call("gather", "ret", b.gath.ptr);
            try F.hcPost(cx, b, R, stream);
            try h.call("hc_post", "xout_out", b.x.ptr);
            try F.hcPre(cx, &r.frame, true, b, R, stream);
            try h.call("hc_pre", "out_out", b.normed.ptr);
            try M.moeBlock(g, mk, lay, &r.moe, sc, b.normed.ptr, b.part.ptr, b.sk.ptr, R, stream);
            try h.call("combine_shared", "out_out", b.part.ptr);
        }
        try gatherRanks(ranks, prefill, R * D * 4, d, stream);
        // the layer's end and the final norm (the MTP head's input)
        for (ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
            const cx: F.Ctx = .{ .g = g, .kk = undefined, .temp = undefined, .prompt_scratch = undefined, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.beginStep(ri, step);
            try F.hcPost(cx, b, R, stream);
            try h.stepArray("x_out", b.x.ptr);
            try g.streamMean(b.x.ptr, b.hidden.ptr, R, stream);
            try g.rmsnorm(b.hidden.ptr, D, r.norm.ptr, r.fnormed.ptr, D, r.fxs.ptr, cfg.eps, R, D, stream);
            try h.stepArray("fnormed", r.fnormed.ptr);
        }
        _ = keep; // the DSA caches commit by position: the next step's pos (the fixture's) is where it writes
        if (h.failures == before) check.pass("BITEXACT glm53 layer 3 step {s} ({s} {d} rows at {d}): both ranks equal to the reference", .{ step, kind, R, pos });
    }
}

/// Run m: the MTP layer on the fixture's rows (each step's next tokens and input rows fed from the reference), its
/// own caches from position 0, decode absorbs, chained drafts and their rewind, a prompt absorb past the dense limit.
fn runMtp(h: *Harness, g: glm.glue.Glue, mk: *const M.Kernels, tk: *const T.Kernels, cfg: *const glm.config.Config, ranks: *[2]Rank, run_: *Run, steps: []const std.json.Value) !void {
    const stream = h.stream;
    const D = cfg.hidden;
    const d = h.gpu.d;
    for (steps, 0..) |sv, s| {
        const kind = sv.object.get("kind").?.string;
        const n: usize = @intCast(sv.object.get("rows").?.integer);
        const mtp_len: usize = @intCast(sv.object.get("mtp_len").?.integer);
        const prefill = n > max_rows;
        const before = h.failures;
        var nb: [16]u8 = undefined;
        const step = try std.fmt.bufPrint(&nb, "m{d}", .{s});
        try run_.window(mtp_len, n, h.stream);
        for (ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
            const b = if (prefill) &r.pmb else &r.mb;
            const ds = if (prefill) &run_.pds[ri] else &run_.ds[ri];
            const cache = &run_.cache[ri];
            try h.beginStep(ri, step);
            const p32: i32 = @intCast(mtp_len);
            try r.pos_dev.upload(0, std.mem.asBytes(&p32));
            try h.feed("next", b.ids);
            try h.feed("hidden_in", b.hin);
            try T.front(g, cfg, &r.mtp, b, r.embed.ptr, n, mtp_len == 0, stream);
            try h.call("mm", "out_out", b.mx.ptr);
            // (enorm and hnorm, traced before eh_proj, write strided halves: eh_proj's output checks them)
            try h.call("rmsnorm", "out_out", b.normed.ptr);
            try A.dsaBlock(g, lay, &r.mtp.dsa, cache, ds, .{ .normed = b.normed.ptr, .part = b.part.ptr, .sk = b.sk.ptr, .pos_dev = r.pos_dev.ptr, .host_pos = mtp_len, .nch = (mtp_len + n + 511) / 512, .R = n }, stream);
            try compareDsa(h, ds, cache, false, prefill, mtp_len, n, cfg.denseLimit(), b.part.ptr);
        }
        try gatherMtp(ranks, prefill, n * D * 4, d, stream);
        for (ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = cfg };
            const b = if (prefill) &r.pmb else &r.mb;
            const sc = if (prefill) &r.pms else &r.ms;
            try h.beginStep(ri, step);
            try h.call("gather", "ret", b.gath.ptr);
            try T.afterAttention(g, cfg, &r.mtp, b, 2, n, stream);
            try h.call("residual_add", "xout_out", b.mx.ptr);
            try M.moeBlock(g, mk, lay, &r.mtp.moe, sc, b.normed.ptr, b.part.ptr, b.sk.ptr, n, stream);
            try h.call("combine_shared", "out_out", b.part.ptr);
        }
        try gatherMtp(ranks, prefill, n * D * 4, d, stream);
        var best: [2]struct { v: f32, id: usize } = undefined;
        for (ranks, 0..) |*r, ri| {
            const b = if (prefill) &r.pmb else &r.mb;
            try h.beginStep(ri, step);
            try T.head(g, tk, cfg, &r.mtp, b, 2, n, stream);
            _ = try h.next("residual_add");
            _ = try h.next("residual_add");
            try h.also("residual_add", "xout_out", b.mx.ptr);
            try h.stepArray("fnormed", b.fnormed.ptr);
            try h.stepArray("logits", b.logits.ptr);
            // the rank's best draft candidate (decode.sample_rows, greedy: top-1 of its vocabulary share)
            try stream.synchronize();
            const nv = r.mtp.head.n;
            const host = try h.gpu.gpa.alloc(u16, nv);
            defer h.gpu.gpa.free(host);
            try b.logits.download(0, std.mem.sliceAsBytes(host));
            var bv: f32 = -std.math.inf(f32);
            var bi: usize = 0;
            for (host, 0..) |x, i| {
                const v: f32 = @bitCast(@as(u32, x) << 16);
                if (v > bv) {
                    bv = v;
                    bi = i;
                }
            }
            best[ri] = .{ .v = bv, .id = ri * nv + bi };
        }
        // both ranks' candidates gathered, the larger value, the lower id on a tie
        const draft: i32 = @intCast(if (best[1].v > best[0].v) best[1].id else best[0].id);
        for (0..2) |ri| {
            var ab: [48]u8 = undefined;
            const want = try h.fx.bytes(try std.fmt.bufPrint(&ab, "r{d}_{s}_draft", .{ ri, step }));
            defer h.gpu.gpa.free(want);
            if (!std.mem.eql(u8, want, std.mem.asBytes(&draft))) {
                std.debug.print("FAIL glm53 {s} rank {d}: draft {d}, the reference's {d}\n", .{ step, ri, draft, std.mem.bytesToValue(i32, want[0..4]) });
                h.failures += 1;
            } else h.passes += 1;
        }
        if (h.failures == before) check.pass("BITEXACT glm53 MTP step {s} ({s} {d} rows at {d}): both ranks equal to the reference, draft {d}", .{ step, kind, n, mtp_len, draft });
    }
}

fn gatherMtp(ranks: *[2]Rank, prefill: bool, n: usize, d: *const cuda.Driver, stream: cuda.Stream) !void {
    var parts: [2]Ptr = undefined;
    var gaths: [2]Ptr = undefined;
    for (ranks, 0..) |*r, i| {
        const b = if (prefill) &r.pmb else &r.mb;
        parts[i] = b.part.ptr;
        gaths[i] = b.gath.ptr;
    }
    try hs.gather(&parts, &gaths, n, d, stream);
}

/// One DSA block's traced calls in order (cuda_dsa.zig `dsaBlock`), then its partial `part`.
fn compareDsa(h: *Harness, ds: *const A.Scratch, cache: *const A.Cache, long: bool, prefill: bool, pos: usize, R: usize, limit: usize, part: Ptr) !void {
    try h.call("mm", "out_out", ds.dp.ptr);
    try h.call("rmsnorm", "out_out", ds.qr.ptr);
    try h.call("rmsnorm", "out_out", ds.lat.ptr);
    try h.call("mm", "out_out", ds.q.ptr);
    if (!long and cache.tier == null) try h.call("latent_write", "cache_out", cache.latent.ptr) else _ = try h.next("latent_write");
    try h.call("mm", "out_out", ds.ikr.ptr);
    try h.call("router", "out_out", ds.igr.ptr);
    _ = try h.next("index_update");
    try h.alsoIf("index_update", "ik_out", cache.ik.ptr);
    try h.alsoIf("index_update", "ig_out", cache.ig.ptr);
    if (!long) try h.alsoIf("index_update", "pk_out", cache.pk.ptr);
    try h.call("absorb_q", "out_out", ds.qa.ptr);
    if (pos < limit) try h.call("dense_attention", "out_out", ds.ol.ptr);
    if (pos + R - 1 >= limit) {
        try h.call("mm", "out_out", ds.qi.ptr);
        try h.call("select_tokens", "ret0", ds.tokens.ptr);
        try h.also("select_tokens", "ret1", ds.counts.ptr);
        try h.call(if (prefill) "sparse_onepass" else "sparse_attention", "out_out", ds.ol.ptr);
    }
    try h.call("expand_v", "out_out", ds.vn.ptr);
    try h.call("mm", "out_out", part);
}

fn gatherRanks(ranks: *[2]Rank, prefill: bool, n: usize, d: *const cuda.Driver, stream: cuda.Stream) !void {
    var parts: [2]Ptr = undefined;
    var gaths: [2]Ptr = undefined;
    for (ranks, 0..) |*r, i| {
        const b = if (prefill) &r.pb else &r.b;
        parts[i] = b.part.ptr;
        gaths[i] = b.gath.ptr;
    }
    try hs.gather(&parts, &gaths, n, d, stream);
}
