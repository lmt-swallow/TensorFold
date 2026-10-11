//! `glm53-kda <work-dir> <model-dir> [isolated]`: GLM-5.3-Flash's embeddings and decoder layer 0 (KDA, the dense
//! MLP, both hyper-connections) on both ranks of the two-rank split against the Mia fork's call-by-call fixture
//! (tools/zig/glm53/dump_kda.py, packed by capture.sh): `work-dir` holds out/fixture (manifest.json + arrays, the
//! call trace), out/weights.json and aot/ (the captured Triton variants); the weights come from the EXL3 pack in
//! `model-dir` through the family's loader, each rank's share checked against the reference's sha256.
//!
//! Both ranks run on this one GPU, stage by stage; their all-gathers are copies of the two partials into each rank's
//! gather buffer (the bytes NCCL would move). Every traced call's outputs are compared with the reference's in trace
//! order, so a first mismatch names the call. By default each stage runs on this engine's own outputs; `isolated`
//! puts the reference's bytes back after every comparison (a mismatch then cannot hide the stages after it).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;

const F = glm.forward;
const Buf = cuda.DeviceBuffer;

/// The trace's calls of one rank's step in order, and the comparison bookkeeping.
const Harness = struct {
    gpu: Gpu,
    fx: fixture.Fixture,
    stream: cuda.Stream,
    isolated: bool,
    failures: usize = 0,
    passes: usize = 0,
    // the current (rank, step)'s calls: their seq numbers and function names; `at` the next unread
    seqs: std.ArrayList(i64) = .empty,
    fns: std.ArrayList([]const u8) = .empty,
    at: usize = 0,
    rank: usize = 0,
    step: usize = 0,
    nbuf: [128]u8 = undefined,

    fn deinit(h: *Harness) void {
        h.seqs.deinit(h.gpu.gpa);
        h.fns.deinit(h.gpu.gpa);
    }

    /// Points the harness at rank r's calls of step s.
    fn begin(h: *Harness, r: usize, s: usize) !void {
        h.seqs.clearRetainingCapacity();
        h.fns.clearRetainingCapacity();
        h.at = 0;
        h.rank = r;
        h.step = s;
        var sb: [16]u8 = undefined;
        const step = try std.fmt.bufPrint(&sb, "s{d}", .{s});
        const trace = h.fx.parsed.value.object.get("trace").?.array.items;
        for (trace) |c| {
            const o = c.object;
            if (o.get("rank").?.integer != r or !std.mem.eql(u8, o.get("step").?.string, step)) continue;
            const f = o.get("fn").?.string;
            try h.seqs.append(h.gpu.gpa, o.get("seq").?.integer);
            try h.fns.append(h.gpu.gpa, f[std.mem.lastIndexOfScalar(u8, f, '.').? + 1 ..]);
        }
    }

    /// The next traced call of `function` (calls of others before it are skipped: group sums no BF16 matmul reads).
    fn next(h: *Harness, function: []const u8) !i64 {
        while (h.at < h.fns.items.len) : (h.at += 1) {
            if (std.mem.eql(u8, h.fns.items[h.at], function)) {
                h.at += 1;
                return h.seqs.items[h.at - 1];
            }
        }
        std.debug.print("FAIL glm53 r{d} s{d}: the trace has no further {s} call\n", .{ h.rank, h.step, function });
        return error.TestFailed;
    }

    fn arrayName(h: *Harness, seq: i64, function: []const u8, what: []const u8) ![]const u8 {
        return std.fmt.bufPrint(&h.nbuf, "r{d}_s{d}_{d:0>3}_{s}_{s}", .{ h.rank, h.step, @as(u64, @intCast(seq)), function, what });
    }

    /// The device bytes at `ptr` against the fixture array `name` (its length): a PASS, or a FAIL counted with the
    /// first differing byte; isolated runs then put the reference's bytes there.
    fn same(h: *Harness, name: []const u8, ptr: cuda.abi.DevicePtr) !void {
        try h.stream.synchronize();
        const want = h.fx.bytes(name) catch |e| {
            std.debug.print("FAIL glm53 {s}: not in the fixture ({t})\n", .{ name, e });
            h.failures += 1;
            return;
        };
        defer h.gpu.gpa.free(want);
        const got = try h.gpu.gpa.alloc(u8, want.len);
        defer h.gpu.gpa.free(got);
        const view: Buf = .{ .d = h.gpu.d, .ptr = ptr, .len = want.len };
        try view.download(0, got);
        if (std.mem.indexOfDiff(u8, got, want)) |at| {
            var diff: usize = 0;
            for (got, want) |a, b| diff += @intFromBool(a != b);
            std.debug.print("FAIL glm53 {s}: {d} of {d} bytes differ, the first at byte {d}\n", .{ name, diff, want.len, at });
            h.failures += 1;
        } else h.passes += 1;
        if (h.isolated) try view.upload(0, want);
    }

    /// The next `function` call's output `what` (an argument's "<arg>_out", or "ret") against `ptr`.
    fn call(h: *Harness, function: []const u8, what: []const u8, ptr: cuda.abi.DevicePtr) !void {
        const seq = try h.next(function);
        try h.same(try h.arrayName(seq, function, what), ptr);
    }

    /// The same call's further outputs (after `call`).
    fn also(h: *Harness, function: []const u8, what: []const u8, ptr: cuda.abi.DevicePtr) !void {
        try h.same(try h.arrayName(h.seqs.items[h.at - 1], function, what), ptr);
    }

    /// A per-step array `r{r}_s{s}_{what}` against `ptr`.
    fn stepArray(h: *Harness, what: []const u8, ptr: cuda.abi.DevicePtr) !void {
        var b: [96]u8 = undefined;
        try h.same(try std.fmt.bufPrint(&b, "r{d}_s{d}_{s}", .{ h.rank, h.step, what }), ptr);
    }

    fn feed(h: *Harness, what: []const u8, buf: Buf) !void {
        var b: [96]u8 = undefined;
        const w = try h.fx.bytes(try std.fmt.bufPrint(&b, "r{d}_s{d}_{s}", .{ h.rank, h.step, what }));
        defer h.gpu.gpa.free(w);
        try buf.upload(0, w);
    }
};

/// One rank's layer 0 and its buffers.
const Rank = struct {
    frame: glm.weights.Frame,
    kda: glm.weights.Kda,
    mlp: glm.weights.Mlp,
    embed: Buf,
    b: F.Buffers,
    pb: F.Buffers,
    st: F.KdaState,

    fn deinit(r: *Rank) void {
        r.frame.deinit();
        r.kda.deinit();
        r.mlp.deinit();
        r.embed.free();
        r.b.deinit();
        r.pb.deinit();
        r.st.deinit();
    }
};

fn sha(gpa: std.mem.Allocator, buf: Buf) ![64]u8 {
    const host = try gpa.alloc(u8, buf.len);
    defer gpa.free(host);
    try buf.download(0, host);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(host, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

/// Each loaded tensor's sha256 against weights.json's (the reference loader's bytes).
fn checkWeights(gpa: std.mem.Allocator, io: std.Io, work: []const u8, ranks: []Rank) !void {
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
        const pairs = [_]struct { []const u8, Buf }{
            .{ "layers.0.attn_hc.fn", r.frame.attn_hc.fn_ },   .{ "layers.0.attn_hc.base", r.frame.attn_hc.base }, .{ "layers.0.attn_hc.scale", r.frame.attn_hc.scale },
            .{ "layers.0.ffn_hc.fn", r.frame.ffn_hc.fn_ },     .{ "layers.0.ffn_hc.base", r.frame.ffn_hc.base },   .{ "layers.0.ffn_hc.scale", r.frame.ffn_hc.scale },
            .{ "layers.0.in_norm", r.frame.in_norm },          .{ "layers.0.post_norm", r.frame.post_norm },       .{ "layers.0.kda.proj.weight", r.kda.proj.buf },
            .{ "layers.0.kda.fb.weight", r.kda.fb.buf },       .{ "layers.0.kda.gb.weight", r.kda.gb.buf },        .{ "layers.0.kda.conv", r.kda.conv },
            .{ "layers.0.kda.a_log", r.kda.a_log },            .{ "layers.0.kda.dt_bias", r.kda.dt_bias },         .{ "layers.0.kda.norm", r.kda.norm },
            .{ "layers.0.kda.o.weight", r.kda.o.buf },         .{ "layers.0.mlp.gu.weight", r.mlp.gu.buf },        .{ "layers.0.mlp.down.weight", r.mlp.down.buf },
            .{ "embed", r.embed },
        };
        for (pairs) |p| {
            const got = try sha(gpa, p[1]);
            n += 1;
            if (!std.mem.eql(u8, &got, want.get(p[0]).?.string)) {
                std.debug.print("FAIL glm53 rank {d} {s}: sha256 {s}, the reference's {s}\n", .{ ri, p[0], got[0..16], want.get(p[0]).?.string[0..16] });
                bad += 1;
            }
        }
    }
    try check.expect(bad == 0, "glm53-kda: {d} of {d} loaded tensors differ from the reference loader's", .{ bad, n });
    check.pass("glm53-kda: every rank's layer-0 tensors equal the reference loader's ({d} sha256)", .{n});
}

pub fn run(gpu: Gpu, work: []const u8, model: []const u8, isolated: bool) !void {
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
    var h: Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir), .stream = try cuda.Stream.init(d, true), .isolated = isolated };
    defer h.fx.deinit();
    defer h.stream.deinit();
    defer h.deinit();
    const stream = h.stream;
    const rows: usize = @intCast(try h.fx.int("rows"));
    const prompt_rows: usize = @intCast(try h.fx.int("prompt_rows"));

    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var kk = try glm.kda.Kernels.load(d);
    defer kk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    var ranks: [2]Rank = undefined;
    var loaded: usize = 0;
    defer for (ranks[0..loaded]) |*r| r.deinit();
    for (&ranks, 0..) |*r, ri| {
        const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
        const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
        r.frame = try glm.weights.Frame.load(l, 0);
        r.kda = try glm.weights.Kda.load(l, 0);
        r.mlp = try glm.weights.Mlp.load(l, "mlp.", 0);
        r.embed = try l.share(glm.weights_map.embed);
        r.b = try F.Buffers.init(d, lay, rows, false);
        r.pb = try F.Buffers.init(d, lay, prompt_rows, true);
        r.st = try F.KdaState.init(d, lay, 1, rows);
        try r.st.rec.fill8(0, null);
        try r.st.conv.fill8(0, null);
        loaded += 1;
    }
    try checkWeights(gpa, gpu.io, work, &ranks);
    const lay0: glm.config.Layout = .{ .rank = 0, .world = 2, .cfg = &cfg };
    const H = lay0.size(.lin);
    var temp = try glm.kda.Temp.init(d, @max(rows, prompt_rows), H);
    defer temp.deinit();
    var pscratch = try glm.kda.Scratch.init(d, 1, prompt_rows, H);
    defer pscratch.deinit();
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams };

    const steps = h.fx.parsed.value.object.get("params").?.object.get("steps").?.array.items;
    const D = cfg.hidden;
    for (steps, 0..) |sv, s| {
        const kind = sv.object.get("kind").?.string;
        const R: usize = @intCast(sv.object.get("rows").?.integer);
        const keep: usize = @intCast(sv.object.get("keep").?.integer);
        const prefill = kind[0] == 'p';
        const before = h.failures;
        // the attention site: embed, hc_pre, the KDA block up to its partial
        for (&ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
            const cx: F.Ctx = .{ .g = g, .kk = &kk, .temp = &temp, .prompt_scratch = &pscratch, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.begin(ri, s);
            try h.stepArray("rec_in", r.st.state(r.st.cur, 0));
            try h.stepArray("conv_in", r.st.convOf(0));
            try h.feed("ids", b.ids);
            try g.embed(b.ids.ptr, r.embed.ptr, b.x.ptr, R, cfg.streams, stream);
            try h.call("embed", "out_out", b.x.ptr);
            try F.hcPre(cx, &r.frame, false, b, R, stream);
            try h.call("hc_pre", "out_out", b.normed.ptr);
            try h.also("hc_pre", "post_out", b.post.ptr);
            try h.also("hc_pre", "comb_out", b.comb.ptr);
            try F.kdaBlock(cx, &r.kda, &r.st, 0, b, R, stream);
            const p = if (prefill) b.kproj.ptr else r.st.proj.ptr;
            try h.call("mm", "out_out", p);
            try h.call("mm", "out_out", b.ka.ptr);
            try h.call("mm", "out_out", b.kg.ptr);
            const sc = if (prefill) pscratch.layer(0) else r.st.scratch.layer(0);
            try h.call("chain", "ret", sc.out);
            try h.also("chain", "state_out_out", r.st.state(1 - r.st.cur, 0));
            if (prefill) try h.call("_shift_conv", "conv_out", r.st.convOf(0));
            try h.call("mm", "out_out", b.part.ptr);
        }
        try gather(&ranks, prefill, R, D, stream);
        // the FFN site: hc_post, hc_pre, the dense MLP up to its partial
        for (&ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
            const cx: F.Ctx = .{ .g = g, .kk = &kk, .temp = &temp, .prompt_scratch = &pscratch, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.begin(ri, s);
            try h.call("gather", "ret", b.gath.ptr);
            try F.hcPost(cx, b, R, stream);
            try h.call("hc_post", "xout_out", b.x.ptr);
            try F.hcPre(cx, &r.frame, true, b, R, stream);
            try h.call("hc_pre", "out_out", b.normed.ptr);
            try F.mlpBlock(cx, &r.mlp, b, R, stream);
            try h.call("mm", "out_out", b.gu.ptr);
            try h.call("swiglu", "out_out", b.act.ptr);
            try h.also("swiglu", "xs_out", b.xs_act.ptr);
            try h.call("mm", "out_out", b.part.ptr);
        }
        try gather(&ranks, prefill, R, D, stream);
        // the layer's end: hc_post, the streams' mean, the commit
        for (&ranks, 0..) |*r, ri| {
            const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
            const cx: F.Ctx = .{ .g = g, .kk = &kk, .temp = &temp, .prompt_scratch = &pscratch, .lay = lay };
            const b = if (prefill) &r.pb else &r.b;
            try h.begin(ri, s);
            try F.hcPost(cx, b, R, stream);
            try h.stepArray("x_out", b.x.ptr);
            try g.streamMean(b.x.ptr, b.hidden.ptr, R, stream);
            try h.stepArray("hidden", b.hidden.ptr);
            try F.kdaCommit(cx, &r.st, R, keep, prefill, stream);
            try h.stepArray("rec_commit", r.st.state(r.st.cur, 0));
            try h.stepArray("conv_commit", r.st.convOf(0));
        }
        if (h.failures == before) check.pass("BITEXACT glm53 layer 0 step {d} ({s} {d} rows, {d} kept): both ranks equal to the reference", .{ s, kind, R, keep });
    }
    try check.expect(h.failures == 0, "glm53-kda: {d} comparisons differ ({d} equal)", .{ h.failures, h.passes });
    check.pass("glm53-kda: {d} steps, {d} comparisons bit-exact{s}", .{ steps.len, h.passes, if (isolated) " (isolated)" else "" });
}

/// The all-gather: rank 0's partial then rank 1's into every rank's gather buffer.
fn gather(ranks: []Rank, prefill: bool, R: usize, D: usize, stream: cuda.Stream) !void {
    const n = R * D * 4;
    for (ranks) |*dst| {
        const gb = if (prefill) dst.pb.gath else dst.b.gath;
        for (ranks, 0..) |*src, k| {
            const part = if (prefill) src.pb.part else src.b.part;
            try gb.copyFrom(k * n, part.ptr, n, stream.handle);
        }
    }
}
