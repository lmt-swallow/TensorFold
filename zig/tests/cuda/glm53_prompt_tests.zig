//! `glm53-prompt <work-dir> <model-dir>`: GLM-5.3-Flash's prompt path (cuda_prompt.zig: chunks, the head of a
//! chunk's last row, the MTP absorb, the commit) on both ranks of the two-rank split, over the model cut down to
//! layers 0 (KDA) and 3 (DSA, MoE) and the MTP layer, against the Mia fork's fixture (tools/zig/glm53/dump_prompt.py,
//! packed by capture.sh prompt): every scenario's chunks from a fresh stream, then the long run over drawn caches.
//!
//! The fixture holds digests, not arrays: each traced call's changed arguments and return values (sha256 and byte
//! length), each chunk's states, and the head's logits of each chunk's last row. A segment of a chunk
//! (`prompt.Chunk.run`) runs on both ranks in turn on this one GPU, the all-gather between segments copies the two
//! partials (the bytes NCCL would move); after each segment its calls' outputs are compared in trace order, so a
//! first mismatch names the call. An argument the call left unchanged has no digest and is not compared.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const hs = @import("glm53_harness.zig");
const Gpu = check.Gpu;

const F = glm.forward;
const M = glm.moe;
const A = glm.dsa;
const T = glm.mtp;
const P = glm.prompt;
const Buf = cuda.DeviceBuffer;
const Ptr = cuda.abi.DevicePtr;

const Json = std.json.Value;

/// One rank's trace cursor over the current step's calls (dump_prompt.py's DigestTracer entries).
const Cursor = struct {
    calls: std.ArrayList(std.json.ObjectMap) = .empty,
    at: usize = 0,
    last: ?std.json.ObjectMap = null,
};

/// The digest comparisons of both ranks.
const Digests = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    trace: []const Json,
    steps: std.json.ObjectMap, // params.digests
    cur: [2]Cursor = .{ .{}, .{} },
    rank: usize = 0,
    step: []const u8 = "",
    failures: usize = 0,
    passes: usize = 0,
    unchanged: usize = 0,
    split: ?P.Split = null, // the fixture's hcsplit settings (dump_prompt.py --split)

    fn deinit(h: *Digests) void {
        for (&h.cur) |*c| c.calls.deinit(h.gpa);
    }

    fn begin(h: *Digests, rank: usize, step: []const u8) !void {
        const c = &h.cur[rank];
        c.calls.clearRetainingCapacity();
        c.at = 0;
        c.last = null;
        for (h.trace) |v| {
            const o = v.object;
            if (o.get("rank").?.integer != rank or !std.mem.eql(u8, o.get("step").?.string, step)) continue;
            try c.calls.append(h.gpa, o);
        }
    }

    fn on(h: *Digests, rank: usize, step: []const u8) void {
        h.rank = rank;
        h.step = step;
    }

    fn fnName(o: std.json.ObjectMap) []const u8 {
        const f = o.get("fn").?.string;
        return f[std.mem.lastIndexOfScalar(u8, f, '.').? + 1 ..];
    }

    /// Moves the rank's cursor to its next call of `function` (calls of others before it are skipped).
    fn next(h: *Digests, function: []const u8) !void {
        const c = &h.cur[h.rank];
        while (c.at < c.calls.items.len) : (c.at += 1) {
            if (std.mem.eql(u8, fnName(c.calls.items[c.at]), function)) {
                c.last = c.calls.items[c.at];
                c.at += 1;
                return;
            }
        }
        std.debug.print("FAIL glm53-prompt r{d} {s}: the trace has no further {s} call\n", .{ h.rank, h.step, function });
        return error.TestFailed;
    }

    fn same(h: *Digests, label: []const u8, what: []const u8, want: Json, ptr: Ptr) !void {
        try h.stream.synchronize();
        const hex = want.array.items[0].string;
        const n: usize = @intCast(want.array.items[1].integer);
        const host = try h.gpa.alloc(u8, n);
        defer h.gpa.free(host);
        const view: Buf = .{ .d = h.d, .ptr = ptr, .len = n };
        try view.download(0, host);
        var out: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &out, .{});
        const got = std.fmt.bytesToHex(out, .lower);
        if (std.mem.eql(u8, &got, hex)) {
            h.passes += 1;
        } else {
            std.debug.print("FAIL glm53-prompt r{d} {s} {s} {s}: sha256 {s} ({d} bytes), the reference's {s}\n", .{ h.rank, h.step, label, what, got[0..16], n, hex[0..16] });
            h.failures += 1;
        }
    }

    /// The next `function` call's output `what` (a changed argument's name, or ret / ret0 / ret1) against `ptr`.
    fn call(h: *Digests, function: []const u8, what: []const u8, ptr: Ptr) !void {
        try h.next(function);
        try h.also(what, ptr);
    }

    /// The same call's further output `what`.
    fn also(h: *Digests, what: []const u8, ptr: Ptr) !void {
        const o = h.cur[h.rank].last.?;
        const want = o.get("digests").?.object.get(what) orelse {
            h.unchanged += 1;
            return;
        };
        try h.same(fnName(o), what, want, ptr);
    }

    /// The step state r{rank}_{step}_{what} of params.digests.
    fn state(h: *Digests, what: []const u8, ptr: Ptr) !void {
        var b: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&b, "r{d}_{s}_{s}", .{ h.rank, h.step, what });
        const want = h.steps.get(name) orelse {
            std.debug.print("FAIL glm53-prompt {s}: not in the fixture's digests\n", .{name});
            h.failures += 1;
            return;
        };
        try h.same("state", what, want, ptr);
    }
};

/// One rank's cut-down model, its stream state and its prompt buffers.
const Rank = struct {
    f0: glm.weights.Frame,
    kda: glm.weights.Kda,
    mlp: glm.weights.Mlp,
    f3: glm.weights.Frame,
    dsa: A.Dsa,
    moe: M.Moe,
    mtp: T.Mtp,
    embed: Buf,
    norm: Buf,
    head_rows: []u8,
    head: Buf,
    p: P.Owned,
    st: F.KdaState,
    pos_dev: Buf,
    mpos_dev: Buf,
    caches: [1]A.Cache = undefined,
    mcache: A.Cache = undefined,
    have_caches: bool = false,
    layers: [2]P.Layer = undefined,
    model: P.Model = undefined,
    stream: P.Stream = undefined,

    fn deinit(r: *Rank, gpa: std.mem.Allocator) void {
        r.f0.deinit();
        r.kda.deinit();
        r.mlp.deinit();
        r.f3.deinit();
        r.dsa.deinit();
        r.moe.deinit();
        r.mtp.deinit();
        r.embed.free();
        r.norm.free();
        gpa.free(r.head_rows);
        r.head.free();
        r.p.deinit();
        r.st.deinit();
        r.pos_dev.free();
        r.mpos_dev.free();
        r.dropCaches();
    }

    fn dropCaches(r: *Rank) void {
        if (!r.have_caches) return;
        r.caches[0].deinit();
        r.mcache.deinit();
        r.have_caches = false;
    }

    /// A fresh stream over `capacity` tokens: zero caches, KDA state and conv windows, position 0.
    fn fresh(r: *Rank, d: *const cuda.Driver, cfg: *const glm.config.Config, capacity: usize, ring: usize, kv: A.Kv) !void {
        r.dropCaches();
        r.caches[0] = try A.Cache.init(d, cfg, capacity, ring, kv);
        r.mcache = try A.Cache.init(d, cfg, capacity, ring, kv);
        r.have_caches = true;
        try r.st.rec.fill8(0, null);
        try r.st.conv.fill8(0, null);
        r.st.cur = 0;
        r.st.pos = 0;
        r.layers = .{
            .{ .frame = &r.f0, .mixer = .{ .kda = .{ .w = &r.kda, .li = 0 } }, .ffn = .{ .mlp = &r.mlp } },
            .{ .frame = &r.f3, .mixer = .{ .dsa = .{ .w = &r.dsa, .di = 0 } }, .ffn = .{ .moe = &r.moe } },
        };
        r.model = .{
            .layers = &r.layers, .embed = r.embed.ptr, .norm = r.norm.ptr,
            .head = .{ .ptr = r.head.ptr, .n = r.head_rows.len / 2 / cfg.hidden, .k = cfg.hidden },
            .mtp = &r.mtp,
        };
        r.stream = .{
            .kda = &r.st, .caches = &r.caches, .mtp_cache = &r.mcache, .pos = 0, .mtp_len = 0, .capacity = capacity,
            .pos_dev = r.pos_dev.ptr, .mtp_pos_dev = r.mpos_dev.ptr,
        };
    }
};

/// dump_prompt.py `draw`: element i of a bf16 tensor drawn with `seed`.
fn drawn(i: u64, seed: u64) u16 {
    var h = (i *% 0x9E3779B1 +% seed *% 0x85EBCA6B) & 0xFFFFFFFF;
    h ^= h >> 15;
    const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((h >> 7) & 255)) - 128)) / 64.0;
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

fn draw(gpa: std.mem.Allocator, b: Buf, seed: u64) !void {
    const n = b.len / 2;
    const step: usize = 1 << 24;
    const host = try gpa.alloc(u16, step);
    defer gpa.free(host);
    var a: usize = 0;
    while (a < n) : (a += step) {
        const m = @min(step, n - a);
        for (host[0..m], 0..) |*x, j| x.* = drawn(a + j, seed);
        try b.upload(a * 2, std.mem.sliceAsBytes(host[0..m]));
    }
}

fn weightsOk(gpa: std.mem.Allocator, io: std.Io, work: []const u8, ranks: []Rank) !void {
    const path = try std.fs.path.join(gpa, &.{ work, "out", "weights.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(Json, gpa, text, .{});
    defer parsed.deinit();
    var bad: usize = 0;
    var n: usize = 0;
    for (ranks, 0..) |*r, ri| {
        var kb: [16]u8 = undefined;
        const want = parsed.value.object.get(try std.fmt.bufPrint(&kb, "rank{d}", .{ri})).?.object;
        const pairs = [_]struct { []const u8, Buf }{
            .{ "layers.0.kda.proj.weight", r.kda.proj.buf }, .{ "layers.0.mlp.down.weight", r.mlp.down.buf },
            .{ "layers.0.attn_hc.fn", r.f0.attn_hc.fn_ },     .{ "layers.3.dsa.proj.weight", r.dsa.proj.buf },
            .{ "layers.3.dsa.absorb.wk", r.dsa.wk },          .{ "layers.3.ffn_hc.fn", r.f3.ffn_hc.fn_ },
            .{ "norm", r.norm },                              .{ "embed", r.embed },
            .{ "head.weight", r.head },
        };
        for (pairs) |p| {
            const got = try hs.sha(gpa, p[1]);
            n += 1;
            const w = want.get(p[0]) orelse {
                std.debug.print("FAIL glm53-prompt rank {d}: weights.json has no {s}\n", .{ ri, p[0] });
                bad += 1;
                continue;
            };
            if (!std.mem.eql(u8, &got, w.string)) {
                std.debug.print("FAIL glm53-prompt rank {d} {s}: sha256 {s}, the reference's {s}\n", .{ ri, p[0], got[0..16], w.string[0..16] });
                bad += 1;
            }
        }
    }
    try check.expect(bad == 0, "glm53-prompt: {d} of {d} loaded tensors differ from the reference loader's", .{ bad, n });
    check.pass("glm53-prompt: the cut-down model's tensors equal the reference loader's ({d} sha256)", .{n});
}

/// The fixture's DSA cache format (params "kv": dump_*.py with TF_GLM53_REF_KV; absent: bf16), required to be `want`
/// when given (the -fp8 subcommands).
fn fixtureKv(params: std.json.ObjectMap, want: ?A.Kv) !A.Kv {
    const kv = try A.Kv.parse(if (params.get("kv")) |v| v.string else "");
    if (want) |w| try check.expect(kv == w, "glm53: the fixture's DSA caches are {t}, not {t} (capture it with KV={t})", .{ kv, w, w });
    return kv;
}

/// `want`: the cache format the fixture must hold (glm53-prompt-fp8), or any (glm53-prompt).
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
    // the manifest (its trace of digests) is larger than fixture.Fixture.open reads
    const fx_dir = try std.fs.path.join(gpa, &.{ work, "out", "fixture" });
    defer gpa.free(fx_dir);
    const mpath = try std.fs.path.join(gpa, &.{ fx_dir, "manifest.json" });
    defer gpa.free(mpath);
    const mtext = try std.Io.Dir.cwd().readFileAlloc(gpu.io, mpath, gpa, .limited(1 << 28));
    defer gpa.free(mtext);
    var fx: fixture.Fixture = .{ .gpa = gpa, .io = gpu.io, .dir = fx_dir, .parsed = try std.json.parseFromSlice(Json, gpa, mtext, .{}) };
    defer fx.deinit();
    const params = fx.parsed.value.object.get("params").?.object;
    const capacity: usize = @intCast(params.get("capacity").?.integer);
    const long_capacity: usize = @intCast(params.get("long_capacity").?.integer);
    const prompt_rows: usize = @intCast(params.get("prompt_rows").?.integer);
    const ring: usize = @intCast(params.get("ring").?.integer);
    const long_ring: usize = @intCast(params.get("long_ring").?.integer);
    const kv = try fixtureKv(params, want);
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var h: Digests = .{ .gpa = gpa, .d = d, .stream = stream, .trace = fx.parsed.value.object.get("trace").?.array.items, .steps = params.get("digests").?.object };
    defer h.deinit();
    if (params.get("split")) |sv| if (sv == .object) {
        h.split = .{
            .pieces = @intCast(sv.object.get("pieces").?.integer), .min_rows = @intCast(sv.object.get("min_rows").?.integer),
            .fronts = sv.object.get("fronts").?.bool,
        };
    };

    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var kk = try glm.kda.Kernels.load(d);
    defer kk.deinit();
    var mk = try M.Kernels.load(d);
    defer mk.deinit();
    var tk = try T.Kernels.load(d);
    defer tk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    var ranks: [2]Rank = undefined;
    var loaded: usize = 0;
    defer for (ranks[0..loaded]) |*r| r.deinit(gpa);
    for (&ranks, 0..) |*r, ri| {
        const lay: glm.config.Layout = .{ .rank = ri, .world = 2, .cfg = &cfg };
        const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
        r.* = .{
            .f0 = try glm.weights.Frame.load(l, 0),
            .kda = try glm.weights.Kda.load(l, 0),
            .mlp = try glm.weights.Mlp.load(l, "mlp.", 0),
            .f3 = try glm.weights.Frame.load(l, 3),
            .dsa = try A.Dsa.load(l, 3),
            .moe = try M.Moe.load(l, 3),
            .mtp = try T.Mtp.load(l),
            .embed = try l.share(glm.weights_map.embed),
            .norm = try l.share(glm.weights_map.norm),
            .head_rows = try l.headRows(),
            .head = undefined,
            .p = undefined,
            .st = undefined,
            .pos_dev = try Buf.alloc(d, 16),
            .mpos_dev = try Buf.alloc(d, 16),
        };
        r.head = try Buf.fromHost(d, r.head_rows);
        r.p = try P.Owned.init(d, lay, prompt_rows, long_capacity);
        r.st = try F.KdaState.init(d, lay, 1, 16);
        loaded += 1;
    }
    try weightsOk(gpa, gpu.io, work, &ranks);
    const g: glm.glue.Glue = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams };
    const k: P.Kernels = .{ .g = g, .kda = &kk, .moe = &mk, .mtp = &tk };

    const steps = params.get("steps").?.object;
    var it = steps.iterator();
    var chunks: usize = 0;
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const long = std.mem.eql(u8, name, "l");
        for (&ranks) |*r| try r.fresh(d, &cfg, if (long) long_capacity else capacity, if (long) long_ring else ring, kv);
        if (long) for (&ranks) |*r| {
            // dump_prompt.py's draws: layer 3's latents and pooled keys, then the MTP layer's
            for ([_]Buf{ r.caches[0].latent, r.caches[0].pk, r.mcache.latent, r.mcache.pk }, 0..) |b, seed| try draw(gpa, b, seed);
        };
        const list = e.value_ptr.array.items;
        for (list, 0..) |sv, si| {
            try chunk(&h, &fx, k, &cfg, &ranks, sv.object, if (si + 1 < list.len) list[si + 1].object else null);
            chunks += 1;
        }
    }
    try check.expect(h.failures == 0, "glm53-prompt: {d} comparisons differ ({d} equal)", .{ h.failures, h.passes });
    check.pass("glm53-prompt: {d} chunks, {d} comparisons bit-exact ({d} unchanged arguments not compared)", .{ chunks, h.passes, h.unchanged });
}

/// One chunk on both ranks, segment by segment, each segment's calls compared after it.
fn chunk(h: *Digests, fx: *fixture.Fixture, k: P.Kernels, cfg: *const glm.config.Config, ranks: *[2]Rank, s: std.json.ObjectMap, after: ?std.json.ObjectMap) !void {
    const gpa = h.gpa;
    const step = s.get("step").?.string;
    const pos: usize = @intCast(s.get("pos").?.integer);
    const R: usize = @intCast(s.get("rows").?.integer);
    const mtp0: usize = @intCast(s.get("mtp_len").?.integer);
    const absorbed: usize = @intCast(s.get("absorbed").?.integer);
    const before = h.failures;
    var raws: [2][]u8 = undefined;
    var ids: [2][]const u32 = undefined;
    var next: [2][]u32 = undefined;
    var cs: [2]P.Chunk = undefined;
    for (ranks, 0..) |*r, ri| {
        var nb: [64]u8 = undefined;
        raws[ri] = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_{s}_ids", .{ ri, step }));
        ids[ri] = @alignCast(std.mem.bytesAsSlice(u32, raws[ri]));
        // the next tokens: the chunk's own past its first row, then the next chunk's first (unless the prompt ends)
        next[ri] = try gpa.alloc(u32, absorbed);
        @memcpy(next[ri][0..@min(absorbed, R - 1)], ids[ri][1..][0..@min(absorbed, R - 1)]);
        if (absorbed == R) {
            const a = after orelse return error.BadFixture;
            const raw2 = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_{s}_ids", .{ ri, a.get("step").?.string }));
            defer gpa.free(raw2);
            next[ri][R - 1] = std.mem.bytesToValue(u32, raw2[0..4]);
        }
        // a prompt continuing a cached context (the long run) starts where the fixture says
        if (r.stream.pos != pos) {
            r.stream.pos = pos;
            r.st.pos = pos;
        }
        r.stream.mtp_len = mtp0;
        cs[ri] = .{ .p = r.p.view(), .m = &r.model, .k = k, .lay = .{ .rank = ri, .world = 2, .cfg = cfg }, .st = &r.stream, .ids = ids[ri], .next = next[ri], .split = h.split };
        try h.begin(ri, step);
    }
    defer for (0..2) |ri| {
        gpa.free(raws[ri]);
        gpa.free(next[ri]);
    };
    const split = cs[0].splits();
    var plans: [2][]P.Op = undefined;
    for (0..2) |ri| plans[ri] = try cs[ri].plan(gpa);
    defer for (plans) |x| gpa.free(x);
    // each rank runs its ops up to its next gather; the gathers then move both ranks' bytes (rank 0's first)
    var at: [2]usize = .{ 0, 0 };
    while (true) {
        for (ranks, 0..) |*r, ri| {
            h.on(ri, step);
            while (at[ri] < plans[ri].len) : (at[ri] += 1) {
                const op = plans[ri][at[ri]];
                switch (op) {
                    .gather => break,
                    .wait => {},
                    else => {
                        try cs[ri].exec(op, h.stream);
                        try compare(h, &cs[ri], r, op, pos, R, mtp0);
                    },
                }
            }
        }
        if (at[0] == plans[0].len and at[1] == plans[1].len) break;
        if (at[0] == plans[0].len or at[1] == plans[1].len) return error.RanksOutOfStep;
        const x0 = plans[0][at[0]].gather;
        const x1 = plans[1][at[1]].gather;
        if (x0.bytes != x1.bytes) return error.RanksOutOfStep;
        const xs = [2]@TypeOf(x0){ x0, x1 };
        for (xs) |x| {
            const dst: Buf = .{ .d = h.d, .ptr = x.recv, .len = 2 * x.bytes };
            for (xs, 0..) |y, k2| try dst.copyFrom(k2 * x.bytes, y.send, x.bytes, h.stream.handle);
        }
        for (xs, 0..) |x, ri| {
            if (x.copy) |cp| {
                const dst: Buf = .{ .d = h.d, .ptr = cp.dst, .len = cp.bytes };
                try dst.copyFrom(0, cp.src, cp.bytes, h.stream.handle);
            } else if (x.recv == ranks[ri].p.mb.gath.ptr or (!split and x.recv == ranks[ri].p.b.gath.ptr)) {
                // the reference's all-gathers are traced (forward.py `gather`); the split's swaps are not
                h.on(ri, step);
                try h.call("gather", "ret", x.recv);
            }
            at[ri] += 1;
        }
    }
    // the chunk's states
    for (ranks, 0..) |*r, ri| {
        h.on(ri, step);
        try h.state("x_out", r.p.b.x.ptr);
        try h.state("rec", r.st.state(r.st.cur, 0));
        try h.state("conv", r.st.convOf(0));
        try h.state("latent", r.caches[0].latent.ptr + r.caches[0].latentAt(cfg, pos));
        try h.state("pk", r.caches[0].pk.ptr);
        if (absorbed > 0) try h.state("mtp_latent", r.mcache.latent.ptr + r.mcache.latentAt(cfg, mtp0));
        var nb: [64]u8 = undefined;
        const want = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_{s}_head", .{ ri, step }));
        defer gpa.free(want);
        const got = try gpa.alloc(u8, want.len);
        defer gpa.free(got);
        try h.stream.synchronize();
        try r.p.logits.download(0, got);
        if (std.mem.eql(u8, got, want)) h.passes += 1 else {
            std.debug.print("FAIL glm53-prompt r{d} {s}: the head's logits differ\n", .{ ri, step });
            h.failures += 1;
        }
        if (r.stream.pos != pos + R or r.stream.mtp_len != mtp0 + absorbed) return error.TestFailed;
    }
    if (h.failures == before) check.pass("BITEXACT glm53-prompt {s}: {d} rows at {d} (MTP {d} rows at {d}), both ranks equal to the reference", .{ step, R, pos, absorbed, mtp0 });
}

/// An op's traced calls (the reference's call order) against this rank's buffers, right after the op.
fn compare(h: *Digests, c: *const P.Chunk, r: *Rank, op: P.Op, pos: usize, R: usize, mtp0: usize) !void {
    const p = &r.p;
    const b = &p.b;
    const cfg = c.lay.cfg;
    const D = cfg.hidden;
    const S = cfg.streams;
    const limit = cfg.denseLimit();
    switch (op) {
        .embed => try h.call("embed", "out", b.x.ptr),
        .zero_pad, .gather, .wait, .commit => {},
        .hc_pre => |x| {
            try h.call("hc_pre", "out", b.normed.ptr + x.lo * D * 2);
            try h.also("post", b.post.ptr + x.lo * S * 4);
            try h.also("comb", b.comb.ptr + x.lo * S * S * 4);
        },
        .hc_post => try h.call("hc_post", "xout", b.x.ptr),
        .post_pair => |x| try h.call("hc_post_pair", "xout", b.x.ptr + x.lo * S * D * 2),
        .mean => |x| try h.call("stream_mean", "out", b.hidden.ptr + x.lo * D * 2),
        .front => |x| switch (x.site) {
            .mixer => switch (c.m.layers[x.l].mixer) {
                .kda => |k| {
                    const LW = k.w.heads * glm.kda.dk;
                    try h.call("mm", "out", b.kproj.ptr + x.lo * c.lay.kdaProj() * 2);
                    try h.call("mm", "out", b.ka.ptr + x.lo * LW * 2);
                    try h.call("mm", "out", b.kg.ptr + x.lo * LW * 2);
                },
                .dsa => |a| {
                    const ds = &p.ds;
                    try h.call("mm", "out", ds.dp.ptr + x.lo * (cfg.q_lora + cfg.kv_lora) * 2);
                    try h.call("rmsnorm", "out", ds.qr.ptr + x.lo * cfg.q_lora * 2);
                    try h.call("rmsnorm", "out", ds.lat.ptr + x.lo * cfg.kv_lora * 2);
                    try h.call("mm", "out", ds.q.ptr + x.lo * a.w.heads * cfg.qk_dim * 2);
                },
            },
            .ffn => switch (c.m.layers[x.l].ffn) {
                .mlp => |w| {
                    try h.call("mm", "out", b.gu.ptr + x.lo * 2 * w.width * 2);
                    try h.call("swiglu", "out", b.act.ptr + x.lo * w.width * 2);
                    try h.also("xs", b.xs_act.ptr + x.lo * w.width / 64 * 4);
                },
                .moe => |w| {
                    const ms = &p.ms;
                    try h.call("mm", "out", ms.sgu.ptr + x.lo * 2 * w.shared.width * 2);
                    try h.call("swiglu", "out", ms.sact.ptr + x.lo * w.shared.width * 2);
                    try h.call("mm", "out", ms.sy.ptr + x.lo * D * 4);
                },
            },
        },
        .rest => |x| switch (x.site) {
            .mixer => switch (c.m.layers[x.l].mixer) {
                .kda => {
                    try h.call("chain", "ret", p.kscratch.layer(0).out);
                    try h.also("state_out", r.st.state(1 - r.st.cur, 0));
                    try h.call("_shift_conv", "conv", r.st.convOf(0));
                },
                .dsa => |a| try compareDsaRows(h, &p.ds, &r.caches[a.di], pos, R, limit),
            },
            .ffn => {
                const ms = &p.ms;
                try h.call("router", "out", ms.logits.ptr);
                try h.call("select", "pick", ms.pick.ptr);
                try h.also("wts", ms.wts.ptr);
            },
        },
        .fill => |x| switch (x.site) {
            .ffn => if (c.m.layers[x.l].ffn == .moe)
                try h.call("combine_shared", "out", b.part.ptr + x.lo * D * 4)
            else
                try h.call("mm", "out", b.part.ptr + x.lo * D * 4),
            .mixer => try h.call("mm", "out", b.part.ptr + x.lo * D * 4),
        },
        .tail => {
            try h.call("rmsnorm", "out", p.fnormed.ptr);
            try h.also("xs", p.fxs.ptr);
            if (c.head) try h.call("mm", "out", p.logits.ptr);
        },
        .mtp_front => {
            const mb = &p.mb;
            // the embedding's first row is zeroed after the call at a stream's first MTP position
            if (mtp0 > 0) try h.call("embed", "out", mb.me.ptr) else try h.next("embed");
            try h.next("rmsnorm"); // enorm and hnorm write strided halves of mcat: eh_proj's output checks them
            try h.next("rmsnorm");
            try h.call("mm", "out", mb.mx.ptr);
            try h.call("rmsnorm", "out", mb.normed.ptr);
            try compareDsa(h, &p.ds, &r.mcache, mtp0, c.next.len, limit);
            try h.call("mm", "out", mb.part.ptr);
        },
        .mtp_mid => {
            const mb = &p.mb;
            try h.call("residual_add", "xout", mb.mx.ptr);
            try h.call("rmsnorm", "out", mb.normed.ptr);
            try compareMoe(h, &p.ms, mb.part.ptr);
        },
        .mtp_end => {
            const mb = &p.mb;
            try h.call("residual_add", "xout", mb.mx.ptr);
            try h.call("rmsnorm", "out", mb.fnormed.ptr);
            try h.call("mm", "out", mb.logits.ptr);
        },
    }
}

/// A DSA block's calls up to (not including) o_proj (cuda_dsa.zig `dsaBlock`, forward.py `dsa_front`, `dsa_rows`).
fn compareDsa(h: *Digests, ds: *const A.Scratch, cache: *const A.Cache, pos: usize, R: usize, limit: usize) !void {
    try h.call("mm", "out", ds.dp.ptr);
    try h.call("rmsnorm", "out", ds.qr.ptr);
    try h.call("rmsnorm", "out", ds.lat.ptr);
    try h.call("mm", "out", ds.q.ptr);
    try compareDsaRows(h, ds, cache, pos, R, limit);
}

/// forward.py `dsa_rows`' calls (cuda_dsa.zig `dsaRows`).
fn compareDsaRows(h: *Digests, ds: *const A.Scratch, cache: *const A.Cache, pos: usize, R: usize, limit: usize) !void {
    try h.call("latent_write", "cache", cache.latent.ptr);
    try h.call("mm", "out", ds.ikr.ptr);
    try h.call("router", "out", ds.igr.ptr);
    try h.next("index_update");
    try h.also("ik", cache.ik.ptr);
    try h.also("ig", cache.ig.ptr);
    try h.also("pk", cache.pk.ptr);
    try h.call("absorb_q", "out", ds.qa.ptr);
    if (pos < limit) try h.call("dense_attention", "out", ds.ol.ptr);
    if (pos + R - 1 >= limit) {
        try h.call("mm", "out", ds.qi.ptr);
        try h.call("select_tokens", "ret0", ds.tokens.ptr);
        try h.also("ret1", ds.counts.ptr);
        try h.call("sparse_onepass", "out", ds.ol.ptr);
    }
    try h.call("expand_v", "out", ds.vn.ptr);
}

/// A MoE block's calls (cuda_moe.zig `moeBlock`): the routing, the shared expert, the weighted sum into `part`.
fn compareMoe(h: *Digests, ms: *const M.Scratch, part: Ptr) !void {
    try h.call("router", "out", ms.logits.ptr);
    try h.call("select", "pick", ms.pick.ptr);
    try h.also("wts", ms.wts.ptr);
    try h.call("mm", "out", ms.sgu.ptr);
    try h.call("swiglu", "out", ms.sact.ptr);
    try h.call("mm", "out", ms.sy.ptr);
    try h.call("combine_shared", "out", part);
}

test "dump_prompt.py's draw" {
    // h = 0 at i = 0, seed 0: (0 - 128) / 64 = -2
    try std.testing.expectEqual(@as(u16, 0xC000), drawn(0, 0));
}
