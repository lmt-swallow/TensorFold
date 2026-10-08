//! DeepSeek-V4.1's mHC glue and MoE routing (cuda_hc.zig, cuda_route.zig) against the fork's stage-by-stage fixture
//! (tools/zig/dsv41/dump_hc.py, packed by capture_hc.sh): `dir` holds out/fixture (manifest.json + arrays) and aot/
//! (the captured Triton variants). Every stage runs on this engine's own outputs of the stage before, and each
//! stage's bytes are compared with the Python fork's, so a first mismatch names the kernel.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;

const Harness = struct {
    gpu: Gpu,
    fx: fixture.Fixture,
    stream: cuda.Stream,
    failures: usize = 0,
    buf: [96]u8 = undefined,

    fn name(self: *Harness, n: usize, what: []const u8) ![]const u8 {
        return std.fmt.bufPrint(&self.buf, "n{d}_{s}", .{ n, what });
    }

    /// The fixture array `n{n}_{what}` uploaded.
    fn upload(self: *Harness, n: usize, what: []const u8) !cuda.DeviceBuffer {
        const bytes = try self.fx.bytes(try self.name(n, what));
        defer self.gpu.gpa.free(bytes);
        return cuda.DeviceBuffer.fromHost(self.gpu.d, bytes);
    }

    fn uploadNamed(self: *Harness, what: []const u8) !cuda.DeviceBuffer {
        const bytes = try self.fx.bytes(what);
        defer self.gpu.gpa.free(bytes);
        return cuda.DeviceBuffer.fromHost(self.gpu.d, bytes);
    }

    /// A zeroed device buffer the size of the fixture array `n{n}_{what}`.
    fn like(self: *Harness, n: usize, what: []const u8) !cuda.DeviceBuffer {
        const bytes = try self.fx.bytes(try self.name(n, what));
        defer self.gpu.gpa.free(bytes);
        var b = try cuda.DeviceBuffer.alloc(self.gpu.d, @max(16, bytes.len));
        errdefer b.free();
        try b.fill8(0, null);
        return b;
    }

    /// `got` against the fixture array `n{n}_{what}`: a PASS line, or a FAIL counted.
    fn stage(self: *Harness, n: usize, what: []const u8, got: cuda.DeviceBuffer) !void {
        try self.stream.synchronize();
        const want = try self.fx.bytes(try self.name(n, what));
        defer self.gpu.gpa.free(want);
        const bytes = try check.download(self.gpu, got);
        defer self.gpu.gpa.free(bytes);
        var label: [64]u8 = undefined;
        const text = try std.fmt.bufPrint(&label, "dsv41 n={d} {s}", .{ n, what });
        check.sameBytes(text, bytes[0..want.len], want) catch {
            self.failures += 1;
            return;
        };
        check.pass("BITEXACT {s}: {d} bytes equal to the Python fork's", .{ text, want.len });
    }
};

fn openSet(gpu: Gpu, dir: []const u8) !cuda.aot.Set {
    const aot_dir = try std.fs.path.join(gpu.gpa, &.{ dir, "aot" });
    defer gpu.gpa.free(aot_dir);
    return cuda.aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
}

fn openFixture(gpu: Gpu, dir: []const u8) !struct { fx: fixture.Fixture, path: []const u8 } {
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, "out", "fixture" });
    errdefer gpu.gpa.free(path);
    return .{ .fx = try fixture.Fixture.open(gpu.gpa, gpu.io, path), .path = path };
}

fn rowCounts(fx: fixture.Fixture, out: []usize) ![]usize {
    const count: usize = @intCast(try fx.int("n_count"));
    var nb: [8]u8 = undefined;
    for (0..count) |i| out[i] = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "n{d}", .{i})));
    return out[0..count];
}

/// `dsv41-hc <dir>`: embed_init -> hc_pre -> hc_post -> hc_pre -> route -> hc_post -> tap -> collapse_norm -> head
/// -> engram_gate, each stage's bytes the fork's.
pub fn hc(gpu: Gpu, dir: []const u8) !void {
    var set = try openSet(gpu, dir);
    defer set.deinit();
    const opened = try openFixture(gpu, dir);
    defer gpu.gpa.free(opened.path);
    var h: Harness = .{ .gpu = gpu, .fx = opened.fx, .stream = try cuda.Stream.init(gpu.d, true) };
    defer h.fx.deinit();
    defer h.stream.deinit();
    const fx = h.fx;
    const d: u32 = @intCast(try fx.int("D"));
    const world: u32 = @intCast(try fx.int("world"));
    const engine = dsv41.hc.Hc.init(&set, d, @intCast(try fx.int("iters")), @floatCast(try fx.float("eps")), @floatCast(try fx.float("hc_eps")));
    const router = dsv41.route.Route.init(&set, @intCast(try fx.int("K")), @intCast(try fx.int("N")), @intCast(try fx.int("topk")), @floatCast(try fx.float("scale")), @intCast(try fx.int("shared_id")));

    const weight_names = [_][]const u8{ "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "attn_norm", "ffn_norm", "norm_w", "gate_w", "gate_b", "engram_qk" };
    var weights: [weight_names.len]cuda.DeviceBuffer = undefined;
    var loaded: usize = 0;
    defer for (weights[0..loaded]) |*w| w.free();
    for (weight_names, &weights) |wn, *w| {
        w.* = try h.uploadNamed(wn);
        loaded += 1;
    }
    const fn_a, const sc_a, const ba_a, const fn_f, const sc_f, const ba_f, const attn_norm, const ffn_norm, const norm_w, const gate_w, const gate_b, const qk = weights;

    var kernels = try dsv41.exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    const head_tr = try fx.bytes("head_trellis");
    defer gpu.gpa.free(head_tr);
    const head_suh = try fx.bytes("head_suh");
    defer gpu.gpa.free(head_suh);
    const head_svh = try fx.bytes("head_svh");
    defer gpu.gpa.free(head_svh);
    const cb: dsv41.exl3.Codebook = if (std.mem.eql(u8, try fx.string("head_cb"), "mul1")) .mul1 else if (std.mem.eql(u8, try fx.string("head_cb"), "mcg")) .mcg else .inst3;
    var head = try dsv41.exl3.Linear.init(gpu.gpa, &kernels, head_tr, @intCast(try fx.int("head_kt")), @intCast(try fx.int("head_nt")), @intCast(try fx.int("head_w16")), head_suh, head_svh, cb);
    defer head.deinit();

    var ns_buf: [8]usize = undefined;
    const ns = try rowCounts(fx, &ns_buf);
    for (ns) |n| {
        const rows: u32 = @intCast(n);
        var ids = try h.upload(n, "ids");
        defer ids.free();
        var emb = try h.upload(n, "emb");
        defer emb.free();
        var h0 = try h.like(n, "h0");
        defer h0.free();
        var pre0 = try h.like(n, "pre0");
        defer pre0.free();
        try engine.embedInit(h.stream, emb.ptr, ids.ptr, h0.ptr, pre0.ptr, rows);
        try h.stage(n, "h0", h0);
        try h.stage(n, "pre0", pre0);

        var part = try cuda.DeviceBuffer.alloc(gpu.d, dsv41.hc.Hc.partWords(n) * 4);
        defer part.free();
        var x1 = try h.like(n, "x1");
        defer x1.free();
        var pre_a = try h.like(n, "pre_a");
        defer pre_a.free();
        var post_a = try h.like(n, "post_a");
        defer post_a.free();
        var comb_a = try h.like(n, "comb_a");
        defer comb_a.free();
        try engine.pre(h.stream, h0.ptr, fn_a.ptr, sc_a.ptr, ba_a.ptr, pre0.ptr, attn_norm.ptr, x1.ptr, pre_a.ptr, post_a.ptr, comb_a.ptr, part.ptr, rows);
        try h.stage(n, "x1", x1);
        try h.stage(n, "pre_a", pre_a);
        try h.stage(n, "post_a", post_a);
        try h.stage(n, "comb_a", comb_a);

        var g_a = try h.upload(n, "g_a");
        defer g_a.free();
        var h1 = try h.like(n, "h1");
        defer h1.free();
        try engine.post(h.stream, g_a.ptr, world, h0.ptr, post_a.ptr, comb_a.ptr, h1.ptr, rows);
        try h.stage(n, "h1", h1);

        var x2 = try h.like(n, "x2");
        defer x2.free();
        var pre_f = try h.like(n, "pre_f");
        defer pre_f.free();
        var post_f = try h.like(n, "post_f");
        defer post_f.free();
        var comb_f = try h.like(n, "comb_f");
        defer comb_f.free();
        try engine.pre(h.stream, h1.ptr, fn_f.ptr, sc_f.ptr, ba_f.ptr, pre_a.ptr, ffn_norm.ptr, x2.ptr, pre_f.ptr, post_f.ptr, comb_f.ptr, part.ptr, rows);
        try h.stage(n, "x2", x2);
        try h.stage(n, "pre_f", pre_f);
        try h.stage(n, "post_f", post_f);
        try h.stage(n, "comb_f", comb_f);

        var parts = try h.like(n, "S");
        defer parts.free();
        var pick = try h.like(n, "pick");
        defer pick.free();
        var wts = try h.like(n, "wts");
        defer wts.free();
        try router.gate(h.stream, x2.ptr, gate_w.ptr, parts.ptr, rows);
        try h.stage(n, "S", parts);
        try router.pick(h.stream, parts.ptr, gate_b.ptr, pick.ptr, wts.ptr, rows);
        try h.stage(n, "pick", pick);
        try h.stage(n, "wts", wts);

        var g_f = try h.upload(n, "g_f");
        defer g_f.free();
        var h2 = try h.like(n, "h2");
        defer h2.free();
        try engine.post(h.stream, g_f.ptr, world, h1.ptr, post_f.ptr, comb_f.ptr, h2.ptr, rows);
        try h.stage(n, "h2", h2);

        var taps = try h.like(n, "taps");
        defer taps.free();
        try engine.tap(h.stream, h2.ptr, taps.ptr, @intCast(3 * d), rows);
        try h.stage(n, "taps", taps);

        var xc = try h.like(n, "xc");
        defer xc.free();
        try engine.collapseNorm(h.stream, h2.ptr, pre_f.ptr, norm_w.ptr, xc.ptr, rows);
        try h.stage(n, "xc", xc);

        var xh = try cuda.DeviceBuffer.alloc(gpu.d, n * head.k * 2);
        defer xh.free();
        var z = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, head.scratchWords(n) * 4));
        defer z.free();
        var logits = try h.like(n, "logits");
        defer logits.free();
        try head.run(xc.ptr, .bf16, n, xh.ptr, z.ptr, logits.ptr, .f32, h.stream);
        try h.stage(n, "logits", logits);

        var kv = try h.upload(n, "kv");
        defer kv.free();
        var h_e = try h.like(n, "h_e");
        defer h_e.free();
        try engine.engramGate(h.stream, h0.ptr, kv.ptr, qk.ptr, h_e.ptr, rows);
        try h.stage(n, "h_e", h_e);
    }
    try check.expect(h.failures == 0, "dsv41-hc: {d} stage(s) differ from the Python fork", .{h.failures});
}

/// `dsv41-route <dir>`: the gate's chunk sums, picks and weights alone, from the fork's own hc_pre output.
pub fn route(gpu: Gpu, dir: []const u8) !void {
    var set = try openSet(gpu, dir);
    defer set.deinit();
    const opened = try openFixture(gpu, dir);
    defer gpu.gpa.free(opened.path);
    var h: Harness = .{ .gpu = gpu, .fx = opened.fx, .stream = try cuda.Stream.init(gpu.d, true) };
    defer h.fx.deinit();
    defer h.stream.deinit();
    const fx = h.fx;
    const r = dsv41.route.Route.init(&set, @intCast(try fx.int("K")), @intCast(try fx.int("N")), @intCast(try fx.int("topk")), @floatCast(try fx.float("scale")), @intCast(try fx.int("shared_id")));
    var gate_w = try h.uploadNamed("gate_w");
    defer gate_w.free();
    var gate_b = try h.uploadNamed("gate_b");
    defer gate_b.free();
    var ns_buf: [8]usize = undefined;
    const ns = try rowCounts(fx, &ns_buf);
    for (ns) |n| {
        const rows: u32 = @intCast(n);
        var x2 = try h.upload(n, "x2");
        defer x2.free();
        var parts = try h.like(n, "S");
        defer parts.free();
        var pick = try h.like(n, "pick");
        defer pick.free();
        var wts = try h.like(n, "wts");
        defer wts.free();
        try r.gate(h.stream, x2.ptr, gate_w.ptr, parts.ptr, rows);
        try h.stage(n, "S", parts);
        try r.pick(h.stream, parts.ptr, gate_b.ptr, pick.ptr, wts.ptr, rows);
        try h.stage(n, "pick", pick);
        try h.stage(n, "wts", wts);
    }
    try check.expect(h.failures == 0, "dsv41-route: {d} stage(s) differ from the Python fork", .{h.failures});
}
