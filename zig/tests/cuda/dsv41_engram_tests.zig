//! DeepSeek-V4.1's Engram device side against the fork's fixture (tools/zig/dsv41/dump_engram.py, packed by
//! capture_engram.sh): the hashes from engram_hash.zig, the rows read from the real FP8 shards (`engram_dir`), their
//! dequantized bf16 `e` a rank, each rank's `engram_wkv` partial, the rank-ordered sum and the gate, every stage's
//! bytes the Python fork's.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;

const rows_mod = dsv41.engram_rows;

const Harness = struct {
    gpu: Gpu,
    fx: fixture.Fixture,
    stream: cuda.Stream,
    failures: usize = 0,

    fn upload(self: *Harness, what: []const u8) !cuda.DeviceBuffer {
        const bytes = try self.fx.bytes(what);
        defer self.gpu.gpa.free(bytes);
        return cuda.DeviceBuffer.fromHost(self.gpu.d, bytes);
    }

    fn like(self: *Harness, what: []const u8) !cuda.DeviceBuffer {
        const bytes = try self.fx.bytes(what);
        defer self.gpu.gpa.free(bytes);
        var b = try cuda.DeviceBuffer.alloc(self.gpu.d, @max(16, bytes.len));
        errdefer b.free();
        try b.fill8(0, null);
        return b;
    }

    fn stage(self: *Harness, what: []const u8, got: cuda.DeviceBuffer) !void {
        try self.stream.synchronize();
        const want = try self.fx.bytes(what);
        defer self.gpu.gpa.free(want);
        const bytes = try check.download(self.gpu, got);
        defer self.gpu.gpa.free(bytes);
        var label: [64]u8 = undefined;
        const text = try std.fmt.bufPrint(&label, "dsv41 engram {s}", .{what});
        check.sameBytes(text, bytes[0..want.len], want) catch {
            self.failures += 1;
            return;
        };
        check.pass("BITEXACT {s}: {d} bytes equal to the Python fork's", .{ text, want.len });
    }

    fn stageHost(self: *Harness, what: []const u8, got: []const u8) !void {
        const want = try self.fx.bytes(what);
        defer self.gpu.gpa.free(want);
        var label: [64]u8 = undefined;
        const text = try std.fmt.bufPrint(&label, "dsv41 engram {s}", .{what});
        check.sameBytes(text, got[0..want.len], want) catch {
            self.failures += 1;
            return;
        };
        check.pass("BITEXACT {s}: {d} bytes equal to the Python fork's", .{ text, want.len });
    }
};

fn nowNs() i128 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// `dsv41-engram <dir> <engram-dir>`.
pub fn engram(gpu: Gpu, dir: []const u8, engram_dir: []const u8) !void {
    const gpa = gpu.gpa;
    const aot_dir = try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_path = try std.fs.path.join(gpa, &.{ dir, "out", "fixture" });
    defer gpa.free(fx_path);
    var h: Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(gpa, gpu.io, fx_path), .stream = try cuda.Stream.init(gpu.d, true) };
    defer h.fx.deinit();
    defer h.stream.deinit();
    const fx = h.fx;
    const n: usize = @intCast(try fx.int("n"));
    const world: usize = @intCast(try fx.int("world"));
    const cols: usize = @intCast(try fx.int("cols"));
    const d: u32 = @intCast(try fx.int("D"));
    const layer = try fx.int("layer");
    const li: usize = @intCast(try fx.int("layer_index"));
    const n_layers: usize = @intCast(try fx.int("n_engram_layers"));

    // 1. hashes: engram_hash.zig from the fixture's token map
    const map_bytes = try fx.bytes("token_map");
    defer gpa.free(map_bytes);
    const token_map = std.mem.bytesAsSlice(i32, map_bytes);
    var layers_buf: [4]i64 = undefined;
    var rows_buf: [4]i64 = undefined;
    var nb: [24]u8 = undefined;
    for (0..n_layers) |i| {
        layers_buf[i] = try fx.int(try std.fmt.bufPrint(&nb, "engram_layer{d}", .{i}));
        rows_buf[i] = try fx.int(try std.fmt.bufPrint(&nb, "engram_rows{d}", .{i}));
    }
    const hasher = try dsv41.engram_hash.Hasher.init(.{
        .layers = layers_buf[0..n_layers],
        .rows = rows_buf[0..n_layers],
        .ngram = @intCast(try fx.int("ngram")),
        .heads = @intCast(try fx.int("heads")),
        .vocab = @intCast(try fx.int("vocab")),
        .cvocab = @intCast(try fx.int("cvocab")),
        .pad_token = @intCast(try fx.int("pad_token")),
    }, @alignCast(token_map), null);
    const tok_bytes = try fx.bytes("tokens");
    defer gpa.free(tok_bytes);
    const tokens = std.mem.bytesAsSlice(i32, tok_bytes);
    const hashes = try gpa.alloc(i64, n * n_layers * cols);
    defer gpa.free(hashes);
    try hasher.hashes(@alignCast(tokens), 0, n, hashes);
    try h.stageHost("hashes", std.mem.sliceAsBytes(hashes));

    // 2. the rows from the shards, this layer's table
    var tables = try rows_mod.Tables.open(gpa, gpu.io, engram_dir);
    defer tables.close();
    const table = tables.get(layer) orelse return error.NoTable;
    var reader = try rows_mod.Reader.init(gpa, 31);
    defer reader.deinit();
    var kernels = try rows_mod.Kernels.load(gpu.d);
    defer kernels.deinit();
    var exl3 = try dsv41.exl3.Kernels.load(gpu.d);
    defer exl3.deinit();
    const per = cols / world; // this rank's columns
    var staging = try rows_mod.Staging.init(gpu.d, n * per);
    defer staging.deinit();
    const idx = try gpa.alloc(i64, n * per);
    defer gpa.free(idx);

    var gathered = try cuda.DeviceBuffer.alloc(gpu.d, world * n * 5 * d * 4);
    defer gathered.free();
    var pbuf: [32]u8 = undefined;
    for (0..world) |rank| {
        const lo: usize = @intCast(try fx.int(try std.fmt.bufPrint(&pbuf, "r{d}_col_lo", .{rank})));
        for (0..n) |p_| for (0..per) |c| {
            idx[p_ * per + c] = hashes[(p_ * n_layers + li) * cols + lo + c];
        };
        try reader.gather(table, idx, staging.codes(n * per), staging.scales(n * per));
        try h.stageHost(try std.fmt.bufPrint(&pbuf, "r{d}_codes", .{rank}), staging.codes(n * per));
        try h.stageHost(try std.fmt.bufPrint(&pbuf, "r{d}_scales", .{rank}), staging.scales(n * per));
        try staging.upload(n * per, h.stream);
        var e = try h.like(try std.fmt.bufPrint(&pbuf, "r{d}_e", .{rank}));
        defer e.free();
        try kernels.decode(h.stream, staging.dev_w.ptr, staging.dev_s.ptr, e.ptr, n * per);
        try h.stage(try std.fmt.bufPrint(&pbuf, "r{d}_e", .{rank}), e);

        // 3. this rank's engram_wkv partial, into the gathered buffer's rank slot
        const tr = try fx.bytes(try std.fmt.bufPrint(&pbuf, "r{d}_wkv_trellis", .{rank}));
        defer gpa.free(tr);
        const suh = try fx.bytes(try std.fmt.bufPrint(&pbuf, "r{d}_wkv_suh", .{rank}));
        defer gpa.free(suh);
        const svh = try fx.bytes(try std.fmt.bufPrint(&pbuf, "r{d}_wkv_svh", .{rank}));
        defer gpa.free(svh);
        const cb_name = try fx.string(try std.fmt.bufPrint(&pbuf, "r{d}_wkv_cb", .{rank}));
        const cb: dsv41.exl3.Codebook = if (std.mem.eql(u8, cb_name, "mul1")) .mul1 else if (std.mem.eql(u8, cb_name, "mcg")) .mcg else .inst3;
        var wkv = try dsv41.exl3.Linear.init(gpa, &exl3, tr, @intCast(try fx.int(try std.fmt.bufPrint(&pbuf, "r{d}_wkv_kt", .{rank}))), @intCast(try fx.int(try std.fmt.bufPrint(&pbuf, "r{d}_wkv_nt", .{rank}))), @intCast(try fx.int(try std.fmt.bufPrint(&pbuf, "r{d}_wkv_w16", .{rank}))), suh, svh, cb);
        defer wkv.deinit();
        var xh = try cuda.DeviceBuffer.alloc(gpu.d, n * wkv.k * 2);
        defer xh.free();
        var z = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, wkv.scratchWords(n) * 4));
        defer z.free();
        const slot = try gathered.at(rank * n * 5 * d * 4);
        try wkv.run(e.ptr, .bf16, n, xh.ptr, z.ptr, slot, .f32, h.stream);
        try h.stream.synchronize();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, n * 5 * d * 4);
        defer part.free();
        try part.copyFrom(0, slot, n * 5 * d * 4, h.stream.handle);
        try h.stage(try std.fmt.bufPrint(&pbuf, "r{d}_part", .{rank}), part);
    }
    try h.stage("gathered", gathered);

    // 4. the ranks' sum to bf16, then the gate
    var kv = try h.like("kv");
    defer kv.free();
    try kernels.sumToBf16(h.stream, gathered.ptr, @intCast(world), n * 5 * d, kv.ptr);
    try h.stage("kv", kv);
    var hbuf = try h.upload("h");
    defer hbuf.free();
    var qk = try h.upload("qk");
    defer qk.free();
    var h_e = try h.like("h_e");
    defer h_e.free();
    const hc_engine = dsv41.hc.Hc.init(&set, d, 20, @floatCast(try fx.float("eps")), 1e-6);
    try hc_engine.engramGate(h.stream, hbuf.ptr, kv.ptr, qk.ptr, h_e.ptr, @intCast(n));
    try h.stage("h_e", h_e);

    // 5. the read path's latency: every column of n positions (both ranks' share), the best of a few rounds
    const all = try gpa.alloc(i64, n * cols);
    defer gpa.free(all);
    for (0..n) |p_| for (0..cols) |c| {
        all[p_ * cols + c] = hashes[(p_ * n_layers + li) * cols + c];
    };
    const ow = try gpa.alloc(u8, n * cols * rows_mod.row_w);
    defer gpa.free(ow);
    const os_ = try gpa.alloc(u8, n * cols * rows_mod.row_s);
    defer gpa.free(os_);
    var best: i128 = std.math.maxInt(i128);
    var first: i128 = 0;
    for (0..5) |i| {
        const t0 = nowNs();
        try reader.gather(table, all, ow, os_);
        const dt = nowNs() - t0;
        if (i == 0) first = dt;
        best = @min(best, dt);
    }
    check.pass("dsv41 engram read: {d} positions x {d} columns = {d} rows x 2 preads; cold {d:.2} ms, best {d:.2} ms", .{ n, cols, n * cols, @as(f64, @floatFromInt(first)) / 1e6, @as(f64, @floatFromInt(best)) / 1e6 });
    try check.expect(h.failures == 0, "dsv41-engram: {d} stage(s) differ from the Python fork", .{h.failures});
}
