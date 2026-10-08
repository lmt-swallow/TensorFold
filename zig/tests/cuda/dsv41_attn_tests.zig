//! `dsv41-attn <dir>`: the DeepSeek-V4.1 attention sublayer (cuda_attention.zig) on tools/zig/dsv41/attn_capture.sh's
//! work dir: the layer's weights and cache planes from the fork's fixture, the fork's Triton kernels from the
//! captured AOT set, the fp32 partial, the picks and every cache plane after the call byte-equal to the fork's.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;
const Fixture = fixture.Fixture;

fn upload(gpu: Gpu, fx: Fixture, name: []const u8) !cuda.DeviceBuffer {
    const bytes = try fx.bytes(name);
    defer gpu.gpa.free(bytes);
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}

fn linear(gpu: Gpu, kernels: *const dsv41.exl3.Kernels, fx: Fixture, name: []const u8) !dsv41.exl3.Linear {
    var nb: [96]u8 = undefined;
    const kt: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "{s}.kt", .{name})));
    const nt: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "{s}.nt", .{name})));
    const w16: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "{s}.w16", .{name})));
    const cb_name = try fx.string(try std.fmt.bufPrint(&nb, "{s}.cb", .{name}));
    const cb: dsv41.exl3.Codebook = if (std.mem.eql(u8, cb_name, "mul1")) .mul1 else if (std.mem.eql(u8, cb_name, "mcg")) .mcg else .inst3;
    const trellis = try fx.bytes(try std.fmt.bufPrint(&nb, "{s}.trellis", .{name}));
    defer gpu.gpa.free(trellis);
    const suh = try fx.bytes(try std.fmt.bufPrint(&nb, "{s}.suh", .{name}));
    defer gpu.gpa.free(suh);
    const svh = try fx.bytes(try std.fmt.bufPrint(&nb, "{s}.svh", .{name}));
    defer gpu.gpa.free(svh);
    return dsv41.exl3.Linear.init(gpu.gpa, kernels, trellis, kt, nt, w16, suh, svh, cb);
}

/// `limit`: compare only the first `limit` bytes (0: all). The compressed planes' scratch row (the last one) is
/// left out: several rows of a window with incomplete groups write it in one launch, in the fork too (torch's
/// index_put with repeated indices), and no row reads it (vis never reaches it).
fn expectSame(gpu: Gpu, fx: Fixture, got: cuda.DeviceBuffer, name: []const u8, label: []const u8, limit: usize) !void {
    var nb: [96]u8 = undefined;
    const want = try fx.bytes(try std.fmt.bufPrint(&nb, "expected_{s}", .{name}));
    defer gpu.gpa.free(want);
    const bytes = try check.download(gpu, got);
    defer gpu.gpa.free(bytes);
    const n = if (limit == 0) want.len else @min(limit, want.len);
    var what: [160]u8 = undefined;
    try check.sameBytes(try std.fmt.bufPrint(&what, "{s} {s}", .{ label, name }), bytes[0..n], want[0..n]);
}

pub fn attention(gpu: Gpu, dir: []const u8) !void {
    const gpa = gpu.gpa;
    const A = dsv41.attention;
    const aot_dir = try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const glue_path = try std.fs.path.join(gpa, &.{ dir, "out", "glue.cubin" });
    defer gpa.free(glue_path);
    const cubin = try std.Io.Dir.cwd().readFileAlloc(gpu.io, glue_path, gpa, .limited(1 << 26));
    defer gpa.free(cubin);
    var glue = try A.Glue.load(gpu.d, cubin);
    defer glue.deinit();
    var kernels = try dsv41.exl3.Kernels.load(gpu.d);
    defer kernels.deinit();

    const wdir = try std.fs.path.join(gpa, &.{ dir, "out", "fixtures", "attn_weights" });
    defer gpa.free(wdir);
    var wf = try Fixture.open(gpa, gpu.io, wdir);
    defer wf.deinit();
    const ratio: usize = @intCast(try wf.int("ratio"));
    const kv_source = (try wf.int("kv_source")) != 0;
    const index_source = (try wf.int("index_source")) != 0;
    const has_idx_wk = (try wf.int("has_idx_wk")) != 0;
    var lay: A.Layer = .{
        .wq_a = try linear(gpu, &kernels, wf, "wq_a"),
        .wq_b = try linear(gpu, &kernels, wf, "wq_b"),
        .wkv = try linear(gpu, &kernels, wf, "wkv"),
        .wo_a = .{ try linear(gpu, &kernels, wf, "wo_a0"), try linear(gpu, &kernels, wf, "wo_a1"), try linear(gpu, &kernels, wf, "wo_a2"), try linear(gpu, &kernels, wf, "wo_a3") },
        .wo_b = try linear(gpu, &kernels, wf, "wo_b"),
        .q_norm = try upload(gpu, wf, "q_norm"),
        .kv_norm = try upload(gpu, wf, "kv_norm"),
        .sink = try upload(gpu, wf, "sink"),
    };
    defer lay.deinit();
    if (kv_source) {
        lay.comp_wkv = try linear(gpu, &kernels, wf, "comp_wkv");
        lay.comp_norm = try upload(gpu, wf, "comp_norm");
        if (ratio > 1) lay.comp_wgate = try linear(gpu, &kernels, wf, "comp_wgate");
    }
    if (index_source) {
        lay.idx_wq_b = try linear(gpu, &kernels, wf, "idx_wq_b");
        lay.idx_proj_h = try upload(gpu, wf, "idx_proj_h");
        if (has_idx_wk) {
            lay.idx_wk = try linear(gpu, &kernels, wf, "idx_wk");
            lay.idx_k_norm = try upload(gpu, wf, "idx_k_norm");
        }
    }
    var rope: A.Rope = .{ .cos = try upload(gpu, wf, "cos"), .sin = try upload(gpu, wf, "sin"), .rows = @intCast(try wf.int("rope_rows")) };
    defer rope.cos.free();
    defer rope.sin.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    const cases = try gpa.dupe(u8, try wf.string("cases"));
    defer gpa.free(cases);
    var attn: ?A.Attention = null;
    defer if (attn) |*at| at.deinit();
    var it = std.mem.splitScalar(u8, cases, ',');
    var failures: usize = 0;
    while (it.next()) |text| {
        const n = try std.fmt.parseInt(usize, text, 10);
        var nb: [96]u8 = undefined;
        const cdir = try std.fs.path.join(gpa, &.{ dir, "out", "fixtures", try std.fmt.bufPrint(&nb, "attn_n{d}", .{n}) });
        defer gpa.free(cdir);
        var fx = try Fixture.open(gpa, gpu.io, cdir);
        defer fx.deinit();
        const start: usize = @intCast(try fx.int("start"));
        const bucket: usize = @intCast(try fx.int("bucket"));
        if (attn == null) {
            const dims: A.Dims = .{
                .heads = @intCast(try wf.int("heads_local")),
                .hd = @intCast(try wf.int("head_dim")),
                .rd = @intCast(try wf.int("rope_dim")),
                .q_rank = @intCast(try wf.int("q_rank")),
                .o_rank = @intCast(try wf.int("o_rank")),
                .groups = @intCast(try wf.int("groups_local")),
                .d = @intCast(try wf.int("dim")),
                .window = @intCast(try wf.int("window")),
                .ring = @intCast(try wf.int("ring_size")),
                .raw = @intCast(try wf.int("raw")),
                .idx_heads = @intCast(try wf.int("idx_heads")),
                .idx_dim = @intCast(try wf.int("idx_dim")),
                .idx_topk = @intCast(try wf.int("idx_topk")),
                .ratio = ratio,
                .kv_source = kv_source,
                .index_source = index_source,
                .has_idx_wk = has_idx_wk,
                .eps = @floatCast(try wf.float("eps")),
                .bucket = bucket,
            };
            attn = try A.Attention.init(gpa, gpu.d, dims, &set, &glue, &lay);
        } else if (attn.?.dims.bucket != bucket) return error.BucketMismatch;
        var caches: A.Caches = .{
            .ring = try upload(gpu, fx, "ring"),
            .comp_codes = try upload(gpu, fx, "comp_codes"),
            .comp_scales = try upload(gpu, fx, "comp_scales"),
            .idxk_codes = try upload(gpu, fx, "idxk_codes"),
            .idxk_scales = try upload(gpu, fx, "idxk_scales"),
            .raw_kv = try upload(gpu, fx, "raw_kv"),
            .raw_score = try upload(gpu, fx, "raw_score"),
            .rows = @intCast(try fx.int("comp_rows")),
        };
        defer caches.deinit();
        var x = try upload(gpu, fx, "x");
        defer x.free();
        const out = try attn.?.forward(&lay, &caches, rope, x.ptr, n, start, stream);
        try stream.synchronize();
        var label_buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "dsv41-attn n={d}", .{n});
        const out_view: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = out, .len = n * attn.?.dims.d * 4 };
        const cidx_view: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = attn.?.cidx.ptr, .len = n * attn.?.dims.kk() * 8 };
        var ok = true;
        expectSame(gpu, fx, out_view, "out", label, 0) catch {
            ok = false;
        };
        if (index_source) expectSame(gpu, fx, cidx_view, "cidx", label, 0) catch {
            ok = false;
        };
        const dm = attn.?.dims;
        const kept = caches.rows - 1; // every row but the scratch row
        const Plane = struct { name: []const u8, limit: usize };
        const planes = [_]Plane{
            .{ .name = "ring", .limit = 0 },                      .{ .name = "comp_codes", .limit = kept * dm.hd / 2 },
            .{ .name = "comp_scales", .limit = kept * dm.hd / 16 }, .{ .name = "idxk_codes", .limit = kept * dm.idx_dim / 2 },
            .{ .name = "idxk_scales", .limit = kept * dm.idx_dim / 32 }, .{ .name = "raw_kv", .limit = 0 },
            .{ .name = "raw_score", .limit = 0 },
        };
        inline for (.{ "ring", "comp_codes", "comp_scales", "idxk_codes", "idxk_scales", "raw_kv", "raw_score" }, 0..) |plane, i| {
            expectSame(gpu, fx, @field(caches, plane), plane, label, planes[i].limit) catch {
                ok = false;
            };
        }
        if (ok) {
            check.pass("BITEXACT {s}: the fp32 partial, {d} picks a row and every cache plane (but the scratch row) equal to the Python fork's (layer {d}, positions {d}..)", .{ label, attn.?.dims.kk(), try wf.int("layer"), start });
        } else failures += 1;
    }
    try check.expect(failures == 0, "{d} attention case(s) differ from the fork", .{failures});
}
