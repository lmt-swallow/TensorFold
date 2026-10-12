//! DFlash2 projection and connected proposer fixtures over real rank weights.
//! Recorded peer partials isolate TP arithmetic; real communication and target verification are separate gates.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const hs = @import("glm53_harness.zig");
const Head = glm.mtp.DraftHead;

fn projection(m: *const glm.dflash_weights.Model, head: *const Head, call: usize) !*const Head {
    if (call == 0) return &m.fc;
    if (call <= 5) return &m.layers[call - 1].kv;
    if (call == 36) return head;
    if (call > 36) return error.BadDflashTrace;
    const layer = &m.layers[(call - 6) / 6];
    return switch ((call - 6) % 6) {
        0 => &layer.a_kp,
        1 => &layer.qkv,
        2 => &layer.o,
        3 => &layer.m_kp,
        4 => &layer.gu,
        5 => &layer.down,
        else => unreachable,
    };
}

pub fn run(gpu: check.Gpu, work: []const u8, target: []const u8, draft: []const u8) !void {
    return replay(gpu, work, target, draft, false, false);
}

pub fn connected(gpu: check.Gpu, work: []const u8, target: []const u8, draft: []const u8) !void {
    return replay(gpu, work, target, draft, true, false);
}

/// Replay committed synthetic taps across two physical ring wraps and reset.
pub fn wrap(gpu: check.Gpu, work: []const u8, target: []const u8, draft: []const u8) !void {
    return replay(gpu, work, target, draft, true, true);
}

fn replay(gpu: check.Gpu, work: []const u8, target: []const u8, draft: []const u8, pipeline: bool, wrapping: bool) !void {
    const a = gpu.gpa;
    const aot_dir = try std.fs.path.join(a, &.{ work, "aot" });
    defer a.free(aot_dir);
    var set = try cuda.aot.Set.load(a, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_dir = try std.fs.path.join(a, &.{ work, "out", "fixture" });
    defer a.free(fx_dir);
    var h: hs.Harness = .{ .gpu = gpu, .fx = try fixture.Fixture.open(a, gpu.io, fx_dir), .stream = try cuda.Stream.init(gpu.d, false), .isolated = false };
    defer h.fx.deinit();
    defer h.stream.deinit();
    defer h.deinit();
    var hc = try glm.glue.Hc.load(gpu.d);
    defer hc.deinit();
    const glue: glm.glue.Glue = .{ .set = &set, .hc = &hc, .hidden = 4096, .streams = 1 };
    var kernels = try glm.dflash_projections.Kernels.load(gpu.d);
    defer kernels.deinit();
    var why: glm.config.Why = .{};
    const cfg = try glm.config.read(a, gpu.io, target, &why);
    var pack = try glm.weights.Pack.open(a, gpu.io, target);
    defer pack.close();
    var scratch = try glm.dflash_buffers.Scratch.init(gpu.d);
    defer scratch.deinit();
    const dk: glm.dflash_kernels.Kernels = .{ .set = &set };
    var lt = try cuda.cublaslt.Library.open();
    defer lt.close();
    var hp = try cuda.cublaslt.Linear.init(&lt, 7, 256, 4096, .bf16, 32 << 20);
    defer hp.deinit();
    const trace = h.fx.parsed.value.object.get("trace").?.array.items;
    for (0..2) |rank| {
        var model = try glm.dflash_weights.Model.open(a, gpu.io, gpu.d, draft, rank);
        defer model.deinit();
        const lay: glm.config.Layout = .{ .rank = rank, .world = 2, .cfg = &cfg };
        var head = try Head.load(.{ .gpa = a, .d = gpu.d, .pack = &pack, .lay = lay });
        defer head.deinit();
        var context = try glm.dflash_buffers.Context.init(gpu.d, if (wrapping) 8192 else 4096);
        defer context.deinit();
        var embedding = try (glm.weights.Loader{ .gpa = a, .d = gpu.d, .pack = &pack, .lay = lay }).share("model.language_model.embed_tokens.weight");
        defer embedding.free();
        var update: usize = 0;
        var tap_start: usize = 0;
        for (0..@as(usize, if (wrapping) 6 else 9)) |step| {
            try h.begin(rank, step);
            var sb: [16]u8 = undefined;
            const name = try std.fmt.bufPrint(&sb, "s{d}", .{step});
            if (pipeline) {
                if (wrapping) try wrapUpdates(&h, &context, &model, scratch, dk, kernels, glue, step, &update, &tap_start);
                try connectedStep(&h, &context, &model, &head, scratch, dk, kernels, glue, embedding, &hp, step, !wrapping);
                continue;
            }
            var count: usize = 0;
            for (trace) |call| {
                const o = call.object;
                if (o.get("rank").?.integer != rank or !std.mem.eql(u8, o.get("step").?.string, name) or !std.mem.endsWith(u8, o.get("fn").?.string, "._mm")) continue;
                const seq = o.get("seq").?.integer;
                const weight = try projection(&model, &head, count);
                const shape = o.get("args").?.object.get("x").?.object.get("shape").?.array.items;
                const rows: usize = @intCast(shape[0].integer);
                try check.expect(shape[1].integer == weight.k, "DFlash2 projection input width", .{});
                const input = try h.fx.bytes(try h.arrayName(seq, "_mm", "x_in"));
                defer a.free(input);
                var x = try cuda.DeviceBuffer.fromHost(gpu.d, input);
                defer x.free();
                const fp32 = std.mem.eql(u8, o.get("ret").?.object.get("dtype").?.string, "torch.float32");
                var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * weight.n * @as(usize, if (fp32) 4 else 2));
                defer out.free();
                const sums = scratch.get("sums");
                try glue.groupSums(x.ptr, weight.k, sums.ptr, rows, weight.k, h.stream);
                try kernels.matmul(weight, x.ptr, sums.ptr, out.ptr, rows, fp32, scratch.get("matrix_partials").ptr, h.stream);
                try h.same(try h.arrayName(seq, "_mm", "ret"), out.ptr);
                count += 1;
            }
            try check.expect(count == 37, "DFlash2 rank {d} step {d}: {d} projections, expected 37", .{ rank, step, count });
        }
    }
    try check.expect(h.failures == 0, "DFlash2 projection replay: {d} failures", .{h.failures});
    if (wrapping) check.pass("DFlash2 two ring wraps and reset equal the captured fixture", .{}) else if (pipeline) check.pass("DFlash2 connected nine-step rank outputs and rings equal the captured fixture", .{}) else check.pass("DFlash2 all 666 rank/step projections equal the captured BF16/FP32 outputs", .{});
}

fn wrapUpdates(h: *hs.Harness, context: *glm.dflash_buffers.Context, model: *const glm.dflash_weights.Model, scratch: glm.dflash_buffers.Scratch, dk: glm.dflash_kernels.Kernels, qk: glm.dflash_projections.Kernels, glue: glm.glue.Glue, step: usize, update: *usize, tap_start: *usize) !void {
    const params = h.fx.parsed.value.object.get("params").?.object;
    const target: usize = @intCast(params.get("checkpoints").?.array.items[step].integer);
    const resetting = step == 5;
    if (resetting) {
        try context.reset(h.stream);
        tap_start.* = 0;
    }
    const raw = try h.fx.bytes("committed_taps");
    defer h.gpu.gpa.free(raw);
    while (context.logical.end < target) {
        const rows: usize = if (resetting) 1 else @intCast(params.get("tap_rows").?.array.items[update.*].integer);
        if (rows == 0 or rows > 64 or context.logical.end + rows > target) return error.BadDflashTrace;
        const bytes_per_row = 5 * 4096 * 2;
        const offset = tap_start.* * bytes_per_row;
        if (offset + rows * bytes_per_row > raw.len) return error.BadDflashTrace;
        try scratch.get("taps").upload(0, raw[offset..][0 .. rows * bytes_per_row]);
        try qk.rotary(scratch.get("cos").ptr, scratch.get("sin").ptr, context.logical.end, rows, model.theta, h.stream);
        var name: [96]u8 = undefined;
        inline for ([_][]const u8{ "cos", "sin" }) |part| {
            const key = if (resetting)
                try std.fmt.bufPrint(&name, "r{d}_reset_{s}", .{ h.rank, part })
            else
                try std.fmt.bufPrint(&name, "r{d}_update{d}_{s}", .{ h.rank, update.*, part });
            try h.same(key, scratch.get(part).ptr);
        }
        try glm.dflash_forward.addTaps(context, model, scratch, dk, qk, glue, scratch.get("taps"), scratch.get("cos"), scratch.get("sin"), rows, model.eps, h.stream);
        tap_start.* += rows;
        if (!resetting) update.* += 1;
    }
    try check.expect(context.logical.end == target, "DFlash wrap checkpoint committed end", .{});
}

/// Each native rank partial is checked before combining it with the recorded peer partial.
/// Both ranks execute this gate; real communication and target acceptance remain separate gates.
const Gather = struct {
    h: *hs.Harness,
    seqs: [10]i64,
    at: usize = 0,
    step: usize,

    fn run(raw: *anyopaque, part: cuda.abi.DevicePtr, gathered: cuda.abi.DevicePtr, floats: usize, stream: cuda.Stream) !void {
        const self: *Gather = @ptrCast(@alignCast(raw));
        if (self.at >= self.seqs.len or floats != 8 * 4096) return error.BadDflashTrace;
        const h = self.h;
        const seq = self.seqs[self.at];
        self.at += 1;
        try h.same(try h.arrayName(seq, "_mm", "ret"), part);
        const own: cuda.DeviceBuffer = .{ .d = h.gpu.d, .ptr = part, .len = floats * 4 };
        const out: cuda.DeviceBuffer = .{ .d = h.gpu.d, .ptr = gathered, .len = 2 * floats * 4 };
        try out.copyFrom(h.rank * floats * 4, own.ptr, floats * 4, stream.handle);
        var name: [128]u8 = undefined;
        const peer_name = try std.fmt.bufPrint(&name, "r{d}_s{d}_{d:0>3}__mm_ret", .{ 1 - h.rank, self.step, @as(u64, @intCast(seq)) });
        const peer = try h.fx.bytes(peer_name);
        defer h.gpu.gpa.free(peer);
        if (peer.len != floats * 4) return error.BadDflashTrace;
        try out.upload((1 - h.rank) * floats * 4, peer);
    }
};

fn feedCall(h: *hs.Harness, seq: i64, function: []const u8, argument: []const u8) !cuda.DeviceBuffer {
    const bytes = try h.fx.bytes(try h.arrayName(seq, function, argument));
    defer h.gpu.gpa.free(bytes);
    return cuda.DeviceBuffer.fromHost(h.gpu.d, bytes);
}

fn connectedStep(h: *hs.Harness, context: *glm.dflash_buffers.Context, model: *const glm.dflash_weights.Model, head: *const Head, scratch: glm.dflash_buffers.Scratch, dk: glm.dflash_kernels.Kernels, qk: glm.dflash_projections.Kernels, glue: glm.glue.Glue, embedding: cuda.DeviceBuffer, hp: *const cuda.cublaslt.Linear, step: usize, update_taps: bool) !void {
    const a = h.gpu.gpa;
    var gathers: Gather = .{ .h = h, .seqs = undefined, .step = step };
    var ng: usize = 0;
    var tap_seq: ?i64 = null;
    var block_seq: ?i64 = null;
    var head_seq: ?i64 = null;
    const trace = h.fx.parsed.value.object.get("trace").?.array.items;
    var sb: [16]u8 = undefined;
    const step_name = try std.fmt.bufPrint(&sb, "s{d}", .{step});
    for (trace) |call| {
        const o = call.object;
        if (o.get("rank").?.integer != h.rank or !std.mem.eql(u8, o.get("step").?.string, step_name)) continue;
        const seq = o.get("seq").?.integer;
        const fn_ = o.get("fn").?.string;
        if (std.mem.endsWith(u8, fn_, "._prep")) {
            if (o.get("args").?.object.get("heads").?.integer == 0 and tap_seq == null) tap_seq = seq;
            if (o.get("args").?.object.get("heads").?.integer == 16 and block_seq == null) block_seq = seq;
        }
        if (std.mem.endsWith(u8, fn_, "._mm")) {
            const ret = o.get("ret").?.object;
            if (std.mem.eql(u8, ret.get("dtype").?.string, "torch.float32")) {
                if (ng == gathers.seqs.len) return error.BadDflashTrace;
                gathers.seqs[ng] = seq;
                ng += 1;
            }
            if (ret.get("shape").?.array.items[1].integer == 77440) head_seq = seq;
        }
    }
    if (ng != 10 or (update_taps and tap_seq == null) or block_seq == null or head_seq == null) return error.BadDflashTrace;
    if (update_taps) {
        // The first projection input is the committed tap tile, not an intermediate replay input.
        var taps = try feedCall(h, 0, "_mm", "x_in");
        defer taps.free();
        var tc = try feedCall(h, tap_seq.?, "_prep", "cos_in");
        defer tc.free();
        var ts = try feedCall(h, tap_seq.?, "_prep", "sin_in");
        defer ts.free();
        const rows: usize = @intCast(h.fx.parsed.value.object.get("params").?.object.get("tap_rows").?.array.items[step].integer);
        try glm.dflash_forward.addTaps(context, model, scratch, dk, qk, glue, taps, tc, ts, rows, 1e-5, h.stream);
    }
    const committed = context.logical.end;
    var bc = try feedCall(h, block_seq.?, "_prep", "cos_in");
    defer bc.free();
    var bs = try feedCall(h, block_seq.?, "_prep", "sin_in");
    defer bs.free();
    var ids: [8]i32 = undefined;
    @memset(&ids, 154856);
    ids[0] = @intCast(try h.fx.int("pending"));
    try scratch.get("ids").upload(0, std.mem.sliceAsBytes(&ids));
    if (!update_taps) {
        try qk.rotary(scratch.get("cos").ptr, scratch.get("sin").ptr, committed, 8, model.theta, h.stream);
        try h.same(try h.arrayName(block_seq.?, "_prep", "cos_in"), scratch.get("cos").ptr);
        try h.same(try h.arrayName(block_seq.?, "_prep", "sin_in"), scratch.get("sin").ptr);
    }
    try glm.dflash_forward.block(context, model, head, scratch, dk, qk, glue, embedding.ptr, if (update_taps) bc.ptr else scratch.get("cos").ptr, if (update_taps) bs.ptr else scratch.get("sin").ptr, 1e-5, .{ .ctx = &gathers, .run = Gather.run }, hp, h.stream);
    try check.expect(gathers.at == 10 and context.logical.end == committed, "DFlash2 block preserves committed position and consumes ten rank partials", .{});
    try h.same(try h.arrayName(head_seq.?, "_mm", "ret"), scratch.get("logits").ptr);
    for (0..5) |i| {
        var name: [64]u8 = undefined;
        try h.same(try std.fmt.bufPrint(&name, "r{d}_s{d}_k{d}", .{ h.rank, step, i }), context.keys[i].ptr);
        try h.same(try std.fmt.bufPrint(&name, "r{d}_s{d}_v{d}", .{ h.rank, step, i }), context.values[i].ptr);
    }
    try h.stream.synchronize();
    const projected_bytes = try h.fx.bytes(try std.fmt.bufPrint(&sb, "r{d}_s{d}_projection", .{ h.rank, step }));
    defer a.free(projected_bytes);
    var native: [7 * 256]u16 = undefined;
    try scratch.get("projected").download(0, std.mem.sliceAsBytes(&native));
    try check.expect(projected_bytes.len == native.len * 8, "DFlash2 captured projection dtype", .{});
    var expanded: [7 * 256]f32 = undefined;
    for (native, 0..) |bits, i| {
        const want: f64 = @bitCast(std.mem.readInt(u64, projected_bytes[i * 8 ..][0..8], .little));
        const got: f32 = @bitCast(@as(u32, bits) << 16);
        expanded[i] = got;
        try check.expect(@as(f64, got) == want, "DFlash2 BF16 hidden projection matches captured value", .{});
    }
    try checkCandidates(h, model, scratch.get("logits"), &expanded, head_seq.?, step);
}

fn checkCandidates(h: *hs.Harness, model: *const glm.dflash_weights.Model, logits: cuda.DeviceBuffer, projected: []const f32, head_seq: i64, step: usize) !void {
    const D = glm.dflash2;
    const a = h.gpu.gpa;
    const own = try a.alloc(u16, 7 * 77440);
    defer a.free(own);
    try logits.download(0, std.mem.sliceAsBytes(own));
    var name: [128]u8 = undefined;
    const peer_raw = try h.fx.bytes(try std.fmt.bufPrint(&name, "r{d}_s{d}_{d:0>3}__mm_ret", .{ 1 - h.rank, step, @as(u64, @intCast(head_seq)) }));
    defer a.free(peer_raw);
    const peer = try a.alloc(u16, own.len);
    defer a.free(peer);
    if (peer_raw.len != peer.len * 2) return error.BadDflashTrace;
    @memcpy(std.mem.sliceAsBytes(peer), peer_raw);
    var candidates: [7 * 16]D.Candidate = undefined;
    const token_raw = try h.fx.bytes(try std.fmt.bufPrint(&name, "r{d}_s{d}_tokens", .{ h.rank, step }));
    defer a.free(token_raw);
    const value_raw = try h.fx.bytes(try std.fmt.bufPrint(&name, "r{d}_s{d}_values", .{ h.rank, step }));
    defer a.free(value_raw);
    if (token_raw.len != candidates.len * 8 or value_raw.len != candidates.len * 8) return error.BadDflashTrace;
    for (0..7) |row| {
        const local = try D.topCandidates(a, own[row * 77440 ..][0..77440], @intCast(h.rank * 77440), 16);
        defer a.free(local);
        const remote = try D.topCandidates(a, peer[row * 77440 ..][0..77440], @intCast((1 - h.rank) * 77440), 16);
        defer a.free(remote);
        var merged: [32]D.Candidate = undefined;
        @memcpy(merged[0..16], local);
        @memcpy(merged[16..32], remote);
        std.mem.sort(D.Candidate, &merged, {}, D.candidateBefore);
        @memcpy(candidates[row * 16 ..][0..16], merged[0..16]);
    }
    for (candidates, 0..) |candidate, i| {
        const token = std.mem.readInt(i64, token_raw[i * 8 ..][0..8], .little);
        const value: f64 = @bitCast(std.mem.readInt(u64, value_raw[i * 8 ..][0..8], .little));
        try check.expect(candidate.token == token and candidate.logit == value, "DFlash2 merged candidate ID/value equals reference", .{});
    }
    const chain = try D.selectGreedy(a, &candidates, 16, projected, 256, model.predecessor, model.successor, @intCast(try h.fx.int("pending")), 0.3);
    defer a.free(chain);
    const raw_chain = try h.fx.bytes(try std.fmt.bufPrint(&name, "r{d}_s{d}_chain", .{ h.rank, step }));
    defer a.free(raw_chain);
    try check.expect(raw_chain.len == chain.len * 4, "DFlash2 confidence chain length equals reference", .{});
    for (chain, 0..) |token, i| try check.expect(token == std.mem.readInt(u32, raw_chain[i * 4 ..][0..4], .little), "DFlash2 selected chain equals reference", .{});
}
