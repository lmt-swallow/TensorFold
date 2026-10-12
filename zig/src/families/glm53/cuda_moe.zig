//! GLM-5.3-Flash's MoE block on one rank, as the Mia fork's `forward.py: moe_block` runs it on the EXL3 pack: the
//! router and top-8 of 288 (Triton, cuda_glue.zig), the pairs grouped by expert (tensorfold/cuda/experts.cu's plan,
//! the experts fatbin), the routed experts through `exl3.cu` (zig/kernels/cuda/glm53/exl3.cu, the glm53_exl3
//! fatbin) and the shared expert through the BF16 matmuls, then the weighted sum into the rank's fp32 partial.
//!
//! Decode windows (exl3_mm.py `routed`, R < 64 rows): one rotated fp16 input a row (`rot_rows`: every expert of a
//! layer shares one gate/up suh), `dec_kernel` for gate and up (4 K splits), the gate/up epilogue, `dec_kernel`
//! for down with its epilogue fused; trellis words through 16-byte non-coherent loads (TF_GLM_EXL3_LOADS=nc).
//! Prompt chunks (a 64-pair plan): `rot_rows`, then the prompt kernels (Y^T = W^T X^T, one fp32 chain a row), their
//! items in `prompt_order_kernel`'s interleaved order. The two forms give a row different bits, each independent
//! of the window's other rows.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const map = @import("weights_map.zig");
const glue_mod = @import("cuda_glue.zig");
const weights = @import("cuda_weights.zig");

const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

/// tensorfold/cuda/experts.py TILE (a decode plan's items) and exl3_mm.py's prompt pass (TF_GLM_EXL3_PASS).
pub const decode_tile = 16;
pub const prompt_tile = 64;
/// exl3_mm.py PROMPT_ROWS: from this many rows a window would take the prompt order on the grouped kernel; decode
/// windows stay below it.
pub const prompt_rows = 64;
/// tensorfold/cuda/experts.py SMALL: pairs the one-block plan takes.
const plan_small = 1024;

/// experts.py `max_items`: an item per used expert, plus one per `tile` pairs past its first.
pub fn maxItems(pairs: usize, experts: usize, tile: usize) usize {
    return @min(pairs, experts) + pairs / tile;
}

pub const Kernels = struct {
    exl3: cuda.Module,
    plan_mod: cuda.Module,
    rot_rows: cuda.Function,
    dec_gateup: cuda.Function, // dec_kernel<8, 4, 16, 0, xrow, nc, 1>
    dec_down: cuda.Function, // dec_kernel<8, 4, 16, 1 (fused down epilogue), no xrow, nc, 1>
    gateup_epilogue: cuda.Function,
    prompt_order: cuda.Function,
    prompt_gateup_shx: cuda.Function,
    prompt_down: cuda.Function,
    smem: [2]u32, // prompt_smem<64>(), prompt_smem<64, 1>()
    plan_small: cuda.Function,
    plan_rank: cuda.Function,
    plan_offsets: cuda.Function,
    plan_scatter: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var x = try cuda.Module.load(d, cuda.kernels.glm53_exl3);
        errdefer x.unload();
        var p = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer p.unload();
        var k: Kernels = undefined;
        k.exl3 = x;
        k.plan_mod = p;
        k.rot_rows = try x.function("_ZN13tf_glm53_exl313rot_in_kernelILb1EEEvPK13__nv_bfloat16iPKiPK6__halfS8_PS6_S9_ii");
        k.dec_gateup = try x.function("_ZN13tf_glm53_exl310dec_kernelILi8ELi4ELi16ELi0ELb1ELi1ELi1EEEvPK6__halfS3_PKjS5_PKiS7_S7_PfiiiiiiS3_S3_S3_PvfPi");
        k.dec_down = try x.function("_ZN13tf_glm53_exl310dec_kernelILi8ELi4ELi16ELi1ELb0ELi1ELi1EEEvPK6__halfS3_PKjS5_PKiS7_S7_PfiiiiiiS3_S3_S3_PvfPi");
        k.gateup_epilogue = try x.function("_ZN13tf_glm53_exl322gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiif");
        k.prompt_order = try x.function("_ZN13tf_glm53_exl319prompt_order_kernelEPKiS1_Piii");
        k.prompt_gateup_shx = try x.function("_ZN13tf_glm53_exl320prompt_gateup_kernelILi64ELi8ELb1EEEvPK6__halfS3_PKjS5_PKiS7_S7_S3_S3_S3_PS1_iiifiS7_");
        k.prompt_down = try x.function("_ZN13tf_glm53_exl318prompt_down_kernelILi64ELi8EEEvPK6__halfPKjPKiS7_S7_S3_PfiiiS7_");
        const g = try x.global("tf_glm53_exl3_prompt_smem");
        var smem: [2]u64 = undefined;
        const view: Buf = .{ .d = d, .ptr = g.ptr, .len = g.len };
        try view.download(0, std.mem.sliceAsBytes(&smem));
        k.smem = .{ @intCast(smem[0]), @intCast(smem[1]) };
        try k.prompt_gateup_shx.allowDynamicShared(k.smem[1]);
        try k.prompt_down.allowDynamicShared(k.smem[0]);
        k.plan_small = try p.function("_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_");
        k.plan_rank = try p.function("_ZN10tf_experts9plan_rankEPKiiiPiS2_");
        k.plan_offsets = try p.function("_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_");
        k.plan_scatter = try p.function("_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.exl3.unload();
        k.plan_mod.unload();
    }
};

/// One layer's routed experts on this rank (exl3_mm.py `Exl3Experts`): trellis words int32 (gate / up [E, D/16,
/// NI/16, 32], down [E, NI/16, D/16, 32]), their fp16 scales, and the gate/up input scale every expert shares.
pub const Experts = struct {
    gt: Buf,
    ut: Buf,
    dt: Buf,
    svh_g: Buf, // [E, NI]
    svh_u: Buf,
    suh_d: Buf, // [E, NI]
    svh_d: Buf, // [E, D]
    shared_suh: Buf, // [D]: exl3_mm.py `shared_suh` (refused when the experts' gate/up suh differ)
    count: usize,
    width: usize, // NI
    dims: usize, // D

    pub fn load(l: weights.Loader, i: usize) !Experts {
        const c = l.lay.cfg;
        const E = c.experts;
        const NI = l.lay.size(.moe);
        const D = c.hidden;
        var x: Experts = undefined;
        x.count = E;
        x.width = NI;
        x.dims = D;
        const sizes = [_]usize{ D / 16 * NI / 16 * 64, D / 16 * NI / 16 * 64, NI / 16 * D / 16 * 64, NI, NI, NI, D };
        const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj", "gate_proj", "up_proj", "down_proj", "down_proj" };
        const parts = [_][]const u8{ "trellis", "trellis", "trellis", "svh", "svh", "suh", "svh" };
        var bufs: [7]*Buf = .{ &x.gt, &x.ut, &x.dt, &x.svh_g, &x.svh_u, &x.suh_d, &x.svh_d };
        var made: usize = 0;
        errdefer for (bufs[0..made]) |b| b.free();
        var n: map.Name = .{};
        for (bufs, sizes, projs, parts) |b, size, proj, part| {
            b.* = try Buf.alloc(l.d, E * size * 2);
            made += 1;
            for (0..E) |e| {
                const name = n.of(map.lm ++ "layers.{d}.mlp.experts.{d}.{s}.{s}", .{ i, e, proj, part });
                const host = try l.hostShare(name);
                defer l.gpa.free(host);
                if (host.len != size * 2) return error.UnexpectedTensor;
                try b.upload(e * size * 2, host);
            }
        }
        // every expert's gate and up suh equal bit for bit: one rotated input a row serves them all
        const g0 = try l.hostShare(n.of(map.lm ++ "layers.{d}.mlp.experts.0.gate_proj.suh", .{i}));
        defer l.gpa.free(g0);
        if (g0.len != D * 2) return error.UnexpectedTensor;
        for (0..E) |e| for ([_][]const u8{ "gate_proj", "up_proj" }) |proj| {
            const s = try l.hostShare(n.of(map.lm ++ "layers.{d}.mlp.experts.{d}.{s}.suh", .{ i, e, proj }));
            defer l.gpa.free(s);
            if (s.len != D * 2) return error.UnexpectedTensor;
            if (!std.mem.eql(u8, s, g0)) return error.ExpertsDoNotShareSuh;
        };
        x.shared_suh = try Buf.fromHost(l.d, g0);
        return x;
    }

    pub fn deinit(x: *Experts) void {
        inline for (.{ "gt", "ut", "dt", "svh_g", "svh_u", "suh_d", "svh_d", "shared_suh" }) |f| @field(x, f).free();
    }
};

/// A MoE layer's rank share (weights.py `MoEW` on the EXL3 pack).
pub const Moe = struct {
    router: Buf, // [E, D] bf16
    bias: Buf, // [E] fp32
    experts: Experts,
    shared: weights.Mlp,

    pub fn load(l: weights.Loader, i: usize) !Moe {
        var n: map.Name = .{};
        var router = try l.share(map.layer(&n, i, "mlp.gate.weight"));
        errdefer router.free();
        var bias = try l.share(map.layer(&n, i, "mlp.gate.e_score_correction_bias"));
        errdefer bias.free();
        var x = try Experts.load(l, i);
        errdefer x.deinit();
        return .{ .router = router, .bias = bias, .experts = x, .shared = try weights.Mlp.load(l, "mlp.shared_experts.", i) };
    }

    pub fn deinit(m: *Moe) void {
        m.router.free();
        m.bias.free();
        m.experts.deinit();
        m.shared.deinit();
    }
};

/// experts.py `Plan` and exl3_mm.py `Scratch` with the block's own buffers, for windows of up to `rows` rows.
pub const Scratch = struct {
    rows: usize,
    slots: usize,
    prefill: bool,
    tile: usize,
    logits: Buf, // [rows, E] fp32
    rpart: Buf, // the router's K-slice partials [8, rows, E] fp32
    pick: Buf, // [rows, slots] i32
    wts: Buf, // [rows, slots] fp32
    members: Buf,
    items: Buf,
    counts: Buf,
    rank: Buf,
    hist: Buf,
    xg: Buf, // [P, D] fp16
    xd: Buf, // [P, NI] fp16
    z: Buf,
    done: Buf,
    order: Buf,
    y: Buf, // [rows, slots, D] fp32: each routed pair's output (the rank's share)
    sgu: Buf, // [rows, 2 W_shared] bf16
    sact: Buf,
    sxs: Buf,
    sy: Buf, // [rows, D] fp32: the shared expert's output

    pub fn init(d: *const cuda.Driver, lay: config.Layout, rows: usize, prefill: bool) !Scratch {
        const c = lay.cfg;
        const slots = c.slots();
        const P = rows * slots;
        const D = c.hidden;
        const NI = lay.size(.moe);
        const SW = lay.size(.shared);
        const tile: usize = if (prefill) prompt_tile else decode_tile;
        var s: Scratch = undefined;
        s.rows = rows;
        s.slots = slots;
        s.prefill = prefill;
        s.tile = tile;
        const a = struct {
            fn f(dd: *const cuda.Driver, n: usize) !Buf {
                return Buf.alloc(dd, @max(n, 16));
            }
        }.f;
        s.logits = try a(d, rows * c.experts * 4);
        s.rpart = try a(d, 8 * rows * c.experts * 4);
        s.pick = try a(d, P * 4);
        s.wts = try a(d, P * 4);
        s.members = try a(d, P * 4);
        s.items = try a(d, maxItems(P, c.experts + 1, decode_tile) * 3 * 4);
        s.counts = try a(d, 2 * 4);
        const wide = P > plan_small;
        s.rank = try a(d, if (wide) P * 4 else 4);
        s.hist = try a(d, if (wide) (P + 1023) / 1024 * (c.experts + 1) * 4 else 4);
        // exl3_mm.py Scratch allocates them zeroed: rows a window does not write keep those zeros (or what an
        // earlier window wrote), as the reference's do
        s.xg = try a(d, P * D * 2);
        try s.xg.fill8(0, null);
        s.xd = try a(d, P * NI * 2);
        try s.xd.fill8(0, null);
        s.z = try a(d, if (prefill) 4 else 2 * 4 * P * @max(NI, D) * 4);
        s.done = try a(d, if (prefill) 4 else (P + P / decode_tile + 1) * @max(1, NI / 128) * 4);
        try s.done.fill8(0, null);
        s.order = try a(d, if (prefill) (P / 16 + 1024) * 4 else 4);
        s.y = try a(d, P * D * 4);
        s.sgu = try a(d, rows * 2 * SW * 2);
        s.sact = try a(d, rows * SW * 2);
        s.sxs = try a(d, rows * SW / 64 * 4);
        s.sy = try a(d, rows * D * 4);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (.{ "logits", "rpart", "pick", "wts", "members", "items", "counts", "rank", "hist", "xg", "xd", "z", "done", "order", "y", "sgu", "sact", "sxs", "sy" }) |f| @field(s, f).free();
    }
};

fn int(v: usize) c_int {
    return @intCast(v);
}

/// experts.py `route`: rows 0 .. R's (row, slot) pairs grouped by expert (ids 0 .. experts, the shared slot's id
/// `experts` included) into items of at most the scratch's tile.
pub fn route(k: *const Kernels, s: *const Scratch, R: usize, experts: usize, stream: cuda.Stream) !void {
    const P = R * s.slots;
    const E = experts + 1;
    if (P <= plan_small) {
        var a: cuda.Args = .{};
        a.add(s.pick.ptr);
        a.add(int(P));
        a.add(int(E));
        a.add(int(s.tile));
        a.add(s.members.ptr);
        a.add(s.items.ptr);
        a.add(s.counts.ptr);
        return cuda.launch.launch(k.plan_small, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &a);
    }
    const nblk = (P + 1023) / 1024;
    var a: cuda.Args = .{};
    a.add(s.pick.ptr);
    a.add(int(P));
    a.add(int(E));
    a.add(s.rank.ptr);
    a.add(s.hist.ptr);
    try cuda.launch.launch(k.plan_rank, .{ .grid = .{ .x = @intCast(nblk) }, .block = .{ .x = 1024 } }, stream, &a);
    var b: cuda.Args = .{};
    b.add(int(nblk));
    b.add(int(E));
    b.add(int(s.tile));
    b.add(s.hist.ptr);
    b.add(s.items.ptr);
    b.add(s.counts.ptr);
    try cuda.launch.launch(k.plan_offsets, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &b);
    var c: cuda.Args = .{};
    c.add(s.pick.ptr);
    c.add(int(P));
    c.add(int(E));
    c.add(s.rank.ptr);
    c.add(s.hist.ptr);
    c.add(s.members.ptr);
    try cuda.launch.launch(k.plan_scatter, .{ .grid = .{ .x = @intCast((P + 255) / 256) }, .block = .{ .x = 256 } }, stream, &c);
}

/// exl3_mm.py `routed`: s.y [R, slots, D] fp32 for every routed pair of rows 0 .. R of the normed rows x (row
/// stride x_stride), from the plan `route` left.
pub fn routed(k: *const Kernels, x: *const Experts, s: *const Scratch, xin: Ptr, x_stride: usize, R: usize, limit: f32, stream: cuda.Stream) !void {
    const P = R * s.slots;
    const D = x.dims;
    const NI = x.width;
    // one rotated fp16 input a row (rot_rows): every pair of the row reads it for gate and up
    {
        var a: cuda.Args = .{};
        a.add(xin);
        a.add(int(x_stride));
        a.add(@as(Ptr, 0));
        a.add(x.shared_suh.ptr);
        a.add(x.shared_suh.ptr);
        a.add(s.xg.ptr);
        a.add(s.xg.ptr);
        a.add(int(D));
        a.add(@as(c_int, 1));
        try cuda.launch.launch(k.rot_rows, .{ .grid = .{ .x = @intCast(R), .y = @intCast(D / 128), .z = 1 }, .block = .{ .x = 32 } }, stream, &a);
    }
    if (s.prefill) {
        const items = maxItems(P, x.count + 1, s.tile);
        var o: cuda.Args = .{};
        o.add(s.items.ptr);
        o.add(s.counts.ptr);
        o.add(s.order.ptr);
        o.add(int(items));
        o.add(int(x.count));
        try cuda.launch.launch(k.prompt_order, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &o);
        var g: cuda.Args = .{};
        g.add(s.xg.ptr);
        g.add(s.xg.ptr);
        g.add(x.gt.ptr);
        g.add(x.ut.ptr);
        g.add(s.items.ptr);
        g.add(s.counts.ptr);
        g.add(s.members.ptr);
        g.add(x.svh_g.ptr);
        g.add(x.svh_u.ptr);
        g.add(x.suh_d.ptr);
        g.add(s.xd.ptr);
        g.add(int(D));
        g.add(int(NI));
        g.add(int(x.count));
        g.add(limit);
        g.add(int(s.slots));
        g.add(s.order.ptr);
        try cuda.launch.launch(k.prompt_gateup_shx, .{ .grid = .{ .x = @intCast(NI / 128), .y = @intCast(items) }, .block = .{ .x = 8 * 32 }, .shared = k.smem[1] }, stream, &g);
        var dn: cuda.Args = .{};
        dn.add(s.xd.ptr);
        dn.add(x.dt.ptr);
        dn.add(s.items.ptr);
        dn.add(s.counts.ptr);
        dn.add(s.members.ptr);
        dn.add(x.svh_d.ptr);
        dn.add(s.y.ptr);
        dn.add(int(NI));
        dn.add(int(D));
        dn.add(int(x.count));
        dn.add(s.order.ptr);
        try cuda.launch.launch(k.prompt_down, .{ .grid = .{ .x = @intCast(D / 256), .y = @intCast(items) }, .block = .{ .x = 8 * 32 }, .shared = k.smem[0] }, stream, &dn);
        return;
    }
    if (R >= prompt_rows) return error.DecodeWindowTooWide;
    const items = maxItems(P, x.count + 1, decode_tile);
    // gate and up: 4 K splits, 8 column tiles a block, Z partials for the epilogue
    var g: cuda.Args = .{};
    g.add(s.xg.ptr);
    g.add(s.xg.ptr);
    g.add(x.gt.ptr);
    g.add(x.ut.ptr);
    g.add(s.items.ptr);
    g.add(s.counts.ptr);
    g.add(s.members.ptr);
    g.add(s.z.ptr);
    g.add(int(D));
    g.add(int(NI));
    g.add(int(P));
    g.add(@as(c_int, 4));
    g.add(int(x.count));
    g.add(int(s.slots));
    g.add(x.svh_g.ptr);
    g.add(x.svh_u.ptr);
    g.add(x.suh_d.ptr);
    g.add(s.xd.ptr);
    g.add(limit);
    g.add(s.done.ptr);
    try cuda.launch.launch(k.dec_gateup, .{ .grid = .{ .x = @intCast(items), .y = @intCast(NI / 128), .z = 2 * 4 }, .block = .{ .x = 128 } }, stream, &g);
    var e: cuda.Args = .{};
    e.add(s.z.ptr);
    e.add(s.pick.ptr);
    e.add(x.svh_g.ptr);
    e.add(x.svh_u.ptr);
    e.add(x.suh_d.ptr);
    e.add(s.xd.ptr);
    e.add(int(P));
    e.add(int(NI));
    e.add(@as(c_int, 4));
    e.add(int(s.slots));
    e.add(limit);
    try cuda.launch.launch(k.gateup_epilogue, .{ .grid = .{ .x = @intCast(P), .y = @intCast(NI / 128) }, .block = .{ .x = 32 } }, stream, &e);
    // down: one split, its epilogue fused (Y written by the kernel)
    var dn: cuda.Args = .{};
    dn.add(s.xd.ptr);
    dn.add(s.xd.ptr);
    dn.add(x.dt.ptr);
    dn.add(x.dt.ptr);
    dn.add(s.items.ptr);
    dn.add(s.counts.ptr);
    dn.add(s.members.ptr);
    dn.add(s.z.ptr);
    dn.add(int(NI));
    dn.add(int(D));
    dn.add(int(P));
    dn.add(@as(c_int, 1));
    dn.add(int(x.count));
    dn.add(int(s.slots));
    dn.add(x.svh_d.ptr);
    dn.add(x.svh_d.ptr);
    dn.add(x.svh_d.ptr);
    dn.add(s.y.ptr);
    dn.add(@as(f32, 0));
    dn.add(s.done.ptr);
    try cuda.launch.launch(k.dec_down, .{ .grid = .{ .x = @intCast(items), .y = @intCast(D / 128), .z = 1 }, .block = .{ .x = 128 } }, stream, &dn);
}

/// forward.py `moe_block` up to the all-gather, on b.normed: router, top-k, the plan, the routed experts, the
/// shared expert, the weighted sum into b.part.
pub fn moeBlock(g: glue_mod.Glue, k: *const Kernels, lay: config.Layout, m: *const Moe, s: *const Scratch, normed: Ptr, part: Ptr, sk: Ptr, R: usize, stream: cuda.Stream) !void {
    const c = lay.cfg;
    try g.router(normed, c.hidden, m.router.ptr, c.experts, R, s.logits.ptr, s.rpart.ptr, stream);
    try g.select(s.logits.ptr, m.bias.ptr, s.pick.ptr, s.wts.ptr, c.top_k, c.experts, c.routed_scale, c.norm_topk, R, stream);
    try route(k, s, R, c.experts, stream);
    try routed(k, &m.experts, s, normed, c.hidden, R, c.limit, stream);
    const sh = &m.shared;
    try g.bmm(normed, c.hidden, sh.gu.b16(), R, s.sgu.ptr, false, sk, stream);
    try g.swiglu(s.sgu.ptr, s.sact.ptr, s.sxs.ptr, c.limit, R, sh.width, stream);
    try g.bmm(s.sact.ptr, sh.width, sh.down.b16(), R, s.sy.ptr, true, sk, stream);
    try g.combineShared(s.y.ptr, s.sy.ptr, s.wts.ptr, part, s.slots, R, stream);
}

test "experts.py max_items" {
    // a 16-row decode window: 144 pairs over 289 ids
    try std.testing.expectEqual(@as(usize, 144 + 9), maxItems(144, 289, 16));
    // a 2,048-row prompt chunk on 64-pair items
    try std.testing.expectEqual(@as(usize, 289 + 288), maxItems(2048 * 9, 289, 64));
}
