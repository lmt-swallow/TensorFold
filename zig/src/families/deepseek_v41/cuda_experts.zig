//! EXL3 routed experts on CUDA: the DSV4.1 fork's `cuda/exl3/experts.py` (prepare, Scratch, routed) in Zig over the
//! fork's kernels (zig/kernels/cuda/dsv41/experts.cu, experts_decode.cu). Decode and verify windows (fewer than 64
//! rows) take the fork's fused path: decode_prep (grouping, rotated inputs), gate/up with its epilogue, down with the
//! rows' combine; every row's bits are its own, whatever shares the window. Prompt chunks are not ported yet.

const std = @import("std");
const cuda = @import("cuda");
const exl3 = @import("cuda_exl3.zig");

pub const Codebook = exl3.Codebook;
pub const Error = error{ BadShape, BadWidth, Unsupported, OutOfMemory } || cuda.Error;

/// experts.py EXACT_ROWS: windows of this many rows or more are prompt chunks (a path not ported here).
pub const exact_rows: usize = 64;
/// experts.py ACT_F32: SwiGLU in fp32 (DeepSeek-V4.1's setting).
pub const act_f32: c_int = 1;
pub const decode_stages: c_int = 3;

/// experts_grouped.cuh's DecodeEpi, passed to grouped_cp_kernel by value (static_assert in experts_decode.cu).
pub const DecodeEpi = extern struct {
    pick: u64 = 0,
    E: c_int = 0,
    svh_g: u64 = 0,
    svh_u: u64 = 0,
    suh_d: u64 = 0,
    xd: u64 = 0,
    limit: f32 = 0,
    act_mode: c_int = 1,
    svh_d: u64 = 0,
    y: u64 = 0,
    wts: u64 = 0,
    add: u64 = 0,
    out: u64 = 0,
    store_y: c_int = 1,
    cnt: u64 = 0,
    ready: u64 = 0,
    ready_cnt: u64 = 0,
    epoch: u64 = 0,
    discard: c_int = 0,
};

comptime {
    std.debug.assert(@sizeOf(DecodeEpi) == 144);
}

/// The K2 range an instantiation of grouped_cp_kernel covers (grouped_cp_launch's choice from the layer's widths).
pub const Range = enum(u2) { r8_8, r2_10, r2_16 };

pub fn rangeOf(lo: u32, hi: u32) Range {
    if (lo == 8 and hi == 8) return .r8_8;
    if (lo >= 2 and hi <= 10) return .r2_10;
    return .r2_16;
}

pub const Kernels = struct {
    d: *const cuda.Driver,
    prep_mod: cuda.Module,
    decode_mod: cuda.Module,
    decode_prep_bf16: cuda.Function,
    /// [range][epi]
    cp: [3][3]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var prep_mod = try cuda.Module.load(d, cuda.kernels.dsv41_exl3_experts);
        errdefer prep_mod.unload();
        var decode_mod = try cuda.Module.load(d, cuda.kernels.dsv41_exl3_experts_decode);
        errdefer decode_mod.unload();
        var k: Kernels = .{ .d = d, .prep_mod = prep_mod, .decode_mod = decode_mod, .decode_prep_bf16 = undefined, .cp = undefined };
        k.decode_prep_bf16 = try prep_mod.function("_ZN16tf_exl3x_experts18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_");
        const ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 16 } };
        for (ranges, 0..) |r, i| for (0..3) |epi| {
            var buf: [192]u8 = undefined;
            const name = try symbol(&buf, "_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi{d}ELi{d}ELi{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE", .{ r[0], r[1], epi });
            k.cp[i][epi] = try decode_mod.function(name);
        };
        return k;
    }

    pub fn deinit(self: *Kernels) void {
        self.decode_mod.unload();
        self.prep_mod.unload();
    }
};

fn symbol(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]const u8 {
    const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args);
    buf[text.len] = 0;
    return buf[0..text.len :0];
}

/// One expert's projection as the checkpoint stores it (this rank's part): trellis int16 [K/16, N/16, w16] raw bytes
/// (the kernels read this stored layout), suh fp16 [K], svh fp16 [N].
pub const Mat = struct { trellis: []const u8, kt: usize, nt: usize, w16: usize, suh: []const u8, svh: []const u8 };
pub const ExpertMats = struct { gate: Mat, up: Mat, down: Mat };

/// A layer's routed experts on the GPU (experts.py Exl3RoutedExperts): every trellis in one buffer (gate of every
/// expert, then up, then down, each at its own width: weights.py pack_trellises), pointer and K2 tables, stacked
/// suh/svh.
pub const Experts = struct {
    trellises: cuda.DeviceBuffer,
    gate_ptr: cuda.DeviceBuffer,
    up_ptr: cuda.DeviceBuffer,
    down_ptr: cuda.DeviceBuffer,
    gate_k2: cuda.DeviceBuffer,
    up_k2: cuda.DeviceBuffer,
    down_k2: cuda.DeviceBuffer,
    suh_g: cuda.DeviceBuffer, // fp16 [E, D]
    suh_u: cuda.DeviceBuffer,
    svh_g: cuda.DeviceBuffer, // fp16 [E, I]
    svh_u: cuda.DeviceBuffer,
    suh_d: cuda.DeviceBuffer, // fp16 [E, I]
    svh_d: cuda.DeviceBuffer, // fp16 [E, D]
    count: usize, // E (the shared expert last)
    dims: usize, // D
    width: usize, // I (this rank's)
    cb: Codebook,
    k2_gu: [2]u32,
    k2_d: [2]u32,
    aligned16: bool,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, experts: []const ExpertMats, cb: Codebook) !Experts {
        return build(gpa, d, experts, cb, null);
    }

    /// `init` with each trellis copied straight from its own bytes on `stream` (no host staging; asynchronous when
    /// they are pinned), waiting for the copies before it returns. The device bytes are `init`'s.
    pub fn initOn(gpa: std.mem.Allocator, d: *const cuda.Driver, experts: []const ExpertMats, cb: Codebook, stream: cuda.Stream) !Experts {
        return build(gpa, d, experts, cb, stream);
    }

    fn build(gpa: std.mem.Allocator, d: *const cuda.Driver, experts: []const ExpertMats, cb: Codebook, stream: ?cuda.Stream) !Experts {
        const E = experts.len;
        if (E == 0) return error.BadShape;
        const D = experts[0].gate.kt * 16;
        const I = experts[0].gate.nt * 16;
        var total: usize = 0;
        for (experts) |ex| {
            if (ex.gate.kt * 16 != D or ex.gate.nt * 16 != I or ex.up.kt * 16 != D or ex.up.nt * 16 != I or ex.down.kt * 16 != I or ex.down.nt * 16 != D) return error.BadShape;
            for ([_]Mat{ ex.gate, ex.up, ex.down }) |m| {
                if (m.trellis.len != m.kt * m.nt * m.w16 * 2 or m.suh.len != m.kt * 16 * 2 or m.svh.len != m.nt * 16 * 2) return error.BadShape;
                total += m.trellis.len;
            }
        }
        const staging: []u8 = if (stream == null) try gpa.alloc(u8, total) else &.{};
        defer if (stream == null) gpa.free(staging);
        var ptrs = try gpa.alloc(i64, 3 * E);
        defer gpa.free(ptrs);
        var k2s = try gpa.alloc(i32, 3 * E);
        defer gpa.free(k2s);
        var trellises = try cuda.DeviceBuffer.alloc(d, total);
        errdefer trellises.free();
        errdefer if (stream) |st| st.synchronize() catch {};
        var off: usize = 0;
        var aligned = true;
        var k2_gu: [2]u32 = .{ 1 << 30, 0 };
        var k2_d: [2]u32 = .{ 1 << 30, 0 };
        for (0..3) |j| for (experts, 0..) |ex, e| {
            const m = switch (j) {
                0 => ex.gate,
                1 => ex.up,
                else => ex.down,
            };
            if (stream) |st| {
                try trellises.uploadAsync(off, m.trellis, st.handle);
            } else @memcpy(staging[off..][0..m.trellis.len], m.trellis);
            ptrs[j * E + e] = @intCast(trellises.ptr + off);
            if ((trellises.ptr + off) % 16 != 0) aligned = false;
            const k2 = try exl3.k2Of(m.w16);
            k2s[j * E + e] = @intCast(k2);
            const range = if (j == 2) &k2_d else &k2_gu;
            range[0] = @min(range[0], k2);
            range[1] = @max(range[1], k2);
            off += m.trellis.len;
        };
        if (stream) |st| try st.synchronize() else try trellises.upload(0, staging);
        var self: Experts = .{
            .trellises = trellises,
            .gate_ptr = undefined,
            .up_ptr = undefined,
            .down_ptr = undefined,
            .gate_k2 = undefined,
            .up_k2 = undefined,
            .down_k2 = undefined,
            .suh_g = undefined,
            .suh_u = undefined,
            .svh_g = undefined,
            .svh_u = undefined,
            .suh_d = undefined,
            .svh_d = undefined,
            .count = E,
            .dims = D,
            .width = I,
            .cb = cb,
            .k2_gu = k2_gu,
            .k2_d = k2_d,
            .aligned16 = aligned,
        };
        self.gate_ptr = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ptrs[0..E]));
        errdefer self.gate_ptr.free();
        self.up_ptr = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ptrs[E .. 2 * E]));
        errdefer self.up_ptr.free();
        self.down_ptr = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ptrs[2 * E .. 3 * E]));
        errdefer self.down_ptr.free();
        self.gate_k2 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(k2s[0..E]));
        errdefer self.gate_k2.free();
        self.up_k2 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(k2s[E .. 2 * E]));
        errdefer self.up_k2.free();
        self.down_k2 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(k2s[2 * E .. 3 * E]));
        errdefer self.down_k2.free();
        self.suh_g = try stack(gpa, d, experts, .gate, .suh, D);
        errdefer self.suh_g.free();
        self.suh_u = try stack(gpa, d, experts, .up, .suh, D);
        errdefer self.suh_u.free();
        self.svh_g = try stack(gpa, d, experts, .gate, .svh, I);
        errdefer self.svh_g.free();
        self.svh_u = try stack(gpa, d, experts, .up, .svh, I);
        errdefer self.svh_u.free();
        self.suh_d = try stack(gpa, d, experts, .down, .suh, I);
        errdefer self.suh_d.free();
        self.svh_d = try stack(gpa, d, experts, .down, .svh, D);
        errdefer self.svh_d.free();
        return self;
    }

    const Which = enum { gate, up, down };
    const Scale = enum { suh, svh };

    fn stack(gpa: std.mem.Allocator, d: *const cuda.Driver, experts: []const ExpertMats, which: Which, scale: Scale, n: usize) !cuda.DeviceBuffer {
        const host = try gpa.alloc(u8, experts.len * n * 2);
        defer gpa.free(host);
        for (experts, 0..) |ex, e| {
            const m = switch (which) {
                .gate => ex.gate,
                .up => ex.up,
                .down => ex.down,
            };
            const src = if (scale == .suh) m.suh else m.svh;
            @memcpy(host[e * n * 2 ..][0 .. n * 2], src);
        }
        return cuda.DeviceBuffer.fromHost(d, host);
    }

    pub fn deinit(self: *Experts) void {
        inline for (.{ "trellises", "gate_ptr", "up_ptr", "down_ptr", "gate_k2", "up_k2", "down_k2", "suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d" }) |f| @field(self, f).free();
    }
};

/// (n tiles a block, warps, K splits, tiles in flight) for a K -> N projection (experts.py default_config).
pub const Tiles = struct { nt: u32, warps: u32, sk: u32, pf: u32 };

pub fn defaultConfig(K: usize, N: usize, gateup: bool) Error!Tiles {
    const first: Tiles = if (gateup) .{ .nt = 8, .warps = 4, .sk = 4, .pf = 1 } else .{ .nt = 8, .warps = 4, .sk = 1, .pf = 1 };
    const cands = [_]Tiles{ first, .{ .nt = 8, .warps = 4, .sk = 2, .pf = 1 }, .{ .nt = 8, .warps = 4, .sk = 1, .pf = 1 }, .{ .nt = 4, .warps = 4, .sk = 2, .pf = 2 }, .{ .nt = 4, .warps = 4, .sk = 1, .pf = 2 } };
    for (cands) |t| {
        if (K % (16 * t.sk * t.warps) == 0 and N % (16 * t.nt) == 0) return t;
    }
    return error.BadShape;
}

/// Buffers for up to `rows` rows of `slots` slots (experts.py Scratch, decode windows only).
pub const Scratch = struct {
    rows: usize,
    slots: usize,
    maxu: usize, // min(rows * slots, E)
    cfg_gu: Tiles,
    cfg_d: Tiles,
    xg: cuda.DeviceBuffer, // fp16 [P, D]
    xu: cuda.DeviceBuffer,
    xd: cuda.DeviceBuffer, // fp16 [P, I]
    z: cuda.DeviceBuffer, // fp32 [max(2 sk_gu I, sk_d D) * P]
    y: cuda.DeviceBuffer, // fp32 [P, D]
    no_y: cuda.DeviceBuffer, // fp32 [1]: a pointer for what a call skips
    cnt_gu: cuda.DeviceBuffer, // int32 [maxu * I / 128]
    cnt_d: cuda.DeviceBuffer, // int32 [rows * D / 128]
    epoch: cuda.DeviceBuffer, // int32 [1]
    ready: cuda.DeviceBuffer, // int32 [maxu]
    ready_cnt: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer, // int32 [maxu]
    count: cuda.DeviceBuffer, // int32 [1]
    members: cuda.DeviceBuffer, // int32 [maxu * ceil(rows / 128) * 128], -1

    pub fn init(d: *const cuda.Driver, ex: *const Experts, rows: usize, slots: usize) !Scratch {
        const D = ex.dims;
        const I = ex.width;
        const P = rows * slots;
        const maxu = @min(P, ex.count);
        const cfg_gu = try defaultConfig(D, I, true);
        const cfg_d = try defaultConfig(I, D, false);
        const zrow = @max(2 * cfg_gu.sk * I, cfg_d.sk * D);
        var s: Scratch = undefined;
        s.rows = rows;
        s.slots = slots;
        s.maxu = maxu;
        s.cfg_gu = cfg_gu;
        s.cfg_d = cfg_d;
        var made: usize = 0;
        const fields = .{ "xg", "xu", "xd", "z", "y", "no_y", "cnt_gu", "cnt_d", "epoch", "ready", "ready_cnt", "ids", "count", "members" };
        const sizes = [_]usize{ P * D * 2, P * D * 2, P * I * 2, zrow * P * 4, P * D * 4, 4, maxu * @max(1, I / 128) * 4, rows * @max(1, D / 128) * 4, 4, @max(1, maxu) * 4, @max(1, maxu) * 4, @max(1, maxu) * 4, 4, maxu * ((rows + 127) / 128) * 128 * 4 };
        errdefer {
            inline for (fields, 0..) |f, i| if (i < made) @field(s, f).free();
        }
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try cuda.DeviceBuffer.alloc(d, sizes[i]);
            made += 1;
            try @field(s, f).fill32(if (i == fields.len - 1) 0xFFFFFFFF else 0, null);
        }
        return s;
    }

    pub fn deinit(self: *Scratch) void {
        inline for (.{ "xg", "xu", "xd", "z", "y", "no_y", "cnt_gu", "cnt_d", "epoch", "ready", "ready_cnt", "ids", "count", "members" }) |f| @field(self, f).free();
    }
};

/// out [R, D] fp32 = the wts-weighted sum over slots of each row's routed experts' outputs (picks >= E skipped): the
/// fork's fused decode window (experts.py _routed_fused) in three launches on `stream`. x: bf16 [R, D] contiguous;
/// pick: int32 [R, slots]; wts: fp32 [R, slots]; limit: the SwiGLU clamp. R < 64 (a decode or verify window).
pub fn routed(k: *const Kernels, ex: *const Experts, s: *const Scratch, x: cuda.abi.DevicePtr, pick: cuda.abi.DevicePtr, wts: cuda.abi.DevicePtr, R: usize, limit: f32, out: cuda.abi.DevicePtr, stream: cuda.Stream) !void {
    const D = ex.dims;
    const I = ex.width;
    const E = ex.count;
    const slots = s.slots;
    if (R < 1 or R > s.rows or R >= exact_rows or R > 128 or slots > 32) return error.Unsupported;
    if (!(ex.aligned16 and s.cfg_gu.nt == 8 and s.cfg_gu.warps == 4 and s.cfg_d.nt == 8 and s.cfg_d.warps == 4 and s.cfg_d.sk == 1 and I % 128 == 0 and D % 128 == 0)) return error.Unsupported;
    const P = R * slots;
    const maxu = @min(P, E);
    const maxm = R; // Scratch.window: the member lists sized for R rows
    const ci = struct {
        fn f(v: usize) c_int {
            return @intCast(v);
        }
    }.f;
    {
        // decode_prep: block 0 groups the picks, the other blocks rotate the routed rows (a warp a program)
        const n = P;
        const items = n * (D / 128) * 2;
        const blocks: u32 = @intCast(1 + (items + 8 - 1) / 8);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(ci(D)); // x_stride
        a.add(pick);
        a.add(ex.suh_g.ptr);
        a.add(ex.suh_u.ptr);
        a.add(s.xg.ptr);
        a.add(s.xu.ptr);
        a.add(ci(D)); // K
        a.add(ci(slots));
        a.add(ci(E));
        a.add(ci(R));
        a.add(s.ids.ptr);
        a.add(s.count.ptr);
        a.add(s.members.ptr);
        a.add(ci(maxm));
        a.add(wts);
        a.add(s.y.ptr);
        a.add(@as(u64, 0)); // add
        a.add(out);
        a.add(ci(D));
        a.add(@as(c_int, 1)); // store_y
        a.add(s.epoch.ptr);
        try cuda.launch.launch(k.decode_prep_bf16, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 }, .shared = @intCast(2 * n * 4) }, stream, &a);
    }
    const MT: u32 = @intCast((maxm + 15) / 16);
    {
        // gate and up, the SwiGLU epilogue in the program that completes each (expert, 128 columns)
        const sk = s.cfg_gu.sk;
        const ep: DecodeEpi = .{
            .pick = pick,
            .E = ci(E),
            .svh_g = ex.svh_g.ptr,
            .svh_u = ex.svh_u.ptr,
            .suh_d = ex.suh_d.ptr,
            .xd = s.xd.ptr,
            .limit = limit,
            .act_mode = act_f32,
            .cnt = s.cnt_gu.ptr,
            .ready = s.ready.ptr,
            .ready_cnt = s.ready_cnt.ptr,
            .epoch = s.epoch.ptr,
            .discard = if (8 * sk <= 32) 1 else 0,
        };
        var a: cuda.Args = .{};
        a.add(s.xg.ptr);
        a.add(s.xu.ptr);
        a.add(ex.gate_ptr.ptr);
        a.add(ex.up_ptr.ptr);
        a.add(ex.gate_k2.ptr);
        a.add(ex.up_k2.ptr);
        a.add(s.ids.ptr);
        a.add(s.count.ptr);
        a.add(s.members.ptr);
        a.add(s.z.ptr);
        a.add(ci(D)); // K
        a.add(ci(I)); // N
        a.add(ci(P));
        a.add(ci(sk));
        a.add(ci(maxm));
        a.add(ci(slots));
        a.add(ep);
        const f = k.cp[@intFromEnum(rangeOf(ex.k2_gu[0], ex.k2_gu[1]))][1];
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(maxu), .y = @intCast(I / 128), .z = 2 * sk * MT }, .block = .{ .x = 128 }, .pdl = true }, stream, &a);
    }
    {
        // down, one split, each slot's output and the rows' combine with wts in the program that completes a row
        const ep: DecodeEpi = .{
            .pick = pick,
            .E = ci(E),
            .svh_d = ex.svh_d.ptr,
            .y = s.y.ptr,
            .wts = wts,
            .add = 0,
            .out = out,
            .store_y = 1,
            .cnt = s.cnt_d.ptr,
            .ready = s.ready.ptr,
            .ready_cnt = s.ready_cnt.ptr,
            .epoch = s.epoch.ptr,
            .discard = if (4 * slots <= 32) 1 else 0,
        };
        var a: cuda.Args = .{};
        a.add(s.xd.ptr);
        a.add(s.xd.ptr);
        a.add(ex.down_ptr.ptr);
        a.add(ex.down_ptr.ptr);
        a.add(ex.down_k2.ptr);
        a.add(ex.down_k2.ptr);
        a.add(s.ids.ptr);
        a.add(s.count.ptr);
        a.add(s.members.ptr);
        a.add(s.z.ptr);
        a.add(ci(I)); // K
        a.add(ci(D)); // N
        a.add(ci(P));
        a.add(@as(c_int, 1)); // SK
        a.add(ci(maxm));
        a.add(ci(slots));
        a.add(ep);
        const f = k.cp[@intFromEnum(rangeOf(ex.k2_d[0], ex.k2_d[1]))][2];
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(maxu), .y = @intCast(D / 128), .z = MT }, .block = .{ .x = 128 }, .pdl = true }, stream, &a);
    }
}

test "the DSV4.1 TP2 rank's tile settings are the fused decode path's" {
    try std.testing.expectEqual(Tiles{ .nt = 8, .warps = 4, .sk = 4, .pf = 1 }, try defaultConfig(5120, 1152, true));
    try std.testing.expectEqual(Tiles{ .nt = 8, .warps = 4, .sk = 1, .pf = 1 }, try defaultConfig(1152, 5120, false));
    try std.testing.expectEqual(Range.r2_10, rangeOf(4, 10));
    try std.testing.expectEqual(Range.r8_8, rangeOf(8, 8));
    try std.testing.expectEqual(Range.r2_16, rangeOf(4, 12));
}
