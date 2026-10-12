//! A layer's routed EXL3 experts (the shared expert as expert `experts`, the last row of every table) on TensorFold
//! 0.6.6's grouped kernel (qwen38fn_experts fatbin: `cuda/exl3/experts.cu`, `experts_grouped.cuh`), launched as its
//! `experts.py: routed` launches them for a decode window: group the picks by expert, rotate each slot's input, the
//! gate and up projections in one grouped launch, the SwiGLU epilogue (fp32), the down projection, its epilogue.
//! Each expert matrix keeps its own width; the tile setting and K splits come from the shapes alone, so a row's
//! output never depends on the window.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const upload = @import("cuda_upload.zig");
const weights = @import("cuda_weights.zig");
const names = @import("weights_map.zig");

const Ptr = cuda.abi.DevicePtr;

/// experts.py ACT_F32: SwiGLU in fp32 (Flash Next's mode; 0 is GLM's bf16 roundings).
pub const act_f32: c_int = 1;

/// experts.py `default_config`: (n tiles a block, warps, K splits, tiles in flight) for a K -> N projection.
pub const Tiles = struct { nt: u32, w: u32, sk: u32, pf: u32 };

pub fn defaultTiles(k: usize, n: usize, gateup: bool) !Tiles {
    const cands = [_]Tiles{
        if (gateup) .{ .nt = 8, .w = 4, .sk = 4, .pf = 1 } else .{ .nt = 8, .w = 4, .sk = 1, .pf = 1 },
        .{ .nt = 8, .w = 4, .sk = 2, .pf = 1 },
        .{ .nt = 8, .w = 4, .sk = 1, .pf = 1 },
        .{ .nt = 4, .w = 4, .sk = 2, .pf = 2 },
        .{ .nt = 4, .w = 4, .sk = 1, .pf = 2 },
    };
    for (cands) |c| if (k % (16 * c.sk * c.w) == 0 and n % (16 * c.nt) == 0) return c;
    return error.NoTileSetting;
}

/// grouped_launch's K2 range for a projection whose experts' half-bits run from lo to hi.
fn range(lo: u32, hi: u32) [2]u32 {
    if (lo == 8 and hi == 8) return .{ 8, 8 };
    if (lo >= 2 and hi <= 10) return .{ 2, 10 };
    return .{ 2, 16 };
}

pub const Kernels = struct {
    module: cuda.Module,
    prompt: cuda.Module, // experts_prompt.cu: a prompt window's experts with the reference pipeline's bytes
    list: cuda.Function,
    plan: cuda.Function,
    diff: cuda.Function,
    group: cuda.Function,
    rot_in_bf16: cuda.Function,
    gateup: cuda.Function,
    down: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.qwen38fn_experts);
        errdefer module.unload();
        var prompt = try cuda.Module.load(d, cuda.kernels.qwen38fn_experts_prompt);
        errdefer prompt.unload();
        return .{
            .module = module,
            .prompt = prompt,
            .list = try prompt.function("tf_qfn_xp_list"),
            .plan = try prompt.function("tf_qfn_xp_plan"),
            .diff = try prompt.function("tf_qfn_xp_count_diff"),
            .group = try module.function("_ZN8tf_qfn_x12group_kernelEPKiPiS2_S2_iiii"),
            .rot_in_bf16 = try module.function("_ZN8tf_qfn_x13rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iii"),
            .gateup = try module.function("_ZN8tf_qfn_x22gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiifi"),
            .down = try module.function("_ZN8tf_qfn_x20down_epilogue_kernelEPKfPKiPK6__halfPfiiii"),
        };
    }

    pub fn deinit(self: *Kernels) void {
        self.module.unload();
        self.prompt.unload();
    }

    /// experts_prompt.cu tf_qfn_xp_{gateup,down}_<cb>_<lo>_<hi>.
    fn promptFn(self: *const Kernels, comptime which: []const u8, cb: weights.Codebook, r: [2]u32) !cuda.Function {
        var buf: [64]u8 = undefined;
        const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "tf_qfn_xp_" ++ which ++ "_{d}_{d}_{d}", .{ @intFromEnum(cb), r[0], r[1] });
        buf[text.len] = 0;
        return self.prompt.function(buf[0..text.len :0]);
    }

    /// grouped_kernel<CB, NT, W, PF, LO, HI> as cuobjdump lists it.
    fn grouped(self: *const Kernels, cb: weights.Codebook, t: Tiles, r: [2]u32) !cuda.Function {
        var buf: [160]u8 = undefined;
        const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "_ZN8tf_exl3x14grouped_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiii", .{ @intFromEnum(cb), t.nt, t.w, t.pf, r[0], r[1] });
        buf[text.len] = 0;
        return self.module.function(buf[0..text.len :0]);
    }
};

/// experts.py `Exl3RoutedExperts`: trellis pointers and half-bits per expert and projection, stacked scales.
pub const Experts = struct {
    trellis: cuda.DeviceBuffer, // every expert's three trellises, 256-byte aligned
    gate_ptr: cuda.DeviceBuffer, // int64 [E]
    up_ptr: cuda.DeviceBuffer,
    down_ptr: cuda.DeviceBuffer,
    gate_k2: cuda.DeviceBuffer, // int32 [E]
    up_k2: cuda.DeviceBuffer,
    down_k2: cuda.DeviceBuffer,
    suh_g: cuda.DeviceBuffer, // fp16 [E, D]
    suh_u: cuda.DeviceBuffer,
    svh_g: cuda.DeviceBuffer, // fp16 [E, I]
    svh_u: cuda.DeviceBuffer,
    suh_d: cuda.DeviceBuffer, // fp16 [E, I]
    svh_d: cuda.DeviceBuffer, // fp16 [E, D]
    count: usize, // E: routed experts + the shared one
    dims: usize, // D
    width: usize, // I
    cb: weights.Codebook,
    k2_gu: [2]u32, // (min, max) over gate and up
    k2_d: [2]u32,

    /// Layer i's routed experts and its shared expert (exl3.py `expert_table`).
    /// With `up`, the trellises' copies are queued on it (landed once it finishes); else each is uploaded here.
    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, pack: *weights.Pack, cfg: *const Config, i: names.Where, up: ?*upload.Uploader) !Experts {
        const E = cfg.experts + 1;
        if (cfg.shared_width != cfg.moe_width) return error.UnsupportedShape; // one width I for the whole table
        const D = cfg.hidden;
        const I = cfg.moe_width;
        const groups = try gpa.alloc(weights.Group, 3 * E);
        defer gpa.free(groups);
        var n: names.Name = .{};
        var total: usize = 0;
        var cb: ?weights.Codebook = null;
        for (0..E) |e| for (0..3) |j| {
            const prefix = switch (j) {
                0 => names.expert(&n, i, e, cfg.experts, "gate_proj"),
                1 => names.expert(&n, i, e, cfg.experts, "up_proj"),
                else => names.expert(&n, i, e, cfg.experts, "down_proj"),
            };
            const g = try pack.group(prefix);
            const k: usize = if (j == 2) I else D;
            const nn: usize = if (j == 2) D else I;
            if (g.k() != k or g.n() != nn or g.w16 % 8 != 0 or g.w16 / 8 < 2 or g.w16 / 8 > 16) return error.UnexpectedTensor;
            if (cb != null and cb.? != g.codebook) return error.MixedCodebooks; // the grouped kernel takes one a layer
            cb = g.codebook;
            groups[3 * e + j] = g;
            total += std.mem.alignForward(usize, g.trellis.len, 256);
        };
        var trellis = try cuda.DeviceBuffer.alloc(d, total);
        errdefer trellis.free();
        // the layer's trellises as one direct read when they share a file (each layer's span is contiguous)
        var span: ?struct { bytes: []const u8, lo: u64 } = null;
        if (up) |u| if (u.span) |*r| span: {
            const first = pack.locate(groups[0].trellis) orelse break :span;
            var lo: u64 = first.offset;
            var hi: u64 = first.offset + groups[0].trellis.len;
            for (groups) |g| {
                const at = pack.locate(g.trellis) orelse break :span;
                if (!std.mem.eql(u8, at.path, first.path)) break :span;
                lo = @min(lo, at.offset);
                hi = @max(hi, at.offset + g.trellis.len);
            }
            if (hi - lo > r.buf[0].bytes.len - 2 * 4096) break :span;
            span = .{ .bytes = try r.read(first.path, lo, @intCast(hi - lo)), .lo = lo };
        };
        if (up != null and span == null) for (groups) |g| upload.prefetch(g.trellis);
        const ptrs = try gpa.alloc(i64, 3 * E);
        defer gpa.free(ptrs);
        const k2s = try gpa.alloc(i32, 3 * E);
        defer gpa.free(k2s);
        var at: usize = 0;
        var k2_gu: [2]u32 = .{ 16, 2 };
        var k2_d: [2]u32 = .{ 16, 2 };
        for (0..E) |e| for (0..3) |j| {
            const g = groups[3 * e + j];
            if (span) |sp| {
                const off: usize = @intCast(pack.locate(g.trellis).?.offset - sp.lo);
                try up.?.putPinned(trellis, at, sp.bytes[off..][0..g.trellis.len]);
            } else if (up) |u| try u.put(trellis, at, g.trellis) else try trellis.upload(at, g.trellis);
            ptrs[j * E + e] = @intCast(trellis.ptr + at);
            const k2: u32 = @intCast(g.w16 / 8);
            k2s[j * E + e] = @intCast(k2);
            const r = if (j == 2) &k2_d else &k2_gu;
            r[0] = @min(r[0], k2);
            r[1] = @max(r[1], k2);
            at += std.mem.alignForward(usize, g.trellis.len, 256);
        };
        if (span != null) try up.?.span.?.done(up.?);
        var bufs: [12]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (0..3) |j| {
            bufs[made] = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ptrs[j * E ..][0..E]));
            made += 1;
        }
        for (0..3) |j| {
            bufs[made] = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(k2s[j * E ..][0..E]));
            made += 1;
        }
        // stacked scales: gate suh, up suh, gate svh, up svh, down suh, down svh (experts.py `prepare`)
        const order = [_]struct { usize, bool }{ .{ 0, true }, .{ 1, true }, .{ 0, false }, .{ 1, false }, .{ 2, true }, .{ 2, false } };
        for (order) |o| {
            const j, const suh = o;
            const len = if (suh) groups[j].suh.len else groups[j].svh.len;
            const host = try gpa.alloc(u8, E * len);
            defer gpa.free(host);
            for (0..E) |e| @memcpy(host[e * len ..][0..len], if (suh) groups[3 * e + j].suh else groups[3 * e + j].svh);
            bufs[made] = try cuda.DeviceBuffer.fromHost(d, host);
            made += 1;
        }
        return .{
            .trellis = trellis, .gate_ptr = bufs[0], .up_ptr = bufs[1], .down_ptr = bufs[2], .gate_k2 = bufs[3], .up_k2 = bufs[4], .down_k2 = bufs[5],
            .suh_g = bufs[6], .suh_u = bufs[7], .svh_g = bufs[8], .svh_u = bufs[9], .suh_d = bufs[10], .svh_d = bufs[11],
            .count = E, .dims = D, .width = I, .cb = cb.?, .k2_gu = k2_gu, .k2_d = k2_d,
        };
    }

    pub fn deinit(self: *Experts) void {
        for ([_]*cuda.DeviceBuffer{ &self.trellis, &self.gate_ptr, &self.up_ptr, &self.down_ptr, &self.gate_k2, &self.up_k2, &self.down_k2, &self.suh_g, &self.suh_u, &self.svh_g, &self.svh_u, &self.suh_d, &self.svh_d }) |b| b.free();
    }
};

/// The kernels of a prompt window's experts. `captured`: 0.6.6's pipeline (`routed`, then the bf16 copy). `fused`:
/// experts_prompt.cu (`promptWindow`), the same bytes in four launches. `check`: both, the fused slots compared with
/// the captured ones byte for byte after every window (error.ExpertsDiffer); the captured ones are kept.
pub const Mode = enum { captured, fused, check };

/// TF_QFN_XP_DOWN: the down programs of a fused window. `loop3` (default) / `loop2`: one program per 48 / 32 members
/// over every column, the members' rows staged once; `tiles`: one per (64 members, 256 columns). The same bytes every
/// way (spark-616b, 2,048-row chunk: loop3 166 ms, tiles 195, loop2 191).
const DownVariant = enum {
    tiles,
    loop2,
    loop3,

    fn fromEnv() !DownVariant {
        const v = std.mem.span(std.c.getenv("TF_QFN_XP_DOWN") orelse return .loop3);
        if (v.len == 0) return .loop3;
        return std.meta.stringToEnum(DownVariant, v) orelse error.BadExpertsVariant;
    }

    fn mt(v: DownVariant) usize {
        return switch (v) {
            .tiles => 4,
            .loop2 => 2,
            .loop3 => 3,
        };
    }
};

/// experts_prompt.cu: members a gate/up program and a down program, and their dynamic shared bytes.
const gu_members = 32;
const down_members = 64;
const gu_shared: u32 = gu_members * (640 + 8) * 2;
const down_shared: u32 = down_members * (640 + 8) * 2;

/// experts.py `Scratch` for windows of up to `rows` rows of `slots` slots.
pub const Scratch = struct {
    xg: cuda.DeviceBuffer, // fp16 [P, D]
    xu: cuda.DeviceBuffer,
    xd: cuda.DeviceBuffer, // fp16 [P, I]
    z: cuda.DeviceBuffer, // fp32 partials
    y: cuda.DeviceBuffer, // fp32 [P, D]: each slot's expert output
    ids: cuda.DeviceBuffer, // int32 [min(P, E)]
    count: cuda.DeviceBuffer, // int32 [1]
    members: cuda.DeviceBuffer, // int32 [min(P, E) * rows]
    rows: usize,
    slots: usize,
    gu: Tiles,
    dn: Tiles,
    mode: Mode = .captured,
    // fused and check: the rows a fused window takes, each expert's slots, their counts, the work items and their
    // count, the SwiGLU outputs
    frows: usize = 0,
    lists: ?cuda.DeviceBuffer = null, // int32 [E, frows]
    counts: ?cuda.DeviceBuffer = null, // int32 [E]
    items: ?cuda.DeviceBuffer = null, // gate/up's, then down's (each maxItems(gu_members) long)
    nitems: ?cuda.DeviceBuffer = null, // int32 [2]
    fxd: ?cuda.DeviceBuffer = null, // fp16 [frows * slots, I]
    // check: the fused slots (bf16 [frows * slots, D]) and the differing words (u64)
    ycheck: ?cuda.DeviceBuffer = null,
    diffs: ?cuda.DeviceBuffer = null,

    fn maxItems(self: *const Scratch, E: usize, members: usize) usize {
        const P = self.frows * self.slots;
        return P / members + @min(P, E) + 1;
    }

    /// Runs prompt windows in `mode` from now on, a fused window taking up to `frows` rows (its buffers allocated
    /// here, once). The fused kernels take Flash Next's shapes: every split of 640 K values (gate/up 4 splits of the
    /// model width, down 1 of the expert width), 4 warps, and at most 1,024 experts.
    pub fn setMode(self: *Scratch, d: *const cuda.Driver, ex: *const Experts, mode: Mode, frows: usize) !void {
        if (mode != .captured and self.lists == null) {
            if (ex.count > 1024 or self.gu.w != 4 or self.dn.w != 4 or ex.dims / self.gu.sk != 640 or ex.width / self.dn.sk != 640 or self.dn.sk != 1 or ex.width % 128 != 0 or ex.dims % 256 != 0) return error.UnsupportedShape;
            self.frows = frows;
            self.lists = try cuda.DeviceBuffer.alloc(d, ex.count * frows * 4);
            self.counts = try cuda.DeviceBuffer.alloc(d, ex.count * 4);
            self.items = try cuda.DeviceBuffer.alloc(d, 2 * self.maxItems(ex.count, gu_members) * 4);
            self.nitems = try cuda.DeviceBuffer.alloc(d, 16);
            self.fxd = try cuda.DeviceBuffer.alloc(d, frows * self.slots * ex.width * 2);
        }
        if (mode == .check and self.ycheck == null) {
            self.ycheck = try cuda.DeviceBuffer.alloc(d, self.frows * self.slots * ex.dims * 2);
            self.diffs = try cuda.DeviceBuffer.alloc(d, 8);
        }
        self.mode = mode;
    }

    pub fn init(d: *const cuda.Driver, ex: *const Experts, rows: usize, slots: usize) !Scratch {
        const D = ex.dims;
        const I = ex.width;
        const P = rows * slots;
        const gu = try defaultTiles(D, I, true);
        const dn = try defaultTiles(I, D, false);
        const maxu = @min(P, ex.count);
        var s: Scratch = .{ .xg = undefined, .xu = undefined, .xd = undefined, .z = undefined, .y = undefined, .ids = undefined, .count = undefined, .members = undefined, .rows = rows, .slots = slots, .gu = gu, .dn = dn };
        const sizes = [_]struct { *cuda.DeviceBuffer, usize }{
            .{ &s.xg, P * D * 2 }, .{ &s.xu, P * D * 2 }, .{ &s.xd, P * I * 2 },
            .{ &s.z, @max(2 * gu.sk * I, dn.sk * D) * P * 4 }, .{ &s.y, P * D * 4 }, .{ &s.ids, maxu * 4 },
            .{ &s.count, 4 }, .{ &s.members, maxu * rows * 4 },
        };
        var made: usize = 0;
        errdefer for (sizes[0..made]) |x| x[0].free();
        for (sizes) |x| {
            x[0].* = try cuda.DeviceBuffer.alloc(d, x[1]);
            try x[0].fill8(0, null);
            made += 1;
        }
        try s.members.fill32(0xFFFFFFFF, null);
        return s;
    }

    pub fn deinit(self: *Scratch) void {
        for ([_]*cuda.DeviceBuffer{ &self.xg, &self.xu, &self.xd, &self.z, &self.y, &self.ids, &self.count, &self.members }) |b| b.free();
        for ([_]*?cuda.DeviceBuffer{ &self.lists, &self.counts, &self.items, &self.nitems, &self.fxd, &self.ycheck, &self.diffs }) |b| if (b.*) |*x| x.free();
    }
};

fn grouped(k: *const Kernels, ex: *const Experts, s: *const Scratch, x0: Ptr, x1: Ptr, tp0: Ptr, tp1: Ptr, b0: Ptr, b1: Ptr, mats: u32, K: usize, N: usize, P: usize, t: Tiles, r: [2]u32, maxu: usize, maxm: usize, stream: cuda.Stream) !void {
    const f = try k.grouped(ex.cb, t, r);
    var a: cuda.Args = .{};
    a.add(x0);
    a.add(x1);
    a.add(tp0);
    a.add(tp1);
    a.add(b0);
    a.add(b1);
    a.add(s.ids.ptr);
    a.add(s.count.ptr);
    a.add(s.members.ptr);
    a.add(s.z.ptr);
    for ([_]usize{ K, N, P, t.sk, maxm, s.slots }) |v| a.add(@as(c_int, @intCast(v)));
    const mt = (maxm + 15) / 16;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(maxu), .y = @intCast(N / (16 * t.nt)), .z = @intCast(mats * t.sk * mt) }, .block = .{ .x = t.w * 32 } }, stream, &a);
}

/// experts.py `routed(x, pick, None, ex, s, None, R)`: each slot's expert output into s.y fp32 [R * slots, D] for the
/// bf16 rows x (row stride x_stride) and their picks int32 [R, slots]; slots whose pick is not an expert are left.
pub fn routed(k: *const Kernels, ex: *const Experts, s: *const Scratch, x: Ptr, x_stride: usize, pick: Ptr, R: usize, stream: cuda.Stream) !Ptr {
    if (R > s.rows) return error.WindowTooLarge;
    const D = ex.dims;
    const I = ex.width;
    const E = ex.count;
    const P = R * s.slots;
    const maxu = @min(P, E);
    const maxm = R; // Scratch.window: members [maxu, R]
    {
        var a: cuda.Args = .{};
        a.add(pick);
        a.add(s.ids.ptr);
        a.add(s.count.ptr);
        a.add(s.members.ptr);
        for ([_]usize{ R, s.slots, E, maxm }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(k.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = @intCast(P * 4) }, stream, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intCast(x_stride)));
        a.add(pick);
        a.add(ex.suh_g.ptr);
        a.add(ex.suh_u.ptr);
        a.add(s.xg.ptr);
        a.add(s.xu.ptr);
        for ([_]usize{ D, s.slots, E }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(k.rot_in_bf16, .{ .grid = .{ .x = @intCast(P), .y = @intCast(D / 128), .z = 2 }, .block = .{ .x = 32 } }, stream, &a);
    }
    try grouped(k, ex, s, s.xg.ptr, s.xu.ptr, ex.gate_ptr.ptr, ex.up_ptr.ptr, ex.gate_k2.ptr, ex.up_k2.ptr, 2, D, I, P, s.gu, range(ex.k2_gu[0], ex.k2_gu[1]), maxu, maxm, stream);
    {
        var a: cuda.Args = .{};
        a.add(s.z.ptr);
        a.add(pick);
        a.add(ex.svh_g.ptr);
        a.add(ex.svh_u.ptr);
        a.add(ex.suh_d.ptr);
        a.add(s.xd.ptr);
        for ([_]usize{ P, I, s.gu.sk, E }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(std.math.inf(f32)); // routed's limit: math.inf
        a.add(act_f32);
        try cuda.launch.launch(k.gateup, .{ .grid = .{ .x = @intCast(P), .y = @intCast(I / 128) }, .block = .{ .x = 32 } }, stream, &a);
    }
    try grouped(k, ex, s, s.xd.ptr, s.xd.ptr, ex.down_ptr.ptr, ex.down_ptr.ptr, ex.down_k2.ptr, ex.down_k2.ptr, 1, I, D, P, s.dn, range(ex.k2_d[0], ex.k2_d[1]), maxu, maxm, stream);
    {
        var a: cuda.Args = .{};
        a.add(s.z.ptr);
        a.add(pick);
        a.add(ex.svh_d.ptr);
        a.add(s.y.ptr);
        for ([_]usize{ P, D, s.dn.sk, E }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(k.down, .{ .grid = .{ .x = @intCast(P), .y = @intCast(D / 128) }, .block = .{ .x = 32 } }, stream, &a);
    }
    return s.y.ptr;
}

/// `routed` for a prompt window (up to s.frows rows; rows are independent, so one window may take what `routed` takes
/// in several) followed by its bf16 copy: each slot's expert output, rounded to bf16, into out bf16
/// [R * slots, D] (row stride D) for the bf16 rows x (row stride x_stride) and their picks int32 [R, slots], on
/// experts_prompt.cu (the bytes `routed` and tf_qfn_f32_to_bf16 write; slots whose pick is not an expert are left).
pub fn promptWindow(k: *const Kernels, ex: *const Experts, s: *const Scratch, x: Ptr, x_stride: usize, pick: Ptr, R: usize, out: Ptr, stream: cuda.Stream) !void {
    if (R > s.frows) return error.WindowTooLarge;
    const D = ex.dims;
    const I = ex.width;
    const E = ex.count;
    const P = R * s.slots;
    const lists = s.lists orelse return error.PromptModeUnset;
    const counts = s.counts.?;
    const items = s.items.?;
    const nitems = s.nitems.?;
    const xd = s.fxd.?;
    {
        var a: cuda.Args = .{};
        a.add(pick);
        a.add(@as(c_int, @intCast(P)));
        a.add(lists.ptr);
        a.add(@as(c_int, @intCast(s.frows)));
        a.add(counts.ptr);
        try cuda.launch.launch(k.list, .{ .grid = .{ .x = @intCast(E) }, .block = .{ .x = 256 } }, stream, &a);
    }
    const dn = try DownVariant.fromEnv();
    const dn_members = 16 * dn.mt();
    const stride = s.maxItems(E, gu_members) * 4;
    for ([_]usize{ gu_members, dn_members }, 0..) |mg, i| {
        var a: cuda.Args = .{};
        a.add(counts.ptr);
        a.add(@as(c_int, @intCast(E)));
        a.add(items.ptr + i * stride);
        a.add(nitems.ptr + i * 4);
        a.add(@as(c_int, @intCast(mg)));
        try cuda.launch.launch(k.plan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &a);
    }
    {
        const f = try k.promptFn("gateup", ex.cb, range(ex.k2_gu[0], ex.k2_gu[1]));
        try f.allowDynamicShared(gu_shared);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intCast(x_stride)));
        a.add(@as(c_int, @intCast(s.slots)));
        a.add(items.ptr);
        a.add(nitems.ptr);
        a.add(lists.ptr);
        a.add(@as(c_int, @intCast(s.frows)));
        a.add(counts.ptr);
        for ([_]Ptr{ ex.gate_ptr.ptr, ex.up_ptr.ptr, ex.gate_k2.ptr, ex.up_k2.ptr, ex.suh_g.ptr, ex.suh_u.ptr, ex.svh_g.ptr, ex.svh_u.ptr, ex.suh_d.ptr, xd.ptr }) |v| a.add(v);
        a.add(@as(c_int, @intCast(D)));
        a.add(@as(c_int, @intCast(I)));
        a.add(std.math.inf(f32)); // routed's limit: math.inf
        a.add(act_f32);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(s.maxItems(E, gu_members)), .y = @intCast(I / 128) }, .block = .{ .x = 256 }, .shared = gu_shared }, stream, &a);
    }
    if (dn != .tiles) {
        var buf: [64]u8 = undefined;
        const r = range(ex.k2_d[0], ex.k2_d[1]);
        const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "tf_qfn_xp_downloop{d}_{d}_{d}_{d}", .{ dn.mt(), @intFromEnum(ex.cb), r[0], r[1] });
        buf[text.len] = 0;
        const f = try k.prompt.function(buf[0..text.len :0]);
        const shared: u32 = @intCast(dn_members * ((640 + 8) * 2 + 128 * 4));
        try f.allowDynamicShared(shared);
        var a: cuda.Args = .{};
        a.add(xd.ptr);
        a.add(items.ptr + stride);
        a.add(nitems.ptr + 4);
        a.add(lists.ptr);
        a.add(@as(c_int, @intCast(s.frows)));
        a.add(counts.ptr);
        a.add(ex.down_ptr.ptr);
        a.add(ex.down_k2.ptr);
        a.add(ex.svh_d.ptr);
        a.add(out);
        a.add(@as(c_int, @intCast(I)));
        a.add(@as(c_int, @intCast(D)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(s.maxItems(E, dn_members)) }, .block = .{ .x = 512 }, .shared = shared }, stream, &a);
    } else {
        const f = try k.promptFn("down", ex.cb, range(ex.k2_d[0], ex.k2_d[1]));
        try f.allowDynamicShared(down_shared);
        var a: cuda.Args = .{};
        a.add(xd.ptr);
        a.add(items.ptr + stride);
        a.add(nitems.ptr + 4);
        a.add(lists.ptr);
        a.add(@as(c_int, @intCast(s.frows)));
        a.add(counts.ptr);
        a.add(ex.down_ptr.ptr);
        a.add(ex.down_k2.ptr);
        a.add(ex.svh_d.ptr);
        a.add(out);
        a.add(@as(c_int, @intCast(I)));
        a.add(@as(c_int, @intCast(D)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(s.maxItems(E, down_members)), .y = @intCast(D / 256) }, .block = .{ .x = 512 }, .shared = down_shared }, stream, &a);
    }
}

/// Check mode: the fused slots (s.ycheck) against the captured ones at `want`, `bytes` long.
pub fn checkWindow(k: *const Kernels, s: *const Scratch, want: Ptr, bytes: usize, stream: cuda.Stream) !void {
    return checkBytes(k, s, s.ycheck.?.ptr, want, bytes, "slots", stream);
}

/// Check mode: the fused SwiGLU outputs of a captured window's rows (from slot `p0`) against the captured ones.
pub fn checkSwiglu(k: *const Kernels, s: *const Scratch, p0: usize, slots_n: usize, stream: cuda.Stream) !void {
    return checkBytes(k, s, s.fxd.?.ptr + p0 * s.xd.len / (s.rows * s.slots), s.xd.ptr, slots_n * s.xd.len / (s.rows * s.slots), "SwiGLU outputs", stream);
}

fn checkBytes(k: *const Kernels, s: *const Scratch, got: Ptr, want: Ptr, bytes: usize, what: []const u8, stream: cuda.Stream) !void {
    const diffs = s.diffs.?;
    try diffs.fill8(0, stream.handle);
    var a: cuda.Args = .{};
    a.add(got);
    a.add(want);
    a.add(@as(c_ulonglong, bytes / 4));
    a.add(diffs.ptr);
    try cuda.launch.launch(k.diff, .{ .grid = .{ .x = 1024 }, .block = .{ .x = 256 } }, stream, &a);
    var n: u64 = 0;
    try diffs.downloadAsync(0, std.mem.asBytes(&n), stream.handle);
    try stream.synchronize();
    if (n != 0) {
        std.debug.print("qwen38fn experts check: the fused {s} differ from the captured ones in {d} of {d} words\n", .{ what, n, bytes / 4 });
        return error.ExpertsDiffer;
    }
}

test "tile settings follow the shapes" {
    try std.testing.expectEqual(Tiles{ .nt = 8, .w = 4, .sk = 4, .pf = 1 }, try defaultTiles(2560, 640, true));
    try std.testing.expectEqual(Tiles{ .nt = 8, .w = 4, .sk = 1, .pf = 1 }, try defaultTiles(640, 2560, false));
    try std.testing.expectEqual([2]u32{ 2, 10 }, range(6, 10));
    try std.testing.expectEqual([2]u32{ 8, 8 }, range(8, 8));
    try std.testing.expectEqual([2]u32{ 2, 16 }, range(6, 12));
}
