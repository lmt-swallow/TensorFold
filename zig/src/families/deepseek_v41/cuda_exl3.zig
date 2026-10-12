//! EXL3 linear layers on CUDA: the DSV4.1 fork's `cuda/exl3/linear.py` and `linear.cpp` in Zig over the fork's
//! kernels (zig/kernels/cuda/dsv41/linear.cu). y [M, N] = x [M, K] @ W + bias for 1 to 128 rows, each row's bits
//! its own: the K split and warps a program come from the layer's shape alone (`plan`), so a layer keeps one
//! reduction order whatever the row count.

const std = @import("std");
const cuda = @import("cuda");

pub const Codebook = enum(c_int) { inst3 = 0, mcg = 1, mul1 = 2 };
pub const DType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

pub const Error = error{ BadShape, BadWidth, OutOfMemory } || cuda.Error;

/// (K splits, warps a program) for a K x N layer (linear.py `plan`): the shape's alone.
pub const Plan = struct { sk: u32, wk: u32 };

pub fn plan(k: usize, n: usize) Plan {
    const blocks: usize = 192;
    const min_tiles: usize = 8;
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) { // a narrow layer needs K splits to fill the SMs at all
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ .sk = @intCast(sk), .wk = @intCast(wk) };
}

/// Trellis words [K/16, N/16, W] (W = 8 * bits 32-bit words a tile) -> strips [N/128, K/16, 8, W]: each 128-column
/// block's tiles in k order (linear.py `strips`). The caller frees the result.
pub fn packStrips(gpa: std.mem.Allocator, words: []const u32, kt: usize, nt: usize, w: usize) ![]u32 {
    if (words.len != kt * nt * w or nt % 8 != 0) return error.BadShape;
    const out = try gpa.alloc(u32, words.len);
    const nb = nt / 8;
    for (0..nb) |b| for (0..kt) |k| for (0..8) |t| {
        const src = ((k * nt) + (b * 8 + t)) * w;
        const dst = (((b * kt) + k) * 8 + t) * w;
        @memcpy(out[dst..][0..w], words[src..][0..w]);
    };
    return out;
}

/// K2 (= 2 * bits) from a trellis' last dimension in int16 words (16 * bits).
pub fn k2Of(w16: usize) Error!u32 {
    if (w16 % 8 != 0) return error.BadWidth;
    const k2 = w16 / 8;
    const known = [_]usize{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 };
    for (known) |v| if (v == k2) return @intCast(k2);
    return error.BadWidth;
}

/// The fatbin's kernels: one module, rot_in and the instantiation a layer's (K2, codebook, warps) picks.
pub const Kernels = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    rot_in: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_exl3_linear);
        errdefer module.unload();
        const rot_in = try module.function("_ZN14tf_exl3_linear13rot_in_kernelEPKviPK6__halfPS2_i");
        return .{ .d = d, .module = module, .rot_in = rot_in };
    }

    pub fn deinit(self: *Kernels) void {
        self.module.unload();
    }

    /// linear_kernel<K2, CB, WK> as cuobjdump lists it for the built fatbin.
    pub fn linear(self: *const Kernels, k2: u32, cb: Codebook, wk: u32) !cuda.Function {
        var buf: [160]u8 = undefined;
        const name = try symbol(&buf, "_ZN14tf_exl3_linear13linear_kernelILi{d}ELi{d}ELi{d}EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii", .{ k2, @intFromEnum(cb), wk });
        return self.module.function(name);
    }

    /// tf_exl3_prep_<K2>_<CB>: rot_in_kernel's rows and unpack_kernel's W_q in one launch (linear.cu `prep`).
    pub fn prep(self: *const Kernels, k2: u32, cb: Codebook) !cuda.Function {
        var buf: [64]u8 = undefined;
        const name = try symbol(&buf, "tf_exl3_prep_{d}_{d}", .{ k2, @intFromEnum(cb) });
        return self.module.function(name);
    }

    /// unpack_kernel<K2, CB>: W_q [K, N] fp16 from the words.
    pub fn unpack(self: *const Kernels, k2: u32, cb: Codebook) !cuda.Function {
        var buf: [160]u8 = undefined;
        const name = try symbol(&buf, "_ZN14tf_exl3_linear13unpack_kernelILi{d}ELi{d}EEEvPKjP6__halfill", .{ k2, @intFromEnum(cb) });
        return self.module.function(name);
    }
};

/// A mangled name formatted into `buf` with its terminator (cuModuleGetFunction wants a C string).
fn symbol(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]const u8 {
    const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args);
    buf[text.len] = 0;
    return buf[0..text.len :0];
}

/// One EXL3 layer on the GPU: strips of trellis words, fp16 suh [K] and svh [N], no bias (the DSV4.1 pack has none).
pub const Linear = struct {
    words: cuda.DeviceBuffer,
    suh: cuda.DeviceBuffer,
    svh: cuda.DeviceBuffer,
    counters: cuda.DeviceBuffer,
    k: usize,
    n: usize,
    k2: u32,
    cb: Codebook,
    sk: u32,
    wk: u32,
    kernel: cuda.Function,
    rot_in: cuda.Function,

    /// From a group's checkpoint tensors: trellis int16 [K/16, N/16, w16] raw bytes, suh / svh fp16 raw bytes.
    pub fn init(gpa: std.mem.Allocator, kernels: *const Kernels, trellis: []const u8, kt: usize, nt: usize, w16: usize, suh: []const u8, svh: []const u8, cb: Codebook) !Linear {
        const d = kernels.d;
        const k = 16 * kt;
        const n = 16 * nt;
        if (k % 128 != 0 or n % 128 != 0) return error.BadShape;
        if (trellis.len != kt * nt * w16 * 2 or suh.len != k * 2 or svh.len != n * 2) return error.BadShape;
        const k2 = try k2Of(w16);
        const w = w16 / 2;
        const stored = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(trellis)));
        const strips = try packStrips(gpa, stored, kt, nt, w);
        defer gpa.free(strips);
        var words = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(strips));
        errdefer words.free();
        var suh_d = try cuda.DeviceBuffer.fromHost(d, suh);
        errdefer suh_d.free();
        var svh_d = try cuda.DeviceBuffer.fromHost(d, svh);
        errdefer svh_d.free();
        var counters = try cuda.DeviceBuffer.alloc(d, 8 * (n / 128) * 4);
        errdefer counters.free();
        try counters.fill32(0, null);
        const p = plan(k, n);
        return .{
            .words = words,
            .suh = suh_d,
            .svh = svh_d,
            .counters = counters,
            .k = k,
            .n = n,
            .k2 = k2,
            .cb = cb,
            .sk = p.sk,
            .wk = p.wk,
            .kernel = try kernels.linear(k2, cb, p.wk),
            .rot_in = kernels.rot_in,
        };
    }

    pub fn deinit(self: *Linear) void {
        self.words.free();
        self.suh.free();
        self.svh.free();
        self.counters.free();
    }

    /// (words between k tiles, words between 128-column blocks) in `words` (strips).
    pub fn strides(self: *const Linear) struct { k: i64, nb: i64 } {
        const tw: i64 = 4 * @as(i64, self.k2);
        return .{ .k = 8 * tw, .nb = @as(i64, @intCast(self.k / 16)) * 8 * tw };
    }

    /// fp32 words of split-K scratch `z` a run of m rows needs (0 when the plan has one split).
    pub fn scratchWords(self: *const Linear, m: usize) usize {
        return if (self.sk > 1) self.sk * m * self.n else 0;
    }

    /// y [m, N] (y_dtype) = x [m, K] (x_dtype: fp16, bf16 or fp32) @ W; xh: fp16 scratch [m, K]; z: fp32 scratch of
    /// `scratchWords(m)` (any pointer when none is needed). Two launches on `stream`, as the Python `__call__`.
    pub fn run(self: *const Linear, x: cuda.abi.DevicePtr, x_dtype: DType, m: usize, xh: cuda.abi.DevicePtr, z: cuda.abi.DevicePtr, y: cuda.abi.DevicePtr, y_dtype: DType, stream: cuda.Stream) !void {
        if (m < 1 or m > 128) return error.BadShape;
        const K: c_int = @intCast(self.k);
        const N: c_int = @intCast(self.n);
        const M: c_int = @intCast(m);
        {
            var a: cuda.Args = .{};
            a.add(x);
            a.add(@intFromEnum(x_dtype));
            a.add(self.suh.ptr);
            a.add(xh);
            a.add(K);
            try cuda.launch.launch(self.rot_in, .{ .grid = .{ .x = @intCast((self.k / 128 + 3) / 4), .y = @intCast(m) }, .block = .{ .x = 128 } }, stream, &a);
        }
        {
            const s = self.strides();
            var a: cuda.Args = .{};
            a.add(xh);
            a.add(self.words.ptr);
            a.add(s.k);
            a.add(s.nb);
            a.add(self.svh.ptr);
            a.add(@as(u64, 0)); // bias
            a.add(y);
            a.add(@intFromEnum(y_dtype));
            a.add(if (self.sk > 1) z else @as(u64, 0));
            a.add(self.counters.ptr);
            a.add(M);
            a.add(K);
            a.add(N);
            a.add(@as(c_int, @intCast(self.sk)));
            const smem: u32 = @intCast(self.wk * @min(m, 8) * 128 * @sizeOf(f32));
            try cuda.launch.launch(self.kernel, .{ .grid = .{ .x = @intCast(self.n / 128), .y = self.sk }, .block = .{ .x = self.wk * 32 }, .shared = smem }, stream, &a);
        }
    }
};

// -- grouped launches (linear_grouped.cu: the fork's Exl3Group) -----------------------------------------------------

/// Layers a grouped launch takes at most (linear_grouped.cu GMAX).
pub const gmax = 8;

/// linear_grouped.cu `GLayer`, field for field (C layout).
pub const GLayer = extern struct {
    xh: u64 = 0,
    t: u64 = 0,
    svh: u64 = 0,
    bias: u64 = 0,
    y: u64 = 0,
    z: u64 = 0,
    counters: u64 = 0,
    ldx: i64 = 0,
    ldy: i64 = 0,
    stride_k: i64 = 0,
    stride_nb: i64 = 0,
    y_dtype: c_int = 0,
    k: c_int = 0,
    n: c_int = 0,
    sk: c_int = 0,
    first: c_int = 0,
    rsuh: u64 = 0,
    rxh: u64 = 0,
    ldr: i64 = 0,
    roff: c_int = 0,
    rcos: u64 = 0,
    rsin: u64 = 0,
    rpos: u64 = 0,
    rhd: c_int = 0,
    rrd: c_int = 0,
};

/// linear_grouped.cu `GArgs`.
pub const GArgs = extern struct {
    l: [gmax]GLayer = @splat(.{}),
    n: c_int = 0,
    m: c_int = 0,
    discard: c_int = 0,
};

/// linear_grouped.cu `RLayer`.
pub const RLayer = extern struct {
    x: u64 = 0,
    suh: u64 = 0,
    xh: u64 = 0,
    ldx: i64 = 0,
    x_dtype: c_int = 0,
    k: c_int = 0,
    first: c_int = 0,
};

/// linear_grouped.cu `RArgs`.
pub const RArgs = extern struct {
    l: [gmax]RLayer = @splat(.{}),
    n: c_int = 0,
    m: c_int = 0,
};

/// The grouped kernels' module: rot_many and glinear<K2, mul1, WK> (WK 4 or 8).
pub const GroupKernels = struct {
    module: cuda.Module,
    rot_many: cuda.Function,

    pub fn load(d: *const cuda.Driver) !GroupKernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_exl3_grouped);
        errdefer module.unload();
        const rot_many = try module.function("_ZN15tf_exl3_grouped15rot_many_kernelENS_5RArgsE");
        return .{ .module = module, .rot_many = rot_many };
    }

    pub fn deinit(self: *GroupKernels) void {
        self.module.unload();
    }

    /// glinear_kernel<K2, CB, WK> (only the mul1 codebook is built).
    pub fn glinear(self: *const GroupKernels, k2: u32, cb: Codebook, wk: u32) !cuda.Function {
        var buf: [128]u8 = undefined;
        const name = try symbol(&buf, "_ZN15tf_exl3_grouped14glinear_kernelILi{d}ELi{d}ELi{d}EEEvNS_5GArgsE", .{ k2, @intFromEnum(cb), wk });
        return self.module.function(name);
    }
};

/// Rows of an input or output of a grouped launch: the address, the dtype and the elements between rows (row-strided
/// views of one tensor: unit column stride, 16-byte aligned rows).
pub const Rows = struct { ptr: cuda.abi.DevicePtr, dtype: DType, ld: usize };

/// A layer's outputs also rotated into the next layer's input rows (linear_grouped.cu rot_out): rxh[row * ldr + roff +
/// col] = rot128_in(y as stored, rsuh[roff + col]).
pub const RotOut = struct { suh: cuda.abi.DevicePtr, xh: cuda.abi.DevicePtr, ld: usize, off: usize };

/// RoPE folded into the epilogue (linear_grouped.cu rope_out): the flagged layers' bf16 outputs leave with rope_heads
/// applied, cos / sin fp32 [*, rd / 2], positions int64 [M].
pub const RopeOut = struct { cos: cuda.abi.DevicePtr, sin: cuda.abi.DevicePtr, pos: cuda.abi.DevicePtr, hd: usize, rd: usize, layers: [gmax]bool };

/// The fork's `Exl3Group` (cuda/exl3/linear.py): up to `gmax` EXL3 layers of the same rows in one `rot_many` launch
/// (every layer's input rotated with its own suh) and one `glinear` launch a (bits, warps) set. Each layer keeps its
/// own plan (K splits, warps, K ranges, reduction order), so each output has the bits of that layer's `Linear.run`
/// whatever shares the launch. Both launches are programmatic dependent launches (the fork's TF_EXL3_PDL default),
/// and the split-K partials leave L2 unwritten once summed (TF_EXL3_L2_DISCARD default; no bit changes).
pub const Group = struct {
    rot_many: cuda.Function, // (the module's: a Group outlives no GroupKernels it was made from)
    layers: [gmax]*const Linear = undefined,
    n: usize = 0,
    /// The launches: each (k2, wk) set's layers, in the order they first appear.
    sets: [gmax]struct { k2: u32, wk: u32, fn_: cuda.Function, idx: [gmax]u8, n: usize } = undefined,
    n_sets: usize = 0,
    pdl: bool = true,
    discard: bool = true,

    pub fn init(kernels: *const GroupKernels, layers: []const *const Linear) !Group {
        if (layers.len < 1 or layers.len > gmax) return error.BadShape;
        var g: Group = .{ .rot_many = kernels.rot_many };
        for (layers, 0..) |l, i| {
            if (l.wk != 4 and l.wk != 8) return error.BadShape;
            for (layers[0..i]) |o| if (o == l) return error.BadShape; // a layer's split-K counters are its own
            g.layers[i] = l;
        }
        g.n = layers.len;
        for (layers, 0..) |l, i| {
            const at = for (g.sets[0..g.n_sets], 0..) |st, j| {
                if (st.k2 == l.k2 and st.wk == l.wk) break j;
            } else blk: {
                g.sets[g.n_sets] = .{ .k2 = l.k2, .wk = l.wk, .fn_ = try kernels.glinear(l.k2, l.cb, l.wk), .idx = undefined, .n = 0 };
                g.n_sets += 1;
                break :blk g.n_sets - 1;
            };
            g.sets[at].idx[g.sets[at].n] = @intCast(i);
            g.sets[at].n += 1;
        }
        return g;
    }

    /// Element offset of layer i's rotated rows in a `buffer` of m rows (each layer's [m, K_i] fp16 in turn).
    pub fn offset(self: *const Group, m: usize, i: usize) usize {
        var o: usize = 0;
        for (self.layers[0..i]) |l| o += m * l.k;
        return o;
    }

    /// Bytes of the rotated-rows buffer for m rows.
    pub fn bufferBytes(self: *const Group, m: usize) usize {
        return self.offset(m, self.n) * 2;
    }

    /// fp32 words of split-K scratch the widest launch needs for m rows (the launches run in turn on one stream).
    pub fn scratchWords(self: *const Group, m: usize) usize {
        var most: usize = 0;
        for (self.sets[0..self.n_sets]) |st| {
            var w: usize = 0;
            for (st.idx[0..st.n]) |i| {
                const l = self.layers[i];
                if (l.sk > 1) w += l.sk * m * l.n;
            }
            most = @max(most, w);
        }
        return most;
    }

    /// Every layer's input rows rotated into `xh` (`bufferBytes(m)` bytes): one rot_many launch.
    pub fn rotate(self: *const Group, xs: []const Rows, m: usize, xh: cuda.abi.DevicePtr, stream: cuda.Stream) !void {
        if (xs.len != self.n or m < 1 or m > 128) return error.BadShape;
        var a: RArgs = .{ .n = @intCast(self.n), .m = @intCast(m) };
        var warps: usize = 0;
        for (self.layers[0..self.n], xs, 0..) |l, x, i| {
            a.l[i] = .{ .x = x.ptr, .suh = l.suh.ptr, .xh = xh + self.offset(m, i) * 2, .ldx = @intCast(x.ld), .x_dtype = @intFromEnum(x.dtype), .k = @intCast(l.k), .first = @intCast(warps) };
            warps += m * (l.k / 128);
        }
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(self.rot_many, .{ .grid = .{ .x = @intCast((warps + 3) / 4) }, .block = .{ .x = 128 }, .pdl = self.pdl }, stream, &args);
    }

    /// The glinear launches on the rotated rows `xh` (`rotate`'s, or written by a kernel that folds the rotation in:
    /// the same bits): outs[i] [m, N_i] = xh_i @ W_i. `rot`: a layer's outputs also rotated into the next layer's
    /// input rows; `rope`: the flagged layers' outputs roped. `z`: `scratchWords(m)` fp32 words.
    pub fn rotated(self: *const Group, m: usize, xh: cuda.abi.DevicePtr, outs: []const Rows, z: cuda.abi.DevicePtr, rot: ?[]const ?RotOut, rope: ?RopeOut, stream: cuda.Stream) !void {
        if (outs.len != self.n or m < 1 or m > 128) return error.BadShape;
        for (self.sets[0..self.n_sets]) |st| {
            var a: GArgs = .{ .n = @intCast(st.n), .m = @intCast(m) };
            var blocks: usize = 0;
            var zoff: usize = 0;
            var any_split = false;
            for (st.idx[0..st.n], 0..) |li, q| {
                const l = self.layers[li];
                const s = l.strides();
                var gl: GLayer = .{
                    .xh = xh + self.offset(m, li) * 2,
                    .t = l.words.ptr,
                    .svh = l.svh.ptr,
                    .y = outs[li].ptr,
                    .counters = l.counters.ptr,
                    .ldx = @intCast(l.k),
                    .ldy = @intCast(outs[li].ld),
                    .stride_k = s.k,
                    .stride_nb = s.nb,
                    .y_dtype = @intFromEnum(outs[li].dtype),
                    .k = @intCast(l.k),
                    .n = @intCast(l.n),
                    .sk = @intCast(l.sk),
                    .first = @intCast(blocks),
                };
                if (l.sk > 1) {
                    gl.z = z + zoff * 4;
                    zoff += l.sk * m * l.n;
                    any_split = true;
                }
                if (rot) |r| if (r[li]) |ro| {
                    gl.rsuh = ro.suh;
                    gl.rxh = ro.xh;
                    gl.ldr = @intCast(ro.ld);
                    gl.roff = @intCast(ro.off);
                };
                if (rope) |rp| if (rp.layers[li]) {
                    gl.rcos = rp.cos;
                    gl.rsin = rp.sin;
                    gl.rpos = rp.pos;
                    gl.rhd = @intCast(rp.hd);
                    gl.rrd = @intCast(rp.rd);
                };
                a.l[q] = gl;
                blocks += (l.n / 128) * l.sk;
            }
            a.discard = @intFromBool(self.discard and any_split);
            var args: cuda.Args = .{};
            args.add(a);
            const smem: u32 = @intCast(st.wk * @min(m, 8) * 128 * @sizeOf(f32));
            try cuda.launch.launch(st.fn_, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = st.wk * 32 }, .shared = smem, .pdl = self.pdl }, stream, &args);
        }
    }
};

// -- L2 prefetch (l2_prefetch.cu: the fork's exl3/prefetch.py) ------------------------------------------------------

/// Ranges a prefetch launch takes at most (l2_prefetch.cu PMAX).
pub const pmax = 24;

/// l2_prefetch.cu `PArgs`.
pub const PArgs = extern struct {
    p: [pmax]u64 = @splat(0),
    n: [pmax]i64 = @splat(0),
    end: [pmax]i64 = @splat(0),
    count: c_int = 0,
    chunk: c_int = 0,
    delay_ns: i64 = 0,
    wave_ns: i64 = 0,
};

/// prefetch.py's pacing: 8 KiB chunks; 3 us before issuing; one block issuing a wave of 128 chunks every
/// 128 x CHUNK / RATE ns (RATE 230 GB/s), so DRAM's queue holds about a wave.
pub const Prefetch = struct {
    module: cuda.Module,
    kernel: cuda.Function,
    chunk: usize = 8 << 10,
    delay_ns: i64 = 3000,
    rate_gbps: f64 = 230,

    pub fn load(d: *const cuda.Driver) !Prefetch {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_l2_prefetch);
        errdefer module.unload();
        return .{ .module = module, .kernel = try module.function("tf_l2_prefetch_kernel") };
    }

    pub fn deinit(self: *Prefetch) void {
        self.module.unload();
    }

    /// prefetch.py `ranges`: the buffers' bytes in order, cut at `budget` bytes in all (each range a multiple of
    /// 16 bytes at a 16-byte aligned start), at most `pmax` ranges.
    pub fn ranges(buffers: []const cuda.DeviceBuffer, budget: usize, out: *PArgs, chunk: usize) void {
        out.* = .{};
        var left: usize = budget;
        var c: i64 = 0;
        for (buffers) |b| {
            if (b.len == 0 or left < 16 or out.count == pmax) continue;
            const n = @min(b.len, left) / 16 * 16;
            if (n == 0 or b.ptr % 16 != 0) continue;
            const i: usize = @intCast(out.count);
            out.p[i] = b.ptr;
            out.n[i] = @intCast(n);
            c += @intCast((n + chunk - 1) / chunk);
            out.end[i] = c;
            out.count += 1;
            left -= n;
        }
    }

    /// One paced prefetch launch of `a`'s ranges on `stream` (nothing when empty).
    pub fn run(self: *const Prefetch, a: *PArgs, stream: cuda.Stream) !void {
        if (a.count == 0 or a.end[@intCast(a.count - 1)] == 0) return;
        a.chunk = @intCast(self.chunk);
        a.delay_ns = self.delay_ns;
        a.wave_ns = if (self.rate_gbps > 0) @intFromFloat(@floor(128.0 * @as(f64, @floatFromInt(self.chunk)) / self.rate_gbps)) else 0;
        var args: cuda.Args = .{};
        args.add(a.*);
        const need: usize = @intCast(@divFloor(a.end[@intCast(a.count - 1)] + 127, 128));
        const grid: u32 = if (a.wave_ns > 0) 1 else @intCast(@min(need, 4));
        try cuda.launch.launch(self.kernel, .{ .grid = .{ .x = grid }, .block = .{ .x = 128 } }, stream, &args);
    }
};

test "the grouped launches' argument structs have linear_grouped.cu's and l2_prefetch.cu's C layout" {
    try std.testing.expectEqual(@as(usize, 176), @sizeOf(GLayer));
    try std.testing.expectEqual(@as(usize, 112), @offsetOf(GLayer, "rsuh"));
    try std.testing.expectEqual(@as(usize, 144), @offsetOf(GLayer, "rcos"));
    try std.testing.expectEqual(@as(usize, 1424), @sizeOf(GArgs));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(RLayer));
    try std.testing.expectEqual(@as(usize, 392), @sizeOf(RArgs));
    try std.testing.expectEqual(@as(usize, 600), @sizeOf(PArgs));
}

test "plan follows linear.py for the DSV4.1 shapes" {
    try std.testing.expectEqual(Plan{ .sk = 8, .wk = 4 }, plan(5120, 1280)); // wq_a
    try std.testing.expectEqual(Plan{ .sk = 8, .wk = 4 }, plan(5120, 512)); // wkv
    try std.testing.expectEqual(Plan{ .sk = 1, .wk = 8 }, plan(1280, 32768)); // wq_b
    try std.testing.expectEqual(Plan{ .sk = 1, .wk = 8 }, plan(5120, 129280)); // head
}

test "strips put each 128-column block's tiles in k order" {
    const kt = 2;
    const nt = 16;
    const w = 1;
    var words: [kt * nt * w]u32 = undefined;
    for (&words, 0..) |*v, i| v.* = @intCast(i);
    const out = try packStrips(std.testing.allocator, &words, kt, nt, w);
    defer std.testing.allocator.free(out);
    // strip 1 (columns 128..255), k tile 1, tile 3 = stored (k=1, nt=11) = 1 * 16 + 11
    try std.testing.expectEqual(@as(u32, 27), out[((1 * kt + 1) * 8 + 3) * w]);
}
