//! Flash Next's MTP head on CUDA (TensorFold 0.6.6 `mtp.py`, `decode.py: absorb / draft`, `exl3.py: requant_rows`):
//! the head reads the main model's final streams and the next tokens, runs its one attention layer on its own cache,
//! the mixer, and a 4-bit group-32 draft head over the draft vocabulary (the main head's rows decoded through the
//! EXL3 linear and quantized as MLX does). Its drafts only choose which tokens a round verifies: the tokens kept are
//! always the main model's, so drafting changes speed, never output.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const Config = @import("config.zig").Config;
const names = @import("weights_map.zig");
const weights = @import("cuda_weights.zig");
const experts = @import("cuda_experts.zig");
const fwd = @import("cuda_forward.zig");
const prompt_mod = @import("cuda_prompt.zig");

const Ptr = cuda.abi.DevicePtr;
const Linear = dsv41.exl3.Linear;

/// qwen4_exp/cuda/__init__.py DEPTH and CONFIDENCE.
pub const depth = 6;
pub const confidence: f64 = 0.7;

/// The draft head's scored ids (0.6.6's draft_vocab.txt, sorted and unique).
pub const draft_vocab = @embedFile("draft_vocab.txt");

/// qmm.py: group size, column tile and the pack's column padding.
const gs = 32;
const bn = 64;

/// A 4-bit group-32 matrix in qmm.py's "frag" layout: words [npad/64][k/32][8][32], bf16 scales and biases [k/32, npad].
pub const Q4 = struct {
    words: cuda.DeviceBuffer,
    scales: cuda.DeviceBuffer,
    biases: cuda.DeviceBuffer,
    n: usize,
    k: usize,
    npad: usize,

    pub fn deinit(self: *Q4) void {
        self.words.free();
        self.scales.free();
        self.biases.free();
    }
};

/// torch.round: nearest, ties to even.
fn rint(x: f32) f32 {
    const r = @round(x); // ties away from zero
    if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2.0) != 0) return r - std.math.sign(x);
    return r;
}

/// exl3.py `requant_rows` + qmm.py `pack`: rows [n, k] fp32 -> MLX 4-bit groups of 32 (scale bf16((hi - lo) / 15)
/// clamped at 1e-8, bias bf16(lo), q = round((x - bias) / scale) in 0..15) packed for the lane matmul.
pub fn pack(gpa: std.mem.Allocator, d: *const cuda.Driver, rows: []const f32, n: usize, k: usize) !Q4 {
    const kg = k / gs;
    const npad = (n + 127) / 128 * 128;
    const q = try gpa.alloc(u8, npad * k);
    defer gpa.free(q);
    @memset(q, 0);
    const scales = try gpa.alloc(u16, kg * npad);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u16, kg * npad);
    defer gpa.free(biases);
    @memset(scales, 0);
    @memset(biases, 0);
    for (0..n) |c| for (0..kg) |g| {
        const grp = rows[c * k + g * gs ..][0..gs];
        var lo = grp[0];
        var hi = grp[0];
        for (grp[1..]) |x| {
            lo = @min(lo, x);
            hi = @max(hi, x);
        }
        // torch's CUDA division by a Python scalar multiplies by its fp32 reciprocal
        const sbits = weights.f32ToBf16(@max((hi - lo) * @as(f32, 1.0 / 15.0), 1e-8));
        const bbits = weights.f32ToBf16(lo);
        scales[g * npad + c] = sbits;
        biases[g * npad + c] = bbits;
        const sf = weights.bf16ToF32(sbits);
        const bf = weights.bf16ToF32(bbits);
        for (grp, 0..) |x, t| q[c * k + g * gs + t] = @intFromFloat(std.math.clamp(rint((x - bf) / sf), 0, 15));
    };
    // word [T][g][j][lane = r * 4 + c4] = sum over p of q[col][g * 32 + 2 * c4 + OFFSETS[p]] << 4p, col = 64T + 8j + r
    const offsets = [_]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };
    const words = try gpa.alloc(u32, npad / bn * kg * 8 * 32);
    defer gpa.free(words);
    for (0..npad / bn) |tt| for (0..kg) |g| for (0..8) |j| for (0..8) |r| for (0..4) |c4| {
        const col = tt * 64 + j * 8 + r;
        var wd: u32 = 0;
        for (offsets, 0..) |o, p| wd |= @as(u32, q[col * k + g * gs + 2 * c4 + o]) << @intCast(4 * p);
        words[(((tt * kg + g) * 8 + j) * 32) + r * 4 + c4] = wd;
    };
    var wbuf = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(words));
    errdefer wbuf.free();
    var sbuf = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(scales));
    errdefer sbuf.free();
    return .{ .words = wbuf, .scales = sbuf, .biases = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(biases)), .n = n, .k = k, .npad = npad };
}

/// qmm.py `split_k` (group 32): K slices of an (n, k) matrix, the shape's alone.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + bn - 1) / bn;
    const groups = k / gs;
    var sk: usize = 1;
    while (sk < 32 and tiles * sk < 160 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

pub const Kernels = struct {
    module: cuda.Module,
    qmm: [3]cuda.Function, // BM 16, 32, 64; no K slices

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.qwen38fn_qmm);
        errdefer module.unload();
        return .{ .module = module, .qmm = .{
            try module.function("_ZN10tf_qfn_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii"),
            try module.function("_ZN10tf_qfn_qmm10qmm_kernelILi32ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii"),
            try module.function("_ZN10tf_qfn_qmm10qmm_kernelILi32ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii"),
        } };
    }

    pub fn deinit(self: *Kernels) void {
        self.module.unload();
    }

    /// qmm.cu `qmm_cuda` for a matrix without K slices: out [m, n] bf16 = x [m, k] bf16 (row stride ldx) @ q^T, with
    /// xs the rows' 32-group sums.
    pub fn matmul(self: *const Kernels, q: *const Q4, x: Ptr, ldx: usize, xs: Ptr, m: usize, out: Ptr, stream: cuda.Stream) !void {
        if (splitK(q.n, q.k) != 1) return error.SplitKNotPorted;
        const bm: usize = if (m <= 16) 16 else if (m <= 32) 32 else 64;
        const f = self.qmm[if (bm == 16) 0 else if (bm == 32) 1 else 2];
        const stage = bm * gs * 2 + bn * gs / 2 + 2 * bn * 2 + bm * 4; // LaneTile: X + W + 2 S + XS
        const partials = (bm / 16) * (bn / 4 / 8) * 4 * 128 * 4;
        const smem: u32 = @intCast(@max(4 * stage, partials));
        const rows_t = (m + bm - 1) / bm;
        const group = @max(1, @min(rows_t, (12 << 20) / (bm * q.k * 2)));
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(q.words.ptr);
        a.add(q.scales.ptr);
        a.add(q.biases.ptr);
        a.add(out);
        a.add(@as(Ptr, 0));
        for ([_]usize{ m, q.n, q.k, 1, q.npad, if (m == 1) q.k else ldx, group }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows_t * ((q.n + bn - 1) / bn)), .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = smem }, stream, &a);
    }
};

/// A window's MTP scratch (state.py Buffers: mtp_e, mtp_eo, mtp_in, mtp_hn, mtp_xh, mtp_hs).
pub const Scratch = struct {
    e: cuda.DeviceBuffer, // bf16 [rows, D]
    eo: cuda.DeviceBuffer, // bf16 [rows, D]
    in: cuda.DeviceBuffer, // bf16 [rows, S * D]
    hn: cuda.DeviceBuffer, // bf16 [rows, S * D]
    xh: cuda.DeviceBuffer, // fp32 [rows, S * D / 32]
    hs: cuda.DeviceBuffer, // bf16 [rows * S, D]

    pub fn init(d: *const cuda.Driver, cfg: *const Config, rows: usize) !Scratch {
        var s: Scratch = undefined;
        const wide = cfg.wide();
        const sizes = [_]struct { *cuda.DeviceBuffer, usize }{
            .{ &s.e, rows * cfg.hidden * 2 }, .{ &s.eo, rows * cfg.hidden * 2 }, .{ &s.in, rows * wide * 2 },
            .{ &s.hn, rows * wide * 2 },      .{ &s.xh, rows * wide / 32 * 4 },  .{ &s.hs, rows * wide * 2 },
        };
        var made: usize = 0;
        errdefer for (sizes[0..made]) |x| x[0].free();
        for (sizes) |x| {
            x[0].* = try cuda.DeviceBuffer.alloc(d, x[1]);
            made += 1;
        }
        return s;
    }

    pub fn deinit(self: *Scratch) void {
        for ([_]*cuda.DeviceBuffer{ &self.e, &self.eo, &self.in, &self.hn, &self.xh, &self.hs }) |b| b.free();
    }
};

pub const Probe = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, prompt: bool, rows: usize, streams: cuda.DeviceBuffer) anyerror!void,
};

pub const Head = struct {
    gpa: std.mem.Allocator,
    norm_e: cuda.DeviceBuffer, // fp32 [D]
    norm_h: cuda.DeviceBuffer, // fp32 [S * D]
    fc_e: Linear,
    fc_h: Linear,
    layer: fwd.Layer,
    mixer: weights.Hc,
    draft_head: Q4,
    ids: []u32, // the draft head's rows' token ids
    k: Kernels,
    logits: cuda.DeviceBuffer, // bf16 [streams, padded draft vocab]: each segment's last row's draft logits
    host: []u16,
    streams: usize, // the most segments one head call takes
    /// A test's hook after each head call (its streams in the buffers it ran on; decode calls' logits in self.logits).
    probe: ?Probe = null,

    /// The pack's MTP head on the GPU, its draft head requantized from the main head (exl3.py `load`, mtp=True).
    /// `streams`: the most streams one head call drafts for.
    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, streams: usize, stream: cuda.Stream) !*Head {
        const cfg = &m.cfg;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.gpa = gpa;
        var n: names.Name = .{};
        const pack_ = &m.pack;
        h.norm_e = try centred(gpa, d, pack_, names.mtp ++ "pre_fc_norm_embedding.weight", cfg.hidden, m.offset);
        h.norm_h = try centred(gpa, d, pack_, names.mtp ++ "pre_fc_norm_hidden.weight", cfg.wide(), m.offset);
        h.fc_e = try linear(gpa, &m.xk, try pack_.group(names.mtp ++ "fc_embedding"));
        h.fc_h = try linear(gpa, &m.xk, try pack_.group(names.mtp ++ "fc_hidden"));
        const at: names.Where = .{ .mtp = 0 };
        const w = try weights.Attn.load(gpa, d, pack_, at, m.offset);
        h.layer = .{
            .index = cfg.layers,
            .attn_hc = try weights.Hc.load(gpa, d, pack_, cfg, n.of(names.mtp ++ "layers.0.{s}", .{"attn_hyper_connection"}), true, m.offset),
            .mlp_hc = try weights.Hc.load(gpa, d, pack_, cfg, n.of(names.mtp ++ "layers.0.{s}", .{"mlp_hyper_connection"}), true, m.offset),
            .mixer = .{ .attn = .{
                .w = w,
                .parts = .{ try linear(gpa, &m.xk, w.q), try linear(gpa, &m.xk, w.k), try linear(gpa, &m.xk, w.v), try linear(gpa, &m.xk, w.iqk) },
                .o = try linear(gpa, &m.xk, w.o),
                .ai = 0,
            } },
            .router = try weights.router(gpa, d, pack_, cfg, at),
            .experts = try experts.Experts.load(gpa, d, pack_, cfg, at),
        };
        h.mixer = try weights.Hc.load(gpa, d, pack_, cfg, names.mtp ++ "hyper_connection_mixer", false, m.offset);
        // the draft vocabulary, then its rows of the main head
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        var it = std.mem.tokenizeAny(u8, draft_vocab, " \n\r\t");
        while (it.next()) |tok| {
            const v = try std.fmt.parseInt(u32, tok, 10);
            if (v < cfg.vocab) try ids.append(gpa, v);
        }
        std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
        h.ids = try gpa.dupe(u32, ids.items);
        const head = &(m.head orelse return error.NoHead);
        const rows = try requantRows(gpa, d, head, h.ids, stream);
        defer gpa.free(rows);
        h.draft_head = try pack(gpa, d, rows, h.ids.len, head.k);
        h.k = try Kernels.load(d);
        h.streams = streams;
        h.logits = try cuda.DeviceBuffer.alloc(d, streams * h.draft_head.npad * 2);
        h.host = try gpa.alloc(u16, h.ids.len);
        m.has_mtp = true;
        return h;
    }

    pub fn deinit(self: *Head) void {
        self.norm_e.free();
        self.norm_h.free();
        self.fc_e.deinit();
        self.fc_h.deinit();
        self.layer.deinit();
        self.mixer.deinit();
        self.draft_head.deinit();
        self.k.deinit();
        self.logits.free();
        self.gpa.free(self.host);
        self.gpa.free(self.ids);
        self.gpa.destroy(self);
    }

    /// mtp.py `mtp_compute` on a decode window's buffers `mb` over segments `segs` (rows a0 .. a0 + n a stream):
    /// sc.in holds each row's main-model final streams and mb.ids its next token; appends each segment's rows to its
    /// stream's head cache (the caller advances its length); each segment's last row's draft logits into row i of
    /// self.logits, the head's own final streams into mb.streams.
    fn forwardSegs(h: *Head, f: *fwd.Forward, segs: []const fwd.Seg, mb: *fwd.Buffers, sc: *Scratch, stream: cuda.Stream) !void {
        const cfg = &f.m.cfg;
        const R = try fwd.segRows(segs);
        if (segs.len > h.streams or R > mb.rows) return error.WindowTooLarge;
        try h.inputs(f, mb.ids.ptr, &mb.w, sc, sc.in.ptr, R, stream, null, mb);
        try f.layerStep(&h.layer, segs, mb, .none, .mtp, stream);
        try f.finish(&h.mixer, mb, R, f.moePending(mb), stream);
        for (segs, 0..) |sg, i| {
            const last = sg.a0 + sg.n - 1;
            try h.k.matmul(&h.draft_head, mb.w.mixed.ptr + last * cfg.hidden * 2, cfg.hidden, mb.w.xs_mixed.ptr + last * cfg.hidden / 32 * 4, 1, h.logits.ptr + i * h.draft_head.npad * 2, stream);
        }
        if (h.probe) |pr| {
            try stream.synchronize();
            try pr.call(pr.ctx, false, R, mb.streams);
        }
    }

    /// One stream's drafting (decode.py `draft` / multi.py `_draft_all`): its kept rows' final streams (`streams`,
    /// rows of S * D) and the token after each (`next`); up to `count` drafts (0: absorb only) into out / probs, their
    /// number into `n`.
    pub const Request = struct {
        st: *fwd.Stream,
        streams: Ptr,
        next: []const i32,
        count: usize,
        out: []u32,
        probs: []f64,
        n: usize = 0,
    };

    /// multi.py `_draft_all` (greedy): every request's head cache drops the drafted entries of its last chain and
    /// absorbs its kept rows, all streams in one head call; then each chains drafts, all chaining streams one call a
    /// depth: the first draft is always kept, a later one below `confidence` ends the chain, as does a first one below
    /// it. A stream's drafts are those decode.py `draft` makes for it alone.
    pub fn draftAll(h: *Head, f: *fwd.Forward, reqs: []Request, mb: *fwd.Buffers, sc: *Scratch, stream: cuda.Stream) !void {
        const wide = f.m.cfg.wide();
        if (reqs.len > h.streams) return error.WindowTooLarge;
        var segs: [16]fwd.Seg = undefined;
        var ids: [256]i32 = undefined;
        var r: usize = 0;
        for (reqs, 0..) |*q, i| {
            q.n = 0;
            if (q.st.mtp_drafted > 0) {
                try q.st.setMtpLen(q.st.mtp_len - q.st.mtp_drafted, stream);
                q.st.mtp_drafted = 0;
            }
            if (r + q.next.len > mb.rows) return error.WindowTooLarge;
            segs[i] = .{ .st = q.st, .a0 = r, .n = q.next.len };
            @memcpy(ids[r..][0..q.next.len], q.next);
            try sc.in.copyFrom(r * wide * 2, q.streams, q.next.len * wide * 2, stream.handle);
            r += q.next.len;
        }
        if (r == 0) return;
        try mb.ids.upload(0, std.mem.sliceAsBytes(ids[0..r]));
        try h.forwardSegs(f, segs[0..reqs.len], mb, sc, stream);
        for (reqs) |q| try q.st.setMtpLen(q.st.mtp_len + q.next.len, stream);
        // the chaining streams: each request's index, its logits row and its streams' row in mb.streams
        var active: [16]struct { q: usize, row: usize } = undefined;
        var na: usize = 0;
        for (reqs, segs[0..reqs.len], 0..) |q, sg, i| if (q.count > 0) {
            active[na] = .{ .q = i, .row = sg.a0 + sg.n - 1 };
            na += 1;
        };
        // logits row of active[i]: its segment's index in the last head call
        var lrow: [16]usize = undefined;
        for (active[0..na], 0..) |a, i| lrow[i] = a.q;
        var j: usize = 0;
        while (na > 0 and j < depth) : (j += 1) {
            try stream.synchronize();
            var nxt: [16]struct { q: usize, row: usize, tok: u32 } = undefined;
            var nn: usize = 0;
            for (active[0..na], lrow[0..na]) |a, lr| {
                const q = &reqs[a.q];
                const sd = try h.sampleDraft(lr);
                const low = sd.prob < confidence;
                if (low and j > 0) continue;
                q.out[q.n] = sd.token;
                q.probs[q.n] = sd.prob;
                q.n += 1;
                if (!low and j + 1 < q.count) {
                    nxt[nn] = .{ .q = a.q, .row = a.row, .tok = sd.token };
                    nn += 1;
                }
            }
            if (nn == 0) break;
            for (nxt[0..nn], 0..) |x, i| {
                segs[i] = .{ .st = reqs[x.q].st, .a0 = i, .n = 1 };
                ids[i] = @intCast(x.tok);
                try sc.in.copyFrom(i * wide * 2, mb.streams.ptr + x.row * wide * 2, wide * 2, stream.handle);
            }
            try mb.ids.upload(0, std.mem.sliceAsBytes(ids[0..nn]));
            try h.forwardSegs(f, segs[0..nn], mb, sc, stream);
            for (nxt[0..nn], 0..) |x, i| {
                const st = reqs[x.q].st;
                try st.setMtpLen(st.mtp_len + 1, stream);
                st.mtp_drafted += 1;
                active[i] = .{ .q = x.q, .row = i };
                lrow[i] = i;
            }
            na = nn;
        }
    }

    /// decode.py `absorb`: the head's cache drops the drafted entries of its last chain, then takes the kept rows
    /// (the main model's final streams and the token after each); the draft logits of the last row.
    pub fn absorb(h: *Head, f: *fwd.Forward, st: *fwd.Stream, mb: *fwd.Buffers, sc: *Scratch, streams_in: Ptr, next: []const i32, stream: cuda.Stream) !void {
        var reqs = [_]Request{.{ .st = st, .streams = streams_in, .next = next, .count = 0, .out = &.{}, .probs = &.{} }};
        try h.draftAll(f, &reqs, mb, sc, stream);
    }

    /// decode.py `draft` (greedy) for one stream: absorb, then chain up to `count` drafts (draftAll's rule); the
    /// drafts into `out` (their head probabilities into `probs`), their count returned.
    pub fn draft(h: *Head, f: *fwd.Forward, st: *fwd.Stream, mb: *fwd.Buffers, sc: *Scratch, streams_in: Ptr, next: []const i32, count: usize, out: []u32, probs: []f64, stream: cuda.Stream) !usize {
        var reqs = [_]Request{.{ .st = st, .streams = streams_in, .next = next, .count = count, .out = out, .probs = probs }};
        try h.draftAll(f, &reqs, mb, sc, stream);
        return reqs[0].n;
    }

    /// The MTP head on a prompt chunk's buffers (mtp.py `mtp_compute` with prefill buffers): absorbs the chunk's rows
    /// with their next tokens; no draft logits are read.
    pub fn forwardPrompt(h: *Head, p: *prompt_mod.Prompt, st: *fwd.Stream, pb: *prompt_mod.Buffers, sc: *Scratch, next: []const i32, streams_in: Ptr, stream: cuda.Stream) !void {
        const n = next.len;
        try pb.ids.upload(0, std.mem.sliceAsBytes(next));
        try h.inputs(p.f, pb.ids.ptr, &pb.w, sc, streams_in, n, stream, .{ .p = p, .pb = pb }, null);
        try p.layerStep(&h.layer, st, pb, n, .none, st.mtpAt(), stream);
        try p.finish(&h.mixer, pb, n, p.moePending(pb), stream);
        if (h.probe) |pr| {
            try stream.synchronize();
            try pr.call(pr.ctx, true, n, pb.streams);
        }
    }

    const PromptCtx = struct { p: *prompt_mod.Prompt, pb: *prompt_mod.Buffers };

    /// The head's input rows into w.h: embedding branch rmsnorm -> fc_embedding, the streams' rmsnorm -> fc_hidden on
    /// each stream, added (mtp.py `mtp_compute`'s first lines).
    fn inputs(h: *Head, f: *fwd.Forward, ids: Ptr, w: *@import("cuda_hc.zig").Window, sc: *Scratch, streams_in: Ptr, n: usize, stream: cuda.Stream, pc: ?PromptCtx, mb: ?*fwd.Buffers) !void {
        const cfg = &f.m.cfg;
        const g = f.g;
        const wide = cfg.wide();
        try g.embedCopies(ids, f.m.embed.ptr, sc.e.ptr, n, 1, stream);
        try g.rmsnorm(sc.e.ptr, cfg.hidden, h.norm_e.ptr, w.mixed.ptr, w.xs_mixed.ptr, cfg.eps, n, cfg.hidden, stream);
        if (streams_in != sc.in.ptr) try sc.in.copyFrom(0, streams_in, n * wide * 2, stream.handle);
        try g.rmsnorm(sc.in.ptr, wide, h.norm_h.ptr, sc.hn.ptr, sc.xh.ptr, cfg.eps, n, wide, stream);
        if (pc) |c| {
            try c.p.matmul(&h.fc_e, w.mixed.ptr, n, sc.eo.ptr, cfg.hidden, c.pb, stream);
            try c.p.matmul(&h.fc_h, sc.hn.ptr, n * cfg.streams, sc.hs.ptr, cfg.hidden, c.pb, stream);
        } else {
            try f.mm(&h.fc_e, w.mixed.ptr, n, sc.eo.ptr, mb.?, stream);
            try f.mm(&h.fc_h, sc.hn.ptr, n * cfg.streams, sc.hs.ptr, mb.?, stream);
        }
        try g.addStreams(sc.eo.ptr, sc.hs.ptr, w.h.ptr, n, stream);
    }

    /// decode.py `Engine.sample_draft`, greedy: the draft head's first largest logit, its token, and its probability
    /// exp(top - logsumexp) (the chain's confidence; computed on the host, so it may differ from torch's in the last
    /// bits, which can only move a chain's cut, never a kept token).
    pub fn sampleDraft(h: *Head, row: usize) !struct { token: u32, prob: f64 } {
        try h.logits.download(row * h.draft_head.npad * 2, std.mem.sliceAsBytes(h.host));
        var best: usize = 0;
        var top = weights.bf16ToF32(h.host[0]);
        for (h.host[1..], 1..) |x, i| {
            const v = weights.bf16ToF32(x);
            if (v > top) {
                top = v;
                best = i;
            }
        }
        var sum: f64 = 0;
        for (h.host) |x| sum += @exp(@as(f64, weights.bf16ToF32(x)) - top);
        return .{ .token = h.ids[best], .prob = 1.0 / sum };
    }
};

fn centred(gpa: std.mem.Allocator, d: *const cuda.Driver, pack_: *weights.Pack, name: []const u8, len: usize, offset: f32) !cuda.DeviceBuffer {
    const t = try pack_.get(name);
    if (t.numel() != len) return error.UnexpectedTensor;
    const v = try gpa.alloc(f32, len);
    defer gpa.free(v);
    for (v, 0..) |*x, j| x.* = @as(f32, @floatCast(try weights.element(t, j))) + offset;
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(v));
}

fn linear(gpa: std.mem.Allocator, k: *const dsv41.exl3.Kernels, g: weights.Group) !Linear {
    const words = try gpa.alignedAlloc(u8, .@"4", g.trellis.len);
    defer gpa.free(words);
    @memcpy(words, g.trellis);
    return Linear.init(gpa, k, words, g.kt, g.nt, g.w16, g.suh, g.svh, @enumFromInt(@intFromEnum(g.codebook)));
}

/// exl3.py `requant_rows`' first half: the head's rows `ids` [n, k] fp32, decoded through its own EXL3 linear on
/// identity blocks of 128 rows (rows[:, k0:k0 + 128] = head(eye)[:, ids]^T).
fn requantRows(gpa: std.mem.Allocator, d: *const cuda.Driver, head: *const Linear, ids: []const u32, stream: cuda.Stream) ![]f32 {
    const k = head.k;
    const n = head.n;
    const blk: usize = 128;
    const rows = try gpa.alloc(f32, ids.len * k);
    errdefer gpa.free(rows);
    var x = try cuda.DeviceBuffer.alloc(d, blk * k * 2);
    defer x.free();
    var xh = try cuda.DeviceBuffer.alloc(d, blk * k * 2);
    defer xh.free();
    var z = try cuda.DeviceBuffer.alloc(d, @max(16, head.scratchWords(blk) * 4));
    defer z.free();
    var out = try cuda.DeviceBuffer.alloc(d, blk * n * 4);
    defer out.free();
    const eye = try gpa.alloc(u16, blk * k);
    defer gpa.free(eye);
    const host = try gpa.alloc(f32, blk * n);
    defer gpa.free(host);
    var k0: usize = 0;
    while (k0 < k) : (k0 += blk) {
        @memset(eye, 0);
        for (0..blk) |i| eye[i * k + k0 + i] = 0x3F80; // bf16 1.0
        try x.upload(0, std.mem.sliceAsBytes(eye));
        try head.run(x.ptr, .bf16, blk, xh.ptr, z.ptr, out.ptr, .f32, stream);
        try stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(host));
        for (ids, 0..) |id, r| for (0..blk) |i| {
            rows[r * k + k0 + i] = host[i * n + id];
        };
    }
    return rows;
}

test "torch.round ties to even" {
    try std.testing.expectEqual(@as(f32, 2), rint(2.5));
    try std.testing.expectEqual(@as(f32, 4), rint(3.5));
    try std.testing.expectEqual(@as(f32, -2), rint(-2.5));
    try std.testing.expectEqual(@as(f32, 3), rint(2.6));
    try std.testing.expectEqual(@as(f32, 0), rint(0.5));
}

test "the draft head takes no K slices at the pack's shape" {
    try std.testing.expectEqual(@as(usize, 1), splitK(79591, 2560));
}

/// What decides between two prompt chunks whether the pass goes on (a cancelled request stops it).
pub const Cancel = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque) bool,
};

/// decode.py `prefill` (chunk loop, no kept points): the prompt in chunks of the prompt buffers' rows; after each
/// chunk's forward its last row's streams into `last` (bf16 [S * D], the head's first absorb reads them) and, with a
/// head, the head absorbing the chunk's rows that have a next token in the prompt; then the commit. The last chunk's
/// logits end in pb.logits.
pub fn prefill(h: ?*Head, p: *prompt_mod.Prompt, st: *fwd.Stream, pb: *prompt_mod.Buffers, scp: ?*Scratch, tokens: []const i32, last: cuda.DeviceBuffer, cancel: ?Cancel, stream: cuda.Stream) !void {
    const wide = p.f.m.cfg.wide();
    var start: usize = 0;
    while (start < tokens.len) {
        if (cancel) |c| if (c.check(c.ctx)) return error.Cancelled;
        const end = @min(start + pb.rows, tokens.len);
        const R = end - start;
        try p.stage(st, pb, tokens[start..end], stream);
        try p.compute(st, pb, R, end == tokens.len, .{}, stream);
        try last.copyFrom(0, pb.streams.ptr + (R - 1) * wide * 2, wide * 2, stream.handle);
        if (h) |head| {
            const next = tokens[start + 1 .. @min(end + 1, tokens.len)];
            if (next.len > 0) {
                try head.forwardPrompt(p, st, pb, scp.?, next, pb.streams.ptr, stream);
                try st.setMtpLen(st.mtp_len + next.len, stream);
            }
        }
        try p.commit(st, pb, R, stream);
        start = end;
    }
}
