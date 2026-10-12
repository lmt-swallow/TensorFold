//! One GLM-5.3-Flash request on one rank, decoded as the Mia fork serves it in its strict setting with MTP drafts
//! (`decode.prefill`, then drafter_choice.py `auto_decode` without DFlash2, greedy): the prompt in 2,048-row chunks,
//! each chunk's rows absorbed into the MTP head; then rounds, each verifying the pending token and its drafts (copy
//! drafts when the context repeats itself, else an MTP chain of up to 3, cut where the product of the drafts'
//! confidences falls under 0.35), keeping the samples up to the first that differs from its draft. Every rank runs the
//! same calls in the same order; their all-gathers and samples are the only exchanges, so the ranks agree on every
//! token without a broadcast. Drafts never change the tokens, only how many a round lands.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const glue_mod = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const weights = @import("cuda_weights.zig");
const A = @import("cuda_dsa.zig");
const Mo = @import("cuda_moe.zig");
const M = @import("cuda_model.zig");
const T = @import("cuda_mtp.zig");
const copy_drafts = @import("copy_drafts.zig");
const P = @import("cuda_pool.zig");
const Tr = @import("cuda_tier.zig");
const RP = @import("round_profile.zig");
const Pm = @import("cuda_prompt.zig");

const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

/// exact_sampling.py MARGIN and decode.py's draft candidates (20 and the margin from each rank).
const margin = 8;
pub const draft_k = 20 + margin;
/// drafter_choice.py: backlog rows the MTP head may owe before it absorbs them.
const cap = 256;
pub const rows = 16; // TF_GLM_MAX_ROWS: the widest verify window
/// Rows of every stream's windows together in one shared forward: the widest window the captured kernel set serves
/// (16 rows; multi.py MAX_WINDOW is 32, below the MoE's 64-row prompt path, which needs the 32-row variants captured).
pub const batch_rows = 16;
pub const prompt_rows = 2048; // decode.PREFILL_ROWS

/// How a rank reaches the other: the all-gather of fp32 partials and of the samples' candidates.
pub const Link = struct {
    ctx: *anyopaque,
    gather: *const fn (ctx: *anyopaque, send: Ptr, recv: Ptr, floats: usize, stream: cuda.Stream) anyerror!void,
};

/// One stream's sequence: the backbone's state, the MTP head's cache and position, the last window's final-normed
/// rows, and the rows the head owes (their hidden rows and the token after each).
pub const Seq = struct {
    gpa: std.mem.Allocator,
    st: M.State,
    mcache: A.Cache,
    mpos: Buf,
    backlog: Buf, // [cap, D]: the MTP head's owed input rows
    last: Buf, // [rows, D]: the last window's (or prompt chunk's last) final-normed rows
    m_next: std.ArrayList(u32) = .empty, // the owed rows' next tokens
    mtp_len: usize = 0,
    mtp_drafted: usize = 0,
    fresh: bool = false, // the head's logits are its last absorbed row's (`absorb` ran a step; a chain may start)
    /// The latent rows on a tier (cuda_tier.zig): the backbone's DSA layers as tier stream `h`, the head's as `hm`.
    tier: ?SeqTier = null,

    pub const SeqTier = struct { t: *Tr.Tier, h: usize, hm: usize };

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, layers: usize, capacity: usize, kv: A.Kv) !Seq {
        const c = lay.cfg;
        const D = c.hidden;
        var st = try M.State.init(gpa, d, lay, layers, capacity, rows, prompt_rows, kv);
        errdefer st.deinit();
        var mcache = try A.Cache.init(d, c, capacity, A.indexRing(capacity, prompt_rows), kv);
        errdefer mcache.deinit();
        var mpos = try Buf.alloc(d, 16);
        errdefer mpos.free();
        var backlog = try Buf.alloc(d, cap * D * 2);
        errdefer backlog.free();
        return .{ .gpa = gpa, .st = st, .mcache = mcache, .mpos = mpos, .backlog = backlog, .last = try Buf.alloc(d, rows * D * 2) };
    }

    /// A stream slot's sequence whose DSA caches (the backbone's and the head's) hold their rings of `ring` rows only;
    /// `bind` points them at a pool's extent, `reset` starts a sequence.
    pub fn initSlot(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, layers: usize, ring: usize, kv: A.Kv) !Seq {
        const D = lay.cfg.hidden;
        var st = try M.State.initSlot(gpa, d, lay, layers, ring, rows, kv);
        errdefer st.deinit();
        var mcache = try A.Cache.rings(d, lay.cfg, ring, kv);
        errdefer mcache.deinit();
        var mpos = try Buf.alloc(d, 16);
        errdefer mpos.free();
        var backlog = try Buf.alloc(d, cap * D * 2);
        errdefer backlog.free();
        return .{ .gpa = gpa, .st = st, .mcache = mcache, .mpos = mpos, .backlog = backlog, .last = try Buf.alloc(d, rows * D * 2) };
    }

    /// The caches on the pool's extent [base, base + size): plane i the i-th DSA layer's, the head's the last.
    pub fn bind(q: *Seq, pool: *const P.Pool, base: usize, size: usize) !void {
        try q.st.bind(pool, base, size);
        try pool.bind(q.st.caches.len, &q.mcache, base, size);
        for (q.st.caches, 0..) |*c, i| c.tier = if (q.tier) |x| .{ .t = x.t, .h = x.h, .layer = i } else null;
        q.mcache.tier = if (q.tier) |x| .{ .t = x.t, .h = x.hm, .layer = 0 } else null;
    }

    /// The positions the stream may reach (its context), which its selections are sized by, as one cache of that
    /// capacity sizes them, whatever its extent.
    pub fn reach(q: *Seq, most: usize) void {
        for (q.st.caches) |*c| c.most = most;
        q.mcache.most = most;
    }

    /// A new sequence from position 0 (the head's too).
    pub fn reset(q: *Seq, stream: cuda.Stream) !void {
        try q.st.reset(stream);
        q.m_next.clearRetainingCapacity();
        q.mtp_len = 0;
        q.mtp_drafted = 0;
        q.fresh = false;
    }

    pub fn deinit(q: *Seq) void {
        q.st.deinit();
        q.mcache.deinit();
        q.mpos.free();
        q.backlog.free();
        q.last.free();
        q.m_next.deinit(q.gpa);
    }

    /// The positions the sequence holds (committed rows).
    pub fn len(q: *const Seq) usize {
        return q.st.pos;
    }
};

/// The model's calls on one rank, shared by every sequence: the window and prompt buffers, the head's buffers, the
/// samples' host row and candidates. Every rank makes the same calls in the same order (the all-gathers inside pair).
pub const Runner = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lay: config.Layout,
    k: M.Kernels,
    tk: *const T.Kernels,
    model: *const M.Model,
    mtp: *const T.Mtp,
    link: Link,
    stream: cuda.Stream,
    w: M.Work,
    mb: T.Buffers,
    pmb: T.Buffers,
    host: []u16, // a logits row
    cand: Buf,
    players: []Pm.Layer, // the prompt path's view of the model (`pmodel`)
    pmodel: Pm.Model,
    stage_rows: Buf, // the prompt path's gather staging (prompt.Prompt)
    argmax_mod: cuda.Module,
    argmax_fn: cuda.Function, // tf_argmax_rows_i32_kernel (torch_ops/argmax.cu): each row's first largest
    picks: Buf, // [batch_rows] i32 ids, then the rows' (value, id) pairs sent and every rank's received
    topk_mod: cuda.Module,
    topk_fn: cuda.Function, // tf_glm53_topk_row (glm53/topk.cu): a row's k best on the device
    /// TF_GLM_DRAFT_TOPK=1: `sample` takes the row's k best on the device (default: the host scan, faster until the
    /// device's runs on more than one block); TF_GLM_TOPK_CHECK=1: both, compared every call.
    topk: enum { device, host, check } = .host,
    /// TF_GLM_ROUND_PROFILE=1 on rank 0: the first rounds' phases logged (round_profile.zig).
    prof: ?RP.Profile = null,
    split: ?Pm.Split = null, // TF_GLM_HC_SPLIT=1: prompt chunks row-split between the ranks
    overlap: ?*Pm.Overlap = null, // TF_GLM_PREFILL_OVERLAP=1|2: their gathers on a second stream

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, k: M.Kernels, tk: *const T.Kernels, model: *const M.Model, mtp: *const T.Mtp, link: Link, stream: cuda.Stream, capacity: usize) !Runner {
        var r: Runner = .{ .gpa = gpa, .d = d, .lay = lay, .k = k, .tk = tk, .model = model, .mtp = mtp, .link = link, .stream = stream, .w = undefined, .mb = undefined, .pmb = undefined, .host = undefined, .cand = undefined, .players = undefined, .pmodel = undefined, .stage_rows = undefined, .argmax_mod = undefined, .argmax_fn = undefined, .picks = undefined, .topk_mod = undefined, .topk_fn = undefined };
        r.w = try M.Work.init(d, lay, batch_rows, prompt_rows, capacity);
        errdefer r.w.deinit();
        r.mb = try T.Buffers.init(d, lay, rows, false);
        errdefer r.mb.deinit();
        // the MTP absorb of a prompt chunk runs after the chunk's backbone: it reuses the chunk's partials, gathers
        // and split-K slices (0.34 GiB a rank at 2,048 rows)
        r.pmb = try T.Buffers.initSharing(d, lay, prompt_rows, true, .{ .part = r.w.pb.part, .gath = r.w.pb.gath, .sk = r.w.pb.sk });
        errdefer r.pmb.deinit();
        r.host = try gpa.alloc(u16, lay.size(.vocab));
        errdefer gpa.free(r.host);
        r.cand = try Buf.alloc(d, 6 * draft_k * 4);
        errdefer r.cand.free();
        r.stage_rows = try Buf.alloc(d, (prompt_rows + 2) * lay.cfg.hidden * 2);
        errdefer r.stage_rows.free();
        r.argmax_mod = try cuda.Module.load(d, cuda.kernels.torch_argmax);
        errdefer r.argmax_mod.unload();
        r.argmax_fn = try r.argmax_mod.function("tf_argmax_rows_i32_kernel");
        r.picks = try Buf.alloc(d, batch_rows * 4 + (1 + lay.world) * 2 * batch_rows * 4);
        errdefer r.picks.free();
        r.topk_mod = try cuda.Module.load(d, cuda.kernels.glm53_topk);
        errdefer r.topk_mod.unload();
        r.topk_fn = try r.topk_mod.function("tf_glm53_topk_row");
        if (std.c.getenv("TF_GLM_DRAFT_TOPK")) |v| if (std.mem.eql(u8, std.mem.span(v), "1")) {
            r.topk = .device;
        };
        if (std.c.getenv("TF_GLM_TOPK_CHECK")) |v| if (std.mem.eql(u8, std.mem.span(v), "1")) {
            r.topk = .check;
        };
        if (lay.rank == 0) r.prof = try RP.Profile.fromEnv(d);
        r.players = try gpa.alloc(Pm.Layer, model.layers.len);
        errdefer gpa.free(r.players);
        r.pmodel = Pm.of(model, lay.cfg, mtp, r.players);
        try r.splitSettings(gpa, d);
        return r;
    }

    pub fn deinit(r: *Runner) void {
        if (r.prof) |*p| p.deinit();
        r.picks.free();
        r.argmax_mod.unload();
        r.topk_mod.unload();
        r.w.deinit();
        r.mb.deinit();
        r.pmb.deinit();
        r.gpa.free(r.host);
        r.cand.free();
        r.gpa.free(r.players);
        r.stage_rows.free();
        if (r.overlap) |o| {
            o.deinit();
            r.gpa.destroy(o);
        }
    }

    /// hcsplit.py `SplitSettings.from_env` of two ranks (send/receive exchanges as all-gathers): TF_GLM_HC_SPLIT=1
    /// splits prompt chunks of TF_GLM_HC_SPLIT_MIN_ROWS rows (default 64) or more; TF_GLM_PREFILL_OVERLAP=1 runs their
    /// gathers on a second stream in TF_GLM_OVERLAP_PIECES row pieces (default 2), =2 also the next block's front on
    /// this rank's rows meanwhile. Every rank must be started alike (the bits do not depend on these; the calls do).
    fn splitSettings(r: *Runner, gpa: std.mem.Allocator, d: *const cuda.Driver) !void {
        const env = struct {
            fn get(name: [:0]const u8, default: usize) !usize {
                const v = std.mem.span(std.c.getenv(name) orelse return default);
                return if (v.len == 0) default else std.fmt.parseInt(usize, v, 10);
            }
        };
        if (try env.get("TF_GLM_HC_SPLIT", 0) != 1) return;
        const level = try env.get("TF_GLM_PREFILL_OVERLAP", 0);
        if (level > 2) return error.BadPrefillOverlap;
        const pieces = try env.get("TF_GLM_OVERLAP_PIECES", 2);
        if (pieces < 1 or pieces > 8) return error.BadOverlapPieces;
        r.split = .{ .pieces = if (level > 0) pieces else 1, .min_rows = try env.get("TF_GLM_HC_SPLIT_MIN_ROWS", 64), .fronts = level == 2 };
        if (level > 0) {
            const o = try gpa.create(Pm.Overlap);
            errdefer gpa.destroy(o);
            o.* = try Pm.Overlap.init(d);
            r.overlap = o;
        }
    }

    fn gatherFn(p: *anyopaque, part: Ptr, gath: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const r: *Runner = @ptrCast(@alignCast(p));
        try r.link.gather(r.link.ctx, part, gath, floats, stream);
    }

    fn gather(r: *Runner) M.Gather {
        return .{ .ctx = r, .run = gatherFn };
    }

    fn promptGatherFn(p: *anyopaque, send: Ptr, recv: Ptr, bytes: usize, stream: cuda.Stream) anyerror!void {
        const r: *Runner = @ptrCast(@alignCast(p));
        try r.link.gather(r.link.ctx, send, recv, bytes / 4, stream);
    }

    fn mtpGather(r: *Runner) T.Gather {
        return .{ .ctx = r, .run = gatherFn };
    }

    const Cand = struct { v: f32, id: u32 };

    fn better(_: void, a: Cand, b: Cand) bool {
        return a.v > b.v or (a.v == b.v and a.id < b.id);
    }

    /// The host scan of row `row` (this rank's vocabulary share): its k best as [values][ids] words.
    fn hostBest(r: *Runner, logits: Buf, row: usize, k: usize) ![2 * draft_k]f32 {
        const nv = r.host.len;
        try r.stream.synchronize();
        try logits.download(row * nv * 2, std.mem.sliceAsBytes(r.host));
        var best: [draft_k]Cand = undefined;
        var n: usize = 0;
        for (r.host, 0..) |x, j| {
            const cand: Cand = .{ .v = @bitCast(@as(u32, x) << 16), .id = @intCast(r.lay.rank * nv + j) };
            if (n < k) {
                best[n] = cand;
                n += 1;
                std.mem.sort(Cand, best[0..n], {}, better);
            } else if (better({}, cand, best[k - 1])) {
                best[k - 1] = cand;
                std.mem.sort(Cand, best[0..k], {}, better);
            }
        }
        var mine: [2 * draft_k]f32 = undefined;
        for (best[0..k], 0..) |b, i| {
            mine[i] = b.v;
            mine[k + i] = @bitCast(b.id);
        }
        return mine;
    }

    /// decode.py `sample_rows`, greedy, of row `row` of `logits` (this rank's vocabulary share): every rank's `k`
    /// best (value, id) gathered, the best by value then the lower id; with `prob`, its probability among the
    /// gathered candidates at temperature 1 (`_probability`, a draft's confidence).
    pub fn sample(r: *Runner, logits: Buf, row: usize, k: usize, prob: ?*f64) !u32 {
        const nv = r.host.len;
        if (r.topk != .host) {
            // the rank's k best on the device, straight into the gather's send words
            var a: cuda.Args = .{};
            a.add(logits.ptr + row * nv * 2);
            a.add(@as(i32, @intCast(nv)));
            a.add(@as(i32, @intCast(k)));
            a.add(@as(i32, @intCast(r.lay.rank * nv)));
            a.add(r.cand.ptr);
            try cuda.launch.launch(r.topk_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, r.stream, &a);
        }
        if (r.topk != .device) {
            const mine = try r.hostBest(logits, row, k);
            if (r.topk == .check) {
                var dev: [2 * draft_k]f32 = undefined;
                try r.stream.synchronize();
                try r.cand.download(0, std.mem.sliceAsBytes(dev[0 .. 2 * k]));
                if (!std.mem.eql(u32, @ptrCast(dev[0 .. 2 * k]), @ptrCast(mine[0 .. 2 * k]))) {
                    std.log.err("glm53: the device top-{d} differs from the host scan (row {d})", .{ k, row });
                    return error.TopkMismatch;
                }
            } else try r.cand.uploadAsync(0, std.mem.sliceAsBytes(mine[0 .. 2 * k]), r.stream.handle);
        }
        try r.link.gather(r.link.ctx, r.cand.ptr, r.cand.ptr + 2 * draft_k * 4, 2 * k, r.stream);
        try r.stream.synchronize();
        var both: [4 * draft_k]f32 = undefined;
        try r.cand.download(2 * draft_k * 4, std.mem.sliceAsBytes(both[0 .. 4 * k]));
        var all: [2 * draft_k]Cand = undefined;
        for (0..2) |rk| for (0..k) |i| {
            all[rk * k + i] = .{ .v = both[rk * 2 * k + i], .id = @bitCast(both[rk * 2 * k + k + i]) };
        };
        std.mem.sort(Cand, all[0 .. 2 * k], {}, better);
        if (prob) |p| {
            var z: f64 = 0;
            const top: f64 = all[0].v;
            for (all[0 .. 2 * k]) |a| z += @exp(@as(f64, a.v) - top);
            p.* = 1.0 / z;
        }
        return all[0].id;
    }

    /// `sample` with k = 1 of rows a0 .. a0 + n of `logits` at once into `out`, the same token for every row whose
    /// logits hold no NaN (the kernel takes a NaN as the largest, `sample` passes it over): each rank's first largest
    /// value a row on the device (the lowest id of equals, -0 equal to +0), every rank's (value, id) a row in one
    /// gather, the larger value, then the lower id.
    pub fn greedyRows(r: *Runner, logits: Buf, a0: usize, n: usize, out: []u32) !void {
        if (n == 0) return;
        if (n > batch_rows) return error.WindowTooWide;
        const nv = r.host.len;
        var a: cuda.Args = .{};
        a.add(logits.ptr + a0 * nv * 2);
        a.add(r.picks.ptr);
        for ([_]usize{ n, nv, nv }) |x| a.add(@as(i64, @intCast(x)));
        a.add(@as(i32, 0)); // bf16
        try cuda.launch.launch(r.argmax_fn, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 } }, r.stream, &a);
        var ids: [batch_rows]i32 = undefined;
        try r.stream.synchronize();
        try r.picks.download(0, std.mem.sliceAsBytes(ids[0..n]));
        var mine: [2 * batch_rows]f32 = undefined;
        for (ids[0..n], 0..) |id, row| {
            if (id < 0) return error.BadLogits;
            var v: u16 = undefined;
            try logits.download(((a0 + row) * nv + @as(usize, @intCast(id))) * 2, std.mem.asBytes(&v));
            mine[row] = @bitCast(@as(u32, v) << 16);
            mine[n + row] = @bitCast(@as(u32, @intCast(r.lay.rank * nv + @as(usize, @intCast(id)))));
        }
        const send = batch_rows * 4;
        const recv = send + 2 * batch_rows * 4;
        try r.picks.uploadAsync(send, std.mem.sliceAsBytes(mine[0 .. 2 * n]), r.stream.handle);
        try r.link.gather(r.link.ctx, r.picks.ptr + send, r.picks.ptr + recv, 2 * n, r.stream);
        try r.stream.synchronize();
        var every: [config.world * 2 * batch_rows]f32 = undefined;
        const world = r.lay.world;
        try r.picks.download(recv, std.mem.sliceAsBytes(every[0 .. world * 2 * n]));
        for (out[0..n], 0..) |*o, row| {
            var best: Cand = .{ .v = every[row], .id = @bitCast(every[n + row]) };
            for (1..world) |rk| {
                const c: Cand = .{ .v = every[rk * 2 * n + row], .id = @bitCast(every[rk * 2 * n + n + row]) };
                if (better({}, c, best)) best = c;
            }
            o.* = best.id;
        }
    }

    /// Before a window of `n` rows at `pos` of the sequence's tier stream (`head`: the head's): its earlier calls
    /// finished on the device, then cuda_tier `beginWindow`. Nothing without a tier.
    fn tierBegin(r: *Runner, q: *const Seq, head: bool, pos: usize, n: usize) !void {
        const x = q.tier orelse return;
        try r.stream.synchronize();
        try x.t.beginWindow(if (head) x.hm else x.h, pos, n);
    }

    /// The tier stream's positions before `pos` final, the ones from it dropped (rejected rows).
    fn tierSettle(r: *Runner, q: *const Seq, head: bool, pos: usize) !void {
        const x = q.tier orelse return;
        const h = if (head) x.hm else x.h;
        x.t.rewind(h, pos);
        try x.t.commit(h, pos, r.stream);
    }

    fn mtpStep(r: *Runner, q: *Seq, b: *const T.Buffers, next: []const u32, hidden: Ptr) !void {
        if (!b.prefill) try r.tierBegin(q, true, q.mtp_len, next.len);
        const pre = b.prefill;
        try T.run(r.k.g, r.tk, r.k.mk, r.lay, r.mtp, b, .{ .cache = &q.mcache, .ds = if (pre) &r.w.pds else &r.w.ds, .ms = if (pre) &r.w.pms else &r.w.ms, .pos_dev = q.mpos }, r.model.embed.ptr, next, hidden, q.mtp_len, r.mtpGather(), r.stream);
    }

    /// decode.py `prefill`'s chunk of `prompt` from `start` (the sequence holds the rows before it): its rows through
    /// the backbone and into the head (the chunk's last row waits for the token after it); returns the next chunk's
    /// start. Once it is the prompt's length, `first` draws the reply's first token.
    pub fn fill(r: *Runner, q: *Seq, prompt: []const u32, start: usize) !usize {
        const D = r.lay.cfg.hidden;
        if (start != q.st.pos) return error.PositionMismatch;
        const R = @min(prompt_rows, prompt.len - start);
        var ps: Pm.Stream = .{
            .kda = &q.st.kda,           .caches = q.st.caches,      .mtp_cache = &q.mcache,  .pos = q.st.pos,
            .mtp_len = q.mtp_len,       .capacity = q.st.caches[0].capacity,
            .pos_dev = q.st.pos_dev.ptr, .mtp_pos_dev = q.mpos.ptr,
        };
        const w = &r.w;
        const c: Pm.Chunk = .{
            .p = .{
                .rows = prompt_rows, .b = &w.pb,  .temp = &w.temp,         .kscratch = &w.pscratch, .ds = &w.pds,
                .ms = &w.pms,        .mb = &r.pmb, .fnormed = w.fnormed.ptr, .fxs = w.fxs.ptr,         .logits = w.logits.ptr,
                .stage_rows = r.stage_rows.ptr,
            },
            .m = &r.pmodel,
            .k = .{ .g = r.k.g, .kda = r.k.kk, .moe = r.k.mk, .mtp = r.tk },
            .lay = r.lay,
            .st = &ps,
            .ids = prompt[start .. start + R],
            .next = prompt[start + 1 .. @min(start + R + 1, prompt.len)],
            .split = r.split,
        };
        try r.tierBegin(q, false, q.st.pos, R);
        if (c.next.len > 0) try r.tierBegin(q, true, q.mtp_len, c.next.len);
        try Pm.prefillChunk(&c, r.gpa, .{ .ctx = r, .gather = promptGatherFn }, r.overlap, r.stream);
        try q.last.copyFrom(0, w.fnormed.ptr + (R - 1) * D * 2, D * 2, r.stream.handle);
        q.st.pos = ps.pos;
        q.mtp_len = ps.mtp_len;
        try r.tierSettle(q, false, q.st.pos);
        try r.tierSettle(q, true, q.mtp_len);
        return start + R;
    }

    /// The reply's first token from the prompt's last row (after its last `fill`); the head owes that row.
    pub fn first(r: *Runner, q: *Seq) !u32 {
        const t = try r.sample(r.w.logits, 0, 1, null);
        try r.owe(q, q.last.ptr, &.{t});
        return t;
    }

    /// One verify window over `ids` (the pending token, then the drafts) at the sequence's length: every row's
    /// greedy sample into `out`; the rows stay uncommitted until `commit`.
    pub fn window(r: *Runner, q: *Seq, ids: []const u32, out: []u32) !void {
        const D = r.lay.cfg.hidden;
        const R = ids.len;
        if (R == 0 or R > rows) return error.WindowTooWide;
        if (q.st.pos + R >= q.st.caches[0].capacity) return error.PromptTooLong;
        try r.tierBegin(q, false, q.st.pos, R);
        try r.w.b.ids.uploadAsync(0, std.mem.sliceAsBytes(ids), r.stream.handle);
        try M.forward(r.k, r.lay, r.model, &q.st, &r.w, R, false, r.gather(), null, r.stream);
        try q.last.copyFrom(0, r.w.fnormed.ptr, R * D * 2, r.stream.handle);
        for (0..R) |row| out[row] = try r.sample(r.w.logits, row, 1, null);
    }

    /// One stream's verify window in a shared forward: its ids (the pending token, then the drafts) and where its
    /// rows' samples go.
    pub const Window = struct { q: *Seq, ids: []const u32, out: []u32 };

    /// Several streams' verify windows as one forward (`M.forwardSegs`): every row's greedy sample into its window's
    /// `out`; each sequence's rows stay uncommitted until its `commit`.
    pub fn windows(r: *Runner, wins: []const Window) !void {
        const D = r.lay.cfg.hidden;
        var segs: [M.max_segs]M.Seg = undefined;
        if (wins.len == 0 or wins.len > M.max_segs) return error.TooManyStreams;
        var R: usize = 0;
        for (wins, segs[0..wins.len]) |w, *sg| {
            const n = w.ids.len;
            if (n == 0 or n > rows) return error.WindowTooWide;
            if (w.q.st.pos + n >= w.q.st.caches[0].capacity) return error.PromptTooLong;
            if (R + n > batch_rows) return error.WindowTooWide;
            try r.tierBegin(w.q, false, w.q.st.pos, n);
            try r.w.b.ids.uploadAsync(R * 4, std.mem.sliceAsBytes(w.ids), r.stream.handle);
            sg.* = .{ .st = &w.q.st, .a0 = R, .n = n };
            R += n;
        }
        if (r.prof) |*p| try p.verifyBegin(R, r.stream);
        try M.forwardSegs(r.k, r.lay, r.model, segs[0..wins.len], &r.w, R, r.gather(), r.stream);
        if (r.prof) |*p| try p.forwardEnd(r.stream);
        const t_draw = RP.nowNs();
        var drawn: [batch_rows]u32 = undefined;
        try r.greedyRows(r.w.logits, 0, R, &drawn);
        if (r.prof) |*p| try p.drawEnd(t_draw);
        for (wins, segs[0..wins.len]) |w, sg| {
            try w.q.last.copyFrom(0, r.w.fnormed.ptr + sg.a0 * D * 2, sg.n * D * 2, r.stream.handle);
            @memcpy(w.out[0..sg.n], drawn[sg.a0..][0..sg.n]);
        }
    }

    /// Keep the last window's first `keep` of its `R` rows.
    pub fn commit(r: *Runner, q: *Seq, R: usize, keep: usize) !void {
        const t0 = RP.nowNs();
        try M.commit(r.k, r.lay, &q.st, R, keep, false, &r.w, r.stream);
        try r.tierSettle(q, false, q.st.pos);
        if (r.prof) |*p| p.committed(t0);
    }

    /// The head owes the rows at `hidden` (one a token of `next`, the token after each); absorbed by `absorb`.
    pub fn owe(r: *Runner, q: *Seq, hidden: Ptr, next: []const u32) !void {
        const D = r.lay.cfg.hidden;
        if (q.m_next.items.len + next.len > cap) return error.BacklogFull;
        try q.backlog.copyFrom(q.m_next.items.len * D * 2, hidden, next.len * D * 2, r.stream.handle);
        try q.m_next.appendSlice(q.gpa, next);
    }

    /// Whether `owe`ing `n` more rows needs an `absorb` first.
    pub fn owesTooMany(q: *const Seq, n: usize) bool {
        return q.m_next.items.len + n > cap;
    }

    /// decode.py `draft`'s first part: the last chain rewound, the owed rows absorbed in windows of `rows`. The
    /// head's buffers are shared by the sequences: a `chain` must follow its sequence's `absorb` directly.
    pub fn absorb(r: *Runner, q: *Seq) !void {
        const D = r.lay.cfg.hidden;
        q.fresh = false;
        q.mtp_len -= q.mtp_drafted;
        q.mtp_drafted = 0;
        if (q.tier) |x| x.t.rewind(x.hm, q.mtp_len);
        const owed = q.m_next.items.len;
        if (r.prof) |*p| if (owed > 0) try p.absorbBegin(r.stream);
        var s0: usize = 0;
        while (s0 < q.m_next.items.len) : (s0 += rows) {
            const part = q.m_next.items[s0..@min(s0 + rows, q.m_next.items.len)];
            try r.mtpStep(q, &r.mb, part, q.backlog.ptr + s0 * D * 2);
            q.mtp_len += part.len;
            q.fresh = true;
        }
        q.m_next.clearRetainingCapacity();
        if (r.prof) |*p| try p.absorbEnd(owed, r.stream);
        // the absorbed rows are final; the chain's drafts after them are not
        try r.tierSettle(q, true, q.mtp_len);
    }

    /// decode.py `draft`'s chain after `absorb`: up to `depth` drafts, cut where the product of their confidences
    /// falls under 0.35 (the first always kept); each draft's confidence into `probs`. Returns the drafts made.
    pub fn chain(r: *Runner, q: *Seq, depth: usize, drafts: []u32, probs: ?[]f64) !usize {
        if (!q.fresh or depth == 0) return 0;
        q.fresh = false;
        var p_chain: f64 = 1.0;
        var nd: usize = 0;
        for (0..depth) |j| {
            const t0 = RP.nowNs();
            defer if (r.prof) |*pr| pr.step(t0);
            var p: f64 = 0;
            const tok = try r.sample(r.mb.logits, 0, draft_k, &p);
            if (j > 0 and p_chain * p < copy_drafts.confidence) break;
            drafts[nd] = tok;
            if (probs) |ps| ps[nd] = p;
            nd += 1;
            p_chain *= p;
            if (p_chain < copy_drafts.confidence) break;
            if (j + 1 < depth) {
                try r.mtpStep(q, &r.mb, &.{tok}, r.mb.fnormed.ptr);
                q.mtp_len += 1;
                q.mtp_drafted += 1;
            }
        }
        return nd;
    }
};

/// One request on one rank as the Mia fork decodes it (the parity driver: its rounds are the reference's rounds).
pub const Engine = struct {
    r: Runner,
    q: Seq,
    copies: ?copy_drafts.CopyDrafts = null,
    out: std.ArrayList(u32) = .empty, // the reply so far (its last token pending)
    // the request's tallies (the engine's stats)
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    arms: std.ArrayList(u8) = .empty,
    keeps: std.ArrayList(usize) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, k: M.Kernels, tk: *const T.Kernels, model: *const M.Model, mtp: *const T.Mtp, link: Link, stream: cuda.Stream, capacity: usize, kv: A.Kv) !Engine {
        var r = try Runner.init(gpa, d, lay, k, tk, model, mtp, link, stream, capacity);
        errdefer r.deinit();
        return .{ .r = r, .q = try Seq.init(gpa, d, lay, model.layers.len, capacity, kv) };
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.r.gpa;
        e.q.deinit();
        e.r.deinit();
        if (e.copies) |*c| c.deinit();
        e.out.deinit(gpa);
        e.arms.deinit(gpa);
        e.keeps.deinit(gpa);
    }

    /// decode.py `prefill` from position 0 with the MTP head: the prompt's chunks, their rows absorbed (the last row
    /// waits for the first token), the first token drawn from the last row; it becomes the pending token.
    pub fn prefill(e: *Engine, prompt: []const u32) !u32 {
        const gpa = e.r.gpa;
        if (prompt.len == 0 or prompt.len + rows >= e.q.st.caches[0].capacity) return error.PromptTooLong;
        var start: usize = 0;
        while (start < prompt.len) start = try e.r.fill(&e.q, prompt, start);
        const t = try e.r.first(&e.q);
        try e.out.append(gpa, t);
        const ctx = try gpa.alloc(u32, prompt.len + 1);
        defer gpa.free(ctx);
        @memcpy(ctx[0..prompt.len], prompt);
        ctx[prompt.len] = t;
        e.copies = try copy_drafts.CopyDrafts.init(gpa, .{}, ctx);
        return t;
    }

    /// One round (auto_decode's loop body) toward a reply of `count` tokens: drafts, the verify window, the keep;
    /// the kept tokens are appended to `out` and returned (a slice of it).
    pub fn round(e: *Engine, count: usize) ![]const u32 {
        const gpa = e.r.gpa;
        const c = e.r.lay.cfg;
        const room = count - e.out.items.len;
        var drafts: [32]u32 = undefined;
        var nd = e.copies.?.propose(count - e.out.items.len - 1, &drafts);
        const arm: u8 = if (nd > 0) 'c' else 'm';
        if (arm == 'm') {
            try e.r.absorb(&e.q);
            nd = try e.r.chain(&e.q, @max(1, @min(copy_drafts.depth_most, room)), &drafts, null);
        }
        var window: [32]u32 = undefined;
        window[0] = e.out.items[e.out.items.len - 1];
        @memcpy(window[1 .. 1 + nd], drafts[0..nd]);
        const R = 1 + nd;
        var sampled: [32]u32 = undefined;
        try e.r.window(&e.q, window[0..R], &sampled);
        var keep: usize = 1;
        for (drafts[0..nd], 0..) |dt, i| {
            if (sampled[i] != dt or std.mem.indexOfScalar(u32, c.eos[0..c.eos_n], sampled[i]) != null) break;
            keep += 1;
        }
        try e.r.commit(&e.q, R, keep);
        try e.r.owe(&e.q, e.q.last.ptr, sampled[0..keep]);
        e.rounds += 1;
        e.drafted += nd;
        e.accepted += keep - 1;
        try e.arms.append(gpa, arm);
        try e.keeps.append(gpa, keep);
        try e.copies.?.extend(sampled[0..keep]);
        const at = e.out.items.len;
        try e.out.appendSlice(gpa, sampled[0..keep]);
        return e.out.items[at..];
    }
};
