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

const Ptr = cuda.abi.DevicePtr;
const Buf = cuda.DeviceBuffer;

/// exact_sampling.py MARGIN and decode.py's draft candidates (20 and the margin from each rank).
const margin = 8;
pub const draft_k = 20 + margin;
/// drafter_choice.py: backlog rows the MTP head may owe before it absorbs them.
const cap = 256;
pub const rows = 16; // TF_GLM_MAX_ROWS: the widest verify window
pub const prompt_rows = 2048; // decode.PREFILL_ROWS

/// How a rank reaches the other: the all-gather of fp32 partials and of the samples' candidates.
pub const Link = struct {
    ctx: *anyopaque,
    gather: *const fn (ctx: *anyopaque, send: Ptr, recv: Ptr, floats: usize, stream: cuda.Stream) anyerror!void,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lay: config.Layout,
    k: M.Kernels,
    tk: *const T.Kernels,
    model: *const M.Model,
    mtp: *const T.Mtp,
    link: Link,
    stream: cuda.Stream,
    st: M.State,
    w: M.Work,
    mcache: A.Cache,
    mb: T.Buffers,
    pmb: T.Buffers,
    mpos: Buf,
    backlog: Buf, // [cap, D]: the MTP head's owed input rows
    last_hidden: Buf,
    host: []u16, // a logits row
    cand: Buf,
    m_next: std.ArrayList(u32) = .empty, // the owed rows' next tokens
    mtp_len: usize = 0,
    mtp_drafted: usize = 0,
    copies: ?copy_drafts.CopyDrafts = null,
    out: std.ArrayList(u32) = .empty, // the reply so far (its last token pending)
    // the request's tallies (the engine's stats)
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    arms: std.ArrayList(u8) = .empty,
    keeps: std.ArrayList(usize) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, k: M.Kernels, tk: *const T.Kernels, model: *const M.Model, mtp: *const T.Mtp, link: Link, stream: cuda.Stream, capacity: usize) !Engine {
        const c = lay.cfg;
        const D = c.hidden;
        var e: Engine = undefined;
        e = .{
            .gpa = gpa, .d = d, .lay = lay, .k = k, .tk = tk, .model = model, .mtp = mtp, .link = link, .stream = stream,
            .st = try M.State.init(gpa, d, lay, model.layers.len, capacity, rows, prompt_rows),
            .w = undefined, .mcache = undefined, .mb = undefined, .pmb = undefined, .mpos = undefined,
            .backlog = undefined, .last_hidden = undefined, .host = undefined, .cand = undefined,
        };
        errdefer e.st.deinit();
        e.w = try M.Work.init(d, lay, rows, prompt_rows, capacity);
        e.mcache = try A.Cache.init(d, c, capacity, A.indexRing(capacity, prompt_rows));
        e.mb = try T.Buffers.init(d, lay, rows, false);
        e.pmb = try T.Buffers.init(d, lay, prompt_rows, true);
        e.mpos = try Buf.alloc(d, 16);
        e.backlog = try Buf.alloc(d, cap * D * 2);
        e.last_hidden = try Buf.alloc(d, D * 2);
        e.host = try gpa.alloc(u16, lay.size(.vocab));
        e.cand = try Buf.alloc(d, 6 * draft_k * 4);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.st.deinit();
        e.w.deinit();
        e.mcache.deinit();
        e.mb.deinit();
        e.pmb.deinit();
        e.mpos.free();
        e.backlog.free();
        e.last_hidden.free();
        e.gpa.free(e.host);
        e.cand.free();
        e.m_next.deinit(e.gpa);
        if (e.copies) |*c| c.deinit();
        e.out.deinit(e.gpa);
        e.arms.deinit(e.gpa);
        e.keeps.deinit(e.gpa);
    }

    fn gatherFn(p: *anyopaque, part: Ptr, gath: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(p));
        try e.link.gather(e.link.ctx, part, gath, floats, stream);
    }

    fn gather(e: *Engine) M.Gather {
        return .{ .ctx = e, .run = gatherFn };
    }

    fn mtpGather(e: *Engine) T.Gather {
        return .{ .ctx = e, .run = gatherFn };
    }

    const Cand = struct { v: f32, id: u32 };

    fn better(_: void, a: Cand, b: Cand) bool {
        return a.v > b.v or (a.v == b.v and a.id < b.id);
    }

    /// decode.py `sample_rows`, greedy, of row `row` of `logits` (this rank's vocabulary share): every rank's `k`
    /// best (value, id) gathered, the best by value then the lower id; with `prob`, its probability among the
    /// gathered candidates at temperature 1 (`_probability`, a draft's confidence).
    fn sample(e: *Engine, logits: Buf, row: usize, k: usize, prob: ?*f64) !u32 {
        const nv = e.host.len;
        try e.stream.synchronize();
        try logits.download(row * nv * 2, std.mem.sliceAsBytes(e.host));
        var best: [draft_k]Cand = undefined;
        var n: usize = 0;
        for (e.host, 0..) |x, j| {
            const cand: Cand = .{ .v = @bitCast(@as(u32, x) << 16), .id = @intCast(e.lay.rank * nv + j) };
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
        try e.cand.upload(0, std.mem.sliceAsBytes(mine[0 .. 2 * k]));
        try e.link.gather(e.link.ctx, e.cand.ptr, e.cand.ptr + 2 * draft_k * 4, 2 * k, e.stream);
        try e.stream.synchronize();
        var both: [4 * draft_k]f32 = undefined;
        try e.cand.download(2 * draft_k * 4, std.mem.sliceAsBytes(both[0 .. 4 * k]));
        var all: [2 * draft_k]Cand = undefined;
        for (0..2) |r| for (0..k) |i| {
            all[r * k + i] = .{ .v = both[r * 2 * k + i], .id = @bitCast(both[r * 2 * k + k + i]) };
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

    fn mtpStep(e: *Engine, b: *const T.Buffers, next: []const u32, hidden: Ptr) !void {
        const pre = b.prefill;
        try T.run(e.k.g, e.tk, e.k.mk, e.lay, e.mtp, b, .{ .cache = &e.mcache, .ds = if (pre) &e.w.pds else &e.w.ds, .ms = if (pre) &e.w.pms else &e.w.ms, .pos_dev = e.mpos }, e.model.embed.ptr, next, hidden, e.mtp_len, e.mtpGather(), e.stream);
    }

    /// decode.py `prefill` from position 0 with the MTP head: the prompt's chunks, their rows absorbed (the last row
    /// waits for the first token), the first token drawn from the last row; it becomes the pending token.
    pub fn prefill(e: *Engine, prompt: []const u32) !u32 {
        const D = e.lay.cfg.hidden;
        if (prompt.len == 0 or prompt.len + rows >= e.st.caches[0].capacity) return error.PromptTooLong;
        var start: usize = 0;
        while (start < prompt.len) : (start += prompt_rows) {
            const R = @min(prompt_rows, prompt.len - start);
            try e.w.pb.ids.upload(0, std.mem.sliceAsBytes(prompt[start .. start + R]));
            try M.forward(e.k, e.lay, e.model, &e.st, &e.w, R, true, e.gather(), null, e.stream);
            try e.last_hidden.copyFrom(0, e.w.fnormed.ptr + (R - 1) * D * 2, D * 2, e.stream.handle);
            const nxt = prompt[start + 1 .. @min(start + R + 1, prompt.len)];
            if (nxt.len > 0) {
                try e.mtpStep(&e.pmb, nxt, e.w.fnormed.ptr);
                e.mtp_len += nxt.len;
            }
            try M.commit(e.k, e.lay, &e.st, R, R, true, &e.w, e.stream);
        }
        const first = try e.sample(e.w.logits, 0, 1, null);
        try e.out.append(e.gpa, first);
        const ctx = try e.gpa.alloc(u32, prompt.len + 1);
        defer e.gpa.free(ctx);
        @memcpy(ctx[0..prompt.len], prompt);
        ctx[prompt.len] = first;
        e.copies = try copy_drafts.CopyDrafts.init(e.gpa, .{}, ctx);
        try e.backlog.copyFrom(0, e.last_hidden.ptr, D * 2, e.stream.handle);
        try e.m_next.append(e.gpa, first);
        return first;
    }

    /// One round (auto_decode's loop body) toward a reply of `count` tokens: drafts, the verify window, the keep;
    /// the kept tokens are appended to `out` and returned (a slice of it).
    pub fn round(e: *Engine, count: usize) ![]const u32 {
        const D = e.lay.cfg.hidden;
        const c = e.lay.cfg;
        const room = count - e.out.items.len;
        var drafts: [32]u32 = undefined;
        var nd = e.copies.?.propose(count - e.out.items.len - 1, &drafts);
        const arm: u8 = if (nd > 0) 'c' else 'm';
        if (arm == 'm') {
            // decode.py `draft`: rewind the last chain, absorb the owed rows in windows of 16, chain up to 3 drafts
            e.mtp_len -= e.mtp_drafted;
            e.mtp_drafted = 0;
            var s0: usize = 0;
            while (s0 < e.m_next.items.len) : (s0 += rows) {
                const part = e.m_next.items[s0..@min(s0 + rows, e.m_next.items.len)];
                try e.mtpStep(&e.mb, part, e.backlog.ptr + s0 * D * 2);
                e.mtp_len += part.len;
            }
            e.m_next.clearRetainingCapacity();
            const depth = @max(1, @min(copy_drafts.depth_most, room));
            var chain: f64 = 1.0;
            nd = 0;
            for (0..depth) |j| {
                var p: f64 = 0;
                const tok = try e.sample(e.mb.logits, 0, draft_k, &p);
                if (j > 0 and chain * p < copy_drafts.confidence) break;
                drafts[nd] = tok;
                nd += 1;
                chain *= p;
                if (chain < copy_drafts.confidence) break;
                if (j + 1 < depth) {
                    try e.mtpStep(&e.mb, &.{tok}, e.mb.fnormed.ptr);
                    e.mtp_len += 1;
                    e.mtp_drafted += 1;
                }
            }
        }
        var window: [32]u32 = undefined;
        window[0] = e.out.items[e.out.items.len - 1];
        @memcpy(window[1 .. 1 + nd], drafts[0..nd]);
        const R = 1 + nd;
        try e.w.b.ids.upload(0, std.mem.sliceAsBytes(window[0..R]));
        try M.forward(e.k, e.lay, e.model, &e.st, &e.w, R, false, e.gather(), null, e.stream);
        var sampled: [32]u32 = undefined;
        for (0..R) |r| sampled[r] = try e.sample(e.w.logits, r, 1, null);
        var keep: usize = 1;
        for (drafts[0..nd], 0..) |dt, i| {
            if (sampled[i] != dt or std.mem.indexOfScalar(u32, c.eos[0..c.eos_n], sampled[i]) != null) break;
            keep += 1;
        }
        try M.commit(e.k, e.lay, &e.st, R, keep, false, &e.w, e.stream);
        if (e.m_next.items.len + keep > cap) return error.BacklogFull;
        try e.backlog.copyFrom(e.m_next.items.len * D * 2, e.w.fnormed.ptr, keep * D * 2, e.stream.handle);
        try e.m_next.appendSlice(e.gpa, sampled[0..keep]);
        e.rounds += 1;
        e.drafted += nd;
        e.accepted += keep - 1;
        try e.arms.append(e.gpa, arm);
        try e.keeps.append(e.gpa, keep);
        try e.copies.?.extend(sampled[0..keep]);
        const at = e.out.items.len;
        try e.out.appendSlice(e.gpa, sampled[0..keep]);
        return e.out.items[at..];
    }
};
