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

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, layers: usize, capacity: usize) !Seq {
        const c = lay.cfg;
        const D = c.hidden;
        var st = try M.State.init(gpa, d, lay, layers, capacity, rows, prompt_rows);
        errdefer st.deinit();
        var mcache = try A.Cache.init(d, c, capacity, A.indexRing(capacity, prompt_rows));
        errdefer mcache.deinit();
        var mpos = try Buf.alloc(d, 16);
        errdefer mpos.free();
        var backlog = try Buf.alloc(d, cap * D * 2);
        errdefer backlog.free();
        return .{ .gpa = gpa, .st = st, .mcache = mcache, .mpos = mpos, .backlog = backlog, .last = try Buf.alloc(d, rows * D * 2) };
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

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, k: M.Kernels, tk: *const T.Kernels, model: *const M.Model, mtp: *const T.Mtp, link: Link, stream: cuda.Stream, capacity: usize) !Runner {
        var r: Runner = .{ .gpa = gpa, .d = d, .lay = lay, .k = k, .tk = tk, .model = model, .mtp = mtp, .link = link, .stream = stream, .w = undefined, .mb = undefined, .pmb = undefined, .host = undefined, .cand = undefined };
        r.w = try M.Work.init(d, lay, rows, prompt_rows, capacity);
        errdefer r.w.deinit();
        r.mb = try T.Buffers.init(d, lay, rows, false);
        errdefer r.mb.deinit();
        r.pmb = try T.Buffers.init(d, lay, prompt_rows, true);
        errdefer r.pmb.deinit();
        r.host = try gpa.alloc(u16, lay.size(.vocab));
        errdefer gpa.free(r.host);
        r.cand = try Buf.alloc(d, 6 * draft_k * 4);
        return r;
    }

    pub fn deinit(r: *Runner) void {
        r.w.deinit();
        r.mb.deinit();
        r.pmb.deinit();
        r.gpa.free(r.host);
        r.cand.free();
    }

    fn gatherFn(p: *anyopaque, part: Ptr, gath: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const r: *Runner = @ptrCast(@alignCast(p));
        try r.link.gather(r.link.ctx, part, gath, floats, stream);
    }

    fn gather(r: *Runner) M.Gather {
        return .{ .ctx = r, .run = gatherFn };
    }

    fn mtpGather(r: *Runner) T.Gather {
        return .{ .ctx = r, .run = gatherFn };
    }

    const Cand = struct { v: f32, id: u32 };

    fn better(_: void, a: Cand, b: Cand) bool {
        return a.v > b.v or (a.v == b.v and a.id < b.id);
    }

    /// decode.py `sample_rows`, greedy, of row `row` of `logits` (this rank's vocabulary share): every rank's `k`
    /// best (value, id) gathered, the best by value then the lower id; with `prob`, its probability among the
    /// gathered candidates at temperature 1 (`_probability`, a draft's confidence).
    pub fn sample(r: *Runner, logits: Buf, row: usize, k: usize, prob: ?*f64) !u32 {
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
        try r.cand.upload(0, std.mem.sliceAsBytes(mine[0 .. 2 * k]));
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

    fn mtpStep(r: *Runner, q: *Seq, b: *const T.Buffers, next: []const u32, hidden: Ptr) !void {
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
        try r.w.pb.ids.upload(0, std.mem.sliceAsBytes(prompt[start .. start + R]));
        try M.forward(r.k, r.lay, r.model, &q.st, &r.w, R, true, r.gather(), null, r.stream);
        try q.last.copyFrom(0, r.w.fnormed.ptr + (R - 1) * D * 2, D * 2, r.stream.handle);
        const nxt = prompt[start + 1 .. @min(start + R + 1, prompt.len)];
        if (nxt.len > 0) {
            try r.mtpStep(q, &r.pmb, nxt, r.w.fnormed.ptr);
            q.mtp_len += nxt.len;
        }
        try M.commit(r.k, r.lay, &q.st, R, R, true, &r.w, r.stream);
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
        try r.w.b.ids.upload(0, std.mem.sliceAsBytes(ids));
        try M.forward(r.k, r.lay, r.model, &q.st, &r.w, R, false, r.gather(), null, r.stream);
        try q.last.copyFrom(0, r.w.fnormed.ptr, R * D * 2, r.stream.handle);
        for (0..R) |row| out[row] = try r.sample(r.w.logits, row, 1, null);
    }

    /// Keep the last window's first `keep` of its `R` rows.
    pub fn commit(r: *Runner, q: *Seq, R: usize, keep: usize) !void {
        try M.commit(r.k, r.lay, &q.st, R, keep, false, &r.w, r.stream);
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
        var s0: usize = 0;
        while (s0 < q.m_next.items.len) : (s0 += rows) {
            const part = q.m_next.items[s0..@min(s0 + rows, q.m_next.items.len)];
            try r.mtpStep(q, &r.mb, part, q.backlog.ptr + s0 * D * 2);
            q.mtp_len += part.len;
            q.fresh = true;
        }
        q.m_next.clearRetainingCapacity();
    }

    /// decode.py `draft`'s chain after `absorb`: up to `depth` drafts, cut where the product of their confidences
    /// falls under 0.35 (the first always kept); each draft's confidence into `probs`. Returns the drafts made.
    pub fn chain(r: *Runner, q: *Seq, depth: usize, drafts: []u32, probs: ?[]f64) !usize {
        if (!q.fresh or depth == 0) return 0;
        q.fresh = false;
        var p_chain: f64 = 1.0;
        var nd: usize = 0;
        for (0..depth) |j| {
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

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lay: config.Layout, k: M.Kernels, tk: *const T.Kernels, model: *const M.Model, mtp: *const T.Mtp, link: Link, stream: cuda.Stream, capacity: usize) !Engine {
        var r = try Runner.init(gpa, d, lay, k, tk, model, mtp, link, stream, capacity);
        errdefer r.deinit();
        return .{ .r = r, .q = try Seq.init(gpa, d, lay, model.layers.len, capacity) };
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
