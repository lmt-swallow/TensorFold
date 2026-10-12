//! Single-stream DFlash2 proposal and exact target verification. Target weights and KV precision are inherited.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const D = @import("dflash2.zig");
const W = @import("cuda_dflash_weights.zig");
const B = @import("cuda_dflash_buffers.zig");
const F = @import("cuda_dflash_forward.zig");
const K = @import("cuda_dflash_kernels.zig").Kernels;
const Q = @import("cuda_dflash_qmm.zig").Kernels;
const T = @import("cuda_mtp.zig");
const Taps = @import("cuda_dflash_taps.zig").Collector;
const Ptr = cuda.abi.DevicePtr;

pub const Session = struct {
    runner: *E.Runner,
    sequence: *E.Seq,
    model: *const W.Model,
    head: *const T.DraftHead,
    kernels: K,
    projections: Q,
    hidden_projection: *const cuda.cublaslt.Linear,
    context: B.Context,
    scratch: B.Scratch,
    taps: Taps,

    /// Borrow rank weights/kernels/target runner; own bounded single-sequence proposer state.
    pub fn init(r: *E.Runner, q: *E.Seq, model: *const W.Model, head: *const T.DraftHead, kernels: K, projections: Q, hp: *const cuda.cublaslt.Linear) !Session {
        if (r.lay.world != 2 or q.st.pos != 0) return error.BadDflashSession;
        var context = try B.Context.init(r.d, q.st.caches[0].capacity);
        errdefer context.deinit();
        var scratch = try B.Scratch.init(r.d);
        errdefer scratch.deinit();
        return .{ .runner = r, .sequence = q, .model = model, .head = head, .kernels = kernels, .projections = projections, .hidden_projection = hp, .context = context, .scratch = scratch, .taps = try Taps.init(r.d, E.prompt_rows, r.k.g, r.stream) };
    }

    pub fn deinit(s: *Session) void {
        s.taps.deinit();
        s.scratch.deinit();
        s.context.deinit();
    }

    fn appendTaps(s: *Session, rows: usize) !void {
        const r = s.runner;
        var start: usize = 0;
        while (start < rows) {
            const n = @min(64, rows - start);
            try s.taps.pack(start, n, s.scratch.get("taps"));
            try s.projections.rotary(s.scratch.get("cos").ptr, s.scratch.get("sin").ptr, s.context.logical.end, n, s.model.theta, r.stream);
            try F.addTaps(&s.context, s.model, s.scratch, s.kernels, s.projections, r.k.g, s.scratch.get("taps"), s.scratch.get("cos"), s.scratch.get("sin"), n, s.model.eps, r.stream);
            start += n;
        }
    }

    /// Direct target prefill captures the exact layer means; prompt MTP is not executed.
    /// Direct/optimized target cache parity must pass before this path is used by serving.
    pub fn fill(s: *Session, prompt: []const u32, start: usize) !usize {
        const r = s.runner;
        if (!s.context.valid or start >= prompt.len or s.context.logical.end != start) return error.BadDflashSession;
        errdefer s.context.valid = false;
        const rows = @min(E.prompt_rows, prompt.len - start);
        const hook = try s.taps.begin(rows);
        const end = try r.fillTargetHook(s.sequence, prompt, start, hook);
        try s.appendTaps(rows);
        if (s.context.logical.end != s.sequence.st.pos) return error.BadDflashSession;
        return end;
    }

    /// Sample the prompt's last target row without adding an MTP backlog entry.
    pub fn first(s: *Session) !u32 {
        if (!s.context.valid or s.context.logical.end == 0 or s.context.logical.end != s.sequence.st.pos) return error.BadDflashSession;
        errdefer s.context.valid = false;
        return s.runner.sample(s.runner.w.logits, 0, 1, null);
    }

    fn gather(raw: *anyopaque, send: Ptr, recv: Ptr, floats: usize, stream: cuda.Stream) !void {
        const s: *Session = @ptrCast(@alignCast(raw));
        try s.runner.link.gather(s.runner.link.ctx, send, recv, floats, stream);
    }

    /// Return confidence-prefix candidates. Speculative KV writes never advance committed context.
    pub fn draft(s: *Session, pending: u32) ![]u32 {
        const r = s.runner;
        if (!s.context.valid or s.context.logical.end != s.sequence.st.pos) return error.BadDflashSession;
        errdefer s.context.valid = false;
        var ids: [8]u32 = undefined;
        @memset(&ids, s.model.mask);
        ids[0] = pending;
        try s.scratch.get("ids").upload(0, std.mem.sliceAsBytes(&ids));
        try s.projections.rotary(s.scratch.get("cos").ptr, s.scratch.get("sin").ptr, s.context.logical.end, 8, s.model.theta, r.stream);
        try F.block(&s.context, s.model, s.head, s.scratch, s.kernels, s.projections, r.k.g, r.model.embed.ptr, s.scratch.get("cos").ptr, s.scratch.get("sin").ptr, s.model.eps, .{ .ctx = s, .run = gather }, s.hidden_projection, r.stream);
        try r.stream.synchronize();
        const logits = try r.gpa.alloc(u16, 7 * 77440);
        defer r.gpa.free(logits);
        try s.scratch.get("logits").download(0, std.mem.sliceAsBytes(logits));
        // Gather only each rank's top16 values/IDs, preserving token IDs as raw uint32 bits.
        var local: [7 * 16 * 2]f32 = undefined;
        for (0..7) |row| {
            const best = try D.topCandidates(r.gpa, logits[row * 77440 ..][0..77440], @intCast(r.lay.rank * 77440), 16);
            defer r.gpa.free(best);
            for (best, 0..) |candidate, i| {
                local[(row * 16 + i) * 2] = @floatCast(candidate.logit);
                local[(row * 16 + i) * 2 + 1] = @bitCast(candidate.token);
            }
        }
        try s.scratch.get("partial").upload(0, std.mem.sliceAsBytes(&local));
        try gather(s, s.scratch.get("partial").ptr, s.scratch.get("gathered").ptr, local.len, r.stream);
        try r.stream.synchronize();
        var all: [2 * 7 * 16 * 2]f32 = undefined;
        try s.scratch.get("gathered").download(0, std.mem.sliceAsBytes(&all));
        var candidates: [7 * 16]D.Candidate = undefined;
        for (0..7) |row| {
            var merged: [32]D.Candidate = undefined;
            for (0..2) |rank| for (0..16) |i| {
                const at = rank * local.len + (row * 16 + i) * 2;
                merged[rank * 16 + i] = .{ .logit = all[at], .token = @bitCast(all[at + 1]) };
            };
            std.mem.sort(D.Candidate, &merged, {}, D.candidateBefore);
            @memcpy(candidates[row * 16 ..][0..16], merged[0..16]);
        }
        var projected: [7 * 256]u16 = undefined;
        try s.scratch.get("projected").download(0, std.mem.sliceAsBytes(&projected));
        var expanded: [7 * 256]f32 = undefined;
        for (projected, 0..) |bits, i| expanded[i] = @bitCast(@as(u32, bits) << 16);
        return D.selectGreedy(r.gpa, &candidates, 16, &expanded, 256, s.model.predecessor, s.model.successor, pending, 0.3);
    }

    /// Verify pending+drafts with the unchanged target; commit only accepted input rows and their taps.
    /// `out` receives emitted greedy samples; the final sample remains pending for the next call.
    pub fn verify(s: *Session, pending: u32, drafts: []const u32, room: usize, out: []u32) !usize {
        const r = s.runner;
        if (!s.context.valid or drafts.len >= E.rows or room == 0 or out.len < @min(1 + drafts.len, room) or s.context.logical.end != s.sequence.st.pos) return error.BadDflashSession;
        errdefer s.context.valid = false;
        var ids: [E.rows]u32 = undefined;
        ids[0] = pending;
        @memcpy(ids[1..][0..drafts.len], drafts);
        const rows = 1 + drafts.len;
        var sampled: [E.rows]u32 = undefined;
        try r.windowHook(s.sequence, ids[0..rows], sampled[0..rows], try s.taps.begin(rows));
        var keep: usize = 1;
        for (drafts, 0..) |drafted, i| {
            if (keep >= room or sampled[i] != drafted or std.mem.indexOfScalar(u32, r.lay.cfg.eos[0..r.lay.cfg.eos_n], sampled[i]) != null) break;
            keep += 1;
        }
        try r.commit(s.sequence, rows, keep);
        try s.appendTaps(keep);
        @memcpy(out[0..keep], sampled[0..keep]);
        if (s.context.logical.end != s.sequence.st.pos) return error.BadDflashSession;
        return keep;
    }
};
