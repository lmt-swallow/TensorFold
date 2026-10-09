//! `dsv41-lanes-parallel <dir> <rank-cache-dir> <model-dir> <rank> [aot-dir]`: the served engine's concurrent steps
//! (cuda_lanes.zig `Streams`) at one rank on dump_lanes_parallel.py's fixture (lanes_parallel_capture.sh's work dir),
//! each step's record encoded as rank 0 announces it and run from the parsed record as every rank runs it, the other
//! rank's partials supplied from the fixture at every gather in the fork's order:
//! - two admissions into the lowest free slots and the first free extents (`Extents`, the fixture's placement): the
//!   gathers' order, each prompt's last row's logits and first token, the pool's planes and the drafter rings after
//!   each;
//! - one drafts step for both streams (one batched pass): their drafts;
//! - one round over both streams' windows (their pending tokens and drafts, the fork's oracle continuation in one):
//!   every row's logits, the taps, the greedy picks and each stream's accepted count;
//! - a keeps step for the first stream, then a drafts step absorbing both streams' kept rows (the second stream's
//!   keep implied) and drafting their next blocks: the pool's planes and the rings after, the drafts.
//! Every byte the Python fork's (each extent's compressed scratch row left out, as the rounds test). The rounds of
//! 1 .. 16 rows and batched drafts of 1 .. 3 streams are timed after (this rank's layers, the gathers local), and a
//! stream's bytes at a 1M-position context printed.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;
const Fixture = fixture.Fixture;
const Ptr = cuda.abi.DevicePtr;
const L = dsv41.serve_lanes;
const F = dsv41.forward;

/// The other rank's partials in gather order from the fixture, this rank's copied in beside them; once they run
/// out with `loose` set, this rank's partial twice (timing runs only). A gather recorded into a graph copies the
/// other rank's partial from its place in `mailbox`; `launched` (the round's and the drafter's launch hook) puts
/// the next partials there before each launch, in the graph's gather order.
const Supply = struct {
    gpu: Gpu,
    rank: usize,
    other: std.ArrayList(cuda.DeviceBuffer) = .empty,
    next: usize = 0,
    loose: bool = false,
    mailbox: ?cuda.DeviceBuffer = null,
    cap_off: usize = 0,

    const box_align = 256;

    fn gather(ctx: *anyopaque, partial: Ptr, words: usize, out: Ptr, stream: cuda.Stream) anyerror!void {
        const self: *Supply = @ptrCast(@alignCast(ctx));
        const dst: cuda.DeviceBuffer = .{ .d = self.gpu.d, .ptr = out, .len = 2 * words * 4 };
        if (!self.loose and try cuda.graph.captureStatus(stream) == .active) {
            const box = self.mailbox orelse return error.NoMailbox;
            const off = self.cap_off;
            self.cap_off += std.mem.alignForward(usize, words * 4, box_align);
            if (self.cap_off > box.len) return error.GatherOverrun;
            try dst.copyFrom(self.rank * words * 4, partial, words * 4, stream.handle);
            try dst.copyFrom((1 - self.rank) * words * 4, box.ptr + off, words * 4, stream.handle);
            return;
        }
        if (self.loose) {
            try dst.copyFrom(0, partial, words * 4, stream.handle);
            try dst.copyFrom(words * 4, partial, words * 4, stream.handle);
            return;
        }
        if (self.next >= self.other.items.len) return error.GatherOverrun;
        const o = self.other.items[self.next];
        self.next += 1;
        if (o.len != words * 4) return error.GatherShape;
        try dst.copyFrom(self.rank * words * 4, partial, words * 4, stream.handle);
        try dst.copyFrom((1 - self.rank) * words * 4, o.ptr, words * 4, stream.handle);
    }

    /// Before a graph launch of `gathers` recorded gathers: the next partials into the mailbox (none once loose).
    fn launched(ctx: *anyopaque, gathers: usize, stream: cuda.Stream) anyerror!void {
        const self: *Supply = @ptrCast(@alignCast(ctx));
        self.cap_off = 0;
        if (self.loose) return;
        const box = self.mailbox orelse return error.NoMailbox;
        var off: usize = 0;
        for (0..gathers) |_| {
            if (self.next >= self.other.items.len) return error.GatherOverrun;
            const o = self.other.items[self.next];
            self.next += 1;
            try box.copyFrom(off, o.ptr, o.len, stream.handle);
            off += std.mem.alignForward(usize, o.len, box_align);
        }
    }

    /// The mailbox: room for every partial at once.
    fn makeMailbox(self: *Supply) !void {
        var total: usize = 0;
        for (self.other.items) |b| total += std.mem.alignForward(usize, b.len, box_align);
        self.mailbox = try cuda.DeviceBuffer.alloc(self.gpu.d, @max(total, box_align));
    }

    fn add(self: *Supply, fx: Fixture, name: []const u8) !void {
        const bytes = try fx.bytes(name);
        defer self.gpu.gpa.free(bytes);
        try self.other.append(self.gpu.gpa, try cuda.DeviceBuffer.fromHost(self.gpu.d, bytes));
    }

    fn deinit(self: *Supply) void {
        if (self.mailbox) |*b| b.free();
        for (self.other.items) |*b| b.free();
        self.other.deinit(self.gpu.gpa);
    }
};

const Check = struct {
    gpu: Gpu,
    fx: Fixture,
    failures: usize = 0,

    /// `len` bytes at `ptr` against `expected_<name>` (its whole length when `len` is 0), the rows `skip` of
    /// `row_bytes` each zeroed on both sides first.
    fn same(self: *Check, label: []const u8, name: []const u8, ptr: Ptr, len: usize, row_bytes: usize, skip: []const usize) !bool {
        const gpa = self.gpu.gpa;
        var nb: [128]u8 = undefined;
        const want = self.fx.bytes(try std.fmt.bufPrint(&nb, "expected_{s}", .{name})) catch {
            std.debug.print("FAIL {s} {s}: no expected array\n", .{ label, name });
            self.failures += 1;
            return false;
        };
        defer gpa.free(want);
        const n = if (len == 0) want.len else len;
        if (n > want.len) {
            std.debug.print("FAIL {s} {s}: {d} bytes held, {d} expected\n", .{ label, name, n, want.len });
            self.failures += 1;
            return false;
        }
        const got = try gpa.alloc(u8, n);
        defer gpa.free(got);
        const view: cuda.DeviceBuffer = .{ .d = self.gpu.d, .ptr = ptr, .len = n };
        try view.download(0, got);
        for (skip) |row| if ((row + 1) * row_bytes <= n) {
            @memset(want[row * row_bytes .. (row + 1) * row_bytes], 0);
            @memset(got[row * row_bytes .. (row + 1) * row_bytes], 0);
        };
        var what: [192]u8 = undefined;
        check.sameBytes(try std.fmt.bufPrint(&what, "{s} {s}", .{ label, name }), got, want[0..n]) catch {
            self.failures += 1;
            return false;
        };
        return true;
    }

    /// The pool's planes and the drafter rings against `<prefix>.<plane>` / `<prefix>.ring<j>`, every busy
    /// extent's compressed scratch row left out (its bytes depend on the fork's launch order).
    fn pool(self: *Check, label: []const u8, prefix: []const u8, st: *const L.Streams) !bool {
        const cfg = st.cfg;
        var ok = true;
        var nb: [96]u8 = undefined;
        for (st.pool.rings, 0..) |r, li| ok = (try self.same(label, try std.fmt.bufPrint(&nb, "{s}.L{d}.ring", .{ prefix, li }), r.ptr, r.len, 0, &.{})) and ok;
        for (st.pool.planes, 0..) |pp, li| {
            const p = pp orelse continue;
            const ratio = cfg.ratio(li);
            var skip: [L.max_slots]usize = undefined;
            var ns: usize = 0;
            for (st.lanes) |l| if (l.busy) {
                skip[ns] = (l.base + l.size) / ratio - 1;
                ns += 1;
            };
            const planes = .{ .{ "comp_codes", p.comp_codes, cfg.head_dim / 2 }, .{ "comp_scales", p.comp_scales, cfg.head_dim / 16 }, .{ "idxk_codes", p.idxk_codes, cfg.idx_dim / 2 }, .{ "idxk_scales", p.idxk_scales, cfg.idx_dim / 32 } };
            inline for (planes) |pl| if (pl[1].len > 0) {
                ok = (try self.same(label, try std.fmt.bufPrint(&nb, "{s}.L{d}.{s}", .{ prefix, li, pl[0] }), pl[1].ptr, pl[1].len, pl[2], skip[0..ns])) and ok;
            };
            if (p.raw_kv.len > 0) {
                ok = (try self.same(label, try std.fmt.bufPrint(&nb, "{s}.L{d}.raw_kv", .{ prefix, li }), p.raw_kv.ptr, p.raw_kv.len, 0, &.{})) and ok;
                ok = (try self.same(label, try std.fmt.bufPrint(&nb, "{s}.L{d}.raw_score", .{ prefix, li }), p.raw_score.ptr, p.raw_score.len, 0, &.{})) and ok;
            }
        }
        if (st.dpool) |dp| for (dp.rings, 0..) |r, j| {
            ok = (try self.same(label, try std.fmt.bufPrint(&nb, "{s}.ring{d}", .{ prefix, j }), r.ptr, r.len, 0, &.{})) and ok;
        };
        return ok;
    }

    /// Each stream's drafts against `expected_<name>` [N, block] (i64) and its confidences within 1e-4.
    fn drafts(self: *Check, label: []const u8, name: []const u8, st: *const L.Streams, slots: []const usize, block: usize) !bool {
        const gpa = self.gpu.gpa;
        var nb: [96]u8 = undefined;
        const wd_bytes = try self.fx.bytes(try std.fmt.bufPrint(&nb, "expected_{s}.drafts", .{name}));
        defer gpa.free(wd_bytes);
        const wd = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(wd_bytes)));
        const wc_bytes = try self.fx.bytes(try std.fmt.bufPrint(&nb, "expected_{s}.conf", .{name}));
        defer gpa.free(wc_bytes);
        const wc = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(wc_bytes)));
        var ok = wd.len == slots.len * block;
        var worst: f32 = 0;
        for (slots, 0..) |slot, s| {
            const l = &st.lanes[slot];
            if (l.held != block or l.drafts.len != block) ok = false;
            for (0..block) |i| {
                if (s * block + i < wd.len and l.drafts.tokens[i] != wd[s * block + i]) ok = false;
                if (s * block + i < wc.len) worst = @max(worst, @abs(l.drafts.conf[i] - wc[s * block + i]) / @max(1.0, @abs(wc[s * block + i])));
            }
            if (!ok) std.debug.print("FAIL {s}: slot {d} drafts {any}, the fork's {any}\n", .{ label, slot, l.drafts.tokens[0..block], wd[s * block ..][0..block] });
        }
        if (worst > 1e-4) {
            std.debug.print("FAIL {s}: the confidences differ by {e} at worst\n", .{ label, worst });
            ok = false;
        }
        if (!ok) self.failures += 1;
        return ok;
    }
};

pub fn parallel(gpu: Gpu, dir: []const u8, cache_dir: []const u8, model_dir: []const u8, rank: usize, aot_override: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const dsp = dsv41.dspark;
    try check.expect(rank < 2, "rank {d}: the fixture holds two ranks", .{rank});
    const aot_dir = if (aot_override) |o| try gpa.dupe(u8, o) else try std.fs.path.join(gpa, &.{ dir, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    std.debug.print("Triton set: {s} ({d} variants)\n", .{ aot_dir, set.variants.len });
    var glue = try dsv41.attention.Glue.load(gpu.d);
    defer glue.deinit();
    var kernels = try dsv41.exl3.Kernels.load(gpu.d);
    defer kernels.deinit();
    var ek = try dsv41.experts.Kernels.load(gpu.d);
    defer ek.deinit();
    var cfg = try dsv41.config.Config.read(gpa, io, model_dir);
    const fdir = try std.fs.path.join(gpa, &.{ dir, "out", "fixtures", "parallel" });
    defer gpa.free(fdir);
    var fx = try Fixture.open(gpa, io, fdir);
    defer fx.deinit();
    const layers: usize = @intCast(try fx.int("layers"));
    const world: u32 = @intCast(try fx.int("world"));
    const slots: usize = @intCast(try fx.int("slots"));
    const cap: usize = @intCast(try fx.int("cap"));
    const total: usize = @intCast(try fx.int("total"));
    const chunk: usize = @intCast(try fx.int("chunk"));
    const bucket: usize = @intCast(try fx.int("bucket"));
    const stages: usize = @intCast(try fx.int("stages"));
    const block: usize = @intCast(try fx.int("block"));
    const n_streams: usize = @intCast(try fx.int("n_streams"));
    try check.expect(world == 2 and n_streams == 2, "the fixture: world {d}, {d} streams", .{ world, n_streams });
    try check.expect(stages == cfg.nextn and block == cfg.dspark_block, "the fixture's drafter: {d} stages of {d} rows", .{ stages, block });
    // the fixture's layers stand in for the tap layers
    cfg.dspark_taps = .{};
    var it = std.mem.splitScalar(u8, try fx.string("taps"), ',');
    while (it.next()) |t| {
        cfg.dspark_taps.items[cfg.dspark_taps.len] = try std.fmt.parseInt(i64, t, 10);
        cfg.dspark_taps.len += 1;
    }

    const rc_path = (try dsv41.rank_cache.RankCache.locate(gpa, io, cache_dir, rank, world)) orelse return error.NoRankCache;
    defer gpa.free(rc_path);
    var rc = try dsv41.rank_cache.RankCache.open(gpa, rc_path);
    defer rc.close();
    var w = try dsv41.weights.load(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1, layers);
    defer w.deinit();
    var ds = try dsv41.weights.loadDSpark(gpa, gpu.d, &cfg, &rc, &kernels, rank, world, .mul1);
    defer ds.deinit(gpa);
    std.debug.print("loaded {d} layers and {d} DSpark stages of rank {d} from {s}\n", .{ layers, ds.blocks.len, rank, rc_path });
    const views = try gpa.alloc(dsp.Stage, ds.blocks.len);
    defer gpa.free(views);
    for (views, ds.blocks) |*v, *b| v.* = dsp.Stage.of(b);
    const rope_rows: usize = @intCast(try fx.int("rope_rows"));
    var ck: Check = .{ .gpu = gpu, .fx = fx };
    var rope_c: dsv41.attention.Rope = .{ .cos = try upload(gpu, fx, "cos_c"), .sin = try upload(gpu, fx, "sin_c"), .rows = rope_rows };
    defer rope_c.cos.free();
    defer rope_c.sin.free();
    var rope_w: dsv41.attention.Rope = .{ .cos = try upload(gpu, fx, "cos_w"), .sin = try upload(gpu, fx, "sin_w"), .rows = rope_rows };
    defer rope_w.cos.free();
    defer rope_w.sin.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    var supply: Supply = .{ .gpu = gpu, .rank = rank };
    defer supply.deinit();
    const gather: F.Gather = .{ .ctx = &supply, .call = Supply.gather };
    const drafter = try dsp.Drafter.initWith(gpa, gpu.d, dsp.dimsOf(&cfg, rank, world), &set, &glue, &ek, views, dsp.headOf(&ds, &w.head), null, .{ .ctx = &supply, .call = Supply.gather });
    defer drafter.deinit();
    try drafter.enableWide(&kernels);
    var fwd = try F.Forward.init(gpa, gpu.d, &cfg, &w, &set, &glue, &ek, world, L.max_rows, L.extent_align, bucket, rope_c, rope_w, stream);
    defer fwd.deinit();
    var st: L.Streams = undefined;
    try st.init(gpa, gpu.d, &fwd, &kernels, null, chunk, stream, drafter, slots, cap);
    defer st.deinit();
    try check.expect(st.extents.total == total, "the window's positions: {d}, the fork's {d}", .{ st.extents.total, total });

    // the fork's gathers in order: each admission's chunks (attention's and the MoE's a layer) and head, the first
    // drafts (each stage's attention and MoE, each Markov step's bests), the round (its layers, its head), the
    // next drafts
    const other: usize = 1 - rank;
    var nb: [128]u8 = undefined;
    var marks: [5]usize = undefined;
    for (0..n_streams) |i| {
        const n_chunks: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_chunks", .{i})));
        for (0..n_chunks) |ci| for (0..layers) |li| {
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_A{d}.C{d}.L{d}.p_attn.r{d}", .{ i, ci, li, other }));
            try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_A{d}.C{d}.L{d}.p_moe.r{d}", .{ i, ci, li, other }));
        };
        try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_A{d}.head.local.r{d}", .{ i, other }));
        marks[i] = supply.other.items.len;
    }
    const addDraft = struct {
        fn f(s: *Supply, f_: Fixture, k: usize, n_stages: usize, n_block: usize, o: usize) !void {
            var b: [96]u8 = undefined;
            for (0..n_stages) |j| {
                try s.add(f_, try std.fmt.bufPrint(&b, "expected_D{d}.s{d}.p_attn.r{d}", .{ k, j, o }));
                try s.add(f_, try std.fmt.bufPrint(&b, "expected_D{d}.s{d}.p_moe.r{d}", .{ k, j, o }));
            }
            for (0..n_block) |i| try s.add(f_, try std.fmt.bufPrint(&b, "expected_D{d}.mk{d}.send.r{d}", .{ k, i, o }));
        }
    }.f;
    try addDraft(&supply, fx, 0, stages, block, other);
    marks[2] = supply.other.items.len;
    for (0..layers) |li| {
        try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_R.L{d}.p_attn.r{d}", .{ li, other }));
        try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_R.L{d}.p_moe.r{d}", .{ li, other }));
    }
    try supply.add(fx, try std.fmt.bufPrint(&nb, "expected_R.head.local.r{d}", .{other}));
    marks[3] = supply.other.items.len;
    try addDraft(&supply, fx, 1, stages, block, other);
    marks[4] = supply.other.items.len;
    try supply.makeMailbox();
    st.runner.launch_hook = .{ .ctx = &supply, .call = Supply.launched };
    drafter.launch_hook = .{ .ctx = &supply, .call = Supply.launched };
    std.debug.print("dsv41-lanes-parallel: graphs {s}\n", .{if (st.runner.graphs) "on" else "off (TF_DS_GRAPHS=0)"});

    const op_buf = try gpa.alloc(u8, L.maxOpBytes(cap));
    defer gpa.free(op_buf);
    const ids_buf = try gpa.alloc(i64, cap);
    defer gpa.free(ids_buf);
    const V = cfg.vocab;
    var slot_of: [2]usize = undefined;
    var first: [2]u32 = undefined;

    // -- the admissions --------------------------------------------------------------------------------------------
    for (0..n_streams) |i| {
        var lb: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&lb, "dsv41-lanes-parallel rank {d} admission {d}", .{ rank, i });
        const plen: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_prompt", .{i})));
        const count: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_count", .{i})));
        const want_slot: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_slot", .{i})));
        const want_base: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_base", .{i})));
        const want_size: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_size", .{i})));
        const pb = try fx.bytes(try std.fmt.bufPrint(&nb, "expected_A{d}.prompt_ids", .{i}));
        defer gpa.free(pb);
        const prompt_ids = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(pb)));
        // the placement rank 0 makes (Lanes.prefillFn): the lowest free slot, the first extent that fits
        const need = plen + count + L.extent_tail;
        var slot: usize = 0;
        while (st.lanes[slot].busy) slot += 1;
        const base = st.extents.first(need) orelse return error.NoRoom;
        const size = L.Extents.size(need);
        try check.expect(slot == want_slot and base == want_base and size == want_size, "{s}: slot {d} at [{d}, +{d}), the fork's {d} at [{d}, +{d})", .{ label, slot, base, size, want_slot, want_base, want_size });
        const op = L.Admit.encode(op_buf, .{ .slot = slot, .engram = false, .drafts = true, .chunk = chunk, .base = base, .size = size, .ids = &.{} }, prompt_ids);
        const a = try L.Admit.parse(op);
        try check.expect(st.admits(a.slot, a.base, a.size, a.len(), a.chunk), "{s}: the record is runnable", .{label});
        const t0 = check.now(io);
        try st.admit(gather, a.slot, a.base, a.size, a.idsInto(ids_buf), a.chunk, a.replay);
        try stream.synchronize();
        const ns = check.now(io) - t0;
        var ok = supply.next == marks[i];
        if (!ok) std.debug.print("FAIL {s}: {d} gathers, the fork's {d}\n", .{ label, supply.next, marks[i] });
        ok = (try ck.same(label, try std.fmt.bufPrint(&nb, "A{d}.logits", .{i}), st.logits, V * 4, 0, &.{})) and ok;
        var one: [1]u32 = undefined;
        try st.argmaxRows(0, &one);
        const want_first: u32 = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_first", .{i})));
        if (one[0] != want_first) {
            std.debug.print("FAIL {s}: the first token {d}, the fork's {d}\n", .{ label, one[0], want_first });
            ok = false;
        }
        ok = (try ck.pool(label, try std.fmt.bufPrint(&nb, "A{d}", .{i}), &st)) and ok;
        slot_of[i] = slot;
        first[i] = one[0];
        if (ok) check.pass("BITEXACT {s}: slot {d}, extent [{d}, {d}), {d} prompt tokens in {d} ms: the gathers' order, the last row's logits, the first token {d}, the pool's planes and the drafter rings equal to the Python fork's", .{ label, slot, base, base + size, plen, @divTrunc(ns, 1_000_000), one[0] }) else ck.failures += 1;
    }

    // -- the first drafts: one batched pass ------------------------------------------------------------------------
    {
        const label = "dsv41-lanes-parallel drafts 0";
        var asks: [2]L.DraftAsk = undefined;
        for (&asks, 0..) |*a, i| a.* = .{ .slot = slot_of[i], .start = st.lanes[slot_of[i]].pos, .absorb = 0, .token = first[i], .q0 = st.lanes[slot_of[i]].pos, .depth = block };
        var rec: L.DraftsRec = undefined;
        try L.DraftsRec.parse(L.DraftsRec.encode(op_buf, &asks), &rec);
        try check.expect(st.draftable(rec.asks[0..rec.count]), "{s}: the record is runnable", .{label});
        try st.draft(rec.asks[0..rec.count]);
        try stream.synchronize();
        var ok = supply.next == marks[2];
        if (!ok) std.debug.print("FAIL {s}: {d} gathers, the fork's {d}\n", .{ label, supply.next, marks[2] });
        ok = (try ck.drafts(label, "D0", &st, &slot_of, block)) and ok;
        if (ok) check.pass("BITEXACT {s}: both streams' blocks in one pass, drafts {any} and {any} equal to the Python fork's", .{ label, st.lanes[slot_of[0]].drafts.tokens[0..block], st.lanes[slot_of[1]].drafts.tokens[0..block] });
    }

    // -- the round -------------------------------------------------------------------------------------------------
    const wb = try fx.bytes("expected_R.window");
    defer gpa.free(wb);
    const window_ids = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(wb)));
    var rws: [2]L.RoundWindow = undefined;
    var starts: [2]usize = undefined;
    var accepted: [2]usize = undefined;
    {
        const label = "dsv41-lanes-parallel round";
        var at: usize = 0;
        for (&rws, 0..) |*rw, i| {
            const rows: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_depth", .{i})) + 1);
            const l = &st.lanes[slot_of[i]];
            rw.* = .{ .slot = slot_of[i], .base = l.base, .size = l.size, .position = l.pos, .ids = window_ids[at..][0..rows] };
            starts[i] = l.pos;
            at += rows;
        }
        try check.expect(at == window_ids.len, "{s}: the windows' rows", .{label});
        var rec: L.Round = undefined;
        try L.Round.parse(L.Round.encode(op_buf, &rws), &rec);
        try check.expect(st.roundable(rec.windows[0..rec.count]), "{s}: the record is runnable", .{label});
        const t0 = check.now(io);
        try st.round(gather, rec.windows[0..rec.count]);
        var picks: [L.max_rows]u32 = undefined;
        try st.argmaxRows(0, picks[0..at]);
        const ns = check.now(io) - t0;
        var ok = supply.next == marks[3];
        if (!ok) std.debug.print("FAIL {s}: {d} gathers, the fork's {d}\n", .{ label, supply.next, marks[3] });
        ok = (try ck.same(label, "R.logits", st.logits, at * V * 4, 0, &.{})) and ok;
        ok = (try ck.same(label, "R.taps", st.runner.taps.ptr, at * cfg.dspark_taps.len * cfg.dim * 2, 0, &.{})) and ok;
        const tb = try fx.bytes("expected_R.target");
        defer gpa.free(tb);
        const targets = std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(tb)));
        for (picks[0..at], targets) |p, t| if (p != t) {
            std.debug.print("FAIL {s}: picks {any}, the fork's {any}\n", .{ label, picks[0..at], targets });
            ok = false;
            break;
        };
        var r0: usize = 0;
        for (rws, 0..) |rw, i| {
            var acc: usize = 0;
            while (acc + 1 < rw.ids.len and rw.ids[acc + 1] == picks[r0 + acc]) acc += 1;
            const want_acc: usize = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "s{d}_accepted", .{i})));
            if (acc != want_acc) {
                std.debug.print("FAIL {s}: stream {d} accepted {d}, the fork's {d}\n", .{ label, i, acc, want_acc });
                ok = false;
            }
            accepted[i] = acc;
            r0 += rw.ids.len;
        }
        if (ok) check.pass("BITEXACT {s}: windows of {d} and {d} rows at {d} and {d} in one forward ({d} ms): every row's logits, the taps, the picks and the accepted counts {d} and {d} equal to the Python fork's", .{ label, rws[0].ids.len, rws[1].ids.len, starts[0], starts[1], @divTrunc(ns, 1_000_000), accepted[0], accepted[1] }) else ck.failures += 1;
    }

    // -- the keep and the next drafts (absorbing the kept rows) ----------------------------------------------------
    {
        const label = "dsv41-lanes-parallel keep + drafts 1";
        // stream 0 keeps through a keeps step; stream 1's keep is its draft's absorb (the lane core's shared order)
        const kept0 = accepted[0] + 1;
        var kasks = [1]L.KeepAsk{.{ .slot = slot_of[0], .start = starts[0], .kept = kept0 }};
        var krec: L.Keeps = undefined;
        try L.Keeps.parse(L.Keeps.encode(op_buf, &kasks), &krec);
        for (krec.asks[0..krec.count]) |k| try st.keep(k);
        var asks: [2]L.DraftAsk = undefined;
        for (&asks, 0..) |*a, i| {
            const kept = accepted[i] + 1;
            const pending: i64 = try fx.int(try std.fmt.bufPrint(&nb, "s{d}_pending", .{i}));
            a.* = .{ .slot = slot_of[i], .start = starts[i], .absorb = kept, .token = pending, .q0 = starts[i] + kept, .depth = block };
        }
        var rec: L.DraftsRec = undefined;
        try L.DraftsRec.parse(L.DraftsRec.encode(op_buf, &asks), &rec);
        try check.expect(st.draftable(rec.asks[0..rec.count]), "{s}: the record is runnable", .{label});
        try st.draft(rec.asks[0..rec.count]);
        try stream.synchronize();
        var ok = supply.next == marks[4];
        if (!ok) std.debug.print("FAIL {s}: {d} gathers, the fork's {d}\n", .{ label, supply.next, marks[4] });
        for (0..2) |i| ok = ok and st.lanes[slot_of[i]].pos == starts[i] + accepted[i] + 1;
        ok = (try ck.pool(label, "K", &st)) and ok;
        ok = (try ck.drafts(label, "D1", &st, &slot_of, block)) and ok;
        if (ok) check.pass("BITEXACT {s}: the streams at {d} and {d}, the pool's planes and the drafter rings after the absorbs, the next drafts {any} and {any} equal to the Python fork's", .{ label, st.lanes[slot_of[0]].pos, st.lanes[slot_of[1]].pos, st.lanes[slot_of[0]].drafts.tokens[0..block], st.lanes[slot_of[1]].drafts.tokens[0..block] });
    }
    try check.expect(supply.next == supply.other.items.len, "{d} of the fork's {d} gathers ran", .{ supply.next, supply.other.items.len });
    st.free(slot_of[0]);
    st.free(slot_of[1]);
    try check.expect(st.idle() and st.extents.n == 1, "the slots and the window are free again", .{});

    // -- timing (this rank's layers, gathers local) and a stream's bytes at 1M positions ---------------------------
    supply.loose = true;
    // the graphs recorded with the fixture's partials read the mailbox: the timed ones are recorded with the local gathers
    st.runner.dropGraphs();
    drafter.dropGraphs();
    if (st.runner.graphs) {
        // the served engine's warm-up at load (Streams.warm): every round and batched draft graph it replays
        const t0 = check.now(io);
        const free0 = (try gpu.ctx.memInfo()).free;
        try st.warm(gather);
        const free1 = (try gpu.ctx.memInfo()).free;
        std.debug.print("dsv41-lanes-parallel rank {d}: warm-up recorded {d} round graphs (1..16 rows, buckets up to {d}) and {d} draft graphs in {d} ms through {d} layers ({d:.1} MiB less free memory)\n", .{ rank, st.runner.graphCount(), @min(L.warm_bucket_max, st.runner.bucket), drafter.captures, @divTrunc(check.now(io) - t0, 1_000_000), layers, @as(f64, @floatFromInt(@as(i64, @intCast(free0)) - @as(i64, @intCast(free1)))) / (1 << 20) });
    }
    var round_ms: [L.max_rows]f64 = undefined;
    for (1..L.max_rows + 1) |n| {
        var best: f64 = std.math.inf(f64);
        for (0..4) |rep| {
            try stream.synchronize();
            const t0 = check.now(io);
            try st.probeRound(gather, n);
            try stream.synchronize();
            if (rep > 0) best = @min(best, @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6);
        }
        round_ms[n - 1] = best;
    }
    var draft_ms: [3]f64 = undefined;
    for (1..4) |n| {
        var best: f64 = std.math.inf(f64);
        for (0..4) |rep| {
            try stream.synchronize();
            const t0 = check.now(io);
            try st.probeDrafts(n);
            try stream.synchronize();
            if (rep > 0) best = @min(best, @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6);
        }
        draft_ms[n - 1] = best;
    }
    std.debug.print("dsv41-lanes-parallel rank {d}: rounds of 1..16 rows {any} ms, batched drafts of 1..3 streams {any} ms through {d} layers (gathers local)\n", .{ rank, round_ms, draft_ms, layers });
    const mcap = (1 << 20) + L.extra;
    const pool_bytes = dsv41.rounds.Pool.bytesFor(&cfg, cfg.n_layers, mcap, L.max_slots);
    const rings = L.max_slots * cfg.nextn * (cfg.window + 16) * cfg.head_dim * 2;
    const per = L.Streams.bytesPerSlot(&cfg, cfg.n_layers, mcap, L.max_slots, cfg.nextn);
    std.debug.print("dsv41-lanes-parallel: at a 1M-position context and {d} slots over {d} layers: the pool {d} bytes ({d:.1} MiB), the drafter rings {d} bytes ({d:.1} MiB), stream_bytes {d} ({d:.1} MiB)\n", .{ L.max_slots, cfg.n_layers, pool_bytes, mib(pool_bytes), rings, mib(rings), per, mib(per) });
    try check.expect(ck.failures == 0, "{d} concurrent step(s) differ from the fork", .{ck.failures});
}

fn mib(b: usize) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 20);
}

fn upload(gpu: Gpu, fx: Fixture, name: []const u8) !cuda.DeviceBuffer {
    const bytes = try fx.bytes(name);
    defer gpu.gpa.free(bytes);
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}
