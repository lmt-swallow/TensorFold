//! `glm53-shared <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir>`: four streams' windows in one forward
//! (cuda_model.zig `forwardSegs`) against each stream's windows alone, on both Sparks. The streams' prompts are
//! prefixes of the forward fixture's (some past the dense limit, some within it); every round each stream verifies a
//! window of its pending token and a few forced tokens (widths changing round to round, up to 32 rows in all) and
//! keeps one row. Every row's logits and sample must be the same in the shared forward as alone, and alone as on the
//! single-stream path the fixtures check (cuda_model.zig `forward`).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm53");
const check = @import("check.zig");
const fixture = @import("fixture.zig");
const Gpu = check.Gpu;

const Ptr = cuda.abi.DevicePtr;
const M = glm.model;
const T = glm.mtp;
const E = glm.engine;
const P = glm.pool;

const Link = struct {
    comm: *const cuda.comm.Comm,

    fn gather(p: *anyopaque, send: Ptr, recv: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(p));
        try l.comm.allGather(send, recv, floats, .f32, stream.handle);
    }
};

const streams = 4;
const rounds = 6;
/// Each stream's window rows by round (the pending token and forced tokens), at most 32 rows a round in all.
const widths = [rounds][streams]usize{ .{ 1, 1, 1, 1 }, .{ 1, 3, 8, 4 }, .{ 16, 8, 4, 4 }, .{ 2, 16, 1, 13 }, .{ 8, 8, 8, 8 }, .{ 1, 16, 1, 1 } };
/// The prompts' lengths: past the dense limit (2,051) and within it.
const lengths = [streams]usize{ 3000, 2100, 1200, 333 };

const Run = struct {
    logits: std.ArrayList(u8) = .empty,
    tokens: std.ArrayList(u32) = .empty,

    fn deinit(x: *Run, gpa: std.mem.Allocator) void {
        x.logits.deinit(gpa);
        x.tokens.deinit(gpa);
    }
};

const Mode = enum { single, alone, shared };

/// Every stream prefilled from 0 in its pool extent; then `rounds` rounds: `shared`, the streams' windows in one
/// forward a round; `alone`, one stream's window a forward (`forwardSegs` of one); `single`, `forward`'s own path. Each
/// stream's rows' logits and samples into its run.
fn decode(r: *E.Runner, pool: *const P.Pool, seqs: *[streams]E.Seq, prompts: [streams][]const u32, forced: []const u32, mode: Mode, runs: *[streams]Run) !void {
    const gpa = r.gpa;
    var pending: [streams]u32 = undefined;
    for (seqs, prompts, 0..) |*q, p, i| {
        try q.bind(pool, i * 8192, 8192);
        try q.reset(r.stream);
        var at: usize = 0;
        while (at < p.len) at = try r.fill(q, p, at);
        pending[i] = try r.first(q);
    }
    var ids: [streams][E.rows]u32 = undefined;
    var outs: [streams][E.rows]u32 = undefined;
    for (widths, 0..) |ws, round| {
        var wins: [streams]E.Runner.Window = undefined;
        for (0..streams) |i| {
            ids[i][0] = pending[i];
            for (ids[i][1..ws[i]], 0..) |*t, j| t.* = forced[(round * 17 + i * 5 + j) % forced.len];
            wins[i] = .{ .q = &seqs[i], .ids = ids[i][0..ws[i]], .out = &outs[i] };
        }
        switch (mode) {
            .shared => {
                try r.windows(&wins);
                for (0..streams) |i| try record(r, &runs[i], &wins, i, outs[i][0..ws[i]]);
            },
            .alone => for (0..streams) |i| {
                try r.windows(wins[i .. i + 1]);
                try record(r, &runs[i], wins[i .. i + 1], 0, outs[i][0..ws[i]]);
            },
            .single => for (0..streams) |i| {
                try r.window(&seqs[i], wins[i].ids, &outs[i]);
                try record(r, &runs[i], wins[i .. i + 1], 0, outs[i][0..ws[i]]);
            },
        }
        for (0..streams) |i| {
            try r.commit(&seqs[i], ws[i], 1);
            pending[i] = outs[i][0];
        }
    }
    _ = gpa;
}

/// Window `i` of the last shared forward: its rows' logits and samples.
fn record(r: *E.Runner, run_: *Run, wins: []const E.Runner.Window, i: usize, samples: []const u32) !void {
    var a0: usize = 0;
    for (wins[0..i]) |w| a0 += w.ids.len;
    const row = r.host.len * 2;
    const n = wins[i].ids.len;
    try r.stream.synchronize();
    const at = run_.logits.items.len;
    try run_.logits.resize(r.gpa, at + n * row);
    try r.w.logits.download(a0 * row, run_.logits.items[at..][0 .. n * row]);
    try run_.tokens.appendSlice(r.gpa, samples);
}

pub fn run(gpu: Gpu, dir: []const u8, model: []const u8, rank: usize, master: []const u8, port: u16, aot_dir: []const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var why: glm.config.Why = .{};
    const cfg = glm.config.read(gpa, gpu.io, model, &why) catch |e| {
        std.debug.print("FAIL glm53 config: {t}: {s}\n", .{ e, why.text() });
        return e;
    };
    const lay: glm.config.Layout = .{ .rank = rank, .world = 2, .cfg = &cfg };
    var set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const fx_dir = try std.fs.path.join(gpa, &.{ dir, "out", "fixture" });
    defer gpa.free(fx_dir);
    var fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir);
    defer fx.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var ip: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, master, '.');
    for (&ip) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    var comm = try cuda.comm.Comm.init(@intCast(rank), 2, ip, port, 600_000);
    defer comm.deinit();
    var hck = try glm.glue.Hc.load(d);
    defer hck.deinit();
    var kk = try glm.kda.Kernels.load(d);
    defer kk.deinit();
    var mk = try glm.moe.Kernels.load(d);
    defer mk.deinit();
    var tk = try T.Kernels.load(d);
    defer tk.deinit();
    var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
    defer pack.close();
    const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
    var mdl = try M.Model.load(l, cfg.layers);
    defer mdl.deinit();
    var mtp = try T.Mtp.load(l);
    defer mtp.deinit();
    pack.dropCache();
    var link: Link = .{ .comm = &comm };
    const k: M.Kernels = .{ .g = .{ .set = &set, .hc = &hck, .hidden = cfg.hidden, .streams = cfg.streams }, .kk = &kk, .mk = &mk };
    var r = try E.Runner.init(gpa, d, lay, k, &tk, &mdl, &mtp, .{ .ctx = &link, .gather = Link.gather }, stream, 8192);
    defer r.deinit();
    var pool = try P.Pool.init(gpa, d, &cfg, cfg.count(.dsa) + 1, streams * 8192, .bf16);
    defer pool.deinit();
    const ring = glm.dsa.indexRing(8192, E.prompt_rows);
    var seqs: [streams]E.Seq = undefined;
    for (&seqs) |*q| q.* = try E.Seq.initSlot(gpa, d, lay, cfg.layers, ring, .bf16);
    defer for (&seqs) |*q| q.deinit();

    var nb: [64]u8 = undefined;
    const prompt_b = try fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_prompt_ids", .{rank}));
    defer gpa.free(prompt_b);
    const ids = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(prompt_b)));
    var prompts: [streams][]const u32 = undefined;
    for (&prompts, lengths) |*p, n| p.* = ids[0..@min(n, ids.len)];
    const forced = ids[100..600];

    var runs: [3][streams]Run = @splat(@splat(.{}));
    defer for (&runs) |*m| for (m) |*x| x.deinit(gpa);
    for (std.enums.values(Mode), 0..) |mode, mi| try decode(&r, &pool, &seqs, prompts, forced, mode, &runs[mi]);
    var bad: usize = 0;
    const row = r.host.len * 2;
    const single = &runs[@intFromEnum(Mode.single)];
    for ([_]Mode{ .alone, .shared }) |mode| for (0..streams) |i| {
        const x = &single[i];
        const y = &runs[@intFromEnum(mode)][i];
        if (!std.mem.eql(u32, x.tokens.items, y.tokens.items)) {
            std.debug.print("FAIL glm53-shared rank {d}: stream {d} samples differ ({t} against the single path)\n", .{ rank, i, mode });
            bad += 1;
        }
        const n = x.logits.items.len / row;
        for (0..n) |j| if (!std.mem.eql(u8, x.logits.items[j * row ..][0..row], y.logits.items[j * row ..][0..row])) {
            std.debug.print("FAIL glm53-shared rank {d}: stream {d} logits row {d} differ ({t} against the single path)\n", .{ rank, i, j, mode });
            bad += 1;
            break;
        };
    };
    try check.expect(bad == 0, "glm53-shared rank {d}: a stream's rows differ in the shared forward", .{rank});
    var total: usize = 0;
    for (single) |x| total += x.tokens.items.len;
    check.pass("BITEXACT glm53-shared rank {d}: {d} streams, {d} rounds, {d} rows: every row's logits and sample the same on the single path, alone and shared", .{ rank, streams, rounds, total });
}
