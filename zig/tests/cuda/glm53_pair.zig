//! What the GLM-5.3-Flash engine tests share: one rank's model, kernels and runner (`Rank`), and the two ranks' link
//! (`launch`): NCCL between the Sparks, or with master "local" both ranks as threads on this GPU, their all-gathers
//! meeting in host memory (rank 0's rows first, as NCCL's). A `layers` count loads the first layers only, so two
//! ranks fit one GPU; the properties the tests check hold for any count.
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

/// One rank's engine pieces at fixed addresses (`load` in place).
pub const Rank = struct {
    cfg: glm.config.Config,
    lay: glm.config.Layout,
    set: cuda.aot.Set,
    hck: glm.glue.Hc,
    kk: glm.kda.Kernels,
    mk: glm.moe.Kernels,
    tk: T.Kernels,
    mdl: M.Model,
    mtp: T.Mtp,
    stream: cuda.Stream,
    fx: fixture.Fixture,
    prompt_bytes: []u8, // the fixture's prompt ids (both ranks' prompt is the same: rank 0's copy)
    gpa: std.mem.Allocator,
    /// TF_GLM_TIER=<dir> (bf16): the latent rows on a tier of `tier_streams` streams whose slab holds the home
    /// blocks and a few runs more, the rest on <dir>/glm53-pair-r<rank>.bin.
    tier: ?*glm.tier.Tier = null,

    pub fn load(self: *Rank, gpu: Gpu, dir: []const u8, model: []const u8, rank: usize, aot_dir: []const u8, layers: ?usize) !void {
        const gpa = gpu.gpa;
        const d = gpu.d;
        var why: glm.config.Why = .{};
        self.cfg = glm.config.read(gpa, gpu.io, model, &why) catch |e| {
            std.debug.print("FAIL glm53 config: {t}: {s}\n", .{ e, why.text() });
            return e;
        };
        self.lay = .{ .rank = rank, .world = 2, .cfg = &self.cfg };
        self.set = try cuda.aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, aot_dir);
        const fx_dir = try std.fs.path.join(gpa, &.{ dir, "out", "fixture" });
        defer gpa.free(fx_dir);
        self.fx = try fixture.Fixture.open(gpa, gpu.io, fx_dir);
        self.gpa = gpa;
        self.prompt_bytes = try self.fx.bytes("r0_prompt_ids");
        self.stream = try cuda.Stream.init(d, true);
        self.hck = try glm.glue.Hc.load(d);
        self.kk = try glm.kda.Kernels.load(d);
        self.mk = try glm.moe.Kernels.load(d);
        self.tk = try T.Kernels.load(d);
        var pack = try glm.weights.Pack.open(gpa, gpu.io, model);
        defer pack.close();
        const l: glm.weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = self.lay };
        self.mdl = try M.Model.load(l, layers orelse self.cfg.layers);
        self.mtp = try T.Mtp.load(l);
        pack.dropCache();
        self.tier = null;
        if (std.c.getenv("TF_GLM_TIER")) |e| {
            const kvt = glm.tier.kv;
            var dsa_layers: usize = 0;
            for (self.mdl.layers) |x| dsa_layers += @intFromBool(x.dsa != null);
            const tl = @max(dsa_layers, 1);
            const path = try std.fmt.allocPrintSentinel(gpa, "{s}/glm53-pair-r{d}.bin", .{ std.mem.span(e), rank }, 0);
            defer gpa.free(path);
            const cap = @max(std.mem.alignForward(usize, tier_capacity, kvt.seg_positions), kvt.home_positions + kvt.seg_positions);
            const runs = tier_streams * tl * kvt.home_runs + 8;
            self.tier = try glm.tier.Tier.init(gpa, d, .{ .streams = tier_streams, .layers = tl, .capacity = cap, .row_bytes = self.cfg.kv_lora * 2, .recent_runs = 2 }, runs * kvt.run_pools * kvt.block, E.prompt_rows, glm.glue.sparse_tokens, path);
        }
    }

    /// Tier streams a test may open, and the positions each holds.
    pub const tier_streams = 8;
    pub const tier_capacity = 8192;

    /// Sequence `i`'s tier streams (backbone 2i, head 2i + 1), opened afresh (nothing without a tier).
    pub fn tierOpen(self: *Rank, q: *E.Seq, i: usize) !void {
        const t = self.tier orelse return;
        const nd = q.st.caches.len;
        try t.open(2 * i, @max(nd, 1));
        try t.open(2 * i + 1, 1);
        q.tier = .{ .t = t, .h = 2 * i, .hm = 2 * i + 1 };
    }

    pub fn deinit(self: *Rank) void {
        if (self.tier) |t| t.deinit(self.gpa);
        self.mtp.deinit();
        self.mdl.deinit();
        self.tk.deinit();
        self.mk.deinit();
        self.kk.deinit();
        self.hck.deinit();
        self.stream.deinit();
        self.gpa.free(self.prompt_bytes);
        self.fx.deinit();
        self.set.deinit();
    }

    pub fn kernels(self: *const Rank) M.Kernels {
        return .{ .g = .{ .set = &self.set, .hc = &self.hck, .hidden = self.cfg.hidden, .streams = self.cfg.streams }, .kk = &self.kk, .mk = &self.mk };
    }

    pub fn runner(self: *Rank, gpa: std.mem.Allocator, d: *const cuda.Driver, link: E.Link, capacity: usize) !E.Runner {
        return E.Runner.init(gpa, d, self.lay, self.kernels(), &self.tk, &self.mdl, &self.mtp, link, self.stream, capacity);
    }

    /// The fixture's prompt ids, read at `load` (a local run's ranks are threads that read no files).
    pub fn prompt(self: *const Rank) []const u32 {
        return std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(self.prompt_bytes)));
    }
};

const Nccl = struct {
    comm: *const cuda.comm.Comm,

    fn gather(p: *anyopaque, send: Ptr, recv: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const l: *Nccl = @ptrCast(@alignCast(p));
        try l.comm.allGather(send, recv, floats, .f32, stream.handle);
    }
};

/// Two ranks' all-gathers in host memory: each copies its rows to its slot, both wait, each reads both slots.
const Host = struct {
    slots: [2][]u8,
    arrived: std.atomic.Value(u32) = .init(0),
    gen: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn barrier(h: *Host) !void {
        const g = h.gen.load(.acquire);
        if (h.arrived.fetchAdd(1, .acq_rel) + 1 == 2) {
            h.arrived.store(0, .release);
            h.gen.store(g +% 1, .release);
            return;
        }
        while (h.gen.load(.acquire) == g) {
            if (h.failed.load(.acquire)) return error.PeerFailed;
            std.Thread.yield() catch {};
        }
    }
};

const HostLink = struct {
    h: *Host,
    rank: usize,
    d: *const cuda.Driver,

    fn gather(p: *anyopaque, send: Ptr, recv: Ptr, floats: usize, stream: cuda.Stream) anyerror!void {
        const l: *HostLink = @ptrCast(@alignCast(p));
        const n = floats * 4;
        if (n > l.h.slots[0].len) return error.GatherTooLarge;
        try stream.synchronize();
        const src: cuda.DeviceBuffer = .{ .d = l.d, .ptr = send, .len = n };
        try src.download(0, l.h.slots[l.rank][0..n]);
        try l.h.barrier();
        const dst: cuda.DeviceBuffer = .{ .d = l.d, .ptr = recv, .len = 2 * n };
        try dst.upload(0, l.h.slots[0][0..n]);
        try dst.upload(n, l.h.slots[1][0..n]);
        try l.h.barrier();
    }
};

/// What a pair test loads: the fixture's directory, the model's, the kernel set's, and the layers (null: all).
pub const Args = struct { dir: []const u8, model: []const u8, aot: []const u8, layers: ?usize };

/// Runs `body(args, gpu, rank, link)` for this process's rank over NCCL, or with master "local" for both ranks as
/// threads on this GPU (each gather at most `most_floats`), each rank loaded first on the calling thread (a rank's
/// files are read there).
pub fn launch(gpu: Gpu, rank: usize, master: []const u8, port: u16, most_floats: usize, args: Args, comptime body: fn (Args, Gpu, *Rank, E.Link) anyerror!void) !void {
    if (!std.mem.eql(u8, master, "local")) {
        var rk: Rank = undefined;
        try rk.load(gpu, args.dir, args.model, rank, args.aot, args.layers);
        defer rk.deinit();
        var ip: [4]u8 = undefined;
        var it = std.mem.splitScalar(u8, master, '.');
        for (&ip) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
        var comm = try cuda.comm.Comm.init(@intCast(rank), 2, ip, port, 600_000);
        defer comm.deinit();
        var link: Nccl = .{ .comm = &comm };
        return body(args, gpu, &rk, .{ .ctx = &link, .gather = Nccl.gather });
    }
    var rks: [2]Rank = undefined;
    try rks[0].load(gpu, args.dir, args.model, 0, args.aot, args.layers);
    defer rks[0].deinit();
    try rks[1].load(gpu, args.dir, args.model, 1, args.aot, args.layers);
    defer rks[1].deinit();
    var host: Host = .{ .slots = .{ try gpu.gpa.alloc(u8, most_floats * 4), try gpu.gpa.alloc(u8, most_floats * 4) } };
    defer for (host.slots) |s| gpu.gpa.free(s);
    var links: [2]HostLink = .{ .{ .h = &host, .rank = 0, .d = gpu.d }, .{ .h = &host, .rank = 1, .d = gpu.d } };
    var errs: [2]?anyerror = .{ null, null };
    const Thread = struct {
        fn run(a: Args, g: Gpu, rk: *Rank, l: *HostLink, h: *Host, err: *?anyerror) void {
            g.ctx.makeCurrent() catch |e| {
                err.* = e;
                h.failed.store(true, .release);
                return;
            };
            body(a, g, rk, .{ .ctx = l, .gather = HostLink.gather }) catch |e| {
                std.debug.print("glm53 local rank {d}: {t}\n", .{ rk.lay.rank, e });
                err.* = e;
                h.failed.store(true, .release);
            };
        }
    };
    var threads: [2]std.Thread = undefined;
    for (&threads, 0..) |*t, r| t.* = try std.Thread.spawn(.{}, Thread.run, .{ args, gpu, &rks[r], &links[r], &host, &errs[r] });
    for (threads) |t| t.join();
    for (errs) |e| if (e) |x| return x;
}
