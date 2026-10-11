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
    }

    pub fn deinit(self: *Rank) void {
        self.mtp.deinit();
        self.mdl.deinit();
        self.tk.deinit();
        self.mk.deinit();
        self.kk.deinit();
        self.hck.deinit();
        self.stream.deinit();
        self.fx.deinit();
        self.set.deinit();
    }

    pub fn kernels(self: *const Rank) M.Kernels {
        return .{ .g = .{ .set = &self.set, .hc = &self.hck, .hidden = self.cfg.hidden, .streams = self.cfg.streams }, .kk = &self.kk, .mk = &self.mk };
    }

    pub fn runner(self: *Rank, gpa: std.mem.Allocator, d: *const cuda.Driver, link: E.Link, capacity: usize) !E.Runner {
        return E.Runner.init(gpa, d, self.lay, self.kernels(), &self.tk, &self.mdl, &self.mtp, link, self.stream, capacity);
    }

    /// The fixture's prompt ids (rank `lay.rank`'s copy), gpa-owned bytes and their u32 view.
    pub fn prompt(self: *Rank) !struct { bytes: []u8, ids: []const u32 } {
        var nb: [64]u8 = undefined;
        const b = try self.fx.bytes(try std.fmt.bufPrint(&nb, "r{d}_prompt_ids", .{self.lay.rank}));
        return .{ .bytes = b, .ids = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(b))) };
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

/// Runs `body(ctx, gpu, rank, link)` for this process's rank over NCCL, or with master "local" for both ranks as
/// threads on this GPU (each gather at most `most_floats`).
pub fn launch(gpu: Gpu, rank: usize, master: []const u8, port: u16, most_floats: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), Gpu, usize, E.Link) anyerror!void) !void {
    if (!std.mem.eql(u8, master, "local")) {
        var ip: [4]u8 = undefined;
        var it = std.mem.splitScalar(u8, master, '.');
        for (&ip) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
        var comm = try cuda.comm.Comm.init(@intCast(rank), 2, ip, port, 600_000);
        defer comm.deinit();
        var link: Nccl = .{ .comm = &comm };
        return body(ctx, gpu, rank, .{ .ctx = &link, .gather = Nccl.gather });
    }
    var host: Host = .{ .slots = .{ try gpu.gpa.alloc(u8, most_floats * 4), try gpu.gpa.alloc(u8, most_floats * 4) } };
    defer for (host.slots) |s| gpu.gpa.free(s);
    var links: [2]HostLink = .{ .{ .h = &host, .rank = 0, .d = gpu.d }, .{ .h = &host, .rank = 1, .d = gpu.d } };
    var errs: [2]?anyerror = .{ null, null };
    const Thread = struct {
        fn run(c: @TypeOf(ctx), g: Gpu, r: usize, l: *HostLink, h: *Host, err: *?anyerror) void {
            g.ctx.makeCurrent() catch |e| {
                err.* = e;
                h.failed.store(true, .release);
                return;
            };
            body(c, g, r, .{ .ctx = l, .gather = HostLink.gather }) catch |e| {
                err.* = e;
                h.failed.store(true, .release);
            };
        }
    };
    var threads: [2]std.Thread = undefined;
    for (&threads, 0..) |*t, r| t.* = try std.Thread.spawn(.{}, Thread.run, .{ ctx, gpu, r, &links[r], &host, &errs[r] });
    for (threads) |t| t.join();
    for (errs) |e| if (e) |x| return x;
}
