//! GLM-5.3-Flash (the EXL3 pack, two DGX Sparks over NCCL) behind the native family interface: rank 0 serves the
//! lane core, rank 1 follows its steps (cuda_lanes.zig). The ranks' settings come from the environment (`--tp/--rank/
//! --master/--master-port` export them: TF_TP_WORLD, TF_TP_RANK, TF_TP_MASTER, TF_TP_PORT); the kernel set
//! (TENSORFOLD_CUDA_KERNELS) must hold the reference's captured Triton kernels (tools/zig/glm53/capture.sh). Up to 4
//! streams decode together, their DSA caches in one pool of TF_GLM_POOL positions (default: one stream's context;
//! at least that, a multiple of 2,048), each stream an extent of it; every slot's other state (KDA states, rings, the
//! head's rows) is made at load. TF_GLM_TIER=<dir> (bf16 or FP8 caches) puts the DSA latent rows on a tier (cuda_tier.zig):
//! a pinned slab of TF_GLM_TIER_GIB GiB of resident pools (default: every slot's at a full context, at most 24), the
//! rest in <dir>/glm53-kv-rank<r>.bin.
//! Both ranks must run with the same --context, TF_GLM_POOL, TF_GLM_TIER and drafts setting.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const plan = @import("plan.zig");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const weights = @import("cuda_weights.zig");
const moe = @import("cuda_moe.zig");
const dsa = @import("cuda_dsa.zig");
const mtp = @import("cuda_mtp.zig");
const M = @import("cuda_model.zig");
const engine = @import("cuda_engine.zig");
const lanesmod = @import("cuda_lanes.zig");
const P = @import("cuda_pool.zig");
const Tr = @import("cuda_tier.zig");
const keep = @import("cuda_keep.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 16384;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
/// Rounds of the decoding streams after each 2,048-row chunk of another stream's prompt pass.
pub const prefill_interleave: u32 = 1;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    pool_positions: usize = 0,
    pool_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
};

/// Everything a rank's engine holds, at fixed addresses (the runner and the lanes point into it).
const Owned = struct {
    gpa: std.mem.Allocator,
    cfg: config.Config,
    set: cuda.aot.Set,
    hc: glue.Hc,
    kk: kda.Kernels,
    mk: moe.Kernels,
    tk: mtp.Kernels,
    model: M.Model,
    head: mtp.Mtp,
    comm: cuda.comm.Comm,
    stream: cuda.Stream,
    runner: engine.Runner,
    pool: P.Pool,
    tier: ?*Tr.Tier,
    lanes: *lanesmod.Lanes,
};

fn getenv(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    return if (v.len == 0) null else v;
}

fn envInt(name: [:0]const u8, default: u32) !u32 {
    const v = getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch error.BadEnvironment;
}

/// Whether this process is a follower (rank > 0), from the same environment `open` reads.
pub fn isFollower() bool {
    return (envInt("TF_TP_RANK", 0) catch 0) != 0;
}

/// A stream's positions at most for a context of `context` tokens: the context and a verify window, a whole extent.
pub fn capacityFor(context: usize) usize {
    return P.Extents.size(context + lanesmod.max_rows + 1);
}

/// TF_GLM_POOL's positions for streams of `capacity` positions: unset, `capacity`; else a whole number at least it
/// and a multiple of the extent alignment (error.BadPool otherwise).
pub fn poolSetting(text: ?[]const u8, capacity: usize) error{BadPool}!usize {
    const v = text orelse return capacity;
    const pool = std.fmt.parseInt(usize, std.mem.trim(u8, v, " "), 10) catch return error.BadPool;
    if (pool < capacity or pool % P.extent_align != 0) return error.BadPool;
    return pool;
}

/// This rank's device bytes of weights (its share of the pack's text tensors and its head rows).
pub fn weightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    var why: config.Why = .{};
    const cfg = config.read(a, io, dir, &why) catch return 0;
    const rank = envInt("TF_TP_RANK", 0) catch return 0;
    var b: plan.Bytes = .{};
    const Ctx = struct { b: *plan.Bytes, lay: config.Layout };
    plan.visit(&cfg, Ctx{ .b = &b, .lay = .{ .rank = rank, .world = config.world, .cfg = &cfg } }, struct {
        fn f(x: Ctx, w: plan.Want) !void {
            try x.b.add(x.lay, w);
        }
    }.f) catch return 0;
    return b.total;
}

fn gatherFn(p: *anyopaque, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, floats: usize, stream: cuda.Stream) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(p));
    try own.comm.allGather(send, recv, floats, .f32, stream.handle);
}

/// Device bytes `x` owns: every `cuda.DeviceBuffer` among its fields, arrays, owned slices and set optionals, recursively.
/// Pointers are not followed (they name what another owner counts), and a DSA cache bound to a pool's extent counts its
/// rings only (the pool counts its rows).
fn deviceBytes(x: anytype) usize {
    const T = @TypeOf(x);
    if (T == cuda.DeviceBuffer) return x.len;
    if (T == dsa.Cache) return x.ik.len + x.ig.len + if (x.owned) x.latent.len + x.pk.len else 0;
    // the MTP prompt buffers' borrowed partials, gathers and split-K slices are the backbone's (`initSharing`)
    if (T == mtp.Buffers and x.borrowed) {
        var n: usize = 0;
        inline for (@typeInfo(T).@"struct".field_names) |name| {
            if (comptime !(std.mem.eql(u8, name, "part") or std.mem.eql(u8, name, "gath") or std.mem.eql(u8, name, "sk"))) n += deviceBytes(@field(x, name));
        }
        return n;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |st| {
            var n: usize = 0;
            inline for (st.field_names) |name| n += deviceBytes(@field(x, name));
            return n;
        },
        .array => {
            var n: usize = 0;
            for (x) |e| n += deviceBytes(e);
            return n;
        },
        .optional => return if (x) |v| deviceBytes(v) else 0,
        .pointer => |pt| {
            if (pt.size != .slice or @typeInfo(pt.child) == .pointer) return 0;
            var n: usize = 0;
            for (x) |e| n += deviceBytes(e);
            return n;
        },
        else => return 0,
    }
}

fn inGib(n: usize) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

/// The device's free bytes between `open`'s phases: what each phase took, beside what the engine's buffers of that
/// phase account for (the rest is the driver's, NCCL's, the modules' and the allocator's).
/// The phases are measured by /proc/meminfo's MemAvailable, which counts the page cache as available (the unified
/// memory's device allocations come out of it), and by cuMemGetInfo's free, which does not (a weight load's page
/// cache shows there, and comes back when an allocation evicts it).
const Phases = struct {
    ctx: *const cuda.Context,
    io: std.Io,
    gpa: std.mem.Allocator,
    rank: usize,
    last: usize,
    last_avail: usize,

    /// Field `key` (a "Name:" line in kB) of `path`, in bytes; 0 when unreadable.
    fn kb(p: *const Phases, path: []const u8, key: []const u8) usize {
        // /proc files report a size of 0: read them as a stream to the end
        var file = std.Io.Dir.cwd().openFile(p.io, path, .{}) catch return 0;
        defer file.close(p.io);
        var buf: [4096]u8 = undefined;
        var r = file.readerStreaming(p.io, &buf);
        const text = r.interface.allocRemaining(p.gpa, .limited(1 << 20)) catch return 0;
        defer p.gpa.free(text);
        var it = std.mem.splitScalar(u8, text, '\n');
        while (it.next()) |line| {
            if (!std.mem.startsWith(u8, line, key) or line.len <= key.len or line[key.len] != ':') continue;
            var f = std.mem.tokenizeAny(u8, line[key.len + 1 ..], " \t");
            const n = std.fmt.parseInt(usize, f.next() orelse return 0, 10) catch return 0;
            return n * 1024;
        }
        return 0;
    }

    fn available(p: *const Phases) usize {
        return p.kb("/proc/meminfo", "MemAvailable");
    }

    /// The process's host side: its resident set, its locked (pinned) pages, and the system's mapped and locked pages.
    fn host(p: *const Phases, name: []const u8) void {
        std.log.info("glm53 rank {d} memory: {s}: process VmRSS {d:.2} GiB, VmLck {d:.2}, VmPin {d:.2}; system Mapped {d:.2}, Mlocked {d:.2}, Shmem {d:.2}", .{
            p.rank,                                         name,                                          inGib(p.kb("/proc/self/status", "VmRSS")), inGib(p.kb("/proc/self/status", "VmLck")),
            inGib(p.kb("/proc/self/status", "VmPin")),      inGib(p.kb("/proc/meminfo", "Mapped")),        inGib(p.kb("/proc/meminfo", "Mlocked")),   inGib(p.kb("/proc/meminfo", "Shmem")),
        });
    }

    fn init(ctx: *const cuda.Context, io: std.Io, gpa: std.mem.Allocator, rank: usize) Phases {
        var p: Phases = .{ .ctx = ctx, .io = io, .gpa = gpa, .rank = rank, .last = 0, .last_avail = 0 };
        p.last_avail = p.available();
        const m = ctx.memInfo() catch return p;
        p.last = m.free;
        std.log.info("glm53 rank {d} memory: at start MemAvailable {d:.2} GiB, device free {d:.2} of {d:.2}", .{ rank, inGib(p.last_avail), inGib(m.free), inGib(m.total) });
        p.host("at start");
        return p;
    }

    /// Phase `name` is done; `counted`: the bytes its buffers hold.
    fn step(p: *Phases, name: []const u8, counted: usize) void {
        const m = p.ctx.memInfo() catch return;
        const avail = p.available();
        const took = @as(f64, @floatFromInt(p.last_avail)) - @as(f64, @floatFromInt(avail));
        const took_dev = @as(f64, @floatFromInt(p.last)) - @as(f64, @floatFromInt(m.free));
        std.log.info("glm53 rank {d} memory: {s}: MemAvailable took {d:.2} GiB (device free {d:.2}), its buffers {d:.2}, the rest {d:.2}; MemAvailable {d:.2}, device free {d:.2} GiB", .{
            p.rank, name, took / (1 << 30), took_dev / (1 << 30), inGib(counted), took / (1 << 30) - inGib(counted), inGib(avail), inGib(m.free),
        });
        p.last = m.free;
        p.last_avail = avail;
        p.host(name);
    }
};

/// What the loaded engine's device buffers hold, by class, in GiB (one log line each).
fn logClasses(own: *const Owned, rank: usize) void {
    var by_kind: [5]usize = @splat(0); // frames (hyper-connections, norms), KDA, DSA, dense MLP, MoE
    for (own.model.layers) |*l| {
        by_kind[0] += deviceBytes(l.frame);
        by_kind[1] += deviceBytes(l.kda);
        by_kind[2] += deviceBytes(l.dsa);
        by_kind[3] += deviceBytes(l.mlp);
        by_kind[4] += deviceBytes(l.moe);
    }
    std.log.info("glm53 rank {d} memory: weights: frames {d:.2}, KDA {d:.2}, DSA {d:.2}, dense MLP {d:.2}, MoE {d:.2}, embed + norm + head {d:.2}, MTP layer + draft head {d:.2} GiB", .{
        rank,                                                                                            inGib(by_kind[0]),            inGib(by_kind[1]), inGib(by_kind[2]), inGib(by_kind[3]), inGib(by_kind[4]),
        inGib(deviceBytes(own.model.embed) + deviceBytes(own.model.norm) + deviceBytes(own.model.head)), inGib(deviceBytes(own.head)),
    });
    const r = &own.runner;
    const w = &r.w;
    std.log.info("glm53 rank {d} memory: window buffers b {d:.2}, ds {d:.2}, ms {d:.2}, mb {d:.2}; prompt buffers pb {d:.2}, pds {d:.2}, pms {d:.2}, pmb {d:.2}; KDA temps {d:.2}; final norm and logits {d:.2}; staging, picks and split {d:.2} GiB", .{
        rank,                                                                       inGib(deviceBytes(w.b)),                                                                              inGib(deviceBytes(w.ds)),  inGib(deviceBytes(w.ms)),  inGib(deviceBytes(r.mb)),
        inGib(deviceBytes(w.pb)),                                                   inGib(deviceBytes(w.pds)),                                                                            inGib(deviceBytes(w.pms)), inGib(deviceBytes(r.pmb)), inGib(deviceBytes(w.temp) + deviceBytes(w.pscratch)),
        inGib(deviceBytes(w.fnormed) + deviceBytes(w.fxs) + deviceBytes(w.logits)), inGib(deviceBytes(r.stage_rows) + deviceBytes(r.picks) + deviceBytes(r.cand) + deviceBytes(r.split)),
    });
    var seqs: usize = 0;
    for (own.lanes.slots) |sl| seqs += deviceBytes(sl.seq);
    std.log.info("glm53 rank {d} memory: pool {d:.2}, stream slots (KDA states, rings, MTP backlog) {d:.2}, kept-prompt snapshots {d:.2} GiB", .{ rank, inGib(deviceBytes(own.pool.latent) + deviceBytes(own.pool.pk)), inGib(seqs), inGib(deviceBytes(own.lanes.snaps)) });
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    if (getenv("TF_GLM_DFLASH_MODEL") != null and (keep.settings().on or !o.drafts)) {
        std.log.err("TF_GLM_DFLASH_MODEL requires TF_GLM_KEEP=0 and drafts enabled", .{});
        return error.BadDflashServingSettings;
    }
    if (getenv("TF_GLM_DFLASH_MODEL") != null)
        _ = try @import("dflash2.zig").Policy.parse(getenv("TF_GLM_DFLASH_POLICY"));
    const world = try envInt("TF_TP_WORLD", 1);
    const rank = try envInt("TF_TP_RANK", 0);
    if (world != config.world or rank >= world) {
        std.log.err("glm53 serves on two ranks: start rank 0 and rank 1 with --tp 2 (TF_TP_WORLD=2)", .{});
        return error.WorldNotTwo;
    }
    const port: u16 = @intCast(try envInt("TF_TP_PORT", 29551));
    var master: [4]u8 = .{ 127, 0, 0, 1 };
    if (getenv("TF_TP_MASTER")) |m| {
        var it = std.mem.splitScalar(u8, m, '.');
        for (&master) |*b| b.* = std.fmt.parseInt(u8, it.next() orelse return error.BadMaster, 10) catch return error.BadMaster;
    }
    const d = ctx.d;
    var phases = Phases.init(ctx, io, gpa, rank);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    var why: config.Why = .{};
    own.cfg = config.read(gpa, io, dir, &why) catch |e| {
        std.log.err("glm53: {s}", .{why.text()});
        return e;
    };
    const lay: config.Layout = .{ .rank = rank, .world = world, .cfg = &own.cfg };
    own.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, kernels);
    errdefer own.set.deinit();
    own.hc = try glue.Hc.load(d);
    errdefer own.hc.deinit();
    own.kk = try kda.Kernels.load(d);
    errdefer own.kk.deinit();
    own.mk = try moe.Kernels.load(d);
    errdefer own.mk.deinit();
    own.tk = try mtp.Kernels.load(d);
    errdefer own.tk.deinit();
    phases.step("kernel modules (Triton set, hc, KDA, MoE, MTP)", 0);
    {
        var pack = try weights.Pack.open(gpa, io, dir);
        defer pack.close();
        const l: weights.Loader = .{ .gpa = gpa, .d = d, .pack = &pack, .lay = lay };
        own.model = try M.Model.load(l, own.cfg.layers);
        errdefer own.model.deinit();
        own.head = try mtp.Mtp.load(l);
        pack.dropCache();
    }
    errdefer own.model.deinit();
    errdefer own.head.deinit();
    phases.step("weights", deviceBytes(own.model.layers) + deviceBytes(own.model.embed) + deviceBytes(own.model.norm) + deviceBytes(own.model.head) + deviceBytes(own.head));
    // NCCL connects a channel's peers and allocates their buffers when a collective first needs them; with
    // NCCL_RUNTIME_CONNECT=0 it does so here, so a served process's memory stops growing once it has loaded (NCCL reads
    // its variables when the communicator is created; a value already set is kept)
    _ = setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    // four channels: at 1M x 4 with FP8 KV the communicator took 2.23 GiB a rank with NCCL's own choice and 0.50
    // with 4, while code decoded at 31.2 tok/s against 30.6 and a 32k prompt filled at 1,518 tok/s against 1,476
    _ = setenv("NCCL_MAX_NCHANNELS", "4", 0);
    // each rank loads on its own; the ranks meet here
    own.comm = try cuda.comm.Comm.init(rank, world, master, port, 1_800_000);
    errdefer own.comm.deinit();
    phases.step("NCCL communicator", 0);
    own.stream = try cuda.Stream.init(d, true);
    errdefer own.stream.deinit();
    const capacity = capacityFor(o.context);
    const k: M.Kernels = .{ .g = .{ .set = &own.set, .hc = &own.hc, .hidden = own.cfg.hidden, .streams = own.cfg.streams }, .kk = &own.kk, .mk = &own.mk };
    own.runner = try engine.Runner.init(gpa, d, lay, k, &own.tk, &own.model, &own.head, .{ .ctx = own, .gather = gatherFn }, own.stream, capacity);
    errdefer own.runner.deinit();
    phases.step("runner (window and prompt buffers)", deviceBytes(own.runner.w) + deviceBytes(own.runner.mb) + deviceBytes(own.runner.pmb) + deviceBytes(own.runner.stage_rows) + deviceBytes(own.runner.picks) + deviceBytes(own.runner.cand) + deviceBytes(own.runner.split));
    const positions = poolSetting(getenv("TF_GLM_POOL"), capacity) catch {
        std.log.err("TF_GLM_POOL is {s}: it takes a number of positions, at least a stream's {d}, a multiple of {d}", .{ getenv("TF_GLM_POOL").?, capacity, P.extent_align });
        return error.BadEnvironment;
    };
    const planes = own.cfg.count(.dsa) + 1;
    const kv = glue.Kv.parse(getenv("TF_GLM_KV") orelse "") catch {
        std.log.err("TF_GLM_KV is {s}: bf16 or fp8", .{getenv("TF_GLM_KV").?});
        return error.BadEnvironment;
    };
    own.tier = null;
    if (getenv("TF_GLM_TIER")) |tdir| {
        const kvt = Tr.kv;
        // default: every slot's latent rows at a full context (nothing evicted), at most 24 GiB (4 x 1M positions
        // keeps about half of them resident beside the weights and the pooled keys)
        const whole = lanesmod.max_streams * capacity * (own.cfg.count(.dsa) + 1) * own.cfg.kv_lora * 2;
        const gib = try envInt("TF_GLM_TIER_GIB", @intCast(@min(24, std.math.divCeil(usize, whole, 1 << 30) catch 24)));
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/glm53-kv-rank{d}.bin", .{ tdir, rank }, 0);
        defer gpa.free(path);
        const tcap = @max(std.mem.alignForward(usize, capacity, kvt.seg_positions), kvt.home_positions + kvt.seg_positions);
        // a slot's backbone and head streams, and the kept prompts' (each until the prompts reading through it end)
        const tstreams = 2 * (lanesmod.max_streams + keep.book.max_kept + 1);
        // map blocks: a backbone stream's DSA layers and its head stream's one, for every pair
        own.tier = try Tr.Tier.init(gpa, d, .{ .streams = tstreams, .layers = own.cfg.count(.dsa), .capacity = tcap, .row_bytes = kv.rowBytes(own.cfg.kv_lora), .blocks = tstreams / 2 * (own.cfg.count(.dsa) + 1) }, @as(usize, gib) << 30, engine.prompt_rows, glue.sparse_tokens, path);
    }
    errdefer if (own.tier) |t| t.deinit(gpa);
    own.pool = try P.Pool.init(gpa, d, &own.cfg, planes, positions, kv, own.tier == null);
    errdefer own.pool.deinit();
    phases.step("tier and pool", deviceBytes(own.pool.latent) + deviceBytes(own.pool.pk));
    own.lanes = try lanesmod.Lanes.init(gpa, d, lay, &own.runner, &own.comm, &own.pool, capacity, o.drafts, own.tier);
    errdefer own.lanes.deinit();
    if (getenv("TF_GLM_DFLASH_MODEL")) |draft_dir| try own.lanes.enableDflash(io, draft_dir);
    {
        var seqs: usize = 0;
        for (own.lanes.slots) |sl| seqs += deviceBytes(sl.seq);
        phases.step("stream slots and kept-prompt snapshots", seqs + deviceBytes(own.lanes.snaps));
    }
    logClasses(own, rank);
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = own.lanes.windowRows(),
        // the pool and every slot are allocated by the load: a stream needs no more
        .stream_bytes = 0,
        .pool_positions = positions,
        .pool_bytes = P.Pool.bytes(&own.cfg, planes, positions, kv, own.tier == null),
        .ctx = own,
        .deinit = release,
    };
}

/// Rank 1's loop: runs rank 0's steps until rank 0 stops; returns when the link closes.
pub fn follow(p: *anyopaque) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    try own.lanes.follow();
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.SamplingNotSupported => "this engine decodes greedily for now: send temperature 0",
        error.NoRoom => "every stream slot is busy, or the pool shared by the running requests has no room for this one: retry when one finishes, or lower max_tokens",
        error.OutOfStep => "the two ranks planned different steps (check both ranks' --context and --no-drafts); restart both ranks",
        error.UnportedPromptSplitSelection => "this prompt's length needs the prompt path's split block selection, which is not ported yet",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lanes.stop();
    own.lanes.deinit();
    own.pool.deinit();
    if (own.tier) |t| t.deinit(own.gpa);
    own.runner.deinit();
    own.stream.deinit();
    own.comm.deinit();
    own.head.deinit();
    own.model.deinit();
    own.tk.deinit();
    own.mk.deinit();
    own.kk.deinit();
    own.hc.deinit();
    own.set.deinit();
    own.gpa.destroy(own);
}

test "a stream's capacity holds the context and a window, a whole extent" {
    for ([_]usize{ 4096, 16384, 1 << 20, 100000 }) |c| {
        const cap = capacityFor(c);
        try std.testing.expect(cap > c + lanesmod.max_rows);
        try std.testing.expectEqual(@as(usize, 0), cap % P.extent_align);
    }
}

test "TF_GLM_POOL: a stream's capacity unless set; at least it, a multiple of the extent alignment" {
    const m: usize = 1 << 20;
    try std.testing.expectEqual(m, try poolSetting(null, m));
    try std.testing.expectEqual(4 * m, try poolSetting("4194304", m));
    try std.testing.expectError(error.BadPool, poolSetting("1046528", m));
    try std.testing.expectError(error.BadPool, poolSetting("2097153", m));
    try std.testing.expectError(error.BadPool, poolSetting("4M", m));
}
