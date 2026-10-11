//! Qwen3.8-Flash-Next on one CUDA GPU behind the native family interface (native/cuda.zig): the model, the captured
//! Triton set, the lane backend and the facts its round loop reads.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const fwd = @import("cuda_forward.zig");
const attn = @import("cuda_attn.zig");
const Lanes = @import("cuda_lanes.zig").Lanes;
const max_rows = @import("cuda_lanes.zig").max_rows;
const prompt_mod = @import("cuda_prompt.zig");

pub const model_type = "qwen4_exp";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 16384;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = prompt_mod.chunk_rows;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
};

const Owned = struct { gpa: std.mem.Allocator, io: std.Io, m: *fwd.Model, set: *cuda.aot.Set, lanes: *Lanes };

/// A stream's positions: the window plus a decode window's rows, a multiple of 64 (the indexer's block count then a
/// multiple of 16, the specialization the captured selection kernels were compiled for).
pub fn capacityFor(context: usize) usize {
    return std.mem.alignForward(usize, context + max_rows, 64);
}

/// Device bytes one stream's sequence takes at `capacity` positions (caches, DeltaNet states, window rows).
pub fn streamBytes(cfg: *const config.Config, capacity: usize) usize {
    var lin: usize = 0;
    for (cfg.kinds[0..cfg.layers]) |k| lin += @intFromBool(k == .linear);
    const att = cfg.layers - lin;
    const nb = (capacity + cfg.index_ratio - 1) / cfg.index_ratio;
    const cache = capacity * cfg.kv_heads * cfg.head_dim * 2 * 2 + capacity * cfg.index_dim * 2 + nb * cfg.index_dim * 2;
    const state = 2 * cfg.nv * cfg.dv * cfg.dk * 4 + (cfg.conv_kernel - 1) * cfg.convDim() * 2;
    const scratch = max_rows * (cfg.nk * cfg.dk * 4 + cfg.nv * cfg.dv * 2 + 2 * cfg.nv * 4) + max_rows * cfg.projWidth() * 2;
    return att * cache + lin * (state + scratch) + (cfg.ple_kernel - 1) * cfg.ngram * cfg.wide() * 2 + max_rows * cfg.wide() * 2;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    var why: config.Why = .{};
    const cfg = config.read(gpa, io, dir, &why) catch |e| {
        std.log.err("qwen38fn: {s}", .{why.text()});
        return e;
    };
    const set = try gpa.create(cuda.aot.Set);
    errdefer gpa.destroy(set);
    set.* = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, kernels);
    errdefer set.deinit();
    const m = try fwd.Model.load(gpa, io, ctx.d, dir, cfg, cfg.layers, true);
    errdefer m.deinit();
    const capacity = capacityFor(o.context);
    const sms: u32 = @intCast(try ctx.attribute(.multiprocessor_count));
    const l = try Lanes.init(gpa, ctx.d, m, set, capacity, sms);
    errdefer l.deinit();
    const own = try gpa.create(Owned);
    own.* = .{ .gpa = gpa, .io = io, .m = m, .set = set, .lanes = l };
    return .{
        .backend = l.backend(),
        .facts = l.facts(),
        .rows = 1,
        .stream_bytes = streamBytes(&m.cfg, capacity),
        .ctx = own,
        .deinit = release,
    };
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.SamplingNotPorted => "this engine decodes greedily for now: send temperature 0",
        error.SelectTilesNotPorted => "past 131,072 positions the indexer's tiled block selection is not ported yet: lower the context",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lanes.deinit();
    own.m.deinit();
    own.set.deinit();
    own.gpa.destroy(own.set);
    own.gpa.destroy(own);
}

test "capacity keeps the indexer's block count a multiple of 16" {
    for ([_]usize{ 4096, 16384, 262144, 100000 }) |c| {
        const cap = capacityFor(c);
        try std.testing.expect(cap >= c + max_rows);
        try std.testing.expectEqual(@as(usize, 0), (cap / 4) % 16);
    }
}
