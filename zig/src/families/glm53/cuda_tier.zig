//! The tiered DSA latent cache on CUDA (kv_tier.zig's device half): the slab and the map in mapped pinned memory
//! (one DRAM on GB10: the NVMe's direct reads and writes land where the kernels read, no copy), the selection read
//! back to the host, fetched, and remapped on the device. cuda_dsa's calls take their latent addresses from here:
//!
//! - before a window (rows pos .. pos + R - 1 of a stream, at a point where the stream's earlier calls have run):
//!   `beginWindow`;
//! - the latent write: `writes(s, l, pos, R, pos_dev)`, one or two (base, pos_dev, rows) pieces;
//! - dense attention (rows before the dense limit): `denseBase(s, l)`, positions [0, kv_tier.home_positions);
//! - sparse attention: after the selection wrote tokens [R, W] and counts [R], `resolve(...)` returns the base the
//!   remapped tokens index (the stream syncs once);
//! - after the round's tokens are known: `commit(s, accepted_end)`, `rewind(s, end)`.
//!
//! The kernels see the bytes a whole cache holds at the same positions in the same order, so the bits are those of
//! a memory-resident cache.
const std = @import("std");
const cuda = @import("cuda");
pub const kv = @import("kv_tier.zig");

const Ptr = cuda.abi.DevicePtr;

pub const Piece = struct { base: Ptr, pos_dev: Ptr, rows: usize };

pub const Tier = struct {
    d: *const cuda.Driver,
    host: kv.Tier,
    slab: cuda.HostBuffer,
    slab_dev: Ptr,
    map: cuda.HostBuffer,
    map_dev: Ptr,
    toks: cuda.HostBuffer, // the selection read back: tokens [rows, W] i32
    cnts: cuda.HostBuffer, // counts [rows] i32
    at: cuda.HostBuffer, // second pieces' first positions (i32, mapped), a ring
    at_dev: Ptr,
    at_next: usize = 0,
    module: cuda.Module,
    remap_fn: cuda.Function,

    /// `slab_bytes` of resident latents (whole runs of kv_tier.run_pools blocks), windows of up to `rows` rows
    /// selecting up to `w` positions a row, the backing file at `path`.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, o: kv.Options, slab_bytes: usize, rows: usize, w: usize, path: [:0]const u8) !*Tier {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const t = try gpa.create(Tier);
        errdefer gpa.destroy(t);
        t.d = d;
        t.at_next = 0;
        t.slab = try cuda.HostBuffer.allocMapped(d, slab_bytes);
        errdefer t.slab.free();
        t.slab_dev = try t.slab.device();
        t.map = try cuda.HostBuffer.allocMapped(d, o.streams * o.layers * o.capacity / kv.pool_rows * 4);
        errdefer t.map.free();
        t.map_dev = try t.map.device();
        t.toks = try cuda.HostBuffer.alloc(d, rows * w * 4);
        errdefer t.toks.free();
        t.cnts = try cuda.HostBuffer.alloc(d, rows * 4);
        errdefer t.cnts.free();
        t.at = try cuda.HostBuffer.allocMapped(d, ring * 4);
        errdefer t.at.free();
        t.at_dev = try t.at.device();
        t.module = try cuda.Module.load(d, cuda.kernels.glm53_tier);
        errdefer t.module.unload();
        t.remap_fn = try t.module.function("tf_glm53_tier_remap");
        const slab: []align(kv.block) u8 = @alignCast(t.slab.bytes);
        t.host = try kv.Tier.init(gpa, slab, t.map.slice(i32), path, o);
        return t;
    }

    /// Second pieces in flight at most before a slot of the position ring is written again (a window's layers
    /// each take one; the stream syncs at least once a window).
    const ring = 256;

    pub fn deinit(t: *Tier, gpa: std.mem.Allocator) void {
        t.host.deinit();
        t.module.unload();
        t.at.free();
        t.cnts.free();
        t.toks.free();
        t.map.free();
        t.slab.free();
        gpa.destroy(t);
    }

    pub fn open(t: *Tier, s: usize) !void {
        try t.host.open(s);
    }

    pub fn close(t: *Tier, s: usize) void {
        t.host.close(s);
    }

    pub fn beginWindow(t: *Tier, s: usize, pos: usize, R: usize) !void {
        try t.host.beginWindow(s, pos, R);
    }

    /// The latent write's pieces for rows pos .. pos + R - 1 of layer `l`: the first at `pos_dev` (the caller's
    /// device position of row 0), a second (past a segment's end) at a position the tier holds.
    pub fn writes(t: *Tier, s: usize, l: usize, pos: usize, R: usize, pos_dev: Ptr) ![2]?Piece {
        const p = try t.host.writeRuns(s, l, pos, R);
        var out: [2]?Piece = .{ null, null };
        for (p, 0..) |x, i| if (x) |q| {
            var at = pos_dev;
            if (i > 0) {
                const k = t.at_next % ring;
                t.at_next += 1;
                t.at.slice(i32)[k] = @intCast(q.pos);
                at = t.at_dev + k * 4;
            }
            out[i] = .{ .base = t.slab_dev +% @as(u64, @bitCast(q.base)), .pos_dev = at, .rows = q.rows };
        };
        return out;
    }

    pub fn denseBase(t: *const Tier, s: usize, l: usize) Ptr {
        return t.slab_dev +% @as(u64, @bitCast(t.host.denseBase(s, l)));
    }

    /// After the selection of rows [R] (tokens [R, w] positions, counts [R]) ran on `stream`: every selected pool
    /// resident, the tokens remapped in place; returns the base the sparse kernels read the latent rows at.
    pub fn resolve(t: *Tier, s: usize, l: usize, tokens: Ptr, counts: Ptr, R: usize, w: usize, stream: cuda.Stream) !Ptr {
        if (R * w * 4 > t.toks.bytes.len or R * 4 > t.cnts.bytes.len) return error.BadShape;
        const api = t.d.api;
        try t.d.check(api.cuMemcpyDtoHAsync_v2(t.cnts.bytes.ptr, counts, R * 4, stream.handle), "cuMemcpyDtoHAsync");
        try t.d.check(api.cuMemcpyDtoHAsync_v2(t.toks.bytes.ptr, tokens, R * w * 4, stream.handle), "cuMemcpyDtoHAsync");
        try stream.synchronize();
        try t.host.fetch(s, l, t.toks.slice(i32)[0 .. R * w], t.cnts.slice(i32)[0..R], w);
        var args: cuda.Args = .{};
        args.add(tokens);
        args.add(counts);
        args.add(t.map_dev + t.host.mapBase(s, l) * 4);
        args.add(@as(i32, @intCast(w)));
        try cuda.launch.launch(t.remap_fn, .{ .grid = .{ .x = @intCast(R) }, .block = .{ .x = 256 } }, stream, &args);
        return t.slab_dev;
    }

    pub fn commit(t: *Tier, s: usize, pos: usize) !void {
        try t.host.commit(s, pos);
    }

    pub fn rewind(t: *Tier, s: usize, pos: usize) void {
        t.host.rewind(s, pos);
    }

    pub fn prefetchRange(t: *Tier, s: usize, l: usize, end: usize) !void {
        try t.host.prefetchRange(s, l, end);
    }

    pub fn ensureRange(t: *Tier, s: usize, l: usize, end: usize) !void {
        try t.host.ensureRange(s, l, end);
    }
};

test {
    std.testing.refAllDecls(kv);
    std.testing.refAllDecls(Tier);
}
