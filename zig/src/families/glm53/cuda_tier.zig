//! The tiered DSA latent cache on CUDA (kv_tier.zig's device half): the slab and the map in mapped pinned memory
//! (one DRAM on GB10: the NVMe's direct reads and writes land where the kernels read, no copy), the selection read
//! back to the host, fetched, and remapped on the device. cuda_dsa's calls take their latent addresses from here:
//!
//! - before a window (rows pos .. pos + R - 1 of a stream, at a point where the stream's earlier calls have run):
//!   `beginWindow`;
//! - the latent write: `writes(s, l, pos, R, pos_dev)`, one or two (base, pos_dev, rows) pieces;
//! - dense attention (rows before the dense limit): `denseBase(s, l)`, positions [0, kv_tier.home_positions);
//! - sparse attention: after the selection wrote tokens [R, W] and counts [R], `resolve(...)` returns the base and
//!   the remapped rows the sparse kernels take (the stream syncs once when some written pool of the layer is on the file only; while
//!   the context fits, no sync);
//! - after the round's tokens are known: `commit(s, accepted_end)`, `rewind(s, end)`;
//! - a kept prompt continued by another stream: `alias(src, dst, rows)` (no copy past the home block; `src` stays
//!   while `dst` reads through it).
//!
//! The kernels see the bytes a whole cache holds at the same positions in the same order, so the bits are those of
//! a memory-resident cache.
const std = @import("std");
const cuda = @import("cuda");
pub const kv = @import("kv_tier.zig");

const Ptr = cuda.abi.DevicePtr;

/// A latent write: rows pos .. pos + rows - 1 at `base` (row p at base + p * row bytes), the first at `pos_dev`.
pub const Piece = struct { base: Ptr, pos_dev: Ptr, pos: usize, rows: usize };

pub const Tier = struct {
    d: *const cuda.Driver,
    host: kv.Tier,
    slab: cuda.HostBuffer,
    slab_dev: Ptr,
    map: cuda.HostBuffer,
    map_dev: Ptr,
    toks: cuda.HostBuffer, // the selection read back: tokens [rows, W] i32
    rows_out: cuda.DeviceBuffer, // the remapped selections: slab rows [rows, W] i32
    cnts: cuda.HostBuffer, // counts [rows] i32
    at: cuda.HostBuffer, // second pieces' first positions (i32 16 bytes apart, mapped: Triton specializes pointers on 16-byte alignment), a ring
    at_dev: Ptr,
    at_next: usize = 0,
    chains: cuda.HostBuffer, // the remap's chains (tier.cu: 10 int64 each, 128 bytes apart, mapped), a ring
    chains_dev: Ptr,
    chain_next: usize = 0,
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
        t.rows_out = try cuda.DeviceBuffer.alloc(d, rows * w * 4 + max_sels * 256);
        errdefer t.rows_out.free();
        t.cnts = try cuda.HostBuffer.alloc(d, rows * 4);
        errdefer t.cnts.free();
        t.at = try cuda.HostBuffer.allocMapped(d, ring * 16);
        errdefer t.at.free();
        t.at_dev = try t.at.device();
        t.chain_next = 0;
        t.chains = try cuda.HostBuffer.allocMapped(d, ring * 128);
        errdefer t.chains.free();
        t.chains_dev = try t.chains.device();
        t.module = try cuda.Module.load(d, cuda.kernels.glm53_tier);
        errdefer t.module.unload();
        t.remap_fn = try t.module.function("tf_glm53_tier_remap");
        const slab: []align(kv.block) u8 = @alignCast(t.slab.bytes);
        t.host = try kv.Tier.init(gpa, slab, t.map.slice(i32), path, o);
        return t;
    }

    /// Second pieces (and remap chains) in flight at most before a slot of their ring is written again: a window's
    /// layers take one each a stream, and the stream syncs at least once a window.
    const ring = 1024;

    pub fn deinit(t: *Tier, gpa: std.mem.Allocator) void {
        t.host.deinit();
        t.module.unload();
        t.chains.free();
        t.at.free();
        t.cnts.free();
        t.rows_out.free();
        t.toks.free();
        t.map.free();
        t.slab.free();
        gpa.destroy(t);
    }

    /// kv_tier.Tier.open: stream `s` with `layers` layers (the backbone's DSA layers, or 1 for the MTP head).
    pub fn open(t: *Tier, s: usize, layers: usize) !void {
        try t.host.open(s, layers);
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
                t.at.slice(i32)[k * 4] = @intCast(q.pos);
                at = t.at_dev + k * 16;
            }
            out[i] = .{ .base = t.slab_dev +% @as(u64, @bitCast(q.base)), .pos_dev = at, .pos = q.pos, .rows = q.rows };
        };
        return out;
    }

    pub fn denseBase(t: *const Tier, s: usize, l: usize) Ptr {
        return t.slab_dev +% @as(u64, @bitCast(t.host.denseBase(s, l)));
    }

    /// Selections a resolveMany takes at most.
    pub const max_sels = 64;

    /// One stream-layer's selection on the device: rows [R], tokens [R, w] positions, counts [R].
    pub const Sel = struct { h: usize, layer: usize, tokens: Ptr, counts: Ptr, R: usize };

    /// Where the sparse kernels read a resolved selection: latent rows at `base`, row indices at `tokens` [R, w]
    /// (the tier's buffer: valid until the next resolve; the selection's positions stay as they were).
    pub const Resolved = struct { base: Ptr, tokens: Ptr };

    /// After the selection of rows [R] (tokens [R, w] positions, counts [R]) ran on `stream`: every selected pool
    /// resident and the positions remapped to slab rows.
    pub fn resolve(t: *Tier, s: usize, l: usize, tokens: Ptr, counts: Ptr, R: usize, w: usize, stream: cuda.Stream) !Resolved {
        var out: [1]Resolved = undefined;
        try t.resolveMany(&.{.{ .h = s, .layer = l, .tokens = tokens, .counts = counts, .R = R }}, w, stream, &out);
        return out[0];
    }

    /// `resolve` of several streams' selections at once (a round's streams on one DSA layer): their missing pools
    /// read as one batch (the drive's latency paid once), then each remapped into out[i].
    pub fn resolveMany(t: *Tier, sels: []const Sel, w: usize, stream: cuda.Stream, out: []Resolved) !void {
        var all_resident = true;
        var rows: usize = 0;
        for (sels) |x| {
            all_resident = all_resident and t.host.resident(x.h, x.layer);
            rows += x.R;
        }
        if (rows * w * 4 > t.toks.bytes.len or rows * 4 > t.cnts.bytes.len or sels.len > max_sels or out.len < sels.len) return error.BadShape;
        // every written pool resident (the whole context fits): the device remap alone, no read-back, no sync
        if (!all_resident) {
            const api = t.d.api;
            var at: usize = 0;
            for (sels) |x| {
                try t.d.check(api.cuMemcpyDtoHAsync_v2(t.cnts.bytes.ptr + at * 4, x.counts, x.R * 4, stream.handle), "cuMemcpyDtoHAsync");
                try t.d.check(api.cuMemcpyDtoHAsync_v2(t.toks.bytes.ptr + at * w * 4, x.tokens, x.R * w * 4, stream.handle), "cuMemcpyDtoHAsync");
                at += x.R;
            }
            try stream.synchronize();
            var hs: [max_sels]kv.Tier.Selection = undefined;
            at = 0;
            for (sels, 0..) |x, i| {
                hs[i] = .{ .s = x.h, .l = x.layer, .tokens = t.toks.slice(i32)[at * w ..][0 .. x.R * w], .counts = t.cnts.slice(i32)[at..][0..x.R], .W = w };
                at += x.R;
            }
            try t.host.fetchMany(hs[0..sels.len]);
        }
        // each selection's rows 256-byte aligned (Triton specializes pointers on 16-byte alignment)
        var at: usize = 0;
        for (sels, out[0..sels.len]) |x, *o| {
            o.* = .{ .base = t.slab_dev, .tokens = t.rows_out.ptr + at };
            try t.remap(x, o.tokens, w, stream);
            at = std.mem.alignForward(usize, at + x.R * w * 4, 256);
        }
    }

    fn remap(t: *Tier, x: Sel, dst: Ptr, w: usize, stream: cuda.Stream) !void {
        var links: [kv.max_chain]kv.Tier.Link = undefined;
        const n = t.host.chain(x.h, &links);
        const k = t.chain_next % ring;
        t.chain_next += 1;
        const c = t.chains.slice(i64)[k * 16 ..][0..10];
        c[0] = @intCast(n);
        c[1] = kv.home_pools;
        for (0..kv.max_chain) |i| {
            c[2 + i] = if (i < n) @intCast(links[i].fork_pool) else 0;
            c[6 + i] = if (i < n) @intCast(t.map_dev + t.host.mapBase(links[i].s, x.layer) * 4) else 0;
        }
        var args: cuda.Args = .{};
        args.add(x.tokens);
        args.add(dst);
        args.add(x.counts);
        args.add(t.chains_dev + k * 128);
        args.add(@as(i32, @intCast(w)));
        try cuda.launch.launch(t.remap_fn, .{ .grid = .{ .x = @intCast(x.R) }, .block = .{ .x = 256 } }, stream, &args);
    }

    /// kv_tier.Tier.alias: stream `dst` continues `src`'s first `rows` positions (a kept prompt), reading through it.
    pub fn alias(t: *Tier, src: usize, dst: usize, rows: usize) !void {
        try t.host.alias(src, dst, rows);
    }

    /// kv_tier.Tier.favor: stream `s`'s pools stay ahead of the others' (on while its prompt is filled).
    pub fn favor(t: *Tier, s: usize, on: bool) void {
        t.host.favor(s, on);
    }

    /// kv_tier.Tier.acquire: a free stream opened with `layers` layers.
    pub fn acquire(t: *Tier, layers: usize) !usize {
        return t.host.acquire(layers);
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
