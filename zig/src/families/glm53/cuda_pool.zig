//! The streams' shared cache pool (Mia's pool.py `Arena` over forward.py `Caches`' planes): per DSA layer, then the
//! MTP layer, a latent plane [rows, kv_lora] and a pooled-key plane [rows / 4 + 2, index_dim] (TF_GLM_KV's rows), all zeroed
//! once. A stream's caches view its extent [base, base + size) of every plane (`Cache.rebind`); extents start and end
//! on multiples of 2,048 positions (so an extent's pooled keys start at a whole row, base / 4), first fit, and a freed
//! extent merges with its neighbours (the same `Extents` the DeepSeek-V4.1 pool uses).
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const dsa = @import("cuda_dsa.zig");
pub const Extents = @import("dsv41").kept_prompts.Extents;
pub const extent_align = @import("dsv41").kept_prompts.extent_align;

const Buf = cuda.DeviceBuffer;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    latent: []Buf,
    pk: []Buf,
    rows: usize,
    kv: dsa.Kv,
    extents: Extents,

    /// Planes for `planes` DSA layers (the backbone's, then the MTP layer's) over `positions` positions (aligned down
    /// to `extent_align`).
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: *const config.Config, planes: usize, positions: usize, kv: dsa.Kv) !Pool {
        const rows = positions / extent_align * extent_align;
        if (rows == 0) return error.BadPool;
        const latent = try gpa.alloc(Buf, planes);
        errdefer gpa.free(latent);
        const pk = try gpa.alloc(Buf, planes);
        errdefer gpa.free(pk);
        var made: usize = 0;
        errdefer for (latent[0..made], pk[0..made]) |*a, *b| {
            a.free();
            b.free();
        };
        for (latent, pk) |*a, *b| {
            a.* = try Buf.alloc(d, rows * kv.rowBytes(c.kv_lora));
            errdefer a.free();
            b.* = try Buf.alloc(d, (rows / 4 + 2) * kv.rowBytes(c.index_dim));
            errdefer b.free();
            try a.fill8(0, null);
            try b.fill8(0, null);
            made += 1;
        }
        return .{ .gpa = gpa, .latent = latent, .pk = pk, .rows = rows, .kv = kv, .extents = Extents.init(rows) };
    }

    pub fn deinit(p: *Pool) void {
        for (p.latent, p.pk) |*a, *b| {
            a.free();
            b.free();
        }
        p.gpa.free(p.latent);
        p.gpa.free(p.pk);
    }

    /// Device bytes of a pool of `planes` planes over `positions` positions.
    pub fn bytes(c: *const config.Config, planes: usize, positions: usize, kv: dsa.Kv) usize {
        const rows = positions / extent_align * extent_align;
        return planes * (rows * kv.rowBytes(c.kv_lora) + (rows / 4 + 2) * kv.rowBytes(c.index_dim));
    }

    /// Plane `i`'s rows of the extent [base, base + size) bound to `cache` (base and size multiples of the alignment).
    pub fn bind(p: *const Pool, i: usize, cache: *dsa.Cache, base: usize, size: usize) !void {
        if (base % extent_align != 0 or size % extent_align != 0 or base + size > p.rows) return error.BadExtent;
        const c_lat = p.latent[i].len / p.rows; // bytes a position
        const c_pk = (p.pk[i].len / (p.rows / 4 + 2)); // bytes a pooled row
        const lat: Buf = .{ .d = p.latent[i].d, .ptr = p.latent[i].ptr + base * c_lat, .len = size * c_lat };
        const pk: Buf = .{ .d = p.pk[i].d, .ptr = p.pk[i].ptr + base / 4 * c_pk, .len = (size / 4 + 2) * c_pk };
        cache.rebind(lat, pk, size);
    }
};

test "an extent's views stay inside the planes, the pooled keys' pad reaching into the next extent" {
    // the arithmetic `bind` does, on a pool of 3 extents
    const rows = 3 * extent_align;
    var ex = Extents.init(rows);
    const a = ex.take(1).?;
    const b = ex.take(extent_align + 1).?;
    try std.testing.expectEqual(@as(usize, 0), a);
    try std.testing.expectEqual(extent_align, b);
    try std.testing.expect(ex.take(1) == null);
    // b's pooled keys end at its last pool plus 2 pad rows: within the plane's rows / 4 + 2
    const size_b = Extents.size(extent_align + 1);
    try std.testing.expect(b / 4 + size_b / 4 + 2 <= rows / 4 + 2);
    ex.give(a, 1);
    try std.testing.expectEqual(@as(?usize, 0), ex.take(extent_align));
}
