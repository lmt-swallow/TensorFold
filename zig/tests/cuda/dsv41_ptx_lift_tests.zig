//! A kernel set lifted from the captured Triton PTX (tools/zig/dsv41/ptx_lift.py) against the captured set it came
//! from: the same variants (function, specialization hash, entry name, runtime parameters, constexprs, PDL), every
//! lifted cubin loaded by the driver under the original entry name with the launch shape its aot.json states, and
//! per variant the registers, spill (local) bytes and static shared bytes of both. The outputs' bit-exactness is the
//! fixture tests' to show: `dsv41-rounds <dir> <rank-cache-dir> <model-dir> <lifted-set>` runs a round on the lifted
//! cubins in place of the captured ones.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Attrs = struct { regs: c_int, local: c_int, static_shared: c_int, max_threads: c_int };

fn attrs(f: anytype) !Attrs {
    return .{
        .regs = try f.attribute(.num_regs),
        .local = try f.attribute(.local_size_bytes),
        .static_shared = try f.attribute(.shared_size_bytes),
        .max_threads = try f.attribute(.max_threads_per_block),
    };
}

fn sameParams(a: anytype, b: anytype) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!std.mem.eql(u8, x.name, y.name) or !std.mem.eql(u8, x.type, y.type) or x.div16 != y.div16 or x.nospec != y.nospec) return false;
    }
    return true;
}

fn sameConsts(a: anytype, b: anytype) bool {
    if (a.map.count() != b.map.count()) return false;
    var it = a.map.iterator();
    while (it.next()) |e| {
        const o = b.map.get(e.key_ptr.*) orelse return false;
        if (!std.meta.eql(e.value_ptr.*, o)) return false;
    }
    return true;
}

/// `dsv41-ptx-lift <captured-set> <lifted-set>`: every captured variant has its lifted counterpart, loadable, without
/// spills at Triton's launch shape; a lifted variant runs at least as many threads as Triton launched (virt: twice).
pub fn ptxLift(gpu: Gpu, orig_dir: []const u8, lifted_dir: []const u8) !void {
    var orig = try cuda.aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, orig_dir);
    defer orig.deinit();
    var lifted = try cuda.aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, lifted_dir);
    defer lifted.deinit();
    check.pass("dsv41 ptx-lift: {d} captured and {d} lifted variants loaded", .{ orig.variants.len, lifted.variants.len });
    try check.expect(orig.variants.len == lifted.variants.len, "dsv41 ptx-lift: {d} captured variants, {d} lifted", .{ orig.variants.len, lifted.variants.len });
    var spills: usize = 0;
    var reg_diff: usize = 0;
    for (orig.variants) |*o| {
        const l = for (lifted.variants) |*v| {
            if (std.mem.eql(u8, v.spec.hash, o.spec.hash)) break v;
        } else {
            std.debug.print("FAIL dsv41 ptx-lift: {s} {s} has no lifted variant\n", .{ o.spec.@"fn", o.spec.hash });
            return error.TestFailed;
        };
        const os = o.spec;
        const ls = l.spec;
        try check.expect(std.mem.eql(u8, os.@"fn", ls.@"fn") and std.mem.eql(u8, os.name, ls.name) and os.pdl == ls.pdl and
            os.num_ctas == ls.num_ctas and sameParams(os.params, ls.params) and sameConsts(os.consts, ls.consts) and
            ls.num_warps >= os.num_warps and ls.shared >= os.shared, "dsv41 ptx-lift: {s} {s}: the lifted variant's spec differs", .{ os.@"fn", os.hash[0..12] });
        const a = try attrs(o.kernel.function);
        const b = try attrs(l.kernel.function);
        try check.expect(b.max_threads >= 32 * @as(c_int, @intCast(ls.num_warps)), "dsv41 ptx-lift: {s} {s}: {d} threads a block at most, the set launches {d}", .{ os.@"fn", os.hash[0..12], b.max_threads, 32 * ls.num_warps });
        // a virt set launches twice the threads, so its register cap halves; spills count only at the same shape
        spills += @intFromBool(ls.num_warps == os.num_warps and b.local > a.local);
        reg_diff += @intFromBool(b.regs != a.regs);
        std.debug.print("  {s:<24} {s} warps {d}->{d} shared {d}->{d} regs {d}->{d} local {d}->{d} static shared {d}->{d}\n", .{
            os.@"fn", os.hash[0..12], os.num_warps, ls.num_warps, os.shared, ls.shared, a.regs, b.regs, a.local, b.local, a.static_shared, b.static_shared,
        });
    }
    try check.expect(spills == 0, "dsv41 ptx-lift: {d} lifted variants spill at the captured launch shape where the captured did not", .{spills});
    check.pass("dsv41 ptx-lift: {d} variants: specs equal, loaded, no new spills; registers differ in {d}", .{ orig.variants.len, reg_diff });
}
