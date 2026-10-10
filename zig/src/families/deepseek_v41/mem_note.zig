//! TF_DS_MEM_REPORT=1: `info:` lines of the device memory each load phase takes. Each note logs, since the previous
//! note, the bytes this process's DeviceBuffers and pinned HostBuffers gained (cuda.usage) and the drop in the
//! device's free memory (cuMemGetInfo: also counts what the driver allocates for modules, graphs and NCCL, and on
//! unified memory whatever the host took meanwhile). Off, a note costs one getenv.
const std = @import("std");
const cuda = @import("cuda");

var last_device: i64 = 0;
var last_host: i64 = 0;
var last_free: i64 = 0;
var started = false;

pub fn on() bool {
    const v = std.c.getenv("TF_DS_MEM_REPORT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Logs the phase `what` ended: the deltas since the last note (the first note: the totals so far).
pub fn note(d: *const cuda.Driver, comptime what: []const u8, args: anytype) void {
    if (!on()) return;
    const u = cuda.usage(false);
    var free: usize = 0;
    var total: usize = 0;
    _ = d.api.cuMemGetInfo_v2(&free, &total);
    const dev: i64 = @intCast(u.device);
    const host: i64 = @intCast(u.host);
    const fr: i64 = @intCast(free);
    var buf: [160]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, what, args) catch what;
    const mib = struct {
        fn of(x: i64) f64 {
            return @as(f64, @floatFromInt(x)) / (1 << 20);
        }
    }.of;
    if (!started) {
        std.log.info("mem: {s}: buffers {d:.1} MiB, pinned {d:.1} MiB, device free {d:.1} MiB of {d:.1} MiB", .{ name, mib(dev), mib(host), mib(fr), mib(@intCast(total)) });
    } else {
        std.log.info("mem: {s}: buffers {d:.1} MiB (total {d:.1}), pinned {d:.1} MiB (total {d:.1}), device free {d:.1} MiB (now {d:.1})", .{ name, mib(dev - last_device), mib(dev), mib(host - last_host), mib(host), mib(fr - last_free), mib(fr) });
    }
    started = true;
    last_device = dev;
    last_host = host;
    last_free = fr;
}
