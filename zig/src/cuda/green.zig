//! SM partitions (green contexts, CUDA 12.4+): streams whose kernels run on a provisioned subset of the device's SMs.
//! Work on such a stream leaves the other SMs to streams of the primary context, which may still use every SM. A
//! kernel's result does not change with the partition when its grid and arithmetic do not depend on how many SMs
//! run it (no co-residency assumption beyond what the partition holds, no SM count read on the device).

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const Stream = @import("stream.zig").Stream;

pub const Partition = struct {
    d: *const Driver,
    ctx: abi.GreenCtx,
    /// The SMs provisioned: at least the count asked for, rounded up to the device's partition granularity, and
    /// fewer than the device's SMs.
    sms: u32,

    /// A green context over at least `sms` of `device`'s SMs (rounded up as the driver requires). error.Invalid
    /// without the driver's green-context entry points, or when the rounded group would take every SM.
    pub fn init(d: *const Driver, device: abi.Device, sms: u32) Error!Partition {
        const g = d.green orelse return error.Invalid;
        if (sms == 0) return error.Invalid;
        var whole: abi.DevResource = .{};
        try d.check(g.cuDeviceGetDevResource(device, &whole, abi.dev_resource_sm), "cuDeviceGetDevResource");
        var group: [1]abi.DevResource = .{.{}};
        var rest: abi.DevResource = .{};
        var n: c_uint = 1;
        try d.check(g.cuDevSmResourceSplitByCount(&group, &n, &whole, &rest, 0, sms), "cuDevSmResourceSplitByCount");
        if (n != 1 or group[0].sm_count >= whole.sm_count) return error.Invalid;
        var desc: abi.DevResourceDesc = null;
        try d.check(g.cuDevResourceGenerateDesc(&desc, &group, 1), "cuDevResourceGenerateDesc");
        var ctx: abi.GreenCtx = null;
        try d.check(g.cuGreenCtxCreate(&ctx, desc, device, abi.green_ctx_default_stream), "cuGreenCtxCreate");
        return .{ .d = d, .ctx = ctx, .sms = group[0].sm_count };
    }

    /// A non-blocking stream on this partition's SMs at the default priority.
    pub fn stream(self: *const Partition) Error!Stream {
        const g = self.d.green.?;
        var s: abi.Stream = null;
        try self.d.check(g.cuGreenCtxStreamCreate(&s, self.ctx, abi.stream_non_blocking, 0), "cuGreenCtxStreamCreate");
        return .{ .d = self.d, .handle = s };
    }

    /// Its streams must be destroyed first.
    pub fn deinit(self: *Partition) void {
        _ = self.d.green.?.cuGreenCtxDestroy(self.ctx);
        self.* = undefined;
    }
};

test "CUdevResource has the driver's size and SM field offsets" {
    try std.testing.expectEqual(@as(usize, 144), @sizeOf(abi.DevResource));
    try std.testing.expectEqual(@as(usize, 96), @offsetOf(abi.DevResource, "sm_count"));
    try std.testing.expectEqual(@as(usize, 104), @offsetOf(abi.DevResource, "coscheduled"));
}
