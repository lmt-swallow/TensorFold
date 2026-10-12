//! Actual served backend/core admissions: cold full/cone/released-slot full, fixed512 greedy outputs.
const std = @import("std");
const dsv41 = @import("dsv41");
const lanes = @import("lanes");
const check = @import("check.zig");

pub fn run(gpu: check.Gpu, cache: []const u8, model: []const u8, kernels: []const u8, rank: u32, tokens: usize, output: []const u8) !void {
    const L = dsv41.serve_lanes;
    if (tokens < 4732 or tokens + 1024 > 1048576) return error.BadPromptLength;
    const port = try std.fmt.parseInt(u16, std.mem.span(std.c.getenv("TF_TP_PORT") orelse return error.MissingPort), 10);
    var master: [4]u8 = undefined;
    var parts = std.mem.splitScalar(u8, std.mem.span(std.c.getenv("TF_TP_MASTER") orelse return error.MissingMaster), '.');
    for (&master) |*part| part.* = try std.fmt.parseInt(u8, parts.next() orelse return error.BadMaster, 10);
    if (parts.next() != null) return error.BadMaster;
    const e = try L.Engine.init(gpu.gpa, gpu.io, gpu.ctx, model, .{
        .rank = rank, .world = 2, .master = master, .port = port, .context = 1048576,
        .bucket = 1048576, .rope_dir = kernels, .cache_dir = cache, .kernels_dir = kernels,
        .drafts = true, .slots = 1, .pool = 1048576, .exact_cone = true, .overlap = false,
    });
    defer e.deinit();
    if (rank != 0) {
        try e.follow();
        check.pass("runtime cone follower agreement completed", .{});
        return;
    }
    defer e.stop();
    if (e.streams.coneCut(tokens, L.default_chunk, false, null, false) == null) return error.UnqualifiedCone;
    const ids = try gpu.gpa.alloc(u32, tokens);
    defer gpu.gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(1000 + i * 73 % 24000);
    const input = try std.json.Stringify.valueAlloc(gpu.gpa, .{ .input_ids = ids }, .{});
    defer gpu.gpa.free(input);
    const input_path = try std.fmt.allocPrint(gpu.gpa, "{s}.input.json", .{output});
    defer gpu.gpa.free(input_path);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = input_path, .data = input });
    var backend = L.Lanes.init(gpu.gpa, e);
    const rows = backend.windowRows();
    var cfg = try lanes.Config.init(gpu.gpa, backend.facts(), rows, rows - 1);
    defer cfg.deinit(gpu.gpa);
    var clock: lanes.backend.WallClock = .{ .io = gpu.io };
    var results: [3][]u32 = undefined;
    var count: usize = 0;
    defer for (results[0..count]) |r| gpu.gpa.free(r);
    for ([_]bool{ false, true, false }, 0..) |cone, arm| {
        e.settings.exact_cone = cone;
        var core = lanes.Engine.init(gpu.gpa, &cfg, backend.backend(), clock.clock());
        defer core.deinit();
        var s = try lanes.Stream.init(gpu.gpa, .{ .id = "runtime-cone", .prompt = ids, .max_new = 512, .eos = &.{}, .drafts = true });
        defer s.deinit(gpu.gpa);
        try core.addStream(&s);
        if (s.cached != 0) return error.UnexpectedReuse;
        while (!s.finished) try core.step();
        if (s.emitted().len != 512 or !e.streams.idle()) return error.UnreleasedStream;
        results[arm] = try gpu.gpa.dupe(u32, s.emitted());
        count += 1;
        const partial = try std.json.Stringify.valueAlloc(gpu.gpa, .{ .input_ids = ids, .output_ids = results[0..count] }, .{});
        defer gpu.gpa.free(partial);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output, .data = partial });
        std.debug.print("RUNTIME_ARM {d} exact_cone={any} outputs={d} rounds={d} accepted={d} drafted={d}\n", .{ arm, cone, s.emitted().len, s.rounds, s.accepted, s.drafted });
    }
    try check.expect(std.mem.eql(u32, results[0], results[1]), "runtime full/cone512 outputs differ", .{});
    try check.expect(std.mem.eql(u32, results[0], results[2]), "runtime released-slot full512 outputs differ", .{});
    check.pass("runtime cold full/cone/released-slot full512 IDs equal", .{});
}
