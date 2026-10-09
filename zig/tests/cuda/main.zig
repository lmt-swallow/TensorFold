//! GPU test runner for the Zig CUDA runtime: `tf-cuda-test <command> [args]`, PASS/FAIL/RESULT lines, exit 1 on failure.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const runtime_tests = @import("runtime_tests.zig");
const bench = @import("bench.zig");
const oracle_tests = @import("oracle_tests.zig");
const libs_tests = @import("libs_tests.zig");
const dsv41_triton = @import("dsv41_triton/aot_tests.zig");
const dsv41_attn = @import("dsv41_attn_tests.zig");
const dsv41_fwd = @import("dsv41_forward_tests.zig");
const dsv41_prompt_tests = @import("dsv41_prompt_tests.zig");
const sample_tests = @import("sample_tests.zig");
const dsv41_tests = @import("dsv41_tests.zig");
const dsv41_hc_tests = @import("dsv41_hc_tests.zig");
const dsv41_engram_tests = @import("dsv41_engram_tests.zig");
const dsv41_dspark_tests = @import("dsv41_dspark_tests.zig");
const dsv41_rounds_tests = @import("dsv41_rounds_tests.zig");

const usage =
    \\usage: tf-cuda-test <command>
    \\  info                      driver, device and library versions
    \\  smoke                     copies, fills, launches, argument packing, module globals
    \\  graph                     stream capture, explicit graphs, node and whole-exec updates
    \\  overhead [n] [reps]       dependent one-thread kernels: plain stream vs one graph
    \\  overhead-pdl [n] [reps]   the same with programmatic dependent launch on every kernel
    \\  launch-ex                 cuLaunchKernelEx: clusters, cooperative grid, PDL on a stream and in a graph
    \\  ptx                       hand-written PTX through the driver JIT, and a refused broken image
    \\  symbols                   every gdn fatbin instantiation resolves by its listed symbol
    \\  cublaslt                  bf16 GEMM through cuBLASLt against an fp64 reference
    \\  nccl                      one-rank NCCL all-reduce and all-gather
    \\  comm <rank> <world> <master-ip> <port>  two-machine NCCL communicator: id exchange, all-gather, control link
    \\  dsv41-attn <dir>          DSV4.1 attention sublayer (decode rows) against the fork's fixture (dsv41/dump_attention.py)
    \\  dsv41-forward <dir> <rank-cache-dir> <model-dir> [rank] [layers|all] [aot-dir]  DSV4.1 layer loop against the fork's fixture (dsv41/dump_forward.py)
    \\  dsv41-prompt <dir> <rank-cache-dir> <model-dir> [rank] [layers]  DSV4.1 prompt path (chunks of 2048 and 512) against the fork's fixture (dsv41/dump_prompt.py)
    \\  gdn-replay <dir>          replay_kernel bits against the Python oracle's fixture
    \\  gdn-tree <dir>            tree_kernel bits against the Python oracle's fixture
    \\  triton <dir>              a Triton cubin's bits against the Python oracle's fixture
    \\  sample                    sample.cu's keyed draws against the Metal rule's host references (synthetic rows)
    \\  exl3-linear <dir>         DSV4.1 EXL3 linear layers against the Python fork's fixture (dsv41/dump_linear.py)
    \\  exl3-experts <dir>        DSV4.1 EXL3 routed experts (decode windows) against the fork's fixture (dsv41/dump_experts.py)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    const cmd = args[1];
    const rest = args[2..];

    run(gpu, cmd, rest) catch |e| {
        std.debug.print("FAIL {s}: {t}\n", .{ cmd, e });
        return 1;
    };
    return 0;
}

fn arg(rest: []const [:0]const u8, i: usize) ![]const u8 {
    if (i >= rest.len) {
        std.debug.print("{s}", .{usage});
        return error.MissingArgument;
    }
    return rest[i];
}

fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "info")) return info(gpu);
    if (std.mem.eql(u8, cmd, "smoke")) return runtime_tests.smoke(gpu);
    if (std.mem.eql(u8, cmd, "graph")) return runtime_tests.graphs(gpu);
    if (std.mem.eql(u8, cmd, "overhead") or std.mem.eql(u8, cmd, "overhead-pdl")) {
        const n = if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 1000;
        const reps = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 20;
        return bench.overhead(gpu, n, reps, std.mem.eql(u8, cmd, "overhead-pdl"));
    }
    if (std.mem.eql(u8, cmd, "launch-ex")) return runtime_tests.launchEx(gpu);
    if (std.mem.eql(u8, cmd, "ptx")) return runtime_tests.ptx(gpu);
    if (std.mem.eql(u8, cmd, "symbols")) return runtime_tests.symbols(gpu);
    if (std.mem.eql(u8, cmd, "cublaslt")) return libs_tests.cublaslt(gpu);
    if (std.mem.eql(u8, cmd, "nccl")) return libs_tests.nccl(gpu);
    if (std.mem.eql(u8, cmd, "dsv41-aot-rmsnorm")) return dsv41_triton.rmsnorm(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "dsv41-aot-kv-norm-rope")) return dsv41_triton.kvNormRope(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "dsv41-attn")) return dsv41_attn.attention(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "dsv41-forward")) return dsv41_fwd.forward(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), if (rest.len > 3) try std.fmt.parseInt(usize, rest[3], 10) else 0, if (rest.len > 4 and rest[4].len > 0 and !std.mem.eql(u8, rest[4], "all")) try std.fmt.parseInt(usize, rest[4], 10) else null, if (rest.len > 5) rest[5] else null);
    if (std.mem.eql(u8, cmd, "dsv41-prompt")) return dsv41_prompt_tests.prompt(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), if (rest.len > 3) try std.fmt.parseInt(usize, rest[3], 10) else 0, if (rest.len > 4) try std.fmt.parseInt(usize, rest[4], 10) else null);
    if (std.mem.eql(u8, cmd, "dsv41-hc")) return dsv41_hc_tests.hc(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "dsv41-route")) return dsv41_hc_tests.route(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "dsv41-engram")) return dsv41_engram_tests.engram(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "dsv41-dspark")) return dsv41_dspark_tests.dspark(gpu, try arg(rest, 0), if (rest.len > 1) rest[1] else null);
    if (std.mem.eql(u8, cmd, "dsv41-draft-pool")) return dsv41_rounds_tests.draftPool(gpu, try arg(rest, 0), if (rest.len > 1) rest[1] else null);
    if (std.mem.eql(u8, cmd, "dsv41-rounds")) return dsv41_rounds_tests.rounds(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), if (rest.len > 3) rest[3] else null, if (rest.len > 4) rest[4] else null);
    if (std.mem.eql(u8, cmd, "comm")) return libs_tests.comm(gpu, try std.fmt.parseInt(u32, try arg(rest, 0), 10), try std.fmt.parseInt(u32, try arg(rest, 1), 10), try arg(rest, 2), try std.fmt.parseInt(u16, try arg(rest, 3), 10));
    if (std.mem.eql(u8, cmd, "gdn-replay")) return oracle_tests.gdnReplay(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "gdn-tree")) return oracle_tests.gdnTree(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "triton")) return oracle_tests.tritonKernel(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "sample")) return sample_tests.draws(gpu);
    if (std.mem.eql(u8, cmd, "exl3-linear")) return dsv41_tests.exl3Linear(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "exl3-experts")) return dsv41_tests.exl3Experts(gpu, try arg(rest, 0));
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}

fn info(gpu: check.Gpu) !void {
    var name_buf: [256]u8 = undefined;
    const name = try gpu.ctx.name(&name_buf);
    const mem = try gpu.ctx.memInfo();
    std.debug.print("RESULT driver CUDA {d}, device {s}, sm_{d}, {d} SMs, {d} MiB total, {d} MiB free, kernels embedded {}\n", .{
        try gpu.d.version(),                                 name,
        try gpu.ctx.capability(),                            try gpu.ctx.attribute(.multiprocessor_count),
        mem.total >> 20,                                     mem.free >> 20,
        cuda.kernels.available,
    });
}
