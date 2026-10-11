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
const dsv41_lanes_tests = @import("dsv41_lanes_tests.zig");
const dsv41_parallel_tests = @import("dsv41_parallel_tests.zig");
const dsv41_topk_tests = @import("dsv41_topk_tests.zig");
const dsv41_rope_tests = @import("dsv41_rope_tests.zig");
const dsv41_prompt_time = @import("dsv41_prompt_time.zig");
const dsv41_index_tests = @import("dsv41_index_tests.zig");
const dsv41_rdma_tests = @import("dsv41_rdma_tests.zig");
const qwen38fn_layer0 = @import("qwen38fn_layer0_tests.zig");
const qwen38fn_layers = @import("qwen38fn_layers_tests.zig");
const qwen38fn_prompt = @import("qwen38fn_prompt_tests.zig");
const qwen38fn_mtp = @import("qwen38fn_mtp_tests.zig");
const qwen38fn_shared = @import("qwen38fn_shared_tests.zig");
const dsv41_ptx_lift_tests = @import("dsv41_ptx_lift_tests.zig");

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
    \\  comm-overlap <rank> <world> <master-ip> <port>  a round-sized all-gather's latency beside another communicator's gathers
    \\  dsv41-rdma <rank> <world> <master-ip> <port> <devices> [gid]  DSV4.1 rounds' RDMA ring: its gathers' bytes against NCCL's (eager and graphs) and both paths' times
    \\  dsv41-ptx-lift <captured-set> <lifted-set>  DSV4.1 kernel set lifted from the captured PTX (dsv41/ptx_lift.py): same variants, loadable, no new spills
    \\  dsv41-attn <dir>          DSV4.1 attention sublayer (decode rows) against the fork's fixture (dsv41/dump_attention.py)
    \\  dsv41-forward <dir> <rank-cache-dir> <model-dir> [rank] [layers|all] [aot-dir]  DSV4.1 layer loop against the fork's fixture (dsv41/dump_forward.py)
    \\  dsv41-prompt <dir> <rank-cache-dir> <model-dir> [rank] [layers]  DSV4.1 prompt path (chunks of 2048 and 512) against the fork's fixture (dsv41/dump_prompt.py)
    \\  dsv41-rmsnorm-torch <dir>  DSV4.1 prompt path's torch-order rms_norm against the fork's ops.rms_norm (dsv41/dump_rmsnorm.py)
    \\  dsv41-topk [aot-dir]      DSV4.1 indexer selection on the device (top-k, candidate pool, its mask) against the host oracle
    \\  dsv41-rope <rope-dir> [rows ...]  DSV4.1 RoPE rows computed on the device against the fork's tables (dsv41/dump_rope.py)
    \\  dsv41-lanes-prefill <dir> <rank-cache-dir> <model-dir> <rank> [aot-dir]  DSV4.1 served engine's prefill step and a round against the fork's fixture (dsv41/dump_prompt.py --decode)
    \\  dsv41-lanes-spec <dir> <rank-cache-dir> <model-dir> <rank> [aot-dir]  DSV4.1 served engine's drafted rounds (DSpark draft, verify, keep) against the fork's fixture (dsv41/dump_spec.py)
    \\  dsv41-index-probe <aot-dir>  DSV4.1 prompt indexer kernel's head-sum order (checked by tools/zig/dsv41/index_tree.py)
    \\  dsv41-gemm-rows           DSV4.1 prompt path's fp32 cuBLAS GEMMs: rows independent of the row count (pinned algorithm)
    \\  dsv41-prompt-time <rank-cache-dir> <model-dir> <aot-dir> <rank> <layers> <tokens> [runs]  DSV4.1 prompt path's speed on one rank (synthetic ids, Engram under TF_DS_ENGRAM, every chunk profiled)
    \\  dsv41-lanes-parallel <dir> <rank-cache-dir> <model-dir> <rank> [aot-dir]  DSV4.1 served engine's concurrent streams (admissions, batched drafts, one shared round, keeps) against the fork's fixture (dsv41/dump_lanes_parallel.py)
    \\  gdn-replay <dir>          replay_kernel bits against the Python oracle's fixture
    \\  gdn-tree <dir>            tree_kernel bits against the Python oracle's fixture
    \\  triton <dir>              a Triton cubin's bits against the Python oracle's fixture
    \\  sample                    sample.cu's keyed draws against the Metal rule's host references (synthetic rows)
    \\  exl3-linear <dir>         DSV4.1 EXL3 linear layers against the Python fork's fixture (dsv41/dump_linear.py)
    \\  exl3-experts <dir>        DSV4.1 EXL3 routed experts (decode windows) against the fork's fixture (dsv41/dump_experts.py)
    \\  qwen38fn-layer0 <work-dir> <model-dir> [isolated]  Qwen3.8-Flash-Next embeddings + DeltaNet layer 0 (decode windows) against TensorFold 0.6.6's fixture (qwen38fn/capture_layer0.sh)
    \\  qwen38fn-prompt <work-dir> <model-dir>  Qwen3.8-Flash-Next prompt path (2,048-row chunks, then decode windows) against 0.6.6's fixture (qwen38fn/capture_prompt.sh)
    \\  qwen38fn-mtp <work-dir> <model-dir>  Qwen3.8-Flash-Next MTP drafting (draft head, head calls, rounds) against 0.6.6's serial drafted decoder (qwen38fn/capture_mtp.sh)
    \\  qwen38fn-shared <kernel-set> <model-dir>  Qwen3.8-Flash-Next shared rounds: four streams' logits, drafts and tokens the same alone and in one forward a round
    \\  qwen38fn-layers <work-dir> <model-dir> [layers] [isolated]  Qwen3.8-Flash-Next decode forward (every layer's boundaries, the logits) against 0.6.6's fixture (qwen38fn/capture_layers.sh)
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
    if (std.mem.eql(u8, cmd, "dsv41-rmsnorm-torch")) return dsv41_topk_tests.rmsnormTorch(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "dsv41-rope")) return dsv41_rope_tests.rope(gpu, try arg(rest, 0), rest[@min(rest.len, 1)..]);
    if (std.mem.eql(u8, cmd, "dsv41-topk")) return dsv41_topk_tests.topk(gpu, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "dsv41-lanes-prefill")) return dsv41_lanes_tests.prefill(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try std.fmt.parseInt(usize, try arg(rest, 3), 10), if (rest.len > 4) rest[4] else null);
    if (std.mem.eql(u8, cmd, "dsv41-lanes-spec")) return dsv41_lanes_tests.spec(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try std.fmt.parseInt(usize, try arg(rest, 3), 10), if (rest.len > 4) rest[4] else null);
    if (std.mem.eql(u8, cmd, "dsv41-prompt-time")) return dsv41_prompt_time.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try std.fmt.parseInt(usize, try arg(rest, 3), 10), try std.fmt.parseInt(usize, try arg(rest, 4), 10), try std.fmt.parseInt(usize, try arg(rest, 5), 10), if (rest.len > 6) try std.fmt.parseInt(usize, rest[6], 10) else 1);
    if (std.mem.eql(u8, cmd, "dsv41-index-probe")) return dsv41_index_tests.probe(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "dsv41-gemm-rows")) return dsv41_prompt_time.gemmRows(gpu);
    if (std.mem.eql(u8, cmd, "dsv41-lanes-parallel")) return dsv41_parallel_tests.parallel(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try std.fmt.parseInt(usize, try arg(rest, 3), 10), if (rest.len > 4) rest[4] else null);
    if (std.mem.eql(u8, cmd, "dsv41-ptx-lift")) return dsv41_ptx_lift_tests.ptxLift(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "dsv41-rdma")) return dsv41_rdma_tests.rdmaRing(gpu, try std.fmt.parseInt(u32, try arg(rest, 0), 10), try std.fmt.parseInt(u32, try arg(rest, 1), 10), try arg(rest, 2), try std.fmt.parseInt(u16, try arg(rest, 3), 10), try arg(rest, 4), if (rest.len > 5) try std.fmt.parseInt(u8, rest[5], 10) else 3);
    if (std.mem.eql(u8, cmd, "comm-overlap")) return libs_tests.commOverlap(gpu, try std.fmt.parseInt(u32, try arg(rest, 0), 10), try std.fmt.parseInt(u32, try arg(rest, 1), 10), try arg(rest, 2), try std.fmt.parseInt(u16, try arg(rest, 3), 10));
    if (std.mem.eql(u8, cmd, "comm")) return libs_tests.comm(gpu, try std.fmt.parseInt(u32, try arg(rest, 0), 10), try std.fmt.parseInt(u32, try arg(rest, 1), 10), try arg(rest, 2), try std.fmt.parseInt(u16, try arg(rest, 3), 10));
    if (std.mem.eql(u8, cmd, "gdn-replay")) return oracle_tests.gdnReplay(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "gdn-tree")) return oracle_tests.gdnTree(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "triton")) return oracle_tests.tritonKernel(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "sample")) return sample_tests.draws(gpu);
    if (std.mem.eql(u8, cmd, "exl3-linear")) return dsv41_tests.exl3Linear(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "exl3-experts")) return dsv41_tests.exl3Experts(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "qwen38fn-shared")) return qwen38fn_shared.shared(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "qwen38fn-mtp")) return qwen38fn_mtp.mtp(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "qwen38fn-prompt")) return qwen38fn_prompt.prompt(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "qwen38fn-layers")) return qwen38fn_layers.layers(gpu, try arg(rest, 0), try arg(rest, 1), if (rest.len > 2 and !std.mem.eql(u8, rest[2], "isolated") and !std.mem.eql(u8, rest[2], "all")) try std.fmt.parseInt(usize, rest[2], 10) else null, for (rest) |r| {
        if (std.mem.eql(u8, r, "isolated")) break true;
    } else false);
    if (std.mem.eql(u8, cmd, "qwen38fn-layer0")) return qwen38fn_layer0.layer0(gpu, try arg(rest, 0), try arg(rest, 1), rest.len > 2 and std.mem.eql(u8, rest[2], "isolated"));
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
