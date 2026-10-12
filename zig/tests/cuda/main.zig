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
const qwen38fn_prompt_time = @import("qwen38fn_prompt_time.zig");
const qwen38fn_select = @import("qwen38fn_select_tests.zig");
const qwen38fn_mtp = @import("qwen38fn_mtp_tests.zig");
const qwen38fn_shared = @import("qwen38fn_shared_tests.zig");
const glm53_kda = @import("glm53_kda_tests.zig");
const glm53_dflash_target = @import("glm53_dflash_target_tests.zig");
const glm53_dflash = @import("glm53_dflash_tests.zig");
const glm53_moe = @import("glm53_moe_tests.zig");
const glm53_dsa = @import("glm53_dsa_tests.zig");
const glm53_prompt = @import("glm53_prompt_tests.zig");
const glm53_forward = @import("glm53_forward_tests.zig");
const glm53_generate = @import("glm53_generate_tests.zig");
const glm53_prompt_time = @import("glm53_prompt_time.zig");
const glm53_select = @import("glm53_select_tests.zig");
const glm53_topk = @import("glm53_topk_tests.zig");
const glm53_keep = @import("glm53_keep_tests.zig");
const glm53_shared = @import("glm53_shared_tests.zig");
const glm53_tier = @import("glm53_tier_tests.zig");
const glm53_seg = @import("glm53_seg_tests.zig");
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
    \\  qwen38fn-prompt-time <kernel-set> <model-dir> [depths] [chunks]  Qwen3.8-Flash-Next served prompt chunks (MTP head included) timed at chosen depths (timing only)
    \\  qwen38fn-select <kernel-set> <model-dir>  Qwen3.8-Flash-Next indexer block selection (_select, _select_tiles past 131,072 keys) against a host oracle up to the capacity
    \\  qwen38fn-mtp <work-dir> <model-dir>  Qwen3.8-Flash-Next MTP drafting (draft head, head calls, rounds) against 0.6.6's serial drafted decoder (qwen38fn/capture_mtp.sh)
    \\  qwen38fn-shared <kernel-set> <model-dir>  Qwen3.8-Flash-Next shared rounds: four streams' logits, drafts and tokens the same alone and in one forward a round
    \\  glm53-tier <aot-dir> <model-dir> <file-dir> [positions]  GLM-5.3-Flash tiered latent cache (slab + NVMe file) against a whole cache: the sparse kernels' outputs byte for byte, decode windows and a prompt block
    \\  glm53-prompt <work-dir> <model-dir>  GLM-5.3-Flash prompt path (chunks, head, MTP absorb, commit) over layers 0 and 3 and the MTP layer on both ranks against the Mia fork's fixture (glm53/capture.sh prompt)
    \\  glm53-prompt-fp8 <work-dir> <model-dir>  glm53-prompt on FP8 DSA caches (TF_GLM_KV=fp8) against a KV=fp8 capture
    \\  glm53-dsa-fp8 <work-dir> <model-dir> [isolated]  glm53-dsa on FP8 DSA caches (TF_GLM_KV=fp8) against a KV=fp8 capture (glm53/capture.sh dsa with KV=fp8)
    \\  glm53-drafted <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir>  GLM-5.3-Flash the fixture's request decoded with MTP and copy drafts on the pair: its reply and rounds against the reference's
    \\  glm53-seg <work-dir> <model-dir>  GLM-5.3-Flash several streams' windows in one forward (forwardSegs, layers 0 and 3) against the Mia fork's batched verify windows, both ranks on this GPU (glm53/capture.sh seg)
    \\  glm53-seg-fp8 <work-dir> <model-dir>  glm53-seg on FP8 DSA caches against a KV=fp8 capture
    \\  glm53-shared <fwd-dir> <model-dir> <rank> <master-ip|local> <port> <aot-dir> [layers]  GLM-5.3-Flash four streams' windows in one forward against each alone and the single-stream path, on the pair
    \\  glm53-keep <fwd-dir> <model-dir> <rank> <master-ip|local> <port> <aot-dir> [layers]  GLM-5.3-Flash a prompt continued from a kept prompt's boundary (pool extents, a snapshot) on the pair against its fresh prefill
    \\  glm53-generate <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir>  GLM-5.3-Flash the fixture's request decoded serially on the pair, its reply against the reference's ids
    \\  glm53-forward <fwd-dir> <model-dir> <rank> <master-ip> <port> <aot-dir> [layers]  GLM-5.3-Flash whole forward on one rank of the pair (NCCL gathers) against the Mia fork's forward fixture (glm53/b1/forward_capture.sh)
    \\  glm53-onepass <aot-dir>  GLM-5.3-Flash onepass.cu against the captured _sparse_onepass
    \\  glm53-scores <aot-dir>  GLM-5.3-Flash scores.cu against the captured _scores (TF_GLM_SCORE_RB's variant)
    \\  glm53-topk  GLM-5.3-Flash draft top-k on the device (one block, two stages) against the host scan
    \\  glm53-select <aot-dir>  GLM-5.3-Flash select.cu's prompt pool selection against the definition and the captured kernels
    \\  glm53-prompt-time <aot-dir> <model-dir> [pos ...]  GLM-5.3-Flash prompt path speed on rank 0 alone: a 2,048-row chunk at each position, its segments timed by kind
    \\  glm53-dsa <work-dir> <model-dir> [isolated]  GLM-5.3-Flash DSA layer 3 (indexer, top-512 pools, latent attention; then its MoE) on both ranks, runs a and l, against the Mia fork's fixture (glm53/capture.sh dsa)
    \\  glm53-dflash-target <reference-json> <model-dir> <draft-dir> <rank> <master-ip> <port> <aot-dir>  Single-stream target 52/512 full-ID gate after prompt cache parity
    \\  glm53-dflash-connected <work-dir> <model-dir> <draft-dir>  DFlash2 tap/ring/block/candidate fixture with checked native partials and recorded peer partials
    \\  glm53-dflash-projections <work-dir> <model-dir> <draft-dir>  DFlash2 Q4 projections against the captured nine-step fixture
    \\  glm53-moe <work-dir> <model-dir> [isolated]  GLM-5.3-Flash MoE block (router, EXL3 experts decode / prompt, shared expert) on both ranks against the Mia fork's fixture (glm53/capture.sh moe)
    \\  glm53-kda <work-dir> <model-dir> [isolated]  GLM-5.3-Flash embeddings + layer 0 (KDA, dense MLP) on both ranks against the Mia fork's fixture (glm53/capture.sh kda)
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
    if (std.mem.eql(u8, cmd, "qwen38fn-select")) return qwen38fn_select.select(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "qwen38fn-prompt-time")) return qwen38fn_prompt_time.run(gpu, try arg(rest, 0), try arg(rest, 1), if (rest.len > 2) rest[2] else null, if (rest.len > 3) try std.fmt.parseInt(usize, rest[3], 10) else 3);
    if (std.mem.eql(u8, cmd, "qwen38fn-prompt")) return qwen38fn_prompt.prompt(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "qwen38fn-layers")) return qwen38fn_layers.layers(gpu, try arg(rest, 0), try arg(rest, 1), if (rest.len > 2 and !std.mem.eql(u8, rest[2], "isolated") and !std.mem.eql(u8, rest[2], "all")) try std.fmt.parseInt(usize, rest[2], 10) else null, for (rest) |r| {
        if (std.mem.eql(u8, r, "isolated")) break true;
    } else false);
    if (std.mem.eql(u8, cmd, "glm53-prompt")) return glm53_prompt.run(gpu, try arg(rest, 0), try arg(rest, 1), null);
    if (std.mem.eql(u8, cmd, "glm53-prompt-fp8")) return glm53_prompt.run(gpu, try arg(rest, 0), try arg(rest, 1), .fp8);
    if (std.mem.eql(u8, cmd, "glm53-dsa")) return glm53_dsa.run(gpu, try arg(rest, 0), try arg(rest, 1), rest.len > 2 and std.mem.eql(u8, rest[2], "isolated"), null);
    if (std.mem.eql(u8, cmd, "glm53-dsa-fp8")) return glm53_dsa.run(gpu, try arg(rest, 0), try arg(rest, 1), rest.len > 2 and std.mem.eql(u8, rest[2], "isolated"), .fp8);
    if (std.mem.eql(u8, cmd, "glm53-drafted")) return glm53_generate.run(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(usize, try arg(rest, 2), 10), try arg(rest, 3), try std.fmt.parseInt(u16, try arg(rest, 4), 10), try arg(rest, 5));
    if (std.mem.eql(u8, cmd, "glm53-seg")) return glm53_seg.run(gpu, try arg(rest, 0), try arg(rest, 1), null);
    if (std.mem.eql(u8, cmd, "glm53-seg-fp8")) return glm53_seg.run(gpu, try arg(rest, 0), try arg(rest, 1), .fp8);
    if (std.mem.eql(u8, cmd, "glm53-shared")) return glm53_shared.run(gpu, .{ .dir = try arg(rest, 0), .model = try arg(rest, 1), .aot = try arg(rest, 5), .layers = if (rest.len > 6) try std.fmt.parseInt(usize, rest[6], 10) else null }, try std.fmt.parseInt(usize, try arg(rest, 2), 10), try arg(rest, 3), try std.fmt.parseInt(u16, try arg(rest, 4), 10));
    if (std.mem.eql(u8, cmd, "glm53-keep")) return glm53_keep.run(gpu, .{ .dir = try arg(rest, 0), .model = try arg(rest, 1), .aot = try arg(rest, 5), .layers = if (rest.len > 6) try std.fmt.parseInt(usize, rest[6], 10) else null }, try std.fmt.parseInt(usize, try arg(rest, 2), 10), try arg(rest, 3), try std.fmt.parseInt(u16, try arg(rest, 4), 10));
    if (std.mem.eql(u8, cmd, "glm53-generate")) return glm53_forward.generate(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(usize, try arg(rest, 2), 10), try arg(rest, 3), try std.fmt.parseInt(u16, try arg(rest, 4), 10), try arg(rest, 5));
    if (std.mem.eql(u8, cmd, "glm53-forward")) return glm53_forward.run(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(usize, try arg(rest, 2), 10), try arg(rest, 3), try std.fmt.parseInt(u16, try arg(rest, 4), 10), try arg(rest, 5), if (rest.len > 6) try std.fmt.parseInt(usize, rest[6], 10) else null);
    if (std.mem.eql(u8, cmd, "glm53-topk")) return glm53_topk.run(gpu);
    if (std.mem.eql(u8, cmd, "glm53-select")) return glm53_select.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "glm53-scores")) return glm53_select.runScores(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "glm53-onepass")) return glm53_select.runOnepass(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "glm53-prompt-time")) {
        var pos: [16]usize = undefined;
        var n: usize = 0;
        for (rest[@min(rest.len, 2)..]) |a| {
            if (n == pos.len) break;
            pos[n] = try std.fmt.parseInt(usize, a, 10);
            n += 1;
        }
        const default = [_]usize{ 0, 30720 };
        return glm53_prompt_time.run(gpu, try arg(rest, 0), try arg(rest, 1), if (n == 0) &default else pos[0..n]);
    }
    if (std.mem.eql(u8, cmd, "glm53-tier")) return glm53_tier.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), if (rest.len > 3) rest[3] else null);
    if (std.mem.eql(u8, cmd, "glm53-dflash-target")) return glm53_dflash_target.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try std.fmt.parseInt(usize, try arg(rest, 3), 10), try arg(rest, 4), try std.fmt.parseInt(u16, try arg(rest, 5), 10), try arg(rest, 6));
    if (std.mem.eql(u8, cmd, "glm53-dflash-connected")) return glm53_dflash.connected(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "glm53-dflash-projections")) return glm53_dflash.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "glm53-moe")) return glm53_moe.run(gpu, try arg(rest, 0), try arg(rest, 1), rest.len > 2 and std.mem.eql(u8, rest[2], "isolated"));
    if (std.mem.eql(u8, cmd, "glm53-kda")) return glm53_kda.run(gpu, try arg(rest, 0), try arg(rest, 1), rest.len > 2 and std.mem.eql(u8, rest[2], "isolated"));
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
