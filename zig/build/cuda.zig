//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, the runtime, Nemotron, the CLI.

const std = @import("std");

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, arch_specific: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

const kernels = [_]Kernel{
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's layouts and the serial feed
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    .{ .name = "dsv41_attn_glue", .src = "dsv41/attn_glue", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the decode attention's torch glue (softmax pool, bf16 scaling, column copies, fp16 cast)
    .{ .name = "dsv41_exl3_linear", .src = "dsv41/linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // the DSV4.1 fork's cuda/exl3/linear.py, tensorfold_exl3_linear_v6
    .{ .name = "dsv41_exl3_experts", .src = "dsv41/experts", .flags = &.{ "-O3", "-lineinfo" } }, // cuda/exl3/experts.py, tensorfold_exl3_experts_v18: grouping, rot_in, epilogues, decode_prep
    .{ .name = "dsv41_exl3_experts_decode", .src = "dsv41/experts_decode", .flags = &.{ "-O3", "-lineinfo" } }, // the same extension's grouped_cp_kernel (decode windows)
    .{ .name = "dsv41_engram", .src = "dsv41/engram", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: model.py Engram._decode and Comm.sum as kernels
    .{ .name = "dsv41_exl3_experts_prompt", .src = "dsv41/experts_prompt", .flags = &.{ "-O3", "-lineinfo" } }, // the experts extension's prompt-chunk kernels (grouped_mma, grouped_rows)
    .{ .name = "dsv41_prompt_glue", .src = "dsv41/prompt_glue", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the prompt path's torch glue (row copies, the compressor's pair pool, fp32 casts)
    .{ .name = "dsv41_topk_glue", .src = "dsv41/topk_glue", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the indexer's selection (topk_indices, the candidate pool and its mask) on the device
    .{ .name = "dsv41_exl3_grouped", .src = "dsv41/linear_grouped", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // the same extension's linear_grouped.cu: rot_many, glinear (grouped launches, rotation / RoPE epilogues)
    .{ .name = "dsv41_l2_prefetch", .src = "dsv41/l2_prefetch", .flags = &.{"-O3"} }, // cuda/exl3/prefetch.py, tensorfold_exl3_l2_prefetch_v6: paced L2 prefetch of weight ranges
    .{ .name = "dsv41_rope", .src = "dsv41/rope", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: ops.py rope_cs's table rows (glibc's sinf / cosf) at given positions
    .{ .name = "dsv41_rdma", .src = "dsv41/rdma_gather", .flags = &.{"-O3"} }, // the DSV4.1 fork's cuda/rdma_gather.cu v7 kernel pair: the rounds' gathers over the RDMA ring
    .{ .name = "qwen38fn_gdn", .src = "qwen38fn/gdn", .flags = &.{ "-O3", "--fmad=false" } }, // 0.6.6 qwen4_exp/cuda/gdn.py, tensorfold_qwen4_exp_gdn
    .{ .name = "qwen38fn_experts", .src = "qwen38fn/experts", .flags = &.{ "-O3", "-lineinfo" } }, // 0.6.6 cuda/exl3/experts.py, tensorfold_exl3_experts_v1
    .{ .name = "qwen38fn_gdn_io", .src = "qwen38fn/gdn_io", .flags = &.{ "-O3", "--fmad=false" } }, // 0.6.6 qwen4_exp/cuda/gdn_io.py, tensorfold_qwen4_exp_gdn_io
    .{ .name = "qwen38fn_gdn_prefill", .src = "qwen38fn/gdn_prefill", .flags = &.{ "-O3", "--fmad=false" } }, // 0.6.6 cuda/kernels/gdn.py's gdn_prefill.cu, tensorfold_gdn_v2
    .{ .name = "qwen38fn_prompt_glue", .src = "qwen38fn/prompt_glue", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the prompt path's torch glue
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// core/stagger.zig, the segment schedule the Metal and CUDA runners share, as its own module (no imports).
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (kernels, images) |k, image| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, tokenizer: *std.Build.Module, dsv41: *std.Build.Module, qwen38fn: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    const dsv41 = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    dsv41.addImport("cuda", cuda);
    dsv41.addImport("core", core);
    dsv41.addImport("lanes", lanes);
    // the RDMA ring's verbs (deepseek_v41/rdma.zig)
    dsv41.addImport("fabric", b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    const qwen38fn = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen38fn/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    qwen38fn.addImport("cuda", cuda);
    qwen38fn.addImport("core", core);
    qwen38fn.addImport("dsv41", dsv41); // its EXL3 linear (0.6.6's kernel body) until that moves to the core
    qwen38fn.addImport("lanes", lanes);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .tokenizer = tokenizer, .dsv41 = dsv41, .qwen38fn = qwen38fn };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (121; later 120,89)") orelse "121";
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("lanes", mods.lanes);
    runner.addImport("dsv41", mods.dsv41);
    runner.addImport("qwen38fn", mods.qwen38fn);
    b.installArtifact(b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner }));
    _ = nativeServer(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.dsv41, mods.qwen38fn, mods.tokenizer, build_options, true);
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, dsv41: *std.Build.Module, qwen38fn: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "nemotron", .module = nemotron }, .{ .name = "dsv41", .module = dsv41 }, .{ .name = "qwen38fn", .module = qwen38fn } },
    });
    return .{ .api = api, .engines = mod };
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, dsv41: *std.Build.Module, qwen38fn: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, lanes, nemotron, dsv41, qwen38fn);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = m.engines }} }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin").dependOn(&install.step);
    return exe;
}

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const seam = engines(b, host, .debug, cuda, mods.lanes, mods.nemotron, mods.dsv41, mods.qwen38fn);
    // the engine seam's own tests (lane_host.zig) too, as the Metal build runs them
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, mods.dsv41, mods.qwen38fn, seam.engines, seam.api, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.lanes, mods.nemotron, mods.dsv41, mods.qwen38fn, mods.tokenizer, build_options, false);
}

/// Validate the complete named input set, including images unused by today's server.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const check = b.addSystemCommand(&.{ "sh", "-c", "for file do [ -f \"$file\" ] && [ -s \"$file\" ] || { echo \"missing or empty CUDA fatbin: $file\" >&2; exit 1; }; done", "check-fatbins" });
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}
