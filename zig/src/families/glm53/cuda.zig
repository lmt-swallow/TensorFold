//! GLM-5.3-Flash on CUDA (two DGX Sparks, tensor parallel, the EXL3 pack): the family's module root. The arithmetic
//! follows the Mia fork's `glm5_next/cuda` (TensorFold v0.6.0 with the 70 patches of the recipe's v1.5 image,
//! ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:ef83797d…) in its strict setting
//! (TF_GLM_DENSE=bf16, TF_GLM_KV=bf16, TF_GLM_KDA_CHUNKED=0, MTP drafts), the reference every piece is checked
//! against bit for bit (tools/zig/glm53/ writes the fixtures and captures its Triton kernels).

pub const config = @import("config.zig");
pub const weights_map = @import("weights_map.zig");
pub const plan = @import("plan.zig");
pub const glue = @import("cuda_glue.zig");
pub const kda = @import("cuda_kda.zig");
pub const weights = @import("cuda_weights.zig");
pub const forward = @import("cuda_forward.zig");
pub const moe = @import("cuda_moe.zig");
pub const dsa = @import("cuda_dsa.zig");
pub const draft_head = @import("draft_head.zig");
pub const mtp = @import("cuda_mtp.zig");
pub const prompt = @import("cuda_prompt.zig");
pub const model = @import("cuda_model.zig");
pub const copy_drafts = @import("copy_drafts.zig");
pub const engine = @import("cuda_engine.zig");
pub const serve_lanes = @import("cuda_lanes.zig");
pub const native = @import("cuda_native.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
