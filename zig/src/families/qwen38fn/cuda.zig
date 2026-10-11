//! Qwen3.8-Flash-Next on CUDA (one DGX Spark, an EXL3 pack): the family's module root. The arithmetic follows
//! TensorFold 0.6.6's `qwen4_exp/cuda` (ashhart/TensorFold v0.6.6, cb2ebf05), the reference every piece is checked
//! against bit for bit (tools/zig/qwen38fn/ writes the fixtures and captures its Triton kernels).

pub const config = @import("config.zig");
pub const weights_map = @import("weights_map.zig");
pub const weights = @import("cuda_weights.zig");
pub const gdn = @import("cuda_gdn.zig");
pub const experts = @import("cuda_experts.zig");
pub const glue = @import("cuda_glue.zig");
pub const hc = @import("cuda_hc.zig");
pub const ngram = @import("ngram.zig");
pub const attn = @import("cuda_attn.zig");
pub const forward = @import("cuda_forward.zig");
pub const prompt = @import("cuda_prompt.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
