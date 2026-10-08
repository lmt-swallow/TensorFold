//! DeepSeek-V4.1-Flash on CUDA (two DGX Sparks over NCCL): the family's module root. The arithmetic follows the
//! Python 0.6.6 fork (bertholomus/TensorFold deepseek-v41-tp2), the reference every piece is checked against.

pub const exl3 = @import("cuda_exl3.zig");
pub const experts = @import("cuda_experts.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
