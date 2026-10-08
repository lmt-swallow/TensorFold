// Zig build: experts_grouped.cuh's decode-window kernel (grouped_cp_kernel, the fork's "fused" decode path) for the
// mul1 codebook at the one decode tile setting (8 tiles a block, 4 warps, 3 stages), every K2 range and epilogue
// grouped_cp_launch picks. The DecodeEpi layout is mirrored by cuda_experts.zig.
#include "experts_grouped.cuh"

namespace tf_exl3x {

static_assert(sizeof(DecodeEpi) == 144, "DecodeEpi layout changed: update cuda_experts.zig");

#define TF_CP(LO_, HI_, EPI_)                                                                                    \
    template __global__ void __launch_bounds__(4 * 32, 4) grouped_cp_kernel<2, 8, 4, 3, LO_, HI_, EPI_>(         \
        const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
        const int*, float*, int, int, int, int, int, int, const DecodeEpi);
TF_CP(8, 8, 0) TF_CP(8, 8, 1) TF_CP(8, 8, 2)
TF_CP(2, 10, 0) TF_CP(2, 10, 1) TF_CP(2, 10, 2)
TF_CP(2, 16, 0) TF_CP(2, 16, 1) TF_CP(2, 16, 2)
#undef TF_CP

}  // namespace tf_exl3x
