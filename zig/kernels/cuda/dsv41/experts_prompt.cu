// Zig build: experts_grouped.cuh's prompt-chunk kernels (experts.py routed() for 64 rows or more, the fork's
// defaults: TF_EXL3_PROMPT_TILES=1, TF_EXL3_PROMPT_KERNEL=mma, TF_EXL3_GROUP_LIST=1) for the mul1 codebook:
// grouped_mma_kernel (gate/up: 64 member rows a program, 4 k tiles a step, the splits folded) and grouped_rows_kernel
// at the down tile setting (8 tiles a block, 4 warps, 1 tile in flight, 2 member tiles a program, no fold), every
// K2 range grouped_mma_launch / grouped_rows_launch pick.
#include "experts_grouped.cuh"

namespace tf_exl3x {

#define TF_MMA(LO_, HI_)                                                                                         \
    template __global__ void __launch_bounds__(256) grouped_mma_kernel<2, LO_, HI_, true, 4>(                    \
        const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
        const int*, float*, int, int, int, int, int, int, int, int, const int*);
TF_MMA(8, 8) TF_MMA(2, 10) TF_MMA(2, 16)
#undef TF_MMA

#define TF_ROWS(LO_, HI_)                                                                                        \
    template __global__ void __launch_bounds__(4 * 32) grouped_rows_kernel<2, 8, 4, 1, LO_, HI_, 2, false>(      \
        const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
        const int*, float*, int, int, int, int, int, int, int, const int*);
TF_ROWS(8, 8) TF_ROWS(2, 10) TF_ROWS(2, 16)
#undef TF_ROWS

}  // namespace tf_exl3x
