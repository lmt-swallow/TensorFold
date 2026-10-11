// ours: the Flash Next prompt path's torch glue as kernels.
//
// tf_qfn_f32_to_bf16: dst[i] = bf16(src[i]) for i < n, torch's float -> bfloat16 conversion (c10 round to nearest
// even, NaN to 0x7FC0): what `buf.y[r0:r0 + n].copy_(y)` does to a routed-expert window's fp32 slots.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256) tf_qfn_f32_to_bf16(const float* __restrict__ src,
                                                                     uint16_t* __restrict__ dst, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = src[i];
    const uint32_t u = __float_as_uint(x);
    dst[i] = x != x ? (uint16_t)0x7FC0 : (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
