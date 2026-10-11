// Zig build: TensorFold 0.6.6's Flash Next DeltaNet front and back kernels verbatim (ashhart/TensorFold v0.6.6,
// cb2ebf05, src/tensorfold/families/qwen4_exp/cuda/gdn_io.cu), host glue dropped, a named namespace. Built with
// the extension's flags (-O3 --fmad=false). cuda_prompt.zig launches them as gdn_io.cpp does.
//
// Flash Next's DeltaNet front and back ends, with gdn.cu's chain-kernel operations and lane order, so the same bits.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tf_qfn_gdn_io {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// conv + SiLU of channel c for row r: taps from the row's window, [conv state (3 rows) | window rows]
__device__ __forceinline__ float conv_act(const __nv_bfloat16* P, const __nv_bfloat16* cs, const __nv_bfloat16* cw,
                                          const int* win, int r, int c, int C, int PW) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int src = win[r * TAPS + tap];
        const float x = src < TAPS - 1 ? __bfloat162float(cs[src * C + c])
                                       : __bfloat162float(P[(size_t)(src - (TAPS - 1)) * PW + c]);
        acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
    }
    return bf(acc / (1.0f + expf(-acc)));
}

// Block (row, head): heads < NK L2-norm q (times DK^-0.5) and k as the chain kernel's warps 0 and 1; others v, g, beta.
template <int NK, int NV>
__global__ void __launch_bounds__(128) front_kernel(
        const __nv_bfloat16* __restrict__ P, const long long* __restrict__ conv_ptrs, const int* __restrict__ sid,
        const int* __restrict__ win, const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float* __restrict__ q, float* __restrict__ k,
        __nv_bfloat16* __restrict__ v, float* __restrict__ g, float* __restrict__ beta) {
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int r = blockIdx.x, head = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const auto* cs = reinterpret_cast<const __nv_bfloat16*>(conv_ptrs[sid[r]]);
    __shared__ float xs[2][DK];
    if (head < NK) {
        xs[0][t] = conv_act(P, cs, cw, win, r, head * DK + t, C, PW);
        xs[1][t] = conv_act(P, cs, cw, win, r, NK * DK + head * DK + t, C, PW);
        __syncthreads();
        if (warp < 2) {
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = xs[warp][lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            float* out = (warp == 0 ? q : k) + ((size_t)r * NK + head) * DK + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = v4[i] * inv;
        }
        return;
    }
    const int hv = head - NK;
    v[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(conv_act(P, cs, cw, win, r, 2 * NK * DK + hv * DV + t,
                                                                     C, PW));
    if (t == 0) {
        const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
        const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
        g[r * NV + hv] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
        beta[r * NV + hv] = bf(sigmoidf_(b));
    }
}

// Block (row, value head): gated RMSNorm on warp 0's sum as the chain kernel, then bf16 out and 32-channel group sums.
template <int NK, int NV>
__global__ void __launch_bounds__(128) back_kernel(
        const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ P,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
        float* __restrict__ xs) {
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int r = blockIdx.x, hv = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV];
    __shared__ float rinv;
    ys[t] = __bfloat162float(y[((size_t)r * NV + hv) * DV + t]);
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float yy = ys[lane * 4 + i]; ss = ss + yy * yy; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
    const float z = __bfloat162float(P[(size_t)r * PW + C + hv * DV + t]);
    const float o = bf(yn * sigmoidf_(z));
    out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
}

}  // namespace tf_qfn_gdn_io

template __global__ void __launch_bounds__(128) tf_qfn_gdn_io::front_kernel<16, 48>(
    const __nv_bfloat16*, const long long*, const int*, const int*, const __nv_bfloat16*, const float*, const float*,
    float*, float*, __nv_bfloat16*, float*, float*);
template __global__ void __launch_bounds__(128) tf_qfn_gdn_io::back_kernel<16, 48>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, __nv_bfloat16*, float*);
