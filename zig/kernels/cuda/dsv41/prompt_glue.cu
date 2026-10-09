// The prompt path's torch glue (model.py attention_k / _compress / moe at chunk rows) as kernels, each the torch
// expression's arithmetic: row copies by index lists, the compressor's pairwise softmax pool over consecutive rows,
// casts to fp32 for the cuBLAS gate and indexer-weight matmuls, int64 fills.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {

// dst[dst_idx ? dst_idx[r] : r] = src[src_idx ? src_idx[r] : r], rows of `words` 4-byte words (any dtype).
__global__ void tf_dsv41_copy_rows(const uint32_t* __restrict__ src, const long long* __restrict__ src_idx,
                                   uint32_t* __restrict__ dst, const long long* __restrict__ dst_idx, int rows,
                                   int words) {
    int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || i >= words) return;
    long long s = src_idx ? src_idx[r] : r;
    long long d = dst_idx ? dst_idx[r] : r;
    dst[(size_t)d * words + i] = src[(size_t)s * words + i];
}

// pooled[g] = bf16(kv[2g] * p0 + kv[2g+1] * p1), p = softmax over the pair of score rows (torch softmax(dim=1) over
// [full, 2, d] in fp32, product and sum in fp32, .to(bf16)): model.py _compress at ratio 2.
__global__ void tf_dsv41_pool_pairs(const float* __restrict__ kv, const float* __restrict__ score, int full, int d,
                                    __nv_bfloat16* __restrict__ pooled) {
    int g = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= full || i >= d) return;
    size_t a = (size_t)(2 * g) * d + i, b = (size_t)(2 * g + 1) * d + i;
    float s0 = score[a], s1 = score[b];
    float mx = fmaxf(s0, s1);
    float e0 = expf(s0 - mx), e1 = expf(s1 - mx);
    float sum = e0 + e1;
    float v = kv[a] * (e0 / sum) + kv[b] * (e1 / sum);
    pooled[(size_t)g * d + i] = __float2bfloat16_rn(v);
}

__global__ void tf_dsv41_bf16_to_f32(const __nv_bfloat16* __restrict__ in, float* __restrict__ out, long long count) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = __bfloat162float(in[i]);
}

__global__ void tf_dsv41_f16_to_f32(const __half* __restrict__ in, float* __restrict__ out, long long count) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = __half2float(in[i]);
}

__global__ void tf_dsv41_fill_i64(long long* __restrict__ out, long long value, long long count) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = value;
}

// ops.rms_norm on bf16 rows [n, d] (d a multiple of 128): torch's `xf * rsqrt(xf.square().mean(-1) + eps)`, then
// `(w.to(F32) * xf).to(bf16)`, with the mean's reduction in torch's order for an inner reduction of contiguous fp32
// (Reduce.cuh, vectorized by 4 over a 32-lane block row): lane t keeps four sums, element 4t + i (+ 128 k) in sum
// i, folded ((s0 + s1) + s2) + s3, then the lanes' shfl_down tree at offsets 16, 8, 4, 2, 1; the mean is that sum
// times 1/d. One warp a row.
__global__ void tf_dsv41_rms_norm_torch(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                        float eps, int d, __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const __nv_bfloat16* row = x + (size_t)r * d;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int base = 0; base < d; base += 128) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float v = __bfloat162float(row[base + 4 * t + i]);
            acc[i] = __fadd_rn(acc[i], __fmul_rn(v, v));
        }
    }
    float s = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xFFFFFFFFu, s, off));
    s = __shfl_sync(0xFFFFFFFFu, s, 0);
    const float rinv = rsqrtf(__fadd_rn(__fmul_rn(s, 1.0f / (float)d), eps));
    for (int c = t; c < d; c += 32) {
        const float v = __fmul_rn(__bfloat162float(row[c]), rinv);
        out[(size_t)r * d + c] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[c]), v));
    }
}

}  // extern "C"
