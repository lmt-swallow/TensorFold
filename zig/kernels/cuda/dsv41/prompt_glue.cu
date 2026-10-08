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

}  // extern "C"
