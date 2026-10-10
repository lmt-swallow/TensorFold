// The torch glue of the DSV4.1 decode attention (graph.py StaticDecoder._attention) that is neither an EXL3 linear
// nor a Triton kernel, as plain CUDA with the same arithmetic: the compressor's softmax pool over a group's rows
// (torch's spatial softmax then the fp32 product sum, rounded to bf16), the indexer weights' bf16 scaling,
// strided bf16 column copies (the views the Python passes), and the round's tail: its logits' rank columns in 16-byte
// words and each row's greedy token on the device. Built to a cubin by tools/zig/dsv41/attn_capture.sh.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {

// out[r, i] = bf16(sum_j kv[(first[r] + j) % raw, i] * softmax_j(score[(first[r] + j) % raw, i])), j < 2
// (compress ratio 2): torch's softmax over the group dim is exp(s - max) / sum with std::exp, the product and the
// sum of the two terms in fp32, then the bf16 rounding of .to(BF16).
__global__ void comp_pool2(const float* __restrict__ kv, const float* __restrict__ score,
                           const long long* __restrict__ first, int raw, int d, __nv_bfloat16* __restrict__ out) {
    int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    long long f = first[r];
    int a = (int)(f % raw);
    int b = (int)((f + 1) % raw);
    float s0 = score[(size_t)a * d + i];
    float s1 = score[(size_t)b * d + i];
    float mx = fmaxf(s0, s1);
    float e0 = expf(s0 - mx);
    float e1 = expf(s1 - mx);
    float sum = e0 + e1;
    float p0 = e0 / sum;
    float p1 = e1 / sum;
    float v = kv[(size_t)a * d + i] * p0 + kv[(size_t)b * d + i] * p1;
    out[(size_t)r * d + i] = __float2bfloat16_rn(v);
}

// out[i] = bf16(float(bf16(in[i])) * scale): rowmm(...).to(BF16) * scale on a bf16 tensor (opmath fp32).
__global__ void bf16_scale(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, float scale, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float a = __bfloat162float(__float2bfloat16_rn(in[i]));
    out[i] = __float2bfloat16_rn(a * scale);
}

// out[i] = half(in[i]): torch's .to(torch.float16) (round to nearest even).
__global__ void f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = __float2half_rn(in[i]);
}

// dst[r * dld + c] = src[r * sld + c] for c < cols (16-bit elements): a column block view made contiguous.
__global__ void copy_cols(const uint16_t* __restrict__ src, int sld, uint16_t* __restrict__ dst, int dld, int rows,
                          int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    dst[(size_t)r * dld + c] = src[(size_t)r * sld + c];
}

// copy_cols's bytes in 16-byte words: dst[r * dld + c] = src[r * sld + c] for c < n16 (strides and lengths in words),
// a word a thread over grid (ceil(n16 / blockDim), rows). Every pointer, stride and row length a multiple of 16 bytes.
__global__ void copy_rows16(const uint4* __restrict__ src, unsigned long long sld, uint4* __restrict__ dst,
                            unsigned long long dld, unsigned long long n16) {
    const unsigned long long r = blockIdx.y;
    const unsigned long long c = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n16) dst[r * dld + c] = src[r * sld + c];
}

// Each row's greedy token as the host takes it (best = 0; best = i where row[i] > row[best]): the first index of the
// largest value, a NaN never larger, 0 when the row's first value is NaN. logits [rows, n] fp32 (row stride ld
// floats), out [rows] u32; a row a block of 1024 threads.
__global__ void __launch_bounds__(1024) argmax_rows(const float* __restrict__ logits, unsigned long long ld, int n,
                                                    unsigned* __restrict__ out) {
    const float* row = logits + (unsigned long long)blockIdx.x * ld;
    const float ninf = -__int_as_float(0x7f800000);
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v == v && (v > bv || (v == bv && i < bi))) {
            bv = v;
            bi = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    const int nw = (blockDim.x + 31) >> 5;
    bv = lane < nw ? sv[lane] : ninf;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) out[blockIdx.x] = row[0] != row[0] ? 0u : (unsigned)bi;
}

}
