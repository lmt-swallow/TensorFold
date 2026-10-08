// The torch glue of the DSV4.1 decode attention (graph.py StaticDecoder._attention) that is neither an EXL3 linear
// nor a Triton kernel, as plain CUDA with the same arithmetic: the compressor's softmax pool over a group's rows
// (torch's spatial softmax then the fp32 product sum, rounded to bf16), the indexer weights' bf16 scaling, and
// strided bf16 column copies (the views the Python passes). Built to a cubin by tools/zig/dsv41/attn_capture.sh.
#include <cuda_bf16.h>
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

// dst[r * dld + c] = src[r * sld + c] for c < cols (16-bit elements): a column block view made contiguous.
__global__ void copy_cols(const uint16_t* __restrict__ src, int sld, uint16_t* __restrict__ dst, int dld, int rows,
                          int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    dst[(size_t)r * dld + c] = src[(size_t)r * sld + c];
}

}
