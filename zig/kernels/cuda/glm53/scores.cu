// GLM-5.3-Flash's prompt indexer scores (sparse.py `_scores`, bf16 pooled keys) with its bits, more rows and pools a
// block. A row r of a prompt chunk's selection block (position P + r) scores pool p as
//   s = sum_h w_h * max(scale * (q_h . k_p), 0)    for p < (P + r + 1) / 4, else -inf,
// w_h = bf16 w * wscale. The reference computes q_h . k_p with mma.sync m16n8k16 (heads the M rows in two 16-head
// tiles, the 128 key dims eight k16 steps in order from a zero accumulator) and sums the heads per column in a fixed
// tree: in each lane (group g = lane / 4) t = m_{g+8} w_{g+8}, t = fma(m_g, w_g, t), t = fma(m_{g+16}, w_{g+16}, t),
// t = fma(m_{g+24}, w_{g+24}, t); then t += shfl_xor(t, 16), += shfl_xor(t, 8), += shfl_xor(t, 4) (the PTX of the
// captured `_scores`). Every output element is a function of its row, its pool and that fixed arithmetic, so this
// kernel's other tiling (256 pools and 32 rows a block, the pool keys' fragments kept in registers across the rows)
// writes the captured kernel's bytes.
//
// The arithmetic was read from the captured cubin's PTX (and its SASS: no fused multiply-adds other than the
// explicit fma.rn chain). Checked by tf-cuda-test glm53-scores (byte for byte against the captured `_scores` at RB 4
// and 8, block positions 2,051 to 600,000, 37 to 512 rows, 2,048 to 150,016 pools) and by glm53-prompt's fixtures
// under TF_GLM_PROMPT_CHECK=1.
#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_glm53_scores {

constexpr int kHeads = 32;
constexpr int kDim = 128;
constexpr int kNt = 4;                // n8 tiles a warp
constexpr int kPools = 8 * kNt * 8;   // pools a block: 8 warps of 32
constexpr int kRows = 32;             // rows a block
constexpr int kStage = 4;             // query rows a shared-memory stage
constexpr int kThreads = 256;
constexpr int kPad = 8;               // shared rows padded by 16 bytes: ldmatrix reads no bank twice
constexpr int kStride = kDim + kPad;  // elements a shared row

// The pool keys first; once each warp holds its keys' fragments, the same memory takes two stages of query rows.
union Shared {
    __nv_bfloat16 k[kPools * kStride];
    __nv_bfloat16 q[2][kStage * kHeads * kStride];
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    const int n = ok ? 16 : 0;  // zero-fill a row past the end
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float relu_scaled(float dot, float scale) {
    return fmaxf(__fmul_rn(dot, scale), 0.0f);
}

// Query rows first .. first + n - 1 [32 heads, 128] into q (shared), 16 bytes a copy.
__device__ __forceinline__ void load_q(__nv_bfloat16* q, const __nv_bfloat16* src, int n) {
    for (int c = threadIdx.x; c < n * kHeads * kDim / 8; c += kThreads) {
        const int h = c / (kDim / 8), j = c % (kDim / 8);   // h: row * 32 + head
        cp_async16(q + h * kStride + j * 8, src + h * kDim + j * 8, true);
    }
    cp_async_commit();
}

// QI: the block's rows [R, 32 heads, 128] bf16; W: each row's 32 head weights (row stride w_stride) bf16; PK: pooled
// keys [pools, 128] bf16; OUT [R, NP] fp32; POS: the first row's position. Grid (pool tiles of 256, row groups of 32).
__global__ void __launch_bounds__(kThreads) scores_kernel(const __nv_bfloat16* __restrict__ QI,
                                                          const __nv_bfloat16* __restrict__ W, int w_stride,
                                                          const __nv_bfloat16* __restrict__ PK, float* __restrict__ OUT,
                                                          const int* __restrict__ POS, int R, int NP, float scale,
                                                          float wscale) {
    extern __shared__ __align__(16) unsigned char smem[];
    Shared& s = *reinterpret_cast<Shared*>(smem);
    const int P = *POS;
    const int pool0 = blockIdx.x * kPools;
    const int row0 = blockIdx.y * kRows;
    const int rows = min(kRows, R - row0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int which = lane >> 3, row8 = lane & 7;
    const int last_pools = (P + row0 + rows) / 4;  // complete pools of the group's last row

    // the block's pools all past every row's complete pools: -inf, no dots (as `_scores`' early tiles)
    if (pool0 >= last_pools) {
        for (int i = threadIdx.x; i < rows * kPools; i += kThreads) {
            const int r = row0 + i / kPools, p = pool0 + i % kPools;
            if (p < NP) OUT[(int64_t)r * NP + p] = -INFINITY;
        }
        return;
    }

    // the pool keys (rows past the last complete pool read as zeros: their scores are -inf whatever they hold)
    for (int c = threadIdx.x; c < kPools * kDim / 8; c += kThreads) {
        const int p = c / (kDim / 8), j = c % (kDim / 8);
        const bool ok = pool0 + p < last_pools;
        cp_async16(s.k + p * kStride + j * 8, PK + (int64_t)(ok ? pool0 + p : 0) * kDim + j * 8, ok);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // this warp's B fragments: pools pool0 + warp * 32 + nb * 8 + n, eight k16 steps
    uint32_t b[kNt][8][2];
#pragma unroll
    for (int nb = 0; nb < kNt; ++nb) {
        const int n0 = warp * 8 * kNt + nb * 8;
#pragma unroll
        for (int ks = 0; ks < 8; ks += 2) {
            // x4: matrices (k ks lo, k ks hi, k ks+1 lo, k ks+1 hi) of the 8 pools
            uint32_t r[4];
            ldmatrix_x4(r, s.k + (n0 + row8) * kStride + ks * 16 + which * 8);
            b[nb][ks][0] = r[0];
            b[nb][ks][1] = r[1];
            b[nb][ks + 1][0] = r[2];
            b[nb][ks + 1][1] = r[3];
        }
    }
    __syncthreads();  // the keys' memory now takes the query stages
    const __nv_bfloat16* qbase = QI + (int64_t)row0 * kHeads * kDim;
    load_q(s.q[0], qbase, min(kStage, rows));

    for (int st = 0; st * kStage < rows; ++st) {
        const int cur = st & 1;
        cp_async_wait_all();
        __syncthreads();
        const int first = (st + 1) * kStage;
        if (first < rows) load_q(s.q[cur ^ 1], qbase + (int64_t)first * kHeads * kDim, min(kStage, rows - first));
        const int in_stage = min(kStage, rows - st * kStage);
        for (int i = 0; i < in_stage; ++i) {
            const int r = row0 + st * kStage + i;
            // the head weights of this lane's four heads: g, g + 8, g + 16, g + 24
            const __nv_bfloat16* wr = W + (int64_t)r * w_stride;
            const float w0 = __fmul_rn(wscale, __bfloat162float(wr[g]));
            const float w1 = __fmul_rn(wscale, __bfloat162float(wr[g + 8]));
            const float w2 = __fmul_rn(wscale, __bfloat162float(wr[g + 16]));
            const float w3 = __fmul_rn(wscale, __bfloat162float(wr[g + 24]));
            const int npool = (P + r + 1) / 4;
            float acc[kNt][2][4];  // [n tile][head tile][c]
#pragma unroll
            for (int nb = 0; nb < kNt; ++nb)
#pragma unroll
                for (int m = 0; m < 2; ++m)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[nb][m][j] = 0.0f;
            const __nv_bfloat16* q = s.q[cur] + i * kHeads * kStride;
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                uint32_t a[2][4];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    // x4: (rows 0-7, k lo), (rows 8-15, k lo), (rows 0-7, k hi), (rows 8-15, k hi) of head tile m
                    const int h = m * 16 + (which & 1) * 8 + row8;
                    ldmatrix_x4(a[m], q + h * kStride + ks * 16 + (which >> 1) * 8);
                }
#pragma unroll
                for (int nb = 0; nb < kNt; ++nb)
#pragma unroll
                    for (int m = 0; m < 2; ++m) mma(acc[nb][m], a[m], b[nb][ks][0], b[nb][ks][1]);
            }
            float val[kNt][2];
#pragma unroll
            for (int nb = 0; nb < kNt; ++nb) {
#pragma unroll
                for (int col = 0; col < 2; ++col) {
                    // head tile 0: rows g (c[col]) and g + 8 (c[2 + col]); tile 1: heads 16 + g, 24 + g
                    const float m0 = relu_scaled(acc[nb][0][col], scale);
                    const float m1 = relu_scaled(acc[nb][0][2 + col], scale);
                    const float m2 = relu_scaled(acc[nb][1][col], scale);
                    const float m3 = relu_scaled(acc[nb][1][2 + col], scale);
                    float v = __fmul_rn(m1, w1);
                    v = __fmaf_rn(m0, w0, v);
                    v = __fmaf_rn(m2, w2, v);
                    v = __fmaf_rn(m3, w3, v);
                    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
                    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
                    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
                    val[nb][col] = v;  // every lane of the column (same t) holds it
                }
            }
            // one coalesced store a warp: lane (g, t) writes tile g / 2's column 2t + g % 2
            float v = 0.0f;
#pragma unroll
            for (int nb = 0; nb < kNt; ++nb)
#pragma unroll
                for (int col = 0; col < 2; ++col)
                    if (g == nb * 2 + col) v = val[nb][col];
            const int p = pool0 + warp * 8 * kNt + (g >> 1) * 8 + 2 * t + (g & 1);
            if (p < NP) OUT[(int64_t)r * NP + p] = p < npool ? v : -INFINITY;
        }
    }
}

}  // namespace tf_glm53_scores

extern "C" __constant__ unsigned int tf_glm53_scores_smem = sizeof(tf_glm53_scores::Shared);
