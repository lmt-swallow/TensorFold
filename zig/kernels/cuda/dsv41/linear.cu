// EXL3 linear, any codebook and width, 1-128 rows: a row's bits depend only on it (mma keeps rows apart, K ranges fixed by (K, N), fixed-order sums).

// Zig build: the fork's kernels verbatim (bertholomus/TensorFold deepseek-v41-tp2, cuda/exl3/linear.cu), the torch
// extension's host glue dropped; namespace tf_exl3_linear so the fatbin exports every instantiation by name.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "linear_common.cuh"

using namespace tf_exl3;

namespace tf_exl3_linear {

__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype,
                                                     const half* __restrict__ suh, half* __restrict__ xh, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * K + k, v);
    load4(suh, F16, k, s);
    rot128_in(v, s, lane);
    store4(xh, F16, (size_t)row * K + k, v);
}

template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32) linear_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int nb = blockIdx.x, split = blockIdx.y, NB = gridDim.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

// W_q [K, N] fp16 from the trellis words (tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb): one warp per tile.
template <int K2, int CB>
__global__ void __launch_bounds__(32) unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N,
                                                    int64_t stride_k, int64_t stride_nb) {
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x;
    uint32_t w[lane_words<K2>()];
    ldg_lane_words<K2>(T + kt * stride_k + (nt >> 3) * stride_nb + (nt & 7) * tile_words<K2>(), lane, w);
    uint32_t b[2][2];
    decode_lane<K2, CB>(w, lane, b[0], b[1]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t v = b[j >> 2][(j >> 1) & 1];
        const unsigned short h = (j & 1) ? (unsigned short)(v >> 16) : (unsigned short)(v & 0xffffu);
        W[(size_t)(kt * 16 + value_row(lane, j)) * N + nt * 16 + value_col(lane, j)] = __ushort_as_half(h);
    }
}

// The prompt GEMM's two preparations in one launch (blocks 0 .. rot_blocks - 1: rot_in_kernel's programs; the rest:
// unpack_kernel's tiles, four a block), so the rows' rotation (DRAM-bound) runs beside the weights' decode. Every
// value is computed by the same code as those kernels': xh rows from rot_in_kernel's warp program (its rows read
// x_stride elements apart instead of K, for a column block of a wider matrix), W_q tiles from unpack_kernel's lane
// decode; the tile is staged in shared memory and stored as 16-byte rows instead of 2-byte values (the same bytes).
// grid (rot_blocks + ceil(K / 16 * N / 16 / 4)), 128 threads; rot_blocks = ceil(K / 512) * M.
template <int K2, int CB>
__device__ __forceinline__ void prep(const void* __restrict__ x, int x_dtype, int x_stride, const half* __restrict__ suh,
                                     half* __restrict__ xh, int K, int rot_blocks, const uint32_t* __restrict__ T,
                                     half* __restrict__ W, int N, int64_t stride_k, int64_t stride_nb) {
    __shared__ __align__(16) unsigned short tile[4][16 * 16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if ((int)blockIdx.x < rot_blocks) {
        const int xb = (K / 128 + 3) / 4;
        const int blk = (blockIdx.x % xb) * 4 + warp, row = blockIdx.x / xb;
        if (blk * 128 >= K) return;
        const int k = blk * 128 + 4 * lane;
        float v[4], sc[4];
        load4(x, x_dtype, (size_t)row * x_stride + k, v);
        load4(suh, F16, k, sc);
        rot128_in(v, sc, lane);
        store4(xh, F16, (size_t)row * K + k, v);
        return;
    }
    const int t = ((int)blockIdx.x - rot_blocks) * 4 + warp, nts = N / 16;
    if (t >= (K / 16) * nts) return;
    const int kt = t / nts, nt = t % nts;
    uint32_t w[lane_words<K2>()];
    ldg_lane_words<K2>(T + kt * stride_k + (nt >> 3) * stride_nb + (nt & 7) * tile_words<K2>(), lane, w);
    uint32_t b[2][2];
    decode_lane<K2, CB>(w, lane, b[0], b[1]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t v = b[j >> 2][(j >> 1) & 1];
        tile[warp][value_row(lane, j) * 16 + value_col(lane, j)] = (j & 1) ? (unsigned short)(v >> 16) : (unsigned short)(v & 0xffffu);
    }
    __syncwarp();
    const int r = lane >> 1, c8 = (lane & 1) * 8;
    *reinterpret_cast<uint4*>(W + (size_t)(kt * 16 + r) * N + nt * 16 + c8) = *reinterpret_cast<const uint4*>(&tile[warp][r * 16 + c8]);
}

#define TF_PREP(K2_, CB_)                                                                                            \
    extern "C" __global__ void __launch_bounds__(128) tf_exl3_prep_##K2_##_##CB_(                                    \
        const void* x, int x_dtype, int x_stride, const half* suh, half* xh, int K, int rot_blocks, const uint32_t* T, \
        half* W, int N, int64_t stride_k, int64_t stride_nb) {                                                       \
        prep<K2_, CB_>(x, x_dtype, x_stride, suh, xh, K, rot_blocks, T, W, N, stride_k, stride_nb);                  \
    }
TF_EXL3_ALL(TF_PREP)
#undef TF_PREP

#define TF_INST(K2_, CB_)                                                                                            \
    template __global__ void __launch_bounds__(2 * 32) linear_kernel<K2_, CB_, 2>(                                  \
        const half*, const uint32_t*, long long, long long, const half*, const half*, void*, int, float*, int*,     \
        int, int, int, int);                                                                                        \
    template __global__ void __launch_bounds__(4 * 32) linear_kernel<K2_, CB_, 4>(                                  \
        const half*, const uint32_t*, long long, long long, const half*, const half*, void*, int, float*, int*,     \
        int, int, int, int);                                                                                        \
    template __global__ void __launch_bounds__(8 * 32) linear_kernel<K2_, CB_, 8>(                                  \
        const half*, const uint32_t*, long long, long long, const half*, const half*, void*, int, float*, int*,     \
        int, int, int, int);                                                                                        \
    template __global__ void __launch_bounds__(32) unpack_kernel<K2_, CB_>(const uint32_t*, half*, int, int64_t,    \
                                                                           int64_t);
TF_EXL3_ALL(TF_INST)
#undef TF_INST

}  // namespace tf_exl3_linear
