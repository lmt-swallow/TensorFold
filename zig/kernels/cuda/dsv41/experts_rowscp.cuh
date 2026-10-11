// Prompt chunks' down projection as grouped_rows_kernel<CB, 8, 4, 1, LO, HI, 2, false> computes it, with its loads
// in flight several k tiles ahead.
//
// grouped_rows_cp_kernel runs grouped_rows_kernel's programs (an expert's 32 member rows by 128 columns, four warps,
// each warp one chain of k tiles of the split, the warps' sums added in warp order) and writes the same bits: every
// output is the same mma.m16n8k16 chain from zero over the same k tiles in the same order, from the same decoded
// fragments and the same A values. Only where the operands come from differs: each warp copies its k tiles' trellis
// words (a step's 8 weight tiles are contiguous) and its 32 rows' 16 columns into its own ring of rcp_stages
// shared-memory stages by cp.async, rcp_stages - 1 k tiles ahead, and reads them from there (grouped_rows holds one
// tile's words and the rows' fragments a tile ahead in registers, its loads exposed). The warps' sums go through the
// same shared memory once every ring is done.
#pragma once

#include "experts_grouped.cuh"

namespace tf_exl3x {

constexpr int RCP_NT = 8;                 // n16 tiles a program (128 columns)
constexpr int RCP_W = 4;                  // warps
constexpr int RCP_G = 2;                  // member tiles a program (32 rows)
// k tiles in a warp's ring: 4, or 3 at 16 half-bits (the rings within 48 KB of static shared memory)
template <int HI>
__host__ __device__ constexpr int rcp_stages() { return HI > 10 ? 3 : 4; }
constexpr int RCP_AW = 12;                // 32-bit words a staged row (16 halfs + 8 pad: fragment loads in distinct banks)

// 32-bit words of a ring stage: the step's trellis words at HI half-bits, then the 32 rows
template <int HI>
__host__ __device__ constexpr int rcp_stage_words() { return RCP_NT * 4 * HI + RCP_G * 16 * RCP_AW; }
template <int HI>
__host__ __device__ constexpr int rcp_smem_words() {
    return (RCP_W * rcp_stages<HI>() * rcp_stage_words<HI>() > RCP_W * 16 * RCP_NT * 16)
               ? RCP_W * rcp_stages<HI>() * rcp_stage_words<HI>()
               : RCP_W * 16 * RCP_NT * 16;
}

// One warp's chain: k tiles [kt0, kt0 + nkt) of n16 tiles nt0 .. + 7 for the program's live member tiles (grouped_rows'
// warp_tiles_rows, its operands from the ring).
template <int CB, int K2, int HI>
__device__ __forceinline__ void rcp_warp(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                         const half* __restrict__ X, int K, const int* rows_sh, int live, int lane,
                                         uint32_t* ring, float (&acc)[RCP_G][RCP_NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    constexpr int SW = rcp_stage_words<HI>();
    constexpr int WOFF = RCP_NT * 4 * HI;
    constexpr int RCP_STAGES = rcp_stages<HI>();
    const LaneMap<K2> map(lane);
    const int g8 = lane >> 2, t = lane & 3;
    // the lane's row copies: rows lane / 2 and lane / 2 + 16 of the 32, 8 halfs (lane % 2) of their 16 columns a k tile
    const int crow = lane >> 1, chalf = lane & 1;
    const int rr0 = rows_sh[crow], rr1 = rows_sh[crow + 16];
    const half* asrc0 = X + (size_t)(rr0 < 0 ? 0 : rr0) * K + chalf * 8;
    const half* asrc1 = X + (size_t)(rr1 < 0 ? 0 : rr1) * K + chalf * 8;
    auto stage = [&](int it) {
        uint32_t* ws = ring + (it % RCP_STAGES) * SW;
        const int kt = kt0 + it;
        const uint32_t* src = T + ((size_t)kt * NTILES + nt0) * TW;
        for (int c = lane; c < RCP_NT * TW / 4; c += 32) cp_async16(ws + 4 * c, src + 4 * c, true);
        cp_async16(ws + WOFF + crow * RCP_AW + chalf * 4, asrc0 + kt * 16, rr0 >= 0);
        cp_async16(ws + WOFF + (crow + 16) * RCP_AW + chalf * 4, asrc1 + kt * 16, rr1 >= 0);
        cp_async_commit();
    };
#pragma unroll
    for (int s = 0; s < RCP_STAGES - 1; ++s) {
        if (s < nkt) stage(s);
        else cp_async_commit();
    }
    for (int it = 0; it < nkt; ++it) {
        if (it + RCP_STAGES - 1 < nkt) stage(it + RCP_STAGES - 1);
        else cp_async_commit();
        cp_async_wait<RCP_STAGES - 1>();
        __syncwarp();
        const uint32_t* ws = ring + (it % RCP_STAGES) * SW;
        const uint32_t* as = ws + WOFF;
        uint32_t a[RCP_G][4];
#pragma unroll
        for (int g = 0; g < RCP_G; ++g) {
            a[g][0] = as[(g * 16 + g8) * RCP_AW + t];
            a[g][1] = as[(g * 16 + g8 + 8) * RCP_AW + t];
            a[g][2] = as[(g * 16 + g8) * RCP_AW + t + 4];
            a[g][3] = as[(g * 16 + g8 + 8) * RCP_AW + t + 4];
        }
#pragma unroll
        for (int i = 0; i < RCP_NT; ++i) {
            uint32_t w[LW];
#pragma unroll
            for (int l = 0; l < LW; ++l) {
                if constexpr ((TW % 32) == 0)
                    w[l] = ws[i * TW + l * 32 + lane];
                else
                    w[l] = (l * 32 + lane < TW) ? ws[i * TW + l * 32 + lane] : 0u;
            }
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w, map, lane, b0, b1);
#pragma unroll
            for (int g = 0; g < RCP_G; ++g) {
                if (g < live) {
                    mma16816(acc[g][i][0], a[g], b0);
                    mma16816(acc[g][i][1], a[g], b1);
                }
            }
        }
        __syncwarp();                                     // the stage is read before a later copy refills it
    }
}

// grouped_rows_kernel<CB, 8, 4, 1, LO, HI, 2, false>'s arguments, grid and output (Z [mats, SK, P, N]).
template <int CB, int LO, int HI>
__global__ void __launch_bounds__(RCP_W * 32) grouped_rows_cp_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, int nexp, const int* __restrict__ work) {
    constexpr int GR = 16 * RCP_G;
    constexpr int TPT = 16 * RCP_NT * 16 / (RCP_W * 32);
    const int zz = (int)(blockIdx.z % nexp);
    const int u = work ? work[2 * zz] : zz;
    if (u >= ucount[0]) return;
    const int mg = work ? work[2 * zz + 1] : (int)blockIdx.x;
    const int zi = (int)(blockIdx.z / nexp);
    const int split = zi % SK;
    const int mat = zi / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[GR];
    __shared__ __align__(16) uint32_t smem[rcp_smem_words<HI>()];
    for (int i = threadIdx.x; i < GR; i += RCP_W * 32) {
        const int m = mg * GR + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;
    int live = 0;
#pragma unroll
    for (int g = 0; g < RCP_G; ++g) live += rows_sh[g * 16] >= 0;

    const int per_split = KT / SK, per_warp = per_split / RCP_W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * RCP_NT;
    float acc[RCP_G][RCP_NT][2][4];
#pragma unroll
    for (int g = 0; g < RCP_G; ++g)
#pragma unroll
        for (int i = 0; i < RCP_NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[g][i][h][c] = 0.f;
    uint32_t* ring = smem + warp * rcp_stages<HI>() * rcp_stage_words<HI>();

    switch (k2) {
#define TF_RCP_CASE(K2_)                                                                                        \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            rcp_warp<CB, K2_, HI>(T, NTILES, kt0, per_warp, nt0, X, K, rows_sh, live, lane, ring, acc);         \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_RCP_CASE(2)
        TF_RCP_CASE(3)
        TF_RCP_CASE(4)
        TF_RCP_CASE(5)
        TF_RCP_CASE(6)
        TF_RCP_CASE(7)
        TF_RCP_CASE(8)
        TF_RCP_CASE(9)
        TF_RCP_CASE(10)
        TF_RCP_CASE(11)
        TF_RCP_CASE(12)
        TF_RCP_CASE(13)
        TF_RCP_CASE(14)
        TF_RCP_CASE(15)
        TF_RCP_CASE(16)
#undef TF_RCP_CASE
        default:
            __trap();
    }
    cp_async_wait<0>();
    __syncthreads();                                      // every ring is done: the shared memory takes the sums

    // the warps' partial sums, added in warp order, one member tile after another (grouped_rows_kernel's)
    float (*red)[16][RCP_NT * 16] = reinterpret_cast<float (*)[16][RCP_NT * 16]>(smem);
#pragma unroll
    for (int g = 0; g < RCP_G; ++g) {
        if (g < live) {
#pragma unroll
            for (int i = 0; i < RCP_NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    red[warp][g8][col] = acc[g][i][h][0];
                    red[warp][g8][col + 1] = acc[g][i][h][1];
                    red[warp][g8 + 8][col] = acc[g][i][h][2];
                    red[warp][g8 + 8][col + 1] = acc[g][i][h][3];
                }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < TPT; ++j) {
                const int idx = threadIdx.x + j * RCP_W * 32;
                const int row = idx / (RCP_NT * 16), col = idx % (RCP_NT * 16);
                float s = red[0][row][col];
#pragma unroll
                for (int w = 1; w < RCP_W; ++w) s += red[w][row][col];
                const int r = rows_sh[g * 16 + row];
                if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
            }
            __syncthreads();
        }
    }
}

}  // namespace tf_exl3x
