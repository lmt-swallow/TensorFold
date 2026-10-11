// Prompt chunks' gate/up as grouped_mma_kernel computes it, with the work spread more evenly over a program's warps.
//
// grouped_mmb_kernel takes grouped_mma_kernel's programs (64 member rows by 64 columns of one expert matrix, eight
// warps, MMA_KB k tiles a step, each step's sixteen weight tiles decoded once into shared memory as mma B fragments,
// the rows staged by cp.async and read by ldmatrix) and writes the same bits: every output is the same mma.m16n8k16
// chains over the same k tiles in the same order (SK x WK chains from zero, a split's chains added in order, the splits
// added in order from 0 under FOLD), from the same decoded fragments and the same A tiles. What differs is only who
// issues which instruction:
//  - the mma of a block with at most 32 member rows (an average routed expert at 2,048 rows has ~32) run on all eight
//    warps (each warp one 16-row group by one n16 tile, or by one n8 half at 16 rows) instead of on the two or four
//    warps whose row groups hold rows, so the step barrier no longer waits on the busiest warps' 16 mma;
//  - the cp.async source and the step's pointers advance by increments instead of being recomputed from the step.
// (GB10, a TP2 rank's gate/up over 40 layers of a 2,048-row chunk: 448 -> 423 ms, 12% fewer instructions; loading the
// trellis words two steps ahead through a register copy instead was slower: 505 ms.)
#pragma once

#include "experts_grouped.cuh"

namespace tf_exl3x {

// a warp's n8 accumulators: 4 (MODE 4: a 16-row group by 32 columns), 2 (MODE 2: by 16), 1 (MODE 1: by 8)
template <int MODE>
__host__ __device__ constexpr int mmb_nj() {
    return MODE == 4 ? 4 : (MODE == 2 ? 2 : 1);
}

template <int CB, int K2, bool FOLD, int KB, int MODE>
__device__ __forceinline__ void mmb_slice(const uint32_t* __restrict__ T, int NTILES, int KT, int per_range, int WK,
                                          int nt0, const half* __restrict__ X, int K, const int* rows_sh, bool live,
                                          int warp, int lane, half (&As)[MMA_STAGES][MMA_ROWS][KB * 16 + 8],
                                          uint4 (&bsh)[2][KB][4][32], float (&tot)[mmb_nj<MODE>()][4]) {
    constexpr int NJ = mmb_nj<MODE>();
    constexpr int DEC = KB / 2;                               // weight tiles a warp decodes a step
    constexpr int LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * Fmt<K2>::TW;
    const size_t kstep = (size_t)KB * kstride;
    const int dk = (warp >> 2) * DEC, dn = warp & 3;          // the tiles this warp decodes (grouped_mma's)
    const uint32_t* tp = T + ((size_t)dk * NTILES + nt0 + dn) * Fmt<K2>::TW + lane;
    // the mma this warp issues: its 16-row group and its first n16 tile / n8 half
    const int rg = MODE == 4 ? (warp & 3) : (MODE == 2 ? (warp & 1) : 0);
    const int nb = MODE == 4 ? 2 * (warp >> 2) : (warp >> 1);
    const int hb = MODE == 1 ? (warp & 1) : 0;
    const int NS = KT / KB;
    constexpr int CPR = KB / 2;
    const int crow = threadIdx.x >> 2, cpart = threadIdx.x & 3;
    const int cr = rows_sh[crow];
    const half* src = X + (size_t)(cr < 0 ? 0 : cr) * K + cpart * 8;
    const int lrow = rg * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

#pragma unroll
    for (int st = 0; st < MMA_STAGES - 1; ++st) {
        if (st < NS)
#pragma unroll
            for (int c = 0; c < CPR; ++c)
                cp_async16(&As[st][crow][(cpart + 4 * c) * 8], src + st * KB * 16 + 32 * c, cr >= 0);
        cp_async_commit();
    }
    // w0: the words of the step decoded next (loaded a step ahead, as grouped_mma_kernel)
    uint32_t w0[DEC][LW];
#pragma unroll
    for (int d = 0; d < DEC; ++d) load_words<K2>(w0[d], tp + (size_t)d * kstride, lane);
#pragma unroll
    for (int d = 0; d < DEC; ++d) {
        uint32_t b0[2], b1[2];
        decode_tile<CB, K2>(w0[d], map, lane, b0, b1);
        bsh[0][dk + d][dn][lane] = make_uint4(b0[0], b0[1], b1[0], b1[1]);
    }
    if (NS > 1)
#pragma unroll
        for (int d = 0; d < DEC; ++d) load_words<K2>(w0[d], tp + kstep + (size_t)d * kstride, lane);
    const uint32_t* tnext = tp + 2 * kstep;                   // the words of step st + 2
    const half* snext = src + (size_t)(MMA_STAGES - 1) * KB * 16;   // the rows of step st + MMA_STAGES - 1
    int sahead = MMA_STAGES - 1, stage = 0;
    cp_async_wait<MMA_STAGES - 2>();
    __syncthreads();

    float acc[NJ][4], sacc[NJ][4];
#pragma unroll
    for (int i = 0; i < NJ; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][c] = sacc[i][c] = 0.f;
    int left = per_range, wk = 0;

    for (int st = 0; st < NS; ++st) {
        const int buf = st & 1;
        if (st + MMA_STAGES - 1 < NS)
#pragma unroll
            for (int c = 0; c < CPR; ++c)
                cp_async16(&As[sahead][crow][(cpart + 4 * c) * 8], snext + 32 * c, cr >= 0);
        cp_async_commit();
        snext += KB * 16;
        sahead = sahead == MMA_STAGES - 1 ? 0 : sahead + 1;
        uint32_t fr[DEC][4];
        if (st + 1 < NS) {
#pragma unroll
            for (int d = 0; d < DEC; ++d) {
                uint32_t b0[2], b1[2];
                decode_tile<CB, K2>(w0[d], map, lane, b0, b1);
                fr[d][0] = b0[0]; fr[d][1] = b0[1]; fr[d][2] = b1[0]; fr[d][3] = b1[1];
            }
            if (st + 2 < NS)
#pragma unroll
                for (int d = 0; d < DEC; ++d) load_words<K2>(w0[d], tnext + (size_t)d * kstride, lane);
            tnext += kstep;
        }
        if (live) {
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) {
                uint32_t a[4];
                ldmatrix_x4(a, &As[stage][lrow][kk * 16 + lcol]);
                if constexpr (MODE == 4) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const uint4 b = bsh[buf][kk][nb + j][lane];
                        const uint32_t bl[2] = {b.x, b.y}, bh[2] = {b.z, b.w};
                        mma16816(acc[2 * j], a, bl);
                        mma16816(acc[2 * j + 1], a, bh);
                    }
                } else if constexpr (MODE == 2) {
                    const uint4 b = bsh[buf][kk][nb][lane];
                    const uint32_t bl[2] = {b.x, b.y}, bh[2] = {b.z, b.w};
                    mma16816(acc[0], a, bl);
                    mma16816(acc[1], a, bh);
                } else {
                    const uint4 b = bsh[buf][kk][nb][lane];
                    const uint32_t bb[2] = {hb ? b.z : b.x, hb ? b.w : b.y};
                    mma16816(acc[0], a, bb);
                }
            }
        }
        left -= KB;
        if (left == 0) {                                      // a chain ends: into its split's sum, in order
#pragma unroll
            for (int i = 0; i < NJ; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sacc[i][c] = wk == 0 ? acc[i][c] : sacc[i][c] + acc[i][c];
                    acc[i][c] = 0.f;
                }
            left = per_range;
            if (++wk == WK) {                                 // a split ends
                wk = 0;
#pragma unroll
                for (int i = 0; i < NJ; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c) tot[i][c] = FOLD ? tot[i][c] + sacc[i][c] : sacc[i][c];
            }
        }
        if (st + 1 < NS)
#pragma unroll
            for (int d = 0; d < DEC; ++d)
                bsh[buf ^ 1][dk + d][dn][lane] = make_uint4(fr[d][0], fr[d][1], fr[d][2], fr[d][3]);
        stage = stage == MMA_STAGES - 1 ? 0 : stage + 1;
        cp_async_wait<MMA_STAGES - 2>();
        __syncthreads();
    }
}

// The program's member rows, its mode (by the 16-row groups that hold rows: they are a prefix) and its output.
template <int CB, int K2, bool FOLD, int KB, int MODE>
__device__ __forceinline__ void mmb_run(const uint32_t* __restrict__ T, int NTILES, int KT, int per_range, int WK,
                                        int nt0, const half* __restrict__ X, int K, const int* rows_sh, int warp,
                                        int lane, half (&As)[MMA_STAGES][MMA_ROWS][KB * 16 + 8],
                                        uint4 (&bsh)[2][KB][4][32], float* __restrict__ Z, size_t zbase, int N) {
    constexpr int NJ = mmb_nj<MODE>();
    const int rg = MODE == 4 ? (warp & 3) : (MODE == 2 ? (warp & 1) : 0);
    const bool live = rows_sh[rg * 16] >= 0;
    float tot[NJ][4];
#pragma unroll
    for (int i = 0; i < NJ; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) tot[i][c] = 0.f;
    mmb_slice<CB, K2, FOLD, KB, MODE>(T, NTILES, KT, per_range, WK, nt0, X, K, rows_sh, live, warp, lane, As, bsh, tot);
    if (!live) return;
    const int g8 = lane >> 2, t = lane & 3;
    const int r0 = rows_sh[rg * 16 + g8], r1 = rows_sh[rg * 16 + g8 + 8];
    // the warp's first column: MODE 4 its 32 (n16 tiles 2 (w / 4), + 1), MODE 2 its n16 tile, MODE 1 its n8 half
    const int c0 = nt0 * 16 + (MODE == 4 ? 32 * (warp >> 2) : (MODE == 2 ? 16 * (warp >> 1) : 16 * (warp >> 1) + 8 * (warp & 1)));
#pragma unroll
    for (int i = 0; i < NJ; ++i) {
        const int col = c0 + i * 8 + 2 * t;
        if (r0 >= 0) *reinterpret_cast<float2*>(Z + zbase + (size_t)r0 * N + col) = make_float2(tot[i][0], tot[i][1]);
        if (r1 >= 0) *reinterpret_cast<float2*>(Z + zbase + (size_t)r1 * N + col) = make_float2(tot[i][2], tot[i][3]);
    }
}

// grouped_mma_kernel's arguments, grid and output (Z [mats, 1, P, N] with FOLD, or the split's sum with SK 1).
template <int CB, int LO, int HI, bool FOLD, int KB>
__global__ void __launch_bounds__(256) grouped_mmb_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int WK, int maxm, int slots, int nexp,
    const int* __restrict__ work) {
    static_assert(KB <= 4, "the static shared memory holds up to 4 k tiles a step");
    const int zz = (int)(blockIdx.z % nexp);
    const int u = work ? work[2 * zz] : zz;
    if (u >= ucount[0]) return;
    const int mb = work ? work[2 * zz + 1] : (int)blockIdx.x;
    const int mat = (int)(blockIdx.z / nexp);
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int KT = K >> 4, NTILES = N >> 4;
    const int per_range = KT / (SK * WK);

    __shared__ int rows_sh[MMA_ROWS];
    __shared__ __align__(16) half As[MMA_STAGES][MMA_ROWS][KB * 16 + 8];
    __shared__ uint4 bsh[2][KB][4][32];
    for (int i = threadIdx.x; i < MMA_ROWS; i += 256) {
        const int m = mb * MMA_ROWS + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                               // members come first, so this block is empty
    const int groups = (rows_sh[16] >= 0) + (rows_sh[32] >= 0) + (rows_sh[48] >= 0) + 1;
    const int nt0 = blockIdx.y * (MMA_COLS / 16);
    const size_t zbase = (size_t)mat * P * N;

    switch (k2) {
#define TF_MMB_CASE(K2_)                                                                                        \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI) {                                                                 \
            if (groups > 2)                                                                                     \
                mmb_run<CB, K2_, FOLD, KB, 4>(T, NTILES, KT, per_range, WK, nt0, X, K, rows_sh, warp, lane, As, \
                                              bsh, Z, zbase, N);                                                \
            else if (groups == 2)                                                                               \
                mmb_run<CB, K2_, FOLD, KB, 2>(T, NTILES, KT, per_range, WK, nt0, X, K, rows_sh, warp, lane, As, \
                                              bsh, Z, zbase, N);                                                \
            else                                                                                                \
                mmb_run<CB, K2_, FOLD, KB, 1>(T, NTILES, KT, per_range, WK, nt0, X, K, rows_sh, warp, lane, As, \
                                              bsh, Z, zbase, N);                                                \
        } else {                                                                                                \
            __trap();                                                                                           \
        }                                                                                                       \
        break;
        TF_MMB_CASE(2)
        TF_MMB_CASE(3)
        TF_MMB_CASE(4)
        TF_MMB_CASE(5)
        TF_MMB_CASE(6)
        TF_MMB_CASE(7)
        TF_MMB_CASE(8)
        TF_MMB_CASE(9)
        TF_MMB_CASE(10)
        TF_MMB_CASE(11)
        TF_MMB_CASE(12)
        TF_MMB_CASE(13)
        TF_MMB_CASE(14)
        TF_MMB_CASE(15)
        TF_MMB_CASE(16)
#undef TF_MMB_CASE
        default:
            __trap();
    }
}

}  // namespace tf_exl3x
