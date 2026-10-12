// Ours: a prompt chunk's routed EXL3 experts (experts.py `routed` over its 1,024-row windows, then the bf16 copy)
// as one window of the whole chunk in five launches, with the bytes of 0.6.6's: the members listed per expert, the
// work items planned per member group, then one program per (expert, 32 members, 128 columns) that rotates its
// members' inputs into shared memory, runs every K split of gate and up, sums them and runs the SwiGLU epilogue
// (gate/up), and one per (expert, 64 members, 256 columns) that runs the down projection and its epilogue straight
// into the chunk's bf16 slots (`buf.y[r0:r0 + n].copy_(y)`).
//
// Bits. Every value is the reference pipeline's (rot_in_kernel, grouped_kernel, gateup_epilogue_kernel,
// grouped_kernel, down_epilogue_kernel, tf_qfn_f32_to_bf16):
// - the rotated inputs are rot_in_kernel's expressions on the same values (one warp a 128-block, fwht128);
// - grouped_kernel's partial sum of split s is ((w0 + w1) + w2) + w3, where chain w is an mma.m16n8k16 chain from 0
//   over the K tiles s * KT/SK + w * KT/(SK * 4) onward in order (4 warps); a member's row of an mma tile never
//   depends on the tile's other rows, so which members share a tile does not matter;
// - the epilogues' split sums are 0 + z0 + z1 + ... in split order, then their expressions verbatim.
// The tiles a program decodes depend on (expert, n tile, K tile) alone, so each is decoded once per member group; a
// program skips its member tiles past the last member.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "experts_grouped.cuh"

namespace tf_qfn_xp {

using namespace tf_exl3x;

constexpr float HAD_SCALE = 0.08838834764831845f;  // experts.cu's
constexpr int MT_GU = 2;                             // gate/up: member tiles (16 rows) a program (two programs an SM)
constexpr int MT_D = 4;                              // down: member tiles a program
constexpr int MG_MAX = 16 * MT_D;
constexpr int WARPS = 8;                             // gate/up: one 16-column n tile each, 128 columns a program
constexpr int WARPS_D = 16;                          // down: 256 columns a program
constexpr int KS = 640;                              // K a split covers (grouped_kernel: K / SK)
constexpr int CHAINS = 4;                            // grouped_kernel's warps: the chains a split is summed from
constexpr int CT = KS / 16 / CHAINS;                 // K tiles a chain
constexpr int APAD = KS + 8;                         // a staged row's halves (bank spread)
// dynamic shared bytes: the staged rows (then the epilogue's sums)
constexpr int SMEM_GU = 16 * MT_GU * APAD * 2;
constexpr int SMEM_D = 16 * MT_D * APAD * 2;
constexpr int ITEM_BITS = 12;                        // an item: expert << ITEM_BITS | member group

// experts.cu's fwht128 after its register strides: the butterflies across lanes.
__device__ __forceinline__ void fwht_lanes(float (&v)[4], int lane) {
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
        }
    }
}

// experts.cu's fwht128 on values computed before it (no product left to fuse into its first additions).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = __fadd_rn(v[0], v[1]), b = __fsub_rn(v[0], v[1]), c = __fadd_rn(v[2], v[3]), d = __fsub_rn(v[2], v[3]);
    v[0] = __fadd_rn(a, c); v[1] = __fadd_rn(b, d); v[2] = __fsub_rn(a, c); v[3] = __fsub_rn(b, d);
    fwht_lanes(v, lane);
}

// rot_in_kernel's fwht128(x * suh) as its SASS computes it (nvcc contracts the products into the first butterflies):
// a = fma(x0, s0, x1 s1), b = fma(x0, s0, -x1 s1), c = fma(x2, s2, x3 s3), d = fma(x2, s2, -x3 s3).
__device__ __forceinline__ void fwht128_products(const float (&x)[4], const float (&s)[4], float (&v)[4], int lane) {
    const float p1 = __fmul_rn(x[1], s[1]), p3 = __fmul_rn(x[3], s[3]);
    const float a = __fmaf_rn(x[0], s[0], p1), b = __fmaf_rn(x[0], s[0], -p1);
    const float c = __fmaf_rn(x[2], s[2], p3), d = __fmaf_rn(x[2], s[2], -p3);
    v[0] = __fadd_rn(a, c); v[1] = __fadd_rn(b, d); v[2] = __fsub_rn(a, c); v[3] = __fsub_rn(b, d);
    fwht_lanes(v, lane);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// tf_qfn_f32_to_bf16's conversion.
__device__ __forceinline__ uint16_t to_bf16(float x) {
    const uint32_t u = __float_as_uint(x);
    return x != x ? (uint16_t)0x7FC0 : (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Program e: expert e's slots (indices into pick, ascending) into lists[e * maxm ..], their count into cnt[e].
__device__ __forceinline__ void list_kernel(const int* __restrict__ pick, int n, int* __restrict__ lists,
                                                   int maxm, int* __restrict__ cnt) {
    __shared__ int wsum[8];
    const int e = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int base = 0;
    for (int i0 = 0; i0 < n; i0 += 256) {
        const int i = i0 + threadIdx.x;
        const bool f = i < n && pick[i] == e;
        const unsigned bal = __ballot_sync(0xffffffffu, f);
        if (lane == 0) wsum[warp] = __popc(bal);
        __syncthreads();
        int before = 0, total = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            before += w < warp ? wsum[w] : 0;
            total += wsum[w];
        }
        if (f) {
            const int r = base + before + __popc(bal & ((1u << lane) - 1u));
            if (r < maxm) lists[(size_t)e * maxm + r] = i;
        }
        base += total;
        __syncthreads();
    }
    if (threadIdx.x == 0) cnt[e] = min(base, maxm);
}

// One block of 1,024 threads (E <= 1,024): the work items (expert, group of mg members) in expert order, their count.
__device__ __forceinline__ void plan_kernel(const int* __restrict__ cnt, int E, int* __restrict__ items,
                                                    int* __restrict__ nitems, int MG) {
    __shared__ int sh[1024];
    const int t = threadIdx.x;
    const int groups = t < E ? (cnt[t] + MG - 1) / MG : 0;
    sh[t] = groups;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        const int v = t >= o ? sh[t - o] : 0;
        __syncthreads();
        sh[t] += v;
        __syncthreads();
    }
    const int off = sh[t] - groups;
    for (int g = 0; g < groups; ++g) items[off + g] = (t << ITEM_BITS) | g;
    if (t == 1023) nitems[0] = sh[1023];
}

// A warp's split: its n tile `nt` over the split's CHAINS chains of CT K tiles from kt_base, the chains summed in
// order into cur, for the first nv member tiles of A (the split's MG staged rows, zero past the members).
template <int CB, int K2, int MT>
__device__ __forceinline__ void split_chains(const uint32_t* __restrict__ T, int NTILES, int nt, int kt_base,
                                             const half* A, int nv, int lane, float (&cur)[MT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const int g = lane >> 2, t = lane & 3;
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt_base * NTILES + nt) * TW + lane;
    uint32_t nxt[LW];
    load_words<K2>(nxt, tp, lane);
#pragma unroll 1
    for (int c = 0; c < CHAINS; ++c) {
        float acc[MT][2][4];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[m][h][q] = 0.f;
#pragma unroll 2
        for (int it = 0; it < CT; ++it) {
            const int kk = c * CT + it;
            uint32_t w[LW];
#pragma unroll
            for (int l = 0; l < LW; ++l) w[l] = nxt[l];
            if (kk + 1 < CHAINS * CT) load_words<K2>(nxt, tp + (size_t)(kk + 1) * kstride, lane);
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w, map, lane, b0, b1);
            const int k = kk * 16 + 2 * t;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < nv) {
                    const half* r0 = A + (m * 16 + g) * APAD + k;
                    const half* r1 = r0 + 8 * APAD;
                    uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(r0), *reinterpret_cast<const uint32_t*>(r1),
                                     *reinterpret_cast<const uint32_t*>(r0 + 8),
                                     *reinterpret_cast<const uint32_t*>(r1 + 8)};
                    mma16816(acc[m][0], a, b0);
                    mma16816(acc[m][1], a, b1);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < 4; ++q) cur[m][h][q] = c == 0 ? acc[m][h][q] : cur[m][h][q] + acc[m][h][q];
    }
}

template <int CB, int LO, int HI, int MT>
__device__ __forceinline__ void split_any(int k2, const uint32_t* T, int NTILES, int nt, int kt_base, const half* A,
                                          int nv, int lane, float (&cur)[MT][2][4]) {
    switch (k2) {
#define TF_QFN_XP_CASE(K2_)                                                         \
    case K2_:                                                                       \
        if constexpr (K2_ >= LO && K2_ <= HI)                                       \
            split_chains<CB, K2_, MT>(T, NTILES, nt, kt_base, A, nv, lane, cur);    \
        else                                                                        \
            __trap();                                                               \
        break;
        TF_QFN_XP_CASE(2)
        TF_QFN_XP_CASE(3)
        TF_QFN_XP_CASE(4)
        TF_QFN_XP_CASE(5)
        TF_QFN_XP_CASE(6)
        TF_QFN_XP_CASE(7)
        TF_QFN_XP_CASE(8)
        TF_QFN_XP_CASE(9)
        TF_QFN_XP_CASE(10)
        TF_QFN_XP_CASE(11)
        TF_QFN_XP_CASE(12)
        TF_QFN_XP_CASE(13)
        TF_QFN_XP_CASE(14)
        TF_QFN_XP_CASE(15)
        TF_QFN_XP_CASE(16)
#undef TF_QFN_XP_CASE
        default:
            __trap();
    }
}

// The item's members (slot indices, -1 past them) into rows_p and its non-empty member tiles into nv; false when the
// program has no item.
template <int MG>
__device__ __forceinline__ bool members(const int* items, const int* nitems, const int* lists, int maxm,
                                        const int* cnt, int* rows_p, int& e, int& nv) {
    if ((int)blockIdx.x >= nitems[0]) return false;
    const int item = items[blockIdx.x];
    e = item >> ITEM_BITS;
    const int grp = item & ((1 << ITEM_BITS) - 1);
    const int c = cnt[e];
    if (threadIdx.x < MG) {
        const int idx = grp * MG + threadIdx.x;
        rows_p[threadIdx.x] = idx < c ? lists[(size_t)e * maxm + idx] : -1;
    }
    nv = (min(c - grp * MG, MG) + 15) / 16;
    __syncthreads();
    return true;
}

// Program (item, 128 columns of the width N): the members' gate and up over every split, summed, then
// gateup_epilogue_kernel's arithmetic into xd fp16 [P, N].
template <int CB, int LO, int HI>
__device__ __forceinline__ void gateup_body(
    const __nv_bfloat16* __restrict__ x, int x_stride, int slots, const int* __restrict__ items,
    const int* __restrict__ nitems, const int* __restrict__ lists, int maxm, const int* __restrict__ cnt,
    const int64_t* __restrict__ TPg, const int64_t* __restrict__ TPu, const int* __restrict__ K2g,
    const int* __restrict__ K2u, const half* __restrict__ suh_g, const half* __restrict__ suh_u,
    const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
    half* __restrict__ xd, int K, int N, float limit, int act_mode) {
    constexpr int MT = MT_GU, MG = 16 * MT;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int rows_p[MG];
    int e, nv;
    if (!members<MG>(items, nitems, lists, maxm, cnt, rows_p, e, nv)) return;
    half* A = reinterpret_cast<half*>(smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int nt = blockIdx.y * WARPS + warp, NTILES = N >> 4, KT = K >> 4, SK = K / KS;
    float tg[MT][2][4], tu[MT][2][4];
#pragma unroll 1
    for (int mat = 0; mat < 2; ++mat) {
        const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TPu[e] : TPg[e]);
        const int k2 = mat ? K2u[e] : K2g[e];
        const half* suh = mat ? suh_u : suh_g;
        float tot[MT][2][4];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < 4; ++q) tot[m][h][q] = 0.f;
#pragma unroll 1
        for (int s = 0; s < SK; ++s) {
            __syncthreads();
            // rot_in_kernel for the split's 128-blocks of every member
            for (int task = warp; task < nv * 16 * (KS / 128); task += WARPS) {
                const int m = task / (KS / 128), b = task % (KS / 128);
                half* dst = A + m * APAD + b * 128 + 4 * lane;
                const int p = rows_p[m];
                if (p < 0) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) dst[j] = __float2half_rn(0.f);
                    continue;
                }
                const int k0 = s * KS + b * 128 + 4 * lane;
                const __nv_bfloat16* xr = x + (size_t)(p / slots) * x_stride + k0;
                const half* su = suh + (size_t)e * K + k0;
                float xv[4], sv[4], v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    xv[j] = __bfloat162float(xr[j]);
                    sv[j] = __half2float(su[j]);
                }
                fwht128_products(xv, sv, v, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) dst[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
            __syncthreads();
            float cur[MT][2][4];
            split_any<CB, LO, HI>(k2, T, NTILES, nt, s * (KT / SK), A, nv, lane, cur);
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int q = 0; q < 4; ++q) tot[m][h][q] += cur[m][h][q];
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    if (mat) tu[m][h][q] = tot[m][h][q];
                    else tg[m][h][q] = tot[m][h][q];
                }
    }
    __syncthreads();
    float* fg = reinterpret_cast<float*>(smem);
    float* fu = fg + MG * 128;
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (m >= nv) continue;
            const int col = warp * 16 + h * 8 + 2 * t, row = m * 16 + g;
            fg[row * 128 + col] = tg[m][h][0];
            fg[row * 128 + col + 1] = tg[m][h][1];
            fg[(row + 8) * 128 + col] = tg[m][h][2];
            fg[(row + 8) * 128 + col + 1] = tg[m][h][3];
            fu[row * 128 + col] = tu[m][h][0];
            fu[row * 128 + col + 1] = tu[m][h][1];
            fu[(row + 8) * 128 + col] = tu[m][h][2];
            fu[(row + 8) * 128 + col + 1] = tu[m][h][3];
        }
    __syncthreads();
    for (int m = warp; m < nv * 16; m += WARPS) {
        const int p = rows_p[m];
        if (p < 0) continue;
        // gateup_epilogue_kernel from its split sums
        const int n = blockIdx.y * 128 + 4 * lane;
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gv[j] = fg[m * 128 + 4 * lane + j];
            uv[j] = fu[m * 128 + 4 * lane + j];
        }
        fwht128(gv, lane);
        fwht128(uv, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float act;
            if (act_mode == 0) {
                float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                                 limit);
                act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
            } else {
                float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
                float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
                act = gg / (1.f + expf(-gg)) * uu;
            }
            v[j] = __fmul_rn(__half2float(suh_d[(size_t)e * N + n + j]), act);  // not fused into fwht128
        }
        fwht128(v, lane);
        half* o = xd + (size_t)p * N + n;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
    }
}

// Program (item, 256 columns of the model width D): the members' down projection (one split of KS), then
// down_epilogue_kernel's arithmetic and tf_qfn_f32_to_bf16's conversion into out bf16 [P, D].
template <int CB, int LO, int HI>
__device__ __forceinline__ void down_body(const half* __restrict__ xd, const int* __restrict__ items,
                                          const int* __restrict__ nitems, const int* __restrict__ lists, int maxm,
                                          const int* __restrict__ cnt, const int64_t* __restrict__ TPd,
                                          const int* __restrict__ K2d, const half* __restrict__ svh_d,
                                          uint16_t* __restrict__ out, int K, int D) {
    constexpr int MT = MT_D, MG = 16 * MT;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int rows_p[MG];
    int e, nv;
    if (!members<MG>(items, nitems, lists, maxm, cnt, rows_p, e, nv)) return;
    half* A = reinterpret_cast<half*>(smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int nt = blockIdx.y * WARPS_D + warp, NTILES = D >> 4;
    for (int i = threadIdx.x; i < nv * 16 * (KS / 8); i += WARPS_D * 32) {
        const int m = i / (KS / 8), c8 = i % (KS / 8);
        const int p = rows_p[m];
        const uint4 v = p >= 0 ? *reinterpret_cast<const uint4*>(xd + (size_t)p * K + c8 * 8) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(A + m * APAD + c8 * 8) = v;
    }
    __syncthreads();
    float cur[MT][2][4];
    split_any<CB, LO, HI>(K2d[e], reinterpret_cast<const uint32_t*>(TPd[e]), NTILES, nt, 0, A, nv, lane, cur);
    __syncthreads();
    constexpr int W = WARPS_D * 16;  // the program's columns
    float* f = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (m >= nv) continue;
            const int col = warp * 16 + h * 8 + 2 * t, row = m * 16 + g;
            f[row * W + col] = 0.f + cur[m][h][0];
            f[row * W + col + 1] = 0.f + cur[m][h][1];
            f[(row + 8) * W + col] = 0.f + cur[m][h][2];
            f[(row + 8) * W + col + 1] = 0.f + cur[m][h][3];
        }
    __syncthreads();
    // down_epilogue_kernel: one warp a (member, 128-block)
    for (int task = warp; task < nv * 16 * (W / 128); task += WARPS_D) {
        const int m = task / (W / 128), blk = task % (W / 128);
        const int p = rows_p[m];
        if (p < 0) continue;
        const int n = blockIdx.y * W + blk * 128 + 4 * lane;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = f[m * W + blk * 128 + 4 * lane + j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[(size_t)p * D + n + j] = to_bf16(v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]));
    }
}

// The down projection with each item's rows staged once (TF_QFN_XP_DOWN=loop2|loop3): one program per item (16 *
// MT members) loops over every 256 columns of D, reusing the staged rows; the epilogue runs a 128-column half at a
// time through a [16 MT][128] fp32 buffer after the rows. The bits are down_body's.
template <int CB, int LO, int HI, int MT>
__device__ __forceinline__ void down_loop_body(const half* __restrict__ xd, const int* __restrict__ items,
                                               const int* __restrict__ nitems, const int* __restrict__ lists, int maxm,
                                               const int* __restrict__ cnt, const int64_t* __restrict__ TPd,
                                               const int* __restrict__ K2d, const half* __restrict__ svh_d,
                                               uint16_t* __restrict__ out, int K, int D) {
    constexpr int MG = 16 * MT;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int rows_p[MG];
    int e, nv;
    if (!members<MG>(items, nitems, lists, maxm, cnt, rows_p, e, nv)) return;
    half* A = reinterpret_cast<half*>(smem);
    float* f = reinterpret_cast<float*>(smem + MG * APAD * 2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = D >> 4;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(TPd[e]);
    const int k2 = K2d[e];
    for (int i = threadIdx.x; i < nv * 16 * (KS / 8); i += WARPS_D * 32) {
        const int m = i / (KS / 8), c8 = i % (KS / 8);
        const int p = rows_p[m];
        const uint4 v = p >= 0 ? *reinterpret_cast<const uint4*>(xd + (size_t)p * K + c8 * 8) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(A + m * APAD + c8 * 8) = v;
    }
    __syncthreads();
    constexpr int W = WARPS_D * 16;  // columns an iteration
#pragma unroll 1
    for (int c0 = 0; c0 < D; c0 += W) {
        float cur[MT][2][4];
        split_any<CB, LO, HI>(k2, T, NTILES, c0 / 16 + warp, 0, A, nv, lane, cur);
#pragma unroll 1
        for (int half_ = 0; half_ < W / 128; ++half_) {
            __syncthreads();
            if (warp / 8 == half_) {
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        if (m >= nv) continue;
                        const int col = (warp % 8) * 16 + h * 8 + 2 * t, row = m * 16 + g;
                        f[row * 128 + col] = 0.f + cur[m][h][0];
                        f[row * 128 + col + 1] = 0.f + cur[m][h][1];
                        f[(row + 8) * 128 + col] = 0.f + cur[m][h][2];
                        f[(row + 8) * 128 + col + 1] = 0.f + cur[m][h][3];
                    }
            }
            __syncthreads();
            // down_epilogue_kernel: one warp a member
            for (int m = warp; m < nv * 16; m += WARPS_D) {
                const int p = rows_p[m];
                if (p < 0) continue;
                const int n = c0 + half_ * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = f[m * 128 + 4 * lane + j];
                fwht128(v, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    out[(size_t)p * D + n + j] = to_bf16(v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]));
            }
        }
    }
}

}  // namespace tf_qfn_xp

extern "C" __global__ void __launch_bounds__(256) tf_qfn_xp_list(const int* pick, int n, int* lists, int maxm,
                                                                  int* cnt) {
    tf_qfn_xp::list_kernel(pick, n, lists, maxm, cnt);
}

// The words where a and b differ, added into *count (TF_QFN_PROMPT_EXPERTS=check).
extern "C" __global__ void tf_qfn_xp_count_diff(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b,
                                                unsigned long long n, unsigned long long* __restrict__ count) {
    unsigned long long c = 0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        c += a[i] != b[i];
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(count, c);
}

// gate/up (256 threads) and down (512 threads) for each codebook and K2 range grouped_launch picks:
// tf_qfn_xp_{gateup,down}_<cb>_<lo>_<hi> with SMEM_GU (41,472) and SMEM_D (82,944) dynamic shared bytes, the
// gate/up items in groups of 32 members, the down items in groups of 64 (tf_qfn_xp_plan's mg)
#define TF_QFN_XP(CB, LO, HI)                                                                                       \
    extern "C" __global__ void __launch_bounds__(256) tf_qfn_xp_gateup_##CB##_##LO##_##HI(                          \
        const __nv_bfloat16* x, int x_stride, int slots, const int* items, const int* nitems, const int* lists,     \
        int maxm, const int* cnt, const int64_t* TPg, const int64_t* TPu, const int* K2g, const int* K2u,           \
        const half* suh_g, const half* suh_u, const half* svh_g, const half* svh_u, const half* suh_d, half* xd,    \
        int K, int N, float limit, int act_mode) {                                                                  \
        tf_qfn_xp::gateup_body<CB, LO, HI>(x, x_stride, slots, items, nitems, lists, maxm, cnt, TPg, TPu, K2g, K2u, \
                                           suh_g, suh_u, svh_g, svh_u, suh_d, xd, K, N, limit, act_mode);           \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(512) tf_qfn_xp_down_##CB##_##LO##_##HI(                            \
        const half* xd, const int* items, const int* nitems, const int* lists, int maxm, const int* cnt,            \
        const int64_t* TPd, const int* K2d, const half* svh_d, uint16_t* out, int K, int D) {                       \
        tf_qfn_xp::down_body<CB, LO, HI>(xd, items, nitems, lists, maxm, cnt, TPd, K2d, svh_d, out, K, D);          \
    }
#define TF_QFN_XP_RANGES(CB) TF_QFN_XP(CB, 8, 8) TF_QFN_XP(CB, 2, 10) TF_QFN_XP(CB, 2, 16)
TF_QFN_XP_RANGES(0)
TF_QFN_XP_RANGES(1)
TF_QFN_XP_RANGES(2)
#undef TF_QFN_XP_RANGES
#undef TF_QFN_XP

extern "C" __global__ void __launch_bounds__(1024) tf_qfn_xp_plan(const int* cnt, int E, int* items, int* nitems,
                                                                   int mg) {
    tf_qfn_xp::plan_kernel(cnt, E, items, nitems, mg);
}

// the looping down projection: tf_qfn_xp_downloop<mt>_<cb>_<lo>_<hi>, 512 threads, grid (items),
// 16 mt (APAD * 2 + 512) dynamic shared bytes
#define TF_QFN_XDL(MT, CB, LO, HI)                                                                                   \
    extern "C" __global__ void __launch_bounds__(512) tf_qfn_xp_downloop##MT##_##CB##_##LO##_##HI(                  \
        const half* xd, const int* items, const int* nitems, const int* lists, int maxm, const int* cnt,            \
        const int64_t* TPd, const int* K2d, const half* svh_d, uint16_t* out, int K, int D) {                       \
        tf_qfn_xp::down_loop_body<CB, LO, HI, MT>(xd, items, nitems, lists, maxm, cnt, TPd, K2d, svh_d, out, K, D);  \
    }
#define TF_QFN_XDL_RANGES(MT, CB) TF_QFN_XDL(MT, CB, 8, 8) TF_QFN_XDL(MT, CB, 2, 10) TF_QFN_XDL(MT, CB, 2, 16)
TF_QFN_XDL_RANGES(2, 0)
TF_QFN_XDL_RANGES(2, 1)
TF_QFN_XDL_RANGES(2, 2)
TF_QFN_XDL_RANGES(3, 0)
TF_QFN_XDL_RANGES(3, 1)
TF_QFN_XDL_RANGES(3, 2)
#undef TF_QFN_XDL_RANGES
#undef TF_QFN_XDL
