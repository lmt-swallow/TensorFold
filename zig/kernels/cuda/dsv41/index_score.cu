// The prompt path's indexer scores with the bits of the fork's kernels.py `_index_score_rows` (prompt chunks: more
// than 16 rows; 32 heads of 128 dims, packed FP4 keys with E8M0 scales, 64-key tiles), at the mma rate.
//
// What `_index_score_rows` computes for a row r and a key t, and what these kernels compute the same way:
//  - the key's dims as bf16: e2m1(code) * 2^(e - 127) in fp32, rounded to bf16 (exact for the codes and scales a
//    cache holds); tf_dsv41_index_keys_bf16 writes them once a call (in segments of keys) instead of once a program;
//  - s[h] = q[r, h] . k[t] as one mma.m16n8k16 (bf16 in, fp32 accumulate) chain from zero over the 128 dims in steps
//    of 16, in order (the captured cubin's: its two 64-dim dots fused into one 8-step chain);
//  - the score = the sum over heads of max(s[h], 0) * w[r, h], added as the captured kernel adds them (recovered by
//    tf-cuda-test dsv41-index-probe and checked by tools/zig/dsv41/index_tree.py against every probe): within each
//    16-head half, n_j = fma(max(s[j], 0), w[j], max(s[j + 8], 0) * w[j + 8]) for j < 8, then
//    ((n_0 + n_4) + (n_2 + n_6)) + ((n_1 + n_5) + (n_3 + n_7)), and the score = half(0..15) + half(16..31);
//  - -inf where t >= vis[r]; with KEYS, topk_indices' int64 key of the score (its bits in total order above, ~t
//    below), and with TMAX each row's largest key of each 64-key tile.
// Built with --fmad=false: every product, sum and fma above is spelled out.

#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int IH = 32;          // heads
constexpr int ID = 128;         // dims
constexpr int BN = 64;          // keys a tile
constexpr int KLD = ID + 8;     // halfs a staged key row (padded: B fragment loads fall in distinct banks)

__device__ __forceinline__ float e2m1(uint32_t code) {
    const uint32_t m = code & 7u;
    uint32_t bits = m >= 2u ? ((((m >> 1) + 126u) << 23) | ((m & 1u) << 22)) : (m == 1u ? 0x3F000000u : 0u);
    bits |= (code & 8u) << 28;
    return __uint_as_float(bits);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

// topk_indices' key of a score at key t: the score's bits in total order (-0 as +0) above, 0xFFFFFFFF - t below
__device__ __forceinline__ long long score_key(float sc, int t) {
    const int bits = sc == 0.0f ? 0 : __float_as_int(sc);
    const int ordered = bits < 0 ? (bits ^ 0x7FFFFFFF) : bits;
    return (long long)((unsigned long long)(long long)ordered << 32) | (0xFFFFFFFFll - (long long)t);
}

// One 64-key tile (staged in `ks`) against the warp's 16 heads of its row: per key the half's sum in the captured
// order (lanes with the same t hold the same sums after the butterflies), for keys 8 j + 2 t and + 1.
__device__ __forceinline__ void tile_halves(const uint32_t (&a)[8][4], const __nv_bfloat16* ks, float w_lo, float w_hi,
                                            int g, int t, float (&half)[8][2]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const __nv_bfloat16* kr = ks + (j * 8 + g) * KLD + 2 * t;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr + kk * 16);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + kk * 16 + 8);
            mma16816(acc, a[kk], b0, b1);
        }
        // acc: heads g (c0, c1) and g + 8 (c2, c3) of the half, keys 8 j + 2 t (c0, c2) and + 1 (c1, c3)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const float lo = fmaxf(acc[c], 0.0f), hi = fmaxf(acc[c + 2], 0.0f);
            float n = __fmaf_rn(lo, w_lo, __fmul_rn(hi, w_hi));
            n = __fadd_rn(n, __shfl_xor_sync(0xffffffffu, n, 16));   // + n_{j+4}
            n = __fadd_rn(n, __shfl_xor_sync(0xffffffffu, n, 8));    // + the pair 2 heads on
            n = __fadd_rn(n, __shfl_xor_sync(0xffffffffu, n, 4));    // + the quad 1 head on
            half[j][c] = n;
        }
    }
}

// RB rows a program, 64 RB threads: warp w takes row w / 2's heads 16 (w % 2) .. + 15.
template <bool KEYS, int RB>
__device__ __forceinline__ void score_rows(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KB,
                                           const __nv_bfloat16* __restrict__ Wt, const long long* __restrict__ VIS,
                                           void* __restrict__ OUT, int n, int rows, long long* __restrict__ TM, int nt,
                                           int tiles, int tile0, int tile1, bool tmax) {
    __shared__ __align__(16) __nv_bfloat16 ks[2][BN * KLD];
    __shared__ float hs[RB][2][BN];
    __shared__ long long tmx[RB][2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rl = warp >> 1, hb = (warp & 1) * 16;
    const int r = blockIdx.x * RB + rl;
    const bool rok = r < rows;
    // the warp's A fragments (its row's 16 heads by the 128 dims) and the heads' weights, once
    uint32_t a[8][4];
    {
        const __nv_bfloat16* q0 = Q + ((size_t)(rok ? r : 0) * IH + hb + g) * ID + 2 * t;
        const __nv_bfloat16* q1 = q0 + 8 * ID;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            a[kk][0] = rok ? *reinterpret_cast<const uint32_t*>(q0 + kk * 16) : 0u;
            a[kk][1] = rok ? *reinterpret_cast<const uint32_t*>(q1 + kk * 16) : 0u;
            a[kk][2] = rok ? *reinterpret_cast<const uint32_t*>(q0 + kk * 16 + 8) : 0u;
            a[kk][3] = rok ? *reinterpret_cast<const uint32_t*>(q1 + kk * 16 + 8) : 0u;
        }
    }
    const float w_lo = rok ? __bfloat162float(Wt[(size_t)r * IH + hb + g]) : 0.f;
    const float w_hi = rok ? __bfloat162float(Wt[(size_t)r * IH + hb + g + 8]) : 0.f;
    // the epilogue's thread: row eo_r of the program, key eo_t of the tile
    const int eo_r = threadIdx.x >> 6, eo_t = threadIdx.x & 63;
    const int er = blockIdx.x * RB + eo_r;
    const long long vis = er < rows ? VIS[er] : 0;

    // this program's tiles, within the segment [tile0, tile1) whose keys KB holds (KB row 0: key tile0 * 64)
    const int tb0 = tile0 + blockIdx.y * tiles, tb1 = min(tile1, tb0 + tiles);
    const int key0 = tile0 * BN;
    // staging: 64 keys x 256 bytes a tile, 16 bytes a copy, 1024 / (64 RB) copies a thread
    auto stage = [&](int buf, int tb) {
#pragma unroll
        for (int c = 0; c < 16 / RB; ++c) {
            const int idx = threadIdx.x + c * 64 * RB;      // 1024 chunks: key idx / 16, 16-byte part idx % 16
            const int key = idx >> 4, part = idx & 15;
            const int tk = tb * BN + key;
            cp_async16(&ks[buf][key * KLD + part * 8], KB + (size_t)(tk < n ? tk - key0 : 0) * ID + part * 8, tk < n);
        }
        cp_async_commit();
    };
    if (tb0 < tb1) stage(0, tb0);
    for (int tb = tb0; tb < tb1; ++tb) {
        const int buf = (tb - tb0) & 1;
        if (tb + 1 < tb1) stage(buf ^ 1, tb + 1);
        else cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        float half[8][2];
        tile_halves(a, ks[buf], w_lo, w_hi, g, t, half);
        if (g == 0) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                hs[rl][warp & 1][j * 8 + 2 * t] = half[j][0];
                hs[rl][warp & 1][j * 8 + 2 * t + 1] = half[j][1];
            }
        }
        __syncthreads();                                      // hs written; every warp is past ks[buf]
        const int tk = tb * BN + eo_t;
        float sc = __fadd_rn(hs[eo_r][0][eo_t], hs[eo_r][1][eo_t]);
        if (!((long long)tk < vis)) sc = -__int_as_float(0x7f800000);
        const bool ok = er < rows && tk < n;
        if constexpr (KEYS) {
            const long long key = score_key(sc, tk);
            if (ok) reinterpret_cast<long long*>(OUT)[(size_t)er * n + tk] = key;
            if (tmax) {
                long long m = tk < n ? key : (long long)0x8000000000000000ull;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    const long long x = __shfl_xor_sync(0xffffffffu, m, o);
                    m = x > m ? x : m;
                }
                if (lane == 0) tmx[eo_r][(threadIdx.x >> 5) & 1] = m;
                __syncthreads();
                if (eo_t == 0 && er < rows) {
                    const long long m0 = tmx[eo_r][0], m1 = tmx[eo_r][1];
                    TM[(size_t)er * nt + tb] = m0 > m1 ? m0 : m1;
                }
            }
        } else {
            if (ok) reinterpret_cast<float*>(OUT)[(size_t)er * n + tk] = sc;
        }
        __syncthreads();                                      // hs / tmx read before the next tile writes them
    }
}

}  // namespace

// KB [n, 128] bf16 from n packed cache rows: codes [n, 64] (byte j: dim j low nibble, dim j + 64 high), E8M0 scales
// [n, 4] (dims 0-31, 32-63, 64-95, 96-127). One thread a byte (two dims).
extern "C" __global__ void tf_dsv41_index_keys_bf16(const uint8_t* __restrict__ K, const uint8_t* __restrict__ KS,
                                                     __nv_bfloat16* __restrict__ KB, int n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * (ID / 2)) return;
    const size_t key = i / (ID / 2);
    const int j = (int)(i % (ID / 2));
    const uint32_t cb = K[i];
    const float sl = __uint_as_float((uint32_t)KS[key * (ID / 32) + j / 32] << 23);
    const float sh = __uint_as_float((uint32_t)KS[key * (ID / 32) + ID / 64 + j / 32] << 23);
    KB[key * ID + j] = __float2bfloat16_rn(__fmul_rn(e2m1(cb & 15u), sl));
    KB[key * ID + ID / 2 + j] = __float2bfloat16_rn(__fmul_rn(e2m1(cb >> 4), sh));
}

// One segment of the keys: tiles [tile0, tile1) of the row block's n keys, KB [their keys, 128] bf16 (from
// tf_dsv41_index_keys_bf16 at the segment's first key). grid (ceil(rows / RB), ceil((tile1 - tile0) / tiles)), 64 RB
// threads: RB rows a program (4 or 8), `tiles` 64-key tiles a program. OUT [rows, n], TM [rows, nt]: absolute keys
// and tiles.
#define TF_INDEX_ROWS(RB_)                                                                                         \
    extern "C" __global__ void __launch_bounds__(64 * RB_) tf_dsv41_index_score_rows##RB_(                         \
        const __nv_bfloat16* Q, const __nv_bfloat16* KB, const __nv_bfloat16* Wt, const long long* VIS, float* OUT, \
        int n, int rows, int nt, int tiles, int tile0, int tile1) {                                                 \
        score_rows<false, RB_>(Q, KB, Wt, VIS, OUT, n, rows, nullptr, nt, tiles, tile0, tile1, false);             \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(64 * RB_) tf_dsv41_index_keys_rows##RB_(                          \
        const __nv_bfloat16* Q, const __nv_bfloat16* KB, const __nv_bfloat16* Wt, const long long* VIS,             \
        long long* OUT, int n, int rows, long long* TM, int nt, int tiles, int tile0, int tile1, int has_tmax) {    \
        score_rows<true, RB_>(Q, KB, Wt, VIS, OUT, n, rows, TM, nt, tiles, tile0, tile1, has_tmax != 0);           \
    }
TF_INDEX_ROWS(4)
TF_INDEX_ROWS(8)
#undef TF_INDEX_ROWS
