// The prompt path's indexer scores with the bits of the fork's kernels.py `_index_score_rows` (prompt chunks: more
// than 16 rows; 32 heads of 128 dims, packed FP4 keys with E8M0 scales, 64-key tiles), at the mma rate.
//
// What `_index_score_rows` computes for a row r and a key t, and what these kernels compute the same way:
//  - the key's dims as bf16: e2m1(code) * 2^(e - 127) in fp32, rounded to bf16 (exact for the codes and scales a
//    cache holds), made from the packed cache rows as each 64-key tile is staged;
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
constexpr int KLD = ID + 8;     // halfs a staged key row (padded: ldmatrix's 8 row addresses fall in distinct banks)

__device__ __forceinline__ float e2m1(uint32_t code) {
    const uint32_t m = code & 7u;
    uint32_t bits = m >= 2u ? ((((m >> 1) + 126u) << 23) | ((m & 1u) << 22)) : (m == 1u ? 0x3F000000u : 0u);
    bits |= (code & 8u) << 28;
    return __uint_as_float(bits);
}

// two dims (codes c0, c1 under scale s) as a bf16 pair, the lower dim in the low half
__device__ __forceinline__ uint32_t dq2(uint32_t c0, uint32_t c1, float s) {
    const __nv_bfloat16 lo = __float2bfloat16_rn(__fmul_rn(e2m1(c0), s));
    const __nv_bfloat16 hi = __float2bfloat16_rn(__fmul_rn(e2m1(c1), s));
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// topk_indices' key of a score at key t: the score's bits in total order (-0 as +0) above, 0xFFFFFFFF - t below
__device__ __forceinline__ long long score_key(float sc, int t) {
    const int bits = sc == 0.0f ? 0 : __float_as_int(sc);
    const int ordered = bits < 0 ? (bits ^ 0x7FFFFFFF) : bits;
    return (long long)((unsigned long long)(long long)ordered << 32) | (0xFFFFFFFFll - (long long)t);
}

// A tile's packed rows as a thread holds them for staging: BPT code bytes of one key (dims j .. j + BPT - 1 in the low
// nibbles, 64 + j .. in the high) and the key's 4 E8M0 scales. 32 RB threads cover the tile's 64 keys x 64 bytes.
template <int RB>
struct Staged {
    static constexpr int BPT = BN * (ID / 2) / (32 * RB);   // 8 (RB 16) or 16 (RB 8)
    uint32_t codes[BPT / 4];
    uint32_t scales;
};

template <int RB>
__device__ __forceinline__ void load_tile(Staged<RB>& s, const uint8_t* __restrict__ K, const uint8_t* __restrict__ KS,
                                          int n, int tb) {
    constexpr int BPT = Staged<RB>::BPT;
    const int key = threadIdx.x / (ID / 2 / BPT), part = threadIdx.x % (ID / 2 / BPT);
    const int tk = tb * BN + key;
    if (tk < n) {
        const uint8_t* src = K + (size_t)tk * (ID / 2) + part * BPT;
        if constexpr (BPT == 16) {
            const uint4 v = *reinterpret_cast<const uint4*>(src);
            s.codes[0] = v.x; s.codes[1] = v.y; s.codes[2] = v.z; s.codes[3] = v.w;
        } else {
            const uint2 v = *reinterpret_cast<const uint2*>(src);
            s.codes[0] = v.x; s.codes[1] = v.y;
        }
        s.scales = *reinterpret_cast<const uint32_t*>(KS + (size_t)tk * (ID / 32));
    } else {
#pragma unroll
        for (int i = 0; i < BPT / 4; ++i) s.codes[i] = 0u;   // keys past n: zero (their scores are never written)
        s.scales = 0u;
    }
}

template <int RB>
__device__ __forceinline__ void store_tile(const Staged<RB>& s, __nv_bfloat16* ks) {
    constexpr int BPT = Staged<RB>::BPT;
    const int key = threadIdx.x / (ID / 2 / BPT), part = threadIdx.x % (ID / 2 / BPT);
    const int j0 = part * BPT;                                       // low dims j0 .., high dims 64 + j0 ..
    const float sl = __uint_as_float(((s.scales >> (8 * (j0 / 32))) & 0xFFu) << 23);
    const float sh = __uint_as_float(((s.scales >> (8 * (2 + j0 / 32))) & 0xFFu) << 23);
    uint32_t lo[BPT / 2], hi[BPT / 2];
#pragma unroll
    for (int i = 0; i < BPT / 4; ++i) {
#pragma unroll
        for (int b = 0; b < 4; b += 2) {
            const uint32_t c0 = (s.codes[i] >> (8 * b)) & 0xFFu, c1 = (s.codes[i] >> (8 * (b + 1))) & 0xFFu;
            lo[i * 2 + b / 2] = dq2(c0 & 15u, c1 & 15u, sl);
            hi[i * 2 + b / 2] = dq2(c0 >> 4, c1 >> 4, sh);
        }
    }
    uint4* dl = reinterpret_cast<uint4*>(ks + key * KLD + j0);
    uint4* dh = reinterpret_cast<uint4*>(ks + key * KLD + ID / 2 + j0);
#pragma unroll
    for (int i = 0; i < BPT / 8; ++i) {
        dl[i] = make_uint4(lo[4 * i], lo[4 * i + 1], lo[4 * i + 2], lo[4 * i + 3]);
        dh[i] = make_uint4(hi[4 * i], hi[4 * i + 1], hi[4 * i + 2], hi[4 * i + 3]);
    }
}

// the half's n_j sum in the captured order for one key (lanes with the same t hold the same sum after the butterflies)
__device__ __forceinline__ float half_sum(float s_lo, float s_hi, float w_lo, float w_hi) {
    float n = __fmaf_rn(fmaxf(s_lo, 0.0f), w_lo, __fmul_rn(fmaxf(s_hi, 0.0f), w_hi));
    n = __fadd_rn(n, __shfl_xor_sync(0xffffffffu, n, 16));   // + n_{j+4}
    n = __fadd_rn(n, __shfl_xor_sync(0xffffffffu, n, 8));    // + the pair 2 heads on
    n = __fadd_rn(n, __shfl_xor_sync(0xffffffffu, n, 4));    // + the quad 1 head on
    return n;
}

// RB rows a program, 32 RB threads: warp w takes row w's 32 heads (two m16 halves sharing every B fragment), each
// staged key tile shared by the program's rows. Programs take `tiles` 64-key tiles from blockIdx.y * tiles.
template <bool KEYS, int RB>
__device__ __forceinline__ void score_rows(const __nv_bfloat16* __restrict__ Q, const uint8_t* __restrict__ K,
                                           const uint8_t* __restrict__ KS, const __nv_bfloat16* __restrict__ Wt,
                                           const long long* __restrict__ VIS, void* __restrict__ OUT, int n, int rows,
                                           long long* __restrict__ TM, int nt, int tiles, bool tmax) {
    __shared__ __align__(16) __nv_bfloat16 ks[2][BN * KLD];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * RB + warp;
    const bool rok = r < rows;
    // the warp's A fragments (its row's 32 heads by the 128 dims, as two 16-head halves) and the heads' weights
    uint32_t a[2][8][4];
    float w_lo[2], w_hi[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const __nv_bfloat16* q0 = Q + ((size_t)(rok ? r : 0) * IH + 16 * h + g) * ID + 2 * t;
        const __nv_bfloat16* q1 = q0 + 8 * ID;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            a[h][kk][0] = rok ? *reinterpret_cast<const uint32_t*>(q0 + kk * 16) : 0u;
            a[h][kk][1] = rok ? *reinterpret_cast<const uint32_t*>(q1 + kk * 16) : 0u;
            a[h][kk][2] = rok ? *reinterpret_cast<const uint32_t*>(q0 + kk * 16 + 8) : 0u;
            a[h][kk][3] = rok ? *reinterpret_cast<const uint32_t*>(q1 + kk * 16 + 8) : 0u;
        }
        w_lo[h] = rok ? __bfloat162float(Wt[(size_t)r * IH + 16 * h + g]) : 0.f;
        w_hi[h] = rok ? __bfloat162float(Wt[(size_t)r * IH + 16 * h + g + 8]) : 0.f;
    }
    const long long vis = rok ? VIS[r] : 0;
    // ldmatrix: lane l addresses row l % 8 of matrix l / 8 (dims +0, +8, +16, +24 of a 32-dim step)
    const int lm_row = lane & 7, lm_col = (lane >> 3) * 8;

    const int tb0 = blockIdx.y * tiles, tb1 = min(nt, tb0 + tiles);
    Staged<RB> st;
    if (tb0 < tb1) {
        load_tile<RB>(st, K, KS, n, tb0);
        store_tile<RB>(st, ks[0]);
    }
    __syncthreads();
    for (int tb = tb0; tb < tb1; ++tb) {
        const int buf = (tb - tb0) & 1;
        const bool next = tb + 1 < tb1;
        if (next) load_tile<RB>(st, K, KS, n, tb + 1);
        if (rok) {
            // lane (g, t) ends with keys 8 g + 2 t and + 1 of the tile
            float sc[2] = {0.f, 0.f};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                const __nv_bfloat16* kr = ks[buf] + (j * 8 + lm_row) * KLD + lm_col;
#pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) {
                    uint32_t b[4];
                    ldsm4(b, kr + k2 * 32);
                    mma16816(acc[0], a[0][2 * k2], b[0], b[1]);
                    mma16816(acc[1], a[1][2 * k2], b[0], b[1]);
                    mma16816(acc[0], a[0][2 * k2 + 1], b[2], b[3]);
                    mma16816(acc[1], a[1][2 * k2 + 1], b[2], b[3]);
                }
                // acc[h]: heads g (c0, c1) and g + 8 (c2, c3) of half h, keys 8 j + 2 t (c0, c2) and + 1 (c1, c3)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const float h0 = half_sum(acc[0][c], acc[0][c + 2], w_lo[0], w_hi[0]);
                    const float h1 = half_sum(acc[1][c], acc[1][c + 2], w_lo[1], w_hi[1]);
                    if (g == j) sc[c] = __fadd_rn(h0, h1);
                }
            }
            long long m = (long long)0x8000000000000000ull;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tk = tb * BN + 8 * g + 2 * t + c;
                const float s = (long long)tk < vis ? sc[c] : -__int_as_float(0x7f800000);
                if constexpr (KEYS) {
                    const long long key = score_key(s, tk);
                    if (tk < n) {
                        reinterpret_cast<long long*>(OUT)[(size_t)r * n + tk] = key;
                        m = key > m ? key : m;
                    }
                } else {
                    if (tk < n) reinterpret_cast<float*>(OUT)[(size_t)r * n + tk] = s;
                }
            }
            if constexpr (KEYS) {
                if (tmax) {
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        const long long x = __shfl_xor_sync(0xffffffffu, m, o);
                        m = x > m ? x : m;
                    }
                    if (lane == 0) TM[(size_t)r * nt + tb] = m;
                }
            }
        }
        if (next) store_tile<RB>(st, ks[buf ^ 1]);   // the buffer every warp finished before the last barrier
        __syncthreads();
    }
}

}  // namespace

// K [n, 64] packed codes (byte j: dim j low nibble, dim j + 64 high), KS [n, 4] E8M0 scales (dims 0-31, 32-63, 64-95,
// 96-127). grid (ceil(rows / RB), ceil(nt / tiles)), 32 RB threads: RB rows a program (8 or 16), `tiles` 64-key tiles
// a program. OUT [rows, n], TM [rows, nt].
#define TF_INDEX_ROWS(RB_)                                                                                         \
    extern "C" __global__ void __launch_bounds__(32 * RB_) tf_dsv41_index_score_rows##RB_(                         \
        const __nv_bfloat16* Q, const uint8_t* K, const uint8_t* KS, const __nv_bfloat16* Wt, const long long* VIS, \
        float* OUT, int n, int rows, int nt, int tiles) {                                                           \
        score_rows<false, RB_>(Q, K, KS, Wt, VIS, OUT, n, rows, nullptr, nt, tiles, false);                        \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(32 * RB_) tf_dsv41_index_keys_rows##RB_(                          \
        const __nv_bfloat16* Q, const uint8_t* K, const uint8_t* KS, const __nv_bfloat16* Wt, const long long* VIS, \
        long long* OUT, int n, int rows, long long* TM, int nt, int tiles, int has_tmax) {                          \
        score_rows<true, RB_>(Q, K, KS, Wt, VIS, OUT, n, rows, TM, nt, tiles, has_tmax != 0);                      \
    }
TF_INDEX_ROWS(8)
TF_INDEX_ROWS(16)
#undef TF_INDEX_ROWS
