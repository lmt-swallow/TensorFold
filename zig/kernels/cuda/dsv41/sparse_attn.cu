// The prompt path's sparse attention with the bits of the fork's kernels.py `_sparse_attn_part` (prompt chunks: one
// split, FINAL; 32 heads of 512 dims, 128 window keys then the row's picks among the packed FP4 compressed keys, 32-key
// blocks), each key block staged once for both 16-head blocks of a row instead of once per head block.
//
// The arithmetic of a (row, head) is the captured cubin's (its SASS; checked on every call by TF_DS_ATTN_CHECK):
//  - a block's keys as bf16: the window rows as they are (0 outside the window), a compressed key's dims
//    e2m1(code) * e4m3(scale) per 16 dims rounded to bf16 (0 for a pick of -1);
//  - s = (q . k) * scale: one mma.m16n8k16 chain from zero over the 512 dims in steps of 16, in order, then one fp32
//    product; -inf for a key outside the window or a pick of -1;
//  - m_new = max(m, max_k s); exp(x) as ex2.approx.f32(x * log2(e)) (the product rounded, the non-ftz ex2);
//    p = exp(s - m_new), alpha = exp(m - m_new);
//  - the block's sum of p over its 32 keys: per warp (8 keys: lane t's keys 2t, 2t + 1 added, then lanes t ^ 2, t ^ 1),
//    then the four warps (SUM_ORDER 0, (w0 + w2) + (w1 + w3), is the captured kernel's; 1 and 2 differ);
//  - l = fma(alpha, l, sum); acc = acc * alpha, then the block's p (as bf16) times its keys accumulated into acc as one
//    mma chain over keys 0-15 then 16-31;
//  - after the last block l = l + exp(sink - m), out = bf16(div.full.f32(acc, l)).
// Built with --fmad=false: every product, sum and fma above is spelled out.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int H = 32, HD = 512, HB = 16, HALF = HD / 2, WIN = 128, BN = 32, WB = WIN / BN;
constexpr int KLD = HD + 8;      // halfs a staged key row (padded)
constexpr int QLD = HD + 8;      // halfs a staged query row

__device__ __forceinline__ float exp_tl(float x) {
    // Triton's tl.exp on fp32: ex2.approx.f32 of the rounded product with log2(e)
    const float t = __fmul_rn(x, 1.4426950216293334961f);
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(t));
    return r;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float e2m1(uint32_t code) {
    const uint32_t m = code & 7u;
    uint32_t bits = m >= 2u ? ((((m >> 1) + 126u) << 23) | ((m & 1u) << 22)) : (m == 1u ? 0x3F000000u : 0u);
    bits |= (code & 8u) << 28;
    return __uint_as_float(bits);
}

__device__ __forceinline__ float e4m3(uint8_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(__half(h));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& b0, uint32_t& b1, const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(s));
}

// The four warps' sums of a head block in the captured order (SUM_ORDER 0: (w0 + w2) + (w1 + w3); 1: (w0 + w1) +
// (w2 + w3); 2: ((w0 + w1) + w2) + w3)
template <int SUM_ORDER>
__device__ __forceinline__ float sum4(float w0, float w1, float w2, float w3) {
    if constexpr (SUM_ORDER == 0) return __fadd_rn(__fadd_rn(w0, w2), __fadd_rn(w1, w3));
    else if constexpr (SUM_ORDER == 1) return __fadd_rn(__fadd_rn(w0, w1), __fadd_rn(w2, w3));
    else return __fadd_rn(__fadd_rn(__fadd_rn(w0, w1), w2), w3);
}

// One query row, both 16-head blocks: 8 warps; warp w takes head block w / 4 and, of each 32-key block, keys
// 8 (w % 4) .. + 7 for the scores and dims 64 (w % 4) .. + 63 of each 256-dim half for the outputs.
template <int SUM_ORDER>
__device__ __forceinline__ void attn_row(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ WSRC,
                                         const long long* __restrict__ WLO, const uint8_t* __restrict__ COMP,
                                         const uint8_t* __restrict__ CSC, const long long* __restrict__ IDX,
                                         const long long* __restrict__ POS, const float* __restrict__ SINK,
                                         __nv_bfloat16* __restrict__ OUT, float scale, int n_idx, int nblk,
                                         bool has_comp) {
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);                 // [32 heads][QLD]
    __nv_bfloat16* ks = qs + H * QLD;                                            // [32 keys][KLD]
    __nv_bfloat16* ps = ks + BN * KLD;                                           // [2][16 heads][BN + 8]
    float* red = reinterpret_cast<float*>(ps + 2 * HB * (BN + 8));              // [2][4 warps][16 heads]
    int* okv = reinterpret_cast<int*>(red + 2 * 4 * HB);                         // [BN]
    long long* rows = reinterpret_cast<long long*>(okv + BN);                    // [BN]

    const int r = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int hb = warp >> 2, nw = warp & 3;
    const long long p = POS[r];
    const long long wlo = WLO[0];

    // the row's queries into shared memory (16 bytes a copy)
    for (int i = threadIdx.x; i < H * HD / 8; i += 256) {
        const int h = i / (HD / 8), c = i % (HD / 8);
        *reinterpret_cast<uint4*>(qs + h * QLD + c * 8) =
            *reinterpret_cast<const uint4*>(Q + ((size_t)r * H + h) * HD + c * 8);
    }

    float m_i[2] = {-1e30f, -1e30f}, l_i[2] = {0.f, 0.f};     // heads g and g + 8 of the warp's block
    float acc[16][4];                                          // dims: 8 n8 tiles of each half
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][c] = 0.f;

    // A thread stages key tid / 8, dims 32 (tid % 8) .. + 31 of each half, a block ahead: its raw bytes (bf16 window
    // rows or packed codes and scales) are loaded into registers while the block before is computed.
    const int skey = threadIdx.x >> 3, spart = threadIdx.x & 7;
    struct Raw {
        uint4 v[8];         // window: the 64 bf16 values; packed: v[0], v[1] the 32 code bytes
        uint32_t sl, sh;    // packed: the two lo-half and the two hi-half scale bytes
        bool ok;
    };
    auto load = [&](int b, Raw& raw) {
        long long row;
        bool packed = false;
        if (b < WB) {
            const long long wp = p - (WIN - 1) + b * BN + skey;
            const long long slot = wp - wlo;
            raw.ok = wp >= 0 && slot >= 0;
            row = raw.ok ? slot : 0;
        } else {
            const int tt = (b - WB) * BN + skey;
            const long long ii = tt < n_idx ? IDX[(size_t)r * n_idx + tt] : -1;
            raw.ok = ii >= 0;
            row = raw.ok ? ii : 0;
            packed = has_comp;
        }
        if (!packed) {
            const __nv_bfloat16* src = (b < WB ? WSRC : reinterpret_cast<const __nv_bfloat16*>(COMP)) + (size_t)row * HD;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int d = (c < 4 ? spart * 32 + c * 8 : HALF + spart * 32 + (c - 4) * 8);
                raw.v[c] = raw.ok ? *reinterpret_cast<const uint4*>(src + d) : make_uint4(0, 0, 0, 0);
            }
        } else {
            const uint8_t* cb = COMP + (size_t)row * HALF + spart * 32;
            const uint8_t* sc = CSC + (size_t)row * (HD / 16) + spart * 2;
            raw.v[0] = raw.ok ? *reinterpret_cast<const uint4*>(cb) : make_uint4(0, 0, 0, 0);
            raw.v[1] = raw.ok ? *reinterpret_cast<const uint4*>(cb + 16) : make_uint4(0, 0, 0, 0);
            raw.sl = raw.ok ? *reinterpret_cast<const uint16_t*>(sc) : 0u;
            raw.sh = raw.ok ? *reinterpret_cast<const uint16_t*>(sc + HALF / 16) : 0u;
        }
    };
    auto store = [&](int b, const Raw& raw) {
        __nv_bfloat16* kr = ks + skey * KLD;
        if (spart == 0) okv[skey] = raw.ok;
        if (b < WB || !has_comp) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int d = (c < 4 ? spart * 32 + c * 8 : HALF + spart * 32 + (c - 4) * 8);
                *reinterpret_cast<uint4*>(kr + d) = raw.v[c];
            }
        } else {
            // byte j: dim j (low nibble) and dim j + 256 (high); a scale for each 16 dims
            const uint32_t w[8] = {raw.v[0].x, raw.v[0].y, raw.v[0].z, raw.v[0].w, raw.v[1].x, raw.v[1].y, raw.v[1].z, raw.v[1].w};
#pragma unroll
            for (int gq = 0; gq < 2; ++gq) {
                const float sl = e4m3((uint8_t)(raw.sl >> (8 * gq)));
                const float sh = e4m3((uint8_t)(raw.sh >> (8 * gq)));
                const int j0 = spart * 32 + gq * 16;
#pragma unroll
                for (int q = 0; q < 2; ++q) {          // 8 dims a 16-byte store
                    uint32_t lo[4], hi[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int byte = gq * 16 + q * 8 + 2 * e;
                        const uint32_t c0 = (w[byte >> 2] >> (8 * (byte & 3))) & 0xffu;
                        const uint32_t c1 = (w[(byte + 1) >> 2] >> (8 * ((byte + 1) & 3))) & 0xffu;
                        __nv_bfloat162 vl, vh;
                        vl.x = __float2bfloat16_rn(__fmul_rn(e2m1(c0 & 15u), sl));
                        vl.y = __float2bfloat16_rn(__fmul_rn(e2m1(c1 & 15u), sl));
                        vh.x = __float2bfloat16_rn(__fmul_rn(e2m1(c0 >> 4), sh));
                        vh.y = __float2bfloat16_rn(__fmul_rn(e2m1(c1 >> 4), sh));
                        lo[e] = *reinterpret_cast<uint32_t*>(&vl);
                        hi[e] = *reinterpret_cast<uint32_t*>(&vh);
                    }
                    *reinterpret_cast<uint4*>(kr + j0 + q * 8) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
                    *reinterpret_cast<uint4*>(kr + HALF + j0 + q * 8) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
                }
            }
        }
    };

    Raw raw;
    load(0, raw);
    for (int b = 0; b < nblk; ++b) {
        __syncthreads();                                       // the previous block's keys and p are read
        store(b, raw);
        __syncthreads();
        if (b + 1 < nblk) load(b + 1, raw);                    // in flight while this block is computed

        // scores: heads 16 hb .. (rows g, g + 8) x keys 8 nw + 2 t, + 1
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        {
            const __nv_bfloat16* qa = qs + (hb * HB + (lane & 15)) * QLD + (lane >> 4) * 8;
            const __nv_bfloat16* kb = ks + (nw * 8 + g) * KLD + 2 * t;
#pragma unroll 8
            for (int kk = 0; kk < HD / 16; ++kk) {
                uint32_t a[4];
                ldmatrix_x4(a, qa + kk * 16);
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kb + kk * 16);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kb + kk * 16 + 8);
                mma16816(s, a, b0, b1);
            }
        }
        const bool ok0 = okv[nw * 8 + 2 * t], ok1 = okv[nw * 8 + 2 * t + 1];
        float sv[4];
        sv[0] = ok0 ? __fmul_rn(s[0], scale) : -__int_as_float(0x7f800000);
        sv[1] = ok1 ? __fmul_rn(s[1], scale) : -__int_as_float(0x7f800000);
        sv[2] = ok0 ? __fmul_rn(s[2], scale) : -__int_as_float(0x7f800000);
        sv[3] = ok1 ? __fmul_rn(s[3], scale) : -__int_as_float(0x7f800000);
        // the block's max a head: the warp's 8 keys, then the head block's 4 warps
        float mx[2] = {fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3])};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
        }
        float* rmax = red + (hb * 4) * HB;
        if (t == 0) {
            rmax[nw * HB + g] = mx[0];
            rmax[nw * HB + g + 8] = mx[1];
        }
        __syncthreads();
        float m_new[2], alpha[2], pr[4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int h = g + 8 * i;
            const float bm = fmaxf(fmaxf(rmax[0 * HB + h], rmax[2 * HB + h]), fmaxf(rmax[1 * HB + h], rmax[3 * HB + h]));
            m_new[i] = fmaxf(bm, m_i[i]);
            alpha[i] = exp_tl(__fsub_rn(m_i[i], m_new[i]));
        }
        pr[0] = exp_tl(__fsub_rn(sv[0], m_new[0]));
        pr[1] = exp_tl(__fsub_rn(sv[1], m_new[0]));
        pr[2] = exp_tl(__fsub_rn(sv[2], m_new[1]));
        pr[3] = exp_tl(__fsub_rn(sv[3], m_new[1]));
        float sm[2] = {__fadd_rn(pr[0], pr[1]), __fadd_rn(pr[2], pr[3])};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sm[i] = __fadd_rn(sm[i], __shfl_xor_sync(0xffffffffu, sm[i], 2));
            sm[i] = __fadd_rn(sm[i], __shfl_xor_sync(0xffffffffu, sm[i], 1));
        }
        __syncthreads();                                       // every warp has read rmax
        if (t == 0) {
            rmax[nw * HB + g] = sm[0];
            rmax[nw * HB + g + 8] = sm[1];
        }
        // p as bf16 for the outputs' mma
        __nv_bfloat16* pb = ps + hb * HB * (BN + 8);
        pb[g * (BN + 8) + nw * 8 + 2 * t] = __float2bfloat16_rn(pr[0]);
        pb[g * (BN + 8) + nw * 8 + 2 * t + 1] = __float2bfloat16_rn(pr[1]);
        pb[(g + 8) * (BN + 8) + nw * 8 + 2 * t] = __float2bfloat16_rn(pr[2]);
        pb[(g + 8) * (BN + 8) + nw * 8 + 2 * t + 1] = __float2bfloat16_rn(pr[3]);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int h = g + 8 * i;
            const float bs = sum4<SUM_ORDER>(rmax[0 * HB + h], rmax[1 * HB + h], rmax[2 * HB + h], rmax[3 * HB + h]);
            l_i[i] = __fmaf_rn(alpha[i], l_i[i], bs);
            m_i[i] = m_new[i];
        }
        // acc = acc * alpha + p . k (keys 0-15, then 16-31)
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j][0] = __fmul_rn(acc[j][0], alpha[0]);
            acc[j][1] = __fmul_rn(acc[j][1], alpha[0]);
            acc[j][2] = __fmul_rn(acc[j][2], alpha[1]);
            acc[j][3] = __fmul_rn(acc[j][3], alpha[1]);
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t a[4];
            ldmatrix_x4(a, pb + (lane & 15) * (BN + 8) + kk * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                // n8 tile j: dims of half j / 8, 64 nw + 8 (j % 8) ..
                const int d0 = (j >> 3) * HALF + nw * 64 + (j & 7) * 8;
                uint32_t b0, b1;
                ldmatrix_x2_trans(b0, b1, ks + (kk * 16 + (lane & 15)) * KLD + d0);
                mma16816(acc[j], a, b0, b1);
            }
        }
    }
    // the sink, the division, bf16
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int h = hb * HB + g + 8 * i;
        const float l = __fadd_rn(l_i[i], exp_tl(__fsub_rn(SINK[h], m_i[i])));
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int d = (j >> 3) * HALF + nw * 64 + (j & 7) * 8 + 2 * t;
            __nv_bfloat162 o;
            o.x = __float2bfloat16_rn(div_full(acc[j][2 * i], l));
            o.y = __float2bfloat16_rn(div_full(acc[j][2 * i + 1], l));
            *reinterpret_cast<__nv_bfloat162*>(OUT + ((size_t)r * H + h) * HD + d) = o;
        }
    }
}

// ---- v3: warp-specialized (TF_DS_SPARSE_ATTN=ws) ----------------------------------------------------------------
// attn_row's arithmetic, order 0, unchanged per (row, head); 12 warps: warps 0-7 compute as attn_row's 8, warps 8-11
// stage the next key block into the other of two key buffers meanwhile. Shared memory is XOR-swizzled instead of
// padded so the queries and two key blocks fit (99 KB): a 1 KB row's 32-bit word w sits at w ^ ((row & 7) << 2).
// Named barriers: 1 the compute warps' own; 2 + b "key buffer b full" (producers arrive, compute warps wait); 4 + b
// "key buffer b free" (compute warps arrive, producers wait).
constexpr int WS_THREADS = 384;
constexpr int WS_QBYTES = H * HD * 2, WS_KBYTES = BN * HD * 2, WS_PBYTES = 2 * HB * BN * 2;

__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// the byte offset of 32-bit word w of 1 KB row `row`
__device__ __forceinline__ int swz(int row, int w) { return row * 1024 + ((w ^ ((row & 7) << 2)) << 2); }
// p tile [16 heads][32 keys] bf16: 64-byte rows of four 16-byte chunks, chunk c at c ^ ((h >> 1) & 3)
__device__ __forceinline__ int pswz(int h, int c) { return h * 64 + ((c ^ ((h >> 1) & 3)) << 4); }

template <int SUM_ORDER>
__device__ __forceinline__ void attn_row_ws(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ WSRC,
                                            const long long* __restrict__ WLO, const uint8_t* __restrict__ COMP,
                                            const uint8_t* __restrict__ CSC, const long long* __restrict__ IDX,
                                            const long long* __restrict__ POS, const float* __restrict__ SINK,
                                            __nv_bfloat16* __restrict__ OUT, float scale, int n_idx, int nblk,
                                            bool has_comp) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* qs = smem;                                      // [32 heads] 1 KB rows, swizzled
    unsigned char* ksb = qs + WS_QBYTES;                           // [2][32 keys] 1 KB rows, swizzled
    unsigned char* psb = ksb + 2 * WS_KBYTES;                      // [2 head blocks][16 heads] 64 B rows
    float* red = reinterpret_cast<float*>(psb + WS_PBYTES);       // [2][4 warps][16 heads]
    int* okv = reinterpret_cast<int*>(red + 2 * 4 * HB);           // [2 buffers][32 keys]

    const int r = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long p = POS[r];
    const long long wlo = WLO[0];

    for (int i = threadIdx.x; i < H * HD / 8; i += WS_THREADS) {
        const int h = i / (HD / 8), c = i % (HD / 8);              // 16-byte chunk c: words 4 c .. + 3
        *reinterpret_cast<uint4*>(qs + swz(h, 4 * c)) = *reinterpret_cast<const uint4*>(Q + ((size_t)r * H + h) * HD + c * 8);
    }
    __syncthreads();

    if (warp >= 8) {
        // producers: 128 threads; key ptid / 4, dims 64 (ptid % 4) .. + 63 of each half
        const int ptid = threadIdx.x - 256, key = ptid >> 2, part = ptid & 3;
        for (int b = 0; b < nblk; ++b) {
            const int buf = b & 1;
            if (b >= 2) bar_sync(4 + buf, WS_THREADS);
            unsigned char* kb = ksb + buf * WS_KBYTES;
            bool ok;
            long long row;
            bool packed = false;
            if (b < WB) {
                const long long wp = p - (WIN - 1) + b * BN + key;
                const long long slot = wp - wlo;
                ok = wp >= 0 && slot >= 0;
                row = ok ? slot : 0;
            } else {
                const int tt = (b - WB) * BN + key;
                const long long ii = tt < n_idx ? IDX[(size_t)r * n_idx + tt] : -1;
                ok = ii >= 0;
                row = ok ? ii : 0;
                packed = has_comp;
            }
            if (part == 0) okv[buf * BN + key] = ok;
            if (!packed) {
                const __nv_bfloat16* src = (b < WB ? WSRC : reinterpret_cast<const __nv_bfloat16*>(COMP)) + (size_t)row * HD;
#pragma unroll
                for (int c = 0; c < 16; ++c) {             // 8 chunks of each half
                    const int d = (c < 8 ? part * 64 + c * 8 : HALF + part * 64 + (c - 8) * 8);
                    const uint4 v = ok ? *reinterpret_cast<const uint4*>(src + d) : make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(kb + swz(key, d >> 1)) = v;
                }
            } else {
                const uint8_t* cb = COMP + (size_t)row * HALF + part * 64;
                const uint8_t* sc = CSC + (size_t)row * (HD / 16) + part * 4;
                uint4 cw[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) cw[q] = ok ? *reinterpret_cast<const uint4*>(cb + 16 * q) : make_uint4(0, 0, 0, 0);
                const uint32_t slw = ok ? *reinterpret_cast<const uint32_t*>(sc) : 0u;
                const uint32_t shw = ok ? *reinterpret_cast<const uint32_t*>(sc + HALF / 16) : 0u;
                const uint32_t w[16] = {cw[0].x, cw[0].y, cw[0].z, cw[0].w, cw[1].x, cw[1].y, cw[1].z, cw[1].w,
                                        cw[2].x, cw[2].y, cw[2].z, cw[2].w, cw[3].x, cw[3].y, cw[3].z, cw[3].w};
#pragma unroll
                for (int gq = 0; gq < 4; ++gq) {               // four 16-dim groups of this part
                    const float sl = e4m3((uint8_t)(slw >> (8 * gq)));
                    const float sh = e4m3((uint8_t)(shw >> (8 * gq)));
                    const int j0 = part * 64 + gq * 16;
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        uint32_t lo[4], hi[4];
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int byte = gq * 16 + q * 8 + 2 * e;
                            const uint32_t c0 = (w[byte >> 2] >> (8 * (byte & 3))) & 0xffu;
                            const uint32_t c1 = (w[(byte + 1) >> 2] >> (8 * ((byte + 1) & 3))) & 0xffu;
                            __nv_bfloat162 vl, vh;
                            vl.x = __float2bfloat16_rn(__fmul_rn(e2m1(c0 & 15u), sl));
                            vl.y = __float2bfloat16_rn(__fmul_rn(e2m1(c1 & 15u), sl));
                            vh.x = __float2bfloat16_rn(__fmul_rn(e2m1(c0 >> 4), sh));
                            vh.y = __float2bfloat16_rn(__fmul_rn(e2m1(c1 >> 4), sh));
                            lo[e] = *reinterpret_cast<uint32_t*>(&vl);
                            hi[e] = *reinterpret_cast<uint32_t*>(&vh);
                        }
                        *reinterpret_cast<uint4*>(kb + swz(key, (j0 + q * 8) >> 1)) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
                        *reinterpret_cast<uint4*>(kb + swz(key, (HALF + j0 + q * 8) >> 1)) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
                    }
                }
            }
            bar_arrive(2 + buf, WS_THREADS);
        }
        return;
    }

    // compute warps: attn_row's
    const int g = lane >> 2, t = lane & 3;
    const int hb = warp >> 2, nw = warp & 3;
    float m_i[2] = {-1e30f, -1e30f}, l_i[2] = {0.f, 0.f};
    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][c] = 0.f;
    unsigned char* pb = psb + hb * (WS_PBYTES / 2);
    float* rmax = red + (hb * 4) * HB;
    const int qrow = hb * HB + (lane & 15);
    for (int b = 0; b < nblk; ++b) {
        const int buf = b & 1;
        const unsigned char* kb = ksb + buf * WS_KBYTES;
        bar_sync(2 + buf, WS_THREADS);
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        const int kr = nw * 8 + g;
#pragma unroll 8
        for (int kk = 0; kk < HD / 16; ++kk) {
            uint32_t a[4];
            ldmatrix_x4(a, qs + swz(qrow, kk * 8 + (lane >> 4) * 4));
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kb + swz(kr, kk * 8 + t));
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kb + swz(kr, kk * 8 + 4 + t));
            mma16816(s, a, b0, b1);
        }
        const bool ok0 = okv[buf * BN + nw * 8 + 2 * t], ok1 = okv[buf * BN + nw * 8 + 2 * t + 1];
        float sv[4];
        sv[0] = ok0 ? __fmul_rn(s[0], scale) : -__int_as_float(0x7f800000);
        sv[1] = ok1 ? __fmul_rn(s[1], scale) : -__int_as_float(0x7f800000);
        sv[2] = ok0 ? __fmul_rn(s[2], scale) : -__int_as_float(0x7f800000);
        sv[3] = ok1 ? __fmul_rn(s[3], scale) : -__int_as_float(0x7f800000);
        float mx[2] = {fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3])};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
        }
        if (t == 0) {
            rmax[nw * HB + g] = mx[0];
            rmax[nw * HB + g + 8] = mx[1];
        }
        bar_sync(1, 256);
        float m_new[2], alpha[2], pr[4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int h = g + 8 * i;
            const float bm = fmaxf(fmaxf(rmax[0 * HB + h], rmax[2 * HB + h]), fmaxf(rmax[1 * HB + h], rmax[3 * HB + h]));
            m_new[i] = fmaxf(bm, m_i[i]);
            alpha[i] = exp_tl(__fsub_rn(m_i[i], m_new[i]));
        }
        pr[0] = exp_tl(__fsub_rn(sv[0], m_new[0]));
        pr[1] = exp_tl(__fsub_rn(sv[1], m_new[0]));
        pr[2] = exp_tl(__fsub_rn(sv[2], m_new[1]));
        pr[3] = exp_tl(__fsub_rn(sv[3], m_new[1]));
        float sm[2] = {__fadd_rn(pr[0], pr[1]), __fadd_rn(pr[2], pr[3])};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sm[i] = __fadd_rn(sm[i], __shfl_xor_sync(0xffffffffu, sm[i], 2));
            sm[i] = __fadd_rn(sm[i], __shfl_xor_sync(0xffffffffu, sm[i], 1));
        }
        bar_sync(1, 256);                                         // every compute warp has read rmax
        if (t == 0) {
            rmax[nw * HB + g] = sm[0];
            rmax[nw * HB + g + 8] = sm[1];
        }
        {
            __nv_bfloat162 p0, p1;
            p0.x = __float2bfloat16_rn(pr[0]);
            p0.y = __float2bfloat16_rn(pr[1]);
            p1.x = __float2bfloat16_rn(pr[2]);
            p1.y = __float2bfloat16_rn(pr[3]);
            *reinterpret_cast<__nv_bfloat162*>(pb + pswz(g, nw) + 4 * t) = p0;
            *reinterpret_cast<__nv_bfloat162*>(pb + pswz(g + 8, nw) + 4 * t) = p1;
        }
        bar_sync(1, 256);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int h = g + 8 * i;
            const float bs = sum4<SUM_ORDER>(rmax[0 * HB + h], rmax[1 * HB + h], rmax[2 * HB + h], rmax[3 * HB + h]);
            l_i[i] = __fmaf_rn(alpha[i], l_i[i], bs);
            m_i[i] = m_new[i];
        }
        bar_sync(1, 256);                                         // the sums are read before block b + 1's maxima
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j][0] = __fmul_rn(acc[j][0], alpha[0]);
            acc[j][1] = __fmul_rn(acc[j][1], alpha[0]);
            acc[j][2] = __fmul_rn(acc[j][2], alpha[1]);
            acc[j][3] = __fmul_rn(acc[j][3], alpha[1]);
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t a[4];
            ldmatrix_x4(a, pb + pswz(lane & 15, kk * 2 + (lane >> 4)));
            const int krow = kk * 16 + (lane & 15);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int d0 = (j >> 3) * HALF + nw * 64 + (j & 7) * 8;
                uint32_t b0, b1;
                ldmatrix_x2_trans(b0, b1, kb + swz(krow, d0 >> 1));
                mma16816(acc[j], a, b0, b1);
            }
        }
        if (b + 2 < nblk) bar_arrive(4 + buf, WS_THREADS);         // this key buffer is free for block b + 2
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int h = hb * HB + g + 8 * i;
        const float l = __fadd_rn(l_i[i], exp_tl(__fsub_rn(SINK[h], m_i[i])));
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int d = (j >> 3) * HALF + nw * 64 + (j & 7) * 8 + 2 * t;
            __nv_bfloat162 o;
            o.x = __float2bfloat16_rn(div_full(acc[j][2 * i], l));
            o.y = __float2bfloat16_rn(div_full(acc[j][2 * i + 1], l));
            *reinterpret_cast<__nv_bfloat162*>(OUT + ((size_t)r * H + h) * HD + d) = o;
        }
    }
}

}  // namespace

// grid (rows), 256 threads, dynamic shared memory tf_dsv41_sparse_attn_smem bytes. `_sparse_attn_part`'s prompt
// arguments (one split, FINAL, RING 0, HAS_BASE 0): Q [rows, 32, 512] bf16, WSRC the window rows (row = position - WLO[0]),
// COMP / CSC the compressed plane (packed FP4 codes and E4M3 scales; has_comp 0: no picks), IDX [rows, n_idx] the picks
// (-1: none), POS [rows], SINK [32], OUT [rows, 32, 512] bf16; nblk = 4 + the picks' 32-key blocks.
#define TF_SPARSE_ATTN(ORD_)                                                                                         \
    extern "C" __global__ void __launch_bounds__(256) tf_dsv41_sparse_attn##ORD_(                                    \
        const __nv_bfloat16* Q, const __nv_bfloat16* WSRC, const long long* WLO, const uint8_t* COMP,               \
        const uint8_t* CSC, const long long* IDX, const long long* POS, const float* SINK, __nv_bfloat16* OUT,       \
        float scale, int n_idx, int nblk, int has_comp) {                                                            \
        attn_row<ORD_>(Q, WSRC, WLO, COMP, CSC, IDX, POS, SINK, OUT, scale, n_idx, nblk, has_comp != 0);             \
    }
TF_SPARSE_ATTN(0)
TF_SPARSE_ATTN(1)
TF_SPARSE_ATTN(2)
#undef TF_SPARSE_ATTN

// v3 (warp-specialized): grid (rows), 384 threads, dynamic shared memory tf_dsv41_sparse_attn_ws_smem bytes
// (sparse_attn.cu: 32 KB queries + two 32 KB key blocks + 2 KB p + 0.75 KB); the same arguments as above, order 0.
extern "C" __global__ void __launch_bounds__(384, 1) tf_dsv41_sparse_attn_ws(
    const __nv_bfloat16* Q, const __nv_bfloat16* WSRC, const long long* WLO, const uint8_t* COMP, const uint8_t* CSC,
    const long long* IDX, const long long* POS, const float* SINK, __nv_bfloat16* OUT, float scale, int n_idx, int nblk,
    int has_comp) {
    attn_row_ws<0>(Q, WSRC, WLO, COMP, CSC, IDX, POS, SINK, OUT, scale, n_idx, nblk, has_comp != 0);
}
