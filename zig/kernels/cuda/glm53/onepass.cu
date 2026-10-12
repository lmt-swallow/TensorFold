// GLM-5.3-Flash's one-pass sparse attention of prompt rows (latent.py `_sparse_onepass`, bf16 cache, HBT 32, KTT 32,
// SPLIT 1) with its bits, the duplicated work and the register staging left out. A row's 32 heads attend to its
// selected tokens in list order, 32 a tile, one online softmax (latent.py `_tile`):
//   s = scale * (q . k) (masked -inf past the row's count), tile_m = max_keys s, next_m = max(m, tile_m) (when the
//   tile has a finite score), alpha = exp(m - next_m) (0 from -inf), p = exp(s - next_m), o = o alpha + bf16(p) . k,
//   l = fma(l, alpha, sum_keys p);  out = bf16(o / l).
// The captured kernel's PTX fixes the arithmetic: the dots are m16n8k16 chains (q . k over 32 k16 steps from zero;
// p . k over 2 k16 steps from o alpha), exp is ex2.approx of x * log2(e), the division div.full, the key sums
// p0 + p1 a lane, then xor 2 and xor 1 lane shuffles, then the four key blocks' sums as (w0 + w2) + (w1 + w3). Its 8
// warps each computed the 32-key tile's dots (warps 4-7 duplicates of 0-3); here warps 0-3 compute them once and
// every warp shares the probabilities through shared memory. The next tile's rows arrive by cp.async while the
// current one is used, where the captured kernel staged them through registers.
//
// The arithmetic was read from the captured cubin's PTX and SASS (its only fused multiply-adds are l's fma.rn).
// Checked by tf-cuda-test glm53-onepass (byte for byte against the captured `_sparse_onepass`: rows with no tokens,
// a few, a partial last tile, 2,048 to 2,051, latents whose tiles' maxima move; TF_GLM_ONEPASS=2 the head-split
// variant) and by glm53-prompt's fixtures under TF_GLM_PROMPT_CHECK=1.
#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_glm53_onepass {

constexpr int kHeads = 32;
constexpr int kLw = 512;
constexpr int kKeys = 32;            // KTT
constexpr int kThreads = 256;
constexpr int kRowBytes = kLw * 2;   // 1,024 bytes a latent row, 64 chunks of 16

struct Shared {
    __nv_bfloat16 q[kHeads * kLw];         // swizzled rows
    __nv_bfloat16 kv[2][kKeys * kLw];      // swizzled rows, two tiles
    __nv_bfloat16 p[kHeads * kKeys];       // bf16(p) [head][key]
    float red[4 * kHeads];                 // per key block (warp 0-3) and head: a reduction's partial values
    float tm[kHeads];                      // the tile's max a head
    float alpha[kHeads];
    float l[kHeads];
};

// 16-byte chunk c of row r of a 1,024-byte-row tile, XOR-swizzled by the row's low three bits.
__device__ __forceinline__ uint32_t swz(int r, int c) { return (uint32_t)(r * kRowBytes + ((c ^ (r & 7)) << 4)); }

__device__ __forceinline__ void cp_async16(uint32_t smem, const void* gmem, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
// tl.exp: ex2.approx of x * log2(e), the product rounded on its own
__device__ __forceinline__ float texp(float x) { return ex2(__fmul_rn(x, 1.4426950408889634f)); }
__device__ __forceinline__ float div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

// Tokens t0 .. t0 + 31 of the row's list into kv (rows past the count zero-filled).
__device__ __forceinline__ void load_tile(uint32_t kv, const __nv_bfloat16* lc, const int* tok, int t0, int n) {
    for (int c = threadIdx.x; c < kKeys * (kRowBytes / 16); c += kThreads) {
        const int r = c / (kRowBytes / 16), j = c % (kRowBytes / 16);
        const bool ok = t0 + r < n;
        const int64_t id = ok ? tok[t0 + r] : 0;
        cp_async16(kv + swz(r, j), lc + id * kLw + j * 8, ok);
    }
    cp_async_commit();
}

// Prompt rows r (grid x): QA [R, 32, 512] bf16, LC the latent cache [tokens, 512] bf16, TOK [R, W] token ids, CNT [R]
// counts (rows with 0 left alone), OUT [R, 32, 512] bf16. The scores' dots: SPLIT false, warps 0-3 each take key
// block `warp` for both head tiles (warps 4-7 wait); SPLIT true, warp w takes key block w % 4 for head tile w / 4
// (every warp half the dots). Either way each score is the same m16n8k16 chain and each sum the same order.
template <bool SPLIT>
__device__ __forceinline__ void onepass_body(const __nv_bfloat16* __restrict__ QA, const __nv_bfloat16* __restrict__ LC,
                                             const int* __restrict__ TOK, const int* __restrict__ CNT,
                                             __nv_bfloat16* __restrict__ OUT, int W, float scale) {
    extern __shared__ __align__(128) unsigned char smem[];
    Shared& s = *reinterpret_cast<Shared*>(smem);
    const int r = blockIdx.x;
    const int n = CNT[r];
    if (n == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int which = lane >> 3, row8 = lane & 7;
    const int kb = SPLIT ? (warp & 3) : warp;       // this warp's key block (scores)
    const int mlo = SPLIT ? (warp >> 2) : 0;        // its head tiles mlo .. mhi - 1
    const int mhi = SPLIT ? mlo + 1 : 2;
    const bool scorer = SPLIT || warp < 4;
    const int* tok = TOK + (int64_t)r * W;
    const uint32_t sq = (uint32_t)__cvta_generic_to_shared(s.q);
    const uint32_t skv[2] = {(uint32_t)__cvta_generic_to_shared(s.kv[0]), (uint32_t)__cvta_generic_to_shared(s.kv[1])};
    const uint32_t sp = (uint32_t)__cvta_generic_to_shared(s.p);

    const __nv_bfloat16* qa = QA + (int64_t)r * kHeads * kLw;
    for (int c = threadIdx.x; c < kHeads * (kRowBytes / 16); c += kThreads) {
        const int h = c / (kRowBytes / 16), j = c % (kRowBytes / 16);
        cp_async16(sq + swz(h, j), qa + h * kLw + j * 8, true);
    }
    load_tile(skv[0], LC, tok, 0, n);
    if (threadIdx.x < kHeads) s.l[threadIdx.x] = 0.0f;

    // the scoring warps: each head's running max m (heads g, g + 8, 16 + g, 24 + g of this lane; a SPLIT warp keeps
    // its tile's two)
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    // every warp: o of heads (g, g + 8) [tile 0], (16 + g, 24 + g) [tile 1] x its 64 latent columns (8 n tiles)
    float o[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) o[mt][nt][j] = 0.0f;

    int cur = 0;
    for (int t0 = 0; t0 < n; t0 += kKeys, cur ^= 1) {
        cp_async_wait_all();
        __syncthreads();  // tile cur arrived; the previous tile's readers are done with tile cur ^ 1, p and alpha
        if (t0 + kKeys < n) load_tile(skv[cur ^ 1], LC, tok, t0 + kKeys, n);
        const uint32_t kv = skv[cur];
        // keys kb * 8 + 2t, + 1 of this lane (the scoring warps)
        const bool v0 = t0 + kb * 8 + 2 * t < n, v1 = t0 + kb * 8 + 2 * t + 1 < n;
        float sc[2][4];
        if (scorer) {
            // the scores: q . k of keys kb * 8 .. + 7, the 512 dims in 32 k16 steps from zero
            float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll 4
            for (int ks = 0; ks < kLw / 16; ks += 2) {
                uint32_t b[4];  // steps ks (lo, hi), ks + 1 (lo, hi)
                ldsm_x4(b, kv + swz(kb * 8 + row8, ks * 2 + which));
#pragma unroll
                for (int hs = 0; hs < 2; ++hs) {
                    const int step = ks + hs;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        if (mt < mlo || mt >= mhi) continue;
                        uint32_t a[4];
                        ldsm_x4(a, sq + swz(mt * 16 + (which & 1) * 8 + row8, step * 2 + (which >> 1)));
                        mma(acc[mt], a, b[2 * hs], b[2 * hs + 1]);
                    }
                }
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                sc[mt][0] = v0 ? __fmul_rn(acc[mt][0], scale) : -INFINITY;
                sc[mt][1] = v1 ? __fmul_rn(acc[mt][1], scale) : -INFINITY;
                sc[mt][2] = v0 ? __fmul_rn(acc[mt][2], scale) : -INFINITY;
                sc[mt][3] = v1 ? __fmul_rn(acc[mt][3], scale) : -INFINITY;
            }
            // each head's max over this warp's 8 keys: the lane's two, then xor 2, xor 1
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int mt = i >> 1, hi = i & 1;
                if (mt < mlo || mt >= mhi) continue;
                float x = fmaxf(sc[mt][2 * hi], sc[mt][2 * hi + 1]);
                x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFFu, x, 2));
                x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFFu, x, 1));
                if (t == 0) s.red[kb * kHeads + mt * 16 + hi * 8 + g] = x;
            }
        }
        __syncthreads();
        if (threadIdx.x < kHeads) {
            const int h = threadIdx.x;
            s.tm[h] = fmaxf(fmaxf(s.red[h], s.red[2 * kHeads + h]), fmaxf(s.red[kHeads + h], s.red[3 * kHeads + h]));
        }
        __syncthreads();
        if (scorer) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int mt = i >> 1, hi = i & 1;
                if (mt < mlo || mt >= mhi) continue;
                const int h = mt * 16 + hi * 8 + g;
                const float tile_m = s.tm[h];
                const bool active = tile_m != -INFINITY;
                const float next_m = active ? fmaxf(m[i], tile_m) : m[i];
                const float al = active ? (m[i] == -INFINITY ? 0.0f : texp(__fsub_rn(m[i], next_m))) : 1.0f;
                const float e0 = texp(__fsub_rn(sc[mt][2 * hi], next_m));
                const float e1 = texp(__fsub_rn(sc[mt][2 * hi + 1], next_m));
                const float p0 = (v0 && active) ? e0 : 0.0f;
                const float p1 = (v1 && active) ? e1 : 0.0f;
                m[i] = next_m;
                *reinterpret_cast<__nv_bfloat162*>(&s.p[h * kKeys + kb * 8 + 2 * t]) = __floats2bfloat162_rn(p0, p1);
                if (kb == 0 && t == 0) s.alpha[h] = al;
                // this warp's key sum of head h: the lane's p0 + p1, then xor 2, xor 1
                float x = __fadd_rn(p0, p1);
                x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, 2));
                x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, 1));
                if (t == 0) s.red[kb * kHeads + h] = x;
            }
        }
        __syncthreads();
        if (threadIdx.x < kHeads) {
            const int h = threadIdx.x;
            // the four key blocks' sums as the reference adds them: (w0 + w2) + (w1 + w3)
            const float sum = __fadd_rn(__fadd_rn(s.red[h], s.red[2 * kHeads + h]), __fadd_rn(s.red[kHeads + h], s.red[3 * kHeads + h]));
            s.l[h] = __fmaf_rn(s.l[h], s.alpha[h], sum);
        }
        // the read-out: o alpha + bf16(p) . k over the tile's 32 keys (two k16 steps), this warp's 64 columns
        float al[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            al[mt][0] = s.alpha[mt * 16 + g];
            al[mt][1] = s.alpha[mt * 16 + 8 + g];
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                o[mt][nt][0] = __fmul_rn(o[mt][nt][0], al[mt][0]);
                o[mt][nt][1] = __fmul_rn(o[mt][nt][1], al[mt][0]);
                o[mt][nt][2] = __fmul_rn(o[mt][nt][2], al[mt][1]);
                o[mt][nt][3] = __fmul_rn(o[mt][nt][3], al[mt][1]);
            }
#pragma unroll
        for (int ks = 0; ks < kKeys / 16; ++ks) {
            uint32_t a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int h = mt * 16 + (which & 1) * 8 + row8;
                ldsm_x4(a[mt], sp + (uint32_t)((h * kKeys + ks * 16 + (which >> 1) * 8) * 2));
            }
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                // x4.trans: keys (lo, hi) x column chunks (2 np, 2 np + 1) of this warp's 64 columns
                uint32_t b[4];
                const int key = ks * 16 + (which & 1) * 8 + row8;
                ldsm_x4_t(b, kv + swz(key, warp * 8 + 2 * np + (which >> 1)));
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    mma(o[mt][2 * np], a[mt], b[0], b[1]);
                    mma(o[mt][2 * np + 1], a[mt], b[2], b[3]);
                }
            }
        }
    }
    __syncthreads();
    // out = bf16(o / l)
    __nv_bfloat16* out = OUT + (int64_t)r * kHeads * kLw;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int h0 = mt * 16 + g, h1 = h0 + 8;
        const float l0 = s.l[h0], l1 = s.l[h1];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int col = warp * 64 + nt * 8 + 2 * t;
            *reinterpret_cast<__nv_bfloat162*>(&out[h0 * kLw + col]) =
                __floats2bfloat162_rn(div_full(o[mt][nt][0], l0), div_full(o[mt][nt][1], l0));
            *reinterpret_cast<__nv_bfloat162*>(&out[h1 * kLw + col]) =
                __floats2bfloat162_rn(div_full(o[mt][nt][2], l1), div_full(o[mt][nt][3], l1));
        }
    }
}

__global__ void __launch_bounds__(kThreads) onepass_kernel(const __nv_bfloat16* __restrict__ QA,
                                                           const __nv_bfloat16* __restrict__ LC,
                                                           const int* __restrict__ TOK, const int* __restrict__ CNT,
                                                           __nv_bfloat16* __restrict__ OUT, int W, float scale) {
    onepass_body<false>(QA, LC, TOK, CNT, OUT, W, scale);
}

__global__ void __launch_bounds__(kThreads) onepass_split_kernel(const __nv_bfloat16* __restrict__ QA,
                                                                 const __nv_bfloat16* __restrict__ LC,
                                                                 const int* __restrict__ TOK, const int* __restrict__ CNT,
                                                                 __nv_bfloat16* __restrict__ OUT, int W, float scale) {
    onepass_body<true>(QA, LC, TOK, CNT, OUT, W, scale);
}

}  // namespace tf_glm53_onepass

extern "C" __constant__ unsigned int tf_glm53_onepass_smem = sizeof(tf_glm53_onepass::Shared);
