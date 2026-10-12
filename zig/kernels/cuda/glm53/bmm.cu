// The BF16-weight matmul of the Mia fork's qmm.py (`_bmm`, with `_reduce` for the split grid) at the mma rate:
// out [m, n] (bf16, or fp32 with F32) = x [m, k] (row stride x_stride) times w^T (w [n, k] bf16, rows contiguous).
//
// What the captured `_bmm` computes for an output (recovered from its SASS), and what this kernel computes the same way:
//  - K in SK equal slices; a slice's sum = one mma.m16n8k16 (bf16 in, fp32 accumulate) chain from +0 over the slice's
//    k values in steps of 16, in order (Triton folds `acc + tl.dot(x, w^T)` into the dot's accumulator, so no add
//    appears between the BK steps);
//  - the output = slice 0's sum, then + slice 1's, + slice 2's, ... (SEQ: in the program; the split grid: `_reduce`),
//    each an fp32 add; SK = 1 is the one chain;
//  - bf16 outputs rounded to nearest even from that fp32.
// Neither the tile shape, the K step nor the pipeline depth enters these: every choice gives the same bits.

#include <cuda_bf16.h>
#include <cstdint>

namespace {


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

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

// BM x BN outputs a program, WM x WN warps (each WTM x WTN), STAGES stages of BK k values in flight. SPLIT: SK > 1 (a
// slice's chain apart from the running total). SWZ (BK 64): staged rows unpadded, their eight 16-byte pieces permuted
// by the row's low 3 bits, so ldmatrix's 8 rows of a piece fall in distinct banks; else rows padded by 8 halfs.
template <int BM, int BN, int BK, int WM, int WN, int STAGES, bool SPLIT, bool F32, bool SWZ = false>
__device__ __forceinline__ void bmm(const __nv_bfloat16* __restrict__ X, int x_stride, const __nv_bfloat16* __restrict__ W,
                                    void* __restrict__ OUT, int M, int N, int K, int SK) {
    constexpr int WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8, T = WM * WN * 32;
    constexpr int LDS = SWZ ? BK : BK + 8;   // halfs a staged row
    static_assert(NI % 2 == 0, "B fragments load two n8 blocks at a time");
    static_assert(!SWZ || BK == 64, "the swizzle permutes a row's eight 16-byte pieces");
    // halfs from a stage's base to row r's 16-byte piece p
    auto soff = [](int r, int p) { return r * LDS + (SWZ ? (p ^ (r & 7)) : p) * 8; };
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);                  // [STAGES][BM][LDS]
    __nv_bfloat16* Bs = As + (size_t)STAGES * BM * LDS;                                // [STAGES][BN][LDS]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / WN, wn = warp % WN;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int KT = K / BK, PT = KT / SK;                                               // stages a slice

    auto load = [&](int kt, int buf) {
        const int k0 = kt * BK;
        __nv_bfloat16* as = As + (size_t)buf * BM * LDS;
        __nv_bfloat16* bs = Bs + (size_t)buf * BN * LDS;
#pragma unroll
        for (int c = 0; c < BM * BK / 8 / T; ++c) {
            const int idx = threadIdx.x + c * T, r = idx / (BK / 8), part = idx % (BK / 8);
            const int row = m0 + r;
            cp_async16(as + soff(r, part), X + (size_t)(row < M ? row : 0) * x_stride + k0 + part * 8, row < M);
        }
#pragma unroll
        for (int c = 0; c < BN * BK / 8 / T; ++c) {
            const int idx = threadIdx.x + c * T, r = idx / (BK / 8), part = idx % (BK / 8);
            const int col = n0 + r;
            cp_async16(bs + soff(r, part), W + (size_t)(col < N ? col : 0) * K + k0 + part * 8, col < N);
        }
    };

    float acc[MI][NI][4];                    // the running chain (SPLIT: the current slice's)
    float tot[SPLIT ? MI : 1][SPLIT ? NI : 1][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    // ldmatrix lanes: A rows (l & 7) + 8 ((l >> 3) & 1), k + 8 (l >> 4); B rows (l & 7) + 8 (l >> 4), k + 8 ((l >> 3) & 1)
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        cp_async_commit();
        const __nv_bfloat16* as = As + (size_t)(kt % STAGES) * BM * LDS;
        const __nv_bfloat16* bs = Bs + (size_t)(kt % STAGES) * BN * LDS;
#pragma unroll
        for (int k16 = 0; k16 < BK / 16; ++k16) {
            uint32_t a[MI][4], b[NI][2];
#pragma unroll
            for (int i = 0; i < MI; ++i) ldsm4(a[i], as + soff(wm * WTM + i * 16 + a_row, (k16 * 16 + a_col) / 8));
#pragma unroll
            for (int j = 0; j < NI; j += 2) {
                uint32_t r[4];
                ldsm4(r, bs + soff(wn * WTN + j * 8 + b_row, (k16 * 16 + b_col) / 8));
                b[j][0] = r[0]; b[j][1] = r[1]; b[j + 1][0] = r[2]; b[j + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j) mma16816(acc[i][j], a[i], b[j][0], b[j][1]);
        }
        if constexpr (SPLIT) {
            if ((kt + 1) % PT == 0) {          // a slice ends: the total takes it, the next slice starts from +0
                const bool first = kt + 1 == PT;
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int j = 0; j < NI; ++j)
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            tot[i][j][c] = first ? acc[i][j][c] : __fadd_rn(tot[i][j][c], acc[i][j][c]);
                            acc[i][j][c] = 0.f;
                        }
            }
        }
    }
    cp_async_wait<0>();

    // acc / tot: rows g (c0, c1) and g + 8 (c2, c3) of each m16 tile, columns 2 t and + 1 of each n8 block
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            const int col = n0 + wn * WTN + j * 8 + 2 * t;
            if (col >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * WTM + i * 16 + g + 8 * h;
                if (row >= M) continue;
                const float v0 = SPLIT ? tot[i % (SPLIT ? MI : 1)][j % (SPLIT ? NI : 1)][2 * h] : acc[i][j][2 * h];
                const float v1 = SPLIT ? tot[i % (SPLIT ? MI : 1)][j % (SPLIT ? NI : 1)][2 * h + 1] : acc[i][j][2 * h + 1];
                if constexpr (F32) {
                    *reinterpret_cast<float2*>(reinterpret_cast<float*>(OUT) + (size_t)row * N + col) = make_float2(v0, v1);
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(OUT) + (size_t)row * N + col) =
                        __floats2bfloat162_rn(v0, v1);
                }
            }
        }
}

}  // namespace

// grid (ceil(m / BM), ceil(n / BN)), WM WN 32 threads, dynamic shared memory STAGES (BM + BN) (BK + 8) 2 bytes.
// Requires k % (BK sk) == 0, x_stride % 8 == 0, n even, and 16-byte aligned x and w. `_1` kernels take sk 1 (one
// chain), `_s` kernels sk > 1; `_b` write bf16, `_f` fp32.
#define TF_BMM(NAME, BM_, BN_, BK_, WM_, WN_, ST_, SPLIT_, F32_, SWZ_)                                                        \
    extern "C" __global__ void __launch_bounds__(WM_ * WN_ * 32) NAME(                                              \
        const __nv_bfloat16* X, int x_stride, const __nv_bfloat16* W, void* OUT, int M, int N, int K, int SK) {     \
        bmm<BM_, BN_, BK_, WM_, WN_, ST_, SPLIT_, F32_, SWZ_>(X, x_stride, W, OUT, M, N, K, SPLIT_ ? SK : 1);                  \
    }
TF_BMM(tf_glm53_bmm_128x256_1b, 128, 256, 32, 2, 4, 3, false, false, false)
TF_BMM(tf_glm53_bmm_128x256k64z_1b, 128, 256, 64, 2, 4, 2, false, false, true)
TF_BMM(tf_glm53_bmm_128x128k64_1b, 128, 128, 64, 2, 4, 2, false, false, false)
TF_BMM(tf_glm53_bmm_128x128k64z_1b, 128, 128, 64, 2, 4, 3, false, false, true)
TF_BMM(tf_glm53_bmm_128x128k64_sb, 128, 128, 64, 2, 4, 2, true, false, false)
TF_BMM(tf_glm53_bmm_128x128k64_sf, 128, 128, 64, 2, 4, 2, true, true, false)
TF_BMM(tf_glm53_bmm_128x128k64z_sb, 128, 128, 64, 2, 4, 3, true, false, true)
TF_BMM(tf_glm53_bmm_128x128k64z_sf, 128, 128, 64, 2, 4, 3, true, true, true)
TF_BMM(tf_glm53_bmm_128x64_sb, 128, 64, 32, 4, 2, 4, true, false, false)
#undef TF_BMM
