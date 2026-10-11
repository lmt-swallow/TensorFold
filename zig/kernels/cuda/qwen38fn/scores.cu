// Ours: attention.py `_scores` (the indexer's block scores) for many rows a program, with the captured Triton
// kernel's bytes. `_scores` runs one program per (row, 64 blocks) and reads the 64 pooled keys again for every row;
// here a program takes RT (32) rows and reads the keys once.
//
// Bits (the captured cubin's PTX, num_warps 4, HI 4, DI 128, BB 64): for block b and head h, lane L (0..15) of the
// row's 16 lanes holds dims d = 8L .. 8L + 7 and computes p_L = k[d+1] q[d+1], then fma over d, d+2, d+3, .., d+7 in
// that order; the lanes are summed by xor butterflies 8, 4, 2, 1 (s(L) = s(L) + s(L ^ m)); the heads are added as
// total = max(s_0, 0) + 0, then total + max(s_h, 0) in head order; the score is div.full.f32(total, sqrt(128)).
// Every lane ends with the same sum, so one thread here computes the tree in that order.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_qfn_sc {

constexpr int DI = 128, HI = 4, BB = 64, RT = 32, THREADS = 256, KPAD = DI + 8;
// dynamic shared bytes: the rows' queries as fp32, then the 64 keys as bf16
constexpr int SMEM = RT * HI * DI * 4 + BB * KPAD * 2;

__device__ __forceinline__ float div_full(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// One head's row . key as _scores sums it: k the key's 128 bf16 dims in registers (two a word), q 128 fp32 dims.
__device__ __forceinline__ float dot(const uint32_t (&k)[64], const float* q) {
    float p[16];
#pragma unroll
    for (int L = 0; L < 16; ++L) {
        const float4 q0 = *reinterpret_cast<const float4*>(q + 8 * L);
        const float4 q1 = *reinterpret_cast<const float4*>(q + 8 * L + 4);
        const uint32_t* w = k + 4 * L;  // dims 8L .. 8L + 7: (lo, hi) of w[0..3]
        float s = __fmul_rn(hi(w[0]), q0.y);
        s = __fmaf_rn(lo(w[0]), q0.x, s);
        s = __fmaf_rn(lo(w[1]), q0.z, s);
        s = __fmaf_rn(hi(w[1]), q0.w, s);
        s = __fmaf_rn(lo(w[2]), q1.x, s);
        s = __fmaf_rn(hi(w[2]), q1.y, s);
        s = __fmaf_rn(lo(w[3]), q1.z, s);
        s = __fmaf_rn(hi(w[3]), q1.w, s);
        p[L] = s;
    }
#pragma unroll
    for (int m = 8; m >= 1; m >>= 1) {
        float n[16];
#pragma unroll
        for (int L = 0; L < 16; ++L) n[L] = __fadd_rn(p[L], p[L ^ m]);
#pragma unroll
        for (int L = 0; L < 16; ++L) p[L] = n[L];
    }
    return p[0];
}

}  // namespace tf_qfn_sc

// SC [rows, NB] fp32: for each sparse row r (complete = (POS0 + r + 1) / RATIO > TOP) and block b < complete, the
// score of row r's indexer queries IQ [rows, HI, DI] bf16 against POOLED [b] bf16 [DI]. Grid (ceil(rows / RT),
// ceil(blocks / BB)) with blocks >= every row's complete count, 256 threads, SMEM dynamic shared bytes; RATIO 4 and
// TOP 512 (Flash Next's).
extern "C" __global__ void __launch_bounds__(256) tf_qfn_scores_rows(const __nv_bfloat16* __restrict__ IQ,
                                                                     const __nv_bfloat16* __restrict__ POOLED,
                                                                     const int* __restrict__ POS0,
                                                                     float* __restrict__ SC, int NB, int rows) {
    using namespace tf_qfn_sc;
    constexpr int RATIO = 4, TOP = 512;
    extern __shared__ __align__(16) unsigned char smem[];
    float* qs = reinterpret_cast<float*>(smem);                                   // [RT][HI * DI]
    __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(smem + RT * HI * DI * 4);  // [BB][KPAD]
    const int r0 = blockIdx.x * RT, b0 = blockIdx.y * BB;
    const int pos0 = POS0[0];
    // the program's last row completes the most blocks
    const int last = min(r0 + RT, rows) - 1;
    const int most = (pos0 + last + 1) / RATIO;
    if (most <= TOP || b0 >= most) return;
    for (int i = threadIdx.x; i < BB * (DI / 8); i += THREADS) {
        const int b = i / (DI / 8), c = 8 * (i % (DI / 8));
        const uint4 v = b0 + b < most ? *reinterpret_cast<const uint4*>(POOLED + (size_t)(b0 + b) * DI + c)
                                      : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(ks + b * KPAD + c) = v;
    }
    for (int i = threadIdx.x; i < RT * HI * DI; i += THREADS) {
        const int r = i / (HI * DI);
        qs[i] = r0 + r < rows ? __bfloat162float(IQ[(size_t)r0 * HI * DI + i]) : 0.f;
    }
    __syncthreads();
    const float scale = 11.313708305358887f;  // _scores' tl.sqrt(DI * 1.0): 0x413504F3
    for (int i = threadIdx.x; i < RT * BB; i += THREADS) {
        const int r = i / BB, b = i % BB;
        const int row = r0 + r;
        if (row >= rows) continue;
        const int complete = (pos0 + row + 1) / RATIO;
        if (complete <= TOP || b0 + b >= complete) continue;
        uint32_t k[64];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const uint4 v = *reinterpret_cast<const uint4*>(ks + b * KPAD + 8 * c);
            k[4 * c] = v.x;
            k[4 * c + 1] = v.y;
            k[4 * c + 2] = v.z;
            k[4 * c + 3] = v.w;
        }
        float total = 0.f;
#pragma unroll
        for (int h = 0; h < HI; ++h) {
            const float s = fmaxf(dot(k, qs + r * HI * DI + h * DI), 0.f);
            total = h == 0 ? __fadd_rn(s, 0.f) : __fadd_rn(total, s);
        }
        SC[(size_t)row * NB + b0 + b] = div_full(total, scale);
    }
}

// The fp32 words where a and b differ over rows [rows, cols] (row stride NB), added into *count.
extern "C" __global__ void tf_qfn_scores_diff(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, int NB,
                                              int rows, int cols, unsigned long long* __restrict__ count) {
    unsigned long long c = 0;
    const long long n = (long long)rows * cols;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / cols, col = i % cols;
        c += a[r * NB + col] != b[r * NB + col];
    }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(count, c);
}
