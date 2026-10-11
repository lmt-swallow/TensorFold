// Ours: exl3_mm.py `F16.__call__` (the `_f16_mm` Triton kernel, then `_reduce` when the matrix has K slices) for a
// prompt chunk's rows in one launch (two with slices), with the Triton kernels' bytes.
//
// Bits. `_f16_mm` computes each (row, column) of K slice s as one mma.m16n8k16 chain from zero over the slice's K in
// ascending 16-steps (`acc = tl.dot(x.to(fp16), w^T, acc)` in BK steps), x rounded to fp16 (cvt.rn); without slices it
// stores bf16(chain) (cvt.rn); with slices `_reduce` adds them to zero in slice order and stores bf16. A row of an mma
// tile never depends on the tile's other rows or columns, so the tile shape here (128 rows, 128 columns a program)
// does not change any value.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_qfn_f16 {

constexpr int BM = 128, BN = 128, BK = 64, PAD = BK + 8, THREADS = 256;

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ half to_half(__nv_bfloat16 v) { return __float2half_rn(__bfloat162float(v)); }
__device__ __forceinline__ half to_half(half v) { return v; }

// Program (128 rows, 128 columns, K slice blockIdx.z): 8 warps, warp w rows 32 * (w % 4) .. + 32, columns
// 64 * (w / 4) .. + 64. Without slices out is bf16 [M, o_stride]; with them part is fp32 [SK, M, N].
template <typename XT>
__device__ __forceinline__ void body(const XT* __restrict__ X, int x_stride, const half* __restrict__ W,
                                     __nv_bfloat16* __restrict__ out, int o_stride, float* __restrict__ part, int M,
                                     int N, int K, int KS) {
    __shared__ __align__(16) half xs[BM][PAD];
    __shared__ __align__(16) half ws[BN][PAD];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, s = blockIdx.z;
    const int wm = (warp & 3) * 32, wn = (warp >> 2) * 64;
    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
#pragma unroll 1
    for (int kk = 0; kk < KS; kk += BK) {
        const int k0 = s * KS + kk;
        __syncthreads();
        for (int i = threadIdx.x; i < BM * (BK / 2); i += THREADS) {
            const int r = i / (BK / 2), c = 2 * (i % (BK / 2));
            const int m = m0 + r;
            half2 v = __half2half2(__float2half_rn(0.f));
            if (m < M) {
                const XT* p = X + (size_t)m * x_stride + k0 + c;
                v = __halves2half2(to_half(p[0]), to_half(p[1]));
            }
            *reinterpret_cast<half2*>(&xs[r][c]) = v;
        }
        for (int i = threadIdx.x; i < BN * (BK / 8); i += THREADS) {
            const int r = i / (BK / 8), c = 8 * (i % (BK / 8));
            const int n = n0 + r;
            const uint4 v = n < N ? *reinterpret_cast<const uint4*>(W + (size_t)n * K + k0 + c) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&ws[r][c]) = v;
        }
        __syncthreads();
#pragma unroll
        for (int k16 = 0; k16 < BK; k16 += 16) {
            uint32_t a[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const half* r0 = &xs[wm + i * 16 + g][k16 + 2 * t];
                const half* r1 = &xs[wm + i * 16 + g + 8][k16 + 2 * t];
                a[i][0] = *reinterpret_cast<const uint32_t*>(r0);
                a[i][1] = *reinterpret_cast<const uint32_t*>(r1);
                a[i][2] = *reinterpret_cast<const uint32_t*>(r0 + 8);
                a[i][3] = *reinterpret_cast<const uint32_t*>(r1 + 8);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const half* c0 = &ws[wn + j * 8 + g][k16 + 2 * t];
                const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(c0), *reinterpret_cast<const uint32_t*>(c0 + 8)};
#pragma unroll
                for (int i = 0; i < 2; ++i) mma16816(acc[i][j], a[i], b);
            }
        }
    }
    const bool slices = K / KS > 1;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + wm + i * 16 + g + 8 * h;
                const int n = n0 + wn + j * 8 + 2 * t;
                if (m >= M) continue;
                if (slices) {
                    float* p = part + ((size_t)s * M + m) * N + n;
                    if (n < N) p[0] = acc[i][j][2 * h];
                    if (n + 1 < N) p[1] = acc[i][j][2 * h + 1];
                } else {
                    if (n < N) out[(size_t)m * o_stride + n] = __float2bfloat16_rn(acc[i][j][2 * h]);
                    if (n + 1 < N) out[(size_t)m * o_stride + n + 1] = __float2bfloat16_rn(acc[i][j][2 * h + 1]);
                }
            }
}

}  // namespace tf_qfn_f16

// x bf16 or fp16 [M, K] (row stride x_stride), W fp16 [N, K]: K slice z of KS (a multiple of 64) over grid
// (ceil(M / 128), ceil(N / 128), K / KS), 256 threads; one slice: out bf16 [M, N] (row stride o_stride), else part
// fp32 [K / KS, M, N] for tf_qfn_f16_reduce.
extern "C" __global__ void __launch_bounds__(256) tf_qfn_f16_rows_bf16(const __nv_bfloat16* X, int x_stride,
                                                                       const half* W, __nv_bfloat16* out, int o_stride,
                                                                       float* part, int M, int N, int K, int KS) {
    tf_qfn_f16::body(X, x_stride, W, out, o_stride, part, M, N, K, KS);
}

extern "C" __global__ void __launch_bounds__(256) tf_qfn_f16_rows_fp16(const half* X, int x_stride, const half* W,
                                                                       __nv_bfloat16* out, int o_stride, float* part,
                                                                       int M, int N, int K, int KS) {
    tf_qfn_f16::body(X, x_stride, W, out, o_stride, part, M, N, K, KS);
}

// `_reduce`: out[m, n] bf16 (row stride o_stride) = 0 + part[0, m, n] + part[1, m, n] + ... in slice order.
extern "C" __global__ void __launch_bounds__(256) tf_qfn_f16_reduce(const float* __restrict__ part, int SK, int M,
                                                                    int N, __nv_bfloat16* __restrict__ out,
                                                                    int o_stride) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    float acc = 0.f;
    for (int s = 0; s < SK; ++s) acc = __fadd_rn(acc, part[((size_t)s * M + m) * N + n]);
    out[(size_t)m * o_stride + n] = __float2bfloat16_rn(acc);
}

// The bf16 values where a [rows, cols] (row stride sa) and b (row stride sb) differ, added into *count.
extern "C" __global__ void tf_qfn_f16_count_diff(const uint16_t* __restrict__ a, int sa, const uint16_t* __restrict__ b,
                                                 int sb, int rows, int cols, unsigned long long* __restrict__ count) {
    unsigned long long c = 0;
    const long long n = (long long)rows * cols;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / cols, col = i % cols;
        c += a[r * sa + col] != b[r * sb + col];
    }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(count, c);
}
