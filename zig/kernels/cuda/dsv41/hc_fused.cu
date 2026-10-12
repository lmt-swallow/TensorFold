// The prompt path's mHC between two sublayers in one pass: the fork's kernels.py `_hc_post` (the residual streams after
// a sublayer) and the next hc_pre's `_hc_partial` (its 24 mixing dots and sum of squares over those streams), so the
// streams written are not read back. S = 4 streams of D columns (D a multiple of 512), WORLD = 2 rank partials,
// NB = 40 K blocks of KB = 4 D / 40 = 512 columns (block b: stream b / (D / 512), columns 512 (b % (D / 512)) ..),
// SUB = 128 columns a step.
//
// What the captured kernels compute, and what this kernel computes the same way:
//  - _hc_post, per row r, column d and stream s: y = bf16(G0 + G1) (rank 0's partial first); out =
//    bf16(fma(ps, y, fma(c3, x3, fma(c2, x2, fma(c0, x0, c1 x1))))) with ps = POST[r, s], c_k = COMB[r, k, s], x_k the
//    bf16 stream k at d (kernels.py _hc_mix_rows spells the same contraction out; c1 x1 its own rounding);
//  - _hc_partial, per row r and block b (recovered earlier from its SASS and confirmed by tf-cuda-test dsv41-hc-probe
//    on pkern-opus-old's hc_partial.cu): acc[m] = (((0 + t_0) + t_1) + t_2) + t_3 over the block's four 128-column
//    steps, a step's t = the 32-lane butterfly (xor 16, 8, 4, 2, 1) of lane L's v = fma(w3, x3, fma(w2, x2, fma(w0,
//    x0, w1 x1))) over its columns 4 L .. 4 L + 3; the sum of squares: per column fma(x, x, ss) over the steps from 0,
//    then a = ss_0 + ss_1 and c = ss_2 + ss_3 of a lane's four columns, each through the butterfly xor 8, 4, 2, 1,
//    then a + c, then that plus its xor 16 partner.
// The partial reads the streams _hc_post wrote, as bf16: here they are the same bf16 values, taken before they are
// stored. The butterfly is taken as a reduce-scatter (each stage halves the mixes a lane carries, lane L ends with
// mix L): every addition in it is one of the butterfly's, between the same two partial sums, so the sums are the
// butterfly's bits with 31 shuffles a stream and step instead of 120. Built with --fmad=false: every product, sum
// and fma above is spelled out.
//
// tf-cuda-test dsv41-hc-fused-probe holds XOUT and PART to the fork's unfused pair at 2,048, 777, 64 and 1 rows, on
// tools/zig/dsv41/dump_hc_mix.py's fixture (a real layer's mixing weights; the comparison that holds the fork's fused
// `_hc_mix_part` to the pair) and on rows spanning 2^-12 .. 2^12 against the captured kernels. VARIANT is its
// sensitivity check (0 is the kernel the prompt path runs): 1 the post without contraction (ps y + (((c0 x0 + c1 x1)
// + c2 x2) + c3 x3)), 2 the dots' butterfly in the other order (xor 1 .. 16), 3 the sum of squares added column by
// column in order; each must change outputs the exact kernel matches. Result (GB10): 0 of every output differs at
// every row count on both inputs; on the fixture's 2,048 rows variant 1 changes 766 streams, 11,545 of 1,966,080 dots
// and 235 of 81,920 sums of squares, variant 2 1,283,809 dots, variant 3 65,802 sums of squares (at 1 row: 0, 615
// dots, 30 sums). 1.32 ms a 2,048-row call against 2.38 for _hc_post + _hc_partial (128 registers, no spills: 8 rows
// a program, two programs an SM).

#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int S = 4;            // streams
constexpr int M = 24;           // mixes
constexpr int NB = 40;          // K blocks of a row
constexpr int KB = 512;         // columns a block
constexpr int SUB = 128;        // columns a step
constexpr int STEPS = KB / SUB;

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ void unpack4(uint2 v, float (&x)[4]) {
    x[0] = __uint_as_float(v.x << 16);
    x[1] = __uint_as_float(v.x & 0xFFFF0000u);
    x[2] = __uint_as_float(v.y << 16);
    x[3] = __uint_as_float(v.y & 0xFFFF0000u);
}

__device__ __forceinline__ uint2 pack4(const float (&o)[4]) {
    return make_uint2((__float_as_uint(o[0]) >> 16) | (__float_as_uint(o[1]) & 0xFFFF0000u),
                      (__float_as_uint(o[2]) >> 16) | (__float_as_uint(o[3]) & 0xFFFF0000u));
}

// The 32-lane sums of v[0..23] (slots 24..31 zero), lane L returned slot L's: stage o (16, 8, 4, 2, 1; VARIANT 2:
// 1 .. 16) keeps the half of the slots whose bit o matches the lane's and adds the partner's copy of them.
template <int VARIANT>
__device__ __forceinline__ float reduce_scatter(float (&v)[32], int lane) {
#pragma unroll
    for (int st = 0; st < 5; ++st) {
        const int o = VARIANT == 2 ? (1 << st) : (16 >> st);
        const bool hi = (lane & o) != 0;
        // slots are kept as a packed array: after the stage with bit o, slot i of the 2n left holds the original slot
        // with bit o equal to the lane's and the remaining bits i (in the stage's order)
        const int n = 16 >> st;
#pragma unroll
        for (int i = 0; i < n; ++i) {
            const float keep = hi ? v[i + n] : v[i];
            const float send = hi ? v[i] : v[i + n];
            v[i] = __fadd_rn(keep, __shfl_xor_sync(0xffffffffu, send, o));
        }
    }
    return v[0];
}

// The slot (mix) lane L ends with: in the stage order, each stage's lane bit picks the upper or lower half.
template <int VARIANT>
__device__ __forceinline__ int slot_of(int lane) {
    if constexpr (VARIANT == 2) {
        // stages 1, 2, 4, 8, 16 split the packed slots 32 -> 16 -> ...: the first stage's bit is the slots' top bit
        int s = 0;
#pragma unroll
        for (int st = 0; st < 5; ++st) s |= ((lane >> st) & 1) << (4 - st);
        return s;
    } else {
        return lane;
    }
}

// grid (ceil(R / ROWS), D / 512), 256 threads (8 warps, ROWS / 8 rows a warp), dynamic shared memory 24 * 4 * 128 * 4
// bytes (a step's FN columns of the 4 streams). X, XOUT [R, 4 D] bf16 (distinct), G0 [R, D] fp32 and G1 = G0 + rs
// (elements), POST [R, 4], COMB [R, 4, 4] fp32, FN [24, 4 D] fp32, PART [R, 40, 32] fp32 (0..23 the dots, 24 the sum
// of squares; 25..31 untouched).
template <int ROWS, int VARIANT>
__device__ __forceinline__ void post_partial(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ XOUT,
                                             const float* __restrict__ G0, long long rs, const float* __restrict__ POST,
                                             const float* __restrict__ COMB, const float* __restrict__ FN,
                                             float* __restrict__ PART, int R, int D) {
    extern __shared__ __align__(16) float fns[];                 // [24][4][128]
    constexpr int RPW = ROWS / 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int cb = blockIdx.y, cbs = D / KB;
    const int wide = S * D;
    const float* G1 = G0 + rs;
    const int me = slot_of<VARIANT>(lane);
    float acc[RPW][S], ss[RPW][S][4];
#pragma unroll
    for (int q = 0; q < RPW; ++q)
#pragma unroll
        for (int s = 0; s < S; ++s) {
            acc[q][s] = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) ss[q][s][j] = 0.f;
        }

#pragma unroll 1
    for (int t = 0; t < STEPS; ++t) {
        __syncthreads();                                          // the step before is done with fns
        for (int i = threadIdx.x; i < M * S * SUB / 4; i += blockDim.x) {
            const int m = i / (S * SUB / 4), rem = i % (S * SUB / 4), s = rem / (SUB / 4), c = rem % (SUB / 4);
            reinterpret_cast<float4*>(fns)[i] =
                *reinterpret_cast<const float4*>(FN + (size_t)m * wide + (size_t)s * D + cb * KB + t * SUB + 4 * c);
        }
        __syncthreads();
        const int d = cb * KB + t * SUB + 4 * lane;               // the lane's first column within a stream
#pragma unroll
        for (int q = 0; q < RPW; ++q) {
            // rows past R compute on row R - 1 and store nothing: every lane of the warp stays in the shuffles (a
            // branch around them makes the compiler emulate them as divergent collectives)
            const int r_ = blockIdx.x * ROWS + warp * RPW + q;
            const bool live = r_ < R;
            const int r = live ? r_ : R - 1;
            float y[4], x[S][4], o[S][4];
            {
                const float4 a = *reinterpret_cast<const float4*>(G0 + (size_t)r * D + d);
                const float4 b = *reinterpret_cast<const float4*>(G1 + (size_t)r * D + d);
                y[0] = bf16r(__fadd_rn(a.x, b.x));
                y[1] = bf16r(__fadd_rn(a.y, b.y));
                y[2] = bf16r(__fadd_rn(a.z, b.z));
                y[3] = bf16r(__fadd_rn(a.w, b.w));
            }
#pragma unroll
            for (int k = 0; k < S; ++k) unpack4(*reinterpret_cast<const uint2*>(X + (size_t)r * wide + (size_t)k * D + d), x[k]);
            const float* cm = COMB + (size_t)r * S * S;
#pragma unroll
            for (int s = 0; s < S; ++s) {
                const float c0 = cm[0 * S + s], c1 = cm[1 * S + s], c2 = cm[2 * S + s], c3 = cm[3 * S + s];
                const float ps = POST[(size_t)r * S + s];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float v;
                    if constexpr (VARIANT == 1) {
                        v = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(c0, x[0][j]), __fmul_rn(c1, x[1][j])), __fmul_rn(c2, x[2][j])), __fmul_rn(c3, x[3][j]));
                        v = __fadd_rn(__fmul_rn(ps, y[j]), v);
                    } else {
                        v = __fmul_rn(c1, x[1][j]);
                        v = __fmaf_rn(c0, x[0][j], v);
                        v = __fmaf_rn(c2, x[2][j], v);
                        v = __fmaf_rn(c3, x[3][j], v);
                        v = __fmaf_rn(ps, y[j], v);
                    }
                    o[s][j] = bf16r(v);
                }
                if (live) *reinterpret_cast<uint2*>(XOUT + (size_t)r * wide + (size_t)s * D + d) = pack4(o[s]);
            }
#pragma unroll
            for (int s = 0; s < S; ++s) {
#pragma unroll
                for (int j = 0; j < 4; ++j) ss[q][s][j] = __fmaf_rn(o[s][j], o[s][j], ss[q][s][j]);
                float v[32];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const float4 w = reinterpret_cast<const float4*>(fns + (m * S + s) * SUB)[lane];
                    float e = __fmul_rn(w.y, o[s][1]);
                    e = __fmaf_rn(w.x, o[s][0], e);
                    e = __fmaf_rn(w.z, o[s][2], e);
                    e = __fmaf_rn(w.w, o[s][3], e);
                    v[m] = e;
                }
#pragma unroll
                for (int m = M; m < 32; ++m) v[m] = 0.f;
                const float sum = reduce_scatter<VARIANT>(v, lane);
                acc[q][s] = __fadd_rn(acc[q][s], sum);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = blockIdx.x * ROWS + warp * RPW + q;
        const bool live = r < R;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            float tot;
            if constexpr (VARIANT == 3) {
                // every column's square sum (lane L's four, in order), then the 128 in column order through lane 0
                float cs[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) cs[j] = ss[q][s][j];
                tot = 0.f;
#pragma unroll 1
                for (int l = 0; l < 32; ++l)
#pragma unroll
                    for (int j = 0; j < 4; ++j) tot = __fadd_rn(tot, __shfl_sync(0xffffffffu, cs[j], l));
            } else {
                float a = __fadd_rn(ss[q][s][0], ss[q][s][1]), c = __fadd_rn(ss[q][s][2], ss[q][s][3]);
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) {
                    a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
                    c = __fadd_rn(c, __shfl_xor_sync(0xffffffffu, c, o));
                }
                tot = __fadd_rn(a, c);
                tot = __fadd_rn(tot, __shfl_xor_sync(0xffffffffu, tot, 16));
            }
            float* row = PART + ((size_t)r * NB + s * cbs + cb) * 32;
            if (live && me < M) row[me] = acc[q][s];
            if (live && lane == 0) row[M] = tot;
        }
    }
}

}  // namespace

#define TF_HC_FUSED(NAME, ROWS_, VARIANT_)                                                                            \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME(                                                           \
        const __nv_bfloat16* X, __nv_bfloat16* XOUT, const float* G0, long long rs, const float* POST,                \
        const float* COMB, const float* FN, float* PART, int R, int D) {                                              \
        post_partial<ROWS_, VARIANT_>(X, XOUT, G0, rs, POST, COMB, FN, PART, R, D);                                   \
    }
TF_HC_FUSED(tf_dsv41_hc_post_partial, 8, 0)
TF_HC_FUSED(tf_dsv41_hc_post_partial_r16, 16, 0)
TF_HC_FUSED(tf_dsv41_hc_post_partial_v1, 8, 1)
TF_HC_FUSED(tf_dsv41_hc_post_partial_v2, 8, 2)
TF_HC_FUSED(tf_dsv41_hc_post_partial_v3, 8, 3)
#undef TF_HC_FUSED
