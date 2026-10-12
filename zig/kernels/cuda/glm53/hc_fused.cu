// glue.py's `_hc_post` (the residual streams after a block) and the next `hc_pre`'s `_hc_partial_mma` (its mixing dots and
// sum of squares over those streams) in one pass, so the streams written are not read back: prompt rows, S = 4 streams
// of D = 4096, WORLD = 2 rank partials, NB = 16 K blocks of 1,024, 24 mixing rows.
//
// What the captured kernels compute (recovered from their SASS), and what this kernel computes the same way:
//  - _hc_post, per row r, column d and stream s: branch = bf16(G0 + G1) (rank 0's partial first); mixed =
//    fma(c3 x3, fma(c2 x2, fma(c0 x0, c1 x1))) with c_k = COMB[r, k, s] and x_k the bf16 stream k at d (c1 x1 its own
//    rounding, the other three fused); out = bf16(fma(branch, POST[r, s], mixed));
//  - _hc_partial_mma, per row r and K block b (columns 1,024 b ..; stream b / 4): the 24 dots as one mma.m16n8k16 chain
//    (bf16 in, fp32 accumulate) from +0 over the block's 64-column tiles in order, 16 columns a step in order; and the
//    sum of squares ss = ((0 + t_0) + t_1) + ... over the tiles, each tile's t = the butterfly (+ chunk c ^ 4, then ^ 2,
//    ^ 1) of its eight 8-column chunks' P_c = fma(x7, x7, ... fma(x2, x2, fma(x0, x0, x1 x1))) (x_i column 8 c + i of
//    the tile, x1 x1 its own rounding).
// The partial reads the streams _hc_post wrote, as bf16: here they are the same bf16 values, taken before they are
// stored. Neither the tile order inside a block nor anything else of the arithmetic changes, so XOUT and PART are the
// captured kernels' bytes. Verified against the captured pair on rows whose values span 2^-12 .. 2^12 (where a plain
// sequential sum of squares differs from the captured one in 95 % of blocks) at 2,048, 777, 64 and 1 rows: XOUT, the
// dots and the sums of squares byte-equal; and in the model by TF_GLM_PROMPT_CHECK.

#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int D = 4096, S = 4, NB = 16, KB = D * S / NB, M24 = 24;
constexpr int BM = 64, BC = 64;              // rows a program, columns a tile
constexpr int TLD = BC + 8;                  // halfs a staged row (padded: ldmatrix rows in distinct banks)

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

__device__ __forceinline__ void unpack8(const uint4 v, float (&x)[8]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[2 * i] = __uint_as_float(w[i] << 16);
        x[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
}

}  // namespace

// grid (ceil(R / 64), D / KB = 4), 128 threads, dynamic shared memory 4 (64 + 32) (64 + 8) 2 bytes. X and XOUT [R, S D]
// bf16 (may be the same), G0 [R, D] fp32 and G1 = G0 + rs (elements, any sign), POST [R, S], COMB [R, S, S] fp32,
// FN [24, S D] bf16, PART [R, NB, 32] fp32 (columns 0..23 the dots, 24 the sum of squares).
extern "C" __global__ void __launch_bounds__(128) tf_glm53_hc_post_partial(
    const __nv_bfloat16* X, __nv_bfloat16* XOUT, const float* __restrict__ G0, long long rs,
    const float* __restrict__ POST, const float* __restrict__ COMB, const __nv_bfloat16* __restrict__ FN,
    float* __restrict__ PART, int R) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* T = reinterpret_cast<__nv_bfloat16*>(smem_raw);              // [S][BM][TLD]: the tile's outputs
    __nv_bfloat16* F = T + S * BM * TLD;                                         // [S][32][TLD]: FN rows (24 + zeros)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int r0 = blockIdx.x * BM, db = blockIdx.y;
    const float* G1 = G0 + rs;

    // elementwise part: pair p = tid + 128 i (i < 4) is (row p / 8, chunk p % 8); a lane's chunk is lane % 8
    float ss[4][S];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int s = 0; s < S; ++s) ss[i][s] = 0.f;
    float acc[S][4][4];   // per stream: the warp's 16 rows x 32 mixing columns (n8 blocks 0..3)
#pragma unroll
    for (int s = 0; s < S; ++s)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[s][j][c] = 0.f;
    // FN rows 24..31: zero once
    for (int x = tid; x < S * 8 * TLD; x += 128) {
        const int s = x / (8 * TLD), rem = x % (8 * TLD);
        F[(size_t)s * 32 * TLD + (M24 + rem / TLD) * TLD + rem % TLD] = __float2bfloat16_rn(0.f);
    }

    for (int tile = 0; tile < KB / BC; ++tile) {
        const int col = db * KB + tile * BC;                                     // within a stream
        __syncthreads();                                                         // T and F free
        // FN: rows 0..23 of K block s * 4 + db (columns s D + col ..): 24 x 64 halfs a stream, 16 bytes a copy
        for (int x = tid; x < S * M24 * (BC / 8); x += 128) {
            const int s = x / (M24 * (BC / 8)), rem = x % (M24 * (BC / 8)), m = rem / (BC / 8), part = rem % (BC / 8);
            *reinterpret_cast<uint4*>(F + (size_t)s * 32 * TLD + m * TLD + part * 8) =
                *reinterpret_cast<const uint4*>(FN + (size_t)m * (S * D) + s * D + col + part * 8);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int p = tid + 128 * i, rl = p / 8, chunk = p % 8, r = r0 + rl;
            const int d = col + chunk * 8;
            float o[S][8];
            if (r < R) {
                float xs[S][8], b[8];
                {
                    const float4 a0 = *reinterpret_cast<const float4*>(G0 + (size_t)r * D + d);
                    const float4 a1 = *reinterpret_cast<const float4*>(G0 + (size_t)r * D + d + 4);
                    const float4 b0 = *reinterpret_cast<const float4*>(G1 + (size_t)r * D + d);
                    const float4 b1 = *reinterpret_cast<const float4*>(G1 + (size_t)r * D + d + 4);
                    const float ga[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    const float gb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                    for (int j = 0; j < 8; ++j) b[j] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(ga[j], gb[j])));
                }
#pragma unroll
                for (int k = 0; k < S; ++k) unpack8(*reinterpret_cast<const uint4*>(X + (size_t)r * (S * D) + k * D + d), xs[k]);
#pragma unroll
                for (int s = 0; s < S; ++s) {
                    const float* cm = COMB + (size_t)r * S * S;
                    const float c0 = cm[0 * S + s], c1 = cm[1 * S + s], c2 = cm[2 * S + s], c3 = cm[3 * S + s];
                    const float ps = POST[(size_t)r * S + s];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float mixed = __fmul_rn(c1, xs[1][j]);
                        mixed = __fmaf_rn(c0, xs[0][j], mixed);
                        mixed = __fmaf_rn(c2, xs[2][j], mixed);
                        mixed = __fmaf_rn(c3, xs[3][j], mixed);
                        o[s][j] = __bfloat162float(__float2bfloat16_rn(__fmaf_rn(b[j], ps, mixed)));
                    }
                }
            } else {
#pragma unroll
                for (int s = 0; s < S; ++s)
#pragma unroll
                    for (int j = 0; j < 8; ++j) o[s][j] = 0.f;
            }
            // XOUT, the staged tile, and this tile's sum of squares
#pragma unroll
            for (int s = 0; s < S; ++s) {
                uint32_t w[4];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    w[q] = (__float_as_uint(o[s][2 * q]) >> 16) | (__float_as_uint(o[s][2 * q + 1]) & 0xFFFF0000u);
                const uint4 v = make_uint4(w[0], w[1], w[2], w[3]);
                if (r < R) *reinterpret_cast<uint4*>(XOUT + (size_t)r * (S * D) + s * D + d) = v;
                *reinterpret_cast<uint4*>(T + (size_t)s * BM * TLD + rl * TLD + chunk * 8) = v;
                float pc = __fmul_rn(o[s][1], o[s][1]);
                pc = __fmaf_rn(o[s][0], o[s][0], pc);
#pragma unroll
                for (int j = 2; j < 8; ++j) pc = __fmaf_rn(o[s][j], o[s][j], pc);
                pc = __fadd_rn(pc, __shfl_xor_sync(0xffffffffu, pc, 4));
                pc = __fadd_rn(pc, __shfl_xor_sync(0xffffffffu, pc, 2));
                pc = __fadd_rn(pc, __shfl_xor_sync(0xffffffffu, pc, 1));
                ss[i][s] = __fadd_rn(pc, ss[i][s]);
            }
        }
        __syncthreads();                                                         // T and F written
        // the dots: warp w's rows 16 w .. + 15 against the 32 FN rows, 4 k16 steps in order, per stream
        const int a_row = 16 * warp + (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
        const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const __nv_bfloat16* ts = T + (size_t)s * BM * TLD;
            const __nv_bfloat16* fs = F + (size_t)s * 32 * TLD;
#pragma unroll
            for (int k16 = 0; k16 < BC / 16; ++k16) {
                uint32_t a[4];
                ldsm4(a, ts + a_row * TLD + k16 * 16 + a_col);
#pragma unroll
                for (int j = 0; j < 4; j += 2) {
                    uint32_t b[4];
                    ldsm4(b, fs + (j * 8 + b_row) * TLD + k16 * 16 + b_col);
                    mma16816(acc[s][j], a, b[0], b[1]);
                    mma16816(acc[s][j + 1], a, b[2], b[3]);
                }
            }
        }
    }

    // PART: the dots (rows g and g + 8 of the warp's 16, columns 8 j + 2 t and + 1) and the sums of squares
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const int b = s * (NB / S) + db;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = r0 + 16 * warp + g + 8 * h, m = 8 * j + 2 * t4;
                if (r >= R || m >= M24) continue;
                float* dst = PART + ((size_t)r * NB + b) * 32 + m;
                dst[0] = acc[s][j][2 * h];
                dst[1] = acc[s][j][2 * h + 1];
            }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int p = tid + 128 * i, r = r0 + p / 8;
        if (r < R && p % 8 == 0)
#pragma unroll
            for (int s = 0; s < S; ++s) PART[((size_t)r * NB + s * (NB / S) + db) * 32 + M24] = ss[i][s];
    }
}
