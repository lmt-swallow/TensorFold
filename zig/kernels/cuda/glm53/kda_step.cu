// kda.cu's step_kernel (the delta rule over a prompt's rows, after prep_kernel) with more rows' work in flight.
// Contract: for any row count, the read-outs y and the final state are byte-equal to step_kernel's on the same inputs
// (built with --fmad=false like kda.cu, so every product and sum below is its own rounding).
//
// step_kernel, for head h, value row v and row r:
//  - s[v][c] = s[v][c] * g[c] for every key column c;
//  - lane l's read p_l = (((0 + s[v][4l] k[4l]) + s[v][4l+1] k[4l+1]) + s[v][4l+2] k[4l+2]) + s[v][4l+3] k[4l+3], then
//    warp_sum's butterfly over the 32 lanes: + lane l ^ 16, then ^ 8, ^ 4, ^ 2, ^ 1;
//  - delta = (v_r - kv) beta_r and s[v][c] = s[v][c] + k[c] delta;
//  - the read-out o the same way from q, rounded to bf16.
// Here TPR threads share a value row: thread T holds lanes l = T + TPR m, adds the butterfly's levels over m's bits
// itself (l ^ 16 first) and the rest (T ^ TPR / 2 .. T ^ 1) through shuffles. Every partial sum is a sum of the same two
// operands as on the lanes (addition commutes), so the read-outs and the state are step_kernel's bits. PIPE runs row
// r's read-out and row r + 1's decay and read in one pass over the columns; each value still goes through the serial
// order (row r's update, its read-out, row r + 1's decay, read and update).

#include <cuda_bf16.h>

namespace {

constexpr int DK = 128, DV = 128;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

// VR value rows a program (TPR VR threads), TR rows a stage, two stages (the next one's k, q, g arrive by cp.async
// while this one's rows run).
template <int VR, int TPR, int TR, bool PIPE = false>
__device__ __forceinline__ void step_rows(int H, const float* __restrict__ state_in, const float* __restrict__ q_in,
                                          const float* __restrict__ k_save, const __nv_bfloat16* __restrict__ v_save,
                                          const float* __restrict__ g_save, const float* __restrict__ b_save, int rows,
                                          __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    constexpr int NT = VR * TPR, M = 32 / TPR;
    __shared__ __align__(16) float kt[2][TR][DK];
    __shared__ __align__(16) float qt[2][TR][DK];
    __shared__ __align__(16) float gt[2][TR][DK];
    __shared__ float vt[2][TR][VR];
    __shared__ float bt[2][TR];
    const int h = blockIdx.x, v0 = blockIdx.y * VR;
    const int T = threadIdx.x % TPR, vl = threadIdx.x / TPR, v = v0 + vl;
    const size_t sbase = ((size_t)h * DV + v) * DK;
    float s[M][4];   // columns 4 (T + TPR m) + i
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float4 x = *reinterpret_cast<const float4*>(state_in + sbase + 4 * (T + TPR * m));
        s[m][0] = x.x; s[m][1] = x.y; s[m][2] = x.z; s[m][3] = x.w;
    }
    auto read = [&](const float* x) {
        float p[M];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float4 c = *reinterpret_cast<const float4*>(x + 4 * (T + TPR * m));
            float a = 0.0f;
            a = a + s[m][0] * c.x;
            a = a + s[m][1] * c.y;
            a = a + s[m][2] * c.z;
            a = a + s[m][3] * c.w;
            p[m] = a;
        }
#pragma unroll
        for (int w = M / 2; w > 0; w >>= 1)            // lane bit log2(TPR) + log2(w): l ^ 16 first
#pragma unroll
            for (int m = 0; m < w; ++m) p[m] = p[m] + p[m + w];
        float r = p[0];
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
        return r;
    };
    // a stage: k, q, g rows by cp.async (16 bytes a copy), v and beta by plain loads (converted)
    auto stage = [&](int buf, int r0) {
        const int n = rows - r0 < TR ? rows - r0 : TR;
        for (int x = threadIdx.x; x < n * DK / 4; x += NT) {
            const int rr = x / (DK / 4), c = (x % (DK / 4)) * 4;
            const size_t base = ((size_t)(r0 + rr) * H + h) * DK + c;
            cp_async16(&kt[buf][rr][c], k_save + base);
            cp_async16(&qt[buf][rr][c], q_in + base);
            cp_async16(&gt[buf][rr][c], g_save + base);
        }
        cp_async_commit();
        for (int x = threadIdx.x; x < n * VR; x += NT) {
            const int rr = x / VR, c = x % VR;
            vt[buf][rr][c] = __bfloat162float(v_save[((size_t)(r0 + rr) * H + h) * DV + v0 + c]);
        }
        if (threadIdx.x < n) bt[buf][threadIdx.x] = b_save[(r0 + threadIdx.x) * H + h];
    };
    if (rows > 0) stage(0, 0);
    for (int r0 = 0, buf = 0; r0 < rows; r0 += TR, buf ^= 1) {
        const int n = rows - r0 < TR ? rows - r0 : TR;
        cp_async_wait<0>();
        __syncthreads();                               // this stage landed; every thread is past the other buffer
        if (r0 + TR < rows) stage(buf ^ 1, r0 + TR);
        if constexpr (PIPE) {
            // row rr's read-out with row rr + 1's decay and read in one pass over the columns (each still applied in
            // the serial order: rr's update, rr's read-out, rr + 1's decay, rr + 1's read, rr + 1's update)
            auto dot = [&](int m, const float* x) {
                const float4 c = *reinterpret_cast<const float4*>(x + 4 * (T + TPR * m));
                float a = 0.0f;
                a = a + s[m][0] * c.x;
                a = a + s[m][1] * c.y;
                a = a + s[m][2] * c.z;
                a = a + s[m][3] * c.w;
                return a;
            };
            auto decay = [&](int m, const float* gr) {
                const float4 g = *reinterpret_cast<const float4*>(gr + 4 * (T + TPR * m));
                s[m][0] = s[m][0] * g.x; s[m][1] = s[m][1] * g.y; s[m][2] = s[m][2] * g.z; s[m][3] = s[m][3] * g.w;
            };
            auto tree = [&](float (&p)[M]) {
#pragma unroll
                for (int w = M / 2; w > 0; w >>= 1)
#pragma unroll
                    for (int m = 0; m < w; ++m) p[m] = p[m] + p[m + w];
            };
            auto update = [&](int rr, float kv) {
                const float delta = (vt[buf][rr][vl] - kv) * bt[buf][rr];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const float4 k = *reinterpret_cast<const float4*>(&kt[buf][rr][4 * (T + TPR * m)]);
                    s[m][0] = s[m][0] + k.x * delta; s[m][1] = s[m][1] + k.y * delta;
                    s[m][2] = s[m][2] + k.z * delta; s[m][3] = s[m][3] + k.w * delta;
                }
            };
            if (n > 0) {
#pragma unroll
                for (int m = 0; m < M; ++m) decay(m, gt[buf][0]);
                update(0, read(kt[buf][0]));
            }
            for (int rr = 0; rr < n; ++rr) {
                float o;
                if (rr + 1 < n) {
                    float po[M], pk[M];
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        po[m] = dot(m, qt[buf][rr]);
                        decay(m, gt[buf][rr + 1]);
                        pk[m] = dot(m, kt[buf][rr + 1]);
                    }
                    tree(po);
                    tree(pk);
                    o = po[0];
                    float kv = pk[0];
#pragma unroll
                    for (int x = TPR / 2; x > 0; x >>= 1) {
                        o += __shfl_xor_sync(0xffffffffu, o, x);
                        kv += __shfl_xor_sync(0xffffffffu, kv, x);
                    }
                    if (T == 0) y_out[((size_t)(r0 + rr) * H + h) * DV + v] = __float2bfloat16_rn(o);
                    update(rr + 1, kv);
                } else {
                    o = read(qt[buf][rr]);
                    if (T == 0) y_out[((size_t)(r0 + rr) * H + h) * DV + v] = __float2bfloat16_rn(o);
                }
            }
            continue;
        }
        for (int rr = 0; rr < n; ++rr) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float4 g = *reinterpret_cast<const float4*>(&gt[buf][rr][4 * (T + TPR * m)]);
                s[m][0] = s[m][0] * g.x; s[m][1] = s[m][1] * g.y; s[m][2] = s[m][2] * g.z; s[m][3] = s[m][3] * g.w;
            }
            const float kv = read(kt[buf][rr]);
            const float delta = (vt[buf][rr][vl] - kv) * bt[buf][rr];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float4 k = *reinterpret_cast<const float4*>(&kt[buf][rr][4 * (T + TPR * m)]);
                s[m][0] = s[m][0] + k.x * delta; s[m][1] = s[m][1] + k.y * delta;
                s[m][2] = s[m][2] + k.z * delta; s[m][3] = s[m][3] + k.w * delta;
            }
            const float o = read(qt[buf][rr]);
            if (T == 0) y_out[((size_t)(r0 + rr) * H + h) * DV + v] = __float2bfloat16_rn(o);
        }
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int m = 0; m < M; ++m)
            *reinterpret_cast<float4*>(state_out + sbase + 4 * (T + TPR * m)) = make_float4(s[m][0], s[m][1], s[m][2], s[m][3]);
    }
}

}  // namespace

// step_kernel's arguments; grid (H, DV / 32), 128 threads (4 a value row), row r's read-out interleaved with row r + 1's
// decay and read.
extern "C" __global__ void __launch_bounds__(128) tf_glm53_kda_step_rows(
    int H, const float* state_in, const float* q_in, const float* k_save, const __nv_bfloat16* v_save,
    const float* g_save, const float* b_save, int rows, __nv_bfloat16* y_out, float* state_out) {
    step_rows<32, 4, 8, true>(H, state_in, q_in, k_save, v_save, g_save, b_save, rows, y_out, state_out);
}
