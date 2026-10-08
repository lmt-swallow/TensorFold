// EXL3 routed experts, any codebook and a width per expert: fixed-order splits, slots and butterflies, no atomics; 4-bit mcg matches GLM's kernel bit for bit.

// Zig build: the fork's kernels verbatim (bertholomus/TensorFold deepseek-v41-tp2, cuda/exl3/experts.cu), the torch
// extension's host glue dropped; namespace tf_exl3x_experts so the fatbin exports every kernel by name.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "experts_grouped.cuh"

namespace tf_exl3x_experts {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ members,
                                                              int R, int slots, int E, int maxm) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;                                   // exclusive per warp
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

// The same grouping for any number of rows (prompt chunks; group_kernel stages every pick in one block's shared memory):
// a block per expert counts its members, then a block per used expert takes its place in id order and lists its
// members in row order, -1 after the last.
constexpr int PLACE_THREADS = 256;

__device__ __forceinline__ int block_sum(int v, int* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    int s = 0;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += sh[w];
    return s;
}

__global__ void __launch_bounds__(PLACE_THREADS) group_count_kernel(const int* __restrict__ pick,
                                                                    int* __restrict__ counts, int n) {
    __shared__ int sh[PLACE_THREADS / 32];
    const int e = blockIdx.x;
    int c = 0;
    for (int i = threadIdx.x; i < n; i += PLACE_THREADS) c += pick[i] == e;
    c = block_sum(c, sh);
    if (threadIdx.x == 0) counts[e] = c;
}

__global__ void __launch_bounds__(PLACE_THREADS) group_place_kernel(const int* __restrict__ pick,
                                                                    const int* __restrict__ counts,
                                                                    int* __restrict__ uids, int* __restrict__ ucount,
                                                                    int* __restrict__ members, int n, int slots, int E,
                                                                    int maxm) {
    __shared__ int sh[PLACE_THREADS / 32];
    __shared__ int warp_tot[PLACE_THREADS / 32];
    const int e = blockIdx.x;
    int before = 0;                                   // used experts with a smaller id: this one's place
    for (int i = threadIdx.x; i < e; i += PLACE_THREADS) before += counts[i] > 0;
    const int u = block_sum(before, sh);
    if (e == 0) {
        int used = 0;
        for (int i = threadIdx.x; i < E; i += PLACE_THREADS) used += counts[i] > 0;
        used = block_sum(used, sh);
        if (threadIdx.x == 0) ucount[0] = used;
    }
    if (counts[e] == 0) return;
    if (threadIdx.x == 0) uids[u] = e;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int base = 0;
    for (int start = 0; start < n; start += PLACE_THREADS) {
        const int i = start + threadIdx.x;
        const int hit = i < n && pick[i] == e;
        int inc = hit;                                // inclusive scan of hits in thread order
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += v;
        }
        __syncthreads();
        if (lane == 31) warp_tot[warp] = inc;
        __syncthreads();
        int prior = 0, total = 0;
        for (int w = 0; w < PLACE_THREADS / 32; ++w) {
            prior += w < warp ? warp_tot[w] : 0;
            total += warp_tot[w];
        }
        const int j = base + prior + inc - hit;
        if (hit && j < maxm) members[u * maxm + j] = (i / slots) * 32 + (i % slots);
        base += total;
    }
    for (int j = base + threadIdx.x; j < maxm; j += PLACE_THREADS) members[u * maxm + j] = -1;
}

// Prompt chunks' work lists (experts.GROUP_LIST): the (expert place, member group) pairs of the used places, groups of
// rows0 (rows1) member rows, in place order and in group order within a place (the order the grids ran them in), then
// sentinel pairs (a place past every used one: its programs return at once) up to each list's length n0 (n1), a bound
// the host takes from the shape alone. One block; no host read of the lists' sizes.
constexpr int LIST_THREADS = 1024;
constexpr int LIST_NONE = 0x7fffffff;

__global__ void __launch_bounds__(LIST_THREADS) work_list_kernel(const int* __restrict__ counts,
                                                                 const int* __restrict__ uids,
                                                                 const int* __restrict__ ucount, int maxu,
                                                                 int* __restrict__ work0, int rows0, int n0,
                                                                 int* __restrict__ work1, int rows1, int n1) {
    __shared__ int warp_tot[LIST_THREADS / 32];
    const int nu = ucount[0];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int l = 0; l < (work1 ? 2 : 1); ++l) {
        int* work = l ? work1 : work0;
        const int rows = l ? rows1 : rows0, n = l ? n1 : n0;
        int base = 0;
        for (int start = 0; start < maxu; start += LIST_THREADS) {
            const int p = start + threadIdx.x;
            const int g = (p < maxu && p < nu) ? (counts[uids[p]] + rows - 1) / rows : 0;
            int inc = g;                              // inclusive scan of the groups in thread (= place) order
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, inc, o);
                if (lane >= o) inc += v;
            }
            if (lane == 31) warp_tot[warp] = inc;
            __syncthreads();
            int prior = 0, total = 0;
            for (int w = 0; w < LIST_THREADS / 32; ++w) {
                prior += w < warp ? warp_tot[w] : 0;
                total += warp_tot[w];
            }
            const int first = base + prior + inc - g;
            for (int j = 0; j < g; ++j) {
                const int q = first + j;
                if (q < n) {                          // (always: n bounds every list the counts allow)
                    work[2 * q] = p;
                    work[2 * q + 1] = j;
                }
            }
            base += total;
            __syncthreads();                          // warp_tot is rewritten by the next round
        }
        for (int q = base + threadIdx.x; q < n; q += LIST_THREADS) {
            work[2 * q] = LIST_NONE;
            work[2 * q + 1] = 0;
        }
    }
}

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32.
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int E) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits). With
// store_y the routed slots' outputs go to y and a non-routed slot's comes from it (the caller's); without (prompt
// chunks: nothing reads them after the combine) y is neither written nor read and a non-routed slot adds w * 0. With
// add (a [rows, D] fp32 term, e.g. the shared expert's output) each combined row gets it added last, the add the
// caller would make after this launch.
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, const float* __restrict__ add,
                                    float* __restrict__ out, int P, int D, int SK, int E, int slots, int store_y) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            if (store_y) y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = store_y ? y[(size_t)p * D + n + j] : 0.f;
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        out[(size_t)r * D + n + j] = add ? __fadd_rn(acc[j], add[(size_t)r * D + n + j]) : acc[j];
}

// out[r][d] = the sum over slots in order of the routed slots' weighted bf16 outputs Y (grouped_down3), fp32, then add.
__global__ void combine_y_kernel(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ pick,
                                 const float* __restrict__ add, float* __restrict__ out, int D, int slots, int E) {
    const int r = blockIdx.x;
    const int d = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (d >= D) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < slots; ++k) {
        const int e = pick[r * slots + k];
        if (e < 0 || e >= E) continue;
        const uint2 v = *reinterpret_cast<const uint2*>(Y + ((size_t)r * slots + k) * D + d);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        acc[0] += a.x; acc[1] += a.y; acc[2] += b.x; acc[3] += b.y;
    }
    float4 o;
    if (add) {
        const float4 s = *reinterpret_cast<const float4*>(add + (size_t)r * D + d);
        o = make_float4(__fadd_rn(acc[0], s.x), __fadd_rn(acc[1], s.y), __fadd_rn(acc[2], s.z), __fadd_rn(acc[3], s.w));
    } else {
        o = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + d) = o;
}

// Decode windows' first launch: block 0 groups the picks (group_kernel's output: distinct experts < E in id order,
// members row * 32 + slot in row order, -1 after the last), each pick in parallel: its expert's place is the number of
// distinct routed experts with a smaller id, its member position the number of earlier picks of the same expert; with
// wts it also combines the rows that route no slot (down_combine's arithmetic: their slots' y, or 0). The other blocks
// run rot_in_kernel's programs, a warp each (the same arithmetic).
constexpr int PREP_THREADS = 256;

template <typename TIN>
__global__ void __launch_bounds__(PREP_THREADS) decode_prep_kernel(
    const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick, const half* __restrict__ suh0,
    const half* __restrict__ suh1, half* __restrict__ out0, half* __restrict__ out1, int K, int slots, int E, int R,
    int* __restrict__ uids, int* __restrict__ ucount, int* __restrict__ members, int maxm,
    const float* __restrict__ wts, const float* __restrict__ y, const float* __restrict__ add, float* __restrict__ out,
    int D, int store_y, int* __restrict__ epoch) {
    const int n = R * slots;
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // gate/up (it waits for this grid's end)
    if (blockIdx.x == 0 && threadIdx.x == 0 && epoch != nullptr) epoch[0] += 1;     // this launch's number
    if (blockIdx.x == 0) {
        extern __shared__ int sh_prep[];
        int* sp = sh_prep;
        int* first = sh_prep + n;
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) sp[i] = pick[i];
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) {
            const int e = sp[i];
            int f = e >= 0 && e < E;
            for (int j = 0; f && j < i; ++j) f = sp[j] != e;
            first[i] = f;
        }
        __syncthreads();
        int used = 0;
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) {
            const int e = sp[i];
            if (e < 0 || e >= E) continue;
            int place = 0, pos = 0, tot = 0;
            for (int j = 0; j < n; ++j) {
                const int ej = sp[j];
                place += first[j] && ej < e;
                tot += ej == e;
                pos += ej == e && j < i;
            }
            if (pos < maxm) members[place * maxm + pos] = (i / slots) * 32 + (i % slots);
            if (first[i]) {
                uids[place] = e;
                for (int k = tot; k < maxm; ++k) members[place * maxm + k] = -1;
                ++used;
            }
        }
        __shared__ int used_sh;
        if (threadIdx.x == 0) used_sh = 0;
        __syncthreads();
        if (used) atomicAdd(&used_sh, used);
        __syncthreads();
        if (threadIdx.x == 0) ucount[0] = used_sh;
        if (wts == nullptr) return;
        for (int r = 0; r < R; ++r) {
            int routed = 0;
            for (int q = 0; q < slots; ++q) routed += sp[r * slots + q] >= 0 && sp[r * slots + q] < E;
            if (routed) continue;
            for (int d = threadIdx.x * 4; d < D; d += PREP_THREADS * 4) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                for (int q = 0; q < slots; ++q) {
                    const float w = wts[r * slots + q];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc[j] = fmaf(w, store_y ? y[((size_t)r * slots + q) * D + d + j] : 0.f, acc[j]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    out[(size_t)r * D + d + j] = add ? __fadd_rn(acc[j], add[(size_t)r * D + d + j]) : acc[j];
            }
        }
        return;
    }
    // rot_in_kernel's program (member row p, 128-block blk, matrix mat), one a warp
    const int nblk = K / 128;
    const int item = (blockIdx.x - 1) * (PREP_THREADS / 32) + (threadIdx.x >> 5);
    if (item >= n * nblk * 2) return;
    const int p = item / (nblk * 2), mat = (item / nblk) % 2, blk = item % nblk;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x & 31;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

template __global__ void rot_in_kernel<__nv_bfloat16>(const __nv_bfloat16*, int, const int*, const half*, const half*, half*, half*, int, int, int);
template __global__ void rot_in_kernel<half>(const half*, int, const int*, const half*, const half*, half*, half*, int, int, int);
template __global__ void __launch_bounds__(PREP_THREADS) decode_prep_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, int, const int*, const half*, const half*, half*, half*, int, int, int, int, int*, int*, int*, int,
    const float*, const float*, const float*, float*, int, int, int*);
template __global__ void __launch_bounds__(PREP_THREADS) decode_prep_kernel<half>(
    const half*, int, const int*, const half*, const half*, half*, half*, int, int, int, int, int*, int*, int*, int,
    const float*, const float*, const float*, float*, int, int, int*);

}  // namespace tf_exl3x_experts
