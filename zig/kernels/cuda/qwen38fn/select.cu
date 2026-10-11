// Ours: attention.py `_select` (each sparse row's TOP blocks of largest score, ties to the lower
// block id, listed in block order, then the unfinished tail; a dense row keeps only its length) for a prompt block's
// rows, one 256-thread program a row. The lists are integers fixed by the scores, so any exact method gives
// `_select`'s bytes; this one finds the cut as `_select_tiles` does (the TOP-th largest order-preserving key, a byte a
// pass from the top) with shared-memory histograms instead of `_select`'s 33 full-row reductions. cuda_prompt.zig uses
// it up to 16,384 blocks (65,536 keys); wider, 0.6.6's `_select` and `_select_tiles` are faster.

#include <stdint.h>

namespace tf_qfn_sel {

constexpr int THREADS = 256;

__device__ __forceinline__ uint32_t key(float v) {
    const uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Exclusive prefix of x over the block (THREADS threads), and the block's total.
__device__ __forceinline__ int block_scan(int x, int& total, int* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) sh[warp] = inc;
    __syncthreads();
    int before = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) {
        before += w < warp ? sh[w] : 0;
        total += sh[w];
    }
    __syncthreads();
    return before + inc - x;
}

}  // namespace tf_qfn_sel

// Row r of rows (grid.x): its list into IDS [rows, IDW], NKR [rows], SPR [rows] from SC [rows, NB] fp32, as
// attention.py `_launch_select` lists it for RATIO and TOP.
extern "C" __global__ void __launch_bounds__(256) tf_qfn_select_rows(const float* __restrict__ SC,
                                                                     const int* __restrict__ POS0, int* __restrict__ IDS,
                                                                     int* __restrict__ NKR, int* __restrict__ SPR, int NB,
                                                                     int RATIO, int TOP, int IDW) {
    using namespace tf_qfn_sel;
    __shared__ int hist[256];
    __shared__ int sh[THREADS / 32];
    __shared__ uint32_t cut_sh;
    __shared__ int need_sh;
    const int r = blockIdx.x;
    const int end = POS0[0] + r + 1;
    const int complete = end / RATIO;
    if (complete <= TOP) {
        if (threadIdx.x == 0) {
            NKR[r] = end;
            SPR[r] = 0;
        }
        return;
    }
    const float* row = SC + (size_t)r * NB;
    uint32_t cut = 0;
    int need = TOP;  // the cut's rank among the keys sharing its leading bytes
    for (int p = 0; p < 4; ++p) {
        const int shift = 24 - 8 * p;
        hist[threadIdx.x] = 0;
        __syncthreads();
        for (int b = threadIdx.x; b < complete; b += THREADS) {
            const uint32_t k = key(row[b]);
            if (p == 0 || (k >> (shift + 8)) == (cut >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 0xFF], 1);
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            // the largest digit d whose keys with digits >= d number at least `need` (one warp, from the top)
            int above = 0, d = -1;
            for (int hi = 255; hi >= 0 && d < 0; hi -= 32) {
                const int h = hist[hi - threadIdx.x];
                int inc = h;  // keys with digits in [hi - lane, hi]
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int v = __shfl_up_sync(0xffffffffu, inc, o);
                    if ((int)threadIdx.x >= o) inc += v;
                }
                const unsigned hit = __ballot_sync(0xffffffffu, above + inc >= need);
                if (hit) {
                    const int l = __ffs(hit) - 1;
                    const int before = __shfl_sync(0xffffffffu, inc - h, l);
                    d = hi - l;
                    if (threadIdx.x == 0) {
                        need_sh = need - (above + before);
                        cut_sh = cut | ((uint32_t)d << shift);
                    }
                } else {
                    above += __shfl_sync(0xffffffffu, inc, 31);
                }
            }
        }
        __syncthreads();
        cut = cut_sh;
        need = need_sh;
        __syncthreads();
    }
    // every key above the cut, then the first `need` keys equal to it, in block order
    int seen = 0, placed = 0;
    for (int b0 = 0; b0 < complete && placed < TOP; b0 += THREADS) {
        const int b = b0 + threadIdx.x;
        const uint32_t k = b < complete ? key(row[b]) : 0u;
        const bool ok = b < complete;
        const int eq = ok && k == cut;
        int teq, tch;
        const int req = seen + block_scan(eq, teq, sh) + eq;  // this key's rank among the equal ones
        const bool chosen = ok && (k > cut || (eq && req <= need));
        const int at = placed + block_scan(chosen, tch, sh);
        if (chosen)
            for (int q = 0; q < RATIO; ++q) IDS[(size_t)r * IDW + at * RATIO + q] = b * RATIO + q;
        seen += teq;
        placed += tch;
    }
    if (threadIdx.x < RATIO) {
        const int tail = RATIO * complete + threadIdx.x;
        if (tail < end) IDS[(size_t)r * IDW + TOP * RATIO + threadIdx.x] = tail;
    }
    if (threadIdx.x == 0) {
        NKR[r] = TOP * RATIO + end - RATIO * complete;
        SPR[r] = 1;
    }
}

// Rows' lists that differ between (IDS, NKR, SPR) and (IDS2, NKR2, SPR2): a row counts when its length or sparse
// flag differs or any of its first NKR entries (sparse rows) does; added into *count.
extern "C" __global__ void tf_qfn_select_diff(const int* __restrict__ IDS, const int* __restrict__ NKR,
                                              const int* __restrict__ SPR, const int* __restrict__ IDS2,
                                              const int* __restrict__ NKR2, const int* __restrict__ SPR2, int rows,
                                              int IDW, unsigned long long* __restrict__ count) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ int bad;
    if (threadIdx.x == 0) bad = NKR[r] != NKR2[r] || SPR[r] != SPR2[r];
    __syncthreads();
    if (!bad && SPR[r])
        for (int i = threadIdx.x; i < NKR[r]; i += blockDim.x)
            if (IDS[(size_t)r * IDW + i] != IDS2[(size_t)r * IDW + i]) bad = 1;
    __syncthreads();
    if (threadIdx.x == 0 && bad) atomicAdd(count, 1ull);
}
