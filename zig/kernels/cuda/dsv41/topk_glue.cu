// The DSV4.1 indexer's selection glue on the device: kernels.topk_indices (each row's k highest scores' indices,
// ascending, ties to the lower index, then `torch.where(top < vis, top, -1)`), model.py `_candidates` (the candidate
// pool's block mask: the blocks with the largest maxima, the newest block pinned in, -inf blocks left out) and
// `apply_candidates` (-inf outside the pool's blocks), with the fork's total order: a score's key is its fp32 bits
// in total order (-0 as +0) above the inverted index, so the k largest keys are one set whatever the algorithm, and
// every row's selection is its own (one block a row).
//
// The selection: a radix select of the k-th largest 32-bit ordered score (four 8-bit passes over the row), then one
// pass in index order taking every score above it and the lowest-indexed scores equal to it (block scans), which
// writes the picks already ascending.
#include <stdint.h>

namespace {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;

// The score's key without its index: torch's `where(bits < 0, bits ^ 0x7FFFFFFF, bits)` on (score + 0.0)'s bits, as
// an unsigned word in the same order.
__device__ __forceinline__ uint32_t ordered(float v) {
    uint32_t b = __float_as_uint(v);
    if (b == 0x80000000u) b = 0u;  // + 0.0: -0 becomes +0
    int32_t s = (int32_t)b;
    s = s < 0 ? (s ^ 0x7FFFFFFF) : s;
    return (uint32_t)s ^ 0x80000000u;
}

// The same word from a key the score kernels wrote (KEYS: the ordered bits above the inverted index).
__device__ __forceinline__ uint32_t ordered(long long key) {
    return (uint32_t)((unsigned long long)key >> 32) ^ 0x80000000u;
}

// Exclusive block-wide prefix sum of `v` over the block's threads; `total` gets the sum. Uses `scratch` [kWarps + 1].
__device__ __forceinline__ uint32_t block_exclusive(uint32_t v, uint32_t* scratch, uint32_t& total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    uint32_t x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint32_t w = lane < kWarps ? scratch[lane] : 0u;
        uint32_t s = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            uint32_t y = __shfl_up_sync(0xFFFFFFFFu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < kWarps) scratch[lane] = s - w;  // exclusive warp offsets
        if (lane == kWarps - 1) scratch[kWarps] = s;
    }
    __syncthreads();
    const uint32_t out = scratch[warp] + x - v;
    total = scratch[kWarps];
    __syncthreads();
    return out;
}

// The k-th largest ordered key of row[0..n) (k >= 1, k <= n) and how many entries equal to it are taken (the rest
// lie above it).
template <typename T>
__device__ void radix_threshold(const T* row, int n, int k, uint32_t& thr, uint32_t& need) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_remaining;
    if (threadIdx.x == 0) {
        s_prefix = 0u;
        s_remaining = (uint32_t)k;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        const uint32_t high = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const uint32_t u = ordered(row[j]);
            if ((u & high) == (prefix & high)) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t rem = s_remaining;
            uint32_t above = 0u;
            int d = 255;
            for (; d > 0; --d) {
                if (above + hist[d] >= rem) break;
                above += hist[d];
            }
            s_remaining = rem - above;
            s_prefix = prefix | ((uint32_t)d << shift);
        }
        __syncthreads();
    }
    thr = s_prefix;
    need = s_remaining;
}

// One row's k picks (k <= n): its k largest keys' indices ascending, -1 at or past v.
template <typename T>
__device__ void topk_row(const T* row, int n, int k, long long v, long long* out) {
    __shared__ uint32_t scan[kWarps + 1];
    uint32_t thr, need;
    radix_threshold(row, n, k, thr, need);
    uint32_t eq_seen = 0u, taken = 0u;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        uint32_t u = 0u;
        if (j < n) u = ordered(row[j]);
        const uint32_t eq = (j < n && u == thr) ? 1u : 0u;
        uint32_t eq_total;
        const uint32_t eq_before = block_exclusive(eq, scan, eq_total);
        const uint32_t sel = (j < n && (u > thr || (eq && eq_seen + eq_before < need))) ? 1u : 0u;
        uint32_t sel_total;
        const uint32_t at = block_exclusive(sel, scan, sel_total);
        if (sel) out[taken + at] = (long long)j < v ? (long long)j : -1LL;
        eq_seen += eq_total;
        taken += sel_total;
    }
}

// hist[bin] += 1 for every lane of the warp with `live`, one shared atomic a distinct bin (the keys of a row share
// their top digits: per-lane atomics on one bin would serialize the whole row). Every lane of the warp calls it (the
// callers' loops are uniform over the block).
__device__ __forceinline__ void hist_add(uint32_t* hist, uint32_t bin, bool live) {
    const unsigned same = __match_any_sync(0xFFFFFFFFu, live ? bin : 0xFFFFFFFFu);
    if (live && (threadIdx.x & 31) == __ffs(same) - 1) atomicAdd(&hist[bin], (uint32_t)__popc(same));
}

// Loads a sweep issues together a thread (the sweeps are latency-bound: one row a block).
constexpr int kUnroll = 8;

// The k-th largest ordered key of row[0..n) as radix_threshold, with hist_add's warp-aggregated counts.
template <typename T>
__device__ void radix_threshold_agg(const T* row, int n, int k, uint32_t& thr, uint32_t& need) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_remaining;
    if (threadIdx.x == 0) {
        s_prefix = 0u;
        s_remaining = (uint32_t)k;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        const uint32_t high = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int base = 0; base < n; base += kUnroll * blockDim.x) {
            T v[kUnroll];
#pragma unroll
            for (int t = 0; t < kUnroll; ++t) {
                const int j = base + t * blockDim.x + threadIdx.x;
                v[t] = j < n ? row[j] : T(0);
            }
#pragma unroll
            for (int t = 0; t < kUnroll; ++t) {
                const int j = base + t * blockDim.x + threadIdx.x;
                const uint32_t u = ordered(v[t]);
                hist_add(hist, (u >> shift) & 255u, j < n && (u & high) == (prefix & high));
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t rem = s_remaining;
            uint32_t above = 0u;
            int d = 255;
            for (; d > 0; --d) {
                if (above + hist[d] >= rem) break;
                above += hist[d];
            }
            s_remaining = rem - above;
            s_prefix = prefix | ((uint32_t)d << shift);
        }
        __syncthreads();
    }
    thr = s_prefix;
    need = s_remaining;
}

// kernels.topk_select's picks a row at most (tf_dsv41_topk_keyset's shared sort).
constexpr int kMaxPicks = 1024;

// An int64 key as an unsigned word in the same order.
__device__ __forceinline__ unsigned long long key_order(long long k) {
    return (unsigned long long)k ^ 0x8000000000000000ull;
}

// The pool's mask from the block maxima s [rows, nblk] (row stride ld): block b of row r is in it when it is among
// the k blocks with the largest keys (k <= nblk) and its maximum is above -inf. BITS false: mask u8 [rows, nblk],
// mask[r, b] = 1 or 0 (row stride mask_ld bytes); BITS true: bit b % 32 of word b / 32 of row r, uint32
// [rows, ceil(nblk / 32)] (row stride mask_ld words; the last word's bits past nblk 0). One block a row (kThreads).
template <bool BITS>
__device__ __forceinline__ void cand_mask_row(const float* __restrict__ s, long long ld, int nblk, int k,
                                              void* __restrict__ mask, long long mask_ld) {
    __shared__ uint32_t scan[kWarps + 1];
    const int r = blockIdx.x;
    const float* row = s + (size_t)r * ld;
    uint32_t thr, need;
    radix_threshold(row, nblk, k, thr, need);
    uint32_t eq_seen = 0u;
    for (int base = 0; base < nblk; base += blockDim.x) {
        const int j = base + threadIdx.x;
        uint32_t u = 0u;
        if (j < nblk) u = ordered(row[j]);
        const uint32_t eq = (j < nblk && u == thr) ? 1u : 0u;
        uint32_t eq_total;
        const uint32_t eq_before = block_exclusive(eq, scan, eq_total);
        const bool sel = j < nblk && (u > thr || (eq && eq_seen + eq_before < need));
        // `s.gather(-1, idx) > -inf`: NaN maxima compare false as well
        const float x = j < nblk ? row[j] : 0.0f;
        const bool in = sel && x > -__int_as_float(0x7F800000);
        if (BITS) {
            // a warp's 32 consecutive blocks are one word (base and the warp's first lane are multiples of 32)
            const unsigned word = __ballot_sync(0xFFFFFFFFu, in);
            if ((threadIdx.x & 31) == 0 && j < nblk) ((uint32_t*)mask)[(size_t)r * mask_ld + j / 32] = word;
        } else if (j < nblk) {
            ((unsigned char*)mask)[(size_t)r * mask_ld + j] = in ? 1 : 0;
        }
        eq_seen += eq_total;
    }
}

}  // namespace

extern "C" {

// Rows of fp32 scores [rows, n] (row stride ld) -> each row's k picks (k <= n) into out [rows, k] (row stride
// out_ld): the indices of its k highest scores ascending, -1 at or past vis[r] (vis null: no mask). One block a row
// (kThreads threads).
__global__ void __launch_bounds__(kThreads) tf_dsv41_topk_rows(const float* __restrict__ score, long long ld, int n,
                                                               int k, const long long* __restrict__ vis,
                                                               long long* __restrict__ out, long long out_ld) {
    const int r = blockIdx.x;
    topk_row(score + (size_t)r * ld, n, k, vis ? vis[r] : (long long)n, out + (size_t)r * out_ld);
}

// tf_dsv41_topk_rows on the int64 keys the score kernels write with KEYS (kernels.topk_select's input): the same
// order, so the same picks as from the scores those kernels computed.
__global__ void __launch_bounds__(kThreads) tf_dsv41_topk_keys(const long long* __restrict__ keys, long long ld, int n,
                                                               int k, const long long* __restrict__ vis,
                                                               long long* __restrict__ out, long long out_ld) {
    const int r = blockIdx.x;
    topk_row(keys + (size_t)r * ld, n, k, vis ? vis[r] : (long long)n, out + (size_t)r * out_ld);
}

// model.py _candidates' block maxima: s[r, b] = max of score[r, b * bsize .. (b + 1) * bsize) (-inf past n, NaN
// propagating as torch's amax), +inf at the block holding the row's newest visible entry ((vis[r] - 1) // bsize,
// floor division: none when vis is 0). out [rows, nblk] (row stride out_ld), nblk = ceil(n / bsize).
__global__ void tf_dsv41_cand_blockmax(const float* __restrict__ score, long long ld, int n,
                                       const long long* __restrict__ vis, int bsize, float* __restrict__ out,
                                       long long out_ld, int nblk) {
    const int r = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const float* row = score + (size_t)r * ld;
    float m = -__int_as_float(0x7F800000);
    bool nan = false;
    for (int t = b * bsize; t < (b + 1) * bsize && t < n; ++t) {
        const float x = row[t];
        if (x != x) nan = true;
        else if (x > m) m = x;
    }
    if (nan) m = __int_as_float(0x7FC00000);
    const long long v = vis[r] - 1;
    const long long last = v >= 0 ? v / bsize : -1;  // floor division of vis - 1 >= -1
    if ((long long)b == last) m = __int_as_float(0x7F800000);
    out[(size_t)r * out_ld + b] = m;
}

// model.py _candidates' mask (cand_mask_row<false>) and the same pool as bits (cand_mask_row<true>).
__global__ void __launch_bounds__(kThreads) tf_dsv41_cand_mask(const float* __restrict__ s, long long ld, int nblk,
                                                               int k, unsigned char* __restrict__ mask,
                                                               long long mask_ld) {
    cand_mask_row<false>(s, ld, nblk, k, mask, mask_ld);
}

__global__ void __launch_bounds__(kThreads) tf_dsv41_cand_bits(const float* __restrict__ s, long long ld, int nblk,
                                                               int k, uint32_t* __restrict__ bits, long long bits_ld) {
    cand_mask_row<true>(s, ld, nblk, k, bits, bits_ld);
}

// kernels.topk_select's tail where torch.topk runs (rows of more than TOPK_FUSED_MAX keys, or keys not in position
// order: the pruned selection's gathered tiles, the candidate pool's keys): each row's k largest int64 keys (k <= n,
// k <= kMaxPicks), their indices (0xFFFFFFFF - the key's low word) ascending, -1 at or past vis[r] (vis null: no mask).
// The keys are unique but for PAD_KEY (int64 min, below every key a score makes), whose index is past any vis: so the
// output is the one torch.topk + _topk_finish write. A radix select of the k-th largest key (eight 8-bit passes), the
// picks gathered in shared memory, a bitonic sort.
__global__ void __launch_bounds__(kThreads) tf_dsv41_topk_keyset(const long long* __restrict__ keys, long long ld,
                                                                 int n, int k, const long long* __restrict__ vis,
                                                                 long long* __restrict__ out, long long out_ld) {
    __shared__ uint32_t hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ uint32_t s_remaining;
    __shared__ long long picks[kMaxPicks];
    __shared__ int s_taken, s_eq;
    const int r = blockIdx.x;
    const long long* row = keys + (size_t)r * ld;
    if (threadIdx.x == 0) {
        s_prefix = 0ull;
        s_remaining = (uint32_t)k;
        s_taken = 0;
        s_eq = 0;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const unsigned long long prefix = s_prefix;
        const unsigned long long high = shift == 56 ? 0ull : (~0ull << (shift + 8));
        for (int base = 0; base < n; base += kUnroll * blockDim.x) {
            unsigned long long v[kUnroll];
#pragma unroll
            for (int t = 0; t < kUnroll; ++t) {
                const int j = base + t * blockDim.x + threadIdx.x;
                v[t] = j < n ? key_order(row[j]) : 0ull;
            }
#pragma unroll
            for (int t = 0; t < kUnroll; ++t) {
                const int j = base + t * blockDim.x + threadIdx.x;
                hist_add(hist, (uint32_t)(v[t] >> shift) & 255u, j < n && (v[t] & high) == (prefix & high));
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t rem = s_remaining;
            uint32_t above = 0u;
            int d = 255;
            for (; d > 0; --d) {
                if (above + hist[d] >= rem) break;
                above += hist[d];
            }
            s_remaining = rem - above;
            s_prefix = prefix | ((unsigned long long)d << shift);
        }
        __syncthreads();
    }
    const unsigned long long thr = s_prefix;
    const int need = (int)s_remaining;
    for (int base = 0; base < n; base += kUnroll * blockDim.x) {
        long long v[kUnroll];
#pragma unroll
        for (int t = 0; t < kUnroll; ++t) {
            const int j = base + t * blockDim.x + threadIdx.x;
            v[t] = j < n ? row[j] : (long long)0x8000000000000000ull;
        }
#pragma unroll
        for (int t = 0; t < kUnroll; ++t) {
            const int j = base + t * blockDim.x + threadIdx.x;
            const unsigned long long u = key_order(v[t]);
            bool take = j < n && u > thr;
            if (!take && j < n && u == thr) take = atomicAdd(&s_eq, 1) < need;
            if (take) picks[atomicAdd(&s_taken, 1)] = 0xFFFFFFFFll - (v[t] & 0xFFFFFFFFll);
        }
    }
    int p2 = 1;
    while (p2 < k) p2 <<= 1;
    __syncthreads();
    for (int i = k + threadIdx.x; i < p2; i += blockDim.x) picks[i] = 0x7FFFFFFFFFFFFFFFll;
    __syncthreads();
    for (int size = 2; size <= p2; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < p2; i += blockDim.x) {
                const int partner = i ^ stride;
                if (partner > i) {
                    const bool up = (i & size) == 0;
                    const long long a = picks[i], b = picks[partner];
                    if ((a > b) == up) {
                        picks[i] = b;
                        picks[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    const long long v = vis ? vis[r] : 0x7FFFFFFFFFFFFFFFll;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const long long idx = picks[i];
        out[(size_t)r * out_ld + i] = idx < v ? idx : -1LL;
    }
}

// rounds.py _candidate_blocks from the block maxima s [rows, nblk] (row stride ld; tf_dsv41_cand_blockmax's, the
// newest block pinned at +inf): the k blocks with the largest keys (k <= nblk; ties to the lower block), ascending,
// each block's id or -1 where its maximum is not above -inf, into cblk [rows, k] int32 (row stride cblk_ld): the pool
// tf_dsv41_cand_mask marks, as the block ids index_keys_cand reads. One block a row (kThreads).
__global__ void __launch_bounds__(kThreads) tf_dsv41_cand_blocks(const float* __restrict__ s, long long ld, int nblk,
                                                                 int k, int* __restrict__ cblk, long long cblk_ld) {
    __shared__ uint32_t w_eq[kWarps], w_sel[kWarps];
    const int r = blockIdx.x;
    const float* row = s + (size_t)r * ld;
    uint32_t thr, need;
    radix_threshold_agg(row, nblk, k, thr, need);
    // each warp a contiguous segment (lanes coalesced), so the warps' counts in warp order order the picks by block
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned below = (1u << lane) - 1u;
    const int seg = (nblk + kWarps * 32 - 1) / (kWarps * 32) * 32;
    const int lo = min(nblk, warp * seg), hi = min(nblk, lo + seg);
    uint32_t eq = 0u;
    for (int base = lo; base < hi; base += 32 * kUnroll) {
        float v[kUnroll];
#pragma unroll
        for (int t = 0; t < kUnroll; ++t) {
            const int j = base + t * 32 + lane;
            v[t] = j < hi ? row[j] : 0.0f;
        }
#pragma unroll
        for (int t = 0; t < kUnroll; ++t) eq += (base + t * 32 + lane < hi && ordered(v[t]) == thr) ? 1u : 0u;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) eq += __shfl_xor_sync(0xFFFFFFFFu, eq, o);
    if (lane == 0) w_eq[warp] = eq;
    __syncthreads();
    uint32_t eq_before = 0u;
    for (int w = 0; w < warp; ++w) eq_before += w_eq[w];
    // two sweeps of the segment: the warp's picks counted, then written at the picks of the warps before
    uint32_t at = 0u;
    for (int sweep = 0; sweep < 2; ++sweep) {
        uint32_t seen = eq_before, taken = 0u;
        for (int base0 = lo; base0 < hi; base0 += 32 * kUnroll) {
            float v[kUnroll];
#pragma unroll
            for (int t = 0; t < kUnroll; ++t) {
                const int j = base0 + t * 32 + lane;
                v[t] = j < hi ? row[j] : 0.0f;
            }
#pragma unroll
            for (int t = 0; t < kUnroll; ++t) {
                const int j = base0 + t * 32 + lane;
                const float x = v[t];
                const uint32_t u = j < hi ? ordered(x) : 0u;
                const bool e = j < hi && u == thr;
                const unsigned eqm = __ballot_sync(0xFFFFFFFFu, e);
                const bool take = j < hi && (u > thr || (e && seen + __popc(eqm & below) < need));
                const unsigned selm = __ballot_sync(0xFFFFFFFFu, take);
                // `where(s.gather(-1, idx) > -inf, idx, -1)`: NaN maxima compare false as well
                if (sweep == 1 && take) cblk[(size_t)r * cblk_ld + at + taken + __popc(selm & below)] = x > -__int_as_float(0x7F800000) ? j : -1;
                seen += __popc(eqm);
                taken += __popc(selm);
            }
        }
        if (sweep == 0) {
            if (lane == 0) w_sel[warp] = taken;
            __syncthreads();
            for (int w = 0; w < warp; ++w) at += w_sel[w];
        }
    }
}

// tf_dsv41_apply_cand on the pool as bits (tf_dsv41_cand_bits'): score[r, t] = -inf where block t / bsize's bit is 0.
__global__ void tf_dsv41_apply_cand_bits(float* __restrict__ score, long long ld, int n,
                                         const uint32_t* __restrict__ bits, long long bits_ld, int bsize) {
    const int r = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int b = t / bsize;
    if (((bits[(size_t)r * bits_ld + b / 32] >> (b & 31)) & 1u) == 0u) score[(size_t)r * ld + t] = -__int_as_float(0x7F800000);
}

// model.py apply_candidates in place: score[r, t] = -inf where mask[r, t / bsize] is 0, t < n.
__global__ void tf_dsv41_apply_cand(float* __restrict__ score, long long ld, int n,
                                    const unsigned char* __restrict__ mask, long long mask_ld, int bsize) {
    const int r = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    if (mask[(size_t)r * mask_ld + t / bsize] == 0) score[(size_t)r * ld + t] = -__int_as_float(0x7F800000);
}

}
