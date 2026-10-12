// GLM-5.3-Flash's prompt pool selection (sparse.py `top_pools` of a prompt chunk's rows: `_select_rows` without
// positions, or `select_split`'s pool form for blocks of fewer than 64 rows over 8,192 scores or more). Its output is
// fixed by its definition alone: each row's k best of its np scores (a score's order is its fp32 bits in total order,
// -0 as +0), ties to the lower pool, the k pool indices ascending. Any exact selection writes the same int64s, so
// this one is free to read the row as few times as it can: integers only, no arithmetic whose order could matter.
//
// One block a row. Pass 1 counts the row's keys by their top 11 bits; the bin holding the k-th best is the threshold
// bin. Pass 2 reads the row again in index order: keys above the threshold bin are taken (their indices kept in
// index order), keys in it are kept as candidates (key and index, in index order). The k-th best among the
// candidates and the ties it takes are then found in shared memory, and the two ascending index lists are merged
// into the row's output. When the threshold bin holds more candidates than shared memory does (many equal or nearly
// equal scores), the block falls back to a radix select over the row (three more counting passes) and one pass in
// index order that writes the picks: the same output, more reads.
//
// Checked by tf-cuda-test glm53-select (the definition computed on the host, and the captured `_select_rows` /
// `select_split` where the Triton set holds the variant, over rows of distinct values, few values, zeros, -inf
// tails, one value, signed zeros and threshold bins filled to the candidate capacity and one past it, np 1,024 to
// 150,016) and by glm53-prompt's fixtures under TF_GLM_PROMPT_CHECK=1 (every call against the captured kernels).
#include <stdint.h>

namespace tf_glm53_select {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kBins = 2048;     // the first pass's 11 key bits
constexpr int kCand = 6144;     // threshold-bin candidates held in shared memory
constexpr int kMaxK = 512;

// The score's key: a word whose unsigned order is the scores' order (sparse.py `_order_key`: (s + 0.0)'s bits, the
// sign bit flipped for positive scores, every bit for negative ones; -0 counts as +0).
__device__ __forceinline__ uint32_t order_key(float s) {
    uint32_t b = __float_as_uint(s);
    if (b == 0x80000000u) b = 0u;
    return b ^ ((uint32_t)((int32_t)b >> 31) | 0x80000000u);
}

// Exclusive block-wide prefix sum of v; `total` gets the block's sum. `scratch`: kWarps + 1 words.
__device__ __forceinline__ uint32_t block_exclusive(uint32_t v, uint32_t* scratch, uint32_t& total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    uint32_t x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        const uint32_t w = lane < kWarps ? scratch[lane] : 0u;
        uint32_t s = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < kWarps) scratch[lane] = s - w;
        if (lane == kWarps - 1) scratch[kWarps] = s;
    }
    __syncthreads();
    const uint32_t out = scratch[warp] + x - v;
    total = scratch[kWarps];
    __syncthreads();
    return out;
}

struct Shared {
    uint32_t hist[kBins];
    uint32_t above[kMaxK];     // indices of keys above the threshold bin, ascending
    uint32_t chosen[kMaxK];    // the candidates taken, ascending
    uint32_t ckey[kCand];
    uint32_t cidx[kCand];
    uint32_t scan[kWarps + 1];
    uint32_t misc[8];
};

// The bin b (of `bins`, counts in s.hist) holding the k-th largest key: the count above it into `above_count`.
__device__ uint32_t threshold_bin(Shared& s, int bins, uint32_t k, uint32_t& above_count) {
    // suffix sums over the bins, highest first: thread t takes bins [t * per, (t + 1) * per) counted from the top
    const int per = (bins + kThreads - 1) / kThreads;
    uint32_t mine = 0;
    for (int j = 0; j < per; ++j) {
        const int bin = bins - 1 - (threadIdx.x * per + j);
        if (bin >= 0) mine += s.hist[bin];
    }
    uint32_t total;
    const uint32_t before = block_exclusive(mine, s.scan, total);   // keys in higher bins than this thread's
    uint32_t run = before;
    for (int j = 0; j < per; ++j) {
        const int bin = bins - 1 - (threadIdx.x * per + j);
        if (bin < 0) break;
        const uint32_t c = s.hist[bin];
        if (run < k && run + c >= k) {
            s.misc[0] = (uint32_t)bin;
            s.misc[1] = run;
        }
        run += c;
    }
    __syncthreads();
    above_count = s.misc[1];
    const uint32_t b = s.misc[0];
    __syncthreads();
    return b;
}

// One pass in index order writing the picks of a fully known threshold: every key above `t`, then the first `ties`
// keys equal to it (scores with fewer keys than k never reach here: k <= np).
__device__ void write_by_threshold(Shared& s, const float* row, int np, uint32_t t, uint32_t ties, int64_t* out) {
    uint32_t written = 0, equal_seen = 0;
    for (int c = 0; c < np; c += kThreads) {
        const int i = c + threadIdx.x;
        const uint32_t key = i < np ? order_key(row[i]) : 0u;
        const uint32_t eq = (i < np && key == t) ? 1u : 0u;
        uint32_t neq;
        const uint32_t eq_before = block_exclusive(eq, s.scan, neq);
        const uint32_t take = (i < np && (key > t || (eq && equal_seen + eq_before < ties))) ? 1u : 0u;
        uint32_t ntake;
        const uint32_t at = block_exclusive(take, s.scan, ntake);
        if (take) out[written + at] = i;
        written += ntake;
        equal_seen += neq;
    }
}

// Radix select of the k-th largest key over the whole row (bits below the 11 already fixed by `prefix`), then the
// ordered write: the fallback for a threshold bin too full for shared memory.
__device__ void fallback(Shared& s, const float* row, int np, uint32_t k, uint32_t prefix, uint32_t above, int64_t* out) {
    uint32_t fixed = 0xFFE00000u;            // the 11 bits pass 1 fixed
    uint32_t need = k - above;               // keys still to take inside the prefix
    const int shifts[3] = {10, 0, 0};
    const int widths[3] = {11, 10, 0};
    for (int p = 0; p < 2; ++p) {
        const int width = widths[p];
        const int shift = shifts[p];
        const uint32_t mask = ((1u << width) - 1u) << shift;
        for (int j = threadIdx.x; j < kBins; j += kThreads) s.hist[j] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < np; i += kThreads) {
            const uint32_t key = order_key(row[i]);
            if ((key & fixed) == prefix) atomicAdd(&s.hist[(key & mask) >> shift], 1u);
        }
        __syncthreads();
        uint32_t higher;
        const uint32_t b = threshold_bin(s, 1 << width, need, higher);
        need -= higher;
        prefix |= b << shift;
        fixed |= mask;
    }
    write_by_threshold(s, row, np, prefix, need, out);
}

// Rows of `np` scores (row r at S + r * s_stride), the k best of each into out [rows, k] int64, ascending.
__global__ void __launch_bounds__(kThreads) select_pools_kernel(const float* __restrict__ S, int s_stride, int np, int k,
                                                                int64_t* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem[];
    Shared& s = *reinterpret_cast<Shared*>(smem);
    const int r = blockIdx.x;
    const float* row = S + (int64_t)r * s_stride;
    int64_t* dst = out + (int64_t)r * k;

    // pass 1: the top 11 bits' counts
    for (int j = threadIdx.x; j < kBins; j += kThreads) s.hist[j] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < np; i += kThreads) atomicAdd(&s.hist[order_key(row[i]) >> 21], 1u);
    __syncthreads();
    uint32_t above;
    const uint32_t bin = threshold_bin(s, kBins, (uint32_t)k, above);
    const uint32_t ncand = s.hist[bin];
    if (ncand > (uint32_t)kCand) {
        fallback(s, row, np, (uint32_t)k, bin << 21, above, dst);
        return;
    }

    // pass 2: the keys above the threshold bin (indices ascending) and its candidates (in index order)
    uint32_t na = 0, nc = 0;
    for (int c = 0; c < np; c += kThreads) {
        const int i = c + threadIdx.x;
        const uint32_t key = i < np ? order_key(row[i]) : 0u;
        const uint32_t kb = key >> 21;
        const uint32_t is_above = (i < np && kb > bin) ? 1u : 0u;
        const uint32_t is_cand = (i < np && kb == bin) ? 1u : 0u;
        uint32_t ta, tc;
        const uint32_t pa = block_exclusive(is_above, s.scan, ta);
        const uint32_t pc = block_exclusive(is_cand, s.scan, tc);
        if (is_above) s.above[na + pa] = (uint32_t)i;
        if (is_cand) {
            s.ckey[nc + pc] = key;
            s.cidx[nc + pc] = (uint32_t)i;
        }
        na += ta;
        nc += tc;
    }
    __syncthreads();
    const uint32_t need = (uint32_t)k - na;   // candidates to take (na == above)

    // the need-th largest candidate key: a radix select over the 21 low bits, in shared memory
    uint32_t prefix = bin << 21, fixed = 0xFFE00000u, still = need;
    const int widths[2] = {11, 10};
    const int shifts[2] = {10, 0};
    for (int p = 0; p < 2; ++p) {
        const uint32_t mask = ((1u << widths[p]) - 1u) << shifts[p];
        for (int j = threadIdx.x; j < kBins; j += kThreads) s.hist[j] = 0;
        __syncthreads();
        for (uint32_t j = threadIdx.x; j < nc; j += kThreads) {
            const uint32_t key = s.ckey[j];
            if ((key & fixed) == prefix) atomicAdd(&s.hist[(key & mask) >> shifts[p]], 1u);
        }
        __syncthreads();
        uint32_t higher;
        const uint32_t b = threshold_bin(s, 1 << widths[p], still, higher);
        still -= higher;
        prefix |= b << shifts[p];
        fixed |= mask;
    }
    const uint32_t t = prefix;               // the k-th best key; `still` of the keys equal to it are taken

    // the candidates taken, in index order: above t, or equal to it among the first `still`
    uint32_t nch = 0, eq_seen = 0;
    for (uint32_t c = 0; c < nc; c += kThreads) {
        const uint32_t j = c + threadIdx.x;
        const uint32_t key = j < nc ? s.ckey[j] : 0u;
        const uint32_t eq = (j < nc && key == t) ? 1u : 0u;
        uint32_t neq;
        const uint32_t eb = block_exclusive(eq, s.scan, neq);
        const uint32_t take = (j < nc && (key > t || (eq && eq_seen + eb < still))) ? 1u : 0u;
        uint32_t nt;
        const uint32_t at = block_exclusive(take, s.scan, nt);
        if (take) s.chosen[nch + at] = s.cidx[j];
        nch += nt;
        eq_seen += neq;
    }
    __syncthreads();

    // merge the two ascending lists (na + nch == k): an element's place is its rank in its list plus the count of
    // the other list's smaller indices (the lists never share an index)
    for (uint32_t j = threadIdx.x; j < na; j += kThreads) {
        const uint32_t v = s.above[j];
        uint32_t lo = 0, hi = nch;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (s.chosen[mid] < v) lo = mid + 1; else hi = mid;
        }
        dst[j + lo] = v;
    }
    for (uint32_t j = threadIdx.x; j < nch; j += kThreads) {
        const uint32_t v = s.chosen[j];
        uint32_t lo = 0, hi = na;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (s.above[mid] < v) lo = mid + 1; else hi = mid;
        }
        dst[j + lo] = v;
    }
}

}  // namespace tf_glm53_select

extern "C" __constant__ unsigned int tf_glm53_select_smem = sizeof(tf_glm53_select::Shared);
