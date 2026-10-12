// A row's k best (value, id) of a bf16 logits row, the order decode.py `sample_rows` keeps them in (cuda_engine.zig
// `Runner.sample`): the larger value first, of equal values the lower id. Each thread keeps its own k best of a
// strided slice, then pairs of lists merge in a fixed tree; the k best of a row under a total order do not depend on
// how the row was split, so the output equals `sample`'s host scan for every row without a NaN. A NaN is not a total
// order's value: the host takes the row's first k values unconditionally and sorts them with a comparator a NaN
// leaves unordered, while here a NaN is never better and enters a thread's list only while it holds fewer than k, so
// a row with a NaN may come out in another order or set (TF_GLM_TOPK_CHECK=1 compares both and stops on a difference).
#include <cuda_bf16.h>
#include <stdint.h>

#define TOPK_THREADS 128
#define TOPK_MOST 32

struct Cand {
    float v;
    int id;
};

__device__ __forceinline__ bool better(Cand a, Cand b) { return a.v > b.v || (a.v == b.v && a.id < b.id); }

// Insert c into the sorted list l[0 .. *n) of at most k.
__device__ __forceinline__ void insert(Cand* l, int* n, int k, Cand c) {
    int i;
    if (*n < k) {
        i = (*n)++;
    } else {
        if (!better(c, l[k - 1])) return;
        i = k - 1;
    }
    while (i > 0 && better(c, l[i - 1])) {
        l[i] = l[i - 1];
        --i;
    }
    l[i] = c;
}

// out: [k values][k ids as int bits], f32 words. `id_base`: the id of column 0 (the rank's vocabulary offset).
extern "C" __global__ void __launch_bounds__(TOPK_THREADS) tf_glm53_topk_row(const __nv_bfloat16* logits, int n,
                                                                            int k, int id_base, float* out) {
    __shared__ Cand lists[TOPK_THREADS][TOPK_MOST];
    __shared__ int counts[TOPK_THREADS];
    const int t = threadIdx.x;
    Cand mine[TOPK_MOST];
    int m = 0;
    for (int j = t; j < n; j += TOPK_THREADS) {
        const Cand c = {__bfloat162float(logits[j]), id_base + j};
        insert(mine, &m, k, c);
    }
    for (int i = 0; i < m; ++i) lists[t][i] = mine[i];
    counts[t] = m;
    __syncthreads();
    for (int s = TOPK_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) {
            Cand merged[TOPK_MOST];
            const int na = counts[t], nb = counts[t + s];
            int a = 0, b = 0, o = 0;
            while (o < k && (a < na || b < nb)) {
                if (b >= nb || (a < na && better(lists[t][a], lists[t + s][b]))) merged[o++] = lists[t][a++];
                else merged[o++] = lists[t + s][b++];
            }
            for (int i = 0; i < o; ++i) lists[t][i] = merged[i];
            counts[t] = o;
        }
        __syncthreads();
    }
    if (t < k) {
        out[t] = t < counts[0] ? lists[0][t].v : -INFINITY;
        reinterpret_cast<int*>(out)[k + t] = t < counts[0] ? lists[0][t].id : -1;
    }
}

// The same k best in two stages (TF_GLM_DRAFT_TOPK=2): blocks of a row's columns each sort their slice's keys and
// keep their k best, then one block sorts the blocks' k bests and writes the row's. A key is the value in total order
// (-0 as +0) above the inverted id, so the order is the host scan's (larger value, then lower id) and the k best of
// the row do not depend on how it was cut; the values written are the logits' own bits (a -0 stays -0). The NaN
// caveat above holds here too: a NaN's key ranks above +inf.
#define TOPK2_THREADS 256
#define TOPK2_KEYS 2048

__device__ __forceinline__ unsigned long long topk_key(float v, int id) {
    uint32_t b = __float_as_uint(v);
    if (b == 0x80000000u) b = 0u;
    const uint32_t k = b ^ ((uint32_t)((int32_t)b >> 31) | 0x80000000u);
    return ((unsigned long long)k << 32) | (unsigned long long)(0xFFFFFFFFu - (uint32_t)id);
}

// keys[0 .. TOPK2_KEYS) sorted descending in place (bitonic), by the block's threads.
__device__ __forceinline__ void sort_desc(unsigned long long* keys) {
    for (int size = 2; size <= TOPK2_KEYS; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int p = threadIdx.x; p < TOPK2_KEYS / 2; p += TOPK2_THREADS) {
                const int i = 2 * p - (p & (stride - 1));
                const int j = i + stride;
                const bool desc = (i & size) == 0;
                const unsigned long long a = keys[i], b = keys[j];
                if ((a < b) == desc) {
                    keys[i] = b;
                    keys[j] = a;
                }
            }
            __syncthreads();
        }
    }
}

// Block b: columns [b * slice, min(n, (b + 1) * slice)) of the row (slice <= TOPK2_KEYS); its k best keys into
// part[b * k ..] (zeros past the slice's end: below every key).
extern "C" __global__ void __launch_bounds__(TOPK2_THREADS) tf_glm53_topk_part(const __nv_bfloat16* logits, int n,
                                                                              int slice, int k, int id_base,
                                                                              unsigned long long* part) {
    __shared__ unsigned long long keys[TOPK2_KEYS];
    const int c0 = blockIdx.x * slice;
    for (int i = threadIdx.x; i < TOPK2_KEYS; i += TOPK2_THREADS) {
        const int c = c0 + i;
        keys[i] = (i < slice && c < n) ? topk_key(__bfloat162float(logits[c]), id_base + c) : 0ull;
    }
    __syncthreads();
    sort_desc(keys);
    for (int i = threadIdx.x; i < k; i += TOPK2_THREADS) part[blockIdx.x * k + i] = keys[i];
}

// One block: the `blocks` k bests (blocks * k <= TOPK2_KEYS) sorted, the row's k best into out [k values][k ids as int
// bits] (f32 words, as tf_glm53_topk_row writes them), each value read back from the row.
extern "C" __global__ void __launch_bounds__(TOPK2_THREADS) tf_glm53_topk_merge(const __nv_bfloat16* logits,
                                                                               const unsigned long long* part,
                                                                               int blocks, int k, int id_base,
                                                                               float* out) {
    __shared__ unsigned long long keys[TOPK2_KEYS];
    for (int i = threadIdx.x; i < TOPK2_KEYS; i += TOPK2_THREADS) keys[i] = i < blocks * k ? part[i] : 0ull;
    __syncthreads();
    sort_desc(keys);
    for (int i = threadIdx.x; i < k; i += TOPK2_THREADS) {
        const int id = (int)(0xFFFFFFFFu - (uint32_t)(keys[i] & 0xFFFFFFFFull));
        out[i] = __bfloat162float(logits[id - id_base]);
        reinterpret_cast<int*>(out)[k + i] = id;
    }
}
