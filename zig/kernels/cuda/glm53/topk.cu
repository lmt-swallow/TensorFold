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
