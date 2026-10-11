// The MTP draft head's pick for each row of its logits (decode.py `sample_draft`, greedy): the first index of the
// row's largest value (a NaN never wins) and that value's probability 1 / sum(exp(x - top)), summed in fp64. One
// block a row and a fixed reduction order, so a row's result depends on its values alone.
#include <cuda_bf16.h>

#define PICK_THREADS 1024

extern "C" __global__ void __launch_bounds__(PICK_THREADS) tf_qfn_draft_pick(const __nv_bfloat16* logits, int n,
                                                                             int stride, int* token, double* prob) {
    __shared__ float top_s[PICK_THREADS];
    __shared__ int idx_s[PICK_THREADS];
    __shared__ double sum_s[PICK_THREADS];
    const __nv_bfloat16* x = logits + (long long)blockIdx.x * stride;
    const int t = threadIdx.x;
    float top = -INFINITY;
    int idx = -1;
    for (int i = t; i < n; i += PICK_THREADS) {
        const float v = __bfloat162float(x[i]);
        if (v > top || (idx < 0 && v == v)) {  // the thread's first largest; -inf rows still pick their first
            top = v;
            idx = i;
        }
    }
    top_s[t] = top;
    idx_s[t] = idx;
    __syncthreads();
    for (int s = PICK_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) {
            const float a = top_s[t], b = top_s[t + s];
            const int ia = idx_s[t], ib = idx_s[t + s];
            if (ib >= 0 && (ia < 0 || b > a || (b == a && ib < ia))) {
                top_s[t] = b;
                idx_s[t] = ib;
            }
        }
        __syncthreads();
    }
    const double best = (double)top_s[0];
    double sum = 0.0;
    for (int i = t; i < n; i += PICK_THREADS) sum += exp((double)__bfloat162float(x[i]) - best);
    sum_s[t] = sum;
    __syncthreads();
    for (int s = PICK_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) sum_s[t] += sum_s[t + s];
        __syncthreads();
    }
    if (t == 0) {
        token[blockIdx.x] = idx_s[0] < 0 ? 0 : idx_s[0];
        prob[blockIdx.x] = 1.0 / sum_s[0];
    }
}
