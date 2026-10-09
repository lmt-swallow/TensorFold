// L2 prefetch of byte ranges: the weights the next kernels of a decode step will read, pulled into L2 while the GPU
// runs kernels that leave DRAM idle (gathers, norms, attention). Threads issue bulk L2 prefetches
// (cp.async.bulk.prefetch.L2, sm_90+: no registers, no shared memory, nothing waits for them) in chunks, the ranges in
// the order given (the order their reader takes them), after an optional delay: all at once, or one block's waves of
// blockDim chunks paced by the global timer (DRAM's queue then holds about a wave: on GB10 a dependent load's latency
// grows several-fold behind a deep queue, which slows the latency-bound kernels running beside). Nothing is written:
// it changes no bit of any output, only where the next kernels find their bytes.

//
// Zig build: the fork's kernel verbatim (cuda/exl3/l2_prefetch.cu), the torch extension's host glue dropped and the
// kernel extern "C" (cuda_exl3.zig's L2Prefetch mirrors PArgs and fills it as exl3_l2_prefetch_cuda does).

#include <cuda_runtime.h>


static constexpr int PMAX = 24;               // ranges a launch

struct PArgs {
    const char* p[PMAX];               // range starts (16-byte aligned)
    long long n[PMAX];                 // range bytes (multiples of 16)
    long long end[PMAX];               // chunk index one past range i (each range starts on a chunk of its own)
    int count;
    int chunk;                         // bytes a prefetch (a multiple of 16)
    long long delay_ns;                // issue this long after the kernel starts (a kernel launched beside it, e.g. a
                                       // gather's staging kernel, first runs undisturbed)
    long long wave_ns;                 // > 0 (one block): a wave of blockDim chunks every wave_ns, in range order
};

static __device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

extern "C" __global__ void __launch_bounds__(128) tf_l2_prefetch_kernel(const __grid_constant__ PArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (a.delay_ns > 0) {
        const unsigned long long t0 = gtime();
        while (gtime() - t0 < (unsigned long long)a.delay_ns) __nanosleep(500);
    }
    const long long total = a.end[a.count - 1];
    const long long stride = (long long)gridDim.x * blockDim.x;
    int r = 0;
    const unsigned long long t0 = gtime();
    long long wave = 0;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c - threadIdx.x < total; c += stride) {
        if (a.wave_ns > 0) {
            // paced waves: the bytes in flight stay about one wave, the order about the ranges' order
            __syncthreads();
            if (threadIdx.x == 0)
                while (gtime() - t0 < (unsigned long long)(wave * a.wave_ns)) __nanosleep(256);
            __syncthreads();
            ++wave;
        }
        if (c >= total) continue;
        while (c >= a.end[r]) ++r;
        const long long off = (c - (r ? a.end[r - 1] : 0)) * a.chunk;
        const long long left = a.n[r] - off;
        const unsigned size = (unsigned)(left < a.chunk ? left : a.chunk);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a.p[r] + off), "r"(size) : "memory");
    }
#endif
}

