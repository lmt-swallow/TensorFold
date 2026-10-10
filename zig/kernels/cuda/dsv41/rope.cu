// ours: the DSV4.1 fork's RoPE table rows (ops.py rope_cs: torch.polar(1, torch.outer(float(positions), freqs)) on
// the CPU, i.e. glibc's cosf / sinf of the fp32 product) computed on the device for given positions, bit for bit:
// row r of the output is the table's row at position pos[r] (or start + r without pos).
//
// freqs: f32 [half], the fork's per-pair frequencies (YaRN ramp applied; dump_rope.py's rope_<kind>_freqs.bin).
// The angle is the fp32 product float(p) * freq (torch.outer of an fp32 position, exact below 2^24, and the fp32
// frequencies); rope_glibc.h reproduces glibc 2.39's aarch64 sinf / cosf of it.

#include <cuda_runtime.h>

#include "rope_glibc.h"

// cos, sin: f32 [n, half]. One thread an entry.
extern "C" __global__ void __launch_bounds__(256) tf_dsv41_rope_rows(const long long* __restrict__ pos, long long start, int n,
                                                                    const float* __restrict__ freqs, int half,
                                                                    float* __restrict__ cos_out, float* __restrict__ sin_out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * half) return;
    const long long r = i / half;
    const int j = (int)(i - r * half);
    const long long p = pos ? pos[r] : start + r;
    const float angle = __fmul_rn(__ll2float_rn(p), freqs[j]);
    float s, c;
    rg_sincosf(angle, &s, &c);
    cos_out[i] = c;
    sin_out[i] = s;
}
