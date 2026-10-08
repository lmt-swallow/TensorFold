// Engram rows on the device, as the Python family's torch ops compute them (model.py `Engram._decode`, `Comm.sum`):
// the FP8 tables' rows (256 e4m3 codes + 8 e8m0 scales a row, one scale a group of 32) to bf16, and the ranks'
// gathered fp32 partials summed in rank order to bf16. Every step is a single rounding the torch op makes too:
// e4m3 -> f32 is exact, the scale is an exact power of two (torch.ldexp = x * pow(2, e)), one f32 multiply, one
// round to bf16; the sum adds the ranks' partials one after another in f32, then rounds once.

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float e4m3fn_to_f32(uint8_t c) {
    const uint32_t sign = (c >> 7) & 1u;
    const uint32_t exp = (c >> 3) & 0xFu;
    const uint32_t man = c & 7u;
    float v;
    if (exp == 0xFu && man == 7u) {
        v = __int_as_float(0x7fc00000);                          // e4m3fn: the only NaN encoding
    } else if (exp == 0u) {
        v = (float)man * 0.001953125f;                           // subnormal: man * 2^-6 / 8 (exact)
    } else {
        v = __uint_as_float(((exp + 120u) << 23) | (man << 20)); // (1 + man / 8) * 2^(exp - 7)
    }
    return sign ? -v : v;
}

// 2^(s - 127) as torch.ldexp(1, s - 127) makes it: 1..254 normal, 0 the subnormal 2^-127, 255 inf.
__device__ __forceinline__ float e8m0_scale(uint8_t s) {
    if (s == 0u) return __uint_as_float(0x00400000u);
    if (s == 255u) return __uint_as_float(0x7f800000u);
    return __uint_as_float((uint32_t)s << 23);
}

}  // namespace

// codes [rows, 256] u8, scales [rows, 8] u8 -> out [rows, 256] bf16; one block of 256 threads a row.
extern "C" __global__ void tf_dsv41_engram_decode(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out, int rows) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    if (row >= rows || t >= 256) return;
    const size_t at = (size_t)row * 256 + t;
    const float v = e4m3fn_to_f32(codes[at]);
    const float sc = e8m0_scale(scales[(size_t)row * 8 + (t >> 5)]);
    out[at] = __float2bfloat16_rn(v * sc);
}

// g [world, n] f32 -> out [n] bf16: acc = g[0]; acc += g[1]; ...; bf16(acc).
extern "C" __global__ void tf_dsv41_engram_sum(const float* __restrict__ g, int world, long long n, __nv_bfloat16* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = g[i];
    for (int r = 1; r < world; ++r) acc = acc + g[(long long)r * n + i];
    out[i] = __float2bfloat16_rn(acc);
}
