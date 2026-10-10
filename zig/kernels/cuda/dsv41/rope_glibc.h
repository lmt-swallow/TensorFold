// glibc 2.39's sinf and cosf (sysdeps/ieee754/flt-32/s_sinf.c, s_cosf.c, s_sincosf.h, sincosf_poly.h) as compiled
// for aarch64 (Ubuntu GLIBC 2.39-0ubuntu8.7, the fork image's libm): TOINT_INTRINSICS (frinta / fcvtas: round half
// away from zero) and GCC's FMA contraction of every `a + b * c` in the polynomials and in reduce_fast's
// `x - round(r) * hpi`. rg_sincosf(y) returns, bit for bit, the pair (sinf(y), cosf(y)) that libm returns for every
// float y (inf and NaN inputs give NaN); tools/zig/dsv41/rope_glibc_check.c compares it with the image's libm over
// all 2^32 inputs.
//
// Shared by the device kernel (rope.cu) and the host check: every double operation is spelled as an explicit
// rounding (RG_MUL / RG_FMA), so no compiler contraction can change a result.

#ifndef TF_DSV41_ROPE_GLIBC_H
#define TF_DSV41_ROPE_GLIBC_H

#include <stdint.h>

#if defined(__CUDACC__)
#define RG_FN static __host__ __device__ __forceinline__
#else
#define RG_FN static inline
#endif

#if defined(__CUDA_ARCH__)
#define RG_MUL(a, b) __dmul_rn((a), (b))
#define RG_FMA(a, b, c) __fma_rn((a), (b), (c))
#define RG_I64_TO_F64(v) __ll2double_rn(v)
#define RG_ROUND(v) round(v)
#else
#include <math.h>
#define RG_MUL(a, b) ((double)(a) * (double)(b))
#define RG_FMA(a, b, c) fma((a), (b), (c))
#define RG_I64_TO_F64(v) ((double)(v))
#define RG_ROUND(v) round(v)
#endif

// __sincosf_table (s_sincosf_data.c, TOINT_INTRINSICS): [0] the polynomials, [1] the cosine polynomial negated
#define RG_HPI_INV 0x1.45F306DC9C883p-1
#define RG_HPI 0x1.921FB54442D18p0
#define RG_PI63 0x1.921FB54442D18p-62
#define RG_S1 -0x1.555545995a603p-3
#define RG_S2 0x1.1107605230bc4p-7
#define RG_S3 -0x1.994eb3774cf24p-13
#define RG_C0 0x1p0
#define RG_C1 -0x1.ffffffd0c621cp-2
#define RG_C2 0x1.55553e1068f19p-5
#define RG_C3 -0x1.6c087e89a359dp-10
#define RG_C4 0x1.99343027bf8c3p-16

RG_FN uint32_t rg_inv_pio4(int i) {
    // __inv_pio4: 4/PI to 192 bits, 8 new bits an entry
    switch (i) {
    case 0: return 0xa2u; case 1: return 0xa2f9u; case 2: return 0xa2f983u; case 3: return 0xa2f9836eu;
    case 4: return 0xf9836e4eu; case 5: return 0x836e4e44u; case 6: return 0x6e4e4415u; case 7: return 0x4e441529u;
    case 8: return 0x441529fcu; case 9: return 0x1529fc27u; case 10: return 0x29fc2757u; case 11: return 0xfc2757d1u;
    case 12: return 0x2757d1f5u; case 13: return 0x57d1f534u; case 14: return 0xd1f534ddu; case 15: return 0xf534ddc0u;
    case 16: return 0x34ddc0dbu; case 17: return 0xddc0db62u; case 18: return 0xc0db6295u; case 19: return 0xdb629599u;
    case 20: return 0x6295993cu; case 21: return 0x95993c43u; case 22: return 0x993c4390u; default: return 0x3c439041u;
    }
}

// sinf_poly, n even: the sine polynomial of xs (x times its quadrant sign) and x2 = x * x
RG_FN float rg_sin_poly(double xs, double x2) {
    const double s1 = RG_FMA(x2, RG_S3, RG_S2);
    const double x3 = RG_MUL(xs, x2);
    const double s = RG_FMA(x3, RG_S1, xs);
    const double x7 = RG_MUL(x2, x3);
    return (float)RG_FMA(s1, x7, s);
}

// sinf_poly, n odd: the cosine polynomial of x2, negated when neg (__sincosf_table[1])
RG_FN float rg_cos_poly(double x2, int neg) {
    const double g = neg ? -1.0 : 1.0; // the table's sign: exact
    const double x4 = RG_MUL(x2, x2);
    const double x6 = RG_MUL(x2, x4);
    const double c1 = RG_FMA(x2, g * RG_C1, g * RG_C0);
    const double c2 = RG_FMA(x2, g * RG_C4, g * RG_C3);
    const double c = RG_FMA(x4, g * RG_C2, c1);
    return (float)RG_FMA(c2, x6, c);
}

// (sinf(y), cosf(y)) of glibc 2.39 aarch64, bit for bit
RG_FN void rg_sincosf(float y, float* sinp, float* cosp) {
    union { float f; uint32_t u; } b;
    b.f = y;
    const uint32_t xi0 = b.u;
    const uint32_t top = (xi0 >> 20) & 0x7ff;
    double x = (double)y;
    if (top < 0x3f4) { // |y| < pi / 4
        if (top < 0x398) { // |y| < 2^-12
            *sinp = y;
            *cosp = 1.0f;
            return;
        }
        const double x2 = RG_MUL(x, x);
        *sinp = rg_sin_poly(x, x2);
        *cosp = rg_cos_poly(x2, 0);
        return;
    }
    int n;
    int q; // the quadrant the sign and the table follow
    if (top < 0x42f) { // |y| < 120: reduce_fast
        const double r = RG_MUL(x, RG_HPI_INV);
        const double rr = RG_ROUND(r);
        n = (int)(long long)rr;
        x = RG_FMA(-rr, RG_HPI, x);
        q = n;
    } else if (top < 0x7f8) { // finite: reduce_large
        const int k = (xi0 >> 26) & 15;
        const int shift = (xi0 >> 23) & 7;
        uint32_t m = ((xi0 & 0xffffff) | 0x800000) << shift;
        uint64_t res0 = (uint64_t)(uint32_t)(m * rg_inv_pio4(k));
        const uint64_t res1 = (uint64_t)m * rg_inv_pio4(k + 4);
        const uint64_t res2 = (uint64_t)m * rg_inv_pio4(k + 8);
        res0 = (res2 >> 32) | (res0 << 32);
        res0 += res1;
        const uint64_t nn = (res0 + (1ULL << 61)) >> 62;
        res0 -= nn << 62;
        x = RG_MUL(RG_I64_TO_F64((long long)res0), RG_PI63);
        n = (int)nn;
        q = n + (int)(xi0 >> 31);
    } else {
        const float nan = y - y;
        *sinp = nan;
        *cosp = nan;
        return;
    }
    const double s = (q & 3) == 1 || (q & 3) == 2 ? -1.0 : 1.0; // __sincosf_table sign[q & 3]
    const int neg = (q & 2) != 0;
    const double x2 = RG_MUL(x, x);
    const double xs = RG_MUL(x, s);
    if (n & 1) {
        *sinp = rg_cos_poly(x2, neg);
        *cosp = rg_sin_poly(xs, x2);
    } else {
        *sinp = rg_sin_poly(xs, x2);
        *cosp = rg_cos_poly(x2, neg);
    }
}

#endif
