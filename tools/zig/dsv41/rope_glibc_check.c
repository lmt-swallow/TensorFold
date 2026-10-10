// Host checks of zig/kernels/cuda/dsv41/rope_glibc.h (no GPU):
//
//   rope_glibc_check libm <libm.so.6>
//       rg_sincosf against that libm's sinf and cosf (loaded in a namespace of its own, so the fork image's libm can be
//       checked on a host with another glibc) over all 2^32 float inputs; prints the mismatch count and the first few.
//   rope_glibc_check table <rope_X_ROWS.bin> <rows> [freqs.bin]
//       Every row p of a dump_rope.py table against rg_sincosf(float(p) * freq[j]) (torch.outer's fp32 product).
//       With freqs.bin (f32 [rope_dim / 2], dump_rope.py's rope_X_freqs.bin) the frequencies are taken from it;
//       without, each column's frequency is recovered from the table (the floats near asin(sin[1][j]) whose rows all
//       match) and printed. Prints the first mismatching row / column and the ulp distance.
//
// Build: gcc -O2 -ffp-contract=off -fopenmp -I zig/kernels/cuda/dsv41 tools/zig/dsv41/rope_glibc_check.c -ldl -lm

#define _GNU_SOURCE
#include <dlfcn.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rope_glibc.h"

static uint32_t bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static int64_t ulps(float a, float b) {
    int32_t ia = (int32_t)bits(a), ib = (int32_t)bits(b);
    if (ia < 0) ia = (int32_t)0x80000000 - ia;
    if (ib < 0) ib = (int32_t)0x80000000 - ib;
    return llabs((long long)ia - (long long)ib);
}

static int check_libm(const char* path) {
    void* h = dlmopen(LM_ID_NEWLM, path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlmopen %s: %s\n", path, dlerror());
        return 2;
    }
    float (*lsin)(float) = (float (*)(float))dlsym(h, "sinf");
    float (*lcos)(float) = (float (*)(float))dlsym(h, "cosf");
    if (!lsin || !lcos) {
        fprintf(stderr, "no sinf / cosf in %s\n", path);
        return 2;
    }
    uint64_t bad = 0, checked = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : bad, checked)
    for (int64_t hi = 0; hi < 65536; hi++) {
        for (uint32_t lo = 0; lo < 65536; lo++) {
            const uint32_t u = ((uint32_t)hi << 16) | lo;
            const float y = from_bits(u);
            float s, c;
            rg_sincosf(y, &s, &c);
            const float ls = lsin(y), lc = lcos(y);
            checked++;
            const int same = (isnan(ls) ? isnan(s) : bits(ls) == bits(s)) && (isnan(lc) ? isnan(c) : bits(lc) == bits(c));
            if (!same) {
                if (bad < 8) {
#pragma omp critical
                    printf("mismatch y=%a (0x%08x): sin %a vs libm %a, cos %a vs libm %a\n", y, u, s, ls, c, lc);
                }
                bad++;
            }
        }
    }
    printf("libm %s: %" PRIu64 " inputs, %" PRIu64 " mismatches\n", path, checked, bad);
    return bad != 0;
}

static float* read_file(const char* path, size_t want) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(2);
    }
    float* p = malloc(want);
    if (fread(p, 1, want, f) != want) {
        fprintf(stderr, "%s: short (want %zu bytes)\n", path, want);
        exit(2);
    }
    fclose(f);
    return p;
}

enum { HALF = 32 };

static int64_t first_bad_row(const float* cos, const float* sin, size_t rows, int j, float f, float* gc, float* gs) {
    for (size_t p = 0; p < rows; p++) {
        float s, c;
        rg_sincosf((float)p * f, &s, &c);
        if (bits(c) != bits(cos[p * HALF + j]) || bits(s) != bits(sin[p * HALF + j])) {
            if (gc) *gc = c, *gs = s;
            return (int64_t)p;
        }
    }
    return -1;
}

static int check_table(const char* path, size_t rows, const char* freqs_path) {
    const size_t n = rows * HALF;
    float* t = read_file(path, 2 * n * 4);
    const float* cos = t;
    const float* sin = t + n;
    float freqs[HALF];
    if (freqs_path) {
        float* fr = read_file(freqs_path, HALF * 4);
        memcpy(freqs, fr, sizeof freqs);
        free(fr);
    } else {
        for (int j = 0; j < HALF; j++) {
            // sin(f) and cos(f) at row 1 pin f to a few ulps; the long rows pick the one float that matches them all
            const double guess = atan2((double)sin[HALF + j], (double)cos[HALF + j]);
            const float g = (float)guess;
            int found = 0;
            float best = g;
            for (int d = -4096; d <= 4096; d++) {
                const float f = from_bits(bits(g) + d);
                float s, c;
                rg_sincosf(f, &s, &c);
                if (bits(c) != bits(cos[HALF + j]) || bits(s) != bits(sin[HALF + j])) continue;
                if (first_bad_row(cos, sin, rows < 4096 ? rows : 4096, j, f, 0, 0) >= 0) continue;
                if (first_bad_row(cos, sin, rows, j, f, 0, 0) >= 0) continue;
                found++;
                best = f;
            }
            if (found != 1) {
                printf("column %d: %d frequencies reproduce every row\n", j, found);
                free(t);
                return 1;
            }
            freqs[j] = best;
        }
        printf("freqs (f32 bits):");
        for (int j = 0; j < HALF; j++) printf(" %08x", bits(freqs[j]));
        printf("\n");
    }
    int64_t bad_row = -1;
    int bad_col = -1;
    float gc = 0, gs = 0;
    for (int j = 0; j < HALF; j++) {
        const int64_t p = first_bad_row(cos, sin, rows, j, freqs[j], &gc, &gs);
        if (p >= 0 && (bad_row < 0 || p < bad_row)) bad_row = p, bad_col = j;
    }
    if (bad_row >= 0) {
        const size_t i = (size_t)bad_row * HALF + bad_col;
        printf("%s: first mismatch row %" PRId64 " column %d: cos %a vs %a (%" PRId64 " ulp), sin %a vs %a (%" PRId64 " ulp)\n",
               path, bad_row, bad_col, gc, cos[i], ulps(gc, cos[i]), gs, sin[i], ulps(gs, sin[i]));
    } else {
        printf("%s: %zu rows x %d columns match\n", path, rows, HALF);
    }
    free(t);
    return bad_row >= 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && strcmp(argv[1], "libm") == 0) return check_libm(argv[2]);
    if (argc >= 4 && strcmp(argv[1], "table") == 0)
        return check_table(argv[2], strtoull(argv[3], 0, 10), argc >= 5 ? argv[4] : 0);
    fprintf(stderr, "usage: %s libm <libm.so.6> | table <rope_X_ROWS.bin> <rows> [freqs.bin]\n", argv[0]);
    return 2;
}
