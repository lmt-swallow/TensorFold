#!/usr/bin/env python3
"""Checks index_score.cu's head-sum order against the captured `_index_score_rows`' answers to the probes of
`tf-cuda-test dsv41-index-probe` (its log as the argument).

The order: within each 16-head half, n_j = fma(r_j, w_j, r_{j+8} * w_{j+8}) for j < 8 (r: a head's max(s, 0)),
then ((n_0 + n_4) + (n_2 + n_6)) + ((n_1 + n_5) + (n_3 + n_7)); the score is half(0..15) + half(16..31). Every
PROBE line (heads a, b, c contributing 2^24, 1, -2^24) and FUSE line (head a's product (1 + 2^-23)(1 + 2^-7) against
head b's -(1 + 2^-7 + 2^-23)) is recomputed in fp32 with that order and compared with the kernel's score.
Prints the mismatches (none: exit 0).

usage: index_tree.py <probe log>
"""

import struct
import sys


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def fma(a: float, b: float, c: float) -> float:
    # a: fp32, b: bf16, so a * b is exact in a double; the sum then rounds once to fp32 (c, a * b well inside range)
    return f32(a * b + c)


def score(r: list, w: list) -> float:
    def half(base: int) -> float:
        n = [fma(r[base + j], w[base + j], f32(r[base + j + 8] * w[base + j + 8])) for j in range(8)]
        m = [f32(n[j] + n[j + 4]) for j in range(4)]
        o = [f32(m[j] + m[j + 2]) for j in range(2)]
        return f32(o[0] + o[1])

    return f32(half(0) + half(16))


def main() -> int:
    bad = total = 0
    for line in open(sys.argv[1]):
        p = line.split()
        if not p:
            continue
        r, w = [0.0] * 32, [0.0] * 32
        if p[0] == "PROBE":
            a, b, c, want = int(p[2]), int(p[3]), int(p[4]), float(p[5])
            r[a], w[a], r[b], w[b], r[c], w[c] = 2.0**24, 1.0, 1.0, 1.0, 2.0**24, -1.0
        elif p[0] == "FUSE":
            a, b, want = int(p[1]), int(p[2]), float(p[3])
            r[a], w[a] = f32(1 + 2.0**-23), 1 + 2.0**-7
            r[b], w[b] = f32(1 + 2.0**-7 + 2.0**-23), -1.0
        else:
            continue
        total += 1
        got = score(r, w)
        if got != f32(want):
            bad += 1
            if bad <= 5:
                print(f"{line.strip()}: the order gives {got}")
    print(f"{bad} of {total} probes differ from the order")
    return 1 if bad or total == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
