#!/usr/bin/env python3
"""Where the device sits idle in an nsys trace (nsys -t cuda): every gap between one GPU operation's end and the next
one's start, summed by the operation before it (a gap after a read-back is host turnaround: the host waited for the
result, then issued the next work), and the share of the traced span the device was busy.

  nsys stats -q --report cuda_gpu_trace --format csv run.nsys-rep > trace.csv
  python3 gpu_gaps.py trace.csv [skip-s [length-s]]

skip-s and length-s window the trace from its first draft pick (decoding started): the seconds skipped, then those
kept (default 0 and all).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def kind(name: str) -> str:
    """A short label for the operation before a gap."""
    for key, label in (("[CUDA memcpy Device-to-Host]", "read-back (DtoH)"), ("[CUDA memcpy Host-to-Device]", "upload (HtoD)"),
                       ("[CUDA memcpy Device-to-Device]", "device copy"), ("tf_argmax", "argmax (verify end)"),
                       ("tf_qfn_draft_pick", "draft pick (head step end)"), ("qmm_kernel", "draft head matmul"),
                       ("[CUDA memset", "memset")):
        if key in name:
            return label
    return "other kernels"


def main() -> int:
    rows = []
    with open(sys.argv[1], newline="") as f:
        for r in csv.DictReader(f):
            try:
                start, dur = int(r["Start (ns)"]), int(r["Duration (ns)"])
            except (KeyError, ValueError):
                continue
            rows.append((start, start + dur, r.get("Name", "")))
    rows.sort()
    first = next((s for s, _, n in rows if "draft_pick" in n), rows[0][0] if rows else 0)
    lo = first + int(float(sys.argv[2]) * 1e9) if len(sys.argv) > 2 else rows[0][0] if rows else 0
    hi = lo + int(float(sys.argv[3]) * 1e9) if len(sys.argv) > 3 else 1 << 62
    rows = [r for r in rows if lo <= r[0] < hi]
    if not rows:
        print("no GPU operations")
        return 1
    span = rows[-1][1] - rows[0][0]
    busy, gaps, counts = 0, defaultdict(int), defaultdict(int)
    end = rows[0][0]
    prev = ""
    for start, stop, name in rows:
        if start > end:
            gap = start - end
            gaps[kind(prev)] += gap
            counts[kind(prev)] += 1 if gap > 20_000 else 0
        busy += max(0, stop - max(start, end))
        if stop > end:
            end, prev = stop, name
    idle = span - busy
    print(f"span {span / 1e9:.2f} s, device busy {100 * busy / span:.1f}%, idle {idle / 1e6:.0f} ms")
    for k, v in sorted(gaps.items(), key=lambda x: -x[1]):
        print(f"  idle after {k:28s} {v / 1e6:8.1f} ms ({100 * v / span:4.1f}% of the span), {counts[k]} gaps over 20 us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
