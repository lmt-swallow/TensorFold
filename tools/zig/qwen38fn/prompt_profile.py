#!/usr/bin/env python3
"""Per-kernel GPU time of `tf-cuda-test qwen38fn-prompt-time` under nsys, by depth.

  nsys profile -t cuda -o /tmp/qpt zig-out/bin/tf-cuda-test qwen38fn-prompt-time <set> <model> 0,65536 3
  nsys export -t sqlite -o /tmp/qpt.sqlite /tmp/qpt.nsys-rep
  python3 prompt_profile.py /tmp/qpt.sqlite --chunks 3 --depths 0,65536

The harness pauses 2 s before each depth, so the kernel timeline splits into the model load and one segment a depth
at gaps of more than a second; the last `len(depths)` segments are the depths. Each segment holds `chunks + 1`
chunks (the warm one included), and every figure is per chunk: launches, GPU ms, share of the summed kernel time.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("db")
    p.add_argument("--chunks", type=int, default=3, help="timed chunks a depth (the harness adds a warm one)")
    p.add_argument("--depths", default="0,65536,204800")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--gap-ms", type=float, default=1000.0)
    a = p.parse_args()
    depths = a.depths.split(",")
    con = sqlite3.connect(a.db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    rows = con.execute(
        "SELECT start, end, demangledName, shortName, gridX, gridY, gridZ, blockX FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    segs: list[list[tuple]] = [[]]
    prev_end = None
    for r in rows:
        if prev_end is not None and r[0] - prev_end > a.gap_ms * 1e6:
            segs.append([])
        segs[-1].append(r)
        prev_end = max(prev_end or 0, r[1])
    segs = segs[-len(depths):]
    per = a.chunks + 1
    for depth, seg in zip(depths, segs):
        acc: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
        for s, e, dn, sn, gx, gy, gz, bx in seg:
            full = names.get(dn, "?")
            short = names.get(sn, "?")
            key = short if len(full) > 160 else full
            if short.startswith("grouped_kernel") or "grouped_kernel" in full:
                key = f"grouped_kernel grid.y={gy}"  # gate+up (N=640) vs down (N=2560)
            acc[key][0] += 1
            acc[key][1] += (e - s) / 1e6
        total = sum(v[1] for v in acc.values())
        wall = (seg[-1][1] - seg[0][0]) / 1e6
        print(f"\n## depth {depth}: {total / per:.1f} ms of kernels a chunk, {wall / per:.1f} ms wall a chunk "
              f"({2048 * per / wall * 1e3:.0f} tok/s)\n")
        print("| kernel | launches/chunk | ms/chunk | share |")
        print("|---|---:|---:|---:|")
        for k, (n, ms) in sorted(acc.items(), key=lambda kv: -kv[1][1])[: a.top]:
            print(f"| {k[:110]} | {n / per:.0f} | {ms / per:.2f} | {100 * ms / total:.1f}% |")


if __name__ == "__main__":
    main()
