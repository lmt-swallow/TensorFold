#!/usr/bin/env python3
"""A fixture of the fork's ops.rms_norm (the torch expression model.py's prompt path uses for the compressed latents
and the index keys) for the Zig engine's torch-order kernel: seeded bf16 rows of 128 and 512 columns, wide enough
that rows near a bf16 rounding boundary show any difference in the mean's summation order.

  python3 -B /w/tools/zig/dsv41/dump_rmsnorm.py --tools /w/tools/zig --out /aot/out [--rows 131072]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("capture_kernel", HERE / "capture_kernel.py")
    ck = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ck)
    from tensorfold.families.deepseek_v41.ops import rms_norm

    g = torch.Generator(device="cuda")
    g.manual_seed(a.seed)
    eps = 1e-6
    for d in (128, 512):
        scale = torch.exp(torch.randn((a.rows, 1), generator=g, device="cuda") * 2.0)
        x = (torch.randn((a.rows, d), generator=g, device="cuda") * scale).to(torch.bfloat16)
        w = (torch.randn((d,), generator=g, device="cuda") * 0.5 + 1.0).to(torch.bfloat16)
        y = rms_norm(x, w, eps)
        ck.save(Path(a.out) / f"rms_{d}", {"rows": a.rows, "d": d, "eps": eps}, {"x": x, "w": w, "expected_y": y})
        print(f"d {d}: {a.rows} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
