#!/usr/bin/env python3
"""GLM-5.3-Flash-2 against GLM-5.3-Flash-1 on the bench prompts: what FP8 KV caches change in the outputs.

Two measures, each per prompt:
- free-running replies (two bench.py results, --base bf16 and --other fp8): the length of the common prefix of the
  token ids and the share of positions whose ids are equal;
- teacher-forced logits (two logit_probe.py captures, each with both ranks' fixture dirs): at every reply position
  of the base reply, both settings' full-vocabulary logits (the ranks' halves side by side) on the same context:
  top-1 agreement (the other setting's greedy token equals the base reply's), the largest and the RMS logit
  difference, and KL(base || other) of the softmax at temperature 1.

  python3 logit_compare.py --base b1-bf16.json --other b1-fp8.json \\
      --base-probe <bf16-work>/r0/out/fixture <bf16-work>/r1/out/fixture \\
      --other-probe <fp8-work>/r0/out/fixture <fp8-work>/r1/out/fixture --out compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(fixture: Path, name: str) -> np.ndarray:
    m = json.loads((fixture / "manifest.json").read_text())["arrays"][name]
    raw = np.fromfile(fixture / m["file"], dtype=np.uint8)
    if m["dtype"] == "bfloat16":
        v = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        v = raw.view({"int32": np.int32, "float32": np.float32}[m["dtype"]])
    return v.reshape(m["shape"])


def full_logits(dirs: list[Path], prompt: str) -> np.ndarray:
    return np.concatenate([load(d, f"r{r}_{prompt}_logits") for r, d in enumerate(dirs)], axis=1).astype(np.float64)


def free_run(base: list[int], other: list[int]) -> dict:
    n = min(len(base), len(other))
    prefix = next((i for i in range(n) if base[i] != other[i]), n)
    equal = sum(int(base[i] == other[i]) for i in range(n))
    return {"tokens": [len(base), len(other)], "common_prefix": prefix, "equal_positions": equal,
            "equal_share": equal / max(len(base), 1), "identical": base == other}


def forced(base: np.ndarray, other: np.ndarray, reply: list[int]) -> dict:
    n = min(len(base), len(other), len(reply))
    a, b = base[:n], other[:n]
    d = b - a
    la = a - a.max(1, keepdims=True)
    lb = b - b.max(1, keepdims=True)
    la -= np.log(np.exp(la).sum(1, keepdims=True))
    lb -= np.log(np.exp(lb).sum(1, keepdims=True))
    kl = (np.exp(la) * (la - lb)).sum(1)
    top_b = b.argmax(1)
    top_a = a.argmax(1)
    return {"rows": n, "base_top1_equals_reply": int((top_a == np.array(reply[:n])).sum()),
            "top1_agreement": float((top_b == np.array(reply[:n])).mean()),
            "max_abs_logit_diff": float(np.abs(d).max()), "rms_logit_diff": float(np.sqrt((d ** 2).mean())),
            "kl_mean": float(kl.mean()), "kl_max": float(kl.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--base-probe", nargs=2, type=Path)
    ap.add_argument("--other-probe", nargs=2, type=Path)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = json.loads(Path(a.base).read_text())["prompts"]
    other = json.loads(Path(a.other).read_text())["prompts"]
    result: dict = {}
    for name in base:
        if name not in other:
            continue
        reply = base[name]["runs"][0]["ids"]
        row = {"free_run": free_run(reply, other[name]["runs"][0]["ids"])}
        if a.base_probe and a.other_probe:
            row["forced"] = forced(full_logits(a.base_probe, name), full_logits(a.other_probe, name), reply)
        result[name] = row
        f = row["free_run"]
        line = f"{name:11s} prefix {f['common_prefix']:4d} equal {f['equal_share']:.3f}"
        if "forced" in row:
            g = row["forced"]
            line += (f" | forced top1 {g['top1_agreement']:.4f} max|dlogit| {g['max_abs_logit_diff']:.3f} "
                     f"rms {g['rms_logit_diff']:.4f} KL mean {g['kl_mean']:.2e} max {g['kl_max']:.2e}")
        print(line)
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
