#!/usr/bin/env python3
"""Needles past 131,072 positions against a running Qwen3.8-Flash-Next server (0.6.6 or native): bench_watch's
haystack (needle_messages) at each length, greedy, 24 tokens; prints and saves each reply's token_sha, prompt
tokens, prefill time and whether the phrase came back, so the two servers' runs compare by sha. Run on the node:

  python3 tools/zig/qwen38fn/long_needles.py --label native --out ~/qpo/needles-native.json [--lengths 200000,255000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsv41"))
from bench_watch import PHRASES, Client, needle_messages  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="qwen38fn")
    p.add_argument("--lengths", default="200000,255000")
    p.add_argument("--label", default="")
    p.add_argument("--out")
    a = p.parse_args()
    c = Client(a.base, a.model)
    rows = []
    for n in map(int, a.lengths.split(",")):
        r = c.chat(needle_messages(c, n, 7391 + n, PHRASES[0]), 24)
        row = {"length": n, "prompt_tokens": r.get("prompt_tokens"), "sha": r.get("token_sha"),
               "prefill_s": r.get("server_prefill_s"), "text": r.get("text"), "error": r.get("error"),
               "correct": PHRASES[0] in (r.get("text") or "")}
        if row["prompt_tokens"] and row["prefill_s"]:
            row["prefill_tps"] = round(row["prompt_tokens"] / row["prefill_s"], 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"label": a.label, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
