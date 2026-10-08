#!/usr/bin/env python3
"""A client workload that makes a DeepSeek-V4.1 `tensorfold serve` (Python fork, recorder shim installed) launch
every Triton variant the Zig engine will replay: prompts at the chunk and warm-up lengths, greedy and sampled,
thinking on and off, and 1..4 concurrent streams over several rounds (verify windows of every width, every
batched-drafter size). Prompt lengths are exact: the server's /tokenize renders the chat template and the filler
is grown until the count matches. Standard library only.

  python3 -B capture_workload.py --base http://127.0.0.1:8000/v1 [--lengths 1,17,33,...] [--max-tokens 64]
          [--streams 1,2,3,4] [--rounds 2] [--no-thinking-only] [--report workload.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_LENGTHS = [1, 17, 33, 131, 514, 1024, 2113, 4096, 9000]
FILLER = ["sea", "road", "lamp", "iron", "cloud", "stone", "river", "glass", "paper", "wheel", "north", "salt"]


def post(base: str, route: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(base.rstrip("/") + route, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def model_name(base: str) -> str:
    with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=60) as r:
        return json.loads(r.read().decode())["data"][0]["id"]


def count_tokens(base: str, model: str, messages: list, thinking: bool) -> int:
    """The rendered chat prompt's length from the server's vLLM-style /v1/tokenize (also served at /tokenize)."""

    return post(base, "/tokenize", {"model": model, "messages": messages,
                                    "chat_template_kwargs": {"enable_thinking": thinking}})["count"]


def prompt_of(base: str, model: str, target: int, thinking: bool, cache: dict) -> list:
    """Messages whose rendered prompt is exactly ``target`` tokens (one user turn of filler words), or the nearest
    length above the template's own tokens when ``target`` is below that."""

    key = (target, thinking)
    if key in cache:
        return cache[key]
    words = max(1, target - 12)
    msgs = [{"role": "user", "content": " ".join(FILLER[i % len(FILLER)] for i in range(words))}]
    for _ in range(12):
        n = count_tokens(base, model, msgs, thinking)
        if n == target or (n > target and words == 1):
            break
        words = max(1, words + (target - n))
        msgs = [{"role": "user", "content": " ".join(FILLER[i % len(FILLER)] for i in range(words))}]
    cache[key] = msgs
    return msgs


def one(base: str, model: str, msgs: list, max_tokens: int, temperature: float, thinking: bool, seed: int) -> dict:
    t0 = time.perf_counter()
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if temperature > 0:
        body.update(top_p=1.0)
    try:
        r = post(base, "/chat/completions", body)
        u = r.get("usage", {})
        return {"ok": True, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "finish": r["choices"][0].get("finish_reason"), "s": round(time.perf_counter() - t0, 2),
                "temperature": temperature, "thinking": thinking}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode()[:200]}", "temperature": temperature,
                "thinking": thinking}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--streams", default="1,2,3,4")
    ap.add_argument("--rounds", type=int, default=2, help="repeat the concurrent phase this many times")
    ap.add_argument("--no-thinking-only", action="store_true")
    ap.add_argument("--report", default="workload.json")
    a = ap.parse_args()
    base = a.base
    model = model_name(base)
    lengths = [int(x) for x in a.lengths.split(",") if x]
    thinkings = [False] if a.no_thinking_only else [False, True]
    cache: dict = {}
    log: list[dict] = []
    print(f"model {model}; lengths {lengths}; thinking {thinkings}", flush=True)

    # 1. every prompt length alone, greedy and sampled, thinking off and on (prompt chunk variants, solo windows)
    for n in lengths:
        for thinking in thinkings:
            msgs = prompt_of(base, model, n, thinking, cache)
            for temperature in (0.0, 1.0):
                r = one(base, model, msgs, a.max_tokens, temperature, thinking, seed=n)
                r.update(phase="solo", target=n)
                log.append(r)
                print(json.dumps(r), flush=True)

    # 2. concurrent streams: mixed lengths so the rounds hold 1..N streams at different positions and depths
    streams = [int(x) for x in a.streams.split(",") if x]
    mix = [17, 131, 514, 1024, 2113, 33, 4096, 1]
    for rnd in range(a.rounds):
        for s in streams:
            jobs = []
            for i in range(s):
                n = mix[(i + rnd * s) % len(mix)]
                thinking = thinkings[(i + rnd) % len(thinkings)]
                temperature = 1.0 if (i + rnd) % 2 else 0.0
                jobs.append((prompt_of(base, model, n, thinking, cache), a.max_tokens * (1 + i % 3), temperature,
                             thinking, 1000 * rnd + 10 * s + i, n))
            with cf.ThreadPoolExecutor(max_workers=s) as ex:
                futs = [ex.submit(one, base, model, m, mt, t, th, sd) for m, mt, t, th, sd, _ in jobs]
                for (m, mt, t, th, sd, n), f in zip(jobs, futs):
                    r = f.result()
                    r.update(phase="concurrent", streams=s, round=rnd, target=n)
                    log.append(r)
                    print(json.dumps(r), flush=True)
    failed = [r for r in log if not r["ok"]]
    with open(a.report, "w") as f:
        json.dump({"model": model, "requests": log, "failed": len(failed)}, f, indent=1)
    print(f"{len(log)} requests, {len(failed)} failed -> {a.report}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
