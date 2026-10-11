#!/usr/bin/env python3
"""Four short code prompts at once against a running native server (greedy, 768 tokens each; a second argument
sends only that prompt): each reply's tokens, decode tok/s and token_sha, and the aggregate. Run where the server listens (the smoke's node), e.g.
  ssh spark-616b python3 - < tools/zig/qwen38fn/code4.py
With the server started under TF_QFN_ROUND_STATS=1 (serve_native.sh ENGINE_ENV), its log then says where the
rounds' time went (round_stats.zig)."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPTS = (
    "Write a Python function that parses an HTTP request line and returns method, path, version.",
    "Write a Python function that parses a CSV line with quoted fields and returns the list of fields.",
    "Write a Python function that parses an ISO 8601 date string and returns year, month and day.",
    "Write a Python function that parses a URL query string and returns a dict of its parameters.",
)


def ask(prompt: str, max_tokens: int, out: dict) -> None:
    body = json.dumps({"model": "qwen38fn", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}),
                                timeout=1800) as r:
        d = json.load(r)
    rt = d.get("tensorfold", {})
    out.update(tokens=d["usage"]["completion_tokens"], wall=time.time() - t0, tps=rt.get("tokens_per_second"),
               ttft=rt.get("time_to_first_token"), sha=rt.get("token_sha"))


def main() -> int:
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 768
    prompts = [PROMPTS[int(sys.argv[2])]] if len(sys.argv) > 2 else list(PROMPTS)  # one prompt alone
    outs = [{} for _ in prompts]
    ts = [threading.Thread(target=ask, args=(p, max_tokens, o)) for p, o in zip(prompts, outs)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    for i, o in enumerate(outs):
        print(f"code{i}: {o.get('tokens')} tokens, decode {o.get('tps') or 0:.1f} tok/s, ttft {o.get('ttft') or 0:.2f} s, "
              f"wall {o.get('wall', 0):.1f} s, sha {o.get('sha')}")
    total = sum(o.get("tokens", 0) for o in outs)
    print(f"aggregate: {total} tokens in {wall:.1f} s = {total / wall:.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
