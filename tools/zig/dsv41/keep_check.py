#!/usr/bin/env python3
"""Kept-prompt scenarios (TF_DS_KEEP) against a running DeepSeek-V4.1 server, greedy, one request at a time.

Each scenario's later requests start with an earlier one's prompt, so a server with kept prompts on admits them at a
kept chunk boundary (usage's cached_tokens) and prefills only the rest:

- conversation: a ~9k-token system prompt and three turns, each appending the last reply and a new question;
- replay_cut: a prompt just past 4096 tokens (its kept boundary 4096 lies inside its own bounded-replay window), then
  the same prompt grown by a few hundred tokens;
- diverge: a ~6.7k-token prompt, then one sharing only its first ~5k tokens (the kept prompt's rows past the shared
  start, and its reply's, are not reused), then the first prompt again (it continues the first one's kept rows: its
  reply must equal the first reply).

Run it against a server with TF_DS_KEEP_CHECK=1 (every continuation is also prefilled from 0 and the server logs
"keep check: ... equal"), then against one with TF_DS_KEEP=0 and compare the replies' token shas (--compare): they
must be equal, request by request.

    python3 keep_check.py --base http://HOST:8000 --out keep.json [--compare keep0.json]
"""

import argparse
import json
import sys
import time
import urllib.request

WORDS = ["amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
         "lumen", "meadow", "nimbus", "onyx", "prairie", "quartz", "russet", "sierra", "tundra"]


def post(base: str, path: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def lines(seed: int, n: int, start: int = 0) -> list[str]:
    """Deterministic note lines (about 20 tokens each)."""

    out = []
    for i in range(start, start + n):
        a, b, c = WORDS[(seed * 7 + i) % 20], WORDS[(seed * 3 + i * 5) % 20], WORDS[(i * 11 + seed) % 20]
        out.append(f"Note {seed}.{i}: the {a} ledger lists {(seed * 7919 + i * 104729) % 100000} {b} crates bound for {c}.")
    return out


def count(base: str, model: str, messages: list) -> int:
    return post(base, "/v1/tokenize", {"model": model, "messages": messages})["count"]


def fit(base: str, model: str, make, lo: int, hi: int) -> int:
    """The least line count n whose prompt make(n) has lo <= tokens < hi."""

    n = max(1, lo // 25)
    while count(base, model, make(n)) < lo:
        n += max(1, (lo - count(base, model, make(n))) // 25)
    while count(base, model, make(n)) >= hi:
        n -= 1
    if not lo <= count(base, model, make(n)) < hi:
        raise SystemExit(f"cannot fit a prompt into [{lo}, {hi}) tokens")
    return n


def chat(base: str, model: str, name: str, messages: list, max_tokens: int) -> dict:
    t0 = time.time()
    r = post(base, "/v1/chat/completions", {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0})
    dt = time.time() - t0
    u = r.get("usage", {})
    tf = r.get("tensorfold", {}) or {}
    row = {
        "name": name,
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "completion_tokens": u.get("completion_tokens"),
        "token_sha": tf.get("token_sha"),
        "seconds": round(dt, 2),
        "content": r["choices"][0]["message"].get("content") or "",
    }
    print(f"{name:24s} prompt {row['prompt_tokens']:>6} cached {row['cached_tokens']:>6} tokens {row['completion_tokens']:>4} "
          f"sha {row['token_sha']} {row['seconds']:6.2f} s", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="an earlier run's --out: every reply's token sha must be equal")
    a = ap.parse_args()
    B, M, K = a.base, a.model, a.max_tokens
    rows = []

    # conversation: system prompt + growing history
    make_sys = lambda n: [{"role": "system", "content": "You keep the shipping notes below.\n" + "\n".join(lines(1, n))},
                          {"role": "user", "content": "Which crates go to tundra? One sentence."}]
    sys_lines = fit(B, M, make_sys, 8900, 9300)
    msgs = make_sys(sys_lines)
    for turn, question in enumerate(["Which crates go to tundra? One sentence.", "And to meadow? One sentence.",
                                     "Sum the two counts you named."]):
        if turn > 0:
            msgs = msgs + [{"role": "assistant", "content": rows[-1]["content"]}, {"role": "user", "content": question}]
        rows.append(chat(B, M, f"conversation.{turn + 1}", msgs, K))

    # replay_cut: just past 4096 tokens, then grown
    make_r = lambda n: [{"role": "user", "content": "Notes:\n" + "\n".join(lines(2, n)) + "\nName the first note's crate color."}]
    n = fit(B, M, make_r, 4110, 4220)
    rows.append(chat(B, M, "replay_cut.1", make_r(n), K))
    rows.append(chat(B, M, "replay_cut.2", make_r(n + 20), K))

    # diverge: a prompt, one sharing its first ~5k tokens, the first again
    make_d = lambda n: [{"role": "user", "content": "Notes:\n" + "\n".join(lines(3, n)) + "\nWhich note lists the most crates?"}]
    n = fit(B, M, make_d, 6600, 6900)
    shared = fit(B, M, make_d, 4950, 5150)
    first = make_d(n)
    rows.append(chat(B, M, "diverge.1", first, K))
    other = [{"role": "user", "content": "Notes:\n" + "\n".join(lines(3, shared) + lines(4, n - shared)) + "\nWhich note lists the most crates?"}]
    rows.append(chat(B, M, "diverge.2", other, K))
    rows.append(chat(B, M, "diverge.3", first, K))

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    ok = True
    by = {r["name"]: r for r in rows}
    if by["diverge.3"]["token_sha"] != by["diverge.1"]["token_sha"]:
        print("DIFFER: diverge.3 (the first prompt again) replied differently from diverge.1")
        ok = False
    if a.compare:
        with open(a.compare) as f:
            ref = {r["name"]: r for r in json.load(f)}
        for r in rows:
            if r["name"] in ref and ref[r["name"]]["token_sha"] != r["token_sha"]:
                print(f"DIFFER: {r['name']}: {r['token_sha']} here, {ref[r['name']]['token_sha']} in {a.compare}")
                ok = False
    print("replies equal" if ok else "replies DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
