#!/usr/bin/env python3
"""The reference's baselines on a served GLM-5.3-Flash (reference.sh start): fixed prompts, greedy, each sent twice
and once without drafts, the reply's token ids compared byte for byte, and what each reply's stats report.

Per prompt (code, prose, structured, a 64k-token needle): the two drafted replies' token ids (they must be equal: the
reference is deterministic), the undrafted reply's (it must be equal too: drafts only propose), the token sha
(sha256 of the ids as JSON, its first 16 hex digits, as the engine's stats print it), the decode rounds, tokens a
round, drafted / accepted tokens, the narrowest verify window, decode tok/s; then a 32k- and a 128k-token prompt
(max_tokens 1) for prefill tok/s. Writes the whole result as JSON (--out) and prints a table.

  python3 bench.py --url http://127.0.0.1:8888 --out b1-<date>.json [--tokens 512]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python function that merges overlapping intervals given as a list of (start, end) pairs, "
            "with type hints, a docstring, and three doctest examples. Then explain its time complexity.",
    "prose": "Describe, in four paragraphs, how a lighthouse keeper on a remote island might spend a winter "
             "without electricity, paying attention to the sounds, the light and the routine of each day.",
    "structured": "Return a JSON object listing the planets of the solar system in order from the Sun, each with "
                  "its name, its mean distance from the Sun in astronomical units and its number of known moons.",
}
NEEDLE = "The verification code for the archive is 4417-CEDAR-LOOM."
FILLER = ("The committee reviewed the quarterly logistics report, noted the delays at the northern depot, and "
          "asked for a revised schedule before the next meeting. ")


def post(url: str, body: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count(url: str, text: str) -> int:
    req = urllib.request.Request(url + "/tokenize", data=json.dumps({"prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return int(json.loads(r.read())["count"])


def long_text(url: str, tokens: int, needle: bool, tag: str) -> str:
    """Filler of about `tokens` tokens after an opening line of its own (`tag`: no two long prompts share a prefix,
    so no kept prompt state shortens a prefill), the needle in its middle when asked for."""
    per = count(url, FILLER * 64) / 64
    n = int(tokens / per)
    parts = [f"Document {tag}, filed for review. "] + [FILLER] * n
    if needle:
        parts.insert(n // 2, NEEDLE + " ")
    return "".join(parts)


def ask(url: str, content: str, max_tokens: int, draft: bool) -> dict:
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0,
            "return_token_ids": True}
    if not draft:
        body["draft"] = False
    t0 = time.time()
    r = post(url, body)
    tf = r.get("tensorfold") or {}
    ids = tf.get("token_ids") or []
    usage = r.get("usage") or {}
    return {"ids": ids, "sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], "wall_s": time.time() - t0,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "stats": {k: v for k, v in tf.items() if k != "token_ids"},
            "text": (r["choices"][0]["message"].get("content") or "")[-200:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--tag", default="1", help="the prefill prompts' own opening (a fresh one each run)")
    ap.add_argument("--only", help="one prompt (code, prose, structured, needle64k) and no prefill runs")
    a = ap.parse_args()
    prompts = dict(PROMPTS)
    if a.only != "code" and a.only not in PROMPTS:
        prompts["needle64k"] = long_text(a.url, 64_000, True, "N-64") + "\n\nWhat is the verification code for the archive?"
    if a.only:
        prompts = {a.only: prompts[a.only]}
    result: dict = {"url": a.url, "max_tokens": a.tokens, "prompts": {}, "prefill": {}}
    ok = True
    for name, text in prompts.items():
        runs = [ask(a.url, text, a.tokens, True), ask(a.url, text, a.tokens, True), ask(a.url, text, a.tokens, False)]
        same = runs[0]["ids"] == runs[1]["ids"]
        serial = runs[0]["ids"] == runs[2]["ids"]
        ok = ok and same and serial and bool(runs[0]["ids"])
        result["prompts"][name] = {"deterministic": same, "drafts_exact": serial, "runs": runs}
        s = runs[0]["stats"]
        dec = (len(runs[0]["ids"]) - 1) / s["decode_s"] if s.get("decode_s") else 0.0
        result["prompts"][name]["decode_tok_s"] = dec
        print(f"{name:11s} sha {runs[0]['sha']} same {same} serial {serial} tokens {len(runs[0]['ids'])} "
              f"prompt {runs[0]['prompt_tokens']} rounds {s.get('rounds')} tok/round {s.get('tokens_per_round')} "
              f"drafted {s.get('drafted')} accepted {s.get('accepted')} min_rows {s.get('min_rows')} "
              f"prefill {s.get('prefill_s')} s (cached {s.get('cached')}), decode {dec:.1f} tok/s", flush=True)
    for k in (() if a.only else (32_000, 128_000)):
        r = ask(a.url, long_text(a.url, k, False, f"P-{k}-{a.tag}") + "\n\nSummarize the text in one sentence.", 1, True)
        s = r["stats"]
        pre = s.get("prefill_s")
        rate = r["prompt_tokens"] / pre if pre else None
        result["prefill"][str(k)] = {"prompt_tokens": r["prompt_tokens"], "prefill_s": pre, "tok_s": rate,
                                     "cached": s.get("cached"),
                                     "wall_s": r["wall_s"], "stats": s}
        print(f"prefill {r['prompt_tokens']} tokens (cached {s.get('cached')}): {pre} s, {rate} tok/s "
              f"(wall {r['wall_s']:.1f} s)", flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("ALL DETERMINISTIC AND EXACT" if ok else "MISMATCH: see the table above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
