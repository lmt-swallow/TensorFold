#!/usr/bin/env python3
"""Long-prompt workload against the Python server with the Triton recorder: prompts of the given token counts
(greedy, then 2 and 4 concurrent streams of the longest ones), so the prompt-chunk and long-position decode variants
get captured. Usage: long_workload.py <base> <model> <lengths comma-separated>"""
import json, sys, time, urllib.request, concurrent.futures

base, model = sys.argv[1], sys.argv[2]
lengths = [int(x) for x in sys.argv[3].split(",")]
filler = "The quick brown fox jumps over the lazy dog near the quiet river while the old clock in the village square keeps ticking through the long afternoon. "


def tokens(text):
    r = urllib.request.urlopen(urllib.request.Request(base + "/tokenize", data=json.dumps({"model": model, "prompt": text, "add_special_tokens": False}).encode(), headers={"Content-Type": "application/json"}), timeout=600)
    d = json.load(r)
    return len(d["tokens"])


def prompt_of(n):
    per = max(1, tokens(filler))
    body = filler * (n // per)
    return body + "\n\nQuestion: summarize the text above in one sentence."


def ask(n, max_tokens=48, temperature=0.0):
    req = {"model": model, "messages": [{"role": "user", "content": prompt_of(n)}], "max_tokens": max_tokens, "temperature": temperature}
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=7200)
    d = json.load(r)
    return n, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"], round(time.time() - t0, 1)


for n in lengths:
    print("solo", ask(n), flush=True)
    print("solo-sampled", ask(n, temperature=1.0), flush=True)
longest = lengths[-2:] if len(lengths) >= 2 else lengths
for streams in (2, 4):
    with concurrent.futures.ThreadPoolExecutor(streams) as ex:
        futs = [ex.submit(ask, longest[i % len(longest)], 64 + 16 * i) for i in range(streams)]
        for f in futs:
            print("conc", streams, f.result(), flush=True)
print("DONE", flush=True)
