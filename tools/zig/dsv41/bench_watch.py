#!/usr/bin/env python3
"""A fixed workload against the live DeepSeek-V4.1 server, so prefill and decode speed can be followed per commit at the
goal configuration (a 4M-position pool, 1M per stream, 4 streams). See bench_watch.md.

  python3 bench_watch.py --host spark-616b [--full]     from the Mac: runs itself on the node over ssh, writes
                                                        results/bench/<date>-<binary sha>.{json,md} here
  python3 bench_watch.py [--full] [--out-dir DIR]       on rank 0's node, against http://127.0.0.1:8000
  python3 bench_watch.py --compare OLD.json [NEW.json]  the table of NEW (default: the newest result) against OLD

Every request is greedy and every prompt fixed, so a reply's token_sha (the server's sha of its token ids) is the same
run to run unless the engine's outputs changed; --compare prints such a change in red. Standard library only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SCHEMA = 1
MARK = "@@bench_watch_result@@"

# kit_bench.py's decode set a (code / prose / structured), unchanged so its text shas stay comparable.
SHORT_PROMPTS = [
    ("code", "Write a complete Python implementation of a thread-safe LRU cache with per-entry TTL expiry. "
             "Include type hints, docstrings, and a set of unittest test cases covering eviction and expiry."),
    ("prose", "Write a long, vivid essay about the history of lighthouses: how they were built, how their lamps "
              "and lenses evolved, and the lives of the keepers who tended them through storms."),
    ("structured", "Return a JSON array of 15 fictional employees. Each object has the fields id (int), name, "
                   "department, title, salary (int), start_date (YYYY-MM-DD), manager_id (int or null) and "
                   "skills (array of strings). Output only the JSON, pretty-printed with two-space indents."),
]
SHORT_TOKENS = 384

# keep-needle.py's haystack generator (word list, sentence shape, 60% depth, question).
NEEDLE_WORDS = "river stone lantern market winter orchard copper signal harbor meadow quiet engine paper garden thunder".split()
PHRASES = ["7391-ALPHA-KITE", "5028-BRAVO-OTTER", "6614-CEDAR-FALCON", "8247-DELTA-HERON"]
NEEDLE_QUESTION = "What is the secret passphrase mentioned in the text above? Reply with the passphrase only."
DEPTH_QUESTION = ("Describe the text above in detail: its recurring words, its sentence shapes, and anything in it that "
                  "stands out. Write about 300 words.")
DEPTH_TOKENS = 256

AGENT_TURN_TOKENS = 192


def log(msg: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# -- HTTP -------------------------------------------------------------------------------------------------------------

class Client:
    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def _open(self, path: str, body: dict | None, timeout: float):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def post(self, path: str, body: dict, timeout: float = 600) -> dict:
        with self._open(path, body, timeout) as r:
            return json.loads(r.read())

    def get(self, path: str, timeout: float = 10) -> dict:
        with self._open(path, None, timeout) as r:
            return json.loads(r.read())

    def count(self, text: str) -> int:
        return int(self.post("/v1/tokenize", {"prompt": text, "add_special_tokens": False})["count"])

    def chat(self, messages: list, max_tokens: int, timeout: float = 7200) -> dict:
        """One streamed greedy chat request (thinking off). Client-side times from the send; the server's own figures
        from the last chunk's `tensorfold` and `speculative` objects."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        start = time.perf_counter()
        first = last = None
        usage, runtime, spec, pieces, finish = None, {}, {}, [], None
        try:
            with self._open("/v1/chat/completions", body, timeout) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    usage = chunk.get("usage") or usage
                    runtime = chunk.get("tensorfold") or runtime
                    spec = chunk.get("speculative") or spec
                    for choice in chunk.get("choices", []):
                        finish = choice.get("finish_reason") or finish
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            now = time.perf_counter()
                            first = now if first is None else first
                            last = now
                            pieces.append(piece)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
            return {"error": f"{type(e).__name__}: {e}", "start": start, "end": time.perf_counter()}
        end = time.perf_counter()
        text = "".join(pieces)
        n = int(usage["completion_tokens"]) if usage else None
        return {
            "start": start, "end": end, "text": text, "finish": finish,
            "prompt_tokens": int(usage["prompt_tokens"]) if usage else None,
            "cached_tokens": int(((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            "tokens": n,
            "ttft_s": (first - start) if first is not None else None,
            # first token excluded, as kit_bench.py
            "decode_tps": (n - 1) / (last - first) if n and first is not None and last > first else None,
            "server_ttft_s": runtime.get("time_to_first_token"),
            "server_prefill_s": runtime.get("prefill_seconds"),
            "server_decode_tps": runtime.get("tokens_per_second"),
            "token_sha": runtime.get("token_sha"),
            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:12],
            "rounds": spec.get("rounds"), "drafted": spec.get("drafted"), "accepted": spec.get("accepted"),
        }


def user(text: str) -> list:
    return [{"role": "user", "content": text}]


def r3(x):
    return round(x, 3) if isinstance(x, float) else x


def row(rid: str, scenario: str, r: dict, **extra) -> dict:
    """A result row: what --compare matches by `id`. `sha` is the bit-exactness check (None: not compared)."""
    keep = ("prompt_tokens", "cached_tokens", "tokens", "ttft_s", "decode_tps", "server_ttft_s", "server_prefill_s",
            "server_decode_tps", "rounds", "drafted", "accepted", "finish", "error")
    out = {"id": rid, "scenario": scenario, "sha": r.get("token_sha"), "text_sha": r.get("text_sha")}
    out.update({k: r3(r.get(k)) for k in keep if r.get(k) is not None})
    p, s = r.get("prompt_tokens"), r.get("server_prefill_s")
    if p and s:
        out["prefill_tps"] = round((p - (r.get("cached_tokens") or 0)) / s, 1)
    out.update({k: r3(v) for k, v in extra.items()})
    return out


# -- workloads ----------------------------------------------------------------------------------------------------------

def needle_messages(c: Client, tokens: int, seed: int, phrase: str, question: str = NEEDLE_QUESTION) -> list:
    """keep-needle.py's prompt: ~`tokens` tokens of filler from Random(seed), the phrase at 60% depth."""
    rng = random.Random(seed)

    def text(n):
        return " ".join(" ".join(rng.choice(NEEDLE_WORDS) for _ in range(rng.randint(8, 16))).capitalize() + "."
                        for _ in range(n))

    per = c.count(text(200)) / 200
    sents = int(tokens / per)
    hay = [text(1) for _ in range(sents)]
    hay.insert(int(sents * 0.6), f"Remember this: the secret passphrase is {phrase}.")
    return user(" ".join(hay) + "\n\n" + question)


AGENT_RULES = """You are opencode, an interactive CLI agent that helps users with software engineering tasks. Use the
instructions below and the tools available to you to assist the user.

# Tone and style
Be concise, direct, and to the point. Answer the user's question directly, without elaboration, explanation, or
details unless asked. Use Github-flavored markdown for formatting. Only use tools to complete tasks; never use tools
or code comments as a means to communicate with the user.

# Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries
and utilities, and follow existing patterns. Never assume that a given library is available, even if it is well known.
Always follow security best practices. Never introduce code that exposes or logs secrets and keys.

# Doing tasks
1. Use the search tools to understand the codebase and the user's query.
2. Implement the solution using all tools available to you.
3. Verify the solution if possible with the project's tests. Never assume a specific test framework or test script.
4. Run the lint and typecheck commands if they were provided to you.

# Tool usage policy
When doing file search, prefer the task tool. You have the capability to call multiple tools in a single response.
Tool results and user messages may include <system-reminder> tags; they bear no direct relation to the specific tool
results or user messages in which they appear.
"""

AGENT_TOOLS = ["bash", "read", "write", "edit", "glob", "grep", "list", "patch", "todowrite", "todoread", "webfetch",
               "task"]

AGENT_TASKS = [
    "The tests in tests/test_scheduler.py fail with a KeyError after my last change. Find the cause and propose a fix.",
    "Here is the output of reading src/scheduler/queue.py. Explain what the retry path does and where the KeyError "
    "comes from.",
    "Here is the output of running the test suite after applying your fix. Summarize which tests still fail and why.",
    "Here is git diff of the working tree. Write a concise commit message for it and list any follow-up work.",
]


def agent_system(seed: int = 4100) -> str:
    """A fixed ~9k-token opencode-like system prompt: rules, tool schemas, an environment block, a project tree."""
    rng = random.Random(seed)
    parts = [AGENT_RULES, "\n# Tools\n"]
    for name in AGENT_TOOLS:
        params = {f"{w}_{i}": {"type": rng.choice(["string", "integer", "boolean", "array"]),
                               "description": " ".join(rng.choice(NEEDLE_WORDS + AGENT_TOOLS) for _ in range(14))}
                  for i, w in enumerate(rng.sample(NEEDLE_WORDS, 6))}
        parts.append(json.dumps({"name": name, "description": f"The {name} tool. " + " ".join(
            rng.choice(NEEDLE_WORDS) for _ in range(40)), "parameters": {"type": "object", "properties": params}},
            indent=1) + "\n")
    parts.append("\n<env>\nWorking directory: /home/dev/project\nIs directory a git repo: yes\nPlatform: linux\n</env>\n")
    parts.append("\n<project>\n")
    for i in range(440):
        depth = rng.randint(1, 4)
        path = "/".join(rng.choice(NEEDLE_WORDS) for _ in range(depth))
        parts.append(f"src/{path}_{i}.{rng.choice(['py', 'ts', 'go', 'md', 'json'])}\n")
    parts.append("</project>\n")
    return "".join(parts)


def tool_result(turn: int, seed: int = 4200) -> str:
    """A fixed synthetic tool output for a turn (~1.5k tokens): a source file, a test log, a diff."""
    rng = random.Random(seed + turn)
    kind = ["read", "test", "diff"][(turn - 1) % 3]
    lines = []
    for i in range(80):
        a, b = rng.choice(NEEDLE_WORDS), rng.choice(NEEDLE_WORDS)
        if kind == "read":
            lines.append(f"{i + 1:4d}\t    {a}_{b} = self.{b}.get({rng.randint(0, 99)}) or {a}_{i}()")
        elif kind == "test":
            lines.append(f"tests/test_{a}.py::test_{b}_{i} {rng.choice(['PASSED', 'PASSED', 'PASSED', 'FAILED'])}")
        else:
            lines.append(f"{rng.choice('+-')}    {a}_{b}({rng.randint(0, 9)}, {b}={a!r})")
    return f"<tool_result tool=\"{['read', 'bash', 'bash'][(turn - 1) % 3]}\">\n" + "\n".join(lines) + "\n</tool_result>"


# -- scenarios ----------------------------------------------------------------------------------------------------------

def scenario_short(c: Client) -> list:
    rows = []
    for name, prompt in SHORT_PROMPTS:
        r = c.chat(user(prompt), SHORT_TOKENS)
        rows.append(row(f"short/{name}", "short", r))
        log(f"short/{name}: {r3(r.get('decode_tps'))} tok/s, {r.get('rounds')} rounds, "
            f"{r.get('accepted')}/{r.get('drafted')}, sha {r.get('token_sha')}")
    return rows


def scenario_agent(c: Client) -> list:
    rows = []
    messages = [{"role": "system", "content": agent_system()}]
    for turn, task in enumerate(AGENT_TASKS, 1):
        content = task if turn == 1 else task + "\n\n" + tool_result(turn - 1)
        messages.append({"role": "user", "content": content})
        r = c.chat(messages, AGENT_TURN_TOKENS)
        rows.append(row(f"agent/turn{turn}", "agent", r))
        log(f"agent/turn{turn}: prompt {r.get('prompt_tokens')} (cached {r.get('cached_tokens')}), prefill "
            f"{r3(r.get('server_prefill_s'))} s, {r3(r.get('decode_tps'))} tok/s")
        if r.get("error"):
            break
        messages.append({"role": "assistant", "content": r["text"]})
    return rows


def needle_row(rid: str, scenario: str, r: dict, phrase: str, **extra) -> dict:
    ok = phrase in (r.get("text") or "")
    return row(rid, scenario, r, correct=ok, answer=(r.get("text") or "").strip()[-40:], **extra)


def scenario_long(c: Client, lengths: list) -> list:
    rows = []
    for n in lengths:
        seed = 7391 + n          # distinct per length, so one haystack is not a prefix of another
        msgs = needle_messages(c, n, seed, PHRASES[0])
        r = c.chat(msgs, 24)
        rows.append(needle_row(f"long/{n // 1024}k", "long", r, PHRASES[0]))
        log(f"long/{n // 1024}k: {r.get('prompt_tokens')} tok in {r3(r.get('server_prefill_s'))} s, "
            f"{'CORRECT' if rows[-1]['correct'] else 'WRONG'}")
        # the same haystack, another question: continues the kept prompt and decodes at depth
        msgs = needle_messages(c, n, seed, PHRASES[0], DEPTH_QUESTION)
        r = c.chat(msgs, DEPTH_TOKENS)
        rows.append(row(f"long/{n // 1024}k-depth", "long", r))
        log(f"long/{n // 1024}k-depth: cached {r.get('cached_tokens')}, {r3(r.get('decode_tps'))} tok/s")
    return rows


def run_concurrent(jobs: list) -> tuple[list, float, float]:
    """Runs the (fn) jobs at once behind a barrier; returns their results and the wall span."""
    res: list = [None] * len(jobs)
    barrier = threading.Barrier(len(jobs))

    def go(i):
        barrier.wait()
        res[i] = jobs[i]()

    ths = [threading.Thread(target=go, args=(i,)) for i in range(len(jobs))]
    [t.start() for t in ths]
    [t.join() for t in ths]
    t0 = min(r["start"] for r in res)
    t1 = max(r["end"] for r in res)
    return res, t0, t1


def scenario_concurrent(c: Client, needle_len: int, short_shas: dict, streams: int = 4) -> list:
    """`streams` short requests at once (decode set a's prompts in turn: code, prose, structured, code, ...), then
    four needles at once. The default of 4 is the goal configuration's; the ids name the stream count."""
    rows = []
    jobs = [SHORT_PROMPTS[i % len(SHORT_PROMPTS)] for i in range(streams)]
    sc = f"conc{streams}-short"
    res, t0, t1 = run_concurrent([lambda p=p: c.chat(user(p), SHORT_TOKENS) for _, p in jobs])
    total = sum(r.get("tokens") or 0 for r in res)
    for i, ((name, _), r) in enumerate(zip(jobs, res)):
        seq = short_shas.get(f"short/{name}")
        rows.append(row(f"{sc}/{i}-{name}", sc, r,
                        same_as_sequential=(seq == r.get("token_sha")) if seq else None))
    per = [r.get("decode_tps") for r in res if r.get("decode_tps")]
    rows.append({"id": f"{sc}/aggregate", "scenario": sc, "sha": None, "wall_s": r3(t1 - t0),
                 "tokens": total, "aggregate_tps": r3(total / (t1 - t0)),
                 "decode_tps": r3(sum(per) / len(per)) if per else None})
    log(f"{sc}: {r3(total / (t1 - t0))} tok/s aggregate, per stream "
        f"{[r3(r.get('decode_tps')) for r in res]}")

    msgs = [needle_messages(c, needle_len, 17391 + i, PHRASES[i]) for i in range(4)]
    res, t0, t1 = run_concurrent([lambda m=m: c.chat(m, 24) for m in msgs])
    prompt_total = sum(r.get("prompt_tokens") or 0 for r in res)
    for i, r in enumerate(res):
        rows.append(needle_row(f"conc4-needle/{i}", "conc4-needle", r, PHRASES[i], done_s=r3(r["end"] - t0)))
    rows.append({"id": "conc4-needle/aggregate", "scenario": "conc4-needle", "sha": None, "length": needle_len,
                 "wall_s": r3(t1 - t0), "prompt_tokens": prompt_total, "aggregate_prefill_tps": r3(prompt_total / (t1 - t0)),
                 "correct": all(x.get("correct") for x in rows if x["scenario"] == "conc4-needle" and "correct" in x)})
    log(f"conc4-needle {needle_len // 1024}k: {prompt_total} tokens in {r3(t1 - t0)} s, "
        f"{sum(1 for x in rows if x['scenario'] == 'conc4-needle' and x.get('correct'))}/4 correct")
    return rows


# -- the server's side: config, binary, rank 0's admission lines --------------------------------------------------------

ADMIT = re.compile(r"slot (\d+) takes extent \[(\d+), \+(\d+)\) \((\w+)\) for a prompt of (\d+); (\d+) streams and "
                   r"(\d+) kept prompts resident, (\d+) of (\d+) positions free")
PREFILL = re.compile(r"rank 0: a prompt of (\d+) rows from (\d+) in (\d+) chunks: (\d+) ms, (\d+) tok/s")
CONTINUED = re.compile(r"kept prompt (\d+): continued at (\d+) of (\d+) \((\w+)\)")
BAD = re.compile(r"differs between the ranks|OutOfStep|MissingTritonVariant|the lane is broken")


def sh(cmd: str, timeout: float = 60) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def docker_log(container: str, since: float, until: float | None = None, host: str | None = None) -> list:
    cmd = f"docker logs --since {since:.0f}" + (f" --until {until + 1:.0f}" if until else "") + f" {container} 2>&1"
    if host:
        cmd = f"ssh -4 -o BatchMode=yes -o ConnectTimeout=5 {host} {shlex.quote(cmd)}"
    return sh(cmd, 120).splitlines()


def server_side(lines: list) -> dict:
    admissions = []
    for ln in lines:
        m = ADMIT.search(ln)
        if m:
            g = [m.group(i) for i in range(1, 10)]
            admissions.append({"slot": int(g[0]), "base": int(g[1]), "size": int(g[2]), "how": g[3],
                               "prompt": int(g[4]), "streams": int(g[5]), "kept": int(g[6]), "free": int(g[7]),
                               "pool": int(g[8])})
    prefills = [{"rows": int(m.group(1)), "from": int(m.group(2)), "ms": int(m.group(4)), "tps": int(m.group(5))}
                for m in map(PREFILL.search, lines) if m]
    continued = [{"kept": int(m.group(1)), "at": int(m.group(2)), "of": int(m.group(3)), "how": m.group(4)}
                 for m in map(CONTINUED.search, lines) if m]
    return {"admissions": admissions, "prefills": prefills, "continued": continued,
            "max_streams": max((a["streams"] for a in admissions), default=0),
            "min_free": min((a["free"] for a in admissions), default=None)}


def binary_sha() -> str:
    out = sh("sha256sum $HOME/bin/tensorfold-native 2>/dev/null")
    return out[:12] if out else "unknown"


def server_config() -> str:
    for ln in reversed(sh("docker logs tfz0 2>&1 | grep -E 'streams at once|serving dsv41' | tail -2").splitlines()):
        if "streams at once" in ln:
            return ln.split("): ", 1)[-1].strip()
    return ""


# -- a run --------------------------------------------------------------------------------------------------------------

def run(a) -> dict:
    c = Client(a.base, a.model)
    c.get("/v1/models")
    started = time.time()
    out = {"schema": SCHEMA, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
           "binary_sha": binary_sha(), "config": server_config(), "full": a.full, "label": a.label,
           "base": a.base, "scenarios": {}, "rows": []}
    log(f"binary {out['binary_sha']}; {out['config']}")
    wanted = a.only.split(",") if a.only else ["short", "agent", "long", "concurrent"]
    long_lengths = [65536, 262144]
    needle_len = 262144 if a.full else 65536
    short_shas: dict = {}
    for name in wanted:
        t0 = time.time()
        if name == "short":
            rows = scenario_short(c)
            short_shas = {r["id"]: r.get("sha") for r in rows}
        elif name == "agent":
            rows = scenario_agent(c)
        elif name == "long":
            rows = scenario_long(c, long_lengths)
        elif name == "concurrent":
            rows = scenario_concurrent(c, needle_len, short_shas, a.streams)
        else:
            raise SystemExit(f"unknown scenario {name}")
        t1 = time.time()
        time.sleep(1.5)  # rank 0's lines of the last request
        side = server_side(docker_log("tfz0", t0 - 1, t1))
        out["scenarios"][name] = {"seconds": round(t1 - t0, 1), "rank0": side}
        out["rows"].extend(rows)
    ended = time.time()
    bad = [ln for ln in docker_log("tfz0", started - 1, ended) if BAD.search(ln)]
    bad += [f"rank 1: {ln}" for ln in docker_log("tfz1", started - 1, ended, host=a.rank1) if BAD.search(ln)]
    out["rank_errors"] = bad[:20]
    out["seconds"] = round(ended - started, 1)
    return out


# -- report -------------------------------------------------------------------------------------------------------------

COLUMNS = [("prompt_tokens", "prompt"), ("cached_tokens", "cached"), ("server_prefill_s", "prefill s"),
           ("prefill_tps", "prefill tok/s"), ("ttft_s", "TTFT s"), ("tokens", "tokens"), ("decode_tps", "decode tok/s"),
           ("rounds", "rounds"), ("accepted", "acc"), ("drafted", "drafted"), ("aggregate_tps", "agg tok/s"),
           ("aggregate_prefill_tps", "agg prefill tok/s"), ("wall_s", "wall s")]
# higher is better for these; the others (times) lower is better
HIGHER = {"prefill_tps", "decode_tps", "aggregate_tps", "aggregate_prefill_tps", "accepted"}
TIMES = {"server_prefill_s", "ttft_s", "wall_s"}


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def markdown(res: dict) -> str:
    lines = [f"# bench_watch {res['started']}", "",
             f"binary `{res['binary_sha']}`; {res.get('config') or 'config unknown'}; "
             f"{'--full' if res.get('full') else 'default'} run, {res['seconds']} s"
             + (f"; label {res['label']}" if res.get("label") else ""), ""]
    cols = [(k, h) for k, h in COLUMNS if any(k in r for r in res["rows"])]
    lines.append("| id | " + " | ".join(h for _, h in cols) + " | ok | token sha |")
    lines.append("|---|" + "---:|" * len(cols) + "---|---|")
    for r in res["rows"]:
        ok = r.get("correct", r.get("same_as_sequential"))
        lines.append(f"| {r['id']} | " + " | ".join(fmt(r.get(k)) for k, _ in cols) +
                     f" | {fmt(ok)} | {r.get('sha') or ''} |")
    lines.append("")
    for name, s in res["scenarios"].items():
        r0 = s["rank0"]
        adm = ", ".join(f"slot {x['slot']} +{x['size']} {x['how']} ({x['streams']} streams, {x['free']} free)"
                        for x in r0["admissions"][:8])
        lines.append(f"- {name}: {s['seconds']} s; {len(r0['admissions'])} admissions, at most {r0['max_streams']} "
                     f"streams, least free {r0['min_free']}" + (f"; {adm}" if adm else ""))
    lines.append("- rank errors: " + ("; ".join(res["rank_errors"]) if res["rank_errors"] else "none"))
    return "\n".join(lines) + "\n"


RED, GREEN, RESET = "\033[31m", "\033[32m", "\033[0m"


def compare(old: dict, new: dict, color: bool) -> tuple[str, int]:
    """The table of new against old: each metric as new (delta %), sha changes in red. Returns (text, sha changes)."""
    paint = (lambda s, c: f"{c}{s}{RESET}") if color else (lambda s, c: s)
    olds = {r["id"]: r for r in old["rows"]}
    keys = ["server_prefill_s", "prefill_tps", "ttft_s", "decode_tps", "aggregate_tps", "aggregate_prefill_tps",
            "wall_s"]
    keys = [k for k in keys if any(k in r for r in new["rows"])]
    out = [f"old {old.get('started')} binary {old.get('binary_sha')}  ->  new {new.get('started')} binary "
           f"{new.get('binary_sha')}", ""]
    header = ["id"] + [dict(COLUMNS)[k] for k in keys] + ["sha"]
    table = [header]
    changed = 0
    for r in new["rows"]:
        o = olds.get(r["id"], {})
        cells = [r["id"]]
        for k in keys:
            v, w = r.get(k), o.get(k)
            if v is None:
                cells.append("")
                continue
            if isinstance(v, (int, float)) and isinstance(w, (int, float)) and w:
                d = (v - w) / w * 100
                good = d > 0 if k in HIGHER else d < 0
                txt = f"{fmt(v)} ({d:+.1f}%)"
                cells.append(paint(txt, GREEN if good else RED) if abs(d) >= 5 else txt)
            else:
                cells.append(fmt(v))
        if r.get("sha") is None:
            s = ""
        elif not o:
            s = f"{r['sha']} (new)"
        elif o.get("sha") == r.get("sha"):
            s = "same"
        else:
            s = paint(f"CHANGED {o.get('sha')} -> {r['sha']}", RED)
            changed += 1
        if r.get("correct") is False:
            s = (s + " " if s else "") + paint("WRONG", RED)
        cells.append(s)
        table.append(cells)
    strip = re.compile(r"\033\[\d+m")
    widths = [max(len(strip.sub("", row[i])) for row in table) for i in range(len(header))]
    for row_ in table:
        out.append("  ".join(cell + " " * (widths[i] - len(strip.sub("", cell))) for i, cell in enumerate(row_)))
    errs = new.get("rank_errors") or []
    out.append("")
    out.append(paint(f"{changed} token sha change(s): bit-exactness regression", RED) if changed
               else "token shas: all same")
    if errs:
        out.append(paint(f"rank errors: {len(errs)}", RED))
    return "\n".join(out), changed


def save(res: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, time.strftime("%Y%m%d-%H%M", time.strptime(res["started"][:19], "%Y-%m-%dT%H:%M:%S"))
                        + f"-{res['binary_sha']}")
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    with open(stem + ".md", "w") as f:
        f.write(markdown(res))
    return stem


def default_out_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results",
                                         "bench"))


def remote(a, argv: list) -> dict:
    """Runs this file on `a.host` over ssh (its source on stdin) and returns the result it prints."""
    src = open(os.path.abspath(__file__), "rb").read()
    args = [x for x in argv if x not in ("--host", a.host)]
    cmd = ["ssh", "-4", "-o", "BatchMode=yes", a.host, "python3 - --emit " + " ".join(shlex.quote(x) for x in args)]
    p = subprocess.run(cmd, input=src, stdout=subprocess.PIPE, timeout=4 * 3600)
    text = p.stdout.decode()
    if MARK not in text:
        raise SystemExit(f"no result from {a.host} (exit {p.returncode})")
    return json.loads(text.split(MARK, 1)[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--full", action="store_true", help="the concurrent needles at 256k instead of 64k")
    p.add_argument("--only", help="comma-separated scenarios: short,agent,long,concurrent")
    p.add_argument("--streams", type=int, default=4,
                   help="short requests the concurrent scenario sends at once (default 4, the goal configuration's)")
    p.add_argument("--label", default="", help="free text kept in the result (e.g. the commit)")
    p.add_argument("--host", help="run on this node over ssh and save the result here")
    p.add_argument("--rank1", default="10.0.1.2", help="rank 1's ssh address, for its error lines (from rank 0)")
    p.add_argument("--out-dir", default=default_out_dir())
    p.add_argument("--compare", metavar="OLD.json")
    p.add_argument("new", nargs="?", help="with --compare: the new result (default: the newest in --out-dir)")
    p.add_argument("--emit", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    color = sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"

    if a.compare:
        new_path = a.new or max(glob.glob(os.path.join(a.out_dir, "*.json")), key=os.path.getmtime)
        text, changed = compare(json.load(open(a.compare)), json.load(open(new_path)), color)
        print(text)
        sys.exit(1 if changed else 0)
    if a.host:
        res = remote(a, sys.argv[1:])
    else:
        res = run(a)
        if a.emit:
            print(MARK + json.dumps(res), flush=True)
            return
    stem = save(res, a.out_dir)
    print(markdown(res))
    print(f"saved {stem}.json and {stem}.md")
    bad = res["rank_errors"] or any(r.get("correct") is False or r.get("same_as_sequential") is False
                                    for r in res["rows"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
