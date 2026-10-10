# bench_watch: the per-commit workload

`bench_watch.py` sends a fixed, realistic workload to the live two-rank server (rank 0's HTTP on spark-616b) and
records prefill and decode speed, the speculative counters, needle correctness, and every reply's `token_sha`. Every
request is greedy with a fixed prompt, so two runs of the same engine give the same shas; a changed sha is a change of
the model's outputs (a bit-exactness regression unless the change was meant to alter outputs, which none may).

It is meant for the goal configuration: `POOL=4194304 CONTEXT=1048576 PARALLEL=4` (4 streams of 1M in a 4M pool).
The result records the server's startup line, so a run against another configuration is visible as such.

## Running it

After your smoke, with the server up (`KEEP=1`) and the GPU granted, from your worktree on the Mac:

```sh
python3 tools/zig/dsv41/bench_watch.py --host spark-616b --label "$(git rev-parse --short HEAD)"
python3 tools/zig/dsv41/bench_watch.py --compare results/bench/<baseline>.json
```

The first command pipes the script to rank 0's node over ssh, runs it there against `http://127.0.0.1:8000`, and
writes `results/bench/<YYYYmmdd-HHMM>-<binary sha>.json` and `.md` in the worktree (the binary sha is the first 12 hex
digits of `sha256sum ~/bin/tensorfold-native` on rank 0). It exits 1 when a needle is wrong, a concurrent reply
differs from its sequential one, or either rank logged `differs between the ranks`, `OutOfStep`,
`MissingTritonVariant` or a broken lane during the run.

The second prints the newest result against the baseline: each metric with its change in percent (green when at least
5% better, red when at least 5% worse), and per reply `same` or a red `CHANGED old -> new` sha. It exits 1 on any sha
change. Commit the baseline the comparison is against, not every run.

Options: `--full` (the four concurrent needles at 256k instead of 64k: about 10 more minutes of prefill), `--only
short,agent,long,concurrent` (a subset; `concurrent`'s sequential-sha check needs `short` in the same run), `--label`
(free text kept in the result). On the node itself, drop `--host`; the result goes to `--out-dir` (default
`results/bench/` beside the script's checkout).

Run time on 47a5326 (baseline `results/bench/20261011-0453-0220ef9c6059.json`): 15 minutes with `--full`, of which
the four 256k needles take 10.5; about 6 minutes without it.

## Scenarios

| scenario | requests | what it watches |
|---|---|---|
| `short` | kit_bench.py decode set a (code, prose, structured), 384 tokens each, one after another | decode tok/s and draft acceptance on short prompts; the same prompts as kit_bench.py, so `text_sha` matches its earlier replies |
| `agent` | an opencode-like session: a fixed ~9k-token system prompt (rules, tool schemas, a project tree), 4 user turns, turns 2-4 carrying a ~1.5k-token tool result, each reply (192 tokens at most) appended to the history | prefill of a growing conversation through the kept prompts: `cached` should cover the earlier turns from turn 2 on |
| `long` | keep-needle.py's haystack (15-word list, the passphrase 7391-ALPHA-KITE at 60% depth) at 64k and 256k, 24 tokens; then the same haystack with a describe-it question, 256 tokens (`-depth` rows) | cold prefill speed at length, the answer, and decode speed at depth after a kept-prompt continuation |
| `concurrent` | the three `short` prompts plus code again, all four at once; then four needles at once (64k, or 256k with `--full`), each with its own haystack and passphrase | aggregate and per-stream decode, TTFT under contention, that batching leaves each reply's sha equal to its sequential one, and admission of four long prompts into the pool |

The haystack seeds differ per length (`7391 + length`, the concurrent ones `17391 + stream`), so no prompt is a prefix
of another and every `long` and `conc4-needle` prompt starts cold. A second run against the same server finds them
kept: the `cached` column then shows it and the prefill figures are not comparable. Compare runs taken right after a
server start (the smoke's restart is one).

## Columns

| column | meaning |
|---|---|
| `prompt`, `cached` | the reply's `usage.prompt_tokens` and `prompt_tokens_details.cached_tokens` (rows a kept prompt supplied) |
| `prefill s` | the server's `tensorfold.prefill_seconds` |
| `prefill tok/s` | (prompt - cached) / prefill s |
| `TTFT s` | client time from the send to the first content chunk (includes queueing behind other streams) |
| `tokens` | completion tokens (a needle's answer is about 9, so its decode tok/s mostly measures the wait behind the other streams' prompts) |
| `decode tok/s` | client-side (tokens - 1) / (last chunk - first chunk), kit_bench.py's definition |
| `rounds`, `acc`, `drafted` | the server's `speculative` counters: decode rounds, draft tokens accepted and proposed |
| `agg tok/s` | concurrent short: all completion tokens / the span from the first send to the last reply |
| `agg prefill tok/s` | concurrent needles: all prompt tokens / that span |
| `wall s` | that span |
| `ok` | needles: the passphrase is in the reply; concurrent short: the sha equals the sequential reply's |
| `token sha` | the server's `tensorfold.token_sha` (sha256 of the comma-joined token ids, 12 hex digits) |

The JSON keeps more per row (`server_ttft_s`, `server_decode_tps`, `text_sha`, `finish`, the needle's answer) and per
scenario rank 0's admission lines (`slot N takes extent [base, +size) (how) ...`: slot, extent, placement, streams and
kept prompts resident, free positions), its prompt lines (rows, start, ms, tok/s per admitted prompt) and its
kept-prompt continuations.
