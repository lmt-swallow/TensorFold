#!/usr/bin/env python3
"""Teacher-forced logits of the Mia fork on the pair, for the logit error between two KV settings (ref.KV:
TF_GLM53_REF_KV=bf16 or fp8): one rank of `forward_capture.sh` (EXTRA_ENV="-e TF_GLM53_REF_KV=fp8"), as
dump_forward.py runs it.

For each prompt of a bench.py result (--bench: its prompt_ids; --replies: the result whose first run's reply ids are
fed back, default the same), the engine decodes
serially (no drafts) and every sampled row's logits (this rank's vocabulary half, bf16) are kept while the token
fed back is the reply's, not the sampled one: both settings see the same context at every step, so their logits
compare position by position. The tokens each run would have sampled are kept too; on the setting that produced the
reply they equal it (the probe's own check). Writes <out>/fixture: r{rank}_{prompt}_logits [n, vocab / 2] bf16,
r{rank}_{prompt}_sampled [n] int32, r{rank}_{prompt}_forced [n] int32; compare two runs with logit_compare.py.

  python3 -B /w/tools/zig/glm53/b1/logit_probe.py --tools /w/tools/zig --out /aot/out --model /model \\
      --rank R --master 10.0.1.2 --port 29561 --bench /w/tools/zig/glm53/b1/results/<bench>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ref  # noqa: E402  (sets the strict environment before TensorFold is imported)

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29561)
    ap.add_argument("--bench", required=True, help="a bench.py result holding prompt_ids")
    ap.add_argument("--replies", help="the bench.py result whose first replies are fed back (default: --bench)")
    ap.add_argument("--context", type=int, default=131072)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref.tools_on_path(a.tools)
    import capture_kernel as ck
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine as Engine

    bench = json.loads(Path(a.bench).read_text())["prompts"]
    replies = json.loads(Path(a.replies or a.bench).read_text())["prompts"]
    eng = Engine(Path(a.model), rank=a.rank, master=a.master, port=a.port, world=2, drafter=None, context=a.context,
                 graphs=False)
    r = a.rank
    arrays: dict[str, torch.Tensor] = {}
    cur: dict = {}
    sample = D.Engine.sample

    def forced_sample(self, logits, positions, sampling, *args, **kw):
        got = sample(self, logits, positions, sampling, *args, **kw)    # every rank's collectives as served
        if kw.get("draft") or not cur:
            return got
        out_ids = []
        for i, p in enumerate(positions):
            k = int(p) - cur["prompt"]
            cur["logits"].append(logits[i].detach().clone())
            cur["sampled"].append(int(got[i]))
            out_ids.append(cur["reply"][k] if 0 <= k < len(cur["reply"]) else int(got[i]))
        return out_ids

    D.Engine.sample = forced_sample
    params: dict = {"rank": r, "kv": ref.KV, "prompts": {}}
    for name, entry in bench.items():
        prompt = [int(t) for t in entry["prompt_ids"]]
        reply = [int(t) for t in replies[name]["runs"][0]["ids"]]
        cur.update(prompt=len(prompt), reply=reply, logits=[], sampled=[])
        got: list[int] = []
        stats = eng.generate(prompt, len(reply), None, lambda t: got.extend(int(x) for x in t) or False, draft=False)
        torch.cuda.synchronize()
        arrays[f"r{r}_{name}_logits"] = torch.stack(cur["logits"]).cpu()
        arrays[f"r{r}_{name}_sampled"] = torch.tensor(cur["sampled"], dtype=torch.int32)
        arrays[f"r{r}_{name}_forced"] = torch.tensor(got, dtype=torch.int32)
        agree = sum(int(s == t) for s, t in zip(cur["sampled"], reply))
        params["prompts"][name] = {"prompt_tokens": len(prompt), "reply_tokens": len(reply), "rows": len(cur["sampled"]),
                                   "sampled_equal_reply": agree, "fed_equal_reply": got == reply[:len(got)],
                                   "stats": stats}
        print(f"rank {r} {name}: {len(prompt)} prompt tokens, {agree} / {len(reply)} sampled tokens equal the reply",
              flush=True)
        cur.clear()
    ck.save(out / "fixture", params, arrays)
    ref.provenance(out, {"dump": "logit_probe", "rank": r, "bench": a.bench, "replies": a.replies or a.bench})
    return 0


if __name__ == "__main__":
    sys.exit(main())
