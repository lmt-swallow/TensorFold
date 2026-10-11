#!/usr/bin/env python3
"""The whole GLM-5.3-Flash forward on the pair (45 layers, both ranks, NCCL between the Sparks) from the Mia fork in
its strict setting, as a fixture for the Zig engine's full forward: one rank of `forward_capture.sh` (this script
runs in the pinned image on each Spark, rank 0 on the head).

One request through the engine as served (`Engine.generate`, greedy, `draft`: MTP drafts as served; `--serial`: one
token a round): its prompt in 2,048-row chunks, then its decode windows. For each forward (`forward._compute`, in
order, numbered f{i}) the arrays r{rank}_f{i}_<name>:
  ids, pos                      the window's token ids and first position
  L{l}                          the streams after layer l (bf16 [rows, 4 D]; a prompt chunk's last 16 rows only)
  logits                        this rank's head logits of the window (bf16 [rows, vocab / 2]; decode windows only)
and the request's prompt and reply ids (prompt_ids, reply_ids) with the engine's stats. --windows bounds the forwards
recorded (later ones still run).

  python3 -B /w/tools/zig/glm53/b1/dump_forward.py --tools /w/tools/zig --out /aot/out --model /model \\
      --rank R --master 10.0.1.2 --port 29561
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ref  # noqa: E402  (sets the strict environment before TensorFold is imported)

import torch  # noqa: E402

PROMPT = ("Explain how a delta-rule recurrent layer and a sparse-attention layer can share one residual stream, "
          "then write a short Python function that checks whether two token lists are byte-identical.")
TAIL_ROWS = 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29561)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=3000, help="the prompt padded to this many tokens")
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--serial", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref.tools_on_path(a.tools)
    import capture_kernel as ck

    rec = ck.load_recorder(a.tools)
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.engine import Engine

    eng = Engine(Path(a.model), rank=a.rank, master=a.master, port=a.port, world=2, drafter=None, context=8192,
                 graphs=False)
    arrays: dict[str, torch.Tensor] = {}
    state = {"i": -1, "R": 0}
    r = a.rank
    compute = F._compute
    layer_forward = F.layer_forward

    def traced_compute(w, st, b, R, **kw):
        state["i"] += 1
        state["R"] = R
        i = state["i"]
        if i < a.windows:
            arrays[f"r{r}_f{i}_ids"] = b.ids[:R].clone()
            arrays[f"r{r}_f{i}_pos"] = torch.tensor([st.pos], dtype=torch.int64)
        with rec.scope(f"f{i}", detail=i < a.windows):
            got = compute(w, st, b, R, **kw)
        if i < a.windows and got is not None and not b.prefill:
            arrays[f"r{r}_f{i}_logits"] = got[:R].clone()
        return got

    def traced_layer(layer, w, st, b, R, *args, **kw):
        layer_forward(layer, w, st, b, R, *args, **kw)
        i = state["i"]
        if i < a.windows:
            lo = max(0, R - TAIL_ROWS) if b.prefill else 0
            arrays[f"r{r}_f{i}_L{layer.index}"] = b.x[lo:R].clone()

    F._compute = traced_compute
    F.layer_forward = traced_layer
    ids = ref.ids_for(a.model, a.prompt_tokens, 20261015)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    prompt = tok.encode(PROMPT).ids
    prompt = (ids[:max(0, a.prompt_tokens - len(prompt))] + prompt)[:a.prompt_tokens]
    reply: list[int] = []
    stats = eng.generate(prompt, a.tokens, None, lambda t: reply.extend(int(x) for x in t) or False,
                         draft=not a.serial)
    torch.cuda.synchronize()
    arrays[f"r{r}_prompt_ids"] = torch.tensor(prompt, dtype=torch.int32)
    arrays[f"r{r}_reply_ids"] = torch.tensor(reply, dtype=torch.int32)
    params = {"rank": r, "forwards": state["i"] + 1, "recorded": min(a.windows, state["i"] + 1),
              "tokens": a.tokens, "serial": a.serial, "stats": stats, "tail_rows": TAIL_ROWS}
    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    ref.provenance(out, {"dump": "forward", "rank": r, "stats": stats})
    print(f"rank {r}: {len(prompt)} prompt tokens, reply {reply[:16]}..., stats {stats}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
