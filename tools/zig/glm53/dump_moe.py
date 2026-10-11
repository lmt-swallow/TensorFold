#!/usr/bin/env python3
"""GLM-5.3-Flash's MoE block (layer 3: the sigmoid router with its correction bias, top-8 of 288 experts, the EXL3
routed experts, the BF16 shared expert, the weighted sum and the rank all-gather) from the Mia fork in its strict
setting, on both ranks of a two-rank split run on one Spark (ref.Pair), as a call-by-call fixture for the Zig engine
(capture.sh runs this inside the reference image, then packs the Triton launches).

The block's input is what the engine gives it: the FFN hyper-connection's normed row (`glue.hc_pre` with the
layer's ffn_hc and post_attention_layernorm) of streams holding the step's token embeddings. Steps: decode windows
of 1, 3 and 16 rows (the decode expert kernel, its fused epilogues and nc trellis loads) and prompt chunks of 2,048
and 37 rows (the prompt kernels on a 64-pair plan, other bits than decode rows'). Then `glue.hc_post` adds the
gathered partials to the streams, as layer_forward does.

Per rank r and step s, the arrays r{r}_{s}_<name>:
  ids, x_in                     the token ids and the streams the step starts from
  normed, xs                    the block's input row and its 64-group sums
  plan_members, plan_items,     the route's plan (experts.route) after the block
  plan_counts
  xg, xd                        the experts' rotated fp16 inputs: gate/up (one row a token when the layer's experts
                                share one suh) and down
  x_out                         the streams after hc_post
  {seq}_{fn}_{arg}_*            every traced call (ref.Tracer), the list in the manifest's "trace"

  python3 -B /w/tools/zig/glm53/dump_moe.py --tools /w/tools/zig --out /aot/out --model /model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref  # noqa: E402  (sets the strict environment before TensorFold is imported)

import torch  # noqa: E402

LAYER = 3
STEPS = (("d", 1), ("d", 3), ("d", 16), ("p", 2048), ("p", 37))
CAPACITY = 4096


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=LAYER, help="a MoE layer (3 to 44)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref.tools_on_path(a.tools)
    import capture_kernel as ck

    rec = ck.load_recorder(a.tools)
    from tensorfold.cuda import experts as grouped
    from tensorfold.families.glm5_next.cuda import exl3_mm, glue, qmm
    from tensorfold.families.glm5_next.cuda import forward as F

    pair = ref.Pair(rec)
    ws = [ref.load_rank(a.model, r, [a.layer], mtp=False, long_context=False) for r in (0, 1)]
    if ws[0].layers[0].moe is None or ws[0].layers[0].moe.shared is None:
        raise SystemExit(f"layer {a.layer} must be a MoE layer with EXL3 experts and a shared expert")
    for r, w in enumerate(ws):
        w.comm = pair.comm(r)
    (out / "weights.json").write_text(json.dumps({f"rank{r}": ref.weight_shas(w) for r, w in enumerate(ws)},
                                                 indent=1, sort_keys=True) + "\n")
    tr = ref.Tracer()
    tr.wrap_all(((glue, ("embed", "hc_pre", "hc_post", "router", "select", "swiglu", "combine", "combine_shared")),
                 (F, ("mm", "gather", "shared_front")), (qmm, ("group_sums",)), (exl3_mm, ("routed",)),
                 (grouped, ("route",))))
    ids = ref.ids_for(a.model, max(R for _, R in STEPS), 20261013)
    arrays = tr.arrays
    rows = max(R for k, R in STEPS if k == "d")
    prompt_rows = max(R for k, R in STEPS if k == "p")

    def rank_run(r: int) -> None:
        w = ws[r]
        c = w.cfg
        layer = w.layers[0]
        b = F.Buffers(w, rows, CAPACITY)
        pb = F.Buffers(w, prompt_rows, CAPACITY, prefill=True)
        for i, (kind, R) in enumerate(STEPS):
            s = f"s{i}"
            buf = pb if kind == "p" else b
            toks = ids[:R]
            buf.ids[:R].copy_(torch.tensor(toks, dtype=torch.int32))
            glue.embed(buf.ids[:R], w.embed, c.hidden, c.streams, buf.x[:R])
            h = layer.ffn_hc
            glue.hc_pre(buf.x[:R], h.fn, h.base, h.scale, layer.post_norm, buf.normed[:R], buf.xs[:R],
                        buf.post[:R], buf.comb[:R], buf.hcpart[:R], c.eps, c.hc_eps, c.hc_iters, prompt=buf.prefill)
            torch.cuda.synchronize()
            arrays[f"r{r}_{s}_ids"] = torch.tensor(toks, dtype=torch.int32)
            arrays[f"r{r}_{s}_x_in"] = buf.x[:R].clone()
            arrays[f"r{r}_{s}_normed"] = buf.normed[:R].clone()
            arrays[f"r{r}_{s}_xs"] = buf.xs[:R].clone()
            ref.Tracer.begin(r, s)
            with rec.scope(f"r{r}_{s}_{kind}{R}", detail=True):
                g = F.moe_block(layer, w, buf, R)
                glue.hc_post(buf.x[:R], buf.x[:R], g, buf.post[:R], buf.comb[:R])
            torch.cuda.synchronize()
            ref.Tracer.begin(r, None)
            P = R * (c.top_k + 1)
            plan = buf.plan
            arrays[f"r{r}_{s}_plan_members"] = plan.members[:P].clone()
            arrays[f"r{r}_{s}_plan_items"] = plan.items.clone()
            arrays[f"r{r}_{s}_plan_counts"] = plan.counts.clone()
            arrays[f"r{r}_{s}_xg"] = buf.exl3.xg[:P].clone()
            arrays[f"r{r}_{s}_xd"] = buf.exl3.xd[:P].clone()
            arrays[f"r{r}_{s}_x_out"] = buf.x[:R].clone()
            print(f"rank {r} step {i}: {kind} rows {R}: pick[0]={buf.pick[0].tolist()} "
                  f"x[0,:4]={buf.x[0, :4].float().tolist()}", flush=True)

    pair.run(rank_run)
    for i in range(len(STEPS)):
        if not ref.same(arrays[f"r0_s{i}_x_out"], arrays[f"r1_s{i}_x_out"]):
            raise SystemExit(f"step {i}: the ranks' streams differ after the MoE (their gathers summed alike)")
    params = {"layer": a.layer, "capacity": CAPACITY, "rows": rows, "prompt_rows": prompt_rows, "world": 2,
              "steps": [{"kind": k, "rows": R} for k, R in STEPS]}
    ck.save(out / "fixture", params, arrays, extra={"trace": tr.calls})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    ref.provenance(out, {"dump": "moe", "steps": STEPS, "calls": len(tr.calls)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
