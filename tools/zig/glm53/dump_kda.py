#!/usr/bin/env python3
"""GLM-5.3-Flash's embeddings and decoder layer 0 (KDA attention, dense MLP, both hyper-connections) from the Mia
fork in its strict setting, on both ranks of a two-rank split run on one Spark (ref.Pair), as a call-by-call fixture
for the Zig engine (capture.sh runs this inside the reference image, then packs the Triton launches).

Layer 0's output depends on the token ids and its own state alone, so a load of layer 0 has the bits the 45-layer
model has there. One stream runs STEPS through `forward.stage`, `forward.compute` (logits off: no head) and
`forward.commit`, as `decode.prefill_chunk` and the drafted decoder run them: prompt chunks on the prompt buffers
(2,048 rows, then a 3-row tail), then decode windows of 1, 3 and 16 rows; a window that keeps fewer rows than it
ran is a drafted window whose tail was rejected, so its commit replays the KDA state (`kda.replay_layers`).

Per rank r and step s, the arrays r{r}_{s}_<name>:
  ids                                   the step's token ids (int32)
  rec_in, conv_in                       the KDA state (fp32 [32, 128, 128] a head) and conv window before the step
  x_out, hidden                         the residual streams after the layer and their mean (forward's b.hidden)
  rec_commit, conv_commit               the state and conv window the commit leaves
  {seq}_{fn}_{arg}_in / _out / _ret*    every traced call (ref.Tracer), the list in the manifest's "trace"
and the manifest's params: per step its kind, rows, keep and position. weights.json: the sha256 of every tensor each
rank's layer 0 holds (the loader's bytes).

  python3 -B /w/tools/zig/glm53/dump_kda.py --tools /w/tools/zig --out /aot/out --model /model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref  # noqa: E402  (sets the strict environment before TensorFold is imported)

import torch  # noqa: E402

# (kind, rows, keep): "p" a prompt chunk (keeps every row), "d" a decode window
STEPS = (("p", 2048, 2048), ("p", 3, 3), ("d", 1, 1), ("d", 3, 3), ("d", 16, 16), ("d", 16, 5), ("d", 3, 1),
         ("d", 1, 1), ("d", 7, 4))
CAPACITY = 4096


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref.tools_on_path(a.tools)
    import capture_kernel as ck

    rec = ck.load_recorder(a.tools)
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda import glue, qmm
    from tensorfold.families.glm5_next.cuda import kda as kda_mod

    pair = ref.Pair(rec)
    ws = [ref.load_rank(a.model, r, [0], mtp=False, long_context=False) for r in (0, 1)]
    if ws[0].layers[0].kind != "kda" or ws[0].layers[0].mlp is None:
        raise SystemExit("layer 0 must be a KDA layer with a dense MLP")
    for r, w in enumerate(ws):
        w.comm = pair.comm(r)
    (out / "weights.json").write_text(json.dumps({f"rank{r}": ref.weight_shas(w) for r, w in enumerate(ws)},
                                                 indent=1, sort_keys=True) + "\n")
    tr = ref.Tracer()
    tr.wrap_all(((glue, ("embed", "hc_pre", "hc_post", "rmsnorm", "swiglu", "stream_mean")),
                 (F, ("mm", "gather", "_shift_conv")), (qmm, ("group_sums",)),
                 (kda_mod, ("chain", "replay_layers"))))
    need = sum(k for _, _, k in STEPS[:-1]) + STEPS[-1][1]
    ids = ref.ids_for(a.model, need, 20261011)
    rows = max(r for kind, r, _ in STEPS if kind == "d")
    prompt_rows = max(r for kind, r, _ in STEPS if kind == "p")
    arrays = tr.arrays

    def rank_run(r: int) -> dict:
        w = ws[r]
        st = F.State(w, CAPACITY, rows)
        b = F.Buffers(w, rows, CAPACITY)
        pb = F.Buffers(w, prompt_rows, CAPACITY, prefill=True)
        pos = 0
        for i, (kind, R, keep) in enumerate(STEPS):
            s = f"s{i}"
            buf = pb if kind == "p" else b
            toks = ids[pos:pos + R]
            cur = st.cur[0]
            arrays[f"r{r}_{s}_ids"] = torch.tensor(toks, dtype=torch.int32)
            arrays[f"r{r}_{s}_rec_in"] = st.rec[cur, 0].clone()
            arrays[f"r{r}_{s}_conv_in"] = st.conv[0].clone()
            ref.Tracer.begin(r, s)
            with rec.scope(f"r{r}_{s}_{kind}{R}", detail=True):
                F.stage(w, st, buf, toks)
                F.compute(w, st, buf, R, logits=False, nch=F.chunks_for(st, R), host_pos=st.pos)
                F.commit(w, st, buf, R, keep)
            torch.cuda.synchronize()
            ref.Tracer.begin(r, None)
            arrays[f"r{r}_{s}_x_out"] = buf.x[:R].clone()
            arrays[f"r{r}_{s}_hidden"] = buf.hidden[:R].clone()
            arrays[f"r{r}_{s}_rec_commit"] = st.rec[st.cur[0], 0].clone()
            arrays[f"r{r}_{s}_conv_commit"] = st.conv[0].clone()
            print(f"rank {r} step {i}: {kind} rows {R} keep {keep} at {pos}: x[0,:4]="
                  f"{buf.x[0, :4].float().tolist()}", flush=True)
            pos += keep
        return {"end": pos}

    ends = pair.run(rank_run)
    for i in range(len(STEPS)):
        a0, a1 = arrays[f"r0_s{i}_x_out"], arrays[f"r1_s{i}_x_out"]
        if not ref.same(a0, a1):
            raise SystemExit(f"step {i}: the ranks' streams differ after layer 0 (their gathers summed alike)")
    params = {"layer": 0, "capacity": CAPACITY, "rows": rows, "prompt_rows": prompt_rows, "world": 2,
              "steps": [{"kind": k, "rows": R, "keep": keep} for k, R, keep in STEPS], "end": ends[0]["end"]}
    pos = 0
    for st_, (_, R, keep) in zip(params["steps"], STEPS):
        st_["pos"] = pos
        pos += keep
    ck.save(out / "fixture", params, arrays, extra={"trace": tr.calls})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    ref.provenance(out, {"dump": "kda", "steps": STEPS, "calls": len(tr.calls)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
