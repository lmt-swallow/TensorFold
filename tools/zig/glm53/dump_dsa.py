#!/usr/bin/env python3
"""GLM-5.3-Flash's first DSA layer (layer 3: the indexer, pooled keys, top-512 pool selection, the latent cache's
absorb / attention / expand, then its MoE) and the MTP layer (layer 45 on its own caches, the draft head) from the
Mia fork in its strict setting, on both ranks of a two-rank split run on one Spark (ref.Pair), as a call-by-call
fixture for the Zig engine (capture.sh runs this inside the reference image, then packs the Triton launches).

The layer's input streams are the embedding of the step's tokens in every hyper-connection stream (what layer 0
reads; layer 3's own input would need layers 0-2): the kernels' bits are checked, not the model's values. The caches
are the served engine's (decode.Engine: one shared pool, the indexer's key and gate ring of 2,112 rows a stream).

Three runs, per rank r, arrays r{r}_{step}_<name> (each traced call as ref.Tracer writes it, the list in the
manifest's "trace") and the manifest's params ("steps": kind, rows, keep, position of each):
  a  one stream from position 0: a 2,048-row prompt chunk (dense: every row before the dense limit 2,051), decode
     windows across the limit (3 rows from 2,049: two dense rows, one sparse) and past it (1, 3, 16 rows, drafted
     windows keeping fewer), a 2,048-row chunk past the limit (the prompt selection, one-pass sparse attention), a
     37-row prompt tail, decode windows again. Per step also x_in / x_out (the streams around the layer) and
     fnormed (the final norm of the streams' mean, the MTP head's input in m).
  m  the MTP layer on run a's rows: the prompt chunk absorbed (2,047 rows, the first position's embedding zeroed),
     decode absorbs, chained one-row drafts and their rewind (decode.absorb / draft), a 2,048-row absorb past the
     dense limit, a sparse draft. Per step also logits (the draft head's) and draft (the sampled token).
  l  synthetic long contexts: latents and pooled keys of positions before P drawn from a seeded generator, then
     decode windows at P = 40,000 (the split top-k selection, past 8,192 pools) and P = 600,000 (past 524,288);
     the drawn caches as l_latent0 and l_pk0 (layer 3's latent rows and pooled keys before the first window).

weights.json: the sha256 of every tensor each rank's layers 3 and 45, head and draft head hold.

  python3 -B /w/tools/zig/glm53/dump_dsa.py --tools /w/tools/zig --out /aot/out --model /model
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
RUN_A = (("p", 2048, 2048), ("d", 1, 1), ("d", 3, 2), ("d", 16, 16), ("d", 16, 3), ("p", 2048, 2048),
         ("p", 37, 37), ("d", 1, 1), ("d", 3, 3), ("d", 16, 9))
CAPACITY = 8192
LONG = ((40_000, (("d", 1, 1), ("d", 3, 2), ("d", 16, 16))), (600_000, (("d", 1, 1), ("d", 16, 4))))
LONG_CAPACITY = 600_064
MAX_ROWS, PREFILL_ROWS = 16, 2048


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--skip-long", action="store_true", help="leave out run l (its caches take 1.2 GiB a rank)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref.tools_on_path(a.tools)
    import capture_kernel as ck

    rec = ck.load_recorder(a.tools)
    from tensorfold.cuda import experts as grouped
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda import exl3_mm, glue, latent, qmm, sparse
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda import mtp as M

    pair = ref.Pair(rec)
    ws = [ref.load_rank(a.model, r, [LAYER], mtp=True, long_context=True) for r in (0, 1)]
    if ws[0].layers[0].kind != "dsa" or ws[0].mtp is None:
        raise SystemExit(f"layer {LAYER} must be a DSA layer and the pack must hold the MTP layer")
    for r, w in enumerate(ws):
        w.comm = pair.comm(r)
    (out / "weights.json").write_text(json.dumps({f"rank{r}": ref.weight_shas(w) for r, w in enumerate(ws)},
                                                 indent=1, sort_keys=True) + "\n")
    tr = ref.Tracer()
    tr.wrap_all(((glue, ("embed", "hc_pre", "hc_post", "rmsnorm", "router", "select", "swiglu", "combine",
                         "combine_shared", "residual_add", "stream_mean")),
                 (F, ("mm", "gather", "dsa_block", "moe_block", "dense_attention")),
                 (M, ("mm", "dsa_block", "moe_block")), (qmm, ("group_sums",)),
                 (latent, ("latent_write", "absorb_q", "attention", "sparse_attention", "sparse_onepass", "expand_v")),
                 (sparse, ("index_update", "select_tokens", "top_pools", "select_split")),
                 (exl3_mm, ("routed",)), (grouped, ("route",))))
    need = sum(k for _, _, k in RUN_A[:-1]) + RUN_A[-1][1] + 1
    ids = ref.ids_for(a.model, need, 20261011)
    arrays = tr.arrays
    steps: dict[str, list] = {"a": [], "m": [], "l": []}

    def engine_state(w, capacity: int):
        """decode.Engine's caches, slot and state for one stream (the shared pool's first ``capacity`` tokens)."""
        caches = F.Caches(w, capacity, streams=1, ring=F.index_ring(capacity, max(MAX_ROWS, PREFILL_ROWS)),
                          kv="bf16")
        slots = F.Slots(w, 1, MAX_ROWS)
        return F.State(w, capacity, MAX_ROWS, caches=caches, slots=slots)

    def main_step(r, w, st, buf, name, toks, R, keep):
        arrays[f"r{r}_{name}_ids"] = torch.tensor(toks, dtype=torch.int32)
        ref.Tracer.begin(r, name)
        with rec.scope(f"r{r}_{name}", detail=True):
            F.stage(w, st, buf, toks)
            F.compute(w, st, buf, R, logits=True, head=False, nch=F.chunks_for(st, R), host_pos=st.pos)
            F.commit(w, st, buf, R, keep)
        torch.cuda.synchronize()
        ref.Tracer.begin(r, None)
        arrays[f"r{r}_{name}_x_out"] = buf.x[:R].clone()
        arrays[f"r{r}_{name}_fnormed"] = buf.fnormed[:R].clone()

    def rank_run(r: int) -> None:
        w = ws[r]
        b = F.Buffers(w, MAX_ROWS, CAPACITY)
        pb = F.Buffers(w, PREFILL_ROWS, CAPACITY, prefill=True)
        mb = F.Buffers(w, MAX_ROWS, CAPACITY)
        st = engine_state(w, CAPACITY)
        hidden: list[torch.Tensor] = []
        pos = 0
        for i, (kind, R, keep) in enumerate(RUN_A):
            main_step(r, w, st, pb if kind == "p" else b, f"a{i}", ids[pos:pos + R], R, keep)
            hidden.append(arrays[f"r{r}_a{i}_fnormed"][:keep])
            if r == 0:
                steps["a"].append({"kind": kind, "rows": R, "keep": keep, "pos": pos})
            pos += keep
        hid = torch.cat(hidden)                          # the committed rows' final-normed rows, by position

        # the MTP head on those rows (decode.prefill_chunk's absorb, then decode.absorb / draft)
        j = 0

        def mtp_step(buf, first: int, n: int, kind: str, chained: torch.Tensor | None = None,
                     draft_token: int | None = None) -> torch.Tensor:
            nonlocal j
            name = f"m{j}"
            j += 1
            nxt = [draft_token] if draft_token is not None else ids[first + 1:first + 1 + n]
            h = chained if chained is not None else hid[first:first + n]
            arrays[f"r{r}_{name}_next"] = torch.tensor(nxt, dtype=torch.int32)
            arrays[f"r{r}_{name}_hidden_in"] = h.clone()
            if r == 0:
                steps["m"].append({"kind": kind, "rows": len(nxt), "mtp_len": st.mtp_len})
            ref.Tracer.begin(r, name)
            with rec.scope(f"r{r}_{name}", detail=True):
                logits = M.mtp_forward(w, st, buf, nxt, h)
            torch.cuda.synchronize()
            draft = D.sample_rows(w, logits[:1], [st.mtp_len + len(nxt)], None)[0]
            ref.Tracer.begin(r, None)
            arrays[f"r{r}_{name}_logits"] = logits.clone()
            arrays[f"r{r}_{name}_fnormed"] = buf.fnormed[:1].clone()
            arrays[f"r{r}_{name}_draft"] = torch.tensor([draft], dtype=torch.int32)
            st.set_mtp_len(st.mtp_len + len(nxt))
            return draft

        mtp_step(pb, 0, 2047, "absorb")                                   # the first chunk (its last row waits)
        d = mtp_step(mb, 2047, 2, "absorb")
        for _ in range(2):                                                # a chain of two drafts
            d = mtp_step(mb, 0, 1, "draft", chained=mb.fnormed[0:1].clone(), draft_token=d)
            st.mtp_drafted += 1
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)                       # the round kept none of them
        st.mtp_drafted = 0
        mtp_step(mb, 2049, 3, "absorb")
        mtp_step(mb, 2052, 16, "absorb")
        mtp_step(pb, 2068, 2048, "absorb")                                # past the dense limit: sparse rows
        d = mtp_step(mb, 4116, 1, "absorb")
        mtp_step(mb, 0, 1, "draft", chained=mb.fnormed[0:1].clone(), draft_token=d)

        if a.skip_long:
            return
        # synthetic long contexts: every DSA cache of positions before P drawn, then decode windows at P
        del st
        torch.cuda.empty_cache()
        st = engine_state(w, LONG_CAPACITY)
        g = ref.seeded(20261012)
        for di in range(len(st.kc)):
            kc = st.kc[di]
            kc.copy_(torch.randn(kc.shape, generator=g, device=ref.DEV).to(kc.dtype))
            pk = st.index[di][2]
            pk.copy_(torch.randn(pk.shape, generator=g, device=ref.DEV).to(pk.dtype))
        torch.cuda.synchronize()
        if r == 0:                       # the drawn caches (the same on both ranks), for the Zig test to start from
            arrays["l_latent0"] = st.kc[0].clone()
            arrays["l_pk0"] = st.index[0][2].clone()
        k = 0
        for P, run in LONG:
            st.set_pos(P)
            for kind, R, keep in run:
                toks = ids[k % 1000:k % 1000 + R]
                main_step(r, w, st, b, f"l{k}", toks, R, keep)
                if r == 0:
                    steps["l"].append({"kind": kind, "rows": R, "keep": keep, "pos": st.pos - keep})
                k += 1

    pair.run(rank_run)
    for name in [n for n in arrays if n.startswith("r0_") and n.endswith(("_x_out", "_fnormed", "_draft"))]:
        if not ref.same(arrays[name], arrays["r1_" + name[3:]]):
            raise SystemExit(f"{name[3:]}: the ranks differ (their gathers summed alike; the logits are each rank's "
                             f"vocabulary half)")
    params = {"layer": LAYER, "mtp_layer": ws[0].cfg.layers, "capacity": CAPACITY, "long_capacity": LONG_CAPACITY,
              "rows": MAX_ROWS, "prompt_rows": PREFILL_ROWS, "world": 2, "steps": steps,
              "dense_limit": ws[0].cfg.dense_limit, "ring": F.index_ring(CAPACITY, max(MAX_ROWS, PREFILL_ROWS))}
    ck.save(out / "fixture", params, arrays, extra={"trace": tr.calls})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    ref.provenance(out, {"dump": "dsa", "run_a": RUN_A, "long": LONG, "calls": len(tr.calls)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
