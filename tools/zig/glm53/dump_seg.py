#!/usr/bin/env python3
"""GLM-5.3-Flash's multi-stream verify windows (verify.BatchedVerify: several streams' rows back to back, KDA layer 0
through kda_segments, DSA layer 3 through dsa_segments) from the Mia fork in its strict setting (TF_GLM53_REF_KV:
bf16 or fp8 DSA caches), on both ranks of a two-rank split run on one Spark (ref.Pair), over a model cut down to
layers 0 and 3, the head, the embedding and the final norm (ref.load_rank, no MTP layer: the fork drafts concurrent
streams with DFlash2 only), as a fixture for the Zig engine's segmented kernels.

The pool is the served one (forward.Caches over POOL_ROWS tokens, four stream slots, index rings of RING rows a
slot): stream s in slot s at base BASES[s] (multiples of 2,048), its extent EXTENTS[s] tokens. Histories, each run
alone through the single-stream prompt path: s0 HISTORY[0] tokens (dense rows), s1 HISTORY[1] (its second
window's 3 rows at 2,050 cross the dense limit: one dense, two sparse), s2 HISTORY[2] (sparse rows); s3's layer-3 latents and pooled keys before LONG_POS are
drawn (dump_dsa.py run l's way, quantized for FP8 caches) and its KDA state is zero. Then the windows WINDOWS, each a
list of (stream, rows, keep): segments of 1, 3 and 16 rows mixed, rewinds (keep < rows: the segmented KDA replay),
windows of one to four segments. Every window runs twice from the same state: each segment alone (forward.forward
and commit on its stream's state, the solo kernels) and all of them as one BatchedVerify window (forward and
commit); the script stops unless both give the same bytes (each segment's logits, latent rows, pooled keys, index
ring, KDA state and conv window): the fork's claim that a segment's rows get the bits of its stream's window alone.

Arrays, per rank r:
  r{r}_init_<name>        the state before the first window: lat (layer 3's latent plane), pk (its pooled keys),
                          ik / ig (its index key and gate rings, every slot), rec / conv (layer 0's KDA states and conv
                          windows, every slot), and the histories' and windows' token ids (r{r}_w{k}_ids)
  r{r}_w{k}_logits        window k's logits (this rank's vocabulary half, its rows in window order)
The manifest's "trace" lists the traced calls of each batched window (digests of what they changed, as
dump_prompt.py's), params "digests" the per-window states after the commit, params "windows" the segments with their
streams' bases, positions, rows and keeps.

  python3 -B /w/tools/zig/glm53/dump_seg.py --tools /w/tools/zig --out /aot/out --model /model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref  # noqa: E402  (sets the strict environment before TensorFold is imported)

import torch  # noqa: E402

from dump_prompt import DigestTracer, sha  # noqa: E402

LAYERS = [0, 3]
SLOT_ROWS, PREFILL_ROWS, WINDOW_ROWS = 16, 2048, 32
HISTORY = (100, 2_049, 4_100)
LONG_POS = 40_000
EXTENTS = (8_192, 8_192, 8_192, 43_008)
BASES = (0, 8_192, 16_384, 24_576)
POOL_ROWS = 67_584
RING = 2_112
# (stream, rows, keep) a segment, in window order
WINDOWS = (
    ((0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)),
    ((0, 3, 2), (1, 3, 3), (2, 16, 16), (3, 1, 1)),
    ((2, 16, 5), (0, 3, 3), (1, 1, 1), (3, 3, 1)),
    ((3, 16, 16),),
    ((1, 3, 3), (3, 3, 2)),
    ((0, 1, 1), (1, 16, 9), (3, 3, 3), (2, 12, 12)),
)


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
    from tensorfold.cuda import experts as grouped
    from tensorfold.families.glm5_next.cuda import exl3_mm, glue, kda as kda_mod, kv8, latent, sparse
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.segments import EXTENT
    from tensorfold.families.glm5_next.cuda.verify import BatchedVerify, Segment

    if any(b % EXTENT for b in BASES) or BASES[-1] + EXTENTS[-1] != POOL_ROWS:
        raise SystemExit("the streams' bases must be multiples of the fork's EXTENT and fill the pool")
    pair = ref.Pair(rec)
    ws = [ref.load_rank(a.model, r, LAYERS, mtp=False, long_context=True) for r in (0, 1)]
    if [l.kind for l in ws[0].layers] != ["kda", "dsa"]:
        raise SystemExit(f"layers {LAYERS} must be a KDA and a DSA layer")
    for r, w in enumerate(ws):
        w.comm = pair.comm(r)
    tr = DigestTracer()
    tr.wrap_all(((glue, ("embed", "hc_pre", "hc_post", "rmsnorm", "router", "select", "swiglu", "combine_shared",
                         "residual_add", "stream_mean")),
                 (F, ("mm", "gather", "conv_shift_segments")),
                 (kda_mod, ("chain_segments", "replay_layers_segments")),
                 (latent, ("seg_latent_write", "absorb_q", "seg_attention", "expand_v")),
                 (sparse, ("seg_index_update", "seg_select_tokens")),
                 (exl3_mm, ("routed",)), (grouped, ("route",))))
    need = sum(HISTORY) + sum(n for win in WINDOWS for _, n, _ in win) + 64
    ids = ref.ids_for(a.model, need, 20261016)
    arrays: dict[str, torch.Tensor] = {}
    digests: dict[str, list] = {}
    windows: list[dict] = []

    def draw(cache: torch.Tensor, g) -> None:
        """dump_dsa.py run l's draw: seeded normal rows, an FP8 cache's quantized from the bf16 draw."""
        if not kv8.quantized(cache):
            cache.copy_(torch.randn(cache.shape, generator=g, device=ref.DEV).to(cache.dtype))
            return
        width = kv8.width(cache)
        for i in range(0, cache.shape[0], 65536):
            n = min(65536, cache.shape[0] - i)
            cache[i:i + n].copy_(kv8.quantize_rows(torch.randn((n, width), generator=g, device=ref.DEV)
                                                   .to(torch.bfloat16)))

    def rank_run(r: int) -> None:
        w = ws[r]
        caches = F.Caches(w, POOL_ROWS, streams=4, ring=RING, kv=ref.KV)
        if caches.ring != F.index_ring(POOL_ROWS, PREFILL_ROWS) or caches.rings is None:
            raise SystemExit("the pool's index rings must be the served ones")
        slots = F.Slots(w, 4, SLOT_ROWS)
        sts = [F.State(w, EXTENTS[s], SLOT_ROWS, caches=caches, base=BASES[s], slots=slots, slot=s) for s in range(4)]
        pb = F.Buffers(w, PREFILL_ROWS, POOL_ROWS, prefill=True)
        b = F.Buffers(w, SLOT_ROWS, POOL_ROWS)
        e = SimpleNamespace(w=w, caches=caches, slots=slots, home=sts[0])
        bv = BatchedVerify(e, rows=WINDOW_ROWS, segs=4)
        at = 0

        def take(n: int) -> list[int]:
            nonlocal at
            at += n
            return ids[at - n:at]

        # the histories, each stream alone through the prompt path
        for s, n in enumerate(HISTORY):
            toks = take(n)
            arrays[f"r{r}_h{s}_ids"] = torch.tensor(toks, dtype=torch.int32)
            for c0 in range(0, n, PREFILL_ROWS):
                R = min(PREFILL_ROWS, n - c0)
                F.stage(w, sts[s], pb, toks[c0:c0 + R])
                F.compute(w, sts[s], pb, R, logits=False, nch=F.chunks_for(sts[s], R), host_pos=sts[s].pos)
                F.commit(w, sts[s], pb, R, R)
        g = ref.seeded(20261017)
        st3 = sts[3]
        draw(st3.kc[0][:LONG_POS], g)
        draw(st3.index[0][2][:LONG_POS // 4], g)
        st3.set_pos(LONG_POS)
        torch.cuda.synchronize()
        planes = caches.arena.planes
        lat, pk = planes[caches.kc[0]].tensor, planes[caches.index[0][2]].tensor
        ik, ig = caches.rings[0]
        for name, t in (("lat", lat), ("pk", pk), ("ik", ik), ("ig", ig), ("rec", slots.rec), ("conv", slots.conv)):
            arrays[f"r{r}_init_{name}"] = t.clone()
        if r == 0:
            params0["positions"] = [st.pos for st in sts]

        def snapshot():
            return ([t.clone() for t in (lat, pk, ik, ig, slots.rec, slots.conv)],
                    [(st.pos, list(st.cur)) for st in sts])

        def restore(snap) -> None:
            for t, x in zip((lat, pk, ik, ig, slots.rec, slots.conv), snap[0]):
                t.copy_(x)
            for st, (pos, cur) in zip(sts, snap[1]):
                st.set_pos(pos)
                st.cur = list(cur)

        def state(win, logits: list[torch.Tensor]) -> dict:
            """What a window leaves: each segment's logits, latent rows and stream's KDA slot, the pooled keys and
            the index rings (every stream's)."""
            got = {"pk": sha(pk), "ik": sha(ik), "ig": sha(ig)}
            for j, (s, n, _) in enumerate(win):
                st = sts[s]
                got[f"s{j}_logits"] = sha(logits[j])
                got[f"s{j}_rec"] = sha(slots.rec[s, st.cur[0] if st.cur else 0])
                got[f"s{j}_conv"] = sha(slots.conv[s])
            return got

        for k, win in enumerate(WINDOWS):
            starts = [sts[s].pos for s, _, _ in win]
            toks = [take(n) for _, n, _ in win]
            arrays[f"r{r}_w{k}_ids"] = torch.tensor([t for tt in toks for t in tt], dtype=torch.int32)
            snap = snapshot()
            # each segment alone: its stream's window and commit (the solo kernels)
            solo_logits = []
            for (s, n, keep), tt in zip(win, toks):
                solo_logits.append(F.forward(w, sts[s], b, tt)[:n].clone())
                F.commit(w, sts[s], b, n, keep)
            torch.cuda.synchronize()
            solo = state(win, solo_logits)
            solo.update({f"s{j}_latent": sha(lat[BASES[s] + p0:BASES[s] + p0 + n])
                         for j, ((s, n, _), p0) in enumerate(zip(win, starts))})
            restore(snap)
            # the same rows as one batched window
            segs = [Segment(sts[s], tt) for (s, _, _), tt in zip(win, toks)]
            ref.Tracer.begin(r, f"w{k}")
            with rec.scope(f"r{r}_w{k}", detail=True):
                v = bv.forward(segs)
                seg_logits = [x.clone() for x in v.logits]
                bv.commit(segs, [keep for _, _, keep in win])
            torch.cuda.synchronize()
            ref.Tracer.begin(r, None)
            got = state(win, seg_logits)
            got.update({f"s{j}_latent": sha(lat[BASES[s] + p0:BASES[s] + p0 + n])
                        for j, ((s, n, _), p0) in enumerate(zip(win, starts))})
            bad = [name for name in got if got[name] != solo[name]]
            if bad:
                raise SystemExit(f"rank {r} window {k}: the batched window differs from its segments alone in {bad}")
            arrays[f"r{r}_w{k}_logits"] = torch.cat(seg_logits)
            for name, d in got.items():
                digests[f"r{r}_w{k}_{name}"] = d
            if r == 0:
                windows.append({"segments": [{"stream": s, "base": BASES[s], "pos": p0, "rows": n, "keep": keep,
                                              "ring_base": caches.ring_base(s)}
                                             for (s, n, keep), p0 in zip(win, starts)]})
            print(f"rank {r} window {k}: {[(s, p0, n, keep) for (s, n, keep), p0 in zip(win, starts)]}: "
                  f"batched == alone", flush=True)

    params0: dict = {}
    pair.run(rank_run)
    for name in [n for n in digests if n.startswith("r0_")]:
        # the latent rows, pooled keys and rings both ranks hold alike; the KDA state is each rank's heads
        if name.endswith(("_latent", "_pk", "_ik", "_ig")) and digests[name] != digests["r1_" + name[3:]]:
            raise SystemExit(f"{name[3:]}: the ranks differ")
    params = {"layers": LAYERS, "world": 2, "kv": ref.KV, "pool_rows": POOL_ROWS, "ring": RING, "bases": BASES,
              "extents": EXTENTS, "history": HISTORY, "long_pos": LONG_POS, "slot_rows": SLOT_ROWS,
              "window_rows": WINDOW_ROWS, "prompt_rows": PREFILL_ROWS, "dense_limit": ws[0].cfg.dense_limit,
              "windows": windows, "digests": digests, **params0}
    ck.save(out / "fixture", params, arrays, extra={"trace": tr.calls})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    ref.provenance(out, {"dump": "seg", "windows": WINDOWS, "calls": len(tr.calls)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
