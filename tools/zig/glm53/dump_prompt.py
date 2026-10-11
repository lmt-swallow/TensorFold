#!/usr/bin/env python3
"""GLM-5.3-Flash's prompt path (decode.prefill_chunk: the chunk's forward, the head on its last row, the MTP head's
absorb of its rows, the commit) from the Mia fork in its strict setting, on both ranks of a two-rank split run on
one Spark (ref.Pair), over a model cut down to layers 0 (KDA, dense MLP) and 3 (DSA, MoE) plus the MTP layer, the
head, the embedding and the final norm (ref.load_rank): a call-by-call fixture for tf-cuda-test glm53-prompt.

Each scenario is one stream from a fresh state, its prompt committed in the chunks listed (start, rows), the way
decode.prefill runs a prompt (chunks of 2,048 rows) or a resumed one (a chunk starting anywhere). The scenarios
are chosen so that every prompt-form kernel variant the served engine can reach is launched at least once:

  s0  35,365 tokens in chunks of 2,048: the chunk at 2,048 crosses the dense limit (3 dense rows, 8 latent chunks),
      chunks past 32,768 score 8,192 pools or more, and the last chunk's 549 rows end in a 37-row selection block
      (the split selection's pool form, select_split without positions)
  s1  4,645 tokens: the last chunk at 4,096 ends in a 37-row block over 2,048 pools (_select_rows, 4,096-wide)
  s2  200 then 300 rows (the 129..255-row and the unaligned SEQ matmuls; one latent chunk)
  s3  600, 500, 1,048 (crossing the limit at 1,100: 951 dense rows in two blocks, 5 latent chunks), then 37 rows
      past it (a prompt tail selecting like decode rows)
  s4  2,048, 1,000 (6 latent chunks), 1,500 past the limit
  s5  2,048, 1,500 (7 latent chunks)
  s6  2, 2,048, 2,048 (the third chunk at 2,050: one dense row, 9 latent chunks)
  s7  4 x 2,048, then 37 rows at 8,192 (decode-form scores over 4,096 pools: 64 programs a row)
  l   drawn caches (layer 3's and the MTP layer's latents and pooled keys from `draw`, KDA state zero), then chunks at 40,000
      (2,048, 549, 37) and 600,000 (549, 37): the selection over 10k-150k pools, positions past 524,288

Per rank r and chunk step <s>_<i> (s0_3, l_1, ...):
  r{r}_<step>_ids        the chunk's token ids (int32)
  r{r}_<step>_head       the head's logits of the chunk's last row (this rank's vocabulary half, bf16)
The manifest's "trace" lists every traced call (ref.Tracer's names) with "digests": the sha256 and byte length of
each tensor argument the call changed (and of its return values), so the fixture holds no intermediate arrays; and
params "digests" the per-step states: x_out (the streams after the last layer), rec / conv (layer 0's
KDA state and conv window after the commit), latent / pk (layer 3's latent rows of the chunk and its pooled keys up
to the chunk's end), mtp_latent (the MTP layer's latent rows the absorb wrote).

  python3 -B /w/tools/zig/glm53/dump_prompt.py --tools /w/tools/zig --out /aot/out --model /model
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref  # noqa: E402  (sets the strict environment before TensorFold is imported)

import torch  # noqa: E402

LAYERS = [0, 3]
MAX_ROWS, PREFILL_ROWS = 16, 2048


def _chunks(total: int, step: int = PREFILL_ROWS, start: int = 0) -> list[tuple[int, int]]:
    return [(a, min(step, start + total - a)) for a in range(start, start + total, step)]


SCENARIOS = {
    "s0": _chunks(35_365),
    "s1": _chunks(4_645),
    "s2": [(0, 200), (200, 300)],
    "s3": [(0, 600), (600, 500), (1_100, 1_048), (2_148, 37)],
    "s4": [(0, 2_048), (2_048, 1_000), (3_048, 1_500)],
    "s5": [(0, 2_048), (2_048, 1_500)],
    "s6": [(0, 2), (2, 2_048), (2_050, 2_048)],
    "s7": _chunks(8_192) + [(8_192, 37)],
}
CAPACITY = 36_864
LONG = [(40_000, (2_048, 549, 37)), (600_000, (549, 37))]
LONG_CAPACITY = 602_112


def draw(t: torch.Tensor, seed: int) -> None:
    """t (bf16) <- a fixed integer formula of each element's flat index i and ``seed``, exact in bf16 (multiples of
    1/64 in [-2, 2)), so the Zig test draws the same bytes on the host (glm53_prompt_tests.zig `draw`):
    h = (i * 0x9E3779B1 + seed * 0x85EBCA6B) mod 2^32, h ^= h >> 15, value = (((h >> 7) & 255) - 128) / 64."""
    flat = t.view(-1)
    n = flat.numel()
    step = 1 << 24
    for a in range(0, n, step):
        i = torch.arange(a, min(n, a + step), dtype=torch.int64, device=t.device)
        h = (i * 0x9E3779B1 + seed * 0x85EBCA6B) & 0xFFFFFFFF
        h = h ^ (h >> 15)
        flat[a:a + i.numel()] = (((h >> 7) & 255) - 128).to(torch.float32).div_(64.0).to(t.dtype)


def sha(t: torch.Tensor) -> list:
    raw = ref.bits(t).cpu().numpy().tobytes()
    return [hashlib.sha256(raw).hexdigest(), len(raw)]


class DigestTracer(ref.Tracer):
    """ref.Tracer's calls and names, keeping the sha256 and byte length of each tensor argument the call changed
    (of 64 MiB or less) and of its return values in the call's "digests" instead of the arrays."""

    def wrap(self, module, name: str) -> None:
        fn = getattr(module, name)
        if getattr(fn, "_glm53_traced", False):
            return
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        tracer = self

        @functools.wraps(fn)
        def traced(*args, **kwargs):
            step = getattr(ref._local, "step", None)
            if step is None:
                return fn(*args, **kwargs)
            bound = None
            if sig is not None:
                try:
                    bound = sig.bind(*args, **kwargs)
                except TypeError:
                    bound = None
            items = (list(bound.arguments.items()) if bound else
                     [(f"a{i}", a) for i, a in enumerate(args)] + list(kwargs.items()))
            seq = ref._local.seq
            ref._local.seq += 1
            meta = {"rank": ref._local.rank, "step": step, "seq": seq, "fn": f"{module.__name__}.{name}",
                    "digests": {}}
            torch.cuda.synchronize()
            before = {k: v.detach().clone() for k, v in items
                      if isinstance(v, torch.Tensor) and v.numel() * v.element_size() <= ref.KEEP_IN_BYTES}
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            for k, v in items:
                if k in before and not ref.same(v, before[k]):
                    meta["digests"][k] = sha(v)
            rets = [out] if isinstance(out, torch.Tensor) else list(out) if isinstance(out, tuple) else []
            for j, x in enumerate(rets):
                if isinstance(x, torch.Tensor):
                    meta["digests"]["ret" if isinstance(out, torch.Tensor) else f"ret{j}"] = sha(x)
            tracer.calls.append(meta)
            return out

        traced._glm53_traced = True
        setattr(module, name, traced)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", default="", help="comma-separated scenarios to run (default: all, and l)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref.tools_on_path(a.tools)
    import capture_kernel as ck

    rec = ck.load_recorder(a.tools)
    from tensorfold.cuda import experts as grouped
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda import exl3_mm, glue, kda as kda_mod, latent, sparse
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda import mtp as M

    only = [s for s in a.only.split(",") if s] or list(SCENARIOS) + ["l"]
    pair = ref.Pair(rec)
    ws = [ref.load_rank(a.model, r, LAYERS, mtp=True, long_context=True) for r in (0, 1)]
    if [l.kind for l in ws[0].layers] != ["kda", "dsa"] or ws[0].mtp is None:
        raise SystemExit(f"layers {LAYERS} must be a KDA and a DSA layer and the pack must hold the MTP layer")
    for r, w in enumerate(ws):
        w.comm = pair.comm(r)
    (out / "weights.json").write_text(json.dumps({f"rank{r}": ref.weight_shas(w) for r, w in enumerate(ws)},
                                                 indent=1, sort_keys=True) + "\n")
    tr = DigestTracer()
    tr.wrap_all(((glue, ("embed", "hc_pre", "hc_post", "rmsnorm", "router", "select", "swiglu", "combine_shared",
                         "residual_add", "stream_mean")),
                 (F, ("mm", "gather", "_shift_conv", "dense_attention")), (M, ("mm",)),
                 (kda_mod, ("chain",)),
                 (latent, ("latent_write", "absorb_q", "sparse_onepass", "expand_v")),
                 (sparse, ("index_update", "select_tokens")),
                 (exl3_mm, ("routed",)), (grouped, ("route",))))
    ids = ref.ids_for(a.model, max(s + r for chunks in SCENARIOS.values() for s, r in chunks), 20261012)
    long_ids = ref.ids_for(a.model, 3_000, 20261013)
    arrays: dict[str, torch.Tensor] = {}
    digests: dict[str, list] = {}
    steps: dict[str, list] = {}

    def engine(w, capacity: int):
        """decode.Engine's prompt side for one stream (the shared pool's first ``capacity`` tokens)."""
        caches = F.Caches(w, capacity, streams=1, ring=F.index_ring(capacity, max(MAX_ROWS, PREFILL_ROWS)),
                          kv=ref.KV)
        slots = F.Slots(w, 1, MAX_ROWS)
        st = F.State(w, capacity, MAX_ROWS, caches=caches, slots=slots)
        pb = F.Buffers(w, PREFILL_ROWS, capacity, prefill=True)
        return SimpleNamespace(w=w, st=st, pbuf=pb, last_hidden=None)

    def chunk(r: int, e, prompt: list[int], start: int, R: int, step: str) -> None:
        st, b = e.st, e.pbuf
        mtp0 = st.mtp_len
        arrays[f"r{r}_{step}_ids"] = torch.tensor(prompt[start:start + R], dtype=torch.int32)
        ref.Tracer.begin(r, step)
        with rec.scope(f"r{r}_{step}", detail=True):
            head = D.prefill_chunk(e, prompt, start, R, mtp=True, head=True)
        torch.cuda.synchronize()
        ref.Tracer.begin(r, None)
        arrays[f"r{r}_{step}_head"] = head
        p = f"r{r}_{step}_"
        digests[p + "x_out"] = sha(b.x[:R])
        digests[p + "rec"] = sha(st.rec[st.cur[0], 0])
        digests[p + "conv"] = sha(st.conv[0])
        digests[p + "latent"] = sha(st.kc[0][start:start + R])
        digests[p + "pk"] = sha(st.index[0][2][:(start + R) // 4 + 2])
        n = st.mtp_len - mtp0
        if n:
            digests[p + "mtp_latent"] = sha(st.mtp_kc[mtp0:mtp0 + n])
        print(f"rank {r} {step}: {R} rows at {start}, MTP {st.mtp_len - mtp0} rows at {mtp0}", flush=True)
        if r == 0:
            steps.setdefault(step.split("_")[0], []).append(
                {"step": step, "pos": start, "rows": R, "mtp_len": mtp0, "absorbed": n, "end": start + R == len(prompt)})

    def rank_run(r: int) -> None:
        w = ws[r]
        for name, chunks in SCENARIOS.items():
            if name not in only:
                continue
            e = engine(w, CAPACITY)
            end = chunks[-1][0] + chunks[-1][1]
            prompt = ids[:end]
            for i, (start, R) in enumerate(chunks):
                chunk(r, e, prompt, start, R, f"{name}_{i}")
            del e
            torch.cuda.empty_cache()
        if "l" not in only:
            return
        e = engine(w, LONG_CAPACITY)
        st = e.st
        if len(st.kc) != 1 or len(st.index) != 2:
            raise SystemExit("the long run draws layer 3's caches and the MTP layer's")
        for seed, t in enumerate((st.kc[0], st.index[0][2], st.mtp_kc, st.index[1][2])):
            draw(t, seed)
        torch.cuda.synchronize()
        i = 0
        for base, rows in LONG:
            # one prompt continuing a cached context of `base` tokens (its ids before base never read)
            st.set_pos(base)
            st.set_mtp_len(base)
            prompt = [0] * base + long_ids[:sum(rows)]
            start = base
            for R in rows:
                chunk(r, e, prompt, start, R, f"l_{i}")
                start += R
                i += 1

    pair.run(rank_run)
    for name in [n for n in digests if n.startswith("r0_")]:
        # the KDA state and conv window are each rank's heads; the streams and the latent caches both ranks hold alike
        if name.endswith(("_x_out", "_latent", "_pk")) and \
                digests[name] != digests["r1_" + name[3:]]:
            raise SystemExit(f"{name[3:]}: the ranks differ")
    params = {"layers": LAYERS, "mtp_layer": ws[0].cfg.layers, "capacity": CAPACITY, "long_capacity": LONG_CAPACITY,
              "rows": MAX_ROWS, "prompt_rows": PREFILL_ROWS, "world": 2, "steps": steps, "digests": digests,
              "dense_limit": ws[0].cfg.dense_limit, "ring": F.index_ring(CAPACITY, max(MAX_ROWS, PREFILL_ROWS)),
              "long_ring": F.index_ring(LONG_CAPACITY, max(MAX_ROWS, PREFILL_ROWS))}
    ck.save(out / "fixture", params, arrays, extra={"trace": tr.calls})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    ref.provenance(out, {"dump": "prompt", "scenarios": SCENARIOS, "long": LONG, "calls": len(tr.calls)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
