#!/usr/bin/env python3
"""A concurrent round from the Python fork (rounds.py RoundDecoder over model.py's PoolCache) as a fixture for the Zig
engine: two streams in a 4-slot pool (slot 0 at extent [0, 2048), slot 2 at [2048, 4096)), each primed with its own
prompt through the single-stream prompt path on its pool view, then one round whose windows are a pending token
and five drafts a stream (12 rows). Both ranks of a TP2 split run in this process, one rank's block resident at a
time (dump_forward.py's RankBlocks); the gathers stack both ranks' partials, so every intermediate is the served
one. The fork's switches as the environment sets them (capture_rounds.sh: its defaults, the served decode path:
rounds.py `_layers_fused` with `_attention2`; UNFUSED=1 there: TF_DS_HC_FUSED=0, TF_DS_ROT_Q=0, TF_DS_ROT_WOB=0,
TF_DS_ROT_ATTN=0, the plain loop); L2 prefetch off (it moves no bit). With the mHC fused, a sublayer's streams are
recorded where the next sublayer's posted mixes (or the stretch's last post) wrote them.
`--engram <dir>` runs Engram on its layers (dump_forward.py's EngramRef): the primes on each stream's prompt, the
round on every stream's rows hashed from that stream's own ids (rounds.py `forward`), one projection, sum and gate
over the round's rows; each rank's partial `L<i>.p_engram.r<r>` and the gated streams `L<i>.h_engram` recorded.

  python3 -B /w/tools/zig/dsv41/dump_rounds.py --tools /w/tools/zig --model /model --out /out [--layers 4] \
      [--engram /models/DeepSeek-V4.1-Flash-Engram --token-map /aot/token_map.json]

`--streams slot:base:size:prompt,...` places the streams elsewhere (a long stream: a scored indexer, and past the
candidate pool's width its mask), `--cap` sizes the pool to hold them; the round's bucket is bucket_for(deepest row).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TF_DS_L2_PREFETCH", "0")
HERE = Path(__file__).resolve().parent
BF16, F32 = torch.bfloat16, torch.float32


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--engram", default=None, help="the Engram shards' dir: run Engram on its layers")
    ap.add_argument("--token-map", default=None, help="the compressed token map JSON (default: built from tokenizer.json)")
    ap.add_argument("--streams", default="0:0:2048:40,2:2048:2048:200",
                    help="the streams as slot:base:size:prompt-length, comma separated")
    a = ap.parse_args()
    ck = load_module("capture_kernel", HERE / "capture_kernel.py")
    fwd = load_module("dump_forward", HERE / "dump_forward.py")
    rec = ck.load_recorder(a.tools)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda.model import KV_QUANT, RAW, Comm, Model, mm
    from tensorfold.families.deepseek_v41.cuda.rounds import RoundDecoder
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, Weights, linear, load_block

    assert KV_QUANT
    fused = K.on("hc")
    torch.cuda.set_device(0)
    world = 2
    L = a.layers
    c = Cfg.read(a.model)
    sh = Shards(a.model)
    vocab_l = c.vocab // world
    heads = [linear(sh, "head", cols=(r * vocab_l, (r + 1) * vocab_l)) for r in range(world)]
    w = Weights(c, 0, world, embed=sh.get("embed.weight").contiguous().cuda(), norm=sh.get("norm.weight").contiguous().cuda(),
                head=heads[0], vocab_lo=0, vocab_hi=vocab_l, model_dir=Path(a.model))
    m = Model(w, Comm(None, 1))
    w.layers = [None] * L                             # new_pool / RoundDecoder count the layers
    slots, cap = 4, a.cap
    pool = m.new_pool(slots, cap)
    blocks = fwd.RankBlocks(sh, c, world, load_block)
    eng = fwd.engram_ref(a, c, world)

    def gather(parts: list) -> torch.Tensor:
        torch.cuda.synchronize()
        return torch.stack([p.contiguous() for p in parts]).clone()

    def mixes(n: int) -> tuple:
        dev = "cuda"
        return (torch.empty((n, c.dim), dtype=BF16, device=dev), torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev),
                torch.empty((n, c.hc), dtype=F32, device=dev), torch.empty((n, c.hc), dtype=F32, device=dev),
                torch.empty((n, c.hc), dtype=F32, device=dev), torch.empty((n, c.hc, c.hc), dtype=F32, device=dev))

    def prime(sc, ids: torch.Tensor) -> None:
        """model.py _forward_k's sublayers over a prompt on a pool view (the prompt path), both ranks a sublayer."""
        n = ids.shape[0]
        start = sc.length
        assert start == 0
        sc.tokens[start:start + n] = ids
        pos = torch.arange(start, start + n, device="cuda")
        h = w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(n)
        shared = [{} for _ in range(world)]
        for li in range(L):
            if eng is not None and blocks.get(0, li).engram_wkv is not None:
                h = eng.apply(blocks, li, h, ids.tolist(), start, n, gather, lambda name, t: None)
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            parts = [m.attention_k(blocks.get(r, li), x, sc, start, shared[r], pos) for r in range(world)]
            K.hc_post(gather(parts), h, post, comb, h)
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            parts = [m.moe(blocks.get(r, li), x) for r in range(world)]
            K.hc_post(gather(parts), h, post, comb, h)
            pre, pre_f = pre_f, pre
        sc.length = start + n

    def pool_planes() -> dict:
        d = {}
        for li in range(L):
            d[f"L{li}.ring"] = pool.ring[li]
            if li in pool.comp:
                d[f"L{li}.comp_codes"], d[f"L{li}.comp_scales"] = pool.comp[li]
            if li in pool.index_k:
                d[f"L{li}.idxk_codes"], d[f"L{li}.idxk_scales"] = pool.index_k[li]
            if li in pool.comp_raw:
                d[f"L{li}.raw_kv"], d[f"L{li}.raw_score"] = pool.comp_raw[li]
        return d

    g = torch.Generator(device="cpu")
    g.manual_seed(a.seed)
    streams = [tuple(int(v) for v in st.split(":")) for st in a.streams.split(",")]   # (slot, base, size, prompt length)
    assert all(b + sz <= cap and sl < slots for sl, b, sz, _ in streams), "a stream outside the pool"
    views, prompts = [], []
    for slot, base, size, n_prompt in streams:
        sc = m.pool_view(pool, slot, base, size)
        ids = torch.randint(0, c.vocab, (n_prompt,), generator=g).to(torch.int64).cuda()
        with rec.scope(f"prime_slot{slot}", detail=False):
            prime(sc, ids)
        views.append(sc)
        prompts.append(ids)
    torch.cuda.synchronize()
    before = {k: v.clone() for k, v in pool_planes().items()}
    # the round: a pending token and five drafts a stream
    k = 5
    windows = []
    ids_all, pos_all, slot_all, base_all, end_all = [], [], [], [], []
    hashes = []                                       # rounds.py forward: each stream's rows from its own host ids
    for (slot, base, size, n_prompt), sc, prompt in zip(streams, views, prompts):
        toks = torch.randint(0, c.vocab, (k + 1,), generator=g).tolist()
        windows.append((slot, base, size, sc.length, toks))
        if eng is not None:
            hashes.append(eng.hashes(prompt.tolist() + toks, sc.length, len(toks)))
        ids_all += toks
        pos_all += list(range(sc.length, sc.length + k + 1))
        slot_all += [slot] * (k + 1)
        base_all += [base] * (k + 1)
        end_all += [base + size] * (k + 1)
    R = len(ids_all)
    deepest = max(p + 1 for p in pos_all)
    from tensorfold.families.deepseek_v41.cuda.graph import bucket_for
    bucket = bucket_for(deepest, cap)
    dec = RoundDecoder(m, pool, R, bucket, taps=False, out=None)
    dec.set(ids_all, pos_all, slot_all, base_all, end_all)
    torch.cuda.synchronize()
    table = dec.table
    arrays = {"ids": torch.tensor(ids_all, dtype=torch.int64), "pos": torch.tensor(pos_all, dtype=torch.int64),
              "slots": torch.tensor(slot_all, dtype=torch.int64), "base": torch.tensor(base_all, dtype=torch.int64),
              "end": torch.tensor(end_all, dtype=torch.int64), "prompt0": prompts[0], "prompt1": prompts[1]}
    arrays.update({f"before_{k_}": v for k_, v in before.items()})
    cos_c, sin_c = m._cs(min(2, L - 1), table)
    cos_w, sin_w = m._cs(0, table)
    arrays.update({"cos_c": cos_c, "sin_c": sin_c, "cos_w": cos_w, "sin_w": sin_w})

    # the round's body (rounds.py _stretch, the unfused loop), each sublayer on both ranks' weights
    dec._idx = {}
    h, pre = K.embed_init(w.embed, dec.ids, c.hc)
    x, part, pre_a, pre_f, post, comb = mixes(R)
    shared = [{} for _ in range(world)]
    if eng is not None:
        import numpy as np

        hashes = np.concatenate(hashes, 0)

    def record(name: str, t: torch.Tensor) -> None:
        arrays[name] = t.clone()

    def fused_body(h, pre, x, part, pre_a, pre_f, post, comb):
        """rounds.py `_layers_fused` over one stretch of every layer, each sublayer on both ranks' weights: a
        sublayer's post fused into the next sublayer's mixes (kernels.hc_pre2, the Sinkhorn half on a side stream when
        hc_defer_on), an Engram layer's post alone before its gate, the last post alone at the end."""

        spare = torch.empty_like(h)
        pending = None
        sink = torch.cuda.Stream() if K.hc_defer_on(c.dim) else None

        def join():
            if sink is not None:
                torch.cuda.current_stream().wait_stream(sink)

        def pre_mix(h, params, pre_in, norm, pre_out):
            nonlocal spare, pending
            fn, scale, base = params
            if pending is None:
                K.hc_pre2(h, fn, scale, base, pre_in, norm, c.eps, c.hc_eps, c.hc_iters, x, pre_out, post, comb, part,
                          sink=sink)
                return h
            out = K.hc_pre2(h, fn, scale, base, pre_in, norm, c.eps, c.hc_eps, c.hc_iters, x, pre_out, post, comb,
                            part, gathered=pending, h_out=spare, sink=sink)
            pending = None
            spare = h
            return out

        for li in range(L):
            if eng is not None and blocks.get(0, li).engram_wkv is not None:
                if pending is not None:
                    with rec.scope(f"L{li}_hc_post_engram", detail=True):
                        K.hc_post(pending, h, post, comb, h)
                    pending = None
                    arrays[f"L{li - 1}.h_moe"] = h.clone()
                with rec.scope(f"L{li}_engram", detail=True):
                    h = eng.apply_hashes(blocks, li, h, hashes, gather, record)
            lay = blocks.get(0, li)
            cos, sin = m._cs(lay.idx, table)
            with rec.scope(f"L{li}_hc_pre_attn", detail=True):
                posted = pending is not None
                h = pre_mix(h, lay.hc_attn, pre, lay.attn_norm, pre_a)
            torch.cuda.synchronize()
            if posted:
                arrays[f"L{li - 1}.h_moe"] = h.clone()
            arrays[f"L{li}.x_attn"] = x.clone()
            parts = []
            for r in range(world):
                with rec.scope(f"L{li}_attention_r{r}", detail=True):
                    parts.append(dec._attention(blocks.get(r, li), x, shared[r], cos, sin))
                arrays[f"L{li}.p_attn.r{r}"] = parts[r].clone()
            join()
            pending = gather(parts)
            lay = blocks.get(0, li)
            with rec.scope(f"L{li}_hc_pre_ffn", detail=True):
                h = pre_mix(h, lay.hc_ffn, pre_a, lay.ffn_norm, pre_f)
            torch.cuda.synchronize()
            arrays[f"L{li}.h_attn"] = h.clone()
            arrays[f"L{li}.x_ffn"] = x.clone()
            parts = []
            for r in range(world):
                with rec.scope(f"L{li}_moe_r{r}", detail=True):
                    parts.append(m.moe(blocks.get(r, li), x, shared_side=False))
                arrays[f"L{li}.p_moe.r{r}"] = parts[r].clone()
            join()
            pending = gather(parts)
            pre, pre_f = pre_f, pre
        with rec.scope(f"L{L - 1}_hc_post_ffn", detail=True):
            K.hc_post(pending, h, post, comb, h)
        arrays[f"L{L - 1}.h_moe"] = h.clone()
        return h, pre

    if fused:
        h, pre = fused_body(h, pre, x, part, pre_a, pre_f, post, comb)
    else:
        for li in range(L):
            if eng is not None and blocks.get(0, li).engram_wkv is not None:
                with rec.scope(f"L{li}_engram", detail=True):
                    h = eng.apply_hashes(blocks, li, h, hashes, gather, record)
            lay = blocks.get(0, li)
            cos, sin = m._cs(lay.idx, table)
            fn, scale, base = lay.hc_attn
            with rec.scope(f"L{li}_hc_pre_attn", detail=True):
                K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            arrays[f"L{li}.x_attn"] = x.clone()
            parts = []
            for r in range(world):
                with rec.scope(f"L{li}_attention_r{r}", detail=True):
                    parts.append(dec._attention(blocks.get(r, li), x, shared[r], cos, sin))
                arrays[f"L{li}.p_attn.r{r}"] = parts[r].clone()
            with rec.scope(f"L{li}_hc_post_attn", detail=True):
                K.hc_post(gather(parts), h, post, comb, h)
            arrays[f"L{li}.h_attn"] = h.clone()
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_ffn
            with rec.scope(f"L{li}_hc_pre_ffn", detail=True):
                K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            arrays[f"L{li}.x_ffn"] = x.clone()
            parts = []
            for r in range(world):
                with rec.scope(f"L{li}_moe_r{r}", detail=True):
                    parts.append(m.moe(blocks.get(r, li), x, shared_side=False))
                arrays[f"L{li}.p_moe.r{r}"] = parts[r].clone()
            with rec.scope(f"L{li}_hc_post_ffn", detail=True):
                K.hc_post(gather(parts), h, post, comb, h)
            arrays[f"L{li}.h_moe"] = h.clone()
            pre, pre_f = pre_f, pre
    with rec.scope("head", detail=True):
        xc = K.collapse_norm(h, pre.contiguous(), w.norm, c.eps)
    arrays["head.xc"] = xc.clone()
    parts = [mm(heads[r], xc, F32) for r in range(world)]
    for r in range(world):
        arrays[f"head.local.r{r}"] = parts[r].clone()
    gth = gather(parts)
    logits = gth.permute(1, 0, 2).reshape(R, -1).contiguous()
    arrays["logits"] = logits
    torch.cuda.synchronize()
    arrays.update({f"after_{k_}": v.clone() for k_, v in pool_planes().items()})
    params = {"layers": L, "world": world, "slots": slots, "cap": cap, "bucket": bucket, "ring_size": pool.ring_size,
              "raw": RAW, "rows": R, "n_windows": len(windows), "rope_rows": cos_c.shape[0], "vocab_local": vocab_l,
              "k": k, "engram": int(eng is not None), "fused": int(fused)}
    for i, (slot, base, size, p0, toks) in enumerate(windows):
        params.update({f"w{i}_slot": slot, f"w{i}_base": base, f"w{i}_size": size, f"w{i}_pos": p0, f"w{i}_rows": len(toks)})
    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    import json
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    print(f"round of {R} rows over {len(windows)} streams at bucket {bucket}: argmax {logits.argmax(-1).tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
