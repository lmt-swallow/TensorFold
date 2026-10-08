#!/usr/bin/env python3
"""A fixture of DeepSeek-V4.1's attention sublayer for the Zig engine: the Python fork's decode-window attention
(graph.py ``StaticDecoder._attention``) on one backbone layer, rank 0 of 2, with its Triton launches recorded for the
AOT set (tools/zig/dsv41/attn_capture.sh packs them).

The caches are primed by the fork's own prompt path over a seeded prompt; each case then runs the sublayer on seeded
rows at the next positions from the same cache state and writes every input (x, positions, the layer's cache planes,
RoPE tables), the layer's weights (the rank's EXL3 parts as the loader slices them, plain tensors) and the outputs
(the fp32 partial, the top-k picks, the cache planes after).

Run inside the fork's image (TRITON_CACHE_DIR set before Python starts):
  python3 -B /w/tools/zig/dsv41/dump_attention.py --tools /w/tools/zig --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
      --out /aot/out --layer 2 --prompt 300 --rows 1,5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True, help="folder of triton_aot_manifest.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--prompt", type=int, default=300)
    ap.add_argument("--rows", default="1,5")
    ap.add_argument("--bucket", type=int, default=1024)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    ck = load_module("capture_kernel", HERE / "capture_kernel.py")
    rec = ck.load_recorder(a.tools)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from tensorfold.families.deepseek_v41.cuda import kernels as K  # noqa: F401  (switch defaults)
    from tensorfold.families.deepseek_v41.cuda.graph import StaticDecoder
    from tensorfold.families.deepseek_v41.cuda.model import KV_QUANT, RAW, Comm, Model
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, exl3_parts, load

    assert KV_QUANT, "the fixture is the packed FP4 cache path (TF_DS_KV native)"
    torch.cuda.set_device(0)
    rank, world = 0, 2
    L = a.layer
    w = load(a.model, rank, world, n_layers=L + 1, log=print, dspark=False)
    c = w.cfg
    m = Model(w, Comm(None, 1))
    lay = w.layers[L]
    ratio = c.compress_ratios[L]
    assert L != c.cand_source and not 0 <= c.cand_source < L, "candidate pools are not in this fixture"
    sc = m.new_cache(a.cap)
    g = torch.Generator(device="cuda")
    g.manual_seed(a.seed)
    ids = torch.randint(1000, 100000, (a.prompt,), generator=g, device="cuda")
    m.forward(sc, ids, 0)                                   # the fork's own prompt path primes every cache
    torch.cuda.synchronize()
    start = sc.length

    def planes() -> dict:
        d = {"ring": sc.ring[L]}
        if L in sc.comp:
            d["comp_codes"], d["comp_scales"] = sc.comp[L]
        if L in sc.index_k:
            d["idxk_codes"], d["idxk_scales"] = sc.index_k[L]
        if L in sc.comp_raw:
            d["raw_kv"], d["raw_score"] = sc.comp_raw[L]
        return d

    before = {k: v.clone() for k, v in planes().items()}
    cos, sin = m._cs(L, sc)
    Hl = c.n_heads // world
    hd = c.head_dim

    # -- the layer's weights, as weights.load_block slices them for this rank ---------------------------------------
    sh = Shards(a.model)
    p = f"layers.{L}"
    gl = c.o_groups // world
    groups = {
        "wq_a": (f"{p}.attn.wq_a", None, None),
        "wq_b": (f"{p}.attn.wq_b", (rank * Hl * hd, (rank + 1) * Hl * hd), None),
        "wkv": (f"{p}.attn.wkv", None, None),
        "wo_b": (f"{p}.attn.wo_b", None, (rank * gl * c.o_rank, (rank + 1) * gl * c.o_rank)),
    }
    for gi, gg in enumerate(range(rank * gl, (rank + 1) * gl)):
        groups[f"wo_a{gi}"] = (f"{p}.attn.wo_a.slice.{gg}", None, None)
    if lay.comp_wkv is not None:
        groups["comp_wkv"] = (f"{p}.attn.compressor.wkv", None, None)
        if lay.comp_wgate is not None:
            groups["comp_wgate"] = (f"{p}.attn.compressor.wgate", None, None)
    if lay.idx_wq_b is not None:
        groups["idx_wq_b"] = (f"{p}.attn.indexer.wq_b", None, None)
        if lay.idx_wk is not None:
            groups["idx_wk"] = (f"{p}.attn.indexer.wk", None, None)
    warr, wpar = {}, {"layer": L, "ratio": ratio, "rank": rank, "world": world, "heads_local": Hl, "head_dim": hd,
                      "rope_dim": c.rope_dim, "q_rank": c.q_rank, "o_rank": c.o_rank, "groups_local": gl,
                      "dim": c.dim, "window": c.window, "ring_size": sc.ring_size, "raw": RAW,
                      "idx_heads": c.idx_heads, "idx_dim": c.idx_dim, "idx_topk": c.idx_topk, "eps": c.eps,
                      "kv_source": int(lay.comp_wkv is not None), "index_source": int(lay.idx_wq_b is not None),
                      "has_idx_wk": int(lay.idx_wk is not None), "rope_rows": cos.shape[0],
                      "groups": ",".join(groups), "cases": a.rows}
    for name, (prefix, cols, rows) in groups.items():
        tr, suh, svh, cb = exl3_parts(sh, prefix, cols, rows, device=None)
        warr[f"{name}.trellis"], warr[f"{name}.suh"], warr[f"{name}.svh"] = tr, suh, svh
        wpar[f"{name}.kt"], wpar[f"{name}.nt"], wpar[f"{name}.w16"] = tr.shape[0], tr.shape[1], tr.shape[2]
        wpar[f"{name}.cb"] = cb
    warr["q_norm"] = lay.q_norm
    warr["kv_norm"] = lay.kv_norm
    warr["sink"] = lay.sink
    if lay.comp_norm is not None:
        warr["comp_norm"] = lay.comp_norm
    if lay.idx_k_norm is not None:
        warr["idx_k_norm"] = lay.idx_k_norm
    if lay.idx_proj is not None:
        warr["idx_proj_h"] = lay.idx_proj_h
    warr["cos"], warr["sin"] = cos, sin
    ck.save(out / "fixtures" / "attn_weights", wpar, warr)
    sh.close()

    # -- the cases: the sublayer on seeded rows at the next positions, from the primed caches --------------------
    cases = []
    for n in [int(v) for v in a.rows.split(",")]:
        for k, v in before.items():
            planes()[k].copy_(v)
        dec = StaticDecoder(m, sc, n, a.bucket, False)
        dec.pos.copy_(torch.arange(start, start + n, dtype=torch.long))
        x = torch.randn(n, c.dim, generator=g, device="cuda").to(torch.bfloat16)
        shared: dict = {}
        with rec.scope(f"attn_n{n}", detail=True):
            y = dec._attention(lay, x, shared)
        torch.cuda.synchronize()
        arrays = {"x": x, "pos": dec.pos.clone(), "expected_out": y}
        if "topk" in shared:
            arrays["expected_cidx"] = shared["topk"]
        for k, v in before.items():
            arrays[k] = v
        for k, v in planes().items():
            arrays[f"expected_{k}"] = v
        params = {"n": n, "start": start, "bucket": a.bucket, "cap": a.cap,
                  "comp_rows": int(before["comp_codes"].shape[0]) if "comp_codes" in before else 0,
                  "cidx_cols": int(shared["topk"].shape[1]) if "topk" in shared else 0}
        ck.save(out / "fixtures" / f"attn_n{n}", params, arrays)
        cases.append(n)
        print(f"case n={n}: out {tuple(y.shape)} {y.dtype}, cidx {tuple(shared['topk'].shape) if 'topk' in shared else None}", flush=True)
    # -- the FP4 kernels the Zig path launches where the graph path runs torch ops: their bytes checked equal ------
    from tensorfold.families.deepseek_v41.ops import fp4_pack, fp4_qd

    with rec.scope("fp4_kernels", detail=True):
        t = torch.randn(5, c.idx_heads * c.idx_dim, generator=g, device="cuda").to(torch.bfloat16)
        assert torch.equal(K.fp4_qd_p2(t), fp4_qd(t, 32, e4m3_scale=False)), "fp4_qd_p2 differs from ops.fp4_qd"
        for d, block, e4m3 in ((c.idx_dim, 32, False), (c.head_dim, 16, True)):
            xx = torch.randn(5, d, generator=g, device="cuda").to(torch.bfloat16)
            rows_t = torch.tensor([7, 1, 3, 0, 5], dtype=torch.long, device="cuda")
            cache = (torch.zeros((8, d // 2), dtype=torch.uint8, device="cuda"),
                     torch.zeros((8, d // block), dtype=torch.uint8, device="cuda"))
            K.fp4_store(xx, cache, rows_t, block, e4m3)
            codes, scales = fp4_pack(xx, block, e4m3)
            assert torch.equal(cache[0][rows_t], codes) and torch.equal(cache[1][rows_t], scales), f"fp4_store differs ({d}, {block}, {e4m3})"
        torch.cuda.synchronize()
    print("fp4_qd_p2 and fp4_store equal the torch ops", flush=True)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "attention.json").write_text(json.dumps({"layer": L, "cases": cases, "prompt": a.prompt, "bucket": a.bucket,
                                                    "cap": a.cap, "seed": a.seed}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
