#!/usr/bin/env python3
"""The DSpark drafter from the Python fork (dspark.py `Drafter.absorb`, `DraftGraph._body`) as a stage-by-stage
fixture for the Zig engine, with the Triton launches recorded for the AOT pack (capture_dspark.sh runs this, then
packs). One rank of world 1 (every head, every expert column), stages limited by --stages.

Absorb: seeded taps [n_abs, 3 d] at positions 0.. -> main_x and every stage's ring. Draft: token at q0 = n_abs ->
the block [token, noise x 4]: h0/pre0 (embed_init), then per stage x1, attention partial, h after post, x2, pick/wts,
MoE partial, h after post; then xc (collapse), xn (rmsnorm), the head's logits, the Markov drafts, the confidence.
The same calls the fork's DraftGraph._body makes, in its order (no CUDA graph: eager); then the real _body on a fresh
cache checks that the step-by-step drafts and confidences equal its own.

Run inside the fork's image with TRITON_CACHE_DIR set and TF_DS_MARKOV_CACHE=0 (no cached bias rows: the same bits,
fewer kernels to capture):
  python3 -B /w/tools/zig/dsv41/dump_dspark.py --tools /w/tools/zig --out /out --model /model [--stages 3] [--n-abs 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernel as ck  # noqa: E402

DEV = torch.device("cuda:0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--n-abs", type=int, default=200)
    ap.add_argument("--token", type=int, default=12345)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)
    import triton
    from tensorfold.families.deepseek_v41.cuda import dspark as D
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda import markov as MK
    from tensorfold.families.deepseek_v41.cuda import model as M
    from tensorfold.families.deepseek_v41.cuda.model import Comm, Model, mm, attn_in, wo_a_out
    from tensorfold.families.deepseek_v41.cuda.weights import load

    assert MK.CACHE_ROWS == 0, "run with TF_DS_MARKOV_CACHE=0"
    w = load(a.model, 0, 1, n_layers=0, dspark=True)
    cfg = w.cfg
    if a.stages < len(w.dspark.blocks):
        w.dspark.blocks = w.dspark.blocks[:a.stages]
    model = Model(w, Comm(None, 1), None)
    drafter = D.Drafter(model)
    d, hc = cfg.dim, cfg.hc
    n = drafter.size
    ring_size = drafter.ring_size
    sc = model.new_cache(4096)
    dc = drafter.new_cache()
    dw = w.dspark
    g = torch.Generator(device=DEV)
    g.manual_seed(20261009)

    # -- absorb --------------------------------------------------------------------------------------------------
    n_abs = a.n_abs
    taps = (torch.randn((n_abs, 3 * d), generator=g, device=DEV, dtype=torch.float32) * 0.8).to(torch.bfloat16)
    cos, sin = drafter._cs(dw.blocks[0], sc)
    with rec.scope("absorb", detail=True):
        drafter.absorb(dc, sc, taps, 0)
    keep = min(n_abs, ring_size)
    main_x = K.rmsnorm(mm(dw.main_proj, taps[-keep:].contiguous()), dw.main_norm, cfg.eps)
    torch.cuda.synchronize()
    arrays = {"taps": taps, "cos": cos, "sin": sin, "main_x": main_x, "main_norm": dw.main_norm, "norm": dw.norm,
              "markov_embed": dw.markov_embed, "markov_head": dw.markov_head, "conf_w": dw.conf,
              "emb2": w.embed[torch.tensor([a.token, cfg.dspark_noise], device=DEV)].contiguous()}
    for j, ring in enumerate(dc.rings):
        arrays[f"ring{j}_absorbed"] = ring.clone()
    params = {"D": d, "HC": hc, "eps": cfg.eps, "hc_eps": cfg.hc_eps, "iters": cfg.hc_iters, "block": n,
              "noise": cfg.dspark_noise, "vocab": cfg.vocab, "world": 1, "rank": 0, "heads": model.Hl, "hd": cfg.head_dim,
              "rd": cfg.rope_dim, "q_rank": cfg.q_rank, "o_rank": cfg.o_rank, "groups": len(dw.blocks[0].wo_a),
              "window": cfg.window, "ring": ring_size, "n_abs": n_abs, "q0": n_abs, "token": a.token,
              "stages": len(dw.blocks), "n_routed": cfg.dspark_routed, "topk": cfg.dspark_topk,
              "swiglu_limit": cfg.swiglu_limit, "route_scale": cfg.route_scale, "markov_rank": cfg.dspark_rank,
              "rope_rows": cos.shape[0], "markov_split": int(MK.SPLIT)}

    # EXL3 groups straight from the pack (the loader's splits at world 1 are the whole tensors)
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, exl3_parts
    sh = Shards(a.model)

    def group(name: str, prefix: str, cols=None, rows=None) -> None:
        tr, suh, svh, cb = exl3_parts(sh, prefix, cols=cols, rows=rows, device=None)
        arrays[f"{name}_trellis"], arrays[f"{name}_suh"], arrays[f"{name}_svh"] = tr, suh, svh
        params[f"{name}_kt"], params[f"{name}_nt"], params[f"{name}_w16"], params[f"{name}_cb"] = tr.shape[0], tr.shape[1], tr.shape[2], cb

    group("main_proj", "mtp.0.main_proj")
    vocab_l = cfg.vocab
    group("head", "head", cols=(0, vocab_l))
    for j, lay in enumerate(dw.blocks):
        p = f"mtp.{j}"
        arrays[f"s{j}_hc_attn_fn"], arrays[f"s{j}_hc_attn_scale"], arrays[f"s{j}_hc_attn_base"] = lay.hc_attn
        arrays[f"s{j}_hc_ffn_fn"], arrays[f"s{j}_hc_ffn_scale"], arrays[f"s{j}_hc_ffn_base"] = lay.hc_ffn
        arrays[f"s{j}_attn_norm"], arrays[f"s{j}_ffn_norm"] = lay.attn_norm, lay.ffn_norm
        arrays[f"s{j}_q_norm"], arrays[f"s{j}_kv_norm"], arrays[f"s{j}_sink"] = lay.q_norm, lay.kv_norm, lay.sink
        arrays[f"s{j}_gate_w"], arrays[f"s{j}_gate_b"] = lay.gate_w, lay.gate_b
        group(f"s{j}_wq_a", f"{p}.attn.wq_a")
        group(f"s{j}_wq_b", f"{p}.attn.wq_b", cols=(0, model.Hl * cfg.head_dim))
        group(f"s{j}_wkv", f"{p}.attn.wkv")
        for gi in range(len(lay.wo_a)):
            group(f"s{j}_wo_a{gi}", f"{p}.attn.wo_a.slice.{gi}")
        group(f"s{j}_wo_b", f"{p}.attn.wo_b", rows=(0, len(lay.wo_a) * cfg.o_rank))
        for e in range(cfg.dspark_routed + 1):
            ep = f"{p}.ffn.experts.{e}" if e < cfg.dspark_routed else f"{p}.ffn.shared_experts"
            group(f"s{j}_e{e}_gate", f"{ep}.w1", cols=(0, cfg.inter))
            group(f"s{j}_e{e}_up", f"{ep}.w3", cols=(0, cfg.inter))
            group(f"s{j}_e{e}_down", f"{ep}.w2", rows=(0, cfg.inter))
    sh.close()

    # -- draft, step by step (DraftGraph._body's calls, eager) --------------------------------------------------
    q0 = n_abs
    assert dc.absorbed == q0, (dc.absorbed, q0)
    ids = torch.full((n,), cfg.dspark_noise, dtype=torch.long, device=DEV)
    ids[0] = a.token
    pos = q0 + torch.arange(n, device=DEV)
    wpos = torch.full((n,), q0 - 1, dtype=torch.int64, device=DEV)
    ids_small = torch.tensor([0] + [1] * (n - 1), dtype=torch.int64, device=DEV)
    with rec.scope("embed", detail=True):
        h, pre = K.embed_init(arrays["emb2"], ids_small, hc)
    h_ref = w.embed[ids].to(torch.bfloat16)[:, None, :].expand(-1, hc, -1).contiguous()
    assert torch.equal(h, h_ref), "embed_init differs from the indexed rows"
    arrays["h0"], arrays["pre0"] = h.clone(), pre.clone()
    x = torch.empty((n, d), dtype=torch.bfloat16, device=DEV)
    part = torch.empty((n * K.HC_BLOCKS * 32,), dtype=torch.float32, device=DEV)
    pre_a = torch.empty((n, hc), dtype=torch.float32, device=DEV)
    pre_f = torch.empty((n, hc), dtype=torch.float32, device=DEV)
    post = torch.empty((n, hc), dtype=torch.float32, device=DEV)
    comb = torch.empty((n, hc, hc), dtype=torch.float32, device=DEV)
    for j, (lay, ring) in enumerate(zip(dw.blocks, dc.rings)):
        fn, scale, base = lay.hc_attn
        with rec.scope(f"s{j}_hc_pre_attn", detail=True):
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x, pre_a, post, comb, part)
        arrays[f"s{j}_x1"] = x.clone()
        with rec.scope(f"s{j}_attention", detail=True):
            pa = drafter._attention(lay, x, sc, ring, None, pos=pos, wpos=wpos)
        arrays[f"s{j}_attn"] = pa.clone()
        with rec.scope(f"s{j}_hc_post_attn", detail=True):
            K.hc_post(model.comm.gather(pa), h, post, comb, h)
        arrays[f"s{j}_h1"] = h.clone()
        fn, scale, base = lay.hc_ffn
        with rec.scope(f"s{j}_hc_pre_ffn", detail=True):
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x, pre_f, post, comb, part)
        arrays[f"s{j}_x2"] = x.clone()
        with rec.scope(f"s{j}_moe", detail=True):
            pm = model.moe(lay, x, topk=drafter.topk)
        arrays[f"s{j}_moe"] = pm.clone()
        with rec.scope(f"s{j}_hc_post_ffn", detail=True):
            K.hc_post(model.comm.gather(pm), h, post, comb, h)
        arrays[f"s{j}_h2"] = h.clone()
        pre, pre_f = pre_f, pre
    with rec.scope("collapse", detail=True):
        xc = K.collapse(h, pre)
    with rec.scope("final_norm", detail=True):
        xn = K.rmsnorm(xc, dw.norm, cfg.eps)
    local = mm(w.head, xn, torch.float32)
    arrays.update({"xc": xc, "xn": xn, "local": local})
    mk = drafter.markov()
    assert mk is not None, "TF_DS_MARKOV must be on"
    out_t = torch.empty((n + 1,), dtype=torch.long, device=DEV)
    out_t[0:1] = torch.tensor([a.token], device=DEV)
    with rec.scope("markov", detail=True):
        mk.steps(mk.logits_of(local), out_t.view(1, n + 1), n, n)
    embs = dw.markov_embed[out_t[:n]]
    conf_in = torch.cat([xc.float(), embs.float()], -1)
    conf = (conf_in @ dw.conf.float().t())[:, 0]
    torch.cuda.synchronize()
    arrays.update({"drafts": out_t[1:].clone(), "conf": conf.clone()})

    # -- the fork's own graph body on a fresh cache, the same absorb: the drafts must be these ---------------------
    dc2 = drafter.new_cache()
    drafter.absorb(dc2, sc, taps, 0)
    dg = D.DraftGraph(drafter, sc, dc2)
    dg.token.fill_(a.token)
    dg.q0.fill_(q0)
    dg._body()
    torch.cuda.synchronize()
    assert torch.equal(dg.out, out_t[1:]), (dg.out.tolist(), out_t[1:].tolist())
    assert torch.equal(dg.conf, conf), (dg.conf.tolist(), conf.tolist())
    for j, (r1, r2) in enumerate(zip(dc.rings, dc2.rings)):
        assert torch.equal(r1, r2), f"ring {j} differs between absorbs"

    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                                                   "pdl": K.SMALL_SWITCHES["pdl"], "stages": len(dw.blocks),
                                                   "markov_on": MK.ON, "markov_split": MK.SPLIT,
                                                   "cache_rows": MK.CACHE_ROWS}, indent=1) + "\n")
    print(f"drafts after {a.token} at q0={q0}: {out_t[1:].tolist()} conf={[round(c, 4) for c in conf.tolist()]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
