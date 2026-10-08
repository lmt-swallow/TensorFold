#!/usr/bin/env python3
"""The DSpark drafter for several streams at once, from the Python fork (dspark.py `DraftPool`, `absorb_many`,
`BatchDraftGraph._body`), as a fixture for the Zig engine's batched drafts: N streams in a 4-slot pool, each
absorbed to its own length (positions 0 ..), then one batched draft pass, stage by stage, unfused
(TF_DS_DRAFT_FUSED=0: the plain hc_pre / hc_post launches and `_attention(fused=False)`, which the fork documents as
the same arithmetic as its fused path). World 1, every column; stages limited by --stages.

Run inside the fork's image with TRITON_CACHE_DIR set, TF_DS_MARKOV_CACHE=0 and TF_DS_DRAFT_FUSED=0:
  python3 -B /w/tools/zig/dsv41/dump_draft_pool.py --tools /w/tools/zig --out /out --model /model [--stages 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernel as ck  # noqa: E402
from dump_dspark import DEV  # noqa: E402

# absorbed rows total under 128 a case: the fork's absorb_many concatenates every stream's taps, and mm takes its
# cuBLAS prompt GEMM above 128 rows (a prompt-time path; a round's absorb is 16 rows at most)
CASES = {2: ([60, 40], [12345, 777], [0, 2]), 3: ([50, 40, 30], [12345, 777, 4321], [1, 0, 3])}
SLOTS = 4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stages", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)
    import triton
    from tensorfold.families.deepseek_v41.cuda import dspark as D
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda import markov as MK
    from tensorfold.families.deepseek_v41.cuda.model import Comm, Model, mm
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, exl3_parts, load

    assert MK.CACHE_ROWS == 0, "run with TF_DS_MARKOV_CACHE=0"
    assert not D.FUSED, "run with TF_DS_DRAFT_FUSED=0"
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
    dw = w.dspark
    cos, sin = drafter._cs(dw.blocks[0], sc)
    arrays = {"cos": cos, "sin": sin, "main_norm": dw.main_norm, "norm": dw.norm, "markov_embed": dw.markov_embed,
              "markov_head": dw.markov_head, "conf_w": dw.conf}
    params = {"D": d, "HC": hc, "eps": cfg.eps, "hc_eps": cfg.hc_eps, "iters": cfg.hc_iters, "block": n,
              "noise": cfg.dspark_noise, "vocab": cfg.vocab, "world": 1, "rank": 0, "heads": model.Hl, "hd": cfg.head_dim,
              "rd": cfg.rope_dim, "q_rank": cfg.q_rank, "o_rank": cfg.o_rank, "groups": len(dw.blocks[0].wo_a),
              "window": cfg.window, "ring": ring_size, "slots": SLOTS, "stages": len(dw.blocks),
              "n_routed": cfg.dspark_routed, "topk": cfg.dspark_topk, "swiglu_limit": cfg.swiglu_limit,
              "route_scale": cfg.route_scale, "markov_rank": cfg.dspark_rank, "rope_rows": cos.shape[0],
              "markov_split": int(MK.SPLIT), "n_cases": len(CASES)}
    sh = Shards(a.model)

    def group(name: str, prefix: str, cols=None, rows=None) -> None:
        tr, suh, svh, cb = exl3_parts(sh, prefix, cols=cols, rows=rows, device=None)
        arrays[f"{name}_trellis"], arrays[f"{name}_suh"], arrays[f"{name}_svh"] = tr, suh, svh
        params[f"{name}_kt"], params[f"{name}_nt"], params[f"{name}_w16"], params[f"{name}_cb"] = tr.shape[0], tr.shape[1], tr.shape[2], cb

    group("main_proj", "mtp.0.main_proj")
    group("head", "head", cols=(0, cfg.vocab))
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

    for ci, (N, (lens, tokens, slots)) in enumerate(CASES.items()):
        g = torch.Generator(device=DEV)
        g.manual_seed(20261009 + N)
        dpool = D.DraftPool(drafter, SLOTS)
        items = []
        for s_, (L, slot) in enumerate(zip(lens, slots)):
            taps = (torch.randn((L, 3 * d), generator=g, device=DEV, dtype=torch.float32) * 0.8).to(torch.bfloat16)
            arrays[f"c{ci}_taps{s_}"] = taps
            items.append((slot, taps, 0))
        with rec.scope(f"c{ci}_absorb", detail=True):
            drafter.absorb_many(dpool, sc, items)
        torch.cuda.synchronize()
        for j, ring in enumerate(dpool.rings):
            arrays[f"c{ci}_ring{j}_absorbed"] = ring.clone()
        q0 = [dpool.views[s].absorbed for s in slots]
        assert q0 == lens, (q0, lens)
        R = N * n
        ids = torch.full((N, n), cfg.dspark_noise, dtype=torch.long, device=DEV)
        ids[:, 0] = torch.tensor(tokens, device=DEV)
        ids = ids.view(R)
        q0_t = torch.tensor(q0, dtype=torch.long, device=DEV)
        slots_t = torch.tensor(slots, dtype=torch.long, device=DEV)
        pos = (q0_t[:, None] + torch.arange(n, device=DEV)[None]).reshape(R)
        wpos = (q0_t - 1)[:, None].expand(N, n).reshape(R).contiguous()
        rbase = (slots_t * dpool.ring_size)[:, None].expand(N, n).reshape(R).contiguous()
        bg = D.BatchDraftGraph(drafter, sc, dpool, N, steps=None)
        emb_small = w.embed[torch.tensor(tokens + [cfg.dspark_noise], device=DEV)].contiguous()
        ids_small = torch.tensor([s_ if r % n == 0 else N for r in range(R) for s_ in [r // n]], dtype=torch.long, device=DEV)
        with rec.scope(f"c{ci}_embed", detail=True):
            h, pre = K.embed_init(emb_small, ids_small, hc)
        assert torch.equal(h, w.embed[ids].to(torch.bfloat16)[:, None, :].expand(-1, hc, -1).contiguous())
        arrays[f"c{ci}_emb"] = emb_small
        arrays[f"c{ci}_h0"] = h.clone()
        x = torch.empty((R, d), dtype=torch.bfloat16, device=DEV)
        part = torch.empty((R * K.HC_BLOCKS * 32,), dtype=torch.float32, device=DEV)
        pre_a = torch.empty((R, hc), dtype=torch.float32, device=DEV)
        pre_f = torch.empty((R, hc), dtype=torch.float32, device=DEV)
        post = torch.empty((R, hc), dtype=torch.float32, device=DEV)
        comb = torch.empty((R, hc, hc), dtype=torch.float32, device=DEV)
        for j, (lay, ring) in enumerate(zip(dw.blocks, dpool.rings)):
            fn, scale, base = lay.hc_attn
            with rec.scope(f"c{ci}_s{j}_hc_pre_attn", detail=True):
                K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x, pre_a, post, comb, part)
            arrays[f"c{ci}_s{j}_x1"] = x.clone()
            with rec.scope(f"c{ci}_s{j}_attention", detail=True):
                pa = bg._attention(lay, x, ring, pos, wpos, rbase)
            arrays[f"c{ci}_s{j}_attn"] = pa.clone()
            with rec.scope(f"c{ci}_s{j}_hc_post_attn", detail=True):
                K.hc_post(model.comm.gather(pa), h, post, comb, h)
            arrays[f"c{ci}_s{j}_h1"] = h.clone()
            fn, scale, base = lay.hc_ffn
            with rec.scope(f"c{ci}_s{j}_hc_pre_ffn", detail=True):
                K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x, pre_f, post, comb, part)
            arrays[f"c{ci}_s{j}_x2"] = x.clone()
            with rec.scope(f"c{ci}_s{j}_moe", detail=True):
                pm = model.moe(lay, x, topk=drafter.topk)
            arrays[f"c{ci}_s{j}_moe"] = pm.clone()
            with rec.scope(f"c{ci}_s{j}_hc_post_ffn", detail=True):
                K.hc_post(model.comm.gather(pm), h, post, comb, h)
            arrays[f"c{ci}_s{j}_h2"] = h.clone()
            pre, pre_f = pre_f, pre
        with rec.scope(f"c{ci}_collapse", detail=True):
            xc = K.collapse(h, pre)
        with rec.scope(f"c{ci}_final_norm", detail=True):
            xn = K.rmsnorm(xc, dw.norm, cfg.eps)
        local = mm(w.head, xn, torch.float32)
        mk = bg.mk
        assert mk is not None
        steps = bg.steps
        out_t = torch.empty((N, steps + 1), dtype=torch.long, device=DEV)
        out_t[:, 0] = torch.tensor(tokens, device=DEV)
        with rec.scope(f"c{ci}_markov", detail=True):
            mk.steps(mk.logits_of(local), out_t, n, steps)
        embs = dw.markov_embed[out_t[:, :steps]]
        conf_in = torch.cat([xc.float().view(N, n, -1)[:, :steps], embs.float()], -1)
        conf = (conf_in @ dw.conf.float().t())[..., 0]
        torch.cuda.synchronize()
        # the fork's own batched pass on the same pool: the same drafts and confidences
        drafts_ref = bg.run(tokens, q0, slots)
        torch.cuda.synchronize()
        assert drafts_ref == out_t[:, 1:].tolist(), (drafts_ref, out_t[:, 1:].tolist())
        conf_ref = torch.tensor(bg.last_conf, device=DEV)
        assert torch.equal(conf_ref, conf), (bg.last_conf, conf.tolist())
        arrays.update({f"c{ci}_xc": xc, f"c{ci}_xn": xn, f"c{ci}_local": local, f"c{ci}_drafts": out_t[:, 1:].contiguous(),
                       f"c{ci}_conf": conf.contiguous()})
        params.update({f"c{ci}_N": N, f"c{ci}_steps": steps})
        for s_ in range(N):
            params.update({f"c{ci}_len{s_}": lens[s_], f"c{ci}_token{s_}": tokens[s_], f"c{ci}_slot{s_}": slots[s_]})
        print(f"case {ci}: N={N} lens={lens} slots={slots} drafts={out_t[:, 1:].tolist()}", flush=True)
    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                                                   "pdl": K.SMALL_SWITCHES["pdl"], "stages": len(dw.blocks),
                                                   "fused": D.FUSED, "cache_rows": MK.CACHE_ROWS}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
