#!/usr/bin/env python3
"""The mHC glue and MoE routing of one DeepSeek-V4.1 layer from the Python fork, as a stage-by-stage fixture for the
Zig engine, with the Triton launches recorded for the AOT pack (capture_hc.sh runs this, then packs).

For rows n in NS, with layer LAYER's real weights (rank 0 of world 2) and seeded inputs:
  embed_init -> h0, pre0; hc_pre(attn) -> x1, pre_a, post_a, comb_a; hc_post(g_a) -> h1; hc_pre(ffn) -> x2, pre_f,
  post_f, comb_f; rowmm_gate + route -> S, pick, wts; hc_post(g_f) -> h2; tap -> taps; collapse_norm -> xc;
  EXL3 head (this rank's vocabulary columns) -> logits; engram_gate(h0, kv) -> h_e.
Every array is written raw (fixture.zig's manifest.json format): inputs and the fork's outputs by stage.

Run inside the fork's image with TRITON_CACHE_DIR set:
  python3 -B /w/tools/zig/dsv41/dump_hc.py --tools /w/tools/zig --out /out --model /models/<pack> --layer 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernel as ck  # noqa: E402  (its recorder loader, fixture writer, jit specialization dump)

DEV = torch.device("cuda:0")
NS = (1, 5, 16)
WORLD = 2
RANK = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--engram-layer", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)
    import triton
    from tensorfold.cuda.exl3.linear import Exl3Linear
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, exl3_parts

    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    d, hc = cfg.dim, cfg.hc
    p = f"layers.{a.layer}"

    def plain(name: str, dtype=None) -> torch.Tensor:
        t = sh.get(name)
        return (t.to(dtype) if dtype is not None else t).contiguous().to(DEV)

    fn_a, sc_a, ba_a = plain(f"{p}.hc_attn_fn", torch.float32), plain(f"{p}.hc_attn_scale", torch.float32), plain(f"{p}.hc_attn_base", torch.float32)
    fn_f, sc_f, ba_f = plain(f"{p}.hc_ffn_fn", torch.float32), plain(f"{p}.hc_ffn_scale", torch.float32), plain(f"{p}.hc_ffn_base", torch.float32)
    attn_norm, ffn_norm, norm_w = plain(f"{p}.attn_norm.weight"), plain(f"{p}.ffn_norm.weight"), plain("norm.weight")
    gate_w, gate_b = plain(f"{p}.ffn.gate.weight", torch.float16), plain(f"{p}.ffn.gate.bias", torch.float32)
    qk = (plain(f"layers.{a.engram_layer}.engram.q_weight", torch.float32) * plain(f"layers.{a.engram_layer}.engram.k_weight", torch.float32)).contiguous()
    embed = sh.get("embed.weight")                                              # CPU bf16 [V, d]
    vocab_lo, vocab_hi = RANK * cfg.vocab // WORLD, (RANK + 1) * cfg.vocab // WORLD
    tr, suh, svh, cb = exl3_parts(sh, "head", cols=(vocab_lo, vocab_hi), device=None)
    head = Exl3Linear.from_tensors(tr.to(DEV), suh.to(DEV), svh.to(DEV), cb)
    shared_id = cfg.n_routed                                                   # the shared expert's slot id
    slots = cfg.topk + 1

    g = torch.Generator(device=DEV)
    g.manual_seed(20261009)
    gc = torch.Generator(device="cpu")
    gc.manual_seed(20261009)
    arrays: dict[str, torch.Tensor] = {
        "hc_attn_fn": fn_a, "hc_attn_scale": sc_a, "hc_attn_base": ba_a, "hc_ffn_fn": fn_f, "hc_ffn_scale": sc_f,
        "hc_ffn_base": ba_f, "attn_norm": attn_norm, "ffn_norm": ffn_norm, "norm_w": norm_w, "gate_w": gate_w,
        "gate_b": gate_b, "engram_qk": qk, "head_trellis": tr, "head_suh": suh, "head_svh": svh,
    }
    params = {"D": d, "HC": hc, "eps": cfg.eps, "hc_eps": cfg.hc_eps, "iters": cfg.hc_iters, "K": d, "N": cfg.n_routed,
              "topk": cfg.topk, "slots": slots, "scale": cfg.route_scale, "shared_id": shared_id, "world": WORLD,
              "vocab_lo": vocab_lo, "vocab_hi": vocab_hi, "head_kt": tr.shape[0], "head_nt": tr.shape[1],
              "head_w16": tr.shape[2], "head_cb": cb, "layer": a.layer, "n_count": len(NS), "HC_BLOCKS": K.HC_BLOCKS}
    for i, n in enumerate(NS):
        params[f"n{i}"] = n
        pre = f"n{n}_"
        ids = torch.randint(0, cfg.vocab, (n,), generator=gc).to(torch.int64)
        emb_small = embed[ids].contiguous().to(DEV)                             # the rows the ids pick, renumbered
        ids_small = torch.arange(n, dtype=torch.int64, device=DEV)
        with rec.scope(f"embed_n{n}", detail=True):
            h0, pre0 = K.embed_init(emb_small, ids_small, hc)
        x1 = torch.empty((n, d), dtype=torch.bfloat16, device=DEV)
        part = torch.empty((n * K.HC_BLOCKS * 32,), dtype=torch.float32, device=DEV)
        pre_a, post_a = torch.empty((n, hc), dtype=torch.float32, device=DEV), torch.empty((n, hc), dtype=torch.float32, device=DEV)
        comb_a = torch.empty((n, hc, hc), dtype=torch.float32, device=DEV)
        with rec.scope(f"hc_pre_attn_n{n}", detail=True):
            K.hc_pre(h0, fn_a, sc_a, ba_a, pre0, attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x1, pre_a, post_a, comb_a, part)
        g_a = torch.randn((WORLD, n, d), generator=g, device=DEV, dtype=torch.float32) * 0.5
        h1 = torch.empty_like(h0)
        with rec.scope(f"hc_post_attn_n{n}", detail=True):
            K.hc_post(g_a, h0, post_a, comb_a, h1)
        x2 = torch.empty((n, d), dtype=torch.bfloat16, device=DEV)
        pre_f, post_f = torch.empty((n, hc), dtype=torch.float32, device=DEV), torch.empty((n, hc), dtype=torch.float32, device=DEV)
        comb_f = torch.empty((n, hc, hc), dtype=torch.float32, device=DEV)
        with rec.scope(f"hc_pre_ffn_n{n}", detail=True):
            K.hc_pre(h1, fn_f, sc_f, ba_f, pre_a, ffn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x2, pre_f, post_f, comb_f, part)
        pick = torch.empty((n, slots), dtype=torch.int32, device=DEV)
        wts = torch.empty((n, slots), dtype=torch.float32, device=DEV)
        with rec.scope(f"route_n{n}", detail=True):
            logits = K.rowmm_gate(x2, gate_w)
            assert logits.dim() == 3, "the fork took the rowmm_parts path at these rows: the Zig port follows it"
            K.route(logits, gate_b, cfg.topk, cfg.route_scale, shared_id, pick, wts)
        g_f = torch.randn((WORLD, n, d), generator=g, device=DEV, dtype=torch.float32) * 0.5
        h2 = torch.empty_like(h0)
        with rec.scope(f"hc_post_ffn_n{n}", detail=True):
            K.hc_post(g_f, h1, post_f, comb_f, h2)
        taps = torch.zeros((n, 3 * d), dtype=torch.bfloat16, device=DEV)
        with rec.scope(f"tap_n{n}", detail=True):
            K.tap(h2, taps[:, 0:d])
        with rec.scope(f"collapse_n{n}", detail=True):
            xc = K.collapse_norm(h2, pre_f, norm_w, cfg.eps)
        logits_v = head(xc, out_dtype=torch.float32)
        kv = (torch.randn((n, 5 * d), generator=g, device=DEV, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        with rec.scope(f"engram_gate_n{n}", detail=True):
            h_e = K.engram_gate(h0, kv, qk, cfg.eps)
        torch.cuda.synchronize()
        arrays.update({
            pre + "ids": ids_small, pre + "emb": emb_small, pre + "h0": h0, pre + "pre0": pre0,
            pre + "x1": x1, pre + "pre_a": pre_a, pre + "post_a": post_a, pre + "comb_a": comb_a,
            pre + "g_a": g_a, pre + "h1": h1,
            pre + "x2": x2, pre + "pre_f": pre_f, pre + "post_f": post_f, pre + "comb_f": comb_f,
            pre + "S": logits, pre + "pick": pick, pre + "wts": wts,
            pre + "g_f": g_f, pre + "h2": h2, pre + "taps": taps, pre + "xc": xc, pre + "logits": logits_v,
            pre + "kv": kv, pre + "h_e": h_e,
        })
        print(f"n={n}: pick[0]={pick[0].tolist()} wts[0]={[round(v, 4) for v in wts[0].tolist()]} "
              f"argmax={int(logits_v[0].argmax())}", flush=True)
    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                                                   "pdl": K.SMALL_SWITCHES["pdl"], "rowmm_parts": K.SMALL_SWITCHES["rowmm_parts"],
                                                   "rowmm": K.SMALL_SWITCHES["rowmm"], "ns": NS}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
