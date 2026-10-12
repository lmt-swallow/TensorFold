#!/usr/bin/env python3
"""The fused mHC (kernels.hc_pre2 with `gathered`: `_hc_mix_part` posted, then `_hc_finish_s`) against the unfused
pair the prompt walk runs (`_hc_post`, then hc_pre's `_hc_partial` + `_hc_finish`) at prompt row counts, byte for
byte, with the fused launches recorded for the AOT pack (capture_hc_mix.sh runs this, then packs).

For rows n in NS, with layer LAYER's real weights (rank 0 of world 2) and seeded inputs:
  embed_init -> h0, pre0; hc_pre(attn) -> pre_a, post_a, comb_a; gathered g [2, n, D] f32;
  unfused: hc_post(g, h0) -> h1; hc_pre(h1, ffn) -> x2, pre_f, post_f, comb_f;
  fused:   hc_pre2(h0, ffn, gathered=g, h_out=h1') -> h1', x2', pre_f', post_f', comb_f'.
Every output pair must be byte-equal. The sensitivity counts say what the comparison resolves: elements of h1 that
the fork's documented post expression without contraction (ps y + (((c0 x0 + c1 x1) + c2 x2) + c3 x3)) changes, and
elements of x2 / pre_f / post_f / comb_f that change when the 40 partial dots are summed in torch's order instead
(a probe that matched by construction would show zero here).
At FIXTURE_ROWS the unfused pair's inputs and outputs (h0, g, post_a, comb_a, fn_f -> h1 and the partial sums part)
are also written as a fixture: `tf-cuda-test dsv41-hc-fused-probe <aot-dir> <out>/fixture` holds hc_fused.cu to the
same comparison.

Run inside the fork's image with TRITON_CACHE_DIR set:
  python3 -B /w/tools/zig/dsv41/dump_hc_mix.py --tools /w/tools/zig --out /out --model /models/<pack> --layer 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernel as ck  # noqa: E402  (its recorder loader, fixture writer, jit specialization dump)

DEV = torch.device("cuda:0")
NS = (1, 2, 3, 7, 64, 777, 2048)       # every rows-a-program value of _hc_rb (1, 2, 3, 4, 8) and the prompt's sizes
WORLD = 2
FIXTURE_ROWS = (2048, 777, 64, 1)


def differ(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements of a and b whose bytes differ (same shape and dtype)."""

    ea = a.contiguous().view(-1)
    eb = b.contiguous().view(-1)
    w = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[ea.element_size()]
    return int((ea.view(w) != eb.view(w)).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)
    import triton
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda.weights import Shards

    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    d, hc, nb = cfg.dim, cfg.hc, K.HC_BLOCKS
    p = f"layers.{a.layer}"

    def plain(name: str, dtype=None) -> torch.Tensor:
        t = sh.get(name)
        return (t.to(dtype) if dtype is not None else t).contiguous().to(DEV)

    fn_a, sc_a, ba_a = plain(f"{p}.hc_attn_fn", torch.float32), plain(f"{p}.hc_attn_scale", torch.float32), plain(f"{p}.hc_attn_base", torch.float32)
    fn_f, sc_f, ba_f = plain(f"{p}.hc_ffn_fn", torch.float32), plain(f"{p}.hc_ffn_scale", torch.float32), plain(f"{p}.hc_ffn_base", torch.float32)
    attn_norm, ffn_norm = plain(f"{p}.attn_norm.weight"), plain(f"{p}.ffn_norm.weight")
    embed = sh.get("embed.weight")
    g = torch.Generator(device=DEV)
    g.manual_seed(20261012)
    gc = torch.Generator(device="cpu")
    gc.manual_seed(20261012)
    report: dict[str, dict] = {}
    arrays: dict[str, torch.Tensor] = {"fn_f": fn_f}
    bad = 0
    for n in NS:
        ids = torch.randint(0, cfg.vocab, (n,), generator=gc).to(torch.int64)
        emb = embed[ids].contiguous().to(DEV)
        h0, pre0 = K.embed_init(emb, torch.arange(n, dtype=torch.int64, device=DEV), hc)

        def mixes():
            return (torch.empty((n, d), dtype=torch.bfloat16, device=DEV),
                    torch.empty((n, hc), dtype=torch.float32, device=DEV),
                    torch.empty((n, hc), dtype=torch.float32, device=DEV),
                    torch.empty((n, hc, hc), dtype=torch.float32, device=DEV),
                    torch.empty((n * nb * 32,), dtype=torch.float32, device=DEV))

        x1, pre_a, post_a, comb_a, part = mixes()
        K.hc_pre(h0, fn_a, sc_a, ba_a, pre0, attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x1, pre_a, post_a, comb_a, part)
        gath = torch.randn((WORLD, n, d), generator=g, device=DEV, dtype=torch.float32) * 0.5

        # unfused: what the prompt walk runs today
        h1 = torch.empty_like(h0)
        K.hc_post(gath, h0, post_a, comb_a, h1)
        x2, pre_f, post_f, comb_f, part_u = mixes()
        K.hc_pre(h1, fn_f, sc_f, ba_f, pre_a, ffn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x2, pre_f, post_f, comb_f, part_u)

        # fused: post / comb are read by the mixes, then rewritten by the finish (one buffer each, as in the walk)
        h1f = torch.empty_like(h0)
        x2f, pre_ff, _, _, part_f = mixes()
        post_ff, comb_ff = post_a.clone(), comb_a.clone()
        with rec.scope(f"hc_mix_n{n}", detail=True):
            src = K.hc_pre2(h0, fn_f, sc_f, ba_f, pre_a, ffn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x2f, pre_ff,
                            post_ff, comb_ff, part_f, gathered=gath, h_out=h1f)
        torch.cuda.synchronize()
        assert src.data_ptr() == h1f.data_ptr()
        eq = {"h": differ(h1, h1f), "x": differ(x2, x2f), "pre": differ(pre_f, pre_ff), "post": differ(post_f, post_ff),
              "comb": differ(comb_f, comb_ff)}

        # sensitivity: the post without contraction, and the partial dots in torch's order
        x = h0.float()
        y = gath.sum(0).to(torch.bfloat16).float()
        cm = comb_a                                                   # [n, j, s]
        alt = (post_a[:, :, None] * y[:, None, :]
               + (((cm[:, 0, :, None] * x[:, 0:1, :] + cm[:, 1, :, None] * x[:, 1:2, :]) + cm[:, 2, :, None] * x[:, 2:3, :])
                  + cm[:, 3, :, None] * x[:, 3:4, :])).to(torch.bfloat16)
        hb = h1.float().view(n, nb, 4 * d // nb)
        fb = fn_f.view(fn_f.shape[0], nb, 4 * d // nb)
        part_t = torch.zeros((n, nb, 32), dtype=torch.float32, device=DEV)
        part_t[:, :, :fn_f.shape[0]] = torch.einsum("rbk,mbk->rbm", hb, fb)
        part_t[:, :, 24] = (hb * hb).sum(-1)
        x2t, pre_t, post_t, comb_t, _ = mixes()
        K._hc_finish[(n,)](h1, part_t.view(-1), ba_f, sc_f, pre_a, ffn_norm, x2t, pre_t, post_t, comb_t, cfg.eps,
                           cfg.hc_eps, D=d, NB=nb, ITERS=cfg.hc_iters, BLOCK=1024, num_warps=8)
        torch.cuda.synchronize()
        sens = {"h": differ(h1, alt), "x": differ(x2, x2t), "pre": differ(pre_f, pre_t), "post": differ(post_f, post_t),
                "comb": differ(comb_f, comb_t)}
        sizes = {"h": h1.numel(), "x": x2.numel(), "pre": pre_f.numel(), "post": post_f.numel(), "comb": comb_f.numel()}
        if n in FIXTURE_ROWS:
            arrays.update({f"n{n}_h0": h0, f"n{n}_g": gath, f"n{n}_post": post_a, f"n{n}_comb": comb_a, f"n{n}_h1": h1,
                           f"n{n}_part": part_u})
        report[str(n)] = {"rb": K._hc_rb(n), "differ": eq, "sensitivity": sens, "elements": sizes}
        bad += sum(eq.values())
        print(f"n={n} rb={K._hc_rb(n)}: fused vs unfused differ {eq}; sensitivity {sens} of {sizes}", flush=True)
    params = {"D": d, "world": WORLD, "layer": a.layer, "n_count": len(FIXTURE_ROWS)}
    params.update({f"n{i}": n for i, n in enumerate(FIXTURE_ROWS)})
    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "probe.json").write_text(json.dumps({"layer": a.layer, "torch": torch.__version__, "triton": triton.__version__,
                                                "switches": {k: K.SMALL_SWITCHES[k] for k in ("hc", "hc_xpf", "hc_rb", "hc_split", "pdl")},
                                                "rows": report}, indent=1) + "\n")
    print("byte-equal at every row count" if bad == 0 else f"MISMATCH: {bad} elements differ", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
