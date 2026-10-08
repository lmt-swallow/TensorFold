#!/usr/bin/env python3
"""Compile a sample of the DeepSeek-V4.1 fork's heavier Triton kernels on small shapes and report what the Zig
engine's aot.Set would refuse: global or profile scratch, clusters (num_ctas > 1). Run inside the fork's image with
TRITON_CACHE_DIR set to an empty directory; prints one line per compiled kernel."""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import torch

DEV = torch.device("cuda:0")


def main() -> int:
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    try:
        from tensorfold.families.deepseek_v41.cuda import markov as M
    except Exception as e:  # noqa: BLE001
        M, m_err = None, e
    n, d, hc = 3, 5120, 4
    g = torch.Generator(device=DEV)
    g.manual_seed(7)
    bf = lambda *s: torch.randn(*s, generator=g, device=DEV).to(torch.bfloat16)  # noqa: E731
    f32 = lambda *s: torch.randn(*s, generator=g, device=DEV)  # noqa: E731
    ran = {}
    # mHC pre (partial dots + Sinkhorn finish; tl.gather butterflies live in the fused variants)
    h = bf(n, hc, d)
    part = torch.empty((n * K.HC_BLOCKS * 32,), dtype=torch.float32, device=DEV)
    K.hc_pre(h, f32(24, hc * d), torch.ones(3, device=DEV), f32(24), torch.ones(n, hc, device=DEV), bf(d), 1e-6, 1e-6, 20,
             torch.empty((n, d), dtype=torch.bfloat16, device=DEV), torch.empty((n, hc), device=DEV),
             torch.empty((n, hc), device=DEV), torch.empty((n, hc, hc), device=DEV), part)
    ran["hc_pre"] = True
    # fused decode mHC (hc_pre2) if the wrapper exists
    if hasattr(K, "hc_pre2"):
        try:
            K.hc_pre2(h, f32(24, hc * d), torch.ones(3, device=DEV), f32(24), torch.ones(n, hc, device=DEV), bf(d), 1e-6, 1e-6,
                      20, torch.empty((n, d), dtype=torch.bfloat16, device=DEV), torch.empty((n, hc), device=DEV),
                      torch.empty((n, hc), device=DEV), torch.empty((n, hc, hc), device=DEV), part)
            ran["hc_pre2"] = True
        except Exception as e:  # noqa: BLE001
            ran["hc_pre2"] = f"skipped: {e}"
    # indexer top-k (tl.sort)
    keys = torch.randint(0, 2**62, (n, 2048), dtype=torch.int64, device=DEV)
    vis = torch.full((n,), 1500, dtype=torch.int64, device=DEV)
    K.topk_select(keys, 512, vis)
    ran["topk_select"] = True
    # router
    pick = torch.empty((n, 7), dtype=torch.int64, device=DEV)
    wts = torch.empty((n, 7), dtype=torch.float32, device=DEV)
    K.route(f32(n, 384), bf(384).to(torch.float16), 6, 1.5, 384, pick, wts)
    ran["route"] = True
    # row matmuls, collapse, embed
    K.rowmm2(bf(2, d), bf(384, d).to(torch.float16))
    ran["rowmm2"] = True
    K.collapse_norm(h, torch.ones(n, hc, device=DEV), bf(d), 1e-6)
    ran["collapse_norm"] = True
    K.embed_init(bf(1000, d), torch.tensor([1, 2, 3], dtype=torch.int64, device=DEV), hc)
    ran["embed_init"] = True
    K.rope_heads(bf(n, 32, 512), f32(4096, 32), f32(4096, 32), torch.tensor([5, 6, 7], dtype=torch.int64, device=DEV), 64)
    ran["rope_heads"] = True
    if hasattr(K, "fp4_qd_p2"):
        K.fp4_qd_p2(bf(n, 32, 128))
        ran["fp4_qd_p2"] = True
    torch.cuda.synchronize()
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    rows = []
    for meta_path in sorted(glob.glob(str(cache / "*" / "*.json"))):
        if "__grp__" in meta_path:
            continue
        meta = json.loads(Path(meta_path).read_text())
        if "num_warps" not in meta:
            continue
        rows.append({"name": meta["name"], "num_warps": meta["num_warps"], "num_ctas": meta.get("num_ctas", 1),
                     "shared": meta.get("shared"), "global_scratch": meta.get("global_scratch_size", 0),
                     "profile_scratch": meta.get("profile_scratch_size", 0), "pdl": meta.get("launch_pdl", False),
                     "cluster_dims": meta.get("cluster_dims")})
    print(json.dumps({"ran": ran, "markov_import": None if M is not None else str(m_err), "kernels": rows}, indent=1))
    bad = [r for r in rows if r["global_scratch"] or r["profile_scratch"] or r["num_ctas"] != 1]
    print(f"REFUSED_BY_AOT_SET: {len(bad)} of {len(rows)}: {[r['name'] for r in bad]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
