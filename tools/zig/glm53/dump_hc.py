#!/usr/bin/env python3
"""GLM-5.3-Flash's hyper-connection glue and the other row kernels around the layers, from the Mia fork in its strict
setting, on seeded inputs, as a fixture for the Zig engine (capture.sh runs this inside the reference image, then
packs the Triton launches). The glue is replicated (every rank runs it on the gathered partials), so one rank's
weights serve: layer 0's two hyper-connections, the final norm and the embedding.

For each row count R (decode windows 1, 3, 16; prompt chunks 2,048 and 37, whose mixing dots run on the tensor
cores: other bits than decode rows'), arrays s{i}_<name>:
  ids, x                        token ids and 4 streams of seeded bf16 values around their embedding
  {site}_normed, _xs, _post,    glue.hc_pre of the attention (site a) and FFN (site f) hyper-connection: the normed
  _comb                         collapsed row, its 64-group sums, the post weights and the Sinkhorn-balanced mix
  part                          two ranks' fp32 partials [2, R, 4096] (seeded)
  post_x                        glue.hc_post of x with them (site a's post and comb)
  pair_x                        glue.hc_post_pair of the same (the row-split prompt path's kernel: the same bits)
  mean, fnormed, fxs            glue.stream_mean of post_x, the final RMSNorm and its group sums
  mtp_x                         glue.residual_add of fnormed with the partials (the MTP layer's plain residual)
and the embedding rows (`glue.embed` of the ids) as s{i}_embed.

  python3 -B /w/tools/zig/glm53/dump_hc.py --tools /w/tools/zig --out /aot/out --model /model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref  # noqa: E402  (sets the strict environment before TensorFold is imported)

import torch  # noqa: E402

ROWS = (("d", 1), ("d", 3), ("d", 16), ("p", 2048), ("p", 37))


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
    from tensorfold.families.glm5_next.cuda import glue

    w = ref.load_rank(a.model, 0, [0], mtp=False, long_context=False)
    c = w.cfg
    layer = w.layers[0]
    D, S = c.hidden, c.streams
    g = ref.seeded(20261014)
    ids = ref.ids_for(a.model, max(R for _, R in ROWS), 20261014)
    arrays: dict[str, torch.Tensor] = {}
    dev = ref.DEV
    for i, (kind, R) in enumerate(ROWS):
        s = f"s{i}"
        prompt = kind == "p"
        idt = torch.tensor(ids[:R], dtype=torch.int32, device=dev)
        emb = torch.empty((R, S * D), dtype=torch.bfloat16, device=dev)
        x = torch.empty((R, S * D), dtype=torch.bfloat16, device=dev)
        normed = torch.empty((R, D), dtype=torch.bfloat16, device=dev)
        xs = torch.empty((R, D // 64), dtype=torch.float32, device=dev)
        post = torch.empty((R, S), dtype=torch.float32, device=dev)
        comb = torch.empty((R, S * S), dtype=torch.float32, device=dev)
        hcpart = torch.empty((R, glue.HC_BLOCKS, 32), dtype=torch.float32, device=dev)
        part = (torch.randn((2, R, D), generator=g, device=dev) * 0.05).contiguous()
        with rec.scope(f"{s}_{kind}{R}", detail=True):
            glue.embed(idt, w.embed, D, S, emb)
            noise = torch.randn((R, S * D), generator=g, device=dev) * 0.02
            x.copy_((emb.float() + noise).to(torch.bfloat16))
            arrays[f"{s}_ids"] = idt.clone()
            arrays[f"{s}_embed"] = emb.clone()
            arrays[f"{s}_x"] = x.clone()
            for site, h, norm in (("a", layer.attn_hc, layer.in_norm), ("f", layer.ffn_hc, layer.post_norm)):
                glue.hc_pre(x, h.fn, h.base, h.scale, norm, normed, xs, post, comb, hcpart, c.eps, c.hc_eps,
                            c.hc_iters, prompt=prompt)
                torch.cuda.synchronize()
                for name, t in (("normed", normed), ("xs", xs), ("post", post), ("comb", comb)):
                    arrays[f"{s}_{site}_{name}"] = t.clone()
            glue.hc_pre(x, layer.attn_hc.fn, layer.attn_hc.base, layer.attn_hc.scale, layer.in_norm, normed, xs,
                        post, comb, hcpart, c.eps, c.hc_eps, c.hc_iters, prompt=prompt)
            y = x.clone()
            glue.hc_post(y, y, part, post, comb)
            z = x.clone()
            glue.hc_post_pair(z, z, part[0], part[1], post, comb)
            mean = torch.empty((R, D), dtype=torch.bfloat16, device=dev)
            glue.stream_mean(y, mean)
            fnormed = torch.empty((R, D), dtype=torch.bfloat16, device=dev)
            fxs = torch.empty((R, D // 64), dtype=torch.float32, device=dev)
            glue.rmsnorm(mean, w.norm, c.eps, fnormed, fxs)
            m = fnormed.clone()
            glue.residual_add(m, m, part)
        torch.cuda.synchronize()
        if not ref.same(y, z):
            raise SystemExit(f"rows {R}: hc_post_pair differs from hc_post")
        for name, t in (("part", part), ("post_x", y), ("pair_x", z), ("mean", mean), ("fnormed", fnormed),
                        ("fxs", fxs), ("mtp_x", m)):
            arrays[f"{s}_{name}"] = t.clone()
        print(f"rows {R} ({kind}): post {post[0].tolist()} fnormed[0,:4]={fnormed[0, :4].float().tolist()}",
              flush=True)
    params = {"layer": 0, "eps": c.eps, "hc_eps": c.hc_eps, "hc_iters": c.hc_iters, "hc_blocks": glue.HC_BLOCKS,
              "steps": [{"kind": k, "rows": R} for k, R in ROWS]}
    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    ref.provenance(out, {"dump": "hc", "rows": ROWS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
