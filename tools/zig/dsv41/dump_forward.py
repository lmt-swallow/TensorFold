#!/usr/bin/env python3
"""A fixture of DeepSeek-V4.1's layer loop for the Zig engine: the Python fork's decode body (graph.py
``StaticDecoder._body``: embed, then per layer mHC pre -> attention -> gather -> mHC post -> mHC pre -> MoE -> gather
-> mHC post, then collapse_norm -> head -> gather) over the first layers of a two-rank split, both ranks computed in
one process: the fork's `comm.gather` is a stack of the ranks' partials in rank order, so each sublayer runs once
with rank 0's weights and once with rank 1's on the same input (the caches are replicated: rank 1's writes repeat
rank 0's bytes), and the gathered pair goes into mHC post as the fork's does.

Memory: one block of one rank is resident at a time (the blocks load on demand from the shards), so the run fits
beside a serving engine. Layers with Engram run as the fork runs them without tables (no gate). L2 prefetch is off
(TF_DS_L2_PREFETCH=0: no data moves, the same bits). The caches are primed by the fork's prompt-path sublayers
(`attention_k`, `moe`) over a seeded prompt in the same two-rank manner.

Written: fixtures/fwd_common (dims, the prompt, rope tables), fixtures/fwd_n<rows> (the window's ids and positions,
every layer's cache planes before the window, both ranks' partials at every gather, h and the mixes after every
sublayer, the logits), launches.json / jit.json for tools/zig/dsv41/forward_capture.sh.

Run inside the fork's image:
  python3 -B /w/tools/zig/dsv41/dump_forward.py --tools /w/tools/zig --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
      --out /aot/out --layers 4 --prompt 40 --rows 1,5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_DS_L2_PREFETCH", "0")

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
BF16, F32 = torch.bfloat16, torch.float32


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RankBlocks:
    """weights.py's blocks of either rank, one resident at a time: `get(rank, i)` loads it (freeing the last)."""

    def __init__(self, sh, cfg, world: int, load_block):
        self.sh, self.cfg, self.world, self.load_block = sh, cfg, world, load_block
        self.key, self.lay = None, None
        self.loads = 0

    def get(self, rank: int, i: int):
        if self.key != (rank, i):
            self.lay = None
            torch.cuda.empty_cache()
            lay = self.load_block(self.sh, self.cfg, f"layers.{i}", i, rank, self.world, self.cfg.n_routed)
            if lay.idx_proj is not None:
                lay.idx_proj_h = lay.idx_proj.to(torch.float16).contiguous()
            self.key, self.lay = (rank, i), lay
            self.loads += 1
        return self.lay


def planes(sc, L: int) -> dict:
    out = {}
    for li in range(L):
        out[f"L{li}.ring"] = sc.ring[li]
        if li in sc.comp:
            out[f"L{li}.comp_codes"], out[f"L{li}.comp_scales"] = sc.comp[li]
        if li in sc.index_k:
            out[f"L{li}.idxk_codes"], out[f"L{li}.idxk_scales"] = sc.index_k[li]
        if li in sc.comp_raw:
            out[f"L{li}.raw_kv"], out[f"L{li}.raw_score"] = sc.comp_raw[li]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=40)
    ap.add_argument("--rows", default="1,5")
    ap.add_argument("--bucket", type=int, default=1024)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    ck = load_module("capture_kernel", HERE / "capture_kernel.py")
    rec = ck.load_recorder(a.tools)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda.graph import StaticDecoder
    from tensorfold.families.deepseek_v41.cuda.model import KV_QUANT, RAW, Comm, Model, mm
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, Weights, linear, load_block

    assert KV_QUANT
    torch.cuda.set_device(0)
    world = 2
    L = a.layers
    c = Cfg.read(a.model)
    rows = [int(v) for v in a.rows.split(",")]
    g = torch.Generator(device="cuda")
    g.manual_seed(a.seed)
    prompt_ids = torch.randint(1000, 100000, (a.prompt,), generator=g, device="cuda")
    window_ids = {n: torch.randint(1000, 100000, (n,), generator=g, device="cuda") for n in rows}

    sh = Shards(a.model)
    vocab_l = c.vocab // world
    heads = [linear(sh, "head", cols=(r * vocab_l, (r + 1) * vocab_l)) for r in range(world)]
    w = Weights(c, 0, world, embed=sh.get("embed.weight").contiguous().cuda(), norm=sh.get("norm.weight").contiguous().cuda(),
                head=heads[0], vocab_lo=0, vocab_hi=vocab_l, model_dir=Path(a.model))
    w.layers = []
    m = Model(w, Comm(None, 1))
    w.layers = [None] * L                             # new_cache counts the layers
    sc = m.new_cache(a.cap)
    w.layers = []
    blocks = RankBlocks(sh, c, world, load_block)

    def gather(parts: list) -> torch.Tensor:
        """The fork's comm.gather: every rank's partial, rank order."""
        torch.cuda.synchronize()
        return torch.stack([p.contiguous() for p in parts]).clone()

    def mixes(n: int) -> tuple:
        dev = "cuda"
        return (torch.empty((n, c.dim), dtype=BF16, device=dev), torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev),
                torch.empty((n, c.hc), dtype=F32, device=dev), torch.empty((n, c.hc), dtype=F32, device=dev),
                torch.empty((n, c.hc), dtype=F32, device=dev), torch.empty((n, c.hc, c.hc), dtype=F32, device=dev))

    def prime(ids: torch.Tensor) -> None:
        """model.py _forward_k's sublayers over the prompt (the eager prompt path), both ranks a sublayer."""
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

    def body(dec, record) -> torch.Tensor:
        """graph.py StaticDecoder._body without the L2 forks, the taps and the Engram rows (none without tables),
        each sublayer on both ranks' weights."""
        n = dec.R
        dec.sc.tokens[dec.pos] = dec.ids
        h = w.embed[dec.ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(n)
        shared = [{} for _ in range(world)]
        for li in range(L):
            lay = blocks.get(0, li)
            assert lay.engram_wkv is None or m.engram is None
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            record(f"L{li}.x_attn", x)
            record(f"L{li}.pre_a", pre_a)
            parts = [dec._attention(blocks.get(r, li), x, shared[r]) for r in range(world)]
            for r in range(world):
                record(f"L{li}.p_attn.r{r}", parts[r])
            gth = gather(parts)
            K.hc_post(gth, h, post, comb, h)
            record(f"L{li}.h_attn", h)
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            record(f"L{li}.x_ffn", x)
            record(f"L{li}.pre_f", pre_f)
            parts = [m.moe(blocks.get(r, li), x, shared_side=False) for r in range(world)]
            for r in range(world):
                record(f"L{li}.p_moe.r{r}", parts[r])
            gth = gather(parts)
            K.hc_post(gth, h, post, comb, h)
            record(f"L{li}.h_moe", h)
            pre, pre_f = pre_f, pre
        xc = K.collapse_norm(h, pre.contiguous(), w.norm, c.eps)
        record("head.xc", xc)
        parts = [mm(heads[r], xc, F32) for r in range(world)]
        for r in range(world):
            record(f"head.local.r{r}", parts[r])
        gth = gather(parts)
        return gth.permute(1, 0, 2).reshape(n, -1)

    with rec.scope("prime", detail=False):
        prime(prompt_ids)
    torch.cuda.synchronize()
    start = sc.length
    before = {k: v.clone() for k, v in planes(sc, L).items()}
    cos_c, sin_c = m._cs(min(2, L - 1), sc)
    cos_w, sin_w = m._cs(0, sc)
    common = {"prompt": a.prompt, "layers": L, "start": start, "bucket": a.bucket, "cap": a.cap, "ring_size": sc.ring_size,
              "raw": RAW, "world": world, "rows": a.rows, "rope_rows": cos_c.shape[0], "vocab_local": vocab_l}
    ck.save(out / "fixtures" / "fwd_common", common, {"prompt_ids": prompt_ids, "cos_c": cos_c, "sin_c": sin_c, "cos_w": cos_w, "sin_w": sin_w})

    for n in rows:
        for k, v in before.items():
            planes(sc, L)[k].copy_(v)
        dec = StaticDecoder(m, sc, n, a.bucket, False)
        dec.ids.copy_(window_ids[n])
        dec.pos.copy_(torch.arange(start, start + n, dtype=torch.long))
        recorded: dict = {}

        def record(name: str, t: torch.Tensor, recorded=recorded) -> None:
            recorded[name] = t.detach().clone()

        with rec.scope(f"fwd_n{n}", detail=True):
            logits = body(dec, record)
        torch.cuda.synchronize()
        arrays = {"ids": dec.ids.clone(), "pos": dec.pos.clone(), "expected_logits": logits.clone()}
        for k, v in before.items():
            arrays[k] = v
        for k, v in planes(sc, L).items():
            arrays[f"expected_{k}"] = v
        for k, v in recorded.items():
            arrays[f"expected_{k}"] = v
        params = {"n": n, "start": start, "bucket": a.bucket, "layers": L,
                  "comp_rows": int(before["L2.comp_codes"].shape[0]) if "L2.comp_codes" in before else 0}
        ck.save(out / "fixtures" / f"fwd_n{n}", params, arrays)
        print(f"case n={n}: logits {tuple(logits.shape)}, {blocks.loads} block loads so far", flush=True)

    # -- the kernel the Zig path launches where the body indexes the embedding: its bytes checked equal ------------
    with rec.scope("embed_init", detail=True):
        for n in rows:
            ids = window_ids[n]
            h, pre = K.embed_init(w.embed, ids, c.hc)
            want = w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
            wpre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
            wpre[:, 0] = 1.0
            assert torch.equal(h, want) and torch.equal(pre, wpre), "embed_init differs from the torch expression"
    torch.cuda.synchronize()
    print("embed_init equals the torch expression", flush=True)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
