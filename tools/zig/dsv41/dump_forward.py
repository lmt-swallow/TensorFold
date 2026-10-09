#!/usr/bin/env python3
"""A fixture of DeepSeek-V4.1's layer loop for the Zig engine: the Python fork's decode body (graph.py
``StaticDecoder._body``: embed, then per layer mHC pre -> attention -> gather -> mHC post -> mHC pre -> MoE -> gather
-> mHC post, then collapse_norm -> head -> gather) over the first layers of a two-rank split, both ranks computed in
one process: the fork's `comm.gather` is a stack of the ranks' partials in rank order, so each sublayer runs once
with rank 0's weights and once with rank 1's on the same input (the caches are replicated: rank 1's writes repeat
rank 0's bytes), and the gathered pair goes into mHC post as the fork's does.

Memory: one block of one rank is resident at a time (the blocks load on demand from the shards), so the run fits
beside a serving engine. Layers with Engram run as the fork runs them without tables (no gate), or with them when
`--engram <dir>` names the shards (model.py `engram_apply` on the host ids prompt + window: each rank's partial
`L<i>.p_engram.r<r>` and the gated streams `L<i>.h_engram` recorded). L2 prefetch is off
(TF_DS_L2_PREFETCH=0: no data moves, the same bits). The caches are primed by the fork's prompt-path sublayers
(`attention_k`, `moe`) over a seeded prompt in the same two-rank manner.

Written: fixtures/fwd_common (dims, the prompt, rope tables), fixtures/fwd_n<rows> (the window's ids and positions,
every layer's cache planes before the window, both ranks' partials at every gather, h and the mixes after every
sublayer, the logits), launches.json / jit.json for tools/zig/dsv41/forward_capture.sh.

Run inside the fork's image:
  python3 -B /w/tools/zig/dsv41/dump_forward.py --tools /w/tools/zig --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
      --out /aot/out --layers 4 --prompt 40 --rows 1,5 [--engram /models/DeepSeek-V4.1-Flash-Engram]
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


class EngramRef:
    """model.py Engram + engram_apply for both ranks of a two-rank split in one process: the row ids from the host ids
    (Engram.hashes), each rank's columns read by offset from the shards and dequantized (Engram._decode), each rank's
    partial mm(engram_wkv, e, F32), their Comm.sum in rank order to bf16, then engram_gate."""

    def __init__(self, engram_dir: str, cfg, token_map: list, world: int):
        import types

        import numpy as np

        from tensorfold.families.deepseek_v41.cuda.model import Engram
        from tensorfold.families.deepseek_v41.ops import engram_multipliers, engram_primes

        de = load_module("dump_engram", HERE / "dump_engram.py")
        self.np, self.E, self.read_rows = np, Engram, de.read_rows
        self.cfg, self.world = cfg, world
        primes = engram_primes(cfg)
        flat = [[p for per in layer for p in per] for layer in primes]
        self.stub = types.SimpleNamespace(
            cfg=cfg, np_primes=np.array(flat, dtype=np.int64),
            np_offsets=np.array([np.cumsum([0, *f[:-1]]) for f in flat], dtype=np.int64),
            np_mult=engram_multipliers(cfg).numpy().astype(np.int64), np_map=np.asarray(token_map, dtype=np.int64),
            pad=int(token_map[cfg.engram_pad]))
        self.n_cols = (cfg.engram_ngram - 1) * cfg.engram_heads
        self.tabs = de.tables(Path(engram_dir))

    def hashes(self, host: list, start: int, n: int):
        """Engram.hashes: the row ids [n, L, cols] of positions start .. start + n - 1 of the host ids."""
        return self.E.hashes(self.stub, host, start, n)

    def apply(self, blocks, li: int, h, host: list, start: int, n: int, gather, record):
        return self.apply_hashes(blocks, li, h, self.hashes(host, start, n), gather, record)

    def apply_hashes(self, blocks, li: int, h, hashes, gather, record):
        """apply on rows whose row ids are given [n, L, cols] (a round: every stream's rows hashed from its own ids)."""
        from tensorfold.families.deepseek_v41.cuda import kernels as K
        from tensorfold.families.deepseek_v41.cuda.model import mm

        np, c = self.np, self.cfg
        n = hashes.shape[0]
        slot = list(c.engram_layers).index(li)
        pw, bw, rw = self.tabs[f"layers.{li}.engram.embed.weight"]
        ps, bs, rs = self.tabs[f"layers.{li}.engram.embed.scale"]
        parts = []
        for r in range(self.world):
            lo, hi = r * self.n_cols // self.world, (r + 1) * self.n_cols // self.world
            idx = np.ascontiguousarray(hashes[:, slot, lo:hi].reshape(-1), dtype=np.int64)
            codes = self.read_rows(pw, bw, rw, idx).cuda()
            scales = self.read_rows(ps, bs, rs, idx).cuda()
            e = self.E._decode(codes, scales, idx.shape[0], rw, n)
            parts.append(mm(blocks.get(r, li).engram_wkv, e, F32))
            record(f"L{li}.p_engram.r{r}", parts[r])
        g = gather(parts)
        acc = g[0].clone()
        for r in range(1, self.world):
            acc += g[r]
        kv = acc.to(BF16)
        h = K.engram_gate(h, kv.contiguous(), blocks.get(0, li).engram_qk, c.eps)
        record(f"L{li}.h_engram", h)
        return h


def engram_ref(a, cfg, world: int):
    """--engram/--token-map -> an EngramRef (the token map read from --token-map, else built from tokenizer.json), or None."""
    if not a.engram:
        return None
    from tensorfold.families.deepseek_v41.ops import compressed_token_map

    if a.token_map:
        tm = json.loads(Path(a.token_map).read_text())
    else:
        tm, n = compressed_token_map(Path(a.model) / "tokenizer.json")
        assert n == cfg.engram_cvocab, (n, cfg.engram_cvocab)
    return EngramRef(a.engram, cfg, tm, world)


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
    ap.add_argument("--engram", default=None, help="the Engram shards' dir: run Engram on its layers")
    ap.add_argument("--token-map", default=None, help="the compressed token map JSON (default: built from tokenizer.json)")
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
    eng = engram_ref(a, c, world)
    host = prompt_ids.tolist()                        # the sequence's host ids (SeqCache.host) for Engram's hashes

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
            if eng is not None and blocks.get(0, li).engram_wkv is not None:
                h = eng.apply(blocks, li, h, host, start, n, gather, lambda name, t: None)
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
        """graph.py StaticDecoder._body without the L2 forks and the taps (Engram's rows when --engram names the
        tables), each sublayer on both ranks' weights."""
        n = dec.R
        dec.sc.tokens[dec.pos] = dec.ids
        h = w.embed[dec.ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(n)
        shared = [{} for _ in range(world)]
        for li in range(L):
            if eng is not None and blocks.get(0, li).engram_wkv is not None:
                h = eng.apply(blocks, li, h, host + dec.ids.tolist(), start, n, gather, record)
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
              "raw": RAW, "world": world, "rows": a.rows, "rope_rows": cos_c.shape[0], "vocab_local": vocab_l,
              "engram": int(eng is not None)}
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
