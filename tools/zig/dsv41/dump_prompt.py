#!/usr/bin/env python3
"""A fixture of DeepSeek-V4.1's prompt (prefill) path for the Zig engine: the Python fork's `_forward_k` sublayers
over a seeded prompt in chunks (engine.py prefill_steps: TF_DS_PREFILL_CHUNK rows a chunk), the first layers of a
two-rank split, both ranks computed in one process (as dump_forward.py: the fork's `comm.gather` is a stack of the
ranks' partials, the caches are replicated, so each sublayer runs once per rank on the same input), then the head on
the prompt's last row. Every chunk records the mixes' inputs, both ranks' partials and the streams after each
sublayer, the index layers' picks, and the cache planes after the chunk. No replay (TF_DS_REPLAY=0), no taps (the
fixture's layers hold none); Engram runs on its layers when `--engram <dir>` names the tables (dump_forward.EngramRef:
each rank's partial `L<i>.p_engram.r<r>` and the gated streams `L<i>.h_engram` recorded), as without tables it is
skipped. With `--decode`, each chunk size's fixture also holds one decode row after the prompt (graph.py
`StaticDecoder._body` at R = 1, as dump_forward.py runs it) on the prompt's greedy token (the argmax of the last row's
logits): its partials, streams, head, logits (`D.<name>`) and the cache planes after it.

Written: fixtures/prompt_common (dims, the prompt ids, rope tables), fixtures/prompt_c<chunk> per chunk size,
launches.json / jit.json for tools/zig/dsv41/prompt_capture.sh.

Run inside the fork's image:
  python3 -B /w/tools/zig/dsv41/dump_prompt.py --tools /w/tools/zig --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
      --out /aot/out --layers 4 --prompt 600 --chunks 2048,512 [--engram /models/DeepSeek-V4.1-Flash-Engram] [--decode]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_DS_L2_PREFETCH", "0")
# the fork's fp32 GEMMs (the prompt chunks' router logits and indexer weights) stay fp32, as cuda_engine sets before
# torch reads it: NVIDIA's PyTorch containers default TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 (TF32)
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import torch  # noqa: E402

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
    ap.add_argument("--prompt", type=int, default=600)
    ap.add_argument("--chunks", default="2048,512")
    ap.add_argument("--bucket", type=int, default=1024)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--engram", default=None, help="the Engram shards' dir: run Engram on its layers")
    ap.add_argument("--token-map", default=None, help="the compressed token map JSON (default: built from tokenizer.json)")
    ap.add_argument("--decode", action="store_true", help="one greedy decode row after the prompt (D.<name>)")
    a = ap.parse_args()
    ck = load_module("capture_kernel", HERE / "capture_kernel.py")
    df = load_module("dump_forward", HERE / "dump_forward.py")
    rec = ck.load_recorder(a.tools)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from tensorfold.families.deepseek_v41 import fp32_gemms
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda.graph import StaticDecoder
    from tensorfold.families.deepseek_v41.cuda.model import KV_QUANT, RAW, Comm, Model, mm
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, Weights, linear, load_block

    assert KV_QUANT
    torch.cuda.set_device(0)
    fp32_gemms()
    world = 2
    L = a.layers
    c = Cfg.read(a.model)
    chunks = [int(v) for v in a.chunks.split(",")]
    g = torch.Generator(device="cuda")
    g.manual_seed(a.seed)
    prompt_ids = torch.randint(1000, 100000, (a.prompt,), generator=g, device="cuda")

    sh = Shards(a.model)
    vocab_l = c.vocab // world
    heads = [linear(sh, "head", cols=(r * vocab_l, (r + 1) * vocab_l)) for r in range(world)]
    w = Weights(c, 0, world, embed=sh.get("embed.weight").contiguous().cuda(), norm=sh.get("norm.weight").contiguous().cuda(),
                head=heads[0], vocab_lo=0, vocab_hi=vocab_l, model_dir=Path(a.model))
    w.layers = []
    m = Model(w, Comm(None, 1))
    blocks = df.RankBlocks(sh, c, world, load_block)
    eng = df.engram_ref(a, c, world)
    host = prompt_ids.tolist()                        # the prompt's host ids (SeqCache.host) for Engram's hashes

    def gather(parts: list) -> torch.Tensor:
        torch.cuda.synchronize()
        return torch.stack([p.contiguous() for p in parts]).clone()

    def mixes(n: int) -> tuple:
        dev = "cuda"
        return (torch.empty((n, c.dim), dtype=BF16, device=dev), torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev),
                torch.empty((n, c.hc), dtype=F32, device=dev), torch.empty((n, c.hc), dtype=F32, device=dev),
                torch.empty((n, c.hc), dtype=F32, device=dev), torch.empty((n, c.hc, c.hc), dtype=F32, device=dev))

    def chunk_forward(sc, ids: torch.Tensor, start: int, shared: list, record) -> tuple:
        """model.py _forward_k over one chunk (no replay, no taps; Engram with --engram), each sublayer on both ranks' weights;
        returns (h, pre) after the layers."""
        n = ids.shape[0]
        assert start == sc.length
        sc.tokens[start:start + n] = ids
        pos = torch.arange(start, start + n, device="cuda")
        h = w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(n)
        for li in range(L):
            if eng is not None and blocks.get(0, li).engram_wkv is not None:
                h = eng.apply(blocks, li, h, host, start, n, gather, record)
            lay = blocks.get(0, li)
            assert lay.engram_wkv is None or m.engram is None
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            record(f"L{li}.x_attn", x)
            record(f"L{li}.pre_a", pre_a)
            if c.compress_ratios[li] == 0 and n > K.DECODE_ROWS:
                # the window-only sublayer's pieces (attention_k with ring_mode off), for the Zig port's probes
                for r in range(world):
                    lay_r = blocks.get(r, li)
                    cos, sin = m._cs(li, sc)
                    qa = mm(lay_r.wq_a, x)
                    qr = K.rmsnorm(qa, lay_r.q_norm, c.eps)
                    q = mm(lay_r.wq_b, qr).view(n, m.Hl, c.head_dim)
                    K.rope_heads(q, cos, sin, pos, c.rope_dim)
                    y = mm(lay_r.wkv, x)
                    lo = max(0, start - (c.window - 1))
                    wsrc = torch.empty((start - lo + n, c.head_dim), dtype=BF16, device="cuda")
                    if start > lo:
                        wsrc[:start - lo] = sc.ring[li][torch.arange(lo, start, device="cuda") % sc.ring_size]
                    K.kv_norm_rope(y, lay_r.kv_norm, cos, sin, pos, sc.ring[li], m._neg(n), c.eps, KV_QUANT, c.rope_dim, out=wsrc[start - lo:])
                    wlo = torch.tensor([lo], dtype=torch.int64, device="cuda")
                    o = K.sparse_attn(q, lay_r.sink, wsrc, wlo, False, None, None, pos, c.head_dim ** -0.5, c.window)
                    record(f"L{li}.dbg.qa.r{r}", qa)
                    record(f"L{li}.dbg.q.r{r}", q)
                    record(f"L{li}.dbg.wsrc.r{r}", wsrc)
                    record(f"L{li}.dbg.o_raw.r{r}", o)
                    K.rope_heads(o, cos, sin, pos, c.rope_dim, inverse=True)
                    record(f"L{li}.dbg.o.r{r}", o)
                    from tensorfold.families.deepseek_v41.cuda.model import wo_a_out
                    record(f"L{li}.dbg.u.r{r}", wo_a_out(lay_r, o))
            # each rank's attention from the caches before the chunk (two ranks keep their own caches): the ring's
            # and the compressor store's rows a rank writes at the end are rows the next rank's window reads
            mutable = [sc.ring[li], *sc.comp_raw.get(li, ())]
            before = [t.clone() for t in mutable]
            parts = []
            for r in range(world):
                for t, b in zip(mutable, before):
                    t.copy_(b)
                parts.append(m.attention_k(blocks.get(r, li), x, sc, start, shared[r], pos))
            for r in range(world):
                record(f"L{li}.p_attn.r{r}", parts[r])
            if "topk" in shared[0] and c.compress_ratios[li] > 0:
                record(f"L{li}.cidx", shared[0]["topk"])
            K.hc_post(gather(parts), h, post, comb, h)
            record(f"L{li}.h_attn", h)
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            record(f"L{li}.x_ffn", x)
            record(f"L{li}.pre_f", pre_f)
            parts = [m.moe(blocks.get(r, li), x) for r in range(world)]
            for r in range(world):
                record(f"L{li}.p_moe.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            record(f"L{li}.h_moe", h)
            pre, pre_f = pre_f, pre
        sc.length = start + n
        return h, pre

    def decode_row(sc, token: int, record) -> torch.Tensor:
        """graph.py StaticDecoder._body at R = 1 on `token` at the cache's end (dump_forward.py's body: no L2 forks, no
        taps), each sublayer on both ranks' weights -> the row's logits [1, V]."""
        dec = StaticDecoder(m, sc, 1, a.bucket, False)
        start = sc.length
        dec.ids.fill_(token)
        dec.pos.fill_(start)
        sc.tokens[start] = token
        h = w.embed[dec.ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((1, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(1)
        shared = [{} for _ in range(world)]
        for li in range(L):
            if eng is not None and blocks.get(0, li).engram_wkv is not None:
                h = eng.apply(blocks, li, h, host + [token], start, 1, gather, record)
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            record(f"L{li}.x_attn", x)
            parts = [dec._attention(blocks.get(r, li), x, shared[r]) for r in range(world)]
            for r in range(world):
                record(f"L{li}.p_attn.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            record(f"L{li}.h_attn", h)
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            record(f"L{li}.x_ffn", x)
            parts = [m.moe(blocks.get(r, li), x, shared_side=False) for r in range(world)]
            for r in range(world):
                record(f"L{li}.p_moe.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            record(f"L{li}.h_moe", h)
            pre, pre_f = pre_f, pre
        xc = K.collapse_norm(h, pre.contiguous(), w.norm, c.eps)
        record("head.xc", xc)
        parts = [mm(heads[r], xc, F32) for r in range(world)]
        for r in range(world):
            record(f"head.local.r{r}", parts[r])
        sc.length = start + 1
        return gather(parts).permute(1, 0, 2).reshape(1, -1)

    cos_c, sin_c = None, None
    for chunk in chunks:
        w.layers = [None] * L                             # new_cache counts the layers
        sc = m.new_cache(a.cap)
        w.layers = []
        m.scratch = {}                                    # the prompt scratch is per (slots, count, prompt)
        if cos_c is None:
            cos_c, sin_c = m._cs(min(2, L - 1), sc)
            cos_w, sin_w = m._cs(0, sc)
            common = {"prompt": a.prompt, "layers": L, "bucket": a.bucket, "cap": a.cap, "ring_size": sc.ring_size,
                      "raw": RAW, "world": world, "chunks": a.chunks, "rope_rows": cos_c.shape[0], "vocab_local": vocab_l,
                      "engram": int(eng is not None)}
            ck.save(out / "fixtures" / "prompt_common", common, {"prompt_ids": prompt_ids, "cos_c": cos_c, "sin_c": sin_c, "cos_w": cos_w, "sin_w": sin_w})
        recorded: dict = {}
        shared = [{} for _ in range(world)]
        starts = list(range(0, a.prompt, chunk))
        last = None
        with rec.scope(f"prompt_c{chunk}", detail=True):
            for ci, s in enumerate(starts):
                e = min(a.prompt, s + chunk)

                def record(name: str, t: torch.Tensor, ci=ci, recorded=recorded) -> None:
                    recorded[f"C{ci}.{name}"] = t.detach().clone()

                h, pre = chunk_forward(sc, prompt_ids[s:e], s, shared, record)
                torch.cuda.synchronize()
                for k, v in df.planes(sc, L).items():
                    recorded[f"C{ci}.{k}"] = v.detach().clone()
                last = (h, pre)
            h, pre = last
            xc = K.collapse_norm(h[-1:].contiguous(), pre[-1:].contiguous(), w.norm, c.eps)
            recorded["head.xc"] = xc.clone()
            parts = [mm(heads[r], xc, F32) for r in range(world)]
            for r in range(world):
                recorded[f"head.local.r{r}"] = parts[r].clone()
            logits = gather(parts).permute(1, 0, 2).reshape(1, -1)
        torch.cuda.synchronize()
        arrays = {"prompt_ids": prompt_ids, "expected_logits": logits.clone()}
        params = {"chunk": chunk, "n_chunks": len(starts), "prompt": a.prompt, "layers": L, "bucket": a.bucket, "cap": a.cap,
                  "comp_rows": int(sc.comp[2][0].shape[0]) if 2 in sc.comp else 0}
        if a.decode:
            token = int(torch.argmax(logits[0]).item())   # torch's first index of the maximum

            def record_d(name: str, t: torch.Tensor, recorded=recorded) -> None:
                recorded[f"D.{name}"] = t.detach().clone()

            with rec.scope(f"prompt_c{chunk}_decode", detail=True):
                d_logits = decode_row(sc, token, record_d)
            torch.cuda.synchronize()
            for k, v in df.planes(sc, L).items():
                recorded[f"D.{k}"] = v.detach().clone()
            arrays["expected_D.logits"] = d_logits.clone()
            params["decode_token"] = token
        for k, v in recorded.items():
            arrays[f"expected_{k}"] = v
        ck.save(out / "fixtures" / f"prompt_c{chunk}", params, arrays)
        print(f"chunk {chunk}: {len(starts)} chunk(s) over {a.prompt} tokens, {blocks.loads} block loads so far", flush=True)
        del sc
        torch.cuda.empty_cache()
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
