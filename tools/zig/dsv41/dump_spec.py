#!/usr/bin/env python3
"""A fixture of DeepSeek-V4.1's speculative decode for the Zig engine's served lane (cuda_lanes.zig with DSpark): the
Python fork's single-stream loop (engine.py `prefill_steps` with a drafter, then `_spec`) over the first layers of a
two-rank split, both ranks computed in one process as dump_forward.py / dump_prompt.py do (each sublayer runs once
with each rank's weights on the same input; the fork's `comm.gather` is the stack of the ranks' partials in rank
order). The DSpark stages run the same way (dspark.py `DraftGraph._body`: each stage's attention and MoE once per
rank), the Markov loop split over the vocabulary (markov.py, TF_DS_MARKOV_SPLIT=1: each rank's (value, index) best a
step, gathered, then `_pick`).

The fixture's layers stand in for the taps' layers: `dspark_taps` becomes the fixture's last three layers (the fork
reads the taps at a tap layer's input, after Engram; no Engram here). Every DSpark stage is loaded (the checkpoint's
drafter, unchanged).

Steps, in the fork's order:
- the prompt (`--prompt` ids, a seeded cycle of `--cycle` ids repeated, so the drafts can land) in chunks of
  `--chunk` rows through `_forward_k` with taps; after each chunk the drafter absorbs the chunk's taps
  (`Drafter.absorb(dc, sc, cat(taps), start)`); the first token is the last row's argmax;
- `--rounds` rounds of `_spec` (TF_DS_ADAPTIVE off): draft 5 after the pending token at P (`DraftGraph._body`), verify
  [token] + drafts at P in one window (graph.py `StaticDecoder._body` with taps, every row's logits), the target's
  picks (greedy), accept the longest prefix of drafts equal to the picks, `sc.length = P + a + 1`, absorb the
  window's taps[:a + 1] at P; then one more draft after the last round. The fixture's few layers make a weak target
  whose picks the drafter rarely matches, so `--oracle k:j` gives round k a window whose first j drafts are the
  target's serial greedy continuation (decoded one row at a time on a copy of the caches, then the caches restored;
  the lane's copy windows look like this): the round keeps j + 1 rows when windows reproduce serial steps.

Recorded (`expected_<name>`): the prompt's partials `C<ci>.L<i>.p_attn.r<r>` / `p_moe`, `head.local.r<r>`, the
logits, each chunk's taps `C<ci>.taps`, the rings after the prompt `P.ring<j>`; each draft k's gathers
`S<k>.s<j>.p_attn.r<r>` / `p_moe`, the Markov steps' bests `S<k>.mk<i>.send.r<r>`, its drafts and confidences;
each round k's gathers `R<k>.L<i>.p_attn.r<r>` / `p_moe`, `R<k>.head.local.r<r>`, the window's logits and taps,
the target's picks, and after the keep and the absorb every layer's cache planes `R<k>.L<i>.<plane>` and every
stage's ring `R<k>.ring<j>`. Params: the window, start, accepted count of each round, the token and q0 of each draft.

Run inside the fork's image (spec_capture.sh does, then packs the launches):
  python3 -B /w/tools/zig/dsv41/dump_spec.py --tools /w/tools/zig --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
      --out /aot/out --layers 4 --prompt 600 --chunk 512 --rounds 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_DS_L2_PREFETCH", "0")
os.environ.setdefault("TF_DS_MARKOV_CACHE", "0")     # no cached bias rows: the same bits, fewer kernels to capture
os.environ.setdefault("TF_DS_MARKOV_SPLIT", "1")
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
BF16, F32 = torch.bfloat16, torch.float32


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Blocks:
    """Both ranks' weights of one block resident at a time: `get(rank, i)` loads block i (a backbone layer, or for
    i >= n_layers the DSpark stage mtp.{i - n_layers}) for every rank, freeing the last block's."""

    def __init__(self, sh, cfg, world: int, load_block):
        self.sh, self.cfg, self.world, self.load_block = sh, cfg, world, load_block
        self.idx, self.lays = None, None
        self.loads = 0

    def get(self, rank: int, i: int):
        if self.idx != i:
            self.lays = None
            torch.cuda.empty_cache()
            c = self.cfg
            prefix = f"layers.{i}" if i < c.n_layers else f"mtp.{i - c.n_layers}"
            n_exp = c.n_routed if i < c.n_layers else c.dspark_routed
            lays = []
            for r in range(self.world):
                lay = self.load_block(self.sh, c, prefix, i, r, self.world, n_exp)
                if lay.idx_proj is not None:
                    lay.idx_proj_h = lay.idx_proj.to(torch.float16).contiguous()
                lays.append(lay)
            self.idx, self.lays = i, lays
            self.loads += self.world
        return self.lays[rank]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=600)
    ap.add_argument("--cycle", type=int, default=24, help="the prompt repeats a seeded cycle of this many ids")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--bucket", type=int, default=1024)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--oracle", default="1:3,2:5", help="round:j pairs: round's first j drafts the serial continuation")
    a = ap.parse_args()
    oracle = {int(k): int(v) for k, v in (p.split(":") for p in a.oracle.split(",") if p)}
    ck = load_module("capture_kernel", HERE / "capture_kernel.py")
    df = load_module("dump_forward", HERE / "dump_forward.py")
    rec = ck.load_recorder(a.tools)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from tensorfold.cuda.sampling import sample_rows
    from tensorfold.families.deepseek_v41 import fp32_gemms
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import dspark as D
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda import markov as MK
    from tensorfold.families.deepseek_v41.cuda.graph import StaticDecoder
    from tensorfold.families.deepseek_v41.cuda.model import KV_QUANT, RAW, Comm, Model, mm
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, Weights, linear, load_block

    assert KV_QUANT
    assert MK.ON and MK.SPLIT and MK.CACHE_ROWS == 0, (MK.ON, MK.SPLIT, MK.CACHE_ROWS)
    torch.cuda.set_device(0)
    fp32_gemms()
    world = 2
    L = a.layers
    c = Cfg.read(a.model)
    assert L >= 4, "the taps stand on layers 1 .. L - 1: at least 4 layers"
    c.dspark_taps = list(range(L - 3, L))
    stages = len(c.compress_ratios) - c.n_layers
    g = torch.Generator(device="cuda")
    g.manual_seed(a.seed)
    cycle = torch.randint(1000, 100000, (a.cycle,), generator=g, device="cuda")
    prompt_ids = cycle.repeat((a.prompt + a.cycle - 1) // a.cycle)[:a.prompt].contiguous()

    sh = Shards(a.model)
    vocab_l = c.vocab // world
    heads = [linear(sh, "head", cols=(r * vocab_l, (r + 1) * vocab_l)) for r in range(world)]
    w = Weights(c, 0, world, embed=sh.get("embed.weight").contiguous().cuda(), norm=sh.get("norm.weight").contiguous().cuda(),
                head=heads[0], vocab_lo=0, vocab_hi=vocab_l, model_dir=Path(a.model))
    w.layers = []
    m = Model(w, Comm(None, 1))
    blocks = Blocks(sh, c, world, load_block)

    def plain(name: str) -> torch.Tensor:
        return sh.get(name).contiguous().cuda()

    last = f"mtp.{stages - 1}"
    dw = types.SimpleNamespace(blocks=None, main_proj=linear(sh, "mtp.0.main_proj"), main_norm=plain("mtp.0.main_norm.weight"),
                               norm=plain(f"{last}.norm.weight"), markov_embed=plain(f"{last}.markov_head.embed.weight"),
                               markov_head=plain(f"{last}.markov_head.head.weight"),
                               conf=plain(f"{last}.confidence_head.proj.weight"))
    # the drafter's methods on this process's model (its weights by rank come from `blocks`)
    drafter = D.Drafter.__new__(D.Drafter)
    drafter.m, drafter.dw = m, dw
    drafter.size, drafter.noise, drafter.topk = c.dspark_block, c.dspark_noise, c.dspark_topk
    drafter.ring_size = c.window + 16
    drafter._block_idx, drafter._markov = {}, {}
    # each rank's Markov loop: its vocabulary half of the head's rows, its own scratch
    mks = [MK.Markov(types.SimpleNamespace(w=types.SimpleNamespace(world=world, rank=r, dspark=dw), cfg=c, comm=None), rows=0)
           for r in range(world)]
    assert all(mk.split for mk in mks)

    def gather(parts: list) -> torch.Tensor:
        torch.cuda.synchronize()
        return torch.stack([p.contiguous() for p in parts]).clone()

    def mixes(n: int) -> tuple:
        dev = "cuda"
        return (torch.empty((n, c.dim), dtype=BF16, device=dev), torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev),
                torch.empty((n, c.hc), dtype=F32, device=dev), torch.empty((n, c.hc), dtype=F32, device=dev),
                torch.empty((n, c.hc), dtype=F32, device=dev), torch.empty((n, c.hc, c.hc), dtype=F32, device=dev))

    recorded: dict = {}
    params: dict = {}

    def recorder(prefix: str):
        def record(name: str, t: torch.Tensor) -> None:
            recorded[f"{prefix}.{name}"] = t.detach().clone()
        return record

    def chunk_forward(sc, ids: torch.Tensor, start: int, shared: list, record) -> tuple:
        """model.py _forward_k over one prompt chunk with taps, each sublayer on both ranks' weights; returns (h, pre,
        taps [n, 3 d])."""
        n = ids.shape[0]
        assert start == sc.length
        sc.tokens[start:start + n] = ids
        pos = torch.arange(start, start + n, device="cuda")
        h = w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(n)
        taps = []
        for li in range(L):
            if li in c.dspark_taps:
                taps.append(h.to(F32).mean(1).to(BF16))
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            # each rank's attention from the caches before the chunk (dump_prompt.py: the rows a rank writes at the
            # end are rows the next rank's window reads)
            mutable = [sc.ring[li], *sc.comp_raw.get(li, ())]
            before = [t.clone() for t in mutable]
            parts = []
            for r in range(world):
                for t, b in zip(mutable, before):
                    t.copy_(b)
                parts.append(m.attention_k(blocks.get(r, li), x, sc, start, shared[r], pos))
            for r in range(world):
                record(f"L{li}.p_attn.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            parts = [m.moe(blocks.get(r, li), x) for r in range(world)]
            for r in range(world):
                record(f"L{li}.p_moe.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            pre, pre_f = pre_f, pre
        sc.length = start + n
        return h, pre, torch.cat(taps, -1)

    def verify(sc, window: list, start: int, record) -> tuple:
        """graph.py StaticDecoder._body at R = len(window) rows with taps (GraphRunner.forward(sc, window, P, True)),
        each sublayer on both ranks' weights -> (every row's logits [R, V], taps [R, 3 d])."""
        n = len(window)
        dec = StaticDecoder(m, sc, n, a.bucket, True)
        dec.ids.copy_(torch.tensor(window, dtype=torch.long))
        dec.pos.copy_(torch.arange(start, start + n, dtype=torch.long))
        sc.tokens[dec.pos] = dec.ids
        h = w.embed[dec.ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(n)
        shared = [{} for _ in range(world)]
        taps = []
        for li in range(L):
            if li in c.dspark_taps:
                taps.append(h.to(F32).mean(1).to(BF16))
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            parts = [dec._attention(blocks.get(r, li), x, shared[r]) for r in range(world)]
            for r in range(world):
                record(f"L{li}.p_attn.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            parts = [m.moe(blocks.get(r, li), x, shared_side=False) for r in range(world)]
            for r in range(world):
                record(f"L{li}.p_moe.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            pre, pre_f = pre_f, pre
        xc = K.collapse_norm(h, pre.contiguous(), w.norm, c.eps)
        parts = [mm(heads[r], xc, F32) for r in range(world)]
        for r in range(world):
            record(f"head.local.r{r}", parts[r])
        sc.length = start + n
        return gather(parts).permute(1, 0, 2).reshape(n, -1), torch.cat(taps, -1)

    def absorb(sc, dc, taps: torch.Tensor, start: int) -> None:
        """dspark.py Drafter.absorb (stage_kv as one mm a stage: the same bits as the grouped launch); the stages' wkv
        and kv_norm are whole on every rank, so rank 0's weights write the rings every rank holds."""
        n = taps.shape[0]
        keep = min(n, drafter.ring_size)
        taps = taps[-keep:]
        first = start + n - keep
        pos = torch.arange(first, first + keep, device="cuda")
        main_x = K.rmsnorm(mm(dw.main_proj, taps.contiguous()), dw.main_norm, c.eps)
        for j, ring in enumerate(dc.rings):
            lay = blocks.get(0, c.n_layers + j)
            y = mm(lay.wkv, main_x)
            cos, sin = m._cs(lay.idx, sc)
            K.kv_norm_rope(y, lay.kv_norm, cos, sin, pos, ring, pos % drafter.ring_size, c.eps, KV_QUANT, c.rope_dim)
        dc.absorbed = start + n

    def draft(sc, dc, token: int, q0: int, record) -> tuple:
        """dspark.py DraftGraph._body for `token` at q0, each stage's attention and MoE on both ranks' weights, the
        Markov loop split over the ranks -> (drafts [block] list, confidences [block])."""
        assert dc.absorbed == q0, (dc.absorbed, q0)
        n = drafter.size
        dev = "cuda"
        ids = torch.full((n,), drafter.noise, dtype=torch.long, device=dev)
        ids[0] = token
        pos = q0 + torch.arange(n, device=dev)
        wpos = torch.full((n,), q0 - 1, dtype=torch.int64, device=dev)
        h = w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device=dev)
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(n)
        for j, ring in enumerate(dc.rings):
            i = c.n_layers + j
            lay = blocks.get(0, i)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            parts = [drafter._attention(blocks.get(r, i), x, sc, ring, None, pos=pos, wpos=wpos) for r in range(world)]
            for r in range(world):
                record(f"s{j}.p_attn.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            lay = blocks.get(0, i)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            parts = [m.moe(blocks.get(r, i), x, topk=drafter.topk) for r in range(world)]
            for r in range(world):
                record(f"s{j}.p_moe.r{r}", parts[r])
            K.hc_post(gather(parts), h, post, comb, h)
            pre, pre_f = pre_f, pre
        xc = K.collapse(h, pre)
        xn = K.rmsnorm(xc, dw.norm, c.eps)
        locals_ = [mm(heads[r], xn, F32) for r in range(world)]
        outs = []
        for r in range(world):
            o = torch.empty((1, n + 1), dtype=torch.long, device=dev)
            o[0, 0] = token
            outs.append(o)
        for i in range(n):
            sends = [mks[r].local_best(locals_[r], outs[r], n, i).clone() for r in range(world)]
            for r in range(world):
                record(f"mk{i}.send.r{r}", sends[r])
            gs = gather(sends)
            for r in range(world):
                mks[r].pick(gs, outs[r], i)
        torch.cuda.synchronize()
        assert torch.equal(outs[0], outs[1]), (outs[0].tolist(), outs[1].tolist())
        embs = dw.markov_embed[outs[0][0, :n]]
        conf = (torch.cat([xc.float(), embs.float()], -1) @ dw.conf.float().t())[:, 0]
        record("drafts", outs[0][0, 1:])
        record("conf", conf)
        return [int(t) for t in outs[0][0, 1:].tolist()], conf

    def cache_tensors(sc) -> list:
        out = []
        for v in vars(sc).values():
            items = v.values() if isinstance(v, dict) else v if isinstance(v, (list, tuple)) else [v]
            for t in items:
                for u in (t if isinstance(t, (list, tuple)) else [t]):
                    if isinstance(u, torch.Tensor):
                        out.append(u)
        return out

    def serial(sc, token: int, start: int, count: int) -> list:
        """The target's next `count` greedy tokens after `token` at `start`, one row a step, on the caches as they
        are; the caches (and the length) are restored after."""
        saved = [t.clone() for t in cache_tensors(sc)]
        length = sc.length
        got, cur = [], token
        for i in range(count):
            lg, _ = verify(sc, [cur], start + i, lambda name, t: None)
            cur = sample_rows(lg, [start + i + 1], None)[0]
            got.append(cur)
        for t, s_ in zip(cache_tensors(sc), saved):
            t.copy_(s_)
        sc.length = length
        torch.cuda.synchronize()
        return got

    w.layers = [None] * L                             # new_cache counts the layers
    sc = m.new_cache(a.cap)
    w.layers = []
    m.scratch = {}
    dc = D.DraftCache([torch.zeros((drafter.ring_size, c.head_dim), dtype=BF16, device="cuda") for _ in range(stages)],
                      drafter.ring_size)
    cos_c, sin_c = m._cs(min(2, L - 1), sc)
    cos_w, sin_w = m._cs(0, sc)
    arrays = {"prompt_ids": prompt_ids, "cos_c": cos_c, "sin_c": sin_c, "cos_w": cos_w, "sin_w": sin_w}
    params.update({"prompt": a.prompt, "layers": L, "chunk": a.chunk, "bucket": a.bucket, "cap": a.cap, "world": world,
                   "rope_rows": cos_c.shape[0], "ring_size": sc.ring_size, "raw": RAW, "vocab_local": vocab_l,
                   "rounds": a.rounds, "stages": stages, "block": drafter.size,
                   "taps": ",".join(str(t) for t in c.dspark_taps)})

    with rec.scope("spec", detail=True):
        # the prompt: engine.py prefill_steps with a drafter
        shared = [{} for _ in range(world)]
        starts = list(range(0, a.prompt, a.chunk))
        for ci, s in enumerate(starts):
            e = min(a.prompt, s + a.chunk)
            h, pre, taps = chunk_forward(sc, prompt_ids[s:e], s, shared, recorder(f"C{ci}"))
            recorded[f"C{ci}.taps"] = taps.clone()
            absorb(sc, dc, taps, s)
        xc = K.collapse_norm(h[-1:].contiguous(), pre[-1:].contiguous(), w.norm, c.eps)
        parts = [mm(heads[r], xc, F32) for r in range(world)]
        for r in range(world):
            recorded[f"head.local.r{r}"] = parts[r].clone()
        logits = gather(parts).permute(1, 0, 2).reshape(1, -1)
        recorded["logits"] = logits.clone()
        for j, ring in enumerate(dc.rings):
            recorded[f"P.ring{j}"] = ring.clone()
        params["n_chunks"] = len(starts)
        tok = sample_rows(logits, [a.prompt], None)[0]
        params["first"] = tok
        # engine.py _spec, greedy, ADAPTIVE off
        P = a.prompt
        for k in range(a.rounds):
            drafts, _conf = draft(sc, dc, tok, P, recorder(f"S{k}"))
            params[f"S{k}_token"], params[f"S{k}_q0"] = tok, P
            kk = len(drafts)
            if k in oracle:
                j = oracle[k]
                drafts = serial(sc, tok, P, j) + drafts[j:]
            window = [tok] + drafts[:kk]
            lg, taps = verify(sc, window, P, recorder(f"R{k}"))
            target = sample_rows(lg, [P + 1 + i for i in range(kk + 1)], None)
            acc = 0
            while acc < kk and drafts[acc] == target[acc]:
                acc += 1
            sc.length = P + acc + 1
            absorb(sc, dc, taps[:acc + 1], P)
            torch.cuda.synchronize()
            recorded[f"R{k}.logits"] = lg.clone()
            recorded[f"R{k}.taps"] = taps.clone()
            recorded[f"R{k}.target"] = torch.tensor(target, dtype=torch.long)
            recorded[f"R{k}.window"] = torch.tensor(window, dtype=torch.long)
            for name, v in df.planes(sc, L).items():
                recorded[f"R{k}.{name}"] = v.detach().clone()
            for j, ring in enumerate(dc.rings):
                recorded[f"R{k}.ring{j}"] = ring.clone()
            params[f"R{k}_start"], params[f"R{k}_rows"], params[f"R{k}_accepted"] = P, len(window), acc
            params[f"R{k}_oracle"] = oracle.get(k, 0)
            print(f"round {k}: at {P} window {window} target {target} accepted {acc}", flush=True)
            tok = target[acc]
            P += acc + 1
        drafts, _conf = draft(sc, dc, tok, P, recorder(f"S{a.rounds}"))
        params[f"S{a.rounds}_token"], params[f"S{a.rounds}_q0"] = tok, P
        print(f"draft {a.rounds}: after {tok} at {P}: {drafts}", flush=True)
    torch.cuda.synchronize()
    for k, v in recorded.items():
        arrays[f"expected_{k}"] = v
    ck.save(out / "fixtures" / "spec", params, arrays)
    print(f"{blocks.loads} block loads", flush=True)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
