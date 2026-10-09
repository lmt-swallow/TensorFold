#!/usr/bin/env python3
"""A fixture of DeepSeek-V4.1's concurrent decode for the Zig engine's served lanes (cuda_lanes.zig `Streams`): the
Python fork's `--parallel` path (multi.py MultiDecoder over model.py's PoolCache, rounds.py RoundDecoder, dspark.py
DraftPool and BatchDraftGraph) over the first layers of a two-rank split, both ranks computed in one process as
dump_rounds.py and dump_spec.py do (each sublayer runs once with each rank's weights on the same input; the fork's
gather is the stack of the ranks' partials in rank order). Unfused launches (TF_DS_HC_FUSED=0, TF_DS_ROT_*=0, the
drafter's `_attention(fused=False)`), the Markov loop split over the vocabulary (TF_DS_MARKOV_SPLIT=1).

The fixture's last three layers stand in for the taps' layers. Steps, in MultiDecoder's order:
- two admissions (`_admit` + `_fill` to the end): each stream takes the lowest free slot of a 4-slot pool and the
  first 2048-aligned extent (multi.py Extents) that holds its prompt, its reply and a round's tail; its prompt
  (`--prompts` lengths, seeded cycles) fills the extent from 0 through the prompt path on its pool view
  (`_forward_k` in chunks of `--chunk` rows with taps), each chunk's taps absorbed into its slot of the DraftPool
  (`Drafter.absorb` on the slot's view); its first token is the last row's argmax;
- the first round's drafts: one batched pass for both streams (BatchDraftGraph `_body`, every stage on both ranks'
  weights, the Markov steps' bests gathered);
- the round: each stream's window [pending] + its first `--depths` drafts (`--oracle s:j`: stream s's first j drafts
  replaced by its target's serial continuation, so a keep of more than one row is exercised), one RoundDecoder
  forward over both windows with taps, each stream's greedy picks and accepted count, its length at P + a + 1, its
  kept rows' taps absorbed into its slot (`absorb_many`, not eager: only the kept rows);
- the next round's drafts after the new pending tokens (a second batched pass).

Recorded (`expected_<name>`): per admission i the chunks' partials `A<i>.C<c>.L<l>.p_attn.r<r>` / `p_moe`, the head's
`A<i>.head.local.r<r>`, the logits `A<i>.logits`, the pool's planes `A<i>.<plane>` and drafter rings `A<i>.ring<j>`
after it; per draft pass k the stages' partials `D<k>.s<j>.p_attn.r<r>` / `p_moe`, the Markov bests
`D<k>.mk<i>.send.r<r>`, the drafts `D<k>.drafts` [N, block] and confidences; the round's partials
`R.L<l>.p_attn.r<r>` / `p_moe`, `R.head.local.r<r>`, its logits, taps, windows, picks; after the keep and the absorb
the pool's planes `K.<plane>` and rings `K.ring<j>`. The serial continuations' rows are not recorded (they run on a
saved copy of the pool, restored after).

Run inside the fork's image (lanes_parallel_capture.sh does, then packs the launches):
  python3 -B /w/tools/zig/dsv41/dump_lanes_parallel.py --tools /w/tools/zig \\
      --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw --out /aot/out --layers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

for k, v in (("TF_DS_HC_FUSED", "0"), ("TF_DS_ROT_Q", "0"), ("TF_DS_ROT_WOB", "0"), ("TF_DS_ROT_ATTN", "0"),
             ("TF_DS_L2_PREFETCH", "0"), ("TF_DS_MARKOV_CACHE", "0"), ("TF_DS_MARKOV_SPLIT", "1"),
             ("TF_DS_DRAFT_FUSED", "0")):
    os.environ.setdefault(k, v)
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
BF16, F32 = torch.bfloat16, torch.float32
SLOTS = 4


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--prompts", default="300,700", help="the streams' prompt lengths, in admission order")
    ap.add_argument("--count", type=int, default=40, help="each stream's reply tokens (its extent's size)")
    ap.add_argument("--cycle", type=int, default=24, help="each prompt repeats a seeded cycle of this many ids")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--context", type=int, default=4096, help="the engine's context: the pool holds context + 24")
    ap.add_argument("--depths", default="5,3", help="each stream's drafts verified in the round")
    ap.add_argument("--oracle", default="1:2", help="stream:j pairs: the stream's first j drafts its serial continuation")
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    lens = [int(x) for x in a.prompts.split(",")]
    depths = [int(x) for x in a.depths.split(",")]
    oracle = {int(k): int(v) for k, v in (p.split(":") for p in a.oracle.split(",") if p)}
    assert len(lens) == len(depths) == 2, "two streams"
    ck = load_module("capture_kernel", HERE / "capture_kernel.py")
    spec = load_module("dump_spec", HERE / "dump_spec.py")
    rec = ck.load_recorder(a.tools)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from tensorfold.families.deepseek_v41 import fp32_gemms
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import dspark as D
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda import markov as MK
    from tensorfold.families.deepseek_v41.cuda.graph import bucket_for
    from tensorfold.families.deepseek_v41.cuda.model import KV_QUANT, RAW, Comm, Model, mm
    from tensorfold.families.deepseek_v41.cuda.multi import Extents
    from tensorfold.families.deepseek_v41.cuda.rounds import MAX_ROWS, RoundDecoder
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, Weights, linear, load_block

    assert KV_QUANT and not K.on("hc") and not K.on("rot_q") and not K.on("rot_wob") and not K.on("rot_attn")
    assert MK.ON and MK.SPLIT and MK.CACHE_ROWS == 0, (MK.ON, MK.SPLIT, MK.CACHE_ROWS)
    torch.cuda.set_device(0)
    fp32_gemms()
    world = 2
    L = a.layers
    c = Cfg.read(a.model)
    assert L >= 4, "the taps stand on layers 1 .. L - 1: at least 4 layers"
    c.dspark_taps = list(range(L - 3, L))
    stages = len(c.compress_ratios) - c.n_layers
    sh = Shards(a.model)
    vocab_l = c.vocab // world
    heads = [linear(sh, "head", cols=(r * vocab_l, (r + 1) * vocab_l)) for r in range(world)]
    w = Weights(c, 0, world, embed=sh.get("embed.weight").contiguous().cuda(), norm=sh.get("norm.weight").contiguous().cuda(),
                head=heads[0], vocab_lo=0, vocab_hi=vocab_l, model_dir=Path(a.model))
    m = Model(w, Comm(None, 1))
    blocks = spec.Blocks(sh, c, world, load_block)

    def plain(name: str) -> torch.Tensor:
        return sh.get(name).contiguous().cuda()

    last = f"mtp.{stages - 1}"
    dw = types.SimpleNamespace(blocks=None, main_proj=linear(sh, "mtp.0.main_proj"), main_norm=plain("mtp.0.main_norm.weight"),
                               norm=plain(f"{last}.norm.weight"), markov_embed=plain(f"{last}.markov_head.embed.weight"),
                               markov_head=plain(f"{last}.markov_head.head.weight"),
                               conf=plain(f"{last}.confidence_head.proj.weight"))
    drafter = D.Drafter.__new__(D.Drafter)
    drafter.m, drafter.dw = m, dw
    drafter.size, drafter.noise, drafter.topk = c.dspark_block, c.dspark_noise, c.dspark_topk
    drafter.ring_size = c.window + 16
    drafter._block_idx, drafter._markov = {}, {}
    mks = [MK.Markov(types.SimpleNamespace(w=types.SimpleNamespace(world=world, rank=r, dspark=dw), cfg=c, comm=None), rows=0)
           for r in range(world)]
    assert all(mk.split for mk in mks)
    n = drafter.size
    ring = drafter.ring_size

    # the pool (engine.py: cap = limit + max_rows + 8, every cache reads the RoPE table of that many positions)
    cap = a.context + MAX_ROWS + 8
    m.rope_cap = cap
    w.layers = [None] * L                             # new_pool / RoundDecoder count the layers
    pool = m.new_pool(SLOTS, cap)
    m.scratch = {}
    extents = Extents(cap)
    table = m.pool_view(pool, 0, 0, extents.total)    # RoPE tables for any position (MultiDecoder._table_view)
    dpool = D.DraftPool.__new__(D.DraftPool)
    dpool.ring_size = ring
    dpool.rings = [torch.zeros((SLOTS * ring, c.head_dim), dtype=BF16, device="cuda") for _ in range(stages)]
    dpool.views = [D.DraftCache([r[s * ring:(s + 1) * ring] for r in dpool.rings], ring) for s in range(SLOTS)]

    def gather(parts: list) -> torch.Tensor:
        torch.cuda.synchronize()
        return torch.stack([p.contiguous() for p in parts]).clone()

    def mixes(rows: int) -> tuple:
        dev = "cuda"
        return (torch.empty((rows, c.dim), dtype=BF16, device=dev), torch.empty((rows * K.HC_BLOCKS * 32,), dtype=F32, device=dev),
                torch.empty((rows, c.hc), dtype=F32, device=dev), torch.empty((rows, c.hc), dtype=F32, device=dev),
                torch.empty((rows, c.hc), dtype=F32, device=dev), torch.empty((rows, c.hc, c.hc), dtype=F32, device=dev))

    recorded: dict = {}
    params: dict = {}

    def recorder(prefix: str):
        def record(name: str, t: torch.Tensor) -> None:
            recorded[f"{prefix}.{name}"] = t.detach().clone()
        return record

    def quiet(name: str, t: torch.Tensor) -> None:
        pass

    def pool_planes() -> dict:
        d = {}
        for li in range(L):
            d[f"L{li}.ring"] = pool.ring[li]
            if li in pool.comp:
                d[f"L{li}.comp_codes"], d[f"L{li}.comp_scales"] = pool.comp[li]
            if li in pool.index_k:
                d[f"L{li}.idxk_codes"], d[f"L{li}.idxk_scales"] = pool.index_k[li]
            if li in pool.comp_raw:
                d[f"L{li}.raw_kv"], d[f"L{li}.raw_score"] = pool.comp_raw[li]
        return d

    def snapshot(prefix: str) -> None:
        torch.cuda.synchronize()
        for k_, v in pool_planes().items():
            recorded[f"{prefix}.{k_}"] = v.clone()
        for j, r in enumerate(dpool.rings):
            recorded[f"{prefix}.ring{j}"] = r.clone()

    def chunk_forward(sc, ids: torch.Tensor, start: int, shared: list, record) -> tuple:
        """model.py _forward_k over one prompt chunk on a pool view with taps, each sublayer on both ranks' weights
        (each rank's attention from the caches before the chunk, as dump_spec.py) -> (h, pre, taps [n, 3 d])."""
        rows = ids.shape[0]
        assert start == sc.length
        sc.tokens[start:start + rows] = ids
        pos = torch.arange(start, start + rows, device="cuda")
        h = w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((rows, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(rows)
        taps = []
        for li in range(L):
            if li in c.dspark_taps:
                taps.append(h.to(F32).mean(1).to(BF16))
            lay = blocks.get(0, li)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
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
        sc.length = start + rows
        return h, pre, torch.cat(taps, -1)

    def absorb_view(dc, taps: torch.Tensor, start: int) -> None:
        """dspark.py Drafter.absorb on a slot's view (dump_spec.py's: stage_kv as one mm a stage)."""
        rows = taps.shape[0]
        keep = min(rows, ring)
        taps = taps[-keep:]
        first = start + rows - keep
        pos = torch.arange(first, first + keep, device="cuda")
        main_x = K.rmsnorm(mm(dw.main_proj, taps.contiguous()), dw.main_norm, c.eps)
        for j, rg in enumerate(dc.rings):
            lay = blocks.get(0, c.n_layers + j)
            y = mm(lay.wkv, main_x)
            cos, sin = m._cs(lay.idx, table)
            K.kv_norm_rope(y, lay.kv_norm, cos, sin, pos, rg, pos % ring, c.eps, KV_QUANT, c.rope_dim)
        dc.absorbed = start + rows

    def absorb_many(items: list) -> None:
        """dspark.py Drafter.absorb_many (not eager): every stream's taps in one pass over the ring planes."""
        taps = torch.cat([t for _, t, _ in items], 0).contiguous()
        pos = torch.tensor([p for _, t, st in items for p in range(st, st + t.shape[0])], dtype=torch.long, device="cuda")
        slot = torch.tensor([sl for sl, t, _ in items for _ in range(t.shape[0])], dtype=torch.long, device="cuda")
        main_x = K.rmsnorm(mm(dw.main_proj, taps), dw.main_norm, c.eps)
        rows = slot * ring + pos % ring
        for j, rg in enumerate(dpool.rings):
            lay = blocks.get(0, c.n_layers + j)
            y = mm(lay.wkv, main_x)
            cos, sin = m._cs(lay.idx, table)
            K.kv_norm_rope(y, lay.kv_norm, cos, sin, pos, rg, rows, c.eps, KV_QUANT, c.rope_dim)
        for sl, t, st in items:
            dpool.views[sl].absorbed = st + t.shape[0]

    def batch_draft(tokens: list, q0: list, slots: list, record) -> tuple:
        """dspark.py BatchDraftGraph._body for N streams (each stage's attention and MoE on both ranks' weights, the
        Markov loop split over the ranks) -> (drafts [N][block], confidences [N, block])."""
        N = len(tokens)
        R = N * n
        dev = "cuda"
        bg = D.BatchDraftGraph.__new__(D.BatchDraftGraph)
        bg.d, bg.sc, bg.dp, bg.N, bg.steps = drafter, table, dpool, N, n
        bg.bidx = (torch.arange(N, device=dev)[:, None, None] * n +
                   torch.arange(n, device=dev)[None, None, :]).expand(N, n, n).reshape(R, n).contiguous()
        bg.zero = torch.zeros((R,), dtype=torch.long, device=dev)
        ids = torch.full((N, n), drafter.noise, dtype=torch.long, device=dev)
        ids[:, 0] = torch.tensor(tokens, device=dev)
        ids = ids.view(R)
        q0_t = torch.tensor(q0, dtype=torch.long, device=dev)
        slots_t = torch.tensor(slots, dtype=torch.long, device=dev)
        pos = (q0_t[:, None] + torch.arange(n, device=dev)[None]).reshape(R)
        wpos = (q0_t - 1)[:, None].expand(N, n).reshape(R).contiguous()
        rbase = (slots_t * ring)[:, None].expand(N, n).reshape(R).contiguous()
        h = w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((R, c.hc), dtype=F32, device=dev)
        pre[:, 0] = 1.0
        x, part, pre_a, pre_f, post, comb = mixes(R)
        for j, rg in enumerate(dpool.rings):
            i = c.n_layers + j
            lay = blocks.get(0, i)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            parts = [bg._attention(blocks.get(r, i), x, rg, pos, wpos, rbase) for r in range(world)]
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
            o = torch.empty((N, n + 1), dtype=torch.long, device=dev)
            o[:, 0] = torch.tensor(tokens, device=dev)
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
        embs = dw.markov_embed[outs[0][:, :n]]
        conf = (torch.cat([xc.float().view(N, n, -1), embs.float()], -1) @ dw.conf.float().t())[..., 0]
        record("drafts", outs[0][:, 1:].contiguous())
        record("conf", conf.contiguous())
        return [[int(t) for t in row] for row in outs[0][:, 1:].tolist()], conf

    def round_forward(windows: list, record) -> tuple:
        """rounds.py RoundDecoder over `windows` [(slot, base, size, P, ids)] with taps, each sublayer on both ranks'
        weights (dump_rounds.py's body) -> (every row's logits [R, V], taps [R, 3 d])."""
        ids_all, pos_all, slot_all, base_all, end_all = [], [], [], [], []
        for slot, base, size, p0, toks in windows:
            ids_all += toks
            pos_all += list(range(p0, p0 + len(toks)))
            slot_all += [slot] * len(toks)
            base_all += [base] * len(toks)
            end_all += [base + size] * len(toks)
        R = len(ids_all)
        bucket = bucket_for(max(p + 1 for p in pos_all), cap)
        dec = RoundDecoder(m, pool, R, bucket, taps=False, out=None)
        dec.set(ids_all, pos_all, slot_all, base_all, end_all)
        torch.cuda.synchronize()
        dec._idx = {}
        h, pre = K.embed_init(w.embed, dec.ids, c.hc)
        x, part, pre_a, pre_f, post, comb = mixes(R)
        shared = [{} for _ in range(world)]
        taps = []
        for li in range(L):
            if li in c.dspark_taps:
                taps.append(h.to(F32).mean(1).to(BF16))
            lay = blocks.get(0, li)
            cos, sin = m._cs(lay.idx, dec.table)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            parts = [dec._attention(blocks.get(r, li), x, shared[r], cos, sin) for r in range(world)]
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
        logits = gather(parts).permute(1, 0, 2).reshape(R, -1).contiguous()
        return logits, torch.cat(taps, -1), bucket

    def serial(stream: dict, token: int, count: int) -> list:
        """The stream's next `count` greedy tokens after `token`, one row a round, on a saved copy of the pool."""
        saved = {k_: v.clone() for k_, v in pool_planes().items()}
        got, cur = [], token
        for i in range(count):
            lg, _, _ = round_forward([(stream["slot"], stream["base"], stream["size"], stream["P"] + i, [cur])], quiet)
            cur = int(lg[0].argmax())
            got.append(cur)
        for k_, v in pool_planes().items():
            v.copy_(saved[k_])
        torch.cuda.synchronize()
        return got

    g = torch.Generator(device="cuda")
    g.manual_seed(a.seed)
    streams = []
    params.update({"layers": L, "world": world, "slots": SLOTS, "cap": cap, "total": extents.total, "chunk": a.chunk,
                   "stages": stages, "block": n, "ring_size": ring, "raw": RAW, "vocab_local": vocab_l,
                   "taps": ",".join(str(t) for t in c.dspark_taps), "n_streams": len(lens)})
    with rec.scope("parallel", detail=True):
        # the admissions (multi.py admit -> _admit -> _fill to the end)
        free = list(range(SLOTS))
        for i, plen in enumerate(lens):
            cycle = torch.randint(1000, 100000, (a.cycle,), generator=g, device="cuda")
            ids = cycle.repeat((plen + a.cycle - 1) // a.cycle)[:plen].contiguous()
            need = plen + a.count + MAX_ROWS + 8 + 4
            base = extents.take(need)
            assert base is not None
            slot = free.pop(0)
            size = extents.size(need)
            sc = m.pool_view(pool, slot, base, size)
            dc = dpool.views[slot]
            shared = [{} for _ in range(world)]
            starts = list(range(0, plen, a.chunk))
            for ci, s in enumerate(starts):
                e = min(plen, s + a.chunk)
                h, pre, taps = chunk_forward(sc, ids[s:e], s, shared, recorder(f"A{i}.C{ci}"))
                absorb_view(dc, taps, s)
            xc = K.collapse_norm(h[-1:].contiguous(), pre[-1:].contiguous(), w.norm, c.eps)
            parts = [mm(heads[r], xc, F32) for r in range(world)]
            for r in range(world):
                recorded[f"A{i}.head.local.r{r}"] = parts[r].clone()
            logits = gather(parts).permute(1, 0, 2).reshape(1, -1)
            recorded[f"A{i}.logits"] = logits.clone()
            first = int(logits[0].argmax())
            snapshot(f"A{i}")
            recorded[f"A{i}.prompt_ids"] = ids.clone()
            params.update({f"s{i}_slot": slot, f"s{i}_base": base, f"s{i}_size": size, f"s{i}_prompt": plen,
                           f"s{i}_count": a.count, f"s{i}_first": first, f"s{i}_chunks": len(starts)})
            streams.append({"slot": slot, "base": base, "size": size, "sc": sc, "P": plen, "pending": first})
            print(f"admission {i}: slot {slot} extent [{base}, {base + size}) prompt {plen} first {first}", flush=True)
        # the first round's drafts: one batched pass (multi.py _round -> _drafts)
        drafts, _ = batch_draft([s["pending"] for s in streams], [s["P"] for s in streams], [s["slot"] for s in streams],
                                recorder("D0"))
        # the round's windows
        windows = []
        for i, (s, d) in enumerate(zip(streams, drafts)):
            k = depths[i]
            dd = d[:k]
            if i in oracle:
                j = oracle[i]
                dd = serial(s, s["pending"], j) + dd[j:]
            s["window"] = [s["pending"]] + dd
            windows.append((s["slot"], s["base"], s["size"], s["P"], s["window"]))
            params[f"s{i}_depth"], params[f"s{i}_oracle"] = k, oracle.get(i, 0)
        logits, taps, bucket = round_forward(windows, recorder("R"))
        params["bucket"] = bucket
        recorded["R.logits"] = logits.clone()
        recorded["R.taps"] = taps.clone()
        recorded["R.window"] = torch.tensor([t for s in streams for t in s["window"]], dtype=torch.long)
        targets = logits.argmax(-1).tolist()
        recorded["R.target"] = torch.tensor(targets, dtype=torch.long)
        r0, items = 0, []
        for i, s in enumerate(streams):
            win = s["window"]
            tg = targets[r0:r0 + len(win)]
            acc = 0
            while acc < len(win) - 1 and win[acc + 1] == tg[acc]:
                acc += 1
            items.append((s["slot"], taps[r0:r0 + acc + 1], s["P"]))
            s["sc"].length = s["P"] + acc + 1
            params[f"s{i}_accepted"] = acc
            print(f"stream {i}: window {win} at {s['P']} target {tg} accepted {acc}", flush=True)
            s["P"] += acc + 1
            s["pending"] = tg[acc]
            r0 += len(win)
        absorb_many(items)
        snapshot("K")
        # the next round's drafts
        drafts2, _ = batch_draft([s["pending"] for s in streams], [s["P"] for s in streams], [s["slot"] for s in streams],
                                 recorder("D1"))
        for i, s in enumerate(streams):
            params[f"s{i}_pending"] = s["pending"]
        print(f"drafts {drafts} then {drafts2}", flush=True)
    torch.cuda.synchronize()
    cos_c, sin_c = m._cs(min(2, L - 1), table)
    cos_w, sin_w = m._cs(0, table)
    arrays = {"cos_c": cos_c, "sin_c": sin_c, "cos_w": cos_w, "sin_w": sin_w}
    params["rope_rows"] = cos_c.shape[0]
    for k_, v in recorded.items():
        arrays[f"expected_{k_}"] = v
    ck.save(out / "fixtures" / "parallel", params, arrays)
    print(f"{blocks.loads} block loads", flush=True)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
