#!/usr/bin/env python3
"""Engram's device side from the Python fork, as a fixture for the Zig engine: hash rows of a token sequence, the raw
code and scale bytes those rows hold in DeepSeek's shards, the fork's dequantized bf16 rows `e` a rank feeds
`engram_wkv`, each rank's fp32 partial, the ranks' sum as bf16 and `engram_gate`'s output, with the gate's Triton
launch recorded for the AOT pack (capture_engram.sh runs this, then packs).

Run inside the fork's image with TRITON_CACHE_DIR set:
  python3 -B /w/tools/zig/dsv41/dump_engram.py --tools /w/tools/zig --out /out --model /model --engram /engram --layer 1
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernel as ck  # noqa: E402

DEV = torch.device("cuda:0")
WORLD = 2


def tables(engram_dir: Path) -> dict:
    """name -> (path, base offset, row bytes) for every Engram tensor in the shards (model.py Engram.__init__)."""

    out = {}
    for path in sorted(engram_dir.glob("*.safetensors")):
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        for name, e in header.items():
            if name.endswith("engram.embed.weight") or name.endswith("engram.embed.scale"):
                lo, hi = e["data_offsets"]
                out[name] = (path, 8 + size + lo, int(np.prod(e["shape"][1:])))
    return out


def read_rows(path: Path, base: int, rb: int, idx: np.ndarray) -> torch.Tensor:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        buf = np.empty((idx.shape[0], rb), dtype=np.uint8)
        for i, r in enumerate(idx.tolist()):
            got = os.pread(fd, rb, base + int(r) * rb)
            assert len(got) == rb, (path, r)
            buf[i] = np.frombuffer(got, dtype=np.uint8)
    finally:
        os.close(fd)
    return torch.from_numpy(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engram", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)
    import triton
    from tensorfold.cuda.exl3.linear import Exl3Linear
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda import model as _model
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, exl3_parts
    from tensorfold.families.deepseek_v41.ops import compressed_token_map, engram_multipliers, engram_primes

    cfg = Cfg.read(a.model)
    token_map, n_classes = compressed_token_map(f"{a.model}/tokenizer.json")
    assert n_classes == cfg.engram_cvocab, (n_classes, cfg.engram_cvocab)
    primes = engram_primes(cfg)
    flat = [[p for per in layer for p in per] for layer in primes]
    stub = types.SimpleNamespace(
        cfg=cfg, np_primes=np.array(flat, dtype=np.int64),
        np_offsets=np.array([np.cumsum([0, *f[:-1]]) for f in flat], dtype=np.int64),
        np_mult=engram_multipliers(cfg).numpy().astype(np.int64), np_map=np.asarray(token_map, dtype=np.int64),
        pad=int(token_map[cfg.engram_pad]))
    n_cols = (cfg.engram_ngram - 1) * cfg.engram_heads
    li = list(cfg.engram_layers).index(a.layer)
    n = a.n
    rng = np.random.default_rng(20261009)
    tokens = [int(t) for t in rng.integers(0, len(token_map), size=n)]
    tokens[0] = cfg.engram_pad
    tokens[1] = 0
    tokens[2] = len(token_map) - 1
    hashes = _model.Engram.hashes(stub, tokens, 0, n)                        # [n, L, cols] int64
    assert hashes.shape == (n, len(cfg.engram_layers), n_cols), hashes.shape

    tabs = tables(Path(a.engram))
    pw, bw, rw = tabs[f"layers.{a.layer}.engram.embed.weight"]
    ps, bs, rs = tabs[f"layers.{a.layer}.engram.embed.scale"]
    assert (rw, rs) == (256, 8), (rw, rs)
    sh = Shards(a.model)
    p = f"layers.{a.layer}"
    d, hc = cfg.dim, cfg.hc
    qk = (sh.get(f"{p}.engram.q_weight").to(torch.float32) * sh.get(f"{p}.engram.k_weight").to(torch.float32)).contiguous().to(DEV)
    per = n_cols // WORLD * cfg.engram_dim

    arrays = {
        "token_map": torch.tensor(token_map, dtype=torch.int32), "tokens": torch.tensor(tokens, dtype=torch.int32),
        "hashes": torch.from_numpy(hashes.astype(np.int64)), "qk": qk,
    }
    params = {"layer": a.layer, "layer_index": li, "n": n, "world": WORLD, "cols": n_cols, "D": d, "HC": hc,
              "eps": cfg.eps, "row_w": rw, "row_s": rs, "shard": pw.name, "ngram": cfg.engram_ngram,
              "heads": cfg.engram_heads, "vocab": cfg.engram_vocab, "cvocab": cfg.engram_cvocab,
              "pad_token": cfg.engram_pad, "n_engram_layers": len(cfg.engram_layers)}
    for i, (l, r) in enumerate(zip(cfg.engram_layers, cfg.engram_rows)):
        params[f"engram_layer{i}"] = int(l)
        params[f"engram_rows{i}"] = int(r)
    parts = []
    for rank in range(WORLD):
        lo, hi = rank * n_cols // WORLD, (rank + 1) * n_cols // WORLD
        idx = np.ascontiguousarray(hashes[:, li, lo:hi].reshape(-1), dtype=np.int64)
        m = idx.shape[0]
        codes = read_rows(pw, bw, rw, idx)
        scales = read_rows(ps, bs, rs, idx)
        e = _model.Engram._decode(codes.to(DEV), scales.to(DEV), m, rw, n)   # [n, (hi - lo) * 256] bf16
        tr, suh, svh, cb = exl3_parts(sh, f"{p}.engram.wkv", rows=(rank * per, (rank + 1) * per), device=None)
        lin = Exl3Linear.from_tensors(tr.to(DEV), suh.to(DEV), svh.to(DEV), cb)
        part = lin(e, out_dtype=torch.float32)                                # [n, 5 d] this rank's partial
        parts.append(part)
        arrays.update({f"r{rank}_codes": codes, f"r{rank}_scales": scales, f"r{rank}_e": e, f"r{rank}_part": part,
                       f"r{rank}_wkv_trellis": tr, f"r{rank}_wkv_suh": suh, f"r{rank}_wkv_svh": svh})
        params.update({f"r{rank}_wkv_kt": tr.shape[0], f"r{rank}_wkv_nt": tr.shape[1], f"r{rank}_wkv_w16": tr.shape[2],
                       f"r{rank}_wkv_cb": cb, f"r{rank}_col_lo": lo, f"r{rank}_col_hi": hi})
        print(f"rank {rank}: cols [{lo}, {hi}), {m} rows, e {tuple(e.shape)}, part {tuple(part.shape)}", flush=True)
    acc = parts[0].clone()
    for r in range(1, WORLD):
        acc += parts[r]
    kv = acc.to(torch.bfloat16).contiguous()                                 # Comm.sum(...).to(BF16)
    g = torch.Generator(device=DEV)
    g.manual_seed(20261009)
    h = (torch.randn((n, hc, d), generator=g, device=DEV, dtype=torch.float32) * 0.7).to(torch.bfloat16)
    with rec.scope(f"engram_gate_n{n}", detail=True):
        h_e = K.engram_gate(h, kv, qk, cfg.eps)
    torch.cuda.synchronize()
    arrays.update({"gathered": torch.stack(parts, 0), "kv": kv, "h": h, "h_e": h_e})
    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                                                   "pdl": K.SMALL_SWITCHES["pdl"], "n": n, "layer": a.layer}, indent=1) + "\n")
    print(f"wrote {out / 'fixture'}: n={n} layer={a.layer} h_e[0,0,:4]={h_e[0, 0, :4].float().tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
