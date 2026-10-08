"""Engram hashing fixture from the Python fork (DeepSeek-V4.1-Flash): primes, offsets, multipliers and the row ids of a
fixed token history, for zig/src/families/deepseek_v41/engram_hash.zig's test.

Run inside a container with the fork installed and the pack mounted:
  python3 engram_fixture.py <model_dir> <out.json>
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np

from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.cuda import model as _model
from tensorfold.families.deepseek_v41.ops import compressed_token_map, engram_multipliers, engram_primes


def main(model_dir: str, out: str) -> None:
    cfg = Cfg.read(model_dir)
    token_map, n_classes = compressed_token_map(f"{model_dir}/tokenizer.json")
    assert n_classes == cfg.engram_cvocab, (n_classes, cfg.engram_cvocab)
    primes = engram_primes(cfg)
    flat = [[p for per in layer for p in per] for layer in primes]
    stub = types.SimpleNamespace(
        cfg=cfg,
        np_primes=np.array(flat, dtype=np.int64),
        np_offsets=np.array([np.cumsum([0, *f[:-1]]) for f in flat], dtype=np.int64),
        np_mult=engram_multipliers(cfg).numpy().astype(np.int64),
        np_map=np.asarray(token_map, dtype=np.int64),
        pad=int(token_map[cfg.engram_pad]),
    )
    rng = np.random.default_rng(4242)
    vocab = len(token_map)
    tokens = [int(t) for t in rng.integers(0, vocab, size=61)]
    tokens[20:23] = [-1, -1, -1]            # an image span: n-grams never reach into or past it
    tokens.append(cfg.engram_pad)
    tokens.append(0)
    tokens.append(vocab - 1)
    cases = [(0, 8), (0, len(tokens)), (3, 5), (17, 10), (19, 9), (60, 4)]
    hashes = {}
    for start, n in cases:
        h = _model.Engram.hashes(stub, tokens, start, n)
        assert h.shape == (n, len(cfg.engram_layers), stub.np_primes.shape[1]), h.shape
        hashes[f"{start}:{n}"] = h.astype(np.int64).tolist()
    used = sorted({t for t in tokens if t >= 0} | {cfg.engram_pad})
    doc = {
        "engram_layers": list(cfg.engram_layers),
        "engram_rows": list(cfg.engram_rows),
        "ngram": cfg.engram_ngram,
        "heads": cfg.engram_heads,
        "vocab": cfg.engram_vocab,
        "cvocab": cfg.engram_cvocab,
        "pad_token": cfg.engram_pad,
        "pad_class": stub.pad,
        "primes": stub.np_primes.tolist(),
        "offsets": stub.np_offsets.tolist(),
        "mult": stub.np_mult.tolist(),
        "token_classes": [[t, int(token_map[t])] for t in used],
        "tokens": tokens,
        "cases": [{"start": s, "n": n, "rows": hashes[f"{s}:{n}"]} for s, n in cases],
    }
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print("wrote", out, "cases", len(cases), "primes sum", [int(s) for s in stub.np_primes.sum(1)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
