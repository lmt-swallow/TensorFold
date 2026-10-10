#!/usr/bin/env python3
"""The Zig engine's RoPE tables from the fork's own arithmetic (ops.py rope_cs: torch CPU float32, freqs_cis below
ROPE_COMPAT, rope_block above): `<out>/rope_c_<rows>.bin` (the compressed layers', YaRN at compress_rope_theta) and
`<out>/rope_w_<rows>.bin` (the window layers'), each cos then sin as f32 [rows, rope_dim / 2], and each kind's
per-pair frequencies `<out>/rope_c_freqs.bin`, `<out>/rope_w_freqs.bin` (f32 [rope_dim / 2], the YaRN ramp applied):
row p of a table is (cosf, sinf)(float(p) * freqs) with glibc's cosf / sinf, which the engine's device path
(TF_DS_ROPE=device, zig/kernels/cuda/dsv41/rope.cu) computes from these frequencies instead of loading the tables.
--freqs-only writes the frequencies alone.

Run in the fork's image: docker run --rm -v $HOME/models:/models -v $HOME/aot:/aot --entrypoint python3 \\
    tensorfold-dsv41:90b68e0-p1 /w/tools/zig/dsv41/dump_rope.py --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \\
    --rows 20480 --out /aot/dsv41-all/rope
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.ops import F32, RopeTables, rope_cs


def freqs_of(dim: int, orig_len: int, base: float, factor: float, beta_fast: float, beta_slow: float) -> torch.Tensor:
    """ops.py freqs_cis / rope_block's fp32 frequencies [dim / 2] (the same torch expressions, so the same bits)."""

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=F32) / dim))
    if orig_len > 0:
        def corr(rot: float) -> float:
            return dim * math.log(orig_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=F32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", type=int, default=0, help="rows (a multiple of 4096: cuda_lanes.ropeRows)")
    ap.add_argument("--freqs-only", action="store_true", help="write rope_<kind>_freqs.bin only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = Cfg.read(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = RopeTables(cfg, device="cpu")
    for name, compressed in (("rope_c", True), ("rope_w", False)):
        args = tables.args(compressed)
        freqs = freqs_of(cfg.rope_dim, *args)
        # the frequencies are the tables': rope_cs's first rows are polar(1, outer(positions, freqs))
        check = 4096
        cos, sin = rope_cs(cfg.rope_dim, check, *args, device="cpu")
        f = torch.polar(torch.ones(check, freqs.numel()), torch.outer(torch.arange(check, dtype=F32), freqs))
        assert torch.equal(cos, f.real) and torch.equal(sin, f.imag), name
        path = out / f"{name}_freqs.bin"
        path.write_bytes(freqs.numpy().astype(np.float32).tobytes())
        print(f"{path}: {freqs.numel()} frequencies")
        if a.freqs_only:
            continue
        assert a.rows > 0, "--rows"
        cos, sin = tables.cs(compressed, a.rows)
        assert cos.shape == (RopeTables.rows(a.rows), cfg.rope_dim // 2), cos.shape
        cos, sin = cos[: a.rows].contiguous(), sin[: a.rows].contiguous()
        path = out / f"{name}_{a.rows}.bin"
        with open(path, "wb") as f:
            f.write(cos.numpy().astype(np.float32).tobytes())
            f.write(sin.numpy().astype(np.float32).tobytes())
        print(f"{path}: {a.rows} rows, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
