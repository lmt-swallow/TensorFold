#!/usr/bin/env python3
"""The Zig engine's RoPE tables from the fork's own arithmetic (ops.py rope_cs: torch CPU float32, freqs_cis below
ROPE_COMPAT, rope_block above): `<out>/rope_c_<rows>.bin` (the compressed layers', YaRN at compress_rope_theta) and
`<out>/rope_w_<rows>.bin` (the window layers'), each cos then sin as f32 [rows, rope_dim / 2].

Run in the fork's image: docker run --rm -v $HOME/models:/models -v $HOME/aot:/aot --entrypoint python3 \\
    tensorfold-dsv41:90b68e0-p1 /w/tools/zig/dsv41/dump_rope.py --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \\
    --rows 20480 --out /aot/dsv41-all/rope
"""
import argparse
from pathlib import Path

import numpy as np

from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.ops import RopeTables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", type=int, required=True, help="rows (a multiple of 4096: cuda_lanes.ropeRows)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = Cfg.read(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = RopeTables(cfg, device="cpu")
    for name, compressed in (("rope_c", True), ("rope_w", False)):
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
