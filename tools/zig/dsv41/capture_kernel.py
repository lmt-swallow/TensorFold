#!/usr/bin/env python3
"""Capture one DeepSeek-V4.1 Triton kernel from the Python fork for the Zig engine: run its wrapper once on seeded
inputs under TRITON_CACHE_DIR with the launch recorder installed, then write launches.json, jit.json and a replay
fixture (the oracle's manifest.json format: raw arrays, args, grid, divisible16, kernel.cubin/json/ptx).

Run inside the fork's image (TRITON_CACHE_DIR set before Python starts):
  python3 -B /w/tools/zig/dsv41/capture_kernel.py --tools /w/tools/zig --out /out rmsnorm kv_norm_rope
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import torch

DEV = torch.device("cuda:0")
EPS = 1e-6


def load_recorder(tools: str):
    spec = importlib.util.spec_from_file_location("triton_aot_manifest", Path(tools) / "triton_aot_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Recorder().install()


def save(out: Path, params: dict, arrays: dict[str, torch.Tensor], extra: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"params": params, "arrays": {}}
    for name, t in arrays.items():
        t = t.detach().contiguous().cpu()
        raw = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
        (out / f"{name}.bin").write_bytes(raw)
        manifest["arrays"][name] = {"file": f"{name}.bin", "dtype": str(t.dtype).replace("torch.", ""),
                                    "shape": list(t.shape)}
    manifest.update(extra or {})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))


def compiled(fn) -> dict:
    """The JIT cache's one specialization of ``fn``: which launch args Triton specialized as multiples of 16."""

    found = {}
    for entry in getattr(fn, "device_caches", {}).values():
        for part in entry if isinstance(entry, tuple) else (entry,):
            if not isinstance(part, dict):
                continue
            for ck in part.values():
                src = getattr(ck, "src", None)
                if src is None:
                    continue
                found["signature"] = {k: str(v) for k, v in src.signature.items()}
                found["constexprs"] = {".".join(map(str, k)): (v.value if hasattr(v, "value") else v)
                                       for k, v in src.constants.items()}
                found["hash"] = ck.hash
                found["metadata"] = ck.metadata._asdict()
                kinds = list(src.signature.values())
                position = {i: p for p, i in enumerate(i for i, k in enumerate(kinds) if k != "constexpr")}
                found["divisible16"] = sorted(position[key[0]] for key, specs in src.attrs.items()
                                              if key[0] in position and ["tt.divisibility", 16] in [list(x) for x in specs])
    return found


def triton_files(cache: Path, name: str, out: Path) -> dict:
    metas = [p for p in glob.glob(str(cache / "**" / f"{name}.json"), recursive=True)]
    if len(metas) != 1:
        raise SystemExit(f"expected one compiled {name} in {cache}, found {metas}")
    base = Path(metas[0]).with_suffix("")
    for ext in ("cubin", "json", "ptx"):
        shutil.copy(f"{base}.{ext}", out / f"kernel.{ext}")
    ptx = (out / "kernel.ptx").read_text()
    entry = re.search(r"\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
    params = re.findall(r"\.param\s+\.(\w+)\s+(?:\.ptr\s+\.global\s+\.align\s+\d+\s+)?(\w+)", entry.group(2))
    return {"ptx_entry": entry.group(1), "ptx_params": params, "cache_dir": str(base.parent.name)}


def f32bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def specialization() -> dict:
    from triton.runtime.jit import JITFunction
    import gc

    out = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            name = f"{fn.fn.__module__}.{fn.fn.__qualname__}"
            out[name] = {"params": [p.name for p in fn.params],
                         "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                         "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return out


def capture_rmsnorm(K, rec, out: Path, cache: Path, g: torch.Generator) -> dict:
    rows, d = 5, 5120
    x = torch.randn(rows, d, generator=g, device=DEV).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(d, generator=g, device=DEV)).to(torch.bfloat16)
    with rec.scope("rmsnorm", detail=True):
        y = K.rmsnorm(x, w, EPS)
    torch.cuda.synchronize()
    fx = out / "fixtures" / "rmsnorm"
    fx.mkdir(parents=True, exist_ok=True)
    abi = triton_files(cache, "_rmsnorm", fx)
    jit = compiled(K._rmsnorm)
    args = [{"kind": "ptr", "array": "x"}, {"kind": "i32", "value": x.stride(0)}, {"kind": "ptr", "array": "w"},
            {"kind": "ptr", "array": "out", "output": True}, {"kind": "i32", "value": y.stride(0)},
            {"kind": "f32", "bits": f32bits(EPS)}]
    save(fx, {"D": d, "BLOCK": 8192, "num_warps": 8, "rows": rows, "eps": EPS},
         {"x": x, "w": w, "out": torch.zeros_like(y), "expected_out": y},
         {"args": args, "grid": [rows, 1, 1], "abi": abi, "jit": jit, "divisible16": jit["divisible16"],
          "aot": {"fn": "_rmsnorm", "ptr_types": {"X": "*bf16", "W": "*bf16", "OUT": "*bf16"},
                  "ints": {"xs": x.stride(0), "os_": y.stride(0)}, "floats": {"eps": EPS},
                  "consts": {k: v for k, v in jit["constexprs"].items()}}})
    return {"hash": jit["hash"], "divisible16": jit["divisible16"], "metadata": jit["metadata"]}


def capture_kv_norm_rope(K, rec, out: Path, cache: Path, g: torch.Generator) -> dict:
    rows, d, rd, ring_rows = 5, 512, 64, 144
    y = torch.randn(rows, d, generator=g, device=DEV).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(d, generator=g, device=DEV)).to(torch.bfloat16)
    cap = 4096
    t = torch.arange(cap, device=DEV, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000.0 ** (torch.arange(0, rd, 2, device=DEV, dtype=torch.float32) / rd))[None]
    cos, sin = torch.cos(t * inv).contiguous(), torch.sin(t * inv).contiguous()
    pos = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64, device=DEV)
    ring = torch.zeros(ring_rows, d, dtype=torch.bfloat16, device=DEV)
    slots = (pos % ring_rows).to(torch.int64)
    ring_before = ring.clone()
    with rec.scope("kv_norm_rope", detail=True):
        o = K.kv_norm_rope(y, w, cos, sin, pos, ring, slots, EPS, True, rd)
    torch.cuda.synchronize()
    fx = out / "fixtures" / "kv_norm_rope"
    fx.mkdir(parents=True, exist_ok=True)
    abi = triton_files(cache, "_kv_norm_rope", fx)
    jit = compiled(K._kv_norm_rope)
    args = [{"kind": "ptr", "array": "y"}, {"kind": "ptr", "array": "w"}, {"kind": "ptr", "array": "cos"},
            {"kind": "ptr", "array": "sin"}, {"kind": "ptr", "array": "pos"},
            {"kind": "ptr", "array": "out", "output": True}, {"kind": "ptr", "array": "ring", "output": True},
            {"kind": "ptr", "array": "slots"}, {"kind": "i32", "value": ring_rows}, {"kind": "f32", "bits": f32bits(EPS)}]
    save(fx, {"D": d, "RD": rd, "QUANT": True, "num_warps": 4, "rows": rows, "eps": EPS, "ring_size": ring_rows},
         {"y": y, "w": w, "cos": cos, "sin": sin, "pos": pos, "out": torch.zeros_like(o), "ring": ring_before,
          "slots": slots, "expected_out": o, "expected_ring": ring},
         {"args": args, "grid": [rows, 1, 1], "abi": abi, "jit": jit, "divisible16": jit["divisible16"],
          "aot": {"fn": "_kv_norm_rope",
                  "ptr_types": {"Y": "*bf16", "W": "*bf16", "COS": "*fp32", "SIN": "*fp32", "POS": "*i64",
                                "OUT": "*bf16", "RING": "*bf16", "SLOT_OF": "*i64"},
                  "ints": {"ring_size": ring_rows}, "floats": {"eps": EPS},
                  "consts": {k: v for k, v in jit["constexprs"].items()}}})
    return {"hash": jit["hash"], "divisible16": jit["divisible16"], "metadata": jit["metadata"]}



def capture_index_score_sweep(K, rec, out: Path, cache: Path, g: torch.Generator) -> dict:
    """Every integer specialization of the prompt indexer's kernels (`_index_score_rows`, and `_index_score` at
    decode rows): rows, n and nt each divisible by 16 or not, keyed and plain, so a served prompt of any length finds
    its variant (Triton specializes ints on 16-divisibility and == 1)."""
    ih, idim = 32, 128
    done = 0
    for rows in (1, 5, 16, 17, 32, 299, 683, 1365, 2048):
        for n in (1, 16, 1024, 1039, 12288, 13312, 56044, 63715, 131072):
            q = torch.randn(rows, ih, idim, generator=g, device=DEV).to(torch.bfloat16)
            codes = torch.randint(0, 256, (max(n, 64), idim // 2), generator=g, device=DEV, dtype=torch.uint8)
            scales = torch.randint(100, 140, (max(n, 64), idim // 32), generator=g, device=DEV, dtype=torch.uint8)
            w = torch.randn(rows, ih, generator=g, device=DEV).to(torch.bfloat16)
            vis = torch.full((rows,), n, dtype=torch.int64, device=DEV)
            for keys in (False, True):
                with rec.scope(f"index_score_rows_{rows}_{n}_{int(keys)}"):
                    K.index_score(q, (codes, scales), w, vis, n, keys=keys, tmax=keys)
                done += 1
            torch.cuda.synchronize()
            del q, codes, scales, w, vis
    return {"launches": done}


CAPTURES = {"rmsnorm": capture_rmsnorm, "kv_norm_rope": capture_kv_norm_rope, "index_score_sweep": capture_index_score_sweep}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True, help="folder of triton_aot_manifest.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("kernels", nargs="+", choices=sorted(CAPTURES))
    a = ap.parse_args()
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = load_recorder(a.tools)
    import triton
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    g = torch.Generator(device=DEV)
    g.manual_seed(20261009)
    report = {"torch": torch.__version__, "triton": triton.__version__, "device": torch.cuda.get_device_name(),
              "pdl_switch": K.SMALL_SWITCHES["pdl"], "kernels": {}}
    for name in a.kernels:
        report["kernels"][name] = CAPTURES[name](K, rec, out, cache, g)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
