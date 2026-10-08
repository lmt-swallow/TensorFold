"""EXL3 linear fixture from the Python fork's own kernels: a few of the DSV4.1 pack's layers, fixed-seed bf16 rows,
the fork's bf16 and fp32 outputs. Run inside the fork's container (tensorfold with torch), e.g.

    python3 dump_linear.py /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw <out_dir>

The Zig test `tf-cuda-test exl3-linear <out_dir>` reads manifest.json (params and arrays as fixture.zig expects)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from tensorfold.cuda.exl3.linear import Exl3Linear

ROWS = (1, 5, 16, 128)
# (name, prefix): sk=8/wk=4 narrow layers, the wide wq_b (sk=1/wk=8), a 3-bit expert and, when the pack has one, a
# 2-bit expert (the step-shuffled decode path)
LAYERS = [
    ("wq_a", "layers.0.attn.wq_a"),
    ("wkv", "layers.0.attn.wkv"),
    ("wq_b", "layers.0.attn.wq_b"),
    ("expert3", "layers.3.ffn.experts.0.w1"),
]


def find_two_bit_expert(model_dir: Path, layer: int = 3) -> str | None:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, file in index.items():
        if name.startswith(f"layers.{layer}.ffn.experts.") and name.endswith(".w1.trellis"):
            by_file.setdefault(file, []).append(name)
    for file, names in by_file.items():
        with safe_open(str(model_dir / file), framework="pt") as f:
            for name in sorted(names):
                if f.get_slice(name).get_shape()[-1] == 32:
                    return name[: -len(".trellis")]
    return None


def main(model_dir: str, out_dir: str) -> None:
    root, out = Path(model_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = list(LAYERS)
    two_bit = find_two_bit_expert(root)
    if two_bit:
        layers.append(("expert2", two_bit))
    params: dict = {"layers": len(layers), "rows": ",".join(str(r) for r in ROWS)}
    arrays: dict = {}

    def put(name: str, t: torch.Tensor) -> None:
        file = name.replace(".", "_") + ".bin"
        (out / file).write_bytes(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                                 if t.dtype in (torch.bfloat16, torch.float16) else
                                 t.detach().cpu().contiguous().numpy().tobytes())
        arrays[name] = {"file": file, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    for li, (name, prefix) in enumerate(layers):
        with safe_open(str(root / index[prefix + ".trellis"]), framework="pt") as f:
            trellis = f.get_tensor(prefix + ".trellis")
            suh = f.get_tensor(prefix + ".suh")
            svh = f.get_tensor(prefix + ".svh")
            assert prefix + ".mul1" in set(f.keys()), f"{prefix}: mul1 codebook expected"
        kt, nt, w16 = trellis.shape
        layer = Exl3Linear.from_tensors(trellis, suh, svh, "mul1")
        params[f"L{li}.name"] = name
        params[f"L{li}.prefix"] = prefix
        params[f"L{li}.kt"], params[f"L{li}.nt"], params[f"L{li}.w16"] = int(kt), int(nt), int(w16)
        params[f"L{li}.sk"], params[f"L{li}.wk"] = int(layer.split[0]), int(layer.split[1])
        put(f"L{li}.trellis", trellis)
        put(f"L{li}.suh", suh.to(torch.float16))
        put(f"L{li}.svh", svh.to(torch.float16))
        for m in ROWS:
            g = torch.Generator(device="cpu").manual_seed(1000 * li + m)
            x = torch.randn((m, layer.k), generator=g, dtype=torch.float32).to(torch.bfloat16).cuda()
            put(f"L{li}.x{m}", x)
            put(f"L{li}.y_bf16{m}", layer(x, out_dtype=torch.bfloat16))
            put(f"L{li}.y_f32{m}", layer(x, out_dtype=torch.float32))
            torch.cuda.synchronize()
        print(f"{name}: K={layer.k} N={layer.n} bits={layer.bits} split={layer.split}", flush=True)
    (out / "manifest.json").write_text(json.dumps({"params": params, "arrays": arrays}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
