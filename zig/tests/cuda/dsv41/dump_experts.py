"""EXL3 routed experts fixture from the Python fork's own kernels: layer 3's 384 routed experts and the shared
expert, each rank's halves (gate/up output columns, down input rows), fixed-seed bf16 rows and routing, the fork's
fp32 partial from its fused decode path (`exl3_experts.routed`, rows < 64). Run inside the fork's container:

    python3 dump_experts.py /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw <out_dir>

Per rank the fixture holds one int16 blob a projection (every expert's trellis in expert order, the fork's
pack_trellises order), the stacked fp16 suh/svh, and per row count x, pick, wts and y. ~2.5 GB a rank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.families.deepseek_v41.cuda.weights import Shards, exl3_parts, pack_trellises

LAYER = 3
ROWS = (1, 6, 16)
WORLD = 2
TOPK = 6
LIMIT = 10.0          # the pack's swiglu_limit (model.py: c.swiglu_limit)


def main(model_dir: str, out_dir: str) -> None:
    root, out = Path(model_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    n_routed = int(tc["n_routed_experts"])
    inter = int(tc["moe_intermediate_size"])
    limit = float(tc.get("swiglu_limit", LIMIT))
    sh = Shards(root)
    params: dict = {"ranks": WORLD, "rows": ",".join(str(r) for r in ROWS), "limit": limit, "topk": TOPK,
                    "slots": TOPK + 1, "n_routed": n_routed, "layer": LAYER}
    arrays: dict = {}

    def put(name: str, t: torch.Tensor) -> None:
        file = name.replace(".", "_") + ".bin"
        t = t.detach().cpu().contiguous()
        raw = t.view(torch.uint8) if t.dtype in (torch.bfloat16, torch.float16) else t
        (out / file).write_bytes(raw.numpy().tobytes())
        arrays[name] = {"file": file, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}

    p = f"layers.{LAYER}"
    for rank in range(WORLD):
        il = inter // WORLD
        cols = (rank * il, (rank + 1) * il)
        gate, up, down = [], [], []
        for e in list(range(n_routed)) + ["shared"]:
            ep = f"{p}.ffn.experts.{e}" if e != "shared" else f"{p}.ffn.shared_experts"
            g = exl3_parts(sh, f"{ep}.w1", cols=cols, device=None)
            u = exl3_parts(sh, f"{ep}.w3", cols=cols, device=None)
            dn = exl3_parts(sh, f"{ep}.w2", rows=cols, device=None)
            assert g[3] == u[3] == dn[3] == "mul1", f"{ep}: mul1 expected"
            gate.append(g[:3])
            up.append(u[:3])
            down.append(dn[:3])
        E = len(gate)
        D, I = gate[0][0].shape[0] * 16, gate[0][0].shape[1] * 16
        for name, mats in (("gate", gate), ("up", up), ("down", down)):
            put(f"R{rank}.{name}.trellis", torch.cat([m[0].reshape(-1) for m in mats]))
            params[f"R{rank}.{name}.shapes"] = [list(m[0].shape) for m in mats]
            put(f"R{rank}.{name}.suh", torch.stack([m[1].reshape(-1).to(torch.float16) for m in mats]))
            put(f"R{rank}.{name}.svh", torch.stack([m[2].reshape(-1).to(torch.float16) for m in mats]))
        trs = pack_trellises([m[0] for m in gate + up + down])
        gate_d = [(trs[k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(gate)]
        up_d = [(trs[E + k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(up)]
        down_d = [(trs[2 * E + k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(down)]
        ex = exl3_experts.prepare(gate_d, up_d, down_d, "mul1")
        s = exl3_experts.Scratch(ex, rows=64, slots=TOPK + 1)
        params[f"R{rank}.E"], params[f"R{rank}.D"], params[f"R{rank}.I"] = E, D, I
        params[f"R{rank}.k2_gu"] = list(ex.k2_gu)
        params[f"R{rank}.k2_d"] = list(ex.k2_d)
        params[f"R{rank}.cfg_gu"] = list(s.cfg_gu)
        params[f"R{rank}.cfg_d"] = list(s.cfg_d)
        params[f"R{rank}.aligned16"] = int(ex.aligned16)
        print(f"rank {rank}: E={E} D={D} I={I} k2_gu={ex.k2_gu} k2_d={ex.k2_d} cfg_gu={s.cfg_gu} cfg_d={s.cfg_d} "
              f"aligned16={ex.aligned16} decode={exl3_experts.DECODE}", flush=True)
        shared_id = E - 1
        for R in ROWS:
            g = torch.Generator(device="cpu").manual_seed(7000 + 100 * rank + R)
            x = torch.randn((R, D), generator=g, dtype=torch.float32).to(torch.bfloat16).cuda()
            pick = torch.empty((R, TOPK + 1), dtype=torch.int32)
            wts = torch.empty((R, TOPK + 1), dtype=torch.float32)
            for r in range(R):
                ids = torch.randperm(n_routed, generator=g)[:TOPK]
                w = torch.rand((TOPK,), generator=g) + 0.05
                pick[r, :TOPK] = ids.to(torch.int32)
                wts[r, :TOPK] = (w / w.sum()) * 1.5
                pick[r, TOPK] = shared_id
                wts[r, TOPK] = 1.0
            pick, wts = pick.cuda().contiguous(), wts.cuda().contiguous()
            y = exl3_experts.routed(x, pick, wts, ex, s, None, R, limit=limit, act_mode=exl3_experts.ACT_F32)
            torch.cuda.synchronize()
            y2 = exl3_experts.routed(x, pick, wts, ex, s, None, R, limit=limit, act_mode=exl3_experts.ACT_F32)
            torch.cuda.synchronize()
            assert torch.equal(y, y2), "the fork's routed() is not deterministic across runs"
            put(f"R{rank}.x{R}", x)
            put(f"R{rank}.pick{R}", pick)
            put(f"R{rank}.wts{R}", wts)
            put(f"R{rank}.y{R}", y)
            print(f"rank {rank} rows {R}: y[0,:4] = {y[0, :4].tolist()}", flush=True)
        del ex, s, trs, gate_d, up_d, down_d
        torch.cuda.empty_cache()
    (out / "manifest.json").write_text(json.dumps({"params": params, "arrays": arrays}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
