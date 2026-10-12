#!/usr/bin/env python3
"""Capture the strict DFlash2 proposer without loading target backbone layers.

Synthetic committed taps test proposer arithmetic, not target tap extraction or
end-to-end generation. Both TP ranks run on one GPU through ref.Pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["TF_GLM_MTP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref  # noqa: E402
import torch  # noqa: E402

STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 64)
CAPACITY = 4096


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ref.tools_on_path(a.tools)
    import capture_kernel as ck
    rec = ck.load_recorder(a.tools)
    from tensorfold.families.glm5_next.cuda import dflash2 as D
    pair = ref.Pair(rec)
    # [] deliberately excludes all 45 target layers and MTP; embedding/head are real.
    ws = [ref.load_rank(a.model, r, [], mtp=False) for r in (0, 1)]
    ds = []
    for r, w in enumerate(ws):
        if w.layers or w.mtp is not None or w.draft_head is None:
            raise RuntimeError("fixture requires no target layers/MTP and a Q4 draft head")
        w.comm = pair.comm(r)
        ds.append(D.Drafter(a.drafter, w, capacity=CAPACITY, ring=True))
    config = Path(a.drafter, "config.json").read_bytes()
    cfg = json.loads(config)
    if cfg["dflash_config"]["block_size"] != 8 or tuple(ds[0].tap_layers) != (5, 14, 24, 33, 42):
        raise RuntimeError("unexpected drafter geometry")
    # Exactly representable BF16 values, identical across ranks and repeated runs.
    count = sum(STEPS) * len(ds[0].tap_layers) * ds[0].D
    taps = ((torch.arange(count, dtype=torch.int32) % 127 - 63).float() / 64).to(torch.bfloat16)
    taps = taps.reshape(sum(STEPS), -1).to(ref.DEV)
    pending = 1234
    tr = ref.Tracer()
    # These outputs cover conv, normalized/rotated QKV and each complete layer.
    # _mm's Q4 object is metadata rather than a giant tensor argument.
    tr.wrap(D, "_dconv")
    tr.wrap(D, "_mm")
    for name in ("_norm", "_prep", "_layer"):
        tr.wrap(D.Drafter, name)
    arrays = tr.arrays
    arrays["committed_taps"] = taps.cpu()

    def rank_run(r: int) -> None:
        d = ds[r]
        start = 0
        for i, n in enumerate(STEPS):
            step = f"s{i}"
            ref.Tracer.begin(r, step)
            with rec.scope(f"r{r}_{step}_taps{n}_block8", detail=True):
                d.add_taps(taps[start:start + n])
                tokens, values, proj = d.candidates(pending, 7)
            torch.cuda.synchronize()
            ref.Tracer.begin(r, None)
            prefix = f"r{r}_{step}_"
            arrays[prefix + "tokens"] = torch.from_numpy(tokens.copy())
            arrays[prefix + "values"] = torch.from_numpy(values.copy())
            arrays[prefix + "projection"] = torch.from_numpy(proj.copy())
            arrays[prefix + "chain"] = torch.tensor(d.chain(tokens, values, proj, pending,
                                                             d.context_end + 1, None, 0.3), dtype=torch.int32)
            # Full physical ring bytes include speculative block writes.
            for layer in range(len(d.layers)):
                arrays[prefix + f"k{layer}"] = d.kc[layer].cpu().clone()
                arrays[prefix + f"v{layer}"] = d.vc[layer].cpu().clone()
            if d.context_end != start + n or int(d.pos_dev.item()) != start + n:
                raise RuntimeError("speculative block advanced committed context")
            start += n
            print(f"rank={r} step={i} committed={start} candidates={tokens.shape}", flush=True)

    pair.run(rank_run)
    for i in range(len(STEPS)):
        for name in ("tokens", "values", "projection", "chain"):
            if not ref.same(arrays[f"r0_s{i}_{name}"], arrays[f"r1_s{i}_{name}"]):
                raise RuntimeError(f"rank disagreement: step {i} {name}")
    params = {"world": 2, "capacity": CAPACITY, "ring_rows": ds[0].cap,
              "block": 8, "pending": pending, "tap_rows": STEPS, "synthetic_taps": True,
              "target_backbone_loaded": False, "mtp_loaded": False,
              "drafter_config_sha256": hashlib.sha256(config).hexdigest()}
    ck.save(out / "fixture", params, arrays, extra={"trace": tr.calls})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "weights.json").write_text(json.dumps({f"rank{r}": ref.weight_shas(w)
                                                  for r, w in enumerate(ws)}, indent=1, sort_keys=True) + "\n")
    ref.provenance(out, {"dump": "dflash2", **params, "calls": len(tr.calls)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
