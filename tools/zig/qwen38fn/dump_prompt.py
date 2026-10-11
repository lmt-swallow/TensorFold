#!/usr/bin/env python3
"""The prompt path of Qwen3.8-Flash-Next from TensorFold 0.6.6 on an EXL3 pack, as fixtures for the Zig engine: prompts
committed in chunks of PREFILL_ROWS rows through `forward.forward(..., logits=final)` and `forward.commit` on the
prompt buffers (decode.py `prefill_chunk` without the MTP head), then a few decode windows on the decode buffers from
the state the prompt left (capture_prompt.sh runs this inside the reference image, then packs the Triton launches).

For each prompt p (--lengths, the token ids a fixed text repeated) and chunk c:
  p{p}_ids                                  the prompt's token ids (int32)
  p{p}_c{c}_L{l}_{h,y,wts,inj}              sha256 of a layer's entering streams and pending MoE write-back, in the
                                            manifest's "sha" table (arrays for the layers in --full)
  p{p}_logits                               the head's logits of the prompt's last row (bf16 [1, vocab])
  p{p}_d{j}_ids, p{p}_d{j}_logits           decode windows after the prompt: ids, logits [R, vocab]

Run inside the reference image with TRITON_CACHE_DIR set:
  python3 -B /w/tools/zig/qwen38fn/dump_prompt.py --tools /w/tools/zig --out /aot/out --model /model --lengths 100,8500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsv41"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernel as ck  # noqa: E402
from dump_layer0 import PROMPT  # noqa: E402

DECODE = ((1, 1), (4, 4), (1, 1))          # (rows, kept) windows after each prompt


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="100,8500")
    ap.add_argument("--full", default="0,1,3", help="layers whose boundary arrays are saved, not only hashed")
    ap.add_argument("--capacity", type=int, default=16384)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)

    import triton
    import tensorfold
    from tokenizers import Tokenizer
    from tensorfold.families.qwen4_exp.cuda import exl3 as X
    from tensorfold.families.qwen4_exp.cuda import forward as F
    from tensorfold.families.qwen4_exp.cuda.decode import PREFILL_ROWS
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = X.load(a.model, mtp=False)
    c = w.cfg
    full = {int(x) for x in a.full.split(",") if x.strip()}
    rows = max(r for r, _ in DECODE)
    st = State(w, a.capacity, rows)
    pb = Buffers(w, PREFILL_ROWS, a.capacity, prefill=True)
    db = Buffers(w, rows, a.capacity, moe_prefill=True)
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    base = tok.encode(PROMPT).ids
    arrays: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    tag = {"prefix": None}

    layer_forward = F.layer_forward

    def boundary(layer, w_, segs, b_, R, pending, **kw):
        p = tag["prefix"]
        if p is not None:
            h = b_.h[:R]
            hashes[f"{p}L{layer.index}_h"] = sha(h)
            if layer.index in full:
                arrays[f"{p}L{layer.index}_h"] = h.detach().clone()
        got = layer_forward(layer, w_, segs, b_, R, pending, **kw)
        if p is not None:
            mode, y, wts, inj = got
            torch.cuda.synchronize()
            for name, t in (("y", y), ("wts", wts), ("inj", inj[:R])):
                hashes[f"{p}L{layer.index}_{name}"] = sha(t)
                if layer.index in full:
                    arrays[f"{p}L{layer.index}_{name}"] = t.detach().clone()
        return got

    F.layer_forward = boundary
    params = {"hidden": c.hidden, "layers": c.layers, "vocab": c.vocab, "capacity": a.capacity, "rows": rows,
              "prefill_rows": PREFILL_ROWS, "full": sorted(full)}
    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    params["prompts"] = len(lengths)
    for pi, n in enumerate(lengths):
        ids = (base * (n // len(base) + 1))[:n]
        arrays[f"p{pi}_ids"] = torch.tensor(ids, dtype=torch.int32)
        params[f"p{pi}_len"] = n
        st.reset(w)
        start, chunk = 0, 0
        last = None
        while start < n:
            end = min(start + PREFILL_ROWS, n)
            final = end == n
            tag["prefix"] = f"p{pi}_c{chunk}_"
            with rec.scope(f"p{pi}_c{chunk}", detail=True):
                logits = F.forward(w, st, pb, ids[start:end], logits=final)
                if final:
                    last = logits.clone()
                F.commit(w, st, pb, end - start, end - start)
            torch.cuda.synchronize()
            start, chunk = end, chunk + 1
        params[f"p{pi}_chunks"] = chunk
        arrays[f"p{pi}_logits"] = last
        nxt = int(last.float().argmax())
        print(f"prompt {pi}: {n} tokens in {chunk} chunks, argmax {nxt}", flush=True)
        # decode windows from the prompt's state: the greedy token, then the prompt text again
        dec = [nxt] + base * 2
        at = 0
        for j, (R, keep) in enumerate(DECODE):
            tag["prefix"] = f"p{pi}_d{j}_"
            toks = dec[at:at + R]
            arrays[f"p{pi}_d{j}_ids"] = torch.tensor(toks, dtype=torch.int32)
            params[f"p{pi}_d{j}_rows"], params[f"p{pi}_d{j}_keep"] = R, keep
            with rec.scope(f"p{pi}_d{j}", detail=True):
                lg = F.forward(w, st, db, toks)
                arrays[f"p{pi}_d{j}_logits"] = lg[:R].clone()
                F.commit(w, st, db, R, keep)
            torch.cuda.synchronize()
            at += keep
        params[f"p{pi}_decodes"] = len(DECODE)
        tag["prefix"] = None

    ck.save(out / "fixture", params, arrays, extra={"sha": hashes})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps({
        "torch": torch.__version__, "triton": triton.__version__, "tensorfold": tensorfold.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "lengths": lengths}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
