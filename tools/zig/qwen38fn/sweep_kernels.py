#!/usr/bin/env python3
"""Launch every Triton specialization the Zig Flash Next engine can need, under the recorder, for its serving kernel
set: TensorFold 0.6.6's prompt path at row counts that hit each of its tile and divisibility choices, decode windows
of 1 to 32 rows (a shared round's four windows), and the indexer's block selection at every power-of-two block count up to the window (`_select`'s
registers, then `_select_tiles` for a decode window and for a prompt block), and the MTP head on each prompt's chunks
and on 1 to 28 rows (four streams' kept rows). Nothing is saved but the launches
(capture_serve_kernels.sh packs them with the fixtures' captures).

Run inside the reference image with TRITON_CACHE_DIR set:
  python3 -B /w/tools/zig/qwen38fn/sweep_kernels.py --tools /w/tools/zig --out /aot/out --model /model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsv41"))
import capture_kernel as ck  # noqa: E402

PROMPTS = (1, 5, 16, 17, 32, 33, 64, 65, 100, 128, 129, 300, 2048, 2049)
CAPACITY = 262208           # cuda_native.capacityFor(262144): the indexer's block count a multiple of 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)

    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import exl3 as X
    from tensorfold.families.qwen4_exp.cuda import forward as F
    from tensorfold.families.qwen4_exp.cuda.decode import PREFILL_ROWS
    from tensorfold.families.qwen4_exp.cuda.mtp import mtp_forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = X.load(a.model, mtp=True, draft_vocab="default")
    cap = 16384                                       # a prompt's cache; selection runs on its own scratch below
    st = State(w, cap, 32)
    pb = Buffers(w, PREFILL_ROWS, cap, prefill=True)
    db = Buffers(w, 32, cap, moe_prefill=True)
    for n in PROMPTS:
        st.reset(w)
        ids = [(7 * i + 11) % 151000 + 1000 for i in range(n)]
        with rec.scope(f"prompt_{n}", detail=True):
            start = 0
            while start < n:
                end = min(start + PREFILL_ROWS, n)
                F.forward(w, st, pb, ids[start:end], logits=end == n)
                nxt = ids[start + 1:end + 1]
                if nxt:
                    mtp_forward(w, st, pb, nxt, pb.streams[:len(nxt)])
                    st.set_mtp_len(st.mtp_len + len(nxt))
                F.commit(w, st, pb, end - start, end - start)
                start = end
        torch.cuda.synchronize()
        print(f"prompt {n}", flush=True)
    for r in range(1, 33):
        with rec.scope(f"decode_{r}", detail=True):
            F.forward(w, st, db, [1000 + i for i in range(r)])
            F.commit(w, st, db, r, max(1, r // 2))
        torch.cuda.synchronize()
    print("decode windows 1..32", flush=True)
    mb = Buffers(w, 32, cap)
    for r in range(1, 29):
        with rec.scope(f"mtp_{r}", detail=True):
            mtp_forward(w, st, mb, [1000 + i for i in range(r)], db.streams[:r])
            st.set_mtp_len(st.mtp_len + r)
        torch.cuda.synchronize()
    print("MTP windows 1..28", flush=True)
    # the selection at every block count up to the window, a decode window's rows and a prompt block's
    for rows in (1, 16, 256):
        s = attn_mod.AttnScratch(rows, w.cfg.heads, w.cfg.head_dim, CAPACITY, torch.device("cuda"),
                                 budget=w.cfg.index_budget, ratio=w.cfg.index_ratio)
        pos0 = torch.zeros((1,), dtype=torch.int32, device="cuda")
        blocks = 1
        while blocks <= s.nb:
            pos0.fill_(blocks * s.ratio - rows)
            with rec.scope(f"select_{rows}_{blocks}", detail=True):
                attn_mod._launch_select(s, pos0, rows, blocks)
            blocks *= 2
        with rec.scope(f"select_{rows}_{s.nb}", detail=True):
            attn_mod._launch_select(s, pos0, rows, s.nb)
        torch.cuda.synchronize()
    print("selection sweep", flush=True)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
