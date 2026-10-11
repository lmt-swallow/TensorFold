#!/usr/bin/env python3
"""TensorFold 0.6.6's MTP drafting on an EXL3 pack, as a fixture for the Zig engine: the serial decoder (decode.py
`prefill` with the MTP head absorbing every prompt chunk, then `mtp_decode`, greedy, depth 6, confidence 0.70) on a
fixed prompt, every MTP head call and draft recorded in order (capture_mtp.sh runs this in the reference image).

  draft_words, draft_scales, draft_biases       the requantized draft head (qmm.py frag layout) and draft_ids
  prompt_ids, tokens                            the prompt and the decoded tokens (pending first)
  m{j}_next, m{j}_in, m{j}_logits, m{j}_streams  MTP call j: its next tokens, input streams, the draft head's logits
                                                 (decode calls) and the head's own final streams
  the manifest's "calls" list: each MTP call's kind (prompt / decode), cache length before, row count; "drafts":
  each sample_draft's token and probability; "rounds": each verify window's tokens, sampled tokens and keep.

Run inside the reference image with TRITON_CACHE_DIR set:
  python3 -B /w/tools/zig/qwen38fn/dump_mtp.py --tools /w/tools/zig --out /aot/out --model /model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsv41"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernel as ck  # noqa: E402
from dump_layer0 import PROMPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--repeat", type=int, default=1, help="the prompt text repeated (a longer prompt: more chunks)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)

    import triton
    import tensorfold
    from tokenizers import Tokenizer
    from tensorfold.families.qwen4_exp.cuda import decode as D
    from tensorfold.families.qwen4_exp.cuda import exl3 as X

    w = X.load(a.model, mtp=True, draft_vocab="default")
    e = D.Engine(w, capacity=16384, max_rows=16)
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    prompt = tok.encode(PROMPT).ids * a.repeat
    arrays: dict[str, torch.Tensor] = {
        "draft_words": w.draft_head.weight, "draft_scales": w.draft_head.scales, "draft_biases": w.draft_head.biases,
        "draft_ids": w.draft_ids.to(torch.int64), "prompt_ids": torch.tensor(prompt, dtype=torch.int32),
    }
    calls: list[dict] = []
    drafts: list[dict] = []
    rounds: list[dict] = []
    mtp_forward = D.mtp_forward

    def traced(w_, st, b, next_tokens, streams, **kw):
        j = len(calls)
        calls.append({"kind": "prompt" if b.prefill else "decode", "mtp_len": st.mtp_len, "rows": len(next_tokens),
                      "drafted": st.mtp_drafted})
        arrays[f"m{j}_next"] = torch.tensor(list(next_tokens), dtype=torch.int32)
        arrays[f"m{j}_in"] = streams.detach().clone()
        with rec.scope(f"mtp_{calls[-1]['kind']}", detail=True):
            got = mtp_forward(w_, st, b, next_tokens, streams, **kw)
        torch.cuda.synchronize()
        if not b.prefill:
            arrays[f"m{j}_logits"] = got.detach().clone()
        arrays[f"m{j}_streams"] = b.streams[:len(next_tokens)].detach().clone()
        return got

    D.mtp_forward = traced
    sample_draft = D.Engine.sample_draft

    def sd(self, logits, position, sampling):
        t, p = sample_draft(self, logits, position, sampling)
        drafts.append({"token": int(t), "prob": float(p), "position": int(position)})
        return t, p

    D.Engine.sample_draft = sd
    forward = D.Engine.forward

    def fw(self, tokens):
        got = forward(self, tokens)
        rounds.append({"tokens": [int(x) for x in tokens], "pos": int(self.st.pos)})
        return got

    D.Engine.forward = fw
    with rec.scope("prefill", detail=True):
        first = D.prefill(e, prompt, None)
    with rec.scope("decode", detail=True):
        res = D.mtp_decode(e, first, a.tokens, None)
    torch.cuda.synchronize()
    for r, k in zip(rounds, res.keeps):
        r["keep"] = int(k)
    arrays["tokens"] = torch.tensor(res.tokens, dtype=torch.int32)
    print(f"{len(prompt)} prompt tokens, first {first}; {len(res.tokens)} tokens in {res.rounds} rounds, "
          f"{res.accepted}/{res.drafted} drafts accepted, {len(calls)} MTP calls", flush=True)
    params = {"prompt": len(prompt), "tokens": len(res.tokens), "calls": len(calls), "depth": 6, "confidence": 0.7,
              "capacity": 16384, "rows": 16, "draft_n": int(w.draft_head.n), "first": int(first)}
    ck.save(out / "fixture", params, arrays, extra={"calls": calls, "drafts": drafts, "rounds": rounds,
                                                    "keeps": [int(k) for k in res.keeps]})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps({
        "torch": torch.__version__, "triton": triton.__version__, "tensorfold": tensorfold.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
