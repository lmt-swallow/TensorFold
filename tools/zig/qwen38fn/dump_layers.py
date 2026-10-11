#!/usr/bin/env python3
"""The whole Qwen3.8-Flash-Next decode forward from TensorFold 0.6.6 on an EXL3 pack, as fixtures for the Zig engine:
every layer's boundary values and a call-by-call trace of chosen layers, with the Triton launches recorded for the AOT
pack (capture_layers.sh runs this inside the reference image, then packs).

One stream decodes a fixed prompt in the windows of dump_layer0.py (R rows, keep), through `forward.forward` and
`forward.commit` as the serial decoder runs them (MTP head not loaded). For each window i:

  w{i}_ids, w{i}_ple_ids, w{i}_ple_rows       the token ids, the n-gram row ids [R, heads] and the staged table rows
  w{i}_L{l}_h                                 the streams entering layer l (the previous layer's write-back pending)
  w{i}_L{l}_y, _wts, _inj                     layer l's pending MoE write-back: expert slots fp32, weights, gates
  w{i}_logits                                 the head's logits [R, vocab] (bf16)

and for each layer in --trace (and the head, as layer "H"), every call of the traced functions in order:
  w{i}_T{l}_{seq}_{fn}_{arg}_in / _out / _ret  a tensor argument before the call, after it (when it changed), and the
                                               returned tensor; the call list with the non-tensor arguments is the
                                               manifest's "trace" entry.

Run inside the reference image with TRITON_CACHE_DIR set:
  python3 -B /w/tools/zig/qwen38fn/dump_layers.py --tools /w/tools/zig --out /aot/out --model /model --trace 1,3
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsv41"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernel as ck  # noqa: E402
from dump_layer0 import PROMPT, WINDOWS  # noqa: E402

KEEP_IN_BYTES = 64 << 20          # tensor arguments larger than this are weights or whole caches: not saved as inputs


class Tracer:
    """Records the traced functions' tensor arguments around each call while `active` names a layer."""

    def __init__(self) -> None:
        self.active: str | None = None
        self.window = 0
        self.seq = 0
        self.arrays: dict[str, torch.Tensor] = {}
        self.calls: list[dict] = []

    def wrap(self, module, name: str) -> None:
        fn = getattr(module, name)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        tracer = self

        @functools.wraps(fn)
        def traced(*args, **kwargs):
            if tracer.active is None:
                return fn(*args, **kwargs)
            bound = None
            if sig is not None:
                try:
                    bound = sig.bind(*args, **kwargs)
                except TypeError:
                    bound = None
            items = list(bound.arguments.items()) if bound else [(f"a{i}", a) for i, a in enumerate(args)] + list(kwargs.items())
            seq = tracer.seq
            tracer.seq += 1
            prefix = f"w{tracer.window}_T{tracer.active}_{seq:03d}_{name}_"
            before = {}
            meta = {"layer": tracer.active, "seq": seq, "fn": f"{module.__name__}.{name}", "args": {}}
            torch.cuda.synchronize()
            for k, v in items:
                if isinstance(v, torch.Tensor):
                    nbytes = v.numel() * v.element_size()
                    meta["args"][k] = {"tensor": True, "dtype": str(v.dtype), "shape": list(v.shape),
                                       "stride": list(v.stride()), "bytes": nbytes}
                    if nbytes <= KEEP_IN_BYTES:
                        before[k] = v.detach().clone()
                else:
                    meta["args"][k] = _plain(v)
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            for k, v in items:
                if isinstance(v, torch.Tensor) and k in before:
                    if torch.equal(_bits(v), _bits(before[k])):
                        tracer.arrays[prefix + k + "_in"] = before[k]
                    else:
                        tracer.arrays[prefix + k + "_in"] = before[k]
                        tracer.arrays[prefix + k + "_out"] = v.detach().clone()
                        meta["args"][k]["changed"] = True
            if isinstance(out, torch.Tensor):
                tracer.arrays[prefix + "ret"] = out.detach().clone()
                meta["ret"] = {"dtype": str(out.dtype), "shape": list(out.shape)}
            elif isinstance(out, tuple):
                meta["ret"] = [_plain(x) if not isinstance(x, torch.Tensor) else "tensor" for x in out]
                for j, x in enumerate(out):
                    if isinstance(x, torch.Tensor):
                        tracer.arrays[prefix + f"ret{j}"] = x.detach().clone()
            meta["window"] = tracer.window
            tracer.calls.append(meta)
            return out

        setattr(module, name, traced)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


def _plain(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    return repr(type(v).__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", default="1,3", help="layers whose calls are traced (comma list; H: the head)")
    ap.add_argument("--layers", type=int, default=0, help="load only the first N layers (0: all)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)

    import triton
    import tensorfold
    from tokenizers import Tokenizer
    from tensorfold.cuda import moe as moe_mod
    from tensorfold.cuda.exl3 import experts as x3experts
    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import exl3 as X
    from tensorfold.families.qwen4_exp.cuda import exl3_mm, exl3_pack
    from tensorfold.families.qwen4_exp.cuda import forward as F
    from tensorfold.families.qwen4_exp.cuda import gdn as G
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    if a.layers:
        from tensorfold.families.qwen4_exp.cuda import weight_types as WT

        read = WT.Config.read.__func__

        def first(cls, model_dir):
            c = read(cls, model_dir)
            c.layers = a.layers
            c.layer_types = c.layer_types[:a.layers]
            c.ple_layers = [i for i in c.ple_layers if i < a.layers]
            return c

        WT.Config.read = classmethod(first)

    w = X.load(a.model, mtp=False)
    c = w.cfg
    traced = {s.strip() for s in a.trace.split(",") if s.strip()}
    tr = Tracer()
    for mod, names in ((glue, ("hc_writeback", "hc_normed", "hc_reduce_act", "hc_act", "hc_mix", "attn_prep",
                               "attn_gate", "ple_gate", "ple_conv")),
                       (attn_mod, ("qsa_select", "qsa_pool", "qsa_rows", "attention")),
                       (F, ("_mm", "_embed", "shift_windows")), (exl3_mm, ("ple_rows",)),
                       (G, ("chain", "replay")), (moe_mod, ("router", "select_rows")), (x3experts, ("routed",))):
        for n in names:
            tr.wrap(mod, n)

    rows = max(r for r, _ in WINDOWS)
    st = State(w, 4096, rows)
    b = Buffers(w, rows, 4096, moe_prefill=True)
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    ids = tok.encode(PROMPT).ids
    arrays = tr.arrays
    cur_window = {"i": 0}

    layer_forward = F.layer_forward

    def boundary(layer, w_, segs, b_, R, pending, **kw):
        i = cur_window["i"]
        arrays[f"w{i}_L{layer.index}_h"] = b_.h[:R].detach().clone()
        tr.active = str(layer.index) if str(layer.index) in traced else None
        try:
            got = layer_forward(layer, w_, segs, b_, R, pending, **kw)
        finally:
            tr.active = None
        mode, y, wts, inj = got
        torch.cuda.synchronize()
        arrays[f"w{i}_L{layer.index}_y"] = y.detach().clone()
        arrays[f"w{i}_L{layer.index}_wts"] = wts.detach().clone()
        arrays[f"w{i}_L{layer.index}_inj"] = inj[:R].detach().clone()
        return got

    F.layer_forward = boundary
    finish = F.finish

    def head(w_, mixer, b_, R, pending, **kw):
        tr.active = "H" if "H" in traced else None
        try:
            return finish(w_, mixer, b_, R, pending, **kw)
        finally:
            tr.active = None

    F.finish = head
    stage_ple = exl3_pack.stage_ple

    def staged(table, sc, row_ids, at=0):
        i = cur_window["i"]
        arrays[f"w{i}_ple_ids"] = torch.from_numpy(np.ascontiguousarray(row_ids).astype(np.int64))
        arrays[f"w{i}_ple_rows"] = torch.from_numpy(np.ascontiguousarray(table.gather(row_ids)))
        return stage_ple(table, sc, row_ids, at=at)

    exl3_pack.stage_ple = staged

    params = {"hidden": c.hidden, "streams": c.streams, "layers": c.layers, "vocab": c.vocab, "rows": rows,
              "windows": len(WINDOWS), "capacity": 4096, "eps": c.eps, "trace": sorted(traced)}
    pos = 0
    for i, (R, keep) in enumerate(WINDOWS):
        cur_window["i"] = i
        tr.window = i
        tr.seq = 0
        toks = ids[pos:pos + R]
        params[f"w{i}_rows"], params[f"w{i}_keep"], params[f"w{i}_pos"] = R, keep, pos
        arrays[f"w{i}_ids"] = torch.tensor(toks, dtype=torch.int32)
        with rec.scope(f"w{i}_forward", detail=True):
            logits = F.forward(w, st, b, toks)
        torch.cuda.synchronize()
        arrays[f"w{i}_logits"] = logits[:R].detach().clone()
        with rec.scope(f"w{i}_commit", detail=True):
            F.commit(w, st, b, R, keep)
        torch.cuda.synchronize()
        print(f"window {i}: rows {R} keep {keep} at {pos}: argmax {logits[:R].float().argmax(-1).tolist()}", flush=True)
        pos += keep

    ck.save(out / "fixture", params, arrays, extra={"trace": tr.calls})
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps({
        "torch": torch.__version__, "triton": triton.__version__, "tensorfold": tensorfold.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "windows": WINDOWS,
        "layers": c.layers, "traced": sorted(traced), "calls": len(tr.calls)}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
