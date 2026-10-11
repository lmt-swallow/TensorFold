#!/usr/bin/env python3
"""The embeddings and decoder layer 0 (a DeltaNet layer) of Qwen3.8-Flash-Next from TensorFold 0.6.6 on an EXL3 pack,
as a stage-by-stage fixture for the Zig engine, with the Triton launches recorded for the AOT pack
(capture_layer0.sh runs this inside the reference image, then packs).

One stream decodes a fixed prompt in windows of WINDOWS rows (R, kept rows): a window of R rows with keep < R is a
drafted window whose tail is rejected, so the next window starts keep rows later and the commit replays the DeltaNet
state. Layer 0's output depends on the tokens and layer 0's own state alone, so a one-layer load of the pack has the
bits the 48-layer model has in its layer 0. For each window i, the arrays w{i}_<name>:

  ids, h0                                          the token ids (int32) and the embedded streams
  a_pss, a_normed, a_xs_normed, a_part, a_act,     the attention hyper-connection: write-back (mode 0) sums,
  a_xs_act, inj_a, a_up, a_mixed, a_xs_mixed       norm, down slices, activation, inject gates, up, mix
  proj                                             the DeltaNet projection rows [qkv | z | b | a]
  conv_in, rec_in                                  the conv window and the recurrent state before the window
  gout, gxs, rec_out, sk, sv, sg, sb               the chain's read-out, its sums, the state after R rows, replay inputs
  branch                                           out_proj's bf16 branch
  h1, m_pss, m_normed, m_xs_normed, m_part, m_act, the MLP hyper-connection: streams after the write-back (mode 1)
  m_xs_act, inj_m, m_up, m_mixed, m_xs_mixed       and its read-out
  logits, pick, wts, y                             router logits, picks, weights, the experts' slots (fp32)
  h2, pss2                                         the streams after the MoE write-back (mode 2): layer 0's output
  conv_out, rec_commit                             the conv window and the state committed after keep rows

Every staged window is also run through the engine's own `layer_forward` from the same state, and its outputs must
equal the staged ones bit for bit (the staging is the engine's call sequence, not a re-implementation).

Run inside the reference image with TRITON_CACHE_DIR set:
  python3 -B /w/tools/zig/qwen38fn/dump_layer0.py --tools /w/tools/zig --out /aot/out --model /model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsv41"))
import capture_kernel as ck  # noqa: E402  (the recorder loader, fixture writer, jit specialization dump)

DEV = torch.device("cuda:0")
WINDOWS = ((1, 1), (1, 1), (1, 1), (2, 2), (3, 3), (7, 4), (16, 16), (5, 1), (1, 1))
PROMPT = ("The DeltaNet layer keeps a fixed-size state per head, so a long conversation costs the same memory per "
          "token as a short one. Explain, step by step, why replaying a window's accepted prefix must reproduce the "
          "serial state bit for bit, and what breaks when it does not.")


def bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bytes (NaNs and signed zeros compare as bits)."""

    return t.contiguous().view(-1).view(torch.uint8)


def one_layer_config():
    """Make the family's Config.read report the first layer only (the loader then reads layer 0's tensors alone)."""

    from tensorfold.families.qwen4_exp.cuda import weight_types as WT

    read = WT.Config.read.__func__

    def first(cls, model_dir):
        c = read(cls, model_dir)
        c.layers = 1
        c.layer_types = c.layer_types[:1]
        return c

    WT.Config.read = classmethod(first)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ck.load_recorder(a.tools)

    import triton
    import tensorfold
    from tokenizers import Tokenizer
    from tensorfold.cuda import moe as moe_mod
    from tensorfold.cuda.exl3.experts import routed
    from tensorfold.families.qwen4_exp.cuda import exl3 as X
    from tensorfold.families.qwen4_exp.cuda import forward as F
    from tensorfold.families.qwen4_exp.cuda import gdn as G
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
    from tensorfold.families.qwen4_exp.cuda.weight_types import Config

    full = Config.read(a.model)
    offset_all = X.centred_offset(Pack(a.model), [f"model.language_model.layers.{i}.attn_hyper_connection.hc_norm.weight"
                                                  for i in range(full.layers)])
    one_layer_config()
    w = X.load(a.model, mtp=False)
    c = w.cfg
    assert c.layers == 1 and w.layers[0].linear, "layer 0 must be a DeltaNet layer"
    assert w.meta["centred_offset"] == offset_all, "the one-layer load must decide the norm offset as the full load"
    layer = w.layers[0]
    S, D, eps, low = c.streams, c.hidden, c.eps, c.low
    rows = max(r for r, _ in WINDOWS)
    st = State(w, 4096, rows)
    b = Buffers(w, rows, 4096, moe_prefill=True)
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    ids = tok.encode(PROMPT).ids
    need = max(sum(k for _, k in WINDOWS[:i]) + r for i, (r, _) in enumerate(WINDOWS))
    assert len(ids) >= need, f"the prompt has {len(ids)} tokens, the windows read {need}"

    arrays: dict[str, torch.Tensor] = {}
    params = {"hidden": D, "streams": S, "low": low, "nk": c.nk, "nv": c.nv, "conv_dim": c.conv_dim,
              "proj_width": G.widths(c.nk, c.nv)[1], "eps": eps, "experts": c.experts, "top_k": c.top_k,
              "slots": c.top_k + 1, "rows": rows, "windows": len(WINDOWS), "centred_offset": offset_all,
              "f16_sk_down": layer.attn_hc.down.sk, "f16_sk_up": layer.attn_hc.up.sk, "f16_sk_ba": None}
    params["f16_sk_ba"] = layer.gdn.proj.parts[2].sk
    hc_a, hc_m, g, m = layer.attn_hc, layer.mlp_hc, layer.gdn, layer.moe
    buf = b.moe
    pos = 0
    for i, (R, keep) in enumerate(WINDOWS):
        p = f"w{i}_"
        toks = ids[pos:pos + R]
        params[p + "rows"], params[p + "keep"], params[p + "pos"] = R, keep, pos
        b.ids[:R].copy_(torch.tensor(toks, dtype=torch.int32))
        cur = st.cur[0]
        conv_in, rec_in = st.conv[0].clone(), st.rec[cur, 0].clone()
        h = b.h
        got: dict[str, torch.Tensor] = {"ids": b.ids[:R].clone(), "conv_in": conv_in, "rec_in": rec_in}

        def keep_(name: str, t: torch.Tensor) -> None:
            got[name] = t.clone()

        def readout(hc, tag: str, inject) -> None:
            glue.hc_normed(h[:R], b.pss[:R], hc.scale, b.normed[:R], b.xs_normed[:R], S, eps)
            keep_(tag + "normed", b.normed[:R])
            keep_(tag + "xs_normed", b.xs_normed[:R])
            part = hc.down.partials(b.normed[:R])
            assert part.dim() == 3
            keep_(tag + "part", part)
            glue.hc_reduce_act(part, b.act[:R], b.xs_act[:R], inject, S, low)
            keep_(tag + "act", b.act[:R])
            keep_(tag + "xs_act", b.xs_act[:R])
            F._mm(b.act[:R], hc.up, b.xs_act[:R], b.up[:R], b)
            keep_(tag + "up", b.up[:R])
            glue.hc_mix(b.up[:R], b.normed[:R], b.mixed[:R], b.xs_mixed[:R], S)
            keep_(tag + "mixed", b.mixed[:R])
            keep_(tag + "xs_mixed", b.xs_mixed[:R])

        with rec.scope(f"w{i}_staged", detail=True):
            F._embed(w, b.ids[:R], S, h[:R])
            keep_("h0", h[:R])
            glue.hc_writeback(h[:R], h[:R], b.pss[:R], S, 0)
            keep_("a_pss", b.pss[:R])
            readout(hc_a, "a_", b.inj_a[:R])
            keep_("inj_a", b.inj_a[:R])
            F._mm(b.mixed[:R], g.proj, b.xs_mixed[:R], b.proj[0, :R], b)
            keep_("proj", b.proj[0, :R])
            G.chain(b.proj[0, :R], st.conv[0], g.conv, st.rec[cur, 0], g.a_log, g.dt_bias, g.norm, eps, R,
                    st.scratch[0], st.rec[1 - cur, 0], b.gout[:R], b.gxs[:R])
            keep_("gout", b.gout[:R])
            keep_("gxs", b.gxs[:R])
            keep_("rec_out", st.rec[1 - cur, 0])
            sc = st.scratch[0]
            for name, t in (("sk", sc.k), ("sv", sc.v), ("sg", sc.g), ("sb", sc.b)):
                keep_(name, t[:R])
            mode, branch = F._out_proj(w, b, b.gout[:R], g.out, b.gxs[:R], R)
            assert mode == 1
            keep_("branch", branch)
            glue.hc_writeback(h[:R], h[:R], b.pss[:R], S, 1, branch=branch, inject=b.inj_a[:R])
            keep_("h1", h[:R])
            keep_("m_pss", b.pss[:R])
            readout(hc_m, "m_", b.inj_m[:R])
            keep_("inj_m", b.inj_m[:R])
            moe_mod.router(b.mixed[:R], m.router, buf.logits[:R])
            keep_("logits", buf.logits[:R])
            moe_mod.select_rows(buf.logits[:R], buf, c.top_k, c.experts)
            keep_("pick", buf.pick[:R])
            keep_("wts", buf.wts[:R])
            y = routed(b.mixed[:R], buf.pick[:R], None, m.experts, w.x3.moe, None, R).view(R, buf.slots, -1)
            keep_("y", y)
            F._writeback(h, b, R, c, (2, y, buf.wts[:R], b.inj_m))
            keep_("h2", h[:R])
            keep_("pss2", b.pss[:R])
        torch.cuda.synchronize()

        # the engine's own call sequence from the same state must give the same bits
        st.conv[0].copy_(conv_in)
        st.rec[cur, 0].copy_(rec_in)
        with rec.scope(f"w{i}_engine", detail=True):
            F._embed(w, b.ids[:R], S, h[:R])
            pending = F.layer_forward(layer, w, [(st, 0, R)], b, R, None)
            F._writeback(h, b, R, c, pending)
        torch.cuda.synchronize()
        for name, t in (("h2", h[:R]), ("m_mixed", b.mixed[:R]), ("inj_m", b.inj_m[:R]), ("rec_out", st.rec[1 - cur, 0]),
                        ("gout", b.gout[:R]), ("wts", buf.wts[:R]), ("pick", buf.pick[:R])):
            if not torch.equal(bits(t), bits(got[name])):
                raise SystemExit(f"window {i}: the staged {name} differs from layer_forward's")

        with rec.scope(f"w{i}_commit", detail=True):
            F.commit(w, st, b, R, keep)
        torch.cuda.synchronize()
        got["conv_out"] = st.conv[0].clone()
        got["rec_commit"] = st.rec[st.cur[0], 0].clone()
        for name, t in got.items():
            arrays[p + name] = t
        print(f"window {i}: rows {R} keep {keep} at {pos}: pick[0]={buf.pick[0].tolist()} "
              f"h2[0,:4]={h[0, :4].float().tolist()}", flush=True)
        pos += keep

    ck.save(out / "fixture", params, arrays)
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(ck.specialization(), indent=1, sort_keys=True) + "\n")
    (out / "capture.json").write_text(json.dumps({
        "torch": torch.__version__, "triton": triton.__version__, "tensorfold": tensorfold.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "windows": WINDOWS,
        "prompt_tokens": len(ids)}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
