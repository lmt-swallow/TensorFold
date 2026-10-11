#!/usr/bin/env python3
"""Merge one or more DeepSeek-V4.1 Triton captures (each: a launches.json from the recorder shim or
capture_kernel.py, its TRITON_CACHE_DIR, the mount path the run saw) into one Zig kernel set: <out>/aot.json +
<out>/cubins/ (each <hash>.cubin with the <hash>.ptx it was compiled from), deduplicated by specialization hash, with
a report of kernel names x variant counts and of every variant aot.Set.load would refuse (global or profile scratch, clusters, a PTX ABI that is not the runtime argument
list). Host-side, no GPU.

  python3 -B pack_capture.py --out <dir> \
      --capture <launches.json> <cache dir> <mount> [--capture ...]
``mount`` is TRITON_CACHE_DIR as the run saw it (e.g. /aot/triton0); pass "-" when it ran outside a container.
jit.json is read beside each launches.json; the specializations are merged (do_not_specialize must agree).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def const(v):
    """A constexpr as Zig compares it (aot_pack.py's rule): ints and bools as integers, floats by their fp32 bits."""

    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, dict) and "fp32_bits" in v:
        return {"f32": int(v["fp32_bits"], 16)}
    raise ValueError(f"constexpr {v!r} has no Zig form")


def refusals(k: dict, jit: dict) -> list[str]:
    """Why aot.Set.load or aot_pack would refuse this variant; empty when it is loadable."""

    why = []
    md = k["metadata"]
    if md.get("global_scratch_size", 0):
        why.append(f"global_scratch {md['global_scratch_size']}")
    if md.get("profile_scratch_size", 0):
        why.append(f"profile_scratch {md['profile_scratch_size']}")
    if md.get("num_ctas", 1) != 1:
        why.append(f"num_ctas {md['num_ctas']}")
    runtime = [n for n, t in k["signature"].items() if t != "constexpr"]
    ptx = [x["name"] for x in k["abi"]]
    if ptx != runtime + ["global_scratch", "profile_scratch"]:
        why.append(f"PTX params {ptx} != runtime args {runtime}")
    if k["function"] not in jit:
        why.append("function missing from jit.json")
    for v in k["constexprs"].values():
        try:
            const(v)
        except ValueError as e:
            why.append(str(e))
    return why


def entry(k: dict, jit: dict) -> dict:
    fn = jit[k["function"]]
    nospec = set(fn["do_not_specialize"])
    attrs = {n for n, v in k["attrs"].items() if v}
    runtime = [n for n, t in k["signature"].items() if t != "constexpr"]
    md = k["metadata"]
    return {
        "fn": k["name"], "hash": k["hash"], "name": md["name"], "num_warps": md["num_warps"],
        "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
        "global_scratch": md.get("global_scratch_size", 0), "global_align": md.get("global_scratch_align", 1),
        "profile_scratch": md.get("profile_scratch_size", 0), "pdl": bool(md.get("launch_pdl", False)),
        "params": [{"name": n, "type": k["signature"][n], "div16": n in attrs, "nospec": n in nospec}
                   for n in runtime],
        "consts": {n: const(v) for n, v in k["constexprs"].items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture", nargs=3, action="append", required=True, metavar=("LAUNCHES", "CACHE", "MOUNT"))
    ap.add_argument("--tools", default=str(TOOLS), help="folder of triton_aot_manifest.py")
    a = ap.parse_args()
    manifest_mod = load_module("triton_aot_manifest", Path(a.tools) / "triton_aot_manifest.py")
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "cubins").mkdir(parents=True)
    (out / "manifests").mkdir()
    jit: dict = {}
    kernels: dict[str, dict] = {}
    refused: list[dict] = []
    launches_total: Counter = Counter()
    sources: dict[str, list[str]] = defaultdict(list)
    for i, (launches, cache, mount) in enumerate(a.capture):
        launches_p, cache_p = Path(launches), Path(cache)
        jit_p = launches_p.parent / "jit.json"
        if jit_p.exists():
            for name, spec in json.loads(jit_p.read_text()).items():
                if name in jit and jit[name]["do_not_specialize"] != spec["do_not_specialize"]:
                    raise SystemExit(f"{name}: do_not_specialize differs between captures")
                jit[name] = spec
        manifest_p = out / "manifests" / f"capture{i}.json"
        m = manifest_mod.build(cache_p, launches_p, manifest_p, None if mount == "-" else mount)
        label = f"capture{i}:{launches_p}"
        for k in m["kernels"]:
            for phase, n in k.get("launches", {}).items():
                launches_total[(k["name"], k["hash"])] += n
            why = refusals(k, jit)
            if why:
                refused.append({"name": k["name"], "hash": k["hash"], "why": why, "source": label})
                continue
            if k["hash"] in kernels:
                sources[k["hash"]].append(label)
                continue
            cubin = cache_p / k["cubin"]
            data = cubin.read_bytes()
            if hashlib.sha256(data).hexdigest() != k["cubin_sha256"]:
                raise SystemExit(f"{cubin} changed since the manifest was written")
            (out / "cubins" / f"{k['hash']}.cubin").write_bytes(data)
            ptx = cubin.with_suffix(".ptx")  # the PTX it was compiled from: ptx_lift.py's input
            if ptx.exists():
                shutil.copy(ptx, out / "cubins" / f"{k['hash']}.ptx")
            kernels[k["hash"]] = entry(k, jit)
            sources[k["hash"]].append(label)
    packed = sorted(kernels.values(), key=lambda x: (x["fn"], x["hash"]))
    (out / "aot.json").write_text(json.dumps({"generator": "tools/zig/dsv41/pack_capture.py", "kernels": packed},
                                             indent=1) + "\n")
    by_fn: dict[str, list[dict]] = defaultdict(list)
    for k in packed:
        by_fn[k["fn"]].append(k)
    lines = [f"# DeepSeek-V4.1 Triton kernel set: {len(packed)} variants of {len(by_fn)} kernels; "
             f"{len(refused)} refused", "", "| kernel | variants | launches | consts seen |", "|---|---:|---:|---|"]
    for fn, ks in sorted(by_fn.items()):
        n_launch = sum(launches_total[(fn, k["hash"])] for k in ks)
        seen = sorted({f"{c}={v.get('int', v.get('f32'))}" for k in ks for c, v in k["consts"].items()})
        lines.append(f"| {fn} | {len(ks)} | {n_launch} | {', '.join(seen)[:160]} |")
    if refused:
        lines += ["", "## Refused (aot.Set.load would not take these)", ""]
        for r in refused:
            lines.append(f"- {r['name']} {r['hash'][:12]} ({r['source']}): {'; '.join(r['why'])}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps({"variants": len(packed), "kernels": len(by_fn), "refused": refused,
                                                 "sources": sources, "per_kernel": {fn: len(ks) for fn, ks in by_fn.items()}},
                                                indent=1) + "\n")
    print("\n".join(lines))
    print(f"-> {out}/aot.json, {len(packed)} cubins")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
