#!/usr/bin/env python3
"""Union several captured Triton sets (aot.json + cubins/, as pack_capture.py / aot_pack.py write them) into one:
variants are keyed by their cubin hash, the first occurrence wins, cubins are copied. Usage:
merge_aot.py --out <dir> <set-dir>... ; rope/ and token_map.json of the first set that has them are copied too."""
import argparse, json, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("sets", nargs="+")
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "cubins"), exist_ok=True)
seen, kernels, generator = set(), [], None
for d in a.sets:
    j = json.load(open(os.path.join(d, "aot.json")))
    generator = generator or j.get("generator")
    for k in j["kernels"]:
        if k["hash"] in seen:
            continue
        src = os.path.join(d, "cubins", k["hash"] + ".cubin")
        if not os.path.exists(src):
            cands = [f for f in os.listdir(os.path.join(d, "cubins")) if f.startswith(k["hash"][:16])]
            if not cands:
                print(f"{d}: no cubin for {k['name']} {k['hash'][:16]}", file=sys.stderr)
                continue
            src = os.path.join(d, "cubins", cands[0])
        shutil.copy2(src, os.path.join(a.out, "cubins", os.path.basename(src)))
        seen.add(k["hash"])
        kernels.append(k)
    for extra in ("rope", "token_map.json"):
        s, t = os.path.join(d, extra), os.path.join(a.out, extra)
        if os.path.exists(s) and not os.path.exists(t):
            (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, t)
json.dump({"generator": generator, "kernels": kernels}, open(os.path.join(a.out, "aot.json"), "w"), indent=1)
by = {}
for k in kernels:
    by[k["name"]] = by.get(k["name"], 0) + 1
print(f"{len(kernels)} variants of {len(by)} kernels -> {a.out}")
