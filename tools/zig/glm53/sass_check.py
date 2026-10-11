#!/usr/bin/env python3
"""Compare two builds of the same CUDA kernels instruction by instruction: the torch extension the reference built
(an .so from a capture's extensions/) against the Zig build's fatbin of the vendored source. Each kernel's SASS is
matched by its name with the namespace removed (the extension's anonymous namespace, ours tf_glm53_*). Instruction
addresses, comments and the function's own symbol inside relocations are ignored; every other byte of the text must
be equal. Every kernel of the fatbin must be in the extension and equal to it (the extension may hold more: the
launch variants a setting other than the strict one picks). Exit 1 otherwise.

  python3 sass_check.py --cuobjdump /usr/local/cuda/bin/cuobjdump <extension.so> <kernels.fatbin>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

FUNC = re.compile(r"^\s*Function : (\S+)")
NS = re.compile(r"_ZN(\d+)")


def strip_ns(mangled: str) -> str:
    """`_ZN<len><ns><len><name>...` -> `<len><name>...`: the kernel without its namespace (the extension's anonymous
    `_GLOBAL__N_...`, ours `tf_glm53_*`); other names as they are."""
    m = NS.match(mangled)
    if not m:
        return mangled
    ns_len = int(m.group(1))
    ns = mangled[m.end():m.end() + ns_len]
    if not (ns.startswith("_GLOBAL__N") or ns.startswith("tf_glm53_")):
        return mangled
    return mangled[m.end() + ns_len:]


def kernels(cuobjdump: str, path: str) -> dict[str, list[str]]:
    out = subprocess.run([cuobjdump, "-sass", path], check=True, capture_output=True, text=True).stdout
    found: dict[str, list[str]] = {}
    cur = None
    for line in out.splitlines():
        m = FUNC.match(line)
        if m:
            cur = strip_ns(m.group(1))
            found[cur] = []
            continue
        if cur is None:
            continue
        line = re.sub(r"/\*[0-9a-f]{4,}\*/", "", line)       # instruction addresses
        line = re.sub(r"/\*.*?\*/", "", line).strip()       # encodings and comments
        line = line.replace(" ;", ";")
        if line and not line.startswith(".") and not line.startswith("-"):
            found[cur].append(line)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cuobjdump", default="cuobjdump")
    ap.add_argument("extension")
    ap.add_argument("fatbin")
    a = ap.parse_args()
    ref = kernels(a.cuobjdump, a.extension)
    ours = kernels(a.cuobjdump, a.fatbin)
    bad = 0
    for name, got in sorted(ours.items()):
        sass = ref.get(name)
        if sass is None:
            print(f"MISSING {name} (not in the extension)")
            bad += 1
        elif got != sass:
            at = next((i for i, (x, y) in enumerate(zip(got, sass)) if x != y), min(len(got), len(sass)))
            print(f"DIFF {name}: {len(sass)} / {len(got)} instructions, first difference at {at}: "
                  f"{sass[at] if at < len(sass) else '-'} | {got[at] if at < len(got) else '-'}")
            bad += 1
        else:
            print(f"SAME {name}: {len(sass)} instructions")
    left = len([n for n in ref if n not in ours and "EmptyKernel" not in n])
    print(f"{len(ours) - bad} of {len(ours)} kernels identical; {left} of the extension's not built here")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
