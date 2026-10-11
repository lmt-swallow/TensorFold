#!/usr/bin/env python3
"""Lift captured DeepSeek-V4.1 Triton kernels from their PTX into CUDA device functions, with a launchable kernel set
that proves the lift: same entry names and parameter lists as the captured set, so aot.Set loads it in place of the
original and every fixture test runs on the lifted code unchanged.

  python3 -B ptx_lift.py collect --set <aot dir> --out <ptx dir> --roots <dir> [<dir> ...]
      <ptx dir>/<hash>.ptx for every variant of the set: the PTX Triton compiled the variant's cubin from, found by
      the cubin's sha256 among the <x>.cubin/<x>.ptx pairs under the roots (Triton caches, capture fixtures).
  python3 -B ptx_lift.py lift --set <aot dir> [--ptx <ptx dir>] --out <dir> [--fn _hc_post ...] [--mode native|virt]
                              [--cuda <toolkit dir>]
      <dir>/include/tf_lift.cuh, <dir>/include/<fn>_<hash12>.cuh (one device function a variant) and
      <dir>/src/<hash>.cu (its __global__ wrapper). With --cuda, the wrappers compiled as Triton compiles (nvcc -ptx,
      then ptxas -lineinfo --regAllocOptLevel=2 --gpu-name=sm_121a, Triton's make_cubin) into <dir>/cubins/, the set's
      <dir>/aot.json (variants not lifted keep their original cubin), and <dir>/report.{md,json}: per variant whether
      the lifted SASS is instruction for instruction the original's, its FFMA/FMUL/FADD counts and registers.

The device function (`tfl_<fn>_<hash12>`) takes the kernel's PTX parameters in order (runtime arguments, then
global_scratch and profile_scratch) and a TflBlock: the virtual block index and grid (%ctaid, %nctaid), the thread's
index in the virtual block (%tid.x), the shared-space address of the block's dynamic shared memory (global_smem), and
the hardware barrier and thread count its `bar.sync 0` becomes. The body is the entry's PTX verbatim apart from those
substitutions, `ld.param` turned into moves from the operands, `ret` into a branch to the body's end, and every
register renamed into its own namespace (%tl_*), so the instructions and their order are Triton's. Kept as they are:
griddepcontrol.wait and .launch_dependents (no-ops in a grid launched without programmatic dependence), warp shuffles
and bar.warp.sync (a virtual block is whole warps). The caller owns the launch shape: `threads` threads, warp-aligned,
run the body together, and `smem` points at `shared` bytes (16-byte aligned) nobody else touches meanwhile.

--mode native: the wrapper runs one virtual block a CTA with the hardware indices (the bit-exactness proof: its SASS is
the original's). --mode virt: the wrapper runs the virtual block in the CTA's second warp group (2x the warps, the
first group idle), on block index (grid - 1 - blockIdx) in every dimension, named barrier 1 and shared memory 1 KiB in,
so a fixture that passes on this set shows that nothing in the body reads the hardware block, thread or barrier state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ARCH = "sm_121a"
# Triton's make_cubin (triton/backends/nvidia/compiler.py, the fork's image): ptxas -lineinfo [--fmad=false] -v
# --regAllocOptLevel=2 --gpu-name=<arch>; every captured variant has enable_fp_fusion (no --fmad=false).
PTXAS_FLAGS = ["-lineinfo", "--regAllocOptLevel=2", f"--gpu-name={ARCH}"]
SMEM_MAX = 101376  # GB10's opt-in dynamic shared memory a block
VIRT_SMEM_OFFSET = 1024

SPECIALS = {
    "%ctaid.x": ("x", "r"), "%ctaid.y": ("y", "r"), "%ctaid.z": ("z", "r"),
    "%nctaid.x": ("nx", "r"), "%nctaid.y": ("ny", "r"), "%nctaid.z": ("nz", "r"),
    "%tid.x": ("tid", "r"),
}
# special registers a body may read unchanged inside a warp-aligned virtual block
SPECIALS_KEPT = {"%laneid", "%lanemask_eq", "%lanemask_lt", "%lanemask_le", "%lanemask_gt", "%lanemask_ge"}


class Refused(Exception):
    """The entry uses something the lift cannot carry into a device function."""


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# -- collect ---------------------------------------------------------------------------------------------------------

def collect(a) -> int:
    kernels = json.loads((Path(a.set) / "aot.json").read_text())["kernels"]
    want: dict[str, list[dict]] = {}  # cubin sha256 -> the variants with that cubin (specializations may share one)
    for k in kernels:
        want.setdefault(sha256(Path(a.set) / "cubins" / f"{k['hash']}.cubin"), []).append(k)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    found = {}
    for root in a.roots:
        for cubin in Path(root).expanduser().rglob("*.cubin"):
            ptx = cubin.with_suffix(".ptx")
            if not ptx.exists():
                continue
            s = sha256(cubin)
            if s in want and s not in found:
                found[s] = ptx
                for k in want[s]:
                    shutil.copy(ptx, out / f"{k['hash']}.ptx")
    missing = [k for s, ks in want.items() if s not in found for k in ks]
    for k in missing:
        print(f"missing PTX: {k['fn']} {k['hash'][:12]}")
    print(f"{len(kernels) - len(missing)} of {len(kernels)} variants' PTX -> {out}")
    return 1 if missing else 0


# -- the PTX entry ---------------------------------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def parse_entry(ptx: str) -> dict:
    """The one .entry of a Triton module: name, parameters [(ptx type, name)], directives, body text, module symbols."""

    text = strip_comments(ptx)
    m = re.search(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)", text, re.S)
    if not m:
        raise Refused("no .entry")
    if len(re.findall(r"\.entry\s", text)) != 1:
        raise Refused("more than one .entry")
    if re.search(r"^\s*\.(visible\s+|extern\s+)?\.?func\b", text, re.M):
        raise Refused("the module has .func definitions")
    params = []
    for decl in m.group(2).split(","):
        decl = decl.strip()
        pm = re.match(r"\.param\s+\.(u64|u32|f32|s32|b32|b64)\b.*?\s(\w+)$", decl)
        if not pm:
            raise Refused(f"parameter {decl!r}")
        params.append((pm.group(1), pm.group(2)))
    rest = text[m.end():]
    open_at = rest.index("{")
    directives = rest[:open_at].split()
    depth = 0
    for i, ch in enumerate(rest[open_at:], open_at):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            body = rest[open_at + 1:i]
            break
    else:
        raise Refused("unbalanced braces")
    head = text[:m.start()]
    symbols = re.findall(r"^\s*\.(?:global|const|shared|extern\s+\.shared)\b[^;]*?\s([\w$]+)\s*(?:\[[^\]]*\])?\s*(?:=[^;]*)?;",
                         head, re.M)
    reqntid = None
    for i, d in enumerate(directives):
        if d in (".reqntid", ".maxntid"):
            reqntid = int(directives[i + 1].rstrip(","))
        elif d.startswith(".") and d not in (".reqntid", ".maxntid"):
            raise Refused(f"entry directive {d}")
    return {"name": m.group(1), "params": params, "reqntid": reqntid, "body": body, "symbols": symbols,
            "directives": " ".join(directives)}


def lift_body(e: dict) -> tuple[list[str], list[tuple[str, str]], dict]:
    """The body as inline-asm lines (with '%%' escapes and %<n> operand references) and its operands [(source, constraint)].

    Operand sources: ("param", index) for a parameter, ("block", field) for a TflBlock field."""

    params = {name: (i, ty) for i, (ty, name) in enumerate(e["params"])}
    regs = set()
    lines = []
    stats = Counter()
    for raw in e["body"].split("\n"):
        line = raw.strip()
        if not line or line.startswith(".loc") or line.startswith(".file"):
            continue
        for rm in re.finditer(r"\.reg\s+\.\w+\s+%(\w+?)<\d+>", line):
            regs.add(rm.group(1))
        lines.append(line)
    operands: list[tuple[tuple, str]] = []
    index: dict[tuple, int] = {}

    def op(src: tuple, constraint: str) -> str:
        if src not in index:
            index[src] = len(operands)
            operands.append((src, constraint))
        return f"\x00{index[src]}\x00"

    module_syms = [s for s in e["symbols"] if s != "global_smem"]
    out = []
    for line in lines:
        pm = re.match(r"^(@!?%\w+\s+)?ld\.param\.(\w+)\s+(%\w+),\s*\[(\w+)\];$", line)
        if pm:
            pred, ty, dst, pname = pm.groups()
            if pname not in params:
                raise Refused(f"ld.param of {pname}")
            i, pty = params[pname]
            wide = pty == "u64"
            ref = op(("param", i), "l" if wide else "r")
            if ty in ("b64", "u64", "s64") and wide:
                line = f"{pred or ''}mov.b64 {dst}, {ref};"
            elif ty in ("b32", "u32", "f32") and not wide:
                line = f"{pred or ''}mov.b32 {dst}, {ref};"
            elif ty == "s32" and not wide and dst.startswith("%rd"):
                line = f"{pred or ''}cvt.s64.s32 {dst}, {ref};"
            else:
                raise Refused(f"ld.param.{ty} of a .{pty} parameter into {dst}")
            stats["ld.param"] += 1
        if "_param_" in line:
            raise Refused(f"parameter reference: {line}")
        for sym in module_syms:
            if re.search(rf"(?<![\w$]){re.escape(sym)}(?![\w$])", line):
                raise Refused(f"module symbol {sym} used: {line}")
        if re.search(r"\bglobal_smem\b", line):
            if re.match(r"^(@!?%\w+\s+)?mov\.(u64|b64|s64)\b", line) or re.search(r"cvta", line):
                raise Refused(f"64-bit use of global_smem: {line}")
            line = re.sub(r"\bglobal_smem\b", op(("block", "smem"), "r"), line)
            stats["global_smem"] += 1
        bm = re.match(r"^(@!?%\w+\s+)?(bar|barrier)\.(sync|arrive|red)(\S*)\s+(.*);$", line)
        if bm:
            pred, kind, what, mods, args = bm.groups()
            if kind != "bar" or what != "sync" or mods or args.strip() != "0":
                raise Refused(f"barrier form {line}")
            line = f"{pred or ''}bar.sync {op(('block', 'bar'), 'r')}, {op(('block', 'threads'), 'r')};"
            stats["bar.sync"] += 1
        rm = re.match(r"^(@!?%\w+\s+)?ret;$", line)
        if rm:
            pred = rm.group(1)
            line = f"{pred}bra $tfl_exit;" if pred else "bra.uni $tfl_exit;"
            stats["ret"] += 1
        if "griddepcontrol" in line:
            stats["griddepcontrol"] += 1

        def special(sm: re.Match) -> str:
            s = sm.group(0)
            if s in SPECIALS:
                field, c = SPECIALS[s]
                stats[s] += 1
                return op(("block", field), c)
            if s in SPECIALS_KEPT:
                return s
            raise Refused(f"special register {s}")

        line = re.sub(r"%(?:ctaid|nctaid|tid|ntid|nctarank|ctarank|cluster\w*|warpid|nwarpid|smid|nsmid|gridid|"
                      r"dynamic_smem_size|total_smem_size|envreg\d+|laneid|lanemask_\w+|clock\w*|globaltimer\w*)"
                      r"(?:\.[xyz])?", special, line)
        if regs:
            line = re.sub(r"%(" + "|".join(sorted(regs, key=len, reverse=True)) + r")(?=<\d+>|\d+\b)", r"%tl_\1", line)
        if re.search(r"%(?!tl_|laneid|lanemask)\w", line):
            raise Refused(f"unhandled % name: {line}")
        out.append(line)
    out.append("$tfl_exit:")
    asm = []
    for line in out:
        line = line.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
        line = re.sub(r"\x00(\d+)\x00", r"%\1", line)
        indent = "" if line.endswith(":") else "\t"
        asm.append(indent + line)
    return asm, operands, dict(stats)


# -- emitted sources -------------------------------------------------------------------------------------------------

COMMON = """\
// Generated by tools/zig/dsv41/ptx_lift.py; do not edit.
// The block a lifted Triton kernel body runs as; see ptx_lift.py for the contract.
#pragma once

struct TflBlock {
  unsigned x, y, z;     // the virtual block index (the body's %ctaid)
  unsigned nx, ny, nz;  // the virtual grid (%nctaid)
  unsigned tid;         // the thread's index in the virtual block (%tid.x): 0 <= tid < threads
  unsigned smem;        // shared-space address of the virtual block's dynamic shared memory (global_smem), 16-aligned
  unsigned bar;         // the hardware barrier the body's `bar.sync 0` waits on (0: the whole CTA)
  unsigned threads;     // the virtual block's threads, a multiple of 32: the count that barrier waits for
};

extern __shared__ __align__(16) unsigned char tfl_dyn_smem[];

__device__ __forceinline__ unsigned tfl_smem_base() { return (unsigned)__cvta_generic_to_shared(tfl_dyn_smem); }

// One virtual block a CTA, with the hardware indices: the body runs as Triton launched it.
__device__ __forceinline__ TflBlock tfl_native_block(unsigned threads) {
  return {blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z, threadIdx.x, tfl_smem_base(), 0u, threads};
}
"""

CTYPE = {"u64": "unsigned long long", "u32": "unsigned", "s32": "int", "b32": "unsigned", "b64": "unsigned long long",
         "f32": "float"}


def cident(name: str) -> str:
    return re.sub(r"\W", "_", name)


def emit(k: dict, e: dict, ptx_sha: str, mode: str) -> tuple[str, str, dict]:
    """(the .cuh, the wrapper .cu, facts) of variant `k` (an aot.json entry) from its parsed entry `e`."""

    asm, operands, stats = lift_body(e)
    names = [p["name"] for p in k["params"]] + ["global_scratch", "profile_scratch"]
    if len(names) != len(e["params"]):
        raise Refused(f"aot.json has {len(names)} parameters, the PTX {len(e['params'])}")
    threads = 32 * k["num_warps"]
    if e["reqntid"] not in (None, threads):
        raise Refused(f".reqntid {e['reqntid']} != num_warps {k['num_warps']} x 32")
    h12 = k["hash"][:12]
    fname = f"tfl{cident(k['fn'])}_{h12}"
    args = [f"{CTYPE[ty]} p_{cident(n)}" for (ty, _), n in zip(e["params"], names)]
    exprs = []
    for (src, c) in operands:
        if src[0] == "param":
            ty = e["params"][src[1]][0]
            v = f"p_{cident(names[src[1]])}"
            exprs.append(f'"{c}"(__float_as_uint({v}))' if ty == "f32" else f'"{c}"({v})')
        else:
            exprs.append(f'"{c}"(b.{src[1]})')
    consts = ", ".join(f"{n}={v.get('int', v.get('f32'))}" for n, v in k["consts"].items())
    body = "\n".join(f'      "{line}\\n"' for line in asm)
    cuh = f"""\
// Generated by tools/zig/dsv41/ptx_lift.py from PTX sha256 {ptx_sha}; do not edit.
// Triton kernel {k['fn']}, specialization {k['hash']}:
//   {consts}
//   {k['num_warps']} warps ({threads} threads), {k['shared']} bytes of dynamic shared memory, PDL {int(k['pdl'])}
#pragma once
#include "tf_lift.cuh"

#define {fname.upper()}_THREADS {threads}
#define {fname.upper()}_SHARED {k['shared']}

// {k['fn']}'s body on virtual block `b`: every thread of the virtual block calls it (see ptx_lift.py).
__device__ __forceinline__ void {fname}(
    {(','+chr(10)+'    ').join(args + ['const TflBlock& b'])}) {{
  asm volatile(
      "{{\\n"
{body}
      "}}\\n"
      :
      : {(','+chr(10)+'        ').join(exprs)}
      : "memory");
}}
"""
    params = ", ".join(args)
    call = ", ".join([f"p_{cident(n)}" for n in names] + ["b"])
    if mode == "native":
        launch_threads, smem_off = threads, 0
        block = f"  const TflBlock b = tfl_native_block({threads}u);\n"
    else:
        launch_threads = 2 * threads
        smem_off = VIRT_SMEM_OFFSET if k["shared"] and k["shared"] + VIRT_SMEM_OFFSET <= SMEM_MAX else 0
        block = (f"  if (threadIdx.x < {threads}u) return;  // the first warp group idles\n"
                 f"  const TflBlock b = {{gridDim.x - 1 - blockIdx.x, gridDim.y - 1 - blockIdx.y, gridDim.z - 1 - blockIdx.z,\n"
                 f"                      gridDim.x, gridDim.y, gridDim.z, threadIdx.x - {threads}u,\n"
                 f"                      tfl_smem_base() + {smem_off}u, 1u, {threads}u}};\n")
    cu = f"""\
// Generated by tools/zig/dsv41/ptx_lift.py ({mode}); do not edit.
#include "{k['fn']}_{h12}.cuh"

extern "C" __global__ void __launch_bounds__({launch_threads}) {e['name']}({params}) {{
{block}  {fname}({call});
}}
"""
    facts = {"threads": launch_threads, "shared": k["shared"] + smem_off, "num_warps": launch_threads // 32,
             "subst": stats, "operands": len(operands)}
    return cuh, cu, facts


# -- build and SASS comparison ---------------------------------------------------------------------------------------

def sass(cuda: Path, cubin: Path) -> list[str]:
    """The cubin's instructions (text without addresses and encodings)."""

    out = subprocess.run([str(cuda / "bin" / "cuobjdump"), "-sass", str(cubin)], check=True, capture_output=True,
                         text=True).stdout
    ins = []
    for line in out.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
        if m:
            ins.append(re.sub(r"\s+", " ", m.group(1)))
    return ins


def opcode_counts(ins: list[str], modifiers: bool = False) -> Counter:
    """Instructions by opcode (with its modifiers, e.g. FFMA.FTZ, when `modifiers`), the NOP padding left out."""

    c = Counter()
    for i in ins:
        i = re.sub(r"^@!?U?P\w+\s+", "", i)
        op = i.split(" ")[0]
        if re.match(r"HFMA2(\.MMA)?\s+\S+,\s*-RZ,\s*RZ,", i):
            op = "MOV(HFMA2)"  # ptxas materializes constants as HFMA2 -RZ, RZ, imm: a move, not arithmetic
        if op != "NOP":
            c[op if modifiers else op.split(".")[0]] += 1
    return c


def resources(cuda: Path, cubin: Path) -> dict:
    out = subprocess.run([str(cuda / "bin" / "cuobjdump"), "-res-usage", str(cubin)], check=True,
                         capture_output=True, text=True).stdout
    m = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", out)
    return {"reg": int(m.group(1)), "stack": int(m.group(2)), "shared": int(m.group(3)), "local": int(m.group(4))} if m else {}


def build(cuda: Path, out: Path, k: dict, threads: int) -> Path:
    src = out / "src" / f"{k['hash']}.cu"
    ptx = out / "build" / f"{k['hash']}.ptx"
    cubin = out / "cubins" / f"{k['hash']}.cubin"
    subprocess.run([str(cuda / "bin" / "nvcc"), f"-arch={ARCH}", "-O3", "-ptx", "-I", str(out / "include"), str(src),
                    "-o", str(ptx)], check=True)
    text = ptx.read_text()
    # Triton's entries carry .reqntid (the exact block size), which __launch_bounds__ can only state as .maxntid
    text, n = re.subn(rf"\.maxntid {threads}(, 1, 1)?\n", f".reqntid {threads}\n", text)
    if n != 1:
        raise SystemExit(f"{src}: expected one .maxntid {threads}")
    ptx.write_text(text)
    subprocess.run([str(cuda / "bin" / "ptxas"), *PTXAS_FLAGS, str(ptx), "-o", str(cubin)], check=True)
    return cubin


FLOAT_OPS = ("FFMA", "FMUL", "FADD", "FFMA2", "FMUL2", "FADD2", "HFMA2", "HMUL2", "HADD2", "MUFU", "HMMA")
# opcodes whose results are floating point or depend on rounding: F* (FFMA, FMUL, FADD, FMNMX, FSETP, F2F, F2FP, F2I,
# FRND, ...), H* (HMMA, HFMA2, ...), MUFU, I2F, and the other MMA families
FP_PREFIXES = ("F", "H", "MUFU", "I2F", "DMMA", "QMMA", "OMMA", "UTC")


def fp_mix(mix: Counter) -> Counter:
    """The floating-point part of an instruction mix (opcode with modifiers -> count)."""

    return Counter({op: n for op, n in mix.items() if op.startswith(FP_PREFIXES)})


def lift(a) -> int:
    src_set = Path(a.set)
    out = Path(a.out)
    for d in ("include", "src", "build", "cubins"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "include" / "tf_lift.cuh").write_text(COMMON)
    kernels = json.loads((src_set / "aot.json").read_text())["kernels"]
    cuda = Path(a.cuda) if a.cuda else None
    rows, packed = [], []
    for k in kernels:
        row = {"fn": k["fn"], "hash": k["hash"], "lifted": False}
        entry = dict(k)
        cubin_src = src_set / "cubins" / f"{k['hash']}.cubin"
        if a.fn and k["fn"] not in a.fn:
            row["note"] = "not selected"
        else:
            ptx_p = Path(a.ptx or src_set / "cubins") / f"{k['hash']}.ptx"
            try:
                if not ptx_p.exists():
                    raise Refused("no PTX collected")
                e = parse_entry(ptx_p.read_text())
                if e["name"] != k["name"]:
                    raise Refused(f"entry {e['name']} != {k['name']}")
                cuh, cu, facts = emit(k, e, sha256(ptx_p), a.mode)
                (out / "include" / f"{k['fn']}_{k['hash'][:12]}.cuh").write_text(cuh)
                (out / "src" / f"{k['hash']}.cu").write_text(cu)
                row.update(lifted=True, **facts)
                entry["num_warps"] = facts["num_warps"]
                entry["shared"] = facts["shared"]
            except Refused as why:
                row["note"] = f"refused: {why}"
        if cuda and row["lifted"]:
            cubin = build(cuda, out, k, row["threads"])
            a_ins, b_ins = sass(cuda, cubin_src), sass(cuda, cubin)
            ca, cb = opcode_counts(a_ins), opcode_counts(b_ins)
            ma, mb = opcode_counts(a_ins, True), opcode_counts(b_ins, True)
            row.update(sass_equal=a_ins == b_ins, ins=(sum(ma.values()), sum(mb.values())), mix_equal=ma == mb,
                       mix_diff={op: (ma[op], mb[op]) for op in sorted(set(ma) | set(mb)) if ma[op] != mb[op]},
                       float_ops={op: (ca[op], cb[op]) for op in FLOAT_OPS if ca[op] or cb[op]},
                       float_equal=fp_mix(ma) == fp_mix(mb),
                       res=(resources(cuda, cubin_src), resources(cuda, cubin)))
        elif cuda:
            shutil.copy(cubin_src, out / "cubins" / f"{k['hash']}.cubin")
        rows.append(row)
        packed.append(entry)
    if cuda:
        (out / "aot.json").write_text(json.dumps({"generator": f"tools/zig/dsv41/ptx_lift.py ({a.mode})",
                                                  "source": str(src_set), "kernels": packed}, indent=1) + "\n")
    (out / "report.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = [f"# ptx_lift ({a.mode}) of {src_set}", "",
             "SASS order: the instruction sequence is the original's; mix: the same instructions (opcode and modifiers) "
             "in any order and registers, NOP padding left out; fp mix: the floating-point and conversion instructions "
             "(F*, H*, MUFU, I2F, MMA) with their modifiers.", "",
             "| kernel | hash | lifted | SASS order | mix | fp mix | instructions | FFMA/FMUL/FADD orig -> lifted | regs | notes |",
             "|---|---|---|---|---|---|---:|---|---|---|"]
    yn = lambda r, key: ("yes" if r[key] else "no") if key in r else ""
    for r in rows:
        fo = r.get("float_ops", {})
        f3 = "/".join(str(fo.get(op, (0, 0))[0]) for op in ("FFMA", "FMUL", "FADD"))
        g3 = "/".join(str(fo.get(op, (0, 0))[1]) for op in ("FFMA", "FMUL", "FADD"))
        res = r.get("res")
        regs = f"{res[0].get('reg')} -> {res[1].get('reg')}" if res else ""
        ins = f"{r['ins'][0]} -> {r['ins'][1]}" if "ins" in r else ""
        note = r.get("note", "")
        if r.get("mix_diff"):
            note = (note + " " + ", ".join(f"{op} {x}->{y}" for op, (x, y) in list(r["mix_diff"].items())[:6])).strip()
        lines.append(f"| {r['fn']} | {r['hash'][:12]} | {'yes' if r['lifted'] else 'no'} | {yn(r, 'sass_equal')} | "
                     f"{yn(r, 'mix_equal')} | {yn(r, 'float_equal')} | {ins} | {f3 + ' -> ' + g3 if fo else ''} | {regs} | {note} |")
    lifted = [r for r in rows if r["lifted"]]
    summary = f"{len(lifted)} of {len(rows)} variants lifted"
    if cuda:
        summary += (f"; SASS order equal {sum(bool(r.get('sass_equal')) for r in lifted)}, instruction mix equal "
                    f"{sum(bool(r.get('mix_equal')) for r in lifted)}, floating-point mix equal "
                    f"{sum(bool(r.get('float_equal')) for r in lifted)}")
    lines += ["", summary]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if cuda and a.mode == "native":
        return 0 if all(r.get("float_equal") for r in lifted) else 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--set", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--roots", nargs="+", required=True)
    l = sub.add_parser("lift")
    l.add_argument("--set", required=True)
    l.add_argument("--ptx", help="the variants' <hash>.ptx (default: beside the cubins, as pack_capture.py keeps them)")
    l.add_argument("--out", required=True)
    l.add_argument("--fn", nargs="*")
    l.add_argument("--mode", choices=("native", "virt"), default="native")
    l.add_argument("--cuda", help="the CUDA toolkit whose ptxas compiled the capture (the fork's image's CUDA 13.3; on a "
                   "host: docker create <image>, docker cp -L <id>:/usr/local/cuda-13.3/. <dir>)")
    a = ap.parse_args()
    return collect(a) if a.cmd == "collect" else lift(a)


if __name__ == "__main__":
    sys.exit(main())
