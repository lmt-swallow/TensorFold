"""Records every Triton launch of a Python-engine process (``tensorfold serve`` on either rank) for the Zig engine's
AOT kernel set, without editing the engine: Python imports ``usercustomize`` at startup from any ``PYTHONPATH``
entry, so this file goes into a directory mounted into the container and named there. It leaves the process
alone unless ``TF_DSV41_CAPTURE_DIR`` is set.

Environment:
  TF_DSV41_CAPTURE_DIR   where launches.json, jit.json and shim.log go (one directory per rank)
  TF_DSV41_TOOLS         directory holding triton_aot_manifest.py (default: this file's directory)
  TF_DSV41_DUMP_EVERY    seconds between periodic dumps (default 60; 0 disables); a dump also runs at exit, when
                         a file named dump.now appears in the capture dir (`touch`; the reliable on-demand way),
                         and on SIGUSR1 where the process still sees Python-level signals (torch's CUDA runtime
                         takes over SIGUSR1 delivery once initialized, so the file trigger is the one to use)
TRITON_CACHE_DIR must be set per rank by the caller: the cubins the manifest maps the launches to live there.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

_dir = os.environ.get("TF_DSV41_CAPTURE_DIR")


def _log(msg: str) -> None:
    try:
        with open(Path(_dir) / "shim.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()} {msg}\n")
    except OSError:
        pass


def _specialization() -> dict:
    """Each JIT function's parameter names and which ones Triton never specializes (aot_pack.py's jit.json)."""

    import gc

    from triton.runtime.jit import JITFunction

    out = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            name = f"{fn.fn.__module__}.{fn.fn.__qualname__}"
            out[name] = {"params": [p.name for p in fn.params],
                         "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                         "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return out


def _install() -> None:
    out = Path(_dir)
    out.mkdir(parents=True, exist_ok=True)
    tools = Path(os.environ.get("TF_DSV41_TOOLS") or Path(__file__).resolve().parent)
    spec = importlib.util.spec_from_file_location("triton_aot_manifest", tools / "triton_aot_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rec = mod.Recorder().install()       # patches triton.runtime.jit.JITFunction.run before any kernel runs
    rec.phase = os.environ.get("TF_DSV41_PHASE", "serve")
    lock = threading.Lock()
    state = {"dumps": 0}

    def dump(why: str) -> None:
        with lock:
            try:
                rec.dump(out / "launches.json")
                (out / "jit.json").write_text(json.dumps(_specialization(), indent=1, sort_keys=True) + "\n")
                (out / "env.json").write_text(json.dumps({k: v for k, v in os.environ.items()
                                                          if k.startswith(("TF_", "NCCL_", "TRITON_"))}, indent=1))
                state["dumps"] += 1
                _log(f"dump #{state['dumps']} ({why}): {len(rec.kernels)} kernels, {sum(rec.counts.values())} launches")
            except Exception as e:  # noqa: BLE001
                _log(f"dump failed ({why}): {e!r}")

    atexit.register(dump, "exit")

    def on_usr1(signum, frame):  # noqa: ARG001
        dump("SIGUSR1")

    signal.signal(signal.SIGUSR1, on_usr1)
    every = float(os.environ.get("TF_DSV41_DUMP_EVERY", "60") or 0)
    trigger = out / "dump.now"

    def watcher() -> None:
        """Every second: a dump when `dump.now` appears in the capture dir (removed after; works from any shell:
        `touch <dir>/dump.now`), and a periodic dump every `every` seconds when anything was launched."""

        last = time.monotonic()
        while True:
            time.sleep(1.0)
            if trigger.exists():
                try:
                    trigger.unlink()
                except OSError:
                    pass
                dump("dump.now")
            if every > 0 and rec.kernels and time.monotonic() - last >= every:
                last = time.monotonic()
                dump("periodic")

    threading.Thread(target=watcher, name="dsv41-capture-dump", daemon=True).start()
    sys.modules["tf_dsv41_capture"] = type(sys)("tf_dsv41_capture")
    sys.modules["tf_dsv41_capture"].recorder = rec
    sys.modules["tf_dsv41_capture"].dump = dump
    _log(f"recorder installed; TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}; argv={sys.argv}")


if _dir:
    try:
        _install()
    except Exception as e:  # noqa: BLE001
        _log(f"install failed: {e!r}")
        sys.stderr.write(f"[dsv41 capture shim] install failed: {e!r}\n")
