# Capturing every DeepSeek-V4.1 Triton variant from a real two-rank serving run

The Zig engine replays the Python fork's Triton kernels from captured cubins (`cuda.aot.Set`), so the kernel set
must hold every specialization a served run launches: prompt chunks at every tail size, decode windows of 1..6
rows, concurrent rounds up to 16 rows, the batched drafter at 1..4 streams, the Engram gate, the DSpark stages,
the Markov head, and both replay modes. Only `tensorfold serve` under a real workload launches all of them, so
the capture is a serving run with a recorder shim, a client workload, and a merge.

Files (tools/zig/dsv41/): `shim/usercustomize.py` (the recorder, installed by `PYTHONPATH`), `prepare_shim.sh`,
`capture_workload.py` (the client), `pack_capture.py` (merge + report), plus `capture_kernel.py` / `capture.sh`
for one-kernel captures and `probe_kernels.py` for a compile-only scan.

## 1. The shim: record launches without editing the fork

Python imports `usercustomize` at startup from any `PYTHONPATH` entry (the image keeps Debian's own
`sitecustomize`, which stays untouched). With `TF_DSV41_CAPTURE_DIR` set, the shim installs
`triton_aot_manifest.Recorder` (it patches `triton.runtime.jit.JITFunction.run`, so launches inside CUDA graph
captures are recorded too) and writes `launches.json`, `jit.json` (every JIT function's `do_not_specialize`) and
`env.json` into that directory: every 60 s (`TF_DSV41_DUMP_EVERY`), whenever a file named `dump.now` appears in
it (`touch ~/aot/serve-capture/out/rank0/dump.now`; the shim removes it after dumping), and at exit. Once torch
has initialized CUDA the process no longer sees Python-level SIGUSR1 (verified: a plain handler never fires after
`import torch` + a kernel launch), so the file trigger is the on-demand path. The server turns SIGTERM into a
KeyboardInterrupt and exits normally, so `docker stop` ends with the exit dump; touch `dump.now` first anyway, so
a hung shutdown still leaves a complete file.

```
bash tools/zig/dsv41/prepare_shim.sh ~/aot/shim        # usercustomize.py + triton_aot_manifest.py
```

Each rank gets its own capture directory and its own Triton cache (the cubins the manifest maps launches to):

```
# added to every rank's docker run (both ranks: rank 1 drafts and verifies the same windows and mirrors the
# head on its own vocab half; constexprs and specializations are expected to match, the merge proves it)
-v ~/aot/shim:/shim:ro -v ~/aot/serve-capture:/aot \
-e PYTHONPATH=/shim -e TF_DSV41_CAPTURE_DIR=/aot/out/rank$R -e TRITON_CACHE_DIR=/aot/triton$R
```

`~/aot/serve-capture` must be writable by the container's user (`--user "$(id -u):$(id -g)" -e HOME=/tmp` keeps the
files owned by you; without it they come out root-owned). Start rank 0 first, rank 1 after the store port listens,
as in scratchpad/home_start_dsv41.sh. Keep every other flag as the production run (`--parallel 4`, `--mtp-drafts 5`,
`--context`, `TF_DS_REPLAY`, `TF_DS_PREFILL_CHUNK=2048`, `TF_DS_ENGRAM`): the variants depend on them.

A fresh `TRITON_CACHE_DIR` is best: the manifest maps each launched hash to its cubin by the kernel's
`metadata_group` paths, which must lie under the mount.

## 2. The workload

```
python3 -B tools/zig/dsv41/capture_workload.py --base http://127.0.0.1:8000/v1 --max-tokens 64 --rounds 2
```

Phase 1 sends each prompt length alone — 1, 17, 33, 131, 514, 1024, 2113, 4096, 9000 tokens (the warm-up lengths,
the 512 and 2048 chunk boundaries and their tails), greedy and temperature 1, thinking off and on — exact lengths
from the server's `/v1/tokenize` rendering of the chat template. Phase 2 runs 1, 2, 3 and 4 concurrent streams of
mixed lengths, several rounds, with mixed `max_tokens` so streams finish at different times: rounds of every width
and the batched drafter at every stream count. `--no-thinking-only` halves it. The run's requests and failures go
to `workload.json`.

What the workload cannot switch: `TF_DS_REPLAY` (CED bounded replay vs full prefill) and `TF_DS_PREFILL_CHUNK` are
server flags, so capture one serving run per setting you intend to ship (e.g. `TF_DS_REPLAY=1` and `0`) into
separate capture dirs; `pack_capture.py` merges them. The Engram gate kernels appear only with the tables present
(`TF_DS_ENGRAM`); the vision kernels only with `--vision` (not needed).

## 3. After the run: dump, merge, report

```
touch ~/aot/serve-capture/out/rank0/dump.now; ssh 10.0.1.2 touch ~/aot/serve-capture/out/rank1/dump.now  # final dumps
sleep 5   # the shim polls once a second; shim.log in each dir shows "dump #N (dump.now)"
python3 -B tools/zig/dsv41/pack_capture.py --out ~/aot/dsv41-set \
    --capture ~/aot/serve-capture/out/rank0/launches.json ~/aot/serve-capture/triton0 /aot/triton0 \
    --capture ~/aot/serve-capture/out/rank1/launches.json ~/aot/serve-capture/triton1 /aot/triton1 \
    [--capture <another run's launches.json> <its cache> <its mount>]...
```

The merge builds each capture's manifest (`tools/zig/triton_aot_manifest.py`), deduplicates variants by Triton's
specialization hash, copies the cubins beside one `aot.json` in `aot_pack.py`'s format, and writes `report.md` /
`report.json`: kernel name x variant count x launches, the constexpr values seen, and every variant
`aot.Set.load` would refuse (global or profile scratch, `num_ctas` > 1, a PTX entry whose parameters are not the
runtime argument list, a function missing from `jit.json`). The exit status is 1 when anything was refused.
Point the engine at the set with `TENSORFOLD_CUDA_KERNELS=~/aot/dsv41-set`.

## Validated so far (2026-10-09, spark-616b)

- Shim: `probe_kernels.py` run with `PYTHONPATH=/shim TF_DSV41_CAPTURE_DIR=/aot/out/rank0` recorded 11 kernels
  with periodic and exit dumps and a `jit.json` of 111 JIT functions; `dump.now` dumps on demand (SIGUSR1 does
  not reach Python once CUDA is up).
- Merge: three captures (capture_kernel.py's rmsnorm + kv_norm_rope and two shim runs) packed into one set of 13
  variants, nothing refused; the merged set loads in `aot.Set` and `tf-cuda-test dsv41-aot-rmsnorm` /
  `dsv41-aot-kv-norm-rope` replay bit-exact from it.
- Not yet run: the serving capture itself (needs the pair free of other GPU jobs and the Engram tables in place).
