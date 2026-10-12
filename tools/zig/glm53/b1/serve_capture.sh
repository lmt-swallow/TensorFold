#!/usr/bin/env bash
# Every Triton variant the Mia fork launches while it serves, on the pair, for the Zig engine's kernel set: the
# recipe's own start.sh (reference.sh's strict setting, KV / PARALLEL / CONTEXT from the caller) with the dsv41
# recorder shim mounted into both ranks, dsv41/capture_workload.py as the client (1..4 concurrent streams: the
# multi-stream seg kernels and the batched verify windows), then each rank's launches packed into one set
# (dsv41/pack_capture.py). Run on the HEAD Spark when neither node serves. Usage:
#   KV=fp8 PARALLEL=4 CONTEXT=1048576 bash serve_capture.sh <work-dir> [capture_workload.py args]
# writes <work-dir>/{shim,recipe,rank0,triton} on the head and the same rank1 / triton dirs on the worker (same
# absolute path), <work-dir>/set (aot.json + cubins + report.md) and <work-dir>/workload.json; the pair is left
# stopped. Merge the set into the served one with dsv41/merge_aot.py.
set -euo pipefail
WORK=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
TF=${TF:-$(cd "$HERE/../../../.." && pwd)}
WORKER=${WORKER:-spark-616b.local}
RECIPE=${RECIPE:-$HOME/src/glm53-mia-recipe-v15}
export KV=${KV:-bf16} PARALLEL=${PARALLEL:-4} CONTEXT=${CONTEXT:-524288}
case "$WORK" in /*) ;; *) echo "serve_capture.sh: an absolute work dir (the same path on both nodes)" >&2; exit 2 ;; esac
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi

mkdir -p "$WORK/rank0" "$WORK/triton"
bash "$TF/tools/zig/dsv41/prepare_shim.sh" "$WORK/shim" >/dev/null
ssh -o BatchMode=yes "$WORKER" "mkdir -p '$WORK/rank1' '$WORK/triton' '$WORK/shim'"
rsync -a "$WORK/shim/" "$WORKER:$WORK/shim/"
# a copy of the recipe whose ranks also mount the shim and record into <work-dir> (the same docker-run arguments
# for both ranks; the containers run as root, so the files they write are root's)
rsync -a "$RECIPE/" "$WORK/recipe/"
python3 - "$WORK" <<'EOF'
import pathlib, sys
work = sys.argv[1]
p = pathlib.Path(work, "recipe", "start.sh")
s = p.read_text()
anchor = "          --ulimit memlock=-1 --ulimit stack=67108864)\n"
if s.count(anchor) != 1:
    raise SystemExit("start.sh: RUN_ARGS not found as expected")
s = s.replace(anchor, anchor + (
    f'RUN_ARGS+=(-v {work}:/aot -e PYTHONPATH=/shim -v {work}/shim:/shim:ro -e TRITON_CACHE_DIR=/aot/triton\n'
    f'          -e TF_DSV41_CAPTURE_DIR=/aot/rankX)\n'
    # concurrent streams without DFlash2 decode without drafts (the server refuses --parallel > 1 otherwise)
    '[[ "$PARALLEL" == 1 ]] || SERVE_ARGS+=(--no-drafts)\n'))
# start.sh serves concurrent streams only with DFlash2 drafts (CC BY-NC-ND, never downloaded here): with DRAFTER=mtp
# and PARALLEL > 1 the ranks run --parallel without a drafter (the MTP head is single-stream only, engine.mtp_head),
# so the concurrent windows are a stream's pending row and its copy drafts
check = '[[ "$DRAFTER" == dflash2 || "$PARALLEL" == 1 ]] || die'
if s.count(check) != 1:
    raise SystemExit("start.sh: the PARALLEL / DRAFTER check not found as expected")
s = s.replace(check, '[[ "$DRAFTER" == dflash2 || "$PARALLEL" == 1 || -n "${CAPTURE:-}" ]] || die')
p.write_text(s)
EOF
# both ranks record into /aot/rankX: a link to rank0 on the head, to rank1 on the worker
ln -s rank0 "$WORK/rankX"
ssh -o BatchMode=yes "$WORKER" "ln -s rank1 '$WORK/rankX'"

cd "$WORK/recipe"
bash "$HERE/reference.sh" check >/dev/null
CAPTURE=1 RECIPE="$WORK/recipe" bash "$HERE/reference.sh" start
bash "$HERE/reference.sh" mem | tee "$WORK/mem.txt"
python3 -B "$TF/tools/zig/dsv41/capture_workload.py" --base "http://127.0.0.1:${PORT:-8888}/v1" --report "$WORK/workload.json" "$@"
touch "$WORK/rank0/dump.now"; ssh -o BatchMode=yes "$WORKER" "touch '$WORK/rank1/dump.now'"
sleep 5
RECIPE="$WORK/recipe" bash "$HERE/reference.sh" stop
rsync -a "$WORKER:$WORK/rank1/" "$WORK/rank1/"
rsync -a "$WORKER:$WORK/triton/" "$WORK/triton1/"
python3 -B "$TF/tools/zig/dsv41/pack_capture.py" --out "$WORK/set" \
  --capture "$WORK/rank0/launches.json" "$WORK/triton" /aot/triton \
  --capture "$WORK/rank1/launches.json" "$WORK/triton1" /aot/triton
echo "set: $WORK/set (report.md); workload: $WORK/workload.json"
