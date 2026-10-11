#!/usr/bin/env bash
# The serving kernel set of the Zig Flash Next engine: sweep_kernels.py's launches (every row-count and block-count
# specialization) packed together with the fixtures' captures into one set (tools/zig/dsv41/pack_capture.py dedupes
# by Triton's hash and refuses what aot.Set cannot load). Usage: bash capture_serve_kernels.sh <work-dir> <out-set>
# [<fixture work-dir> ...]; the fixture dirs are capture_*.sh work dirs (their out/launches.json and triton/).
# Point the server at the set with TENSORFOLD_CUDA_KERNELS=<out-set>.
set -euo pipefail
NEED_GIB=${NEED_GIB:-80} "$(dirname "$0")/../dsv41/gpu_guard.sh" || exit $?
WORK=$1; OUT=$2; shift 2
TF=${TF:-$HOME/src/TensorFold}
MODEL=${MODEL:-$HOME/models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
IMAGE=${IMAGE:-tensorfold-qwen38fn:0.6.6}
CACHE=${CACHE:-$HOME/.cache/tensorfold-qwen38fn-0.6.6}
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton" "$CACHE/torch_extensions"
docker run --rm --gpus all --ipc=host --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e TRITON_CACHE_DIR=/aot/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -v "$TF:/w:ro" -v "$WORK:/aot" -v "$MODEL:/model:ro" -v "$CACHE:/cache" --entrypoint python3 "$IMAGE" \
  -B /w/tools/zig/qwen38fn/sweep_kernels.py --tools /w/tools/zig --out /aot/out --model /model
args=(--capture "$WORK/out/launches.json" "$WORK/triton" /aot/triton)
for f in "$@"; do args+=(--capture "$f/out/launches.json" "$f/triton" /aot/triton); done
rm -rf "$OUT"
python3 -B "$TF/tools/zig/dsv41/pack_capture.py" --out "$OUT" "${args[@]}"
echo "serving kernel set: $OUT (report.md beside aot.json)"
