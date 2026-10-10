#!/usr/bin/env bash
# Capture DeepSeek-V4.1 Triton kernels from the Python fork's image on a DGX Spark, then pack them for the Zig engine.
# Usage: bash capture.sh <work-dir> <kernel>... ; needs the TensorFold checkout at $TF (default ~/src/TensorFold)
# and the image tensorfold-dsv41:90b68e0-p1. Writes <work-dir>/out/{launches,jit,manifest,capture}.json,
# <work-dir>/out/fixtures/<kernel>/ (replay fixtures) and <work-dir>/aot/{aot.json,cubins/}.
set -euo pipefail
"$(dirname "$0")/gpu_guard.sh" || exit $?   # never beside a serving rank: the node swaps into a hang
WORK=$1; shift
TF=${TF:-$HOME/src/TensorFold}
IMAGE=${IMAGE:-tensorfold-dsv41:90b68e0-p1}
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton"
docker run --rm --gpus all --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/aot/triton -e TF_DS_TRITON_PDL="${TF_DS_TRITON_PDL:-1}" \
  -v "$TF:/w:ro" -v "$WORK:/aot" --entrypoint python3 "$IMAGE" \
  -B /w/tools/zig/dsv41/capture_kernel.py --tools /w/tools/zig --out /aot/out "$@"
python3 -B "$TF/tools/zig/triton_aot_manifest.py" --cache "$WORK/triton" --launches "$WORK/out/launches.json" \
  --mount /aot/triton --out "$WORK/out/manifest.json"
rm -rf "$WORK/aot"
python3 -B "$TF/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" --cache "$WORK/triton" \
  --jit "$WORK/out/jit.json" --out "$WORK/aot"
echo "captured: $WORK/aot (aot.json + cubins), fixtures: $WORK/out/fixtures"
