#!/usr/bin/env bash
# The DSpark drafter fixture (dump_dspark.py) from the Python fork's image on a DGX Spark, with its Triton launches
# packed for the Zig engine. Usage: bash capture_dspark.sh <work-dir> [--stages N] [--n-abs N]; needs the checkout at
# $TF (default ~/src/TensorFold), the pack at $MODEL and the image tensorfold-dsv41:90b68e0-p1. Writes
# <work-dir>/out/fixture (several GB: the stages' experts), <work-dir>/out/launches.json and <work-dir>/aot/;
# `tf-cuda-test dsv41-dspark <work-dir>` reads them. Needs ~9 GiB of GPU memory at 3 stages (~4.5 at 1).
set -euo pipefail
"$(dirname "$0")/gpu_guard.sh" || exit $?   # never beside a serving rank: the node swaps into a hang
WORK=$1; shift
TF=${TF:-$HOME/src/TensorFold}
MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw}
IMAGE=${IMAGE:-tensorfold-dsv41:90b68e0-p1}
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton"
docker run --rm --gpus all --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/aot/triton \
  -e TF_DS_TRITON_PDL="${TF_DS_TRITON_PDL:-1}" -e TF_DS_MARKOV_CACHE=0 -e TF_DS_RANK_CACHE= \
  -v "$TF:/w:ro" -v "$WORK:/aot" -v "$MODEL:/model:ro" --entrypoint python3 "$IMAGE" \
  -B /w/tools/zig/dsv41/dump_dspark.py --tools /w/tools/zig --out /aot/out --model /model "$@"
python3 -B "$TF/tools/zig/triton_aot_manifest.py" --cache "$WORK/triton" --launches "$WORK/out/launches.json" \
  --mount /aot/triton --out "$WORK/out/manifest.json"
rm -rf "$WORK/aot"
python3 -B "$TF/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" --cache "$WORK/triton" \
  --jit "$WORK/out/jit.json" --out "$WORK/aot"
echo "captured: $WORK/aot (aot.json + cubins), fixture: $WORK/out/fixture"
