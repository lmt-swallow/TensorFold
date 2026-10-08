#!/usr/bin/env bash
# The Engram fixture (dump_engram.py) from the Python fork's image on a DGX Spark, with the gate's Triton launch packed
# for the Zig engine. Usage: bash capture_engram.sh <work-dir> [--layer N] [--n N]; needs the checkout at $TF (default
# ~/src/TensorFold), the pack at $MODEL, the Engram shards at $ENGRAM (default ~/models/DeepSeek-V4.1-Flash-Engram) and
# the image tensorfold-dsv41:90b68e0-p1. Writes <work-dir>/out/fixture, <work-dir>/out/launches.json and
# <work-dir>/aot/; `tf-cuda-test dsv41-engram <work-dir> <engram-dir>` reads them.
set -euo pipefail
WORK=$1; shift
TF=${TF:-$HOME/src/TensorFold}
MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw}
ENGRAM=${ENGRAM:-$HOME/models/DeepSeek-V4.1-Flash-Engram}
IMAGE=${IMAGE:-tensorfold-dsv41:90b68e0-p1}
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton"
docker run --rm --gpus all --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/aot/triton -e TF_DS_TRITON_PDL="${TF_DS_TRITON_PDL:-1}" \
  -v "$TF:/w:ro" -v "$WORK:/aot" -v "$MODEL:/model:ro" -v "$ENGRAM:/engram:ro" --entrypoint python3 "$IMAGE" \
  -B /w/tools/zig/dsv41/dump_engram.py --tools /w/tools/zig --out /aot/out --model /model --engram /engram "$@"
python3 -B "$TF/tools/zig/triton_aot_manifest.py" --cache "$WORK/triton" --launches "$WORK/out/launches.json" \
  --mount /aot/triton --out "$WORK/out/manifest.json"
rm -rf "$WORK/aot"
python3 -B "$TF/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" --cache "$WORK/triton" \
  --jit "$WORK/out/jit.json" --out "$WORK/aot"
echo "captured: $WORK/aot (aot.json + cubins), fixture: $WORK/out/fixture"
