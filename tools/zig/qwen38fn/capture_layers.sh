#!/usr/bin/env bash
# The whole-forward fixture (dump_layers.py: layer boundaries, traced layers) from TensorFold 0.6.6 on a DGX Spark, with its Triton launches packed for the
# Zig engine. Usage: bash capture_layers.sh <work-dir>; needs the TensorFold checkout at $TF (default ~/src/TensorFold),
# the pack at $MODEL (default ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5) and the image $IMAGE
# (tensorfold-qwen38fn:0.6.6, image.Dockerfile). Writes <work-dir>/out/fixture (manifest.json + arrays),
# <work-dir>/out/launches.json and <work-dir>/aot/{aot.json,cubins/}; `tf-cuda-test qwen38fn-layers <work-dir> <model>`
# reads them.
set -euo pipefail
NEED_GIB=${NEED_GIB:-80} "$(dirname "$0")/../dsv41/gpu_guard.sh" || exit $?   # never beside a serving model
WORK=$1; shift
TF=${TF:-$HOME/src/TensorFold}
MODEL=${MODEL:-$HOME/models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
IMAGE=${IMAGE:-tensorfold-qwen38fn:0.6.6}
CACHE=${CACHE:-$HOME/.cache/tensorfold-qwen38fn-0.6.6}   # the extensions' builds, kept between runs
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton" "$CACHE/torch_extensions"
docker run --rm --gpus all --ipc=host --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e TRITON_CACHE_DIR=/aot/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -v "$TF:/w:ro" -v "$WORK:/aot" -v "$MODEL:/model:ro" -v "$CACHE:/cache" --entrypoint python3 "$IMAGE" \
  -B /w/tools/zig/qwen38fn/dump_layers.py --tools /w/tools/zig --out /aot/out --model /model "$@"
python3 -B "$TF/tools/zig/triton_aot_manifest.py" --cache "$WORK/triton" --launches "$WORK/out/launches.json" \
  --mount /aot/triton --out "$WORK/out/manifest.json"
rm -rf "$WORK/aot"
python3 -B "$TF/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" --cache "$WORK/triton" \
  --jit "$WORK/out/jit.json" --out "$WORK/aot"
echo "captured: $WORK/aot (aot.json + cubins), fixture: $WORK/out/fixture"
