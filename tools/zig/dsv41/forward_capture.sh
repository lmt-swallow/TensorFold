#!/usr/bin/env bash
# The layer-loop fixture (dump_forward.py) and its Triton AOT set, from the Python fork's image on a DGX Spark.
# Usage: bash forward_capture.sh <work-dir> [dump_forward.py args...]; needs the checkout at $TF (default
# ~/src/TensorFold), the pack at $MODELS/DeepSeek-V4.1-Flash-EXL3-2.9bpw and the image tensorfold-dsv41:90b68e0-p1.
# Writes <work-dir>/out/fixtures/fwd_common, fwd_n<rows>/ and <work-dir>/aot/{aot.json,cubins/}.
set -euo pipefail
WORK=$1; shift
TF=${TF:-$HOME/src/TensorFold}
MODELS=${MODELS:-$HOME/models}
IMAGE=${IMAGE:-tensorfold-dsv41:90b68e0-p1}
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton"
docker run --rm --gpus all --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/aot/triton \
  -e TF_DS_TRITON_PDL="${TF_DS_TRITON_PDL:-1}" -e TF_DS_L2_PREFETCH=0 -v "$TF:/w:ro" -v "$WORK:/aot" -v "$MODELS:/models:ro" \
  --entrypoint python3 "$IMAGE" -B /w/tools/zig/dsv41/dump_forward.py --tools /w/tools/zig \
  --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw --out /aot/out "$@"
python3 -B "$TF/tools/zig/triton_aot_manifest.py" --cache "$WORK/triton" --launches "$WORK/out/launches.json" \
  --mount /aot/triton --out "$WORK/out/manifest.json"
rm -rf "$WORK/aot"
python3 -B "$TF/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" --cache "$WORK/triton" \
  --jit "$WORK/out/jit.json" --out "$WORK/aot"
echo "captured: $WORK/aot (aot.json + cubins), fixtures: $WORK/out/fixtures"
