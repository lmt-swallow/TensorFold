#!/usr/bin/env bash
# The concurrent-decode fixture (dump_lanes_parallel.py: two admissions into a 4-slot pool, a batched draft, one round
# over both streams' windows, the keeps and absorbs, the next batched draft) and its Triton AOT set, from the Python
# fork's image on a DGX Spark.
# Usage: bash lanes_parallel_capture.sh <work-dir> [dump_lanes_parallel.py args...]; needs the checkout at $TF
# (default ~/src/TensorFold), the pack at $MODELS/DeepSeek-V4.1-Flash-EXL3-2.9bpw and the image
# tensorfold-dsv41:90b68e0-p1. Writes <work-dir>/out/fixtures/parallel and <work-dir>/aot/{aot.json,cubins/};
# `tf-cuda-test dsv41-lanes-parallel` reads them.
set -euo pipefail
"$(dirname "$0")/gpu_guard.sh" || exit $?   # never beside a serving rank: the node swaps into a hang
WORK=$1; shift
TF=${TF:-$HOME/src/TensorFold}
MODELS=${MODELS:-$HOME/models}
IMAGE=${IMAGE:-tensorfold-dsv41:90b68e0-p1}
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton"
docker run --rm --gpus all --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/aot/triton \
  -e TF_DS_TRITON_PDL="${TF_DS_TRITON_PDL:-1}" -e TF_DS_HC_FUSED=0 -e TF_DS_ROT_Q=0 -e TF_DS_ROT_WOB=0 -e TF_DS_ROT_ATTN=0 \
  -e TF_DS_L2_PREFETCH=0 -e TF_DS_MARKOV_CACHE=0 -e TF_DS_MARKOV_SPLIT=1 -e TF_DS_DRAFT_FUSED=0 -e TF_DS_RANK_CACHE= \
  -e TF_DS_REPLAY=0 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -v "$TF:/w:ro" -v "$WORK:/aot" -v "$MODELS:/models:ro" \
  --entrypoint python3 "$IMAGE" -B /w/tools/zig/dsv41/dump_lanes_parallel.py --tools /w/tools/zig \
  --model /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw --out /aot/out "$@"
python3 -B "$TF/tools/zig/triton_aot_manifest.py" --cache "$WORK/triton" --launches "$WORK/out/launches.json" \
  --mount /aot/triton --out "$WORK/out/manifest.json"
rm -rf "$WORK/aot"
python3 -B "$TF/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" --cache "$WORK/triton" \
  --jit "$WORK/out/jit.json" --out "$WORK/aot"
echo "captured: $WORK/aot (aot.json + cubins), fixtures: $WORK/out/fixtures"
