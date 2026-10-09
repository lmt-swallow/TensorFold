#!/usr/bin/env bash
# A concurrent round's fixture (dump_rounds.py) from the Python fork's image on a DGX Spark, with its Triton launches
# packed for the Zig engine. Usage: bash capture_rounds.sh <work-dir> [--layers N] [--engram /models/<engram-dir>
# --token-map /tm/token_map.json]; needs the checkout at $TF, the pack at $MODEL and the image
# tensorfold-dsv41:90b68e0-p1 (UNFUSED=1: the unfused switches); $MODELS is mounted at /models and $TOKEN_MAP_DIR at /tm for Engram.
# `tf-cuda-test dsv41-rounds <work-dir> <rank-cache-dir> <model-dir>` reads <work-dir>/out/fixture and <work-dir>/aot.
set -euo pipefail
WORK=$1; shift
TF=${TF:-$HOME/src/TensorFold}
MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw}
IMAGE=${IMAGE:-tensorfold-dsv41:90b68e0-p1}
MODELS=${MODELS:-$HOME/models}
TOKEN_MAP_DIR=${TOKEN_MAP_DIR:-$HOME/aot/dsv41-all}
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton"
# the fork's default switches (its served decode path); UNFUSED=1: the plain loop the fork documents as the same bits
SWITCHES=()
if [ "${UNFUSED:-0}" = 1 ]; then SWITCHES=(-e TF_DS_HC_FUSED=0 -e TF_DS_ROT_Q=0 -e TF_DS_ROT_WOB=0 -e TF_DS_ROT_ATTN=0); fi
docker run --rm --gpus all --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/aot/triton \
  -e TF_DS_TRITON_PDL="${TF_DS_TRITON_PDL:-1}" "${SWITCHES[@]}" -e TF_DS_L2_PREFETCH=0 -e TF_DS_RANK_CACHE= -v "$TF:/w:ro" -v "$WORK:/aot" -v "$MODEL:/model:ro" \
  -v "$MODELS:/models:ro" -v "$TOKEN_MAP_DIR:/tm:ro" --entrypoint python3 "$IMAGE" \
  -B /w/tools/zig/dsv41/dump_rounds.py --tools /w/tools/zig --out /aot/out --model /model "$@"
python3 -B "$TF/tools/zig/triton_aot_manifest.py" --cache "$WORK/triton" --launches "$WORK/out/launches.json" \
  --mount /aot/triton --out "$WORK/out/manifest.json"
rm -rf "$WORK/aot"
python3 -B "$TF/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" --cache "$WORK/triton" \
  --jit "$WORK/out/jit.json" --out "$WORK/aot"
echo "captured: $WORK/aot (aot.json + cubins), fixture: $WORK/out/fixture"
