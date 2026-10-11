#!/usr/bin/env bash
# A GLM-5.3-Flash fixture (dump_<name>.py) from the Mia fork's pinned image on ONE DGX Spark, with its Triton
# launches packed for the Zig engine and its CUDA extensions kept for the SASS comparison. Usage:
#   bash capture.sh <kda|dsa|moe|hc|prompt|seg> <work-dir> [dump args...]
# needs the TensorFold checkout at $TF (default: this script's tree), the pack in the Hugging Face cache ($HF_REPO,
# default ~/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold, its one snapshot) and the
# image $IMAGE (the recipe's v1.5 pin, present on the node: never pulled). Writes <work-dir>/out/fixture
# (manifest.json + arrays), weights.json, launches.json, capture.json, <work-dir>/aot/{aot.json,cubins/} and
# <work-dir>/extensions/ (the torch extensions the run built: kda, exl3, hc, experts, qmm). Host memory: the dumps
# need 6 (hc) to 20 (dsa, its 600k-token caches) GiB; NEED_GIB raises the guard's bar. KV=fp8: the dsa and prompt
# dumps on FP8 DSA caches (ref.py TF_GLM53_REF_KV, GLM-5.3-Flash-2's reference); default bf16.
set -euo pipefail
NAME=$1; shift
WORK=$1; shift
case "$NAME" in kda|dsa|moe|hc|prompt|seg) ;; *) echo "capture.sh: kda, dsa, moe, hc, prompt or seg, not $NAME" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
TF=${TF:-$(cd "$HERE/../../.." && pwd)}
NEED_GIB=${NEED_GIB:-40} bash "$TF/tools/zig/dsv41/gpu_guard.sh" || exit $?   # never beside a serving model
IMAGE=${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:ef83797d791fef96c4605e8d37367aca6de5aeac7bb672792cb682e2e55d4237}
HF_REPO=${HF_REPO:-$HOME/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold}
REV=${REV:-$(ls "$HF_REPO/snapshots" | head -1)}
CACHE=${CACHE:-$HOME/.cache/tensorfold-glm53-v1.5}   # the extensions' builds, kept between runs
docker image inspect "$IMAGE" >/dev/null || { echo "capture.sh: $IMAGE is not on this node (never pulled here)" >&2; exit 2; }
if [ -e "$WORK" ]; then echo "$WORK exists; pass a fresh work dir" >&2; exit 1; fi
mkdir -p "$WORK/out" "$WORK/triton" "$CACHE/torch_extensions"
# the snapshot's files link into ../../blobs: mount the repo folder whole
docker run --rm --pull=never --gpus all --ipc=host --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e TF_GLM53_REF_KV="${KV:-bf16}" -e TRITON_CACHE_DIR=/aot/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TF_GLM53_IMAGE_PATCHES="$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")" \
  -v "$TF:/w:ro" -v "$WORK:/aot" -v "$HF_REPO:/hf:ro" -v "$CACHE:/cache" --entrypoint python3 "$IMAGE" \
  -B "/w/tools/zig/glm53/dump_$NAME.py" --tools /w/tools/zig --out /aot/out --model "/hf/snapshots/$REV" "$@"
python3 -B "$TF/tools/zig/triton_aot_manifest.py" --cache "$WORK/triton" --launches "$WORK/out/launches.json" \
  --mount /aot/triton --out "$WORK/out/manifest.json"
rm -rf "$WORK/aot"
python3 -B "$TF/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" --cache "$WORK/triton" \
  --jit "$WORK/out/jit.json" --out "$WORK/aot"
mkdir -p "$WORK/extensions"
find "$CACHE/torch_extensions" -name '*.so' -exec cp {} "$WORK/extensions/" \;
find "$CACHE/torch_extensions" -name '*.so' -exec sha256sum {} + > "$WORK/extensions/sha256.txt"
echo "captured: $WORK/aot (aot.json + cubins), fixture: $WORK/out/fixture, extensions: $WORK/extensions"
