#!/usr/bin/env bash
# The 45-layer forward fixture on the pair (b1/dump_forward.py), run on the HEAD Spark (rank 0) with WORKER (rank 1):
# both ranks in the pinned image, NCCL over the CX7 rails (TF_GLM_COMM=nccl), the pack from each node's Hugging Face
# cache. Usage: bash forward_capture.sh <work-dir> [dump args]; writes <work-dir>/r{0,1}/out on the head and the worker.
# Run only when neither node serves (reference.sh stop; gpu_guard both).
set -euo pipefail
WORK=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
TF=${TF:-$(cd "$HERE/../../../.." && pwd)}
WORKER=${WORKER:-spark-616b.local}   # ssh name; the ranks meet over the CX7 link at MASTER
MASTER=${MASTER:-10.0.1.2}        # the head's address on the link
PORT=${PORT:-29561}
IMAGE=${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:ef83797d791fef96c4605e8d37367aca6de5aeac7bb672792cb682e2e55d4237}
REPO=models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold
REV=078455ffe6472f9a52fbc1139f58b9db2881b25c
RAILS=${RAILS:-rocep1s0f0,roceP2p1s0f0}
DOCK="--rm --gpus all --ipc=host --net=host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --pull=never"
ENVS="-e TF_GLM_COMM=nccl -e NCCL_IB_HCA=$RAILS -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e HF_HUB_OFFLINE=1"
ENVS="$ENVS -e TRITON_CACHE_DIR=/aot/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e HOME=/tmp"
run_rank() { # rank -> the docker command line for it
  echo "mkdir -p $WORK/r$1/out $WORK/r$1/triton \$HOME/.cache/tensorfold-glm53-v1.5/torch_extensions && docker run $DOCK $ENVS --user \$(id -u):\$(id -g) \
 -v \$HOME/src/TensorFold-wt-glm-opus:/w:ro -v $WORK/r$1:/aot -v \$HOME/.cache/huggingface/hub/$REPO:/hf:ro \
 -v \$HOME/.cache/tensorfold-glm53-v1.5:/cache --entrypoint python3 $IMAGE -B /w/tools/zig/glm53/b1/dump_forward.py \
 --tools /w/tools/zig --out /aot/out --model /hf/snapshots/$REV --rank $1 --master $MASTER --port $PORT $*"
}
for w in local worker; do
  if [ $w = local ]; then bash -s < "$TF/tools/zig/dsv41/gpu_guard.sh"; else ssh -o BatchMode=yes "$WORKER" "NEED_GIB=100 bash -s" < "$TF/tools/zig/dsv41/gpu_guard.sh"; fi
done
ssh -o BatchMode=yes "$WORKER" "$(run_rank 1 "$@")" > "$WORK.r1.log" 2>&1 &
w1=$!
bash -c "$(run_rank 0 "$@")" 2>&1 | tee "$WORK.r0.log"
wait $w1 || { echo "rank 1 failed: $WORK.r1.log" >&2; exit 1; }
echo "fixtures: head $WORK/r0/out, worker $WORKER:$WORK/r1/out"
