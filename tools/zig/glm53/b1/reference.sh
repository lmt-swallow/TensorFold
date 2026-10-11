#!/usr/bin/env bash
# The bit-exact reference of GLM-5.3-Flash on the pair: the Mia recipe's own start.sh (v1.5, 83d653f: patches
# 9f73cca659a1, the image pinned by digest) in the strict setting, run on the HEAD Spark (rank 0, the API) with the
# WORKER Spark as rank 1. Usage, on the head:
#   bash reference.sh check   both nodes: no serving container, gpu_guard, the image, the pack's snapshot complete
#   bash reference.sh sync    the head's snapshot (blobs and links) to the worker's cache, the bytes it lacks only,
#                             and this worktree to the worker (forward_capture.sh runs it there)
#   bash reference.sh start   start.sh with DENSE=bf16 KV=bf16 KDA_CHUNKED=0 DRAFTER=mtp PARALLEL=1 CONTEXT=524288
#   bash reference.sh mem     MemAvailable on both nodes (after the load)
#   bash reference.sh stop    the recipe's stop.sh (both ranks); the pair is left stopped
# RECIPE (default ~/src/glm53-mia-recipe-v15, an rsync of the recipe at 83d653f), WORKER (default spark-616b.local,
# the other Spark's ssh name; FABRIC_PEER its CX7 address), WORKER_WEIGHTS (copy: the recipe copies what the worker's
# cache lacks; nfs). VISION (default 0; 1: the recipe's default, rank 0 also runs the image tower).
set -euo pipefail
RECIPE=${RECIPE:-$HOME/src/glm53-mia-recipe-v15}
export WORKER=${WORKER:-spark-616b.local}      # the ssh target (key-based from the head)
export FABRIC_PEER=${FABRIC_PEER:-10.0.1.1}     # the worker's address on the CX7 link (the ranks' rendezvous route)
export WORKER_WEIGHTS=${WORKER_WEIGHTS:-copy}
HERE=$(cd "$(dirname "$0")" && pwd)
TF=${TF:-$(cd "$HERE/../../../.." && pwd)}
REPO=models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold
REV=078455ffe6472f9a52fbc1139f58b9db2881b25c
DIGEST=sha256:ef83797d791fef96c4605e8d37367aca6de5aeac7bb672792cb682e2e55d4237

# the strict setting (design-glm53-native: the oracle) and what start.sh needs to serve it without the network
export DENSE=bf16 KV=bf16 KDA_CHUNKED=0 DRAFTER=mtp PARALLEL=1 CONTEXT=524288 VISION=${VISION:-0}
export PULL=0 HF_HUB_OFFLINE=1

node() { # where, command
  if [[ "$1" == head ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$WORKER" "$2"; fi
}

case "${1:-}" in
  check)
    for w in head worker; do
      echo "== $w"
      node $w "docker ps --format '{{.Names}} {{.Image}}'"
      node $w "NEED_GIB=100 bash -s" < "$TF/tools/zig/dsv41/gpu_guard.sh" && echo "gpu_guard ok"
      node $w "docker images --digests --format '{{.Repository}}:{{.Tag}} {{.Digest}} {{.ID}}' | grep -iE 'glm-5.3-flash-exl3-2x|tensorfold-glm53' || true"
      node $w "s=\$HOME/.cache/huggingface/hub/$REPO/snapshots/$REV; ls \$s | wc -l; du -sLh \$s 2>/dev/null | cut -f1; free -g | sed -n 2p; df -h \$HOME | tail -1"
    done
    echo "(the head's image must carry digest $DIGEST; the snapshot holds 94 entries, 164 GB)"
    ;;
  sync)
    ssh -o BatchMode=yes "$WORKER" "mkdir -p \$HOME/.cache/huggingface/hub/$REPO"
    rsync -a --partial "$HOME/.cache/huggingface/hub/$REPO/" "$WORKER:.cache/huggingface/hub/$REPO/"
    node worker "s=\$HOME/.cache/huggingface/hub/$REPO/snapshots/$REV; ls \$s | wc -l; du -sLh \$s | cut -f1"
    rsync -a --exclude .git --exclude zig-out --exclude .zig-cache "$TF/" "$WORKER:src/TensorFold-wt-glm-opus/"
    ;;
  start)
    cd "$RECIPE"
    git -C "$RECIPE" rev-parse --short HEAD 2>/dev/null || true
    ./start.sh
    ;;
  mem)
    for w in head worker; do echo "$w: $(node $w "awk '/MemAvailable/ {printf \"%.1f GiB\", \$2 / 1048576}' /proc/meminfo")"; done
    ;;
  stop)
    cd "$RECIPE" && ./stop.sh
    ;;
  *)
    sed -n 2,13p "$0"
    exit 2
    ;;
esac
