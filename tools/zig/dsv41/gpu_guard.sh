#!/usr/bin/env bash
# Refuse GPU work on a DGX Spark that is already serving or testing. Host and GPU share one unified memory: a second
# model beside a serving rank (~106 GiB) does not fail an allocation, it swaps the host until nothing (sshd included)
# answers, and a two-rank server takes the other node down with it. Run it on the node before any `docker run --gpus`:
#
#   bash gpu_guard.sh            exit 0 when no compute process holds the GPU, no serving container is up and
#                                MemAvailable >= NEED_GIB (40); otherwise print why and exit 2
#   ssh node bash -s < gpu_guard.sh   the same from another machine
#
# NEED_GIB sets the memory bar (a 25-layer fixture test needs ~70, a 4-layer one ~15); FORCE=1 skips every check.
set -u
NEED_GIB=${NEED_GIB:-40}
[ "${FORCE:-0}" = 1 ] && exit 0

apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
if [ -n "$apps" ]; then
  echo "gpu_guard: compute processes hold the GPU on $(hostname):" >&2
  echo "$apps" >&2
  exit 2
fi
up=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(tfz[01]|dsv41-(master|worker))$' || true)
if [ -n "$up" ]; then
  echo "gpu_guard: serving containers up on $(hostname): $(echo "$up" | tr '\n' ' ')(stop them first: serve_smoke.sh stop)" >&2
  exit 2
fi
avail=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if [ "$avail" -lt "$NEED_GIB" ]; then
  echo "gpu_guard: $avail GiB available on $(hostname), $NEED_GIB needed (NEED_GIB)" >&2
  exit 2
fi
