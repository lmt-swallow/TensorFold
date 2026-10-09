#!/usr/bin/env bash
# The two-rank smoke of serve_smoke.md: tensorfold-native serving DeepSeek-V4.1-Flash, rank 0 (HTTP) on $R0 and rank 1
# (the follower) on $R1, each in nvcr.io/nvidia/pytorch:26.07-py3 as containers tfz0 / tfz1. Run it from any machine
# that reaches both nodes over ssh (the Mac, or rank 0 itself).
#
#   bash serve_smoke.sh [run]   copy the binary, start rank 0, wait for its master port, start rank 1, wait for
#                               "serving", send one fixed greedy chat request, print the reply's token ids and tok/s,
#                               then remove both containers (KEEP=1 leaves them serving)
#   bash serve_smoke.sh stop    remove both containers
#
# Environment: R0/R1 (ssh names, default spark-616b/spark-9e43), MASTER (rank 0's address on the NCCL link, 10.0.1.1),
# BUILD_HOST and BUILD_BIN (where the built binary is: spark-9e43, ~/src/TensorFold/zig-out/native/bin/tensorfold-native),
# CONTEXT (16384), MAX_TOKENS (64), ENGRAM (the Engram tables under ~/models, DeepSeek-V4.1-Flash-Engram; empty: run
# without them, degraded; the token map is ~/aot/$KERNELS/token_map.json on both nodes), LOG_DIR (where both ranks' logs are kept), FORCE=1 (start even when the GPU
# already has compute processes), NO_COPY=1 (use the ~/bin/tensorfold-native already on each node).
set -euo pipefail

R0=${R0:-spark-616b}
R1=${R1:-spark-9e43}
MASTER=${MASTER:-10.0.1.1}
MASTER_PORT=${MASTER_PORT:-29551}
HTTP_PORT=${HTTP_PORT:-8000}
BUILD_HOST=${BUILD_HOST:-spark-9e43}
BUILD_BIN=${BUILD_BIN:-src/TensorFold/zig-out/native/bin/tensorfold-native}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
MODEL=${MODEL:-DeepSeek-V4.1-Flash-EXL3-2.9bpw}
CONTEXT=${CONTEXT:-16384}
MAX_TOKENS=${MAX_TOKENS:-64}
ENGRAM=${ENGRAM-DeepSeek-V4.1-Flash-Engram}
KERNELS=${KERNELS:-dsv41-all}   # the captured Triton set under ~/aot (aot.json, cubins, rope/, token_map.json)
DRAFTS=${DRAFTS:-1}            # 0: --no-drafts
PARALLEL=${PARALLEL:-1}        # streams decoded together (--parallel)
RESERVE=${RESERVE:-4}          # TENSORFOLD_MEMORY_RESERVE_GIB
PORT_TIMEOUT=${PORT_TIMEOUT:-600}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
LOG_DIR=${LOG_DIR:-${TMPDIR:-/tmp}/dsv41-smoke-$(date +%Y%m%d-%H%M%S)}
PROMPT=${PROMPT:-Write a Python function that parses an HTTP request line and returns method, path, version.}

SSH=(ssh -4 -o BatchMode=yes -o ConnectTimeout=10)
on() { local host=$1; shift; "${SSH[@]}" "$host" "$@"; }
say() { echo "[smoke $(date +%H:%M:%S)] $*" >&2; }
die() { say "error: $*"; exit 1; }

# RoPE rows for a context (cuda_lanes.ropeRows): context + 24 extra positions, up to a multiple of 4096.
rope_rows() { local need=$(($1 + 24)); local rows=$(((need + 4095) / 4096 * 4096)); echo $((rows < 4096 ? 4096 : rows)); }

stop_all() {
  for pair in "$R0 tfz0" "$R1 tfz1"; do
    set -- $pair
    on "$1" "docker rm -f $2 >/dev/null 2>&1 || true" || say "cannot reach $1 to remove $2"
  done
}

if [ "${1:-run}" = stop ]; then
  stop_all
  say "removed tfz0 on $R0 and tfz1 on $R1"
  exit 0
fi
[ "${1:-run}" = run ] || die "usage: $0 [run|stop]"
command -v python3 >/dev/null || die "python3 is needed here to read the replies"

ROWS=$(rope_rows "$CONTEXT")
mkdir -p "$LOG_DIR"
LOG_PIDS=()

cleanup() {
  local code=$?
  if [ "${KEEP:-0}" = 1 ] && [ "$code" = 0 ]; then
    say "KEEP=1: tfz0 and tfz1 stay up; '$0 stop' removes them"
  else
    stop_all
  fi
  for p in "${LOG_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  say "logs: $LOG_DIR/tfz0.log, $LOG_DIR/tfz1.log"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# -- preflight: reachability, the GPU free, each node's inputs ------------------------------------------------------
for h in "$R0" "$R1"; do on "$h" true || die "cannot ssh to $h"; done
for h in "$R0" "$R1"; do
  others=$(on "$h" "docker ps --format '{{.Names}}' | grep -E '^(dsv41-master|dsv41-worker|tfz0|tfz1)\$' || true")
  [ -z "$others" ] || [ "${FORCE:-0}" = 1 ] || die "$h runs $others (stop the Python fork with docker stop, or '$0 stop'); FORCE=1 overrides"
  busy=$(on "$h" "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true")
  [ -z "$busy" ] || [ "${FORCE:-0}" = 1 ] || die "the GPU on $h is in use: $busy; FORCE=1 overrides"
done

rank=0
for h in "$R0" "$R1"; do
  on "$h" "test -f \$HOME/models/$MODEL/config.json" || die "$h has no ~/models/$MODEL"
  on "$h" "ls \$HOME/tfcache/rank/rank${rank}of2-*.bin >/dev/null 2>&1" || die "$h has no ~/tfcache/rank/rank${rank}of2-*.bin"
  on "$h" "test -f \$HOME/aot/$KERNELS/aot.json" || die "$h has no ~/aot/$KERNELS/aot.json (the captured Triton set)"
  rank=$((rank + 1))
done
# the RoPE tables for this context: copy them from the node that has them to the one that lacks them
for name in rope_c rope_w; do
  f="aot/$KERNELS/rope/${name}_${ROWS}.bin"
  has0=$(on "$R0" "test -f \$HOME/$f && echo y || true")
  has1=$(on "$R1" "test -f \$HOME/$f && echo y || true")
  if [ -z "$has0" ] && [ -z "$has1" ]; then
    die "no ~/$f on either node: run dump_rope.py --rows $ROWS (tools/zig/dsv41/dump_rope.py)"
  elif [ -z "$has0" ]; then
    say "copying ~/$f from $R1 to $R0"
    on "$R1" "cat \$HOME/$f" | on "$R0" "mkdir -p \$HOME/aot/$KERNELS/rope && cat > \$HOME/$f.part && mv \$HOME/$f.part \$HOME/$f"
  elif [ -z "$has1" ]; then
    say "copying ~/$f from $R0 to $R1"
    on "$R0" "cat \$HOME/$f" | on "$R1" "mkdir -p \$HOME/aot/$KERNELS/rope && cat > \$HOME/$f.part && mv \$HOME/$f.part \$HOME/$f"
  fi
done

# -- the binary on each node ----------------------------------------------------------------------------------------
if [ "${NO_COPY:-0}" != 1 ]; then
  on "$BUILD_HOST" "test -x $BUILD_BIN" || die "$BUILD_HOST has no $BUILD_BIN: build it (zig build ... install native)"
  for h in "$R0" "$R1"; do
    say "copying $BUILD_HOST:$BUILD_BIN to $h:~/bin/tensorfold-native"
    on "$BUILD_HOST" "cat $BUILD_BIN" | on "$h" "mkdir -p \$HOME/bin && cat > \$HOME/bin/tensorfold-native.part && chmod +x \$HOME/bin/tensorfold-native.part && mv \$HOME/bin/tensorfold-native.part \$HOME/bin/tensorfold-native"
  done
fi
sum0=$(on "$R0" "sha256sum \$HOME/bin/tensorfold-native | cut -c1-16")
sum1=$(on "$R1" "sha256sum \$HOME/bin/tensorfold-native | cut -c1-16")
[ "$sum0" = "$sum1" ] || die "the binaries differ between the nodes ($sum0 vs $sum1)"
say "binary $sum0 on both nodes; context $CONTEXT, RoPE rows $ROWS"

# -- the ranks ----------------------------------------------------------------------------------------------------
DOCK="--gpus all --ipc=host --net=host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1"
DOCK="$DOCK -v \$HOME/models:/models:ro -v \$HOME/tfcache/rank:/rc:ro -v \$HOME/aot/$KERNELS:/k:ro -v \$HOME/bin:/b:ro"
ENVS="-e TENSORFOLD_CUDA_KERNELS=/k -e TF_DS_RANK_CACHE=/rc -e TENSORFOLD_MEMORY_RESERVE_GIB=$RESERVE"
ENVS="$ENVS -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=enp1s0f0np0"
[ -n "$ENGRAM" ] && ENVS="$ENVS -e TF_DS_ENGRAM=/models/$ENGRAM"
SERVE="/b/tensorfold-native serve /models/$MODEL --tp 2 --master $MASTER --master-port $MASTER_PORT"
[ "$DRAFTS" = 0 ] && SERVE="$SERVE --no-drafts"
SERVE="$SERVE --context $CONTEXT --parallel $PARALLEL --temperature 0 --no-thinking --no-update-check --name dsv41"
SERVE="$SERVE --host 0.0.0.0 --port $HTTP_PORT"

start() { # host name rank
  on "$1" "docker rm -f $2 >/dev/null 2>&1 || true; docker run --rm -d --name $2 $DOCK $ENVS $IMAGE bash -c '$SERVE --rank $3'" >/dev/null
  # follow the log from the start: --rm removes the container (and its log) when the process exits
  on "$1" "docker logs -f $2" >"$LOG_DIR/$2.log" 2>&1 &
  LOG_PIDS+=($!)
}
running() { [ -n "$(on "$1" "docker ps -q --filter name=^$2\$")" ]; }
tail_log() { say "last lines of $1:"; tail -n 30 "$LOG_DIR/$1.log" >&2 || true; }

say "starting rank 0 (tfz0) on $R0"
start "$R0" tfz0 0
waited=0
until on "$R0" "ss -ltn 'sport = :$MASTER_PORT' | grep -q LISTEN"; do
  running "$R0" tfz0 || { tail_log tfz0; die "rank 0 exited before listening on $MASTER_PORT"; }
  [ $waited -lt "$PORT_TIMEOUT" ] || { tail_log tfz0; die "rank 0 did not listen on $MASTER_PORT within ${PORT_TIMEOUT}s"; }
  sleep 2; waited=$((waited + 2))
done
say "rank 0 listens on $MASTER:$MASTER_PORT after ${waited}s; starting rank 1 (tfz1) on $R1"
start "$R1" tfz1 1

waited=0
until grep -qE "serving .* at http://" "$LOG_DIR/tfz0.log" 2>/dev/null; do
  running "$R0" tfz0 || { tail_log tfz0; tail_log tfz1; die "rank 0 exited before serving"; }
  running "$R1" tfz1 || { tail_log tfz1; tail_log tfz0; die "rank 1 exited before rank 0 served"; }
  [ $waited -lt "$READY_TIMEOUT" ] || { tail_log tfz0; tail_log tfz1; die "rank 0 did not serve within ${READY_TIMEOUT}s"; }
  sleep 5; waited=$((waited + 5))
done
say "rank 0 serves after ${waited}s: $(grep -m1 -E 'serving .* at http://' "$LOG_DIR/tfz0.log")"

# -- one greedy request -----------------------------------------------------------------------------------------------
BODY=$(python3 -c 'import json,sys; print(json.dumps({"model":"dsv41","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$PROMPT" "$MAX_TOKENS")
say "POST /v1/chat/completions (max_tokens $MAX_TOKENS, temperature 0)"
on "$R0" "curl -sS --max-time $REQUEST_TIMEOUT http://127.0.0.1:$HTTP_PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" \
  <<<"$BODY" >"$LOG_DIR/reply.json" || { tail_log tfz0; tail_log tfz1; die "the request failed"; }
CONTENT=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({"prompt":r["choices"][0]["message"]["content"] or "","add_special_tokens":False}))' "$LOG_DIR/reply.json") \
  || { cat "$LOG_DIR/reply.json" >&2; tail_log tfz0; die "the reply has no message"; }
on "$R0" "curl -sS --max-time 60 http://127.0.0.1:$HTTP_PORT/v1/tokenize -H 'Content-Type: application/json' -d @-" \
  <<<"$CONTENT" >"$LOG_DIR/reply_ids.json"
python3 - "$LOG_DIR/reply.json" "$LOG_DIR/reply_ids.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
ids = json.load(open(sys.argv[2])).get("tokens")
tf = r.get("tensorfold") or {}
u = r.get("usage") or {}
print("reply:", json.dumps(r["choices"][0]["message"]["content"]))
print("finish:", r["choices"][0].get("finish_reason"), "| prompt tokens:", u.get("prompt_tokens"), "| completion tokens:", u.get("completion_tokens"))
print("reply ids (retokenized content):", ids)
print("token_sha:", tf.get("token_sha"))
print("tok/s: %.2f | prefill %.2fs | ttft %s | total %.2fs" % (tf.get("tokens_per_second") or 0, tf.get("prefill_seconds") or 0, tf.get("time_to_first_token"), tf.get("seconds") or 0))
EOF
say "done; the reply is in $LOG_DIR/reply.json"
