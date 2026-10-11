#!/usr/bin/env bash
# The native Zig engine serving Qwen3.8-Flash-Next on one DGX Spark (container tfq0 on $NODE, NVIDIA's PyTorch image
# for the CUDA runtime), with the smoke request of reference_serve.sh so the token_sha compares with 0.6.6's
# (83e5ff518a84 for the default prompt, 64 tokens, thinking on).
#
#   bash serve_native.sh [run]   copy the binary, start the server, wait, send the smoke request; KEEP=1 leaves it up
#   bash serve_native.sh stop    remove the container
#
# Environment: NODE (spark-616b), BIN (the local tensorfold-native, default zig-out/native/bin/tensorfold-native),
# KERNELS (~/aot/... serving kernel set on the node, capture_serve_kernels.sh), MODEL, PARALLEL (4), CONTEXT (262144),
# RESERVE (2), MAX_TOKENS (64), LOG_DIR, EXTRA_ARGS, ENGINE_ENV (space-separated NAME=VALUE for the server, e.g.
# TF_QFN_GRAPHS=0).
set -euo pipefail

NODE=${NODE:-spark-616b}
BIN=${BIN:-zig-out/native/bin/tensorfold-native}
KERNELS=${KERNELS:-qfn-serve-kernels}
MODEL=${MODEL:-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
PARALLEL=${PARALLEL:-4}
CONTEXT=${CONTEXT:-262144}
RESERVE=${RESERVE:-2}
HTTP_PORT=${HTTP_PORT:-8000}
MAX_TOKENS=${MAX_TOKENS:-64}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
LOG_DIR=${LOG_DIR:-${TMPDIR:-/tmp}/tfq0-$(date +%Y%m%d-%H%M%S)}
PROMPT=${PROMPT:-Write a Python function that parses an HTTP request line and returns method, path, version.}
NAME=tfq0
HERE=$(cd "$(dirname "$0")" && pwd)

SSH=(ssh -4 -o BatchMode=yes -o ConnectTimeout=10)
on() { "${SSH[@]}" "$NODE" "$@"; }
say() { echo "[tfq0 $(date +%H:%M:%S)] $*" >&2; }
die() { say "error: $*"; exit 1; }

if [ "${1:-run}" = stop ]; then
  on "docker rm -f $NAME >/dev/null 2>&1 || true"
  say "removed $NAME on $NODE"
  exit 0
fi
[ "${1:-run}" = run ] || die "usage: $0 [run|stop]"

NEED_GIB=100 on bash -s < "$HERE/../dsv41/gpu_guard.sh" || die "the GPU on $NODE is not free (gpu_guard)"
on "test -f \$HOME/aot/$KERNELS/aot.json" || die "$NODE has no ~/aot/$KERNELS/aot.json"
mkdir -p "$LOG_DIR"
on "mkdir -p \$HOME/bin/qfn && cat > \$HOME/bin/qfn/tensorfold-native.new && chmod +x \$HOME/bin/qfn/tensorfold-native.new && mv \$HOME/bin/qfn/tensorfold-native.new \$HOME/bin/qfn/tensorfold-native" < "$BIN"
sum=$(on "sha256sum \$HOME/bin/qfn/tensorfold-native" | cut -c1-12)
[ "$sum" = "$(shasum -a 256 "$BIN" | cut -c1-12)" ] || die "the copied binary differs"
say "binary $sum; --parallel $PARALLEL --context $CONTEXT, kernels ~/aot/$KERNELS"
on "docker run --rm -d --name $NAME --gpus all --ipc=host --net=host -v \$HOME/models/$MODEL:/model:ro -v \$HOME/aot/$KERNELS:/k:ro -v \$HOME/bin/qfn:/b:ro \
  -e TENSORFOLD_CUDA_KERNELS=/k -e TENSORFOLD_MEMORY_RESERVE_GIB=$RESERVE $(for e in ${ENGINE_ENV:-}; do printf -- '-e %s ' "$e"; done) $IMAGE \
  /b/tensorfold-native serve /model --context $CONTEXT --parallel $PARALLEL --temperature 0 --no-update-check --name qwen38fn \
  --host 0.0.0.0 --port $HTTP_PORT ${EXTRA_ARGS:-}" >/dev/null
on "docker logs -f $NAME" >"$LOG_DIR/$NAME.log" 2>&1 &
LOG_PID=$!
trap 'kill $LOG_PID 2>/dev/null || true; [ "${KEEP:-0}" = 1 ] || on "docker rm -f $NAME >/dev/null 2>&1 || true"; say "log: $LOG_DIR/$NAME.log"' EXIT

waited=0
until grep -qE "serving .* at http://" "$LOG_DIR/$NAME.log" 2>/dev/null; do
  on "docker ps -q -f name=^$NAME\$" | grep -q . || { tail -30 "$LOG_DIR/$NAME.log" >&2; die "the server exited"; }
  [ $waited -lt "$READY_TIMEOUT" ] || die "not serving within ${READY_TIMEOUT}s"
  sleep 5; waited=$((waited + 5))
done
say "serving after ${waited}s: $(grep -m1 -E 'streams at once' "$LOG_DIR/$NAME.log" || true)"

BODY=$(python3 -c 'import json,sys; print(json.dumps({"model":"qwen38fn","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$PROMPT" "$MAX_TOKENS")
on "curl -sS --max-time 1800 http://127.0.0.1:$HTTP_PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" \
  <<<"$BODY" >"$LOG_DIR/reply.json" || die "the request failed"
python3 - "$LOG_DIR/reply.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
tf = r.get("tensorfold") or {}
u = r.get("usage") or {}
m = r["choices"][0]["message"]
print("reply:", json.dumps(m.get("content")), "| reasoning:", json.dumps((m.get("reasoning_content") or m.get("reasoning") or "")[:120]))
print("finish:", r["choices"][0].get("finish_reason"), "| prompt tokens:", u.get("prompt_tokens"), "| completion tokens:", u.get("completion_tokens"))
print("token_sha:", tf.get("token_sha"), "(0.6.6: 83e5ff518a84)")
print("tok/s: %.2f | prefill %.2fs | ttft %s" % (tf.get("tokens_per_second") or 0, tf.get("prefill_seconds") or 0, tf.get("time_to_first_token")))
EOF
say "done; the reply is in $LOG_DIR/reply.json"
