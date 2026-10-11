#!/usr/bin/env bash
# The bit-exact reference for Qwen3.8-Flash-Next on one DGX Spark: upstream TensorFold 0.6.6 (`tensorfold serve`,
# image tensorfold-qwen38fn:0.6.6) on the EXL3 3.05bpw pack, bf16 KV, MTP drafts on, at the goal configuration
# (4 streams, 262,144 positions each), as container qfn066 on $NODE. Run it from any machine that reaches the node.
#
#   bash reference_serve.sh [run]   start the server, wait for it, send the smoke request (greedy, the DSV4.1 smoke's
#                                   prompt) and print the reply's token_sha and speed; KEEP=1 leaves it serving
#   bash reference_serve.sh stop    remove the container
#
# Then, with the server kept: python3 tools/zig/dsv41/bench_watch.py --host $NODE --model qwen38fn --only short,concurrent
# --label tf066-exl3-305 (kit_bench decode set a, then four streams at once).
#
# Environment: NODE (spark-616b), MODEL (~/models/... pack directory name), IMAGE, PARALLEL (4), CONTEXT (262144),
# MAX_TOKENS (64), LOG_DIR, CAPTURE=1 (record every Triton launch through the recorder shim into
# ~/aot/qfn-serve-capture/{out,triton}: tools/zig/dsv41/serve_capture.md), EXTRA_ARGS (more serve flags).
# The GPU must be free (tools/zig/dsv41/gpu_guard.sh runs first): host and GPU share one memory.
set -euo pipefail

NODE=${NODE:-spark-616b}
MODEL=${MODEL:-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
IMAGE=${IMAGE:-tensorfold-qwen38fn:0.6.6}
PARALLEL=${PARALLEL:-4}
CONTEXT=${CONTEXT:-262144}
HTTP_PORT=${HTTP_PORT:-8000}
MAX_TOKENS=${MAX_TOKENS:-64}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
LOG_DIR=${LOG_DIR:-${TMPDIR:-/tmp}/qfn066-$(date +%Y%m%d-%H%M%S)}
PROMPT=${PROMPT:-Write a Python function that parses an HTTP request line and returns method, path, version.}
NAME=qfn066
HERE=$(cd "$(dirname "$0")" && pwd)

SSH=(ssh -4 -o BatchMode=yes -o ConnectTimeout=10)
on() { "${SSH[@]}" "$NODE" "$@"; }
say() { echo "[qfn066 $(date +%H:%M:%S)] $*" >&2; }
die() { say "error: $*"; exit 1; }

if [ "${1:-run}" = stop ]; then
  on "docker rm -f $NAME >/dev/null 2>&1 || true"
  say "removed $NAME on $NODE"
  exit 0
fi
[ "${1:-run}" = run ] || die "usage: $0 [run|stop]"

NEED_GIB=100 on bash -s < "$HERE/../dsv41/gpu_guard.sh" || die "the GPU on $NODE is not free (gpu_guard)"
on "test -f \$HOME/models/$MODEL/config.json" || die "$NODE has no ~/models/$MODEL"
mkdir -p "$LOG_DIR"

capture=()
if [ "${CAPTURE:-0}" = 1 ]; then
  scp -q "$HERE/../dsv41/shim/usercustomize.py" "$HERE/../triton_aot_manifest.py" "$NODE:/tmp/"
  on "rm -rf \$HOME/aot/qfn-serve-capture && mkdir -p \$HOME/aot/qfn-serve-capture/{shim,out,triton} && mv /tmp/usercustomize.py /tmp/triton_aot_manifest.py \$HOME/aot/qfn-serve-capture/shim/"
  capture=(-v "\$HOME/aot/qfn-serve-capture:/aot" -e PYTHONPATH=/aot/shim -e TF_DSV41_CAPTURE_DIR=/aot/out/rank0
           -e TRITON_CACHE_DIR=/aot/triton)
fi

say "starting $NAME on $NODE: --parallel $PARALLEL --context $CONTEXT"
on "mkdir -p \$HOME/.cache/tensorfold-qwen38fn-0.6.6/torch_extensions \$HOME/.cache/tensorfold-qwen38fn-0.6.6/triton && \
  docker run -d --name $NAME --gpus all --ipc=host --net=host --user \$(id -u):\$(id -g) -e HOME=/tmp \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
  -v \$HOME/.cache/tensorfold-qwen38fn-0.6.6:/cache -v \$HOME/models/$MODEL:/model:ro ${capture[*]:-} $IMAGE \
  serve /model --name qwen38fn --host 0.0.0.0 --port $HTTP_PORT --parallel $PARALLEL --context $CONTEXT ${EXTRA_ARGS:-}" >/dev/null
on "docker logs -f $NAME" >"$LOG_DIR/$NAME.log" 2>&1 &
LOG_PID=$!
trap 'kill $LOG_PID 2>/dev/null || true; [ "${KEEP:-0}" = 1 ] || on "docker rm -f $NAME >/dev/null 2>&1 || true"; say "log: $LOG_DIR/$NAME.log"' EXIT

waited=0
until on "curl -fsS --max-time 5 http://127.0.0.1:$HTTP_PORT/v1/models" >/dev/null 2>&1; do
  on "docker ps -q -f name=^$NAME\$" | grep -q . || { tail -40 "$LOG_DIR/$NAME.log" >&2; die "the server exited"; }
  [ $waited -lt "$READY_TIMEOUT" ] || die "not serving within ${READY_TIMEOUT}s"
  sleep 5; waited=$((waited + 5))
done
say "serving after ${waited}s"
grep -iE "context|stream|window|memory|GiB|parallel" "$LOG_DIR/$NAME.log" | tail -12 >&2 || true

BODY=$(python3 -c 'import json,sys; print(json.dumps({"model":"qwen38fn","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$PROMPT" "$MAX_TOKENS")
on "curl -sS --max-time 1800 http://127.0.0.1:$HTTP_PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" \
  <<<"$BODY" >"$LOG_DIR/reply.json" || die "the request failed"
python3 - "$LOG_DIR/reply.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
tf = r.get("tensorfold") or {}
u = r.get("usage") or {}
print("reply:", json.dumps(r["choices"][0]["message"].get("content")))
print("finish:", r["choices"][0].get("finish_reason"), "| prompt tokens:", u.get("prompt_tokens"), "| completion tokens:", u.get("completion_tokens"))
print("token_sha:", tf.get("token_sha"))
print("stats:", json.dumps({k: v for k, v in tf.items() if k != "token_sha"}), "| speculative:", json.dumps(r.get("speculative")))
EOF
say "done; the reply is in $LOG_DIR/reply.json"
