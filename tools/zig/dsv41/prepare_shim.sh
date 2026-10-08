#!/usr/bin/env bash
# Assemble the recorder shim directory mounted into both ranks' containers: usercustomize.py plus a copy of
# tools/zig/triton_aot_manifest.py (the recorder). Usage: bash prepare_shim.sh <shim-dir>
set -euo pipefail
DEST=$1
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DEST"
cp "$HERE/shim/usercustomize.py" "$DEST/usercustomize.py"
cp "$HERE/../triton_aot_manifest.py" "$DEST/triton_aot_manifest.py"
echo "shim at $DEST: mount it and set PYTHONPATH=<mount> TF_DSV41_CAPTURE_DIR=<per-rank dir> TRITON_CACHE_DIR=<per-rank dir>"
