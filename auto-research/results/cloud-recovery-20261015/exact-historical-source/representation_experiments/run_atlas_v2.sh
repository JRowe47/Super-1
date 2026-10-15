#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG OUTPUT_DIRECTORY" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")
OUTPUT=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")
PYTHON_BIN=${PYTHON_BIN:-python3}
mkdir -p "$OUTPUT"

PYTHONPATH="$PROJECT_ROOT" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" -u "$SCRIPT_DIR/atlas_v2_experiment.py" \
  --config "$CONFIG" --out "$OUTPUT"
