#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG OUTPUT_ROOT" >&2
  exit 2
fi

CONFIG=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")
OUTPUT_ROOT=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_ID=${GPU_ID:-0}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
mkdir -p "$OUTPUT_ROOT/logs"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
  "$SCRIPT_DIR/extract_predictive_atlas.py" \
  --config "$CONFIG" --out "$OUTPUT_ROOT/atlas" \
  > "$OUTPUT_ROOT/logs/atlas.log" 2>&1

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
  "$SCRIPT_DIR/bake_predictive_sequences.py" \
  --config "$CONFIG" --atlas "$OUTPUT_ROOT/atlas/atlas.pt" \
  --split train --out "$OUTPUT_ROOT/baked-train" \
  > "$OUTPUT_ROOT/logs/bake-train.log" 2>&1

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
  "$SCRIPT_DIR/bake_predictive_sequences.py" \
  --config "$CONFIG" --atlas "$OUTPUT_ROOT/atlas/atlas.pt" \
  --split val --sequences 2048 --out "$OUTPUT_ROOT/baked-val" \
  > "$OUTPUT_ROOT/logs/bake-val.log" 2>&1

printf 'DONE\n'
