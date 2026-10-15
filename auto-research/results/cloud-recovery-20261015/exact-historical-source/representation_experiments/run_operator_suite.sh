#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG OUTPUT_ROOT" >&2
  exit 2
fi

CONFIG=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")
OUTPUT_ROOT=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_LIST=${GPU_LIST:-0,1,2,3}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
ARMS=(positive_kernel fourier_local wavelet_rbf atlas)
if [[ ${#GPUS[@]} -lt ${#ARMS[@]} ]]; then
  echo "GPU_LIST must provide at least four devices" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
mkdir -p "$OUTPUT_ROOT/logs"

pids=()
for index in "${!ARMS[@]}"; do
  arm=${ARMS[$index]}
  gpu=${GPUS[$index]}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
    "$SCRIPT_DIR/operator_experiment.py" \
    --config "$CONFIG" --arm "$arm" --out "$OUTPUT_ROOT/$arm" \
    > "$OUTPUT_ROOT/logs/$arm.log" 2>&1 &
  pids+=("$!")
done

failure=0
for pid in "${pids[@]}"; do
  wait "$pid" || failure=1
done
if [[ $failure -ne 0 ]]; then
  echo "one or more operator arms failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" \
  "$SCRIPT_DIR/summarize_operator_suite.py" \
  --config "$CONFIG" --root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/summary.json"

