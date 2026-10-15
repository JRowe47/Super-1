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
if [[ ${#GPUS[@]} -lt 4 ]]; then
  echo "GPU_LIST must provide four devices" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
mkdir -p "$OUTPUT_ROOT/logs"

free_gb=$(df -BG / | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
minimum_free_gb=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["safety"]["minimum_free_gb"])' "$CONFIG")
if (( free_gb < minimum_free_gb )); then
  echo "refusing to start: ${free_gb}GB free is below ${minimum_free_gb}GB" >&2
  exit 1
fi

ARMS=(
  recurrent_full_kl
  hierarchical_full_kl
  hierarchical_embed_full_kl
  hierarchical_branch_distill
)
pids=()
for index in "${!ARMS[@]}"; do
  arm=${ARMS[$index]}
  gpu=${GPUS[$index]}
  if [[ -s "$OUTPUT_ROOT/$arm/result.json" ]]; then
    continue
  fi
  mkdir -p "$OUTPUT_ROOT/$arm"
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 PYTHONPATH="$PROJECT_ROOT" \
    "$PYTHON_BIN" -u "$SCRIPT_DIR/distribution_distillation.py" \
    --config "$CONFIG" --arm "$arm" --out "$OUTPUT_ROOT/$arm" \
    > "$OUTPUT_ROOT/logs/$arm.log" 2>&1 &
  pids+=("$!")
done

failure=0
for pid in "${pids[@]}"; do
  wait "$pid" || failure=1
done
if [[ $failure -ne 0 ]]; then
  echo "one or more distribution students failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

free_gb=$(df -BG / | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if (( free_gb < minimum_free_gb )); then
  echo "pausing before benchmarks: ${free_gb}GB free is below ${minimum_free_gb}GB" >&2
  exit 1
fi

BENCHMARK_LIMIT=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["benchmark"]["limit"])' "$CONFIG")
pids=()
for index in "${!ARMS[@]}"; do
  arm=${ARMS[$index]}
  gpu=${GPUS[$index]}
  if [[ -s "$OUTPUT_ROOT/$arm/benchmarks.json" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 PYTHONPATH="$PROJECT_ROOT" \
    "$PYTHON_BIN" -u "$SCRIPT_DIR/distribution_benchmarks.py" \
    --config "$CONFIG" --arm "$arm" \
    --checkpoint "$OUTPUT_ROOT/$arm/student_final.pt" \
    --limit "$BENCHMARK_LIMIT" --out "$OUTPUT_ROOT/$arm/benchmarks.json" \
    > "$OUTPUT_ROOT/logs/$arm-benchmarks.log" 2>&1 &
  pids+=("$!")
done
failure=0
for pid in "${pids[@]}"; do
  wait "$pid" || failure=1
done
if [[ $failure -ne 0 ]]; then
  echo "one or more distribution benchmarks failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" \
  "$SCRIPT_DIR/summarize_distribution_suite.py" \
  --config "$CONFIG" --root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/summary.json"
