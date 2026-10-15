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
WAIT_SECONDS=${WAIT_SECONDS:-60}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -lt 4 ]]; then
  echo "GPU_LIST must provide four devices" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
mkdir -p "$OUTPUT_ROOT/logs"

while [[ ! -s "$OUTPUT_ROOT/atlas/atlas.pt" ||
         ! -s "$OUTPUT_ROOT/baked-train/manifest.json" ||
         ! -s "$OUTPUT_ROOT/baked-val/manifest.json" ||
         ! -f "$OUTPUT_ROOT/sync.complete" ]]; do
  sleep "$WAIT_SECONDS"
done

STUDENTS=(transformer recurrent_logits hierarchical_predictive hierarchical_logit_control)
pids=()
for index in "${!STUDENTS[@]}"; do
  student=${STUDENTS[$index]}
  gpu=${GPUS[$index]}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
    "$SCRIPT_DIR/predictive_experiment.py" \
    --config "$CONFIG" --student "$student" \
    --train_manifest "$OUTPUT_ROOT/baked-train/manifest.json" \
    --val_manifest "$OUTPUT_ROOT/baked-val/manifest.json" \
    --atlas "$OUTPUT_ROOT/atlas/atlas.pt" --out "$OUTPUT_ROOT/$student" \
    > "$OUTPUT_ROOT/logs/$student.log" 2>&1 &
  pids+=("$!")
done

failure=0
for pid in "${pids[@]}"; do
  wait "$pid" || failure=1
done
if [[ $failure -ne 0 ]]; then
  echo "one or more predictive students failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

BENCHMARK_LIMIT=${BENCHMARK_LIMIT:-1000}
pids=()
for index in "${!STUDENTS[@]}"; do
  student=${STUDENTS[$index]}
  gpu=${GPUS[$index]}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
    "$SCRIPT_DIR/predictive_benchmarks.py" \
    --config "$CONFIG" --student "$student" \
    --checkpoint "$OUTPUT_ROOT/$student/student_final.pt" \
    --limit "$BENCHMARK_LIMIT" --out "$OUTPUT_ROOT/$student/benchmarks.json" \
    > "$OUTPUT_ROOT/logs/$student-benchmarks.log" 2>&1 &
  pids+=("$!")
done
failure=0
for pid in "${pids[@]}"; do
  wait "$pid" || failure=1
done
if [[ $failure -ne 0 ]]; then
  echo "one or more benchmark evaluations failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" \
  "$SCRIPT_DIR/summarize_predictive_suite.py" \
  --config "$CONFIG" --root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/summary.json"
