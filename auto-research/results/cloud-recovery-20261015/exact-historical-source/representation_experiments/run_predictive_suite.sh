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

CUDA_VISIBLE_DEVICES=${GPUS[0]} PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
  "$SCRIPT_DIR/extract_predictive_atlas.py" \
  --config "$CONFIG" --out "$OUTPUT_ROOT/atlas" \
  > "$OUTPUT_ROOT/logs/atlas.log" 2>&1

CUDA_VISIBLE_DEVICES=${GPUS[0]} PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
  "$SCRIPT_DIR/bake_predictive_sequences.py" \
  --config "$CONFIG" --atlas "$OUTPUT_ROOT/atlas/atlas.pt" \
  --split train --out "$OUTPUT_ROOT/baked-train" \
  > "$OUTPUT_ROOT/logs/bake-train.log" 2>&1 &
train_bake_pid=$!
CUDA_VISIBLE_DEVICES=${GPUS[1]} PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -u \
  "$SCRIPT_DIR/bake_predictive_sequences.py" \
  --config "$CONFIG" --atlas "$OUTPUT_ROOT/atlas/atlas.pt" \
  --split val --sequences 2048 --out "$OUTPUT_ROOT/baked-val" \
  > "$OUTPUT_ROOT/logs/bake-val.log" 2>&1 &
val_bake_pid=$!
wait "$train_bake_pid"
wait "$val_bake_pid"

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
