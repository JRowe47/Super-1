#!/usr/bin/env bash
# Replicate matched-scratch-fineweb-20261017-v1 on one or more NVIDIA GPUs (tested: H100 80GB, torch 2.7.1+cu128).
#   bash run.sh parity                 # FP32 + FP64 exactness of FastHierarchy vs the recovered original class
#   bash run.sh headline               # nanoGPT vs hybrid1, 3 seeds each, 503M tokens (~3.5 H100-hours)
#   bash run.sh all                    # every run in configs/ (~15 H100-hours)
#   bash run.sh configs/main_hybrid1_s1.json ...   # specific runs
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export DATA_DIR="${DATA_DIR:-$HERE/data}" OUT_DIR="${OUT_DIR:-$HERE/runs}"
mkdir -p "$OUT_DIR"
[ -f "$DATA_DIR/fineweb_train_000010.bin" ] || python "$HERE/download_data.py"
cd "$HERE/src"
run_cfg() {  # $1 = config path
  local name; name="$(basename "$1" .json)"
  python - "$1" "$DATA_DIR" > "$OUT_DIR/$name.cfg.json" <<'P'
import json, sys; c = json.load(open(sys.argv[1])); c["data_root"] = sys.argv[2]; print(json.dumps(c))
P
  echo ">> $name"; python train.py --config "$OUT_DIR/$name.cfg.json" --out "$OUT_DIR/$name" 2>&1 | tee "$OUT_DIR/$name.log" | grep -E "RESULT|selection_ce" || true
}
case "${1:-headline}" in
  parity)   python parity.py "$OUT_DIR/parity_fp32.json"; python parity64.py "$OUT_DIR/parity_fp64.json" ;;
  headline) for s in 1 2 3; do run_cfg "$HERE/configs/main_nanogpt_s$s.json"; run_cfg "$HERE/configs/main_hybrid1_s$s.json"; done
            python "$HERE/summarize.py" "$OUT_DIR" ;;
  all)      for c in "$HERE"/configs/lr_*.json "$HERE"/configs/main_*.json; do run_cfg "$c"; done; python "$HERE/summarize.py" "$OUT_DIR" ;;
  *)        for c in "$@"; do run_cfg "$c"; done; python "$HERE/summarize.py" "$OUT_DIR" ;;
esac
