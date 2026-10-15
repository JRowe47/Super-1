# Hierarchy result code

This ZIP contains the hierarchy-vs-transformer experiment source, exact
10M-token config, runners, benchmark harness, requirements, original result
JSONs, and the PDF report.

Run from this directory after placing the Qwen-tokenized `train.bin` and
`val.bin` at `data/qwen3-06b-representation` and prefetching Hugging Face
assets:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 python -u \
  representation_experiments/run_matched_hierarchy_single_gpu.py \
  --config representation_experiments/configs/qwen06b_matched_hierarchy_function_v1.json \
  --out runs/qwen06b-matched-hierarchy-function-v1 \
  --gpu 0
```

The corpus and model checkpoints are excluded because they are several
gigabytes; the source and recorded results are included.
