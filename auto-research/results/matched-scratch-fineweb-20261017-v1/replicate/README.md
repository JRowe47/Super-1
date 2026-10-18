# Replicate: matched scratch-CE FineWeb benchmark (nanoGPT vs hierarchy arms)

Reproduces `auto-research/reports/matched-scratch-fineweb-20261017-v1.md`.

## Requirements
NVIDIA GPU (tested on H100 80GB), CUDA PyTorch 2.7.1, numpy and gcc (for `torch.compile`). Tested image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`. About 2.2 GB of disk for data.

## Run
```sh
bash run.sh parity      # FastHierarchy == recovered HierarchicalDistributionStudent (FP64 ~1e-14)
bash run.sh headline    # nanoGPT vs hybrid1, seeds 1-3, 503M tokens each (~3.5 H100-hours)
bash run.sh all         # all 30 configs: LR trials + main runs (~15 H100-hours)
python summarize.py runs
```
Data is downloaded automatically from HF `kjj0/fineweb10B-gpt2`: train shards 1–10 and the val shard. Val tokens [0, 50M) are the selection split and [50M, 100M) the confirmation split.

## Expected (confirmation CE, nats; ±~0.005 seed noise)
| arm | s1 | s2 | s3 | mean |
|---|---|---|---|---|
| nanogpt | 3.7537 | 3.7448 | 3.7514 | 3.7499 |
| hybrid1 | 3.7405 | 3.7388 | 3.7367 | 3.7387 (−0.0112 vs nanoGPT) |
| hier (original) | 4.3178 | 4.3163 | – | 4.3170 |
| hybrid (HAHAHA) | 3.8230 | – | – | 3.8230 |

GPU kernels (cuDNN, flash attention) are not bitwise deterministic, so exact values will differ slightly between runs.

## Files
- `src/train.py`: data loader, all arms, training and eval. `FastHierarchy` executes the original hierarchy equations.
- `src/nanogpt_model.py`: the pinned upstream nanoGPT, sha256 `7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`.
- `src/representation_experiments/`: the recovered exact original source; `predictive_models.py` sha256 `b73a7c07801f5672adb52c0cb16a3c772d62e50eebc0366491d497f76d13f73f`.
- `configs/`: one JSON per run.
  - Batch-2 configs are copied verbatim from the results.
  - Batch-1 configs are regenerated with the same generator. The regenerated `main_nanogpt_s3` matches the config that actually ran.
- `src/parity.py`, `src/parity64.py`: exactness tests. `src/bench.py`: prefill latency and memory.

## Caveats
- hybrid1 was designed post-hoc. Its gain (0.011 nats) is below the preregistered 0.02-nat useful effect.
- hybrid1 trains 1.8× slower than nanoGPT.
- This is not a SOTA claim. See the report.
