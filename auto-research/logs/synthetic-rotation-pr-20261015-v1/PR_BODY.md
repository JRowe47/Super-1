The GitHub auto-research prompt now has an executed experiment-to-PR result: three paired seeds, four arms, 80 optimization updates and 10,240 matched target exposures per arm/seed. Oriented fast-state rotation **does not pass the preregistered promotion criterion**. This PR preserves the negative/inconclusive finding and every checkpoint; it does not promote an architecture.

The native arm imports the recovered byte-exact hierarchy. Geometry and alignment controls use the existing wrapper with equal 8,832 parameters and common native initialization. nanoGPT is pinned to `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` and explicitly not parameter/FLOP matched. All use the same deterministic synthetic repeating-motif data, independent train/selection/confirmation generator streams, FP32, one LR trial, final checkpoint only, and matched train/evaluation context 32.

| Seed | Native CE | Oriented CE | Alignment CE | nanoGPT CE |
|---|---:|---:|---:|---:|
| 73 | 2.727595448 | 2.726783395 | 2.728287667 | 2.663441926 |
| 191 | 2.687289476 | 2.687752277 | 2.690792680 | 2.707339019 |
| 307 | 2.756008893 | 2.754041553 | 2.755060375 | 2.681283355 |

CE is nats/target on 1,024 reserved synthetic targets. Oriented-minus-alignment mean: **-0.001854499**, below the required 0.02-nat useful improvement. Descriptive paired 95% t interval: **[-0.004476049, +0.000767051]**. Orientation also loses to native on seed 191. No promotion; n=3 and short optimization cannot establish convergence, language quality, novelty, SOTA or a deployment advantage.

All 12 training runs completed. The first process then failed in a secondary nanoGPT reset-window diagnostic because sliced targets were noncontiguous. Its code, exit-1 receipt and checkpoints remain immutable. A separately preregistered evaluation-only recovery made adapter inputs contiguous and completed all 12 exact duplicate confirmations without retraining or changing the primary criterion. Previously observed primary results match exactly. This is a disclosed recovery after partial metric exposure, not an uninterrupted protocol pass. AGENT_PROMPTS.md records the failure lesson.

Total experiment CPU time: 9.089 seconds; approximately 1.8 MB of artifacts. No GPU/MPS/cloud, external data, installations, official test, background training or automation. Language training stays paused.

Evidence:
- `auto-research/reports/synthetic-rotation-pr-20261015-v1.md`: full report, limitations, protocol hashes, all seeds and next proposed disconfirming control.
- `auto-research/results/synthetic-rotation-pr-20261015-v1/`: preregistered training code/data/protocol, all 960 raw update rows, 12 model/optimizer checkpoints with hashes, original failure.
- `auto-research/results/synthetic-rotation-pr-20261015-eval-v2/`: frozen recovery protocol/code, complete metrics and successful terminal receipt.
- Registry and global manifest updated; historical model sources unchanged.

Validation passed:
```sh
python3 -B scripts/verify_evidence.py
python3 -B auto-research/results/synthetic-rotation-pr-20261015-eval-v2/verify_results.py
```

Reproduce the successful evaluation using preserved checkpoints and a fresh SSD directory:
```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -B auto-research/results/synthetic-rotation-pr-20261015-eval-v2/evaluate_checkpoints.py \
  --output '/Volumes/SP AI 01_16/CodexWorkspaces/trin-codex/auto-research/results/rotation-eval-replay-CHOOSE-UNIQUE-ID'
```
The report also preserves the original training command and explicitly documents its expected diagnostic failure. Do not merge automatically.
