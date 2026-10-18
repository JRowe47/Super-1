# Experiment report: matched-scratch-fineweb-20261017-v1

Status: **completed, with partial loss of batch-1 artifacts.** The primary hypothesis was rejected. One post-hoc arm beat nanoGPT on all 3 seeds, but the gap is smaller than the preregistered useful effect.
Approval reference: Cowork session 2026-10-17. The user supplied the Daytona key, asked to "keep trying to beat nanoGPT ASAP", and chose "~$100 / up to 2×H100" and "Branch + PR, no merge". After the batch-1 loss the user approved a relaunch under the same cap.
Protocol: `results/matched-scratch-fineweb-20261017-v1/PROTOCOL.json` (sha256 `eef9a0e85c783fbb7cbe8ca6cebb061969edc99271bd5cd7c6d57f6d2ca77ba5`), frozen before any language training. Every post-freeze decision is timestamped in `PROTOCOL_ADDENDA.md`.

## Question and controls

**Primary hypothesis (preregistered):** the exact original 3-timescale GRU hierarchy reaches lower held-out CE than pinned upstream nanoGPT. The conditions are ~125M parameters, the same FineWeb tokens, the GPT-2 tokenizer, the same optimizer recipe and the same LR-trial budget. Both models train and evaluate at context 1024, which removes the historical 512-train / 2K-eval extrapolation confound.

**Comparator:** frozen Aug-8 nanoGPT `model.py` (sha256 `7c0170…d11c`), 12L/12H/768, learned absolute positions.

**Matching track:** parameters and training tokens are matched (503M tokens, 65,536 tokens/update). FLOPs and walltime are measured but not matched.

**Selection:** each arm gets 3 LR trials (3e-4, 1e-3, 3e-3) at 105M tokens with seed 1. The arm's LR is the one with the lowest selection-half CE. Main runs then train 3 seeds at 503M tokens and are scored once on the confirmation half.

## What actually ran

- **Data:** HF `kjj0/fineweb10B-gpt2`.
  - Train: shards 1–10 (1.0B tokens).
  - Selection: `fineweb_val_000000` tokens [0, 50M).
  - Confirmation: tokens [50M, 100M), first 33.6M scored, non-overlapping 1024 windows.
  - FineWeb had never been used in this repository before.
- **Hardware:** Daytona H100 80GB HBM3, image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, torch 2.7.1+cu128, cuDNN 90701, bf16 autocast, TF32 on.
- **Execution of the original hierarchy:** `FastHierarchy` in `train.py`. Each 32-token segment's fast GRUCell recurrence runs as one cuDNN GRU call with the *same* weights; the slow states are constant inside a segment. Parameter names match `HierarchicalDistributionStudent`, so state_dicts load strictly. The equations are unchanged.
- **Fidelity:**
  - FP64 parity (`parity64.py`, tolerance 1e-9): **pass**, max error ~2e-14 on logits, loss, gradients and state across 4 configs, including saturated weights.
  - FP32 parity (`parity.py`): **one row failed** the preregistered 1e-5 state tolerance (2.42e-5 at weight scale 3.0). Logits, loss and gradients passed. The tolerance was not relaxed.
  - Both receipts were re-run identically in batch 2 (`batch2/parity_fp*.json`).
  - The first FP64 attempt hit a harness defect: the loss cast float64 logits to float32. Only that cast was fixed. This is recorded in the addenda.
- **Arms:**

| Arm | Status | Params | Train tok/s (H100) |
|---|---|---:|---:|
| `nanogpt` | preregistered | 124,475,904 | 453–457K |
| `hier` (original hierarchy, width 1344, reset per window) | preregistered | 125,442,240 | 317K |
| `hier_stateful` (same model, contiguous lanes, carried state) | preregistered secondary | 125,442,240 | 318K |
| `deep_hier` (4 × [hierarchy mixer + MLP], width 736) | **post-hoc new arch** | 123,784,896 | 177K |
| `hybrid` (H,A,H,A,H,A blocks, width 736, no pos. emb.) | **post-hoc new arch** | 121,624,736 | 196K |
| `hybrid1` (1 H block + 9 nanoGPT-style A blocks, width 768, no pos. emb.) | **post-hoc new arch** | 126,041,856 (+1.3%) | 255K |

- **Deviations:**
  - The original hierarchy got 2 main seeds, not 3: the futility rule fired at the LR stage, and the batch-1 sandboxes were then lost.
  - `hybrid` got 1 main seed. It lost by 0.069 nats, so seeds 2 and 3 were not run.
  - `deep_hier` main runs were cancelled as futile.
  - `hier_stateful` ran LR trials only.
- **Preservation incident:** the batch-1 sandboxes were ephemeral with a 90-minute idle auto-stop. They were deleted while the agent's command channel was blocked for about 6 hours by a safety-classifier outage.
  - Lost: batch-1 logs, files and checkpoints, including the checkpoint hashes for nanoGPT seeds 1–2, hierarchy seeds 1–2 and hybrid seed 1.
  - Their metrics in `RECOVERED_BATCH1.json` were transcribed from tool outputs read before the deletion.
  - Batch 2 wrote everything to a persistent Daytona volume and mirrored the small files into `batch2/`.

## Results

**Primary metric:** confirmation CE in nats at 503M tokens, context 1024, with state reset per window.

| Seed | nanoGPT | Original hierarchy | hybrid (HAHAHA) | **hybrid1 (1H+9A)** | hybrid1 − nanoGPT |
|---|---:|---:|---:|---:|---:|
| 1 | 3.75365 | 4.31784 | 3.82302 | **3.74049** | −0.01316 |
| 2 | 3.74476 | 4.31625 | — | **3.73883** | −0.00593 |
| 3 | 3.75136 | — | — | **3.73673** | −0.01463 |
| mean | 3.74993 | 4.31705 | 3.82302 | **3.73868** | **−0.01124** (sd 0.0047) |

- **Primary hypothesis: rejected.** The original hierarchy is 0.567 nats worse than nanoGPT at 503M tokens and 0.247 worse at the 105M LR stage, which triggered the futility rule. When train context equals eval context, the historical "hierarchy beats nanoGPT" result does not reproduce.
- **hybrid1 vs nanoGPT (exploratory, post-hoc):**
  - Lower CE on 3 of 3 paired seeds; mean −0.0112 nats (−0.30% relative).
  - That is below the preregistered minimum useful effect of 0.02 nats, and the arm was designed after looking at results. Status: **inconclusive / exploratory**. It needs a fresh preregistered confirmation.
  - The advantage holds along the whole curve: about −0.023 at 100M, −0.013 at 250M and −0.014 at 500M tokens (selection CE, nanoGPT mean of the 2 seeds with saved curves).
- **Short-budget regime:** at 105M tokens, `hybrid` (4.455) and `hybrid1` (4.507) both beat nanoGPT (4.613). Recurrent arms learn faster early, and nanoGPT catches up with more tokens.
  - `hybrid` crosses nanoGPT at about 100M tokens and ends 0.069 worse.
  - `deep_hier` leads early and ends 0.028 worse at 105M.
- **LR trials (selection CE at 105M tokens):**
  - nanoGPT: 4.788 / **4.613** / 6.604
  - hier: 5.189 / **4.860** / 5.714
  - hier_stateful: 5.216 / **4.895** / 7.670
  - deep_hier: n/a (lost) / 4.701 / **4.641**
  - hybrid: n/a (lost) / **4.455** / 4.590
  - hybrid1: 4.626 / **4.507** / 4.787
- **Stateful lanes** (carrying hierarchy state across windows) did not help at 105M tokens: 4.895 vs 4.860.
- **No FP32 duplicate evaluation** was performed. Evaluation uses bf16 autocast with FP32 CE accumulation. This is a disclosed limitation.

Graphs (`graphs/matched-scratch-fineweb-20261017-v1/`): `learning_curves`, `confirmation_per_seed`, `paired_gap`, `lr_trials`. They are rendered by `render.py` from `METRICS.json` only.

## Deployment measurements

These are the batch-2 checkpoints `main_nanogpt_s3` and `main_hybrid1_s2` (`batch2/bench.json`). Conditions: same H100, bf16 autocast, 1024-token prefill, 5 warmup passes and 30 timed passes, synchronized.

| | nanoGPT B=1 | hybrid1 B=1 | nanoGPT B=16 | hybrid1 B=16 |
|---|---:|---:|---:|---:|
| prefill p50 / p95 (ms) | 1.95 / 2.06 | 17.49 / 18.31 | 10.09 / 10.13 | 26.50 / 27.62 |
| prefill tok/s | 525K | 59K | 1.62M | 618K |
| peak allocated (GB) | 0.82 | 0.90 | 1.20 | 2.59 |

- **Speed cost.** The recurrent block is sequential over 32-token segments, so hybrid1 prefill is 9× slower at batch 1 and 2.6× slower at batch 16. Training throughput is 1.8× lower. **At matched walltime, nanoGPT would train on about 1.8× more tokens; walltime-matched quality was not measured.**
- **Not measured:** decode latency, because pinned nanoGPT has no KV cache.
- **Analytic per-sequence state at 1024 context:**
  - hybrid1 keeps KV for 9 attention layers plus a 4×768 recurrent state.
  - nanoGPT keeps KV for 12 layers.
  - That is a ~25% smaller KV cache for hybrid1 (analytic, not a total-memory measurement).

## Decision

- **Predeclared criterion:** not met for the primary arm; the original hierarchy is rejected at matched context.
- **hybrid1:** inconclusive. It is consistent across 3 seeds but under the 0.02 useful-effect threshold, slower, slightly larger and post-hoc.
- **Dominant alternative explanation:** hybrid1's gain may come from removing learned absolute positions (recurrence supplying position information) and from the extra 1.3% of parameters, not from the multi-timescale hierarchy.
- **Next best experiment** (≈ 6 H100-hours, ≈ $35 GPU-only), preregistered fresh. Three arms, 3 seeds, 500M tokens:
  1. hybrid1;
  2. nanoGPT with RoPE (the strong positional baseline required by `auto-research.md`);
  3. the hybrid1 equal-parameter control, with the first block a plain causal-conv or single-GRU mixer without segment/document states.

  Interpretation: if (2) or (3) matches hybrid1, the hierarchy is not the cause. Also add a walltime-matched nanoGPT arm.
- **New authority:** required for any further GPU spend.

## Integrity and cleanup

- **Terminal receipt:** `TERMINAL.json`. Cleanup: `CLEANUP.json`.
- **GPU sandboxes:** verified at 0 remaining (2026-10-18T01:40Z).
- **Retained:** Daytona volume `super1-results` (3.3 GB) with the 7 batch-2 checkpoints (hashes in `batch2/runs/*/result.json`).
- **Approximate usage:** about 11 H100-hours, roughly $44 GPU-only at list price plus CPU/RAM add-ons. The Daytona bill is authoritative.
- **Lost:** batch-1 checkpoints, logs and hashes (see above).
- **Official WikiText-2 test:** untouched. No SOTA or novelty claim. No automations scheduled.

## Reproduce

```sh
# on an H100 with the FineWeb shards in /work/data and sources under /work/src
python train.py --config '<config.json from batch2/runs/<run>/>' --out /work/runs/<run>
python parity64.py out.json
python bench.py bench.json <run> ...
python ../../graphs/matched-scratch-fineweb-20261017-v1/render.py
```

`train.py` in this directory is the final version. Batch-1 runs used earlier revisions of the same file; the nanoGPT, FastHierarchy, data and eval code paths were unchanged, and only new arms and the float64 loss-cast fix were added.
