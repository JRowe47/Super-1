# Linear minGRU CPU benchmarks: scoped upstream import

The longer matched CPU recipes favor the linear minGRU fast-cell adaptation
over ordinary GRU within the original fast/segment/document hierarchy:

| Batch / threads / peak LR | Linear minGRU CE | Ordinary GRU CE | Perplexity reduction | Training throughput ratio |
|---|---:|---:|---:|---:|
| 4 / 2 / 0.004 | 4.454564 | 4.541924 | 8.37% | 1.93× |
| 16 / 4 / 0.008 | 4.382792 | 4.534900 | 14.11% | 1.69× |

Each recipe receives a 180-second **training-wall** allowance. The B16/T4 pair
uses 668.9 versus 677.2 aggregate CPU seconds and processes 5,038,080 versus
2,973,696 training targets. Parameters are 996,320 versus 995,925. This is a
practical time-budget result: cheaper updates permit more data exposure and
optimizer steps. The earlier equal-token experiment favored ordinary GRU
(4.864115 versus 4.988805), and remains in upstream PR #3. The equal-CPU
follow-up is already in PR #4.

These are single-seed exploratory CPU runs on a preserved FineWeb-Edu sample,
with a train-only 2,048-token BPE vocabulary and reset 512-token windows.
Validation was reused. The two 60-second stages share seed 101603; the longer
stage uses seed 101604 and a fresh data order. They are not three independent
confirmations, a GPU speedrun, or evidence of large-model superiority.

## Included evidence

| Stage | Scope | Complete report |
|---|---|---|
| `gru-batch-wall-20261016-v1` | 27 throughput cells and six 60-second quality trials | [Batch/thread comparison](gru-batch-wall-20261016-v1.md) |
| `mingru-thread-lr-20261016-v1` | Seven 60-second minGRU thread/LR trials | [Thread/LR screen](mingru-thread-lr-20261016-v1.md) |
| `gru-longer-wall-20261016-v1` | Six 180-second recipes, including both nanoGPT controls | [Longer comparison](gru-longer-wall-20261016-v1.md) |

All 19 trials, including the weaker configurations, retain frozen protocols,
complete compressed traces, results, checks, checkpoint hashes, logs, figures,
source, and licenses. The batch/thread helpers are dependencies of the later
two runners. The thread/LR screen records how the longer recipes were chosen.
The nanoGPT wrapper's redundant CE computation was removed before these runs;
its corrected version and controls are retained.

All 139 imported files (2,522,435 bytes) are unchanged from contributor commit
`4b905f740f87ec037c9c9e1e158703c81cc6aed4`. The branch starts at upstream commit
`884c07ad289361c3bb0ccfe1ae4931d7a60258eb`; only the three named stages are copied.
Context/capacity, curve conversion, spectral, and online-learning experiments
are outside this change. Existing model implementation files are unchanged.

The original reports and protocols retain their historical fork-only wording.
The user's explicit 2026-10-18 request authorizes this upstream submission for
these three studies; [the import record](../results/mingru-cpu-upstream-20261018-v1/IMPORT.json)
documents that scope without rewriting historical evidence.

## Verify the checked-in records

From the repository root, with Python 3 and no third-party dependencies:

```bash
python3 scripts/verify_evidence.py
python3 auto-research/results/mingru-cpu-upstream-20261018-v1/verify.py
```

The first command checks repository hashes and existing saved evidence. The
second compares the imported files with their original Git object hashes,
checks protocol/source hashes and Python syntax, and reconciles all 19 stored
training traces, learning-rate schedules, target counts, clocks, final-window
means, perplexities, and the 27-cell throughput selection. It does not import
Torch, execute models, or access datasets. The publication-time output is
[VERIFICATION.json](../results/mingru-cpu-upstream-20261018-v1/VERIFICATION.json).

Historical checkpoint replays passed with zero per-window CE error in the
saved receipts. This import does **not** rerun those evaluations. Checkpoints,
optimizer/RNG state, and exact token arrays remain in contributor-held archives
outside Git; names and SHA256 values are in each study's `PRESERVATION.json`.
Those payloads are required for full training or saved-checkpoint replay, so a
source checkout alone is not a complete numerical-reproduction package.

To regenerate figures/reports, use the per-study `report.py` with Matplotlib.
To rerun training, follow each original report in a separate output copy with
the exact hash-bound data and preserved sibling dependencies. Do not overwrite
completed records. Wall-time schedules depend on hardware speed, so a fresh
training run is not expected to be bitwise identical.

This publication performs no new training, model evaluation, official-test
access, cloud provisioning, architecture promotion, or automatic merge.
