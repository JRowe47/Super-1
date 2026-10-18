# Current research handoff

Updated 2026-10-18 UTC. Latest completed run: `magnitude-feedback-20261018-v1`.

## Latest decision

Adaptive feedback reduces post-change CE by **26.45%** against a frozen-direction control with same-state adaptive step magnitudes (three seed reductions 27.00%,26.26%,26.08%). Final still-valid-old accuracy is **86.90% versus93.42%** for that control. Losses 4.91–8.26pp fail the unchanged 2pp retention gate in every seed. **No promotion.** Step size alone does not explain the adaptation gain in this synthetic task. Exact BP still has the strongest final balanced accuracy 92.44% versus 81.04%.

- [Report](../reports/magnitude-feedback-20261018-v1.md)
- [Frozen source and reproduction](../results/magnitude-feedback-20261018-v1/README.md)
- [Protocol](../results/magnitude-feedback-20261018-v1/PROTOCOL.json) and [registration/hashes](../results/magnitude-feedback-20261018-v1/REGISTRATION.json)
- [Metrics](../results/magnitude-feedback-20261018-v1/SUMMARY.json) and [terminal receipt](../results/magnitude-feedback-20261018-v1/TERMINAL.json)
- [Checkpoint locations/hashes](../results/magnitude-feedback-20261018-v1/ARTIFACT_MANIFEST.json)

All 18 main+6 pilot runs finished; all 24 full online replays are bitwise exact. Original FP32/FP64 tolerances and every magnitude audit pass. Charged experiment CPU 226.078s. Four individual checkpoint files total 1,910,052 bytes; uploaded copies were downloaded and hash-verified. No jobs remain. Exact source was committed before execution at 644d69ed8e797e3e06de0af416fa561a8bd6f148. A missing-dependency pilot import failure happened before data/updates, was retained, and was repaired through a committed runtime-only amendment. No source or thresholds changed.

## Next stage, not running

Test a fixed-budget replay retention intervention: split the same 16 replay slots between recent FIFO and an old-example reservoir. Include adaptive/frozen × FIFO/mixed replay controls, fixed current exposure, fresh 3 paired seeds, unchanged effect/retention gates and one preregistered mixture. Keep historical labels, including stale conflicts; do not use heldout/phase information to discard or relabel memory. Budget approximately 250–450 aggregate CPU seconds, no more than the standing 900s/3 single-thread workers. Freeze/publish source and protocol before executing.

Use this directory for discussion and handoffs. Keep artifacts small and individual. The temporary restored runtime was removed after completion; restore exact versions from RUNTIME_AMENDMENT.json when needed. Reuse the existing repo and project folder; do not create cumulative archives or duplicate old bundles.

## Historical preservation caveat

`bounded-feedback-20261018-v1` remains a post-run transcription: its original source/registration/checkpoints were lost at the prior workspace reset. Its observed 28.31% gain and retention failure remain [separately documented](../reports/bounded-feedback-20261018-v1.md). The present run uses a fresh implementation from the hash-verified preserved predecessor in `curve_feedback_experiments_2026-10-18.zip`; it does not repair the old missing bytes or claim exact reproduction of that study.
