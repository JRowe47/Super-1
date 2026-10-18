# Current research handoff

Updated 2026-10-18 UTC / 2026-10-17 America/Denver.

## Latest result

Run: `bounded-feedback-20261018-v1`.

Bounding learned feedback prevented the earlier numerical blow-ups in all 12 bounded learned-feedback main runs. With bounds throughout warmup, continued feedback adaptation reduced primary post-change cross-entropy by **28.31%** versus freezing, improving all three seeds. Final still-valid old accuracy fell from **94.903% to 90.365%**; the two-point retention guard failed in every seed. **No promotion.** Exact backpropagation remains strongest overall.

- [Report](../reports/bounded-feedback-20261018-v1.md)
- [Recorded metrics](../results/bounded-feedback-20261018-v1/RECORDED_SUMMARY.json)
- [Protocol record](../results/bounded-feedback-20261018-v1/PROTOCOL_RECORD.json)
- [Execution and preservation status](../results/bounded-feedback-20261018-v1/TERMINAL_RECORD.json)

## First action on resume

Recover the completed run directory `auto-research/results/bounded-feedback-20261018-v1` from the prior workspace if it is accessible. Export its exact source, original registration/protocol, compact result records and necessary replay states individually. Verify against the original registration before claiming a complete reproducible release.

Training and all 27 main plus 9 pilot replays finished before the workspace returned `environment_offline`. This Git update contains transcribed findings, not the original source/receipt/checkpoint bytes. Original source hashes are unresolved here; do not invent them or claim these records reproduce the run.

The preceding study is `adaptive-feedback-20261018-v1`, saved in `curve_feedback_experiments_2026-10-18.zip` in Library. The earlier seven-study handoff is `curve_online_learning_handoff_2026-10-18.zip` in the Geometric Self Attention folder. Reuse those existing copies when needed. The user cancelled further cumulative archive work.

## Next scientific question

Does adaptive feedback help after matching update magnitude? Its sampled update norm is larger, while directional alignment is not consistently better. Freeze a magnitude-matched frozen-feedback control first; then investigate a separate replay-based retention intervention. Keep exact hidden gradients diagnostic only, preserve the same retention gate, and retain every seed and failed arm.

Neither follow-up is implemented or running. Prioritize preserving the exact implementation over reconstructing a supposedly identical run from prose.
