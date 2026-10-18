# Current research handoff

Updated 2026-10-18 UTC. Active run: `replay-retention-20261018-v1`, frozen before experiment verification/training. No results yet.

## Registered next stage

Six arms: exact backprop, frozen feedback and adaptive feedback, each with FIFO or mixed replay. Both policies store 512 examples and train on 16 current +16 replay examples. Mixed keeps 256 old examples and a 256-example recent queue, drawing 8+8. FIFO keeps a rolling512 queue and draws16. Historical labels stay unchanged, including conflicts after rule changes. One fixed mixture, fresh seeds801791/2/3 and pilot97987; no tuning.

Primary: adaptive_mixed versus frozen_mixed, same >=5% mean CE reduction/no-negative-seed and <=2pp old-window/final retention-loss gates. Source/protocol must be committed before verification and training. At most3 single-thread workers,900 aggregate CPU seconds, no GPU/cloud/official-test access.

- [Protocol](../results/replay-retention-20261018-v1/PROTOCOL.json)
- [Source hashes](../results/replay-retention-20261018-v1/REGISTRATION.json)
- [Reproduction](../results/replay-retention-20261018-v1/README.md)

On resume inspect VERIFY/PILOT/MAIN receipts and per-seed results before starting anything. Do not rerun an active or completed run or overwrite evidence.

## Previous completed stage

[magnitude-feedback-20261018-v1 report](../reports/magnitude-feedback-20261018-v1.md): adaptive feedback lowers post-change CE by26.45% after same-state magnitude matching, but all3 seeds fail retention (final-old losses4.91–8.26pp). All24 replays exact. Exact source and small checkpoints are fully preserved. No promotion.

The earlier bounded-feedback study still has a source/checkpoint preservation gap after the prior workspace reset. Its findings are transcriptions, not original bytes. This stage derives from the fully preserved magnitude study.

Use this research scratch channel and existing project folder. Keep files small and individual; no new cumulative archive. Freeze source and protocol before each new run.
