# Current research handoff

Active authorized run: `refresh-replay-20261018-v1`. Preregistered, execution pending. Nine arms: FIFO/permanent/refreshed replay crossed with backprop/frozen/adaptive feedback. Refresh1 bank row per update,256 recent+256 history and8+8 sampling fixed. Fresh main seeds901801..3, pilot98997. Same joint CE/retention gate,900 aggregate CPU-second envelope, max3 one-thread workers. Source publication/readback precedes verification and optimization.

Protocol: [PROTOCOL.json](../results/refresh-replay-20261018-v1/PROTOCOL.json). Registration: [REGISTRATION.json](../results/refresh-replay-20261018-v1/REGISTRATION.json). Completed results will be placed alongside these files; do not infer a completed result from this preregistration.

The previous permanent bank improved retention but sharply harmed changed-fact learning. This run tests whether a causal age bound improves that tradeoff. No phase-aware refresh or relabeling. If the joint gate fails, pivot to feedback plasticity instead of extending replay-rate searches.

## Prior completed handoff

# Current research handoff

Updated 2026-10-18 UTC. Latest completed run: `replay-retention-20261018-v1`.

## Latest result and decision

Permanent old-example replay improves retention but harms adaptation. Adaptive mixed replay raises final still-valid-old accuracy **88.19%→91.57%**, while changed-rule accuracy falls **73.05%→49.09%** and primary post-change CE rises **30.71%** versus adaptive FIFO.

Within mixed replay, adaptive feedback still beats frozen feedback by **21.94%** mean paired post-change CE reduction (15.09%,23.85%,26.89%). Final-old losses **3.52,5.25,5.19pp** against frozen mixed fail the unchanged 2pp guard in every seed; old-window losses also fail. **No promotion.** Mixed replay hurts changed-rule learning for BP and frozen feedback too. BP/FIFO remains strongest on final balanced accuracy 90.81%.

- [Report](../reports/replay-retention-20261018-v1.md)
- [Frozen source/reproduction](../results/replay-retention-20261018-v1/README.md)
- [Protocol](../results/replay-retention-20261018-v1/PROTOCOL.json) and [registration](../results/replay-retention-20261018-v1/REGISTRATION.json)
- [Metrics](../results/replay-retention-20261018-v1/SUMMARY.json), [endpoint components](../results/replay-retention-20261018-v1/ENDPOINT_COMPONENTS.json)
- [Terminal](../results/replay-retention-20261018-v1/TERMINAL.json), [checkpoint identities/hashes](../results/replay-retention-20261018-v1/ARTIFACT_MANIFEST.json)

All 18 main and 6 pilot arms completed; all 24 full online replays bitwise exact; original numerical tolerances pass. Both replay policies hold 512 examples and use 16 current+16 replay samples. The mixed policy pins 256 old examples and keeps 256 recent, sampling 8+8; historical labels are never refreshed. It exposes 3,736–4,948 stale binary targets versus 30–54 for FIFO. This supports a stale-memory concern but does not isolate it from changed data allocation/recent coverage.

Source/protocol committed before execution at 34f559d1454a067d175e55c372f3b20387b2423d. No source or gate changes, no execution failures. Charged experiment CPU 199.535s. Four NPZ state files total 754,883 bytes, remotely downloaded and SHA256 verified. Temporary runtime removed; exact pins in RUNTIME.json. All jobs stopped.

## Next stage, proposed and not running

Test a causally refreshed long-history bank with the same 256 recent+256 long-history storage and 8+8 replay samples: replace one old-bank row after each update using one randomly selected current example, fixed circular replacement. After initial turnover this caps long-history age at 256 updates without phase metadata or oracle relabeling. Compare FIFO/permanent/refreshed policies under adaptive/frozen/BP, fresh 3 paired seeds, one preregistered refresh rate, unchanged >=5% effect and <=2pp retention gates. Keep the permanent-bank control to isolate refresh from storage/sample allocation. Estimated 250–450 aggregate CPU seconds, standing 900s maximum with at most 3 single-thread workers; no GPU/cloud/official-test access.

If this cannot improve the joint tradeoff, reconsider feedback plasticity rather than keep tuning replay. Freeze and publish source/protocol before any new run. Check receipts and worker state before resuming; do not duplicate completed experiments.

## Continuity and artifact policy

Use this research scratch channel and the existing project folder. Keep individual files small; no cumulative archive. Previous [magnitude study](../reports/magnitude-feedback-20261018-v1.md) found a 26.45% adaptation gain after magnitude matching but failed retention; its source/checkpoints are fully preserved. The older bounded-feedback study remains a post-run transcription with missing original bytes after a workspace reset. Do not claim those old bytes have been recovered.
