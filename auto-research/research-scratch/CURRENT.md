# Current research handoff

Latest completed run: **refresh-replay-20261018-v1**. **No promotion: retention gate failed.** Nothing is running.

[Report](../reports/refresh-replay-20261018-v1.md) · [Summary](../results/refresh-replay-20261018-v1/SUMMARY.json) · [Protocol](../results/refresh-replay-20261018-v1/PROTOCOL.json) · [Terminal](../results/refresh-replay-20261018-v1/TERMINAL.json) · [Checkpoints](../results/refresh-replay-20261018-v1/ARTIFACT_MANIFEST.json).

## Latest finding

Nine arms cross backprop/frozen/adaptive feedback with FIFO512, permanent256+256 mixed replay, and causally refreshed256+256 replay. Both mixed modes sample8+8; refreshed bank replaces one circular row per update using a uniform current example with observed label. No phase or oracle relabeling.

Fresh main seeds901801/901802/901803, pilot98997. Primary adaptive_refreshed versus frozen_refreshed CE reductions26.11/25.72/23.46%, mean25.0964%, SD1.4345pp. Old-window deltas-6.78/-10.14/-7.04pp; final-old-4.35/-3.79/-1.23pp. Effect gate passes; retention requires both measures>=-2pp in EVERY seed and fails all3 via old-window.

Relative to adaptive permanent replay, refresh reduces CE11.70%, restores21.48pp final changed accuracy and7.02pp balanced accuracy, but loses1.02pp final-old and6.20pp old-window accuracy. Compared with adaptive FIFO it still raises CE13.17%, with only0.28pp final-old gain and2.34pp changed accuracy loss. Refresh also helps BP/frozen changed accuracy, so it is not a feedback-specific rescue.

Obsolete replay targets FIFO/permanent/refreshed:901801=38/3244/164;901802=24/3544/132;901803=54/3796/126. Refreshed labels become current after complete turnover, but novel-context learning also improves before any rule changes: age/content and staleness are not independently isolated.

## Evidence and authority

Source frozen and published before execution: `4faa9eb28162de359366e77a7ff1e980dc8826fe`. Protocol SHA256 `7def232286ce614b3d3b34e3c2717e1ce789b5d35067bea4d8d7ed6fc5015235`. Verification/pilot commit `386d8addee0316275094e2c1f8554596077f54c2`. All27 main+9 pilot arms completed; all36 full online replays bitwise exact at all4 phases/final state. Numerical tolerances unchanged; independent refresh references passed.

261.082 aggregate CPU seconds of900; no experiment failures or source changes. Four compact checkpoints total1,025,296bytes, remote SHA256 readback verified. Temporary runtime removed; all workers exited. No official tests, external datasets, GPU/cloud/paid compute or language training.

Work only in JRowe47/Super-1, branch research/curve-inception-20261017-v1, existing draft PR3. Do not merge or target upstream. Use auto-research/research-scratch for stable handoff; preserve previous results and minimize artifacts. Existing project folder contains each seed's NPZ file, not a new cumulative archive. Current local staging tree is not a full Git checkout; use current remote branch as authority. The older bounded-feedback source gap remains unresolved; recent runs are fully preserved.

## Next proposed stage

Pivot from replay allocation to **feedback plasticity**. Fix FIFO replay512/16, common warmup, forward lr/decay, exposure and spectral cap. Four arms: frozen, normal adaptive, one slow-feedback setting, backprop. A candidate single fixed setting is0.1 times the entire feedback SGD update including decay after warmup; leave forward updates unchanged. Freeze source/equations and three fresh paired main seeds plus separate pilot BEFORE any experiment execution. No rate sweep or phase metadata. Primary slow-versus-frozen:>=5% mean paired CE gain, no negative seed, both old-window and final-old losses<=2pp in every seed; full original numerical/source/preservation/replay gates. Compare with normal adaptive to expose whether apparent retention simply sacrifices the useful effect.

Estimated150–300 CPU seconds, within standing900 aggregate CPU seconds, max3 concurrent one-thread workers and four small state files. Runtime restoration uses pinned versions in latest RUNTIME.json; remove it after durable evidence checks. Follow-on is proposed only, not implemented/launched; await the user's continuation. Do not enlarge the compute scope or start language/1B work.

## Earlier checkpoints

- [Permanent replay retention](../reports/replay-retention-20261018-v1.md): adaptive mixed vs frozen21.94% CE gain, retention fails all3; permanent bank raises adaptive CE30.71% versus FIFO.
- [Magnitude feedback](../reports/magnitude-feedback-20261018-v1.md): magnitude-controlled CE advantage26.45%, retention still fails. Do not infer closer BP gradients.
- Historical bounded-feedback run lost source/checkpoints; keep its transcription caveat. Do not describe it as exactly reproduced.
