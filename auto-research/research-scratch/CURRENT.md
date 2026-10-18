# Current research handoff

Active authorized run: **feedback-family-20261018-v1**, preregistered and awaiting source publication/readback before verification or training. Five arms: BP, frozen, B-only, V/U-only, full learning-only. Online feedback decay off; inactive matrices literally frozen. FIFO512/16, full rates, common warmup, forward rule, cap2 and exposure/endpoints fixed. Primary B-only versus frozen uses unchanged5%/no-negative-seed and per-seed2pp old-window/final-old guards. Fresh main seeds1201831..3; pilot101927. No outcome yet.

Same900 aggregate CPU-second budget, max3 one-thread workers; no GPU/cloud/language expansion. [Protocol](../results/feedback-family-20261018-v1/PROTOCOL.json) · [Registration](../results/feedback-family-20261018-v1/REGISTRATION.json).

## Prior completed handoff

# Current research handoff

Latest completed run: **feedback-decay-20261018-v2**. **No promotion: effect passes, retention fails all three main seeds. Nothing is running.**

[Report](../reports/feedback-decay-20261018-v2.md) · [Summary](../results/feedback-decay-20261018-v2/SUMMARY.json) · [Protocol](../results/feedback-decay-20261018-v2/PROTOCOL.json) · [Terminal](../results/feedback-decay-20261018-v2/TERMINAL.json) · [Checkpoints](../results/feedback-decay-20261018-v2/ARTIFACT_MANIFEST.json).

## Latest result

Fixed FIFO512/16 and five arms: frozen, decay-only, learning-only, normal adaptive and BP. Full feedback rates; identical4096-step common warmup, forward optimizer, cap2, stream/exposure. Main seeds1101821/1101822/1101823 each2048 online steps per arm; pilot100917 with256 steps.

Primary normal versus decay-only CE reductions29.0876/23.0375/28.9714%, mean27.0322%, SD3.4600pp. Old-window deltas-9.2494/-6.4593/-6.4907pp; final-old-3.0134/-3.5714/-2.6228pp. Effect passes, both2pp retention guards fail every seed. Normal versus frozen improves24.4650% but also fails retention all3.

Decay-only worsens CE versus frozen by3.5722% on average and is worse in every seed. Learning-only improves26.8814% versus frozen, but old-window loss8.06–11.89pp and final-old4.52–4.69pp still fail. Data-dependent feedback learning adds benefit beyond passive decay in these worlds; retention remains unresolved. Benefits remain concentrated in changed-rule learning; novel-window CE worsens versus frozen. Mean final changed accuracy: BP89.58%, normal73.96%, learning-only73.57%, frozen48.31%, decay-only35.94%. BP final balanced93.92% remains strongest.

## Failure, repair and evidence

v1 (`8d5a000f7572962bc4eb29e072bbe71b1bb5f348`) failed verification before any training: FP64 seed15817 norm2.0000000000000004 exceeded strict cap2. Preserve its failed receipt/diagnosis; VERIFICATION.json there is inherited-checks-only, not overall PASS. [Failure report](../reports/feedback-decay-20261018-v1.md).

v2 keeps the original projection and adds common decoded readback/overshoot-only inward correction for every updated-feedback arm and warmup. Cap2, original FP32/64 tolerances and all scientific gates unchanged. Known failure retained in regression; all checks pass. Frozen v2 source/protocol commit `801d1e3147c47bd5b6004f1d343ea9a2c9185d6b`; protocol SHA256 `9631d01c94420a3b22cc4e721897cdebf80270dd58e44d25eaa7faa0ac47fa44`. Pilot/verification commit `cd316e68f256a4dd4236ed3e47bb09761db217c6`. All18 v2 and16 v1 registered files remain unchanged.

All15 main+5 pilot arms complete,20 full online replays bitwise exact. Four checkpoints total921,878 bytes, remote SHA256 verified. 250.633 aggregate CPU seconds/900, including failedv1 verification/diagnosis and import canary once; max3 one-thread workers. All workers exited, runtime/readback copies removed. No official tests, external datasets, GPU/cloud/paid compute, language training or extra seed/rate search.

## Stable working instructions

Work only in JRowe47/Super-1, branch research/curve-inception-20261017-v1, existing draft PR3. No merge/upstream submission. Use auto-research/research-scratch for stable handoff and all run files under auto-research/results and reports. Project folder holds individual NPZ checkpoints; no cumulative archive or duplicate repo export. Local worktree is a staging tree, not a full clone; remote branch is authority. Preserve original archives and older evidence. Historical bounded-feedback source/checkpoints remain lost; never claim exact reconstruction.

## Next proposed stage

Proposed only: separate learning of error-transport matrices B from learning of feedback encoder/readout matrices V/U, with online feedback decay disabled in all local arms to make the switches literal freezes. Fixed FIFO, original rates, cap, common full warmup, exposure and endpoints. Arms frozen, B-only, V/U-only, full learning-only, and BP. Primary B-only versus frozen with unchanged >=5% mean CE/no negative seed and <=2pp per-seed old-window/final-old guard. Three fresh paired main seeds plus a separate pilot; no rate sweep. Test whether B plasticity can retain adaptation while fixed V/U reduces forgetting; do not assume it will. Not implemented or running.

Groups: B-only learns B1/B2 while V1/V2/U0/U1 remain fixed; V/U-only does the reverse. Common warmup remains full normal feedback learning+decay; online feedback learning uses full original rate and common repaired cap. Frozen and inactive families receive neither feedback learning nor decay. Full learning-only is the positive adaptation/reference control. This is a matrix-family intervention; B changes still propagate through the coupled feedback path.

Expected250–450 aggregate CPU seconds within standing900s, max3 one-thread workers, four compact NPZ checkpoints. Reuse pinned versions in latest RUNTIME.json, publish exact source/protocol before checks/training, preserve failures and all endpoints. Await the next user continuation; no proposed experiment is launched. No expansion to language/1B/cloud work is implied.

## Earlier checkpoints

- [Slow feedback](../reports/slow-feedback-20261018-v1.md): retention passes but1.68% CE gain misses5%; slowing learning and decay together surrenders most adaptation.
- [Refreshed replay](../reports/refresh-replay-20261018-v1.md):25.10% CE gain, retention fails; refreshing helps permanent-bank adaptation but remains slower than FIFO.
- [Permanent replay](../reports/replay-retention-20261018-v1.md): old retention improves but changed-rule learning worsens sharply.
- [Magnitude control](../reports/magnitude-feedback-20261018-v1.md):26.45% CE gain survives matched update magnitude, retention fails; BP alignment improvement is not established.
- Earlier bounded-feedback missing-source transcription remains explicitly caveated.
