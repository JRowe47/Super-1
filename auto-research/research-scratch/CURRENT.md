# Current research handoff

Active authorized run: **feedback-decay-20261018-v1**, preregistered, execution pending. Fixed FIFO512/16, original full feedback rate; five arms=frozen/decay-only/learning-only/normal adaptive/backprop. Primary normal adaptive vs decay-only uses unchanged CE/retention thresholds; frozen reference remains separately visible. Fresh main seeds1101821..3 and pilot100917. Source publication/readback precedes verification and optimization.900 aggregate CPU seconds, max3 one-thread workers, no rate sweep.

[Protocol](../results/feedback-decay-20261018-v1/PROTOCOL.json) · [Registration](../results/feedback-decay-20261018-v1/REGISTRATION.json). No outcome yet; no language/architecture promotion is implied by this mechanism control.

## Prior completed handoff

# Current research handoff

Latest completed run: **slow-feedback-20261018-v1**. **No promotion: useful-effect gate failed; retention passed.** Nothing is running.

[Report](../reports/slow-feedback-20261018-v1.md) · [Summary](../results/slow-feedback-20261018-v1/SUMMARY.json) · [Protocol](../results/slow-feedback-20261018-v1/PROTOCOL.json) · [Terminal](../results/slow-feedback-20261018-v1/TERMINAL.json) · [Checkpoints](../results/slow-feedback-20261018-v1/ARTIFACT_MANIFEST.json).

## Latest result

Four arms share FIFO512/16, current16, unchanged4096-step warmup, forward lr/decay and spectral cap2. Slow adaptive uses0.1 of the entire feedback SGD drive including decay before projection. Controls: frozen, normal adaptive, backprop. Fresh main seeds1001811/1001812/1001813, pilot99907; main2048 online steps per arm, pilot256.

Primary slow-versus-frozen CE reductions1.9965/2.8885/0.1649%, mean1.6833%, sample SD1.3885pp: below5% minimum. Old-window deltas-1.1265/-1.3602/-0.9521pp, final-old-0.2790/-0.5022/-0.3906pp: every retention comparison is within2pp. Joint gate fails.

Normal adaptive gains25.3142% CE versus frozen but fails retention all3. Slow preserves7.248% of normal's benefit when averaging per-seed gain fractions, gains2.9762pp final-old accuracy versus normal, but loses31.6406pp final changed accuracy and10.7050pp balanced accuracy. Mean final changed accuracy: frozen36.46%, slow37.63%, normal69.27%, BP81.12%. Normal's benefit is concentrated in the changed-rule window; novel-window CE is worse than frozen. No rate/comparator selection.

## Evidence and authority

Source/protocol published/readback before experiment: `272e90a306793d60ba9265bcf6b1ce21985e819a`; protocol SHA256 `dc30899a06125b20216eaf4dabbb9a2ca367b0d9c7eeb2454732fd431c5c2674`. Pilot/verification commit `3b95cd068c33e604f67da12e1349d6efe41a11da`. All12 main+4 pilot arms complete;16 full online replays bitwise exact at four phases/final state. Original FP32/FP64 tolerances pass, with independent dense slow-step and exact multiplier0/1 endpoint/trajectory checks.

163.694 aggregate CPU seconds/900, max3 one-thread workers; no experiment failures or source changes. Pinned dependency setup had transient network timeouts resolved by automatic retries before experiment execution. Four compact checkpoints total703,631bytes, remote SHA256 readback verified. All workers exited; temporary runtime/readback copies removed. Source and report hashes in global MANIFEST.sha256.json. No official tests, external datasets, GPU/cloud/paid compute or language training.

Work only in JRowe47/Super-1 on research/curve-inception-20261017-v1, existing draft PR3. No merge/upstream submission. Use auto-research/research-scratch for stable back-and-forth; all per-run files under auto-research/results and reports. Existing project folder holds individual NPZ checkpoints; no new cumulative archive. Local worktree is a staging tree, not a full checkout; current remote branch is authority. Keep original archives and older evidence intact. Earlier bounded-feedback source/checkpoints remain lost; never claim exact reconstruction of that run.

## Next proposed stage

Separate **feedback local learning from feedback weight decay**, both of which were scaled in this experiment. Fixed FIFO512/16, original full feedback step0.05, common warmup, forward optimizer, cap2, exposure and endpoints.2x2: feedback data-dependent drive on/off xfeedback decay on/off, giving frozen(neither), decay-only, learning-only, normal adaptive(both), plus BP. This tests a missing simpler explanation: passive feedback decay may account for some apparent adaptive-feedback benefit.

Proposed primary: normal adaptive versus decay-only, with unchanged>=5% mean paired post-change CE reduction, no negative seed, and both old-window/final-old loss<=2pp in every seed. Frozen and learning-only are explanatory controls; report all contrasts, do not select a favorable primary afterward. Fresh3 paired main seeds and separate mechanical pilot; prepublish exact source/protocol, original tolerances, all full replays and compact remote-preserved states. No rate sweep.

Expected200–350 aggregate CPU seconds within standing900s, max3 one-thread workers; four small NPZ files. Restore exact versions from latest RUNTIME.json and remove temporary runtime after preservation. Follow-on is proposed only, not implemented/launched; await the user's continuation. No permission for language/1B/cloud expansion is implied.

## Earlier checkpoints

- [Refreshed replay](../reports/refresh-replay-20261018-v1.md): primary adaptive/frozen CE gain25.10%, retention fails; refresh helps permanent-bank adaptation but remains slower than FIFO.
- [Permanent replay](../reports/replay-retention-20261018-v1.md): improved old retention but sharply impaired changed-rule learning.
- [Magnitude control](../reports/magnitude-feedback-20261018-v1.md):26.45% CE gain survives matched update magnitude, retention fails; do not infer improved BP alignment.
- Earlier bounded-feedback missing-source transcription stays explicitly caveated.
