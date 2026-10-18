# Current research handoff

Latest completed run: **feedback-family-20261018-v1**. **No promotion: B-only effect passes, both retention guards fail all three main seeds. Nothing is running.**

[Report](../reports/feedback-family-20261018-v1.md) · [Summary](../results/feedback-family-20261018-v1/SUMMARY.json) · [Protocol](../results/feedback-family-20261018-v1/PROTOCOL.json) · [Terminal](../results/feedback-family-20261018-v1/TERMINAL.json) · [Checkpoints](../results/feedback-family-20261018-v1/ARTIFACT_MANIFEST.json).

## Latest result

Five arms: BP, frozen, B-only, V/U-only and full learning-only. Online feedback decay off; inactive matrices remain bitwise frozen. Same FIFO512/16, full rates, common4096-step warmup, forward rule, cap2 and data/exposure. Main seeds1201831/1201832/1201833 each2048 online steps/arm; pilot101927 with256.

Primary B-only versus frozen: CE reductions9.8871/3.1267/7.8729%, mean6.9622%, SD3.4710pp. Old-window deltas-8.8309/-4.8584/-6.3651pp; final-old-12.2210/-10.0446/-14.7321pp. Effect passes; both2pp guards fail every seed. B-only final changed accuracy40.76% is slightly below frozen41.54%; final balanced66.00% versus70.80%. Earlier CE benefit is not sustained final superiority.

V/U-only versus frozen improves22.3748% CE (SD0.0377pp), close to full learning21.8719%. V/U-only old-window deltas-6.9406/-5.1200/-7.5823pp fail all3; final-old-0.2790/+3.0134/-2.6228pp fails one. Mean final-old+0.0372pp must not be substituted for those per-seed failures. V/U-only final balanced82.62%, full78.78%, BP92.71%. Full loses3.5261pp old-window and4.3155pp final-old versus V/U-only. No secondary promotion.

Structural inference from preserved equations: with B fixed, V appears only in the disabled B-learning drive, so V changes cannot alter W/U/B trajectories. V/U-only's useful non-V behavior therefore comes from U learning; no new U-only experiment was run here. This motivates the next proposed readout-layer ablation.

## Evidence and authority

Exact source/protocol before execution: `553fb843756fa98ca534196f84902bf2d78aa863`; protocol SHA256 `b6d41588d346510748df8507d1448dc22a44183c192313b1e3576ebf4360f673`. Pilot/verification preserved at `036f4e6bf5485d8f5916231f7926e2b1610d6212`. All19 frozen files unchanged. Original FP32 absolute2e-6/FP64 absolute2e-12 gates pass; new family dense-step errors3.57628e-7/6.66134e-16. Inactive12-step trajectories and frozen/full endpoints bitwise exact. Prior cap-failure regression retained; strict cap2 unchanged.

All15 main+5 pilot arms complete without experiment failures/retries/source changes.20 full online replays exact at all phases/final states, every inactive matrix equal to warmup. Four compact active-family checkpoint overlays total729,783 bytes, remote SHA256 and schema verified. 273.358 aggregate CPU seconds/900, max3 one-thread workers. Workers exited and temporary runtime/readbacks removed.

Use only JRowe47/Super-1 branch research/curve-inception-20261017-v1 and existing draft PR3. No merge/upstream submission. Stable back-and-forth in auto-research/research-scratch; run files under auto-research/results and reports. Individual NPZ states in existing Geometric Self Attention project folder, no cumulative archive or duplicate repo export. Local worktree is a staging tree, not a full clone; remote branch is authority. No official tests, external data, GPU/cloud/paid compute or language training. Preserve original archives and older evidence. Earlier bounded-feedback source/checkpoints remain lost; never claim exact reconstruction. feedback-decay-v1 failed cap-verification evidence remains intact.

## Next proposed stage

Proposed only: separate feedback readout learning in U0 versus U1, with B1/B2/V1/V2 fixed at common warmup and all online feedback decay off. Arms frozen, U0-only, U1-only, both-U, and BP. Primary U1-only versus frozen, unchanged >=5% mean CE/no negative seed and <=2pp per-seed old-window/final-old guard. Keep full original rates/FIFO/warmup/cap/exposure/endpoints. Fresh3 paired main seeds and separate pilot. Verify both-U and preserved V/U-only have identical W/U/B trajectories despite differing V, as implied by the equations with B fixed. Test whether adapting only the feedback readout into the second hidden layer reduces forgetting. No rate sweep; not implemented or running.

U0 drives W0 learning through e1; U1 drives W1 learning through e2. The candidate fixes U0 while adapting U1, but W0 continues to learn through its fixed feedback decoder; this does not freeze early forward features. Both-U should match the prior V/U-only non-V trajectory because B is fixed. Verify that relationship before execution rather than assuming numerical equivalence. Keep the same forward optimizer and full common warmup; online B/V fixed and feedback decay off.

Expected250–450 aggregate CPU seconds within standing900s, max3 one-thread workers; four compact states. Restore exact packages from RUNTIME.json, prepublish/readback frozen source, retain all failure evidence, run the separate pilot before3 main seeds, preserve full replays/checkpoints then remove runtime. Await the next user continuation. No proposed follow-on is launched or language/1B/cloud budget implied.

## Earlier checkpoints

- [Learning versus decay](../reports/feedback-decay-20261018-v2.md): learned feedback beats decay-only27.03% CE but retention fails; decay alone worsens versus frozen. v1 cap failure was repaired separately without relaxing gates.
- [Slow feedback](../reports/slow-feedback-20261018-v1.md): retention passes but1.68% CE gain misses5%; most adaptation benefit is lost.
- [Refreshed replay](../reports/refresh-replay-20261018-v1.md):25.10% CE gain, retention fails; refreshing helps permanent replay but remains slower than FIFO.
- [Permanent replay](../reports/replay-retention-20261018-v1.md): old retention improves while changed-rule learning worsens sharply.
- [Magnitude control](../reports/magnitude-feedback-20261018-v1.md):26.45% CE gain survives matched update magnitude, retention fails; no improved BP alignment established.
- Older bounded-feedback missing-source transcription remains explicitly caveated.
