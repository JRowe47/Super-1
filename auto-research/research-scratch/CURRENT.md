# Current research handoff

Latest completed run: **feedback-readout-20261018-v1**. **No promotion: U1-only effect passes; old-window retention fails all 3 main seeds, final-old retention passes all 3. Nothing is running.**

[Report](../reports/feedback-readout-20261018-v1.md) · [Summary](../results/feedback-readout-20261018-v1/SUMMARY.json) · [Protocol](../results/feedback-readout-20261018-v1/PROTOCOL.json) · [Terminal](../results/feedback-readout-20261018-v1/TERMINAL.json) · [Checkpoints](../results/feedback-readout-20261018-v1/ARTIFACT_MANIFEST.json).

## Latest outcome

Five arms: BP, frozen, U0-only, U1-only, both-U. B/V fixed and online feedback decay off. Full rates, FIFO 512/16, common4096 warmup, W optimizer, cap 2 and data/exposure unchanged. Main seeds1301841/1301842/1301843 each 2048 online steps per arm; pilot 102937 with 256.

Sole primary U1-only versus frozen: CE reductions20.9574/13.8837/18.1288%, mean 17.6566%, SD 3.5604pp. Old-window deltas-7.0940/-6.2779/-6.0163pp fail2pp in all 3; final-old+0.4464/+0.2790/-0.0558pp passes all 3. Final recovery does not substitute for accessibility during adaptation; final mixed exposure includes old data again.

Explanatory U0-only: mean 9.8430% CE reduction, SD 0.6148pp; old-window-2.8878/-3.1215/-1.3009pp fails2; final-old-1.8415/-0.6138/-0.3906pp passes all 3. Closest observed tradeoff, but no promotion or primary substitution. Both-U:28.6811% CE reduction, SD 1.0172pp; old-window-11.0944/-9.4971/-8.7402pp fails all 3; final-old-1.2835/+0.7254/+0.3906pp passes all 3. Final balanced accuracy: BP92.10%, frozen71.59%, U0-only 79.65%, U1-only 73.98%, both-U81.02%. U1-only early CE gain is concentrated in changed-rule windows; novel-window mean slightly worsens.

Both-U versus preserved V/U-only W/U/B updates and 12-step trajectories are now checked bitwise despite differing V, with B fixed. This confirms the dependency inference on registered verification cases, not a second main experiment. W0 still learns under U1-only; early features are not frozen. Cross-stage means use different worlds and must not be treated as paired improvements.

## Evidence and authority

Exact source before experiment execution:`38d685eab9df1306168fd9b068c61456dcf7b29e`. Pilot/verification:`71470554ea17ac257bba8739ad01cd43174d1f9d`. Main metrics/checkpoint audit:`f8fd1d3a656bb8eb80aefe3f370a53ab3a74c918`. Protocol SHA256:`605f328f1e6d1ed4744a9ee6b93b9c929f4ad050ffbe8ad10a4be1102ae76fd3`. All21 frozen files unchanged. Original absoluteFP32 2e-6/FP64 2e-12 limits, prior cap regressions and strict cap 2 pass. New readout dense errors2.98023224e-7/4.44089210e-16. All20 complete online replays exact at every phase/final state; inactive matrices bitwise fixed. All15 main+5 pilot arms complete without experiment failures, retries or source changes. Four active-U checkpoint overlays:469,853bytes,28arrays each; remote SHA256/schema verified. CPU 296.103s/900, max3 one-thread workers; all workers exited, runtime/readbacks removed.

Use only JRowe47/Super-1 branch research/curve-inception-20261017-v1 and existing draft PR3. No merge/upstream submission for this branch. Stable coordination in auto-research/research-scratch; run source/evidence under auto-research/results and reports. Local worktree is a partial staging tree, not a clone; remote branch is authority. Checkpoints are individually preserved in the existing Geometric Self Attention folder. No cumulative archive or duplicate repo export. Keep original uploaded archive and prior artifacts intact. Earlier bounded-feedback source/checkpoints remain lost; do not claim recovery. Original feedback-decay-v1 cap-verification failure remains preserved. No official tests/external data/GPU/cloud/paid compute/language training.

## Next proposed stage

Proposed only: test one fixed half-rate U0-only setting (beta=0.5 on the U0 local learning drive only), with B1/B2/V1/V2/U1 fixed at common warmup and online feedback decay off. Four arms: BP, frozen feedback, original full-rate U0-only, half-rate U0-only. Sole primary half-rate U0-only versus frozen, unchanged >=5% mean paired CE reduction/no negative seed and <=2pp per-seed old-window/final-old loss. Full-rate U0-only is the positive adaptation control. Keep W optimizer, common4096 warmup, FIFO 512/16, strict cap 2, data/exposure and endpoints. Three fresh paired main seeds plus separate mechanical pilot. Verify beta 1 matches preserved U0-only and beta 0 matches frozen bitwise, independent dense half-rate steps, inactive matrices, strict cap and all full replays. No added rate grid or outcome-driven tuning. Expected200-350 aggregate CPU seconds inside 900, max3 one-thread workers; four compact checkpoint files. This follow-on is not implemented or running; await continuation.

Rationale: U0-only cleared5% adaptation while two old-window losses only modestly exceeded2pp. Halving just that readout drive might reach the joint target, or might lose too much adaptation; neither is assumed. This is a new prospectively registered rate counterfactual on fresh worlds, not retroactive selection for this stage. Only the one coarse half-rate setting is proposed, no sweep or additional rates/seeds after outcomes. Common warmup stays full rate; beta applies only online to U0. At beta 0 all feedback is frozen while W still learns; beta 1 must reproduce the current U0-only implementation exactly.

Restore pinned RUNTIME.json packages; freeze/publish/readback source before numerical checks, pilot then3 main seeds, full exact replays, compact checkpoint preservation and cleanup. Preserve failures before any separately registered repair; never relax tolerances/gates. Await the next continuation; no next run is implemented or running.

## Earlier checkpoints

- [B versus V/U](../reports/feedback-family-20261018-v1.md): B-only 6.96% CE improvement with severe retention failure; V/U-only 22.37% but old-window retention fails all 3 and final-old one. Structural U-only inference motivated this stage.
- [Learning versus decay](../reports/feedback-decay-20261018-v2.md): learned feedback beats decay-only 27.03%, retention fails; decay alone worsens versus frozen. Earlier v1 cap failure preserved separately.
- [Slow whole feedback](../reports/slow-feedback-20261018-v1.md):0.1 scale passed retention but1.68% CE gain missed 5%; this is not the proposed U0-only/zero-decay half-rate condition.
- [Refreshed replay](../reports/refresh-replay-20261018-v1.md), [permanent replay](../reports/replay-retention-20261018-v1.md), [magnitude control](../reports/magnitude-feedback-20261018-v1.md): earlier adaptation/retention tradeoffs; no joint promotion or BP-alignment improvement established.
