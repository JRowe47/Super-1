# Experiment report: feedback-readout-20261018-v1

**Completed; no promotion.** U1-only learning reduces early post-change cross-entropy by **17.66% versus frozen feedback**, but old-fact accuracy during the adaptation windows falls **6.02–7.09 percentage points** in every seed. Final-old accuracy is within the 2pp guard in all three seeds. The registered joint criterion still fails: later recovery cannot replace retention during adaptation.

U0-only is the closest explanatory tradeoff: **9.84%** CE improvement with old-window losses **1.30–3.12pp**, failing two seeds. Both-U improves **28.68%** but loses **8.74–11.09pp** old-window accuracy. All three learned-readout arms pass the final-old guard on these worlds; none passes both retention guards. No secondary result replaces the sole U1-only primary.

Approval: user “Go ahead with that”, accepting the preceding U0/U1 proposal. Source/protocol commit: `38d685eab9df1306168fd9b068c61456dcf7b29e`; protocol SHA256 `605f328f1e6d1ed4744a9ee6b93b9c929f4ad050ffbe8ad10a4be1102ae76fd3`. All 25 source-publication files were read back byte-equal before verification or training. [Protocol](../results/feedback-readout-20261018-v1/PROTOCOL.json) · [Registration](../results/feedback-readout-20261018-v1/REGISTRATION.json) · [Summary](../results/feedback-readout-20261018-v1/SUMMARY.json).

## Question and controls

Can learning only U1, the decoder of the second hidden layer's error signal, preserve useful adaptation with less old-fact loss than learning both readouts?

Five arms share the same stored model, common warmup, forward optimizer, stream, queries, FIFO memory and exposure. Exact backprop is the strong control. The four local arms cross learning U0 and U1 on/off, with B1/B2/V1/V2 fixed bitwise at common warmup and all online feedback decay off.

| Local arm | U0 learning | U1 learning | B and V learning | Online feedback decay |
|---|---|---|---|---|
| Frozen | Off | Off | Off | Off |
| U0-only | On | Off | Off | Off |
| U1-only | Off | On | Off | Off |
| Both-U | On | On | Off | Off |

U0 decodes e1 for W0; U1 decodes e2 for W1. **W0 still learns under U1-only** using its fixed U0 decoder. Thus this does not freeze early forward features. All forward weights keep step size 0.05 and decay0.001. Active U uses the original0.05 step and 0.8 local-rule factor. Common4096-step warmup retains original BP-forward/full-feedback-learning-and-decay. Inactive feedback remains exact after the shared repaired cap 2; no rate or cap is changed in this experiment.

The sole primary is U1-only versus frozen: >=5% mean paired relative CE reduction, no negative seed, and at most 2pp additional loss in both old-window and final-old accuracy for every seed. U0-only/both-U contrasts, interaction, endpoint decomposition and displacement are preregistered explanatory analyses. A smaller final loss cannot override the old-window guard.

The strongest alternative is that local forward learning still disrupts old knowledge during concentrated new-context/rule exposure even when U0 is fixed. A second alternative is that U0-only offers less adaptation but a better retention tradeoff. These arms distinguish the readout choices; they do not isolate every causal path inside the evolving network. [Rule mapping](../results/feedback-readout-20261018-v1/RULE_MAPPING.md).

## What ran and verification

Fresh main seeds **1301841, 1301842, 1301843** and separate mechanical pilot **102937**. Main:4096 shared warmup steps and 2048 online updates per arm in four 512-step phases. Pilot:same warmup plus 256 updates per arm. All15 main and 5 pilot arms completed without experiment exceptions, source changes after freeze, retries, seed exclusions or rate search. Pilot outcomes did not tune the main configuration.

This remains an11-input, four-binary-target synthetic threshold-world task: old contexts, novel contexts, changed rules and mixed exposure. Main heldout sets use128 examples per context; pilot 32. FIFO holds 512 examples and samples 16 before appending current 16. Historical labels remain unchanged, including obsolete replay targets:44/30/24 by main seed. Warmup exposes1,048,576 binary targets per seed; main online exposure262,144 per arm and pilot 32,768. No external corpus or official test was accessed.

All preserved Fourier-transform, local-equation, exact-gradient, component/family, repaired-cap and independent FIFO checks pass. The known earlier failed-cap case remains covered without changing the strict cap or original absolute tolerances. Six new seed/dtype cases cover bounded and active-projection states for all four readout masks. Maximum dense-step errors are **2.9802322e-07 FP32** and **4.4408921e-16 FP64**, below 2e-6 and 2e-12. All decoded feedback norms satisfy cap 2.

Same-state W updates are bitwise identical across masks. Every inactive feedback matrix remains bitwise frozen through12-step checks. Inputs are unmodified, outputs detached, active readouts change, nonboolean switches reject, and local paths avoid autograd. U1-only updates all forward layers in the checked trajectory. Frozen endpoint and trajectory match the preserved reference exactly.

The previously inferred relationship is now checked numerically: **both-U and preserved V/U-only have bitwise-identical W/U/B updates and 12-step trajectories**, even though V changes only in the latter. With B fixed, V enters only the disabled B-learning drive. This verifies the expected dependency structure on the registered verification cases; it is not another full main-world experiment or an independent replication seed. [Verification](../results/feedback-readout-20261018-v1/VERIFICATION.json).

All **20 complete online trajectories** replayed from serialized warmup and independently regenerated/hash-verified data. Four phase hashes, final full-state hashes and every serialized tensor match bitwise. Every inactive feedback matrix also equals common warmup at the end. Replays establish reproducibility, not additional independent samples. Verification/pilot evidence was preserved at `71470554ea17ac257bba8739ad01cd43174d1f9d`; full main metrics and checkpoint audit at `f8fd1d3a656bb8eb80aefe3f370a53ab3a74c918` before this report.

Charged experiment CPU is **296.103s of 900s**, including the import canary, verification, pilot/main imports, warmup, online updates, diagnostic exact-gradient probes, data regeneration, full replays, launchers and frozen-analysis startup. Main children used 259.792s aggregate over 88.019s elapsed. Included main warmup cost was 60.663s and replay 81.671s. Dependency setup, source/report preparation and transfers are excluded. Maximum concurrency was three one-thread CPU workers. Pinned Python 3.12.14, PyTorch 2.14.1+cpu and NumPy 2.3.5 are recorded in RUNTIME.json.

## Results

Primary CE averages novel-context checkpoints544–768 and changed-rule checkpoints1056–1280, every 32 steps:16 checkpoints, nats per binary target. The reported effect averages each seed's paired relative reduction. Old-window accuracy averages stable facts over the same checkpoints; final-old is terminal stable accuracy. Stable facts exclude the deliberately changed relations. Negative accuracy deltas indicate additional loss versus the control.

| Seed | U1-only CE | Frozen CE | CE reduction | Old-window delta | Final-old delta |
|---|---:|---:|---:|---:|---:|
| 1301841 | 0.826738 | 1.045939 | 20.96% | -7.09pp | +0.45pp |
| 1301842 | 0.908854 | 1.055380 | 13.88% | -6.28pp | +0.28pp |
| 1301843 | 0.808212 | 0.987175 | 18.13% | -6.02pp | -0.06pp |

Mean primary reduction **17.6566%**, sample SD **3.5604pp**. All seeds improve and the mean exceeds 5%, so the effect gate passes. The old-window guard fails all three seeds; the final-old guard passes all three. **The joint primary gate fails.** Final-old deltas average+0.2232pp, which does not erase the intervening losses.

All registered secondary contrasts, seed order1301841/1301842/1301843:

| Contrast | Mean CE reduction | Sample SD | Old-window deltas by seed | Final-old deltas by seed |
|---|---:|---:|---|---|
| U0-only vs frozen | +9.84% | 0.615pp | -2.89 / -3.12 / -1.30pp | -1.84 / -0.61 / -0.39pp |
| Both-U vs frozen | +28.68% | 1.017pp | -11.09 / -9.50 / -8.74pp | -1.28 / +0.73 / +0.39pp |
| Both-U vs U1-only | +13.29% | 3.727pp | -4.00 / -3.22 / -2.72pp | -1.73 / +0.45 / +0.45pp |
| Both-U vs U0-only | +20.90% | 0.588pp | -8.21 / -6.38 / -7.44pp | +0.56 / +1.34 / +0.78pp |

U0-only reductions are10.3154/10.0658/9.1478%, mean 9.8430%, sample SD 0.6148pp. Its old-window deltas are-2.8878/-3.1215/-1.3009pp; final-old deltas-1.8415/-0.6138/-0.3906pp. The first two old-window losses exceed2pp. This is a closer tradeoff, not a passed result or a replacement primary.

Both-U reductions are29.4896/29.0148/27.5390%, mean 28.6811%, sample SD 1.0172pp. It fails all old-window guards but passes every final-old guard. Adding U0 learning to U1-only lowers primary CE by 13.2863% on average while worsening old-window accuracy another2.72–4.00pp. Neither single readout captures all of both-U's adaptation benefit.

All-arm primary CE:

| Arm | CE 1301841 | CE 1301842 | CE 1301843 |
|---|---:|---:|---:|
| Backprop | 0.780341 | 0.753420 | 0.751759 |
| Frozen | 1.045939 | 1.055380 | 0.987175 |
| U0-only | 0.938047 | 0.949148 | 0.896870 |
| U1-only | 0.826738 | 0.908854 | 0.808212 |
| Both-U | 0.737496 | 0.749164 | 0.715317 |

Final heldout means:

| Arm | Final old | Final novel | Final changed | Final balanced |
|---|---:|---:|---:|---:|
| Backprop | 96.95% | 94.86% | 84.51% | 92.10% |
| Frozen | 94.68% | 82.58% | 37.50% | 71.59% |
| U0-only | 93.73% | 90.66% | 54.56% | 79.65% |
| U1-only | 94.90% | 82.39% | 44.66% | 73.98% |
| Both-U | 94.62% | 91.15% | 57.29% | 81.02% |

U1-only's stronger early CE benefit does not translate into the best terminal local-policy performance. Its final balanced accuracy is73.98%, compared with 79.65% for U0-only and 81.02% for both-U. Frozen reaches71.59%; backprop remains strongest at92.10%. U1-only's final novel accuracy82.39% is slightly below frozen82.58%, while final changed accuracy rises from 37.50% to 44.66%. These final endpoints do not replace the registered early-window metric.

Registered window components and mean L2 readout displacement from warmup:

| Arm | Novel-window CE | Changed-window CE | U0 displacement | U1 displacement |
|---|---:|---:|---:|---:|
| Backprop | 0.622957 | 0.900723 | 0.0000 | 0.0000 |
| Frozen | 0.845206 | 1.213790 | 0.0000 | 0.0000 |
| U0-only | 0.813030 | 1.043014 | 2.8688 | 0.0000 |
| U1-only | 0.852378 | 0.843492 | 0.0000 | 3.1987 |
| Both-U | 0.832058 | 0.635927 | 2.2837 | 3.1840 |

B and V displacement is exactly zero in every arm. U1-only's improvement comes from lower changed-window CE; its mean novel-window CE is slightly worse than frozen. U0-only improves both component means. Both-U's particularly strong changed-window improvement comes with the largest old-window accuracy loss. Same-state direction probes and complete prequential/heldout traces remain in RESULT.json; no improved BP alignment is inferred.

The CE interaction `(both-U - U1-only) - (U0-only - frozen)` is+0.018650/-0.053458/-0.002590 nats, mean **-0.012466**. Its sign varies by seed; it does not support a uniform synergy claim. The mean old-window accuracy interaction is-0.8777pp. Per-seed U1 fractions of both-U CE gain are71.07/47.85/65.83% (mean 61.58%); U0 fractions34.98/34.69/33.22% (mean 34.30%). These ratios are descriptive and need not add to 100%.

## Interpretation and limits

The proposed U1-only protective-retention hypothesis fails under the unchanged joint objective. Fixing U0 reduces old-window loss relative to both-U, but does not reduce it enough. U0-only is less damaging during the measured windows and still clears the useful-effect threshold in this explanatory comparison.

Old-fact accuracy recovers by the final mixed-exposure phase, when old data returns. That is consistent with relearning or recovery; the experiment does not show that old knowledge stayed accessible during adaptation. The large window failures remain substantively important even with favorable terminal accuracy. They also show why a terminal-only guard would give a different answer.

All local policies store the same model and receive the same data/exposure, but update different parameter counts. Same model/exposure is not FLOP matching or equality of trainable capacity. Three fresh paired synthetic worlds do not establish population significance, language transfer, novelty, scaling or deployment efficiency. Earlier V/U-only results used different worlds; this stage's28.68% both-U mean is not evidence of improvement over that earlier22.37% mean. The dedicated numerical equivalence check, rather than cross-stage aggregate comparison, grounds the mechanism claim.

## Deployment and preservation

Stored forward parameters:1504,6016 FP32 bytes. Feedback:12544,50176 FP32 bytes. B/V fixed parameters:8448; U0 and U1 each 2048. Online updated counts are1504 for frozen/BP,3552 for each single-U arm and 5600 for both-U. BP only uses forward weights; shared experiment bookkeeping retains a feedback bank.

FIFO payload is30720 FP32 bytes plus 4096 int64 ID bytes, excluding containers/RNG. Precomputed streams and queries are benchmark bookkeeping. Main peak RSS is **385,676–388,832 KiB per worker**, including framework, streams and diagnostics. Aggregate simultaneous peak, workspace/allocator breakdown, FLOPs, throughput, quantization and deployment latency are unmeasured; no prefill/decode benchmark applies.

Four individually compressed NPZ checkpoints total **469,853 bytes**. Each holds 28 arrays: common warmup, all final forward weights and only the U readouts learned by that arm. Complete final state is warmup overlaid with the arm. All four remote SHA256 readbacks and overlay-schema checks pass. All21 frozen source/protocol/runtime files retain their hashes. [Artifacts](../results/feedback-readout-20261018-v1/ARTIFACT_MANIFEST.json) · [Audit](../results/feedback-readout-20261018-v1/EVIDENCE_AUDIT.json) · [Terminal](../results/feedback-readout-20261018-v1/TERMINAL.json).

All workers exited. After preservation,965,051,069 bytes of temporary runtime and redundant readback copies were removed. Completed progress snapshots that reappeared were compared exactly with retained RESULT.json before removal; no partial or failed evidence was discarded. Original archives and older results remain intact. Earlier bounded-feedback missing-source/checkpoint evidence and the original feedback-decay-v1 cap failure remain explicitly caveated. This stage does not recover historical missing bytes.

## Decision and next proposed experiment

**No promotion.** The primary passes adaptation but fails retention during adaptation in every seed. Final recovery is encouraging for reversibility, yet it does not satisfy the declared objective.

The next bounded hypothesis follows the closest observed tradeoff: **half-rate U0-only** may reduce its modest excess old-window loss while retaining at least 5% CE improvement. This is a prospective hypothesis based on the explanatory arm, not a change to the completed primary. A key disconfirmation is losing the useful adaptation benefit when the U0 drive is halved; linear scaling of either gain or forgetting is not assumed.

Proposed only: test one fixed half-rate U0-only setting (beta=0.5 on the U0 local learning drive only), with B1/B2/V1/V2/U1 fixed at common warmup and online feedback decay off. Four arms: BP, frozen feedback, original full-rate U0-only, half-rate U0-only. Sole primary half-rate U0-only versus frozen, unchanged >=5% mean paired CE reduction/no negative seed and <=2pp per-seed old-window/final-old loss. Full-rate U0-only is the positive adaptation control. Keep W optimizer, common4096 warmup, FIFO 512/16, strict cap 2, data/exposure and endpoints. Three fresh paired main seeds plus separate mechanical pilot. Verify beta 1 matches preserved U0-only and beta 0 matches frozen bitwise, independent dense half-rate steps, inactive matrices, strict cap and all full replays. No added rate grid or outcome-driven tuning. Expected200-350 aggregate CPU seconds inside 900, max3 one-thread workers; four compact checkpoint files. This follow-on is not implemented or running; await continuation.

The single new factor halves U0's learning drive; it does not halve W updates, add online decay, change replay or select a new endpoint. Full-rate U0-only reproduces the current explanatory condition on fresh worlds. No further rate is tried if half-rate fails. This differs from the earlier0.1 whole-feedback/decay study, and does not inherit a passed claim from it. Await the user's continuation; no follow-on or language/cloud expansion is launched.

## Cleanup reconciliation

After main-evidence publication, a partial temporary runtime was present again despite the earlier recorded deletion. No matching experiment workers were running. The additional 51,513,836 bytes were removed and all 21 frozen source files still matched. The cause is undetermined; no experiment was rerun. [Cleanup receipt](../results/feedback-readout-20261018-v1/POST_PUBLICATION_CLEANUP.json).
