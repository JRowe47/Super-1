# Experiment report: feedback-decay-20261018-v2

**Completed; no promotion.** Learned feedback changes improve post-change cross-entropy beyond passive decay, but the predeclared retention guard fails in all three main seeds. Normal adaptive feedback reduces CE by **27.03% versus decay-only**, with old-window losses of **6.46–9.25 percentage points** and final-old losses of **2.62–3.57pp**. Both exceed the allowed 2pp in every seed. Decay-only worsens CE versus frozen; learning-only retains the adaptation benefit and still forgets.

The user authorized the fixed-FIFO learning/decay factorial. Exact v2 source/protocol was published and read back before execution at `801d1e3147c47bd5b6004f1d343ea9a2c9185d6b`; protocol SHA256 `9631d01c94420a3b22cc4e721897cdebf80270dd58e44d25eaa7faa0ac47fa44`. [Protocol](../results/feedback-decay-20261018-v2/PROTOCOL.json) · [Registration](../results/feedback-decay-20261018-v2/REGISTRATION.json) · [Summary](../results/feedback-decay-20261018-v2/SUMMARY.json).

## Question and fixed design

Does data-dependent feedback learning add useful adaptation beyond passive feedback weight decay while limiting extra old-fact loss to 2pp?

Five arms share the same model, initial state, 4096-step common warmup, forward optimizer, data/query stream, FIFO storage of 512 examples, 16 replay plus 16 current examples per update, and training exposure. Backprop is the exact-gradient control. The four local arms cross feedback learning on/off with feedback decay on/off:

| Arm | Local feedback learning | Feedback decay |
|---|---|---|
| Frozen | Off | Off |
| Decay-only | Off | On |
| Learning-only | On | Off |
| Normal adaptive | On | On |

Forward step size is 0.05 with weight decay 0.001. Feedback uses the original 0.05 step and local-rule factor 0.8; when enabled, feedback decay is 0.001 for B and 0.01 for V/U. A common spectral cap of 2 follows enabled feedback updates. Common warmup uses BP forward learning plus full local feedback learning/decay. No phase, query outcome, source ID or online tuning signal controls these switches. [Rule mapping](../results/feedback-decay-20261018-v2/RULE_MAPPING.md).

Primary: normal adaptive versus decay-only. The preregistered effect gate requires mean paired CE reduction at least 5%, with no negative seed; old-window and final-old accuracy may each lose at most 2pp in every seed. Secondary factorial contrasts and the original normal-versus-frozen comparison remain visible. A narrower mechanism result cannot override the original retention objective.

## Verification failure and numerical repair

The original v1 source was frozen at `8d5a000f7572962bc4eb29e072bbe71b1bb5f348`. Verification seed 15817 in FP64 exposed decoded B1 spectral norm **2.0000000000000004**, violating the strict cap by 4.44e-16 after the inherited four-epsilon projection. v1 stopped **before any pilot or main training**. Its source, failed launcher receipt and read-only diagnosis remain intact. Its VERIFICATION.json contains only inherited successful checks; the overall status is FAILED in VERIFY_RECEIPT.json. [Failure report](feedback-decay-20261018-v1.md).

Separately frozen v2 retains the original projection, reads back the decoded norm, and corrects only an actual overshoot using a 32-epsilon inward margin, with at most three corrections and a hard failure for any remaining violation. The shared operator covers all updated-feedback arms and common warmup. **Cap 2, FP32 tolerance 2e-6, FP64 tolerance 2e-12, success thresholds and unseen training seeds are unchanged.** No outcome-driven tuning occurred.

All original and new checks pass. Three seeds × two dtypes × bounded/active-projection states × four feedback switch combinations are compared with independent dense equations. Maximum component errors are 1.81794e-6 (FP32) and 4.88498e-15 (FP64), within original tolerances. Frozen/full endpoints and 12-step trajectories match their common reference policies bitwise; forward updates at a common state match; decay feedback is independent of changed inputs/labels and matches its analytic 12-step trajectory. Local training paths avoid autograd, outputs are detached, input state is unchanged, and independent FIFO references pass.

The additional projection regression covers 288 matrix outputs: 285 already-compliant outputs remain bitwise unchanged; three overshoots receive corrections. Maximum coefficient differences are 1.84774e-6 in FP32 and 3.77476e-15 in FP64. The known failing case is exercised, not removed. [Verification](../results/feedback-decay-20261018-v2/VERIFICATION.json).

## Execution and cost

Fresh paired main seeds **1101821, 1101822, 1101823**; separate mechanical pilot **100917**. Main runs use 4096 common warmup steps and 2048 online updates per arm across four 512-step phases. Pilot uses the same warmup plus 256 updates per arm. All 15 main and five pilot arms completed. v2 had no execution exception, source change, retry or seed exclusion.

The unchanged synthetic threshold-world has 11 inputs and four binary targets. Novel contexts precede a later rule change. Heldout queries use 128 examples per context for main and 32 for pilot. Warmup exposes 1,048,576 binary targets per seed; online main exposure is 262,144 per arm. Replay retains historically observed labels. Obsolete replay-target counts are 40/44/36 by main seed. No official test or external dataset was accessed.

All **20 complete online trajectories** replayed from serialized warmup with regenerated/hash-verified data. Four phase hashes, final complete-state hashes and stored final tensors match bitwise. Exact source and per-seed data/state hashes remain in RESULT.json. Pilot/verification evidence was preserved during unchanged main execution at `cd316e68f256a4dd4236ed3e47bb09761db217c6`.

Charged experiment CPU is **250.633s of 900s**, including the failed v1 verification/diagnosis (3.378945s), the import canary once, v2 verification, imports, warmup, all arms, probes, full replays, launchers and analysis startup. Main child CPU is 216.438s aggregate over 73.040s elapsed; warmup is 53.329s and replay 65.937s, both included in the total. Dependency setup, source/report preparation and file transfers are excluded. At most three one-thread CPU workers ran; no GPU, cloud or paid resource was created. Python 3.12.14, PyTorch 2.14.1+cpu and NumPy 2.3.5 are pinned in RUNTIME.json.

## Results

Primary CE averages novel-context checkpoints at steps 544–768 and changed-rule checkpoints at 1056–1280, every 32 steps: 16 checkpoints, nats per binary target. Paired relative reductions are averaged across seeds. Old-window accuracy uses the same checkpoints; final-old uses the terminal state. Stable facts exclude the two deliberately changed relations. Positive CE reductions favor the candidate; negative accuracy deltas indicate loss.

| Seed | Normal CE | Decay-only CE | CE reduction | Old-window delta | Final-old delta |
|---|---:|---:|---:|---:|---:|
| 1101821 | 0.760853 | 1.072949 | 29.09% | -9.25pp | -3.01pp |
| 1101822 | 0.764590 | 0.993458 | 23.04% | -6.46pp | -3.57pp |
| 1101823 | 0.719046 | 1.012333 | 28.97% | -6.49pp | -2.62pp |

Mean CE reduction **27.0322%**, sample SD **3.4600pp**: effect passes. Both retention measures fail in all three seeds; mean old-window delta is -7.3998pp and mean final-old delta -3.0692pp. **The joint primary gate fails.**

Preregistered secondary contrasts, with per-seed deltas ordered 1101821/1101822/1101823:

| Contrast (candidate vs control) | Mean CE reduction | Sample SD | Old-window deltas by seed | Final-old deltas by seed |
|---|---:|---:|---|---|
| Learning-only vs frozen | +26.88% | 4.58pp | -11.89 / -9.43 / -8.06pp | -4.58 / -4.52 / -4.69pp |
| Decay-only vs frozen | -3.57% | 4.11pp | -1.90 / -2.20 / -0.84pp | -0.61 / -1.00 / -1.40pp |
| Normal vs learning-only | -3.38% | 1.79pp | +0.74 / +0.77 / +0.73pp | +0.95 / -0.06 / +0.67pp |
| Normal vs frozen | +24.47% | 3.51pp | -11.15 / -8.66 / -7.33pp | -3.63 / -4.58 / -4.02pp |

Decay-only worsens CE versus frozen in every seed (mean **3.57% worse**); it does not explain the observed adaptive gain in these worlds. Learning-only improves CE by **26.88% versus frozen**, but loses 8.06–11.89pp old-window and 4.52–4.69pp final-old accuracy. Normal adaptive improves **24.47% versus frozen** and likewise fails retention in every seed. No learned-feedback variant passes the original joint objective.

Adding decay to learned feedback worsens the combined early-window CE by 3.38% on average, while improving mean old-window accuracy by 0.75pp and final-old by 0.52pp; one final-old seed is slightly worse. Thus decay's role depends on the learning path and endpoint. It is not an adequate retention remedy here.

All-arm primary CE:

| Arm | CE 1101821 | CE 1101822 | CE 1101823 |
|---|---:|---:|---:|
| Backprop | 0.726915 | 0.747643 | 0.696218 |
| Frozen | 0.990875 | 0.977090 | 1.004714 |
| Decay-only | 1.072949 | 0.993458 | 1.012333 |
| Learning-only | 0.744717 | 0.745733 | 0.681964 |
| Normal adaptive | 0.760853 | 0.764590 | 0.719046 |

Final mean heldout accuracy:

| Arm | Old accuracy | Novel accuracy | Changed accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| Backprop | 96.54% | 95.64% | 89.58% | 93.92% |
| Frozen | 93.25% | 87.70% | 48.31% | 76.42% |
| Decay-only | 92.24% | 82.00% | 35.94% | 70.06% |
| Learning-only | 88.65% | 85.94% | 73.57% | 82.72% |
| Normal adaptive | 89.17% | 86.43% | 73.96% | 83.19% |

Backprop has the strongest final balanced accuracy (93.92%). Normal adaptive reaches 73.96% changed-rule accuracy versus 35.94% decay-only and 48.31% frozen, but final old accuracy is 89.17% versus 92.24% and 93.25%. An early-window CE advantage does not establish overall superiority over backprop.

Preregistered endpoint decomposition and mean L2 feedback displacement from warmup:

| Arm | Novel-window CE | Changed-window CE | Feedback displacement |
|---|---:|---:|---:|
| Backprop | 0.554589 | 0.892596 | 0.0000 |
| Frozen | 0.758083 | 1.223703 | 0.0000 |
| Decay-only | 0.829776 | 1.222717 | 2.0839 |
| Learning-only | 0.801340 | 0.646936 | 6.7999 |
| Normal adaptive | 0.871535 | 0.624791 | 6.7776 |

Learning-only and normal adaptive both gain mainly in the changed-rule window; their novel-window CE is worse than frozen. Normal adaptive's changed-window CE is slightly better than learning-only, despite its worse combined primary CE. These are fixed registered components, not replacement endpoints.

The CE interaction `(normal - decay-only) - (learning-only - frozen)` is -0.065937, +0.002489, +0.029463 nats by seed, mean **-0.011329**; its sign varies. The corresponding old-window accuracy interaction averages +2.3914pp. The descriptive ratio of decay-only CE gain to normal CE gain is negative in every seed (-35.68%, -7.70%, -2.67%; mean -15.35%). These intervention contrasts are not additive causal shares. Full interactions, direction probes and prequential traces remain in the compact results; no improved alignment with BP is inferred.

Three paired synthetic worlds support this bounded mechanism conclusion. They do not establish population significance, language-model transfer, novelty, scalability or deployment efficiency.

## Preservation and deployment measurements

All 18 registered v2 files and all 16 failed-v1 files retain their frozen hashes. Four compressed NPZ checkpoints total **921,878 bytes**; every remote readback SHA256 matches. Each stores common warmup W/F and final W for every arm; decay/learning/normal arms additionally store final F, while BP/frozen refer to warmup F. Data regenerate from registered seeds. [Artifact manifest](../results/feedback-decay-20261018-v2/ARTIFACT_MANIFEST.json) · [Audit](../results/feedback-decay-20261018-v2/EVIDENCE_AUDIT.json) · [Terminal](../results/feedback-decay-20261018-v2/TERMINAL.json).

Forward parameters: 1504 (6016 FP32 bytes); feedback: 12544 (50176 bytes). Replay holds 512 examples, 30720 FP32 payload bytes plus 4096 int64 provenance bytes, excluding container/RNG overhead. Active BP requires only forward weights; benchmark bookkeeping retains shared feedback. Full precomputed streams/queries are benchmark storage, not deployed replay memory.

Main peak RSS is 379,172–379,364 KiB per worker, including framework, full streams and diagnostics. Aggregate simultaneous peak, allocator/workspace breakdown, FLOPs, quantization, throughput and deployment latency are unmeasured. Prefill/decode does not apply to this feedforward task.

All workers exited. After exact remote preservation, 912,225,523 bytes of reproducible runtime files and redundant readback copies were removed. A completed pilot progress snapshot was compared exactly with RESULT.json before removal. Completed artifacts, original archives and all failure evidence remain intact. The historical bounded-feedback missing-source/checkpoint gap remains explicit; this run does not reconstruct those lost bytes.

## Decision and next proposed stage

**No promotion.** Data-dependent feedback updates add adaptation beyond decay, but the joint retention objective fails. Removing decay also fails to resolve forgetting. This weakens the passive-decay explanation at the tested rates and worlds; it does not prove that every decay schedule or feedback mechanism behaves the same way.

Next, localize learning to feedback matrix families. Freeze V1/V2/U0/U1 at their warmup values while allowing B1/B2 to learn; compare with frozen, V/U-only, full learning-only and BP. Disable online feedback decay in all local arms so an inactive family is literally unchanged. The hypothesis is that adaptive error transport B can preserve useful adaptation while fixed encoder/readout V/U reduces forgetting. B changes still propagate through the whole feedback path; this is a family ablation, not an isolation of every hidden-layer causal effect.

Proposed primary: **B-only versus frozen**, restoring the original useful-adaptation-plus-retention question, with unchanged 5%/no-negative-seed and per-seed 2pp guards. Full learning-only and V/U-only remain explanatory controls; do not select a primary after outcomes. Same FIFO, full feedback step, common warmup, spectral cap, exposure and endpoints; three fresh paired seeds and separate pilot, exact source-before-execution, full replays and compact checkpoints. No rate sweep. Expected roughly 250–450 aggregate CPU seconds inside the standing 900s envelope, maximum three one-thread workers. This follow-on is **proposed only, not implemented or running**. Language/1B training and broader resources remain paused.
