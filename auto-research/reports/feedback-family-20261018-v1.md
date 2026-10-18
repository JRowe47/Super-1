# Experiment report: feedback-family-20261018-v1

**Completed; no promotion.** Learning only the B feedback matrices does not solve retention. Its early post-change CE improves **6.96% versus frozen feedback**, but final old-fact accuracy falls **10.04–14.73 percentage points** in every seed, far beyond the 2pp guard. Old-window retention also fails in every seed. The proposed protective role of freezing V/U is rejected for this configuration.

The explanatory **V/U-only** arm is more promising: **22.37%** CE improvement, close to full learning's **21.87%**, and much better final retention than B-only. However, V/U-only loses **5.12–7.58pp** old-window accuracy in all three seeds and **2.62pp** final-old accuracy in one seed. Its near-zero mean final-old loss does not satisfy the per-seed guards. No learned-feedback arm passes the original joint objective.

Approval: user “Ok, go ahead with that”, accepting the B versus V/U family-learning proposal. Exact source/protocol commit: `553fb843756fa98ca534196f84902bf2d78aa863`; protocol SHA256 `b6d41588d346510748df8507d1448dc22a44183c192313b1e3576ebf4360f673`. All 23 publication files were read back byte-equal and the branch verified before numerical checks or training. [Protocol](../results/feedback-family-20261018-v1/PROTOCOL.json) · [Registration](../results/feedback-family-20261018-v1/REGISTRATION.json) · [Summary](../results/feedback-family-20261018-v1/SUMMARY.json).

## Question and controls

Does allowing B to learn while keeping V/U fixed preserve useful post-change adaptation and old knowledge?

Five arms share the same model, initial state, 4096-step common warmup, forward optimizer, data/query stream, FIFO storage of 512 examples, 16 replay plus 16 current examples per update, and exposure. Exact backprop is the strong control. Four local arms cross B learning on/off with V/U learning on/off:

| Arm | B1/B2 learning | V1/V2/U0/U1 learning | Online feedback decay |
|---|---|---|---|
| Frozen | Off | Off | Off |
| B-only | On | Off | Off |
| V/U-only | Off | On | Off |
| Full learning-only | On | On | Off |

Forward step size is 0.05 with decay 0.001. Active feedback uses the original 0.05 step and local-rule factor 0.8. Inactive matrices remain bitwise equal to common warmup. Common warmup retains original BP forward plus full local feedback learning/decay; online feedback decay is then off. Strict cap 2 uses the byte-preserved shared numerical repair from feedback-decay-v2. Partial-family updates restore inactive coefficients after the common cap, so redundant numerical projection cannot move them. All computation, including redundant SVDs, is charged.

The B family has 4352 parameters; V/U has 8192. Every local policy stores the same 1504 forward and 12544 feedback parameters, while the intended ablation changes which feedback parameters update. Online updated counts are 5856 for B-only, 9696 for V/U-only and 14048 for full learning, versus 1504 for frozen/BP. This is matched model/data/exposure, not matched FLOPs or trainable parameter count.

Sole primary: B-only versus frozen. Effect requires mean paired relative CE reduction >=5% and no negative seed. Both old-window and final-old accuracy may lose at most 2pp in every seed. Secondary outcomes cannot replace the primary. The strongest alternative is that adaptation needs U readout learning and that B changes with fixed readouts disrupt the mapping; the V/U-only and full arms test that alternative. [Rule mapping](../results/feedback-family-20261018-v1/RULE_MAPPING.md).

## What ran and verification

Main seeds **1201831, 1201832, 1201833**; independent mechanical pilot **101927**. Main: 4096 shared warmup steps plus 2048 online updates per arm in four 512-step phases. Pilot: same warmup plus 256 updates per arm. All 15 main and five pilot arms completed, without experiment exceptions, source changes, seed exclusions, retries or a rate sweep.

The unchanged synthetic threshold-world has 11 inputs and four binary labels. Novel contexts precede a rule change. Main query sets use 128 examples per context, pilot 32. Warmup exposes 1,048,576 binary targets per seed; main online exposure is 262,144 per arm. FIFO preserves observed historical labels, including briefly obsolete targets; obsolete replay-target counts are 28/36/36 by main seed. No external corpus or official test was accessed.

All prior transform, local-equation, exact-gradient, component-switch, spectral-cap and independent FIFO checks remain and pass. The known prior cap failure remains covered by the preserved regression; no tolerance or cap changed in this stage. New family verification covers three fresh verification seeds, two dtypes, bounded/active-projection states and all four masks against independent dense per-example equations. Maximum family-step errors are **3.57628e-7 FP32** and **6.66134e-16 FP64**, below the original 2e-6 and 2e-12 limits. Maximum decoded capped norm is exactly 2.0.

Same-state forward updates are bitwise identical; inactive matrices remain exact over 12-step trajectories; frozen/full endpoints and trajectories match the preserved reference bitwise. Active learning is nonzero in the checked cases, inputs remain unchanged, outputs detach, and local paths avoid autograd. No new result is attributed merely to nonzero gradients. [Verification](../results/feedback-family-20261018-v1/VERIFICATION.json).

Every one of **20 complete online trajectories** replayed from serialized common warmup with independently regenerated/hash-verified data. All four phase hashes, final full-state hashes and stored tensors match bitwise. Every inactive feedback matrix also matches warmup exactly at the end. The replay is a reproducibility check, not another independent seed. Verification/pilot evidence was preserved during unchanged main execution at `036f4e6bf5485d8f5916231f7926e2b1610d6212`.

Charged experiment CPU: **273.358s / 900s**, including runtime import canary, verification, all child imports/warmup/online work, exact-gradient probes, data regeneration, full replays, launchers and frozen analysis startup. Main children used 241.148s aggregate over 81.113s elapsed; included main warmup was 62.305s and full replay 68.344s. Dependency setup, source/report preparation and transfers are excluded. At most three one-thread CPU workers ran. Python 3.12.14, PyTorch 2.14.1+cpu and NumPy 2.3.5 are pinned in RUNTIME.json.

## Results

Primary CE averages novel-context checkpoints at steps 544–768 and changed-rule checkpoints at 1056–1280, every 32 steps: 16 checkpoints, nats per binary target. Each seed's paired relative reduction is averaged across seeds. Old-window accuracy averages stable facts at those same checkpoints; final-old is the terminal score. Stable facts exclude the two intentionally changed relations. Positive CE reduction favors the candidate; negative accuracy deltas indicate loss.

| Seed | B-only CE | Frozen CE | CE reduction | Old-window delta | Final-old delta |
|---|---:|---:|---:|---:|---:|
| 1201831 | 0.866618 | 0.961703 | 9.89% | -8.83pp | -12.22pp |
| 1201832 | 0.995693 | 1.027830 | 3.13% | -4.86pp | -10.04pp |
| 1201833 | 0.936074 | 1.016068 | 7.87% | -6.37pp | -14.73pp |

Mean reduction **6.9622%**, sample SD **3.4710pp**. The mean exceeds 5% and all three seeds are positive, so the effect gate passes; a seed below 5% is allowed by the declared rule. Old-window losses **4.86–8.83pp** and final-old losses **10.04–14.73pp** fail both guards in every seed. Mean old-window delta is -6.6848pp; final-old is -12.3326pp. **Joint primary gate fails.**

All preregistered secondary contrasts; per-seed deltas ordered 1201831/1201832/1201833:

| Contrast (candidate vs control) | Mean CE reduction | Sample SD | Old-window deltas by seed | Final-old deltas by seed |
|---|---:|---:|---|---|
| V/U-only vs frozen | +22.37% | 0.0377pp | -6.94 / -5.12 / -7.58pp | -0.28 / +3.01 / -2.62pp |
| Full learning-only vs frozen | +21.87% | 0.9832pp | -11.01 / -10.37 / -8.83pp | -1.12 / -2.96 / -8.76pp |
| Full vs B-only | +15.97% | 2.1947pp | -2.18 / -5.51 / -2.47pp | +11.10 / +7.09 / +5.97pp |
| Full vs V/U-only | -0.65% | 1.2834pp | -4.07 / -5.25 / -1.25pp | -0.84 / -5.97 / -6.14pp |

V/U-only's CE reductions are 22.3314/22.3935/22.3994% (mean **22.3748%**, sample SD **0.0377pp**). Its final-old deltas are -0.2790/+3.0134/-2.6228pp: mean **+0.0372pp**, but one seed still exceeds the allowed loss. All three old-window deltas fail, averaging -6.5476pp. Neither the favorable mean final retention nor the narrow CE spread establishes a passed retention criterion or population robustness.

Full learning-only improves **21.8719%** CE versus frozen, but loses 8.83–11.01pp old-window accuracy; two final-old seeds also exceed 2pp. Relative to V/U-only, full learning has 0.65% higher primary CE on average, loses **3.5261pp old-window** and **4.3155pp final-old** accuracy. CE is mixed across seeds; the accuracy deltas favor V/U-only in all three. Adding B learning is not supported as beneficial in that comparison.

All-arm primary CE:

| Arm | CE 1201831 | CE 1201832 | CE 1201833 |
|---|---:|---:|---:|
| Backprop | 0.705604 | 0.710131 | 0.819155 |
| Frozen | 0.961703 | 1.027830 | 1.016068 |
| B-only | 0.866618 | 0.995693 | 0.936074 |
| V/U-only | 0.746941 | 0.797663 | 0.788474 |
| Full learning-only | 0.747047 | 0.814613 | 0.786934 |

Final mean heldout accuracy:

| Arm | Old accuracy | Novel accuracy | Changed accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| Backprop | 96.56% | 96.03% | 85.55% | 92.71% |
| Frozen | 92.06% | 78.81% | 41.54% | 70.80% |
| B-only | 79.72% | 77.51% | 40.76% | 66.00% |
| V/U-only | 92.09% | 83.89% | 71.88% | 82.62% |
| Full learning-only | 87.78% | 87.24% | 61.33% | 78.78% |

B-only's final changed-rule accuracy (40.76%) is slightly below frozen (41.54%), despite its earlier CE improvement; final balanced accuracy is lower (66.00% versus 70.80%). Its primary benefit is not sustained superiority at the final endpoint. V/U-only reaches 71.88% changed accuracy and 82.62% balanced, versus 61.33% and 78.78% for full learning. Backprop remains strongest at 92.71% final balanced accuracy.

Registered endpoint components and mean feedback L2 displacement from warmup:

| Arm | Novel-window CE | Changed-window CE | B displacement | V/U displacement |
|---|---:|---:|---:|---:|
| Backprop | 0.566880 | 0.923047 | 0.0000 | 0.0000 |
| Frozen | 0.841192 | 1.162542 | 0.0000 | 0.0000 |
| B-only | 0.850883 | 1.014707 | 3.4731 | 0.0000 |
| V/U-only | 0.794768 | 0.760618 | 0.0000 | 5.3257 |
| Full learning-only | 0.854138 | 0.711591 | 4.0843 | 5.1897 |

V/U-only improves both component-window means versus frozen; B-only and full learning have slightly worse novel-window CE and lower changed-window CE. Full learning has lower changed-window CE than V/U-only but worse novel-window CE. These are fixed registered components, not replacement primary metrics.

The family CE interaction `(full - B-only) - (V/U-only - frozen)` is **+0.095190 / +0.049087 / +0.078454 nats**, mean **+0.074244**: combined learning delivers less CE reduction than adding the separate observed reductions would suggest. Mean old-window interaction is +3.1587pp and final-old interaction +8.0171pp; interactions depend on the endpoint and are not additive causal shares.

Per-seed fractions of full CE gain are 44.30/15.07/34.91% for B-only, mean **31.43%**; V/U-only fractions are 100.05/107.95/99.33%, mean **102.44%**. Fractions are descriptive intervention ratios, not attribution percentages that must sum to 100. All direction probes and prequential traces remain in RESULT.json; no improved BP alignment is inferred.

## Mechanistic interpretation

The B-only protective-retention hypothesis fails at these settings. Fixed readout matrices do not make changing error transport safe; final retention is worse than with full learning. V/U-only shows that substantial adaptation does not require changing B in this benchmark, while the old-window failures still prevent promotion.

There is a useful structural implication of the preserved equations: **when B is fixed, changing V cannot affect W or U updates**. V enters the B-learning drive; the W and U drives depend on activations, B-transmitted errors and U, not V. With B updates disabled and matrixwise cap operations, differences confined to V remain confined there. Thus the non-V trajectory of V/U-only is attributable to U learning under these equations. This is an inference from the explicit dependency structure, not an additional U-only experiment conducted here. Updating V still consumed compute and its state was retained.

Three fresh paired synthetic worlds provide bounded mechanism evidence, not population significance, language transfer, novelty, scalability or deployment efficiency. No rate, comparator or endpoint was chosen after seeing outcomes for the completed experiment.

## Deployment and preservation

Stored forward parameters: 1504 (6016 FP32 bytes). Local-policy feedback parameters: 12544 (50176 bytes). Every replay memory contains 512 examples, 30720 FP32 payload bytes and 4096 int64 provenance bytes, excluding containers/RNG. BP's active inference/training model uses only forward weights; shared experiment bookkeeping retains feedback. Full precomputed streams and queries are benchmark storage, not deployed replay memory.

Main peak RSS is 386,168–389,024 KiB **per worker**, including framework, streams and diagnostics. Aggregate simultaneous peak, allocator/workspace breakdown, FLOPs, quantization, throughput and deployment latency are unmeasured. No prefill/decode benchmark applies to this feedforward synthetic classifier.

Four individual compressed NPZ files total **729,783 bytes**. Each contains 36 arrays: common warmup, all final forward weights, and only each arm's actively learned feedback matrices. Complete final state is warmup overlaid with that arm. Every remote SHA256 readback and compact overlay schema check passes. All 19 frozen source/protocol/runtime files retain their hashes. [Artifact manifest](../results/feedback-family-20261018-v1/ARTIFACT_MANIFEST.json) · [Audit](../results/feedback-family-20261018-v1/EVIDENCE_AUDIT.json) · [Terminal](../results/feedback-family-20261018-v1/TERMINAL.json).

All workers exited. After preservation, 1,066,500,797 bytes of reproducible temporary runtime and redundant readback copies were removed. A completed pilot progress snapshot was checked exactly against RESULT.json and removed. No failed/partial evidence was discarded, no archive was duplicated, and prior completed artifacts remain intact. The older bounded-feedback missing-source/checkpoint gap and feedback-decay-v1 numerical failure remain explicitly preserved; this stage does not reconstruct lost historical bytes.

## Decision and next proposed experiment

**No promotion.** B-only passes useful-effect but fails retention decisively. V/U-only is the better explanatory direction, yet it also fails the predeclared joint objective. The result does not justify choosing its favorable terminal mean as a substitute for the old-window guard.

Next isolate **U0 versus U1 learning**, with B1/B2/V1/V2 fixed and online feedback decay off. U0 decodes feedback for the first hidden-layer forward update; U1 does so for the second hidden layer. Candidate hypothesis: U1-only retains a useful adaptation benefit while limiting the additional plasticity introduced into the first hidden-layer update. W0 still learns, and its error depends on the evolving network; this is not a frozen-feature model.

Five proposed arms: frozen, U0-only, U1-only, both-U and BP. Primary **U1-only versus frozen**, with the same mean >=5% CE reduction/no negative seed and both per-seed 2pp retention guards. Both-U serves as the positive adaptation control. Before training, verify that both-U and the preserved V/U-only policy have bitwise-identical W/U/B trajectories even though V differs, along with the original numerical/cap/replay checks. Then use three fresh paired main seeds and a separate pilot, same full feedback rates, FIFO, warmup, cap, exposure and endpoints. No rate sweep or post-outcome primary selection.

Expected **250–450 aggregate CPU seconds**, inside the standing 900s budget and maximum three one-thread workers, with four small checkpoint files. This next experiment is **proposed only, not implemented or running**; await the user's continuation. No language/1B/cloud expansion is implied.

## Final cleanup reconciliation

After initial publication, a final local check found part of the temporary runtime had reappeared (646,352,286 bytes in 10,212 files); the cause is not established. No experiment worker was running and no incomplete progress remained. Those reproducible files were removed again, with all registered source and completed evidence unchanged. [Cleanup receipt](../results/feedback-family-20261018-v1/POST_PUBLICATION_CLEANUP.json).
