# Experiment report: slow-feedback-20261018-v1

Status: **completed; no promotion, useful-effect gate failed**.

Slowing the whole feedback update to0.1 passes old-fact retention in all three fresh seeds, but retains little of normal adaptive feedback's learning benefit. Primary post-change CE reduction is **1.6833%**, below the predeclared5% minimum; normal adaptive feedback gains25.3142% but fails retention. Neither setting passes the joint criterion.

Approval: user “Ok, go ahead and continue”, following the feedback-plasticity proposal. Exact source/protocol commit: `272e90a306793d60ba9265bcf6b1ce21985e819a`; protocol SHA256 `dc30899a06125b20216eaf4dabbb9a2ca367b0d9c7eeb2454732fd431c5c2674`. All20 publication files were read back byte-equal and the branch updated before numerical verification or optimization. [Protocol](../results/slow-feedback-20261018-v1/PROTOCOL.json) · [Registration](../results/slow-feedback-20261018-v1/REGISTRATION.json) · [Publication receipt](../results/slow-feedback-20261018-v1/PUBLICATION.json).

## Question and controls

Does slowing feedback plasticity retain a useful post-change adaptation advantage while keeping old-fact loss within2pp of frozen feedback?

Four arms share the same model, FIFO512 memory with16 replay samples,16 current examples per update,4096-step common warmup, forward optimizer, current/query data, training exposure and one configuration. Controls are exact backprop, frozen warmup-learned feedback, and normal bounded adaptive feedback. The candidate uses0.1 of the entire feedback SGD drive, including its weight decay, **before** the unchanged spectral cap2. Its forward update formula is unchanged. Warmup uses the original full feedback update in every arm.

The base feedback-rule factor0.8 remains part of the local drive. Online effective feedback step is0.005 for slow versus0.05 for normal adaptive, with the same pre-update local equations. Feedback decay is0.001 for B matrices and0.01 for V/U inside that scaled drive. Forward lr0.05 and decay0.001 remain fixed. No phase, query, error threshold or source classification enters the multiplier. Projection can alter the final displacement; a0.1 pre-projection step is not a claim of exactly0.1 cumulative movement.

The sole primary comparator is slow versus frozen. Normal adaptive reveals whether apparent retention sacrifices the established adaptation benefit; backprop remains the strong exact-gradient control. This run changes feedback learning and decay together, so it cannot distinguish their roles. See [rule mapping](../results/slow-feedback-20261018-v1/RULE_MAPPING.md).

## What actually ran

Fresh paired main seeds1001811,1001812,1001813; separate mechanical pilot99907. Main4096 common warmup plus2048 online steps per arm, four512-step phases. Pilot4096+256, four64-step phases. All12 main and4 pilot arms completed without an experiment failure, retry, configuration change or seed exclusion.

The unchanged synthetic threshold-world has11 inputs and4 binary labels; new contexts precede a later rule change. Queries use128 examples per context main and32 pilot. Warmup exposes1,048,576 binary targets per seed; main online262,144 per arm and pilot32,768. FIFO preserves historically observed labels, including temporarily obsolete targets; all arms share identical replay IDs. Obsolete binary replay-target counts are24/22/30 by main seed. No external dataset or official test was accessed.

Python3.12.14, PyTorch2.14.1+cpu, NumPy2.3.5 and pinned dependencies in RUNTIME.json. Temporary dependency setup encountered network timeouts, resolved by automatic retries before any experiment execution; versions were unchanged. One CPU thread per worker, at most three concurrent workers. The exact predecessor registration hashes were checked before copying seven reference files unchanged. Earlier missing-source evidence is not silently reconstructed.

All original FP32 absolute2e-6 / FP64 absolute2e-12 checks pass. Independent dense slow-step checks cover three seeds, two dtypes, normal and active-projection states: maximum errors3.5763e-7 and4.9960e-16. Same-state forward updates are bitwise identical for frozen/slow/normal feedback. Multiplier0/1 endpoints and12-step trajectories match the preserved frozen/normal policies exactly. Input isolation, detached outputs, no-autograd local path, spectral cap and independent3x128-step FIFO references also pass. [Verification](../results/slow-feedback-20261018-v1/VERIFICATION.json).

Every one of16 online trajectories was replayed from serialized warmup with independently regenerated/hash-verified data. All four phase and final complete-state hashes and stored final tensors are bitwise identical. No source or gate changed after registration. Per-seed data, source, warmup and state hashes remain in RESULT.json.

Total charged experiment CPU **163.694s /900s**. Main children including imports used139.111s aggregate and47.014s elapsed. Main warmup used40.180s; exact online replay used39.778s. These are components of the aggregate, not additional costs. Import canary, verification, pilot, diagnostics, launchers and analysis startup are charged; dependency setup/source/report preparation and transfers are excluded.

## Results

Primary metric: mean novel CE at steps544..768 and changed-rule CE at1056..1280, every32 steps,16 checkpoints total, nats per binary target. Each seed's paired relative reduction is averaged across seeds. Old-window accuracy averages stable facts at the same checkpoints; final-old checks the terminal state. Stable facts exclude the two deliberately changed relations in context0. Positive CE reductions favor the candidate; negative retention deltas indicate loss.

| Seed | Slow CE | Frozen CE | CE reduction | Old-window delta | Final-old delta |
|---|---:|---:|---:|---:|---:|
| 1001811 | 0.969424 | 0.989173 | 2.00% | -1.13pp | -0.28pp |
| 1001812 | 1.015704 | 1.045915 | 2.89% | -1.36pp | -0.50pp |
| 1001813 | 0.937379 | 0.938928 | 0.16% | -0.95pp | -0.39pp |

Mean CE reduction **1.6833%**, sample SD1.3885 percentage points. Every seed is positive, but the mean is below5%; **effect gate fails**. Old-window losses0.95–1.36pp and final-old losses0.28–0.50pp satisfy both<=2pp limits in every seed; **retention gate passes**. The joint gate fails. Mean old-window delta is-1.1463pp; mean final-old delta-0.3906pp.

Normal adaptive versus frozen, a preregistered secondary comparison:

| Seed | Normal adaptive CE reduction vs frozen | Old-window delta | Final-old delta | Normal CE gain preserved by slow |
|---|---:|---:|---:|---:|
| 1001811 | 23.66% | -8.93pp | -2.06pp | 8.44% |
| 1001812 | 22.66% | -6.80pp | -4.02pp | 12.75% |
| 1001813 | 29.63% | -10.74pp | -4.02pp | 0.56% |

Normal adaptive gains25.3142% CE, sample SD3.7673pp, but fails both retention measures in every seed. Slow preserves only7.248% of the normal adaptive CE gain on average **when averaging the three per-seed fractions**. This differs from dividing the two mean relative reductions. Descriptive slow-versus-normal CE increases average31.9333%.

All-arm primary CE for every seed:

| Arm | CE1001811 | CE1001812 | CE1001813 |
|---|---:|---:|---:|
| Backprop | 0.736537 | 0.779363 | 0.677176 |
| Frozen | 0.989173 | 1.045915 | 0.938928 |
| Normal adaptive | 0.755130 | 0.808947 | 0.660766 |
| Slow adaptive | 0.969424 | 1.015704 | 0.937379 |

Final mean heldout accuracy over the three seeds:

| Arm | Old accuracy | Novel accuracy | Changed accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| Backprop | 96.95% | 94.50% | 81.12% | 90.86% |
| Frozen | 94.27% | 79.92% | 36.46% | 70.21% |
| Normal adaptive | 90.90% | 84.90% | 69.27% | 81.69% |
| Slow adaptive | 93.88% | 81.45% | 37.63% | 70.99% |

Relative to normal adaptive, slow gains2.9762pp final-old accuracy and7.6776pp old-window accuracy, but loses31.6406pp final changed accuracy and10.7050pp balanced accuracy. Its final changed accuracy37.63% is close to frozen36.46%, compared with69.27% for normal adaptive. Backprop has the strongest final balanced accuracy90.86%; early post-change CE does not establish overall model superiority.

Registered endpoint decomposition, descriptive means:

| Arm | Novel-window CE | Changed-window CE |
|---|---:|---:|
| Backprop | 0.577723 | 0.884327 |
| Frozen | 0.770855 | 1.211822 |
| Normal adaptive | 0.843980 | 0.639249 |
| Slow adaptive | 0.775335 | 1.173004 |

Normal feedback's CE advantage arises from the changed-rule window while its novel-context window is worse than frozen. Slow also produces only a small changed-window gain. This is an observed decomposition of the fixed endpoint, not a new selected metric.

Mean final feedback displacement from warmup is0.7642 for slow versus6.6250 for normal adaptive, measured as L2 distance in the complete orthonormal Fourier coordinates. Frozen and backprop have zero feedback displacement. This confirms differing trajectories but does not attribute the change to local learning versus decay or prove alignment with backprop. Full same-state direction probes, prequential traces and costs are retained in per-seed RESULT.json. [Frozen summary](../results/slow-feedback-20261018-v1/SUMMARY.json).

Three paired synthetic worlds support a bounded mechanism result, not population significance, language-model transfer, novelty, scalability or deployment efficiency. No rate or comparator was selected after observing outcomes.

## Deployment measurements

Forward parameters1504 (6016 FP32 bytes), feedback12544 (50176 bytes) before metadata/container overhead. Active backprop needs only forward weights; experiment bookkeeping retains shared feedback. Every replay memory holds512 examples:30720 FP32 payload bytes plus4096 int64 provenance bytes, excluding container/RNG overhead. Slow adds only a fixed scalar, not a new model/state bank. These components are not total RAM; precomputed full data/query arrays serve benchmark reproducibility.

Observed main-process peak RSS378,636–378,848KiB **per worker**, including framework, full streams and diagnostics. Aggregate simultaneous peak, allocator reservation, activations/workspace breakdown, FLOPs, quantization, throughput and deployment latency are **unmeasured**. Prefill/decode is not applicable to this feedforward synthetic task.

Four compressed NPZ checkpoints total703,631bytes. Each stores common warmup W/F, final W for all arms, and final feedback for normal/slow adaptive arms; BP/frozen reference warmup feedback. All four remote readback hashes match. [Artifact manifest and restore instructions](../results/slow-feedback-20261018-v1/ARTIFACT_MANIFEST.json).

## Decision

**No promotion.** Slowing the entire feedback update to0.1 meets retention by surrendering most of the measured adaptation advantage. The result rejects this specific useful-effect hypothesis at the predeclared threshold; it does not prove that every intermediate rate or feedback mechanism must fail. Do not start an unregistered rate sweep.

The simplest unresolved explanation is the role of **feedback decay**. Normal adaptive feedback changes both local learning drives and decay relative to frozen; this run slows both together. We have not shown that its adaptation benefit requires learned feedback changes beyond passive decay.

Next proposed experiment: hold FIFO, normal feedback step, common warmup, forward optimizer and all endpoints fixed, and cross feedback data-dependent learning on/off with feedback decay on/off. Arms: frozen(neither), decay-only, learning-only, normal adaptive(both), plus backprop. Candidate primary: normal adaptive versus decay-only with the same>=5% CE effect/no negative seed and<=2pp per-seed old-window/final-old guard. Learning-only and frozen identify decay interactions. If decay-only explains most of normal adaptive's advantage, the learned-feedback mechanism claim is weakened. If data-dependent updates add a useful effect beyond decay but retention still fails, continue to report failure of the joint objective.

Use three fresh paired seeds and a separate mechanical pilot, one fixed design, full source-before-execution and exact-replay requirements. Expected200–350 aggregate CPU seconds, inside the standing900s envelope, max3 one-thread workers, no GPU/cloud and four small state files. This follow-on is **proposed only, not implemented or running**; wait for the user's next continuation. No expanded resource authority is implied. Language/1B training remains paused.

## Integrity and cleanup

All16 registered source/protocol/runtime files retain their frozen hashes. Verification/pilot evidence was preserved at `3b95cd068c33e604f67da12e1349d6efe41a11da` while main ran unchanged. All16 full online replays and four remote checkpoint SHA256 checks pass. [Audit](../results/slow-feedback-20261018-v1/EVIDENCE_AUDIT.json) · [Terminal](../results/slow-feedback-20261018-v1/TERMINAL.json).

All workers exited. Redundant readback copies and1,083,929,843bytes of temporary runtime files were removed after durable preservation. A redundant completed pilot progress snapshot was checked exactly against RESULT.json and removed; no partial/failed experiment evidence was discarded. Completed artifacts and original archives remain intact. No cloud resources, official tests, external datasets, emails, alerts or scheduled jobs were used or created. The older bounded-feedback missing-source gap remains explicit; current evidence derives from fully preserved predecessor source.
