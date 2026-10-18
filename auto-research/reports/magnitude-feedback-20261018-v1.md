# Experiment report: magnitude-feedback-20261018-v1

Status: completed; primary effect passes, retention fails; **no promotion**.

Adaptive feedback's post-change CE advantage survives a control that matches hidden-layer update magnitude: **26.45% mean paired reduction**, with all three seeds positive. It still loses **4.91–8.26 percentage points** of final still-valid-old accuracy against that control, exceeding the unchanged 2pp limit in every seed. Larger updates alone do not explain the adaptation gain in this synthetic task; useful retention remains unresolved.

Approval: user asked to continue to the next stage, using the existing repo/research scratch and small project artifacts. Exact frozen source/protocol were committed and read back before verification/training at `644d69ed8e797e3e06de0af416fa561a8bd6f148`. [Run files](../results/magnitude-feedback-20261018-v1/), [protocol](../results/magnitude-feedback-20261018-v1/PROTOCOL.json), [source registration](../results/magnitude-feedback-20261018-v1/REGISTRATION.json). Protocol SHA256: `28b8b77f0941208696d3fd2965fccd44359a223974c9b3c860d26c68a9627b1d`.

## Question and controls

The primary comparison is `adaptive` versus `frozen_match_adaptive`. The latter keeps learned feedback directions fixed and uses a separately adapting local shadow bank to supply the two hidden-layer SGD-drive norms, evaluated at its own current forward state and training batch. The reciprocal control normalizes adaptive drives to an immutable frozen bank's norms. Decay is included in the matched drive; readout W2 is unchanged. No exact hidden gradient or heldout information enters either local update policy.

All arms share the same 4096-step BP forward warmup with bounded local shadow feedback, data, fresh world seeds, step count and fixed optimizer. Six arms include exact backprop and original random frozen feedback. Cap 2 uniformly scales each complete Fourier feedback store after initialization/updates; full dense SVD cost is charged. No low-rank approximation, tuning, early selection or discarded seed.

Matching occurs at each control's own state; trajectories can diverge. The matched arms require an extra 12544-coefficient feedback bank and two local calls, so this is neither a compute nor a deployed-memory matched comparison. Adaptive feedback can outperform this control without becoming a closer approximation to BP.

## What actually ran

Python 3.12.14, numpy 2.3.5, torch 2.14.1+cpu, deterministic FP32 CPU execution, max 3 concurrent one-thread workers. Main seeds 701781/2/3 each received 4096 shared warmup steps and 2048 updates per arm; pilot 96977 received 4096+256. Warmup exposes 1048576 binary targets per seed; main online exposes 262144 per arm, pilot 32768. The stream uses 16 current plus 16 replay examples per update; FIFO 512 keeps historical labels, including stale labels after drift. Disjoint RNG streams generate warmup/current/replay/heldout sets. All data hashes and source hashes are in each seed's RESULT.json.

All 18 main and6 pilot runs finished finite. All 24 full online replays, from serialized warmup and independently regenerated/hash-verified data, reproduce all four phase hashes, final actual states and shadow states bitwise. The preserved independent dense-rule checks pass original FP32 absolute 2e-6 and FP64 absolute 2e-12 tolerances, without relaxation. Extension max absolute errors are 2.384e-7 and 5.551e-16. All 25600 hidden-layer step matches pass; maximum relative norm error `2.72932152e-07` against 2e-6.

A first pilot launch failed at runtime import before data generation or model updates because sympy was missing. Its unchanged failed receipt, exact dependency repair and successful deterministic import canary are preserved. Runtime-only amendment was committed before retry; source, seeds, gates and parameters did not change. The old `bounded-feedback-20261018-v1` source remains unavailable after the workspace reset. This is a fresh extension of the hash-verified preserved predecessor, not a byte-identical reconstruction of the lost study.

## Results

Primary CE is the equal 16-checkpoint mean: novel-context CE at steps 544..768 every 32, changed-rule CE at 1056..1280 every 32. Lower is better. Old-window retention uses still-valid-old accuracy over those same checkpoints; final retention uses step 2048. Deltas below are adaptive minus magnitude-matched frozen, in percentage points.

| Seed | Adaptive CE | Matched frozen CE | CE reduction | Old-window delta pp | Final-old delta pp |
|---|---:|---:|---:|---:|---:|
| 701781 | 0.700232 | 0.959273 | 27.00% | -11.89 | -6.36 |
| 701782 | 0.711086 | 0.964268 | 26.26% | -11.43 | -8.26 |
| 701783 | 0.714017 | 0.965916 | 26.08% | -6.62 | -4.91 |

Mean paired reduction **26.4464%**, sample SD **0.4910 percentage points**. The effect gate requires mean >=5% and no negative seed; it passes. The retention gate requires both old-window and final losses <=2pp in every seed; it fails in all three. Three paired synthetic worlds provide descriptive evidence, not a population significance or language-quality claim.

Secondary comparisons are consistent: unnormalized adaptive versus frozen improves CE by 28.4448%; adaptive with frozen magnitudes versus frozen improves by 28.9224%. Both also fail retention. These were declared secondary and cannot replace the primary decision.

Final mean accuracies (%):

| Arm | Still-valid old | Novel | Changed | Balanced |
|---|---:|---:|---:|---:|
| Exact backprop | 96.91 | 95.25 | 85.16 | 92.44 |
| Random frozen feedback | 92.71 | 90.53 | 67.06 | 83.43 |
| Learned frozen feedback | 93.45 | 83.63 | 39.71 | 72.26 |
| Adaptive feedback | 86.90 | 84.21 | 72.01 | 81.04 |
| Frozen direction, adaptive magnitude | 93.42 | 84.34 | 55.73 | 77.83 |
| Adaptive direction, frozen magnitude | 87.93 | 83.01 | 69.01 | 79.98 |

Backprop remains strongest on final balanced accuracy,92.44% versus 81.04% for adaptive. The adaptation endpoint and final broad quality are different questions.

Same-state hidden-layer diagnostic probes do **not** establish better BP alignment. On the adaptive trajectory, mean exact-drive cosine is 0.360 for adaptive versus 0.455 for frozen at W0, and0.397 versus 0.523 at W1. These are 12 samples per layer across 3 seeds and 4 post-warmup boundaries; they use training batches and never affect optimization. The gain survives magnitude normalization, but its direction may favor rapid changes at the cost of old constraints. That is an interpretation, not a proven causal mechanism. Exact aggregates are in DIAGNOSTIC_SUMMARY.json.

## Compute and deployment measurements

Charged experiment CPU totals **226.078s**, including the failed import, runtime canary, successful stages/launchers and frozen analysis, below 900s. Main workers including imports used 198.803s combined; main wall time 66.829s with 3 workers. Main common warmup cost 40.646s; full replay 65.504s. Per-arm main execution below sums all 3 seeds and includes evaluation/probes, so it is experimental cost rather than a pure training-speed benchmark:

| Arm | Aggregate CPU seconds |
|---|---:|
| Exact backprop | 9.836 |
| Random frozen feedback | 7.992 |
| Learned frozen feedback | 7.649 |
| Adaptive feedback | 16.491 |
| Frozen direction, adaptive magnitude | 21.189 |
| Adaptive direction, frozen magnitude | 21.555 |

The forward model has 1504 FP32 coefficients; ordinary local policies add 12544 feedback coefficients (14048 total), and matched policies add another 12544 (26592 total). The BP control's active model has 1504; its stored checkpoint retains unused feedback for the shared runner. Compressed retained checkpoints total **1,910,052 bytes** across four files. Main per-process peak RSS is 379388–379648KiB; this is not a measured concurrent aggregate or model-only memory. Activation/workspace breakdown, allocator peaks, streaming latency/p50/p95, deployment throughput and FLOPs are unmeasured. No deployment-efficiency claim.

## Decision and next experiment

**No promotion.** The fixed primary retention criterion was not met. A plausible next intervention is to split the same 16 replay slots between recent FIFO and a small fixed reservoir of old examples, while keeping current exposure, total minibatch and compute budget fixed. Use the full adaptive/frozen × FIFO/mixed-replay factorial so an improvement cannot be attributed to changed data allocation alone. Preserve historical labels, including conflicts after drift; no oracle relabeling or phase-aware removal of invalid old labels. Keep the same adaptation-effect and per-seed 2pp retention gates, fresh paired seeds, and preregister one mixture without tuning. Estimated cost remains roughly 250–450 aggregate CPU seconds under the standing 900s ceiling. This proposal is not implemented or running; substantial scaling would need new authority.

## Integrity, preservation and cleanup

[Terminal receipt](../results/magnitude-feedback-20261018-v1/TERMINAL.json), [evidence audit](../results/magnitude-feedback-20261018-v1/EVIDENCE_AUDIT.json), [checkpoint identities/hashes](../results/magnitude-feedback-20261018-v1/ARTIFACT_MANIFEST.json). All four saved state files were downloaded again and SHA256 verified. Source/protocol and compact results are in Git; checkpoint files are individual companions in the existing project folder. No cumulative archive or dataset copy was created. Temporary restored runtime, predecessor download and verification copies were removed (about 1.05 GB). All workers exited; no GPU/cloud jobs or scheduled alerts were created; official test untouched.

For reproduction, follow the frozen README and RUNTIME_AMENDMENT version pins. Download each state file under its manifest run_relative_path when inspecting saved states. For a fresh rerun, copy only the registered source files plus REGISTRATION.json to a new writable run directory; the launcher intentionally refuses existing receipts/seed outputs. Preserve this evidence rather than overwrite it. Current continuation entry: [research scratch](../research-scratch/CURRENT.md).
