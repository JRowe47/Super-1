# Experiment report: replay-retention-20261018-v1

Status: completed; effect passes, retention fails; **no promotion**.

Keeping a permanent bank of old examples improves retention but slows adaptation. For adaptive feedback, mixed replay raises final still-valid-old accuracy from **88.19% to 91.57%**, while changed-rule accuracy falls from **73.05% to 49.09%** and primary post-change CE rises **30.71%**. Adaptive feedback still beats frozen feedback under the same mixed replay by **21.94%** on CE, but its old-fact losses exceed the unchanged 2pp limit in every seed. This fixed replay policy is not a satisfactory retention solution.

Approval: user said “Alright, go ahead with the next,” referring to the fixed-budget replay experiment in the preceding report/handoff. Exact source and protocol were committed and byte-verified before experiment verification or training at `34f559d1454a067d175e55c372f3b20387b2423d`. [Run files](../results/replay-retention-20261018-v1/), [protocol](../results/replay-retention-20261018-v1/PROTOCOL.json), [registration](../results/replay-retention-20261018-v1/REGISTRATION.json). Protocol SHA256: `7449bfbfa874107ac3f8d64fe2e2dd8f82236882058b3bd018a1686af8972ee3`.

## Question and controls

Does reserving replay for old examples preserve a useful adaptive-feedback advantage without violating the original retention gate? Six arms cross backprop, frozen learned feedback and adaptive feedback with FIFO and mixed replay. The primary contrast is **adaptive_mixed versus frozen_mixed**. FIFO, within-policy replay differences and interaction effects are explanatory controls.

Both policies start with the same 512 old training examples and store exactly 512 examples. Each update uses 16 current examples plus 16 replay samples. FIFO samples 16 from a rolling queue of 512; mixed pins the first 256 old examples and rolls the other 256, sampling eight from each. Both consume the same 16 uniform replay RNG draws per step; modulo256 yields uniform draws for each mixed bank. Current examples enter memory only after constructing the batch. The recent horizon changes from 32 to 16 batches in mixed replay, so the treatment is an explicit memory-allocation policy, not an isolated sample-age change.

Historical labels remain unchanged, including old labels that later conflict with changed rules. There is no phase-aware filtering or oracle relabeling. Both schedules share identical current examples, warmup, query data and seed worlds. The model, complete Fourier representation, local equations, spectral cap 2, fixed learning rate and weight decay are unchanged. Previous magnitude-normalization controls are not active here; that question was tested in the preceding stage. No mixture, seed or learning-rate search.

## What actually ran

Three fresh main seeds **801791, 801792, 801793** and mechanical pilot **97987**. Each seed has a common 4096-step BP-forward/local-feedback warmup. Every main arm then runs 2048 updates across four phases of 512; every pilot arm runs 256 updates. Warmup exposes 1,048,576 binary targets per seed; online main exposes 262,144 per arm, pilot 32,768. One fixed configuration per arm, identical total training exposure and memory slots.

Python 3.12.14, numpy 2.3.5, torch 2.14.1+cpu; exact dependency pins are in RUNTIME.json. Deterministic FP32 CPU, at most three concurrent single-thread workers. All 18 main and six pilot runs finish finite. All 24 full online replays reproduce every phase and final complete-state hash bitwise from serialized warmup and independently regenerated/hash-verified schedules. Saved final tensors also match exactly.

Original independent dense/Fourier, gradient, cap and local-rule checks pass FP32 absolute 2e-6 and FP64 absolute 2e-12 without relaxation. The new mixed schedule matches an independent deque/per-example reference across three verification worlds. Tests confirm unchanged FIFO behavior, shared current/query arrays, causal replay, exactly 512 stored examples, 32-example batches, and persistent historical labels. No execution failures or post-registration changes.

## Primary result

CE is the equal mean of eight heldout novel-context checkpoints at steps 544–768 every 32 and eight changed-rule checkpoints at 1056–1280 every 32. Old-window retention uses still-valid-old accuracy at those same 16 checkpoints; final retention is step 2048. Deltas are adaptive mixed minus frozen mixed, in percentage points.

| Seed | Adaptive mixed CE | Frozen mixed CE | CE reduction | Old-window delta pp | Final-old delta pp |
|---|---:|---:|---:|---:|---:|
| 801791 | 1.001266 | 1.179227 | 15.09% | -3.75 | -3.52 |
| 801792 | 0.909231 | 1.193992 | 23.85% | -6.61 | -5.25 |
| 801793 | 0.893558 | 1.222238 | 26.89% | -6.53 | -5.19 |

Mean paired CE reduction **21.9442%**, sample SD **6.1265 percentage points**. The effect rule requires mean >=5% with no negative seed; it passes. The retention rule permits at most 2pp loss in both windows and final state for every seed; it fails in all three. The unchanged FIFO adaptive-versus-frozen secondary comparison also passes effect (29.08% mean CE reduction) and fails retention.

Three paired synthetic worlds provide descriptive evidence. No population significance, language-model quality, architecture promotion or universal claim about replay follows.

## Replay intervention within adaptive feedback

Mixed minus FIFO, keeping the same adaptive learner:

| Seed | Final-old gain pp | Old-window gain pp | Post-change CE increase | Final changed-rule delta pp |
|---|---:|---:|---:|---:|
| 801791 | +5.86 | +8.56 | +30.44% | -31.25 |
| 801792 | +0.33 | +10.01 | +31.10% | -26.95 |
| 801793 | +3.96 | +7.04 | +30.59% | -13.67 |

Mean final-old gain is **3.39pp**, old-window gain **8.54pp**, but post-change CE increases **30.71%** and final changed-rule accuracy falls **23.96pp**. Every seed exhibits the retention/adaptation tradeoff. Final novel accuracy also falls **3.71pp**. The adaptive-minus-frozen interaction in final-old replay benefit is +3.79, −0.73 and +2.96pp across the three seeds. It is mixed, so the retention gain is not uniformly specific to adaptive feedback. No interaction gate was registered.

Final mean accuracy (%), with balanced accuracy the equal mean of the three groups:

| Arm | Still-valid old | Novel | Changed | Balanced |
|---|---:|---:|---:|---:|
| Backprop / FIFO | 96.84 | 96.29 | 79.30 | 90.81 |
| Backprop / mixed | 97.54 | 94.47 | 57.55 | 83.19 |
| Frozen / FIFO | 94.85 | 85.16 | 41.93 | 73.98 |
| Frozen / mixed | 96.22 | 83.98 | 23.31 | 67.84 |
| Adaptive / FIFO | 88.19 | 84.08 | 73.05 | 81.77 |
| Adaptive / mixed | 91.57 | 80.37 | 49.09 | 73.68 |

The mixed policy also harms changed-rule accuracy for BP (**−21.74pp**) and frozen feedback (**−18.62pp**). This points to a replay-policy problem that extends beyond the local-feedback mechanism. BP with FIFO remains the strongest final balanced control at **90.81%**. Adaptive/FIFO is **81.77%** and adaptive/mixed **73.68%**.

Mixed replay exposes the three runs to **4,948 / 3,736 / 3,774 stale binary targets**, compared with **54 / 30 / 40** under FIFO. These are diagnostic counts, not inputs to the learner. A persistent conflict with outdated labels is a plausible contributor to the loss on changed rules. It is not isolated causally here: mixed replay also reduces recent coverage and allocates fewer replay slots to new contexts. Both primary endpoint components worsen under adaptive mixed: novel-window CE 0.8385→1.0501 and changed-window CE 0.5918→0.8193. [Component decomposition](../results/replay-retention-20261018-v1/ENDPOINT_COMPONENTS.json) is a descriptive aggregation of the registered checkpoints.

## Compute and deployment measurements

Charged experiment CPU: **199.535s** under the 900-second ceiling, including runtime import canary, numerical checks, pilot/main imports and work, complete online replays, launchers and frozen analysis. Main combined child CPU 175.218s; main wall 58.580s with three workers. Shared main warmup 40.729s and main replay 50.567s. Source preparation, dependency setup, transfer and report formatting are outside the training-cost measurement.

Per-arm main CPU sums across all three seeds and includes evaluation/probes; this is experimental cost, not a pure training-speed benchmark:

| Arm | Aggregate CPU seconds |
|---|---:|
| Backprop / FIFO | 9.791 |
| Backprop / mixed | 9.768 |
| Frozen / FIFO | 9.337 |
| Frozen / mixed | 9.687 |
| Adaptive / FIFO | 19.590 |
| Adaptive / mixed | 17.297 |

Forward model: 1,504 FP32 coefficients; local feedback adds 12,544 (14,048 total). BP's active model has 1,504; shared-runner checkpoints retain unused feedback for common-start reconstruction. Replay payload is **30,720 bytes** of FP32 x/y plus **4,096 bytes** of int64 IDs per policy, excluding object/allocator overhead. Both policies store the same 512 rows. Main per-process peak RSS 390132–390260KiB; concurrent total, activation/workspace breakdown, FLOPs and deployment latency are unmeasured. No efficiency claim.

The four compressed state files total **754,883 bytes**, using one common warm feedback bank for BP/frozen finals. No stream tensor dump or cumulative archive.

## Decision and next experiment

**No promotion of permanent old-example replay.** The intervention buys retention by sacrificing the ability to learn new and changed information, and still fails the primary retention guard.

The next useful test is a **causally refreshed long-history bank**. Keep the same 256 recent +256 long-history slots and 8+8 samples, but replace one old-bank row after each update with one uniformly chosen current example, in fixed circular order. Once the initial bank turns over, retained long-history examples are at most 256 updates old, without phase metadata or oracle relabeling. Compare FIFO, permanent-bank and refreshed-bank policies under adaptive feedback, frozen feedback and BP; register one refresh rate and fresh paired seeds. The permanent-bank control keeps storage, sample allocation and recent horizon identical, helping isolate the cost of never refreshing old examples. Retain the same primary adaptation/retention gates and report both novel and changed-rule components.

This is a proposed hypothesis, not an implemented or running follow-up. Estimated 250–450 aggregate CPU seconds, within the standing 900-second envelope; larger training still needs separate authority. If refreshing the bank cannot improve the joint tradeoff, reconsider the feedback plasticity rule rather than tuning replay mixtures indefinitely.

## Integrity, preservation and cleanup

[Terminal](../results/replay-retention-20261018-v1/TERMINAL.json), [audit](../results/replay-retention-20261018-v1/EVIDENCE_AUDIT.json), [checkpoint identities and hashes](../results/replay-retention-20261018-v1/ARTIFACT_MANIFEST.json). All four saved files were downloaded again and SHA256 verified. Source/protocol were saved before execution; results and receipts are in Git; individual NPZ companions are in the existing project folder. All workers exited; temporary runtime and verification copies removed; no GPU/cloud resources or scheduled alerts created; official test untouched.

The previous bounded-feedback source-loss caveat remains historical. This stage derives from the fully preserved magnitude-feedback source and has a complete preservation record. Follow the frozen README for reproduction in a fresh output directory; never overwrite completed receipts or seed outputs. Current continuation entry: [research scratch](../research-scratch/CURRENT.md).
