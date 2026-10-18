# Super-1 source reconciliation and next experiment

Date: 2026-10-16. Status: completed read-only review; no Super-1 training or evaluation executed; no PR submitted.

Repository inspected: `rohanarun/Super-1`, main commit `e9afef41fe85de826e8edc97dcf2ffeb6917f2f3`. Twenty-one retrieved text files match their Git blob identities and the pinned tree. This is selected-source verification, not a full repository/checkpoint audit or a benchmark replay. The repository is reported public by GitHub at inspection, despite historical documents describing a private repository.

## Main finding

The recovered implementation already contains the strongest concrete connection from our earlier discussion: **GRU gates whose affine maps include cubic-spline-generated low-rank weight residuals**. We should now ground experiments in that code instead of inferring the architecture from descriptions.

The original `HierarchicalDistributionStudent` and the separate `HierarchicalLocalFunctionWeightsStudent` are distinct arms. The former uses ordinary GRU cells; the latter uses function-generated gate maps and also adds compressed local attention. Our prior rotating-vector and coefficient-state experiments are external prototypes, not replications of either arm.

Authoritative source: [recovered predictive_models.py](https://github.com/rohanarun/Super-1/blob/e9afef41fe85de826e8edc97dcf2ffeb6917f2f3/auto-research/results/cloud-recovery-20261015/exact-historical-source/representation_experiments/predictive_models.py), SHA-256 `b73a7c07801f5672adb52c0cb16a3c772d62e50eebc0366491d497f76d13f73f`. The retrieved bytes also match the latest synthetic experiment's source manifest.

## What the code actually does

| Component | Recovered Super-1 implementation | Our completed external pass 7 |
|---|---|---|
| Fast update | GRU proposal from token embedding, segment state, document state and previous fast state | Four parallel rotating-vector banks or coefficient delta-memory banks |
| Slow updates | Accumulate fast states; update segment GRU at segment boundary; update document GRU after the configured number of segments | Every bank advances each token, with different learned leak scales |
| Persistent native state | Fast, segment, document and segment-sum tensors, plus counters | Bank state plus spectral history and three prior embeddings |
| Readout | Concatenation of fast/segment/document, learned two-layer combine MLP, final LayerNorm, tied embedding logits | Spectral likelihood-shaped bank mixing, curve FFN refinement, final RMSNorm, tied embedding logits |
| Weight curves | Separate function-weight arm: dense affine base plus a sum of spline-weighted low-rank residual maps | Conditional full/low-rank refinement bases; pass 7 additionally stores coefficients of associative weight matrices |
| Meaning of curve coordinate | Integer phase cycles over a sampled spline design; fast phase follows token position, slower phases follow their own update counters | Refinement-depth coordinates and persistent coefficient-memory coordinates are separate experiments |
| Local context | Function/local arms include bounded compressed attention and its key/value state | Four-tap depthwise local embedding context |
| Training/data | Historical configuration uses Qwen vocabulary/teacher, width selected toward a 120M budget, 10M training tokens | 2,048-token vocabulary, small local teacher, 215,040 text input positions per arm, one seed |
| State carry | Native `step` accepts/returns state; `forward` resets at each call; training carry must be supplied explicitly | Full-prefix model; random training windows reset state, including the occasional 1,024-token episode |

For row-vector input x, the function-weight affine is

`base(x) + sum_k phi_k(phase) * ((x @ L_k) @ R_k)`.

The spline coefficients are fixed samples of an open-uniform cubic B-spline design, while the low-rank factors are trained. The historical function configuration specifies 24 virtual phases, 6 controls and rank 32. The code does **not** execute 24 refinement layers per token: it selects one phase row. It does not remove the dense base, embedding, gates or local attention storage.

The native clocks in that configuration are segment length 32 and 8 segments per document update. A slower state is not simply an exponential bank with a longer half-life. The nonlinear GRU state dependence prevents treating our affine scan as an equivalent acceleration.

The familiar 2,752 scalars at width 688 are four *different* native tensors. Our earlier four banks happened to total the same number before extra history was added; equality of scalar counts does not establish architecture equivalence. The function/local arm additionally carries local keys and values.

## A documentation distinction worth making explicit

`EXPERIMENTS.md` E04 describes an original mean readout and proposes a transport/readout factorial. That assumption does not describe the recovered historical `HierarchicalDistributionStudent`, whose combine MLP already learns from concatenated scales. The instruction may target another preserved hierarchy family. Before implementing E04, bind its "original" to a specific class/hash. Do not replace the historical combine MLP with a mean and label that an unchanged baseline.

Similarly, the catalog contains historical missing-artifact statements. The newer cloud-recovery README supersedes them for the recovered native source and nanoGPT checkpoint. No renewed missing-file search is warranted from those older paragraphs alone.

## Existing negative result that changes the next step

The latest merged experiment, `synthetic-rotation-radial-20261016-v1`, reports:

| Seed | Native readout CE | Rotation | Equal-count radial control | Rotation minus radial |
|---|---:|---:|---:|---:|
| 73 | 3.906930447 | 3.912286192 | 3.909294099 | +0.002992094 |
| 191 | 4.037421405 | 4.042921543 | 4.040547907 | +0.002373636 |
| 307 | 3.879141450 | 3.911892533 | 3.871451885 | +0.040440649 |

These are saved synthetic confirmation results in nats per readout target, not newly evaluated language scores. Rotation loses to radial and native in all three seeds. The saved mean paired gap is +0.015268793 and the recorded interval crosses zero. The existing decision is no promotion.

Reading the generator and runner supports the report's objective concern: each document supplies 28 unpredictable command targets and 4 state-readout targets, while training uses all-token CE and the primary metric scores only readouts. Readout losses worsened for the hierarchy arms. The nanoGPT readout loss is close to ln(8), which is consistent with little useful accumulator learning. This does not establish that a loss-mask change will solve the task; it makes that a cheap next diagnostic.

Previous readout tokens are fed as ordinary teacher-forced context. Consequently, each subsequent group supplies a fresh true-state anchor: this task is not answer-free retention across the entire document. That distinction should remain explicit if it is reused for a long-context claim.

Sources: [merged report](https://github.com/rohanarun/Super-1/blob/e9afef41fe85de826e8edc97dcf2ffeb6917f2f3/auto-research/reports/synthetic-rotation-radial-20261016-v1.md), saved `execution/RESULT.json`, `generate_data.py`, `geometry.py` and `run_experiment.py` in the same run directory. Metrics were reconciled to the saved JSON and the paired gaps recomputed; no checkpoint inference was run.

## External findings worth carrying forward, with their limits

Our completed pass 7 is useful hypothesis-generating evidence. All ten endpoints are retained in `EXTERNAL_RESULTS.json`, including the worse and null variants. The principal observations are:

- Learned coefficient memory reduced analytical persistent state from 3,072 to 1,008 scalars, while CE128 rose from 5.188497 to 5.294224. It also used fewer trainable parameters, so this is not an equal-capacity comparison.
- A learned basis beat the fixed DCT basis in that screen, with 473,344 additional trainable basis parameters across the four banks.
- Scalar memory feedback into the write gate and midpoint plasticity effectively tied the learned-coefficient control. They are not full self-referential weight matrices or faithful SRWM reproductions.
- Slowing all banks with independent writes hurt this short text task. The follow-up preserving fast banks was built and passed targeted checks but **was not trained** when repository access arrived.
- Retention supervision improved query loss without establishing reliable old-association retrieval. The extra-objective arms also received 100 additional updates, disclosed separately.

Every pass-7 arm has one training seed. These token-CE scores cannot be ranked against the repository's byte CE, synthetic readout CE or Qwen-vocabulary CE. No external prototype is promoted into Super-1 by this review. Model checkpoints, teacher and corpus remain in the separately preserved external project; this contribution includes compact results, recipes and source hashes, not a new self-contained training replication.

## Concrete next experiment, not launched

The draft keeps the exact recovered native hierarchy and existing angular/radial wrappers, changes training to **readout-only loss**, and uses fresh generated documents. Four arms: native, angular, radial, pinned nanoGPT; three paired seeds; one optimizer setting each. Angular versus radial remains the primary comparison. The native and nanoGPT arms diagnose whether the task is learnable at the budget.

Proposed ceiling: one CPU thread, 300 CPU seconds, 10 minutes wall time, 50 MiB artifacts, no external corpus, dependency installation, paid compute or GPU. A fixed 400-update budget per arm/seed is proposed, with timeout recorded as incomplete rather than a shortened successful run. This is a resource cap, not a measured runtime prediction for this host.

The draft includes both full-vocabulary and eight-choice readout CE, readout accuracy, an explicit chance/learning check, unchanged geometry controls, fresh train/selection/confirmation seeds, final-checkpoint selection only, and exact duplicate scoring after checkpoint freeze. All arms see the same token histories; the known positional-feature difference between native hierarchy and nanoGPT remains a caveat. If no arm learns the state task beyond a simple answer-vocabulary predictor, stop interpreting it as a useful geometry discriminator. A timing-feature control or simpler task can then be proposed separately.

`NEXT_EXPERIMENT_DRAFT.json` is planning material. It is not preregistration, execution authorization or a running job. No existing run ID, source or confirmation region is overwritten.

## What is blocked and what is complete

The connected GitHub account can read upstream but has no push permission. No accessible `JRowe47/Super-1` fork was found, and the available connector has no fork-creation operation. A PR has therefore not been opened. The contribution is supplied as an applyable patch against the pinned commit, with an explicit review registry entry and manifest additions. It contains no claimed new native experiment.

The repository's `auto-research.md` and `SCALING_1B.md` require execution on a genuinely mounted `/Volumes/SP AI 01_16/CodexWorkspaces` destination and explicitly disallow silently substituting another path. That mount is absent in this hosted workspace. The draft can be reviewed here, but running the repository workflow here requires the user's explicit workspace exception or access to the specified mounted environment. No mount, authorization or successful execution receipt was fabricated.

This session read source/evidence, verified selected text identities, reconciled saved numbers and prepared a contribution. It did not execute repository model code, read official test data, fetch model weights/corpora, provision resources, create alerts or send messages. The earlier external hybrid smoke check finished before the new repository instructions were read; it is separately labeled and contains no optimizer training run.
