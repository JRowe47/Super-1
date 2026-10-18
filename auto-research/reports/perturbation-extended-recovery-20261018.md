# What the W560 perturbation research actually tested

The [extended recovery bundle](../results/perturbation-extended-recovery-20261018-v1/) adds source and saved evidence from the SSD and resident iCloud backup files. It supplements the [original frozen optimizer recovery](../PERTURBATION_OPTIMIZATION.md). This is an evidence audit, not a fresh numerical replay or a complete inventory of every historical research run.

## Optimization experiments

| Experiment | Preserved evidence and outcome |
|---|---|
| Paired structured perturbations | The original solver scores positive/negative directions, combines promising directions, performs line search and adapts its radius. Groups include base affines, left/right factors and codecs. |
| Winning-direction line search | The later decision records contain **45 line-search candidate rows** along previously successful changes. |
| Tensor-block refinement | **219 candidates across 73 block searches**, with **19 accepted block refinements** in those records. |
| Single-weight refinement | **264 candidates across 88 scalar searches**, with **11 accepted scalar refinements**. |
| Gradient-informed scalar diagnostic | A separate CPU FP32 pilot performed **one backward pass**, ranked scalar changes using gradient/delta information, and screened **eight modifications**. It wrote no checkpoints or heads and had no promotion authority. Its CPU scores are explicitly not numerically comparable to canonical MPS BF16 scores. |
| CE versus knowledge distillation | Both arms used the same starting checkpoint and raw random seed stream, with **180 search evaluations per arm** and **13 provisional search acceptances each**. Adaptive trajectories could diverge after acceptance. Both selected outputs were ultimately `parent_noop`, with final mean CE **6.224467061460018**, identical to the parent. The method label selected CE, but **the unchanged parent was retained and no output was promoted**. |

Counts of local block/scalar acceptances are not counts of independently promoted checkpoints. The later hybrid ledger contains **21 heads and 21 decisions: 20 accepted decisions and one rejection**. Its initial head imports model generation 8, and its last head records **model generation 28**, with selected duplicated mean CE **6.212864637374878**. Authority generation and model generation are different counters.

The same ledger contains a decision recording the chart's **6.218782722949982** tensor-block result as a materialized generation-11 promotion. This provides later documentary evidence beyond the PNG's earlier “G11 materializing” annotation. The external checkpoint bytes were not rehashed or evaluated in this recovery, so this is a verified saved record, not fresh checkpoint validation.

All these canonical optimization scores use reused selection data. They do not establish held-out generalization. The later partial-verification policy uses strict duplicated improvement with **zero margin**, whereas the earlier campaign threshold was **0.0001 relative improvement**. Six later decisions explicitly report that not every completed-worker candidate was evaluated twice. Do not claim exhaustive verification or the original strict threshold for every later promotion.

The CE/KD final bank is labeled one-time method-selection data and forbidden for continuation reuse. It is separate from the later reused verification bank. Do not pool its loss with the continuation trajectory.

## Downstream benchmark canary

A saved reporting-only canary compared two checkpoints on **128 examples per task**, using fixed first-N pinned source order. The result explicitly labels repeated/adaptive benchmark use as audit data, not held-out evaluation.

| Task | Prior accuracy | Current accuracy |
|---|---:|---:|
| ARC-Easy | 28.90625% | 28.90625% |
| HellaSwag | 19.53125% | 19.53125% |
| LAMBADA | 0% | 0% |
| PIQA | 53.125% | 52.34375% |

Saved likelihood metrics improved on these samples, but accuracy did not improve; PIQA lost one correct answer. This small canary does not support broad capability claims. It was not permitted to gate optimizer promotion.

## Integrity tests and unsuccessful attempts

The historical host/worker/binder suite reports **43 tests: 42 passed and one skipped**. The manager suite reports **74 tests: 69 passed and five skipped**. They cover exact candidate storage, numerical failures, duplicate evaluation, runtime/manifest binding, credentials handling, recovery and cloud-resource lifecycle behavior. These are stored historical logs, not newly rerun tests.

The earlier recovery freshly reran all **11 original frozen integrity tests**, successfully. That separate receipt remains in the original bundle.

This supplement also preserves unsuccessful pipeline attempts: empty logs, a missing NumPy import, writable upload staging rejected by its seal, and a manager-stage failure. These logs do not themselves prove a successful cloud launch or cleanup. A separate parent-versus-child audit scaffold was explicitly cancelled before loading a checkpoint or evaluating the model. Neither a configured experiment nor a “started” marker is counted as a completed evaluation.

## Can this learn new concepts?

Perturbation-based optimization changes weights in response to feedback. It can therefore learn useful behavior without backpropagation; [MeZO](https://arxiv.org/abs/2305.17333) demonstrates forward-only language-model fine-tuning. That result is not evidence that this W560 procedure acquired a new concept, nor that arbitrary new concepts are efficiently learnable with this particular search space or budget.

Our preserved experiments show local optimization on repeatedly reused data and a small downstream canary without accuracy gains. They do **not** distinguish memorization, recalibration, better use of existing representations, and acquisition of a generalizable new rule. The separate gradient diagnostic must also not be described as purely forward-only.

An exact function-preserving permutation leaves the network's behavior unchanged; it cannot teach a concept by itself. [Git Re-Basin](https://arxiv.org/abs/2209.04836) uses permutation symmetries to align parameterizations. Symmetry could potentially inform subsequent function-changing perturbations, but we found no completed permutation-symmetry-guided perturbation experiment in the records inspected. A permutation must be applied consistently to all coupled tensors/states to be a true symmetry; arbitrary shuffling is not guaranteed to preserve the model.

A proposed concept-learning test would first demonstrate that the initial model cannot apply a specified rule, then compare perturbation training with backpropagation and an unchanged baseline using matched data and disclosed compute budgets. Freeze choices before testing unseen combinations, instances and paraphrases, with memorization controls and multiple seeds. Any symmetry-guided variant should be a separate arm against the existing perturbation policy. This is a proposal only; no such experiment was launched or authorized by the archival upload.

## Fresh verification and limits

The [standard-library verifier](../results/perturbation-extended-recovery-20261018-v1/verify_recovery.py) checks all imported hashes/sizes, parses the source and JSON, verifies content-addressed decisions/heads and parent-child checkpoint identifiers, recomputes saved duplicate score means, derives the refinement counts, recomputes benchmark accuracy from example correctness, and checks the CE/KD parent-retention result. The [receipt](../results/perturbation-extended-recovery-20261018-v1/VERIFICATION.json) records the results.

The audit checks **1,128 saved duplicate vector pairs**; repeated parent evaluations across decisions are not independent experiments. It does not validate the external model/data payloads or replay model scores. Historical source, failures, metadata and receipts remain byte-identical, and all exclusions/external dependencies are disclosed in the import inventory.
