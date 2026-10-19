# SmolLM2-135M: learned depth curves, phase 3

This phase turns the first block's MLP residual transformation into a learned
curve over depth. It tests whether one fitted representation can be evaluated
with 8, 4, 2, or 1 steps. Attention executes once with its original weights;
all 29 later blocks, embeddings and final head remain frozen. One prespecified
composition also uses the phase-2 shared Q/K factors.

The principal deliverables are `results/phase3_report.html`, the summary and
trajectory figures, detailed result JSONs, and four reloadable checkpoints.
This is an experimental first-block distillation study, not a full-model
retraining or a reproduction of an unreleased GSA implementation.

## The curve and its integration

Let `a` be the residual state immediately after the original first attention
operation. Define a depth-dependent SwiGLU vector field `F(h,t)`:

    F(h,t) = D(t) [SiLU(G(t) RMSNorm(h)) * U(t) RMSNorm(h)]
    u[0] = 0
    u[i+1] = u[i] + F(a + u[i], i/N) / N
    block_output = a + u[N]

The original post-attention RMSNorm scales are frozen and reused inside the
field. `G(t)`, `U(t)` and `D(t)` are sampled from four clamped Catmull-Rom spline
control vectors at depths 0, 1/3, 2/3 and 1. These are C1 matrix-valued curves.
Euler evaluation uses the left endpoint `i/N` and integrates the same interval
`[0,1]` at every step count. The factor `1/N` is essential.

Unlike phase 1's flattened scalar-parameter index, this curve's coordinate is
depth. The spline coefficients do not depend on the input; the vector field
does, through its evolving hidden state. No quaternion, imposed sphere,
adaptive solver, hierarchical memory, or attention-cache reduction is used.
Autograd differentiates through the unrolled steps; this is not an adjoint
implementation or a claim of constant-memory training.

## Main comparison

| Student | Width | Spline controls | Trainable MLP values | Training depth |
|---|---:|---:|---:|---|
| Direct 192 | 192 | 1 | 331,776 | 1 |
| Direct 768 | 768 | 1 | 1,327,104 | 1 |
| Fixed-depth curve | 192 | 4 | 1,327,104 | 8 |
| Mixed-depth curve | 192 | 4 | 1,327,104 | 1, 2, 4, 8 in a balanced cycle |

The original MLP has width 1,536 and 2,654,208 weights. Both curves impose half
that parameter budget. Direct 192 matches a curve field evaluation's width;
Direct 768 matches the curve's stored parameter count and its four-step
projection MAC count. A curve evaluated once at depth zero accesses only its first control
vector; the other controls remain stored. Equal stored parameter counts do not
mean equal active computation at every resolution.

Each curve checkpoint is evaluated at all four depths without changing its
weights. Training uses 3,000 updates, batch 512 cached teacher examples, and the
same seeded minibatch stream for each student. AdamW has peak LR 0.0003,
40-update warmup, cosine decay to 10%, weight decay 0.01, and gradient clipping
at norm 1. The objective is endpoint MSE divided by mean training teacher-update
energy. No gradients pass through the other transformer blocks or a logit loss.

Initialization selects teacher neurons by contribution energy, copies their
gate/up weights, and refits the down projection with training-only ridge
regression. All four curve controls initially contain the same fitted weights.
There is no random-initialized baseline in this phase. Every method gets the
same number of examples and optimizer updates, but field evaluations, training
FLOPs and elapsed time differ and are recorded.

The short pilot used 32,768 training tokens and 40 updates with a separate
minibatch seed. It established finite gradients, declining validation error,
and affordable run time. The main run then expanded to 131,072 unique training
tokens and 3,000 updates; it recomputes initialization and restarts training.
Pilot weights are excluded from the main results. Pilot inputs and metadata
are retained for provenance.

## Data and selection

Pinned official SmolLM2-135M revision:
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

Pinned WikiText-2 raw converted-Parquet revision:
`3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e`.

- Main training: 1,024 windows / 131,072 input tokens / 417 articles.
- Validation: 32 windows / 4,096 predicted tokens / 18 articles.
- Test: 128 windows / 16,384 predicted tokens / 16 articles.

Each 129-token window supplies 128 inputs and 128 shifted targets and stays
within one article. Validation and test exclude **every article sampled in
phase 2**, not merely the same windows. Official train/validation/test article
titles are disjoint. This is a sampled fixed-128-context comparison, not standard
full WikiText perplexity. The pretrained model may already have seen the text.

Checkpoint selection uses validation endpoint MSE every 100 updates, including
initialization and the final update. Fixed-depth uses its 8-step MSE; mixed-depth
uses the equal mean over all four depths. No test results select a checkpoint.

After training, full-model validation compares both direct students and both
curves at all four depths, plus the fixed phase-2 Q/K baseline. The curve for
composition, cache checks and detailed timing is the fewest-step curve passing
both delta NLL <= 0.05 and KL <= 0.05, then lowest KL. If none passes, the
lowest-KL curve is explicitly a failed-gate representative. That choice is
written before test evaluation. All trained curves/depths and direct controls
are reported on test regardless of quality.

Test confidence intervals resample the 16 selected documents in 2,000 paired
bootstrap replicates. These are pointwise sample intervals, not evidence of
model-family generality or training-seed robustness. This phase has one main
minibatch seed and one model revision.

## Measurements and safeguards

- A full-width one-step replacement must be bit-identical to the original.
- Full-model next-token NLL, KL, argmax agreement and first-block residual error.
- Unique coefficient values and linear MACs, separate from spline assembly.
- Optional materialized weight banks versus on-demand sampling: extra cache
  storage is counted, and predictions must agree exactly.
- Paired CPU latency at batch 1, 128-token prefill, four threads, FP32/eager.
- Incremental KV-cache versus prefill agreement for the selected curve.
- Exact hashes for all learned tensors outside the first block.
- Exported-state round trips and reproduction of measured held-out losses.
- Incremental-cache checks for both exported curve students at eight steps.

All 30 outer blocks still execute. A depth change alters only the repeated
MLP field evaluations inside block 0. At 8 steps, a width-192 curve has the same
learned MLP projection MAC count as the original width-1,536 MLP; spline assembly,
eight RMSNorms and kernel overhead are additional work. Its 4-, 2- and 1-step
forms use fewer projection MACs. Stored coefficients compare at equal FP32
precision, not against the source checkpoint's BF16 byte count.

On-demand sampling constructs one field's matrices as needed. The optional
materialized mode precomputes N sets of matrices; it can improve runtime while
increasing resident memory. Neither mode has a demonstrated hardware speedup
until the measurements establish one. Timing ranges and raw pairs are retained.

The curve plot uses the last token of the first validation window, chosen in
advance. Finer 16/32/64-step trajectories and a frozen-state ablation on that
window are exploratory diagnostics, never used for checkpoint selection.
The 64-step path is not ground truth. Smooth weights do not guarantee a model
trained at eight steps is invariant to solver resolution.

## Proposed follow-up

Start from a function-preserving decomposition before imposing compression:
partition the original 1,536 MLP neurons into eight groups of 192, keep the
original attention-output state as each group's input, and sum the eight
contributions. This equals the original MLP algebraically; a future implementation
must check floating-point agreement. It keeps all original weights and is not
a completed experiment here. Once full-resolution reconstruction is verified,
fit an ordered curve through the contributions and measure reduced sampling.
Compare anchor-conditioned fields and intermediate-path supervision with this
phase's evolving-state, endpoint-only student. The current curves simultaneously
change representation, depth and parameter budget, so their failure cannot
identify one of those factors as the sole cause.

## Reproduce

Use Python 3.12, at least 4 GB free memory, and the pinned CPU environment in
`requirements.txt`. The archive omits the original model and teacher caches.

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt
    python download_model.py
    # Included token arrays are ready; regenerate only if wanted:
    # python prepare_data.py
    mv results recorded_results
    mkdir results
    cp recorded_results/pilot.json results/pilot.json
    python run_phase3.py --stage all
    python verify_exports.py
    python build_report.py

The supplied checkpoints and initializations are overwritten by a fresh run.
Copy the whole directory first if preserving the supplied weights. The `mv`
step preserves supplied measurements and prevents the resume logic from
skipping completed work. The pilot metadata copy preserves the separate, already
completed pilot while rerunning every main student. Use an empty directory for a changed protocol/model or
dataset; caches are specific to this experiment. `--stage train` stops before
language-model evaluation; `--stage evaluate` resumes from the saved students.

Load any trained student into a fresh pinned reference model:

    python load_candidate.py curve_fixed8 --steps 8
    python load_candidate.py curve_mixed --steps 2
    python load_candidate.py direct_768

`--shared-qk` composes with the phase-2 checkpoint; `--materialized` caches sampled
matrices. Only the prespecified combinations in the results were evaluated.

## Research context and sources

- Neural ODEs: https://arxiv.org/abs/1806.07366 — learned vector fields and
  numerical integration as a depth/accuracy tradeoff. This study uses ordinary
  unrolled Euler updates, not that paper's adjoint method.
- Universal Transformers: https://arxiv.org/abs/1807.03819 — recurrence over
  depth. This study does not repeat attention or implement adaptive halting.
- Reference model: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Dataset: https://huggingface.co/datasets/Salesforce/wikitext
- Reference implementation: https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py

The user's source archive motivated sampling learned weight functions at
variable depth, but its quoted GSA comments leave the construction unspecified.
The spline, initialization, integration rule and distillation choices here are
ours. This does not test the comments' cache, online-perturbation or scaling
claims. The original archive and phase-1/phase-2 artifacts are unchanged.
