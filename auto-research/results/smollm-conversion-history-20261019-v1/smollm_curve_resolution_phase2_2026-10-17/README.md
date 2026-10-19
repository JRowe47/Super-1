# SmolLM2 first-block curve research — phase 2

This experiment continues the exact first-block conversion. It compares
structured factor curves, coarse-to-fine Haar coefficients, reduced spline
knots, and groups of original MLP neurons. **Only transformer block 0 changes.**
The other 29 blocks, embeddings, final normalization and output head retain
their original values. There are no gradient updates or distillation steps.

See `results/phase2_report.html` for the measured findings and
`results/summary.json` for completion status. Detailed per-window measurements,
selection rules, spectra, and CPU timing samples are retained in `results/`.

## What a curve means here

The phase-1 curve indexed every scalar first-block parameter. In this phase,
unchanged matrices and normalization scales remain exact cardinal curves.
Decomposed matrices use curves for their two factors, evaluated at exact knots:

    W approximately equals L R
    y = L (R x)

The order of retained SVD components defines the resolution of that factor
family. Curve interpolation itself does not create the low-rank structure.
This remains a coordinate representation; no special benefit is claimed merely
from wrapping coefficients in a curve. Full-factor decompositions have a
floating-point roundoff floor, unlike phase 1's bitwise-exact knot lookup.

For a matrix with shape m by n, the retained factor cost is r(m+n), including
both factors. A nominal 25% coefficient budget means r(m+n) <= 0.25mn, not
that 25% of the singular modes remain. Full-rank controls are separate and can
store more values than the original dense matrix.

## Experimental arms

1. Ordinary matrix SVD, applied to query, key, or both independently.
2. Activation-aware SVD using the uncentered input second moment after the
   first RMSNorm. A ridge of 0.0001 times its mean diagonal stabilizes the
   Cholesky factor. Decompose W L and map the input factor back with L inverse.
   This adapts the approximation to observed input directions. It is inspired
   by activation-aware SVD methods; it is not a reproduction of the complete
   SVD-LLM pipeline or its published benchmarks.
3. Joint Q/K ordinary and activation-aware SVD, stacking their output rows and
   sharing the input factor. The implementation evaluates the shared input
   projection once, then produces Q and K in their ORIGINAL output coordinates.
   RoPE, grouped-query layout, masks and attention calculations stay unchanged.
4. Haar wavelet coefficients, ordered from coarse to fine. A prefix is kept;
   omitted detail coefficients become zero. These are storage/functional
   controls: a dense matrix is reconstructed once, and inference still performs
   the full dense matrix multiplication. The dense working cache is reported.
5. Thinned spline knots at deterministic evenly spaced integer indices, with
   cubic Hermite reconstruction. These also materialize dense weights and do
   not claim fewer inference multiply-accumulates.
6. MLP contribution groups: 1,536 gated neurons are arranged into 16 groups of
   96. Gate/up rows and down columns remain coupled. We evaluate 12, 8, 4 or 2
   groups by importance, original contiguous order, or three seeded random
   orderings. The actual retained submatrices are compact and use fewer MACs.
   Importance is E[activation_j squared] times squared norm of down column j;
   it measures individual contribution energy, not interactions or causality.
7. Full-resolution controls for exact curves, SVD, activation-aware SVD,
   joint SVD, Haar, spline knots and all 16 MLP groups. An MLP-skip arm measures
   the zero-group endpoint while retaining attention and residual connections.

There are 84 validation configurations. No MLP neurons are rescaled or refitted
after selection. All-group evaluation restores the original computation.

## Dataset and selection

Use `Salesforce/wikitext`, `wikitext-2-raw-v1`, pinned by the converted-Parquet
revision in `data/dataset_manifest.json`. Official train, validation, and test
articles are disjoint, and this preparation verifies title disjointness.

- Calibration: 128 windows, 16,384 input tokens, from the official train split.
- Candidate selection: 32 windows, 4,096 predicted tokens, official validation.
- Held-out final evaluation: 128 windows, 16,384 predicted tokens, official test.

Each sampled window contains 129 tokens within one article. The first 128 are
model inputs, and their 128 shifted next tokens are targets. Context is reset
between windows. Selected windows are nonoverlapping within each document.
This is a sampled, fixed-128-token comparison, NOT the standard complete
WikiText-2 perplexity evaluation. Held out means held out of THIS experiment;
the original model may have encountered similar or identical material in
pretraining. The manifest records the exact windows, seeds and source hashes.

Selection is fixed before evaluating test results: for ordinary Q/K,
activation-aware Q/K, jointly activation-aware Q/K and importance-ranked MLP,
choose the least linear-projection compute passing validation delta NLL <=
0.05 nats/token and KL <= 0.05. If none passes, report the lowest-KL candidate
as a failed-gate representative. Matched random and contiguous MLPs plus
half-budget Haar and spline controls are included regardless of their quality.
This is an engineering selection gate, not a statistical significance test.

Reported test uncertainty uses 2,000 paired bootstrap resamples of documents,
retaining all selected windows of each sampled document. It is a small,
single-model study; random MLP ordering has three seeds, but the data-window
sampling and model checkpoint each have one seed/revision.

## Measurements

- Next-token NLL and paired change from the original model.
- Perplexity ratio, exp(delta NLL); not an independently computed benchmark.
- KL of the original next-token distribution to the candidate distribution.
- Agreement with the original argmax, which is not ground-truth accuracy.
- First-block error relative to the original RESIDUAL UPDATE, not only the
  much larger residual-stream state.
- All stored coefficient values, materialized dense caches, and linear MACs.
- First-block and full-model prefill latency: 12 alternating-order CPU pairs,
  warmup first, four threads, batch 1, 128 tokens, FP32 and eager attention.
- Joint-Q/K prefill versus incremental-cache compatibility.
- Hashes of every parameter tensor outside block 0 before and after the run.

Coefficient budgets apply to the targeted projections, not the full block or
whole model. All models still have 30 blocks and their original attention-cache
dimensions. Grouped MLP computation reduces evaluated contributions, not the
number of sequential transformer blocks. Whole-model speed gains can be much
smaller than the first-block projection MAC reductions. Timing on a shared CPU
is noisy; retain the paired samples and do not infer a GPU speedup.

## Reproduce

Use Python 3.12 and at least 4 GB free RAM. The tested CPU environment is
recorded in `environment.json`; exact package versions are in requirements.

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt
    python download_model.py
    python prepare_data.py
    mv results recorded_results
    python run_phase2.py
    python verify_saved_candidates.py
    python build_report.py

The reference model downloader pins SmolLM2 to revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The full reference checkpoint and raw
Parquet files are excluded from the archive. Token windows, selection manifests,
and selected converted first-block checkpoints are included.

The `mv` command preserves the supplied measurements and makes the run recompute
every candidate. Without that step, the runner resumes existing result JSONs.
`prepare_data.py` is optional when using the included token arrays; it downloads
the original Parquet splits and regenerates the identical sample when called.
Use a fresh directory if a previous reproduction already has `recorded_results`.

`run_phase2.py --stage sweep` stops after selection; `--stage finish` resumes
completed validation work and performs the held-out phase. Do not reuse the
cache/results directories after changing the model, data, or protocol; start
a new experiment directory for such changes. The frozen first-block checkpoint
loader is `load_candidate.py`.

For example, restore a saved candidate without rerunning the sweep:

    python load_candidate.py joint_activation_svd_0.125

`verify_saved_candidates.py` checks every saved candidate's complete block-state
round trip and reproduces its first held-out window loss. The model must be a
fresh instance of the pinned CPU FP32/eager reference before installing a block.

Coefficient-count reductions compare at equal precision. Saved curve checkpoints
are FP32; they are not claimed to be smaller in bytes than the original BF16
checkpoint. The integer neuron-index array and format metadata add small costs.

## Sources

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- https://huggingface.co/datasets/Salesforce/wikitext
- https://arxiv.org/abs/2403.07378
- https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py

The supplied October 15 project archive and the phase-1 experiment are
unchanged. This phase does not train a continuous-depth model or change
TinyLlama. Reduced-depth architectural distillation remains a separate test.
