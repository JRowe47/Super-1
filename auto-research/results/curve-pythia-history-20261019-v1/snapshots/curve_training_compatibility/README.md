# Curve-space training compatibility

This phase tests actual learning through full native-order Haar coefficients
in Pythia-70m, then separates shared depth fields, restricted conversion and
variable execution depth in small finite-task models. Read
curve_training_compatibility_report.html for results, limits and figures.

## Main findings

- Full-resolution curve encoding supports continuation. Native and transported
  Adam first updates agree to about 3e-14 per coordinate, with small numerical
  drift by step 24. Their aggregate held-out losses are almost identical.
- Adam directly on coefficients changes the update direction (first-step
  cosine 0.639 against native Adam, despite almost equal norm). Native Adam
  finishes at validation NLL 4.373488; coefficient Adam at 4.409748, from 4.740709.
- Both reduce copying accuracy substantially: 64.06% initially, then 41.76%
  native and 42.96% coefficient Adam. Conversion alone does not explain this.
- Two-mode depth curves fit the finite toys, but all final arms saturate at
  100% six-step accuracy. Hard projection changes logits without changing any
  labels. A no-stage diagnostic exposes the weak depth demands of these tasks.
  This does not establish full GSA capacity or faithful off-grid execution.

## Scope

Pythia uses all 72 tensors in six decoder blocks (18,914,304 trainable values).
Embeddings, final normalization and output projection are fixed. Tensor order
and layer identities are preserved. No coefficients are truncated. Master
values are float64; decoded forward computation is float32, with dense matrix
operators. This is not a curve-native execution kernel or the full GSA model.

Each Pythia arm sees the same 24 batches of two 96-token sequences, 4,560
next-token targets total. Fresh optimizer states are used. The original five
arms include an unstable SGD setting; the disclosed addendum adds two small-step
SGD controls. All original results are retained. No hyperparameter sweep.

The toys have five paired seeds, all 16 four-bit inputs and six target functions.
They use 1,200 scratch updates or 400 updates from a native step-800 checkpoint.
All input/output maps train freely. Multi-depth training is not compute matched.
The shared depth field is a controlled residual-network proxy, not GSA attention.

## Reproduce

Use Python 3.12 and the dependencies in requirements.txt. The tested environment
is recorded in results/saved_model_audit.json. The precise PyTorch build may
not be available on all package indexes; other builds may change numerical
trajectories. Use a copy of this directory because reruns overwrite results.

~~~bash
python verify_bundle.py
python download_sources.py
python pythia_train.py
python stable_sgd.py
python toy_depth.py
python analyze.py
python verify_saved.py
python make_figures.py
python build_report.py
~~~

download_sources.py downloads only the pinned 281.7 MB base checkpoint unless
--dataset is supplied. Exact token IDs, tokenizer, configuration and probe inputs
are already bundled. To regenerate source selection, run:

~~~bash
python download_sources.py --dataset
python prepare_data.py
~~~

The base checkpoint is step143000, commit
de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852. Its SHA-256 and source details are in
checkpoint_manifest.json. Raw data provenance is in data/source.json and
DATA_ATTRIBUTION.txt. WikiText is not verified absent from pretraining.

The Pythia commands also accept --checkpoints PATH, where PATH contains
step143000/model.safetensors, step143000/config.json and tokenizer/.
The stable control needs the baseline result files from the original run.
Budget about 5.5 minutes of measured component runtimes on the source CPU;
hardware and dependency changes can alter this. Original and stable Pythia
runs use three CPU threads, toys one. Temporary reference snapshots require
several GB of extra scratch storage when rerunning. They are excluded here.

## Files

- protocol.json: original frozen protocol, hash recorded in run metadata.
- protocol_addendum.json: instability trigger, fixed training-gradient rate
  rule, validation-reuse disclosure and added budget for the SGD pair.
- transforms.py: complete Haar transform, inverse and explicit adjoint.
- pythia_train.py / stable_sgd.py: real coefficient-autograd continuation.
- scoring.py: retention probes, also used in the preceding verified phase.
- toy_depth.py: exact/full and restricted shared depth-field controls.
- analyze.py: paired bootstrap intervals, seed summaries, post-hoc stage removal.
- verify_saved.py: independent final-model reload and inference checks.
- results/: per-item metrics, original outputs, summaries, numerics and runtimes.
- responses/: retained final-query hidden states at each evaluated step.
- models/: two adapted Pythia decoders and all toy final models/native teachers.
- figures/: reproducible PNG and SVG figures.

Pythia saved NPZ files contain named decoded float32 decoder tensors. Restore
them into the pinned base checkpoint using the pattern in verify_saved.py.
They reproduce the recorded final evaluations. They do not contain the original
float64 coefficient masters or optimizer moments, so are inference/analysis
checkpoints rather than exact optimizer-resume snapshots. Re-encoding these
saved tensors is verified to preserve their float32 weights and checked logits.

Toy NPZ files include trainable arrays, curve kind/mode count, bridge buffers
and alpha. The bridge's frozen residual arrays remain in the saved file but
alpha is zero at the final checkpoint. They are not active final parameters.

## Interpretation and uncertainty

Pythia intervals resample the same 24 validation blocks or 24 copying sequences
(stratified by prefix length), with 4,000 paired resamples. Prose intervals
resample eight passages with token weighting. These intervals do not measure
variability across independent Pythia training runs. The prose set was reused
from the previous phase. Fresh copying inputs excluded prior probe token IDs.

All toy seeds and outputs are retained. Exhaustive truth-table performance
does not test out-of-distribution semantics. The post-hoc zero-residual control
was prompted by ceiling accuracy and did not select training hyperparameters.

The proposed next experiments (retention-aware adaptation and more demanding
depth-dependent sequence tasks) are not results of this bundle.
