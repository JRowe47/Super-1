Pythia-70M: matched component controls and checkpoint context
17 October 2026

Open pythia_matched_components_report.html for the measured findings.

This continues the native full-resolution curve analysis. It compares all 64
depth-six packet branches in the existing block-3, head-6 Q/K window at equal
update norm, then independently confirms selected controls and measures fixed
directions at checkpoints 512, 1000 and 8000. No training or permutation.

Reproduce:
python -m pip install -r requirements.txt
python download_checkpoints.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python run_experiment.py
OPENBLAS_NUM_THREADS=1 python analyze.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python verify_scoring.py
OPENBLAS_NUM_THREADS=1 python figures.py
python build_report.py

run_experiment.py and verify_scoring.py accept --checkpoints /absolute/path
to reuse the pinned model files. Downloads total approximately 845 MB.
The experiment is deterministic on this CPU/software setup; inference kernels
on different hardware may have small numerical differences.

Exact design: protocol.json. The protocol hash is retained in environment.json.
64 equal-norm scan variants on 24 prefixes and 4 original prose passages.
36 confirmation variants on 48 fresh prefixes and 8 different prose passages.
Both exclude all random-prefix token IDs from the preceding causality phase.
Selection is locked after the scan and before any confirmation scoring.

Transform: Haar packet, a=(even+odd)/sqrt(2), d=(even-odd)/sqrt(2), recursively.
The full six-level tree is orthonormal. Tree order is not monotonic Fourier
frequency order. A path contains 1024 coefficients, reconstructed over all
65536 native positions. It is not one coefficient or 1024 edited weights.
Native flat interval: [589824,655360), in the original packed QKV tensor;
block and head indices are zero-based. No neuron/order alignment is used.

E = selected projection of theta1000-theta512.
L = selected projection of theta8000-theta512.
r = norm(E) for the prior-selected path aaaaad.
All unit interventions use norm r, including each scan branch separately.
Model patch: theta_s - lambda * v. Positive lambda subtracts the direction;
negative lambda adds it. lambda=0 is the unchanged background checkpoint.
At steps 1000 and 8000, lambda=norm(L)/r is also evaluated for both directions.

Files:
native_local_curves.npz: original float32 window at all three checkpoints;
    all 64 x 1024 float64 packet coefficients for both training intervals;
    explicit full-length selected curves and original native flat indices.
results/probe_inputs.json: every probe token, query order and prose passage.
results/prior_token_ids_excluded.json: the preceding phase's excluded IDs.
results/scan.json and confirmation.json: every sequence, query position,
    passage, intervention norm, rounding error and restoration check.
results/selection_locked.json: scan-derived selection and checksum.
results/curve_metadata.json: path norms, cosine, scaling and reconstruction.
results/*effects.json: paired effects and 95% conditional bootstrap intervals.
results/paired_contrasts.json: paired branch, direction and context contrasts.
results/dose_curvature.json: f(+1)+f(-1)-2f(0) per direction and checkpoint.
results/direct_scoring_verification.json: ordinary full-model/direct-index check.
results/environment.json: runtime, protocol hash and restore results.
responses/*.npz: full normalized hidden vectors at the last query.
results/logit_gram_step*.npy: centered full-vocabulary output-head Gram matrices.
checkpoint_manifest.json: source revisions and model SHA-256 hashes.
results/checkpoint_verification.json: verification of the actual reused files.
SHA256SUMS: hashes of all bundled files other than itself.

Export a full-resolution component, without loading model checkpoints:
python inspect_component.py aaaaad --direction early --matched --output early.npz
python inspect_component.py ddaaaa --direction early --matched --output scan_winner.npz

Interpretation:
Equal norm removes one magnitude confound; it does not equalize Q/K balance,
activation sensitivity or every other structural property of a direction.
Every same-depth branch has the same coefficient count and native support.
Changing the checkpoint with the vector fixed measures background dependence,
including all baseline weights. It does not localize that dependence to the
parameters outside the selected group. Comparing E versus L within one
checkpoint at equal norm measures orientation-dependent response.
Small synthetic probes and eight prose passages cannot establish a natural
language concept, grokking, general-capability preservation or cross-seed
universality. Rankings are descriptive, not statistically distinct ordering.
95% intervals are paired, length-stratified and unadjusted for multiplicity.
