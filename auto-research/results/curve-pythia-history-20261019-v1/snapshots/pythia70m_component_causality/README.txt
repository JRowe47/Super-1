Pythia-70M native component causality — 17 October 2026

Open pythia_component_causality_report.html.

Research scope: native parameter-curve changes, full-resolution Fourier and
wavelet hierarchies, behavioral rollback/transplant tests. No training, neuron
permutation, alignment search or compression contest.

Reproduce:
python -m pip install -r requirements.txt
python download_checkpoints.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python causal_probe.py discover
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python causal_probe.py confirm
OPENBLAS_NUM_THREADS=1 python summarize.py
OPENBLAS_NUM_THREADS=1 python figures.py
python build_report.py

Use --checkpoints /absolute/path to reuse existing pinned checkpoints. The
three model downloads total about 845 MB; no new model fitting is needed.

Files:
protocol.json: design, discovery/confirmation separation, frozen selection rule.
checkpoint_manifest.json: source revisions and SHA-256 model checksums.
native_qkv_curves.npz: all six QKV tensors at steps 512 and 1,000, original
    float32 values. Keys step512_layer0 ... step1000_layer5.
selected_local_component_curves.npz: complete native 65,536-position window,
    before/after, full update, depth-five parent and both depth-six children.
results/probe_inputs.json: every token input and original prose passage.
results/candidate_catalog.json: every native component definition.
results/selection_locked.json: discovery-only selection, locked before confirmation.
results/discovery.json, confirmation.json, later_context.json: every recorded
    per-sequence / per-position and prose measurement.
results/confirmation_effects.json: stratified paired bootstrap intervals.
results/behavioral_interactions.json: every available parent/child interaction.
responses/*.npz: complete final-normalized hidden states at the last query
    for each sequence and each condition. Columns retain original hidden order.
results/logit_gram_step*.npy: centered full-vocabulary output-head Gram matrices.
results/*environment.json: runtime, numerical restore and selection checks.
SHA256SUMS: hashes of every bundled file except itself.

Extract any component from the native source curves without loading the LLM:
python inspect_component.py H6_aaaaad --output selected_component.npz

Index conventions: block 3 and head 6 are zero-based. The QKV tensor is packed
by head, then Q/K/V, 64 rows each; row-major flat order is unchanged. The
head-6 Q/K window is [589824,655360), with Q [589824,622592) and K
[622592,655360). The earlier 64-weight Haar example belongs to another head;
its visible curve trajectory did not establish a copying role.

A packet depth-six group contains 1024 coefficients but changes a curve over
65536 native weights. Do not report it as one coefficient or 1024 edited weights.
The transform calculation is float64; installed inference weights are float32.

Limits: one model training seed, synthetic content-following probes, eight
collateral prose passages, unadjusted descriptive confidence intervals. No
natural-language feature identity, sparse training advantage, universal basis
preference, grokking claim, or guarantee that unrelated capabilities survive.

Next bounded question: compare selected recursive masks against sibling masks
at equal update norm and test interpolation dose responses on separate prompts.
Use identical update directions across checkpoints to distinguish changes in
the component itself from changes in the surrounding trained computation.
