# Curve retention and depth compatibility

This phase continues the native-order, full-resolution curve research with two
bounded experiments. Read curve_retention_depth_report.html for all results,
uncertainty, figures, limitations and the continuous-flow counterexample.

## Findings

Coefficient Adam with a moderate teacher-preservation penalty improves short-copy
accuracy from 44.04% to 53.46%, a paired gain of 9.43 percentage points [95% interval
5.39, 13.59]. Held-out text NLL changes from 4.533716 to 4.536775; the paired
difference is +0.003060 [-0.016719, +0.024109]. This is not a formal equivalence or
non-inferiority claim. Copying remains below the starting model's 63.51%.

The strong penalty improves copying further but sacrifices much more text
learning. Both penalty strengths and all native/curve endpoints are reported.
Regularized models process additional replay examples, so improvements over
unregularized controls are not an equal-compute or equal-exposure comparison.

A 12-site local routing task requires six hops, with fixed identity input and
readout maps. Native training learns the six-hop function; full and two-mode
curve conversions preserve it at six steps. Changing execution depth changes
the operator even for the constant field a=-6,b=6. Its N-step map is
(I + 6(S-I)/N)^N: exactly S^6 at N=6, a mixing operator at other depths.

Within this real scalar one-hop continuous-flow family, no choice of arbitrary
depth-varying a(s),b(s) produces S^6 in the continuous limit. The report gives
the Fourier-mode proof. This is a restricted-family result, not an impossibility
claim about GSA, augmented-state models, arbitrary ODEs, or general attention.

## Protocol and scope

protocol.json was fixed before main training; its hash is stored in
results/environment.json. Full Pythia Haar coefficients are retained; only
19M decoder values are trainable, not the embedding/output/final norm.
The dense network uses differentiably decoded weights. No curve-native execution
speedup, recursive mask, permutation, or full GSA architecture is claimed.

There are 96 new Pythia optimizer steps across four arms (24 each). Two
unregularized endpoints from the previous compatibility phase are hash-verified
and reevaluated on new held-out inputs. Without the reuse option, the source
regenerates those controls too, for 144 steps total. All Adam rates are 2e-5.

Teacher KL uses 24 separate copy sequences and 24 separate 96-token text blocks.
Each replay batch is seen twice. The copying and text KL terms receive equal
weight, with total coefficient lambda 1 or 4. No held-out targets are used to
choose a rate, penalty or checkpoint. Actual new-task data is the same 24
adaptation batches as the preceding phase, 4,560 next-token targets per arm.

The new validation set has 24 blocks. Short copy evaluation has 24 sequences;
long-copy diagnostics have six. All replay/test copy token IDs are disjoint and
exclude preceding probe IDs. Text replay is disjoint by source row from
adaptation; validation uses a separate split and excludes previous evaluation
rows. The eight prose passages are reused retention probes, never replayed.
Absence of WikiText from model pretraining is not established.

Toy training uses 19,000 updates across five paired seeds: three 600-step scratch
arms and five 400-step native-teacher continuation arms per seed. Fresh paired
Gaussian minibatches contain 64 rings. Intermediate hop targets supervise the
scratch phase only. Multi-depth continuation has 5,592 executed stages versus
2,400 in fixed-depth continuation. This is not a compute-matched comparison.

## Reproduce

Use Python 3.12 and requirements.txt. Exact tested versions are recorded in
results/saved_model_audit.json. The precise source PyTorch build may be
unavailable on some package indexes; other versions can change numerical paths.
Use a copy of the directory: reruns overwrite results and saved models.

~~~bash
python verify_bundle.py
python download_sources.py
python run_retention.py
python transport_toy.py
python analyze_phase.py
python verify_saved.py
python make_figures.py
python build_report.py
~~~

download_sources.py retrieves the pinned 281.7 MB Pythia checkpoint and verifies
its SHA-256. Configuration, tokenizer, exact token blocks and probes are bundled.
Optional source-data regeneration:

~~~bash
python download_sources.py --dataset
python prepare_phase.py
~~~

run_retention.py and verify_saved.py accept --checkpoints PATH. To reuse the
old unregularized endpoints, run_retention.py accepts --reuse-controls PATH,
where PATH is the extracted curve_training_compatibility_2026-10-17 directory.
The old decoder hashes and exact adaptation token blocks are checked.

The successful Pythia run took 202.28 seconds on three CPU threads, including
teacher preparation and evaluation. An initial implementation error happened
before any regularized optimizer update and was fixed. Toy training used one
thread. Exact toy timers were not persisted before a post-training audit error;
the 29.17-second saved-checkpoint span is only a lower bound.

## Numerical audit and recovery

The teacher cache first used inference-mode indexing tensors; PyTorch refused
their reuse in autograd. Cache creation was changed to no_grad before the first
regularized update. The objective and data were unchanged. Initial teacher KL
is zero; initial replay-gradient numerical residuals are below 4.1e-13.

All 40 toy models finished training and were saved. A post-training oracle audit
reconstructed JSON values as float32 and failed a float64 comparison. The source
now explicitly uses float64. recover_transport_checks.py reconstructs the
post-training diagnostics from saved models, with no new optimizer steps.
It independently matched 39 available final expected-MSE records exactly and
reconstructed one missing final record (seed4, curve2_multidepth, step400), marked
recovered_from_saved_model in the trajectory file. Audit metadata and all raw
records are retained. The recovery script is historical; normal reproduction
uses the fixed transport_toy.py and does not require recovery.

Both prespecified lambda4 Pythia decoder files were reloaded into fresh models
and reproduced every checked validation-block loss, copying metric and prose
metric exactly. Files contain decoded float32 weights, not float64 masters or
optimizer moments, and are not exact optimizer-resume snapshots. The lambda4
pair was chosen for saving before outcomes. Lambda1 evidence includes the full
per-item evaluations and source; its final decoder tensors are not bundled.

## Contents

- results/evaluations.json: per-block/per-sequence raw Pythia measurements.
- results/effects.json and contrasts.json: paired effects and intervals.
- results/haar_update_bands.json: original tensor/layer/detail-scale update data.
- results/haar_band_summary.json: energies normalized by coefficient count.
- results/transport_trajectories.json: all toy evaluations and exact operators.
- results/transport_continuous_limit.json: mixing limit, Fourier gains and proof.
- results/implementation_notes.json: fixes and recovery disclosure.
- models/: two lambda4 Pythia decoders and 40 toy final/source checkpoints.
- data/: exact adaptation/replay/evaluation token inputs and row provenance.
- provenance/: preceding protocol, prior probe IDs, model hashes, and relevant
  context from the supplied original archive. Bank coordinate and depth differ.
- figures/: PNG and SVG versions; report embeds its PNGs without external files.

Paired 95% intervals resample evaluation items, not independent training seeds.
The longer copy probes are descriptive. Full-resolution transforms do not imply
that a band has a unique semantic function. Coarse-band enrichment is a
descriptive optimizer footprint; no causal masking intervention was run here.
All proposed follow-up experiments in the report remain proposals.
