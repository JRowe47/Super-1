# Native curve and Pythia research history

## Publication summary

This is a retrospective publication of ten saved research stages dated 17 October 2026. It preserves the progression from broad Fourier diagnostics to full-resolution native component analysis, behavioral interventions, actual curve-coordinate continuation, retention controls, and explicit depth-operator tests. It introduces no new training or model evaluation.

The central distinction is between representation, optimization, and execution. An invertible full coefficient representation can preserve an ordinary network at its original coordinates. That does not imply compression, an optimizer-independent training trajectory, semantic meaning for a frequency address, or function preservation when the execution depth/operator changes.

The main findings, with their boundaries, are:

- Broad SmolLM/Pythia weight spectra do not exhibit a strong generic low-frequency collapse. This does not rule out useful localized, phase-sensitive components.
- Native component interventions identify a strong contribution of Pythia block 3, head 6's Q/K update to synthetic successor copying. The direction of an update and the surrounding checkpoint matter alongside its address and magnitude.
- Pythia can continue learning through a complete native-order Haar parameterization. Adam on coefficients takes a different direction from native Adam; transporting the native optimizer provides the appropriate equivalence control.
- Teacher replay improves a measured retention/learning tradeoff. A later static gradient-selected update mask helps initially but underperforms the matched random-mask mean after 24 updates.
- Exact conversion at one depth does not guarantee faithful execution at another. A restricted scalar local-flow counterexample is followed by a constructive two-channel, noncommuting local remedy. The remedy costs extra state and operations and is specific to the routing task.

These are scoped experiments, not a complete implementation or validation of geometric self-attention, a general architecture-conversion algorithm, broad language-capability preservation, or a demonstrated compute/inference speedup.

## What is included

[IMPORT.json](IMPORT.json) records ten source archive identities, 430 preserved source/report/figure/measurement files, and 429 excluded archive members. Large JSON/log files are stored with lossless gzip compression. The import manifest distinguishes original content hashes from stored-byte hashes. Historical checksum manifests describe the original archives, not the filtered publication tree.

Checkpoint/model arrays, coefficient archives, cached hidden states, token-array bundles, and other NumPy binary dependencies are excluded. Five unnecessary local-path or research-decision context files were also excluded. All source archives remain unchanged. Historical reports and READMEs sometimes describe their original full bundles; those descriptions must be read together with the exclusions in IMPORT.json. This filtered publication is an evidence/source snapshot, not a self-contained model-reproduction package.

## Chronological stage index

All stages are dated 17 October 2026. The order below follows their documented research dependencies, rather than asserting finer timestamp precision.

### 1. SmolLM depth Fourier probe

[Report](snapshots/smollm_fourier_probe/results/smollm_fourier_probe.html) · [Source](snapshots/smollm_fourier_probe/run_probe.py)

The seven projection families retain 28.72% of squared weight norm in the lowest 8 of 28 cosine modes, close to the 28.57% equal-energy reference. All 28 modes are required to exceed 99% in frequency order. Normalization vectors are smoother but account for only 0.0325% of block parameters.

This is a post-hoc weight-only analysis of saved SmolLM phase-6 endpoints. It performs no new training or language-quality evaluation. The rank-six update stack is explained by the six adapted block positions, not evidence of a discovered six-curve compression mechanism. The proposed toy experiment was not run in this stage.

Historical checks: float64 FFT/DCT roundtrip maximum error 1.78e-15, source hashes, Gram-based permutation controls. Reproducing the original weight transforms requires the separately recovered phase-6 weights. The bundle has no historical SHA256SUMS manifest.

### 2. SmolLM recursive spectral update sufficiency

[Report](snapshots/smollm_recursive_spectral/results/recursive_spectral_report.html) · [Protocol](snapshots/smollm_recursive_spectral/protocol.json)

A gradient-guided mask retains 25.26% of the update's real spectral degrees of freedom and recovers 54.8% of the full update's audit NLL improvement. At 28 blocks, audit perplexity relative to the original 30-block model is 1.2236, versus 1.2552 for random spectral bands and 1.1090 for the complete update. A phase-shifted control loses the benefit. Function-preserving MLP permutations strongly change fine-bin spectra, demonstrating coordinate dependence.

This is retrospective sufficiency using an already trained dense update. It does not show sparse training or reduced training compute. The exploratory guided individual-weight comparison does not resolve a reliable Fourier advantage. The audit is reused; one source training seed and six adapted blocks limit generality. Global flattened FFTs and localized intervention FFTs use different axes.

Historical checks: unchanged original tensor hashes, reproduced phase-6 metrics, transform/Parseval checks, preserved predictions under coherent MLP permutation. Their model-based portions are not rerun here.

### 3. Pythia-70M spectral checkpoint study

[Report](snapshots/pythia70m_spectral/pythia70m_spectral_report.html) · [Protocol](snapshots/pythia70m_spectral/protocol.json)

Thirteen published checkpoints show a synthetic copying transition without a corresponding broad Fourier collapse. On the initial probe, accuracy rises from 0% at step 512 to 38.5% at step 1,000 and 76.4% at step 8,000. A fresh-sequence ablation at step 1,000 reduces accuracy from 39.1% to 0.8% when layer 3/head 6's output contribution is removed.

An ordering fitted on an earlier update modestly improves a later retrospective MLP update-recovery comparison at 25% budget, but individual weights win at 10%. There is no general Fourier advantage, training-efficiency result, or grokking claim. Frozen ordering is an exploratory earlier stage; later studies return to fixed native coordinates.

Historical checks: hashes for all 13 checkpoints, inverse/Parseval/tree reconstruction, function-preserving neuron-order controls, and full-update restoration. The original validate.py requires external checkpoints; it is not a publication-only test.

### 4. Pythia native full-resolution component atlas

[Report](snapshots/pythia70m_native_component_analysis/pythia70m_native_component_report.html) · [Functional-baseline receipt](snapshots/pythia70m_native_component_analysis/results/functional_baseline.json)

The analysis preserves all 72 decoder tensors, six layers, 18,914,304 native parameters per checkpoint, and every Fourier/Haar coordinate across 13 checkpoints. It follows individual signed coefficients, phase changes, local supports, and recursive parent/child structure. Large movement is a targeting candidate, not a behavioral-importance label.

Full-precision reconstruction gives identical checked float32 logits at steps 1,000 and 143,000. Rounded float32/complex64 analysis coefficients change one of 520 checked final-checkpoint argmaxes. Full resolution and numerical precision therefore must remain separate claims.

The archive's internal root is pythia70m_in_situ_2026-10-17. Its six large layer-coefficient NPZs and exact step-1,000 decoder-curve NPZ were separately supplied artifacts and are not inside the recovered ZIP. Their hashes/metadata are preserved; their existence or function is not freshly verified by this publication. Original component extraction requires those arrays or regeneration from pinned checkpoints.

### 5. Toy component calibration

[Report](snapshots/toy_component_calibration/toy_component_calibration_report.html) · [Protocol](snapshots/toy_component_calibration/protocol.json)

Twenty small models cover readout, parity, context routing, and joint tasks over every one of 16 inputs, with five seeds per setting. All reach perfect truth-table accuracy. Exhaustive declared component rollback/transplant cases link native updates to behavior and show that additive parameter components generally have non-additive hidden-network effects.

A selected two-weight Haar rollback reduces routing accuracy to 87.5% while preserving readout/parity choices. This is a post-hoc illustration within an enumerable task, not held-out generalization, universal semantic frequency identity, or proof that a component carries the behavior in isolation.

Historical checks: full reconstruction/choice agreement, parent-child additivity, direct numerical replay. Model-array exclusion prevents repeating the full intervention/reconstruction audit from this publication. Some detailed component evidence exists only in the original NPZs; the historical report's embedded explorer and JSON summaries remain available.

### 6. Pythia component causality

[Report](snapshots/pythia70m_component_causality/pythia_component_causality_report.html) · [Locked selection](snapshots/pythia70m_component_causality/results/selection_locked.json)

Discovery and confirmation are separated. Rolling block 3/head 6's native Q/K interval back from step 1,000 to step 512 reduces fresh repeated-successor accuracy from 34.07% to 2.69%. The prior visibly changing Fourier/Haar atoms have much smaller effects. A discovery-selected depth-six packet group retains a measurable effect; its 1,024 coefficients reconstruct an update over 65,536 weights, not merely 1,024 edited weights.

Transplanting the interval into the earlier network does not install top-choice copying. Effects are conditional on the trained surroundings; a small prose-average loss change does not guarantee preservation of unrelated capabilities. Selection is behavior-guided and conditional intervals are not multiplicity-adjusted.

Historical checks: endpoint restoration, parameter additivity, locked selection checksum, direct full-forward scoring, and centered-logit Gram comparison. Fresh publication checks reaggregate saved position/sequence/prose scores only; excluded hidden-state/Gram arrays prevent independently rebuilding interaction geometry.

### 7. Pythia matched component directions and context

[Report](snapshots/pythia70m_matched_components/pythia_matched_components_report.html) · [Direct-scoring receipt](snapshots/pythia70m_matched_components/results/direct_scoring_verification.json)

All 64 depth-six packet branches are compared at equal update norm, followed by independent confirmation. The prior-selected branch remains competitive, but its difference from the scan winner is not statistically resolved. At step 8,000, equal-norm early and late directions at the same address have opposite effects on copying NLL. Address and norm do not specify an intervention's behavior.

The background comparison includes every baseline parameter, including the component's existing values; it does not localize dependence solely outside the selected group. Results remain one pretraining seed and small synthetic/prose probes.

Historical checks: all 100 interventions restore original parameters, unchanged outside-window positions, zero restored loss differences, direct scoring max NLL discrepancy 1.43e-6 with all checked top-1 matches, and pinned checkpoint hashes. Saved-score arithmetic is freshly checked, not the model forward passes.

### 8. Curve training compatibility

[Report](snapshots/curve_training_compatibility/curve_training_compatibility_report.html) · [Saved-model receipt](snapshots/curve_training_compatibility/results/saved_model_audit.json)

Pythia actually learns through complete native-order Haar coefficients. After 24 paired batches, native and transported Adam reach almost identical held-out loss; direct coefficient Adam changes the first-update direction (cosine 0.639 versus native Adam) and follows a different trajectory. Native and coefficient Adam both degrade copying while learning the adaptation text.

This stage keeps every decoder degree of freedom and dense forward computation. It does not implement a compressed curve model or curve-native execution kernel. The unstable original SGD runs and disclosed stable-step addendum are both preserved. Five-seed shared-depth truth-table toys saturate; a no-stage control shows that they do not force substantial depth computation.

Historical saved-native/coefficient decoder reload checks reproduce all checked metrics exactly. Equivalent optimization implementations exhibit later numerical drift rather than bitwise-identical full trajectories. Publication checks reaggregate original and stable-SGD saved scores without rerunning either model.

### 9. Curve retention and execution-depth boundary

[Report](snapshots/curve_retention_depth/curve_retention_depth_report.html) · [Saved-model receipt](snapshots/curve_retention_depth/results/saved_model_audit.json)

A moderate teacher penalty raises coefficient-Adam short-copy accuracy from 44.04% to 53.46%, a 9.43-point gain, while observed text NLL changes by +0.0031. Its interval permits changes in either direction; this is not a non-inferiority result. Replay uses additional training examples/compute and all adapted arms remain below baseline copying retention.

A six-hop local ring task establishes a distinct depth issue. The constant scalar field a=-6, b=6 implements exact six-hop routing at six Euler steps but approaches a different mixing operator with finer execution. The impossibility proof covers only real scalar one-hop commuting continuous flows, not general ODEs, augmented-state models, or GSA.

Historical audits reload the two prespecified lambda-4 Pythia decoders. The headline lambda-1 endpoint has full score evidence but no saved decoder in the original bundle. A toy audit dtype failure was repaired from existing saved models; 39 final records matched and one missing final record was explicitly recovered without retraining. Exact toy timing was not preserved. Publication checks verify score means and point contrasts.

### 10. Static masks and augmented local transport

[Report](snapshots/curve_mask_transport/curve_mask_transport_report.html) · [Saved-model receipt](snapshots/curve_mask_transport/results/saved_model_audit.json)

The static targeted mask beats the random-mask mean by 3.78 copying points after one update, but trails by 9.69 points after 24; final text NLL is also higher by 0.0188. Counts match within each native tensor/Haar band, and each update's global norm is matched. This rejects the tested fixed mask as an endpoint retention improvement, without showing that recursive targeting is intrinsically ineffective or that refreshing masks will solve the problem.

The companion two-real-channel construction realizes six shifts using twelve alternating local rotations and smooth, noncommuting pulses. A boundary-aware exact solver preserves the operator under tested subdivisions; five seeds for each of two parameterizations learn it. State doubles and at least twelve local operations remain necessary. A nominal one-cell grid does not perform six-hop communication in one local operation. This is a constructive task-specific remedy, not arbitrary Transformer conversion.

Historical audits reproduce the full/targeted saved-decoder metrics exactly. The initial odd-tile correction occurred before training or evaluation. Publication checks verify point contrasts, mask-count equality, recorded step norms/frozen-master checks, and summaries of all 80 saved toy/grid records.

## Validation: current checks versus historical evidence

### Performed for this publication

- Read all ten saved reports, relevant protocols/READMEs, verification receipts, and verifier source.
- Checked archive-byte identities against IMPORT.json and all 839 archive members covered by the nine historical checksum manifests; no missing or mismatched covered member was found. This validates archive integrity, not scientific correctness. The first Fourier archive has no such manifest. [Archive receipt](SOURCE_ARCHIVE_VERIFICATION.json); rerun with `python verify_source_archives.py ARCHIVE_DIRECTORY` against the ten original ZIPs.
- Reviewed retained text for credential-shaped strings, signed URLs, private-session links, personal-contact records, unrelated PDS/regional content, and local filesystem traces. No credential or unrelated-user-data finding remained after image-encoding/matrix-operator false positives were resolved. Five unnecessary path/context files were excluded with provenance.
- Ran [verify_saved_metrics.py](verify_saved_metrics.py), a standalone standard-library checker: 46,807 arithmetic/consistency checks across 21 preserved result files passed. [Machine-readable result](SAVED_METRICS_VERIFICATION.json).

The arithmetic checker recomputes 20,052 sequence NLL means from saved position scores and 20,052 sequence accuracies, sequence and block means, token-weighted prose means, retention/mask point contrasts, mask-count and step-norm consistency, and saved toy/grid aggregates. It reads gzip directly. It imports no experiment code, loads no checkpoints, makes no network calls, runs no model, and writes no files.

### Preserved historical evidence; not rerun

Training/evaluation outcomes, model reload and logits checks, source-checkpoint hash receipts, transform checks that need omitted arrays, semantic/causal interpretations, bootstrap confidence intervals, and interaction geometry remain historical evidence from the original runs. The publication verifier does not independently validate historical forward-pass correctness, recompute confidence intervals, or cover every claim in all ten stages.

The original verify_saved.py, verify_scoring.py, baseline_check.py, and training/probe commands execute models and require their recorded dependencies. The historical verify_bundle.py scripts/checksum lists target the complete original archives and should not be advertised as passing on this filtered tree. Rebuilding figures/reports may require decompressing saved JSON and, for some stages, restoring excluded arrays. Perform such regeneration only in a separate copy.

## Interpretation limits retained across the history

Most Pythia findings condition on one published pretraining seed and short synthetic copying/limited prose evaluations. Paired bootstrap intervals primarily describe evaluation-item variability, not training-seed uncertainty or general capability preservation. Later phases deliberately use new probe/validation inputs, so absolute scores across reports are not directly comparable. WikiText is not verified absent from Pythia pretraining. Complete coordinates preserve available parameter information; they do not by themselves establish useful compression, a universally preferred basis, semantic feature identity, or interchangeable execution depths.

## Archived reproduction entry points

The following are the original README entry points, preserved as documentation. They were not executed for this publication. Use each stage's original complete bundle, recorded dependency versions, and required external checkpoints/arrays in a separate working copy. Many commands train models or run model inference; this list is not a model-free verification recipe. Snapshot JSON/logs may first need restoration from gzip to their original filenames. The two publication-specific verifiers above are the model-free checks.

1. SmolLM Fourier: recover phase-6 inputs and the pinned model with that bundle's download_model.py, then `python run_probe.py --phase6 PATH` and `python build_report.py`.
2. SmolLM recursive: `python global_probe.py --phase6 PATH`, `python interventions.py --phase6 PATH`, `python targeting_control.py --phase6 PATH`, `python layer_permutation_probe.py --phase6 PATH`, then `python build_report.py`.
3. Pythia spectral: `python download_checkpoints.py`, then `python analyze_spectra.py`, `python behavior.py`, `python transition_probe.py`, `python low_frequency_controls.py`, `python validate.py`, and `python build_report.py`.
4. Native atlas: `python download_checkpoints.py`, `python insitu.py --checkpoints ./checkpoints`, `python baseline_check.py --checkpoints ./checkpoints`, `python make_exact_baseline.py --checkpoints ./checkpoints`, and `python build_report.py --checkpoints ./checkpoints`. The separately supplied analysis coefficient files belong in coefficients/.
5. Toy calibration: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python calibrate.py`, then the same thread settings for `python summarize.py` and `python build_report.py`. The archived no-fitting inspector is `python inspect_component.py --task joint --seed 1 --component 197 --start 0 --end 800`; it still requires excluded saved model arrays.
6. Component causality: `python download_checkpoints.py`, then `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python causal_probe.py discover` and the same command with `confirm`; `OPENBLAS_NUM_THREADS=1 python summarize.py`, `OPENBLAS_NUM_THREADS=1 python figures.py`, and `python build_report.py`.
7. Matched components: `python download_checkpoints.py`, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python run_experiment.py`, `OPENBLAS_NUM_THREADS=1 python analyze.py`, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python verify_scoring.py`, `OPENBLAS_NUM_THREADS=1 python figures.py`, and `python build_report.py`.
8. Training compatibility: `python verify_bundle.py`, `python download_sources.py`, `python pythia_train.py`, `python stable_sgd.py`, `python toy_depth.py`, `python analyze.py`, `python verify_saved.py`, `python make_figures.py`, and `python build_report.py`.
9. Retention/depth: `python verify_bundle.py`, `python download_sources.py`, `python run_retention.py`, `python transport_toy.py`, `python analyze_phase.py`, `python verify_saved.py`, `python make_figures.py`, and `python build_report.py`. The historical recovery helper is not needed for normal reproduction using the fixed source.
10. Masks/transport: `python verify_bundle.py`, `python download_sources.py`, `python run_masks.py --checkpoints checkpoints`, `python augmented_transport.py`, `python analyze_phase.py`, `python make_figures.py`, `python verify_saved.py --checkpoints checkpoints`, and `python build_report.py`.

The SmolLM commands depend on the earlier phase-6 experiment. Pythia downloaders use pinned public checkpoint identities, but downloading/restoring a model is outside the present arithmetic-only publication audit. No new experiments are implied by listing these commands.
