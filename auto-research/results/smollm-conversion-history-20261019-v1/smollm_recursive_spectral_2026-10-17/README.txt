Recursive spectral update sufficiency study, 17 October 2026.
Open results/recursive_spectral_report.html.
This is post-hoc masking of an already trained model update, not a new training run.
The global waveform FFT and localized intervention FFT use explicitly different axes; see protocol.json.
Requires the existing smollm_initialization_phase6 bundle and pinned original model.
Run global_probe.py --phase6 PATH, interventions.py --phase6 PATH, targeting_control.py --phase6 PATH, layer_permutation_probe.py --phase6 PATH, then build_report.py.
The targeting control is an exploratory addendum after the primary outcomes; see protocol_addendum.json.
The layer/permutation extension is exploratory and has its own frozen protocol. It does not align different layers.
No large model checkpoints or teacher caches are duplicated here.
