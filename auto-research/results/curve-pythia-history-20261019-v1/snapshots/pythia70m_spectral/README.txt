Pythia-70M spectral checkpoint study — 17 October 2026

Open pythia70m_spectral_report.html for methods and results.
No model training is performed.

Reproduction:
1. Install requirements.txt.
2. Run download_checkpoints.py (about 3.66 GB public weights).
3. Run analyze_spectra.py, behavior.py, transition_probe.py, low_frequency_controls.py, validate.py, build_report.py.
Use OPENBLAS_NUM_THREADS=1 or 2 and OMP_NUM_THREADS=3 to avoid oversubscription.

All checkpoints have pinned revisions and SHA-256 hashes in checkpoint_manifest.json.
The initial download resolves branch refs only if no manifest is present.
Report figures and machine-readable metrics are in results/.
No external code repository is modified.
