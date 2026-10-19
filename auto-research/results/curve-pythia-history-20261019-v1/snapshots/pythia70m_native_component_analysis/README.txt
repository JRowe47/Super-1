Pythia native-coordinate component atlas, 17 October 2026

Open pythia70m_native_component_report.html.

The user corrected the research scope: preserve native order; convert each layer to an exact functional curve baseline; examine full-resolution recursively decomposed components and changes across training. Do not substitute permutation alignment or compression benchmarks for this objective.

Reproduction:
python -m pip install -r requirements.txt
python download_checkpoints.py
python insitu.py --checkpoints ./checkpoints
python baseline_check.py --checkpoints ./checkpoints
python make_exact_baseline.py --checkpoints ./checkpoints
python build_report.py --checkpoints ./checkpoints

Set OPENBLAS_NUM_THREADS=1 for these analysis runs.
The separately supplied coefficient NPZ files belong in coefficients/.
Use inspect_components.py for any full-resolution Fourier node or individual Haar detail.
The full-precision step1000 decoder baseline can be restored with make_exact_baseline.restore_decoder_from_curves(model, path), using the pinned step1000 model for the unchanged architecture, embeddings, final norm and output head.

Full resolution and numerical precision are distinct: time-series files retain every coordinate at float32/complex64 precision. Use the complex128 baseline for function-preserving execution. See the report for the measured precision check.
