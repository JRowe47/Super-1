# SmolLM2 initialization study — Phase 6

Open results/phase6_report.html for protocol, results, plots and limitations.

Two initializations, 1,800 matched updates each, N=3 only, 28 total decoder blocks. The retained policy keeps offsets [0,2,3] in original windows 4–7 and 12–15. Both primary arms have ordinary unit residual strength. Original 30-block weights remain immutable. The new audit excludes phase-5 audit documents.

Python 3.12, CPU FP32, four threads per worker. Install requirements, download_model.py (or set SMOLLM_MODEL_DIR), prepare.py, train.py, norm_control.py, global_curve_feasibility.py, evaluate_phase6.py, verify_export.py, build_report.py. Prepared tokens and selected checkpoints are included; the teacher caches and original pinned model are regenerated. train.py is sequential by default. Optional separate --arm interpolated / --arm retained workers have independent checkpoints and locks. Completed training records are skipped; move their records/checkpoints aside to retrain. Resume state is scratch-only and excluded from this archive.

from models import load_candidate
model = load_candidate('retained', as_curve=False, steps=3)
curve = load_candidate('retained', as_curve=True, steps=3)

Either initialization is supported. Curves expose the fitted N=3 or exact-original N=4 schedule. No N=2 quality claim. Curves are reconstructed deterministically from selected blocks plus reference/phase4_checkpoints; no separate curve checkpoint is needed. Original knots and the previous base curve add storage: coefficient count is not a compression ratio. Native models use ordinary Transformers serialization. Start a new KV cache whenever depth changes.

reference/phase5_dense_n3.safetensors contains the six reference native blocks compiled from the selected phase-5 dense model, with 4/3 residual scaling folded into output/down projections. export_phase5_reference.py records its provenance recipe. This reference is an evaluation control and never supplies phase-6 training weights.
