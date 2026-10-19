# SmolLM2 joint curves — Phase 5

Open results/phase5_report.html. The report explains the matched protocol, fresh article-held-out audit, four-mode projection and five-mode exact dense-to-curve lift.

Python 3.12, CPU FP32, four threads per worker. Install requirements, run download_model.py (or set SMOLLM_MODEL_DIR), prepare.py, train.py, run_evaluation.py, verify_export.py, build_report.py. Prepared data are included. Teacher caches are regenerated. Default training is sequential; train_dense_parallel.py is optional and must not overlap another dense worker.

600 updates per arm, same recorded batches; one seed. The selected dense control and its five-mode lift have the same learned degrees of freedom, 25% more than the four-mode curve arms. No compression claim. Audit articles are reserved from the official train partition, not an official benchmark test split.

load_candidate('orthogonal_curve', steps=3) returns the function-weight prototype. load_candidate('lifted_dense', steps=3, native_export=True) returns an ordinary native decoder with sampled weights and fixed residual scaling folded in. Supported resolutions are 2,3,4; the independent dense control supports 2,3. Start a new cache when changing resolution. The function-weight prototype needs this custom loader; native_export is suitable for ordinary model serialization.

Selected checkpoints hold increments relative to reference/phase4_checkpoints. The four-mode projection and complete five-mode lift are regenerated deterministically from dense.safetensors. No separate projected/lifted checkpoint is necessary.

Initial preparation calibration was superseded after a gradient-clearing correction; only calibration_verified.log and results/calibration.json supplied the frozen training learning rates. See reference/preparation_notes.json.
