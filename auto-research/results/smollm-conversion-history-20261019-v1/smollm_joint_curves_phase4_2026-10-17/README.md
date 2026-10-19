# SmolLM2 joint curves — phase 4

Open results/phase4_report.html for methods, measurements and limitations.

Run with Python 3.12 on CPU, at least 4 GiB available memory (8 GiB recommended). Install requirements, download the pinned original model, then run prepare.py, train.py, run_evaluation.py and build_report.py. Use SMOLLM_MODEL_DIR to point to existing model files. Token windows are supplied; teacher caches are regenerated.

To replay selected models, use load_candidate.load_candidate({4:3,12:3}); supported windows are 4 and 12 and supported resolutions are 2,3,4. This example is not a quality recommendation. Set the resolution before a sequence and start a fresh DynamicCache after changing it. All original model weights are retained. This CPU prototype requires its custom loader; Hugging Face save_pretrained does not export the external curve banks.

checkpoints/window_*.safetensors contain final selected slopes. *_local files are selected local-stage checkpoints. Coefficients are depth-only. This is not a GSA reproduction or a compressed-weight model. See results/protocol.json for prespecified settings and selection.

The earlier survey archive was recovered and rebuilt separately after a truncated stored ZIP was detected. Its complete entries passed CRC checks; the survey code, raw arrays and reports were then regenerated from the pinned original model.
