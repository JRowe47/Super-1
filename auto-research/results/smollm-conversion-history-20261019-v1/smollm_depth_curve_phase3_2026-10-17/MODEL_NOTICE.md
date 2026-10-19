# Model attribution

Reference: HuggingFaceTB/SmolLM2-135M, by Hugging Face and the SmolLM team.
https://huggingface.co/HuggingFaceTB/SmolLM2-135M

Pinned revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
File SHA-256 hashes are in `model_manifest.json`.

The model is provided under Apache License 2.0; its license text is retained in
`MODEL_LICENSE.txt`. Phase-3 checkpoints are modified derivatives initialized
from selected teacher neurons, then fitted to the teacher's MLP outputs. Learned
FP32 spline control vectors represent gate, up and down matrices over depth.
One-pass controls are distilled with the same teacher examples. Original
RMSNorm scales are retained. The phase-2 shared-Q/K reference checkpoint is also
included for the declared baseline and composition test. The complete original
model is not included; `download_model.py` retrieves it.

These are experimental inference checkpoints, not official SmolLM releases.
