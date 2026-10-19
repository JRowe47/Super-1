# Model attribution

Reference: HuggingFaceTB/SmolLM2-135M, by Hugging Face and the SmolLM team.
https://huggingface.co/HuggingFaceTB/SmolLM2-135M

Pinned revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
File SHA-256 hashes are in `model_manifest.json`.

The model is provided under Apache License 2.0; its license text is retained in
`MODEL_LICENSE.txt`. Saved first-block checkpoints in `checkpoints/` are modified
derivatives of that model: the selected query/key projections are replaced by
FP32 low-rank factors, or the selected MLP retains fewer original neurons.
Other first-block tensors retain their original values in FP32 curve buffers.
The complete original model is not included; `download_model.py` retrieves it.

These are experimental inference checkpoints, not official SmolLM releases.
