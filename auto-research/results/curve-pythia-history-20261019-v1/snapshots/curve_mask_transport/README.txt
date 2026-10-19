CURVE MASK TARGETING AND AUGMENTED LOCAL TRANSPORT — 2026-10-17

Start with curve_mask_transport_report.html (standalone, embedded figures).
Raw measurements are under results/. The protocol was fixed before results;
results/execution_notes.json documents one pre-training shape correction.

Pythia: same pinned 70m step143000 checkpoint as earlier phases. All72 decoder
parameter tensors are complete orthonormal Haar curves in native C order.
No coefficients are discarded. Only their optimizer updates are masked.
Five arms,24 updates each; all use the same lambda1 teacher replay objective.
Four masks have identical coefficient counts per tensor/detail level, and all
five runs have matching per-step coefficient update L2 norms. Evaluation is
fresh48 text blocks and48 short copying sequences, plus6 long probes and8
reused prose diagnostics. A static gradient-based target is tested against
2 random masks and an opposite-ranked mask. This is not a GSA implementation.

The routing toy adds a zero-initialized auxiliary real channel at each of12
sites. Six alternating onsite/neighbor rotation pairs can represent the
six-hop shift exactly. Twelve smooth pulses and a boundary-aware exact solver
preserve the operator under subdivision. Five seeds each for12-control and
2-control training;400 updates each. Costs: doubled state and at least12
rotations. Coarse sampling/Euler failures remain visible in the report.

REPRODUCE
Python versions are in requirements.txt and results/saved_model_audit.json.
From this directory:
  python verify_bundle.py
  python download_sources.py
  python run_masks.py --checkpoints checkpoints
  python augmented_transport.py
  python analyze_phase.py
  python make_figures.py
  python verify_saved.py --checkpoints checkpoints
  python build_report.py

Training/evaluation token arrays, synthetic probes and tokenizer are bundled.
Recreating token selection additionally needs the pinned dataset parquet:
  python download_sources.py --dataset
  python prepare_inputs.py --source-data data --checkpoints checkpoints
Token block source rows/hashes, exclusion records and pinned model metadata
are included. WikiText was not verified absent from Pythia pretraining.

CHECKPOINTS AND MASKS
models/full_final_decoder.npz and targeted_final_decoder.npz contain decoded
float32 decoder weights only. Embedding, output head and final norm remain
from the base model. No final Adam states or float64 coefficient masters are
saved; these files reproduce evaluated inference, not bit-exact continuation.
Other arms have full metrics and deterministic source, but no final weights.
mask_*.npz arrays pack boolean masks with NumPy bitorder='little'; unpack and
truncate to tensor_mapping.json counts. Haar packing is coarse approximation,
then detail bands from broadest to finest. tile_scores.npz has explicit column
names and tensor indices into tensor_mapping.json. Score tiles are64 adjacent
coefficients within one band. In native weights, a level-j tile covers64*2^j
successive positions. Tensors and their rows are never reordered.

Reports and plots use raw saved measurements; uncertainty resamples held-out
items, not training seeds. The mean random-control result is an average of two
models' per-item metrics, not a weight ensemble. No compression, speedup or
full-language semantic safety claim follows from these measurements.

See DATA_ATTRIBUTION.txt for data/model sources and licenses. The base281MB
checkpoint and source parquet are excluded from the ZIP; immutable downloads
are supplied. SHA256SUMS.json covers every released file except itself.
