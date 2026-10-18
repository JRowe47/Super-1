# Super-1 function-weight compression: progress and handoff

Updated: 2026-10-18, America/New_York.

## Objective and non-negotiable architecture

Reduce actual stored/resident model weights while retaining language quality. Preserve function-generated weights, learned token-conditioned t sampling and selectable inference depth. Hierarchy is a separate KV-storage/readout mechanism, not permission to replace the backbone with GRU/RWKV. More training compute is acceptable. No SOTA,70B or100K recall claim is established.

## Authoritative workspace and safety

Work here: `/Volumes/SP AI 01_16/CodexWorkspaces/Super-1-memory-frontier-20261018`.
Branch: `research/memory-quality-frontier-20261018`. Verify SSD mount/writability before any write. Never write development outputs into Documents/Desktop/iCloud. All subprocesses must use this SSD workdir. No paid/cloud resources or unbounded training are authorized. Do not overwrite frozen runs or delete failed/partial artifacts. No subagents were requested.

Preserve the unrelated untracked `auto-research/results/recurrent-capacity-rwkv-20261018-v1/`. Its interrupted experiment is retained on local branch `research/recurrent-capacity-rwkv-20261018`, commit f10c27f. It is not a completed RWKV result. Preserve the user edit `wonko ` in `sampling_library/README.md` in the separate PR7 worktree `/Volumes/SP AI 01_16/CodexWorkspaces/Super-1-adaptive-depth-20261018`.

## Published progress before this experiment

- PR7 https://github.com/rohanarun/Super-1/pull/7 adds cached inference retaining original function weights and sampling.
- Draft PR8 https://github.com/rohanarun/Super-1/pull/8 adds the optional hierarchical KV backend, two three-seed byte-text pilots, learned pooling libraries and a chart. Last verified head before this task:5a2c5a501dd21214555957eef26326a4dca63eb9. It is stacked on PR7 and was not merged.
- PR9 https://github.com/rohanarun/Super-1/pull/9 is a scoped import of minGRU CPU-training-throughput benchmarks, head4d0dfed819e91e2b7c0efc60ac98692f5bf01c94. Approximately equal1M parameter counts; not evidence of weight compression. Curve-conversion/spectral stages are excluded. No modification/merge of PR9 was requested.
- Report: `auto-research/reports/function-weight-kv-text-descent-20261018.md`.
- At512-byte context, mean CE: pinned nanoGPT2.280836, full curve2.131311,16-entry hierarchy2.134549. Narrow quality gates passed. A separate8K passage favored hierarchy but is not distant-fact recall evidence.
- Separate100K-byte-token mechanics on the128-context tiny model: hierarchy185.7MiB vs full KV1729.0MiB peak RSS,9.31x lower. Original short/8K4x total-memory gates failed and remain failed. These are distinct experiments, not a joint100K quality/memory result.
- All96 prior checkpoints preserved;12 new hierarchy profiles independently reproduce confirmation logits.12 runtime tests+13 library checks+5 depth checks passed;1405 repository hashes verified at previous head. Reverify after edits.
- Current curve model has68,905 unique parameters:55,296 basis values,12,288 tied embedding/output values,864 normalization,409 router,48 curve coefficients. Basis-only4-bit payload estimate54,866 bytes, basis+embedding36,434 bytes versus137,810 FP16, excluding metadata. These were estimates, not measured quantized quality.

## Current experiment status

Preparing a bounded three-seed weight-only experiment. KV stays full causal attention for every arm. Compare full-precision continuation controls,4-bit shared bases,4-bit shared bases+embedding and4-bit nanoGPT; train with next-byte CE+frozen-teacher KL. All arms get matched recovery targets. Select on validation, freeze, then evaluate an unused byte region. Measure packed file bytes and unique packed tensor bytes, including scales, separately from peak process RAM.

Critical requirement: a fake-quant training result alone is insufficient. Export true nibble-packed tensors, load an inference model with no full-precision weight masters, and decode only bounded matrix tiles or requested embedding rows. Verify packed-vs-fake-quant logits and preserve t sampling at depths2/3/4. Keep high-precision coefficient/router/norm parameters initially. Do not claim the prototype is an optimized quantized kernel or a drop-in cached-serving runtime.

## Continuation instructions

1. Read this file and current `git status` before doing anything. Check active processes before launching duplicate runs.
2. Read the new experiment PROTOCOL/CONFIG/SOURCE_MANIFEST and TERMINAL/FREEZE/RESULT files when they exist. A missing TERMINAL is not completion; preserve partial output.
3. Never tune on confirmation bytes or overwrite a frozen run. Keep all seeds, zero-update controls, failed candidates and checkpoints. Do not retroactively loosen quality/storage gates.
4. Inspect actual packed bytes, ties/aliases, resident buffers and tile scratch; never call FP32 fake quantization a RAM-saving inference implementation.
5. Preserve existing architecture and old evidence. Distinguish extra training gains from quantization using matched continuation controls.
6. Update this handoff with run commands, receipts, results, limitations and precise next steps. Publish authorized PR8 changes only after verification; do not merge automatically.

## Journal

- Started this task: verified mounted/writable SSD (~4.7GiB free), preserved unrelated work, inspected trained-model parameter layout and prior protocol. No new training started yet.

- Implemented experimental nibble-packed inference in `sampling_library/packed_weights.py`; random mechanics canary passes all five arms, including curve depths2/3/4, sampled positions, causal prefixes and gradients. Protocol/config/source hashes are in `auto-research/results/function-weight-int4-20261018-v1/`. Controls use FP16 rounding to match actual FP16 deployment; original teachers remain FP32. Training has not started yet.
