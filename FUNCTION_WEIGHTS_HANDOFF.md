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

Completed `function-weight-int4-20261018-v1`, with KV held at full causal attention. All five arms and three seeds finished, with 60 immutable recovery checkpoints, 30 original packed exports and 15 compact exports. No process is intentionally left running.

- Primary model (int4 shared bases + tied embedding): 38,546 weight bytes versus 137,810 FP16 bytes, **3.575x smaller**.
- Mean confirmation CE 2.210507 versus matched FP16 recovery 2.203687: **0.684% worse mean perplexity**, passing the preset 2% tolerance on every seed and the paired upper confidence bound.
- Original torch deployment files: 53,925 versus 151,418 bytes, **2.808x**. The original 3x file-storage gate FAILED and remains failed.
- Separately preregistered lossless packaging follow-up: 46,088 versus 144,941 bytes with both controls using the same compact format, **3.145x**. Every tensor and confirmation logit is exactly unchanged.
- Quantized nanoGPT control: CE 2.359474, compact file 92,507 bytes. This is a tiny single-book comparison, not SOTA evidence.
- Compact curve peak process RSS stays around 182–184 MiB; weight compression did not establish dramatic total-process RAM savings at this scale.
- Function curves, token-conditioned t and selectable depths 2/3/4 are preserved. Packed inference is experimental full-sequence CPU inference; it is not yet integrated with cached/hierarchical-KV serving. Language quality was tested only at depth 4.

Results: `auto-research/reports/function-weight-int4-20261018.md`.
Protocols and raw evidence: `auto-research/results/function-weight-int4-20261018-v1/`.
Hash-bound learned profile index: `auto-research/results/function-weight-int4-20261018-v1/LIBRARY.json` (all 15 research profiles inactive by default).
Loader instructions: `sampling_library/PACKED_WEIGHTS.md`.
Chart: `graphs/function_weight_int4.png` and reproducible renderer `scripts/render_function_weight_int4.py`.

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

- Recovery run completed at preregistration commit96f7d72:60 checkpoints,30 packed exports,45 confirmation rows,113.35CPU seconds/121.86wall seconds. All30 exports independently replay exactly;27 trained depth2/3/4 checks pass. Curve bases+embedding:3.575x smaller payload,0.684% mean PPL regression vs matched FP16 control, quality gate passes. Torch serialization yields only2.808x file reduction, so original joint/storage gate FAILS. A separately committed lossless packaging follow-up will test a coalesced byte buffer on every selected arm with no weight/quality changes.

## Exact commands and durable receipts

Workdir for every command: `/Volumes/SP AI 01_16/CodexWorkspaces/Super-1-memory-frontier-20261018`. Use `python3 -B` to avoid bytecode churn. Never rerun a write command into an existing immutable output.

Executed training (already complete; do not launch again):

```sh
python3 -B auto-research/results/function-weight-int4-20261018-v1/run.py canary
python3 -B auto-research/results/function-weight-int4-20261018-v1/run.py train
python3 -B auto-research/results/function-weight-int4-20261018-v1/analyze.py
python3 -B auto-research/results/function-weight-int4-20261018-v1/verify.py --output auto-research/results/function-weight-int4-20261018-v1/execution/VERIFICATION.json
python3 -B auto-research/results/function-weight-int4-20261018-v1/package.py
```

Recovery protocol/source commit: `96f7d72`. Packaging follow-up protocol/source commit: `160d243`. Recovery used 113.35 CPU seconds / 121.86 wall seconds. All runs used one CPU thread. Recovery log is `TRAIN.log`; the terminal receipt says COMPLETED. `FREEZE.json` precedes new confirmation. Full-precision recovery masters and optimizer states are research artifacts; only `.packed.pt` and `.s1pw` are compressed deployment formats.

Receipts in the experiment directory:

- `SOURCE_MANIFEST.json`: model, corpus, parent checkpoints and preregistered code/config hashes.
- `execution/START.json`, `TERMINAL.json`, `FREEZE.json`, `RESULT.json`, `SUMMARY.json`: run identity, completion, selection, all quality results and original gates.
- `execution/VERIFICATION.json`: 60 checkpoint hashes, matching minibatches/selection, 30 exact deployment replays and 27 trained depth/t checks.
- `execution/memory/`: 15 original-format fresh-worker RSS receipts.
- `execution/compact/START.json`, `RESULT.json`: packaging commit/hashes and 15 exact tensor/logit round trips with compact-vs-compact ratios.
- `execution/compact/memory/`: 15 compact-format fresh-worker RSS receipts.
- `LIBRARY.json`: selected checkpoint/source hashes, compact files, tokenizer, context, tested depth and quality.

Safe read-only replay: run `verify.py --output` with a NEW absolute file path on the SSD. For isolated memory replay, use `run.py memory --path <original-packed-file> --destination <new-json>` or `package.py --memory <compact-file> --destination <new-json>`. The destination must not already exist.

Regression commands completed: 13 unittest cases across `sampling_library.test_packed_weights_archive`, `test_inference`, `test_hierarchical_kv`, `test_hierarchy_profiles`; 13 checks in `python3 -B -m sampling_library.test_library`; 5 checks in `python3 -B -m sampling_library.test_profiles`. Chart command: `python3 -B scripts/render_function_weight_int4.py`. Repository hash check: `python3 -B scripts/verify_evidence.py`.

## Next research and integration steps

1. Keep this result scoped to a byte-level Alice model; the new confirmation region is [114128,119771). It is now consumed and cannot become future tuning data while still called untouched. Old confirmation/diagnostic regions are recorded in the parent protocols.
2. Test broader held-out text and larger models with quantized nanoGPT and equal recovery budgets. Preregister a new run/directory before any training; do not modify these frozen sources or relax gates. Do not infer authorization for paid/cloud or unbounded runs.
3. Integrate the packed projection/embedding store with the existing cached function-weight runtime, preserving full-KV/hierarchy behavior and cache identity checks. First require full/cached/chunk and t parity, then measure joint weight+KV peak RAM. Current packed inference does not offer this yet.
4. After the above, investigate 3/2-bit learned codebooks or activation-aware low-rank residuals. Do not conflate the current full-rank quantized bases with a demonstrated low-rank compression result.
5. Preserve the original file-gate failure, all seeds and negative results. Do not multiply the earlier 100K KV RSS ratio by the weight-storage ratio; they are different measurements and not jointly validated.

A report-generation helper initially hit a Python3.9 f-string syntax error; it was corrected before producing documentation. No training, source manifests or numerical results were changed by that documentation-only error.

- Completed final checks: all 30 original exports replayed exactly; all 15 compact exports preserve tensors/logits; 27 trained depth checks and 31 regression checks passed. Both formats have all 15 isolated RSS receipts. Library hashes and both preregistered source sets remain intact. Results/chart/report and this handoff are prepared for draft PR8; remote publication will be verified next.
