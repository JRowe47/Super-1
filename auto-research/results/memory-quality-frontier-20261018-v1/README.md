# Memory and quality pilot: relaxing compression did not improve quality

The compact gated-delta hybrid had the lowest held-out byte loss in this three-seed microbenchmark. Replacing two of its recurrent blocks with full attention increased loss in every seed. Both hybrids beat the local Transformer control, but neither demonstrated a fourfold reduction in total inference memory. The preregistered combined gate failed.

This is a new, simplified gated-delta hybrid, not a less-compressed checkpoint of the original hierarchy. Its byte losses must not be compared numerically with the earlier FineWeb token losses of 4.317 and 3.750. The control is our nanoGPT-style RoPE microbaseline, not pinned upstream nanoGPT or a SOTA reproduction.

## Quality at the trained 256-byte context

All arms have four blocks, width 64, four heads and a tied 256-byte embedding/readout. A is full causal RoPE attention; D is a normalized gated-delta memory block. Each seed/arm received the same 500 minibatches, 512,000 target exposures and optimizer recipe. Parameters differ by less than 1%. All selected checkpoints were frozen before fresh confirmation; all selected step 500.

| Arm | Pattern | Parameters | Seed 73 CE | Seed 191 CE | Seed 307 CE | Mean CE | Mean training CPU seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer | AAAA | 214,144 | 2.275262 | 2.286304 | 2.270385 | 2.277317 | 12.899 |
| compact | DDDA | 215,752 | 2.223631 | 2.194202 | 2.195396 | 2.204410 | 21.953 |
| relaxed | DAAA | 214,680 | 2.241904 | 2.248856 | 2.238726 | 2.243162 | 16.083 |

CE is nats per byte; lower is better. Each endpoint scores 32,768 targets from 128 disjoint 257-byte windows. The fresh confirmation region is `[1070414,1103310)` of the pinned corpus; train and selection regions were previously exposed research data. This is one short contiguous text region, not a broad benchmark. Paired intervals reflect three training seeds only, not uncertainty across domains or documents.

| Contrast | Mean CE difference | Paired 95% t interval | Perplexity ratio from mean CE |
|---|---:|---|---:|
| compact minus transformer | -0.072907 | [-0.123374, -0.022440] | 0.929687 |
| relaxed minus compact | +0.038752 | [-0.007497, +0.085002] | 1.039513 |
| relaxed minus transformer | -0.034155 | [-0.041546, -0.026763] | 0.966422 |

Replacing recurrent blocks with attention changes the model's processing and inductive bias as well as information retention. These arms are not nested capacities of one fixed memory mechanism. The result therefore does not isolate compression as the cause of the difference. A cleaner memory-capacity ablation would hold the update mechanism and block arrangement fixed while varying recurrent state capacity.

The relaxed arm passed the preregistered 1% perplexity noninferiority criterion against the control. The relaxation-improvement hypothesis failed: relaxed loss was higher than compact loss in every seed, and its mean difference was +0.038752 nats/byte. The compact result is reported as a secondary comparison, without changing the primary arm after seeing results.

## Inference measurements

Each of 27 measurements ran in a fresh CPU FP32 process, one thread, batch one, with 64-byte streaming prefill chunks and 32 teacher-forced cached decode steps. Peak RSS includes runtime, allocator, weights, state and transients. Retained tensor bytes count weights and the final cache/state only. These are different memory measures. Latency is one sample per trained checkpoint/context and is exploratory, not a stable production throughput benchmark.

| Context | Arm | Mean peak RSS MiB | Retained weights + state MiB | Mean paired RSS ratio | Mean paired prefill time ratio | Mean paired decode time ratio |
|---:|---|---:|---:|---:|---:|---:|
| 256 | transformer | 171.625 | 1.379 | 1.0000 | 1.0000 | 1.0000 |
| 256 | compact | 182.917 | 0.975 | 1.0660 | 1.3887 | 1.1624 |
| 256 | relaxed | 176.672 | 1.245 | 1.0297 | 1.1862 | 1.0307 |
| 2048 | transformer | 185.328 | 4.879 | 1.0000 | 1.0000 | 1.0000 |
| 2048 | compact | 182.031 | 1.850 | 0.9844 | 0.6404 | 0.7849 |
| 2048 | relaxed | 192.104 | 3.870 | 1.0384 | 0.8585 | 0.9757 |
| 8192 | transformer | 247.411 | 16.879 | 1.0000 | 1.0000 | 1.0000 |
| 8192 | compact | 202.406 | 4.850 | 0.8178 | 0.3703 | 0.4741 |
| 8192 | relaxed | 244.161 | 12.870 | 0.9864 | 0.7745 | 0.7650 |

Ratios are candidate/control, paired by seed; lower is better. Full per-seed measurements, startup/loaded peaks, checkpoint hashes and times are in `execution/MEMORY.json` and `execution/memory/`. Quality was trained and tested only at 256 bytes. The 2K and 8K rows test cached execution and memory mechanics; they do not establish useful long-context language quality.

At 8K, compact retained 71.3% fewer tensor bytes, but reduced paired total peak RSS by only 18.2%. Relaxed retained 23.8% fewer tensor bytes, with only a 1.4% mean paired total-RSS reduction. The relaxed arm failed the predeclared fourfold total-memory criterion; its latency criterion passed. Its three full-attention blocks structurally limit cache savings. There is no demonstrated fourfold total-memory result.

## What this changes

Training efficiency can be a secondary objective when deployment savings justify the expense. Compact used approximately 1.70 times the control training CPU here. That tradeoff is acceptable to investigate, but these equal-exposure short runs do not establish optimal training compute allocation. All arms used one optimizer recipe and may be undertrained.

The result motivates retaining the compact candidate for a separately specified stronger benchmark. It does not support relaxing compression in this particular hybrid. No new larger run is launched by this report. A future comparison should use a tuned, pinned nanoGPT baseline, broader text, longer matched train/eval contexts, and the intended inference hardware. Training compute and tuning effort should remain disclosed even if inference memory is the primary objective.

All four blocks execute on every token in this pilot. Dynamic layer sampling is not measured here. Skipping layers may reduce compute, but does not automatically evict weights or preserve valid historical caches when a skipped layer is later activated. Any dynamic policy needs its own quality, cache-consistency, latency and resident-memory evaluation.

## Provenance and verification

- Preregistered source commit: `96a6cff489f0d186a0cfee737caeb159fa6c2700`.
- Configuration, split, selection and gates: `CONFIG.json`, `PROTOCOL.json`; original source hashes: `SOURCE_MANIFEST.json`.
- Pinned FLA recurrence, paper and corpus sources: `SOURCES.json`; vendored source and license are retained.
- Initial upstream-module import failed because `einops` was absent. `CANARY_INITIAL_FAILURE.json` preserves the failure. The unchanged recurrent function was extracted through Python AST, avoiding unrelated chunk-function imports. No dependency was installed and no numerical tolerance was relaxed.
- `CANARY.json` records upstream FP32 reference agreement, FP64 chunk/sequential output and gradient agreement, optimizer agreement, causal behavior and cached/full equivalence.
- `execution/VERIFICATION.json` records exact independent recomputation of all nine confirmation endpoints, selected minima, matched minibatches, checkpoint/source hashes, and trained cached/full agreement.
- `execution/MEMORY_VERIFICATION.json` audits all 27 workers, checkpoint identities, exact cache/model tensor accounting, ratios and decision arithmetic. Its audit script was added after measurements; it did not affect the frozen protocol or select results.
- All 54 training checkpoints and nine inference checkpoints are preserved, including worse configurations. `FREEZE.json` precedes confirmation. There was no adaptive retraining on confirmation.
- Training and confirmation finished in 159.579 CPU seconds and 163.899 wall seconds. Memory workers took 34.858 wall seconds separately. No GPU or paid cloud resources were used.
- Environment: Python 3.9, PyTorch 2.8.0, macOS CPU. There are no quantization or optimized GPU-kernel claims.

## Reproduction

Use a separate writable SSD checkout and output location for any new execution; do not overwrite the stored evidence. The training runner refuses to reuse an output directory. The memory runner and verifier use the run directory's `execution/` and write receipts exclusively, so replay them in a copy with that directory absent or the relevant new receipt paths configured. Preserve this committed execution.

```sh
/usr/bin/python3 -B auto-research/results/memory-quality-frontier-20261018-v1/canary.py
/usr/bin/python3 -B auto-research/results/memory-quality-frontier-20261018-v1/run.py --output "$PWD/auto-research/results/memory-quality-frontier-20261018-v1/execution"
/usr/bin/python3 -B auto-research/results/memory-quality-frontier-20261018-v1/verify.py
/usr/bin/python3 -B auto-research/results/memory-quality-frontier-20261018-v1/measure_all.py
/usr/bin/python3 -B auto-research/results/memory-quality-frontier-20261018-v1/audit_measurements.py
```

Set `TMPDIR` to an existing directory on the SSD and `PYTHONDONTWRITEBYTECODE=1`. The read-only repository evidence audit is `python3 -B scripts/verify_evidence.py`; unlike the commands above it can be run directly against the preserved evidence.
