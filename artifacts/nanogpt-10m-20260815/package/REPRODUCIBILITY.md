# Reproducibility contract

## Two supported modes

### Recorded-evidence reproduction

This mode is CPU-only. It recomputes the comparison JSON, all state formulas,
CE deltas, benchmark means, charts, and report from the included result
artifacts. It is fully reproducible from the package bytes.

### Fresh training reproduction

This mode trains both models from initialization under the same declared
contract. It requires the external corpus, teacher snapshot, benchmark caches,
one CUDA GPU, and sufficient disk space. The protocol, steps, examples,
objective, and evaluations are reproducible. A bit-identical final checkpoint
is not promised across different CUDA, PyTorch, driver, and kernel versions.

## Identity levels

### Official nanoGPT

The comparator is frozen exactly. The package binds:

- upstream nanoGPT commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`;
- vendored `model.py` SHA-256
  `7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`;
- parameter-free adapter SHA-256
  `1adfffbd843a55f00491b50cb46bc6c5d99041f54d9f48251ce3e79ad672a870`;
- frozen manifest SHA-256
  `dc4e324d7f43ddf417c6641ce3c6ec856cbec69a6d1b83e93f2fe5b0690c833c`.

### Historical pure hierarchy

The historical config is byte-identical with SHA-256
`f47d8cabafa5f16ff32c1169fba2064bf8124907cfc99e7d8374212cc25c58fa`.
The included code is the registered compatible snapshot that was previously
validated against the historical artifacts. It is not byte-identical to all
four originally deployed core files. The original deployed hashes are retained
in `provenance/source_identity.json`.

Therefore:

- recorded historical result verification is exact;
- the historical experiment can be structurally and statistically reproduced;
- a bit-identical historical checkpoint rerun is not claimed from this source
  snapshot alone.

## External input identity

Exact bytes, not regenerated content, define the input identity:

- `train.bin`: 4,000,000,000 bytes, SHA-256
  `834c82d2ea76e2f17aa43d15862f5ef444322c37d9b29d07fe803bb21ac163e4`;
- `val.bin`: 8,000,000 bytes, SHA-256
  `8517985c4979dbc95c83137ee534e30e789e2275e4ba1ee1fd31e1f0d566e4d3`;
- Qwen3-0.6B `model.safetensors`: 1,192,135,096 bytes, SHA-256
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`;
- teacher revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

The corpus was prepared using the Qwen3-1.7B tokenizer family and historically
consumed under the `qwen3-06b-representation` path. Recreated data with a
different hash is a different experiment.

## Benchmark data

The declared Hugging Face dataset revisions are bound in `package.json`.
For offline reruns, populate the corresponding dataset cache before launch.
The four benchmark result JSON files are included, but the dataset caches are
not copied into this code archive.

## Recorded evidence included

The package includes the historical result and benchmark JSON as well as the
official nanoGPT result, preflight, completion receipt, resolved config,
246-record training log, benchmark, and benchmark completion receipt. The
large final checkpoints are identified by hash but not duplicated.

The three historical training-curve rows were preserved from immutable live-log
transcripts; the original full historical `train.jsonl` is not included. They
are used only for the supporting training chart and not for the primary fixed
evaluation or state claim.

## State accounting

State values are analytic BF16 per-sequence architecture accounting:

- hierarchy: `4 * width 688 * 2 bytes = 5,504 bytes`;
- nanoGPT: `2 * 21 layers * 7 heads * 64 head dimension * context * 2 bytes`.

They are not measurements of total GPU memory and do not include model
parameters, activations, optimizer state, allocator overhead, or runtime
workspace memory.

## Create-only execution

The package wrappers reject an existing output path. The official comparator
also writes preflight and completion receipts create-only. Use a fresh output
directory for every attempt and preserve failed attempts.
