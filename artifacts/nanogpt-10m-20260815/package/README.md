# Official nanoGPT vs. Proven Pure Hierarchy - 10M-token reproduction

This package reproduces the matched experiment reported on 2026-08-15:

- the original proven `hierarchical_qwen_full_kl` design;
- a pinned official nanoGPT comparator;
- 4,883 optimizer steps and 10,000,384 executed tokens per arm;
- the same byte-identified corpus, Qwen3-0.6B teacher objective, optimizer
  schedule, deterministic sample schedule, and fixed evaluation protocol;
- fixed evaluation at contexts 512, 1,024, and 2,048;
- the four-task limit-1,000 benchmark.

The code archive intentionally does **not** contain the PDF. The PDF is a
separate deliverable. The archive does contain the report source, metrics,
charts, and portable report builder.

This email-safe edition contains no nested archives. The exact expanded
official nanoGPT source tree and its frozen manifest are included directly.

## Recorded result

| Context | Official nanoGPT CE | Pure hierarchy CE | nanoGPT state / hierarchy state |
|---:|---:|---:|---:|
| 512 | **5.126457** | 5.260253 | 3,500.65x |
| 1,024 | 5.261362 | **5.070329** | 7,001.30x |
| 2,048 | 5.988185 | **5.291337** | 14,002.60x |

The hierarchy uses 5,504 analytic BF16 state bytes per sequence at every
tested context. nanoGPT's theoretical KV state grows from 19,267,584 bytes at
context 512 to 77,070,336 bytes at context 2,048.

## Package map

- `source/historical_hierarchy/` - registered compatible historical snapshot,
  exact historical config, and recorded result/benchmark JSON.
- `source/official_nanogpt/` - exact frozen comparator with pinned upstream
  nanoGPT model bytes and its 7-test contract suite.
- The original source archives are hash-referenced but omitted. Their expanded
  source trees remain included and manifest-bound. This avoids archive nesting
  while preserving the executable code and provenance.
- `evidence/recorded/` - the historical result/benchmark and the completed
  nanoGPT result, contract receipts, benchmark, resolved config, and training
  log.
- `provenance/` - source identities, external input identities, and recorded
  artifact hashes.
- `tools/` - package verifier, input verifier/runner, and comparison generator.
- `report/` - report Markdown, graph data, four charts, and portable builder.
- `package.json` - the machine-readable experiment contract.
- `MANIFEST.sha256` - exact package inventory and content hashes.

## Verify the package

```bash
python3 tools/verify_package.py --run-tests
```

This verifies every package file, both expanded source trees, the frozen
nanoGPT manifest, recorded result hashes, report calculations, and the
comparator's contract tests.

## Verify external inputs

The 4 GB training corpus, validation data, teacher snapshot, checkpoints, and
benchmark caches are not duplicated in this small code archive.

```bash
python3 tools/reproduce.py verify-inputs \
  --runtime-root "$RUNTIME_ROOT" \
  --teacher-snapshot-root "$TEACHER_SNAPSHOT"
```

Expected runtime layout:

```text
$RUNTIME_ROOT/data/qwen3-06b-representation/train.bin
$RUNTIME_ROOT/data/qwen3-06b-representation/val.bin
$TEACHER_SNAPSHOT/model.safetensors
```

The verifier checks byte size and SHA-256 before any launch.

## Run the two arms

All run commands require one process and exactly one explicitly visible GPU.
Every output path must be new.

```bash
python3 tools/reproduce.py run-hierarchy \
  --runtime-root "$RUNTIME_ROOT" \
  --teacher-snapshot-root "$TEACHER_SNAPSHOT" \
  --out "$HIERARCHY_RUN" --gpu 0

python3 tools/reproduce.py benchmark-hierarchy \
  --runtime-root "$RUNTIME_ROOT" \
  --teacher-snapshot-root "$TEACHER_SNAPSHOT" \
  --run-root "$HIERARCHY_RUN" --gpu 0

python3 tools/reproduce.py preflight-nanogpt \
  --runtime-root "$RUNTIME_ROOT" \
  --teacher-snapshot-root "$TEACHER_SNAPSHOT" \
  --out "$NANOGPT_RUN" --gpu 0

python3 tools/reproduce.py run-nanogpt \
  --runtime-root "$RUNTIME_ROOT" \
  --teacher-snapshot-root "$TEACHER_SNAPSHOT" \
  --out "$NANOGPT_RUN" --gpu 0

python3 tools/reproduce.py benchmark-nanogpt \
  --runtime-root "$RUNTIME_ROOT" \
  --teacher-snapshot-root "$TEACHER_SNAPSHOT" \
  --run-root "$NANOGPT_RUN" --gpu 0
```

Add `--dry-run` to any launch command to print the exact command after input
validation without starting GPU work.

## Compare fresh outputs

```bash
python3 tools/compare_results.py \
  --hierarchy-result "$HIERARCHY_RUN/result.json" \
  --hierarchy-benchmark "$HIERARCHY_RUN/benchmarks.json" \
  --hierarchy-training-rows "$HIERARCHY_RUN/train.jsonl" \
  --nanogpt-result "$NANOGPT_RUN/result.json" \
  --nanogpt-benchmark "$NANOGPT_RUN/benchmarks.json" \
  --nanogpt-training-log "$NANOGPT_RUN/train.jsonl" \
  --out comparison.json
```

For the included recorded artifacts:

```bash
python3 tools/compare_results.py --recorded --check report/metrics.json
```

## Rebuild charts or a PDF

The PDF is generated separately and is not part of the code archive.

```bash
python3 report/build_report.py --mode charts
python3 report/build_report.py --mode pdf --pdf /absolute/output/report.pdf
```

See `REPRODUCIBILITY.md` for identity levels and exact claim boundaries.
