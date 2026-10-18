# Extended W560 perturbation research recovery

This supplement preserves the additional SSD and iCloud research located after the initial optimizer import. It contains **271 unchanged historical files** with file-level provenance in [IMPORT.json](IMPORT.json). Read [the findings and interpretation](../../reports/perturbation-extended-recovery-20261018.md).

| Directory | Contents |
|---|---|
| `source/icloud-trin/` | Direction descent, candidate verification/search, scalar diagnostic, chart generation and orchestration scripts recovered from resident iCloud backup files |
| `source/hybrid-runtime/` | Historical distributed worker/controller source, configuration and tests |
| `source/ce-vs-kd/` | Matched objective comparison source, configuration and tests |
| `source/benchmark/` | Historical benchmark implementation |
| `evidence/hybrid-lineage/` | All 21 located decision records and 21 heads, plus available top-level lineage records |
| `evidence/ce-vs-kd/` | Search, selection and final-result JSON records, including parent retention |
| `evidence/scalar-cpu-pilot/` | Available v1/v2/v3 metadata and logs, including the completed v3 diagnostic |
| `evidence/benchmark/` | Canary results, receipts, configuration and diagnostics |
| `evidence/hybrid-tests/` | Historical 43-test and 74-test logs, including skips |
| `evidence/hybrid-pipeline-logs/` | Empty and failed pipeline attempt logs retained without relabeling them as success |
| `evidence/cancelled-audit/` | Audit scaffold explicitly cancelled before any model evaluation |

Verify copied bytes, content addresses, lineage references, saved score arithmetic and benchmark accuracy without loading a model:

```sh
python3 -B auto-research/results/perturbation-extended-recovery-20261018-v1/verify_recovery.py
```

[VERIFICATION.json](VERIFICATION.json) records that fresh audit. The imported source is an archival snapshot, not a self-contained runnable environment: full frozen/vendor dependencies, checkpoints, token banks, teacher logits, benchmark assets and cloud resources remain external. The verifier never opens them. Historical launch scripts/configuration do not authorize execution. No training, model inference, cloud operation or historical test-suite rerun occurred for this supplement.

The two original chart HTML files in the iCloud visualization folder were marked dataless and were not hydrated. Their absence from this supplement is not evidence of deletion. The user's original PNGs remain in the repository's `graphs/` directory. Source files retain original absolute paths and bytes; no originals were modified.
