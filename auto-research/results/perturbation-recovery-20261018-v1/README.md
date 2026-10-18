# Recovered perturbation-based weight optimizer

Start with [the optimizer guide](../../PERTURBATION_OPTIMIZATION.md) for algorithm details, source entry points, evidence and reproduction commands.

- `source/hfw-w560-continuous-optimizer-20260815-v1r1/`: original 85-file frozen source release, unchanged.
- `source/mps-runtime/`: later MPS controller/model snapshot, separately inventoried.
- `provenance/frozen-release/`: original release manifests and receipts.
- `evidence/mps-20260822/`: all three located completed decisions and both heads, including rejected proposals.
- `IMPORT.json`: file-level provenance, exclusions and external dependencies.
- `TOY_TESTS.json`: fresh run of all 11 original frozen integrity tests.
- `verify_recovery.py` and `VERIFICATION.json`: read-only hash, source parse and saved-score audit.

No new language training or W560 inference was performed. Saved scores use reused optimization data, not a held-out generalization test. Large checkpoints and token arrays are external, with historical identifiers preserved. Historical launch flags are not current authorization.
