# Full-Fourier parameters and hybrid predicted updates

Completed bounded CPU study. **No promotion:** the conservative hybrid fell back on all 1,000 main updates per seed. Confidence-only gating avoided 91.0% of main backward calls but impaired new learning and still cost more CPU. Complete Fourier numerical qualification passed original tolerances.

Read the [report](../../reports/hybrid-curve-updates-20261017-v1.md), [aggregate results](SUMMARY.json), and [frozen protocol](PROTOCOL.json). Main evidence comprises three paired seeds and five arms. Pilot seed 90917 is excluded from main analysis.

## Evidence files

- `experiment.py`: unchanged frozen main model, update predictor, gate, training and evaluation implementation.
- `launch.py`: bounded three-worker CPU launcher; exclusive output creation prevents overwriting completed runs.
- `PILOT_PROTOCOL.json` and `pilot_source/`: original four-arm pilot protocol and exact source.
- `PROTOCOL.json`, `REGISTRATION.json`, `SOURCE_MANIFEST.json`: immutable local registration before main execution. The shared registry entry was created after pilot inspection and before main; no external preregistration is claimed.
- `VERIFICATION.json`: original-tolerance FP32/FP64 fidelity checks and no-autograd application check.
- `main/` and `pilot/`: calibration, progress traces, timing/count summaries and seed terminal receipts. Traces include every audit plus periodic progress, not every unaudited step.
- `CHECKPOINT_FREEZE.json`: 18 main checkpoint hashes frozen before final confirmation.
- `CONFIRMATION.json`: all 15 final scores, repeated exactly after checkpoint reload.
- `analyze.py`: descriptive aggregation/report/plot generation from saved evidence. It performs no training or model evaluation.
- `benchmark.py`, `COMPUTE_DIAGNOSTIC.json`: post-main optional timing diagnostic, including dense re-export of updated Fourier parameters; not the main training-step cost.
- `PRESERVATION.json`: companion archive identity, checksum and extraction instructions.
- `EVIDENCE_MANIFEST.json`: final evidence hashes; does not replace the preregistered source manifest.

## Replay saved checkpoints

The companion `hybrid_curve_update_checkpoints_2026-10-17.zip` contains 23 checkpoints: five pilot and eighteen main snapshots. Extract its contents directly into this directory; paths begin with `main/` or `pilot/`. The archive's `CHECKPOINTS.sha256.json` covers all checkpoint bytes. Checkpoints contain tensor state and basic data, not arbitrary model classes.

From this directory, with Python 3.12, NumPy, and the recorded PyTorch 2.14.1+cpu environment available:

```bash
python3 experiment.py evaluate
python3 analyze.py
```

The first command verifies each main final checkpoint against its preexisting summary hash, scores it twice, and writes `CONFIRMATION.json`. The second verifies all main checkpoint hashes and frozen protocol/source hashes before writing the descriptive report and plots. Matplotlib is required only for analysis plots. Timing will differ across hardware; exact parity claims concern this recorded environment.

## Repeat training separately

Do not delete or overwrite the preserved results. In a fresh repository-shaped directory, copy `experiment.py`, `launch.py`, `PROTOCOL.json`, `REGISTRATION.json`, and `SOURCE_MANIFEST.json`. Keep the same relative layout `auto-research/results/hybrid-curve-updates-20261017-v1/` so launcher logs have the intended location. Run:

```bash
python3 experiment.py verify
python3 launch.py
python3 experiment.py evaluate
```

The launch starts three one-thread workers, each capped at 120 process CPU seconds. It rejects existing seed/log outputs. The source digest must match the frozen main protocol. For the original pilot, use the original source and protocol in `pilot_source/` in a separate empty directory; its source hash intentionally differs from main because main added a diagnostic arm.

This experiment uses labeled synthetic operators and an external composition executor. It does not implement token-context memory, a learned semantic hierarchy, variable depth, neural ODEs or an LLM. The controller is frozen after warmup; continuously training it on fallback remains a follow-up hypothesis. Full Fourier coordinates preserve this learner's parameter family, but no transform speedup was demonstrated.
