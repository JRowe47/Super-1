# Magnitude-matched feedback diagnostic

Small CPU synthetic experiment, not language-model training. Frozen source and protocol are committed before verification or training. Read PROTOCOL.json, REGISTRATION.json and SOURCE_PROVENANCE.json first.

Runtime: Python 3.12.14, numpy 2.3.5, torch 2.14.1+cpu. Use an environment containing these versions and one CPU thread. From this directory, in order:

```sh
python3 launch.py verify
python3 launch.py pilot
python3 launch.py main
python3 analyze.py
```

The launcher refuses to overwrite receipts and workers refuse existing seed directories. Reproduction should use a fresh checkout/output directory with frozen source files; preserve the original outputs first. Main is gated on a successful numerical verification and mechanical pilot. Pilot is not used to tune parameters or select configurations.

`main/<seed>/RESULT.json` includes compact metric traces, matched-drive audits, data/state hashes and full deterministic replay receipts. `pilot/<seed>/RESULT.json` is the mechanical check. `STATES.npz` per seed holds common warmup W/F, original random feedback, all final states and final adaptive shadow. No pickle is used. Data are regenerated from the byte-preserved generator and verified by a canonical recursive SHA-256. Phase states are checked by hash on complete replay; large streams and intermediate tensors are not exported.

The primary contrast is adaptive versus frozen_match_adaptive. The unscaled and reciprocal contrasts are descriptive. All three seeds and all failures are retained. A primary gate pass would remain a synthetic diagnostic, not an architecture promotion or an efficiency result.
