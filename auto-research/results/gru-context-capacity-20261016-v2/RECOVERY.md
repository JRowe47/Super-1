# Fresh recovery sweep

The previous context/capacity study lost execution access after one observed
completion. Its original source/protocol/checkpoint files were not recovered
from the reset working directory. Its console-only observation and interruption
receipt remain unchanged under `../gru-context-capacity-20261016-v1/` in Git.

This is a **fresh v2 study**, not a continuation of a recovered checkpoint or an
independent seed replication. It retains the ten declared model configurations,
initialization/data seeds, common-tail scoring, and 90-second training budgets.
All ten trials start again; no earlier result is silently substituted.

Pinned model source is recovered from commit
`1a7111ac68d5760d52b01cd060f397b1b776ed47`; its six executable dependency hashes
match the preceding completed study. FineWeb-Edu train and validation arrays
come from `gru_extended_checkpoints_20261016.zip`, archive SHA256
`c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6`.
The tokenizer comes from the pinned repository's `data_metadata/tokenizer.json`.
All three input hashes match the prior protocol exactly. No test is accessed.

PyTorch 2.14.1+cpu and NumPy 2.3.5 are restored on the available CPU host.
Actual runtime, clocks and memory are recorded per worker. Source and protocol
are pushed before training; every terminal checkpoint is archived and sent
through the supported preservation helper before launching the next trial.
Per-item upload results must be inspected before claiming preservation.

The inherited optimizer uses the number of embedding rows to choose exclusions.
Both nanoGPT token and position tables have 2,048 rows and receive zero weight
decay. V1's inherited prose omitted the position-table effect. V2 explicitly
declares and checks the existing behavior; no hidden optimizer change is made.

## Reproduction

Use Python with PyTorch 2.14.1+cpu and NumPy 2.3.5, and materialize the hash-bound
parent directories alongside this study. `prepare.py` documents registration;
do not rerun it over an existing frozen protocol. Use a fresh execution/output
copy when reproducing a completed study.

```bash
python experiment.py preflight --data-dir /path/to/exact/data
python launch.py --data-dir /path/to/exact/data --archive-dir /path/to/checkpoint_archives
python analyze.py --data-dir /path/to/exact/data
```

The optional `--upload-helper` points to the current supported Library helper
in the hosted execution environment. External reproduction may omit it and
preserve the generated checkpoint archives through its own durable storage.
Time-budget runs can process different update counts on different hardware.
