This bounded synthetic experiment tests whether norm-preserving angular fast-state modulation beats simpler equal-parameter radial scaling on fresh modular-accumulator documents. It did not: rotation lost to the radial control and native hierarchy on all three seeds. No architecture is promoted.

Protocol and code were committed before training at `754aae8760321138fcf6b743d7117ffd16c7d851`, based on current main `ab0fbf60bc4cc9f2b9bb397d69ea80139008b1ff`. This changes both the task and the explanatory control from the previous repeating-motif/orientation experiment; previous negative/failure receipts remain intact.

| Seed | Native readout CE | Rotation | Radial | Pinned nanoGPT |
|---|---:|---:|---:|---:|
| 73 | 3.906930447 | 3.912286192 | 3.909294099 | 2.097609997 |
| 191 | 4.037421405 | 4.042921543 | 4.040547907 | 2.109694898 |
| 307 | 3.879141450 | 3.911892533 | 3.871451885 | 2.098059624 |

Rotation−radial mean: **+0.015268793 nats**, descriptive paired 95% t interval **[−0.038889533, +0.069427118]**. The preregistered useful-effect threshold was −0.02 nats plus all-seed/native and uncertainty gates. Decision: `NO_PROMOTION_INCONCLUSIVE_OR_NEGATIVE`.

All 12 arm/seed runs completed 80 updates and 10,240 target exposures each. Native source is unchanged; geometry/control have 8,832 parameters, native 8,816, nanoGPT 8,136. All checkpoint hashes froze before confirmation; all 12 duplicate FP32 confirmations match exactly. Checkpoints with optimizer states, 960 progress rows, source/data hashes, reports and logs are preserved.

One CPU thread, synthetic data only; no GPU/MPS/cloud/install/external corpus/official test. Training plus all evaluations used 7.873963 measured CPU seconds and 8.449360 wall seconds. Both verifiers passed, consuming 0.111004 additional measured CPU seconds; final artifact accounting is in `STORAGE.json`, well below 50 MiB. No runtime failures, recovery, posthoc threshold changes or retraining occurred.

Limits: n=3, 128 fixed confirmation readouts; all-token training weights unpredictable commands seven times more than readouts; hierarchy readout CE worsened from initialization and nanoGPT remains near uniform readout prediction. This does not establish successful modular computation. FLOPs/walltime/parameters are not matched across native and nanoGPT; no strong RoPE baseline or deployment benchmark. This is synthetic workflow evidence, not language quality, SOTA, novelty or a scaling result.

Report: [full results and exact commands](https://github.com/rohanarun/Super-1/blob/codex/synthetic-rotation-radial-20261016/auto-research/reports/synthetic-rotation-radial-20261016-v1.md).

Read-only verification:

```sh
python3 -B auto-research/results/synthetic-rotation-radial-20261016-v1/verify_results.py
python3 -B scripts/verify_evidence.py
```

The exact optimization command (already executed; do not overwrite immutable output) is documented in the report and `EXIT.json`. A fresh authorized replay uses the frozen runner with `--output` pointing at a new SSD directory and the recorded single-thread environment variables. Future training requires new authorization. Leave this PR unmerged.
