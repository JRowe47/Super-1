# minLSTM retention and compute experiment

Frozen run: `minlstm-retention-20261017-v1`. See the [report](../../reports/minlstm-retention-20261017-v1.md).

Requirements: Python3.11+, CPU PyTorch2.14.1 (recorded execution version), NumPy2.3.5, Matplotlib3.10.8. Other versions may have different timing and floating-point results.

This research branch preserves the original source, protocols and receipts. Extract the companion checkpoint archive at this run root to add all pilot/main checkpoints and full traces. Work on a separate copy to preserve the original receipts. To replay the primary results and inspect the strict fidelity audit, enter this run directory and use:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python analyze.py
python strict_fidelity_audit.py
python stress.py
python report.py
```

The main recipe was `python verify.py`, `python benchmark.py --canary`, then `python launch.py --split pilot --workers 3` and `python launch.py --split main --workers 3`, followed by analysis and `python benchmark.py`. The launcher refuses changed source hashes and existing trial directories. For fresh training, use a separate directory and preserve the original registration. Do not treat the archived adjusted verification PASS as satisfying the original tolerance: LSTM64 failed the original FP32 optimizer-parameter parity gate and its threshold was relaxed contrary to repository instructions. It is a descriptive supplemental control. The three equal-state models passed the original tolerances. This deviation must be resolved before using LSTM64 as a confirmation control in future work.

All main arms use final checkpoints after 3,000 updates. Pilot outputs are separate. Sources listed in `REGISTRATION.json` remained unchanged after registration. `analyze.py` reloads all 12 checkpoints and exactly replays the primary endpoint, and checks paired train/evaluation hashes. The initial pre-training verification failure, adjusted check, and final strict audit are retained. `stress.py` refuses changed checkpoint hashes; its protocol was frozen after the main ceiling was observed, before the extended evaluation.

`COMPUTE.json` contains isolated timing distributions. `training_summary.json` contains counts, timing, stream hashes, and every 100th training step. Full traces and all pilot/main checkpoint tensors are preserved in the companion archive referenced by `PRESERVATION.json`; extract it at this run root to replay the archived models. Artifact hashes are in that archive and `CHECKPOINT_MANIFEST.json`.

The user explicitly authorized publishing this experiment to the JRowe47/Super-1 fork after the public-publication review requested confirmation. The research PR remains unmerged.

The model deliberately implements normalized signed linear minLSTM and linear minGRU. No rotation, curve controller, multilevel hierarchy, depth adaptation, convolution, or positive-candidate log-space variant is silently substituted. A normal LSTM stores both hidden and cell state.
