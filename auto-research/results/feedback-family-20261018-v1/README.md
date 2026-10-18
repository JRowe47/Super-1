# Feedback matrix-family learning factorial

Five arms: backprop, frozen, B-only, V/U-only and full learning-only. Online feedback decay is off. Full original feedback rates, common4096-step warmup, forward optimizer, repaired strict cap2 and FIFO512/16 are fixed. Main seeds1201831/1201832/1201833; pilot101927. Main2048 online steps/arm, pilot256. Primary B-only versus frozen uses the unchanged >=5% CE reduction/no negative seed and <=2pp old-window/final-old loss per seed. All20 trajectories must replay bitwise.

With the pinned RUNTIME.json packages restored and PYTHONPATH set:

```sh
python3 launch.py verify
python3 launch.py pilot
python3 launch.py main
python3 analyze.py
```

Use a fresh output copy for reproduction; never overwrite completed receipts. All frozen source/protocol must be published/read back before numerical checks or training. launch.py bounds each child including imports. Compact source/JSON evidence lives in Git; four compressed NPZ states are preserved individually. The complete final state is shared warmup overlaid with stored arm; only learned feedback families need an arm-specific copy. Streams regenerate from seeds, with no archive or data dump.

Original FP32 absolute2e-6/FP64 absolute2e-12 tolerances and all previous cap/component/FIFO regressions remain. No rate search, cloud/GPU or language-model run. Synthetic results do not establish transfer, novelty, scalability or deployment efficiency. Earlier missing-source evidence and failed-cap chronology remain explicitly caveated.
