# Feedback-readout learning factorial

Five arms: BP, frozen, U0-only, U1-only and both-U. B/V fixed and online feedback decay off. Full original rates, shared4096-step warmup, forward optimizer, repaired strict cap2 and FIFO512/16 stay fixed. Main seeds1301841/1301842/1301843; pilot102937. Main2048 online steps/arm; pilot256. Sole primary U1-only versus frozen keeps>=5% CE reduction/no negative seed and<=2pp old-window/final-old loss in every seed. All20 online trajectories must replay bitwise.

With the pinned RUNTIME.json packages restored and PYTHONPATH set:

```sh
python3 launch.py verify
python3 launch.py pilot
python3 launch.py main
python3 analyze.py
```

Use fresh output paths to reproduce; never overwrite completed receipts. Frozen source/protocol must be published and read back before numerical checks or training. launch.py bounds every child including imports. Compact source/JSON evidence lives in Git; four individual compressed NPZ checkpoints store shared warmup plus each arm's finalW and learnedU only. No cumulative archive or data dump.

All previous checks and original absoluteFP32 2e-6/FP64 2e-12 tolerances remain. New checks verify inactive matrices, U masks and both-U/VU W/U/B equivalence with B fixed. W0 still learns under U1-only. No rate search, paid/cloud/GPU or language-model run; no transfer, novelty, scalability or deployment claim. Prior missing-source and failed-cap caveats remain.
