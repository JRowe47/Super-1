# Slow feedback with FIFO replay fixed

Four arms: backprop, frozen learned feedback, normal adaptive feedback and adaptive feedback with0.1 of the entire feedback SGD step including decay. Normal forward updates, common warmup, cap2 and FIFO512/16 remain unchanged. One rate fixed before execution, no sweep.

Fresh main seeds1001811/1001812/1001813, pilot99907. Main4096 common warmup plus2048 online steps per arm; pilot4096+256. Same primary CE windows and <=2pp old-window/final-old per-seed retention guard. All16 online trajectories must replay bitwise exactly.

With pinned RUNTIME.json versions installed and PYTHONPATH set to their directory:

```sh
python3 launch.py verify
python3 launch.py pilot
python3 launch.py main
python3 analyze.py
```

Run only in a fresh output copy; never overwrite receipts of a completed run. launch.py caps every child including imports. Exact source/protocol must be published and read back before verification or optimization. Store compact JSON/source in Git and four small individually compressed NPZ states in the project folder; no cumulative archive or stream dump. Full state reconstruction: shared warmup plus each stored final arm overlay.

Synthetic mechanism test only. No language-quality, novelty, scaling or deployment-speed claim. Historical missing-source caveat remains in SOURCE_PROVENANCE.json.
