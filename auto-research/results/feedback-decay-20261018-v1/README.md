# Fixed-FIFO feedback learning/decay factorial

Five arms: frozen feedback(neither component), decay-only, learning-only, normal adaptive(both), and backprop. Original full feedback rate, common warmup, forward optimizer, spectral cap2 and FIFO512/16 remain fixed. Independent learning/decay switches, no new rate sweep.

Main seeds1101821/1101822/1101823, pilot100917. Main4096 common warmup+2048 online updates/arm; pilot4096+256. Primary normal adaptive versus decay-only uses the same>=5% CE reduction/no negative seed and<=2pp old-window/final-old loss in every seed. Report normal-versus-frozen separately. All20 online trajectories must replay exactly.

With pinned RUNTIME.json versions installed and PYTHONPATH set:

```sh
python3 launch.py verify
python3 launch.py pilot
python3 launch.py main
python3 analyze.py
```

Reproduce in a fresh output copy; do not overwrite completed receipts. Exact source/protocol must be published/read back before checks or optimization. launch.py caps imports and all experiment children. Store compact source/JSON in Git and four individual compressed NPZ states in the project folder, without stream dumps or a cumulative archive. Full final state is the shared warmup overlaid with the stored arm state.

Synthetic mechanism test only. A primary pass would not override a failed original frozen-reference retention gate or establish language quality, novelty, scalability or deployment efficiency. Historical missing-source caveat remains in SOURCE_PROVENANCE.json.
