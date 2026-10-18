# Refresh replay: fixed-capacity causal bank experiment

Nine arms: backprop, frozen learned feedback and adaptive learned feedback crossed with FIFO, permanent mixed bank and refreshed mixed bank. One bank row replaced per update, no phase signal or oracle relabeling. Source/protocol must be committed and read back before verification or optimization.

Main seeds901801,901802,901803; pilot98997. Main4096 common warmup plus2048 updates per arm; pilot4096+256. Same primary CE windows and <=2pp per-seed old retention guard. All36 online trajectories must replay exactly from serialized warmup. No seed/configuration selection.

With pinned versions in RUNTIME.json installed and PYTHONPATH set to their directory:

```sh
python3 launch.py verify
python3 launch.py pilot
python3 launch.py main
python3 analyze.py
```

Do not overwrite existing run receipts; reproduce into a fresh output copy. launch.py reads PROTOCOL.json and caps every child including imports. Store compact JSON/source in Git, individual compressed NPZ states in the project folder, no stream dumps or cumulative archive. Final decision and preservation are in SUMMARY.json, ARTIFACT_MANIFEST.json and TERMINAL.json.

Synthetic mechanism study only. No language-quality, novelty, scalability or deployment-speed claim. Historical missing-source caveat remains in SOURCE_PROVENANCE.json.
