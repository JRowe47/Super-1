# Fixed-storage replay retention experiment

Six arms cross exact backprop, frozen learned feedback and adaptive feedback with FIFO and mixed replay. One fixed configuration, three fresh main seeds and a mechanical pilot. This is a synthetic diagnostic, not language-model training.

The complete source/protocol/registration are committed and read back before numerical verification, pilot or main. `feedback_core.py` and reference verification files are byte-preserved from the previous registered run. Original numerical tolerances and the primary adaptation/retention gate are unchanged.

Both policies store 512 examples and train on 16 current plus16 replay examples. FIFO samples16 from a queue512. Mixed samples8 from a recent queue256 and8 from a fixed old bank256. Both start with the same512 examples. The old bank retains its original labels, including stale labels after a rule changes. No source or phase metadata enters the learner.

Use the exact Python/package versions in RUNTIME.json. From a fresh output directory containing the registered source files and REGISTRATION.json:

```sh
python3 launch.py verify
python3 launch.py pilot
python3 launch.py main
python3 analyze.py
```

The launcher refuses existing receipts and each worker refuses an existing seed directory. Do not overwrite completed evidence. Single-thread CPU workers; at most3 concurrently; hard child CPU caps total830 seconds plus limited termination grace within the900-second aggregate experiment ceiling. The pilot is a mechanics check and cannot be used for tuning.

Each seed's RESULT.json contains metric traces, canonical data hashes and full deterministic online replay receipts. STATES.npz contains the common warmup W/F and all final W; only adaptive arms need separate final F. Frozen/BP final feedback is the unchanged warm bank. For inspection reconstruct each final state as `{**warm, **stored_arm}`. Phase/final hashes cover the complete state. Files use NumPy NPZ without pickle. Data are regenerated from registered seeds, not archived.

The primary comparison is adaptive_mixed versus frozen_mixed: mean paired post-change CE reduction>=5%, no negative seed, and no more than2pp old-window or final-old accuracy loss in any seed. FIFO and within-policy replay effects are descriptive controls. A gate pass remains synthetic evidence; it does not authorize architecture promotion or larger training.
