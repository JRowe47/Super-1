# Iterative use-case sampling optimization

This experiment implements the user's requested benchmark / gradient-descent / validation loop. It does not add recursive execution to inference. The shared model weights are frozen before adapting the sampler.

Workloads are next-byte text continuation, synthetic dictionary retrieval, and synthetic modular arithmetic. Policies include frozen uniform sampling, one shared token-conditioned sampler, per-workload global positions, and per-workload token-conditioned positions. Workload identity is supplied explicitly; trained policies determine positions without hand-written token-category rules.

Each policy receives three gradient-descent rounds. Selection chooses the best of the initial state and all three rounds; confirmation is never used for tuning. Every rejected checkpoint, optimizer state, gradient statistic and baseline failure is preserved.

- [Frozen protocol](PROTOCOL.json)
- [Configuration](CONFIG.json)
- [Full report](../../reports/usecase-sampling-descent-20261018-v1.md)
- [Raw results](execution/RESULT.json)
- [Round histories](execution/SAMPLING_HISTORY.json)
- [Selected policies and checkpoint hashes](execution/FREEZE.json)

Run `python3 -B auto-research/results/usecase-sampling-descent-20261018-v1/verify.py` from the repository root to independently replay the frozen confirmation results without training.

Training replay uses `run.py --output` and requires an authorized budget, an unused directory on the mounted SSD, and SSD-directed TMPDIR. The resource monitor checks the actual replay destination. Existing results refuse overwrite.

The baseline capability gate is separate from sampler improvement. A backbone that has not learned retrieval/arithmetic cannot support a claim that minor loss changes demonstrate those capabilities. Extra per-workload router storage is reported; active depth remains four blocks. This study does not make a new nanoGPT performance claim.
