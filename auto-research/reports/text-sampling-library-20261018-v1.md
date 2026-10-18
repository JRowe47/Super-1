# Text sampling library and continuation

A versioned library now stores learned token-conditioned sampling functions for an exact use case, backbone, tokenizer and context. Nine entries preserve three retained references and six unpromoted candidates. Reloaded references reproduce their logits exactly; incompatible or corrupt entries are rejected.

Both continuation arms start at the previous shared sampler, freeze all backbone weights, and receive the same 1,000 text-tuning updates per seed. Four rounds of 250 updates compare Adam learning rates 0.003 and 0.0003. Each arm retains its best validation round including round zero. One primary arm is selected by mean validation loss before fresh confirmation.

| Arm | Seed 73 | Seed 191 | Seed 307 | Mean CE (nats/byte) |
|---|---:|---:|---:|---:|
| incumbent | 2.184679151 | 2.186048590 | 2.202432867 | 2.191053536 |
| continued | 2.179671090 | 2.186172958 | 2.200430825 | 2.188758291 |
| small_step | 2.183185101 | 2.186414387 | 2.201727483 | 2.190442324 |

Validation selects **continued**. Its mean paired change is -0.002295244 nats/byte; paired 95% interval [-0.00870120590982329, 0.004110716939543597]. One seed regresses. The preregistered 0.02-nat useful-effect gate fails; **no replacement is promoted**.

The confirmation region is [1037134, 1053774), using 256 nonoverlapping 65-byte chunks from Tiny Shakespeare. Tuning and validation regions are reused research data; this confirmation region was not scored in either earlier pilot. Scores from those earlier regions are not directly comparable.

All nine selected endpoints independently replay exactly. All 24 round checkpoints, optimizer/scheduler states, gradients, matched minibatch digests and original fallback checkpoints are preserved. Thirteen library checks passed. Training took 70.37 CPU seconds and 70.97 wall seconds, one CPU thread.

The pre-run cross-length bitwise comparison failed with a 5.7220458984375e-06 maximum FP32 difference. This failure remains in CANARY.json. Changing future tokens at the same tensor shape leaves earlier logits exactly unchanged. No tolerance was relaxed, no weights were updated to mask the failure, and cross-length bitwise invariance is not claimed.

Each model still executes four blocks. The first and last coordinates are fixed at 0 and 1; two interior coordinates are learned per token. The library stores the learned function, with observed coordinate distributions in POSITIONS.json. Layer-count adaptation is a separate subsequent experiment. No broad language-quality, nanoGPT superiority, universal optimality or speed claim follows.

Replay: `python3 -B auto-research/results/text-sampling-library-20261018-v1/check.py` (write-once verification receipt; use a fresh location or inspect the existing receipt on repeated execution). Library checks likewise preserve their first receipt.

Library API: [sampling_library](../../sampling_library/README.md).
