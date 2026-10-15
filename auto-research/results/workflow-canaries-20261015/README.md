# Synthetic acceptance evidence

Executed in the parent chat on CPU, one thread, three seeds (73, 191, 307). These are mechanics checks, not language benchmarks or SOTA results.

- `workflow_synthetic_canary.py`: later October 2 multi-scale reference and pinned nanoGPT. See CLI help for required roots.
- `historical_synthetic_canary.py`: compatible historical GRU hierarchy snapshot and recovered checkpoint layout. Not byte-exact deployed source.
- `conditioned_fast_rotation.py`: candidate wrapper preserving the native fast GRU transition.
- `geometry_execution_snapshot.py`: exact in-memory execution snapshot; requires `GEOMETRY_SOURCE` containing the wrapper source before execution and the recorded local package path. Not a standalone portable launcher.

All three paths passed synthetic checks. The historical loader matched 19 tensor keys/shapes and 119,694,800 parameters. Initial harness failures are retained: an identity-reference isolation bug and a checkpoint-container parsing bug were corrected without changing the original model/checkpoint or tolerances.

No task corpus or official test was evaluated. No cloud resources were created. Separate-chat creation was denied by app approval policy; this work does not fulfill that separate-chat requirement.
