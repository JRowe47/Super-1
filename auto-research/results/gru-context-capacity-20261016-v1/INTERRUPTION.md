# Context/capacity sweep — execution connection interrupted

Status: **INCOMPLETE; remaining trial/process status unverified.**
Study: `gru-context-capacity-20261016-v1`.
Parent experiment/code commit: `3d886ed13c82ef914d07ce7a3f0b455df202aff7`.
This is a recovery receipt, not a completed benchmark report.

The user authorized the next rapid research phase, continuing the existing
CPU-only exploration and publication to JRowe47/Super-1. No GPU was provisioned,
no new paid resources were requested, and no official test was accessed.

## Declared screen

Ten fresh trials, each with a 90-second training-wall allowance, four CPU
threads, seed 101605 and common data-order seed 101614:

| Model | Width | Contexts | Slow-state chunk |
|---|---:|---|---:|
| Linear minGRU | 160 | 512, 1024, 2048 | 32 |
| Linear minGRU | 224 | 512, 2048 | 32 |
| Linear minGRU | 160 | 2048 | 64 |
| Ordinary GRU | 147 | 2048 | 32 |
| nanoGPT | 108 | 512, 1024, 2048 | n/a |

Counts: minGRU160 996,320 parameters; minGRU224 1,767,584; ordinary GRU
995,925; nanoGPT 1,008,072, including its fixed 2,048-position table.

All lengths tile the same ordered 2,048-target training blocks. Recurrent
models use four blocks (8,192 targets) per update at peak LR 0.008; nanoGPT
uses one block (2,048 targets) at peak LR 0.002. Full BPTT, reset examples,
AdamW, 5% wall-time warmup followed by cosine decay to 10% of peak.
Evaluation scores identical final-512 targets of 16 validation windows.
Monitoring uses four separate windows. The validation split is reused,
not a pristine test set. Training uses the existing preserved FineWeb-Edu
sample and 2,048-BPE tokenizer.

The wider model changes parameters and compute as well as persistent state.
Chunk64 changes feedback timing and document-state update period as well as
parallel scan length. Neither is an isolated state-capacity intervention.

## What was observed before access failed

The ten preflight configurations passed finite forward/backward/AdamW checks.
Training-target identity across lengths and validation-tail identity passed.
For every 2K recurrent configuration, a split at token777 preserved outputs
within 2.4e-6 and state within 1.2e-7, using trained canary weights that were
discarded before the actual runs.

The console confirmed completion of the first trial:

- Recipe: `linear_w160_c1024_s32`.
- Final common-tail validation CE: **5.068231076002121**.
- 281 updates; 2,301,952 target exposures.
- Training wall: 89.40397835103431 seconds.
- Aggregate training CPU: 336.69062161399995 seconds.
- Swapping the prefix older than 512 tokens worsened CE by
  **0.005067706108093262**.
- Final monitoring CE: 5.155896186828613 on the separate monitoring windows.

These values are from observed execution output. The runner wrote its
checkpoint and RESULT.json before printing COMPLETE. They have **not yet
received the post-run checkpoint replay** and are not a completed comparison.
The prefix-swap probe changes the input distribution; it is not proof of
long-range retrieval.

Subsequent shell commands failed with an exec-server transport disconnection.
Both process-status access and file-edit access failed to reconnect.
The launcher might have progressed further; do not infer that other trials
completed or failed, or that the launcher stopped. No trial was restarted.
This receipt is the only new file published in this interruption commit;
the local source, frozen protocol and raw outputs still require retrieval.

## Recovery

Local working directory:
`/workspace/scratch/eaefc77ff824/research/fork_work/gru-extended-20261016-v1`.

Local study directory:
`auto-research/results/gru-context-capacity-20261016-v1/`.

Local interpreter:
`/workspace/scratch/eaefc77ff824/.venv/bin/python`.

Local data:
`/workspace/scratch/eaefc77ff824/research/experiments/compact_state_v1/data`.

Local files created before the disconnection: experiment.py, prepare.py,
launch.py, analyze.py, PROTOCOL.json, REGISTRATION.json, PREFLIGHT.json,
and the execution/log directories. A later IMPLEMENTATION_NOTES.md write
had an uncertain outcome; check before editing.

1. Reconnect and inspect live processes and all RESULT/FAILURE/checkpoint
   receipts before launching anything. Do not duplicate an active launcher.
2. Preserve the existing protocol and completed/failed trials. The launcher
   skips terminal receipts and the worker can resume its own unfinished
   checkpoint with model, AdamW, RNG and training-clock state.
3. If the source files and protocol are recovered and no launcher is active,
   resume launch.py with the existing data path. Do not rerun prepare.py
   against the frozen protocol.
4. Once all declared trials have terminal receipts, run analyze.py with the
   same data path. Inspect the complete results, prefix diagnostics, charts
   and exact checkpoint replays, including every failed trial.
5. Preserve the checkpoints, publish source and raw results, update the
   registry/manifest and this existing draft PR. Do not submit upstream.

## Optimizer-description issue noticed during this phase

The inherited frozen optimizer helper exempts any nn.Embedding with
num_embeddings == 2048. NanoGPT's new 2,048-position table therefore receives
zero weight decay along with its token table. The inherited protocol prose
mentions token-table/one-dimensional exclusions but omits that position-table
effect. All three current nanoGPT context arms share this actual policy.
The prior 512-position control had decayed positions.

Disclose this description mismatch in the eventual report without rewriting
the frozen source/protocol or silently rerunning trials. In a future study,
identify embedding roles by parameter identity and explicitly choose the
position-table decay policy. No model ranking or promotion follows from
this interrupted screen.
