# Synthetic rotation experiment: no promotion

Status: completed comparison after a preserved evaluation-harness failure and a separately preregistered evaluation-only recovery. This is not an uninterrupted protocol pass.

Authorization: the user-requested experiment-to-PR test and GitHub main `auto-research.md` at `d25da5777a2a49021588bd7ef5158e7aa400b2d5`; one local CPU thread, five CPU minutes, 50 MiB, no external data/installations/GPU/MPS/cloud/official test. GitHub main was fetched before branching; SSD mount `/dev/disk4s1` and resolved writable path were verified. The checkout was clean and no matching experiment process was running.

## Question and controls

Does oriented cross-scale conditioning improve fast-state rotation beyond an equally sized alignment-dot query? Prior energy-vs-shape evidence failed its overall gate on one seed, motivating an explanatory control rather than an energy extension. The exact recovered `HierarchicalDistributionStudent` and native autograd remain unchanged. The preserved wrapper rotates the native fast GRU proposal using segment-plus-document context; the only difference between geometry arms is normalized oriented cross versus dot query. Zero phase/gain yields exact native logits initially. Each geometry arm has 8 phase and 8 gain parameters.

Primary comparator: alignment-dot. Native hierarchy and pinned nanoGPT are separately reported. Native/geometry/control counts are 8,816/8,832/8,832; nanoGPT is 8,136 parameters, one layer, width 24, two heads, learned absolute positions, no dropout. nanoGPT revision `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, source SHA-256 `7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`. Historical hierarchy source SHA-256 `b73a7c07801f5672adb52c0cb16a3c772d62e50eebc0366491d497f76d13f73f`.

## What actually ran

All 12 seed/arm runs completed 80 AdamW updates at LR 0.003, batch four, sequence length 32: 10,240 target exposures each, 122,880 total. Three paired seeds (73, 191, 307), one LR and one final-checkpoint opportunity per arm. Fresh length-eight random motifs repeat within 33-token documents over vocabulary 16. Independent generator seeds define 320 training, 32 selection and 32 confirmation documents. CE includes all targets, including the unpredictable initial motif. Both initial and final selection CE were recorded; every arm/seed decreased. Selection did not choose checkpoints.

Hierarchy runs width 16, segment length four, document-clock period eight. States carry across adjacent 16-token chunks with full 32-token BPTT and reset only at new synthetic documents and split boundaries. nanoGPT sees the same 32-token input and targets, trained/evaluated at that length. The secondary reset-window diagnostic resets each arm after 16 tokens without changing targets. Full source, generator/data/config hashes, all 960 update metrics and all model/optimizer checkpoints are retained.

Preregistration was committed before training (`6a7baba`, finalized `1493c93`); the final protocol hash is `fd679f15252dff20501919bb4bfb164c97c31ac3c8775ff81d6f6f1eea2b48fc`. All checkpoint identities were frozen before primary confirmation began. The first run completed training but exited 1 when nanoGPT secondary reset-window evaluation passed noncontiguous sliced targets to upstream `view`. That run remains FAILED in its immutable terminal receipt. No training was repeated.

The recovery protocol was committed before execution (`7e75739`); hash `285fa3f05b16e11fd7f380397338276448ad97ed6f48ac82f9a6be9c09965cf0`. It supplied contiguous chunk inputs/targets at the adapter boundary, retained every source/checkpoint/primary criterion, and completed all 12 confirmations with exact duplicate CE and logit hashes. The four primary seed-73 results already observed in the first attempt agree exactly with recovery. This is a disclosed deviation from the original no-repairs rule: separate evaluation recovery after partial metric exposure, not a new blinded scientific replication. The failure and original runnable code remain intact. AGENT_PROMPTS.md records the adapter-layout lesson.

Runtime: macOS Python 3.9.6 / PyTorch 2.8.0, FP32 CPU, one intra-op and one inter-op thread. This is a compatibility runtime, not the recovered Linux runtime. Training/first evaluation consumed 7.937996 CPU seconds; recovery 1.151001; total 9.088997 CPU seconds. Wall times: 8.326993 and 1.361120 seconds. Evidence is approximately 1.8 MB, below 50 MiB.

## Results

Confirmation cross entropy in nats per target, lower is better; 1,024 reserved synthetic targets per arm. All seeds and control results:

| Seed | Native | Oriented | Alignment | nanoGPT | Oriented minus alignment | Oriented minus native |
|---|---:|---:|---:|---:|---:|---:|
| 73 | 2.727595448 | 2.726783395 | 2.728287667 | 2.663441926 | -0.001504272 | -0.000812054 |
| 191 | 2.687289476 | 2.687752277 | 2.690792680 | 2.707339019 | -0.003040403 | +0.000462800 |
| 307 | 2.756008893 | 2.754041553 | 2.755060375 | 2.681283355 | -0.001018822 | -0.001967341 |

The oriented-minus-alignment mean is -0.001854499 nats; descriptive paired Student-t 95% interval (df=2) [-0.004476049, +0.000767051]. It wins the alignment comparison in all three seeds but misses the predeclared 0.02-nat useful-effect threshold, the interval includes zero, and it loses to native on seed 191. The conjunctive advance gate is false. NanoGPT has lower CE than every hierarchy arm on seeds 73 and 307; native has lower CE than nanoGPT on seed 191. No winner is promoted.

Three seeds provide weak uncertainty estimation; t-interval assumptions are unverified, and all initialization seeds share fixed data. These short training budgets do not establish convergence. All arms match exposure and tuning opportunities; nanoGPT is not parameter/FLOP/walltime matched, and no modern positional baseline is included. These deliberate workflow-test limits prevent architecture-superiority, transfer, language-quality, novelty or SOTA claims.

## Deployment measurements

No deployment benchmark was performed. The original process peak RSS was 270,352,384 bytes and recovery peak RSS was 194,674,688 bytes, including Python/runtime and multiple sequential arms. These are not per-arm deployed memory measurements. Weights/states are FP32, batch four, no quantization. Activations/workspaces/allocator memory, isolated weight bytes, warmed prefill/decode latency, p50/p95 and throughput are unmeasured. CPU times above describe this small experiment only.

## Decision

No promotion: the predeclared criterion was not met. Generic extra flexibility and insufficient training remain plausible explanations. A useful next synthetic disconfirmation would keep orientation and parameter count fixed but shuffle segment/document context across batch lanes; any real context-conditioning benefit should disappear. Proposed cost: one CPU thread, at most five CPU minutes, at most 10 MiB, no paid resources, three paired seeds; requires a new frozen protocol and approval. It is not running.

The existing unapproved [language proposal](../results/new-chat-exact-source-canary-20261015-v1/LANGUAGE_PROPOSAL.json) remains a proposal; data bindings and a separately approved full-shape cost profile must be resolved before language/1B training. Nothing in this result authorizes it.

## Integrity and reproduction

From the repository root, verify preserved artifacts without running models:

```sh
python3 -B scripts/verify_evidence.py
python3 -B auto-research/results/synthetic-rotation-pr-20261015-eval-v2/verify_results.py
```

Reproduce the completed evaluation on the preserved frozen checkpoints, choosing a fresh SSD output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -B auto-research/results/synthetic-rotation-pr-20261015-eval-v2/evaluate_checkpoints.py \
  --output "/Volumes/SP AI 01_16/CodexWorkspaces/trin-codex/auto-research/results/rotation-eval-replay-CHOOSE-UNIQUE-ID"
```

The exact first execution command is retained below for provenance. It reproduces training and the known secondary diagnostic failure (exit 1); it is not the successful recovery command. Do not overwrite the original output:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -B auto-research/results/synthetic-rotation-pr-20261015-v1/run_experiment.py \
  --output "/Volumes/SP AI 01_16/CodexWorkspaces/trin-codex/auto-research/results/rotation-training-replay-CHOOSE-UNIQUE-ID"
```

- [Training protocol and source](../results/synthetic-rotation-pr-20261015-v1/)
- [Every checkpoint hash](../results/synthetic-rotation-pr-20261015-v1/execution/CHECKPOINT_FREEZE.json)
- [Original failed terminal receipt](../results/synthetic-rotation-pr-20261015-v1/execution/TERMINAL.json)
- [Recovered results](../results/synthetic-rotation-pr-20261015-eval-v2/execution/RESULT.json)
- [Completed recovery terminal receipt](../results/synthetic-rotation-pr-20261015-eval-v2/execution/TERMINAL.json)
- [Independent verification receipt](../logs/synthetic-rotation-pr-20261015-v1/verification.json)

Both registry entries link the completed recovered metrics and this report while preserving original failure status. Global manifest verification and the independent no-training verifier passed. The verifier reconstructs synthetic data, verifies source/protocol/checkpoint hashes, all 960 update rows and matched exposures, all 12 exact duplicate evaluations and the failed decision gate. All artifacts are retained. No external data, official test, installation, cloud resources, background training, email or automation were used. Original and archived source files are unchanged; language training remains paused.
