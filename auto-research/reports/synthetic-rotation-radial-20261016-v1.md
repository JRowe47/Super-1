# Synthetic angular versus radial accumulator experiment

Status: completed, confirmed, **NO_PROMOTION_INCONCLUSIVE_OR_NEGATIVE**. Synthetic workflow evidence only; no language-quality, SOTA, novelty, historical-score reproduction or deployment claim.

## Authorization and preregistration

The user explicitly authorized this separate-chat end-to-end test, bounded optimization, branch push and unmerged PR against `rohanarun/Super-1` main. Envelope: one local CPU thread, at most 300 CPU seconds and 50 MiB new artifacts, synthetic generated data only, no installs, GPU/MPS/cloud/paid API, downloads of datasets or official test. Main process was restricted to 220 CPU seconds soft / 230 hard, reserving 70 seconds outside it.

Current GitHub main was fetched at `ab0fbf60bc4cc9f2b9bb397d69ea80139008b1ff`. Protocol and executable code were committed **before training** at `754aae8760321138fcf6b743d7117ffd16c7d851`. Protocol SHA-256: `407b233af92db599efaf79db0a7148703fcc120f4840fe123ae9c93c2ed8982c`. Evidence: `auto-research/results/synthetic-rotation-radial-20261016-v1/`. All source/data pins are in `SOURCE_MANIFEST.json` and `PROTOCOL.json`. The current GitHub main instructions, agent prompts, catalog, registry, scaling brief and previous negative synthetic result were read before choosing this hypothesis.

The previous motif experiment compared oriented versus alignment-conditioned rotation and was inconclusive. This experiment changes the question and data: identical alignment conditioning drives either rotation or a simpler radial scale on a cumulative modular-accumulator task. No previous run was overwritten or called a new discovery.

## Question and controls

Does angular, norm-preserving fast-state modulation improve held-out state-readout CE relative to equal-count radial modulation? Let p be the native GRU proposal, q the existing segment+document state, and a = phase + tanh(gain) tanh(dot(p,q)/sqrt(norm2(p) norm2(q)+epsilon)), per adjacent 2D pair. Candidate applies R(a)p; control applies (1+tanh(a))p. Both add 16 active parameters, start at exact native identity, and share base initialization for each seed. The control captures conditional scaling flexibility without rotation. It does not match angular/radial derivatives or FLOPs.

The immutable exact recovered `HierarchicalDistributionStudent` retains its native GRU gates, fast/segment/document clocks, accumulator, coupling order, combine MLP, normalization and tied readout. Only the separately named candidate/control fast-cell wrappers differ. Native source and pinned upstream nanoGPT source remain byte unchanged. Hierarchy/native: 8,816 parameters; rotation/radial: 8,832 each; nanoGPT: 8,136 (one layer, two heads, width24, learned positions, dropout0). NanoGPT is a scoped small fidelity baseline, not a strong positional control or language benchmark.

## What actually ran

Seeds 73,191,307; one AdamW trial per arm at LR0.003, betas0.9/0.999, eps1e-8, weight decay0.01, gradient norm clip1. Each arm/seed completed80 updates ×128 targets =10,240 target exposures. Four documents per update,32 targets per document. There are28 command and4 readout targets per document: **8,960 command and1,280 readout exposures per arm/seed**, totaling10,240. All960 optimizer updates and122,880 total target exposures are preserved. No best-checkpoint selection, tuning, adaptive stopping, retraining or recovery.

Each document begins with a random state0..7, followed by four groups of seven random modular commands8..15 and a modulo8 state readout0..7. The readouts are cumulative within each group; commands are unpredictable. Independent generator seeds train2026101601 / selection2026101602 / confirmation2026101603 produce320 /32 /32 full unique documents. Full-document disjointness and exact data regeneration pass; finite local substrings may coincide. This is not an independent-language transfer test.

All arms train CE on every next token. Primary confirmation CE uses only the128 state-readout targets; all-token CE uses1024 targets. Hierarchy state carries across adjacent16-token chunks without detaching within the32-token document, resets at true document/split boundaries, and never leaks from training to evaluation. NanoGPT sees the same32-token context. Reset-window16 is a secondary diagnostic. All nanoGPT sliced inputs and targets are made contiguous; no upstream patch was needed.

Runtime: 3.9.6 (default, Aug 25 2026, 21:26:21) , PyTorch2.8.0, macOS-27.0.1-arm64-arm-64bit, CPU FP32, one intraop and one interop thread. This macOS compatibility runtime is not the recovered historical Linux runtime. Checkpoint files include model and optimizer state, updates and exposures. All12 hashes were frozen in `CHECKPOINT_FREEZE.json` before any reserved confirmation scoring; each confirmation row binds that freeze hash. Every arm has two exact FP32 confirmations (CE, count and full logit hash). Zero-modulation native identity, direct/streamed equality, active modulation gradients and angular norm preservation checks pass.

## Results

Primary confirmation readout CE in nats/target (lower is better):

| Seed | Native | Rotation | Radial control | nanoGPT | Rotation−radial |
|---|---:|---:|---:|---:|---:|
| 73 | 3.906930447 | 3.912286192 | 3.909294099 | 2.097609997 | +0.002992094 |
| 191 | 4.037421405 | 4.042921543 | 4.040547907 | 2.109694898 | +0.002373636 |
| 307 | 3.879141450 | 3.911892533 | 3.871451885 | 2.098059624 | +0.040440649 |

Mean rotation−radial: **+0.015268793 nats**; descriptive paired Student-t95% interval (df2): **[-0.038889533, +0.069427118]**. The preregistered gate required all three gaps<0, mean≤−0.02, interval upper<0, and rotation beating native on all seeds. Rotation loses to both radial and native on all three seeds. The gain criterion fails; the interval also crosses zero. No threshold changed. Three model seeds on one fixed synthetic holdout provide weak uncertainty estimation; normality is unverified and duplicate evaluations are not independent samples.

All arm/seed diagnostics and measured training+selection process CPU:

| Seed | Arm | Initial selection readout CE | Final selection readout CE | Confirmation all-token CE | Reset16 readout CE | Train+selection CPU s |
|---|---|---:|---:|---:|---:|---:|
| 73 | native | 3.256671548 | 3.914790213 | 2.444556743 | 4.048395187 | 0.897932 |
| 73 | rotation | 3.256671548 | 3.920391381 | 2.445292711 | 4.052745998 | 0.773878 |
| 73 | radial | 3.256671548 | 3.917326659 | 2.444704235 | 4.049397111 | 0.720973 |
| 73 | pinned_nanogpt | 2.818513781 | 2.102446944 | 2.096091509 | 2.098068982 | 0.070925 |
| 191 | native | 3.264946043 | 4.024950236 | 2.456473887 | 4.107158303 | 0.450473 |
| 191 | rotation | 3.264946043 | 4.030543774 | 2.456635475 | 4.107813716 | 0.801677 |
| 191 | radial | 3.264946043 | 4.027831137 | 2.456574261 | 4.107899427 | 0.745640 |
| 191 | pinned_nanogpt | 2.821804643 | 2.102874011 | 2.103268027 | 2.108764410 | 0.065614 |
| 307 | native | 2.878403515 | 3.883271188 | 2.443274200 | 4.159393191 | 0.461572 |
| 307 | rotation | 2.878403515 | 3.916678995 | 2.438893348 | 4.208360255 | 0.773567 |
| 307 | radial | 2.878403515 | 3.875939757 | 2.443897635 | 4.147108078 | 0.731403 |
| 307 | pinned_nanogpt | 2.801630676 | 2.105717629 | 2.099233061 | 2.098791778 | 0.071905 |

The hierarchy readout losses worsened from initialization despite lower all-token training/selection losses. The objective gives seven times as much weight to unpredictable commands as to readouts; this is a preregistered limitation, not a posthoc fix. nanoGPT readout CE is near ln(8)=2.07944, the ideal uniform predictor over eight known readout classes. This does not demonstrate successful modular computation by any arm. Native hierarchy clock position is not an explicit readout-position embedding; nanoGPT has learned position embeddings. Parameter, FLOP and walltime matching across hierarchy and nanoGPT were not established; no compute-efficiency conclusion follows.

## Resources and deployment measurements

The supervised training+all evaluation process consumed **7.873963 CPU seconds** (7.420857 user + 0.453106 system), 8.449360 wall seconds. Supervisor consumed 0.038989 CPU seconds. Internal process timing reports7.700447s; external child usage includes startup/exit. Verification usage and final logical artifact bytes are recorded separately in `VERIFICATION.json` and `STORAGE.json`. Git and interactive authoring overhead are outside the measured experiment process; no claim of whole-desktop CPU accounting. A large margin remains under the300-second scientific-compute envelope.

Observed whole-process peak RSS: 269352960 bytes. This includes all arms, runtime and workspace and is not a per-arm deployment peak. Weight dtype FP32; no quantization. Per-arm deployment activations/workspace, allocator peaks, prefill/decode latency p50/p95 and throughput are unmeasured. Per-arm CPU times include selection and canaries, not just training; FLOPs are unmeasured.

## Decision and next hypothesis

Do not promote angular geometry. A next disconfirming test would freeze this architecture/control pair and use readout-only training loss on fresh documents, with a positional-feature control to separate accumulation learning from readout timing. This proposal is **not running or authorized**. Estimated small local budget: one CPU thread, under60 CPU seconds and5MiB (request a300-second/50MiB ceiling before launch). No language/1B/GPU/paid run is justified.

## Integrity, failures and cleanup

No runtime failures, protocol recovery, metric-exposed repair, automatic rerun or retraining occurred. No scientific threshold was changed. The inherited contiguous-target failure was avoided at the adapter as instructed. All negative results,12 checkpoints,960 progress rows, config/source/data/checkpoint hashes, freeze, per-row duplicate receipts, terminal and console log are preserved. Post-result reporting and hash-verification artifacts are explicitly separate from the preregistered code and protocol.

SSD was genuinely mounted as HFS disk4s1 at `/Volumes/SP AI 01_16`; resolved writable checkout `/Volumes/SP AI 01_16/CodexWorkspaces/Super-1`. Parent declared no concurrent writes; no matching run processes were found. The unrelated untracked `RELEASE_ASSET_COPY.json` remains untracked and untouched. No original repo, cloud workload, unrelated process or old evidence was modified. No corpus, teacher, official-test data, downloads, package installs, GPU/MPS or cloud resources were used. The synchronous experiment exited successfully; no training process remains from this run. Future training remains unauthorized.

## Exact commands

From `/Volumes/SP AI 01_16/CodexWorkspaces/Super-1`:

```sh
git fetch origin main
git switch -c codex/synthetic-rotation-radial-20261016 origin/main
# Protocol, generator, geometry, runner and verifier committed before execution:
git commit -m 'Preregister angular versus radial synthetic accumulator experiment'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B auto-research/results/synthetic-rotation-radial-20261016-v1/run_experiment.py --output '/Volumes/SP AI 01_16/CodexWorkspaces/Super-1/auto-research/results/synthetic-rotation-radial-20261016-v1/execution'
python3 -B auto-research/results/synthetic-rotation-radial-20261016-v1/verify_results.py
python3 -B scripts/verify_evidence.py
```

The execution command was launched by a stdlib subprocess supervisor redirecting stdout/stderr to `auto-research/logs/synthetic-rotation-radial-20261016-v1/training.log`; exact argv, exit code and child CPU are in `EXIT.json`. Do not rerun into the existing immutable execution directory. A fresh replay requires new explicit authorization, a fresh SSD output path and a new resource ledger. Verification commands do not optimize models.
