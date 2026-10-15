# Prioritized frontier experiments

**Recovery update 2026-10-15:** Read [RESEARCH_CATALOG.md](RESEARCH_CATALOG.md) first. E00 has recovered the original nanoGPT package/results and exact external corpus/teacher. The E03 dedicated odd-query control already exists and failed the overall energy advance gate; do not rerun it unchanged or claim it is missing. The next E03 hypothesis must explain and control the seed-dependent result. Check archived transfer and optimizer-fidelity failures before requesting scale-up.

All entries are **draft proposals, not authorized runs**. Fill/freeze a protocol and approve resource budgets before canaries, profiling, evaluation or training. Use bounded pilots before requesting scale-up. Costs and exact budgets must be measured/proposed, not invented.

## E00 — Recover provenance and establish the original reference

**Question:** Can the historical 120M comparison and the exact original recurrence be reconstructed faithfully?

Actions: locate original source/configs, full upstream nanoGPT revision, tokenizer, split/token hashes, teacher settings, selected checkpoints, evaluation commands and task harness versions. Do not mistake empty preserved directories for complete recovery. Map each headline to a source artifact. Verify the newer distributed hashes without executing models.

Deliverables: provenance table, original-design specification with normalization/coupling order, split chronology, missing-artifact list and replay feasibility. The original paper PDF remains report-backed unless raw evidence is recovered.

Gate: unresolved historical artifacts do not block a clearly labeled fresh experiment, but they do block claiming an exact historical replication. Never substitute the current small pilot for the 120M design.

## E01 — Preserve the recurrence while removing execution overhead

**Hypothesis:** Most long-context cost is implementation overhead, not required mathematical work.

Arms: exact eager reference; grouped input projections/compiled unchanged recurrence; then an explicitly justified fused-kernel or block implementation. No approximate scan arm is labeled equivalent.

Canary matrix: multiple seeds; 128/512/1024/2048-token shapes; batch-1 and intended microbatches; zero/nonzero state; nonzero cross-scale coupling and gains; chunked vs uninterrupted stream; FP32 reference and intended fast precision. Compare outputs, all state, gradients and optimizer updates. Declare tolerances and supported shapes first. Verify causality and document reset semantics.

Measurements: wall time per forward/backward/update, peak memory, compilation overhead and projected approved-run cost on actual intended hardware. No long full-model run before this canary.

Disconfirmation: failed semantic/gradient/optimizer parity, or no useful amortized speedup. Retain reference; do not relabel a changed model as an optimization.

## E02 — Decisive fair-context hierarchy versus Transformer test

**Primary question:** Does the exact original hierarchy retain a useful quality/state advantage when the baseline is trained at the tested context rather than extrapolating?

Core arms: original learned-gate hierarchy; pinned upstream nanoGPT; strong pinned RoPE Transformer. First recover/pin the nanoGPT implementation and reproduce its training-context behavior. Do not assert the archived short revision specifies a complete runnable baseline.

Primary matrix: train at 512 and evaluate at 512; train at 1024 and evaluate at 1024; train at 2048 and evaluate at 2048. Each is its own matched comparison. Secondary extrapolation matrix: train at 512 and evaluate at 1024/2048, explicitly separated from primary results. A 2K positional embedding table with untrained extra positions is not a fair 2K-trained baseline.

At each context re-resolve Transformer width/depth/head constraints and profile compute. Conduct separately labeled parameter-matched and FLOP-matched tracks when both cannot be matched simultaneously. Preserve loss/tokenization/teacher/exposure/update-token and tuning budgets. At least three paired seeds; confirmation sample size chosen before results. Use contiguous recurrent lanes; include reset-state versus carry-state ablation without giving the carry arm information unavailable within the baseline's declared context.

Do not conflate BPTT length, accessible history and inference state size. If hierarchy carries history beyond the Transformer window, label that experiment as a deployment tradeoff and run an equal-history comparison separately.

Stages: approved small faithful pilot; affordable intermediate model; proposed historical ~120M scale only after E01 and pilot gates. A 10M-token ~120M run reproduces a short-budget regime, not evidence of convergence. Plot learning curves at matched exposure and compute and propose longer budgets before claiming asymptotic advantage.

Endpoints: reserved held-out CE; task accuracy with frozen harness; streaming total-memory/latency. Show nanoGPT's wins as well as losses.

Advance: preregistered useful quality or deployment-Pareto gain survives fair positional controls and seed uncertainty. Otherwise identify which confound explains the historical reversal.

## E03 — Isolate pair-energy from added flexibility

**Question:** Is relative energy specifically useful, rather than just extra learnable nonlinear query parameters?

Arms: overlapping rotation baseline; pair-energy; equal-parameter relative-phase key; equal-parameter non-energy nonlinear query-shape control; frozen/disabled-energy control; competent parameter-matched Transformer. Match tuning opportunities. Pin exact equations and parameter counts; do not call unequal hidden capacity matched.

First use the established byte task with a fresh preregistered confirmation region, acknowledging research-wide exposure history. Then test a separately approved independent corpus and a word/token-level model. Do not use byte CE to satisfy word-PPL gates.

Diagnostics: energy coefficient learning, gradient/use sensitivity, state/drive norm ranges, angle saturation, state-dependent Jacobian behavior. Rotation norm preservation does not imply a globally orthogonal Jacobian or stable full recurrence. Diagnostic interventions must not replace the frozen primary evaluation.

Advance only if the mechanism beats the dedicated shape control on the declared paired criterion; otherwise describe it as generic extra flexibility. The incomplete word sweep already motivates this control; do not relaunch it unchanged.

## E04 — Original-hierarchy transport × learned readout factorial

**Question:** Can cross-scale transport help because the output actually observes differences between scales?

Keep the exact original hierarchy. Four arms: original mean readout; learned scale readout only; transport with mean readout; transport plus learned readout. Preserve shared initial base parameters, identity transport and mean-equivalent readout initialization. Report added parameters and use a justified matched-capacity control.

Critical distinction: mean-preserving scale transport can be immediately invisible to mean readout. Test that mathematically and in synthetic parity checks first. It may still affect later recurrent dynamics; do not infer no temporal effect from same-step invariance.

Measure each factorial main effect and interaction on the same stream/seeds/selection protocol, plus gate/transport/readout gradients. Use nonzero gains and perturbed readout tests to expose disconnected branches. Confirm quality effects, not just nonzero gradients.

Advance only if transport plus readout beats readout-only and original by the preregistered useful effect with uncertainty. Otherwise retain the simpler model. Do not mix this factorial with E03 before isolation.

## E05 — Scale and measure a real inference frontier

**Question:** Does a confirmed mechanism deliver useful quality per deployed byte at acceptable latency across scales/tasks?

Select the best confirmed contender by frozen selection criteria, not test outcomes. Scale from pilot to intermediate to historical ~120M only after budget approval. Proposed exact dimensions/token budgets remain unresolved until measured hardware costs and data availability are reviewed.

Controls: competent dense Transformer, positional baseline, and a relevant modern recurrent/state-space baseline with pinned implementation and comparable tuning. Consult current primary sources before choosing the latter; do not assume a named old baseline is the frontier.

Deployment matrix: train/eval matched contexts from E02; batch-1 and multi-sequence serving; FP32/BF16 and, separately, equally tuned quantization; measured prefill/decode latency and throughput. Show resident parameter bytes, sequence state, activations/workspace and observed device/process peaks separately. Treat compilation/startup and OOMs honestly.

Ablate where the gain comes from: recurrence/hierarchy, function-generated transport, parameter count, distillation, data, history budget and optimization. Change one factor at a time. Confirm on at least one independent task/corpus before a generalization claim.

Output: all-seed quality-vs-total-memory and quality-vs-latency Pareto plots, plus quality-vs-training-compute. A 14,003x state ratio alone is not the success criterion.

## E06 — Freeze, compare to current published targets, then test once

Prerequisites: completed validated run, architecture/checkpoint/tokenizer freeze, duplicate canonical FP32 confirmation, relevant word-level validation eligibility gate, valid primary-source apples-to-apples target comparison and explicit test authority.

Static and dynamic evaluation are separate protocols. Never count a cached/adaptive test score as static. Respect data provenance, training exposure, tuning and test-time compute.

Only after the sealed protocol authorizes it: exactly one official test. No retuning/retry based on test results. Preserve final report, hashes, charts, target comparison and zero matching cloud-resource cleanup receipt. If any prerequisite is absent, stop at validation and say why.

## How to choose among proposals

Before requesting a run, write: (1) the uncertainty it resolves; (2) the strongest competing explanation; (3) the cheapest decisive control; (4) falsifying outcome; (5) estimated compute, money and local CPU cost; (6) evidence that would justify the next stage.

Prefer a smaller experiment that rules out an explanation to a larger sweep with the same confound. Do not launch all proposals merely because this queue exists.
