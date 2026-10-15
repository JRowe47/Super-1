# Roadmap to a defensible 1B-parameter frontier result

Updated 2026-10-15. The target is **approximately 1B stored/trainable parameters**, not 32B. This is a staged experimental plan, not a guarantee of SOTA or authority for paid compute.

## What a 1B success must establish

Choose and freeze the claim before training: (a) better held-out language quality at matched deployed bytes and acceptable latency, or (b) comparable quality at materially less measured memory/latency. Compare against competent ~1B Transformers and current recurrent models under an explicit training-data/compute/tuning contract. A local nanoGPT win alone is not SOTA. Distillation, scratch pretraining and test-time adaptation are different tracks.

The historical 120M experiment is useful motivation, not a scaling law. Its 10M tokens are a short-budget regime; its longer-context advantage is confounded by 512-token training. The tiny byte pilots do not demonstrate 1B behavior. The energy-vs-shape advance gate failed one seed and the optimized backward failed strict optimizer fidelity: neither may be silently promoted.

## Stage 0 — Make reproduction executable

1. Verify repository and private release manifests, checkpoint digests, corpus/teacher hashes and pinned dataset revisions.
2. Use the recovered byte-exact hierarchy core and historical nanoGPT final checkpoint linked from results/cloud-recovery-20261015/. Their hashes now match the historical records. Keep the old compatible snapshot distinct and unchanged. Do not keep searching for these same artifacts as though they remain unlocated.
3. The installed Linux environment is recovered and passed a CPU arithmetic/import check, including an extracted-archive check using Debian 12 host libraries. A complete independent container/GPU replay and proof that every runtime file is unchanged since August remain unestablished. Pin packages and compiler/device/container identity for any new runtime and record drift explicitly.
4. The registered separate-chat exact-source self-test passed all 12 arm/seed rows: shapes, causality, state carry, gradients, optimizer updates and all three clocks. See results/new-chat-exact-source-canary-20261015-v1/. It used macOS PyTorch 2.8.0, not the recovered cloud environment. This closes the bounded workflow-canary step, not historical quality reproduction or execution-performance parity.
5. For a faithful small language pilot, preregister train/selection/confirmation partitions, same tokenizer, explicit objective and paired sampler. Distilled historical replay and new scratch-CE comparison must have separate run IDs.

Gate: another agent can follow the reproduction instructions without hidden paths, missing imports, mutable seals or unexplained source substitutions.

## Stage 1 — Optimize the unchanged recurrence

Group input projections and compile the same recurrent equations before attempting custom kernels. A nonlinear state-dependent update cannot be declared an associative scan without proof.

Test nonzero coupling/gains and saturated regimes, not only identity initialization. Compare FP32 outputs, intermediate/final state, gradients and optimizer steps, then document mixed-precision error. A fused backend is accepted only within preregistered tolerances; matching logits alone is insufficient.

Measure actual intended hardware before full training: tokens/s, time/update, peak allocated/reserved/device memory, compile overhead and checkpoint time. Verify deterministic restart and failed-run preservation. Don't extrapolate the tiny CPU pilot's speed to a 1B GPU job.

## Stage 2 — Isolate new geometries cheaply

Keep the original architecture as the immutable reference. Propose at most one primary geometric change per experiment; the model should choose it from evidence and competing explanations, not an unconditional list of favorite mechanisms.

Candidate questions:
- **Transport × readout factorial:** original mean, learned-readout only, cross-scale transport only, both. Test whether benefit survives the readout-only control. Mean preservation can hide same-step effects; later dynamics still matter.
- **Conditioned transport geometry:** compare oriented angle-only, relative-energy angle, and equal-parameter non-energy nonlinear query shape. The prior shape experiment is completed with a failed overall energy gate; inspect it before designing a different test.
- **Timescale/coupling learnability:** learned original gates vs controlled fixed timescales, local-only vs cross-scale coupling, with equal capacity/tuning. Do not replace the reference architecture under an optimization label.
- **Residual/depth allocation:** only as an explicitly new architecture arm, test identity-safe residual refinements and input reinjection with fixed compute, versus equal-compute wider/deeper controls. Recurrence over depth is not the same as recurrence over time.

Primary report: complete per-seed reserved CE, useful effect, uncertainty, native optimizer fidelity and cost. Separate hypotheses. At least three paired seeds for confirmation. A synthetic canary is implementation evidence, never a language win.

## Stage 3 — Fit scaling behavior before committing to 1B

Use a resource-approved ladder, e.g. faithful ~120M -> intermediate ~300M -> ~1B, with independently resolved dimensions. These are planning scales, not hard-coded architecture choices.

Resolve the embedding/core allocation before choosing those dimensions. For the recovered native HierarchicalDistributionStudent only, P(V,d)=V*d+32*d^2+23*d: vocabulary 151936 and width 688 give 119694800 parameters, about 87.3% in the embedding. Width 3696 gives 998773776 parameters, with about 56.2% in the embedding. These are analytic counts, not trained scaling evidence, and are not the parameter formula for every function-weight arm. Widening, stacking, vocabulary changes and function-generated depth must be separately declared ablations.

Run multiple exposure checkpoints at each scale, with equal LR/schedule tuning opportunities and fresh reserved confirmation. Estimate quality-versus-training-compute and uncertainty; test whether a pilot gain shrinks or reverses. Compare train/eval-matched 512/1024/2048 contexts and separately label length extrapolation.

For scratch pretraining, examine a *range* of token budgets informed by measured loss curves and available clean data. As an illustrative planning anchor only, 20 tokens/parameter gives 20B tokens for 1B parameters, not a promise of optimum, competitiveness or sufficiency. Chinchilla studied Transformers; do not transplant its optimum to a novel recurrence without fitting.

Compute estimate: a dense Transformer rough proxy is C ≈ 6ND; at N=1e9 and D=20e9 this is 1.2e20 FLOPs. This is a heuristic, **not the actual recurrent/teacher compute**. Measure operator FLOPs, recurrent iterations, optimizer and teacher overhead separately. For DDP/FSDP/tensor parallel options estimate sustained achieved throughput, communication, checkpointing and failure recovery, then price the approved hardware. No made-up GPU count or dollar figure.

Memory lower-bound planning at 1B: BF16 weights alone ~2GB (decimal); Adam moments/gradients/master weights may add many GB depending on implementation. Activations, sequence state, workspace and communication are additional. Quantization and parameter sharding are explicit variants, not invisible fairness changes.

Gate: complete intermediate-scale evidence, stable execution, preserved checkpoints, feasible measured cost and a predeclared expected useful effect before requesting the main 1B budget.

## Stage 4 — Proper 1B comparison

Train the selected original-hierarchy-derived candidate and tuned pinned baselines on the same licensed/deduplicated corpus and tokenizer. Use a strong positional Transformer, the vendored nanoGPT fidelity baseline, and a relevant current recurrent/state-space baseline. Mamba-3 is a primary-source comparator candidate, not an automatically fair or globally strongest baseline. Match parameter and FLOP/deployed-byte tracks separately when they cannot all coincide.

Keep independent corpora/tasks and research-wide contamination chronology. Disclose teacher, teacher tokens/FLOPs and any pretrained embeddings. Scratch-pretrained contender vs distilled baseline is not an architecture-only comparison.

Evaluate held-out CE, frozen downstream harness, long-context retrieval/state tracking, streaming state carry/reset, and measured deployment Pareto curves. Include batch-1 and multi-sequence service; report prefill/decode p50/p95, throughput, total memory and persistent state separately. Compare GQA/MQA/quantized/windowed baselines explicitly rather than projecting the old 14,003x ratio onto all Transformers.

## Stage 5 — Validate a frontier claim

Freeze primary benchmark, score direction, model/active parameter count, data exposure, tuning budget and training/test-time compute before the main result. Verify current primary-source published baselines at the time of claim. Use all seeds, null results and uncertainty; register failed runs.

A 1B model can lead a **defined size/compute/memory regime** without being the world's best LLM. Do not blur those claims. Novelty needs an exact mechanism/prior-art comparison independent of a score win.

Official Salesforce WT2 test restrictions remain in force. Its historic static validation PPL <61.93 / dynamic <40.27 gates are eligibility, not current SOTA thresholds or broad 1B performance targets. No official test until a frozen eligible protocol permits the one-shot test.

## Execution authority for the new-chat self-test

The user's request authorizes a separate chat to follow this guide and test the workflow. Start with bounded local synthetic import/forward/backward/optimizer canaries, one CPU thread, no cloud or long training. Verify SSD mount/writability first. If its environment cannot write the SSD, stop execution and report that precise blocker rather than writing to Documents or silently changing design.

The completed separate-chat test produced LANGUAGE_PROPOSAL.json with a rough cost envelope, but its data hashes/bindings and execution approval remain unresolved. A separately approved full-shape profile must precede training. Do not infer an unlimited sweep, twelve-hour CPU authorization, or a main 1B training budget. Store results/receipts under auto-research/ and preserve failure output. Repair demonstrated harness defects only within the approved scope; never change the reference model or relax tolerances to turn a real failure into a pass. A new run needs its own frozen protocol and run ID.

## Primary-source anchors (checked 2026-10-15)

- [Hoffmann et al., Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556): train-data/model-size scaling evidence for Transformers; use as prior, not proof for this recurrence.
- [Lahoti et al., Mamba-3](https://arxiv.org/abs/2603.15569): relevant inference-first recurrent quality/efficiency comparisons and state tracking; motivates a stronger baseline.
- [Geiping et al., Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171): depth-recurrent comparison context; does not justify substituting depth recurrence for original time hierarchy.
