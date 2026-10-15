# Separate-chat exact-source acceptance report

Run: `new-chat-exact-source-canary-20261015-v1`. The requested separate-chat synthetic execution completed. All 12 seed/arm rows passed, with no model-check failures and no harness repairs. This result is mechanics evidence only: no language-quality, SOTA, novelty or historical benchmark-reproduction claim.

## Scope and identity

Base commit: `6f0015c7a02fffa9daa889eb7369056e3f8cf528`.
Protocol SHA-256: `d873b2feeffda7847078f99735bc6cfff8049a885fdefe845091d184ce9d05f2`.
Recovered hierarchy SHA-256: `b73a7c07801f5672adb52c0cb16a3c772d62e50eebc0366491d497f76d13f73f`.
Pinned nanoGPT SHA-256: `7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`.
See SOURCE_MANIFEST.json for every executed model/wrapper/harness hash.

Used the exact recovered `HierarchicalDistributionStudent`, preserving the learned fast/segment/document GRUs, concatenation order, combine network, layer norm, tied output embedding and state counters. Only the named cloned geometry arms wrap the native fast-cell output; the baseline and wrapper source files remain unchanged. Each geometry arm has 9,600 parameters versus native 9,584. nanoGPT has 8,496 and is explicitly not parameter-matched.

CPU only, one intra-op and one inter-op thread, float64, PyTorch 2.8.0, existing Python 3.9.6. This is a compatibility runtime, not the recovered historical cloud runtime. Synthetic shape: batch 1, vocabulary 64, width 16, 272 tokens, segment length 32, eight segments per document update. No dependency installation, datasets, official tests, GPU/MPS, cloud API/SSH or training job. Local process inventory was read only; no other workload was changed.

## Every seed and check

| Seed | Native | Oriented | Alignment-dot | nanoGPT |
|---|---|---|---|---|
| 73 | PASS | PASS | PASS | PASS |
| 191 | PASS | PASS | PASS | PASS |
| 307 | PASS | PASS | PASS | PASS |

All nine hierarchy rows pass exact identity logits/hidden/final state, exact causal prefix, full-forward versus token-step logits/hidden/final-state/counters, finite loss and every parameter gradient, nonzero gradients in each GRU scale, and a finite optimizer update. Both geometry arms additionally pass equal parameter count and nonzero gain gradients at gain 0.3. Every sequence invokes fast 272 times, segment eight times and document once. All hierarchy streaming errors are exactly zero.

All three nanoGPT rows pass exact causal prefix, finite loss and every parameter gradient, finite optimizer update, and growing-prefix inference consistency. Its source exposes no recurrent `step` or KV-cache API: prefix recomputation is explicitly used, not a claimed cached-streaming test. Maximum prefix discrepancies are 1.1102230246251565e-16 (73), 1.1102230246251565e-16 (191), and 1.6653345369377348e-16 (307), within the unchanged absolute 1e-9 tolerance, rtol=0. Geometry identity/gain and historical clock checks do not apply to nanoGPT.

Optimizer checks establish finite changes, not equivalence of an optimized backward implementation, multi-step training stability or historical optimizer replay. No mixed-precision, long-context or full-size quality claim follows.

## Execution and preservation

Wall time 3.119914 seconds; process CPU 2.524425 seconds; peak process RSS 296,845,312 bytes. The registered five-minute budget was not approached. First attempt exited 0. No reruns or repairs. All seeds, per-check metrics, synthetic losses, source hashes and attempt log are retained.


The SSD was confirmed mounted and writable, and output was resolved outside iCloud before writing. Approximately 738 MiB was free at initial inspection. Bytecode was disabled and temporary/cache destinations were redirected inside this run directory. No model checkpoints were needed or saved for this one-update mechanics test. No cleanup affected any parent-owned cloud resource. The shared instructions, protocol, registry and tracked baseline files were left unchanged; no commit or push.

## Workflow gaps retained

The instruction-linked `evidence/publication_audit.json` and recovered receipt audit directory are absent here. Of 270 publication-manifest entries, 129 paths are absent from this lightweight checkout; this is not evidence of deletion elsewhere. Full publication audit / benchmark replay therefore remains unestablished. The initial sandbox process-list command was denied; the approved read-only retry succeeded. These are workflow/inventory issues, not suppressed model failures. WORKFLOW_GAPS.json records the missing paths and an instruction correction for the parent to reconcile.

The archived energy-vs-shape and optimizer receipts were read from the existing SSD research-evidence directory, without opening corpus or checkpoint contents. Energy loses seed 301892410 (1.6235875611 versus shape 1.6214159419); its overall advance gate is false despite zero duplicate drift. Energy wins the other two paired seeds. The native-versus-analytic optimizer trajectory archive retains 49 strict failures and its failed status. This synthetic run does not supersede those failures. ARCHIVED_EVIDENCE_REVIEW.json preserves paths, hashes and exact paired values.

## Next decision: proposed only

LANGUAGE_PROPOSAL.json freezes a proposed oriented-versus-alignment contrast, equal-count controls, three paired seeds, two LRs per arm, 1,000 updates per trial, 24 trials and 12.288 million byte-target exposures total. Native hierarchy and approximately count-matched pinned nanoGPT are secondary controls. The primary advance rule requires at least 0.01 nats/byte mean improvement, every paired seed improving and a paired 95% interval below zero. Failure or uncertainty stays visible.

Proposed local rental spend is $0, with a 12 CPU-hour ceiling. A deliberately rough extrapolation from this tiny canary budgets about 9.07 CPU-hours of training and 9.97 including 10% evaluation/preservation; it is not measured language throughput or a guaranteed bound. A separately approved 20-update full-shape profile must precede execution. Electricity and opportunity cost are unpriced. Data version, document splits, hashes and contamination audit remain unresolved and block launch. The design proposal is recorded, but is not a complete launch-ready preregistration or approved resource budget.

At approximately 1B parameters and illustrative 20B tokens, the dense 6ND proxy is approximately 1.20e20 FLOPs. At hypothetical achieved 50/100/200 TFLOP/s this is approximately 666/333/166 device-hours per arm/seed, before overhead; it is not a measured cost for the recurrence. Actual money equals measured device-hours times a current approved quote plus storage/evaluation/checkpoint overhead. No provider quote or cloud access occurred. A credible dollar budget and 1B launch remain blocked on measured scaling, strong baselines, data and explicit resource approval.

The parent owns registry/report-index reconciliation. It should record this run's synthetic PASS while retaining the standalone documentation gaps and the unlaunched, data-unbound language proposal. No new authorization is being inferred.
