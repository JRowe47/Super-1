# Bounded feedback experiment

Run: `bounded-feedback-20261018-v1`. Execution completed; retention criterion failed; original source and checkpoint export pending.

The user authorized this continuation with "Cool - good results. Go ahead and continue." A later instruction authorized compact repository handoffs and replaced further cumulative archive packaging. This report records results inspected during execution; it is not a byte-exact export of the original evidence.

## Question and controls

Can bounded local feedback continue adapting after exact-backpropagation warmup, improve learning after changes, and retain old knowledge? The primary comparison is feedback bounded throughout warmup and online learning versus identically warmed feedback frozen at handover. Exact backpropagation, original and bounded fixed random feedback, unbounded learned feedback, and bounds beginning only at handover are controls.

## What actually ran

A bias-free 11 -> 32 ReLU -> 32 ReLU -> 4 network has 1,504 forward and 12,544 feedback parameters. Each matrix retains all orthonormal real Fourier coefficients. All arms use the same 4,096-step BP warmup and online data schedule. Online learning has four 512-step blocks: old stationary knowledge, novel contexts, changed rules and mixed data. Each step uses 16 current and 16 FIFO replay examples; the buffer holds 512 examples. All learning rates are 0.05. The B/V/U spectral norm cap is fixed at 2, implemented by uniform matrix rescaling using full SVD values.

Main seeds: 601771, 601772 and 601773. Pilot seed: 95967, full warmup and four 64-step online blocks. The source/configuration description is in [PROTOCOL_RECORD.json](../results/bounded-feedback-20261018-v1/PROTOCOL_RECORD.json). Original frozen registration time: 2026-10-18T03:50:18.009520+00:00. Original source/config/data/checkpoint hashes and runtime versions remain unavailable for this export; they must be recovered rather than reconstructed.

Verification, pilot and main used 242.514369 recorded child CPU seconds in total. Main wall time was 74.3856 seconds, with at most three single-thread CPU workers. No new training ran during this repository export.

## Results

The primary metric averages novel-context CE at steps 544..768 and changed-rule CE at steps 1056..1280, every 32 steps. Relative reductions are calculated within seed and then averaged.

| Seed | Relative CE reduction | Old accuracy delta in primary window | Final old accuracy delta |
| --- | ---: | ---: | ---: |
| 601771 | 26.53% | -9.80 pp | -6.25 pp |
| 601772 | 29.65% | -8.01 pp | -5.02 pp |
| 601773 | 28.77% | -9.20 pp | -2.34 pp |
| Mean | 28.31% | -9.00 pp | -4.54 pp |

The >=5% effect threshold and nonnegative-seed condition passed. The allowed old-accuracy loss was <=2 percentage points during the primary window and at final, in every seed. Both retention checks failed in every seed.

Final accuracy percentages, means over all three seeds:

| Method | Old | Novel | Changed | Balanced |
| --- | ---: | ---: | ---: | ---: |
| Backpropagation | 97.340 | 95.443 | 83.984 | 92.256 |
| Fixed random | 94.215 | 91.602 | 70.573 | 85.463 |
| Bounded fixed random | 90.327 | 77.702 | 39.062 | 69.031 |
| Bound from handover, adaptive | 91.183 | 86.686 | 61.979 | 79.949 |
| Bound from handover, frozen | 95.219 | 83.366 | 46.875 | 75.153 |
| Bound throughout, adaptive | 90.365 | 85.840 | 69.401 | 81.868 |
| Bound throughout, frozen | 94.903 | 81.673 | 41.536 | 72.704 |

All 12 bounded learned-feedback main runs completed with finite values. Five of six unbounded learned-feedback main runs became nonfinite: adaptive at steps 3/3/4, frozen at steps 4/4 in the first two seeds. The third unbounded frozen run completed with catastrophic CE (primary 7.33e30; final old 2.10e33). It is not counted as a useful stable result. No survivor-only average is reported.

All seven methods in the final-accuracy table completed across all three seeds. Complete recorded paired values and failure rows are in [RECORDED_SUMMARY.json](../results/bounded-feedback-20261018-v1/RECORDED_SUMMARY.json). Three seeds support the observed paired result, not a broad architecture superiority or significance claim.

## Update and deployment measurements

Sampled global update cosine with exact BP was approximately 0.784 for the primary adaptive arm and 0.799 for frozen; update-norm ratios were approximately 1.047 and 0.809. Different trajectories and the exact readout contribution prevent a causal directional interpretation. Larger updates remain a competing explanation.

Mean policy CPU was approximately 5.45 seconds for bounded adaptive feedback versus 3.12 seconds for BP in this setup. The SVD cap is a global matrix operation. No compute advantage is established. Total deployment memory, prefill/decode latency and language-model quality were not measured.

## Decision

**Combined criterion failed; no promotion.** Continued bounded feedback plasticity improves post-change adaptation relative to its frozen control, with excessive forgetting. Exact BP and original fixed random feedback still beat it on balanced accuracy.

Next: isolate magnitude using a magnitude-matched frozen control, then test a separate replay-based retention intervention. Exact hidden gradients remain diagnostic. The 242.5-CPU-second current run is a cost reference; freeze the follow-up design and bounded resource accounting before execution. Neither proposed follow-up is running.

## Integrity and preservation

Recorded numerical checks passed original FP32 2e-6 and FP64 2e-12 tolerances (observed maxima about 3.58e-7 and 5.00e-16). All 27 main and nine pilot trajectories, including failure prefixes, replayed exactly. All launchers exited zero, and the launcher joined all workers before postprocessing. These facts were inspected before the environment went offline; [TERMINAL_RECORD.json](../results/bounded-feedback-20261018-v1/TERMINAL_RECORD.json) is a transcription, not the original receipt.

The source, original registration/protocol and raw checkpoint files were written in the prior workspace, but their durable export is unconfirmed. No archive or rendered figure is claimed complete. The user requested small files and the stable [research scratch handoff](../research-scratch/CURRENT.md) instead. Synthetic task only; no official-test access, cloud provisioning or new scheduled work occurred.
