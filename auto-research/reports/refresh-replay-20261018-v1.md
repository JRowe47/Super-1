# Experiment report: refresh-replay-20261018-v1

Status: **completed; no promotion, retention gate failed**.

Refreshing the replay bank reduces the permanent bank's adaptation penalty, but does not solve adaptive feedback's old-fact loss. Adaptive refreshed feedback lowers the registered post-change cross-entropy by **25.10%** versus frozen refreshed feedback; the old-window retention limit fails in all three fresh seeds. Compared with adaptive FIFO, refresh still increases post-change CE by13.17% for only0.28pp mean final-old accuracy gain.

Approval: user “Alright, go ahead with the next”, following the causal-refresh proposal and standing bounded local CPU/research-branch authority. Exact source/protocol commit: `4faa9eb28162de359366e77a7ff1e980dc8826fe`; protocol SHA256: `7def232286ce614b3d3b34e3c2717e1ce789b5d35067bea4d8d7ed6fc5015235`. All22 publication files were read back byte-equal and the branch updated before experiment verification or optimization. [Protocol](../results/refresh-replay-20261018-v1/PROTOCOL.json), [registration](../results/refresh-replay-20261018-v1/REGISTRATION.json), [publication receipt](../results/refresh-replay-20261018-v1/PUBLICATION.json).

## Question and controls

Can a bounded-age replay bank support useful adaptive local feedback without exceeding2pp old-fact accuracy loss against frozen feedback?

All policies store512 examples and replay16 with16 current examples per update. FIFO stores512 recent examples and samples16. Permanent mixed stores256 recent plus256 initial old examples and samples8+8. Refreshed mixed uses the same split, initial examples, sample draws and recent queue, but replaces one bank row after each sample/update. Replacement is circular and the new entry is selected uniformly from the current16; its observed label is retained. After initial turnover, bank ages are1..256 updates. No phase, query label, loss or correctness controls refresh.

Cross all three replay policies with backprop, frozen warmup-learned feedback and continually adaptive bounded feedback: nine arms. Shared4096-update warmup per seed. Forward/feedback shapes, Fourier representation, spectral cap2, optimizer/decay, current stream, queries, exposure and tuning count remain fixed. One configuration, no LR trials or seed selection. The primary comparator is adaptive_refreshed versus frozen_refreshed; all other comparisons are descriptive.

The refresh intervention changes bank age and content together, including obsolete-label exposure. It does not isolate staleness alone. FIFO additionally differs in allocation and recent-history duration. Precomputed full streams are experimental bookkeeping; the512-example claim concerns replay capacity, not total program RAM.

## What actually ran

Fresh paired seeds901801,901802,901803; mechanical pilot98997. Main4096 common warmup+2048 online updates per arm, four512-update phases. Pilot4096+256, four64-update phases. All27 main and9 pilot arms completed. Warmup exposes1,048,576 binary targets per seed; main262,144 per arm and pilot32,768 per arm. These counts describe training; exact replay repeats optimization for verification and its CPU is charged.

Synthetic threshold-world inputs have11 dimensions and4 binary relations, with new contexts and a later rule change. Heldout queries are128 per context main and32 pilot. The current/query/warmup generator and existing replay controls are byte-preserved; additional refresh sampling uses independent seed+66,000,000. Per-seed hashes cover full generated schedules, shared warmup states, every phase/final state, and compressed checkpoints. See [source provenance](../results/refresh-replay-20261018-v1/SOURCE_PROVENANCE.json) and [per-seed results](../results/refresh-replay-20261018-v1/main/).

Python3.12.14, PyTorch2.14.1+cpu, NumPy2.3.5; complete pinned versions in RUNTIME.json. At most three concurrent single-thread CPU workers, no GPU/cloud/external datasets. Verification retains FP32 absolute2e-6 and FP64 absolute2e-12 tolerances. New independent scalar/deque references check three640-step streams, one-row replacement, causal IDs, unchanged labels, fixed capacity/batch, original controls, shared currents/queries, age bounds and removal of all obsolete labels after a complete post-change turnover. Every check passed.

All36 online trajectories were independently regenerated and replayed from the serialized warmup. Four phase and final complete-state hashes and stored final tensors match bitwise. No experiment failure, source/config/gate change, retry, excluded seed or outcome-driven adjustment occurred. [Verification](../results/refresh-replay-20261018-v1/VERIFICATION.json).

Total charged experiment CPU **261.082s /900s**. Main children including imports used232.080s aggregate, with77.950s elapsed. Main common warmup used39.647s; full main replay used76.147s. These components overlap the aggregate and must not be added again. Import canary, verification, pilot, diagnostics, launchers and frozen analysis startup are included. Dependency setup, source/report preparation and artifact transfer are excluded.

## Results

Primary CE is the per-seed equal mean of novel-context CE at steps544..768 and changed-rule CE at1056..1280, sampled every32 steps:16 registered checkpoints, nats per binary target. Old retention averages stable accuracy across the same windows and also checks final stable accuracy. Stable facts exclude the two deliberately changed relations in context0. Positive CE reduction favors adaptive feedback; negative retention deltas mean accuracy loss.

| Seed | Adaptive refreshed CE | Frozen refreshed CE | CE reduction | Old-window delta | Final-old delta |
|---|---:|---:|---:|---:|---:|
| 901801 | 0.877788 | 1.187988 | 26.11% | -6.78pp | -4.35pp |
| 901802 | 0.814861 | 1.097051 | 25.72% | -10.14pp | -3.79pp |
| 901803 | 0.847419 | 1.107089 | 23.46% | -7.04pp | -1.23pp |

Mean paired CE reduction **25.0964%**, sample SD1.4345 percentage points; all three effects positive and above5%. Retention fails every old-window comparison; final-old fails two of three. The combined gate requires both old measures within2pp in every seed, so **the predeclared gate fails**. Mean final-old gap is-3.1250pp and mean old-window gap-7.9869pp. Three paired worlds provide descriptive evidence, not population significance or transfer validation.

All-arm post-change CE, with every main seed shown:

| Learning / replay | CE901801 | CE901802 | CE901803 |
|---|---:|---:|---:|
| backprop / fifo | 0.792078 | 0.752749 | 0.771706 |
| backprop / permanent | 0.985749 | 0.936705 | 0.956723 |
| backprop / refreshed | 0.891050 | 0.838924 | 0.856652 |
| frozen / fifo | 1.096971 | 1.017720 | 1.020923 |
| frozen / permanent | 1.266743 | 1.178852 | 1.213051 |
| frozen / refreshed | 1.187988 | 1.097051 | 1.107089 |
| adaptive / fifo | 0.771154 | 0.725634 | 0.747324 |
| adaptive / permanent | 1.011912 | 0.929216 | 0.936789 |
| adaptive / refreshed | 0.877788 | 0.814861 | 0.847419 |

Final mean heldout accuracy over the three seeds:

| Learning / replay | Old accuracy | Novel accuracy | Changed accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| backprop / fifo | 97.02% | 95.87% | 87.76% | 93.55% |
| backprop / permanent | 97.15% | 94.56% | 66.54% | 86.08% |
| backprop / refreshed | 96.88% | 96.03% | 87.50% | 93.47% |
| frozen / fifo | 94.23% | 81.97% | 37.76% | 71.32% |
| frozen / permanent | 95.96% | 79.20% | 22.27% | 65.81% |
| frozen / refreshed | 94.66% | 81.67% | 37.63% | 71.32% |
| adaptive / fifo | 91.26% | 84.15% | 72.92% | 82.77% |
| adaptive / permanent | 92.56% | 82.32% | 49.09% | 74.66% |
| adaptive / refreshed | 91.54% | 82.91% | 70.57% | 81.67% |

The secondary adaptive-versus-frozen comparisons improve CE21.3558% with the permanent bank and28.4002% with FIFO; both fail retention. The primary was not switched to the more favorable FIFO contrast. Backprop retains higher final accuracy and is not compute-matched to local feedback. Adaptive feedback's favorable early post-change CE does not establish better final accuracy.

Within adaptive feedback, refreshing versus permanent replay:

| Seed | Refresh vs permanent CE reduction | Final-old delta | Final-changed delta | Refresh vs FIFO CE increase |
|---|---:|---:|---:|---:|
| 901801 | 13.25% | -1.73pp | +35.16pp | 13.83% |
| 901802 | 12.31% | -0.11pp | +14.06pp | 12.30% |
| 901803 | 9.54% | -1.23pp | +15.23pp | 13.39% |

Refresh lowers primary CE11.7004%, raises final changed accuracy21.4844pp and balanced accuracy7.0157pp. It reduces final-old accuracy1.0231pp and old-window accuracy6.1954pp. This is a tradeoff, not dominance. Backprop and frozen feedback also recover changed accuracy with refresh (+20.9635pp and+15.3646pp), so the benefit is not specific to learning the feedback matrices.

Against adaptive FIFO, refresh still raises primary CE13.1726%, changes final-old accuracy by+0.2790pp, old-window by+1.6846pp, final-changed by-2.3438pp and balanced by-1.1006pp. It does not supply a strong retention benefit at the tested rate.

Obsolete replay labels, counted as individual binary targets and shared across learning policies:

| Seed | FIFO obsolete targets | Permanent bank | Refreshed bank |
|---|---:|---:|---:|
| 901801 | 38 | 3244 | 164 |
| 901802 | 24 | 3544 | 132 |
| 901803 | 54 | 3796 | 126 |

Refreshed schedules have zero obsolete targets from zero-based update1280 onward (1024-change point+256-bank lifetime), and maximum sampled bank age256. The audit is diagnostic only. The novel CE window improves from1.037657 with a permanent bank to0.929582 with refresh, before any labels change; changed-rule window CE improves0.880954→0.763797. **Inference:** stale-label persistence contributes plausibly, but cannot explain the whole bank effect; age/content allocation also matters. FIFO remains lower on both components (0.813784 and0.682291). No mediation or staleness-only causal claim is made.

All per-seed traces, prequential training scores, same-state direction probes, costs and hashes remain in RESULT.json. [Frozen analysis](../results/refresh-replay-20261018-v1/SUMMARY.json) reports all declared descriptive interactions and endpoint components. No diagnostic gradient affects local learning.

## Deployment measurements

Forward parameters1504, local feedback12544:6016 and50176 FP32 bytes respectively before container overhead. Active backprop requires only forward parameters; experiment bookkeeping retains common feedback for exact starts. Replay payload30720 FP32 bytes plus4096 int64 ID bytes in every policy; refreshed replay additionally needs a counter and independent RNG state. These analytical components are not total deployed memory.

Measured main-process maximum RSS ranges398,980–399,192KiB **per worker**, including precomputed benchmark streams, framework and diagnostics. Simultaneous aggregate peak, allocator reservation, activation/workspace breakdown, training FLOPs, quantized storage, deployment throughput and latency are **unmeasured**. No prefill/decode task or deployment efficiency claim applies.

Four individually compressed NPZ states total1,025,296bytes. Warmup W/F is stored once per seed; BP/frozen arms store final W and reference unchanged warmup F, while adaptive arms retain final W/F. No whole-stream dump or cumulative archive. [Artifact manifest](../results/refresh-replay-20261018-v1/ARTIFACT_MANIFEST.json).

## Decision

**No promotion.** Causal refresh reduces the adaptation damage of permanent old replay but fails the unchanged joint adaptation/retention criterion. This negative primary result and the positive descriptive control are both retained. Two replay interventions have now shifted the retention/adaptation balance without resolving feedback-related old-fact loss.

Next best hypothesis: slower feedback plasticity can preserve useful post-change learning while reducing old-fact drift. Hold FIFO replay, forward lr/decay, cap, common warmup and exposure fixed. Compare frozen feedback, current adaptive feedback, a single phase-blind slow-feedback setting and backprop. A candidate prespecified setting is0.1 of the existing feedback SGD step (including its decay term), applied only after warmup; normal forward updates and spectral projection remain unchanged. It is a new hypothesis, not a rate selected by this run. Freeze the exact setting and fresh seeds before execution; no sweep or phase oracle. Primary slow-versus-frozen uses the same CE and per-seed retention gate; normal adaptive is the disconfirming control for losing the adaptation benefit. A failed effect gate would reject merely approaching the frozen endpoint.

Expected150–300 aggregate CPU seconds, within the standing900-second bounded envelope, max3 single-thread workers, no GPU/cloud, and four compact state files. This follow-on is **proposed, not implemented or running**; wait for the user's next-stage instruction. No additional paid resource authority is required for this bounded design; language/large-scale runs remain paused.

## Integrity and cleanup

All18 frozen source/protocol/runtime files retain their registered hashes. Source commit preceded all experiment checks/training; verification/pilot evidence was separately committed as `386d8addee0316275094e2c1f8554596077f54c2` while main ran unchanged. Each of the four final checkpoint files was retrieved again and matched its original SHA256. [Evidence audit](../results/refresh-replay-20261018-v1/EVIDENCE_AUDIT.json), [terminal receipt](../results/refresh-replay-20261018-v1/TERMINAL.json).

All workers exited. Temporary readback copies and1,075,541,235bytes of reproducible runtime files were removed after preservation checks. Completed evidence and earlier archives remain intact. Official tests were untouched, no cloud resource was created, no alert/email/automation was created or resumed, and no language training ran. The earlier bounded-feedback missing-source gap remains historical and unresolved; this study derives only from fully preserved predecessor source.
