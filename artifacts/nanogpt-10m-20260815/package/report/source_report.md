# Official nanoGPT vs. Proven Pure Hierarchy

## Matched 10-million-token comparison

**Date:** 2026-08-15  
**Status:** Training, fixed evaluation, and four-task benchmark complete  
**Primary question:** How does the proven Aug-8 pure-hierarchy design compare with official nanoGPT under the same data, teacher-distillation objective, training exposure, and evaluation protocol?

## Executive summary

The official nanoGPT comparator and the historical pure-hierarchy model were closely parameter matched and trained for the same **4,883 optimizer steps / 10,000,384 tokens** on the same corpus and deterministic sample schedule.

- At the trained **512-token context**, nanoGPT achieved **2.54% lower fixed-evaluation CE**.
- At **1,024 tokens**, hierarchy achieved **3.63% lower CE** while nanoGPT's theoretical state was **7,001.30× as large**.
- At **2,048 tokens**, hierarchy achieved **11.64% lower CE** while nanoGPT's theoretical state was **14,002.60× as large**.
- The hierarchy state remained **5,504 bytes (5.375 KiB)** per sequence at every tested context. nanoGPT's theoretical BF16 KV state grew linearly from **18.375 MiB** at context 512 to **73.5 MiB** at context 2,048.
- nanoGPT's four-task benchmark mean was **0.286338**, compared with hierarchy's **0.280206**: a **0.006132 absolute / 2.19% relative** nanoGPT advantage.

The strongest result is therefore not a universal quality win. It is a long-context state-efficiency result: at context 2,048, hierarchy produced lower CE using **99.9929% fewer state bytes**.

## Matched experimental contract

| Variable | Matched value |
|---|---:|
| Teacher | Qwen3-0.6B-Base |
| Training corpus | Byte-identical `train.bin` |
| Validation corpus | Byte-identical `val.bin` |
| Training context | 512 tokens |
| Batch size | 4 sequences |
| Optimizer steps | 4,883 |
| Executed tokens | 10,000,384 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Warmup | 500 steps |
| Betas | 0.9, 0.95 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Training seed | 12001 |
| Objective | 0.1 next-token CE + 1.0 multi-temperature forward KL |
| Evaluation seed | 14001 |

Parameter counts were nearly exact:

- Pure hierarchy: **119,694,800** parameters
- Official nanoGPT: **119,685,440** parameters
- Difference: **9,360 parameters**, or **0.00782%**

The nanoGPT comparator used pinned upstream nanoGPT model code with geometry **width 448 / 21 layers / 7 heads / head dimension 64**. The hierarchy used the original proven width-688 pure recurrent hierarchy, unchanged.

## Fixed-evaluation cross-entropy

Lower CE is better.

| Context | Official nanoGPT CE | Pure hierarchy CE | Result |
|---:|---:|---:|---|
| 512 | **5.126457** | 5.260253 | nanoGPT 2.54% lower relative to hierarchy |
| 1,024 | 5.261362 | **5.070329** | hierarchy 3.63% lower relative to nanoGPT |
| 2,048 | 5.988185 | **5.291337** | hierarchy 11.64% lower relative to nanoGPT |

The 512 result is based on 16 batches of 2 sequences. Each 1,024- and 2,048-token result is based on 2 batches of 1 sequence.

## State comparison

These values are nominal BF16 per-sequence inference-state accounting, not measured total GPU memory.

| Context | nanoGPT theoretical KV state | Hierarchy recurrent state | nanoGPT state / hierarchy state | Hierarchy reduction |
|---:|---:|---:|---:|---:|
| 512 | 19,267,584 B (18.375 MiB) | 5,504 B (5.375 KiB) | **3,500.65×** | **99.9714% fewer bytes** |
| 1,024 | 38,535,168 B (36.75 MiB) | 5,504 B (5.375 KiB) | **7,001.30×** | **99.9857% fewer bytes** |
| 2,048 | 77,070,336 B (73.5 MiB) | 5,504 B (5.375 KiB) | **14,002.60×** | **99.9929% fewer bytes** |

State formulas:

- Hierarchy: `4 recurrent tensors × width 688 × 2 BF16 bytes = 5,504 bytes`.
- nanoGPT: `K/V × 21 layers × 7 heads × 64 head dimension × context × 2 BF16 bytes`.

At the two long-context points where hierarchy also had lower CE, it achieved:

- **Context 1,024:** 3.63% lower CE using 1/7,001.30 as many state bytes.
- **Context 2,048:** 11.64% lower CE using 1/14,002.60 as many state bytes.

## Matched training checkpoints

These are instantaneous CE values from the same deterministic training minibatch at the same step, not held-out evaluation averages.

| Step | Tokens | nanoGPT CE | Hierarchy CE | Difference |
|---:|---:|---:|---:|---|
| 3,821 | 7,825,408 | **4.975600** | 4.985255 | nanoGPT 0.19% lower |
| 4,521 | 9,259,008 | **4.993795** | 5.076864 | nanoGPT 1.64% lower |
| 4,883 | 10,000,384 | **5.317025** | 5.426507 | nanoGPT 2.02% lower |

The nanoGPT values are included in the complete 246-record training log. The
three historical hierarchy rows are preserved transcript values; the original
full historical training log is not part of this package. These rows are
supporting evidence and are not used for the primary fixed-evaluation result.

## Four-task benchmark

The historical limit-1,000 evaluation was run against the completed official nanoGPT checkpoint.

| Task | Official nanoGPT | Pure hierarchy | Higher score |
|---|---:|---:|---|
| ARC-Easy | **0.340351** | 0.329825 | nanoGPT |
| HellaSwag | **0.275000** | 0.257000 | nanoGPT |
| LAMBADA | 0.000000 | 0.000000 | Tie |
| PIQA | 0.530000 | **0.534000** | Hierarchy |
| Mean | **0.286338** | 0.280206 | nanoGPT |

nanoGPT's mean was **0.006132 points higher**, a **2.19% relative advantage**. This is one seed with no confidence interval.

## Interpretation

1. **nanoGPT is slightly better at the trained context.** Its fixed context-512 CE was 2.54% lower, and its benchmark mean was 2.19% higher.
2. **Hierarchy generalizes better to longer contexts in this protocol.** At context 2,048, hierarchy's CE was 5.291337 versus nanoGPT's 5.988185.
3. **The state gap is exceptionally large.** The hierarchy maintains 5.375 KiB of recurrent state while nanoGPT's theoretical KV state grows with context, reaching 73.5 MiB at context 2,048.
4. **The result is architecture-specific, not a blanket transformer claim.** The official nanoGPT model used learned absolute position embeddings; training exposed only positions 0–511, so positions 512–2,047 were not trained. This is an important explanation for the long-context degradation and must accompany the result.

## Reproducibility and provenance

Historical hierarchy evidence:

- Config SHA-256: `f47d8cabafa5f16ff32c1169fba2064bf8124907cfc99e7d8374212cc25c58fa`
- Result SHA-256: `ffe3dea20685e88381eab65f02bba504e6870975a0ec06854de73718b88f769d`
- Benchmark SHA-256: `109f61c95d03dba065a31c4cbc683e7b96ebb52a507afab9cae8136e94188b43`
- Final checkpoint SHA-256: `beb12f7a9fce10195c639365cc673200e04c97affa9557239b70edd28bd9c66f`

Official nanoGPT evidence:

- Frozen package manifest SHA-256: `dc4e324d7f43ddf417c6641ce3c6ec856cbec69a6d1b83e93f2fe5b0690c833c`
- Result SHA-256: `2120eac7d5f699d810425a8b793eccc0fd80674430751a2cc6d33361d81e3d44`
- Training completion SHA-256: `6e3c1eda103ff89e6ae687e800f2cbf3c0cb76dd3a25b12fb2bc68c0df9419a7`
- Final model SHA-256: `7de4e9d81b606ec6b0d7e9fbaf4dcdcb30aa55456e775a07371aaf65a6591241`
- Training log SHA-256: `867e64674ed291d3eb8e8594d316c68c19c830fc7a70206d9542680ddcce532c`
- Benchmark SHA-256: `1df9a3e0f410ffee5189e111137acd2755717a6a7c81cfb099e54c4cbb7f340e`
- Benchmark completion SHA-256: `fc28be724d939e0f619d935d47b1ffc356a06e2c66b6530efd41523ca3042be6`

The reproduction package contains the historical result JSON, benchmark JSON,
the frozen official nanoGPT comparator, the completed nanoGPT result chain, and
portable verification and rerun tools. All paths in the package are relative.
