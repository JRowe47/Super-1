# Representation-first Qwen3 experiments

This package preregisters two independent tests against the frozen
`Qwen/Qwen3-0.6B-Base` teacher.  It does not modify or depend on the completed
Qwen3-1.7B conversion queues.

## Study A: replace attention with a function

Only Qwen's attention modules are replaced. Embeddings, RMSNorms, MLPs, residual
connections, tokenizer, and tied language-model head are copied from the frozen
teacher and kept frozen. Four causal operators are trained end-to-end against
the same teacher traces and token batches:

| Arm | Global path | Local/multiscale path | Shared across depth |
|---|---|---|---|
| `positive_kernel` | positive causal kernel features | none | no |
| `fourier_local` | causal spectrally parameterized convolution | exact sliding-window attention | no |
| `wavelet_rbf` | learned RBF charts | causal Haar-like wavelets | no |
| `atlas` | adaptive chart x basis features | exact local attention plus wavelets | yes |

The training objective contains four separately logged terms: attention-output
NMSE, decoder residual-stream NMSE, final-logit NMSE, and full teacher KL. The
evaluator measures identical fixed tokens at 128, 256, 512, and 1024-token
contexts, top-1 agreement, language-model loss, throughput, peak memory, unique
stored parameters, analytic attention FLOPs, and profiler-counted full-forward
FLOPs on identical inputs. Promotion fails if measured FLOPs are unavailable.

Promotion requires every preregistered gate to pass. In particular, a low loss
does not compensate for larger storage, larger FLOPs, or long-context collapse.
The thresholds live in the JSON manifest rather than in the trainer.

Run on four GPUs:

```bash
PYTHON_BIN=v4-venv/bin/python GPU_LIST=0,1,2,3 \
  representation_experiments/run_operator_suite.sh \
  representation_experiments/configs/qwen06b_representation_studies.json \
  runs/qwen06b-operator-study
```

The authoritative decision is `runs/qwen06b-operator-study/summary.json`.

## Atlas V2: state-oriented recovery

Atlas V2 is an isolated follow-up to the failed one-bank Atlas arm. It uses four
depth-grouped shared banks, rank-32 coordinate adapters at every layer, eight
functional heads, signed features, sparse top-2 routing among 32 charts, and
exact causal local windows of 32 and 128 tokens. The original dense attention
modules are frozen training bypasses; their contribution is annealed to zero,
and every validation result sets the functional path to 100 percent.

The objective emphasizes properties that need not share the teacher's internal
coordinate system: teacher KL and CE, residual-stream Gram geometry,
communication magnitude, future-state probes, counterfactual response deltas,
and routing balance. Future probes are training-only and excluded from the
deployable parameter count and final checkpoint.

The 50M-token run has a strict held-out gate at 10M tokens. It continues only if
the 512-token student-minus-teacher CE gap is at most 0.5. The final target is a
gap at most 0.2, teacher KL at most 0.2, and top-1 agreement at least 0.9.

```bash
PYTHON_BIN=v4-venv/bin/python \
  representation_experiments/run_atlas_v2.sh \
  representation_experiments/configs/qwen06b_atlas_v2.json \
  runs/qwen06b-atlas-v2
```

## Study B: compile behavior into predictive states

This study never uses a teacher neuron as a representation coordinate. A
context is represented by a behavioral fingerprint containing:

1. a CountSketch of the teacher's next-token distribution; and
2. the change in that sketch after an input span is replaced.

The atlas fitter evaluates 32 causal positions per teacher sequence, giving two
million input-defined contexts without two million separate forward passes. It
fits 4,096 online atoms and records atom utilization and collapse diagnostics.

The bake stage creates immutable training shards containing original and
intervened inputs, teacher top-k log probabilities, current atom IDs, and future
atom IDs at horizons 1, 4, and 16. Student training never calls the teacher.

Three primary students are fitted under the same 120M stored-parameter ceiling:

| Student | Objective | Sequence state |
|---|---|---|
| `transformer` | ordinary next-token CE | standard per-layer KV cache |
| `recurrent_logits` | sparse teacher-logit distillation only | fixed recurrent state |
| `hierarchical_predictive` | CE, teacher logits, atoms, future atoms, interventions, causal deltas | token, segment, and document states |

A fourth confirmation arm, `hierarchical_logit_control`, uses the identical
hierarchical architecture and storage but receives only CE and teacher logits.
The representation-first claim is rejected unless the full predictive objective
beats this same-architecture control as well as both primary baselines.

Widths are selected programmatically at the manifest's configured granularity so
each architecture is the widest model that does not exceed the shared storage ceiling. The result
records the exact utilization; a promotion is rejected if the utilization
spread exceeds the configured tolerance.

The students roll forward autonomously for 512 tokens. Evaluation measures
next-token loss, teacher-top1 agreement, future-probe reconstruction, degradation
by rollout position, stored state bytes, throughput, memory, and matched public
benchmarks (LAMBADA, HellaSwag, ARC-Easy, and PIQA).

Run on four GPUs:

```bash
PYTHON_BIN=v4-venv/bin/python GPU_LIST=0,1,2,3 BENCHMARK_LIMIT=1000 \
  representation_experiments/run_predictive_suite.sh \
  representation_experiments/configs/qwen06b_representation_studies.json \
  runs/qwen06b-predictive-state-study
```

The authoritative decision is `runs/qwen06b-predictive-state-study/summary.json`.

## Prepare the fixed corpus

Both studies use the same Qwen-tokenized FineWeb-Edu corpus and held-out split.
The existing resume-safe tokenizer supports Qwen's uint32 vocabulary:

```bash
python prepare_pythia_data.py \
  --tokenizer Qwen/Qwen3-0.6B-Base \
  --dataset HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --train_tokens 1200000000 \
  --val_tokens 10000000 \
  --token_dtype uint32 \
  --out data/qwen3-06b-representation
```

The 1.2B-token file supplies a large context pool for atlas fitting. The first
decision run bakes and trains on approximately 25M distinct student tokens;
longer training is allowed only after the preregistered gate passes. Validation
shards are baked from `val.bin`, never from the training file.

## Interpretation rules

- Operator Study A asks whether a functional communication operator can replace
  attention. It does not claim a language-model improvement.
- Predictive Study B asks whether behavioral-state supervision beats ordinary
  CE and ordinary logit distillation at equal storage.
- A failed gate is a result. Do not promote an arm because one chart looks good.
- The public benchmark comparison is exploratory until repeated with at least
  three seeds and the full benchmark sets.
- Analytic FLOPs are a gate, but measured latency and peak memory are also
  reported. A nominally linear operator can still be kernel-inefficient.

## Resume and preservation

All long stages write atomic checkpoints and skip final artifacts that already
exist. Results, checkpoints, atlas definitions, and baked shards are never
deleted by either launcher. Re-running the launcher resumes or verifies the
existing output directory.
