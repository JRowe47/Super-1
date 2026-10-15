# Official nanoGPT comparator for the completed Aug-8 hierarchy study

This fresh package adds one comparator only: pinned upstream nanoGPT trained
under the byte-bound Aug-8 teacher-distillation contract. It does not modify or
resume the historical hierarchy study and contains no perturbation solver,
energy telemetry, extra representation objective, or distributed launch path.

The unchanged contract is Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, the exact 1B/2M-token corpus,
restart-stable random sampling, batch 4 at context 512, AdamW, scalar CE weight
0.1, scalar multi-temperature forward-KL weight 1.0, seed 12001, 4,883 steps,
10,000,384 executed tokens, and fixed evaluation at 512/1,024/2,048 tokens.

The model is vendored nanoGPT commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291` with its tensor-free adapter.
An exhaustive, precommitted search over head dimension 64, 1-16 heads and 1-24
layers chooses W448/L21/H7: 119,685,440 stored parameters, only 9,360 below the
historical hierarchy's 119,694,800.

Run the focused local suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
python3 -B -m unittest discover -s tests -v
```

After independent audit and freeze, use a brand-new output path on one visible
GPU. `TEACHER_SNAPSHOT` must be the exact Hugging Face snapshot directory whose
leaf is the bound revision and whose `model.safetensors` has the configured
SHA-256.

```bash
cd /home/rohanarun/funcweights
env CUDA_VISIBLE_DEVICES=0 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=/absolute/frozen/aug8_official_nanogpt_comparator_20260815_v1 \
  /absolute/python -B -m representation_experiments.run_exact \
  --runtime-root /home/rohanarun/funcweights \
  --teacher-snapshot-root "$TEACHER_SNAPSHOT" \
  --out /home/rohanarun/funcweights/runs/FRESH_ATTEMPT
```

The training phase includes all fixed CE evaluations. After its exact PASS
completion, the historical four-task limit-1000 benchmark is a separate
create-only phase:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=/absolute/frozen/aug8_official_nanogpt_comparator_20260815_v1 \
  /absolute/python -B -m representation_experiments.run_benchmark_exact \
  --runtime-root /home/rohanarun/funcweights \
  --teacher-snapshot-root "$TEACHER_SNAPSHOT" \
  --run-root /home/rohanarun/funcweights/runs/FRESH_ATTEMPT
```

This candidate is deliberately unfrozen and grants no staging or launch
authority until independent audit passes.
