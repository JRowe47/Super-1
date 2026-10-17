# Reproducing the TinyLlama depth-curve screen

This directory preserves a completed exploratory measurement. Do not overwrite
its protocol or results to rerun it. The companion directory
`../tinyllama-depth-anchor-20261017-v1` is a separately registered post-hoc
restoration diagnostic.

## Measured environment

Python 3.12, CPU PyTorch 2.14.1+cpu, Transformers 4.57.6, safetensors 0.8.0,
SciPy and NumPy versions recorded in `PROTOCOL.json`. Supporting packages used:
tokenizers 0.22.2, huggingface-hub 0.36.2, accelerate 1.15.0,
sentencepiece 0.2.2, pyarrow 26.0.0, psutil 7.2.2, matplotlib.

The execution environment reused an existing CPU PyTorch installation. Use an
appropriate CPU build in a separate environment; GPU packages are unnecessary.
Keep the recorded versions for an exact-environment attempt. Different numerical
libraries or CPUs may produce small differences; fidelity tolerances are frozen
in the protocol and must not be loosened to claim reproduction.

Allow roughly 4.4 GB for the downloaded checkpoint, 10 GB or more of memory
headroom for the materialized reconstruction workflow, and additional disk for
the runtime. This is **not** an implementation of low-memory curve inference.
The experiment ran with four CPU threads. It performs no gradient training.

## Fresh replay

From the repository root, create a new directory at the same depth and copy
only source files. The script obtains its output identity from that directory.
Never copy the completed protocol/result files into a new experiment and then
silently overwrite them.

```bash
curve_run=auto-research/results/tinyllama-depth-curves-replay
mkdir "$curve_run"
cp auto-research/results/tinyllama-depth-curves-20261017-v1/acquire.py "$curve_run/"
cp auto-research/results/tinyllama-depth-curves-20261017-v1/experiment.py "$curve_run/"

python auto-research/results/tinyllama-depth-curves-20261017-v1/replay_inputs.py \
  --reference auto-research/results/tinyllama-depth-curves-20261017-v1/INPUTS.json \
  --inputs /path/to/replay-inputs \
  --manifest "$curve_run/INPUTS.json"

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python "$curve_run/experiment.py" prepare \
  --inputs /path/to/replay-inputs --cache /path/to/replay-cache

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python "$curve_run/experiment.py" structure \
  --inputs /path/to/replay-inputs --cache /path/to/replay-cache

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
timeout 1800s python "$curve_run/experiment.py" functional \
  --inputs /path/to/replay-inputs --cache /path/to/replay-cache
```

The replay helper downloads only the model's configuration/tokenizer/safetensors
and WikiText-2 raw train/validation. It uses the **pinned URLs already recorded in
the published manifest** and checks every file's SHA256 and size, without
resolving a moving model or dataset ref. It never fetches the official test.
The original `acquire.py` resolved and recorded the dataset revision for the
measured experiment; `replay_inputs.py` reuses that exact revision for replay.
Freeze the same source/config/input hashes before scoring.

The optional diagnostic script is intentionally bound to the completed parent
run by hashes. To repeat it in a fresh experiment, copy it to a new sibling run,
explicitly set its `PARENT` to your completed primary replay, register with
`prepare`, then execute `run` using the same input and cache arguments. That
adaptation is new source and must be included in the new protocol's source hash.

## Review and plot regeneration

The original completed outcomes can be checked and rendered without downloading
the model:

```bash
python auto-research/results/tinyllama-depth-curves-20261017-v1/report.py
```

This verifies saved bindings/aggregates, regenerates the two reports and figures,
and updates the two registry entries. It does not rerun inference or training.

`PROJECTORS.json` contains the small exact depth-mixing matrices used to
reconstruct each FFN family from the pinned checkpoint. The hypothetical
compact representation consists of matrix bases and depth coefficients; no
compact checkpoint was exported or speed benchmark claimed.

New experiment code follows the repository MIT license. External TinyLlama
weights retain their Apache-2.0 model license; WikiText retains its upstream
dataset terms. No model weights or corpus passages are redistributed here.
