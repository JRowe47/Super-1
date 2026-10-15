# Recovered original study: reproduction map

## What is recovered

The nanoGPT final checkpoint matches its original completion receipt SHA-256 exactly: `7de4e9d81b606ec6b0d7e9fbaf4dcdcb30aa55456e775a07371aaf65a6591241`.

All four recorded deployed hierarchy core hashes match the recovered source under `exact-historical-source/representation_experiments/`. The previous compatible snapshot is preserved elsewhere; it must not be silently substituted or relabeled.

Full per-arm historical training logs, result/config files, failed benchmark logs and summary are recovered. Five final model checkpoints and five resumable checkpoints are being preserved in the checkpoint release; only assets whose terminal upload receipt and GitHub digest match are complete.

The installed training2 Python environment is captured separately with an interpreter, standard library, package-version/metadata hashes, archive hash and six part hashes. The environment passed a CPU import/arithmetic test in its original read-only filesystem with ephemeral scratch space. This is not proof that the exported archive is independently portable, that the environment is unchanged since August, or that historical metrics have been rerun.

## Download and verify before running

Use the private repository releases, authenticated with your own GitHub account:

- `recovered-120m-checkpoints-20261015`: historical final and resumable checkpoints.
- `replication-inputs-20261015`: corpus, teacher, tokenizer/config, pinned benchmark caches and their manifest.
- `recovered-runtime-20261015`: installed environment archive parts and runtime manifest.

Use the release asset names and SHA-256 values in the terminal manifests. Do not load a checkpoint until its hash matches. Treat checkpoint loading as code-sensitive; prefer a pinned trusted loader and `weights_only=True` when its actual schema supports that.

Download runtime parts in numbered order, verify every part, then concatenate and verify the complete archive hash before extraction. Inspect archive paths before extraction and use an isolated destination. Do not extract an archive containing absolute-path environment symlinks over the host root filesystem. It is a Linux x86-64 environment, not a macOS virtualenv.

Disk space is a prerequisite: the current SSD is nearly full. Do not launch reproduction until sufficient free space exists for the downloaded inputs, checkpoints, runtime parts, expanded environment and outputs. The historical configuration's safety floor must also be satisfied. Never fall back to iCloud.

## Map historical paths explicitly

The recovered configuration uses vocabulary 151936, uint32 corpus tokens, 120M target parameters, seed 12001, training context 512 and approximately 10M tokens. Read the actual frozen configuration rather than transcribing these notes as a replacement config.

Set PYTHONPATH to the parent of the recovered `representation_experiments` directory. Restore corpus, teacher and cache paths into a new derived config with a recorded diff and hash; retain the original config unchanged. Preserve tokenizer identity, distillation objective, training exposure, evaluation offsets and precision. Use the frozen official nanoGPT adapter/package from the original replication package, not an arbitrary current nanoGPT checkout.

The historical launch interface is:

```sh
python -B -m representation_experiments.run_matched_hierarchy_single_gpu \
  --config /absolute/path/to/derived-path-config.json \
  --out /absolute/ssd/path/to/new-run \
  --gpu 0
```

This command trains all configured arms and runs their historical benchmarks. It is documentation, NOT authorization to launch it. Read the source before execution; obtain an approved GPU/resource budget first. A partial checkpoint load or synthetic canary is not a complete reproduction.

## Runtime provenance

The training2 disk records base image `projects/ml-images/global/images/common-cu129-ubuntu-2204-nvidia-580-v20260722`. The recovered OS reports Ubuntu 22.04.5 LTS; the tested virtualenv reports Python 3.10.12 and PyTorch 2.11.0+cu130. Do not infer that the base-image CUDA label equals the installed Torch CUDA build. Required system libraries and drivers must be checked separately when constructing a portable image. Third-party licenses remain applicable.

## Evidence limits and seals

Recovered results are old evidence, not new wins. The historical 2K nanoGPT comparison trained absolute positions only through 512; preserve that extrapolation caveat. Persistent-state ratios are not total memory ratios. No new SOTA, novelty or quality improvement is claimed here. The official WikiText-2 test remains sealed under the existing validation eligibility protocol.
