"""Fit predictive atoms from Qwen behavior on input-defined contexts."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import torch

from representation_experiments.common import TokenMemmap, atomic_json, load_json, seed_everything
from representation_experiments.predictive_atlas import (
    OnlineKMeans,
    atlas_diagnostics,
    behavioral_fingerprint,
    corrupt_context,
    hash_projection,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--contexts", type=int, default=0)
    return parser.parse_args()


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else contextlib.nullcontext())


@torch.no_grad()
def fingerprints(model, input_ids, replacement, fit, buckets, signs, seed):
    intervention = corrupt_context(
        input_ids, replacement, float(fit["intervention_span_fraction"]), seed)
    with autocast(input_ids.device):
        original = model(input_ids=input_ids, use_cache=False).logits
        changed = model(input_ids=intervention, use_cache=False).logits
    positions = torch.linspace(
        max(0, input_ids.shape[1] // 4), input_ids.shape[1] - 1,
        int(fit["positions_per_sequence"]), device=input_ids.device).long().unique()
    value = behavioral_fingerprint(
        original.index_select(1, positions), changed.index_select(1, positions),
        buckets, signs, int(fit["teacher_topk"]),
        int(fit["sketch_dimensions"]))
    return value.reshape(-1, value.shape[-1])


def main():
    args = arguments()
    config = load_json(args.config)
    fit = config["atlas_fit"]
    seed_everything(int(fit["seed"]))
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final = output / "atlas.pt"
    if final.exists():
        print(json.dumps({"event": "already_complete", "path": str(final)}))
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install requirements-representation.txt on the worker") from error

    device = torch.device("cuda", int(fit.get("device", 0))) if torch.cuda.is_available() \
        else torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        config["teacher"], torch_dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="sdpa").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config["teacher"])
    replacement = tokenizer.eos_token_id
    buckets, signs = hash_projection(
        int(model.config.vocab_size), int(fit["sketch_dimensions"]),
        int(fit["hash_seed"]), device)
    data = TokenMemmap.open(
        Path(config["data"]) / "train.bin", config["token_dtype"])
    target_contexts = args.contexts or int(fit["contexts"])
    batch_size = int(fit["batch_size"])
    samples_per_step = batch_size * int(fit["positions_per_sequence"])
    steps = (target_contexts + samples_per_step - 1) // samples_per_step
    atoms = int(fit["atoms"])
    initializer = []
    kmeans = None
    processed = 0
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        kmeans = OnlineKMeans(payload["centers"].to(device), payload["counts"].to(device))
        processed = int(payload["processed"])

    for step in range(processed // samples_per_step, steps):
        raw = data.batch_for_step(
            int(fit["context"]), batch_size, step, int(fit["seed"])).to(device)
        sample = fingerprints(
            model, raw[:, :-1], replacement, fit, buckets, signs,
            int(fit["seed"]) + step).float()
        sample = sample[:max(0, target_contexts - processed)]
        if kmeans is None:
            initializer.append(sample.cpu())
            available = sum(item.shape[0] for item in initializer)
            if available >= atoms:
                initial = torch.cat(initializer, dim=0)[:atoms].to(device)
                kmeans = OnlineKMeans(initial)
                initializer = []
        else:
            kmeans.update(sample, float(fit["rate_floor"]))
        processed = min(target_contexts, processed + sample.shape[0])
        if kmeans is not None and (
                (step + 1) % int(fit["checkpoint_every"]) == 0 or step + 1 == steps):
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({
                "centers": kmeans.centers.cpu(), "counts": kmeans.counts.cpu(),
                "processed": processed,
            }, temporary)
            os.replace(temporary, checkpoint)
        if (step + 1) % int(fit["log_every"]) == 0:
            print(json.dumps({"event": "atlas_fit", "contexts": processed}), flush=True)

    if kmeans is None:
        raise RuntimeError("not enough contexts to initialize every predictive atom")
    payload = {
        "format": "behavioral_predictive_atlas_v1",
        "teacher": config["teacher"],
        "centers": kmeans.centers.cpu(),
        "counts": kmeans.counts.cpu(),
        "hash_buckets": buckets.cpu(),
        "hash_signs": signs.cpu(),
        "teacher_topk": int(fit["teacher_topk"]),
        "sketch_dimensions": int(fit["sketch_dimensions"]),
        "contexts": processed,
        "intervention_span_fraction": float(fit["intervention_span_fraction"]),
        "diagnostics": atlas_diagnostics(kmeans.centers, kmeans.counts),
    }
    temporary = output / "atlas.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, final)
    atomic_json(payload["diagnostics"], output / "summary.json")
    print("DONE", json.dumps(payload["diagnostics"]), flush=True)


if __name__ == "__main__":
    main()
