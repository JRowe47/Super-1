"""Bake teacher logits, predictive atoms, future probes, and interventions."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import torch

from representation_experiments.common import TokenMemmap, atomic_json, load_json
from representation_experiments.predictive_atlas import (
    behavioral_fingerprint,
    corrupt_context,
    nearest_atoms,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequences", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    return parser.parse_args()


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else contextlib.nullcontext())


def future_atoms(atom_ids: torch.Tensor, horizons: list[int]):
    result = []
    for horizon in horizons:
        shifted = torch.full_like(atom_ids, -100)
        if horizon < atom_ids.shape[1]:
            shifted[:, :-horizon] = atom_ids[:, horizon:]
        result.append(shifted)
    return result


def main():
    args = arguments()
    config = load_json(args.config)
    bake = config["predictive_bake"]
    atlas = torch.load(args.atlas, map_location="cpu", weights_only=False)
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install requirements-representation.txt on the worker") from error
    device = torch.device("cuda", int(bake.get("device", 0))) if torch.cuda.is_available() \
        else torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        config["teacher"], torch_dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="sdpa").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config["teacher"])
    centers = atlas["centers"].to(device)
    buckets = atlas["hash_buckets"].to(device)
    signs = atlas["hash_signs"].to(device)
    data = TokenMemmap.open(
        Path(config["data"]) / f"{args.split}.bin", config["token_dtype"])
    sequences = args.sequences or int(bake["sequences"])
    batch_size = int(bake["batch_size"])
    context = int(bake["context"])
    batches_per_shard = int(bake["sequences_per_shard"]) // batch_size
    steps = (sequences + batch_size - 1) // batch_size
    manifest = []
    for shard_start in range(0, steps, batches_per_shard):
        shard_index = shard_start // batches_per_shard
        shard_path = output / f"shard-{shard_index:05d}.pt"
        if shard_path.exists():
            manifest.append(shard_path.name)
            continue
        records = []
        for step in range(shard_start, min(steps, shard_start + batches_per_shard)):
            raw = data.batch_for_step(
                context, batch_size, step, int(bake["seed"])).to(device)
            inputs = raw[:, :-1]
            targets = raw[:, 1:]
            intervention = corrupt_context(
                inputs, tokenizer.eos_token_id,
                float(atlas["intervention_span_fraction"]), int(bake["seed"]) + step)
            with torch.no_grad(), autocast(device):
                original_logits = model(input_ids=inputs, use_cache=False).logits
                changed_logits = model(input_ids=intervention, use_cache=False).logits
            fingerprint = behavioral_fingerprint(
                original_logits, changed_logits, buckets, signs,
                int(atlas["teacher_topk"]), int(atlas["sketch_dimensions"]))
            changed_fingerprint = behavioral_fingerprint(
                changed_logits, original_logits, buckets, signs,
                int(atlas["teacher_topk"]), int(atlas["sketch_dimensions"]))
            flat_atoms = nearest_atoms(fingerprint.reshape(-1, fingerprint.shape[-1]), centers)
            changed_atoms = nearest_atoms(
                changed_fingerprint.reshape(-1, changed_fingerprint.shape[-1]), centers)
            atom_ids = flat_atoms.view_as(inputs)
            changed_atom_ids = changed_atoms.view_as(inputs)
            log_probability = torch.log_softmax(original_logits.float(), dim=-1)
            top_values, top_indices = torch.topk(
                log_probability, int(bake["teacher_topk"]), dim=-1)
            records.append({
                "input_ids": inputs.cpu(), "target_ids": targets.cpu(),
                "teacher_topk_indices": top_indices.int().cpu(),
                "teacher_topk_logprobs": top_values.to(torch.float16).cpu(),
                "atom_ids": atom_ids.int().cpu(),
                "future_atom_ids": [value.int().cpu() for value in future_atoms(
                    atom_ids, [int(x) for x in config["students"]["hierarchical_predictive"]["future_horizons"]])],
                "intervention_ids": intervention.cpu(),
                "intervention_atom_ids": changed_atom_ids.int().cpu(),
            })
        merged = {
            key: ([torch.cat([item[key][head] for item in records], dim=0)
                   for head in range(len(records[0][key]))]
                  if isinstance(records[0][key], list)
                  else torch.cat([item[key] for item in records], dim=0))
            for key in records[0]
        }
        temporary = shard_path.with_suffix(".tmp")
        torch.save(merged, temporary)
        os.replace(temporary, shard_path)
        manifest.append(shard_path.name)
        print(json.dumps({"event": "baked_shard", "path": str(shard_path)}), flush=True)
    summary = {
        "format": "predictive_sequence_bake_v1", "teacher": config["teacher"],
        "atlas": str(Path(args.atlas).resolve()), "shards": manifest,
        "sequences": sequences, "context": context, "split": args.split,
        "teacher_topk": int(bake["teacher_topk"]),
    }
    atomic_json(summary, output / "manifest.json")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
