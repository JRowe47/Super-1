"""Train one equal-storage student and evaluate autonomous 512-token rollout."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from representation_experiments.common import (
    atomic_json,
    cosine_with_warmup,
    load_json,
    seed_everything,
)
from representation_experiments.predictive_models import build_to_budget


class BakedBatches:
    def __init__(self, manifest_path: str | Path, batch_size: int, seed: int):
        manifest_path = Path(manifest_path).expanduser().resolve()
        manifest = load_json(manifest_path)
        self.paths = []
        for value in manifest["shards"]:
            path = Path(value)
            if not path.is_absolute():
                path = manifest_path.parent / path
            elif not path.exists():
                portable = manifest_path.parent / path.name
                if portable.exists():
                    path = portable
            self.paths.append(path)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.entries = []
        for path in self.paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            count = int(payload["input_ids"].shape[0])
            self.entries.extend((path, start) for start in range(0, count, batch_size)
                                if start + batch_size <= count)
        if not self.entries:
            raise ValueError("baked manifest contains no full student batches")
        self.cached_path = None
        self.cached_payload = None

    def batch(self, step: int):
        epoch, offset = divmod(step, len(self.entries))
        order = np.random.default_rng(self.seed + epoch).permutation(len(self.entries))
        path, start = self.entries[int(order[offset])]
        if path != self.cached_path:
            self.cached_payload = torch.load(path, map_location="cpu", weights_only=False)
            self.cached_path = path
        end = start + self.batch_size
        return {
            key: ([item[start:end] for item in value]
                  if isinstance(value, list) else value[start:end])
            for key, value in self.cached_payload.items()
        }


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--student", required=True,
                        choices=("transformer", "recurrent_logits", "hierarchical_predictive",
                                 "hierarchical_logit_control"))
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else contextlib.nullcontext())


def to_device(payload, device):
    return {
        key: ([item.to(device) for item in value]
              if isinstance(value, list) else value.to(device))
        for key, value in payload.items()
    }


def sparse_teacher_loss(student_logits, indices, logprobs):
    teacher_probability = logprobs.float().exp()
    teacher_probability = teacher_probability / teacher_probability.sum(-1, keepdim=True).clamp_min(1e-8)
    student_logprob = torch.log_softmax(student_logits.float(), dim=-1)
    selected = student_logprob.gather(-1, indices.long())
    return -(teacher_probability * selected).sum(-1).mean()


def predictive_losses(model, kind, batch, objective, atom_centers):
    output = model(batch["input_ids"].long())
    ce = F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        batch["target_ids"].reshape(-1).long())
    sparse = sparse_teacher_loss(
        output.logits, batch["teacher_topk_indices"],
        batch["teacher_topk_logprobs"])
    losses = {"ce": ce, "sparse_teacher": sparse}
    total = float(objective.get("ce", 0.0)) * ce + float(
        objective.get("sparse_teacher", 0.0)) * sparse

    if kind == "hierarchical_predictive":
        atom = F.cross_entropy(
            output.atom_logits.reshape(-1, output.atom_logits.shape[-1]),
            batch["atom_ids"].reshape(-1).long())
        future_values = []
        for logits, target in zip(output.future_atom_logits, batch["future_atom_ids"]):
            future_values.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1).long(),
                ignore_index=-100))
        future = torch.stack(future_values).mean()
        changed = model(batch["intervention_ids"].long())
        intervention_atom = F.cross_entropy(
            changed.atom_logits.reshape(-1, changed.atom_logits.shape[-1]),
            batch["intervention_atom_ids"].reshape(-1).long())

        stride = int(objective.get("causal_stride", 8))
        original_probability = torch.softmax(output.atom_logits[:, ::stride].float(), dim=-1)
        changed_probability = torch.softmax(changed.atom_logits[:, ::stride].float(), dim=-1)
        predicted_delta = (original_probability - changed_probability) @ atom_centers.float()
        original_target = atom_centers.index_select(
            0, batch["atom_ids"][:, ::stride].reshape(-1).long()).view_as(predicted_delta)
        changed_target = atom_centers.index_select(
            0, batch["intervention_atom_ids"][:, ::stride].reshape(-1).long()).view_as(predicted_delta)
        causal = F.mse_loss(predicted_delta, original_target - changed_target)
        losses.update({
            "atom": atom, "future": future,
            "intervention_atom": intervention_atom, "causal": causal,
        })
        total = (total + float(objective["atom"]) * atom
                 + float(objective["future"]) * future
                 + float(objective["intervention_atom"]) * intervention_atom
                 + float(objective["causal"]) * causal)
    losses["total"] = total
    return output, losses


@torch.no_grad()
def evaluate(model, kind, batches, atlas_centers, objective, device, count):
    model.eval()
    positions = {}
    totals = {"ce": 0.0, "sparse_teacher": 0.0, "teacher_top1": 0.0,
              "atom_accuracy": 0.0, "future_probe_accuracy": 0.0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = 0.0
    tokens = 0
    for step in range(count):
        batch = to_device(batches.batch(step), device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with autocast(device):
            output, losses = predictive_losses(
                model, kind, batch, objective, atlas_centers)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - started
        totals["ce"] += float(losses["ce"])
        totals["sparse_teacher"] += float(losses["sparse_teacher"])
        totals["teacher_top1"] += float((
            output.logits.argmax(-1)
            == batch["teacher_topk_indices"][..., 0].long()).float().mean())
        if kind in ("hierarchical_predictive", "hierarchical_logit_control"):
            totals["atom_accuracy"] += float((
                output.atom_logits.argmax(-1) == batch["atom_ids"].long()).float().mean())
            accuracies = []
            for logits, target in zip(output.future_atom_logits, batch["future_atom_ids"]):
                valid = target >= 0
                accuracies.append((logits.argmax(-1)[valid] == target[valid]).float().mean())
            totals["future_probe_accuracy"] += float(torch.stack(accuracies).mean())

        context = output.logits.shape[1]
        for end in (64, 128, 256, 512):
            if end > context:
                continue
            begin = 0 if end == 64 else end // 2
            logits = output.logits[:, begin:end]
            targets = batch["target_ids"][:, begin:end].long()
            value = float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))
            positions.setdefault(str(end), []).append(value)
        tokens += batch["target_ids"].numel()
    metrics = {key: value / count for key, value in totals.items()}
    if kind not in ("hierarchical_predictive", "hierarchical_logit_control"):
        metrics.pop("atom_accuracy")
        metrics.pop("future_probe_accuracy")
    metrics["rollout_ce"] = {
        key: sum(values) / len(values) for key, values in positions.items()
    }
    if "64" in metrics["rollout_ce"] and "512" in metrics["rollout_ce"]:
        metrics["rollout_degradation_64_to_512"] = (
            metrics["rollout_ce"]["512"] - metrics["rollout_ce"]["64"])
    metrics["tokens_per_second"] = tokens / max(elapsed, 1e-8)
    metrics["peak_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0)
    return metrics


def main():
    args = arguments()
    config = load_json(args.config)
    training = config["student_training"]
    student_config = config["students"][args.student]
    objective = config["objectives"][args.student]
    seed_everything(int(training["seed"]))
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", int(training.get("device", 0))) if torch.cuda.is_available() \
        else torch.device("cpu")
    atlas = torch.load(args.atlas, map_location="cpu", weights_only=False)
    atom_centers = atlas["centers"].to(device)
    budgeted = build_to_budget(
        args.student, int(config["vocab_size"]), int(training["context"]),
        student_config, int(config["student_parameter_budget"]))
    model = budgeted.model.to(device)
    train_batches = BakedBatches(
        args.train_manifest, int(training["batch_size"]), int(training["seed"]))
    val_batches = BakedBatches(
        args.val_manifest, int(training["eval_batch_size"]), int(training["seed"]) + 999)
    steps = args.steps or int(training["steps"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        betas=tuple(training.get("betas", (0.9, 0.95))),
        weight_decay=float(training.get("weight_decay", 0.1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(
            step, steps, int(training["warmup_steps"]),
            float(training.get("minimum_lr_ratio", 0.1))))
    checkpoint = output / "checkpoint.pt"
    start_step = 0
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])

    log_path = output / "train.jsonl"
    model.train()
    for step in range(start_step, steps):
        batch = to_device(train_batches.batch(step), device)
        with autocast(device):
            _, losses = predictive_losses(
                model, args.student, batch, objective, atom_centers)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        if step % int(training["log_every"]) == 0 or step + 1 == steps:
            record = {
                "event": "train", "step": step + 1,
                "learning_rate": scheduler.get_last_lr()[0],
                **{key: float(value.detach()) for key, value in losses.items()},
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if (step + 1) % int(training["checkpoint_every"]) == 0 or step + 1 == steps:
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({
                "step": step + 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            }, temporary)
            os.replace(temporary, checkpoint)

    metrics = evaluate(
        model, args.student, val_batches, atom_centers, objective, device,
        int(training["eval_batches"]))
    metrics["state_elements_at_512"] = model.stored_state_elements(
        int(training["eval_batch_size"]), 512)
    metrics["state_bytes_bf16_at_512"] = 2 * metrics["state_elements_at_512"]
    result = {
        "student": args.student, "width": budgeted.width,
        "stored_parameters": budgeted.parameters,
        "target_parameters": budgeted.target_parameters,
        "budget_utilization": budgeted.parameters / budgeted.target_parameters,
        "steps": steps, "metrics": metrics,
    }
    atomic_json(result, output / "result.json")
    temporary = output / "student_final.pt.tmp"
    torch.save({
        "model": model.state_dict(), "result": result,
        "student_config": student_config,
    }, temporary)
    os.replace(temporary, output / "student_final.pt")
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
