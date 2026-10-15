"""Compile frozen Qwen distributions into constant-state recurrent students."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from representation_experiments.common import (
    TokenMemmap,
    atomic_json,
    cosine_with_warmup,
    load_json,
    seed_everything,
)
from representation_experiments.predictive_models import build_to_budget


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} GPUs are visible")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if distributed:
        arguments = {"backend": backend}
        if device.type == "cuda":
            arguments["device_id"] = device
        dist.init_process_group(**arguments)
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError("torchrun environment and process group disagree")
    return DistributedContext(rank, local_rank, world_size, distributed, device)


def distributed_barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=0)
    return parser.parse_args()


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else contextlib.nullcontext())


def chunked_logsumexp(logits: torch.Tensor, temperature: float,
                      chunk_size: int) -> torch.Tensor:
    accumulator = None
    for start in range(0, logits.shape[-1], chunk_size):
        current = torch.logsumexp(
            logits[..., start:start + chunk_size].float() / temperature, dim=-1)
        accumulator = current if accumulator is None else torch.logaddexp(
            accumulator, current)
    if accumulator is None:
        raise ValueError("distribution has an empty vocabulary")
    return accumulator


def exact_forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     temperature: float, chunk_size: int) -> torch.Tensor:
    """Exact forward KL without constructing full-vocabulary probabilities."""
    teacher_logits = teacher_logits.detach()
    with torch.no_grad():
        teacher_lse = chunked_logsumexp(teacher_logits, temperature, chunk_size)
    student_lse = chunked_logsumexp(student_logits, temperature, chunk_size)
    expected_student = torch.zeros_like(student_lse)
    expected_teacher_logprob = torch.zeros_like(teacher_lse)
    for start in range(0, student_logits.shape[-1], chunk_size):
        teacher_scaled = (
            teacher_logits[..., start:start + chunk_size].float() / temperature)
        student_scaled = (
            student_logits[..., start:start + chunk_size].float() / temperature)
        teacher_probability = torch.exp(teacher_scaled - teacher_lse.unsqueeze(-1))
        expected_student = expected_student + (
            teacher_probability * student_scaled).sum(dim=-1)
        with torch.no_grad():
            expected_teacher_logprob = expected_teacher_logprob + (
                teacher_probability * (
                    teacher_scaled - teacher_lse.unsqueeze(-1))).sum(dim=-1)
    cross_entropy = student_lse - expected_student
    return (cross_entropy + expected_teacher_logprob).mean() * temperature * temperature


def _sequential_multi_temperature_kl(
        student_logits: torch.Tensor, teacher_logits: torch.Tensor,
        temperature_specs: list[dict], chunk_size: int):
    """Historical independent-temperature implementation used as a reference."""
    weighted = student_logits.new_zeros((), dtype=torch.float32)
    metrics = {}
    weight_sum = 0.0
    for spec in temperature_specs:
        temperature = float(spec["value"])
        weight = float(spec["weight"])
        value = exact_forward_kl(
            student_logits, teacher_logits, temperature, chunk_size)
        weighted = weighted + weight * value
        weight_sum += weight
        metrics[f"kl_t{temperature:g}"] = value
    if weight_sum <= 0:
        raise ValueError("temperature weights must sum to a positive value")
    return weighted / weight_sum, metrics


def fused_multi_temperature_kl(
        student_logits: torch.Tensor, teacher_logits: torch.Tensor,
        temperature_specs: list[dict], chunk_size: int):
    """Exact multi-temperature KL with two shared vocabulary traversals.

    ``multi_temperature_kl`` evaluates each temperature independently, so three
    configured temperatures traverse every vocabulary chunk nine times: twice
    for their log normalizers and once for their expectations.  This prototype
    preserves the per-temperature reduction order while moving the vocabulary
    loop outside the temperature loop.  Each teacher and student chunk is cast
    to float once in each of two passes and then reused by every temperature.

    The fallback retains the historical broadcasting/error behavior for
    non-matching logit shapes.  Production teacher/student logits have identical
    shapes and use the fused path.
    """
    if student_logits.shape != teacher_logits.shape:
        return _sequential_multi_temperature_kl(
            student_logits, teacher_logits, temperature_specs, chunk_size)

    temperatures = [float(spec["value"]) for spec in temperature_specs]
    weights = [float(spec["weight"]) for spec in temperature_specs]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("temperature weights must sum to a positive value")
    if student_logits.shape[-1] == 0:
        raise ValueError("distribution has an empty vocabulary")

    teacher_logits = teacher_logits.detach()
    teacher_lses = [None] * len(temperatures)
    student_lses = [None] * len(temperatures)
    for start in range(0, student_logits.shape[-1], chunk_size):
        teacher_chunk = teacher_logits[..., start:start + chunk_size].float()
        student_chunk = student_logits[..., start:start + chunk_size].float()
        for index, temperature in enumerate(temperatures):
            teacher_current = torch.logsumexp(
                teacher_chunk / temperature, dim=-1)
            student_current = torch.logsumexp(
                student_chunk / temperature, dim=-1)
            teacher_lses[index] = (
                teacher_current if teacher_lses[index] is None
                else torch.logaddexp(teacher_lses[index], teacher_current))
            student_lses[index] = (
                student_current if student_lses[index] is None
                else torch.logaddexp(student_lses[index], student_current))

    expected_students = [torch.zeros_like(value) for value in student_lses]
    expected_teacher_logprobs = [
        torch.zeros_like(value) for value in teacher_lses]
    for start in range(0, student_logits.shape[-1], chunk_size):
        teacher_chunk = teacher_logits[..., start:start + chunk_size].float()
        student_chunk = student_logits[..., start:start + chunk_size].float()
        for index, temperature in enumerate(temperatures):
            teacher_scaled = teacher_chunk / temperature
            student_scaled = student_chunk / temperature
            teacher_probability = torch.exp(
                teacher_scaled - teacher_lses[index].unsqueeze(-1))
            expected_students[index] = expected_students[index] + (
                teacher_probability * student_scaled).sum(dim=-1)
            expected_teacher_logprobs[index] = (
                expected_teacher_logprobs[index] + (
                    teacher_probability * (
                        teacher_scaled
                        - teacher_lses[index].unsqueeze(-1))).sum(dim=-1))

    weighted = student_logits.new_zeros((), dtype=torch.float32)
    metrics = {}
    for index, (temperature, weight) in enumerate(zip(temperatures, weights)):
        cross_entropy = student_lses[index] - expected_students[index]
        value = (
            cross_entropy + expected_teacher_logprobs[index]
        ).mean() * temperature * temperature
        weighted = weighted + weight * value
        metrics[f"kl_t{temperature:g}"] = value
    return weighted / weight_sum, metrics


def multi_temperature_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         temperature_specs: list[dict], chunk_size: int,
                         implementation: str = "sequential"):
    implementations = {
        "sequential": _sequential_multi_temperature_kl,
        "fused_two_pass": fused_multi_temperature_kl,
    }
    try:
        function = implementations[str(implementation)]
    except KeyError as error:
        raise ValueError(
            f"unknown multi-temperature KL implementation {implementation!r}; "
            f"expected one of {sorted(implementations)}") from error
    return function(
        student_logits, teacher_logits, temperature_specs, chunk_size)


def scheduled_weight(spec, progress: float) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    initial = float(spec["initial"])
    final = float(spec["final"])
    begin = float(spec.get("begin_fraction", 0.0))
    end = float(spec.get("end_fraction", 1.0))
    if not 0.0 <= begin <= end <= 1.0:
        raise ValueError("loss schedule fractions must satisfy 0 <= begin <= end <= 1")
    if progress <= begin:
        return initial
    if progress >= end or end == begin:
        return final
    ratio = (progress - begin) / (end - begin)
    curve = str(spec.get("curve", "linear"))
    if curve == "linear":
        interpolation = ratio
    elif curve == "cosine":
        interpolation = 0.5 - 0.5 * math.cos(math.pi * ratio)
    else:
        raise ValueError(f"unknown loss schedule curve: {curve}")
    return initial + interpolation * (final - initial)


def projected_cosine_loss(student_hidden: torch.Tensor,
                          teacher_hidden: torch.Tensor,
                          teacher_projection: torch.Tensor,
                          stride: int, horizon: int = 0) -> torch.Tensor:
    if stride <= 0:
        raise ValueError("representation stride must be positive")
    length = min(student_hidden.shape[1], teacher_hidden.shape[1])
    if horizon < 0 or horizon >= length:
        raise ValueError("future-state horizon is outside the training context")
    positions = torch.arange(
        0, length - horizon, stride, device=student_hidden.device)
    student_values = student_hidden.index_select(1, positions).float()
    teacher_positions = positions + int(horizon)
    teacher_values = teacher_hidden.index_select(1, teacher_positions).float()
    projected_teacher = torch.matmul(teacher_values, teacher_projection.float())
    return (1.0 - F.cosine_similarity(
        student_values, projected_teacher, dim=-1)).mean()


def auxiliary_representation_losses(student_hidden: torch.Tensor,
                                    teacher_hidden: torch.Tensor,
                                    teacher_projection: torch.Tensor,
                                    training: dict):
    losses = {}
    representation = training.get("representation_matching")
    if representation is not None:
        losses["representation"] = projected_cosine_loss(
            student_hidden, teacher_hidden, teacher_projection,
            int(representation["token_stride"]))
    future = training.get("future_state_prediction")
    if future is not None:
        values = [
            projected_cosine_loss(
                student_hidden, teacher_hidden, teacher_projection,
                int(future["token_stride"]), int(horizon))
            for horizon in future["horizons"]
        ]
        if not values:
            raise ValueError("future-state prediction needs at least one horizon")
        losses["future_state"] = torch.stack(values).mean()
    return losses


@torch.no_grad()
def initialize_from_teacher_embedding(student, teacher, projection_config: dict):
    teacher_weight = teacher.get_input_embeddings().weight.detach()
    if student.width > teacher_weight.shape[1]:
        raise ValueError("student width exceeds teacher embedding dimension")
    chunk_size = int(projection_config["chunk_size"])
    dimensions = int(teacher_weight.shape[1])
    rows = int(teacher_weight.shape[0])
    mean = torch.zeros(dimensions, device=teacher_weight.device, dtype=torch.float32)
    original_energy = torch.zeros((), device=teacher_weight.device, dtype=torch.float64)
    for start in range(0, rows, chunk_size):
        values = teacher_weight[start:start + chunk_size].float()
        mean += values.sum(dim=0)
        original_energy += values.double().square().sum()
    mean /= rows
    covariance = torch.zeros(
        dimensions, dimensions, device=teacher_weight.device, dtype=torch.float32)
    for start in range(0, rows, chunk_size):
        centered = teacher_weight[start:start + chunk_size].float() - mean
        covariance.addmm_(centered.t(), centered)
    covariance /= rows
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -student.width:]
    projected_energy = torch.zeros((), device=teacher_weight.device, dtype=torch.float64)
    for start in range(0, rows, chunk_size):
        centered = teacher_weight[start:start + chunk_size].float() - mean
        projected = centered @ components
        projected_energy += projected.double().square().sum()
    preserve_norm = bool(projection_config.get("preserve_global_norm", True))
    scale = (torch.sqrt(original_energy / projected_energy).float()
             if preserve_norm else torch.ones((), device=teacher_weight.device))
    for start in range(0, rows, chunk_size):
        centered = teacher_weight[start:start + chunk_size].float() - mean
        projected = (centered @ components) * scale
        student.embedding.weight[start:start + chunk_size].copy_(
            projected.to(student.embedding.weight.dtype))
    metadata = {
        "method": "centered_teacher_embedding_covariance_eigendecomposition",
        "teacher_dimensions": dimensions,
        "student_dimensions": student.width,
        "rows": rows,
        "preserve_global_norm": preserve_norm,
        "scale": float(scale),
        "retained_variance": float(
            eigenvalues[-student.width:].sum() / eigenvalues.clamp_min(0).sum()),
    }
    return metadata, components * scale


def sample_from_logits(logits: torch.Tensor, temperature: float,
                       generator: torch.Generator):
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def branch_candidates(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                      specs: list[dict], seed: int):
    generator = torch.Generator(device=teacher_logits.device)
    generator.manual_seed(int(seed))
    candidates = []
    for spec in specs:
        kind = spec["kind"]
        if kind == "teacher_top1":
            value = teacher_logits.argmax(dim=-1)
        elif kind == "teacher_sample":
            value = sample_from_logits(
                teacher_logits, float(spec["temperature"]), generator)
        elif kind == "teacher_rank":
            rank = int(spec["rank"])
            if rank < 1:
                raise ValueError("teacher_rank is one-indexed and must be positive")
            value = torch.topk(teacher_logits, rank, dim=-1).indices[:, rank - 1]
        elif kind == "student_sample":
            value = sample_from_logits(
                student_logits.detach(), float(spec["temperature"]), generator)
        else:
            raise ValueError(f"unknown branch candidate kind: {kind}")
        candidates.append(value)
    if not candidates:
        raise ValueError("branch candidate list is empty")
    return torch.stack(candidates, dim=1)


def student_last_logits(student, input_ids: torch.Tensor):
    if hasattr(student, "last_logits"):
        return student.last_logits(input_ids)
    return student(input_ids).logits[:, -1:]


def teacher_last_logits(teacher, input_ids: torch.Tensor):
    hidden = teacher.model(
        input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state[:, -1:]
    return teacher.lm_head(hidden)


def branching_transition_loss(student, teacher, original_inputs,
                              original_student_logits, original_teacher_logits,
                              training: dict, step: int, device):
    branch = training["branching"]
    minimum = int(branch["minimum_prefix"])
    maximum = min(int(branch["maximum_prefix"]), original_inputs.shape[1] - 1)
    if minimum > maximum:
        raise ValueError("branch prefix range is outside the training context")
    generator = np.random.default_rng(int(training["seed"]) + int(step))
    prefix_length = int(generator.integers(minimum, maximum + 1))
    teacher_at_branch = original_teacher_logits[:, prefix_length - 1]
    student_at_branch = original_student_logits[:, prefix_length - 1]
    candidates = branch_candidates(
        teacher_at_branch, student_at_branch, branch["candidates"],
        int(training["seed"]) + 1_000_003 * int(step))
    count = candidates.shape[1]
    prefix = original_inputs[:, :prefix_length].repeat_interleave(count, dim=0)
    branch_inputs = torch.cat((prefix, candidates.reshape(-1, 1)), dim=1)
    values = []
    metric_values = {}
    rollout_steps = int(branch["rollout_steps"])
    for depth in range(rollout_steps):
        with torch.no_grad(), autocast(device):
            teacher_logits = teacher_last_logits(teacher, branch_inputs)
        with autocast(device):
            student_logits = student_last_logits(student, branch_inputs)
        value, metrics = multi_temperature_kl(
            student_logits, teacher_logits, training["temperatures"],
            int(training["vocab_chunk_size"]),
            str(training.get("kl_implementation", "sequential")))
        values.append(value)
        for name, metric in metrics.items():
            metric_values.setdefault(name, []).append(metric)
        if depth + 1 < rollout_steps:
            next_tokens = teacher_logits[:, -1].argmax(dim=-1, keepdim=True)
            branch_inputs = torch.cat((branch_inputs, next_tokens), dim=1)
    averaged = torch.stack(values).mean()
    metrics = {
        f"branch_{name}": torch.stack(items).mean()
        for name, items in metric_values.items()
    }
    metrics["branch_prefix_length"] = averaged.new_tensor(float(prefix_length))
    return averaged, metrics


@torch.no_grad()
def evaluate(student, teacher, val_data, config, arm, device, batches):
    training = config["training"]
    evaluation = config["evaluation"]
    context = int(evaluation["context"])
    batch_size = int(evaluation["batch_size"])
    totals = {
        "student_ce": 0.0, "teacher_ce": 0.0, "top1_agreement": 0.0,
    }
    for spec in evaluation["temperatures"]:
        totals[f"kl_t{float(spec['value']):g}"] = 0.0
    branch_totals = {}
    position_values = {}
    elapsed = 0.0
    tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    student.eval()
    teacher.eval()
    for step in range(batches):
        raw = val_data.batch_for_step(
            context, batch_size, step, int(evaluation["seed"])).to(device)
        inputs, targets = raw[:, :-1], raw[:, 1:]
        with autocast(device):
            teacher_logits = teacher(
                input_ids=inputs, use_cache=False, return_dict=True).logits
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with autocast(device):
            student_output = student(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - started
        student_logits = student_output.logits
        totals["student_ce"] += float(F.cross_entropy(
            student_logits.float().reshape(-1, student_logits.shape[-1]),
            targets.reshape(-1)))
        totals["teacher_ce"] += float(F.cross_entropy(
            teacher_logits.float().reshape(-1, teacher_logits.shape[-1]),
            targets.reshape(-1)))
        totals["top1_agreement"] += float(
            (student_logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean())
        _, kl_metrics = multi_temperature_kl(
            student_logits, teacher_logits, evaluation["temperatures"],
            int(training["vocab_chunk_size"]),
            str(training.get("kl_implementation", "sequential")))
        for name, value in kl_metrics.items():
            totals[name] += float(value)
        for end in evaluation["rollout_positions"]:
            end = int(end)
            if end > context:
                continue
            begin = 0 if end == min(evaluation["rollout_positions"]) else end // 2
            logits = student_logits[:, begin:end]
            selected_targets = targets[:, begin:end]
            position_values.setdefault(str(end), []).append(float(F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                selected_targets.reshape(-1))))
        if step < int(evaluation["branch_batches"]):
            _, metrics = branching_transition_loss(
                student, teacher, inputs, student_logits, teacher_logits,
                training, int(evaluation["seed"]) + step, device)
            for name, value in metrics.items():
                if name == "branch_prefix_length":
                    continue
                branch_totals[name] = branch_totals.get(name, 0.0) + float(value)
        tokens += targets.numel()
    result = {name: value / batches for name, value in totals.items()}
    result["student_minus_teacher_ce"] = result["student_ce"] - result["teacher_ce"]
    result["rollout_ce"] = {
        name: sum(values) / len(values) for name, values in position_values.items()
    }
    positions = [str(int(value)) for value in evaluation["rollout_positions"]]
    if positions[0] in result["rollout_ce"] and positions[-1] in result["rollout_ce"]:
        result["rollout_degradation"] = (
            result["rollout_ce"][positions[-1]] - result["rollout_ce"][positions[0]])
    if int(evaluation["branch_batches"]) > 0:
        result.update({
            name: value / int(evaluation["branch_batches"])
            for name, value in branch_totals.items()
        })
    result["tokens_per_second"] = tokens / max(elapsed, 1e-8)
    result["peak_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0)
    long_context_metrics = {}
    long_batches = int(evaluation.get("long_context_batches", 0))
    for long_context_value in evaluation.get("long_contexts", []):
        long_context = int(long_context_value)
        if long_context <= context or long_batches <= 0:
            continue
        long_totals = {
            "student_ce": 0.0,
            "teacher_ce": 0.0,
            "top1_agreement": 0.0,
            "kl_t1": 0.0,
        }
        long_batch_size = int(evaluation.get("long_context_batch_size", 1))
        for step in range(long_batches):
            raw = val_data.batch_for_step(
                long_context, long_batch_size, step,
                int(evaluation["seed"]) + 10_000_019 * long_context).to(device)
            inputs, targets = raw[:, :-1], raw[:, 1:]
            with autocast(device):
                teacher_logits = teacher(
                    input_ids=inputs, use_cache=False, return_dict=True).logits
                student_logits = student(inputs).logits
            long_totals["student_ce"] += float(F.cross_entropy(
                student_logits.float().reshape(-1, student_logits.shape[-1]),
                targets.reshape(-1)))
            long_totals["teacher_ce"] += float(F.cross_entropy(
                teacher_logits.float().reshape(-1, teacher_logits.shape[-1]),
                targets.reshape(-1)))
            long_totals["top1_agreement"] += float(
                (student_logits.argmax(-1) == teacher_logits.argmax(-1))
                .float().mean())
            long_totals["kl_t1"] += float(exact_forward_kl(
                student_logits, teacher_logits, 1.0,
                int(training["vocab_chunk_size"])))
        values = {
            name: value / long_batches for name, value in long_totals.items()
        }
        values["student_minus_teacher_ce"] = (
            values["student_ce"] - values["teacher_ce"])
        long_context_metrics[str(long_context)] = values
    result["long_context"] = long_context_metrics
    return result


def save_checkpoint(path, step, model, optimizer, scheduler, projection_metadata):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "embedding_projection": projection_metadata,
    }, temporary)
    os.replace(temporary, path)


def main():
    args = arguments()
    config = load_json(args.config)
    if args.arm not in config["arms"]:
        raise ValueError(f"arm is not in config: {args.arm}")
    arm = config["arms"][args.arm]
    training = config["training"]
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    final_path = output / "student_final.pt"
    if result_path.exists() and final_path.exists():
        print("SKIP", result_path, flush=True)
        return
    if result_path.exists() and not final_path.exists():
        raise RuntimeError(
            "result.json exists without student_final.pt; preserve the partial "
            "finalization and recover it explicitly")
    if final_path.exists() and not result_path.exists():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        recovered = payload.get("result")
        if not isinstance(recovered, dict) or not recovered:
            raise RuntimeError(
                "student_final.pt exists without a recoverable result payload")
        atomic_json(recovered, result_path)
        print("RECOVERED", result_path, flush=True)
        return
    distributed = setup_distributed()
    seed_everything(int(training["seed"]))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = distributed.device
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("install requirements-representation.txt on the GPU worker") from error

    teacher = AutoModelForCausalLM.from_pretrained(
        config["teacher"], torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to(device).eval()
    teacher.config.use_cache = False
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    architecture = arm["architecture"]
    model_max_context = int(config.get("model_max_context", training["context"]))
    parameter_budget = int(arm.get(
        "parameter_budget", config["student_parameter_budget"]))
    budgeted = build_to_budget(
        architecture, int(config["vocab_size"]), model_max_context,
        config["models"][architecture], parameter_budget)
    base_student = budgeted.model.to(device)
    checkpoint = output / "checkpoint.pt"
    projection_metadata = {"method": "random_initialization"}
    projection_path = output / "teacher_projection.pt"
    auxiliary_enabled = any(
        name in training for name in (
            "representation_matching", "future_state_prediction"))
    teacher_projection = None
    if not checkpoint.exists() and arm["embedding_initialization"] == "qwen_pca":
        projection_metadata, teacher_projection = initialize_from_teacher_embedding(
            base_student, teacher, config["embedding_projection"])
        if auxiliary_enabled and distributed.primary:
            temporary_projection = projection_path.with_suffix(".tmp")
            torch.save(teacher_projection.detach().cpu(), temporary_projection)
            os.replace(temporary_projection, projection_path)
    elif auxiliary_enabled:
        if projection_path.exists():
            teacher_projection = torch.load(
                projection_path, map_location=device, weights_only=True)
        elif arm["embedding_initialization"] == "qwen_pca":
            _, teacher_projection = initialize_from_teacher_embedding(
                base_student, teacher, config["embedding_projection"])
            if distributed.primary:
                temporary_projection = projection_path.with_suffix(".tmp")
                torch.save(teacher_projection.detach().cpu(), temporary_projection)
                os.replace(temporary_projection, projection_path)
        else:
            raise ValueError(
                "representation objectives require qwen_pca initialization")
    distributed_barrier(distributed)
    if teacher_projection is not None:
        teacher_projection = teacher_projection.to(device)
    if distributed.primary:
        atomic_json({
            "arm": args.arm,
            "architecture": architecture,
            "width": budgeted.width,
            "stored_parameters": budgeted.parameters,
            "embedding_projection": projection_metadata,
            "world_size": distributed.world_size,
            "config": config,
        }, output / "resolved_config.json")

    optimizer = torch.optim.AdamW(
        base_student.parameters(), lr=float(training["learning_rate"]),
        betas=tuple(training["betas"]), weight_decay=float(training["weight_decay"]))
    train_data = TokenMemmap.open(
        Path(config["data"]) / "train.bin", config["token_dtype"])
    val_data = TokenMemmap.open(
        Path(config["data"]) / "val.bin", config["token_dtype"])
    tokens_per_step = (
        int(training["context"]) * int(training["batch_size"])
        * distributed.world_size)
    sampling = str(training.get("sampling", "restart_stable_random"))
    if sampling not in ("restart_stable_random", "non_replacement_blocks"):
        raise ValueError(f"unknown training sampling mode: {sampling}")
    if args.steps:
        steps = int(args.steps)
        training_target_tokens = steps * tokens_per_step
        training_blocks = (
            steps * int(training["batch_size"]) * distributed.world_size
            if sampling == "non_replacement_blocks" else None)
    elif sampling == "non_replacement_blocks":
        context_tokens = int(training["context"])
        available_blocks = (len(train_data.tokens) - 1) // context_tokens
        requested_blocks = int(training["max_tokens"]) // context_tokens
        training_blocks = min(requested_blocks, available_blocks)
        if training_blocks % distributed.world_size:
            raise ValueError(
                "non-replacement block count must divide across distributed ranks")
        training_target_tokens = training_blocks * context_tokens
        sequences_per_step = int(training["batch_size"]) * distributed.world_size
        steps = math.ceil(training_blocks / sequences_per_step)
        if steps < 1:
            raise ValueError("token budget is smaller than one distributed step")
    else:
        steps = math.ceil(int(training["max_tokens"]) / tokens_per_step)
        training_target_tokens = steps * tokens_per_step
        training_blocks = None
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(
            step, steps, int(training["warmup_steps"]),
            float(training["minimum_lr_ratio"])))
    start_step = 0
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        base_student.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        projection_metadata = payload.get("embedding_projection", projection_metadata)
        start_step = int(payload["step"])
    student = (DDP(
        base_student, device_ids=[distributed.local_rank],
        output_device=distributed.local_rank, broadcast_buffers=False)
        if distributed.distributed and device.type == "cuda"
        else DDP(base_student, broadcast_buffers=False)
        if distributed.distributed else base_student)
    log_path = output / "train.jsonl"
    student.train()
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        if sampling == "non_replacement_blocks":
            completed_blocks = (
                step * int(training["batch_size"]) * distributed.world_size)
            remaining_blocks = int(training_blocks) - completed_blocks
            local_batch_size = min(
                int(training["batch_size"]),
                max(0, math.ceil(
                    (remaining_blocks - distributed.rank)
                    / distributed.world_size)))
            if local_batch_size < 1:
                raise RuntimeError(
                    "the final non-replacement step left a rank without data")
            raw = train_data.affine_block_batch_for_step(
                int(training["context"]), local_batch_size, step,
                int(training["seed"]), distributed.rank,
                distributed.world_size).to(device)
        else:
            raw = train_data.batch_for_step(
                int(training["context"]), int(training["batch_size"]),
                step * distributed.world_size + distributed.rank,
                int(training["seed"])).to(device)
        inputs, targets = raw[:, :-1], raw[:, 1:]
        with torch.no_grad(), autocast(device):
            if auxiliary_enabled:
                teacher_hidden = teacher.model(
                    input_ids=inputs, use_cache=False,
                    return_dict=True).last_hidden_state
                teacher_logits = teacher.lm_head(teacher_hidden)
            else:
                teacher_hidden = None
                teacher_logits = teacher(
                    input_ids=inputs, use_cache=False,
                    return_dict=True).logits
        with autocast(device):
            student_output = student(inputs)
        student_logits = student_output.logits
        ce = F.cross_entropy(
            student_logits.float().reshape(-1, student_logits.shape[-1]),
            targets.reshape(-1))
        distribution, kl_metrics = multi_temperature_kl(
            student_logits, teacher_logits, training["temperatures"],
            int(training["vocab_chunk_size"]),
            str(training.get("kl_implementation", "sequential")))
        progress = (step + 1) / steps
        ce_weight = scheduled_weight(training["ce_weight"], progress)
        distribution_weight = scheduled_weight(
            training["distribution_weight"], progress)
        total = ce_weight * ce + distribution_weight * distribution
        auxiliary_losses = {}
        auxiliary_weights = {}
        if auxiliary_enabled:
            if teacher_projection is None:
                raise RuntimeError("teacher projection was not initialized")
            if teacher_hidden is None:
                raise RuntimeError("teacher hidden state was not retained")
            auxiliary_losses = auxiliary_representation_losses(
                student_output.hidden_states, teacher_hidden,
                teacher_projection, training)
            for name, value in auxiliary_losses.items():
                config_name = (
                    "representation_matching" if name == "representation"
                    else "future_state_prediction")
                weight = scheduled_weight(
                    training[config_name]["weight"], progress)
                auxiliary_weights[name] = weight
                total = total + weight * value
            del teacher_hidden
        branch_metrics = {}
        if bool(arm["branching"]) and step % int(training["branching"]["every"]) == 0:
            branch_value, branch_metrics = branching_transition_loss(
                student, teacher, inputs, student_logits, teacher_logits,
                training, step, device)
            total = total + float(training["branching"]["weight"]) * branch_value
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        if distributed.primary and (
                step % int(training["log_every"]) == 0 or step + 1 == steps):
            record = {
                "event": "train", "step": step + 1,
                "tokens": min((step + 1) * tokens_per_step,
                              training_target_tokens),
                "learning_rate": scheduler.get_last_lr()[0],
                "loss": float(total.detach()), "ce": float(ce.detach()),
                "distribution": float(distribution.detach()),
                "ce_weight": ce_weight,
                "distribution_weight": distribution_weight,
                "gradient_norm": float(gradient_norm),
                **{name: float(value.detach()) for name, value in kl_metrics.items()},
                **{
                    f"{name}_loss": float(value.detach())
                    for name, value in auxiliary_losses.items()
                },
                **{
                    f"{name}_weight": float(value)
                    for name, value in auxiliary_weights.items()
                },
                **{name: float(value.detach()) for name, value in branch_metrics.items()},
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if distributed.primary and (
                (step + 1) % int(training["checkpoint_every"]) == 0
                or step + 1 == steps):
            save_checkpoint(
                checkpoint, step + 1, base_student, optimizer, scheduler,
                projection_metadata)

    distributed_barrier(distributed)
    if distributed.primary:
        eval_batches = args.eval_batches or int(config["evaluation"]["batches"])
        metrics = evaluate(
            base_student, teacher, val_data, config, arm, device, eval_batches)
        metrics["state_elements_per_sequence_at_512"] = (
            base_student.stored_state_elements(1, 512))
        metrics["state_bytes_bf16_per_sequence_at_512"] = (
            2 * metrics["state_elements_per_sequence_at_512"])
        metrics["estimated_macs_per_token_at_512"] = (
            base_student.estimated_macs_per_token(512))
        for context_value in config["evaluation"].get("long_contexts", []):
            context_value = int(context_value)
            metrics[f"state_bytes_bf16_per_sequence_at_{context_value}"] = (
                2 * base_student.stored_state_elements(1, context_value))
            metrics[f"estimated_macs_per_token_at_{context_value}"] = (
                base_student.estimated_macs_per_token(context_value))
        result = {
            "arm": args.arm,
            "architecture": architecture,
            "embedding_initialization": arm["embedding_initialization"],
            "branching": bool(arm["branching"]),
            "width": budgeted.width,
            "stored_parameters": budgeted.parameters,
            "target_parameters": budgeted.target_parameters,
            "budget_utilization": budgeted.parameters / budgeted.target_parameters,
            "steps": steps,
            "tokens": training_target_tokens,
            "sampling": sampling,
            "world_size": distributed.world_size,
            "embedding_projection": projection_metadata,
            "metrics": metrics,
        }
        temporary = output / "student_final.pt.tmp"
        torch.save({
            "model": base_student.state_dict(), "result": result,
            "model_config": config["models"][architecture],
        }, temporary)
        os.replace(temporary, final_path)
        atomic_json(result, result_path)
        print("DONE", json.dumps(result), flush=True)
    distributed_barrier(distributed)
    if distributed.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
