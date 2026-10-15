"""Train and evaluate one frozen-Qwen functional-attention replacement arm."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from representation_experiments.common import (
    TokenMemmap,
    atomic_json,
    cosine_with_warmup,
    load_json,
    normalized_mse,
    seed_everything,
    teacher_kl,
)
from representation_experiments.operator_attention import install_qwen_operators


class AttentionCapture:
    def __init__(self, model):
        self.outputs = []
        self.handles = [
            layer.self_attn.register_forward_hook(self._capture)
            for layer in model.model.layers
        ]

    def _capture(self, _module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        self.outputs.append(value)

    def clear(self):
        self.outputs.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True,
                        choices=("positive_kernel", "fourier_local", "wavelet_rbf", "atlas"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=0,
                        help="smoke-test override; zero uses the manifest")
    return parser.parse_args()


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else contextlib.nullcontext())


def trainable_state(model):
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def load_trainable_state(model, state):
    parameters = dict(model.named_parameters())
    missing = set(state) - set(parameters)
    if missing:
        raise ValueError(f"checkpoint parameters missing from model: {sorted(missing)[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


@torch.no_grad()
def profile_forward(model, inputs, device):
    """Measure profiler-counted inference FLOPs for one identical forward pass."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    model.eval()
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        with autocast(device):
            model(input_ids=inputs, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    flops = int(sum(int(event.flops or 0) for event in profiler.key_averages()))
    device_time_us = sum(
        float(getattr(event, "device_time_total", 0.0) or 0.0)
        for event in profiler.key_averages())
    cpu_time_us = sum(float(event.cpu_time_total) for event in profiler.key_averages())
    return {
        "flops": flops,
        "profiled_device_time_us": device_time_us,
        "profiled_cpu_time_us": cpu_time_us,
    }


def distillation_losses(student_output, teacher_output, student_attention,
                        teacher_attention, weights, logit_stride):
    if len(student_attention) != len(teacher_attention):
        raise RuntimeError("teacher and student emitted different attention depths")
    attention = torch.stack([
        normalized_mse(student, teacher.detach())
        for student, teacher in zip(student_attention, teacher_attention)
    ]).mean()
    student_hidden = student_output.hidden_states[1:]
    teacher_hidden = teacher_output.hidden_states[1:]
    residual = torch.stack([
        normalized_mse(student, teacher.detach())
        for student, teacher in zip(student_hidden, teacher_hidden)
    ]).mean()
    student_logits = student_output.logits[:, ::logit_stride]
    teacher_logits = teacher_output.logits[:, ::logit_stride]
    logits = normalized_mse(student_logits, teacher_logits.detach())
    kl = teacher_kl(student_logits, teacher_logits.detach())
    total = (
        float(weights["attention"]) * attention
        + float(weights["residual"]) * residual
        + float(weights["logits"]) * logits
        + float(weights["kl"]) * kl
    )
    return total, {
        "attention_nmse": float(attention.detach()),
        "residual_nmse": float(residual.detach()),
        "logit_nmse": float(logits.detach()),
        "teacher_kl": float(kl.detach()),
    }


@torch.no_grad()
def evaluate(teacher, student, teacher_capture, student_capture, memmap,
             contexts, batches, batch_size, device, seed, weights, logit_stride):
    result = {}
    teacher.eval()
    student.eval()
    for context in contexts:
        totals = {key: 0.0 for key in (
            "attention_nmse", "residual_nmse", "logit_nmse", "teacher_kl",
            "student_ce", "teacher_ce", "top1_agreement")}
        total_tokens = 0
        elapsed = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for raw in memmap.batches(context, batch_size, batches, seed + context,
                                  sequential=True):
            raw = raw.to(device)
            inputs, targets = raw[:, :-1], raw[:, 1:]
            teacher_capture.clear()
            student_capture.clear()
            with autocast(device):
                teacher_output = teacher(
                    input_ids=inputs, use_cache=False, output_hidden_states=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with autocast(device):
                student_output = student(
                    input_ids=inputs, use_cache=False, output_hidden_states=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            _, losses = distillation_losses(
                student_output, teacher_output, student_capture.outputs,
                teacher_capture.outputs, weights, logit_stride)
            for key, value in losses.items():
                totals[key] += value
            student_logits = student_output.logits.float()
            teacher_logits = teacher_output.logits.float()
            totals["student_ce"] += float(F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                targets.reshape(-1)))
            totals["teacher_ce"] += float(F.cross_entropy(
                teacher_logits.reshape(-1, teacher_logits.shape[-1]),
                targets.reshape(-1)))
            totals["top1_agreement"] += float(
                (student_logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean())
            total_tokens += targets.numel()
        metrics = {key: value / batches for key, value in totals.items()}
        metrics["student_minus_teacher_ce"] = metrics["student_ce"] - metrics["teacher_ce"]
        metrics["tokens_per_second"] = total_tokens / max(elapsed, 1e-8)
        metrics["peak_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0)
        metrics["tokens"] = total_tokens
        result[str(context)] = metrics
    return result


def main():
    args = arguments()
    config = load_json(args.config)
    arm_config = config["arms"][args.arm]
    training = config["training"]
    seed_everything(int(training["seed"]))
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", int(training.get("device", 0))) if torch.cuda.is_available() \
        else torch.device("cpu")

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("install requirements-representation.txt on the GPU worker") from error

    model_name = config["teacher"]
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    teacher.config.use_cache = False
    student.config.use_cache = False
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    installed = install_qwen_operators(
        student, args.arm, arm_config, int(training["context"]))
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("operator installation produced no trainable parameters")
    teacher.to(device).eval()
    student.to(device)
    teacher_capture = AttentionCapture(teacher)
    student_capture = AttentionCapture(student)

    train_data = TokenMemmap.open(
        Path(config["data"]) / "train.bin", config["token_dtype"])
    val_data = TokenMemmap.open(
        Path(config["data"]) / "val.bin", config["token_dtype"])
    steps = args.steps or int(training["steps"])
    optimizer = torch.optim.AdamW(
        trainable, lr=float(training["learning_rate"]),
        betas=tuple(training.get("betas", (0.9, 0.95))),
        weight_decay=float(training.get("weight_decay", 0.01)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(
            step, steps, int(training["warmup_steps"]),
            float(training.get("minimum_lr_ratio", 0.1))))
    checkpoint = output / "checkpoint.pt"
    start_step = 0
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        load_trainable_state(student, payload["operator"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])

    log_path = output / "train.jsonl"
    student.train()
    for step in range(start_step, steps):
        raw = train_data.batch_for_step(
            int(training["context"]), int(training["batch_size"]), step,
            int(training["seed"]))
        raw = raw.to(device)
        inputs = raw[:, :-1]
        teacher_capture.clear()
        student_capture.clear()
        with torch.no_grad(), autocast(device):
            teacher_output = teacher(
                input_ids=inputs, use_cache=False, output_hidden_states=True)
        with autocast(device):
            student_output = student(
                input_ids=inputs, use_cache=False, output_hidden_states=True)
            total, losses = distillation_losses(
                student_output, teacher_output, student_capture.outputs,
                teacher_capture.outputs, training["loss_weights"],
                int(training["logit_stride"]))
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        if step % int(training["log_every"]) == 0 or step + 1 == steps:
            record = {
                "event": "train", "step": step + 1, "loss": float(total.detach()),
                "learning_rate": scheduler.get_last_lr()[0], **losses,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if (step + 1) % int(training["checkpoint_every"]) == 0 or step + 1 == steps:
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({
                "step": step + 1, "operator": trainable_state(student),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            }, temporary)
            os.replace(temporary, checkpoint)

    metrics = evaluate(
        teacher, student, teacher_capture, student_capture, val_data,
        [int(value) for value in config["evaluation"]["contexts"]],
        int(config["evaluation"]["batches"]),
        int(config["evaluation"]["batch_size"]), device,
        int(training["seed"]) + 100_000, training["loss_weights"],
        int(training["logit_stride"]))
    profile_context = int(config["promotion_gates"]["reference_context"])
    profile_input = val_data.batch_for_step(
        profile_context, 1, 0, int(training["seed"]) + 200_000
    )[:, :-1].to(device)
    teacher_capture.clear()
    student_capture.clear()
    teacher_profile = profile_forward(teacher, profile_input, device)
    student_profile = profile_forward(student, profile_input, device)
    measured_flop_ratio = (
        student_profile["flops"] / teacher_profile["flops"]
        if teacher_profile["flops"] > 0 else None)
    result = {
        "arm": args.arm,
        "teacher": model_name,
        "steps": steps,
        "dense_attention_parameters": installed.dense_attention_parameters,
        "operator_parameters": installed.operator_parameters,
        "parameter_ratio": installed.operator_parameters / installed.dense_attention_parameters,
        "estimated_dense_attention_flops": installed.estimated_dense_flops,
        "estimated_operator_flops": installed.estimated_operator_flops,
        "estimated_flop_ratio": installed.estimated_operator_flops / installed.estimated_dense_flops,
        "teacher_profile": teacher_profile,
        "operator_profile": student_profile,
        "measured_flop_ratio": measured_flop_ratio,
        "metrics": metrics,
        "config": str(Path(args.config).resolve()),
    }
    atomic_json(result, output / "result.json")
    temporary = output / "operator_final.pt.tmp"
    torch.save({"operator": trainable_state(student), "result": result}, temporary)
    os.replace(temporary, output / "operator_final.pt")
    teacher_capture.close()
    student_capture.close()
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
