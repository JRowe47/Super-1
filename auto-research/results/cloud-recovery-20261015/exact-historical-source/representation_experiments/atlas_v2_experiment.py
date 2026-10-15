"""Train Atlas V2 against a frozen Qwen3-0.6B communication teacher.

The dense attention modules are retained only as a frozen training bypass.  All
milestone evaluations set the bypass to zero and therefore measure the
deployable functional operator.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from representation_experiments.atlas_v2_attention import (
    auxiliary_losses,
    clear_auxiliary,
    install_atlas_v2,
    set_functional_alpha,
)
from representation_experiments.common import (
    TokenMemmap,
    atomic_json,
    cosine_with_warmup,
    load_json,
    normalized_mse,
    seed_everything,
    symmetric_geometry_loss,
    teacher_kl,
)
from representation_experiments.operator_experiment import (
    AttentionCapture,
    autocast,
    evaluate,
    load_trainable_state,
    profile_forward,
    trainable_state,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--max-tokens", type=int, default=0,
        help="isolated smoke-test override; zero uses the configured token budget")
    parser.add_argument(
        "--skip-profile", action="store_true",
        help="skip expensive profiler measurement in isolated smoke tests")
    parser.add_argument(
        "--evaluation-contexts", type=int, nargs="+", default=None,
        help="isolated validation override; production uses the manifest")
    parser.add_argument(
        "--evaluation-batches", type=int, default=0,
        help="isolated validation override; production uses the manifest")
    return parser.parse_args()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(path: Path, step: int, model: nn.Module,
                    optimizer, scheduler, milestones: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "trainable": trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "milestones": milestones,
    }, temporary)
    os.replace(temporary, path)


def deployable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "_atlas_v2_future_probes" not in name
    }


def bypass_alpha(step: int, tokens_per_step: int, training: dict) -> float:
    start = float(training["bypass_alpha_start"])
    duration = max(1, math.ceil(
        int(training["bypass_tokens"]) / tokens_per_step))
    progress = min(1.0, (step + 1) / duration)
    return start + (1.0 - start) * progress


def attention_scale_loss(student_attention, teacher_attention):
    """Match communication magnitude without demanding identical coordinates."""
    if len(student_attention) != len(teacher_attention):
        raise RuntimeError("teacher and student emitted different attention depths")
    values = []
    for student, teacher in zip(student_attention, teacher_attention):
        student_rms = student.float().square().mean(dim=-1).clamp_min(1e-8).sqrt()
        teacher_rms = teacher.float().square().mean(dim=-1).clamp_min(1e-8).sqrt()
        values.append((student_rms.log() - teacher_rms.log()).square().mean())
    return torch.stack(values).mean()


def geometry_loss(student_output, teacher_output, layers: list[int]) -> torch.Tensor:
    losses = []
    for layer in layers:
        losses.append(symmetric_geometry_loss(
            student_output.hidden_states[layer],
            teacher_output.hidden_states[layer].detach()))
    return torch.stack(losses).mean()


def future_probe_loss(probes: nn.ModuleDict, student_hidden: torch.Tensor,
                      teacher_hidden: torch.Tensor, horizons: list[int],
                      stride: int) -> torch.Tensor:
    values = []
    for horizon in horizons:
        if horizon >= student_hidden.shape[1]:
            continue
        source = student_hidden[:, :-horizon:stride]
        target = teacher_hidden[:, horizon::stride].detach()
        values.append(normalized_mse(probes[str(horizon)](source), target))
    if not values:
        return student_hidden.sum() * 0.0
    return torch.stack(values).mean()


def corrupted_inputs(inputs: torch.Tensor, step: int, fraction: float,
                     replacement_token: int, seed: int) -> torch.Tensor:
    changed = inputs.clone()
    sequence = inputs.shape[1]
    width = max(1, min(sequence, round(sequence * fraction)))
    locations = max(1, sequence - width + 1)
    start = (int(seed) + 104729 * int(step)) % locations
    changed[:, start:start + width] = int(replacement_token)
    return changed


def counterfactual_loss(student, teacher, inputs, student_logits,
                        teacher_logits, step: int, training: dict,
                        replacement_token: int, device) -> torch.Tensor:
    changed = corrupted_inputs(
        inputs, step, float(training["intervention_fraction"]),
        replacement_token, int(training["seed"]))
    with torch.no_grad(), autocast(device):
        teacher_changed = teacher(input_ids=changed, use_cache=False).logits
    with autocast(device):
        student_changed = student(input_ids=changed, use_cache=False).logits
    stride = int(training["intervention_stride"])
    teacher_delta = teacher_changed[:, ::stride].float() - teacher_logits[:, ::stride].float()
    student_delta = student_changed[:, ::stride].float() - student_logits[:, ::stride].float()
    return normalized_mse(student_delta, teacher_delta.detach())


def milestone_result(teacher, student, installed, teacher_capture,
                     student_capture, validation, config, device, token_count,
                     training) -> dict:
    set_functional_alpha(installed, 1.0)
    metrics = evaluate(
        teacher, student, teacher_capture, student_capture, validation,
        [int(value) for value in config["evaluation"]["contexts"]],
        int(config["evaluation"]["batches"]),
        int(config["evaluation"]["batch_size"]), device,
        int(training["seed"]) + 100_000 + int(token_count),
        config["evaluation"]["diagnostic_weights"],
        int(training["logit_stride"]))
    reference = metrics[str(config["evaluation"]["reference_context"])]
    gate = config["gate"]
    ce_pass = reference["student_minus_teacher_ce"] <= float(gate["max_ce_gap"])
    return {
        "tokens": int(token_count),
        "metrics": metrics,
        "continue_gate": {
            "reference_context": int(config["evaluation"]["reference_context"]),
            "max_ce_gap": float(gate["max_ce_gap"]),
            "observed_ce_gap": reference["student_minus_teacher_ce"],
            "passed": bool(ce_pass),
        },
    }


def main():
    args = arguments()
    config = load_json(args.config)
    if args.evaluation_contexts:
        config["evaluation"]["contexts"] = [
            int(value) for value in args.evaluation_contexts]
        if int(config["evaluation"]["reference_context"]) not in \
                config["evaluation"]["contexts"]:
            config["evaluation"]["reference_context"] = int(
                config["evaluation"]["contexts"][0])
    if args.evaluation_batches:
        config["evaluation"]["batches"] = int(args.evaluation_batches)
    training = config["training"]
    seed_everything(int(training["seed"]))
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    device = torch.device("cuda", int(training.get("device", 0))) \
        if torch.cuda.is_available() else torch.device("cpu")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install requirements-representation.txt on the worker") from error

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
    installed = install_atlas_v2(
        student, config["atlas_v2"], int(training["context"]))

    hidden = int(student.config.hidden_size)
    horizons = [int(value) for value in training["future_horizons"]]
    probes = nn.ModuleDict({
        str(horizon): nn.Linear(hidden, hidden, bias=False)
        for horizon in horizons
    })
    for probe in probes.values():
        nn.init.eye_(probe.weight)
    student.add_module("_atlas_v2_future_probes", probes)

    teacher.to(device).eval()
    student.to(device)
    teacher_capture = AttentionCapture(teacher)
    student_capture = AttentionCapture(student)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    replacement_token = tokenizer.eos_token_id
    if replacement_token is None:
        replacement_token = tokenizer.pad_token_id or 0

    train_data = TokenMemmap.open(
        Path(config["data"]) / "train.bin", config["token_dtype"])
    validation = TokenMemmap.open(
        Path(config["data"]) / "val.bin", config["token_dtype"])
    tokens_per_step = int(training["context"]) * int(training["batch_size"])
    maximum_tokens = int(args.max_tokens or training["max_tokens"])
    steps = math.ceil(maximum_tokens / tokens_per_step)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Atlas V2 installation produced no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(training["learning_rate"]),
        betas=tuple(training["betas"]),
        weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(
            step, steps, int(training["warmup_steps"]),
            float(training["minimum_lr_ratio"])))

    checkpoint = output / "checkpoint.pt"
    start_step = 0
    milestones: list[dict] = []
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        load_trainable_state(student, payload["trainable"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])
        milestones = list(payload.get("milestones", []))

    configured_milestones = sorted(
        int(value) for value in config["milestones_tokens"]
        if int(value) <= maximum_tokens)
    completed_milestones = {int(value["tokens"]) for value in milestones}
    stopped_at_gate = False
    weights = training["loss_weights"]
    started_at = time.time()
    student.train()
    for step in range(start_step, steps):
        alpha = bypass_alpha(step, tokens_per_step, training)
        set_functional_alpha(installed, alpha)
        raw = train_data.batch_for_step(
            int(training["context"]), int(training["batch_size"]), step,
            int(training["seed"])).to(device)
        inputs, targets = raw[:, :-1], raw[:, 1:]

        teacher_capture.clear()
        student_capture.clear()
        clear_auxiliary(installed)
        with torch.no_grad(), autocast(device):
            teacher_output = teacher(
                input_ids=inputs, use_cache=False, output_hidden_states=True)
        with autocast(device):
            student_output = student(
                input_ids=inputs, use_cache=False, output_hidden_states=True)
            stride = int(training["logit_stride"])
            student_sample = student_output.logits[:, ::stride]
            teacher_sample = teacher_output.logits[:, ::stride].detach()
            losses = {
                "kl": teacher_kl(student_sample, teacher_sample),
                "ce": F.cross_entropy(
                    student_output.logits.float().reshape(
                        -1, student_output.logits.shape[-1]),
                    targets.reshape(-1)),
                "logit_nmse": normalized_mse(student_sample, teacher_sample),
                "geometry": geometry_loss(
                    student_output, teacher_output,
                    [int(value) for value in training["geometry_layers"]]),
                "attention_scale": attention_scale_loss(
                    student_capture.outputs, teacher_capture.outputs),
            }
            auxiliary = auxiliary_losses(installed)
            losses.update(auxiliary)
            if (step + 1) % int(training["future_every"]) == 0:
                losses["future"] = future_probe_loss(
                    probes, student_output.hidden_states[-1],
                    teacher_output.hidden_states[-1], horizons,
                    int(training["future_stride"]))
            else:
                losses["future"] = student_output.logits.sum() * 0.0

            if (step + 1) % int(training["intervention_every"]) == 0:
                teacher_capture.clear()
                student_capture.clear()
                clear_auxiliary(installed)
                losses["counterfactual"] = counterfactual_loss(
                    student, teacher, inputs, student_output.logits,
                    teacher_output.logits, step, training,
                    replacement_token, device)
            else:
                losses["counterfactual"] = student_output.logits.sum() * 0.0

            total = sum(
                float(weights[name]) * value
                for name, value in losses.items() if name in weights)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        token_count = min(maximum_tokens, (step + 1) * tokens_per_step)

        if ((step + 1) % int(training["log_every"]) == 0
                or step == 0 or step + 1 == steps):
            record = {
                "event": "train", "step": step + 1,
                "tokens": token_count, "functional_alpha": alpha,
                "loss": float(total.detach()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                **{name: float(value.detach()) for name, value in losses.items()},
            }
            append_jsonl(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

        reached = [
            value for value in configured_milestones
            if value not in completed_milestones and token_count >= value]
        for milestone_tokens in reached:
            milestone = milestone_result(
                teacher, student, installed, teacher_capture, student_capture,
                validation, config, device, milestone_tokens, training)
            milestones.append(milestone)
            completed_milestones.add(milestone_tokens)
            atomic_json(
                milestone, output / f"milestone-{milestone_tokens}.json")
            append_jsonl(log_path, {"event": "milestone", **milestone})
            print(json.dumps({"event": "milestone", **milestone}), flush=True)
            if (milestone_tokens == configured_milestones[0]
                    and not milestone["continue_gate"]["passed"]
                    and maximum_tokens > milestone_tokens):
                stopped_at_gate = True
                break
            student.train()
            set_functional_alpha(installed, bypass_alpha(
                step, tokens_per_step, training))
        if ((step + 1) % int(training["checkpoint_every"]) == 0
                or step + 1 == steps or stopped_at_gate):
            save_checkpoint(
                checkpoint, step + 1, student, optimizer, scheduler, milestones)
        if stopped_at_gate:
            break

    completed_steps = step + 1 if steps > start_step else start_step
    completed_tokens = min(maximum_tokens, completed_steps * tokens_per_step)
    if completed_tokens not in completed_milestones:
        milestone = milestone_result(
            teacher, student, installed, teacher_capture, student_capture,
            validation, config, device, completed_tokens, training)
        milestones.append(milestone)
        atomic_json(milestone, output / f"milestone-{completed_tokens}.json")

    reference_context = int(config["evaluation"]["reference_context"])
    profile_input = validation.batch_for_step(
        reference_context, 1, 0, int(training["seed"]) + 200_000
    )[:, :-1].to(device)
    set_functional_alpha(installed, 1.0)
    if args.skip_profile:
        teacher_profile = {"skipped": True}
        operator_profile = {"skipped": True}
        measured_flop_ratio = None
    else:
        teacher_capture.clear()
        student_capture.clear()
        teacher_profile = profile_forward(teacher, profile_input, device)
        operator_profile = profile_forward(student, profile_input, device)
        measured_flop_ratio = (
            operator_profile["flops"] / teacher_profile["flops"]
            if teacher_profile["flops"] > 0 else None)

    final_metrics = milestones[-1]["metrics"]
    reference = final_metrics[str(reference_context)]
    final_goal = config["final_goal"]
    goal_pass = (
        reference["student_minus_teacher_ce"] <= float(final_goal["max_ce_gap"])
        and reference["teacher_kl"] <= float(final_goal["max_teacher_kl"])
        and reference["top1_agreement"] >= float(final_goal["min_top1_agreement"])
    )
    result = {
        "experiment": "atlas_v2",
        "teacher": model_name,
        "decision": "stopped_at_10m_gate" if stopped_at_gate else (
            "final_goal_passed" if goal_pass else "final_goal_failed"),
        "steps": completed_steps,
        "tokens": completed_tokens,
        "wall_time_seconds_this_process": time.time() - started_at,
        "dense_attention_parameters": installed.dense_attention_parameters,
        "deployable_attention_parameters": installed.deployable_parameters,
        "parameter_ratio": (
            installed.deployable_parameters / installed.dense_attention_parameters),
        "estimated_dense_attention_flops": installed.estimated_dense_flops,
        "estimated_operator_flops": installed.estimated_operator_flops,
        "estimated_flop_ratio": (
            installed.estimated_operator_flops / installed.estimated_dense_flops),
        "teacher_profile": teacher_profile,
        "operator_profile": operator_profile,
        "measured_flop_ratio": measured_flop_ratio,
        "milestones": milestones,
        "final_metrics": final_metrics,
        "final_goal": {**final_goal, "passed": bool(goal_pass)},
        "training_only_probe_parameters": sum(
            parameter.numel() for parameter in probes.parameters()),
        "config": str(Path(args.config).resolve()),
    }
    atomic_json(result, output / "result.json")
    temporary = output / "operator_final.pt.tmp"
    torch.save({"operator": deployable_state(student), "result": result}, temporary)
    os.replace(temporary, output / "operator_final.pt")
    teacher_capture.close()
    student_capture.close()
    print("DONE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
