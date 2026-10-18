#!/usr/bin/env python3
"""Read-only CPU gradient ranking and scalar-delta screening for a promoted W560 winner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mean(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise RuntimeError("loss inventory is empty")
    return torch.stack(values).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--wave-root", required=True, type=Path)
    parser.add_argument("--verification-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--winner-candidate-order", required=True, type=int)
    parser.add_argument("--batch-indices", default="0")
    parser.add_argument("--sequences-per-batch", default=1, type=int)
    parser.add_argument("--token-limit", default=65, type=int)
    parser.add_argument("--rank-top-k", default=100, type=int)
    parser.add_argument("--screen-top-k-per-mode", default=4, type=int)
    parser.add_argument("--cpu-threads", default=4, type=int)
    args = parser.parse_args()

    if (
        args.token_limit < 2
        or args.sequences_per_batch <= 0
        or args.rank_top_k <= 0
        or args.screen_top_k_per_mode < 0
    ):
        raise RuntimeError("invalid positive pilot bounds")
    batch_indices = [int(value) for value in args.batch_indices.split(",")]
    if not batch_indices or len(set(batch_indices)) != len(batch_indices):
        raise RuntimeError("batch indices must be a unique nonempty list")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)

    package_root = args.package_root.resolve()
    wave_root = args.wave_root.resolve()
    verification_root = args.verification_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = package_root / "runtime"
    sys.path.insert(0, str(runtime_root))

    from hfw_dist.common import hash_file, read_json
    from hfw_dist.ledger import DistributedLedger
    from hfw_dist.model_io import build_loaded_parent
    from hfw_dist.promote_mps import _load_vendor

    started_ns = time.time_ns()
    campaign, campaign_identity = read_json(wave_root / "snapshot/CAMPAIGN.json")
    parent_record, parent_identity = read_json(verification_root / "PARENT.json")
    result_paths = sorted(
        (verification_root / "candidate-results").glob(
            f"{args.winner_candidate_order:06d}-*.json"
        )
    )
    if len(result_paths) != 1:
        raise RuntimeError("winner candidate result is not unique")
    winner, winner_identity = read_json(result_paths[0])
    if winner.get("passes_minimum_replicated_improvement") is not True:
        raise RuntimeError("winner is not a duplicated canonical strict improver")
    group = str(winner["group"])

    frozen = runtime_root / "frozen"
    ledger = DistributedLedger(runtime_root / "authority/lineage")
    current_pointer, current_pointer_identity = read_json(ledger.root / "CURRENT.json")
    current_head, current_head_identity = read_json(
        Path(str(current_pointer["record_path"])), str(current_pointer["record_sha256"])
    )
    if int(current_head["model_generation"]) != 10:
        raise RuntimeError("pilot requires the bound G10 head")
    if current_head.get("replacement_group_flat_sha256") != winner["candidate_group_flat_sha256"]:
        raise RuntimeError("G10 is not the selected winner materialization")

    source_head_sha = str(parent_record["base_head_record_sha256"])
    source_head_paths = list((ledger.root / "heads").glob(f"*-{source_head_sha}.head.json"))
    if len(source_head_paths) != 1:
        raise RuntimeError("winner source head is not unique")
    source_head, source_head_identity = read_json(source_head_paths[0], source_head_sha)
    branch, bridge, vendor_manifest = _load_vendor(frozen)

    print("LOAD source checkpoint", flush=True)
    loaded_source, _ = build_loaded_parent(
        frozen,
        Path(str(source_head["checkpoint"]["path"])),
        str(source_head["checkpoint"]["sha256"]),
        str(source_head["model_state_digest"]),
        torch.device("cpu"),
        bridge,
        vendor_manifest,
    )
    source_parts = {
        name: value.detach().cpu().float().clone()
        for name, value in loaded_source.model.named_parameters()
        if name in loaded_source.groups[group]
    }
    del loaded_source

    print("LOAD current checkpoint", flush=True)
    loaded_current, _ = build_loaded_parent(
        frozen,
        Path(str(current_head["checkpoint"]["path"])),
        str(current_head["checkpoint"]["sha256"]),
        str(current_head["model_state_digest"]),
        torch.device("cpu"),
        bridge,
        vendor_manifest,
    )
    group_names = tuple(loaded_current.groups[group])
    current_parameters = {
        name: value.detach().cpu()
        for name, value in loaded_current.model.named_parameters()
    }
    current_buffers = {
        name: value.detach().cpu().clone()
        for name, value in loaded_current.model.named_buffers()
    }
    current_flat = torch.cat(
        [current_parameters[name].reshape(-1).float() for name in group_names]
    )
    if hashlib.sha256(branch._tensor_bytes(current_flat)).hexdigest() != winner[
        "candidate_group_flat_sha256"
    ]:
        raise RuntimeError("G10 group bytes differ from winner bytes")
    direction_parts = {
        name: current_parameters[name].float() - source_parts[name]
        for name in group_names
    }
    source_parameters = dict(current_parameters)
    source_parameters.update(source_parts)
    direction_flat = torch.cat([direction_parts[name].reshape(-1) for name in group_names])
    nonzero = direction_flat != 0
    if not torch.any(nonzero) or not torch.isfinite(direction_flat).all():
        raise RuntimeError("winner scalar delta is zero or nonfinite")

    verify_shape = tuple(int(value) for value in campaign["promotion"]["verify_bank_shape"])
    if any(index < 0 or index >= verify_shape[0] for index in batch_indices):
        raise RuntimeError("pilot batch index is outside VERIFY bank")
    if args.token_limit > verify_shape[2]:
        raise RuntimeError("token limit exceeds VERIFY sequence length")
    if args.sequences_per_batch > verify_shape[1]:
        raise RuntimeError("sequence count exceeds VERIFY batch width")
    verify_identity = hash_file(
        frozen / "verify.u32le", str(campaign["promotion"]["verify_bank_sha256"])
    )
    verify = np.memmap(frozen / "verify.u32le", mode="r", dtype="<u4", shape=verify_shape)
    batches = [
        torch.from_numpy(
            np.array(
                verify[index, : args.sequences_per_batch, : args.token_limit],
                dtype=np.int64,
                copy=True,
            )
        )
        for index in batch_indices
    ]

    manifest = {
        "schema": "hfw_w560_scalar_delta_cpu_pilot_manifest_v1",
        "status": "BOUND_CPU_FP32_SHADOW_AUTOGRAD_PILOT",
        "created_unix_ns": started_ns,
        "campaign": campaign_identity,
        "winner_result": winner_identity,
        "winner_candidate_order": args.winner_candidate_order,
        "winner_task_index": int(winner["task_index"]),
        "group": group,
        "source_head": source_head_identity,
        "source_checkpoint_sha256": source_head["checkpoint"]["sha256"],
        "current_pointer": current_pointer_identity,
        "current_head": current_head_identity,
        "current_checkpoint_sha256": current_head["checkpoint"]["sha256"],
        "verify_bank": verify_identity,
        "verify_bank_shape": verify_shape,
        "batch_indices": batch_indices,
        "sequences_per_batch": args.sequences_per_batch,
        "token_limit": args.token_limit,
        "rank_top_k": args.rank_top_k,
        "screen_top_k_per_mode": args.screen_top_k_per_mode,
        "cpu_threads": args.cpu_threads,
        "scalar_count": int(direction_flat.numel()),
        "nonzero_scalar_delta_count": int(nonzero.sum()),
        "canonical_promotion_authority": False,
        "cpu_fp32_shadow_numerically_comparable_to_canonical_mps_bf16": False,
        "local_verify_reused_optimization_data_not_held_out": True,
        "checkpoint_or_head_writes_authorized": False,
    }
    atomic_json(output_root / "MANIFEST.json", manifest)
    print(
        f"BOUND group={group} scalars={direction_flat.numel()} "
        f"nonzero={int(nonzero.sum())} batches={batch_indices} tokens={args.token_limit}",
        flush=True,
    )

    gradient_parameters = dict(current_parameters)
    gradient_leaves = {}
    for name in group_names:
        leaf = current_parameters[name].detach().clone().requires_grad_(True)
        gradient_parameters[name] = leaf
        gradient_leaves[name] = leaf

    def losses_for(
        parameters: dict[str, torch.Tensor], selected_batches: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        losses = []
        for raw in selected_batches:
            inputs, targets = raw[:, :-1], raw[:, 1:]
            output = torch.func.functional_call(
                loaded_current.model,
                (parameters, dict(current_buffers)),
                (inputs,),
                strict=True,
            )
            token_losses = F.cross_entropy(
                output.logits.float().reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            )
            losses.append(token_losses.mean())
        return losses

    print("GRADIENT start", flush=True)
    gradient_started_ns = time.time_ns()
    pilot_loss = mean(losses_for(gradient_parameters, batches))
    if not torch.isfinite(pilot_loss):
        raise RuntimeError("pilot loss is nonfinite")
    pilot_loss.backward()
    unused_gradient_tensors = [
        name for name in group_names if gradient_leaves[name].grad is None
    ]
    gradient_flat = torch.cat(
        [
            (
                torch.zeros_like(gradient_leaves[name])
                if gradient_leaves[name].grad is None
                else gradient_leaves[name].grad.detach()
            ).reshape(-1).float()
            for name in group_names
        ]
    )
    if not torch.isfinite(gradient_flat).all():
        raise RuntimeError("pilot gradient is nonfinite")
    contribution = gradient_flat * direction_flat
    extension_improvement = -contribution
    ablation_improvement = contribution
    valid_indices = torch.nonzero(nonzero, as_tuple=False).flatten()

    spans = []
    cursor = 0
    for name in group_names:
        shape = tuple(int(value) for value in current_parameters[name].shape)
        count = int(current_parameters[name].numel())
        spans.append((cursor, cursor + count, name, shape))
        cursor += count
    if cursor != direction_flat.numel():
        raise RuntimeError("parameter span inventory drift")

    def describe(flat_index: int, score: float, mode: str) -> dict[str, object]:
        for start, end, parameter_name, shape in spans:
            if start <= flat_index < end:
                local = flat_index - start
                multi = list(np.unravel_index(local, shape))
                return {
                    "mode": mode,
                    "predicted_improvement": score,
                    "flat_index": flat_index,
                    "parameter_name": parameter_name,
                    "parameter_local_flat_index": local,
                    "parameter_multi_index": [int(value) for value in multi],
                    "source_value": float(current_flat[flat_index] - direction_flat[flat_index]),
                    "current_value": float(current_flat[flat_index]),
                    "delta": float(direction_flat[flat_index]),
                    "gradient": float(gradient_flat[flat_index]),
                    "gradient_times_delta": float(contribution[flat_index]),
                }
        raise RuntimeError("flat index outside parameter spans")

    top_count = min(args.rank_top_k, int(valid_indices.numel()))
    extension_values, extension_positions = torch.topk(
        extension_improvement[valid_indices], top_count
    )
    ablation_values, ablation_positions = torch.topk(
        ablation_improvement[valid_indices], top_count
    )
    extension_rows = [
        describe(int(valid_indices[position]), float(score), "extend_same_delta")
        for score, position in zip(extension_values, extension_positions)
    ]
    ablation_rows = [
        describe(int(valid_indices[position]), float(score), "ablate_to_source")
        for score, position in zip(ablation_values, ablation_positions)
    ]
    absolute_values, absolute_positions = torch.topk(
        torch.abs(contribution[valid_indices]), top_count
    )
    absolute_rows = [
        describe(int(valid_indices[position]), float(score), "absolute_local_contribution")
        for score, position in zip(absolute_values, absolute_positions)
    ]
    gradient_norm = float(torch.linalg.vector_norm(gradient_flat))
    direction_norm = float(torch.linalg.vector_norm(direction_flat))
    if gradient_norm == 0.0 or direction_norm == 0.0:
        raise RuntimeError("gradient or direction norm is zero")
    dot = float(torch.dot(gradient_flat, direction_flat))
    ranking = {
        "schema": "hfw_w560_scalar_delta_cpu_fp32_shadow_ranking_v1",
        "status": "COMPLETED_ONE_BACKWARD_GRADIENT_DELTA_RANKING",
        "manifest": hash_file(output_root / "MANIFEST.json"),
        "pilot_mean_loss": float(pilot_loss.detach()),
        "gradient_seconds": (time.time_ns() - gradient_started_ns) / 1e9,
        "autograd_backward_calls": 1,
        "optimizer_steps": 0,
        "unused_zero_gradient_tensors": unused_gradient_tensors,
        "gradient_norm": gradient_norm,
        "direction_norm": direction_norm,
        "gradient_dot_direction": dot,
        "gradient_direction_cosine": dot / (gradient_norm * direction_norm),
        "first_order_full_extension_predicted_improvement": -dot,
        "positive_extension_score_count": int((extension_improvement[nonzero] > 0).sum()),
        "positive_ablation_score_count": int((ablation_improvement[nonzero] > 0).sum()),
        "top_extension": extension_rows,
        "top_ablation": ablation_rows,
        "top_absolute_local_contribution": absolute_rows,
        "cpu_fp32_shadow_numerically_comparable_to_canonical_mps_bf16": False,
        "local_verify_reused_optimization_data_not_held_out": True,
    }
    atomic_json(output_root / "RANKING.json", ranking)
    print(
        f"GRADIENT complete loss={float(pilot_loss.detach()):.12f} "
        f"dot={dot:.12g} cosine={ranking['gradient_direction_cosine']:.12g}",
        flush=True,
    )
    del gradient_parameters, gradient_leaves, pilot_loss

    with torch.no_grad():
        current_baseline = float(mean(losses_for(dict(current_parameters), batches)))
        source_baseline = float(mean(losses_for(dict(source_parameters), batches)))
    screening = []
    shortlist = absolute_rows[: args.screen_top_k_per_mode]
    screen_count = len(shortlist) * 2
    print(
        f"SCREEN current_baseline={current_baseline:.12f} "
        f"source_baseline={source_baseline:.12f} count={screen_count}",
        flush=True,
    )
    ordinal = 0
    for row in shortlist:
        flat_index = int(row["flat_index"])
        parameter_name = str(row["parameter_name"])
        local_index = int(row["parameter_local_flat_index"])
        signed_delta = float(row["delta"])
        for mode, base_parameters, baseline, multiplier in (
            ("apply_delta_to_g9_source", source_parameters, source_baseline, 1.0),
            ("ablate_delta_from_g10_current", current_parameters, current_baseline, -1.0),
        ):
            ordinal += 1
            trial_parameters = dict(base_parameters)
            trial = base_parameters[parameter_name].detach().clone()
            trial.reshape(-1)[local_index] += signed_delta * multiplier
            trial_parameters[parameter_name] = trial
            with torch.no_grad():
                trial_loss = float(mean(losses_for(trial_parameters, batches)))
            result = {
                **row,
                "mode": mode,
                "screen_ordinal": ordinal,
                "cpu_fp32_shadow_baseline_loss": baseline,
                "cpu_fp32_shadow_trial_loss": trial_loss,
                "cpu_fp32_shadow_observed_improvement": baseline - trial_loss,
                "strict_cpu_shadow_improvement": trial_loss < baseline,
            }
            screening.append(result)
            print(
                f"SCREEN {ordinal}/{screen_count} mode={mode} "
                f"index={flat_index} loss={trial_loss:.12f} "
                f"improvement={baseline - trial_loss:.12g}",
                flush=True,
            )

    result = {
        "schema": "hfw_w560_scalar_delta_cpu_fp32_shadow_pilot_result_v1",
        "status": "COMPLETED_READ_ONLY_SCALAR_DELTA_PILOT",
        "manifest": hash_file(output_root / "MANIFEST.json"),
        "ranking": hash_file(output_root / "RANKING.json"),
        "completed_unix_ns": time.time_ns(),
        "cpu_fp32_shadow_g9_source_baseline_loss": source_baseline,
        "cpu_fp32_shadow_g10_current_baseline_loss": current_baseline,
        "scalar_screen_count": len(screening),
        "strict_cpu_shadow_improver_count": sum(
            int(row["strict_cpu_shadow_improvement"]) for row in screening
        ),
        "screening": screening,
        "canonical_mps_bf16_verification_required_before_any_promotion": True,
        "canonical_promotion_authority": False,
        "checkpoint_or_head_writes": 0,
        "cpu_fp32_shadow_numerically_comparable_to_canonical_mps_bf16": False,
        "local_verify_reused_optimization_data_not_held_out": True,
    }
    atomic_json(output_root / "RESULT.json", result)
    print(
        f"DONE screens={len(screening)} cpu_shadow_improvers="
        f"{result['strict_cpu_shadow_improver_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
