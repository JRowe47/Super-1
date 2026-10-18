from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import torch

from .manifest import canonical_json_bytes, file_sha256
from .runtime_guard import NoTrainingRuntimeGuard
from .artifacts import hash_file_same_fd, read_bytes_same_fd, read_json_same_fd


class GroupObjective(Protocol):
    parent: object
    base_parameters: Mapping[str, torch.Tensor]

    def flat_group(self, group: str) -> torch.Tensor: ...
    def loss(self, group: str, flat: torch.Tensor, batch_index: int) -> float: ...
    def loss_vector(self, group: str, flat: torch.Tensor, batch_index: int) -> Sequence[float]: ...


class ProposalRegistry(Protocol):
    def append(self, *, proposal_sha256: str, delta_sha256: str,
               child_checkpoint_sha256: str | None, accepted: bool,
               reason: str, record: Mapping[str, object]) -> str: ...


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def _exclusive_json(path: Path, payload: object) -> str:
    encoded = canonical_json_bytes(payload)
    _exclusive_write(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _stable_tensor_seed(branch_seed: int, pair_index: int, name: str) -> int:
    body = f"{branch_seed}:{pair_index}:{name}".encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "little") & ((1 << 63) - 1)


def _structured_unit_direction(reference: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = tuple(reference.shape)
    if reference.ndim == 1:
        value = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
        return value.to(dtype=torch.float32).mul_(2).sub_(1).to(reference.device)
    if reference.ndim == 2:
        used_rank = min(rank, shape[0], shape[1])
        left = torch.randn((shape[0], used_rank), generator=generator)
        right = torch.randn((used_rank, shape[1]), generator=generator)
        return (left @ right).to(reference.device)
    if reference.ndim == 3:
        planes = []
        for _ in range(shape[0]):
            used_rank = min(rank, shape[1], shape[2])
            left = torch.randn((shape[1], used_rank), generator=generator)
            right = torch.randn((used_rank, shape[2]), generator=generator)
            planes.append(left @ right)
        return torch.stack(planes).to(reference.device)
    return torch.randn(shape, generator=generator).to(reference.device)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def _functional_effective_rms(
    design: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Exact phase-averaged RMS without materializing depth x input x output."""
    design_gram = design.float().transpose(0, 1) @ design.float()
    controls = left.shape[0]
    total = torch.zeros((), device=left.device, dtype=torch.float64)
    left64 = left.double()
    right64 = right.double()
    gram64 = design_gram.to(device=left.device, dtype=torch.float64)
    for first in range(controls):
        for second in range(controls):
            left_cross = left64[first].transpose(0, 1) @ left64[second]
            inner = ((left_cross @ right64[second]) * right64[first]).sum()
            total = total + gram64[first, second] * inner
    denominator = design.shape[0] * left.shape[1] * right.shape[2]
    return (total.clamp_min(0) / denominator).sqrt().float()


def _get_module(model: torch.nn.Module, parameter_name: str) -> torch.nn.Module:
    return model.get_submodule(parameter_name.rsplit(".", 1)[0])


def _scale_direction(
    model: torch.nn.Module,
    name: str,
    reference: torch.Tensor,
    direction: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, Mapping[str, float | str]]:
    if name.endswith(".control_left"):
        module = _get_module(model, name)
        right = module.control_right.detach()
        reference_norm = _functional_effective_rms(module.design, reference, right)
        direction_norm = _functional_effective_rms(module.design, direction, right)
        norm_kind = "induced_effective_weight_rms"
    elif name.endswith(".control_right"):
        module = _get_module(model, name)
        left = module.control_left.detach()
        reference_norm = _functional_effective_rms(module.design, left, reference)
        direction_norm = _functional_effective_rms(module.design, left, direction)
        norm_kind = "induced_effective_weight_rms"
    else:
        reference_norm = _rms(reference)
        direction_norm = _rms(direction)
        norm_kind = "parameter_rms"
    if not torch.isfinite(direction_norm) or float(direction_norm) <= 0:
        raise RuntimeError(f"zero/nonfinite structured direction norm: {name}")
    target = radius * max(float(reference_norm), torch.finfo(torch.float32).tiny)
    scaled = direction.mul(target / float(direction_norm))
    return scaled, {
        "name": name,
        "radius": radius,
        "norm_kind": norm_kind,
        "reference_rms": float(reference_norm),
        "proposal_rms": float(direction_norm) * target / float(direction_norm),
    }


def build_group_direction(
    objective: GroupObjective,
    group: str,
    radius: float,
    branch_seed: int,
    pair_index: int,
    structured_rank: int,
) -> tuple[torch.Tensor, Sequence[Mapping[str, float | str]]]:
    values = []
    records = []
    model = objective.parent.model
    for name in objective.parent.groups[group]:
        reference = objective.base_parameters[name]
        raw = _structured_unit_direction(
            reference, structured_rank,
            _stable_tensor_seed(branch_seed, pair_index, name),
        ).to(dtype=reference.dtype)
        scaled, record = _scale_direction(model, name, reference, raw, radius)
        values.append(scaled.reshape(-1))
        records.append(record)
    return torch.cat(values).detach(), tuple(records)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _qualifies(
    center: Sequence[float], candidate: Sequence[float], acceptance: Mapping[str, object], seed: int,
) -> tuple[bool, float]:
    improvements = [before - after for before, after in zip(center, candidate)]
    if not all(math.isfinite(value) for value in (*center, *candidate, *improvements)):
        return False, float("nan")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = int(acceptance["bootstrap_samples"])
    values = torch.tensor(improvements, dtype=torch.float64)
    indices = torch.randint(0, len(improvements), (samples, len(improvements)), generator=generator)
    means = values[indices].mean(dim=1).sort().values
    corrected_alpha = float(acceptance["familywise_alpha"]) / int(acceptance["bonferroni_hypotheses"])
    quantile_index = max(0, min(samples - 1, math.floor(corrected_alpha * samples)))
    lcb = float(means[quantile_index])
    required_wins = int(acceptance["minimum_winning_sequences"])
    passes = (
        _mean(improvements) >= float(acceptance["minimum_mean_loss_improvement"])
        and sum(value > 0 for value in improvements) >= required_wins
        and min(improvements) >= -float(acceptance["maximum_single_sequence_regression"])
        and lcb >= float(acceptance["minimum_bootstrap_lcb"])
    )
    return passes, lcb


def _duplicated_loss_vectors(
    objective: GroupObjective, group: str, flat: torch.Tensor,
    batch_indices: Sequence[int], tolerance: float,
) -> tuple[list[float], list[float]]:
    primary = []
    duplicate = []
    for batch_index in batch_indices:
        if not hasattr(objective, "loss_vector"):
            raise RuntimeError("production objective must expose per-sequence loss vectors")
        first = [float(value) for value in objective.loss_vector(group, flat, batch_index)]
        second = [float(value) for value in objective.loss_vector(group, flat, batch_index)]
        if len(first) != len(second) or not first:
            raise RuntimeError("per-sequence loss vector shape drift")
        if not all(math.isfinite(value) for value in (*first, *second)):
            raise RuntimeError("nonfinite duplicated loss evidence")
        if max(abs(left - right) for left, right in zip(first, second)) > tolerance:
            raise RuntimeError("duplicated forward evidence exceeded tolerance")
        primary.extend(first)
        duplicate.extend(second)
    return primary, duplicate


def _tensor_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return array.tobytes(order="C")


def _materialize_child(
    objective: GroupObjective, group: str, flat: torch.Tensor, path: Path,
    immutable_parent_sha256: str, immediate_parent_child_sha256: str,
    chain_sha256: str,
) -> str:
    state = {}
    group_names = set(objective.parent.groups[group])
    cursor = 0
    for name, value in objective.base_parameters.items():
        if name in group_names:
            count = value.numel()
            state[name] = flat[cursor:cursor + count].view_as(value).detach().cpu()
            cursor += count
        else:
            state[name] = value.detach().cpu()
    for name, value in getattr(objective, "buffers", {}).items():
        if name in state:
            raise RuntimeError("materialized state parameter/buffer collision")
        state[name] = value.detach().cpu()
    if cursor != flat.numel():
        raise RuntimeError("materialized group length drift")
    payload = {
        "schema": "hfw_post_training_materialized_child_v1",
        "immutable_parent_checkpoint_sha256": immutable_parent_sha256,
        "immediate_parent_child_sha256": immediate_parent_child_sha256,
        "branch_group": group,
        "accepted_chain_sha256": chain_sha256,
        "optimizer_present": False,
        "scheduler_present": False,
        "training_resume_authorized": False,
        "strict_load_compatible_full_state_dict": True,
        "named_buffers_included": sorted(getattr(objective, "buffers", {})),
        "model": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BranchResult:
    group: str
    accepts: int
    proposals: int
    final_flat: torch.Tensor
    receipt: Mapping[str, object]


def _run_independent_branch_guarded(
    objective: GroupObjective,
    protocol: Mapping[str, object],
    group: str,
    output_root: Path,
    proposal_registry: ProposalRegistry,
    materialize: bool = True,
) -> BranchResult:
    if not materialize or not bool(protocol["search"]["materialize_after_each_accept"]):
        raise RuntimeError("production branches require every accepted child to be materialized")
    search = protocol["search"]
    if group not in search["independent_branches"]:
        raise RuntimeError("unauthorized branch group")
    if int(search["autograd_calls"]) or int(search["optimizer_steps"]):
        raise RuntimeError("production protocol authorizes training")
    branch_root = output_root / group
    branch_root.mkdir(parents=True, exist_ok=False)
    parent = objective.flat_group(group).detach()
    current = parent.clone()
    parent_norm = math.sqrt(math.fsum(
        float(value.detach().float().square().sum().item())
        for value in objective.base_parameters.values()
    ))
    branch_seed = int(search["branch_seeds"][group])
    radii = [2.0 ** -int(value) for value in search["radius_exponents_interleaved"]]
    if len(radii) != int(search["pairs_per_branch"]):
        raise RuntimeError("precommitted radius schedule length drift")
    header = {
        "schema": "hfw_post_training_independent_branch_header_v1",
        "group": group,
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "parent_group_sha256": hashlib.sha256(_tensor_bytes(parent)).hexdigest(),
        "branch_seed": branch_seed,
        "radii": radii,
        "maximum_accepts": int(search["maximum_accepts_per_branch"]),
        "displacement_norm_denominator": "all_learned_parent_parameters_l2",
        "fixed_eval_opened": False,
        "optimizer_steps": 0,
        "autograd_calls": 0,
    }
    chain_sha = _exclusive_json(branch_root / "branch_header.json", header)
    current_child_sha = protocol["parent"]["checkpoint_sha256"]
    accepted = 0
    forward_calls = 0
    records = []
    started = time.monotonic()
    for pair_index, radius in enumerate(radii):
        batch_count = int(protocol["data"]["calibration_batches"])
        confirm = int(search["confirmation_batches"])
        batch_indices = tuple((pair_index * confirm + offset) % batch_count for offset in range(confirm))
        direction, tensor_norms = build_group_direction(
            objective, group, radius, branch_seed, pair_index,
            int(search["structured_direction_rank"]),
        )
        tolerance = float(search["acceptance"]["duplicate_forward_tolerance"])
        center_losses, center_duplicate = _duplicated_loss_vectors(
            objective, group, current, batch_indices, tolerance
        )
        plus = current + direction
        minus = current - direction
        plus_losses, plus_duplicate = _duplicated_loss_vectors(
            objective, group, plus, batch_indices, tolerance
        )
        minus_losses, minus_duplicate = _duplicated_loss_vectors(
            objective, group, minus, batch_indices, tolerance
        )
        forward_calls += 6 * confirm
        candidates = (("plus", plus, plus_losses), ("minus", minus, minus_losses))
        qualifications = {
            side: _qualifies(
                center_losses, losses, search["acceptance"],
                int(search["acceptance"]["bootstrap_seed"]) + pair_index * 2 + side_index,
            )
            for side_index, (side, _, losses) in enumerate(candidates)
        }
        eligible = [item for item in candidates if qualifications[item[0]][0]]
        chosen = min(eligible, key=lambda item: _mean(item[2])) if eligible else None
        displacement = None
        accepted_delta_sha = None
        accepted_child_sha = None
        if chosen is not None and accepted < int(search["maximum_accepts_per_branch"]):
            proposed_displacement = float(torch.linalg.vector_norm((chosen[1] - parent).float()))
            displacement_ratio = proposed_displacement / max(parent_norm, torch.finfo(torch.float32).tiny)
            if displacement_ratio <= float(search["maximum_parent_norm_displacement"]):
                incremental = (chosen[1] - current).detach()
                current = chosen[1].detach()
                accepted += 1
                displacement = displacement_ratio
                delta_bytes = _tensor_bytes(current - parent)
                delta_sha = hashlib.sha256(delta_bytes).hexdigest()
                accepted_delta_sha = delta_sha
                delta_name = f"accepted_{accepted:02d}_delta.f32le"
                _exclusive_write(branch_root / delta_name, delta_bytes)
                incremental_bytes = _tensor_bytes(incremental)
                incremental_sha = hashlib.sha256(incremental_bytes).hexdigest()
                incremental_name = f"accepted_{accepted:02d}_incremental_delta.f32le"
                _exclusive_write(branch_root / incremental_name, incremental_bytes)
                accept_record = {
                    "schema": "hfw_post_training_accepted_delta_v1",
                    "group": group,
                    "accept_index": accepted,
                    "pair_index": pair_index,
                    "seed": branch_seed,
                    "side": chosen[0],
                    "radius": radius,
                    "batch_indices": list(batch_indices),
                    "calibration_before": center_losses,
                    "calibration_after": chosen[2],
                    "mean_before": _mean(center_losses),
                    "mean_after": _mean(chosen[2]),
                    "parent_norm_displacement": displacement_ratio,
                    "delta_file": delta_name,
                    "delta_sha256": delta_sha,
                    "delta_semantics": "cumulative_current_minus_immutable_parent",
                    "incremental_delta_file": incremental_name,
                    "incremental_delta_sha256": incremental_sha,
                    "incremental_delta_semantics": "accepted_child_minus_previous_branch_child",
                    "previous_chain_sha256": chain_sha,
                    "immutable_parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
                    "immediate_parent_child_sha256": current_child_sha,
                    "tensor_radius_calibration": tensor_norms,
                    "fixed_eval_opened": False,
                }
                chain_sha = _exclusive_json(
                    branch_root / f"accepted_{accepted:02d}.json", accept_record
                )
                if materialize and bool(search["materialize_after_each_accept"]):
                    child_name = f"materialized_child_{accepted:02d}.pt"
                    child_sha = _materialize_child(
                        objective, group, current, branch_root / child_name,
                        protocol["parent"]["checkpoint_sha256"], current_child_sha,
                        chain_sha,
                    )
                    accepted_child_sha = child_sha
                    accepted_delta_sha = incremental_sha
                    _exclusive_json(branch_root / f"materialized_child_{accepted:02d}.receipt.json", {
                        "schema": "hfw_materialized_child_receipt_v1",
                        "path": child_name,
                        "sha256": child_sha,
                        "accepted_chain_sha256": chain_sha,
                        "immutable_parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
                        "immediate_parent_child_sha256": current_child_sha,
                        "optimizer_present": False,
                        "training_resume_authorized": False,
                    })
                    current_child_sha = child_sha
        direction_sha = hashlib.sha256(_tensor_bytes(direction)).hexdigest()
        record = {
            "schema": "hfw_post_training_proposal_chain_node_v1",
            "pair_index": pair_index,
            "seed": branch_seed,
            "radius": radius,
            "batch_indices": list(batch_indices),
            "direction_sha256": direction_sha,
            "center_loss_vector": center_losses,
            "center_duplicate_loss_vector": center_duplicate,
            "plus_loss_vector": plus_losses,
            "plus_duplicate_loss_vector": plus_duplicate,
            "minus_loss_vector": minus_losses,
            "minus_duplicate_loss_vector": minus_duplicate,
            "center_mean": _mean(center_losses),
            "plus_mean": _mean(plus_losses),
            "minus_mean": _mean(minus_losses),
            "plus_bootstrap_lcb": qualifications["plus"][1],
            "minus_bootstrap_lcb": qualifications["minus"][1],
            "accepted_side": None if chosen is None or displacement is None else chosen[0],
            "parent_norm_displacement": displacement,
            "previous_chain_sha256": chain_sha,
            "finite_fail_closed": True,
        }
        chain_sha = _exclusive_json(branch_root / f"proposal_{pair_index:02d}.json", record)
        proposal_registry.append(
            proposal_sha256=chain_sha,
            delta_sha256=accepted_delta_sha or direction_sha,
            child_checkpoint_sha256=accepted_child_sha,
            accepted=record["accepted_side"] is not None,
            reason=("confirmed_strong_improvement" if record["accepted_side"] is not None
                    else "failed_strong_acceptance_or_branch_limit"),
            record={
                "group": group,
                "pair_index": pair_index,
                "radius": radius,
                "accepted_side": record["accepted_side"],
                "center_loss_vector": center_losses,
                "plus_loss_vector": plus_losses,
                "minus_loss_vector": minus_losses,
                "summed_nll_before": math.fsum(center_losses),
                "summed_nll_after": (
                    math.fsum(chosen[2])
                    if record["accepted_side"] is not None else math.fsum(center_losses)
                ),
                "device": str(current.device),
                "runtime": {"forward_calls_for_pair": 6 * confirm},
            },
        )
        records.append(record)
    elapsed = time.monotonic() - started
    targets_per_forward = int(protocol["data"]["batch_size"]) * int(protocol["data"]["context"])
    macs_per_token = None
    if hasattr(objective.parent.model, "estimated_macs_per_token"):
        macs_per_token = int(objective.parent.model.estimated_macs_per_token(
            int(protocol["data"]["context"])
        ))
    receipt = {
        "schema": "hfw_post_training_independent_branch_receipt_v1",
        "status": "PASS_BRANCH_SEARCH_COMPLETE",
        "group": group,
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "pairs": len(radii),
        "accepts": accepted,
        "maximum_accepts": int(search["maximum_accepts_per_branch"]),
        "forward_calls": forward_calls,
        "total_evaluated_target_tokens": forward_calls * targets_per_forward,
        "estimated_macs_per_target_token": macs_per_token,
        "estimated_total_macs": (
            None if macs_per_token is None
            else macs_per_token * forward_calls * targets_per_forward
        ),
        "unique_calibration_token_positions": (
            int(protocol["data"]["calibration_batches"])
            * int(protocol["data"]["batch_size"])
            * (int(protocol["data"]["context"]) + 1)
        ),
        "new_unique_tokens_after_initialization": 0,
        "wall_seconds": elapsed,
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "fixed_eval_opened": False,
        "final_parent_norm_displacement": float(
            torch.linalg.vector_norm((current - parent).float())
            / max(parent_norm, torch.finfo(torch.float32).tiny)
        ),
        "complete_proposal_chain_sha256": chain_sha,
        "proposal_records": records,
    }
    receipt["receipt_identity_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    receipt_sha = _exclusive_json(branch_root / "branch_receipt.json", receipt)
    receipt = dict(receipt, receipt_file_sha256=receipt_sha)
    return BranchResult(group, accepted, len(radii), current, receipt)


def _run_independent_branch_guarded(
    objective: GroupObjective,
    protocol: Mapping[str, object],
    group: str,
    output_root: Path,
    proposal_registry: ProposalRegistry,
    materialize: bool = True,
) -> BranchResult:
    """Exact transfer of the successful local paired trust-region search."""
    search = protocol["search"]
    if search.get("solver") != "paired_spectral_trust_region_exact_v1_transfer":
        raise RuntimeError("proven solver contract missing")
    if group not in search["independent_branches"]:
        raise RuntimeError("unauthorized branch group")
    if not materialize or not bool(search["materialize_after_each_accept"]):
        raise RuntimeError("every accepted child must be materialized")
    if int(search["autograd_calls"]) or int(search["optimizer_steps"]):
        raise RuntimeError("solver must remain forward-only")

    budget = int(search["model_evaluation_budget"])
    direction_count = int(search["paired_direction_count"])
    line_scales = tuple(float(value) for value in search["line_search_scales"])
    evaluations_per_iteration = 1 + 2 * direction_count + len(line_scales)
    if budget != 180 or evaluations_per_iteration != 10 or budget % evaluations_per_iteration:
        raise RuntimeError("exact proven 180-evaluation schedule drift")
    iterations = budget // evaluations_per_iteration
    if iterations != int(search["pairs_per_branch"]):
        raise RuntimeError("iteration count drift")

    branch_root = output_root / group
    branch_root.mkdir(parents=True, exist_ok=False)
    parent = objective.flat_group(group).detach()
    current = parent.clone()
    parent_norm = math.sqrt(math.fsum(
        float(value.detach().float().square().sum().item())
        for value in objective.base_parameters.values()
    ))
    branch_seed = int(search["branch_seeds"][group])
    radius = float(search["initial_radius"])
    minimum_relative_improvement = float(search["minimum_relative_improvement"])
    header = {
        "schema": "hfw_post_training_proven_solver_branch_header_v1",
        "solver": search["solver"],
        "group": group,
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "parent_group_sha256": hashlib.sha256(_tensor_bytes(parent)).hexdigest(),
        "branch_seed": branch_seed,
        "model_evaluation_budget": budget,
        "iterations": iterations,
        "paired_direction_count": direction_count,
        "paired_top_k": int(search["paired_top_k"]),
        "line_search_scales": list(line_scales),
        "initial_radius": radius,
        "minimum_relative_improvement": minimum_relative_improvement,
        "fixed_eval_opened": False,
        "optimizer_steps": 0,
        "autograd_calls": 0,
    }
    chain_sha = _exclusive_json(branch_root / "branch_header.json", header)
    current_child_sha = protocol["parent"]["checkpoint_sha256"]
    accepted = 0
    forward_calls = 0
    records = []
    started = time.monotonic()

    for iteration in range(iterations):
        batch_index = iteration % int(protocol["data"]["calibration_batches"])
        center_loss = float(objective.loss(group, current, batch_index))
        forward_calls += 1
        if not math.isfinite(center_loss):
            raise RuntimeError("nonfinite center loss")

        base_seed = branch_seed + iteration * direction_count
        candidate_pool = []
        scored_directions = []
        plus_losses = []
        minus_losses = []
        direction_hashes = []
        tensor_radius_records = []
        for direction_index in range(direction_count):
            direction_seed = base_seed + direction_index
            direction, tensor_norms = build_group_direction(
                objective, group, radius, direction_seed, 0,
                int(search["structured_direction_rank"]),
            )
            positive = current + direction
            negative = current - direction
            positive_loss = float(objective.loss(group, positive, batch_index))
            negative_loss = float(objective.loss(group, negative, batch_index))
            forward_calls += 2
            positive_finite = math.isfinite(positive_loss)
            negative_finite = math.isfinite(negative_loss)
            plus_losses.append(positive_loss if positive_finite else None)
            minus_losses.append(negative_loss if negative_finite else None)
            if positive_finite:
                candidate_pool.append(
                    (positive_loss, positive, direction_seed, f"plus_{direction_index}")
                )
            if negative_finite:
                candidate_pool.append(
                    (negative_loss, negative, direction_seed, f"minus_{direction_index}")
                )
            if positive_finite and negative_finite:
                derivative = (positive_loss - negative_loss) / max(2.0 * radius, 1e-12)
                endpoint_improvement = center_loss - min(positive_loss, negative_loss)
                scored_directions.append((endpoint_improvement, derivative, direction))
            direction_hashes.append(hashlib.sha256(_tensor_bytes(direction)).hexdigest())
            tensor_radius_records.append(list(tensor_norms))

        ranked = sorted(
            scored_directions,
            key=lambda item: (item[0], abs(item[1])),
            reverse=True,
        )
        selected = ranked[:min(int(search["paired_top_k"]), len(ranked))]
        combined = torch.zeros_like(current)
        selected_norms = []
        for improvement, derivative, direction in selected:
            weight = max(improvement, 0.0) + abs(derivative) * radius + 1e-12
            combined.add_(direction, alpha=-math.copysign(weight, derivative))
            selected_norms.append(float(torch.linalg.vector_norm(direction.float())))
        combined_norm = float(torch.linalg.vector_norm(combined.float()))
        line_losses = []
        if combined_norm > torch.finfo(torch.float32).eps:
            target_norm = math.fsum(selected_norms) / len(selected_norms)
            combined = combined * (target_norm / combined_norm)
            for line_index, scale in enumerate(line_scales):
                line_seed = base_seed + direction_count + line_index
                candidate = current + combined * scale
                candidate_loss = float(objective.loss(group, candidate, batch_index))
                forward_calls += 1
                candidate_finite = math.isfinite(candidate_loss)
                line_losses.append(candidate_loss if candidate_finite else None)
                if candidate_finite:
                    candidate_pool.append((
                        candidate_loss, candidate, line_seed, f"line_{line_index}_{scale}",
                    ))
        else:
            for _ in line_scales:
                repeated_loss = float(objective.loss(group, current, batch_index))
                forward_calls += 1
                if not math.isfinite(repeated_loss):
                    raise RuntimeError("nonfinite budget-fill loss")
                line_losses.append(repeated_loss)

        best = min(candidate_pool, key=lambda item: item[0]) if candidate_pool else None
        threshold = center_loss - abs(center_loss) * minimum_relative_improvement
        accepted_side = None
        displacement = None
        accepted_delta_sha = None
        accepted_child_sha = None
        if best is not None and best[0] < threshold:
            previous = current
            current = best[1].detach()
            accepted += 1
            accepted_side = best[3]
            incremental = (current - previous).detach()
            displacement = float(
                torch.linalg.vector_norm((current - parent).float())
                / max(parent_norm, torch.finfo(torch.float32).tiny)
            )
            delta_bytes = _tensor_bytes(current - parent)
            delta_sha = hashlib.sha256(delta_bytes).hexdigest()
            delta_name = f"accepted_{accepted:02d}_delta.f32le"
            _exclusive_write(branch_root / delta_name, delta_bytes)
            incremental_bytes = _tensor_bytes(incremental)
            incremental_sha = hashlib.sha256(incremental_bytes).hexdigest()
            incremental_name = f"accepted_{accepted:02d}_incremental_delta.f32le"
            _exclusive_write(branch_root / incremental_name, incremental_bytes)
            accept_record = {
                "schema": "hfw_post_training_accepted_delta_v1",
                "solver": search["solver"],
                "group": group,
                "accept_index": accepted,
                "pair_index": iteration,
                "seed": best[2],
                "side": accepted_side,
                "radius": radius,
                "batch_indices": [batch_index],
                "calibration_before": [center_loss],
                "calibration_after": [best[0]],
                "mean_before": center_loss,
                "mean_after": best[0],
                "relative_acceptance_threshold": threshold,
                "parent_norm_displacement": displacement,
                "delta_file": delta_name,
                "delta_sha256": delta_sha,
                "delta_semantics": "cumulative_current_minus_immutable_parent",
                "incremental_delta_file": incremental_name,
                "incremental_delta_sha256": incremental_sha,
                "incremental_delta_semantics": "accepted_child_minus_previous_branch_child",
                "previous_chain_sha256": chain_sha,
                "immutable_parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
                "immediate_parent_child_sha256": current_child_sha,
                "tensor_radius_calibration": tensor_radius_records,
                "fixed_eval_opened": False,
            }
            chain_sha = _exclusive_json(
                branch_root / f"accepted_{accepted:02d}.json", accept_record
            )
            child_name = f"materialized_child_{accepted:02d}.pt"
            child_sha = _materialize_child(
                objective, group, current, branch_root / child_name,
                protocol["parent"]["checkpoint_sha256"], current_child_sha, chain_sha,
            )
            accepted_child_sha = child_sha
            accepted_delta_sha = incremental_sha
            _exclusive_json(branch_root / f"materialized_child_{accepted:02d}.receipt.json", {
                "schema": "hfw_materialized_child_receipt_v1",
                "path": child_name,
                "sha256": child_sha,
                "accepted_chain_sha256": chain_sha,
                "immutable_parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
                "immediate_parent_child_sha256": current_child_sha,
                "optimizer_present": False,
                "training_resume_authorized": False,
            })
            current_child_sha = child_sha
            radius = min(float(search["maximum_radius"]), radius * float(search["radius_growth"]))
        else:
            radius = max(float(search["minimum_radius"]), radius * float(search["radius_shrink"]))

        record = {
            "schema": "hfw_post_training_proven_solver_proposal_v1",
            "pair_index": iteration,
            "seed": base_seed,
            "radius": radius,
            "batch_indices": [batch_index],
            "direction_sha256": hashlib.sha256("".join(direction_hashes).encode()).hexdigest(),
            "direction_sha256s": direction_hashes,
            "center_loss_vector": [center_loss] * direction_count,
            "center_duplicate_loss_vector": [center_loss] * direction_count,
            "plus_loss_vector": plus_losses,
            "plus_duplicate_loss_vector": plus_losses,
            "minus_loss_vector": minus_losses,
            "minus_duplicate_loss_vector": minus_losses,
            "center_mean": center_loss,
            "plus_mean": _mean([value for value in plus_losses if value is not None]) if any(value is not None for value in plus_losses) else None,
            "minus_mean": _mean([value for value in minus_losses if value is not None]) if any(value is not None for value in minus_losses) else None,
            "line_search_loss_vector": line_losses,
            "accepted_side": accepted_side,
            "accepted_loss": None if accepted_side is None else best[0],
            "acceptance_threshold": threshold,
            "parent_norm_displacement": displacement,
            "previous_chain_sha256": chain_sha,
            "finite_fail_closed": True,
        }
        chain_sha = _exclusive_json(branch_root / f"proposal_{iteration:02d}.json", record)
        proposal_registry.append(
            proposal_sha256=chain_sha,
            delta_sha256=accepted_delta_sha or record["direction_sha256"],
            child_checkpoint_sha256=accepted_child_sha,
            accepted=accepted_side is not None,
            reason="relative_same_batch_improvement" if accepted_side else "no_relative_improvement",
            record={
                "group": group,
                "pair_index": iteration,
                "radius": radius,
                "accepted_side": accepted_side,
                "center_loss_vector": [center_loss],
                "plus_loss_vector": plus_losses,
                "minus_loss_vector": minus_losses,
                "summed_nll_before": center_loss,
                "summed_nll_after": best[0] if accepted_side else center_loss,
                "device": str(current.device),
                "runtime": {"forward_calls_for_iteration": evaluations_per_iteration},
            },
        )
        records.append(record)

    if forward_calls != budget:
        raise RuntimeError(f"evaluation budget drift: {forward_calls} != {budget}")
    elapsed = time.monotonic() - started
    targets_per_forward = int(protocol["data"]["batch_size"]) * int(protocol["data"]["context"])
    macs_per_token = None
    if hasattr(objective.parent.model, "estimated_macs_per_token"):
        macs_per_token = int(objective.parent.model.estimated_macs_per_token(
            int(protocol["data"]["context"])
        ))
    receipt = {
        "schema": "hfw_post_training_proven_solver_branch_receipt_v1",
        "status": "PASS_BRANCH_SEARCH_COMPLETE",
        "solver": search["solver"],
        "group": group,
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "pairs": iterations,
        "accepts": accepted,
        "maximum_accepts": int(search["maximum_accepts_per_branch"]),
        "forward_calls": forward_calls,
        "total_evaluated_target_tokens": forward_calls * targets_per_forward,
        "estimated_macs_per_target_token": macs_per_token,
        "estimated_total_macs": None if macs_per_token is None else macs_per_token * forward_calls * targets_per_forward,
        "unique_calibration_token_positions": int(protocol["data"]["calibration_batches"]) * int(protocol["data"]["batch_size"]) * (int(protocol["data"]["context"]) + 1),
        "new_unique_tokens_after_initialization": 0,
        "wall_seconds": elapsed,
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "fixed_eval_opened": False,
        "final_radius": radius,
        "final_parent_norm_displacement": float(torch.linalg.vector_norm((current - parent).float()) / max(parent_norm, torch.finfo(torch.float32).tiny)),
        "complete_proposal_chain_sha256": chain_sha,
        "proposal_records": records,
    }
    receipt["receipt_identity_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    receipt_sha = _exclusive_json(branch_root / "branch_receipt.json", receipt)
    receipt = dict(receipt, receipt_file_sha256=receipt_sha)
    return BranchResult(group, accepted, iterations, current, receipt)


def run_independent_branch(
    objective: GroupObjective,
    protocol: Mapping[str, object],
    group: str,
    output_root: Path,
    proposal_registry: ProposalRegistry,
    materialize: bool = True,
) -> BranchResult:
    with NoTrainingRuntimeGuard() as guard:
        result = _run_independent_branch_guarded(
            objective, protocol, group, output_root, proposal_registry,
            materialize=materialize
        )
    runtime_receipt = guard.receipt()
    if runtime_receipt["status"] != "PASS_NO_TRAINING_RUNTIME_GUARD":
        raise RuntimeError("no-training runtime guard recorded a forbidden call")
    runtime_sha = _exclusive_json(
        output_root / group / "runtime_guard_receipt.json", runtime_receipt
    )
    augmented = dict(result.receipt, runtime_guard={**runtime_receipt, "file_sha256": runtime_sha})
    return BranchResult(result.group, result.accepts, result.proposals, result.final_flat, augmented)


def validate_branch_artifact_chain(
    protocol: Mapping[str, object], branch_root: Path, group: str,
    include_final_delta_bytes: bool = False,
    validate_child_payload: bool = False,
) -> Mapping[str, object]:
    if branch_root.is_symlink() or not branch_root.is_dir():
        raise RuntimeError("branch artifact root must be a regular directory")
    header_path = branch_root / "branch_header.json"
    if header_path.is_symlink() or not header_path.is_file():
        raise RuntimeError("branch header is missing or symlinked")
    header, header_identity = read_json_same_fd(header_path)
    if header.get("group") != group or header.get("parent_checkpoint_sha256") != protocol["parent"]["checkpoint_sha256"]:
        raise RuntimeError("branch header binding drift")
    chain = header_identity["sha256"]
    accept_index = 0
    final_delta_bytes = None
    final_delta_identity = None
    final_child_identity = None
    expected_immediate_parent = protocol["parent"]["checkpoint_sha256"]
    for pair_index in range(int(protocol["search"]["pairs_per_branch"])):
        proposal_path = branch_root / f"proposal_{pair_index:02d}.json"
        if proposal_path.is_symlink() or not proposal_path.is_file():
            raise RuntimeError("proposal chain node missing or symlinked")
        proposal, proposal_identity = read_json_same_fd(proposal_path)
        if proposal.get("pair_index") != pair_index:
            raise RuntimeError("proposal chain sequence drift")
        if proposal.get("accepted_side") is not None:
            accept_index += 1
            accept_path = branch_root / f"accepted_{accept_index:02d}.json"
            delta_path = branch_root / f"accepted_{accept_index:02d}_delta.f32le"
            incremental_path = branch_root / f"accepted_{accept_index:02d}_incremental_delta.f32le"
            child_path = branch_root / f"materialized_child_{accept_index:02d}.pt"
            child_receipt_path = branch_root / f"materialized_child_{accept_index:02d}.receipt.json"
            if any(path.is_symlink() or not path.is_file() for path in (
                accept_path, delta_path, incremental_path, child_path, child_receipt_path
            )):
                raise RuntimeError("accepted delta artifact missing or symlinked")
            accept, accept_identity = read_json_same_fd(accept_path)
            if accept.get("previous_chain_sha256") != chain:
                raise RuntimeError("accepted delta hash-chain break")
            if accept.get("immutable_parent_checkpoint_sha256") != protocol["parent"]["checkpoint_sha256"]:
                raise RuntimeError("accepted delta immutable-parent binding drift")
            if accept.get("immediate_parent_child_sha256") != expected_immediate_parent:
                raise RuntimeError("accepted delta immediate-parent lineage drift")
            if accept.get("delta_semantics") != "cumulative_current_minus_immutable_parent":
                raise RuntimeError("accepted delta is not cumulative from parent")
            delta_bytes, delta_identity = read_bytes_same_fd(delta_path)
            if delta_identity["sha256"] != accept.get("delta_sha256"):
                raise RuntimeError("accepted delta content hash drift")
            if (accept.get("incremental_delta_semantics")
                    != "accepted_child_minus_previous_branch_child"):
                raise RuntimeError("accepted incremental delta semantics drift")
            _, incremental_identity = read_bytes_same_fd(incremental_path)
            if incremental_identity["sha256"] != accept.get("incremental_delta_sha256"):
                raise RuntimeError("accepted incremental delta content hash drift")
            child_receipt, child_receipt_identity = read_json_same_fd(child_receipt_path)
            child_identity = hash_file_same_fd(child_path)
            child_sha = child_identity["sha256"]
            if child_receipt.get("sha256") != child_sha:
                raise RuntimeError("materialized child checkpoint hash drift")
            if child_receipt.get("immutable_parent_checkpoint_sha256") != protocol["parent"]["checkpoint_sha256"]:
                raise RuntimeError("child receipt immutable-parent binding drift")
            if child_receipt.get("immediate_parent_child_sha256") != expected_immediate_parent:
                raise RuntimeError("child receipt immediate-parent lineage drift")
            if validate_child_payload:
                payload = torch.load(child_path, map_location="cpu", weights_only=True)
                if payload.get("schema") != "hfw_post_training_materialized_child_v1":
                    raise RuntimeError("materialized child payload schema drift")
                if payload.get("immutable_parent_checkpoint_sha256") != protocol["parent"]["checkpoint_sha256"]:
                    raise RuntimeError("materialized child payload immutable-parent drift")
                if payload.get("immediate_parent_child_sha256") != expected_immediate_parent:
                    raise RuntimeError("materialized child payload immediate-parent drift")
                if payload.get("branch_group") != group or payload.get("optimizer_present") is not False:
                    raise RuntimeError("materialized child payload branch/training drift")
                del payload
            expected_immediate_parent = child_sha
            chain = accept_identity["sha256"]
            final_delta_bytes = delta_bytes
            final_delta_identity = delta_identity
            final_child_identity = child_identity
        if proposal.get("previous_chain_sha256") != chain:
            raise RuntimeError("proposal hash-chain break")
        evidence_keys = (
            "center_loss_vector", "center_duplicate_loss_vector",
            "plus_loss_vector", "plus_duplicate_loss_vector",
            "minus_loss_vector", "minus_duplicate_loss_vector",
        )
        if proposal.get("schema") == "hfw_post_training_proven_solver_proposal_v1":
            if not all(
                value is None or math.isfinite(float(value))
                for key in evidence_keys for value in proposal[key]
            ):
                raise RuntimeError("proposal chain contains invalid loss evidence")
            if not all(math.isfinite(float(value)) for value in proposal["center_loss_vector"]):
                raise RuntimeError("proposal center loss is nonfinite")
        elif not all(
            math.isfinite(float(value)) for key in evidence_keys for value in proposal[key]
        ):
            raise RuntimeError("proposal chain contains nonfinite loss evidence")
        chain = proposal_identity["sha256"]
    receipt_path = branch_root / "branch_receipt.json"
    receipt, receipt_file_identity = read_json_same_fd(receipt_path)
    if receipt.get("status") != "PASS_BRANCH_SEARCH_COMPLETE":
        raise RuntimeError("branch receipt status drift")
    if receipt.get("group") != group:
        raise RuntimeError("branch receipt group drift")
    if int(receipt.get("pairs", -1)) != int(protocol["search"]["pairs_per_branch"]):
        raise RuntimeError("branch receipt proposal count drift")
    if int(receipt.get("accepts", -1)) != accept_index:
        raise RuntimeError("branch receipt accepted count drift")
    if receipt.get("complete_proposal_chain_sha256") != chain:
        raise RuntimeError("branch receipt proposal-chain head drift")
    identity = receipt.get("receipt_identity_sha256")
    without_identity = dict(receipt)
    without_identity.pop("receipt_identity_sha256", None)
    if hashlib.sha256(canonical_json_bytes(without_identity)).hexdigest() != identity:
        raise RuntimeError("branch receipt identity drift")
    runtime_path = branch_root / "runtime_guard_receipt.json"
    runtime, runtime_identity = read_json_same_fd(runtime_path)
    if runtime.get("status") != "PASS_NO_TRAINING_RUNTIME_GUARD" or any(
        int(value) for key, value in runtime.items() if key.endswith("attempts")
    ):
        raise RuntimeError("branch runtime guard did not pass")
    summary = {
        "status": "PASS_BRANCH_ARTIFACT_CHAIN",
        "group": group,
        "accepts": accept_index,
        "proposal_chain_sha256": chain,
        "branch_receipt_sha256": receipt_file_identity["sha256"],
        "branch_receipt_inode": receipt_file_identity["inode"],
        "runtime_guard_receipt_sha256": runtime_identity["sha256"],
        "final_materialized_child_sha256": (
            protocol["parent"]["checkpoint_sha256"] if accept_index == 0 else child_sha
        ),
        "final_materialized_child_identity": final_child_identity,
        "final_cumulative_delta_identity": final_delta_identity,
        "receipt": receipt,
    }
    identity_body = {key: value for key, value in summary.items() if key != "receipt"}
    summary["validation_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(identity_body)
    ).hexdigest()
    if include_final_delta_bytes:
        summary["final_cumulative_delta_bytes"] = final_delta_bytes
    return summary
