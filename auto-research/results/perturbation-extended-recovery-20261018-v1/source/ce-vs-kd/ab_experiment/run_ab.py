from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    canonical_json_bytes, file_identity, read_bytes_same_fd, read_json_same_fd,
    sha256_bytes, tensor_f32le_bytes, write_exclusive, write_exclusive_json,
)
from .runtime import (
    AppendOnlyRegistry, MPSStudentObjective, TeacherLogitStore,
    assert_no_conflicting_mps_solver, claim_phase, derived_protocol,
    exact_candidate_materialization, load_bank, load_config,
    load_frozen_kl_module, load_promoted_w560, mps_normalization_adapter,
    require_bound_output, validate_input_closure,
)


def load_exact_flat(record: Mapping[str, object], device: torch.device) -> torch.Tensor:
    body = read_bytes_same_fd(Path(record["path"]), record["sha256"])
    if len(body) != int(record["size_bytes"]):
        raise RuntimeError("candidate exact group-flat size drift")
    values = np.frombuffer(body, dtype="<f4").copy()
    tensor = torch.from_numpy(values).to(device)
    if not torch.isfinite(tensor).all():
        raise RuntimeError("candidate exact group-flat contains nonfinite values")
    return tensor


def state_for_flat(objective: MPSStudentObjective, group: str, flat: torch.Tensor):
    state = dict(objective.group_mapping(group, flat))
    state.update(objective.buffers)
    return state


def run_arm(
    config: Mapping[str, object], mode: str, parent, search_batches: torch.Tensor,
    teacher_store: TeacherLogitStore | None, root: Path,
) -> Mapping[str, object]:
    from vendor.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver import branch_search

    if root.exists():
        raise FileExistsError(f"arm root must be fresh: {root}")
    root.mkdir(parents=True)
    device = torch.device("mps")
    objective = MPSStudentObjective(
        parent, list(search_batches), device, mode=mode, config=config,
        teacher_store=(teacher_store if mode == "kd" else None), teacher_role="SEARCH",
    )
    protocol = derived_protocol(config)
    registry = AppendOnlyRegistry(root / "proposal_registry")
    group = config["experiment"]["group"]
    started = time.monotonic()
    with mps_normalization_adapter(
        branch_search, config["vendor"]["branch_search_sha256"],
    ) as normalization_calls, exact_candidate_materialization(
        branch_search, objective, group, root / "candidate_lineage",
    ) as candidates:
        result = branch_search.run_independent_branch(
            objective, protocol, group, root / "branch", registry, materialize=True,
        )
    if result.proposals != 18 or objective.forward_calls != 180:
        raise RuntimeError(
            f"arm budget drift: proposals={result.proposals}, forwards={objective.forward_calls}"
        )
    if objective.bfloat16_logits != objective.forward_calls:
        raise RuntimeError("not every arm forward produced BF16 logits")
    if len(candidates) != result.accepts:
        raise RuntimeError("accepted candidate sidecar count drift")
    arm = {
        "schema": "hfw_w560_ce_vs_kd_search_arm_terminal_v1",
        "status": "PASS_MATCHED_ARM_SEARCH_COMPLETE",
        "mode": mode,
        "objective": config["objectives"][mode],
        "starting_checkpoint_sha256": config["inputs"]["starting_checkpoint"]["sha256"],
        "starting_head_sha256": config["experiment"]["starting_head_sha256"],
        "group": group,
        "seed": int(config["experiment"]["seed"]),
        "raw_rng_seed_stream_shared_across_arms": True,
        "adaptive_paths_may_diverge_only_after_objective_specific_accepts": True,
        "later_radii_directions_and_candidate_bytes_not_claimed_identical": True,
        "vendor_branch_search_sha256": config["vendor"]["branch_search_sha256"],
        "vendor_solver_bytes_edited": False,
        "model_evaluation_budget": objective.forward_calls,
        "pairs": result.proposals,
        "accepts": result.accepts,
        "final_radius": result.receipt["final_radius"],
        "final_flat_sha256": sha256_bytes(tensor_f32le_bytes(result.final_flat)),
        "candidate_descriptors": candidates,
        "branch_receipt": result.receipt,
        "normalization_calls": normalization_calls["count"],
        "mps_bfloat16_logits": objective.bfloat16_logits,
        "teacher_forward_calls_during_arm": 0,
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "search_bank_sha256": config["banks"]["SEARCH"]["sha256"],
        "verify_bank_opened": False,
        "final_bank_opened": False,
        "wall_seconds": time.monotonic() - started,
    }
    artifact = write_exclusive_json(root / "ARM_SEARCH_TERMINAL.json", arm)
    return {**arm, "terminal_artifact": artifact}


def score_ce_vectors(
    objective: MPSStudentObjective, group: str, flat: torch.Tensor,
    *, duplicate: bool,
) -> tuple[list[float], list[float]]:
    first = []
    second = []
    for batch_index in range(len(objective.calibration)):
        first.extend(float(value) for value in objective.loss_vector(group, flat, batch_index))
        if duplicate:
            second.extend(float(value) for value in objective.loss_vector(group, flat, batch_index))
    if duplicate:
        if len(first) != len(second) or max(
            abs(left - right) for left, right in zip(first, second)
        ) > float(objective.config["selection"]["duplicate_forward_tolerance"]):
            raise RuntimeError("duplicated CE score drift")
    return first, second


def score_kd_vectors(
    config: Mapping[str, object], objective: MPSStudentObjective,
    teacher_store: TeacherLogitStore | None, group: str, flat: torch.Tensor,
) -> Mapping[str, object]:
    if teacher_store is None:
        raise RuntimeError("KD VERIFY selection lacks VERIFY-only teacher capability")
    first_ce, first_kl, first_temperatures = per_sequence_metrics(
        config, objective, teacher_store, "VERIFY", group, flat,
    )
    second_ce, second_kl, second_temperatures = per_sequence_metrics(
        config, objective, teacher_store, "VERIFY", group, flat,
    )
    settings = config["objectives"]["kd"]
    ce_weight = float(settings["ce_weight"])
    distribution_weight = float(settings["distribution_weight"])
    first = [
        ce_weight * ce + distribution_weight * divergence
        for ce, divergence in zip(first_ce, first_kl)
    ]
    second = [
        ce_weight * ce + distribution_weight * divergence
        for ce, divergence in zip(second_ce, second_kl)
    ]
    tolerance = float(config["selection"]["duplicate_forward_tolerance"])
    if any(set(first_row) != set(second_row) for first_row, second_row in zip(
        first_temperatures, second_temperatures,
    )):
        raise RuntimeError("duplicated KD per-temperature metric key drift")
    temperature_duplicate_max = max(
        abs(first_row[key] - second_row[key])
        for first_row, second_row in zip(first_temperatures, second_temperatures)
        for key in first_row
    )
    duplicate_max = max(
        max(abs(a - b) for a, b in zip(first, second)),
        max(abs(a - b) for a, b in zip(first_ce, second_ce)),
        max(abs(a - b) for a, b in zip(first_kl, second_kl)),
        temperature_duplicate_max,
    )
    if duplicate_max > tolerance:
        raise RuntimeError("duplicated KD VERIFY score drift")
    return {
        "objective_first": first,
        "objective_second": second,
        "ce_first": first_ce,
        "ce_second": second_ce,
        "teacher_divergence_first": first_kl,
        "teacher_divergence_second": second_kl,
        "per_temperature_first": first_temperatures,
        "per_temperature_second": second_temperatures,
        "duplicate_max_abs_difference": duplicate_max,
    }


def verify_and_select(
    config: Mapping[str, object], parent, verify_batches: torch.Tensor,
    arm: Mapping[str, object], root: Path, *, mode: str,
    teacher_store: TeacherLogitStore | None,
) -> Mapping[str, object]:
    if root.exists():
        raise FileExistsError("selection root must be fresh")
    root.mkdir(parents=True)
    device = torch.device("mps")
    objective = MPSStudentObjective(
        parent, list(verify_batches), device, mode=mode, config=config,
        teacher_store=teacher_store, teacher_role="VERIFY",
    )
    group = config["experiment"]["group"]
    parent_flat = objective.flat_group(group)
    if mode == "ce":
        parent_first, parent_second = score_ce_vectors(
            objective, group, parent_flat, duplicate=True,
        )
        parent_metrics = {"ce_first": parent_first, "ce_second": parent_second}
        parent_objective = parent_first
    else:
        parent_metrics = score_kd_vectors(config, objective, teacher_store, group, parent_flat)
        parent_objective = parent_metrics["objective_first"]
    parent_mean = math.fsum(parent_objective) / len(parent_objective)
    candidate_results = []
    for descriptor in arm["candidate_descriptors"]:
        flat = load_exact_flat(descriptor["exact_candidate_group_flat"], device)
        if mode == "ce":
            first, second = score_ce_vectors(objective, group, flat, duplicate=True)
            metrics = {"ce_first": first, "ce_second": second}
            objective_values = first
        else:
            metrics = score_kd_vectors(config, objective, teacher_store, group, flat)
            objective_values = metrics["objective_first"]
        mean = math.fsum(objective_values) / len(objective_values)
        candidate_results.append({
            "accept_index": int(descriptor["accept_index"]),
            "exact_candidate_group_flat": descriptor["exact_candidate_group_flat"],
            "cumulative_delta": descriptor["cumulative_delta"],
            "incremental_delta": descriptor["incremental_delta"],
            **metrics,
            "mean_selection_objective": mean,
            "relative_parent_improvement": (parent_mean - mean) / parent_mean,
        })
    minimum = float(config["selection"]["minimum_relative_verify_improvement"])
    passing = [
        item for item in candidate_results
        if item["relative_parent_improvement"] > minimum
    ]
    selected = min(
        passing, key=lambda item: (item["mean_selection_objective"], item["accept_index"]),
    ) if passing else None
    terminal = candidate_results[-1] if candidate_results else None
    selection = {
        "schema": "hfw_w560_ce_vs_kd_identical_verify_selection_v1",
        "status": "PASS_VERIFY_SELECTION_IRREVOCABLY_WRITTEN",
        "arm_mode": arm["mode"],
        "verify_role": "VERIFY_REUSED_SELECTION_DATA_NOT_FINAL",
        "verify_bank_sha256": config["banks"]["VERIFY"]["sha256"],
        "selection_objective": (
            "mean_per_sequence_cross_entropy" if mode == "ce"
            else "frozen_initial_0.1_CE_plus_1.0_exact_sequential_multi_temperature_forward_KL"
        ),
        "objective_specific_verify_selection_is_part_of_each_method": True,
        "minimum_relative_verify_improvement_strictly_greater_than": minimum,
        "parent": {
            "checkpoint_sha256": config["inputs"]["starting_checkpoint"]["sha256"],
            **parent_metrics,
            "mean_selection_objective": parent_mean,
        },
        "all_saved_candidates_evaluated": True,
        "candidate_count": len(candidate_results),
        "candidates": candidate_results,
        "terminal_solver_state": {
            "kind": "parent" if terminal is None else "child",
            "record": {
                "checkpoint_sha256": config["inputs"]["starting_checkpoint"]["sha256"],
            } if terminal is None else terminal,
        },
        "selected_kind": "parent_noop" if selected is None else "child",
        "selected_noop_parent": selected is None,
        "selected": {
            "checkpoint_sha256": config["inputs"]["starting_checkpoint"]["sha256"],
            "exact_candidate_group_flat": None,
            "mean_selection_objective": parent_mean,
        } if selected is None else selected,
        "final_bank_opened": False,
        "teacher_forward_calls": 0,
        "student_forward_calls": objective.forward_calls,
        "optimizer_steps": 0,
        "autograd_calls": 0,
    }
    artifact = write_exclusive_json(root / "VERIFY_SELECTION_TERMINAL.json", selection)
    return {**selection, "terminal_artifact": artifact}


def per_sequence_metrics(
    config: Mapping[str, object], objective: MPSStudentObjective,
    teacher_store: TeacherLogitStore, role: str, group: str, flat: torch.Tensor,
) -> tuple[list[float], list[float], list[Mapping[str, float]]]:
    kl_module = load_frozen_kl_module(config)
    ce_values = []
    distribution_values = []
    per_temperature = []
    offset = 0
    settings = config["objectives"]["kd"]
    for batch_index, raw in enumerate(objective.calibration):
        state = state_for_flat(objective, group, flat)
        logits, targets = objective.logits_and_targets(state, batch_index)
        teacher = teacher_store.batch(
            role, range(offset, offset + raw.shape[0]), torch.device("mps"),
        )
        token_ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.to("mps").reshape(-1),
            reduction="none",
        ).view_as(targets.to("mps"))
        ce_values.extend(float(value) for value in token_ce.mean(dim=1).cpu())
        for index in range(raw.shape[0]):
            distribution, metrics = kl_module.multi_temperature_kl(
                logits[index:index + 1], teacher[index:index + 1],
                settings["temperatures"], int(settings["vocab_chunk_size"]),
            )
            distribution_values.append(float(distribution))
            per_temperature.append({key: float(value) for key, value in metrics.items()})
        offset += raw.shape[0]
    return ce_values, distribution_values, per_temperature


def selected_flat(
    config: Mapping[str, object], objective: MPSStudentObjective,
    selection: Mapping[str, object],
) -> torch.Tensor:
    if selection["selected_kind"] == "parent_noop":
        return objective.flat_group(config["experiment"]["group"])
    return load_exact_flat(selection["selected"]["exact_candidate_group_flat"], torch.device("mps"))


def materialize_winner_checkpoint(
    config: Mapping[str, object], payload: Mapping[str, object], parent,
    flat: torch.Tensor, method: str, selection: Mapping[str, object],
    final_result_sha256: str, path: Path,
) -> Mapping[str, object]:
    from vendor.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver import production_bridge

    group = config["experiment"]["group"]
    state = {name: value.detach().cpu() for name, value in parent.model.state_dict().items()}
    cursor = 0
    for name in parent.groups[group]:
        reference = state[name]
        count = reference.numel()
        state[name] = flat[cursor:cursor + count].view_as(reference).detach().cpu()
        cursor += count
    if cursor != flat.numel():
        raise RuntimeError("winner group-flat length drift")
    model_digest = production_bridge.model_state_digest(state)
    base_flat = torch.cat([
        dict(parent.model.named_parameters())[name].detach().reshape(-1)
        for name in parent.groups[group]
    ])
    cumulative_delta_sha256 = sha256_bytes(tensor_f32le_bytes(flat - base_flat))
    selected = selection["selected"]
    if selected["cumulative_delta"]["sha256"] != cumulative_delta_sha256:
        raise RuntimeError("winning exact group-flat/cumulative-delta relation drift")
    exact_group_bytes = tensor_f32le_bytes(flat)
    cumulative_delta_bytes = tensor_f32le_bytes(flat - base_flat)
    replacement_artifact = write_exclusive(
        path.parent / f"{sha256_bytes(exact_group_bytes)}.group_flat.f32le",
        exact_group_bytes,
    )
    cumulative_delta_artifact = write_exclusive(
        path.parent / f"{cumulative_delta_sha256}.group_delta.f32le",
        cumulative_delta_bytes,
    )
    checkpoint = {
        "schema": "hfw_w560_continuous_promoted_checkpoint_v1",
        "origin_checkpoint_sha256": payload["origin_checkpoint_sha256"],
        "parent_checkpoint_sha256": config["inputs"]["starting_checkpoint"]["sha256"],
        "generation": int(payload["generation"]) + 1,
        "accepted_proposal_sha256": final_result_sha256,
        "selected_candidate_order": int(selected["accept_index"]),
        "selected_candidate_delta_sha256": cumulative_delta_sha256,
        "solver_cumulative_delta_sha256": cumulative_delta_sha256,
        "replacement_group_flat_sha256": replacement_artifact["sha256"],
        "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
        "solver_cumulative_delta_used_to_reconstruct_group": False,
        "model_state_digest": model_digest,
        "optimizer_present": False,
        "scheduler_present": False,
        "training_resume_authorized": False,
        "model": state,
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    checkpoint_identity = file_identity(path)
    reloaded = torch.load(
        io.BytesIO(read_bytes_same_fd(path, checkpoint_identity["sha256"])),
        map_location="cpu", weights_only=True,
    )
    expected_checkpoint_keys = {
        "schema", "origin_checkpoint_sha256", "parent_checkpoint_sha256", "generation",
        "accepted_proposal_sha256", "selected_candidate_order",
        "selected_candidate_delta_sha256", "solver_cumulative_delta_sha256",
        "replacement_group_flat_sha256", "group_replacement_materialization_source",
        "solver_cumulative_delta_used_to_reconstruct_group", "model_state_digest",
        "optimizer_present", "scheduler_present", "training_resume_authorized", "model",
    }
    if set(reloaded) != expected_checkpoint_keys:
        raise RuntimeError("winner checkpoint key inventory drift after same-FD reload")
    if production_bridge.model_state_digest(reloaded["model"]) != model_digest:
        raise RuntimeError("winner checkpoint model digest drift after same-FD reload")
    reloaded_flat = torch.cat([
        reloaded["model"][name].reshape(-1) for name in parent.groups[group]
    ])
    if sha256_bytes(tensor_f32le_bytes(reloaded_flat)) != replacement_artifact["sha256"]:
        raise RuntimeError("winner checkpoint does not contain exact replacement bytes")
    return {
        "checkpoint": {
            **checkpoint_identity,
            "format": "hfw_w560_continuous_promoted_checkpoint_v1",
            "model_state_digest": model_digest,
        },
        "replacement_group_flat": replacement_artifact,
        "solver_cumulative_delta": cumulative_delta_artifact,
    }


def final_evaluate(
    config_path: Path, teacher_manifest_path: Path, run_root: Path,
) -> Mapping[str, object]:
    raise RuntimeError(
        "monolithic A/B execution is forbidden; use least-authority arm, verify, and final phases"
    )


def run_arm_phase(
    config_path: Path, mode: str, output_root: Path,
    teacher_capability_path: Path | None = None,
) -> Mapping[str, object]:
    config = load_config(config_path)
    if config["phase"] != f"{mode.upper()}_SEARCH":
        raise RuntimeError("search phase/config identity drift")
    if set(config["banks"]) != {"SEARCH"}:
        raise RuntimeError("SEARCH process received a non-SEARCH token role")
    encoded_config = canonical_json_bytes(config).lower()
    if b"final" in encoded_config or b"verify" in encoded_config:
        raise RuntimeError("SEARCH phase config exposes VERIFY or FINAL capability")
    assert_no_conflicting_mps_solver(config)
    validate_input_closure(config)
    require_bound_output(config, output_root)
    teacher_store = None
    if mode == "ce":
        if teacher_capability_path is not None or "teacher_outputs" in config["inputs"]:
            raise RuntimeError("CE SEARCH must not receive teacher-logit capability")
    else:
        if teacher_capability_path is None:
            raise RuntimeError("KD SEARCH requires SEARCH-only teacher capability")
        teacher_store = TeacherLogitStore(config, teacher_capability_path)
        if set(teacher_store.by_role) != {"SEARCH"}:
            raise RuntimeError("KD SEARCH teacher capability role drift")
    search, _ = load_bank(config["banks"]["SEARCH"])
    claim_phase(config, config_path)
    parent, _ = load_promoted_w560(config, torch.device("mps"))
    result = run_arm(
        config, mode, parent, search.reshape(12, 2, 513), teacher_store, output_root,
    )
    if teacher_store is not None:
        teacher_store.clear()
    return result


def run_verify_phase(
    config_path: Path, mode: str, arm_terminal_path: Path,
    output_root: Path, teacher_capability_path: Path | None = None,
) -> Mapping[str, object]:
    config = load_config(config_path)
    if config["phase"] != f"{mode.upper()}_VERIFY":
        raise RuntimeError("VERIFY phase/config identity drift")
    if set(config["banks"]) != {"VERIFY"}:
        raise RuntimeError("VERIFY process received a non-VERIFY token role")
    encoded_config = canonical_json_bytes(config).lower()
    if b"final" in encoded_config or b"search.u32" in encoded_config:
        raise RuntimeError("VERIFY phase config exposes SEARCH or FINAL token capability")
    assert_no_conflicting_mps_solver(config)
    validate_input_closure(config)
    require_bound_output(config, output_root)
    expected_arm = config["phase_inputs"]["arm_terminal"]
    if arm_terminal_path.resolve() != Path(expected_arm["path"]).resolve():
        raise RuntimeError("arm terminal path is not the unique preregistered output")
    arm = read_json_same_fd(arm_terminal_path, expected_arm["sha256"])
    if arm.get("status") != "PASS_MATCHED_ARM_SEARCH_COMPLETE" or arm.get("mode") != mode:
        raise RuntimeError("arm terminal phase binding drift")
    teacher_store = None
    if mode == "ce":
        if teacher_capability_path is not None or "teacher_outputs" in config["inputs"]:
            raise RuntimeError("CE VERIFY must not receive teacher-logit capability")
    else:
        if teacher_capability_path is None:
            raise RuntimeError("KD VERIFY requires VERIFY-only teacher capability")
        teacher_store = TeacherLogitStore(config, teacher_capability_path)
        if set(teacher_store.by_role) != {"VERIFY"}:
            raise RuntimeError("KD VERIFY teacher capability role drift")
    verify, _ = load_bank(config["banks"]["VERIFY"])
    claim_phase(config, config_path)
    parent, _ = load_promoted_w560(config, torch.device("mps"))
    result = verify_and_select(
        config, parent, verify.reshape(8, 4, 513), arm, output_root,
        mode=mode, teacher_store=teacher_store,
    )
    if teacher_store is not None:
        teacher_store.clear()
    return result


def run_final_phase(
    config_path: Path, final_teacher_capability_path: Path,
    ce_selection_path: Path, kd_selection_path: Path, run_root: Path,
) -> Mapping[str, object]:
    config = load_config(config_path)
    if config["phase"] != "FINAL":
        raise RuntimeError("FINAL phase/config identity drift")
    if set(config["banks"]) != {"FINAL"}:
        raise RuntimeError("FINAL process received a non-FINAL token role")
    assert_no_conflicting_mps_solver(config)
    validate_input_closure(config)
    require_bound_output(config, run_root)
    if run_root.exists():
        raise FileExistsError("A/B run root must be fresh")
    config_identity = file_identity(config_path)
    teacher_manifest_identity = file_identity(final_teacher_capability_path)
    expected_teacher_manifest = config["inputs"]["teacher_outputs"]["manifest_sha256"]
    if teacher_manifest_identity["sha256"] != expected_teacher_manifest:
        raise RuntimeError("FINAL teacher capability launch binding drift")
    teacher_store = TeacherLogitStore(config, final_teacher_capability_path)
    if set(teacher_store.by_role) != {"FINAL"}:
        raise RuntimeError("FINAL process received non-FINAL teacher logits")
    ce_expected = config["phase_inputs"]["ce_selection"]
    kd_expected = config["phase_inputs"]["kd_selection"]
    if ce_selection_path.resolve() != Path(ce_expected["path"]).resolve():
        raise RuntimeError("CE selection path is not preregistered")
    if kd_selection_path.resolve() != Path(kd_expected["path"]).resolve():
        raise RuntimeError("KD selection path is not preregistered")
    ce_selection = read_json_same_fd(ce_selection_path, ce_expected["sha256"])
    kd_selection = read_json_same_fd(kd_selection_path, kd_expected["sha256"])
    for mode, selection in (("ce", ce_selection), ("kd", kd_selection)):
        if selection.get("status") != "PASS_VERIFY_SELECTION_IRREVOCABLY_WRITTEN":
            raise RuntimeError(f"{mode} selection is not terminal")
        if selection.get("arm_mode") != mode:
            raise RuntimeError(f"{mode} selection mode drift")
    claim_phase(config, config_path)
    run_root.mkdir(parents=True)
    device = torch.device("mps")
    parent, starting_payload = load_promoted_w560(config, device)
    selection_seal = {
        "schema": "hfw_w560_ce_vs_kd_both_selections_sealed_v1",
        "status": "PASS_FINAL_MAY_NOW_OPEN_ONCE",
        "config_sha256": config_identity["sha256"],
        "teacher_final_capability_sha256": teacher_manifest_identity["sha256"],
        "ce_selection_sha256": ce_expected["sha256"],
        "kd_selection_sha256": kd_expected["sha256"],
        "both_selected_exact_candidate_bytes_irrevocably_written": True,
        "final_student_open_count_before_seal": 0,
    }
    seal_artifact = write_exclusive_json(run_root / "BOTH_SELECTIONS_SEALED.json", selection_seal)

    final_bank, final_bank_identity = load_bank(config["banks"]["FINAL"])
    final_batches = final_bank.reshape(16, 4, 513)
    labels = {
        "parent": None,
        "ce_selected_output": ce_selection,
        "kd_selected_output": kd_selection,
    }
    scores = {}
    group = config["experiment"]["group"]
    for label, selection in labels.items():
        objective = MPSStudentObjective(
            parent, list(final_batches), device, mode="ce", config=config,
        )
        flat = objective.flat_group(group) if selection is None else selected_flat(
            config, objective, selection,
        )
        first_ce, first_kl, first_temperatures = per_sequence_metrics(
            config, objective, teacher_store, "FINAL", group, flat,
        )
        second_ce, second_kl, second_temperatures = per_sequence_metrics(
            config, objective, teacher_store, "FINAL", group, flat,
        )
        tolerance = float(config["selection"]["duplicate_forward_tolerance"])
        if any(set(first_row) != set(second_row) for first_row, second_row in zip(
            first_temperatures, second_temperatures,
        )):
            raise RuntimeError("FINAL duplicate per-temperature metric key drift")
        temperature_duplicate_max = max(
            abs(first_row[key] - second_row[key])
            for first_row, second_row in zip(first_temperatures, second_temperatures)
            for key in first_row
        )
        duplicate_max = max(
            max(abs(a - b) for a, b in zip(first_ce, second_ce)),
            max(abs(a - b) for a, b in zip(first_kl, second_kl)),
            temperature_duplicate_max,
        )
        if duplicate_max > tolerance:
            raise RuntimeError("FINAL duplicate score drift")
        scores[label] = {
            "selected_kind": "parent" if selection is None else selection["selected_kind"],
            "exact_group_flat_sha256": sha256_bytes(tensor_f32le_bytes(flat)),
            "ce_first": first_ce,
            "ce_second": second_ce,
            "teacher_divergence_first": first_kl,
            "teacher_divergence_second": second_kl,
            "per_temperature_first": first_temperatures,
            "per_temperature_second": second_temperatures,
            "mean_ce": math.fsum(first_ce) / len(first_ce),
            "mean_teacher_divergence": math.fsum(first_kl) / len(first_kl),
            "duplicate_max_abs_difference": duplicate_max,
            "student_forward_calls": objective.forward_calls,
        }
        teacher_store.clear()
        del objective, flat
        gc.collect()
        torch.mps.empty_cache()

    left = scores["ce_selected_output"]
    right = scores["kd_selected_output"]
    if left["mean_ce"] < right["mean_ce"]:
        winning_method = "ce"
    elif right["mean_ce"] < left["mean_ce"]:
        winning_method = "kd"
    elif left["mean_teacher_divergence"] <= right["mean_teacher_divergence"]:
        winning_method = "ce"
    else:
        winning_method = "kd"
    winning_label = f"{winning_method}_selected_output"
    winner = scores[winning_label]
    parent_score = scores["parent"]
    relative_parent_improvement = (
        parent_score["mean_ce"] - winner["mean_ce"]
    ) / parent_score["mean_ce"]
    promotion_gate = float(config["selection"]["minimum_relative_final_parent_improvement"])
    promote = relative_parent_improvement > promotion_gate
    paired = {
        "ce_selected_output_minus_parent_ce": [
            child - base for child, base in zip(scores["ce_selected_output"]["ce_first"], parent_score["ce_first"])
        ],
        "kd_selected_output_minus_parent_ce": [
            child - base for child, base in zip(scores["kd_selected_output"]["ce_first"], parent_score["ce_first"])
        ],
        "kd_selected_output_minus_ce_selected_output_ce": [
            kd - ce for kd, ce in zip(scores["kd_selected_output"]["ce_first"], scores["ce_selected_output"]["ce_first"])
        ],
        "ce_selected_output_wins_vs_kd_selected_output": sum(
            ce < kd for ce, kd in zip(scores["ce_selected_output"]["ce_first"], scores["kd_selected_output"]["ce_first"])
        ),
        "kd_selected_output_wins_vs_ce_selected_output": sum(
            kd < ce for ce, kd in zip(scores["ce_selected_output"]["ce_first"], scores["kd_selected_output"]["ce_first"])
        ),
        "ties": sum(
            kd == ce for ce, kd in zip(scores["ce_selected_output"]["ce_first"], scores["kd_selected_output"]["ce_first"])
        ),
    }
    result_without_sha = {
        "schema": "hfw_w560_ce_vs_kd_matched_ab_final_result_v1",
        "status": "PASS_FINAL_OPENED_ONCE_AFTER_BOTH_SELECTIONS",
        "config": config_identity,
        "teacher_outputs_manifest": teacher_manifest_identity,
        "selection_seal": seal_artifact,
        "final_bank": final_bank_identity,
        "final_disclosure": "ONE_TIME_METHOD_SELECTION_DATA_NOT_REUSABLE_HELD_OUT_EVALUATION",
        "scores": scores,
        "paired_sequence_statistics": paired,
        "method_winner_rule": "lower_FINAL_mean_CE_then_lower_FINAL_teacher_divergence_exact_tie_break",
        "winning_method": winning_method,
        "winning_label": winning_label,
        "winner_relative_parent_ce_improvement": relative_parent_improvement,
        "promotion_rule": "strict_relative_parent_CE_improvement_greater_than_0.0001",
        "promotion_gate": promotion_gate,
        "winning_method_selected_kind": (
            ce_selection["selected_kind"] if winning_method == "ce"
            else kd_selection["selected_kind"]
        ),
        "promote_winner_selected_output": promote,
        "continuation_checkpoint_choice": "winning_selected_output" if promote else "unchanged_parent",
        "continuation_objective": winning_method,
        "no_regression_gate_passed": promote,
        "teacher_forward_calls_during_search_verify_final": 0,
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "final_must_never_be_reused_by_continuation": True,
    }
    result_sha = sha256_bytes(canonical_json_bytes(result_without_sha))
    result = {**result_without_sha, "result_identity_sha256": result_sha}
    result_artifact = write_exclusive_json(run_root / "FINAL_RESULT.json", result)

    if promote:
        selection = ce_selection if winning_method == "ce" else kd_selection
        objective = MPSStudentObjective(
            parent, [final_batches[0]], device, mode="ce", config=config,
        )
        flat = selected_flat(config, objective, selection)
        promoted = materialize_winner_checkpoint(
            config, starting_payload, parent, flat, winning_method, selection,
            result_artifact["sha256"], run_root / "winner_checkpoint.pt",
        )
    else:
        promoted = None
    source_head = read_json_same_fd(
        Path(config["inputs"]["starting_head"]["path"]),
        config["inputs"]["starting_head"]["sha256"],
    )
    if promote:
        continuation_head = {
            "schema": "hfw_w560_ce_vs_kd_winner_export_snapshot_v1",
            "status": "PASS_A_B_WINNER_EXPORT_REQUIRES_FRESH_CONTINUATION_CONTRACT",
            "generation": int(starting_payload["generation"]) + 1,
            "local_promotions": int(source_head["local_promotions"]) + 1,
            "parent_head_record_sha256": config["experiment"]["starting_head_sha256"],
            "source_current_head_record_sha256": source_head["source_current_head_record_sha256"],
            "parent_checkpoint_sha256": config["inputs"]["starting_checkpoint"]["sha256"],
            "checkpoint": promoted["checkpoint"],
            "origin_checkpoint_sha256": starting_payload["origin_checkpoint_sha256"],
            "model_state_digest": promoted["checkpoint"]["model_state_digest"],
            "accepted_proposal_sha256": result_artifact["sha256"],
            "selected_candidate_order": int(selection["selected"]["accept_index"]),
            "selection_result_sha256": result_artifact["sha256"],
            "delta_sha256": promoted["solver_cumulative_delta"]["sha256"],
            "solver_cumulative_delta_sha256": promoted["solver_cumulative_delta"]["sha256"],
            "replacement_group_flat_sha256": promoted["replacement_group_flat"]["sha256"],
            "solver_cumulative_delta": promoted["solver_cumulative_delta"],
            "replacement_group_flat": promoted["replacement_group_flat"],
            "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
            "group": group,
            "local_verify_evidence": {
                "role": "A_B_OBJECTIVE_SPECIFIC_VERIFY_THEN_ONE_TIME_FINAL_SELECTION",
                "final_result_sha256": result_artifact["sha256"],
                "winning_method": winning_method,
                "selected_output_kind": selection["selected_kind"],
                "strict_final_parent_ce_gate_passed": True,
            },
            "winning_objective": winning_method,
            "old_attempt3_contract_or_runtime_identity_inherited": False,
        }
    else:
        continuation_head = {
            "schema": "hfw_w560_ce_vs_kd_winner_export_snapshot_v1",
            "status": "PASS_A_B_PARENT_RETAINED_EXPORT_REQUIRES_FRESH_CONTINUATION_CONTRACT",
            "generation": int(source_head["generation"]),
            "local_promotions": int(source_head["local_promotions"]),
            "parent_head_record_sha256": source_head["parent_head_record_sha256"],
            "source_current_head_record_sha256": config["experiment"]["starting_head_sha256"],
            "parent_checkpoint_sha256": source_head["parent_checkpoint_sha256"],
            "checkpoint": {
                **config["inputs"]["starting_checkpoint"],
                "format": "hfw_w560_continuous_promoted_checkpoint_v1",
            },
            "origin_checkpoint_sha256": source_head["origin_checkpoint_sha256"],
            "model_state_digest": source_head["model_state_digest"],
            "accepted_proposal_sha256": source_head["accepted_proposal_sha256"],
            "selected_candidate_order": int(source_head["selected_candidate_order"]),
            "selection_result_sha256": result_artifact["sha256"],
            "delta_sha256": source_head["delta_sha256"],
            "solver_cumulative_delta_sha256": source_head["solver_cumulative_delta_sha256"],
            "replacement_group_flat_sha256": source_head["replacement_group_flat_sha256"],
            "solver_cumulative_delta": source_head["solver_cumulative_delta"],
            "replacement_group_flat": source_head["replacement_group_flat"],
            "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
            "group": source_head["group"],
            "local_verify_evidence": source_head["local_verify_evidence"],
            "winning_objective": winning_method,
            "old_attempt3_contract_or_runtime_identity_inherited": False,
        }
    head_snapshot_artifact = write_exclusive_json(
        run_root / "WINNER_EXPORT_SNAPSHOT.json", continuation_head,
    )
    handoff = {
        "schema": "hfw_w560_ce_vs_kd_winner_continuation_handoff_v1",
        "status": "PASS_WINNER_SELECTED_NO_REGRESSION_GATE_APPLIED",
        "final_result_sha256": result_artifact["sha256"],
        "winning_method": winning_method,
        "checkpoint": continuation_head["checkpoint"],
        "head_snapshot": head_snapshot_artifact,
        "head_snapshot_record": continuation_head,
        "replacement_group_flat": continuation_head["replacement_group_flat"],
        "solver_cumulative_delta": continuation_head["solver_cumulative_delta"],
        "winner_selected_output_promoted": promote,
        "final_bank_or_teacher_final_logits_authorized_for_continuation": False,
        "continuation_search_bank_sha256": config["phase_inputs"]["continuation_search_bank_sha256"],
        "continuation_verify_bank_sha256": config["phase_inputs"]["continuation_verify_bank_sha256"],
    }
    handoff_artifact = write_exclusive_json(run_root / "WINNER_HANDOFF.json", handoff)
    return {**result, "result_artifact": result_artifact, "handoff_artifact": handoff_artifact}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm_parser = subparsers.add_parser("arm")
    arm_parser.add_argument("--config", type=Path, required=True)
    arm_parser.add_argument("--mode", choices=("ce", "kd"), required=True)
    arm_parser.add_argument("--output-root", type=Path, required=True)
    arm_parser.add_argument("--teacher-capability", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--config", type=Path, required=True)
    verify_parser.add_argument("--mode", choices=("ce", "kd"), required=True)
    verify_parser.add_argument("--arm-terminal", type=Path, required=True)
    verify_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser.add_argument("--teacher-capability", type=Path)
    final_parser = subparsers.add_parser("final")
    final_parser.add_argument("--config", type=Path, required=True)
    final_parser.add_argument("--teacher-capability", type=Path, required=True)
    final_parser.add_argument("--ce-selection", type=Path, required=True)
    final_parser.add_argument("--kd-selection", type=Path, required=True)
    final_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "arm":
        result = run_arm_phase(
            args.config, args.mode, args.output_root, args.teacher_capability,
        )
    elif args.command == "verify":
        result = run_verify_phase(
            args.config, args.mode, args.arm_terminal, args.output_root,
            args.teacher_capability,
        )
    else:
        result = run_final_phase(
            args.config, args.teacher_capability, args.ce_selection,
            args.kd_selection, args.run_root,
        )
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
