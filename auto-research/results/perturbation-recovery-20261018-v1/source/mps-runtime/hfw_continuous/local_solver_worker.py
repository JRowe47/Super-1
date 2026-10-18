from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import torch

from .common import (
    canonical_json_bytes, hash_file_same_fd, read_bytes_same_fd, read_json_same_fd,
    write_content_addressed_bytes, write_content_addressed_json,
    write_exclusive_bytes, write_exclusive_json,
)
from .mps_adapter import (
    NUMERICAL_LANE, adapter_contract, make_mps_bfloat16_objective_class,
    mps_functional_normalization,
)
from .local_candidate_import import LOCAL_SCHEMA
from .local_config import load_config, runtime_root
from .local_ledger import LocalLedger
from .relocation import vendor_modules
from .state import load_head_model


class JobLocalTraceRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.count = 0

    def append(
        self, *, proposal_sha256: str, delta_sha256: str,
        child_checkpoint_sha256: str | None, accepted: bool,
        reason: str, record: Mapping[str, object],
    ) -> str:
        event = {
            "schema": "hfw_w560_mps_job_local_provisional_solver_event_v1",
            "event_index": self.count,
            "proposal_sha256": proposal_sha256,
            "delta_sha256": delta_sha256,
            "artifact_only_child_sha256": child_checkpoint_sha256,
            "provisionally_accepted_inside_solver": bool(accepted),
            "locally_promoted": False,
            "local_checkpoint_lineage_changed": False,
            "reason": reason,
            "record": dict(record),
        }
        payload = canonical_json_bytes(event)
        digest = hashlib.sha256(payload).hexdigest()
        write_exclusive_json(self.root / f"{self.count:04d}-{digest}.event.json", event)
        self.count += 1
        return digest


@contextmanager
def artifact_only_projection(branch_module, adapter_identity: Mapping[str, object]):
    original = branch_module._materialize_child

    def _write_delta_child(
        objective, group: str, flat: torch.Tensor, path: Path,
        immutable_parent_sha256: str, immediate_parent_child_sha256: str,
        chain_sha256: str,
    ) -> str:
        match = re.fullmatch(r"materialized_child_(\d+)\.pt", path.name)
        if match is None:
            raise RuntimeError("unexpected provisional child filename")
        accept_index = int(match.group(1))
        cumulative_path = path.parent / f"accepted_{accept_index:02d}_delta.f32le"
        cumulative = hash_file_same_fd(cumulative_path)
        expected = branch_module._tensor_bytes(flat - objective.flat_group(group))
        if cumulative["sha256"] != hashlib.sha256(expected).hexdigest():
            raise RuntimeError("artifact adapter cumulative delta differs from exact solver state")
        exact_group_payload = branch_module._tensor_bytes(flat)
        exact_group_artifact = write_exclusive_bytes(
            path.parent / f"accepted_{accept_index:02d}_group_flat.f32le",
            exact_group_payload,
        )
        descriptor = {
            "schema": "hfw_w560_mps_provisional_delta_child_descriptor_v1",
            "artifact_projection_only": True,
            "target_branch_seed_changed_by_append_only_job_allocator": True,
            "parent_checkpoint_sha256_job_bound": True,
            "non_seed_solver_settings_radius_policy_proposal_count_and_evaluation_order_unchanged": True,
            "numerical_backend_adapted": True,
            "numerical_lane": NUMERICAL_LANE,
            "adapter_source_sha256": adapter_identity["adapter_source"]["sha256"],
            "cuda_bit_identical_claimed": False,
            "group": group,
            "immutable_job_base_checkpoint_sha256": immutable_parent_sha256,
            "immediate_parent_artifact_sha256": immediate_parent_child_sha256,
            "accepted_chain_sha256": chain_sha256,
            "cumulative_group_delta": cumulative,
            "exact_candidate_group_flat": exact_group_artifact,
            "candidate_group_flat_sha256": exact_group_artifact["sha256"],
            "cumulative_delta_relation": "float32_exact_candidate_group_flat_minus_exact_job_base_group",
            "base_plus_cumulative_delta_is_not_an_authoritative_reconstruction": True,
            "full_provisional_checkpoint_materialized": False,
            "local_head_write_authorized": False,
            "duplicated_local_verify_required": True,
        }
        return write_exclusive_json(path, descriptor)["sha256"]

    branch_module._materialize_child = _write_delta_child
    try:
        yield
    finally:
        branch_module._materialize_child = original


def _job_protocol(
    config: Mapping[str, object], base_checkpoint_sha256: str, group: str, seed: int,
) -> Mapping[str, object]:
    package_root, _, _, _, _ = vendor_modules(config)
    protocol, _ = read_json_same_fd(
        package_root / "config/production_protocol.json",
        config["vendor"]["production_protocol_sha256"],
    )
    protocol = deepcopy(protocol)
    protocol["parent"]["checkpoint_sha256"] = base_checkpoint_sha256
    protocol["search"]["branch_seeds"][group] = int(seed)
    return protocol


def validate_job_protocol_overrides(
    base_protocol: Mapping[str, object], protocol: Mapping[str, object],
    base_checkpoint_sha256: str, group: str, seed: int,
) -> Mapping[str, object]:
    expected = deepcopy(base_protocol)
    expected["parent"]["checkpoint_sha256"] = str(base_checkpoint_sha256)
    expected["search"]["branch_seeds"][group] = int(seed)
    if protocol != expected:
        raise RuntimeError(
            "job protocol changed a field other than parent checkpoint SHA or target branch seed"
        )
    frozen_seed = int(base_protocol["search"]["branch_seeds"][group])
    if int(seed) == frozen_seed:
        raise RuntimeError("job allocator did not change the target branch seed")
    for other_group, frozen_value in base_protocol["search"]["branch_seeds"].items():
        if other_group == group:
            continue
        if int(protocol["search"]["branch_seeds"][other_group]) != int(frozen_value):
            raise RuntimeError("non-target branch seed changed")
    return {
        "schema": "hfw_w560_mps_job_protocol_override_validation_v1",
        "status": "PASS_ONLY_PARENT_CHECKPOINT_SHA_AND_TARGET_BRANCH_SEED_JOB_BOUND",
        "job_bound_fields": [
            "parent.checkpoint_sha256",
            f"search.branch_seeds.{group}",
        ],
        "target_branch_seed_changed_by_append_only_job_allocator": True,
        "frozen_target_branch_seed": frozen_seed,
        "job_target_branch_seed": int(seed),
        "parent_checkpoint_sha256_job_bound": True,
        "non_target_branch_seeds_unchanged": True,
        "non_seed_solver_settings_unchanged": True,
        "radius_policy_unchanged": True,
        "proposal_count_unchanged": True,
        "within_job_evaluation_order_unchanged": True,
    }


def validate_exact_candidate_sidecar_relation(
    branch_module, base_flat: torch.Tensor, previous_flat: torch.Tensor,
    exact_path: Path, exact_sha256: str,
    delta_path: Path, delta_sha256: str,
    incremental_path: Path, incremental_sha256: str,
) -> tuple[torch.Tensor, Mapping[str, object]]:
    exact_payload, _ = read_bytes_same_fd(exact_path, exact_sha256)
    delta_payload, _ = read_bytes_same_fd(delta_path, delta_sha256)
    incremental_payload, _ = read_bytes_same_fd(incremental_path, incremental_sha256)
    exact_values = np.frombuffer(exact_payload, dtype="<f4")
    delta_values = np.frombuffer(delta_payload, dtype="<f4")
    incremental_values = np.frombuffer(incremental_payload, dtype="<f4")
    if not (
        exact_values.size == delta_values.size == incremental_values.size == base_flat.numel()
    ):
        raise RuntimeError("MPS candidate sidecar element count drift")
    candidate_flat = torch.from_numpy(exact_values.copy()).to(
        device=base_flat.device, dtype=base_flat.dtype,
    )
    delta = torch.from_numpy(delta_values.copy()).to(
        device=base_flat.device, dtype=base_flat.dtype,
    )
    incremental = torch.from_numpy(incremental_values.copy()).to(
        device=base_flat.device, dtype=base_flat.dtype,
    )
    if not all(torch.isfinite(value).all() for value in (candidate_flat, delta, incremental)):
        raise RuntimeError("MPS candidate sidecars contain nonfinite values")
    candidate_minus_base = branch_module._tensor_bytes(candidate_flat - base_flat)
    if hashlib.sha256(candidate_minus_base).hexdigest() != delta_sha256:
        raise RuntimeError("MPS cumulative delta is not exact float32 candidate-minus-base")
    candidate_minus_previous = branch_module._tensor_bytes(candidate_flat - previous_flat)
    if hashlib.sha256(candidate_minus_previous).hexdigest() != incremental_sha256:
        raise RuntimeError("MPS incremental delta is not exact float32 candidate-minus-previous")
    return candidate_flat, {
        "schema": "hfw_w560_mps_exact_candidate_delta_relation_v1",
        "exact_candidate_group_flat_sha256": exact_sha256,
        "cumulative_delta_sha256": delta_sha256,
        "incremental_delta_sha256": incremental_sha256,
        "serialized_float32_subtraction_matches_cumulative_delta_bitwise": True,
        "serialized_float32_candidate_minus_previous_matches_incremental_delta_bitwise": True,
        "evaluation_uses_exact_candidate_group_flat_not_base_plus_delta": True,
    }


def validate_artifact_candidate_chain(
    branch_module, protocol: Mapping[str, object], branch_root: Path, group: str,
    base_flat: torch.Tensor,
) -> tuple[Mapping[str, object], list[Mapping[str, object]]]:
    frozen_validation = branch_module.validate_branch_artifact_chain(
        protocol, branch_root, group, include_final_delta_bytes=False,
        validate_child_payload=False,
    )
    header, header_identity = read_json_same_fd(branch_root / "branch_header.json")
    search = protocol["search"]
    exact_header = {
        "schema": "hfw_post_training_proven_solver_branch_header_v1",
        "solver": search["solver"],
        "group": group,
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "branch_seed": int(search["branch_seeds"][group]),
        "model_evaluation_budget": int(search["model_evaluation_budget"]),
        "iterations": int(search["pairs_per_branch"]),
        "paired_direction_count": int(search["paired_direction_count"]),
        "paired_top_k": int(search["paired_top_k"]),
        "line_search_scales": [float(value) for value in search["line_search_scales"]],
        "initial_radius": float(search["initial_radius"]),
        "minimum_relative_improvement": float(search["minimum_relative_improvement"]),
        "fixed_eval_opened": False,
        "optimizer_steps": 0,
        "autograd_calls": 0,
    }
    for key, expected in exact_header.items():
        if header.get(key) != expected:
            raise RuntimeError(f"frozen solver branch header drift: {key}")
    receipt = frozen_validation["receipt"]
    receipt_requirements = {
        "schema": "hfw_post_training_proven_solver_branch_receipt_v1",
        "status": "PASS_BRANCH_SEARCH_COMPLETE",
        "solver": search["solver"],
        "group": group,
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "pairs": int(search["pairs_per_branch"]),
        "maximum_accepts": int(search["maximum_accepts_per_branch"]),
        "forward_calls": int(search["model_evaluation_budget"]),
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "new_unique_tokens_after_initialization": 0,
        "fixed_eval_opened": False,
    }
    for key, expected in receipt_requirements.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"frozen solver branch receipt drift: {key}")
    candidates = []
    previous_flat = base_flat
    for accept_index in range(1, int(frozen_validation["accepts"]) + 1):
        accepted, accepted_identity = read_json_same_fd(
            branch_root / f"accepted_{accept_index:02d}.json"
        )
        delta_identity = hash_file_same_fd(
            branch_root / f"accepted_{accept_index:02d}_delta.f32le"
        )
        incremental_identity = hash_file_same_fd(
            branch_root / f"accepted_{accept_index:02d}_incremental_delta.f32le"
        )
        exact_group_identity = hash_file_same_fd(
            branch_root / f"accepted_{accept_index:02d}_group_flat.f32le"
        )
        descriptor, descriptor_identity = read_json_same_fd(
            branch_root / f"materialized_child_{accept_index:02d}.pt"
        )
        child_receipt, child_receipt_identity = read_json_same_fd(
            branch_root / f"materialized_child_{accept_index:02d}.receipt.json"
        )
        requirements = {
            "schema": "hfw_w560_mps_provisional_delta_child_descriptor_v1",
            "artifact_projection_only": True,
            "target_branch_seed_changed_by_append_only_job_allocator": True,
            "parent_checkpoint_sha256_job_bound": True,
            "non_seed_solver_settings_radius_policy_proposal_count_and_evaluation_order_unchanged": True,
            "numerical_backend_adapted": True,
            "numerical_lane": NUMERICAL_LANE,
            "cuda_bit_identical_claimed": False,
            "full_provisional_checkpoint_materialized": False,
            "local_head_write_authorized": False,
            "duplicated_local_verify_required": True,
            "group": group,
            "immutable_job_base_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
            "accepted_chain_sha256": accepted_identity["sha256"],
            "candidate_group_flat_sha256": exact_group_identity["sha256"],
            "cumulative_delta_relation": "float32_exact_candidate_group_flat_minus_exact_job_base_group",
            "base_plus_cumulative_delta_is_not_an_authoritative_reconstruction": True,
        }
        for key, expected in requirements.items():
            if descriptor.get(key) != expected:
                raise RuntimeError(f"MPS candidate descriptor drift: {key}")
        if int(accepted.get("accept_index", -1)) != accept_index:
            raise RuntimeError("accepted candidate order drift")
        if accepted.get("group") != group:
            raise RuntimeError("accepted candidate group drift")
        if accepted.get("delta_semantics") != "cumulative_current_minus_immutable_parent":
            raise RuntimeError("accepted candidate delta semantics drift")
        if accepted.get("delta_sha256") != delta_identity["sha256"]:
            raise RuntimeError("accepted candidate delta identity drift")
        if accepted.get("incremental_delta_sha256") != incremental_identity["sha256"]:
            raise RuntimeError("accepted candidate incremental delta identity drift")
        if descriptor.get("cumulative_group_delta", {}).get("sha256") != delta_identity["sha256"]:
            raise RuntimeError("MPS descriptor cumulative delta SHA drift")
        if descriptor.get("exact_candidate_group_flat", {}).get("sha256") != exact_group_identity["sha256"]:
            raise RuntimeError("MPS descriptor exact group-flat SHA drift")
        if child_receipt.get("sha256") != descriptor_identity["sha256"]:
            raise RuntimeError("artifact-only child receipt descriptor binding drift")
        if child_receipt.get("accepted_chain_sha256") != accepted_identity["sha256"]:
            raise RuntimeError("artifact-only child receipt accepted-chain drift")
        candidate_flat, relation = validate_exact_candidate_sidecar_relation(
            branch_module, base_flat, previous_flat,
            branch_root / f"accepted_{accept_index:02d}_group_flat.f32le",
            exact_group_identity["sha256"],
            branch_root / f"accepted_{accept_index:02d}_delta.f32le",
            delta_identity["sha256"],
            branch_root / f"accepted_{accept_index:02d}_incremental_delta.f32le",
            incremental_identity["sha256"],
        )
        previous_flat = candidate_flat
        candidates.append({
            "candidate_order": accept_index,
            "accept_index": accept_index,
            "pair_index": int(accepted["pair_index"]),
            "accepted_record": accepted_identity,
            "cumulative_delta": delta_identity,
            "incremental_delta": incremental_identity,
            "exact_candidate_group_flat": exact_group_identity,
            "delta_semantics": accepted["delta_semantics"],
            "candidate_group_flat_sha256": descriptor["candidate_group_flat_sha256"],
            "artifact_only_child_descriptor": descriptor_identity,
            "artifact_only_child_receipt": child_receipt_identity,
            "accepted_chain_sha256": accepted_identity["sha256"],
            "immediate_parent_artifact_sha256": descriptor["immediate_parent_artifact_sha256"],
            "exact_candidate_delta_relation": relation,
        })
    if candidates:
        final_delta = frozen_validation.get("final_cumulative_delta_identity") or {}
        if final_delta.get("sha256") != candidates[-1]["cumulative_delta"]["sha256"]:
            raise RuntimeError("frozen validation final delta is not final ordered candidate")
    elif frozen_validation.get("final_cumulative_delta_identity") is not None:
        raise RuntimeError("frozen validation reported final delta without candidates")
    return {
        **frozen_validation,
        "controller_verified_branch_header_sha256": header_identity["sha256"],
        "controller_verified_exact_job_seed_solver_budget_and_no_training": True,
    }, candidates


def _load_search_bank(
    config: Mapping[str, object], root: Path,
) -> tuple[np.ndarray, Mapping[str, object]]:
    bank = config["search"]["bank"]
    path = root / str(bank["relative_path"])
    payload, identity = read_bytes_same_fd(path, str(bank["sha256"]))
    if int(identity["size_bytes"]) != int(bank["size_bytes"]):
        raise RuntimeError("SEARCH bank size drift")
    shape = tuple(int(value) for value in bank["shape"])
    values = np.frombuffer(payload, dtype="<u4").reshape(shape).copy()
    return values, {
        "schema": "hfw_w560_mps_local_search_capability_v1",
        "role": "SEARCH",
        "bank": identity,
        "shape": list(shape),
        "new_unique_token_positions_after_initialization": 0,
    }


def _localize_solver_candidates(
    ledger: LocalLedger, job_identity: Mapping[str, object],
    candidates: list[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], str]:
    relative_root = Path("proposal_imports") / f"local-job-{job_identity['sha256']}"
    root = ledger.root / relative_root
    if root.exists():
        raise FileExistsError("local solver candidate root already exists")
    root.mkdir(parents=True)
    localized = []
    for candidate in candidates:
        order = int(candidate["candidate_order"])
        destination = root / f"candidate-{order:04d}"
        exact_payload, _ = read_bytes_same_fd(
            Path(candidate["exact_candidate_group_flat"]["path"]),
            candidate["exact_candidate_group_flat"]["sha256"],
        )
        cumulative_payload, _ = read_bytes_same_fd(
            Path(candidate["cumulative_delta"]["path"]), candidate["cumulative_delta"]["sha256"],
        )
        incremental_payload, _ = read_bytes_same_fd(
            Path(candidate["incremental_delta"]["path"]), candidate["incremental_delta"]["sha256"],
        )
        exact = write_content_addressed_bytes(destination, ".group_flat.f32le", exact_payload)
        cumulative = write_content_addressed_bytes(destination, ".cumulative_delta.f32le", cumulative_payload)
        incremental = write_content_addressed_bytes(destination, ".incremental_delta.f32le", incremental_payload)
        localized.append({
            "candidate_order": order,
            "accept_index": int(candidate["accept_index"]),
            "pair_index": int(candidate["pair_index"]),
            "exact_candidate_group_flat": exact,
            "cumulative_delta": cumulative,
            "incremental_delta": incremental,
            "candidate_group_flat_sha256": exact["sha256"],
            "delta_semantics": "cumulative_current_minus_immutable_job_base",
        })
    return localized, relative_root.as_posix()


def run_local_solver_job(
    config_path: Path, ledger_root: Path, job_path: Path, output_root: Path,
) -> Mapping[str, object]:
    config, config_identity = load_config(config_path)
    if config["authority"].get("local_solver_launch_authorized") is not True:
        raise RuntimeError("long local solver launch is not authorized in this frozen build")
    numerical_contract = adapter_contract(config)
    ledger = LocalLedger(ledger_root, config_path)
    job, job_identity = ledger.validate_job(job_path)
    current = ledger.current_head()
    if job["base_head_record_sha256"] != current["record_sha256"]:
        raise RuntimeError("stale local solver job rejected before SEARCH")
    root = runtime_root(config_path)
    output_root = Path(output_root).resolve()
    try:
        output_root.relative_to(root)
    except ValueError as error:
        raise RuntimeError("local solver output is outside runtime") from error
    if output_root.exists():
        raise FileExistsError(f"local solver output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    search_values, search_capability = _load_search_bank(config, root)
    base_head = ledger.head_by_record_sha(job["base_head_record_sha256"])
    if base_head["checkpoint"]["sha256"] != job["base_checkpoint_sha256"]:
        raise RuntimeError("local solver job base checkpoint binding drift")
    device = torch.device("mps")
    loaded, relocation = load_head_model(config_path, base_head, device, config=config)
    batches = [torch.from_numpy(batch.astype("int64", copy=True)) for batch in search_values]
    _, branch, bridge, _, _ = vendor_modules(config)
    objective_class = make_mps_bfloat16_objective_class(bridge)
    objective = objective_class(loaded, batches, device)
    group = str(job["group"])
    base_flat = objective.flat_group(group).detach().clone()
    if base_flat.numel() > int(config["solver_limits"]["maximum_group_elements"]):
        raise RuntimeError("local solver group exceeds frozen bound")
    protocol = _job_protocol(config, job["base_checkpoint_sha256"], group, int(job["seed"]))
    base_protocol, _ = read_json_same_fd(
        Path(branch.__file__).resolve().parent / "config/production_protocol.json",
        config["vendor"]["production_protocol_sha256"],
    )
    protocol_overrides = validate_job_protocol_overrides(
        base_protocol, protocol, job["base_checkpoint_sha256"], group, int(job["seed"]),
    )
    registry = JobLocalTraceRegistry(output_root / "job_local_solver_events")
    solver_root = output_root / "solver_trace"
    with mps_functional_normalization(branch, str(config["vendor"]["branch_search_sha256"])) as normalization_calls:
        with artifact_only_projection(branch, numerical_contract):
            result = branch.run_independent_branch(
                objective, protocol, group, solver_root, registry, materialize=True,
            )
    normalization_count = int(normalization_calls["count"])
    if group in {"left_factors", "right_factors"} and normalization_count <= 0:
        raise RuntimeError("MPS factor proposal completed without CPU float64 normalization")
    if group in {"base_affines", "codecs"} and normalization_count != 0:
        raise RuntimeError("non-factor proposal unexpectedly used functional normalization")
    branch_root = solver_root / group
    frozen_validation, candidates = validate_artifact_candidate_chain(
        branch, protocol, branch_root, group, base_flat,
    )
    if int(result.accepts) != len(candidates):
        raise RuntimeError("local solver result/candidate-chain count drift")
    result_group_sha = hashlib.sha256(branch._tensor_bytes(result.final_flat)).hexdigest()
    expected_result_sha = candidates[-1]["candidate_group_flat_sha256"] if candidates else hashlib.sha256(branch._tensor_bytes(base_flat)).hexdigest()
    if result_group_sha != expected_result_sha:
        raise RuntimeError("local solver final tensor differs from exact candidate chain")
    localized, artifact_root_relative = _localize_solver_candidates(
        ledger, job_identity, candidates,
    )
    solver_settings_sha = hashlib.sha256(canonical_json_bytes(protocol["search"])).hexdigest()
    proposal = {
        "schema": LOCAL_SCHEMA,
        "status": "PASS_COMPLETE_LOCAL_SOLVER_CHAIN_AWAITING_DUPLICATED_LOCAL_VERIFY",
        "source_proposal_sha256": None,
        "source_terminal_receipt_sha256": None,
        "local_solver_job_record_sha256": job_identity["sha256"],
        "artifact_root_relative": artifact_root_relative,
        "base_head_record_sha256": job["base_head_record_sha256"],
        "base_generation": int(job["base_generation"]),
        "base_checkpoint_sha256": job["base_checkpoint_sha256"],
        "group": group,
        "seed": int(job["seed"]),
        "base_group_flat_sha256": hashlib.sha256(branch._tensor_bytes(base_flat)).hexdigest(),
        "group_elements": int(base_flat.numel()),
        "solver_source_sha256": config["vendor"]["branch_search_sha256"],
        "solver_settings_sha256": solver_settings_sha,
        "search_bank_sha256": search_capability["bank"]["sha256"],
        "candidate_count": len(localized),
        "ordered_candidates": localized,
        "every_saved_candidate_imported": True,
        "candidate_subset_selection_performed": False,
        "local_verify_opened_during_import": False,
        "held_out_generalization_claimed": False,
        "cloud_access": False,
        "training_or_backprop": False,
        "adam_or_any_optimizer": False,
        "solver_receipt": dict(result.receipt),
        "solver_validation_identity_sha256": frozen_validation["validation_identity_sha256"],
        "job_protocol_overrides": protocol_overrides,
        "numerical_lane": {
            **numerical_contract,
            "normalization_calls": normalization_count,
            "objective_runtime": objective.numerical_lane_receipt(),
        },
        "relocated_parent_receipt": relocation,
        "config_sha256": config_identity["sha256"],
        "new_unique_token_positions_after_initialization": 0,
        "solver_may_write_local_head": False,
        "stale_policy": "reject_without_merge_or_rebase",
    }
    proposal_artifact = write_content_addressed_json(
        ledger.root / "proposal_imports", ".local-proposal.json", proposal,
    )
    terminal = write_exclusive_json(output_root / "LOCAL_SOLVER_COMPLETE.json", {
        "schema": "hfw_w560_mps_local_solver_terminal_receipt_v1",
        "status": "PASS_CANDIDATE_CHAIN_REQUIRES_DUPLICATED_LOCAL_VERIFY",
        "proposal_artifact": proposal_artifact,
        "local_checkpoint_lineage_changed": False,
        "local_verify_opened": False,
    })
    return {**proposal, "proposal_artifact": proposal_artifact, "terminal_artifact": terminal}
