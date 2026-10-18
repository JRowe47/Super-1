from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import torch

from .common import (
    canonical_json_bytes, hash_file_same_fd, read_json_same_fd,
    write_content_addressed_bytes, write_content_addressed_json,
    write_exclusive_bytes, write_exclusive_json,
)
from .disk_safety import validated_disk_reservations
from .ledger import Ledger
from .relocation import load_config, vendor_modules
from .state import load_head_model
from .token_commitment import load_bank


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
            "schema": "hfw_w560_job_local_provisional_solver_event_v1",
            "event_index": self.count,
            "proposal_sha256": proposal_sha256,
            "delta_sha256": delta_sha256,
            "artifact_only_child_sha256": child_checkpoint_sha256,
            "provisionally_accepted_inside_solver": bool(accepted),
            "globally_committed": False,
            "reason": reason,
            "record": dict(record),
        }
        payload = canonical_json_bytes(event)
        digest = hashlib.sha256(payload).hexdigest()
        write_exclusive_json(self.root / f"{self.count:04d}-{digest}.event.json", event)
        self.count += 1
        return digest


@contextmanager
def artifact_only_projection(branch_module):
    """Replace only provisional full-checkpoint projection; solver arithmetic is untouched."""
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
            "schema": "hfw_w560_provisional_delta_child_descriptor_v1",
            "artifact_projection_only": True,
            "solver_arithmetic_seed_radius_evaluation_order_changed": False,
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
            "global_commit_authorized": False,
        }
        return write_exclusive_json(path, descriptor)["sha256"]

    branch_module._materialize_child = _write_delta_child
    try:
        yield
    finally:
        branch_module._materialize_child = original


def _job_protocol(config: Mapping[str, object], base_checkpoint_sha256: str, group: str, seed: int):
    package_root, _, _, _, _ = vendor_modules(config)
    protocol, _ = read_json_same_fd(
        package_root / "config/production_protocol.json",
        config["vendor"]["production_protocol_sha256"],
    )
    protocol = deepcopy(protocol)
    protocol["parent"]["checkpoint_sha256"] = base_checkpoint_sha256
    protocol["search"]["branch_seeds"][group] = int(seed)
    return protocol


def validate_artifact_candidate_chain(
    branch_module, protocol: Mapping[str, object], branch_root: Path, group: str,
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
        if descriptor.get("schema") != "hfw_w560_provisional_delta_child_descriptor_v1":
            raise RuntimeError("artifact-only candidate descriptor schema drift")
        if descriptor.get("artifact_projection_only") is not True:
            raise RuntimeError("artifact-only candidate descriptor policy drift")
        if descriptor.get("solver_arithmetic_seed_radius_evaluation_order_changed") is not False:
            raise RuntimeError("artifact-only adapter claims solver arithmetic drift")
        if descriptor.get("full_provisional_checkpoint_materialized") is not False:
            raise RuntimeError("provisional candidate unexpectedly materialized a full checkpoint")
        if descriptor.get("global_commit_authorized") is not False:
            raise RuntimeError("provisional candidate claimed global commit authority")
        if descriptor.get("group") != group:
            raise RuntimeError("artifact-only candidate descriptor group drift")
        if descriptor.get("immutable_job_base_checkpoint_sha256") != protocol["parent"]["checkpoint_sha256"]:
            raise RuntimeError("artifact-only candidate base checkpoint drift")
        if descriptor.get("accepted_chain_sha256") != accepted_identity["sha256"]:
            raise RuntimeError("artifact-only candidate accepted-chain drift")
        descriptor_delta = descriptor.get("cumulative_group_delta", {})
        if descriptor_delta.get("sha256") != delta_identity["sha256"]:
            raise RuntimeError("artifact-only descriptor delta SHA drift")
        if int(descriptor_delta.get("size_bytes", -1)) != int(delta_identity["size_bytes"]):
            raise RuntimeError("artifact-only descriptor delta size drift")
        descriptor_group = descriptor.get("exact_candidate_group_flat", {})
        if descriptor_group.get("sha256") != exact_group_identity["sha256"]:
            raise RuntimeError("artifact-only descriptor exact candidate SHA drift")
        if int(descriptor_group.get("size_bytes", -1)) != int(exact_group_identity["size_bytes"]):
            raise RuntimeError("artifact-only descriptor exact candidate size drift")
        if descriptor.get("candidate_group_flat_sha256") != exact_group_identity["sha256"]:
            raise RuntimeError("artifact-only candidate group SHA is not exact-flat artifact SHA")
        if descriptor.get("cumulative_delta_relation") != (
            "float32_exact_candidate_group_flat_minus_exact_job_base_group"
        ):
            raise RuntimeError("artifact-only cumulative delta relation drift")
        if descriptor.get("base_plus_cumulative_delta_is_not_an_authoritative_reconstruction") is not True:
            raise RuntimeError("artifact-only reconstruction authority drift")
        if child_receipt.get("sha256") != descriptor_identity["sha256"]:
            raise RuntimeError("artifact-only child receipt descriptor binding drift")
        if child_receipt.get("accepted_chain_sha256") != accepted_identity["sha256"]:
            raise RuntimeError("artifact-only child receipt accepted-chain drift")
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
        })
    if candidates:
        final_delta = frozen_validation.get("final_cumulative_delta_identity") or {}
        if final_delta.get("sha256") != candidates[-1]["cumulative_delta"]["sha256"]:
            raise RuntimeError("frozen validation final delta is not final ordered candidate")
    elif frozen_validation.get("final_cumulative_delta_identity") is not None:
        raise RuntimeError("frozen validation reported a final delta with no accepted candidates")
    return {
        **frozen_validation,
        "controller_verified_branch_header_sha256": header_identity["sha256"],
        "controller_verified_exact_job_seed_solver_budget_and_no_training": True,
    }, candidates


def run_proposal_job(
    config_path: Path, token_manifest_path: Path, ledger_root: Path,
    job_path: Path, output_root: Path,
) -> Mapping[str, object]:
    ledger = Ledger(ledger_root)
    runtime_inputs = ledger.validate_runtime_inputs(config_path, token_manifest_path)
    config = runtime_inputs["config_value"]
    job, job_identity = ledger.validate_job_path(job_path)
    if Path(output_root).exists():
        raise FileExistsError(f"worker output root must be fresh: {output_root}")
    Path(output_root).mkdir(parents=True)
    base_head = ledger.head_by_record_sha(job["base_head_record_sha256"])
    if base_head["checkpoint"]["sha256"] != job["base_checkpoint_sha256"]:
        raise RuntimeError("job base checkpoint binding drift")
    device = torch.device(config["controller"]["forward_device"])
    loaded, relocation = load_head_model(config_path, base_head, device, config=config)
    search_values, token_manifest = load_bank(
        token_manifest_path, "SEARCH",
        expected_manifest_sha256=runtime_inputs["token_manifest"]["sha256"],
    )
    if token_manifest["final_bank_present"] is not False:
        raise RuntimeError("worker token capability contains FINAL")
    batches = [torch.from_numpy(batch.astype("int64", copy=True)) for batch in search_values]
    _, branch, bridge, _, _ = vendor_modules(config)
    objective = bridge.W560ForwardOnlyObjective(loaded, batches, device)
    group = str(job["group"])
    base_flat = objective.flat_group(group).detach().clone()
    disk_reservations = validated_disk_reservations(config)
    if base_flat.numel() > int(disk_reservations["maximum_group_elements"]):
        raise RuntimeError("worker group exceeds formula-bound disk reservation")
    protocol = _job_protocol(config, job["base_checkpoint_sha256"], group, int(job["seed"]))
    base_protocol, _ = read_json_same_fd(
        Path(branch.__file__).resolve().parent / "config/production_protocol.json",
        config["vendor"]["production_protocol_sha256"],
    )
    base_search = base_protocol["search"]
    changed_search_fields = [
        key for key in protocol["search"]
        if protocol["search"][key] != base_search[key] and key != "branch_seeds"
    ]
    if changed_search_fields:
        raise RuntimeError(f"solver settings changed: {changed_search_fields}")
    registry = JobLocalTraceRegistry(Path(output_root) / "job_local_solver_events")
    solver_root = Path(output_root) / "solver_trace"
    with artifact_only_projection(branch):
        result = branch.run_independent_branch(
            objective, protocol, group, solver_root, registry, materialize=True
        )
    if hash_file_same_fd(Path(branch.__file__))["sha256"] != config["vendor"]["branch_search_sha256"]:
        raise RuntimeError("frozen solver source changed during artifact-only adapter call")
    branch_root = solver_root / group
    frozen_validation, candidates = validate_artifact_candidate_chain(
        branch, protocol, branch_root, group
    )
    result_group_sha = hashlib.sha256(branch._tensor_bytes(result.final_flat)).hexdigest()
    if int(result.accepts) != len(candidates):
        raise RuntimeError("solver result/candidate chain accepted-count drift")
    if len(candidates) > int(disk_reservations["maximum_accepts_per_branch"]):
        raise RuntimeError("worker candidate chain exceeds formula-bound accept ceiling")
    expected_result_sha = (
        candidates[-1]["candidate_group_flat_sha256"] if candidates
        else hashlib.sha256(branch._tensor_bytes(base_flat)).hexdigest()
    )
    if result_group_sha != expected_result_sha:
        raise RuntimeError("solver final tensor differs from ordered candidate chain")
    validation_receipt = {
        "schema": "hfw_w560_controller_branch_artifact_validation_v1",
        "status": "PASS_FROZEN_CHAIN_VALIDATED_WITH_ARTIFACT_ONLY_CHILD_PAYLOAD_DISABLED",
        "job_record_sha256": job_identity["sha256"],
        "branch_root": str(branch_root.resolve()),
        "group": group,
        "frozen_validator_called": True,
        "frozen_validator_validate_child_payload": False,
        "frozen_validation": dict(frozen_validation),
        "ordered_candidates": candidates,
        "final_cumulative_delta": frozen_validation.get("final_cumulative_delta_identity"),
    }
    validation_artifact = write_exclusive_json(
        Path(output_root) / "CONTROLLER_BRANCH_VALIDATION.json", validation_receipt
    )
    solver_settings_sha = hashlib.sha256(
        canonical_json_bytes(protocol["search"])
    ).hexdigest()
    proposal = {
        "schema": "hfw_w560_continuous_candidate_chain_proposal_v1",
        "status": "PASS_UNTRUSTED_ORDERED_CANDIDATES_READY_FOR_SINGLE_COMMITTER",
        "job_record_sha256": job_identity["sha256"],
        "job": dict(job),
        "base_head_record_sha256": job["base_head_record_sha256"],
        "base_checkpoint_sha256": job["base_checkpoint_sha256"],
        "group": group,
        "seed": int(job["seed"]),
        "ordered_candidates": candidates,
        "candidate_count": len(candidates),
        "candidate_selection_authority": "SINGLE_SERIALIZED_COMMITTER_ONLY",
        "group_elements": int(base_flat.numel()),
        "group_shape": [int(base_flat.numel())],
        "base_group_flat_sha256": hashlib.sha256(branch._tensor_bytes(base_flat)).hexdigest(),
        "provisional_accepts": result.accepts,
        "solver_proposals": result.proposals,
        "solver_receipt": dict(result.receipt),
        "solver_source_sha256": config["vendor"]["branch_search_sha256"],
        "solver_settings_sha256": solver_settings_sha,
        "frozen_solver_trace_branch_root": str(branch_root.resolve()),
        "controller_branch_validation_artifact": validation_artifact,
        "frozen_branch_validation_identity_sha256": frozen_validation["validation_identity_sha256"],
        "final_cumulative_delta": frozen_validation.get("final_cumulative_delta_identity"),
        "seed_is_job_input_only": True,
        "artifact_adapter": {
            "provisional_full_checkpoints_suppressed": True,
            "exact_solver_deltas_and_traces_preserved": True,
            "solver_callable_executed_unchanged": True,
        },
        "relocated_parent_receipt": relocation,
        "runtime_input_contract": runtime_inputs,
        "token_manifest_sha256": runtime_inputs["token_manifest"]["sha256"],
        "token_role_opened": "SEARCH",
        "commit_bank_opened_by_worker": False,
        "final_bank_present": False,
        "worker_commit_authority": False,
    }
    proposal_artifact = write_content_addressed_json(
        Path(ledger_root) / "proposals", ".proposal.json", proposal
    )
    write_exclusive_json(Path(output_root) / "PROPOSAL_COMPLETE.json", {
        "schema": "hfw_w560_worker_terminal_receipt_v1",
        "status": "PASS",
        "proposal_artifact": proposal_artifact,
    })
    return {**proposal, "proposal_artifact": proposal_artifact}
