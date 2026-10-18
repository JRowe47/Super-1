from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
import torch

from .common import (
    canonical_json_bytes, fsync_directory, hash_file_same_fd, read_bytes_same_fd,
    read_json_same_fd, write_content_addressed_bytes, write_content_addressed_json,
)
from .disk_safety import validated_disk_reservations
from .ledger import Ledger
from .proposal_worker import _job_protocol, validate_artifact_candidate_chain
from .relocation import load_config, vendor_modules
from .state import flatten_losses, load_head_model, replace_group, state_for_objective
from .token_commitment import load_bank


def _write_promoted_checkpoint(
    path: Path, *, state: Mapping[str, torch.Tensor], origin_sha256: str,
    parent_sha256: str, generation: int, proposal_sha256: str,
    selected_candidate: Mapping[str, object], model_state_digest: str,
) -> Mapping[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    payload = {
        "schema": "hfw_w560_continuous_promoted_checkpoint_v1",
        "origin_checkpoint_sha256": origin_sha256,
        "parent_checkpoint_sha256": parent_sha256,
        "generation": generation,
        "accepted_proposal_sha256": proposal_sha256,
        "selected_candidate_order": int(selected_candidate["candidate_order"]),
        "selected_candidate_delta_sha256": selected_candidate["cumulative_delta"]["sha256"],
        "solver_cumulative_delta_sha256": selected_candidate["cumulative_delta"]["sha256"],
        "replacement_group_flat_sha256": selected_candidate["exact_candidate_group_flat"]["sha256"],
        "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
        "solver_cumulative_delta_used_to_reconstruct_group": False,
        "model_state_digest": model_state_digest,
        "optimizer_present": False,
        "scheduler_present": False,
        "training_resume_authorized": False,
        "model": {name: value.detach().cpu() for name, value in state.items()},
    }
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)
    return hash_file_same_fd(path)


def _validate_proposal_path(
    ledger: Ledger, proposal_path: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    path = Path(proposal_path)
    if path.resolve().parent != (ledger.root / "proposals").resolve():
        raise RuntimeError("proposal is outside the append-only ledger proposal directory")
    proposal, identity = read_json_same_fd(path)
    if path.name != f"{identity['sha256']}.proposal.json":
        raise RuntimeError("proposal filename/content hash drift")
    if proposal.get("schema") != "hfw_w560_continuous_candidate_chain_proposal_v1":
        raise RuntimeError("proposal schema drift")
    return proposal, identity


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def load_exact_candidate_group(
    branch_module, candidate: Mapping[str, object], base_flat: torch.Tensor,
    previous_flat: torch.Tensor | None = None, expected_branch_root: Path | None = None,
) -> tuple[torch.Tensor, bytes, bytes, Mapping[str, object]]:
    if base_flat.dtype != torch.float32 or not torch.isfinite(base_flat).all():
        raise RuntimeError("exact candidate relation requires a finite float32 base group")
    expected_root = None if expected_branch_root is None else Path(expected_branch_root).resolve()

    def artifact_path(record: Mapping[str, object], label: str) -> Path:
        path = Path(record["path"])
        if expected_root is not None and path.resolve().parent != expected_root:
            raise RuntimeError(f"{label} is outside the frozen branch artifact root")
        return path

    exact_record = candidate["exact_candidate_group_flat"]
    exact_payload, exact_identity = read_bytes_same_fd(
        artifact_path(exact_record, "exact candidate group-flat"), exact_record["sha256"]
    )
    if int(exact_identity["size_bytes"]) != int(exact_record["size_bytes"]):
        raise RuntimeError("exact candidate group-flat size drift")
    exact_values = np.frombuffer(exact_payload, dtype="<f4")
    if exact_values.size != base_flat.numel():
        raise RuntimeError("exact candidate group-flat length drift")
    candidate_flat = torch.from_numpy(exact_values.copy()).to(
        device=base_flat.device, dtype=base_flat.dtype
    )
    if not torch.isfinite(candidate_flat).all():
        raise RuntimeError("exact candidate group-flat contains nonfinite values")
    if hashlib.sha256(branch_module._tensor_bytes(candidate_flat)).hexdigest() != exact_identity["sha256"]:
        raise RuntimeError("exact candidate group-flat tensor SHA drift")
    if exact_identity["sha256"] != candidate["candidate_group_flat_sha256"]:
        raise RuntimeError("candidate group SHA is not bound to exact group-flat bytes")

    delta_record = candidate["cumulative_delta"]
    delta_payload, delta_identity = read_bytes_same_fd(
        artifact_path(delta_record, "candidate cumulative delta"), delta_record["sha256"]
    )
    if int(delta_identity["size_bytes"]) != int(delta_record["size_bytes"]):
        raise RuntimeError("candidate cumulative delta size drift")
    delta_values = np.frombuffer(delta_payload, dtype="<f4")
    if delta_values.size != base_flat.numel():
        raise RuntimeError("candidate cumulative delta length drift")
    delta = torch.from_numpy(delta_values.copy()).to(
        device=base_flat.device, dtype=base_flat.dtype
    )
    if not torch.isfinite(delta).all():
        raise RuntimeError("candidate cumulative delta contains nonfinite values")
    exact_relation_payload = branch_module._tensor_bytes(candidate_flat - base_flat)
    if hashlib.sha256(exact_relation_payload).hexdigest() != delta_identity["sha256"]:
        raise RuntimeError("cumulative delta is not exact float32 candidate-minus-base")
    previous = base_flat if previous_flat is None else previous_flat
    incremental_record = candidate["incremental_delta"]
    incremental_payload, incremental_identity = read_bytes_same_fd(
        artifact_path(incremental_record, "candidate incremental delta"),
        incremental_record["sha256"],
    )
    if int(incremental_identity["size_bytes"]) != int(incremental_record["size_bytes"]):
        raise RuntimeError("candidate incremental delta size drift")
    incremental_values = np.frombuffer(incremental_payload, dtype="<f4")
    if incremental_values.size != base_flat.numel():
        raise RuntimeError("candidate incremental delta length drift")
    incremental = torch.from_numpy(incremental_values.copy()).to(
        device=base_flat.device, dtype=base_flat.dtype
    )
    if not torch.isfinite(incremental).all():
        raise RuntimeError("candidate incremental delta contains nonfinite values")
    if hashlib.sha256(branch_module._tensor_bytes(candidate_flat - previous)).hexdigest() != incremental_identity["sha256"]:
        raise RuntimeError("incremental delta is not exact float32 candidate-minus-previous")
    reconstructed_payload = branch_module._tensor_bytes(base_flat + delta)
    exact_bits = np.frombuffer(exact_payload, dtype="<u4")
    reconstructed_bits = np.frombuffer(reconstructed_payload, dtype="<u4")
    mismatch_count = int(np.count_nonzero(exact_bits != reconstructed_bits))
    maximum_abs_error = float(torch.max(torch.abs((base_flat + delta) - candidate_flat)).item())
    relation = {
        "schema": "hfw_w560_exact_candidate_delta_relation_v1",
        "exact_candidate_group_flat_sha256": exact_identity["sha256"],
        "cumulative_delta_sha256": delta_identity["sha256"],
        "declared_relation": "float32_exact_candidate_group_flat_minus_exact_job_base_group",
        "serialized_float32_subtraction_matches_cumulative_delta_bitwise": True,
        "incremental_delta_sha256": incremental_identity["sha256"],
        "serialized_float32_candidate_minus_previous_matches_incremental_delta_bitwise": True,
        "evaluation_uses_exact_candidate_group_flat_not_base_plus_delta": True,
        "base_plus_delta_bitwise_equals_exact_candidate": mismatch_count == 0,
        "base_plus_delta_mismatched_float32_elements": mismatch_count,
        "base_plus_delta_maximum_absolute_error": maximum_abs_error,
    }
    return candidate_flat, exact_payload, delta_payload, relation


def _candidate_evidence(
    objective, parent_first: list[float], parent_second: list[float],
    candidate_state: Mapping[str, torch.Tensor], candidate: Mapping[str, object],
    batch_count: int, tolerance: float, minimum_relative: float,
) -> Mapping[str, object]:
    first = flatten_losses(objective, candidate_state, batch_count)
    second = flatten_losses(objective, candidate_state, batch_count)
    if not all(math.isfinite(value) for value in (*first, *second)):
        raise RuntimeError("nonfinite candidate COMMIT evidence")
    duplicate_max = max(abs(left - right) for left, right in zip(first, second))
    parent_means = [_mean(parent_first), _mean(parent_second)]
    candidate_means = [_mean(first), _mean(second)]
    thresholds = [value - abs(value) * minimum_relative for value in parent_means]
    duplicates_pass = duplicate_max <= tolerance
    passes = (
        duplicates_pass
        and candidate_means[0] < thresholds[0]
        and candidate_means[1] < thresholds[1]
    )
    return {
        "candidate_order": int(candidate["candidate_order"]),
        "accept_index": int(candidate["accept_index"]),
        "pair_index": int(candidate["pair_index"]),
        "cumulative_delta_sha256": candidate["cumulative_delta"]["sha256"],
        "candidate_group_flat_sha256": candidate["candidate_group_flat_sha256"],
        "per_sequence_first": first,
        "per_sequence_second": second,
        "mean_first": candidate_means[0],
        "mean_second": candidate_means[1],
        "replicated_mean": math.fsum(candidate_means) / 2.0,
        "duplicate_max_abs_sequence_difference": duplicate_max,
        "duplicates_pass": duplicates_pass,
        "thresholds": thresholds,
        "passes_minimum_replicated_improvement": passes,
    }


def commit_proposal(
    config_path: Path, token_manifest_path: Path, ledger_root: Path,
    proposal_path: Path,
) -> Mapping[str, object]:
    ledger = Ledger(ledger_root)
    initial_runtime_inputs = ledger.validate_runtime_inputs(config_path, token_manifest_path)
    config = initial_runtime_inputs["config_value"]
    proposal, proposal_identity = _validate_proposal_path(ledger, proposal_path)
    with ledger.lock():
        runtime_inputs = ledger.validate_runtime_inputs(config_path, token_manifest_path)
        config = runtime_inputs["config_value"]
        job_path = ledger.root / "jobs" / (
            f"{int(proposal['job']['job_index']):012d}-{proposal['job_record_sha256']}.job.json"
        )
        job, job_identity = ledger.validate_job_path(
            job_path, expected_sha256=str(proposal["job_record_sha256"])
        )
        if canonical_json_bytes(job) != canonical_json_bytes(proposal["job"]):
            raise RuntimeError("proposal embedded job differs from exact ledger job")
        current = ledger.current_head()
        if proposal["base_head_record_sha256"] != current["record_sha256"]:
            decision = {
                "schema": "hfw_w560_continuous_commit_decision_v1",
                "status": "REJECTED_STALE_HEAD_NO_MERGE_NO_REBASE",
                "accepted": False,
                "proposal_sha256": proposal_identity["sha256"],
                "job_record_sha256": job_identity["sha256"],
                "proposal_base_head_record_sha256": proposal["base_head_record_sha256"],
                "observed_head_record_sha256": current["record_sha256"],
                "observed_generation": current["generation"],
                "commit_bank_opened": False,
                "child_checkpoint_materialized": False,
            }
            artifact = ledger.append_decision(decision)
            return {**decision, "decision_artifact": artifact}
        if proposal["base_checkpoint_sha256"] != current["checkpoint"]["sha256"]:
            raise RuntimeError("proposal base checkpoint differs from current head")
        if proposal["solver_source_sha256"] != config["vendor"]["branch_search_sha256"]:
            raise RuntimeError("proposal solver source drift")
        if proposal.get("worker_commit_authority") is not False:
            raise RuntimeError("proposal worker claimed commit authority")
        group = str(proposal["group"])
        if group != job["group"] or int(proposal["seed"]) != int(job["seed"]):
            raise RuntimeError("proposal group/seed differs from exact ledger job")
        if group not in config["controller"]["groups"]:
            raise RuntimeError("proposal group is outside controller contract")
        _, branch_module, bridge, _, _ = vendor_modules(config)
        if hash_file_same_fd(Path(branch_module.__file__))["sha256"] != config["vendor"]["branch_search_sha256"]:
            raise RuntimeError("frozen branch solver changed before commit")
        protocol = _job_protocol(
            config, job["base_checkpoint_sha256"], group, int(job["seed"])
        )
        disk_reservations = validated_disk_reservations(config)
        expected_solver_settings_sha = hashlib.sha256(
            canonical_json_bytes(protocol["search"])
        ).hexdigest()
        if proposal["solver_settings_sha256"] != expected_solver_settings_sha:
            raise RuntimeError("proposal frozen solver-settings SHA drift")
        trace_root = Path(proposal["frozen_solver_trace_branch_root"])
        try:
            trace_root.resolve().relative_to((ledger.root / "worker_attempts").resolve())
        except ValueError as error:
            raise RuntimeError("proposal solver trace is outside worker-attempt ledger") from error
        frozen_validation, observed_candidates = validate_artifact_candidate_chain(
            branch_module, protocol, trace_root, group
        )
        if canonical_json_bytes(observed_candidates) != canonical_json_bytes(proposal["ordered_candidates"]):
            raise RuntimeError("proposal ordered candidates differ from frozen solver trace")
        if int(proposal["candidate_count"]) != len(observed_candidates):
            raise RuntimeError("proposal candidate count drift")
        if frozen_validation["validation_identity_sha256"] != proposal["frozen_branch_validation_identity_sha256"]:
            raise RuntimeError("proposal frozen branch validation identity drift")
        validation_record = proposal["controller_branch_validation_artifact"]
        controller_validation, validation_identity = read_json_same_fd(
            Path(validation_record["path"]), validation_record["sha256"]
        )
        if validation_identity["size_bytes"] != validation_record["size_bytes"]:
            raise RuntimeError("controller branch validation size drift")
        if controller_validation.get("job_record_sha256") != job_identity["sha256"]:
            raise RuntimeError("controller branch validation job binding drift")
        if controller_validation.get("frozen_validator_validate_child_payload") is not False:
            raise RuntimeError("controller branch validation used the wrong child validator")
        if canonical_json_bytes(controller_validation["ordered_candidates"]) != canonical_json_bytes(observed_candidates):
            raise RuntimeError("controller validation candidate chain drift")
        if not observed_candidates:
            decision = {
                "schema": "hfw_w560_continuous_commit_decision_v1",
                "status": "REJECTED_NO_PROVISIONALLY_ACCEPTED_CANDIDATES",
                "accepted": False,
                "proposal_sha256": proposal_identity["sha256"],
                "job_record_sha256": job_identity["sha256"],
                "base_head_record_sha256": current["record_sha256"],
                "base_checkpoint_sha256": current["checkpoint"]["sha256"],
                "group": group,
                "commit_bank_opened": False,
                "child_checkpoint_materialized": False,
                "stale_rebase_or_merge_performed": False,
                "frozen_branch_validation_identity_sha256": frozen_validation["validation_identity_sha256"],
            }
            artifact = ledger.append_decision(decision)
            return {**decision, "decision_artifact": artifact}

        device = torch.device(config["controller"]["forward_device"])
        loaded, relocation = load_head_model(config_path, current, device, config=config)
        lineage_objective = bridge.W560ForwardOnlyObjective(loaded, (), device)
        lineage_base_flat = lineage_objective.flat_group(group).detach()
        if hashlib.sha256(branch_module._tensor_bytes(lineage_base_flat)).hexdigest() != proposal["base_group_flat_sha256"]:
            raise RuntimeError("proposal base group does not match current committed checkpoint")
        if int(proposal["group_elements"]) != lineage_base_flat.numel():
            raise RuntimeError("proposal group length drift")
        if lineage_base_flat.numel() > int(disk_reservations["maximum_group_elements"]):
            raise RuntimeError("proposal group exceeds formula-bound disk reservation")
        if len(observed_candidates) > int(disk_reservations["maximum_accepts_per_branch"]):
            raise RuntimeError("proposal candidate chain exceeds formula-bound accept ceiling")
        precommit_relations = {}
        previous_candidate_flat = lineage_base_flat
        for candidate in observed_candidates:
            candidate_flat, _, _, relation = load_exact_candidate_group(
                branch_module, candidate, lineage_base_flat, previous_candidate_flat,
                trace_root,
            )
            precommit_relations[int(candidate["candidate_order"])] = relation
            previous_candidate_flat = candidate_flat
        commit_values, token_manifest = load_bank(
            token_manifest_path, "COMMIT",
            expected_manifest_sha256=runtime_inputs["token_manifest"]["sha256"],
        )
        if token_manifest["banks"]["COMMIT"]["provenance"].get("reclassified_role") != (
            "PERMANENT_REUSABLE_COMMIT_OBJECTIVE_NOT_HELD_OUT"
        ):
            raise RuntimeError("committer received an unauthorized COMMIT objective")
        batches = [torch.from_numpy(batch.astype("int64", copy=True)) for batch in commit_values]
        objective = bridge.W560ForwardOnlyObjective(loaded, batches, device)
        base_flat = objective.flat_group(group).detach()
        if not torch.equal(base_flat, lineage_base_flat):
            raise RuntimeError("COMMIT objective base group differs from pre-COMMIT lineage base")
        parent_state = state_for_objective(objective)
        parent_first = flatten_losses(objective, parent_state, len(batches))
        parent_second = flatten_losses(objective, parent_state, len(batches))
        if not all(math.isfinite(value) for value in (*parent_first, *parent_second)):
            raise RuntimeError("nonfinite parent COMMIT evidence")
        tolerance = float(config["controller"]["duplicate_forward_tolerance"])
        parent_duplicate_max = max(
            abs(left - right) for left, right in zip(parent_first, parent_second)
        )
        if parent_duplicate_max > tolerance:
            raise RuntimeError("parent COMMIT duplicate evaluation drift")
        minimum_relative = float(config["controller"]["minimum_relative_commit_improvement"])
        candidate_rows = []
        selected_row = None
        selected_candidate = None
        selected_state = None
        selected_payload = None
        selected_exact_payload = None
        previous_candidate_flat = base_flat
        for candidate in observed_candidates:
            candidate_flat, exact_payload, delta_payload, relation = load_exact_candidate_group(
                branch_module, candidate, base_flat, previous_candidate_flat, trace_root
            )
            previous_candidate_flat = candidate_flat
            if canonical_json_bytes(relation) != canonical_json_bytes(
                precommit_relations[int(candidate["candidate_order"])]
            ):
                raise RuntimeError("candidate relation changed after COMMIT objective opened")
            candidate_state = replace_group(objective, group, candidate_flat)
            row = _candidate_evidence(
                objective, parent_first, parent_second, candidate_state, candidate,
                len(batches), tolerance, minimum_relative,
            )
            row = {**row, "exact_candidate_delta_relation": relation}
            candidate_rows.append(row)
            if row["passes_minimum_replicated_improvement"] and (
                selected_row is None
                or (row["replicated_mean"], row["candidate_order"])
                < (selected_row["replicated_mean"], selected_row["candidate_order"])
            ):
                selected_row = row
                selected_candidate = candidate
                selected_state = candidate_state
                selected_payload = delta_payload
                selected_exact_payload = exact_payload
        accepted = selected_candidate is not None
        generation = int(current["generation"]) + 1
        evidence = {
            "role": "PERMANENT_REUSABLE_COMMIT_OBJECTIVE_NOT_HELD_OUT",
            "all_provisionally_accepted_cumulative_candidates_evaluated": True,
            "candidate_count": len(observed_candidates),
            "candidate_results": candidate_rows,
            "selection_rule": "LOWEST_REPLICATED_MEAN_AMONG_PASSING_CANDIDATES_THEN_EARLIEST_ACCEPT_INDEX",
            "selected_candidate_order": None if selected_row is None else selected_row["candidate_order"],
            "parent_per_sequence_first": parent_first,
            "parent_per_sequence_second": parent_second,
            "parent_mean_first": _mean(parent_first),
            "parent_mean_second": _mean(parent_second),
            "parent_duplicate_max_abs_sequence_difference": parent_duplicate_max,
            "minimum_relative_improvement": minimum_relative,
            "duplicate_forward_tolerance": tolerance,
            "sequences": len(parent_first),
        }
        child_identity = None
        ledger_delta = None
        ledger_replacement = None
        model_digest = None
        if accepted:
            free_bytes = shutil.disk_usage(Path(ledger_root)).free
            required_free = (
                int(disk_reservations["minimum_unreserved_free_bytes"])
                + int(disk_reservations["reserved_bytes_for_possible_promotion"])
            )
            if free_bytes < required_free:
                failure = {
                    "schema": "hfw_w560_continuous_promotion_disk_gate_failure_v1",
                    "status": "BLOCK_INSUFFICIENT_DISK_FOR_VERIFIED_PROMOTION",
                    "proposal_sha256": proposal_identity["sha256"],
                    "job_record_sha256": job_identity["sha256"],
                    "selected_candidate_order": int(selected_candidate["candidate_order"]),
                    "free_bytes": free_bytes,
                    "required_free_bytes": required_free,
                    "proposal_remains_pending": True,
                    "no_delta_or_checkpoint_materialized": True,
                }
                artifact = write_content_addressed_json(
                    Path(ledger_root) / "failure_receipts", ".failure.json", failure
                )
                return {**failure, "failure_artifact": artifact}
            ledger_delta = write_content_addressed_bytes(
                Path(ledger_root) / "deltas", ".group_delta.f32le", selected_payload,
            )
            ledger_replacement = write_content_addressed_bytes(
                Path(ledger_root) / "replacement_group_flats", ".group_flat.f32le",
                selected_exact_payload,
            )
            if ledger_replacement["sha256"] != selected_candidate["exact_candidate_group_flat"]["sha256"]:
                raise RuntimeError("promoted exact replacement differs from selected candidate")
            model_digest = bridge.model_state_digest(selected_state)
            checkpoint_prefix = f"generation-{generation:08d}-{proposal_identity['sha256'][:16]}-attempt-"
            prior_attempts = list((Path(ledger_root) / "promoted_checkpoints").glob(
                f"{checkpoint_prefix}*.pt"
            ))
            checkpoint_path = Path(ledger_root) / "promoted_checkpoints" / (
                f"{checkpoint_prefix}{len(prior_attempts) + 1:04d}.pt"
            )
            child_identity = _write_promoted_checkpoint(
                checkpoint_path, state=selected_state,
                origin_sha256=config["origin"]["checkpoint_sha256"],
                parent_sha256=current["checkpoint"]["sha256"], generation=generation,
                proposal_sha256=proposal_identity["sha256"],
                selected_candidate=selected_candidate, model_state_digest=model_digest,
            )
        decision = {
            "schema": "hfw_w560_continuous_commit_decision_v1",
            "status": "ACCEPTED_VERIFIED_BEST_CANDIDATE_PROMOTION" if accepted else "REJECTED_NO_CANDIDATE_IMPROVED_COMMIT_OBJECTIVE",
            "accepted": accepted,
            "proposal_sha256": proposal_identity["sha256"],
            "job_record_sha256": job_identity["sha256"],
            "base_head_record_sha256": current["record_sha256"],
            "base_checkpoint_sha256": current["checkpoint"]["sha256"],
            "group": group,
            "commit_bank_opened": True,
            "commit_evidence": evidence,
            "selected_candidate": selected_candidate,
            "child_checkpoint": child_identity,
            "child_model_state_digest": model_digest,
            "ledger_delta": ledger_delta,
            "ledger_replacement_group_flat": ledger_replacement,
            "full_child_materialized_from_exact_replacement_group_flat": accepted,
            "solver_cumulative_delta_used_as_candidate_reconstruction": False,
            "stale_rebase_or_merge_performed": False,
            "relocated_parent_receipt": relocation,
            "runtime_input_contract": runtime_inputs,
            "frozen_branch_validation_identity_sha256": frozen_validation["validation_identity_sha256"],
            "final_bank_present": False,
        }
        decision_artifact = ledger.append_decision(decision)
        head_artifact = None
        if accepted:
            head = {
                "schema": "hfw_w560_continuous_head_v1",
                "generation": generation,
                "parent_head_record_sha256": current["record_sha256"],
                "parent_checkpoint_sha256": current["checkpoint"]["sha256"],
                "checkpoint": {**child_identity, "format": "hfw_w560_continuous_promoted_checkpoint_v1"},
                "origin_checkpoint_sha256": config["origin"]["checkpoint_sha256"],
                "model_state_digest": model_digest,
                "runtime_manifest_file_sha256": current["runtime_manifest_file_sha256"],
                "runtime_manifest_identity_sha256": current["runtime_manifest_identity_sha256"],
                "accepted_proposal_sha256": proposal_identity["sha256"],
                "selected_candidate_order": int(selected_candidate["candidate_order"]),
                "decision_sha256": decision_artifact["sha256"],
                "delta_sha256": ledger_delta["sha256"],
                "solver_cumulative_delta_sha256": ledger_delta["sha256"],
                "replacement_group_flat_sha256": ledger_replacement["sha256"],
                "solver_cumulative_delta": ledger_delta,
                "replacement_group_flat": ledger_replacement,
                "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
                "group": group,
                "commit_objective": evidence,
                "contract_sha256": current["contract_sha256"],
            }
            head_artifact = ledger.append_head(head)
        return {
            **decision, "decision_artifact": decision_artifact,
            "new_head_artifact": head_artifact, "current_after": ledger.current_head(),
        }
