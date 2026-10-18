from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .common import (
    canonical_json_bytes, fsync_directory, hash_file_same_fd, read_bytes_same_fd,
    read_json_same_fd, write_content_addressed_bytes,
)
from .local_candidate_import import LOCAL_SCHEMA
from .local_config import load_config, runtime_root
from .local_ledger import DECISION_SCHEMA, HEAD_SCHEMA, LocalLedger
from .local_verify_bank import load_local_verify
from .mps_adapter import adapter_contract, make_mps_bfloat16_objective_class
from .relocation import vendor_modules
from .state import flatten_losses, load_head_model, replace_group, state_for_objective


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _load_proposal(
    ledger: LocalLedger, path: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    proposal, identity = read_json_same_fd(path)
    if Path(path).resolve().parent != (ledger.root / "proposal_imports").resolve():
        raise RuntimeError("localized proposal is outside append-only proposal directory")
    if Path(path).name != f"{identity['sha256']}.local-proposal.json":
        raise RuntimeError("localized proposal filename/content SHA drift")
    if proposal.get("schema") != LOCAL_SCHEMA:
        raise RuntimeError("localized proposal schema drift")
    candidates = proposal.get("ordered_candidates", [])
    if int(proposal.get("candidate_count", -1)) != len(candidates):
        raise RuntimeError("localized proposal candidate-count drift")
    if [int(row.get("candidate_order", -1)) for row in candidates] != list(range(1, len(candidates) + 1)):
        raise RuntimeError("localized candidate order drift")
    if proposal.get("every_saved_candidate_imported") is not True or proposal.get("candidate_subset_selection_performed") is not False:
        raise RuntimeError("localized proposal is not the full saved candidate chain")
    source_sha = proposal.get("source_proposal_sha256")
    job_sha = proposal.get("local_solver_job_record_sha256")
    if (source_sha is None) == (job_sha is None):
        raise RuntimeError("localized proposal must have exactly one source-chain identity")
    expected_root = (
        Path("proposal_imports") / str(source_sha)
        if source_sha is not None
        else Path("proposal_imports") / f"local-job-{job_sha}"
    )
    declared_root = Path(str(proposal.get("artifact_root_relative", "")))
    if declared_root.is_absolute() or ".." in declared_root.parts or declared_root != expected_root:
        raise RuntimeError("localized proposal artifact-root contract drift")
    return proposal, identity


def _artifact_payload(record: Mapping[str, object], expected_parent: Path) -> bytes:
    path = Path(record["path"])
    try:
        path.resolve().relative_to(expected_parent.resolve())
    except ValueError as error:
        raise RuntimeError("localized candidate artifact escaped proposal import root") from error
    payload, identity = read_bytes_same_fd(path, str(record["sha256"]))
    if int(identity["size_bytes"]) != int(record["size_bytes"]):
        raise RuntimeError("localized candidate artifact size drift")
    return payload


def _load_exact_candidate(
    branch, candidate: Mapping[str, object], base_flat: torch.Tensor,
    previous_flat: torch.Tensor, artifact_root: Path,
) -> tuple[torch.Tensor, bytes, bytes, Mapping[str, object]]:
    exact_payload = _artifact_payload(candidate["exact_candidate_group_flat"], artifact_root)
    cumulative_payload = _artifact_payload(candidate["cumulative_delta"], artifact_root)
    incremental_payload = _artifact_payload(candidate["incremental_delta"], artifact_root)
    exact_array = np.frombuffer(exact_payload, dtype="<f4")
    cumulative_array = np.frombuffer(cumulative_payload, dtype="<f4")
    incremental_array = np.frombuffer(incremental_payload, dtype="<f4")
    if not (exact_array.size == cumulative_array.size == incremental_array.size == base_flat.numel()):
        raise RuntimeError("localized candidate sidecar element-count drift")
    candidate_flat = torch.from_numpy(exact_array.copy()).to(base_flat.device, base_flat.dtype)
    cumulative = torch.from_numpy(cumulative_array.copy()).to(base_flat.device, base_flat.dtype)
    incremental = torch.from_numpy(incremental_array.copy()).to(base_flat.device, base_flat.dtype)
    if not all(torch.isfinite(value).all() for value in (candidate_flat, cumulative, incremental)):
        raise RuntimeError("localized candidate sidecar contains nonfinite values")
    exact_sha = hashlib.sha256(branch._tensor_bytes(candidate_flat)).hexdigest()
    if exact_sha != candidate["exact_candidate_group_flat"]["sha256"] or exact_sha != candidate["candidate_group_flat_sha256"]:
        raise RuntimeError("localized exact candidate tensor SHA drift")
    cumulative_sha = hashlib.sha256(branch._tensor_bytes(candidate_flat - base_flat)).hexdigest()
    if cumulative_sha != candidate["cumulative_delta"]["sha256"]:
        raise RuntimeError("cumulative delta is not exact float32 candidate-minus-job-base")
    incremental_sha = hashlib.sha256(branch._tensor_bytes(candidate_flat - previous_flat)).hexdigest()
    if incremental_sha != candidate["incremental_delta"]["sha256"]:
        raise RuntimeError("incremental delta is not exact float32 candidate-minus-previous")
    return candidate_flat, exact_payload, cumulative_payload, {
        "schema": "hfw_w560_mps_local_exact_candidate_relation_v1",
        "exact_candidate_group_flat_sha256": exact_sha,
        "cumulative_delta_sha256": cumulative_sha,
        "incremental_delta_sha256": incremental_sha,
        "serialized_float32_candidate_minus_job_base_matches_cumulative_delta_bitwise": True,
        "serialized_float32_candidate_minus_previous_matches_incremental_delta_bitwise": True,
        "evaluation_and_checkpoint_materialization_use_exact_candidate_group_flat": True,
        "base_plus_delta_used_to_reconstruct_candidate": False,
    }


def _candidate_evidence(
    objective, candidate_state: Mapping[str, torch.Tensor], candidate: Mapping[str, object],
    parent_first: list[float], parent_second: list[float], batch_count: int,
    tolerance: float, minimum_relative: float,
) -> Mapping[str, object]:
    first = flatten_losses(objective, candidate_state, batch_count)
    second = flatten_losses(objective, candidate_state, batch_count)
    if not all(math.isfinite(value) for value in (*first, *second)):
        raise RuntimeError("nonfinite LOCAL_VERIFY candidate evidence")
    duplicate_max = max(abs(left - right) for left, right in zip(first, second))
    candidate_means = [_mean(first), _mean(second)]
    parent_means = [_mean(parent_first), _mean(parent_second)]
    thresholds = [value - abs(value) * minimum_relative for value in parent_means]
    passes = (
        duplicate_max <= tolerance
        and candidate_means[0] < thresholds[0]
        and candidate_means[1] < thresholds[1]
    )
    return {
        "candidate_order": int(candidate["candidate_order"]),
        "accept_index": int(candidate["accept_index"]),
        "pair_index": int(candidate["pair_index"]),
        "candidate_group_flat_sha256": candidate["candidate_group_flat_sha256"],
        "cumulative_delta_sha256": candidate["cumulative_delta"]["sha256"],
        "per_sequence_first": first,
        "per_sequence_second": second,
        "mean_first": candidate_means[0],
        "mean_second": candidate_means[1],
        "replicated_mean": math.fsum(candidate_means) / 2.0,
        "duplicate_max_abs_sequence_difference": duplicate_max,
        "duplicates_pass": duplicate_max <= tolerance,
        "thresholds": thresholds,
        "passes_minimum_replicated_improvement": passes,
    }


def _write_promoted_checkpoint(
    path: Path, state: Mapping[str, torch.Tensor], *, origin_sha256: str,
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


def evaluate_and_promote(
    config_path: Path, ledger_root: Path, proposal_path: Path,
) -> Mapping[str, object]:
    ledger = LocalLedger(ledger_root, config_path)
    proposal, proposal_identity = _load_proposal(ledger, proposal_path)
    with ledger.lock():
        current = ledger.current_head()
        if proposal["base_head_record_sha256"] != current["record_sha256"]:
            decision = {
                "schema": DECISION_SCHEMA,
                "status": "REJECTED_STALE_HEAD_NO_MERGE_NO_REBASE_NO_LOCAL_VERIFY_OPEN",
                "accepted": False,
                "proposal_sha256": proposal_identity["sha256"],
                "proposal_base_head_record_sha256": proposal["base_head_record_sha256"],
                "observed_head_record_sha256": current["record_sha256"],
                "observed_generation": current["generation"],
                "local_verify_opened": False,
                "child_checkpoint_materialized": False,
                "stale_rebase_or_merge_performed": False,
            }
            artifact = ledger.append_decision(decision)
            return {**decision, "decision_artifact": artifact, "current_after": current}
        if proposal["base_checkpoint_sha256"] != current["checkpoint"]["sha256"]:
            raise RuntimeError("localized proposal base checkpoint differs from current local head")
        config, config_identity = load_config(config_path)
        adapter_contract(config)
        verify_values, verify_capability = load_local_verify(config, runtime_root(config_path))
        device = torch.device("mps")
        loaded, relocation = load_head_model(config_path, current, device, config=config)
        _, branch, bridge, _, _ = vendor_modules(config)
        batches = [torch.from_numpy(batch.astype("int64", copy=True)) for batch in verify_values]
        objective_class = make_mps_bfloat16_objective_class(bridge)
        objective = objective_class(loaded, batches, device)
        group = str(proposal["group"])
        base_flat = objective.flat_group(group).detach().clone()
        if int(proposal["group_elements"]) != base_flat.numel():
            raise RuntimeError("localized proposal group length differs from current head")
        if hashlib.sha256(branch._tensor_bytes(base_flat)).hexdigest() != proposal["base_group_flat_sha256"]:
            raise RuntimeError("localized proposal base group differs from current head")
        parent_state = state_for_objective(objective)
        parent_first = flatten_losses(objective, parent_state, len(batches))
        parent_second = flatten_losses(objective, parent_state, len(batches))
        if not all(math.isfinite(value) for value in (*parent_first, *parent_second)):
            raise RuntimeError("nonfinite parent LOCAL_VERIFY evidence")
        tolerance = float(config["controller"]["duplicate_forward_tolerance"])
        parent_duplicate_max = max(abs(left - right) for left, right in zip(parent_first, parent_second))
        if parent_duplicate_max > tolerance:
            raise RuntimeError("parent duplicated LOCAL_VERIFY score drift")
        minimum_relative = float(config["controller"]["minimum_relative_local_verify_improvement"])
        artifact_root = ledger.root / str(proposal["artifact_root_relative"])
        try:
            artifact_root.resolve().relative_to((ledger.root / "proposal_imports").resolve())
        except ValueError as error:
            raise RuntimeError("localized proposal artifact root escaped ledger") from error
        candidate_rows = []
        selected_row = None
        selected_candidate = None
        selected_state = None
        selected_exact_payload = None
        selected_delta_payload = None
        previous_flat = base_flat
        for candidate in proposal["ordered_candidates"]:
            candidate_flat, exact_payload, delta_payload, relation = _load_exact_candidate(
                branch, candidate, base_flat, previous_flat, artifact_root,
            )
            previous_flat = candidate_flat
            candidate_state = replace_group(objective, group, candidate_flat)
            row = _candidate_evidence(
                objective, candidate_state, candidate, parent_first, parent_second,
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
                selected_exact_payload = exact_payload
                selected_delta_payload = delta_payload
        if len(candidate_rows) != int(proposal["candidate_count"]):
            raise RuntimeError("not every saved candidate was evaluated")
        accepted = selected_candidate is not None
        generation = int(current["generation"]) + 1
        evidence = {
            "role": "LOCAL_VERIFY",
            "disclosure": "PERMANENT_REUSED_OPTIMIZATION_DATA_NOT_HELD_OUT",
            "held_out_generalization_claimed": False,
            "bank_sha256": verify_capability["bank"]["sha256"],
            "candidate_count": len(candidate_rows),
            "all_saved_candidates_evaluated": True,
            "candidate_results": candidate_rows,
            "selection_rule": "LOWEST_REPLICATED_MEAN_AMONG_CANDIDATES_PASSING_STRICT_RELATIVE_THRESHOLD_THEN_EARLIEST_CANDIDATE_ORDER",
            "selected_candidate_order": None if selected_row is None else selected_row["candidate_order"],
            "parent_per_sequence_first": parent_first,
            "parent_per_sequence_second": parent_second,
            "parent_mean_first": _mean(parent_first),
            "parent_mean_second": _mean(parent_second),
            "parent_duplicate_max_abs_sequence_difference": parent_duplicate_max,
            "minimum_relative_improvement": minimum_relative,
            "duplicate_forward_tolerance": tolerance,
            "sequences": len(parent_first),
            "duplicated_parent_scoring": True,
            "duplicated_every_candidate_scoring": True,
            "objective_runtime": objective.numerical_lane_receipt(),
        }
        child_identity = None
        ledger_delta = None
        ledger_replacement = None
        model_digest = None
        if accepted:
            ledger_delta = write_content_addressed_bytes(
                ledger.root / "deltas", ".group_delta.f32le", selected_delta_payload,
            )
            ledger_replacement = write_content_addressed_bytes(
                ledger.root / "replacement_group_flats", ".group_flat.f32le", selected_exact_payload,
            )
            if ledger_replacement["sha256"] != selected_candidate["exact_candidate_group_flat"]["sha256"]:
                raise RuntimeError("local replacement bytes differ from selected exact candidate")
            model_digest = bridge.model_state_digest(selected_state)
            prefix = f"generation-{generation:08d}-{proposal_identity['sha256'][:16]}-attempt-"
            prior = list((ledger.root / "promoted_checkpoints").glob(f"{prefix}*.pt"))
            checkpoint_path = ledger.root / "promoted_checkpoints" / f"{prefix}{len(prior) + 1:04d}.pt"
            child_identity = _write_promoted_checkpoint(
                checkpoint_path, selected_state,
                origin_sha256=config["origin"]["checkpoint_sha256"],
                parent_sha256=current["checkpoint"]["sha256"], generation=generation,
                proposal_sha256=proposal_identity["sha256"],
                selected_candidate=selected_candidate, model_state_digest=model_digest,
            )
        decision = {
            "schema": DECISION_SCHEMA,
            "status": "ACCEPTED_BEST_DUPLICATED_LOCAL_VERIFY_CANDIDATE" if accepted else "REJECTED_NO_CANDIDATE_PASSED_DUPLICATED_LOCAL_VERIFY_THRESHOLD",
            "accepted": accepted,
            "proposal_sha256": proposal_identity["sha256"],
            "base_head_record_sha256": current["record_sha256"],
            "base_checkpoint_sha256": current["checkpoint"]["sha256"],
            "group": group,
            "local_verify_opened": True,
            "local_verify_evidence": evidence,
            "selected_candidate": selected_candidate,
            "child_checkpoint": child_identity,
            "child_model_state_digest": model_digest,
            "ledger_delta": ledger_delta,
            "ledger_replacement_group_flat": ledger_replacement,
            "full_child_materialized_from_exact_replacement_group_flat": accepted,
            "solver_cumulative_delta_used_to_reconstruct_group": False,
            "stale_rebase_or_merge_performed": False,
            "relocated_parent_receipt": relocation,
            "config_sha256": config_identity["sha256"],
            "cloud_access": False,
            "training_or_backprop": False,
            "adam_or_any_optimizer": False,
        }
        decision_artifact = ledger.append_decision(decision)
        head_artifact = None
        if accepted:
            head = {
                "schema": HEAD_SCHEMA,
                "generation": generation,
                "local_promotions": int(current["local_promotions"]) + 1,
                "parent_head_record_sha256": current["record_sha256"],
                "source_current_head_record_sha256": current["source_current_head_record_sha256"],
                "parent_checkpoint_sha256": current["checkpoint"]["sha256"],
                "checkpoint": {**child_identity, "format": "hfw_w560_continuous_promoted_checkpoint_v1"},
                "origin_checkpoint_sha256": current["origin_checkpoint_sha256"],
                "model_state_digest": model_digest,
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
                "local_verify_evidence": evidence,
                "contract_sha256": current["contract_sha256"],
                "runtime_manifest_file_sha256": current["runtime_manifest_file_sha256"],
                "runtime_manifest_identity_sha256": current["runtime_manifest_identity_sha256"],
            }
            head_artifact = ledger.append_head(head)
        return {
            **decision,
            "decision_artifact": decision_artifact,
            "new_head_artifact": head_artifact,
            "current_after": ledger.current_head(),
        }


def evaluate_promote_and_allocate_next(
    config_path: Path, ledger_root: Path, proposal_path: Path,
) -> Mapping[str, object]:
    result = evaluate_and_promote(config_path, ledger_root, proposal_path)
    ledger = LocalLedger(ledger_root, config_path)
    next_job = ledger.allocate_next_job()
    current = ledger.current_head()
    if next_job["base_head_record_sha256"] != current["record_sha256"]:
        raise RuntimeError("next local solver job did not bind current append-only head")
    if next_job["base_checkpoint_sha256"] != current["checkpoint"]["sha256"]:
        raise RuntimeError("next local solver job did not bind current checkpoint")
    return {
        **result,
        "next_solver_job": next_job,
        "next_job_reads_current_append_only_head_after_promotion": True,
        "next_job_base_head_record_sha256": current["record_sha256"],
        "next_job_base_checkpoint_sha256": current["checkpoint"]["sha256"],
    }
