from __future__ import annotations

import copy
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hfw_dist.common import canonical_json_bytes, hash_file, sha256_bytes, write_exclusive
from hfw_dist.daytona_manager import (
    DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH,
    DaytonaManager,
    DeterministicNumericalWorkerFailure,
    EXPECTED_WHEEL_SHA256,
    LOCKED_RUNTIME_PACKAGES,
    MANAGER_LABEL,
    ManagerConfig,
    ManagerError,
    ReceiptJournal,
    WORKER_ENVIRONMENT,
    _atomic_stream_download,
    _build_parser,
    _contains_probe_labels,
    _contains_required_labels,
    _probe_labels,
    _probe_name,
    _immediate_predecessor_quota_failure_value,
    _immediate_predecessor_terminal_absence_value,
    _predecessor_terminal_absence_value,
    _probe_launch_identity,
    _recovered_task0_provenance_value,
    _sandbox_identity,
    _snapshot_identity,
    _snapshot_name,
    _task_labels,
    _task_name,
    _validate_upload_root_path,
    _validate_immediate_predecessor_quota_failure,
    _validate_predecessor_recovery_failure,
    _validate_worker_failure,
    _validate_worker_result,
    validate_recovered_task0_provenance,
    validate_upload_staging_receipt,
    validate_upload_snapshot_root,
)
from hfw_dist.seal_upload_staging import build_upload_staging_receipt


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ATTEMPT8_PACKAGE_ROOT = Path(
    "/Users/rohanarun/Documents/trin/"
    "hfw-w560-daytona-distributed-ce-optimizer-20260821-attempt8"
)
WAVE_ROOT = PACKAGE_ROOT / "runtime/waves/wave-00000000"
UPLOAD_SNAPSHOT_ROOT = Path(
    "/private/tmp/hfw_w560_attempt8_test_only_upload_snapshot"
)
BASE_IMAGE = (
    "python:3.12.11-slim-bookworm@sha256:"
    "519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
)
FINAL_WORKER_FAILURE_POLICY = {
    "machine_readable_filename": "CPU_SEARCH_FAILURE.json",
    "schema": "hfw_w560_daytona_cpu_search_failure_v1",
    "deterministic_numerical_status": "FAIL_DETERMINISTIC_NUMERICAL_FAILURE",
    "deterministic_numerical_classification": "DETERMINISTIC_NUMERICAL_FAILURE",
    "allowed_stages": [
        "BASE_CENTER_PREFLIGHT", "FORWARD_DTYPE",
        "SEARCH_CENTER", "TERMINAL_RANKING",
    ],
    "deterministic_numerical_exit_code": 42,
    "deterministic_numerical_retry_policy": "TERMINAL_DO_NOT_RETRY",
    "traceback_substring_classification_forbidden": True,
    "unclassified_or_infrastructure_failures_use_bounded_retry": True,
    "candidate_acceptance_gate_failure_is_nonterminal_rejection": True,
    "terminal_reason_codes_by_stage": {
        "BASE_CENTER_PREFLIGHT": [
            "SEALED_PREFLIGHT_OPERATION_EXCEPTION",
            "BASE_CENTER_NUMERICAL_OR_DTYPE_GATE_FAILED",
            "PREFLIGHT_RSS_LIMIT_REACHED",
        ],
        "FORWARD_DTYPE": ["OBSERVED_FP64_SHADOW_DTYPE_MISMATCH"],
        "SEARCH_CENTER": ["NONFINITE_SEARCH_OBJECTIVE"],
        "TERMINAL_RANKING": [
            "NONFINITE_TERMINAL_RANKING",
            "TERMINAL_CACHE_FLAT_IDENTITY_MISMATCH",
            "TERMINAL_DUPLICATE_BATCH_COUNT_MISMATCH",
            "TERMINAL_DUPLICATE_TOLERANCE_EXCEEDED",
            "WORKER_RSS_LIMIT_REACHED_AFTER_SEARCH",
        ],
    },
}


def config(root: Path) -> ManagerConfig:
    return ManagerConfig(
        package_root=root,
        wave_root=root / "wave",
        upload_snapshot_root=UPLOAD_SNAPSHOT_ROOT,
        base_image=BASE_IMAGE,
        api_url="https://app.daytona.io/api",
        target="us",
        worker_timeout_seconds=60,
        campaign_timeout_seconds=600,
        api_timeout_seconds=10,
        poll_seconds=1,
        maximum_attempts=2,
        create_concurrency=2,
        retry_initial_seconds=0.001,
        retry_maximum_seconds=0.002,
        vcpu_hour_rate_usd="0.0504",
        memory_gib_hour_rate_usd="0.0162",
        disk_gib_hour_rate_usd="0.000108",
    )


def attempt5_campaign(worker_count: int = 2):
    tasks = [
        {
            "worker_id": f"cpu-{index:03d}", "group": "left_factors",
            "seed": index + 1, "expected_group_elements": 4,
            "expected_final_flat_size_bytes": 16,
        }
        for index in range(worker_count)
    ]
    lane = {
        "memory_limit_gib": 4,
        "candidate_artifact_dtype": "float32",
        "candidate_codec_dtype": "float32",
        "forward_device": "cpu", "forward_autocast_enabled": False,
        "forward_autocast_dtype": None,
        "evaluation_parameter_dtype": "float64",
        "floating_buffer_dtype": "float64",
        "evaluation_floating_buffer_dtype": "float64",
        "persistent_state_dtype": "float64",
        "evaluation_persistent_state_dtype": "float64",
        "initial_compressed_state_dtype": "float64",
        "hidden_dtype": "float64", "logits_dtype": "float64",
        "ce_dtype": "float64",
        "distinct_fp64_shadow_state_surrogate": True,
        "canonical_checkpoint_and_model_source_bytes_untouched": True,
        "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
        "proposal_shadow_replaces_canonical_fp32_state_storage": True,
        "raw_ce_cross_lane_comparable": False,
        "mps_sole_promotion_authority": True,
        "base_full_bank_gate_schema": (
            "hfw_w560_cpu_fp64_all_search_base_preflight_v1"
        ),
        "full_bank_gate_schema": (
            "hfw_w560_cpu_fp64_all_search_acceptance_gate_v1"
        ),
        "objective_receipt_schema": (
            "hfw_w560_cpu_fp64_stable_microbatch_objective_receipt_v1"
        ),
        "fp64_full_bank_duplicate_absolute_tolerance": 1e-12,
        "logical_batch_shape": [2, 513],
        "physical_microbatch_shape": [1, 513],
        "normalization_adapter": {
            "schema": "maxabs_scaled_fp64_layer_norm_rescaled_epsilon_v1",
        },
        "sealed_environment": {
            "python_major_minor": "3.12", "platform_system": "Linux",
            "machine": "x86_64", "torch_version": "2.8.0+cpu",
            "torch_cuda_version": None, "numpy_version": "2.0.2",
        },
    }
    return {
        "wave_index": 0, "worker_count": worker_count,
        "snapshot_manifest_core_sha256": "1" * 64,
        "base_head_record_sha256": "2" * 64,
        "base_checkpoint_sha256": "3" * 64,
        "base_model_state_digest": "4" * 64,
        "search_bank_shape": [12, 2, 513],
        "tasks": tasks,
        "proposal_lane": lane,
        "worker_failure_policy": dict(FINAL_WORKER_FAILURE_POLICY),
        "solver": {
            "iterations": 18, "logical_search_evaluations": 180,
            "physical_search_model_forwards": 360,
            "base_center_preflight_logical_evaluations": 12,
            "base_center_preflight_physical_model_forwards": 24,
            "terminal_full_search_ranking_logical_evaluations": 12,
            "terminal_full_search_ranking_physical_model_forwards": 24,
            "acceptance_gate_logical_evaluations_per_attempt": 12,
            "acceptance_gate_physical_model_forwards_per_attempt": 24,
            "acceptance_gate_maximum_attempts": 18,
            "total_logical_evaluations_minimum": 204,
            "total_physical_model_forwards_minimum": 408,
            "total_logical_evaluations_maximum": 420,
            "total_physical_model_forwards_maximum": 840,
            "physical_model_forwards_per_logical_evaluation": 2,
        },
    }


def deterministic_failure_value(
    campaign, campaign_sha: str, manifest_sha: str, task_index: int = 0,
    stage: str = "BASE_CENTER_PREFLIGHT",
    reason_code: str = "BASE_CENTER_NUMERICAL_OR_DTYPE_GATE_FAILED",
):
    task = campaign["tasks"][task_index]
    lane = campaign["proposal_lane"]
    solver = campaign["solver"]
    is_base = stage == "BASE_CENTER_PREFLIGHT"
    partial_base_operation = (
        is_base and reason_code == "SEALED_PREFLIGHT_OPERATION_EXCEPTION"
    )
    preflight_logical = 1 if partial_base_operation else solver[
        "base_center_preflight_logical_evaluations"
    ]
    preflight_physical = 2 if partial_base_operation else solver[
        "base_center_preflight_physical_model_forwards"
    ]
    phase_logical = {"SEARCH": 0, "ACCEPTANCE_COMMIT_GATE": 0, "TERMINAL_RANKING": 0}
    phase_physical = {"SEARCH": 0, "ACCEPTANCE_COMMIT_GATE": 0, "TERMINAL_RANKING": 0}
    if stage == "FORWARD_DTYPE":
        phase_logical["SEARCH"], phase_physical["SEARCH"] = 1, 2
    elif stage == "SEARCH_CENTER":
        phase_logical["SEARCH"], phase_physical["SEARCH"] = 10, 20
    elif stage == "TERMINAL_RANKING":
        phase_logical["SEARCH"] = solver["logical_search_evaluations"]
        phase_physical["SEARCH"] = solver["physical_search_model_forwards"]
        phase_logical["TERMINAL_RANKING"] = 12
        phase_physical["TERMINAL_RANKING"] = 24
    solver_logical = sum(phase_logical.values())
    solver_physical = sum(phase_physical.values())
    diagnostics = {
        "reason_code": reason_code,
        "maximum_observed_rss_bytes": 1,
        "memory_limit_bytes": lane["memory_limit_gib"] * 1024 ** 3,
        "peak_rss_strictly_below_memory_limit": True,
        "preflight_logical_evaluations": preflight_logical,
        "preflight_physical_model_forwards": preflight_physical,
        "solver_logical_evaluations_completed": solver_logical,
        "solver_physical_model_forwards_completed": solver_physical,
        "acceptance_gate_attempts_completed": 0,
        "acceptance_gate_passes_completed": 0,
        "acceptance_gate_failures_completed": 0,
        "phase_logical_evaluations": phase_logical,
        "phase_physical_model_forwards": phase_physical,
        "float64_logits_observed": preflight_physical + solver_physical,
        "float64_hidden_observed": preflight_physical + solver_physical,
        "float64_persistent_state_observed": preflight_physical + solver_physical,
        "float64_ce_evaluations": preflight_physical + solver_physical,
    }
    if is_base:
        if reason_code == "SEALED_PREFLIGHT_OPERATION_EXCEPTION":
            diagnostics["preflight"] = {
                "schema": lane["base_full_bank_gate_schema"],
                "status": "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT",
                "failure_reasons": ["SEALED_PREFLIGHT_OPERATION_EXCEPTION"],
                "completed_batches": [], "failed_batch_index": 0,
                "failed_sequence_index": 1,
                "operation_exception_type": "RuntimeError",
                "operation_exception_message_sha256": "7" * 64,
            }
        elif reason_code == "PREFLIGHT_RSS_LIMIT_REACHED":
            diagnostics["maximum_observed_rss_bytes"] = (
                lane["memory_limit_gib"] * 1024 ** 3
            )
            diagnostics["peak_rss_strictly_below_memory_limit"] = False
            diagnostics["preflight"] = {
                "schema": lane["base_full_bank_gate_schema"],
                "status": "PASS_FINITE_CPU_FP64_ALL_SEARCH_BASE",
                "failure_reasons": [], "base_full_search_mean_ce": 6.0,
            }
        else:
            batch_count = int(campaign["search_bank_shape"][0])
            diagnostics["preflight"] = {
                "schema": lane["base_full_bank_gate_schema"],
                "status": "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT",
                "failure_reasons": ["NONFINITE_OR_DTYPE_BASE_SEQUENCE"],
                "batch_indices": list(range(batch_count)),
                "batches": [
                    {
                        "batch_index": index, "passed": index != 0,
                        "per_sequence": ([{
                            "sequence_index": 0,
                            "logits_dtype": "float32",
                            "expected_logits_dtype": "float64",
                            "logits_dtype_matches": False,
                            "hidden_dtype": "float64",
                            "expected_hidden_dtype": "float64",
                            "hidden_dtype_matches": True,
                            "persistent_state_floating_dtypes": ["float64"],
                            "persistent_state_dtype_matches": True,
                            "ce_dtype": "float64",
                            "expected_ce_dtype": "float64",
                            "loss_class": "FINITE",
                            "loss": 6.0,
                        }] if index == 0 else []),
                    }
                    for index in range(batch_count)
                ],
                "base_full_search_batch_losses": [None] + [6.0] * (batch_count - 1),
                "base_full_search_mean_ce": None,
            }
    elif stage == "FORWARD_DTYPE":
        diagnostics.update({
            "group": task["group"], "batch_index": 0, "sequence_index": 0,
            "sequence": {
                "logits_dtype": "float32", "expected_logits_dtype": "float64",
                "logits_dtype_matches": False,
                "hidden_dtype": "float64", "expected_hidden_dtype": "float64",
                "hidden_dtype_matches": True,
                "persistent_state_floating_dtypes": ["float64"],
                "persistent_state_dtype_matches": True, "ce_dtype": "float64",
                "expected_ce_dtype": "float64",
            },
        })
    elif stage == "SEARCH_CENTER":
        diagnostics.update({
            "reason": "VENDOR_REJECTED_NONFINITE_SEARCH_CENTER",
            "objective": {
                "group": task["group"], "batch_index": 0,
                "sequence_index": 0, "sequence": {
                    "loss_class": "NAN", "loss": None,
                    "logits": {"all_finite": False},
                },
            },
        })
    elif reason_code == "NONFINITE_TERMINAL_RANKING":
        diagnostics.update({
            "group": task["group"], "batch_index": 0, "sequence_index": 0,
            "sequence": {
                "loss_class": "NAN", "loss": None,
                "per_token_loss": {"all_finite": False},
            },
        })
    elif reason_code == "TERMINAL_CACHE_FLAT_IDENTITY_MISMATCH":
        diagnostics.update({"terminal_flat_sha256": "5" * 64, "cached_flat_sha256": "6" * 64})
    elif reason_code == "TERMINAL_DUPLICATE_BATCH_COUNT_MISMATCH":
        diagnostics.update({"terminal_batch_count": 12, "cached_batch_count": 11})
    elif reason_code == "TERMINAL_DUPLICATE_TOLERANCE_EXCEEDED":
        diagnostics.update({
            "maximum_absolute_difference": 2e-12, "absolute_tolerance": 1e-12,
            "flat_sha256": "5" * 64,
        })
    elif reason_code == "WORKER_RSS_LIMIT_REACHED_AFTER_SEARCH":
        observed_rss = lane["memory_limit_gib"] * 1024 ** 3
        diagnostics.update({
            "observed_peak_rss_bytes": observed_rss,
            "maximum_observed_rss_bytes": observed_rss,
            "peak_rss_strictly_below_memory_limit": False,
        })
    return {
        "schema": campaign["worker_failure_policy"]["schema"],
        "status": campaign["worker_failure_policy"]["deterministic_numerical_status"],
        "classification": campaign["worker_failure_policy"]["deterministic_numerical_classification"],
        "retry_policy": campaign["worker_failure_policy"]["deterministic_numerical_retry_policy"],
        "stage": stage,
        "campaign_sha256": campaign_sha,
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_manifest_core_sha256": campaign["snapshot_manifest_core_sha256"],
        "task_sha256": sha256_bytes(canonical_json_bytes(task)),
        "task_index": task_index,
        "worker_id": task["worker_id"],
        "wave_index": campaign["wave_index"],
        "group": task["group"],
        "seed": task["seed"],
        "base_head_record_sha256": campaign["base_head_record_sha256"],
        "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
        "base_model_state_digest": campaign["base_model_state_digest"],
        "numerical_lane": {
            key: lane[key] for key in (
                "candidate_artifact_dtype", "candidate_codec_dtype", "forward_device",
                "forward_autocast_enabled", "forward_autocast_dtype",
                "evaluation_parameter_dtype", "floating_buffer_dtype",
                "evaluation_floating_buffer_dtype", "persistent_state_dtype",
                "evaluation_persistent_state_dtype", "initial_compressed_state_dtype",
                "hidden_dtype", "logits_dtype", "ce_dtype",
                "distinct_fp64_shadow_state_surrogate",
                "canonical_checkpoint_and_model_source_bytes_untouched",
                "canonical_fp32_state_contract_executed_by_proposal_shadow",
                "proposal_shadow_replaces_canonical_fp32_state_storage",
                "raw_ce_cross_lane_comparable", "mps_sole_promotion_authority",
            )
        },
        "environment": {
            key: lane["sealed_environment"][key]
            for key in (
                "python_major_minor", "platform_system", "machine", "torch_version",
                "torch_cuda_version", "numpy_version",
            )
        },
        "diagnostics": diagnostics,
        "solver_artifacts_created": not is_base,
        "credential_environment_scan": {
            "forbidden_names_checked": ["DAYTONA_API_KEY"],
            "forbidden_name_matches": 0,
            "forbidden_value_pattern": "dtn_ followed by at least 32 hexadecimal characters",
            "forbidden_value_pattern_matches": 0,
        },
        "worker_checkpoint_or_head_writes": False,
        "mps_verify_opened": False,
        "cloud_credentials_available_to_worker": False,
        "training_or_backprop": False,
        "optimizer_steps": 0,
    }


def attempt5_success_value(
    campaign, campaign_sha: str, manifest_sha: str,
    flat_identity, bundle_identity, task_index: int = 0,
    with_passed_gate: bool = True,
):
    task = campaign["tasks"][task_index]
    solver = campaign["solver"]
    lane = campaign["proposal_lane"]
    batch_count = int(solver["base_center_preflight_logical_evaluations"])
    sequence_count = int(campaign["search_bank_shape"][1])
    base_mean = 6.0
    terminal_mean = 5.5 if with_passed_gate else base_mean
    finite_tensor = {
        "all_finite": True,
        "first_nonfinite_timestep": None,
        "maximum_finite_absolute_value": 1.0,
    }
    batches = []
    for batch_index in range(batch_count):
        rows = []
        for sequence_index in range(sequence_count):
            rows.append({
                "sequence_index": sequence_index,
                "logits_dtype": "float64", "expected_logits_dtype": "float64",
                "logits_dtype_matches": True,
                "hidden_dtype": "float64", "expected_hidden_dtype": "float64",
                "hidden_dtype_matches": True,
                "persistent_state_floating_dtypes": ["float64"],
                "persistent_state_dtype_matches": True,
                "ce_dtype": "float64", "expected_ce_dtype": "float64",
                "loss_class": "FINITE", "loss": base_mean,
                "logits": dict(finite_tensor), "hidden": dict(finite_tensor),
                "per_token_loss": dict(finite_tensor),
            })
        batches.append({
            "batch_index": batch_index, "per_sequence": rows,
            "base_center_mean_ce": base_mean, "passed": True,
        })
    base_sha = "9" * 64 if with_passed_gate else flat_identity["sha256"]
    base_losses = [base_mean] * batch_count
    terminal_losses = [terminal_mean] * batch_count
    gate_logical = (
        int(solver["acceptance_gate_logical_evaluations_per_attempt"])
        if with_passed_gate else 0
    )
    gate_physical = (
        int(solver["acceptance_gate_physical_model_forwards_per_attempt"])
        if with_passed_gate else 0
    )
    preflight_logical = int(solver["base_center_preflight_logical_evaluations"])
    preflight_physical = int(solver["base_center_preflight_physical_model_forwards"])
    search_logical = int(solver["logical_search_evaluations"])
    search_physical = int(solver["physical_search_model_forwards"])
    terminal_logical = int(solver["terminal_full_search_ranking_logical_evaluations"])
    terminal_physical = int(solver["terminal_full_search_ranking_physical_model_forwards"])
    solver_logical = search_logical + gate_logical + terminal_logical
    solver_physical = search_physical + gate_physical + terminal_physical
    total_logical = solver_logical + preflight_logical
    total_physical = solver_physical + preflight_physical
    preflight = {
        "schema": lane["base_full_bank_gate_schema"],
        "status": "PASS_FINITE_CPU_FP64_ALL_SEARCH_BASE",
        "failure_reasons": [], "batch_indices": list(range(batch_count)),
        "logical_batch_shape": lane["logical_batch_shape"],
        "physical_microbatch_shape": lane["physical_microbatch_shape"],
        "checkpoint_candidate_dtype": "float32",
        "evaluation_parameter_dtype": "float64", "forward_device": "cpu",
        "forward_autocast_dtype": None, "logits_dtype": "float64",
        "hidden_dtype": "float64", "persistent_state_dtype": "float64",
        "ce_dtype": "float64",
        "final_norm_algorithm": lane["normalization_adapter"]["schema"],
        "base_candidate_flat_sha256": base_sha, "batches": batches,
        "base_full_search_batch_losses": base_losses,
        "base_full_search_mean_ce": base_mean,
        "preflight_logical_evaluations": preflight_logical,
        "preflight_physical_model_forwards": preflight_physical,
        "maximum_observed_rss_bytes": 1,
    }
    gate_record = {
        "schema": lane["full_bank_gate_schema"], "gate_index": 0,
        "iteration": 0, "group": task["group"],
        "current_flat_sha256": base_sha,
        "candidate_flat_sha256": flat_identity["sha256"],
        "batch_indices": list(range(gate_logical)),
        "candidate_batch_losses": terminal_losses,
        "current_full_search_mean_ce": base_mean,
        "candidate_full_search_mean_ce": terminal_mean,
        "all_finite": True, "strict_full_search_mean_improvement": True,
        "passed": True, "logical_evaluations": gate_logical,
        "physical_model_forwards": gate_physical,
        "current_mutation_or_artifact_write_before_gate": False,
    }
    environment = {
        **lane["sealed_environment"],
        "platform_processor": "sealed-test-cpu",
        "torch_cpu_capability": "AVX2",
        "proc_cpuinfo_distinct_model_names": ["sealed-test-cpu"],
        "peak_rss_bytes": 2, "memory_limit_bytes": lane["memory_limit_gib"] * 1024 ** 3,
        "peak_rss_strictly_below_memory_limit": True,
    }
    return {
        "schema": "hfw_w560_daytona_cpu_search_result_v1",
        "status": "PASS_PROVISIONAL_CPU_SEARCH_ONLY",
        "campaign_sha256": campaign_sha,
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_manifest_core_sha256": campaign["snapshot_manifest_core_sha256"],
        "task_sha256": sha256_bytes(canonical_json_bytes(task)),
        "task_index": task_index, "worker_id": task["worker_id"],
        "wave_index": int(campaign["wave_index"]),
        "base_head_record_sha256": campaign["base_head_record_sha256"],
        "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
        "base_model_state_digest": campaign["base_model_state_digest"],
        "group": task["group"], "seed": int(task["seed"]),
        "group_elements": int(task["expected_group_elements"]),
        "changed": with_passed_gate,
        "provisional_search_accepts": int(with_passed_gate),
        "pairs": int(solver["iterations"]),
        "logical_search_evaluations": search_logical,
        "physical_search_model_forwards": search_physical,
        "terminal_ranking_logical_evaluations": terminal_logical,
        "terminal_ranking_physical_model_forwards": terminal_physical,
        "acceptance_gate_attempts": int(with_passed_gate),
        "acceptance_gate_passes": int(with_passed_gate),
        "acceptance_gate_failures": 0,
        "accepted_updates": int(with_passed_gate),
        "acceptance_gate_logical_evaluations": gate_logical,
        "acceptance_gate_physical_model_forwards": gate_physical,
        "solver_logical_evaluations_excluding_preflight": solver_logical,
        "solver_physical_model_forwards_excluding_preflight": solver_physical,
        "base_center_preflight_logical_evaluations": preflight_logical,
        "base_center_preflight_physical_model_forwards": preflight_physical,
        "total_logical_evaluations": total_logical,
        "total_physical_model_forwards": total_physical,
        "base_center_preflight": preflight,
        "base_full_search_batch_losses": base_losses,
        "base_full_search_mean_ce": base_mean,
        "terminal_full_search_batch_losses": terminal_losses,
        "terminal_full_search_mean_ce": terminal_mean,
        "terminal_full_search_duplicate": {
            "status": "PASS_TERMINAL_FULL_BANK_DUPLICATES_CACHED_COMMIT_GATE",
            "flat_sha256": flat_identity["sha256"], "batch_count": terminal_logical,
            "absolute_tolerance": lane["fp64_full_bank_duplicate_absolute_tolerance"],
            "maximum_absolute_difference": 0.0,
        },
        "self_calibrated_full_search_improvement": base_mean - terminal_mean,
        "self_calibrated_improvement_semantics": (
            "base_full_search_mean_ce_minus_terminal_full_search_mean_ce"
        ),
        "trajectory_final_group_flat": flat_identity,
        "accepted_candidate_bundle": bundle_identity,
        "accepted_candidates": ([{
            "candidate_order": 1,
            "accept_index": 1,
            "pair_index": 0,
            "group": task["group"],
            "offset_bytes": 0,
            "size_bytes": int(task["expected_final_flat_size_bytes"]),
            "group_flat_sha256": flat_identity["sha256"],
            "accepted_record_sha256": "a" * 64,
            "artifact_only_child_sha256": "b" * 64,
            "cumulative_delta_sha256": "c" * 64,
            "candidate_cpu_full_search_mean_ce": terminal_mean,
        }] if with_passed_gate else []),
        "accepted_candidate_count": int(with_passed_gate),
        "every_provisionally_accepted_candidate_in_bundle": True,
        "candidate_subset_selection_performed": False,
        "cpu_ranking_used_for_candidate_selection": False,
        "objective_runtime": {
            "schema": lane["objective_receipt_schema"], "device": "cpu",
            "checkpoint_candidate_dtype": "float32",
            "evaluation_parameter_dtype": "float64",
            "floating_buffer_dtype": "float64", "persistent_state_dtype": "float64",
            "forward_autocast_dtype": None, "logits_dtype": "float64",
            "hidden_dtype": "float64", "ce_dtype": "float64",
            "final_norm_algorithm": lane["normalization_adapter"]["schema"],
            "canonical_fp32_state_storage_contract_replaced_in_proposal_shadow": True,
            "canonical_model_source_or_checkpoint_mutated": False,
            "candidate_f32_widened_to_fp64_for_every_score": True,
            "logical_batch_shape": lane["logical_batch_shape"],
            "physical_microbatch_shape": lane["physical_microbatch_shape"],
            "logical_evaluations_excluding_preflight": solver_logical,
            "physical_model_forwards_excluding_preflight": solver_physical,
            "phase_logical_evaluations": {
                "SEARCH": search_logical, "ACCEPTANCE_COMMIT_GATE": gate_logical,
                "TERMINAL_RANKING": terminal_logical,
            },
            "phase_physical_model_forwards": {
                "SEARCH": search_physical, "ACCEPTANCE_COMMIT_GATE": gate_physical,
                "TERMINAL_RANKING": terminal_physical,
            },
            "preflight_logical_evaluations": preflight_logical,
            "preflight_physical_model_forwards": preflight_physical,
            "total_logical_evaluations_including_preflight": total_logical,
            "total_physical_model_forwards_including_preflight": total_physical,
            "acceptance_gate_attempts": int(with_passed_gate),
            "acceptance_gate_passes": int(with_passed_gate),
            "acceptance_gate_failures": 0,
            "acceptance_gate_records": [gate_record] if with_passed_gate else [],
            "float64_logits_observed": total_physical,
            "float64_hidden_observed": total_physical,
            "float64_persistent_state_observed": total_physical,
            "float64_ce_evaluations": total_physical,
            "physical_model_forwards_per_logical_evaluation": int(
                solver["physical_model_forwards_per_logical_evaluation"]
            ),
            "maximum_observed_rss_bytes": 1,
            "inference_only": True, "optimizer_steps": 0, "autograd_calls": 0,
        },
        "environment": environment,
        "timing": {"search_seconds": 1.0, "ranking_seconds": 1.0, "total_seconds": 3.0},
        "worker_checkpoint_or_head_writes": 0,
        "distinct_fp64_shadow_state_surrogate": True,
        "canonical_checkpoint_and_model_source_bytes_untouched": True,
        "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
        "proposal_shadow_replaces_canonical_fp32_state_storage": True,
        "raw_ce_cross_lane_comparable": False,
        "mps_sole_promotion_authority": True,
        "mps_verify_opened": False, "local_verify_available_to_worker": False,
        "cloud_credentials_available_to_worker": False,
        "credential_environment_scan": {
            "forbidden_name_matches": 0, "forbidden_value_pattern_matches": 0,
        },
        "training_or_backprop": False, "optimizer_steps": 0,
        "provisional_not_held_out_or_promoted": True,
    }


class JournalTests(unittest.TestCase):
    def test_receipts_are_create_only_and_chained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign_sha = "a" * 64
            journal = ReceiptJournal(root, campaign_sha)
            journal.append("ONE", value=1)
            journal.append("TWO", value=2)
            recovered = ReceiptJournal(root, campaign_sha)
            self.assertEqual([row["event"] for row in recovered.records], ["ONE", "TWO"])
            first = sorted(root.iterdir())[0]
            first.chmod(0o644)
            first.write_bytes(first.read_bytes() + b" ")
            with self.assertRaises(RuntimeError):
                ReceiptJournal(root, campaign_sha)


class DownloadTests(unittest.TestCase):
    def test_atomic_stream_hash_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "artifact.bin"
            payload = b"fixed-output" * 100
            identity = _atomic_stream_download(
                (payload[:31], payload[31:]), destination,
                hashlib.sha256(payload).hexdigest(), len(payload),
            )
            self.assertEqual(identity["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertFalse(destination.with_name(".artifact.bin.part").exists())


class ResultValidationTests(unittest.TestCase):
    def test_exact_campaign_environment_and_mean_are_enforced(self) -> None:
        campaign_path = WAVE_ROOT / "snapshot/CAMPAIGN.json"
        manifest_path = WAVE_ROOT / "snapshot/SNAPSHOT_MANIFEST.json"
        campaign = json.loads(campaign_path.read_text())
        campaign_sha = hash_file(campaign_path)["sha256"]
        manifest_sha = hash_file(manifest_path)["sha256"]
        task = campaign["tasks"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flat = root / "trajectory_final_group_flat.f32le"
            flat_identity = write_exclusive(
                flat, b"\x00" * int(task["expected_final_flat_size_bytes"]),
            )
            bundle = root / "accepted_candidates.f32le.bundle"
            os.link(flat, bundle)
            bundle_identity = hash_file(bundle)
            result = attempt5_success_value(
                campaign, campaign_sha, manifest_sha, flat_identity,
                bundle_identity,
            )
            result_path = root / "CPU_SEARCH_RESULT.json"
            result_path.write_bytes(canonical_json_bytes(result))
            validated = _validate_worker_result(
                campaign, campaign_sha, manifest_sha, 0, result_path, flat, bundle,
                campaign["proposal_lane"]["sealed_environment"],
            )
            self.assertEqual(
                validated["self_calibrated_full_search_improvement"], 0.5,
            )
            wrong_write_type = dict(result)
            wrong_write_type["worker_checkpoint_or_head_writes"] = False
            result_path.write_bytes(canonical_json_bytes(wrong_write_type))
            with self.assertRaises(ManagerError):
                _validate_worker_result(
                    campaign, campaign_sha, manifest_sha, 0,
                    result_path, flat, bundle, None,
                )
            wrong_duplicate = json.loads(json.dumps(result))
            wrong_duplicate["terminal_full_search_batch_losses"][0] += 0.25
            wrong_duplicate["terminal_full_search_batch_losses"][1] -= 0.25
            result_path.write_bytes(canonical_json_bytes(wrong_duplicate))
            with self.assertRaises(ManagerError):
                _validate_worker_result(
                    campaign, campaign_sha, manifest_sha, 0,
                    result_path, flat, bundle, None,
                )
        return
        campaign_path = WAVE_ROOT / "snapshot/CAMPAIGN.json"
        manifest_path = WAVE_ROOT / "snapshot/SNAPSHOT_MANIFEST.json"
        campaign = json.loads(campaign_path.read_text())
        campaign["worker_failure_policy"] = dict(FINAL_WORKER_FAILURE_POLICY)
        campaign_sha = hash_file(campaign_path)["sha256"]
        manifest_sha = hash_file(manifest_path)["sha256"]
        task = campaign["tasks"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flat = root / "trajectory_final_group_flat.f32le"
            payload = b"\x00" * int(task["expected_final_flat_size_bytes"])
            flat_identity = write_exclusive(flat, payload)
            losses = [6.0] * int(campaign["solver"]["terminal_full_search_ranking_logical_evaluations"])
            preflight_rows = []
            for sequence_index in range(campaign["proposal_lane"]["logical_batch_shape"][0]):
                finite_tensor = {
                    "all_finite": True,
                    "first_nonfinite_timestep": None,
                    "maximum_finite_absolute_value": 1.0,
                }
                preflight_rows.append({
                    "sequence_index": sequence_index,
                    "logits_dtype": "bfloat16",
                    "expected_logits_dtype": "bfloat16",
                    "logits_dtype_matches": True,
                    "hidden_dtype": "bfloat16",
                    "ce_dtype": "float32",
                    "expected_ce_dtype": "float32",
                    "loss_class": "FINITE",
                    "loss": 6.0,
                    "logits": dict(finite_tensor),
                    "hidden": dict(finite_tensor),
                    "per_token_loss": dict(finite_tensor),
                })
            preflight = {
                "schema": "hfw_w560_cpu_bfloat16_base_center_preflight_v1",
                "status": "PASS_FINITE_CPU_BFLOAT16_BASE_CENTER",
                "failure_reasons": [],
                "batch_index": 0,
                "logical_batch_shape": campaign["proposal_lane"]["logical_batch_shape"],
                "physical_microbatch_shape": campaign["proposal_lane"]["physical_microbatch_shape"],
                "parameter_dtype": "float32",
                "forward_device": "cpu",
                "forward_autocast_dtype": "bfloat16",
                "ce_dtype": "float32",
                "per_sequence": preflight_rows,
                "base_center_mean_ce": 6.0,
                "preflight_logical_evaluations": 1,
                "preflight_physical_model_forwards": 2,
                "total_logical_evaluations_including_preflight": 1,
                "total_physical_model_forwards_including_preflight": 2,
                "maximum_observed_rss_bytes": 1,
            }
            solver_logical = (
                campaign["solver"]["logical_search_evaluations"]
                + campaign["solver"]["terminal_full_search_ranking_logical_evaluations"]
            )
            solver_physical = (
                campaign["solver"]["physical_search_model_forwards"]
                + campaign["solver"]["terminal_full_search_ranking_physical_model_forwards"]
            )
            result = {
                "schema": "hfw_w560_daytona_cpu_search_result_v1",
                "status": "PASS_PROVISIONAL_CPU_SEARCH_ONLY",
                "campaign_sha256": campaign_sha,
                "snapshot_manifest_sha256": manifest_sha,
                "snapshot_manifest_core_sha256": campaign["snapshot_manifest_core_sha256"],
                "task_sha256": sha256_bytes(canonical_json_bytes(task)),
                "task_index": 0,
                "worker_id": task["worker_id"],
                "wave_index": campaign["wave_index"],
                "base_head_record_sha256": campaign["base_head_record_sha256"],
                "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
                "base_model_state_digest": campaign["base_model_state_digest"],
                "group": task["group"],
                "seed": task["seed"],
                "group_elements": task["expected_group_elements"],
                "pairs": campaign["solver"]["iterations"],
                "logical_search_evaluations": campaign["solver"]["logical_search_evaluations"],
                "physical_search_model_forwards": campaign["solver"]["physical_search_model_forwards"],
                "terminal_ranking_logical_evaluations": campaign["solver"]["terminal_full_search_ranking_logical_evaluations"],
                "terminal_ranking_physical_model_forwards": campaign["solver"]["terminal_full_search_ranking_physical_model_forwards"],
                "solver_logical_evaluations_excluding_preflight": solver_logical,
                "solver_physical_model_forwards_excluding_preflight": solver_physical,
                "base_center_preflight_logical_evaluations": 1,
                "base_center_preflight_physical_model_forwards": 2,
                "total_logical_evaluations": campaign["solver"]["total_logical_evaluations"],
                "total_physical_model_forwards": campaign["solver"]["total_physical_model_forwards"],
                "base_center_preflight": preflight,
                "terminal_full_search_batch_losses": losses,
                "terminal_full_search_mean_ce": 6.0,
                "trajectory_final_group_flat": flat_identity,
                "objective_runtime": {
                    "schema": "hfw_w560_cpu_physical_microbatch_objective_receipt_v1",
                    "device": "cpu",
                    "parameter_dtype": "float32",
                    "forward_autocast_dtype": "bfloat16",
                    "logits_dtype": "bfloat16",
                    "ce_dtype": "float32",
                    "logical_batch_shape": campaign["proposal_lane"]["logical_batch_shape"],
                    "physical_microbatch_shape": campaign["proposal_lane"]["physical_microbatch_shape"],
                    "logical_evaluations": solver_logical,
                    "physical_model_forwards": solver_physical,
                    "preflight_logical_evaluations": 1,
                    "preflight_physical_model_forwards": 2,
                    "cpu_bfloat16_autocast_entries": campaign["solver"]["total_physical_model_forwards"],
                    "bfloat16_logits_observed": campaign["solver"]["total_physical_model_forwards"],
                    "float32_ce_evaluations": campaign["solver"]["total_physical_model_forwards"],
                    "physical_model_forwards_per_logical_evaluation": 2,
                    "maximum_observed_rss_bytes": 1,
                },
                "environment": {
                    **campaign["proposal_lane"]["sealed_environment"],
                    "peak_rss_bytes": 1,
                    "peak_rss_strictly_below_memory_limit": True,
                },
                "worker_checkpoint_or_head_writes": 0,
                "mps_verify_opened": False,
                "local_verify_available_to_worker": False,
                "cloud_credentials_available_to_worker": False,
                "credential_environment_scan": {
                    "forbidden_name_matches": 0,
                    "forbidden_value_pattern_matches": 0,
                },
                "training_or_backprop": False,
                "optimizer_steps": 0,
                "provisional_not_held_out_or_promoted": True,
            }
            result_path = root / "CPU_SEARCH_RESULT.json"
            result_path.write_bytes(canonical_json_bytes(result))
            value = _validate_worker_result(
                campaign, campaign_sha, manifest_sha, 0, result_path, flat,
                campaign["proposal_lane"]["sealed_environment"],
            )
            self.assertEqual(value["peak_rss_bytes"], 1)
            wrong_write_type = dict(result)
            wrong_write_type["worker_checkpoint_or_head_writes"] = False
            result_path.write_bytes(canonical_json_bytes(wrong_write_type))
            with self.assertRaises(ManagerError):
                _validate_worker_result(
                    campaign, campaign_sha, manifest_sha, 0, result_path, flat, None,
                )
            bad = dict(result)
            bad["campaign_sha256"] = "b" * 64
            result_path.write_bytes(canonical_json_bytes(bad))
            with self.assertRaises(ManagerError):
                _validate_worker_result(
                    campaign, campaign_sha, manifest_sha, 0, result_path, flat, None,
                )

    def test_deterministic_failure_requires_exact_machine_readable_identity(self) -> None:
        campaign = attempt5_campaign()
        campaign_sha = "a" * 64
        manifest_sha = "b" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "CPU_SEARCH_FAILURE.json"
            value = deterministic_failure_value(campaign, campaign_sha, manifest_sha)
            path.write_bytes(canonical_json_bytes(value))
            observed, identity = _validate_worker_failure(
                campaign, campaign_sha, manifest_sha, 0, path,
            )
            self.assertEqual(observed["stage"], "BASE_CENTER_PREFLIGHT")
            self.assertEqual(len(identity["sha256"]), 64)
            wrong_write_type = dict(value)
            wrong_write_type["worker_checkpoint_or_head_writes"] = 0
            path.write_bytes(canonical_json_bytes(wrong_write_type))
            with self.assertRaises(ManagerError):
                _validate_worker_failure(
                    campaign, campaign_sha, manifest_sha, 0, path,
                )

    def test_zero_pass_terminal_duplicate_is_recomputed_against_base_vector(self) -> None:
        campaign_path = WAVE_ROOT / "snapshot/CAMPAIGN.json"
        manifest_path = WAVE_ROOT / "snapshot/SNAPSHOT_MANIFEST.json"
        campaign = json.loads(campaign_path.read_text())
        campaign_sha = hash_file(campaign_path)["sha256"]
        manifest_sha = hash_file(manifest_path)["sha256"]
        task = campaign["tasks"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flat = root / "trajectory_final_group_flat.f32le"
            flat_identity = write_exclusive(
                flat, b"\x00" * int(task["expected_final_flat_size_bytes"]),
            )
            bundle = root / "accepted_candidates.f32le.bundle"
            bundle_identity = write_exclusive(bundle, b"")
            result = attempt5_success_value(
                campaign, campaign_sha, manifest_sha, flat_identity,
                bundle_identity,
                with_passed_gate=False,
            )
            result_path = root / "CPU_SEARCH_RESULT.json"
            result_path.write_bytes(canonical_json_bytes(result))
            _validate_worker_result(
                campaign, campaign_sha, manifest_sha, 0, result_path, flat,
                bundle, None,
            )
            tampered = json.loads(json.dumps(result))
            tampered["terminal_full_search_batch_losses"][0] += 0.25
            tampered["terminal_full_search_batch_losses"][1] -= 0.25
            result_path.write_bytes(canonical_json_bytes(tampered))
            with self.assertRaises(ManagerError):
                _validate_worker_result(
                    campaign, campaign_sha, manifest_sha, 0,
                    result_path, flat, bundle, None,
                )

    def test_every_sealed_attempt5_exit42_reason_validates_exactly(self) -> None:
        campaign = attempt5_campaign()
        campaign_sha = "a" * 64
        manifest_sha = "b" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "CPU_SEARCH_FAILURE.json"
            for stage, reason_codes in campaign["worker_failure_policy"][
                "terminal_reason_codes_by_stage"
            ].items():
                for reason_code in reason_codes:
                    value = deterministic_failure_value(
                        campaign, campaign_sha, manifest_sha,
                        stage=stage, reason_code=reason_code,
                    )
                    path.write_bytes(canonical_json_bytes(value))
                    observed, unused_identity = _validate_worker_failure(
                        campaign, campaign_sha, manifest_sha, 0, path,
                    )
                    self.assertEqual(observed["stage"], stage)
                    self.assertEqual(observed["diagnostics"]["reason_code"], reason_code)
            wrong_task = dict(value)
            wrong_task["task_index"] = 1
            path.write_bytes(canonical_json_bytes(wrong_task))
            with self.assertRaises(ManagerError):
                _validate_worker_failure(
                    campaign, campaign_sha, manifest_sha, 0, path,
                )

    def test_exit42_reason_claims_must_be_proven_by_diagnostics(self) -> None:
        campaign = attempt5_campaign()
        campaign_sha = "a" * 64
        manifest_sha = "b" * 64
        cases = []

        operation = deterministic_failure_value(
            campaign, campaign_sha, manifest_sha,
            reason_code="SEALED_PREFLIGHT_OPERATION_EXCEPTION",
        )
        operation["diagnostics"]["preflight"][
            "operation_exception_message_sha256"
        ] = "not-a-digest"
        cases.append(operation)

        numerical = deterministic_failure_value(campaign, campaign_sha, manifest_sha)
        numerical["diagnostics"]["preflight"]["status"] = (
            "PASS_FINITE_CPU_FP64_ALL_SEARCH_BASE"
        )
        cases.append(numerical)

        rss = deterministic_failure_value(
            campaign, campaign_sha, manifest_sha,
            reason_code="PREFLIGHT_RSS_LIMIT_REACHED",
        )
        rss["diagnostics"]["maximum_observed_rss_bytes"] = 1
        rss["diagnostics"]["peak_rss_strictly_below_memory_limit"] = True
        cases.append(rss)

        dtype = deterministic_failure_value(
            campaign, campaign_sha, manifest_sha,
            stage="FORWARD_DTYPE",
            reason_code="OBSERVED_FP64_SHADOW_DTYPE_MISMATCH",
        )
        dtype["diagnostics"]["sequence"].update({
            "logits_dtype": "float64", "logits_dtype_matches": True,
            "hidden_dtype_matches": True,
            "persistent_state_dtype_matches": True, "ce_dtype": "float64",
        })
        cases.append(dtype)
        dtype_missing = deterministic_failure_value(
            campaign, campaign_sha, manifest_sha,
            stage="FORWARD_DTYPE",
            reason_code="OBSERVED_FP64_SHADOW_DTYPE_MISMATCH",
        )
        dtype_missing["diagnostics"]["sequence"] = {}
        cases.append(dtype_missing)

        base_missing = deterministic_failure_value(
            campaign, campaign_sha, manifest_sha,
        )
        base_missing["diagnostics"]["preflight"]["batches"][0][
            "per_sequence"
        ] = [{}]
        cases.append(base_missing)

        search = deterministic_failure_value(
            campaign, campaign_sha, manifest_sha,
            stage="SEARCH_CENTER", reason_code="NONFINITE_SEARCH_OBJECTIVE",
        )
        del search["diagnostics"]["objective"]["sequence_index"]
        cases.append(search)

        terminal = deterministic_failure_value(
            campaign, campaign_sha, manifest_sha,
            stage="TERMINAL_RANKING", reason_code="NONFINITE_TERMINAL_RANKING",
        )
        terminal["diagnostics"]["sequence"] = {
            "loss_class": "FINITE", "loss": 6.0,
            "logits": {"all_finite": True},
            "hidden": {"all_finite": True},
            "per_token_loss": {"all_finite": True},
        }
        cases.append(terminal)

        cache = deterministic_failure_value(
            campaign, campaign_sha, manifest_sha,
            stage="TERMINAL_RANKING",
            reason_code="TERMINAL_CACHE_FLAT_IDENTITY_MISMATCH",
        )
        cache["diagnostics"]["cached_flat_sha256"] = cache["diagnostics"][
            "terminal_flat_sha256"
        ]
        cases.append(cache)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "CPU_SEARCH_FAILURE.json"
            for value in cases:
                path.write_bytes(canonical_json_bytes(value))
                with self.assertRaises(ManagerError):
                    _validate_worker_failure(
                        campaign, campaign_sha, manifest_sha, 0, path,
                    )


class DeterministicRetryTests(unittest.TestCase):
    def _manager(self, temporary: str):
        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = replace(
            config(Path(temporary)),
            wave_root=Path(temporary) / "wave",
            maximum_attempts=3,
        )
        manager.config.wave_root.mkdir(parents=True, exist_ok=True)
        manager.download_root = manager.config.wave_root / "downloads"
        manager.download_root.mkdir(exist_ok=True)
        manager.orchestration_root = manager.config.wave_root / "orchestration"
        manager.orchestration_root.mkdir(exist_ok=True)
        manager.campaign = attempt5_campaign()
        manager.campaign_sha256 = "a" * 64
        manager.snapshot_manifest_sha256 = "b" * 64
        manager.wave_index = int(manager.campaign["wave_index"])
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal(manager.campaign_sha256)
        manager._cancel = threading.Event()
        manager._cleanup_mode = False
        manager._cleanup_deadline_monotonic = 0.0
        manager._run_config = {
            "campaign_deadline_unix_ns": __import__("time").time_ns()
            + 10_000_000_000,
        }
        return manager

    def _terminal_error(self, manager, attempt=0):
        relative = (
            "orchestration/worker-failures/worker-000/"
            f"attempt-{attempt}/CPU_SEARCH_FAILURE.json"
        )
        path = manager.config.wave_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        failure = deterministic_failure_value(
            manager.campaign,
            manager.campaign_sha256,
            manager.snapshot_manifest_sha256,
        )
        path.write_bytes(canonical_json_bytes(failure))
        identity = hash_file(path)
        return DeterministicNumericalWorkerFailure(
            task_index=0,
            attempt=attempt,
            exit_code=42,
            failure=failure,
            failure_identity=identity,
            failure_relative_path=relative,
        )

    def test_validated_exit42_uses_one_attempt_despite_retry_budget(self) -> None:
        class Sandbox:
            id = "sandbox-id"

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            failure = deterministic_failure_value(
                manager.campaign,
                manager.campaign_sha256,
                manager.snapshot_manifest_sha256,
            )

            def download(unused_sandbox, unused_remote, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(canonical_json_bytes(failure))

            manager._download_remote = download
            terminal_error = manager._validated_deterministic_failure(
                Sandbox(), 0, 0, 42,
            )
            self.assertIsInstance(
                terminal_error, DeterministicNumericalWorkerFailure,
            )
            self.assertTrue(
                (manager.config.wave_root / DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH).is_file()
            )
            counts = {"create": 0, "delete": 0}

            def create(*unused):
                counts["create"] += 1
                return Sandbox()

            manager._validate_existing_output = lambda *unused: None
            manager._create_task_sandbox = create
            manager._ensure_session_command = lambda *unused: ("session", "command", 1)
            manager._wait_command = lambda *unused: (_ for _ in ()).throw(terminal_error)
            manager._delete_sandbox = lambda *unused: counts.__setitem__(
                "delete", counts["delete"] + 1,
            )
            with self.assertRaises(DeterministicNumericalWorkerFailure):
                manager._run_task(object(), 0, None)
            self.assertEqual(counts, {"create": 1, "delete": 1})

    def test_unclassified_failure_retains_bounded_retry(self) -> None:
        class Sandbox:
            id = "sandbox-id"

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            counts = {"create": 0, "delete": 0}

            def create(*unused):
                counts["create"] += 1
                return Sandbox()

            manager._validate_existing_output = lambda *unused: None
            manager._create_task_sandbox = create
            manager._ensure_session_command = lambda *unused: ("session", "command", 1)
            manager._wait_command = lambda *unused: (_ for _ in ()).throw(
                ManagerError("transient infrastructure failure")
            )
            manager._delete_sandbox = lambda *unused: counts.__setitem__(
                "delete", counts["delete"] + 1,
            )
            with self.assertRaises(ManagerError):
                manager._run_task(object(), 0, None)
            self.assertEqual(counts, {"create": 3, "delete": 3})

    def test_exit42_wrong_task_artifact_is_rejected_not_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            wrong = deterministic_failure_value(
                manager.campaign,
                manager.campaign_sha256,
                manager.snapshot_manifest_sha256,
            )
            wrong["task_index"] = 1

            def download(unused_sandbox, unused_remote, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(canonical_json_bytes(wrong))

            manager._download_remote = download
            observed = manager._validated_deterministic_failure(
                object(), 0, 0, 42,
            )
            self.assertIsNone(observed)
            self.assertEqual(
                manager.journal.events[-1][0],
                "WORKER_FAILURE_ARTIFACT_REJECTED",
            )

    def test_exit42_missing_malformed_and_wrong_campaign_remain_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)

            def missing(unused_sandbox, unused_remote, unused_destination):
                raise ManagerError("remote failure artifact is absent")

            manager._download_remote = missing
            self.assertIsNone(manager._validated_deterministic_failure(
                object(), 0, 0, 42,
            ))

            def malformed(unused_sandbox, unused_remote, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"{}\n")

            manager._download_remote = malformed
            self.assertIsNone(manager._validated_deterministic_failure(
                object(), 0, 1, 42,
            ))

            wrong_campaign = deterministic_failure_value(
                manager.campaign,
                "f" * 64,
                manager.snapshot_manifest_sha256,
            )

            def wrong_identity(unused_sandbox, unused_remote, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(canonical_json_bytes(wrong_campaign))

            manager._download_remote = wrong_identity
            self.assertIsNone(manager._validated_deterministic_failure(
                object(), 0, 2, 42,
            ))
            self.assertEqual(
                [event for event, unused in manager.journal.events],
                ["WORKER_FAILURE_ARTIFACT_REJECTED"] * 3,
            )
            self.assertFalse(
                (manager.config.wave_root / DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH).exists()
            )

    def test_decision_restart_is_cleanup_only_and_never_relaunches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            terminal_error = self._terminal_error(manager)
            manager._seal_deterministic_failure_decision(terminal_error)
            manager._read_deterministic_failure_terminal = lambda: None
            manager._ensure_snapshot = lambda: self.fail("restart attempted snapshot creation")
            manager._cleanup = lambda *unused: ["injected cleanup failure"]
            with self.assertRaises(ManagerError):
                manager._run_locked()
            self.assertFalse(
                (manager.config.wave_root / "CLOUD_DETERMINISTIC_FAILURE_COMPLETE.json").exists()
            )

            recovered = self._manager(temporary)
            recovered._read_deterministic_failure_terminal = lambda: None
            recovered._ensure_snapshot = lambda: self.fail("restart attempted snapshot creation")
            cleanup_calls = []
            def cleanup(*args):
                cleanup_calls.append(args)
                recovered._last_cleanup_evidence = {
                    "final_remaining_count": 0,
                    "final_relist_succeeded": True,
                    "snapshot_present": False,
                    "snapshot_relist_succeeded": True,
                    "exact_campaign_label_relist_passed": True,
                    "snapshot_exact_name_not_found_confirmed": True,
                }
                return []
            recovered._cleanup = cleanup
            result = recovered._run_locked()
            self.assertEqual(
                result["status"],
                "FAIL_DETERMINISTIC_NUMERICAL_FAILURE_CLEANED_TERMINAL",
            )
            self.assertEqual(len(cleanup_calls), 1)
            self.assertTrue(
                (recovered.config.wave_root / DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH).is_file()
            )

    def test_task_n_exit42_seals_truthful_activity_and_restart_is_cloud_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            journal_root = manager.orchestration_root / "receipts"
            manager.journal = ReceiptJournal(journal_root, manager.campaign_sha256)
            manager.journal.append(
                "WORKER_OUTPUT_DOWNLOADED_AND_VALIDATED", task_index=0,
            )
            manager.journal.append("WORKER_COMMAND_STARTED", task_index=1, attempt=0)
            relative = (
                "orchestration/worker-failures/worker-001/"
                "attempt-0/CPU_SEARCH_FAILURE.json"
            )
            failure_path = manager.config.wave_root / relative
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure = deterministic_failure_value(
                manager.campaign, manager.campaign_sha256,
                manager.snapshot_manifest_sha256, task_index=1,
                stage="SEARCH_CENTER", reason_code="NONFINITE_SEARCH_OBJECTIVE",
            )
            failure_path.write_bytes(canonical_json_bytes(failure))
            error = DeterministicNumericalWorkerFailure(
                task_index=1, attempt=0, exit_code=42, failure=failure,
                failure_identity=hash_file(failure_path),
                failure_relative_path=relative,
            )
            manager._seal_deterministic_failure_decision(error)
            manager._last_cleanup_evidence = {
                "final_remaining_count": 0,
                "final_relist_succeeded": True,
                "snapshot_present": False,
                "snapshot_relist_succeeded": True,
                "exact_campaign_label_relist_passed": True,
                "snapshot_exact_name_not_found_confirmed": True,
            }
            terminal = manager._seal_deterministic_failure_terminal(error)
            self.assertEqual(terminal["task_index"], 1)
            self.assertEqual(terminal["fanout_tasks_started"], 1)
            self.assertEqual(terminal["cloud_results_downloaded"], 1)

            recovered = self._manager(temporary)
            recovered.journal = ReceiptJournal(journal_root, recovered.campaign_sha256)
            recovered._ensure_snapshot = lambda: self.fail(
                "terminal restart attempted a cloud resource"
            )
            observed = recovered._run_locked()
            self.assertEqual(observed["task_index"], 1)

    def test_seed_measured_deadline_gate_passes_only_when_both_windows_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            manager.config = replace(manager.config, worker_timeout_seconds=1000)
            manager._run_config = {
                "campaign_deadline_unix_ns": __import__("time").time_ns()
                + 2_000_000_000_000,
            }
            seed = {
                "worker_total_seconds": 10.0,
                "total_physical_model_forwards": 408,
            }
            receipt = manager._fanout_deadline_safety_receipt(seed)
            self.assertLessEqual(
                receipt["worker_window_required_seconds"],
                manager.config.worker_timeout_seconds,
            )
            manager._enforce_fanout_deadline_safety(receipt)

            manager.config = replace(manager.config, worker_timeout_seconds=40)
            with self.assertRaises(ManagerError):
                manager._fanout_deadline_safety_receipt(seed)

    def test_seed_measured_deadline_gate_rejects_insufficient_campaign_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            manager.config = replace(manager.config, worker_timeout_seconds=1000)
            manager._run_config = {
                "campaign_deadline_unix_ns": __import__("time").time_ns()
                + 600_000_000_000,
            }
            with self.assertRaises(ManagerError):
                manager._fanout_deadline_safety_receipt({
                    "worker_total_seconds": 10.0,
                    "total_physical_model_forwards": 408,
                })


class ImageAndConfigTests(unittest.TestCase):
    @contextmanager
    def _disposable_manager(self):
        temporary_parent = PACKAGE_ROOT / "runtime/tests/.temporary"
        temporary_parent.mkdir(exist_ok=True)
        with ExitStack() as stack:
            temporary = stack.enter_context(tempfile.TemporaryDirectory(
                prefix="hfw_w560_manager_source_",
                dir=temporary_parent,
            ))
            upload_temporary = stack.enter_context(tempfile.TemporaryDirectory(
                prefix="hfw_w560_manager_upload_",
                dir="/private/tmp",
            ))
            temporary_root = Path(temporary)
            package_root = temporary_root / "package"
            wave_root = package_root / "runtime/waves/wave-00000001"
            snapshot_root = wave_root / "snapshot"
            shutil.copytree(WAVE_ROOT / "snapshot", snapshot_root)
            upload_root = Path(upload_temporary) / "upload"
            shutil.copytree(snapshot_root, upload_root)
            entries = [upload_root, *upload_root.rglob("*")]
            for path in (value for value in entries if value.is_file()):
                path.chmod(0o400)
            for path in sorted(
                (value for value in entries if value.is_dir()),
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                path.chmod(0o500)
            manager = DaytonaManager.__new__(DaytonaManager)
            manager.config = replace(
                config(package_root),
                wave_root=wave_root,
                upload_snapshot_root=upload_root,
            )
            manager.snapshot_root = snapshot_root
            manager.campaign = json.loads(
                (manager.snapshot_root / "CAMPAIGN.json").read_text()
            )
            manager.campaign_sha256 = hash_file(
                manager.snapshot_root / "CAMPAIGN.json"
            )["sha256"]
            manager.snapshot_manifest_sha256 = hash_file(
                manager.snapshot_root / "SNAPSHOT_MANIFEST.json"
            )["sha256"]
            manager.wave_index = int(manager.campaign["wave_index"])
            manager.upload_snapshot_identity = validate_upload_snapshot_root(
                manager.snapshot_root, upload_root,
            )
            self._write_staging_receipt(manager, package_root)
            manager.upload_staging_receipt_identity = hash_file(
                package_root / "runtime/config/UPLOAD_SNAPSHOT_STAGING.json"
            )
            manager.credential = "not-a-real-key"
            manager.journal = FakeJournal(manager.campaign_sha256)
            manager._cancel = threading.Event()
            manager._cleanup_mode = False
            manager._run_config = {
                "campaign_deadline_unix_ns": __import__("time").time_ns()
                + 10_000_000_000,
            }
            try:
                yield manager, upload_root, package_root
            finally:
                if temporary_root.exists():
                    all_entries = [temporary_root, *temporary_root.rglob("*")]
                    for path in all_entries:
                        if path.is_file():
                            path.chmod(0o600)
                    for path in sorted(
                        (value for value in all_entries if value.is_dir()),
                        key=lambda value: len(value.parts),
                    ):
                        path.chmod(0o700)

    def _assert_sealed_modes(self, upload_root: Path) -> None:
        entries = [upload_root, *upload_root.rglob("*")]
        self.assertTrue(entries)
        self.assertFalse(any(path.stat().st_mode & 0o222 for path in entries))
        for path in entries:
            expected = 0o500 if path.is_dir() else 0o400
            self.assertEqual(path.stat().st_mode & 0o777, expected)

    def _write_staging_receipt(
        self,
        manager: DaytonaManager,
        package_root: Path,
    ) -> None:
        receipt = build_upload_staging_receipt(
            package_root,
            manager.snapshot_root,
            manager.config.upload_snapshot_root,
        )
        path = package_root / "runtime/config/UPLOAD_SNAPSHOT_STAGING.json"
        path.parent.mkdir(parents=True)
        write_exclusive(path, canonical_json_bytes(receipt))

    @contextmanager
    def _disposable_staging_receipt_fixture(self):
        with tempfile.TemporaryDirectory(
            prefix="hfw_w560_receipt_",
            dir="/private/tmp",
        ) as temporary:
            root = Path(temporary)
            package_root = root / "package"
            snapshot_root = (
                package_root / "runtime/waves/wave-00000001/snapshot"
            )
            snapshot_root.mkdir(parents=True)
            (snapshot_root / "CAMPAIGN.json").write_bytes(
                canonical_json_bytes({"wave_index": 1})
            )
            (snapshot_root / "SNAPSHOT_MANIFEST.json").write_bytes(
                canonical_json_bytes({"schema": "disposable-test-manifest"})
            )
            (snapshot_root / "payload.bin").write_bytes(b"disposable-staging")
            upload_root = root / "upload"
            shutil.copytree(snapshot_root, upload_root)
            for path in upload_root.rglob("*"):
                if path.is_file():
                    path.chmod(0o400)
            upload_root.chmod(0o500)
            files = {
                path.relative_to(upload_root).as_posix(): {
                    "sha256": hash_file(path)["sha256"],
                    "size_bytes": hash_file(path)["size_bytes"],
                }
                for path in sorted(upload_root.rglob("*"))
                if path.is_file()
            }
            identity = {
                "campaign_sha256": files["CAMPAIGN.json"]["sha256"],
                "snapshot_manifest_sha256": files[
                    "SNAPSHOT_MANIFEST.json"
                ]["sha256"],
                "files": files,
                "file_count": len(files),
                "total_size_bytes": sum(
                    int(value["size_bytes"]) for value in files.values()
                ),
            }
            manager = DaytonaManager.__new__(DaytonaManager)
            manager.config = replace(
                config(package_root),
                wave_root=snapshot_root.parent,
                upload_snapshot_root=upload_root,
            )
            manager.snapshot_root = snapshot_root
            manager.upload_snapshot_identity = identity
            self._write_staging_receipt(manager, package_root)
            try:
                yield manager, upload_root, package_root
            finally:
                if upload_root.exists():
                    upload_root.chmod(0o700)
                    for path in upload_root.rglob("*"):
                        if path.is_file():
                            path.chmod(0o600)

    def test_sdk_image_render_restores_sealed_staging_read_only_modes(self) -> None:
        with self._disposable_manager() as (manager, upload_root, unused_package):
            manager._build_image().dockerfile()
            self._assert_sealed_modes(upload_root)

    def test_parser_constructs_complete_config_and_image_is_fully_locked(self) -> None:
        with self._disposable_manager() as (manager, upload_root, package_root):
            parser = _build_parser()
            args = parser.parse_args([
                "--package-root", str(package_root),
                "--wave-root", str(WAVE_ROOT),
                "--upload-snapshot-root", str(upload_root),
                "--base-image", BASE_IMAGE,
            ])
            self.assertEqual(args.target, "us")
            self.assertEqual(args.api_url, "https://app.daytona.io/api")
            dockerfile = manager._build_image().dockerfile()
            for requirement in LOCKED_RUNTIME_PACKAGES:
                self.assertIn(requirement, dockerfile)
            for filename, digest in EXPECTED_WHEEL_SHA256.items():
                self.assertIn(filename, dockerfile)
                self.assertIn(digest, dockerfile)
            self.assertIn("--no-index --no-deps", dockerfile)

    def test_expected_wheel_rows_render_as_one_dockerfile_instruction(self) -> None:
        with self._disposable_manager() as (manager, unused_upload, unused_package):
            lines = manager._build_image().dockerfile().splitlines()
            expected_commands = [
                line for line in lines
                if line.startswith("RUN printf '%s\\n' ")
                and line.endswith(
                    " > /opt/w560-build/expected-wheel-sha256.txt"
                )
            ]
            self.assertEqual(len(expected_commands), 1)
            self.assertEqual(
                expected_commands[0].count("  /tmp/w560-wheelhouse/"),
                len(EXPECTED_WHEEL_SHA256),
            )
            for line in lines:
                hash_at_column_zero = (
                    len(line) > 64
                    and all(
                        character in "0123456789abcdef"
                        for character in line[:64]
                    )
                    and line[64].isspace()
                )
                self.assertFalse(hash_at_column_zero, line)

    def test_upload_copy_is_single_source_and_has_no_space(self) -> None:
        with self._disposable_manager() as (manager, upload_root, unused_package):
            dockerfile = manager._build_image().dockerfile()
            copy_lines = [
                line for line in dockerfile.splitlines()
                if line.startswith("COPY ")
            ]
            self.assertEqual(len(copy_lines), 1)
            self.assertEqual(
                copy_lines[0].split(),
                ["COPY", str(upload_root).lstrip("/"), "/opt/w560"],
            )
            self.assertNotIn("SP AI 01_16", copy_lines[0])

    def test_upload_snapshot_is_exact_authoritative_inventory(self) -> None:
        with self._disposable_staging_receipt_fixture() as (
            manager, unused_upload, unused_package,
        ):
            identity = manager.upload_snapshot_identity
            self.assertEqual(identity["file_count"], len(identity["files"]))
            self.assertEqual(
                identity["total_size_bytes"],
                sum(
                    int(record["size_bytes"])
                    for record in identity["files"].values()
                ),
            )
            receipt_identity = validate_upload_staging_receipt(
                manager.config, manager.snapshot_root, identity,
            )
            self.assertEqual(len(receipt_identity["sha256"]), 64)

    def test_upload_root_rejects_spaces_and_symlink_components(self) -> None:
        with tempfile.TemporaryDirectory(prefix="manager upload space ") as temporary:
            with self.assertRaises(ManagerError):
                _validate_upload_root_path(Path(temporary))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ManagerError):
                _validate_upload_root_path(linked)

    def test_alternate_exact_copy_path_is_not_the_sealed_staging_root(self) -> None:
        with self._disposable_staging_receipt_fixture() as (
            manager, unused_upload, package_root,
        ):
            alternate_root = package_root.parent / "alternate"
            alternate_root.mkdir()
            alternate = replace(
                manager.config,
                upload_snapshot_root=alternate_root,
            )
            with self.assertRaises(ManagerError):
                validate_upload_staging_receipt(
                    alternate, manager.snapshot_root,
                    manager.upload_snapshot_identity,
                )

    def test_live_staging_proof_rejects_exact_byte_hardlink_replacement(self) -> None:
        with self._disposable_staging_receipt_fixture() as (
            manager, upload_root, unused_package,
        ):
            source = manager.snapshot_root / "payload.bin"
            uploaded = upload_root / "payload.bin"
            upload_root.chmod(0o700)
            uploaded.unlink()
            os.link(source, uploaded)
            uploaded.chmod(0o400)
            upload_root.chmod(0o500)
            with self.assertRaisesRegex(
                ManagerError, "live upload staging proof failed",
            ):
                validate_upload_staging_receipt(
                    manager.config,
                    manager.snapshot_root,
                    manager.upload_snapshot_identity,
                )

    def test_image_build_rejects_identity_drift_before_add_local_dir(self) -> None:
        with self._disposable_manager() as (manager, unused_upload, unused_package):
            manager.upload_snapshot_identity = {"inventory_sha256": "0" * 64}
            with self.assertRaises(ManagerError):
                manager._build_image()

    def test_snapshot_create_restores_modes_after_deferred_sdk_consumption(self) -> None:
        with self._disposable_manager() as (manager, upload_root, unused_package):
            manager.config = replace(manager.config, maximum_attempts=1)
            manager._find_snapshot = lambda: None
            manager._build_image = lambda: BASE_IMAGE
            target = upload_root / "CAMPAIGN.json"

            class Snapshot:
                id = "created-snapshot-id"
                name = _snapshot_name(manager.campaign_sha256, manager.wave_index)
                state = "active"
                size = 1.25
                cpu = int(manager.campaign["proposal_lane"]["cpu_limit"])
                mem = int(manager.campaign["proposal_lane"]["memory_limit_gib"])
                disk = int(manager.campaign["proposal_lane"]["disk_limit_gib"])
                gpu = 0
                region_ids = [manager.config.target]

            class SnapshotService:
                def create(self, unused_params, timeout=None):
                    target.chmod(0o600)
                    return Snapshot()

            class Client:
                snapshot = SnapshotService()

            manager._client = lambda: Client()
            observed = manager._ensure_snapshot()
            self.assertEqual(observed.id, Snapshot.id)
            self.assertEqual(
                manager.journal.events[-1][1]["snapshot"]["size_bytes"],
                1.25,
            )
            self._assert_sealed_modes(upload_root)

    def test_snapshot_create_rejects_bytes_changed_during_sdk_consumption(self) -> None:
        with self._disposable_manager() as (manager, upload_root, unused_package):
            manager.config = replace(manager.config, maximum_attempts=1)
            manager._find_snapshot = lambda: None
            manager._build_image = lambda: BASE_IMAGE
            target = upload_root / "CAMPAIGN.json"

            class SnapshotService:
                def create(self, unused_params, timeout=None):
                    target.chmod(0o600)
                    with target.open("ab") as stream:
                        stream.write(b"tamper")
                    return object()

            class Client:
                snapshot = SnapshotService()

            manager._client = lambda: Client()
            with self.assertRaisesRegex(
                ManagerError, "changed sealed upload snapshot bytes",
            ):
                manager._ensure_snapshot()
            self._assert_sealed_modes(upload_root)

    def test_snapshot_create_rejects_identical_byte_inode_replacement(self) -> None:
        with self._disposable_manager() as (manager, upload_root, unused_package):
            manager.config = replace(manager.config, maximum_attempts=1)
            manager._find_snapshot = lambda: None
            manager._build_image = lambda: BASE_IMAGE
            target = upload_root / "CAMPAIGN.json"

            class SnapshotService:
                def create(self, unused_params, timeout=None):
                    payload = target.read_bytes()
                    upload_root.chmod(0o700)
                    target.unlink()
                    target.write_bytes(payload)
                    return object()

            class Client:
                snapshot = SnapshotService()

            manager._client = lambda: Client()
            with self.assertRaisesRegex(
                ManagerError,
                "changed sealed upload staging filesystem identity",
            ):
                manager._ensure_snapshot()
            self._assert_sealed_modes(upload_root)


class FakeJournal:
    def __init__(self, campaign_sha256: str = "f" * 64) -> None:
        self.events = []
        self.campaign_sha256 = campaign_sha256
        self._records = []

    def append(self, event: str, **fields):
        self.events.append((event, fields))
        previous = None
        if self._records:
            previous = sha256_bytes(canonical_json_bytes(self._records[-1]))
        record = {
            "schema": "hfw_w560_daytona_manager_mutation_receipt_v1",
            "receipt_index": len(self._records),
            "campaign_sha256": self.campaign_sha256,
            "previous_receipt_sha256": previous,
            "event": event,
            "recorded_unix_ns": len(self._records) + 1,
            **fields,
        }
        self._records.append(record)
        return record

    @property
    def records(self):
        return tuple(self._records)

    def latest(self, event: str, **matches):
        for record in reversed(self._records):
            if record["event"] == event and all(
                record.get(key) == value for key, value in matches.items()
            ):
                return record
        return None


class FakeSandbox:
    def __init__(self, campaign_sha: str) -> None:
        self.id = "sandbox-id"
        self.name = _task_name(campaign_sha, 0, 1, 0)
        self.state = "started"
        self.desired_state = "started"
        self.snapshot = "snapshot"
        self.target = "us"
        self.cpu = 2
        self.memory = 4
        self.disk = 8
        self.gpu = 0
        self.network_block_all = True
        self.public = False
        self.auto_stop_interval = 31
        self.auto_delete_interval = 0
        self.labels = _task_labels(campaign_sha, 0, 1, 0)
        self.deleted = False

    def delete(self, timeout=None):
        self.deleted = True


class CleanupTests(unittest.TestCase):
    def test_cleanup_ignores_fanout_cancel_and_proves_zero(self) -> None:
        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = config(Path("/tmp/manager-test"))
        manager.campaign_sha256 = "c" * 64
        manager.wave_index = 0
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal(manager.campaign_sha256)
        manager._cancel = threading.Event()
        manager._cancel.set()
        manager._cleanup_mode = False
        manager._cleanup_deadline_monotonic = 0.0
        manager._cleanup_evidence_lock = threading.Lock()
        manager._confirmed_absent_sandbox_ids = set()
        sandbox = FakeSandbox(manager.campaign_sha256)
        calls = {"list": 0}

        def list_sandboxes():
            calls["list"] += 1
            return [sandbox] if calls["list"] == 1 else []

        manager._all_campaign_sandboxes = list_sandboxes
        manager._find_snapshot = lambda: None
        class Client:
            def get(self, unused_sandbox_id):
                from daytona import DaytonaNotFoundError
                raise DaytonaNotFoundError("absent after accepted delete")
        manager._client = lambda: Client()
        failures = manager._cleanup(None, "INJECTED_FAILURE")
        self.assertEqual(failures, [])
        self.assertTrue(sandbox.deleted)
        self.assertGreaterEqual(calls["list"], 2)
        self.assertEqual(manager._last_cleanup_evidence["initial_campaign_sandbox_count"], 1)
        self.assertEqual(manager._last_cleanup_evidence["final_remaining_count"], 0)
        self.assertEqual(manager._last_cleanup_evidence["final_remaining_ids"], [])
        event, fields = manager.journal.events[-1]
        self.assertEqual(event, "CLEANUP_TERMINAL")
        self.assertEqual(fields["initial_campaign_sandbox_count"], 1)
        self.assertEqual(fields["campaign_sandbox_count"], 0)

    def test_delete_not_found_is_success(self) -> None:
        from daytona import DaytonaNotFoundError

        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = config(Path("/tmp/manager-test"))
        manager.campaign_sha256 = "d" * 64
        manager.wave_index = 0
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal()
        manager._cancel = threading.Event()
        manager._cleanup_mode = True
        manager._cleanup_deadline_monotonic = __import__("time").monotonic() + 10
        manager._cleanup_evidence_lock = threading.Lock()
        manager._confirmed_absent_sandbox_ids = set()
        sandbox = FakeSandbox(manager.campaign_sha256)

        def absent(timeout=None):
            raise DaytonaNotFoundError("already absent")

        sandbox.delete = absent
        class Client:
            def get(self, unused_sandbox_id):
                raise DaytonaNotFoundError("exact id absent")
        manager._client = lambda: Client()
        manager._delete_sandbox(sandbox, "TEST")
        self.assertEqual(manager.journal.events[-1][0], "SANDBOX_DELETED")

    def test_delete_ownership_allows_service_labels_and_rejects_new_hfw(self) -> None:
        from daytona import DaytonaNotFoundError

        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = config(Path("/tmp/manager-test"))
        manager.campaign = attempt5_campaign()
        manager.campaign_sha256 = "d" * 64
        manager.wave_index = 0
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal(manager.campaign_sha256)
        manager._cancel = threading.Event()
        manager._cleanup_mode = True
        manager._cleanup_deadline_monotonic = (
            __import__("time").monotonic() + 10
        )
        manager._cleanup_evidence_lock = threading.Lock()
        manager._confirmed_absent_sandbox_ids = set()

        class Client:
            def get(self, unused_sandbox_id):
                raise DaytonaNotFoundError("exact id absent")

        manager._client = lambda: Client()
        sandbox = FakeSandbox(manager.campaign_sha256)
        sandbox.labels["code-toolbox-language"] = "python"
        manager._delete_sandbox(sandbox, "TEST_SERVICE_LABEL")

        tampered = FakeSandbox(manager.campaign_sha256)
        tampered.labels["hfw_unexpected_owner"] = "forbidden"
        with self.assertRaisesRegex(ManagerError, "unexpected hfw labels"):
            manager._delete_sandbox(tampered, "TEST_UNEXPECTED_HFW")

    def test_delete_accept_returns_while_get_lingers_then_notfound(self) -> None:
        from daytona import DaytonaNotFoundError

        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = replace(
            config(Path("/tmp/manager-test")), poll_seconds=0,
        )
        manager.campaign_sha256 = "7" * 64
        manager.wave_index = 0
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal()
        manager._cancel = threading.Event()
        manager._cleanup_mode = True
        manager._cleanup_deadline_monotonic = __import__("time").monotonic() + 10
        manager._cleanup_evidence_lock = threading.Lock()
        manager._confirmed_absent_sandbox_ids = set()
        sandbox = FakeSandbox(manager.campaign_sha256)
        observations = iter((sandbox, sandbox, DaytonaNotFoundError("gone")))

        class Client:
            def get(self, unused_sandbox_id):
                value = next(observations)
                if isinstance(value, BaseException):
                    raise value
                return value

        manager._client = lambda: Client()
        manager._delete_sandbox(sandbox, "TEST_LINGER")
        self.assertTrue(sandbox.deleted)
        events = [event for event, unused in manager.journal.events]
        self.assertEqual(events.count("SANDBOX_DELETE_ACCEPTED"), 1)
        self.assertEqual(events.count("SANDBOX_DELETE_VISIBILITY_STALE"), 2)
        self.assertEqual(events[-1], "SANDBOX_DELETED")

    def test_delete_404_still_polls_exact_get_until_notfound(self) -> None:
        from daytona import DaytonaNotFoundError

        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = replace(
            config(Path("/tmp/manager-test")), poll_seconds=0,
        )
        manager.campaign_sha256 = "8" * 64
        manager.wave_index = 0
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal()
        manager._cancel = threading.Event()
        manager._cleanup_mode = True
        manager._cleanup_deadline_monotonic = __import__("time").monotonic() + 10
        manager._cleanup_evidence_lock = threading.Lock()
        manager._confirmed_absent_sandbox_ids = set()
        sandbox = FakeSandbox(manager.campaign_sha256)
        sandbox.delete = lambda timeout=None: (_ for _ in ()).throw(
            DaytonaNotFoundError("delete endpoint says absent")
        )
        observations = iter((sandbox, DaytonaNotFoundError("exact get absent")))

        class Client:
            def get(self, unused_sandbox_id):
                value = next(observations)
                if isinstance(value, BaseException):
                    raise value
                return value

        manager._client = lambda: Client()
        manager._delete_sandbox(sandbox, "TEST_DELETE_404")
        events = [event for event, unused in manager.journal.events]
        self.assertIn("SANDBOX_DELETE_VISIBILITY_STALE", events)
        self.assertEqual(events[-1], "SANDBOX_DELETED")

    def test_delete_transient_get_is_not_absence(self) -> None:
        from daytona import DaytonaNotFoundError

        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = replace(
            config(Path("/tmp/manager-test")), poll_seconds=0,
            maximum_attempts=3,
        )
        manager.campaign_sha256 = "6" * 64
        manager.wave_index = 0
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal()
        manager._cancel = threading.Event()
        manager._cleanup_mode = True
        manager._cleanup_deadline_monotonic = __import__("time").monotonic() + 10
        manager._cleanup_evidence_lock = threading.Lock()
        manager._confirmed_absent_sandbox_ids = set()
        sandbox = FakeSandbox(manager.campaign_sha256)
        observations = iter((RuntimeError("transient"), sandbox, DaytonaNotFoundError("gone")))

        class Client:
            def get(self, unused_sandbox_id):
                value = next(observations)
                if isinstance(value, BaseException):
                    raise value
                return value

        manager._client = lambda: Client()
        manager._delete_sandbox(sandbox, "TEST_TRANSIENT_GET")
        events = [event for event, unused in manager.journal.events]
        self.assertIn("SDK_CALL_RETRY", events)
        self.assertIn("SANDBOX_DELETE_VISIBILITY_STALE", events)
        self.assertEqual(events[-1], "SANDBOX_DELETED")

    def test_delete_stuck_exact_id_times_out_without_deleted_receipt(self) -> None:
        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = replace(
            config(Path("/tmp/manager-test")), poll_seconds=0,
            api_timeout_seconds=0.002,
        )
        manager.campaign_sha256 = "5" * 64
        manager.wave_index = 0
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal()
        manager._cancel = threading.Event()
        manager._cleanup_mode = True
        manager._cleanup_deadline_monotonic = __import__("time").monotonic() + 10
        manager._cleanup_evidence_lock = threading.Lock()
        manager._confirmed_absent_sandbox_ids = set()
        sandbox = FakeSandbox(manager.campaign_sha256)

        class Client:
            def get(self, unused_sandbox_id):
                return sandbox

        manager._client = lambda: Client()
        with self.assertRaises(ManagerError):
            manager._delete_sandbox(sandbox, "TEST_STUCK")
        events = [event for event, unused in manager.journal.events]
        self.assertNotIn("SANDBOX_DELETED", events)

    def test_build_failed_snapshot_is_owned_and_deleted(self) -> None:
        class Snapshot:
            id = "failed-snapshot-id"
            state = "build_failed"
            size = 1.25
            cpu = 2
            mem = 4
            disk = 8
            gpu = 0
            region_ids = ["us"]

        class SnapshotService:
            def __init__(self):
                self.deleted = False

            def delete(self, snapshot):
                self.deleted = True

        class Client:
            def __init__(self):
                self.snapshot = SnapshotService()

        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = config(Path("/tmp/manager-test"))
        manager.campaign_sha256 = "e" * 64
        manager.wave_index = 0
        manager.campaign = {
            "proposal_lane": {"cpu_limit": 2, "memory_limit_gib": 4, "disk_limit_gib": 8},
        }
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal()
        manager._cancel = threading.Event()
        manager._cleanup_mode = False
        manager._cleanup_deadline_monotonic = 0.0
        manager._all_campaign_sandboxes = lambda: []
        snapshot = Snapshot()
        snapshot.name = _snapshot_name(manager.campaign_sha256, 0)
        calls = {"find": 0}

        def find():
            calls["find"] += 1
            return snapshot if calls["find"] == 1 else None

        manager._find_snapshot = find
        client = Client()
        manager._daytona = client
        manager._client = lambda: client
        failures = manager._cleanup(None, "INJECTED_BUILD_FAILURE")
        self.assertEqual(failures, [])
        self.assertTrue(client.snapshot.deleted)

    def test_snapshot_delete_polls_lingering_identity_until_not_found(self) -> None:
        class Snapshot:
            id = "lingering-snapshot-id"
            state = "build_failed"
            size = 1.25
            cpu = 2
            mem = 4
            disk = 8
            gpu = 0
            region_ids = ["us"]

        class SnapshotService:
            def __init__(self):
                self.delete_calls = 0

            def delete(self, snapshot):
                self.delete_calls += 1

        class Client:
            def __init__(self):
                self.snapshot = SnapshotService()

        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = replace(config(Path("/tmp/manager-test")), poll_seconds=0)
        manager.campaign_sha256 = "f" * 64
        manager.wave_index = 0
        manager.campaign = {
            "proposal_lane": {"cpu_limit": 2, "memory_limit_gib": 4, "disk_limit_gib": 8},
        }
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal()
        manager._cancel = threading.Event()
        manager._cleanup_mode = True
        manager._cleanup_deadline_monotonic = __import__("time").monotonic() + 10
        snapshot = Snapshot()
        snapshot.name = _snapshot_name(manager.campaign_sha256, 0)
        observations = iter((snapshot, snapshot, None))
        manager._find_snapshot = lambda: next(observations)
        client = Client()
        manager._daytona = client
        manager._client = lambda: client
        manager._delete_snapshot_exact(snapshot, "TEST_LINGER")
        self.assertEqual(client.snapshot.delete_calls, 1)
        event, fields = manager.journal.events[-1]
        self.assertEqual(event, "SNAPSHOT_DELETED")
        self.assertEqual(fields["snapshot"]["size_bytes"], 1.25)
        self.assertTrue(fields["exact_name_not_found_confirmed"])

    def test_snapshot_delete_timeout_never_emits_deleted_receipt(self) -> None:
        class Snapshot:
            id = "stuck-snapshot-id"
            state = "removing"
            size = None
            cpu = 2
            mem = 4
            disk = 8
            gpu = 0
            region_ids = ["us"]

        class SnapshotService:
            def delete(self, snapshot):
                return None

        class Client:
            snapshot = SnapshotService()

        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = replace(config(Path("/tmp/manager-test")), poll_seconds=0)
        manager.campaign_sha256 = "9" * 64
        manager.wave_index = 0
        manager.campaign = {
            "proposal_lane": {"cpu_limit": 2, "memory_limit_gib": 4, "disk_limit_gib": 8},
        }
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal()
        manager._cancel = threading.Event()
        manager._cleanup_mode = True
        manager._cleanup_deadline_monotonic = __import__("time").monotonic() + 0.002
        snapshot = Snapshot()
        snapshot.name = _snapshot_name(manager.campaign_sha256, 0)
        manager._find_snapshot = lambda: snapshot
        client = Client()
        manager._daytona = client
        manager._client = lambda: client
        with self.assertRaises(TimeoutError):
            manager._delete_snapshot_exact(snapshot, "TEST_TIMEOUT")
        self.assertNotIn("SNAPSHOT_DELETED", [event for event, _ in manager.journal.events])


class RecoveryTests(unittest.TestCase):
    @staticmethod
    def _recovery_rows(
        source_sandbox,
        predecessor_probe_sandbox,
        immediate_probe_sandbox,
    ):
        return [
            {
                "role": "ORIGINAL_SOURCE_TASK0",
                "sandbox": source_sandbox,
            },
            {
                "role": "PREDECESSOR_LABEL_DRIFT_PROBE",
                "sandbox": predecessor_probe_sandbox,
            },
            {
                "role": "IMMEDIATE_PREDECESSOR_QUOTA_API_PROBE",
                "sandbox": immediate_probe_sandbox,
            },
        ]

    def _manager(self, temporary: str) -> DaytonaManager:
        campaign_path = WAVE_ROOT / "snapshot/CAMPAIGN.json"
        manifest_path = WAVE_ROOT / "snapshot/SNAPSHOT_MANIFEST.json"
        manager = DaytonaManager.__new__(DaytonaManager)
        manager.config = config(Path(temporary))
        manager.config.wave_root.mkdir(parents=True)
        manager.download_root = manager.config.wave_root / "downloads"
        manager.download_root.mkdir()
        manager.orchestration_root = manager.config.wave_root / "orchestration"
        manager.orchestration_root.mkdir()
        manager.campaign = json.loads(campaign_path.read_text())
        manager.campaign_sha256 = hash_file(campaign_path)["sha256"]
        manager.snapshot_manifest_sha256 = hash_file(manifest_path)["sha256"]
        manager.wave_index = 0
        manager.credential = "not-a-real-key"
        manager.journal = FakeJournal(manager.campaign_sha256)
        manager._cancel = threading.Event()
        manager._cleanup_mode = False
        manager._run_config = {
            "campaign_deadline_unix_ns": __import__("time").time_ns() + 10_000_000_000,
        }
        return manager

    def _probe_fixture(self, manager: DaytonaManager, labels):
        class Snapshot:
            id = "snapshot-id"
            name = _snapshot_name(manager.campaign_sha256, manager.wave_index)

        class Sandbox:
            id = "probe-sandbox-id"
            name = _probe_name(manager.campaign_sha256, manager.wave_index)
            state = "started"
            desired_state = "started"
            snapshot = Snapshot.name
            target = manager.config.target
            cpu = int(manager.campaign["proposal_lane"]["cpu_limit"])
            memory = int(manager.campaign["proposal_lane"]["memory_limit_gib"])
            disk = int(manager.campaign["proposal_lane"]["disk_limit_gib"])
            gpu = 0
            network_block_all = True
            public = False
            auto_stop_interval = (
                (manager.config.worker_timeout_seconds + 59) // 60 + 30
            )
            auto_delete_interval = 0
            env = dict(WORKER_ENVIRONMENT)

        Sandbox.labels = dict(labels)
        return Snapshot(), Sandbox()

    @staticmethod
    def _usage_row(
        region_id="us", sandbox_class="linux-vm",
        current_cpu=2, current_memory=4, current_disk=8, current_gpu=0,
        total_cpu=1000, total_memory=2000, total_disk=4000, total_gpu=0,
        max_cpu_per_sandbox=2, max_memory_per_sandbox=4,
        max_disk_per_sandbox=8,
    ):
        return SimpleNamespace(
            region_id=region_id,
            sandbox_class=SimpleNamespace(value=sandbox_class),
            current_cpu_usage=current_cpu,
            current_memory_usage=current_memory,
            current_disk_usage=current_disk,
            current_gpu_usage=current_gpu,
            total_cpu_quota=total_cpu,
            total_memory_quota=total_memory,
            total_disk_quota=total_disk,
            total_gpu_quota=total_gpu,
            max_cpu_per_sandbox=max_cpu_per_sandbox,
            max_memory_per_sandbox=max_memory_per_sandbox,
            max_disk_per_sandbox=max_disk_per_sandbox,
        )

    def _quota_probe(self, manager: DaytonaManager):
        unused_snapshot, probe = self._probe_fixture(
            manager,
            _probe_labels(manager.campaign_sha256, manager.wave_index),
        )
        probe.organization_id = "organization-id"
        probe.sandbox_class = "linux-vm"
        return probe

    def _call_quota_receipt(
        self, manager: DaytonaManager, probe, rows, additional_workers=None,
    ):
        api_client = object()
        manager._client = lambda: SimpleNamespace(_api_client=api_client)
        manager._retry = lambda unused_operation, call: call()
        calls = []

        class FakeOrganizationsApi:
            def __init__(self, observed_api_client):
                if observed_api_client is not api_client:
                    raise AssertionError("organization usage client binding drift")

            def get_organization_usage_overview(
                self, organization_id, _request_timeout=None,
            ):
                calls.append((organization_id, _request_timeout))
                return SimpleNamespace(region_usage=rows)

        if additional_workers is None:
            additional_workers = int(manager.campaign["worker_count"]) - 1
        with patch(
            "daytona_api_client.api.organizations_api.OrganizationsApi",
            FakeOrganizationsApi,
        ):
            value = manager._quota_receipt(probe, additional_workers)
        return value, calls

    def test_sdk_resource_identities_require_nonnegative_integral_resources(self) -> None:
        sandbox = SimpleNamespace(
            id="sandbox-id",
            name="sandbox-name",
            state="started",
            desired_state="started",
            snapshot="snapshot-name",
            target="us",
            cpu=2.0,
            memory=4.0,
            disk=8.0,
            gpu=0.0,
            network_block_all=True,
            public=False,
            auto_stop_interval=31,
            auto_delete_interval=0,
            labels={},
        )
        snapshot = SimpleNamespace(
            id="snapshot-id",
            name="snapshot-name",
            state="active",
            size=16.25,
            cpu=2.0,
            mem=4.0,
            disk=8.0,
            gpu=0.0,
            region_ids=["us"],
        )
        self.assertEqual(
            {
                key: _sandbox_identity(sandbox)[key]
                for key in ("cpu", "memory_gib", "disk_gib", "gpu")
            },
            {"cpu": 2, "memory_gib": 4, "disk_gib": 8, "gpu": 0},
        )
        self.assertEqual(
            {
                key: _snapshot_identity(snapshot)[key]
                for key in (
                    "size_bytes", "cpu", "memory_gib", "disk_gib", "gpu",
                )
            },
            {
                "size_bytes": 16.25,
                "cpu": 2,
                "memory_gib": 4,
                "disk_gib": 8,
                "gpu": 0,
            },
        )

        invalid_values = {
            "bool": True,
            "nan": float("nan"),
            "infinite": float("inf"),
            "negative": -1,
            "fractional": 1.5,
            "nonnumeric": "2",
        }
        identity_fields = (
            (_sandbox_identity, sandbox, "cpu"),
            (_sandbox_identity, sandbox, "memory"),
            (_sandbox_identity, sandbox, "disk"),
            (_sandbox_identity, sandbox, "gpu"),
            (_snapshot_identity, snapshot, "cpu"),
            (_snapshot_identity, snapshot, "mem"),
            (_snapshot_identity, snapshot, "disk"),
            (_snapshot_identity, snapshot, "gpu"),
        )
        for identity, original, field in identity_fields:
            for name, invalid in invalid_values.items():
                with self.subTest(
                    identity=identity.__name__, field=field, value=name,
                ):
                    observed = copy.deepcopy(original)
                    setattr(observed, field, invalid)
                    with self.assertRaisesRegex(
                        ManagerError, "invalid Daytona SDK integer quantity",
                    ):
                        identity(observed)

        invalid_sizes = {
            "bool": True,
            "nan": float("nan"),
            "infinite": float("inf"),
            "negative": -1,
            "nonnumeric": "16.25",
        }
        for name, invalid in invalid_sizes.items():
            with self.subTest(snapshot_size=name):
                observed = copy.deepcopy(snapshot)
                observed.size = invalid
                with self.assertRaisesRegex(
                    ManagerError, "invalid Daytona SDK numeric quantity",
                ):
                    _snapshot_identity(observed)

    def test_quota_receipt_enforces_nullable_per_sandbox_maximums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            probe = self._quota_probe(manager)
            equal, unused_calls = self._call_quota_receipt(
                manager, probe, [self._usage_row()],
            )
            self.assertEqual(equal["max_cpu_per_sandbox"], 2.0)
            self.assertEqual(equal["max_memory_per_sandbox"], 4.0)
            self.assertEqual(equal["max_disk_per_sandbox"], 8.0)
            self.assertEqual(
                equal["per_sandbox_maximum_null_semantics"]["null_fields"],
                [],
            )

        below_cases = {
            "cpu": {"max_cpu_per_sandbox": 1},
            "memory": {"max_memory_per_sandbox": 3},
            "disk": {"max_disk_per_sandbox": 7},
        }
        for name, replacements in below_cases.items():
            with (
                self.subTest(name=f"below_{name}"),
                tempfile.TemporaryDirectory() as temporary,
            ):
                manager = self._manager(temporary)
                probe = self._quota_probe(manager)
                with self.assertRaisesRegex(
                    ManagerError, "below the exact probe resource",
                ):
                    self._call_quota_receipt(
                        manager, probe, [self._usage_row(**replacements)],
                    )

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            probe = self._quota_probe(manager)
            nullable, unused_calls = self._call_quota_receipt(
                manager,
                probe,
                [self._usage_row(
                    max_cpu_per_sandbox=None,
                    max_memory_per_sandbox=None,
                    max_disk_per_sandbox=None,
                )],
            )
            semantics = nullable["per_sandbox_maximum_null_semantics"]
            self.assertEqual(semantics["null_fields"], [
                "max_cpu_per_sandbox",
                "max_disk_per_sandbox",
                "max_memory_per_sandbox",
            ])
            self.assertFalse(semantics["null_means_unlimited"])
            self.assertTrue(semantics["exact_resource_probe_already_live"])
            self.assertEqual(semantics["exact_resource_probe_resources"], {
                "cpu": 2,
                "memory_gib": 4,
                "disk_gib": 8,
                "gpu": 0,
            })

        for field in (
            "max_cpu_per_sandbox",
            "max_memory_per_sandbox",
            "max_disk_per_sandbox",
        ):
            for name, invalid in (
                ("bool", True),
                ("nan", float("nan")),
                ("infinite", float("inf")),
            ):
                with (
                    self.subTest(field=field, value=name),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    manager = self._manager(temporary)
                    probe = self._quota_probe(manager)
                    with self.assertRaisesRegex(
                        ManagerError, "invalid live Daytona quota value",
                    ):
                        self._call_quota_receipt(
                            manager,
                            probe,
                            [self._usage_row(**{field: invalid})],
                        )

    def test_quota_receipt_uses_exact_probe_region_and_class_without_forbidden_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            probe = self._quota_probe(manager)
            exact = self._usage_row()
            rows = [
                self._usage_row(region_id="eu"),
                self._usage_row(sandbox_class="container"),
                exact,
            ]
            value, calls = self._call_quota_receipt(
                manager, probe, rows,
            )
            self.assertEqual(
                [organization_id for organization_id, unused_timeout in calls],
                [probe.organization_id],
            )
            self.assertGreater(calls[0][1], 0)
            self.assertLessEqual(
                calls[0][1], manager.config.api_timeout_seconds,
            )
            self.assertEqual(value["probe_sandbox_id"], probe.id)
            self.assertEqual(value["region_id"], probe.target)
            self.assertEqual(value["sandbox_class"], probe.sandbox_class)
            self.assertEqual(
                value["status"],
                "PASS_EXACT_REGION_SANDBOX_CLASS_FANOUT_FITS",
            )
            self.assertFalse(
                value["forbidden_sandbox_region_quota_endpoint_used"],
            )
            self.assertEqual(value["projected"], {
                "cpu": 200.0,
                "memory_gib": 400.0,
                "disk_gib": 800.0,
                "gpu": 0.0,
            })
            self.assertEqual(manager.journal.events[-1][0], "LIVE_QUOTA_PASS")

    def test_quota_receipt_rejects_missing_or_ambiguous_probe_identity_before_api_call(self) -> None:
        cases = {
            "organization": ("organization_id", ""),
            "sandbox_class_missing": ("sandbox_class", None),
            "sandbox_class_unknown": (
                "sandbox_class", "unknown_default_open_api",
            ),
            "region_missing": ("target", ""),
        }
        for name, (attribute, replacement) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                manager = self._manager(temporary)
                probe = self._quota_probe(manager)
                setattr(probe, attribute, replacement)
                with self.assertRaisesRegex(
                    ManagerError, "missing or ambiguous",
                ):
                    self._call_quota_receipt(
                        manager, probe, [self._usage_row()],
                    )

    def test_quota_receipt_rejects_probe_region_drift_before_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            probe = self._quota_probe(manager)
            probe.target = "eu"
            with self.assertRaisesRegex(ManagerError, "probe region drift"):
                self._call_quota_receipt(
                    manager, probe, [self._usage_row(region_id="eu")],
                )

    def test_quota_receipt_requires_one_exact_region_class_row(self) -> None:
        cases = {
            "zero": [self._usage_row(region_id="eu")],
            "duplicate": [self._usage_row(), self._usage_row()],
        }
        for name, rows in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                manager = self._manager(temporary)
                probe = self._quota_probe(manager)
                with self.assertRaisesRegex(
                    ManagerError, "unique live region usage row",
                ):
                    self._call_quota_receipt(manager, probe, rows)

    def test_quota_receipt_rejects_invalid_usage_numbers(self) -> None:
        cases = {
            "nan": {"current_cpu_usage": float("nan")},
            "infinite": {"total_cpu_quota": float("inf")},
            "negative": {"current_memory_usage": -1},
            "boolean": {"current_gpu_usage": False},
            "current_exceeds_total": {
                "current_disk_usage": 10,
                "total_disk_quota": 9,
            },
        }
        for name, replacements in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                manager = self._manager(temporary)
                probe = self._quota_probe(manager)
                row = self._usage_row()
                for attribute, replacement in replacements.items():
                    setattr(row, attribute, replacement)
                with self.assertRaises(ManagerError):
                    self._call_quota_receipt(manager, probe, [row])

    def test_quota_receipt_rejects_insufficient_fanout_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            probe = self._quota_probe(manager)
            row = self._usage_row(
                total_cpu=199, total_memory=399, total_disk=799,
            )
            with self.assertRaisesRegex(
                ManagerError, "cannot fit fanout",
            ):
                self._call_quota_receipt(manager, probe, [row])

    def test_quota_receipt_rejects_additional_worker_count_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            probe = self._quota_probe(manager)
            with self.assertRaisesRegex(
                ManagerError, "additional worker count drift",
            ):
                self._call_quota_receipt(
                    manager, probe, [self._usage_row()],
                    additional_workers=98,
                )

    def test_partial_committed_download_is_terminal_before_cloud_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            worker = manager.download_root / "worker-000"
            worker.mkdir()
            (worker / "CPU_SEARCH_RESULT.json").write_text("{}")
            with self.assertRaises(ManagerError):
                manager._validate_existing_output(0, None)
            self.assertEqual(
                manager.journal.events[-1][0],
                "PARTIAL_COMMITTED_OUTPUT_TERMINAL_LOCAL_CORRUPTION",
            )

    def test_validated_output_delete_failure_never_starts_second_worker_attempt(self) -> None:
        class Sandbox:
            id = "sandbox-one"

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            counts = {"create": 0, "command": 0, "delete": 0}
            value = {
                "result": {"sha256": "1" * 64, "size_bytes": 10},
                "flat": {"sha256": "2" * 64, "size_bytes": 20},
            }
            manager._validate_existing_output = lambda *unused: None

            def create(*unused):
                counts["create"] += 1
                return Sandbox()

            def command(*unused):
                counts["command"] += 1
                return "session", "command", 1

            def delete(*unused):
                counts["delete"] += 1
                raise ManagerError("injected exact-ID deletion timeout")

            manager._create_task_sandbox = create
            manager._ensure_session_command = command
            manager._wait_command = lambda *unused: None
            manager._download_and_validate = lambda *unused: value
            manager._delete_sandbox = delete
            with self.assertRaisesRegex(ManagerError, "model rerun forbidden"):
                manager._run_task(object(), 1, None)
            self.assertEqual(counts, {"create": 1, "command": 1, "delete": 1})

    def test_failed_attempt_delete_failure_blocks_next_worker_attempt(self) -> None:
        class Sandbox:
            id = "sandbox-one"

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            counts = {"create": 0, "command": 0, "delete": 0}
            manager._validate_existing_output = lambda *unused: None

            def create(*unused):
                counts["create"] += 1
                return Sandbox()

            def command(*unused):
                counts["command"] += 1
                return "session", "command", 1

            def delete(*unused):
                counts["delete"] += 1
                raise ManagerError("injected exact-ID deletion timeout")

            manager._create_task_sandbox = create
            manager._ensure_session_command = command
            manager._wait_command = lambda *unused: (_ for _ in ()).throw(
                ManagerError("transient worker failure")
            )
            manager._delete_sandbox = delete
            with self.assertRaisesRegex(ManagerError, "retry blocked"):
                manager._run_task(object(), 1, None)
            self.assertEqual(counts, {"create": 1, "command": 1, "delete": 1})

    @unittest.skip("legacy two-file recovery fixture is not valid for hybrid bundles")
    def test_post_atomic_rename_receipt_failure_recovers_without_model_rerun(self) -> None:
        class Sandbox:
            id = "sandbox-post-rename"

        class FailOnceJournal(FakeJournal):
            def __init__(self):
                super().__init__()
                self.failed = False

            def append(self, event: str, **fields):
                if event == "WORKER_OUTPUT_DOWNLOADED_AND_VALIDATED" and not self.failed:
                    self.failed = True
                    raise OSError("injected receipt failure after atomic rename")
                return super().append(event, **fields)

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            manager.journal = FailOnceJournal()
            counts = {"create": 0, "command": 0, "delete": 0}
            source_root = WAVE_ROOT / "downloads/worker-000"

            def create(*unused):
                counts["create"] += 1
                return Sandbox()

            def command(*unused):
                counts["command"] += 1
                return "session", "command", 1

            def download(unused_sandbox, remote_path, local_path, **unused):
                source = source_root / (
                    "CPU_SEARCH_RESULT.json"
                    if remote_path.endswith("CPU_SEARCH_RESULT.json")
                    else "trajectory_final_group_flat.f32le"
                )
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, local_path)
                return hash_file(local_path)

            def delete(*unused):
                counts["delete"] += 1

            manager._create_task_sandbox = create
            manager._ensure_session_command = command
            manager._wait_command = lambda *unused: None
            manager._download_remote = download
            manager._delete_sandbox = delete
            result = manager._run_task(object(), 0, None)
            self.assertEqual(
                result["result"]["sha256"],
                "29e982cdadbd0f1d2d79d06512b7a44feac5cb23f204df6f76311959777cfbfe",
            )
            self.assertEqual(counts, {"create": 1, "command": 1, "delete": 1})
            self.assertTrue(
                (manager.download_root / "worker-000/trajectory_final_group_flat.f32le").is_file()
            )

    @unittest.skip("legacy two-file recovery fixture is not valid for hybrid bundles")
    def test_flat_download_failure_leaves_fixed_root_absent(self) -> None:
        class Sandbox:
            id = "sandbox-partial-staging"

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            source_result = WAVE_ROOT / "downloads/worker-000/CPU_SEARCH_RESULT.json"

            def download(unused_sandbox, remote_path, local_path, **unused):
                if remote_path.endswith("CPU_SEARCH_RESULT.json"):
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_result, local_path)
                    return hash_file(local_path)
                raise ManagerError("injected flat download interruption")

            manager._download_remote = download
            with self.assertRaisesRegex(ManagerError, "flat download interruption"):
                manager._download_and_validate(Sandbox(), 0, 0, None)
            self.assertFalse((manager.download_root / "worker-000").exists())

    def test_recovery_precloud_gate_reuses_sealed_pass_with_zero_cloud_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            provenance_identity = {"sha256": "a" * 64, "size_bytes": 123}
            source_id = "source-task0-sandbox"
            failed_probe_id = "failed-probe-sandbox"
            immediate_probe_id = "immediate-probe-sandbox"
            rows = self._recovery_rows(
                {"id": source_id},
                {"id": failed_probe_id},
                {"id": immediate_probe_id},
            )
            manager.recovered_task0 = {
                "identity": provenance_identity,
                "required_exact_id_reconciliation": rows,
            }
            terminal = {
                "schema": "hfw_w560_recovery_precloud_zero_resource_gate_v3",
                "status": (
                    "PASS_ALL_REQUIRED_EXACT_IDS_AND_CAMPAIGN_RESOURCES_ABSENT"
                ),
                "campaign_sha256": manager.campaign_sha256,
                "snapshot_manifest_sha256": manager.snapshot_manifest_sha256,
                "wave_index": manager.wave_index,
                "recovery_provenance": provenance_identity,
                "required_exact_id_count": 3,
                "required_exact_sandbox_ids": [
                    source_id, failed_probe_id, immediate_probe_id,
                ],
                "exact_id_reconciliations": [
                    {
                        "role": row["role"],
                        "sandbox_id": row["sandbox"]["id"],
                        "exact_id_not_found_confirmed": True,
                        "delete_required": False,
                    }
                    for row in rows
                ],
                "all_required_exact_ids_not_found_confirmed": True,
                "exact_campaign_label_relist_passed": True,
                "final_campaign_sandboxes_remaining": 0,
                "snapshot_exact_name_not_found_confirmed": True,
                "snapshot_present": False,
                "sandbox_snapshot_or_probe_create_before_gate": False,
                "task0_model_command_started": False,
                "checkpoint_or_head_writes": 0,
            }
            write_exclusive(
                manager.config.wave_root / "RECOVERY_PRECLOUD_ZERO_RESOURCE_COMPLETE.json",
                canonical_json_bytes(terminal),
            )
            manager._client = lambda: self.fail("sealed gate attempted a cloud call")
            observed = manager._recovery_precloud_zero_resource_gate()
            self.assertEqual(observed, terminal)
            self.assertEqual(
                manager.journal.events[-1][0],
                "RECOVERY_PRECLOUD_ZERO_RESOURCE_GATE_REUSED_NO_CLOUD_CALLS",
            )

    def test_recovered_source_exact_id_transient_error_never_implies_absence(self) -> None:
        from daytona import DaytonaNotFoundError

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            manager.config = replace(
                manager.config, maximum_attempts=3, retry_initial_seconds=0,
                retry_maximum_seconds=0,
            )
            source_id = "known-source-task0-id"
            source_sandbox = {
                "id": source_id,
                "labels": _task_labels(
                    manager.campaign_sha256, manager.wave_index, 0, 0,
                ),
            }
            manager.recovered_task0 = {
                "required_exact_id_reconciliation": self._recovery_rows(
                    source_sandbox,
                    {"id": "earlier-probe"},
                    {"id": "immediate-probe"},
                ),
            }
            observations = iter((
                RuntimeError("transient exact-get failure"),
                DaytonaNotFoundError("now proven absent"),
            ))

            class Client:
                def get(self, unused_sandbox_id):
                    value = next(observations)
                    if isinstance(value, BaseException):
                        raise value
                    return value

            manager._client = lambda: Client()
            result = manager._reconcile_recovered_source_task0_exact_id()
            self.assertTrue(result["exact_id_not_found_confirmed"])
            events = [event for event, unused in manager.journal.events]
            self.assertEqual(events[0], "SDK_CALL_RETRY")
            self.assertEqual(events[-1], "RECOVERED_PREDECESSOR_EXACT_ID_ABSENT")

    def test_recovered_failed_probe_transient_error_never_implies_absence(self) -> None:
        from daytona import DaytonaNotFoundError

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            manager.config = replace(
                manager.config, maximum_attempts=3, retry_initial_seconds=0,
                retry_maximum_seconds=0,
            )
            failed_probe_id = "known-failed-probe-id"
            manager.recovered_task0 = {
                "required_exact_id_reconciliation": self._recovery_rows(
                    {"id": "source-task0"},
                    {"id": failed_probe_id},
                    {"id": "immediate-probe"},
                ),
            }
            observations = iter((
                RuntimeError("transient exact-get failure"),
                DaytonaNotFoundError("now proven absent"),
            ))

            class Client:
                def get(self, unused_sandbox_id):
                    value = next(observations)
                    if isinstance(value, BaseException):
                        raise value
                    return value

            manager._client = lambda: Client()
            result = manager._reconcile_recovered_failed_probe_exact_id()
            self.assertEqual(result["sandbox_id"], failed_probe_id)
            self.assertTrue(result["exact_id_not_found_confirmed"])
            events = [event for event, unused in manager.journal.events]
            self.assertEqual(events[0], "SDK_CALL_RETRY")
            self.assertEqual(events[-1], "RECOVERED_PREDECESSOR_EXACT_ID_ABSENT")

    def test_recovery_precloud_gate_rejects_non_distinct_exact_ids_before_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            repeated_id = "same-sandbox-id"
            manager.recovered_task0 = {
                "identity": {"sha256": "e" * 64, "size_bytes": 1},
                "required_exact_id_reconciliation": self._recovery_rows(
                    {"id": repeated_id},
                    {"id": repeated_id},
                    {"id": "third-id"},
                ),
            }
            manager._client = lambda: self.fail("non-distinct IDs reached cloud")
            with self.assertRaisesRegex(ManagerError, "not distinct"):
                manager._recovery_precloud_zero_resource_gate()

    def test_live_failed_probe_additive_service_label_passes_and_is_journaled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            required = _probe_labels(manager.campaign_sha256, manager.wave_index)
            sealed_labels = {**required, "code-toolbox-language": "python"}
            unused_snapshot, sealed = self._probe_fixture(manager, sealed_labels)
            unused_snapshot, live = self._probe_fixture(
                manager,
                {**sealed_labels, "daytona-region": manager.config.target},
            )
            sealed_sandbox = {
                "id": sealed.id,
                "name": sealed.name,
                "snapshot": sealed.snapshot,
                "target": sealed.target,
                "cpu": sealed.cpu,
                "memory_gib": sealed.memory,
                "disk_gib": sealed.disk,
                "gpu": sealed.gpu,
                "network_block_all": sealed.network_block_all,
                "public": sealed.public,
                "auto_stop_interval": sealed.auto_stop_interval,
                "auto_delete_interval": sealed.auto_delete_interval,
                "labels": dict(sealed.labels),
            }
            manager.recovered_task0 = {
                "required_exact_id_reconciliation": self._recovery_rows(
                    {"id": "source-task0"},
                    sealed_sandbox,
                    {"id": "immediate-probe"},
                ),
            }

            class Client:
                def get(self, sandbox_id):
                    return live

            manager._client = lambda: Client()
            deleted = []
            manager._delete_sandbox = lambda sandbox, reason: deleted.append(
                (sandbox, reason)
            )
            result = manager._reconcile_recovered_failed_probe_exact_id()
            self.assertTrue(result["exact_id_not_found_confirmed"])
            self.assertEqual(len(deleted), 1)
            event, fields = manager.journal.events[-1]
            self.assertEqual(event, "RECOVERED_PREDECESSOR_EXACT_ID_RECONCILED")
            self.assertEqual(
                fields["observed_sandbox"]["labels"]["daytona-region"],
                manager.config.target,
            )

    def test_probe_platform_label_passes_and_full_observed_map_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            required = _probe_labels(manager.campaign_sha256, manager.wave_index)
            observed = {**required, "code-toolbox-language": "python"}
            snapshot, sandbox = self._probe_fixture(manager, observed)
            manager._find_probe_sandboxes = lambda: [sandbox]
            recovered = manager._create_probe_sandbox(snapshot)
            self.assertIs(recovered, sandbox)
            event, fields = manager.journal.events[-1]
            self.assertEqual(event, "PROBE_SANDBOX_RECOVERED")
            self.assertEqual(fields["sandbox"]["labels"], observed)

    def test_worker_platform_labels_pass_but_unexpected_hfw_labels_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)

            class Snapshot:
                id = "snapshot"
                name = "snapshot"

            sandbox = FakeSandbox(manager.campaign_sha256)
            sandbox.snapshot = Snapshot.name
            sandbox.env = dict(WORKER_ENVIRONMENT)
            sandbox.labels = {
                **sandbox.labels,
                "code-toolbox-language": "python",
                "daytona-region": manager.config.target,
            }
            manager._validate_sandbox(sandbox, 1, 0, Snapshot())
            sandbox.labels["hfw_probe_generation"] = "unexpected"
            with self.assertRaisesRegex(ManagerError, "sandbox labels drift"):
                manager._validate_sandbox(sandbox, 1, 0, Snapshot())

    def test_probe_missing_or_changed_required_owner_label_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            required = _probe_labels(manager.campaign_sha256, manager.wave_index)
            for name, labels in (
                ("missing", {key: value for key, value in required.items() if key != "hfw_role"}),
                ("changed", {**required, "hfw_campaign_sha256": "b" * 64}),
            ):
                with self.subTest(name=name):
                    snapshot, sandbox = self._probe_fixture(manager, labels)
                    with self.assertRaisesRegex(ManagerError, "probe labels drift"):
                        manager._validate_probe_sandbox(sandbox, snapshot)

    def test_probe_unexpected_hfw_ownership_label_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            required = _probe_labels(manager.campaign_sha256, manager.wave_index)
            observed = {**required, "hfw_task_index": "000"}
            self.assertFalse(_contains_probe_labels(observed, required))
            snapshot, sandbox = self._probe_fixture(manager, observed)
            with self.assertRaisesRegex(ManagerError, "probe labels drift"):
                manager._validate_probe_sandbox(sandbox, snapshot)

    def test_predecessor_failure_provenance_rejects_bound_evidence_tamper(self) -> None:
        provenance = json.loads(
            (
                ATTEMPT8_PACKAGE_ROOT
                / "runtime/config/RECOVERED_TASK0_PROVENANCE.json"
            ).read_text()
        )
        evidence = provenance["predecessor_recovery_failure"]
        binding = provenance["immutable_binding"]
        source_sandbox = provenance["source"]["task0_sandbox"]
        mutations = {
            "failure": lambda value: value["receipt_chain"][10].__setitem__(
                "error", "different failure",
            ),
            "probe_identity": lambda value: value[
                "failed_probe_sandbox"
            ]["labels"].__setitem__("hfw_task_index", "000"),
            "cleanup": lambda value: value["receipt_chain"][19].__setitem__(
                "final_remaining_count", 1,
            ),
            "precloud": lambda value: value[
                "precloud_zero_resource_terminal"
            ]["value"].__setitem__(
                "source_task0_sandbox_id", "different-source-id",
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                tampered = copy.deepcopy(evidence)
                mutate(tampered)
                with self.assertRaises(ManagerError):
                    _validate_predecessor_recovery_failure(
                        tampered,
                        campaign_sha256=binding["campaign_sha256"],
                        snapshot_manifest_sha256=binding[
                            "snapshot_manifest_sha256"
                        ],
                        wave_index=1,
                        original_task0_sandbox=source_sandbox,
                    )

    @unittest.skip("fresh hybrid campaign does not import legacy task0 output")
    def test_immediate_predecessor_binds_exact_quota_failure_chain(self) -> None:
        provenance = json.loads(
            (
                ATTEMPT8_PACKAGE_ROOT
                / "runtime/config/RECOVERED_TASK0_PROVENANCE.json"
            ).read_text()
        )
        binding = provenance["immutable_binding"]
        source_sandbox = provenance["source"]["task0_sandbox"]
        earlier_probe = provenance[
            "predecessor_recovery_failure"
        ]["failed_probe_sandbox"]
        evidence = _immediate_predecessor_quota_failure_value(
            ATTEMPT8_PACKAGE_ROOT,
            campaign_sha256=binding["campaign_sha256"],
            snapshot_manifest_sha256=binding["snapshot_manifest_sha256"],
            wave_index=1,
            original_task0_sandbox=source_sandbox,
            earlier_failed_probe_sandbox=earlier_probe,
        )
        self.assertEqual(
            evidence["terminal_absence"],
            _immediate_predecessor_terminal_absence_value(
                ATTEMPT8_PACKAGE_ROOT, wave_index=1,
            ),
        )
        self.assertEqual(evidence["receipt_chain_length"], 25)
        self.assertEqual(
            evidence["failed_probe_sandbox"]["id"],
            "e2238232-07c1-4cd9-ab04-c0e2cbc99fab",
        )
        self.assertEqual(
            evidence["receipt_chain"][20]["error"],
            "read live region quota exhausted retries: "
            + evidence["receipt_chain"][14]["error"],
        )
        self.assertEqual(
            evidence["terminal_absence"][
                "local_authority_checkpoint_file_count"
            ],
            0,
        )

        mutations = {
            "retry": lambda value: value["receipt_chain"][14].__setitem__(
                "error", "different 403",
            ),
            "failure": lambda value: value.__setitem__(
                "failure_receipt", {"sha256": "f" * 64, "size_bytes": 1},
            ),
            "probe": lambda value: value["failed_probe_sandbox"][
                "labels"
            ].__setitem__("hfw_task_index", "000"),
            "cleanup": lambda value: value["receipt_chain"][24].__setitem__(
                "final_remaining_count", 1,
            ),
            "build_evidence": lambda value: value["build_evidence"].__setitem__(
                "pip_freeze_lines", [],
            ),
            "terminal_absence": lambda value: value["terminal_absence"].__setitem__(
                "local_authority_checkpoint_file_count", 1,
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                tampered = copy.deepcopy(evidence)
                mutate(tampered)
                with self.assertRaises(ManagerError):
                    _validate_immediate_predecessor_quota_failure(
                        tampered,
                        campaign_sha256=binding["campaign_sha256"],
                        snapshot_manifest_sha256=binding[
                            "snapshot_manifest_sha256"
                        ],
                        wave_index=1,
                        original_task0_sandbox=source_sandbox,
                        earlier_failed_probe_sandbox=earlier_probe,
                    )

    @unittest.skip("fresh hybrid campaign rejects legacy two-file recovery provenance")
    def test_v2_recovery_is_only_accepted_for_explicit_external_validation(self) -> None:
        with self.assertRaisesRegex(ManagerError, "schema/status drift"):
            validate_recovered_task0_provenance(
                ATTEMPT8_PACKAGE_ROOT, wave_index=1,
            )
        observed = validate_recovered_task0_provenance(
            ATTEMPT8_PACKAGE_ROOT,
            wave_index=1,
            _allow_external_v2=True,
        )
        self.assertIsNotNone(observed)
        assert observed is not None
        self.assertEqual(
            observed["schema"], "hfw_w560_recovered_task0_provenance_v2",
        )

    @unittest.skip("fresh hybrid campaign has no recovered task0 provenance")
    def test_v3_provenance_exposes_unique_top_level_three_id_contract(self) -> None:
        attempt8 = json.loads(
            (
                ATTEMPT8_PACKAGE_ROOT
                / "runtime/config/RECOVERED_TASK0_PROVENANCE.json"
            ).read_text()
        )
        value = _recovered_task0_provenance_value(
            Path(attempt8["source"]["package_root"]),
            Path(
                attempt8["predecessor_recovery_failure"]["package_root"]
            ),
            ATTEMPT8_PACKAGE_ROOT,
            PACKAGE_ROOT,
            wave_index=1,
            recovered_unix_ns=1,
        )
        rows = value["required_exact_id_reconciliation"]
        self.assertEqual(
            value["schema"], "hfw_w560_recovered_task0_provenance_v3",
        )
        self.assertEqual(
            [row["role"] for row in rows],
            [
                "ORIGINAL_SOURCE_TASK0",
                "PREDECESSOR_LABEL_DRIFT_PROBE",
                "IMMEDIATE_PREDECESSOR_QUOTA_API_PROBE",
            ],
        )
        self.assertEqual(len({row["sandbox"]["id"] for row in rows}), 3)
        self.assertEqual(
            value["immediate_predecessor_recovery_failure"][
                "receipt_chain_length"
            ],
            25,
        )

    def test_predecessor_terminal_absence_rejects_coexisting_post_probe_artifacts(self) -> None:
        forbidden = (
            "LAUNCH_MANIFEST.json",
            "CLOUD_DOWNLOADS_COMPLETE.json",
            "CLOUD_CLEANUP_COMPLETE.json",
            "REAL_PIPELINE_DETERMINISTIC_FAILURE.json",
            "COLLECTION_COMPLETE.json",
            "PROMOTION_COMPLETE.json",
        )
        for relative in forbidden:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                package = Path(temporary)
                wave = package / "runtime/waves/wave-00000001"
                heads = package / "runtime/authority/lineage/heads"
                checkpoints = package / "runtime/authority/checkpoints"
                wave.mkdir(parents=True)
                heads.mkdir(parents=True)
                checkpoints.mkdir(parents=True)
                (heads / ("00000000-" + "a" * 64 + ".head.json")).write_text("{}")
                (checkpoints / ("b" * 64 + ".pt")).write_bytes(b"checkpoint")
                marker = wave / relative
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}")
                with self.assertRaisesRegex(ManagerError, "unexpectedly exists"):
                    _predecessor_terminal_absence_value(package, wave_index=1)

        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            wave = package / "runtime/waves/wave-00000001"
            heads = package / "runtime/authority/lineage/heads"
            checkpoints = package / "runtime/authority/checkpoints"
            shortlist = package / "runtime/authority/lineage/shortlists"
            wave.mkdir(parents=True)
            heads.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            shortlist.mkdir(parents=True)
            (heads / ("00000000-" + "a" * 64 + ".head.json")).write_text("{}")
            (checkpoints / ("b" * 64 + ".pt")).write_bytes(b"checkpoint")
            (shortlist / ("c" * 64 + ".shortlist.json")).write_text("{}")
            with self.assertRaisesRegex(ManagerError, "lineage artifacts"):
                _predecessor_terminal_absence_value(package, wave_index=1)

    def test_recovered_sandbox_required_labels_allow_sealed_platform_extra(self) -> None:
        required = _task_labels("a" * 64, 1, 0, 0)
        observed = {**required, "code-toolbox-language": "python"}
        self.assertTrue(_contains_required_labels(observed, required))
        self.assertEqual(observed["code-toolbox-language"], "python")

    def test_recovered_sandbox_required_labels_reject_missing_owner_label(self) -> None:
        required = _task_labels("a" * 64, 1, 0, 0)
        observed = {**required, "code-toolbox-language": "python"}
        del observed["hfw_role"]
        self.assertFalse(_contains_required_labels(observed, required))

    def test_recovered_sandbox_required_labels_reject_tampered_campaign(self) -> None:
        required = _task_labels("a" * 64, 1, 0, 0)
        observed = {
            **required,
            "hfw_campaign_sha256": "b" * 64,
            "code-toolbox-language": "python",
        }
        self.assertFalse(_contains_required_labels(observed, required))

    def test_recovery_precloud_gate_proves_all_three_ids_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            provenance_identity = {"sha256": "b" * 64, "size_bytes": 321}
            source_id = "source-task0-sandbox"
            failed_probe_id = "failed-probe-sandbox"
            immediate_probe_id = "immediate-probe-sandbox"
            rows = self._recovery_rows(
                {"id": source_id},
                {"id": failed_probe_id},
                {"id": immediate_probe_id},
            )
            manager.recovered_task0 = {
                "identity": provenance_identity,
                "required_exact_id_reconciliation": rows,
            }
            calls = []

            def exact_id(sandbox, *, role):
                calls.append((role, sandbox["id"]))
                return {
                    "role": role,
                    "sandbox_id": sandbox["id"],
                    "exact_id_not_found_confirmed": True,
                    "delete_required": False,
                }

            def cleanup(*unused):
                calls.append(("cleanup", None))
                manager._last_cleanup_evidence = {
                    "initial_campaign_sandbox_count": 0,
                    "unique_campaign_sandboxes_discovered": 0,
                    "unique_campaign_sandbox_ids_discovered": [],
                    "exact_id_not_found_confirmation_count": 0,
                    "final_remaining_count": 0,
                    "final_remaining_ids": [],
                    "final_relist_succeeded": True,
                    "snapshot_present": False,
                    "snapshot_relist_succeeded": True,
                    "exact_campaign_label_relist_passed": True,
                    "snapshot_exact_name_not_found_confirmed": True,
                }
                return []

            manager._reconcile_recovered_exact_sandbox_id = exact_id
            manager._cleanup = cleanup
            observed = manager._recovery_precloud_zero_resource_gate()
            self.assertEqual(
                calls,
                [
                    ("ORIGINAL_SOURCE_TASK0", source_id),
                    ("PREDECESSOR_LABEL_DRIFT_PROBE", failed_probe_id),
                    (
                        "IMMEDIATE_PREDECESSOR_QUOTA_API_PROBE",
                        immediate_probe_id,
                    ),
                    ("cleanup", None),
                ],
            )
            self.assertEqual(
                observed["required_exact_sandbox_ids"],
                [source_id, failed_probe_id, immediate_probe_id],
            )
            self.assertEqual(observed["required_exact_id_count"], 3)
            self.assertEqual(observed["final_campaign_sandboxes_remaining"], 0)
            self.assertFalse(observed["sandbox_snapshot_or_probe_create_before_gate"])

    def test_recovery_precloud_gate_exact_gets_all_three_ids_before_cleanup(self) -> None:
        from daytona import DaytonaNotFoundError

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            source_id = "source-task0-sandbox"
            failed_probe_id = "failed-probe-sandbox"
            immediate_probe_id = "immediate-probe-sandbox"
            manager.recovered_task0 = {
                "identity": {"sha256": "f" * 64, "size_bytes": 444},
                "required_exact_id_reconciliation": self._recovery_rows(
                    {"id": source_id},
                    {"id": failed_probe_id},
                    {"id": immediate_probe_id},
                ),
            }
            calls = []

            class Client:
                def get(self, sandbox_id):
                    calls.append(("exact-get", sandbox_id))
                    raise DaytonaNotFoundError("proven absent")

            manager._client = lambda: Client()

            def cleanup(*unused):
                calls.append(("campaign-and-snapshot-cleanup", None))
                manager._last_cleanup_evidence = {
                    "initial_campaign_sandbox_count": 0,
                    "unique_campaign_sandboxes_discovered": 0,
                    "unique_campaign_sandbox_ids_discovered": [],
                    "exact_id_not_found_confirmation_count": 0,
                    "final_remaining_count": 0,
                    "final_remaining_ids": [],
                    "final_relist_succeeded": True,
                    "snapshot_present": False,
                    "snapshot_relist_succeeded": True,
                    "exact_campaign_label_relist_passed": True,
                    "snapshot_exact_name_not_found_confirmed": True,
                }
                return []

            manager._cleanup = cleanup
            observed = manager._recovery_precloud_zero_resource_gate()
            self.assertEqual(calls, [
                ("exact-get", source_id),
                ("exact-get", failed_probe_id),
                ("exact-get", immediate_probe_id),
                ("campaign-and-snapshot-cleanup", None),
            ])
            self.assertTrue(
                observed["all_required_exact_ids_not_found_confirmed"]
            )

    def test_run_locked_executes_fresh_recovery_gate_before_snapshot_ensure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            manager.recovered_task0 = {
                "identity": {"sha256": "c" * 64, "size_bytes": 1},
            }
            order = []
            manager._existing_download_terminal = lambda: None
            manager._read_existing_launch_manifest = lambda **unused: None
            manager._recovery_precloud_zero_resource_gate = lambda: order.append("gate")

            def ensure_snapshot():
                self.assertEqual(order, ["gate"])
                order.append("snapshot")
                raise ManagerError("stop after ordering assertion")

            manager._ensure_snapshot = ensure_snapshot
            manager._cleanup = lambda *unused: []
            with self.assertRaisesRegex(ManagerError, "ordering assertion"):
                manager._run_locked()
            self.assertEqual(order, ["gate", "snapshot"])

    def test_run_locked_existing_launch_skips_destructive_recovery_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            manager.recovered_task0 = {
                "identity": {"sha256": "d" * 64, "size_bytes": 1},
            }
            manager._existing_download_terminal = lambda: None
            manager._read_existing_launch_manifest = lambda **unused: {
                "sealed_worker_environment": {},
            }
            manager._recovery_precloud_zero_resource_gate = lambda: self.fail(
                "partial-fanout restart ran destructive precloud gate"
            )
            manager._ensure_snapshot = lambda: (_ for _ in ()).throw(
                ManagerError("stop after skip assertion")
            )
            manager._cleanup = lambda *unused: []
            with self.assertRaisesRegex(ManagerError, "skip assertion"):
                manager._run_locked()

    def test_execute_timeout_after_remote_accept_recovers_one_command(self) -> None:
        class Command:
            id = "accepted-command"
            command = None

        class Session:
            commands = []

        class Process:
            def __init__(self, expected):
                self.expected = expected
                self.session = Session()
                self.execute_calls = 0

            def get_session(self, session_id):
                return self.session

            def execute_session_command(self, session_id, request, timeout=None):
                self.execute_calls += 1
                command = Command()
                command.command = request.command
                self.session.commands = [command]
                raise RuntimeError("injected timeout after remote accept")

        class Sandbox:
            id = "sandbox-id"

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            sandbox = Sandbox()
            sandbox.process = Process(None)
            session_id, command_id, _ = manager._ensure_session_command(sandbox, 0, 0)
            self.assertEqual(session_id, "w560-000-a0")
            self.assertEqual(command_id, "accepted-command")
            self.assertEqual(sandbox.process.execute_calls, 1)

    def test_existing_launch_manifest_is_read_without_dynamic_recompute(self) -> None:
        class Snapshot:
            id = "snapshot-id"
            name = "snapshot-name"
            state = "active"
            size = 1
            cpu = 2
            mem = 4
            disk = 8
            gpu = 0
            region_ids = ["us"]

        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            manager.config = replace(manager.config, base_image=BASE_IMAGE)
            environment = manager.campaign["proposal_lane"]["sealed_environment"]
            seed = {
                "result": {"sha256": "1" * 64, "size_bytes": 10},
                "flat": {"sha256": "2" * 64, "size_bytes": 20},
                "bundle": {"sha256": "3" * 64, "size_bytes": 20},
                "accepted_candidate_count": 1,
                "peak_rss_bytes": 30,
                "base_full_search_mean_ce": 6.0,
                "terminal_full_search_mean_ce": 5.5,
                "self_calibrated_full_search_improvement": 0.5,
                "acceptance_gate_attempts": 1,
                "acceptance_gate_passes": 1,
                "accepted_updates": 1,
                "total_physical_model_forwards": int(
                    manager.campaign["solver"]["total_physical_model_forwards_minimum"]
                ),
                "worker_total_seconds": 10.0,
                "environment": dict(environment),
            }
            manager._run_config = {
                "campaign_deadline_unix_ns": __import__("time").time_ns()
                + 2_000_000_000_000,
            }
            deadline = manager._fanout_deadline_safety_receipt(seed)
            class Probe:
                id = "probe-id"
                organization_id = "organization-id"
                name = _probe_name(
                    manager.campaign_sha256, manager.wave_index,
                )
                state = "started"
                desired_state = "started"
                snapshot = Snapshot.name
                target = manager.config.target
                sandbox_class = "linux-vm"
                cpu = 2
                memory = 4
                disk = 8
                gpu = 0
                network_block_all = True
                public = False
                auto_stop_interval = 31
                auto_delete_interval = 0
                labels = {
                    **_probe_labels(
                        manager.campaign_sha256, manager.wave_index,
                    ),
                    "code-toolbox-language": "python",
                }

                def delete(self, timeout=None):
                    return None

            probe = Probe()
            probe_identity = _probe_launch_identity(probe)
            quota, quota_calls = self._call_quota_receipt(
                manager, probe, [self._usage_row()],
            )
            self.assertEqual(
                [
                    organization_id
                    for organization_id, unused_timeout in quota_calls
                ],
                [probe.organization_id],
            )

            class Client:
                def get(self, sandbox_id):
                    from daytona import DaytonaNotFoundError
                    raise DaytonaNotFoundError(
                        f"{sandbox_id} is exactly absent"
                    )

            manager._client = lambda: Client()
            deletion_proof = manager._delete_sandbox(
                probe, "BUILD_EVIDENCE_AND_QUOTA_PROBE_COMPLETE",
            )
            deletion_records = [
                record for record in manager.journal.records
                if isinstance(record.get("sandbox"), dict)
                and record.get("event") in {
                    "SANDBOX_DELETE_INTENT",
                    "SANDBOX_DELETE_ACCEPTED",
                    "SANDBOX_DELETED",
                }
            ]
            self.assertEqual(len(deletion_records), 3)
            for record in deletion_records:
                self.assertNotIn("organization_id", record["sandbox"])
                self.assertNotIn("sandbox_class", record["sandbox"])
            manager._launch_manifest(
                Snapshot(), seed,
                {"wheels": [], "dynamic_marker": "sealed"},
                quota,
                deadline,
                probe_identity,
                deletion_proof,
            )
            recovered = manager._read_existing_launch_manifest(
                snapshot=Snapshot(), seed_result=seed,
            )
            self.assertEqual(
                recovered["live_quota_before_fanout"]["schema"],
                "hfw_w560_daytona_live_quota_before_fanout_v3",
            )

            manifest_path = manager.config.wave_root / "LAUNCH_MANIFEST.json"
            sealed_manifest = json.loads(manifest_path.read_text())
            tamper_cases = {
                "full_probe_identity": lambda value: value[
                    "build_evidence_quota_probe"
                ]["sandbox"].__setitem__("desired_state", "tampered"),
                "probe_organization_identity": lambda value: value[
                    "build_evidence_quota_probe"
                ]["sandbox"].__setitem__("organization_id", "tampered"),
                "probe_sandbox_class_identity": lambda value: value[
                    "build_evidence_quota_probe"
                ]["sandbox"].__setitem__("sandbox_class", "container"),
                "deletion_receipt_identity": lambda value: value[
                    "build_evidence_quota_probe"
                ]["exact_id_deletion_proof"]["receipt_identities"][
                    "exact_id_absent"
                ].__setitem__("sha256", "0" * 64),
                "quota_probe_binding": lambda value: value[
                    "live_quota_before_fanout"
                ].__setitem__("probe_sandbox_id", "tampered"),
                "quota_region_binding": lambda value: value[
                    "live_quota_before_fanout"
                ].__setitem__("region_id", "eu"),
                "quota_organization_binding": lambda value: value[
                    "live_quota_before_fanout"
                ].__setitem__("organization_id", "tampered"),
                "quota_sandbox_class_binding": lambda value: value[
                    "live_quota_before_fanout"
                ].__setitem__("sandbox_class", "container"),
                "quota_forbidden_endpoint": lambda value: value[
                    "live_quota_before_fanout"
                ].__setitem__(
                    "forbidden_sandbox_region_quota_endpoint_used", True,
                ),
                "quota_projection": lambda value: value[
                    "live_quota_before_fanout"
                ]["projected"].__setitem__("cpu", 0.0),
                "quota_request_timeout": lambda value: value[
                    "live_quota_before_fanout"
                ].__setitem__(
                    "organization_usage_request_timeout_seconds", 0,
                ),
                "quota_per_sandbox_maximum": lambda value: value[
                    "live_quota_before_fanout"
                ].__setitem__("max_cpu_per_sandbox", 1),
                "quota_null_semantics": lambda value: value[
                    "live_quota_before_fanout"
                ]["per_sandbox_maximum_null_semantics"].__setitem__(
                    "null_means_unlimited", True,
                ),
            }
            for name, mutate in tamper_cases.items():
                with self.subTest(name=name):
                    tampered = copy.deepcopy(sealed_manifest)
                    mutate(tampered)
                    manifest_path.chmod(0o600)
                    manifest_path.write_bytes(canonical_json_bytes(tampered))
                    with self.assertRaises(ManagerError):
                        manager._read_existing_launch_manifest(
                            snapshot=Snapshot(), seed_result=seed,
                        )
                    manifest_path.write_bytes(
                        canonical_json_bytes(sealed_manifest)
                    )
                    manifest_path.chmod(0o400)


if __name__ == "__main__":
    unittest.main()
