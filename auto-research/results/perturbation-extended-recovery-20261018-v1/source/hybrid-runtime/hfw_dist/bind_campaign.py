from __future__ import annotations

import argparse
from copy import deepcopy
import errno
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Mapping

import torch

from .common import (
    canonical_json_bytes, fsync_directory, hash_file, read_json, relative_inventory,
    sha256_bytes, write_json_exclusive,
)
from .ledger import DistributedLedger


WORKER_RUNTIME_FILES = (
    "__init__.py", "common.py", "model_io.py", "cpu_proposal_worker.py",
)

_HARDLINK_FALLBACK_ERRNOS = frozenset({
    errno.EXDEV,
    errno.EACCES,
    errno.EMLINK,
    errno.ENOSYS,
    errno.EPERM,
    *(value for value in (
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    ) if value is not None),
})


def _hardlink_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def _validated_checkpoint(
    path: Path, expected_sha256: str, expected_size_bytes: int,
) -> Mapping[str, object]:
    identity = hash_file(path, expected_sha256)
    if int(identity["size_bytes"]) != int(expected_size_bytes):
        raise RuntimeError(f"checkpoint size drift: {path}")
    return identity


def _atomic_copy_checkpoint_create_only(
    source: Path, destination: Path, *, expected_sha256: str,
    expected_size_bytes: int,
) -> Mapping[str, object]:
    """Copy through a durable staging inode, then publish without replacement."""
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent,
    )
    staging = Path(staging_name)
    destination_created = False
    cleanup_changed_directory = False
    try:
        source_descriptor = os.open(
            source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            with os.fdopen(source_descriptor, "rb") as source_handle:
                source_descriptor = -1
                with os.fdopen(descriptor, "wb") as destination_handle:
                    descriptor = -1
                    shutil.copyfileobj(
                        source_handle, destination_handle,
                        length=1024 * 1024,
                    )
                    destination_handle.flush()
                    os.fchmod(destination_handle.fileno(), 0o444)
                    os.fsync(destination_handle.fileno())
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)
        staged_identity = _validated_checkpoint(
            staging, expected_sha256, expected_size_bytes,
        )
        # A same-directory staging hardlink is an atomic, create-only publish:
        # unlike replace(), it cannot overwrite a checkpoint from another run.
        os.link(staging, destination)
        destination_created = True
        installed_identity = _validated_checkpoint(
            destination, expected_sha256, expected_size_bytes,
        )
        if (
            installed_identity["device"], installed_identity["inode"]
        ) != (staged_identity["device"], staged_identity["inode"]):
            raise RuntimeError("atomically copied checkpoint inode drift")
        staging.unlink()
        fsync_directory(destination.parent)
        return installed_identity
    except BaseException:
        if destination_created:
            destination.unlink(missing_ok=True)
            cleanup_changed_directory = True
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if staging.exists():
            staging.unlink()
            cleanup_changed_directory = True
        if cleanup_changed_directory:
            fsync_directory(destination.parent)


def _import_checkpoint_create_only(
    source: Path, destination: Path, *, expected_sha256: str,
    expected_size_bytes: int,
) -> Mapping[str, object]:
    """Import a checkpoint by hardlink or a cross-filesystem atomic copy."""
    source = Path(source)
    destination = Path(destination)
    _validated_checkpoint(source, expected_sha256, expected_size_bytes)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in _HARDLINK_FALLBACK_ERRNOS:
            raise
        return _atomic_copy_checkpoint_create_only(
            source, destination,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
    try:
        identity = _validated_checkpoint(
            destination, expected_sha256, expected_size_bytes,
        )
        fsync_directory(destination.parent)
        return identity
    except BaseException:
        destination.unlink(missing_ok=True)
        fsync_directory(destination.parent)
        raise


def _scan_snapshot_credentials(
    snapshot: Path, inventory: Mapping[str, object],
) -> Mapping[str, object]:
    credential_pattern = re.compile(rb"dtn_[0-9A-Fa-f]{32,}")
    files_scanned = 0
    bytes_scanned = 0
    scan_digest = hashlib.sha256()
    for relative in sorted(inventory):
        path = snapshot / relative
        files_scanned += 1
        overlap = b""
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                bytes_scanned += len(chunk)
                scan_digest.update(chunk)
                scanned = overlap + chunk
                if credential_pattern.search(scanned):
                    raise RuntimeError(
                        f"credential-like material found in worker payload: {relative}"
                    )
                overlap = scanned[-128:]
    return {
        "value_pattern": "dtn_ followed by at least 32 hexadecimal characters",
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "concatenated_payload_sha256": scan_digest.hexdigest(),
        "matches": 0,
    }


def _source_current_head(source_ledger: Path) -> Mapping[str, object]:
    paths = sorted((Path(source_ledger) / "heads").glob("*.head.json"))
    if not paths:
        raise RuntimeError("source ledger has no heads")
    value, identity = read_json(paths[-1])
    if not paths[-1].name.endswith(f"-{identity['sha256']}.head.json"):
        raise RuntimeError("source current head filename/content drift")
    checkpoint = hash_file(
        Path(str(value["checkpoint"]["path"])), str(value["checkpoint"]["sha256"]),
    )
    if int(checkpoint["size_bytes"]) != int(value["checkpoint"]["size_bytes"]):
        raise RuntimeError("source current checkpoint size drift")
    return {**value, "record_sha256": identity["sha256"], "record_path": identity["path"]}


def _source_verify_shape(source_ledger: Path, frozen: Path) -> tuple[int, ...]:
    source_contract, _ = read_json(Path(source_ledger) / "contract.json")
    config_record = source_contract.get("config")
    if not isinstance(config_record, Mapping):
        raise RuntimeError("source ledger contract does not bind its controller")
    controller_path = Path(str(config_record.get("path")))
    controller, _ = read_json(controller_path, str(config_record.get("sha256")))
    shape = tuple(int(value) for value in controller["local_verify"]["bank"]["runtime_batch_shape"])
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise RuntimeError("source authoritative VERIFY runtime shape drift")
    record = controller["local_verify"]["bank"]
    identity = hash_file(frozen / "verify.u32le", str(record["sha256"]))
    if int(identity["size_bytes"]) != math.prod(shape) * 4:
        raise RuntimeError("source authoritative VERIFY shape/bytes drift")
    return shape


def _solver_contract(frozen_root: Path) -> tuple[tuple[str, ...], Mapping[str, object]]:
    protocol, _ = read_json(
        frozen_root / "vendor/hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver"
        / "config/production_protocol.json"
    )
    groups = tuple(str(value) for value in protocol["search"]["independent_branches"])
    if not groups or len(set(groups)) != len(groups):
        raise RuntimeError("frozen vendor group order is empty or ambiguous")
    return groups, protocol


def _group_sizes(
    frozen_root: Path, groups: tuple[str, ...], protocol: Mapping[str, object],
) -> dict[str, int]:
    sys.path.insert(0, str(frozen_root))
    try:
        manifest = importlib.import_module(
            "vendor.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.manifest"
        )
        source_root = frozen_root / "source"
        source_config, _ = read_json(source_root / "config/bottleneck_hfw_1k_100m.json")
        protocol = deepcopy(protocol)
        protocol["frozen_source"]["source_root"] = str(source_root)
        with manifest.isolated_predictive_models(source_root) as module:
            model_config = source_config["models"][protocol["model"]["kind"]]
            with torch.device("meta"):
                model = module.build_student(
                    protocol["model"]["kind"], int(source_config["vocab_size"]),
                    int(protocol["model"]["width"]),
                    int(source_config["model_max_context"]), model_config,
                )
        sizes = {group: 0 for group in groups}
        for name, value in model.named_parameters():
            group = manifest.classify_parameter(name)
            if group in sizes:
                sizes[group] += value.numel()
        if any(value <= 0 for value in sizes.values()):
            raise RuntimeError("derived empty optimizer group")
        return sizes
    finally:
        sys.path.remove(str(frozen_root))


def _initialize_authority(
    package_root: Path, source_ledger: Path, frozen: Path, *,
    minimum_relative_verify_improvement: float,
    duplicate_forward_tolerance: float, host_mps_lock_path: Path,
) -> DistributedLedger:
    ledger = DistributedLedger(package_root / "runtime/authority/lineage")
    if ledger.root.exists():
        contract, _ = ledger.contract()
        expected = {
            "minimum_relative_verify_improvement": minimum_relative_verify_improvement,
            "duplicate_forward_tolerance": duplicate_forward_tolerance,
            "host_mps_lock_path": str(host_mps_lock_path.resolve()),
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                raise RuntimeError(f"immutable authority policy differs from requested wave: {key}")
        return ledger
    source_before = _source_current_head(source_ledger)
    checkpoint_source = Path(str(source_before["checkpoint"]["path"]))
    checkpoint_dir = package_root / "runtime/authority/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{source_before['checkpoint']['sha256']}.pt"
    checkpoint = _import_checkpoint_create_only(
        checkpoint_source, checkpoint_path,
        expected_sha256=str(source_before["checkpoint"]["sha256"]),
        expected_size_bytes=int(source_before["checkpoint"]["size_bytes"]),
    )
    source_after = _source_current_head(source_ledger)
    if source_after["record_sha256"] != source_before["record_sha256"]:
        raise RuntimeError("source head changed while importing distributed authority")
    contract = {
        "status": "ACTIVE_REAL_CHECKPOINT_WRITING_AUTHORITY",
        "source_attempt_ledger": str(Path(source_ledger).resolve()),
        "source_import_head_record_sha256": source_before["record_sha256"],
        "source_import_checkpoint_sha256": source_before["checkpoint"]["sha256"],
        "search_bank_sha256": hash_file(frozen / "search.u32le")["sha256"],
        "verify_bank_sha256": hash_file(frozen / "verify.u32le")["sha256"],
        "vendor_solver_sha256": hash_file(
            frozen / "vendor/hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver/branch_search.py"
        )["sha256"],
        "objective": "mean_next_token_cross_entropy",
        "proposal_lane": "daytona_distinct_fp64_shadow_state_surrogate_stable_layernorm_v1",
        "authority_lane": "mac_mps_bfloat16_duplicate_local_verify_v1",
        "local_verify_disclosure": "PERMANENT_REUSED_SELECTION_DATA_NOT_HELD_OUT",
        "minimum_relative_verify_improvement": minimum_relative_verify_improvement,
        "duplicate_forward_tolerance": duplicate_forward_tolerance,
        "host_mps_lock_path": str(host_mps_lock_path.resolve()),
        "host_mps_lock_contract": "all_MPS_solver_verify_and_benchmark_processes_must_hold_this_exclusive_lock",
        "stale_policy": "reject_entire_wave_without_merge_or_rebase",
        "worker_checkpoint_write_authorized": False,
        "only_mps_promoter_may_append_head": True,
        "training_or_backprop": False,
        "optimizer_steps": 0,
    }
    initial_head = {
        "status": "IMPORTED_CURRENT_MAC_HEAD",
        "model_generation": int(source_before["generation"]),
        "local_promotions": int(source_before["local_promotions"]),
        "source_head_record_sha256": source_before["record_sha256"],
        "source_head_record_path": source_before["record_path"],
        "parent_checkpoint_sha256": source_before["parent_checkpoint_sha256"],
        "checkpoint": {**checkpoint, "format": "hfw_w560_continuous_promoted_checkpoint_v1"},
        "origin_checkpoint_sha256": source_before["origin_checkpoint_sha256"],
        "model_state_digest": source_before["model_state_digest"],
        "accepted_proposal_sha256": source_before["accepted_proposal_sha256"],
        "selected_candidate_order": int(source_before["selected_candidate_order"]),
        "group": source_before["group"],
        "decision_sha256": None,
        "replacement_group_flat_sha256": source_before["replacement_group_flat_sha256"],
        "solver_cumulative_delta_sha256": source_before["solver_cumulative_delta_sha256"],
    }
    return_value = ledger.initialize(contract=contract, initial_head=initial_head)
    if return_value["checkpoint"]["sha256"] != source_before["checkpoint"]["sha256"]:
        raise RuntimeError("initial distributed head checkpoint binding drift")
    return ledger


def bind_campaign(
    package_root: Path, source_ledger: Path, *, workers: int,
    shortlist_per_group: int, nonce: str, cpu_limit: int,
    memory_limit_gib: int, disk_limit_gib: int,
    torch_threads: int, interop_threads: int,
    minimum_relative_verify_improvement: float,
    duplicate_forward_tolerance: float,
    worker_python_major_minor: str, worker_platform_system: str,
    worker_machine: str, worker_torch_version: str, worker_numpy_version: str,
    host_mps_lock_path: Path,
) -> Mapping[str, object]:
    frozen = Path(package_root).resolve() / "runtime/frozen"
    groups, vendor_protocol = _solver_contract(frozen)
    if workers <= 0 or workers % len(groups):
        raise ValueError(f"workers must be a positive multiple of the {len(groups)} derived groups")
    if shortlist_per_group <= 0 or shortlist_per_group > workers // len(groups):
        raise ValueError("invalid shortlist-per-group")
    if min(cpu_limit, memory_limit_gib, disk_limit_gib, torch_threads, interop_threads) <= 0:
        raise ValueError("worker resources and thread counts must be positive")
    if not 0 < minimum_relative_verify_improvement < 1:
        raise ValueError("minimum relative VERIFY improvement must be between zero and one")
    if duplicate_forward_tolerance < 0:
        raise ValueError("duplicate forward tolerance must be nonnegative")
    package_root = Path(package_root).resolve()
    ledger = _initialize_authority(
        package_root, source_ledger, frozen,
        minimum_relative_verify_improvement=minimum_relative_verify_improvement,
        duplicate_forward_tolerance=duplicate_forward_tolerance,
        host_mps_lock_path=host_mps_lock_path,
    )
    group_sizes = _group_sizes(frozen, groups, vendor_protocol)
    verify_shape = _source_verify_shape(source_ledger, frozen)
    with ledger.lock():
        head = ledger.recover_current_pointer_locked()
        wave_index = ledger.next_wave_index()
        wave_root = package_root / "runtime/waves" / f"wave-{wave_index:08d}"
        if wave_root.exists():
            raise FileExistsError(f"wave path exists: {wave_root}")
        snapshot = wave_root / "snapshot"
        snapshot.mkdir(parents=True)
        _hardlink_tree(frozen / "source", snapshot / "source")
        _hardlink_tree(frozen / "vendor", snapshot / "vendor")
        os.link(frozen / "search.u32le", snapshot / "search.u32le")
        os.link(Path(str(head["checkpoint"]["path"])), snapshot / "checkpoint.pt")
        runtime_source = package_root / "runtime/hfw_dist"
        worker_runtime = snapshot / "hfw_dist"
        worker_runtime.mkdir()
        for name in WORKER_RUNTIME_FILES:
            os.link(runtime_source / name, worker_runtime / name)
        inventory = relative_inventory(snapshot)
        credential_scan = _scan_snapshot_credentials(snapshot, inventory)
        snapshot_manifest_core = {
            "schema": "hfw_w560_daytona_snapshot_manifest_v1",
            "status": "BOUND_IMMUTABLE_WORKER_PAYLOAD",
            "wave_index": wave_index,
            "base_head_record_sha256": head["record_sha256"],
            "base_checkpoint_sha256": head["checkpoint"]["sha256"],
            "base_model_state_digest": head["model_state_digest"],
            "files": inventory,
            "verify_bank_included": False,
            "lineage_or_promoter_included": False,
            "cloud_credentials_included": False,
            "credential_pattern_scan": credential_scan,
        }
        snapshot_core_sha = sha256_bytes(canonical_json_bytes(snapshot_manifest_core))
        tasks = []
        per_group = workers // len(groups)
        for worker_index in range(workers):
            group = groups[worker_index % len(groups)]
            ordinal = worker_index // len(groups)
            seed_material = (
                f"{nonce}:{wave_index}:{head['record_sha256']}:{group}:{ordinal}:{worker_index}"
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") & ((1 << 63) - 1)
            tasks.append({
                "task_index": worker_index,
                "worker_id": f"wave-{wave_index:08d}-worker-{worker_index:03d}",
                "group": group,
                "group_ordinal": ordinal,
                "seed": seed,
                "expected_group_elements": group_sizes[group],
                "expected_final_flat_size_bytes": group_sizes[group] * 4,
            })
        campaign = {
            "schema": "hfw_w560_daytona_cpu100_campaign_v1",
            "status": "BOUND_NOT_LAUNCHED",
            "created_unix_ns": time.time_ns(),
            "wave_index": wave_index,
            "campaign_nonce_sha256": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
            "base_head_record_sha256": head["record_sha256"],
            "base_authority_generation": int(head["authority_generation"]),
            "base_checkpoint_sha256": head["checkpoint"]["sha256"],
            "base_checkpoint_size_bytes": int(head["checkpoint"]["size_bytes"]),
            "base_model_state_digest": head["model_state_digest"],
            "origin_checkpoint_sha256": head["origin_checkpoint_sha256"],
            "snapshot_manifest_core_sha256": snapshot_core_sha,
            "search_bank_sha256": inventory["search.u32le"]["sha256"],
            "search_bank_shape": [
                int(vendor_protocol["data"]["calibration_batches"]),
                int(vendor_protocol["data"]["batch_size"]),
                int(vendor_protocol["data"]["context"]) + 1,
            ],
            "worker_count": workers,
            "group_order": list(groups),
            "workers_per_group": per_group,
            "group_elements": group_sizes,
            "tasks": tasks,
            "solver": {
                "vendor_branch_search_sha256": inventory[
                    "vendor/hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver/branch_search.py"
                ]["sha256"],
                "logical_search_evaluations": int(vendor_protocol["search"]["model_evaluation_budget"]),
                "iterations": int(vendor_protocol["search"]["pairs_per_branch"]),
                "physical_model_forwards_per_logical_evaluation": int(vendor_protocol["data"]["batch_size"]),
                "physical_search_model_forwards": int(vendor_protocol["search"]["model_evaluation_budget"]) * int(vendor_protocol["data"]["batch_size"]),
                "terminal_full_search_ranking_logical_evaluations": int(vendor_protocol["data"]["calibration_batches"]),
                "terminal_full_search_ranking_physical_model_forwards": int(vendor_protocol["data"]["calibration_batches"]) * int(vendor_protocol["data"]["batch_size"]),
                "base_center_preflight_logical_evaluations": int(vendor_protocol["data"]["calibration_batches"]),
                "base_center_preflight_physical_model_forwards": int(vendor_protocol["data"]["calibration_batches"]) * int(vendor_protocol["data"]["batch_size"]),
                "acceptance_gate_logical_evaluations_per_attempt": int(vendor_protocol["data"]["calibration_batches"]),
                "acceptance_gate_physical_model_forwards_per_attempt": int(vendor_protocol["data"]["calibration_batches"]) * int(vendor_protocol["data"]["batch_size"]),
                "acceptance_gate_maximum_attempts": int(vendor_protocol["search"]["pairs_per_branch"]),
                "acceptance_gate_requires_all_batches_finite": True,
                "acceptance_gate_requires_strict_full_search_mean_improvement": True,
                "acceptance_gate_before_current_mutation_and_artifact_write": True,
                "total_logical_evaluations_minimum": int(vendor_protocol["search"]["model_evaluation_budget"]) + 2 * int(vendor_protocol["data"]["calibration_batches"]),
                "total_physical_model_forwards_minimum": (int(vendor_protocol["search"]["model_evaluation_budget"]) + 2 * int(vendor_protocol["data"]["calibration_batches"])) * int(vendor_protocol["data"]["batch_size"]),
                "total_logical_evaluations_maximum": int(vendor_protocol["search"]["model_evaluation_budget"]) + 2 * int(vendor_protocol["data"]["calibration_batches"]) + int(vendor_protocol["search"]["pairs_per_branch"]) * int(vendor_protocol["data"]["calibration_batches"]),
                "total_physical_model_forwards_maximum": (int(vendor_protocol["search"]["model_evaluation_budget"]) + 2 * int(vendor_protocol["data"]["calibration_batches"]) + int(vendor_protocol["search"]["pairs_per_branch"]) * int(vendor_protocol["data"]["calibration_batches"])) * int(vendor_protocol["data"]["batch_size"]),
                "forward_only": True,
                "autograd_calls": 0,
                "optimizer_steps": 0,
                "new_unique_token_positions_after_initialization": 0,
            },
            "proposal_lane": {
                "device": "cpu", "forward_device": "cpu",
                "candidate_artifact_dtype": "float32",
                "candidate_codec_dtype": "float32",
                "forward_autocast_enabled": False,
                "forward_autocast_dtype": None,
                "evaluation_parameter_dtype": "float64",
                "floating_buffer_dtype": "float64",
                "evaluation_floating_buffer_dtype": "float64",
                "persistent_state_dtype": "float64",
                "evaluation_persistent_state_dtype": "float64",
                "initial_compressed_state_dtype": "float64",
                "hidden_dtype": "float64",
                "logits_dtype": "float64",
                "ce_dtype": "float64",
                "distinct_fp64_shadow_state_surrogate": True,
                "canonical_checkpoint_and_model_source_bytes_untouched": True,
                "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
                "proposal_shadow_replaces_canonical_fp32_state_storage": True,
                "candidate_and_checkpoint_domain_remains_exact_float32": True,
                "candidate_float32_widened_to_float64_before_every_score": True,
                "canonical_source_forces_codec_and_persistent_state_float32": True,
                "shadow_surrogate_removes_per_step_float32_state_quantization": True,
                "normalization_adapter": {
                    "schema": "maxabs_scaled_fp64_layer_norm_rescaled_epsilon_v1",
                    "real_arithmetic_equivalent_within_fp64_shadow_surrogate": True,
                    "bitwise_equivalent_to_native_layer_norm_claimed": False,
                    "scale_floor": "sqrt_layer_norm_epsilon",
                    "epsilon_rescaling": "(epsilon_divided_by_scale)_divided_by_scale",
                },
                "full_bank_gate_schema": "hfw_w560_cpu_fp64_all_search_acceptance_gate_v1",
                "base_full_bank_gate_schema": "hfw_w560_cpu_fp64_all_search_base_preflight_v1",
                "objective_receipt_schema": "hfw_w560_cpu_fp64_stable_microbatch_objective_receipt_v1",
                "fp64_full_bank_duplicate_absolute_tolerance": duplicate_forward_tolerance,
                "logical_batch_shape": [int(vendor_protocol["data"]["batch_size"]), int(vendor_protocol["data"]["context"]) + 1],
                "physical_microbatch_shape": [1, int(vendor_protocol["data"]["context"]) + 1],
                "torch_threads": torch_threads, "interop_threads": interop_threads,
                "cpu_limit": cpu_limit, "memory_limit_gib": memory_limit_gib, "disk_limit_gib": disk_limit_gib,
                "raw_ce_comparable_to_cpu_fp32_claimed": False,
                "raw_ce_comparable_to_mps_bfloat16_claimed": False,
                "raw_ce_comparable_to_cuda_claimed": False,
                "raw_ce_cross_numerical_lane_comparisons_forbidden": True,
                "raw_ce_cross_lane_comparable": False,
                "mps_sole_promotion_authority": True,
                "base_center_preflight": {
                    "all_workers": True,
                    "seed_worker_task_index": 0,
                    "batch_indices": list(range(int(vendor_protocol["data"]["calibration_batches"]))),
                    "before_solver_scratch_or_artifacts": True,
                    "seed_must_pass_before_fanout": True,
                    "requires_all_logits_hidden_and_per_token_ce_finite": True,
                    "requires_float64_logits_hidden_state_and_ce": True,
                },
                "sealed_environment": {
                    "python_major_minor": worker_python_major_minor,
                    "platform_system": worker_platform_system,
                    "machine": worker_machine,
                    "torch_version": worker_torch_version,
                    "torch_cuda_version": None,
                    "numpy_version": worker_numpy_version,
                    "torch_cpu_index_url": "https://download.pytorch.org/whl/cpu",
                    "cpu_only_torch_required": True,
                },
            },
            "worker_failure_policy": {
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
            },
            "selection_rule": {
                "mode": "all_unique_provisionally_accepted_candidates",
                "ordering": [
                    "task_index_ascending",
                    "worker_candidate_order_ascending",
                ],
                "every_accepted_candidate_downloaded": True,
                "every_unique_accepted_candidate_evaluated_twice_on_canonical_mps_bf16": True,
                "cpu_ranking_filter_permitted": False,
                "cpu_ranking_filter_applied": False,
                "cpu_scores_are_provisional_and_cross_lane_incomparable": True,
                "predeclared_before_worker_launch": True,
                "deduplicate_within_group_by_exact_candidate_flat_sha256": True,
                "keep_first_in_declared_order": True,
                "zero_accepted_candidates_permitted": True,
                "legacy_shortlist_per_group_argument_ignored": shortlist_per_group,
            },
            "promotion": {
                "authority": "single_local_mac_mps_promoter",
                "verify_bank_sha256": hash_file(frozen / "verify.u32le")["sha256"],
                "verify_bank_shape": list(verify_shape),
                "duplicate_parent_and_every_shortlisted_candidate": True,
                "minimum_relative_improvement": minimum_relative_verify_improvement,
                "duplicate_forward_tolerance": duplicate_forward_tolerance,
                "host_mps_lock_path": str(host_mps_lock_path.resolve()),
                "at_most_one_checkpoint_and_head_append": True,
                "stale_wave_rejected_without_merge_or_rebase": True,
                "local_verify_is_reused_selection_data_not_held_out": True,
                "winner_ordering": [
                    "passes_both_duplicated_relative_thresholds",
                    "mps_verify_replicated_mean_ascending",
                    "shortlist_candidate_order_ascending",
                ],
            },
            "worker_checkpoint_write_authorized": False,
            "remote_checkpoint_write_authorized": False,
            "cloud_secret_material_present": False,
            "credential_pattern_scan": credential_scan,
        }
        campaign_payload = canonical_json_bytes(campaign)
        campaign_sha = sha256_bytes(campaign_payload)
        campaign_path = snapshot / "CAMPAIGN.json"
        campaign_identity = write_json_exclusive(campaign_path, campaign)
        if campaign_identity["sha256"] != campaign_sha:
            raise RuntimeError("campaign content-address drift")
        manifest = {
            **snapshot_manifest_core,
            "snapshot_manifest_core_sha256": snapshot_core_sha,
            "campaign_sha256": campaign_sha,
            "campaign_size_bytes": campaign_identity["size_bytes"],
        }
        manifest_identity = write_json_exclusive(snapshot / "SNAPSHOT_MANIFEST.json", manifest)
        ledger_campaign = ledger.append_named_record(
            "campaigns", ".campaign.json", {**campaign, "campaign_sha256": campaign_sha,
                "snapshot_manifest_sha256": manifest_identity["sha256"],
                "wave_root": str(wave_root)},
        )
        write_json_exclusive(wave_root / "BOUND_NOT_LAUNCHED.json", {
            "schema": "hfw_w560_daytona_bound_wave_receipt_v1",
            "status": "BOUND_NOT_LAUNCHED",
            "wave_index": wave_index,
            "campaign": campaign_identity,
            "snapshot_manifest": manifest_identity,
            "ledger_campaign": ledger_campaign,
            "base_head_record_sha256": head["record_sha256"],
            "source_checkpoint_written_by_cloud": False,
        })
        return {
            "wave_index": wave_index, "wave_root": str(wave_root),
            "campaign": campaign_identity, "snapshot_manifest": manifest_identity,
            "base_head": head, "group_elements": group_sizes,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--shortlist-per-group", type=int, default=4)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--cpu-limit", type=int, default=2)
    parser.add_argument("--memory-limit-gib", type=int, default=4)
    parser.add_argument("--disk-limit-gib", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--minimum-relative-verify-improvement", type=float, default=0.0001)
    parser.add_argument("--duplicate-forward-tolerance", type=float, default=1e-7)
    parser.add_argument("--worker-python-major-minor", default="3.12")
    parser.add_argument("--worker-platform-system", default="Linux")
    parser.add_argument("--worker-machine", default="x86_64")
    parser.add_argument("--worker-torch-version", default="2.8.0+cpu")
    parser.add_argument("--worker-numpy-version", default="2.0.2")
    parser.add_argument("--host-mps-lock-path", type=Path, required=True)
    args = parser.parse_args()
    result = bind_campaign(
        args.package_root, args.source_ledger, workers=args.workers,
        shortlist_per_group=args.shortlist_per_group, nonce=args.nonce,
        cpu_limit=args.cpu_limit, memory_limit_gib=args.memory_limit_gib,
        disk_limit_gib=args.disk_limit_gib, torch_threads=args.torch_threads,
        interop_threads=args.interop_threads,
        minimum_relative_verify_improvement=args.minimum_relative_verify_improvement,
        duplicate_forward_tolerance=args.duplicate_forward_tolerance,
        worker_python_major_minor=args.worker_python_major_minor,
        worker_platform_system=args.worker_platform_system,
        worker_machine=args.worker_machine,
        worker_torch_version=args.worker_torch_version,
        worker_numpy_version=args.worker_numpy_version,
        host_mps_lock_path=args.host_mps_lock_path,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
