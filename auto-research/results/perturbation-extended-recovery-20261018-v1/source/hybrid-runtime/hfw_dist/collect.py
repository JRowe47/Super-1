from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

from .common import (
    canonical_json_bytes, hash_file, hash_file_region, read_json, sha256_bytes,
    write_json_exclusive,
)
from .ledger import DistributedLedger


def _contract_value_matches(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, int):
        return (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed == expected
        )
    return observed == expected


def _validate_terminal_duplicate_against_cached(
    *,
    terminal_losses: list[float],
    cached_losses: list[float],
    duplicate: Mapping[str, object],
    current_flat_sha256: str,
    sealed_absolute_tolerance: float,
) -> float:
    if (
        len(terminal_losses) != len(cached_losses)
        or not terminal_losses
        or not all(math.isfinite(value) for value in terminal_losses)
        or not all(math.isfinite(value) for value in cached_losses)
    ):
        raise RuntimeError("terminal/cached full-bank vector drift")
    recomputed_maximum_difference = max(
        abs(terminal - cached)
        for terminal, cached in zip(terminal_losses, cached_losses)
    )
    declared_maximum_difference = float(
        duplicate.get("maximum_absolute_difference", float("inf"))
    )
    declared_tolerance = float(duplicate.get("absolute_tolerance", -1))
    if (
        duplicate.get("status")
        != "PASS_TERMINAL_FULL_BANK_DUPLICATES_CACHED_COMMIT_GATE"
        or duplicate.get("flat_sha256") != current_flat_sha256
        or int(duplicate.get("batch_count", -1)) != len(cached_losses)
        or declared_tolerance != sealed_absolute_tolerance
        or declared_maximum_difference != recomputed_maximum_difference
        or recomputed_maximum_difference > sealed_absolute_tolerance
    ):
        raise RuntimeError("terminal duplicate evidence drift")
    return recomputed_maximum_difference


def _validate_cloud_handoff(
    wave_root: Path, campaign: Mapping[str, object], campaign_identity: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[int, Mapping[str, object]]]:
    downloads, downloads_identity = read_json(wave_root / "CLOUD_DOWNLOADS_COMPLETE.json")
    cleanup, cleanup_identity = read_json(wave_root / "CLOUD_CLEANUP_COMPLETE.json")
    if downloads.get("schema") != "hfw_w560_daytona_cloud_download_terminal_v1":
        raise RuntimeError("cloud download terminal schema drift")
    if downloads.get("status") != "PASS_ALL_100_PROVISIONAL_CPU_OUTPUTS_DOWNLOADED":
        raise RuntimeError("cloud download terminal did not pass")
    if cleanup.get("schema") != "hfw_w560_daytona_cloud_cleanup_terminal_v1":
        raise RuntimeError("cloud cleanup terminal schema drift")
    if cleanup.get("status") != "PASS_ZERO_CAMPAIGN_SANDBOXES_AND_SNAPSHOT_ABSENT":
        raise RuntimeError("cloud cleanup terminal did not pass")
    for record in (downloads, cleanup):
        if (
            record.get("campaign_sha256") != campaign_identity["sha256"]
            or int(record.get("wave_index", -1)) != int(campaign["wave_index"])
        ):
            raise RuntimeError("cloud terminal campaign/wave binding drift")
    if (
        int(cleanup.get("campaign_sandboxes_remaining", -1)) != 0
        or cleanup.get("snapshot_present") is not False
        or cleanup.get("exact_campaign_label_query") is not True
        or cleanup.get("snapshot_not_found_confirmed") is not True
        or cleanup.get("cleanup_failures") != []
    ):
        raise RuntimeError("cloud cleanup did not prove zero exact-label resources")
    if int(downloads.get("worker_count", -1)) != int(campaign["worker_count"]):
        raise RuntimeError("cloud download worker count drift")
    worker_rows = downloads.get("workers", [])
    by_task = {int(row["task_index"]): row for row in worker_rows}
    if len(by_task) != int(campaign["worker_count"]) or sorted(by_task) != list(
        range(int(campaign["worker_count"]))
    ):
        raise RuntimeError("cloud download task inventory is incomplete or duplicated")
    launch_record = downloads.get("launch_manifest", {})
    launch_path = wave_root / "LAUNCH_MANIFEST.json"
    launch, launch_identity = read_json(launch_path, str(launch_record["sha256"]))
    if int(launch_identity["size_bytes"]) != int(launch_record["size_bytes"]):
        raise RuntimeError("launch manifest size drift")
    if (
        launch.get("schema") != "hfw_w560_daytona_launch_manifest_v1"
        or launch.get("status") != "SEED_WORKER_0_PASS_BEFORE_TASKS_1_THROUGH_99"
        or launch.get("campaign_sha256") != campaign_identity["sha256"]
        or int(launch.get("wave_index", -1)) != int(campaign["wave_index"])
    ):
        raise RuntimeError("launch manifest binding/status drift")
    sealed = campaign["proposal_lane"]["sealed_environment"]
    environment = launch.get("sealed_worker_environment", {})
    for key in (
        "python_major_minor", "platform_system", "machine", "torch_version",
        "torch_cuda_version", "numpy_version",
    ):
        if environment.get(key) != sealed[key]:
            raise RuntimeError(f"launch manifest sealed environment drift: {key}")
    seed = launch.get("seed_worker", {})
    if int(seed.get("task_index", -1)) != 0:
        raise RuntimeError("launch manifest lacks worker-0 seed gate")
    if int(seed.get("peak_rss_bytes", -1)) >= int(seed.get("memory_limit_bytes", -1)):
        raise RuntimeError("seed worker did not prove strict memory bound")
    seed_download = by_task[0]
    if (
        seed.get("result_sha256") != seed_download.get("result_sha256")
        or seed.get("flat_sha256") != seed_download.get("flat_sha256")
        or seed.get("accepted_candidate_bundle_sha256")
        != seed_download.get("accepted_candidate_bundle_sha256")
        or seed.get("accepted_candidate_count")
        != seed_download.get("accepted_candidate_count")
    ):
        raise RuntimeError("seed launch/download identity drift")
    return (
        {**downloads, "identity": downloads_identity, "launch_identity": launch_identity},
        {**cleanup, "identity": cleanup_identity}, by_task,
    )


def _ensure_hardlink(source: Path, destination: Path, expected_sha256: str) -> Mapping[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return hash_file(destination, expected_sha256)
    os.link(source, destination)
    os.chmod(destination, 0o444)
    return hash_file(destination, expected_sha256)


def _existing_complete(wave_root: Path) -> Mapping[str, object] | None:
    path = wave_root / "COLLECTION_COMPLETE.json"
    if not path.exists():
        return None
    value, identity = read_json(path)
    if value.get("status") != "PASS_ALL_WORKERS_COLLECTED_AND_ALL_ACCEPTED_CANDIDATES_SEALED":
        raise RuntimeError("existing collection terminal has non-PASS status")
    shortlist, shortlist_identity = read_json(
        Path(str(value["shortlist"]["path"])), str(value["shortlist"]["sha256"]),
    )
    return {**value, "terminal_identity": identity, "shortlist_value": shortlist,
            "shortlist_identity": shortlist_identity}


def collect_wave(package_root: Path, wave_root: Path) -> Mapping[str, object]:
    package_root = Path(package_root).resolve()
    wave_root = Path(wave_root).resolve()
    existing = _existing_complete(wave_root)
    if existing is not None:
        return existing
    campaign, campaign_identity = read_json(wave_root / "snapshot/CAMPAIGN.json")
    cloud_downloads, cloud_cleanup, cloud_worker_by_task = _validate_cloud_handoff(
        wave_root, campaign, campaign_identity,
    )
    ledger = DistributedLedger(package_root / "runtime/authority/lineage")
    downloads = wave_root / "downloads"
    collected = wave_root / "collected"
    collected.mkdir(parents=True, exist_ok=True)
    expected_logical = int(campaign["solver"]["logical_search_evaluations"])
    expected_search_physical = int(campaign["solver"]["physical_search_model_forwards"])
    expected_rank_logical = int(campaign["solver"]["terminal_full_search_ranking_logical_evaluations"])
    expected_rank_physical = int(campaign["solver"]["terminal_full_search_ranking_physical_model_forwards"])
    expected_preflight_logical = int(campaign["solver"]["base_center_preflight_logical_evaluations"])
    expected_preflight_physical = int(campaign["solver"]["base_center_preflight_physical_model_forwards"])
    expected_total_logical_min = int(campaign["solver"]["total_logical_evaluations_minimum"])
    expected_total_physical_min = int(campaign["solver"]["total_physical_model_forwards_minimum"])
    expected_total_logical_max = int(campaign["solver"]["total_logical_evaluations_maximum"])
    expected_total_physical_max = int(campaign["solver"]["total_physical_model_forwards_maximum"])
    gate_logical = int(campaign["solver"]["acceptance_gate_logical_evaluations_per_attempt"])
    gate_physical = int(campaign["solver"]["acceptance_gate_physical_model_forwards_per_attempt"])
    gate_maximum = int(campaign["solver"]["acceptance_gate_maximum_attempts"])
    memory_limit = int(campaign["proposal_lane"]["memory_limit_gib"]) * 1024 ** 3
    sealed_environment = campaign["proposal_lane"]["sealed_environment"]
    rows = []
    for task in campaign["tasks"]:
        task_index = int(task["task_index"])
        worker_root = downloads / f"worker-{task_index:03d}"
        result_path = worker_root / "CPU_SEARCH_RESULT.json"
        flat_path = worker_root / "trajectory_final_group_flat.f32le"
        bundle_path = worker_root / "accepted_candidates.f32le.bundle"
        if not result_path.is_file() or not flat_path.is_file() or not bundle_path.is_file():
            raise RuntimeError(f"worker {task_index} has not returned all fixed output objects")
        extra = sorted(
            path.name for path in worker_root.iterdir()
            if path.is_file() and path.name not in {
                "CPU_SEARCH_RESULT.json", "trajectory_final_group_flat.f32le",
                "accepted_candidates.f32le.bundle",
            }
        )
        if extra:
            raise RuntimeError(f"worker {task_index} download directory has unexpected files: {extra}")
        result, result_identity = read_json(result_path)
        gate_attempts = int(result.get("acceptance_gate_attempts", -1))
        gate_passes = int(result.get("acceptance_gate_passes", -1))
        gate_failures = int(result.get("acceptance_gate_failures", -1))
        accepted_updates = int(result.get("accepted_updates", -1))
        if not (
            0 <= accepted_updates == gate_passes <= gate_attempts <= gate_maximum
            and gate_failures == gate_attempts - gate_passes
            and int(result.get("provisional_search_accepts", -1)) == accepted_updates
        ):
            raise RuntimeError(f"worker {task_index} acceptance gate relation drift")
        expected_total_logical = expected_total_logical_min + gate_attempts * gate_logical
        expected_total_physical = expected_total_physical_min + gate_attempts * gate_physical
        task_sha = sha256_bytes(canonical_json_bytes(task))
        requirements = {
            "schema": "hfw_w560_daytona_cpu_search_result_v1",
            "status": "PASS_PROVISIONAL_CPU_SEARCH_ONLY",
            "campaign_sha256": campaign_identity["sha256"],
            "task_sha256": task_sha,
            "task_index": task_index,
            "worker_id": task["worker_id"],
            "base_head_record_sha256": campaign["base_head_record_sha256"],
            "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
            "base_model_state_digest": campaign["base_model_state_digest"],
            "group": task["group"],
            "seed": int(task["seed"]),
            "group_elements": int(task["expected_group_elements"]),
            "pairs": int(campaign["solver"]["iterations"]),
            "logical_search_evaluations": expected_logical,
            "physical_search_model_forwards": expected_search_physical,
            "terminal_ranking_logical_evaluations": expected_rank_logical,
            "terminal_ranking_physical_model_forwards": expected_rank_physical,
            "solver_logical_evaluations_excluding_preflight": expected_logical + expected_rank_logical + gate_attempts * gate_logical,
            "solver_physical_model_forwards_excluding_preflight": expected_search_physical + expected_rank_physical + gate_attempts * gate_physical,
            "base_center_preflight_logical_evaluations": expected_preflight_logical,
            "base_center_preflight_physical_model_forwards": expected_preflight_physical,
            "acceptance_gate_logical_evaluations": gate_attempts * gate_logical,
            "acceptance_gate_physical_model_forwards": gate_attempts * gate_physical,
            "total_logical_evaluations": expected_total_logical,
            "total_physical_model_forwards": expected_total_physical,
            "worker_checkpoint_or_head_writes": 0,
            "mps_verify_opened": False,
            "local_verify_available_to_worker": False,
            "cloud_credentials_available_to_worker": False,
            "training_or_backprop": False,
            "optimizer_steps": 0,
            "provisional_not_held_out_or_promoted": True,
            "distinct_fp64_shadow_state_surrogate": True,
            "canonical_checkpoint_and_model_source_bytes_untouched": True,
            "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
            "proposal_shadow_replaces_canonical_fp32_state_storage": True,
            "raw_ce_cross_lane_comparable": False,
            "mps_sole_promotion_authority": True,
            "accepted_candidate_count": accepted_updates,
            "every_provisionally_accepted_candidate_in_bundle": True,
            "candidate_subset_selection_performed": False,
            "cpu_ranking_used_for_candidate_selection": False,
        }
        for key, expected in requirements.items():
            observed = result.get(key)
            if not _contract_value_matches(observed, expected):
                raise RuntimeError(f"worker {task_index} result contract drift: {key}")
        preflight = result.get("base_center_preflight", {})
        preflight_batches = preflight.get("batches", [])
        if (
            preflight.get("schema") != "hfw_w560_cpu_fp64_all_search_base_preflight_v1"
            or preflight.get("status") != "PASS_FINITE_CPU_FP64_ALL_SEARCH_BASE"
            or preflight.get("batch_indices") != list(range(expected_preflight_logical))
            or preflight.get("checkpoint_candidate_dtype") != "float32"
            or preflight.get("evaluation_parameter_dtype") != "float64"
            or preflight.get("forward_device") != "cpu"
            or preflight.get("forward_autocast_dtype") is not None
            or preflight.get("ce_dtype") != "float64"
            or len(preflight_batches) != expected_preflight_logical
            or int(preflight.get("preflight_logical_evaluations", -1))
            != expected_preflight_logical
            or int(preflight.get("preflight_physical_model_forwards", -1))
            != expected_preflight_physical
        ):
            raise RuntimeError(f"worker {task_index} base-center preflight contract drift")
        for batch_index, batch in enumerate(preflight_batches):
            per_sequence = batch.get("per_sequence", [])
            if int(batch.get("batch_index", -1)) != batch_index or batch.get("passed") is not True:
                raise RuntimeError(f"worker {task_index} base preflight batch drift")
            if len(per_sequence) != int(campaign["search_bank_shape"][1]):
                raise RuntimeError(f"worker {task_index} base sequence count drift")
            for sequence_index, evidence in enumerate(per_sequence):
                if (
                    int(evidence.get("sequence_index", -1)) != sequence_index
                    or evidence.get("logits_dtype") != "float64"
                    or evidence.get("logits_dtype_matches") is not True
                    or evidence.get("hidden_dtype") != "float64"
                    or evidence.get("hidden_dtype_matches") is not True
                    or evidence.get("persistent_state_dtype_matches") is not True
                    or evidence.get("ce_dtype") != "float64"
                    or evidence.get("loss_class") != "FINITE"
                    or not math.isfinite(float(evidence.get("loss", float("nan"))))
                    or any(
                        evidence.get(name, {}).get("all_finite") is not True
                        or evidence.get(name, {}).get("first_nonfinite_timestep") is not None
                        for name in ("logits", "hidden", "per_token_loss")
                    )
                ):
                    raise RuntimeError(f"worker {task_index} base per-sequence evidence drift")
        base_losses = [float(value) for value in result["base_full_search_batch_losses"]]
        if len(base_losses) != expected_preflight_logical or not all(math.isfinite(v) for v in base_losses):
            raise RuntimeError(f"worker {task_index} base loss count drift")
        base_score = math.fsum(base_losses) / len(base_losses)
        if (
            base_losses != [float(value) for value in preflight["base_full_search_batch_losses"]]
            or not math.isclose(base_score, float(preflight["base_full_search_mean_ce"]), rel_tol=0, abs_tol=1e-12)
            or not math.isclose(base_score, float(result["base_full_search_mean_ce"]), rel_tol=0, abs_tol=1e-12)
        ):
            raise RuntimeError(f"worker {task_index} base mean does not recompute")
        batch_losses = [float(value) for value in result["terminal_full_search_batch_losses"]]
        if len(batch_losses) != expected_rank_logical or not all(
            math.isfinite(value) for value in batch_losses
        ):
            raise RuntimeError(f"worker {task_index} ranking loss count drift")
        score = math.fsum(batch_losses) / len(batch_losses)
        if not math.isclose(
            score, float(result["terminal_full_search_mean_ce"]), rel_tol=0, abs_tol=1e-12,
        ):
            raise RuntimeError(f"worker {task_index} declared ranking mean does not recompute")
        improvement = base_score - score
        if not math.isclose(
            improvement, float(result["self_calibrated_full_search_improvement"]),
            rel_tol=0, abs_tol=1e-12,
        ):
            raise RuntimeError(f"worker {task_index} self-calibrated improvement drift")
        peak_rss = int(result["environment"]["peak_rss_bytes"])
        if peak_rss >= memory_limit or result["environment"].get(
            "peak_rss_strictly_below_memory_limit"
        ) is not True:
            raise RuntimeError(f"worker {task_index} did not prove the memory bound")
        if (
            not isinstance(result["environment"].get("torch_cpu_capability"), str)
            or not result["environment"]["torch_cpu_capability"]
            or not isinstance(
                result["environment"].get("proc_cpuinfo_distinct_model_names"), list
            )
            or not result["environment"]["proc_cpuinfo_distinct_model_names"]
        ):
            raise RuntimeError(f"worker {task_index} CPU capability/model disclosure missing")
        for key in (
            "python_major_minor", "platform_system", "machine", "torch_version",
            "torch_cuda_version", "numpy_version",
        ):
            if result["environment"].get(key) != sealed_environment[key]:
                raise RuntimeError(f"worker {task_index} sealed environment drift: {key}")
        objective = result.get("objective_runtime", {})
        phase_logical = objective.get("phase_logical_evaluations", {})
        phase_physical = objective.get("phase_physical_model_forwards", {})
        gate_records = objective.get("acceptance_gate_records", [])
        if (
            objective.get("schema") != "hfw_w560_cpu_fp64_stable_microbatch_objective_receipt_v1"
            or int(objective.get("logical_evaluations_excluding_preflight", -1))
            != expected_logical + expected_rank_logical + gate_attempts * gate_logical
            or int(objective.get("physical_model_forwards_excluding_preflight", -1))
            != expected_search_physical + expected_rank_physical + gate_attempts * gate_physical
            or int(objective.get("preflight_logical_evaluations", -1))
            != expected_preflight_logical
            or int(objective.get("preflight_physical_model_forwards", -1))
            != expected_preflight_physical
            or int(objective.get("total_logical_evaluations_including_preflight", -1))
            != expected_total_logical
            or int(objective.get("total_physical_model_forwards_including_preflight", -1))
            != expected_total_physical
            or int(objective.get("float64_logits_observed", -1))
            != expected_total_physical
            or int(objective.get("float64_hidden_observed", -1))
            != expected_total_physical
            or int(objective.get("float64_persistent_state_observed", -1))
            != expected_total_physical
            or int(objective.get("float64_ce_evaluations", -1))
            != expected_total_physical
            or objective.get("checkpoint_candidate_dtype") != "float32"
            or objective.get("evaluation_parameter_dtype") != "float64"
            or objective.get("forward_autocast_dtype") is not None
            or objective.get("logits_dtype") != "float64"
            or objective.get("ce_dtype") != "float64"
            or objective.get("candidate_f32_widened_to_fp64_for_every_score") is not True
            or objective.get("canonical_model_source_or_checkpoint_mutated") is not False
            or int(objective.get("acceptance_gate_attempts", -1)) != gate_attempts
            or int(objective.get("acceptance_gate_passes", -1)) != gate_passes
            or int(objective.get("acceptance_gate_failures", -1)) != gate_failures
            or len(gate_records) != gate_attempts
            or int(phase_logical.get("SEARCH", -1)) != expected_logical
            or int(phase_physical.get("SEARCH", -1)) != expected_search_physical
            or int(phase_logical.get("ACCEPTANCE_COMMIT_GATE", -1)) != gate_attempts * gate_logical
            or int(phase_physical.get("ACCEPTANCE_COMMIT_GATE", -1)) != gate_attempts * gate_physical
            or int(phase_logical.get("TERMINAL_RANKING", -1)) != expected_rank_logical
            or int(phase_physical.get("TERMINAL_RANKING", -1)) != expected_rank_physical
            or objective.get("optimizer_steps") != 0
            or objective.get("autograd_calls") != 0
        ):
            raise RuntimeError(f"worker {task_index} objective runtime accounting drift")
        observed_passes = 0
        current_flat_sha = str(preflight["base_candidate_flat_sha256"])
        cached_full_search_losses = list(base_losses)
        cached_full_search_mean = base_score
        for gate_index, gate in enumerate(gate_records):
            losses = gate.get("candidate_batch_losses", [])
            finite_losses = all(
                value is not None and math.isfinite(float(value)) for value in losses
            )
            candidate_mean = (
                math.fsum(float(value) for value in losses) / len(losses)
                if finite_losses and losses else None
            )
            expected_pass = bool(
                finite_losses
                and candidate_mean < cached_full_search_mean
            )
            if (
                gate.get("schema") != "hfw_w560_cpu_fp64_all_search_acceptance_gate_v1"
                or int(gate.get("gate_index", -1)) != gate_index
                or gate.get("group") != task["group"]
                or gate.get("current_flat_sha256") != current_flat_sha
                or not math.isclose(
                    float(gate.get("current_full_search_mean_ce", float("nan"))),
                    cached_full_search_mean, rel_tol=0, abs_tol=1e-12,
                )
                or gate.get("batch_indices") != list(range(gate_logical))
                or len(losses) != gate_logical
                or int(gate.get("logical_evaluations", -1)) != gate_logical
                or int(gate.get("physical_model_forwards", -1)) != gate_physical
                or gate.get("current_mutation_or_artifact_write_before_gate") is not False
                or gate.get("all_finite") is not finite_losses
                or gate.get("strict_full_search_mean_improvement") is not expected_pass
                or gate.get("passed") is not expected_pass
                or (
                    candidate_mean is not None and not math.isclose(
                        candidate_mean, float(gate["candidate_full_search_mean_ce"]),
                        rel_tol=0, abs_tol=1e-12,
                    )
                )
                or (
                    candidate_mean is None
                    and gate.get("candidate_full_search_mean_ce") is not None
                )
            ):
                raise RuntimeError(f"worker {task_index} acceptance gate evidence drift")
            observed_passes += int(expected_pass)
            if expected_pass:
                current_flat_sha = str(gate["candidate_flat_sha256"])
                cached_full_search_losses = [float(value) for value in losses]
                cached_full_search_mean = float(candidate_mean)
        if observed_passes != gate_passes:
            raise RuntimeError(f"worker {task_index} acceptance gate pass count drift")
        duplicate = result.get("terminal_full_search_duplicate", {})
        try:
            _validate_terminal_duplicate_against_cached(
                terminal_losses=batch_losses,
                cached_losses=cached_full_search_losses,
                duplicate=duplicate,
                current_flat_sha256=current_flat_sha,
                sealed_absolute_tolerance=float(
                    campaign["proposal_lane"][
                        "fp64_full_bank_duplicate_absolute_tolerance"
                    ]
                ),
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"worker {task_index} terminal duplicate evidence drift"
            ) from error
        if not math.isclose(
            score, cached_full_search_mean, rel_tol=0,
            abs_tol=float(
                campaign["proposal_lane"][
                    "fp64_full_bank_duplicate_absolute_tolerance"
                ]
            ),
        ):
            raise RuntimeError(f"worker {task_index} terminal/cached mean drift")
        declared_flat = result["trajectory_final_group_flat"]
        local_flat = hash_file(flat_path, str(declared_flat["sha256"]))
        if (
            int(local_flat["size_bytes"]) != int(declared_flat["size_bytes"])
            or int(local_flat["size_bytes"]) != int(task["expected_final_flat_size_bytes"])
            or local_flat["sha256"] != duplicate["flat_sha256"]
            or local_flat["sha256"] != current_flat_sha
        ):
            raise RuntimeError(f"worker {task_index} fixed flat file size drift")
        declared_bundle = result.get("accepted_candidate_bundle")
        accepted_candidates = result.get("accepted_candidates")
        if not isinstance(declared_bundle, Mapping) or not isinstance(
            accepted_candidates, list,
        ):
            raise RuntimeError(f"worker {task_index} accepted candidate bundle missing")
        local_bundle = hash_file(bundle_path, str(declared_bundle.get("sha256")))
        expected_flat_size = int(task["expected_final_flat_size_bytes"])
        passing_gates = [gate for gate in gate_records if gate.get("passed") is True]
        if (
            len(accepted_candidates) != accepted_updates
            or len(passing_gates) != accepted_updates
            or int(local_bundle["size_bytes"]) != accepted_updates * expected_flat_size
            or int(local_bundle["size_bytes"])
            != int(declared_bundle.get("size_bytes", -1))
        ):
            raise RuntimeError(f"worker {task_index} accepted candidate bundle size drift")
        validated_candidates = []
        for candidate_order, (candidate, gate) in enumerate(
            zip(accepted_candidates, passing_gates), start=1,
        ):
            if not isinstance(candidate, Mapping):
                raise RuntimeError(f"worker {task_index} accepted candidate record drift")
            candidate_sha = str(gate["candidate_flat_sha256"])
            offset_bytes = (candidate_order - 1) * expected_flat_size
            if (
                candidate.get("candidate_order") != candidate_order
                or candidate.get("accept_index") != candidate_order
                or candidate.get("pair_index") != gate.get("iteration")
                or candidate.get("group") != task["group"]
                or candidate.get("offset_bytes") != offset_bytes
                or candidate.get("size_bytes") != expected_flat_size
                or candidate.get("group_flat_sha256") != candidate_sha
                or not math.isclose(
                    float(candidate.get(
                        "candidate_cpu_full_search_mean_ce", float("nan")
                    )),
                    float(gate["candidate_full_search_mean_ce"]),
                    rel_tol=0, abs_tol=1e-12,
                )
            ):
                raise RuntimeError(
                    f"worker {task_index} accepted candidate provenance drift"
                )
            hash_file_region(
                bundle_path, offset_bytes, expected_flat_size, candidate_sha,
            )
            validated_candidates.append(dict(candidate))
        if accepted_updates and validated_candidates[-1]["group_flat_sha256"] != current_flat_sha:
            raise RuntimeError(
                f"worker {task_index} terminal flat is not the final accepted candidate"
            )
        cloud_row = cloud_worker_by_task[task_index]
        if (
            cloud_row.get("result_sha256") != result_identity["sha256"]
            or int(cloud_row.get("result_size_bytes", -1)) != int(result_identity["size_bytes"])
            or cloud_row.get("flat_sha256") != local_flat["sha256"]
            or int(cloud_row.get("flat_size_bytes", -1)) != int(local_flat["size_bytes"])
            or cloud_row.get("accepted_candidate_bundle_sha256")
            != local_bundle["sha256"]
            or int(cloud_row.get("accepted_candidate_bundle_size_bytes", -1))
            != int(local_bundle["size_bytes"])
            or int(cloud_row.get("accepted_candidate_count", -1))
            != accepted_updates
        ):
            raise RuntimeError(f"worker {task_index} download terminal/file identity drift")
        worker_collected = collected / f"worker-{task_index:03d}"
        flat_copy = _ensure_hardlink(
            flat_path,
            worker_collected / f"{local_flat['sha256']}.trajectory_final_group_flat.f32le",
            str(local_flat["sha256"]),
        )
        result_copy = _ensure_hardlink(
            result_path,
            worker_collected / f"{result_identity['sha256']}.CPU_SEARCH_RESULT.json",
            str(result_identity["sha256"]),
        )
        bundle_copy = _ensure_hardlink(
            bundle_path,
            worker_collected / f"{local_bundle['sha256']}.accepted_candidates.f32le.bundle",
            str(local_bundle["sha256"]),
        )
        rows.append({
            "task_index": task_index, "worker_id": task["worker_id"],
            "group": task["group"], "seed": int(task["seed"]),
            "task_sha256": task_sha,
            "terminal_full_search_mean_ce": score,
            "base_full_search_mean_ce": base_score,
            "self_calibrated_full_search_improvement": improvement,
            "provisional_search_accepts": int(result["provisional_search_accepts"]),
            "peak_rss_bytes": peak_rss,
            "trajectory_final_group_flat": flat_copy,
            "cpu_search_result": result_copy,
            "accepted_candidate_bundle": bundle_copy,
            "accepted_candidates": validated_candidates,
            "remote_declared_absolute_path_trusted": False,
            "fixed_download_filename_hash_and_size_validated": True,
        })
    if len(rows) != int(campaign["worker_count"]):
        raise RuntimeError("not every campaign worker was collected")
    groups = tuple(str(value) for value in campaign["group_order"])
    shortlisted = []
    skipped_duplicates = []
    for group in groups:
        eligible = [row for row in rows if row["group"] == group]
        if len(eligible) != int(campaign["workers_per_group"]):
            raise RuntimeError(f"collected group balance drift: {group}")
    keeper_by_key = {}
    all_candidates = []
    for row in sorted(rows, key=lambda item: int(item["task_index"])):
        for candidate in row["accepted_candidates"]:
            all_candidates.append((row, candidate))
    for row, candidate in all_candidates:
        digest = str(candidate["group_flat_sha256"])
        group = str(row["group"])
        dedupe_key = (group, digest)
        if dedupe_key in keeper_by_key:
            keeper = keeper_by_key[dedupe_key]
            skipped_duplicates.append({
                "group": group,
                "duplicate_task_index": row["task_index"],
                "duplicate_worker_candidate_order": candidate["candidate_order"],
                "duplicate_flat_sha256": digest,
                "keeper_task_index": keeper["task_index"],
                "keeper_worker_candidate_order": keeper["worker_candidate_order"],
                "dedupe_key": "group_and_group_flat_sha256",
            })
            continue
        shortlist_row = {
            "candidate_order": len(shortlisted) + 1,
            "task_index": row["task_index"],
            "worker_id": row["worker_id"],
            "worker_candidate_order": candidate["candidate_order"],
            "accept_index": candidate["accept_index"],
            "pair_index": candidate["pair_index"],
            "group": group,
            "seed": row["seed"],
            "task_sha256": row["task_sha256"],
            "candidate_group_flat_sha256": digest,
            "candidate_cpu_full_search_mean_ce": candidate[
                "candidate_cpu_full_search_mean_ce"
            ],
            "accepted_record_sha256": candidate["accepted_record_sha256"],
            "artifact_only_child_sha256": candidate[
                "artifact_only_child_sha256"
            ],
            "cumulative_delta_sha256": candidate["cumulative_delta_sha256"],
            "accepted_candidate_bundle": row["accepted_candidate_bundle"],
            "offset_bytes": candidate["offset_bytes"],
            "size_bytes": candidate["size_bytes"],
            "cpu_search_result": row["cpu_search_result"],
            "cpu_score_used_for_selection": False,
        }
        keeper_by_key[dedupe_key] = shortlist_row
        shortlisted.append(shortlist_row)
    shortlist = {
        "schema": "hfw_w560_daytona_all_accepted_candidates_v1",
        "status": "PASS_ALL_UNIQUE_ACCEPTED_CANDIDATES_SEALED",
        "campaign_sha256": campaign_identity["sha256"],
        "wave_index": int(campaign["wave_index"]),
        "base_head_record_sha256": campaign["base_head_record_sha256"],
        "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
        "base_model_state_digest": campaign["base_model_state_digest"],
        "selection_rule": campaign["selection_rule"],
        "candidate_count": len(shortlisted),
        "accepted_candidate_count_before_exact_dedupe": len(all_candidates),
        "candidates": shortlisted,
        "skipped_exact_flat_duplicates": skipped_duplicates,
        "dedupe_scope": "within_group_only",
        "dedupe_key": "group_and_group_flat_sha256",
        "first_in_task_then_accept_order_kept": True,
        "candidate_order": "task_index_ascending_then_worker_candidate_order_ascending",
        "all_worker_results_validated_before_selection": True,
        "every_accepted_candidate_downloaded": True,
        "every_unique_accepted_candidate_selected_for_mps_verify": True,
        "cpu_ranking_filter_permitted": False,
        "cpu_ranking_filter_applied": False,
        "cpu_fp64_shadow_surrogate_scores_are_provisional_not_cross_lane_comparable": True,
        "canonical_mps_full_context_verify_is_sole_promotion_authority": True,
        "each_mixed_group_candidate_is_independent_against_same_base": True,
        "no_cross_group_composition": True,
        "mps_duplicate_verify_required": True,
    }
    with ledger.lock():
        head = ledger.recover_current_pointer_locked()
        if head["record_sha256"] != campaign["base_head_record_sha256"]:
            raise RuntimeError("campaign became stale before collection; refusing shortlist")
        collection = {
            "schema": "hfw_w560_daytona_complete_collection_v1",
            "status": "PASS_ALL_100_PROVISIONAL_RESULTS_VALIDATED",
            "campaign_sha256": campaign_identity["sha256"],
            "wave_index": int(campaign["wave_index"]),
            "worker_count": len(rows),
            "workers": rows,
            "all_fixed_flat_filenames_hashes_and_sizes_validated": True,
            "all_accepted_candidate_bundle_hashes_sizes_and_regions_validated": True,
            "accepted_candidate_count_before_exact_dedupe": len(all_candidates),
            "all_peak_rss_strictly_below_bound": True,
            "checkpoint_or_head_writes": 0,
            "cloud_download_terminal_sha256": cloud_downloads["identity"]["sha256"],
            "cloud_cleanup_terminal_sha256": cloud_cleanup["identity"]["sha256"],
            "launch_manifest_sha256": cloud_downloads["launch_identity"]["sha256"],
            "zero_cloud_resources_proven_before_collection": True,
        }
        collection_identity = ledger.append_named_record(
            "collections", ".collection.json", collection,
        )
        shortlist_identity = ledger.append_named_record(
            "shortlists", ".shortlist.json", shortlist,
        )
    terminal = {
        "schema": "hfw_w560_daytona_collection_terminal_v1",
        "status": "PASS_ALL_WORKERS_COLLECTED_AND_ALL_ACCEPTED_CANDIDATES_SEALED",
        "campaign_sha256": campaign_identity["sha256"],
        "wave_index": int(campaign["wave_index"]),
        "collection": collection_identity,
        "shortlist": shortlist_identity,
        "candidate_count": len(shortlisted),
        "mps_verify_opened": False,
        "checkpoint_or_head_writes": 0,
    }
    terminal_identity = write_json_exclusive(wave_root / "COLLECTION_COMPLETE.json", terminal)
    return {**terminal, "terminal_identity": terminal_identity,
            "shortlist_value": shortlist, "shortlist_identity": shortlist_identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    args = parser.parse_args()
    result = collect_wave(args.package_root, args.wave_root)
    print(json.dumps({
        "status": result["status"], "wave_index": result["wave_index"],
        "candidate_count": result["candidate_count"],
        "shortlist_sha256": result["shortlist"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
