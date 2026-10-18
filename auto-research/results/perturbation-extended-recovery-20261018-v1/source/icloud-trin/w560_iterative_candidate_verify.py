from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical one-candidate-at-a-time MPS verification for an iterative promotion loop."
    )
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--wave-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument(
        "--minimum-relative-improvement-override",
        type=float,
        help=(
            "Override the campaign promotion margin. Zero means any strict duplicated "
            "canonical improvement over the parent is retained as passing."
        ),
    )
    parser.add_argument(
        "--reuse-verification-root",
        type=Path,
        help=(
            "Import already-completed duplicated MPS scores from another immutable "
            "verification root after validating their numerical and artifact bindings."
        ),
    )
    parser.add_argument(
        "--successor-snapshot-root",
        type=Path,
        help=(
            "Create a successor fixed snapshot from this completed verification root, "
            "bound to the current promoted head and containing only candidates after "
            "--start-candidate-order."
        ),
    )
    parser.add_argument("--start-candidate-order", type=int, default=1)
    parser.add_argument(
        "--promote-after-exhaustive",
        action="store_true",
        help="Append an operator-stopped partial-campaign decision and at most one head after exhaustive scoring.",
    )
    parser.add_argument(
        "--promote-after-evaluated-count",
        type=int,
        help=(
            "Explicit operator checkpoint boundary. Promote the best strict improver "
            "after this many fixed-snapshot candidates instead of waiting for the full set."
        ),
    )
    parser.add_argument(
        "--promote-first-strict-improver",
        action="store_true",
        help=(
            "Keep one canonical parent loaded, stop at the first strict duplicated "
            "improver, and promote it immediately. Exhaust the snapshot on no pass."
        ),
    )
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.staged"
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _head(package_root: Path, read_json) -> tuple[dict[str, object], dict[str, object]]:
    current, current_identity = read_json(package_root / "runtime/authority/lineage/CURRENT.json")
    head, head_identity = read_json(
        Path(str(current["record_path"])), str(current["record_sha256"])
    )
    if head_identity["sha256"] != current["record_sha256"]:
        raise RuntimeError("current head pointer identity drift")
    head = dict(head)
    head["record_sha256"] = head_identity["sha256"]
    return head, dict(current_identity)


def _validated_operator_stop_cleanup(wave_root: Path, campaign_sha256: str, read_json, hash_file):
    rows = []
    for path in sorted((wave_root / "orchestration/receipts").glob("*.receipt.json")):
        value, identity = read_json(path)
        if value.get("event") == "CLEANUP_TERMINAL":
            rows.append((value, identity))
    if len(rows) != 1:
        raise RuntimeError("expected exactly one operator-stop cleanup terminal receipt")
    cleanup, identity = rows[0]
    if (
        cleanup.get("campaign_sha256") != campaign_sha256
        or cleanup.get("reason") != "FAILURE"
        or cleanup.get("exact_campaign_label_relist_passed") is not True
        or cleanup.get("final_relist_succeeded") is not True
        or int(cleanup.get("campaign_sandbox_count", -1)) != 0
        or int(cleanup.get("final_remaining_count", -1)) != 0
        or cleanup.get("final_remaining_ids") != []
        or cleanup.get("snapshot_present") is not False
        or cleanup.get("snapshot_relist_succeeded") is not True
        or cleanup.get("snapshot_exact_name_not_found_confirmed") is not True
        or int(cleanup.get("failure_count", -1)) != 0
        or cleanup.get("failures") != []
    ):
        raise RuntimeError("operator-stop cleanup receipt does not prove zero exact resources")
    return cleanup, hash_file(Path(str(identity["path"])), str(identity["sha256"]))


def _snapshot_candidates(package_root: Path, wave_root: Path, output_root: Path, api) -> dict[str, object]:
    read_json = api["read_json"]
    hash_file = api["hash_file"]
    hash_file_region = api["hash_file_region"]
    sha256_bytes = api["sha256_bytes"]
    canonical_json_bytes = api["canonical_json_bytes"]

    campaign, campaign_identity = read_json(wave_root / "snapshot/CAMPAIGN.json")
    head, current_identity = _head(package_root, read_json)
    if head["record_sha256"] != campaign["base_head_record_sha256"]:
        raise RuntimeError("campaign base head is no longer current")
    if head["checkpoint"]["sha256"] != campaign["base_checkpoint_sha256"]:
        raise RuntimeError("campaign base checkpoint is no longer current")

    task_by_index = {int(row["task_index"]): row for row in campaign["tasks"]}
    workers: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    keeper: dict[tuple[str, str], dict[str, object]] = {}
    downloads = wave_root / "downloads"
    for worker_root in sorted(downloads.glob("worker-*")):
        if not worker_root.is_dir():
            continue
        task_index = int(worker_root.name.split("-")[1])
        task = task_by_index.get(task_index)
        if task is None:
            raise RuntimeError(f"unexpected task index {task_index}")
        result_path = worker_root / "CPU_SEARCH_RESULT.json"
        bundle_path = worker_root / "accepted_candidates.f32le.bundle"
        flat_path = worker_root / "trajectory_final_group_flat.f32le"
        if not (result_path.is_file() and bundle_path.is_file() and flat_path.is_file()):
            continue
        result, result_identity = read_json(result_path)
        if (
            result.get("schema") != "hfw_w560_daytona_cpu_search_result_v1"
            or result.get("status") != "PASS_PROVISIONAL_CPU_SEARCH_ONLY"
            or result.get("campaign_sha256") != campaign_identity["sha256"]
            or int(result.get("task_index", -1)) != task_index
            or result.get("task_sha256") != sha256_bytes(canonical_json_bytes(task))
            or result.get("base_head_record_sha256") != campaign["base_head_record_sha256"]
            or result.get("base_checkpoint_sha256") != campaign["base_checkpoint_sha256"]
            or result.get("group") != task["group"]
            or int(result.get("seed", -1)) != int(task["seed"])
            or result.get("every_provisionally_accepted_candidate_in_bundle") is not True
            or result.get("cpu_ranking_used_for_candidate_selection") is not False
            or result.get("mps_verify_opened") is not False
        ):
            raise RuntimeError(f"worker {task_index} result contract drift")
        declared_bundle = result["accepted_candidate_bundle"]
        bundle_identity = hash_file(bundle_path, str(declared_bundle["sha256"]))
        if int(bundle_identity["size_bytes"]) != int(declared_bundle["size_bytes"]):
            raise RuntimeError(f"worker {task_index} bundle size drift")
        accepted = result.get("accepted_candidates")
        if (
            not isinstance(accepted, list)
            or len(accepted) != int(result.get("accepted_candidate_count", -1))
            or len(accepted) != int(result.get("accepted_updates", -1))
        ):
            raise RuntimeError(f"worker {task_index} accepted candidate inventory drift")
        workers.append(
            {
                "task_index": task_index,
                "group": str(task["group"]),
                "seed": int(task["seed"]),
                "result": result_identity,
                "bundle": bundle_identity,
                "accepted_candidate_count": len(accepted),
            }
        )
        for worker_order, candidate in enumerate(accepted, start=1):
            group = str(candidate["group"])
            digest = str(candidate["group_flat_sha256"])
            offset = int(candidate["offset_bytes"])
            size = int(candidate["size_bytes"])
            if (
                group != task["group"]
                or int(candidate["candidate_order"]) != worker_order
                or int(candidate["accept_index"]) != worker_order
                or size != int(task["expected_final_flat_size_bytes"])
                or offset != (worker_order - 1) * size
            ):
                raise RuntimeError(f"worker {task_index} candidate {worker_order} provenance drift")
            hash_file_region(bundle_path, offset, size, digest)
            key = (group, digest)
            if key in keeper:
                duplicates.append(
                    {
                        "group": group,
                        "flat_sha256": digest,
                        "duplicate_task_index": task_index,
                        "duplicate_worker_candidate_order": worker_order,
                        "keeper_candidate_order": keeper[key]["candidate_order"],
                    }
                )
                continue
            row = {
                "candidate_order": len(candidates) + 1,
                "task_index": task_index,
                "worker_candidate_order": worker_order,
                "group": group,
                "seed": int(task["seed"]),
                "candidate_group_flat_sha256": digest,
                "candidate_cpu_full_search_mean_ce": float(
                    candidate["candidate_cpu_full_search_mean_ce"]
                ),
                "bundle_path": str(bundle_path.resolve()),
                "bundle_sha256": bundle_identity["sha256"],
                "bundle_size_bytes": int(bundle_identity["size_bytes"]),
                "offset_bytes": offset,
                "size_bytes": size,
            }
            keeper[key] = row
            candidates.append(row)

    snapshot = {
        "schema": "hfw_w560_partial_non_authoritative_mps_snapshot_v1",
        "status": "SEALED_FIXED_PARTIAL_DOWNLOAD_SET",
        # Bind retries to the immutable downloaded artifacts instead of wall time.
        "created_unix_ns": max(
            max(int(row["result"]["mtime_ns"]), int(row["bundle"]["mtime_ns"]))
            for row in workers
        ),
        "package_root": str(package_root),
        "wave_root": str(wave_root),
        "campaign_sha256": campaign_identity["sha256"],
        "base_head_record_sha256": campaign["base_head_record_sha256"],
        "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
        "base_model_state_digest": campaign["base_model_state_digest"],
        "current_pointer_identity": current_identity,
        "downloaded_worker_count": len(workers),
        "downloaded_task_indexes": [row["task_index"] for row in workers],
        "missing_task_indexes": sorted(set(range(int(campaign["worker_count"]))) - {row["task_index"] for row in workers}),
        "accepted_candidate_count_before_exact_dedupe": sum(row["accepted_candidate_count"] for row in workers),
        "unique_candidate_count": len(candidates),
        "workers": workers,
        "candidates": candidates,
        "skipped_exact_within_group_duplicates": duplicates,
        "cpu_scores_provisional_and_numerically_incomparable_to_mps_bf16": True,
        "local_verify_reused_optimization_data_not_held_out": True,
        "authoritative_promotion_permitted": False,
        "checkpoint_or_head_writes_permitted": False,
    }
    snapshot["snapshot_sha256"] = _sha256_json(snapshot)
    snapshot_path = output_root / "SNAPSHOT.json"
    if snapshot_path.exists():
        existing = json.loads(snapshot_path.read_text())
        declared_sha = existing.get("snapshot_sha256")
        existing_payload = dict(existing)
        existing_payload.pop("snapshot_sha256", None)
        if not isinstance(declared_sha, str) or _sha256_json(existing_payload) != declared_sha:
            raise RuntimeError("existing fixed snapshot self-identity drift")
        comparable_existing = dict(existing_payload)
        comparable_new = dict(snapshot)
        comparable_new.pop("snapshot_sha256", None)
        comparable_existing.pop("created_unix_ns", None)
        comparable_new.pop("created_unix_ns", None)
        if comparable_existing != comparable_new:
            raise RuntimeError("existing fixed snapshot differs from current immutable downloads")
        return existing
    _atomic_json(snapshot_path, snapshot)
    return snapshot


def _reusable_candidate_result(
    *,
    result_path: Path,
    candidate: dict[str, object],
    snapshot_sha256: str,
    authoritative: bool,
    observed_region_identity: dict[str, object],
    thresholds: list[float],
    tolerance: float,
) -> dict[str, object] | None:
    if not result_path.exists():
        return None
    row = json.loads(result_path.read_text())
    expected_schema = (
        "hfw_w560_operator_stopped_authoritative_candidate_mps_verify_v1"
        if authoritative
        else "hfw_w560_partial_non_authoritative_candidate_mps_verify_v1"
    )
    expected_extra_keys = {
        "schema",
        "snapshot_sha256",
        "region_identity",
        "mps_verify_per_sequence_first",
        "mps_verify_per_sequence_second",
        "mps_verify_mean_first",
        "mps_verify_mean_second",
        "mps_verify_replicated_mean",
        "duplicate_max_abs_sequence_difference",
        "passes_minimum_replicated_improvement",
        "authoritative_promotion_permitted",
        "checkpoint_or_head_writes",
        "local_verify_reused_optimization_data_not_held_out",
    }
    if set(row) != set(candidate) | expected_extra_keys:
        raise RuntimeError(f"existing candidate result key inventory drift: {result_path}")
    if any(row.get(key) != value for key, value in candidate.items()):
        raise RuntimeError(f"existing candidate result provenance drift: {result_path}")
    if (
        row.get("schema") != expected_schema
        or row.get("snapshot_sha256") != snapshot_sha256
        or row.get("region_identity") != observed_region_identity
        or row.get("authoritative_promotion_permitted") is not authoritative
        or int(row.get("checkpoint_or_head_writes", -1)) != 0
        or row.get("local_verify_reused_optimization_data_not_held_out") is not True
    ):
        raise RuntimeError(f"existing candidate result authority binding drift: {result_path}")
    first = row.get("mps_verify_per_sequence_first")
    second = row.get("mps_verify_per_sequence_second")
    if (
        not isinstance(first, list)
        or not isinstance(second, list)
        or len(first) == 0
        or len(first) != len(second)
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in first + second)
    ):
        raise RuntimeError(f"existing candidate result loss inventory drift: {result_path}")
    mean_first = math.fsum(first) / len(first)
    mean_second = math.fsum(second) / len(second)
    duplicate = max(abs(a - b) for a, b in zip(first, second))
    replicated = math.fsum([mean_first, mean_second]) / 2
    passes = duplicate <= tolerance and mean_first < thresholds[0] and mean_second < thresholds[1]
    if (
        row.get("mps_verify_mean_first") != mean_first
        or row.get("mps_verify_mean_second") != mean_second
        or row.get("mps_verify_replicated_mean") != replicated
        or row.get("duplicate_max_abs_sequence_difference") != duplicate
        or row.get("passes_minimum_replicated_improvement") is not passes
    ):
        raise RuntimeError(f"existing candidate result numerical derivation drift: {result_path}")
    return row


def _rebind_reusable_candidate_result(
    *,
    source_result_path: Path,
    destination_result_path: Path,
    candidate: dict[str, object],
    snapshot_sha256: str,
    authoritative: bool,
    observed_region_identity: dict[str, object],
    source_thresholds: list[float],
    destination_thresholds: list[float],
    tolerance: float,
) -> dict[str, object] | None:
    source = _reusable_candidate_result(
        result_path=source_result_path,
        candidate=candidate,
        snapshot_sha256=snapshot_sha256,
        authoritative=authoritative,
        observed_region_identity=observed_region_identity,
        thresholds=source_thresholds,
        tolerance=tolerance,
    )
    if source is None:
        return None
    first = source["mps_verify_per_sequence_first"]
    second = source["mps_verify_per_sequence_second"]
    duplicate = float(source["duplicate_max_abs_sequence_difference"])
    passes = (
        duplicate <= tolerance
        and float(source["mps_verify_mean_first"]) < destination_thresholds[0]
        and float(source["mps_verify_mean_second"]) < destination_thresholds[1]
    )
    rebound = {
        **source,
        "passes_minimum_replicated_improvement": passes,
    }
    _atomic_json(destination_result_path, rebound)
    return rebound


def main() -> None:
    args = _parse_args()
    package_root = args.package_root.resolve()
    wave_root = args.wave_root.resolve()
    output_root = args.output_root.resolve()
    if (
        args.minimum_relative_improvement_override is not None
        and (
            not math.isfinite(args.minimum_relative_improvement_override)
            or args.minimum_relative_improvement_override < 0
        )
    ):
        raise RuntimeError("minimum relative improvement override must be finite and nonnegative")
    reuse_verification_root = (
        None
        if args.reuse_verification_root is None
        else args.reuse_verification_root.resolve()
    )
    successor_snapshot_root = (
        None
        if args.successor_snapshot_root is None
        else args.successor_snapshot_root.resolve()
    )
    if reuse_verification_root == output_root:
        raise RuntimeError("reuse verification root must differ from output root")
    if args.promote_after_evaluated_count is not None:
        if not args.promote_after_exhaustive or args.promote_after_evaluated_count <= 0:
            raise RuntimeError(
                "promote-after-evaluated-count requires promotion mode and a positive count"
            )
    if args.promote_first_strict_improver and not args.promote_after_exhaustive:
        raise RuntimeError("promote-first-strict-improver requires promotion mode")
    if args.promote_first_strict_improver and args.promote_after_evaluated_count is not None:
        raise RuntimeError("first-improver mode is incompatible with a fixed count boundary")
    if args.start_candidate_order <= 0:
        raise RuntimeError("start candidate order must be positive")
    if successor_snapshot_root is not None and reuse_verification_root is not None:
        raise RuntimeError("successor scoring cannot reuse scores from a different parent")
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = package_root / "runtime"
    sys.path.insert(0, str(runtime_root))

    from hfw_dist.common import (
        canonical_json_bytes,
        exclusive_lock,
        hash_file,
        hash_file_region,
        read_file_region,
        read_json,
        sha256_bytes,
        write_exclusive,
    )
    from hfw_dist.model_io import build_loaded_parent
    from hfw_dist.ledger import DistributedLedger, HEAD_SCHEMA
    from hfw_dist.promote_mps import (
        MPSDuplicateVerifyObjective,
        _load_vendor,
        _mean,
        _validate_accepted_decision_artifacts,
        _write_checkpoint_atomic,
    )

    if not torch.backends.mps.is_available():
        raise RuntimeError("canonical MPS device is unavailable")
    if (wave_root / "COLLECTION_COMPLETE.json").exists() or (wave_root / "MPS_DECISION.json").exists():
        raise RuntimeError("official collection/verification already started; partial verifier refused")

    api = {
        "canonical_json_bytes": canonical_json_bytes,
        "hash_file": hash_file,
        "hash_file_region": hash_file_region,
        "read_json": read_json,
        "sha256_bytes": sha256_bytes,
    }
    if successor_snapshot_root is None:
        snapshot = _snapshot_candidates(package_root, wave_root, output_root, api)
    else:
        source_snapshot = json.loads(
            (successor_snapshot_root / "SNAPSHOT.json").read_text()
        )
        source_declared_sha = source_snapshot.get("snapshot_sha256")
        source_payload = dict(source_snapshot)
        source_payload.pop("snapshot_sha256", None)
        if (
            not isinstance(source_declared_sha, str)
            or _sha256_json(source_payload) != source_declared_sha
        ):
            raise RuntimeError("successor source snapshot self-identity drift")
        successor_head, successor_current_identity = _head(package_root, read_json)
        successor_candidates = [
            row
            for row in source_snapshot["candidates"]
            if int(row["candidate_order"]) >= args.start_candidate_order
        ]
        if not successor_candidates:
            raise RuntimeError("successor snapshot candidate set is empty")
        snapshot = {
            **source_snapshot,
            "schema": "hfw_w560_operator_checkpoint_successor_mps_snapshot_v1",
            "status": "SEALED_FIXED_SUCCESSOR_CANDIDATE_SET",
            "source_snapshot_sha256": source_declared_sha,
            "source_candidate_count": int(source_snapshot["unique_candidate_count"]),
            "start_candidate_order": args.start_candidate_order,
            "base_head_record_sha256": successor_head["record_sha256"],
            "base_checkpoint_sha256": successor_head["checkpoint"]["sha256"],
            "base_model_state_digest": successor_head["model_state_digest"],
            "current_pointer_identity": successor_current_identity,
            "unique_candidate_count": len(successor_candidates),
            "candidates": successor_candidates,
            "authoritative_promotion_permitted": True,
            "checkpoint_or_head_writes_permitted": True,
        }
        snapshot.pop("snapshot_sha256", None)
        snapshot["snapshot_sha256"] = _sha256_json(snapshot)
        _atomic_json(output_root / "SNAPSHOT.json", snapshot)
    if reuse_verification_root is not None:
        prior_snapshot = json.loads(
            (reuse_verification_root / "SNAPSHOT.json").read_text()
        )
        prior_declared_sha = prior_snapshot.get("snapshot_sha256")
        prior_payload = dict(prior_snapshot)
        prior_payload.pop("snapshot_sha256", None)
        if (
            not isinstance(prior_declared_sha, str)
            or _sha256_json(prior_payload) != prior_declared_sha
        ):
            raise RuntimeError("reuse snapshot self-identity drift")
        comparable_prior = dict(prior_payload)
        comparable_current = dict(snapshot)
        comparable_current.pop("snapshot_sha256", None)
        comparable_prior.pop("created_unix_ns", None)
        comparable_current.pop("created_unix_ns", None)
        if comparable_prior != comparable_current:
            raise RuntimeError("reuse snapshot candidate inventory differs")
        snapshot = prior_snapshot
        _atomic_json(output_root / "SNAPSHOT.json", snapshot)
    print(
        f"SNAPSHOT workers={snapshot['downloaded_worker_count']} "
        f"accepted={snapshot['accepted_candidate_count_before_exact_dedupe']} "
        f"unique={snapshot['unique_candidate_count']}",
        flush=True,
    )

    campaign, campaign_identity = read_json(wave_root / "snapshot/CAMPAIGN.json")
    frozen = package_root / "runtime/frozen"
    contract, _ = read_json(package_root / "runtime/authority/lineage/contract.json")
    ledger = DistributedLedger(package_root / "runtime/authority/lineage")
    lock_path = Path(str(contract["host_mps_lock_path"]))
    if str(campaign["promotion"]["host_mps_lock_path"]) != str(lock_path):
        raise RuntimeError("campaign/authority MPS lock drift")

    cleanup = cleanup_identity = None
    if args.promote_after_exhaustive:
        cleanup, cleanup_identity = _validated_operator_stop_cleanup(
            wave_root, str(campaign_identity["sha256"]), read_json, hash_file,
        )

    with ExitStack() as stack:
        stack.enter_context(exclusive_lock(lock_path))
        if args.promote_after_exhaustive:
            stack.enter_context(ledger.lock())
        if not args.promote_after_exhaustive and (wave_root / "CLOUD_DOWNLOADS_COMPLETE.json").exists():
            print("STOP official cloud download terminal exists; yielding to final exhaustive verifier", flush=True)
            return
        head = ledger.current_head() if args.promote_after_exhaustive else _head(package_root, read_json)[0]
        if head["record_sha256"] != snapshot["base_head_record_sha256"]:
            raise RuntimeError("head advanced before partial MPS verification")
        collection_identity = shortlist_identity = intent_identity = None
        if args.promote_after_exhaustive:
            assert cleanup is not None and cleanup_identity is not None
            operator_collection = {
                "schema": "hfw_w560_operator_stopped_daytona_collection_v1",
                "status": "PASS_ALL_COMPLETED_DOWNLOADS_PRESERVED_AND_CLOUD_CLEANED",
                "campaign_sha256": campaign_identity["sha256"],
                "base_head_record_sha256": head["record_sha256"],
                "base_checkpoint_sha256": head["checkpoint"]["sha256"],
                "operator_stop_reason": "USER_REQUESTED_STOP_REMAINING_PROPOSAL_SEARCH_AND_PROMOTE_COMPLETED_SET",
                "downloaded_worker_count": snapshot["downloaded_worker_count"],
                "missing_task_indexes": snapshot["missing_task_indexes"],
                "accepted_candidate_count_before_exact_dedupe": snapshot[
                    "accepted_candidate_count_before_exact_dedupe"
                ],
                "unique_candidate_count": snapshot["unique_candidate_count"],
                "fixed_snapshot_sha256": snapshot["snapshot_sha256"],
                "cleanup_terminal_receipt": cleanup_identity,
                "cleanup_zero_exact_campaign_sandboxes": True,
                "cleanup_exact_snapshot_absent": True,
                "cloud_search_intentionally_terminated_before_all_100_workers": True,
                "cloud_checkpoint_or_head_writes": 0,
            }
            collection_identity = ledger.append_named_record(
                "collections", ".collection.json", operator_collection,
            )
            operator_shortlist = {
                "schema": "hfw_w560_operator_stopped_all_completed_candidates_v1",
                "status": "PASS_EVERY_UNIQUE_CANDIDATE_FROM_EVERY_COMPLETED_WORKER_SEALED",
                "campaign_sha256": campaign_identity["sha256"],
                "collection_sha256": collection_identity["sha256"],
                "base_head_record_sha256": head["record_sha256"],
                "base_checkpoint_sha256": head["checkpoint"]["sha256"],
                "fixed_snapshot_sha256": snapshot["snapshot_sha256"],
                "downloaded_worker_count": snapshot["downloaded_worker_count"],
                "missing_task_indexes": snapshot["missing_task_indexes"],
                "candidate_count": snapshot["unique_candidate_count"],
                "accepted_candidate_count_before_exact_dedupe": snapshot[
                    "accepted_candidate_count_before_exact_dedupe"
                ],
                "candidates": snapshot["candidates"],
                "skipped_exact_within_group_duplicates": snapshot[
                    "skipped_exact_within_group_duplicates"
                ],
                "every_completed_worker_candidate_downloaded": True,
                "every_unique_completed_worker_candidate_selected_for_mps_verify": True,
                "cpu_ranking_filter_applied": False,
                "cpu_scores_used_for_selection": False,
                "operator_stopped_before_all_campaign_workers_completed": True,
            }
            shortlist_identity = ledger.append_named_record(
                "shortlists", ".shortlist.json", operator_shortlist,
            )
            intent_identity = ledger.append_named_record(
                "promotion_intents", ".intent.json", {
                    "schema": "hfw_w560_operator_stopped_partial_promotion_intent_v1",
                    "status": "SEALED_BEFORE_CANONICAL_MPS_VERIFY",
                    "campaign_sha256": campaign_identity["sha256"],
                    "collection_sha256": collection_identity["sha256"],
                    "shortlist_sha256": shortlist_identity["sha256"],
                    "base_head_record_sha256": head["record_sha256"],
                    "base_checkpoint_sha256": head["checkpoint"]["sha256"],
                    "candidate_count": snapshot["unique_candidate_count"],
                    "at_most_one_checkpoint_and_head_append": True,
                    "cpu_ranking_filter_applied": False,
                    "every_candidate_must_be_evaluated_twice": True,
                    "retain_every_passing_candidate_in_authoritative_decision": True,
                    "promotion_margin_source": (
                        "OPERATOR_OVERRIDE"
                        if args.minimum_relative_improvement_override is not None
                        else "SEALED_CAMPAIGN"
                    ),
                    "minimum_relative_improvement": (
                        float(args.minimum_relative_improvement_override)
                        if args.minimum_relative_improvement_override is not None
                        else float(campaign["promotion"]["minimum_relative_improvement"])
                    ),
                    "strict_parent_improvement_required": True,
                },
            )
        branch, bridge, vendor_manifest = _load_vendor(frozen)
        loaded, _ = build_loaded_parent(
            frozen,
            Path(str(head["checkpoint"]["path"])),
            str(head["checkpoint"]["sha256"]),
            str(head["model_state_digest"]),
            torch.device("mps"),
            bridge,
            vendor_manifest,
        )
        verify_shape = tuple(int(value) for value in campaign["promotion"]["verify_bank_shape"])
        verify_identity = hash_file(
            frozen / "verify.u32le", str(campaign["promotion"]["verify_bank_sha256"])
        )
        expected_verify_bytes = math.prod(verify_shape) * np.dtype("<u4").itemsize
        if int(verify_identity["size_bytes"]) != expected_verify_bytes:
            raise RuntimeError("VERIFY shape/byte-size binding drift")
        verify = np.memmap(frozen / "verify.u32le", mode="r", dtype="<u4", shape=verify_shape)
        batches = [
            torch.from_numpy(np.array(verify[index], dtype=np.int64, copy=True))
            for index in range(verify_shape[0])
        ]
        objective = MPSDuplicateVerifyObjective(loaded, batches)
        parent_state = objective.parent_state()
        parent_first = objective.sequence_losses(parent_state)
        parent_second = objective.sequence_losses(parent_state)
        tolerance = float(campaign["promotion"]["duplicate_forward_tolerance"])
        parent_duplicate = max(abs(a - b) for a, b in zip(parent_first, parent_second))
        if parent_duplicate > tolerance:
            raise RuntimeError("duplicated parent MPS VERIFY drift")
        parent_means = [_mean(parent_first), _mean(parent_second)]
        campaign_minimum_relative = float(
            campaign["promotion"]["minimum_relative_improvement"]
        )
        minimum_relative = (
            campaign_minimum_relative
            if args.minimum_relative_improvement_override is None
            else float(args.minimum_relative_improvement_override)
        )
        thresholds = [value - abs(value) * minimum_relative for value in parent_means]
        promotion_policy = (
            "STRICT_DUPLICATED_CANONICAL_IMPROVEMENT_OVER_PARENT_ZERO_MARGIN"
            if minimum_relative == 0.0
            else "DUPLICATED_CANONICAL_MINIMUM_RELATIVE_IMPROVEMENT"
        )
        parent_record = {
            "schema": "hfw_w560_partial_non_authoritative_parent_mps_verify_v1",
            "campaign_sha256": campaign_identity["sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "base_head_record_sha256": head["record_sha256"],
            "base_checkpoint_sha256": head["checkpoint"]["sha256"],
            "verify_bank": verify_identity,
            "per_sequence_first": parent_first,
            "per_sequence_second": parent_second,
            "mean_first": parent_means[0],
            "mean_second": parent_means[1],
            "duplicate_max_abs_sequence_difference": parent_duplicate,
            "thresholds": thresholds,
            "minimum_relative_improvement": minimum_relative,
            "campaign_minimum_relative_improvement": campaign_minimum_relative,
            "promotion_margin_source": (
                "OPERATOR_OVERRIDE"
                if args.minimum_relative_improvement_override is not None
                else "SEALED_CAMPAIGN"
            ),
            "promotion_policy": promotion_policy,
            "retain_every_passing_candidate": True,
            "duplicate_forward_tolerance": tolerance,
            "authoritative_promotion_permitted": bool(args.promote_after_exhaustive),
        }
        _atomic_json(output_root / "PARENT.json", parent_record)
        print(
            f"PARENT mean1={parent_means[0]:.15f} mean2={parent_means[1]:.15f} "
            f"duplicate={parent_duplicate:.3g}",
            flush=True,
        )

        reuse_source_parent = None
        reuse_source_thresholds = None
        if reuse_verification_root is not None:
            reuse_source_parent, _ = read_json(
                reuse_verification_root / "PARENT.json"
            )
            invariant_keys = (
                "campaign_sha256",
                "snapshot_sha256",
                "base_head_record_sha256",
                "base_checkpoint_sha256",
                "verify_bank",
                "per_sequence_first",
                "per_sequence_second",
                "mean_first",
                "mean_second",
                "duplicate_max_abs_sequence_difference",
                "duplicate_forward_tolerance",
                "authoritative_promotion_permitted",
            )
            if any(
                reuse_source_parent.get(key) != parent_record.get(key)
                for key in invariant_keys
            ):
                raise RuntimeError(
                    "reuse verification parent differs from current canonical parent"
                )
            reuse_source_thresholds = [
                float(value) for value in reuse_source_parent["thresholds"]
            ]

        completed = 0
        reused = 0
        imported = 0
        passing = 0
        best: dict[str, object] | None = None
        candidate_rows: list[dict[str, object]] = []
        stop_reason = (
            "OPERATOR_STOPPED_COMPLETED_WORKER_SNAPSHOT_EXHAUSTED"
            if args.promote_after_exhaustive
            else "PARTIAL_SNAPSHOT_EXHAUSTED"
        )
        for candidate in snapshot["candidates"]:
            if (
                args.promote_after_evaluated_count is not None
                and completed >= args.promote_after_evaluated_count
            ):
                stop_reason = "OPERATOR_CHECKPOINT_BOUNDARY_REACHED"
                break
            if args.max_candidates is not None and completed >= args.max_candidates:
                stop_reason = "MAX_CANDIDATES_REACHED"
                break
            if not args.promote_after_exhaustive and (wave_root / "CLOUD_DOWNLOADS_COMPLETE.json").exists():
                stop_reason = "YIELDED_TO_FINAL_EXHAUSTIVE_VERIFIER_AFTER_CLOUD_DOWNLOAD_TERMINAL"
                break
            # The authoritative run holds ledger.lock() for the entire scoring loop,
            # so its head cannot change here. Re-hashing the 400 MB checkpoint for
            # every candidate would add hundreds of redundant full-file reads.
            current_head = head if args.promote_after_exhaustive else _head(package_root, read_json)[0]
            if current_head["record_sha256"] != snapshot["base_head_record_sha256"]:
                stop_reason = "HEAD_ADVANCED_STOPPED_WITHOUT_WRITES"
                break
            payload, identity = read_file_region(
                Path(str(candidate["bundle_path"])),
                int(candidate["offset_bytes"]),
                int(candidate["size_bytes"]),
                str(candidate["candidate_group_flat_sha256"]),
            )
            result_path = output_root / "candidate-results" / (
                f"{int(candidate['candidate_order']):06d}-{candidate['candidate_group_flat_sha256']}.json"
            )
            reusable = _reusable_candidate_result(
                result_path=result_path,
                candidate=candidate,
                snapshot_sha256=str(snapshot["snapshot_sha256"]),
                authoritative=bool(args.promote_after_exhaustive),
                observed_region_identity=identity,
                thresholds=thresholds,
                tolerance=tolerance,
            )
            imported_this_candidate = False
            if reusable is None and reuse_verification_root is not None:
                assert reuse_source_thresholds is not None
                source_result_path = (
                    reuse_verification_root
                    / "candidate-results"
                    / result_path.name
                )
                reusable = _rebind_reusable_candidate_result(
                    source_result_path=source_result_path,
                    destination_result_path=result_path,
                    candidate=candidate,
                    snapshot_sha256=str(snapshot["snapshot_sha256"]),
                    authoritative=bool(args.promote_after_exhaustive),
                    observed_region_identity=identity,
                    source_thresholds=reuse_source_thresholds,
                    destination_thresholds=thresholds,
                    tolerance=tolerance,
                )
                imported_this_candidate = reusable is not None
            if reusable is not None:
                candidate_rows.append(reusable)
                completed += 1
                reused += 1
                imported += int(imported_this_candidate)
                passes = bool(reusable["passes_minimum_replicated_improvement"])
                passing += int(passes)
                if passes and (
                    best is None
                    or (reusable["mps_verify_replicated_mean"], reusable["candidate_order"])
                    < (best["mps_verify_replicated_mean"], best["candidate_order"])
                ):
                    best = reusable
                print(
                    f"{'IMPORTED' if imported_this_candidate else 'REUSED'} "
                    f"{completed}/{snapshot['unique_candidate_count']} "
                    f"task={candidate['task_index']} group={candidate['group']} "
                    f"mean={reusable['mps_verify_replicated_mean']:.15f} pass={passes} "
                    f"passing={passing}",
                    flush=True,
                )
                del payload
                if passes and args.promote_first_strict_improver:
                    stop_reason = "FIRST_STRICT_DUPLICATED_IMPROVER_REACHED"
                    break
                continue
            values = np.frombuffer(payload, dtype="<f4")
            group = str(candidate["group"])
            expected_elements = int(campaign["group_elements"][group])
            if values.size != expected_elements or not np.isfinite(values).all():
                raise RuntimeError("candidate element count or finiteness drift")
            candidate_flat = torch.from_numpy(values.copy()).to("mps")
            candidate_state = objective.state_with_group(group, candidate_flat)
            first = objective.sequence_losses(candidate_state)
            second = objective.sequence_losses(candidate_state)
            duplicate = max(abs(a - b) for a, b in zip(first, second))
            means = [_mean(first), _mean(second)]
            passes = (
                duplicate <= tolerance
                and means[0] < thresholds[0]
                and means[1] < thresholds[1]
            )
            row = {
                "schema": (
                    "hfw_w560_operator_stopped_authoritative_candidate_mps_verify_v1"
                    if args.promote_after_exhaustive
                    else "hfw_w560_partial_non_authoritative_candidate_mps_verify_v1"
                ),
                "snapshot_sha256": snapshot["snapshot_sha256"],
                **candidate,
                "region_identity": identity,
                "mps_verify_per_sequence_first": first,
                "mps_verify_per_sequence_second": second,
                "mps_verify_mean_first": means[0],
                "mps_verify_mean_second": means[1],
                "mps_verify_replicated_mean": math.fsum(means) / 2,
                "duplicate_max_abs_sequence_difference": duplicate,
                "passes_minimum_replicated_improvement": passes,
                "authoritative_promotion_permitted": bool(args.promote_after_exhaustive),
                "checkpoint_or_head_writes": 0,
                "local_verify_reused_optimization_data_not_held_out": True,
            }
            _atomic_json(result_path, row)
            candidate_rows.append(row)
            completed += 1
            passing += int(passes)
            if passes and (
                best is None
                or (row["mps_verify_replicated_mean"], row["candidate_order"])
                < (best["mps_verify_replicated_mean"], best["candidate_order"])
            ):
                best = row
            print(
                f"CANDIDATE {completed}/{snapshot['unique_candidate_count']} "
                f"task={candidate['task_index']} group={group} "
                f"mean={row['mps_verify_replicated_mean']:.15f} pass={passes} "
                f"passing={passing}",
                flush=True,
            )
            del payload, values, candidate_flat, candidate_state, first, second
            torch.mps.empty_cache()
            if passes and args.promote_first_strict_improver:
                stop_reason = "FIRST_STRICT_DUPLICATED_IMPROVER_REACHED"
                break

        improving_candidates = []
        for row in candidate_rows:
            if row["passes_minimum_replicated_improvement"] is not True:
                continue
            result_path = output_root / "candidate-results" / (
                f"{int(row['candidate_order']):06d}-{row['candidate_group_flat_sha256']}.json"
            )
            improving_candidates.append({
                "candidate_order": int(row["candidate_order"]),
                "task_index": int(row["task_index"]),
                "worker_candidate_order": int(row["worker_candidate_order"]),
                "group": str(row["group"]),
                "candidate_group_flat_sha256": str(
                    row["candidate_group_flat_sha256"]
                ),
                "mps_verify_mean_first": float(row["mps_verify_mean_first"]),
                "mps_verify_mean_second": float(row["mps_verify_mean_second"]),
                "mps_verify_replicated_mean": float(
                    row["mps_verify_replicated_mean"]
                ),
                "duplicate_max_abs_sequence_difference": float(
                    row["duplicate_max_abs_sequence_difference"]
                ),
                "candidate_result": hash_file(result_path),
            })
        improving_candidates.sort(
            key=lambda row: (
                row["mps_verify_replicated_mean"], row["candidate_order"]
            )
        )
        improving_artifact = {
            "schema": "hfw_w560_operator_stopped_improving_candidates_v1",
            "status": (
                "EXHAUSTIVE_STRICT_IMPROVERS_RETAINED"
                if completed == int(snapshot["unique_candidate_count"])
                else "PARTIAL_STRICT_IMPROVERS_RETAINED"
            ),
            "campaign_sha256": campaign_identity["sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "base_head_record_sha256": head["record_sha256"],
            "base_checkpoint_sha256": head["checkpoint"]["sha256"],
            "promotion_policy": promotion_policy,
            "minimum_relative_improvement": minimum_relative,
            "parent_mean_first": parent_means[0],
            "parent_mean_second": parent_means[1],
            "duplicate_forward_tolerance": tolerance,
            "candidate_count_evaluated_twice": completed,
            "improving_candidate_count": len(improving_candidates),
            "all_improving_candidates_retained": True,
            "candidates": improving_candidates,
        }
        improving_artifact_path = output_root / "IMPROVING_CANDIDATES.json"
        _atomic_json(improving_artifact_path, improving_artifact)
        improving_artifact_identity = hash_file(improving_artifact_path)

        promotion_record = None
        if args.promote_after_exhaustive:
            if (
                completed != int(snapshot["unique_candidate_count"])
                and args.promote_after_evaluated_count is None
                and not args.promote_first_strict_improver
            ):
                raise RuntimeError("authoritative operator-stop promotion requires exhaustive scoring")
            assert collection_identity is not None
            assert shortlist_identity is not None
            assert intent_identity is not None
            child_identity = delta_identity = replacement_identity = None
            model_digest = None
            if best is not None:
                selected_payload, selected_region = read_file_region(
                    Path(str(best["bundle_path"])),
                    int(best["offset_bytes"]),
                    int(best["size_bytes"]),
                    str(best["candidate_group_flat_sha256"]),
                )
                group = str(best["group"])
                selected_values = np.frombuffer(selected_payload, dtype="<f4")
                if (
                    selected_values.size != int(campaign["group_elements"][group])
                    or not np.isfinite(selected_values).all()
                ):
                    raise RuntimeError("selected candidate flat drift before checkpoint materialization")
                selected_flat_cpu = torch.from_numpy(selected_values.copy())
                if hashlib.sha256(branch._tensor_bytes(selected_flat_cpu)).hexdigest() != best[
                    "candidate_group_flat_sha256"
                ]:
                    raise RuntimeError("selected candidate CPU serialization drift")
                base_cpu = objective.flat_group(group).detach().cpu().float()
                delta_payload = branch._tensor_bytes(selected_flat_cpu - base_cpu)
                delta_sha = hashlib.sha256(delta_payload).hexdigest()
                delta_path = ledger.root / "deltas" / f"{delta_sha}.group_delta.f32le"
                delta_identity = (
                    hash_file(delta_path, delta_sha)
                    if delta_path.exists()
                    else write_exclusive(delta_path, delta_payload)
                )
                replacement_path = ledger.root / "replacement_group_flats" / (
                    f"{best['candidate_group_flat_sha256']}.group_flat.f32le"
                )
                replacement_identity = (
                    hash_file(replacement_path, str(best["candidate_group_flat_sha256"]))
                    if replacement_path.exists()
                    else write_exclusive(replacement_path, selected_payload)
                )
                selected_state_mps = objective.state_with_group(
                    group, selected_flat_cpu.to("mps"),
                )
                selected_state_cpu = {
                    name: value.detach().cpu() for name, value in selected_state_mps.items()
                }
                exact_roundtrip = torch.cat([
                    selected_state_cpu[name].reshape(-1) for name in loaded.groups[group]
                ])
                if hashlib.sha256(branch._tensor_bytes(exact_roundtrip)).hexdigest() != replacement_identity[
                    "sha256"
                ]:
                    raise RuntimeError("selected checkpoint group roundtrip drift")
                model_digest = bridge.model_state_digest(selected_state_cpu)
                child_generation = int(head["model_generation"]) + 1
                checkpoint_payload = {
                    "schema": "hfw_w560_continuous_promoted_checkpoint_v1",
                    "origin_checkpoint_sha256": head["origin_checkpoint_sha256"],
                    "parent_checkpoint_sha256": head["checkpoint"]["sha256"],
                    "generation": child_generation,
                    "accepted_proposal_sha256": shortlist_identity["sha256"],
                    "selected_candidate_order": int(best["candidate_order"]),
                    "selected_candidate_delta_sha256": delta_identity["sha256"],
                    "solver_cumulative_delta_sha256": delta_identity["sha256"],
                    "replacement_group_flat_sha256": replacement_identity["sha256"],
                    "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
                    "solver_cumulative_delta_used_to_reconstruct_group": False,
                    "model_state_digest": model_digest,
                    "optimizer_present": False,
                    "scheduler_present": False,
                    "training_resume_authorized": False,
                    "model": selected_state_cpu,
                }
                checkpoint_path = ledger.root / "promoted_checkpoints" / (
                    f"generation-{child_generation:08d}-{shortlist_identity['sha256'][:16]}.pt"
                )
                child_identity = _write_checkpoint_atomic(
                    checkpoint_path, checkpoint_payload, bridge,
                )
                best = {**best, "selected_region_identity": selected_region}
                del (
                    selected_payload, selected_values, selected_flat_cpu, base_cpu,
                    delta_payload, selected_state_mps, selected_state_cpu,
                    exact_roundtrip, checkpoint_payload,
                )
                torch.mps.empty_cache()

            decision = {
                "schema": "hfw_w560_operator_stopped_partial_mps_verify_decision_v1",
                "status": (
                    (
                        "ACCEPTED_BEST_EXHAUSTIVELY_VERIFIED_COMPLETED_WORKER_CANDIDATE"
                        if completed == int(snapshot["unique_candidate_count"])
                        else "ACCEPTED_STRICTLY_VERIFIED_OPERATOR_CHECKPOINT_CANDIDATE"
                    )
                    if best is not None
                    else "REJECTED_NO_EVALUATED_CANDIDATE_PASSED_DUPLICATED_MPS_VERIFY"
                ),
                "accepted": best is not None,
                "campaign_sha256": campaign_identity["sha256"],
                "collection_sha256": collection_identity["sha256"],
                "shortlist_sha256": shortlist_identity["sha256"],
                "promotion_intent_sha256": intent_identity["sha256"],
                "base_head_record_sha256": head["record_sha256"],
                "base_checkpoint_sha256": head["checkpoint"]["sha256"],
                "operator_stopped_before_all_campaign_workers_completed": True,
                "downloaded_worker_count": snapshot["downloaded_worker_count"],
                "missing_task_indexes": snapshot["missing_task_indexes"],
                "promotion_policy": promotion_policy,
                "minimum_relative_improvement": minimum_relative,
                "campaign_minimum_relative_improvement": campaign_minimum_relative,
                "improving_candidate_count": len(improving_candidates),
                "all_improving_candidates_retained": True,
                "improving_candidates_artifact": improving_artifact_identity,
                "local_verify": {
                    "role": "VERIFY",
                    "disclosure": "PERMANENT_REUSED_SELECTION_DATA_NOT_HELD_OUT",
                    "held_out_generalization_claimed": False,
                    "bank_sha256": verify_identity["sha256"],
                    "parent_per_sequence_first": parent_first,
                    "parent_per_sequence_second": parent_second,
                    "parent_mean_first": parent_means[0],
                    "parent_mean_second": parent_means[1],
                    "parent_duplicate_max_abs_sequence_difference": parent_duplicate,
                    "thresholds": thresholds,
                    "minimum_relative_improvement": minimum_relative,
                    "campaign_minimum_relative_improvement": campaign_minimum_relative,
                    "promotion_policy": promotion_policy,
                    "duplicate_forward_tolerance": tolerance,
                    "candidate_results": candidate_rows,
                    "strict_improving_candidates": improving_candidates,
                    "improving_candidates_artifact": improving_artifact_identity,
                    "all_completed_worker_unique_candidates_evaluated_twice": (
                        completed == int(snapshot["unique_candidate_count"])
                    ),
                    "mixed_groups_evaluated_as_independent_replacements": True,
                    "cross_group_composition_performed": False,
                    "runtime": objective.runtime_receipt(),
                },
                "selected_candidate": best,
                "child_checkpoint": child_identity,
                "child_model_state_digest": model_digest,
                "ledger_delta": delta_identity,
                "ledger_replacement_group_flat": replacement_identity,
                "checkpoint_materialized_from_exact_group_flat": best is not None,
                "training_or_backprop": False,
                "optimizer_steps": 0,
            }
            decision_identity = ledger.append_named_record(
                "decisions", ".decision.json", decision,
            )
            head_after = head
            if best is not None:
                if ledger.current_head()["record_sha256"] != snapshot["base_head_record_sha256"]:
                    raise RuntimeError("head changed inside serialized operator-stop promoter")
                _validate_accepted_decision_artifacts(
                    ledger, frozen, decision, head, str(shortlist_identity["sha256"]),
                )
                child = decision["child_checkpoint"]
                new_head = {
                    "schema": HEAD_SCHEMA,
                    "status": "PROMOTED_EXACT_MPS_VERIFIED_OPERATOR_STOPPED_DAYTONA_PROPOSAL",
                    "authority_generation": int(head["authority_generation"]) + 1,
                    "model_generation": int(head["model_generation"]) + 1,
                    "local_promotions": int(head["local_promotions"]) + 1,
                    "parent_head_record_sha256": head["record_sha256"],
                    "contract_sha256": head["contract_sha256"],
                    "source_head_record_sha256": head["source_head_record_sha256"],
                    "source_head_record_path": head["source_head_record_path"],
                    "parent_checkpoint_sha256": head["checkpoint"]["sha256"],
                    "checkpoint": {**child, "format": "hfw_w560_continuous_promoted_checkpoint_v1"},
                    "origin_checkpoint_sha256": head["origin_checkpoint_sha256"],
                    "model_state_digest": model_digest,
                    "accepted_proposal_sha256": shortlist_identity["sha256"],
                    "selected_candidate_order": int(best["candidate_order"]),
                    "group": best["group"],
                    "decision_sha256": decision_identity["sha256"],
                    "replacement_group_flat_sha256": replacement_identity["sha256"],
                    "solver_cumulative_delta_sha256": delta_identity["sha256"],
                    "campaign_sha256": campaign_identity["sha256"],
                    "shortlist_sha256": shortlist_identity["sha256"],
                }
                head_after = ledger.append_head(new_head)
            promotion_record = {
                "schema": "hfw_w560_operator_stopped_partial_promotion_terminal_v1",
                "status": decision["status"],
                "accepted": bool(decision["accepted"]),
                "campaign_sha256": campaign_identity["sha256"],
                "collection": collection_identity,
                "shortlist": shortlist_identity,
                "promotion_intent": intent_identity,
                "decision": decision_identity,
                "child_checkpoint": child_identity,
                "current_head_record_sha256": head_after["record_sha256"],
                "parent_ce": parent_means[0],
                "child_ce": None if best is None else best["mps_verify_replicated_mean"],
                "absolute_improvement": (
                    None if best is None else parent_means[0] - best["mps_verify_replicated_mean"]
                ),
                "relative_improvement": (
                    None if best is None else (parent_means[0] - best["mps_verify_replicated_mean"]) / abs(parent_means[0])
                ),
                "promotion_policy": promotion_policy,
                "minimum_relative_improvement": minimum_relative,
                "campaign_minimum_relative_improvement": campaign_minimum_relative,
                "improving_candidate_count": len(improving_candidates),
                "all_improving_candidates_retained": True,
                "improving_candidates_artifact": improving_artifact_identity,
            }
            _atomic_json(output_root / "PROMOTION.json", promotion_record)

        summary = {
            "schema": (
                "hfw_w560_operator_stopped_authoritative_mps_verify_summary_v1"
                if args.promote_after_exhaustive
                else "hfw_w560_partial_non_authoritative_mps_verify_summary_v1"
            ),
            "status": stop_reason,
            "completed_unix_ns": time.time_ns(),
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "snapshot_unique_candidate_count": snapshot["unique_candidate_count"],
            "candidate_count_evaluated_twice": completed,
            "candidate_result_files_reused_after_validated_restart": reused,
            "candidate_result_files_imported_from_prior_policy_run": imported,
            "passing_candidate_count": passing,
            "promotion_policy": promotion_policy,
            "minimum_relative_improvement": minimum_relative,
            "campaign_minimum_relative_improvement": campaign_minimum_relative,
            "improving_candidate_count": len(improving_candidates),
            "all_improving_candidates_retained": True,
            "improving_candidates_artifact": improving_artifact_identity,
            "best_provisional_passing_candidate": best,
            "all_snapshot_candidates_evaluated_twice": completed == snapshot["unique_candidate_count"],
            "runtime": objective.runtime_receipt(),
            "authoritative_promotion_permitted": bool(args.promote_after_exhaustive),
            "checkpoint_or_head_writes": int(
                bool(promotion_record and promotion_record["accepted"])
            ),
            "final_exhaustive_promotion_still_required": not bool(args.promote_after_exhaustive),
            "promotion": promotion_record,
            "cpu_scores_provisional_and_numerically_incomparable_to_mps_bf16": True,
            "local_verify_reused_optimization_data_not_held_out": True,
        }
        _atomic_json(output_root / "SUMMARY.json", summary)
        print(
            f"SUMMARY status={stop_reason} evaluated={completed} passing={passing}",
            flush=True,
        )


if __name__ == "__main__":
    main()
