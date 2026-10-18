from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

from .collect import _validate_cloud_handoff, collect_wave
from .common import hash_file, read_json, write_json_exclusive
from .daytona_manager import validate_recovered_task0_provenance
from .promote_mps import promote_wave
from .runtime_manifest import validate_host_runtime


DETERMINISTIC_FAILURE_TERMINAL_FILENAME = (
    "CLOUD_DETERMINISTIC_FAILURE_COMPLETE.json"
)
PIPELINE_DETERMINISTIC_TERMINAL_FILENAME = (
    "REAL_PIPELINE_DETERMINISTIC_FAILURE.json"
)


def _validated_deterministic_failure_terminal(
    wave_root: Path,
    campaign: Mapping[str, object],
    campaign_identity: Mapping[str, object],
) -> Mapping[str, object] | None:
    path = wave_root / DETERMINISTIC_FAILURE_TERMINAL_FILENAME
    if not path.exists():
        return None
    terminal, terminal_identity = read_json(path)
    required = {
        "schema": "hfw_w560_daytona_deterministic_failure_terminal_v1",
        "status": "FAIL_DETERMINISTIC_NUMERICAL_FAILURE_CLEANED_TERMINAL",
        "campaign_sha256": campaign_identity["sha256"],
        "snapshot_manifest_sha256": hash_file(
            wave_root / "snapshot/SNAPSHOT_MANIFEST.json"
        )["sha256"],
        "wave_index": int(campaign["wave_index"]),
        "campaign_sandboxes_remaining": 0,
        "snapshot_present": False,
        "exact_campaign_label_relist_passed": True,
        "snapshot_exact_name_not_found_confirmed": True,
        "cleanup_failures": [],
        "local_collection_or_verify_started": False,
        "checkpoint_or_head_writes": 0,
        "deterministic_numerical_attempts": 1,
        "failed_task_numerical_attempts": 1,
        "blind_numerical_retries": 0,
        "terminal_restart_requires_no_cloud_calls": True,
    }
    for key, expected in required.items():
        if terminal.get(key) != expected:
            raise RuntimeError(f"deterministic manager terminal drift: {key}")
    worker_count = int(campaign["worker_count"])
    task_index = terminal.get("task_index")
    fanout_started = terminal.get("fanout_tasks_started")
    downloads = terminal.get("cloud_results_downloaded")
    if (
        not isinstance(task_index, int)
        or isinstance(task_index, bool)
        or not 0 <= task_index < worker_count
        or not isinstance(fanout_started, int)
        or isinstance(fanout_started, bool)
        or not 0 <= fanout_started < worker_count
        or not isinstance(downloads, int)
        or isinstance(downloads, bool)
        or not 0 <= downloads <= worker_count
        or (task_index == 0 and fanout_started != 0)
        or (task_index > 0 and fanout_started == 0)
    ):
        raise RuntimeError("deterministic manager terminal activity drift")
    policy = campaign["worker_failure_policy"]
    if (
        terminal.get("classification")
        != policy["deterministic_numerical_classification"]
        or terminal.get("retry_policy")
        != policy["deterministic_numerical_retry_policy"]
        or int(terminal.get("process_exit_code", -1))
        != int(policy["deterministic_numerical_exit_code"])
        or terminal.get("stage") not in policy["allowed_stages"]
    ):
        raise RuntimeError("deterministic manager terminal policy drift")
    for key in ("worker_failure_artifact", "validated_decision"):
        artifact = terminal.get(key, {})
        relative = artifact.get("relative_path")
        if not isinstance(relative, str):
            raise RuntimeError(f"deterministic manager terminal path drift: {key}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe deterministic manager terminal path: {key}")
        identity = hash_file(wave_root / relative_path, str(artifact.get("sha256")))
        if int(identity["size_bytes"]) != int(artifact.get("size_bytes", -1)):
            raise RuntimeError(f"deterministic manager terminal size drift: {key}")
    return {**terminal, "terminal_identity": terminal_identity}


def _seal_pipeline_deterministic_terminal(
    wave_root: Path,
    campaign: Mapping[str, object],
    campaign_identity: Mapping[str, object],
    manager_terminal: Mapping[str, object],
) -> Mapping[str, object]:
    value = {
        "schema": "hfw_w560_real_pipeline_deterministic_failure_v1",
        "status": "FAIL_DETERMINISTIC_NUMERICAL_FAILURE_CLEANED_TERMINAL",
        "campaign_sha256": campaign_identity["sha256"],
        "wave_index": int(campaign["wave_index"]),
        "manager_terminal": manager_terminal["terminal_identity"],
        "worker_failure_artifact": manager_terminal["worker_failure_artifact"],
        "classification": manager_terminal["classification"],
        "stage": manager_terminal["stage"],
        "failed_task_index": manager_terminal["task_index"],
        "failed_worker_attempt": manager_terminal["worker_attempt"],
        "failed_task_numerical_attempts": manager_terminal[
            "failed_task_numerical_attempts"
        ],
        "fanout_tasks_started": manager_terminal["fanout_tasks_started"],
        "cloud_results_downloaded": manager_terminal["cloud_results_downloaded"],
        "seed_gate_completed_before_failure": manager_terminal["task_index"] > 0,
        "blind_numerical_retries": 0,
        "cloud_cleanup_passed": True,
        "cloud_calls_on_terminal_restart": False,
        "collection_or_mps_verify_started": False,
        "checkpoint_promoted": False,
        "promotion_status": "NOT_RUN_DETERMINISTIC_NUMERICAL_FAILURE",
        "current_head_record_sha256": campaign["base_head_record_sha256"],
    }
    path = wave_root / PIPELINE_DETERMINISTIC_TERMINAL_FILENAME
    if path.exists():
        existing, identity = read_json(path)
        if existing != value:
            raise RuntimeError("pipeline deterministic terminal changed across restart")
    else:
        identity = write_json_exclusive(path, value)
    return {**value, "terminal_identity": identity}


def _manager_command(args, package_root: Path, wave_root: Path) -> list[str]:
    expected_entry = os.path.abspath(os.fspath(
        package_root / "runtime/daytona-manager-venv/bin/python"
    ))
    requested_entry = expected_entry if args.manager_python is None else os.path.abspath(
        os.fspath(args.manager_python)
    )
    if requested_entry != expected_entry:
        raise RuntimeError("manager interpreter must be the sealed package-local venv entrypoint")
    return [
        requested_entry, "-P", "-B", "-m", "hfw_dist.daytona_manager",
        "--package-root", str(package_root),
        "--wave-root", str(wave_root),
        "--upload-snapshot-root", os.path.abspath(os.fspath(args.upload_snapshot_root)),
        "--base-image", args.base_image,
        "--api-url", args.api_url,
        "--target", args.target,
        "--worker-timeout-seconds", str(args.worker_timeout_seconds),
        "--campaign-timeout-seconds", str(args.campaign_timeout_seconds),
        "--api-timeout-seconds", str(args.api_timeout_seconds),
        "--poll-seconds", str(args.poll_seconds),
        "--maximum-attempts", str(args.maximum_attempts),
        "--create-concurrency", str(args.create_concurrency),
        "--retry-initial-seconds", str(args.retry_initial_seconds),
        "--retry-maximum-seconds", str(args.retry_maximum_seconds),
        "--vcpu-hour-rate-usd", args.vcpu_hour_rate_usd,
        "--memory-gib-hour-rate-usd", args.memory_gib_hour_rate_usd,
        "--disk-gib-hour-rate-usd", args.disk_gib_hour_rate_usd,
    ]


def _cleanup_passed(wave_root: Path, campaign, campaign_identity) -> bool:
    return all((wave_root / name).is_file() for name in (
        "CLOUD_DOWNLOADS_COMPLETE.json", "LAUNCH_MANIFEST.json",
        "CLOUD_CLEANUP_COMPLETE.json",
    )) and bool(_validate_cloud_handoff(wave_root, campaign, campaign_identity))


def run_pipeline(args) -> Mapping[str, object]:
    package_root = Path(args.package_root).resolve()
    wave_root = Path(args.wave_root).resolve()
    if package_root not in wave_root.parents:
        raise RuntimeError("wave root must be inside package root")
    host_runtime = validate_host_runtime(package_root)
    campaign, campaign_identity = read_json(wave_root / "snapshot/CAMPAIGN.json")
    snapshot_manifest_identity = hash_file(wave_root / "snapshot/SNAPSHOT_MANIFEST.json")
    for name, observed in (
        ("campaign", campaign_identity),
        ("snapshot_manifest", snapshot_manifest_identity),
    ):
        sealed = host_runtime[name]
        if (
            observed["sha256"] != sealed["sha256"]
            or int(observed["size_bytes"]) != int(sealed["size_bytes"])
        ):
            raise RuntimeError(f"requested wave {name} differs from sealed host runtime")
    wave_index = int(campaign["wave_index"])
    if wave_index != int(host_runtime["wave_index"]):
        raise RuntimeError("requested wave index differs from sealed host runtime")
    expected_wave_root = package_root / "runtime/waves" / f"wave-{wave_index:08d}"
    if wave_root != expected_wave_root:
        raise RuntimeError("requested wave path differs from sealed host runtime allocation")
    recovery = validate_recovered_task0_provenance(
        package_root, wave_index=wave_index,
    )
    recovery_identity = None if recovery is None else recovery["identity"]
    if recovery_identity != host_runtime.get("recovered_task0_provenance"):
        raise RuntimeError("recovered task0 provenance differs from sealed host runtime")
    upload_snapshot_root = os.path.abspath(os.fspath(args.upload_snapshot_root))
    sealed_upload = host_runtime["upload_snapshot_staging"]
    if upload_snapshot_root != sealed_upload["upload_snapshot_root"]:
        raise RuntimeError("requested upload snapshot root differs from sealed host runtime")
    manager_source = package_root / "runtime/hfw_dist/daytona_manager.py"
    collector_source = package_root / "runtime/hfw_dist/collect.py"
    promoter_source = package_root / "runtime/hfw_dist/promote_mps.py"
    command = _manager_command(args, package_root, wave_root)
    manager_entrypoint = Path(command[0])
    entrypoint_lstat = manager_entrypoint.lstat()
    runtime_inventory = host_runtime["inventory"]
    secret = os.environ.get("DAYTONA_API_KEY")
    command_text = "\0".join(command)
    if secret and secret in command_text:
        raise RuntimeError("refusing manager command containing Daytona credential")
    intent = {
        "schema": "hfw_w560_real_wave_pipeline_intent_v2",
        "status": "SEALED_BEFORE_MANAGER_SUBPROCESS",
        "campaign_sha256": campaign_identity["sha256"],
        "wave_index": int(campaign["wave_index"]),
        "manager_source": hash_file(manager_source),
        "collector_source": hash_file(collector_source),
        "promoter_source": hash_file(promoter_source),
        "manager_entrypoint": {
            "lexical_path": str(manager_entrypoint),
            "package_local": True,
            "is_symlink": manager_entrypoint.is_symlink(),
            "symlink_target": (
                os.readlink(manager_entrypoint) if manager_entrypoint.is_symlink() else None
            ),
            "lstat_mode": entrypoint_lstat.st_mode,
            "lstat_size_bytes": entrypoint_lstat.st_size,
        },
        "manager_requirements_lock": runtime_inventory[
            "runtime/daytona-manager-requirements.lock"
        ],
        "manager_venv_audit": runtime_inventory[
            "runtime/config/DAYTONA_MANAGER_VENV_AUDIT.json"
        ],
        "upload_snapshot_staging": sealed_upload,
        "recovered_task0_provenance": recovery_identity,
        "recovered_task0_execution_origin": (
            None if recovery is None else recovery["status"]
        ),
        "recovery_failure_chain": (
            [] if recovery is None else [
                {
                    "role": "PREDECESSOR_LABEL_DRIFT_FAILURE",
                    "package_root": recovery[
                        "predecessor_recovery_failure"
                    ]["package_root"],
                    "receipt_chain_length": recovery[
                        "predecessor_recovery_failure"
                    ]["receipt_chain_length"],
                    "receipt_chain_head": recovery[
                        "predecessor_recovery_failure"
                    ]["receipt_chain_head"],
                    "failure_receipt": recovery[
                        "predecessor_recovery_failure"
                    ]["failure_receipt"],
                    "cleanup_terminal_receipt": recovery[
                        "predecessor_recovery_failure"
                    ]["cleanup_terminal_receipt"],
                },
                {
                    "role": "IMMEDIATE_PREDECESSOR_QUOTA_API_FAILURE",
                    "package_root": recovery[
                        "immediate_predecessor_recovery_failure"
                    ]["package_root"],
                    "receipt_chain_length": recovery[
                        "immediate_predecessor_recovery_failure"
                    ]["receipt_chain_length"],
                    "receipt_chain_head": recovery[
                        "immediate_predecessor_recovery_failure"
                    ]["receipt_chain_head"],
                    "failure_receipt": recovery[
                        "immediate_predecessor_recovery_failure"
                    ]["failure_receipt"],
                    "cleanup_terminal_receipt": recovery[
                        "immediate_predecessor_recovery_failure"
                    ]["cleanup_terminal_receipt"],
                },
            ]
        ),
        "recovery_required_exact_id_reconciliation": (
            [] if recovery is None else [
                {
                    "role": row["role"],
                    "sandbox_id": row["sandbox"]["id"],
                }
                for row in recovery["required_exact_id_reconciliation"]
            ]
        ),
        "recovery_required_exact_sandbox_ids": (
            [] if recovery is None else [
                row["sandbox"]["id"]
                for row in recovery["required_exact_id_reconciliation"]
            ]
        ),
        "recovery_all_required_exact_ids_gate_required_before_cloud_create": (
            recovery is not None
        ),
        "task0_model_command_permitted": recovery is None,
        "external_recovery_fork_dependency_revalidated_prelaunch": (
            recovery is not None
        ),
        "manager_argv_without_credential": command,
        "credential_source": "inherited_process_environment_only",
        "credential_in_argv_or_receipt": False,
        "cloud_calls_after_manager_subprocess": False,
        "host_runtime_manifest_sha256": host_runtime["identity"]["sha256"],
    }
    intent_path = wave_root / "REAL_PIPELINE_INTENT.json"
    if intent_path.exists():
        existing, _ = read_json(intent_path)
        if existing != intent:
            raise RuntimeError("real pipeline launch contract changed across restart")
    else:
        write_json_exclusive(intent_path, intent)

    deterministic_terminal = _validated_deterministic_failure_terminal(
        wave_root, campaign, campaign_identity,
    )
    manager_required = deterministic_terminal is None and not all((wave_root / name).is_file() for name in (
        "CLOUD_DOWNLOADS_COMPLETE.json", "CLOUD_CLEANUP_COMPLETE.json",
    ))
    manager_error = None
    try:
        if manager_required:
            if not secret:
                raise RuntimeError("DAYTONA_API_KEY is required only for unfinished manager stage")
            child_environment = dict(os.environ)
            child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
            child_environment["PYTHONNOUSERSITE"] = "1"
            child_environment["PYTHONPATH"] = str(package_root / "runtime")
            try:
                subprocess.run(command, env=child_environment, check=True)
            except BaseException as error:
                manager_error = error
            finally:
                child_environment.pop("DAYTONA_API_KEY", None)
                child_environment.clear()
    finally:
        os.environ.pop("DAYTONA_API_KEY", None)
        secret = None
    if "DAYTONA_API_KEY" in os.environ:
        raise RuntimeError("Daytona credential remained in pipeline environment")
    deterministic_terminal = _validated_deterministic_failure_terminal(
        wave_root, campaign, campaign_identity,
    )
    if deterministic_terminal is not None:
        return _seal_pipeline_deterministic_terminal(
            wave_root, campaign, campaign_identity, deterministic_terminal,
        )
    if manager_error is not None:
        if (wave_root / "CLOUD_CLEANUP_COMPLETE.json").is_file():
            cleanup, _ = read_json(wave_root / "CLOUD_CLEANUP_COMPLETE.json")
            if cleanup.get("status") != "PASS_ZERO_CAMPAIGN_SANDBOXES_AND_SNAPSHOT_ABSENT":
                raise RuntimeError("manager failed and cleanup terminal did not pass") from manager_error
        raise RuntimeError("Daytona manager stage failed after cleanup attempt") from manager_error

    _validate_cloud_handoff(wave_root, campaign, campaign_identity)
    collection = collect_wave(package_root, wave_root)
    promotion = promote_wave(package_root, wave_root)
    terminal = {
        "schema": "hfw_w560_real_checkpoint_writing_pipeline_terminal_v1",
        "status": "PASS_MANAGER_CLEANUP_COLLECTION_AND_MPS_DECISION_COMPLETE",
        "campaign_sha256": campaign_identity["sha256"],
        "wave_index": int(campaign["wave_index"]),
        "cloud_calls_after_manager_subprocess": False,
        "daytona_credential_present_during_collection_or_promotion": False,
        "collection_terminal_sha256": collection["terminal_identity"]["sha256"],
        "promotion_terminal_sha256": promotion["terminal_identity"]["sha256"],
        "promotion_status": promotion["status"],
        "checkpoint_promoted": bool(promotion["accepted"]),
        "current_head_record_sha256": promotion["current_head_record_sha256"],
    }
    terminal_path = wave_root / "REAL_PIPELINE_COMPLETE.json"
    if terminal_path.exists():
        existing, identity = read_json(terminal_path)
        if existing != terminal:
            raise RuntimeError("real pipeline terminal changed across restart")
    else:
        identity = write_json_exclusive(terminal_path, terminal)
    return {**terminal, "terminal_identity": identity}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--upload-snapshot-root", type=Path, required=True)
    parser.add_argument("--manager-python")
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--api-url", default="https://app.daytona.io/api")
    parser.add_argument("--target", default="us")
    parser.add_argument("--worker-timeout-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--campaign-timeout-seconds", type=int, default=8 * 60 * 60)
    parser.add_argument("--api-timeout-seconds", type=int, default=15 * 60)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--create-concurrency", type=int, default=16)
    parser.add_argument("--retry-initial-seconds", type=float, default=2.0)
    parser.add_argument("--retry-maximum-seconds", type=float, default=60.0)
    parser.add_argument("--vcpu-hour-rate-usd", default="0.0504")
    parser.add_argument("--memory-gib-hour-rate-usd", default="0.0162")
    parser.add_argument("--disk-gib-hour-rate-usd", default="0.000108")
    return parser


def main() -> None:
    result = run_pipeline(_build_parser().parse_args())
    print(json.dumps({
        "status": result["status"],
        "checkpoint_promoted": result["checkpoint_promoted"],
        "promotion_status": result["promotion_status"],
        "current_head_record_sha256": result["current_head_record_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
