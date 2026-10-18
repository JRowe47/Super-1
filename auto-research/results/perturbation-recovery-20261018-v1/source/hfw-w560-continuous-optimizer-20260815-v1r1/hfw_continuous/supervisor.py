from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Mapping

from .common import read_json_same_fd, write_content_addressed_json, write_exclusive_json
from .committer import commit_proposal
from .disk_safety import validated_disk_reservations
from .ledger import Ledger
from .relocation import load_config


def recover_pending_proposals(
    config_path: Path, token_manifest_path: Path, ledger_root: Path,
) -> list[Mapping[str, object]]:
    ledger = Ledger(ledger_root)
    ledger.validate_runtime_inputs(config_path, token_manifest_path)
    heads = ledger.heads()
    referenced_decisions = {head["decision_sha256"] for head in heads if head["decision_sha256"]}
    decisions = []
    for path in sorted((Path(ledger_root) / "decisions").glob("*.decision.json")):
        value, identity = read_json_same_fd(path)
        decisions.append((value, identity))
    results = []
    for proposal_path in sorted((Path(ledger_root) / "proposals").glob("*.proposal.json")):
        _, proposal_identity = read_json_same_fd(proposal_path)
        rows = [
            (value, identity) for value, identity in decisions
            if value.get("proposal_sha256") == proposal_identity["sha256"]
        ]
        terminal = any(
            (not value.get("accepted", False)) or identity["sha256"] in referenced_decisions
            for value, identity in rows
        )
        if not terminal:
            results.append(commit_proposal(
                config_path, token_manifest_path, ledger_root, proposal_path
            ))
    return results


def _worker_command(
    config_path: Path, token_manifest_path: Path, ledger_root: Path,
    job_path: Path, output_root: Path,
) -> list[str]:
    return [
        sys.executable, "-m", "hfw_continuous.cli", "worker",
        "--config", str(config_path),
        "--token-manifest", str(token_manifest_path),
        "--ledger", str(ledger_root),
        "--job", str(job_path),
        "--output", str(output_root),
    ]


def run_continuously(
    config_path: Path, token_manifest_path: Path, ledger_root: Path,
    *, max_waves_override: int | None = None,
) -> Mapping[str, object]:
    ledger = Ledger(ledger_root)
    with ledger.supervisor_lock():
        initial_runtime_inputs = ledger.validate_runtime_inputs(config_path, token_manifest_path)
        config = initial_runtime_inputs["config_value"]
        groups = tuple(config["controller"]["groups"])
        workers = int(config["controller"]["parallel_workers"])
        if workers < 1:
            raise RuntimeError("parallel_workers must be positive")
        initial_disk = validated_disk_reservations(config)
        initial_required = (
            int(initial_disk["minimum_unreserved_free_bytes"])
            + workers * int(initial_disk["reserved_bytes_per_worker"])
            + int(initial_disk["reserved_bytes_for_possible_promotion"])
        )
        initial_free = shutil.disk_usage(Path(ledger_root)).free
        if initial_free < initial_required:
            failure = {
                "schema": "hfw_w560_continuous_disk_gate_failure_v1",
                "status": "BLOCK_INSUFFICIENT_DISK_BEFORE_RECOVERY_OR_WAVE",
                "free_bytes": initial_free,
                "required_free_bytes": initial_required,
                "no_recovery_or_jobs_started": True,
            }
            artifact = write_content_addressed_json(
                Path(ledger_root) / "failure_receipts", ".failure.json", failure
            )
            return {**failure, "failure_artifact": artifact, "current_head": ledger.current_head()}
        try:
            recovery = recover_pending_proposals(config_path, token_manifest_path, ledger_root)
        except Exception as error:
            failure = {
                "schema": "hfw_w560_continuous_recovery_exception_v1",
                "status": "BLOCK_PENDING_PROPOSAL_RECOVERY_EXCEPTION_SUPERVISOR_STOPPED",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "artifacts_preserved": True,
                "no_jobs_created": True,
                "no_new_wave_will_start": True,
                "in_process_retry_performed": False,
                "head_after": ledger.current_head(),
            }
            artifact = write_content_addressed_json(
                Path(ledger_root) / "failure_receipts", ".failure.json", failure
            )
            return {**failure, "failure_artifact": artifact}
        recovery_blocks = [
            row for row in recovery if str(row.get("status", "")).startswith("BLOCK_")
        ]
        if recovery_blocks:
            return {
                "status": "BLOCK_PENDING_PROPOSAL_RECOVERY_FAILED_CLOSED",
                "recovered_pending_proposals": recovery,
                "current_head": ledger.current_head(),
            }
        configured_limit = config["controller"].get("max_waves")
        limit = max_waves_override if max_waves_override is not None else configured_limit
        wave = 0
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root) + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        while limit is None or wave < int(limit):
            runtime_inputs = ledger.validate_runtime_inputs(config_path, token_manifest_path)
            config = runtime_inputs["config_value"]
            disk = validated_disk_reservations(config)
            disk_usage = shutil.disk_usage(Path(ledger_root))
            reserved = (
                int(disk["minimum_unreserved_free_bytes"])
                + workers * int(disk["reserved_bytes_per_worker"])
                + int(disk["reserved_bytes_for_possible_promotion"])
            )
            if disk_usage.free < reserved:
                failure = {
                    "schema": "hfw_w560_continuous_disk_gate_failure_v1",
                    "status": "BLOCK_INSUFFICIENT_DISK_BEFORE_WAVE",
                    "wave": wave,
                    "free_bytes": disk_usage.free,
                    "required_free_bytes": reserved,
                    "disk_safety_config": dict(disk),
                    "no_jobs_created": True,
                }
                artifact = write_content_addressed_json(
                    Path(ledger_root) / "failure_receipts", ".failure.json", failure
                )
                return {
                    **failure, "failure_artifact": artifact,
                    "recovered_pending_proposals": recovery,
                    "current_head": ledger.current_head(),
                }
            launched = []
            for _ in range(workers):
                job_index = len(list((Path(ledger_root) / "jobs").glob("*.job.json")))
                group = groups[job_index % len(groups)]
                job = ledger.create_job(
                    group=group, root_seed=int(config["controller"]["root_seed"]),
                    allowed_groups=groups,
                )
                output = Path(ledger_root) / "worker_attempts" / (
                    f"job-{job['job_index']:012d}-{job['job_record_sha256'][:16]}-attempt-0001"
                )
                command = _worker_command(
                    config_path, token_manifest_path, ledger_root,
                    Path(job["job_artifact"]["path"]), output,
                )
                process = subprocess.Popen(
                    command, cwd=package_root, env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                launched.append((job, output, command, process))
            completed = []
            failures = []
            for job, output, command, process in launched:
                stdout, stderr = process.communicate()
                row = (job, output, command, process, stdout, stderr)
                completed.append(row)
                if process.returncode:
                    failure = {
                        "schema": "hfw_w560_continuous_worker_failure_v1",
                        "status": "BLOCK_WORKER_FAILED_SUPERVISOR_STOPPED",
                        "job_record_sha256": job["job_record_sha256"],
                        "exit_code": process.returncode,
                        "command": command,
                        "stdout": stdout,
                        "stderr": stderr,
                        "artifacts_preserved": True,
                        "job_will_not_be_reused": True,
                        "no_new_wave_will_start": True,
                    }
                    artifact = write_content_addressed_json(
                        Path(ledger_root) / "failure_receipts", ".failure.json", failure
                    )
                    failures.append({**failure, "failure_artifact": artifact})
            if failures:
                wave_receipt = {
                    "schema": "hfw_w560_continuous_wave_receipt_v1",
                    "status": "BLOCK_WORKER_FAILURE_NO_SUCCESSFUL_PROPOSALS_COMMITTED_THIS_WAVE",
                    "wave": wave,
                    "jobs": [job["job_record_sha256"] for job, _, _, _ in launched],
                    "failures": failures,
                    "successful_worker_proposals_left_pending_for_explicit_recovery": True,
                    "head_after": ledger.current_head(),
                }
                receipt = write_content_addressed_json(
                    Path(ledger_root) / "worker_attempts", ".wave.json", wave_receipt
                )
                return {
                    **wave_receipt, "wave_receipt_artifact": receipt,
                    "recovered_pending_proposals": recovery,
                }
            outcomes = []
            for job, output, command, process, stdout, stderr in completed:
                terminal, _ = read_json_same_fd(output / "PROPOSAL_COMPLETE.json")
                proposal_path = Path(terminal["proposal_artifact"]["path"])
                head_before = ledger.current_head()
                try:
                    outcome = commit_proposal(
                        config_path, token_manifest_path, ledger_root, proposal_path
                    )
                except Exception as error:
                    failure = {
                        "schema": "hfw_w560_continuous_committer_exception_v1",
                        "status": "BLOCK_COMMITTER_EXCEPTION_SUPERVISOR_STOPPED",
                        "wave": wave,
                        "job_record_sha256": job["job_record_sha256"],
                        "proposal_sha256": terminal["proposal_artifact"]["sha256"],
                        "proposal_path": str(proposal_path),
                        "head_before": head_before,
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                        "traceback": traceback.format_exc(),
                        "artifacts_preserved": True,
                        "proposal_remains_uncommitted": True,
                        "no_new_wave_will_start": True,
                        "in_process_retry_performed": False,
                    }
                    failure_artifact = write_content_addressed_json(
                        Path(ledger_root) / "failure_receipts", ".failure.json", failure
                    )
                    wave_receipt = {
                        "schema": "hfw_w560_continuous_wave_receipt_v1",
                        "status": "BLOCK_COMMITTER_EXCEPTION_NO_FURTHER_PROPOSALS_COMMITTED",
                        "wave": wave,
                        "jobs": [row[0]["job_record_sha256"] for row in launched],
                        "outcomes": outcomes,
                        "committer_failure": {**failure, "failure_artifact": failure_artifact},
                        "remaining_successful_proposals_left_pending": True,
                        "head_after": ledger.current_head(),
                    }
                    receipt = write_content_addressed_json(
                        Path(ledger_root) / "worker_attempts", ".wave.json", wave_receipt
                    )
                    return {
                        **wave_receipt, "wave_receipt_artifact": receipt,
                        "recovered_pending_proposals": recovery,
                    }
                outcomes.append(outcome)
                if str(outcome.get("status", "")).startswith("BLOCK_"):
                    wave_receipt = {
                        "schema": "hfw_w560_continuous_wave_receipt_v1",
                        "status": "BLOCK_COMMITTER_FAILED_CLOSED_NO_FURTHER_PROPOSALS_COMMITTED",
                        "wave": wave,
                        "jobs": [row[0]["job_record_sha256"] for row in launched],
                        "outcomes": outcomes,
                        "remaining_successful_proposals_left_pending": True,
                        "head_after": ledger.current_head(),
                    }
                    receipt = write_content_addressed_json(
                        Path(ledger_root) / "worker_attempts", ".wave.json", wave_receipt
                    )
                    return {
                        **wave_receipt, "wave_receipt_artifact": receipt,
                        "recovered_pending_proposals": recovery,
                    }
            wave_receipt = {
                "schema": "hfw_w560_continuous_wave_receipt_v1",
                "status": "PASS_WAVE_COMPLETE",
                "wave": wave,
                "jobs": [job["job_record_sha256"] for job, _, _, _ in launched],
                "outcomes": outcomes,
                "disk_gate": {
                    "free_bytes": disk_usage.free,
                    "required_free_bytes": reserved,
                    "passed": True,
                },
                "runtime_input_contract": runtime_inputs,
                "head_after": ledger.current_head(),
            }
            write_content_addressed_json(
                Path(ledger_root) / "worker_attempts", ".wave.json", wave_receipt
            )
            wave += 1
        return {
            "status": "STOPPED_AT_EXPLICIT_WAVE_LIMIT",
            "waves_completed_this_process": wave,
            "recovered_pending_proposals": recovery,
            "current_head": ledger.current_head(),
        }
