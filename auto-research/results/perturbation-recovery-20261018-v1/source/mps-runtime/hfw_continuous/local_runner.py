from __future__ import annotations

import fcntl
from pathlib import Path
import time
import traceback
from typing import Mapping

from .common import read_json_same_fd, write_exclusive_bytes, write_exclusive_json
from .local_config import load_config, runtime_root
from .local_ledger import LocalLedger
from .local_promoter import evaluate_and_promote
from .local_solver_worker import run_local_solver_job


def initialize_continuous_run(
    config_path: Path, ledger_root: Path, run_root: Path,
) -> Mapping[str, object]:
    config, config_identity = load_config(config_path)
    ledger = LocalLedger(ledger_root, config_path)
    current = ledger.current_head()
    incomplete = ledger.incomplete_job_states()
    if len(incomplete) > 1:
        raise RuntimeError("cumulative handoff has ambiguous multiple incomplete jobs")
    if incomplete and incomplete[0]["stage"] == "ACCEPTED_DECISION_UNHEADED_BLOCKED":
        raise RuntimeError("cumulative handoff is blocked on an accepted decision without an anchored head")
    root = runtime_root(config_path)
    run_root = Path(run_root).resolve()
    try:
        run_root.relative_to(root)
    except ValueError as error:
        raise RuntimeError("cumulative run is outside runtime") from error
    if run_root.exists():
        raise FileExistsError("cumulative run root must be fresh")
    run_root.mkdir(parents=True)
    (run_root / "attempts").mkdir()
    write_exclusive_bytes(run_root / "runner.lock", b"", mode=0o600)
    contract = {
        "schema": "hfw_w560_mps_cumulative_local_run_v1",
        "status": "INITIALIZED_NOT_LAUNCHED",
        "config_sha256": config_identity["sha256"],
        "ledger_contract_sha256": read_json_same_fd(ledger.root / "contract.json")[1]["sha256"],
        "initial_head_record_sha256": current["record_sha256"],
        "initial_checkpoint_sha256": current["checkpoint"]["sha256"],
        "local_verify_role": "LOCAL_VERIFY",
        "local_verify_disclosure": "PERMANENT_REUSED_OPTIMIZATION_DATA_NOT_HELD_OUT",
        "single_worker": True,
        "next_job_always_allocated_from_current_append_only_head": True,
        "incomplete_job_at_initialization": None if not incomplete else incomplete[0]["job_record_sha256"],
        "incomplete_stage_at_initialization": None if not incomplete else incomplete[0]["stage"],
        "cloud_access": False,
        "training_or_backprop": False,
        "adam_or_any_optimizer": False,
    }
    artifact = write_exclusive_json(run_root / "contract.json", contract)
    return {**contract, "contract_artifact": artifact}


def run_continuously(
    config_path: Path, ledger_root: Path, run_root: Path,
    *, maximum_cycles: int | None = None,
) -> Mapping[str, object]:
    config, config_identity = load_config(config_path)
    if config["authority"]["local_solver_launch_authorized"] is not True:
        raise RuntimeError("long cumulative local solver launch is not authorized in this frozen build")
    if maximum_cycles is not None and int(maximum_cycles) <= 0:
        raise RuntimeError("maximum_cycles must be positive")
    ledger = LocalLedger(ledger_root, config_path)
    run_root = Path(run_root).resolve()
    contract, _ = read_json_same_fd(run_root / "contract.json")
    if contract.get("config_sha256") != config_identity["sha256"]:
        raise RuntimeError("cumulative run/config binding drift")
    lock_path = run_root / "runner.lock"
    completed = 0
    started = time.monotonic()
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("cumulative local runner is already active") from error
        incomplete = ledger.incomplete_job_states()
        if len(incomplete) > 1:
            raise RuntimeError("cumulative run refuses ambiguous multiple incomplete jobs")
        if incomplete:
            job_state = incomplete[0]
        else:
            allocated = ledger.allocate_next_job()
            job_state = ledger.job_states()[-1]
            if job_state["job_record_sha256"] != allocated["job_record_sha256"]:
                raise RuntimeError("newly allocated job state binding drift")
        while maximum_cycles is None or completed < int(maximum_cycles):
            if job_state["stage"] == "ACCEPTED_DECISION_UNHEADED_BLOCKED":
                raise RuntimeError("accepted LOCAL_VERIFY decision lacks an anchored append-only head")
            if job_state["stage"] not in {"SOLVER_PENDING", "VERIFY_PENDING"}:
                raise RuntimeError(f"unexpected incomplete job stage: {job_state['stage']}")
            job = job_state
            current_before = ledger.current_head()
            if job_state["stage"] == "SOLVER_PENDING":
                if job["base_head_record_sha256"] != current_before["record_sha256"]:
                    raise RuntimeError("pending local solver job is stale against current head")
                if job["base_checkpoint_sha256"] != current_before["checkpoint"]["sha256"]:
                    raise RuntimeError("pending local solver job checkpoint is stale")
            job_index = int(job["job_index"])
            attempt_root = run_root / "attempts" / f"job-{job_index:012d}-attempt-0001"
            try:
                if job_state["stage"] == "SOLVER_PENDING":
                    proposal = run_local_solver_job(
                        config_path, ledger.root, Path(job["job_artifact"]["path"]), attempt_root,
                    )
                    proposal_path = Path(proposal["proposal_artifact"]["path"])
                else:
                    proposal = job_state["proposal"]
                    proposal_path = Path(proposal["artifact"]["path"])
                result = evaluate_and_promote(
                    config_path, ledger.root, proposal_path,
                )
                completed_state = [
                    row for row in ledger.job_states()
                    if row["job_record_sha256"] == job["job_record_sha256"]
                ]
                if len(completed_state) != 1 or completed_state[0]["stage"] != "COMPLETE":
                    raise RuntimeError("job did not reach decision-and-head-validated COMPLETE state")
                next_job = ledger.allocate_next_job()
                current = ledger.current_head()
                if next_job["base_head_record_sha256"] != current["record_sha256"]:
                    raise RuntimeError("post-cycle next job did not read current append-only head")
                if next_job["base_checkpoint_sha256"] != current["checkpoint"]["sha256"]:
                    raise RuntimeError("post-cycle next job did not read current checkpoint")
                incomplete_after = ledger.incomplete_job_states()
                if (
                    len(incomplete_after) != 1
                    or incomplete_after[0]["job_record_sha256"] != next_job["job_record_sha256"]
                    or incomplete_after[0]["stage"] != "SOLVER_PENDING"
                ):
                    raise RuntimeError("post-cycle successor is not the unique solver-pending job")
                cycle = {
                    "schema": "hfw_w560_mps_cumulative_local_cycle_receipt_v1",
                    "status": "PASS_CYCLE_COMPLETE_NEXT_JOB_BOUND_CURRENT_HEAD",
                    "job_record_sha256": job["job_record_sha256"],
                    "proposal_sha256": result["proposal_sha256"],
                    "decision_sha256": result["decision_artifact"]["sha256"],
                    "accepted": result["accepted"],
                    "current_head_record_sha256": current["record_sha256"],
                    "current_checkpoint_sha256": current["checkpoint"]["sha256"],
                    "next_job_record_sha256": next_job["job_record_sha256"],
                    "next_job_base_head_record_sha256": next_job["base_head_record_sha256"],
                    "next_job_base_checkpoint_sha256": next_job["base_checkpoint_sha256"],
                    "next_job_reads_current_append_only_head_after_promotion": True,
                    "local_verify_is_reused_optimization_data_not_held_out": True,
                }
                write_exclusive_json(attempt_root / "CUMULATIVE_CYCLE_COMPLETE.json", cycle)
                job_state = incomplete_after[0]
            except BaseException as error:
                attempt_root.mkdir(parents=True, exist_ok=True)
                write_exclusive_json(attempt_root / "CUMULATIVE_CYCLE_FAILED.json", {
                    "schema": "hfw_w560_mps_cumulative_local_cycle_failure_v1",
                    "status": "BLOCK_PRESERVED_FAILED_CYCLE_RUN_STOPPED",
                    "job_index": job_index,
                    "job_record_sha256": job["job_record_sha256"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                })
                raise
            completed += 1
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    receipt = {
        "schema": "hfw_w560_mps_cumulative_local_call_receipt_v1",
        "status": "PASS_REQUESTED_CUMULATIVE_CYCLES_COMPLETE",
        "completed_cycles": completed,
        "maximum_cycles": maximum_cycles,
        "wall_seconds": time.monotonic() - started,
        "cloud_access": False,
        "training_or_backprop": False,
        "adam_or_any_optimizer": False,
    }
    artifact = write_exclusive_json(run_root / f"call-{time.time_ns()}.receipt.json", receipt)
    return {**receipt, "receipt_artifact": artifact}
