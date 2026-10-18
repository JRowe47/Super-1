from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Mapping

from .assets import validate_assets
from .benchmark import benchmark
from .common import canonical_json_bytes, exclusive_lock, hash_file_same_fd, read_json_same_fd, sha256_bytes, write_exclusive_bytes, write_exclusive_json
from .config import runtime_root, validate_pinned_inputs
from .lineage import active_modules


def initialize(config_path: Path) -> Mapping[str, object]:
    package = validate_pinned_inputs(config_path)
    assets = validate_assets(config_path)
    solver_child_diagnostic = validate_solver_child_environment_diagnostic(config_path, package)
    config = package["config"]
    root = runtime_root(config_path)
    output = root / "supervisor/contract.json"
    if output.exists():
        raise FileExistsError("create-only supervisor contract already exists")
    command = list(config["supervisor"]["solver_command"])
    if command[-2:] != ["--maximum-cycles", "1"]:
        raise RuntimeError("supervisor command is not exactly one cycle")
    if len(command) < 3 or command[1] != "-B":
        raise RuntimeError("supervisor command must disable bytecode with -B")
    baseline_cycles = sorted(_cycles(config))
    baseline_head = _current_head(config)
    contract = {
        "schema": "hfw_w560_promotion_cycle_supervisor_contract_v1",
        "status": "INITIALIZED_NOT_LAUNCHED",
        "config_sha256": package["config_identity"]["sha256"],
        "runtime_manifest_sha256": package["runtime_manifest_identity"]["sha256"],
        "asset_manifest_sha256": assets["manifest_identity"]["sha256"],
        "asset_import_receipt_sha256": assets["import_identity"]["sha256"],
        "solver_child_environment_diagnostic_sha256": solver_child_diagnostic["identity"]["sha256"],
        "solver_command": command,
        "baseline_cycle_receipt_sha256s": baseline_cycles,
        "baseline_current_head_sha256": baseline_head["record_sha256"],
        "baseline_current_checkpoint_sha256": baseline_head["checkpoint"]["sha256"],
        "one_cycle_per_subprocess": True,
        "canary_after_every_accepted_promotion": True,
        "full_after_accepted_promotion_count": config["supervisor"]["full_every_accepted_promotions"],
        "full_after_wall_seconds": config["supervisor"]["full_every_seconds"],
        "initialized_time_ns": time.time_ns(),
        "benchmark_may_gate_or_promote": False,
        "network_access": False,
    }
    identity = write_exclusive_json(output, contract)
    write_exclusive_bytes(root / "supervisor/supervisor.lock", b"", mode=0o600)
    return {**contract, "contract": identity}


def _cycles(config: Mapping[str, object]) -> dict[str, Path]:
    root = Path(str(config["active_controller"]["run_attempts_root"]))
    return {
        str(hash_file_same_fd(path)["sha256"]): path
        for path in root.glob("job-*-attempt-*/CUMULATIVE_CYCLE_COMPLETE.json")
    }


def _current_head(config: Mapping[str, object]) -> Mapping[str, object]:
    active = config["active_controller"]
    with active_modules(config) as (ledger_module, _):
        ledger = ledger_module.LocalLedger(
            Path(str(active["lineage_root"])), Path(str(active["config_path"])),
        )
        return ledger.current_head()


def _solver_environment(config: Mapping[str, object]) -> dict[str, str]:
    environment = {
        "PATH": str(config["supervisor"]["solver_path"]),
        "PYTHONPATH": str(config["supervisor"]["solver_pythonpath"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(config["supervisor"]["solver_pycache_prefix"]),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "DO_NOT_TRACK": "1",
    }
    prohibited = {"PYTHONNOUSERSITE", "PYTHONSAFEPATH"}
    if prohibited & set(environment):
        raise RuntimeError("benchmark interpreter isolation leaked into solver child")
    return environment


def _solver_child_probe_command(config: Mapping[str, object]) -> list[str]:
    code = (
        "import json,os,sys,torch,numpy;"
        "import hfw_continuous.local_cli;"
        "print(json.dumps({"
        "'status':'PASS_SOLVER_CHILD_TORCH_IMPORT_NO_MODEL',"
        "'torch_version':torch.__version__,"
        "'numpy_version':numpy.__version__,"
        "'mps_available':torch.backends.mps.is_available(),"
        "'mps_built':torch.backends.mps.is_built(),"
        "'hfw_continuous_local_cli_imported':True,"
        "'python':sys.executable,"
        "'pythonpath':os.environ.get('PYTHONPATH'),"
        "'python_no_user_site':os.environ.get('PYTHONNOUSERSITE'),"
        "'python_safe_path':os.environ.get('PYTHONSAFEPATH'),"
        "'model_forward_calls':0,'tokenizer_encode_calls':0},sort_keys=True))"
    )
    return [str(config["supervisor"]["solver_command"][0]), "-B", "-c", code]


def diagnose_solver_child_environment(config_path: Path) -> Mapping[str, object]:
    package = validate_pinned_inputs(config_path)
    config = package["config"]
    root = runtime_root(config_path)
    output = root / "validation/SOLVER_CHILD_ENVIRONMENT_DIAGNOSTIC.json"
    if output.exists():
        raise FileExistsError("create-only solver child diagnostic already exists")
    environment = _solver_environment(config)
    command = _solver_child_probe_command(config)
    completed = subprocess.run(
        command, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60,
    )
    lines = [line for line in completed.stdout.decode("utf-8").splitlines() if line.strip()]
    if completed.returncode != 0 or len(lines) != 1:
        raise RuntimeError(
            f"solver child torch import failed: returncode={completed.returncode} "
            f"stderr={completed.stderr.decode('utf-8', errors='replace')}"
        )
    observed = json.loads(lines[0])
    if (
        observed.get("status") != "PASS_SOLVER_CHILD_TORCH_IMPORT_NO_MODEL"
        or observed.get("torch_version") != config["supervisor"]["solver_dependency_versions"]["torch"]
        or observed.get("numpy_version") != config["supervisor"]["solver_dependency_versions"]["numpy"]
        or observed.get("mps_available") is not True
        or observed.get("mps_built") is not True
        or observed.get("hfw_continuous_local_cli_imported") is not True
        or observed.get("pythonpath") != config["supervisor"]["solver_pythonpath"]
        or observed.get("python_no_user_site") is not None
        or observed.get("python_safe_path") is not None
        or int(observed.get("model_forward_calls", -1)) != 0
        or int(observed.get("tokenizer_encode_calls", -1)) != 0
    ):
        raise RuntimeError("solver child diagnostic output drift")
    body = {
        "schema": "hfw_w560_solver_child_environment_diagnostic_v1",
        "status": "PASS_EXACT_SOLVER_CHILD_TORCH_IMPORT_NO_MODEL",
        "config_sha256": package["config_identity"]["sha256"],
        "runtime_manifest_sha256": package["runtime_manifest_identity"]["sha256"],
        "command": command,
        "environment": environment,
        "environment_identity_sha256": sha256_bytes(canonical_json_bytes(environment)),
        "observed": observed,
        "returncode": completed.returncode,
        "stderr": completed.stderr.decode("utf-8", errors="replace"),
        "model_forward_calls": 0,
        "tokenizer_encode_calls": 0,
    }
    identity = write_exclusive_json(output, body)
    return {**body, "identity": identity}


def validate_solver_child_environment_diagnostic(
    config_path: Path, package: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    package = package or validate_pinned_inputs(config_path)
    config = package["config"]
    root = runtime_root(config_path)
    body, identity = read_json_same_fd(root / "validation/SOLVER_CHILD_ENVIRONMENT_DIAGNOSTIC.json")
    expected_environment = _solver_environment(config)
    observed = body.get("observed", {})
    if (
        body.get("schema") != "hfw_w560_solver_child_environment_diagnostic_v1"
        or body.get("status") != "PASS_EXACT_SOLVER_CHILD_TORCH_IMPORT_NO_MODEL"
        or body.get("config_sha256") != package["config_identity"]["sha256"]
        or body.get("runtime_manifest_sha256") != package["runtime_manifest_identity"]["sha256"]
        or body.get("command") != _solver_child_probe_command(config)
        or body.get("environment") != expected_environment
        or body.get("environment_identity_sha256") != sha256_bytes(canonical_json_bytes(expected_environment))
        or observed.get("torch_version") != config["supervisor"]["solver_dependency_versions"]["torch"]
        or observed.get("numpy_version") != config["supervisor"]["solver_dependency_versions"]["numpy"]
        or observed.get("mps_available") is not True
        or observed.get("mps_built") is not True
        or observed.get("hfw_continuous_local_cli_imported") is not True
        or observed.get("pythonpath") != config["supervisor"]["solver_pythonpath"]
        or observed.get("python_no_user_site") is not None
        or observed.get("python_safe_path") is not None
        or int(body.get("model_forward_calls", -1)) != 0
        or int(body.get("tokenizer_encode_calls", -1)) != 0
    ):
        raise RuntimeError("solver child environment diagnostic drift")
    return {**body, "identity": identity}


def _run_one_solver_cycle(
    config_path: Path, config: Mapping[str, object], cycle_index: int,
) -> Mapping[str, object]:
    root = runtime_root(config_path)
    before_cycles = _cycles(config)
    before_head = _current_head(config)
    command = [str(value) for value in config["supervisor"]["solver_command"]]
    started_ns = time.time_ns()
    with exclusive_lock(Path(str(config["locks"]["host_mps_lane"])), nonblocking=False):
        completed = subprocess.run(
            command, env=_solver_environment(config), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    log_root = root / "supervisor/logs"
    stdout_identity = write_exclusive_bytes(
        log_root / f"{cycle_index:012d}-{started_ns}.solver.stdout", completed.stdout,
    )
    stderr_identity = write_exclusive_bytes(
        log_root / f"{cycle_index:012d}-{started_ns}.solver.stderr", completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"one-cycle solver exited {completed.returncode}; "
            f"stdout={stdout_identity['sha256']} stderr={stderr_identity['sha256']}"
        )
    lines = [line for line in completed.stdout.decode("utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("one-cycle solver produced no call receipt")
    call = json.loads(lines[-1])
    if (
        call.get("schema") != "hfw_w560_mps_cumulative_local_call_receipt_v1"
        or call.get("status") != "PASS_REQUESTED_CUMULATIVE_CYCLES_COMPLETE"
        or int(call.get("completed_cycles", -1)) != 1
        or int(call.get("maximum_cycles", -1)) != 1
    ):
        raise RuntimeError("one-cycle solver call receipt drift")
    after_cycles = _cycles(config)
    new = sorted(set(after_cycles) - set(before_cycles))
    if len(new) != 1:
        raise RuntimeError(f"one-cycle solver emitted {len(new)} new cycle receipts")
    cycle, cycle_identity = read_json_same_fd(after_cycles[new[0]], new[0])
    after_head = _current_head(config)
    accepted = cycle.get("accepted") is True
    if accepted != (after_head["record_sha256"] != before_head["record_sha256"]):
        raise RuntimeError("accepted flag/head-change mismatch")
    if accepted and after_head.get("parent_head_record_sha256") != before_head["record_sha256"]:
        raise RuntimeError("accepted cycle did not create exactly one child head")
    if not accepted and after_head["record_sha256"] != before_head["record_sha256"]:
        raise RuntimeError("rejected cycle changed head")
    if (
        cycle.get("current_head_record_sha256") != after_head["record_sha256"]
        or cycle.get("current_checkpoint_sha256") != after_head["checkpoint"]["sha256"]
    ):
        raise RuntimeError("cycle receipt/post-cycle head mismatch")
    return {
        "call": call, "cycle": cycle, "cycle_identity": cycle_identity,
        "before_head": before_head, "after_head": after_head,
        "solver_stdout": stdout_identity, "solver_stderr": stderr_identity,
    }


def _completed_events(root: Path) -> list[Mapping[str, object]]:
    events = []
    for path in sorted((root / "supervisor/events").glob("*.cycle.json")):
        value, identity = read_json_same_fd(path)
        events.append({**value, "record_sha256": identity["sha256"]})
    return events


def _validate_clean_restart_boundary(
    config: Mapping[str, object], contract: Mapping[str, object],
    contract_sha256: str, events: list[Mapping[str, object]],
) -> None:
    if [int(event.get("cycle_index", -1)) for event in events] != list(range(len(events))):
        raise RuntimeError("supervisor event sequence is not contiguous")
    if any(event.get("contract_sha256") != contract_sha256 for event in events):
        raise RuntimeError("supervisor event/contract binding drift")
    event_cycles = [str(event.get("cycle_receipt_sha256")) for event in events]
    if len(event_cycles) != len(set(event_cycles)):
        raise RuntimeError("duplicate supervisor cycle receipt")
    baseline_cycles = [str(value) for value in contract.get("baseline_cycle_receipt_sha256s", [])]
    if len(baseline_cycles) != len(set(baseline_cycles)):
        raise RuntimeError("duplicate baseline cycle receipt")
    observed_cycles = set(_cycles(config))
    expected_cycles = set(baseline_cycles) | set(event_cycles)
    if observed_cycles != expected_cycles:
        missing = sorted(expected_cycles - observed_cycles)[:3]
        unrecorded = sorted(observed_cycles - expected_cycles)[:3]
        raise RuntimeError(
            f"unclean restart boundary; refusing to skip or replay cycle: "
            f"missing={missing} unrecorded={unrecorded}"
        )
    current = _current_head(config)
    expected_head = (
        str(events[-1]["after_head_sha256"])
        if events else str(contract["baseline_current_head_sha256"])
    )
    expected_checkpoint = (
        str(events[-1]["after_checkpoint_sha256"])
        if events else str(contract["baseline_current_checkpoint_sha256"])
    )
    if (
        current["record_sha256"] != expected_head
        or current["checkpoint"]["sha256"] != expected_checkpoint
    ):
        raise RuntimeError("unclean restart boundary; current lineage differs from recorded supervisor state")


def supervise(
    config_path: Path, maximum_supervised_cycles: int | None,
) -> Mapping[str, object]:
    package = validate_pinned_inputs(config_path)
    assets = validate_assets(config_path)
    solver_child_diagnostic = validate_solver_child_environment_diagnostic(config_path, package)
    config = package["config"]
    root = runtime_root(config_path)
    contract, contract_identity = read_json_same_fd(root / "supervisor/contract.json")
    if (
        contract.get("schema") != "hfw_w560_promotion_cycle_supervisor_contract_v1"
        or contract.get("status") != "INITIALIZED_NOT_LAUNCHED"
        or contract.get("one_cycle_per_subprocess") is not True
        or contract.get("benchmark_may_gate_or_promote") is not False
        or contract.get("network_access") is not False
    ):
        raise RuntimeError("supervisor contract policy drift")
    if contract.get("config_sha256") != package["config_identity"]["sha256"]:
        raise RuntimeError("supervisor contract/config drift")
    expected_contract_bindings = {
        "runtime_manifest_sha256": package["runtime_manifest_identity"]["sha256"],
        "asset_manifest_sha256": assets["manifest_identity"]["sha256"],
        "asset_import_receipt_sha256": assets["import_identity"]["sha256"],
        "solver_child_environment_diagnostic_sha256": solver_child_diagnostic["identity"]["sha256"],
        "solver_command": list(config["supervisor"]["solver_command"]),
    }
    for key, expected in expected_contract_bindings.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"supervisor immutable contract drift: {key}")
    if maximum_supervised_cycles is not None and maximum_supervised_cycles <= 0:
        raise RuntimeError("maximum supervised cycles must be positive")
    completed_this_call = 0
    with exclusive_lock(root / "supervisor/supervisor.lock", nonblocking=True):
        while maximum_supervised_cycles is None or completed_this_call < maximum_supervised_cycles:
            existing = _completed_events(root)
            _validate_clean_restart_boundary(
                config, contract, contract_identity["sha256"], existing,
            )
            cycle_index = len(existing)
            started_ns = time.time_ns()
            try:
                run = _run_one_solver_cycle(config_path, config, cycle_index)
                benchmark_records = []
                if run["cycle"]["accepted"] is True:
                    accepted_count = sum(int(event.get("accepted") is True) for event in existing) + 1
                    modes = ["canary"]
                    last_full_ns = max(
                        (
                            int(event["completed_time_ns"])
                            for event in existing
                            if any(
                                record.get("mode") == "full" and record.get("status") == "PASS"
                                for record in event.get("benchmark_records", [])
                            )
                        ),
                        default=int(contract["initialized_time_ns"]),
                    )
                    if (
                        accepted_count % int(config["supervisor"]["full_every_accepted_promotions"]) == 0
                        or started_ns - last_full_ns >= int(config["supervisor"]["full_every_seconds"]) * 1_000_000_000
                    ):
                        modes.append("full")
                    for mode in modes:
                        try:
                            benchmark_records.append({
                                "mode": mode, "status": "PASS",
                                "receipt": benchmark(
                                    config_path, Path(str(run["cycle_identity"]["path"])), mode,
                                ),
                            })
                        except BaseException as error:
                            benchmark_records.append({
                                "mode": mode,
                                "status": "REPORTING_FAILURE_DID_NOT_GATE_HEAD",
                                "error_type": type(error).__name__, "error": str(error),
                                "traceback": traceback.format_exc(),
                            })
                event = {
                    "schema": "hfw_w560_promotion_cycle_supervisor_event_v1",
                    "status": "PASS_ONE_SOLVER_CYCLE_AND_REPORTING_BOUNDARY_COMPLETE",
                    "contract_sha256": contract_identity["sha256"],
                    "cycle_index": cycle_index, "completed_time_ns": time.time_ns(),
                    "cycle_receipt_sha256": run["cycle_identity"]["sha256"],
                    "accepted": run["cycle"]["accepted"],
                    "before_head_sha256": run["before_head"]["record_sha256"],
                    "after_head_sha256": run["after_head"]["record_sha256"],
                    "after_checkpoint_sha256": run["after_head"]["checkpoint"]["sha256"],
                    "benchmark_modes": [value["mode"] for value in benchmark_records],
                    "benchmark_records": benchmark_records,
                    "benchmark_may_gate_or_promote": False,
                }
                write_exclusive_json(
                    root / "supervisor/events" / f"{cycle_index:012d}-{event['completed_time_ns']}.cycle.json",
                    event,
                )
            except BaseException as error:
                failure = {
                    "schema": "hfw_w560_promotion_cycle_supervisor_failure_v1",
                    "status": "BLOCK_SOLVER_CYCLE_FAILED_SUPERVISOR_STOPPED",
                    "contract_sha256": contract_identity["sha256"], "cycle_index": cycle_index,
                    "error_type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                write_exclusive_json(
                    root / "supervisor/failures" / f"{cycle_index:012d}-{time.time_ns()}.failure.json",
                    failure,
                )
                raise
            completed_this_call += 1
    return {
        "status": "PASS_REQUESTED_SUPERVISED_CYCLES_COMPLETE",
        "completed_cycles": completed_this_call,
    }
