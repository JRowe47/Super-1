from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
import sys
from typing import Iterator, Mapping

from .common import exclusive_lock, read_json_same_fd
from .config import validate_pinned_inputs


@contextlib.contextmanager
def active_modules(config: Mapping[str, object]) -> Iterator[tuple[object, object]]:
    runtime = str(config["active_controller"]["runtime_root"])
    if runtime in sys.path:
        raise RuntimeError("active controller runtime already present in sys.path")
    preexisting = [name for name in sys.modules if name == "hfw_continuous" or name.startswith("hfw_continuous.")]
    if preexisting:
        raise RuntimeError(f"active controller module namespace already loaded: {preexisting}")
    sys.path.insert(0, runtime)
    try:
        ledger_module = importlib.import_module("hfw_continuous.local_ledger")
        state_module = importlib.import_module("hfw_continuous.state")
        yield ledger_module, state_module
    finally:
        sys.path.remove(runtime)
        for name in [name for name in sys.modules if name == "hfw_continuous" or name.startswith("hfw_continuous.") or name == "vendor" or name.startswith("vendor.")]:
            sys.modules.pop(name, None)


@contextlib.contextmanager
def solver_idle_boundary(config: Mapping[str, object]) -> Iterator[None]:
    active = config["active_controller"]
    with exclusive_lock(Path(str(config["locks"]["host_mps_lane"])), nonblocking=True):
        with exclusive_lock(Path(str(active["runner_lock_path"])), nonblocking=True):
            with exclusive_lock(Path(str(active["lineage_lock_path"])), nonblocking=True):
                yield


def _cycle_path(config: Mapping[str, object], cycle_receipt: Path) -> Path:
    path = Path(cycle_receipt).resolve()
    attempts = Path(str(config["active_controller"]["run_attempts_root"])).resolve()
    try:
        path.relative_to(attempts)
    except ValueError as error:
        raise RuntimeError("cycle receipt is outside the pinned active run") from error
    if path.name != "CUMULATIVE_CYCLE_COMPLETE.json":
        raise RuntimeError("unexpected cycle receipt filename")
    return path


def validate_accepted_promotion(config_path: Path, cycle_receipt: Path) -> Mapping[str, object]:
    package = validate_pinned_inputs(config_path)
    config = package["config"]
    cycle, cycle_identity = read_json_same_fd(_cycle_path(config, cycle_receipt))
    if cycle.get("schema") != "hfw_w560_mps_cumulative_local_cycle_receipt_v1" or cycle.get("status") != "PASS_CYCLE_COMPLETE_NEXT_JOB_BOUND_CURRENT_HEAD":
        raise RuntimeError("cycle receipt schema/status drift")
    if cycle.get("accepted") is not True:
        raise RuntimeError("benchmark trigger must be an accepted promotion cycle")
    active = config["active_controller"]
    with active_modules(config) as (ledger_module, _):
        ledger = ledger_module.LocalLedger(Path(str(active["lineage_root"])), Path(str(active["config_path"])))
        heads = ledger.heads()
    current = heads[-1]
    if len(heads) < 2:
        raise RuntimeError("accepted promotion has no prior head")
    prior = heads[-2]
    required = {
        "current_checkpoint_sha256": current["checkpoint"]["sha256"],
        "current_head_record_sha256": current["record_sha256"],
        "decision_sha256": current["decision_sha256"],
        "next_job_base_checkpoint_sha256": current["checkpoint"]["sha256"],
        "next_job_base_head_record_sha256": current["record_sha256"],
    }
    for key, expected in required.items():
        if cycle.get(key) != expected:
            raise RuntimeError(f"cycle/current lineage binding drift: {key}")
    if current.get("parent_head_record_sha256") != prior["record_sha256"] or current.get("parent_checkpoint_sha256") != prior["checkpoint"]["sha256"]:
        raise RuntimeError("current/prior promotion chain drift")
    decision_path = Path(str(active["lineage_root"])) / "decisions" / f"{current['decision_sha256']}.decision.json"
    decision, decision_identity = read_json_same_fd(decision_path, str(current["decision_sha256"]))
    if decision.get("accepted") is not True:
        raise RuntimeError("current head decision is not accepted")
    if decision.get("base_head_record_sha256") != prior["record_sha256"] or decision.get("base_checkpoint_sha256") != prior["checkpoint"]["sha256"]:
        raise RuntimeError("decision prior binding drift")
    if decision.get("child_checkpoint", {}).get("sha256") != current["checkpoint"]["sha256"]:
        raise RuntimeError("decision child/current checkpoint drift")
    return {
        "status": "PASS_ACCEPTED_PROMOTION_TRIGGER_BOUND",
        "package": package,
        "cycle": cycle,
        "cycle_identity": cycle_identity,
        "decision": decision,
        "decision_identity": decision_identity,
        "prior_head": prior,
        "current_head": current,
    }


def load_exact_head_model(config: Mapping[str, object], head: Mapping[str, object], device: object):
    active = config["active_controller"]
    with active_modules(config) as (_, state_module):
        active_config = state_module.load_config(Path(str(active["config_path"])))
        loaded, protocol_receipt = state_module.load_head_model(
            Path(str(active["config_path"])), head, device, config=active_config,
        )
    if loaded.checkpoint_sha256 != head["checkpoint"]["sha256"]:
        raise RuntimeError("loader returned wrong checkpoint")
    return loaded, protocol_receipt
