from __future__ import annotations

import argparse
import json
from pathlib import Path

from .local_candidate_import import import_completed_candidate_chain
from .local_config import load_config, runtime_root
from .local_ledger import LocalLedger, initialize_local_ledger
from .local_promoter import evaluate_promote_and_allocate_next
from .local_runner import initialize_continuous_run, run_continuously
from .local_verify_bank import load_local_verify


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def status(config_path: Path, ledger_root: Path | None) -> dict[str, object]:
    config, identity = load_config(config_path)
    values, capability = load_local_verify(config, runtime_root(config_path))
    result = {
        "schema": "hfw_w560_mps_cumulative_local_status_v1",
        "status": "PASS_CONTINUOUS_CANONICAL_MPS_SOLVER_LAUNCH_AUTHORIZED",
        "config": identity,
        "local_verify": capability,
        "local_verify_shape": list(values.shape),
        "local_verify_is_reused_optimization_data_not_held_out": True,
        "long_local_solver_authorized": config["authority"]["local_solver_launch_authorized"],
        "cloud_access_authorized": False,
        "training_or_backprop_authorized": False,
        "adam_or_any_optimizer_authorized": False,
    }
    if ledger_root is not None:
        ledger = LocalLedger(ledger_root, config_path)
        result["current_head"] = ledger.current_head()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="All-local cumulative MPS solver with duplicated LOCAL_VERIFY promotion")
    commands = parser.add_subparsers(dest="command", required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--config", type=Path, required=True)
    status_parser.add_argument("--ledger", type=Path)
    init = commands.add_parser("init-ledger")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--ledger", type=Path, required=True)
    imported = commands.add_parser("import-completed-chain")
    imported.add_argument("--config", type=Path, required=True)
    imported.add_argument("--ledger", type=Path, required=True)
    imported.add_argument("--source-proposal", type=Path, required=True)
    imported.add_argument("--source-terminal", type=Path, required=True)
    imported.add_argument("--expected-source-proposal-sha256", required=True)
    verify = commands.add_parser("verify-promote-allocate-next")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--ledger", type=Path, required=True)
    verify.add_argument("--proposal", type=Path, required=True)
    allocate = commands.add_parser("allocate-next-job")
    allocate.add_argument("--config", type=Path, required=True)
    allocate.add_argument("--ledger", type=Path, required=True)
    init_run = commands.add_parser("init-continuous-run")
    init_run.add_argument("--config", type=Path, required=True)
    init_run.add_argument("--ledger", type=Path, required=True)
    init_run.add_argument("--run", type=Path, required=True)
    continuous = commands.add_parser("run-continuous")
    continuous.add_argument("--config", type=Path, required=True)
    continuous.add_argument("--ledger", type=Path, required=True)
    continuous.add_argument("--run", type=Path, required=True)
    continuous.add_argument("--maximum-cycles", type=int)
    args = parser.parse_args()
    if args.command == "status":
        result = status(args.config, args.ledger)
    elif args.command == "init-ledger":
        result = initialize_local_ledger(args.config, args.ledger)
    elif args.command == "import-completed-chain":
        ledger = LocalLedger(args.ledger, args.config)
        result = import_completed_candidate_chain(
            ledger, args.source_proposal, args.source_terminal,
            args.expected_source_proposal_sha256,
        )
    elif args.command == "verify-promote-allocate-next":
        result = evaluate_promote_and_allocate_next(args.config, args.ledger, args.proposal)
    elif args.command == "allocate-next-job":
        result = LocalLedger(args.ledger, args.config).allocate_next_job()
    elif args.command == "init-continuous-run":
        result = initialize_continuous_run(args.config, args.ledger, args.run)
    else:
        result = run_continuously(
            args.config, args.ledger, args.run, maximum_cycles=args.maximum_cycles,
        )
    _print(result)
    return 1 if str(result.get("status", "")).startswith("BLOCK_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
