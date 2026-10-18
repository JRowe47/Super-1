from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import bootstrap_ledger
from .committer import commit_proposal
from .ledger import Ledger
from .proposal_worker import run_proposal_job
from .relocation import load_config, prepare_relocated_parent
from .supervisor import run_continuously
from .token_commitment import prepare_token_commitment


def _print(value) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def result_exit_code(value) -> int:
    return 1 if str(value.get("status", "")).startswith("BLOCK_") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Append-only continuous W560 perturbation controller")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parent = commands.add_parser("prepare-parent")
    prepare_parent.add_argument("--config", type=Path, required=True)
    prepare_parent.add_argument("--output", type=Path, required=True)
    tokens = commands.add_parser("prepare-tokens")
    tokens.add_argument("--config", type=Path, required=True)
    tokens.add_argument("--output", type=Path, required=True)
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--config", type=Path, required=True)
    bootstrap.add_argument("--token-manifest", type=Path, required=True)
    bootstrap.add_argument("--ledger", type=Path, required=True)
    job = commands.add_parser("create-job")
    job.add_argument("--config", type=Path, required=True)
    job.add_argument("--token-manifest", type=Path, required=True)
    job.add_argument("--ledger", type=Path, required=True)
    job.add_argument("--group", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--token-manifest", type=Path, required=True)
    worker.add_argument("--ledger", type=Path, required=True)
    worker.add_argument("--job", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    commit = commands.add_parser("commit")
    commit.add_argument("--config", type=Path, required=True)
    commit.add_argument("--token-manifest", type=Path, required=True)
    commit.add_argument("--ledger", type=Path, required=True)
    commit.add_argument("--proposal", type=Path, required=True)
    supervise = commands.add_parser("supervise")
    supervise.add_argument("--config", type=Path, required=True)
    supervise.add_argument("--token-manifest", type=Path, required=True)
    supervise.add_argument("--ledger", type=Path, required=True)
    supervise.add_argument("--max-waves", type=int)
    status = commands.add_parser("status")
    status.add_argument("--config", type=Path, required=True)
    status.add_argument("--token-manifest", type=Path, required=True)
    status.add_argument("--ledger", type=Path, required=True)
    status.add_argument("--verify-checkpoints", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "prepare-parent":
        result = prepare_relocated_parent(arguments.config, arguments.output)
    elif arguments.command == "prepare-tokens":
        result = prepare_token_commitment(arguments.config, arguments.output)
    elif arguments.command == "bootstrap":
        result = bootstrap_ledger(arguments.config, arguments.token_manifest, arguments.ledger)
    elif arguments.command == "create-job":
        ledger = Ledger(arguments.ledger)
        runtime_inputs = ledger.validate_runtime_inputs(
            arguments.config, arguments.token_manifest
        )
        config = runtime_inputs["config_value"]
        result = ledger.create_job(
            group=arguments.group,
            root_seed=int(config["controller"]["root_seed"]),
            allowed_groups=config["controller"]["groups"],
        )
    elif arguments.command == "worker":
        result = run_proposal_job(
            arguments.config, arguments.token_manifest, arguments.ledger,
            arguments.job, arguments.output,
        )
    elif arguments.command == "commit":
        result = commit_proposal(
            arguments.config, arguments.token_manifest, arguments.ledger,
            arguments.proposal,
        )
    elif arguments.command == "supervise":
        result = run_continuously(
            arguments.config, arguments.token_manifest, arguments.ledger,
            max_waves_override=arguments.max_waves,
        )
    else:
        ledger = Ledger(arguments.ledger)
        ledger.validate_runtime_inputs(arguments.config, arguments.token_manifest)
        result = ledger.validate_history(arguments.verify_checkpoints)
    _print(result)
    return result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
