from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import bind_imported_assets, validate_assets
from .benchmark import benchmark
from .config import validate_pinned_inputs
from .lineage import solver_idle_boundary, validate_accepted_promotion
from .supervisor import diagnose_solver_child_environment, initialize, supervise


def _compact(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _compact(item) for key, item in value.items()
            if key not in ("config", "model", "prior_head", "current_head", "decision", "cycle", "manifest", "examples")
        }
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(prog="w560-promotion-benchmark")
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-package")
    commands.add_parser("bind-local-assets")
    commands.add_parser("validate-assets")
    commands.add_parser("init-supervisor")
    commands.add_parser("diagnose-solver-child")
    run = commands.add_parser("supervise")
    run.add_argument("--maximum-supervised-cycles", type=int)
    trigger = commands.add_parser("validate-trigger")
    trigger.add_argument("--cycle-receipt", type=Path, required=True)
    evaluate = commands.add_parser("benchmark")
    evaluate.add_argument("--cycle-receipt", type=Path, required=True)
    evaluate.add_argument("--mode", choices=("canary", "full"), required=True)
    args = parser.parse_args()
    if args.command == "validate-package":
        result = validate_pinned_inputs(args.config)
    elif args.command == "bind-local-assets":
        result = bind_imported_assets(args.config)
    elif args.command == "validate-assets":
        result = validate_assets(args.config)
    elif args.command == "init-supervisor":
        result = initialize(args.config)
    elif args.command == "diagnose-solver-child":
        result = diagnose_solver_child_environment(args.config)
    elif args.command == "supervise":
        result = supervise(args.config, args.maximum_supervised_cycles)
    elif args.command == "validate-trigger":
        config = validate_pinned_inputs(args.config)["config"]
        with solver_idle_boundary(config):
            result = validate_accepted_promotion(args.config, args.cycle_receipt)
    else:
        result = benchmark(args.config, args.cycle_receipt, args.mode)
    print(json.dumps(_compact(result), sort_keys=True))


if __name__ == "__main__":
    main()
