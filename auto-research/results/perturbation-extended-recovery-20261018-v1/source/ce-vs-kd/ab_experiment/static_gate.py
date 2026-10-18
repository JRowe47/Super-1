from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Mapping

from .common import file_identity, read_json_same_fd, sha256_bytes, write_exclusive_json


RUNTIME_PREFIXES = ("ab_experiment", "vendor")


def runtime_inventory(runtime_root: Path) -> Mapping[str, object]:
    files = {}
    for prefix in RUNTIME_PREFIXES:
        root = runtime_root / prefix
        for path in sorted(root.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                raise RuntimeError(f"stale/unmanifested Python bytecode is forbidden: {path}")
            if not path.is_file():
                continue
            files[str(path.relative_to(runtime_root))] = file_identity(path)
    identity = sha256_bytes(json.dumps(
        {name: {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}
         for name, row in files.items()},
        sort_keys=True, separators=(",", ":"),
    ).encode())
    return {
        "schema": "hfw_w560_ce_vs_kd_runtime_manifest_v1",
        "file_count": len(files),
        "files": files,
        "inventory_identity_sha256": identity,
    }


def freeze(runtime_root: Path, output: Path) -> Mapping[str, object]:
    manifest = runtime_inventory(runtime_root)
    return {**manifest, "artifact": write_exclusive_json(output, manifest)}


def _assert_no_training_calls(runtime_root: Path) -> None:
    for path in sorted((runtime_root / "ab_experiment").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                if any(name.startswith(("requests", "urllib", "boto", "google.cloud")) for name in names):
                    raise RuntimeError(f"network-capable import in model runtime: {path.name}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"backward", "step"}:
                    raise RuntimeError(f"training call {node.func.attr} in {path.name}")


def validate_phase_config(path: Path) -> Mapping[str, object]:
    config = read_json_same_fd(path)
    phase = str(config["phase"])
    encoded = path.read_bytes().lower()
    if phase.endswith("_SEARCH"):
        if set(config["banks"]) != {"SEARCH"}:
            raise RuntimeError("SEARCH phase has non-SEARCH token role")
        if b"final" in encoded or b"verify" in encoded:
            raise RuntimeError("SEARCH phase exposes VERIFY/FINAL capability")
        if phase == "CE_SEARCH" and "teacher_outputs" in config["inputs"]:
            raise RuntimeError("CE SEARCH exposes teacher logits")
        if phase == "KD_SEARCH" and config["inputs"]["teacher_outputs"]["roles_available"] != ["SEARCH"]:
            raise RuntimeError("KD SEARCH teacher role drift")
    elif phase.endswith("_VERIFY"):
        if set(config["banks"]) != {"VERIFY"}:
            raise RuntimeError("VERIFY phase has non-VERIFY token role")
        if b"final" in encoded or b"search.u32" in encoded:
            raise RuntimeError("VERIFY phase exposes SEARCH/FINAL token capability")
        if phase == "CE_VERIFY" and "teacher_outputs" in config["inputs"]:
            raise RuntimeError("CE VERIFY exposes teacher logits")
        if phase == "KD_VERIFY" and config["inputs"]["teacher_outputs"]["roles_available"] != ["VERIFY"]:
            raise RuntimeError("KD VERIFY teacher role drift")
    elif phase == "FINAL":
        if set(config["banks"]) != {"FINAL"}:
            raise RuntimeError("FINAL phase has non-FINAL token role")
        if config["inputs"]["teacher_outputs"]["roles_available"] != ["FINAL"]:
            raise RuntimeError("FINAL teacher role drift")
        if set(config["phase_inputs"]) != {
            "preregistered_config_sha256", "ce_selection", "kd_selection",
            "continuation_search_bank_sha256", "continuation_verify_bank_sha256",
        }:
            raise RuntimeError("FINAL phase selection/opaque continuation binding drift")
    else:
        raise RuntimeError(f"unsupported phase config: {phase}")
    return {"phase": phase, "sha256": file_identity(path)["sha256"]}


def gate(
    runtime_root: Path, manifest_path: Path, phase_configs: list[Path],
) -> Mapping[str, object]:
    expected = read_json_same_fd(manifest_path)
    observed = runtime_inventory(runtime_root)
    if expected["inventory_identity_sha256"] != observed["inventory_identity_sha256"]:
        raise RuntimeError("runtime inventory drift")
    if expected["file_count"] != observed["file_count"]:
        raise RuntimeError("runtime file-count drift")
    _assert_no_training_calls(runtime_root)
    phases = [validate_phase_config(path) for path in phase_configs]
    return {
        "schema": "hfw_w560_ce_vs_kd_static_gate_receipt_v1",
        "status": "PASS_RUNTIME_AND_LEAST_AUTHORITY_PHASE_GATES",
        "runtime_manifest_sha256": file_identity(manifest_path)["sha256"],
        "runtime_inventory_identity_sha256": observed["inventory_identity_sha256"],
        "phase_configs": phases,
        "cloud_backprop_adam_or_optimizer_capability": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--runtime-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--runtime-root", type=Path, required=True)
    gate_parser.add_argument("--manifest", type=Path, required=True)
    gate_parser.add_argument("--phase-config", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.runtime_root, args.output)
    else:
        result = gate(args.runtime_root, args.manifest, args.phase_config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
