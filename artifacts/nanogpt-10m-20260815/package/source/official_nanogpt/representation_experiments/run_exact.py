"""Create-only launcher for the exact Aug-8 official nanoGPT comparator."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from representation_experiments.contract import (
    ARM_NAME,
    require_single_process_environment,
    sha256_file,
    validate_runtime_data,
)


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def validate_completed_result(output: Path, static: dict) -> dict:
    result_path = output / "result.json"
    checkpoint_path = output / "student_final.pt"
    if result_path.is_symlink() or not result_path.is_file():
        raise RuntimeError("training completed without a regular result.json")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise RuntimeError("training completed without a regular student_final.pt")
    result = json.loads(result_path.read_text())
    expected = static["static_contract"]
    resolved = expected["resolved_model"]
    required = {
        "arm": ARM_NAME,
        "architecture": "nanogpt",
        "steps": expected["steps"],
        "tokens": expected["executed_tokens"],
        "width": resolved["width"],
        "stored_parameters": resolved["stored_parameters"],
        "target_parameters": resolved["target_parameters"],
    }
    differences = {
        key: {"expected": value, "observed": result.get(key)}
        for key, value in required.items()
        if result.get(key) != value
    }
    if differences:
        raise RuntimeError(f"completed result identity drifted: {differences}")
    metrics = result.get("metrics", {})
    if set(metrics.get("long_context", {})) != {"1024", "2048"}:
        raise RuntimeError("completed result lacks exact long-context evaluations")
    return {
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "identity": required,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--teacher-snapshot-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent.parent
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    teacher_snapshot_root = Path(args.teacher_snapshot_root).expanduser().resolve()
    runtime = validate_runtime_data(
        package_root, runtime_root, teacher_snapshot_root)
    process = require_single_process_environment()
    public_runtime = {
        **runtime,
        "process": process,
    }
    if args.validate_only:
        print(json.dumps(public_runtime, indent=2, sort_keys=True))
        return 0

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("production launch requires one explicit CUDA_VISIBLE_DEVICES")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("production launch requires exactly one visible CUDA GPU")
    if output.exists() or output.is_symlink():
        raise RuntimeError("output path already exists; use a fresh attempt path")
    output.mkdir(parents=True, exist_ok=False)
    preflight_path = output / "exact_contract_preflight.json"
    write_json_exclusive(preflight_path, {
        "schema": "official_nanogpt_aug8_exact_preflight_v1",
        "status": "PASS",
        **public_runtime,
    })

    config_path = (
        package_root / "config" / "official_nanogpt_aug8_10m.json")
    previous_cwd = Path.cwd()
    previous_argv = list(sys.argv)
    previous_teacher = os.environ.get("AUG8_TEACHER_SNAPSHOT_ROOT")
    try:
        os.environ["AUG8_TEACHER_SNAPSHOT_ROOT"] = str(teacher_snapshot_root)
        os.chdir(runtime_root)
        sys.argv = [
            "distribution_distillation",
            "--config", str(config_path),
            "--arm", ARM_NAME,
            "--out", str(output),
        ]
        from representation_experiments import distribution_distillation
        distribution_distillation.main()
        completion = validate_completed_result(output, runtime)
        write_json_exclusive(output / "exact_contract_completion.json", {
            "schema": "official_nanogpt_aug8_exact_completion_v1",
            "status": "PASS",
            "preflight": {
                "path": str(preflight_path),
                "sha256": sha256_file(preflight_path),
            },
            **completion,
        })
        return 0
    except BaseException as error:
        failure_path = output / "exact_contract_failure.json"
        if not failure_path.exists():
            write_json_exclusive(failure_path, {
                "schema": "official_nanogpt_aug8_exact_failure_v1",
                "status": "BLOCKED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "preflight_sha256": sha256_file(preflight_path),
            })
        raise
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        if previous_teacher is None:
            os.environ.pop("AUG8_TEACHER_SNAPSHOT_ROOT", None)
        else:
            os.environ["AUG8_TEACHER_SNAPSHOT_ROOT"] = previous_teacher


if __name__ == "__main__":
    raise SystemExit(main())
