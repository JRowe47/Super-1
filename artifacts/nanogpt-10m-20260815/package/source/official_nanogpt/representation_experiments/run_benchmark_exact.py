"""Create-only historical four-task benchmark phase for the comparator."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from representation_experiments.contract import (
    ARM_NAME,
    require_single_process_environment,
    sha256_file,
    validate_runtime_data,
)
from representation_experiments.run_exact import write_json_exclusive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--teacher-snapshot-root", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent.parent
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    teacher_root = Path(args.teacher_snapshot_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    runtime = validate_runtime_data(package_root, runtime_root, teacher_root)
    require_single_process_environment()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("benchmark requires one explicit CUDA_VISIBLE_DEVICES")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible CUDA GPU")

    completion_path = run_root / "exact_contract_completion.json"
    if completion_path.is_symlink() or not completion_path.is_file():
        raise RuntimeError("benchmark requires an exact PASS training completion")
    completion = json.loads(completion_path.read_text())
    if completion.get("status") != "PASS":
        raise RuntimeError("training completion is not PASS")
    checkpoint_path = run_root / "student_final.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != completion.get("checkpoint_sha256"):
        raise RuntimeError("final checkpoint differs from training completion")
    benchmark_path = run_root / "benchmarks.json"
    receipt_path = run_root / "exact_benchmark_completion.json"
    if benchmark_path.exists() or receipt_path.exists():
        raise RuntimeError("benchmark artifacts already exist; never overwrite them")

    config_path = package_root / "config" / "official_nanogpt_aug8_10m.json"
    previous_cwd = Path.cwd()
    previous_argv = list(sys.argv)
    previous_teacher = os.environ.get("AUG8_TEACHER_SNAPSHOT_ROOT")
    try:
        os.environ["AUG8_TEACHER_SNAPSHOT_ROOT"] = str(teacher_root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.chdir(runtime_root)
        sys.argv = [
            "distribution_benchmarks",
            "--config", str(config_path),
            "--arm", ARM_NAME,
            "--checkpoint", str(checkpoint_path),
            "--limit", "1000",
            "--out", str(benchmark_path),
        ]
        from representation_experiments import distribution_benchmarks
        distribution_benchmarks.main()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        if previous_teacher is None:
            os.environ.pop("AUG8_TEACHER_SNAPSHOT_ROOT", None)
        else:
            os.environ["AUG8_TEACHER_SNAPSHOT_ROOT"] = previous_teacher

    result = json.loads(benchmark_path.read_text())
    expected = runtime["static_contract"]["resolved_model"]
    if (result.get("arm") != ARM_NAME
            or result.get("limit") != 1000
            or result.get("stored_parameters") != expected["stored_parameters"]
            or set(result.get("scores", {}))
            != {"lambada", "hellaswag", "arc_easy", "piqa"}):
        raise RuntimeError("benchmark result identity drifted")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during benchmark")
    write_json_exclusive(receipt_path, {
        "schema": "official_nanogpt_aug8_exact_benchmark_completion_v1",
        "status": "PASS",
        "training_completion_sha256": sha256_file(completion_path),
        "checkpoint_sha256": checkpoint_sha256,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "limit": 1000,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
