#!/usr/bin/env python3
"""Config-driven verifier and create-only launcher for both experiment arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_package() -> dict:
    with (ROOT / "package.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_regular(path: Path, expected_size: int, expected_sha256: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required regular file is missing: {path}")
    stat = path.stat()
    if stat.st_size != expected_size:
        raise RuntimeError(f"size mismatch for {path}: {stat.st_size} != {expected_size}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed}")
    return {"path": str(path), "bytes": stat.st_size, "sha256": observed}


def verify_inputs(package: dict, runtime_root: Path, teacher_root: Path) -> dict:
    runtime_root = runtime_root.expanduser().resolve()
    teacher_root = teacher_root.expanduser().resolve()
    inputs = package["inputs"]
    data_root = runtime_root / inputs["data_directory_relative_to_runtime"]
    train = verify_regular(
        data_root / inputs["train"]["filename"],
        int(inputs["train"]["bytes"]),
        inputs["train"]["sha256"],
    )
    validation = verify_regular(
        data_root / inputs["validation"]["filename"],
        int(inputs["validation"]["bytes"]),
        inputs["validation"]["sha256"],
    )
    teacher = verify_regular(
        teacher_root / inputs["teacher"]["weights_filename"],
        int(inputs["teacher"]["weights_bytes"]),
        inputs["teacher"]["weights_sha256"],
    )
    return {
        "status": "PASS_INPUT_IDENTITY",
        "runtime_root": str(runtime_root),
        "data_root": str(data_root),
        "teacher_snapshot_root": str(teacher_root),
        "train": train,
        "validation": validation,
        "teacher": teacher,
    }


def require_absent(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"{label} already exists; use a fresh path: {path}")


def base_environment(source_root: Path, gpu: str) -> dict[str, str]:
    for key, expected in {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}.items():
        observed = os.environ.get(key)
        if observed is not None and observed != expected:
            raise RuntimeError(f"{key} must be {expected}, observed {observed}")
    env = dict(os.environ)
    env.update({
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "CUDA_VISIBLE_DEVICES": gpu,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": str(source_root),
    })
    return env


def display_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    public_env = {
        key: env[key]
        for key in [
            "CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_RANK",
            "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE",
            "PYTHONPATH",
        ]
    }
    print(json.dumps({
        "cwd": str(cwd),
        "environment": public_env,
        "command": command,
        "shell_display": " ".join(shlex.quote(part) for part in command),
    }, indent=2, sort_keys=True))


def gpu_preflight(python: str, env: dict[str, str], cwd: Path) -> None:
    code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "assert torch.cuda.device_count() == 1, torch.cuda.device_count(); "
        "print(torch.cuda.get_device_name(0))"
    )
    subprocess.run([python, "-B", "-c", code], cwd=cwd, env=env, check=True)


def run_command(command: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> int:
    display_command(command, cwd, env)
    if dry_run:
        return 0
    gpu_preflight(command[0], env, cwd)
    subprocess.run(command, cwd=cwd, env=env, check=True)
    return 0


def materialize_hierarchy_config(
    package: dict,
    runtime_root: Path,
    teacher_root: Path,
    output: Path,
    execute: bool,
) -> Path:
    source_path = ROOT / package["paths"]["historical_config"]
    control = output.parent / f"{output.name}.reproduction-control"
    derived = control / "hierarchy_config.json"
    require_absent(control, "hierarchy control directory")
    if not execute:
        return derived
    with source_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    config["data"] = str(
        runtime_root.resolve() / package["inputs"]["data_directory_relative_to_runtime"])
    config["teacher"] = str(teacher_root.resolve())
    config["reproduction_binding"] = {
        "immutable_parent_config": package["paths"]["historical_config"],
        "immutable_parent_config_sha256": package["provenance"]["historical_config_sha256"],
        "only_runtime_paths_changed": ["data", "teacher"],
    }
    control.mkdir(parents=False, exist_ok=False)
    with derived.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return derived


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--runtime-root", type=Path, required=True)
    verify.add_argument("--teacher-snapshot-root", type=Path, required=True)

    for name in [
        "run-hierarchy", "benchmark-hierarchy", "preflight-nanogpt",
        "run-nanogpt", "benchmark-nanogpt",
    ]:
        command = subparsers.add_parser(name)
        command.add_argument("--runtime-root", type=Path, required=True)
        command.add_argument("--teacher-snapshot-root", type=Path, required=True)
        target = "--run-root" if name.startswith("benchmark-") else "--out"
        command.add_argument(target, type=Path, required=True)
        command.add_argument("--gpu", default="0")
        command.add_argument("--python", default=sys.executable)
        command.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "tools/verify_package.py")],
        cwd=ROOT,
        check=True,
    )
    package = load_package()
    inputs = verify_inputs(package, args.runtime_root, args.teacher_snapshot_root)
    if args.action == "verify-inputs":
        print(json.dumps(inputs, indent=2, sort_keys=True))
        return 0

    runtime_root = args.runtime_root.expanduser().resolve()
    teacher_root = args.teacher_snapshot_root.expanduser().resolve()
    historical_source = ROOT / package["paths"]["historical_source"]
    nano_source = ROOT / package["paths"]["nanogpt_source"]

    if args.action == "run-hierarchy":
        output = args.out.expanduser().resolve()
        require_absent(output, "hierarchy output")
        config = materialize_hierarchy_config(
            package, runtime_root, teacher_root, output, not args.dry_run)
        env = base_environment(historical_source, args.gpu)
        command = [
            args.python, "-B", "-u",
            str(historical_source / "representation_experiments/distribution_distillation.py"),
            "--config", str(config),
            "--arm", package["comparison"]["hierarchy_arm"],
            "--out", str(output),
        ]
        return run_command(command, historical_source, env, args.dry_run)

    if args.action == "benchmark-hierarchy":
        run_root = args.run_root.expanduser().resolve()
        checkpoint = run_root / "student_final.pt"
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise RuntimeError(f"missing hierarchy final checkpoint: {checkpoint}")
        benchmark = run_root / "benchmarks.json"
        require_absent(benchmark, "hierarchy benchmark")
        config = run_root.parent / f"{run_root.name}.reproduction-control/hierarchy_config.json"
        if config.is_symlink() or not config.is_file():
            raise RuntimeError(f"missing derived hierarchy config: {config}")
        env = base_environment(historical_source, args.gpu)
        command = [
            args.python, "-B", "-u",
            str(historical_source / "representation_experiments/distribution_benchmarks.py"),
            "--config", str(config),
            "--arm", package["comparison"]["hierarchy_arm"],
            "--checkpoint", str(checkpoint),
            "--limit", str(package["benchmark_limit_per_task"]),
            "--out", str(benchmark),
        ]
        return run_command(command, historical_source, env, args.dry_run)

    output = (
        args.run_root.expanduser().resolve()
        if args.action == "benchmark-nanogpt"
        else args.out.expanduser().resolve()
    )
    env = base_environment(nano_source, args.gpu)
    if args.action == "preflight-nanogpt":
        require_absent(output, "nanoGPT output")
        command = [
            args.python, "-B", "-m", "representation_experiments.run_exact",
            "--runtime-root", str(runtime_root),
            "--teacher-snapshot-root", str(teacher_root),
            "--out", str(output), "--validate-only",
        ]
    elif args.action == "run-nanogpt":
        require_absent(output, "nanoGPT output")
        command = [
            args.python, "-B", "-u", "-m", "representation_experiments.run_exact",
            "--runtime-root", str(runtime_root),
            "--teacher-snapshot-root", str(teacher_root),
            "--out", str(output),
        ]
    else:
        command = [
            args.python, "-B", "-u", "-m",
            "representation_experiments.run_benchmark_exact",
            "--runtime-root", str(runtime_root),
            "--teacher-snapshot-root", str(teacher_root),
            "--run-root", str(output),
        ]
    return run_command(command, nano_source, env, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
