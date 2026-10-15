"""Resume-safe launcher for the matched hierarchy/function-weight study."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from representation_experiments.common import load_json


def free_gb(path: Path) -> int:
    return shutil.disk_usage(path).free // (1024 ** 3)


def run_parallel(commands, gpu_ids, logs, environment):
    if len(commands) > len(gpu_ids):
        raise ValueError("the manifest has more simultaneous arms than GPUs")
    processes = []
    for gpu, (name, command) in zip(gpu_ids, commands):
        log_path = logs / f"{name}.log"
        stream = log_path.open("ab")
        child_environment = dict(environment)
        child_environment["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT,
            env=child_environment)
        processes.append((name, process, stream, log_path))
    failures = []
    for name, process, stream, log_path in processes:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append(f"{name} exited {return_code}; inspect {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default=os.environ.get("GPU_LIST", "0,1,2,3"))
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    config = load_json(config_path)
    minimum_free = int(config["safety"]["minimum_free_gb"])
    if free_gb(output) < minimum_free:
        raise RuntimeError("free disk is below the manifest safety floor")
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    arms = list(config["arms"])
    if len(gpu_ids) != len(arms):
        raise ValueError("the matched study requires exactly one GPU per arm")
    source_root = Path(__file__).resolve().parent.parent
    script_root = Path(__file__).resolve().parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    environment["HF_HUB_OFFLINE"] = "1"

    training_commands = []
    for arm in arms:
        if (output / arm / "result.json").is_file():
            continue
        (output / arm).mkdir(exist_ok=True)
        training_commands.append((arm, [
            sys.executable, "-u", str(script_root / "distribution_distillation.py"),
            "--config", str(config_path), "--arm", arm,
            "--out", str(output / arm),
        ]))
    run_parallel(training_commands, gpu_ids, logs, environment)

    if free_gb(output) < minimum_free:
        raise RuntimeError("paused before benchmarks: disk is below safety floor")
    benchmark_commands = []
    limit = str(int(config["benchmark"]["limit"]))
    for arm in arms:
        benchmark = output / arm / "benchmarks.json"
        if benchmark.is_file():
            continue
        benchmark_commands.append((f"{arm}-benchmarks", [
            sys.executable, "-u", str(script_root / "distribution_benchmarks.py"),
            "--config", str(config_path), "--arm", arm,
            "--checkpoint", str(output / arm / "student_final.pt"),
            "--limit", limit, "--out", str(benchmark),
        ]))
    run_parallel(benchmark_commands, gpu_ids, logs, environment)

    subprocess.run([
        sys.executable, str(script_root / "summarize_matched_hierarchy_suite.py"),
        "--config", str(config_path), "--root", str(output),
        "--out", str(output / "summary.json"),
    ], check=True, env=environment)


if __name__ == "__main__":
    main()
