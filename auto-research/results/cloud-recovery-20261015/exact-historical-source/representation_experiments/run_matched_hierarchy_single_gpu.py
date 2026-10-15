"""Resume-safe sequential launcher for one authorized GPU worker."""
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


def run_logged(command, log_path, environment):
    with log_path.open("ab") as stream:
        subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT,
            check=True, env=environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    config = load_json(config_path)
    minimum_free = int(config["safety"]["minimum_free_gb"])
    source_root = Path(__file__).resolve().parent.parent
    script_root = Path(__file__).resolve().parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for arm in config["arms"]:
        if free_gb(output) < minimum_free:
            raise RuntimeError("free disk is below the manifest safety floor")
        result = output / arm / "result.json"
        if result.is_file():
            continue
        (output / arm).mkdir(exist_ok=True)
        run_logged([
            sys.executable, "-u", str(script_root / "distribution_distillation.py"),
            "--config", str(config_path), "--arm", arm,
            "--out", str(output / arm),
        ], logs / f"{arm}.log", environment)

    limit = str(int(config["benchmark"]["limit"]))
    for arm in config["arms"]:
        if free_gb(output) < minimum_free:
            raise RuntimeError("paused before benchmark: disk is below safety floor")
        benchmark = output / arm / "benchmarks.json"
        if benchmark.is_file():
            continue
        run_logged([
            sys.executable, "-u", str(script_root / "distribution_benchmarks.py"),
            "--config", str(config_path), "--arm", arm,
            "--checkpoint", str(output / arm / "student_final.pt"),
            "--limit", limit, "--out", str(benchmark),
        ], logs / f"{arm}-benchmarks.log", environment)

    subprocess.run([
        sys.executable, str(script_root / "summarize_matched_hierarchy_suite.py"),
        "--config", str(config_path), "--root", str(output),
        "--out", str(output / "summary.json"),
    ], check=True, env=environment)


if __name__ == "__main__":
    main()
