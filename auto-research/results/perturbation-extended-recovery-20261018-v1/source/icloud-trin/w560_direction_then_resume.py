#!/usr/bin/env python3
"""Wait for winner-direction descent, refresh history, then resume fixed candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-pid", type=int, required=True)
    parser.add_argument("--direction-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--source-snapshot-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--start-candidate-order", type=int, required=True)
    parser.add_argument("--verifier-script", type=Path, required=True)
    parser.add_argument("--chart-watch-script", type=Path, required=True)
    parser.add_argument("--chart-refresh-script", type=Path, required=True)
    parser.add_argument("--fragment-renderer", type=Path, required=True)
    parser.add_argument("--png-renderer", type=Path, required=True)
    parser.add_argument("--inline-html", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--node-path", type=Path, required=True)
    args = parser.parse_args()

    while alive(args.direction_pid):
        time.sleep(10)
    direction_terminal = args.direction_root / "PROMOTION.json"
    if not direction_terminal.exists():
        atomic_json(args.direction_root / "POST_DIRECTION_RESUME.json", {
            "schema": "hfw_w560_post_direction_resume_v1",
            "status": "BLOCKED_DIRECTION_RUN_FAILED_WITHOUT_TERMINAL",
            "direction_pid": args.direction_pid,
            "completed_unix_ns": time.time_ns(),
        })
        raise RuntimeError("direction run exited without PROMOTION.json")

    run([
        args.python, str(args.chart_refresh_script),
        "--package-root", str(args.package_root),
        "--wave-root", str(args.wave_root),
        "--verification-root", str(args.direction_root),
        "--inline-html", str(args.inline_html),
    ])
    standalone = args.direction_root / "w560-full-improvement-history-standalone.html"
    run([args.python, str(args.fragment_renderer), str(args.inline_html), str(standalone)])
    run([
        str(args.node), str(args.png_renderer), standalone.as_uri(), str(args.png),
    ], env={**os.environ, "NODE_PATH": str(args.node_path)})

    args.resume_root.mkdir(parents=True, exist_ok=True)
    log_path = args.resume_root / "verifier.log"
    with log_path.open("ab", buffering=0) as log:
        verifier = subprocess.Popen([
            args.python, "-B", str(args.verifier_script),
            "--package-root", str(args.package_root),
            "--wave-root", str(args.wave_root),
            "--output-root", str(args.resume_root),
            "--promote-after-exhaustive",
            "--minimum-relative-improvement-override", "0",
            "--successor-snapshot-root", str(args.source_snapshot_root),
            "--start-candidate-order", str(args.start_candidate_order),
        ], stdout=log, stderr=subprocess.STDOUT)
    chart_log_path = args.resume_root / "chart-watch.log"
    with chart_log_path.open("ab", buffering=0) as chart_log:
        chart_watch = subprocess.Popen([
            args.python, "-B", str(args.chart_watch_script),
            "--pid", str(verifier.pid),
            "--poll-seconds", "5",
            "--package-root", str(args.package_root),
            "--wave-root", str(args.wave_root),
            "--verification-root", str(args.resume_root),
            "--inline-html", str(args.inline_html),
            "--png", str(args.png),
            "--refresh-script", str(args.chart_refresh_script),
            "--fragment-renderer", str(args.fragment_renderer),
            "--png-renderer", str(args.png_renderer),
            "--python", args.python,
            "--node", str(args.node),
            "--node-path", str(args.node_path),
        ], stdout=chart_log, stderr=subprocess.STDOUT)
    sleep_guard = subprocess.Popen(["/usr/bin/caffeinate", "-dimsu", "-w", str(verifier.pid)])
    returncode = verifier.wait()
    chart_returncode = chart_watch.wait()
    sleep_guard.wait()
    terminal = {
        "schema": "hfw_w560_post_direction_resume_v1",
        "status": "COMPLETED" if returncode == 0 and chart_returncode == 0 else "FAILED",
        "direction_promotion": {
            "path": str(direction_terminal), "sha256": sha256(direction_terminal),
        },
        "resume_pid": verifier.pid,
        "resume_returncode": returncode,
        "chart_watch_pid": chart_watch.pid,
        "chart_watch_returncode": chart_returncode,
        "start_candidate_order": args.start_candidate_order,
        "completed_unix_ns": time.time_ns(),
    }
    atomic_json(args.direction_root / "POST_DIRECTION_RESUME.json", terminal)
    if terminal["status"] != "COMPLETED":
        raise RuntimeError("post-direction resume failed")


if __name__ == "__main__":
    main()
