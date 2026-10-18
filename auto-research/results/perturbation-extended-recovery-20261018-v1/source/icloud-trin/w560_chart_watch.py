#!/usr/bin/env python3
"""Refresh and archive the W560 chart whenever a new strict MPS improver appears."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def passing_inventory(root: Path) -> tuple[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "candidate-results").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("passes_minimum_replicated_improvement") is not True:
            continue
        rows.append({
            "candidate_order": int(row["candidate_order"]),
            "task_index": int(row["task_index"]),
            "group": str(row["group"]),
            "ce": float(row["mps_verify_replicated_mean"]),
            "duplicate": float(row["duplicate_max_abs_sequence_difference"]),
            "result_path": str(path),
            "result_sha256": sha256(path),
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), rows


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def refresh(args: argparse.Namespace, inventory_sha: str, passers: list[dict[str, object]]) -> None:
    run_checked([
        args.python,
        str(args.refresh_script),
        "--package-root", str(args.package_root),
        "--wave-root", str(args.wave_root),
        "--verification-root", str(args.verification_root),
        "--inline-html", str(args.inline_html),
    ])
    standalone = args.verification_root / "w560-full-improvement-history-standalone.html"
    run_checked([
        args.python,
        str(args.fragment_renderer),
        str(args.inline_html),
        str(standalone),
    ])
    staged_png = args.png.parent / f".{args.png.name}.{os.getpid()}.staged.png"
    environment = {**os.environ, "NODE_PATH": str(args.node_path)}
    run_checked([
        str(args.node),
        str(args.png_renderer),
        standalone.as_uri(),
        str(staged_png),
    ], env=environment)
    os.replace(staged_png, args.png)

    best = min(
        passers,
        key=lambda row: (float(row["ce"]), int(row["candidate_order"])),
    ) if passers else None
    archive_identity = None
    if best is not None:
        archive = args.verification_root / "chart-history" / (
            f"strict-improvers-{len(passers):03d}-best-candidate-"
            f"{int(best['candidate_order']):06d}.png"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.png, archive)
        archive_identity = {"path": str(archive), "sha256": sha256(archive)}
    state = {
        "schema": "hfw_w560_live_chart_watch_state_v1",
        "updated_unix_ns": time.time_ns(),
        "passing_inventory_sha256": inventory_sha,
        "passing_candidate_count": len(passers),
        "best_passing_candidate": best,
        "inline_html": {"path": str(args.inline_html), "sha256": sha256(args.inline_html)},
        "shareable_png": {"path": str(args.png), "sha256": sha256(args.png)},
        "archived_png": archive_identity,
    }
    atomic_json(args.verification_root / "CHART_WATCH_STATE.json", state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument("--inline-html", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--refresh-script", type=Path, required=True)
    parser.add_argument("--fragment-renderer", type=Path, required=True)
    parser.add_argument("--png-renderer", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--node-path", type=Path, required=True)
    args = parser.parse_args()

    state_path = args.verification_root / "CHART_WATCH_STATE.json"
    prior_sha = None
    if state_path.exists():
        prior_sha = json.loads(state_path.read_text()).get("passing_inventory_sha256")
    while True:
        inventory_sha, passers = passing_inventory(args.verification_root)
        if inventory_sha != prior_sha:
            refresh(args, inventory_sha, passers)
            prior_sha = inventory_sha
        if not process_exists(args.pid):
            refresh(args, inventory_sha, passers)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
