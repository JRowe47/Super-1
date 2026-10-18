#!/usr/bin/env python3
"""Refresh the W560 inline chart from append-only optimizer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


CANONICAL_RE = re.compile(
    r"\{generation:(?P<generation>\d+),ce:(?P<ce>[0-9.]+),"
    r"time:'(?P<time>[^']+)',group:'(?P<group>[^']+)'\}"
)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def iso_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, ZoneInfo("America/New_York")).isoformat()


def replace_once(text: str, pattern: str, replacement: str, *, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"expected one chart replacement for {pattern!r}, observed {count}")
    return updated


def canonical_points(html: str, package_root: Path) -> list[dict[str, object]]:
    array_match = re.search(r"const canonical = \[(.*?)\n\s*\];", html, re.S)
    if not array_match:
        raise RuntimeError("canonical chart data is missing")
    points = [
        {
            "generation": int(match.group("generation")),
            "ce": float(match.group("ce")),
            "time": match.group("time"),
            "group": match.group("group"),
        }
        for match in CANONICAL_RE.finditer(array_match.group(1))
    ]
    if not points:
        raise RuntimeError("canonical chart data is empty")
    by_generation = {int(point["generation"]): point for point in points}
    lineage = package_root / "runtime/authority/lineage"
    for head_path in sorted((lineage / "heads").glob("*.head.json")):
        head = read_json(head_path)
        generation = int(head["model_generation"])
        decision_sha = head.get("decision_sha256")
        if generation in by_generation or not isinstance(decision_sha, str):
            continue
        decision_path = lineage / "decisions" / f"{decision_sha}.decision.json"
        if not decision_path.exists():
            raise RuntimeError(f"promoted head decision is missing: {decision_path}")
        decision = read_json(decision_path)
        selected = decision.get("selected_candidate")
        if not isinstance(selected, dict):
            selected = decision.get("winner")
        if not isinstance(selected, dict):
            raise RuntimeError("accepted decision has no selected candidate")
        ce = selected.get("mps_verify_replicated_mean")
        if not isinstance(ce, (int, float)):
            ce = selected.get("canonical_mps_bf16_replicated_mean_ce")
        if not isinstance(ce, (int, float)):
            raise RuntimeError("accepted decision has no canonical replicated mean")
        checkpoint = head.get("checkpoint")
        time_ns = checkpoint.get("mtime_ns") if isinstance(checkpoint, dict) else None
        if not isinstance(time_ns, int):
            time_ns = decision_path.stat().st_mtime_ns
        point = {
            "generation": generation,
            "ce": float(ce),
            "time": iso_from_ns(time_ns),
            "group": str(selected["group"]),
        }
        by_generation[generation] = point
    return [by_generation[generation] for generation in sorted(by_generation)]


def worker_rows(wave_root: Path) -> list[dict[str, object]]:
    receipts = [
        read_json(path)
        for path in sorted((wave_root / "orchestration/receipts").glob("*.receipt.json"))
    ]
    downloads = [
        receipt
        for receipt in receipts
        if receipt.get("event") == "WORKER_OUTPUT_DOWNLOADED_AND_VALIDATED"
    ]
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for receipt in downloads:
        task = int(receipt["task_index"])
        if task in seen:
            raise RuntimeError(f"duplicate validated download receipt for task {task}")
        seen.add(task)
        result = read_json(wave_root / "downloads" / f"worker-{task:03d}" / "CPU_SEARCH_RESULT.json")
        rows.append(
            {
                "task": task,
                "group": result["group"],
                "base": round(float(result["base_full_search_mean_ce"]), 12),
                "terminal": round(float(result["terminal_full_search_mean_ce"]), 12),
                "improvement": round(float(result["self_calibrated_full_search_improvement"]), 12),
                "accepted": int(result["accepted_candidate_count"]),
                "time": iso_from_ns(int(receipt["recorded_unix_ns"])),
            }
        )
    downloaded = sorted(path.parent.name for path in (wave_root / "downloads").glob("worker-*/CPU_SEARCH_RESULT.json"))
    if len(rows) != len(downloaded):
        raise RuntimeError("validated receipt count differs from downloaded result count")
    return rows


def provisional_best(verification_root: Path) -> dict[str, object] | None:
    passing: list[tuple[Path, dict[str, object]]] = []
    for path in sorted((verification_root / "candidate-results").glob("*.json")):
        row = read_json(path)
        if row.get("passes_minimum_replicated_improvement") is True:
            passing.append((path, row))
    if not passing:
        return None
    path, row = min(
        passing,
        key=lambda item: (
            float(item[1]["mps_verify_replicated_mean"]),
            int(item[1]["candidate_order"]),
        ),
    )
    parent = read_json(verification_root / "PARENT.json")
    parent_ce = float(parent["mean_first"])
    ce = float(row["mps_verify_replicated_mean"])
    return {
        "ce": ce,
        "time": iso_from_ns(path.stat().st_mtime_ns),
        "group": str(row["group"]),
        "candidateOrder": int(row["candidate_order"]),
        "taskIndex": int(row["task_index"]),
        "duplicate": float(row["duplicate_max_abs_sequence_difference"]),
        "improvement": parent_ce - ce,
        "passingCount": len(passing),
        "resultSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument("--inline-html", type=Path, required=True)
    args = parser.parse_args()

    html = args.inline_html.read_text()
    canonical = canonical_points(html, args.package_root)
    workers = worker_rows(args.wave_root)
    live_best = provisional_best(args.verification_root)
    if live_best is not None and float(live_best["ce"]) >= float(canonical[-1]["ce"]):
        live_best = None
    accepted = sum(int(row["accepted"]) for row in workers)
    latest_time = datetime.fromisoformat(str(workers[-1]["time"]))
    latest_label = latest_time.strftime("%H:%M ET")
    summary_path = args.verification_root / "SUMMARY.json"
    complete = False
    if summary_path.exists():
        summary = read_json(summary_path)
        complete = bool(summary.get("all_snapshot_candidates_evaluated_twice"))
    last_generation = int(canonical[-1]["generation"])
    next_generation = last_generation + 1
    if live_best is not None:
        live_best["generation"] = next_generation
        panel_status = (
            f"G{next_generation} verified winner pending promotion · "
            f"{live_best['passingCount']} strict improver · best candidate "
            f"{live_best['candidateOrder']}"
        )
    elif complete:
        panel_status = f"G{last_generation} is the latest promoted generation"
    else:
        panel_status = f"G{next_generation} remains pending"

    canonical_js = ",\n        ".join(
        "{generation:%d,ce:%.15f,time:'%s',group:'%s'}"
        % (point["generation"], point["ce"], point["time"], point["group"])
        for point in canonical
    )
    workers_js = json.dumps(workers, separators=(",", ":"))
    provisional_js = (
        "null"
        if live_best is None
        else json.dumps(live_best, separators=(",", ":"), sort_keys=True)
    )
    min_terminal = min(float(row["terminal"]) for row in workers)
    max_terminal = max(float(row["terminal"]) for row in workers)

    html = replace_once(
        html,
        r'<p class="figure-subtitle">.*?</p>',
        f'<p class="figure-subtitle">Canonical checkpoint progress, Daytona CPU search outcomes, and every accepted candidate downloaded through {latest_label}</p>',
    )
    html = replace_once(
        html,
        r'<p class="panel-subtitle">Duplicated LOCAL_VERIFY · lower is better · .*?</p>',
        f'<p class="panel-subtitle">Duplicated LOCAL_VERIFY · lower is better · {panel_status}</p>',
    )
    html = replace_once(
        html,
        r'aria-label="Canonical MPS BF16 cross entropy fell from .*?\.">',
        (
            'aria-label="Canonical MPS BF16 cross entropy fell from '
            f'{float(canonical[0]["ce"]):.6f} at generation {canonical[0]["generation"]} to '
            f'{float(canonical[-1]["ce"]):.6f} at generation {last_generation}. '
            + (
                f'Candidate {live_best["candidateOrder"]} is a duplicated verified winner at '
                f'{float(live_best["ce"]):.6f} awaiting generation {next_generation} promotion.'
                if live_best is not None
                else (f'Generation {next_generation} is pending.' if not complete else f'Generation {last_generation} is current.')
            )
            + '">' 
        ),
    )
    html = replace_once(
        html,
        r'aria-label="\d+ completed Daytona workers lowered their CPU FP64 shadow cross entropy from a common base of .*?\.">',
        (
            f'aria-label="{len(workers)} completed Daytona workers lowered their CPU FP64 shadow cross entropy '
            f'from a common base of {float(workers[0]["base"]):.6f} to provisional terminal values between '
            f'{min_terminal:.6f} and {max_terminal:.6f}.">'
        ),
    )
    html = replace_once(
        html,
        r'<p class="panel-subtitle">\d+ of 100 bundles validated · \d+ accepted candidates preserved · \d+ bundles (?:pending|stopped before download)</p>',
        f'<p class="panel-subtitle">{len(workers)} of 100 bundles validated · {accepted} accepted candidates preserved · {100-len(workers)} bundles stopped before download</p>',
    )
    html = replace_once(
        html,
        r'aria-label="Cumulative accepted candidates reached \d+ after \d+ of 100 worker bundles were downloaded and validated\.">',
        f'aria-label="Cumulative accepted candidates reached {accepted} after {len(workers)} of 100 worker bundles were downloaded and validated.">',
    )
    html = replace_once(
        html,
        r"const canonical = \[.*?\n\s*\];",
        "const canonical = [\n        " + canonical_js + "\n      ];",
        flags=re.S,
    )
    html = replace_once(
        html,
        r"const provisionalBest = .*?;",
        f"const provisionalBest = {provisional_js};",
    )
    html = replace_once(html, r"const workers = \[.*?\];", f"const workers = {workers_js};", flags=re.S)
    html = replace_once(
        html,
        r"const verificationPending = (?:true|false); const pendingGeneration = \d+;",
        f"const verificationPending = {str(not complete).lower()}; const pendingGeneration = {next_generation};",
    )
    args.inline_html.write_text(html)
    print(json.dumps({"canonical_generation": last_generation, "canonical_ce": canonical[-1]["ce"], "workers": len(workers), "accepted_candidates": accepted, "verification_pending": not complete, "provisional_best": live_best}, sort_keys=True))


if __name__ == "__main__":
    main()
