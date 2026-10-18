#!/usr/bin/env python3
"""Iterate canonical candidates and exhaust exact winner directions after every promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


STATE_SCHEMA = "hfw_w560_iterative_candidate_direction_loop_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--wave-root", required=True, type=Path)
    parser.add_argument("--source-snapshot-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--candidate-verifier", required=True, type=Path)
    parser.add_argument("--direction-verifier", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--alpha-scales", default="1.125,1.25,1.5,2")
    parser.add_argument("--block-scales", default="0.125,0.25,0.5")
    parser.add_argument("--scalar-scales", default="0.125,0.25,0.5")
    parser.add_argument("--scalar-top-k", type=int, default=8)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--chart-refresh", type=Path)
    parser.add_argument("--fragment-renderer", type=Path)
    parser.add_argument("--png-renderer", type=Path)
    parser.add_argument("--inline-html", type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--node-path", type=Path)
    args = parser.parse_args()
    chart_values = (
        args.chart_refresh,
        args.fragment_renderer,
        args.png_renderer,
        args.inline_html,
        args.png,
        args.node,
        args.node_path,
    )
    if any(value is not None for value in chart_values) and not all(
        value is not None for value in chart_values
    ):
        raise RuntimeError("chart refresh is either fully configured or fully omitted")
    if args.scalar_top_k < 0:
        raise RuntimeError("scalar-top-k must be nonnegative")
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise RuntimeError("max-candidates must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def current_head(package_root: Path) -> dict[str, Any]:
    lineage = package_root / "runtime/authority/lineage"
    pointer = read_json(lineage / "CURRENT.json")
    record = Path(str(pointer["record_path"]))
    if sha256(record) != pointer["record_sha256"]:
        raise RuntimeError("CURRENT head content hash drift")
    head = read_json(record)
    if int(head["authority_generation"]) != int(pointer["authority_generation"]):
        raise RuntimeError("CURRENT generation drift")
    return {**head, "record_path": str(record), "record_sha256": pointer["record_sha256"]}


def source_candidates(source_root: Path) -> list[dict[str, Any]]:
    snapshot = read_json(source_root / "SNAPSHOT.json")
    candidates = list(snapshot["candidates"])
    orders = [int(row["candidate_order"]) for row in candidates]
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise RuntimeError("source candidate orders are not unique and sorted")
    return candidates


def initial_next_order(source_root: Path, candidates: list[dict[str, Any]]) -> int | None:
    completed = {
        int(path.name.split("-", 1)[0])
        for path in (source_root / "candidate-results").glob("*.json")
    }
    encountered_gap = False
    for candidate in candidates:
        order = int(candidate["candidate_order"])
        if order not in completed:
            encountered_gap = True
            return order
        if encountered_gap:
            raise RuntimeError("completed source candidates are not a contiguous prefix")
    return None


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(
            f"child failed with return code {completed.returncode}; see {log_path}"
        )


def selected_ce_and_duplicate(package_root: Path, head: dict[str, Any]) -> tuple[float, float]:
    decision_path = (
        package_root
        / "runtime/authority/lineage/decisions"
        / f"{head['decision_sha256']}.decision.json"
    )
    if sha256(decision_path) != head["decision_sha256"]:
        raise RuntimeError("current decision content hash drift")
    selected = read_json(decision_path).get("selected_candidate")
    if not isinstance(selected, dict):
        raise RuntimeError("current accepted decision has no selected candidate")
    ce = selected.get("mps_verify_replicated_mean")
    duplicate = selected.get("duplicate_max_abs_sequence_difference")
    if not isinstance(ce, (int, float)) or not isinstance(duplicate, (int, float)):
        raise RuntimeError("current selected candidate lacks canonical duplicate evidence")
    return float(ce), float(duplicate)


def prepare_direction_input(
    package_root: Path,
    stage_root: Path,
    head: dict[str, Any],
) -> tuple[Path, int]:
    lineage = package_root / "runtime/authority/lineage"
    group_sha = str(head["replacement_group_flat_sha256"])
    group_path = lineage / "replacement_group_flats" / f"{group_sha}.group_flat.f32le"
    if sha256(group_path) != group_sha:
        raise RuntimeError("current replacement-group flat hash drift")
    ce, duplicate = selected_ce_and_duplicate(package_root, head)
    if duplicate != 0.0:
        raise RuntimeError("recursive direction requires exact duplicate equality")
    input_root = stage_root / "input"
    candidate_order = int(head["model_generation"])
    atomic_json(input_root / "PARENT.json", {
        "schema": "hfw_w560_recursive_direction_parent_adapter_v1",
        "base_head_record_sha256": head["parent_head_record_sha256"],
        "current_head_record_sha256": head["record_sha256"],
    })
    atomic_json(
        input_root
        / "candidate-results"
        / f"{candidate_order:06d}-{group_sha}.json",
        {
            "schema": "hfw_w560_recursive_direction_winner_adapter_v1",
            "passes_minimum_replicated_improvement": True,
            "duplicate_max_abs_sequence_difference": duplicate,
            "group": head["group"],
            "bundle_path": str(group_path),
            "offset_bytes": 0,
            "size_bytes": group_path.stat().st_size,
            "candidate_group_flat_sha256": group_sha,
            "mps_verify_replicated_mean": ce,
            "source_head_record_sha256": head["parent_head_record_sha256"],
            "current_head_record_sha256": head["record_sha256"],
        },
    )
    return input_root, candidate_order


def chart_configured(args: argparse.Namespace) -> bool:
    return args.chart_refresh is not None


def refresh_chart(args: argparse.Namespace, verification_root: Path, method: str) -> dict[str, str]:
    assert args.chart_refresh is not None
    assert args.fragment_renderer is not None
    assert args.png_renderer is not None
    assert args.inline_html is not None
    assert args.png is not None
    assert args.node is not None
    assert args.node_path is not None
    run_logged([
        args.python,
        str(args.chart_refresh),
        "--package-root", str(args.package_root),
        "--wave-root", str(args.wave_root),
        "--verification-root", str(verification_root),
        "--inline-html", str(args.inline_html),
    ], args.run_root / "logs/chart-refresh.log")
    standalone = args.run_root / "chart-standalone.html"
    run_logged([
        args.python,
        str(args.fragment_renderer),
        str(args.inline_html),
        str(standalone),
    ], args.run_root / "logs/chart-render-fragment.log")
    staged_png = args.png.parent / f".{args.png.name}.{os.getpid()}.staged.png"
    environment = {**os.environ, "NODE_PATH": str(args.node_path)}
    with (args.run_root / "logs/chart-render-png.log").open("ab", buffering=0) as log:
        completed = subprocess.run(
            [str(args.node), str(args.png_renderer), standalone.as_uri(), str(staged_png)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if completed.returncode != 0:
        raise RuntimeError("chart PNG render failed")
    os.replace(staged_png, args.png)
    head = current_head(args.package_root)
    archive = args.run_root / "chart-history" / (
        f"generation-{int(head['model_generation']):08d}-{method}.png"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.png, archive)
    return {
        "inline_html_sha256": sha256(args.inline_html),
        "png_sha256": sha256(args.png),
        "archive_path": str(archive),
        "archive_sha256": sha256(archive),
    }


def load_or_initialize_state(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    state_path = args.run_root / "STATE.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("schema") != STATE_SCHEMA:
            raise RuntimeError("iterative state schema drift")
        return state
    next_order = initial_next_order(args.source_snapshot_root, candidates)
    head = current_head(args.package_root)
    return {
        "schema": STATE_SCHEMA,
        "status": "READY",
        "phase": "CANDIDATE",
        "next_candidate_order": next_order,
        "active_candidate_order": None,
        "current_head_record_sha256": head["record_sha256"],
        "current_model_generation": int(head["model_generation"]),
        "completed_candidate_orders": [],
        "promotion_events": [],
        "chart_pending": None,
        "updated_unix_ns": time.time_ns(),
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_unix_ns"] = time.time_ns()
    atomic_json(path, state)


def main() -> None:
    args = parse_args()
    args.package_root = args.package_root.resolve()
    args.wave_root = args.wave_root.resolve()
    args.source_snapshot_root = args.source_snapshot_root.resolve()
    args.run_root = args.run_root.resolve()
    args.candidate_verifier = args.candidate_verifier.resolve()
    args.direction_verifier = args.direction_verifier.resolve()
    candidates = source_candidates(args.source_snapshot_root)
    candidate_by_order = {int(row["candidate_order"]): row for row in candidates}
    next_order = initial_next_order(args.source_snapshot_root, candidates)
    head = current_head(args.package_root)
    plan = {
        "current_head_record_sha256": head["record_sha256"],
        "current_model_generation": int(head["model_generation"]),
        "source_candidate_count": len(candidates),
        "next_candidate_order": next_order,
        "remaining_candidate_count": sum(
            int(row["candidate_order"]) >= int(next_order)
            for row in candidates
        ) if next_order is not None else 0,
        "scalar_top_k": args.scalar_top_k,
        "alpha_scales": args.alpha_scales,
        "block_scales": args.block_scales,
        "scalar_scales": args.scalar_scales,
    }
    print("PLAN " + json.dumps(plan, sort_keys=True), flush=True)
    if args.plan_only:
        return
    args.run_root.mkdir(parents=True, exist_ok=True)
    state_path = args.run_root / "STATE.json"
    state = load_or_initialize_state(args, candidates)
    save_state(state_path, state)
    completed_this_run = 0

    while True:
        if state.get("chart_pending") is not None:
            pending = dict(state["chart_pending"])
            state["last_chart"] = refresh_chart(
                args, Path(str(pending["verification_root"])), str(pending["method"]),
            )
            state["chart_pending"] = None
            save_state(state_path, state)

        phase = str(state["phase"])
        if phase == "DIRECTION":
            before = current_head(args.package_root)
            stage_root = args.run_root / "direction" / (
                f"from-generation-{int(before['model_generation']):08d}-"
                f"{before['record_sha256'][:12]}"
            )
            output_root = stage_root / "output"
            terminal_path = output_root / "PROMOTION.json"
            if not terminal_path.exists():
                input_root, winner_order = prepare_direction_input(
                    args.package_root, stage_root, before,
                )
                run_logged([
                    args.python,
                    "-B",
                    str(args.direction_verifier),
                    "--package-root", str(args.package_root),
                    "--wave-root", str(args.wave_root),
                    "--verification-root", str(input_root),
                    "--output-root", str(output_root),
                    "--winner-candidate-order", str(winner_order),
                    "--alpha-scales", args.alpha_scales,
                    "--block-scales", args.block_scales,
                    "--scalar-scales", args.scalar_scales,
                    "--scalar-top-k", str(args.scalar_top_k),
                ], stage_root / "direction.log")
            terminal = read_json(terminal_path)
            after = current_head(args.package_root)
            event = {
                "kind": "DIRECTION",
                "accepted": bool(terminal["accepted"]),
                "parent_head_record_sha256": terminal["parent_head_record_sha256"],
                "current_head_record_sha256": terminal["current_head_record_sha256"],
                "parent_ce": terminal["parent_ce"],
                "child_ce": terminal["child_ce"],
                "output_root": str(output_root),
            }
            if event not in state["promotion_events"]:
                state["promotion_events"].append(event)
            if terminal["accepted"]:
                if after["record_sha256"] == before["record_sha256"]:
                    raise RuntimeError("accepted direction did not advance current head")
                state["current_head_record_sha256"] = after["record_sha256"]
                state["current_model_generation"] = int(after["model_generation"])
                state["chart_pending"] = {
                    "verification_root": str(output_root),
                    "method": "direction-descent",
                }
                save_state(state_path, state)
                print(
                    f"PROMOTED direction G{after['model_generation']} "
                    f"ce={terminal['child_ce']:.15f}; continuing direction descent",
                    flush=True,
                )
                continue
            order = int(state["active_candidate_order"])
            state["completed_candidate_orders"].append(order)
            state["next_candidate_order"] = order + 1
            state["active_candidate_order"] = None
            state["phase"] = "CANDIDATE"
            save_state(state_path, state)
            print(f"DIRECTION exhausted after candidate {order}", flush=True)
            continue

        order_value = state.get("next_candidate_order")
        if order_value is None or int(order_value) not in candidate_by_order:
            state["status"] = "COMPLETED_ALL_FIXED_CANDIDATES_AND_DIRECTIONS"
            state["phase"] = "COMPLETE"
            save_state(state_path, state)
            print("COMPLETE all fixed candidates and recursive directions exhausted", flush=True)
            return
        if args.max_candidates is not None and completed_this_run >= args.max_candidates:
            state["status"] = "STOPPED_AT_REQUESTED_CANDIDATE_BOUNDARY"
            save_state(state_path, state)
            print("STOP requested candidate boundary reached", flush=True)
            return

        order = int(order_value)
        before = current_head(args.package_root)
        stage_root = args.run_root / "regular" / (
            f"candidate-{order:06d}-from-{before['record_sha256'][:12]}"
        )
        terminal_path = stage_root / "PROMOTION.json"
        state["phase"] = "CANDIDATE"
        state["active_candidate_order"] = order
        save_state(state_path, state)
        if not terminal_path.exists():
            run_logged([
                args.python,
                "-B",
                str(args.candidate_verifier),
                "--package-root", str(args.package_root),
                "--wave-root", str(args.wave_root),
                "--output-root", str(stage_root),
                "--promote-after-exhaustive",
                "--promote-first-strict-improver",
                "--minimum-relative-improvement-override", "0",
                "--successor-snapshot-root", str(args.source_snapshot_root),
                "--start-candidate-order", str(order),
            ], stage_root / "verifier.log")
        terminal = read_json(terminal_path)
        summary = read_json(stage_root / "SUMMARY.json")
        result_paths = sorted((stage_root / "candidate-results").glob("*.json"))
        if not result_paths:
            raise RuntimeError("iterative verifier produced no candidate results")
        results = [read_json(path) for path in result_paths]
        evaluated_orders = [int(result["candidate_order"]) for result in results]
        expected_orders = list(range(order, order + len(results)))
        if evaluated_orders != expected_orders:
            raise RuntimeError("iterative scan results are not a contiguous candidate prefix")
        if int(summary["candidate_count_evaluated_twice"]) != len(results):
            raise RuntimeError("iterative summary/result count drift")
        passers = [
            result
            for result in results
            if result["passes_minimum_replicated_improvement"] is True
        ]
        if len(passers) > 1:
            raise RuntimeError("first-improver scan retained more than one passer")
        accepted = bool(terminal["accepted"])
        if accepted is not bool(passers):
            raise RuntimeError("candidate pass/promotion decision drift")
        result = passers[0] if passers else results[-1]
        after = current_head(args.package_root)
        event = {
            "kind": "REGULAR_CANDIDATE",
            "candidate_order": int(result["candidate_order"]),
            "scan_start_candidate_order": order,
            "evaluated_candidate_orders": evaluated_orders,
            "evaluated_candidate_count": len(results),
            "task_index": int(result["task_index"]),
            "group": result["group"],
            "accepted": accepted,
            "ce": result["mps_verify_replicated_mean"],
            "duplicate": result["duplicate_max_abs_sequence_difference"],
            "parent_head_record_sha256": before["record_sha256"],
            "current_head_record_sha256": after["record_sha256"],
            "stage_root": str(stage_root),
        }
        state["promotion_events"].append(event)
        completed_this_run += len(results)
        if accepted:
            if after["record_sha256"] == before["record_sha256"]:
                raise RuntimeError("accepted regular candidate did not advance current head")
            state["phase"] = "DIRECTION"
            state["active_candidate_order"] = int(result["candidate_order"])
            state["completed_candidate_orders"].extend(evaluated_orders[:-1])
            state["next_candidate_order"] = evaluated_orders[-1] + 1
            state["current_head_record_sha256"] = after["record_sha256"]
            state["current_model_generation"] = int(after["model_generation"])
            state["chart_pending"] = {
                "verification_root": str(stage_root),
                "method": "candidate-search",
            }
            save_state(state_path, state)
            print(
                f"PROMOTED candidate {result['candidate_order']} G{after['model_generation']} "
                f"ce={result['mps_verify_replicated_mean']:.15f}; starting direction descent",
                flush=True,
            )
            continue
        state["completed_candidate_orders"].extend(evaluated_orders)
        state["next_candidate_order"] = evaluated_orders[-1] + 1
        state["active_candidate_order"] = None
        state["phase"] = "CANDIDATE"
        save_state(state_path, state)
        print(
            f"NO PASS in candidates {evaluated_orders[0]}-{evaluated_orders[-1]}; "
            "fixed candidate set exhausted",
            flush=True,
        )


if __name__ == "__main__":
    main()
