#!/usr/bin/env python3
"""Wait for the authoritative verifier, then refresh chart and audit outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_exit(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(30)


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_terminal(
    package_root: Path,
    wave_root: Path,
    verification_root: Path,
) -> dict[str, object]:
    summary_path = verification_root / "SUMMARY.json"
    promotion_path = verification_root / "PROMOTION.json"
    snapshot_path = verification_root / "SNAPSHOT.json"
    parent_path = verification_root / "PARENT.json"
    require(summary_path.exists(), "authoritative verifier exited without SUMMARY.json")
    require(promotion_path.exists(), "authoritative verifier exited without PROMOTION.json")
    require(snapshot_path.exists(), "authoritative verifier exited without SNAPSHOT.json")
    require(parent_path.exists(), "authoritative verifier exited without PARENT.json")

    summary = read_json(summary_path)
    promotion = read_json(promotion_path)
    snapshot = read_json(snapshot_path)
    parent = read_json(parent_path)
    expected = int(snapshot["unique_candidate_count"])
    require(expected > 0, "fixed candidate snapshot is empty")
    require(
        summary.get("schema") == "hfw_w560_operator_stopped_authoritative_mps_verify_summary_v1",
        "terminal summary schema is not authoritative",
    )
    require(
        summary.get("status") == "OPERATOR_STOPPED_COMPLETED_WORKER_SNAPSHOT_EXHAUSTED",
        "authoritative verifier did not exhaust the fixed snapshot",
    )
    require(
        int(summary["candidate_count_evaluated_twice"]) == expected
        and summary.get("all_snapshot_candidates_evaluated_twice") is True
        and summary.get("authoritative_promotion_permitted") is True
        and summary.get("final_exhaustive_promotion_still_required") is False,
        "authoritative verifier terminal summary is not exhaustive",
    )
    require(summary.get("promotion") == promotion, "summary/promotion payload drift")
    require(
        promotion.get("campaign_sha256") == snapshot.get("campaign_sha256")
        == parent.get("campaign_sha256"),
        "campaign binding drift across terminal artifacts",
    )
    require(
        parent.get("snapshot_sha256") == snapshot.get("snapshot_sha256"),
        "parent/fixed-snapshot binding drift",
    )
    require(
        parent.get("authoritative_promotion_permitted") is True,
        "parent verification was not authoritative",
    )

    result_paths = sorted((verification_root / "candidate-results").glob("*.json"))
    require(len(result_paths) == expected, "candidate result file count is not exhaustive")
    orders: list[int] = []
    passers = 0
    maximum_duplicate = 0.0
    tolerance = float(parent["duplicate_forward_tolerance"])
    for path in result_paths:
        row = read_json(path)
        order = int(row["candidate_order"])
        orders.append(order)
        require(
            row.get("schema")
            == "hfw_w560_operator_stopped_authoritative_candidate_mps_verify_v1",
            f"candidate {order} result schema is not authoritative",
        )
        require(
            row.get("snapshot_sha256") == snapshot.get("snapshot_sha256")
            and row.get("authoritative_promotion_permitted") is True
            and int(row.get("checkpoint_or_head_writes", -1)) == 0,
            f"candidate {order} authority binding drift",
        )
        first = row.get("mps_verify_per_sequence_first")
        second = row.get("mps_verify_per_sequence_second")
        require(
            isinstance(first, list)
            and isinstance(second, list)
            and len(first) > 0
            and len(first) == len(second),
            f"candidate {order} duplicated sequence inventory drift",
        )
        require(
            all(isinstance(value, (int, float)) and math.isfinite(value) for value in first + second),
            f"candidate {order} contains a non-finite loss",
        )
        duplicate = max(abs(float(a) - float(b)) for a, b in zip(first, second))
        require(
            duplicate == float(row["duplicate_max_abs_sequence_difference"])
            and duplicate <= tolerance,
            f"candidate {order} duplicate verification drift",
        )
        maximum_duplicate = max(maximum_duplicate, duplicate)
        passers += int(row.get("passes_minimum_replicated_improvement") is True)
    require(orders == list(range(1, expected + 1)), "candidate result order is not contiguous")
    require(
        passers == int(summary["passing_candidate_count"]),
        "passing-candidate count drift",
    )

    cleanup_receipt = None
    receipt_paths = sorted((wave_root / "orchestration/receipts").glob("*.receipt.json"))
    for path in reversed(receipt_paths):
        candidate = read_json(path)
        if candidate.get("event") == "CLEANUP_TERMINAL":
            cleanup_receipt = candidate
            cleanup_receipt_path = path
            break
    require(cleanup_receipt is not None, "cleanup terminal receipt is missing")
    require(
        cleanup_receipt.get("campaign_sha256") == snapshot.get("campaign_sha256")
        and int(cleanup_receipt.get("campaign_sandbox_count", -1)) == 0
        and int(cleanup_receipt.get("final_remaining_count", -1)) == 0
        and cleanup_receipt.get("snapshot_present") is False
        and cleanup_receipt.get("final_relist_succeeded") is True
        and cleanup_receipt.get("snapshot_relist_succeeded") is True,
        "cleanup terminal does not prove local zero-resource reconciliation",
    )

    lineage = package_root / "runtime/authority/lineage"
    current_path = lineage / "CURRENT.json"
    current = read_json(current_path)
    require(
        current.get("record_sha256") == promotion.get("current_head_record_sha256"),
        "CURRENT pointer does not match promotion terminal",
    )
    head_path = Path(str(current["record_path"]))
    require(head_path.exists(), "CURRENT head record is missing")
    require(sha256(head_path) == current["record_sha256"], "CURRENT head hash drift")

    decision_identity = promotion.get("decision")
    require(isinstance(decision_identity, dict), "promotion decision identity is missing")
    decision_path = Path(str(decision_identity["path"]))
    require(
        decision_path.exists() and sha256(decision_path) == decision_identity["sha256"],
        "promotion decision artifact hash drift",
    )
    decision = read_json(decision_path)
    accepted = promotion.get("accepted") is True
    require(decision.get("accepted") is accepted, "decision/promotion acceptance drift")
    require(
        bool(summary.get("checkpoint_or_head_writes")) is accepted,
        "summary checkpoint/head write count drift",
    )
    if accepted:
        child = promotion.get("child_checkpoint")
        require(isinstance(child, dict), "accepted promotion has no child checkpoint")
        child_path = Path(str(child["path"]))
        require(
            child_path.exists() and sha256(child_path) == child["sha256"],
            "accepted child checkpoint hash drift",
        )
        parent_ce = float(promotion["parent_ce"])
        child_ce = float(promotion["child_ce"])
        require(
            child_ce < float(parent["thresholds"][0])
            and math.isclose(float(promotion["absolute_improvement"]), parent_ce - child_ce)
            and math.isclose(
                float(promotion["relative_improvement"]),
                (parent_ce - child_ce) / abs(parent_ce),
            ),
            "accepted promotion numerical derivation drift",
        )
        require(
            current.get("record_sha256") != snapshot.get("base_head_record_sha256"),
            "accepted promotion did not advance the head",
        )
    else:
        require(promotion.get("child_checkpoint") is None, "rejected promotion has a child checkpoint")
        require(
            current.get("record_sha256") == snapshot.get("base_head_record_sha256"),
            "rejected promotion unexpectedly advanced the head",
        )

    return {
        "summary": summary,
        "promotion": promotion,
        "snapshot": snapshot,
        "expected": expected,
        "passers": passers,
        "maximum_duplicate": maximum_duplicate,
        "cleanup_receipt_path": str(cleanup_receipt_path),
        "cleanup_receipt_sha256": sha256(cleanup_receipt_path),
        "current_head_record_sha256": current["record_sha256"],
        "decision_path": str(decision_path),
        "decision_sha256": decision_identity["sha256"],
        "accepted": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument("--inline-html", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    wait_for_exit(args.pid)
    terminal = validate_terminal(
        args.package_root,
        args.wave_root,
        args.verification_root,
    )
    summary_path = args.verification_root / "SUMMARY.json"
    promotion_path = args.verification_root / "PROMOTION.json"
    expected = int(terminal["expected"])

    run([
        "python3",
        str(Path(__file__).with_name("w560_chart_refresh.py")),
        "--package-root", str(args.package_root),
        "--wave-root", str(args.wave_root),
        "--verification-root", str(args.verification_root),
        "--inline-html", str(args.inline_html),
    ])
    standalone = args.verification_root / "w560-full-improvement-history-standalone.html"
    run([
        "python3",
        "/Users/rohanarun/.codex/plugins/cache/openai-bundled/visualize/1.0.22/skills/visualize/scripts/render.py",
        str(args.inline_html), str(standalone),
    ])
    node = "/Users/rohanarun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    node_path = "/Users/rohanarun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
    run([
        node,
        str(Path(__file__).with_name("w560_chart_render.js")),
        standalone.as_uri(), str(args.png),
    ], env={**os.environ, "NODE_PATH": node_path})

    test_run = run([
        str(args.package_root / "runtime/daytona-manager-venv/bin/python"),
        "-B", "-m", "unittest", "discover", "-s", "runtime/tests", "-v",
    ], cwd=args.package_root, capture_output=True)
    audit = {
        "schema": "hfw_w560_operator_stopped_postrun_audit_v1",
        "completed_unix_ns": time.time_ns(),
        "summary_sha256": sha256(summary_path),
        "promotion_sha256": sha256(promotion_path),
        "candidate_count_evaluated_twice": expected,
        "passing_candidate_count": terminal["passers"],
        "maximum_duplicate_abs_sequence_difference": terminal["maximum_duplicate"],
        "promotion_accepted": terminal["accepted"],
        "current_head_record_sha256": terminal["current_head_record_sha256"],
        "decision_path": terminal["decision_path"],
        "decision_sha256": terminal["decision_sha256"],
        "cleanup_receipt_path": terminal["cleanup_receipt_path"],
        "cleanup_receipt_sha256": terminal["cleanup_receipt_sha256"],
        "cleanup_zero_resource_local_receipt_validated": True,
        "chart_inline_sha256": sha256(args.inline_html),
        "chart_png_sha256": sha256(args.png),
        "test_returncode": test_run.returncode,
        "test_stdout_tail": test_run.stdout[-4000:],
        "test_stderr_tail": test_run.stderr[-4000:],
    }
    destination = args.verification_root / "POSTRUN_AUDIT.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
