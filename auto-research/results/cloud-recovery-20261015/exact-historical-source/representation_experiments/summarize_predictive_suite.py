"""Compare equal-storage students and apply the predictive-state decision gate."""
from __future__ import annotations

import argparse
from pathlib import Path

from representation_experiments.common import atomic_json, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    rows = []
    for student in config["students"]:
        path = Path(args.root) / student / "result.json"
        if not path.exists():
            rows.append({"student": student, "complete": False})
            continue
        payload = load_json(path)
        benchmark_path = Path(args.root) / student / "benchmarks.json"
        if benchmark_path.exists():
            payload["benchmarks"] = load_json(benchmark_path)
        rows.append({"student": student, "complete": True, **payload})
    complete = {row["student"]: row for row in rows if row.get("complete")}
    decision = "incomplete"
    checks = {}
    if set(config["students"]).issubset(complete):
        predictive = complete["hierarchical_predictive"]
        hierarchy_control = complete["hierarchical_logit_control"]
        recurrent = complete["recurrent_logits"]
        transformer = complete["transformer"]
        gates = config["predictive_promotion_gates"]
        checks = {
            "beats_logit_recurrent_ce": predictive["metrics"]["ce"]
            <= recurrent["metrics"]["ce"] - gates["minimum_ce_improvement"],
            "beats_transformer_ce": predictive["metrics"]["ce"]
            <= transformer["metrics"]["ce"] - gates["minimum_ce_improvement"],
            "beats_same_architecture_logit_control": predictive["metrics"]["ce"]
            <= hierarchy_control["metrics"]["ce"] - gates["minimum_ce_improvement"],
            "future_probe_reconstruction": predictive["metrics"]["future_probe_accuracy"]
            >= gates["minimum_future_probe_accuracy"],
            "rollout_stability": predictive["metrics"].get(
                "rollout_degradation_64_to_512", 999)
            <= gates["maximum_rollout_ce_degradation"],
            "memory_advantage": predictive["metrics"]["state_bytes_bf16_at_512"]
            < transformer["metrics"]["state_bytes_bf16_at_512"],
            "equal_storage": max(row["budget_utilization"] for row in complete.values())
            - min(row["budget_utilization"] for row in complete.values())
            <= gates["maximum_budget_utilization_spread"],
        }
        if all("benchmarks" in row for row in complete.values()):
            averages = {
                name: sum(row["benchmarks"]["scores"].values())
                / len(row["benchmarks"]["scores"])
                for name, row in complete.items()
            }
            checks["benchmark_average"] = averages["hierarchical_predictive"] >= (
                max(averages["transformer"], averages["recurrent_logits"],
                    averages["hierarchical_logit_control"])
                + gates["minimum_benchmark_average_improvement"])
        decision = "promote" if all(checks.values()) else "reject_or_iterate"
    summary = {
        "study": "qwen06b_predictive_state_compilation",
        "rows": rows, "checks": checks, "decision": decision,
    }
    atomic_json(summary, args.out)
    print(summary)


if __name__ == "__main__":
    main()
