"""Summarize the equal-supervision hierarchy/function-weight experiment."""
from __future__ import annotations

import argparse

from representation_experiments.common import atomic_json, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    rows = []
    for arm in config["arms"]:
        result = load_json(f"{args.root}/{arm}/result.json")
        result["benchmarks"] = load_json(f"{args.root}/{arm}/benchmarks.json")
        rows.append(result)
    by_name = {row["arm"]: row for row in rows}
    gates = config["promotion_gates"]
    transformer = by_name[gates["transformer_reference"]]
    hierarchy = by_name[gates["hierarchy_reference"]]
    candidates = [
        by_name[gates["local_reference"]],
        by_name[gates["function_candidate"]],
    ]

    def ce(row):
        return float(row["metrics"]["student_ce"])

    def benchmark(row):
        return float(row["benchmarks"]["mean_score"])

    def eligible(row):
        long_metrics = row["metrics"].get("long_context", {})
        long_context = str(max(int(value) for value in long_metrics)) \
            if long_metrics else None
        long_degradation = (float(long_metrics[long_context]["student_ce"]) - ce(row)
                            if long_context else float("inf"))
        return {
            "beats_matched_transformer_ce": ce(row) <= ce(transformer) - float(
                gates["minimum_transformer_ce_improvement"]),
            "beats_fixed_hierarchy_ce": ce(row) <= ce(hierarchy) - float(
                gates["minimum_current_ce_improvement"]),
            "rollout_stable": float(row["metrics"]["rollout_degradation"]) <= float(
                gates["maximum_rollout_ce_degradation"]),
            "long_context_stable": long_degradation <= float(
                gates["maximum_long_context_ce_degradation"]),
            "state_within_limit": int(row["metrics"][
                "state_bytes_bf16_per_sequence_at_512"]) <= int(
                    gates["maximum_state_bytes_per_sequence"]),
            "estimated_macs_not_above_transformer": float(row["metrics"][
                "estimated_macs_per_token_at_512"]) <= float(transformer["metrics"][
                    "estimated_macs_per_token_at_512"]) * float(
                        gates["maximum_inference_macs_ratio"]),
            "benchmark_not_worse": benchmark(row) >= benchmark(transformer),
        }

    candidate_checks = {row["arm"]: eligible(row) for row in candidates}
    utilization = [float(row["budget_utilization"]) for row in rows]
    equal_storage = max(utilization) - min(utilization) <= float(
        gates["maximum_budget_utilization_spread"])
    qualified = [row for row in candidates
                 if equal_storage and all(candidate_checks[row["arm"]].values())]
    winner = min(qualified, key=ce) if qualified else None
    summary = {
        "study": config["study"],
        "rows": rows,
        "candidate_checks": candidate_checks,
        "equal_storage": equal_storage,
        "winner": winner["arm"] if winner else None,
        "decision": (f"promote_{winner['arm']}" if winner
                     else "reject_or_iterate"),
    }
    atomic_json(summary, args.out)
    print(summary)


if __name__ == "__main__":
    main()
