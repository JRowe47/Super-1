"""Summarize the equal-budget distribution-compilation study."""
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
    arms = list(config["arms"])
    rows = []
    for arm in arms:
        result = load_json(f"{args.root}/{arm}/result.json")
        benchmark = load_json(f"{args.root}/{arm}/benchmarks.json")
        row = dict(result)
        row["benchmarks"] = benchmark
        rows.append(row)
    by_name = {row["arm"]: row for row in rows}
    recurrent = by_name["recurrent_full_kl"]
    hierarchical = by_name["hierarchical_full_kl"]
    projected = by_name["hierarchical_embed_full_kl"]
    branched = by_name["hierarchical_branch_distill"]
    threshold = config["promotion_gates"]

    def ce(row):
        return float(row["metrics"]["student_ce"])

    def kl(row):
        return float(row["metrics"]["kl_t1"])

    def benchmark(row):
        return float(row["benchmarks"]["mean_score"])

    minimum = float(threshold["minimum_ce_improvement"])
    checks = {
        "hierarchy_beats_recurrent_ce": ce(hierarchical) <= ce(recurrent) - minimum,
        "embedding_improves_hierarchy_ce": ce(projected) <= ce(hierarchical) - minimum,
        "branching_improves_projected_ce": ce(branched) <= ce(projected) - minimum,
        "branching_best_teacher_kl": kl(branched) <= min(kl(row) for row in rows),
        "branching_best_benchmark_mean": benchmark(branched) >= max(
            benchmark(row) for row in rows),
        "branching_rollout_stable": float(
            branched["metrics"]["rollout_degradation"]) <= float(
                threshold["maximum_rollout_ce_degradation"]),
        "branching_transition_kl_improves": float(
            branched["metrics"]["branch_kl_t1"]) <= float(
                projected["metrics"]["branch_kl_t1"]) - float(
                    threshold["minimum_branch_kl_improvement"]),
        "equal_storage": max(float(row["budget_utilization"]) for row in rows)
        - min(float(row["budget_utilization"]) for row in rows) <= float(
            threshold["maximum_budget_utilization_spread"]),
    }
    decision = "promote_branching_distribution_compiler" if all(checks.values()) \
        else "reject_or_iterate"
    summary = {
        "study": "qwen06b_hierarchical_distribution_compilation_v1",
        "rows": rows,
        "checks": checks,
        "decision": decision,
    }
    atomic_json(summary, args.out)
    print(summary)


if __name__ == "__main__":
    main()
