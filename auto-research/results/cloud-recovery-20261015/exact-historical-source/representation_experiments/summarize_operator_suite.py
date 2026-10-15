"""Apply preregistered promotion gates to all four operator arms."""
from __future__ import annotations

import argparse
from pathlib import Path

from representation_experiments.common import atomic_json, load_json


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = arguments()
    config = load_json(args.config)
    gates = config["promotion_gates"]
    reference_context = str(gates["reference_context"])
    long_context = str(gates["long_context"])
    rows = []
    for arm in config["arms"]:
        path = Path(args.root) / arm / "result.json"
        if not path.exists():
            rows.append({"arm": arm, "complete": False, "promoted": False})
            continue
        result = load_json(path)
        reference = result["metrics"][reference_context]
        long = result["metrics"][long_context]
        checks = {
            "teacher_kl": reference["teacher_kl"] <= gates["max_teacher_kl"],
            "top1_agreement": reference["top1_agreement"] >= gates["min_top1_agreement"],
            "attention_nmse": reference["attention_nmse"] <= gates["max_attention_nmse"],
            "residual_nmse": reference["residual_nmse"] <= gates["max_residual_nmse"],
            "stored_attention": result["parameter_ratio"] < 1.0,
            "estimated_flops": result["estimated_flop_ratio"] <= 1.0,
            "measured_flops_available": result["measured_flop_ratio"] is not None,
            "measured_flops": (result["measured_flop_ratio"] is not None
                               and result["measured_flop_ratio"] <= 1.0),
            "long_context": long["teacher_kl"] <= gates["max_long_context_kl"],
            "long_context_degradation": long["teacher_kl"] <= (
                gates["max_long_context_multiplier"]
                * max(reference["teacher_kl"], 1e-8)),
        }
        rows.append({
            "arm": arm, "complete": True, "checks": checks,
            "promoted": all(checks.values()),
            "reference": reference, "long_context": long,
            "parameter_ratio": result["parameter_ratio"],
            "estimated_flop_ratio": result["estimated_flop_ratio"],
            "measured_flop_ratio": result["measured_flop_ratio"],
        })
    promoted = [row for row in rows if row.get("promoted")]
    promoted.sort(key=lambda row: (
        row["reference"]["teacher_kl"], row["parameter_ratio"]))
    summary = {
        "study": "qwen06b_functional_attention_operator_replacement",
        "rows": rows,
        "winner": promoted[0]["arm"] if promoted else None,
        "promotion_count": len(promoted),
        "decision": "promote" if promoted else "no_operator_met_all_gates",
    }
    atomic_json(summary, args.out)
    print(summary)


if __name__ == "__main__":
    main()
