#!/usr/bin/env python3
"""Recompute comparison metrics from raw result and benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_training_rows(path: Path) -> dict[int, dict]:
    if path.suffix == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row.get("event") == "train":
                        rows.append(row)
    else:
        payload = load_json(path)
        rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return {int(row["step"]): row for row in rows}


def fixed_metrics(result: dict, contexts: list[int]) -> tuple[list[float], list[int]]:
    metrics = result["metrics"]
    ce: list[float] = []
    state: list[int] = []
    for context in contexts:
        if context == 512:
            ce.append(float(metrics["student_ce"]))
        else:
            ce.append(float(metrics["long_context"][str(context)]["student_ce"]))
        state.append(int(metrics[f"state_bytes_bf16_per_sequence_at_{context}"]))
    return ce, state


def build_metrics(
    package: dict,
    hierarchy_result: dict,
    hierarchy_benchmark: dict,
    hierarchy_training_rows: dict[int, dict],
    nanogpt_result: dict,
    nanogpt_benchmark: dict,
    nanogpt_training_rows: dict[int, dict],
    recorded_artifacts: dict,
) -> dict:
    contexts = [int(value) for value in package["comparison"]["contexts"]]
    hierarchy_ce, hierarchy_state = fixed_metrics(hierarchy_result, contexts)
    nanogpt_ce, nanogpt_state = fixed_metrics(nanogpt_result, contexts)
    ratios = [round(n / h, 12) for n, h in zip(nanogpt_state, hierarchy_state)]
    reductions = [
        round(100.0 * (n - h) / n, 12)
        for n, h in zip(nanogpt_state, hierarchy_state)
    ]

    selected_steps = sorted(set(hierarchy_training_rows) & set(nanogpt_training_rows))
    selected_tokens = [int(hierarchy_training_rows[step]["tokens"]) for step in selected_steps]
    for step, tokens in zip(selected_steps, selected_tokens):
        if int(nanogpt_training_rows[step]["tokens"]) != tokens:
            raise RuntimeError(f"training token mismatch at step {step}")

    task_keys = [item["key"] for item in package["benchmark_task_order"]]
    task_labels = [item["label"] for item in package["benchmark_task_order"]]
    nano_scores = [float(nanogpt_benchmark["scores"][key]) for key in task_keys]
    hierarchy_scores = [float(hierarchy_benchmark["scores"][key]) for key in task_keys]
    nano_mean = sum(nano_scores) / len(nano_scores)
    hierarchy_mean = sum(hierarchy_scores) / len(hierarchy_scores)
    if not math.isclose(nano_mean, float(nanogpt_benchmark["mean_score"]), rel_tol=0, abs_tol=1e-15):
        raise RuntimeError("nanoGPT benchmark mean does not recompute")
    if not math.isclose(hierarchy_mean, float(hierarchy_benchmark["mean_score"]), rel_tol=0, abs_tol=1e-15):
        raise RuntimeError("hierarchy benchmark mean does not recompute")

    recorded_h = recorded_artifacts["historical_hierarchy"]
    recorded_n = recorded_artifacts["official_nanogpt"]
    inputs = package["inputs"]
    source = load_json(ROOT / "provenance/source_identity.json")
    return {
        "report_date": package["created_utc"][:10],
        "steps": int(package["execution_contract"]["steps"]),
        "tokens": int(package["execution_contract"]["executed_tokens"]),
        "parameters": {
            "hierarchy": int(hierarchy_result["stored_parameters"]),
            "nanogpt": int(nanogpt_result["stored_parameters"]),
        },
        "fixed_eval": {
            "contexts": contexts,
            "nanogpt_ce": nanogpt_ce,
            "hierarchy_ce": hierarchy_ce,
            "eval_shapes": [
                f"{package['fixed_evaluation'][str(context)]['batches']} batches x "
                f"{package['fixed_evaluation'][str(context)]['batch_size']} "
                f"{'sequence' if package['fixed_evaluation'][str(context)]['batch_size'] == 1 else 'sequences'}"
                for context in contexts
            ],
        },
        "state": {
            "hierarchy_bytes": hierarchy_state,
            "nanogpt_bytes": nanogpt_state,
            "nanogpt_to_hierarchy_ratio": ratios,
            "hierarchy_percent_fewer": reductions,
        },
        "training_checkpoints": {
            "steps": selected_steps,
            "tokens": selected_tokens,
            "nanogpt_ce": [float(nanogpt_training_rows[step]["ce"]) for step in selected_steps],
            "hierarchy_ce": [float(hierarchy_training_rows[step]["ce"]) for step in selected_steps],
        },
        "benchmark": {
            "tasks": task_labels + ["Mean"],
            "nanogpt": nano_scores + [nano_mean],
            "hierarchy": hierarchy_scores + [hierarchy_mean],
            "limit_per_task": int(nanogpt_benchmark["limit"]),
        },
        "provenance": {
            "hierarchy_config_sha256": recorded_h["config"],
            "hierarchy_result_sha256": recorded_h["result"],
            "hierarchy_benchmark_sha256": recorded_h["benchmark"],
            "hierarchy_checkpoint_sha256": recorded_h["final_checkpoint_not_included"],
            "nanogpt_manifest_sha256": source["official_nanogpt"]["freeze_manifest_sha256"],
            "nanogpt_result_sha256": recorded_n["result"],
            "nanogpt_training_completion_sha256": recorded_n["training_completion"],
            "nanogpt_final_model_sha256": recorded_n["final_model_not_included"],
            "nanogpt_training_log_sha256": recorded_n["training_log"],
            "nanogpt_benchmark_sha256": recorded_n["benchmark"],
            "nanogpt_benchmark_completion_sha256": recorded_n["benchmark_completion"],
            "nanogpt_upstream_commit": package["models"]["nanogpt"]["upstream_commit"],
            "nanogpt_model_sha256": source["official_nanogpt"]["vendored_model_sha256"],
            "train_data_sha256": inputs["train"]["sha256"],
            "val_data_sha256": inputs["validation"]["sha256"],
            "teacher_weights_sha256": inputs["teacher"]["weights_sha256"],
        },
    }


def recorded_paths(package: dict) -> dict[str, Path]:
    paths = package["paths"]
    return {key: ROOT / value for key, value in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorded", action="store_true")
    parser.add_argument("--hierarchy-result", type=Path)
    parser.add_argument("--hierarchy-benchmark", type=Path)
    parser.add_argument("--hierarchy-training-rows", type=Path)
    parser.add_argument("--nanogpt-result", type=Path)
    parser.add_argument("--nanogpt-benchmark", type=Path)
    parser.add_argument("--nanogpt-training-log", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    package = load_json(ROOT / "package.json")
    if args.recorded:
        defaults = recorded_paths(package)
        hierarchy_result = defaults["recorded_hierarchy_result"]
        hierarchy_benchmark = defaults["recorded_hierarchy_benchmark"]
        hierarchy_rows = defaults["reported_hierarchy_training_rows"]
        nanogpt_result = defaults["recorded_nanogpt_result"]
        nanogpt_benchmark = defaults["recorded_nanogpt_benchmark"]
        nanogpt_log = defaults["recorded_nanogpt_training_log"]
    else:
        required = {
            "hierarchy_result": args.hierarchy_result,
            "hierarchy_benchmark": args.hierarchy_benchmark,
            "hierarchy_training_rows": args.hierarchy_training_rows,
            "nanogpt_result": args.nanogpt_result,
            "nanogpt_benchmark": args.nanogpt_benchmark,
            "nanogpt_training_log": args.nanogpt_training_log,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"missing fresh-run inputs: {', '.join(missing)}")
        hierarchy_result = args.hierarchy_result
        hierarchy_benchmark = args.hierarchy_benchmark
        hierarchy_rows = args.hierarchy_training_rows
        nanogpt_result = args.nanogpt_result
        nanogpt_benchmark = args.nanogpt_benchmark
        nanogpt_log = args.nanogpt_training_log

    metrics = build_metrics(
        package,
        load_json(hierarchy_result),
        load_json(hierarchy_benchmark),
        load_training_rows(hierarchy_rows),
        load_json(nanogpt_result),
        load_json(nanogpt_benchmark),
        load_training_rows(nanogpt_log),
        load_json(ROOT / "provenance/recorded_artifacts.json"),
    )
    if args.check:
        expected = load_json(args.check)
        if metrics != expected:
            raise RuntimeError("recomputed comparison differs from expected metrics")
        print("PASS_RECOMPUTED_COMPARISON")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(args.out)
    if not args.check and not args.out:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
