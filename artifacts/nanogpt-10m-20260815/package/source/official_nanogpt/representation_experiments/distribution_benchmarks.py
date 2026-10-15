"""Matched public-task evaluation for distribution-compiled recurrent students."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from representation_experiments.common import atomic_json, load_json
from representation_experiments.predictive_benchmarks import (
    greedy_match,
    load_tasks,
    sequence_score,
)
from representation_experiments.predictive_models import build_to_budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        print("SKIP", output, flush=True)
        return
    config = load_json(args.config)
    arm = config["arms"][args.arm]
    architecture = arm["architecture"]
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    from transformers import AutoTokenizer
    teacher_snapshot_root = os.environ.get("AUG8_TEACHER_SNAPSHOT_ROOT")
    if not teacher_snapshot_root:
        raise RuntimeError("exact teacher snapshot root was not bound by launcher")
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_snapshot_root, local_files_only=True)
    budgeted = build_to_budget(
        architecture, int(config["vocab_size"]), int(config.get(
            "model_max_context", config["training"]["context"])),
        config["models"][architecture], int(arm.get(
            "parameter_budget", config["student_parameter_budget"])))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    budgeted.model.load_state_dict(payload["model"])
    model = budgeted.model.to(device).eval()
    tasks = load_tasks(args.limit)
    scores = {}
    for task, items in tasks.items():
        correct = 0
        for index, item in enumerate(items, start=1):
            if task == "lambada":
                correct += int(greedy_match(
                    model, tokenizer, item[0], item[1], model.max_context, device))
            else:
                context, choices, gold = item
                values = []
                for choice in choices:
                    score, tokens = sequence_score(
                        model, tokenizer, context, choice, model.max_context, device)
                    values.append(score / max(tokens, 1))
                correct += int(int(np.argmax(values)) == gold)
            if index % int(config["benchmark"]["log_every"]) == 0:
                print(json.dumps({
                    "task": task, "items": index, "accuracy": correct / index,
                }), flush=True)
        scores[task] = correct / len(items)
    result = {
        "arm": args.arm,
        "architecture": architecture,
        "limit": args.limit,
        "stored_parameters": budgeted.parameters,
        "scores": scores,
        "mean_score": sum(scores.values()) / len(scores),
        "metrics": {
            "lambada": "greedy_exact_match",
            "multiple_choice": "length_normalized_loglikelihood",
        },
    }
    atomic_json(result, output)
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
