"""Matched public-task evaluation for all three predictive-state students."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from representation_experiments.common import atomic_json, load_json
from representation_experiments.predictive_models import build_to_budget


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else contextlib.nullcontext())


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


@torch.no_grad()
def sequence_score(model, tokenizer, context, continuation, length, device):
    prefix = encode(tokenizer, context) or [tokenizer.eos_token_id]
    suffix = encode(tokenizer, continuation)
    combined = (prefix + suffix)[-(length + 1):]
    continuation_tokens = min(len(suffix), len(combined) - 1)
    inputs = torch.tensor([combined[:-1]], device=device)
    with autocast(device):
        logits = model(inputs).logits[0]
    targets = torch.tensor(combined[1:], device=device)
    values = F.log_softmax(logits.float(), dim=-1)[
        torch.arange(targets.numel(), device=device), targets]
    return float(values[-continuation_tokens:].sum()), continuation_tokens


@torch.no_grad()
def greedy_match(model, tokenizer, context, continuation, length, device):
    prefix = encode(tokenizer, context) or [tokenizer.eos_token_id]
    suffix = encode(tokenizer, continuation)
    combined = (prefix + suffix)[-(length + 1):]
    count = min(len(suffix), len(combined) - 1)
    inputs = torch.tensor([combined[:-1]], device=device)
    with autocast(device):
        prediction = model(inputs).logits[0, -count:].argmax(-1)
    return bool(torch.equal(prediction, torch.tensor(combined[-count:], device=device)))


def load_tasks(limit):
    from datasets import load_dataset

    def take(dataset):
        return dataset.select(range(min(limit, len(dataset)))) if limit > 0 else dataset

    lambada = take(load_dataset("EleutherAI/lambada_openai", "en", split="test"))
    hellaswag = take(load_dataset("Rowan/hellaswag", split="validation"))
    arc = take(load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation"))
    piqa = take(load_dataset("lighteval/piqa", split="validation"))
    return {
        "lambada": [(" ".join(row["text"].split()[:-1]), " " + row["text"].split()[-1])
                    for row in lambada],
        "hellaswag": [(row["ctx"], [" " + value for value in row["endings"]],
                       int(row["label"])) for row in hellaswag],
        "arc_easy": [(row["question"], [" " + value for value in row["choices"]["text"]],
                      row["choices"]["label"].index(row["answerKey"])) for row in arc],
        "piqa": [(row["goal"], [" " + row["sol1"], " " + row["sol2"]],
                  int(row["label"])) for row in piqa],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--student", required=True,
                        choices=("transformer", "recurrent_logits", "hierarchical_predictive",
                                 "hierarchical_logit_control"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["teacher"])
    budgeted = build_to_budget(
        args.student, int(config["vocab_size"]),
        int(config["student_training"]["context"]), config["students"][args.student],
        int(config["student_parameter_budget"]))
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
            if index % 100 == 0:
                print(json.dumps({"task": task, "items": index,
                                  "accuracy": correct / index}), flush=True)
        scores[task] = correct / len(items)
    result = {
        "student": args.student, "limit": args.limit,
        "stored_parameters": budgeted.parameters, "scores": scores,
        "metrics": {
            "lambada": "greedy_exact_match",
            "multiple_choice": "length_normalized_loglikelihood",
        },
    }
    atomic_json(result, args.out)
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
