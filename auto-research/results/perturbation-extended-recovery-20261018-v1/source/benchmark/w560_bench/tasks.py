from __future__ import annotations

import hashlib
import json
from typing import Mapping

from .common import canonical_json_bytes


def normalize_row(task: str, row: Mapping[str, object], index: int) -> Mapping[str, object]:
    if task == "lambada":
        text = str(row["text"]).strip()
        words = text.split()
        if len(words) < 2:
            raise RuntimeError(f"LAMBADA row {index} has fewer than two words")
        normalized = {"index": index, "context": " ".join(words[:-1]), "continuation": " " + words[-1]}
    elif task == "hellaswag":
        endings = list(row["endings"])
        if len(endings) != 4:
            raise RuntimeError(f"HellaSwag row {index} choice-count drift")
        normalized = {"index": index, "context": str(row["ctx"]), "choices": [" " + str(value).lstrip() for value in endings], "gold": int(row["label"])}
    elif task == "arc_easy":
        choices = row["choices"]
        labels = [str(value) for value in choices["label"]]
        texts = [str(value) for value in choices["text"]]
        answer = str(row["answerKey"])
        if answer not in labels or len(texts) < 2:
            raise RuntimeError(f"ARC-Easy row {index} answer/choices drift")
        normalized = {"index": index, "context": str(row["question"]), "choices": [" " + value.lstrip() for value in texts], "gold": labels.index(answer)}
    elif task == "piqa":
        normalized = {"index": index, "context": str(row["goal"]), "choices": [" " + str(row["sol1"]).lstrip(), " " + str(row["sol2"]).lstrip()], "gold": int(row["label"])}
    else:
        raise RuntimeError(f"unsupported task: {task}")
    digest_body = {"task": task, **normalized}
    return {**normalized, "example_sha256": hashlib.sha256(canonical_json_bytes(digest_body)).hexdigest()}


def parse_jsonl_line(line: bytes, task: str, expected_index: int) -> Mapping[str, object]:
    row = json.loads(line)
    if int(row.get("index", -1)) != expected_index:
        raise RuntimeError(f"{task} normalized index drift")
    digest = row.pop("example_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes({"task": task, **row})).hexdigest()
    if digest != expected:
        raise RuntimeError(f"{task} normalized example SHA drift at {expected_index}")
    return {**row, "example_sha256": digest}
