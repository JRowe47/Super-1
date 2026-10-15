"""Official nanoGPT-only model factory for the historical comparison contract."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from representation_experiments.common import parameter_count
from representation_experiments.nanogpt_control import NanoGPTStudent


@dataclass
class BudgetedStudent:
    model: NanoGPTStudent
    width: int
    parameters: int
    target_parameters: int


def official_bias_parameter_count(vocab_size: int, block_size: int,
                                  width: int, layers: int) -> int:
    """Exact tied-embedding upstream nanoGPT count when ``bias=True``."""
    embeddings = int(vocab_size) * int(width) + int(block_size) * int(width)
    per_block = 12 * int(width) * int(width) + 13 * int(width)
    final_norm = 2 * int(width)
    return embeddings + int(layers) * per_block + final_norm


def resolve_budget(kind: str, vocab_size: int, max_context: int, config: dict,
                   target_parameters: int) -> dict:
    if kind != "nanogpt":
        raise ValueError(
            "this comparator package authorizes only the official nanoGPT arm")
    search = dict(config["geometry_search"])
    if int(search["maximum_stored_parameters"]) != int(target_parameters):
        raise ValueError("geometry ceiling differs from the parameter budget")
    target_match = int(search["target_stored_parameters"])
    records = []
    for heads in range(
            int(search["minimum_heads"]), int(search["maximum_heads"]) + 1):
        width = heads * int(search["head_dimension"])
        for layers in range(
                int(search["minimum_layers"]),
                int(search["maximum_layers"]) + 1):
            count = official_bias_parameter_count(
                vocab_size, max_context, width, layers)
            if count <= target_parameters:
                records.append((
                    abs(count - target_match), -count, layers, heads,
                    width, count))
    if not records:
        raise ValueError("official nanoGPT cannot fit under the parameter budget")
    chosen = min(records)
    result = {
        "width": chosen[4],
        "layers": chosen[2],
        "heads": chosen[3],
        "stored_parameters": chosen[5],
        "target_match_parameters": target_match,
        "target_parameters": int(target_parameters),
    }
    with torch.device("meta"):
        candidate = build_student(
            kind, vocab_size, result["width"], max_context,
            {**config, "heads": result["heads"], "layers": result["layers"]})
    if parameter_count(candidate) != result["stored_parameters"]:
        raise RuntimeError("analytic and upstream nanoGPT parameter counts disagree")
    return result


def build_student(kind: str, vocab_size: int, width: int, max_context: int,
                  config: dict) -> NanoGPTStudent:
    if kind != "nanogpt":
        raise ValueError(
            "this comparator package authorizes only the official nanoGPT arm")
    return NanoGPTStudent(
        vocab_size=vocab_size,
        width=width,
        max_context=max_context,
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        bias=bool(config["bias"]),
    )


def build_to_budget(kind: str, vocab_size: int, max_context: int, config: dict,
                    target_parameters: int) -> BudgetedStudent:
    if kind != "nanogpt":
        raise ValueError(
            "this comparator package authorizes only the official nanoGPT arm")
    resolved = resolve_budget(
        kind, vocab_size, max_context, config, target_parameters)
    width = int(resolved["width"])
    expected_count = int(resolved["stored_parameters"])
    model_config = {
        **config,
        "layers": int(resolved["layers"]),
        "heads": int(resolved["heads"]),
    }
    model = build_student(kind, vocab_size, width, max_context, model_config)
    observed_count = parameter_count(model)
    if observed_count != expected_count:
        raise RuntimeError(
            "meta-device and materialized nanoGPT parameter counts disagree")
    return BudgetedStudent(
        model=model,
        width=width,
        parameters=observed_count,
        target_parameters=int(target_parameters),
    )
