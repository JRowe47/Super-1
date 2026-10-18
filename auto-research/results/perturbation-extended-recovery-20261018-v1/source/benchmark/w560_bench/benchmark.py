from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
from typing import Mapping

from .assets import validate_assets
from .common import hash_file_same_fd, read_json_same_fd, write_exclusive_json
from .config import runtime_root
from .lineage import load_exact_head_model, solver_idle_boundary, validate_accepted_promotion
from .tasks import parse_jsonl_line


def _load_tokenizer(config_path: Path, config: Mapping[str, object]):
    from transformers import AutoTokenizer

    root = runtime_root(config_path)
    identity, identity_artifact = read_json_same_fd(root / config["tokenizer"]["identity_relative_path"])
    tokenizer_root = root / "inputs/tokenizer"
    for name, expected in config["tokenizer"]["files"].items():
        hash_file_same_fd(tokenizer_root / name, expected["sha256"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    mapping = tokenizer.get_vocab()
    observed = {
        "base_vocab_size": int(tokenizer.vocab_size), "tokenizer_length": len(tokenizer),
        "mapping_count": len(mapping), "maximum_token_id": max(mapping.values()),
        "minimum_token_id": min(mapping.values()), "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id, "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
    }
    for key, value in observed.items():
        if identity.get(key) != value:
            raise RuntimeError(f"AutoTokenizer runtime identity drift: {key}")
    if int(identity["maximum_token_id"]) >= int(identity["model_vocab_size"]):
        raise RuntimeError("tokenizer ID exceeds model vocabulary")
    return tokenizer, identity, identity_artifact


def _encode(tokenizer: object, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _continuation_stats(
    model: object, tokenizer: object, context: str, continuation: str, *,
    device: object, maximum_combined_tokens: int, eos_id: int,
) -> tuple[Mapping[str, object], int]:
    import torch

    context_ids = _encode(tokenizer, context)
    continuation_ids = _encode(tokenizer, continuation)
    if not continuation_ids:
        raise RuntimeError("empty continuation tokenization")
    if not context_ids:
        context_ids = [eos_id]
    combined = (context_ids + continuation_ids)[-maximum_combined_tokens:]
    n_cont = min(len(continuation_ids), len(combined) - 1)
    if n_cont <= 0:
        raise RuntimeError("no continuation target positions remain")
    inputs = torch.tensor([combined[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(combined[-n_cont:], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="mps", dtype=torch.bfloat16, enabled=True):
        logits = model(inputs).logits.float()[0, -n_cont:]
    log_probs = torch.log_softmax(logits, dim=-1)
    gold_log_probs = log_probs.gather(-1, targets[:, None]).squeeze(-1)
    gold_logits = logits.gather(-1, targets[:, None]).squeeze(-1)
    predicted = logits.argmax(dim=-1)
    masked = logits.clone()
    masked.scatter_(1, targets[:, None], float("-inf"))
    best_wrong = masked.max(dim=-1).values
    return {
        "mean_loglikelihood": float(gold_log_probs.mean().cpu()),
        "mean_token_nll": float((-gold_log_probs.mean()).cpu()),
        "mean_gold_vs_best_wrong_token_logit_margin": float((gold_logits - best_wrong).mean().cpu()),
        "teacher_forced_exact_match_under_gold_prefixes": bool(torch.equal(predicted, targets)),
        "scored_token_count": n_cont,
    }, 1


def _evaluate(
    model: object, tokenizer: object, rows: Mapping[str, list[Mapping[str, object]]],
    config: Mapping[str, object], device: object, tokenizer_identity: Mapping[str, object],
) -> Mapping[str, object]:
    task_results = {}
    total_forwards = 0
    maximum = int(config["evaluation"]["maximum_combined_tokens"])
    eos_id = int(tokenizer_identity["eos_token_id"])
    for task in config["task_order"]:
        examples = []
        for row in rows[task]:
            if task == "lambada":
                stats, forwards = _continuation_stats(
                    model, tokenizer, row["context"], row["continuation"],
                    device=device, maximum_combined_tokens=maximum, eos_id=eos_id,
                )
                total_forwards += forwards
                examples.append({
                    "index": row["index"], "example_sha256": row["example_sha256"],
                    "correct": stats["teacher_forced_exact_match_under_gold_prefixes"],
                    "nll": stats["mean_token_nll"],
                    "margin": stats["mean_gold_vs_best_wrong_token_logit_margin"],
                    "scored_token_count": stats["scored_token_count"],
                })
            else:
                choices = []
                for choice in row["choices"]:
                    stats, forwards = _continuation_stats(
                        model, tokenizer, row["context"], choice, device=device,
                        maximum_combined_tokens=maximum, eos_id=eos_id,
                    )
                    choices.append(stats)
                    total_forwards += forwards
                gold = int(row["gold"])
                normalized = [float(value["mean_loglikelihood"]) for value in choices]
                best_wrong = max(value for index, value in enumerate(normalized) if index != gold)
                selected = max(range(len(normalized)), key=normalized.__getitem__)
                examples.append({
                    "index": row["index"], "example_sha256": row["example_sha256"],
                    "correct": selected == gold, "nll": -normalized[gold],
                    "margin": normalized[gold] - best_wrong, "gold": gold, "selected": selected,
                    "choice_normalized_loglikelihoods": normalized,
                })
        count = len(examples)
        if task == "lambada":
            scored_tokens = sum(int(value["scored_token_count"]) for value in examples)
            mean_nll = sum(
                float(value["nll"]) * int(value["scored_token_count"])
                for value in examples
            ) / scored_tokens
            nll_aggregation = "TOKEN_WEIGHTED_OVER_ALL_SCORED_TARGET_TOKENS"
            mean_margin = sum(
                float(value["margin"]) * int(value["scored_token_count"])
                for value in examples
            ) / scored_tokens
            margin_aggregation = "TOKEN_WEIGHTED_OVER_ALL_SCORED_TARGET_TOKENS"
        else:
            scored_tokens = None
            mean_nll = sum(float(value["nll"]) for value in examples) / count
            nll_aggregation = "EQUAL_EXAMPLE_MEAN_OF_GOLD_CHOICE_NORMALIZED_NLL"
            mean_margin = sum(float(value["margin"]) for value in examples) / count
            margin_aggregation = "EQUAL_EXAMPLE_MEAN_GOLD_MINUS_BEST_WRONG_MARGIN"
        task_results[task] = {
            "count": count,
            "accuracy": sum(int(value["correct"]) for value in examples) / count,
            "mean_nll": mean_nll,
            "perplexity_from_mean_nll": __import__("math").exp(mean_nll),
            "nll_aggregation": nll_aggregation,
            "scored_target_token_count": scored_tokens,
            "mean_margin": mean_margin,
            "margin_aggregation": margin_aggregation,
            "examples": examples,
        }
    return {"tasks": task_results, "model_forward_calls": total_forwards}


def _bootstrap(
    values: list[float], seed_material: str, resamples: int,
    weights: list[int] | None = None,
) -> Mapping[str, float]:
    if not values:
        raise RuntimeError("cannot bootstrap empty paired vector")
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little"))
    count = len(values)
    if weights is not None and (len(weights) != count or any(value <= 0 for value in weights)):
        raise RuntimeError("invalid paired bootstrap weights")
    def estimate(indices: list[int]) -> float:
        if weights is None:
            return sum(values[index] for index in indices) / count
        denominator = sum(weights[index] for index in indices)
        return sum(values[index] * weights[index] for index in indices) / denominator
    estimates = sorted(estimate([rng.randrange(count) for _ in range(count)]) for _ in range(resamples))
    point = estimate(list(range(count)))
    return {
        "mean": point,
        "ci95_lower": estimates[int(0.025 * (resamples - 1))],
        "ci95_upper": estimates[int(0.975 * (resamples - 1))],
        "resamples": resamples,
    }


def _paired(
    prior: Mapping[str, object], current: Mapping[str, object],
    config: Mapping[str, object], head_pair: str,
) -> Mapping[str, object]:
    result = {}
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    for task in config["task_order"]:
        left = prior["tasks"][task]["examples"]
        right = current["tasks"][task]["examples"]
        if len(left) != len(right):
            raise RuntimeError(f"paired row-count drift: {task}")
        accuracy_delta: list[float] = []
        nll_delta: list[float] = []
        margin_delta: list[float] = []
        token_weights: list[int] = []
        wins = losses = stable_correct = stable_incorrect = 0
        flips = []
        examples = []
        for before, after in zip(left, right):
            if before["example_sha256"] != after["example_sha256"] or before["index"] != after["index"]:
                raise RuntimeError(f"paired example identity drift: {task}")
            before_correct, after_correct = bool(before["correct"]), bool(after["correct"])
            if not before_correct and after_correct:
                wins += 1
                flips.append({"index": before["index"], "direction": "INCORRECT_TO_CORRECT", "example_sha256": before["example_sha256"]})
            elif before_correct and not after_correct:
                losses += 1
                flips.append({"index": before["index"], "direction": "CORRECT_TO_INCORRECT", "example_sha256": before["example_sha256"]})
            else:
                stable_correct += int(before_correct)
                stable_incorrect += int(not before_correct)
            accuracy_delta.append(float(after_correct) - float(before_correct))
            nll_delta.append(float(after["nll"]) - float(before["nll"]))
            margin_delta.append(float(after["margin"]) - float(before["margin"]))
            if task == "lambada":
                if int(before["scored_token_count"]) != int(after["scored_token_count"]):
                    raise RuntimeError("paired LAMBADA scored-token count drift")
                token_weights.append(int(before["scored_token_count"]))
            examples.append({
                "index": before["index"], "example_sha256": before["example_sha256"],
                "prior_correct": before_correct, "current_correct": after_correct,
                "prior_nll": before["nll"], "current_nll": after["nll"],
                "prior_margin": before["margin"], "current_margin": after["margin"],
            })
        result[task] = {
            "incorrect_to_correct": wins, "correct_to_incorrect": losses,
            "stable_correct": stable_correct, "stable_incorrect": stable_incorrect,
            "all_correctness_flips": flips,
            "accuracy_current_minus_prior": _bootstrap(accuracy_delta, f"{head_pair}:{task}:accuracy", resamples),
            "nll_current_minus_prior": _bootstrap(
                nll_delta, f"{head_pair}:{task}:nll", resamples,
                token_weights if task == "lambada" else None,
            ),
            "margin_current_minus_prior": _bootstrap(
                margin_delta, f"{head_pair}:{task}:margin", resamples,
                token_weights if task == "lambada" else None,
            ),
            "examples": examples,
        }
    return result


def _load_rows(
    config_path: Path, assets: Mapping[str, object], mode: str,
) -> Mapping[str, list[Mapping[str, object]]]:
    config = assets["package"]["config"]
    root = runtime_root(config_path)
    limit = int(
        config["evaluation"]["canary_first_n"]
        if mode == "canary" else config["evaluation"]["full_first_n"]
    )
    rows = {}
    for task in config["task_order"]:
        values = []
        path = root / "assets" / assets["manifest"]["datasets"][task]["normalized_relative_path"]
        with path.open("rb") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    break
                values.append(parse_jsonl_line(line, task, index))
        expected = min(int(config["datasets"][task]["expected_rows"]), limit)
        if len(values) != expected:
            raise RuntimeError(f"{task} fixed-first-N row-count drift")
        rows[task] = values
    return rows


def benchmark(config_path: Path, cycle_receipt: Path, mode: str) -> Mapping[str, object]:
    if mode not in ("canary", "full"):
        raise RuntimeError("mode must be canary or full")
    assets = validate_assets(config_path)
    config = assets["package"]["config"]
    os.environ.update({
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "DO_NOT_TRACK": "1",
    })
    with solver_idle_boundary(config):
        trigger = validate_accepted_promotion(config_path, cycle_receipt)
        current_sha = trigger["current_head"]["record_sha256"]
        root = runtime_root(config_path)
        output = root / "results" / f"{current_sha}.{mode}.result.json"
        receipt = root / "receipts" / f"{current_sha}.{mode}.receipt.json"
        if output.exists() or receipt.exists():
            raise FileExistsError("create-only benchmark result already exists for head/mode")
        import torch
        if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
            raise RuntimeError("MPS is unavailable")
        device = torch.device("mps")
        tokenizer, tokenizer_identity, tokenizer_identity_artifact = _load_tokenizer(config_path, config)
        rows = _load_rows(config_path, assets, mode)
        prior_loaded, prior_protocol = load_exact_head_model(config, trigger["prior_head"], device)
        prior_scores = _evaluate(prior_loaded.model, tokenizer, rows, config, device, tokenizer_identity)
        del prior_loaded
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        current_loaded, current_protocol = load_exact_head_model(config, trigger["current_head"], device)
        current_scores = _evaluate(current_loaded.model, tokenizer, rows, config, device, tokenizer_identity)
        paired = _paired(
            prior_scores, current_scores, config,
            f"{trigger['prior_head']['record_sha256']}:{current_sha}:{mode}",
        )
        result = {
            "schema": "hfw_w560_promotion_benchmark_result_v2",
            "status": "PASS_REPORTING_ONLY_BENCHMARK_COMPLETE",
            "mode": mode, "disclosure": config["disclosure"],
            "held_out_generalization_claimed": False, "benchmark_may_gate_or_promote": False,
            "prior_head_sha256": trigger["prior_head"]["record_sha256"],
            "prior_checkpoint_sha256": trigger["prior_head"]["checkpoint"]["sha256"],
            "current_head_sha256": current_sha,
            "current_checkpoint_sha256": trigger["current_head"]["checkpoint"]["sha256"],
            "cycle_receipt_sha256": trigger["cycle_identity"]["sha256"],
            "decision_sha256": trigger["decision_identity"]["sha256"],
            "asset_manifest_sha256": assets["manifest_identity"]["sha256"],
            "asset_import_receipt_sha256": assets["import_identity"]["sha256"],
            "tokenizer_revision": config["tokenizer"]["revision"],
            "tokenizer_identity_sha256": tokenizer_identity_artifact["sha256"],
            "runtime_manifest_sha256": assets["package"]["runtime_manifest_identity"]["sha256"],
            "prior": prior_scores, "current": current_scores, "paired": paired,
            "device": "mps", "autocast_dtype": "bfloat16", "batch_size": 1,
            "selection": "FIXED_FIRST_N_PINNED_SOURCE_ORDER_NO_SHUFFLE",
            "maximum_combined_tokens": config["evaluation"]["maximum_combined_tokens"],
            "training_or_backprop": False, "optimizer_steps": 0,
            "checkpoint_writes": 0, "lineage_writes": 0, "network_access": False,
            "protocol_receipts": {"prior": prior_protocol, "current": current_protocol},
        }
        result_identity = write_exclusive_json(output, result)
        receipt_body = {
            "schema": "hfw_w560_promotion_benchmark_receipt_v2",
            "status": "PASS_CREATE_ONLY_REPORTING_RECEIPT",
            "result_sha256": result_identity["sha256"], "result_size_bytes": result_identity["size_bytes"],
            "current_head_sha256": current_sha, "current_checkpoint_sha256": result["current_checkpoint_sha256"],
            "prior_head_sha256": result["prior_head_sha256"], "prior_checkpoint_sha256": result["prior_checkpoint_sha256"],
            "cycle_receipt_sha256": result["cycle_receipt_sha256"], "decision_sha256": result["decision_sha256"],
            "asset_manifest_sha256": result["asset_manifest_sha256"],
            "asset_import_receipt_sha256": result["asset_import_receipt_sha256"],
            "tokenizer_identity_sha256": result["tokenizer_identity_sha256"],
            "runtime_manifest_sha256": result["runtime_manifest_sha256"],
            "mode": mode, "benchmark_may_gate_or_promote": False,
            "held_out_generalization_claimed": False, "disclosure": config["disclosure"],
            "checkpoint_writes": 0, "lineage_writes": 0, "network_access": False,
        }
        receipt_identity = write_exclusive_json(receipt, receipt_body)
        return {**receipt_body, "result": result_identity, "receipt": receipt_identity}
