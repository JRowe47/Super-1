from __future__ import annotations

import hashlib
import os
import math
import time
import stat
from pathlib import Path
from typing import Mapping

import torch
import numpy as np

from .branch_search import validate_branch_artifact_chain
from .manifest import canonical_json_bytes, file_sha256
from .production_bridge import LoadedW560Parent, W560ForwardOnlyObjective


def _load_materialized_child_same_fd(
    path: Path, expected_sha256: str,
) -> Mapping[str, object]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("materialized child is not a regular same-FD file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("materialized child pre-load SHA drift")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("materialized child post-load SHA drift")
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError("materialized child changed during exact load")
        return payload
    finally:
        os.close(descriptor)


def execute_report_only_fixed_eval(
    protocol: Mapping[str, object], loaded_parent: LoadedW560Parent, *,
    authorization: Mapping[str, object], validation_path: Path,
    search_root: Path, device: torch.device,
) -> Mapping[str, object]:
    started = time.monotonic()
    if authorization.get("status") != "PASS_FIXED_EVAL_MAY_NOW_OPEN":
        raise RuntimeError("fixed evaluation is not authorized")
    if authorization.get("all_four_branches_complete") is not True:
        raise RuntimeError("fixed evaluation requires all four branches")
    data = protocol["data"]
    if file_sha256(validation_path) != data["sha256"]:
        raise RuntimeError("fixed evaluation validation bytes drift")
    tokens = np.memmap(validation_path, dtype=np.dtype(data["dtype"]), mode="r")
    length = int(authorization["sample_length_tokens"])
    examples = [
        np.asarray(tokens[int(start):int(start) + length], dtype=np.int64)
        for start in authorization["evaluation_offsets"]
    ]
    batch_size = int(data["batch_size"])
    if len(examples) % batch_size:
        raise RuntimeError("fixed evaluation example count is not batch-aligned")
    batches = [
        torch.from_numpy(np.stack(examples[index:index + batch_size]))
        for index in range(0, len(examples), batch_size)
    ]
    objective = W560ForwardOnlyObjective(loaded_parent, batches, device)
    groups = list(protocol["search"]["independent_branches"])
    arm_specs = {"immutable_parent": None}
    for group in groups:
        validation = validate_branch_artifact_chain(
            protocol, search_root / group, group, include_final_delta_bytes=True
        )
        accepts = int(validation["accepts"])
        bound_finals = authorization.get("bound_branch_finals", {})
        canonical_group = protocol["canonical_weight_address_binding"]["internal_to_canonical_group"][group]
        if canonical_group not in bound_finals:
            raise RuntimeError(f"fixed evaluation lacks terminal-bound branch final: {group}")
        if (validation["final_materialized_child_sha256"]
                != bound_finals[canonical_group]["final_checkpoint_sha256"]):
            raise RuntimeError(f"fixed evaluation child SHA differs from terminal bundle: {group}")
        arm_specs[group] = {
            "accepts": accepts,
            "expected_child_sha256": validation["final_materialized_child_sha256"],
            "child_path": (
                None if accepts == 0 else
                search_root / group / f"materialized_child_{accepts:02d}.pt"
            ),
        }
    results = {}
    total_forward_calls = 0
    parent_state = {**objective.base_parameters, **objective.buffers}
    for arm, spec in arm_specs.items():
        if spec is None or spec["accepts"] == 0:
            state = parent_state
            evaluated_state_sha = protocol["parent"]["checkpoint_sha256"]
        else:
            payload = _load_materialized_child_same_fd(
                spec["child_path"], spec["expected_child_sha256"]
            )
            if payload.get("schema") != "hfw_post_training_materialized_child_v1":
                raise RuntimeError(f"fixed evaluation child payload schema drift: {arm}")
            if payload.get("branch_group") != arm:
                raise RuntimeError(f"fixed evaluation child payload group drift: {arm}")
            state = {
                name: value.to(device)
                for name, value in payload["model"].items()
            }
            evaluated_state_sha = spec["expected_child_sha256"]
        vectors = [objective.loss_state_vector(state, index) for index in range(len(batches))]
        duplicate_vectors = [
            objective.loss_state_vector(state, index) for index in range(len(batches))
        ]
        total_forward_calls += 2 * len(batches)
        flat_losses = [value for vector in vectors for value in vector]
        duplicate_losses = [value for vector in duplicate_vectors for value in vector]
        if not all(math.isfinite(value) for value in (*flat_losses, *duplicate_losses)):
            raise RuntimeError("fixed evaluation contains nonfinite evidence")
        maximum_duplicate_delta = max(
            abs(left - right) for left, right in zip(flat_losses, duplicate_losses)
        )
        if maximum_duplicate_delta > float(protocol["search"]["acceptance"]["duplicate_forward_tolerance"]):
            raise RuntimeError("fixed evaluation duplicate-forward drift")
        results[arm] = {
            "per_sequence_losses": flat_losses,
            "duplicate_per_sequence_losses": duplicate_losses,
            "maximum_duplicate_absolute_delta": maximum_duplicate_delta,
            "mean_per_token_cross_entropy": sum(flat_losses) / len(flat_losses),
            "fixed_examples": len(flat_losses),
            "evaluated_exact_state_artifact_sha256": evaluated_state_sha,
            "evaluation_source": (
                "immutable_parent" if spec is None or spec["accepts"] == 0
                else "same_fd_loaded_terminal_bound_materialized_child"
            ),
        }
        if spec is not None and spec["accepts"]:
            del state, payload
            if device.type == "cuda":
                torch.cuda.empty_cache()
    parent_losses = results["immutable_parent"]["per_sequence_losses"]
    bootstrap_samples = int(protocol["search"]["acceptance"]["bootstrap_samples"])
    family_alpha = float(protocol["search"]["acceptance"]["familywise_alpha"])
    paired = {}
    for arm in groups:
        improvements = [
            parent - child for parent, child in zip(
                parent_losses, results[arm]["per_sequence_losses"]
            )
        ]
        generator = torch.Generator(device="cpu").manual_seed(
            int(protocol["search"]["acceptance"]["bootstrap_seed"])
            + groups.index(arm)
        )
        values = torch.tensor(improvements, dtype=torch.float64)
        indices = torch.randint(
            0, len(values), (bootstrap_samples, len(values)), generator=generator
        )
        means = values[indices].mean(dim=1).sort().values
        tail = family_alpha / (2 * len(groups))
        lower_index = max(0, min(bootstrap_samples - 1, math.floor(tail * bootstrap_samples)))
        upper_index = max(0, min(bootstrap_samples - 1, math.ceil((1 - tail) * bootstrap_samples) - 1))
        paired[arm] = {
            "per_sequence_parent_minus_branch_ce": improvements,
            "mean_parent_minus_branch_ce": sum(improvements) / len(improvements),
            "bonferroni_familywise_alpha": family_alpha,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_mean_ci": [float(means[lower_index]), float(means[upper_index])],
        }
    return {
        "schema": "hfw_w560_parent_plus_exact_four_report_only_fixed_eval_v1",
        "status": "PASS_FIXED_EVAL_REPORTED_NO_SELECTION",
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "results": results,
        "paired_branch_vs_parent": paired,
        "arms_evaluated": ["immutable_parent", *groups],
        "branch_selected_or_combined": False,
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "forward_calls": total_forward_calls,
        "unique_fixed_eval_token_positions": len(examples) * length,
        "total_evaluated_target_tokens": total_forward_calls * batch_size * (length - 1),
        "estimated_macs_per_target_token": (
            int(loaded_parent.model.estimated_macs_per_token(int(data["context"])))
            if hasattr(loaded_parent.model, "estimated_macs_per_token") else None
        ),
        "estimated_total_macs": (
            int(loaded_parent.model.estimated_macs_per_token(int(data["context"])))
            * total_forward_calls * batch_size * (length - 1)
            if hasattr(loaded_parent.model, "estimated_macs_per_token") else None
        ),
        "wall_seconds": time.monotonic() - started,
    }


def write_create_only_result_registry(output_root: Path, result: Mapping[str, object]) -> Mapping[str, object]:
    encoded = canonical_json_bytes(result)
    digest = hashlib.sha256(encoded).hexdigest()
    path = output_root / "final_result_registry" / f"{digest}.result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "sha256": digest, "size_bytes": len(encoded), "create_only": True}
