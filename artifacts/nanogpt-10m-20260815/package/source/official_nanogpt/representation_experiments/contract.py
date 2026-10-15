"""Fail-closed validation for the exact historical nanoGPT comparison."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from representation_experiments.predictive_models import resolve_budget


HISTORICAL_CONFIG_SHA256 = (
    "f47d8cabafa5f16ff32c1169fba2064bf8124907cfc99e7d8374212cc25c58fa")
UPSTREAM_MODEL_SHA256 = (
    "7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c")
UPSTREAM_LICENSE_SHA256 = (
    "a59ec5cdb1c1e447e5266a014b3e7ded511f8d6e5a08931e931c87490ff821fe")
UPSTREAM_ADAPTER_SHA256 = (
    "1adfffbd843a55f00491b50cb46bc6c5d99041f54d9f48251ce3e79ad672a870")
UPSTREAM_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
ARM_NAME = "official_nanogpt_aug8_10m"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required regular file is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _without(mapping: dict, names: set[str]) -> dict:
    return {key: value for key, value in mapping.items() if key not in names}


def validate_static_contract(package_root: Path) -> dict:
    package_root = package_root.expanduser().resolve()
    config_path = package_root / "config" / "official_nanogpt_aug8_10m.json"
    historical_path = (
        package_root / "bindings" / "qwen06b_matched_hierarchy_function_v1.json")
    config = load_json(config_path)
    historical = load_json(historical_path)
    if sha256_file(historical_path) != HISTORICAL_CONFIG_SHA256:
        raise RuntimeError("historical Aug-8 config binding drifted")

    exact_top_level = (
        "teacher", "vocab_size", "data", "token_dtype",
        "student_parameter_budget", "model_max_context", "embedding_projection",
        "evaluation", "benchmark", "safety")
    for key in exact_top_level:
        if config.get(key) != historical.get(key):
            raise RuntimeError(f"comparator differs from historical contract: {key}")

    training = dict(config.get("training", {}))
    historical_training = dict(historical.get("training", {}))
    permitted_explicit_defaults = {
        "expected_steps", "expected_executed_tokens", "sampling",
        "kl_implementation"}
    if _without(training, permitted_explicit_defaults) != historical_training:
        raise RuntimeError("training contract differs from historical Aug-8 training")
    if training.get("sampling") != "restart_stable_random":
        raise RuntimeError("historical restart-stable sample order is not explicit")
    if training.get("kl_implementation") != "sequential":
        raise RuntimeError("historical sequential multi-temperature KL is not explicit")
    if any(name in training for name in (
            "representation_matching", "future_state_prediction")):
        raise RuntimeError("extra representation objectives are forbidden")
    if config.get("required_world_size") != 1:
        raise RuntimeError("historical comparator must use one process and one GPU")
    if set(config.get("arms", {})) != {ARM_NAME}:
        raise RuntimeError("comparator must contain exactly one official nanoGPT arm")
    arm = config["arms"][ARM_NAME]
    if arm != {
            "architecture": "nanogpt",
            "embedding_initialization": "qwen_pca",
            "branching": False}:
        raise RuntimeError("official nanoGPT arm declaration drifted")

    model = config.get("models", {}).get("nanogpt")
    if not isinstance(model, dict):
        raise RuntimeError("official nanoGPT model config is missing")
    if model.get("dropout") != 0.0 or model.get("bias") is not True:
        raise RuntimeError("official nanoGPT defaults require dropout=0 and bias=true")
    if model.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("official nanoGPT upstream commit drifted")
    search = model.get("geometry_search")
    required_search = {
        "target_stored_parameters": 119694800,
        "maximum_stored_parameters": 120000000,
        "head_dimension": 64,
        "minimum_heads": 1,
        "maximum_heads": 16,
        "minimum_layers": 1,
        "maximum_layers": 24,
        "tie_break": [
            "absolute_parameter_distance", "larger_parameter_count",
            "shallower_layer_count", "smaller_head_count"],
    }
    if search != required_search:
        raise RuntimeError("official nanoGPT exhaustive geometry search drifted")
    teacher_identity = config.get("teacher_identity", {})
    if teacher_identity != {
            "model.safetensors": {
                "sha256": (
                    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba")}}:
        raise RuntimeError("teacher weights identity drifted")

    tokens_per_step = int(training["context"]) * int(training["batch_size"])
    steps = math.ceil(int(training["max_tokens"]) / tokens_per_step)
    executed_tokens = steps * tokens_per_step
    if steps != int(training["expected_steps"]) or steps != 4883:
        raise RuntimeError("historical optimizer-step count drifted")
    if (executed_tokens != int(training["expected_executed_tokens"])
            or executed_tokens != 10000384):
        raise RuntimeError("historical executed-token count drifted")

    resolved = resolve_budget(
        "nanogpt", int(config["vocab_size"]), int(config["model_max_context"]),
        model, int(config["student_parameter_budget"]))
    expected_model = config.get("resolved_model_expectation")
    if resolved != expected_model:
        raise RuntimeError(
            f"resolved official nanoGPT model drifted: {resolved!r}")

    bound_files = {
        "third_party/nanogpt/model.py": UPSTREAM_MODEL_SHA256,
        "third_party/nanogpt/LICENSE": UPSTREAM_LICENSE_SHA256,
        "representation_experiments/nanogpt_control.py": UPSTREAM_ADAPTER_SHA256,
    }
    for relative, expected_sha256 in bound_files.items():
        if sha256_file(package_root / relative) != expected_sha256:
            raise RuntimeError(f"pinned upstream artifact drifted: {relative}")
    return {
        "status": "PASS_STATIC_CONTRACT",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "historical_config_sha256": HISTORICAL_CONFIG_SHA256,
        "upstream_commit": UPSTREAM_COMMIT,
        "resolved_model": resolved,
        "steps": steps,
        "executed_tokens": executed_tokens,
        "bound_files": bound_files,
        "config": config,
    }


def validate_runtime_data(package_root: Path, runtime_root: Path,
                          teacher_snapshot_root: Path) -> dict:
    static = validate_static_contract(package_root)
    runtime_root = runtime_root.expanduser().resolve()
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise RuntimeError("runtime root must be a regular existing directory")
    data_root = runtime_root / static["config"]["data"]
    records = {}
    for name, expected in static["config"]["data_identity"].items():
        path = data_root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"data artifact must be a regular file: {path}")
        before = path.stat()
        if before.st_size != int(expected["bytes"]):
            raise RuntimeError(f"data size drifted: {name}")
        observed_sha256 = sha256_file(path)
        after = path.stat()
        identity_before = (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise RuntimeError(f"data changed while hashing: {name}")
        if observed_sha256 != expected["sha256"]:
            raise RuntimeError(f"data SHA-256 drifted: {name}")
        records[name] = {
            "path": str(path),
            "bytes": before.st_size,
            "sha256": observed_sha256,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
        }
    teacher_snapshot_root = teacher_snapshot_root.expanduser().resolve()
    if (teacher_snapshot_root.is_symlink()
            or not teacher_snapshot_root.is_dir()
            or teacher_snapshot_root.name != static["config"]["teacher_revision"]):
        raise RuntimeError("teacher snapshot root is not the exact bound revision")
    teacher_path = teacher_snapshot_root / "model.safetensors"
    if teacher_path.is_symlink() or not teacher_path.is_file():
        raise RuntimeError("teacher model.safetensors must be a regular file")
    teacher_before = teacher_path.stat()
    teacher_sha256 = sha256_file(teacher_path)
    teacher_after = teacher_path.stat()
    if ((teacher_before.st_dev, teacher_before.st_ino, teacher_before.st_size,
         teacher_before.st_mtime_ns)
            != (teacher_after.st_dev, teacher_after.st_ino, teacher_after.st_size,
                teacher_after.st_mtime_ns)):
        raise RuntimeError("teacher weights changed while hashing")
    expected_teacher_sha256 = static["config"]["teacher_identity"][
        "model.safetensors"]["sha256"]
    if teacher_sha256 != expected_teacher_sha256:
        raise RuntimeError("teacher model.safetensors SHA-256 drifted")
    return {
        "status": "PASS_RUNTIME_DATA",
        "runtime_root": str(runtime_root),
        "data": records,
        "teacher_snapshot": {
            "path": str(teacher_snapshot_root),
            "weights_path": str(teacher_path),
            "weights_sha256": teacher_sha256,
            "bytes": teacher_before.st_size,
            "device": teacher_before.st_dev,
            "inode": teacher_before.st_ino,
            "mtime_ns": teacher_before.st_mtime_ns,
        },
        "static_contract": _without(static, {"config"}),
    }


def require_single_process_environment() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (world_size, rank, local_rank) != (1, 0, 0):
        raise RuntimeError("exact Aug-8 comparator forbids distributed execution")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if len(devices) != 1:
            raise RuntimeError("exact comparator requires exactly one visible GPU")
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "cuda_visible_devices": visible,
    }
