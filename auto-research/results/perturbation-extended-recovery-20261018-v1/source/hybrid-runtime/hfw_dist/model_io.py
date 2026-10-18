from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping

import torch

from .common import hash_file, read_json


PROMOTED_SCHEMA = "hfw_w560_continuous_promoted_checkpoint_v1"
PROMOTED_KEYS = {
    "schema", "origin_checkpoint_sha256", "parent_checkpoint_sha256", "generation",
    "accepted_proposal_sha256", "selected_candidate_order",
    "selected_candidate_delta_sha256", "solver_cumulative_delta_sha256",
    "replacement_group_flat_sha256", "group_replacement_materialization_source",
    "solver_cumulative_delta_used_to_reconstruct_group", "model_state_digest",
    "optimizer_present", "scheduler_present", "training_resume_authorized", "model",
}


def load_checkpoint_same_fd(path: Path, expected_sha256: str) -> Mapping[str, object]:
    """Hash and torch.load one descriptor without buffering another checkpoint copy."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("checkpoint is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("checkpoint SHA drift")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError("checkpoint changed during load")
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint payload is not a mapping")
    return payload


def validate_promoted_payload(
    payload: Mapping[str, object], expected_model_digest: str,
    bridge_module,
) -> None:
    if set(payload) != PROMOTED_KEYS or payload.get("schema") != PROMOTED_SCHEMA:
        raise RuntimeError("promoted checkpoint schema/key inventory drift")
    if payload.get("optimizer_present") is not False:
        raise RuntimeError("checkpoint contains optimizer state")
    if payload.get("scheduler_present") is not False:
        raise RuntimeError("checkpoint contains scheduler state")
    if payload.get("training_resume_authorized") is not False:
        raise RuntimeError("checkpoint authorizes training resume")
    if payload.get("group_replacement_materialization_source") != "exact_candidate_group_flat_bytes":
        raise RuntimeError("checkpoint group replacement source drift")
    if payload.get("solver_cumulative_delta_used_to_reconstruct_group") is not False:
        raise RuntimeError("checkpoint reconstructed group from lossy delta")
    observed = bridge_module.model_state_digest(payload["model"])
    if observed != expected_model_digest or observed != payload.get("model_state_digest"):
        raise RuntimeError("checkpoint model-state digest drift")


def build_loaded_parent(
    snapshot_root: Path, checkpoint_path: Path, checkpoint_sha256: str,
    model_state_digest: str, device: torch.device, bridge_module, manifest_module,
):
    payload = load_checkpoint_same_fd(checkpoint_path, checkpoint_sha256)
    validate_promoted_payload(payload, model_state_digest, bridge_module)
    state = payload["model"]
    source_root = Path(snapshot_root) / "source"
    source_config, _ = read_json(source_root / "config/bottleneck_hfw_1k_100m.json")
    protocol, _ = read_json(
        Path(snapshot_root)
        / "vendor/hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver"
        / "config/production_protocol.json"
    )
    protocol = dict(protocol)
    protocol["frozen_source"] = dict(protocol["frozen_source"])
    protocol["frozen_source"]["source_root"] = str(source_root)
    manifest_module.verify_source_inventory(protocol, source_root)
    group_manifest = manifest_module.derive_group_manifest(protocol, source_root)
    with manifest_module.isolated_predictive_models(source_root) as module:
        model_config = source_config["models"][protocol["model"]["kind"]]
        with torch.device("meta"):
            model = module.build_student(
                protocol["model"]["kind"], int(source_config["vocab_size"]),
                int(protocol["model"]["width"]),
                int(source_config["model_max_context"]), model_config,
            )
        model.load_state_dict(state, strict=True, assign=True)
    del state
    del payload
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("no-autograd model gate failed")
    groups = {name: [] for name in protocol["search"]["independent_branches"]}
    for name, _ in model.named_parameters():
        group = manifest_module.classify_parameter(name)
        if group in groups:
            groups[group].append(name)
    loaded = bridge_module.LoadedW560Parent(
        model=model,
        groups={name: tuple(values) for name, values in groups.items()},
        parent_gate={"status": "PASS_PROMOTED_CHECKPOINT_EXACT_SHA_AND_MODEL_DIGEST"},
        group_manifest=group_manifest,
        checkpoint_sha256=checkpoint_sha256,
    )
    return loaded, protocol


def model_state_from_parent(parent) -> dict[str, torch.Tensor]:
    result = {name: value.detach() for name, value in parent.model.named_parameters()}
    for name, value in parent.model.named_buffers():
        if name in result:
            raise RuntimeError("parameter/buffer collision")
        result[name] = value.detach()
    return result


def replace_group_in_state(parent, group: str, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    state = model_state_from_parent(parent)
    cursor = 0
    for name in parent.groups[group]:
        reference = state[name]
        count = reference.numel()
        state[name] = flat[cursor:cursor + count].view_as(reference)
        cursor += count
    if cursor != flat.numel():
        raise RuntimeError("candidate flat group length drift")
    return state
