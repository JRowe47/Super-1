from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

import torch

from .common import hash_file_same_fd
from .relocation import derive_relocated_protocol, load_config, vendor_modules


def _torch_load_same_fd(path: Path, expected_sha256: str) -> Mapping[str, object]:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("checkpoint is not a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("checkpoint pre-load SHA drift")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("checkpoint post-load SHA drift")
        if (before.st_dev, before.st_ino, before.st_size, before.st_nlink, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_nlink, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RuntimeError("checkpoint changed during load")
        return payload
    finally:
        os.close(descriptor)


def load_head_model(
    config_path: Path, head: Mapping[str, object], device: torch.device,
    *, config: Mapping[str, object] | None = None,
):
    config = load_config(config_path) if config is None else config
    derived, relocation_receipt = derive_relocated_protocol(config_path, config=config)
    _, branch, bridge, _, _ = vendor_modules(config)
    source_root = Path(config["source"]["root"])
    loaded = bridge.load_completed_w560_parent(derived, source_root, device)
    checkpoint = head["checkpoint"]
    if checkpoint["sha256"] == config["origin"]["checkpoint_sha256"]:
        if Path(checkpoint["path"]).resolve() != Path(config["origin"]["relocated_checkpoint_path"]).resolve():
            raise RuntimeError("generation-zero checkpoint path drift")
    else:
        payload = _torch_load_same_fd(Path(checkpoint["path"]), checkpoint["sha256"])
        if payload.get("schema") != "hfw_w560_continuous_promoted_checkpoint_v1":
            raise RuntimeError("promoted checkpoint schema drift")
        if payload.get("origin_checkpoint_sha256") != config["origin"]["checkpoint_sha256"]:
            raise RuntimeError("promoted checkpoint origin drift")
        if payload.get("parent_checkpoint_sha256") != head["parent_checkpoint_sha256"]:
            raise RuntimeError("promoted checkpoint immediate-parent drift")
        if int(payload.get("generation", -1)) != int(head["generation"]):
            raise RuntimeError("promoted checkpoint generation drift")
        if payload.get("accepted_proposal_sha256") != head["accepted_proposal_sha256"]:
            raise RuntimeError("promoted checkpoint proposal lineage drift")
        if int(payload.get("selected_candidate_order", -1)) != int(head["selected_candidate_order"]):
            raise RuntimeError("promoted checkpoint selected candidate drift")
        if payload.get("selected_candidate_delta_sha256") != head["delta_sha256"]:
            raise RuntimeError("promoted checkpoint selected delta drift")
        if payload.get("solver_cumulative_delta_sha256") != head["solver_cumulative_delta_sha256"]:
            raise RuntimeError("promoted checkpoint solver cumulative delta drift")
        if payload.get("replacement_group_flat_sha256") != head["replacement_group_flat_sha256"]:
            raise RuntimeError("promoted checkpoint exact replacement group-flat drift")
        if payload.get("group_replacement_materialization_source") != "exact_candidate_group_flat_bytes":
            raise RuntimeError("promoted checkpoint replacement materialization source drift")
        if payload.get("solver_cumulative_delta_used_to_reconstruct_group") is not False:
            raise RuntimeError("promoted checkpoint reconstructed group from noninvertible delta")
        if head.get("group_replacement_materialization_source") != "exact_candidate_group_flat_bytes":
            raise RuntimeError("promoted head replacement materialization source drift")
        for label, head_key, expected_sha in (
            (
                "solver cumulative delta", "solver_cumulative_delta",
                head["solver_cumulative_delta_sha256"],
            ),
            (
                "exact replacement group-flat", "replacement_group_flat",
                head["replacement_group_flat_sha256"],
            ),
        ):
            artifact = head.get(head_key, {})
            identity = hash_file_same_fd(Path(artifact["path"]))
            if identity["sha256"] != expected_sha or identity["sha256"] != artifact["sha256"]:
                raise RuntimeError(f"promoted checkpoint {label} artifact SHA drift")
            if int(identity["size_bytes"]) != int(artifact["size_bytes"]):
                raise RuntimeError(f"promoted checkpoint {label} artifact size drift")
        if payload.get("model_state_digest") != head["model_state_digest"]:
            raise RuntimeError("promoted checkpoint/head model digest drift")
        if payload.get("optimizer_present") is not False or payload.get("scheduler_present") is not False:
            raise RuntimeError("promoted checkpoint contains training state")
        if payload.get("training_resume_authorized") is not False:
            raise RuntimeError("promoted checkpoint authorizes training resume")
        if bridge.model_state_digest(payload["model"]) != payload["model_state_digest"]:
            raise RuntimeError("promoted checkpoint model state digest drift")
        replacement_from_checkpoint = torch.cat([
            payload["model"][name].reshape(-1) for name in loaded.groups[head["group"]]
        ])
        if hashlib.sha256(branch._tensor_bytes(replacement_from_checkpoint)).hexdigest() != head["replacement_group_flat_sha256"]:
            raise RuntimeError("promoted checkpoint model does not contain exact replacement group-flat")
        loaded.model.load_state_dict(payload["model"], strict=True)
        loaded.checkpoint_sha256 = checkpoint["sha256"]
    return loaded, relocation_receipt


def state_for_objective(objective) -> dict[str, torch.Tensor]:
    return {**objective.base_parameters, **objective.buffers}


def replace_group(
    objective, group: str, flat: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = dict(objective.group_mapping(group, flat))
    state.update(objective.buffers)
    return state


def flatten_losses(objective, state: Mapping[str, torch.Tensor], batch_count: int) -> list[float]:
    result = []
    for batch_index in range(batch_count):
        result.extend(float(value) for value in objective.loss_state_vector(state, batch_index))
    return result
