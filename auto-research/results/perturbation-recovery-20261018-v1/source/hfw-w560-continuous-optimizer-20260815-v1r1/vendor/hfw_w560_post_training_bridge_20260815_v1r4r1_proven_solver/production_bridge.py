from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .manifest import (
    classify_parameter, derive_group_manifest, isolated_predictive_models,
    verify_source_inventory,
)
from .parent_gate import validate_immutable_parent


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def model_state_digest(mapping: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in mapping.items():
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


@dataclass
class LoadedW560Parent:
    model: torch.nn.Module
    groups: Mapping[str, tuple[str, ...]]
    parent_gate: Mapping[str, object]
    group_manifest: Mapping[str, object]
    checkpoint_sha256: str


def _load_checkpoint_nofollow(path: Path, expected_sha256: str) -> Mapping[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("checkpoint is not a regular same-FD file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("checkpoint same-FD pre-load SHA drift")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("checkpoint same-FD post-load SHA drift")
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError("checkpoint changed during load")
    finally:
        os.close(descriptor)
    if set(payload) != {"step", "model", "optimizer", "scheduler", "embedding_projection", "checkpoint_identity"}:
        raise RuntimeError("checkpoint payload schema drift")
    return payload


def load_completed_w560_parent(
    protocol: Mapping[str, object], source_root: Path, device: torch.device
) -> LoadedW560Parent:
    gate = validate_immutable_parent(protocol)
    verify_source_inventory(protocol, source_root)
    manifest = derive_group_manifest(protocol, source_root)
    checkpoint_path = Path(protocol["parent"]["checkpoint_path"])
    payload = _load_checkpoint_nofollow(checkpoint_path, protocol["parent"]["checkpoint_sha256"])
    if int(payload["step"]) != int(protocol["parent"]["completed_step"]):
        raise RuntimeError("checkpoint payload step drift")
    state = payload["model"]
    if model_state_digest(state) != protocol["parent"]["model_state_digest"]:
        raise RuntimeError("checkpoint model state digest drift")
    del payload
    with isolated_predictive_models(source_root) as module:
        config = json.loads((source_root / "config/bottleneck_hfw_1k_100m.json").read_text())
        model_config = config["models"][protocol["model"]["kind"]]
        with torch.device("meta"):
            model = module.build_student(
                protocol["model"]["kind"], int(config["vocab_size"]),
                int(protocol["model"]["width"]), int(config["model_max_context"]),
                model_config,
            )
        model.load_state_dict(state, strict=True, assign=True)
        del state
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("no-autograd gate failed")
        groups = {name: [] for name in protocol["search"]["independent_branches"]}
        for name, _ in model.named_parameters():
            group = classify_parameter(name)
            if group in groups:
                groups[group].append(name)
        return LoadedW560Parent(
            model=model,
            groups={name: tuple(values) for name, values in groups.items()},
            parent_gate=gate,
            group_manifest=manifest,
            checkpoint_sha256=protocol["parent"]["checkpoint_sha256"],
        )


class W560ForwardOnlyObjective:
    def __init__(self, parent: LoadedW560Parent, calibration_batches: Sequence[torch.Tensor], device: torch.device):
        self.parent = parent
        self.device = device
        self.calibration = tuple(batch.detach().cpu().clone() for batch in calibration_batches)
        self.base_parameters = {name: value.detach() for name, value in parent.model.named_parameters()}
        self.buffers = {name: value.detach() for name, value in parent.model.named_buffers()}

    def flat_group(self, group: str) -> torch.Tensor:
        return torch.cat([self.base_parameters[name].reshape(-1) for name in self.parent.groups[group]]).detach()

    def group_mapping(self, group: str, flat: torch.Tensor) -> Mapping[str, torch.Tensor]:
        mapping = dict(self.base_parameters)
        cursor = 0
        for name in self.parent.groups[group]:
            reference = self.base_parameters[name]
            count = reference.numel()
            mapping[name] = flat[cursor:cursor + count].view_as(reference)
            cursor += count
        if cursor != flat.numel():
            raise RuntimeError("candidate group vector length drift")
        return mapping

    def loss_state_vector(
        self, state: Mapping[str, torch.Tensor], batch_index: int
    ) -> tuple[float, ...]:
        expected = set(self.base_parameters) | set(self.buffers)
        if set(state) != expected:
            raise RuntimeError("candidate full state_dict inventory drift")
        if any(value.requires_grad for value in state.values()):
            raise RuntimeError("autograd is forbidden")
        parameters = {}
        buffers = {}
        for name, reference in self.base_parameters.items():
            value = state[name]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise RuntimeError(f"candidate parameter shape/dtype drift: {name}")
            parameters[name] = value.to(self.device)
        for name, reference in self.buffers.items():
            value = state[name]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise RuntimeError(f"candidate buffer shape/dtype drift: {name}")
            buffers[name] = value.to(self.device)
        raw = self.calibration[batch_index].to(self.device)
        inputs, targets = raw[:, :-1], raw[:, 1:]
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                output = torch.func.functional_call(
                    self.parent.model,
                    (parameters, buffers),
                    (inputs,), strict=True,
                )
            per_token = F.cross_entropy(
                output.logits.float().reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1), reduction="none",
            ).view(targets.shape)
            values = per_token.mean(dim=1)
            if not torch.isfinite(values).all():
                raise RuntimeError("nonfinite per-sequence calibration loss")
            return tuple(float(value) for value in values.cpu().tolist())

    def loss_vector(self, group: str, flat: torch.Tensor, batch_index: int) -> tuple[float, ...]:
        if flat.requires_grad or any(value.requires_grad for value in self.base_parameters.values()):
            raise RuntimeError("autograd is forbidden")
        state = dict(self.group_mapping(group, flat))
        state.update(self.buffers)
        return self.loss_state_vector(state, batch_index)

    def loss(self, group: str, flat: torch.Tensor, batch_index: int) -> float:
        try:
            values = self.loss_vector(group, flat, batch_index)
        except RuntimeError as error:
            if str(error) == "nonfinite per-sequence calibration loss":
                return float("inf")
            raise
        return sum(values) / len(values)
