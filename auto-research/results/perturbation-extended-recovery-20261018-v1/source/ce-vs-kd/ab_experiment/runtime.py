from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    canonical_json_bytes, exact_keys, file_identity, read_bytes_same_fd,
    read_json_same_fd, sha256_bytes, tensor_f32le_bytes, write_exclusive,
    write_exclusive_json,
)


TEACHER_VOCAB_SIZE = 151936
SEQUENCE_TOKENS = 513
CONTEXT_TOKENS = 512
BF16_LOGIT_SHARD_BYTES = CONTEXT_TOKENS * TEACHER_VOCAB_SIZE * 2


def load_config(path: Path) -> Mapping[str, object]:
    config = read_json_same_fd(path)
    if config.get("schema") != "hfw_w560_ce_vs_kd_matched_ab_config_v1":
        raise RuntimeError("A/B config schema drift")
    return config


def assert_no_conflicting_mps_solver(config: Mapping[str, object]) -> None:
    conflict = config["runtime"]["conflicting_mps_process_substring"]
    output = subprocess.check_output(["/bin/ps", "-axo", "pid=,command="], text=True)
    own_pid = os.getpid()
    matches = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != own_pid and conflict in command:
            matches.append({"pid": pid, "command": command})
    if matches:
        raise RuntimeError(f"conflicting active MPS solver blocks model work: {matches}")


def require_bound_output(config: Mapping[str, object], actual: Path) -> None:
    output = config["phase_output"]
    expected = output.get("result_root", output.get("result_path"))
    if expected is None or Path(actual).resolve() != Path(expected).resolve():
        raise RuntimeError("phase output path is not the unique preregistered destination")


def claim_phase(config: Mapping[str, object], config_path: Path) -> Mapping[str, object]:
    claim_path = Path(config["phase_output"]["claim_path"])
    output = config["phase_output"].get(
        "result_root", config["phase_output"].get("result_path"),
    )
    return write_exclusive_json(claim_path, {
        "schema": "hfw_w560_ce_vs_kd_unique_phase_claim_v1",
        "status": "CLAIMED_NO_RETRY_OR_ALTERNATE_OUTPUT_WITHIN_THIS_A_B_PACKAGE",
        "phase": config["phase"],
        "config": file_identity(config_path),
        "bound_output": output,
        "failure_requires_fresh_whole_a_b_package": True,
    })


def validate_input_closure(config: Mapping[str, object]) -> Mapping[str, object]:
    observed = {}
    for label, record in config["inputs"].items():
        if label == "teacher_outputs":
            continue
        if isinstance(record, dict) and {"path", "sha256", "size_bytes"} <= set(record):
            identity = file_identity(Path(record["path"]), record["sha256"])
            if identity["size_bytes"] != int(record["size_bytes"]):
                raise RuntimeError(f"input size drift: {label}")
            observed[label] = identity
            if label == "environment_manifest":
                environment = read_json_same_fd(Path(record["path"]), record["sha256"])
                if environment.get("status") != "PASS_EXACT_LOCAL_WHEEL_AND_CRITICAL_SOURCE_CLOSURE":
                    raise RuntimeError("isolated environment manifest status drift")
                for category in ("wheels", "critical_installed_sources"):
                    for name, dependency in environment[category].items():
                        dependency_identity = file_identity(
                            Path(dependency["path"]), dependency["sha256"],
                        )
                        if int(dependency_identity["size_bytes"]) != int(dependency["size_bytes"]):
                            raise RuntimeError(f"environment dependency size drift: {name}")
                        observed[f"environment:{category}:{name}"] = dependency_identity
            if label == "runtime_manifest":
                manifest = read_json_same_fd(Path(record["path"]), record["sha256"])
                if manifest.get("schema") != "hfw_w560_ce_vs_kd_runtime_manifest_v1":
                    raise RuntimeError("runtime manifest schema drift")
                runtime_root = Path(config["package_root"]) / "runtime"
                reduced = {}
                for relative, runtime_record in manifest["files"].items():
                    path = runtime_root / relative
                    identity = file_identity(path, runtime_record["sha256"])
                    if int(identity["size_bytes"]) != int(runtime_record["size_bytes"]):
                        raise RuntimeError(f"runtime file size drift: {relative}")
                    reduced[relative] = {
                        "sha256": identity["sha256"],
                        "size_bytes": identity["size_bytes"],
                    }
                    observed[f"runtime:{relative}"] = identity
                inventory_identity = sha256_bytes(json.dumps(
                    reduced, sort_keys=True, separators=(",", ":"),
                ).encode())
                if inventory_identity != manifest["inventory_identity_sha256"]:
                    raise RuntimeError("runtime manifest inventory identity drift")
    source_root = Path(config["source"]["root"])
    for relative, expected_sha256 in config["source"]["files"].items():
        identity = file_identity(source_root / relative, expected_sha256)
        observed[f"source:{relative}"] = identity
    for relative, record in config["vendor"]["files"].items():
        identity = file_identity(Path(record["path"]), record["sha256"])
        if int(identity["size_bytes"]) != int(record["size_bytes"]):
            raise RuntimeError(f"vendor file size drift: {relative}")
        observed[f"vendor:{relative}"] = identity
    teacher = config.get("teacher")
    if teacher is not None:
        for label, record in teacher["snapshot_files"].items():
            identity = file_identity(Path(record["path"]), record["sha256"])
            if identity["size_bytes"] != int(record["size_bytes"]):
                raise RuntimeError(f"teacher file size drift: {label}")
            observed[f"teacher:{label}"] = identity
    return observed


def load_bank(record: Mapping[str, object]) -> tuple[torch.Tensor, Mapping[str, object]]:
    body = read_bytes_same_fd(Path(record["path"]), record["sha256"])
    if len(body) != int(record["size_bytes"]):
        raise RuntimeError("bank size drift")
    shape = tuple(int(value) for value in record["shape"])
    values = np.frombuffer(body, dtype="<u4")
    if values.size != math.prod(shape):
        raise RuntimeError("bank element count drift")
    maximum = int(values.max())
    if maximum > int(record["maximum_token_id"]):
        raise RuntimeError("bank contains a token outside teacher tokenizer IDs")
    return torch.from_numpy(values.copy()).view(*shape), {
        "sha256": sha256_bytes(body), "shape": list(shape),
        "maximum_token_id": maximum,
    }


def _torch_load_same_fd(path: Path, expected_sha256: str):
    body = read_bytes_same_fd(path, expected_sha256)
    import io
    return torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)


def load_promoted_w560(config: Mapping[str, object], device: torch.device):
    from vendor.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver import (
        manifest, production_bridge,
    )
    checkpoint = config["inputs"]["starting_checkpoint"]
    payload = _torch_load_same_fd(Path(checkpoint["path"]), checkpoint["sha256"])
    expected_keys = {
        "schema", "origin_checkpoint_sha256", "parent_checkpoint_sha256", "generation",
        "accepted_proposal_sha256", "selected_candidate_order",
        "selected_candidate_delta_sha256", "solver_cumulative_delta_sha256",
        "replacement_group_flat_sha256", "group_replacement_materialization_source",
        "solver_cumulative_delta_used_to_reconstruct_group", "model_state_digest",
        "optimizer_present", "scheduler_present", "training_resume_authorized", "model",
    }
    exact_keys(payload, expected_keys, "starting promoted checkpoint")
    if payload["schema"] != "hfw_w560_continuous_promoted_checkpoint_v1":
        raise RuntimeError("starting checkpoint format drift")
    if payload["model_state_digest"] != checkpoint["model_state_digest"]:
        raise RuntimeError("starting model digest config drift")
    if production_bridge.model_state_digest(payload["model"]) != payload["model_state_digest"]:
        raise RuntimeError("starting model state digest mismatch")
    if payload["optimizer_present"] is not False or payload["scheduler_present"] is not False:
        raise RuntimeError("starting checkpoint contains training state")
    if payload["training_resume_authorized"] is not False:
        raise RuntimeError("starting checkpoint authorizes training resume")

    source_root = Path(config["source"]["root"])
    source_config = json.loads(
        read_bytes_same_fd(
            source_root / "config/bottleneck_hfw_1k_100m.json",
            config["source"]["files"]["config/bottleneck_hfw_1k_100m.json"],
        )
    )
    frozen_protocol = read_json_same_fd(
        Path(config["vendor"]["production_protocol_path"]),
        config["vendor"]["production_protocol_sha256"],
    )
    manifest.verify_source_inventory(frozen_protocol, source_root)
    group_manifest = manifest.derive_group_manifest(frozen_protocol, source_root)
    protocol = derived_protocol(config)
    with manifest.isolated_predictive_models(source_root) as module:
        model_config = source_config["models"][protocol["model"]["kind"]]
        with torch.device("meta"):
            model = module.build_student(
                protocol["model"]["kind"], int(source_config["vocab_size"]),
                int(protocol["model"]["width"]), int(source_config["model_max_context"]),
                model_config,
            )
        model.load_state_dict(payload["model"], strict=True, assign=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = {name: [] for name in protocol["search"]["independent_branches"]}
    for name, _ in model.named_parameters():
        group = manifest.classify_parameter(name)
        if group in groups:
            groups[group].append(name)
    return production_bridge.LoadedW560Parent(
        model=model,
        groups={name: tuple(values) for name, values in groups.items()},
        parent_gate={"status": "PASS_LOCAL_PROMOTED_CHECKPOINT_EXACT_SHA_AND_MODEL_DIGEST"},
        group_manifest=group_manifest,
        checkpoint_sha256=checkpoint["sha256"],
    ), payload


def derived_protocol(config: Mapping[str, object]) -> Mapping[str, object]:
    vendor = config["vendor"]
    protocol = read_json_same_fd(
        Path(vendor["production_protocol_path"]),
        vendor["production_protocol_sha256"],
    )
    protocol = deepcopy(protocol)
    checkpoint = config["inputs"]["starting_checkpoint"]
    protocol["parent"]["checkpoint_path"] = checkpoint["path"]
    protocol["parent"]["checkpoint_sha256"] = checkpoint["sha256"]
    protocol["parent"]["size_bytes"] = checkpoint["size_bytes"]
    protocol["parent"]["model_state_digest"] = checkpoint["model_state_digest"]
    protocol["search"]["branch_seeds"][config["experiment"]["group"]] = int(
        config["experiment"]["seed"]
    )
    expected = config["experiment"]["frozen_solver_settings"]
    for key, value in expected.items():
        if protocol["search"].get(key) != value:
            raise RuntimeError(f"frozen solver setting drift: {key}")
    return protocol


def functional_effective_rms_cpu_float64(
    design: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
) -> torch.Tensor:
    design_gram = design.float().transpose(0, 1) @ design.float()
    controls = left.shape[0]
    total = torch.zeros((), device="cpu", dtype=torch.float64)
    left64 = left.detach().to("cpu").double()
    right64 = right.detach().to("cpu").double()
    gram64 = design_gram.detach().to("cpu").double()
    for first in range(controls):
        for second in range(controls):
            left_cross = left64[first].transpose(0, 1) @ left64[second]
            inner = ((left_cross @ right64[second]) * right64[first]).sum()
            total = total + gram64[first, second] * inner
    denominator = design.shape[0] * left.shape[1] * right.shape[2]
    return (total.clamp_min(0) / denominator).sqrt().float().to(left.device)


@contextmanager
def mps_normalization_adapter(branch_module, expected_sha256: str):
    before = file_identity(Path(branch_module.__file__), expected_sha256)
    original = branch_module._functional_effective_rms
    calls = {"count": 0}

    def adapted(design, left, right):
        if left.device.type != "mps" or right.device.type != "mps":
            raise RuntimeError("normalization adapter requires MPS factor tensors")
        calls["count"] += 1
        return functional_effective_rms_cpu_float64(design, left, right)

    branch_module._functional_effective_rms = adapted
    try:
        yield calls
    finally:
        branch_module._functional_effective_rms = original
        after = file_identity(Path(branch_module.__file__), expected_sha256)
        if before["sha256"] != after["sha256"]:
            raise RuntimeError("vendor branch solver bytes changed")


def load_frozen_kl_module(config: Mapping[str, object]):
    record = config["inputs"]["frozen_distillation_source"]
    path = Path(record["path"])
    read_bytes_same_fd(path, record["sha256"])
    module_name = (
        "frozen_w560_distribution_distillation_"
        f"{record['sha256'][:16]}"
    )
    existing = sys.modules.get(module_name)
    if existing is not None:
        if getattr(existing, "__hfw_source_sha256__", None) != record["sha256"]:
            raise RuntimeError("frozen distillation module cache identity drift")
        return existing
    source_root = str(Path(config["source"]["root"]))
    inserted = source_root not in sys.path
    if inserted:
        sys.path.insert(0, source_root)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load frozen distillation source")
        module = importlib.util.module_from_spec(spec)
        module.__hfw_source_sha256__ = record["sha256"]
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(module_name) is module:
                del sys.modules[module_name]
            raise
    finally:
        if inserted:
            sys.path.remove(source_root)
    return module


class TeacherLogitStore:
    def __init__(self, config: Mapping[str, object], manifest_path: Path):
        expected = config["inputs"]["teacher_outputs"]
        if Path(manifest_path).resolve() != Path(expected["capability_path"]).resolve():
            raise RuntimeError("teacher-logit capability path is not preregistered")
        self.manifest = read_json_same_fd(manifest_path, expected["manifest_sha256"])
        if self.manifest.get("schema") != "hfw_w560_ce_vs_kd_teacher_logits_capability_v1":
            raise RuntimeError("teacher-logit capability schema drift")
        allowed_statuses = {
            "PASS_ALL_FULL_VOCAB_TEACHER_LOGITS_FROZEN",
            "PASS_SEARCH_ONLY_TEACHER_LOGITS_CAPABILITY",
            "PASS_VERIFY_ONLY_TEACHER_LOGITS_CAPABILITY",
            "PASS_FINAL_ONLY_TEACHER_LOGITS_CAPABILITY_AFTER_SELECTION_SEAL",
        }
        if self.manifest.get("status") not in allowed_statuses:
            raise RuntimeError("teacher outputs are not terminal/frozen/capability-scoped")
        if self.manifest.get("teacher_forward_calls_after_freeze") != 0:
            raise RuntimeError("teacher was forwarded after output freeze")
        if self.manifest.get("master_manifest_sha256") != expected["master_manifest_sha256"]:
            raise RuntimeError("teacher-logit master-manifest identity drift")
        self.by_role = self.manifest["banks"]
        expected_roles = set(expected["roles_available"])
        if set(self.by_role) != expected_roles:
            raise RuntimeError("teacher-logit capability role inventory drift")
        if set(self.manifest.get("roles_available", ())) != expected_roles:
            raise RuntimeError("teacher-logit declared role inventory drift")
        capability_root = Path(manifest_path).resolve().parent
        for role in sorted(expected_roles):
            if role not in config["banks"]:
                raise RuntimeError("teacher-logit role lacks matching phase token bank")
            token_bank, token_identity = load_bank(config["banks"][role])
            sequences = token_bank.reshape(-1, SEQUENCE_TOKENS)
            role_record = self.by_role[role]
            if role_record.get("bank") != token_identity:
                raise RuntimeError("teacher-logit capability/token-bank identity drift")
            if int(role_record.get("sequence_count", -1)) != int(sequences.shape[0]):
                raise RuntimeError("teacher-logit sequence count drift")
            shards = role_record.get("shards")
            if not isinstance(shards, list) or len(shards) != int(sequences.shape[0]):
                raise RuntimeError("teacher-logit shard inventory drift")
            expected_role_root = (capability_root / role.lower()).resolve()
            for index, (shard, sequence) in enumerate(zip(shards, sequences)):
                exact_keys(
                    shard,
                    {
                        "index", "path", "sha256", "size_bytes", "shape", "dtype",
                        "input_sequence_sha256",
                    },
                    "teacher-logit shard",
                )
                shard_path = Path(shard["path"])
                if shard_path.is_symlink() or shard_path.resolve().parent != expected_role_root:
                    raise RuntimeError("teacher-logit shard path escaped role capability root")
                if int(shard["index"]) != index:
                    raise RuntimeError("teacher-logit shard index drift")
                if int(shard["size_bytes"]) != BF16_LOGIT_SHARD_BYTES:
                    raise RuntimeError("teacher-logit shard declared size drift")
                if shard["shape"] != [CONTEXT_TOKENS, TEACHER_VOCAB_SIZE]:
                    raise RuntimeError("teacher-logit shard shape drift")
                if shard["dtype"] != "bfloat16_little_endian_uint16_payload":
                    raise RuntimeError("teacher-logit shard dtype drift")
                sequence_sha = sha256_bytes(
                    sequence.numpy().astype("<u4", copy=False).tobytes(order="C")
                )
                if shard["input_sequence_sha256"] != sequence_sha:
                    raise RuntimeError("teacher-logit shard/token-sequence binding drift")
        self.cache: dict[tuple[str, int], torch.Tensor] = {}
        self.device_cache_key: tuple[str, tuple[int, ...], str] | None = None
        self.device_cache_value: torch.Tensor | None = None

    def batch(self, role: str, indices: Sequence[int], device: torch.device) -> torch.Tensor:
        indices = tuple(int(index) for index in indices)
        device_key = (role, indices, str(device))
        if self.device_cache_key == device_key and self.device_cache_value is not None:
            return self.device_cache_value
        values = []
        records = self.by_role[role]["shards"]
        for index in indices:
            key = (role, int(index))
            if key not in self.cache:
                record = records[int(index)]
                body = read_bytes_same_fd(Path(record["path"]), record["sha256"])
                if len(body) != BF16_LOGIT_SHARD_BYTES:
                    raise RuntimeError("teacher logit shard size drift")
                raw = np.frombuffer(body, dtype="<u2").copy()
                tensor = torch.from_numpy(raw).view(torch.bfloat16).view(
                    CONTEXT_TOKENS, TEACHER_VOCAB_SIZE,
                )
                self.cache[key] = tensor
            values.append(self.cache[key])
        self.device_cache_value = torch.stack(values).to(device)
        self.device_cache_key = device_key
        return self.device_cache_value

    def clear(self) -> None:
        self.cache.clear()
        self.device_cache_key = None
        self.device_cache_value = None


class MPSStudentObjective:
    def __init__(
        self, parent, batches: Sequence[torch.Tensor], device: torch.device,
        *, mode: str, config: Mapping[str, object], teacher_store: TeacherLogitStore | None = None,
        teacher_role: str = "SEARCH",
    ):
        if device.type != "mps" or not torch.backends.mps.is_available():
            raise RuntimeError("objective requires available MPS")
        if mode not in {"ce", "kd"}:
            raise ValueError("unknown objective mode")
        self.parent = parent
        self.device = device
        self.calibration = tuple(batch.detach().cpu().clone() for batch in batches)
        self.base_parameters = {name: value.detach() for name, value in parent.model.named_parameters()}
        self.buffers = {name: value.detach() for name, value in parent.model.named_buffers()}
        self.mode = mode
        self.config = config
        self.teacher_store = teacher_store
        self.teacher_role = teacher_role
        self.kl_module = load_frozen_kl_module(config) if mode == "kd" else None
        self.forward_calls = 0
        self.bfloat16_logits = 0
        self.last_metrics: dict[str, float] = {}

    def flat_group(self, group: str) -> torch.Tensor:
        return torch.cat([
            self.base_parameters[name].reshape(-1) for name in self.parent.groups[group]
        ]).detach()

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

    def logits_and_targets(
        self, state: Mapping[str, torch.Tensor], batch_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = set(self.base_parameters) | set(self.buffers)
        if set(state) != expected or any(value.requires_grad for value in state.values()):
            raise RuntimeError("candidate state inventory/autograd drift")
        parameters = {name: state[name].to(self.device) for name in self.base_parameters}
        buffers = {name: state[name].to(self.device) for name in self.buffers}
        raw = self.calibration[batch_index].to(
            device=self.device, dtype=torch.long,
        )
        if raw.dtype != torch.long:
            raise RuntimeError("student token inputs and targets are not int64")
        inputs, targets = raw[:, :-1], raw[:, 1:]
        with torch.inference_mode(), torch.autocast(
            device_type="mps", dtype=torch.bfloat16, enabled=True,
        ):
            output = torch.func.functional_call(
                self.parent.model, (parameters, buffers), (inputs,), strict=True,
            )
        self.forward_calls += 1
        if output.logits.dtype != torch.bfloat16:
            raise RuntimeError("student forward did not produce MPS bfloat16 logits")
        if tuple(output.logits.shape) != (
            raw.shape[0], CONTEXT_TOKENS, TEACHER_VOCAB_SIZE,
        ):
            raise RuntimeError("student full-vocabulary logit shape drift")
        self.bfloat16_logits += 1
        return output.logits, targets

    def scalar_state_loss(self, state: Mapping[str, torch.Tensor], batch_index: int) -> float:
        logits, targets = self.logits_and_targets(state, batch_index)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        )
        if self.mode == "ce":
            value = ce
            self.last_metrics = {"ce": float(ce)}
        else:
            if self.teacher_store is None or self.kl_module is None:
                raise RuntimeError("KD objective lacks frozen teacher logits")
            batch_size = int(targets.shape[0])
            first_index = batch_index * batch_size
            teacher = self.teacher_store.batch(
                self.teacher_role, range(first_index, first_index + batch_size), self.device,
            )
            settings = self.config["objectives"]["kd"]
            distribution, metrics = self.kl_module.multi_temperature_kl(
                logits, teacher, settings["temperatures"], int(settings["vocab_chunk_size"]),
            )
            value = float(settings["ce_weight"]) * ce + float(
                settings["distribution_weight"]
            ) * distribution
            self.last_metrics = {
                "ce": float(ce), "distribution": float(distribution),
                **{key: float(item) for key, item in metrics.items()},
            }
        if not torch.isfinite(value):
            return float("inf")
        return float(value)

    def loss(self, group: str, flat: torch.Tensor, batch_index: int) -> float:
        if flat.requires_grad:
            raise RuntimeError("autograd is forbidden")
        state = dict(self.group_mapping(group, flat))
        state.update(self.buffers)
        return self.scalar_state_loss(state, batch_index)

    def loss_vector(self, group: str, flat: torch.Tensor, batch_index: int):
        state = dict(self.group_mapping(group, flat))
        state.update(self.buffers)
        logits, targets = self.logits_and_targets(state, batch_index)
        per_token = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
        ).view_as(targets)
        return tuple(float(value) for value in per_token.mean(dim=1).cpu())


@contextmanager
def exact_candidate_materialization(branch_module, objective, group: str, root: Path):
    original = branch_module._materialize_child
    base = objective.flat_group(group).detach()
    previous = base
    records: list[Mapping[str, object]] = []
    artifacts = root / "exact_candidate_bytes"
    artifacts.mkdir(parents=True, exist_ok=False)

    def materialize(
        _objective, _group, flat, path, immutable_parent_sha256,
        immediate_parent_child_sha256, chain_sha256,
    ):
        nonlocal previous
        if _objective is not objective or _group != group:
            raise RuntimeError("materialization objective/group drift")
        exact = tensor_f32le_bytes(flat)
        cumulative = tensor_f32le_bytes(flat - base)
        incremental = tensor_f32le_bytes(flat - previous)
        exact_sha = sha256_bytes(exact)
        cumulative_sha = sha256_bytes(cumulative)
        incremental_sha = sha256_bytes(incremental)
        exact_artifact = write_exclusive(artifacts / f"{exact_sha}.group_flat.f32le", exact)
        cumulative_artifact = write_exclusive(
            artifacts / f"{cumulative_sha}.cumulative_delta.f32le", cumulative,
        )
        incremental_artifact = write_exclusive(
            artifacts / f"{incremental_sha}.incremental_delta.f32le", incremental,
        )
        descriptor = {
            "schema": "hfw_w560_ab_exact_candidate_descriptor_v1",
            "group": group,
            "accept_index": len(records) + 1,
            "immutable_parent_checkpoint_sha256": immutable_parent_sha256,
            "immediate_parent_descriptor_sha256": immediate_parent_child_sha256,
            "accepted_chain_sha256": chain_sha256,
            "exact_candidate_group_flat": exact_artifact,
            "cumulative_delta": cumulative_artifact,
            "incremental_delta": incremental_artifact,
            "candidate_minus_base_matches_cumulative_delta_bitwise": (
                sha256_bytes(tensor_f32le_bytes(flat - base)) == cumulative_sha
            ),
            "candidate_minus_previous_matches_incremental_delta_bitwise": (
                sha256_bytes(tensor_f32le_bytes(flat - previous)) == incremental_sha
            ),
            "full_checkpoint_materialized_during_search": False,
            "optimizer_steps": 0,
            "autograd_calls": 0,
        }
        descriptor_artifact = write_exclusive_json(path, descriptor)
        records.append({**descriptor, "descriptor_artifact": descriptor_artifact})
        previous = flat.detach().clone()
        return descriptor_artifact["sha256"]

    branch_module._materialize_child = materialize
    try:
        yield records
    finally:
        branch_module._materialize_child = original


class AppendOnlyRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.index = 0

    def append(self, **record):
        value = {"schema": "hfw_w560_ab_proposal_registry_node_v1", **record}
        digest = sha256_bytes(canonical_json_bytes(value))
        write_exclusive_json(self.root / f"{self.index:03d}-{digest}.json", value)
        self.index += 1
        return digest
