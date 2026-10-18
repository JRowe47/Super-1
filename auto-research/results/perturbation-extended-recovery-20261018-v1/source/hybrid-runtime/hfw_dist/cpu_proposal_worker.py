from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import types
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    canonical_json_bytes, fsync_directory, hash_file, read_json, sha256_bytes,
    validate_inventory, write_exclusive, write_json_exclusive,
)
from .model_io import build_loaded_parent


VENDOR_PACKAGE = "vendor.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver"
FAILURE_SCHEMA = "hfw_w560_daytona_cpu_search_failure_v1"
FAILURE_FILENAME = "CPU_SEARCH_FAILURE.json"
DETERMINISTIC_NUMERICAL_EXIT_CODE = 42


class DeterministicNumericalFailure(RuntimeError):
    def __init__(
        self, stage: str, diagnostics: Mapping[str, object], *,
        solver_artifacts_created: bool,
    ):
        super().__init__(f"deterministic numerical failure at {stage}")
        self.stage = stage
        self.diagnostics = dict(diagnostics)
        self.solver_artifacts_created = bool(solver_artifacts_created)


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def deterministic_failure_peak_rss(
    objective_peak_rss_bytes: int, diagnostics: Mapping[str, object],
) -> int:
    return max(
        int(objective_peak_rss_bytes),
        int(diagnostics.get("observed_peak_rss_bytes", 0)),
    )


def cpu_model_disclosure() -> Mapping[str, object]:
    names = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line and line.split(":", 1)[0].strip() in {"model name", "Processor"}:
                value = line.split(":", 1)[1].strip()
                if value and value not in names:
                    names.append(value)
    return {
        "platform_processor": platform.processor(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "proc_cpuinfo_distinct_model_names": names,
    }


def require_finite_terminal_ranking(
    losses: list[float], objective: "CPUPhysicalMicrobatchObjective",
) -> None:
    if all(math.isfinite(value) for value in losses):
        return
    details = dict(objective.last_nonfinite_diagnostics or {
        "reason": "NONFINITE_TERMINAL_FULL_SEARCH_CE",
    })
    details["reason_code"] = "NONFINITE_TERMINAL_RANKING"
    raise DeterministicNumericalFailure(
        "TERMINAL_RANKING", details, solver_artifacts_created=True,
    )


def expected_variable_totals(
    solver: Mapping[str, object], gate_attempts: int,
) -> tuple[int, int]:
    maximum = int(solver["acceptance_gate_maximum_attempts"])
    if not 0 <= gate_attempts <= maximum:
        raise RuntimeError("acceptance gate attempts outside sealed bound")
    logical = int(solver["total_logical_evaluations_minimum"]) + gate_attempts * int(
        solver["acceptance_gate_logical_evaluations_per_attempt"]
    )
    physical = int(solver["total_physical_model_forwards_minimum"]) + gate_attempts * int(
        solver["acceptance_gate_physical_model_forwards_per_attempt"]
    )
    if (
        logical > int(solver["total_logical_evaluations_maximum"])
        or physical > int(solver["total_physical_model_forwards_maximum"])
    ):
        raise RuntimeError("acceptance gate accounting exceeded sealed maximum")
    return logical, physical


def validate_terminal_duplicate(
    objective: "CPUFP64StableMicrobatchObjective", group: str,
    final_flat: torch.Tensor, losses: list[float], tolerance: float,
) -> Mapping[str, object]:
    terminal_flat_sha = objective._flat_sha256(final_flat)
    cached = objective._current_full_bank.get(group)
    if cached is None or cached["flat_sha256"] != terminal_flat_sha:
        raise DeterministicNumericalFailure(
            "TERMINAL_RANKING", {
                "reason_code": "TERMINAL_CACHE_FLAT_IDENTITY_MISMATCH",
                "terminal_flat_sha256": terminal_flat_sha,
                "cached_flat_sha256": None if cached is None else cached["flat_sha256"],
            }, solver_artifacts_created=True,
        )
    if len(losses) != len(cached["batch_losses"]):
        raise DeterministicNumericalFailure(
            "TERMINAL_RANKING", {
                "reason_code": "TERMINAL_DUPLICATE_BATCH_COUNT_MISMATCH",
                "terminal_batch_count": len(losses),
                "cached_batch_count": len(cached["batch_losses"]),
            }, solver_artifacts_created=True,
        )
    maximum = max(
        abs(float(left) - float(right))
        for left, right in zip(losses, cached["batch_losses"])
    )
    if maximum > tolerance:
        raise DeterministicNumericalFailure(
            "TERMINAL_RANKING", {
                "reason_code": "TERMINAL_DUPLICATE_TOLERANCE_EXCEEDED",
                "maximum_absolute_difference": maximum,
                "absolute_tolerance": tolerance,
                "flat_sha256": terminal_flat_sha,
            }, solver_artifacts_created=True,
        )
    return {
        "status": "PASS_TERMINAL_FULL_BANK_DUPLICATES_CACHED_COMMIT_GATE",
        "flat_sha256": terminal_flat_sha,
        "batch_count": len(losses),
        "absolute_tolerance": tolerance,
        "maximum_absolute_difference": maximum,
    }


class CPUPhysicalMicrobatchObjective:
    """CPU BF16-autocast forward with FP32 CE and one sequence per invocation.

    The unchanged vendor solver sees one logical loss evaluation for each [2,513]
    SEARCH batch. This class performs two [1,513] forwards and averages their CE,
    bounding peak logits memory for a 4 GiB worker.
    """

    def __init__(self, parent, calibration_batches, device: torch.device, expected_shape):
        if device.type != "cpu":
            raise RuntimeError("CPU proposal objective received non-CPU device")
        self.parent = parent
        self.device = device
        self.calibration = tuple(batch.detach().cpu().to(torch.long) for batch in calibration_batches)
        expected_shape = tuple(int(value) for value in expected_shape)
        if len(expected_shape) != 3 or expected_shape[1] <= 0:
            raise RuntimeError("invalid sealed SEARCH shape")
        if len(self.calibration) != expected_shape[0]:
            raise RuntimeError("SEARCH batch count drift")
        if any(tuple(batch.shape) != expected_shape[1:] for batch in self.calibration):
            raise RuntimeError("SEARCH batch shape drift")
        self.expected_shape = expected_shape
        self.base_parameters = {
            name: value.detach() for name, value in parent.model.named_parameters()
        }
        self.buffers = {name: value.detach() for name, value in parent.model.named_buffers()}
        self.logical_evaluations = 0
        self.physical_model_forwards = 0
        self.preflight_logical_evaluations = 0
        self.preflight_physical_model_forwards = 0
        self.cpu_bfloat16_autocast_entries = 0
        self.bfloat16_logits_observed = 0
        self.float32_ce_evaluations = 0
        self.last_nonfinite_diagnostics: Mapping[str, object] | None = None
        self.maximum_observed_rss_bytes = peak_rss_bytes()
        self._flat_cache: dict[str, torch.Tensor] = {}

    def flat_group(self, group: str) -> torch.Tensor:
        if group not in self._flat_cache:
            self._flat_cache[group] = torch.cat([
                self.base_parameters[name].reshape(-1) for name in self.parent.groups[group]
            ]).detach()
        return self._flat_cache[group]

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

    @staticmethod
    def _tensor_time_diagnostics(value: torch.Tensor) -> Mapping[str, object]:
        if value.ndim < 2:
            raise RuntimeError("diagnostic tensor lacks batch/time axes")
        first_bad = None
        maximum_finite_abs = 0.0
        for timestep in range(int(value.shape[1])):
            row = value[:, timestep]
            finite = torch.isfinite(row)
            if first_bad is None and not bool(finite.all()):
                first_bad = timestep
            if bool(finite.any()):
                maximum_finite_abs = max(
                    maximum_finite_abs,
                    float(row[finite].detach().float().abs().max()),
                )
        return {
            "all_finite": first_bad is None,
            "first_nonfinite_timestep": first_bad,
            "maximum_finite_absolute_value": maximum_finite_abs,
        }

    @staticmethod
    def _scalar_class(value: torch.Tensor) -> str:
        scalar = value.detach().float()
        if bool(torch.isfinite(scalar)):
            return "FINITE"
        if bool(torch.isnan(scalar)):
            return "NAN"
        return "POSITIVE_INFINITY" if float(scalar) > 0 else "NEGATIVE_INFINITY"

    def _evaluate_sequence(
        self, parameters: Mapping[str, torch.Tensor], sequence: torch.Tensor,
        *, preflight: bool, include_diagnostics: bool,
    ) -> tuple[float, Mapping[str, object] | None]:
        inputs, targets = sequence[:, :-1], sequence[:, 1:]
        with torch.inference_mode(), torch.autocast(
            device_type="cpu", dtype=torch.bfloat16, enabled=True,
        ):
            if not torch.is_autocast_enabled("cpu"):
                raise RuntimeError("CPU bfloat16 autocast did not activate")
            self.cpu_bfloat16_autocast_entries += 1
            if preflight:
                self.preflight_physical_model_forwards += 1
            else:
                self.physical_model_forwards += 1
            output = torch.func.functional_call(
                self.parent.model, (parameters, self.buffers), (inputs,), strict=True,
            )
        logits_dtype_matches = output.logits.dtype == torch.bfloat16
        if logits_dtype_matches:
            self.bfloat16_logits_observed += 1
        logits_float = output.logits.float()
        per_token = F.cross_entropy(
            logits_float.reshape(-1, logits_float.shape[-1]),
            targets.reshape(-1), reduction="none",
        ).view_as(targets)
        if per_token.dtype != torch.float32:
            raise RuntimeError("CPU proposal CE did not execute in FP32")
        self.float32_ce_evaluations += 1
        loss = per_token.mean()
        finite = bool(torch.isfinite(loss))
        diagnostics = None
        if include_diagnostics or not finite or not logits_dtype_matches:
            hidden = getattr(output, "hidden_states", None)
            if not torch.is_tensor(hidden):
                raise RuntimeError("model output lacks tensor hidden_states for preflight")
            logits_diagnostics = self._tensor_time_diagnostics(output.logits)
            hidden_diagnostics = self._tensor_time_diagnostics(hidden)
            loss_diagnostics = self._tensor_time_diagnostics(per_token)
            loss_class = self._scalar_class(loss)
            diagnostics = {
                "logits_dtype": str(output.logits.dtype).removeprefix("torch."),
                "expected_logits_dtype": "bfloat16",
                "logits_dtype_matches": logits_dtype_matches,
                "hidden_dtype": str(hidden.dtype).removeprefix("torch."),
                "ce_dtype": str(per_token.dtype).removeprefix("torch."),
                "expected_ce_dtype": "float32",
                "loss_class": loss_class,
                "loss": float(loss) if loss_class == "FINITE" else None,
                "logits": logits_diagnostics,
                "hidden": hidden_diagnostics,
                "per_token_loss": loss_diagnostics,
            }
        value = float(loss) if finite else float("inf")
        del loss, per_token, logits_float, output
        self.maximum_observed_rss_bytes = max(
            self.maximum_observed_rss_bytes, peak_rss_bytes(),
        )
        return value, diagnostics

    def preflight_base_center(
        self, group: str, flat: torch.Tensor, batch_index: int,
    ) -> Mapping[str, object]:
        if flat.requires_grad:
            raise RuntimeError("autograd is forbidden")
        parameters = self.group_mapping(group, flat)
        if any(value.requires_grad for value in parameters.values()):
            raise RuntimeError("candidate parameters unexpectedly require gradients")
        raw = self.calibration[batch_index]
        per_sequence = []
        values = []
        self.preflight_logical_evaluations += 1
        for sequence_index in range(int(raw.shape[0])):
            try:
                value, diagnostics = self._evaluate_sequence(
                    parameters, raw[sequence_index:sequence_index + 1],
                    preflight=True, include_diagnostics=True,
                )
            except Exception as error:
                self.maximum_observed_rss_bytes = max(
                    self.maximum_observed_rss_bytes, peak_rss_bytes(),
                )
                exception_evidence = {
                    "schema": "hfw_w560_cpu_bfloat16_base_center_preflight_v1",
                    "status": "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT",
                    "failure_reasons": ["SEALED_PREFLIGHT_OPERATION_EXCEPTION"],
                    "batch_index": int(batch_index),
                    "logical_batch_shape": list(raw.shape),
                    "physical_microbatch_shape": [1, int(raw.shape[1])],
                    "parameter_dtype": "float32",
                    "forward_device": "cpu",
                    "forward_autocast_dtype": "bfloat16",
                    "ce_dtype": "float32",
                    "per_sequence": per_sequence,
                    "failed_sequence_index": sequence_index,
                    "operation_exception_type": type(error).__name__,
                    "operation_exception_message_sha256": sha256_bytes(
                        str(error).encode("utf-8", errors="replace")
                    ),
                    "base_center_mean_ce": None,
                    "preflight_logical_evaluations": self.preflight_logical_evaluations,
                    "preflight_physical_model_forwards": self.preflight_physical_model_forwards,
                    "total_logical_evaluations_including_preflight": (
                        self.logical_evaluations + self.preflight_logical_evaluations
                    ),
                    "total_physical_model_forwards_including_preflight": (
                        self.physical_model_forwards + self.preflight_physical_model_forwards
                    ),
                    "maximum_observed_rss_bytes": self.maximum_observed_rss_bytes,
                }
                raise DeterministicNumericalFailure(
                    "BASE_CENTER_PREFLIGHT", {
                        "reason_code": "SEALED_PREFLIGHT_OPERATION_EXCEPTION",
                        "preflight": exception_evidence,
                    }, solver_artifacts_created=False,
                ) from error
            assert diagnostics is not None
            values.append(value)
            per_sequence.append({"sequence_index": sequence_index, **diagnostics})
        passed = all(
            math.isfinite(value)
            and row["logits_dtype_matches"]
            and row["ce_dtype"] == "float32"
            and row["logits"]["all_finite"]
            and row["hidden"]["all_finite"]
            and row["per_token_loss"]["all_finite"]
            for value, row in zip(values, per_sequence)
        )
        failure_reasons = set()
        for value, row in zip(values, per_sequence):
            if not math.isfinite(value):
                failure_reasons.add("NONFINITE_MEAN_CE")
            if not row["logits_dtype_matches"]:
                failure_reasons.add("LOGITS_DTYPE_MISMATCH")
            if not row["logits"]["all_finite"]:
                failure_reasons.add("NONFINITE_LOGITS")
            if not row["hidden"]["all_finite"]:
                failure_reasons.add("NONFINITE_HIDDEN")
            if not row["per_token_loss"]["all_finite"]:
                failure_reasons.add("NONFINITE_PER_TOKEN_CE")
        diagnostics = {
            "schema": "hfw_w560_cpu_bfloat16_base_center_preflight_v1",
            "status": (
                "PASS_FINITE_CPU_BFLOAT16_BASE_CENTER"
                if passed else "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT"
            ),
            "failure_reasons": sorted(failure_reasons),
            "batch_index": int(batch_index),
            "logical_batch_shape": list(raw.shape),
            "physical_microbatch_shape": [1, int(raw.shape[1])],
            "parameter_dtype": "float32",
            "forward_device": "cpu",
            "forward_autocast_dtype": "bfloat16",
            "ce_dtype": "float32",
            "per_sequence": per_sequence,
            "base_center_mean_ce": math.fsum(values) / len(values) if passed else None,
            "preflight_logical_evaluations": self.preflight_logical_evaluations,
            "preflight_physical_model_forwards": self.preflight_physical_model_forwards,
            "total_logical_evaluations_including_preflight": (
                self.logical_evaluations + self.preflight_logical_evaluations
            ),
            "total_physical_model_forwards_including_preflight": (
                self.physical_model_forwards + self.preflight_physical_model_forwards
            ),
            "maximum_observed_rss_bytes": self.maximum_observed_rss_bytes,
        }
        if not passed:
            raise DeterministicNumericalFailure(
                "BASE_CENTER_PREFLIGHT", {
                    "reason_code": "BASE_CENTER_NUMERICAL_OR_DTYPE_GATE_FAILED",
                    "preflight": diagnostics,
                },
                solver_artifacts_created=False,
            )
        return diagnostics

    def loss(self, group: str, flat: torch.Tensor, batch_index: int) -> float:
        if flat.requires_grad:
            raise RuntimeError("autograd is forbidden")
        parameters = self.group_mapping(group, flat)
        if any(value.requires_grad for value in parameters.values()):
            raise RuntimeError("candidate parameters unexpectedly require gradients")
        raw = self.calibration[batch_index]
        per_sequence = []
        first_nonfinite_diagnostics = None
        for sequence_index in range(int(raw.shape[0])):
            value, diagnostics = self._evaluate_sequence(
                parameters, raw[sequence_index:sequence_index + 1],
                preflight=False, include_diagnostics=False,
            )
            if diagnostics is not None and not diagnostics["logits_dtype_matches"]:
                raise DeterministicNumericalFailure(
                    "FORWARD_DTYPE", {
                        "reason_code": "OBSERVED_LOGITS_DTYPE_MISMATCH",
                        "group": group, "batch_index": int(batch_index),
                        "sequence_index": sequence_index, "sequence": diagnostics,
                    }, solver_artifacts_created=True,
                )
            if not math.isfinite(value):
                if first_nonfinite_diagnostics is None:
                    first_nonfinite_diagnostics = {
                        "group": group, "batch_index": int(batch_index),
                        "sequence_index": sequence_index, "sequence": diagnostics,
                    }
            per_sequence.append(value)
        self.logical_evaluations += 1
        if first_nonfinite_diagnostics is not None:
            self.last_nonfinite_diagnostics = first_nonfinite_diagnostics
            return float("inf")
        return math.fsum(per_sequence) / len(per_sequence)

    def receipt(self) -> Mapping[str, object]:
        return {
            "schema": "hfw_w560_cpu_physical_microbatch_objective_receipt_v1",
            "device": "cpu", "parameter_dtype": "float32",
            "forward_autocast_dtype": "bfloat16", "logits_dtype": "bfloat16",
            "ce_dtype": "float32",
            "logical_batch_shape": list(self.expected_shape[1:]),
            "physical_microbatch_shape": [1, self.expected_shape[2]],
            "logical_evaluations": self.logical_evaluations,
            "physical_model_forwards": self.physical_model_forwards,
            "preflight_logical_evaluations": self.preflight_logical_evaluations,
            "preflight_physical_model_forwards": self.preflight_physical_model_forwards,
            "total_logical_evaluations_including_preflight": (
                self.logical_evaluations + self.preflight_logical_evaluations
            ),
            "total_physical_model_forwards_including_preflight": (
                self.physical_model_forwards + self.preflight_physical_model_forwards
            ),
            "cpu_bfloat16_autocast_entries": self.cpu_bfloat16_autocast_entries,
            "bfloat16_logits_observed": self.bfloat16_logits_observed,
            "float32_ce_evaluations": self.float32_ce_evaluations,
            "physical_model_forwards_per_logical_evaluation": self.expected_shape[1],
            "maximum_observed_rss_bytes": self.maximum_observed_rss_bytes,
            "inference_only": True, "optimizer_steps": 0, "autograd_calls": 0,
        }


def stable_fp64_layer_norm(module: torch.nn.LayerNorm, values: torch.Tensor) -> torch.Tensor:
    """LayerNorm-equivalent real arithmetic without squaring unscaled huge values."""
    if values.dtype != torch.float64:
        raise RuntimeError("stable proposal LayerNorm requires FP64 input")
    maximum = values.detach().abs().amax(dim=-1, keepdim=True)
    floor = torch.full_like(maximum, math.sqrt(float(module.eps)))
    scale = torch.maximum(maximum, floor)
    scaled = values / scale
    centered = scaled - scaled.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + (float(module.eps) / scale) / scale)
    if module.elementwise_affine:
        normalized = normalized * module.weight + module.bias
    return normalized


class CPUFP64StableMicrobatchObjective(CPUPhysicalMicrobatchObjective):
    """Exact-F32 proposal domain scored by an FP64 shadow forward.

    The canonical checkpoint/model source remains unchanged.  Candidate tensors,
    cumulative deltas, and terminal flats stay FP32.  Only scoring widens those
    exact FP32 tensors into a separate FP64 functional state.  The canonical
    compressed recurrent state is FP32, so this proposal-only lane explicitly
    replaces its state/codec dtype with FP64 and installs a numerically scaled
    final LayerNorm.  It is not raw-CE comparable with any canonical lane.
    """

    PHASE_SEARCH = "SEARCH"
    PHASE_ACCEPTANCE_GATE = "ACCEPTANCE_COMMIT_GATE"
    PHASE_TERMINAL = "TERMINAL_RANKING"

    def __init__(self, parent, calibration_batches, device: torch.device, expected_shape):
        if device.type != "cpu":
            raise RuntimeError("FP64 proposal objective received non-CPU device")
        self.parent = parent
        self.device = device
        self.calibration = tuple(batch.detach().cpu().to(torch.long) for batch in calibration_batches)
        expected_shape = tuple(int(value) for value in expected_shape)
        if len(expected_shape) != 3 or expected_shape[1] <= 0:
            raise RuntimeError("invalid sealed SEARCH shape")
        if len(self.calibration) != expected_shape[0] or any(
            tuple(batch.shape) != expected_shape[1:] for batch in self.calibration
        ):
            raise RuntimeError("SEARCH batch shape drift")
        self.expected_shape = expected_shape

        # Retain the immutable checkpoint-domain FP32 tensors before converting
        # the live module into a proposal-only FP64 shadow evaluator.
        self.base_parameters = {
            name: value.detach() for name, value in parent.model.named_parameters()
        }
        self.buffers = {name: value.detach() for name, value in parent.model.named_buffers()}
        if any(value.dtype != torch.float32 for value in self.base_parameters.values()):
            raise RuntimeError("checkpoint proposal parameter domain is not exact FP32")
        if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in self.buffers.values()
        ):
            raise RuntimeError("checkpoint proposal floating-buffer domain is not exact FP32")

        checkpoint_domain_digest_before = hashlib.sha256()
        for name in sorted(self.base_parameters):
            checkpoint_domain_digest_before.update(name.encode("utf-8"))
            checkpoint_domain_digest_before.update(self._flat_bytes(
                self.base_parameters[name].reshape(-1)
            ))
        checkpoint_domain_digest_before = checkpoint_domain_digest_before.hexdigest()
        parent.model.double()
        checkpoint_domain_digest_after = hashlib.sha256()
        for name in sorted(self.base_parameters):
            if self.base_parameters[name].dtype != torch.float32:
                raise RuntimeError("retained checkpoint parameter changed dtype during shadow conversion")
            checkpoint_domain_digest_after.update(name.encode("utf-8"))
            checkpoint_domain_digest_after.update(self._flat_bytes(
                self.base_parameters[name].reshape(-1)
            ))
        if checkpoint_domain_digest_after.hexdigest() != checkpoint_domain_digest_before:
            raise RuntimeError("retained checkpoint FP32 parameter bytes changed during shadow conversion")
        if not hasattr(parent.model, "state_codecs") or not hasattr(parent.model, "final_norm"):
            raise RuntimeError("sealed FP64 adapter model surface drift")

        def initial_state_native(model, batch_size, state_device, dtype=None):
            del dtype
            if bool(model.identity_codec_mode):
                raise RuntimeError("sealed FP64 adapter unexpectedly entered identity codec mode")
            actual_dtype = model.embedding.weight.dtype
            zero = lambda: torch.zeros(
                batch_size, model.state_width, device=state_device, dtype=actual_dtype,
            )
            return {
                "fast": zero(), "segment": zero(), "document": zero(),
                "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
                "position": 0, "segment_updates": 0, "document_updates": 0,
            }

        def encode_native(codec, hidden):
            return F.linear(hidden, codec.weight)

        def final_norm_forward(layer_norm, hidden):
            return stable_fp64_layer_norm(layer_norm, hidden)

        parent.model.initial_state = types.MethodType(initial_state_native, parent.model)
        for codec in parent.model.state_codecs.values():
            codec.encode = types.MethodType(encode_native, codec)
        parent.model.final_norm.forward = types.MethodType(
            final_norm_forward, parent.model.final_norm,
        )

        self.evaluation_base_parameters = {
            name: value.detach() for name, value in parent.model.named_parameters()
        }
        self.evaluation_buffers = {
            name: value.detach() for name, value in parent.model.named_buffers()
        }
        if any(value.dtype != torch.float64 for value in self.evaluation_base_parameters.values()):
            raise RuntimeError("FP64 shadow parameter conversion failed")
        if any(
            value.is_floating_point() and value.dtype != torch.float64
            for value in self.evaluation_buffers.values()
        ):
            raise RuntimeError("FP64 shadow floating-buffer conversion failed")

        self.logical_evaluations = 0
        self.physical_model_forwards = 0
        self.preflight_logical_evaluations = 0
        self.preflight_physical_model_forwards = 0
        self.phase_logical_evaluations = {
            self.PHASE_SEARCH: 0, self.PHASE_ACCEPTANCE_GATE: 0,
            self.PHASE_TERMINAL: 0,
        }
        self.phase_physical_model_forwards = {
            self.PHASE_SEARCH: 0, self.PHASE_ACCEPTANCE_GATE: 0,
            self.PHASE_TERMINAL: 0,
        }
        self.float64_logits_observed = 0
        self.float64_hidden_observed = 0
        self.float64_ce_evaluations = 0
        self.float64_persistent_state_observed = 0
        self.last_nonfinite_diagnostics: Mapping[str, object] | None = None
        self.maximum_observed_rss_bytes = peak_rss_bytes()
        self._flat_cache: dict[str, torch.Tensor] = {}
        self._active_phase = self.PHASE_SEARCH
        self.acceptance_gate_records: list[Mapping[str, object]] = []
        self.acceptance_gate_passes = 0
        self.acceptance_gate_failures = 0
        self._current_full_bank: dict[str, Mapping[str, object]] = {}

    @staticmethod
    def _flat_bytes(flat: torch.Tensor) -> bytes:
        if flat.device.type != "cpu" or flat.dtype != torch.float32 or flat.requires_grad:
            raise RuntimeError("proposal candidate left exact CPU FP32 domain")
        return flat.detach().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")

    @classmethod
    def _flat_sha256(cls, flat: torch.Tensor) -> str:
        payload = cls._flat_bytes(flat)
        roundtrip = torch.from_numpy(np.frombuffer(payload, dtype="<f4").copy()).view_as(flat)
        if not torch.equal(roundtrip, flat):
            raise RuntimeError("FP32 candidate serialization roundtrip drift")
        return hashlib.sha256(payload).hexdigest()

    def group_mapping(self, group: str, flat: torch.Tensor) -> Mapping[str, torch.Tensor]:
        self._flat_sha256(flat)
        mapping = dict(self.evaluation_base_parameters)
        cursor = 0
        for name in self.parent.groups[group]:
            reference = self.base_parameters[name]
            count = reference.numel()
            candidate = flat[cursor:cursor + count].view_as(reference)
            mapping[name] = candidate.to(dtype=torch.float64)
            cursor += count
        if cursor != flat.numel():
            raise RuntimeError("candidate group vector length drift")
        return mapping

    @staticmethod
    def _tensor_time_diagnostics(value: torch.Tensor) -> Mapping[str, object]:
        if value.ndim < 2:
            raise RuntimeError("diagnostic tensor lacks batch/time axes")
        first_bad = None
        maximum_finite_abs = 0.0
        for timestep in range(int(value.shape[1])):
            row = value[:, timestep]
            finite = torch.isfinite(row)
            if first_bad is None and not bool(finite.all()):
                first_bad = timestep
            if bool(finite.any()):
                maximum_finite_abs = max(
                    maximum_finite_abs, float(row[finite].detach().abs().max().item()),
                )
        return {
            "all_finite": first_bad is None,
            "first_nonfinite_timestep": first_bad,
            "maximum_finite_absolute_value": maximum_finite_abs,
        }

    @staticmethod
    def _scalar_class(value: torch.Tensor) -> str:
        scalar = value.detach()
        if bool(torch.isfinite(scalar)):
            return "FINITE"
        if bool(torch.isnan(scalar)):
            return "NAN"
        return "NEGATIVE_INFINITY" if bool(torch.signbit(scalar)) else "POSITIVE_INFINITY"

    def _evaluate_sequence(
        self, parameters: Mapping[str, torch.Tensor], sequence: torch.Tensor,
        *, preflight: bool, include_diagnostics: bool,
    ) -> tuple[float, Mapping[str, object] | None]:
        inputs, targets = sequence[:, :-1], sequence[:, 1:]
        if preflight:
            self.preflight_physical_model_forwards += 1
        else:
            self.physical_model_forwards += 1
            self.phase_physical_model_forwards[self._active_phase] += 1
        with torch.inference_mode():
            output = torch.func.functional_call(
                self.parent.model, (parameters, self.evaluation_buffers), (inputs,), strict=True,
            )
        logits_dtype_matches = output.logits.dtype == torch.float64
        hidden = getattr(output, "hidden_states", None)
        if not torch.is_tensor(hidden):
            raise RuntimeError("model output lacks tensor hidden_states")
        hidden_dtype_matches = hidden.dtype == torch.float64
        final_state = getattr(output, "final_state", None)
        state_tensors = [] if not isinstance(final_state, Mapping) else [
            value for value in final_state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        ]
        state_dtype_matches = bool(state_tensors) and all(
            value.dtype == torch.float64 for value in state_tensors
        )
        if logits_dtype_matches:
            self.float64_logits_observed += 1
        if hidden_dtype_matches:
            self.float64_hidden_observed += 1
        if state_dtype_matches:
            self.float64_persistent_state_observed += 1
        per_token = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]),
            targets.reshape(-1), reduction="none",
        ).view_as(targets)
        if per_token.dtype == torch.float64:
            self.float64_ce_evaluations += 1
        loss = per_token.mean()
        finite = bool(torch.isfinite(loss))
        diagnostics = None
        if (
            include_diagnostics or not finite or not logits_dtype_matches
            or not hidden_dtype_matches or not state_dtype_matches
            or per_token.dtype != torch.float64
        ):
            loss_class = self._scalar_class(loss)
            diagnostics = {
                "logits_dtype": str(output.logits.dtype).removeprefix("torch."),
                "expected_logits_dtype": "float64",
                "logits_dtype_matches": logits_dtype_matches,
                "hidden_dtype": str(hidden.dtype).removeprefix("torch."),
                "expected_hidden_dtype": "float64",
                "hidden_dtype_matches": hidden_dtype_matches,
                "persistent_state_floating_dtypes": sorted({
                    str(value.dtype).removeprefix("torch.") for value in state_tensors
                }),
                "persistent_state_dtype_matches": state_dtype_matches,
                "ce_dtype": str(per_token.dtype).removeprefix("torch."),
                "expected_ce_dtype": "float64",
                "loss_class": loss_class,
                "loss": float(loss.item()) if loss_class == "FINITE" else None,
                "logits": self._tensor_time_diagnostics(output.logits),
                "hidden": self._tensor_time_diagnostics(hidden),
                "per_token_loss": self._tensor_time_diagnostics(per_token),
            }
        value = float(loss.item()) if finite else float("inf")
        del loss, per_token, output, hidden, final_state, state_tensors
        self.maximum_observed_rss_bytes = max(self.maximum_observed_rss_bytes, peak_rss_bytes())
        return value, diagnostics

    @staticmethod
    def _diagnostics_pass(value: float, row: Mapping[str, object]) -> bool:
        return bool(
            math.isfinite(value)
            and row["logits_dtype_matches"]
            and row["hidden_dtype_matches"]
            and row["persistent_state_dtype_matches"]
            and row["ce_dtype"] == "float64"
            and row["logits"]["all_finite"]
            and row["hidden"]["all_finite"]
            and row["per_token_loss"]["all_finite"]
        )

    def preflight_all_base_batches(self, group: str, flat: torch.Tensor) -> Mapping[str, object]:
        parameters = self.group_mapping(group, flat)
        batch_rows = []
        batch_losses = []
        failure_reasons = set()
        for batch_index, raw in enumerate(self.calibration):
            self.preflight_logical_evaluations += 1
            values = []
            per_sequence = []
            for sequence_index in range(int(raw.shape[0])):
                try:
                    value, diagnostics = self._evaluate_sequence(
                        parameters, raw[sequence_index:sequence_index + 1],
                        preflight=True, include_diagnostics=True,
                    )
                except Exception as error:
                    evidence = {
                        "schema": "hfw_w560_cpu_fp64_all_search_base_preflight_v1",
                        "status": "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT",
                        "failure_reasons": ["SEALED_PREFLIGHT_OPERATION_EXCEPTION"],
                        "completed_batches": batch_rows,
                        "failed_batch_index": batch_index,
                        "failed_sequence_index": sequence_index,
                        "operation_exception_type": type(error).__name__,
                        "operation_exception_message_sha256": sha256_bytes(
                            str(error).encode("utf-8", errors="replace")
                        ),
                    }
                    raise DeterministicNumericalFailure(
                        "BASE_CENTER_PREFLIGHT", {
                            "reason_code": "SEALED_PREFLIGHT_OPERATION_EXCEPTION",
                            "preflight": evidence,
                        }, solver_artifacts_created=False,
                    ) from error
                assert diagnostics is not None
                values.append(value)
                per_sequence.append({"sequence_index": sequence_index, **diagnostics})
                if not self._diagnostics_pass(value, diagnostics):
                    failure_reasons.add("NONFINITE_OR_DTYPE_BASE_SEQUENCE")
            mean = math.fsum(values) / len(values) if all(math.isfinite(v) for v in values) else None
            batch_losses.append(mean)
            batch_rows.append({
                "batch_index": batch_index, "per_sequence": per_sequence,
                "base_center_mean_ce": mean,
                "passed": mean is not None and all(
                    self._diagnostics_pass(value, row)
                    for value, row in zip(values, per_sequence)
                ),
            })
        passed = not failure_reasons and all(value is not None for value in batch_losses)
        evidence = {
            "schema": "hfw_w560_cpu_fp64_all_search_base_preflight_v1",
            "status": (
                "PASS_FINITE_CPU_FP64_ALL_SEARCH_BASE"
                if passed else "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT"
            ),
            "failure_reasons": sorted(failure_reasons),
            "batch_indices": list(range(len(self.calibration))),
            "logical_batch_shape": list(self.expected_shape[1:]),
            "physical_microbatch_shape": [1, self.expected_shape[2]],
            "checkpoint_candidate_dtype": "float32",
            "evaluation_parameter_dtype": "float64",
            "forward_device": "cpu", "forward_autocast_dtype": None,
            "logits_dtype": "float64", "hidden_dtype": "float64",
            "persistent_state_dtype": "float64", "ce_dtype": "float64",
            "final_norm_algorithm": "maxabs_scaled_fp64_layer_norm_rescaled_epsilon_v1",
            "base_candidate_flat_sha256": self._flat_sha256(flat),
            "batches": batch_rows,
            "base_full_search_batch_losses": batch_losses,
            "base_full_search_mean_ce": (
                math.fsum(batch_losses) / len(batch_losses) if passed else None
            ),
            "preflight_logical_evaluations": self.preflight_logical_evaluations,
            "preflight_physical_model_forwards": self.preflight_physical_model_forwards,
            "maximum_observed_rss_bytes": self.maximum_observed_rss_bytes,
        }
        if not passed:
            raise DeterministicNumericalFailure(
                "BASE_CENTER_PREFLIGHT", {
                    "reason_code": "BASE_CENTER_NUMERICAL_OR_DTYPE_GATE_FAILED",
                    "preflight": evidence,
                }, solver_artifacts_created=False,
            )
        self._current_full_bank[group] = {
            "flat_sha256": evidence["base_candidate_flat_sha256"],
            "batch_losses": list(batch_losses),
            "mean_ce": evidence["base_full_search_mean_ce"],
        }
        return evidence

    def loss(self, group: str, flat: torch.Tensor, batch_index: int) -> float:
        parameters = self.group_mapping(group, flat)
        raw = self.calibration[batch_index]
        per_sequence = []
        first_nonfinite = None
        for sequence_index in range(int(raw.shape[0])):
            value, diagnostics = self._evaluate_sequence(
                parameters, raw[sequence_index:sequence_index + 1],
                preflight=False, include_diagnostics=False,
            )
            if diagnostics is not None and not (
                diagnostics["logits_dtype_matches"]
                and diagnostics["hidden_dtype_matches"]
                and diagnostics["persistent_state_dtype_matches"]
                and diagnostics["ce_dtype"] == "float64"
            ):
                raise DeterministicNumericalFailure(
                    "FORWARD_DTYPE", {
                        "reason_code": "OBSERVED_FP64_SHADOW_DTYPE_MISMATCH",
                        "group": group, "batch_index": int(batch_index),
                        "sequence_index": sequence_index, "sequence": diagnostics,
                    }, solver_artifacts_created=True,
                )
            if not math.isfinite(value) and first_nonfinite is None:
                first_nonfinite = {
                    "group": group, "batch_index": int(batch_index),
                    "sequence_index": sequence_index, "sequence": diagnostics,
                }
            per_sequence.append(value)
        self.logical_evaluations += 1
        self.phase_logical_evaluations[self._active_phase] += 1
        if first_nonfinite is not None:
            self.last_nonfinite_diagnostics = first_nonfinite
            return float("inf")
        return math.fsum(per_sequence) / len(per_sequence)

    def acceptance_gate(
        self, group: str, current: torch.Tensor, candidate: torch.Tensor, iteration: int,
    ) -> Mapping[str, object]:
        current_sha = self._flat_sha256(current)
        candidate_sha = self._flat_sha256(candidate)
        cached = self._current_full_bank.get(group)
        if cached is None or cached["flat_sha256"] != current_sha:
            raise RuntimeError("acceptance gate current/full-bank cache drift")
        previous_phase = self._active_phase
        self._active_phase = self.PHASE_ACCEPTANCE_GATE
        before_logical = self.phase_logical_evaluations[self.PHASE_ACCEPTANCE_GATE]
        before_physical = self.phase_physical_model_forwards[self.PHASE_ACCEPTANCE_GATE]
        losses = []
        try:
            for batch_index in range(len(self.calibration)):
                value = self.loss(group, candidate, batch_index)
                losses.append(value if math.isfinite(value) else None)
        finally:
            self._active_phase = previous_phase
        logical = self.phase_logical_evaluations[self.PHASE_ACCEPTANCE_GATE] - before_logical
        physical = self.phase_physical_model_forwards[self.PHASE_ACCEPTANCE_GATE] - before_physical
        all_finite = all(value is not None for value in losses)
        candidate_mean = math.fsum(losses) / len(losses) if all_finite else None
        improves = bool(all_finite and candidate_mean < float(cached["mean_ce"]))
        passed = all_finite and improves
        record = {
            "schema": "hfw_w560_cpu_fp64_all_search_acceptance_gate_v1",
            "gate_index": len(self.acceptance_gate_records),
            "iteration": int(iteration), "group": group,
            "current_flat_sha256": current_sha,
            "candidate_flat_sha256": candidate_sha,
            "batch_indices": list(range(len(self.calibration))),
            "candidate_batch_losses": losses,
            "current_full_search_mean_ce": float(cached["mean_ce"]),
            "candidate_full_search_mean_ce": candidate_mean,
            "all_finite": all_finite,
            "strict_full_search_mean_improvement": improves,
            "passed": passed,
            "logical_evaluations": logical,
            "physical_model_forwards": physical,
            "current_mutation_or_artifact_write_before_gate": False,
        }
        if logical != len(self.calibration) or physical != len(self.calibration) * self.expected_shape[1]:
            raise RuntimeError("acceptance gate fixed all-bank accounting drift")
        self.acceptance_gate_records.append(record)
        if passed:
            self.acceptance_gate_passes += 1
            self._current_full_bank[group] = {
                "flat_sha256": candidate_sha,
                "batch_losses": list(losses), "mean_ce": candidate_mean,
            }
        else:
            self.acceptance_gate_failures += 1
        return record

    def begin_terminal_ranking(self) -> None:
        if self._active_phase != self.PHASE_SEARCH:
            raise RuntimeError("objective phase drift before terminal ranking")
        self._active_phase = self.PHASE_TERMINAL

    def receipt(self) -> Mapping[str, object]:
        return {
            "schema": "hfw_w560_cpu_fp64_stable_microbatch_objective_receipt_v1",
            "device": "cpu", "checkpoint_candidate_dtype": "float32",
            "evaluation_parameter_dtype": "float64",
            "floating_buffer_dtype": "float64", "persistent_state_dtype": "float64",
            "forward_autocast_dtype": None, "logits_dtype": "float64",
            "hidden_dtype": "float64", "ce_dtype": "float64",
            "final_norm_algorithm": "maxabs_scaled_fp64_layer_norm_rescaled_epsilon_v1",
            "canonical_fp32_state_storage_contract_replaced_in_proposal_shadow": True,
            "canonical_model_source_or_checkpoint_mutated": False,
            "candidate_f32_widened_to_fp64_for_every_score": True,
            "logical_batch_shape": list(self.expected_shape[1:]),
            "physical_microbatch_shape": [1, self.expected_shape[2]],
            "logical_evaluations_excluding_preflight": self.logical_evaluations,
            "physical_model_forwards_excluding_preflight": self.physical_model_forwards,
            "phase_logical_evaluations": dict(self.phase_logical_evaluations),
            "phase_physical_model_forwards": dict(self.phase_physical_model_forwards),
            "preflight_logical_evaluations": self.preflight_logical_evaluations,
            "preflight_physical_model_forwards": self.preflight_physical_model_forwards,
            "total_logical_evaluations_including_preflight": (
                self.logical_evaluations + self.preflight_logical_evaluations
            ),
            "total_physical_model_forwards_including_preflight": (
                self.physical_model_forwards + self.preflight_physical_model_forwards
            ),
            "acceptance_gate_attempts": len(self.acceptance_gate_records),
            "acceptance_gate_passes": self.acceptance_gate_passes,
            "acceptance_gate_failures": self.acceptance_gate_failures,
            "acceptance_gate_records": list(self.acceptance_gate_records),
            "float64_logits_observed": self.float64_logits_observed,
            "float64_hidden_observed": self.float64_hidden_observed,
            "float64_persistent_state_observed": self.float64_persistent_state_observed,
            "float64_ce_evaluations": self.float64_ce_evaluations,
            "physical_model_forwards_per_logical_evaluation": self.expected_shape[1],
            "maximum_observed_rss_bytes": self.maximum_observed_rss_bytes,
            "inference_only": True, "optimizer_steps": 0, "autograd_calls": 0,
        }


class WorkerTraceRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.count = 0

    def append(
        self, *, proposal_sha256: str, delta_sha256: str,
        child_checkpoint_sha256: str | None, accepted: bool,
        reason: str, record: Mapping[str, object],
    ) -> str:
        event = {
            "schema": "hfw_w560_daytona_cpu_provisional_solver_event_v1",
            "event_index": self.count,
            "proposal_sha256": proposal_sha256,
            "delta_sha256": delta_sha256,
            "artifact_only_child_sha256": child_checkpoint_sha256,
            "provisional_search_accept": bool(accepted),
            "checkpoint_or_head_written": False,
            "reason": reason,
            "record": dict(record),
        }
        payload = canonical_json_bytes(event)
        digest = sha256_bytes(payload)
        write_exclusive(self.root / f"{self.count:04d}-{digest}.event.json", payload)
        self.count += 1
        return digest


@contextmanager
def artifact_only_children(branch_module):
    original = branch_module._materialize_child

    def project(
        objective, group: str, flat: torch.Tensor, path: Path,
        immutable_parent_sha256: str, immediate_parent_child_sha256: str,
        chain_sha256: str,
    ) -> str:
        match = re.fullmatch(r"materialized_child_(\d+)\.pt", path.name)
        if match is None:
            raise RuntimeError("unexpected artifact-only child filename")
        accept_index = int(match.group(1))
        cumulative = hash_file(path.parent / f"accepted_{accept_index:02d}_delta.f32le")
        exact_payload = branch_module._tensor_bytes(flat)
        exact = write_exclusive(
            path.parent / f"accepted_{accept_index:02d}_group_flat.f32le", exact_payload,
        )
        expected_delta = hashlib.sha256(
            branch_module._tensor_bytes(flat - objective.flat_group(group))
        ).hexdigest()
        if cumulative["sha256"] != expected_delta:
            raise RuntimeError("artifact-only cumulative delta relation drift")
        descriptor = {
            "schema": "hfw_w560_daytona_cpu_artifact_only_child_v1",
            "artifact_projection_only": True,
            "group": group,
            "immutable_parent_checkpoint_sha256": immutable_parent_sha256,
            "immediate_parent_artifact_sha256": immediate_parent_child_sha256,
            "accepted_chain_sha256": chain_sha256,
            "cumulative_group_delta": cumulative,
            "exact_candidate_group_flat": exact,
            "full_checkpoint_materialized": False,
            "checkpoint_or_lineage_write_authorized": False,
            "mps_duplicate_verify_required": True,
        }
        return write_json_exclusive(path, descriptor)["sha256"]

    branch_module._materialize_child = project
    try:
        yield
    finally:
        branch_module._materialize_child = original


def build_accepted_candidate_bundle(
    branch_root: Path, output_root: Path, group: str, accepts: int,
    gate_records: list[Mapping[str, object]], expected_flat_size_bytes: int,
) -> tuple[Mapping[str, object], list[Mapping[str, object]]]:
    passing_gates = [gate for gate in gate_records if gate.get("passed") is True]
    if len(passing_gates) != accepts:
        raise RuntimeError("accepted candidate bundle gate count drift")
    bundle_path = output_root / "accepted_candidates.f32le.bundle"
    descriptor = os.open(
        bundle_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    entries: list[Mapping[str, object]] = []
    bundle_digest = hashlib.sha256()
    offset = 0
    try:
        for accept_index, gate in enumerate(passing_gates, start=1):
            accepted_path = branch_root / f"accepted_{accept_index:02d}.json"
            accepted, accepted_identity = read_json(accepted_path)
            child_path = branch_root / f"materialized_child_{accept_index:02d}.pt"
            child, child_identity = read_json(child_path)
            flat_path = branch_root / f"accepted_{accept_index:02d}_group_flat.f32le"
            flat_identity = hash_file(flat_path)
            declared_flat = child.get("exact_candidate_group_flat")
            if (
                accepted.get("schema") != "hfw_post_training_accepted_delta_v1"
                or accepted.get("accept_index") != accept_index
                or accepted.get("pair_index") != gate.get("iteration")
                or accepted.get("group") != group
                or child.get("schema")
                != "hfw_w560_daytona_cpu_artifact_only_child_v1"
                or child.get("group") != group
                or not isinstance(declared_flat, Mapping)
                or declared_flat.get("sha256") != flat_identity["sha256"]
                or int(declared_flat.get("size_bytes", -1))
                != int(flat_identity["size_bytes"])
                or int(flat_identity["size_bytes"]) != expected_flat_size_bytes
                or gate.get("candidate_flat_sha256") != flat_identity["sha256"]
            ):
                raise RuntimeError("accepted candidate bundle provenance drift")
            source = os.open(flat_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                remaining = expected_flat_size_bytes
                while remaining:
                    chunk = os.read(source, min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError("short accepted candidate flat read")
                    written = 0
                    while written < len(chunk):
                        count = os.write(descriptor, chunk[written:])
                        if count <= 0:
                            raise RuntimeError("short accepted candidate bundle write")
                        written += count
                    bundle_digest.update(chunk)
                    remaining -= len(chunk)
                if os.read(source, 1):
                    raise RuntimeError("accepted candidate flat grew during bundle write")
            finally:
                os.close(source)
            entries.append({
                "candidate_order": accept_index,
                "accept_index": accept_index,
                "pair_index": int(accepted["pair_index"]),
                "group": group,
                "offset_bytes": offset,
                "size_bytes": expected_flat_size_bytes,
                "group_flat_sha256": flat_identity["sha256"],
                "accepted_record_sha256": accepted_identity["sha256"],
                "artifact_only_child_sha256": child_identity["sha256"],
                "cumulative_delta_sha256": accepted["delta_sha256"],
                "candidate_cpu_full_search_mean_ce": gate[
                    "candidate_full_search_mean_ce"
                ],
            })
            offset += expected_flat_size_bytes
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    fsync_directory(output_root)
    bundle_identity = hash_file(bundle_path, bundle_digest.hexdigest())
    if int(bundle_identity["size_bytes"]) != accepts * expected_flat_size_bytes:
        raise RuntimeError("accepted candidate bundle size drift")
    return bundle_identity, entries


def _load_vendor(snapshot_root: Path):
    sys.path.insert(0, str(snapshot_root))
    branch = importlib.import_module(f"{VENDOR_PACKAGE}.branch_search")
    bridge = importlib.import_module(f"{VENDOR_PACKAGE}.production_bridge")
    manifest = importlib.import_module(f"{VENDOR_PACKAGE}.manifest")
    return branch, bridge, manifest


def _worker_protocol(snapshot_root: Path, campaign, task, branch):
    protocol_path = (
        snapshot_root / "vendor/hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver"
        / "config/production_protocol.json"
    )
    protocol, _ = read_json(protocol_path)
    derived = deepcopy(protocol)
    derived["parent"]["checkpoint_sha256"] = campaign["base_checkpoint_sha256"]
    derived["frozen_source"]["source_root"] = str(snapshot_root / "source")
    derived["search"]["branch_seeds"][task["group"]] = int(task["seed"])
    if int(derived["search"]["model_evaluation_budget"]) != int(
        campaign["solver"]["logical_search_evaluations"]
    ):
        raise RuntimeError("vendor logical SEARCH budget drift")
    if int(derived["search"]["pairs_per_branch"]) != int(
        campaign["solver"]["iterations"]
    ):
        raise RuntimeError("vendor iteration count drift")
    return derived


def run_worker(
    snapshot_root: Path, task_index: int, expected_campaign_sha256: str,
    scratch_root: Path, output_root: Path,
) -> Mapping[str, object]:
    started = time.monotonic()
    campaign, campaign_identity = read_json(
        snapshot_root / "CAMPAIGN.json", expected_campaign_sha256,
    )
    manifest, manifest_identity = read_json(snapshot_root / "SNAPSHOT_MANIFEST.json")
    if campaign.get("schema") != "hfw_w560_daytona_cpu100_campaign_v1":
        raise RuntimeError("campaign schema drift")
    if manifest.get("campaign_sha256") != campaign_identity["sha256"]:
        raise RuntimeError("snapshot/campaign binding drift")
    manifest_core = dict(manifest)
    for key in ("snapshot_manifest_core_sha256", "campaign_sha256", "campaign_size_bytes"):
        manifest_core.pop(key, None)
    if sha256_bytes(canonical_json_bytes(manifest_core)) != manifest["snapshot_manifest_core_sha256"]:
        raise RuntimeError("snapshot manifest core identity drift")
    if manifest["snapshot_manifest_core_sha256"] != campaign["snapshot_manifest_core_sha256"]:
        raise RuntimeError("campaign snapshot core binding drift")
    validate_inventory(snapshot_root, manifest["files"])
    if any(path.exists() for path in (
        snapshot_root / "verify.u32le", snapshot_root / "lineage",
        snapshot_root / "hfw_dist/promote_mps.py",
    )):
        raise RuntimeError("worker snapshot received VERIFY, lineage, or promoter")
    if not 0 <= task_index < int(campaign["worker_count"]):
        raise RuntimeError("task index outside campaign")
    task = campaign["tasks"][task_index]
    if int(task["task_index"]) != task_index:
        raise RuntimeError("campaign task index drift")
    task_sha = sha256_bytes(canonical_json_bytes(task))
    forbidden_names = {
        "DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN", "DAYTONA_TOKEN",
        "DTN_API_KEY", "DTN_JWT_TOKEN", "DTN_TOKEN",
    }
    forbidden_environment = sorted(key for key in os.environ if key in forbidden_names)
    credential_value_pattern = re.compile(r"dtn_[0-9a-f]{32,}", re.IGNORECASE)
    forbidden_value_pattern_matches = sum(
        1 for value in os.environ.values()
        if credential_value_pattern.search(value) is not None
    )
    if forbidden_environment or forbidden_value_pattern_matches:
        raise RuntimeError(
            "cloud credential environment leaked to worker: "
            f"forbidden_name_count={len(forbidden_environment)} "
            f"forbidden_value_pattern_count={forbidden_value_pattern_matches}"
        )
    credential_environment_scan = {
        "forbidden_names_checked": sorted(forbidden_names),
        "forbidden_name_matches": 0,
        "forbidden_value_pattern": "dtn_ followed by at least 32 hexadecimal characters",
        "forbidden_value_pattern_matches": 0,
    }
    sealed_environment = campaign["proposal_lane"]["sealed_environment"]
    observed_environment = {
        "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
        "platform_system": platform.system(), "machine": platform.machine(),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
    }
    for key in (
        "python_major_minor", "platform_system", "machine", "torch_version",
        "torch_cuda_version", "numpy_version",
    ):
        if observed_environment[key] != sealed_environment[key]:
            raise RuntimeError(f"sealed worker environment drift: {key}")
    threads = int(campaign["proposal_lane"]["torch_threads"])
    interop_threads = int(campaign["proposal_lane"]["interop_threads"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    branch, bridge, vendor_manifest = _load_vendor(snapshot_root)
    vendor_identity = hash_file(Path(branch.__file__), campaign["solver"]["vendor_branch_search_sha256"])
    loaded, _ = build_loaded_parent(
        snapshot_root, snapshot_root / "checkpoint.pt",
        str(campaign["base_checkpoint_sha256"]),
        str(campaign["base_model_state_digest"]), torch.device("cpu"),
        bridge, vendor_manifest,
    )
    if set(loaded.groups) != set(campaign["group_order"]):
        raise RuntimeError("loaded parameter group inventory drift")
    search_shape = tuple(int(value) for value in campaign["search_bank_shape"])
    expected_search_size = math.prod(search_shape) * np.dtype("<u4").itemsize
    search_identity = hash_file(snapshot_root / "search.u32le", str(campaign["search_bank_sha256"]))
    if int(search_identity["size_bytes"]) != expected_search_size:
        raise RuntimeError("SEARCH shape/byte-size binding drift")
    search = np.memmap(snapshot_root / "search.u32le", mode="r", dtype="<u4", shape=search_shape)
    batches = [torch.from_numpy(np.array(search[index], dtype=np.int64, copy=True)) for index in range(search_shape[0])]
    objective = CPUFP64StableMicrobatchObjective(
        loaded, batches, torch.device("cpu"), search_shape,
    )
    group = str(task["group"])
    base_flat = objective.flat_group(group)
    if base_flat.numel() != int(task["expected_group_elements"]):
        raise RuntimeError("task group element count drift")
    protocol = _worker_protocol(snapshot_root, campaign, task, branch)
    output_root.mkdir(parents=True, exist_ok=False)
    proposal_lane = campaign["proposal_lane"]
    expected_lane = {
        "candidate_artifact_dtype": "float32",
        "candidate_codec_dtype": "float32",
        "forward_device": "cpu",
        "forward_autocast_enabled": False,
        "forward_autocast_dtype": None,
        "evaluation_parameter_dtype": "float64",
        "floating_buffer_dtype": "float64",
        "evaluation_floating_buffer_dtype": "float64",
        "persistent_state_dtype": "float64",
        "evaluation_persistent_state_dtype": "float64",
        "initial_compressed_state_dtype": "float64",
        "hidden_dtype": "float64",
        "logits_dtype": "float64",
        "ce_dtype": "float64",
        "distinct_fp64_shadow_state_surrogate": True,
            "canonical_checkpoint_and_model_source_bytes_untouched": True,
            "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
            "proposal_shadow_replaces_canonical_fp32_state_storage": True,
        "raw_ce_cross_lane_comparable": False,
        "mps_sole_promotion_authority": True,
    }
    for key, expected in expected_lane.items():
        if proposal_lane.get(key) != expected:
            raise RuntimeError(f"sealed CPU numerical lane drift: {key}")
    memory_limit = int(proposal_lane["memory_limit_gib"]) * 1024 ** 3
    failure_policy = campaign["worker_failure_policy"]
    if (
        failure_policy.get("machine_readable_filename") != FAILURE_FILENAME
        or failure_policy.get("schema") != FAILURE_SCHEMA
        or int(failure_policy.get("deterministic_numerical_exit_code", -1))
        != DETERMINISTIC_NUMERICAL_EXIT_CODE
        or failure_policy.get("deterministic_numerical_retry_policy")
        != "TERMINAL_DO_NOT_RETRY"
    ):
        raise RuntimeError("sealed deterministic numerical failure policy drift")

    def record_deterministic_numerical_failure(
        error: DeterministicNumericalFailure,
    ) -> Mapping[str, object]:
        allowed_stages = tuple(campaign["worker_failure_policy"]["allowed_stages"])
        if error.stage not in allowed_stages:
            raise RuntimeError("deterministic numerical failure stage is not campaign-bound")
        reason_code = error.diagnostics.get("reason_code")
        allowed_reasons = campaign["worker_failure_policy"][
            "terminal_reason_codes_by_stage"
        ][error.stage]
        if reason_code not in allowed_reasons:
            raise RuntimeError("deterministic numerical reason code is not campaign-bound")
        failure_peak_rss = deterministic_failure_peak_rss(
            objective.maximum_observed_rss_bytes, error.diagnostics,
        )
        diagnostics = {
            **error.diagnostics,
            "maximum_observed_rss_bytes": failure_peak_rss,
            "memory_limit_bytes": memory_limit,
            "peak_rss_strictly_below_memory_limit": (
                failure_peak_rss < memory_limit
            ),
            "preflight_logical_evaluations": objective.preflight_logical_evaluations,
            "preflight_physical_model_forwards": objective.preflight_physical_model_forwards,
            "solver_logical_evaluations_completed": objective.logical_evaluations,
            "solver_physical_model_forwards_completed": objective.physical_model_forwards,
            "acceptance_gate_attempts_completed": len(objective.acceptance_gate_records),
            "acceptance_gate_passes_completed": objective.acceptance_gate_passes,
            "acceptance_gate_failures_completed": objective.acceptance_gate_failures,
            "phase_logical_evaluations": dict(objective.phase_logical_evaluations),
            "phase_physical_model_forwards": dict(objective.phase_physical_model_forwards),
            "float64_logits_observed": objective.float64_logits_observed,
            "float64_hidden_observed": objective.float64_hidden_observed,
            "float64_persistent_state_observed": objective.float64_persistent_state_observed,
            "float64_ce_evaluations": objective.float64_ce_evaluations,
        }
        policy = failure_policy
        failure = {
            "schema": policy["schema"],
            "status": policy["deterministic_numerical_status"],
            "classification": policy["deterministic_numerical_classification"],
            "retry_policy": policy["deterministic_numerical_retry_policy"],
            "stage": error.stage,
            "campaign_sha256": campaign_identity["sha256"],
            "snapshot_manifest_sha256": manifest_identity["sha256"],
            "snapshot_manifest_core_sha256": manifest["snapshot_manifest_core_sha256"],
            "task_sha256": task_sha,
            "task_index": task_index,
            "worker_id": task["worker_id"],
            "wave_index": int(campaign["wave_index"]),
            "group": group,
            "seed": int(task["seed"]),
            "base_head_record_sha256": campaign["base_head_record_sha256"],
            "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
            "base_model_state_digest": campaign["base_model_state_digest"],
            "numerical_lane": expected_lane,
            "environment": observed_environment,
            "diagnostics": diagnostics,
            "solver_artifacts_created": error.solver_artifacts_created,
            "worker_checkpoint_or_head_writes": False,
            "mps_verify_opened": False,
            "cloud_credentials_available_to_worker": False,
            "credential_environment_scan": credential_environment_scan,
            "training_or_backprop": False,
            "optimizer_steps": 0,
        }
        identity = write_json_exclusive(output_root / policy["machine_readable_filename"], failure)
        error.failure_identity = identity
        return identity

    try:
        base_center_preflight = objective.preflight_all_base_batches(group, base_flat)
    except DeterministicNumericalFailure as error:
        record_deterministic_numerical_failure(error)
        raise
    expected_preflight_logical = int(
        campaign["solver"]["base_center_preflight_logical_evaluations"]
    )
    expected_preflight_physical = int(
        campaign["solver"]["base_center_preflight_physical_model_forwards"]
    )
    if (
        objective.preflight_logical_evaluations != expected_preflight_logical
        or objective.preflight_physical_model_forwards != expected_preflight_physical
    ):
        raise RuntimeError("base-center preflight accounting drift")
    if objective.maximum_observed_rss_bytes >= memory_limit:
        failure = DeterministicNumericalFailure(
            "BASE_CENTER_PREFLIGHT", {
                "reason_code": "PREFLIGHT_RSS_LIMIT_REACHED",
                "preflight": base_center_preflight,
            }, solver_artifacts_created=False,
        )
        record_deterministic_numerical_failure(failure)
        raise failure
    scratch_root.mkdir(parents=True, exist_ok=False)
    registry = WorkerTraceRegistry(scratch_root / "events")
    search_started = time.monotonic()
    try:
        with artifact_only_children(branch):
            result = branch.run_independent_branch(
                objective, protocol, group, scratch_root / "branch", registry,
                materialize=True,
            )
    except DeterministicNumericalFailure as error:
        record_deterministic_numerical_failure(error)
        raise
    except RuntimeError as error:
        reason_by_message = {
            "nonfinite center loss": "VENDOR_REJECTED_NONFINITE_SEARCH_CENTER",
            "nonfinite budget-fill loss": "VENDOR_REJECTED_NONFINITE_SEARCH_BUDGET_FILL",
        }
        if str(error) not in reason_by_message:
            raise
        failure = DeterministicNumericalFailure(
            "SEARCH_CENTER", {
                "reason_code": "NONFINITE_SEARCH_OBJECTIVE",
                "reason": reason_by_message[str(error)],
                "objective": objective.last_nonfinite_diagnostics,
            }, solver_artifacts_created=True,
        )
        record_deterministic_numerical_failure(failure)
        raise failure from error
    search_seconds = time.monotonic() - search_started
    validation = branch.validate_branch_artifact_chain(
        protocol, scratch_root / "branch" / group, group,
        include_final_delta_bytes=False, validate_child_payload=False,
    )
    expected_search_logical = int(campaign["solver"]["logical_search_evaluations"])
    expected_search_physical = int(campaign["solver"]["physical_search_model_forwards"])
    if result.proposals != int(protocol["search"]["pairs_per_branch"]) or int(result.receipt["forward_calls"]) != expected_search_logical:
        raise RuntimeError("vendor SEARCH schedule drift")
    if (
        objective.phase_logical_evaluations[objective.PHASE_SEARCH] != expected_search_logical
        or objective.phase_physical_model_forwards[objective.PHASE_SEARCH]
        != expected_search_physical
    ):
        raise RuntimeError("SEARCH logical/physical forward accounting drift")
    ranking_started = time.monotonic()
    try:
        objective.begin_terminal_ranking()
        full_search_losses = [
            objective.loss(group, result.final_flat, index)
            for index in range(search_shape[0])
        ]
    except DeterministicNumericalFailure as error:
        record_deterministic_numerical_failure(error)
        raise
    ranking_seconds = time.monotonic() - ranking_started
    expected_rank_logical = int(campaign["solver"]["terminal_full_search_ranking_logical_evaluations"])
    expected_rank_physical = int(campaign["solver"]["terminal_full_search_ranking_physical_model_forwards"])
    if (
        objective.phase_logical_evaluations[objective.PHASE_TERMINAL]
        != expected_rank_logical
        or objective.phase_physical_model_forwards[objective.PHASE_TERMINAL]
        != expected_rank_physical
    ):
        raise RuntimeError("terminal ranking logical/physical forward accounting drift")
    gate_attempts = len(objective.acceptance_gate_records)
    gate_passes = objective.acceptance_gate_passes
    gate_failures = objective.acceptance_gate_failures
    maximum_gate_attempts = int(campaign["solver"]["acceptance_gate_maximum_attempts"])
    gate_logical_per_attempt = int(
        campaign["solver"]["acceptance_gate_logical_evaluations_per_attempt"]
    )
    gate_physical_per_attempt = int(
        campaign["solver"]["acceptance_gate_physical_model_forwards_per_attempt"]
    )
    if not (
        0 <= int(result.accepts) == gate_passes <= gate_attempts <= maximum_gate_attempts
        and gate_failures == gate_attempts - gate_passes
        and objective.phase_logical_evaluations[objective.PHASE_ACCEPTANCE_GATE]
        == gate_attempts * gate_logical_per_attempt
        and objective.phase_physical_model_forwards[objective.PHASE_ACCEPTANCE_GATE]
        == gate_attempts * gate_physical_per_attempt
    ):
        raise RuntimeError("pre-mutation acceptance gate accounting/relation drift")
    total_logical = objective.logical_evaluations + objective.preflight_logical_evaluations
    total_physical = (
        objective.physical_model_forwards + objective.preflight_physical_model_forwards
    )
    expected_total_logical, expected_total_physical = expected_variable_totals(
        campaign["solver"], gate_attempts,
    )
    if (
        total_logical != expected_total_logical
        or total_physical != expected_total_physical
    ):
        raise RuntimeError("total preflight/SEARCH/ranking accounting drift")
    if (
        objective.float64_logits_observed != total_physical
        or objective.float64_hidden_observed != total_physical
        or objective.float64_persistent_state_observed != total_physical
        or objective.float64_ce_evaluations != total_physical
    ):
        raise RuntimeError("not every CPU forward used the sealed FP64 shadow lane")
    try:
        require_finite_terminal_ranking(full_search_losses, objective)
    except DeterministicNumericalFailure as error:
        record_deterministic_numerical_failure(error)
        raise
    terminal_mean = math.fsum(full_search_losses) / len(full_search_losses)
    duplicate_tolerance = float(
        proposal_lane["fp64_full_bank_duplicate_absolute_tolerance"]
    )
    try:
        terminal_duplicate = validate_terminal_duplicate(
            objective, group, result.final_flat, full_search_losses, duplicate_tolerance,
        )
    except DeterministicNumericalFailure as error:
        record_deterministic_numerical_failure(error)
        raise
    final_payload = branch._tensor_bytes(result.final_flat)
    if final_payload != objective._flat_bytes(result.final_flat):
        raise RuntimeError("vendor terminal flat serialization differs from exact FP32 domain")
    final_roundtrip = torch.from_numpy(
        np.frombuffer(final_payload, dtype="<f4").copy()
    ).view_as(result.final_flat)
    if not torch.equal(final_roundtrip, result.final_flat):
        raise RuntimeError("terminal exact FP32 flat failed byte roundtrip")
    if len(final_payload) != int(task["expected_final_flat_size_bytes"]):
        raise RuntimeError("terminal group-flat size drift")
    final_identity = write_exclusive(
        output_root / "trajectory_final_group_flat.f32le", final_payload,
    )
    accepted_bundle_identity, accepted_candidates = build_accepted_candidate_bundle(
        scratch_root / "branch" / group,
        output_root,
        group,
        int(result.accepts),
        objective.acceptance_gate_records,
        int(task["expected_final_flat_size_bytes"]),
    )
    rss = max(objective.maximum_observed_rss_bytes, peak_rss_bytes())
    if rss >= memory_limit:
        failure = DeterministicNumericalFailure(
            "TERMINAL_RANKING", {
                "reason_code": "WORKER_RSS_LIMIT_REACHED_AFTER_SEARCH",
                "observed_peak_rss_bytes": rss,
                "memory_limit_bytes": memory_limit,
            }, solver_artifacts_created=True,
        )
        record_deterministic_numerical_failure(failure)
        raise failure
    no_training = result.receipt["runtime_guard"]
    if no_training.get("status") != "PASS_NO_TRAINING_RUNTIME_GUARD" or any(
        int(value) for key, value in no_training.items() if key.endswith("attempts")
    ):
        raise RuntimeError("vendor no-training guard failed")
    worker_result = {
        "schema": "hfw_w560_daytona_cpu_search_result_v1",
        "status": "PASS_PROVISIONAL_CPU_SEARCH_ONLY",
        "campaign_sha256": campaign_identity["sha256"],
        "snapshot_manifest_sha256": manifest_identity["sha256"],
        "snapshot_manifest_core_sha256": manifest["snapshot_manifest_core_sha256"],
        "task_sha256": task_sha,
        "task_index": task_index,
        "worker_id": task["worker_id"],
        "wave_index": int(campaign["wave_index"]),
        "base_head_record_sha256": campaign["base_head_record_sha256"],
        "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
        "base_model_state_digest": campaign["base_model_state_digest"],
        "group": group, "seed": int(task["seed"]),
        "group_elements": int(result.final_flat.numel()),
        "changed": final_identity["sha256"] != hashlib.sha256(branch._tensor_bytes(base_flat)).hexdigest(),
        "provisional_search_accepts": int(result.accepts),
        "pairs": int(result.proposals),
        "logical_search_evaluations": expected_search_logical,
        "physical_search_model_forwards": expected_search_physical,
        "terminal_ranking_logical_evaluations": expected_rank_logical,
        "terminal_ranking_physical_model_forwards": expected_rank_physical,
        "acceptance_gate_attempts": gate_attempts,
        "acceptance_gate_passes": gate_passes,
        "acceptance_gate_failures": gate_failures,
        "accepted_updates": int(result.accepts),
        "acceptance_gate_logical_evaluations": gate_attempts * gate_logical_per_attempt,
        "acceptance_gate_physical_model_forwards": gate_attempts * gate_physical_per_attempt,
        "solver_logical_evaluations_excluding_preflight": objective.logical_evaluations,
        "solver_physical_model_forwards_excluding_preflight": objective.physical_model_forwards,
        "base_center_preflight_logical_evaluations": objective.preflight_logical_evaluations,
        "base_center_preflight_physical_model_forwards": objective.preflight_physical_model_forwards,
        "total_logical_evaluations": (
            objective.logical_evaluations + objective.preflight_logical_evaluations
        ),
        "total_physical_model_forwards": (
            objective.physical_model_forwards + objective.preflight_physical_model_forwards
        ),
        "base_center_preflight": base_center_preflight,
        "base_full_search_batch_losses": base_center_preflight["base_full_search_batch_losses"],
        "base_full_search_mean_ce": base_center_preflight["base_full_search_mean_ce"],
        "terminal_full_search_batch_losses": full_search_losses,
        "terminal_full_search_mean_ce": terminal_mean,
        "terminal_full_search_duplicate": terminal_duplicate,
        "self_calibrated_full_search_improvement": (
            float(base_center_preflight["base_full_search_mean_ce"]) - terminal_mean
        ),
        "self_calibrated_improvement_semantics": "base_full_search_mean_ce_minus_terminal_full_search_mean_ce",
        "trajectory_final_group_flat": final_identity,
        "accepted_candidate_bundle": accepted_bundle_identity,
        "accepted_candidates": accepted_candidates,
        "accepted_candidate_count": len(accepted_candidates),
        "every_provisionally_accepted_candidate_in_bundle": True,
        "candidate_subset_selection_performed": False,
        "cpu_ranking_used_for_candidate_selection": False,
        "complete_proposal_chain_sha256": result.receipt["complete_proposal_chain_sha256"],
        "branch_receipt_sha256": validation["branch_receipt_sha256"],
        "runtime_guard_receipt_sha256": validation["runtime_guard_receipt_sha256"],
        "vendor_branch_search": vendor_identity,
        "objective_runtime": objective.receipt(),
        "environment": {
            **observed_environment,
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count_visible": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            **cpu_model_disclosure(),
            "peak_rss_bytes": rss, "memory_limit_bytes": memory_limit,
            "peak_rss_strictly_below_memory_limit": rss < memory_limit,
        },
        "timing": {
            "search_seconds": search_seconds, "ranking_seconds": ranking_seconds,
            "total_seconds": time.monotonic() - started,
        },
        "worker_checkpoint_or_head_writes": 0,
        "distinct_fp64_shadow_state_surrogate": True,
        "canonical_checkpoint_and_model_source_bytes_untouched": True,
        "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
        "proposal_shadow_replaces_canonical_fp32_state_storage": True,
        "raw_ce_cross_lane_comparable": False,
        "mps_sole_promotion_authority": True,
        "mps_verify_opened": False,
        "local_verify_available_to_worker": False,
        "cloud_credentials_available_to_worker": False,
        "credential_environment_scan": credential_environment_scan,
        "training_or_backprop": False,
        "optimizer_steps": 0,
        "provisional_not_held_out_or_promoted": True,
    }
    result_identity = write_json_exclusive(output_root / "CPU_SEARCH_RESULT.json", worker_result)
    return {**worker_result, "result_identity": result_identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_worker(
            args.snapshot_root.resolve(), args.task_index, args.campaign_sha256,
            args.scratch_root.resolve(), args.output_root.resolve(),
        )
    except DeterministicNumericalFailure as error:
        identity = getattr(error, "failure_identity", None)
        print(json.dumps({
            "status": "FAIL_DETERMINISTIC_NUMERICAL_FAILURE",
            "classification": "DETERMINISTIC_NUMERICAL_FAILURE",
            "retry_policy": "TERMINAL_DO_NOT_RETRY",
            "failure_filename": FAILURE_FILENAME,
            "failure_sha256": None if identity is None else identity["sha256"],
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(DETERMINISTIC_NUMERICAL_EXIT_CODE)
    print(json.dumps({
        "status": result["status"], "worker_id": result["worker_id"],
        "terminal_full_search_mean_ce": result["terminal_full_search_mean_ce"],
        "peak_rss_bytes": result["environment"]["peak_rss_bytes"],
        "result_sha256": result["result_identity"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
