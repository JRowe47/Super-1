from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .common import canonical_json_bytes, hash_file_same_fd


NUMERICAL_LANE = "mps_bfloat16_forward_cpu_float64_functional_normalization_v1"


def functional_effective_rms_cpu_float64(
    design: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen formula with only its float64 portion moved to CPU.

    The expression and loop order intentionally match the frozen vendor function.
    MPS tensors are copied to CPU before conversion because MPS cannot represent
    float64.  The scalar result is copied back to the caller's device.
    """
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


def adapter_source_identity() -> Mapping[str, object]:
    return hash_file_same_fd(Path(__file__))


def adapter_contract(config: Mapping[str, object]) -> Mapping[str, object]:
    lane = config.get("mps_local_lane", {})
    required = {
        "numerical_lane": NUMERICAL_LANE,
        "model_forward_device": "mps",
        "model_forward_autocast_dtype": "bfloat16",
        "functional_normalization_device": "cpu",
        "functional_normalization_dtype": "float64",
        "float64_transfer_order": "mps_to_cpu_then_cpu_float64",
        "vendor_solver_bytes_edited": False,
        "solver_settings_edited": False,
        "local_verify_role": "LOCAL_VERIFY",
        "local_verify_is_reused_optimization_data_not_held_out": True,
        "cloud_access": False,
        "remote_checkpoint_authority": False,
    }
    for key, expected in required.items():
        if lane.get(key) != expected:
            raise RuntimeError(f"MPS numerical-lane contract drift: {key}")
    if config.get("controller", {}).get("forward_device") != "mps":
        raise RuntimeError("MPS proposal lane requires forward_device=mps")
    if int(config.get("controller", {}).get("parallel_workers", -1)) != 1:
        raise RuntimeError("MPS proposal lane requires exactly one local worker")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    source = adapter_source_identity()
    if source["sha256"] != lane.get("adapter_source_sha256"):
        raise RuntimeError("MPS adapter source SHA drift")
    body = dict(lane)
    claimed = body.pop("contract_identity_sha256", None)
    observed = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if claimed != observed:
        raise RuntimeError("MPS adapter contract identity drift")
    return {
        "schema": "hfw_w560_mps_numerical_lane_validation_v1",
        "status": "PASS_EXPLICIT_MPS_BF16_FORWARD_CPU_FLOAT64_NORMALIZATION_LANE",
        "adapter_source": source,
        "contract_identity_sha256": observed,
        "cuda_bit_identical_claimed": False,
        "candidate_requires_duplicated_local_verify_before_local_promotion": True,
        "local_verify_is_reused_optimization_data_not_held_out": True,
    }


@contextmanager
def mps_functional_normalization(branch_module, expected_vendor_sha256: str):
    """Temporarily adapt the one unsupported vendor float64 operation."""
    vendor_path = Path(branch_module.__file__)
    before = hash_file_same_fd(vendor_path)
    if before["sha256"] != expected_vendor_sha256:
        raise RuntimeError("frozen vendor solver SHA drift before MPS adapter")
    original = branch_module._functional_effective_rms
    calls = {"count": 0}

    def adapted(design: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.device.type != "mps" or right.device.type != "mps":
            raise RuntimeError("MPS normalization adapter received non-MPS factor tensors")
        calls["count"] += 1
        return functional_effective_rms_cpu_float64(design, left, right)

    branch_module._functional_effective_rms = adapted
    try:
        yield calls
    finally:
        branch_module._functional_effective_rms = original
        after = hash_file_same_fd(vendor_path)
        if before["sha256"] != after["sha256"] or after["sha256"] != expected_vendor_sha256:
            raise RuntimeError("frozen vendor solver bytes changed during MPS adapter")


def make_mps_bfloat16_objective_class(bridge_module):
    base = bridge_module.W560ForwardOnlyObjective

    class MPSBFloat16ForwardOnlyObjective(base):
        def __init__(
            self, parent, calibration_batches: Sequence[torch.Tensor], device: torch.device,
        ):
            if device.type != "mps":
                raise RuntimeError("MPS objective received a non-MPS device")
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS objective cannot start because MPS is unavailable")
            super().__init__(parent, calibration_batches, device)
            self.mps_bfloat16_autocast_entries = 0
            self.mps_bfloat16_logits = 0
            self.forward_calls = 0

        def loss_state_vector(
            self, state: Mapping[str, torch.Tensor], batch_index: int,
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
                with torch.autocast(device_type="mps", dtype=torch.bfloat16, enabled=True):
                    if not torch.is_autocast_enabled("mps"):
                        raise RuntimeError("MPS bfloat16 autocast did not activate")
                    self.mps_bfloat16_autocast_entries += 1
                    output = torch.func.functional_call(
                        self.parent.model,
                        (parameters, buffers),
                        (inputs,), strict=True,
                    )
                self.forward_calls += 1
                if output.logits.dtype == torch.bfloat16:
                    self.mps_bfloat16_logits += 1
                per_token = F.cross_entropy(
                    output.logits.float().reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1), reduction="none",
                ).view(targets.shape)
                values = per_token.mean(dim=1)
                if not torch.isfinite(values).all():
                    raise RuntimeError("nonfinite per-sequence calibration loss")
                return tuple(float(value) for value in values.cpu().tolist())

        def numerical_lane_receipt(self) -> Mapping[str, object]:
            if self.forward_calls != self.mps_bfloat16_autocast_entries:
                raise RuntimeError("not every model forward entered MPS bfloat16 autocast")
            if self.forward_calls != self.mps_bfloat16_logits:
                raise RuntimeError("not every MPS model forward produced bfloat16 logits")
            return {
                "schema": "hfw_w560_mps_objective_runtime_receipt_v1",
                "numerical_lane": NUMERICAL_LANE,
                "forward_device": "mps",
                "autocast_dtype": "bfloat16",
                "forward_calls": self.forward_calls,
                "mps_bfloat16_autocast_entries": self.mps_bfloat16_autocast_entries,
                "bfloat16_logits_observed": self.mps_bfloat16_logits,
                "every_forward_produced_bfloat16_logits": True,
                "inference_only": True,
                "optimizer_steps": 0,
                "autograd_calls": 0,
            }

    return MPSBFloat16ForwardOnlyObjective
