from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from hfw_dist.cpu_proposal_worker import (
    CPUPhysicalMicrobatchObjective,
    DeterministicNumericalFailure,
    require_finite_terminal_ranking,
)


class TinyAutocastModel(torch.nn.Module):
    def __init__(
        self, *, nonfinite_timestep: int | None = None,
        force_fp32_logits: bool = False, operation_error: bool = False,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16, bias=False)
        self.nonfinite_timestep = nonfinite_timestep
        self.force_fp32_logits = force_fp32_logits
        self.operation_error = operation_error

    def forward(self, input_ids: torch.Tensor):
        if self.operation_error:
            raise RuntimeError("test-only deterministic operation failure")
        hidden = self.embedding(input_ids)
        logits = self.head(hidden)
        if self.nonfinite_timestep is not None:
            logits = logits.clone()
            logits[:, self.nonfinite_timestep] = float("nan")
        if self.force_fp32_logits:
            logits = logits.float()
        return SimpleNamespace(
            logits=logits, hidden_states=hidden, final_state={},
        )


def objective(
    *, nonfinite_timestep: int | None = None, force_fp32_logits: bool = False,
    operation_error: bool = False,
):
    model = TinyAutocastModel(
        nonfinite_timestep=nonfinite_timestep,
        force_fp32_logits=force_fp32_logits,
        operation_error=operation_error,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parent = SimpleNamespace(
        model=model,
        groups={"group": tuple(name for name, _ in model.named_parameters())},
    )
    batch = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    return CPUPhysicalMicrobatchObjective(
        parent, [batch], torch.device("cpu"), (1, 2, 4),
    )


class CpuBfloat16WorkerTests(unittest.TestCase):
    def test_preflight_requires_bfloat16_logits_and_float32_ce(self):
        value = objective()
        flat = value.flat_group("group")
        evidence = value.preflight_base_center("group", flat, 0)
        self.assertEqual(evidence["status"], "PASS_FINITE_CPU_BFLOAT16_BASE_CENTER")
        self.assertEqual(evidence["forward_autocast_dtype"], "bfloat16")
        self.assertEqual(evidence["ce_dtype"], "float32")
        self.assertEqual(len(evidence["per_sequence"]), 2)
        for row in evidence["per_sequence"]:
            self.assertEqual(row["logits_dtype"], "bfloat16")
            self.assertEqual(row["ce_dtype"], "float32")
            self.assertEqual(row["loss_class"], "FINITE")
            self.assertTrue(row["logits"]["all_finite"])
            self.assertTrue(row["hidden"]["all_finite"])
            self.assertTrue(row["per_token_loss"]["all_finite"])
        receipt = value.receipt()
        self.assertEqual(receipt["preflight_logical_evaluations"], 1)
        self.assertEqual(receipt["preflight_physical_model_forwards"], 2)
        self.assertEqual(receipt["total_logical_evaluations_including_preflight"], 1)
        self.assertEqual(receipt["total_physical_model_forwards_including_preflight"], 2)
        self.assertEqual(receipt["cpu_bfloat16_autocast_entries"], 2)
        self.assertEqual(receipt["bfloat16_logits_observed"], 2)
        self.assertEqual(receipt["float32_ce_evaluations"], 2)

    def test_nonfinite_preflight_is_machine_classifiable_before_search(self):
        value = objective(nonfinite_timestep=1)
        flat = value.flat_group("group")
        with self.assertRaises(DeterministicNumericalFailure) as captured:
            value.preflight_base_center("group", flat, 0)
        self.assertEqual(captured.exception.stage, "BASE_CENTER_PREFLIGHT")
        self.assertFalse(captured.exception.solver_artifacts_created)
        self.assertEqual(
            captured.exception.diagnostics["reason_code"],
            "BASE_CENTER_NUMERICAL_OR_DTYPE_GATE_FAILED",
        )
        evidence = captured.exception.diagnostics["preflight"]
        self.assertEqual(
            evidence["status"], "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT",
        )
        self.assertEqual(
            evidence["failure_reasons"],
            ["NONFINITE_LOGITS", "NONFINITE_MEAN_CE", "NONFINITE_PER_TOKEN_CE"],
        )
        for row in evidence["per_sequence"]:
            self.assertEqual(row["logits"]["first_nonfinite_timestep"], 1)
            self.assertEqual(row["per_token_loss"]["first_nonfinite_timestep"], 1)
            self.assertIsNone(row["hidden"]["first_nonfinite_timestep"])
            self.assertEqual(row["loss_class"], "NAN")
        self.assertEqual(value.logical_evaluations, 0)
        self.assertEqual(value.physical_model_forwards, 0)

    def test_preflight_classifies_logits_dtype_mismatch(self):
        value = objective(force_fp32_logits=True)
        flat = value.flat_group("group")
        with self.assertRaises(DeterministicNumericalFailure) as captured:
            value.preflight_base_center("group", flat, 0)
        self.assertEqual(captured.exception.stage, "BASE_CENTER_PREFLIGHT")
        evidence = captured.exception.diagnostics["preflight"]
        self.assertEqual(evidence["failure_reasons"], ["LOGITS_DTYPE_MISMATCH"])
        self.assertEqual(value.bfloat16_logits_observed, 0)
        self.assertEqual(value.float32_ce_evaluations, 2)

    def test_nonfinite_search_loss_retains_diagnostics_without_accepting(self):
        value = objective(nonfinite_timestep=1)
        flat = value.flat_group("group")
        self.assertEqual(value.loss("group", flat, 0), float("inf"))
        self.assertEqual(value.physical_model_forwards, 2)
        self.assertEqual(value.logical_evaluations, 1)
        self.assertEqual(
            value.last_nonfinite_diagnostics["sequence"]["loss_class"], "NAN",
        )

    def test_nonfinite_terminal_ranking_reaches_classified_envelope(self):
        value = objective(nonfinite_timestep=1)
        flat = value.flat_group("group")
        losses = [value.loss("group", flat, 0)]
        with self.assertRaises(DeterministicNumericalFailure) as captured:
            require_finite_terminal_ranking(losses, value)
        self.assertEqual(captured.exception.stage, "TERMINAL_RANKING")
        self.assertEqual(
            captured.exception.diagnostics["reason_code"],
            "NONFINITE_TERMINAL_RANKING",
        )
        self.assertTrue(captured.exception.solver_artifacts_created)
        self.assertEqual(value.logical_evaluations, 1)
        self.assertEqual(value.physical_model_forwards, 2)

    def test_preflight_operation_exception_is_deterministic_and_machine_readable(self):
        value = objective(operation_error=True)
        flat = value.flat_group("group")
        with self.assertRaises(DeterministicNumericalFailure) as captured:
            value.preflight_base_center("group", flat, 0)
        self.assertEqual(captured.exception.stage, "BASE_CENTER_PREFLIGHT")
        diagnostics = captured.exception.diagnostics
        self.assertEqual(
            diagnostics["reason_code"], "SEALED_PREFLIGHT_OPERATION_EXCEPTION",
        )
        evidence = diagnostics["preflight"]
        self.assertEqual(
            evidence["failure_reasons"], ["SEALED_PREFLIGHT_OPERATION_EXCEPTION"],
        )
        self.assertEqual(evidence["failed_sequence_index"], 0)
        self.assertEqual(evidence["operation_exception_type"], "RuntimeError")
        self.assertEqual(len(evidence["operation_exception_message_sha256"]), 64)
        self.assertEqual(value.preflight_logical_evaluations, 1)
        self.assertEqual(value.preflight_physical_model_forwards, 1)

    def test_post_preflight_logits_dtype_failure_is_terminal_and_artifact_bound(self):
        value = objective(force_fp32_logits=True)
        flat = value.flat_group("group")
        with self.assertRaises(DeterministicNumericalFailure) as captured:
            value.loss("group", flat, 0)
        self.assertEqual(captured.exception.stage, "FORWARD_DTYPE")
        self.assertEqual(
            captured.exception.diagnostics["reason_code"],
            "OBSERVED_LOGITS_DTYPE_MISMATCH",
        )
        self.assertTrue(captured.exception.solver_artifacts_created)


if __name__ == "__main__":
    unittest.main()
