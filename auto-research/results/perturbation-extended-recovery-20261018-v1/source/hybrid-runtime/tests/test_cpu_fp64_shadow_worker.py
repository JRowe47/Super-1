from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from hfw_dist.cpu_proposal_worker import (
    CPUFP64StableMicrobatchObjective,
    DeterministicNumericalFailure,
    build_accepted_candidate_bundle,
    deterministic_failure_peak_rss,
    expected_variable_totals,
    stable_fp64_layer_norm,
    validate_terminal_duplicate,
)
from hfw_dist.common import hash_file, write_exclusive, write_json_exclusive


class _Codec(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))

    def encode(self, hidden):
        return F.linear(hidden.float(), self.weight.float())

    def decode(self, persistent):
        return F.linear(persistent, self.weight.transpose(0, 1))


class _ToyShadowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 4)
        self.final_norm = nn.LayerNorm(4)
        self.state_width = 4
        self.identity_codec_mode = False
        self.state_codecs = nn.ModuleDict({
            name: _Codec(4) for name in ("fast", "segment", "document", "segment_sum")
        })

    def initial_state(self, batch_size, device, dtype=None):
        del dtype
        zero = lambda: torch.zeros(batch_size, 4, device=device, dtype=torch.float32)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
            "position": 0, "segment_updates": 0, "document_updates": 0,
        }

    def forward(self, input_ids):
        state = self.initial_state(input_ids.shape[0], input_ids.device)
        hidden_rows = []
        for index in range(input_ids.shape[1]):
            hidden = self.embedding(input_ids[:, index]) + self.state_codecs["fast"].decode(
                state["fast"]
            ) * 0.01
            state["fast"] = self.state_codecs["fast"].encode(hidden)
            state["segment"] = self.state_codecs["segment"].encode(hidden)
            state["document"] = self.state_codecs["document"].encode(hidden)
            state["segment_sum"] = self.state_codecs["segment_sum"].encode(hidden)
            hidden_rows.append(hidden)
        stacked = torch.stack(hidden_rows, dim=1)
        logits = F.linear(self.final_norm(stacked), self.embedding.weight)
        return SimpleNamespace(logits=logits, hidden_states=stacked, final_state=state)


def _objective(batch_count: int = 2):
    torch.manual_seed(0)
    model = _ToyShadowModel().float().eval()
    groups = {"group": tuple(name for name, _ in model.named_parameters())}
    parent = SimpleNamespace(model=model, groups=groups)
    batches = []
    for index in range(batch_count):
        first = torch.tensor([1 + index, 2, 3, 4], dtype=torch.long)
        second = torch.tensor([4, 3, 2, 1 + index], dtype=torch.long)
        batches.append(torch.stack((first, second)))
    value = CPUFP64StableMicrobatchObjective(
        parent, batches, torch.device("cpu"), (batch_count, 2, 4),
    )
    return value


class StableLayerNormTests(unittest.TestCase):
    def test_matches_native_fp64_on_moderate_values(self):
        module = nn.LayerNorm(4).double()
        values = torch.tensor(
            [[1.25, -2.5, 3.75, 0.5], [0.0, 0.1, -0.2, 0.3]],
            dtype=torch.float64,
        )
        expected = F.layer_norm(
            values, module.normalized_shape, module.weight, module.bias, module.eps,
        )
        observed = stable_fp64_layer_norm(module, values)
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)

    def test_huge_values_and_zero_row_remain_finite_and_scale_stable(self):
        module = nn.LayerNorm(4, elementwise_affine=False).double()
        huge = torch.tensor(
            [[1e197, -2e197, 3e197, -4e197], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        observed = stable_fp64_layer_norm(module, huge)
        scaled = stable_fp64_layer_norm(module, huge * 1e20)
        self.assertTrue(bool(torch.isfinite(observed).all()))
        self.assertTrue(bool(torch.isfinite(scaled).all()))
        torch.testing.assert_close(observed[0], scaled[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(observed[1], torch.zeros(4, dtype=torch.float64))


class FP64ShadowObjectiveTests(unittest.TestCase):
    def test_failure_peak_rss_includes_post_objective_process_peak(self):
        self.assertEqual(
            deterministic_failure_peak_rss(
                3_000_000_000,
                {"observed_peak_rss_bytes": 4_300_000_000},
            ),
            4_300_000_000,
        )
        self.assertEqual(
            deterministic_failure_peak_rss(
                3_000_000_000,
                {"observed_peak_rss_bytes": 2_900_000_000},
            ),
            3_000_000_000,
        )

    def test_retains_exact_f32_candidate_and_runs_every_shadow_tensor_fp64(self):
        value = _objective()
        flat = value.flat_group("group")
        self.assertEqual(flat.dtype, torch.float32)
        before = value._flat_bytes(flat)
        evidence = value.preflight_all_base_batches("group", flat)
        self.assertEqual(value._flat_bytes(flat), before)
        self.assertEqual(evidence["status"], "PASS_FINITE_CPU_FP64_ALL_SEARCH_BASE")
        self.assertEqual(evidence["preflight_logical_evaluations"], 2)
        self.assertEqual(evidence["preflight_physical_model_forwards"], 4)
        for batch in evidence["batches"]:
            for row in batch["per_sequence"]:
                self.assertEqual(row["logits_dtype"], "float64")
                self.assertEqual(row["hidden_dtype"], "float64")
                self.assertEqual(row["persistent_state_floating_dtypes"], ["float64"])
                self.assertEqual(row["ce_dtype"], "float64")

    def test_gate_completes_full_bank_and_requires_strict_mean_improvement(self):
        value = _objective()
        flat = value.flat_group("group")
        value.preflight_all_base_batches("group", flat)
        value._current_full_bank["group"] = {
            **value._current_full_bank["group"], "mean_ce": 1_000.0,
        }
        gate = value.acceptance_gate("group", flat, flat.clone(), 0)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["logical_evaluations"], 2)
        self.assertEqual(gate["physical_model_forwards"], 4)
        self.assertEqual(value.acceptance_gate_passes, 1)

    def test_nonfinite_candidate_is_rejected_after_fixed_full_bank_not_fatal(self):
        value = _objective()
        flat = value.flat_group("group")
        value.preflight_all_base_batches("group", flat)
        candidate = flat.clone()
        candidate[0] = float("inf")
        gate = value.acceptance_gate("group", flat, candidate, 0)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["all_finite"])
        self.assertEqual(gate["logical_evaluations"], 2)
        self.assertEqual(gate["physical_model_forwards"], 4)
        self.assertEqual(value.acceptance_gate_failures, 1)

    def test_fp64_diagnostics_do_not_downcast_huge_finite_values(self):
        value = torch.tensor([[[1e197, -1e197], [2.0, -3.0]]], dtype=torch.float64)
        evidence = CPUFP64StableMicrobatchObjective._tensor_time_diagnostics(value)
        self.assertTrue(evidence["all_finite"])
        self.assertEqual(evidence["maximum_finite_absolute_value"], 1e197)
        scalar = torch.tensor(1e197, dtype=torch.float64)
        self.assertEqual(CPUFP64StableMicrobatchObjective._scalar_class(scalar), "FINITE")

    def test_terminal_duplicate_binds_cached_flat_and_losses(self):
        value = _objective()
        flat = value.flat_group("group")
        preflight = value.preflight_all_base_batches("group", flat)
        receipt = validate_terminal_duplicate(
            value, "group", flat,
            list(preflight["base_full_search_batch_losses"]), 0.0,
        )
        self.assertEqual(
            receipt["status"],
            "PASS_TERMINAL_FULL_BANK_DUPLICATES_CACHED_COMMIT_GATE",
        )
        with self.assertRaises(DeterministicNumericalFailure) as captured:
            changed = flat.clone()
            changed[0] = changed[0] + 1
            validate_terminal_duplicate(value, "group", changed, [0.0, 0.0], 0.0)
        self.assertEqual(
            captured.exception.diagnostics["reason_code"],
            "TERMINAL_CACHE_FLAT_IDENTITY_MISMATCH",
        )
        with self.assertRaises(DeterministicNumericalFailure) as captured:
            validate_terminal_duplicate(value, "group", flat, [0.0], 0.0)
        self.assertEqual(
            captured.exception.diagnostics["reason_code"],
            "TERMINAL_DUPLICATE_BATCH_COUNT_MISMATCH",
        )
        with self.assertRaises(DeterministicNumericalFailure) as captured:
            losses = list(preflight["base_full_search_batch_losses"])
            losses[0] += 1e-3
            validate_terminal_duplicate(value, "group", flat, losses, 1e-6)
        self.assertEqual(
            captured.exception.diagnostics["reason_code"],
            "TERMINAL_DUPLICATE_TOLERANCE_EXCEEDED",
        )

    def test_variable_accounting_accepts_exact_zero_and_maximum_gate_bounds(self):
        solver = {
            "acceptance_gate_maximum_attempts": 18,
            "acceptance_gate_logical_evaluations_per_attempt": 12,
            "acceptance_gate_physical_model_forwards_per_attempt": 24,
            "total_logical_evaluations_minimum": 204,
            "total_physical_model_forwards_minimum": 408,
            "total_logical_evaluations_maximum": 420,
            "total_physical_model_forwards_maximum": 840,
        }
        self.assertEqual(expected_variable_totals(solver, 0), (204, 408))
        self.assertEqual(expected_variable_totals(solver, 18), (420, 840))
        with self.assertRaisesRegex(RuntimeError, "outside sealed bound"):
            expected_variable_totals(solver, 19)

    def test_every_accepted_flat_is_streamed_into_ordered_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            branch = root / "branch"
            output = root / "output"
            branch.mkdir()
            output.mkdir()
            payload = b"candidate-flat"
            flat_identity = write_exclusive(
                branch / "accepted_01_group_flat.f32le", payload,
            )
            accepted_identity = write_json_exclusive(
                branch / "accepted_01.json", {
                    "schema": "hfw_post_training_accepted_delta_v1",
                    "accept_index": 1, "pair_index": 7, "group": "codecs",
                    "delta_sha256": "d" * 64,
                },
            )
            child_identity = write_json_exclusive(
                branch / "materialized_child_01.pt", {
                    "schema": "hfw_w560_daytona_cpu_artifact_only_child_v1",
                    "group": "codecs",
                    "exact_candidate_group_flat": flat_identity,
                },
            )
            bundle, entries = build_accepted_candidate_bundle(
                branch, output, "codecs", 1,
                [{
                    "passed": True, "iteration": 7,
                    "candidate_flat_sha256": flat_identity["sha256"],
                    "candidate_full_search_mean_ce": 5.0,
                }],
                len(payload),
            )
            self.assertEqual(
                hash_file(output / "accepted_candidates.f32le.bundle")["sha256"],
                flat_identity["sha256"],
            )
            self.assertEqual(bundle["size_bytes"], len(payload))
            self.assertEqual(entries[0]["accepted_record_sha256"], accepted_identity["sha256"])
            self.assertEqual(entries[0]["artifact_only_child_sha256"], child_identity["sha256"])
            self.assertEqual(entries[0]["offset_bytes"], 0)

    def test_vendor_gate_is_before_current_mutation_and_artifact_write(self):
        source = Path(__file__).parents[1] / (
            "frozen/vendor/hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver/"
            "branch_search.py"
        )
        text = source.read_text()
        gate = text.index("acceptance_gate = dict(objective.acceptance_gate(")
        mutation = text.index("current = best[1].detach()", gate)
        artifact = text.index("_exclusive_write(branch_root / delta_name", mutation)
        self.assertLess(gate, mutation)
        self.assertLess(mutation, artifact)


if __name__ == "__main__":
    unittest.main()
