from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from representation_experiments.common import TokenMemmap, parameter_count
from representation_experiments.contract import (
    HISTORICAL_CONFIG_SHA256,
    require_single_process_environment,
    sha256_file,
    validate_static_contract,
)
from representation_experiments.distribution_distillation import (
    exact_forward_kl,
    multi_temperature_kl,
)
from representation_experiments.predictive_models import (
    build_student,
    official_bias_parameter_count,
    resolve_budget,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "official_nanogpt_aug8_10m.json"
HISTORICAL_PATH = ROOT / "bindings" / "qwen06b_matched_hierarchy_function_v1.json"


class ExactComparatorTests(unittest.TestCase):
    def test_static_contract_and_historical_binding(self):
        receipt = validate_static_contract(ROOT)
        self.assertEqual(receipt["status"], "PASS_STATIC_CONTRACT")
        self.assertEqual(sha256_file(HISTORICAL_PATH), HISTORICAL_CONFIG_SHA256)
        self.assertEqual(receipt["steps"], 4883)
        self.assertEqual(receipt["executed_tokens"], 10000384)

    def test_training_and_evaluation_are_exact_historical_values(self):
        current = json.loads(CONFIG_PATH.read_text())
        historical = json.loads(HISTORICAL_PATH.read_text())
        for key in (
                "teacher", "vocab_size", "data", "token_dtype",
                "student_parameter_budget", "model_max_context",
                "embedding_projection", "evaluation", "benchmark", "safety"):
            self.assertEqual(current[key], historical[key])
        training = dict(current["training"])
        for extra in (
                "expected_steps", "expected_executed_tokens", "sampling",
                "kl_implementation"):
            training.pop(extra)
        self.assertEqual(training, historical["training"])
        self.assertNotIn("representation_matching", current["training"])
        self.assertNotIn("future_state_prediction", current["training"])
        self.assertEqual(current["evaluation"]["long_contexts"], [1024, 2048])

    def test_exhaustive_official_geometry_is_unique_closest_match(self):
        config = json.loads(CONFIG_PATH.read_text())
        model_config = config["models"]["nanogpt"]
        search = model_config["geometry_search"]
        candidates = []
        for heads in range(search["minimum_heads"], search["maximum_heads"] + 1):
            width = heads * search["head_dimension"]
            for layers in range(
                    search["minimum_layers"], search["maximum_layers"] + 1):
                count = official_bias_parameter_count(
                    config["vocab_size"], config["model_max_context"],
                    width, layers)
                if count <= search["maximum_stored_parameters"]:
                    candidates.append((
                        abs(count - search["target_stored_parameters"]),
                        -count, layers, heads, width, count))
        winner = min(candidates)
        self.assertEqual(winner, (9360, -119685440, 21, 7, 448, 119685440))
        self.assertEqual(sum(row[0] == winner[0] for row in candidates), 1)
        resolved = resolve_budget(
            "nanogpt", config["vocab_size"], config["model_max_context"],
            model_config, config["student_parameter_budget"])
        self.assertEqual(resolved, config["resolved_model_expectation"])

    def test_adapter_is_tensor_free_and_matches_upstream_parameter_count(self):
        config = json.loads(CONFIG_PATH.read_text())
        resolved = config["resolved_model_expectation"]
        model_config = {
            **config["models"]["nanogpt"],
            "layers": resolved["layers"],
            "heads": resolved["heads"],
        }
        with torch.device("meta"):
            model = build_student(
                "nanogpt", config["vocab_size"], resolved["width"],
                config["model_max_context"], model_config)
        self.assertEqual(parameter_count(model), resolved["stored_parameters"])
        self.assertTrue(all(name.startswith("gpt.") for name, _ in model.named_parameters()))
        state = model.state_accounting(batch_size=1, context=2048)
        self.assertFalse(state["actual_runtime_measured"])
        self.assertEqual(state["layers"], 21)
        self.assertEqual(state["heads"], 7)

    def test_restart_stable_batch_order_matches_historical_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            values = np.arange(10000, dtype=np.uint32)
            values.tofile(path)
            dataset = TokenMemmap.open(path, "uint32")
            observed = dataset.batch_for_step(32, 4, 19, 12001).numpy()
            generator = np.random.default_rng(12001 + 1_000_003 * 19)
            starts = generator.integers(0, len(values) - 33, size=4)
            expected = np.stack([values[start:start + 33] for start in starts])
            np.testing.assert_array_equal(observed, expected)

    def test_historical_multitemperature_objective_matches_direct_kl(self):
        generator = torch.Generator().manual_seed(41)
        student = torch.randn(2, 3, 11, generator=generator)
        teacher = torch.randn(2, 3, 11, generator=generator)
        temperatures = [
            {"value": 1.0, "weight": 0.5},
            {"value": 2.0, "weight": 0.3},
            {"value": 4.0, "weight": 0.2},
        ]
        observed, metrics = multi_temperature_kl(
            student, teacher, temperatures, 4, "sequential")
        expected = sum(
            spec["weight"] * exact_forward_kl(
                student, teacher, spec["value"], 4)
            for spec in temperatures)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(set(metrics), {"kl_t1", "kl_t2", "kl_t4"})

    def test_process_gate_rejects_ddp_and_multiple_visible_gpus(self):
        with mock.patch.dict(os.environ, {
                "WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "forbids distributed"):
                require_single_process_environment()
        with mock.patch.dict(os.environ, {
                "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
                "CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "exactly one visible GPU"):
                require_single_process_environment()
        with mock.patch.dict(os.environ, {
                "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
                "CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            self.assertEqual(require_single_process_environment()["world_size"], 1)


if __name__ == "__main__":
    unittest.main()
