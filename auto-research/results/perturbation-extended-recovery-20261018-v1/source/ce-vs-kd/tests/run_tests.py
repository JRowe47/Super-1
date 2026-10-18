from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


RUNTIME = Path(__file__).resolve().parents[1]
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from ab_experiment.common import file_identity, sha256_bytes, tensor_f32le_bytes, write_exclusive_json
from ab_experiment.run_ab import materialize_winner_checkpoint
from ab_experiment.runtime import TeacherLogitStore, claim_phase, require_bound_output
from ab_experiment.static_gate import validate_phase_config


class TinyParent:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.model.weight.requires_grad_(False)
        self.groups = {"base_affines": ("weight",)}


class MatchedABTests(unittest.TestCase):
    def test_teacher_capability_is_exactly_bound_to_token_sequence_and_master(self):
        root = Path(tempfile.mkdtemp(prefix="teacher-capability-"))
        role_root = root / "search"
        role_root.mkdir()
        token_path = root / "search.u32le"
        token = np.arange(513, dtype="<u4")
        token_path.write_bytes(token.tobytes())
        token_identity = {
            "sha256": hashlib.sha256(token.tobytes()).hexdigest(),
            "shape": [1, 513],
            "maximum_token_id": 512,
        }
        shard = role_root / "00000-placeholder.bf16le"
        shard.write_bytes(b"")
        sequence_sha = sha256_bytes(token.tobytes())
        capability = {
            "schema": "hfw_w560_ce_vs_kd_teacher_logits_capability_v1",
            "status": "PASS_SEARCH_ONLY_TEACHER_LOGITS_CAPABILITY",
            "master_manifest_sha256": "a" * 64,
            "roles_available": ["SEARCH"],
            "banks": {"SEARCH": {
                "bank": token_identity,
                "sequence_count": 1,
                "shards": [{
                    "index": 0,
                    "path": str(shard),
                    "sha256": "b" * 64,
                    "size_bytes": 512 * 151936 * 2,
                    "shape": [512, 151936],
                    "dtype": "bfloat16_little_endian_uint16_payload",
                    "input_sequence_sha256": sequence_sha,
                }],
                "total_bytes": 512 * 151936 * 2,
                "target_aligned_logits_only": True,
                "unused_513th_logit_not_stored": True,
            }},
            "verify_or_final_role_present": False,
            "teacher_forward_calls_after_freeze": 0,
        }
        capability_path = root / "SEARCH_TEACHER_CAPABILITY.json"
        capability_artifact = write_exclusive_json(capability_path, capability)
        config = {
            "inputs": {"teacher_outputs": {
                "capability_path": str(capability_path.resolve()),
                "manifest_sha256": capability_artifact["sha256"],
                "master_manifest_sha256": "a" * 64,
                "roles_available": ["SEARCH"],
            }},
            "banks": {"SEARCH": {
                **file_identity(token_path), "shape": [1, 513],
                "maximum_token_id": 512,
            }},
        }
        store = TeacherLogitStore(config, capability_path)
        self.assertEqual(set(store.by_role), {"SEARCH"})
        tampered = json.loads(json.dumps(capability))
        tampered["banks"]["SEARCH"]["shards"][0]["input_sequence_sha256"] = "c" * 64
        tampered_path = root / "SEARCH_TEACHER_CAPABILITY_TAMPERED.json"
        tampered_artifact = write_exclusive_json(tampered_path, tampered)
        tampered_config = json.loads(json.dumps(config))
        tampered_config["inputs"]["teacher_outputs"]["capability_path"] = str(
            tampered_path.resolve()
        )
        tampered_config["inputs"]["teacher_outputs"]["manifest_sha256"] = tampered_artifact["sha256"]
        with self.assertRaisesRegex(RuntimeError, "token-sequence binding"):
            TeacherLogitStore(tampered_config, tampered_path)

    def test_winner_checkpoint_uses_exact_replacement_and_parent_cumulative_delta(self):
        root = Path(tempfile.mkdtemp(prefix="winner-materialization-"))
        parent = TinyParent()
        base = torch.cat([parameter.reshape(-1) for parameter in parent.model.parameters()])
        flat = base + torch.tensor([0.1, -0.2, 0.3, -0.4])
        delta = tensor_f32le_bytes(flat - base)
        exact = tensor_f32le_bytes(flat)
        exact_path = root / "selected.group_flat.f32le"
        delta_path = root / "selected.cumulative_delta.f32le"
        exact_path.write_bytes(exact)
        delta_path.write_bytes(delta)
        selection = {
            "selected": {
                "accept_index": 3,
                "exact_candidate_group_flat": file_identity(exact_path),
                "cumulative_delta": file_identity(delta_path),
                "incremental_delta": file_identity(delta_path),
            }
        }
        config = {
            "experiment": {"group": "base_affines"},
            "inputs": {"starting_checkpoint": {"sha256": "d" * 64}},
        }
        result = materialize_winner_checkpoint(
            config,
            {"origin_checkpoint_sha256": "e" * 64, "generation": 2},
            parent, flat, "ce", selection, "f" * 64, root / "winner.pt",
        )
        payload = torch.load(result["checkpoint"]["path"], map_location="cpu", weights_only=True)
        self.assertEqual(payload["selected_candidate_delta_sha256"], sha256_bytes(delta))
        self.assertEqual(payload["solver_cumulative_delta_sha256"], sha256_bytes(delta))
        self.assertEqual(payload["replacement_group_flat_sha256"], sha256_bytes(exact))
        self.assertEqual(payload["group_replacement_materialization_source"], "exact_candidate_group_flat_bytes")
        self.assertFalse(payload["solver_cumulative_delta_used_to_reconstruct_group"])
        self.assertEqual(result["replacement_group_flat"]["sha256"], sha256_bytes(exact))
        self.assertEqual(result["solver_cumulative_delta"]["sha256"], sha256_bytes(delta))

    def test_phase_gate_rejects_final_smuggling_into_search(self):
        root = Path(tempfile.mkdtemp(prefix="phase-capability-"))
        valid = {
            "phase": "CE_SEARCH", "banks": {"SEARCH": {}},
            "inputs": {},
        }
        valid_path = root / "valid.json"
        valid_path.write_text(json.dumps(valid))
        self.assertEqual(validate_phase_config(valid_path)["phase"], "CE_SEARCH")
        invalid = {**valid, "hidden_capability": {"role": "FINAL"}}
        invalid_path = root / "invalid.json"
        invalid_path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(RuntimeError, "VERIFY/FINAL"):
            validate_phase_config(invalid_path)

    def test_unique_phase_claim_and_bound_output_are_single_use(self):
        root = Path(tempfile.mkdtemp(prefix="unique-phase-"))
        config_path = root / "phase.json"
        output_root = root / "bound-output"
        config = {
            "phase": "CE_SEARCH",
            "phase_output": {
                "claim_path": str(root / "CE_SEARCH_STARTED.json"),
                "result_root": str(output_root),
            },
        }
        config_path.write_text(json.dumps(config))
        require_bound_output(config, output_root)
        with self.assertRaisesRegex(RuntimeError, "unique preregistered"):
            require_bound_output(config, root / "alternate-output")
        first = claim_phase(config, config_path)
        self.assertEqual(first["sha256"], file_identity(root / "CE_SEARCH_STARTED.json")["sha256"])
        with self.assertRaises(FileExistsError):
            claim_phase(config, config_path)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MatchedABTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
