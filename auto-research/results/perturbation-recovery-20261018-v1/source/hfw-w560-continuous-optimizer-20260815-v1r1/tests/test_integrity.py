from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hfw_continuous.common import (
    hash_file_same_fd, read_json_same_fd, write_content_addressed_json, write_exclusive_bytes,
    write_exclusive_json,
)
from hfw_continuous.committer import commit_proposal, load_exact_candidate_group
from hfw_continuous.cli import result_exit_code
from hfw_continuous.disk_safety import validated_disk_reservations
from hfw_continuous.ledger import Ledger, initialize_ledger
from hfw_continuous.proposal_worker import (
    JobLocalTraceRegistry, artifact_only_projection, validate_artifact_candidate_chain,
)
from hfw_continuous.relocation import load_config, vendor_modules
from hfw_continuous.release_binding import validate_runtime_manifest
from hfw_continuous.state import load_head_model
from hfw_continuous.token_commitment import prepare_token_commitment
from hfw_continuous.supervisor import run_continuously


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / "config/controller.json"


def write_test_runtime_config(root: Path, config: dict) -> tuple[Path, dict]:
    runtime = root / "hfw_continuous/runtime.py"
    vendor = root / "vendor/__init__.py"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    vendor.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("VALUE = 1\n")
    vendor.write_text("\n")
    files = {}
    for path in (runtime, vendor):
        payload = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
        }
    body = {
        "schema": "hfw_w560_continuous_runtime_manifest_v1",
        "namespace": root.name,
        "file_count": len(files),
        "total_size_bytes": sum(value["size_bytes"] for value in files.values()),
        "files": files,
        "coverage": {"hfw_continuous": "all_python_modules", "vendor": "all_regular_noncache_files"},
        "excluded_from_runtime_manifest": [
            "config/controller.json", "config/continuous_runtime_manifest.v1.json",
            "tests", "top_level_legacy_copy",
        ],
    }
    canonical = lambda value: (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    manifest = {
        **body, "manifest_identity_sha256": hashlib.sha256(canonical(body)).hexdigest(),
    }
    manifest_path = root / "config/continuous_runtime_manifest.v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical(manifest))
    manifest_payload = manifest_path.read_bytes()
    config["runtime_release"] = {
        "manifest_relative_path": "config/continuous_runtime_manifest.v1.json",
        "manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_size_bytes": len(manifest_payload),
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
    }
    config_path = root / "config/controller.json"
    config_path.write_text(json.dumps(config))
    return config_path, validate_runtime_manifest(config_path, config)


class ToyObjective:
    def __init__(self):
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0, -0.75, 0.5])))
        self.parent = SimpleNamespace(model=model, groups={"base_affines": ("weight",)})
        self.base_parameters = {"weight": model.weight.detach()}
        self.buffers = {}
        self.target = torch.tensor([0.2, -0.1, 0.05])

    def flat_group(self, group: str) -> torch.Tensor:
        assert group == "base_affines"
        return self.base_parameters["weight"].reshape(-1).detach()

    def loss(self, group: str, flat: torch.Tensor, batch_index: int) -> float:
        return float((flat - self.target).square().mean() + batch_index * 1e-5)

    def loss_vector(self, group: str, flat: torch.Tensor, batch_index: int):
        return (self.loss(group, flat, batch_index),)


class IntegrityTests(unittest.TestCase):
    def test_cli_returns_nonzero_for_block_receipts(self) -> None:
        self.assertEqual(result_exit_code({"status": "BLOCK_FAIL_CLOSED"}), 1)
        self.assertEqual(result_exit_code({"status": "PASS"}), 0)

    def test_vendored_solver_and_bridge_exact_hashes(self) -> None:
        config = load_config(CONFIG)
        root, branch, bridge, _, _ = vendor_modules(config)
        self.assertEqual(
            hash_file_same_fd(Path(branch.__file__))["sha256"],
            "0aced15800caf4042ffd1863c7cd1c5415bed6d9dc799220be17635bd5c5f349",
        )
        self.assertEqual(
            hash_file_same_fd(Path(bridge.__file__))["sha256"],
            "919c70204c5fefd86a65410f1256c14f54c4ffb34b4db15e5b993ff705cc0ac7",
        )
        self.assertEqual(
            hash_file_same_fd(root / "config/production_protocol.json")["sha256"],
            "be700d482822ac2e9317e43e6714d0d1100c03d8d60f4b944d8ddcb6ad7f7356",
        )

    def test_token_commitment_is_create_only_disjoint_and_has_no_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = np.arange(4096, dtype="<u4")
            corpus_path = root / "val.bin"
            corpus.tofile(corpus_path)
            corpus_sha = hashlib.sha256(corpus.tobytes()).hexdigest()
            origin_sha = "a" * 64
            reveal = {
                "schema": "sealed_evaluation_reveal_preimage_v1",
                "validation_sha256": corpus_sha,
                "validation_token_count": int(corpus.size),
                "evaluation_offsets": [100, 200, 300, 400],
                "sample_length_tokens": 5,
                "salt_hex": "11" * 32,
            }
            reveal_identity = write_exclusive_json(root / "published.reveal.json", reveal)
            authorization = {
                "schema": "exact_four_branch_final_eval_authorization_v1",
                "status": "AUTHORIZED",
                "opaque_eval_commitment_sha256": reveal_identity["sha256"],
                "published_reveal_sha256": reveal_identity["sha256"],
                "validation_sha256": corpus_sha,
                "parent_checkpoint_sha256": origin_sha,
                "final_evaluation_calls_authorized": 1,
                "search_or_calibration_calls_after_reveal_authorized": 0,
            }
            authorization_identity = write_exclusive_json(
                root / "prior.authorization.json", authorization
            )
            config = {
                "origin": {"checkpoint_sha256": origin_sha},
                "corpus": {
                    "path": str(corpus_path),
                    "sha256": corpus_sha,
                    "size_bytes": corpus.nbytes,
                    "dtype": "uint32",
                },
                "token_policy": {
                    "context": 4,
                    "batch_size": 2,
                    "prior_public_exclusion": {"batches": 1, "seed": 3},
                    "search": {"batches": 2, "seed": 7},
                    "commit": {
                        "batches": 2,
                        "starts": [100, 200, 300, 400],
                        "published_reveal": {
                            **reveal_identity,
                            "schema": reveal["schema"],
                        },
                        "prior_authorization": {
                            **authorization_identity,
                            "schema": authorization["schema"],
                            "status": authorization["status"],
                        },
                        "reclassified_role": "PERMANENT_REUSABLE_COMMIT_OBJECTIVE_NOT_HELD_OUT",
                    },
                },
            }
            config_path, _ = write_test_runtime_config(root, config)
            result = prepare_token_commitment(config_path, root / "tokens")
            self.assertFalse(result["final_bank_present"])
            self.assertEqual(set(result["banks"]), {"SEARCH", "COMMIT"})
            intervals = []
            for role in ("SEARCH", "COMMIT"):
                for record in result["selection_records"][role]:
                    for interval in record["intervals"]:
                        candidate = tuple(interval)
                        self.assertFalse(any(candidate[0] < end and start < candidate[1] for start, end in intervals))
                        intervals.append(candidate)
            with self.assertRaises(FileExistsError):
                prepare_token_commitment(config_path, root / "tokens")

    def _toy_protocol(self, branch_module) -> dict:
        config = load_config(CONFIG)
        root = Path(branch_module.__file__).resolve().parent
        protocol = json.loads((root / "config/production_protocol.json").read_text())
        protocol["parent"]["checkpoint_sha256"] = "a" * 64
        protocol["search"]["branch_seeds"]["base_affines"] = 123456
        protocol["data"]["calibration_batches"] = 12
        return protocol

    def test_artifact_adapter_changes_projection_only_not_solver_trace(self) -> None:
        config = load_config(CONFIG)
        _, branch, _, _, _ = vendor_modules(config)
        before_sha = hash_file_same_fd(Path(branch.__file__))["sha256"]
        protocol = self._toy_protocol(branch)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_registry = JobLocalTraceRegistry(root / "original-events")
            original = branch.run_independent_branch(
                ToyObjective(), protocol, "base_affines", root / "original",
                original_registry, materialize=True,
            )
            adapted_registry = JobLocalTraceRegistry(root / "adapted-events")
            with artifact_only_projection(branch):
                adapted = branch.run_independent_branch(
                    ToyObjective(), protocol, "base_affines", root / "adapted",
                    adapted_registry, materialize=True,
                )
            frozen_validation, candidates = validate_artifact_candidate_chain(
                branch, protocol, root / "adapted/base_affines", "base_affines"
            )
            self.assertEqual(frozen_validation["status"], "PASS_BRANCH_ARTIFACT_CHAIN")
            self.assertEqual(len(candidates), adapted.accepts)
            self.assertTrue(
                frozen_validation["controller_verified_exact_job_seed_solver_budget_and_no_training"]
            )
            self.assertTrue(torch.equal(original.final_flat, adapted.final_flat))
            self.assertEqual(original.accepts, adapted.accepts)
            self.assertEqual(original.proposals, adapted.proposals)
            self.assertEqual(original.receipt["forward_calls"], adapted.receipt["forward_calls"])
            self.assertEqual(original.receipt["final_radius"], adapted.receipt["final_radius"])
            for left, right in zip(
                original.receipt["proposal_records"], adapted.receipt["proposal_records"]
            ):
                for key in (
                    "center_loss_vector", "plus_loss_vector", "minus_loss_vector",
                    "line_search_loss_vector", "accepted_side", "accepted_loss",
                    "acceptance_threshold", "direction_sha256s",
                ):
                    self.assertEqual(left[key], right[key])
            descriptors = sorted((root / "adapted/base_affines").glob("materialized_child_*.pt"))
            self.assertTrue(descriptors)
            self.assertEqual(json.loads(descriptors[-1].read_text())["artifact_projection_only"], True)
            exact_groups = sorted((root / "adapted/base_affines").glob("accepted_*_group_flat.f32le"))
            self.assertEqual(len(exact_groups), adapted.accepts)
            self.assertEqual(
                candidates[-1]["exact_candidate_group_flat"]["sha256"],
                hash_file_same_fd(exact_groups[-1])["sha256"],
            )
        self.assertEqual(hash_file_same_fd(Path(branch.__file__))["sha256"], before_sha)

    def test_exact_candidate_bytes_survive_noninvertible_float32_delta(self) -> None:
        config = load_config(CONFIG)
        _, branch, _, _, _ = vendor_modules(config)
        base_bits = np.array([0xBFCE039A], dtype="<u4")
        candidate_bits = np.array([0x3DA93BAF], dtype="<u4")
        base = torch.from_numpy(base_bits.view("<f4").copy())
        exact_candidate = torch.from_numpy(candidate_bits.view("<f4").copy())
        delta = exact_candidate - base
        self.assertFalse(torch.equal(base + delta, exact_candidate))
        self.assertEqual(int((base + delta).numpy().view("<u4")[0]), 0x3DA93BB0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exact_artifact = write_exclusive_bytes(
                root / "candidate.group_flat.f32le", branch._tensor_bytes(exact_candidate)
            )
            delta_artifact = write_exclusive_bytes(
                root / "candidate.group_delta.f32le", branch._tensor_bytes(delta)
            )
            incremental_artifact = write_exclusive_bytes(
                root / "candidate.incremental_delta.f32le", branch._tensor_bytes(delta)
            )
            candidate = {
                "candidate_group_flat_sha256": exact_artifact["sha256"],
                "exact_candidate_group_flat": exact_artifact,
                "cumulative_delta": delta_artifact,
                "incremental_delta": incremental_artifact,
            }
            loaded, exact_payload, delta_payload, relation = load_exact_candidate_group(
                branch, candidate, base, expected_branch_root=root
            )
            self.assertTrue(torch.equal(loaded, exact_candidate))
            self.assertEqual(exact_payload, branch._tensor_bytes(exact_candidate))
            self.assertEqual(delta_payload, branch._tensor_bytes(delta))
            self.assertTrue(
                relation["serialized_float32_subtraction_matches_cumulative_delta_bitwise"]
            )
            self.assertFalse(relation["base_plus_delta_bitwise_equals_exact_candidate"])
            self.assertGreater(relation["base_plus_delta_mismatched_float32_elements"], 0)
            tampered_exact = root / "tampered.group_flat.f32le"
            tampered_exact.write_bytes(exact_payload[:-4] + np.float32(3.0).tobytes())
            bad_candidate = {
                **candidate,
                "exact_candidate_group_flat": {
                    **exact_artifact, "path": str(tampered_exact),
                },
            }
            with self.assertRaises(RuntimeError):
                load_exact_candidate_group(
                    branch, bad_candidate, base, expected_branch_root=root
                )
            tampered_delta = root / "tampered.group_delta.f32le"
            tampered_delta.write_bytes(delta_payload[:-4] + np.float32(4.0).tobytes())
            bad_delta = {
                **candidate,
                "cumulative_delta": {
                    **delta_artifact, "path": str(tampered_delta),
                },
            }
            with self.assertRaises(RuntimeError):
                load_exact_candidate_group(branch, bad_delta, base, expected_branch_root=root)
            outside = root / "outside"
            outside.mkdir()
            outside_exact = write_exclusive_bytes(
                outside / "candidate.group_flat.f32le", exact_payload
            )
            outside_candidate = {
                **candidate, "exact_candidate_group_flat": outside_exact,
            }
            with self.assertRaises(RuntimeError):
                load_exact_candidate_group(
                    branch, outside_candidate, base, expected_branch_root=root
                )
            nonfinite_payload = np.array([np.inf], dtype="<f4").tobytes()
            nonfinite_artifact = write_exclusive_bytes(
                root / "nonfinite.group_flat.f32le", nonfinite_payload
            )
            nonfinite_candidate = {
                **candidate,
                "candidate_group_flat_sha256": nonfinite_artifact["sha256"],
                "exact_candidate_group_flat": nonfinite_artifact,
            }
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                load_exact_candidate_group(
                    branch, nonfinite_candidate, base, expected_branch_root=root
                )
            legacy_candidate = {
                key: value for key, value in candidate.items()
                if key not in {"exact_candidate_group_flat", "candidate_group_flat_sha256"}
            }
            with self.assertRaises((KeyError, RuntimeError)):
                load_exact_candidate_group(branch, legacy_candidate, base)

    def test_stale_proposal_rejected_before_commit_bank_opens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin_path = root / "origin.pt"
            origin_identity = write_exclusive_bytes(origin_path, b"origin")
            config_path, runtime_binding = write_test_runtime_config(root, {})
            token_path = root / "tokens.json"
            token_path.write_text(json.dumps({
                "config_sha256": hash_file_same_fd(config_path)["sha256"],
                "runtime_source_binding": runtime_binding,
            }))
            prepared = {
                "relocated_checkpoint": origin_identity,
            }
            ledger_root = root / "ledger"
            initialize_ledger(
                ledger_root,
                config_identity=hash_file_same_fd(config_path),
                token_manifest_identity=hash_file_same_fd(token_path),
                prepared_parent=prepared,
                baseline_evidence={"mean_ce": 1.0},
                origin={
                    "checkpoint_sha256": origin_identity["sha256"], "size_bytes": 6,
                    "model_state_digest": "f" * 64,
                },
                runtime_source_binding=runtime_binding,
            )
            ledger = Ledger(ledger_root)
            job = ledger.create_job(
                group="base_affines", root_seed=1, allowed_groups=["base_affines"]
            )
            proposal = {
                "schema": "hfw_w560_continuous_candidate_chain_proposal_v1",
                "job_record_sha256": job["job_record_sha256"],
                "job": {key: value for key, value in job.items() if key not in {
                    "job_record_sha256", "job_artifact"
                }},
                "base_head_record_sha256": "0" * 64,
            }
            proposal_artifact = write_content_addressed_json(
                ledger_root / "proposals", ".proposal.json", proposal
            )
            decision = commit_proposal(
                config_path, token_path, ledger_root, Path(proposal_artifact["path"])
            )
            self.assertEqual(decision["status"], "REJECTED_STALE_HEAD_NO_MERGE_NO_REBASE")
            self.assertFalse(decision["commit_bank_opened"])
            self.assertEqual(Ledger(ledger_root).current_head()["generation"], 0)

    def test_ledger_rejects_config_or_token_substitution_and_ignores_staged_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, runtime_binding = write_test_runtime_config(
                root, {"controller": {}}
            )
            token_path = root / "tokens.json"
            token_path.write_text(json.dumps({
                "config_sha256": hash_file_same_fd(config_path)["sha256"],
                "runtime_source_binding": runtime_binding,
            }))
            origin = write_exclusive_bytes(root / "origin.pt", b"origin")
            ledger_root = root / "ledger"
            initialize_ledger(
                ledger_root,
                config_identity=hash_file_same_fd(config_path),
                token_manifest_identity=hash_file_same_fd(token_path),
                prepared_parent={"relocated_checkpoint": origin},
                baseline_evidence={},
                origin={
                    "checkpoint_sha256": origin["sha256"], "size_bytes": origin["size_bytes"],
                    "model_state_digest": "e" * 64,
                },
                runtime_source_binding=runtime_binding,
            )
            ledger = Ledger(ledger_root)
            ledger.validate_runtime_inputs(config_path, token_path)
            runtime_path = root / "hfw_continuous/runtime.py"
            original_runtime = runtime_path.read_bytes()
            runtime_path.write_bytes(original_runtime + b"TAMPER = True\n")
            with self.assertRaises(RuntimeError):
                ledger.validate_runtime_inputs(config_path, token_path)
            runtime_path.write_bytes(original_runtime)
            manifest_path = root / "config/continuous_runtime_manifest.v1.json"
            original_manifest = manifest_path.read_bytes()
            manifest_path.write_bytes(original_manifest + b" ")
            with self.assertRaises(RuntimeError):
                ledger.validate_runtime_inputs(config_path, token_path)
            manifest_path.write_bytes(original_manifest)
            ledger.validate_runtime_inputs(config_path, token_path)
            substituted = root / "other-config.json"
            substituted.write_text('{"controller":{"changed":true}}')
            with self.assertRaises(RuntimeError):
                ledger.validate_runtime_inputs(substituted, token_path)
            write_exclusive_bytes(ledger_root / "heads/.00000001-deadbeef.staged", b"partial")
            self.assertEqual(ledger.current_head()["generation"], 0)
            contract_path = ledger_root / "contract.json"
            contract_path.chmod(0o644)
            tampered = json.loads(contract_path.read_text())
            tampered["append_only"] = False
            contract_path.write_text(json.dumps(tampered))
            with self.assertRaises(RuntimeError):
                ledger.validate_runtime_inputs(config_path, token_path)

    def test_supervisor_singleton_and_disk_gate_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "controller": {
                    "groups": ["base_affines"], "parallel_workers": 1, "root_seed": 1,
                    "max_waves": 1,
                }
            }
            config_path, runtime_binding = write_test_runtime_config(root, config)
            token_path = root / "tokens.json"
            token_path.write_text(json.dumps({
                "config_sha256": hash_file_same_fd(config_path)["sha256"],
                "runtime_source_binding": runtime_binding,
            }))
            origin = write_exclusive_bytes(root / "origin.pt", b"origin")
            ledger_root = root / "ledger"
            initialize_ledger(
                ledger_root,
                config_identity=hash_file_same_fd(config_path),
                token_manifest_identity=hash_file_same_fd(token_path),
                prepared_parent={"relocated_checkpoint": origin}, baseline_evidence={},
                origin={
                    "checkpoint_sha256": origin["sha256"], "size_bytes": origin["size_bytes"],
                    "model_state_digest": "d" * 64,
                },
                runtime_source_binding=runtime_binding,
            )
            ledger = Ledger(ledger_root)
            with ledger.supervisor_lock():
                with self.assertRaises(RuntimeError):
                    with ledger.supervisor_lock():
                        pass
            reservation = {
                "minimum_unreserved_free_bytes": 1 << 62,
                "maximum_group_elements": 1,
                "maximum_accepts_per_branch": 1,
                "worker_artifact_bytes_before_overhead": 1,
                "reserved_bytes_per_worker": 1,
                "promotion_group_artifact_bytes_before_overhead": 1,
                "reserved_bytes_for_possible_promotion": 1,
            }
            with patch(
                "hfw_continuous.supervisor.validated_disk_reservations",
                return_value=reservation,
            ):
                result = run_continuously(config_path, token_path, ledger_root)
            self.assertEqual(result["status"], "BLOCK_INSUFFICIENT_DISK_BEFORE_RECOVERY_OR_WAVE")
            self.assertEqual(len(list((ledger_root / "jobs").glob("*.job.json"))), 0)

    def test_formula_bound_disk_reservations_cover_exact_candidate_sidecars(self) -> None:
        config = load_config(CONFIG)
        result = validated_disk_reservations(config)
        raw_worker_bytes = 10_046_960 * 18 * 3 * 4
        self.assertEqual(result["worker_artifact_bytes_before_overhead"], raw_worker_bytes)
        self.assertGreater(result["reserved_bytes_per_worker"], raw_worker_bytes)
        self.assertEqual(result["reserved_bytes_per_worker"], 2_707_014_272)
        self.assertEqual(result["reserved_bytes_for_possible_promotion"], 1_815_805_367)
        bad = deepcopy(config)
        bad["controller"]["disk_safety"]["worker_reservation_formula"][
            "computed_reserved_bytes_per_worker"
        ] -= 1
        with self.assertRaises(RuntimeError):
            validated_disk_reservations(bad)

    def test_supervisor_receipts_committer_and_recovery_exceptions_and_stops(self) -> None:
        class SuccessfulWorker:
            returncode = 0

            def communicate(self):
                return "", ""

        def initialized_test_ledger(root: Path):
            config = {
                "controller": {
                    "groups": ["base_affines"], "parallel_workers": 1,
                    "root_seed": 1, "max_waves": 2,
                }
            }
            config_path, runtime_binding = write_test_runtime_config(root, config)
            token_path = root / "tokens.json"
            token_path.write_text(json.dumps({
                "config_sha256": hash_file_same_fd(config_path)["sha256"],
                "runtime_source_binding": runtime_binding,
            }))
            origin = write_exclusive_bytes(root / "origin.pt", b"origin")
            ledger_root = root / "ledger"
            initialize_ledger(
                ledger_root,
                config_identity=hash_file_same_fd(config_path),
                token_manifest_identity=hash_file_same_fd(token_path),
                prepared_parent={"relocated_checkpoint": origin}, baseline_evidence={},
                origin={
                    "checkpoint_sha256": origin["sha256"],
                    "size_bytes": origin["size_bytes"], "model_state_digest": "c" * 64,
                },
                runtime_source_binding=runtime_binding,
            )
            return config_path, token_path, ledger_root

        reservation = {
            "minimum_unreserved_free_bytes": 1,
            "maximum_group_elements": 1,
            "maximum_accepts_per_branch": 1,
            "worker_artifact_bytes_before_overhead": 1,
            "reserved_bytes_per_worker": 1,
            "promotion_group_artifact_bytes_before_overhead": 1,
            "reserved_bytes_for_possible_promotion": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, token_path, ledger_root = initialized_test_ledger(root)
            terminal = {
                "proposal_artifact": {
                    "path": str(ledger_root / "proposals" / ("a" * 64 + ".proposal.json")),
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                }
            }
            with (
                patch(
                    "hfw_continuous.supervisor.validated_disk_reservations",
                    return_value=reservation,
                ),
                patch("hfw_continuous.supervisor.subprocess.Popen", return_value=SuccessfulWorker()),
                patch("hfw_continuous.supervisor.read_json_same_fd", return_value=(terminal, {})),
                patch(
                    "hfw_continuous.supervisor.commit_proposal",
                    side_effect=RuntimeError("injected commit failure"),
                ),
            ):
                result = run_continuously(config_path, token_path, ledger_root)
            self.assertEqual(result["status"], "BLOCK_COMMITTER_EXCEPTION_NO_FURTHER_PROPOSALS_COMMITTED")
            self.assertEqual(result_exit_code(result), 1)
            self.assertEqual(Ledger(ledger_root).current_head()["generation"], 0)
            self.assertEqual(len(list((ledger_root / "jobs").glob("*.job.json"))), 1)
            failure_paths = sorted((ledger_root / "failure_receipts").glob("*.failure.json"))
            self.assertEqual(len(failure_paths), 1)
            failure, failure_identity = read_json_same_fd(failure_paths[0])
            self.assertEqual(failure["status"], "BLOCK_COMMITTER_EXCEPTION_SUPERVISOR_STOPPED")
            self.assertEqual(failure_paths[0].name, f"{failure_identity['sha256']}.failure.json")
            self.assertTrue(failure["no_new_wave_will_start"])
            self.assertFalse(failure["in_process_retry_performed"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, token_path, ledger_root = initialized_test_ledger(root)
            with (
                patch(
                    "hfw_continuous.supervisor.validated_disk_reservations",
                    return_value=reservation,
                ),
                patch(
                    "hfw_continuous.supervisor.recover_pending_proposals",
                    side_effect=RuntimeError("injected recovery failure"),
                ),
            ):
                result = run_continuously(config_path, token_path, ledger_root)
            self.assertEqual(
                result["status"],
                "BLOCK_PENDING_PROPOSAL_RECOVERY_EXCEPTION_SUPERVISOR_STOPPED",
            )
            self.assertEqual(result_exit_code(result), 1)
            self.assertEqual(Ledger(ledger_root).current_head()["generation"], 0)
            self.assertEqual(len(list((ledger_root / "jobs").glob("*.job.json"))), 0)

    def test_accepted_history_and_loader_bind_exact_replacement_and_solver_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, runtime_binding = write_test_runtime_config(root, {})
            token_path = root / "tokens.json"
            token_path.write_text(json.dumps({
                "config_sha256": hash_file_same_fd(config_path)["sha256"],
                "runtime_source_binding": runtime_binding,
            }))
            origin = write_exclusive_bytes(root / "origin.pt", b"origin")
            ledger_root = root / "ledger"
            initialize_ledger(
                ledger_root,
                config_identity=hash_file_same_fd(config_path),
                token_manifest_identity=hash_file_same_fd(token_path),
                prepared_parent={"relocated_checkpoint": origin}, baseline_evidence={},
                origin={
                    "checkpoint_sha256": origin["sha256"],
                    "size_bytes": origin["size_bytes"], "model_state_digest": "b" * 64,
                },
                runtime_source_binding=runtime_binding,
            )
            ledger = Ledger(ledger_root)
            current = ledger.current_head()
            exact_tensor = torch.tensor([0.25], dtype=torch.float32)
            delta_tensor = torch.tensor([-0.75], dtype=torch.float32)
            exact_payload = exact_tensor.numpy().astype("<f4", copy=False).tobytes()
            delta_payload = delta_tensor.numpy().astype("<f4", copy=False).tobytes()
            replacement = write_exclusive_bytes(
                ledger_root / "replacement_group_flats" / "replacement.group_flat.f32le",
                exact_payload,
            )
            delta = write_exclusive_bytes(
                ledger_root / "deltas" / "delta.group_delta.f32le", delta_payload
            )
            proposal_sha = "9" * 64
            model_digest = "8" * 64
            checkpoint_path = ledger_root / "promoted_checkpoints/generation-00000001.pt"
            checkpoint_payload = {
                "schema": "hfw_w560_continuous_promoted_checkpoint_v1",
                "origin_checkpoint_sha256": origin["sha256"],
                "parent_checkpoint_sha256": origin["sha256"],
                "generation": 1,
                "accepted_proposal_sha256": proposal_sha,
                "selected_candidate_order": 1,
                "selected_candidate_delta_sha256": delta["sha256"],
                "solver_cumulative_delta_sha256": delta["sha256"],
                "replacement_group_flat_sha256": replacement["sha256"],
                "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
                "solver_cumulative_delta_used_to_reconstruct_group": False,
                "model_state_digest": model_digest,
                "optimizer_present": False,
                "scheduler_present": False,
                "training_resume_authorized": False,
                "model": {"weight": exact_tensor},
            }
            torch.save(checkpoint_payload, checkpoint_path)
            checkpoint = hash_file_same_fd(checkpoint_path)
            decision = {
                "schema": "hfw_w560_continuous_commit_decision_v1",
                "accepted": True,
                "proposal_sha256": proposal_sha,
                "selected_candidate": {
                    "candidate_order": 1,
                    "cumulative_delta": delta,
                    "exact_candidate_group_flat": replacement,
                },
                "child_checkpoint": checkpoint,
                "ledger_delta": delta,
                "ledger_replacement_group_flat": replacement,
            }
            decision_artifact = ledger.append_decision(decision)
            head = {
                "schema": "hfw_w560_continuous_head_v1",
                "generation": 1,
                "parent_head_record_sha256": current["record_sha256"],
                "parent_checkpoint_sha256": origin["sha256"],
                "checkpoint": checkpoint,
                "origin_checkpoint_sha256": origin["sha256"],
                "model_state_digest": model_digest,
                "runtime_manifest_file_sha256": current["runtime_manifest_file_sha256"],
                "runtime_manifest_identity_sha256": current["runtime_manifest_identity_sha256"],
                "accepted_proposal_sha256": proposal_sha,
                "selected_candidate_order": 1,
                "decision_sha256": decision_artifact["sha256"],
                "delta_sha256": delta["sha256"],
                "solver_cumulative_delta_sha256": delta["sha256"],
                "replacement_group_flat_sha256": replacement["sha256"],
                "solver_cumulative_delta": delta,
                "replacement_group_flat": replacement,
                "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
                "group": "base_affines",
                "commit_objective": {},
                "contract_sha256": current["contract_sha256"],
            }
            ledger.append_head(head)
            self.assertEqual(
                ledger.validate_history(verify_checkpoints=True)["status"],
                "PASS_APPEND_ONLY_HEAD_HISTORY",
            )

            class FakeBridge:
                @staticmethod
                def load_completed_w560_parent(derived, source_root, device):
                    model = torch.nn.Module()
                    model.register_parameter(
                        "weight", torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
                    )
                    return SimpleNamespace(
                        model=model, groups={"base_affines": ("weight",)},
                        checkpoint_sha256=origin["sha256"],
                    )

                @staticmethod
                def model_state_digest(state):
                    return model_digest

            fake_branch = SimpleNamespace(
                _tensor_bytes=lambda value: value.detach().cpu().contiguous().numpy().astype(
                    "<f4", copy=False
                ).tobytes()
            )
            loader_config = {
                "origin": {
                    "checkpoint_sha256": origin["sha256"],
                    "relocated_checkpoint_path": origin["path"],
                },
                "source": {"root": str(root)},
            }
            promoted_head = ledger.current_head()
            with (
                patch("hfw_continuous.state.derive_relocated_protocol", return_value=({}, {})),
                patch(
                    "hfw_continuous.state.vendor_modules",
                    return_value=(root, fake_branch, FakeBridge, None, None),
                ),
            ):
                loaded, _ = load_head_model(
                    config_path, promoted_head, torch.device("cpu"), config=loader_config
                )
                self.assertTrue(torch.equal(loaded.model.weight.detach(), exact_tensor))
                replacement_path = Path(replacement["path"])
                replacement_path.chmod(0o644)
                replacement_path.write_bytes(np.array([0.5], dtype="<f4").tobytes())
                with self.assertRaises(RuntimeError):
                    load_head_model(
                        config_path, promoted_head, torch.device("cpu"), config=loader_config
                    )
                with self.assertRaises(RuntimeError):
                    ledger.validate_history(verify_checkpoints=True)


if __name__ == "__main__":
    unittest.main()
