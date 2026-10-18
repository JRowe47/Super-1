from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hfw_continuous.common import hash_file_same_fd, write_content_addressed_json
from hfw_continuous.local_config import load_config, runtime_root
from hfw_continuous.local_ledger import HEAD_SCHEMA, DECISION_SCHEMA, LocalLedger, initialize_local_ledger
from hfw_continuous.local_promoter import _load_proposal, evaluate_and_promote
from hfw_continuous.local_runner import initialize_continuous_run, run_continuously
from hfw_continuous.local_solver_worker import run_local_solver_job
from hfw_continuous.local_verify_bank import load_local_verify


RUNTIME = Path(__file__).resolve().parents[1]
CONFIG = RUNTIME / "config/controller.json"


class CumulativeLocalTests(unittest.TestCase):
    def test_exact_separate_local_verify_bank_and_disclosure(self):
        config, _ = load_config(CONFIG)
        values, receipt = load_local_verify(config, runtime_root(CONFIG))
        self.assertEqual(values.shape, (8, 4, 513))
        self.assertEqual(receipt["bank"]["sha256"], "b4592e6295a6e3d8b0f3979df2caea459355467bc97d301bfc62caba15dd7047")
        self.assertEqual(receipt["disclosure"], "PERMANENT_REUSED_OPTIMIZATION_DATA_NOT_HELD_OUT")
        self.assertFalse(receipt["held_out_generalization_claimed"])
        self.assertNotEqual(receipt["bank"]["sha256"], config["search"]["bank"]["sha256"])

    def test_local_verify_intervals_are_disjoint_from_search_and_prior_public(self):
        selection = json.loads((RUNTIME / "config/local_verify_selection.v1.json").read_text())
        search = json.loads((RUNTIME / "inputs/search/search_selection_records.json").read_text())
        length = int(selection["selection"]["sequence_tokens"])
        verify_intervals = [(int(start), int(start) + length) for start in selection["selection"]["starts"]]
        excluded = []
        for row in search["search_records"]:
            excluded.extend(tuple(int(value) for value in interval) for interval in row["intervals"])
        for row in search["prior_public_exclusion"]:
            excluded.extend(tuple(int(value) for value in interval) for interval in row["intervals"])
        for left, right in verify_intervals:
            for other_left, other_right in excluded:
                self.assertTrue(right <= other_left or other_right <= left)

    def test_stale_proposal_rejects_before_local_verify_loader(self):
        test_root = Path(tempfile.mkdtemp(prefix="stale-reject-", dir=RUNTIME / "tests"))
        ledger_root = test_root / "ledger"
        current = initialize_local_ledger(CONFIG, ledger_root)
        source_sha = "0" * 64
        proposal = {
            "schema": "hfw_w560_mps_local_candidate_chain_v1",
            "source_proposal_sha256": source_sha,
            "source_terminal_receipt_sha256": "1" * 64,
            "local_solver_job_record_sha256": None,
            "artifact_root_relative": f"proposal_imports/{source_sha}",
            "base_head_record_sha256": "f" * 64,
            "base_generation": current["generation"],
            "base_checkpoint_sha256": current["checkpoint"]["sha256"],
            "candidate_count": 0,
            "ordered_candidates": [],
            "every_saved_candidate_imported": True,
            "candidate_subset_selection_performed": False,
        }
        artifact = write_content_addressed_json(ledger_root / "proposal_imports", ".local-proposal.json", proposal)
        with mock.patch("hfw_continuous.local_promoter.load_config", side_effect=AssertionError("LOCAL_VERIFY path reached")):
            result = evaluate_and_promote(CONFIG, ledger_root, Path(artifact["path"]))
        self.assertEqual(result["status"], "REJECTED_STALE_HEAD_NO_MERGE_NO_REBASE_NO_LOCAL_VERIFY_OPEN")
        self.assertFalse(result["local_verify_opened"])

    def test_source_and_local_job_artifact_roots_are_exact_and_confined(self):
        test_root = Path(tempfile.mkdtemp(prefix="proposal-roots-", dir=RUNTIME / "tests"))
        ledger_root = test_root / "ledger"
        initialize_local_ledger(CONFIG, ledger_root)
        ledger = LocalLedger(ledger_root, CONFIG)
        current = ledger.current_head()
        forms = (
            {
                "source_proposal_sha256": "3" * 64,
                "source_terminal_receipt_sha256": "4" * 64,
                "local_solver_job_record_sha256": None,
                "artifact_root_relative": f"proposal_imports/{'3' * 64}",
            },
            {
                "source_proposal_sha256": None,
                "source_terminal_receipt_sha256": None,
                "local_solver_job_record_sha256": "5" * 64,
                "artifact_root_relative": f"proposal_imports/local-job-{'5' * 64}",
            },
        )
        for form in forms:
            proposal = {
                "schema": "hfw_w560_mps_local_candidate_chain_v1",
                **form,
                "base_head_record_sha256": current["record_sha256"],
                "base_generation": current["generation"],
                "base_checkpoint_sha256": current["checkpoint"]["sha256"],
                "candidate_count": 0,
                "ordered_candidates": [],
                "every_saved_candidate_imported": True,
                "candidate_subset_selection_performed": False,
            }
            artifact = write_content_addressed_json(ledger_root / "proposal_imports", ".local-proposal.json", proposal)
            loaded, _ = _load_proposal(ledger, Path(artifact["path"]))
            self.assertEqual(loaded["artifact_root_relative"], form["artifact_root_relative"])

    def test_next_job_reads_new_current_head_after_append_only_promotion(self):
        test_root = Path(tempfile.mkdtemp(prefix="allocator-current-head-", dir=RUNTIME / "tests"))
        ledger_root = test_root / "ledger"
        initialize_local_ledger(CONFIG, ledger_root)
        ledger = LocalLedger(ledger_root, CONFIG)
        current = ledger.current_head()
        decision = {
            "schema": DECISION_SCHEMA,
            "status": "TEST_ACCEPTED_APPEND_ONLY_HEAD",
            "accepted": True,
            "proposal_sha256": "2" * 64,
            "child_checkpoint": current["checkpoint"],
        }
        decision_artifact = ledger.append_decision(decision)
        head = deepcopy({key: value for key, value in current.items() if key not in {"record_sha256", "record_path"}})
        head.update({
            "schema": HEAD_SCHEMA,
            "generation": int(current["generation"]) + 1,
            "local_promotions": int(current["local_promotions"]) + 1,
            "parent_head_record_sha256": current["record_sha256"],
            "parent_checkpoint_sha256": current["checkpoint"]["sha256"],
            "accepted_proposal_sha256": "2" * 64,
            "decision_sha256": decision_artifact["sha256"],
        })
        new_head_artifact = ledger.append_head(head)
        job = ledger.allocate_next_job()
        self.assertEqual(job["base_head_record_sha256"], new_head_artifact["sha256"])
        self.assertEqual(job["base_checkpoint_sha256"], head["checkpoint"]["sha256"])
        self.assertEqual(job["base_generation"], head["generation"])

    def test_saved_proposal_is_verify_pending_and_cannot_be_skipped(self):
        test_root = Path(tempfile.mkdtemp(prefix="no-orphan-jobs-", dir=RUNTIME / "tests"))
        ledger_root = test_root / "ledger"
        initialize_local_ledger(CONFIG, ledger_root)
        ledger = LocalLedger(ledger_root, CONFIG)
        first = ledger.allocate_next_job()
        states = ledger.job_states()
        self.assertEqual([(row["job_index"], row["stage"]) for row in states], [(0, "SOLVER_PENDING")])
        proposal = {
            "schema": "hfw_w560_mps_local_candidate_chain_v1",
            "source_proposal_sha256": None,
            "source_terminal_receipt_sha256": None,
            "local_solver_job_record_sha256": first["job_record_sha256"],
            "artifact_root_relative": f"proposal_imports/local-job-{first['job_record_sha256']}",
            "base_head_record_sha256": first["base_head_record_sha256"],
            "base_generation": first["base_generation"],
            "base_checkpoint_sha256": first["base_checkpoint_sha256"],
            "candidate_count": 0,
            "ordered_candidates": [],
            "every_saved_candidate_imported": True,
            "candidate_subset_selection_performed": False,
        }
        proposal_artifact = write_content_addressed_json(ledger_root / "proposal_imports", ".local-proposal.json", proposal)
        states = ledger.job_states()
        self.assertEqual([(row["job_index"], row["stage"]) for row in states], [(0, "VERIFY_PENDING")])
        self.assertEqual(states[0]["proposal"]["artifact"]["sha256"], proposal_artifact["sha256"])
        with self.assertRaisesRegex(RuntimeError, "prior local job is incomplete"):
            ledger.allocate_next_job()
        decision = {
            "schema": DECISION_SCHEMA,
            "status": "REJECTED_NO_CANDIDATE_PASSED_DUPLICATED_LOCAL_VERIFY_THRESHOLD",
            "accepted": False,
            "proposal_sha256": proposal_artifact["sha256"],
            "local_verify_opened": True,
            "local_verify_evidence": {
                "all_saved_candidates_evaluated": True,
                "candidate_count": 0,
            },
            "child_checkpoint": None,
        }
        ledger.append_decision(decision)
        self.assertEqual(ledger.job_states()[0]["stage"], "COMPLETE")
        successor = ledger.allocate_next_job()
        self.assertEqual([row["job_index"] for row in ledger.jobs()], [0, 1])
        incomplete = ledger.incomplete_job_states()
        self.assertEqual(len(incomplete), 1)
        self.assertEqual(incomplete[0]["stage"], "SOLVER_PENDING")
        self.assertEqual(incomplete[0]["job_record_sha256"], successor["job_record_sha256"])
        self.assertEqual(successor["base_head_record_sha256"], ledger.current_head()["record_sha256"])

    def test_accepted_decision_without_head_blocks_successor(self):
        test_root = Path(tempfile.mkdtemp(prefix="accepted-unheaded-", dir=RUNTIME / "tests"))
        ledger_root = test_root / "ledger"
        initialize_local_ledger(CONFIG, ledger_root)
        ledger = LocalLedger(ledger_root, CONFIG)
        job = ledger.allocate_next_job()
        proposal = {
            "schema": "hfw_w560_mps_local_candidate_chain_v1",
            "source_proposal_sha256": None,
            "source_terminal_receipt_sha256": None,
            "local_solver_job_record_sha256": job["job_record_sha256"],
            "artifact_root_relative": f"proposal_imports/local-job-{job['job_record_sha256']}",
            "base_head_record_sha256": job["base_head_record_sha256"],
            "base_generation": job["base_generation"],
            "base_checkpoint_sha256": job["base_checkpoint_sha256"],
            "candidate_count": 1,
            "ordered_candidates": [{"candidate_order": 1}],
            "every_saved_candidate_imported": True,
            "candidate_subset_selection_performed": False,
        }
        proposal_artifact = write_content_addressed_json(ledger_root / "proposal_imports", ".local-proposal.json", proposal)
        current = ledger.current_head()
        decision = {
            "schema": DECISION_SCHEMA,
            "status": "ACCEPTED_BEST_DUPLICATED_LOCAL_VERIFY_CANDIDATE",
            "accepted": True,
            "proposal_sha256": proposal_artifact["sha256"],
            "local_verify_opened": True,
            "local_verify_evidence": {
                "all_saved_candidates_evaluated": True,
                "candidate_count": 1,
            },
            "child_checkpoint": current["checkpoint"],
        }
        ledger.append_decision(decision)
        state = ledger.job_states()[0]
        self.assertEqual(state["stage"], "ACCEPTED_DECISION_UNHEADED_BLOCKED")
        with self.assertRaisesRegex(RuntimeError, "prior local job is incomplete"):
            ledger.allocate_next_job()

    def test_runner_resumes_saved_chain_at_verify_before_successor(self):
        test_root = Path(tempfile.mkdtemp(prefix="resume-verify-before-successor-", dir=RUNTIME / "tests"))
        ledger_root = test_root / "ledger"
        run_root = test_root / "run"
        initialize_local_ledger(CONFIG, ledger_root)
        ledger = LocalLedger(ledger_root, CONFIG)
        job = ledger.allocate_next_job()
        proposal = {
            "schema": "hfw_w560_mps_local_candidate_chain_v1",
            "source_proposal_sha256": None,
            "source_terminal_receipt_sha256": None,
            "local_solver_job_record_sha256": job["job_record_sha256"],
            "artifact_root_relative": f"proposal_imports/local-job-{job['job_record_sha256']}",
            "base_head_record_sha256": job["base_head_record_sha256"],
            "base_generation": job["base_generation"],
            "base_checkpoint_sha256": job["base_checkpoint_sha256"],
            "candidate_count": 0,
            "ordered_candidates": [],
            "every_saved_candidate_imported": True,
            "candidate_subset_selection_performed": False,
        }
        proposal_artifact = write_content_addressed_json(ledger_root / "proposal_imports", ".local-proposal.json", proposal)
        initialize_continuous_run(CONFIG, ledger_root, run_root)
        config, config_identity = load_config(CONFIG)
        authorized = deepcopy(config)
        authorized["authority"] = dict(config["authority"])
        authorized["authority"]["local_solver_launch_authorized"] = True
        observed = []

        def fake_evaluate(config_path, observed_ledger_root, proposal_path):
            self.assertEqual(Path(proposal_path), Path(proposal_artifact["path"]))
            observed.append(Path(proposal_path))
            decision = {
                "schema": DECISION_SCHEMA,
                "status": "REJECTED_NO_CANDIDATE_PASSED_DUPLICATED_LOCAL_VERIFY_THRESHOLD",
                "accepted": False,
                "proposal_sha256": proposal_artifact["sha256"],
                "local_verify_opened": True,
                "local_verify_evidence": {
                    "all_saved_candidates_evaluated": True,
                    "candidate_count": 0,
                },
                "child_checkpoint": None,
            }
            artifact = LocalLedger(observed_ledger_root, config_path).append_decision(decision)
            return {**decision, "decision_artifact": artifact}

        with mock.patch("hfw_continuous.local_runner.load_config", return_value=(authorized, config_identity)), mock.patch(
            "hfw_continuous.local_runner.run_local_solver_job",
            side_effect=AssertionError("saved proposal was incorrectly rerun through solver"),
        ), mock.patch("hfw_continuous.local_runner.evaluate_and_promote", side_effect=fake_evaluate):
            receipt = run_continuously(CONFIG, ledger_root, run_root, maximum_cycles=1)
        self.assertEqual(receipt["completed_cycles"], 1)
        self.assertEqual(observed, [Path(proposal_artifact["path"])])
        states = ledger.job_states()
        self.assertEqual([(row["job_index"], row["stage"]) for row in states], [(0, "COMPLETE"), (1, "SOLVER_PENDING")])

    def test_long_solver_launch_is_explicitly_authorized(self):
        config, _ = load_config(CONFIG)
        self.assertTrue(config["authority"]["local_solver_launch_authorized"])

    def test_no_training_cloud_or_optimizer_runtime_capability(self):
        forbidden = ("torch.optim", ".backward(", "requires_grad_(True)", "import requests", "import subprocess", "google.cloud", "boto3")
        for path in sorted((RUNTIME / "hfw_continuous").glob("*.py")):
            source = path.read_text()
            for fragment in forbidden:
                self.assertNotIn(fragment, source, f"forbidden capability in {path.name}: {fragment}")
        config, _ = load_config(CONFIG)
        self.assertFalse(config["authority"]["cloud_access_authorized"])
        self.assertFalse(config["authority"]["training_or_backprop_authorized"])
        self.assertFalse(config["authority"]["adam_or_any_optimizer_authorized"])


if __name__ == "__main__":
    unittest.main()
