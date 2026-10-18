from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import hfw_dist.run_pipeline as pipeline_module
import hfw_dist.promote_mps as promoter_module
import hfw_dist.daytona_manager as manager_module
import hfw_dist.runtime_manifest as runtime_manifest_module
from hfw_dist.bind_campaign import _group_sizes, _solver_contract
from hfw_dist.collect import (
    _contract_value_matches,
    _validate_cloud_handoff,
    _validate_terminal_duplicate_against_cached,
)
from hfw_dist.common import (
    atomic_current_pointer, hash_file, write_exclusive, write_json_exclusive,
)
from hfw_dist.daytona_manager import ManagerError
from hfw_dist.ledger import DistributedLedger, HEAD_SCHEMA
from hfw_dist.run_pipeline import _manager_command
from hfw_dist.runtime_manifest import (
    _normalized_test_run_declarations,
    _validate_test_run_receipt,
    _validate_test_staging_path_binding,
    _validate_upload_snapshot_staging,
    create_test_run_receipt,
    validate_host_runtime,
)
from hfw_dist.seal_upload_staging import seal_upload_staging


RUNTIME = Path(__file__).resolve().parents[1]
FROZEN = RUNTIME / "frozen"


class StaticContractTests(unittest.TestCase):
    def test_recovery_v3_uses_generic_failure_and_three_id_intent_fields(self):
        manager_source = (
            RUNTIME / "hfw_dist/daytona_manager.py"
        ).read_text()
        manifest_source = (
            RUNTIME / "hfw_dist/runtime_manifest.py"
        ).read_text()
        pipeline_source = (
            RUNTIME / "hfw_dist/run_pipeline.py"
        ).read_text()
        self.assertIn(
            "hfw_w560_recovered_task0_provenance_v3", manager_source,
        )
        self.assertIn(
            "hfw_w560_recovery_precloud_zero_resource_gate_v3",
            manager_source,
        )
        self.assertIn(
            '"required_exact_id_reconciliation"', manager_source,
        )
        self.assertEqual(
            manifest_source.count(
                "hfw_w560_recovered_task0_provenance_v3"
            ),
            2,
        )
        for field in (
            '"recovery_failure_chain"',
            '"recovery_required_exact_id_reconciliation"',
            '"recovery_required_exact_sandbox_ids"',
            '"recovery_all_required_exact_ids_gate_required_before_cloud_create"',
        ):
            self.assertIn(field, pipeline_source)
        for obsolete in (
            '"recovery_predecessor_receipt_chain_head"',
            '"recovery_predecessor_failure_receipt"',
            '"recovery_predecessor_cleanup_terminal_receipt"',
            '"recovery_dual_exact_id_gate_required_before_cloud_create"',
        ):
            self.assertNotIn(obsolete, pipeline_source)

    @staticmethod
    def _make_upload_tree_read_only(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(root, 0o555)

    @staticmethod
    def _make_upload_tree_writable(root: Path) -> None:
        if not root.exists():
            return
        os.chmod(root, 0o755)
        for path in root.rglob("*"):
            if path.is_symlink():
                continue
            os.chmod(path, 0o755 if path.is_dir() else 0o644)

    @staticmethod
    def _staging_fixture(root: Path) -> tuple[Path, Path, Path]:
        package = root / "package"
        wave = package / "runtime/waves/wave-00000001"
        snapshot = wave / "snapshot"
        (snapshot / "nested/empty").mkdir(parents=True)
        write_json_exclusive(snapshot / "CAMPAIGN.json", {"wave_index": 1})
        write_json_exclusive(
            snapshot / "SNAPSHOT_MANIFEST.json", {"status": "TEST"},
        )
        write_exclusive(snapshot / "nested/payload.bin", b"staging-v2-test")
        upload = root / "upload"
        shutil.copytree(snapshot, upload, copy_function=shutil.copy2)
        StaticContractTests._make_upload_tree_read_only(upload)
        return package, wave, upload

    def test_staging_v3_accepts_distinct_non_nested_device_independent_copies(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            package, wave, upload = self._staging_fixture(Path(directory))
            try:
                receipt = seal_upload_staging(package, wave, upload)
                self.assertEqual(
                    receipt["schema"],
                    "hfw_w560_daytona_upload_snapshot_staging_v3",
                )
                self.assertEqual(
                    receipt["status"],
                    "PASS_IMMUTABLE_NO_SPACE_DEVICE_INDEPENDENT_COPY",
                )
                self.assertEqual(receipt["complete_directories"], [
                    "nested", "nested/empty",
                ])
                self.assertEqual(
                    receipt["package_root_device"], receipt["upload_root_device"],
                )
                self.assertTrue(
                    receipt[
                        "source_and_upload_corresponding_files_have_disjoint_device_inode_identities"
                    ]
                )
                for pair in receipt["corresponding_file_identities"].values():
                    self.assertNotEqual(
                        (pair["source"]["device"], pair["source"]["inode"]),
                        (pair["upload"]["device"], pair["upload"]["inode"]),
                    )
                    self.assertEqual(pair["upload"]["link_count"], 1)
                validated = _validate_upload_snapshot_staging(package, wave)
                self.assertEqual(validated["identity"], receipt["identity"])
            finally:
                self._make_upload_tree_writable(upload)

    def test_staging_v3_rejects_upload_hardlinks_and_nested_roots(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            package, wave, copied_upload = self._staging_fixture(root)
            self._make_upload_tree_writable(copied_upload)
            shutil.rmtree(copied_upload)
            source = wave / "snapshot"
            hardlinked_upload = root / "hardlinked"
            for source_path in sorted(source.rglob("*")):
                relative = source_path.relative_to(source)
                destination = hardlinked_upload / relative
                if source_path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source_path, destination)
            self._make_upload_tree_read_only(hardlinked_upload)
            try:
                with self.assertRaisesRegex(RuntimeError, "hardlink"):
                    seal_upload_staging(package, wave, hardlinked_upload)
            finally:
                self._make_upload_tree_writable(hardlinked_upload)

            nested_upload = package / "nested-upload"
            shutil.copytree(source, nested_upload, copy_function=shutil.copy2)
            self._make_upload_tree_read_only(nested_upload)
            try:
                with self.assertRaisesRegex(RuntimeError, "distinct and non-nested"):
                    seal_upload_staging(package, wave, nested_upload)
            finally:
                self._make_upload_tree_writable(nested_upload)

    @staticmethod
    def _runner_fixture(
        root: Path, names: tuple[str, ...] = ("test_alpha", "test_beta"),
    ) -> tuple[Path, Path, list[dict[str, object]]]:
        package = root / "package"
        tests = package / "runtime/tests"
        tests.mkdir(parents=True)
        interpreter = Path(sys.executable).absolute()
        runs = []
        for index, name in enumerate(names):
            source = tests / f"{name}.py"
            source.write_text(
                "from pathlib import Path\n"
                "import os\n"
                "import sys\n"
                "import unittest\n\n"
                "class RunnerOwnedTest(unittest.TestCase):\n"
                "    def test_exact_runner_context(self):\n"
                "        self.assertEqual(Path.cwd(), Path(__file__).resolve().parents[2])\n"
                "        self.assertEqual(os.environ['PYTHONDONTWRITEBYTECODE'], '1')\n"
                "        self.assertNotIn('HOME', os.environ)\n"
                f"        print('stdout-{name}')\n"
                f"        print('stderr-{name}', file=sys.stderr)\n",
                encoding="utf-8",
            )
            relative = source.relative_to(package).as_posix()
            module = f"tests.{source.stem}"
            runs.append({
                "name": f"host-{index}",
                "interpreter": str(interpreter),
                "environment_profile": "host",
                "test_modules": {module: relative},
                "timeout_seconds": 30,
            })
        return package, root / "production-upload", runs

    def test_test_run_receipt_executes_and_derives_exact_evidence(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            package, production_staging, runs = self._runner_fixture(
                Path(directory),
            )
            receipt = create_test_run_receipt(
                package,
                production_staging_path=production_staging,
                runs=runs,
            )
            self.assertEqual(receipt["run_count"], 2)
            self.assertEqual(receipt["total_tests_run"], 2)
            self.assertTrue(receipt["runner_owned_subprocess_execution"])
            self.assertEqual(
                receipt["production_staging_path"], str(production_staging),
            )
            self.assertFalse(production_staging.exists())
            for index, run in enumerate(receipt["runs"]):
                self.assertEqual(run["exit_code"], 0)
                self.assertEqual(run["tests_run"], 1)
                self.assertEqual(run["command"][1:5], [
                    "-B", "-m", "unittest", "-v",
                ])
                log = package / run["result_log"]["path"]
                output = log.read_text()
                self.assertIn(f"stdout-test_{'alpha' if index == 0 else 'beta'}", output)
                self.assertIn(f"stderr-test_{'alpha' if index == 0 else 'beta'}", output)
            validated = _validate_test_run_receipt(package)
            self.assertEqual(validated["identity"], receipt["identity"])
            idempotent = create_test_run_receipt(
                package,
                production_staging_path=production_staging,
                runs=runs,
            )
            self.assertEqual(idempotent["identity"], receipt["identity"])

    def test_test_run_commands_use_profile_specific_isolation_flags(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            package, unused_staging, runs = self._runner_fixture(
                Path(directory), names=("test_alpha",),
            )
            unused_sources, host = _normalized_test_run_declarations(
                package, runs=runs,
            )
            manager_run = [{**runs[0], "environment_profile": "manager"}]
            unused_sources, manager = _normalized_test_run_declarations(
                package, runs=manager_run,
            )
            self.assertEqual(host[0]["command"][1], "-B")
            self.assertNotIn("-P", host[0]["command"])
            self.assertEqual(manager[0]["command"][1:3], ["-P", "-B"])

    def test_test_run_rejects_fake_log_without_execution(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            package, production_staging, runs = self._runner_fixture(
                Path(directory), names=("test_alpha",),
            )
            fake = package / "runtime/test-results/run-000-host-0.log"
            write_exclusive(fake, b"Ran 1 test in 0.1s\n\nOK\n")
            with self.assertRaisesRegex(RuntimeError, "exists before runner execution"):
                create_test_run_receipt(
                    package,
                    production_staging_path=production_staging,
                    runs=runs,
                )
            self.assertFalse(
                (package / "runtime/config/TEST_RUN_RECEIPT.json").exists()
            )

    def test_test_run_rejects_module_source_mapping_drift(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            package, production_staging, runs = self._runner_fixture(
                Path(directory), names=("test_alpha", "test_beta"),
            )
            bad = [dict(run) for run in runs]
            bad[0]["test_modules"] = {
                "tests.test_alpha": "runtime/tests/test_beta.py",
            }
            with self.assertRaisesRegex(RuntimeError, "not canonical"):
                create_test_run_receipt(
                    package,
                    production_staging_path=production_staging,
                    runs=bad,
                )
            self.assertFalse((package / "runtime/test-results").exists())

    def test_test_run_validation_rejects_tampered_runner_log(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            package, production_staging, runs = self._runner_fixture(
                Path(directory), names=("test_alpha",),
            )
            receipt = create_test_run_receipt(
                package,
                production_staging_path=production_staging,
                runs=runs,
            )
            log = package / receipt["runs"][0]["result_log"]["path"]
            log.chmod(0o644)
            with log.open("ab") as handle:
                handle.write(b"tamper\n")
            with self.assertRaisesRegex(RuntimeError, "identity drift"):
                _validate_test_run_receipt(package)

    def test_test_receipt_staging_path_must_match_upload_receipt(self):
        path = "/private/tmp/hfw_attempt8_production_upload"
        test_receipt = {"production_staging_path": path}
        self.assertEqual(
            _validate_test_staging_path_binding(
                test_receipt, {"upload_snapshot_root": path},
            ),
            path,
        )
        with self.assertRaisesRegex(RuntimeError, "differs"):
            _validate_test_staging_path_binding(
                test_receipt,
                {"upload_snapshot_root": path + "-different"},
            )

    def test_worker_contract_keeps_boolean_flags_distinct_from_integer_counts(self):
        self.assertTrue(_contract_value_matches(False, False))
        self.assertTrue(_contract_value_matches(0, 0))
        self.assertFalse(_contract_value_matches(False, 0))
        self.assertFalse(_contract_value_matches(0, False))

    def test_terminal_duplicate_is_recomputed_from_cached_loss_vector(self):
        duplicate = {
            "status": "PASS_TERMINAL_FULL_BANK_DUPLICATES_CACHED_COMMIT_GATE",
            "flat_sha256": "a" * 64,
            "batch_count": 2,
            "maximum_absolute_difference": 0.0,
            "absolute_tolerance": 1e-7,
        }
        self.assertEqual(
            _validate_terminal_duplicate_against_cached(
                terminal_losses=[1.0, 2.0],
                cached_losses=[1.0, 2.0],
                duplicate=duplicate,
                current_flat_sha256="a" * 64,
                sealed_absolute_tolerance=1e-7,
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate evidence"):
            _validate_terminal_duplicate_against_cached(
                terminal_losses=[1.0, 2.0 + 5e-8],
                cached_losses=[1.0, 2.0],
                duplicate=duplicate,
                current_flat_sha256="a" * 64,
                sealed_absolute_tolerance=1e-7,
            )
        with self.assertRaisesRegex(RuntimeError, "duplicate evidence"):
            _validate_terminal_duplicate_against_cached(
                terminal_losses=[1.0, 2.0 + 2e-7],
                cached_losses=[1.0, 2.0],
                duplicate={
                    **duplicate,
                    "maximum_absolute_difference": 2e-7,
                },
                current_flat_sha256="a" * 64,
                sealed_absolute_tolerance=1e-7,
            )

    def test_host_manifest_v2_validates_disposable_sealed_dependencies(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            package = Path(directory) / "package"
            payload = package / "runtime/hfw_dist/payload.py"
            payload_identity = write_exclusive(payload, b"# sealed test payload\n")
            manager_identity = {"sha256": "a" * 64, "size_bytes": 1}
            test_receipt_identity = {"sha256": "b" * 64, "size_bytes": 2}
            upload_identity = {"sha256": "c" * 64, "size_bytes": 3}
            manifest = {
                "schema": "hfw_w560_distributed_host_runtime_manifest_v2",
                "status": "SEALED_NOT_LAUNCHED",
                "wave_index": 1,
                "inventory": {
                    "runtime/hfw_dist/payload.py": {
                        "sha256": payload_identity["sha256"],
                        "size_bytes": payload_identity["size_bytes"],
                    },
                },
                "manager_venv_audit": manager_identity,
                "test_run_receipt": {
                    "receipt": test_receipt_identity,
                    "run_count": 2,
                    "total_tests_run": 5,
                    "test_sources_sha256": "d" * 64,
                    "production_staging_untouched_by_tests": True,
                    "production_staging_path": str(Path(directory) / "upload"),
                },
                "upload_snapshot_staging": {
                    "receipt": upload_identity,
                    "upload_snapshot_root": str(Path(directory) / "upload"),
                    "complete_inventory_sha256": "e" * 64,
                    "file_count": 3,
                    "directory_count": 1,
                    "total_size_bytes": 10,
                },
                "recovered_task0_provenance": None,
            }
            write_json_exclusive(
                package / "runtime/config/HOST_RUNTIME_MANIFEST.json", manifest,
            )
            with patch.object(
                runtime_manifest_module, "_host_files", return_value=[payload],
            ), patch.object(
                runtime_manifest_module, "_validate_manager_venv",
                return_value={"identity": manager_identity},
            ), patch.object(
                runtime_manifest_module, "_validate_test_run_receipt",
                return_value={
                    "identity": test_receipt_identity,
                    "run_count": 2,
                    "total_tests_run": 5,
                    "test_sources_sha256": "d" * 64,
                    "production_staging_path": str(Path(directory) / "upload"),
                },
            ), patch.object(
                runtime_manifest_module, "_validate_upload_snapshot_staging",
                return_value={
                    "identity": upload_identity,
                    "upload_snapshot_root": str(Path(directory) / "upload"),
                    "complete_inventory_sha256": "e" * 64,
                    "file_count": 3,
                    "directory_count": 1,
                    "total_size_bytes": 10,
                },
            ):
                observed = validate_host_runtime(package)
            self.assertEqual(observed["schema"], manifest["schema"])
            self.assertEqual(observed["inventory"], manifest["inventory"])

    def test_manager_command_has_no_credential_channel(self):
        package_root = RUNTIME.parent
        manager_python = package_root / "runtime/daytona-manager-venv/bin/python"
        args = SimpleNamespace(
            manager_python=manager_python, base_image="python:test@sha256:" + "a" * 64,
            upload_snapshot_root=Path(
                "/private/tmp/hfw_w560_attempt8_test_only_upload_snapshot"
            ),
            api_url="https://app.daytona.io/api", target="us",
            worker_timeout_seconds=10, campaign_timeout_seconds=20,
            api_timeout_seconds=5, poll_seconds=1, maximum_attempts=1,
            create_concurrency=1, retry_initial_seconds=1.0,
            retry_maximum_seconds=2.0, vcpu_hour_rate_usd="0.0504",
            memory_gib_hour_rate_usd="0.0162", disk_gib_hour_rate_usd="0.000108",
        )
        command = _manager_command(args, package_root, package_root / "wave")
        joined = "\0".join(command)
        self.assertNotIn("DAYTONA_API_KEY", joined)
        self.assertNotIn("dtn_", joined)
        self.assertIn("hfw_dist.daytona_manager", command)
        self.assertEqual(command[1:3], ["-P", "-B"])
        self.assertEqual(
            command[command.index("--disk-gib-hour-rate-usd") + 1], "0.000108",
        )
        self.assertEqual(
            command[command.index("--upload-snapshot-root") + 1],
            str(args.upload_snapshot_root),
        )
        self.assertEqual(command[0], os.path.abspath(os.fspath(manager_python)))
        audit = subprocess.run([
            command[0], "-c",
            "import importlib.metadata,json,sys; "
            "print(json.dumps({'prefix':sys.prefix,'sdk':importlib.metadata.version('daytona')}))",
        ], check=True, capture_output=True, text=True)
        observed = json.loads(audit.stdout)
        self.assertEqual(Path(observed["prefix"]), manager_python.parents[1])
        self.assertEqual(observed["sdk"], "0.194.0")

    def test_frozen_identities_and_derived_groups(self):
        campaign = json.loads(
            (
                RUNTIME
                / "waves/wave-00000000/snapshot/CAMPAIGN.json"
            ).read_text()
        )
        self.assertEqual(
            hash_file(
                FROZEN / "vendor/hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver"
                / "branch_search.py"
            )["sha256"],
            campaign["solver"]["vendor_branch_search_sha256"],
        )
        self.assertEqual(
            hash_file(FROZEN / "search.u32le")["sha256"],
            campaign["search_bank_sha256"],
        )
        groups, protocol = _solver_contract(FROZEN)
        sizes = _group_sizes(FROZEN, groups, protocol)
        self.assertEqual(tuple(sizes), groups)
        self.assertTrue(all(value > 0 for value in sizes.values()))

    def test_hybrid_contract_forbids_cpu_filter_and_requires_all_candidates(self):
        campaign = json.loads(
            (RUNTIME / "waves/wave-00000000/snapshot/CAMPAIGN.json").read_text()
        )
        rule = campaign["selection_rule"]
        self.assertEqual(rule["mode"], "all_unique_provisionally_accepted_candidates")
        self.assertIs(rule["every_accepted_candidate_downloaded"], True)
        self.assertIs(rule["cpu_ranking_filter_permitted"], False)
        self.assertIs(
            rule["every_unique_accepted_candidate_evaluated_twice_on_canonical_mps_bf16"],
            True,
        )
        worker_source = (RUNTIME / "hfw_dist/cpu_proposal_worker.py").read_text()
        manager_source = (RUNTIME / "hfw_dist/daytona_manager.py").read_text()
        promoter_source = (RUNTIME / "hfw_dist/promote_mps.py").read_text()
        for source in (worker_source, manager_source):
            self.assertIn("accepted_candidates.f32le.bundle", source)
        self.assertIn("read_file_region", promoter_source)

    def test_current_pointer_is_recoverable_derived_state(self):
        temporary_parent = RUNTIME / "tests/.temporary"
        temporary_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint_identity = write_exclusive(checkpoint, b"checkpoint-test")
            ledger = DistributedLedger(root / "lineage")
            initial = ledger.initialize(
                contract={"status": "TEST"},
                initial_head={
                    "status": "TEST", "model_generation": 0, "local_promotions": 0,
                    "source_head_record_sha256": "0" * 64,
                    "source_head_record_path": str(root / "source.head.json"),
                    "parent_checkpoint_sha256": "1" * 64,
                    "checkpoint": checkpoint_identity,
                    "origin_checkpoint_sha256": "2" * 64,
                    "model_state_digest": "3" * 64,
                    "accepted_proposal_sha256": "4" * 64,
                    "selected_candidate_order": 1, "group": "test",
                    "decision_sha256": None,
                    "replacement_group_flat_sha256": "5" * 64,
                    "solver_cumulative_delta_sha256": "6" * 64,
                },
            )
            second = ledger.append_head({
                **{key: value for key, value in initial.items()
                   if key not in {"record_sha256", "record_path"}},
                "schema": HEAD_SCHEMA, "authority_generation": 1,
                "parent_head_record_sha256": initial["record_sha256"],
                "parent_checkpoint_sha256": checkpoint_identity["sha256"],
            })
            atomic_current_pointer(ledger.root / "CURRENT.json", {
                "schema": "hfw_w560_distributed_current_head_pointer_v1",
                "authority_generation": 0,
                "record_sha256": initial["record_sha256"],
                "record_path": initial["record_path"],
            })
            with self.assertRaisesRegex(RuntimeError, "CURRENT pointer"):
                ledger.current_head()
            with ledger.lock():
                recovered = ledger.recover_current_pointer_locked()
            self.assertEqual(recovered["record_sha256"], second["record_sha256"])

    @unittest.skip("fresh hybrid campaign has no recovered task0 dependency")
    def test_promoter_revalidates_recovery_dependency_under_both_locks_before_verify_or_append(self):
        temporary_parent = RUNTIME / "tests/.temporary"
        temporary_parent.mkdir(exist_ok=True)
        sealed_package = RUNTIME.parent
        provenance = json.loads(
            (
                sealed_package
                / "runtime/config/RECOVERED_TASK0_PROVENANCE.json"
            ).read_text()
        )
        dependency = provenance["external_lineage_dependency"]
        predecessor = provenance["predecessor_recovery_failure"]
        immediate_predecessor = provenance[
            "immediate_predecessor_recovery_failure"
        ]
        required_exact_ids = provenance["required_exact_id_reconciliation"]
        self.assertEqual(
            provenance["schema"],
            "hfw_w560_recovered_task0_provenance_v3",
        )
        self.assertEqual(len(required_exact_ids), 3)
        self.assertEqual(
            len({row["sandbox"]["id"] for row in required_exact_ids}),
            3,
        )
        tamper_targets = {
            "external_head": Path(dependency["source_head_path"]).resolve(),
            "external_checkpoint": Path(
                dependency["source_checkpoint_path"]
            ).resolve(),
            "local_current": (
                sealed_package / "runtime/authority/lineage/CURRENT.json"
            ).resolve(),
            "predecessor_host": (
                Path(predecessor["package_root"])
                / "runtime/config/HOST_RUNTIME_MANIFEST.json"
            ).resolve(),
            "immediate_predecessor_host": (
                Path(immediate_predecessor["package_root"])
                / "runtime/config/HOST_RUNTIME_MANIFEST.json"
            ).resolve(),
        }
        real_content_record = manager_module._content_record
        real_read_json = manager_module.read_json
        for tamper in (
            "external_head", "external_checkpoint", "local_current",
            "predecessor_host", "immediate_predecessor_host",
        ):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory(
                dir=temporary_parent,
            ) as directory:
                package = Path(directory) / "package"
                wave = package / "runtime/waves/wave-00000001"
                snapshot = wave / "snapshot"
                snapshot.mkdir(parents=True)
                host_lock_path = package / "runtime/host-mps.lock"
                campaign = {
                    "wave_index": 1,
                    "promotion": {
                        "host_mps_lock_path": str(host_lock_path),
                        "winner_ordering": [
                            "passes_both_duplicated_relative_thresholds",
                            "mps_verify_replicated_mean_ascending",
                            "shortlist_candidate_order_ascending",
                        ],
                    },
                }
                write_json_exclusive(snapshot / "CAMPAIGN.json", campaign)
                write_json_exclusive(
                    wave / "COLLECTION_COMPLETE.json", {"status": "TEST"},
                )
                lineage = package / "runtime/authority/lineage"
                (lineage / "heads").mkdir(parents=True)
                (lineage / "decisions").mkdir()
                events = []
                objective_constructions = []
                validation_calls = 0

                def validate_recovery(unused_package, *, wave_index):
                    nonlocal validation_calls
                    validation_calls += 1
                    events.append((
                        f"validate-{validation_calls}",
                        "host-lock" in events,
                        "ledger-lock" in events,
                    ))
                    return manager_module.validate_recovered_task0_provenance(
                        sealed_package, wave_index=wave_index,
                    )

                def content_record(path):
                    value = real_content_record(path)
                    if (
                        validation_calls == 2
                        and tamper in {
                            "external_head", "external_checkpoint",
                            "predecessor_host", "immediate_predecessor_host",
                        }
                        and Path(path).resolve() == tamper_targets[tamper]
                    ):
                        return {**value, "sha256": "f" * 64}
                    return value

                def recovery_read_json(path, expected_sha256=None):
                    value, identity = real_read_json(path, expected_sha256)
                    if (
                        validation_calls == 2
                        and tamper == "local_current"
                        and Path(path).resolve() == tamper_targets[tamper]
                    ):
                        value = {
                            **value,
                            "record_sha256": "f" * 64,
                            "record_path": str(sealed_package / "wrong.head.json"),
                        }
                    return value, identity

                @contextmanager
                def host_lock(unused_path):
                    events.append("host-lock")
                    yield

                class FakeLedger:
                    def __init__(self, root):
                        self.root = root

                    def contract(self):
                        return {"host_mps_lock_path": str(host_lock_path)}, {}

                    @contextmanager
                    def lock(self):
                        events.append("ledger-lock")
                        yield

                    def recover_current_pointer_locked(self):
                        events.append("current-recovery")
                        raise AssertionError("CURRENT recovery reached after tamper")

                    def append_named_record(self, *unused, **unused_keywords):
                        raise AssertionError("decision or intent append reached after tamper")

                    def append_head(self, *unused, **unused_keywords):
                        raise AssertionError("head append reached after tamper")

                def objective(*unused, **unused_keywords):
                    objective_constructions.append(True)
                    raise AssertionError("MPS VERIFY objective reached after tamper")

                with patch.object(
                    promoter_module, "validate_recovered_task0_provenance",
                    side_effect=validate_recovery,
                ), patch.object(
                    manager_module, "_content_record",
                    side_effect=content_record,
                ), patch.object(
                    manager_module, "read_json",
                    side_effect=recovery_read_json,
                ), patch.object(
                    promoter_module, "DistributedLedger", FakeLedger,
                ), patch.object(
                    promoter_module, "exclusive_lock", side_effect=host_lock,
                ), patch.object(
                    promoter_module, "_validated_shortlist",
                    return_value=(
                        {"candidate_count": 0, "candidates": []},
                        {"sha256": "b" * 64, "size_bytes": 1},
                        {},
                    ),
                ), patch.object(
                    promoter_module, "MPSDuplicateVerifyObjective",
                    side_effect=objective,
                ), self.assertRaises(ManagerError):
                    promoter_module.promote_wave(package, wave)

                self.assertEqual(validation_calls, 2)
                self.assertEqual(events[:4], [
                    ("validate-1", False, False),
                    "host-lock",
                    "ledger-lock",
                    ("validate-2", True, True),
                ])
                self.assertNotIn("current-recovery", events)
                self.assertEqual(objective_constructions, [])
                self.assertEqual(
                    list((lineage / "decisions").iterdir()), [],
                )
                self.assertEqual(
                    list((lineage / "heads").iterdir()), [],
                )

    def test_cloud_handoff_requires_download_launch_and_zero_resource_cleanup(self):
        temporary_parent = RUNTIME / "tests/.temporary"
        temporary_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
            wave = Path(directory)
            campaign = {
                "wave_index": 0, "worker_count": 1,
                "proposal_lane": {"sealed_environment": {
                    "python_major_minor": "3.12", "platform_system": "Linux",
                    "machine": "x86_64", "torch_version": "2.8.0+cpu",
                    "torch_cuda_version": None, "numpy_version": "2.0.2",
                }},
            }
            campaign_identity = write_json_exclusive(wave / "CAMPAIGN.json", campaign)
            launch = {
                "schema": "hfw_w560_daytona_launch_manifest_v1",
                "status": "SEED_WORKER_0_PASS_BEFORE_TASKS_1_THROUGH_99",
                "campaign_sha256": campaign_identity["sha256"], "wave_index": 0,
                "sealed_worker_environment": campaign["proposal_lane"]["sealed_environment"],
                "seed_worker": {
                    "task_index": 0, "result_sha256": "a" * 64,
                    "flat_sha256": "b" * 64, "peak_rss_bytes": 10,
                    "memory_limit_bytes": 20,
                },
            }
            launch_identity = write_json_exclusive(wave / "LAUNCH_MANIFEST.json", launch)
            downloads = {
                "schema": "hfw_w560_daytona_cloud_download_terminal_v1",
                "status": "PASS_ALL_100_PROVISIONAL_CPU_OUTPUTS_DOWNLOADED",
                "campaign_sha256": campaign_identity["sha256"], "wave_index": 0,
                "worker_count": 1, "launch_manifest": launch_identity,
                "workers": [{
                    "task_index": 0, "result_sha256": "a" * 64,
                    "result_size_bytes": 1, "flat_sha256": "b" * 64,
                    "flat_size_bytes": 4, "peak_rss_bytes": 10,
                }],
            }
            cleanup = {
                "schema": "hfw_w560_daytona_cloud_cleanup_terminal_v1",
                "status": "PASS_ZERO_CAMPAIGN_SANDBOXES_AND_SNAPSHOT_ABSENT",
                "campaign_sha256": campaign_identity["sha256"], "wave_index": 0,
                "campaign_sandboxes_remaining": 0, "snapshot_present": False,
                "exact_campaign_label_query": True,
                "snapshot_not_found_confirmed": True, "cleanup_failures": [],
            }
            write_json_exclusive(wave / "CLOUD_DOWNLOADS_COMPLETE.json", downloads)
            write_json_exclusive(wave / "CLOUD_CLEANUP_COMPLETE.json", cleanup)
            observed_downloads, observed_cleanup, by_task = _validate_cloud_handoff(
                wave, campaign, campaign_identity,
            )
            self.assertEqual(observed_downloads["worker_count"], 1)
            self.assertEqual(observed_cleanup["campaign_sandboxes_remaining"], 0)
            self.assertEqual(sorted(by_task), [0])

    def test_pipeline_removes_credential_before_local_authority_stages(self):
        temporary_parent = RUNTIME / "tests/.temporary"
        temporary_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
            package = Path(directory) / "package"
            wave = package / "runtime/waves/wave-00000000"
            module_root = package / "runtime/hfw_dist"
            snapshot = wave / "snapshot"
            module_root.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            for name in ("daytona_manager.py", "collect.py", "promote_mps.py"):
                write_exclusive(module_root / name, name.encode("utf-8"))
            write_exclusive(
                package / "runtime/daytona-manager-venv/bin/python",
                b"test-only-manager-entrypoint",
            )
            campaign_identity = write_json_exclusive(snapshot / "CAMPAIGN.json", {
                "wave_index": 0,
            })
            snapshot_identity = write_json_exclusive(
                snapshot / "SNAPSHOT_MANIFEST.json", {"test": "snapshot"},
            )
            args = SimpleNamespace(
                package_root=package, wave_root=wave,
                manager_python=package / "runtime/daytona-manager-venv/bin/python",
                upload_snapshot_root=package / "upload-snapshot",
                base_image="python:test@sha256:" + "a" * 64,
                api_url="https://app.daytona.io/api", target="us",
                worker_timeout_seconds=10, campaign_timeout_seconds=20,
                api_timeout_seconds=5, poll_seconds=1, maximum_attempts=1,
                create_concurrency=1, retry_initial_seconds=1.0,
                retry_maximum_seconds=2.0, vcpu_hour_rate_usd="0.0504",
                memory_gib_hour_rate_usd="0.0162", disk_gib_hour_rate_usd="0.000108",
            )
            stages = []

            def fake_manager(command, env, check):
                self.assertTrue(check)
                self.assertEqual(env["DAYTONA_API_KEY"], "test-only-secret")
                self.assertEqual(env["PYTHONNOUSERSITE"], "1")
                self.assertEqual(env["PYTHONPATH"], str(package / "runtime"))
                self.assertNotIn("test-only-secret", "\0".join(command))
                self.assertEqual(
                    command[command.index("--upload-snapshot-root") + 1],
                    str(args.upload_snapshot_root),
                )
                stages.append("manager")
                for filename in (
                    "CLOUD_DOWNLOADS_COMPLETE.json", "LAUNCH_MANIFEST.json",
                    "CLOUD_CLEANUP_COMPLETE.json",
                ):
                    write_json_exclusive(wave / filename, {"test": filename})

            def fake_handoff(*unused):
                self.assertNotIn("DAYTONA_API_KEY", os.environ)
                self.assertTrue((wave / "CLOUD_CLEANUP_COMPLETE.json").is_file())
                stages.append("cleanup_gate")
                return {}, {}, {}

            def fake_collect(*unused):
                self.assertNotIn("DAYTONA_API_KEY", os.environ)
                stages.append("collect")
                return {"terminal_identity": {"sha256": "b" * 64}}

            def fake_promote(*unused):
                self.assertNotIn("DAYTONA_API_KEY", os.environ)
                stages.append("promote")
                return {
                    "terminal_identity": {"sha256": "c" * 64},
                    "status": "REJECTED_TEST_ONLY", "accepted": False,
                    "current_head_record_sha256": "d" * 64,
                }

            with patch.dict(os.environ, {"DAYTONA_API_KEY": "test-only-secret"}), patch.object(
                pipeline_module, "validate_host_runtime",
                return_value={
                    "identity": {"sha256": "e" * 64},
                    "wave_index": 0,
                    "campaign": {
                        "sha256": campaign_identity["sha256"],
                        "size_bytes": campaign_identity["size_bytes"],
                    },
                    "snapshot_manifest": {
                        "sha256": snapshot_identity["sha256"],
                        "size_bytes": snapshot_identity["size_bytes"],
                    },
                    "upload_snapshot_staging": {
                        "upload_snapshot_root": str(args.upload_snapshot_root),
                        "receipt": {"sha256": "1" * 64, "size_bytes": 1},
                        "complete_inventory_sha256": "2" * 64,
                        "file_count": 1, "total_size_bytes": 1,
                    },
                    "inventory": {
                        "runtime/daytona-manager-requirements.lock": {
                            "sha256": "f" * 64, "size_bytes": 1,
                        },
                        "runtime/config/DAYTONA_MANAGER_VENV_AUDIT.json": {
                            "sha256": "0" * 64, "size_bytes": 1,
                        },
                    },
                },
            ), patch.object(
                pipeline_module.subprocess, "run", side_effect=fake_manager,
            ), patch.object(
                pipeline_module, "_validate_cloud_handoff", side_effect=fake_handoff,
            ), patch.object(
                pipeline_module, "collect_wave", side_effect=fake_collect,
            ), patch.object(
                pipeline_module, "promote_wave", side_effect=fake_promote,
            ):
                terminal = pipeline_module.run_pipeline(args)
            self.assertEqual(stages, ["manager", "cleanup_gate", "collect", "promote"])
            self.assertFalse(terminal["checkpoint_promoted"])
            self.assertEqual(campaign_identity["sha256"], terminal["campaign_sha256"])

    def test_pipeline_rejects_wave_bytes_outside_host_runtime_seal(self):
        temporary_parent = RUNTIME / "tests/.temporary"
        temporary_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
            package = Path(directory) / "package"
            wave = package / "runtime/waves/wave-00000000"
            snapshot = wave / "snapshot"
            snapshot.mkdir(parents=True)
            campaign_identity = write_json_exclusive(
                snapshot / "CAMPAIGN.json", {"wave_index": 0},
            )
            snapshot_identity = write_json_exclusive(
                snapshot / "SNAPSHOT_MANIFEST.json", {"test": "snapshot"},
            )
            sealed = {
                "wave_index": 0,
                "campaign": {
                    "sha256": "f" * 64,
                    "size_bytes": campaign_identity["size_bytes"],
                },
                "snapshot_manifest": {
                    "sha256": snapshot_identity["sha256"],
                    "size_bytes": snapshot_identity["size_bytes"],
                },
            }
            with patch.object(
                pipeline_module, "validate_host_runtime", return_value=sealed,
            ), self.assertRaisesRegex(RuntimeError, "campaign differs"):
                pipeline_module.run_pipeline(SimpleNamespace(
                    package_root=package, wave_root=wave,
                ))


if __name__ == "__main__":
    unittest.main()
