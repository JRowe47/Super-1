from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
from typing import Mapping

import numpy as np

from .common import (
    canonical_json_bytes, file_identity, read_json_same_fd, sha256_bytes,
    write_exclusive_json,
)


SOURCE_FILES = (
    "representation_experiments/predictive_models.py",
    "representation_experiments/common.py",
    "representation_experiments/nanogpt_control.py",
    "third_party/nanogpt/model.py",
    "third_party/nanogpt/LICENSE",
    "config/bottleneck_hfw_1k_100m.json",
    "protocol.v1.json",
    "release_manifest.v1.json",
)
CRITICAL_RUNTIME_SOURCES = (
    "transformers/models/qwen3/modeling_qwen3.py",
    "transformers/models/qwen3/configuration_qwen3.py",
    "transformers/models/qwen3/__init__.py",
    "transformers/modeling_utils.py",
    "transformers/tokenization_utils_tokenizers.py",
    "transformers/tokenization_utils_base.py",
    "transformers/utils/hub.py",
)


def _record(path: Path, **extra: object) -> Mapping[str, object]:
    return {**file_identity(path), **extra}


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def freeze_environment(
    package_root: Path, recorded_environment: Path, output: Path,
) -> Mapping[str, object]:
    import huggingface_hub
    import safetensors
    import tokenizers
    import torch
    import transformers

    wheels_root = package_root / "environment/wheels"
    pydeps_root = package_root / "environment/pydeps"
    stale_bytecode = [
        path for path in pydeps_root.rglob("*")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
    ]
    if stale_bytecode:
        raise RuntimeError(
            f"isolated environment contains readable stale bytecode: {stale_bytecode[0]}"
        )
    wheels = {
        path.name: _record(path)
        for path in sorted(wheels_root.glob("*.whl"))
    }
    if not wheels:
        raise RuntimeError("isolated wheel closure is empty")
    sources = {}
    for relative in CRITICAL_RUNTIME_SOURCES:
        path = pydeps_root / relative
        sources[relative] = _record(path)
    versions = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
        "tokenizers": tokenizers.__version__,
    }
    manifest = {
        "schema": "hfw_w560_ce_vs_kd_isolated_mps_environment_v1",
        "status": "PASS_EXACT_LOCAL_WHEEL_AND_CRITICAL_SOURCE_CLOSURE",
        "package_root": str(package_root.resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "pydeps_root": str(pydeps_root.resolve()),
        "wheels_root": str(wheels_root.resolve()),
        "versions": versions,
        "wheels": wheels,
        "critical_installed_sources": sources,
        "recorded_original_execution_environment": _record(recorded_environment),
        "recorded_original_transformers_version": "5.12.1",
        "teacher_publication_metadata_transformers_version": "4.51.0",
        "local_numerical_lane_adaptation": {
            "python": versions["python"],
            "torch": versions["torch"],
            "device": "mps",
            "student_and_teacher_autocast_dtype": "bfloat16",
            "functional_normalization": "cpu_float64_same_operation_order",
        },
        "system_environment_mutated": False,
    }
    return {**manifest, "artifact": write_exclusive_json(output, manifest)}


def _safetensors_summary(path: Path) -> Mapping[str, object]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        header_size = struct.unpack("<Q", os.read(descriptor, 8))[0]
        header = json.loads(os.read(descriptor, header_size))
    finally:
        os.close(descriptor)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    parameter_count = 0
    dtypes = set()
    for record in tensors.values():
        count = 1
        for extent in record["shape"]:
            count *= int(extent)
        parameter_count += count
        dtypes.add(record["dtype"])
    return {
        "tensor_count": len(tensors),
        "parameter_count": parameter_count,
        "dtypes": sorted(dtypes),
        "header_size_bytes": header_size,
    }


def _maximum_token_id(tokenizer_json: Path) -> tuple[int, int]:
    payload = json.loads(tokenizer_json.read_text())
    vocabulary = payload["model"]["vocab"]
    ids = [int(value) for value in vocabulary.values()]
    ids.extend(int(item["id"]) for item in payload.get("added_tokens", ()))
    return int(len(vocabulary)), max(ids)


def _runtime_tokenizer_mapping(snapshot: Path) -> tuple[Mapping[str, object], frozenset[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False,
    )
    mapping = sorted(
        (str(token), int(identifier))
        for token, identifier in tokenizer.get_vocab().items()
    )
    identifiers = frozenset(identifier for _, identifier in mapping)
    if not identifiers:
        raise RuntimeError("runtime tokenizer mapping is empty")
    added = sorted(
        (str(token), int(identifier))
        for token, identifier in tokenizer.get_added_vocab().items()
    )
    summary = {
        "tokenizer_class": type(tokenizer).__name__,
        "declared_base_vocab_size": int(tokenizer.vocab_size),
        "mapping_entry_count": len(mapping),
        "unique_mapped_id_count": len(identifiers),
        "minimum_mapped_id": min(identifiers),
        "maximum_mapped_id": max(identifiers),
        "mapping_sha256": sha256_bytes(canonical_json_bytes(mapping)),
        "added_mapping_entry_count": len(added),
        "added_mapping_sha256": sha256_bytes(canonical_json_bytes(added)),
        "mapped_ids_contiguous_observation": (
            len(identifiers) == max(identifiers) - min(identifiers) + 1
        ),
        "contiguity_required_for_validity": False,
    }
    return summary, identifiers


def _bank_record(path: Path, shape: list[int]) -> Mapping[str, object]:
    body = path.read_bytes()
    values = np.frombuffer(body, dtype="<u4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise RuntimeError(f"bank shape drift: {path}: {values.size} != {expected}")
    return {
        **_record(path),
        "shape": shape,
        "dtype": "uint32_le",
        "maximum_token_id": int(values.max()),
    }


def _vendor(package_root: Path) -> Mapping[str, object]:
    package_name = "hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver"
    root = package_root / "runtime/vendor" / package_name
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files[str(path.relative_to(root))] = _record(path)
    protocol_path = root / "config/production_protocol.json"
    branch_path = root / "branch_search.py"
    bridge_path = root / "production_bridge.py"
    return {
        "package": f"vendor.{package_name}",
        "root": str(root.resolve()),
        "files": files,
        "production_protocol_path": str(protocol_path.resolve()),
        "production_protocol_sha256": files["config/production_protocol.json"]["sha256"],
        "branch_search_sha256": files["branch_search.py"]["sha256"],
        "production_bridge_sha256": files["production_bridge.py"]["sha256"],
    }


def _source(package_root: Path) -> Mapping[str, object]:
    root = package_root / "runtime/inputs/frozen_source"
    return {
        "root": str(root.resolve()),
        "files": {relative: file_identity(root / relative)["sha256"] for relative in SOURCE_FILES},
    }


def _starting_inputs(package_root: Path, environment_manifest: Path) -> Mapping[str, object]:
    inputs = package_root / "runtime/inputs"
    checkpoints = sorted((inputs / "checkpoint").glob("*.pt"))
    heads = sorted((inputs / "checkpoint").glob("*.source.json"))
    if len(checkpoints) != 1 or len(heads) != 1:
        raise RuntimeError("starting checkpoint/head staged inventory is not exact")
    checkpoint = checkpoints[0]
    head = heads[0]
    source_head = read_json_same_fd(head)
    if source_head["checkpoint"]["sha256"] != file_identity(checkpoint)["sha256"]:
        raise RuntimeError("starting checkpoint/head identity drift")
    return {
        "starting_checkpoint": _record(
            checkpoint,
            model_state_digest=source_head["model_state_digest"],
            format="hfw_w560_continuous_promoted_checkpoint_v1",
        ),
        "starting_head": _record(head),
        "environment_manifest": _record(environment_manifest),
    }


def _experiment(
    vendor: Mapping[str, object], precommitted_job_path: Path,
    starting_inputs: Mapping[str, object],
) -> Mapping[str, object]:
    protocol = read_json_same_fd(Path(vendor["production_protocol_path"]))
    search = protocol["search"]
    job = read_json_same_fd(precommitted_job_path)
    job_identity = file_identity(precommitted_job_path)
    starting_head_sha = starting_inputs["starting_head"]["sha256"]
    starting_checkpoint_sha = starting_inputs["starting_checkpoint"]["sha256"]
    if job.get("schema") != "hfw_w560_mps_local_solver_job_v1" or int(job["job_index"]) != 0:
        raise RuntimeError("precommitted A/B job schema/index drift")
    if job["base_head_record_sha256"] != starting_head_sha:
        raise RuntimeError("precommitted A/B job/start head drift")
    if job["base_checkpoint_sha256"] != starting_checkpoint_sha:
        raise RuntimeError("precommitted A/B job/start checkpoint drift")
    starting_head = read_json_same_fd(Path(starting_inputs["starting_head"]["path"]))
    if int(job["base_generation"]) != int(starting_head["generation"]):
        raise RuntimeError("precommitted A/B job/start generation drift")
    if job["group"] not in search["independent_branches"]:
        raise RuntimeError("precommitted A/B job group is outside frozen branch inventory")
    frozen = {
        key: value for key, value in search.items()
        if key != "branch_seeds"
    }
    return {
        "group": job["group"],
        "seed": int(job["seed"]),
        "seed_source": job_identity,
        "starting_head_sha256": starting_head_sha,
        "starting_checkpoint_sha256": starting_checkpoint_sha,
        "starting_generation": int(job["base_generation"]),
        "frozen_solver_settings": frozen,
        "student_forward_budget_per_arm": int(search["model_evaluation_budget"]),
        "paired_generations_per_arm": int(search["pairs_per_branch"]),
        "only_objective_differs_at_shared_initial_state": True,
        "adaptive_accepts_may_cause_later_direction_radius_and_candidate_byte_divergence": True,
    }


def _objectives(source_root: Path) -> Mapping[str, object]:
    frozen_config = json.loads((source_root / "config/bottleneck_hfw_1k_100m.json").read_text())
    training = frozen_config["training"]
    return {
        "ce": {
            "name": "mean_next_token_cross_entropy",
            "full_vocabulary": True,
        },
        "kd": {
            "name": "original_initial_soft_target_distillation_objective",
            "ce_weight": float(training["ce_weight"]["initial"]),
            "distribution_weight": float(training["distribution_weight"]["initial"]),
            "temperatures": training["temperatures"],
            "vocab_chunk_size": int(training["vocab_chunk_size"]),
            "forward_kl_implementation": "frozen_exact_forward_kl_then_sequential_multi_temperature_kl",
            "full_teacher_logits": True,
            "top_k_projection_or_temperature_substitution": False,
        },
    }


def _runtime() -> Mapping[str, object]:
    return {
        "device": "mps",
        "student_autocast_dtype": "bfloat16",
        "teacher_autocast_dtype": "bfloat16",
        "normalization_device": "cpu",
        "normalization_dtype": "float64",
        "conflicting_mps_process_substring": "hfw_continuous.mps_cli run-continuous",
        "cloud_access": False,
        "optimizer_or_backprop": False,
    }


def _output_contract(package_root: Path) -> Mapping[str, object]:
    run_root = package_root / "runtime/runs"
    teacher_root = run_root / "teacher-precompute-attempt1"
    matched_root = run_root / "matched-ab-attempt1"
    return {
        "teacher_smoke": {
            "claim_path": str((teacher_root / "TEACHER_SMOKE_STARTED.json").resolve()),
            "result_path": str((teacher_root / "TEACHER_SMOKE.json").resolve()),
        },
        "teacher_precompute": {
            "claim_path": str((teacher_root / "TEACHER_PRECOMPUTE_STARTED.json").resolve()),
            "result_root": str((teacher_root / "frozen-logits").resolve()),
        },
        "CE_SEARCH": {
            "claim_path": str((matched_root / "CE_SEARCH_STARTED.json").resolve()),
            "result_root": str((matched_root / "ce-search").resolve()),
        },
        "KD_SEARCH": {
            "claim_path": str((matched_root / "KD_SEARCH_STARTED.json").resolve()),
            "result_root": str((matched_root / "kd-search").resolve()),
        },
        "CE_VERIFY": {
            "claim_path": str((matched_root / "CE_VERIFY_STARTED.json").resolve()),
            "result_root": str((matched_root / "ce-verify").resolve()),
        },
        "KD_VERIFY": {
            "claim_path": str((matched_root / "KD_VERIFY_STARTED.json").resolve()),
            "result_root": str((matched_root / "kd-verify").resolve()),
        },
        "FINAL": {
            "claim_path": str((matched_root / "FINAL_STARTED.json").resolve()),
            "result_root": str((matched_root / "final-result").resolve()),
        },
        "failure_policy": "ANY_CLAIMED_PHASE_FAILURE_REQUIRES_FRESH_WHOLE_A_B_PACKAGE_NO_RETRY_OR_SUBSET_SELECTION",
        "attempt_index": 1,
    }


def preregister(
    package_root: Path, teacher_root: Path, environment_manifest: Path,
    recorded_environment: Path, precommitted_job: Path,
    runtime_manifest: Path, output: Path,
) -> Mapping[str, object]:
    source = _source(package_root)
    vendor = _vendor(package_root)
    inputs = _starting_inputs(package_root, environment_manifest)
    inputs["runtime_manifest"] = _record(runtime_manifest)
    frozen_distillation = package_root / "runtime/inputs/frozen_source/distribution_distillation.py"
    inputs = {**inputs, "frozen_distillation_source": _record(frozen_distillation)}
    banks_root = package_root / "runtime/inputs/banks"
    banks = {
        "SEARCH": _bank_record(banks_root / "search/search.u32le", [12, 2, 513]),
        "VERIFY": _bank_record(banks_root / "verify/verify.u32le", [32, 513]),
        "FINAL": _bank_record(banks_root / "final/final.u32le", [64, 513]),
    }
    teacher_files = {
        path.name: _record(path)
        for path in sorted(teacher_root.iterdir()) if path.is_file()
    }
    required_teacher_files = {
        "LICENSE", "config.json", "generation_config.json", "merges.txt",
        "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    }
    if set(teacher_files) != required_teacher_files:
        raise RuntimeError("teacher snapshot file inventory drift")
    safe_summary = _safetensors_summary(teacher_root / "model.safetensors")
    base_vocab, tokenizer_json_maximum_token_id = _maximum_token_id(
        teacher_root / "tokenizer.json"
    )
    runtime_tokenizer_mapping, runtime_mapped_ids = _runtime_tokenizer_mapping(teacher_root)
    teacher_config = json.loads((teacher_root / "config.json").read_text())
    bank_token_validation = {}
    for role, record in banks.items():
        values = np.frombuffer(Path(record["path"]).read_bytes(), dtype="<u4")
        observed = frozenset(int(value) for value in np.unique(values))
        missing = observed - runtime_mapped_ids
        if missing:
            raise RuntimeError(
                f"{role} bank contains IDs absent from pinned runtime tokenizer: "
                f"{sorted(missing)[:8]}"
            )
        if int(values.max()) >= int(teacher_config["vocab_size"]):
            raise RuntimeError(f"{role} bank token exceeds teacher model vocabulary")
        bank_token_validation[role] = {
            "unique_token_id_count": len(observed),
            "maximum_token_id": int(values.max()),
            "all_ids_present_in_runtime_tokenizer_mapping": True,
            "all_ids_below_teacher_model_vocab_size": True,
        }
    environment = read_json_same_fd(environment_manifest)
    if environment.get("status") != "PASS_EXACT_LOCAL_WHEEL_AND_CRITICAL_SOURCE_CLOSURE":
        raise RuntimeError("isolated environment manifest is not terminal")
    if file_identity(recorded_environment)["sha256"] != environment[
        "recorded_original_execution_environment"
    ]["sha256"]:
        raise RuntimeError("recorded original environment binding drift")
    inputs["precommitted_ab_job"] = _record(precommitted_job)
    experiment = _experiment(vendor, precommitted_job, inputs)
    config = {
        "schema": "hfw_w560_ce_vs_kd_matched_ab_config_v1",
        "status": "PREREGISTERED_BEFORE_ANY_AB_TEACHER_OR_AB_STUDENT_MODEL_FORWARD",
        "phase": "PRECOMPUTE",
        "package_root": str(package_root.resolve()),
        "inputs": inputs,
        "source": source,
        "vendor": vendor,
        "banks": banks,
        "experiment": experiment,
        "objectives": _objectives(Path(source["root"])),
        "selection": {
            "minimum_relative_verify_improvement": 0.0001,
            "minimum_relative_final_parent_improvement": 0.0001,
            "duplicate_forward_tolerance": 1e-7,
            "verify_selects_lowest_objective_among_all_saved_candidates_and_parent_noop": True,
            "final_method_winner": "lower_selected_output_mean_CE_then_teacher_KL_exact_tie_break",
            "promotion_is_separate_strict_parent_CE_no_regression_gate": True,
        },
        "teacher": {
            "repository": "Qwen/Qwen3-1.7B-Base",
            "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
            "snapshot_root": str(teacher_root.resolve()),
            "snapshot_files": teacher_files,
            "publication_metadata_transformers_version": teacher_config["transformers_version"],
            "actual_original_execution_transformers_version": environment[
                "recorded_original_transformers_version"
            ],
            "vocab_size": int(teacher_config["vocab_size"]),
            "tokenizer_json_base_vocab_size": base_vocab,
            "tokenizer_json_maximum_mapped_id": tokenizer_json_maximum_token_id,
            "runtime_tokenizer_mapping": runtime_tokenizer_mapping,
            "bank_token_validation": bank_token_validation,
            "parameter_count": safe_summary["parameter_count"],
            "safetensors_summary": safe_summary,
            "fully_offline_local_files_only": True,
            "trust_remote_code": False,
        },
        "teacher_outputs_contract": {
            "roles": {"SEARCH": 24, "VERIFY": 32, "FINAL": 64},
            "shape_per_sequence": [512, int(teacher_config["vocab_size"])],
            "dtype": "bfloat16_little_endian_uint16_payload",
            "expected_total_bytes": (24 + 32 + 64) * 512 * int(teacher_config["vocab_size"]) * 2,
            "teacher_preprocessing_forwards": 120,
            "bounded_smoke_forwards": 1,
            "aggregate_teacher_forwards": 121,
            "teacher_forward_after_freeze": 0,
        },
        "environment": environment,
        "runtime": _runtime(),
        "output_contract": _output_contract(package_root),
        "authority": {
            "local_mps_only": True,
            "cloud": False,
            "teacher_precompute_authorized_after_conflict_free_boundary": True,
            "ce_kd_matched_ab_authorized": True,
            "optimizer_backprop_or_adam": False,
        },
        "final_disclosure": {
            "materialized_before_model_work": True,
            "teacher_only_prelabel_read_before_arms_authorized": True,
            "unopened_by_student_optimizer_or_selection_until_both_selection_receipts": True,
            "one_time_method_selection_data_not_held_out": True,
            "never_available_to_continuation": True,
        },
    }
    return {**config, "artifact": write_exclusive_json(output, config)}


def _phase_common(preregistered: Mapping[str, object], phase: str) -> dict[str, object]:
    return {
        "schema": preregistered["schema"],
        "status": f"BOUND_LEAST_AUTHORITY_{phase}",
        "phase": phase,
        "package_root": preregistered["package_root"],
        "source": preregistered["source"],
        "vendor": preregistered["vendor"],
        "experiment": preregistered["experiment"],
        "runtime": preregistered["runtime"],
        "environment": {"versions": preregistered["environment"]["versions"]},
        "authority": preregistered["authority"],
    }


def bind_search(
    prereg_path: Path, mode: str, output: Path,
    teacher_capability: Path | None,
) -> Mapping[str, object]:
    prereg = read_json_same_fd(prereg_path)
    phase = f"{mode.upper()}_SEARCH"
    config = _phase_common(prereg, phase)
    inputs = {
        key: prereg["inputs"][key]
        for key in (
            "starting_checkpoint", "starting_head", "environment_manifest",
            "precommitted_ab_job", "runtime_manifest",
        )
    }
    if mode == "kd":
        if teacher_capability is None:
            raise RuntimeError("KD SEARCH binding requires teacher capability")
        capability = read_json_same_fd(teacher_capability)
        expected_capability = Path(
            prereg["output_contract"]["teacher_precompute"]["result_root"]
        ) / "SEARCH_TEACHER_CAPABILITY.json"
        if teacher_capability.resolve() != expected_capability.resolve():
            raise RuntimeError("KD SEARCH teacher capability path is not preregistered")
        if set(capability["banks"]) != {"SEARCH"}:
            raise RuntimeError("KD SEARCH capability role drift")
        inputs["frozen_distillation_source"] = prereg["inputs"]["frozen_distillation_source"]
        inputs["teacher_outputs"] = {
            "capability_path": str(teacher_capability.resolve()),
            "manifest_sha256": file_identity(teacher_capability)["sha256"],
            "master_manifest_sha256": capability["master_manifest_sha256"],
            "roles_available": ["SEARCH"],
        }
    elif teacher_capability is not None:
        raise RuntimeError("CE SEARCH cannot bind teacher capability")
    config.update({
        "inputs": inputs,
        "banks": {"SEARCH": prereg["banks"]["SEARCH"]},
        "objectives": {mode: prereg["objectives"][mode]},
        "phase_inputs": {"preregistered_config_sha256": file_identity(prereg_path)["sha256"]},
        "phase_output": prereg["output_contract"][phase],
    })
    return {**config, "artifact": write_exclusive_json(output, config)}


def bind_verify(
    prereg_path: Path, mode: str, arm_terminal: Path, output: Path,
    teacher_capability: Path | None,
) -> Mapping[str, object]:
    prereg = read_json_same_fd(prereg_path)
    phase = f"{mode.upper()}_VERIFY"
    expected_arm = Path(
        prereg["output_contract"][f"{mode.upper()}_SEARCH"]["result_root"]
    ) / "ARM_SEARCH_TERMINAL.json"
    if arm_terminal.resolve() != expected_arm.resolve():
        raise RuntimeError("VERIFY arm terminal path is not the unique preregistered output")
    config = _phase_common(prereg, phase)
    inputs = {
        key: prereg["inputs"][key]
        for key in (
            "starting_checkpoint", "starting_head", "environment_manifest",
            "precommitted_ab_job", "runtime_manifest",
        )
    }
    if mode == "kd":
        if teacher_capability is None:
            raise RuntimeError("KD VERIFY binding requires teacher capability")
        capability = read_json_same_fd(teacher_capability)
        expected_capability = Path(
            prereg["output_contract"]["teacher_precompute"]["result_root"]
        ) / "VERIFY_TEACHER_CAPABILITY.json"
        if teacher_capability.resolve() != expected_capability.resolve():
            raise RuntimeError("KD VERIFY teacher capability path is not preregistered")
        if set(capability["banks"]) != {"VERIFY"}:
            raise RuntimeError("KD VERIFY capability role drift")
        inputs["frozen_distillation_source"] = prereg["inputs"]["frozen_distillation_source"]
        inputs["teacher_outputs"] = {
            "capability_path": str(teacher_capability.resolve()),
            "manifest_sha256": file_identity(teacher_capability)["sha256"],
            "master_manifest_sha256": capability["master_manifest_sha256"],
            "roles_available": ["VERIFY"],
        }
    elif teacher_capability is not None:
        raise RuntimeError("CE VERIFY cannot bind teacher capability")
    config.update({
        "inputs": inputs,
        "banks": {"VERIFY": prereg["banks"]["VERIFY"]},
        "objectives": {mode: prereg["objectives"][mode]},
        "selection": {
            "minimum_relative_verify_improvement": prereg["selection"][
                "minimum_relative_verify_improvement"
            ],
            "duplicate_forward_tolerance": prereg["selection"]["duplicate_forward_tolerance"],
        },
        "phase_inputs": {
            "preregistered_config_sha256": file_identity(prereg_path)["sha256"],
            "arm_terminal": _record(arm_terminal),
        },
        "phase_output": prereg["output_contract"][phase],
    })
    return {**config, "artifact": write_exclusive_json(output, config)}


def bind_final(
    prereg_path: Path, ce_selection: Path, kd_selection: Path,
    teacher_capability: Path, output: Path,
) -> Mapping[str, object]:
    prereg = read_json_same_fd(prereg_path)
    expected_ce_selection = Path(
        prereg["output_contract"]["CE_VERIFY"]["result_root"]
    ) / "VERIFY_SELECTION_TERMINAL.json"
    expected_kd_selection = Path(
        prereg["output_contract"]["KD_VERIFY"]["result_root"]
    ) / "VERIFY_SELECTION_TERMINAL.json"
    expected_teacher_capability = Path(
        prereg["output_contract"]["teacher_precompute"]["result_root"]
    ) / "FINAL_TEACHER_CAPABILITY.json"
    if ce_selection.resolve() != expected_ce_selection.resolve():
        raise RuntimeError("FINAL CE selection is not the unique preregistered output")
    if kd_selection.resolve() != expected_kd_selection.resolve():
        raise RuntimeError("FINAL KD selection is not the unique preregistered output")
    if teacher_capability.resolve() != expected_teacher_capability.resolve():
        raise RuntimeError("FINAL teacher capability path is not preregistered")
    capability = read_json_same_fd(teacher_capability)
    if set(capability["banks"]) != {"FINAL"}:
        raise RuntimeError("FINAL capability role drift")
    config = _phase_common(prereg, "FINAL")
    config.update({
        "inputs": {
            key: prereg["inputs"][key]
            for key in (
                "starting_checkpoint", "starting_head", "environment_manifest",
                "frozen_distillation_source", "precommitted_ab_job", "runtime_manifest",
            )
        },
        "banks": {"FINAL": prereg["banks"]["FINAL"]},
        "objectives": prereg["objectives"],
        "selection": {
            "minimum_relative_final_parent_improvement": prereg["selection"][
                "minimum_relative_final_parent_improvement"
            ],
            "duplicate_forward_tolerance": prereg["selection"]["duplicate_forward_tolerance"],
        },
        "phase_inputs": {
            "preregistered_config_sha256": file_identity(prereg_path)["sha256"],
            "ce_selection": _record(ce_selection),
            "kd_selection": _record(kd_selection),
            "continuation_search_bank_sha256": prereg["banks"]["SEARCH"]["sha256"],
            "continuation_verify_bank_sha256": prereg["banks"]["VERIFY"]["sha256"],
        },
        "phase_output": prereg["output_contract"]["FINAL"],
    })
    config["inputs"]["teacher_outputs"] = {
        "capability_path": str(teacher_capability.resolve()),
        "manifest_sha256": file_identity(teacher_capability)["sha256"],
        "master_manifest_sha256": capability["master_manifest_sha256"],
        "roles_available": ["FINAL"],
    }
    return {**config, "artifact": write_exclusive_json(output, config)}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    env = subparsers.add_parser("environment")
    env.add_argument("--package-root", type=Path, default=_package_root())
    env.add_argument("--recorded-environment", type=Path, required=True)
    env.add_argument("--output", type=Path, required=True)
    pre = subparsers.add_parser("preregister")
    pre.add_argument("--package-root", type=Path, default=_package_root())
    pre.add_argument("--teacher-root", type=Path, required=True)
    pre.add_argument("--environment-manifest", type=Path, required=True)
    pre.add_argument("--recorded-environment", type=Path, required=True)
    pre.add_argument("--precommitted-job", type=Path, required=True)
    pre.add_argument("--runtime-manifest", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    search = subparsers.add_parser("bind-search")
    search.add_argument("--preregistered", type=Path, required=True)
    search.add_argument("--mode", choices=("ce", "kd"), required=True)
    search.add_argument("--teacher-capability", type=Path)
    search.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("bind-verify")
    verify.add_argument("--preregistered", type=Path, required=True)
    verify.add_argument("--mode", choices=("ce", "kd"), required=True)
    verify.add_argument("--arm-terminal", type=Path, required=True)
    verify.add_argument("--teacher-capability", type=Path)
    verify.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("bind-final")
    final.add_argument("--preregistered", type=Path, required=True)
    final.add_argument("--ce-selection", type=Path, required=True)
    final.add_argument("--kd-selection", type=Path, required=True)
    final.add_argument("--teacher-capability", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "environment":
        result = freeze_environment(args.package_root, args.recorded_environment, args.output)
    elif args.command == "preregister":
        result = preregister(
            args.package_root, args.teacher_root, args.environment_manifest,
            args.recorded_environment, args.precommitted_job,
            args.runtime_manifest, args.output,
        )
    elif args.command == "bind-search":
        result = bind_search(
            args.preregistered, args.mode, args.output, args.teacher_capability,
        )
    elif args.command == "bind-verify":
        result = bind_verify(
            args.preregistered, args.mode, args.arm_terminal, args.output,
            args.teacher_capability,
        )
    else:
        result = bind_final(
            args.preregistered, args.ce_selection, args.kd_selection,
            args.teacher_capability, args.output,
        )
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
