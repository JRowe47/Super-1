from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping

from .common import canonical_json_bytes, hash_file_same_fd, read_json_same_fd, sha256_bytes


CONFIG_SCHEMA = "hfw_w560_promotion_benchmark_config_v1"
MANIFEST_SCHEMA = "hfw_w560_promotion_benchmark_runtime_manifest_v1"
ENVIRONMENT_MANIFEST_SCHEMA = "hfw_w560_benchmark_environment_manifest_v1"


def runtime_root(config_path: Path) -> Path:
    path = Path(config_path).resolve()
    if path.name != "benchmark.json" or path.parent.name != "config" or path.parent.parent.name != "runtime":
        raise RuntimeError("benchmark config must be runtime/config/benchmark.json")
    return path.parent.parent


def load_config(config_path: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    config, identity = read_json_same_fd(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("benchmark config schema drift")
    authority = config.get("authority", {})
    required_false = (
        "network_access_authorized", "cloud_access_authorized",
        "training_or_backprop_authorized", "optimizer_authorized",
        "adam_authorized", "checkpoint_write_authorized", "lineage_write_authorized",
        "promotion_gate_authorized", "benchmark_result_may_gate_promotion",
    )
    if any(authority.get(key) is not False for key in required_false):
        raise RuntimeError("benchmark authority drift")
    if authority.get("public_dataset_prefetch_authorized") is not False:
        raise RuntimeError("attempt2 must not authorize public dataset prefetch")
    if authority.get("local_frozen_asset_import_authorized") is not True:
        raise RuntimeError("local frozen asset import authority missing")
    if config.get("disclosure") != "REPEATED_OR_ADAPTIVELY_USED_BENCHMARKS_ARE_AUDIT_DATA_NOT_HELD_OUT":
        raise RuntimeError("benchmark disclosure drift")
    return config, identity


def _validate_environment_inventory(
    root: Path, config: Mapping[str, object],
) -> Mapping[str, object]:
    environment = config["benchmark_environment"]
    environment_root = root / "environment/benchmark_pydeps"
    expected_pythonpath = f"{root}:{environment_root}"
    if environment.get("pythonpath") != expected_pythonpath:
        raise RuntimeError("benchmark Python path drift")
    if os.environ.get("PYTHONPATH") != expected_pythonpath:
        raise RuntimeError("benchmark process PYTHONPATH drift")
    if (
        environment.get("python_no_site_required") is not True
        or sys.flags.no_site != 1
    ):
        raise RuntimeError("benchmark interpreter must run with -S")
    if (
        environment.get("python_safe_path_required") is not True
        or sys.flags.safe_path is not True
        or os.environ.get("PYTHONSAFEPATH") != "1"
    ):
        raise RuntimeError("benchmark interpreter must run with -P/PYTHONSAFEPATH")
    if list(sys.path) != list(environment["sys_path_allowlist"]):
        raise RuntimeError(f"benchmark interpreter sys.path closure drift: {sys.path}")
    if (
        environment.get("all_regular_environment_files_sha_bound") is not True
        or environment.get("symlinks_authorized") is not False
        or environment.get("readable_pyc_or_pyo_authorized") is not False
        or int(environment.get("readable_bytecode_count", -1)) != 0
    ):
        raise RuntimeError("benchmark environment closure policy drift")
    expected_pycache = str(environment["external_pycache_prefix"])
    if not sys.dont_write_bytecode:
        raise RuntimeError("benchmark interpreter must run with -B/PYTHONDONTWRITEBYTECODE")
    if os.environ.get("PYTHONPYCACHEPREFIX") != expected_pycache:
        raise RuntimeError("benchmark external pycache prefix drift")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("benchmark interpreter must disable user site packages")
    pycache_path = Path(expected_pycache).resolve()
    try:
        pycache_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("benchmark pycache must be external to the sealed package")
    manifest, manifest_file = read_json_same_fd(
        root / str(environment["inventory_relative_path"]),
        str(environment["inventory_file_sha256"]),
    )
    files = manifest.get("files")
    if (
        manifest.get("schema") != ENVIRONMENT_MANIFEST_SCHEMA
        or manifest.get("status") != "SEALED_ALL_REGULAR_ENVIRONMENT_FILES_NO_BYTECODE"
        or not isinstance(files, Mapping)
        or int(manifest.get("file_count", -1)) != len(files)
        or int(environment["inventory_file_count"]) != len(files)
    ):
        raise RuntimeError("benchmark environment manifest drift")
    content_identity = sha256_bytes(canonical_json_bytes({
        "schema": ENVIRONMENT_MANIFEST_SCHEMA,
        "files": files,
    }))
    if (
        manifest.get("manifest_identity_sha256") != content_identity
        or environment["inventory_content_identity_sha256"] != content_identity
    ):
        raise RuntimeError("benchmark environment content identity drift")
    observed_paths = set()
    total_size = 0
    for path in environment_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"benchmark environment symlink is forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(environment_root).as_posix()
        if path.suffix in (".pyc", ".pyo") or "__pycache__" in path.parts:
            raise RuntimeError(f"readable benchmark bytecode is forbidden: {relative}")
        observed_paths.add(relative)
    if observed_paths != set(files):
        missing = sorted(set(files) - observed_paths)[:3]
        extra = sorted(observed_paths - set(files))[:3]
        raise RuntimeError(f"benchmark environment closure drift: missing={missing} extra={extra}")
    for relative, expected in files.items():
        observed = hash_file_same_fd(
            environment_root / str(relative), str(expected["sha256"]),
        )
        if int(observed["size_bytes"]) != int(expected["size_bytes"]):
            raise RuntimeError(f"benchmark environment size drift: {relative}")
        total_size += int(observed["size_bytes"])
    if (
        total_size != int(manifest.get("total_size_bytes", -1))
        or total_size != int(environment["inventory_total_size_bytes"])
    ):
        raise RuntimeError("benchmark environment total-size drift")
    return {
        "manifest_file": manifest_file,
        "manifest_content_identity_sha256": content_identity,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "readable_bytecode_count": 0,
        "closure_exact": True,
    }


def validate_runtime(config_path: Path) -> Mapping[str, object]:
    config, config_identity = load_config(config_path)
    root = runtime_root(config_path)
    manifest_path = root / "config/runtime_manifest.v1.json"
    manifest, manifest_identity = read_json_same_fd(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("runtime manifest schema drift")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or int(manifest.get("file_count", -1)) != len(files):
        raise RuntimeError("runtime manifest inventory drift")
    total = 0
    for relative, expected in files.items():
        path = root / str(relative)
        observed = hash_file_same_fd(path, str(expected["sha256"]))
        if int(observed["size_bytes"]) != int(expected["size_bytes"]):
            raise RuntimeError(f"runtime size drift: {relative}")
        total += int(observed["size_bytes"])
    if total != int(manifest.get("total_size_bytes", -1)):
        raise RuntimeError("runtime total-size drift")
    observed_content_identity = sha256_bytes(canonical_json_bytes({
        "schema": MANIFEST_SCHEMA,
        "files": files,
    }))
    if manifest.get("manifest_identity_sha256") != observed_content_identity:
        raise RuntimeError("runtime manifest content identity drift")
    return {
        "status": "PASS_IMMUTABLE_BENCHMARK_RUNTIME",
        "config": config,
        "config_identity": config_identity,
        "runtime_manifest_identity": manifest_identity,
        "runtime_manifest_content_identity_sha256": observed_content_identity,
    }


def validate_pinned_inputs(config_path: Path) -> Mapping[str, object]:
    validated = validate_runtime(config_path)
    config = validated["config"]
    tokenizer_files = {}
    for relative, expected in config["tokenizer"]["files"].items():
        tokenizer_files[relative] = hash_file_same_fd(
            runtime_root(config_path) / "inputs/tokenizer" / relative, str(expected["sha256"])
        )
    tokenizer_identity, tokenizer_identity_artifact = read_json_same_fd(
        runtime_root(config_path) / config["tokenizer"]["identity_relative_path"],
        str(config["tokenizer"]["identity_sha256"]),
    )
    if (
        tokenizer_identity.get("schema") != "hfw_w560_tokenizer_runtime_identity_v1"
        or tokenizer_identity.get("status") != "PASS_DERIVED_FROM_EXACT_LOCAL_AUTOTOKENIZER"
        or tokenizer_identity.get("all_token_ids_fit_model_vocabulary") is not True
        or tokenizer_identity.get("bos_token") is not None
        or tokenizer_identity.get("bos_token_id") is not None
    ):
        raise RuntimeError("tokenizer identity receipt drift")
    for name, observed in tokenizer_files.items():
        bound = tokenizer_identity["files"][name]
        if bound["sha256"] != observed["sha256"] or int(bound["size_bytes"]) != int(observed["size_bytes"]):
            raise RuntimeError(f"tokenizer identity/file drift: {name}")
    active = config["active_controller"]
    controller_body, controller = read_json_same_fd(
        Path(str(active["config_path"])), str(active["config_sha256"]),
    )
    release_body, release = read_json_same_fd(
        Path(str(active["runtime_manifest_path"])), str(active["runtime_manifest_sha256"]),
    )
    contract_body, contract = read_json_same_fd(
        Path(str(active["lineage_contract_path"])), str(active["lineage_contract_sha256"]),
    )
    run_contract_body, run_contract = read_json_same_fd(
        Path(str(active["run_contract_path"])), str(active["run_contract_sha256"]),
    )
    initial_head_body, initial_head = read_json_same_fd(
        Path(str(active["initial_head_path"])), str(active["initial_head_sha256"]),
    )
    loader_diagnostic_body, loader_diagnostic = read_json_same_fd(
        Path(str(active["loader_diagnostic_path"])), str(active["loader_diagnostic_sha256"]),
    )
    verify_shape_diagnostic_body, verify_shape_diagnostic = read_json_same_fd(
        Path(str(active["verify_shape_diagnostic_path"])), str(active["verify_shape_diagnostic_sha256"]),
    )
    if (
        controller_body.get("schema") != "hfw_w560_mps_winner_continuation_config_v1"
        or controller_body.get("objective", {}).get("mode") != "ce"
        or controller_body.get("authority", {}).get("cloud_access_authorized") is not False
        or controller_body.get("authority", {}).get("training_or_backprop_authorized") is not False
        or controller_body.get("authority", {}).get("adam_or_any_optimizer_authorized") is not False
    ):
        raise RuntimeError("attempt6 controller authority/objective drift")
    if (
        controller_body.get("runtime_release", {}).get("manifest_file_sha256") != release["sha256"]
        or controller_body.get("runtime_release", {}).get("manifest_identity_sha256") != active["runtime_manifest_identity_sha256"]
        or release_body.get("manifest_identity_sha256") != active["runtime_manifest_identity_sha256"]
        or controller_body.get("source", {}).get("files") != active["frozen_source_inventory"]
    ):
        raise RuntimeError("attempt6 release/source binding drift")
    if (
        contract_body.get("config", {}).get("sha256") != controller["sha256"]
        or contract_body.get("runtime_manifest_file_sha256") != release["sha256"]
        or contract_body.get("runtime_manifest_identity_sha256") != active["runtime_manifest_identity_sha256"]
        or contract_body.get("objective") != "ce"
    ):
        raise RuntimeError("attempt6 lineage contract binding drift")
    if (
        initial_head_body.get("contract_sha256") != contract["sha256"]
        or initial_head_body.get("checkpoint", {}).get("sha256") != active["initial_checkpoint_sha256"]
        or initial_head_body.get("objective") != "ce"
    ):
        raise RuntimeError("attempt6 bootstrap head binding drift")
    if (
        loader_diagnostic_body.get("status") != "PASS_NO_FORWARD_CURRENT_WINNER_CHECKPOINT_LOADER"
        or loader_diagnostic_body.get("config_sha256") != controller["sha256"]
        or loader_diagnostic_body.get("current_checkpoint_sha256") != active["initial_checkpoint_sha256"]
        or int(loader_diagnostic_body.get("model_forward_calls", -1)) != 0
    ):
        raise RuntimeError("attempt6 loader diagnostic drift")
    if (
        verify_shape_diagnostic_body.get("config_sha256") != controller["sha256"]
        or verify_shape_diagnostic_body.get("head_record_sha256") != initial_head["sha256"]
        or verify_shape_diagnostic_body.get("checkpoint_sha256") != active["initial_checkpoint_sha256"]
    ):
        raise RuntimeError("attempt6 verify-shape diagnostic drift")
    if (
        run_contract_body.get("config_sha256") != controller["sha256"]
        or run_contract_body.get("ledger_contract_sha256") != contract["sha256"]
        or run_contract_body.get("initial_head_record_sha256") != initial_head["sha256"]
        or run_contract_body.get("loader_diagnostic_sha256") != loader_diagnostic["sha256"]
        or run_contract_body.get("verify_shape_diagnostic_sha256") != verify_shape_diagnostic["sha256"]
    ):
        raise RuntimeError("attempt6 run contract binding drift")
    source_root = Path(str(active["runtime_root"])) / "inputs/frozen_source"
    frozen_source = {}
    for relative, expected_sha256 in active["frozen_source_inventory"].items():
        frozen_source[relative] = hash_file_same_fd(source_root / relative, str(expected_sha256))
    provenance = hash_file_same_fd(
        runtime_root(config_path) / "inputs/dataset_provenance/assets.json",
        str(config["dataset_provenance"]["sha256"]),
    )
    environment_inventory = _validate_environment_inventory(runtime_root(config_path), config)
    python_executable = hash_file_same_fd(
        Path(str(config["benchmark_environment"]["python_executable"])),
        str(config["benchmark_environment"]["python_executable_sha256"]),
    )
    return {
        **validated,
        "status": "PASS_PINNED_CONTROLLER_SOURCE_TOKENIZER_AND_DATASET_REVISIONS",
        "tokenizer_files": tokenizer_files,
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_identity_artifact": tokenizer_identity_artifact,
        "active_controller_config": controller,
        "active_runtime_manifest": release,
        "active_lineage_contract": contract,
        "active_run_contract": run_contract,
        "active_initial_head": initial_head,
        "active_loader_diagnostic": loader_diagnostic,
        "active_verify_shape_diagnostic": verify_shape_diagnostic,
        "frozen_source": frozen_source,
        "dataset_provenance": provenance,
        "benchmark_environment_inventory": environment_inventory,
        "benchmark_python_executable": python_executable,
    }
