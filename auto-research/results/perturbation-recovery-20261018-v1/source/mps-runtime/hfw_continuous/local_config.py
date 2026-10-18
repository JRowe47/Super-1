from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .common import canonical_json_bytes, hash_file_same_fd, read_json_same_fd
from .release_binding import validate_runtime_manifest


CONFIG_SCHEMA = "hfw_w560_mps_cumulative_local_optimizer_config_v1"
PACKAGE_NAMESPACE = "hfw-w560-continuous-mps-cumulative-optimizer-20260822-v2-attempt1"
EXACT_SOURCE_FILES = {
    "representation_experiments/predictive_models.py": "05560f064eddfe069be3d4459ea1fff8c644286aba101058370516ea5e298db8",
    "representation_experiments/common.py": "eed2cad9f3313b5c889bf72bc72aa78254466ff1e1b5548d363a26b42f7757dd",
    "representation_experiments/nanogpt_control.py": "1adfffbd843a55f00491b50cb46bc6c5d99041f54d9f48251ce3e79ad672a870",
    "third_party/nanogpt/model.py": "7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c",
    "third_party/nanogpt/LICENSE": "a59ec5cdb1c1e447e5266a014b3e7ded511f8d6e5a08931e931c87490ff821fe",
    "config/bottleneck_hfw_1k_100m.json": "f0640b60349940ff69c6ae2f62785954edd07d68984e90132dd428af0ddf83c1",
    "protocol.v1.json": "7a013d9640241577844da13ffb85883cce6b7fab5d50b6e4de924a09fbd62663",
    "release_manifest.v1.json": "d61d003ede83dcfb302be5da854b3628dd53bdd13632e9d92c7fa8a75a16a190",
}


def runtime_root(config_path: Path) -> Path:
    root = Path(config_path).resolve().parents[1]
    expected_mount = Path("/Volumes/SP AI 01_16").resolve()
    if root.name != "runtime" or root.parent.name != PACKAGE_NAMESPACE:
        raise RuntimeError("cumulative local optimizer runtime namespace drift")
    try:
        root.relative_to(expected_mount)
    except ValueError as error:
        raise RuntimeError("cumulative local optimizer is outside the attached SSD") from error
    return root


def _require_runtime_path(root: Path, path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} is outside the cumulative runtime") from error
    if Path(path).is_symlink():
        raise RuntimeError(f"{label} is symlinked")
    return resolved


def load_config(config_path: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    config, identity = read_json_same_fd(config_path)
    root = runtime_root(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("cumulative local optimizer config schema drift")
    runtime = validate_runtime_manifest(config_path, config)
    source = config.get("source", {})
    if source.get("files") != EXACT_SOURCE_FILES:
        raise RuntimeError("exact frozen source inventory drift")
    source_root = _require_runtime_path(root, Path(source["root"]), "frozen source root")
    for relative, expected in EXACT_SOURCE_FILES.items():
        if hash_file_same_fd(source_root / relative)["sha256"] != expected:
            raise RuntimeError(f"frozen source byte drift: {relative}")
    authority = config.get("authority", {})
    exact_authority = {
        "user_authorized_cumulative_local_optimizer_build": True,
        "local_solver_launch_authorized": True,
        "local_verify_and_bounded_smoke_authorized": True,
        "cloud_access_authorized": False,
        "remote_checkpoint_write_authorized": False,
        "training_or_backprop_authorized": False,
        "adam_or_any_optimizer_authorized": False,
    }
    if authority != exact_authority:
        raise RuntimeError("cumulative local optimizer authority drift")
    controller = config.get("controller", {})
    required_controller = {
        "forward_device": "mps",
        "forward_autocast_dtype": "bfloat16",
        "parallel_workers": 1,
        "minimum_relative_local_verify_improvement": 0.0001,
        "duplicate_forward_tolerance": 1e-7,
        "stale_policy": "reject_without_merge_or_rebase",
        "groups": ["base_affines", "left_factors", "right_factors", "codecs"],
        "root_seed": 5609001,
    }
    if controller != required_controller:
        raise RuntimeError("cumulative controller contract drift")
    local_verify = config.get("local_verify", {})
    if local_verify.get("role") != "LOCAL_VERIFY":
        raise RuntimeError("verification role must be exactly LOCAL_VERIFY")
    if local_verify.get("disclosure") != "PERMANENT_REUSED_OPTIMIZATION_DATA_NOT_HELD_OUT":
        raise RuntimeError("LOCAL_VERIFY reuse disclosure drift")
    if local_verify.get("held_out_generalization_claimed") is not False:
        raise RuntimeError("LOCAL_VERIFY cannot be described as held out")
    bank = local_verify.get("bank", {})
    expected_bank = {
        "relative_path": "inputs/local_verify/local_verify.u32le",
        "sha256": "b4592e6295a6e3d8b0f3979df2caea459355467bc97d301bfc62caba15dd7047",
        "size_bytes": 65664,
        "logical_shape": [32, 513],
        "runtime_batch_shape": [8, 4, 513],
        "dtype": "uint32_le",
    }
    if bank != expected_bank:
        raise RuntimeError("LOCAL_VERIFY bank contract drift")
    bank_identity = hash_file_same_fd(root / expected_bank["relative_path"])
    if bank_identity["sha256"] != expected_bank["sha256"] or int(bank_identity["size_bytes"]) != expected_bank["size_bytes"]:
        raise RuntimeError("LOCAL_VERIFY bank bytes drift")
    lineage = config.get("lineage", {})
    exact_lineage = {
        "source_current_head_record_sha256": "980c95f6d938a4661423d08ac0c9f97bdfea542db824b6062f4d6f7229839f01",
        "source_decision_sha256": "0ecf3c0645683edd3189899069682a8a6b279c7231e936d7b723c840a6f1caae",
        "source_parent_checkpoint_sha256": "8b65f885c2e1a81437bedca34511a6c6692c1bcec7671764b2a431f26757b4a6",
        "source_local_promotions": 6,
        "current_checkpoint_sha256": "04830cf33045b19dadb610873770b97e8d0cdb1446b5bccf96389f35b35b76fd",
        "origin_checkpoint_sha256": "0c69fbf81526282d7d68ac258a98867d7de0f0e88e69e5f9f0543fdf1c535fba",
        "source_generation": 7,
    }
    for key, expected in exact_lineage.items():
        if lineage.get(key) != expected:
            raise RuntimeError(f"exact local lineage drift: {key}")
    source_head = hash_file_same_fd(
        _require_runtime_path(
            root, Path(lineage["source_current_head_record_path"]),
            "source authority head",
        )
    )
    if source_head["sha256"] != lineage["source_current_head_record_sha256"]:
        raise RuntimeError("source authority head byte drift")
    source_decision = hash_file_same_fd(
        _require_runtime_path(
            root, Path(lineage["source_decision_path"]),
            "source authority decision",
        )
    )
    if source_decision["sha256"] != lineage["source_decision_sha256"]:
        raise RuntimeError("source authority decision byte drift")
    return config, {**identity, "runtime_source_binding": runtime, "local_verify_bank": bank_identity}


def config_contract_identity(config: Mapping[str, object]) -> str:
    body = dict(config)
    body.pop("runtime_release", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()
