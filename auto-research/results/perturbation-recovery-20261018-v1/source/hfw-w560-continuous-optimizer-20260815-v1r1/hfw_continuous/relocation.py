from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
from typing import Mapping

from .common import hash_file_same_fd, read_json_same_fd, write_exclusive_json
from .release_binding import validate_runtime_manifest


def load_config(config_path: Path) -> Mapping[str, object]:
    config = read_json_same_fd(config_path)[0]
    validate_runtime_manifest(config_path, config)
    return config


def vendor_modules(config: Mapping[str, object]):
    package = str(config["vendor"]["package"])
    branch = importlib.import_module(f"{package}.branch_search")
    bridge = importlib.import_module(f"{package}.production_bridge")
    manifest = importlib.import_module(f"{package}.manifest")
    parent_gate = importlib.import_module(f"{package}.parent_gate")
    package_root = Path(branch.__file__).resolve().parent
    _, protocol_identity = read_json_same_fd(
        package_root / "config/production_protocol.json",
        config["vendor"]["production_protocol_sha256"],
    )
    observed = {
        "branch_search_sha256": hash_file_same_fd(package_root / "branch_search.py")["sha256"],
        "production_bridge_sha256": hash_file_same_fd(package_root / "production_bridge.py")["sha256"],
        "production_protocol_sha256": protocol_identity["sha256"],
    }
    for key, value in observed.items():
        if value != config["vendor"][key]:
            raise RuntimeError(f"vendored source drift: {key}")
    return package_root, branch, bridge, manifest, parent_gate


def derive_relocated_protocol(
    config_path: Path, *, config: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    config = load_config(config_path) if config is None else config
    runtime_binding = validate_runtime_manifest(config_path, config)
    package_root, _, _, _, parent_gate = vendor_modules(config)
    protocol, _ = read_json_same_fd(
        package_root / "config/production_protocol.json",
        config["vendor"]["production_protocol_sha256"],
    )
    frozen = protocol["parent"]
    origin = config["origin"]
    exact_fields = {
        "checkpoint_path": "original_checkpoint_path",
        "checkpoint_sha256": "checkpoint_sha256",
        "size_bytes": "size_bytes",
        "minimum_link_count": "minimum_link_count",
        "review_receipt_sha256": "review_receipt_sha256",
        "review_schema": "review_schema",
        "checkpoint_identity_sha256": "checkpoint_identity_sha256",
        "model_state_digest": "model_state_digest",
        "completed_step": "step",
        "completed_tokens": "tokens",
    }
    for frozen_key, config_key in exact_fields.items():
        if frozen[frozen_key] != origin[config_key]:
            raise RuntimeError(f"origin contract drift: {frozen_key}")
    checkpoint_path = Path(origin["relocated_checkpoint_path"])
    checkpoint = hash_file_same_fd(checkpoint_path)
    if checkpoint["sha256"] != origin["checkpoint_sha256"]:
        raise RuntimeError("relocated checkpoint SHA drift")
    if int(checkpoint["size_bytes"]) != int(origin["size_bytes"]):
        raise RuntimeError("relocated checkpoint size drift")
    link_count = int(checkpoint["link_count"])
    if link_count < int(origin["minimum_link_count"]):
        raise RuntimeError("relocated checkpoint hardlink count is below contract")
    review_path = Path(origin["relocated_review_receipt_path"])
    review = hash_file_same_fd(review_path)
    if review["sha256"] != origin["review_receipt_sha256"]:
        raise RuntimeError("relocated review receipt SHA drift")
    source_root = Path(config["source"]["root"])
    for relative, expected in config["source"]["files"].items():
        if hash_file_same_fd(source_root / relative)["sha256"] != expected:
            raise RuntimeError(f"relocated frozen source drift: {relative}")
    derived = deepcopy(protocol)
    derived["parent"]["checkpoint_path"] = str(checkpoint_path)
    derived["parent"]["inode"] = int(checkpoint["inode"])
    derived["parent"]["minimum_link_count"] = int(origin["minimum_link_count"])
    derived["parent"]["review_receipt_path"] = str(review_path)
    derived["frozen_source"]["source_root"] = str(source_root)
    gate = parent_gate.validate_immutable_parent(derived)
    receipt = {
        "schema": "hfw_w560_relocated_parent_prepared_receipt_v1",
        "status": "PASS_RELOCATED_PARENT_EXACT_BYTES_NEW_STAT_BOUND_IN_MEMORY",
        "original_checkpoint_path": origin["original_checkpoint_path"],
        "relocated_checkpoint": {**checkpoint, "link_count": link_count},
        "relocated_review_receipt": review,
        "original_checkpoint_sha256": origin["checkpoint_sha256"],
        "original_size_bytes": origin["size_bytes"],
        "review_receipt_sha256": origin["review_receipt_sha256"],
        "checkpoint_identity_sha256": origin["checkpoint_identity_sha256"],
        "model_state_digest": origin["model_state_digest"],
        "source_root": str(source_root),
        "parent_gate": gate,
        "vendored_production_protocol_edited": False,
        "runtime_source_binding": runtime_binding,
    }
    return derived, receipt


def prepare_relocated_parent(config_path: Path, output_path: Path) -> Mapping[str, object]:
    _, receipt = derive_relocated_protocol(config_path)
    artifact = write_exclusive_json(output_path, receipt)
    return {**receipt, "prepared_receipt_artifact": artifact}
