from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .manifest import derive_group_manifest, file_sha256, load_protocol
from .static_gate import validate_no_training_source_gate


def validate_local_release(
    package_root: Path, source_root: Path | None = None
) -> Mapping[str, object]:
    freeze = package_root / "freeze"
    required = (
        "exact_group_manifest.v1.json",
        "no_training_source_gate.v1.json",
        "local_test_receipt.v1.json",
        "source_manifest.v1.json",
        "release_manifest.v1.json",
        "package_archive.v1.tar",
        "package_archive_receipt.v1.json",
    )
    if any(not (freeze / name).is_file() or (freeze / name).is_symlink() for name in required):
        raise RuntimeError("release inventory is incomplete or symlinked")
    source_manifest = json.loads((freeze / "source_manifest.v1.json").read_text())
    for relative, expected in source_manifest["files"].items():
        path = package_root / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise RuntimeError(f"frozen package source drift: {relative}")
    release = json.loads((freeze / "release_manifest.v1.json").read_text())
    for relative, expected in release["freeze_artifacts"].items():
        if file_sha256(freeze / relative) != expected:
            raise RuntimeError(f"frozen receipt drift: {relative}")
    protocol = load_protocol(package_root / "config/production_protocol.json")
    bound_source = (
        Path(protocol["frozen_source"]["local_review_root"])
        if source_root is None else Path(source_root)
    )
    derived = derive_group_manifest(protocol, bound_source)
    frozen_group = json.loads((freeze / "exact_group_manifest.v1.json").read_text())
    if derived != frozen_group:
        raise RuntimeError("frozen exact group manifest is not reproducible")
    static = validate_no_training_source_gate(package_root)
    if static["status"] != "PASS_NO_AUTOGRAD_NO_OPTIMIZER_SOURCE":
        raise RuntimeError("no-training source gate failed at release validation")
    tests = json.loads((freeze / "local_test_receipt.v1.json").read_text())
    if tests.get("status") != "PASS" or int(tests.get("failed", -1)) != 0:
        raise RuntimeError("local tests did not pass")
    archive = json.loads((freeze / "package_archive_receipt.v1.json").read_text())
    if file_sha256(freeze / "package_archive.v1.tar") != archive["sha256"]:
        raise RuntimeError("package archive SHA drift")
    if protocol["authority"] != {
        "stage_authorized": False,
        "launch_authorized": False,
        "remote_access_authorized_by_this_package": False,
    }:
        raise RuntimeError("local package unexpectedly grants execution authority")
    return {
        "schema": "hfw_w560_local_release_validation_receipt_v1",
        "status": "PASS_LOCAL_RELEASE_CREATE_ONLY",
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "canonical_weight_manifest_sha256": protocol["canonical_weight_address_binding"]["manifest_sha256"],
        "exact_group_manifest_sha256": file_sha256(freeze / "exact_group_manifest.v1.json"),
        "source_manifest_sha256": file_sha256(freeze / "source_manifest.v1.json"),
        "release_manifest_sha256": file_sha256(freeze / "release_manifest.v1.json"),
        "archive_sha256": archive["sha256"],
        "tests_passed": tests["passed"],
        "tests_failed": 0,
        "parent_accessed": False,
        "remote_accessed": False,
        "stage_authorized": False,
        "launch_authorized": False,
    }
