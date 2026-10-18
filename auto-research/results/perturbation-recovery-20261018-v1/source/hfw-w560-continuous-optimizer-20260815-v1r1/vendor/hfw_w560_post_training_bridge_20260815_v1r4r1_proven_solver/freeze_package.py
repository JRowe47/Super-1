from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

from .manifest import (
    canonical_json_bytes, derive_group_manifest, file_sha256, load_protocol,
)
from .release_validator import validate_local_release
from .static_gate import validate_no_training_source_gate


def _exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_json(path: Path, payload: object) -> None:
    _exclusive(path, canonical_json_bytes(payload))


def _source_files(package_root: Path):
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(package_root)
        if relative.parts[0] == "freeze" or "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        yield relative, path


def main() -> int:
    package_root = Path(__file__).resolve().parent
    freeze = package_root / "freeze"
    freeze.mkdir(exist_ok=True)
    if any(freeze.iterdir()):
        raise RuntimeError("create-only freeze directory is not empty")
    protocol = load_protocol(package_root / "config/production_protocol.json")
    group = derive_group_manifest(protocol, Path(protocol["frozen_source"]["local_review_root"]))
    static = validate_no_training_source_gate(package_root)
    environment = dict(os.environ)
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/hfw_w560_freeze_pyc"
    test = subprocess.run(
        [sys.executable, "-m", f"{__package__}.tests.run_tests"],
        cwd=package_root.parents[1], env=environment,
        check=False, capture_output=True, text=True,
    )
    output = test.stdout + test.stderr
    match = re.search(r"RESULT passed=(\d+) failed=(\d+) seconds=([0-9.]+)", output)
    if test.returncode or match is None or int(match.group(2)) != 0:
        raise RuntimeError(f"local test gate failed before freeze:\n{output}")
    test_receipt = {
        "schema": "hfw_w560_local_test_receipt_v1",
        "status": "PASS",
        "passed": int(match.group(1)),
        "failed": int(match.group(2)),
        "seconds": float(match.group(3)),
        "command": f"{sys.executable} -m {__package__}.tests.run_tests",
        "output": output,
        "parent_accessed": False,
        "remote_accessed": False,
    }
    _exclusive_json(freeze / "exact_group_manifest.v1.json", group)
    _exclusive_json(freeze / "no_training_source_gate.v1.json", static)
    _exclusive_json(freeze / "local_test_receipt.v1.json", test_receipt)
    source_manifest = {
        "schema": "hfw_w560_post_training_source_manifest_v1",
        "namespace": package_root.name,
        "files": {str(relative): file_sha256(path) for relative, path in _source_files(package_root)},
    }
    source_manifest["manifest_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(source_manifest)
    ).hexdigest()
    _exclusive_json(freeze / "source_manifest.v1.json", source_manifest)
    release = {
        "schema": "hfw_w560_post_training_local_release_manifest_v1",
        "status": "LOCAL_FROZEN_NOT_STAGED_NOT_LAUNCHED",
        "namespace": package_root.name,
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "parent_review_receipt_sha256": protocol["parent"]["review_receipt_sha256"],
        "canonical_weight_manifest_sha256": protocol["canonical_weight_address_binding"]["manifest_sha256"],
        "freeze_artifacts": {
            name: file_sha256(freeze / name) for name in (
                "exact_group_manifest.v1.json",
                "no_training_source_gate.v1.json",
                "local_test_receipt.v1.json",
                "source_manifest.v1.json",
            )
        },
        "parent_accessed": False,
        "remote_accessed": False,
        "stage_authorized": False,
        "launch_authorized": False,
    }
    _exclusive_json(freeze / "release_manifest.v1.json", release)
    archive_path = freeze / "package_archive.v1.tar"
    descriptor = os.open(archive_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as raw:
        with tarfile.open(fileobj=raw, mode="w") as archive:
            for relative, path in _source_files(package_root):
                info = archive.gettarinfo(str(path), arcname=str(Path(package_root.name) / relative))
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            for name in (
                "exact_group_manifest.v1.json", "no_training_source_gate.v1.json",
                "local_test_receipt.v1.json", "source_manifest.v1.json", "release_manifest.v1.json",
            ):
                path = freeze / name
                info = archive.gettarinfo(str(path), arcname=str(Path(package_root.name) / "freeze" / name))
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    _exclusive_json(freeze / "package_archive_receipt.v1.json", {
        "schema": "hfw_w560_package_archive_receipt_v1",
        "path": archive_path.name,
        "sha256": file_sha256(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "create_only": True,
    })
    validation = validate_local_release(package_root)
    _exclusive_json(freeze / "release_validation_receipt.v1.json", validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
