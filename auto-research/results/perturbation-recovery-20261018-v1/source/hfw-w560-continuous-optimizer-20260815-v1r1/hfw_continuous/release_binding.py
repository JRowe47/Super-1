from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Mapping

from .common import canonical_json_bytes, hash_file_same_fd, read_json_same_fd


def _discover_runtime_files(package_root: Path) -> set[str]:
    discovered: set[str] = set()
    for relative_root in (Path("hfw_continuous"), Path("vendor")):
        absolute_root = package_root / relative_root
        if absolute_root.is_symlink() or not absolute_root.is_dir():
            raise RuntimeError(f"runtime source root missing or symlinked: {relative_root}")
        for directory, directory_names, file_names in os.walk(absolute_root, followlinks=False):
            directory_path = Path(directory)
            for name in list(directory_names):
                child = directory_path / name
                if name == "__pycache__":
                    directory_names.remove(name)
                elif child.is_symlink():
                    raise RuntimeError(f"runtime source directory is symlinked: {child}")
            for name in file_names:
                path = directory_path / name
                if name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
                    continue
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"runtime source file is not regular: {path}")
                if relative_root.name == "hfw_continuous" and path.suffix != ".py":
                    continue
                discovered.add(path.relative_to(package_root).as_posix())
    return discovered


def validate_runtime_manifest(
    config_path: Path, config: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    config_path = Path(config_path)
    if config is None:
        config = read_json_same_fd(config_path)[0]
    release = config.get("runtime_release")
    if not isinstance(release, Mapping):
        raise RuntimeError("controller config has no pinned runtime release manifest")
    package_root = config_path.resolve().parents[1]
    relative = PurePosixPath(str(release["manifest_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("runtime manifest path is not a safe package-relative path")
    manifest_path = package_root.joinpath(*relative.parts)
    manifest, manifest_file_identity = read_json_same_fd(
        manifest_path, str(release["manifest_file_sha256"])
    )
    if int(manifest_file_identity["size_bytes"]) != int(release["manifest_size_bytes"]):
        raise RuntimeError("runtime manifest file size drift")
    if manifest.get("schema") != "hfw_w560_continuous_runtime_manifest_v1":
        raise RuntimeError("runtime manifest schema drift")
    if manifest.get("namespace") != package_root.name:
        raise RuntimeError("runtime manifest namespace drift")
    claimed_content_identity = manifest.get("manifest_identity_sha256")
    body = dict(manifest)
    body.pop("manifest_identity_sha256", None)
    observed_content_identity = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if claimed_content_identity != observed_content_identity:
        raise RuntimeError("runtime manifest claimed content identity drift")
    if claimed_content_identity != release["manifest_identity_sha256"]:
        raise RuntimeError("controller config/runtime manifest content identity drift")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("runtime manifest file map is empty")
    discovered = _discover_runtime_files(package_root)
    if set(files) != discovered:
        missing = sorted(discovered - set(files))
        extra = sorted(set(files) - discovered)
        raise RuntimeError(f"runtime manifest coverage drift: missing={missing}, extra={extra}")
    total_size = 0
    for relative_name in sorted(files):
        pure = PurePosixPath(relative_name)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError("runtime manifest contains an unsafe relative path")
        expected = files[relative_name]
        observed = hash_file_same_fd(package_root.joinpath(*pure.parts))
        if observed["sha256"] != expected["sha256"]:
            raise RuntimeError(f"runtime source SHA drift: {relative_name}")
        if int(observed["size_bytes"]) != int(expected["size_bytes"]):
            raise RuntimeError(f"runtime source size drift: {relative_name}")
        total_size += int(observed["size_bytes"])
    if int(manifest.get("file_count", -1)) != len(files):
        raise RuntimeError("runtime manifest file count drift")
    if int(manifest.get("total_size_bytes", -1)) != total_size:
        raise RuntimeError("runtime manifest total size drift")
    return {
        "schema": "hfw_w560_continuous_runtime_binding_v1",
        "status": "PASS_CONFIG_PINNED_RUNTIME_MANIFEST_AND_ALL_LISTED_BYTES",
        "package_root": str(package_root),
        "manifest_file": manifest_file_identity,
        "manifest_identity_sha256": claimed_content_identity,
        "validated_file_count": len(files),
        "validated_total_size_bytes": total_size,
    }
