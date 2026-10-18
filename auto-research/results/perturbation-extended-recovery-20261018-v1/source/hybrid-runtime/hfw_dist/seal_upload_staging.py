from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Mapping

from .common import (
    canonical_json_bytes,
    hash_file,
    read_json,
    sha256_bytes,
    write_json_exclusive,
)


RECEIPT_RELATIVE_PATH = Path("runtime/config/UPLOAD_SNAPSHOT_STAGING.json")
SYSTEM_DATA_ROOT = Path("/System/Volumes/Data")
STAGING_SCHEMA = "hfw_w560_daytona_upload_snapshot_staging_v3"
STAGING_STATUS = "PASS_IMMUTABLE_NO_SPACE_DEVICE_INDEPENDENT_COPY"


def _roots_are_non_nested(left: Path, right: Path) -> bool:
    return left != right and left not in right.parents and right not in left.parents


def _strict_tree_inventory(
    root: Path,
    *,
    reject_whitespace_components: bool,
    require_read_only: bool,
    require_single_link_files: bool,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    """Inventory a real tree without following links or accepting special entries."""
    root = Path(root)
    root_stat = os.lstat(root)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"staging proof root is not a real directory: {root}")
    if require_read_only and root_stat.st_mode & 0o222:
        raise RuntimeError(f"upload staging directory is writable: {root}")

    directories: list[str] = []
    files: dict[str, dict[str, object]] = {}
    file_identities: dict[str, Mapping[str, object]] = {}

    def visit(directory: Path, relative_directory: str) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            if entry.name in {"", ".", ".."}:
                raise RuntimeError("staging proof tree contains an unsafe name")
            if reject_whitespace_components and any(
                character.isspace() for character in entry.name
            ):
                raise RuntimeError(
                    f"upload staging tree contains whitespace: {entry.name!r}"
                )
            relative = (
                entry.name
                if not relative_directory
                else f"{relative_directory}/{entry.name}"
            )
            observed = entry.stat(follow_symlinks=False)
            entry_path = directory / entry.name
            if stat.S_ISLNK(observed.st_mode):
                raise RuntimeError(f"staging proof tree contains a symlink: {relative}")
            if stat.S_ISDIR(observed.st_mode):
                if require_read_only and observed.st_mode & 0o222:
                    raise RuntimeError(
                        f"upload staging directory is writable: {relative}"
                    )
                directories.append(relative)
                visit(entry_path, relative)
                continue
            if not stat.S_ISREG(observed.st_mode):
                raise RuntimeError(
                    f"staging proof tree contains a special entry: {relative}"
                )
            if require_read_only and observed.st_mode & 0o222:
                raise RuntimeError(f"upload staging file is writable: {relative}")
            identity = hash_file(entry_path)
            if require_single_link_files and int(identity["link_count"]) != 1:
                raise RuntimeError(
                    f"upload staging file has a hardlink: {relative}"
                )
            files[relative] = {
                "sha256": identity["sha256"],
                "size_bytes": int(identity["size_bytes"]),
            }
            file_identities[relative] = identity

    visit(root, "")
    return {
        "directories": sorted(directories),
        "files": {name: files[name] for name in sorted(files)},
    }, {name: file_identities[name] for name in sorted(file_identities)}


def build_upload_staging_receipt(
    package_root: Path,
    source_snapshot_root: Path,
    upload_root: Path,
) -> dict[str, object]:
    package = Path(package_root).resolve()
    source = Path(source_snapshot_root).resolve()
    upload_input = Path(upload_root)
    if (
        not upload_input.is_absolute()
        or upload_input.resolve() != upload_input
        or not upload_input.is_dir()
    ):
        raise RuntimeError("upload root must be an absolute real directory")
    upload = upload_input
    if not package.is_dir() or not source.is_dir():
        raise RuntimeError("package and wave snapshot roots must be real directories")
    if package not in source.parents:
        raise RuntimeError("wave snapshot root is outside the package root")
    if not _roots_are_non_nested(package, upload):
        raise RuntimeError("package and upload roots must be distinct and non-nested")
    if not _roots_are_non_nested(source, upload):
        raise RuntimeError("source snapshot and upload roots must be distinct and non-nested")
    if any(any(character.isspace() for character in part) for part in upload.parts):
        raise RuntimeError("upload root path contains whitespace")

    system_data_device = int(os.stat(SYSTEM_DATA_ROOT).st_dev)
    package_device = int(os.stat(package).st_dev)
    source_device = int(os.stat(source).st_dev)
    upload_device = int(os.stat(upload).st_dev)
    if source_device != package_device:
        raise RuntimeError("source snapshot root device differs from package root")
    if upload_device != system_data_device:
        raise RuntimeError("upload root is not on the system data volume")

    source_tree, source_identities = _strict_tree_inventory(
        source,
        reject_whitespace_components=False,
        require_read_only=False,
        require_single_link_files=False,
    )
    upload_tree, upload_identities = _strict_tree_inventory(
        upload,
        reject_whitespace_components=True,
        require_read_only=True,
        require_single_link_files=True,
    )
    if source_tree != upload_tree:
        raise RuntimeError("upload staging complete inventory differs from source snapshot")

    corresponding_file_identities: dict[str, object] = {}
    source_identity_set: set[tuple[int, int]] = set()
    upload_identity_set: set[tuple[int, int]] = set()
    files = source_tree["files"]
    for relative in files:
        source_identity = source_identities[relative]
        upload_identity = upload_identities[relative]
        source_device_inode = (
            int(source_identity["device"]), int(source_identity["inode"]),
        )
        upload_device_inode = (
            int(upload_identity["device"]), int(upload_identity["inode"]),
        )
        if source_device_inode == upload_device_inode:
            raise RuntimeError(
                f"upload staging file is a hardlink to its source: {relative}"
            )
        source_identity_set.add(source_device_inode)
        upload_identity_set.add(upload_device_inode)
        corresponding_file_identities[relative] = {
            "source": {
                "device": source_device_inode[0],
                "inode": source_device_inode[1],
                "link_count": int(source_identity["link_count"]),
            },
            "upload": {
                "device": upload_device_inode[0],
                "inode": upload_device_inode[1],
                "link_count": int(upload_identity["link_count"]),
            },
        }
    if not source_identity_set.isdisjoint(upload_identity_set):
        raise RuntimeError("source and upload trees share a file identity")

    campaign, _ = read_json(source / "CAMPAIGN.json")
    campaign_identity = hash_file(source / "CAMPAIGN.json")
    snapshot_identity = hash_file(source / "SNAPSHOT_MANIFEST.json")
    directories = source_tree["directories"]
    return {
        "schema": STAGING_SCHEMA,
        "status": STAGING_STATUS,
        "wave_index": int(campaign["wave_index"]),
        "package_root": str(package),
        "source_wave_snapshot_path": str(source),
        "upload_snapshot_root": str(upload),
        "system_data_device": system_data_device,
        "package_root_device": package_device,
        "source_snapshot_root_device": source_device,
        "upload_root_device": upload_device,
        "campaign": {
            "sha256": campaign_identity["sha256"],
            "size_bytes": campaign_identity["size_bytes"],
        },
        "snapshot_manifest": {
            "sha256": snapshot_identity["sha256"],
            "size_bytes": snapshot_identity["size_bytes"],
        },
        "complete_inventory": files,
        "complete_directories": directories,
        "complete_inventory_sha256": sha256_bytes(canonical_json_bytes(files)),
        "complete_tree_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source_tree)
        ),
        "file_count": len(files),
        "directory_count": len(directories),
        "total_size_bytes": sum(
            int(record["size_bytes"]) for record in files.values()
        ),
        "corresponding_file_identities": corresponding_file_identities,
        "corresponding_file_identities_sha256": sha256_bytes(
            canonical_json_bytes(corresponding_file_identities)
        ),
        "source_and_upload_inventory_exact_match": True,
        "source_and_upload_roots_distinct": True,
        "source_and_upload_roots_non_nested": True,
        "source_and_upload_corresponding_files_have_disjoint_device_inode_identities": True,
        "source_and_upload_corresponding_files_are_not_hardlinks": True,
        "upload_root_absolute_real_directory": True,
        "upload_root_and_all_components_contain_no_whitespace": True,
        "upload_tree_contains_no_symlinks": True,
        "upload_tree_contains_no_special_entries": True,
        "upload_tree_directories_read_only": True,
        "upload_tree_files_read_only": True,
        "upload_tree_files_have_single_link_count": True,
        "package_root_device_equals_source_snapshot_device": True,
        "package_root_device_equals_system_data_device": (
            package_device == system_data_device
        ),
        "source_snapshot_root_device_equals_system_data_device": (
            source_device == system_data_device
        ),
        "upload_root_device_equals_system_data_device": True,
        "source_and_upload_devices_distinct": source_device != upload_device,
    }


def seal_upload_staging(
    package_root: Path,
    wave_root: Path,
    upload_root: Path,
) -> dict[str, object]:
    package = Path(package_root).resolve()
    value = build_upload_staging_receipt(
        package, Path(wave_root).resolve() / "snapshot", upload_root,
    )
    receipt_path = package / RECEIPT_RELATIVE_PATH
    if receipt_path.exists():
        existing, identity = read_json(receipt_path)
        if canonical_json_bytes(existing) != canonical_json_bytes(value):
            raise RuntimeError("upload staging receipt already exists with drift")
    else:
        identity = write_json_exclusive(receipt_path, value)
    return {**value, "identity": identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--upload-root", type=Path, required=True)
    args = parser.parse_args()
    result = seal_upload_staging(
        args.package_root, args.wave_root, args.upload_root,
    )
    print(json.dumps({
        "status": result["status"],
        "sha256": result["identity"]["sha256"],
        "file_count": result["file_count"],
        "directory_count": result["directory_count"],
        "total_size_bytes": result["total_size_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
