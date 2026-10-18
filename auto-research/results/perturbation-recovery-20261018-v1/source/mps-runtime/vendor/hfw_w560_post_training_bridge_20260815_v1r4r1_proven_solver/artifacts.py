from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import socket
import stat
from pathlib import Path
from typing import Mapping, Sequence

from .manifest import canonical_json_bytes


def write_exclusive_bytes(path: Path, payload: bytes, mode: int = 0o444) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)
    return hashlib.sha256(payload).hexdigest()


def write_content_addressed_json(
    directory: Path, suffix: str, payload: Mapping[str, object], mode: int = 0o444,
) -> Mapping[str, object]:
    encoded = canonical_json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    path = directory / f"{digest}{suffix}"
    write_exclusive_bytes(path, encoded, mode=mode)
    return {"path": str(path), "sha256": digest, "size_bytes": len(encoded)}


def read_json_same_fd(path: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("receipt is not a regular same-FD file")
        digest = hashlib.sha256()
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError("receipt changed during same-FD read")
        payload = b"".join(chunks)
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise RuntimeError("receipt must be a JSON object")
        return value, {
            "sha256": digest.hexdigest(), "inode": before.st_ino,
            "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
        }
    finally:
        os.close(descriptor)


def read_bytes_same_fd(path: Path) -> tuple[bytes, Mapping[str, object]]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("artifact is not a regular same-FD file")
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError("artifact changed during same-FD read")
        return b"".join(chunks), {
            "sha256": digest.hexdigest(), "inode": before.st_ino,
            "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
        }
    finally:
        os.close(descriptor)


def hash_file_same_fd(path: Path) -> Mapping[str, object]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("artifact is not a regular same-FD file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError("artifact changed during same-FD hash")
        return {
            "sha256": digest.hexdigest(), "inode": before.st_ino,
            "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
        }
    finally:
        os.close(descriptor)


def require_strong_path_separation(paths: Mapping[str, Path]) -> Mapping[str, str]:
    resolved = {name: Path(path).resolve(strict=False) for name, path in paths.items()}
    items = list(resolved.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise RuntimeError(f"protected roots overlap: {left_name}/{right_name}")
    return {name: str(path) for name, path in resolved.items()}


def validate_fresh_no_writer_receipt(
    protocol: Mapping[str, object], path: Path,
) -> Mapping[str, object]:
    receipt, identity = read_json_same_fd(path)
    if receipt.get("status") != "PASS_NO_WRITERS":
        raise RuntimeError("no-writer receipt did not pass")
    if receipt.get("bound_parent_checkpoint_sha256") != protocol["parent"]["checkpoint_sha256"]:
        raise RuntimeError("no-writer receipt parent binding drift")
    if receipt.get("hostname") != socket.gethostname():
        raise RuntimeError("no-writer receipt host binding drift")
    if receipt.get("observed_writer_pids") != []:
        raise RuntimeError("no-writer receipt contains active writer PIDs")
    created = dt.datetime.fromisoformat(str(receipt["created_at_utc"]).replace("Z", "+00:00"))
    age = (dt.datetime.now(dt.timezone.utc) - created.astimezone(dt.timezone.utc)).total_seconds()
    if age < 0 or age > int(protocol["operations"]["no_writer_receipt_max_age_seconds"]):
        raise RuntimeError("no-writer receipt is stale")
    return {
        **receipt,
        "receipt_file_sha256": identity["sha256"],
        "receipt_inode": identity["inode"],
        "validated_age_seconds": age,
    }
