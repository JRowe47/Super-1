from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Iterator, Mapping


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_file(path: Path, expected_sha256: str | None = None) -> dict[str, object]:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = (
            before.st_dev, before.st_ino, before.st_size, before.st_nlink,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        if identity != (
            after.st_dev, after.st_ino, after.st_size, after.st_nlink,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise RuntimeError(f"file changed during hash: {path}")
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        return {
            "path": str(path.resolve()), "sha256": observed,
            "size_bytes": before.st_size, "device": before.st_dev,
            "inode": before.st_ino, "link_count": before.st_nlink,
            "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
        }
    finally:
        os.close(descriptor)


def hash_file_region(
    path: Path, offset_bytes: int, size_bytes: int,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    path = Path(path)
    if offset_bytes < 0 or size_bytes < 0:
        raise RuntimeError("file region bounds must be nonnegative")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        if offset_bytes + size_bytes > before.st_size:
            raise RuntimeError(f"file region exceeds file size: {path}")
        os.lseek(descriptor, offset_bytes, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = size_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"short read while hashing file region: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = (
            before.st_dev, before.st_ino, before.st_size, before.st_nlink,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        if identity != (
            after.st_dev, after.st_ino, after.st_size, after.st_nlink,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise RuntimeError(f"file changed during region hash: {path}")
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise RuntimeError(f"SHA-256 region mismatch: {path}")
        return {
            "path": str(path.resolve()), "sha256": observed,
            "offset_bytes": offset_bytes, "size_bytes": size_bytes,
            "file_size_bytes": before.st_size, "device": before.st_dev,
            "inode": before.st_ino, "link_count": before.st_nlink,
            "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
        }
    finally:
        os.close(descriptor)


def read_file_region(
    path: Path, offset_bytes: int, size_bytes: int,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, object]]:
    identity = hash_file_region(path, offset_bytes, size_bytes, expected_sha256)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.lseek(descriptor, offset_bytes, os.SEEK_SET)
        payload = bytearray()
        remaining = size_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"short read while reading file region: {path}")
            payload.extend(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    if sha256_bytes(payload) != identity["sha256"]:
        raise RuntimeError(f"file changed between region hash and read: {path}")
    return bytes(payload), identity


def read_bytes(path: Path, expected_sha256: str | None = None) -> tuple[bytes, dict[str, object]]:
    identity = hash_file(path, expected_sha256)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            payload.extend(chunk)
    finally:
        os.close(descriptor)
    if sha256_bytes(payload) != identity["sha256"]:
        raise RuntimeError(f"file changed between hash and read: {path}")
    return bytes(payload), identity


def read_json(path: Path, expected_sha256: str | None = None) -> tuple[Mapping[str, object], dict[str, object]]:
    payload, identity = read_bytes(path, expected_sha256)
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value, identity


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)
    return hash_file(path)


def write_json_exclusive(path: Path, value: object) -> dict[str, object]:
    return write_exclusive(path, canonical_json_bytes(value))


def write_content_addressed(directory: Path, suffix: str, payload: bytes) -> dict[str, object]:
    digest = sha256_bytes(payload)
    path = Path(directory) / f"{digest}{suffix}"
    if path.exists():
        identity = hash_file(path, digest)
        if int(identity["size_bytes"]) != len(payload):
            raise RuntimeError("content-addressed file size collision")
        return identity
    return write_exclusive(path, payload)


def write_content_addressed_json(directory: Path, suffix: str, value: object) -> dict[str, object]:
    return write_content_addressed(directory, suffix, canonical_json_bytes(value))


def atomic_current_pointer(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    path = Path(path)
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging, 0o444)
        os.replace(staging, path)
        fsync_directory(path.parent)
    finally:
        staging.unlink(missing_ok=True)
    return hash_file(path)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def relative_inventory(root: Path, excluded: set[str] | None = None) -> dict[str, dict[str, object]]:
    root = Path(root)
    excluded = excluded or set()
    result: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        identity = hash_file(path)
        result[relative] = {
            "sha256": identity["sha256"], "size_bytes": identity["size_bytes"],
        }
    return result


def validate_inventory(root: Path, inventory: Mapping[str, object]) -> None:
    root = Path(root)
    expected = set(inventory)
    for relative, record in inventory.items():
        path = root / relative
        identity = hash_file(path, str(record["sha256"]))
        if int(identity["size_bytes"]) != int(record["size_bytes"]):
            raise RuntimeError(f"snapshot file size drift: {relative}")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    allowed_runtime = {
        "CAMPAIGN.json", "TASK.json", "SNAPSHOT_MANIFEST.json",
    }
    unexpected = observed - expected - allowed_runtime
    if unexpected:
        raise RuntimeError(f"unexpected snapshot files: {sorted(unexpected)}")
