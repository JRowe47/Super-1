from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Iterator, Mapping


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_file_same_fd(path: Path) -> Mapping[str, object]:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_nlink, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_nlink, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RuntimeError(f"file changed while hashing: {path}")
        return {
            "path": str(path.resolve()),
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
            "device": before.st_dev,
            "inode": before.st_ino,
            "link_count": before.st_nlink,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_bytes(path: Path, payload: bytes, mode: int = 0o444) -> Mapping[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    fsync_directory(path.parent)
    return {"path": str(path.resolve()), "sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def write_exclusive_json(path: Path, value: object, mode: int = 0o444) -> Mapping[str, object]:
    return write_exclusive_bytes(path, canonical_json_bytes(value), mode=mode)


def write_content_addressed_bytes(
    directory: Path, suffix: str, payload: bytes, mode: int = 0o444,
) -> Mapping[str, object]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    path = directory / f"{digest}{suffix}"
    if path.exists():
        identity = hash_file_same_fd(path)
        if identity["sha256"] != digest or int(identity["size_bytes"]) != len(payload):
            raise RuntimeError(f"content-address collision: {path}")
        return identity
    temporary = directory / f".{digest}.{os.getpid()}.{time.time_ns()}.tmp"
    write_exclusive_bytes(temporary, payload, mode=mode)
    try:
        try:
            os.link(temporary, path)
            fsync_directory(directory)
        except FileExistsError:
            identity = hash_file_same_fd(path)
            if identity["sha256"] != digest or int(identity["size_bytes"]) != len(payload):
                raise RuntimeError(f"concurrent content-address collision: {path}")
    finally:
        temporary.unlink(missing_ok=True)
        fsync_directory(directory)
    return hash_file_same_fd(path)


def write_content_addressed_json(
    directory: Path, suffix: str, value: object, mode: int = 0o444,
) -> Mapping[str, object]:
    return write_content_addressed_bytes(directory, suffix, canonical_json_bytes(value), mode=mode)


def read_json_same_fd(path: Path, expected_sha256: str | None = None) -> tuple[Mapping[str, object], Mapping[str, object]]:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular JSON file: {path}")
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_nlink, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_nlink, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RuntimeError(f"JSON changed while reading: {path}")
    finally:
        os.close(descriptor)
    digest = sha256_bytes(bytes(payload))
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"JSON SHA mismatch: {path}")
    value = json.loads(payload)
    return value, {
        "path": str(path.resolve()), "sha256": digest, "size_bytes": len(payload),
        "device": before.st_dev, "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "link_count": before.st_nlink,
        "ctime_ns": before.st_ctime_ns,
    }


def read_bytes_same_fd(path: Path, expected_sha256: str | None = None) -> tuple[bytes, Mapping[str, object]]:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_nlink, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_nlink, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RuntimeError(f"file changed while reading: {path}")
    finally:
        os.close(descriptor)
    digest = sha256_bytes(bytes(payload))
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return bytes(payload), {
        "path": str(path.resolve()), "sha256": digest, "size_bytes": len(payload),
        "device": before.st_dev, "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "link_count": before.st_nlink,
        "ctime_ns": before.st_ctime_ns,
    }


@contextmanager
def exclusive_lock(path: Path, *, nonblocking: bool = False) -> Iterator[None]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"lock file missing or unsafe: {path}")
    with path.open("r+") as handle:
        operation = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as error:
            raise RuntimeError(f"exclusive singleton lock is already held: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
