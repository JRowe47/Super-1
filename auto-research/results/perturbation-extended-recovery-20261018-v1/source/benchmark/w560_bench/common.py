from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Iterator, Mapping


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_bytes_same_fd(path: Path, expected_sha256: str | None = None) -> tuple[bytes, Mapping[str, object]]:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular same-FD file: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f"file changed during same-FD read: {path}")
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise RuntimeError(f"SHA drift for {path}: expected {expected_sha256}, observed {observed}")
        return b"".join(chunks), {
            "path": str(path.resolve()), "sha256": observed, "size_bytes": before.st_size,
            "device": before.st_dev, "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        }
    finally:
        os.close(descriptor)


def read_json_same_fd(path: Path, expected_sha256: str | None = None) -> tuple[Mapping[str, object], Mapping[str, object]]:
    data, identity = read_bytes_same_fd(path, expected_sha256)
    value = json.loads(data)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value, identity


def hash_file_same_fd(path: Path, expected_sha256: str | None = None) -> Mapping[str, object]:
    return read_bytes_same_fd(path, expected_sha256)[1]


def write_exclusive_bytes(path: Path, data: bytes, mode: int = 0o444) -> Mapping[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    return hash_file_same_fd(path)


def write_exclusive_json(path: Path, value: object) -> Mapping[str, object]:
    return write_exclusive_bytes(path, canonical_json_bytes(value))


@contextlib.contextmanager
def exclusive_lock(path: Path, *, nonblocking: bool) -> Iterator[int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    try:
        try:
            fcntl.flock(descriptor, flags)
        except BlockingIOError as error:
            raise RuntimeError(f"lock busy; refusing MPS/solver overlap: {path}") from error
        yield descriptor
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def require_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RuntimeError(f"{label} key inventory drift: missing={sorted(expected-set(value))}, extra={sorted(set(value)-expected)}")
