from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_bytes_same_fd(path: Path, expected_sha256: str | None = None) -> bytes:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        body = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            body.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if tuple(getattr(before, key) for key in fields) != tuple(
        getattr(after, key) for key in fields
    ):
        raise RuntimeError(f"file changed while reading: {path}")
    observed = sha256_bytes(body)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"SHA mismatch for {path}: {observed}")
    return bytes(body)


def read_json_same_fd(path: Path, expected_sha256: str | None = None) -> Mapping[str, object]:
    body = read_bytes_same_fd(path, expected_sha256)
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def file_identity(path: Path, expected_sha256: str | None = None) -> Mapping[str, object]:
    body = read_bytes_same_fd(path, expected_sha256)
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_bytes(body),
        "size_bytes": len(body),
    }


def write_exclusive(path: Path, body: bytes, mode: int = 0o444) -> Mapping[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return {"path": str(path.resolve()), "sha256": sha256_bytes(body), "size_bytes": len(body)}


def write_exclusive_json(path: Path, value: object) -> Mapping[str, object]:
    return write_exclusive(path, canonical_json_bytes(value))


def exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RuntimeError(
            f"{label} key inventory drift: missing={sorted(expected-set(value))} "
            f"extra={sorted(set(value)-expected)}"
        )


def tensor_f32le_bytes(value) -> bytes:
    return value.detach().float().cpu().contiguous().numpy().astype(
        "<f4", copy=False,
    ).tobytes(order="C")
