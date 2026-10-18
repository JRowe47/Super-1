from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_exclusive(path: Path, value: object) -> None:
    body = canonical(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--released-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.ready.exists() or args.release.exists() or args.released_receipt.exists():
        raise FileExistsError("boundary guard paths must be create-only and fresh")
    if args.lock.is_symlink() or not args.lock.is_file():
        raise RuntimeError("allocator lock is missing or unsafe")
    with args.lock.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        stat = os.fstat(handle.fileno())
        ready = {
            "schema": "hfw_w560_attempt5_allocator_boundary_guard_v1",
            "status": "PASS_ALLOCATOR_LOCK_HELD_WAITING_FOR_ACTIVE_JOB_TERMINAL",
            "guard_pid": os.getpid(),
            "allocator_lock_path": str(args.lock.resolve()),
            "allocator_lock_device": stat.st_dev,
            "allocator_lock_inode": stat.st_ino,
            "acquired_time_ns": time.time_ns(),
            "release_requires_explicit_create_only_sentinel": str(args.release.resolve()),
        }
        ready["receipt_identity_sha256"] = hashlib.sha256(canonical(ready)).hexdigest()
        write_exclusive(args.ready, ready)
        while not args.release.is_file():
            time.sleep(0.05)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    released = {
        "schema": "hfw_w560_attempt5_allocator_boundary_guard_release_v1",
        "status": "PASS_ALLOCATOR_LOCK_RELEASED_AFTER_EXPLICIT_SENTINEL",
        "guard_pid": os.getpid(),
        "released_time_ns": time.time_ns(),
        "release_sentinel_path": str(args.release.resolve()),
    }
    released["receipt_identity_sha256"] = hashlib.sha256(canonical(released)).hexdigest()
    write_exclusive(args.released_receipt, released)


if __name__ == "__main__":
    main()
