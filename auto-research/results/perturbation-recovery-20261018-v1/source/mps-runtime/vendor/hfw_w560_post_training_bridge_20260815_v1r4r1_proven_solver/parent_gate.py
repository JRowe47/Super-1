from __future__ import annotations

import json
import os
import hashlib
import re
import stat
from pathlib import Path
from typing import Mapping

from .manifest import file_sha256


def _require_sha256_hex(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{label} is not an exact lowercase SHA-256 digest")
    return value


def _read_hash_same_fd(path: Path, keep_bytes: bool = False):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("bound artifact is not a regular file")
        digest = hashlib.sha256()
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if keep_bytes:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_nlink, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_nlink, after.st_mtime_ns)
        if identity_before != identity_after:
            raise RuntimeError("bound artifact changed during same-FD validation")
        return digest.hexdigest(), before, b"".join(chunks) if keep_bytes else None
    finally:
        os.close(descriptor)


def validate_immutable_parent(protocol: Mapping[str, object]) -> Mapping[str, object]:
    contract = protocol["parent"]
    for field in (
        "checkpoint_sha256", "review_receipt_sha256",
        "checkpoint_identity_sha256", "model_state_digest",
    ):
        _require_sha256_hex(contract.get(field), f"parent {field}")
    checkpoint = Path(contract["checkpoint_path"])
    review = Path(contract["review_receipt_path"])
    checkpoint_digest, checkpoint_info, _ = _read_hash_same_fd(checkpoint)
    if checkpoint_info.st_size != int(contract["size_bytes"]):
        raise RuntimeError("checkpoint size drift")
    if checkpoint_info.st_ino != int(contract["inode"]):
        raise RuntimeError("checkpoint inode drift")
    if checkpoint_info.st_nlink < int(contract["minimum_link_count"]):
        raise RuntimeError("checkpoint link-count drift")
    if checkpoint_digest != contract["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA drift")
    review_digest, _, review_bytes = _read_hash_same_fd(review, keep_bytes=True)
    if review_digest != contract["review_receipt_sha256"]:
        raise RuntimeError("review receipt SHA drift")
    receipt = json.loads(review_bytes.decode("utf-8"))
    if receipt.get("schema") != contract["review_schema"]:
        raise RuntimeError("review schema drift")
    if int(receipt.get("step", -1)) != int(contract["completed_step"]):
        raise RuntimeError("review step drift")
    if int(receipt.get("tokens", -1)) != int(contract["completed_tokens"]):
        raise RuntimeError("review token count drift")
    if receipt.get("training_resume_permanently_forbidden") is not True:
        raise RuntimeError("review does not permanently forbid training resume")
    if receipt.get("checkpoint_identity_sha256") != contract["checkpoint_identity_sha256"]:
        raise RuntimeError("checkpoint identity drift")
    publication = receipt.get("checkpoint_publication", {})
    control = publication.get("control", {})
    _require_sha256_hex(
        control.get("model_state_digest"), "review model_state_digest"
    )
    if publication.get("sha256") != contract["checkpoint_sha256"]:
        raise RuntimeError("review checkpoint publication SHA drift")
    if control.get("model_state_digest") != contract["model_state_digest"]:
        raise RuntimeError("review model-state digest drift")
    return {
        "status": "PASS_IMMUTABLE_COMPLETED_PARENT",
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "review_receipt_sha256": contract["review_receipt_sha256"],
        "step": int(contract["completed_step"]),
        "tokens": int(contract["completed_tokens"]),
        "inode": checkpoint_info.st_ino,
        "size_bytes": checkpoint_info.st_size,
        "link_count": checkpoint_info.st_nlink,
        "training_resume_permanently_forbidden": True,
    }
