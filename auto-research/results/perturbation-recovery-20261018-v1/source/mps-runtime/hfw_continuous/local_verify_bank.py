from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np

from .common import canonical_json_bytes, hash_file_same_fd, read_bytes_same_fd, write_exclusive_bytes, write_exclusive_json


SELECTION_SCHEMA = "hfw_w560_local_verify_public_selection_v1"
BANK_ROLE = "LOCAL_VERIFY"
DISCLOSURE = "PERMANENT_REUSED_OPTIMIZATION_DATA_NOT_HELD_OUT"


def _read_source_same_fd(path: Path, expected_sha256: str, expected_size: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise RuntimeError("LOCAL_VERIFY source identity drift")
        payload = bytearray()
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError("LOCAL_VERIFY source changed during same-FD read")
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("LOCAL_VERIFY source SHA drift")
        return bytes(payload)
    finally:
        os.close(descriptor)


def load_selection(path: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    payload, identity = read_bytes_same_fd(path)
    selection = json.loads(payload)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise RuntimeError("LOCAL_VERIFY selection schema drift")
    if selection.get("role") != BANK_ROLE or selection.get("disclosure") != DISCLOSURE:
        raise RuntimeError("LOCAL_VERIFY role/disclosure drift")
    starts = selection.get("selection", {}).get("starts", [])
    if len(starts) != 32 or len(set(int(value) for value in starts)) != 32:
        raise RuntimeError("LOCAL_VERIFY requires exactly 32 distinct public starts")
    materialization = selection.get("materialization", {})
    required = {
        "before_any_local_optimization": True,
        "zero_new_unique_token_positions_after_materialization": True,
        "reused_for_candidate_selection_across_generations": True,
        "held_out_generalization_claimed": False,
    }
    if materialization != required:
        raise RuntimeError("LOCAL_VERIFY optimization-data disclosure drift")
    return selection, identity


def materialize_local_verify(selection_path: Path, runtime_root: Path) -> Mapping[str, object]:
    selection, selection_identity = load_selection(selection_path)
    source = selection["source"]
    raw = _read_source_same_fd(
        Path(source["path"]), str(source["sha256"]), int(source["size_bytes"]),
    )
    tokens = np.frombuffer(raw, dtype="<u4")
    if tokens.size != int(source["token_count"]):
        raise RuntimeError("LOCAL_VERIFY source token-count drift")
    length = int(selection["selection"]["sequence_tokens"])
    starts = [int(value) for value in selection["selection"]["starts"]]
    sequences = np.stack([tokens[start:start + length] for start in starts], axis=0)
    if sequences.shape != tuple(selection["materialized_bank"]["logical_shape"]):
        raise RuntimeError("LOCAL_VERIFY materialized logical shape drift")
    payload = sequences.astype("<u4", copy=False).tobytes(order="C")
    expected = selection["materialized_bank"]
    if len(payload) != int(expected["size_bytes"]):
        raise RuntimeError("LOCAL_VERIFY materialized size drift")
    if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise RuntimeError("LOCAL_VERIFY materialized SHA drift")
    bank_path = Path(runtime_root) / str(expected["relative_path"])
    bank_identity = write_exclusive_bytes(bank_path, payload)
    receipt = {
        "schema": "hfw_w560_local_verify_materialization_receipt_v1",
        "status": "PASS_EXACT_32_SEQUENCE_LOCAL_VERIFY_MATERIALIZED_BEFORE_OPTIMIZATION",
        "role": BANK_ROLE,
        "disclosure": DISCLOSURE,
        "selection_spec": selection_identity,
        "source_identity": {
            "sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
            "token_count": source["token_count"],
        },
        "public_starts": starts,
        "sequence_tokens": length,
        "bank": bank_identity,
        "logical_shape": expected["logical_shape"],
        "runtime_batch_shape": expected["runtime_batch_shape"],
        "materialized_before_any_local_optimization": True,
        "zero_new_unique_token_positions_after_materialization": True,
        "reused_for_candidate_selection_across_generations": True,
        "held_out_generalization_claimed": False,
        "source_corpus_bytes_copied_into_runtime": False,
    }
    receipt_path = bank_path.parent / "LOCAL_VERIFY_MATERIALIZED.json"
    receipt_identity = write_exclusive_json(receipt_path, receipt)
    return {**receipt, "receipt_artifact": receipt_identity}


def load_local_verify(
    config: Mapping[str, object], runtime_root: Path,
) -> tuple[np.ndarray, Mapping[str, object]]:
    contract = config["local_verify"]
    if contract.get("role") != BANK_ROLE or contract.get("disclosure") != DISCLOSURE:
        raise RuntimeError("LOCAL_VERIFY config disclosure drift")
    bank = contract["bank"]
    bank_path = Path(runtime_root) / str(bank["relative_path"])
    payload, identity = read_bytes_same_fd(bank_path, str(bank["sha256"]))
    if int(identity["size_bytes"]) != int(bank["size_bytes"]):
        raise RuntimeError("LOCAL_VERIFY runtime bank size drift")
    logical_shape = tuple(int(value) for value in bank["logical_shape"])
    values = np.frombuffer(payload, dtype="<u4").reshape(logical_shape).copy()
    runtime_shape = tuple(int(value) for value in bank["runtime_batch_shape"])
    if int(np.prod(runtime_shape)) != int(np.prod(logical_shape)):
        raise RuntimeError("LOCAL_VERIFY runtime batching changes token inventory")
    receipt, receipt_identity = read_bytes_same_fd(
        Path(runtime_root) / str(contract["materialization_receipt"]["relative_path"]),
        str(contract["materialization_receipt"]["sha256"]),
    )
    if len(receipt) != int(contract["materialization_receipt"]["size_bytes"]):
        raise RuntimeError("LOCAL_VERIFY materialization receipt size drift")
    return values.reshape(runtime_shape), {
        "schema": "hfw_w560_local_verify_loaded_capability_v1",
        "role": BANK_ROLE,
        "disclosure": DISCLOSURE,
        "bank": identity,
        "materialization_receipt_sha256": receipt_identity["sha256"],
        "logical_shape": list(logical_shape),
        "runtime_batch_shape": list(runtime_shape),
        "new_unique_token_positions_opened": 0,
        "held_out_generalization_claimed": False,
    }
