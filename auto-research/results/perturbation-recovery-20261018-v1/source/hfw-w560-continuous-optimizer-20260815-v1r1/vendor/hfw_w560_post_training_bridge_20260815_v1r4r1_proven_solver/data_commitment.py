from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

def resolve_validation_corpus(
    protocol: Mapping[str, object], runtime_root: Path,
) -> tuple[Path, os.stat_result]:
    """Resolve validation bytes only below the independently bound runtime root."""
    contract = protocol["data"]
    relative = Path(str(contract["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("validation corpus relative path is unsafe")
    path = runtime_root / relative
    expected_path = protocol.get("runtime", {}).get("validation_path")
    if expected_path is not None and str(path) != str(expected_path):
        raise RuntimeError("validation corpus canonical path binding drift")
    if path.is_symlink():
        raise RuntimeError("validation corpus must not be a symlink")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("validation corpus is not a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError("validation corpus changed while hashing")
        if digest.hexdigest() != contract["sha256"]:
            raise RuntimeError("validation corpus identity drift")
        if "size_bytes" in contract and before.st_size != int(contract["size_bytes"]):
            raise RuntimeError("validation corpus size drift")
        if "inode" in contract and before.st_ino != int(contract["inode"]):
            raise RuntimeError("validation corpus inode drift")
        if "minimum_link_count" in contract and before.st_nlink < int(contract["minimum_link_count"]):
            raise RuntimeError("validation corpus link-count drift")
        return path, before
    finally:
        os.close(descriptor)


def _batch_starts(token_count: int, context: int, batch_size: int, step: int, seed: int) -> tuple[int, ...]:
    generator = np.random.default_rng(int(seed) + 1_000_003 * int(step))
    return tuple(int(value) for value in generator.integers(
        0, token_count - context - 1, size=batch_size
    ))


def _overlaps(candidate: tuple[int, int], occupied: Sequence[tuple[int, int]]) -> bool:
    return any(candidate[0] < stop and start < candidate[1] for start, stop in occupied)


def _resolve_fresh_batches(
    tokens: np.memmap,
    context: int,
    batch_size: int,
    count: int,
    seed: int,
    occupied: list[tuple[int, int]],
) -> list[Mapping[str, object]]:
    records = []
    step = 0
    while len(records) < count:
        starts = _batch_starts(len(tokens), context, batch_size, step, seed)
        intervals = [(start, start + context + 1) for start in starts]
        if not any(_overlaps(interval, occupied + intervals[:index]) for index, interval in enumerate(intervals)):
            batch = np.stack([tokens[start:start + context + 1] for start in starts]).astype(np.int64)
            records.append({
                "selection_step": step,
                "starts": list(starts),
                "intervals": [list(value) for value in intervals],
                "raw_batch_sha256": hashlib.sha256(batch.astype("<i8", copy=False).tobytes()).hexdigest(),
                "example_sha256s": [
                    hashlib.sha256(row.astype("<i8", copy=False).tobytes()).hexdigest()
                    for row in batch
                ],
            })
            occupied.extend(intervals)
        step += 1
        if step > count * 10000:
            raise RuntimeError("could not resolve disjoint data commitment")
    return records


def resolve_public_calibration_layout(
    protocol: Mapping[str, object], runtime_root: Path
) -> Mapping[str, object]:
    contract = protocol["data"]
    path, identity = resolve_validation_corpus(protocol, runtime_root)
    tokens = np.memmap(path, dtype=np.dtype(contract["dtype"]), mode="r")
    context = int(contract["context"])
    batch_size = int(contract["batch_size"])
    occupied = []
    prior = []
    for step in range(16):
        starts = _batch_starts(len(tokens), context, batch_size, step, int(contract["prior_public_seed"]))
        intervals = [(start, start + context + 1) for start in starts]
        occupied.extend(intervals)
        prior.append({"step": step, "starts": list(starts), "intervals": [list(value) for value in intervals]})
    calibration = _resolve_fresh_batches(
        tokens, context, batch_size, int(contract["calibration_batches"]),
        int(contract["calibration_seed"]), occupied,
    )
    body = {
        "schema": "hfw_public_calibration_layout_v3",
        "runtime_root": str(runtime_root),
        "data_relative_path": contract["relative_path"],
        "data_sha256": contract["sha256"],
        "data_size_bytes": identity.st_size,
        "data_inode": identity.st_ino,
        "dtype": contract["dtype"],
        "token_count": len(tokens),
        "context": context,
        "batch_size": batch_size,
        "prior_public_eval_exclusion": prior,
        "calibration": calibration,
        "calibration_offsets": [start for record in calibration for start in record["starts"]],
        "fresh_fixed_eval_offsets_present": False,
        "fixed_eval_seed_present": False,
        "canonical_commit_reveal_required": True,
        "new_unique_tokens_after_commitment": 0,
    }
    return body


def load_committed_batches(
    protocol: Mapping[str, object], runtime_root: Path,
    commitment: Mapping[str, object], section: str,
):
    path, identity = resolve_validation_corpus(protocol, runtime_root)
    contract = protocol["data"]
    if commitment.get("runtime_root") != str(runtime_root):
        raise RuntimeError("committed runtime-root binding drift")
    if commitment.get("data_relative_path") != contract["relative_path"]:
        raise RuntimeError("committed validation relative-path drift")
    if commitment.get("data_sha256") != contract["sha256"]:
        raise RuntimeError("committed validation SHA drift")
    if int(commitment.get("data_size_bytes", -1)) != identity.st_size:
        raise RuntimeError("committed validation size drift")
    tokens = np.memmap(
        path, dtype=np.dtype(commitment["dtype"]), mode="r"
    )
    context = int(commitment["context"])
    for record in commitment[section]:
        batch = np.stack([
            tokens[int(start):int(start) + context + 1]
            for start in record["starts"]
        ]).astype(np.int64)
        digest = hashlib.sha256(batch.astype("<i8", copy=False).tobytes()).hexdigest()
        if digest != record["raw_batch_sha256"]:
            raise RuntimeError("committed data batch drift")
        yield batch
