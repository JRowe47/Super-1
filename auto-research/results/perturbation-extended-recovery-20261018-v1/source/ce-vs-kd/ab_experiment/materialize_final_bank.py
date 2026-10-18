from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_exact(path: Path, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        body = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            body.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError(f"input changed while reading: {path}")
    observed = sha256_bytes(body)
    if observed != expected_sha256:
        raise RuntimeError(f"input SHA mismatch: {path}: {observed}")
    return bytes(body)


def write_exclusive(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def intervals_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def require_disjoint(intervals: Iterable[tuple[int, int]]) -> None:
    ordered = sorted(intervals)
    for left, right in zip(ordered, ordered[1:]):
        if intervals_overlap(left, right):
            raise RuntimeError(f"token intervals overlap: {left} and {right}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--val-sha256", required=True)
    parser.add_argument("--search-records", type=Path, required=True)
    parser.add_argument("--search-records-sha256", required=True)
    parser.add_argument("--verify-selection", type=Path, required=True)
    parser.add_argument("--verify-selection-sha256", required=True)
    parser.add_argument("--starting-checkpoint-sha256", required=True)
    parser.add_argument("--starting-head-sha256", required=True)
    parser.add_argument("--teacher-weights-sha256", required=True)
    parser.add_argument("--literal", required=True)
    parser.add_argument("--sequence-count", type=int, required=True)
    parser.add_argument("--sequence-tokens", type=int, required=True)
    parser.add_argument("--maximum-token-id", type=int, required=True)
    parser.add_argument("--bank-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()

    if args.sequence_count != 64 or args.sequence_tokens != 513:
        raise RuntimeError("FINAL contract requires exactly 64x513 tokens")
    val_body = read_exact(args.val, args.val_sha256)
    if len(val_body) % 4:
        raise RuntimeError("validation corpus is not uint32-aligned")
    tokens = np.frombuffer(val_body, dtype="<u4")
    search_body = read_exact(args.search_records, args.search_records_sha256)
    verify_body = read_exact(args.verify_selection, args.verify_selection_sha256)
    search = json.loads(search_body)
    verify = json.loads(verify_body)

    search_starts = [
        int(start)
        for record in search["search_records"]
        for start in record["starts"]
    ]
    verify_starts = [int(value) for value in verify["selection"]["starts"]]
    occupied = [
        (start, start + args.sequence_tokens)
        for start in (*search_starts, *verify_starts)
    ]
    require_disjoint(occupied)

    derivation = "|".join((
        args.literal,
        args.starting_checkpoint_sha256,
        args.starting_head_sha256,
        args.teacher_weights_sha256,
        sha256_bytes(search_body),
        sha256_bytes(verify_body),
    )).encode("utf-8")
    derivation_sha256 = sha256_bytes(derivation)
    maximum_start = len(tokens) - args.sequence_tokens
    if maximum_start < 0:
        raise RuntimeError("validation corpus is shorter than one sequence")
    modulus = maximum_start + 1
    starts: list[int] = []
    counter = 0
    while len(starts) < args.sequence_count:
        digest = hashlib.sha256(
            derivation + b"|" + counter.to_bytes(8, "big")
        ).digest()
        counter += 1
        start = int.from_bytes(digest[:8], "big") % modulus
        interval = (start, start + args.sequence_tokens)
        if any(intervals_overlap(interval, existing) for existing in occupied):
            continue
        occupied.append(interval)
        starts.append(start)
    require_disjoint(occupied)

    all_selected_starts = (*search_starts, *verify_starts, *starts)
    selected_maximum = max(
        int(tokens[start:start + args.sequence_tokens].max())
        for start in all_selected_starts
    )
    if selected_maximum > args.maximum_token_id:
        raise RuntimeError(
            f"selected token ID exceeds teacher tokenizer: "
            f"{selected_maximum} > {args.maximum_token_id}"
        )

    bank = np.stack([
        tokens[start:start + args.sequence_tokens] for start in starts
    ]).astype("<u4", copy=False).tobytes(order="C")
    expected_size = args.sequence_count * args.sequence_tokens * 4
    if len(bank) != expected_size:
        raise RuntimeError("FINAL bank size drift")
    bank_sha256 = sha256_bytes(bank)
    write_exclusive(args.bank_out, bank)
    receipt = {
        "schema": "hfw_w560_ce_vs_kd_final_bank_materialization_v1",
        "status": "PASS_FINAL_MATERIALIZED_BEFORE_ANY_AB_TEACHER_OR_AB_STUDENT_FORWARD",
        "role": "FINAL",
        "future_teacher_only_prelabel_read_before_arms_is_authorized": True,
        "not_claimed_unopened_by_teacher_until_arm_selections": True,
        "must_remain_unopened_by_student_optimizer_or_selection_until_both_arm_selections": True,
        "selection_algorithm": "sha256_counter_first_u64_mod_valid_start_reject_overlap_v1",
        "literal": args.literal,
        "derivation_sha256": derivation_sha256,
        "counters_examined": counter,
        "starts": starts,
        "intervals": [[start, start + args.sequence_tokens] for start in starts],
        "disjoint_from_search_and_verify": True,
        "selected_maximum_token_id": selected_maximum,
        "teacher_tokenizer_maximum_valid_token_id": args.maximum_token_id,
        "every_search_verify_final_token_id_valid_for_teacher_tokenizer": True,
        "source": {
            "sha256": args.val_sha256,
            "size_bytes": len(val_body),
            "dtype": "uint32_le",
            "token_count": len(tokens),
        },
        "exclusions": {
            "search_records_sha256": args.search_records_sha256,
            "search_sequence_count": len(search_starts),
            "verify_selection_sha256": args.verify_selection_sha256,
            "verify_sequence_count": len(verify_starts),
        },
        "identity_inputs": {
            "starting_checkpoint_sha256": args.starting_checkpoint_sha256,
            "starting_head_sha256": args.starting_head_sha256,
            "teacher_weights_sha256": args.teacher_weights_sha256,
        },
        "bank": {
            "path": str(args.bank_out.resolve()),
            "sha256": bank_sha256,
            "size_bytes": len(bank),
            "shape": [args.sequence_count, args.sequence_tokens],
            "dtype": "uint32_le",
        },
        "ab_teacher_forward_calls_before_materialization": 0,
        "ab_student_forward_calls_before_materialization": 0,
        "optimizer_steps": 0,
        "autograd_calls": 0,
    }
    write_exclusive(args.receipt_out, canonical_json_bytes(receipt))
    print(json.dumps({
        "bank_sha256": bank_sha256,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt)),
        "starts": starts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
