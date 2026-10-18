from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .common import (
    hash_file_same_fd, read_bytes_same_fd, read_json_same_fd,
    write_exclusive_bytes, write_exclusive_json,
)
from .release_binding import validate_runtime_manifest


def _starts(token_count: int, context: int, batch_size: int, step: int, seed: int) -> tuple[int, ...]:
    generator = np.random.default_rng(int(seed) + 1_000_003 * int(step))
    return tuple(int(value) for value in generator.integers(
        0, token_count - context - 1, size=batch_size
    ))


def _overlaps(interval: tuple[int, int], occupied: Sequence[tuple[int, int]]) -> bool:
    return any(interval[0] < stop and start < interval[1] for start, stop in occupied)


def _select_bank(
    tokens: np.ndarray, *, context: int, batch_size: int, batches: int,
    seed: int, occupied: list[tuple[int, int]],
) -> tuple[np.ndarray, list[Mapping[str, object]]]:
    records: list[Mapping[str, object]] = []
    arrays = []
    step = 0
    while len(records) < batches:
        starts = _starts(len(tokens), context, batch_size, step, seed)
        intervals = [(start, start + context + 1) for start in starts]
        if not any(
            _overlaps(interval, occupied + intervals[:index])
            for index, interval in enumerate(intervals)
        ):
            batch = np.stack([tokens[start:start + context + 1] for start in starts]).astype("<u4")
            records.append({
                "selection_step": step,
                "starts": list(starts),
                "intervals": [list(value) for value in intervals],
                "batch_sha256": hashlib.sha256(batch.tobytes(order="C")).hexdigest(),
            })
            arrays.append(batch)
            occupied.extend(intervals)
        step += 1
        if step > max(10_000, batches * 100_000):
            raise RuntimeError("could not construct a disjoint token bank")
    return np.stack(arrays), records


def _materialize_explicit_bank(
    tokens: np.ndarray, *, context: int, batch_size: int,
    starts: Sequence[int], occupied: list[tuple[int, int]],
) -> tuple[np.ndarray, list[Mapping[str, object]]]:
    if len(starts) % batch_size:
        raise RuntimeError("explicit bank starts do not divide into configured batches")
    intervals = [(int(start), int(start) + context + 1) for start in starts]
    for index, interval in enumerate(intervals):
        if interval[0] < 0 or interval[1] > len(tokens):
            raise RuntimeError("explicit token interval is outside corpus")
        if _overlaps(interval, occupied + intervals[:index]):
            raise RuntimeError("explicit token bank overlaps a precommitted interval")
    arrays = []
    records = []
    for batch_index in range(0, len(starts), batch_size):
        selected = [int(value) for value in starts[batch_index:batch_index + batch_size]]
        batch = np.stack([tokens[start:start + context + 1] for start in selected]).astype("<u4")
        batch_intervals = intervals[batch_index:batch_index + batch_size]
        arrays.append(batch)
        records.append({
            "selection_step": None,
            "published_explicit_starts": selected,
            "starts": selected,
            "intervals": [list(value) for value in batch_intervals],
            "batch_sha256": hashlib.sha256(batch.tobytes(order="C")).hexdigest(),
        })
    occupied.extend(intervals)
    return np.stack(arrays), records


def _verify_published_commit_provenance(
    config: Mapping[str, object], commit_spec: Mapping[str, object],
) -> Mapping[str, object]:
    reveal_spec = commit_spec.get("published_reveal")
    authorization_spec = commit_spec.get("prior_authorization")
    if not isinstance(reveal_spec, Mapping) or not isinstance(authorization_spec, Mapping):
        raise RuntimeError("published COMMIT provenance paths and full identities are required")
    reveal, reveal_identity = read_json_same_fd(
        Path(reveal_spec["path"]), str(reveal_spec["sha256"])
    )
    if reveal.get("schema") != reveal_spec["schema"]:
        raise RuntimeError("published reveal schema drift")
    if reveal.get("validation_sha256") != config["corpus"]["sha256"]:
        raise RuntimeError("published reveal validation SHA drift")
    expected_token_count = int(config["corpus"]["size_bytes"]) // np.dtype(
        config["corpus"]["dtype"]
    ).itemsize
    if int(reveal.get("validation_token_count", -1)) != expected_token_count:
        raise RuntimeError("published reveal validation length drift")
    if int(reveal.get("sample_length_tokens", -1)) != int(config["token_policy"]["context"]) + 1:
        raise RuntimeError("published reveal sample length drift")
    if [int(value) for value in reveal.get("evaluation_offsets", [])] != [
        int(value) for value in commit_spec["starts"]
    ]:
        raise RuntimeError("published reveal offsets drift")
    authorization, authorization_identity = read_json_same_fd(
        Path(authorization_spec["path"]), str(authorization_spec["sha256"])
    )
    if authorization.get("schema") != authorization_spec["schema"]:
        raise RuntimeError("prior authorization schema drift")
    if authorization.get("status") != authorization_spec["status"]:
        raise RuntimeError("prior authorization status drift")
    if authorization.get("published_reveal_sha256") != reveal_identity["sha256"]:
        raise RuntimeError("prior authorization reveal binding drift")
    if authorization.get("opaque_eval_commitment_sha256") != reveal_identity["sha256"]:
        raise RuntimeError("prior authorization opaque commitment binding drift")
    if authorization.get("validation_sha256") != config["corpus"]["sha256"]:
        raise RuntimeError("prior authorization validation SHA drift")
    if authorization.get("parent_checkpoint_sha256") != config["origin"]["checkpoint_sha256"]:
        raise RuntimeError("prior authorization immutable parent binding drift")
    if int(authorization.get("final_evaluation_calls_authorized", -1)) != 1:
        raise RuntimeError("prior authorization call count drift")
    if int(authorization.get("search_or_calibration_calls_after_reveal_authorized", -1)) != 0:
        raise RuntimeError("prior authorization post-reveal policy drift")
    return {
        "published_reveal": reveal_identity,
        "prior_one_call_authorization": authorization_identity,
        "published_reveal_schema": reveal["schema"],
        "prior_authorization_schema": authorization["schema"],
        "prior_authorization_status": authorization["status"],
        "validation_sha256": reveal["validation_sha256"],
        "validation_token_count": expected_token_count,
        "sample_length_tokens": int(reveal["sample_length_tokens"]),
        "evaluation_offsets": [int(value) for value in reveal["evaluation_offsets"]],
        "prior_authorization_reclassified_as_provenance_only": True,
    }


def prepare_token_commitment(config_path: Path, output_root: Path) -> Mapping[str, object]:
    config, config_identity = read_json_same_fd(config_path)
    runtime_binding = validate_runtime_manifest(config_path, config)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"token commitment root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    corpus = config["corpus"]
    corpus_path = Path(corpus["path"])
    corpus_payload, corpus_identity = read_bytes_same_fd(corpus_path, corpus["sha256"])
    if corpus_identity["sha256"] != corpus["sha256"]:
        raise RuntimeError("corpus SHA drift")
    if int(corpus_identity["size_bytes"]) != int(corpus["size_bytes"]):
        raise RuntimeError("corpus size drift")
    tokens = np.frombuffer(corpus_payload, dtype=np.dtype(corpus["dtype"]))
    policy = config["token_policy"]
    published_commit_provenance = _verify_published_commit_provenance(
        config, policy["commit"]
    )
    context = int(policy["context"])
    batch_size = int(policy["batch_size"])
    occupied: list[tuple[int, int]] = []
    prior_records = []
    prior = policy["prior_public_exclusion"]
    for step in range(int(prior["batches"])):
        starts = _starts(len(tokens), context, batch_size, step, int(prior["seed"]))
        intervals = [(start, start + context + 1) for start in starts]
        occupied.extend(intervals)
        prior_records.append({"step": step, "starts": list(starts), "intervals": [list(v) for v in intervals]})
    bank_values = {}
    bank_records = {}
    for role in ("SEARCH", "COMMIT"):
        spec = policy[role.lower()]
        if "starts" in spec:
            values, records = _materialize_explicit_bank(
                tokens, context=context, batch_size=batch_size,
                starts=spec["starts"], occupied=occupied,
            )
        else:
            values, records = _select_bank(
                tokens, context=context, batch_size=batch_size,
                batches=int(spec["batches"]), seed=int(spec["seed"]), occupied=occupied,
            )
        if values.shape[0] != int(spec["batches"]):
            raise RuntimeError(f"{role} bank count drift")
        payload = values.astype("<u4", copy=False).tobytes(order="C")
        observed_bank_sha = hashlib.sha256(payload).hexdigest()
        expected_bank_sha = spec.get("expected_u32le_sha256")
        if expected_bank_sha is not None and observed_bank_sha != expected_bank_sha:
            raise RuntimeError(f"{role} precommitted bank SHA drift")
        artifact = write_exclusive_bytes(output_root / f"{role.lower()}.u32le", payload)
        bank_values[role] = {
            **artifact,
            "shape": list(values.shape),
            "dtype": "uint32_le",
            "seed": None if "starts" in spec else int(spec["seed"]),
            "batches": int(spec["batches"]),
            "unique_token_positions": int(values.size),
            "published_explicit_starts": bool("starts" in spec),
            "provenance": {
                key: value for key, value in spec.items()
                if key not in {"starts", "seed", "batches", "expected_u32le_sha256"}
            },
            "expected_u32le_sha256": expected_bank_sha,
        }
        bank_records[role] = records
    manifest = {
        "schema": "hfw_w560_continuous_token_commitment_v1",
        "status": "PASS_SEARCH_AND_COMMIT_PREGENERATED_NO_FINAL_BANK",
        "config_sha256": config_identity["sha256"],
        "runtime_source_binding": runtime_binding,
        "corpus": corpus_identity,
        "context": context,
        "batch_size": batch_size,
        "prior_public_exclusion": prior_records,
        "published_commit_provenance": published_commit_provenance,
        "banks": bank_values,
        "selection_records": bank_records,
        "roles_generated_before_optimization": ["SEARCH", "COMMIT"],
        "final_bank_present": False,
        "new_unique_token_positions_after_initialization": 0,
    }
    manifest_artifact = write_exclusive_json(output_root / "token_manifest.json", manifest)
    return {**manifest, "token_manifest_artifact": manifest_artifact}


def load_bank(
    token_manifest_path: Path, role: str, *, expected_manifest_sha256: str | None = None,
) -> tuple[np.ndarray, Mapping[str, object]]:
    role = role.upper()
    if role not in {"SEARCH", "COMMIT"}:
        raise RuntimeError("only precommitted SEARCH or COMMIT banks may be loaded")
    manifest, _ = read_json_same_fd(token_manifest_path, expected_manifest_sha256)
    if manifest.get("final_bank_present") is not False:
        raise RuntimeError("continuous token manifest unexpectedly contains FINAL")
    record = manifest["banks"][role]
    path = Path(record["path"])
    payload, identity = read_bytes_same_fd(path, record["sha256"])
    if identity["sha256"] != record["sha256"] or identity["size_bytes"] != record["size_bytes"]:
        raise RuntimeError(f"{role} bank identity drift")
    shape = tuple(int(value) for value in record["shape"])
    values = np.frombuffer(payload, dtype="<u4")
    if values.size != int(np.prod(shape)):
        raise RuntimeError(f"{role} bank shape drift")
    return values.reshape(shape), manifest


def validate_token_manifest_against_config(
    config: Mapping[str, object], manifest: Mapping[str, object],
) -> Mapping[str, object]:
    policy = config["token_policy"]
    if manifest.get("schema") != "hfw_w560_continuous_token_commitment_v1":
        raise RuntimeError("token manifest schema drift")
    if manifest.get("final_bank_present") is not False:
        raise RuntimeError("token manifest contains an unauthorized FINAL bank")
    if manifest.get("roles_generated_before_optimization") != ["SEARCH", "COMMIT"]:
        raise RuntimeError("token manifest role initialization drift")
    runtime_binding = validate_runtime_manifest(
        Path(manifest["runtime_source_binding"]["manifest_file"]["path"]).parents[1]
        / "config/controller.json",
        config,
    )
    recorded_runtime = manifest.get("runtime_source_binding", {})
    if recorded_runtime.get("manifest_file", {}).get("sha256") != runtime_binding["manifest_file"]["sha256"]:
        raise RuntimeError("token manifest runtime source file SHA drift")
    if recorded_runtime.get("manifest_identity_sha256") != runtime_binding["manifest_identity_sha256"]:
        raise RuntimeError("token manifest runtime source content identity drift")
    if int(manifest.get("new_unique_token_positions_after_initialization", -1)) != 0:
        raise RuntimeError("token manifest permits new post-initialization tokens")
    if set(manifest.get("banks", {})) != {"SEARCH", "COMMIT"}:
        raise RuntimeError("token manifest bank set drift")
    for role in ("SEARCH", "COMMIT"):
        spec = policy[role.lower()]
        record = manifest["banks"][role]
        expected_shape = [
            int(spec["batches"]), int(policy["batch_size"]), int(policy["context"]) + 1,
        ]
        if record.get("shape") != expected_shape or record.get("dtype") != "uint32_le":
            raise RuntimeError(f"{role} token manifest shape/dtype drift")
        if record.get("sha256") != spec["expected_u32le_sha256"]:
            raise RuntimeError(f"{role} token manifest bank SHA differs from config")
        if int(record.get("batches", -1)) != int(spec["batches"]):
            raise RuntimeError(f"{role} token manifest batch count drift")
    commit_records = manifest["selection_records"]["COMMIT"]
    observed_starts = [
        int(value) for record in commit_records for value in record["starts"]
    ]
    if observed_starts != [int(value) for value in policy["commit"]["starts"]]:
        raise RuntimeError("COMMIT token manifest offsets differ from config")
    provenance = manifest.get("published_commit_provenance", {})
    for label, config_key in (
        ("published_reveal", "published_reveal"),
        ("prior_one_call_authorization", "prior_authorization"),
    ):
        observed = provenance.get(label, {})
        expected = policy["commit"][config_key]
        if observed.get("sha256") != expected["sha256"]:
            raise RuntimeError(f"token manifest {label} SHA drift")
        if Path(observed.get("path", "")).resolve() != Path(expected["path"]).resolve():
            raise RuntimeError(f"token manifest {label} path drift")
    return {
        "status": "PASS_TOKEN_MANIFEST_EXACT_CONFIG_BINDING",
        "search_bank_sha256": manifest["banks"]["SEARCH"]["sha256"],
        "commit_bank_sha256": manifest["banks"]["COMMIT"]["sha256"],
        "published_reveal_sha256": provenance["published_reveal"]["sha256"],
        "prior_authorization_sha256": provenance["prior_one_call_authorization"]["sha256"],
        "final_bank_present": False,
    }
