"""Commit-reveal evaluation split and append-only terminal registries.

The search-facing capability contains no evaluation offsets, salt, validation
identity, or authority path.  Proposal decisions, including rejections, form a
content-addressed hash chain.  A reveal can be published only after a terminal
entry irrevocably seals that chain.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time
from typing import Mapping, Sequence


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EVENT_PATTERN = re.compile(r"^(\d{20})-([0-9a-f]{64})\.event\.json$")
TERMINAL_PATTERN = re.compile(r"^(\d{20})-([0-9a-f]{64})\.terminal\.json$")


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    value = str(value)
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _ensure_nonoverlapping_offsets(
    offsets: Sequence[int], sample_length_tokens: int, val_token_count: int, label: str
) -> list[int]:
    length = int(sample_length_tokens)
    token_count = int(val_token_count)
    if length <= 0 or token_count <= 0:
        raise ValueError(f"{label} lengths must be positive")
    canonical = sorted({int(value) for value in offsets})
    if len(canonical) != len(offsets) or not canonical:
        raise ValueError(f"{label} offsets must be nonempty and unique")
    intervals = []
    for offset in canonical:
        if offset < 0 or offset + length > token_count:
            raise ValueError(f"{label} offset outside validation token bounds")
        intervals.append((offset, offset + length))
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(f"{label} windows overlap internally")
    return canonical


def _open_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        raise


def _write_content_addressed(
    directory: Path, suffix: str, payload: bytes, mode: int = 0o444
) -> tuple[str, Path]:
    digest = sha256_bytes(payload)
    path = directory / f"{digest}{suffix}"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"content-address collision: {path}")
        return digest, path
    _open_exclusive(path, payload, mode)
    return digest, path


def _read_content_addressed(path: Path, suffix: str) -> tuple[dict, str]:
    if not path.name.endswith(suffix):
        raise RuntimeError(f"unexpected content-address suffix: {path}")
    digest = path.name[: -len(suffix)]
    _require_sha256(digest, "content address")
    payload = path.read_bytes()
    if sha256_bytes(payload) != digest:
        raise RuntimeError(f"content-address mutation detected: {path}")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"content-addressed JSON object required: {path}")
    if canonical_json_bytes(decoded) != payload:
        raise RuntimeError(f"noncanonical content-addressed JSON: {path}")
    return decoded, digest


@contextmanager
def _registry_lock(public_root: Path):
    lock_path = public_root / "locks" / "sealed_evaluation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class SearchCapability:
    public_root: Path
    opaque_eval_commitment_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_root", Path(self.public_root).resolve())
        _require_sha256(
            self.opaque_eval_commitment_sha256, "opaque evaluation commitment"
        )

    def public_payload(self) -> dict:
        return {
            "schema": "sealed_evaluation_search_capability_v1",
            "public_root": str(self.public_root),
            "opaque_eval_commitment_sha256": self.opaque_eval_commitment_sha256,
        }


def _session_path(capability: SearchCapability) -> Path:
    return (
        capability.public_root
        / "sessions"
        / f"{capability.opaque_eval_commitment_sha256}.session.json"
    )


def _validate_session(capability: SearchCapability) -> dict:
    path = _session_path(capability)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("opaque evaluation session is missing or symlinked")
    payload = json.loads(path.read_bytes())
    expected = {
        "schema": "opaque_eval_commitment_session_v1",
        "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
    }
    if payload != expected or path.read_bytes() != canonical_json_bytes(expected):
        raise RuntimeError("opaque evaluation session mutation detected")
    return payload


def create_eval_commitment(
    *,
    authority_root: Path,
    public_root: Path,
    validation_sha256: str,
    validation_token_count: int,
    evaluation_offsets: Sequence[int],
    sample_length_tokens: int,
    salt: bytes | None = None,
) -> SearchCapability:
    """Create the private preimage and return only a public opaque capability."""

    authority_root = Path(authority_root).resolve()
    public_root = Path(public_root).resolve()
    if authority_root == public_root:
        raise ValueError("authority and search-visible roots must be distinct")
    try:
        authority_root.relative_to(public_root)
        raise ValueError("authority root cannot be inside the search-visible root")
    except ValueError as error:
        if str(error).startswith("authority root"):
            raise
    try:
        public_root.relative_to(authority_root)
        raise ValueError("search-visible root cannot be inside the authority root")
    except ValueError as error:
        if str(error).startswith("search-visible root"):
            raise

    validation_sha256 = _require_sha256(validation_sha256, "validation artifact")
    offsets = _ensure_nonoverlapping_offsets(
        evaluation_offsets,
        sample_length_tokens,
        validation_token_count,
        "evaluation",
    )
    salt = secrets.token_bytes(32) if salt is None else bytes(salt)
    if len(salt) < 32:
        raise ValueError("evaluation commitment salt must contain at least 256 bits")
    preimage = {
        "schema": "sealed_evaluation_reveal_preimage_v1",
        "validation_sha256": validation_sha256,
        "validation_token_count": int(validation_token_count),
        "evaluation_offsets": offsets,
        "sample_length_tokens": int(sample_length_tokens),
        "salt_hex": salt.hex(),
    }
    encoded = canonical_json_bytes(preimage)
    commitment = sha256_bytes(encoded)
    private_path = authority_root / "sealed_reveals" / f"{commitment}.reveal.json"
    _open_exclusive(private_path, encoded, mode=0o400)
    session = {
        "schema": "opaque_eval_commitment_session_v1",
        "opaque_eval_commitment_sha256": commitment,
    }
    _open_exclusive(
        public_root / "sessions" / f"{commitment}.session.json",
        canonical_json_bytes(session),
        mode=0o444,
    )
    return SearchCapability(public_root, commitment)


def create_calibration_manifest(
    capability: SearchCapability,
    *,
    validation_sha256: str,
    validation_token_count: int,
    calibration_offsets: Sequence[int],
    sample_length_tokens: int,
) -> str:
    _validate_session(capability)
    if terminal_entry_for_commitment(capability) is not None:
        raise RuntimeError("terminal registry already sealed")
    payload = {
        "schema": "sealed_evaluation_calibration_manifest_v1",
        "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
        "validation_sha256": _require_sha256(validation_sha256, "validation artifact"),
        "validation_token_count": int(validation_token_count),
        "calibration_offsets": _ensure_nonoverlapping_offsets(
            calibration_offsets,
            sample_length_tokens,
            validation_token_count,
            "calibration",
        ),
        "sample_length_tokens": int(sample_length_tokens),
    }
    digest, _ = _write_content_addressed(
        capability.public_root / "calibration_manifests",
        ".calibration.json",
        canonical_json_bytes(payload),
    )
    return digest


def _event_files(public_root: Path) -> list[Path]:
    directory = public_root / "proposal_events"
    return sorted(directory.glob("*.event.json")) if directory.exists() else []


def validate_proposal_chain(capability: SearchCapability) -> dict:
    _validate_session(capability)
    events = []
    previous = None
    accepted = 0
    rejected = 0
    for expected_sequence, path in enumerate(_event_files(capability.public_root)):
        match = EVENT_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unexpected proposal event filename: {path.name}")
        sequence = int(match.group(1))
        digest = match.group(2)
        if sequence != expected_sequence:
            raise RuntimeError("proposal event sequence is not contiguous")
        payload = path.read_bytes()
        if sha256_bytes(payload) != digest:
            raise RuntimeError("proposal event content-address mutation detected")
        event = json.loads(payload)
        if canonical_json_bytes(event) != payload:
            raise RuntimeError("proposal event JSON is noncanonical")
        if event.get("sequence") != sequence:
            raise RuntimeError("proposal event embedded sequence drift")
        if event.get("previous_event_sha256") != previous:
            raise RuntimeError("proposal event hash-chain break")
        if event.get("opaque_eval_commitment_sha256") != capability.opaque_eval_commitment_sha256:
            raise RuntimeError("proposal event commitment drift")
        decision = event.get("decision")
        if decision == "accepted":
            accepted += 1
        elif decision == "rejected":
            rejected += 1
        else:
            raise RuntimeError("proposal event decision drift")
        events.append({"sha256": digest, "payload": event})
        previous = digest
    return {
        "status": "PASS",
        "event_count": len(events),
        "accepted": accepted,
        "rejected": rejected,
        "head_event_sha256": previous,
        "events": events,
    }


def append_proposal_decision(
    capability: SearchCapability,
    *,
    proposal_sha256: str,
    parent_checkpoint_sha256: str,
    delta_sha256: str,
    decision: str,
    decision_reason: str,
    record: Mapping[str, object] | None = None,
) -> str:
    if decision not in ("accepted", "rejected"):
        raise ValueError("proposal decision must be accepted or rejected")
    if not decision_reason:
        raise ValueError("proposal decision reason cannot be empty")
    with _registry_lock(capability.public_root):
        if terminal_entry_for_commitment(capability) is not None:
            raise RuntimeError("terminal registry already sealed; proposals are closed")
        chain = validate_proposal_chain(capability)
        event = {
            "schema": "sealed_evaluation_proposal_event_v1",
            "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
            "sequence": chain["event_count"],
            "previous_event_sha256": chain["head_event_sha256"],
            "proposal_sha256": _require_sha256(proposal_sha256, "proposal"),
            "parent_checkpoint_sha256": _require_sha256(
                parent_checkpoint_sha256, "parent checkpoint"
            ),
            "delta_sha256": _require_sha256(delta_sha256, "delta"),
            "decision": decision,
            "decision_reason": decision_reason,
            "record": dict(record or {}),
        }
        encoded = canonical_json_bytes(event)
        digest = sha256_bytes(encoded)
        path = (
            capability.public_root
            / "proposal_events"
            / f"{chain['event_count']:020d}-{digest}.event.json"
        )
        _open_exclusive(path, encoded)
        return digest


def _terminal_files(public_root: Path) -> list[Path]:
    directory = public_root / "terminal_registry"
    return sorted(directory.glob("*.terminal.json")) if directory.exists() else []


def validate_terminal_registry(public_root: Path) -> dict:
    public_root = Path(public_root).resolve()
    entries = []
    previous = None
    commitments = set()
    for expected_sequence, path in enumerate(_terminal_files(public_root)):
        match = TERMINAL_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unexpected terminal entry filename: {path.name}")
        sequence = int(match.group(1))
        digest = match.group(2)
        if sequence != expected_sequence:
            raise RuntimeError("terminal registry sequence is not contiguous")
        payload = path.read_bytes()
        if sha256_bytes(payload) != digest:
            raise RuntimeError("terminal registry content-address mutation detected")
        entry = json.loads(payload)
        if canonical_json_bytes(entry) != payload:
            raise RuntimeError("terminal registry JSON is noncanonical")
        if entry.get("sequence") != sequence or entry.get("previous_terminal_sha256") != previous:
            raise RuntimeError("terminal registry hash-chain break")
        commitment = _require_sha256(
            entry.get("opaque_eval_commitment_sha256"), "terminal commitment"
        )
        if commitment in commitments:
            raise RuntimeError("evaluation commitment has multiple terminal seals")
        commitments.add(commitment)
        entries.append({"sha256": digest, "payload": entry})
        previous = digest
    return {
        "status": "PASS",
        "entry_count": len(entries),
        "head_terminal_sha256": previous,
        "entries": entries,
    }


def terminal_entry_for_commitment(capability: SearchCapability) -> dict | None:
    registry = validate_terminal_registry(capability.public_root)
    matches = [
        entry
        for entry in registry["entries"]
        if entry["payload"]["opaque_eval_commitment_sha256"]
        == capability.opaque_eval_commitment_sha256
    ]
    if len(matches) > 1:
        raise RuntimeError("multiple terminal entries for evaluation commitment")
    return matches[0] if matches else None


def seal_terminal_registry(
    capability: SearchCapability,
    *,
    final_checkpoint_sha256: str,
    calibration_manifest_sha256: str,
    terminal_status: str,
) -> str:
    if not terminal_status:
        raise ValueError("terminal status cannot be empty")
    with _registry_lock(capability.public_root):
        _validate_session(capability)
        if terminal_entry_for_commitment(capability) is not None:
            raise RuntimeError("evaluation commitment is already terminal")
        calibration_path = (
            capability.public_root
            / "calibration_manifests"
            / f"{_require_sha256(calibration_manifest_sha256, 'calibration manifest')}.calibration.json"
        )
        calibration, _ = _read_content_addressed(calibration_path, ".calibration.json")
        if calibration.get("opaque_eval_commitment_sha256") != capability.opaque_eval_commitment_sha256:
            raise RuntimeError("calibration manifest commitment drift")
        proposals = validate_proposal_chain(capability)
        registry = validate_terminal_registry(capability.public_root)
        entry = {
            "schema": "sealed_evaluation_terminal_registry_entry_v1",
            "sequence": registry["entry_count"],
            "previous_terminal_sha256": registry["head_terminal_sha256"],
            "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
            "terminal_status": terminal_status,
            "final_checkpoint_sha256": _require_sha256(
                final_checkpoint_sha256, "final checkpoint"
            ),
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "proposal_chain": {
                "head_event_sha256": proposals["head_event_sha256"],
                "event_count": proposals["event_count"],
                "accepted": proposals["accepted"],
                "rejected": proposals["rejected"],
            },
        }
        encoded = canonical_json_bytes(entry)
        digest = sha256_bytes(encoded)
        path = (
            capability.public_root
            / "terminal_registry"
            / f"{registry['entry_count']:020d}-{digest}.terminal.json"
        )
        _open_exclusive(path, encoded)
        return digest


def publish_reveal(
    capability: SearchCapability, *, authority_root: Path
) -> str:
    """Publish the exact private preimage only after terminal closure."""

    with _registry_lock(capability.public_root):
        terminal = terminal_entry_for_commitment(capability)
        if terminal is None:
            raise PermissionError("evaluation reveal forbidden before terminal registry seal")
        commitment = capability.opaque_eval_commitment_sha256
        private_path = (
            Path(authority_root).resolve()
            / "sealed_reveals"
            / f"{commitment}.reveal.json"
        )
        if not private_path.is_file() or private_path.is_symlink():
            raise RuntimeError("authority reveal artifact is missing or symlinked")
        encoded = private_path.read_bytes()
        if sha256_bytes(encoded) != commitment:
            raise RuntimeError("authority reveal artifact mutation detected")
        preimage = json.loads(encoded)
        if canonical_json_bytes(preimage) != encoded:
            raise RuntimeError("authority reveal artifact is noncanonical")
        public_path = capability.public_root / "reveals" / f"{commitment}.reveal.json"
        if public_path.exists():
            raise FileExistsError("evaluation reveal was already published")
        _open_exclusive(public_path, encoded)
        return commitment


def read_public_reveal(capability: SearchCapability) -> dict:
    terminal = terminal_entry_for_commitment(capability)
    if terminal is None:
        raise PermissionError("evaluation reveal unavailable before terminal registry seal")
    commitment = capability.opaque_eval_commitment_sha256
    path = capability.public_root / "reveals" / f"{commitment}.reveal.json"
    if not path.is_file() or path.is_symlink():
        raise PermissionError("evaluation reveal has not been published")
    encoded = path.read_bytes()
    if sha256_bytes(encoded) != commitment:
        raise RuntimeError("published evaluation reveal mutation detected")
    payload = json.loads(encoded)
    if canonical_json_bytes(payload) != encoded:
        raise RuntimeError("published evaluation reveal is noncanonical")
    return payload


def prove_calibration_evaluation_disjoint(
    capability: SearchCapability,
    *,
    calibration_manifest_sha256: str,
    validation_path: Path | None = None,
) -> tuple[str, dict]:
    reveal = read_public_reveal(capability)
    terminal = terminal_entry_for_commitment(capability)
    if terminal is None:
        raise AssertionError("terminal disappeared after reveal read")
    calibration_manifest_sha256 = _require_sha256(
        calibration_manifest_sha256, "calibration manifest"
    )
    if terminal["payload"]["calibration_manifest_sha256"] != calibration_manifest_sha256:
        raise RuntimeError("terminal seal binds a different calibration manifest")
    calibration_path = (
        capability.public_root
        / "calibration_manifests"
        / f"{calibration_manifest_sha256}.calibration.json"
    )
    calibration, _ = _read_content_addressed(calibration_path, ".calibration.json")
    if calibration["validation_sha256"] != reveal["validation_sha256"]:
        raise RuntimeError("calibration/evaluation validation SHA mismatch")
    if calibration["validation_token_count"] != reveal["validation_token_count"]:
        raise RuntimeError("calibration/evaluation validation length mismatch")
    if validation_path is not None:
        actual = sha256_file(Path(validation_path).resolve())
        if actual != reveal["validation_sha256"]:
            raise RuntimeError("validation artifact bytes do not match revealed SHA")

    calibration_intervals = [
        (offset, offset + calibration["sample_length_tokens"])
        for offset in calibration["calibration_offsets"]
    ]
    evaluation_intervals = [
        (offset, offset + reveal["sample_length_tokens"])
        for offset in reveal["evaluation_offsets"]
    ]
    overlaps = []
    for calibration_interval in calibration_intervals:
        for evaluation_interval in evaluation_intervals:
            if max(calibration_interval[0], evaluation_interval[0]) < min(
                calibration_interval[1], evaluation_interval[1]
            ):
                overlaps.append(
                    {
                        "calibration": list(calibration_interval),
                        "evaluation": list(evaluation_interval),
                    }
                )
    if overlaps:
        raise RuntimeError(f"calibration/evaluation windows overlap: {overlaps}")
    proof = {
        "schema": "sealed_evaluation_disjointness_proof_v1",
        "status": "PASS",
        "blockers": [],
        "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
        "terminal_entry_sha256": terminal["sha256"],
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "validation_sha256": reveal["validation_sha256"],
        "validation_bytes_rehashed": validation_path is not None,
        "calibration_window_count": len(calibration_intervals),
        "evaluation_window_count": len(evaluation_intervals),
        "overlap_count": 0,
    }
    proof_sha, _ = _write_content_addressed(
        capability.public_root / "disjointness_proofs",
        ".proof.json",
        canonical_json_bytes(proof),
    )
    return proof_sha, proof


def audit_sealed_evaluation(capability: SearchCapability) -> dict:
    proposals = validate_proposal_chain(capability)
    terminal = terminal_entry_for_commitment(capability)
    revealed = False
    if terminal is not None:
        reveal_path = (
            capability.public_root
            / "reveals"
            / f"{capability.opaque_eval_commitment_sha256}.reveal.json"
        )
        if reveal_path.exists():
            read_public_reveal(capability)
            revealed = True
        sealed = terminal["payload"]["proposal_chain"]
        current = {
            key: proposals[key]
            for key in ("head_event_sha256", "event_count", "accepted", "rejected")
        }
        if sealed != current:
            raise RuntimeError("proposal chain changed after terminal seal")
    return {
        "status": "PASS",
        "blockers": [],
        "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
        "proposal_chain": {
            key: proposals[key]
            for key in ("head_event_sha256", "event_count", "accepted", "rejected")
        },
        "terminal_sealed": terminal is not None,
        "terminal_entry_sha256": None if terminal is None else terminal["sha256"],
        "revealed": revealed,
    }
