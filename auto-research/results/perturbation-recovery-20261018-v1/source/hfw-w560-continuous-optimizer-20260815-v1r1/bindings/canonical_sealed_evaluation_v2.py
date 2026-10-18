"""Production integration helpers layered over the immutable v1 sealer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
from typing import Mapping, Sequence

from sealed_evaluation import (
    SearchCapability,
    append_proposal_decision,
    canonical_json_bytes,
    create_eval_commitment,
    prove_calibration_evaluation_disjoint,
    publish_reveal,
    read_public_reveal,
    seal_terminal_registry,
    sha256_bytes,
    terminal_entry_for_commitment,
    validate_proposal_chain,
)


def _require_sha256(value: object, label: str) -> str:
    value = str(value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _write_content_addressed(
    directory: Path, suffix: str, payload: bytes, mode: int = 0o444
) -> tuple[str, Path]:
    digest = sha256_bytes(payload)
    path = directory / f"{digest}{suffix}"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("content-address collision")
        return digest, path
    _write_exclusive(path, payload, mode)
    return digest, path


def _read_content_addressed(path: Path, suffix: str) -> tuple[dict, str]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("content-addressed artifact is missing or symlinked")
    if not path.name.endswith(suffix):
        raise RuntimeError("content-addressed suffix drift")
    digest = _require_sha256(path.name[: -len(suffix)], "content address")
    payload = path.read_bytes()
    if sha256_bytes(payload) != digest:
        raise RuntimeError("content-addressed artifact mutation detected")
    decoded = json.loads(payload)
    if canonical_json_bytes(decoded) != payload:
        raise RuntimeError("content-addressed artifact is noncanonical")
    return decoded, digest


def _normalize_intervals(
    intervals: Sequence[Mapping[str, object] | Sequence[int]],
    validation_token_count: int,
    label: str,
) -> list[dict[str, int]]:
    normalized = []
    for interval in intervals:
        if isinstance(interval, Mapping):
            start = int(interval["start"])
            end = int(interval["end_exclusive"])
        else:
            if len(interval) != 2:
                raise ValueError(f"{label} interval needs start and end")
            start, end = (int(interval[0]), int(interval[1]))
        if start < 0 or end <= start or end > int(validation_token_count):
            raise ValueError(f"{label} interval outside validation bounds")
        normalized.append({"start": start, "end_exclusive": end})
    normalized.sort(key=lambda row: (row["start"], row["end_exclusive"]))
    return normalized


def _offset_intervals(
    offsets: Sequence[int], sample_length_tokens: int, validation_token_count: int
) -> list[dict[str, int]]:
    intervals = []
    seen = set()
    for value in offsets:
        start = int(value)
        if start in seen:
            raise ValueError("calibration offsets must be unique")
        seen.add(start)
        end = start + int(sample_length_tokens)
        if start < 0 or end > int(validation_token_count):
            raise ValueError("calibration interval outside validation bounds")
        intervals.append({"start": start, "end_exclusive": end})
    return intervals


def _intervals_overlap(left: Mapping[str, int], right: Mapping[str, int]) -> bool:
    return max(left["start"], right["start"]) < min(
        left["end_exclusive"], right["end_exclusive"]
    )


def _private_offset_plan(
    *,
    validation_sha256: str,
    validation_token_count: int,
    evaluation_window_count: int,
    sample_length_tokens: int,
    offset_stride_tokens: int,
    excluded_intervals: Sequence[Mapping[str, object] | Sequence[int]],
    private_entropy: bytes,
) -> dict:
    validation_sha256 = _require_sha256(validation_sha256, "validation artifact")
    token_count = int(validation_token_count)
    window_count = int(evaluation_window_count)
    window_length = int(sample_length_tokens)
    stride = int(offset_stride_tokens)
    if token_count <= 0 or window_count <= 0 or window_length <= 0:
        raise ValueError("validation length, window count, and window length must be positive")
    if stride < window_length:
        raise ValueError("offset stride must be at least one evaluation window")
    if len(private_entropy) < 32:
        raise ValueError("private evaluation entropy must contain at least 256 bits")
    exclusions = _normalize_intervals(
        excluded_intervals, token_count, "public exclusion"
    )
    derivation_contract = {
        "schema": "authority_private_eval_offset_derivation_v1",
        "algorithm": "HMAC_SHA256_score_sort_over_stride_aligned_candidates",
        "validation_sha256": validation_sha256,
        "validation_token_count": token_count,
        "evaluation_window_count": window_count,
        "sample_length_tokens": window_length,
        "offset_stride_tokens": stride,
        "excluded_intervals": exclusions,
    }
    context = canonical_json_bytes(derivation_contract)
    candidates = []
    for start in range(0, token_count - window_length + 1, stride):
        interval = {"start": start, "end_exclusive": start + window_length}
        if any(_intervals_overlap(interval, excluded) for excluded in exclusions):
            continue
        score = hmac.new(
            private_entropy,
            context + start.to_bytes(8, "little", signed=False),
            hashlib.sha256,
        ).digest()
        candidates.append((score, start))
    if len(candidates) < window_count:
        raise RuntimeError(
            "insufficient validation capacity after public/calibration exclusions"
        )
    selected = sorted(start for _, start in sorted(candidates)[:window_count])
    return {
        **derivation_contract,
        "private_entropy_hex": private_entropy.hex(),
        "candidate_count_after_exclusions": len(candidates),
        "evaluation_offsets": selected,
    }


@dataclass(frozen=True)
class AuthorityGeneratedCommitment:
    search_capability: SearchCapability
    authority_root: Path
    generation_record_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "authority_root", Path(self.authority_root).resolve())
        _require_sha256(self.generation_record_sha256, "generation record")


def create_authority_generated_eval_commitment(
    *,
    authority_root: Path,
    public_root: Path,
    validation_sha256: str,
    validation_token_count: int,
    evaluation_window_count: int,
    sample_length_tokens: int,
    offset_stride_tokens: int,
    calibration_offsets: Sequence[int],
    calibration_sample_length_tokens: int,
    public_excluded_intervals: Sequence[Mapping[str, object] | Sequence[int]] = (),
    private_entropy: bytes | None = None,
) -> AuthorityGeneratedCommitment:
    """Select secret eval offsets without exposing them to the search process."""

    authority_root = Path(authority_root).resolve()
    private_entropy = secrets.token_bytes(32) if private_entropy is None else bytes(private_entropy)
    calibration_intervals = _offset_intervals(
        calibration_offsets,
        calibration_sample_length_tokens,
        validation_token_count,
    )
    exclusions = calibration_intervals + _normalize_intervals(
        public_excluded_intervals,
        validation_token_count,
        "public exclusion",
    )
    plan = _private_offset_plan(
        validation_sha256=validation_sha256,
        validation_token_count=validation_token_count,
        evaluation_window_count=evaluation_window_count,
        sample_length_tokens=sample_length_tokens,
        offset_stride_tokens=offset_stride_tokens,
        excluded_intervals=exclusions,
        private_entropy=private_entropy,
    )
    search_capability = create_eval_commitment(
        authority_root=authority_root,
        public_root=public_root,
        validation_sha256=validation_sha256,
        validation_token_count=validation_token_count,
        evaluation_offsets=plan["evaluation_offsets"],
        sample_length_tokens=sample_length_tokens,
        salt=private_entropy,
    )
    generation_record = {
        "schema": "authority_private_eval_generation_record_v1",
        "opaque_eval_commitment_sha256": (
            search_capability.opaque_eval_commitment_sha256
        ),
        "plan": plan,
    }
    generation_sha, _ = _write_content_addressed(
        authority_root / "generation_records",
        ".generation.json",
        canonical_json_bytes(generation_record),
        mode=0o400,
    )
    return AuthorityGeneratedCommitment(
        search_capability=search_capability,
        authority_root=authority_root,
        generation_record_sha256=generation_sha,
    )


def append_branch_event(
    capability: SearchCapability,
    *,
    branch_event: Mapping[str, object],
    allowed_groups: Sequence[str],
) -> str:
    """Fail-closed adapter from a solver branch decision into the canonical chain."""

    required = {
        "branch_id",
        "group",
        "proposal_sha256",
        "parent_checkpoint_sha256",
        "child_checkpoint_sha256",
        "delta_sha256",
        "decision",
        "decision_reason",
        "calibration_manifest_sha256",
        "summed_nll_before",
        "summed_nll_after",
    }
    missing = sorted(required - set(branch_event))
    if missing:
        raise ValueError(f"branch event missing required fields: {missing}")
    branch_id = str(branch_event["branch_id"])
    group = str(branch_event["group"])
    if not branch_id:
        raise ValueError("branch id cannot be empty")
    allowed_groups = tuple(str(value) for value in allowed_groups)
    if len(set(allowed_groups)) != len(allowed_groups) or group not in allowed_groups:
        raise ValueError(f"branch group is not authorized: {group}")
    decision = str(branch_event["decision"])
    if decision not in ("accepted", "rejected"):
        raise ValueError("every branch proposal must be explicitly accepted or rejected")
    parent = _require_sha256(branch_event["parent_checkpoint_sha256"], "branch parent")
    child = _require_sha256(branch_event["child_checkpoint_sha256"], "branch child")
    before = float(branch_event["summed_nll_before"])
    after = float(branch_event["summed_nll_after"])
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError("branch calibration NLL values must be finite")
    if decision == "accepted":
        if child == parent:
            raise ValueError("accepted branch proposal must advance checkpoint identity")
        if not after < before:
            raise ValueError("accepted branch proposal must improve summed calibration NLL")
    elif child != parent:
        raise ValueError("rejected branch proposal must retain its parent checkpoint")
    record = {
        "schema": "canonical_branch_event_adapter_record_v1",
        "branch_id": branch_id,
        "group": group,
        "child_checkpoint_sha256": child,
        "calibration_manifest_sha256": _require_sha256(
            branch_event["calibration_manifest_sha256"], "calibration manifest"
        ),
        "summed_nll_before": before,
        "summed_nll_after": after,
        "device": branch_event.get("device"),
        "runtime": branch_event.get("runtime"),
        "precision": branch_event.get("precision"),
        "metadata": dict(branch_event.get("metadata", {})),
    }
    return append_proposal_decision(
        capability,
        proposal_sha256=_require_sha256(branch_event["proposal_sha256"], "proposal"),
        parent_checkpoint_sha256=parent,
        delta_sha256=_require_sha256(branch_event["delta_sha256"], "delta"),
        decision=decision,
        decision_reason=str(branch_event["decision_reason"]),
        record=record,
    )


def _audit_exact_four_branch_finals(
    capability: SearchCapability,
    *,
    parent_checkpoint_sha256: str,
    calibration_manifest_sha256: str,
    required_groups: Sequence[str],
    branch_finals: Sequence[Mapping[str, object]],
) -> tuple[dict, list[dict]]:
    parent_checkpoint_sha256 = _require_sha256(
        parent_checkpoint_sha256, "common branch parent"
    )
    calibration_manifest_sha256 = _require_sha256(
        calibration_manifest_sha256, "calibration manifest"
    )
    required_groups = tuple(str(value) for value in required_groups)
    if len(required_groups) != 4 or len(set(required_groups)) != 4:
        raise ValueError("final authorization requires exactly four unique groups")
    if len(branch_finals) != 4:
        raise ValueError("final authorization requires exactly four branch finals")
    chain = validate_proposal_chain(capability)
    events_by_branch: dict[str, list[dict]] = {}
    for event in chain["events"]:
        record = event["payload"].get("record", {})
        if record.get("schema") != "canonical_branch_event_adapter_record_v1":
            raise RuntimeError("proposal chain contains a non-branch-adapter event")
        if record.get("group") not in required_groups:
            raise RuntimeError("proposal chain contains an unauthorized branch group")
        if record.get("calibration_manifest_sha256") != calibration_manifest_sha256:
            raise RuntimeError("branch event calibration manifest drift")
        events_by_branch.setdefault(record["branch_id"], []).append(event)

    canonical_finals = []
    seen_branches = set()
    seen_groups = set()
    for final in branch_finals:
        branch_id = str(final["branch_id"])
        group = str(final["group"])
        if branch_id in seen_branches or group in seen_groups:
            raise ValueError("branch finals must have unique branch ids and groups")
        seen_branches.add(branch_id)
        seen_groups.add(group)
        if group not in required_groups:
            raise ValueError(f"unexpected final branch group: {group}")
        events = events_by_branch.get(branch_id, [])
        if not events:
            raise RuntimeError(f"branch final has no canonical events: {branch_id}")
        current = parent_checkpoint_sha256
        for event in events:
            payload = event["payload"]
            record = payload["record"]
            if record["group"] != group:
                raise RuntimeError("one branch id changed groups")
            if payload["parent_checkpoint_sha256"] != current:
                raise RuntimeError("branch checkpoint lineage is not contiguous")
            current = record["child_checkpoint_sha256"]
        expected_final = _require_sha256(final["final_checkpoint_sha256"], "branch final")
        expected_event = _require_sha256(final["last_branch_event_sha256"], "last branch event")
        if current != expected_final or events[-1]["sha256"] != expected_event:
            raise RuntimeError("branch final does not bind its canonical terminal event")
        canonical_finals.append(
            {
                "branch_id": branch_id,
                "group": group,
                "final_checkpoint_sha256": expected_final,
                "last_branch_event_sha256": expected_event,
                "proposal_event_count": len(events),
            }
        )
    if seen_groups != set(required_groups) or seen_branches != set(events_by_branch):
        raise RuntimeError("branch finals are not exhaustive over the canonical event chain")
    canonical_finals.sort(key=lambda row: (row["group"], row["branch_id"]))
    return chain, canonical_finals


def _verify_private_generation(
    authority: AuthorityGeneratedCommitment,
) -> tuple[dict, dict]:
    record_path = (
        authority.authority_root
        / "generation_records"
        / f"{authority.generation_record_sha256}.generation.json"
    )
    record, _ = _read_content_addressed(record_path, ".generation.json")
    commitment = authority.search_capability.opaque_eval_commitment_sha256
    if record.get("opaque_eval_commitment_sha256") != commitment:
        raise RuntimeError("private generation record commitment drift")
    plan = record["plan"]
    recomputed = _private_offset_plan(
        validation_sha256=plan["validation_sha256"],
        validation_token_count=plan["validation_token_count"],
        evaluation_window_count=plan["evaluation_window_count"],
        sample_length_tokens=plan["sample_length_tokens"],
        offset_stride_tokens=plan["offset_stride_tokens"],
        excluded_intervals=plan["excluded_intervals"],
        private_entropy=bytes.fromhex(plan["private_entropy_hex"]),
    )
    if recomputed != plan:
        raise RuntimeError("private evaluation offset derivation is not reproducible")
    private_reveal_path = (
        authority.authority_root / "sealed_reveals" / f"{commitment}.reveal.json"
    )
    if not private_reveal_path.is_file() or private_reveal_path.is_symlink():
        raise RuntimeError("private reveal is missing or symlinked")
    private_bytes = private_reveal_path.read_bytes()
    if sha256_bytes(private_bytes) != commitment:
        raise RuntimeError("private reveal commitment mutation detected")
    private_reveal = json.loads(private_bytes)
    if canonical_json_bytes(private_reveal) != private_bytes:
        raise RuntimeError("private reveal is noncanonical")
    if (
        private_reveal["evaluation_offsets"] != plan["evaluation_offsets"]
        or private_reveal["salt_hex"] != plan["private_entropy_hex"]
        or private_reveal["validation_sha256"] != plan["validation_sha256"]
    ):
        raise RuntimeError("private reveal differs from generated offset plan")
    return record, private_reveal


def seal_reveal_authorize_four_branch_final_eval(
    authority: AuthorityGeneratedCommitment,
    *,
    parent_checkpoint_sha256: str,
    calibration_manifest_sha256: str,
    required_groups: Sequence[str],
    branch_finals: Sequence[Mapping[str, object]],
    terminal_status: str,
    validation_path: Path,
) -> dict:
    """Atomically ordered terminal gate for the one final held-out evaluation."""

    capability = authority.search_capability
    if terminal_entry_for_commitment(capability) is not None:
        raise RuntimeError("terminal seal already exists; final authorization is create-only")
    generation_record, private_reveal = _verify_private_generation(authority)
    chain, canonical_finals = _audit_exact_four_branch_finals(
        capability,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        calibration_manifest_sha256=calibration_manifest_sha256,
        required_groups=required_groups,
        branch_finals=branch_finals,
    )
    branch_bundle = {
        "schema": "exact_four_branch_final_bundle_v1",
        "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
        "parent_checkpoint_sha256": _require_sha256(
            parent_checkpoint_sha256, "common branch parent"
        ),
        "calibration_manifest_sha256": _require_sha256(
            calibration_manifest_sha256, "calibration manifest"
        ),
        "required_groups": sorted(str(value) for value in required_groups),
        "branch_finals": canonical_finals,
        "canonical_proposal_chain_head_sha256": chain["head_event_sha256"],
        "canonical_proposal_event_count": chain["event_count"],
        "canonical_accepted_count": chain["accepted"],
        "canonical_rejected_count": chain["rejected"],
    }
    bundle_sha, _ = _write_content_addressed(
        capability.public_root / "branch_final_bundles",
        ".branches.json",
        canonical_json_bytes(branch_bundle),
    )
    terminal_sha = seal_terminal_registry(
        capability,
        final_checkpoint_sha256=bundle_sha,
        calibration_manifest_sha256=calibration_manifest_sha256,
        terminal_status=terminal_status,
    )
    publish_reveal(capability, authority_root=authority.authority_root)
    public_reveal = read_public_reveal(capability)
    if public_reveal != private_reveal:
        raise RuntimeError("published reveal differs from pre-seal verified reveal")
    proof_sha, proof = prove_calibration_evaluation_disjoint(
        capability,
        calibration_manifest_sha256=calibration_manifest_sha256,
        validation_path=validation_path,
    )
    authorization = {
        "schema": "exact_four_branch_final_eval_authorization_v1",
        "status": "AUTHORIZED",
        "blockers": [],
        "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
        "parent_checkpoint_sha256": branch_bundle["parent_checkpoint_sha256"],
        "branch_final_bundle_sha256": bundle_sha,
        "terminal_entry_sha256": terminal_sha,
        "published_reveal_sha256": capability.opaque_eval_commitment_sha256,
        "authority_generation_record_sha256": authority.generation_record_sha256,
        "disjointness_proof_sha256": proof_sha,
        "validation_sha256": proof["validation_sha256"],
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "required_branch_final_count": 4,
        "actual_branch_final_count": len(canonical_finals),
        "final_evaluation_calls_authorized": 1,
        "search_or_calibration_calls_after_reveal_authorized": 0,
    }
    authorization_sha, authorization_path = _write_content_addressed(
        capability.public_root / "final_eval_authorizations",
        ".authorization.json",
        canonical_json_bytes(authorization),
    )
    return {
        "status": "AUTHORIZED",
        "blockers": [],
        "authorization_sha256": authorization_sha,
        "authorization_path": str(authorization_path),
        "branch_final_bundle_sha256": bundle_sha,
        "terminal_entry_sha256": terminal_sha,
        "reveal_sha256": capability.opaque_eval_commitment_sha256,
        "disjointness_proof_sha256": proof_sha,
        "generation_record_sha256": authority.generation_record_sha256,
        "branch_final_count": len(canonical_finals),
        "accepted_proposals": chain["accepted"],
        "rejected_proposals": chain["rejected"],
        "final_evaluation_calls_authorized": 1,
    }
