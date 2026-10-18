from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

from .common import (
    canonical_json_bytes, exclusive_lock, fsync_directory, hash_file_same_fd,
    read_json_same_fd, sha256_bytes, write_content_addressed_json,
    write_exclusive_bytes, write_exclusive_json,
)


LEDGER_DIRECTORIES = (
    "heads", "jobs", "worker_attempts", "proposals", "decisions", "deltas",
    "promoted_checkpoints", "failure_receipts", "recovery_receipts",
    "replacement_group_flats",
)


def _head_path(root: Path, generation: int, digest: str) -> Path:
    return root / "heads" / f"{generation:08d}-{digest}.head.json"


def _write_head_crash_atomic(root: Path, generation: int, payload: bytes) -> Mapping[str, object]:
    digest = sha256_bytes(payload)
    final_path = _head_path(root, generation, digest)
    if final_path.exists():
        raise FileExistsError(f"head already exists: {final_path}")
    staging = root / "heads" / f".{generation:08d}-{digest}.{os.getpid()}.{time.time_ns()}.staged"
    write_exclusive_bytes(staging, payload)
    try:
        os.link(staging, final_path)
        fsync_directory(final_path.parent)
    except FileExistsError:
        raise RuntimeError(f"concurrent head publication collision: {final_path}")
    staging.unlink()
    fsync_directory(final_path.parent)
    return hash_file_same_fd(final_path)


def initialize_ledger(
    root: Path, *, config_identity: Mapping[str, object],
    token_manifest_identity: Mapping[str, object], prepared_parent: Mapping[str, object],
    baseline_evidence: Mapping[str, object], origin: Mapping[str, object],
    runtime_source_binding: Mapping[str, object],
) -> Mapping[str, object]:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to reuse ledger: {root}")
    root.mkdir(parents=True)
    for name in LEDGER_DIRECTORIES:
        (root / name).mkdir()
    write_exclusive_bytes(root / "committer.lock", b"", mode=0o600)
    write_exclusive_bytes(root / "supervisor.lock", b"", mode=0o600)
    contract = {
        "schema": "hfw_w560_continuous_ledger_contract_v1",
        "config": dict(config_identity),
        "token_manifest": dict(token_manifest_identity),
        "prepared_parent": dict(prepared_parent),
        "origin_checkpoint_sha256": origin["checkpoint_sha256"],
        "stale_policy": "reject_without_merge_or_rebase",
        "single_serialized_committer": True,
        "append_only": True,
        "final_bank_present": False,
        "runtime_source_binding": dict(runtime_source_binding),
    }
    contract_artifact = write_exclusive_json(root / "contract.json", contract)
    head = {
        "schema": "hfw_w560_continuous_head_v1",
        "generation": 0,
        "parent_head_record_sha256": None,
        "parent_checkpoint_sha256": None,
        "checkpoint": {
            "path": prepared_parent["relocated_checkpoint"]["path"],
            "sha256": origin["checkpoint_sha256"],
            "size_bytes": origin["size_bytes"],
            "format": "immutable_training_root_checkpoint",
        },
        "origin_checkpoint_sha256": origin["checkpoint_sha256"],
        "model_state_digest": origin["model_state_digest"],
        "runtime_manifest_file_sha256": runtime_source_binding["manifest_file"]["sha256"],
        "runtime_manifest_identity_sha256": runtime_source_binding["manifest_identity_sha256"],
        "accepted_proposal_sha256": None,
        "decision_sha256": None,
        "delta_sha256": None,
        "group": None,
        "commit_objective": dict(baseline_evidence),
        "contract_sha256": contract_artifact["sha256"],
    }
    payload = canonical_json_bytes(head)
    digest = sha256_bytes(payload)
    artifact = _write_head_crash_atomic(root, 0, payload)
    return {**head, "head_record_artifact": artifact, "contract_artifact": contract_artifact}


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "contract.json").is_file():
            raise RuntimeError(f"not an initialized ledger: {self.root}")

    def lock(self):
        return exclusive_lock(self.root / "committer.lock")

    def supervisor_lock(self):
        return exclusive_lock(self.root / "supervisor.lock", nonblocking=True)

    def contract(self) -> Mapping[str, object]:
        contract, contract_identity = read_json_same_fd(self.root / "contract.json")
        generation_zero_paths = sorted((self.root / "heads").glob("00000000-*.head.json"))
        if len(generation_zero_paths) != 1:
            raise RuntimeError("ledger generation-zero head is absent or ambiguous")
        generation_zero, zero_identity = read_json_same_fd(generation_zero_paths[0])
        if generation_zero_paths[0].name != f"00000000-{zero_identity['sha256']}.head.json":
            raise RuntimeError("generation-zero head filename/content hash drift")
        if generation_zero.get("schema") != "hfw_w560_continuous_head_v1":
            raise RuntimeError("generation-zero head schema drift")
        if generation_zero.get("contract_sha256") != contract_identity["sha256"]:
            raise RuntimeError("ledger contract bytes differ from generation-zero anchor")
        return contract

    def validate_runtime_inputs(
        self, config_path: Path, token_manifest_path: Path,
    ) -> Mapping[str, object]:
        contract = self.contract()
        from .release_binding import validate_runtime_manifest
        config_value, config_identity = read_json_same_fd(config_path)
        if config_identity["sha256"] != contract["config"]["sha256"]:
            raise RuntimeError("ledger config SHA drift")
        if int(config_identity["size_bytes"]) != int(contract["config"]["size_bytes"]):
            raise RuntimeError("ledger config size drift")
        runtime_binding = validate_runtime_manifest(config_path, config_value)
        expected_runtime = contract["runtime_source_binding"]
        if runtime_binding["manifest_file"]["sha256"] != expected_runtime["manifest_file"]["sha256"]:
            raise RuntimeError("ledger runtime manifest file SHA drift")
        if int(runtime_binding["manifest_file"]["size_bytes"]) != int(expected_runtime["manifest_file"]["size_bytes"]):
            raise RuntimeError("ledger runtime manifest file size drift")
        if runtime_binding["manifest_identity_sha256"] != expected_runtime["manifest_identity_sha256"]:
            raise RuntimeError("ledger runtime manifest content identity drift")
        token_manifest, token_identity = read_json_same_fd(token_manifest_path)
        if token_identity["sha256"] != contract["token_manifest"]["sha256"]:
            raise RuntimeError("ledger token manifest SHA drift")
        if int(token_identity["size_bytes"]) != int(contract["token_manifest"]["size_bytes"]):
            raise RuntimeError("ledger token manifest size drift")
        if token_manifest.get("config_sha256") != config_identity["sha256"]:
            raise RuntimeError("token manifest/config ledger binding drift")
        if token_manifest.get("runtime_source_binding", {}).get("manifest_file", {}).get("sha256") != runtime_binding["manifest_file"]["sha256"]:
            raise RuntimeError("token/ledger runtime manifest binding drift")
        return {
            "config": config_identity, "token_manifest": token_identity,
            "runtime_source_binding": runtime_binding,
            "config_value": config_value,
            "token_manifest_value": token_manifest,
        }

    def validate_job_path(
        self, job_path: Path, *, expected_sha256: str | None = None,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        path = Path(job_path)
        if path.resolve().parent != (self.root / "jobs").resolve():
            raise RuntimeError("proposal job is outside the append-only ledger job directory")
        match = re.fullmatch(r"(\d{12})-([0-9a-f]{64})\.job\.json", path.name)
        if match is None:
            raise RuntimeError("proposal job filename contract drift")
        job, identity = read_json_same_fd(path, expected_sha256)
        if identity["sha256"] != match.group(2):
            raise RuntimeError("proposal job filename/content hash drift")
        if job.get("schema") != "hfw_w560_continuous_proposal_job_v1":
            raise RuntimeError("proposal job schema drift")
        if int(job.get("job_index", -1)) != int(match.group(1)):
            raise RuntimeError("proposal job index drift")
        base = self.head_by_record_sha(str(job["base_head_record_sha256"]))
        if int(job["base_generation"]) != int(base["generation"]):
            raise RuntimeError("proposal job base generation drift")
        if job["base_checkpoint_sha256"] != base["checkpoint"]["sha256"]:
            raise RuntimeError("proposal job base checkpoint drift")
        if job.get("stale_policy") != "reject_without_merge_or_rebase":
            raise RuntimeError("proposal job stale policy drift")
        if job.get("worker_may_commit") is not False:
            raise RuntimeError("proposal job granted worker commit authority")
        return job, identity

    def heads(self) -> list[Mapping[str, object]]:
        rows = []
        contract_identity = hash_file_same_fd(self.root / "contract.json")
        contract = self.contract()
        paths = sorted((self.root / "heads").glob("*.head.json"))
        for generation, path in enumerate(paths):
            value, identity = read_json_same_fd(path)
            prefix, digest = path.name.split("-", 1)
            digest = digest.removesuffix(".head.json")
            if int(prefix) != generation or int(value["generation"]) != generation:
                raise RuntimeError("head generations are not contiguous")
            if identity["sha256"] != digest:
                raise RuntimeError("head filename/content hash drift")
            if value.get("schema") != "hfw_w560_continuous_head_v1":
                raise RuntimeError("head schema drift")
            if value.get("contract_sha256") != contract_identity["sha256"]:
                raise RuntimeError("head/ledger contract SHA drift")
            if value.get("origin_checkpoint_sha256") != contract["origin_checkpoint_sha256"]:
                raise RuntimeError("head immutable origin drift")
            if value.get("runtime_manifest_file_sha256") != contract["runtime_source_binding"]["manifest_file"]["sha256"]:
                raise RuntimeError("head runtime manifest file identity drift")
            if value.get("runtime_manifest_identity_sha256") != contract["runtime_source_binding"]["manifest_identity_sha256"]:
                raise RuntimeError("head runtime manifest content identity drift")
            if generation:
                parent = rows[-1]
                if value["parent_head_record_sha256"] != parent["record_sha256"]:
                    raise RuntimeError("head record chain break")
                if value["parent_checkpoint_sha256"] != parent["checkpoint"]["sha256"]:
                    raise RuntimeError("head checkpoint chain break")
                decision_path = self.root / "decisions" / f"{value['decision_sha256']}.decision.json"
                decision, decision_identity = read_json_same_fd(
                    decision_path, value["decision_sha256"]
                )
                if decision.get("accepted") is not True:
                    raise RuntimeError("promoted head points to a non-accepted decision")
                if decision.get("proposal_sha256") != value["accepted_proposal_sha256"]:
                    raise RuntimeError("promoted head proposal/decision lineage drift")
                if decision.get("child_checkpoint", {}).get("sha256") != value["checkpoint"]["sha256"]:
                    raise RuntimeError("promoted head checkpoint/decision lineage drift")
                if decision.get("ledger_delta", {}).get("sha256") != value["delta_sha256"]:
                    raise RuntimeError("promoted head delta/decision lineage drift")
                if value.get("solver_cumulative_delta_sha256") != value["delta_sha256"]:
                    raise RuntimeError("promoted head solver cumulative delta alias drift")
                selected = decision.get("selected_candidate", {})
                if selected.get("cumulative_delta", {}).get("sha256") != value["solver_cumulative_delta_sha256"]:
                    raise RuntimeError("promoted head solver delta/selected candidate drift")
                if selected.get("exact_candidate_group_flat", {}).get("sha256") != value.get("replacement_group_flat_sha256"):
                    raise RuntimeError("promoted head exact replacement/selected candidate drift")
                if decision.get("ledger_replacement_group_flat", {}).get("sha256") != value.get("replacement_group_flat_sha256"):
                    raise RuntimeError("promoted head exact replacement/decision artifact drift")
                if value.get("solver_cumulative_delta", {}).get("sha256") != value.get("solver_cumulative_delta_sha256"):
                    raise RuntimeError("promoted head solver delta artifact drift")
                if value.get("replacement_group_flat", {}).get("sha256") != value.get("replacement_group_flat_sha256"):
                    raise RuntimeError("promoted head exact replacement artifact drift")
            rows.append({**value, "record_sha256": identity["sha256"], "record_path": identity["path"]})
        if not rows:
            raise RuntimeError("ledger has no head")
        return rows

    def current_head(self) -> Mapping[str, object]:
        return self.heads()[-1]

    def head_by_record_sha(self, digest: str) -> Mapping[str, object]:
        matches = [row for row in self.heads() if row["record_sha256"] == digest]
        if len(matches) != 1:
            raise RuntimeError("job base head is absent or ambiguous")
        return matches[0]

    def create_job(self, *, group: str, root_seed: int, allowed_groups: Sequence[str]) -> Mapping[str, object]:
        if group not in allowed_groups:
            raise RuntimeError(f"unauthorized perturbation group: {group}")
        with self.lock():
            head = self.current_head()
            existing = sorted((self.root / "jobs").glob("*.job.json"))
            job_index = len(existing)
            seed_material = f"{root_seed}:{job_index}:{group}:{head['record_sha256']}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") & ((1 << 63) - 1)
            job = {
                "schema": "hfw_w560_continuous_proposal_job_v1",
                "job_index": job_index,
                "group": group,
                "seed": seed,
                "base_head_record_sha256": head["record_sha256"],
                "base_generation": head["generation"],
                "base_checkpoint_sha256": head["checkpoint"]["sha256"],
                "stale_policy": "reject_without_merge_or_rebase",
                "worker_may_commit": False,
            }
            payload = canonical_json_bytes(job)
            digest = sha256_bytes(payload)
            artifact = write_exclusive_bytes(
                self.root / "jobs" / f"{job_index:012d}-{digest}.job.json", payload
            )
            return {**job, "job_record_sha256": digest, "job_artifact": artifact}

    def append_decision(self, decision: Mapping[str, object]) -> Mapping[str, object]:
        return write_content_addressed_json(self.root / "decisions", ".decision.json", decision)

    def append_head(self, head: Mapping[str, object]) -> Mapping[str, object]:
        current = self.current_head()
        if int(head["generation"]) != int(current["generation"]) + 1:
            raise RuntimeError("new head generation is not the unique successor")
        if head["parent_head_record_sha256"] != current["record_sha256"]:
            raise RuntimeError("new head parent record mismatch")
        if head["parent_checkpoint_sha256"] != current["checkpoint"]["sha256"]:
            raise RuntimeError("new head parent checkpoint mismatch")
        payload = canonical_json_bytes(head)
        digest = sha256_bytes(payload)
        return _write_head_crash_atomic(self.root, int(head["generation"]), payload)

    def validate_history(self, verify_checkpoints: bool = False) -> Mapping[str, object]:
        heads = self.heads()
        if verify_checkpoints:
            for head in heads:
                identity = hash_file_same_fd(Path(head["checkpoint"]["path"]))
                if identity["sha256"] != head["checkpoint"]["sha256"]:
                    raise RuntimeError("head checkpoint SHA drift")
                if int(identity["size_bytes"]) != int(head["checkpoint"]["size_bytes"]):
                    raise RuntimeError("head checkpoint size drift")
                if int(head["generation"]) == 0:
                    continue
                for label, directory, key, sha_key in (
                    (
                        "solver cumulative delta", "deltas",
                        "solver_cumulative_delta", "solver_cumulative_delta_sha256",
                    ),
                    (
                        "exact replacement group-flat", "replacement_group_flats",
                        "replacement_group_flat", "replacement_group_flat_sha256",
                    ),
                ):
                    artifact = head.get(key, {})
                    artifact_path = Path(artifact["path"])
                    if artifact_path.resolve().parent != (self.root / directory).resolve():
                        raise RuntimeError(f"head {label} is outside its append-only directory")
                    artifact_identity = hash_file_same_fd(artifact_path)
                    if artifact_identity["sha256"] != head[sha_key]:
                        raise RuntimeError(f"head {label} SHA drift")
                    if artifact_identity["sha256"] != artifact.get("sha256"):
                        raise RuntimeError(f"head {label} record SHA drift")
                    if int(artifact_identity["size_bytes"]) != int(artifact.get("size_bytes", -1)):
                        raise RuntimeError(f"head {label} size drift")
        return {
            "status": "PASS_APPEND_ONLY_HEAD_HISTORY",
            "generations": len(heads),
            "accepted_promotions": len(heads) - 1,
            "current_head_record_sha256": heads[-1]["record_sha256"],
            "current_checkpoint_sha256": heads[-1]["checkpoint"]["sha256"],
            "checkpoint_and_dual_lineage_bytes_verified": verify_checkpoints,
        }
