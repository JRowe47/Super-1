from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import time
from typing import Mapping

from .common import (
    canonical_json_bytes, exclusive_lock, fsync_directory, hash_file_same_fd,
    read_json_same_fd, sha256_bytes, write_content_addressed_json,
    write_exclusive_bytes, write_exclusive_json,
)
from .local_config import load_config


HEAD_SCHEMA = "hfw_w560_mps_local_head_v1"
JOB_SCHEMA = "hfw_w560_mps_local_solver_job_v1"
CONTRACT_SCHEMA = "hfw_w560_mps_local_append_only_lineage_contract_v1"
DECISION_SCHEMA = "hfw_w560_mps_local_verify_decision_v1"
LEDGER_DIRECTORIES = (
    "heads", "jobs", "proposal_imports", "decisions", "deltas",
    "replacement_group_flats", "promoted_checkpoints", "failure_receipts",
)


def _write_head_atomic(root: Path, generation: int, head: Mapping[str, object]) -> Mapping[str, object]:
    payload = canonical_json_bytes(head)
    digest = sha256_bytes(payload)
    final_path = root / "heads" / f"{generation:08d}-{digest}.head.json"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite local head: {final_path}")
    staging = root / "heads" / f".{generation:08d}-{digest}.{os.getpid()}.{time.time_ns()}.staged"
    write_exclusive_bytes(staging, payload)
    try:
        os.link(staging, final_path)
        fsync_directory(final_path.parent)
    finally:
        staging.unlink(missing_ok=True)
        fsync_directory(final_path.parent)
    return hash_file_same_fd(final_path)


def initialize_local_ledger(config_path: Path, ledger_root: Path) -> Mapping[str, object]:
    config, config_identity = load_config(config_path)
    root = Path(ledger_root)
    if root.exists():
        raise FileExistsError(f"local lineage root must be fresh: {root}")
    root.mkdir(parents=True)
    for name in LEDGER_DIRECTORIES:
        (root / name).mkdir()
    write_exclusive_bytes(root / "lineage.lock", b"", mode=0o600)
    runtime = config_identity["runtime_source_binding"]
    contract = {
        "schema": CONTRACT_SCHEMA,
        "config": {key: config_identity[key] for key in ("path", "sha256", "size_bytes")},
        "runtime_manifest_file_sha256": runtime["manifest_file"]["sha256"],
        "runtime_manifest_identity_sha256": runtime["manifest_identity_sha256"],
        "source_current_head_record_sha256": config["lineage"]["source_current_head_record_sha256"],
        "source_current_checkpoint_sha256": config["lineage"]["current_checkpoint_sha256"],
        "origin_checkpoint_sha256": config["lineage"]["origin_checkpoint_sha256"],
        "local_verify_bank_sha256": config["local_verify"]["bank"]["sha256"],
        "local_verify_materialization_receipt_sha256": config["local_verify"]["materialization_receipt"]["sha256"],
        "local_verify_disclosure": "PERMANENT_REUSED_OPTIMIZATION_DATA_NOT_HELD_OUT",
        "held_out_generalization_claimed": False,
        "stale_policy": "reject_without_merge_or_rebase",
        "append_only": True,
        "single_serialized_local_promoter": True,
        "cloud_access": False,
        "training_or_backprop": False,
        "adam_or_any_optimizer": False,
    }
    contract_artifact = write_exclusive_json(root / "contract.json", contract)
    lineage = config["lineage"]
    checkpoint = hash_file_same_fd(Path(lineage["current_checkpoint_path"]))
    historical_delta = hash_file_same_fd(Path(lineage["current_solver_delta_path"]))
    historical_replacement = hash_file_same_fd(Path(lineage["current_replacement_group_flat_path"]))
    head = {
        "schema": HEAD_SCHEMA,
        "generation": int(lineage["source_generation"]),
        "local_promotions": int(lineage["source_local_promotions"]),
        "parent_head_record_sha256": None,
        "source_current_head_record_sha256": lineage["source_current_head_record_sha256"],
        "parent_checkpoint_sha256": lineage["source_parent_checkpoint_sha256"],
        "checkpoint": {**checkpoint, "format": "hfw_w560_continuous_promoted_checkpoint_v1"},
        "origin_checkpoint_sha256": lineage["origin_checkpoint_sha256"],
        "model_state_digest": lineage["current_model_state_digest"],
        "accepted_proposal_sha256": lineage["current_accepted_proposal_sha256"],
        "selected_candidate_order": int(lineage["current_selected_candidate_order"]),
        "decision_sha256": None,
        "delta_sha256": historical_delta["sha256"],
        "solver_cumulative_delta_sha256": historical_delta["sha256"],
        "replacement_group_flat_sha256": historical_replacement["sha256"],
        "solver_cumulative_delta": historical_delta,
        "replacement_group_flat": historical_replacement,
        "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
        "group": lineage["current_group"],
        "local_verify_evidence": None,
        "contract_sha256": contract_artifact["sha256"],
        "runtime_manifest_file_sha256": runtime["manifest_file"]["sha256"],
        "runtime_manifest_identity_sha256": runtime["manifest_identity_sha256"],
    }
    head_artifact = _write_head_atomic(root, int(head["generation"]), head)
    return {**head, "record_sha256": head_artifact["sha256"], "record_path": head_artifact["path"], "contract_artifact": contract_artifact}


class LocalLedger:
    def __init__(self, root: Path, config_path: Path):
        self.root = Path(root)
        self.config_path = Path(config_path)
        if not (self.root / "contract.json").is_file():
            raise RuntimeError("local lineage is not initialized")

    def lock(self):
        return exclusive_lock(self.root / "lineage.lock")

    def contract(self) -> tuple[Mapping[str, object], Mapping[str, object]]:
        contract, identity = read_json_same_fd(self.root / "contract.json")
        if contract.get("schema") != CONTRACT_SCHEMA:
            raise RuntimeError("local lineage contract schema drift")
        config, config_identity = load_config(self.config_path)
        if identity["sha256"] not in [head.get("contract_sha256") for head in self._raw_heads()]:
            raise RuntimeError("local lineage contract is not anchored by any head")
        if contract["config"]["sha256"] != config_identity["sha256"]:
            raise RuntimeError("local lineage config SHA drift")
        if contract["local_verify_bank_sha256"] != config["local_verify"]["bank"]["sha256"]:
            raise RuntimeError("local lineage LOCAL_VERIFY bank drift")
        return contract, identity

    def _raw_heads(self) -> list[Mapping[str, object]]:
        rows = []
        for path in sorted((self.root / "heads").glob("*.head.json")):
            value, identity = read_json_same_fd(path)
            rows.append({**value, "record_sha256": identity["sha256"], "record_path": identity["path"]})
        return rows

    def heads(self) -> list[Mapping[str, object]]:
        rows = self._raw_heads()
        if not rows:
            raise RuntimeError("local lineage has no head")
        contract, contract_identity = read_json_same_fd(self.root / "contract.json")
        first_generation = int(rows[0]["generation"])
        for index, row in enumerate(rows):
            generation = first_generation + index
            path = Path(row["record_path"])
            match = re.fullmatch(r"(\d{8})-([0-9a-f]{64})\.head\.json", path.name)
            if match is None or int(match.group(1)) != generation or match.group(2) != row["record_sha256"]:
                raise RuntimeError("local head filename/content generation drift")
            if row.get("schema") != HEAD_SCHEMA or int(row["generation"]) != generation:
                raise RuntimeError("local head schema/generation drift")
            if row.get("contract_sha256") != contract_identity["sha256"]:
                raise RuntimeError("local head contract drift")
            if row.get("origin_checkpoint_sha256") != contract["origin_checkpoint_sha256"]:
                raise RuntimeError("local head origin drift")
            checkpoint = hash_file_same_fd(Path(row["checkpoint"]["path"]))
            if checkpoint["sha256"] != row["checkpoint"]["sha256"] or int(checkpoint["size_bytes"]) != int(row["checkpoint"]["size_bytes"]):
                raise RuntimeError("local head checkpoint byte drift")
            if index:
                parent = rows[index - 1]
                if row["parent_head_record_sha256"] != parent["record_sha256"]:
                    raise RuntimeError("local head record chain break")
                if row["parent_checkpoint_sha256"] != parent["checkpoint"]["sha256"]:
                    raise RuntimeError("local checkpoint chain break")
                if int(row["local_promotions"]) != int(parent["local_promotions"]) + 1:
                    raise RuntimeError("local promotion count drift")
                decision, _ = read_json_same_fd(
                    self.root / "decisions" / f"{row['decision_sha256']}.decision.json",
                    row["decision_sha256"],
                )
                if decision.get("accepted") is not True or decision["child_checkpoint"]["sha256"] != row["checkpoint"]["sha256"]:
                    raise RuntimeError("local promoted head decision lineage drift")
        return rows

    def current_head(self) -> Mapping[str, object]:
        return self.heads()[-1]

    def head_by_record_sha(self, digest: str) -> Mapping[str, object]:
        matches = [row for row in self.heads() if row["record_sha256"] == digest]
        if len(matches) != 1:
            raise RuntimeError("local head record is absent or ambiguous")
        return matches[0]

    def allocate_next_job(self) -> Mapping[str, object]:
        config, _ = load_config(self.config_path)
        with self.lock():
            incomplete = [row for row in self.job_states() if row["stage"] != "COMPLETE"]
            if incomplete:
                raise RuntimeError(
                    "refusing to allocate a successor while a prior local job is incomplete: "
                    f"{[(row['job_index'], row['stage']) for row in incomplete]}"
                )
            current = self.current_head()
            existing = sorted((self.root / "jobs").glob("*.job.json"))
            job_index = len(existing)
            groups = config["controller"]["groups"]
            group = groups[job_index % len(groups)]
            seed_material = f"{config['controller']['root_seed']}:{job_index}:{group}:{current['record_sha256']}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") & ((1 << 63) - 1)
            job = {
                "schema": JOB_SCHEMA,
                "job_index": job_index,
                "group": group,
                "seed": seed,
                "base_head_record_sha256": current["record_sha256"],
                "base_generation": current["generation"],
                "base_checkpoint_sha256": current["checkpoint"]["sha256"],
                "stale_policy": "reject_without_merge_or_rebase",
                "solver_may_promote": False,
                "local_promoter_is_only_head_writer": True,
            }
            payload = canonical_json_bytes(job)
            digest = sha256_bytes(payload)
            artifact = write_exclusive_bytes(
                self.root / "jobs" / f"{job_index:012d}-{digest}.job.json", payload,
            )
            return {**job, "job_record_sha256": digest, "job_artifact": artifact}

    def validate_job(self, path: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
        job, identity = read_json_same_fd(path)
        if Path(path).resolve().parent != (self.root / "jobs").resolve():
            raise RuntimeError("local solver job is outside append-only queue")
        match = re.fullmatch(r"(\d{12})-([0-9a-f]{64})\.job\.json", Path(path).name)
        if match is None or match.group(2) != identity["sha256"]:
            raise RuntimeError("local solver job filename/content drift")
        if job.get("schema") != JOB_SCHEMA or int(job["job_index"]) != int(match.group(1)):
            raise RuntimeError("local solver job schema/index drift")
        base = self.head_by_record_sha(job["base_head_record_sha256"])
        if int(job["base_generation"]) != int(base["generation"]) or job["base_checkpoint_sha256"] != base["checkpoint"]["sha256"]:
            raise RuntimeError("local solver job base lineage drift")
        return job, identity

    def jobs(self) -> list[Mapping[str, object]]:
        rows = []
        paths = sorted((self.root / "jobs").glob("*.job.json"))
        for expected_index, path in enumerate(paths):
            job, identity = self.validate_job(path)
            if int(job["job_index"]) != expected_index:
                raise RuntimeError("local solver job indices are not contiguous")
            rows.append({**job, "job_record_sha256": identity["sha256"], "job_artifact": identity})
        return rows

    def job_states(self) -> list[Mapping[str, object]]:
        proposals_by_job: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
        for path in sorted((self.root / "proposal_imports").glob("*.local-proposal.json")):
            proposal, identity = read_json_same_fd(path)
            job_sha = proposal.get("local_solver_job_record_sha256")
            if job_sha is None:
                continue
            proposals_by_job.setdefault(str(job_sha), []).append((proposal, identity))
        if any(len(rows) != 1 for rows in proposals_by_job.values()):
            raise RuntimeError("local solver job has ambiguous multiple proposal chains")
        decisions_by_proposal: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
        for path in sorted((self.root / "decisions").glob("*.decision.json")):
            decision, identity = read_json_same_fd(path)
            if decision.get("schema") != DECISION_SCHEMA:
                raise RuntimeError("local decision schema drift during job-state audit")
            decisions_by_proposal.setdefault(str(decision["proposal_sha256"]), []).append(
                (decision, identity)
            )
        heads_by_decision: dict[str, list[Mapping[str, object]]] = {}
        for head in self.heads():
            decision_sha = head.get("decision_sha256")
            if decision_sha is not None:
                heads_by_decision.setdefault(str(decision_sha), []).append(head)
        jobs = self.jobs()
        known = {row["job_record_sha256"] for row in jobs}
        if not set(proposals_by_job).issubset(known):
            raise RuntimeError("localized proposal references an unknown local solver job")
        states = []
        for job in jobs:
            proposal_rows = proposals_by_job.get(job["job_record_sha256"], [])
            if not proposal_rows:
                states.append({**job, "stage": "SOLVER_PENDING", "proposal": None, "decision": None})
                continue
            proposal, proposal_identity = proposal_rows[0]
            decision_rows = decisions_by_proposal.get(proposal_identity["sha256"], [])
            if len(decision_rows) > 1:
                raise RuntimeError("local proposal has ambiguous multiple verification decisions")
            if not decision_rows:
                states.append({
                    **job,
                    "stage": "VERIFY_PENDING",
                    "proposal": {**proposal, "artifact": proposal_identity},
                    "decision": None,
                })
                continue
            decision, decision_identity = decision_rows[0]
            evidence = decision.get("local_verify_evidence")
            if decision.get("local_verify_opened") is True:
                if not isinstance(evidence, Mapping):
                    raise RuntimeError("LOCAL_VERIFY decision has no evidence mapping")
                if evidence.get("all_saved_candidates_evaluated") is not True:
                    raise RuntimeError("LOCAL_VERIFY decision did not evaluate every saved candidate")
                if int(evidence.get("candidate_count", -1)) != int(proposal["candidate_count"]):
                    raise RuntimeError("LOCAL_VERIFY decision/proposal candidate-count drift")
            elif decision.get("status") != "REJECTED_STALE_HEAD_NO_MERGE_NO_REBASE_NO_LOCAL_VERIFY_OPEN":
                raise RuntimeError("non-stale local decision closed without LOCAL_VERIFY")
            anchored_heads = heads_by_decision.get(decision_identity["sha256"], [])
            if decision.get("accepted") is True:
                if not anchored_heads:
                    states.append({
                        **job,
                        "stage": "ACCEPTED_DECISION_UNHEADED_BLOCKED",
                        "proposal": {**proposal, "artifact": proposal_identity},
                        "decision": {**decision, "artifact": decision_identity},
                    })
                    continue
                if len(anchored_heads) != 1:
                    raise RuntimeError("accepted local decision is anchored by an ambiguous head count")
                head = anchored_heads[0]
                if head["checkpoint"]["sha256"] != decision.get("child_checkpoint", {}).get("sha256"):
                    raise RuntimeError("accepted local decision/head child checkpoint drift")
            elif anchored_heads:
                raise RuntimeError("rejected local decision unexpectedly has a promoted head")
            states.append({
                **job,
                "stage": "COMPLETE",
                "proposal": {**proposal, "artifact": proposal_identity},
                "decision": {**decision, "artifact": decision_identity},
                "anchored_head_record_sha256": (
                    None if not anchored_heads else anchored_heads[0]["record_sha256"]
                ),
            })
        return states

    def incomplete_job_states(self) -> list[Mapping[str, object]]:
        return [row for row in self.job_states() if row["stage"] != "COMPLETE"]

    def append_decision(self, decision: Mapping[str, object]) -> Mapping[str, object]:
        if decision.get("schema") != DECISION_SCHEMA:
            raise RuntimeError("local decision schema drift")
        return write_content_addressed_json(self.root / "decisions", ".decision.json", decision)

    def append_head(self, head: Mapping[str, object]) -> Mapping[str, object]:
        current = self.current_head()
        if head.get("schema") != HEAD_SCHEMA:
            raise RuntimeError("local promoted head schema drift")
        if int(head["generation"]) != int(current["generation"]) + 1:
            raise RuntimeError("local promoted generation is not next")
        if head["parent_head_record_sha256"] != current["record_sha256"] or head["parent_checkpoint_sha256"] != current["checkpoint"]["sha256"]:
            raise RuntimeError("local promoted head does not extend current head")
        return _write_head_atomic(self.root, int(head["generation"]), head)
