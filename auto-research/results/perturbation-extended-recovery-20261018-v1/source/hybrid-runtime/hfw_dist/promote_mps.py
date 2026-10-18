from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    canonical_json_bytes, exclusive_lock, fsync_directory, hash_file,
    hash_file_region, read_file_region, read_json,
    write_exclusive, write_json_exclusive,
)
from .ledger import DistributedLedger, HEAD_SCHEMA
from .model_io import build_loaded_parent, load_checkpoint_same_fd, validate_promoted_payload
from .daytona_manager import validate_recovered_task0_provenance


VENDOR_PACKAGE = "vendor.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver"


class MPSDuplicateVerifyObjective:
    def __init__(self, parent, batches):
        if not torch.backends.mps.is_available():
            raise RuntimeError("authoritative promoter requires local MPS")
        self.parent = parent
        self.device = torch.device("mps")
        self.batches = tuple(batch.detach().cpu().to(torch.long) for batch in batches)
        self.base_parameters = {
            name: value.detach() for name, value in parent.model.named_parameters()
        }
        self.buffers = {name: value.detach() for name, value in parent.model.named_buffers()}
        self.forward_calls = 0
        self.bfloat16_logits = 0
        self._flat_cache: dict[str, torch.Tensor] = {}

    def flat_group(self, group: str) -> torch.Tensor:
        if group not in self._flat_cache:
            self._flat_cache[group] = torch.cat([
                self.base_parameters[name].reshape(-1) for name in self.parent.groups[group]
            ]).detach()
        return self._flat_cache[group]

    def state_with_group(self, group: str, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        state = dict(self.base_parameters)
        cursor = 0
        for name in self.parent.groups[group]:
            reference = self.base_parameters[name]
            count = reference.numel()
            state[name] = flat[cursor:cursor + count].view_as(reference)
            cursor += count
        if cursor != flat.numel():
            raise RuntimeError("MPS VERIFY group vector length drift")
        state.update(self.buffers)
        return state

    def parent_state(self) -> dict[str, torch.Tensor]:
        return {**self.base_parameters, **self.buffers}

    def sequence_losses(self, state: Mapping[str, torch.Tensor]) -> list[float]:
        expected = set(self.base_parameters) | set(self.buffers)
        if set(state) != expected or any(value.requires_grad for value in state.values()):
            raise RuntimeError("MPS VERIFY state inventory/autograd drift")
        parameters = {name: state[name].to(self.device) for name in self.base_parameters}
        buffers = {name: state[name].to(self.device) for name in self.buffers}
        values = []
        for batch in self.batches:
            raw = batch.to(self.device)
            inputs, targets = raw[:, :-1], raw[:, 1:]
            with torch.inference_mode(), torch.autocast(
                device_type="mps", dtype=torch.bfloat16, enabled=True,
            ):
                output = torch.func.functional_call(
                    self.parent.model, (parameters, buffers), (inputs,), strict=True,
                )
            self.forward_calls += 1
            if output.logits.dtype != torch.bfloat16:
                raise RuntimeError("MPS VERIFY forward did not produce BF16 logits")
            self.bfloat16_logits += 1
            losses = F.cross_entropy(
                output.logits.float().reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1), reduction="none",
            ).view_as(targets).mean(dim=1)
            if not torch.isfinite(losses).all():
                raise RuntimeError("nonfinite MPS VERIFY sequence CE")
            values.extend(float(value) for value in losses.detach().cpu().tolist())
            del output, losses
        return values

    def runtime_receipt(self) -> Mapping[str, object]:
        if self.forward_calls != self.bfloat16_logits:
            raise RuntimeError("not every MPS VERIFY forward produced BF16 logits")
        return {
            "schema": "hfw_w560_distributed_mps_verify_runtime_v1",
            "device": "mps", "autocast_dtype": "bfloat16",
            "forward_calls": self.forward_calls,
            "bfloat16_logits_observed": self.bfloat16_logits,
            "inference_only": True, "optimizer_steps": 0, "autograd_calls": 0,
        }


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _load_vendor(frozen: Path):
    sys.path.insert(0, str(frozen))
    branch = importlib.import_module(f"{VENDOR_PACKAGE}.branch_search")
    bridge = importlib.import_module(f"{VENDOR_PACKAGE}.production_bridge")
    manifest = importlib.import_module(f"{VENDOR_PACKAGE}.manifest")
    return branch, bridge, manifest


def _find_decision(ledger: DistributedLedger, shortlist_sha256: str):
    rows = []
    for path in sorted((ledger.root / "decisions").glob("*.decision.json")):
        value, identity = read_json(path)
        if value.get("shortlist_sha256") == shortlist_sha256:
            rows.append(({**value, "record_sha256": identity["sha256"]}, identity))
    if len(rows) > 1:
        raise RuntimeError("multiple decisions exist for one shortlist")
    return rows[0] if rows else None


def _validated_shortlist(
    ledger: DistributedLedger, wave_root: Path, campaign: Mapping[str, object],
    campaign_identity: Mapping[str, object], collection_terminal: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    shortlist_path = Path(str(collection_terminal["shortlist"]["path"]))
    expected_shortlist_parent = (ledger.root / "shortlists").resolve()
    if shortlist_path.resolve().parent != expected_shortlist_parent:
        raise RuntimeError("shortlist is outside authoritative shortlist directory")
    shortlist, shortlist_identity = read_json(
        shortlist_path, str(collection_terminal["shortlist"]["sha256"]),
    )
    if shortlist_path.name != f"{shortlist_identity['sha256']}.shortlist.json":
        raise RuntimeError("shortlist filename/content address drift")
    collection_path = Path(str(collection_terminal["collection"]["path"]))
    if collection_path.resolve().parent != (ledger.root / "collections").resolve():
        raise RuntimeError("collection is outside authoritative collection directory")
    collection, collection_identity = read_json(
        collection_path, str(collection_terminal["collection"]["sha256"]),
    )
    if collection_path.name != f"{collection_identity['sha256']}.collection.json":
        raise RuntimeError("collection filename/content address drift")
    if (
        shortlist.get("campaign_sha256") != campaign_identity["sha256"]
        or collection.get("campaign_sha256") != campaign_identity["sha256"]
    ):
        raise RuntimeError("collection/shortlist campaign binding drift")
    candidates = shortlist.get("candidates", [])
    if (
        shortlist.get("schema") != "hfw_w560_daytona_all_accepted_candidates_v1"
        or shortlist.get("status")
        != "PASS_ALL_UNIQUE_ACCEPTED_CANDIDATES_SEALED"
        or not isinstance(candidates, list)
        or int(shortlist.get("candidate_count", -1)) != len(candidates)
        or shortlist.get("every_accepted_candidate_downloaded") is not True
        or shortlist.get("every_unique_accepted_candidate_selected_for_mps_verify")
        is not True
        or shortlist.get("cpu_ranking_filter_permitted") is not False
        or shortlist.get("cpu_ranking_filter_applied") is not False
    ):
        raise RuntimeError("all-accepted-candidate shortlist contract drift")
    if shortlist.get("selection_rule") != campaign["selection_rule"]:
        raise RuntimeError("shortlist selection rule differs from sealed campaign")
    worker_by_task = {int(row["task_index"]): row for row in collection["workers"]}
    if len(worker_by_task) != int(campaign["worker_count"]):
        raise RuntimeError("collection task inventory is incomplete or duplicated")
    task_by_index = {int(row["task_index"]): row for row in campaign["tasks"]}
    expected_candidates = []
    expected_duplicates = []
    keeper = {}
    for row in sorted(collection["workers"], key=lambda item: int(item["task_index"])):
        task_index = int(row["task_index"])
        accepted = row.get("accepted_candidates")
        if not isinstance(accepted, list):
            raise RuntimeError("collection accepted candidate inventory missing")
        for worker_order, accepted_candidate in enumerate(accepted, start=1):
            if accepted_candidate.get("candidate_order") != worker_order:
                raise RuntimeError("collection worker candidate order drift")
            group = str(row["group"])
            digest = str(accepted_candidate["group_flat_sha256"])
            key = (group, digest)
            if key in keeper:
                prior = keeper[key]
                expected_duplicates.append({
                    "group": group,
                    "duplicate_task_index": task_index,
                    "duplicate_worker_candidate_order": worker_order,
                    "duplicate_flat_sha256": digest,
                    "keeper_task_index": prior["task_index"],
                    "keeper_worker_candidate_order": prior["worker_candidate_order"],
                    "dedupe_key": "group_and_group_flat_sha256",
                })
                continue
            expected = {
                "task_index": task_index,
                "worker_candidate_order": worker_order,
                "group": group,
                "digest": digest,
                "row": row,
                "accepted": accepted_candidate,
            }
            keeper[key] = expected
            expected_candidates.append(expected)
    if shortlist.get("skipped_exact_flat_duplicates") != expected_duplicates:
        raise RuntimeError("shortlist duplicate audit differs from authoritative recomputation")
    seen_group_shas = set()
    for order, (candidate, expected) in enumerate(zip(candidates, expected_candidates), start=1):
        group = expected["group"]
        collection_row = expected["row"]
        accepted_candidate = expected["accepted"]
        task_index = int(candidate["task_index"])
        if (
            candidate.get("candidate_order") != order
            or candidate.get("group") != group
            or task_index != expected["task_index"]
            or candidate.get("worker_candidate_order")
            != expected["worker_candidate_order"]
            or candidate.get("accept_index") != accepted_candidate["accept_index"]
            or candidate.get("pair_index") != accepted_candidate["pair_index"]
        ):
            raise RuntimeError("shortlist deterministic candidate order drift")
        flat_sha = str(candidate["candidate_group_flat_sha256"])
        if (group, flat_sha) in seen_group_shas:
            raise RuntimeError("shortlist exact flat duplicated within group")
        seen_group_shas.add((group, flat_sha))
        task = task_by_index.get(task_index)
        if task is None or task["group"] != group or int(task["seed"]) != int(candidate["seed"]):
            raise RuntimeError("shortlist candidate is not a sealed campaign task")
        if (
            flat_sha != expected["digest"]
            or candidate.get("accepted_candidate_bundle")
            != collection_row.get("accepted_candidate_bundle")
            or candidate.get("offset_bytes") != accepted_candidate["offset_bytes"]
            or candidate.get("size_bytes") != accepted_candidate["size_bytes"]
            or candidate.get("cpu_search_result")
            != collection_row.get("cpu_search_result")
            or candidate.get("task_sha256") != collection_row.get("task_sha256")
            or candidate.get("seed") != collection_row.get("seed")
            or candidate.get("worker_id") != collection_row.get("worker_id")
            or candidate.get("cpu_score_used_for_selection") is not False
        ):
            raise RuntimeError("shortlist candidate differs from collection")
        bundle_path = Path(str(candidate["accepted_candidate_bundle"]["path"]))
        try:
            bundle_path.resolve().relative_to((wave_root / "collected").resolve())
        except ValueError as error:
            raise RuntimeError("shortlisted bundle escaped collected wave root") from error
        hash_file(bundle_path, str(candidate["accepted_candidate_bundle"]["sha256"]))
        hash_file_region(
            bundle_path,
            int(candidate["offset_bytes"]),
            int(candidate["size_bytes"]),
            flat_sha,
        )
    if len(candidates) != len(expected_candidates):
        raise RuntimeError("shortlist omitted an accepted unique candidate")
    return shortlist, shortlist_identity, collection


def _ensure_link(source: Path, destination: Path, sha256: str) -> Mapping[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return hash_file(destination, sha256)
    os.link(source, destination)
    os.chmod(destination, 0o444)
    return hash_file(destination, sha256)


def _write_checkpoint_atomic(
    final_path: Path, payload: Mapping[str, object], bridge,
) -> Mapping[str, object]:
    if final_path.exists():
        identity = hash_file(final_path)
        existing = load_checkpoint_same_fd(final_path, str(identity["sha256"]))
        validate_promoted_payload(existing, str(payload["model_state_digest"]), bridge)
        if (
            existing["accepted_proposal_sha256"] != payload["accepted_proposal_sha256"]
            or existing["replacement_group_flat_sha256"] != payload["replacement_group_flat_sha256"]
        ):
            raise RuntimeError("existing promoted checkpoint is not this transaction")
        return identity
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging = final_path.parent / f".{final_path.name}.{os.getpid()}.{time.time_ns()}.staged"
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, final_path)
        fsync_directory(final_path.parent)
    finally:
        staging.unlink(missing_ok=True)
        fsync_directory(final_path.parent)
    return hash_file(final_path)


def _validate_accepted_decision_artifacts(
    ledger: DistributedLedger, frozen: Path, decision: Mapping[str, object],
    head: Mapping[str, object], shortlist_sha256: str,
) -> None:
    child = decision["child_checkpoint"]
    child_path = Path(str(child["path"]))
    if child_path.resolve().parent != (ledger.root / "promoted_checkpoints").resolve():
        raise RuntimeError("accepted child checkpoint escaped authoritative directory")
    child_identity = hash_file(child_path, str(child["sha256"]))
    if int(child_identity["size_bytes"]) != int(child["size_bytes"]):
        raise RuntimeError("accepted child checkpoint size drift")
    branch, bridge, vendor_manifest = _load_vendor(frozen)
    payload = load_checkpoint_same_fd(child_path, str(child["sha256"]))
    validate_promoted_payload(payload, str(decision["child_model_state_digest"]), bridge)
    selected = decision["selected_candidate"]
    delta = decision["ledger_delta"]
    replacement = decision["ledger_replacement_group_flat"]
    expected = {
        "origin_checkpoint_sha256": head["origin_checkpoint_sha256"],
        "parent_checkpoint_sha256": head["checkpoint"]["sha256"],
        "generation": int(head["model_generation"]) + 1,
        "accepted_proposal_sha256": shortlist_sha256,
        "selected_candidate_order": int(selected["candidate_order"]),
        "selected_candidate_delta_sha256": delta["sha256"],
        "solver_cumulative_delta_sha256": delta["sha256"],
        "replacement_group_flat_sha256": replacement["sha256"],
        "model_state_digest": decision["child_model_state_digest"],
        "optimizer_present": False,
        "scheduler_present": False,
        "training_resume_authorized": False,
        "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
        "solver_cumulative_delta_used_to_reconstruct_group": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"accepted decision checkpoint transaction drift: {key}")
    for record, directory in (
        (delta, "deltas"), (replacement, "replacement_group_flats"),
    ):
        path = Path(str(record["path"]))
        if path.resolve().parent != (ledger.root / directory).resolve():
            raise RuntimeError(f"accepted decision artifact escaped {directory}")
        identity = hash_file(path, str(record["sha256"]))
        if int(identity["size_bytes"]) != int(record["size_bytes"]):
            raise RuntimeError("accepted decision sidecar size drift")
    loaded, _ = build_loaded_parent(
        frozen, child_path, str(child["sha256"]),
        str(decision["child_model_state_digest"]), torch.device("cpu"),
        bridge, vendor_manifest,
    )
    group = str(selected["group"])
    exact = torch.cat([
        dict(loaded.model.named_parameters())[name].reshape(-1)
        for name in loaded.groups[group]
    ])
    if hashlib.sha256(branch._tensor_bytes(exact)).hexdigest() != replacement["sha256"]:
        raise RuntimeError("accepted checkpoint selected group differs from exact replacement")


def _terminal_path(wave_root: Path) -> Path:
    return wave_root / "PROMOTION_COMPLETE.json"


def _write_terminal_once(wave_root: Path, value: Mapping[str, object]) -> Mapping[str, object]:
    path = _terminal_path(wave_root)
    if path.exists():
        existing, identity = read_json(path)
        if existing != value:
            raise RuntimeError("promotion terminal already exists with different content")
        return {**existing, "terminal_identity": identity}
    identity = write_json_exclusive(path, value)
    return {**value, "terminal_identity": identity}


def promote_wave(package_root: Path, wave_root: Path) -> Mapping[str, object]:
    package_root = Path(package_root).resolve()
    wave_root = Path(wave_root).resolve()
    if _terminal_path(wave_root).exists():
        value, identity = read_json(_terminal_path(wave_root))
        return {**value, "terminal_identity": identity}
    campaign, campaign_identity = read_json(wave_root / "snapshot/CAMPAIGN.json")
    recovery = validate_recovered_task0_provenance(
        package_root, wave_index=int(campaign["wave_index"]),
    )
    collection_terminal, _ = read_json(wave_root / "COLLECTION_COMPLETE.json")
    ledger = DistributedLedger(package_root / "runtime/authority/lineage")
    shortlist, shortlist_identity, collection = _validated_shortlist(
        ledger, wave_root, campaign, campaign_identity, collection_terminal,
    )
    frozen = package_root / "runtime/frozen"
    contract, _ = ledger.contract()
    host_mps_lock_path = Path(str(contract["host_mps_lock_path"]))
    if str(campaign["promotion"]["host_mps_lock_path"]) != str(host_mps_lock_path):
        raise RuntimeError("campaign/authority host MPS lock binding drift")
    expected_winner_order = [
        "passes_both_duplicated_relative_thresholds",
        "mps_verify_replicated_mean_ascending",
        "shortlist_candidate_order_ascending",
    ]
    if campaign["promotion"].get("winner_ordering") != expected_winner_order:
        raise RuntimeError("MPS winner ordering was not sealed as required")
    with exclusive_lock(host_mps_lock_path), ledger.lock():
        if recovery is not None:
            validate_recovered_task0_provenance(
                package_root, wave_index=int(campaign["wave_index"]),
            )
        head = ledger.recover_current_pointer_locked()
        existing_decision = _find_decision(ledger, str(shortlist_identity["sha256"]))
        if existing_decision is not None:
            decision, decision_identity = existing_decision
            if head.get("decision_sha256") == decision_identity["sha256"]:
                return _write_terminal_once(wave_root, {
                    "schema": "hfw_w560_distributed_promotion_terminal_v1",
                    "status": decision["status"],
                    "accepted": bool(decision["accepted"]),
                    "campaign_sha256": campaign_identity["sha256"],
                    "shortlist_sha256": shortlist_identity["sha256"],
                    "decision": decision_identity,
                    "current_head_record_sha256": head["record_sha256"],
                    "recovered_after_head_append": True,
                })
            if not bool(decision["accepted"]):
                return _write_terminal_once(wave_root, {
                    "schema": "hfw_w560_distributed_promotion_terminal_v1",
                    "status": decision["status"], "accepted": False,
                    "campaign_sha256": campaign_identity["sha256"],
                    "shortlist_sha256": shortlist_identity["sha256"],
                    "decision": decision_identity,
                    "current_head_record_sha256": head["record_sha256"],
                    "recovered_after_head_append": False,
                })
            if head["record_sha256"] != campaign["base_head_record_sha256"]:
                return _write_terminal_once(wave_root, {
                    "schema": "hfw_w560_distributed_promotion_terminal_v1",
                    "status": "REJECTED_ORPHANED_ACCEPTED_DECISION_HEAD_ADVANCED_BY_OTHER_TRANSACTION",
                    "accepted": False,
                    "campaign_sha256": campaign_identity["sha256"],
                    "shortlist_sha256": shortlist_identity["sha256"],
                    "decision": decision_identity,
                    "current_head_record_sha256": head["record_sha256"],
                    "recovered_after_head_append": False,
                })
        if head["record_sha256"] != campaign["base_head_record_sha256"]:
            if existing_decision is None:
                stale = {
                    "schema": "hfw_w560_distributed_mps_verify_decision_v1",
                    "status": "REJECTED_STALE_HEAD_NO_VERIFY_NO_MERGE_NO_REBASE",
                    "accepted": False,
                    "campaign_sha256": campaign_identity["sha256"],
                    "shortlist_sha256": shortlist_identity["sha256"],
                    "base_head_record_sha256": campaign["base_head_record_sha256"],
                    "observed_head_record_sha256": head["record_sha256"],
                    "mps_verify_opened": False,
                    "checkpoint_materialized": False,
                }
                identity = ledger.append_named_record("decisions", ".decision.json", stale)
            else:
                stale, identity = existing_decision
            return _write_terminal_once(wave_root, {
                "schema": "hfw_w560_distributed_promotion_terminal_v1",
                "status": stale["status"], "accepted": False,
                "campaign_sha256": campaign_identity["sha256"],
                "shortlist_sha256": shortlist_identity["sha256"],
                "decision": identity,
                "current_head_record_sha256": head["record_sha256"],
                "recovered_after_head_append": False,
            })
        if existing_decision is None:
            intent = {
                "schema": "hfw_w560_distributed_promotion_intent_v1",
                "status": "SEALED_BEFORE_MPS_VERIFY",
                "campaign_sha256": campaign_identity["sha256"],
                "shortlist_sha256": shortlist_identity["sha256"],
                "base_head_record_sha256": head["record_sha256"],
                "base_checkpoint_sha256": head["checkpoint"]["sha256"],
                "candidate_count": int(shortlist["candidate_count"]),
                "mixed_groups_independent_against_same_base": True,
                "at_most_one_head_append": True,
            }
            intent_identity = ledger.append_named_record(
                "promotion_intents", ".intent.json", intent,
            )
            branch, bridge, vendor_manifest = _load_vendor(frozen)
            loaded, _ = build_loaded_parent(
                frozen, Path(str(head["checkpoint"]["path"])),
                str(head["checkpoint"]["sha256"]), str(head["model_state_digest"]),
                torch.device("mps"), bridge, vendor_manifest,
            )
            verify_shape = tuple(int(value) for value in campaign["promotion"]["verify_bank_shape"])
            verify_identity = hash_file(
                frozen / "verify.u32le", str(campaign["promotion"]["verify_bank_sha256"]),
            )
            expected_verify_bytes = math.prod(verify_shape) * np.dtype("<u4").itemsize
            if int(verify_identity["size_bytes"]) != expected_verify_bytes:
                raise RuntimeError("VERIFY shape/byte-size binding drift")
            verify = np.memmap(frozen / "verify.u32le", mode="r", dtype="<u4", shape=verify_shape)
            batches = [torch.from_numpy(np.array(verify[index], dtype=np.int64, copy=True))
                       for index in range(verify_shape[0])]
            objective = MPSDuplicateVerifyObjective(loaded, batches)
            parent_state = objective.parent_state()
            parent_first = objective.sequence_losses(parent_state)
            parent_second = objective.sequence_losses(parent_state)
            tolerance = float(campaign["promotion"]["duplicate_forward_tolerance"])
            parent_duplicate = max(abs(a - b) for a, b in zip(parent_first, parent_second))
            if parent_duplicate > tolerance:
                raise RuntimeError("duplicated parent MPS VERIFY drift")
            parent_means = [_mean(parent_first), _mean(parent_second)]
            minimum_relative = float(campaign["promotion"]["minimum_relative_improvement"])
            thresholds = [value - abs(value) * minimum_relative for value in parent_means]
            candidate_rows = []
            selected = None
            selected_flat_cpu = None
            selected_flat_payload = None
            for candidate_order, candidate in enumerate(shortlist["candidates"], start=1):
                candidate_path = Path(str(candidate["accepted_candidate_bundle"]["path"]))
                try:
                    candidate_path.resolve().relative_to((wave_root / "collected").resolve())
                except ValueError as error:
                    raise RuntimeError("shortlisted bundle escaped collected wave root") from error
                payload, identity = read_file_region(
                    candidate_path,
                    int(candidate["offset_bytes"]),
                    int(candidate["size_bytes"]),
                    str(candidate["candidate_group_flat_sha256"]),
                )
                group = str(candidate["group"])
                expected_elements = int(campaign["group_elements"][group])
                if int(identity["size_bytes"]) != expected_elements * 4:
                    raise RuntimeError("shortlisted flat size drift")
                values = np.frombuffer(payload, dtype="<f4")
                if values.size != expected_elements or not np.isfinite(values).all():
                    raise RuntimeError("shortlisted flat element/finiteness drift")
                candidate_flat_cpu = torch.from_numpy(values.copy())
                candidate_flat_mps = candidate_flat_cpu.to("mps")
                candidate_state = objective.state_with_group(group, candidate_flat_mps)
                first = objective.sequence_losses(candidate_state)
                second = objective.sequence_losses(candidate_state)
                duplicate = max(abs(a - b) for a, b in zip(first, second))
                means = [_mean(first), _mean(second)]
                passes = (
                    duplicate <= tolerance
                    and means[0] < thresholds[0]
                    and means[1] < thresholds[1]
                )
                row = {
                    "candidate_order": candidate_order,
                    "task_index": int(candidate["task_index"]),
                    "group": group, "seed": int(candidate["seed"]),
                    "candidate_group_flat_sha256": identity["sha256"],
                    "cpu_candidate_full_search_mean_ce": candidate[
                        "candidate_cpu_full_search_mean_ce"
                    ],
                    "cpu_score_used_for_selection": False,
                    "mps_verify_per_sequence_first": first,
                    "mps_verify_per_sequence_second": second,
                    "mps_verify_mean_first": means[0],
                    "mps_verify_mean_second": means[1],
                    "mps_verify_replicated_mean": math.fsum(means) / 2,
                    "duplicate_max_abs_sequence_difference": duplicate,
                    "passes_minimum_replicated_improvement": passes,
                    "independent_replacement_against_same_immutable_base": True,
                    "composed_with_other_shortlist_groups": False,
                }
                candidate_rows.append(row)
                if passes and (
                    selected is None
                    or (row["mps_verify_replicated_mean"], row["candidate_order"])
                    < (selected["mps_verify_replicated_mean"], selected["candidate_order"])
                ):
                    selected = row
                    selected_flat_cpu = candidate_flat_cpu
                    selected_flat_payload = payload
            accepted = selected is not None
            child_identity = None
            delta_identity = None
            replacement_identity = None
            model_digest = None
            if accepted:
                group = str(selected["group"])
                base_cpu = objective.flat_group(group).detach().cpu().float()
                if hashlib.sha256(branch._tensor_bytes(selected_flat_cpu)).hexdigest() != selected[
                    "candidate_group_flat_sha256"
                ]:
                    raise RuntimeError("selected CPU flat serialization drift")
                delta_payload = branch._tensor_bytes(selected_flat_cpu - base_cpu)
                delta_sha = hashlib.sha256(delta_payload).hexdigest()
                delta_path = ledger.root / "deltas" / f"{delta_sha}.group_delta.f32le"
                if delta_path.exists():
                    delta_identity = hash_file(delta_path, delta_sha)
                else:
                    delta_identity = write_exclusive(delta_path, delta_payload)
                replacement_path = ledger.root / "replacement_group_flats" / (
                    f"{selected['candidate_group_flat_sha256']}.group_flat.f32le"
                )
                if replacement_path.exists():
                    replacement_identity = hash_file(
                        replacement_path, str(selected["candidate_group_flat_sha256"]),
                    )
                else:
                    replacement_identity = write_exclusive(
                        replacement_path, selected_flat_payload,
                    )
                selected_state_mps = objective.state_with_group(
                    group, selected_flat_cpu.to("mps"),
                )
                selected_state_cpu = {
                    name: value.detach().cpu() for name, value in selected_state_mps.items()
                }
                exact_roundtrip = torch.cat([
                    selected_state_cpu[name].reshape(-1) for name in loaded.groups[group]
                ])
                if hashlib.sha256(branch._tensor_bytes(exact_roundtrip)).hexdigest() != replacement_identity["sha256"]:
                    raise RuntimeError("checkpoint exact replacement flat roundtrip drift")
                model_digest = bridge.model_state_digest(selected_state_cpu)
                child_generation = int(head["model_generation"]) + 1
                checkpoint_payload = {
                    "schema": "hfw_w560_continuous_promoted_checkpoint_v1",
                    "origin_checkpoint_sha256": head["origin_checkpoint_sha256"],
                    "parent_checkpoint_sha256": head["checkpoint"]["sha256"],
                    "generation": child_generation,
                    "accepted_proposal_sha256": shortlist_identity["sha256"],
                    "selected_candidate_order": int(selected["candidate_order"]),
                    "selected_candidate_delta_sha256": delta_identity["sha256"],
                    "solver_cumulative_delta_sha256": delta_identity["sha256"],
                    "replacement_group_flat_sha256": replacement_identity["sha256"],
                    "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
                    "solver_cumulative_delta_used_to_reconstruct_group": False,
                    "model_state_digest": model_digest,
                    "optimizer_present": False, "scheduler_present": False,
                    "training_resume_authorized": False,
                    "model": selected_state_cpu,
                }
                checkpoint_path = ledger.root / "promoted_checkpoints" / (
                    f"generation-{child_generation:08d}-{shortlist_identity['sha256'][:16]}.pt"
                )
                child_identity = _write_checkpoint_atomic(checkpoint_path, checkpoint_payload, bridge)
                transaction_fields = {
                    key: checkpoint_payload[key] for key in (
                        "schema", "origin_checkpoint_sha256", "parent_checkpoint_sha256",
                        "generation", "accepted_proposal_sha256", "selected_candidate_order",
                        "selected_candidate_delta_sha256", "solver_cumulative_delta_sha256",
                        "replacement_group_flat_sha256",
                        "group_replacement_materialization_source",
                        "solver_cumulative_delta_used_to_reconstruct_group", "model_state_digest",
                        "optimizer_present", "scheduler_present", "training_resume_authorized",
                    )
                }
                del checkpoint_payload, selected_state_cpu, selected_state_mps, exact_roundtrip
                reloaded = load_checkpoint_same_fd(
                    Path(str(child_identity["path"])), str(child_identity["sha256"]),
                )
                validate_promoted_payload(reloaded, str(model_digest), bridge)
                for key, expected in transaction_fields.items():
                    if reloaded.get(key) != expected:
                        raise RuntimeError(f"new checkpoint same-FD transaction reload drift: {key}")
                reloaded_roundtrip = torch.cat([
                    reloaded["model"][name].reshape(-1) for name in loaded.groups[group]
                ])
                if hashlib.sha256(branch._tensor_bytes(reloaded_roundtrip)).hexdigest() != replacement_identity["sha256"]:
                    raise RuntimeError("new checkpoint same-FD selected group roundtrip drift")
                if bridge.model_state_digest(reloaded["model"]) != model_digest:
                    raise RuntimeError("new checkpoint same-FD model digest drift")
                del reloaded, reloaded_roundtrip
            decision = {
                "schema": "hfw_w560_distributed_mps_verify_decision_v1",
                "status": (
                    "ACCEPTED_BEST_INDEPENDENT_MIXED_GROUP_CANDIDATE"
                    if accepted else "REJECTED_NO_CANDIDATE_PASSED_DUPLICATED_MPS_VERIFY"
                ),
                "accepted": accepted,
                "campaign_sha256": campaign_identity["sha256"],
                "shortlist_sha256": shortlist_identity["sha256"],
                "promotion_intent_sha256": intent_identity["sha256"],
                "base_head_record_sha256": head["record_sha256"],
                "base_checkpoint_sha256": head["checkpoint"]["sha256"],
                "local_verify": {
                    "role": "VERIFY",
                    "disclosure": "PERMANENT_REUSED_SELECTION_DATA_NOT_HELD_OUT",
                    "held_out_generalization_claimed": False,
                    "bank_sha256": verify_identity["sha256"],
                    "parent_per_sequence_first": parent_first,
                    "parent_per_sequence_second": parent_second,
                    "parent_mean_first": parent_means[0],
                    "parent_mean_second": parent_means[1],
                    "parent_duplicate_max_abs_sequence_difference": parent_duplicate,
                    "thresholds": thresholds,
                    "minimum_relative_improvement": minimum_relative,
                    "duplicate_forward_tolerance": tolerance,
                    "candidate_results": candidate_rows,
                    "all_shortlisted_candidates_evaluated_twice": True,
                    "mixed_groups_evaluated_as_independent_replacements": True,
                    "cross_group_composition_performed": False,
                    "runtime": objective.runtime_receipt(),
                },
                "selected_candidate": selected,
                "child_checkpoint": child_identity,
                "child_model_state_digest": model_digest,
                "ledger_delta": delta_identity,
                "ledger_replacement_group_flat": replacement_identity,
                "checkpoint_materialized_from_exact_group_flat": accepted,
                "training_or_backprop": False, "optimizer_steps": 0,
            }
            decision_identity = ledger.append_named_record(
                "decisions", ".decision.json", decision,
            )
        else:
            decision, decision_identity = existing_decision
            accepted = bool(decision["accepted"])
        head_after = head
        if accepted:
            if ledger.current_head()["record_sha256"] != campaign["base_head_record_sha256"]:
                raise RuntimeError("head changed inside serialized promoter lock")
            _validate_accepted_decision_artifacts(
                ledger, frozen, decision, head, str(shortlist_identity["sha256"]),
            )
            selected = decision["selected_candidate"]
            child = decision["child_checkpoint"]
            new_head = {
                "schema": HEAD_SCHEMA,
                "status": "PROMOTED_EXACT_MPS_VERIFIED_CLOUD_PROPOSAL",
                "authority_generation": int(head["authority_generation"]) + 1,
                "model_generation": int(head["model_generation"]) + 1,
                "local_promotions": int(head["local_promotions"]) + 1,
                "parent_head_record_sha256": head["record_sha256"],
                "contract_sha256": head["contract_sha256"],
                "source_head_record_sha256": head["source_head_record_sha256"],
                "source_head_record_path": head["source_head_record_path"],
                "parent_checkpoint_sha256": head["checkpoint"]["sha256"],
                "checkpoint": {**child, "format": "hfw_w560_continuous_promoted_checkpoint_v1"},
                "origin_checkpoint_sha256": head["origin_checkpoint_sha256"],
                "model_state_digest": decision["child_model_state_digest"],
                "accepted_proposal_sha256": shortlist_identity["sha256"],
                "selected_candidate_order": int(selected["candidate_order"]),
                "group": selected["group"],
                "decision_sha256": decision_identity["sha256"],
                "replacement_group_flat_sha256": decision["ledger_replacement_group_flat"]["sha256"],
                "solver_cumulative_delta_sha256": decision["ledger_delta"]["sha256"],
                "campaign_sha256": campaign_identity["sha256"],
                "shortlist_sha256": shortlist_identity["sha256"],
            }
            head_after = ledger.append_head(new_head)
        terminal = {
            "schema": "hfw_w560_distributed_promotion_terminal_v1",
            "status": decision["status"], "accepted": bool(decision["accepted"]),
            "campaign_sha256": campaign_identity["sha256"],
            "shortlist_sha256": shortlist_identity["sha256"],
            "decision": decision_identity,
            "current_head_record_sha256": head_after["record_sha256"],
            "recovered_after_head_append": False,
        }
        return _write_terminal_once(wave_root, terminal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    args = parser.parse_args()
    result = promote_wave(args.package_root, args.wave_root)
    print(json.dumps({
        "status": result["status"], "accepted": result["accepted"],
        "current_head_record_sha256": result["current_head_record_sha256"],
        "decision_sha256": result["decision"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
