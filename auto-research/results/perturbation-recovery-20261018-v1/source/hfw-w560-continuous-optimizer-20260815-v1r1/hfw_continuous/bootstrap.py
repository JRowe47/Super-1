from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch

from .common import read_json_same_fd
from .ledger import initialize_ledger
from .release_binding import validate_runtime_manifest
from .relocation import derive_relocated_protocol, vendor_modules
from .state import flatten_losses, state_for_objective
from .token_commitment import load_bank, validate_token_manifest_against_config


def bootstrap_ledger(
    config_path: Path, token_manifest_path: Path, ledger_root: Path,
) -> Mapping[str, object]:
    config, config_identity = read_json_same_fd(config_path)
    validate_runtime_manifest(config_path, config)
    protocol, prepared = derive_relocated_protocol(config_path, config=config)
    _, _, bridge, _, _ = vendor_modules(config)
    device = torch.device(config["controller"]["forward_device"])
    loaded = bridge.load_completed_w560_parent(protocol, Path(config["source"]["root"]), device)
    _, token_identity = read_json_same_fd(token_manifest_path)
    commit_values, token_manifest = load_bank(
        token_manifest_path, "COMMIT", expected_manifest_sha256=token_identity["sha256"]
    )
    search_values, search_manifest = load_bank(
        token_manifest_path, "SEARCH", expected_manifest_sha256=token_identity["sha256"]
    )
    if search_manifest != token_manifest:
        raise RuntimeError("SEARCH and COMMIT opened from different token manifests")
    token_contract = validate_token_manifest_against_config(config, token_manifest)
    if token_manifest["runtime_source_binding"]["manifest_file"]["sha256"] != prepared["runtime_source_binding"]["manifest_file"]["sha256"]:
        raise RuntimeError("prepared parent/token runtime manifest file binding drift")
    if token_manifest["runtime_source_binding"]["manifest_identity_sha256"] != prepared["runtime_source_binding"]["manifest_identity_sha256"]:
        raise RuntimeError("prepared parent/token runtime manifest content identity drift")
    if token_manifest["banks"]["COMMIT"]["provenance"].get("reclassified_role") != (
        "PERMANENT_REUSABLE_COMMIT_OBJECTIVE_NOT_HELD_OUT"
    ):
        raise RuntimeError("COMMIT bank is not the explicitly reclassified published fixed512 bank")
    batches = [torch.from_numpy(batch.astype("int64", copy=True)) for batch in commit_values]
    objective = bridge.W560ForwardOnlyObjective(loaded, batches, device)
    state = state_for_objective(objective)
    first = flatten_losses(objective, state, len(batches))
    second = flatten_losses(objective, state, len(batches))
    tolerance = float(config["controller"]["duplicate_forward_tolerance"])
    duplicate_max = max(abs(left - right) for left, right in zip(first, second))
    if duplicate_max > tolerance:
        raise RuntimeError("parent COMMIT duplicate evaluation drift")
    first_mean = math.fsum(first) / len(first)
    second_mean = math.fsum(second) / len(second)
    expected = float(config["controller"]["expected_parent_commit_mean_ce"])
    reproduction_tolerance = float(config["controller"]["parent_commit_reproduction_tolerance"])
    if max(abs(first_mean - expected), abs(second_mean - expected)) > reproduction_tolerance:
        raise RuntimeError(
            f"parent fixed512 baseline did not reproduce: {first_mean}, {second_mean} != {expected}"
        )
    baseline = {
        "role": "PERMANENT_REUSABLE_COMMIT_OBJECTIVE_NOT_HELD_OUT",
        "parent_expected_mean_ce": expected,
        "duplicate_mean_ce": [first_mean, second_mean],
        "duplicate_max_abs_sequence_difference": duplicate_max,
        "per_sequence_ce_first": first,
        "per_sequence_ce_second": second,
        "sequences": len(first),
        "target_tokens_per_sequence": int(config["token_policy"]["context"]),
        "passed_expected_parent_reproduction": True,
    }
    if token_manifest["config_sha256"] != config_identity["sha256"]:
        raise RuntimeError("token commitment was not generated from this controller config")
    result = initialize_ledger(
        ledger_root,
        config_identity=config_identity,
        token_manifest_identity=token_identity,
        prepared_parent=prepared,
        baseline_evidence=baseline,
        origin=config["origin"],
        runtime_source_binding=prepared["runtime_source_binding"],
    )
    return {
        "schema": "hfw_w560_continuous_bootstrap_receipt_v1",
        "status": "PASS_LEDGER_INITIALIZED_NO_PROPOSALS_STARTED",
        "baseline": baseline,
        "token_contract": token_contract,
        "prepared_parent": prepared,
        "generation_zero_head": result,
        "final_bank_present": False,
    }
