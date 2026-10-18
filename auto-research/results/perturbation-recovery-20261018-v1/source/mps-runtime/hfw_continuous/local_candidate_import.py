from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .common import (
    canonical_json_bytes, read_bytes_same_fd, read_json_same_fd,
    write_content_addressed_bytes, write_content_addressed_json,
)
from .local_ledger import LocalLedger


SOURCE_SCHEMA = "hfw_w560_mps_untrusted_candidate_chain_proposal_v1"
LOCAL_SCHEMA = "hfw_w560_mps_local_candidate_chain_v1"


def _copy_candidate_artifact(
    source: Mapping[str, object], directory: Path, suffix: str,
) -> Mapping[str, object]:
    payload, identity = read_bytes_same_fd(Path(source["path"]), str(source["sha256"]))
    if int(identity["size_bytes"]) != int(source["size_bytes"]):
        raise RuntimeError("source candidate artifact size drift")
    copied = write_content_addressed_bytes(directory, suffix, payload)
    if copied["sha256"] != source["sha256"] or int(copied["size_bytes"]) != int(source["size_bytes"]):
        raise RuntimeError("localized candidate artifact changed during copy")
    return copied


def import_completed_candidate_chain(
    ledger: LocalLedger, source_proposal_path: Path, source_terminal_path: Path,
    expected_source_proposal_sha256: str,
) -> Mapping[str, object]:
    current = ledger.current_head()
    source, source_identity = read_json_same_fd(
        source_proposal_path, expected_source_proposal_sha256,
    )
    if source.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError("source candidate-chain schema drift")
    if source.get("base_checkpoint_sha256") != current["checkpoint"]["sha256"]:
        raise RuntimeError("source candidate-chain base is not current local head")
    if source.get("token_role_opened") != "SEARCH" or source.get("roles_available_to_worker") != ["SEARCH"]:
        raise RuntimeError("source candidate-chain did not use SEARCH-only solver data")
    if source.get("new_unique_token_positions_after_initialization") != 0:
        raise RuntimeError("source candidate-chain opened new token positions")
    if source.get("worker_commit_authority") is not False or source.get("official_checkpoint_lineage_changed") is not False:
        raise RuntimeError("source candidate-chain claimed lineage authority")
    terminal, terminal_identity = read_json_same_fd(source_terminal_path)
    if terminal.get("status") != "PASS_UNTRUSTED_PROPOSAL_ONLY":
        raise RuntimeError("source candidate-chain terminal status drift")
    if terminal.get("proposal_artifact", {}).get("sha256") != source_identity["sha256"]:
        raise RuntimeError("source terminal/proposal SHA drift")
    candidates = source.get("ordered_candidates", [])
    if int(source.get("candidate_count", -1)) != len(candidates):
        raise RuntimeError("source candidate count drift")
    if [int(row.get("candidate_order", -1)) for row in candidates] != list(range(1, len(candidates) + 1)):
        raise RuntimeError("source candidate order is not complete and contiguous")
    destination = ledger.root / "proposal_imports" / source_identity["sha256"]
    if destination.exists():
        raise FileExistsError("source proposal was already imported")
    destination.mkdir(parents=True)
    localized = []
    for row in candidates:
        order = int(row["candidate_order"])
        artifact_root = destination / f"candidate-{order:04d}"
        exact = _copy_candidate_artifact(
            row["exact_candidate_group_flat"], artifact_root, ".group_flat.f32le",
        )
        cumulative = _copy_candidate_artifact(
            row["cumulative_delta"], artifact_root, ".cumulative_delta.f32le",
        )
        incremental = _copy_candidate_artifact(
            row["incremental_delta"], artifact_root, ".incremental_delta.f32le",
        )
        if exact["sha256"] != row["candidate_group_flat_sha256"]:
            raise RuntimeError("localized exact group-flat SHA drift")
        localized.append({
            "candidate_order": order,
            "accept_index": int(row["accept_index"]),
            "pair_index": int(row["pair_index"]),
            "exact_candidate_group_flat": exact,
            "cumulative_delta": cumulative,
            "incremental_delta": incremental,
            "candidate_group_flat_sha256": exact["sha256"],
            "delta_semantics": "cumulative_current_minus_immutable_job_base",
        })
    proposal = {
        "schema": LOCAL_SCHEMA,
        "status": "PASS_COMPLETE_CHAIN_LOCALIZED_AWAITING_DUPLICATED_LOCAL_VERIFY",
        "source_proposal_sha256": source_identity["sha256"],
        "source_terminal_receipt_sha256": terminal_identity["sha256"],
        "local_solver_job_record_sha256": None,
        "artifact_root_relative": (Path("proposal_imports") / source_identity["sha256"]).as_posix(),
        "base_head_record_sha256": current["record_sha256"],
        "base_generation": current["generation"],
        "base_checkpoint_sha256": current["checkpoint"]["sha256"],
        "group": source["group"],
        "seed": int(source["seed"]),
        "base_group_flat_sha256": source["base_group_flat_sha256"],
        "group_elements": int(source["group_elements"]),
        "solver_source_sha256": source["solver_source_sha256"],
        "solver_settings_sha256": source["solver_settings_sha256"],
        "search_bank_sha256": source["search_bank_sha256"],
        "candidate_count": len(localized),
        "ordered_candidates": localized,
        "every_saved_candidate_imported": True,
        "candidate_subset_selection_performed": False,
        "local_verify_opened_during_import": False,
        "held_out_generalization_claimed": False,
        "cloud_access": False,
        "training_or_backprop": False,
        "adam_or_any_optimizer": False,
    }
    artifact = write_content_addressed_json(
        ledger.root / "proposal_imports", ".local-proposal.json", proposal,
    )
    receipt = {
        "schema": "hfw_w560_mps_local_candidate_import_receipt_v1",
        "status": "PASS_ALL_SOURCE_CANDIDATES_LOCALIZED_CREATE_ONLY",
        "source_proposal_sha256": source_identity["sha256"],
        "localized_proposal": artifact,
        "candidate_count": len(localized),
        "every_saved_candidate_imported": True,
        "source_paths_required_after_import": False,
        "local_verify_opened": False,
    }
    receipt_artifact = write_content_addressed_json(
        destination, ".import-receipt.json", receipt,
    )
    return {**proposal, "proposal_artifact": artifact, "import_receipt": receipt_artifact}
