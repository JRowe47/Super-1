#!/usr/bin/env python3
"""Canonical forward-only line, tensor-block, and scalar descent along a verified delta."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_scales(value: str) -> list[float]:
    scales = [float(item) for item in value.split(",")]
    if not scales or any(not math.isfinite(item) for item in scales):
        raise argparse.ArgumentTypeError("scales must be finite comma-separated numbers")
    return scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--winner-candidate-order", type=int, required=True)
    parser.add_argument(
        "--alpha-scales",
        type=parse_scales,
        default=parse_scales("0.5,0.75,1,1.125,1.25,1.5,2"),
    )
    parser.add_argument(
        "--block-scales",
        type=parse_scales,
        default=parse_scales("0.125,0.25,0.5"),
    )
    parser.add_argument(
        "--scalar-scales",
        type=parse_scales,
        default=parse_scales("0.125,0.25,0.5"),
    )
    parser.add_argument(
        "--scalar-top-k",
        type=int,
        default=8,
        help="Test this many largest-magnitude nonzero coordinates from the exact delta.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        help="Operator checkpoint boundary: evaluate only this many leading tensor blocks.",
    )
    args = parser.parse_args()
    if args.max_blocks is not None and args.max_blocks <= 0:
        raise RuntimeError("max-blocks must be positive")
    if args.scalar_top_k < 0:
        raise RuntimeError("scalar-top-k must be nonnegative")
    if any(scale <= 0 for scale in args.scalar_scales):
        raise RuntimeError("scalar-scales must be positive")

    package_root = args.package_root.resolve()
    wave_root = args.wave_root.resolve()
    verification_root = args.verification_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = package_root / "runtime"
    sys.path.insert(0, str(runtime_root))

    from hfw_dist.common import (
        exclusive_lock, hash_file, read_file_region, read_json, write_exclusive,
    )
    from hfw_dist.ledger import DistributedLedger, HEAD_SCHEMA
    from hfw_dist.model_io import build_loaded_parent
    from hfw_dist.promote_mps import (
        MPSDuplicateVerifyObjective,
        _load_vendor,
        _mean,
        _validate_accepted_decision_artifacts,
        _write_checkpoint_atomic,
    )

    if not torch.backends.mps.is_available():
        raise RuntimeError("canonical MPS is unavailable")

    campaign, campaign_identity = read_json(wave_root / "snapshot/CAMPAIGN.json")
    parent_record, _ = read_json(verification_root / "PARENT.json")
    result_paths = sorted(
        (verification_root / "candidate-results").glob(
            f"{args.winner_candidate_order:06d}-*.json"
        )
    )
    if len(result_paths) != 1:
        raise RuntimeError("winner result artifact is not unique")
    winner, winner_identity = read_json(result_paths[0])
    if (
        winner.get("passes_minimum_replicated_improvement") is not True
        or float(winner["duplicate_max_abs_sequence_difference"]) != 0.0
    ):
        raise RuntimeError("selected winner lacks exact duplicated strict improvement")
    group = str(winner["group"])
    winner_payload, winner_region = read_file_region(
        Path(str(winner["bundle_path"])),
        int(winner["offset_bytes"]),
        int(winner["size_bytes"]),
        str(winner["candidate_group_flat_sha256"]),
    )
    winner_flat_cpu = torch.from_numpy(
        np.frombuffer(winner_payload, dtype="<f4").copy()
    )

    frozen = package_root / "runtime/frozen"
    contract, _ = read_json(package_root / "runtime/authority/lineage/contract.json")
    ledger = DistributedLedger(package_root / "runtime/authority/lineage")
    lock_path = Path(str(contract["host_mps_lock_path"]))
    tolerance = float(campaign["promotion"]["duplicate_forward_tolerance"])

    with ExitStack() as stack:
        stack.enter_context(exclusive_lock(lock_path))
        stack.enter_context(ledger.lock())
        head = ledger.current_head()
        if int(head["model_generation"]) < 10:
            raise RuntimeError("winner has not been promoted before directional descent")
        if head.get("replacement_group_flat_sha256") != winner["candidate_group_flat_sha256"]:
            raise RuntimeError("current head is not the selected winner")

        source_head_sha = str(parent_record["base_head_record_sha256"])
        source_head_paths = list((ledger.root / "heads").glob(f"*-{source_head_sha}.head.json"))
        if len(source_head_paths) != 1:
            raise RuntimeError("winner source head is not unique")
        source_head, _ = read_json(source_head_paths[0], source_head_sha)

        branch, bridge, vendor_manifest = _load_vendor(frozen)
        loaded_source, _ = build_loaded_parent(
            frozen,
            Path(str(source_head["checkpoint"]["path"])),
            str(source_head["checkpoint"]["sha256"]),
            str(source_head["model_state_digest"]),
            torch.device("mps"),
            bridge,
            vendor_manifest,
        )
        loaded_current, _ = build_loaded_parent(
            frozen,
            Path(str(head["checkpoint"]["path"])),
            str(head["checkpoint"]["sha256"]),
            str(head["model_state_digest"]),
            torch.device("mps"),
            bridge,
            vendor_manifest,
        )
        verify_shape = tuple(int(value) for value in campaign["promotion"]["verify_bank_shape"])
        verify_identity = hash_file(
            frozen / "verify.u32le",
            str(campaign["promotion"]["verify_bank_sha256"]),
        )
        expected_bytes = math.prod(verify_shape) * np.dtype("<u4").itemsize
        if int(verify_identity["size_bytes"]) != expected_bytes:
            raise RuntimeError("VERIFY bank byte size drift")
        verify = np.memmap(frozen / "verify.u32le", mode="r", dtype="<u4", shape=verify_shape)
        batches = [
            torch.from_numpy(np.array(verify[index], dtype=np.int64, copy=True))
            for index in range(verify_shape[0])
        ]
        source_objective = MPSDuplicateVerifyObjective(loaded_source, batches)
        current_objective = MPSDuplicateVerifyObjective(loaded_current, batches)
        source_flat = source_objective.flat_group(group).detach().cpu().float()
        current_flat = current_objective.flat_group(group).detach().cpu().float()
        if hashlib.sha256(branch._tensor_bytes(current_flat)).hexdigest() != winner[
            "candidate_group_flat_sha256"
        ]:
            raise RuntimeError("current promoted group is not exact winner bytes")
        direction = winner_flat_cpu - source_flat
        if not torch.isfinite(direction).all() or not torch.any(direction != 0):
            raise RuntimeError("winner direction is zero or nonfinite")

        current_state = current_objective.parent_state()
        parent_first = current_objective.sequence_losses(current_state)
        parent_second = current_objective.sequence_losses(current_state)
        parent_means = [_mean(parent_first), _mean(parent_second)]
        parent_duplicate = max(abs(a - b) for a, b in zip(parent_first, parent_second))
        if parent_duplicate > tolerance:
            raise RuntimeError("directional parent duplicate drift")
        if not math.isclose(
            parent_means[0], float(winner["mps_verify_replicated_mean"]), abs_tol=0.0
        ):
            raise RuntimeError("promoted winner CE differs from winner result")

        def evaluate(flat_cpu: torch.Tensor) -> dict[str, object]:
            state = source_objective.state_with_group(group, flat_cpu.to("mps"))
            first = source_objective.sequence_losses(state)
            second = source_objective.sequence_losses(state)
            means = [_mean(first), _mean(second)]
            duplicate = max(abs(a - b) for a, b in zip(first, second))
            return {
                "mps_verify_per_sequence_first": first,
                "mps_verify_per_sequence_second": second,
                "mps_verify_mean_first": means[0],
                "mps_verify_mean_second": means[1],
                "mps_verify_replicated_mean": math.fsum(means) / 2,
                "duplicate_max_abs_sequence_difference": duplicate,
                "passes_current_parent": (
                    duplicate <= tolerance
                    and means[0] < parent_means[0]
                    and means[1] < parent_means[1]
                ),
            }

        alpha_rows = []
        best_flat = current_flat.clone()
        best_means = parent_means
        best_ce = math.fsum(parent_means) / 2
        best_alpha = 1.0
        for alpha in args.alpha_scales:
            trial = winner_flat_cpu.clone() if alpha == 1.0 else source_flat + direction * alpha
            row = {
                "stage": "winner_delta_line_search",
                "alpha": alpha,
                "group_flat_sha256": hashlib.sha256(branch._tensor_bytes(trial)).hexdigest(),
                **evaluate(trial),
            }
            alpha_rows.append(row)
            print(
                f"ALPHA {alpha:g} CE={row['mps_verify_replicated_mean']:.15f} "
                f"pass={row['passes_current_parent']}", flush=True,
            )
            ce = float(row["mps_verify_replicated_mean"])
            if ce < best_ce and bool(row["passes_current_parent"]):
                best_flat = trial.detach().cpu().float()
                best_means = [float(row["mps_verify_mean_first"]), float(row["mps_verify_mean_second"])]
                best_ce = ce
                best_alpha = alpha
            del trial
            torch.mps.empty_cache()

        block_rows = []
        accepted_blocks = []
        cursor = 0
        for block_index, parameter_name in enumerate(loaded_source.groups[group]):
            if args.max_blocks is not None and block_index >= args.max_blocks:
                break
            count = int(source_objective.base_parameters[parameter_name].numel())
            start, end = cursor, cursor + count
            cursor = end
            candidate_rows = []
            selected = None
            for beta in args.block_scales:
                trial = best_flat.clone()
                trial[start:end].add_(direction[start:end], alpha=beta)
                row = {
                    "stage": "tensor_block_direction_extension",
                    "parameter_name": parameter_name,
                    "start": start,
                    "end": end,
                    "beta": beta,
                    "group_flat_sha256": hashlib.sha256(branch._tensor_bytes(trial)).hexdigest(),
                    **evaluate(trial),
                }
                candidate_rows.append(row)
                if (
                    float(row["duplicate_max_abs_sequence_difference"]) <= tolerance
                    and float(row["mps_verify_mean_first"]) < best_means[0]
                    and float(row["mps_verify_mean_second"]) < best_means[1]
                    and (selected is None or float(row["mps_verify_replicated_mean"]) < float(selected[0]["mps_verify_replicated_mean"]))
                ):
                    selected = (row, trial.detach().cpu().float())
                del trial
                torch.mps.empty_cache()
            accepted = selected is not None
            if accepted:
                selected_row, selected_flat = selected
                best_flat = selected_flat
                best_means = [
                    float(selected_row["mps_verify_mean_first"]),
                    float(selected_row["mps_verify_mean_second"]),
                ]
                best_ce = float(selected_row["mps_verify_replicated_mean"])
                accepted_blocks.append({
                    "parameter_name": parameter_name,
                    "beta": selected_row["beta"],
                    "ce": best_ce,
                })
            block_rows.append({
                "parameter_name": parameter_name,
                "start": start,
                "end": end,
                "candidates": candidate_rows,
                "accepted": accepted,
                "accepted_beta": None if not accepted else selected[0]["beta"],
            })
            print(
                f"BLOCK {parameter_name} accepted={accepted} current_ce={best_ce:.15f}",
                flush=True,
            )
        block_inventory_complete = cursor == winner_flat_cpu.numel()
        if args.max_blocks is None and not block_inventory_complete:
            raise RuntimeError("tensor block inventory does not cover winner group")

        scalar_rows = []
        accepted_scalars = []
        nonzero_indices = torch.nonzero(direction != 0, as_tuple=False).reshape(-1)
        ranked_scalar_indices: list[int] = []
        if args.scalar_top_k and int(nonzero_indices.numel()):
            magnitudes = direction[nonzero_indices].abs()
            rank_count = min(args.scalar_top_k, int(nonzero_indices.numel()))
            ranked_positions = torch.topk(
                magnitudes, k=rank_count, largest=True, sorted=True,
            ).indices
            ranked_scalar_indices = [
                int(nonzero_indices[position].item()) for position in ranked_positions
            ]
        for scalar_rank, flat_index in enumerate(ranked_scalar_indices, start=1):
            candidate_rows = []
            selected = None
            delta_value = float(direction[flat_index].item())
            for gamma in args.scalar_scales:
                trial = best_flat.clone()
                trial[flat_index].add_(delta_value * gamma)
                row = {
                    "stage": "single_weight_direction_extension",
                    "scalar_rank": scalar_rank,
                    "flat_index": flat_index,
                    "direction_value": delta_value,
                    "gamma": gamma,
                    "group_flat_sha256": hashlib.sha256(
                        branch._tensor_bytes(trial)
                    ).hexdigest(),
                    **evaluate(trial),
                }
                candidate_rows.append(row)
                if (
                    float(row["duplicate_max_abs_sequence_difference"]) <= tolerance
                    and float(row["mps_verify_mean_first"]) < best_means[0]
                    and float(row["mps_verify_mean_second"]) < best_means[1]
                    and (
                        selected is None
                        or float(row["mps_verify_replicated_mean"])
                        < float(selected[0]["mps_verify_replicated_mean"])
                    )
                ):
                    selected = (row, trial.detach().cpu().float())
                del trial
                torch.mps.empty_cache()
            accepted = selected is not None
            if accepted:
                selected_row, selected_flat = selected
                best_flat = selected_flat
                best_means = [
                    float(selected_row["mps_verify_mean_first"]),
                    float(selected_row["mps_verify_mean_second"]),
                ]
                best_ce = float(selected_row["mps_verify_replicated_mean"])
                accepted_scalars.append({
                    "scalar_rank": scalar_rank,
                    "flat_index": flat_index,
                    "gamma": selected_row["gamma"],
                    "ce": best_ce,
                })
            scalar_rows.append({
                "scalar_rank": scalar_rank,
                "flat_index": flat_index,
                "direction_value": delta_value,
                "candidates": candidate_rows,
                "accepted": accepted,
                "accepted_gamma": None if not accepted else selected[0]["gamma"],
            })
            print(
                f"SCALAR rank={scalar_rank} index={flat_index} "
                f"accepted={accepted} current_ce={best_ce:.15f}",
                flush=True,
            )

        final_sha = hashlib.sha256(branch._tensor_bytes(best_flat)).hexdigest()
        search = {
            "schema": "hfw_w560_canonical_winner_direction_descent_v1",
            "status": (
                "COMPLETED_DUPLICATED_CANONICAL_DIRECTION_AND_BLOCK_SEARCH"
                if block_inventory_complete
                else "COMPLETED_OPERATOR_CHECKPOINT_DUPLICATED_CANONICAL_DIRECTION_AND_BLOCK_PREFIX_SEARCH"
            ),
            "campaign_sha256": campaign_identity["sha256"],
            "source_head_record_sha256": source_head_sha,
            "parent_head_record_sha256": head["record_sha256"],
            "parent_checkpoint_sha256": head["checkpoint"]["sha256"],
            "winner_candidate_result": winner_identity,
            "winner_region": winner_region,
            "winner_candidate_order": args.winner_candidate_order,
            "group": group,
            "direction_sha256": hashlib.sha256(branch._tensor_bytes(direction)).hexdigest(),
            "parent_per_sequence_first": parent_first,
            "parent_per_sequence_second": parent_second,
            "parent_mean_first": parent_means[0],
            "parent_mean_second": parent_means[1],
            "parent_duplicate": parent_duplicate,
            "alpha_rows": alpha_rows,
            "best_alpha_before_blocks": best_alpha,
            "block_rows": block_rows,
            "operator_checkpoint_max_blocks": args.max_blocks,
            "block_inventory_complete": block_inventory_complete,
            "block_elements_covered": cursor,
            "accepted_blocks": accepted_blocks,
            "scalar_selection_policy": "largest_absolute_nonzero_exact_delta_coordinates",
            "scalar_top_k": args.scalar_top_k,
            "scalar_scales": args.scalar_scales,
            "nonzero_direction_coordinate_count": int(nonzero_indices.numel()),
            "ranked_scalar_indices": ranked_scalar_indices,
            "scalar_rows": scalar_rows,
            "accepted_scalars": accepted_scalars,
            "final_group_flat_sha256": final_sha,
            "final_ce": best_ce,
            "strictly_improves_parent": best_means[0] < parent_means[0] and best_means[1] < parent_means[1],
            "local_verify_reused_optimization_data_not_held_out": True,
            "training_or_backprop": False,
            "autograd_calls": 0,
            "optimizer_steps": 0,
        }
        atomic_json(output_root / "SEARCH.json", search)
        proposal_identity = ledger.append_named_record(
            "shortlists", ".shortlist.json", search,
        )

        accepted = bool(search["strictly_improves_parent"])
        child_identity = delta_identity = replacement_identity = None
        model_digest = None
        decision_identity = None
        head_after = head
        if accepted:
            delta_payload = branch._tensor_bytes(best_flat - current_flat)
            delta_sha = hashlib.sha256(delta_payload).hexdigest()
            delta_path = ledger.root / "deltas" / f"{delta_sha}.group_delta.f32le"
            delta_identity = hash_file(delta_path, delta_sha) if delta_path.exists() else write_exclusive(delta_path, delta_payload)
            replacement_payload = branch._tensor_bytes(best_flat)
            replacement_path = ledger.root / "replacement_group_flats" / f"{final_sha}.group_flat.f32le"
            replacement_identity = hash_file(replacement_path, final_sha) if replacement_path.exists() else write_exclusive(replacement_path, replacement_payload)
            final_state_mps = current_objective.state_with_group(group, best_flat.to("mps"))
            final_state_cpu = {name: value.detach().cpu() for name, value in final_state_mps.items()}
            model_digest = bridge.model_state_digest(final_state_cpu)
            child_generation = int(head["model_generation"]) + 1
            selected = {
                "candidate_order": 1,
                "group": group,
                "winner_candidate_order": args.winner_candidate_order,
                "best_alpha_before_blocks": best_alpha,
                "accepted_block_count": len(accepted_blocks),
                "accepted_scalar_count": len(accepted_scalars),
                "candidate_group_flat_sha256": final_sha,
                "mps_verify_mean_first": best_means[0],
                "mps_verify_mean_second": best_means[1],
                "mps_verify_replicated_mean": best_ce,
                "duplicate_max_abs_sequence_difference": 0.0,
            }
            checkpoint_payload = {
                "schema": "hfw_w560_continuous_promoted_checkpoint_v1",
                "origin_checkpoint_sha256": head["origin_checkpoint_sha256"],
                "parent_checkpoint_sha256": head["checkpoint"]["sha256"],
                "generation": child_generation,
                "accepted_proposal_sha256": proposal_identity["sha256"],
                "selected_candidate_order": 1,
                "selected_candidate_delta_sha256": delta_identity["sha256"],
                "solver_cumulative_delta_sha256": delta_identity["sha256"],
                "replacement_group_flat_sha256": replacement_identity["sha256"],
                "group_replacement_materialization_source": "exact_candidate_group_flat_bytes",
                "solver_cumulative_delta_used_to_reconstruct_group": False,
                "model_state_digest": model_digest,
                "optimizer_present": False,
                "scheduler_present": False,
                "training_resume_authorized": False,
                "model": final_state_cpu,
            }
            checkpoint_path = ledger.root / "promoted_checkpoints" / (
                f"generation-{child_generation:08d}-{proposal_identity['sha256'][:16]}.pt"
            )
            child_identity = _write_checkpoint_atomic(checkpoint_path, checkpoint_payload, bridge)
            decision = {
                "schema": "hfw_w560_canonical_winner_direction_descent_decision_v1",
                "status": "ACCEPTED_WINNER_DIRECTION_DESCENT" ,
                "accepted": True,
                "campaign_sha256": campaign_identity["sha256"],
                "shortlist_sha256": proposal_identity["sha256"],
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
                    "duplicate_forward_tolerance": tolerance,
                    "alpha_rows": alpha_rows,
                    "block_rows": block_rows,
                    "scalar_rows": scalar_rows,
                },
                "selected_candidate": selected,
                "child_checkpoint": child_identity,
                "child_model_state_digest": model_digest,
                "ledger_delta": delta_identity,
                "ledger_replacement_group_flat": replacement_identity,
                "checkpoint_materialized_from_exact_group_flat": True,
                "training_or_backprop": False,
                "optimizer_steps": 0,
            }
            decision_identity = ledger.append_named_record("decisions", ".decision.json", decision)
            _validate_accepted_decision_artifacts(
                ledger, frozen, decision, head, str(proposal_identity["sha256"]),
            )
            new_head = {
                "schema": HEAD_SCHEMA,
                "status": "PROMOTED_EXACT_MPS_VERIFIED_WINNER_DIRECTION_DESCENT",
                "authority_generation": int(head["authority_generation"]) + 1,
                "model_generation": child_generation,
                "local_promotions": int(head["local_promotions"]) + 1,
                "parent_head_record_sha256": head["record_sha256"],
                "contract_sha256": head["contract_sha256"],
                "source_head_record_sha256": head["source_head_record_sha256"],
                "source_head_record_path": head["source_head_record_path"],
                "parent_checkpoint_sha256": head["checkpoint"]["sha256"],
                "checkpoint": {**child_identity, "format": "hfw_w560_continuous_promoted_checkpoint_v1"},
                "origin_checkpoint_sha256": head["origin_checkpoint_sha256"],
                "model_state_digest": model_digest,
                "accepted_proposal_sha256": proposal_identity["sha256"],
                "selected_candidate_order": 1,
                "group": group,
                "decision_sha256": decision_identity["sha256"],
                "replacement_group_flat_sha256": replacement_identity["sha256"],
                "solver_cumulative_delta_sha256": delta_identity["sha256"],
                "campaign_sha256": campaign_identity["sha256"],
                "shortlist_sha256": proposal_identity["sha256"],
            }
            head_after = ledger.append_head(new_head)

        promotion = {
            "schema": "hfw_w560_canonical_winner_direction_descent_terminal_v1",
            "status": "ACCEPTED" if accepted else "REJECTED_NO_STRICT_DIRECTIONAL_IMPROVEMENT",
            "accepted": accepted,
            "parent_head_record_sha256": head["record_sha256"],
            "current_head_record_sha256": head_after["record_sha256"],
            "parent_ce": math.fsum(parent_means) / 2,
            "child_ce": best_ce if accepted else None,
            "absolute_improvement": (math.fsum(parent_means) / 2 - best_ce) if accepted else None,
            "relative_improvement": ((math.fsum(parent_means) / 2 - best_ce) / abs(math.fsum(parent_means) / 2)) if accepted else None,
            "best_alpha_before_blocks": best_alpha,
            "accepted_blocks": accepted_blocks,
            "accepted_scalars": accepted_scalars,
            "proposal": proposal_identity,
            "decision": decision_identity,
            "child_checkpoint": child_identity,
        }
        atomic_json(output_root / "PROMOTION.json", promotion)
        print(json.dumps(promotion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
