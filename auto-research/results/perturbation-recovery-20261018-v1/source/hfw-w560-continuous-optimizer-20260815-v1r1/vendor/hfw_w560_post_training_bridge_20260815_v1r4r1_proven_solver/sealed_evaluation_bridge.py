from __future__ import annotations

import importlib.util
import json
import secrets
import sys
from pathlib import Path
from types import ModuleType
from typing import Mapping

from .manifest import file_sha256


def load_canonical_sealed_evaluation(
    protocol: Mapping[str, object], package_root: Path,
) -> ModuleType:
    binding = protocol["canonical_sealed_evaluation_binding"]
    source = package_root / binding["source_relative_path"]
    receipt_path = package_root / binding["validation_receipt_relative_path"]
    if source.is_symlink() or not source.is_file() or file_sha256(source) != binding["source_sha256"]:
        raise RuntimeError("canonical sealed-evaluation source drift")
    if (receipt_path.is_symlink() or not receipt_path.is_file()
            or file_sha256(receipt_path) != binding["validation_receipt_sha256"]):
        raise RuntimeError("canonical sealed-evaluation validation receipt drift")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or receipt.get("blockers") != []:
        raise RuntimeError("canonical sealed-evaluation receipt does not pass")
    properties = receipt.get("properties", {})
    required = (
        "accepted_and_rejected_proposals_share_one_hash_chain",
        "authority_preimage_is_outside_search_visible_root",
        "calibration_eval_disjointness_proved_after_reveal",
        "early_reveal_fails_closed",
        "post_terminal_proposals_fail_closed",
        "search_receives_only_opaque_eval_commitment",
        "validation_bytes_rehashed_against_reveal",
    )
    if not all(properties.get(name) is True for name in required):
        raise RuntimeError("canonical sealed-evaluation property drift")
    spec = importlib.util.spec_from_file_location(
        "hfw_w560_bound_canonical_sealed_evaluation", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import canonical sealed-evaluation source")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    if Path(module.__file__).resolve() != source.resolve():
        raise RuntimeError("canonical sealed-evaluation import path drift")
    return module


def load_canonical_sealed_evaluation_v2(
    protocol: Mapping[str, object], package_root: Path,
) -> tuple[ModuleType, ModuleType]:
    binding = protocol["canonical_sealed_evaluation_binding"]
    artifacts = (
        ("v2_source_relative_path", "v2_source_sha256"),
        ("v2_vendor_manifest_relative_path", "v2_vendor_manifest_sha256"),
        ("v2_validation_receipt_relative_path", "v2_validation_receipt_sha256"),
    )
    for path_key, sha_key in artifacts:
        path = package_root / binding[path_key]
        if path.is_symlink() or not path.is_file() or file_sha256(path) != binding[sha_key]:
            raise RuntimeError(f"canonical sealed-evaluation v2 binding drift: {path_key}")
    receipt = json.loads((package_root / binding["v2_validation_receipt_relative_path"]).read_text())
    if receipt.get("status") != "PASS" or receipt.get("blockers") != []:
        raise RuntimeError("canonical sealed-evaluation v2 validation did not pass")
    v1 = load_canonical_sealed_evaluation(protocol, package_root)
    previous_v1 = sys.modules.get("sealed_evaluation")
    sys.modules["sealed_evaluation"] = v1
    source = package_root / binding["v2_source_relative_path"]
    spec = importlib.util.spec_from_file_location("hfw_w560_bound_canonical_sealed_evaluation_v2", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import canonical sealed-evaluation v2")
    v2 = importlib.util.module_from_spec(spec)
    previous_v2 = sys.modules.get(spec.name)
    sys.modules[spec.name] = v2
    try:
        spec.loader.exec_module(v2)
    except BaseException:
        if previous_v2 is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_v2
        if previous_v1 is None:
            sys.modules.pop("sealed_evaluation", None)
        else:
            sys.modules["sealed_evaluation"] = previous_v1
        raise
    if Path(v2.__file__).resolve() != source.resolve():
        raise RuntimeError("canonical sealed-evaluation v2 import path drift")
    return v1, v2


def create_production_search_capability(
    protocol: Mapping[str, object], package_root: Path, *,
    authority_root: Path, public_root: Path,
    calibration_layout: Mapping[str, object],
    salt: bytes | None = None,
):
    canonical, canonical_v2 = load_canonical_sealed_evaluation_v2(protocol, package_root)
    data = protocol["data"]
    expected_eval_examples = int(data["fresh_fixed_eval_batches"]) * int(data["batch_size"])
    if calibration_layout.get("fresh_fixed_eval_offsets_present") is not False:
        raise RuntimeError("search-visible calibration leaked fixed-eval offsets")
    sample_length = int(data["context"]) + 1
    occupied = [
        tuple(interval)
        for section in ("prior_public_eval_exclusion", "calibration")
        for record in calibration_layout[section]
        for interval in record["intervals"]
    ]
    token_count = int(calibration_layout["token_count"])
    authority = canonical_v2.create_authority_generated_eval_commitment(
        authority_root=authority_root,
        public_root=public_root,
        validation_sha256=data["sha256"],
        validation_token_count=int(calibration_layout["token_count"]),
        evaluation_window_count=expected_eval_examples,
        sample_length_tokens=sample_length,
        offset_stride_tokens=sample_length,
        calibration_offsets=calibration_layout["calibration_offsets"],
        calibration_sample_length_tokens=sample_length,
        public_excluded_intervals=occupied[:sum(len(record["intervals"]) for record in calibration_layout["prior_public_eval_exclusion"])],
        private_entropy=salt,
    )
    capability = authority.search_capability
    calibration_sha = canonical.create_calibration_manifest(
        capability,
        validation_sha256=data["sha256"],
        validation_token_count=int(calibration_layout["token_count"]),
        calibration_offsets=calibration_layout["calibration_offsets"],
        sample_length_tokens=sample_length,
    )
    public = capability.public_payload()
    encoded = json.dumps(public, sort_keys=True)
    for forbidden in protocol["canonical_sealed_evaluation_binding"]["search_capability_must_exclude"]:
        if forbidden in encoded:
            raise RuntimeError(f"search capability leaked authority field: {forbidden}")
    return canonical, canonical_v2, authority, capability, calibration_sha


class CanonicalProposalRegistry:
    def __init__(
        self, canonical_v2: ModuleType, capability, parent_checkpoint_sha256: str,
        calibration_manifest_sha256: str, group: str,
        group_aliases: Mapping[str, str], allowed_groups,
    ):
        self.canonical_v2 = canonical_v2
        self.capability = capability
        self.parent_checkpoint_sha256 = parent_checkpoint_sha256
        self.calibration_manifest_sha256 = calibration_manifest_sha256
        self.group = group
        self.canonical_group = group_aliases[group]
        self.allowed_groups = tuple(allowed_groups)
        self.branch_id = f"w560-{self.canonical_group}"
        self.current_child_sha256 = parent_checkpoint_sha256
        self.last_branch_event_sha256 = None

    def append(
        self, *, proposal_sha256: str, delta_sha256: str,
        child_checkpoint_sha256: str | None, accepted: bool,
        reason: str, record: Mapping[str, object],
    ) -> str:
        parent = self.current_child_sha256
        if accepted:
            if child_checkpoint_sha256 is None:
                raise RuntimeError("accepted canonical event requires actual materialized child SHA")
            child = str(child_checkpoint_sha256)
        else:
            if child_checkpoint_sha256 is not None:
                raise RuntimeError("rejected canonical event cannot advance a child checkpoint")
            child = parent
        event = {
            "branch_id": self.branch_id,
            "group": self.canonical_group,
            "proposal_sha256": proposal_sha256,
            "parent_checkpoint_sha256": parent,
            "child_checkpoint_sha256": child,
            "delta_sha256": delta_sha256,
            "decision": "accepted" if accepted else "rejected",
            "decision_reason": reason,
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "summed_nll_before": record["summed_nll_before"],
            "summed_nll_after": record["summed_nll_after"],
            "device": record.get("device"),
            "precision": "cuda_bfloat16_logits_float32_ce",
            "runtime": record.get("runtime"),
            "metadata": dict(record),
        }
        event_sha = self.canonical_v2.append_branch_event(
            self.capability, branch_event=event, allowed_groups=self.allowed_groups,
        )
        self.current_child_sha256 = child
        self.last_branch_event_sha256 = event_sha
        return event_sha

    def branch_final(self) -> Mapping[str, object]:
        if self.last_branch_event_sha256 is None:
            raise RuntimeError("branch has no canonical proposal events")
        return {
            "branch_id": self.branch_id,
            "group": self.canonical_group,
            "final_checkpoint_sha256": self.current_child_sha256,
            "last_branch_event_sha256": self.last_branch_event_sha256,
        }
