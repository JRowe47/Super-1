from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .common import read_json_same_fd
from .relocation import vendor_modules


def validated_disk_reservations(config: Mapping[str, object]) -> Mapping[str, int]:
    """Derive and validate storage bounds from the frozen model/solver protocol."""
    package_root, _, _, _, _ = vendor_modules(config)
    protocol, _ = read_json_same_fd(
        Path(package_root) / "config/production_protocol.json",
        str(config["vendor"]["production_protocol_sha256"]),
    )
    configured_groups = tuple(str(value) for value in config["controller"]["groups"])
    protocol_counts = protocol["model"]["expected_group_counts"]
    if set(configured_groups) != set(protocol_counts):
        raise RuntimeError("disk reservation groups differ from frozen protocol groups")
    maximum_group_elements = max(int(protocol_counts[group]) for group in configured_groups)
    maximum_accepts = int(protocol["search"]["maximum_accepts_per_branch"])
    if maximum_accepts != int(protocol["search"]["pairs_per_branch"]):
        raise RuntimeError("frozen solver accept ceiling is not bounded by its pair count")

    disk = config["controller"]["disk_safety"]
    worker = disk["worker_reservation_formula"]
    worker_requirements = {
        "maximum_group_elements_from_frozen_protocol": maximum_group_elements,
        "maximum_accepts_per_branch_from_frozen_protocol": maximum_accepts,
        "artifacts_per_accept": 3,
        "bytes_per_float32_element": 4,
    }
    for key, expected in worker_requirements.items():
        if int(worker.get(key, -1)) != expected:
            raise RuntimeError(f"worker disk reservation input drift: {key}")
    worker_overhead = int(worker["fixed_metadata_and_filesystem_overhead_bytes"])
    if worker_overhead <= 0:
        raise RuntimeError("worker disk reservation overhead must be positive")
    worker_artifact_bytes = (
        maximum_group_elements
        * maximum_accepts
        * worker_requirements["artifacts_per_accept"]
        * worker_requirements["bytes_per_float32_element"]
    )
    worker_reserved = worker_artifact_bytes + worker_overhead
    if int(worker.get("computed_reserved_bytes_per_worker", -1)) != worker_reserved:
        raise RuntimeError("worker disk reservation formula/result drift")

    promotion = disk["promotion_reservation_formula"]
    promotion_requirements = {
        "full_checkpoint_source_size_bytes": int(config["origin"]["size_bytes"]),
        "maximum_group_elements_from_frozen_protocol": maximum_group_elements,
        "selected_group_artifacts": 2,
        "bytes_per_float32_element": 4,
    }
    for key, expected in promotion_requirements.items():
        if int(promotion.get(key, -1)) != expected:
            raise RuntimeError(f"promotion disk reservation input drift: {key}")
    promotion_overhead = int(promotion["fixed_metadata_and_filesystem_overhead_bytes"])
    if promotion_overhead <= 0:
        raise RuntimeError("promotion disk reservation overhead must be positive")
    promotion_group_artifact_bytes = (
        maximum_group_elements
        * promotion_requirements["selected_group_artifacts"]
        * promotion_requirements["bytes_per_float32_element"]
    )
    promotion_reserved = (
        promotion_requirements["full_checkpoint_source_size_bytes"]
        + promotion_group_artifact_bytes
        + promotion_overhead
    )
    if int(promotion.get("computed_reserved_bytes_for_possible_promotion", -1)) != promotion_reserved:
        raise RuntimeError("promotion disk reservation formula/result drift")
    minimum_unreserved = int(disk["minimum_unreserved_free_bytes"])
    if minimum_unreserved <= 0:
        raise RuntimeError("minimum unreserved disk safety floor must be positive")
    return {
        "minimum_unreserved_free_bytes": minimum_unreserved,
        "maximum_group_elements": maximum_group_elements,
        "maximum_accepts_per_branch": maximum_accepts,
        "worker_artifact_bytes_before_overhead": worker_artifact_bytes,
        "reserved_bytes_per_worker": worker_reserved,
        "promotion_group_artifact_bytes_before_overhead": promotion_group_artifact_bytes,
        "reserved_bytes_for_possible_promotion": promotion_reserved,
    }
