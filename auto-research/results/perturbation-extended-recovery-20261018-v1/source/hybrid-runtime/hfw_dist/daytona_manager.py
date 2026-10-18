from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import stat
import sys
import threading
import time
from typing import Callable, Iterable, Mapping, Sequence, TypeVar

from .common import (
    canonical_json_bytes,
    fsync_directory,
    hash_file,
    hash_file_region,
    read_json,
    sha256_bytes,
    validate_inventory,
    write_exclusive,
    write_json_exclusive,
)
from .seal_upload_staging import build_upload_staging_receipt


SDK_VERSION = "0.194.0"
REMOTE_SNAPSHOT_ROOT = "/opt/w560"
REMOTE_BUILD_EVIDENCE_ROOT = "/opt/w560-build"
REMOTE_SCRATCH_ROOT = "/tmp/w560-scratch"
REMOTE_OUTPUT_ROOT = "/tmp/w560-output"
REMOTE_RESULT_PATH = f"{REMOTE_OUTPUT_ROOT}/CPU_SEARCH_RESULT.json"
REMOTE_FLAT_PATH = f"{REMOTE_OUTPUT_ROOT}/trajectory_final_group_flat.f32le"
REMOTE_BUNDLE_PATH = f"{REMOTE_OUTPUT_ROOT}/accepted_candidates.f32le.bundle"
REMOTE_FAILURE_PATH = f"{REMOTE_OUTPUT_ROOT}/CPU_SEARCH_FAILURE.json"
DETERMINISTIC_FAILURE_TERMINAL_FILENAME = (
    "CLOUD_DETERMINISTIC_FAILURE_COMPLETE.json"
)
DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH = (
    "orchestration/DETERMINISTIC_FAILURE_DECISION.json"
)
MANAGER_LABEL = "hfw-w560-daytona-manager-v1"
UPLOAD_STAGING_RECEIPT_RELATIVE_PATH = Path(
    "runtime/config/UPLOAD_SNAPSHOT_STAGING.json"
)
RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH = Path(
    "runtime/config/RECOVERED_TASK0_PROVENANCE.json"
)
RECOVERED_TASK0_PROVENANCE_SCHEMA_V2 = (
    "hfw_w560_recovered_task0_provenance_v2"
)
RECOVERED_TASK0_PROVENANCE_SCHEMA_V3 = (
    "hfw_w560_recovered_task0_provenance_v3"
)
RECOVERY_PRECLOUD_SCHEMA_V3 = (
    "hfw_w560_recovery_precloud_zero_resource_gate_v3"
)
OUTPUT_FILENAMES = (
    "CPU_SEARCH_RESULT.json",
    "trajectory_final_group_flat.f32le",
    "accepted_candidates.f32le.bundle",
)
WORKER_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPYCACHEPREFIX": "/tmp/w560-pycache",
}
LOCKED_RUNTIME_PACKAGES = (
    "filelock==3.32.3",
    "fsspec==2026.7.0",
    "jinja2==3.1.6",
    "markupsafe==3.0.3",
    "mpmath==1.3.0",
    "networkx==3.6.1",
    "setuptools==80.9.0",
    "sympy==1.14.0",
    "typing-extensions==4.16.0",
)
EXPECTED_WHEEL_SHA256 = {
    "filelock-3.32.3-py3-none-any.whl": "7f0ca4bcc0e181c60dbbd8aa9ab5b120ebb99e4e064e83636340056f833a1f09",
    "fsspec-2026.7.0-py3-none-any.whl": "b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279",
    "jinja2-3.1.6-py3-none-any.whl": "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67",
    "markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": "d6dd0be5b5b189d31db7cda48b91d7e0a9795f31430b7f271219ab30f1d3ac9d",
    "mpmath-1.3.0-py3-none-any.whl": "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c",
    "networkx-3.6.1-py3-none-any.whl": "d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762",
    "numpy-2.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "0123ffdaa88fa4ab64835dcbde75dcdf89c453c922f18dced6e27c90d1d0ec5a",
    "setuptools-80.9.0-py3-none-any.whl": "062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922",
    "sympy-1.14.0-py3-none-any.whl": "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5",
    "torch-2.8.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl": "cb9a8ba8137ab24e36bf1742cb79a1294bd374db570f09fc15a5e1318160db4e",
    "typing_extensions-4.16.0-py3-none-any.whl": "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8",
}
PRICING_SOURCE_URL = "https://www.daytona.io/pricing"
_KEY_PATTERN = re.compile(r"(?:dtn_|DAYTONA_(?:API_KEY|JWT_TOKEN))[A-Za-z0-9._-]*")
_DIGEST_PINNED_PYTHON_IMAGE = re.compile(
    r"^python:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")
FP64_FULL_BANK_GATE_SCHEMA = "hfw_w560_cpu_fp64_all_search_acceptance_gate_v1"
FP64_BASE_PREFLIGHT_SCHEMA = "hfw_w560_cpu_fp64_all_search_base_preflight_v1"
FP64_OBJECTIVE_RECEIPT_SCHEMA = "hfw_w560_cpu_fp64_stable_microbatch_objective_receipt_v1"


class ManagerError(RuntimeError):
    pass


class Cancelled(ManagerError):
    pass


class DeterministicNumericalWorkerFailure(ManagerError):
    def __init__(
        self,
        *,
        task_index: int,
        attempt: int,
        exit_code: int,
        failure: Mapping[str, object],
        failure_identity: Mapping[str, object],
        failure_relative_path: str,
    ):
        super().__init__(
            f"worker {task_index} produced a validated terminal numerical "
            "failure"
        )
        self.task_index = task_index
        self.attempt = attempt
        self.exit_code = exit_code
        self.failure = dict(failure)
        self.failure_identity = dict(failure_identity)
        self.failure_relative_path = failure_relative_path


@dataclass(frozen=True)
class ManagerConfig:
    package_root: Path
    wave_root: Path
    upload_snapshot_root: Path
    base_image: str
    api_url: str
    target: str
    worker_timeout_seconds: int
    campaign_timeout_seconds: int
    api_timeout_seconds: int
    poll_seconds: int
    maximum_attempts: int
    create_concurrency: int
    retry_initial_seconds: float
    retry_maximum_seconds: float
    vcpu_hour_rate_usd: str
    memory_gib_hour_rate_usd: str
    disk_gib_hour_rate_usd: str

    def receipt_value(
        self,
        campaign_sha256: str,
        started_unix_ns: int,
        upload_snapshot_identity: Mapping[str, object],
        upload_staging_receipt_identity: Mapping[str, object],
        recovered_task0_provenance_identity: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        return {
            "schema": "hfw_w560_daytona_manager_run_config_v1",
            "status": "SEALED_BEFORE_CLOUD_MUTATION",
            "campaign_sha256": campaign_sha256,
            "package_root": str(self.package_root),
            "wave_root": str(self.wave_root),
            "upload_snapshot_root": str(self.upload_snapshot_root),
            "upload_snapshot_identity": dict(upload_snapshot_identity),
            "upload_snapshot_staging_receipt": dict(
                upload_staging_receipt_identity
            ),
            "recovered_task0_provenance": (
                None if recovered_task0_provenance_identity is None
                else dict(recovered_task0_provenance_identity)
            ),
            "base_image": self.base_image,
            "base_image_digest": self.base_image.rsplit("@sha256:", 1)[1],
            "api_url": self.api_url,
            "target": self.target,
            "worker_timeout_seconds": self.worker_timeout_seconds,
            "campaign_timeout_seconds": self.campaign_timeout_seconds,
            "campaign_started_unix_ns": started_unix_ns,
            "campaign_deadline_unix_ns": (
                started_unix_ns + self.campaign_timeout_seconds * 1_000_000_000
            ),
            "api_timeout_seconds": self.api_timeout_seconds,
            "poll_seconds": self.poll_seconds,
            "maximum_attempts": self.maximum_attempts,
            "create_concurrency": self.create_concurrency,
            "retry_initial_seconds": self.retry_initial_seconds,
            "retry_maximum_seconds": self.retry_maximum_seconds,
            "rates_usd": {
                "vcpu_hour": self.vcpu_hour_rate_usd,
                "memory_gib_hour": self.memory_gib_hour_rate_usd,
                "disk_gib_hour": self.disk_gib_hour_rate_usd,
            },
            "pricing_source": PRICING_SOURCE_URL,
            "host_runtime": {
                "python_executable": str(Path(sys.executable).resolve()),
                "python_version": ".".join(str(value) for value in sys.version_info[:3]),
                "daytona_sdk_version": importlib.metadata.version("daytona"),
            },
            "credentials_source": "DAYTONA_API_KEY_PROCESS_ENVIRONMENT_ONLY",
            "credentials_written_to_disk_or_worker": False,
            "worker_network_block_all": True,
            "worker_secrets": None,
            "worker_environment_keys": sorted(WORKER_ENVIRONMENT),
        }


class ReceiptJournal:
    """Append-only, self-validating receipts for every remote mutation."""

    def __init__(self, root: Path, campaign_sha256: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.campaign_sha256 = campaign_sha256
        self._lock = threading.Lock()
        self._records = self._read_all()

    def _read_all(self) -> list[Mapping[str, object]]:
        records: list[Mapping[str, object]] = []
        expected_index = 0
        previous = None
        for path in sorted(self.root.glob("*.receipt.json")):
            match = re.fullmatch(r"(\d{8})-([0-9a-f]{64})\.receipt\.json", path.name)
            if match is None or int(match.group(1)) != expected_index:
                raise ManagerError(f"receipt journal sequence drift: {path.name}")
            value, identity = read_json(path, match.group(2))
            if value.get("schema") != "hfw_w560_daytona_manager_mutation_receipt_v1":
                raise ManagerError("receipt journal schema drift")
            if int(value.get("receipt_index", -1)) != expected_index:
                raise ManagerError("receipt body index drift")
            if value.get("previous_receipt_sha256") != previous:
                raise ManagerError("receipt chain drift")
            if value.get("campaign_sha256") != self.campaign_sha256:
                raise ManagerError("receipt campaign binding drift")
            records.append(value)
            previous = identity["sha256"]
            expected_index += 1
        return records

    @property
    def records(self) -> tuple[Mapping[str, object], ...]:
        with self._lock:
            return tuple(self._records)

    def append(self, event: str, **fields: object) -> Mapping[str, object]:
        with self._lock:
            index = len(self._records)
            previous = None
            if self._records:
                previous = sha256_bytes(canonical_json_bytes(self._records[-1]))
            value = {
                "schema": "hfw_w560_daytona_manager_mutation_receipt_v1",
                "receipt_index": index,
                "campaign_sha256": self.campaign_sha256,
                "previous_receipt_sha256": previous,
                "event": event,
                "recorded_unix_ns": time.time_ns(),
                **fields,
            }
            payload = canonical_json_bytes(value)
            digest = sha256_bytes(payload)
            write_exclusive(
                self.root / f"{index:08d}-{digest}.receipt.json",
                payload,
            )
            self._records.append(value)
            return value

    def latest(self, event: str, **matches: object) -> Mapping[str, object] | None:
        with self._lock:
            for value in reversed(self._records):
                if value.get("event") != event:
                    continue
                if all(value.get(key) == expected for key, expected in matches.items()):
                    return value
        return None


def _mutation_receipt_identity(
    receipt: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the immutable file identity of one validated journal receipt."""
    index = receipt.get("receipt_index")
    event = receipt.get("event")
    if (
        receipt.get("schema")
        != "hfw_w560_daytona_manager_mutation_receipt_v1"
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or not isinstance(event, str)
        or not event
    ):
        raise ManagerError("mutation receipt identity contract drift")
    payload = canonical_json_bytes(receipt)
    return {
        "receipt_index": index,
        "event": event,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _safe_error(error: BaseException, credential: str) -> str:
    value = str(error)
    if credential:
        value = value.replace(credential, "[REDACTED]")
    value = _KEY_PATTERN.sub("[REDACTED]", value)
    return value[:2000]


def _state_text(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _sdk_nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManagerError(f"invalid Daytona SDK integer quantity: {field}")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise ManagerError(f"invalid Daytona SDK integer quantity: {field}")
    normalized = int(value)
    if normalized < 0:
        raise ManagerError(f"invalid Daytona SDK integer quantity: {field}")
    return normalized


def _sdk_nonnegative_number(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManagerError(f"invalid Daytona SDK numeric quantity: {field}")
    if not math.isfinite(float(value)) or value < 0:
        raise ManagerError(f"invalid Daytona SDK numeric quantity: {field}")
    return value


def _sdk_identity_text(value: object, field: str) -> str:
    raw = getattr(value, "value", value)
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or raw == "unknown_default_open_api"
    ):
        raise ManagerError(f"Daytona SDK {field} is missing or ambiguous")
    return raw


def _snapshot_identity(snapshot: object) -> Mapping[str, object]:
    return {
        "id": str(getattr(snapshot, "id")),
        "name": str(getattr(snapshot, "name")),
        "state": _state_text(getattr(snapshot, "state")),
        "size_bytes": (
            None if getattr(snapshot, "size", None) is None
            else _sdk_nonnegative_number(
                getattr(snapshot, "size"), "snapshot size",
            )
        ),
        "cpu": _sdk_nonnegative_integer(
            getattr(snapshot, "cpu"), "snapshot CPU",
        ),
        "memory_gib": _sdk_nonnegative_integer(
            getattr(snapshot, "mem"), "snapshot memory",
        ),
        "disk_gib": _sdk_nonnegative_integer(
            getattr(snapshot, "disk"), "snapshot disk",
        ),
        "gpu": _sdk_nonnegative_integer(
            getattr(snapshot, "gpu"), "snapshot GPU",
        ),
        "region_ids": list(getattr(snapshot, "region_ids", None) or []),
    }


def _sandbox_identity(sandbox: object) -> Mapping[str, object]:
    return {
        "id": str(getattr(sandbox, "id")),
        "name": str(getattr(sandbox, "name")),
        "state": _state_text(getattr(sandbox, "state", None)),
        "desired_state": _state_text(getattr(sandbox, "desired_state", None)),
        "snapshot": getattr(sandbox, "snapshot", None),
        "target": str(getattr(sandbox, "target", "")),
        "cpu": _sdk_nonnegative_integer(
            getattr(sandbox, "cpu"), "sandbox CPU",
        ),
        "memory_gib": _sdk_nonnegative_integer(
            getattr(sandbox, "memory"), "sandbox memory",
        ),
        "disk_gib": _sdk_nonnegative_integer(
            getattr(sandbox, "disk"), "sandbox disk",
        ),
        "gpu": _sdk_nonnegative_integer(
            getattr(sandbox, "gpu"), "sandbox GPU",
        ),
        "network_block_all": bool(getattr(sandbox, "network_block_all", False)),
        "public": bool(getattr(sandbox, "public", False)),
        "auto_stop_interval": int(getattr(sandbox, "auto_stop_interval", 0) or 0),
        "auto_delete_interval": int(getattr(sandbox, "auto_delete_interval", 0) or 0),
        "labels": dict(getattr(sandbox, "labels", {}) or {}),
    }


def _probe_launch_identity(sandbox: object) -> Mapping[str, object]:
    identity = dict(_sandbox_identity(sandbox))
    identity.update({
        "organization_id": _sdk_identity_text(
            getattr(sandbox, "organization_id", None),
            "probe organization ID",
        ),
        "sandbox_class": _sdk_identity_text(
            getattr(sandbox, "sandbox_class", None),
            "probe sandbox class",
        ),
    })
    return identity


def _task_labels(campaign_sha256: str, wave_index: int, task_index: int, attempt: int) -> dict[str, str]:
    return {
        "hfw_manager": MANAGER_LABEL,
        "hfw_campaign_sha256": campaign_sha256,
        "hfw_wave_index": f"{wave_index:08d}",
        "hfw_role": "cpu-proposal-worker",
        "hfw_task_index": f"{task_index:03d}",
        "hfw_attempt": str(attempt),
    }


def _contains_required_labels(
    observed: object,
    required: Mapping[str, str],
) -> bool:
    """Accept sealed platform-added labels without weakening manager ownership."""
    return isinstance(observed, Mapping) and all(
        observed.get(name) == value for name, value in required.items()
    )


def _contains_probe_labels(
    observed: object,
    required: Mapping[str, str],
) -> bool:
    """Allow service labels while keeping the command-free probe role exact."""
    if not _contains_required_labels(observed, required):
        return False
    assert isinstance(observed, Mapping)
    return all(
        isinstance(name, str)
        and (not name.startswith("hfw_") or name in required)
        for name in observed
    )


def _contains_task_labels(
    observed: object,
    required: Mapping[str, str],
) -> bool:
    """Allow service labels while keeping every worker ownership label exact."""
    return _contains_probe_labels(observed, required)


def _task_name(campaign_sha256: str, wave_index: int, task_index: int, attempt: int) -> str:
    return (
        f"hfw-{campaign_sha256[:12]}-v{wave_index:04d}-"
        f"w{task_index:03d}-a{attempt}"
    )


def _probe_labels(campaign_sha256: str, wave_index: int) -> dict[str, str]:
    return {
        "hfw_manager": MANAGER_LABEL,
        "hfw_campaign_sha256": campaign_sha256,
        "hfw_wave_index": f"{wave_index:08d}",
        "hfw_role": "build-evidence-quota-probe",
    }


def _probe_name(campaign_sha256: str, wave_index: int) -> str:
    return f"hfw-{campaign_sha256[:12]}-v{wave_index:04d}-probe"


def _snapshot_name(campaign_sha256: str, wave_index: int) -> str:
    return f"hfw-w560-{campaign_sha256[:16]}-v{wave_index:04d}-pristine"


def _worker_command(task_index: int, campaign_sha256: str) -> str:
    values = [
        "python", "-B", "-m", "hfw_dist.cpu_proposal_worker",
        "--snapshot-root", REMOTE_SNAPSHOT_ROOT,
        "--task-index", str(task_index),
        "--campaign-sha256", campaign_sha256,
        "--scratch-root", REMOTE_SCRATCH_ROOT,
        "--output-root", REMOTE_OUTPUT_ROOT,
    ]
    return " ".join(shlex.quote(value) for value in values)


def _validate_config(config: ManagerConfig) -> None:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("manager must run in the sealed external Python 3.12 environment")
    if importlib.metadata.version("daytona") != SDK_VERSION:
        raise ValueError(f"manager requires Daytona SDK {SDK_VERSION}")
    if not _DIGEST_PINNED_PYTHON_IMAGE.fullmatch(config.base_image):
        raise ValueError("--base-image must be a digest-pinned official python image")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", config.target):
        raise ValueError("invalid Daytona target")
    if config.api_url != "https://app.daytona.io/api":
        raise ValueError("this authority is sealed to the official Daytona API URL")
    _validate_upload_root_path(config.upload_snapshot_root)
    integers = (
        config.worker_timeout_seconds, config.campaign_timeout_seconds,
        config.api_timeout_seconds, config.poll_seconds, config.maximum_attempts,
        config.create_concurrency,
    )
    if any(value <= 0 for value in integers):
        raise ValueError("manager timeouts, attempts, and concurrency must be positive")
    if config.worker_timeout_seconds >= config.campaign_timeout_seconds:
        raise ValueError("worker timeout must be shorter than campaign timeout")
    if not 0 < config.retry_initial_seconds <= config.retry_maximum_seconds:
        raise ValueError("invalid retry interval bounds")
    for value in (
        config.vcpu_hour_rate_usd,
        config.memory_gib_hour_rate_usd,
        config.disk_gib_hour_rate_usd,
    ):
        if Decimal(value) < 0:
            raise ValueError("cost rates must be nonnegative")


def validate_worker_snapshot(snapshot_root: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    snapshot_root = Path(snapshot_root).resolve()
    campaign, campaign_identity = read_json(snapshot_root / "CAMPAIGN.json")
    manifest, manifest_identity = read_json(snapshot_root / "SNAPSHOT_MANIFEST.json")
    if campaign.get("schema") != "hfw_w560_daytona_cpu100_campaign_v1":
        raise ManagerError("campaign schema drift")
    if manifest.get("campaign_sha256") != campaign_identity["sha256"]:
        raise ManagerError("snapshot manifest campaign binding drift")
    if manifest.get("verify_bank_included") is not False:
        raise ManagerError("worker payload declares VERIFY access")
    if manifest.get("lineage_or_promoter_included") is not False:
        raise ManagerError("worker payload declares authority access")
    if manifest.get("cloud_credentials_included") is not False:
        raise ManagerError("worker payload declares cloud credentials")
    scan = manifest.get("credential_pattern_scan", {})
    if scan.get("matches") != 0:
        raise ManagerError("credential scan did not pass")
    validate_inventory(snapshot_root, manifest["files"])
    forbidden = []
    for path in snapshot_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot_root).as_posix().lower()
        if (
            relative == "verify.u32le"
            or "/lineage/" in f"/{relative}/"
            or relative.endswith("promote_mps.py")
        ):
            forbidden.append(relative)
    if forbidden:
        raise ManagerError(f"forbidden worker payload files: {forbidden}")
    if int(campaign["worker_count"]) != len(campaign["tasks"]):
        raise ManagerError("campaign task inventory drift")
    if int(campaign["worker_count"]) != 100:
        raise ManagerError("this manager is sealed for exactly 100 workers")
    proposal = campaign["proposal_lane"]
    expected_proposal = {
        "device": "cpu",
        "forward_device": "cpu",
        "candidate_artifact_dtype": "float32",
        "candidate_codec_dtype": "float32",
        "initial_compressed_state_dtype": "float64",
        "evaluation_parameter_dtype": "float64",
        "evaluation_floating_buffer_dtype": "float64",
        "evaluation_persistent_state_dtype": "float64",
        "hidden_dtype": "float64",
        "logits_dtype": "float64",
        "ce_dtype": "float64",
        "forward_autocast_enabled": False,
        "forward_autocast_dtype": None,
        "distinct_fp64_shadow_state_surrogate": True,
        "canonical_checkpoint_and_model_source_bytes_untouched": True,
        "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
        "proposal_shadow_replaces_canonical_fp32_state_storage": True,
        "candidate_and_checkpoint_domain_remains_exact_float32": True,
        "candidate_float32_widened_to_float64_before_every_score": True,
        "canonical_source_forces_codec_and_persistent_state_float32": True,
        "shadow_surrogate_removes_per_step_float32_state_quantization": True,
        "raw_ce_cross_lane_comparable": False,
        "mps_sole_promotion_authority": True,
        "full_bank_gate_schema": FP64_FULL_BANK_GATE_SCHEMA,
        "base_full_bank_gate_schema": FP64_BASE_PREFLIGHT_SCHEMA,
        "objective_receipt_schema": FP64_OBJECTIVE_RECEIPT_SCHEMA,
    }
    for key, expected in expected_proposal.items():
        if proposal.get(key) != expected:
            raise ManagerError(f"campaign FP64 shadow lane drift: {key}")
    if (
        int(proposal["cpu_limit"]) != 2
        or int(proposal["memory_limit_gib"]) != 4
        or int(proposal["disk_limit_gib"]) != 8
    ):
        raise ManagerError("campaign is not the sealed 2 CPU / 4 GiB / 8 GiB CPU lane")
    normalization = proposal.get("normalization_adapter")
    expected_normalization = {
        "schema": "maxabs_scaled_fp64_layer_norm_rescaled_epsilon_v1",
        "real_arithmetic_equivalent_within_fp64_shadow_surrogate": True,
        "bitwise_equivalent_to_native_layer_norm_claimed": False,
        "scale_floor": "sqrt_layer_norm_epsilon",
        "epsilon_rescaling": "(epsilon_divided_by_scale)_divided_by_scale",
    }
    if not isinstance(normalization, Mapping):
        raise ManagerError("campaign FP64 normalization adapter missing")
    for key, expected in expected_normalization.items():
        if normalization.get(key) != expected:
            raise ManagerError(f"campaign FP64 normalization adapter drift: {key}")
    solver = campaign["solver"]
    search_shape = campaign.get("search_bank_shape")
    if (
        not isinstance(search_shape, list)
        or len(search_shape) != 3
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in search_shape
        )
    ):
        raise ManagerError("campaign SEARCH bank shape drift")
    bank_batches, logical_batch_size, _ = search_shape
    iterations = solver.get("iterations")
    search_logical = solver.get("logical_search_evaluations")
    if (
        not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0
        or not isinstance(search_logical, int) or isinstance(search_logical, bool)
        or search_logical <= 0
    ):
        raise ManagerError("campaign solver base schedule drift")
    minimum_logical = search_logical + 2 * bank_batches
    maximum_logical = minimum_logical + iterations * bank_batches
    expected_solver = {
        "base_center_preflight_logical_evaluations": bank_batches,
        "base_center_preflight_physical_model_forwards": (
            bank_batches * logical_batch_size
        ),
        "physical_search_model_forwards": search_logical * logical_batch_size,
        "acceptance_gate_logical_evaluations_per_attempt": bank_batches,
        "acceptance_gate_physical_model_forwards_per_attempt": (
            bank_batches * logical_batch_size
        ),
        "acceptance_gate_maximum_attempts": iterations,
        "acceptance_gate_requires_all_batches_finite": True,
        "acceptance_gate_requires_strict_full_search_mean_improvement": True,
        "acceptance_gate_before_current_mutation_and_artifact_write": True,
        "terminal_full_search_ranking_logical_evaluations": bank_batches,
        "terminal_full_search_ranking_physical_model_forwards": (
            bank_batches * logical_batch_size
        ),
        "total_logical_evaluations_minimum": minimum_logical,
        "total_physical_model_forwards_minimum": (
            minimum_logical * logical_batch_size
        ),
        "total_logical_evaluations_maximum": maximum_logical,
        "total_physical_model_forwards_maximum": (
            maximum_logical * logical_batch_size
        ),
        "physical_model_forwards_per_logical_evaluation": logical_batch_size,
    }
    for key, expected in expected_solver.items():
        observed = solver.get(key)
        if isinstance(expected, bool):
            matches = observed is expected
        else:
            matches = (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed == expected
            )
        if not matches:
            raise ManagerError(f"campaign FP64 solver accounting drift: {key}")
    failure_policy = campaign.get("worker_failure_policy", {})
    expected_failure_policy = {
        "machine_readable_filename": "CPU_SEARCH_FAILURE.json",
        "schema": "hfw_w560_daytona_cpu_search_failure_v1",
        "deterministic_numerical_status": "FAIL_DETERMINISTIC_NUMERICAL_FAILURE",
        "deterministic_numerical_classification": "DETERMINISTIC_NUMERICAL_FAILURE",
        "deterministic_numerical_exit_code": 42,
        "deterministic_numerical_retry_policy": "TERMINAL_DO_NOT_RETRY",
        "traceback_substring_classification_forbidden": True,
        "unclassified_or_infrastructure_failures_use_bounded_retry": True,
        "candidate_acceptance_gate_failure_is_nonterminal_rejection": True,
        "allowed_stages": [
            "BASE_CENTER_PREFLIGHT", "FORWARD_DTYPE",
            "SEARCH_CENTER", "TERMINAL_RANKING",
        ],
        "terminal_reason_codes_by_stage": {
            "BASE_CENTER_PREFLIGHT": [
                "SEALED_PREFLIGHT_OPERATION_EXCEPTION",
                "BASE_CENTER_NUMERICAL_OR_DTYPE_GATE_FAILED",
                "PREFLIGHT_RSS_LIMIT_REACHED",
            ],
            "FORWARD_DTYPE": ["OBSERVED_FP64_SHADOW_DTYPE_MISMATCH"],
            "SEARCH_CENTER": ["NONFINITE_SEARCH_OBJECTIVE"],
            "TERMINAL_RANKING": [
                "NONFINITE_TERMINAL_RANKING",
                "TERMINAL_CACHE_FLAT_IDENTITY_MISMATCH",
                "TERMINAL_DUPLICATE_BATCH_COUNT_MISMATCH",
                "TERMINAL_DUPLICATE_TOLERANCE_EXCEEDED",
                "WORKER_RSS_LIMIT_REACHED_AFTER_SEARCH",
            ],
        },
    }
    for key, expected in expected_failure_policy.items():
        if failure_policy.get(key) != expected:
            raise ManagerError(f"campaign worker failure policy drift: {key}")
    return campaign, {
        "campaign": campaign_identity,
        "snapshot_manifest": manifest_identity,
    }


def _validate_upload_root_path(upload_snapshot_root: Path) -> Path:
    """Reject ambiguous local build contexts before the Daytona SDK sees them."""
    raw = os.fspath(upload_snapshot_root)
    if not raw or not os.path.isabs(raw):
        raise ManagerError("--upload-snapshot-root must be an absolute path")
    if any(character.isspace() for character in raw):
        raise ManagerError("--upload-snapshot-root must not contain whitespace")
    if os.path.normpath(raw) != raw:
        raise ManagerError("--upload-snapshot-root must be a canonical lexical path")
    path = Path(raw)
    current = Path(path.anchor)
    root_stat = os.lstat(current)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ManagerError("upload snapshot path anchor is not a real directory")
    for component in path.parts[1:]:
        if component in {"", ".", ".."} or any(
            character.isspace() for character in component
        ):
            raise ManagerError("upload snapshot path contains an unsafe component")
        current = current / component
        try:
            observed = os.lstat(current)
        except FileNotFoundError as error:
            raise ManagerError(f"upload snapshot path does not exist: {current}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ManagerError(f"upload snapshot path contains a symlink: {current}")
        if not stat.S_ISDIR(observed.st_mode):
            raise ManagerError(f"upload snapshot path component is not a directory: {current}")
    return path


def _restore_upload_snapshot_read_only(upload_snapshot_root: Path) -> None:
    root = _validate_upload_root_path(upload_snapshot_root)
    directories: list[Path] = []
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False,
    ):
        current_path = Path(current)
        directories.append(current_path)
        for name in directory_names:
            path = current_path / name
            observed = os.lstat(path)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ManagerError("upload staging directory identity drift during mode restore")
        for name in file_names:
            path = current_path / name
            observed = os.lstat(path)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
                raise ManagerError("upload staging file identity drift during mode restore")
            files.append(path)
    for path in files:
        os.chmod(path, 0o400, follow_symlinks=False)
    for path in reversed(directories):
        os.chmod(path, 0o500, follow_symlinks=False)
    if any(os.lstat(path).st_mode & 0o222 for path in (*files, *directories)):
        raise ManagerError("upload staging remained writable after SDK mode restore")


def _strict_tree_inventory(
    root: Path,
    *,
    reject_whitespace_components: bool,
) -> Mapping[str, object]:
    """Hash every regular file and reject links or special filesystem entries."""
    root = Path(root)
    files: dict[str, Mapping[str, object]] = {}
    directories: list[str] = []

    def visit(directory: Path, relative_directory: str) -> None:
        directory_stat = os.lstat(directory)
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
            raise ManagerError(f"snapshot tree directory is unsafe: {directory}")
        if relative_directory:
            directories.append(relative_directory)
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            if entry.name in {"", ".", ".."}:
                raise ManagerError("snapshot tree contains an unsafe name")
            if reject_whitespace_components and any(
                character.isspace() for character in entry.name
            ):
                raise ManagerError(
                    f"upload snapshot tree contains whitespace: {entry.name!r}"
                )
            relative = (
                entry.name if not relative_directory
                else f"{relative_directory}/{entry.name}"
            )
            observed = entry.stat(follow_symlinks=False)
            entry_path = directory / entry.name
            if stat.S_ISLNK(observed.st_mode):
                raise ManagerError(f"snapshot tree contains a symlink: {relative}")
            if stat.S_ISDIR(observed.st_mode):
                visit(entry_path, relative)
                continue
            if not stat.S_ISREG(observed.st_mode):
                raise ManagerError(f"snapshot tree contains a special file: {relative}")
            identity = hash_file(entry_path)
            files[relative] = {
                "sha256": identity["sha256"],
                "size_bytes": int(identity["size_bytes"]),
            }

    visit(root, "")
    return {
        "directories": sorted(directories),
        "files": {name: files[name] for name in sorted(files)},
    }


def validate_upload_snapshot_root(
    authoritative_snapshot_root: Path,
    upload_snapshot_root: Path,
) -> Mapping[str, object]:
    """Prove the no-space upload copy is exactly the authoritative worker payload."""
    upload_root = _validate_upload_root_path(upload_snapshot_root)
    authoritative_root = Path(authoritative_snapshot_root).resolve()
    authoritative_inventory = _strict_tree_inventory(
        authoritative_root, reject_whitespace_components=False,
    )
    upload_inventory = _strict_tree_inventory(
        upload_root, reject_whitespace_components=True,
    )
    if upload_inventory != authoritative_inventory:
        raise ManagerError("upload file/directory inventory differs from authoritative snapshot")
    authoritative_campaign, authoritative_identities = validate_worker_snapshot(
        authoritative_root
    )
    upload_campaign, upload_identities = validate_worker_snapshot(upload_root)
    authoritative_manifest, _ = read_json(
        authoritative_root / "SNAPSHOT_MANIFEST.json"
    )
    upload_manifest, _ = read_json(upload_root / "SNAPSHOT_MANIFEST.json")
    if upload_campaign != authoritative_campaign:
        raise ManagerError("upload campaign differs from authoritative wave snapshot")
    if upload_manifest != authoritative_manifest:
        raise ManagerError("upload snapshot manifest differs from authoritative wave snapshot")
    if upload_identities["campaign"]["sha256"] != authoritative_identities["campaign"]["sha256"]:
        raise ManagerError("upload campaign identity differs from authoritative wave snapshot")
    if (
        upload_identities["snapshot_manifest"]["sha256"]
        != authoritative_identities["snapshot_manifest"]["sha256"]
    ):
        raise ManagerError("upload snapshot identity differs from authoritative wave snapshot")
    inventory_payload = canonical_json_bytes(upload_inventory)
    files = upload_inventory["files"]
    return {
        "schema": "hfw_w560_daytona_upload_snapshot_identity_v1",
        "root": str(upload_root),
        "authoritative_snapshot_root": str(authoritative_root),
        "campaign_sha256": upload_identities["campaign"]["sha256"],
        "snapshot_manifest_sha256": upload_identities["snapshot_manifest"]["sha256"],
        "inventory_sha256": sha256_bytes(inventory_payload),
        "file_count": len(files),
        "directory_count": len(upload_inventory["directories"]),
        "total_size_bytes": sum(
            int(record["size_bytes"]) for record in files.values()
        ),
        "directories": list(upload_inventory["directories"]),
        "files": dict(files),
        "symlinks_or_special_files": 0,
        "whitespace_path_components": 0,
    }


def validate_upload_staging_receipt(
    config: ManagerConfig,
    authoritative_snapshot_root: Path,
    upload_snapshot_identity: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind direct manager launches to the package-sealed staging location."""
    receipt, receipt_identity = read_json(
        config.package_root / UPLOAD_STAGING_RECEIPT_RELATIVE_PATH
    )
    try:
        live_receipt = build_upload_staging_receipt(
            config.package_root,
            authoritative_snapshot_root,
            config.upload_snapshot_root,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ManagerError(
            f"live upload staging proof failed: {error}"
        ) from error
    if canonical_json_bytes(receipt) != canonical_json_bytes(live_receipt):
        raise ManagerError("sealed upload staging receipt differs from live proof")
    if (
        receipt.get("schema") != "hfw_w560_daytona_upload_snapshot_staging_v3"
        or receipt.get("status")
        != "PASS_IMMUTABLE_NO_SPACE_DEVICE_INDEPENDENT_COPY"
    ):
        raise ManagerError("upload snapshot staging receipt schema/status drift")
    if receipt.get("upload_snapshot_root") != str(config.upload_snapshot_root):
        raise ManagerError("manager upload root differs from package-sealed staging root")
    if receipt.get("source_wave_snapshot_path") != str(
        Path(authoritative_snapshot_root).resolve()
    ):
        raise ManagerError("upload staging receipt source snapshot path drift")
    if receipt.get("campaign", {}).get("sha256") != upload_snapshot_identity.get(
        "campaign_sha256"
    ):
        raise ManagerError("upload staging receipt campaign identity drift")
    if receipt.get("snapshot_manifest", {}).get(
        "sha256"
    ) != upload_snapshot_identity.get("snapshot_manifest_sha256"):
        raise ManagerError("upload staging receipt snapshot identity drift")
    files = upload_snapshot_identity.get("files")
    if not isinstance(files, Mapping) or receipt.get("complete_inventory") != files:
        raise ManagerError("upload staging receipt complete inventory drift")
    files_sha256 = sha256_bytes(canonical_json_bytes(files))
    if (
        receipt.get("complete_inventory_sha256") != files_sha256
        or int(receipt.get("file_count", -1))
        != int(upload_snapshot_identity.get("file_count", -2))
        or int(receipt.get("total_size_bytes", -1))
        != int(upload_snapshot_identity.get("total_size_bytes", -2))
    ):
        raise ManagerError("upload staging receipt inventory summary drift")
    required_true = (
        "source_and_upload_inventory_exact_match",
        "source_and_upload_roots_distinct",
        "source_and_upload_roots_non_nested",
        "source_and_upload_corresponding_files_have_disjoint_device_inode_identities",
        "source_and_upload_corresponding_files_are_not_hardlinks",
        "upload_root_absolute_real_directory",
        "upload_root_and_all_components_contain_no_whitespace",
        "upload_tree_contains_no_symlinks",
        "upload_tree_contains_no_special_entries",
        "upload_tree_directories_read_only",
        "upload_tree_files_read_only",
        "upload_tree_files_have_single_link_count",
        "package_root_device_equals_source_snapshot_device",
        "upload_root_device_equals_system_data_device",
    )
    if any(receipt.get(name) is not True for name in required_true):
        raise ManagerError("upload staging receipt proof flag drift")
    return receipt_identity


def _atomic_stream_download(
    chunks: Iterable[bytes], destination: Path,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> Mapping[str, object]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        identity = hash_file(destination, expected_sha256)
        if expected_size_bytes is not None and int(identity["size_bytes"]) != expected_size_bytes:
            raise ManagerError(f"existing download size drift: {destination}")
        return identity
    part = destination.with_name(f".{destination.name}.part")
    if part.exists():
        part.unlink()
    descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise ManagerError("download stream yielded non-bytes")
            offset = 0
            while offset < len(chunk):
                offset += os.write(descriptor, chunk[offset:])
            digest.update(chunk)
            size += len(chunk)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        part.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    observed = digest.hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        part.unlink(missing_ok=True)
        raise ManagerError(f"download SHA-256 mismatch: {destination.name}")
    if expected_size_bytes is not None and size != expected_size_bytes:
        part.unlink(missing_ok=True)
        raise ManagerError(f"download size mismatch: {destination.name}")
    os.chmod(part, 0o444)
    os.replace(part, destination)
    fsync_directory(destination.parent)
    return hash_file(destination, observed)


def _finite_nonnegative(value: object, label: str) -> float:
    observed = float(value)
    if not math.isfinite(observed) or observed < 0:
        raise ManagerError(f"invalid finite numerical diagnostic: {label}")
    return observed


def _validate_tensor_time_diagnostic(
    value: Mapping[str, object],
    *,
    context: int,
    require_finite: bool,
    label: str,
) -> bool:
    all_finite = value.get("all_finite")
    first_bad = value.get("first_nonfinite_timestep")
    _finite_nonnegative(
        value.get("maximum_finite_absolute_value"),
        f"{label}.maximum_finite_absolute_value",
    )
    if all_finite is True:
        if first_bad is not None:
            raise ManagerError(f"finite diagnostic has a bad timestep: {label}")
    elif all_finite is False:
        if not isinstance(first_bad, int) or not 0 <= first_bad < context:
            raise ManagerError(f"nonfinite diagnostic timestep drift: {label}")
    else:
        raise ManagerError(f"finite diagnostic flag drift: {label}")
    if require_finite and all_finite is not True:
        raise ManagerError(f"successful worker has nonfinite diagnostic: {label}")
    return bool(all_finite)


def _sequence_diagnostic_proves_nonfinite(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("loss_class") in {
        "NAN", "POSITIVE_INFINITY", "NEGATIVE_INFINITY",
    } and value.get("loss") is None:
        return True
    return any(
        isinstance(value.get(name), Mapping)
        and value[name].get("all_finite") is False
        for name in ("logits", "hidden", "per_token_loss")
    )


def _sequence_diagnostic_proves_dtype_mismatch(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_pairs = (
        ("logits_dtype", "expected_logits_dtype", "logits_dtype_matches"),
        ("hidden_dtype", "expected_hidden_dtype", "hidden_dtype_matches"),
    )
    matches = []
    for observed_key, expected_key, match_key in expected_pairs:
        observed = value.get(observed_key)
        declared_expected = value.get(expected_key)
        declared_match = value.get(match_key)
        if (
            not isinstance(observed, str)
            or not observed
            or declared_expected != "float64"
            or not isinstance(declared_match, bool)
            or declared_match is not (observed == "float64")
        ):
            return False
        matches.append(declared_match)
    ce_dtype = value.get("ce_dtype")
    if (
        not isinstance(ce_dtype, str)
        or not ce_dtype
        or value.get("expected_ce_dtype") != "float64"
    ):
        return False
    matches.append(ce_dtype == "float64")
    state_dtypes = value.get("persistent_state_floating_dtypes")
    state_match = value.get("persistent_state_dtype_matches")
    if (
        not isinstance(state_dtypes, list)
        or not state_dtypes
        or any(not isinstance(dtype, str) or not dtype for dtype in state_dtypes)
        or not isinstance(state_match, bool)
        or state_match is not all(dtype == "float64" for dtype in state_dtypes)
    ):
        return False
    return not all((*matches, state_match))


def _fp64_numerical_lane(campaign: Mapping[str, object]) -> Mapping[str, object]:
    lane = campaign["proposal_lane"]
    return {
        "candidate_artifact_dtype": lane["candidate_artifact_dtype"],
        "candidate_codec_dtype": lane["candidate_codec_dtype"],
        "forward_device": lane["forward_device"],
        "forward_autocast_enabled": lane["forward_autocast_enabled"],
        "forward_autocast_dtype": lane["forward_autocast_dtype"],
        "evaluation_parameter_dtype": lane["evaluation_parameter_dtype"],
        "floating_buffer_dtype": lane["floating_buffer_dtype"],
        "evaluation_floating_buffer_dtype": lane["evaluation_floating_buffer_dtype"],
        "persistent_state_dtype": lane["persistent_state_dtype"],
        "evaluation_persistent_state_dtype": lane["evaluation_persistent_state_dtype"],
        "initial_compressed_state_dtype": lane["initial_compressed_state_dtype"],
        "hidden_dtype": lane["hidden_dtype"],
        "logits_dtype": lane["logits_dtype"],
        "ce_dtype": lane["ce_dtype"],
        "distinct_fp64_shadow_state_surrogate": lane[
            "distinct_fp64_shadow_state_surrogate"
        ],
        "canonical_checkpoint_and_model_source_bytes_untouched": lane[
            "canonical_checkpoint_and_model_source_bytes_untouched"
        ],
        "canonical_fp32_state_contract_executed_by_proposal_shadow": lane[
            "canonical_fp32_state_contract_executed_by_proposal_shadow"
        ],
        "proposal_shadow_replaces_canonical_fp32_state_storage": lane[
            "proposal_shadow_replaces_canonical_fp32_state_storage"
        ],
        "raw_ce_cross_lane_comparable": lane["raw_ce_cross_lane_comparable"],
        "mps_sole_promotion_authority": lane["mps_sole_promotion_authority"],
    }


def _validate_fp64_sequence(
    campaign: Mapping[str, object], row: Mapping[str, object],
    *, batch_index: int, sequence_index: int,
) -> tuple[bool, float | None]:
    lane = campaign["proposal_lane"]
    context = int(lane["logical_batch_shape"][1]) - 1
    expected = {
        "batch_index": batch_index,
        "sequence_index": sequence_index,
        "expected_hidden_dtype": lane["hidden_dtype"],
        "expected_logits_dtype": lane["logits_dtype"],
        "expected_ce_dtype": lane["ce_dtype"],
        "hidden_dtype": lane["hidden_dtype"],
        "logits_dtype": lane["logits_dtype"],
        "ce_dtype": lane["ce_dtype"],
        "hidden_dtype_matches": True,
        "logits_dtype_matches": True,
        "ce_dtype_matches": True,
    }
    for key, expected_value in expected.items():
        if row.get(key) != expected_value:
            raise ManagerError(f"FP64 sequence diagnostic drift: {key}")
    finite_flags = []
    for name in ("logits", "hidden", "per_token_loss"):
        diagnostic = row.get(name)
        if not isinstance(diagnostic, Mapping):
            raise ManagerError(f"FP64 sequence diagnostic missing: {name}")
        finite_flags.append(_validate_tensor_time_diagnostic(
            diagnostic,
            context=context,
            require_finite=False,
            label=f"batch-{batch_index}.sequence-{sequence_index}.{name}",
        ))
    loss_class = row.get("loss_class")
    loss: float | None
    if loss_class == "FINITE":
        loss = float(row.get("loss"))
        if not math.isfinite(loss):
            raise ManagerError("FP64 finite sequence loss is nonfinite")
    elif loss_class in {"NAN", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"}:
        if row.get("loss") is not None:
            raise ManagerError("FP64 nonfinite sequence loss must be null")
        loss = None
    else:
        raise ManagerError("FP64 sequence loss class drift")
    all_finite = all(finite_flags) and loss is not None
    if row.get("all_finite") is not all_finite:
        raise ManagerError("FP64 sequence aggregate finite flag drift")
    return all_finite, loss


def _validate_full_bank_gate(
    campaign: Mapping[str, object], evidence: Mapping[str, object],
    *, expected_kind: str, expected_attempt_index: int | None,
    require_pass: bool,
) -> Mapping[str, object]:
    lane = campaign["proposal_lane"]
    solver = campaign["solver"]
    logical_shape = [int(value) for value in lane["logical_batch_shape"]]
    batch_count = int(solver["acceptance_gate_logical_evaluations_per_attempt"])
    expected = {
        "schema": lane["full_bank_gate_schema"],
        "gate_kind": expected_kind,
        "gate_attempt_index": expected_attempt_index,
        "logical_evaluations": batch_count,
        "physical_model_forwards": int(
            solver["acceptance_gate_physical_model_forwards_per_attempt"]
        ),
        "logical_batch_shape": logical_shape,
        "physical_microbatch_shape": [
            int(value) for value in lane["physical_microbatch_shape"]
        ],
        "numerical_lane": _fp64_numerical_lane(campaign),
        "raw_ce_cross_lane_comparable": False,
        "canonical_checkpoint_and_model_source_bytes_untouched": True,
        "mps_state_opened_or_mutated": False,
    }
    for key, expected_value in expected.items():
        if evidence.get(key) != expected_value:
            raise ManagerError(f"FP64 full-bank gate contract drift: {key}")
    candidate_sha = evidence.get("candidate_state_sha256")
    if not isinstance(candidate_sha, str) or not _SHA256.fullmatch(candidate_sha):
        raise ManagerError("FP64 full-bank candidate state identity drift")
    batches = evidence.get("batches")
    if not isinstance(batches, list) or len(batches) != batch_count:
        raise ManagerError("FP64 full-bank batch inventory drift")
    recomputed_batch_losses: list[float | None] = []
    all_sequences_finite = True
    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, Mapping) or batch.get("batch_index") != batch_index:
            raise ManagerError("FP64 full-bank batch index drift")
        rows = batch.get("per_sequence")
        if not isinstance(rows, list) or len(rows) != logical_shape[0]:
            raise ManagerError("FP64 full-bank sequence inventory drift")
        sequence_losses = []
        batch_finite = True
        for sequence_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ManagerError("FP64 full-bank sequence row missing")
            row_finite, loss = _validate_fp64_sequence(
                campaign, row,
                batch_index=batch_index,
                sequence_index=sequence_index,
            )
            batch_finite = batch_finite and row_finite
            if loss is not None:
                sequence_losses.append(loss)
        expected_batch_loss = (
            math.fsum(sequence_losses) / logical_shape[0]
            if batch_finite else None
        )
        observed_batch_loss = batch.get("batch_mean_ce")
        if expected_batch_loss is None:
            if observed_batch_loss is not None:
                raise ManagerError("nonfinite FP64 batch mean must be null")
        elif float(observed_batch_loss) != expected_batch_loss:
            raise ManagerError("FP64 full-bank batch mean relation drift")
        if batch.get("all_finite") is not batch_finite:
            raise ManagerError("FP64 full-bank batch finite flag drift")
        recomputed_batch_losses.append(expected_batch_loss)
        all_sequences_finite = all_sequences_finite and batch_finite
    if evidence.get("batch_losses") != recomputed_batch_losses:
        raise ManagerError("FP64 full-bank declared batch losses drift")
    observed_all_finite = evidence.get("all_finite")
    if observed_all_finite is not all_sequences_finite:
        raise ManagerError("FP64 full-bank aggregate finite flag drift")
    finite_mean = (
        math.fsum(float(value) for value in recomputed_batch_losses) / batch_count
        if all_sequences_finite else None
    )
    if finite_mean is None:
        if evidence.get("mean_ce") is not None:
            raise ManagerError("nonfinite FP64 full-bank mean must be null")
    elif float(evidence.get("mean_ce")) != finite_mean:
        raise ManagerError("FP64 full-bank mean relation drift")
    gate_passed = evidence.get("gate_passed")
    mutation_applied = evidence.get("mutation_applied")
    if not isinstance(gate_passed, bool) or not isinstance(mutation_applied, bool):
        raise ManagerError("FP64 full-bank decision flags drift")
    expected_status = (
        "PASS_FINITE_CPU_FLOAT64_FULL_BANK"
        if gate_passed else "REJECT_CPU_FLOAT64_FULL_BANK"
    )
    if evidence.get("status") != expected_status:
        raise ManagerError("FP64 full-bank status drift")
    if require_pass and (not gate_passed or not all_sequences_finite):
        raise ManagerError("required FP64 full-bank gate did not pass")
    if gate_passed and not all_sequences_finite:
        raise ManagerError("nonfinite FP64 gate cannot pass")
    if mutation_applied and not gate_passed:
        raise ManagerError("rejected FP64 gate cannot mutate vendor current")
    if expected_kind != "ACCEPTANCE_CANDIDATE" and mutation_applied:
        raise ManagerError("base/terminal FP64 gate cannot mutate vendor current")
    reference_mean = evidence.get("reference_mean_ce")
    improvement = evidence.get("improvement")
    if expected_kind == "ACCEPTANCE_CANDIDATE":
        if not math.isfinite(float(reference_mean)):
            raise ManagerError("FP64 acceptance reference mean drift")
        expected_improvement = (
            float(reference_mean) - float(finite_mean)
            if finite_mean is not None else None
        )
        if expected_improvement is None:
            if improvement is not None:
                raise ManagerError("nonfinite FP64 acceptance improvement must be null")
        elif float(improvement) != expected_improvement:
            raise ManagerError("FP64 acceptance improvement relation drift")
        if gate_passed and not expected_improvement > 0:
            raise ManagerError("nonimproving FP64 acceptance gate passed")
    elif reference_mean is not None or improvement is not None:
        raise ManagerError("base/terminal FP64 gate has acceptance-only fields")
    maximum_rss = int(evidence.get("maximum_observed_rss_bytes", -1))
    memory_limit = int(lane["memory_limit_gib"]) * 1024 ** 3
    if not 0 < maximum_rss < memory_limit:
        raise ManagerError("FP64 full-bank RSS bound drift")
    return {
        "batch_losses": recomputed_batch_losses,
        "mean_ce": finite_mean,
        "all_finite": all_sequences_finite,
        "gate_passed": gate_passed,
        "mutation_applied": mutation_applied,
        "maximum_observed_rss_bytes": maximum_rss,
    }


def _validate_worker_result(
    campaign: Mapping[str, object], campaign_sha256: str,
    snapshot_manifest_sha256: str,
    task_index: int, result_path: Path,
    flat_path: Path, bundle_path: Path,
    launch_environment: Mapping[str, object] | None,
) -> Mapping[str, object]:
    result, result_identity = read_json(result_path)
    task = campaign["tasks"][task_index]
    solver = campaign["solver"]
    lane = campaign["proposal_lane"]
    gate_attempts = result.get("acceptance_gate_attempts")
    gate_passes = result.get("acceptance_gate_passes")
    gate_failures = result.get("acceptance_gate_failures")
    accepted_updates = result.get("accepted_updates")
    for name, value in (
        ("acceptance_gate_attempts", gate_attempts),
        ("acceptance_gate_passes", gate_passes),
        ("acceptance_gate_failures", gate_failures),
        ("accepted_updates", accepted_updates),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ManagerError(f"worker {task_index} result counter type drift: {name}")
    gate_maximum = int(solver["acceptance_gate_maximum_attempts"])
    if not (
        0 <= accepted_updates == gate_passes <= gate_attempts <= gate_maximum
        and gate_failures == gate_attempts - gate_passes
    ):
        raise ManagerError(f"worker {task_index} acceptance gate relation drift")
    expected_search_logical = int(solver["logical_search_evaluations"])
    expected_search_physical = int(solver["physical_search_model_forwards"])
    expected_terminal_logical = int(
        solver["terminal_full_search_ranking_logical_evaluations"]
    )
    expected_terminal_physical = int(
        solver["terminal_full_search_ranking_physical_model_forwards"]
    )
    expected_preflight_logical = int(solver["base_center_preflight_logical_evaluations"])
    expected_preflight_physical = int(
        solver["base_center_preflight_physical_model_forwards"]
    )
    gate_logical_per_attempt = int(
        solver["acceptance_gate_logical_evaluations_per_attempt"]
    )
    gate_physical_per_attempt = int(
        solver["acceptance_gate_physical_model_forwards_per_attempt"]
    )
    gate_logical = gate_attempts * gate_logical_per_attempt
    gate_physical = gate_attempts * gate_physical_per_attempt
    solver_logical = expected_search_logical + expected_terminal_logical + gate_logical
    solver_physical = expected_search_physical + expected_terminal_physical + gate_physical
    total_logical = int(solver["total_logical_evaluations_minimum"]) + gate_logical
    total_physical = int(solver["total_physical_model_forwards_minimum"]) + gate_physical
    requirements = {
        "schema": "hfw_w560_daytona_cpu_search_result_v1",
        "status": "PASS_PROVISIONAL_CPU_SEARCH_ONLY",
        "campaign_sha256": campaign_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "snapshot_manifest_core_sha256": campaign["snapshot_manifest_core_sha256"],
        "task_sha256": sha256_bytes(canonical_json_bytes(task)),
        "task_index": task_index,
        "worker_id": task["worker_id"],
        "wave_index": int(campaign["wave_index"]),
        "base_head_record_sha256": campaign["base_head_record_sha256"],
        "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
        "base_model_state_digest": campaign["base_model_state_digest"],
        "group": task["group"], "seed": int(task["seed"]),
        "group_elements": int(task["expected_group_elements"]),
        "pairs": int(solver["iterations"]),
        "logical_search_evaluations": expected_search_logical,
        "physical_search_model_forwards": expected_search_physical,
        "terminal_ranking_logical_evaluations": expected_terminal_logical,
        "terminal_ranking_physical_model_forwards": expected_terminal_physical,
        "acceptance_gate_logical_evaluations": gate_logical,
        "acceptance_gate_physical_model_forwards": gate_physical,
        "solver_logical_evaluations_excluding_preflight": solver_logical,
        "solver_physical_model_forwards_excluding_preflight": solver_physical,
        "base_center_preflight_logical_evaluations": expected_preflight_logical,
        "base_center_preflight_physical_model_forwards": expected_preflight_physical,
        "total_logical_evaluations": total_logical,
        "total_physical_model_forwards": total_physical,
        "provisional_search_accepts": accepted_updates,
        "self_calibrated_improvement_semantics": (
            "base_full_search_mean_ce_minus_terminal_full_search_mean_ce"
        ),
        "worker_checkpoint_or_head_writes": 0,
        "distinct_fp64_shadow_state_surrogate": True,
        "canonical_checkpoint_and_model_source_bytes_untouched": True,
        "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
        "proposal_shadow_replaces_canonical_fp32_state_storage": True,
        "raw_ce_cross_lane_comparable": False,
        "mps_sole_promotion_authority": True,
        "mps_verify_opened": False,
        "local_verify_available_to_worker": False,
        "cloud_credentials_available_to_worker": False,
        "training_or_backprop": False,
        "optimizer_steps": 0,
        "provisional_not_held_out_or_promoted": True,
        "accepted_candidate_count": accepted_updates,
        "every_provisionally_accepted_candidate_in_bundle": True,
        "candidate_subset_selection_performed": False,
        "cpu_ranking_used_for_candidate_selection": False,
    }
    for key, expected in requirements.items():
        observed = result.get(key)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, int):
            matches = isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
        else:
            matches = observed == expected
        if not matches:
            raise ManagerError(f"worker {task_index} result contract drift: {key}")
    if not (
        int(solver["total_logical_evaluations_minimum"]) <= total_logical
        <= int(solver["total_logical_evaluations_maximum"])
        and int(solver["total_physical_model_forwards_minimum"]) <= total_physical
        <= int(solver["total_physical_model_forwards_maximum"])
    ):
        raise ManagerError(f"worker {task_index} total evaluation bound drift")
    preflight = result.get("base_center_preflight")
    if not isinstance(preflight, Mapping):
        raise ManagerError(f"worker {task_index} base preflight missing")
    batches = preflight.get("batches")
    if (
        preflight.get("schema") != lane["base_full_bank_gate_schema"]
        or preflight.get("status") != "PASS_FINITE_CPU_FP64_ALL_SEARCH_BASE"
        or preflight.get("failure_reasons") != []
        or preflight.get("batch_indices") != list(range(expected_preflight_logical))
        or preflight.get("logical_batch_shape") != lane["logical_batch_shape"]
        or preflight.get("physical_microbatch_shape") != lane["physical_microbatch_shape"]
        or preflight.get("checkpoint_candidate_dtype") != "float32"
        or preflight.get("evaluation_parameter_dtype") != "float64"
        or preflight.get("forward_device") != "cpu"
        or preflight.get("forward_autocast_dtype") is not None
        or preflight.get("logits_dtype") != "float64"
        or preflight.get("hidden_dtype") != "float64"
        or preflight.get("persistent_state_dtype") != "float64"
        or preflight.get("ce_dtype") != "float64"
        or preflight.get("final_norm_algorithm")
        != lane["normalization_adapter"]["schema"]
        or not isinstance(batches, list) or len(batches) != expected_preflight_logical
        or preflight.get("preflight_logical_evaluations") != expected_preflight_logical
        or preflight.get("preflight_physical_model_forwards") != expected_preflight_physical
    ):
        raise ManagerError(f"worker {task_index} base preflight contract drift")
    recomputed_base_losses = []
    sequence_count = int(campaign["search_bank_shape"][1])
    context = int(campaign["search_bank_shape"][2]) - 1
    for batch_index, batch in enumerate(batches):
        rows = batch.get("per_sequence") if isinstance(batch, Mapping) else None
        if (
            not isinstance(batch, Mapping) or batch.get("batch_index") != batch_index
            or batch.get("passed") is not True
            or not isinstance(rows, list) or len(rows) != sequence_count
        ):
            raise ManagerError(f"worker {task_index} base preflight batch drift")
        sequence_losses = []
        for sequence_index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("sequence_index") != sequence_index:
                raise ManagerError(f"worker {task_index} base sequence index drift")
            expected_row = {
                "logits_dtype": "float64", "expected_logits_dtype": "float64",
                "logits_dtype_matches": True,
                "hidden_dtype": "float64", "expected_hidden_dtype": "float64",
                "hidden_dtype_matches": True,
                "persistent_state_floating_dtypes": ["float64"],
                "persistent_state_dtype_matches": True,
                "ce_dtype": "float64", "expected_ce_dtype": "float64",
                "loss_class": "FINITE",
            }
            if any(row.get(key) != expected for key, expected in expected_row.items()):
                raise ManagerError(f"worker {task_index} base sequence dtype drift")
            loss = float(row.get("loss"))
            if not math.isfinite(loss):
                raise ManagerError(f"worker {task_index} base sequence loss drift")
            for name in ("logits", "hidden", "per_token_loss"):
                diagnostic = row.get(name)
                if not isinstance(diagnostic, Mapping) or not _validate_tensor_time_diagnostic(
                    diagnostic, context=context, require_finite=True,
                    label=f"worker-{task_index}.batch-{batch_index}.{name}",
                ):
                    raise ManagerError(f"worker {task_index} base finite evidence drift")
            sequence_losses.append(loss)
        batch_mean = math.fsum(sequence_losses) / sequence_count
        if not math.isclose(
            batch_mean, float(batch.get("base_center_mean_ce")), rel_tol=0, abs_tol=1e-12,
        ):
            raise ManagerError(f"worker {task_index} base batch mean drift")
        recomputed_base_losses.append(batch_mean)
    base_losses = [float(value) for value in result.get("base_full_search_batch_losses", [])]
    base_mean = math.fsum(base_losses) / len(base_losses) if base_losses else float("nan")
    if (
        len(base_losses) != expected_preflight_logical
        or base_losses != recomputed_base_losses
        or base_losses != [float(value) for value in preflight["base_full_search_batch_losses"]]
        or not math.isclose(base_mean, float(preflight["base_full_search_mean_ce"]), rel_tol=0, abs_tol=1e-12)
        or not math.isclose(base_mean, float(result.get("base_full_search_mean_ce")), rel_tol=0, abs_tol=1e-12)
    ):
        raise ManagerError(f"worker {task_index} base full-search score drift")
    terminal_losses = [float(value) for value in result.get("terminal_full_search_batch_losses", [])]
    if len(terminal_losses) != expected_terminal_logical or not all(
        math.isfinite(value) for value in terminal_losses
    ):
        raise ManagerError(f"worker {task_index} terminal loss inventory drift")
    terminal_mean = math.fsum(terminal_losses) / len(terminal_losses)
    improvement = base_mean - terminal_mean
    if (
        not math.isclose(terminal_mean, float(result.get("terminal_full_search_mean_ce")), rel_tol=0, abs_tol=1e-12)
        or not math.isclose(
            improvement, float(result.get("self_calibrated_full_search_improvement")),
            rel_tol=0, abs_tol=1e-12,
        )
    ):
        raise ManagerError(f"worker {task_index} self-calibrated improvement drift")
    objective = result.get("objective_runtime")
    if not isinstance(objective, Mapping):
        raise ManagerError(f"worker {task_index} objective receipt missing")
    phase_logical = objective.get("phase_logical_evaluations")
    phase_physical = objective.get("phase_physical_model_forwards")
    gate_records = objective.get("acceptance_gate_records")
    expected_objective = {
        "schema": lane["objective_receipt_schema"], "device": "cpu",
        "checkpoint_candidate_dtype": "float32",
        "evaluation_parameter_dtype": "float64",
        "floating_buffer_dtype": "float64", "persistent_state_dtype": "float64",
        "forward_autocast_dtype": None, "logits_dtype": "float64",
        "hidden_dtype": "float64", "ce_dtype": "float64",
        "final_norm_algorithm": lane["normalization_adapter"]["schema"],
        "canonical_fp32_state_storage_contract_replaced_in_proposal_shadow": True,
        "canonical_model_source_or_checkpoint_mutated": False,
        "candidate_f32_widened_to_fp64_for_every_score": True,
        "logical_batch_shape": lane["logical_batch_shape"],
        "physical_microbatch_shape": lane["physical_microbatch_shape"],
        "logical_evaluations_excluding_preflight": solver_logical,
        "physical_model_forwards_excluding_preflight": solver_physical,
        "preflight_logical_evaluations": expected_preflight_logical,
        "preflight_physical_model_forwards": expected_preflight_physical,
        "total_logical_evaluations_including_preflight": total_logical,
        "total_physical_model_forwards_including_preflight": total_physical,
        "acceptance_gate_attempts": gate_attempts,
        "acceptance_gate_passes": gate_passes,
        "acceptance_gate_failures": gate_failures,
        "float64_logits_observed": total_physical,
        "float64_hidden_observed": total_physical,
        "float64_persistent_state_observed": total_physical,
        "float64_ce_evaluations": total_physical,
        "physical_model_forwards_per_logical_evaluation": int(
            solver["physical_model_forwards_per_logical_evaluation"]
        ),
        "inference_only": True, "optimizer_steps": 0, "autograd_calls": 0,
    }
    if any(objective.get(key) != expected for key, expected in expected_objective.items()):
        raise ManagerError(f"worker {task_index} objective runtime accounting drift")
    expected_phase_logical = {
        "SEARCH": expected_search_logical,
        "ACCEPTANCE_COMMIT_GATE": gate_logical,
        "TERMINAL_RANKING": expected_terminal_logical,
    }
    expected_phase_physical = {
        "SEARCH": expected_search_physical,
        "ACCEPTANCE_COMMIT_GATE": gate_physical,
        "TERMINAL_RANKING": expected_terminal_physical,
    }
    if phase_logical != expected_phase_logical or phase_physical != expected_phase_physical:
        raise ManagerError(f"worker {task_index} objective phase accounting drift")
    if not isinstance(gate_records, list) or len(gate_records) != gate_attempts:
        raise ManagerError(f"worker {task_index} acceptance gate inventory drift")
    observed_passes = 0
    current_flat_sha = preflight.get("base_candidate_flat_sha256")
    if not isinstance(current_flat_sha, str) or not _SHA256.fullmatch(current_flat_sha):
        raise ManagerError(f"worker {task_index} base candidate identity drift")
    base_flat_sha = current_flat_sha
    current_full_search_mean = base_mean
    current_full_search_losses = list(base_losses)
    previous_iteration = -1
    for gate_index, gate in enumerate(gate_records):
        losses = gate.get("candidate_batch_losses") if isinstance(gate, Mapping) else None
        finite = isinstance(losses, list) and len(losses) == gate_logical_per_attempt and all(
            value is not None and math.isfinite(float(value)) for value in losses
        )
        candidate_mean = (
            math.fsum(float(value) for value in losses) / len(losses) if finite else None
        )
        expected_pass = bool(
            finite and candidate_mean < float(gate.get("current_full_search_mean_ce"))
        )
        if (
            not isinstance(gate, Mapping)
            or gate.get("schema") != lane["full_bank_gate_schema"]
            or gate.get("gate_index") != gate_index
            or not isinstance(gate.get("iteration"), int)
            or isinstance(gate.get("iteration"), bool)
            or not previous_iteration < gate.get("iteration") < int(solver["iterations"])
            or gate.get("group") != task["group"]
            or gate.get("current_flat_sha256") != current_flat_sha
            or not math.isclose(
                float(gate.get("current_full_search_mean_ce", float("nan"))),
                current_full_search_mean,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or gate.get("batch_indices") != list(range(gate_logical_per_attempt))
            or gate.get("logical_evaluations") != gate_logical_per_attempt
            or gate.get("physical_model_forwards") != gate_physical_per_attempt
            or gate.get("current_mutation_or_artifact_write_before_gate") is not False
            or gate.get("all_finite") is not finite
            or gate.get("strict_full_search_mean_improvement") is not expected_pass
            or gate.get("passed") is not expected_pass
            or (candidate_mean is None) != (gate.get("candidate_full_search_mean_ce") is None)
            or (
                candidate_mean is not None and not math.isclose(
                    candidate_mean, float(gate["candidate_full_search_mean_ce"]),
                    rel_tol=0, abs_tol=1e-12,
                )
            )
        ):
            raise ManagerError(f"worker {task_index} acceptance gate evidence drift")
        for digest_key in ("current_flat_sha256", "candidate_flat_sha256"):
            digest = gate.get(digest_key)
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ManagerError(f"worker {task_index} gate identity drift")
        previous_iteration = int(gate["iteration"])
        observed_passes += int(expected_pass)
        if expected_pass:
            current_flat_sha = str(gate["candidate_flat_sha256"])
            assert candidate_mean is not None
            current_full_search_mean = candidate_mean
            current_full_search_losses = [float(value) for value in losses]
    if observed_passes != gate_passes:
        raise ManagerError(f"worker {task_index} acceptance gate pass count drift")
    duplicate = result.get("terminal_full_search_duplicate")
    recomputed_duplicate_maximum = max(
        abs(terminal - cached)
        for terminal, cached in zip(terminal_losses, current_full_search_losses)
    )
    if (
        not isinstance(duplicate, Mapping)
        or duplicate.get("status")
        != "PASS_TERMINAL_FULL_BANK_DUPLICATES_CACHED_COMMIT_GATE"
        or duplicate.get("flat_sha256") != current_flat_sha
        or duplicate.get("batch_count") != expected_terminal_logical
        or float(duplicate.get("maximum_absolute_difference", float("inf")))
        != recomputed_duplicate_maximum
        or not 0 <= float(duplicate.get("maximum_absolute_difference", float("inf")))
        <= float(duplicate.get("absolute_tolerance", -1))
        or float(duplicate.get("absolute_tolerance", -1))
        != float(lane["fp64_full_bank_duplicate_absolute_tolerance"])
    ):
        raise ManagerError(f"worker {task_index} terminal duplicate evidence drift")
    memory_limit = int(lane["memory_limit_gib"]) * 1024 ** 3
    environment = result.get("environment")
    if not isinstance(environment, Mapping):
        raise ManagerError(f"worker {task_index} environment missing")
    peak_rss = int(environment.get("peak_rss_bytes", -1))
    objective_rss = int(objective.get("maximum_observed_rss_bytes", -1))
    if (
        not 0 < objective_rss <= peak_rss < memory_limit
        or environment.get("memory_limit_bytes") != memory_limit
        or environment.get("peak_rss_strictly_below_memory_limit") is not True
        or not isinstance(environment.get("torch_cpu_capability"), str)
        or not environment["torch_cpu_capability"]
        or not isinstance(environment.get("platform_processor"), str)
        or not isinstance(environment.get("proc_cpuinfo_distinct_model_names"), list)
        or not environment["proc_cpuinfo_distinct_model_names"]
        or any(not isinstance(name, str) or not name for name in environment["proc_cpuinfo_distinct_model_names"])
    ):
        raise ManagerError(f"worker {task_index} environment/RSS disclosure drift")
    sealed = lane["sealed_environment"]
    environment_keys = (
        "python_major_minor", "platform_system", "machine", "torch_version",
        "torch_cuda_version", "numpy_version",
    )
    for key in environment_keys:
        if environment.get(key) != sealed[key]:
            raise ManagerError(f"worker {task_index} sealed environment drift: {key}")
        if launch_environment is not None and environment.get(key) != launch_environment.get(key):
            raise ManagerError(f"worker {task_index} launch environment drift: {key}")
    declared = result.get("trajectory_final_group_flat")
    if not isinstance(declared, Mapping):
        raise ManagerError(f"worker {task_index} final flat declaration missing")
    flat_identity = hash_file(flat_path, str(declared.get("sha256")))
    if (
        int(flat_identity["size_bytes"]) != int(declared.get("size_bytes", -1))
        or int(flat_identity["size_bytes"]) != int(task["expected_final_flat_size_bytes"])
        or flat_identity["sha256"] != current_flat_sha
        or result.get("changed") is not (current_flat_sha != base_flat_sha)
    ):
        raise ManagerError(f"worker {task_index} final flat size drift")
    declared_bundle = result.get("accepted_candidate_bundle")
    candidates = result.get("accepted_candidates")
    if not isinstance(declared_bundle, Mapping) or not isinstance(candidates, list):
        raise ManagerError(f"worker {task_index} accepted candidate bundle missing")
    bundle_identity = hash_file(bundle_path, str(declared_bundle.get("sha256")))
    expected_flat_size = int(task["expected_final_flat_size_bytes"])
    if (
        int(bundle_identity["size_bytes"])
        != accepted_updates * expected_flat_size
        or int(bundle_identity["size_bytes"])
        != int(declared_bundle.get("size_bytes", -1))
        or len(candidates) != accepted_updates
    ):
        raise ManagerError(f"worker {task_index} accepted candidate bundle size drift")
    passing_gates = [gate for gate in gate_records if gate.get("passed") is True]
    for candidate_order, (candidate, gate) in enumerate(
        zip(candidates, passing_gates), start=1,
    ):
        if not isinstance(candidate, Mapping):
            raise ManagerError(f"worker {task_index} accepted candidate record drift")
        candidate_sha = gate.get("candidate_flat_sha256")
        if (
            candidate.get("candidate_order") != candidate_order
            or candidate.get("accept_index") != candidate_order
            or candidate.get("pair_index") != gate.get("iteration")
            or candidate.get("group") != task["group"]
            or candidate.get("offset_bytes")
            != (candidate_order - 1) * expected_flat_size
            or candidate.get("size_bytes") != expected_flat_size
            or candidate.get("group_flat_sha256") != candidate_sha
            or not all(
                isinstance(candidate.get(key), str)
                and _SHA256.fullmatch(str(candidate[key]))
                for key in (
                    "group_flat_sha256", "accepted_record_sha256",
                    "artifact_only_child_sha256", "cumulative_delta_sha256",
                )
            )
            or not math.isclose(
                float(candidate.get("candidate_cpu_full_search_mean_ce", float("nan"))),
                float(gate.get("candidate_full_search_mean_ce", float("nan"))),
                rel_tol=0, abs_tol=1e-12,
            )
        ):
            raise ManagerError(f"worker {task_index} accepted candidate provenance drift")
        try:
            hash_file_region(
                bundle_path,
                int(candidate["offset_bytes"]),
                int(candidate["size_bytes"]),
                str(candidate_sha),
            )
        except RuntimeError as error:
            raise ManagerError(
                f"worker {task_index} accepted candidate region drift"
            ) from error
    if accepted_updates and candidates[-1]["group_flat_sha256"] != current_flat_sha:
        raise ManagerError(f"worker {task_index} terminal flat is not final accepted candidate")
    credential_scan = result.get("credential_environment_scan")
    if (
        not isinstance(credential_scan, Mapping)
        or credential_scan.get("forbidden_name_matches") != 0
        or credential_scan.get("forbidden_value_pattern_matches") != 0
    ):
        raise ManagerError(f"worker {task_index} credential scan did not pass")
    timing = result.get("timing")
    if not isinstance(timing, Mapping):
        raise ManagerError(f"worker {task_index} timing receipt missing")
    timing_values = {
        name: _finite_nonnegative(timing.get(name), f"worker-{task_index}.{name}")
        for name in ("search_seconds", "ranking_seconds", "total_seconds")
    }
    if (
        timing_values["total_seconds"] <= 0
        or timing_values["total_seconds"]
        < timing_values["search_seconds"] + timing_values["ranking_seconds"]
    ):
        raise ManagerError(f"worker {task_index} timing relation drift")
    return {
        "task_index": task_index, "result": result_identity, "flat": flat_identity,
        "bundle": bundle_identity,
        "accepted_candidate_count": accepted_updates,
        "peak_rss_bytes": peak_rss,
        "base_full_search_mean_ce": base_mean,
        "terminal_full_search_mean_ce": terminal_mean,
        "self_calibrated_full_search_improvement": improvement,
        "acceptance_gate_attempts": gate_attempts,
        "acceptance_gate_passes": gate_passes,
        "accepted_updates": accepted_updates,
        "total_physical_model_forwards": total_physical,
        "worker_total_seconds": timing_values["total_seconds"],
        "environment": dict(environment),
    }


def _content_record(path: Path) -> Mapping[str, object]:
    identity = hash_file(Path(path))
    return {
        "sha256": identity["sha256"],
        "size_bytes": int(identity["size_bytes"]),
    }


def _canonical_record_identity(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = canonical_json_bytes(value)
    return {
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _content_identity_fields(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    sha256 = value.get("sha256")
    size_bytes = value.get("size_bytes")
    if (
        not isinstance(sha256, str)
        or _SHA256.fullmatch(sha256) is None
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes <= 0
    ):
        return None
    return {"sha256": sha256, "size_bytes": size_bytes}


def _predecessor_terminal_absence_value(
    predecessor_package_root: Path,
    *,
    wave_index: int,
) -> Mapping[str, object]:
    predecessor_package_root = Path(predecessor_package_root).resolve()
    predecessor_wave_root = (
        predecessor_package_root
        / "runtime/waves"
        / f"wave-{wave_index:08d}"
    )
    forbidden_wave_artifacts = (
        "LAUNCH_MANIFEST.json",
        "CLOUD_DOWNLOADS_COMPLETE.json",
        "CLOUD_CLEANUP_COMPLETE.json",
        "CLOUD_DETERMINISTIC_FAILURE_COMPLETE.json",
        "REAL_PIPELINE_DETERMINISTIC_FAILURE.json",
        "REAL_PIPELINE_COMPLETE.json",
        "COLLECTION_COMPLETE.json",
        "MPS_DECISION.json",
        "PROMOTION_COMPLETE.json",
        "orchestration/DETERMINISTIC_FAILURE_DECISION.json",
    )
    present = [
        relative
        for relative in forbidden_wave_artifacts
        if (predecessor_wave_root / relative).exists()
    ]
    if present:
        raise ManagerError(
            f"predecessor post-probe terminal artifact unexpectedly exists: {present}"
        )
    collected_files = []
    collected_root = predecessor_wave_root / "collected"
    if collected_root.exists():
        collected_files = sorted(
            path.relative_to(predecessor_wave_root).as_posix()
            for path in collected_root.rglob("*")
            if path.is_file()
        )
    if collected_files:
        raise ManagerError("predecessor collection artifacts unexpectedly exist")
    extra_download_entries = sorted(
        path.relative_to(predecessor_wave_root).as_posix()
        for path in (predecessor_wave_root / "downloads").glob("worker-*")
        if path.name != "worker-000"
    )
    if extra_download_entries:
        raise ManagerError("predecessor fanout worker download artifacts unexpectedly exist")
    lineage_root = predecessor_package_root / "runtime/authority/lineage"
    required_empty_lineage_directories = (
        "collections",
        "shortlists",
        "decisions",
        "promotion_intents",
        "deltas",
        "replacement_group_flats",
        "promoted_checkpoints",
        "failures",
    )
    nonempty_lineage = {
        name: sorted(
            path.relative_to(lineage_root).as_posix()
            for path in (lineage_root / name).rglob("*")
            if path.is_file()
        )
        for name in required_empty_lineage_directories
    }
    nonempty_lineage = {
        name: paths for name, paths in nonempty_lineage.items() if paths
    }
    if nonempty_lineage:
        raise ManagerError(
            f"predecessor collection/promotion lineage artifacts exist: {nonempty_lineage}"
        )
    head_files = sorted((lineage_root / "heads").glob("*.head.json"))
    checkpoint_files = sorted(
        (predecessor_package_root / "runtime/authority/checkpoints").glob("*.pt")
    )
    if (
        len(head_files) != 1
        or not head_files[0].name.startswith("00000000-")
        or len(checkpoint_files) != 1
    ):
        raise ManagerError("predecessor authority progressed beyond generation zero")
    return {
        "schema": "hfw_w560_predecessor_prefanout_terminal_absence_v1",
        "forbidden_wave_artifacts_absent": list(forbidden_wave_artifacts),
        "collected_files": [],
        "fanout_worker_download_entries": [],
        "required_empty_lineage_directories": list(
            required_empty_lineage_directories
        ),
        "authoritative_generation0_head": _content_record(head_files[0]),
        "authoritative_base_checkpoint": _content_record(checkpoint_files[0]),
        "authoritative_head_file_count": 1,
        "authority_checkpoint_file_count": 1,
        "checkpoint_or_head_append_count": 0,
        "local_verify_started": False,
    }


def _validate_predecessor_recovery_failure(
    evidence: object,
    *,
    campaign_sha256: str,
    snapshot_manifest_sha256: str,
    wave_index: int,
    original_task0_sandbox: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(evidence, Mapping):
        raise ManagerError("predecessor recovery failure evidence missing")
    required_top_level = {
        "schema": "hfw_w560_predecessor_recovery_failure_evidence_v1",
        "status": "PREDECESSOR_PROBE_LABEL_DRIFT_CLEANED_TERMINAL",
        "campaign_sha256": campaign_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "wave_index": wave_index,
        "task0_model_command_count": 0,
        "fanout_worker_sandbox_create_count": 0,
        "local_verify_started": False,
        "checkpoint_or_head_append_count": 0,
        "task0_cloud_search_rerun": False,
        "failed_probe_worker_command_permitted": False,
        "cleanup_terminal": True,
    }
    if any(evidence.get(name) != value for name, value in required_top_level.items()):
        raise ManagerError("predecessor recovery failure disclosure drift")
    predecessor_package_root = Path(str(evidence.get("package_root"))).resolve()
    predecessor_wave_root = Path(str(evidence.get("wave_root"))).resolve()
    if predecessor_wave_root != (
        predecessor_package_root / "runtime/waves" / f"wave-{wave_index:08d}"
    ):
        raise ManagerError("predecessor recovery failure path binding drift")
    terminal_absence = _predecessor_terminal_absence_value(
        predecessor_package_root, wave_index=wave_index,
    )
    if evidence.get("terminal_absence") != terminal_absence:
        raise ManagerError("predecessor prefanout terminal absence proof drift")
    predecessor_paths = {
        "host_runtime_manifest": (
            predecessor_package_root / "runtime/config/HOST_RUNTIME_MANIFEST.json"
        ),
        "real_pipeline_intent": predecessor_wave_root / "REAL_PIPELINE_INTENT.json",
        "run_config": predecessor_wave_root / "orchestration/RUN_CONFIG.json",
        "recovered_task0_provenance": (
            predecessor_package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
        ),
    }
    for name, predecessor_path in predecessor_paths.items():
        identity = evidence.get(name)
        if (
            not isinstance(identity, Mapping)
            or not isinstance(identity.get("sha256"), str)
            or _SHA256.fullmatch(str(identity["sha256"])) is None
            or not isinstance(identity.get("size_bytes"), int)
            or isinstance(identity.get("size_bytes"), bool)
            or int(identity["size_bytes"]) <= 0
            or _content_record(predecessor_path) != identity
        ):
            raise ManagerError(f"predecessor recovery {name} identity drift")
    host_runtime, _ = read_json(predecessor_paths["host_runtime_manifest"])
    pipeline_intent, _ = read_json(predecessor_paths["real_pipeline_intent"])
    run_config, _ = read_json(predecessor_paths["run_config"])
    predecessor_recovery, _ = read_json(
        predecessor_paths["recovered_task0_provenance"]
    )
    recovered_identity = evidence["recovered_task0_provenance"]
    if (
        pipeline_intent.get("campaign_sha256") != campaign_sha256
        or pipeline_intent.get("wave_index") != wave_index
        or pipeline_intent.get("host_runtime_manifest_sha256")
        != evidence["host_runtime_manifest"]["sha256"]
        or pipeline_intent.get("recovered_task0_provenance", {}).get("sha256")
        != recovered_identity["sha256"]
        or pipeline_intent.get("recovered_task0_provenance", {}).get("size_bytes")
        != recovered_identity["size_bytes"]
        or pipeline_intent.get("task0_model_command_permitted") is not False
        or run_config.get("campaign_sha256") != campaign_sha256
        or Path(str(run_config.get("package_root"))).resolve()
        != predecessor_package_root
        or Path(str(run_config.get("wave_root"))).resolve()
        != predecessor_wave_root
        or run_config.get("recovered_task0_provenance", {}).get("sha256")
        != recovered_identity["sha256"]
        or run_config.get("recovered_task0_provenance", {}).get("size_bytes")
        != recovered_identity["size_bytes"]
        or predecessor_recovery.get("source", {}).get("task0_sandbox")
        != original_task0_sandbox
        or host_runtime.get("campaign", {}).get("sha256") != campaign_sha256
        or host_runtime.get("snapshot_manifest", {}).get("sha256")
        != snapshot_manifest_sha256
    ):
        raise ManagerError("predecessor host/pipeline/run-config cross-binding drift")

    chain = evidence.get("receipt_chain")
    if not isinstance(chain, list) or len(chain) != 20:
        raise ManagerError("predecessor recovery receipt chain length drift")
    previous_sha256 = None
    for index, record in enumerate(chain):
        if (
            not isinstance(record, Mapping)
            or record.get("schema")
            != "hfw_w560_daytona_manager_mutation_receipt_v1"
            or record.get("campaign_sha256") != campaign_sha256
            or record.get("receipt_index") != index
            or record.get("previous_receipt_sha256") != previous_sha256
        ):
            raise ManagerError("predecessor recovery receipt chain drift")
        previous_sha256 = str(_canonical_record_identity(record)["sha256"])
    expected_chain_head = _canonical_record_identity(chain[-1])
    if (
        evidence.get("receipt_chain_length") != len(chain)
        or evidence.get("receipt_chain_head") != expected_chain_head
    ):
        raise ManagerError("predecessor recovery receipt chain head drift")
    expected_events = [
        "MANAGER_RUN_STARTED",
        "RECOVERED_SOURCE_TASK0_EXACT_ID_ABSENT",
        "CAMPAIGN_SANDBOX_ABSENCE_RECONCILED",
        "CLEANUP_TERMINAL",
        "RECOVERY_PRECLOUD_ZERO_RESOURCE_GATE_PASSED",
        "SNAPSHOT_CREATE_INTENT",
        "SNAPSHOT_CREATED",
        "RECOVERED_AND_REVALIDATED_NO_RERUN",
        "TASK_SANDBOX_ABSENCE_RECONCILED",
        "PROBE_SANDBOX_CREATE_INTENT",
        "MANAGER_RUN_FAILED",
        "SANDBOX_DELETE_INTENT",
        "SANDBOX_DELETE_ACCEPTED",
        "SANDBOX_DELETE_VISIBILITY_STALE",
        "SANDBOX_EXACT_ID_ABSENT",
        "SANDBOX_DELETED",
        "SNAPSHOT_DELETE_INTENT",
        "SNAPSHOT_DELETED",
        "CAMPAIGN_SANDBOX_ABSENCE_RECONCILED",
        "CLEANUP_TERMINAL",
    ]
    if [record.get("event") for record in chain] != expected_events:
        raise ManagerError("predecessor recovery terminal event sequence drift")
    external_journal = ReceiptJournal(
        predecessor_wave_root / "orchestration/receipts", campaign_sha256,
    )
    if [dict(record) for record in external_journal.records] != chain:
        raise ManagerError("preserved predecessor receipt chain identity drift")

    probe_labels = _probe_labels(campaign_sha256, wave_index)
    probe_intent = chain[9]
    failure = chain[10]
    delete_intent = chain[11]
    delete_absent = chain[14]
    deleted = chain[15]
    snapshot_deleted = chain[17]
    campaign_absent = chain[18]
    cleanup = chain[19]
    source_exact_absent = chain[1]
    precloud_pass = chain[4]
    created_snapshot = chain[6].get("snapshot")
    if (
        source_exact_absent.get("event")
        != "RECOVERED_SOURCE_TASK0_EXACT_ID_ABSENT"
        or source_exact_absent.get("sandbox_id")
        != original_task0_sandbox.get("id")
        or source_exact_absent.get("exact_id_not_found_confirmed") is not True
        or source_exact_absent.get("source_label_list_used_as_substitute") is not False
    ):
        raise ManagerError("predecessor original task0 exact-ID absence drift")
    if (
        probe_intent.get("event") != "PROBE_SANDBOX_CREATE_INTENT"
        or probe_intent.get("labels") != probe_labels
        or probe_intent.get("worker_command_permitted") is not False
        or probe_intent.get("sandbox_name")
        != _probe_name(campaign_sha256, wave_index)
    ):
        raise ManagerError("predecessor probe create intent drift")
    if (
        failure.get("event") != "MANAGER_RUN_FAILED"
        or failure.get("error") != "build-evidence quota probe labels drift"
        or failure.get("classification")
        != "UNCLASSIFIED_OR_INFRASTRUCTURE_FAILURE"
        or evidence.get("failure_receipt") != _canonical_record_identity(failure)
    ):
        raise ManagerError("predecessor probe-label failure receipt drift")
    failed_probe = evidence.get("failed_probe_sandbox")
    failed_probe_labels = (
        {} if not isinstance(failed_probe, Mapping)
        else failed_probe.get("labels", {})
    )
    platform_added_labels = (
        {} if not isinstance(failed_probe_labels, Mapping)
        else {
            name: value
            for name, value in failed_probe_labels.items()
            if name not in probe_labels
        }
    )
    if (
        not isinstance(failed_probe, Mapping)
        or delete_intent.get("event") != "SANDBOX_DELETE_INTENT"
        or delete_intent.get("reason") != "FAILURE"
        or delete_intent.get("sandbox") != failed_probe
        or evidence.get("probe_delete_intent_receipt")
        != _canonical_record_identity(delete_intent)
        or failed_probe.get("name") != _probe_name(campaign_sha256, wave_index)
        or not isinstance(failed_probe.get("id"), str)
        or not failed_probe.get("id")
        or not _contains_probe_labels(failed_probe.get("labels"), probe_labels)
        or not platform_added_labels
        or any(str(name).startswith("hfw_") for name in platform_added_labels)
        or evidence.get("platform_added_probe_labels") != platform_added_labels
        or failed_probe.get("network_block_all") is not True
        or failed_probe.get("public") is not False
        or not isinstance(created_snapshot, Mapping)
        or failed_probe.get("snapshot") != created_snapshot.get("name")
        or failed_probe.get("cpu") != created_snapshot.get("cpu")
        or failed_probe.get("memory_gib") != created_snapshot.get("memory_gib")
        or failed_probe.get("disk_gib") != created_snapshot.get("disk_gib")
        or failed_probe.get("gpu") != created_snapshot.get("gpu")
        or failed_probe.get("target") != run_config.get("target")
        or failed_probe.get("auto_stop_interval")
        != math.ceil(int(run_config.get("worker_timeout_seconds", -1)) / 60) + 30
        or failed_probe.get("auto_delete_interval") != 0
        or probe_intent.get("snapshot_id") != created_snapshot.get("id")
        or probe_intent.get("snapshot_name") != created_snapshot.get("name")
    ):
        raise ManagerError("predecessor failed probe full identity drift")
    failed_probe_id = str(failed_probe["id"])
    if (
        delete_absent.get("event") != "SANDBOX_EXACT_ID_ABSENT"
        or delete_absent.get("sandbox_id") != failed_probe_id
        or delete_absent.get("reason") != "FAILURE"
        or delete_absent.get("exact_id_not_found_confirmed") is not True
        or evidence.get("probe_exact_id_absent_receipt")
        != _canonical_record_identity(delete_absent)
        or deleted.get("event") != "SANDBOX_DELETED"
        or deleted.get("sandbox") != failed_probe
        or deleted.get("reason") != "FAILURE"
        or deleted.get("exact_id_not_found_confirmed") is not True
        or evidence.get("probe_deleted_receipt")
        != _canonical_record_identity(deleted)
    ):
        raise ManagerError("predecessor failed probe exact-ID deletion drift")
    if (
        snapshot_deleted.get("event") != "SNAPSHOT_DELETED"
        or snapshot_deleted.get("reason") != "FAILURE"
        or snapshot_deleted.get("exact_name_not_found_confirmed") is not True
        or snapshot_deleted.get("snapshot") != created_snapshot
        or created_snapshot.get("name")
        != _snapshot_name(campaign_sha256, wave_index)
        or evidence.get("snapshot_deleted_receipt")
        != _canonical_record_identity(snapshot_deleted)
        or campaign_absent.get("event")
        != "CAMPAIGN_SANDBOX_ABSENCE_RECONCILED"
        or campaign_absent.get("reason") != "FAILURE"
        or campaign_absent.get("final_remaining_count") != 0
        or campaign_absent.get("exact_campaign_labels") is not True
    ):
        raise ManagerError("predecessor recovery resource absence receipt drift")
    required_cleanup = {
        "event": "CLEANUP_TERMINAL",
        "reason": "FAILURE",
        "failure_count": 0,
        "failures": [],
        "initial_campaign_sandbox_count": 1,
        "unique_campaign_sandboxes_discovered": 1,
        "unique_campaign_sandbox_ids_discovered": [failed_probe_id],
        "exact_id_not_found_confirmation_count": 1,
        "final_remaining_count": 0,
        "final_remaining_ids": [],
        "final_relist_succeeded": True,
        "campaign_sandbox_count": 0,
        "exact_campaign_label_relist_passed": True,
        "snapshot_present": False,
        "snapshot_relist_succeeded": True,
        "snapshot_exact_name_not_found_confirmed": True,
    }
    if (
        any(cleanup.get(name) != value for name, value in required_cleanup.items())
        or evidence.get("cleanup_terminal_receipt")
        != _canonical_record_identity(cleanup)
    ):
        raise ManagerError("predecessor recovery cleanup terminal drift")

    precloud = evidence.get("precloud_zero_resource_terminal")
    if not isinstance(precloud, Mapping):
        raise ManagerError("predecessor precloud terminal missing")
    precloud_value = precloud.get("value")
    precloud_identity = precloud.get("identity")
    if (
        not isinstance(precloud_value, Mapping)
        or precloud_identity != _canonical_record_identity(precloud_value)
        or precloud_value.get("schema")
        != "hfw_w560_recovery_precloud_zero_resource_gate_v1"
        or precloud_value.get("status")
        != "PASS_EXACT_SOURCE_ID_AND_CAMPAIGN_RESOURCES_ABSENT"
        or precloud_value.get("campaign_sha256") != campaign_sha256
        or precloud_value.get("snapshot_manifest_sha256")
        != snapshot_manifest_sha256
        or precloud_value.get("wave_index") != wave_index
        or precloud_value.get("source_task0_sandbox_id")
        != original_task0_sandbox.get("id")
        or precloud_value.get("source_task0_exact_id_not_found_confirmed") is not True
        or precloud_value.get("final_campaign_sandboxes_remaining") != 0
        or precloud_value.get("snapshot_present") is not False
        or precloud_value.get("snapshot_exact_name_not_found_confirmed") is not True
        or precloud_value.get("sandbox_snapshot_or_probe_create_before_gate") is not False
        or precloud_value.get("task0_model_command_started") is not False
        or precloud_value.get("checkpoint_or_head_writes") != 0
        or precloud_value.get("recovery_provenance", {}).get("sha256")
        != recovered_identity["sha256"]
        or precloud_value.get("recovery_provenance", {}).get("size_bytes")
        != recovered_identity["size_bytes"]
        or precloud_pass.get("event")
        != "RECOVERY_PRECLOUD_ZERO_RESOURCE_GATE_PASSED"
        or precloud_pass.get("terminal") != precloud_identity
        or precloud_pass.get("no_cloud_create_before_gate") is not True
        or precloud_pass.get("source_task0_exact_id", {}).get("sandbox_id")
        != original_task0_sandbox.get("id")
        or precloud_pass.get("source_task0_exact_id", {}).get(
            "exact_id_not_found_confirmed"
        ) is not True
    ):
        raise ManagerError("predecessor precloud zero-resource terminal drift")

    required_ids = evidence.get("required_exact_id_reconciliation")
    expected_required_ids = [
        {
            "role": "original-source-task0",
            "sandbox": dict(original_task0_sandbox),
        },
        {
            "role": "failed-build-evidence-quota-probe",
            "sandbox": dict(failed_probe),
        },
    ]
    if required_ids != expected_required_ids:
        raise ManagerError("predecessor dual exact-ID reconciliation binding drift")
    if original_task0_sandbox.get("id") == failed_probe_id:
        raise ManagerError("predecessor recovery sandbox identities are not distinct")
    recovered = chain[7]
    if (
        recovered.get("event") != "RECOVERED_AND_REVALIDATED_NO_RERUN"
        or recovered.get("task_index") != 0
        or recovered.get("sandbox_create_permitted") is not False
        or recovered.get("worker_command_permitted") is not False
        or recovered.get("cloud_search_rerun") is not False
    ):
        raise ManagerError("predecessor recovered task0 no-rerun receipt drift")
    return dict(evidence)


def _predecessor_recovery_failure_value(
    predecessor_package_root: Path,
    *,
    campaign_sha256: str,
    snapshot_manifest_sha256: str,
    wave_index: int,
    original_task0_sandbox: Mapping[str, object],
) -> Mapping[str, object]:
    predecessor_package_root = Path(predecessor_package_root).resolve()
    predecessor_wave_root = (
        predecessor_package_root
        / "runtime/waves"
        / f"wave-{wave_index:08d}"
    )
    predecessor_campaign, predecessor_identities = validate_worker_snapshot(
        predecessor_wave_root / "snapshot"
    )
    if (
        predecessor_identities["campaign"]["sha256"] != campaign_sha256
        or predecessor_identities["snapshot_manifest"]["sha256"]
        != snapshot_manifest_sha256
        or int(predecessor_campaign["wave_index"]) != wave_index
    ):
        raise ManagerError("predecessor recovery campaign/snapshot binding drift")
    predecessor_recovery_path = (
        predecessor_package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
    )
    predecessor_recovery, _ = read_json(predecessor_recovery_path)
    if (
        predecessor_recovery.get("schema")
        != "hfw_w560_recovered_task0_provenance_v1"
        or predecessor_recovery.get("status")
        != "RECOVERED_AND_REVALIDATED_NO_RERUN"
        or predecessor_recovery.get("source", {}).get("task0_sandbox")
        != original_task0_sandbox
        or predecessor_recovery.get("immutable_binding", {}).get("campaign_sha256")
        != campaign_sha256
    ):
        raise ManagerError("predecessor recovered task0 provenance drift")
    journal = ReceiptJournal(
        predecessor_wave_root / "orchestration/receipts", campaign_sha256,
    )
    records = [dict(record) for record in journal.records]
    if len(records) != 20:
        raise ManagerError("predecessor recovery receipt chain is not terminal")
    precloud_path = (
        predecessor_wave_root / "RECOVERY_PRECLOUD_ZERO_RESOURCE_COMPLETE.json"
    )
    precloud_value, _ = read_json(precloud_path)
    failure = records[10]
    delete_intent = records[11]
    failed_probe = delete_intent.get("sandbox")
    if not isinstance(failed_probe, Mapping):
        raise ManagerError("predecessor failed probe identity missing")
    terminal_absence = _predecessor_terminal_absence_value(
        predecessor_package_root, wave_index=wave_index,
    )
    evidence = {
        "schema": "hfw_w560_predecessor_recovery_failure_evidence_v1",
        "status": "PREDECESSOR_PROBE_LABEL_DRIFT_CLEANED_TERMINAL",
        "package_root": str(predecessor_package_root),
        "wave_root": str(predecessor_wave_root),
        "campaign_sha256": campaign_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "wave_index": wave_index,
        "host_runtime_manifest": _content_record(
            predecessor_package_root / "runtime/config/HOST_RUNTIME_MANIFEST.json"
        ),
        "real_pipeline_intent": _content_record(
            predecessor_wave_root / "REAL_PIPELINE_INTENT.json"
        ),
        "run_config": _content_record(
            predecessor_wave_root / "orchestration/RUN_CONFIG.json"
        ),
        "recovered_task0_provenance": _content_record(
            predecessor_recovery_path
        ),
        "precloud_zero_resource_terminal": {
            "identity": _canonical_record_identity(precloud_value),
            "value": precloud_value,
        },
        "receipt_chain": records,
        "receipt_chain_length": len(records),
        "receipt_chain_head": _canonical_record_identity(records[-1]),
        "failure_receipt": _canonical_record_identity(failure),
        "probe_delete_intent_receipt": _canonical_record_identity(delete_intent),
        "probe_exact_id_absent_receipt": _canonical_record_identity(records[14]),
        "probe_deleted_receipt": _canonical_record_identity(records[15]),
        "snapshot_deleted_receipt": _canonical_record_identity(records[17]),
        "cleanup_terminal_receipt": _canonical_record_identity(records[19]),
        "terminal_absence": terminal_absence,
        "failed_probe_sandbox": dict(failed_probe),
        "platform_added_probe_labels": {
            name: value
            for name, value in failed_probe["labels"].items()
            if name not in _probe_labels(campaign_sha256, wave_index)
        },
        "required_exact_id_reconciliation": [
            {
                "role": "original-source-task0",
                "sandbox": dict(original_task0_sandbox),
            },
            {
                "role": "failed-build-evidence-quota-probe",
                "sandbox": dict(failed_probe),
            },
        ],
        "task0_model_command_count": 0,
        "fanout_worker_sandbox_create_count": 0,
        "local_verify_started": False,
        "checkpoint_or_head_append_count": 0,
        "task0_cloud_search_rerun": False,
        "failed_probe_worker_command_permitted": False,
        "cleanup_terminal": True,
    }
    return _validate_predecessor_recovery_failure(
        evidence,
        campaign_sha256=campaign_sha256,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        wave_index=wave_index,
        original_task0_sandbox=original_task0_sandbox,
    )


def _immediate_predecessor_terminal_absence_value(
    predecessor_package_root: Path,
    *,
    wave_index: int,
) -> Mapping[str, object]:
    predecessor_package_root = Path(predecessor_package_root).resolve()
    predecessor_wave_root = (
        predecessor_package_root
        / "runtime/waves"
        / f"wave-{wave_index:08d}"
    )
    forbidden_wave_artifacts = (
        "LAUNCH_MANIFEST.json",
        "CLOUD_DOWNLOADS_COMPLETE.json",
        "CLOUD_CLEANUP_COMPLETE.json",
        "CLOUD_DETERMINISTIC_FAILURE_COMPLETE.json",
        "REAL_PIPELINE_DETERMINISTIC_FAILURE.json",
        "REAL_PIPELINE_COMPLETE.json",
        "COLLECTION_COMPLETE.json",
        "MPS_DECISION.json",
        "PROMOTION_COMPLETE.json",
        "orchestration/DETERMINISTIC_FAILURE_DECISION.json",
    )
    present = [
        relative
        for relative in forbidden_wave_artifacts
        if (predecessor_wave_root / relative).exists()
    ]
    if present:
        raise ManagerError(
            "immediate predecessor post-fanout artifact unexpectedly exists: "
            f"{present}"
        )

    expected_downloads = (
        "downloads/worker-000/CPU_SEARCH_RESULT.json",
        "downloads/worker-000/trajectory_final_group_flat.f32le",
    )
    download_root = predecessor_wave_root / "downloads"
    observed_downloads = tuple(sorted(
        path.relative_to(predecessor_wave_root).as_posix()
        for path in download_root.rglob("*")
        if path.is_file()
    ))
    if observed_downloads != expected_downloads:
        raise ManagerError(
            "immediate predecessor download set is not exactly imported task0"
        )

    collected_root = predecessor_wave_root / "collected"
    collected_files = [] if not collected_root.exists() else sorted(
        path.relative_to(predecessor_wave_root).as_posix()
        for path in collected_root.rglob("*")
        if path.is_file()
    )
    if collected_files:
        raise ManagerError(
            "immediate predecessor collection artifacts unexpectedly exist"
        )

    lineage_root = predecessor_package_root / "runtime/authority/lineage"
    required_empty_lineage_directories = (
        "collections",
        "shortlists",
        "decisions",
        "promotion_intents",
        "deltas",
        "replacement_group_flats",
        "promoted_checkpoints",
        "failures",
    )
    nonempty_lineage = {
        name: sorted(
            path.relative_to(lineage_root).as_posix()
            for path in (lineage_root / name).rglob("*")
            if path.is_file()
        )
        for name in required_empty_lineage_directories
    }
    nonempty_lineage = {
        name: paths for name, paths in nonempty_lineage.items() if paths
    }
    if nonempty_lineage:
        raise ManagerError(
            "immediate predecessor collection/promotion lineage artifacts exist: "
            f"{nonempty_lineage}"
        )

    head_files = sorted((lineage_root / "heads").glob("*.head.json"))
    local_checkpoint_files = sorted(
        path
        for path in (
            predecessor_package_root / "runtime/authority/checkpoints"
        ).glob("*")
        if path.is_file()
    )
    if (
        len(head_files) != 1
        or not head_files[0].name.startswith("00000000-")
        or local_checkpoint_files
    ):
        raise ManagerError(
            "immediate predecessor authority progressed beyond imported generation zero"
        )
    expected_head_sha256 = head_files[0].name.split("-", 1)[1].split(".", 1)[0]
    head, head_identity = read_json(head_files[0], expected_head_sha256)
    checkpoint = head.get("checkpoint")
    if (
        head.get("authority_generation") != 0
        or not isinstance(checkpoint, Mapping)
        or _content_identity_fields(checkpoint) is None
        or not isinstance(checkpoint.get("path"), str)
        or not Path(str(checkpoint["path"])).is_absolute()
    ):
        raise ManagerError(
            "immediate predecessor imported head/checkpoint metadata drift"
        )
    current_path = lineage_root / "CURRENT.json"
    current, current_identity = read_json(current_path)
    if (
        current.get("authority_generation") != 0
        or current.get("record_sha256") != head_identity["sha256"]
        or Path(str(current.get("record_path"))).resolve()
        != head_files[0].resolve()
    ):
        raise ManagerError("immediate predecessor CURRENT pointer drift")

    return {
        "schema": (
            "hfw_w560_immediate_predecessor_prefanout_terminal_absence_v1"
        ),
        "forbidden_wave_artifacts_absent": list(forbidden_wave_artifacts),
        "collected_files": [],
        "download_files": list(expected_downloads),
        "download_file_identities": {
            relative: _content_record(predecessor_wave_root / relative)
            for relative in expected_downloads
        },
        "fanout_worker_download_entries": [],
        "required_empty_lineage_directories": list(
            required_empty_lineage_directories
        ),
        "authoritative_generation0_head": _content_record(head_files[0]),
        "authoritative_generation0_head_checkpoint": dict(checkpoint),
        "authoritative_current_pointer": _content_record(current_path),
        "authoritative_head_file_count": 1,
        "local_authority_checkpoint_file_count": 0,
        "checkpoint_or_head_append_count": 0,
        "local_verify_started": False,
    }


def _validate_immediate_predecessor_quota_failure(
    evidence: object,
    *,
    campaign_sha256: str,
    snapshot_manifest_sha256: str,
    wave_index: int,
    original_task0_sandbox: Mapping[str, object],
    earlier_failed_probe_sandbox: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(evidence, Mapping):
        raise ManagerError("immediate predecessor quota failure evidence missing")
    required_top_level = {
        "schema": "hfw_w560_immediate_predecessor_quota_failure_evidence_v1",
        "status": "PREDECESSOR_REGION_QUOTA_AUTH_CONTEXT_FAILURE_CLEANED_TERMINAL",
        "campaign_sha256": campaign_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "wave_index": wave_index,
        "task0_model_command_count": 0,
        "fanout_worker_sandbox_create_count": 0,
        "local_verify_started": False,
        "checkpoint_or_head_append_count": 0,
        "task0_cloud_search_rerun": False,
        "failed_probe_worker_command_permitted": False,
        "cleanup_terminal": True,
    }
    if any(evidence.get(name) != value for name, value in required_top_level.items()):
        raise ManagerError("immediate predecessor quota failure disclosure drift")

    predecessor_package_root = Path(str(evidence.get("package_root"))).resolve()
    predecessor_wave_root = Path(str(evidence.get("wave_root"))).resolve()
    if predecessor_wave_root != (
        predecessor_package_root
        / "runtime/waves"
        / f"wave-{wave_index:08d}"
    ):
        raise ManagerError("immediate predecessor quota failure path binding drift")
    terminal_absence = _immediate_predecessor_terminal_absence_value(
        predecessor_package_root, wave_index=wave_index,
    )
    if evidence.get("terminal_absence") != terminal_absence:
        raise ManagerError("immediate predecessor terminal absence proof drift")

    predecessor_paths = {
        "host_runtime_manifest": (
            predecessor_package_root / "runtime/config/HOST_RUNTIME_MANIFEST.json"
        ),
        "real_pipeline_intent": predecessor_wave_root / "REAL_PIPELINE_INTENT.json",
        "run_config": predecessor_wave_root / "orchestration/RUN_CONFIG.json",
        "recovered_task0_provenance": (
            predecessor_package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
        ),
    }
    for name, predecessor_path in predecessor_paths.items():
        identity = _content_identity_fields(evidence.get(name))
        if identity is None or _content_record(predecessor_path) != identity:
            raise ManagerError(
                f"immediate predecessor quota failure {name} identity drift"
            )

    recovered = validate_recovered_task0_provenance(
        predecessor_package_root,
        wave_index=wave_index,
        _allow_external_v2=True,
    )
    if (
        recovered is None
        or recovered.get("schema") != RECOVERED_TASK0_PROVENANCE_SCHEMA_V2
        or _content_identity_fields(recovered.get("identity"))
        != evidence["recovered_task0_provenance"]
        or recovered.get("source", {}).get("task0_sandbox")
        != original_task0_sandbox
        or recovered.get("predecessor_recovery_failure", {}).get(
            "failed_probe_sandbox"
        ) != earlier_failed_probe_sandbox
    ):
        raise ManagerError(
            "immediate predecessor external v2 recovery provenance drift"
        )

    host_runtime, _ = read_json(predecessor_paths["host_runtime_manifest"])
    pipeline_intent, _ = read_json(predecessor_paths["real_pipeline_intent"])
    run_config, _ = read_json(predecessor_paths["run_config"])
    recovered_identity = evidence["recovered_task0_provenance"]
    predecessor_failure = recovered["predecessor_recovery_failure"]
    campaign_identity = _content_record(
        predecessor_wave_root / "snapshot/CAMPAIGN.json"
    )
    snapshot_identity = _content_record(
        predecessor_wave_root / "snapshot/SNAPSHOT_MANIFEST.json"
    )
    inherited_ids = [
        str(original_task0_sandbox.get("id")),
        str(earlier_failed_probe_sandbox.get("id")),
    ]
    if (
        host_runtime.get("schema")
        != "hfw_w560_distributed_host_runtime_manifest_v2"
        or host_runtime.get("status") != "SEALED_NOT_LAUNCHED"
        or _content_identity_fields(host_runtime.get("campaign"))
        != campaign_identity
        or campaign_identity["sha256"] != campaign_sha256
        or _content_identity_fields(host_runtime.get("snapshot_manifest"))
        != snapshot_identity
        or snapshot_identity["sha256"] != snapshot_manifest_sha256
        or _content_identity_fields(host_runtime.get("recovered_task0_provenance"))
        != recovered_identity
        or pipeline_intent.get("schema")
        != "hfw_w560_real_wave_pipeline_intent_v1"
        or pipeline_intent.get("status")
        != "SEALED_BEFORE_MANAGER_SUBPROCESS"
        or pipeline_intent.get("campaign_sha256") != campaign_sha256
        or pipeline_intent.get("wave_index") != wave_index
        or pipeline_intent.get("host_runtime_manifest_sha256")
        != evidence["host_runtime_manifest"]["sha256"]
        or _content_identity_fields(pipeline_intent.get("recovered_task0_provenance"))
        != recovered_identity
        or pipeline_intent.get("recovery_predecessor_receipt_chain_head")
        != predecessor_failure["receipt_chain_head"]
        or pipeline_intent.get("recovery_predecessor_failure_receipt")
        != predecessor_failure["failure_receipt"]
        or pipeline_intent.get("recovery_predecessor_cleanup_terminal_receipt")
        != predecessor_failure["cleanup_terminal_receipt"]
        or pipeline_intent.get("recovery_required_exact_sandbox_ids")
        != inherited_ids
        or pipeline_intent.get(
            "recovery_dual_exact_id_gate_required_before_cloud_create"
        ) is not True
        or pipeline_intent.get("task0_model_command_permitted") is not False
        or pipeline_intent.get("upload_snapshot_staging")
        != host_runtime.get("upload_snapshot_staging")
        or run_config.get("campaign_sha256") != campaign_sha256
        or Path(str(run_config.get("package_root"))).resolve()
        != predecessor_package_root
        or Path(str(run_config.get("wave_root"))).resolve()
        != predecessor_wave_root
        or _content_identity_fields(run_config.get("recovered_task0_provenance"))
        != recovered_identity
    ):
        raise ManagerError(
            "immediate predecessor host/pipeline/run-config cross-binding drift"
        )

    precloud_path = (
        predecessor_wave_root / "RECOVERY_PRECLOUD_ZERO_RESOURCE_COMPLETE.json"
    )
    precloud = evidence.get("precloud_zero_resource_terminal")
    if not isinstance(precloud, Mapping):
        raise ManagerError("immediate predecessor precloud terminal missing")
    precloud_value, _ = read_json(precloud_path)
    if (
        precloud.get("identity") != _content_record(precloud_path)
        or precloud.get("value") != precloud_value
        or precloud_value.get("schema")
        != "hfw_w560_recovery_precloud_zero_resource_gate_v2"
        or precloud_value.get("status")
        != "PASS_DUAL_EXACT_IDS_AND_CAMPAIGN_RESOURCES_ABSENT"
        or precloud_value.get("campaign_sha256") != campaign_sha256
        or precloud_value.get("snapshot_manifest_sha256")
        != snapshot_manifest_sha256
        or precloud_value.get("wave_index") != wave_index
        or precloud_value.get("recovery_provenance", {}).get("sha256")
        != recovered_identity["sha256"]
        or precloud_value.get("recovery_provenance", {}).get("size_bytes")
        != recovered_identity["size_bytes"]
        or precloud_value.get("required_exact_sandbox_ids") != inherited_ids
        or precloud_value.get("source_task0_sandbox_id") != inherited_ids[0]
        or precloud_value.get("source_task0_exact_id_not_found_confirmed")
        is not True
        or precloud_value.get("predecessor_failed_probe_sandbox_id")
        != inherited_ids[1]
        or precloud_value.get(
            "predecessor_failed_probe_exact_id_not_found_confirmed"
        ) is not True
        or precloud_value.get("dual_exact_id_not_found_confirmed") is not True
        or precloud_value.get("final_campaign_sandboxes_remaining") != 0
        or precloud_value.get("snapshot_present") is not False
        or precloud_value.get("snapshot_exact_name_not_found_confirmed")
        is not True
        or precloud_value.get("sandbox_snapshot_or_probe_create_before_gate")
        is not False
        or precloud_value.get("task0_model_command_started") is not False
        or precloud_value.get("checkpoint_or_head_writes") != 0
    ):
        raise ManagerError("immediate predecessor precloud terminal drift")

    build_evidence = evidence.get("build_evidence")
    evidence_root = predecessor_wave_root / "orchestration/build-evidence"
    pip_freeze_path = evidence_root / "pip-freeze.txt"
    wheel_sha256_path = evidence_root / "wheel-sha256.txt"
    if not isinstance(build_evidence, Mapping):
        raise ManagerError("immediate predecessor build evidence missing")
    wheel_lines = [
        line.strip()
        for line in wheel_sha256_path.read_text().splitlines()
        if line.strip()
    ]
    observed_wheels: dict[str, str] = {}
    for line in wheel_lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or _SHA256.fullmatch(parts[0]) is None:
            raise ManagerError("immediate predecessor wheel evidence malformed")
        observed_wheels[Path(parts[1]).name] = parts[0]
    pip_freeze_lines = [
        line.strip()
        for line in pip_freeze_path.read_text().splitlines()
        if line.strip()
    ]
    sealed_environment = recovered.get("immutable_binding", {}).get("environment")
    required_runtime_lines: set[str] = set()
    if isinstance(sealed_environment, Mapping):
        required_runtime_lines.update({
            f"numpy=={sealed_environment.get('numpy_version')}",
            f"torch=={sealed_environment.get('torch_version')}",
        })
    if (
        build_evidence.get("pip_freeze_file") != _content_record(pip_freeze_path)
        or build_evidence.get("wheel_sha256_file")
        != _content_record(wheel_sha256_path)
        or build_evidence.get("wheel_closure_sha256_by_filename")
        != observed_wheels
        or observed_wheels != EXPECTED_WHEEL_SHA256
        or build_evidence.get("pip_freeze_lines") != pip_freeze_lines
        or not required_runtime_lines.issubset(set(pip_freeze_lines))
    ):
        raise ManagerError("immediate predecessor build evidence drift")

    chain = evidence.get("receipt_chain")
    if not isinstance(chain, list) or len(chain) != 25:
        raise ManagerError("immediate predecessor receipt chain length drift")
    previous_sha256 = None
    for index, record in enumerate(chain):
        if (
            not isinstance(record, Mapping)
            or record.get("schema")
            != "hfw_w560_daytona_manager_mutation_receipt_v1"
            or record.get("campaign_sha256") != campaign_sha256
            or record.get("receipt_index") != index
            or record.get("previous_receipt_sha256") != previous_sha256
        ):
            raise ManagerError("immediate predecessor receipt chain drift")
        previous_sha256 = str(_canonical_record_identity(record)["sha256"])
    expected_chain_head = _canonical_record_identity(chain[-1])
    if (
        evidence.get("receipt_chain_length") != 25
        or evidence.get("receipt_chain_head") != expected_chain_head
    ):
        raise ManagerError("immediate predecessor receipt chain head drift")
    expected_events = [
        "MANAGER_RUN_STARTED",
        "RECOVERED_PREDECESSOR_EXACT_ID_ABSENT",
        "RECOVERED_PREDECESSOR_EXACT_ID_ABSENT",
        "CAMPAIGN_SANDBOX_ABSENCE_RECONCILED",
        "CLEANUP_TERMINAL",
        "RECOVERY_PRECLOUD_ZERO_RESOURCE_GATE_PASSED",
        "SNAPSHOT_CREATE_INTENT",
        "SNAPSHOT_CREATED",
        "RECOVERED_AND_REVALIDATED_NO_RERUN",
        "TASK_SANDBOX_ABSENCE_RECONCILED",
        "PROBE_SANDBOX_CREATE_INTENT",
        "PROBE_SANDBOX_CREATED",
        "SDK_CALL_RETRY",
        "SDK_CALL_RETRY",
        "SDK_CALL_RETRY",
        "SANDBOX_DELETE_INTENT",
        "SANDBOX_DELETE_ACCEPTED",
        "SANDBOX_DELETE_VISIBILITY_STALE",
        "SANDBOX_EXACT_ID_ABSENT",
        "SANDBOX_DELETED",
        "MANAGER_RUN_FAILED",
        "SNAPSHOT_DELETE_INTENT",
        "SNAPSHOT_DELETED",
        "CAMPAIGN_SANDBOX_ABSENCE_RECONCILED",
        "CLEANUP_TERMINAL",
    ]
    if [record.get("event") for record in chain] != expected_events:
        raise ManagerError("immediate predecessor terminal event sequence drift")
    external_journal = ReceiptJournal(
        predecessor_wave_root / "orchestration/receipts", campaign_sha256,
    )
    if [dict(record) for record in external_journal.records] != chain:
        raise ManagerError("immutable immediate predecessor receipt chain drift")

    if (
        chain[1].get("role") != "ORIGINAL_SOURCE_TASK0"
        or chain[1].get("sandbox_id") != inherited_ids[0]
        or chain[1].get("exact_id_not_found_confirmed") is not True
        or chain[2].get("role") != "FAILED_BUILD_EVIDENCE_QUOTA_PROBE"
        or chain[2].get("sandbox_id") != inherited_ids[1]
        or chain[2].get("exact_id_not_found_confirmed") is not True
        or chain[5].get("terminal") != precloud["identity"]
        or chain[5].get("source_task0_exact_id", {}).get("sandbox_id")
        != inherited_ids[0]
        or chain[5].get("predecessor_failed_probe_exact_id", {}).get(
            "sandbox_id"
        ) != inherited_ids[1]
    ):
        raise ManagerError(
            "immediate predecessor inherited exact-ID reconciliation drift"
        )

    snapshot = chain[7].get("snapshot")
    probe_intent = chain[10]
    probe_created = chain[11]
    failed_probe = evidence.get("failed_probe_sandbox")
    probe_labels = _probe_labels(campaign_sha256, wave_index)
    platform_added_labels = (
        {}
        if not isinstance(failed_probe, Mapping)
        or not isinstance(failed_probe.get("labels"), Mapping)
        else {
            name: value
            for name, value in failed_probe["labels"].items()
            if name not in probe_labels
        }
    )
    if (
        not isinstance(snapshot, Mapping)
        or not isinstance(failed_probe, Mapping)
        or probe_intent.get("labels") != probe_labels
        or probe_intent.get("worker_command_permitted") is not False
        or probe_intent.get("sandbox_name")
        != _probe_name(campaign_sha256, wave_index)
        or probe_intent.get("snapshot_id") != snapshot.get("id")
        or probe_intent.get("snapshot_name") != snapshot.get("name")
        or probe_created.get("sandbox") != failed_probe
        or probe_created.get("worker_command_permitted") is not False
        or failed_probe.get("name") != _probe_name(campaign_sha256, wave_index)
        or not isinstance(failed_probe.get("id"), str)
        or not failed_probe.get("id")
        or not _contains_probe_labels(failed_probe.get("labels"), probe_labels)
        or evidence.get("platform_added_probe_labels") != platform_added_labels
        or any(str(name).startswith("hfw_") for name in platform_added_labels)
        or failed_probe.get("network_block_all") is not True
        or failed_probe.get("public") is not False
        or failed_probe.get("snapshot") != snapshot.get("name")
        or failed_probe.get("cpu") != snapshot.get("cpu")
        or failed_probe.get("memory_gib") != snapshot.get("memory_gib")
        or failed_probe.get("disk_gib") != snapshot.get("disk_gib")
        or failed_probe.get("gpu") != snapshot.get("gpu")
        or failed_probe.get("target") != run_config.get("target")
        or failed_probe.get("auto_stop_interval")
        != math.ceil(int(run_config.get("worker_timeout_seconds", -1)) / 60) + 30
        or failed_probe.get("auto_delete_interval") != 0
    ):
        raise ManagerError("immediate predecessor failed probe identity drift")
    failed_probe_id = str(failed_probe["id"])

    retry_path = f"/api/sandbox/{failed_probe_id}/region-quota"
    for attempt, retry in enumerate(chain[12:15]):
        error = retry.get("error")
        if (
            retry.get("attempt") != attempt
            or retry.get("operation") != "read live region quota"
            or not isinstance(error, str)
            or "(403)" not in error
            or "Invalid authentication context" not in error
            or retry_path not in error
        ):
            raise ManagerError(
                "immediate predecessor region-quota 403 retry evidence drift"
            )
    failure = chain[20]
    if (
        failure.get("classification")
        != "UNCLASSIFIED_OR_INFRASTRUCTURE_FAILURE"
        or failure.get("error")
        != "read live region quota exhausted retries: " + str(chain[14]["error"])
        or evidence.get("failure_receipt")
        != _canonical_record_identity(failure)
    ):
        raise ManagerError("immediate predecessor quota failure receipt drift")

    for record in chain[15:20]:
        if record.get("reason") != "BUILD_EVIDENCE_AND_QUOTA_PROBE_COMPLETE":
            raise ManagerError("immediate predecessor probe deletion reason drift")
    if (
        chain[15].get("sandbox") != failed_probe
        or chain[16].get("sandbox") != failed_probe
        or chain[17].get("sandbox_id") != failed_probe_id
        or chain[18].get("sandbox_id") != failed_probe_id
        or chain[18].get("exact_id_not_found_confirmed") is not True
        or chain[19].get("sandbox") != failed_probe
        or chain[19].get("exact_id_not_found_confirmed") is not True
        or evidence.get("probe_created_receipt")
        != _canonical_record_identity(chain[11])
        or evidence.get("probe_exact_id_absent_receipt")
        != _canonical_record_identity(chain[18])
        or evidence.get("probe_deleted_receipt")
        != _canonical_record_identity(chain[19])
    ):
        raise ManagerError("immediate predecessor exact-ID probe deletion drift")
    cleanup = chain[24]
    if (
        chain[21].get("reason") != "FAILURE"
        or chain[21].get("snapshot") != snapshot
        or chain[22].get("reason") != "FAILURE"
        or chain[22].get("snapshot") != snapshot
        or chain[22].get("exact_name_not_found_confirmed") is not True
        or chain[23].get("reason") != "FAILURE"
        or chain[23].get("final_remaining_count") != 0
        or chain[23].get("exact_campaign_labels") is not True
        or cleanup.get("reason") != "FAILURE"
        or cleanup.get("failure_count") != 0
        or cleanup.get("failures") != []
        or cleanup.get("initial_campaign_sandbox_count") != 0
        or cleanup.get("unique_campaign_sandboxes_discovered") != 0
        or cleanup.get("unique_campaign_sandbox_ids_discovered") != []
        or cleanup.get("exact_id_not_found_confirmation_count") != 0
        or cleanup.get("final_remaining_count") != 0
        or cleanup.get("final_remaining_ids") != []
        or cleanup.get("campaign_sandbox_count") != 0
        or cleanup.get("snapshot_present") is not False
        or cleanup.get("snapshot_exact_name_not_found_confirmed") is not True
        or evidence.get("snapshot_deleted_receipt")
        != _canonical_record_identity(chain[22])
        or evidence.get("cleanup_terminal_receipt")
        != _canonical_record_identity(cleanup)
    ):
        raise ManagerError("immediate predecessor terminal cleanup drift")

    recovered_task0 = chain[8]
    forbidden_command_events = [
        record.get("event")
        for record in chain
        if "COMMAND" in str(record.get("event"))
    ]
    if (
        recovered_task0.get("task_index") != 0
        or recovered_task0.get("sandbox_create_permitted") is not False
        or recovered_task0.get("worker_command_permitted") is not False
        or recovered_task0.get("cloud_search_rerun") is not False
        or _content_identity_fields(recovered_task0.get("provenance_receipt"))
        != recovered_identity
        or forbidden_command_events
        or any(record.get("event") == "SANDBOX_CREATED" for record in chain)
    ):
        raise ManagerError("immediate predecessor task0 no-rerun proof drift")
    return dict(evidence)


def _immediate_predecessor_quota_failure_value(
    predecessor_package_root: Path,
    *,
    campaign_sha256: str,
    snapshot_manifest_sha256: str,
    wave_index: int,
    original_task0_sandbox: Mapping[str, object],
    earlier_failed_probe_sandbox: Mapping[str, object],
) -> Mapping[str, object]:
    predecessor_package_root = Path(predecessor_package_root).resolve()
    predecessor_wave_root = (
        predecessor_package_root
        / "runtime/waves"
        / f"wave-{wave_index:08d}"
    )
    predecessor_campaign, predecessor_identities = validate_worker_snapshot(
        predecessor_wave_root / "snapshot"
    )
    if (
        predecessor_identities["campaign"]["sha256"] != campaign_sha256
        or predecessor_identities["snapshot_manifest"]["sha256"]
        != snapshot_manifest_sha256
        or int(predecessor_campaign["wave_index"]) != wave_index
    ):
        raise ManagerError(
            "immediate predecessor campaign/snapshot binding drift"
        )
    predecessor_recovery_path = (
        predecessor_package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
    )
    recovered = validate_recovered_task0_provenance(
        predecessor_package_root,
        wave_index=wave_index,
        _allow_external_v2=True,
    )
    if (
        recovered is None
        or recovered.get("schema") != RECOVERED_TASK0_PROVENANCE_SCHEMA_V2
        or recovered.get("source", {}).get("task0_sandbox")
        != original_task0_sandbox
        or recovered.get("predecessor_recovery_failure", {}).get(
            "failed_probe_sandbox"
        ) != earlier_failed_probe_sandbox
    ):
        raise ManagerError("immediate predecessor v2 recovery binding drift")

    journal = ReceiptJournal(
        predecessor_wave_root / "orchestration/receipts", campaign_sha256,
    )
    records = [dict(record) for record in journal.records]
    if len(records) != 25:
        raise ManagerError("immediate predecessor receipt chain is not terminal")
    failed_probe = records[11].get("sandbox")
    if not isinstance(failed_probe, Mapping):
        raise ManagerError("immediate predecessor failed probe identity missing")
    precloud_path = (
        predecessor_wave_root / "RECOVERY_PRECLOUD_ZERO_RESOURCE_COMPLETE.json"
    )
    precloud_value, _ = read_json(precloud_path)
    pip_freeze_path = (
        predecessor_wave_root / "orchestration/build-evidence/pip-freeze.txt"
    )
    wheel_sha256_path = (
        predecessor_wave_root / "orchestration/build-evidence/wheel-sha256.txt"
    )
    observed_wheels = {
        Path(parts[1]).name: parts[0]
        for parts in (
            line.strip().split(maxsplit=1)
            for line in wheel_sha256_path.read_text().splitlines()
            if line.strip()
        )
        if len(parts) == 2
    }
    pip_freeze_lines = [
        line.strip()
        for line in pip_freeze_path.read_text().splitlines()
        if line.strip()
    ]
    terminal_absence = _immediate_predecessor_terminal_absence_value(
        predecessor_package_root, wave_index=wave_index,
    )
    evidence = {
        "schema": "hfw_w560_immediate_predecessor_quota_failure_evidence_v1",
        "status": "PREDECESSOR_REGION_QUOTA_AUTH_CONTEXT_FAILURE_CLEANED_TERMINAL",
        "package_root": str(predecessor_package_root),
        "wave_root": str(predecessor_wave_root),
        "campaign_sha256": campaign_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "wave_index": wave_index,
        "host_runtime_manifest": _content_record(
            predecessor_package_root / "runtime/config/HOST_RUNTIME_MANIFEST.json"
        ),
        "real_pipeline_intent": _content_record(
            predecessor_wave_root / "REAL_PIPELINE_INTENT.json"
        ),
        "run_config": _content_record(
            predecessor_wave_root / "orchestration/RUN_CONFIG.json"
        ),
        "recovered_task0_provenance": _content_record(
            predecessor_recovery_path
        ),
        "precloud_zero_resource_terminal": {
            "identity": _content_record(precloud_path),
            "value": precloud_value,
        },
        "build_evidence": {
            "pip_freeze_file": _content_record(pip_freeze_path),
            "wheel_sha256_file": _content_record(wheel_sha256_path),
            "pip_freeze_lines": pip_freeze_lines,
            "wheel_closure_sha256_by_filename": observed_wheels,
        },
        "receipt_chain": records,
        "receipt_chain_length": len(records),
        "receipt_chain_head": _canonical_record_identity(records[-1]),
        "failure_receipt": _canonical_record_identity(records[20]),
        "probe_created_receipt": _canonical_record_identity(records[11]),
        "probe_exact_id_absent_receipt": _canonical_record_identity(records[18]),
        "probe_deleted_receipt": _canonical_record_identity(records[19]),
        "snapshot_deleted_receipt": _canonical_record_identity(records[22]),
        "cleanup_terminal_receipt": _canonical_record_identity(records[24]),
        "terminal_absence": terminal_absence,
        "failed_probe_sandbox": dict(failed_probe),
        "platform_added_probe_labels": {
            name: value
            for name, value in failed_probe["labels"].items()
            if name not in _probe_labels(campaign_sha256, wave_index)
        },
        "task0_model_command_count": 0,
        "fanout_worker_sandbox_create_count": 0,
        "local_verify_started": False,
        "checkpoint_or_head_append_count": 0,
        "task0_cloud_search_rerun": False,
        "failed_probe_worker_command_permitted": False,
        "cleanup_terminal": True,
    }
    return _validate_immediate_predecessor_quota_failure(
        evidence,
        campaign_sha256=campaign_sha256,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        wave_index=wave_index,
        original_task0_sandbox=original_task0_sandbox,
        earlier_failed_probe_sandbox=earlier_failed_probe_sandbox,
    )


def _recovered_task0_provenance_value(
    source_package_root: Path,
    predecessor_package_root: Path,
    immediate_predecessor_package_root: Path,
    destination_package_root: Path,
    *,
    wave_index: int,
    recovered_unix_ns: int,
) -> Mapping[str, object]:
    source_package_root = Path(source_package_root).resolve()
    predecessor_package_root = Path(predecessor_package_root).resolve()
    immediate_predecessor_package_root = Path(
        immediate_predecessor_package_root
    ).resolve()
    destination_package_root = Path(destination_package_root).resolve()
    if len({
        source_package_root,
        predecessor_package_root,
        immediate_predecessor_package_root,
        destination_package_root,
    }) != 4:
        raise ManagerError(
            "task0 recovery source, both failed predecessors, and destination "
            "must differ"
        )
    source_wave = (
        source_package_root / "runtime/waves" / f"wave-{wave_index:08d}"
    )
    destination_wave = (
        destination_package_root / "runtime/waves" / f"wave-{wave_index:08d}"
    )
    source_snapshot = source_wave / "snapshot"
    destination_snapshot = destination_wave / "snapshot"
    source_campaign, source_identities = validate_worker_snapshot(source_snapshot)
    destination_campaign, destination_identities = validate_worker_snapshot(
        destination_snapshot
    )
    if source_campaign != destination_campaign:
        raise ManagerError("recovered task0 campaign bytes differ from source campaign")
    if any(
        source_identities["campaign"][key]
        != destination_identities["campaign"][key]
        for key in ("sha256", "size_bytes")
    ):
        raise ManagerError("recovered task0 campaign identity differs from source")
    if any(
        source_identities["snapshot_manifest"][key]
        != destination_identities["snapshot_manifest"][key]
        for key in ("sha256", "size_bytes")
    ):
        raise ManagerError("recovered task0 snapshot manifest differs from source")
    if int(source_campaign["wave_index"]) != wave_index:
        raise ManagerError("recovered task0 wave index drift")

    relative_result = Path("downloads/worker-000/CPU_SEARCH_RESULT.json")
    relative_flat = Path(
        "downloads/worker-000/trajectory_final_group_flat.f32le"
    )
    relative_bundle = Path(
        "downloads/worker-000/accepted_candidates.f32le.bundle"
    )
    source_result_path = source_wave / relative_result
    source_flat_path = source_wave / relative_flat
    source_bundle_path = source_wave / relative_bundle
    destination_result_path = destination_wave / relative_result
    destination_flat_path = destination_wave / relative_flat
    destination_bundle_path = destination_wave / relative_bundle
    campaign_sha256 = str(source_identities["campaign"]["sha256"])
    snapshot_manifest_sha256 = str(
        source_identities["snapshot_manifest"]["sha256"]
    )
    source_validation = _validate_worker_result(
        source_campaign,
        campaign_sha256,
        snapshot_manifest_sha256,
        0,
        source_result_path,
        source_flat_path,
        source_bundle_path,
        None,
    )
    destination_validation = _validate_worker_result(
        destination_campaign,
        campaign_sha256,
        snapshot_manifest_sha256,
        0,
        destination_result_path,
        destination_flat_path,
        destination_bundle_path,
        None,
    )
    source_result = _content_record(source_result_path)
    source_flat = _content_record(source_flat_path)
    source_bundle = _content_record(source_bundle_path)
    destination_result = _content_record(destination_result_path)
    destination_flat = _content_record(destination_flat_path)
    destination_bundle = _content_record(destination_bundle_path)
    if (
        source_result != destination_result
        or source_flat != destination_flat
        or source_bundle != destination_bundle
    ):
        raise ManagerError("recovered task0 output copy is not byte-identical")

    journal = ReceiptJournal(
        source_wave / "orchestration/receipts", campaign_sha256,
    )
    output_receipt = journal.latest(
        "WORKER_OUTPUT_DOWNLOADED_AND_VALIDATED", task_index=0,
    )
    if output_receipt is None:
        raise ManagerError("source task0 validated-output receipt is absent")
    if (
        output_receipt.get("result") != source_result
        or output_receipt.get("flat") != source_flat
        or output_receipt.get("accepted_candidate_bundle", {}).get("sha256")
        != source_bundle["sha256"]
    ):
        raise ManagerError("source task0 receipt/output identity drift")
    source_sandbox_id = output_receipt.get("sandbox_id")
    if not isinstance(source_sandbox_id, str) or not source_sandbox_id:
        raise ManagerError("source task0 validated-output sandbox ID is absent")
    journal_records = journal.records
    if not journal_records or journal_records[-1].get("event") != "CLEANUP_TERMINAL":
        raise ManagerError("source receipt chain lacks cleanup terminal head")
    created_receipts = [
        record for record in journal_records
        if record.get("event") == "SANDBOX_CREATED"
        and record.get("task_index") == 0
        and record.get("attempt") == 0
        and isinstance(record.get("sandbox"), Mapping)
        and record["sandbox"].get("id") == source_sandbox_id
    ]
    if len(created_receipts) != 1:
        raise ManagerError("source task0 sandbox creation identity is not unique")
    created_receipt = created_receipts[0]
    manager_failure = journal.latest("MANAGER_RUN_FAILED")
    cleanup_terminal = journal_records[-1]
    if (
        manager_failure is None
        or manager_failure.get("error")
        != "seed sandbox remained after validated output cleanup"
        or manager_failure.get("classification")
        != "UNCLASSIFIED_OR_INFRASTRUCTURE_FAILURE"
    ):
        raise ManagerError("source manager failure is not the sealed deletion visibility race")
    if (
        cleanup_terminal.get("reason") != "FAILURE"
        or cleanup_terminal.get("failure_count") != 0
        or cleanup_terminal.get("failures") != []
        or cleanup_terminal.get("snapshot_present") is not False
        or cleanup_terminal.get("campaign_sandbox_count") != 1
    ):
        raise ManagerError("source cleanup receipt does not match the known stale-count terminal")

    receipt_paths = sorted(
        (source_wave / "orchestration/receipts").glob("*.receipt.json")
    )
    if len(receipt_paths) != len(journal_records):
        raise ManagerError("source receipt path/count drift")
    output_index = int(output_receipt["receipt_index"])
    output_receipt_identity = _content_record(receipt_paths[output_index])
    created_receipt_identity = _content_record(
        receipt_paths[int(created_receipt["receipt_index"])]
    )
    manager_failure_identity = _content_record(
        receipt_paths[int(manager_failure["receipt_index"])]
    )
    chain_head_identity = _content_record(receipt_paths[-1])
    host_runtime_path = (
        source_package_root / "runtime/config/HOST_RUNTIME_MANIFEST.json"
    )
    pipeline_intent_path = source_wave / "REAL_PIPELINE_INTENT.json"
    host_runtime = _content_record(host_runtime_path)
    pipeline_intent = _content_record(pipeline_intent_path)
    intent, _ = read_json(pipeline_intent_path, str(pipeline_intent["sha256"]))
    if (
        intent.get("campaign_sha256") != campaign_sha256
        or intent.get("host_runtime_manifest_sha256")
        != host_runtime["sha256"]
    ):
        raise ManagerError("source pipeline intent recovery binding drift")

    source_head_path = (
        source_package_root / "runtime/authority/lineage/heads"
        / f"00000000-{source_campaign['base_head_record_sha256']}.head.json"
    )
    destination_head_path = (
        destination_package_root / "runtime/authority/lineage/heads"
        / f"00000000-{source_campaign['base_head_record_sha256']}.head.json"
    )
    source_head, source_head_identity_full = read_json(
        source_head_path, str(source_campaign["base_head_record_sha256"]),
    )
    source_head_identity = _content_record(source_head_path)
    destination_head_identity = _content_record(destination_head_path)
    if source_head_identity != destination_head_identity:
        raise ManagerError("recovery fork copied generation-0 head differs from source")
    source_checkpoint_path = Path(str(source_head["checkpoint"]["path"]))
    source_checkpoint_identity = _content_record(source_checkpoint_path)
    if (
        source_checkpoint_identity["sha256"]
        != source_campaign["base_checkpoint_sha256"]
        or source_checkpoint_identity["size_bytes"]
        != int(source_campaign["base_checkpoint_size_bytes"])
        or source_head_identity_full["sha256"]
        != source_campaign["base_head_record_sha256"]
    ):
        raise ManagerError("attempt5 external head/checkpoint dependency drift")
    current, _ = read_json(
        destination_package_root / "runtime/authority/lineage/CURRENT.json"
    )
    if (
        current.get("record_sha256") != source_campaign["base_head_record_sha256"]
        or Path(str(current.get("record_path"))).resolve()
        != destination_head_path.resolve()
    ):
        raise ManagerError("recovery fork derived CURRENT pointer is not repaired locally")

    task = source_campaign["tasks"][0]
    task_sha256 = sha256_bytes(canonical_json_bytes(task))
    validation_summary = {
        "task_index": 0,
        "task_sha256": task_sha256,
        "worker_id": task["worker_id"],
        "group": task["group"],
        "seed": int(task["seed"]),
        "peak_rss_bytes": int(source_validation["peak_rss_bytes"]),
        "worker_total_seconds": float(source_validation["worker_total_seconds"]),
        "total_physical_model_forwards": int(
            source_validation["total_physical_model_forwards"]
        ),
        "base_full_search_mean_ce": float(
            source_validation["base_full_search_mean_ce"]
        ),
        "terminal_full_search_mean_ce": float(
            source_validation["terminal_full_search_mean_ce"]
        ),
        "self_calibrated_full_search_improvement": float(
            source_validation["self_calibrated_full_search_improvement"]
        ),
        "acceptance_gate_attempts": int(
            source_validation["acceptance_gate_attempts"]
        ),
        "acceptance_gate_passes": int(
            source_validation["acceptance_gate_passes"]
        ),
        "accepted_updates": int(source_validation["accepted_updates"]),
        "environment": dict(source_validation["environment"]),
    }
    if destination_validation["result"]["sha256"] != source_result["sha256"]:
        raise ManagerError("destination task0 revalidation result identity drift")
    if destination_validation["flat"]["sha256"] != source_flat["sha256"]:
        raise ManagerError("destination task0 revalidation flat identity drift")
    if destination_validation["bundle"]["sha256"] != source_bundle["sha256"]:
        raise ManagerError("destination task0 revalidation bundle identity drift")

    predecessor_failure = _predecessor_recovery_failure_value(
        predecessor_package_root,
        campaign_sha256=campaign_sha256,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        wave_index=wave_index,
        original_task0_sandbox=dict(created_receipt["sandbox"]),
    )
    immediate_predecessor_failure = _immediate_predecessor_quota_failure_value(
        immediate_predecessor_package_root,
        campaign_sha256=campaign_sha256,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        wave_index=wave_index,
        original_task0_sandbox=dict(created_receipt["sandbox"]),
        earlier_failed_probe_sandbox=dict(
            predecessor_failure["failed_probe_sandbox"]
        ),
    )
    required_exact_id_reconciliation = [
        {
            "role": "ORIGINAL_SOURCE_TASK0",
            "sandbox": dict(created_receipt["sandbox"]),
        },
        {
            "role": "PREDECESSOR_LABEL_DRIFT_PROBE",
            "sandbox": dict(predecessor_failure["failed_probe_sandbox"]),
        },
        {
            "role": "IMMEDIATE_PREDECESSOR_QUOTA_API_PROBE",
            "sandbox": dict(
                immediate_predecessor_failure["failed_probe_sandbox"]
            ),
        },
    ]
    required_ids = [
        str(row["sandbox"]["id"])
        for row in required_exact_id_reconciliation
    ]
    if len(required_ids) != 3 or len(set(required_ids)) != 3:
        raise ManagerError(
            "recovery requires three unique exact sandbox identities"
        )

    return {
        "schema": RECOVERED_TASK0_PROVENANCE_SCHEMA_V3,
        "status": "RECOVERED_AND_REVALIDATED_NO_RERUN",
        "recovered_unix_ns": recovered_unix_ns,
        "source": {
            "package_root": str(source_package_root),
            "wave_root": str(source_wave),
            "host_runtime_manifest": host_runtime,
            "real_pipeline_intent": pipeline_intent,
            "validated_output_receipt": output_receipt_identity,
            "sandbox_created_receipt": created_receipt_identity,
            "manager_failure_receipt": manager_failure_identity,
            "receipt_chain_head": chain_head_identity,
            "receipt_chain_length": len(journal_records),
            "cleanup_terminal_reported_campaign_sandbox_count": int(
                journal_records[-1].get("campaign_sandbox_count", -1)
            ),
            "manager_failure_error": manager_failure["error"],
            "task0_sandbox": dict(created_receipt["sandbox"]),
            "result": source_result,
            "flat": source_flat,
            "bundle": source_bundle,
        },
        "destination": {
            "package_root": str(destination_package_root),
            "wave_root": str(destination_wave),
            "result": destination_result,
            "flat": destination_flat,
            "bundle": destination_bundle,
            "copied_generation0_head": destination_head_identity,
        },
        "external_lineage_dependency": {
            "status": "REQUIRED_PRESERVED_ATTEMPT5_GENERATION0_HEAD_AND_CHECKPOINT",
            "source_head_path": str(source_head_path),
            "source_head": source_head_identity,
            "source_checkpoint_path": str(source_checkpoint_path),
            "source_checkpoint": source_checkpoint_identity,
            "destination_copied_head_path": str(destination_head_path),
            "destination_copied_head": destination_head_identity,
            "required_through_recovery_wave_promotion_terminal": True,
            "recovery_fork_is_not_self_contained_until_post_wave_successor_rebind": True,
        },
        "predecessor_recovery_failure": predecessor_failure,
        "immediate_predecessor_recovery_failure": (
            immediate_predecessor_failure
        ),
        "required_exact_id_reconciliation": (
            required_exact_id_reconciliation
        ),
        "immutable_binding": {
            "campaign_sha256": campaign_sha256,
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "snapshot_manifest_core_sha256": source_campaign[
                "snapshot_manifest_core_sha256"
            ],
            "base_head_record_sha256": source_campaign[
                "base_head_record_sha256"
            ],
            "base_checkpoint_sha256": source_campaign[
                "base_checkpoint_sha256"
            ],
            "base_model_state_digest": source_campaign[
                "base_model_state_digest"
            ],
            **validation_summary,
        },
        "validation": {
            "source_worker_result_revalidated": True,
            "destination_worker_result_revalidated": True,
            "campaign_bytes_identical": True,
            "snapshot_manifest_bytes_identical": True,
            "result_bytes_identical": True,
            "flat_bytes_identical": True,
            "task0_sandbox_or_worker_command_permitted": False,
            "task0_cloud_search_rerun": False,
            "source_cleanup_terminal_is_not_reused_as_zero_resource_proof": True,
            "fresh_manager_must_reconcile_exact_campaign_resources": True,
            "predecessor_probe_failure_and_cleanup_bound": True,
            "immediate_predecessor_quota_failure_and_cleanup_bound": True,
            "three_unique_exact_id_reconciliations_required_before_cloud_create": True,
            "top_level_required_exact_id_reconciliation_is_authoritative": True,
        },
    }


def build_recovered_task0_provenance(
    source_package_root: Path,
    predecessor_package_root: Path,
    immediate_predecessor_package_root: Path,
    destination_package_root: Path,
    *,
    wave_index: int,
) -> Mapping[str, object]:
    destination_package_root = Path(destination_package_root).resolve()
    path = destination_package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
    if path.exists():
        existing, identity = read_json(path)
        value = _recovered_task0_provenance_value(
            source_package_root,
            predecessor_package_root,
            immediate_predecessor_package_root,
            destination_package_root,
            wave_index=wave_index,
            recovered_unix_ns=int(existing.get("recovered_unix_ns", -1)),
        )
        if existing != value:
            raise ManagerError("recovered task0 provenance already exists with drift")
        return {**existing, "identity": identity}
    value = _recovered_task0_provenance_value(
        source_package_root,
        predecessor_package_root,
        immediate_predecessor_package_root,
        destination_package_root,
        wave_index=wave_index,
        recovered_unix_ns=time.time_ns(),
    )
    identity = write_json_exclusive(path, value)
    return {**value, "identity": identity}


def validate_recovered_task0_provenance(
    package_root: Path,
    *,
    wave_index: int,
    _allow_external_v2: bool = False,
) -> Mapping[str, object] | None:
    package_root = Path(package_root).resolve()
    path = package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
    if not path.exists():
        return None
    observed, identity = read_json(path)
    schema = observed.get("schema")
    if (
        schema not in {
            RECOVERED_TASK0_PROVENANCE_SCHEMA_V2,
            RECOVERED_TASK0_PROVENANCE_SCHEMA_V3,
        }
        or (
            schema == RECOVERED_TASK0_PROVENANCE_SCHEMA_V2
            and not _allow_external_v2
        )
        or observed.get("status") != "RECOVERED_AND_REVALIDATED_NO_RERUN"
        or not isinstance(observed.get("recovered_unix_ns"), int)
        or isinstance(observed.get("recovered_unix_ns"), bool)
    ):
        raise ManagerError("recovered task0 provenance schema/status drift")
    source = observed.get("source")
    destination = observed.get("destination")
    binding = observed.get("immutable_binding")
    validation = observed.get("validation")
    dependency = observed.get("external_lineage_dependency")
    predecessor_failure = observed.get("predecessor_recovery_failure")
    immediate_predecessor_failure = observed.get(
        "immediate_predecessor_recovery_failure"
    )
    required_exact_ids = observed.get("required_exact_id_reconciliation")
    if any(
        not isinstance(value, Mapping)
        for value in (
            source,
            destination,
            binding,
            validation,
            dependency,
            predecessor_failure,
        )
    ):
        raise ManagerError("recovered task0 provenance section missing")
    expected_validation_v2 = {
        "source_worker_result_revalidated": True,
        "destination_worker_result_revalidated": True,
        "campaign_bytes_identical": True,
        "snapshot_manifest_bytes_identical": True,
        "result_bytes_identical": True,
        "flat_bytes_identical": True,
        "task0_sandbox_or_worker_command_permitted": False,
        "task0_cloud_search_rerun": False,
        "source_cleanup_terminal_is_not_reused_as_zero_resource_proof": True,
        "fresh_manager_must_reconcile_exact_campaign_resources": True,
        "predecessor_probe_failure_and_cleanup_bound": True,
        "dual_exact_id_reconciliation_required_before_cloud_create": True,
    }
    expected_validation_v3 = {
        "source_worker_result_revalidated": True,
        "destination_worker_result_revalidated": True,
        "campaign_bytes_identical": True,
        "snapshot_manifest_bytes_identical": True,
        "result_bytes_identical": True,
        "flat_bytes_identical": True,
        "task0_sandbox_or_worker_command_permitted": False,
        "task0_cloud_search_rerun": False,
        "source_cleanup_terminal_is_not_reused_as_zero_resource_proof": True,
        "fresh_manager_must_reconcile_exact_campaign_resources": True,
        "predecessor_probe_failure_and_cleanup_bound": True,
        "immediate_predecessor_quota_failure_and_cleanup_bound": True,
        "three_unique_exact_id_reconciliations_required_before_cloud_create": True,
        "top_level_required_exact_id_reconciliation_is_authoritative": True,
    }
    expected_validation = (
        expected_validation_v2
        if schema == RECOVERED_TASK0_PROVENANCE_SCHEMA_V2
        else expected_validation_v3
    )
    if validation != expected_validation:
        raise ManagerError("recovered task0 provenance validation flags drift")
    if schema == RECOVERED_TASK0_PROVENANCE_SCHEMA_V2:
        if (
            immediate_predecessor_failure is not None
            or required_exact_ids is not None
        ):
            raise ManagerError("external v2 recovery contains v3-only fields")
    elif (
        not isinstance(immediate_predecessor_failure, Mapping)
        or not isinstance(required_exact_ids, list)
    ):
        raise ManagerError("recovered task0 v3 recovery sections missing")
    wave_root = package_root / "runtime/waves" / f"wave-{wave_index:08d}"
    if (
        Path(str(destination.get("package_root"))).resolve() != package_root
        or Path(str(destination.get("wave_root"))).resolve() != wave_root
    ):
        raise ManagerError("recovered task0 destination path binding drift")
    campaign, identities = validate_worker_snapshot(wave_root / "snapshot")
    if (
        identities["campaign"]["sha256"] != binding.get("campaign_sha256")
        or identities["snapshot_manifest"]["sha256"]
        != binding.get("snapshot_manifest_sha256")
        or campaign.get("snapshot_manifest_core_sha256")
        != binding.get("snapshot_manifest_core_sha256")
        or int(campaign.get("wave_index", -1)) != wave_index
    ):
        raise ManagerError("recovered task0 campaign/snapshot binding drift")
    result_path = wave_root / "downloads/worker-000/CPU_SEARCH_RESULT.json"
    flat_path = (
        wave_root / "downloads/worker-000/trajectory_final_group_flat.f32le"
    )
    bundle_path = (
        wave_root / "downloads/worker-000/accepted_candidates.f32le.bundle"
    )
    result_identity = _content_record(result_path)
    flat_identity = _content_record(flat_path)
    bundle_identity = _content_record(bundle_path)
    if (
        destination.get("result") != result_identity
        or destination.get("flat") != flat_identity
        or source.get("result") != result_identity
        or source.get("flat") != flat_identity
        or destination.get("bundle") != bundle_identity
        or source.get("bundle") != bundle_identity
    ):
        raise ManagerError("recovered task0 destination output identity drift")
    result_validation = _validate_worker_result(
        campaign,
        str(identities["campaign"]["sha256"]),
        str(identities["snapshot_manifest"]["sha256"]),
        0,
        result_path,
        flat_path,
        bundle_path,
        None,
    )
    task = campaign["tasks"][0]
    expected_binding = {
        "campaign_sha256": identities["campaign"]["sha256"],
        "snapshot_manifest_sha256": identities["snapshot_manifest"]["sha256"],
        "snapshot_manifest_core_sha256": campaign["snapshot_manifest_core_sha256"],
        "base_head_record_sha256": campaign["base_head_record_sha256"],
        "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
        "base_model_state_digest": campaign["base_model_state_digest"],
        "task_index": 0,
        "task_sha256": sha256_bytes(canonical_json_bytes(task)),
        "worker_id": task["worker_id"],
        "group": task["group"],
        "seed": int(task["seed"]),
        "peak_rss_bytes": int(result_validation["peak_rss_bytes"]),
        "worker_total_seconds": float(result_validation["worker_total_seconds"]),
        "total_physical_model_forwards": int(
            result_validation["total_physical_model_forwards"]
        ),
        "base_full_search_mean_ce": float(
            result_validation["base_full_search_mean_ce"]
        ),
        "terminal_full_search_mean_ce": float(
            result_validation["terminal_full_search_mean_ce"]
        ),
        "self_calibrated_full_search_improvement": float(
            result_validation["self_calibrated_full_search_improvement"]
        ),
        "acceptance_gate_attempts": int(
            result_validation["acceptance_gate_attempts"]
        ),
        "acceptance_gate_passes": int(
            result_validation["acceptance_gate_passes"]
        ),
        "accepted_updates": int(result_validation["accepted_updates"]),
        "environment": dict(result_validation["environment"]),
    }
    if binding != expected_binding:
        raise ManagerError("recovered task0 revalidation summary drift")
    source_sandbox = source.get("task0_sandbox")
    if not isinstance(source_sandbox, Mapping):
        raise ManagerError("recovered source task0 sandbox identity missing")
    expected_labels = _task_labels(
        str(binding["campaign_sha256"]), wave_index, 0, 0,
    )
    if (
        not isinstance(source_sandbox.get("id"), str)
        or not source_sandbox.get("id")
        or not _contains_task_labels(
            source_sandbox.get("labels"), expected_labels,
        )
    ):
        raise ManagerError("recovered source task0 sandbox binding drift")
    required_dependency = {
        "status": "REQUIRED_PRESERVED_ATTEMPT5_GENERATION0_HEAD_AND_CHECKPOINT",
        "required_through_recovery_wave_promotion_terminal": True,
        "recovery_fork_is_not_self_contained_until_post_wave_successor_rebind": True,
    }
    if any(dependency.get(key) != value for key, value in required_dependency.items()):
        raise ManagerError("recovery fork external dependency disclosure drift")
    source_head_path = Path(str(dependency.get("source_head_path")))
    source_checkpoint_path = Path(str(dependency.get("source_checkpoint_path")))
    destination_head_path = Path(str(dependency.get("destination_copied_head_path")))
    if (
        _content_record(source_head_path) != dependency.get("source_head")
        or _content_record(source_checkpoint_path) != dependency.get("source_checkpoint")
        or _content_record(destination_head_path)
        != dependency.get("destination_copied_head")
        or destination.get("copied_generation0_head")
        != dependency.get("destination_copied_head")
    ):
        raise ManagerError("recovery fork external head/checkpoint dependency drift")
    head, head_identity = read_json(
        destination_head_path, str(binding["base_head_record_sha256"]),
    )
    if (
        head_identity["sha256"] != binding["base_head_record_sha256"]
        or Path(str(head["checkpoint"]["path"])).resolve()
        != source_checkpoint_path.resolve()
        or head["checkpoint"]["sha256"] != binding["base_checkpoint_sha256"]
    ):
        raise ManagerError("recovery fork copied head content drift")
    current, _ = read_json(package_root / "runtime/authority/lineage/CURRENT.json")
    if (
        current.get("record_sha256") != binding["base_head_record_sha256"]
        or Path(str(current.get("record_path"))).resolve()
        != destination_head_path.resolve()
    ):
        raise ManagerError("recovery fork derived CURRENT pointer drift")
    _validate_predecessor_recovery_failure(
        predecessor_failure,
        campaign_sha256=str(binding["campaign_sha256"]),
        snapshot_manifest_sha256=str(binding["snapshot_manifest_sha256"]),
        wave_index=wave_index,
        original_task0_sandbox=source_sandbox,
    )
    if schema == RECOVERED_TASK0_PROVENANCE_SCHEMA_V3:
        assert isinstance(immediate_predecessor_failure, Mapping)
        validated_immediate = _validate_immediate_predecessor_quota_failure(
            immediate_predecessor_failure,
            campaign_sha256=str(binding["campaign_sha256"]),
            snapshot_manifest_sha256=str(binding["snapshot_manifest_sha256"]),
            wave_index=wave_index,
            original_task0_sandbox=source_sandbox,
            earlier_failed_probe_sandbox=predecessor_failure[
                "failed_probe_sandbox"
            ],
        )
        expected_required_exact_ids = [
            {
                "role": "ORIGINAL_SOURCE_TASK0",
                "sandbox": dict(source_sandbox),
            },
            {
                "role": "PREDECESSOR_LABEL_DRIFT_PROBE",
                "sandbox": dict(predecessor_failure["failed_probe_sandbox"]),
            },
            {
                "role": "IMMEDIATE_PREDECESSOR_QUOTA_API_PROBE",
                "sandbox": dict(validated_immediate["failed_probe_sandbox"]),
            },
        ]
        if required_exact_ids != expected_required_exact_ids:
            raise ManagerError(
                "recovered task0 top-level exact-ID reconciliation drift"
            )
        sandbox_ids = [
            str(row["sandbox"]["id"])
            for row in required_exact_ids
        ]
        roles = [str(row["role"]) for row in required_exact_ids]
        if (
            len(sandbox_ids) != 3
            or len(set(sandbox_ids)) != 3
            or len(set(roles)) != 3
        ):
            raise ManagerError(
                "recovered task0 required exact-ID reconciliations are not unique"
            )
    return {**observed, "identity": identity}


def _validate_worker_failure(
    campaign: Mapping[str, object],
    campaign_sha256: str,
    snapshot_manifest_sha256: str,
    task_index: int,
    failure_path: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    policy = campaign["worker_failure_policy"]
    if Path(failure_path).name != policy["machine_readable_filename"]:
        raise ManagerError("worker failure artifact filename drift")
    failure, identity = read_json(failure_path)
    task = campaign["tasks"][task_index]
    requirements = {
        "schema": policy["schema"],
        "status": policy["deterministic_numerical_status"],
        "classification": policy["deterministic_numerical_classification"],
        "retry_policy": policy["deterministic_numerical_retry_policy"],
        "campaign_sha256": campaign_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "snapshot_manifest_core_sha256": campaign["snapshot_manifest_core_sha256"],
        "task_sha256": sha256_bytes(canonical_json_bytes(task)),
        "task_index": task_index,
        "worker_id": task["worker_id"],
        "wave_index": int(campaign["wave_index"]),
        "group": task["group"],
        "seed": int(task["seed"]),
        "base_head_record_sha256": campaign["base_head_record_sha256"],
        "base_checkpoint_sha256": campaign["base_checkpoint_sha256"],
        "base_model_state_digest": campaign["base_model_state_digest"],
        "worker_checkpoint_or_head_writes": False,
        "mps_verify_opened": False,
        "cloud_credentials_available_to_worker": False,
        "training_or_backprop": False,
        "optimizer_steps": 0,
    }
    for key, expected in requirements.items():
        observed = failure.get(key)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, int):
            matches = (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed == expected
            )
        else:
            matches = observed == expected
        if not matches:
            raise ManagerError(f"worker {task_index} failure contract drift: {key}")
    stage = failure.get("stage")
    if stage not in policy["allowed_stages"]:
        raise ManagerError(f"worker {task_index} failure stage drift")
    lane = campaign["proposal_lane"]
    expected_lane = _fp64_numerical_lane(campaign)
    if failure.get("numerical_lane") != expected_lane:
        raise ManagerError(f"worker {task_index} failure numerical lane drift")
    sealed_environment = lane["sealed_environment"]
    environment = failure.get("environment")
    environment_keys = (
        "python_major_minor", "platform_system", "machine", "torch_version",
        "torch_cuda_version", "numpy_version",
    )
    if not isinstance(environment, Mapping) or set(environment) != set(environment_keys):
        raise ManagerError(f"worker {task_index} failure environment inventory drift")
    for key in environment_keys:
        if environment.get(key) != sealed_environment[key]:
            raise ManagerError(f"worker {task_index} failure environment drift: {key}")
    diagnostics = failure.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ManagerError(f"worker {task_index} failure diagnostics missing")
    memory_limit = int(lane["memory_limit_gib"]) * 1024 ** 3
    if int(diagnostics.get("memory_limit_bytes", -1)) != memory_limit:
        raise ManagerError(f"worker {task_index} failure memory limit drift")
    peak_rss = int(diagnostics.get("maximum_observed_rss_bytes", -1))
    if peak_rss <= 0:
        raise ManagerError(f"worker {task_index} failure RSS missing")
    rss_below = diagnostics.get("peak_rss_strictly_below_memory_limit")
    if rss_below is not (peak_rss < memory_limit):
        raise ManagerError(f"worker {task_index} failure RSS relation drift")
    counter_names = (
        "preflight_logical_evaluations",
        "preflight_physical_model_forwards",
        "solver_logical_evaluations_completed",
        "solver_physical_model_forwards_completed",
        "acceptance_gate_attempts_completed",
        "acceptance_gate_passes_completed",
        "acceptance_gate_failures_completed",
        "float64_logits_observed",
        "float64_hidden_observed",
        "float64_persistent_state_observed",
        "float64_ce_evaluations",
    )
    counters = {}
    for name in counter_names:
        value = diagnostics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ManagerError(f"worker {task_index} failure counter drift: {name}")
        counters[name] = value
    solver = campaign["solver"]
    if (
        counters["preflight_logical_evaluations"]
        > int(solver["base_center_preflight_logical_evaluations"])
        or counters["preflight_physical_model_forwards"]
        > int(solver["base_center_preflight_physical_model_forwards"])
        or counters["solver_logical_evaluations_completed"]
        > int(solver["total_logical_evaluations_maximum"])
        - int(solver["base_center_preflight_logical_evaluations"])
        or counters["solver_physical_model_forwards_completed"]
        > int(solver["total_physical_model_forwards_maximum"])
        - int(solver["base_center_preflight_physical_model_forwards"])
    ):
        raise ManagerError(f"worker {task_index} failure counters exceed campaign")
    gate_attempts = counters["acceptance_gate_attempts_completed"]
    if (
        not 0 <= counters["acceptance_gate_passes_completed"] <= gate_attempts
        <= int(solver["acceptance_gate_maximum_attempts"])
        or counters["acceptance_gate_failures_completed"]
        != gate_attempts - counters["acceptance_gate_passes_completed"]
    ):
        raise ManagerError(f"worker {task_index} failure gate summary drift")
    phase_logical = diagnostics.get("phase_logical_evaluations")
    phase_physical = diagnostics.get("phase_physical_model_forwards")
    expected_phase_keys = {"SEARCH", "ACCEPTANCE_COMMIT_GATE", "TERMINAL_RANKING"}
    if (
        not isinstance(phase_logical, Mapping)
        or not isinstance(phase_physical, Mapping)
        or set(phase_logical) != expected_phase_keys
        or set(phase_physical) != expected_phase_keys
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (*phase_logical.values(), *phase_physical.values())
        )
        or sum(phase_logical.values()) != counters["solver_logical_evaluations_completed"]
        or sum(phase_physical.values()) != counters["solver_physical_model_forwards_completed"]
        or phase_logical["ACCEPTANCE_COMMIT_GATE"]
        != gate_attempts * int(solver["acceptance_gate_logical_evaluations_per_attempt"])
        or phase_physical["ACCEPTANCE_COMMIT_GATE"]
        != gate_attempts * int(solver["acceptance_gate_physical_model_forwards_per_attempt"])
    ):
        raise ManagerError(f"worker {task_index} failure phase accounting drift")
    completed_forwards = (
        counters["preflight_physical_model_forwards"]
        + counters["solver_physical_model_forwards_completed"]
    )
    if (
        counters["float64_hidden_observed"] > completed_forwards
        or counters["float64_logits_observed"] > completed_forwards
        or counters["float64_persistent_state_observed"] > completed_forwards
        or counters["float64_ce_evaluations"] > completed_forwards
    ):
        raise ManagerError(f"worker {task_index} failure forward accounting drift")
    artifacts_created = failure.get("solver_artifacts_created")
    if not isinstance(artifacts_created, bool):
        raise ManagerError(f"worker {task_index} failure artifact flag drift")
    credential_scan = failure.get("credential_environment_scan")
    if (
        not isinstance(credential_scan, Mapping)
        or credential_scan.get("forbidden_name_matches") != 0
        or credential_scan.get("forbidden_value_pattern_matches") != 0
    ):
        raise ManagerError(f"worker {task_index} failure credential scan drift")
    reason_code = diagnostics.get("reason_code")
    allowed_reason_by_stage = policy.get("terminal_reason_codes_by_stage")
    if (
        not isinstance(allowed_reason_by_stage, Mapping)
        or reason_code not in allowed_reason_by_stage.get(stage, [])
    ):
        raise ManagerError(f"worker {task_index} failure reason code drift")
    if stage == "BASE_CENTER_PREFLIGHT":
        if artifacts_created is not False or any(phase_logical.values()) or any(
            phase_physical.values()
        ):
            raise ManagerError(f"worker {task_index} base failure started solver")
        preflight = diagnostics.get("preflight")
        if not isinstance(preflight, Mapping) or preflight.get("schema") != lane[
            "base_full_bank_gate_schema"
        ]:
            raise ManagerError(f"worker {task_index} base failure preflight drift")
        if reason_code == "SEALED_PREFLIGHT_OPERATION_EXCEPTION":
            exception_type = preflight.get("operation_exception_type")
            failed_batch = preflight.get("failed_batch_index")
            failed_sequence = preflight.get("failed_sequence_index")
            if (
                preflight.get("status") != "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT"
                or preflight.get("failure_reasons")
                != ["SEALED_PREFLIGHT_OPERATION_EXCEPTION"]
                or not isinstance(preflight.get("completed_batches"), list)
                or not isinstance(failed_batch, int)
                or isinstance(failed_batch, bool)
                or not 0 <= failed_batch < int(campaign["search_bank_shape"][0])
                or not isinstance(failed_sequence, int)
                or isinstance(failed_sequence, bool)
                or not 0 <= failed_sequence < int(campaign["search_bank_shape"][1])
                or not isinstance(exception_type, str)
                or not 0 < len(exception_type) <= 128
                or not isinstance(
                    preflight.get("operation_exception_message_sha256"), str,
                )
                or not _SHA256.fullmatch(
                    str(preflight["operation_exception_message_sha256"])
                )
                or len(preflight["completed_batches"]) != failed_batch
                or counters["preflight_logical_evaluations"] != failed_batch + 1
                or counters["preflight_physical_model_forwards"]
                != failed_batch * int(campaign["search_bank_shape"][1])
                + failed_sequence + 1
            ):
                raise ManagerError(
                    f"worker {task_index} preflight operation failure evidence drift"
                )
        elif reason_code == "BASE_CENTER_NUMERICAL_OR_DTYPE_GATE_FAILED":
            failure_reasons = preflight.get("failure_reasons")
            batches = preflight.get("batches")
            batch_losses = preflight.get("base_full_search_batch_losses")
            failed_rows = []
            if isinstance(batches, list):
                for batch in batches:
                    if not isinstance(batch, Mapping):
                        continue
                    rows = batch.get("per_sequence")
                    if isinstance(rows, list):
                        failed_rows.extend(
                            row for row in rows
                            if isinstance(row, Mapping) and (
                                _sequence_diagnostic_proves_dtype_mismatch(row)
                                or _sequence_diagnostic_proves_nonfinite(row)
                            )
                        )
            if (
                preflight.get("status") != "FAIL_DETERMINISTIC_NUMERICAL_PREFLIGHT"
                or not isinstance(failure_reasons, list)
                or not failure_reasons
                or any(not isinstance(value, str) or not value for value in failure_reasons)
                or preflight.get("batch_indices")
                != list(range(int(campaign["search_bank_shape"][0])))
                or not isinstance(batches, list)
                or len(batches) != int(campaign["search_bank_shape"][0])
                or not any(
                    isinstance(batch, Mapping) and batch.get("passed") is False
                    for batch in batches
                )
                or not isinstance(batch_losses, list)
                or len(batch_losses) != len(batches)
                or not failed_rows
                or preflight.get("base_full_search_mean_ce") is not None
                or counters["preflight_logical_evaluations"]
                != int(solver["base_center_preflight_logical_evaluations"])
                or counters["preflight_physical_model_forwards"]
                != int(solver["base_center_preflight_physical_model_forwards"])
            ):
                raise ManagerError(
                    f"worker {task_index} numerical preflight failure evidence drift"
                )
        elif reason_code == "PREFLIGHT_RSS_LIMIT_REACHED":
            if (
                preflight.get("status") != "PASS_FINITE_CPU_FP64_ALL_SEARCH_BASE"
                or preflight.get("failure_reasons") != []
                or not math.isfinite(float(preflight.get("base_full_search_mean_ce")))
                or peak_rss < memory_limit
                or rss_below is not False
                or counters["preflight_logical_evaluations"]
                != int(solver["base_center_preflight_logical_evaluations"])
                or counters["preflight_physical_model_forwards"]
                != int(solver["base_center_preflight_physical_model_forwards"])
            ):
                raise ManagerError(
                    f"worker {task_index} preflight RSS failure evidence drift"
                )
    else:
        if artifacts_created is not True:
            raise ManagerError(f"worker {task_index} post-preflight failure lacks artifacts")
        if (
            counters["preflight_logical_evaluations"]
            != int(solver["base_center_preflight_logical_evaluations"])
            or counters["preflight_physical_model_forwards"]
            != int(solver["base_center_preflight_physical_model_forwards"])
        ):
            raise ManagerError(f"worker {task_index} post-preflight base gate drift")
    if "group" in diagnostics and diagnostics.get("group") != task["group"]:
        raise ManagerError(f"worker {task_index} failure group drift")
    if stage == "FORWARD_DTYPE":
        sequence = diagnostics.get("sequence")
        dtype_mismatch = _sequence_diagnostic_proves_dtype_mismatch(sequence)
        if (
            diagnostics.get("group") != task["group"]
            or not isinstance(diagnostics.get("batch_index"), int)
            or isinstance(diagnostics.get("batch_index"), bool)
            or not 0 <= int(diagnostics["batch_index"]) < int(
                campaign["search_bank_shape"][0]
            )
            or not isinstance(diagnostics.get("sequence_index"), int)
            or isinstance(diagnostics.get("sequence_index"), bool)
            or not 0 <= int(diagnostics["sequence_index"]) < int(
                campaign["search_bank_shape"][1]
            )
            or not dtype_mismatch
        ):
            raise ManagerError(f"worker {task_index} dtype failure diagnostics drift")
    elif stage == "SEARCH_CENTER":
        objective_failure = diagnostics.get("objective")
        if (
            diagnostics.get("reason") not in {
                "VENDOR_REJECTED_NONFINITE_SEARCH_CENTER",
                "VENDOR_REJECTED_NONFINITE_SEARCH_BUDGET_FILL",
            }
            or not isinstance(objective_failure, Mapping)
            or objective_failure.get("group") != task["group"]
            or not isinstance(objective_failure.get("batch_index"), int)
            or isinstance(objective_failure.get("batch_index"), bool)
            or not 0 <= int(objective_failure["batch_index"]) < int(
                campaign["search_bank_shape"][0]
            )
            or not isinstance(objective_failure.get("sequence_index"), int)
            or isinstance(objective_failure.get("sequence_index"), bool)
            or not 0 <= int(objective_failure["sequence_index"]) < int(
                campaign["search_bank_shape"][1]
            )
            or (
                not _sequence_diagnostic_proves_nonfinite(
                    objective_failure.get("sequence")
                )
            )
        ):
            raise ManagerError(f"worker {task_index} SEARCH failure diagnostics drift")
    elif stage == "TERMINAL_RANKING":
        expected_search_logical = int(solver["logical_search_evaluations"])
        expected_search_physical = int(solver["physical_search_model_forwards"])
        maximum_terminal_logical = int(
            solver["terminal_full_search_ranking_logical_evaluations"]
        )
        maximum_terminal_physical = int(
            solver["terminal_full_search_ranking_physical_model_forwards"]
        )
        if (
            phase_logical["SEARCH"] != expected_search_logical
            or phase_physical["SEARCH"] != expected_search_physical
        ):
            raise ManagerError(
                f"worker {task_index} terminal failure lacks completed SEARCH"
            )
        if reason_code == "NONFINITE_TERMINAL_RANKING":
            if (
                not 1 <= phase_logical["TERMINAL_RANKING"] <= maximum_terminal_logical
                or phase_physical["TERMINAL_RANKING"]
                != phase_logical["TERMINAL_RANKING"]
                * int(solver["physical_model_forwards_per_logical_evaluation"])
            ):
                raise ManagerError(
                    f"worker {task_index} partial terminal ranking accounting drift"
                )
        elif (
            phase_logical["TERMINAL_RANKING"] != maximum_terminal_logical
            or phase_physical["TERMINAL_RANKING"] != maximum_terminal_physical
        ):
            raise ManagerError(
                f"worker {task_index} post-ranking failure accounting drift"
            )
        if reason_code == "NONFINITE_TERMINAL_RANKING":
            has_indexed_nonfinite = (
                diagnostics.get("group") == task["group"]
                and isinstance(diagnostics.get("batch_index"), int)
                and not isinstance(diagnostics.get("batch_index"), bool)
                and 0 <= int(diagnostics["batch_index"])
                < int(campaign["search_bank_shape"][0])
                and isinstance(diagnostics.get("sequence_index"), int)
                and not isinstance(diagnostics.get("sequence_index"), bool)
                and 0 <= int(diagnostics["sequence_index"])
                < int(campaign["search_bank_shape"][1])
                and _sequence_diagnostic_proves_nonfinite(
                    diagnostics.get("sequence")
                )
            )
            if not has_indexed_nonfinite:
                raise ManagerError(
                    f"worker {task_index} terminal nonfinite diagnostics drift"
                )
        elif reason_code == "TERMINAL_CACHE_FLAT_IDENTITY_MISMATCH":
            terminal_sha = diagnostics.get("terminal_flat_sha256")
            cached_sha = diagnostics.get("cached_flat_sha256")
            if (
                not isinstance(terminal_sha, str)
                or not _SHA256.fullmatch(terminal_sha)
                or (
                    cached_sha is not None
                    and (
                        not isinstance(cached_sha, str)
                        or not _SHA256.fullmatch(cached_sha)
                    )
                )
                or cached_sha == terminal_sha
            ):
                raise ManagerError(
                    f"worker {task_index} terminal cache identity diagnostics drift"
                )
        elif reason_code == "TERMINAL_DUPLICATE_BATCH_COUNT_MISMATCH":
            terminal_count = diagnostics.get("terminal_batch_count")
            cached_count = diagnostics.get("cached_batch_count")
            if (
                not isinstance(terminal_count, int)
                or isinstance(terminal_count, bool)
                or terminal_count < 0
                or not isinstance(cached_count, int)
                or isinstance(cached_count, bool)
                or cached_count < 0
                or terminal_count == cached_count
            ):
                raise ManagerError(
                    f"worker {task_index} terminal batch-count diagnostics drift"
                )
        elif reason_code == "TERMINAL_DUPLICATE_TOLERANCE_EXCEEDED":
            maximum = _finite_nonnegative(
                diagnostics.get("maximum_absolute_difference"),
                "terminal_duplicate.maximum_absolute_difference",
            )
            tolerance = _finite_nonnegative(
                diagnostics.get("absolute_tolerance"),
                "terminal_duplicate.absolute_tolerance",
            )
            if (
                maximum <= tolerance
                or tolerance
                != float(lane["fp64_full_bank_duplicate_absolute_tolerance"])
                or not isinstance(diagnostics.get("flat_sha256"), str)
                or not _SHA256.fullmatch(str(diagnostics["flat_sha256"]))
            ):
                raise ManagerError(
                    f"worker {task_index} terminal tolerance diagnostics drift"
                )
        elif reason_code == "WORKER_RSS_LIMIT_REACHED_AFTER_SEARCH":
            observed_peak = diagnostics.get("observed_peak_rss_bytes")
            if (
                not isinstance(observed_peak, int)
                or isinstance(observed_peak, bool)
                or observed_peak < memory_limit
                or observed_peak != peak_rss
                or rss_below is not False
                or diagnostics.get("memory_limit_bytes") != memory_limit
            ):
                raise ManagerError(f"worker {task_index} terminal RSS diagnostics drift")
    return failure, identity


class DaytonaManager:
    def __init__(self, config: ManagerConfig):
        _validate_config(config)
        self.config = config
        self.snapshot_root = config.wave_root / "snapshot"
        self.campaign, identities = validate_worker_snapshot(self.snapshot_root)
        self.campaign_sha256 = str(identities["campaign"]["sha256"])
        self.snapshot_manifest_sha256 = str(identities["snapshot_manifest"]["sha256"])
        self.upload_snapshot_identity = validate_upload_snapshot_root(
            self.snapshot_root, config.upload_snapshot_root,
        )
        if self.upload_snapshot_identity["campaign_sha256"] != self.campaign_sha256:
            raise ManagerError("upload campaign is not bound to manager campaign")
        if (
            self.upload_snapshot_identity["snapshot_manifest_sha256"]
            != self.snapshot_manifest_sha256
        ):
            raise ManagerError("upload manifest is not bound to manager snapshot manifest")
        self.upload_staging_receipt_identity = validate_upload_staging_receipt(
            config, self.snapshot_root, self.upload_snapshot_identity,
        )
        self.wave_index = int(self.campaign["wave_index"])
        self.orchestration_root = config.wave_root / "orchestration"
        self.orchestration_root.mkdir(parents=True, exist_ok=True)
        self.download_root = config.wave_root / "downloads"
        self.download_root.mkdir(parents=True, exist_ok=True)
        self.recovered_task0 = validate_recovered_task0_provenance(
            config.package_root, wave_index=self.wave_index,
        )
        self.journal = ReceiptJournal(
            self.orchestration_root / "receipts", self.campaign_sha256,
        )
        self.credential = os.environ.get("DAYTONA_API_KEY", "")
        if not self.credential:
            raise ManagerError("DAYTONA_API_KEY must be provided in process environment")
        self._cancel = threading.Event()
        self._deterministic_failure_lock = threading.Lock()
        self._create_semaphore = threading.BoundedSemaphore(config.create_concurrency)
        self._daytona = None
        self._cleanup_mode = False
        self._cleanup_deadline_monotonic = 0.0
        self._last_cleanup_evidence: Mapping[str, object] | None = None
        self._cleanup_evidence_lock = threading.Lock()
        self._confirmed_absent_sandbox_ids: set[str] = set()
        self._run_config = self._seal_run_config()

    @property
    def deadline_unix_ns(self) -> int:
        return int(self._run_config["campaign_deadline_unix_ns"])

    def _seal_run_config(self) -> Mapping[str, object]:
        path = self.orchestration_root / "RUN_CONFIG.json"
        if path.exists():
            value, _ = read_json(path)
            expected = self.config.receipt_value(
                self.campaign_sha256,
                int(value["campaign_started_unix_ns"]),
                self.upload_snapshot_identity,
                self.upload_staging_receipt_identity,
                None if self.recovered_task0 is None
                else self.recovered_task0["identity"],
            )
            if value != expected:
                raise ManagerError("manager configuration changed across restart")
            return value
        value = self.config.receipt_value(
            self.campaign_sha256,
            time.time_ns(),
            self.upload_snapshot_identity,
            self.upload_staging_receipt_identity,
            None if self.recovered_task0 is None
            else self.recovered_task0["identity"],
        )
        write_json_exclusive(path, value)
        return value

    def _remaining_seconds(self) -> float:
        if self._cleanup_mode:
            remaining = self._cleanup_deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("bounded cleanup deadline expired")
            return remaining
        remaining = (self.deadline_unix_ns - time.time_ns()) / 1_000_000_000
        if remaining <= 0:
            raise TimeoutError("sealed campaign deadline expired")
        return remaining

    def _retry(self, name: str, operation: Callable[[], _T]) -> _T:
        delay = self.config.retry_initial_seconds
        last_error: BaseException | None = None
        for attempt in range(self.config.maximum_attempts):
            if self._cancel.is_set() and not self._cleanup_mode:
                raise Cancelled(f"cancelled before {name}")
            self._remaining_seconds()
            try:
                return operation()
            except BaseException as error:
                last_error = error
                self.journal.append(
                    "SDK_CALL_RETRY",
                    operation=name,
                    attempt=attempt,
                    error=_safe_error(error, self.credential),
                )
                if attempt + 1 >= self.config.maximum_attempts:
                    break
                bounded = min(delay, self.config.retry_maximum_seconds, self._remaining_seconds())
                jitter = secrets.randbelow(1001) / 1000.0
                time.sleep(min(bounded * (0.75 + 0.5 * jitter), self._remaining_seconds()))
                delay = min(delay * 2, self.config.retry_maximum_seconds)
        assert last_error is not None
        raise ManagerError(f"{name} exhausted retries: {_safe_error(last_error, self.credential)}")

    def _client(self):
        if self._daytona is not None:
            return self._daytona
        if importlib.metadata.version("daytona") != SDK_VERSION:
            raise ManagerError(f"Daytona SDK must be exactly {SDK_VERSION}")
        from daytona import Daytona, DaytonaConfig
        self._daytona = Daytona(DaytonaConfig(
            api_url=self.config.api_url,
            target=self.config.target,
            connection_pool_maxsize=250,
        ))
        return self._daytona

    def _build_image(self):
        from daytona import Image
        sealed = self.campaign["proposal_lane"]["sealed_environment"]
        numpy_version = str(sealed["numpy_version"])
        torch_runtime_version = str(sealed["torch_version"])
        if not torch_runtime_version.endswith("+cpu"):
            raise ManagerError("campaign torch version is not explicitly CPU-only")
        torch_package_version = torch_runtime_version[:-4]
        requirements = (
            f"numpy=={numpy_version}",
            f"torch=={torch_package_version}",
            *LOCKED_RUNTIME_PACKAGES,
        )
        requirement_arguments = " ".join(shlex.quote(value) for value in requirements)
        expected_rows = tuple(
            f"{digest}  /tmp/w560-wheelhouse/{filename}"
            for filename, digest in sorted(EXPECTED_WHEEL_SHA256.items())
        )
        expected_row_arguments = " ".join(
            shlex.quote(row) for row in expected_rows
        )
        image = Image.base(self.config.base_image)
        image = image.run_commands(
            "mkdir -p /opt/w560-build /tmp/w560-wheelhouse",
            (
                "python -m pip download --only-binary=:all: "
                "--index-url https://download.pytorch.org/whl/cpu "
                "--extra-index-url https://pypi.org/simple "
                f"--dest /tmp/w560-wheelhouse {requirement_arguments}"
            ),
            (
                "sha256sum /tmp/w560-wheelhouse/*.whl | LC_ALL=C sort "
                "> /opt/w560-build/wheel-sha256.txt"
            ),
            (
                "printf '%s\\n' " + expected_row_arguments
                + " > /opt/w560-build/expected-wheel-sha256.txt"
            ),
            (
                "sha256sum -c /opt/w560-build/expected-wheel-sha256.txt"
            ),
            (
                "python -m pip install --no-cache-dir --no-index --no-deps "
                "--find-links=/tmp/w560-wheelhouse " + requirement_arguments
            ),
            "python -m pip freeze --all | LC_ALL=C sort > /opt/w560-build/pip-freeze.txt",
            "rm -rf /tmp/w560-wheelhouse",
        )
        upload_snapshot_identity = validate_upload_snapshot_root(
            self.snapshot_root, self.config.upload_snapshot_root,
        )
        if upload_snapshot_identity != self.upload_snapshot_identity:
            raise ManagerError(
                "upload snapshot changed after sealed manager configuration"
            )
        try:
            image = image.add_local_dir(
                self.config.upload_snapshot_root, REMOTE_SNAPSHOT_ROOT,
            )
        finally:
            self._restore_and_revalidate_upload_snapshot(
                "SDK image build-context registration",
            )
        image = image.workdir(REMOTE_SNAPSHOT_ROOT)
        return image

    def _restore_and_revalidate_upload_snapshot(self, operation: str) -> None:
        """Restore sealed modes and prove SDK consumption did not change bytes."""
        _restore_upload_snapshot_read_only(self.config.upload_snapshot_root)
        try:
            restored_upload_identity = validate_upload_snapshot_root(
                self.snapshot_root, self.config.upload_snapshot_root,
            )
        except ManagerError as error:
            raise ManagerError(
                f"{operation} changed sealed upload snapshot bytes or inventory: "
                f"{error}"
            ) from error
        if restored_upload_identity != self.upload_snapshot_identity:
            raise ManagerError(
                f"{operation} changed sealed upload snapshot bytes or inventory"
            )
        try:
            restored_receipt_identity = validate_upload_staging_receipt(
                self.config,
                self.snapshot_root,
                restored_upload_identity,
            )
        except ManagerError as error:
            raise ManagerError(
                f"{operation} changed sealed upload staging filesystem identity: "
                f"{error}"
            ) from error
        if (
            restored_receipt_identity
            != self.upload_staging_receipt_identity
        ):
            raise ManagerError(
                f"{operation} changed sealed upload staging receipt identity"
            )

    def _snapshot_expected(self) -> tuple[int, int, int]:
        lane = self.campaign["proposal_lane"]
        return (
            int(lane["cpu_limit"]),
            int(lane["memory_limit_gib"]),
            int(lane["disk_limit_gib"]),
        )

    def _validate_snapshot(self, snapshot: object) -> None:
        identity = _snapshot_identity(snapshot)
        cpu, memory, disk = self._snapshot_expected()
        if identity["name"] != _snapshot_name(self.campaign_sha256, self.wave_index):
            raise ManagerError("snapshot deterministic name drift")
        if identity["state"] != "active":
            raise ManagerError(f"snapshot is not active: {identity['state']}")
        if (
            identity["cpu"] != cpu
            or identity["memory_gib"] != memory
            or identity["disk_gib"] != disk
            or identity["gpu"] != 0
        ):
            raise ManagerError("snapshot resources drift")

    def _validate_snapshot_for_delete(self, snapshot: object) -> None:
        identity = _snapshot_identity(snapshot)
        cpu, memory, disk = self._snapshot_expected()
        if identity["name"] != _snapshot_name(self.campaign_sha256, self.wave_index):
            raise ManagerError("refusing to delete snapshot with non-campaign name")
        if identity["state"] not in {
            "active", "error", "build_failed", "inactive", "pending",
            "building", "pulling", "removing",
        }:
            raise ManagerError("refusing to delete snapshot with unknown state")
        if (
            identity["cpu"] != cpu
            or identity["memory_gib"] != memory
            or identity["disk_gib"] != disk
            or identity["gpu"] != 0
        ):
            raise ManagerError("refusing to delete snapshot with resource drift")

    def _find_snapshot(self):
        from daytona import DaytonaNotFoundError
        name = _snapshot_name(self.campaign_sha256, self.wave_index)
        try:
            return self._client().snapshot.get(name)
        except DaytonaNotFoundError:
            return None

    def _delete_snapshot_exact(self, snapshot: object, reason: str) -> None:
        from daytona import DaytonaNotFoundError
        self._validate_snapshot_for_delete(snapshot)
        identity = _snapshot_identity(snapshot)
        self.journal.append(
            "SNAPSHOT_DELETE_INTENT", snapshot=identity, reason=reason,
        )

        def delete_or_absent() -> None:
            try:
                self._client().snapshot.delete(snapshot)
            except DaytonaNotFoundError:
                return

        self._retry(f"delete snapshot {identity['id']}", delete_or_absent)
        while True:
            observed = self._retry(
                f"confirm snapshot {identity['id']} deletion",
                self._find_snapshot,
            )
            if observed is None:
                break
            observed_identity = _snapshot_identity(observed)
            if observed_identity["id"] != identity["id"]:
                raise ManagerError(
                    "a different snapshot appeared under the deterministic campaign name"
                )
            self._validate_snapshot_for_delete(observed)
            time.sleep(min(self.config.poll_seconds, self._remaining_seconds()))
        self.journal.append(
            "SNAPSHOT_DELETED", snapshot=identity, reason=reason,
            exact_name_not_found_confirmed=True,
        )

    def _reconcile_snapshot(self, snapshot: object | None):
        if snapshot is None:
            return None
        while snapshot is not None:
            state = _state_text(getattr(snapshot, "state"))
            if state == "active":
                self._validate_snapshot(snapshot)
                return snapshot
            if state in {"error", "build_failed", "inactive"}:
                self._delete_snapshot_exact(snapshot, f"RECONCILE_{state.upper()}")
                return None
            if state == "removing":
                time.sleep(min(self.config.poll_seconds, self._remaining_seconds()))
                snapshot = self._retry("poll removing snapshot", self._find_snapshot)
                continue
            if state in {"building", "pending", "pulling"}:
                time.sleep(min(self.config.poll_seconds, self._remaining_seconds()))
                snapshot = self._retry("poll snapshot build", self._find_snapshot)
                continue
            raise ManagerError(f"unknown snapshot state during reconciliation: {state}")
        return None

    def _ensure_snapshot(self):
        existing = self._reconcile_snapshot(self._find_snapshot())
        if existing is not None:
            self._validate_snapshot(existing)
            self.journal.append("SNAPSHOT_RECOVERED", snapshot=_snapshot_identity(existing))
            return existing
        from daytona import CreateSnapshotParams, Resources
        name = _snapshot_name(self.campaign_sha256, self.wave_index)
        cpu, memory, disk = self._snapshot_expected()
        self.journal.append(
            "SNAPSHOT_CREATE_INTENT",
            snapshot_name=name,
            resources={"cpu": cpu, "memory_gib": memory, "disk_gib": disk, "gpu": 0},
            base_image=self.config.base_image,
            snapshot_root_sha256=self.campaign["snapshot_manifest_core_sha256"],
        )

        def create():
            self._restore_and_revalidate_upload_snapshot(
                "snapshot.create preflight",
            )
            recovered = self._reconcile_snapshot(self._find_snapshot())
            if recovered is not None:
                return recovered
            create_params = CreateSnapshotParams(
                name=name,
                image=self._build_image(),
                resources=Resources(cpu=cpu, memory=memory, disk=disk, gpu=0),
            )
            try:
                return self._client().snapshot.create(
                    create_params,
                    timeout=min(
                        self.config.worker_timeout_seconds,
                        int(self._remaining_seconds()),
                    ),
                )
            finally:
                self._restore_and_revalidate_upload_snapshot(
                    "snapshot.create SDK consumption",
                )

        snapshot = self._retry("create pristine image snapshot", create)
        self._validate_snapshot(snapshot)
        self.journal.append("SNAPSHOT_CREATED", snapshot=_snapshot_identity(snapshot))
        return snapshot

    def _find_task_sandboxes(self, task_index: int) -> list[object]:
        from daytona import DaytonaNotFoundError, ListSandboxesQuery
        labels = {
            "hfw_manager": MANAGER_LABEL,
            "hfw_campaign_sha256": self.campaign_sha256,
            "hfw_wave_index": f"{self.wave_index:08d}",
            "hfw_role": "cpu-proposal-worker",
            "hfw_task_index": f"{task_index:03d}",
        }
        partial = list(self._retry(
            f"list task {task_index} sandboxes",
            lambda: list(self._client().list(ListSandboxesQuery(labels=labels))),
        ))
        hydrated = []
        for sandbox in partial:
            def get_or_absent(sandbox_id=str(sandbox.id)):
                try:
                    return self._client().get(sandbox_id)
                except DaytonaNotFoundError:
                    return None
            value = self._retry(
                f"hydrate task {task_index} sandbox {sandbox.id}",
                get_or_absent,
            )
            if value is not None:
                hydrated.append(value)
        return hydrated

    def _find_probe_sandboxes(self) -> list[object]:
        from daytona import DaytonaNotFoundError, ListSandboxesQuery

        labels = _probe_labels(self.campaign_sha256, self.wave_index)
        partial = list(self._retry(
            "list exact build-evidence quota probe sandboxes",
            lambda: list(self._client().list(ListSandboxesQuery(labels=labels))),
        ))
        hydrated = []
        for sandbox in partial:
            def get_or_absent(sandbox_id=str(sandbox.id)):
                try:
                    return self._client().get(sandbox_id)
                except DaytonaNotFoundError:
                    return None
            value = self._retry(
                f"hydrate build-evidence quota probe {sandbox.id}",
                get_or_absent,
            )
            if value is not None:
                hydrated.append(value)
        return hydrated

    def _reconcile_task_sandboxes_absent(
        self, task_index: int, reason: str,
    ) -> None:
        deadline = time.monotonic() + min(
            float(self.config.api_timeout_seconds), self._remaining_seconds(),
        )
        relist_count = 0
        while True:
            remaining = self._find_task_sandboxes(task_index)
            relist_count += 1
            if not remaining:
                self.journal.append(
                    "TASK_SANDBOX_ABSENCE_RECONCILED",
                    task_index=task_index,
                    reason=reason,
                    final_remaining_count=0,
                    relist_count=relist_count,
                    exact_campaign_task_labels=True,
                )
                return
            self.journal.append(
                "TASK_SANDBOX_ABSENCE_RELIST_STALE",
                task_index=task_index,
                reason=reason,
                remaining_count=len(remaining),
                remaining_ids=sorted(str(getattr(value, "id")) for value in remaining),
                relist_count=relist_count,
            )
            if time.monotonic() >= deadline:
                raise ManagerError(
                    f"task {task_index} sandboxes remained after bounded exact-label reconciliation"
                )
            for sandbox in remaining:
                self._delete_sandbox(sandbox, reason)
            now = time.monotonic()
            if now >= deadline:
                raise ManagerError(
                    f"task {task_index} sandbox reconciliation deadline expired"
                )
            time.sleep(min(
                float(self.config.poll_seconds),
                deadline - now,
                self._remaining_seconds(),
            ))

    def _validate_sandbox(self, sandbox: object, task_index: int, attempt: int, snapshot: object) -> None:
        identity = _sandbox_identity(sandbox)
        expected_labels = _task_labels(
            self.campaign_sha256, self.wave_index, task_index, attempt,
        )
        cpu, memory, disk = self._snapshot_expected()
        if identity["name"] != _task_name(
            self.campaign_sha256, self.wave_index, task_index, attempt,
        ):
            raise ManagerError("sandbox deterministic name drift")
        if not _contains_task_labels(identity["labels"], expected_labels):
            raise ManagerError("sandbox labels drift")
        if (
            identity["cpu"] != cpu
            or identity["memory_gib"] != memory
            or identity["disk_gib"] != disk
            or identity["gpu"] != 0
        ):
            raise ManagerError("sandbox resources drift")
        if identity["network_block_all"] is not True or identity["public"] is not False:
            raise ManagerError("sandbox isolation drift")
        expected_auto_stop = math.ceil(self.config.worker_timeout_seconds / 60) + 30
        if (
            identity["auto_stop_interval"] != expected_auto_stop
            or identity["auto_delete_interval"] != 0
        ):
            raise ManagerError("sandbox fail-safe lifecycle interval drift")
        if identity["snapshot"] not in {str(getattr(snapshot, "id")), str(getattr(snapshot, "name"))}:
            raise ManagerError("sandbox snapshot binding drift")
        if identity["target"] != self.config.target:
            raise ManagerError("sandbox target region drift")
        observed_env = dict(getattr(sandbox, "env", {}) or {})
        if observed_env != WORKER_ENVIRONMENT:
            raise ManagerError("sandbox user environment is not the sealed minimal environment")
        if any(
            marker in key.upper()
            for key in observed_env
            for marker in ("KEY", "TOKEN", "SECRET", "DAYTONA")
        ):
            raise ManagerError("sandbox environment contains a credential-like key")

    def _create_task_sandbox(self, snapshot: object, task_index: int, attempt: int):
        from daytona import CreateSandboxFromSnapshotParams
        matches = [
            sandbox for sandbox in self._find_task_sandboxes(task_index)
            if dict(getattr(sandbox, "labels", {})).get("hfw_attempt") == str(attempt)
        ]
        if len(matches) > 1:
            raise ManagerError(f"duplicate task sandbox: {task_index}/{attempt}")
        if matches:
            sandbox = matches[0]
            self._validate_sandbox(sandbox, task_index, attempt, snapshot)
            self.journal.append(
                "SANDBOX_RECOVERED", task_index=task_index, attempt=attempt,
                sandbox=_sandbox_identity(sandbox),
            )
            return sandbox
        labels = _task_labels(self.campaign_sha256, self.wave_index, task_index, attempt)
        name = _task_name(self.campaign_sha256, self.wave_index, task_index, attempt)
        auto_stop_minutes = math.ceil(self.config.worker_timeout_seconds / 60) + 30
        auto_delete_minutes = 0
        self.journal.append(
            "SANDBOX_CREATE_INTENT",
            task_index=task_index, attempt=attempt, sandbox_name=name,
            snapshot_id=str(getattr(snapshot, "id")),
            snapshot_name=str(getattr(snapshot, "name")),
            labels=labels,
            network_block_all=True,
            secrets=None,
            environment=WORKER_ENVIRONMENT,
            ephemeral_via_auto_delete_zero=True,
            auto_stop_interval_minutes=auto_stop_minutes,
            auto_delete_interval_minutes=auto_delete_minutes,
        )

        def create():
            recovered = [
                sandbox for sandbox in self._find_task_sandboxes(task_index)
                if dict(getattr(sandbox, "labels", {})).get("hfw_attempt") == str(attempt)
            ]
            if len(recovered) > 1:
                raise ManagerError("duplicate sandbox appeared during create reconciliation")
            if recovered:
                return recovered[0]
            with self._create_semaphore:
                return self._client().create(
                    CreateSandboxFromSnapshotParams(
                        name=name,
                        snapshot=str(getattr(snapshot, "name")),
                        env_vars=dict(WORKER_ENVIRONMENT),
                        labels=labels,
                        public=False,
                        auto_stop_interval=auto_stop_minutes,
                        auto_archive_interval=0,
                        auto_delete_interval=auto_delete_minutes,
                        secrets=None,
                        network_block_all=True,
                        network_allow_list=None,
                        domain_allow_list=None,
                        ephemeral=True,
                    ),
                    timeout=min(self.config.api_timeout_seconds, int(self._remaining_seconds())),
                )

        sandbox = self._retry(f"create worker sandbox {task_index}/{attempt}", create)
        self._validate_sandbox(sandbox, task_index, attempt, snapshot)
        self.journal.append(
            "SANDBOX_CREATED", task_index=task_index, attempt=attempt,
            sandbox=_sandbox_identity(sandbox),
        )
        return sandbox

    def _validate_probe_sandbox(self, sandbox: object, snapshot: object) -> None:
        identity = _sandbox_identity(sandbox)
        expected_labels = _probe_labels(self.campaign_sha256, self.wave_index)
        cpu, memory, disk = self._snapshot_expected()
        if identity["name"] != _probe_name(self.campaign_sha256, self.wave_index):
            raise ManagerError("build-evidence quota probe deterministic name drift")
        if not _contains_probe_labels(identity["labels"], expected_labels):
            raise ManagerError("build-evidence quota probe labels drift")
        if (
            identity["cpu"] != cpu
            or identity["memory_gib"] != memory
            or identity["disk_gib"] != disk
            or identity["gpu"] != 0
        ):
            raise ManagerError("build-evidence quota probe resources drift")
        if identity["network_block_all"] is not True or identity["public"] is not False:
            raise ManagerError("build-evidence quota probe isolation drift")
        expected_auto_stop = math.ceil(self.config.worker_timeout_seconds / 60) + 30
        if (
            identity["auto_stop_interval"] != expected_auto_stop
            or identity["auto_delete_interval"] != 0
        ):
            raise ManagerError("build-evidence quota probe lifecycle drift")
        if identity["snapshot"] not in {
            str(getattr(snapshot, "id")), str(getattr(snapshot, "name")),
        }:
            raise ManagerError("build-evidence quota probe snapshot binding drift")
        if identity["target"] != self.config.target:
            raise ManagerError("build-evidence quota probe target drift")
        if dict(getattr(sandbox, "env", {}) or {}) != WORKER_ENVIRONMENT:
            raise ManagerError("build-evidence quota probe environment drift")

    def _create_probe_sandbox(self, snapshot: object):
        from daytona import CreateSandboxFromSnapshotParams

        matches = self._find_probe_sandboxes()
        if len(matches) > 1:
            raise ManagerError("duplicate build-evidence quota probe sandbox")
        if matches:
            sandbox = matches[0]
            self._validate_probe_sandbox(sandbox, snapshot)
            self.journal.append(
                "PROBE_SANDBOX_RECOVERED",
                sandbox=_sandbox_identity(sandbox),
                worker_command_permitted=False,
            )
            return sandbox
        labels = _probe_labels(self.campaign_sha256, self.wave_index)
        name = _probe_name(self.campaign_sha256, self.wave_index)
        auto_stop_minutes = math.ceil(self.config.worker_timeout_seconds / 60) + 30
        self.journal.append(
            "PROBE_SANDBOX_CREATE_INTENT",
            sandbox_name=name,
            snapshot_id=str(getattr(snapshot, "id")),
            snapshot_name=str(getattr(snapshot, "name")),
            labels=labels,
            network_block_all=True,
            secrets=None,
            environment=WORKER_ENVIRONMENT,
            worker_command_permitted=False,
        )

        def create():
            recovered = self._find_probe_sandboxes()
            if len(recovered) > 1:
                raise ManagerError(
                    "duplicate build-evidence quota probe appeared during create reconciliation"
                )
            if recovered:
                return recovered[0]
            with self._create_semaphore:
                return self._client().create(
                    CreateSandboxFromSnapshotParams(
                        name=name,
                        snapshot=str(getattr(snapshot, "name")),
                        env_vars=dict(WORKER_ENVIRONMENT),
                        labels=labels,
                        public=False,
                        auto_stop_interval=auto_stop_minutes,
                        auto_archive_interval=0,
                        auto_delete_interval=0,
                        secrets=None,
                        network_block_all=True,
                        network_allow_list=None,
                        domain_allow_list=None,
                        ephemeral=True,
                    ),
                    timeout=min(
                        self.config.api_timeout_seconds,
                        int(self._remaining_seconds()),
                    ),
                )

        sandbox = self._retry("create build-evidence quota probe", create)
        self._validate_probe_sandbox(sandbox, snapshot)
        self.journal.append(
            "PROBE_SANDBOX_CREATED",
            sandbox=_sandbox_identity(sandbox),
            worker_command_permitted=False,
        )
        return sandbox

    def _ensure_session_command(self, sandbox: object, task_index: int, attempt: int) -> tuple[str, str, int]:
        from daytona import DaytonaNotFoundError
        session_id = f"w560-{task_index:03d}-a{attempt}"
        command = _worker_command(task_index, self.campaign_sha256)
        try:
            session = sandbox.process.get_session(session_id)
        except DaytonaNotFoundError:
            self._retry(
                f"create session {task_index}/{attempt}",
                lambda: sandbox.process.create_session(session_id),
            )
            session = sandbox.process.get_session(session_id)
        commands = [item for item in (session.commands or []) if item.command == command]
        if len(commands) > 1:
            raise ManagerError(f"duplicate worker commands: {task_index}/{attempt}")
        intent = self.journal.latest(
            "WORKER_COMMAND_INTENT", task_index=task_index, attempt=attempt,
        )
        started_unix_ns = (
            int(intent["started_unix_ns"]) if intent is not None else time.time_ns()
        )
        if commands:
            self.journal.append(
                "WORKER_COMMAND_RECOVERED", task_index=task_index, attempt=attempt,
                sandbox_id=str(getattr(sandbox, "id")), session_id=session_id,
                command_id=str(commands[0].id),
            )
            return session_id, str(commands[0].id), started_unix_ns
        if intent is None:
            self.journal.append(
                "WORKER_COMMAND_INTENT", task_index=task_index, attempt=attempt,
                sandbox_id=str(getattr(sandbox, "id")), session_id=session_id,
                command_sha256=sha256_bytes(command.encode("utf-8")),
                started_unix_ns=started_unix_ns,
                worker_timeout_seconds=self.config.worker_timeout_seconds,
            )
        from daytona import SessionExecuteRequest
        def execute_or_recover() -> str:
            recovered_session = sandbox.process.get_session(session_id)
            recovered_commands = [
                item for item in (recovered_session.commands or [])
                if item.command == command
            ]
            if len(recovered_commands) > 1:
                raise ManagerError(f"duplicate worker commands: {task_index}/{attempt}")
            if recovered_commands:
                return str(recovered_commands[0].id)
            response = sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=command, run_async=True),
                timeout=self.config.api_timeout_seconds,
            )
            return str(response.cmd_id)

        command_id = self._retry(
            f"launch worker command {task_index}/{attempt}",
            execute_or_recover,
        )
        self.journal.append(
            "WORKER_COMMAND_STARTED", task_index=task_index, attempt=attempt,
            sandbox_id=str(getattr(sandbox, "id")), session_id=session_id,
            command_id=command_id, started_unix_ns=started_unix_ns,
        )
        return session_id, command_id, started_unix_ns

    def _wait_command(
        self, sandbox: object, task_index: int, attempt: int,
        session_id: str, command_id: str, started_unix_ns: int,
    ) -> None:
        worker_deadline = started_unix_ns + self.config.worker_timeout_seconds * 1_000_000_000
        while True:
            if self._cancel.is_set():
                raise Cancelled(f"worker {task_index} cancelled")
            self._remaining_seconds()
            if time.time_ns() >= worker_deadline:
                raise TimeoutError(f"worker {task_index} exceeded sealed worker timeout")
            command = self._retry(
                f"poll worker command {task_index}/{attempt}",
                lambda: sandbox.process.get_session_command(session_id, command_id),
            )
            if command.exit_code is not None:
                exit_code = int(command.exit_code)
                self.journal.append(
                    "WORKER_COMMAND_TERMINAL", task_index=task_index, attempt=attempt,
                    sandbox_id=str(getattr(sandbox, "id")), command_id=command_id,
                    exit_code=exit_code,
                )
                if exit_code != 0:
                    deterministic_failure = self._validated_deterministic_failure(
                        sandbox, task_index, attempt, exit_code,
                    )
                    if deterministic_failure is not None:
                        raise deterministic_failure
                    logs = self._retry(
                        f"read failed worker logs {task_index}/{attempt}",
                        lambda: sandbox.process.get_session_command_logs(session_id, command_id),
                    )
                    tail = _safe_error(
                        RuntimeError((logs.stderr or logs.output or "")[-2000:]),
                        self.credential,
                    )
                    raise ManagerError(f"worker {task_index} failed ({exit_code}): {tail}")
                return
            try:
                sandbox.refresh_activity()
            except BaseException:
                pass
            time.sleep(min(self.config.poll_seconds, self._remaining_seconds()))

    def _validated_deterministic_failure(
        self,
        sandbox: object,
        task_index: int,
        attempt: int,
        exit_code: int,
    ) -> DeterministicNumericalWorkerFailure | None:
        policy = self.campaign["worker_failure_policy"]
        if exit_code != int(policy["deterministic_numerical_exit_code"]):
            return None
        relative = (
            f"orchestration/worker-failures/worker-{task_index:03d}/"
            f"attempt-{attempt}/{policy['machine_readable_filename']}"
        )
        local_path = self.config.wave_root / relative
        try:
            self._download_remote(sandbox, REMOTE_FAILURE_PATH, local_path)
            failure, failure_identity = _validate_worker_failure(
                self.campaign,
                self.campaign_sha256,
                self.snapshot_manifest_sha256,
                task_index,
                local_path,
            )
        except BaseException as error:
            self.journal.append(
                "WORKER_FAILURE_ARTIFACT_REJECTED",
                task_index=task_index,
                attempt=attempt,
                exit_code=exit_code,
                local_relative_path=relative,
                error=_safe_error(error, self.credential),
                classified_terminal=False,
                bounded_retry_remains_enabled=True,
            )
            return None
        terminal_error = DeterministicNumericalWorkerFailure(
            task_index=task_index,
            attempt=attempt,
            exit_code=exit_code,
            failure=failure,
            failure_identity=failure_identity,
            failure_relative_path=relative,
        )
        decision_lock = getattr(self, "_deterministic_failure_lock", None)
        if decision_lock is None:
            decision_lock = threading.Lock()
            self._deterministic_failure_lock = decision_lock
        with decision_lock:
            selected_error = self._read_deterministic_failure_decision()
            if selected_error is None:
                selected_error = terminal_error
                decision_identity = self._seal_deterministic_failure_decision(
                    selected_error,
                )
            else:
                decision_identity = hash_file(
                    self.config.wave_root
                    / DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH,
                )
        self.journal.append(
            "DETERMINISTIC_NUMERICAL_WORKER_FAILURE_VALIDATED",
            task_index=selected_error.task_index,
            attempt=selected_error.attempt,
            exit_code=selected_error.exit_code,
            stage=selected_error.failure["stage"],
            failure={
                "sha256": selected_error.failure_identity["sha256"],
                "size_bytes": selected_error.failure_identity["size_bytes"],
                "relative_path": selected_error.failure_relative_path,
            },
            retry_policy=policy["deterministic_numerical_retry_policy"],
            traceback_substring_used_for_classification=False,
            concurrently_observed_task_index=task_index,
            first_validated_failure_decision_wins=True,
            decision={
                "sha256": decision_identity["sha256"],
                "size_bytes": decision_identity["size_bytes"],
            },
        )
        return selected_error

    def _seal_deterministic_failure_decision(
        self,
        error: DeterministicNumericalWorkerFailure,
    ) -> Mapping[str, object]:
        value = {
            "schema": "hfw_w560_daytona_deterministic_failure_decision_v1",
            "status": "VALIDATED_TERMINAL_DO_NOT_RETRY_PENDING_CLEANUP",
            "campaign_sha256": self.campaign_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "wave_index": self.wave_index,
            "task_index": error.task_index,
            "worker_attempt": error.attempt,
            "process_exit_code": error.exit_code,
            "classification": error.failure["classification"],
            "stage": error.failure["stage"],
            "retry_policy": error.failure["retry_policy"],
            "worker_failure_artifact": {
                "relative_path": error.failure_relative_path,
                "sha256": error.failure_identity["sha256"],
                "size_bytes": error.failure_identity["size_bytes"],
            },
            "traceback_substring_used_for_classification": False,
            "cloud_cleanup_pending": True,
            "sandbox_create_or_worker_command_permitted_on_restart": False,
        }
        path = self.config.wave_root / DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH
        if path.exists():
            existing, identity = read_json(path)
            if existing != value:
                raise ManagerError("deterministic failure decision changed")
            return identity
        return write_json_exclusive(path, value)

    def _read_deterministic_failure_decision(
        self,
    ) -> DeterministicNumericalWorkerFailure | None:
        path = self.config.wave_root / DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH
        if not path.exists():
            return None
        decision, _ = read_json(path)
        required = {
            "schema": "hfw_w560_daytona_deterministic_failure_decision_v1",
            "status": "VALIDATED_TERMINAL_DO_NOT_RETRY_PENDING_CLEANUP",
            "campaign_sha256": self.campaign_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "wave_index": self.wave_index,
            "retry_policy": self.campaign["worker_failure_policy"][
                "deterministic_numerical_retry_policy"
            ],
            "traceback_substring_used_for_classification": False,
            "cloud_cleanup_pending": True,
            "sandbox_create_or_worker_command_permitted_on_restart": False,
        }
        for key, expected in required.items():
            if decision.get(key) != expected:
                raise ManagerError(f"deterministic failure decision drift: {key}")
        task_index = decision.get("task_index")
        if (
            not isinstance(task_index, int)
            or isinstance(task_index, bool)
            or not 0 <= task_index < int(self.campaign["worker_count"])
        ):
            raise ManagerError("deterministic failure decision task drift")
        artifact = decision.get("worker_failure_artifact", {})
        relative = artifact.get("relative_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ManagerError("deterministic failure decision path drift")
        failure, identity = _validate_worker_failure(
            self.campaign,
            self.campaign_sha256,
            self.snapshot_manifest_sha256,
            task_index,
            self.config.wave_root / relative,
        )
        if (
            identity["sha256"] != artifact.get("sha256")
            or int(identity["size_bytes"]) != int(artifact.get("size_bytes", -1))
            or failure["classification"] != decision.get("classification")
            or failure["stage"] != decision.get("stage")
            or decision.get("process_exit_code")
            != int(self.campaign["worker_failure_policy"][
                "deterministic_numerical_exit_code"
            ])
        ):
            raise ManagerError("deterministic failure decision artifact drift")
        return DeterministicNumericalWorkerFailure(
            task_index=task_index,
            attempt=int(decision.get("worker_attempt", -1)),
            exit_code=int(decision.get("process_exit_code", -1)),
            failure=failure,
            failure_identity=identity,
            failure_relative_path=relative,
        )

    def _download_remote(
        self, sandbox: object, remote_path: str, local_path: Path,
        expected_sha256: str | None = None, expected_size_bytes: int | None = None,
    ) -> Mapping[str, object]:
        return self._retry(
            f"download {remote_path}",
            lambda: _atomic_stream_download(
                sandbox.fs.download_file_stream(
                    remote_path, timeout=self.config.api_timeout_seconds,
                ),
                local_path,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
            ),
        )

    def _download_and_validate(
        self, sandbox: object, task_index: int, attempt: int,
        launch_environment: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        sandbox_id = str(getattr(sandbox, "id"))
        sandbox_key = hashlib.sha256(sandbox_id.encode("utf-8")).hexdigest()
        staging_root = (
            self.orchestration_root / "download-staging"
            / f"worker-{task_index:03d}" / f"attempt-{attempt}"
            / f"sandbox-{sandbox_key}"
        )
        staging_root.mkdir(parents=True, exist_ok=True)
        result_path = staging_root / OUTPUT_FILENAMES[0]
        flat_path = staging_root / OUTPUT_FILENAMES[1]
        bundle_path = staging_root / OUTPUT_FILENAMES[2]
        self._download_remote(sandbox, REMOTE_RESULT_PATH, result_path)
        result, _ = read_json(result_path)
        declared = result["trajectory_final_group_flat"]
        self._download_remote(
            sandbox, REMOTE_FLAT_PATH, flat_path,
            expected_sha256=str(declared["sha256"]),
            expected_size_bytes=int(declared["size_bytes"]),
        )
        declared_bundle = result["accepted_candidate_bundle"]
        self._download_remote(
            sandbox, REMOTE_BUNDLE_PATH, bundle_path,
            expected_sha256=str(declared_bundle["sha256"]),
            expected_size_bytes=int(declared_bundle["size_bytes"]),
        )
        observed = sorted(path.name for path in staging_root.iterdir() if path.is_file())
        if observed != sorted(OUTPUT_FILENAMES):
            raise ManagerError(f"worker {task_index} attempt-scoped output set drift")
        staged_value = _validate_worker_result(
            self.campaign, self.campaign_sha256, self.snapshot_manifest_sha256,
            task_index, result_path, flat_path, bundle_path, launch_environment,
        )
        if staged_value["environment"].get("campaign_sha256") not in (
            None, self.campaign_sha256,
        ):
            raise ManagerError("unexpected environment campaign identity")
        root = self.download_root / f"worker-{task_index:03d}"
        if root.exists():
            committed = self._validate_existing_output(
                task_index, launch_environment,
            )
            if committed is None:
                raise ManagerError(
                    f"worker {task_index} committed output directory is partial; refusing cross-attempt mix"
                )
            if (
                committed["result"]["sha256"]
                != staged_value["result"]["sha256"]
                or committed["flat"]["sha256"]
                != staged_value["flat"]["sha256"]
                or committed["bundle"]["sha256"]
                != staged_value["bundle"]["sha256"]
            ):
                raise ManagerError(
                    f"worker {task_index} committed output differs from attempt-scoped pair"
                )
            value = committed
        else:
            os.rename(staging_root, root)
            fsync_directory(root.parent)
            value = _validate_worker_result(
                self.campaign,
                self.campaign_sha256,
                self.snapshot_manifest_sha256,
                task_index,
                root / OUTPUT_FILENAMES[0],
                root / OUTPUT_FILENAMES[1],
                root / OUTPUT_FILENAMES[2],
                launch_environment,
            )
        self.journal.append(
            "WORKER_OUTPUT_DOWNLOADED_AND_VALIDATED",
            task_index=task_index,
            attempt=attempt,
            sandbox_id=sandbox_id,
            attempt_scoped_staging=True,
            complete_pair_atomically_adopted=True,
            result={
                "sha256": value["result"]["sha256"],
                "size_bytes": value["result"]["size_bytes"],
            },
            flat={
                "sha256": value["flat"]["sha256"],
                "size_bytes": value["flat"]["size_bytes"],
            },
            accepted_candidate_bundle={
                "sha256": value["bundle"]["sha256"],
                "size_bytes": value["bundle"]["size_bytes"],
                "candidate_count": value["accepted_candidate_count"],
            },
            peak_rss_bytes=value["peak_rss_bytes"],
        )
        return value

    def _validate_existing_output(
        self, task_index: int, launch_environment: Mapping[str, object] | None,
    ) -> Mapping[str, object] | None:
        root = self.download_root / f"worker-{task_index:03d}"
        result = root / OUTPUT_FILENAMES[0]
        flat = root / OUTPUT_FILENAMES[1]
        bundle = root / OUTPUT_FILENAMES[2]
        if not root.exists():
            return None
        if not result.is_file() or not flat.is_file() or not bundle.is_file():
            self.journal.append(
                "PARTIAL_COMMITTED_OUTPUT_TERMINAL_LOCAL_CORRUPTION",
                task_index=task_index,
                result_present=result.is_file(),
                flat_present=flat.is_file(),
                bundle_present=bundle.is_file(),
                cloud_worker_retry_permitted=False,
            )
            raise ManagerError(
                f"worker {task_index} committed output directory is incomplete; "
                "atomic output recovery forbids a cloud rerun"
            )
        return _validate_worker_result(
            self.campaign, self.campaign_sha256, self.snapshot_manifest_sha256,
            task_index, result, flat, bundle, launch_environment,
        )

    def _validate_campaign_sandbox_labels_for_delete(
        self,
        labels: object,
    ) -> None:
        if not isinstance(labels, Mapping):
            raise ManagerError("sandbox deletion labels are missing")
        role = labels.get("hfw_role")
        if role == "build-evidence-quota-probe":
            valid = _contains_probe_labels(
                labels,
                _probe_labels(self.campaign_sha256, self.wave_index),
            )
        elif role == "cpu-proposal-worker":
            task_index = labels.get("hfw_task_index")
            attempt = labels.get("hfw_attempt")
            campaign = getattr(self, "campaign", None)
            worker_count = (
                None if not isinstance(campaign, Mapping)
                else int(campaign["worker_count"])
            )
            if (
                not isinstance(task_index, str)
                or re.fullmatch(r"[0-9]{3}", task_index) is None
                or (
                    worker_count is not None
                    and int(task_index) >= worker_count
                )
                or not isinstance(attempt, str)
                or re.fullmatch(r"0|[1-9][0-9]*", attempt) is None
            ):
                valid = False
            else:
                valid = _contains_task_labels(
                    labels,
                    _task_labels(
                        self.campaign_sha256,
                        self.wave_index,
                        int(task_index),
                        int(attempt),
                    ),
                )
        else:
            valid = False
        if not valid:
            raise ManagerError(
                "refusing to delete sandbox with missing or unexpected hfw labels"
            )

    def _delete_sandbox(
        self, sandbox: object, reason: str,
    ) -> Mapping[str, object]:
        from daytona import DaytonaNotFoundError
        identity = _sandbox_identity(sandbox)
        labels = identity["labels"]
        self._validate_campaign_sandbox_labels_for_delete(labels)
        delete_intent = self.journal.append(
            "SANDBOX_DELETE_INTENT", sandbox=identity, reason=reason,
        )
        delete_observation = {"already_absent": False}
        deadline = time.monotonic() + min(
            float(self.config.api_timeout_seconds), self._remaining_seconds(),
        )
        delete_error: BaseException | None = None
        for attempt in range(self.config.maximum_attempts):
            try:
                sandbox.delete(timeout=self.config.api_timeout_seconds)
                delete_error = None
                break
            except DaytonaNotFoundError:
                delete_observation["already_absent"] = True
                delete_error = None
                break
            except BaseException as error:
                delete_error = error
                self.journal.append(
                    "SDK_CALL_RETRY",
                    operation=f"delete sandbox {identity['id']}",
                    attempt=attempt,
                    error=_safe_error(error, self.credential),
                    bounded_by_exact_deletion_deadline=True,
                )
                now = time.monotonic()
                if attempt + 1 >= self.config.maximum_attempts or now >= deadline:
                    break
                time.sleep(min(
                    self.config.retry_initial_seconds * (2 ** attempt),
                    self.config.retry_maximum_seconds,
                    deadline - now,
                    self._remaining_seconds(),
                ))
        if delete_error is not None:
            self.journal.append(
                "SANDBOX_DELETE_FAILED", sandbox=identity, reason=reason,
                error=_safe_error(delete_error, self.credential),
            )
            raise ManagerError(
                f"delete sandbox {identity['id']} exhausted bounded attempts"
            ) from delete_error
        delete_accepted = self.journal.append(
            "SANDBOX_DELETE_ACCEPTED",
            sandbox=identity,
            reason=reason,
            already_absent=delete_observation["already_absent"],
            exact_id_not_found_confirmed=False,
        )
        exact_id_absent = self._wait_sandbox_id_absent(
            identity, reason, deadline=deadline,
        )
        deleted = self.journal.append(
            "SANDBOX_DELETED",
            sandbox=identity,
            reason=reason,
            exact_id_not_found_confirmed=True,
        )
        return {
            "schema": "hfw_w560_daytona_exact_sandbox_deletion_proof_v1",
            "sandbox_id": identity["id"],
            "reason": reason,
            "receipt_identities": {
                "delete_intent": _mutation_receipt_identity(delete_intent),
                "delete_accepted": _mutation_receipt_identity(delete_accepted),
                "exact_id_absent": _mutation_receipt_identity(exact_id_absent),
                "deleted": _mutation_receipt_identity(deleted),
            },
        }

    def _wait_sandbox_id_absent(
        self,
        identity: Mapping[str, object],
        reason: str,
        *,
        deadline: float,
    ) -> Mapping[str, object]:
        from daytona import DaytonaNotFoundError

        sandbox_id = str(identity["id"])
        poll_count = 0
        error_count = 0
        while True:
            poll_count += 1
            try:
                observed = self._client().get(sandbox_id)
                error_count = 0
            except DaytonaNotFoundError:
                confirmed = getattr(self, "_confirmed_absent_sandbox_ids", None)
                lock = getattr(self, "_cleanup_evidence_lock", None)
                if confirmed is not None:
                    if lock is None:
                        confirmed.add(sandbox_id)
                    else:
                        with lock:
                            confirmed.add(sandbox_id)
                receipt = self.journal.append(
                    "SANDBOX_EXACT_ID_ABSENT",
                    sandbox_id=sandbox_id,
                    reason=reason,
                    exact_id_not_found_confirmed=True,
                    poll_count=poll_count,
                )
                return receipt
            except BaseException as error:
                error_count += 1
                self.journal.append(
                    "SDK_CALL_RETRY",
                    operation=f"confirm sandbox {sandbox_id} deletion",
                    attempt=error_count - 1,
                    error=_safe_error(error, self.credential),
                    bounded_by_exact_deletion_deadline=True,
                    absence_not_inferred=True,
                )
                now = time.monotonic()
                if error_count >= self.config.maximum_attempts or now >= deadline:
                    raise ManagerError(
                        f"exact sandbox {sandbox_id} absence could not be proven before bounded deadline"
                    ) from error
                time.sleep(min(
                    self.config.retry_initial_seconds * (2 ** (error_count - 1)),
                    self.config.retry_maximum_seconds,
                    deadline - now,
                    self._remaining_seconds(),
                ))
                continue
            if str(getattr(observed, "id")) != sandbox_id:
                raise ManagerError(
                    "exact sandbox lookup returned a different sandbox identity"
                )
            labels = dict(getattr(observed, "labels", {}) or {})
            self._validate_campaign_sandbox_labels_for_delete(labels)
            sealed_hfw_labels = {
                name: value
                for name, value in identity["labels"].items()
                if isinstance(name, str) and name.startswith("hfw_")
            }
            if any(
                labels.get(name) != value
                for name, value in sealed_hfw_labels.items()
            ):
                raise ManagerError(
                    "sandbox manager-owned label changed while deleting"
                )
            now = time.monotonic()
            if now >= deadline:
                raise ManagerError(
                    f"sandbox {sandbox_id} remained visible after bounded exact-ID deletion polling"
                )
            self.journal.append(
                "SANDBOX_DELETE_VISIBILITY_STALE",
                sandbox_id=sandbox_id,
                reason=reason,
                poll_count=poll_count,
                observed_state=_state_text(getattr(observed, "state", None)),
            )
            time.sleep(min(
                float(self.config.poll_seconds),
                deadline - now,
                self._remaining_seconds(),
            ))

    def _run_task(
        self, snapshot: object, task_index: int,
        launch_environment: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        existing = self._validate_existing_output(task_index, launch_environment)
        if existing is not None:
            recovery = getattr(self, "recovered_task0", None)
            if task_index == 0 and recovery is not None:
                prior = self.journal.latest(
                    "RECOVERED_AND_REVALIDATED_NO_RERUN", task_index=0,
                )
                expected_result = {
                    "sha256": existing["result"]["sha256"],
                    "size_bytes": existing["result"]["size_bytes"],
                }
                expected_flat = {
                    "sha256": existing["flat"]["sha256"],
                    "size_bytes": existing["flat"]["size_bytes"],
                }
                if prior is None:
                    self.journal.append(
                        "RECOVERED_AND_REVALIDATED_NO_RERUN",
                        task_index=0,
                        provenance_receipt=recovery["identity"],
                        result=expected_result,
                        flat=expected_flat,
                        sandbox_create_permitted=False,
                        worker_command_permitted=False,
                        cloud_search_rerun=False,
                    )
                elif (
                    prior.get("provenance_receipt") != recovery["identity"]
                    or prior.get("result") != expected_result
                    or prior.get("flat") != expected_flat
                    or prior.get("sandbox_create_permitted") is not False
                    or prior.get("worker_command_permitted") is not False
                    or prior.get("cloud_search_rerun") is not False
                ):
                    raise ManagerError("recovered task0 journal receipt drift")
            return existing
        if task_index == 0 and getattr(self, "recovered_task0", None) is not None:
            raise ManagerError(
                "sealed recovered task0 output is missing or invalid; cloud rerun is forbidden"
            )
        last_error: BaseException | None = None
        for attempt in range(self.config.maximum_attempts):
            sandbox = None
            try:
                sandbox = self._create_task_sandbox(snapshot, task_index, attempt)
                session_id, command_id, started = self._ensure_session_command(
                    sandbox, task_index, attempt,
                )
                self._wait_command(
                    sandbox, task_index, attempt, session_id, command_id, started,
                )
                value = self._download_and_validate(
                    sandbox, task_index, attempt, launch_environment,
                )
            except BaseException as error:
                # The atomic directory rename is the durable-output boundary.  A
                # crash or receipt-write failure after that rename must never
                # be classified as a model-attempt failure or trigger another
                # worker command.  Revalidate the fixed root before deciding
                # whether this exception is retryable.
                try:
                    durable_value = self._validate_existing_output(
                        task_index, launch_environment,
                    )
                except BaseException as committed_error:
                    committed_root = self.download_root / f"worker-{task_index:03d}"
                    if committed_root.exists():
                        if sandbox is not None:
                            self._delete_sandbox(
                                sandbox, "TERMINAL_LOCAL_OUTPUT_CORRUPTION",
                            )
                        raise ManagerError(
                            f"worker {task_index} has terminal committed-output "
                            "corruption; cloud rerun forbidden"
                        ) from committed_error
                    durable_value = None
                if durable_value is not None:
                    self.journal.append(
                        "DURABLE_OUTPUT_RECOVERED_AFTER_POST_COMMIT_EXCEPTION_NO_RERUN",
                        task_index=task_index,
                        attempt=attempt,
                        sandbox_id=(
                            None if sandbox is None
                            else str(getattr(sandbox, "id"))
                        ),
                        original_error=_safe_error(error, self.credential),
                        result={
                            "sha256": durable_value["result"]["sha256"],
                            "size_bytes": durable_value["result"]["size_bytes"],
                        },
                        flat={
                            "sha256": durable_value["flat"]["sha256"],
                            "size_bytes": durable_value["flat"]["size_bytes"],
                        },
                        next_worker_attempt_permitted=False,
                    )
                    if sandbox is not None:
                        try:
                            self._delete_sandbox(
                                sandbox, "OUTPUT_DURABLE_POST_COMMIT_RECOVERY",
                            )
                        except BaseException as cleanup_error:
                            self.journal.append(
                                "VALIDATED_OUTPUT_CLEANUP_PENDING_NO_RERUN",
                                task_index=task_index,
                                attempt=attempt,
                                sandbox_id=str(getattr(sandbox, "id")),
                                result={
                                    "sha256": durable_value["result"]["sha256"],
                                    "size_bytes": durable_value["result"]["size_bytes"],
                                },
                                flat={
                                    "sha256": durable_value["flat"]["sha256"],
                                    "size_bytes": durable_value["flat"]["size_bytes"],
                                },
                                error=_safe_error(cleanup_error, self.credential),
                                next_worker_attempt_permitted=False,
                            )
                            raise ManagerError(
                                f"worker {task_index} output is durably validated "
                                "but sandbox cleanup remains pending; model rerun forbidden"
                            ) from cleanup_error
                    return durable_value
                last_error = error
                self.journal.append(
                    "WORKER_ATTEMPT_FAILED", task_index=task_index, attempt=attempt,
                    sandbox_id=None if sandbox is None else str(getattr(sandbox, "id")),
                    error=_safe_error(error, self.credential),
                )
                if sandbox is not None:
                    try:
                        self._delete_sandbox(sandbox, "FAILED_ATTEMPT")
                    except BaseException as cleanup_error:
                        self.journal.append(
                            "FAILED_ATTEMPT_CLEANUP_BLOCKED_RETRY",
                            task_index=task_index,
                            attempt=attempt,
                            sandbox_id=str(getattr(sandbox, "id")),
                            error=_safe_error(cleanup_error, self.credential),
                            next_worker_attempt_permitted=False,
                        )
                        raise ManagerError(
                            f"worker {task_index} attempt {attempt} cleanup did not prove exact-ID absence; retry blocked"
                        ) from cleanup_error
                if isinstance(error, DeterministicNumericalWorkerFailure):
                    raise
                if isinstance(error, (Cancelled, TimeoutError)):
                    raise
                if attempt + 1 < self.config.maximum_attempts:
                    time.sleep(min(
                        self.config.retry_initial_seconds * (2 ** attempt),
                        self.config.retry_maximum_seconds,
                        self._remaining_seconds(),
                    ))
                continue
            try:
                assert sandbox is not None
                self._delete_sandbox(sandbox, "OUTPUT_VALIDATED")
            except BaseException as cleanup_error:
                self.journal.append(
                    "VALIDATED_OUTPUT_CLEANUP_PENDING_NO_RERUN",
                    task_index=task_index,
                    attempt=attempt,
                    sandbox_id=str(getattr(sandbox, "id")),
                    result={
                        "sha256": value["result"]["sha256"],
                        "size_bytes": value["result"]["size_bytes"],
                    },
                    flat={
                        "sha256": value["flat"]["sha256"],
                        "size_bytes": value["flat"]["size_bytes"],
                    },
                    error=_safe_error(cleanup_error, self.credential),
                    next_worker_attempt_permitted=False,
                )
                raise ManagerError(
                    f"worker {task_index} output is durably validated but sandbox cleanup remains pending; model rerun forbidden"
                ) from cleanup_error
            return value
        assert last_error is not None
        raise ManagerError(
            f"worker {task_index} exhausted attempts: {_safe_error(last_error, self.credential)}"
        )

    def _read_build_evidence(self, seed: object) -> Mapping[str, object]:
        evidence_root = self.orchestration_root / "build-evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        wheel_path = evidence_root / "wheel-sha256.txt"
        freeze_path = evidence_root / "pip-freeze.txt"
        wheel_identity = self._download_remote(
            seed, f"{REMOTE_BUILD_EVIDENCE_ROOT}/wheel-sha256.txt", wheel_path,
        )
        freeze_identity = self._download_remote(
            seed, f"{REMOTE_BUILD_EVIDENCE_ROOT}/pip-freeze.txt", freeze_path,
        )
        wheel_lines = [line.strip() for line in wheel_path.read_text().splitlines() if line.strip()]
        if len(wheel_lines) != len(EXPECTED_WHEEL_SHA256):
            raise ManagerError("snapshot build wheel closure count drift")
        wheels = []
        for line in wheel_lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not _SHA256.fullmatch(parts[0]):
                raise ManagerError("malformed wheel SHA-256 evidence")
            wheels.append({"sha256": parts[0], "filename": Path(parts[1]).name})
        observed_wheels = {value["filename"]: value["sha256"] for value in wheels}
        if observed_wheels != EXPECTED_WHEEL_SHA256:
            raise ManagerError("snapshot build wheel closure hash drift")
        if not any("torch-" in value["filename"] and "+cpu" in value["filename"] for value in wheels):
            raise ManagerError("recorded torch wheel is not explicitly CPU-only")
        if not any("numpy-" in value["filename"] for value in wheels):
            raise ManagerError("numpy wheel hash is missing")
        return {
            "wheel_sha256_file": wheel_identity,
            "pip_freeze_file": freeze_identity,
            "wheels": wheels,
            "locked_runtime_packages": list(LOCKED_RUNTIME_PACKAGES),
            "torch_cpu_index_url": self.campaign["proposal_lane"]["sealed_environment"]["torch_cpu_index_url"],
            "offline_no_deps_install_from_verified_wheelhouse": True,
        }

    def _quota_receipt(self, seed: object, additional_workers: int) -> Mapping[str, object]:
        from daytona_api_client.api.organizations_api import OrganizationsApi

        def quota_number(value: object, field: str) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ManagerError(f"invalid live Daytona quota value: {field}")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ManagerError(f"invalid live Daytona quota value: {field}")
            return number

        expected_additional_workers = int(self.campaign["worker_count"]) - 1
        if (
            type(additional_workers) is not int
            or additional_workers != expected_additional_workers
        ):
            raise ManagerError("live quota additional worker count drift")

        probe_id = _sdk_identity_text(
            getattr(seed, "id", None), "quota probe sandbox ID",
        )
        organization_id = _sdk_identity_text(
            getattr(seed, "organization_id", None), "organization ID",
        )
        region_id = _sdk_identity_text(
            getattr(seed, "target", None), "quota probe region ID",
        )
        sandbox_class = _sdk_identity_text(
            getattr(seed, "sandbox_class", None), "sandbox class",
        )
        if region_id != self.config.target:
            raise ManagerError("live quota probe region drift")

        cpu, memory, disk = self._snapshot_expected()
        exact_probe_resources = {
            "cpu": cpu,
            "memory_gib": memory,
            "disk_gib": disk,
            "gpu": 0,
        }
        probe_identity = _sandbox_identity(seed)
        if (
            probe_identity.get("id") != probe_id
            or probe_identity.get("target") != region_id
            or any(
                probe_identity.get(resource) != expected
                for resource, expected in exact_probe_resources.items()
            )
        ):
            raise ManagerError("live quota probe exact resource identity drift")

        request_timeout_seconds = min(
            float(self.config.api_timeout_seconds),
            float(self._remaining_seconds()),
        )
        if (
            not math.isfinite(request_timeout_seconds)
            or request_timeout_seconds <= 0
        ):
            raise ManagerError("live organization usage request timeout is invalid")

        client = self._client()
        overview = self._retry(
            "read live organization usage",
            lambda: OrganizationsApi(
                client._api_client,
            ).get_organization_usage_overview(
                organization_id,
                _request_timeout=request_timeout_seconds,
            ),
        )
        region_usage = getattr(overview, "region_usage", None)
        if not isinstance(region_usage, list):
            raise ManagerError("live organization usage rows are missing")
        rows = []
        for row in region_usage:
            row_region_id = _sdk_identity_text(
                getattr(row, "region_id", None), "usage row region ID",
            )
            if row_region_id != region_id:
                continue
            row_sandbox_class = _sdk_identity_text(
                getattr(row, "sandbox_class", None),
                "usage row sandbox class",
            )
            if row_sandbox_class == sandbox_class:
                rows.append(row)
        if len(rows) != 1:
            raise ManagerError("could not bind a unique live region usage row")
        row = rows[0]

        current = {
            "cpu": quota_number(row.current_cpu_usage, "current CPU"),
            "memory_gib": quota_number(
                row.current_memory_usage, "current memory",
            ),
            "disk_gib": quota_number(row.current_disk_usage, "current disk"),
            "gpu": quota_number(row.current_gpu_usage, "current GPU"),
        }
        totals = {
            "cpu": quota_number(row.total_cpu_quota, "total CPU"),
            "memory_gib": quota_number(row.total_memory_quota, "total memory"),
            "disk_gib": quota_number(row.total_disk_quota, "total disk"),
            "gpu": quota_number(row.total_gpu_quota, "total GPU"),
        }
        if any(current[key] > totals[key] for key in current):
            raise ManagerError("current live Daytona usage exceeds quota")

        maximum_fields = {
            "max_cpu_per_sandbox": cpu,
            "max_memory_per_sandbox": memory,
            "max_disk_per_sandbox": disk,
        }
        per_sandbox_maximum = {}
        null_fields = []
        missing = object()
        for field, required in maximum_fields.items():
            raw = getattr(row, field, missing)
            if raw is missing:
                raise ManagerError(
                    f"live Daytona usage row is missing {field}"
                )
            if raw is None:
                per_sandbox_maximum[field] = None
                null_fields.append(field)
                continue
            maximum = quota_number(raw, field)
            if maximum < required:
                raise ManagerError(
                    f"live Daytona {field} is below the exact probe resource"
                )
            per_sandbox_maximum[field] = maximum

        projected = {
            "cpu": current["cpu"] + additional_workers * cpu,
            "memory_gib": current["memory_gib"] + additional_workers * memory,
            "disk_gib": current["disk_gib"] + additional_workers * disk,
            "gpu": current["gpu"],
        }
        if any(projected[key] > totals[key] for key in projected):
            raise ManagerError(f"live Daytona quota cannot fit fanout: {projected} > {totals}")
        value = {
            "schema": "hfw_w560_daytona_live_quota_before_fanout_v3",
            "status": "PASS_EXACT_REGION_SANDBOX_CLASS_FANOUT_FITS",
            "probe_sandbox_id": probe_id,
            "organization_id": organization_id,
            "region_id": region_id,
            "sandbox_class": sandbox_class,
            "region_binding_source": "API_KEY_AUTHENTICATED_SANDBOX_TARGET",
            "sandbox_class_binding_source": (
                "API_KEY_AUTHENTICATED_SANDBOX_CLASS"
            ),
            "quota_usage_source": (
                "API_KEY_AUTHENTICATED_ORGANIZATION_USAGE"
            ),
            "organization_usage_request_timeout_seconds": (
                request_timeout_seconds
            ),
            "forbidden_sandbox_region_quota_endpoint_used": False,
            "current": current,
            "additional_workers": additional_workers,
            "projected": projected,
            "quota": totals,
            **per_sandbox_maximum,
            "per_sandbox_maximum_null_semantics": {
                "schema": (
                    "hfw_w560_daytona_per_sandbox_maximum_null_semantics_v1"
                ),
                "null_fields": sorted(null_fields),
                "null_means_unlimited": False,
                "null_numeric_limit_asserted": False,
                "null_acceptance_basis": (
                    "ALREADY_LIVE_COMMAND_FREE_PROBE_AT_EXACT_REQUESTED_RESOURCES"
                ),
                "exact_resource_probe_already_live": True,
                "exact_resource_probe_resources": exact_probe_resources,
                "non_null_limits_enforced": True,
            },
            "fits": True,
            "measured_after_seed_pass_before_fanout": True,
        }
        self.journal.append("LIVE_QUOTA_PASS", quota=value)
        return value

    def _validate_live_quota_receipt(
        self,
        receipt: object,
        probe_identity: Mapping[str, object],
    ) -> None:
        expected_fields = {
            "schema", "status", "probe_sandbox_id", "organization_id",
            "region_id", "sandbox_class", "region_binding_source",
            "sandbox_class_binding_source", "quota_usage_source",
            "organization_usage_request_timeout_seconds",
            "forbidden_sandbox_region_quota_endpoint_used", "current",
            "additional_workers", "projected", "quota", "fits",
            "max_cpu_per_sandbox", "max_memory_per_sandbox",
            "max_disk_per_sandbox", "per_sandbox_maximum_null_semantics",
            "measured_after_seed_pass_before_fanout",
        }
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != expected_fields
            or receipt.get("schema")
            != "hfw_w560_daytona_live_quota_before_fanout_v3"
            or receipt.get("status")
            != "PASS_EXACT_REGION_SANDBOX_CLASS_FANOUT_FITS"
            or receipt.get("probe_sandbox_id") != probe_identity.get("id")
            or receipt.get("region_id") != probe_identity.get("target")
            or receipt.get("region_id") != self.config.target
            or receipt.get("organization_id")
            != probe_identity.get("organization_id")
            or receipt.get("sandbox_class")
            != probe_identity.get("sandbox_class")
            or receipt.get("region_binding_source")
            != "API_KEY_AUTHENTICATED_SANDBOX_TARGET"
            or receipt.get("sandbox_class_binding_source")
            != "API_KEY_AUTHENTICATED_SANDBOX_CLASS"
            or receipt.get("quota_usage_source")
            != "API_KEY_AUTHENTICATED_ORGANIZATION_USAGE"
            or receipt.get("forbidden_sandbox_region_quota_endpoint_used")
            is not False
            or receipt.get("fits") is not True
            or receipt.get("measured_after_seed_pass_before_fanout") is not True
            or type(receipt.get("additional_workers")) is not int
            or receipt.get("additional_workers")
            != int(self.campaign["worker_count"]) - 1
        ):
            raise ManagerError("live quota receipt identity or source drift")

        request_timeout = receipt.get(
            "organization_usage_request_timeout_seconds"
        )
        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, (int, float))
            or not math.isfinite(float(request_timeout))
            or float(request_timeout) <= 0
            or float(request_timeout) > float(self.config.api_timeout_seconds)
        ):
            raise ManagerError("live quota receipt request timeout drift")

        resource_keys = {"cpu", "memory_gib", "disk_gib", "gpu"}
        mappings = {}
        for name in ("current", "projected", "quota"):
            values = receipt.get(name)
            if not isinstance(values, Mapping) or set(values) != resource_keys:
                raise ManagerError(f"live quota receipt {name} shape drift")
            normalized = {}
            for resource, value in values.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise ManagerError(
                        f"live quota receipt {name} value drift: {resource}"
                    )
                normalized[resource] = float(value)
            mappings[name] = normalized

        current = mappings["current"]
        projected = mappings["projected"]
        totals = mappings["quota"]
        additional_workers = int(receipt["additional_workers"])
        cpu, memory, disk = self._snapshot_expected()
        exact_probe_resources = {
            "cpu": cpu,
            "memory_gib": memory,
            "disk_gib": disk,
            "gpu": 0,
        }
        if any(
            probe_identity.get(resource) != expected
            for resource, expected in exact_probe_resources.items()
        ):
            raise ManagerError("live quota receipt exact probe resource drift")

        maximum_fields = {
            "max_cpu_per_sandbox": cpu,
            "max_memory_per_sandbox": memory,
            "max_disk_per_sandbox": disk,
        }
        null_fields = []
        for field, required in maximum_fields.items():
            value = receipt.get(field)
            if value is None:
                null_fields.append(field)
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < required
            ):
                raise ManagerError(
                    f"live quota receipt {field} value drift"
                )
        expected_null_semantics = {
            "schema": (
                "hfw_w560_daytona_per_sandbox_maximum_null_semantics_v1"
            ),
            "null_fields": sorted(null_fields),
            "null_means_unlimited": False,
            "null_numeric_limit_asserted": False,
            "null_acceptance_basis": (
                "ALREADY_LIVE_COMMAND_FREE_PROBE_AT_EXACT_REQUESTED_RESOURCES"
            ),
            "exact_resource_probe_already_live": True,
            "exact_resource_probe_resources": exact_probe_resources,
            "non_null_limits_enforced": True,
        }
        if canonical_json_bytes(
            receipt.get("per_sandbox_maximum_null_semantics")
        ) != canonical_json_bytes(expected_null_semantics):
            raise ManagerError(
                "live quota receipt per-sandbox null semantics drift"
            )

        expected_projected = {
            "cpu": current["cpu"] + additional_workers * cpu,
            "memory_gib": current["memory_gib"] + additional_workers * memory,
            "disk_gib": current["disk_gib"] + additional_workers * disk,
            "gpu": current["gpu"],
        }
        if any(
            not math.isclose(
                projected[key], expected_projected[key],
                rel_tol=0.0, abs_tol=1e-9,
            )
            for key in resource_keys
        ):
            raise ManagerError("live quota receipt projection arithmetic drift")
        if any(
            current[key] > totals[key] or projected[key] > totals[key]
            for key in resource_keys
        ):
            raise ManagerError("live quota receipt capacity drift")

    def _cost_receipt(self) -> Mapping[str, object]:
        workers = int(self.campaign["worker_count"])
        cpu, memory, disk = self._snapshot_expected()
        billable_disk = max(0, disk - 5)
        hourly = (
            Decimal(self.config.vcpu_hour_rate_usd) * cpu * workers
            + Decimal(self.config.memory_gib_hour_rate_usd) * memory * workers
            + Decimal(self.config.disk_gib_hour_rate_usd) * billable_disk * workers
        )
        maximum = hourly * Decimal(self.config.campaign_timeout_seconds) / Decimal(3600)
        return {
            "workers": workers,
            "per_worker": {
                "cpu": cpu, "memory_gib": memory, "disk_gib": disk,
                "free_disk_gib": 5, "billable_disk_gib": billable_disk,
            },
            "rates_usd": {
                "vcpu_hour": self.config.vcpu_hour_rate_usd,
                "memory_gib_hour": self.config.memory_gib_hour_rate_usd,
                "disk_gib_hour": self.config.disk_gib_hour_rate_usd,
            },
            "estimated_full_fanout_usd_per_hour": format(hourly, "f"),
            "sealed_campaign_timeout_hours": format(
                Decimal(self.config.campaign_timeout_seconds) / Decimal(3600), "f",
            ),
            "estimated_timeout_cap_usd": format(maximum, "f"),
            "estimate_not_invoice": True,
        }

    def _fanout_deadline_safety_receipt(
        self, seed_result: Mapping[str, object],
    ) -> Mapping[str, object]:
        observed_seconds = float(seed_result["worker_total_seconds"])
        observed_physical = int(seed_result["total_physical_model_forwards"])
        maximum_physical = int(
            self.campaign["solver"]["total_physical_model_forwards_maximum"]
        )
        if (
            not math.isfinite(observed_seconds)
            or observed_seconds <= 0
            or observed_physical <= 0
            or maximum_physical < observed_physical
        ):
            raise ManagerError("seed timing/forward evidence cannot project fanout")
        projected_seconds = observed_seconds * maximum_physical / observed_physical
        infrastructure_margin_seconds = 2 * self.config.api_timeout_seconds
        worker_window_required = math.ceil(
            projected_seconds + infrastructure_margin_seconds
        )
        cleanup_reserve_seconds = max(
            10 * 60, self.config.api_timeout_seconds * 2,
        )
        campaign_window_required = worker_window_required + cleanup_reserve_seconds
        remaining_seconds = math.floor(self._remaining_seconds())
        receipt = {
            "schema": "hfw_w560_seed_measured_fanout_deadline_gate_v1",
            "status": "PASS_WORST_CASE_WORKER_AND_CAMPAIGN_WINDOWS_FIT",
            "seed_observed_total_seconds": observed_seconds,
            "seed_observed_total_physical_model_forwards": observed_physical,
            "campaign_maximum_total_physical_model_forwards": maximum_physical,
            "projection_formula": (
                "seed_seconds_times_maximum_physical_forwards_divided_by_"
                "seed_observed_physical_forwards"
            ),
            "projected_worst_case_worker_seconds": projected_seconds,
            "infrastructure_margin_seconds": infrastructure_margin_seconds,
            "infrastructure_margin_formula": "two_times_api_timeout_seconds",
            "worker_window_required_seconds": worker_window_required,
            "sealed_worker_timeout_seconds": self.config.worker_timeout_seconds,
            "cleanup_reserve_seconds": cleanup_reserve_seconds,
            "campaign_window_required_seconds": campaign_window_required,
            "campaign_remaining_seconds_at_gate": remaining_seconds,
            "fanout_worker_count": int(self.campaign["worker_count"]) - 1,
            "seed_full_result_validated_before_projection": True,
            "no_fanout_worker_created_before_gate": True,
        }
        self._enforce_fanout_deadline_safety(receipt)
        return receipt

    def _validate_fanout_deadline_safety_receipt(
        self,
        receipt: Mapping[str, object],
        seed_result: Mapping[str, object],
    ) -> None:
        observed_seconds = float(seed_result["worker_total_seconds"])
        observed_physical = int(seed_result["total_physical_model_forwards"])
        maximum_physical = int(
            self.campaign["solver"]["total_physical_model_forwards_maximum"]
        )
        projected_seconds = observed_seconds * maximum_physical / observed_physical
        infrastructure_margin = 2 * self.config.api_timeout_seconds
        worker_required = math.ceil(projected_seconds + infrastructure_margin)
        cleanup_reserve = max(10 * 60, self.config.api_timeout_seconds * 2)
        campaign_required = worker_required + cleanup_reserve
        exact = {
            "schema": "hfw_w560_seed_measured_fanout_deadline_gate_v1",
            "status": "PASS_WORST_CASE_WORKER_AND_CAMPAIGN_WINDOWS_FIT",
            "seed_observed_total_seconds": observed_seconds,
            "seed_observed_total_physical_model_forwards": observed_physical,
            "campaign_maximum_total_physical_model_forwards": maximum_physical,
            "projection_formula": (
                "seed_seconds_times_maximum_physical_forwards_divided_by_"
                "seed_observed_physical_forwards"
            ),
            "infrastructure_margin_seconds": infrastructure_margin,
            "infrastructure_margin_formula": "two_times_api_timeout_seconds",
            "worker_window_required_seconds": worker_required,
            "sealed_worker_timeout_seconds": self.config.worker_timeout_seconds,
            "cleanup_reserve_seconds": cleanup_reserve,
            "campaign_window_required_seconds": campaign_required,
            "fanout_worker_count": int(self.campaign["worker_count"]) - 1,
            "seed_full_result_validated_before_projection": True,
            "no_fanout_worker_created_before_gate": True,
        }
        for key, expected in exact.items():
            if receipt.get(key) != expected:
                raise ManagerError(f"fanout deadline receipt drift: {key}")
        if not math.isclose(
            float(receipt.get("projected_worst_case_worker_seconds")),
            projected_seconds,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ManagerError("fanout deadline projection arithmetic drift")
        remaining_at_gate = receipt.get("campaign_remaining_seconds_at_gate")
        if (
            not isinstance(remaining_at_gate, int)
            or isinstance(remaining_at_gate, bool)
            or remaining_at_gate < campaign_required
        ):
            raise ManagerError("fanout deadline original campaign window drift")

    def _enforce_fanout_deadline_safety(
        self, receipt: Mapping[str, object],
    ) -> None:
        if (
            receipt.get("schema")
            != "hfw_w560_seed_measured_fanout_deadline_gate_v1"
            or receipt.get("status")
            != "PASS_WORST_CASE_WORKER_AND_CAMPAIGN_WINDOWS_FIT"
        ):
            raise ManagerError("fanout deadline gate receipt drift")
        worker_required = int(receipt["worker_window_required_seconds"])
        campaign_required = int(receipt["campaign_window_required_seconds"])
        if worker_required > self.config.worker_timeout_seconds:
            raise ManagerError(
                "seed-projected worst-case worker exceeds sealed worker timeout"
            )
        if campaign_required > math.floor(self._remaining_seconds()):
            raise ManagerError(
                "seed-projected worst-case fanout plus cleanup exceeds remaining campaign"
            )

    def _validate_probe_identity_value(
        self,
        probe_identity: Mapping[str, object],
        snapshot: object | None,
    ) -> None:
        expected_fields = {
            "id", "name", "state", "desired_state", "snapshot", "target",
            "cpu", "memory_gib", "disk_gib", "gpu", "network_block_all",
            "public", "auto_stop_interval", "auto_delete_interval", "labels",
            "organization_id", "sandbox_class",
        }
        organization_id = _sdk_identity_text(
            probe_identity.get("organization_id"),
            "sealed probe organization ID",
        )
        sandbox_class = _sdk_identity_text(
            probe_identity.get("sandbox_class"),
            "sealed probe sandbox class",
        )
        required_labels = _probe_labels(
            self.campaign_sha256, self.wave_index,
        )
        cpu, memory, disk = self._snapshot_expected()
        expected_auto_stop = math.ceil(
            self.config.worker_timeout_seconds / 60,
        ) + 30
        if (
            set(probe_identity) != expected_fields
            or probe_identity.get("organization_id") != organization_id
            or probe_identity.get("sandbox_class") != sandbox_class
            or not isinstance(probe_identity.get("id"), str)
            or not probe_identity.get("id")
            or probe_identity.get("name")
            != _probe_name(self.campaign_sha256, self.wave_index)
            or not isinstance(probe_identity.get("state"), str)
            or not isinstance(probe_identity.get("desired_state"), str)
            or not isinstance(probe_identity.get("snapshot"), str)
            or not probe_identity.get("snapshot")
            or probe_identity.get("target") != self.config.target
            or any(
                type(probe_identity.get(name)) is not int
                for name in (
                    "cpu", "memory_gib", "disk_gib", "gpu",
                    "auto_stop_interval", "auto_delete_interval",
                )
            )
            or probe_identity.get("cpu") != cpu
            or probe_identity.get("memory_gib") != memory
            or probe_identity.get("disk_gib") != disk
            or probe_identity.get("gpu") != 0
            or probe_identity.get("network_block_all") is not True
            or probe_identity.get("public") is not False
            or probe_identity.get("auto_stop_interval") != expected_auto_stop
            or probe_identity.get("auto_delete_interval") != 0
            or not _contains_probe_labels(
                probe_identity.get("labels"), required_labels,
            )
        ):
            raise ManagerError("command-free probe full sandbox identity drift")
        if snapshot is not None and probe_identity["snapshot"] not in {
            str(getattr(snapshot, "id")), str(getattr(snapshot, "name")),
        }:
            raise ManagerError("command-free probe snapshot binding drift")

    def _journal_receipt_by_identity(
        self,
        identity: object,
        expected_event: str,
    ) -> Mapping[str, object]:
        if not isinstance(identity, Mapping) or set(identity) != {
            "receipt_index", "event", "sha256", "size_bytes",
        }:
            raise ManagerError("probe deletion receipt identity shape drift")
        index = identity.get("receipt_index")
        records = self.journal.records
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(records)
        ):
            raise ManagerError("probe deletion receipt index drift")
        record = records[index]
        if (
            _mutation_receipt_identity(record) != dict(identity)
            or record.get("event") != expected_event
            or record.get("campaign_sha256") != self.campaign_sha256
        ):
            raise ManagerError("probe deletion receipt identity drift")
        return record

    def _validate_probe_deletion_proof(
        self,
        probe_identity: Mapping[str, object],
        proof: object,
    ) -> None:
        reason = "BUILD_EVIDENCE_AND_QUOTA_PROBE_COMPLETE"
        if (
            not isinstance(proof, Mapping)
            or set(proof) != {
                "schema", "sandbox_id", "reason", "receipt_identities",
            }
            or proof.get("schema")
            != "hfw_w560_daytona_exact_sandbox_deletion_proof_v1"
            or proof.get("sandbox_id") != probe_identity.get("id")
            or proof.get("reason") != reason
        ):
            raise ManagerError("command-free probe exact deletion proof drift")
        identities = proof.get("receipt_identities")
        expected_events = {
            "delete_intent": "SANDBOX_DELETE_INTENT",
            "delete_accepted": "SANDBOX_DELETE_ACCEPTED",
            "exact_id_absent": "SANDBOX_EXACT_ID_ABSENT",
            "deleted": "SANDBOX_DELETED",
        }
        if not isinstance(identities, Mapping) or set(identities) != set(
            expected_events
        ):
            raise ManagerError("command-free probe deletion receipt set drift")
        records = {
            name: self._journal_receipt_by_identity(
                identities[name], expected_event,
            )
            for name, expected_event in expected_events.items()
        }
        indices = [
            int(identities[name]["receipt_index"])
            for name in expected_events
        ]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ManagerError("command-free probe deletion receipt order drift")
        historical_sandbox_identity = dict(probe_identity)
        historical_sandbox_identity.pop("organization_id", None)
        historical_sandbox_identity.pop("sandbox_class", None)
        for name in ("delete_intent", "delete_accepted", "deleted"):
            if (
                canonical_json_bytes(records[name].get("sandbox"))
                != canonical_json_bytes(historical_sandbox_identity)
                or records[name].get("reason") != reason
            ):
                raise ManagerError(
                    "command-free probe deletion receipt sandbox binding drift"
                )
        accepted = records["delete_accepted"]
        exact_absent = records["exact_id_absent"]
        deleted = records["deleted"]
        if (
            not isinstance(accepted.get("already_absent"), bool)
            or accepted.get("exact_id_not_found_confirmed") is not False
            or exact_absent.get("sandbox_id") != probe_identity["id"]
            or exact_absent.get("reason") != reason
            or exact_absent.get("exact_id_not_found_confirmed") is not True
            or not isinstance(exact_absent.get("poll_count"), int)
            or isinstance(exact_absent.get("poll_count"), bool)
            or int(exact_absent["poll_count"]) < 1
            or deleted.get("exact_id_not_found_confirmed") is not True
        ):
            raise ManagerError("command-free probe exact-ID deletion receipt drift")

    def _launch_manifest(
        self, snapshot: object, seed_result: Mapping[str, object],
        build_evidence: Mapping[str, object], quota: Mapping[str, object],
        fanout_deadline_safety: Mapping[str, object],
        probe_identity: Mapping[str, object],
        probe_deletion_proof: Mapping[str, object],
    ) -> Mapping[str, object]:
        path = self.config.wave_root / "LAUNCH_MANIFEST.json"
        environment = dict(seed_result["environment"])
        recovery = getattr(self, "recovered_task0", None)
        recovered_seed = recovery is not None
        self._validate_fanout_deadline_safety_receipt(
            fanout_deadline_safety, seed_result,
        )
        required_probe_labels = _probe_labels(
            self.campaign_sha256, self.wave_index,
        )
        self._validate_probe_identity_value(probe_identity, snapshot)
        self._validate_probe_deletion_proof(
            probe_identity, probe_deletion_proof,
        )
        self._validate_live_quota_receipt(quota, probe_identity)
        value = {
            "schema": "hfw_w560_daytona_launch_manifest_v1",
            "status": "SEED_WORKER_0_PASS_BEFORE_TASKS_1_THROUGH_99",
            "campaign_sha256": self.campaign_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "snapshot_manifest_core_sha256": self.campaign["snapshot_manifest_core_sha256"],
            "wave_index": self.wave_index,
            "base_head_record_sha256": self.campaign["base_head_record_sha256"],
            "base_checkpoint_sha256": self.campaign["base_checkpoint_sha256"],
            "snapshot": _snapshot_identity(snapshot),
            "snapshot_pristine_image_built_not_post_worker_filesystem": True,
            "base_image": self.config.base_image,
            "base_image_digest": self.config.base_image.rsplit("@sha256:", 1)[1],
            "target": self.config.target,
            "api_url": self.config.api_url,
            "build_evidence": build_evidence,
            "torch_install": {
                "index_url": self.campaign["proposal_lane"]["sealed_environment"]["torch_cpu_index_url"],
                "runtime_version": self.campaign["proposal_lane"]["sealed_environment"]["torch_version"],
                "full_dependency_closure_pinned_and_wheel_hash_verified": True,
                "offline_install_no_deps_after_verification": True,
            },
            "seed_worker": {
                "task_index": 0,
                "result_sha256": seed_result["result"]["sha256"],
                "result_size_bytes": seed_result["result"]["size_bytes"],
                "flat_sha256": seed_result["flat"]["sha256"],
                "flat_size_bytes": seed_result["flat"]["size_bytes"],
                "accepted_candidate_bundle_sha256": seed_result["bundle"]["sha256"],
                "accepted_candidate_bundle_size_bytes": seed_result["bundle"]["size_bytes"],
                "accepted_candidate_count": seed_result["accepted_candidate_count"],
                "peak_rss_bytes": seed_result["peak_rss_bytes"],
                "base_full_search_mean_ce": seed_result[
                    "base_full_search_mean_ce"
                ],
                "terminal_full_search_mean_ce": seed_result[
                    "terminal_full_search_mean_ce"
                ],
                "self_calibrated_full_search_improvement": seed_result[
                    "self_calibrated_full_search_improvement"
                ],
                "acceptance_gate_attempts": seed_result[
                    "acceptance_gate_attempts"
                ],
                "acceptance_gate_passes": seed_result[
                    "acceptance_gate_passes"
                ],
                "accepted_updates": seed_result["accepted_updates"],
                "memory_limit_bytes": int(
                    self.campaign["proposal_lane"]["memory_limit_gib"]
                ) * 1024 ** 3,
                "peak_rss_strictly_below_memory_limit": True,
                "execution_origin": (
                    "RECOVERED_AND_REVALIDATED_NO_RERUN"
                    if recovered_seed else "EXECUTED_IN_THIS_MANAGER_RUN"
                ),
                "recovery_provenance_receipt": (
                    None if not recovered_seed
                    else recovery["identity"]
                ),
                "task0_worker_sandbox_created_by_this_manager": False,
                "task0_model_command_started_by_this_manager": False,
            },
            "sealed_worker_environment": environment,
            "live_quota_before_fanout": quota,
            "cost_estimate": self._cost_receipt(),
            "fanout_deadline_safety": dict(fanout_deadline_safety),
            "fanout_task_indices": list(range(1, int(self.campaign["worker_count"]))),
            "build_evidence_quota_probe": {
                "role": "build-evidence-quota-probe",
                "task_index": None,
                "worker_command_permitted": False,
                "model_forward_permitted": False,
                "sandbox": dict(probe_identity),
                "full_observed_labels": dict(probe_identity["labels"]),
                "required_manager_labels": required_probe_labels,
                "required_manager_labels_subset_passed": True,
                "unexpected_hfw_labels_absent": True,
                "exact_id_deletion_confirmed_before_manifest": True,
                "exact_id_deletion_proof": dict(probe_deletion_proof),
            },
            "recovered_historical_same_campaign_output_count": (
                1 if recovered_seed else 0
            ),
            "new_worker_outputs_required_after_launch_manifest": (
                int(self.campaign["worker_count"]) - (1 if recovered_seed else 0)
            ),
            "worker_count": int(self.campaign["worker_count"]),
            "worker_network_block_all": True,
            "worker_secrets": None,
            "worker_credentials_present": False,
            "worker_verify_or_lineage_present": False,
            "distinct_fp64_shadow_state_surrogate": True,
            "canonical_checkpoint_and_model_source_bytes_untouched": True,
            "canonical_fp32_state_contract_executed_by_proposal_shadow": False,
            "proposal_shadow_replaces_canonical_fp32_state_storage": True,
            "raw_ce_cross_lane_comparable": False,
            "worker_selection_metric": "SELF_CALIBRATED_BASE_MINUS_TERMINAL_CE",
            "mps_sole_promotion_authority": True,
            "cloud_results_provisional_not_promoted": True,
            "checkpoint_or_head_writes": 0,
        }
        if path.exists():
            existing, _ = read_json(path)
            if canonical_json_bytes(existing) != canonical_json_bytes(value):
                raise ManagerError("launch manifest changed across restart")
            return existing
        write_json_exclusive(path, value)
        self.journal.append(
            "LAUNCH_MANIFEST_SEALED",
            launch_manifest=hash_file(path),
            seed_peak_rss_bytes=seed_result["peak_rss_bytes"],
        )
        return value

    def _read_existing_launch_manifest(
        self, snapshot: object | None = None,
        seed_result: Mapping[str, object] | None = None,
    ) -> Mapping[str, object] | None:
        path = self.config.wave_root / "LAUNCH_MANIFEST.json"
        if not path.exists():
            return None
        value, _ = read_json(path)
        recovery = getattr(self, "recovered_task0", None)
        required = {
            "schema": "hfw_w560_daytona_launch_manifest_v1",
            "status": "SEED_WORKER_0_PASS_BEFORE_TASKS_1_THROUGH_99",
            "campaign_sha256": self.campaign_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "snapshot_manifest_core_sha256": self.campaign["snapshot_manifest_core_sha256"],
            "wave_index": self.wave_index,
            "base_head_record_sha256": self.campaign["base_head_record_sha256"],
            "base_checkpoint_sha256": self.campaign["base_checkpoint_sha256"],
            "base_image": self.config.base_image,
            "target": self.config.target,
            "api_url": self.config.api_url,
            "worker_count": int(self.campaign["worker_count"]),
            "recovered_historical_same_campaign_output_count": (
                1 if recovery is not None else 0
            ),
            "new_worker_outputs_required_after_launch_manifest": (
                int(self.campaign["worker_count"])
                - (1 if recovery is not None else 0)
            ),
            "checkpoint_or_head_writes": 0,
        }
        for key, expected in required.items():
            if canonical_json_bytes(value.get(key)) != canonical_json_bytes(expected):
                raise ManagerError(f"existing launch manifest drift: {key}")
        sealed = self.campaign["proposal_lane"]["sealed_environment"]
        seed_worker = value.get("seed_worker")
        if not isinstance(seed_worker, Mapping):
            raise ManagerError("existing launch seed worker receipt missing")
        expected_origin = (
            "RECOVERED_AND_REVALIDATED_NO_RERUN"
            if recovery is not None
            else "EXECUTED_IN_THIS_MANAGER_RUN"
        )
        expected_provenance = (
            None if recovery is None
            else recovery["identity"]
        )
        if (
            seed_worker.get("execution_origin") != expected_origin
            or seed_worker.get("recovery_provenance_receipt")
            != expected_provenance
            or seed_worker.get("task0_worker_sandbox_created_by_this_manager")
            is not False
            or seed_worker.get("task0_model_command_started_by_this_manager")
            is not False
        ):
            raise ManagerError("existing launch recovered-seed disclosure drift")
        probe = value.get("build_evidence_quota_probe")
        if not isinstance(probe, Mapping):
            raise ManagerError("existing launch probe disclosure missing")
        probe_sandbox = probe.get("sandbox")
        required_probe_labels = _probe_labels(
            self.campaign_sha256, self.wave_index,
        )
        probe_deletion_proof = probe.get("exact_id_deletion_proof")
        expected_probe_disclosure = {
            "role": "build-evidence-quota-probe",
            "task_index": None,
            "worker_command_permitted": False,
            "model_forward_permitted": False,
            "full_observed_labels": (
                None if not isinstance(probe_sandbox, Mapping)
                else probe_sandbox.get("labels")
            ),
            "required_manager_labels": required_probe_labels,
            "required_manager_labels_subset_passed": True,
            "unexpected_hfw_labels_absent": True,
            "exact_id_deletion_confirmed_before_manifest": True,
            "exact_id_deletion_proof": probe_deletion_proof,
        }
        if (
            not isinstance(probe_sandbox, Mapping)
            or set(probe) != {"sandbox", *expected_probe_disclosure}
            or canonical_json_bytes({
                name: probe.get(name)
                for name in expected_probe_disclosure
            }) != canonical_json_bytes(expected_probe_disclosure)
        ):
            raise ManagerError("existing launch probe disclosure drift")
        self._validate_probe_identity_value(probe_sandbox, snapshot)
        self._validate_probe_deletion_proof(
            probe_sandbox, probe_deletion_proof,
        )
        self._validate_live_quota_receipt(
            value.get("live_quota_before_fanout"), probe_sandbox,
        )
        for key in (
            "python_major_minor", "platform_system", "machine", "torch_version",
            "torch_cuda_version", "numpy_version",
        ):
            if value["sealed_worker_environment"].get(key) != sealed[key]:
                raise ManagerError(f"existing launch environment drift: {key}")
        if snapshot is not None:
            current = _snapshot_identity(snapshot)
            if (
                value["snapshot"].get("id") != current["id"]
                or value["snapshot"].get("name") != current["name"]
            ):
                raise ManagerError("existing launch manifest snapshot binding drift")
        if seed_result is not None:
            seed = value["seed_worker"]
            expected_seed = {
                "result_sha256": seed_result["result"]["sha256"],
                "result_size_bytes": seed_result["result"]["size_bytes"],
                "flat_sha256": seed_result["flat"]["sha256"],
                "flat_size_bytes": seed_result["flat"]["size_bytes"],
                "accepted_candidate_bundle_sha256": seed_result["bundle"]["sha256"],
                "accepted_candidate_bundle_size_bytes": seed_result["bundle"]["size_bytes"],
                "accepted_candidate_count": seed_result["accepted_candidate_count"],
                "peak_rss_bytes": seed_result["peak_rss_bytes"],
                "base_full_search_mean_ce": seed_result[
                    "base_full_search_mean_ce"
                ],
                "terminal_full_search_mean_ce": seed_result[
                    "terminal_full_search_mean_ce"
                ],
                "self_calibrated_full_search_improvement": seed_result[
                    "self_calibrated_full_search_improvement"
                ],
                "acceptance_gate_attempts": seed_result[
                    "acceptance_gate_attempts"
                ],
                "acceptance_gate_passes": seed_result[
                    "acceptance_gate_passes"
                ],
                "accepted_updates": seed_result["accepted_updates"],
            }
            for key, expected in expected_seed.items():
                if canonical_json_bytes(seed.get(key)) != canonical_json_bytes(expected):
                    raise ManagerError(f"existing launch seed drift: {key}")
            deadline_safety = value.get("fanout_deadline_safety")
            if not isinstance(deadline_safety, Mapping):
                raise ManagerError("existing launch fanout deadline safety missing")
            self._validate_fanout_deadline_safety_receipt(
                deadline_safety, seed_result,
            )
        return value

    def _existing_download_terminal(self) -> Mapping[str, object] | None:
        path = self.config.wave_root / "CLOUD_DOWNLOADS_COMPLETE.json"
        if not path.exists():
            return None
        terminal, _ = read_json(path)
        if (
            terminal.get("schema") != "hfw_w560_daytona_cloud_download_terminal_v1"
            or terminal.get("status") != "PASS_ALL_100_PROVISIONAL_CPU_OUTPUTS_DOWNLOADED"
            or terminal.get("campaign_sha256") != self.campaign_sha256
            or int(terminal.get("worker_count", -1)) != int(self.campaign["worker_count"])
        ):
            raise ManagerError("existing cloud download terminal drift")
        recovered_count = (
            1 if getattr(self, "recovered_task0", None) is not None else 0
        )
        worker_count = int(self.campaign["worker_count"])
        if (
            terminal.get("recovered_historical_same_campaign_output_count")
            != recovered_count
            or terminal.get("new_worker_outputs_downloaded_this_recovery")
            != worker_count - recovered_count
            or terminal.get("recovered_task_indices")
            != ([0] if recovered_count else [])
            or terminal.get("new_worker_task_indices")
            != list(range(recovered_count, worker_count))
        ):
            raise ManagerError("existing cloud download recovery accounting drift")
        launch = self._read_existing_launch_manifest()
        if launch is None:
            raise ManagerError("download terminal exists without launch manifest")
        environment = launch["sealed_worker_environment"]
        for task_index in range(int(self.campaign["worker_count"])):
            value = self._validate_existing_output(task_index, environment)
            if value is None:
                raise ManagerError("download terminal exists with missing worker output")
            row = terminal["workers"][task_index]
            expected_origin = (
                "RECOVERED_AND_REVALIDATED_NO_RERUN"
                if recovered_count and task_index == 0
                else "EXECUTED_AND_DOWNLOADED_IN_THIS_RECOVERY_RUN"
            )
            if (
                int(row["task_index"]) != task_index
                or row["result_sha256"] != value["result"]["sha256"]
                or row["flat_sha256"] != value["flat"]["sha256"]
                or row.get("accepted_candidate_bundle_sha256")
                != value["bundle"]["sha256"]
                or row.get("accepted_candidate_count")
                != value["accepted_candidate_count"]
                or row.get("execution_origin") != expected_origin
            ):
                raise ManagerError(f"download terminal worker binding drift: {task_index}")
        return terminal

    def _all_campaign_sandboxes(self) -> list[object]:
        from daytona import DaytonaNotFoundError, ListSandboxesQuery
        labels = {
            "hfw_manager": MANAGER_LABEL,
            "hfw_campaign_sha256": self.campaign_sha256,
            "hfw_wave_index": f"{self.wave_index:08d}",
        }
        partial = list(self._retry(
            "list all exact campaign sandboxes",
            lambda: list(self._client().list(ListSandboxesQuery(labels=labels))),
        ))
        hydrated = []
        for sandbox in partial:
            def get_or_absent(sandbox_id=str(sandbox.id)):
                try:
                    return self._client().get(sandbox_id)
                except DaytonaNotFoundError:
                    return None
            value = self._retry(
                f"hydrate cleanup sandbox {sandbox.id}",
                get_or_absent,
            )
            if value is not None:
                hydrated.append(value)
        return hydrated

    def _reconcile_all_campaign_sandboxes_absent(
        self,
        reason: str,
        discovered_ids: set[str] | None = None,
    ) -> list[object]:
        relist_count = 0
        while True:
            remaining = self._all_campaign_sandboxes()
            if discovered_ids is not None:
                discovered_ids.update(
                    str(getattr(value, "id")) for value in remaining
                )
            relist_count += 1
            if not remaining:
                self.journal.append(
                    "CAMPAIGN_SANDBOX_ABSENCE_RECONCILED",
                    reason=reason,
                    final_remaining_count=0,
                    relist_count=relist_count,
                    exact_campaign_labels=True,
                )
                return []
            self.journal.append(
                "CAMPAIGN_SANDBOX_ABSENCE_RELIST_STALE",
                reason=reason,
                remaining_count=len(remaining),
                remaining_ids=sorted(str(getattr(value, "id")) for value in remaining),
                relist_count=relist_count,
            )
            for sandbox in remaining:
                self._delete_sandbox(sandbox, reason)
            time.sleep(min(
                float(self.config.poll_seconds), self._remaining_seconds(),
            ))

    def _reconcile_recovered_exact_sandbox_id(
        self,
        sealed_sandbox: Mapping[str, object],
        *,
        role: str,
    ) -> Mapping[str, object]:
        from daytona import DaytonaNotFoundError

        if role not in {
            "ORIGINAL_SOURCE_TASK0",
            "PREDECESSOR_LABEL_DRIFT_PROBE",
            "IMMEDIATE_PREDECESSOR_QUOTA_API_PROBE",
        }:
            raise ManagerError("recovered exact-ID reconciliation role drift")
        sandbox_id = str(sealed_sandbox["id"])
        deadline = time.monotonic() + min(
            float(self.config.api_timeout_seconds), self._remaining_seconds(),
        )
        error_count = 0
        while True:
            try:
                observed = self._client().get(sandbox_id)
                break
            except DaytonaNotFoundError:
                self.journal.append(
                    "RECOVERED_PREDECESSOR_EXACT_ID_ABSENT",
                    role=role,
                    sandbox_id=sandbox_id,
                    exact_id_not_found_confirmed=True,
                    source_label_list_used_as_substitute=False,
                )
                return {
                    "role": role,
                    "sandbox_id": sandbox_id,
                    "exact_id_not_found_confirmed": True,
                    "delete_required": False,
                }
            except BaseException as error:
                error_count += 1
                self.journal.append(
                    "SDK_CALL_RETRY",
                    operation=(
                        f"reconcile recovered {role.lower()} sandbox {sandbox_id}"
                    ),
                    attempt=error_count - 1,
                    error=_safe_error(error, self.credential),
                    absence_not_inferred=True,
                    bounded_by_recovery_precloud_deadline=True,
                )
                now = time.monotonic()
                if error_count >= self.config.maximum_attempts or now >= deadline:
                    raise ManagerError(
                        f"recovered {role.lower()} exact-ID absence could not be established"
                    ) from error
                time.sleep(min(
                    self.config.retry_initial_seconds * (2 ** (error_count - 1)),
                    self.config.retry_maximum_seconds,
                    deadline - now,
                    self._remaining_seconds(),
                ))
        observed_identity = _sandbox_identity(observed)
        if role == "ORIGINAL_SOURCE_TASK0":
            required_labels = _task_labels(
                self.campaign_sha256, self.wave_index, 0, 0,
            )
            labels_valid = _contains_task_labels(
                observed_identity["labels"], required_labels,
            )
        else:
            required_labels = _probe_labels(
                self.campaign_sha256, self.wave_index,
            )
            labels_valid = _contains_probe_labels(
                observed_identity["labels"], required_labels,
            )
        stable_fields = (
            "id", "name", "snapshot", "target", "cpu", "memory_gib",
            "disk_gib", "gpu", "network_block_all", "public",
            "auto_stop_interval", "auto_delete_interval",
        )
        if not labels_valid or any(
            observed_identity[name] != sealed_sandbox[name]
            for name in stable_fields
        ):
            raise ManagerError(f"recovered {role.lower()} live sandbox identity drift")
        self._delete_sandbox(observed, f"RECOVERY_PRECLOUD_{role}")
        self.journal.append(
            "RECOVERED_PREDECESSOR_EXACT_ID_RECONCILED",
            role=role,
            sandbox_id=sandbox_id,
            observed_sandbox=observed_identity,
            exact_id_not_found_confirmed=True,
            delete_required=True,
        )
        return {
            "role": role,
            "sandbox_id": sandbox_id,
            "exact_id_not_found_confirmed": True,
            "delete_required": True,
        }

    def _required_recovered_exact_id_reconciliations(
        self,
    ) -> list[Mapping[str, object]]:
        recovery = getattr(self, "recovered_task0", None)
        if recovery is None:
            raise ManagerError("recovered exact-ID reconciliation lacks provenance")
        rows = recovery.get("required_exact_id_reconciliation")
        expected_roles = [
            "ORIGINAL_SOURCE_TASK0",
            "PREDECESSOR_LABEL_DRIFT_PROBE",
            "IMMEDIATE_PREDECESSOR_QUOTA_API_PROBE",
        ]
        if not isinstance(rows, list) or len(rows) != 3:
            raise ManagerError(
                "recovered exact-ID reconciliation must contain three entries"
            )
        normalized: list[Mapping[str, object]] = []
        for index, expected_role in enumerate(expected_roles):
            row = rows[index]
            if (
                not isinstance(row, Mapping)
                or row.get("role") != expected_role
                or not isinstance(row.get("sandbox"), Mapping)
                or not isinstance(row["sandbox"].get("id"), str)
                or not row["sandbox"].get("id")
            ):
                raise ManagerError(
                    "recovered exact-ID reconciliation entry drift"
                )
            normalized.append({
                "role": expected_role,
                "sandbox": dict(row["sandbox"]),
            })
        sandbox_ids = [str(row["sandbox"]["id"]) for row in normalized]
        if len(set(sandbox_ids)) != 3:
            raise ManagerError("recovery precloud exact sandbox IDs are not distinct")
        return normalized

    def _reconcile_recovered_source_task0_exact_id(self) -> Mapping[str, object]:
        rows = self._required_recovered_exact_id_reconciliations()
        return self._reconcile_recovered_exact_sandbox_id(
            rows[0]["sandbox"], role=str(rows[0]["role"]),
        )

    def _reconcile_recovered_failed_probe_exact_id(self) -> Mapping[str, object]:
        rows = self._required_recovered_exact_id_reconciliations()
        return self._reconcile_recovered_exact_sandbox_id(
            rows[1]["sandbox"], role=str(rows[1]["role"]),
        )

    def _recovery_precloud_zero_resource_gate(self) -> Mapping[str, object]:
        recovery = getattr(self, "recovered_task0", None)
        if recovery is None:
            raise ManagerError("recovery precloud gate requires sealed provenance")
        required_rows = self._required_recovered_exact_id_reconciliations()
        required_ids = [
            str(row["sandbox"]["id"])
            for row in required_rows
        ]
        path = self.config.wave_root / "RECOVERY_PRECLOUD_ZERO_RESOURCE_COMPLETE.json"
        if path.exists():
            existing, identity = read_json(path)
            required = {
                "schema": RECOVERY_PRECLOUD_SCHEMA_V3,
                "status": (
                    "PASS_ALL_REQUIRED_EXACT_IDS_AND_CAMPAIGN_RESOURCES_ABSENT"
                ),
                "campaign_sha256": self.campaign_sha256,
                "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
                "wave_index": self.wave_index,
                "recovery_provenance": recovery["identity"],
                "required_exact_id_count": 3,
                "required_exact_sandbox_ids": required_ids,
                "all_required_exact_ids_not_found_confirmed": True,
                "exact_campaign_label_relist_passed": True,
                "final_campaign_sandboxes_remaining": 0,
                "snapshot_exact_name_not_found_confirmed": True,
                "snapshot_present": False,
                "sandbox_snapshot_or_probe_create_before_gate": False,
                "task0_model_command_started": False,
                "checkpoint_or_head_writes": 0,
            }
            reconciliations = existing.get("exact_id_reconciliations")
            expected_reconciliations = []
            if isinstance(reconciliations, list):
                for index, row in enumerate(reconciliations):
                    if not isinstance(row, Mapping):
                        break
                    expected_reconciliations.append({
                        "role": required_rows[index]["role"]
                        if index < len(required_rows) else None,
                        "sandbox_id": required_ids[index]
                        if index < len(required_ids) else None,
                        "exact_id_not_found_confirmed": True,
                        "delete_required": row.get("delete_required"),
                    })
            if (
                len(expected_reconciliations) != 3
                or reconciliations != expected_reconciliations
                or any(
                    not isinstance(row.get("delete_required"), bool)
                    for row in expected_reconciliations
                )
                or existing != {
                    **required,
                    "exact_id_reconciliations": reconciliations,
                }
            ):
                raise ManagerError("recovery precloud zero-resource terminal drift")
            self.journal.append(
                "RECOVERY_PRECLOUD_ZERO_RESOURCE_GATE_REUSED_NO_CLOUD_CALLS",
                terminal=identity,
                recovery_provenance=recovery["identity"],
                cloud_calls=0,
            )
            return existing
        exact_id_reconciliations = [
            self._reconcile_recovered_exact_sandbox_id(
                row["sandbox"], role=str(row["role"]),
            )
            for row in required_rows
        ]
        all_exact_ids_confirmed = all(
            evidence["exact_id_not_found_confirmed"] is True
            for evidence in exact_id_reconciliations
        )
        if not all_exact_ids_confirmed:
            raise ManagerError("recovery precloud exact-ID proof did not pass")
        failures = self._cleanup(None, "RECOVERY_PRECLOUD_ZERO_RESOURCE_GATE")
        if failures:
            raise ManagerError(
                f"recovery precloud cleanup did not prove zero resources: {failures}"
            )
        cleanup = self._validated_zero_cleanup_evidence()
        value = {
            "schema": RECOVERY_PRECLOUD_SCHEMA_V3,
            "status": (
                "PASS_ALL_REQUIRED_EXACT_IDS_AND_CAMPAIGN_RESOURCES_ABSENT"
            ),
            "campaign_sha256": self.campaign_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "wave_index": self.wave_index,
            "recovery_provenance": recovery["identity"],
            "required_exact_id_count": 3,
            "required_exact_sandbox_ids": required_ids,
            "exact_id_reconciliations": [
                dict(evidence) for evidence in exact_id_reconciliations
            ],
            "all_required_exact_ids_not_found_confirmed": (
                all_exact_ids_confirmed
            ),
            "exact_campaign_label_relist_passed": cleanup[
                "exact_campaign_label_relist_passed"
            ],
            "final_campaign_sandboxes_remaining": cleanup[
                "final_remaining_count"
            ],
            "snapshot_exact_name_not_found_confirmed": cleanup[
                "snapshot_exact_name_not_found_confirmed"
            ],
            "snapshot_present": cleanup["snapshot_present"],
            "sandbox_snapshot_or_probe_create_before_gate": False,
            "task0_model_command_started": False,
            "checkpoint_or_head_writes": 0,
        }
        identity = write_json_exclusive(path, value)
        self.journal.append(
            "RECOVERY_PRECLOUD_ZERO_RESOURCE_GATE_PASSED",
            terminal={
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            },
            cleanup_evidence=dict(cleanup),
            exact_id_reconciliations=[
                dict(evidence) for evidence in exact_id_reconciliations
            ],
            no_cloud_create_before_gate=True,
        )
        return value

    def _cleanup(self, snapshot: object | None, reason: str) -> list[str]:
        failures: list[str] = []
        self._cleanup_mode = True
        self._cleanup_deadline_monotonic = time.monotonic() + max(
            10 * 60, self.config.api_timeout_seconds * 2,
        )
        initial_count: int | None = None
        discovered_ids: set[str] = set()
        try:
            sandboxes = self._all_campaign_sandboxes()
            initial_count = len(sandboxes)
            discovered_ids.update(
                str(getattr(value, "id")) for value in sandboxes
            )
        except BaseException as error:
            failures.append(_safe_error(error, self.credential))
            sandboxes = []
        with ThreadPoolExecutor(max_workers=min(32, max(1, len(sandboxes)))) as executor:
            futures = {
                executor.submit(self._delete_sandbox, sandbox, reason): sandbox
                for sandbox in sandboxes
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as error:
                    failures.append(_safe_error(error, self.credential))
        candidate = snapshot
        if candidate is None:
            try:
                candidate = self._find_snapshot()
            except BaseException as error:
                failures.append(_safe_error(error, self.credential))
        if candidate is not None:
            try:
                self._delete_snapshot_exact(candidate, reason)
            except BaseException as error:
                failures.append(_safe_error(error, self.credential))
        final_remaining_count: int | None = None
        final_remaining_ids: list[str] | None = None
        final_relist_succeeded = False
        try:
            remaining = self._reconcile_all_campaign_sandboxes_absent(
                reason, discovered_ids,
            )
            final_remaining_count = len(remaining)
            final_remaining_ids = sorted(
                str(getattr(value, "id")) for value in remaining
            )
            final_relist_succeeded = True
            if remaining:
                failures.append(
                    f"{len(remaining)} exact campaign-labelled sandboxes remain after cleanup"
                )
        except BaseException as error:
            failures.append(_safe_error(error, self.credential))
            try:
                remaining = self._all_campaign_sandboxes()
                discovered_ids.update(
                    str(getattr(value, "id")) for value in remaining
                )
                final_remaining_count = len(remaining)
                final_remaining_ids = sorted(
                    str(getattr(value, "id")) for value in remaining
                )
                final_relist_succeeded = True
                if remaining:
                    failures.append(
                        f"{len(remaining)} exact campaign-labelled sandboxes remain after cleanup fallback relist"
                    )
            except BaseException as relist_error:
                failures.append(_safe_error(relist_error, self.credential))
        snapshot_present_after_cleanup: bool | None = None
        snapshot_relist_succeeded = False
        try:
            snapshot_present_after_cleanup = self._find_snapshot() is not None
            snapshot_relist_succeeded = True
            if snapshot_present_after_cleanup:
                failures.append("exact campaign snapshot remains after cleanup")
        except BaseException as error:
            failures.append(_safe_error(error, self.credential))
        exact_campaign_label_relist_passed = (
            final_relist_succeeded and final_remaining_count == 0
        )
        snapshot_exact_name_not_found_confirmed = (
            snapshot_relist_succeeded and snapshot_present_after_cleanup is False
        )
        confirmed_absent = set(
            getattr(self, "_confirmed_absent_sandbox_ids", set())
        )
        cleanup_evidence = {
            "initial_campaign_sandbox_count": initial_count,
            "unique_campaign_sandboxes_discovered": len(discovered_ids),
            "unique_campaign_sandbox_ids_discovered": sorted(discovered_ids),
            "exact_id_not_found_confirmation_count": len(
                discovered_ids & confirmed_absent
            ),
            "final_remaining_count": final_remaining_count,
            "final_remaining_ids": final_remaining_ids,
            "final_relist_succeeded": final_relist_succeeded,
            "snapshot_present": snapshot_present_after_cleanup,
            "snapshot_relist_succeeded": snapshot_relist_succeeded,
            "exact_campaign_label_relist_passed": exact_campaign_label_relist_passed,
            "snapshot_exact_name_not_found_confirmed": snapshot_exact_name_not_found_confirmed,
        }
        self._last_cleanup_evidence = cleanup_evidence
        self.journal.append(
            "CLEANUP_TERMINAL",
            reason=reason,
            failure_count=len(failures),
            failures=failures,
            initial_campaign_sandbox_count=initial_count,
            unique_campaign_sandboxes_discovered=len(discovered_ids),
            unique_campaign_sandbox_ids_discovered=sorted(discovered_ids),
            exact_id_not_found_confirmation_count=len(
                discovered_ids & confirmed_absent
            ),
            campaign_sandbox_count=final_remaining_count,
            final_remaining_count=final_remaining_count,
            final_remaining_ids=final_remaining_ids,
            final_relist_succeeded=final_relist_succeeded,
            snapshot_present=snapshot_present_after_cleanup,
            snapshot_relist_succeeded=snapshot_relist_succeeded,
            exact_campaign_label_relist_passed=exact_campaign_label_relist_passed,
            snapshot_exact_name_not_found_confirmed=snapshot_exact_name_not_found_confirmed,
        )
        self._cleanup_mode = False
        return failures

    def _validated_zero_cleanup_evidence(self) -> Mapping[str, object]:
        evidence = self._last_cleanup_evidence
        if not isinstance(evidence, Mapping):
            raise ManagerError("structured cleanup evidence is missing")
        required = {
            "final_remaining_count": 0,
            "final_relist_succeeded": True,
            "snapshot_present": False,
            "snapshot_relist_succeeded": True,
            "exact_campaign_label_relist_passed": True,
            "snapshot_exact_name_not_found_confirmed": True,
        }
        for key, expected in required.items():
            observed = evidence.get(key)
            if isinstance(expected, bool):
                matches = observed is expected
            else:
                matches = (
                    isinstance(observed, int)
                    and not isinstance(observed, bool)
                    and observed == expected
                )
            if not matches:
                raise ManagerError(f"cleanup evidence does not prove zero resources: {key}")
        return evidence

    def _read_deterministic_failure_terminal(self) -> Mapping[str, object] | None:
        path = self.config.wave_root / DETERMINISTIC_FAILURE_TERMINAL_FILENAME
        if not path.exists():
            return None
        terminal, _ = read_json(path)
        required = {
            "schema": "hfw_w560_daytona_deterministic_failure_terminal_v1",
            "status": "FAIL_DETERMINISTIC_NUMERICAL_FAILURE_CLEANED_TERMINAL",
            "campaign_sha256": self.campaign_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "wave_index": self.wave_index,
            "campaign_sandboxes_remaining": 0,
            "snapshot_present": False,
            "cleanup_failures": [],
            "checkpoint_or_head_writes": 0,
            "deterministic_numerical_attempts": 1,
            "failed_task_numerical_attempts": 1,
            "blind_numerical_retries": 0,
        }
        for key, expected in required.items():
            if terminal.get(key) != expected:
                raise ManagerError(f"deterministic failure terminal drift: {key}")
        task_index = terminal.get("task_index")
        worker_count = int(self.campaign["worker_count"])
        if (
            not isinstance(task_index, int)
            or isinstance(task_index, bool)
            or not 0 <= task_index < worker_count
        ):
            raise ManagerError("sealed deterministic terminal task index drift")
        started_tasks, downloaded_tasks = self._terminal_activity_task_sets()
        fanout_started = terminal.get("fanout_tasks_started")
        downloads = terminal.get("cloud_results_downloaded")
        if (
            not isinstance(fanout_started, int)
            or isinstance(fanout_started, bool)
            or fanout_started != len(started_tasks)
            or not isinstance(downloads, int)
            or isinstance(downloads, bool)
            or downloads != len(downloaded_tasks)
            or (task_index > 0 and task_index not in started_tasks)
            or (task_index == 0 and fanout_started != 0)
        ):
            raise ManagerError("sealed deterministic terminal activity count drift")
        artifact = terminal.get("worker_failure_artifact", {})
        relative = artifact.get("relative_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ManagerError("deterministic terminal failure path drift")
        failure_path = self.config.wave_root / relative
        failure, identity = _validate_worker_failure(
            self.campaign,
            self.campaign_sha256,
            self.snapshot_manifest_sha256,
            task_index,
            failure_path,
        )
        if (
            identity["sha256"] != artifact.get("sha256")
            or int(identity["size_bytes"]) != int(artifact.get("size_bytes", -1))
            or failure["stage"] != terminal.get("stage")
        ):
            raise ManagerError("deterministic terminal artifact binding drift")
        decision = terminal.get("validated_decision", {})
        if decision.get("relative_path") != DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH:
            raise ManagerError("deterministic terminal decision path drift")
        decision_identity = hash_file(
            self.config.wave_root / DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH,
            str(decision.get("sha256")),
        )
        if int(decision_identity["size_bytes"]) != int(decision.get("size_bytes", -1)):
            raise ManagerError("deterministic terminal decision size drift")
        return terminal

    def _terminal_activity_task_sets(self) -> tuple[set[int], set[int]]:
        worker_count = int(self.campaign["worker_count"])
        command_events = {
            "WORKER_COMMAND_STARTED",
            "WORKER_COMMAND_RECOVERED",
            "WORKER_COMMAND_TERMINAL",
            "WORKER_OUTPUT_DOWNLOADED_AND_VALIDATED",
        }
        started: set[int] = set()
        downloaded: set[int] = set()
        for record in self.journal.records:
            task_index = record.get("task_index")
            if (
                not isinstance(task_index, int)
                or isinstance(task_index, bool)
                or not 0 <= task_index < worker_count
            ):
                continue
            if record.get("event") in command_events and task_index > 0:
                started.add(task_index)
            if record.get("event") == "WORKER_OUTPUT_DOWNLOADED_AND_VALIDATED":
                downloaded.add(task_index)
        return started, downloaded

    def _seal_deterministic_failure_terminal(
        self,
        error: DeterministicNumericalWorkerFailure,
    ) -> Mapping[str, object]:
        if not 0 <= error.task_index < int(self.campaign["worker_count"]):
            raise ManagerError("deterministic terminal task index drift")
        observed = hash_file(
            self.config.wave_root / error.failure_relative_path,
            str(error.failure_identity["sha256"]),
        )
        if int(observed["size_bytes"]) != int(error.failure_identity["size_bytes"]):
            raise ManagerError("deterministic failure artifact size drift")
        decision_identity = hash_file(
            self.config.wave_root / DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH
        )
        started_tasks, downloaded_tasks = self._terminal_activity_task_sets()
        if error.task_index > 0 and error.task_index not in started_tasks:
            raise ManagerError("failed fanout task has no command activity receipt")
        cleanup_evidence = self._validated_zero_cleanup_evidence()
        terminal = {
            "schema": "hfw_w560_daytona_deterministic_failure_terminal_v1",
            "status": "FAIL_DETERMINISTIC_NUMERICAL_FAILURE_CLEANED_TERMINAL",
            "campaign_sha256": self.campaign_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "wave_index": self.wave_index,
            "task_index": error.task_index,
            "worker_attempt": error.attempt,
            "process_exit_code": error.exit_code,
            "classification": error.failure["classification"],
            "stage": error.failure["stage"],
            "retry_policy": error.failure["retry_policy"],
            "worker_failure_artifact": {
                "relative_path": error.failure_relative_path,
                "sha256": observed["sha256"],
                "size_bytes": observed["size_bytes"],
            },
            "validated_decision": {
                "relative_path": DETERMINISTIC_FAILURE_DECISION_RELATIVE_PATH,
                "sha256": decision_identity["sha256"],
                "size_bytes": decision_identity["size_bytes"],
            },
            "deterministic_numerical_attempts": 1,
            "failed_task_numerical_attempts": 1,
            "blind_numerical_retries": 0,
            "fanout_tasks_started": len(started_tasks),
            "campaign_sandboxes_remaining": cleanup_evidence[
                "final_remaining_count"
            ],
            "snapshot_present": cleanup_evidence["snapshot_present"],
            "exact_campaign_label_relist_passed": cleanup_evidence[
                "exact_campaign_label_relist_passed"
            ],
            "snapshot_exact_name_not_found_confirmed": cleanup_evidence[
                "snapshot_exact_name_not_found_confirmed"
            ],
            "cleanup_failures": [],
            "cloud_results_downloaded": len(downloaded_tasks),
            "local_collection_or_verify_started": False,
            "checkpoint_or_head_writes": 0,
            "terminal_restart_requires_no_cloud_calls": True,
        }
        path = self.config.wave_root / DETERMINISTIC_FAILURE_TERMINAL_FILENAME
        if path.exists():
            existing, _ = read_json(path)
            if existing != terminal:
                raise ManagerError("deterministic failure terminal changed across restart")
        else:
            write_json_exclusive(path, terminal)
        self.journal.append(
            "DETERMINISTIC_FAILURE_TERMINAL_SEALED",
            terminal=hash_file(path),
            task_index=error.task_index,
            attempt=error.attempt,
            stage=error.failure["stage"],
        )
        return terminal

    def _run_locked(self) -> Mapping[str, object]:
        existing_failure = self._read_deterministic_failure_terminal()
        if existing_failure is not None:
            return existing_failure
        recovered_decision = self._read_deterministic_failure_decision()
        if recovered_decision is not None:
            self.journal.append(
                "DETERMINISTIC_FAILURE_DECISION_RECOVERED_CLEANUP_ONLY",
                task_index=recovered_decision.task_index,
                attempt=recovered_decision.attempt,
                sandbox_create_or_worker_command_permitted=False,
            )
            failures = self._cleanup(
                None, "RECOVER_VALIDATED_DETERMINISTIC_FAILURE",
            )
            if failures:
                raise ManagerError(
                    f"deterministic failure cleanup recovery incomplete: {failures}"
                )
            return self._seal_deterministic_failure_terminal(recovered_decision)
        snapshot = None
        outcome = "FAILURE"
        deterministic_failure = None
        deterministic_terminal = None
        try:
            self.journal.append(
                "MANAGER_RUN_STARTED",
                sdk_version=importlib.metadata.version("daytona"),
                pid=os.getpid(),
                credentials_source="PROCESS_ENVIRONMENT_ONLY",
            )
            existing_terminal = self._existing_download_terminal()
            if existing_terminal is not None:
                outcome = "SUCCESS"
                return existing_terminal
            self._remaining_seconds()
            # A fresh recovery must prove the exact source task0 sandbox, every
            # exact-campaign sandbox, and the deterministic snapshot absent
            # before creating any new cloud resource.  Once a launch manifest
            # exists, this is a partial-fanout restart and its resources must be
            # resumed rather than deleted.  A previously sealed gate is reused
            # locally without any cloud call.
            existing_launch = self._read_existing_launch_manifest()
            if (
                getattr(self, "recovered_task0", None) is not None
                and existing_launch is None
            ):
                self._recovery_precloud_zero_resource_gate()
            snapshot = self._ensure_snapshot()
            existing_launch = self._read_existing_launch_manifest(snapshot=snapshot)
            seed_environment = (
                None if existing_launch is None
                else existing_launch["sealed_worker_environment"]
            )
            seed = self._run_task(snapshot, 0, seed_environment)
            self._reconcile_task_sandboxes_absent(
                0, "SEED_OUTPUT_VALIDATED_BEFORE_FANOUT",
            )
            if existing_launch is None:
                fanout_deadline_safety = self._fanout_deadline_safety_receipt(seed)
                # This pristine clone never runs the model. It reads immutable build
                # evidence and live quota, then is deleted before fanout.
                probe = self._create_probe_sandbox(snapshot)
                probe_identity = _probe_launch_identity(probe)
                try:
                    build_evidence = self._read_build_evidence(probe)
                    quota = self._quota_receipt(
                        probe, int(self.campaign["worker_count"]) - 1,
                    )
                finally:
                    probe_deletion_proof = self._delete_sandbox(
                        probe, "BUILD_EVIDENCE_AND_QUOTA_PROBE_COMPLETE",
                    )
                launch = self._launch_manifest(
                    snapshot, seed, build_evidence, quota,
                    fanout_deadline_safety, probe_identity,
                    probe_deletion_proof,
                )
            else:
                launch = self._read_existing_launch_manifest(
                    snapshot=snapshot, seed_result=seed,
                )
                assert launch is not None
            launch_environment = launch["sealed_worker_environment"]
            self._enforce_fanout_deadline_safety(
                launch["fanout_deadline_safety"],
            )

            results: dict[int, Mapping[str, object]] = {0: seed}
            task_indices = range(1, int(self.campaign["worker_count"]))
            with ThreadPoolExecutor(max_workers=len(list(task_indices))) as executor:
                futures = {
                    executor.submit(
                        self._run_task, snapshot, task_index, launch_environment,
                    ): task_index
                    for task_index in task_indices
                }
                try:
                    for future in as_completed(futures):
                        task_index = futures[future]
                        results[task_index] = future.result()
                except BaseException:
                    self._cancel.set()
                    for future in futures:
                        future.cancel()
                    raise
            if sorted(results) != list(range(int(self.campaign["worker_count"]))):
                raise ManagerError("not all 100 worker downloads completed")
            recovered_count = (
                1 if getattr(self, "recovered_task0", None) is not None else 0
            )
            terminal = {
                "schema": "hfw_w560_daytona_cloud_download_terminal_v1",
                "status": "PASS_ALL_100_PROVISIONAL_CPU_OUTPUTS_DOWNLOADED",
                "campaign_sha256": self.campaign_sha256,
                "wave_index": self.wave_index,
                "launch_manifest": hash_file(self.config.wave_root / "LAUNCH_MANIFEST.json"),
                "worker_count": len(results),
                "recovered_historical_same_campaign_output_count": recovered_count,
                "new_worker_outputs_downloaded_this_recovery": (
                    len(results) - recovered_count
                ),
                "recovered_task_indices": ([0] if recovered_count else []),
                "new_worker_task_indices": list(range(recovered_count, len(results))),
                "workers": [
                    {
                        "task_index": index,
                        "execution_origin": (
                            "RECOVERED_AND_REVALIDATED_NO_RERUN"
                            if recovered_count and index == 0
                            else "EXECUTED_AND_DOWNLOADED_IN_THIS_RECOVERY_RUN"
                        ),
                        "result_sha256": results[index]["result"]["sha256"],
                        "result_size_bytes": results[index]["result"]["size_bytes"],
                        "flat_sha256": results[index]["flat"]["sha256"],
                        "flat_size_bytes": results[index]["flat"]["size_bytes"],
                        "accepted_candidate_bundle_sha256": results[index]["bundle"]["sha256"],
                        "accepted_candidate_bundle_size_bytes": results[index]["bundle"]["size_bytes"],
                        "accepted_candidate_count": results[index]["accepted_candidate_count"],
                        "peak_rss_bytes": results[index]["peak_rss_bytes"],
                        "base_full_search_mean_ce": results[index][
                            "base_full_search_mean_ce"
                        ],
                        "terminal_full_search_mean_ce": results[index][
                            "terminal_full_search_mean_ce"
                        ],
                        "self_calibrated_full_search_improvement": results[index][
                            "self_calibrated_full_search_improvement"
                        ],
                        "acceptance_gate_attempts": results[index][
                            "acceptance_gate_attempts"
                        ],
                        "acceptance_gate_passes": results[index][
                            "acceptance_gate_passes"
                        ],
                        "accepted_updates": results[index]["accepted_updates"],
                    }
                    for index in sorted(results)
                ],
                "all_downloads_fixed_name_sha256_and_size_validated": True,
                "cloud_results_provisional_not_mps_verified_or_promoted": True,
                "raw_ce_cross_lane_comparable": False,
                "worker_selection_metric": "SELF_CALIBRATED_BASE_MINUS_TERMINAL_CE",
                "selection_tie_breakers": [
                    "task_index_ascending", "result_sha256_ascending",
                ],
                "mps_sole_promotion_authority": True,
                "checkpoint_or_head_writes": 0,
            }
            terminal_path = self.config.wave_root / "CLOUD_DOWNLOADS_COMPLETE.json"
            if terminal_path.exists():
                existing, _ = read_json(terminal_path)
                if existing != terminal:
                    raise ManagerError("cloud download terminal changed across restart")
            else:
                write_json_exclusive(terminal_path, terminal)
            self.journal.append(
                "ALL_DOWNLOADS_TERMINAL_SEALED",
                terminal=hash_file(terminal_path),
                worker_count=len(results),
            )
            outcome = "SUCCESS"
            return terminal
        except BaseException as error:
            self._cancel.set()
            self.journal.append(
                "MANAGER_RUN_FAILED",
                error=_safe_error(error, self.credential),
                classification=(
                    "DETERMINISTIC_NUMERICAL_FAILURE"
                    if isinstance(error, DeterministicNumericalWorkerFailure)
                    else "UNCLASSIFIED_OR_INFRASTRUCTURE_FAILURE"
                ),
            )
            if isinstance(error, DeterministicNumericalWorkerFailure):
                deterministic_failure = error
            else:
                raise
        finally:
            failures = self._cleanup(snapshot, outcome)
            if outcome == "SUCCESS" and not failures:
                cleanup_evidence = self._validated_zero_cleanup_evidence()
                cleanup = {
                    "schema": "hfw_w560_daytona_cloud_cleanup_terminal_v1",
                    "status": "PASS_ZERO_CAMPAIGN_SANDBOXES_AND_SNAPSHOT_ABSENT",
                    "campaign_sha256": self.campaign_sha256,
                    "wave_index": self.wave_index,
                    "campaign_sandboxes_remaining": cleanup_evidence[
                        "final_remaining_count"
                    ],
                    "snapshot_present": cleanup_evidence["snapshot_present"],
                    "exact_campaign_label_query": cleanup_evidence[
                        "exact_campaign_label_relist_passed"
                    ],
                    "snapshot_not_found_confirmed": cleanup_evidence[
                        "snapshot_exact_name_not_found_confirmed"
                    ],
                    "cleanup_failures": [],
                    "credentials_must_be_unset_before_local_collection": True,
                    "next_step": "LOCAL_COLLECT_THEN_LOCAL_MPS_DUPLICATE_VERIFY_AND_AT_MOST_ONE_PROMOTION",
                    "cloud_checkpoint_or_head_writes": 0,
                }
                cleanup_path = self.config.wave_root / "CLOUD_CLEANUP_COMPLETE.json"
                if cleanup_path.exists():
                    existing, _ = read_json(cleanup_path)
                    if existing != cleanup:
                        raise ManagerError("cloud cleanup terminal drift")
                else:
                    write_json_exclusive(cleanup_path, cleanup)
            if deterministic_failure is not None and not failures:
                deterministic_terminal = self._seal_deterministic_failure_terminal(
                    deterministic_failure,
                )
            if failures and sys.exc_info()[0] is None:
                raise ManagerError(f"cloud cleanup incomplete: {failures}")
        if deterministic_terminal is not None:
            return deterministic_terminal
        raise ManagerError("manager reached an impossible terminal state")

    def run(self) -> Mapping[str, object]:
        lock_path = self.orchestration_root / "manager-run.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ManagerError("another manager process holds the campaign run lock") from error
            return self._run_locked()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--upload-snapshot-root", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--api-url", default="https://app.daytona.io/api")
    parser.add_argument("--target", default="us")
    parser.add_argument("--worker-timeout-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--campaign-timeout-seconds", type=int, default=8 * 60 * 60)
    parser.add_argument("--api-timeout-seconds", type=int, default=15 * 60)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--create-concurrency", type=int, default=16)
    parser.add_argument("--retry-initial-seconds", type=float, default=2.0)
    parser.add_argument("--retry-maximum-seconds", type=float, default=60.0)
    parser.add_argument("--vcpu-hour-rate-usd", default="0.0504")
    parser.add_argument("--memory-gib-hour-rate-usd", default="0.0162")
    parser.add_argument("--disk-gib-hour-rate-usd", default="0.000108")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    package_root = args.package_root.resolve()
    wave_root = args.wave_root.resolve()
    if package_root not in wave_root.parents:
        raise SystemExit("wave root must be inside package root")
    config = ManagerConfig(
        package_root=package_root,
        wave_root=wave_root,
        upload_snapshot_root=args.upload_snapshot_root,
        base_image=args.base_image,
        api_url=args.api_url,
        target=args.target,
        worker_timeout_seconds=args.worker_timeout_seconds,
        campaign_timeout_seconds=args.campaign_timeout_seconds,
        api_timeout_seconds=args.api_timeout_seconds,
        poll_seconds=args.poll_seconds,
        maximum_attempts=args.maximum_attempts,
        create_concurrency=args.create_concurrency,
        retry_initial_seconds=args.retry_initial_seconds,
        retry_maximum_seconds=args.retry_maximum_seconds,
        vcpu_hour_rate_usd=args.vcpu_hour_rate_usd,
        memory_gib_hour_rate_usd=args.memory_gib_hour_rate_usd,
        disk_gib_hour_rate_usd=args.disk_gib_hour_rate_usd,
    )
    result = DaytonaManager(config).run()
    output = {
        "status": result["status"],
        "campaign_sha256": result["campaign_sha256"],
        "cloud_resources_cleaned": True,
        "checkpoint_or_head_writes": 0,
    }
    if "worker_count" in result:
        output["worker_count"] = result["worker_count"]
    if result["status"] == "FAIL_DETERMINISTIC_NUMERICAL_FAILURE_CLEANED_TERMINAL":
        output.update({
            "task_index": result["task_index"],
            "stage": result["stage"],
            "blind_numerical_retries": 0,
        })
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
