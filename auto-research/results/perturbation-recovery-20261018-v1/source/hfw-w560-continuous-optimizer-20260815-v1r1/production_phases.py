from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import torch

from .artifacts import (
    read_json_same_fd, require_strong_path_separation,
    validate_fresh_no_writer_receipt, write_content_addressed_json,
)
from .branch_search import run_independent_branch, validate_branch_artifact_chain
from .data_commitment import (
    load_committed_batches, resolve_public_calibration_layout,
    resolve_validation_corpus,
)
from .manifest import file_sha256, load_protocol, verify_source_inventory
from .parent_gate import validate_immutable_parent
from .power_sampler import power_sampler_from_protocol
from .production_bridge import W560ForwardOnlyObjective, load_completed_w560_parent
from .production_runner import execute_report_only_fixed_eval
from .resource_gate import validate_no_autograd_resource_gate
from .sealed_evaluation_bridge import (
    CanonicalProposalRegistry, create_production_search_capability,
    load_canonical_sealed_evaluation_v2,
)


def _validate_final_cuda_isolation(
    protocol: Mapping[str, object],
) -> int:
    physical_index = int(
        protocol["operations"]["final_evaluation_physical_cuda_index"]
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_index):
        raise RuntimeError("final evaluator CUDA physical isolation drift")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("final evaluator requires exactly one isolated CUDA device")
    return physical_index


def authority_prepare_core(
    protocol: Mapping[str, object], package_root: Path, source_root: Path,
    runtime_root: Path, *,
    authority_root: Path, public_root: Path,
) -> Mapping[str, object]:
    require_strong_path_separation({"authority": authority_root, "public": public_root})
    if authority_root.exists() or public_root.exists():
        raise RuntimeError("authority and public roots must be fresh create-only paths")
    calibration = resolve_public_calibration_layout(protocol, runtime_root)
    canonical, canonical_v2, authority, capability, calibration_sha = (
        create_production_search_capability(
            protocol, package_root,
            authority_root=authority_root,
            public_root=public_root,
            calibration_layout=calibration,
        )
    )
    del canonical, canonical_v2
    public_manifest = {
        "schema": "hfw_w560_search_capability_manifest_v1",
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "protocol_sha256": file_sha256(package_root / "config/production_protocol.json"),
        "capability": capability.public_payload(),
        "calibration_manifest_sha256": calibration_sha,
        "calibration_layout": calibration,
        "authority_fields_present": False,
        "authority_process_pid": os.getpid(),
        "search_process_must_differ_from_authority_pid": True,
    }
    public_artifact = write_content_addressed_json(
        public_root / "production_search_inputs", ".search.json", public_manifest
    )
    private_receipt = {
        "schema": "hfw_w560_authority_prepare_private_receipt_v1",
        "status": "PASS_AUTHORITY_PREPARED_SEARCH_NOT_STARTED",
        "generation_record_sha256": authority.generation_record_sha256,
        "opaque_eval_commitment_sha256": capability.opaque_eval_commitment_sha256,
        "public_search_manifest_sha256": public_artifact["sha256"],
        "authority_process_pid": os.getpid(),
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "fixed_eval_revealed": False,
    }
    private_artifact = write_content_addressed_json(
        authority_root / "authority_prepare_receipts", ".authority.json",
        private_receipt, mode=0o400,
    )
    return {
        "status": "PASS_AUTHORITY_PREPARED_SEARCH_NOT_STARTED",
        "public_search_manifest": public_artifact,
        "private_authority_receipt": private_artifact,
        "search_payload": capability.public_payload(),
    }


def _load_search_manifest(
    path: Path, expected_sha256: str,
    protocol: Mapping[str, object] | None = None, package_root: Path | None = None,
) -> Mapping[str, object]:
    value, identity = read_json_same_fd(path)
    if identity["sha256"] != expected_sha256:
        raise RuntimeError("public search manifest SHA drift")
    if value.get("authority_fields_present") is not False:
        raise RuntimeError("search manifest contains authority fields")
    if protocol is not None:
        if value.get("parent_checkpoint_sha256") != protocol["parent"]["checkpoint_sha256"]:
            raise RuntimeError("search manifest parent checkpoint binding drift")
        if package_root is None or value.get("protocol_sha256") != file_sha256(
            package_root / "config/production_protocol.json"
        ):
            raise RuntimeError("search manifest protocol binding drift")
    encoded = json.dumps(value, sort_keys=True)
    for forbidden in ("authority_root", "generation_record_sha256", "private_entropy", "evaluation_offsets", "salt_hex"):
        if forbidden in encoded:
            raise RuntimeError(f"search manifest leaked private authority data: {forbidden}")
    if int(value["authority_process_pid"]) == os.getpid():
        raise RuntimeError("search worker cannot be the authority process")
    return value


def run_group_worker_core(
    protocol: Mapping[str, object], package_root: Path, source_root: Path,
    runtime_root: Path, *,
    search_manifest_path: Path, search_manifest_sha256: str,
    output_root: Path, group: str,
    resource_receipt_path: Path, resource_receipt_sha256: str,
) -> Mapping[str, object]:
    manifest = _load_search_manifest(
        search_manifest_path, search_manifest_sha256, protocol, package_root
    )
    resource, resource_identity = read_json_same_fd(resource_receipt_path)
    if resource_identity["sha256"] != resource_receipt_sha256:
        raise RuntimeError("resource receipt SHA drift in worker")
    if resource.get("status") != "PASS_NO_AUTOGRAD_RESOURCE_GATE":
        raise RuntimeError("worker requires a passed measured resource receipt")
    expected_physical = int(protocol["operations"]["worker_group_to_physical_cuda_index"][group])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(expected_physical):
        raise RuntimeError("worker CUDA isolation drift")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("isolated worker requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    pre_parent = validate_immutable_parent(protocol)
    verify_source_inventory(protocol, source_root)
    loaded = load_completed_w560_parent(protocol, source_root, device)
    batches = [
        torch.from_numpy(batch.copy())
        for batch in load_committed_batches(
            protocol, runtime_root, manifest["calibration_layout"], "calibration"
        )
    ]
    objective = W560ForwardOnlyObjective(loaded, batches, device)
    canonical, canonical_v2 = load_canonical_sealed_evaluation_v2(protocol, package_root)
    capability_payload = manifest["capability"]
    capability = canonical.SearchCapability(
        Path(capability_payload["public_root"]),
        capability_payload["opaque_eval_commitment_sha256"],
    )
    aliases = protocol["canonical_weight_address_binding"]["internal_to_canonical_group"]
    allowed = tuple(aliases[value] for value in protocol["search"]["independent_branches"])
    registry = CanonicalProposalRegistry(
        canonical_v2, capability, protocol["parent"]["checkpoint_sha256"],
        manifest["calibration_manifest_sha256"], group, aliases, allowed,
    )
    with power_sampler_from_protocol(protocol, expected_physical) as power_sampler:
        result = run_independent_branch(
            objective, protocol, group, output_root, registry, materialize=True
        )
    power_receipt = power_sampler.receipt()
    if result.receipt.get("estimated_total_macs") is None:
        raise RuntimeError("production branch did not account estimated MACs")
    branch_validation = validate_branch_artifact_chain(
        protocol, output_root / group, group, validate_child_payload=True
    )
    post_parent = validate_immutable_parent(protocol)
    verify_source_inventory(protocol, source_root)
    if pre_parent != post_parent:
        raise RuntimeError("immutable parent evidence changed across worker")
    worker = {
        "schema": "hfw_w560_isolated_group_worker_receipt_v1",
        "status": "PASS_ISOLATED_GROUP_WORKER_COMPLETE",
        "group": group,
        "canonical_group": aliases[group],
        "physical_cuda_index": expected_physical,
        "visible_cuda_devices": 1,
        "worker_pid": os.getpid(),
        "authority_pid": manifest["authority_process_pid"],
        "branch_final": registry.branch_final(),
        "branch_artifact_validation": {
            key: value for key, value in branch_validation.items() if key != "receipt"
        },
        "branch_receipt_identity_sha256": result.receipt["receipt_identity_sha256"],
        "runtime_guard": result.receipt["runtime_guard"],
        "parent_pre": pre_parent,
        "parent_post": post_parent,
        "source_revalidated_post": True,
        "resource_receipt_sha256": resource_receipt_sha256,
        "gpu_power_draw_energy": power_receipt,
        "forward_calls": result.receipt["forward_calls"],
        "unique_calibration_token_positions": result.receipt["unique_calibration_token_positions"],
        "total_evaluated_target_tokens": result.receipt["total_evaluated_target_tokens"],
        "estimated_total_macs": result.receipt["estimated_total_macs"],
        "wall_seconds": result.receipt["wall_seconds"],
        "fixed_eval_opened": False,
    }
    artifact = write_content_addressed_json(
        output_root / group / "worker_receipts", ".worker.json", worker
    )
    return {**worker, "worker_receipt_artifact": artifact}


def search_coordinator_core(
    protocol: Mapping[str, object], package_root: Path, source_root: Path,
    runtime_root: Path, *,
    search_manifest_path: Path, search_manifest_sha256: str,
    output_root: Path, no_writer_receipt_path: Path,
) -> Mapping[str, object]:
    manifest = _load_search_manifest(
        search_manifest_path, search_manifest_sha256, protocol, package_root
    )
    no_writer = validate_fresh_no_writer_receipt(protocol, no_writer_receipt_path)
    if output_root.exists():
        raise RuntimeError("search output root must be fresh")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    resource = validate_no_autograd_resource_gate(protocol, output_root, no_writer)
    output_root.mkdir(exist_ok=False)
    resource_artifact = write_content_addressed_json(
        output_root / "resource_receipts", ".resource.json", resource
    )
    processes = []
    for group in protocol["search"]["independent_branches"]:
        physical = int(protocol["operations"]["worker_group_to_physical_cuda_index"][group])
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(physical)
        command = [
            sys.executable, "-m", f"{__package__}.production_entrypoint", "worker",
            "--protocol", str(package_root / "config/production_protocol.json"),
            "--source-root", str(source_root),
            "--runtime-root", str(runtime_root),
            "--search-manifest", str(search_manifest_path),
            "--search-manifest-sha256", search_manifest_sha256,
            "--output-root", str(output_root),
            "--group", group,
            "--resource-receipt", resource_artifact["path"],
            "--resource-receipt-sha256", resource_artifact["sha256"],
        ]
        processes.append((group, command, subprocess.Popen(
            command, cwd=package_root.parents[1], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )))
    failures = []
    worker_outputs = {}
    for group, command, process in processes:
        stdout, stderr = process.communicate()
        worker_outputs[group] = {
            "exit_code": process.returncode,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            "command": command,
        }
        if process.returncode:
            failures.append(group)
            write_content_addressed_json(output_root / "failure_receipts", ".failure.json", {
                "schema": "hfw_w560_worker_failure_receipt_v1",
                "status": "BLOCK",
                "group": group,
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            })
    if failures:
        raise RuntimeError(f"isolated GPU workers failed: {failures}")
    canonical, _ = load_canonical_sealed_evaluation_v2(protocol, package_root)
    payload = manifest["capability"]
    capability = canonical.SearchCapability(
        Path(payload["public_root"]), payload["opaque_eval_commitment_sha256"]
    )
    branch_receipts = {}
    for group in protocol["search"]["independent_branches"]:
        candidates = list((output_root / group / "worker_receipts").glob("*.worker.json"))
        if len(candidates) != 1 or candidates[0].is_symlink():
            raise RuntimeError(f"worker receipt inventory drift: {group}")
        worker, identity = read_json_same_fd(candidates[0])
        if worker.get("status") != "PASS_ISOLATED_GROUP_WORKER_COMPLETE":
            raise RuntimeError(f"worker did not pass: {group}")
        validate_branch_artifact_chain(protocol, output_root / group, group)
        branch_receipts[group] = {"receipt": worker, "file_sha256": identity["sha256"]}
    chain = canonical.validate_proposal_chain(capability)
    expected = len(branch_receipts) * int(protocol["search"]["pairs_per_branch"])
    if chain["event_count"] != expected:
        raise RuntimeError("canonical four-worker proposal chain count drift")
    search_receipt = {
        "schema": "hfw_w560_four_gpu_search_coordinator_receipt_v1",
        "status": "PASS_SEARCH_TERMINAL_READY_EVAL_STILL_SEALED",
        "search_coordinator_pid": os.getpid(),
        "authority_pid": manifest["authority_process_pid"],
        "workers": worker_outputs,
        "branches": branch_receipts,
        "resource_receipt": resource_artifact,
        "no_writer_receipt_sha256": no_writer["receipt_file_sha256"],
        "canonical_chain": {
            key: chain[key] for key in ("event_count", "accepted", "rejected", "head_event_sha256")
        },
        "fixed_eval_opened": False,
        "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        "protocol_sha256": file_sha256(package_root / "config/production_protocol.json"),
        "public_search_manifest_sha256": search_manifest_sha256,
    }
    artifact = write_content_addressed_json(
        output_root / "search_coordinator_receipts", ".search_complete.json", search_receipt
    )
    return {**search_receipt, "search_receipt_artifact": artifact}


def authority_finalize_core(
    protocol: Mapping[str, object], package_root: Path, source_root: Path,
    runtime_root: Path, *,
    authority_root: Path, public_root: Path, output_root: Path,
    final_registry_root: Path,
    private_authority_receipt_path: Path, private_authority_receipt_sha256: str,
    search_manifest_path: Path, search_manifest_sha256: str,
    search_receipt_path: Path, search_receipt_sha256: str,
    final_no_writer_receipt_path: Path,
) -> Mapping[str, object]:
    require_strong_path_separation({
        "authority": authority_root, "public": public_root,
        "search_output": output_root, "final_registry": final_registry_root,
    })
    if final_registry_root.exists():
        raise RuntimeError("final result registry root must be fresh")
    final_registry_root.mkdir(parents=True, exist_ok=False)
    private, private_identity = read_json_same_fd(private_authority_receipt_path)
    if private_identity["sha256"] != private_authority_receipt_sha256:
        raise RuntimeError("private authority receipt SHA drift")
    if private.get("status") != "PASS_AUTHORITY_PREPARED_SEARCH_NOT_STARTED":
        raise RuntimeError("authority prepare receipt did not pass")
    manifest = _load_search_manifest(
        search_manifest_path, search_manifest_sha256, protocol, package_root
    )
    search, search_identity = read_json_same_fd(search_receipt_path)
    if search_identity["sha256"] != search_receipt_sha256:
        raise RuntimeError("search coordinator receipt SHA drift")
    if search.get("status") != "PASS_SEARCH_TERMINAL_READY_EVAL_STILL_SEALED":
        raise RuntimeError("four-worker search is not terminal-ready")
    if search.get("parent_checkpoint_sha256") != protocol["parent"]["checkpoint_sha256"]:
        raise RuntimeError("search receipt parent binding drift")
    if search.get("protocol_sha256") != file_sha256(package_root / "config/production_protocol.json"):
        raise RuntimeError("search receipt protocol binding drift")
    if search.get("public_search_manifest_sha256") != search_manifest_sha256:
        raise RuntimeError("search receipt public-manifest binding drift")
    if int(search["search_coordinator_pid"]) == os.getpid():
        raise RuntimeError("final authority cannot be the search coordinator process")
    if int(search["authority_pid"]) != int(manifest["authority_process_pid"]):
        raise RuntimeError("authority/search process identity binding drift")
    final_no_writer = validate_fresh_no_writer_receipt(protocol, final_no_writer_receipt_path)
    final_physical_index = _validate_final_cuda_isolation(protocol)
    final_resource = validate_no_autograd_resource_gate(
        protocol, final_registry_root / "pending", final_no_writer,
        physical_cuda_indices=(final_physical_index,),
    )
    final_resource_artifact = write_content_addressed_json(
        final_registry_root / "resource_receipts", ".resource.json", final_resource
    )
    validation_path, _ = resolve_validation_corpus(protocol, runtime_root)
    canonical, canonical_v2 = load_canonical_sealed_evaluation_v2(protocol, package_root)
    payload = manifest["capability"]
    capability = canonical.SearchCapability(
        Path(payload["public_root"]), payload["opaque_eval_commitment_sha256"]
    )
    authority = canonical_v2.AuthorityGeneratedCommitment(
        capability, authority_root, private["generation_record_sha256"]
    )
    aliases = protocol["canonical_weight_address_binding"]["internal_to_canonical_group"]
    branch_finals = []
    for group in protocol["search"]["independent_branches"]:
        branch_validation = validate_branch_artifact_chain(protocol, output_root / group, group)
        bound = search["branches"][group]
        candidates = list((output_root / group / "worker_receipts").glob("*.worker.json"))
        if len(candidates) != 1:
            raise RuntimeError("final authority worker receipt inventory drift")
        worker, identity = read_json_same_fd(candidates[0])
        if identity["sha256"] != bound["file_sha256"] or worker != bound["receipt"]:
            raise RuntimeError("final authority worker receipt binding drift")
        current_validation = {
            key: value for key, value in branch_validation.items() if key != "receipt"
        }
        if worker["branch_artifact_validation"] != current_validation:
            raise RuntimeError("terminal full branch artifact identity drift")
        branch_finals.append(worker["branch_final"])
    pre_parent = validate_immutable_parent(protocol)
    verify_source_inventory(protocol, source_root)
    authorization_v2 = canonical_v2.seal_reveal_authorize_four_branch_final_eval(
        authority,
        parent_checkpoint_sha256=protocol["parent"]["checkpoint_sha256"],
        calibration_manifest_sha256=manifest["calibration_manifest_sha256"],
        required_groups=[aliases[group] for group in protocol["search"]["independent_branches"]],
        branch_finals=branch_finals,
        terminal_status="ALL_FOUR_ISOLATED_GPU_BRANCHES_COMPLETE",
        validation_path=validation_path,
    )
    reveal = canonical.read_public_reveal(capability)
    fixed_authorization = {
        "status": "PASS_FIXED_EVAL_MAY_NOW_OPEN",
        "all_four_branches_complete": True,
        "evaluation_offsets": reveal["evaluation_offsets"],
        "sample_length_tokens": reveal["sample_length_tokens"],
        "canonical_v2_authorization": authorization_v2,
        "bound_branch_finals": {
            final["group"]: final for final in branch_finals
        },
    }
    device = torch.device("cuda:0")
    loaded = load_completed_w560_parent(protocol, source_root, device)
    with power_sampler_from_protocol(protocol, final_physical_index) as power_sampler:
        fixed_eval = execute_report_only_fixed_eval(
            protocol, loaded,
            authorization=fixed_authorization,
            validation_path=validation_path,
            search_root=output_root,
            device=device,
        )
    power_receipt = power_sampler.receipt()
    fixed_eval = {
        **fixed_eval,
        "gpu_power_draw_energy": power_receipt,
    }
    for group in protocol["search"]["independent_branches"]:
        validate_branch_artifact_chain(protocol, output_root / group, group)
    post_parent = validate_immutable_parent(protocol)
    verify_source_inventory(protocol, source_root)
    if pre_parent != post_parent:
        raise RuntimeError("parent changed across final fixed evaluation")
    audit = canonical.audit_sealed_evaluation(capability)
    if audit.get("status") != "PASS" or not audit.get("terminal_sealed") or not audit.get("revealed"):
        raise RuntimeError("canonical sealed-evaluation post-audit failed")
    result = {
        "schema": "hfw_w560_post_training_complete_result_v2",
        "status": "PASS_COMPLETE",
        "parent_pre": pre_parent,
        "parent_post": post_parent,
        "search_receipt_sha256": search_receipt_sha256,
        "final_no_writer_receipt_sha256": final_no_writer["receipt_file_sha256"],
        "final_resource_receipt": final_resource_artifact,
        "branch_finals": branch_finals,
        "canonical_final_authorization": authorization_v2,
        "fixed_eval": fixed_eval,
        "canonical_post_audit": audit,
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "training_resume_authorized": False,
        "branch_selected_or_combined": False,
    }
    artifact = write_content_addressed_json(
        final_registry_root, ".final_result.json", result
    )
    return {**result, "final_result_artifact": artifact}
