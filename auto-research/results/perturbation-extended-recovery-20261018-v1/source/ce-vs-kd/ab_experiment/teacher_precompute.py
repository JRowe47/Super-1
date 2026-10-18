from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import torch

from .common import (
    canonical_json_bytes, file_identity, read_json_same_fd, sha256_bytes,
    write_exclusive, write_exclusive_json,
)
from .runtime import (
    BF16_LOGIT_SHARD_BYTES, CONTEXT_TOKENS, SEQUENCE_TOKENS, TEACHER_VOCAB_SIZE,
    assert_no_conflicting_mps_solver, load_bank, load_config,
    validate_input_closure,
)


def _require_exact_path(actual: Path, expected: str, label: str) -> None:
    if Path(actual).resolve() != Path(expected).resolve():
        raise RuntimeError(f"{label} path is not the unique preregistered destination")


def _claim(config: Mapping[str, object], config_path: Path, role: str) -> Mapping[str, object]:
    contract = config["output_contract"][role]
    return write_exclusive_json(Path(contract["claim_path"]), {
        "schema": "hfw_w560_ce_vs_kd_unique_teacher_phase_claim_v1",
        "status": "CLAIMED_NO_RETRY_OR_ALTERNATE_TEACHER_OUTPUT_WITHIN_THIS_A_B_PACKAGE",
        "role": role,
        "config": file_identity(config_path),
        "bound_output": contract.get("result_root", contract.get("result_path")),
        "failure_requires_fresh_whole_a_b_package": True,
    })


def validate_environment(config: Mapping[str, object]) -> Mapping[str, object]:
    import huggingface_hub
    import numpy
    import safetensors
    import tokenizers
    import transformers

    observed = {
        "python": ".".join(str(value) for value in os.sys.version_info[:3]),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
        "tokenizers": tokenizers.__version__,
    }
    expected = config["environment"]["versions"]
    for key, value in expected.items():
        if observed.get(key) != value:
            raise RuntimeError(f"runtime version drift: {key}: {observed.get(key)} != {value}")
    offline = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    if any(value != "1" for value in offline.values()):
        raise RuntimeError(f"teacher runtime is not fully offline: {offline}")
    observed["offline_environment"] = offline
    return observed


def validate_tokenizer(config: Mapping[str, object]) -> Mapping[str, object]:
    from transformers import AutoTokenizer

    teacher = config["teacher"]
    snapshot = Path(teacher["snapshot_root"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False,
    )
    mapping = sorted(
        (str(token), int(identifier))
        for token, identifier in tokenizer.get_vocab().items()
    )
    ids = frozenset(identifier for _, identifier in mapping)
    added = sorted(
        (str(token), int(identifier))
        for token, identifier in tokenizer.get_added_vocab().items()
    )
    observed = {
        "tokenizer_class": type(tokenizer).__name__,
        "declared_base_vocab_size": int(tokenizer.vocab_size),
        "mapping_entry_count": len(mapping),
        "unique_mapped_id_count": len(ids),
        "minimum_mapped_id": min(ids),
        "maximum_mapped_id": max(ids),
        "mapping_sha256": sha256_bytes(canonical_json_bytes(mapping)),
        "added_mapping_entry_count": len(added),
        "added_mapping_sha256": sha256_bytes(canonical_json_bytes(added)),
        "mapped_ids_contiguous_observation": (
            len(ids) == max(ids) - min(ids) + 1
        ),
        "contiguity_required_for_validity": False,
    }
    if observed != teacher["runtime_tokenizer_mapping"]:
        raise RuntimeError("pinned runtime tokenizer mapping identity drift")
    bank_validation = {}
    for role, record in config["banks"].items():
        bank, _ = load_bank(record)
        bank_values = bank.numpy()
        bank_ids = frozenset(int(value) for value in np.unique(bank_values))
        missing = bank_ids - ids
        if missing:
            raise RuntimeError(
                f"{role} bank contains IDs absent from pinned runtime tokenizer: "
                f"{sorted(missing)[:8]}"
            )
        bank_maximum = int(bank_values.max())
        if bank_maximum >= int(teacher["vocab_size"]):
            raise RuntimeError(f"{role} bank token exceeds teacher model vocabulary")
        bank_validation[role] = {
            "unique_token_id_count": len(bank_ids),
            "maximum_token_id": bank_maximum,
            "all_ids_present_in_runtime_tokenizer_mapping": True,
            "all_ids_below_teacher_model_vocab_size": True,
        }
    if bank_validation != teacher["bank_token_validation"]:
        raise RuntimeError("pinned runtime tokenizer/bank validation drift")
    return {**observed, "bank_token_validation": bank_validation}


def load_teacher(config: Mapping[str, object], device: torch.device):
    from transformers import AutoModelForCausalLM

    teacher = config["teacher"]
    snapshot = Path(teacher["snapshot_root"])
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("teacher no-autograd gate failed")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != int(teacher["parameter_count"]):
        raise RuntimeError("teacher parameter count drift")
    if int(model.config.vocab_size) != TEACHER_VOCAB_SIZE:
        raise RuntimeError("teacher model vocabulary size drift")
    return model


def forward_teacher(model, raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = raw[:, :-1].to(device=device, dtype=torch.long)
    if inputs.dtype != torch.long:
        raise RuntimeError("teacher embedding inputs are not int64")
    with torch.inference_mode(), torch.autocast(
        device_type="mps", dtype=torch.bfloat16, enabled=True,
    ):
        logits = model(input_ids=inputs, use_cache=False).logits
    if logits.dtype != torch.bfloat16:
        raise RuntimeError("teacher forward did not produce bfloat16 logits")
    expected = (raw.shape[0], CONTEXT_TOKENS, TEACHER_VOCAB_SIZE)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"teacher output shape drift: {tuple(logits.shape)} != {expected}")
    if not torch.isfinite(logits.float()).all():
        raise RuntimeError("teacher output contains nonfinite values")
    return logits


def logits_bytes(logits: torch.Tensor) -> bytes:
    if logits.dtype != torch.bfloat16 or logits.shape[0] != 1:
        raise RuntimeError("logit serialization requires one BF16 sequence")
    raw = logits.detach().cpu().contiguous().view(torch.uint16).numpy()
    body = raw.astype("<u2", copy=False).tobytes(order="C")
    if len(body) != BF16_LOGIT_SHARD_BYTES:
        raise RuntimeError("serialized teacher shard size drift")
    return body


def smoke(config_path: Path, output: Path) -> Mapping[str, object]:
    config = load_config(config_path)
    _require_exact_path(
        output, config["output_contract"]["teacher_smoke"]["result_path"],
        "teacher smoke",
    )
    assert_no_conflicting_mps_solver(config)
    inputs = validate_input_closure(config)
    environment = validate_environment(config)
    tokenizer = validate_tokenizer(config)
    search, bank_identity = load_bank(config["banks"]["SEARCH"])
    first_batch = search.reshape(-1, 2, 513)[0]
    _claim(config, config_path, "teacher_smoke")
    device = torch.device("mps")
    started = time.monotonic()
    model = load_teacher(config, device)
    logits = forward_teacher(model, first_batch, device)
    digest = hashlib.sha256()
    for sequence in range(logits.shape[0]):
        digest.update(logits_bytes(logits[sequence:sequence + 1]))
    receipt = {
        "schema": "hfw_w560_teacher_one_search_batch_mps_smoke_v1",
        "status": "PASS_EXACT_LOCAL_TEACHER_FULL_VOCAB_MPS_BF16_FORWARD",
        "config_sha256": file_identity(config_path)["sha256"],
        "validated_input_count": len(inputs),
        "environment": environment,
        "tokenizer": tokenizer,
        "search_bank": bank_identity,
        "batch_index": 0,
        "input_shape": list(first_batch.shape),
        "output_shape": list(logits.shape),
        "output_dtype": str(logits.dtype),
        "full_batch_logits_sha256": digest.hexdigest(),
        "teacher_forward_calls": 1,
        "teacher_logits_materialized_for_search": False,
        "student_forward_calls": 0,
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "wall_seconds": time.monotonic() - started,
        "active_conflicting_mps_solver_detected": False,
    }
    write_exclusive_json(output, receipt)
    return receipt


def precompute(config_path: Path, smoke_receipt_path: Path, output_root: Path) -> Mapping[str, object]:
    config = load_config(config_path)
    _require_exact_path(
        smoke_receipt_path, config["output_contract"]["teacher_smoke"]["result_path"],
        "teacher smoke receipt",
    )
    _require_exact_path(
        output_root, config["output_contract"]["teacher_precompute"]["result_root"],
        "teacher precompute",
    )
    assert_no_conflicting_mps_solver(config)
    validate_input_closure(config)
    environment = validate_environment(config)
    tokenizer = validate_tokenizer(config)
    smoke_receipt = read_json_same_fd(smoke_receipt_path)
    if smoke_receipt.get("status") != "PASS_EXACT_LOCAL_TEACHER_FULL_VOCAB_MPS_BF16_FORWARD":
        raise RuntimeError("teacher precompute requires successful bounded smoke")
    if smoke_receipt.get("config_sha256") != file_identity(config_path)["sha256"]:
        raise RuntimeError("teacher smoke/preregistered config identity drift")
    if output_root.exists():
        raise FileExistsError("teacher output root must be fresh")
    _claim(config, config_path, "teacher_precompute")
    output_root.mkdir(parents=True)
    device = torch.device("mps")
    started = time.monotonic()
    model = load_teacher(config, device)
    bank_receipts = {}
    total_forwards = 0
    total_bytes = 0
    for role in ("SEARCH", "VERIFY", "FINAL"):
        bank, bank_identity = load_bank(config["banks"][role])
        sequences = bank.reshape(-1, SEQUENCE_TOKENS)
        role_root = output_root / role.lower()
        role_root.mkdir()
        shards = []
        for index, raw in enumerate(sequences):
            logits = forward_teacher(model, raw.unsqueeze(0), device)
            body = logits_bytes(logits)
            digest = sha256_bytes(body)
            artifact = write_exclusive(
                role_root / f"{index:05d}-{digest}.bf16le", body,
            )
            shards.append({
                "index": index, **artifact,
                "shape": [CONTEXT_TOKENS, TEACHER_VOCAB_SIZE],
                "dtype": "bfloat16_little_endian_uint16_payload",
                "input_sequence_sha256": sha256_bytes(
                    raw.numpy().astype("<u4", copy=False).tobytes(order="C")
                ),
            })
            total_forwards += 1
            total_bytes += len(body)
            del logits, body
            if index % 4 == 3:
                gc.collect()
                torch.mps.empty_cache()
        bank_receipts[role] = {
            "bank": bank_identity,
            "sequence_count": int(sequences.shape[0]),
            "shards": shards,
            "total_bytes": sum(int(item["size_bytes"]) for item in shards),
            "target_aligned_logits_only": True,
            "unused_513th_logit_not_stored": True,
        }
    expected_total = int(config["teacher_outputs_contract"]["expected_total_bytes"])
    if total_bytes != expected_total:
        raise RuntimeError(f"teacher output total bytes drift: {total_bytes} != {expected_total}")
    precompute_wall_seconds = time.monotonic() - started
    manifest = {
        "schema": "hfw_w560_ce_vs_kd_full_teacher_logits_manifest_v1",
        "status": "PASS_ALL_FULL_VOCAB_TEACHER_LOGITS_FROZEN",
        "config_sha256": file_identity(config_path)["sha256"],
        "smoke_receipt": file_identity(smoke_receipt_path),
        "environment": environment,
        "tokenizer": tokenizer,
        "teacher": config["teacher"],
        "inference": {
            "device": "mps",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "use_cache": False,
            "local_files_only": True,
            "trust_remote_code": False,
            "input_slice": "raw[:, :-1]",
            "output_shape_per_sequence": [CONTEXT_TOKENS, TEACHER_VOCAB_SIZE],
            "full_vocabulary": True,
            "top_k_or_projection": False,
        },
        "banks": bank_receipts,
        "teacher_forward_calls_preprocessing": total_forwards,
        "bounded_smoke_teacher_forward_calls": int(smoke_receipt["teacher_forward_calls"]),
        "aggregate_one_time_teacher_forward_calls_including_smoke": (
            total_forwards + int(smoke_receipt["teacher_forward_calls"])
        ),
        "teacher_forward_calls_after_freeze": 0,
        "total_payload_bytes": total_bytes,
        "bounded_smoke_wall_seconds": float(smoke_receipt["wall_seconds"]),
        "preprocessing_wall_seconds": precompute_wall_seconds,
        "aggregate_one_time_teacher_wall_seconds_including_smoke": (
            precompute_wall_seconds + float(smoke_receipt["wall_seconds"])
        ),
        "excluded_from_each_arm_180_student_forward_budget": True,
        "optimizer_steps": 0,
        "autograd_calls": 0,
    }
    artifact = write_exclusive_json(output_root / "TEACHER_OUTPUTS_FROZEN.json", manifest)
    search_capability = {
        "schema": "hfw_w560_ce_vs_kd_teacher_logits_capability_v1",
        "status": "PASS_SEARCH_ONLY_TEACHER_LOGITS_CAPABILITY",
        "master_manifest_sha256": artifact["sha256"],
        "roles_available": ["SEARCH"],
        "banks": {"SEARCH": bank_receipts["SEARCH"]},
        "verify_or_final_role_present": False,
        "teacher_forward_calls_after_freeze": 0,
    }
    search_artifact = write_exclusive_json(
        output_root / "SEARCH_TEACHER_CAPABILITY.json", search_capability,
    )
    verify_capability = {
        "schema": "hfw_w560_ce_vs_kd_teacher_logits_capability_v1",
        "status": "PASS_VERIFY_ONLY_TEACHER_LOGITS_CAPABILITY",
        "master_manifest_sha256": artifact["sha256"],
        "roles_available": ["VERIFY"],
        "banks": {"VERIFY": bank_receipts["VERIFY"]},
        "search_or_final_role_present": False,
        "teacher_forward_calls_after_freeze": 0,
    }
    verify_artifact = write_exclusive_json(
        output_root / "VERIFY_TEACHER_CAPABILITY.json", verify_capability,
    )
    final_capability = {
        "schema": "hfw_w560_ce_vs_kd_teacher_logits_capability_v1",
        "status": "PASS_FINAL_ONLY_TEACHER_LOGITS_CAPABILITY_AFTER_SELECTION_SEAL",
        "master_manifest_sha256": artifact["sha256"],
        "roles_available": ["FINAL"],
        "banks": {"FINAL": bank_receipts["FINAL"]},
        "search_or_verify_role_present": False,
        "teacher_forward_calls_after_freeze": 0,
    }
    final_artifact = write_exclusive_json(
        output_root / "FINAL_TEACHER_CAPABILITY.json", final_capability,
    )
    seal = {
        "schema": "hfw_w560_final_tokens_and_teacher_outputs_presearch_seal_v1",
        "status": "PASS_TEACHER_ONLY_FINAL_LABELING_COMPLETE_AND_FINAL_SEALED",
        "final_bank": config["banks"]["FINAL"],
        "master_teacher_manifest": artifact,
        "search_capability": search_artifact,
        "verify_capability": verify_artifact,
        "final_only_capability": final_artifact,
        "final_access_before_seal": "teacher_only_precompute",
        "student_optimizer_or_selection_final_access_before_seal": False,
        "teacher_forward_calls_after_seal": 0,
        "final_capability_must_not_be_provisioned_until_both_selection_receipts_exist": True,
    }
    seal_artifact = write_exclusive_json(
        output_root / "FINAL_AND_TEACHER_OUTPUTS_SEALED.json", seal,
    )
    return {
        **manifest,
        "manifest_artifact": artifact,
        "search_capability_artifact": search_artifact,
        "verify_capability_artifact": verify_artifact,
        "final_capability_artifact": final_artifact,
        "combined_seal_artifact": seal_artifact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", type=Path, required=True)
    smoke_parser.add_argument("--output", type=Path, required=True)
    precompute_parser = subparsers.add_parser("precompute")
    precompute_parser.add_argument("--config", type=Path, required=True)
    precompute_parser.add_argument("--smoke-receipt", type=Path, required=True)
    precompute_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        result = smoke(args.config, args.output)
    else:
        result = precompute(args.config, args.smoke_receipt, args.output_root)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
