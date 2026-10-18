from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .common import hash_file_same_fd, read_json_same_fd, write_exclusive_json
from .config import runtime_root, validate_pinned_inputs


ASSET_SCHEMA = "hfw_w560_promotion_benchmark_assets_v1"


def bind_imported_assets(config_path: Path) -> Mapping[str, object]:
    package = validate_pinned_inputs(config_path)
    config = package["config"]
    root = runtime_root(config_path)
    output = root / "receipts/ASSET_IMPORT_COMPLETE.json"
    if output.exists():
        raise FileExistsError("create-only asset import receipt already exists")
    source = config["asset_import"]
    source_manifest = hash_file_same_fd(Path(str(source["source_manifest_path"])), str(source["source_manifest_sha256"]))
    source_receipt = hash_file_same_fd(Path(str(source["source_receipt_path"])), str(source["source_receipt_sha256"]))
    manifest, manifest_identity = read_json_same_fd(root / "assets/manifest.v1.json", str(source["source_manifest_sha256"]))
    if manifest.get("schema") != ASSET_SCHEMA or manifest.get("status") != "PASS_REVISION_PINNED_PUBLIC_DATASETS_NORMALIZED":
        raise RuntimeError("imported asset manifest schema/status drift")
    files = {}
    for task in config["task_order"]:
        task_record = manifest["datasets"][task]
        path = root / "assets" / task_record["normalized_relative_path"]
        files[task] = hash_file_same_fd(path, str(task_record["normalized_sha256"]))
        if int(files[task]["size_bytes"]) != int(task_record["normalized_size_bytes"]):
            raise RuntimeError(f"imported asset size drift: {task}")
        spec = config["datasets"][task]
        expected = {
            "repo_id": spec["repo_id"], "config": spec.get("config"), "split": spec["split"],
            "requested_revision": spec["revision"], "row_count": spec["expected_rows"],
        }
        for key, value in expected.items():
            if task_record.get(key) != value:
                raise RuntimeError(f"imported asset provenance drift: {task}.{key}")
    receipt = {
        "schema": "hfw_w560_promotion_benchmark_asset_import_receipt_v1",
        "status": "PASS_CREATE_ONLY_LOCAL_ASSET_IMPORT_BOUND",
        "source_asset_manifest": source_manifest,
        "source_asset_receipt": source_receipt,
        "local_asset_manifest": manifest_identity,
        "dataset_files": {task: {"sha256": value["sha256"], "size_bytes": value["size_bytes"]} for task, value in files.items()},
        "config_sha256": package["config_identity"]["sha256"],
        "runtime_manifest_sha256": package["runtime_manifest_identity"]["sha256"],
        "network_access": False,
        "tokenizer_encode_calls": 0,
        "model_forward_calls": 0,
    }
    identity = write_exclusive_json(output, receipt)
    return {**receipt, "receipt": identity}


def validate_assets(config_path: Path) -> Mapping[str, object]:
    package = validate_pinned_inputs(config_path)
    config = package["config"]
    root = runtime_root(config_path)
    import_receipt, import_identity = read_json_same_fd(root / "receipts/ASSET_IMPORT_COMPLETE.json")
    if import_receipt.get("schema") != "hfw_w560_promotion_benchmark_asset_import_receipt_v1" or import_receipt.get("status") != "PASS_CREATE_ONLY_LOCAL_ASSET_IMPORT_BOUND":
        raise RuntimeError("asset import receipt drift")
    if import_receipt.get("config_sha256") != package["config_identity"]["sha256"] or import_receipt.get("runtime_manifest_sha256") != package["runtime_manifest_identity"]["sha256"]:
        raise RuntimeError("asset import/package binding drift")
    manifest, manifest_identity = read_json_same_fd(root / "assets/manifest.v1.json", config["asset_import"]["source_manifest_sha256"])
    if import_receipt.get("local_asset_manifest", {}).get("sha256") != manifest_identity["sha256"]:
        raise RuntimeError("asset import/manifest binding drift")
    for task in config["task_order"]:
        record = manifest["datasets"][task]
        observed = hash_file_same_fd(root / "assets" / record["normalized_relative_path"], record["normalized_sha256"])
        imported = import_receipt["dataset_files"][task]
        if observed["sha256"] != imported["sha256"] or int(observed["size_bytes"]) != int(imported["size_bytes"]):
            raise RuntimeError(f"asset/import receipt byte drift: {task}")
    return {
        "status": "PASS_LOCAL_FROZEN_ASSET_BANK_AND_IMPORT_RECEIPT",
        "package": package,
        "manifest": manifest,
        "manifest_identity": manifest_identity,
        "import_receipt": import_receipt,
        "import_identity": import_identity,
    }
