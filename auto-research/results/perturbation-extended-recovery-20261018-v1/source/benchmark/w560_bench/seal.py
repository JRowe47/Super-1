from __future__ import annotations

import argparse
from pathlib import Path

from .common import hash_file_same_fd, sha256_bytes, canonical_json_bytes, write_exclusive_json


INCLUDED_PREFIXES = (
    "w560_bench/",
    "config/benchmark.json",
    "config/benchmark_environment_manifest.v1.json",
    "inputs/tokenizer/",
    "inputs/dataset_provenance/",
)
ENVIRONMENT_MANIFEST_SCHEMA = "hfw_w560_benchmark_environment_manifest_v1"


def seal_environment(runtime_root: Path) -> dict[str, object]:
    root = Path(runtime_root).resolve()
    environment_root = root / "environment/benchmark_pydeps"
    output = root / "config/benchmark_environment_manifest.v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite environment manifest: {output}")
    files = {}
    for path in sorted(environment_root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"environment symlink is forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(environment_root).as_posix()
        if path.suffix in (".pyc", ".pyo") or "__pycache__" in path.parts:
            raise RuntimeError(f"readable bytecode is forbidden: {relative}")
        identity = hash_file_same_fd(path)
        files[relative] = {
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        }
    identity_body = {"schema": ENVIRONMENT_MANIFEST_SCHEMA, "files": files}
    manifest = {
        "schema": ENVIRONMENT_MANIFEST_SCHEMA,
        "status": "SEALED_ALL_REGULAR_ENVIRONMENT_FILES_NO_BYTECODE",
        "file_count": len(files),
        "total_size_bytes": sum(int(value["size_bytes"]) for value in files.values()),
        "manifest_identity_sha256": sha256_bytes(canonical_json_bytes(identity_body)),
        "files": files,
    }
    return {**manifest, "artifact": write_exclusive_json(output, manifest)}


def seal(runtime_root: Path) -> dict[str, object]:
    root = Path(runtime_root).resolve()
    output = root / "config/runtime_manifest.v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime manifest: {output}")
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.endswith(".pyc") or "/__pycache__/" in f"/{relative}/":
            continue
        if not any(relative == prefix or relative.startswith(prefix) for prefix in INCLUDED_PREFIXES):
            continue
        identity = hash_file_same_fd(path)
        files[relative] = {"sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}
    identity_body = {"schema": "hfw_w560_promotion_benchmark_runtime_manifest_v1", "files": files}
    manifest = {
        "schema": "hfw_w560_promotion_benchmark_runtime_manifest_v1",
        "status": "SEALED_CREATE_ONLY_REPORTING_RUNTIME",
        "file_count": len(files),
        "total_size_bytes": sum(int(value["size_bytes"]) for value in files.values()),
        "manifest_identity_sha256": sha256_bytes(canonical_json_bytes(identity_body)),
        "files": files,
    }
    return {**manifest, "artifact": write_exclusive_json(output, manifest)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--environment-only", action="store_true")
    args = parser.parse_args()
    print(seal_environment(args.runtime_root) if args.environment_only else seal(args.runtime_root))


if __name__ == "__main__":
    main()
