from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from .manifest import load_protocol
from .production_phases import (
    authority_finalize_core, authority_prepare_core, run_group_worker_core,
    search_coordinator_core,
)
from .release_validator import validate_local_release
from .artifacts import write_content_addressed_json


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)


def _validate_runtime_root_argument(
    protocol: dict[str, object], runtime_root: Path,
) -> Path:
    expected_runtime = str(protocol["runtime"]["runtime_root"])
    if os.path.abspath(os.fspath(runtime_root)) != expected_runtime:
        raise RuntimeError("production runtime root is not the exact canonical protocol path")
    if runtime_root.resolve() != Path(expected_runtime).resolve():
        raise RuntimeError("production runtime root resolved identity drift")
    return runtime_root


def _validated(arguments):
    package_root = Path(__file__).resolve().parent
    expected_protocol = package_root / "config/production_protocol.json"
    if arguments.protocol.resolve() != expected_protocol.resolve():
        raise RuntimeError("production entrypoint requires its own frozen protocol")
    release = validate_local_release(package_root, arguments.source_root)
    if release.get("status") != "PASS_LOCAL_RELEASE_CREATE_ONLY":
        raise RuntimeError("frozen package release validation did not pass")
    protocol = load_protocol(expected_protocol)
    if arguments.source_root.resolve() != Path(protocol["frozen_source"]["source_root"]).resolve():
        raise RuntimeError("production source root is not the exact protocol-bound frozen source")
    _validate_runtime_root_argument(protocol, arguments.runtime_root)
    if arguments.source_root.resolve() == arguments.runtime_root.resolve():
        raise RuntimeError("frozen source and runtime roots must be distinct")
    return package_root, protocol


def main() -> int:
    parser = argparse.ArgumentParser(description="W560 post-training create-only production phases")
    commands = parser.add_subparsers(dest="phase", required=True)
    authority = commands.add_parser("authority-prepare")
    _common(authority)
    authority.add_argument("--authority-root", type=Path, required=True)
    authority.add_argument("--public-root", type=Path, required=True)
    worker = commands.add_parser("worker")
    _common(worker)
    worker.add_argument("--search-manifest", type=Path, required=True)
    worker.add_argument("--search-manifest-sha256", required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--group", required=True)
    worker.add_argument("--resource-receipt", type=Path, required=True)
    worker.add_argument("--resource-receipt-sha256", required=True)
    search = commands.add_parser("search-all")
    _common(search)
    search.add_argument("--search-manifest", type=Path, required=True)
    search.add_argument("--search-manifest-sha256", required=True)
    search.add_argument("--output-root", type=Path, required=True)
    search.add_argument("--no-writer-receipt", type=Path, required=True)
    final = commands.add_parser("authority-finalize")
    _common(final)
    final.add_argument("--authority-root", type=Path, required=True)
    final.add_argument("--public-root", type=Path, required=True)
    final.add_argument("--output-root", type=Path, required=True)
    final.add_argument("--final-registry-root", type=Path, required=True)
    final.add_argument("--private-authority-receipt", type=Path, required=True)
    final.add_argument("--private-authority-receipt-sha256", required=True)
    final.add_argument("--search-manifest", type=Path, required=True)
    final.add_argument("--search-manifest-sha256", required=True)
    final.add_argument("--search-receipt", type=Path, required=True)
    final.add_argument("--search-receipt-sha256", required=True)
    final.add_argument("--final-no-writer-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    package_root, protocol = _validated(arguments)
    if arguments.phase == "authority-prepare":
        result = authority_prepare_core(
            protocol, package_root, arguments.source_root, arguments.runtime_root,
            authority_root=arguments.authority_root, public_root=arguments.public_root,
        )
    elif arguments.phase == "worker":
        result = run_group_worker_core(
            protocol, package_root, arguments.source_root, arguments.runtime_root,
            search_manifest_path=arguments.search_manifest,
            search_manifest_sha256=arguments.search_manifest_sha256,
            output_root=arguments.output_root, group=arguments.group,
            resource_receipt_path=arguments.resource_receipt,
            resource_receipt_sha256=arguments.resource_receipt_sha256,
        )
    elif arguments.phase == "search-all":
        result = search_coordinator_core(
            protocol, package_root, arguments.source_root, arguments.runtime_root,
            search_manifest_path=arguments.search_manifest,
            search_manifest_sha256=arguments.search_manifest_sha256,
            output_root=arguments.output_root,
            no_writer_receipt_path=arguments.no_writer_receipt,
        )
    else:
        result = authority_finalize_core(
            protocol, package_root, arguments.source_root, arguments.runtime_root,
            authority_root=arguments.authority_root,
            public_root=arguments.public_root,
            output_root=arguments.output_root,
            final_registry_root=arguments.final_registry_root,
            private_authority_receipt_path=arguments.private_authority_receipt,
            private_authority_receipt_sha256=arguments.private_authority_receipt_sha256,
            search_manifest_path=arguments.search_manifest,
            search_manifest_sha256=arguments.search_manifest_sha256,
            search_receipt_path=arguments.search_receipt,
            search_receipt_sha256=arguments.search_receipt_sha256,
            final_no_writer_receipt_path=arguments.final_no_writer_receipt,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


def _failure_root_from_argv() -> Path | None:
    for flag in ("--final-registry-root", "--output-root", "--public-root", "--authority-root"):
        if flag in sys.argv:
            index = sys.argv.index(flag)
            if index + 1 < len(sys.argv):
                return Path(sys.argv[index + 1]) / "failure_receipts"
    return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        root = _failure_root_from_argv()
        if root is not None:
            try:
                write_content_addressed_json(root, ".failure.json", {
                    "schema": "hfw_w560_create_only_phase_failure_receipt_v1",
                    "status": "BLOCK",
                    "phase_argv": sys.argv[1:],
                    "pid": os.getpid(),
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                })
            except BaseException:
                pass
        raise
