#!/usr/bin/env python3
"""Verify the complete code package without loading model checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def verify_outer_manifest() -> int:
    manifest_path = ROOT / "MANIFEST.sha256"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("MANIFEST.sha256 is missing")
    expected: dict[str, str] = {}
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        digest, separator, relative = raw.partition("  ")
        if separator != "  " or len(digest) != 64 or not safe_member(relative):
            raise RuntimeError(f"invalid manifest line: {raw!r}")
        if relative in expected:
            raise RuntimeError(f"duplicate manifest path: {relative}")
        expected[relative] = digest
    if list(expected) != sorted(expected):
        raise RuntimeError("MANIFEST.sha256 paths are not sorted")

    actual: set[str] = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"package contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"package contains a special file: {relative}")
        if relative == "MANIFEST.sha256":
            continue
        actual.add(relative)
        file_stat = path.stat()
        if file_stat.st_size > 0 and hasattr(file_stat, "st_blocks") and file_stat.st_blocks == 0:
            raise RuntimeError(f"package contains a dataless/sparse file: {relative}")
    if set(expected) != actual:
        raise RuntimeError(
            f"manifest inventory mismatch; missing={sorted(set(expected)-actual)}, "
            f"extra={sorted(actual-set(expected))}")
    for relative, digest in expected.items():
        observed = sha256_file(ROOT / relative)
        if observed != digest:
            raise RuntimeError(f"manifest hash mismatch: {relative}")
    return len(expected)


def verify_zip(path: Path, expanded: Path) -> int:
    observed: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not safe_member(info.filename):
                raise RuntimeError(f"unsafe ZIP path: {info.filename}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"ZIP contains link: {info.filename}")
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            relative = PurePosixPath(*parts[1:]).as_posix()
            if not relative:
                raise RuntimeError(f"ZIP member lacks namespace: {info.filename}")
            data = archive.read(info)
            target = expanded / relative
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(f"expanded ZIP member missing: {relative}")
            if sha256_bytes(data) != sha256_file(target):
                raise RuntimeError(f"expanded ZIP member differs: {relative}")
            observed.add(relative)
    expanded_files = {
        item.relative_to(expanded).as_posix()
        for item in expanded.rglob("*") if item.is_file()
    }
    if observed != expanded_files:
        raise RuntimeError("expanded ZIP inventory differs from archive")
    return len(observed)


def verify_tar(path: Path, expanded: Path) -> int:
    observed: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not safe_member(member.name):
                raise RuntimeError(f"unsafe tar path: {member.name}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"tar contains link/special member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"tar contains unsupported member: {member.name}")
            parts = PurePosixPath(member.name).parts
            relative = PurePosixPath(*parts[1:]).as_posix()
            if not relative:
                raise RuntimeError(f"tar member lacks namespace: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read tar member: {member.name}")
            data = handle.read()
            target = expanded / relative
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(f"expanded tar member missing: {relative}")
            if sha256_bytes(data) != sha256_file(target):
                raise RuntimeError(f"expanded tar member differs: {relative}")
            observed.add(relative)
    expanded_files = {
        item.relative_to(expanded).as_posix()
        for item in expanded.rglob("*") if item.is_file()
    }
    if observed != expanded_files:
        raise RuntimeError("expanded tar inventory differs from archive")
    return len(observed)


def verify_archives(package: dict) -> dict:
    result: dict[str, int] = {}
    for name, spec in package["archives"].items():
        if spec.get("included", True) is False:
            continue
        path = ROOT / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"source archive hash mismatch: {name}")
        expanded = ROOT / package["paths"][
            "historical_source" if name == "historical_hierarchy" else "nanogpt_source"]
        if spec["format"] == "zip":
            result[name] = verify_zip(path, expanded)
        elif spec["format"] == "tar.gz":
            result[name] = verify_tar(path, expanded)
        else:
            raise RuntimeError(f"unsupported archive format: {spec['format']}")
    return result


def verify_source_identity(package: dict) -> None:
    identity = load_json(ROOT / "provenance/source_identity.json")
    historical = ROOT / package["paths"]["historical_source"]
    for relative, digest in identity["historical_hierarchy"]["included_snapshot"].items():
        if sha256_file(historical / relative) != digest:
            raise RuntimeError(f"historical source identity mismatch: {relative}")
    nano = ROOT / package["paths"]["nanogpt_source"]
    internal_manifest = nano / "freeze/manifest.v1.json"
    expected_manifest = identity["official_nanogpt"]["freeze_manifest_sha256"]
    if sha256_file(internal_manifest) != expected_manifest:
        raise RuntimeError("official nanoGPT freeze manifest hash mismatch")
    manifest = load_json(internal_manifest)
    if manifest.get("status") != "FROZEN_AFTER_INDEPENDENT_PASS":
        raise RuntimeError("official nanoGPT freeze status is not PASS")
    if manifest.get("active_file_count") != len(manifest.get("files", {})):
        raise RuntimeError("official nanoGPT manifest count mismatch")
    for relative, record in manifest["files"].items():
        path = nano / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"official nanoGPT file missing: {relative}")
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"official nanoGPT file drift: {relative}")


def verify_recorded_artifacts(package: dict) -> None:
    recorded = load_json(ROOT / "provenance/recorded_artifacts.json")
    paths = package["paths"]
    included = {
        ROOT / paths["historical_config"]: recorded["historical_hierarchy"]["config"],
        ROOT / paths["recorded_hierarchy_result"]: recorded["historical_hierarchy"]["result"],
        ROOT / paths["recorded_hierarchy_benchmark"]: recorded["historical_hierarchy"]["benchmark"],
        ROOT / paths["recorded_nanogpt_result"]: recorded["official_nanogpt"]["result"],
        ROOT / paths["recorded_nanogpt_training_log"]: recorded["official_nanogpt"]["training_log"],
        ROOT / paths["recorded_nanogpt_benchmark"]: recorded["official_nanogpt"]["benchmark"],
        ROOT / "evidence/recorded/nanogpt/exact_contract_preflight.json": recorded["official_nanogpt"]["training_preflight"],
        ROOT / "evidence/recorded/nanogpt/exact_contract_completion.json": recorded["official_nanogpt"]["training_completion"],
        ROOT / "evidence/recorded/nanogpt/exact_benchmark_completion.json": recorded["official_nanogpt"]["benchmark_completion"],
        ROOT / "evidence/recorded/nanogpt/resolved_config.json": recorded["official_nanogpt"]["resolved_config"],
    }
    for path, digest in included.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"recorded artifact mismatch: {path.relative_to(ROOT)}")

    completion = load_json(ROOT / "evidence/recorded/nanogpt/exact_contract_completion.json")
    benchmark_completion = load_json(ROOT / "evidence/recorded/nanogpt/exact_benchmark_completion.json")
    if completion.get("status") != "PASS" or completion.get("result_sha256") != recorded["official_nanogpt"]["result"]:
        raise RuntimeError("nanoGPT training completion binding is invalid")
    if completion.get("checkpoint_sha256") != recorded["official_nanogpt"]["final_model_not_included"]:
        raise RuntimeError("nanoGPT final model binding is invalid")
    if benchmark_completion.get("status") != "PASS" or benchmark_completion.get("benchmark_sha256") != recorded["official_nanogpt"]["benchmark"]:
        raise RuntimeError("nanoGPT benchmark completion binding is invalid")
    if benchmark_completion.get("training_completion_sha256") != recorded["official_nanogpt"]["training_completion"]:
        raise RuntimeError("nanoGPT benchmark does not bind training completion")

    log_path = ROOT / paths["recorded_nanogpt_training_log"]
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 246 or rows[-1].get("step") != package["execution_contract"]["steps"]:
        raise RuntimeError("nanoGPT training log shape is invalid")


def verify_report(package: dict) -> None:
    markdown = (ROOT / "report/source_report.md").read_text(encoding="utf-8")
    builder = (ROOT / "report/build_report.py").read_text(encoding="utf-8")
    if "Critical limitations" in markdown or "Critical limitations" in builder:
        raise RuntimeError("removed report section is still present")
    if any(path.suffix.lower() == ".pdf" for path in ROOT.rglob("*")):
        raise RuntimeError("the code package must not contain a PDF")
    for name in ["fixed_eval_ce.png", "state_scaling.png", "matched_training_ce.png", "benchmark_scores.png"]:
        path = ROOT / "report/charts" / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"report chart missing: {name}")
    subprocess.run([
        sys.executable, "-B", str(ROOT / "tools/compare_results.py"),
        "--recorded", "--check", str(ROOT / package["paths"]["report_metrics"]),
    ], cwd=ROOT, check=True)


def compile_sources() -> int:
    count = 0
    for path in ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        count += 1
    return count


def run_tests(package: dict) -> None:
    nano = ROOT / package["paths"]["nanogpt_source"]
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(nano),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
    })
    subprocess.run([
        sys.executable, "-B", "-W", "error", "-m", "unittest", "discover",
        "-s", str(nano / "tests"), "-p", "test_exact_comparator.py", "-v",
    ], cwd=nano, env=env, check=True)
    historical = ROOT / package["paths"]["historical_source"]
    historical_env = dict(env)
    historical_env["PYTHONPATH"] = str(historical)
    subprocess.run([
        sys.executable, "-B",
        str(historical / "representation_experiments/smoke_matched_hierarchy.py"),
        "--config", str(ROOT / package["paths"]["historical_config"]),
    ], cwd=historical, env=historical_env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    package = load_json(ROOT / "package.json")
    manifest_count = verify_outer_manifest()
    archive_counts = verify_archives(package)
    verify_source_identity(package)
    verify_recorded_artifacts(package)
    verify_report(package)
    compiled = compile_sources()
    if args.run_tests:
        run_tests(package)
    print(json.dumps({
        "status": "PASS_REPRODUCTION_PACKAGE",
        "manifest_files": manifest_count,
        "archive_regular_files": archive_counts,
        "compiled_python_files": compiled,
        "comparator_tests_run": args.run_tests,
        "pdf_in_archive": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
