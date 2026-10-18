from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Mapping, Sequence

from .common import (
    canonical_json_bytes, fsync_directory, hash_file, read_bytes, read_json,
    sha256_bytes, write_json_exclusive,
)
from .seal_upload_staging import (
    STAGING_SCHEMA,
    STAGING_STATUS,
    build_upload_staging_receipt,
)


MANIFEST_RELATIVE_PATH = Path("runtime/config/HOST_RUNTIME_MANIFEST.json")
VENV_AUDIT_RELATIVE_PATH = Path("runtime/config/DAYTONA_MANAGER_VENV_AUDIT.json")
VENV_LOCK_RELATIVE_PATH = Path("runtime/daytona-manager-requirements.lock")
VENV_PYTHON_RELATIVE_PATH = Path("runtime/daytona-manager-venv/bin/python")
UPLOAD_STAGING_RELATIVE_PATH = Path("runtime/config/UPLOAD_SNAPSHOT_STAGING.json")
LOCAL_ARM_PREFLIGHT_RELATIVE_PATH = Path(
    "runtime/config/LOCAL_ARM_FP64_SHADOW_DIAGNOSTIC.json"
)
RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH = Path(
    "runtime/config/RECOVERED_TASK0_PROVENANCE.json"
)
TEST_RUN_RECEIPT_RELATIVE_PATH = Path("runtime/config/TEST_RUN_RECEIPT.json")
TEST_RESULTS_RELATIVE_PATH = Path("runtime/test-results")
TEST_RUN_SPEC_SCHEMA = "hfw_w560_test_run_spec_v2"
TEST_RUN_RECEIPT_SCHEMA = "hfw_w560_test_run_receipt_v2"
TEST_RUN_RECEIPT_STATUS = "PASS_RUNNER_EXECUTED_ALL_DECLARED_UNITTEST_MODULES"
HOST_MANIFEST_SCHEMA = "hfw_w560_distributed_host_runtime_manifest_v2"


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _locked_distributions(package_root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in (package_root / VENV_LOCK_RELATIVE_PATH).read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if match is None:
            raise RuntimeError("manager venv requirements lock contains a malformed line")
        name = _normalized_distribution_name(match.group(1))
        if name in rows:
            raise RuntimeError("manager venv requirements lock contains a duplicate distribution")
        rows[name] = match.group(2)
    if not rows:
        raise RuntimeError("manager venv requirements lock is empty")
    return rows


def _validate_manager_venv(package_root: Path) -> Mapping[str, object]:
    audit, audit_identity = read_json(package_root / VENV_AUDIT_RELATIVE_PATH)
    lock_identity = hash_file(
        package_root / VENV_LOCK_RELATIVE_PATH,
        str(audit["requirements_lock"]["sha256"]),
    )
    if int(lock_identity["size_bytes"]) != int(audit["requirements_lock"]["size_bytes"]):
        raise RuntimeError("manager venv lock size drift")
    locked = _locked_distributions(package_root)
    if len(locked) != int(audit["requirements_lock"]["distribution_count"]):
        raise RuntimeError("manager venv lock distribution count drift")
    interpreter = os.path.abspath(os.fspath(package_root / VENV_PYTHON_RELATIVE_PATH))
    probe = subprocess.run(
        [interpreter, "-I", "-B", "-c", (
            "import importlib.metadata as m,json,platform,re,sys;"
            "norm=lambda x:re.sub(r'[-_.]+','-',x).lower();"
            "print(json.dumps({'implementation':platform.python_implementation(),"
            "'python_version':platform.python_version(),'platform_system':platform.system(),"
            "'machine':platform.machine(),'prefix':sys.prefix,'daytona_version':m.version('daytona'),"
            "'distributions':{norm(d.metadata['Name']):d.version for d in m.distributions()}},"
            "sort_keys=True))"
        )],
        check=True, capture_output=True, text=True, timeout=60,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    observed = json.loads(probe.stdout)
    expected = {
        "schema": "hfw_w560_daytona_manager_venv_audit_v1",
        "status": "PASS_EXACT_LOCKED_EXTERNAL_MANAGER_ENVIRONMENT",
        "manager_python_relative_path": VENV_PYTHON_RELATIVE_PATH.as_posix(),
        "manager_prefix_relative_path": "runtime/daytona-manager-venv",
        "requirements_lock": {
            "relative_path": VENV_LOCK_RELATIVE_PATH.as_posix(),
            "sha256": lock_identity["sha256"],
            "size_bytes": lock_identity["size_bytes"],
            "distribution_count": len(locked),
        },
        "python": {
            "implementation": observed["implementation"],
            "version": observed["python_version"],
            "major_minor": ".".join(observed["python_version"].split(".")[:2]),
            "platform_system": observed["platform_system"],
            "machine": observed["machine"],
        },
        "daytona_sdk_version": observed["daytona_version"],
        "installed_distributions_exactly_equal_lock": observed["distributions"] == locked,
        "manager_prefix_is_package_local_venv": (
            Path(observed["prefix"]) == package_root / "runtime/daytona-manager-venv"
        ),
        "credential_environment_forwarded_to_audit_probe": False,
    }
    if observed["distributions"] != locked:
        raise RuntimeError("manager venv installed distribution inventory differs from lock")
    if canonical_json_bytes(audit) != canonical_json_bytes(expected):
        raise RuntimeError("manager venv audit receipt differs from live sanitized probe")
    return {**audit, "identity": audit_identity}


def _package_relative_file(
    package_root: Path,
    value: object,
    *,
    field: str,
) -> tuple[Path, str]:
    supplied = Path(str(value))
    candidate = supplied if supplied.is_absolute() else package_root / supplied
    candidate = candidate.resolve()
    if package_root not in candidate.parents:
        raise RuntimeError(f"{field} is outside the package root")
    relative = candidate.relative_to(package_root).as_posix()
    if not candidate.is_file():
        raise RuntimeError(f"{field} is not a regular file: {relative}")
    return candidate, relative


def _test_source_paths(package_root: Path) -> dict[str, Path]:
    tests_root = package_root / "runtime/tests"
    paths = sorted(tests_root.glob("test_*.py"))
    if not paths:
        raise RuntimeError("test source inventory is empty")
    return {
        path.relative_to(package_root).as_posix(): path
        for path in paths
    }


def _test_result_paths(package_root: Path) -> dict[str, Path]:
    results_root = package_root / TEST_RESULTS_RELATIVE_PATH
    if not results_root.is_dir():
        raise RuntimeError("test result directory is absent")
    result: dict[str, Path] = {}
    for path in sorted(results_root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("test result tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("test result tree contains a special entry")
        relative = path.relative_to(package_root).as_posix()
        result[relative] = path
    if not result:
        raise RuntimeError("test result inventory is empty")
    return result


def _identity_record(path: Path) -> dict[str, object]:
    identity = hash_file(path)
    return {
        "sha256": identity["sha256"],
        "size_bytes": int(identity["size_bytes"]),
    }


def _successful_unittest_result_record(
    path: Path,
    *,
    expected_tests_run: int | None = None,
) -> tuple[dict[str, object], int]:
    payload, identity = read_bytes(path)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("test result log is not UTF-8") from error
    counts = [
        int(value)
        for value in re.findall(r"(?m)^Ran ([0-9]+) tests? in .+$", text)
    ]
    success_lines = re.findall(r"(?m)^OK(?: \([^\n]+\))?$", text)
    failure_lines = re.findall(r"(?m)^FAILED(?: \([^\n]+\))?$", text)
    if (
        len(counts) != 1
        or counts[0] <= 0
        or len(success_lines) != 1
        or failure_lines
        or (
            expected_tests_run is not None
            and counts[0] != expected_tests_run
        )
    ):
        raise RuntimeError("test result log does not prove one nonempty unittest run passed")
    return {
        "sha256": identity["sha256"],
        "size_bytes": int(identity["size_bytes"]),
    }, counts[0]


def _canonical_interpreter(value: object) -> tuple[Path, Path, dict[str, object]]:
    supplied = Path(str(value))
    if not supplied.is_absolute():
        raise RuntimeError("test interpreter path must be absolute")
    entrypoint = Path(os.path.abspath(os.fspath(supplied)))
    if entrypoint != supplied:
        raise RuntimeError("test interpreter path must be normalized")
    interpreter = entrypoint.resolve()
    if not entrypoint.exists() or not interpreter.is_file():
        raise RuntimeError("test interpreter is not a real regular file")
    return entrypoint, interpreter, {
        "path": str(entrypoint),
        "resolved_path": str(interpreter),
        **_identity_record(interpreter),
    }


def _test_environment_for_profile(
    package_root: Path,
    value: object,
) -> tuple[str, dict[str, str]]:
    host_profile = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str((package_root / "runtime").resolve()),
    }
    manager_profile = {
        **host_profile,
        "PYTHONNOUSERSITE": "1",
    }
    if value == "host":
        return "host", host_profile
    if value == "manager":
        return "manager", manager_profile
    raise RuntimeError("test environment profile is not sanctioned")


def _canonical_test_module(relative: str) -> str:
    path = Path(relative)
    if (
        not relative.startswith("runtime/tests/test_")
        or path.suffix != ".py"
        or path.parent.as_posix() != "runtime/tests"
        or path.as_posix() != relative
    ):
        raise RuntimeError(f"test source is not a canonical test module: {relative}")
    return f"tests.{path.stem}"


def _validated_production_staging_path(
    package_root: Path,
    value: object,
) -> Path:
    supplied = Path(str(value))
    if not supplied.is_absolute():
        raise RuntimeError("production staging path must be absolute")
    lexical = Path(os.path.abspath(os.fspath(supplied)))
    if lexical != supplied or lexical.resolve(strict=False) != lexical:
        raise RuntimeError("production staging path must be normalized with real parents")
    if (
        lexical == package_root
        or lexical in package_root.parents
        or package_root in lexical.parents
    ):
        raise RuntimeError("production staging and package roots must be non-nested")
    if any(any(character.isspace() for character in part) for part in lexical.parts):
        raise RuntimeError("production staging path contains whitespace")
    return lexical


def _require_production_staging_absent(path: Path, *, phase: str) -> None:
    if os.path.lexists(path):
        raise RuntimeError(f"production staging path exists {phase}: {path}")


def _atomic_write_bytes_create_only(path: Path, payload: bytes) -> dict[str, object]:
    """Publish a complete immutable log in one create-only directory operation."""
    path = Path(path)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent,
    )
    staging = Path(staging_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(staging, path)
        published = True
        staging.unlink()
        fsync_directory(path.parent)
        return hash_file(path)
    except BaseException:
        if published:
            path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        staging.unlink(missing_ok=True)
        fsync_directory(path.parent)


def _normalized_test_run_declarations(
    package_root: Path,
    *,
    runs: Sequence[Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    expected_sources = _test_source_paths(package_root)
    source_records = {
        relative: _identity_record(path)
        for relative, path in sorted(expected_sources.items())
    }
    if not runs:
        raise RuntimeError("test run spec requires at least one run")
    declarations: list[dict[str, object]] = []
    names: set[str] = set()
    covered_sources: set[str] = set()
    allowed_keys = {
        "name", "interpreter", "environment_profile", "test_modules",
        "timeout_seconds",
    }
    for index, run in enumerate(runs):
        if set(run) != allowed_keys:
            raise RuntimeError("test run declaration contains missing or unexpected fields")
        name = run.get("name")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None
            or name in names
        ):
            raise RuntimeError("test run names must be unique safe filenames")
        names.add(name)
        interpreter_input = run.get("interpreter")
        if isinstance(interpreter_input, Mapping):
            interpreter_input = interpreter_input.get("path")
        interpreter_entrypoint, _, interpreter_record = _canonical_interpreter(
            interpreter_input
        )
        profile, environment = _test_environment_for_profile(
            package_root, run.get("environment_profile"),
        )
        modules_input = run.get("test_modules")
        if not isinstance(modules_input, Mapping) or not modules_input:
            raise RuntimeError("test run requires a nonempty test_modules mapping")
        modules: dict[str, str] = {}
        for module, relative in sorted(modules_input.items()):
            if not isinstance(module, str) or not isinstance(relative, str):
                raise RuntimeError("test module mapping must contain string pairs")
            if relative not in expected_sources:
                raise RuntimeError(f"test module maps to an unknown source: {relative}")
            if module != _canonical_test_module(relative):
                raise RuntimeError(
                    f"test module/source mapping is not canonical: {module} -> {relative}"
                )
            if relative in covered_sources:
                raise RuntimeError(f"test source is assigned to multiple runs: {relative}")
            covered_sources.add(relative)
            modules[module] = relative
        timeout_seconds = run.get("timeout_seconds")
        if (
            type(timeout_seconds) is not int
            or timeout_seconds <= 0
            or timeout_seconds > 86400
        ):
            raise RuntimeError("test timeout must be an integer from 1 through 86400")
        isolation_flags = ["-P", "-B"] if profile == "manager" else ["-B"]
        command = [
            str(interpreter_entrypoint), *isolation_flags,
            "-m", "unittest", "-v", *sorted(modules),
        ]
        result_relative = (
            TEST_RESULTS_RELATIVE_PATH / f"run-{index:03d}-{name}.log"
        ).as_posix()
        declarations.append({
            "name": name,
            "interpreter": interpreter_record,
            "environment_profile": profile,
            "environment": environment,
            "command": command,
            "command_sha256": sha256_bytes(canonical_json_bytes(command)),
            "test_modules": modules,
            "working_directory": str(package_root),
            "timeout_seconds": timeout_seconds,
            "result_log_path": result_relative,
        })
    if covered_sources != set(expected_sources):
        missing = sorted(set(expected_sources) - covered_sources)
        raise RuntimeError(f"test run declarations do not cover every test source: {missing}")
    return source_records, declarations


def create_test_run_receipt(
    package_root: Path,
    *,
    production_staging_path: Path,
    runs: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Execute a declarative unittest plan and seal runner-owned evidence."""
    package = Path(package_root).resolve()
    staging = _validated_production_staging_path(
        package, production_staging_path,
    )
    source_records, declarations = _normalized_test_run_declarations(
        package, runs=runs,
    )
    spec_binding = {
        "schema": TEST_RUN_SPEC_SCHEMA,
        "package_root": str(package),
        "production_staging_path": str(staging),
        "test_sources": source_records,
        "runs": declarations,
    }
    spec_sha256 = sha256_bytes(canonical_json_bytes(spec_binding))
    receipt_path = package / TEST_RUN_RECEIPT_RELATIVE_PATH
    if receipt_path.exists():
        existing = _validate_test_run_receipt(package)
        if existing.get("declarative_spec_sha256") != spec_sha256:
            raise RuntimeError("test run receipt already exists for a different spec")
        return existing

    results_root = package / TEST_RESULTS_RELATIVE_PATH
    if os.path.lexists(results_root):
        raise RuntimeError("test result directory exists before runner execution")
    results_root.mkdir(parents=True)
    fsync_directory(results_root.parent)
    run_records: list[dict[str, object]] = []
    for declaration in declarations:
        name = str(declaration["name"])
        _require_production_staging_absent(staging, phase=f"before test run {name}")
        output: bytes
        return_code: int
        try:
            try:
                completed = subprocess.run(
                    declaration["command"],
                    cwd=str(package),
                    env=dict(declaration["environment"]),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=int(declaration["timeout_seconds"]),
                )
                output = completed.stdout
                return_code = int(completed.returncode)
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout if isinstance(error.stdout, bytes) else b""
                output = stdout + b"\nRUNNER_TIMEOUT\n"
                return_code = 124
            except OSError as error:
                output = f"RUNNER_EXEC_ERROR: {error}\n".encode("utf-8", "replace")
                return_code = 126
            result_relative = str(declaration["result_log_path"])
            result_path = package / result_relative
            result_identity = _atomic_write_bytes_create_only(result_path, output)
        finally:
            _require_production_staging_absent(
                staging, phase=f"after test run {name}",
            )
        if return_code != 0:
            raise RuntimeError(
                f"runner-owned test subprocess failed: {name} exit {return_code}"
            )
        parsed_identity, tests_run = _successful_unittest_result_record(result_path)
        if (
            parsed_identity["sha256"] != result_identity["sha256"]
            or parsed_identity["size_bytes"] != result_identity["size_bytes"]
        ):
            raise RuntimeError("test result identity changed after atomic publication")
        run_records.append({
            **declaration,
            "result_log": {
                "path": result_relative,
                **parsed_identity,
            },
            "exit_code": return_code,
            "tests_run": tests_run,
            "production_staging_absent_before_run": True,
            "production_staging_absent_after_run": True,
            "stdout_and_stderr_combined_by_runner": True,
        })

    expected_logs = {
        str(declaration["result_log_path"]) for declaration in declarations
    }
    if set(_test_result_paths(package)) != expected_logs:
        raise RuntimeError("runner-owned result inventory is incomplete or unexpected")
    value = {
        "schema": TEST_RUN_RECEIPT_SCHEMA,
        "status": TEST_RUN_RECEIPT_STATUS,
        "package_root": str(package),
        "declarative_spec_schema": TEST_RUN_SPEC_SCHEMA,
        "declarative_spec_sha256": spec_sha256,
        "production_staging_path": str(staging),
        "test_sources": source_records,
        "test_sources_sha256": sha256_bytes(canonical_json_bytes(source_records)),
        "runs": run_records,
        "run_count": len(run_records),
        "total_tests_run": sum(int(run["tests_run"]) for run in run_records),
        "production_staging_untouched_by_tests": True,
        "production_staging_absent_before_and_after_every_run": True,
        "runner_owned_subprocess_execution": True,
        "runner_derived_commands_from_explicit_module_source_mapping": True,
        "runner_derived_exit_codes_test_counts_and_result_logs": True,
    }
    identity = write_json_exclusive(receipt_path, value)
    return {**value, "identity": identity}


def _validate_test_run_receipt(package_root: Path) -> Mapping[str, object]:
    receipt, receipt_identity = read_json(
        package_root / TEST_RUN_RECEIPT_RELATIVE_PATH
    )
    expected_receipt_fields = {
        "schema", "status", "package_root", "declarative_spec_schema",
        "declarative_spec_sha256", "production_staging_path", "test_sources",
        "test_sources_sha256", "runs", "run_count", "total_tests_run",
        "production_staging_untouched_by_tests",
        "production_staging_absent_before_and_after_every_run",
        "runner_owned_subprocess_execution",
        "runner_derived_commands_from_explicit_module_source_mapping",
        "runner_derived_exit_codes_test_counts_and_result_logs",
    }
    if (
        set(receipt) != expected_receipt_fields
        or
        receipt.get("schema") != TEST_RUN_RECEIPT_SCHEMA
        or receipt.get("status") != TEST_RUN_RECEIPT_STATUS
        or receipt.get("package_root") != str(package_root)
        or receipt.get("declarative_spec_schema") != TEST_RUN_SPEC_SCHEMA
        or receipt.get("production_staging_untouched_by_tests") is not True
        or receipt.get(
            "production_staging_absent_before_and_after_every_run"
        ) is not True
        or receipt.get("runner_owned_subprocess_execution") is not True
        or receipt.get(
            "runner_derived_commands_from_explicit_module_source_mapping"
        ) is not True
        or receipt.get(
            "runner_derived_exit_codes_test_counts_and_result_logs"
        ) is not True
    ):
        raise RuntimeError("test run receipt binding drift")

    staging = _validated_production_staging_path(
        package_root, receipt.get("production_staging_path"),
    )

    expected_sources = _test_source_paths(package_root)
    source_records = receipt.get("test_sources")
    if not isinstance(source_records, Mapping) or set(source_records) != set(expected_sources):
        raise RuntimeError("test run receipt source inventory drift")
    for relative, path in expected_sources.items():
        if canonical_json_bytes(source_records[relative]) != canonical_json_bytes(
            _identity_record(path)
        ):
            raise RuntimeError(f"test source identity drift: {relative}")
    if receipt.get("test_sources_sha256") != sha256_bytes(
        canonical_json_bytes(source_records)
    ):
        raise RuntimeError("test source inventory summary drift")

    runs = receipt.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RuntimeError("test run receipt has no runs")
    referenced_logs: set[str] = set()
    total_tests = 0
    declaration_inputs: list[dict[str, object]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            raise RuntimeError("test run receipt contains a malformed run")
        declaration_inputs.append({
            "name": run.get("name"),
            "interpreter": (
                run.get("interpreter", {}).get("path")
                if isinstance(run.get("interpreter"), Mapping)
                else None
            ),
            "environment_profile": run.get("environment_profile"),
            "test_modules": run.get("test_modules"),
            "timeout_seconds": run.get("timeout_seconds"),
        })

    expected_source_records, expected_declarations = (
        _normalized_test_run_declarations(
            package_root, runs=declaration_inputs,
        )
    )
    if canonical_json_bytes(source_records) != canonical_json_bytes(
        expected_source_records
    ):
        raise RuntimeError("test source record normalization drift")

    if len(runs) != len(expected_declarations):
        raise RuntimeError("test run declaration count drift")
    for run, declaration in zip(runs, expected_declarations):
        name = str(declaration["name"])
        expected_run_fields = set(declaration) | {
            "result_log", "exit_code", "tests_run",
            "production_staging_absent_before_run",
            "production_staging_absent_after_run",
            "stdout_and_stderr_combined_by_runner",
        }
        if set(run) != expected_run_fields:
            raise RuntimeError(f"runner-owned test record fields drift: {name}")
        for field in declaration:
            if canonical_json_bytes(run.get(field)) != canonical_json_bytes(
                declaration[field]
            ):
                raise RuntimeError(f"runner-owned test declaration drift: {name}/{field}")
        result_record = run.get("result_log")
        if not isinstance(result_record, Mapping):
            raise RuntimeError("test result identity is absent")
        result_path, result_relative = _package_relative_file(
            package_root, result_record.get("path"), field="test result log",
        )
        if result_relative != declaration["result_log_path"]:
            raise RuntimeError(f"runner-owned result path drift: {name}")
        exit_code = run.get("exit_code")
        tests_run = run.get("tests_run")
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or isinstance(tests_run, bool)
            or not isinstance(tests_run, int)
            or exit_code != 0
            or tests_run <= 0
            or run.get("production_staging_absent_before_run") is not True
            or run.get("production_staging_absent_after_run") is not True
            or run.get("stdout_and_stderr_combined_by_runner") is not True
        ):
            raise RuntimeError(f"test run is not successful: {name}")
        parsed_identity, parsed_tests_run = _successful_unittest_result_record(
            result_path, expected_tests_run=tests_run,
        )
        expected_result = {
            "path": result_relative,
            **parsed_identity,
        }
        if (
            not result_relative.startswith(TEST_RESULTS_RELATIVE_PATH.as_posix() + "/")
            or result_relative in referenced_logs
            or canonical_json_bytes(result_record)
            != canonical_json_bytes(expected_result)
        ):
            raise RuntimeError(f"test result identity drift: {name}")
        referenced_logs.add(result_relative)
        total_tests += parsed_tests_run
    if referenced_logs != set(_test_result_paths(package_root)):
        raise RuntimeError("test result inventory differs from test run receipt")
    spec_binding = {
        "schema": TEST_RUN_SPEC_SCHEMA,
        "package_root": str(package_root),
        "production_staging_path": str(staging),
        "test_sources": expected_source_records,
        "runs": expected_declarations,
    }
    if receipt.get("declarative_spec_sha256") != sha256_bytes(
        canonical_json_bytes(spec_binding)
    ):
        raise RuntimeError("test declarative spec identity drift")
    run_count = receipt.get("run_count")
    total_tests_run = receipt.get("total_tests_run")
    if (
        type(run_count) is not int
        or type(total_tests_run) is not int
        or run_count != len(runs)
        or total_tests_run != total_tests
    ):
        raise RuntimeError("test run receipt summary drift")
    return {**receipt, "identity": receipt_identity}


def _validate_upload_snapshot_staging(
    package_root: Path, wave_root: Path,
) -> Mapping[str, object]:
    receipt, receipt_identity = read_json(package_root / UPLOAD_STAGING_RELATIVE_PATH)
    if (
        receipt.get("schema") != STAGING_SCHEMA
        or receipt.get("status") != STAGING_STATUS
    ):
        raise RuntimeError("upload snapshot staging receipt binding drift")
    expected = build_upload_staging_receipt(
        package_root,
        Path(wave_root).resolve() / "snapshot",
        Path(str(receipt.get("upload_snapshot_root", ""))),
    )
    if canonical_json_bytes(receipt) != canonical_json_bytes(expected):
        raise RuntimeError("upload snapshot staging receipt differs from live proof")
    return {**receipt, "identity": receipt_identity}


def _validate_test_staging_path_binding(
    test_run_receipt: Mapping[str, object],
    upload_staging: Mapping[str, object],
) -> str:
    test_path = test_run_receipt.get("production_staging_path")
    upload_path = upload_staging.get("upload_snapshot_root")
    if (
        not isinstance(test_path, str)
        or not isinstance(upload_path, str)
        or test_path != upload_path
    ):
        raise RuntimeError(
            "test-time absent production staging path differs from sealed upload root"
        )
    return test_path


def _host_files(package_root: Path) -> list[Path]:
    package_root = Path(package_root).resolve()
    paths = [
        package_root / "README.md",
        package_root / VENV_LOCK_RELATIVE_PATH,
        package_root / VENV_AUDIT_RELATIVE_PATH,
        package_root / UPLOAD_STAGING_RELATIVE_PATH,
        package_root / LOCAL_ARM_PREFLIGHT_RELATIVE_PATH,
        package_root / TEST_RUN_RECEIPT_RELATIVE_PATH,
    ]
    recovery = package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
    if recovery.exists():
        paths.append(recovery)
    paths.extend(sorted((package_root / "runtime/hfw_dist").glob("*.py")))
    paths.extend(sorted((package_root / "runtime/tests").glob("*.py")))
    paths.extend(sorted(_test_result_paths(package_root).values()))
    paths.extend(sorted(
        path for path in (package_root / "runtime/frozen").rglob("*") if path.is_file()
    ))
    return paths


def build_manifest(package_root: Path, wave_root: Path) -> Mapping[str, object]:
    package_root = Path(package_root).resolve()
    wave_root = Path(wave_root).resolve()
    test_run_receipt = _validate_test_run_receipt(package_root)
    inventory = {}
    for path in _host_files(package_root):
        identity = hash_file(path)
        inventory[path.relative_to(package_root).as_posix()] = {
            "sha256": identity["sha256"], "size_bytes": identity["size_bytes"],
        }
    test_run_receipt_after_inventory = _validate_test_run_receipt(package_root)
    if canonical_json_bytes(test_run_receipt) != canonical_json_bytes(
        test_run_receipt_after_inventory
    ):
        raise RuntimeError("test run evidence changed while building host inventory")
    campaign = hash_file(wave_root / "snapshot/CAMPAIGN.json")
    snapshot = hash_file(wave_root / "snapshot/SNAPSHOT_MANIFEST.json")
    manager_venv = _validate_manager_venv(package_root)
    upload_staging = _validate_upload_snapshot_staging(package_root, wave_root)
    production_staging_path = _validate_test_staging_path_binding(
        test_run_receipt, upload_staging,
    )
    recovery_path = package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
    recovery = None
    if recovery_path.exists():
        recovery_value, recovery_identity = read_json(recovery_path)
        if (
            recovery_value.get("schema")
            != "hfw_w560_recovered_task0_provenance_v3"
            or recovery_value.get("status")
            != "RECOVERED_AND_REVALIDATED_NO_RERUN"
        ):
            raise RuntimeError("recovered task0 provenance schema/status drift")
        recovery = recovery_identity
    return {
        "schema": HOST_MANIFEST_SCHEMA,
        "status": "SEALED_NOT_LAUNCHED",
        "inventory": inventory,
        "inventory_file_count": len(inventory),
        "wave_index": int(read_json(wave_root / "snapshot/CAMPAIGN.json")[0]["wave_index"]),
        "campaign": campaign,
        "snapshot_manifest": snapshot,
        "manager_venv_audit": manager_venv["identity"],
        "test_run_receipt": {
            "receipt": test_run_receipt["identity"],
            "run_count": test_run_receipt["run_count"],
            "total_tests_run": test_run_receipt["total_tests_run"],
            "test_sources_sha256": test_run_receipt["test_sources_sha256"],
            "production_staging_untouched_by_tests": True,
            "production_staging_path": production_staging_path,
        },
        "upload_snapshot_staging": {
            "receipt": upload_staging["identity"],
            "upload_snapshot_root": upload_staging["upload_snapshot_root"],
            "complete_inventory_sha256": upload_staging["complete_inventory_sha256"],
            "file_count": upload_staging["file_count"],
            "directory_count": upload_staging["directory_count"],
            "total_size_bytes": upload_staging["total_size_bytes"],
            "campaign": upload_staging["campaign"],
            "snapshot_manifest": upload_staging["snapshot_manifest"],
        },
        "recovered_task0_provenance": recovery,
        "mutable_runtime_excluded": [
            "runtime/authority", "runtime/waves/*/downloads",
            "runtime/waves/*/collected", "runtime/waves/*/orchestration",
        ],
        "cloud_credentials_included": False,
    }


def seal_host_runtime(package_root: Path, wave_root: Path) -> Mapping[str, object]:
    package_root = Path(package_root).resolve()
    value = build_manifest(package_root, wave_root)
    path = package_root / MANIFEST_RELATIVE_PATH
    if path.exists():
        existing, identity = read_json(path)
        if canonical_json_bytes(existing) != canonical_json_bytes(value):
            raise RuntimeError("host runtime differs from existing immutable manifest")
        return {**existing, "identity": identity}
    identity = write_json_exclusive(path, value)
    return {**value, "identity": identity}


def validate_host_runtime(package_root: Path) -> Mapping[str, object]:
    package_root = Path(package_root).resolve()
    manifest, identity = read_json(package_root / MANIFEST_RELATIVE_PATH)
    if (
        manifest.get("schema") != HOST_MANIFEST_SCHEMA
        or manifest.get("status") != "SEALED_NOT_LAUNCHED"
        or type(manifest.get("wave_index")) is not int
    ):
        raise RuntimeError("host runtime manifest schema/status drift")
    manifest_inventory = manifest.get("inventory")
    if not isinstance(manifest_inventory, Mapping):
        raise RuntimeError("host runtime manifest inventory is malformed")
    observed_paths = {
        path.relative_to(package_root).as_posix() for path in _host_files(package_root)
    }
    if observed_paths != set(manifest_inventory):
        raise RuntimeError("host runtime file inventory drift")
    for relative, record in manifest_inventory.items():
        if (
            not isinstance(record, Mapping)
            or set(record) != {"sha256", "size_bytes"}
            or not isinstance(record.get("sha256"), str)
            or type(record.get("size_bytes")) is not int
        ):
            raise RuntimeError(f"host runtime inventory record malformed: {relative}")
        observed = hash_file(package_root / relative, record["sha256"])
        if int(observed["size_bytes"]) != record["size_bytes"]:
            raise RuntimeError(f"host runtime file size drift: {relative}")
    manager_venv = _validate_manager_venv(package_root)
    if canonical_json_bytes(manager_venv["identity"]) != canonical_json_bytes(
        manifest.get("manager_venv_audit")
    ):
        raise RuntimeError("manager venv audit identity differs from host runtime manifest")
    test_run_receipt = _validate_test_run_receipt(package_root)
    expected_test_run_receipt = manifest.get("test_run_receipt", {})
    if (
        not isinstance(expected_test_run_receipt, Mapping)
        or type(expected_test_run_receipt.get("run_count")) is not int
        or type(expected_test_run_receipt.get("total_tests_run")) is not int
        or canonical_json_bytes(test_run_receipt["identity"])
        != canonical_json_bytes(expected_test_run_receipt.get("receipt"))
        or test_run_receipt["run_count"]
        != expected_test_run_receipt.get("run_count")
        or test_run_receipt["total_tests_run"]
        != expected_test_run_receipt.get("total_tests_run")
        or test_run_receipt["test_sources_sha256"]
        != expected_test_run_receipt.get("test_sources_sha256")
        or expected_test_run_receipt.get(
            "production_staging_untouched_by_tests"
        ) is not True
    ):
        raise RuntimeError("test run receipt differs from host runtime manifest")
    wave_root = package_root / "runtime/waves" / f"wave-{manifest['wave_index']:08d}"
    upload_staging = _validate_upload_snapshot_staging(package_root, wave_root)
    production_staging_path = _validate_test_staging_path_binding(
        test_run_receipt, upload_staging,
    )
    expected_upload = manifest.get("upload_snapshot_staging", {})
    if (
        not isinstance(expected_upload, Mapping)
        or type(expected_upload.get("file_count")) is not int
        or type(expected_upload.get("directory_count")) is not int
        or type(expected_upload.get("total_size_bytes")) is not int
        or canonical_json_bytes(upload_staging["identity"])
        != canonical_json_bytes(expected_upload.get("receipt"))
        or upload_staging["upload_snapshot_root"]
        != expected_upload.get("upload_snapshot_root")
        or upload_staging["complete_inventory_sha256"]
        != expected_upload.get("complete_inventory_sha256")
        or upload_staging["file_count"] != expected_upload.get("file_count")
        or upload_staging["directory_count"]
        != expected_upload.get("directory_count")
        or upload_staging["total_size_bytes"]
        != expected_upload.get("total_size_bytes")
    ):
        raise RuntimeError("upload snapshot staging differs from host runtime manifest")
    if expected_test_run_receipt.get(
        "production_staging_path"
    ) != production_staging_path:
        raise RuntimeError(
            "test staging path binding differs from host runtime manifest"
        )
    recovery_path = package_root / RECOVERED_TASK0_PROVENANCE_RELATIVE_PATH
    recovery_identity = None
    if recovery_path.exists():
        recovery, recovery_identity = read_json(recovery_path)
        if (
            recovery.get("schema") != "hfw_w560_recovered_task0_provenance_v3"
            or recovery.get("status") != "RECOVERED_AND_REVALIDATED_NO_RERUN"
        ):
            raise RuntimeError("recovered task0 provenance schema/status drift")
    if canonical_json_bytes(recovery_identity) != canonical_json_bytes(
        manifest.get("recovered_task0_provenance")
    ):
        raise RuntimeError("recovered task0 provenance differs from host runtime manifest")
    return {**manifest, "identity": identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path)
    parser.add_argument("--create-test-run-receipt", action="store_true")
    parser.add_argument("--test-run-spec", type=Path)
    args = parser.parse_args()
    if args.create_test_run_receipt:
        if args.wave_root is not None or args.test_run_spec is None:
            parser.error(
                "test receipt creation requires --test-run-spec and forbids --wave-root"
            )
        spec, _ = read_json(args.test_run_spec)
        runs = spec.get("runs")
        if (
            set(spec) != {
                "schema", "production_staging_path", "runs",
            }
            or spec.get("schema") != TEST_RUN_SPEC_SCHEMA
            or not isinstance(runs, list)
            or any(not isinstance(run, Mapping) for run in runs)
        ):
            parser.error(
                "test run spec requires the v2 schema, production staging path, "
                "and declarative runs"
            )
        result = create_test_run_receipt(
            args.package_root,
            production_staging_path=Path(str(spec["production_staging_path"])),
            runs=runs,
        )
        print(json.dumps({
            "status": result["status"],
            "sha256": result["identity"]["sha256"],
            "run_count": result["run_count"],
            "total_tests_run": result["total_tests_run"],
        }, sort_keys=True))
        return
    if args.wave_root is None or args.test_run_spec is not None:
        parser.error("host sealing requires --wave-root and forbids --test-run-spec")
    result = seal_host_runtime(args.package_root, args.wave_root)
    print(json.dumps({
        "status": result["status"], "sha256": result["identity"]["sha256"],
        "file_count": result["inventory_file_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
