from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def load_protocol(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_source_inventory(protocol: Mapping[str, object], source_root: Path) -> None:
    for relative, expected in protocol["frozen_source"]["files"].items():
        path = source_root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"frozen source identity mismatch: {relative}")


@contextmanager
def isolated_predictive_models(source_root: Path):
    prefixes = ("representation_experiments", "third_party")
    saved = {
        name: module for name, module in tuple(sys.modules.items())
        if name == prefixes[0] or name.startswith(prefixes[0] + ".")
        or name == prefixes[1] or name.startswith(prefixes[1] + ".")
    }
    for name in saved:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("representation_experiments.predictive_models")
        expected = (source_root / "representation_experiments/predictive_models.py").resolve()
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError("predictive model import did not use the bound source")
        for name, loaded in tuple(sys.modules.items()):
            if name.startswith("representation_experiments.") or name.startswith("third_party."):
                origin = getattr(loaded, "__file__", None)
                if origin is not None and not Path(origin).resolve().is_relative_to(source_root.resolve()):
                    raise RuntimeError(f"source import escaped frozen root: {name}")
        yield module
    finally:
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == prefixes[0] or name.startswith(prefixes[0] + ".") \
                    or name == prefixes[1] or name.startswith(prefixes[1] + "."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def classify_parameter(name: str) -> str:
    if re.search(r"\.base\.(weight|bias)$", name):
        return "base_affines"
    if name.endswith(".control_left"):
        return "left_factors"
    if name.endswith(".control_right"):
        return "right_factors"
    if name.startswith("state_codecs.") and name.endswith(".weight"):
        return "codecs"
    return "frozen"


def verify_canonical_weight_binding(
    protocol: Mapping[str, object], groups: Mapping[str, list[Mapping[str, object]]]
) -> Mapping[str, object]:
    package_root = Path(__file__).resolve().parent
    binding = protocol["canonical_weight_address_binding"]
    artifacts = (
        ("manifest_relative_path", "manifest_sha256"),
        ("contract_relative_path", "contract_sha256"),
        ("validation_receipt_relative_path", "validation_receipt_sha256"),
    )
    for path_key, digest_key in artifacts:
        path = package_root / binding[path_key]
        if not path.is_file() or path.is_symlink() or file_sha256(path) != binding[digest_key]:
            raise RuntimeError(f"canonical binding artifact drift: {path_key}")
    canonical = json.loads((package_root / binding["manifest_relative_path"]).read_text())
    aliases = binding["internal_to_canonical_group"]
    for internal, canonical_name in aliases.items():
        derived = {
            (row["name"], tuple(row["shape"]), int(row["numel"]))
            for row in groups[internal]
        }
        published = {
            (row["state_dict_key"], tuple(row["shape"]), int(row["parameters"]))
            for row in canonical["allowed"]["groups"][canonical_name]["entries"]
        }
        if derived != published:
            raise RuntimeError(f"canonical tensor-address mismatch: {internal}")
    contract = json.loads((package_root / binding["contract_relative_path"]).read_text())
    if contract["immutable_parent"]["checkpoint_sha256"] != protocol["parent"]["checkpoint_sha256"]:
        raise RuntimeError("canonical binding parent identity drift")
    return {
        "manifest_sha256": binding["manifest_sha256"],
        "contract_sha256": binding["contract_sha256"],
        "validation_receipt_sha256": binding["validation_receipt_sha256"],
        "canonical_group_aliases": aliases,
    }


def derive_group_manifest(protocol: Mapping[str, object], source_root: Path) -> Mapping[str, object]:
    verify_source_inventory(protocol, source_root)
    with isolated_predictive_models(source_root) as module:
        config = json.loads((source_root / "config/bottleneck_hfw_1k_100m.json").read_text())
        model_config = config["models"][protocol["model"]["kind"]]
        width = int(model_config["minimum_width"])
        if width != int(model_config["maximum_width"]) or width != int(protocol["model"]["width"]):
            raise RuntimeError("W560 topology drift")
        with torch.device("meta"):
            model = module.build_student(
                protocol["model"]["kind"], int(config["vocab_size"]), width,
                int(config["model_max_context"]), model_config,
            )
        groups = {name: [] for name in (*protocol["search"]["independent_branches"], "frozen")}
        for name, parameter in model.named_parameters():
            group = classify_parameter(name)
            groups[group].append({
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "dtype": "float32",
            })
        counts = {name: sum(value["numel"] for value in records) for name, records in groups.items()}
        if counts["frozen"] != 85085280:
            raise RuntimeError("frozen embedding/norm parameter count drift")
        if set(value["name"] for value in groups["frozen"]) != set(protocol["model"]["frozen_parameters"]):
            raise RuntimeError("frozen parameter inventory drift")
        expected = {name: int(value) for name, value in protocol["model"]["expected_group_counts"].items()}
        actual = {name: counts[name] for name in protocol["search"]["independent_branches"]}
        if actual != expected:
            raise RuntimeError(f"group count drift: expected={expected} actual={actual}")
        if sum(counts.values()) != int(protocol["model"]["total_parameters"]):
            raise RuntimeError("total W560 parameter count drift")
        canonical_binding = verify_canonical_weight_binding(protocol, groups)
        body = {
            "schema": "hfw_w560_exact_group_manifest_v1",
            "model_kind": protocol["model"]["kind"],
            "width": width,
            "vocab_size": int(config["vocab_size"]),
            "total_parameters": sum(counts.values()),
            "group_counts": actual,
            "frozen_count": counts["frozen"],
            "groups": groups,
            "derived_from_meta_model": True,
            "learned_spline_or_fourier_group": False,
            "canonical_weight_address_binding": canonical_binding,
        }
        body["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        return body
