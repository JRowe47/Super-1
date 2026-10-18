from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Mapping

from .manifest import canonical_json_bytes, file_sha256


def validate_no_training_source_gate(package_root: Path) -> Mapping[str, object]:
    files = sorted(
        path for path in package_root.glob("*.py")
        if path.name != "static_gate.py"
    )
    inspected = {}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"backward", "grad"}:
                    raise RuntimeError(f"autograd call is forbidden: {path.name}:{node.lineno}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                if any(name == "torch.optim" or name.startswith("torch.optim.") for name in names):
                    raise RuntimeError(f"optimizer import is forbidden: {path.name}:{node.lineno}")
        inspected[path.name] = file_sha256(path)
    body = {
        "schema": "hfw_no_training_source_gate_v1",
        "status": "PASS_NO_AUTOGRAD_NO_OPTIMIZER_SOURCE",
        "inspected_files": inspected,
        "optimizer_imports": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
    }
    body["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body

