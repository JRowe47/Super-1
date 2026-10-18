from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import hash_file_same_fd, write_exclusive_json


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def derive_identity(tokenizer_root: Path, model_config_path: Path, output: Path) -> dict[str, object]:
    from transformers import AutoTokenizer

    tokenizer_root = Path(tokenizer_root)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), local_files_only=True, trust_remote_code=False,
        use_fast=True,
    )
    mapping = tokenizer.get_vocab()
    model_config = json.loads(Path(model_config_path).read_text())
    identity = {
        "schema": "hfw_w560_tokenizer_runtime_identity_v1",
        "status": "PASS_DERIVED_FROM_EXACT_LOCAL_AUTOTOKENIZER",
        "base_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_length": len(tokenizer),
        "mapping_count": len(mapping),
        "maximum_token_id": max(mapping.values()),
        "minimum_token_id": min(mapping.values()),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "model_vocab_size": int(model_config["vocab_size"]),
        "all_token_ids_fit_model_vocabulary": max(mapping.values()) < int(model_config["vocab_size"]),
        "files": {name: hash_file_same_fd(tokenizer_root / name) for name in TOKENIZER_FILES},
        "tokenizer_encode_calls": 0,
        "model_forward_calls": 0,
        "network_access": False,
    }
    if identity["bos_token"] is not None or identity["bos_token_id"] is not None:
        raise RuntimeError("unexpected BOS token in exact local tokenizer")
    if identity["eos_token_id"] is None:
        raise RuntimeError("exact local tokenizer has no EOS token")
    artifact = write_exclusive_json(output, identity)
    return {**identity, "artifact": artifact}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(derive_identity(args.tokenizer_root, args.model_config, args.output))


if __name__ == "__main__":
    main()
