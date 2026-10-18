from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Mapping

from .common import (
    atomic_current_pointer, canonical_json_bytes, exclusive_lock, hash_file,
    read_json, sha256_bytes, write_exclusive, write_json_exclusive,
)


HEAD_SCHEMA = "hfw_w560_distributed_authoritative_head_v1"
CONTRACT_SCHEMA = "hfw_w560_distributed_append_only_lineage_contract_v1"


class DistributedLedger:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def lock_path(self) -> Path:
        return self.root / "authority.lock"

    def lock(self):
        return exclusive_lock(self.lock_path)

    def initialize(
        self, *, contract: Mapping[str, object], initial_head: Mapping[str, object],
    ) -> Mapping[str, object]:
        if self.root.exists():
            raise FileExistsError(f"authoritative lineage already exists: {self.root}")
        self.root.mkdir(parents=True)
        for name in (
            "heads", "campaigns", "collections", "shortlists", "decisions",
            "promotion_intents", "deltas", "replacement_group_flats",
            "promoted_checkpoints", "failures",
        ):
            (self.root / name).mkdir()
        write_exclusive(self.lock_path, b"", mode=0o600)
        contract_artifact = write_json_exclusive(
            self.root / "contract.json", {"schema": CONTRACT_SCHEMA, **contract},
        )
        head = {
            "schema": HEAD_SCHEMA,
            "authority_generation": 0,
            "parent_head_record_sha256": None,
            "contract_sha256": contract_artifact["sha256"],
            **initial_head,
        }
        return self.append_head(head)

    def contract(self) -> tuple[Mapping[str, object], Mapping[str, object]]:
        value, identity = read_json(self.root / "contract.json")
        if value.get("schema") != CONTRACT_SCHEMA:
            raise RuntimeError("distributed lineage contract schema drift")
        return value, identity

    def heads(self, *, validate_current_pointer: bool = True) -> list[Mapping[str, object]]:
        contract, contract_identity = self.contract()
        rows = []
        for path in sorted((self.root / "heads").glob("*.head.json")):
            value, identity = read_json(path)
            match = re.fullmatch(r"(\d{8})-([0-9a-f]{64})\.head\.json", path.name)
            if match is None or identity["sha256"] != match.group(2):
                raise RuntimeError("head filename/content SHA drift")
            if value.get("schema") != HEAD_SCHEMA:
                raise RuntimeError("distributed head schema drift")
            if int(value["authority_generation"]) != int(match.group(1)):
                raise RuntimeError("distributed head generation drift")
            if value.get("contract_sha256") != contract_identity["sha256"]:
                raise RuntimeError("distributed head contract drift")
            checkpoint = value["checkpoint"]
            observed = hash_file(Path(str(checkpoint["path"])), str(checkpoint["sha256"]))
            if int(observed["size_bytes"]) != int(checkpoint["size_bytes"]):
                raise RuntimeError("distributed head checkpoint size drift")
            rows.append({**value, "record_sha256": identity["sha256"], "record_path": identity["path"]})
        if not rows:
            raise RuntimeError("distributed authoritative lineage has no head")
        for index, row in enumerate(rows):
            if int(row["authority_generation"]) != index:
                raise RuntimeError("distributed head generations are not contiguous")
            if index:
                parent = rows[index - 1]
                if row["parent_head_record_sha256"] != parent["record_sha256"]:
                    raise RuntimeError("distributed head record chain break")
                if row["parent_checkpoint_sha256"] != parent["checkpoint"]["sha256"]:
                    raise RuntimeError("distributed checkpoint chain break")
        if validate_current_pointer:
            pointer, _ = read_json(self.root / "CURRENT.json")
            if pointer.get("record_sha256") != rows[-1]["record_sha256"]:
                raise RuntimeError("CURRENT pointer does not name terminal head")
        return rows

    def current_head(self) -> Mapping[str, object]:
        return self.heads()[-1]

    def recover_current_pointer_locked(self) -> Mapping[str, object]:
        """Repair only the derived CURRENT pointer after an interrupted head append.

        The append-only head is authoritative; CURRENT is a replaceable cache. Callers
        must hold ``self.lock()`` so a promoter cannot append concurrently.
        """
        terminal = self.heads(validate_current_pointer=False)[-1]
        atomic_current_pointer(self.root / "CURRENT.json", {
            "schema": "hfw_w560_distributed_current_head_pointer_v1",
            "authority_generation": int(terminal["authority_generation"]),
            "record_sha256": terminal["record_sha256"],
            "record_path": terminal["record_path"],
            "recovered_from_append_only_terminal_head": True,
        })
        return self.current_head()

    def append_head(self, head: Mapping[str, object]) -> Mapping[str, object]:
        payload = canonical_json_bytes(head)
        digest = sha256_bytes(payload)
        generation = int(head["authority_generation"])
        path = self.root / "heads" / f"{generation:08d}-{digest}.head.json"
        identity = write_exclusive(path, payload)
        enriched = {**head, "record_sha256": identity["sha256"], "record_path": identity["path"]}
        atomic_current_pointer(self.root / "CURRENT.json", {
            "schema": "hfw_w560_distributed_current_head_pointer_v1",
            "authority_generation": generation,
            "record_sha256": identity["sha256"],
            "record_path": identity["path"],
        })
        return enriched

    def next_wave_index(self) -> int:
        paths = sorted((self.root / "campaigns").glob("*.campaign.json"))
        indices = []
        for path in paths:
            value, _ = read_json(path)
            indices.append(int(value["wave_index"]))
        if indices != list(range(len(indices))):
            raise RuntimeError("campaign wave indices are not contiguous")
        return len(indices)

    def append_named_record(
        self, directory: str, suffix: str, value: Mapping[str, object],
    ) -> Mapping[str, object]:
        payload = canonical_json_bytes(value)
        digest = sha256_bytes(payload)
        path = self.root / directory / f"{digest}{suffix}"
        if path.exists():
            identity = hash_file(path, digest)
            if int(identity["size_bytes"]) != len(payload):
                raise RuntimeError("content-addressed lineage record size collision")
            return identity
        return write_exclusive(path, payload)
