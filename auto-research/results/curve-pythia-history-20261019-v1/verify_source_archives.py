#!/usr/bin/env python3
"""Verify original ZIP identities and historical member manifests, without models.

Usage: python verify_source_archives.py ARCHIVE_DIRECTORY
Reads the ten original ZIPs named in IMPORT.json, not the filtered snapshots.
Writes JSON to standard output only. No archive extraction or code execution.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_directory", type=Path)
    args = parser.parse_args()
    imported = json.loads((Path(__file__).resolve().parent / "IMPORT.json").read_text())
    records = []
    for item in imported["archives"]:
        path = args.archive_directory / item["name"]
        data = path.read_bytes()
        identity = sha(data) == item["sha256"] and len(data) == item["bytes"]
        assert identity, "Archive identity mismatch: " + item["name"]
        with zipfile.ZipFile(path) as archive:
            files = [n for n in archive.namelist() if not n.endswith("/")]
            root = files[0].split("/")[0]
            manifest_name = next((n for n in files if Path(n).name in ("SHA256SUMS", "SHA256SUMS.json")), None)
            checked, missing, mismatches = 0, [], []
            if manifest_name:
                text = archive.read(manifest_name).decode("utf-8")
                if manifest_name.endswith(".json"):
                    manifest = json.loads(text)
                else:
                    manifest = {}
                    for line in text.splitlines():
                        if line.strip():
                            digest, name = line.split(maxsplit=1)
                            manifest[name.lstrip("*")] = digest
                for name, digest in manifest.items():
                    name = name.removeprefix("./")
                    member = name if name.startswith(root + "/") else root + "/" + name
                    if member not in files:
                        missing.append(name)
                        continue
                    if sha(archive.read(member)) != digest:
                        mismatches.append(name)
                    checked += 1
            records.append({
                "archive": item["name"], "stage": item["stage"], "archive_root": root,
                "archive_bytes": len(data), "archive_sha256": sha(data),
                "archive_identity_matches_import": identity, "archive_files": len(files),
                "historical_manifest": manifest_name,
                "member_sha256_checks": checked, "missing": missing, "mismatched": mismatches,
                "status": "PASS" if manifest_name and not missing and not mismatches else (
                    "NO_HISTORICAL_MANIFEST" if not manifest_name else "FAIL"),
            })
    result = {
        "scope": "Source archive byte identity and historical member SHA-256 integrity only; no model execution or scientific-result validation.",
        "source_archives": len(records),
        "historical_manifests": sum(r["historical_manifest"] is not None for r in records),
        "member_sha256_checks": sum(r["member_sha256_checks"] for r in records),
        "status": "PASS" if all(r["status"] != "FAIL" for r in records) else "FAIL",
        "archives": records,
    }
    print(json.dumps(result, indent=2))
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
