"""Verify every release file in SHA256SUMS.json, without network or training."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
for name,expected in manifest.items():
    p=ROOT/name;assert p.is_file(),name
    with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected,name
print('Verified',len(manifest),'files.')
