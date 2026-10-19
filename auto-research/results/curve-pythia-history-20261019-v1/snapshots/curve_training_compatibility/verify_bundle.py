"""Verify every file recorded in the release checksum manifest; no network or training."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
for name,expected in manifest.items():
    path=ROOT/name
    assert path.is_file(),'Missing: '+name
    with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected,'Hash mismatch: '+name
print('Verified',len(manifest),'files.')
