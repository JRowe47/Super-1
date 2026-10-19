#!/usr/bin/env python3
"""Verify publication bytes and syntax without importing research code or models."""
import ast, gzip, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parents[3]
run=Path(__file__).resolve().parent
manifest=json.loads((run/'IMPORT.json').read_text())
json_count=python_count=0
for row in manifest['files']:
 p=root/row['path']; stored=p.read_bytes()
 assert hashlib.sha256(stored).hexdigest()==row['stored_sha256'],p
 raw=gzip.decompress(stored) if row.get('compression')=='gzip' else stored
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],p
 if row['archive_member'].endswith('.json'):json.loads(raw);json_count+=1
 if row['archive_member'].endswith('.py'):ast.parse(raw);python_count+=1
 assert not p.suffix.lower() in {'.npz','.npy','.pt','.pth','.bin','.safetensors'},p
 assert 'checkpoints' not in p.relative_to(run).parts,p
assert len(manifest['archives'])==10
print(json.dumps({'status':'PASS','archives':10,'preserved_files':len(manifest['files']),'excluded_dependencies':len(manifest['excluded']),'json_parsed':json_count,'python_syntax_checked':python_count,'new_model_execution':False},indent=2))
