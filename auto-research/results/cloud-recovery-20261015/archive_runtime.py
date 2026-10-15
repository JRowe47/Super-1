"""Archive allowlisted recovered runtime files without executing them."""
import email.parser
import hashlib
import json
import pathlib
import tarfile

root = pathlib.Path('/mnt/recovery-training2')
venv = root / 'home/rohanarun/funcweights/representation-venv'
output = pathlib.Path('/tmp/hierarchy-runtime-recovered.tar.gz')
packages = []
for metadata in sorted(venv.glob('lib/python*/site-packages/*.dist-info/METADATA')):
    raw = metadata.read_bytes()
    msg = email.parser.BytesParser().parsebytes(raw)
    packages.append({'name': msg['Name'], 'version': msg['Version'],
                     'metadata_sha256': hashlib.sha256(raw).hexdigest()})

def keep(info):
    parts = pathlib.PurePosixPath(info.name).parts
    if '__pycache__' in parts or info.name.endswith('.pyc') or parts[-1] == 'direct_url.json':
        return None
    return info

with tarfile.open(output, 'w:gz', compresslevel=1) as archive:
    for path in [venv, root / 'usr/bin/python3.10', root / 'usr/lib/python3.10', root / 'etc/os-release']:
        if path.exists():
            archive.add(path, arcname=str(path.relative_to(root)), filter=keep)
manifest = {'scope': 'Recovered training2 Python environment, interpreter and stdlib; not a full OS/container or verified historical executable replay',
            'excluded': ['__pycache__', '*.pyc', 'direct_url.json'],
            'packages': packages, 'archive_bytes': output.stat().st_size,
            'archive_sha256': hashlib.file_digest(output.open('rb'), 'sha256').hexdigest()}
parts = []
with output.open('rb') as source:
    index = 1
    while True:
        block = source.read(512 * 1024 * 1024)
        if not block:
            break
        path = pathlib.Path(str(output) + '.part-' + str(index).zfill(3))
        path.write_bytes(block)
        parts.append({'name': path.name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
        index += 1
manifest['parts'] = parts
pathlib.Path('/tmp/hierarchy-runtime-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest), flush=True)
