"""Download a pinned public reference checkpoint with no training or mutation."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'reference_model'
REPO = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
FILES = ['config.json', 'generation_config.json', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'model.safetensors']

def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.load(r)

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    existing_manifest = ROOT/'model_manifest.json'
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text())
        if existing['revision'] != REVISION:
            raise RuntimeError('Existing model belongs to another revision; use a new model directory.')
        for filename, info in existing['files'].items():
            path = DEST/filename
            if path.exists() and hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != info['sha256']:
                raise RuntimeError(f'Existing file checksum mismatch: {filename}')
    metadata = get_json(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}')
    revision = metadata['sha']
    assert revision == REVISION
    files = {x['rfilename'] for x in metadata['siblings']}
    manifest = {'repository': REPO, 'revision': revision, 'files': {}}
    for filename in FILES:
        if filename not in files:
            continue
        path = DEST / filename
        if not path.exists():
            print('Downloading', filename, flush=True)
            url = f'https://huggingface.co/{REPO}/resolve/{revision}/{filename}'
            with urllib.request.urlopen(url, timeout=180) as r, path.with_suffix(path.suffix+'.part').open('wb') as w:
                while chunk := r.read(8*1024*1024):
                    w.write(chunk)
            path.with_suffix(path.suffix+'.part').replace(path)
        h = hashlib.sha256()
        with path.open('rb') as f:
            while chunk := f.read(8*1024*1024):
                h.update(chunk)
        manifest['files'][filename] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    (ROOT/'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'revision': revision, 'bytes': sum(x['bytes'] for x in manifest['files'].values())}), flush=True)

if __name__ == '__main__':
    main()
