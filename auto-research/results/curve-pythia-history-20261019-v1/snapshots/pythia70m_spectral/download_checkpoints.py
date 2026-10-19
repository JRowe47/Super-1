"""Fetch immutable public Pythia checkpoint revisions; no training."""
import concurrent.futures, hashlib, json, pathlib, time, urllib.request
ROOT = pathlib.Path(__file__).resolve().parent
REPO = 'EleutherAI/pythia-70m'
STEPS = [0, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
MANIFEST = ROOT / 'checkpoint_manifest.json'
PINNED = {r['step']:r for r in json.loads(MANIFEST.read_text())} if MANIFEST.exists() else {}

def get(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()

def one(step):
    d = ROOT / 'checkpoints' / f'step{step}'
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / 'source.json'
    if meta_path.exists() and (d / 'model.safetensors').exists():
        cached = json.loads(meta_path.read_text())
        if step in PINNED:
            assert cached['commit']==PINNED[step]['commit'], 'Cached checkpoint revision differs from pinned manifest'
        return cached
    if step in PINNED:
        sha = PINNED[step]['commit']
    else:
        meta = json.loads(get(f'https://huggingface.co/api/models/{REPO}/revision/step{step}'))
        sha = meta['sha']
    prefix = f'https://huggingface.co/{REPO}/resolve/{sha}/'
    config = get(prefix + 'config.json')
    (d / 'config.json').write_bytes(config)
    dest = d / 'model.safetensors'
    t = time.time()
    h = hashlib.sha256()
    with urllib.request.urlopen(prefix + 'model.safetensors', timeout=120) as r, open(str(dest)+'.part', 'wb') as f:
        while chunk := r.read(2**20):
            h.update(chunk)
            f.write(chunk)
    if step in PINNED:
        assert h.hexdigest()==PINNED[step]['sha256'], 'Downloaded bytes fail pinned SHA-256 check'
    pathlib.Path(str(dest)+'.part').replace(dest)
    info = {'repo': REPO, 'revision': f'step{step}', 'step':step, 'commit':sha,
            'sha256':h.hexdigest(), 'size_bytes':dest.stat().st_size,
            'elapsed_seconds':time.time()-t}
    meta_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return info

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(one, STEPS))
    (ROOT / 'checkpoint_manifest.json').write_text(json.dumps(results, indent=2))
    target = ROOT / 'checkpoints' / 'tokenizer'
    target.mkdir(exist_ok=True)
    sha = results[-1]['commit']
    for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        (target/name).write_bytes(get(f'https://huggingface.co/{REPO}/resolve/{sha}/{name}'))
    print('DOWNLOAD COMPLETE', flush=True)

if __name__ == '__main__':
    main()
