"""Fetch public, pinned TinyLlama and WikiText train/validation only.

Large inputs are reproducible external artifacts, never committed to Git.
The official WikiText test file is deliberately excluded.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

MODEL = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
REVISION = '59f6f375b26bde864a6ca194a9a3044570490064'
DATASET = 'Salesforce/wikitext'


def read_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch(url, path, expected_sha=None, expected_size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.partial')
        started = time.monotonic()
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size % (512 * 1024 * 1024) == 0:
                    print(f'DOWNLOAD {path.name} {size / 2**30:.2f} GiB {time.monotonic()-started:.1f}s', flush=True)
        sha = digest.hexdigest()
        if expected_sha and sha != expected_sha:
            raise ValueError(f'SHA mismatch: {path.name}')
        if expected_size and size != expected_size:
            raise ValueError(f'Size mismatch: {path.name}')
        partial.replace(path)
    else:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        sha, size = digest.hexdigest(), path.stat().st_size
        if expected_sha and sha != expected_sha:
            raise ValueError(f'Existing SHA mismatch: {path.name}')
        if expected_size and size != expected_size:
            raise ValueError(f'Existing size mismatch: {path.name}')
    print(f'VERIFIED {path.name} {size} {sha}', flush=True)
    return {'path': str(path), 'sha256': sha, 'bytes': size, 'source_url': url}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    args = ap.parse_args()
    tree = read_json(f'https://huggingface.co/api/models/{MODEL}/tree/{REVISION}')
    by_name = {entry['path']: entry for entry in tree}
    files = ['config.json', 'generation_config.json', 'special_tokens_map.json',
             'tokenizer_config.json', 'tokenizer.json', 'tokenizer.model', 'model.safetensors']
    manifest = {'model': MODEL, 'model_revision': REVISION, 'files': [], 'official_test_accessed': False}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    for name in files:
        entry = by_name[name]
        manifest['files'].append(fetch(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',
                                      args.inputs / 'model' / name,
                                      entry.get('lfs', {}).get('oid'), entry['size']))
        args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    dataset_revision = read_json(f'https://huggingface.co/api/datasets/{DATASET}')['sha']
    manifest['dataset'], manifest['dataset_revision'] = DATASET, dataset_revision
    tree = read_json(f'https://huggingface.co/api/datasets/{DATASET}/tree/{dataset_revision}/wikitext-2-raw-v1')
    by_name = {entry['path']: entry for entry in tree}
    for split in ['train', 'validation']:
        name = f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
        entry = by_name[name]
        manifest['files'].append(fetch(f'https://huggingface.co/datasets/{DATASET}/resolve/{dataset_revision}/{name}',
                                      args.inputs / f'{split}.parquet', entry['lfs']['oid'], entry['size']))
        args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
