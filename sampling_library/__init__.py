"""Evidence-backed sampler weights, selected by explicit workload and exact backbone."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = Path(__file__).with_name('index.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(record):
    path = (ROOT / record['path']).resolve()
    if ROOT not in path.parents or digest(path) != record['sha256']:
        raise ValueError('Artifact escaped repository or failed hash verification')
    return path


def fingerprint(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def validate(entry):
    for field in ('id', 'use_case', 'base_fingerprint', 'tokenizer', 'context',
                  'status', 'active', 'seed', 'training', 'metrics', 'evidence'):
        if field not in entry:
            raise ValueError('Missing field: ' + field)
    if entry['status'] not in ('research_reference', 'candidate', 'confirmed_improvement'):
        raise ValueError('Unknown evidence status')
    if entry['active'] and entry['status'] == 'candidate':
        raise ValueError('Unconfirmed candidate cannot replace reference')
    for key in ('checkpoint', 'base_checkpoint'):
        artifact(entry[key])
    if not entry['evidence']:
        raise ValueError('Evidence is required')
    for record in entry['evidence']:
        artifact(record)
    # JSON serialization also rejects NaN/Infinity in measurements.
    json.dumps(entry, allow_nan=False)


def read(index=DEFAULT_INDEX):
    data = json.loads(Path(index).read_text())
    if data['schema_version'] != 1:
        raise ValueError('Unsupported library schema')
    return data


def register(entry, index=DEFAULT_INDEX):
    """Append an immutable entry; caller supplies a benchmark-backed promotion decision.

    This is a single-writer research index. Never concurrently mutate this file.
    """
    validate(entry)
    index = Path(index)
    data = read(index) if index.exists() else {'schema_version': 1, 'entries': []}
    if any(e['id'] == entry['id'] for e in data['entries']):
        raise ValueError('Entry ID already exists')
    if entry['active']:
        for existing in data['entries']:
            if all(existing[k] == entry[k] for k in ('use_case', 'base_fingerprint', 'tokenizer', 'context')):
                existing['active'] = False
    data['entries'].append(entry)
    with tempfile.NamedTemporaryFile(mode='w', dir=index.parent, delete=False) as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        temporary = f.name
    os.replace(temporary, index)


def resolve(use_case, base_fingerprint, tokenizer, context, index=DEFAULT_INDEX):
    matches = [e for e in read(index)['entries'] if e['active'] and
               (e['use_case'], e['base_fingerprint'], e['tokenizer'], e['context']) ==
               (use_case, base_fingerprint, tokenizer, context)]
    if len(matches) != 1:
        raise LookupError('No unique validated sampler for this workload/model/tokenizer/context')
    validate(matches[0])
    return matches[0]


def load_router(model, use_case, tokenizer, context, index=DEFAULT_INDEX):
    """Load an exact-compatible active sampler; unknown workloads fail explicitly."""
    import torch
    if getattr(model, 'mode', None) != 'token':
        raise ValueError('Library entries require token-conditioned SampledCurve')
    entry = resolve(use_case, fingerprint(model.base), tokenizer, context, index)
    state = torch.load(artifact(entry['checkpoint']), map_location='cpu', weights_only=True)
    if digest(artifact(entry['base_checkpoint'])) != state['base_sha256']:
        raise ValueError('Sampler checkpoint is bound to a different backbone')
    model.router.load_state_dict(state['router'], strict=True)
    return entry
