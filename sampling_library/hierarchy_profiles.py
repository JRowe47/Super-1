"""Explicitly load measured experimental hierarchy profiles without promotion."""
import importlib.util
import json
from pathlib import Path
from . import artifact


def load_hierarchy_profile(entry_id, index):
    """Load hashed backbone and trained pooling weights from a research library.

    This explicit loader does not select a use case or claim that a profile has
    passed long-context quality or total-memory gates. Returned evidence carries
    the measured context and limitations. A new request must start a new cache.
    """
    import torch
    from .hierarchical_kv import HierarchicalKV
    from .inference import CachedSampledCurve

    data = json.loads(Path(index).read_text())
    entries = [entry for entry in data['entries'] if entry['id'] == entry_id]
    if len(entries) != 1:
        raise LookupError('No unique measured hierarchy profile')
    entry = entries[0]
    manifest = json.loads(artifact(entry['source_manifest']).read_text())
    for path, digest in manifest.items():
        artifact({'path': path, 'sha256': digest})
    config = json.loads(artifact(entry['config']).read_text())
    measured = json.loads(artifact(entry['selection_record']).read_text())
    if (measured['sha256'] != entry['pool_checkpoint']['sha256']
            or measured['base_sha256'] != entry['base_checkpoint']['sha256']
            or config['candidates'].get(measured['arm']) != entry['capacity']):
        raise ValueError('Pooling weights, backbone or capacity differ from the measured pair')
    if entry['context'] != config['context'] or entry['execution_depth'] != config['layers']:
        raise ValueError('Profile context or execution depth differs from its protocol')
    if entry['capacity'] not in config['candidates'].values():
        raise ValueError('Hierarchy capacity was not measured in the protocol')
    source = artifact(entry['model_source'])
    if manifest.get(entry['model_source']['path']) != entry['model_source']['sha256']:
        raise ValueError('Model source is not bound to the frozen manifest')
    spec = importlib.util.spec_from_file_location('measured_hierarchy_model', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        model = module.SampledCurve(config, 'token', entry['execution_depth'])
    model.load_state_dict(torch.load(artifact(entry['base_checkpoint']), weights_only=True, map_location='cpu')['model'])
    runtime = CachedSampledCurve(model, lambda i, h, d: HierarchicalKV(h, d, **entry['capacity']))
    runtime.memory.load_state_dict(torch.load(artifact(entry['pool_checkpoint']), weights_only=True, map_location='cpu')['model'])
    return runtime.eval(), entry
