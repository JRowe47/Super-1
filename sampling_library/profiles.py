"""Load an explicitly selected, measured layer-count profile from the library."""
import importlib.util
import json
from pathlib import Path
from . import ROOT, artifact, fingerprint, validate
INDEX=Path(__file__).with_name('depth_profiles.json')

def load_profile(entry_id, index=INDEX):
    """Return model and evidence; experimental low-depth profiles are explicit opt-ins."""
    import torch
    data=json.loads(Path(index).read_text());matches=[e for e in data['entries'] if e['id']==entry_id]
    if len(matches)!=1:raise LookupError('No unique measured depth profile')
    entry=matches[0];validate(entry)
    config=json.loads(artifact(entry['config']).read_text());source=artifact(entry['model_source'])
    # A model module's transitive dependencies are bound by its frozen source manifest.
    manifest=json.loads(artifact(entry['source_manifest']).read_text())
    for path,sha in manifest.items():artifact({'path':path,'sha256':sha})
    depth=entry['execution_depth']
    if depth not in config['depths']:raise ValueError('Depth is outside the measured protocol')
    spec=importlib.util.spec_from_file_location('library_depth_model',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):model=module.SampledCurve(config,'token',depth)
    base=torch.load(artifact(entry['base_checkpoint']),weights_only=True,map_location='cpu')
    model.base.load_state_dict(base['base'])
    if fingerprint(model.base)!=entry['base_fingerprint']:raise ValueError('Backbone mismatch')
    state=torch.load(artifact(entry['checkpoint']),weights_only=True,map_location='cpu')
    if state['execution_depth']!=depth or state['base_sha256']!=entry['base_checkpoint']['sha256']:raise ValueError('Checkpoint depth or backbone mismatch')
    model.router.load_state_dict(state['router']);model.eval()
    return model,entry

if __name__=='__main__':
    data=json.loads(INDEX.read_text())
    for entry in data['entries']:
        print(json.dumps({k:entry[k] for k in ('id','use_case','execution_depth','status','metrics')}))
