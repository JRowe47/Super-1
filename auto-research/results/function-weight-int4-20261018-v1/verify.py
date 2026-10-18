"""Independently reload packed exports and audit selection without training."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent; ROOT=H.parents[2]; E=H/'execution'
sys.path.insert(0,str(ROOT))
from sampling_library.packed_weights import PackedModel, fake_parameters

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main(destination):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    c=json.loads((H/'CONFIG.json').read_text()); freeze=json.loads((E/'FREEZE.json').read_text())
    result=json.loads((E/'RESULT.json').read_text()); checks=[]; depth_checks=[]
    spec=importlib.util.spec_from_file_location('curve_verify',ROOT/c['model_source'])
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for name,digest in json.loads((H/'SOURCE_MANIFEST.json').read_text()).items(): assert sha(ROOT/name)==digest,name
    assert json.loads((E/'TERMINAL.json').read_text())['status']=='COMPLETED'
    assert len(freeze['history'])==len(c['seeds'])*len(c['arms'])*(c['rounds']+1)
    for r in freeze['history']: assert sha(E/r['checkpoint'])==r['sha256']
    for selected in freeze['selected']:
        group=[r for r in freeze['history'] if r['arm']==selected['arm'] and r['seed']==selected['seed']]
        assert min(group,key=lambda r:r['selection']['ce'])==selected
    for seed in c['seeds']:
        for rnd in range(c['rounds']+1):
            assert len({r['batch_sha256'] for r in freeze['history'] if r['seed']==seed and r['round']==rnd})==1
    a,b=c['regions']['confirmation']; raw=(ROOT/c['parent_run']/'corpus.txt').read_bytes()[a:b]
    examples=torch.tensor(list(raw),dtype=torch.long).view(c['confirmation_examples'],c['context']+1)
    x,y=examples[:,:-1].contiguous(),examples[:,1:].contiguous()
    for r in result['rows']:
        if 'packed_file' not in r: continue
        path=E/r['packed_file']; assert sha(path)==r['packed_sha256']
        model=PackedModel(torch.load(path,weights_only=True)); loss=0.; digest=hashlib.sha256()
        with torch.inference_mode():
            for i in range(0,len(x),c['batch']):
                z=model(x[i:i+c['batch']]); target=y[i:i+c['batch']]
                loss+=F.cross_entropy(z.flatten(0,1),target.flatten(),reduction='sum').item(); digest.update(z.numpy().tobytes())
        assert loss/y.numel()==r['confirmation']['ce']
        assert digest.hexdigest()==r['confirmation']['logits_sha256']
        assert path.stat().st_size==r['file_bytes'] and model.parameter_bytes==r['payload_bytes']
        assert not any(isinstance(v,torch.nn.Module) for v in vars(model).values())
        checks.append(dict(file=path.name,sha256=sha(path),exact_logits=True,ce=loss/y.numel()))
        if r['arm'].startswith('curve') and r['variant']=='selected':
            dense=mod.SampledCurve(c,'token',c['layers']).eval()
            dense.load_state_dict(torch.load(E/r['checkpoint'],weights_only=True)['model'])
            # Mechanics only; no depth selection or lower-depth quality claim.
            sample=x[:1,:37]
            for depth in (2,3,4):
                dense.depth=dense.execution_depth=depth
                with torch.no_grad():
                    expected=torch.func.functional_call(dense,fake_parameters(dense,r['arm'],c['group']),(sample,),{'positions':True},tie_weights=True)
                    actual=model(sample,positions=True,depth=depth)
                torch.testing.assert_close(actual[0],expected[0],atol=3e-5,rtol=3e-4)
                torch.testing.assert_close(actual[1],expected[1],atol=2e-6,rtol=2e-5)
                depth_checks.append(dict(file=path.name,depth=depth,max_logit_error=(actual[0]-expected[0]).abs().max().item()))
    with destination.open('x') as f: json.dump(dict(status='PASS',source_manifest=True,matched_targets=True,selection_recomputed=True,checkpoint_hashes=len(freeze['history']),packed_replays=checks,trained_depth_checks=depth_checks),f,indent=2); f.write('\n')
    print(f'PASS: {len(checks)} packed exports independently replayed; all checkpoints and selection audited')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); main(p.parse_args().output)
