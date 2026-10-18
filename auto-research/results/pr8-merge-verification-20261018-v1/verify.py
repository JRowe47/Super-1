"""Read-only numerical/runtime audit before the explicitly requested PR8 merge."""
import hashlib,json,subprocess,sys
from pathlib import Path
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent; ROOT=H.parents[2]; sys.path.insert(0,str(ROOT))
from sampling_library.hierarchy_profiles import load_hierarchy_profile
from sampling_library import artifact
from sampling_library.packed_weights_archive import read_archive

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,sort_keys=True); f.write('\n')
def command(name,args):
    result=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,capture_output=True,text=True)
    with (H/(name+'.log')).open('x') as f: f.write(result.stdout+result.stderr)
    assert result.returncode==0,(name,result.stderr)
    print(name,'PASS',flush=True)
    return dict(name=name,command=[sys.executable,'-B',*args],exit_code=result.returncode,log_sha256=sha(H/(name+'.log')))

def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    commands=[]
    commands.append(command('repository-hashes',['scripts/verify_evidence.py']))
    commands.append(command('unit-tests',['-m','unittest','sampling_library.test_packed_weights_archive','sampling_library.test_inference','sampling_library.test_hierarchical_kv','sampling_library.test_hierarchy_profiles']))
    commands.append(command('library',['-m','sampling_library.test_library']))
    commands.append(command('depth-profiles',['-m','sampling_library.test_profiles']))
    commands.append(command('packed-replay',['auto-research/results/function-weight-int4-20261018-v1/verify.py','--output',str(H/'PACKED_REPLAY.json')]))
    hierarchy=[]
    for run in ['function-kv-text-descent-20261018-v1','function-kv-text-descent-512-20261018-v1']:
        directory=ROOT/'auto-research/results'/run; index=directory/'LOADABLE_LIBRARY.json'; result=json.loads((directory/'execution/RESULT.json').read_text())
        for entry in json.loads(index.read_text())['entries']:
            m,_=load_hierarchy_profile(entry['id'],index); c=json.loads(artifact(entry['config']).read_text()); a,b=c['text_regions']['confirmation']
            d=torch.tensor(list((directory/'corpus.txt').read_bytes()[a:b]),dtype=torch.long)
            ix=torch.arange(c['confirmation_examples'])[:,None]*(c['context']+1)+torch.arange(c['context'])[None,:]
            x,y=d[ix].contiguous(),d[ix+1].contiguous(); digest=hashlib.sha256(); loss=0.
            with torch.no_grad():
                for i in range(0,len(x),c['batch']):
                    z,_=m(x[i:i+c['batch']]); target=y[i:i+c['batch']]; digest.update(z.numpy().tobytes()); loss+=F.cross_entropy(z.flatten(0,1),target.flatten(),reduction='sum').item()
            expected=next(r for r in result['rows'] if r.get('variant')=='selected' and r['checkpoint']==Path(entry['pool_checkpoint']['path']).name)['confirmation']
            assert digest.hexdigest()==expected['logits_sha256'] and loss/y.numel()==expected['ce']
            hierarchy.append(dict(run=run,id=entry['id'],ce=loss/y.numel(),exact_logits=True))
        print(run,'6 exact profile replays PASS',flush=True)
    weight=ROOT/'auto-research/results/function-weight-int4-20261018-v1'; packed=[]
    for entry in json.loads((weight/'LIBRARY.json').read_text())['entries']:
        path=ROOT/entry['file']; assert sha(path)==entry['sha256']; compact=read_archive(path)
        original=weight/'execution'/f"{entry['arm']}-{entry['seed']}-selected.packed.pt"
        assert sha(original)==entry['original_packed_sha256']; old=torch.load(original,weights_only=True)
        assert old['aliases']==compact['aliases'] and old['config']==compact['config']
        for name,record in old['entries'].items():
            for k,v in record.items():
                other=compact['entries'][name][k]
                assert torch.equal(v,other) if isinstance(v,torch.Tensor) else v==other
        packed.append(dict(id=entry['id'],exact_tensors=True,sha256=sha(path)))
    charts=[]; readme=(ROOT/'README.md').read_text()
    for name in ['function_weight_kv_text_descent','function_weight_int4']:
        path=ROOT/'graphs'/(name+'.json'); receipt=json.loads(path.read_text())
        # Renderer receipts record exact input and output hashes.
        for section in ('inputs','outputs'):
            for filename,digest in receipt[section].items(): assert sha(ROOT/filename)==digest
        assert f'graphs/{name}.png' in readme
        charts.append(dict(name=name,readme_embedded=True,receipt_sha256=sha(path)))
    save(H/'VERIFICATION.json',dict(status='PASS',tested_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),commands=commands,hierarchy_profiles=hierarchy,compact_profiles=packed,charts=charts,no_training=True))
    print('PASS: all numerical replays, compact formats and README chart bindings',flush=True)
if __name__=='__main__': main()
