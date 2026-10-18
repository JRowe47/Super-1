"""Losslessly compact all selected exports and retain exact inference receipts."""
import argparse,hashlib,json,resource,subprocess,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent; ROOT=H.parents[2]; E=H/'execution'
sys.path.insert(0,str(ROOT))
from sampling_library.packed_weights import PackedModel
from sampling_library.packed_weights_archive import write_archive,read_archive

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    with p.open('x') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n')
def main():
    c=json.loads((H/'CONFIG.json').read_text()); rows=json.loads((E/'RESULT.json').read_text())['rows']
    d=E/'compact'; d.mkdir(exist_ok=False)
    sources=[ROOT/'sampling_library/packed_weights_archive.py',H/'package.py',H/'PACKAGING_PROTOCOL.md']
    save(d/'START.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    a,b=c['regions']['confirmation']; data=torch.tensor(list((ROOT/c['parent_run']/'corpus.txt').read_bytes()[a:b]),dtype=torch.long).view(c['confirmation_examples'],c['context']+1)
    x,y=data[:,:-1].contiguous(),data[:,1:].contiguous(); result=[]
    for r in rows:
        if r['variant']!='selected': continue
        old=E/r['packed_file']; assert sha(old)==r['packed_sha256']
        original=torch.load(old,weights_only=True); path=d/old.name.replace('.packed.pt','.s1pw')
        write_archive(path,original); restored=read_archive(path)
        for name,entry in original['entries'].items():
            for k,v in entry.items():
                other=restored['entries'][name][k]
                if isinstance(v,torch.Tensor): assert torch.equal(v,other)
                else: assert v==other
        assert original['aliases']==restored['aliases'] and original['config']==restored['config']
        m=PackedModel(restored); total=0.; digest=hashlib.sha256()
        with torch.inference_mode():
            for i in range(0,len(x),c['batch']):
                z=m(x[i:i+c['batch']]); yy=y[i:i+c['batch']]
                total+=F.cross_entropy(z.flatten(0,1),yy.flatten(),reduction='sum').item(); digest.update(z.numpy().tobytes())
        assert total/y.numel()==r['confirmation']['ce'] and digest.hexdigest()==r['confirmation']['logits_sha256']
        result.append(dict(arm=r['arm'],seed=r['seed'],source=old.name,source_sha256=sha(old),file=path.name,sha256=sha(path),file_bytes=path.stat().st_size,payload_bytes=m.parameter_bytes,confirmation_ce=total/y.numel(),exact_logits=True,exact_tensors=True))
    ratios={}
    for arm in ('curve_b4','curve_be4','nano_q4'):
        control='nano_fp16' if arm.startswith('nano') else 'curve_fp16'
        ratios[arm]=[next(r['file_bytes'] for r in result if r['arm']==control and r['seed']==s)/next(r['file_bytes'] for r in result if r['arm']==arm and r['seed']==s) for s in c['seeds']]
    save(d/'RESULT.json',dict(status='COMPLETED',rows=result,compact_file_reductions=ratios,original_gate_unchanged=True,scope='Lossless packaging follow-up after quality exposure; no new quality evidence'))
    print(json.dumps(dict(files=len(result),ratios=ratios),indent=2))

def memory(path,destination):
    start=time.perf_counter(); m=PackedModel(read_archive(path)); c=m.c
    raw=(ROOT/c['parent_run']/'corpus.txt').read_bytes()[:c['context']]
    with torch.inference_mode(): z=m(torch.tensor(list(raw),dtype=torch.long)[None])
    assert torch.isfinite(z).all()
    factor=1 if sys.platform=='darwin' else 1024
    save(destination,dict(file=path.name,sha256=sha(path),payload_bytes=m.parameter_bytes,file_bytes=path.stat().st_size,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor,max_decoded_weight_bytes=m.max_decoded_values*4,load_and_forward_seconds=time.perf_counter()-start,scope='Fresh process, compact export only,512-token full attention; not latency benchmark'))

if __name__=='__main__':
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser(); p.add_argument('--memory',type=Path); p.add_argument('--destination',type=Path); a=p.parse_args()
    if a.memory: memory(a.memory,a.destination)
    else: main()
