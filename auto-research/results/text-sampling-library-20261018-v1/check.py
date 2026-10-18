"""Independent checkpoint scoring and library reload checks; no optimizer execution."""
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
import run

def main():
    run.setup();h=run.H;root=run.ROOT;out=h/'execution'
    result=json.loads((out/'RESULT.json').read_text());freeze=json.loads((out/'FREEZE.json').read_text())
    assert all(run.sha(root/k)==v for k,v in json.loads((h/'SOURCE_MANIFEST.json').read_text()).items())
    raw=(h.parent/'adaptive-depth-shakespeare-20261018-v1/corpus.txt').read_bytes()
    start,end=run.C['text_regions']['confirmation'];length=run.C['context'];n=run.C['confirmation_examples'];batch=run.C['batch']
    chunks=torch.tensor([list(raw[start+i*(length+1):start+(i+1)*(length+1)]) for i in range(n)])
    assert start+n*(length+1)==end
    checked=[]
    for row in result['rows']:
        ref=next(r for r in run.references() if r['seed']==row['seed']);assert run.sha(root/row['checkpoint'])==row['sha256']
        m=run.model(ref,root/row['checkpoint']);total=0.;correct=0;m.eval()
        with torch.no_grad():
            for i in range(0,n,batch):
                x=chunks[i:i+batch,:-1].contiguous();y=chunks[i:i+batch,1:].contiguous();logits=m(x)
                total+=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum').item();correct+=(logits.argmax(-1)==y).sum().item()
        assert total/(n*length)==row['confirmation']['ce'] and correct/(n*length)==row['confirmation']['accuracy']
        checked.append({'seed':row['seed'],'arm':row['arm'],'ce':total/(n*length)})
    for seed in run.C['seeds']:
        for rnd in range(1,run.C['rounds']+1):
            pair=[r for r in freeze['history'] if r['seed']==seed and r['round']==rnd]
            assert len({r['batch_sha256'] for r in pair})==1
        incumbent=next(r for r in freeze['choices'] if r['seed']==seed and r['arm']=='incumbent')
        for arm in run.C['arms']:
            candidates=[incumbent]+[r for r in freeze['history'] if r['seed']==seed and r['arm']==arm]
            chosen=next(r for r in freeze['choices'] if r['seed']==seed and r['arm']==arm)
            assert chosen['selection']['ce']==min(r['selection']['ce'] for r in candidates)
    run.save(out/'VERIFICATION.json',{'status':'PASS','independent_replays':checked,'matched_batches':True,'selection_minima':True,'source_hashes':True})
    print('PASS:',len(checked),'independent confirmation replays')
if __name__=='__main__':main()
