"""Recompute confirmation using independently constructed windows and frozen weights."""
import json,math
import torch
from torch.nn import functional as F
import run
from model import build
run.setup();out=run.H/'execution';C=run.C;r=json.loads((out/'RESULT.json').read_text());freeze=json.loads((out/'FREEZE.json').read_text())
assert all(run.sha(run.ROOT/k)==v for k,v in json.loads((run.H/'SOURCE_MANIFEST.json').read_text()).items())
raw=(run.H/'corpus.txt').read_bytes();a,b=C['text_regions']['confirmation'];n=C['confirmation_examples'];length=C['context'];assert a+n*(length+1)==b
chunks=torch.tensor([list(raw[a+i*(length+1):a+(i+1)*(length+1)]) for i in range(n)]);rows=[]
for row in r['rows']:
    assert run.sha(out/row['inference_checkpoint'])==row['inference_sha256'];m=build(C,row['arm'],row['seed']);m.load_state_dict(torch.load(out/row['inference_checkpoint'],weights_only=True)['model']);m.eval();total=0.;correct=0
    with torch.no_grad():
        for offset in range(0,n,C['batch']):
            x=chunks[offset:offset+C['batch'],:-1].contiguous();y=chunks[offset:offset+C['batch'],1:].contiguous();pred,_=m(x);total+=F.cross_entropy(pred.flatten(0,1),y.flatten(),reduction='sum').item();correct+=(pred.argmax(-1)==y).sum().item()
        x=chunks[:1,:-1].contiguous();full,_=m(x);cache=None;parts=[]
        for offset in range(0,length,16):
            pred,cache=m(x[:,offset:offset+16],cache);parts.append(pred)
        difference=float((full-torch.cat(parts,1)).abs().max());assert torch.allclose(full,torch.cat(parts,1),atol=C['forward_atol'],rtol=C['forward_rtol'])
    assert total/(n*length)==row['confirmation']['ce'];assert correct/(n*length)==row['confirmation']['accuracy']
    history=[h for h in freeze['history'] if h['arm']==row['arm'] and h['seed']==row['seed']];assert row['selection']['ce']==min(h['selection']['ce'] for h in history)
    rows.append({'arm':row['arm'],'seed':row['seed'],'ce':total/(n*length),'trained_cached_max_error':difference})
for seed in C['seeds']:
    for step in range(0,C['updates']+1,C['selection_every']):
        group=[h for h in freeze['history'] if h['seed']==seed and h['step']==step];assert len(group)==len(C['patterns']) and len({h['batch_sha256'] for h in group})==1
for row in freeze['history']:assert run.sha(out/row['checkpoint'])==row['sha256']
run.save(out/'VERIFICATION.json',{'status':'PASS','independent_replays':rows,'selection_minima':True,'matched_training_batches':True,'all_checkpoint_hashes':True,'source_hashes':True})
print('PASS:',len(rows),'independent endpoints and trained cached decoding')
