"""Independent forward scoring plus depth-selector/checkpoint selection audit."""
import json
import torch
from torch.nn import functional as F
import run
run.setup();out=run.H/'execution';freeze=json.loads((out/'FREEZE.json').read_text());result=json.loads((out/'RESULT.json').read_text())
assert all(run.sha(run.ROOT/k)==v for k,v in json.loads((run.H/'SOURCE_MANIFEST.json').read_text()).items())
raw=(run.H.parent/'adaptive-depth-shakespeare-20261018-v1/corpus.txt').read_bytes();start,end=run.C['text_regions']['confirmation'];n=run.C['confirmation_examples'];length=run.C['context'];batch=run.C['batch'];chunks=torch.tensor([list(raw[start+i*(length+1):start+(i+1)*(length+1)]) for i in range(n)])
assert start+n*(length+1)==end
checked=[]
for row in result['rows']:
    assert run.sha(run.ROOT/row['checkpoint'])==row['sha256'];ref=next(r for r in run.references() if r['seed']==row['seed']);m=run.model(ref,row['depth'],run.ROOT/row['checkpoint']);m.eval();total=0.;correct=0
    with torch.no_grad():
        for i in range(0,n,batch):
            x=chunks[i:i+batch,:-1].contiguous();y=chunks[i:i+batch,1:].contiguous();pred=m(x)
            total+=F.cross_entropy(pred.flatten(0,1),y.flatten(),reduction='sum').item();correct+=(pred.argmax(-1)==y).sum().item()
    assert total/(n*length)==row['confirmation']['ce'] and correct/(n*length)==row['confirmation']['accuracy']
    checked.append({'seed':row['seed'],'kind':row['kind'],'depth':row['depth'],'ce':total/(n*length)})
for ref in run.references():
    seed=ref['seed'];history=[r for r in freeze['history'] if r['seed']==seed];policy=next(p for p in freeze['policies'] if p['seed']==seed);eligible=[policy['incumbent']['selection']['ce']]
    for record in history:
        state=torch.load(out/f"depth-selector-{seed}-round{record['round']}.pt",weights_only=True)
        assert state['logits'].softmax(0).tolist()==record['depth_probabilities']
        assert run.C['depths'][int(state['logits'].argmax())]==record['argmax_depth']
        eligible.append(next(r['selection']['ce'] for r in record['selection'] if r['depth']==record['argmax_depth']))
    assert policy['selected']['selection']['ce']==min(eligible)
    for depth in run.C['depths']:
        profile=next(r for r in freeze['profiles'] if r['seed']==seed and r['depth']==depth)
        assert profile['selection']['ce']==min(r['selection']['ce'] for h in history for r in h['selection'] if r['depth']==depth)
run.save(out/'VERIFICATION.json',{'status':'PASS','independent_replays':checked,'selector_probability_and_argmax':True,'validation_selection_minima':True,'source_hashes':True})
print('PASS:',len(checked),'independent confirmation replays and learned depth selection')
