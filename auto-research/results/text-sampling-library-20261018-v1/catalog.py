"""Catalog the independently replayed study, preserving unpromoted candidates."""
import sys
import json
import torch
import run
sys.path.insert(0,str(run.ROOT))
import sampling_library as library
run.setup()
out=run.H/'execution';result=json.loads((out/'RESULT.json').read_text())
assert json.loads((out/'VERIFICATION.json').read_text())['status']=='PASS'
assert not library.DEFAULT_INDEX.exists(), 'Do not overwrite an existing library'
def artifact(path):return {'path':str(path.relative_to(run.ROOT)),'sha256':run.sha(path)}
positions=[]
for row in result['rows']:
    ref=next(r for r in run.references() if r['seed']==row['seed'])
    m=run.model(ref,run.ROOT/row['checkpoint']);m.eval()
    with torch.no_grad():
        _,t=m(run.evaluation('selection')[0],positions=True)
        pos={'seed':row['seed'],'arm':row['arm'],'mean':t.mean((0,1)).tolist(),'std':t.std((0,1)).tolist(),'minimum':t.amin((0,1)).tolist(),'maximum':t.amax((0,1)).tolist()}
    positions.append(pos)
    promoted=result['promoted'] and row['arm']==result['winner_on_validation']
    reference=row['arm']=='incumbent'
    entry={'id':f"{run.C['run_id']}/{row['arm']}/{row['seed']}",'use_case':run.C['use_case'],
           'base_fingerprint':ref['base_fingerprint'],'tokenizer':run.C['tokenizer'],'context':run.C['context'],'seed':row['seed'],
           'status':'confirmed_improvement' if promoted else 'research_reference' if reference else 'candidate',
           'active':promoted or (reference and not result['promoted']),'execution_depth':run.C['layers'],
           'checkpoint':artifact(run.ROOT/row['checkpoint']),'base_checkpoint':artifact(run.PRIOR/'execution'/ref['base_checkpoint']),
           'training':{'run_id':run.C['run_id'],'selected_round':row['round'],'initial_lr':run.C['arms'].get(row['arm']),'updates_per_round':run.C['updates_per_round'],'backbone_frozen':True},
           'metrics':{'selection':row['selection'],'confirmation':row['confirmation'],'unit':'nats/byte','confirmation_byte_region':run.C['text_regions']['confirmation'],'positions':pos},
           'evidence':[artifact(run.H/'PROTOCOL.json'),artifact(run.H/'SOURCE_MANIFEST.json'),artifact(out/'FREEZE.json'),artifact(out/'RESULT.json'),artifact(out/'VERIFICATION.json')],
           'limitations':'Single Shakespeare corpus; reference means retained research baseline, not universal optimum or established architecture improvement.'}
    library.register(entry)
run.save(out/'POSITIONS.json',positions)
print('Cataloged',len(result['rows']),'entries')
