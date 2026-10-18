"""Preserve measured depth profiles and learned depth preferences in the library."""
import json
import sys
import statistics
import run
sys.path.insert(0,str(run.ROOT))
import sampling_library as library
from sampling_library.profiles import INDEX,load_profile
run.setup();out=run.H/'execution';r=json.loads((out/'RESULT.json').read_text());freeze=json.loads((out/'FREEZE.json').read_text());assert json.loads((out/'VERIFICATION.json').read_text())['status']=='PASS'
def artifact(path):return {'path':str(path.relative_to(run.ROOT)),'sha256':run.sha(path)}
entries=[];checks=[]
for row in r['rows']:
    if row['kind']!='profile':continue
    ref=next(x for x in run.references() if x['seed']==row['seed'])
    e={'id':f"{run.C['run_id']}/layers-{row['depth']}/{row['seed']}",'use_case':run.C['use_case'],'base_fingerprint':ref['base_fingerprint'],'tokenizer':run.C['tokenizer'],'context':run.C['context'],'seed':row['seed'],
       'status':'candidate','active':False,'execution_depth':row['depth'],
       'checkpoint':artifact(run.ROOT/row['checkpoint']),'base_checkpoint':artifact(run.PRIOR/'execution'/ref['base_checkpoint']),
       'config':artifact(run.H/'CONFIG.json'),'model_source':artifact(run.H/'model.py'),'source_manifest':artifact(run.H/'SOURCE_MANIFEST.json'),
       'training':{'run_id':run.H.name,'selected_round':row['round'],'coordinate_updates':row['round']*run.C['updates_per_round'] if row['depth']>2 else 0,'backbone_frozen':True},
       'metrics':{'selection':row['selection'],'confirmation':row['confirmation'],'benchmark':row['benchmark'],'unit':'nats/byte','confirmation_byte_region':run.C['text_regions']['confirmation']},
       'evidence':[artifact(run.H/'PROTOCOL.json'),artifact(out/'FREEZE.json'),artifact(out/'RESULT.json'),artifact(out/'VERIFICATION.json')],
       'limitations':'Explicit experimental profile, not an automatic replacement. Backbone trained at4 layers; low-depth normalization-prefix distribution shift. No GPU speed claim.'}
    library.validate(e);entries.append(e)
policies=[]
for p in freeze['policies']:
    policies.append({'seed':p['seed'],'use_case':run.C['use_case'],'learned_argmax_layers':p['final_argmax_depth'],'learned_probabilities':dict(zip(map(str,run.C['depths']),p['final_probabilities'])),'validation_selected_layers':p['selected']['depth'],'validation_selected_checkpoint':artifact(run.ROOT/p['selected']['checkpoint']),'selector_checkpoint':artifact(out/f"depth-selector-{p['seed']}-round{run.C['rounds']}.pt"),'promoted':r['promoted'],'default_reference_retained':not r['promoted']})
run.save(INDEX,{'schema_version':1,'entries':entries,'policies':policies,'scope':'Explicit measured depth profiles. Lower CE is better; smaller layer count is not an automatic quality win.'})
for entry in entries:
    m,loaded=load_profile(entry['id']);metric=run.score(m,run.evaluation('confirmation'))
    assert metric==entry['metrics']['confirmation']
    counts=[0]*run.C['layers'];handles=[]
    for i,block in enumerate(m.base.blocks):
        def hook(module,args,result,index=i):counts[index]+=1
        handles.append(block.ln1.register_forward_hook(hook))
    with run.torch.no_grad():m(run.evaluation('selection')[0][:1])
    for handle in handles:handle.remove()
    assert counts==[int(i<entry['execution_depth']) for i in range(run.C['layers'])]
    checks.append({'id':entry['id'],'exact_library_reload':True,'executed_blocks':counts})
run.save(out/'LIBRARY_VERIFICATION.json',{'status':'PASS','checks':checks})
print('PASS:9 depth profiles reload exactly and execute only their chosen layer count')
