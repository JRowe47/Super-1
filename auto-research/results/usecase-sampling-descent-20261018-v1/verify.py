"""Independently audit selection, frozen weights, samplers and every confirmation score."""
import hashlib,json,math,statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from policy import build,fingerprint
from tasks import load_data,evaluation,generate
H=Path(__file__).resolve().parent;ROOT=H.parents[2];O=H/'execution';c=json.loads((H/'CONFIG.json').read_text())
torch.set_num_threads(c['threads']);torch.set_num_interop_threads(c['threads']);torch.use_deterministic_algorithms(True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((H/'SOURCE_MANIFEST.json').read_text());assert all(sha(ROOT/k)==v for k,v in manifest.items())
assert generate(c)==json.loads((H/'DATA.json').read_text())
r=json.loads((O/'RESULT.json').read_text());freeze=json.loads((O/'FREEZE.json').read_text());hist=json.loads((O/'SAMPLING_HISTORY.json').read_text());terminal=json.loads((O/'TERMINAL.json').read_text())
assert terminal['status']=='COMPLETED' and len(hist)==len(c['seeds'])*(1+2*len(c['tasks']))*(c['rounds']+1)
base={b['seed']:b for b in freeze['base_records']}
for seed,b in base.items():assert sha(O/b['checkpoint'])==b['sha256']
for row in hist:
 assert sha(O/row['checkpoint'])==row['sha256'];assert row['base_fingerprint']==base[row['seed']]['base_fingerprint']
 assert (O/row['checkpoint']).stat().st_mtime_ns<=(O/'FREEZE.json').stat().st_mtime_ns
for z in freeze['selected']:
 history=[h for h in hist if h['seed']==z['seed'] and h['candidate']==z['candidate']]
 best=min(history,key=lambda h:(h['criterion'],h['round']))
 assert best['round']==z['best_round'] and best['checkpoint']==z['checkpoint']
 assert len(history)==c['rounds']+1
for seed in c['seeds']:
 for round_id in range(1,c['rounds']+1):
  shared=next(h for h in hist if h['seed']==seed and h['round']==round_id and h['candidate']=='shared_token')
  for task in c['tasks']:
   for variant in ['global','token']:
    expert=next(h for h in hist if h['seed']==seed and h['round']==round_id and h['candidate']==task+'_'+variant)
    assert expert['sampler_sha256'][task]==shared['sampler_sha256'][task]
    assert expert['updates_this_round']==c['updates_per_round_per_task']
  assert shared['updates_this_round']==len(c['tasks'])*c['updates_per_round_per_task']
d=load_data(c);expected={(s,t,a) for s in c['seeds'] for t in c['tasks'] for a in ['uniform','shared_token',t+'_global',t+'_token']}
assert len(r['rows'])==len(expected) and {(v['seed'],v['task'],v['candidate']) for v in r['rows']}==expected
for row in r['rows']:
 seed,task,name=row['seed'],row['task'],row['candidate'];b=base[seed]
 state=torch.load(O/b['checkpoint'],weights_only=True)['base'];mode='fixed' if name=='uniform' else ('global' if name.endswith('_global') else 'token')
 m=build(c,seed,mode,state)
 if row['checkpoint']:m.router.load_state_dict(torch.load(O/row['checkpoint'],weights_only=True)['router'])
 assert fingerprint(m.base)==b['base_fingerprint'];m.eval();x,y,mask=evaluation(d,c,task,'confirmation')
 with torch.inference_mode():
  _,positions=m(x[:c['batch']],positions=True)
  assert torch.isfinite(positions).all() and (positions>=0).all() and (positions<=1).all()
  assert (positions.diff(dim=-1)>=0).all() and (positions[:,:,0]==0).all() and (positions[:,:,-1]==1).all()
 total=correct=targets=0;digest=hashlib.sha256()
 with torch.inference_mode():
  for i in range(0,len(x),c['batch']):
   selected=m(x[i:i+c['batch']])[mask[i:i+c['batch']]];target=y[i:i+c['batch']][mask[i:i+c['batch']]]
   total+=F.cross_entropy(selected,target,reduction='sum').item();correct+=(selected.argmax(-1)==target).sum().item();targets+=target.numel();digest.update(selected.numpy().tobytes())
 actual={'ce':total/targets,'accuracy':correct/targets,'targets':targets,'logits_sha256':digest.hexdigest()}
 assert actual==row['confirmation']==row['duplicate'],(seed,task,name)
lookup={(v['candidate'],v['task'],v['seed']):v['confirmation'] for v in r['rows']}
for task in c['tasks']:
 for control,z in r['contrasts'][task].items():
  gaps=[lookup[task+'_token',task,s]['ce']-lookup[control,task,s]['ce'] for s in c['seeds']]
  assert gaps==z['gaps_nats_per_target'];mean=statistics.mean(gaps);sem=statistics.stdev(gaps)/math.sqrt(len(gaps));assert mean==z['mean']
  for key,critical in [('paired_t95',c['t95_df2']),('familywise_interval',c['bonferroni_t_df2'])]:assert z[key]==[mean-critical*sem,mean+critical*sem]
 def gate(control):
  z=r['contrasts'][task][control];return max(z['gaps_nats_per_target'])<0 and z['mean']<=-c['minimum_gain_nats'] and z['familywise_interval'][1]<0
 if task=='text':qualified=all(b['final_selection'][task]['ce']<b['initial_selection'][task]['ce']*(1-c['minimum_text_relative_gain']) for b in base.values())
 else:qualified=all(b['final_selection'][task]['accuracy']>=c['minimum_task_accuracy'] for b in base.values())
 assert r['decisions'][task]=={'base_learning_qualified':qualified,'improves_frozen_uniform':qualified and gate('uniform'),'specialization_supported':qualified and gate('uniform') and gate('shared_token'),'token_conditioning_supported':qualified and gate('uniform') and gate('shared_token') and gate(task+'_global')}
print(json.dumps({'status':'PASS','base_checkpoints':len(base),'sampler_round_checkpoints':len(hist),'independent_confirmation_replays':len(expected),'selection_and_paired_samplers_verified':True,'source_data_and_frozen_backbone_hashes_verified':True,'decisions_verified':True},indent=2))
