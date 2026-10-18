"""Independently replay all frozen confirmation endpoints; no training."""
import hashlib,json,math,statistics,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from model import build,logits
H=Path(__file__).resolve().parent; ROOT=H.parents[2]; O=H/'execution'
c=json.loads((H/'CONFIG.json').read_text()); p=json.loads((H/'PROTOCOL.json').read_text())
torch.set_num_threads(c['threads']); torch.set_num_interop_threads(c['threads']); torch.use_deterministic_algorithms(True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
sources=json.loads((H/'SOURCE_MANIFEST.json').read_text())
assert all(sha(ROOT/k)==v for k,v in sources.items())
frozen=json.loads((O/'FREEZE.json').read_text()); result=json.loads((O/'RESULT.json').read_text()); terminal=json.loads((O/'TERMINAL.json').read_text())
assert terminal['status']=='COMPLETED'
expected={(a,s) for a in c['arms'] for s in c['seeds']}
assert {(r['arm'],r['seed']) for r in result['rows']}==expected
assert len(result['rows'])==len(expected)==len(frozen['checkpoints'])
raw=(H/'corpus.txt').read_bytes(); a=int(len(raw)*c['train_fraction']); b=int(len(raw)*c['selection_end_fraction'])
for name,blob in [('train',raw[:a]),('selection',raw[a:b]),('confirmation',raw[b:])]:
    assert hashlib.sha256(blob).hexdigest()==p['data'][name+'_sha256']
blob=torch.tensor(list(raw[b:]),dtype=torch.long)
count=c['confirmation_windows']; length=c['context']
x=torch.stack([blob[i*(length+1):i*(length+1)+length] for i in range(count)])
y=torch.stack([blob[i*(length+1)+1:i*(length+1)+length+1] for i in range(count)])
progress=[json.loads(s) for s in (O/'progress.jsonl').read_text().splitlines()]
for row in result['rows']:
    arm,seed=row['arm'],row['seed']; path=O/row['checkpoint']
    assert sha(path)==row['checkpoint_sha256']==frozen['checkpoints'][path.name]
    assert row['updates']==c['updates'] and row['target_exposures']==c['updates']*c['batch']*length
    assert row['confirmation']==row['duplicate']
    steps=[r['update'] for r in progress if r['phase']=='train' and r['arm']==arm and r['seed']==seed]
    assert steps==list(range(c['selection_every'],c['updates']+1,c['selection_every']))
    m=build(c,arm,seed); m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')); m.eval()
    total=0.; digest=hashlib.sha256()
    with torch.inference_mode():
        for i in range(0,count,c['batch']):
            pred=logits(m,arm,x[i:i+c['batch']],y[i:i+c['batch']])
            total+=F.cross_entropy(pred.flatten(0,1),y[i:i+c['batch']].flatten(),reduction='sum').item()
            digest.update(pred.numpy().tobytes())
    assert total/y.numel()==row['confirmation']['ce']
    assert digest.hexdigest()==row['confirmation']['logits_sha256']
    assert row['parameters']==sum(p.numel() for p in m.parameters())
    assert all(math.isfinite(row['benchmark'][key]) and row['benchmark'][key]>0 for key in ['prefill_ms_p50','prefill_ms_p95','decode_ms_per_token_p50','process_peak_rss_bytes'])
for seed in c['seeds']:
    assert len({r['sampler_sha256'] for r in result['rows'] if r['seed']==seed})==1
for control,contrast in result['contrasts'].items():
    ce={(r['arm'],r['seed']):r['confirmation']['ce'] for r in result['rows']}
    gaps=[ce['token',s]-ce[control,s] for s in c['seeds']]
    assert gaps==contrast['token_minus_control_nats']
    assert statistics.mean(gaps)==contrast['mean']
    half=c['t_critical_df2']*statistics.stdev(gaps)/math.sqrt(len(gaps))
    assert contrast['descriptive_paired_t95']==[statistics.mean(gaps)-half,statistics.mean(gaps)+half]
def gate(control):
    z=result['contrasts'][control]
    return max(z['token_minus_control_nats'])<0 and z['mean']<=-c['minimum_gain_nats'] and z['descriptive_paired_t95'][1]<0
learning=all(r['final_selection']['ce']<r['initial_selection']['ce'] for r in result['rows'])
assert result['primary_fixed_comparison_passed']==(gate('fixed') and learning)
assert result['token_specific_control_passed']==(gate('fixed') and gate('global') and learning)
assert result['nanoGPT_comparison_passed']==(gate('fixed') and gate('global') and gate('dense') and gate('nanogpt') and learning)
print(json.dumps({'status':'PASS','frozen_checkpoints_verified':len(expected),'independent_FP32_confirmation_replays':len(expected),'all_sampler_exposures_and_decision_rules_verified':True,'outcome':result['outcome']},indent=2))
