"""Independent artifact verification; no torch, training, or evaluation."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'auto-research/results/synthetic-rotation-pr-20261015-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
p=read(OLD/'PROTOCOL.json'); c=p['config']
registry=read(ROOT/'auto-research/RUN_REGISTRY.json')
for directory in [OLD,HERE]:
    entry=next(r for r in registry['runs'] if r['run_id']==directory.name)
    assert sha(directory/'PROTOCOL.json')==entry['protocol_sha256']
    assert all(sha(ROOT/k)==v for k,v in read(directory/'SOURCE_MANIFEST.json').items())
data=read(OLD/'DATA.json')
assert data==runpy.run_path(str(OLD/'generate_data.py'))['generate'](c)
for split in data:
    digest=hashlib.sha256(json.dumps(data[split],separators=(',',':')).encode()).hexdigest()
    assert digest==p['data'][split+'_sha256']
    assert len(data[split])==c['split_batches'][split]
    assert all(len(batch)==c['batch_size'] for batch in data[split])
    assert all(len(doc)==c['sequence_length']+1 for batch in data[split] for doc in batch)
progress=[json.loads(s) for s in (OLD/'execution/progress.jsonl').read_text().splitlines()]
assert len(progress)==len(c['arms'])*len(c['seeds'])*p['training']['updates']
frozen=read(OLD/'execution/CHECKPOINT_FREEZE.json')
terminal=read(HERE/'execution/TERMINAL.json')
assert terminal['status']=='COMPLETED_EVALUATION_RECOVERY_NO_RETRAINING' and terminal['training_updates']==0
assert read(OLD/'EXIT.json')['exit_code']==1 and read(HERE/'EXIT.json')['exit_code']==0
assert len(frozen)==len(terminal['rows'])==12
rows=terminal['rows']
for row in rows:
    seed,arm=row['seed'],row['arm']
    series=[r for r in progress if r['seed']==seed and r['arm']==arm]
    assert [r['update'] for r in series]==list(range(1,81))
    assert [r['targets'] for r in series]==[128*i for i in range(1,81)]
    assert all(math.isfinite(r['train_ce']) and math.isfinite(r['preclip_grad_norm']) for r in series)
    assert row['updates']==80 and row['target_exposures']==10240
    assert sha(OLD/'execution'/row['checkpoint'])==row['checkpoint_sha256']
    assert row['confirmation']==row['duplicate_confirmation']
    assert row['confirmation']['targets']==row['reset_window_confirmation']['targets']==1024
    assert read(HERE/'execution'/f'{seed}-{arm}.json')==row
    assert row['final_selection']['ce_nats_per_target']<row['initial_selection']['ce_nats_per_target']
result=read(HERE/'execution/RESULT.json')
for pair in result['paired']:
    matching={r['arm']:r for r in rows if r['seed']==pair['seed']}
    assert matching['oriented_cross']['parameters']==matching['alignment_dot']['parameters']==8832
    assert matching['native']['parameters']==8816 and matching['pinned_nanogpt']['parameters']==8136
    assert pair['ce']=={arm:r['confirmation']['ce_nats_per_target'] for arm,r in matching.items()}
    assert pair['oriented_minus_alignment']==pair['ce']['oriented_cross']-pair['ce']['alignment_dot']
gaps=[r['oriented_minus_alignment'] for r in result['paired']]
mean=statistics.mean(gaps); half=c['t_critical_95_df2']*statistics.stdev(gaps)/math.sqrt(len(gaps))
assert result['mean_oriented_minus_alignment']==mean
assert result['paired_t_95_interval']==[mean-half,mean+half]
gate=all(x<0 for x in gaps) and mean<=-c['minimum_effect_nats'] and mean+half<0 and all(r['oriented_minus_native']<0 for r in result['paired'])
assert result['primary_gate_passed']==gate
cpu=terminal['cpu_seconds']+read(OLD/'execution/TERMINAL.json')['cpu_seconds']
assert cpu<300
files=[p for d in [OLD,HERE,ROOT/'auto-research/logs'/OLD.name] for p in d.rglob('*') if p.is_file()]
size=sum(p.stat().st_size for p in files)
assert size<c['max_artifact_bytes']
print(json.dumps({'status':'PASS','arm_seed_rows':len(rows),'training_updates':len(progress),'targets_per_arm_seed':10240,'duplicate_rows_exact':12,'data_regenerated_exactly':True,'all_selection_losses_decreased':True,'paired_mean':mean,'primary_gate_passed':gate,'experiment_cpu_seconds_total':cpu,'evidence_bytes_checked':size},indent=2))
