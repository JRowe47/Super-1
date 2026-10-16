"""Register fresh recovery trials; preserve the lost v1 attempt separately."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import experiment as ex
import numpy as np
assert not (ROOT/'PROTOCOL.json').exists(),'Frozen protocol already exists'
old=json.loads((ROOT.parent/'gru-longer-wall-20261016-v1/PROTOCOL.json').read_text())
p={k:old[k] for k in ('data','data_hashes','tokenizer_sha256')}
p.update(run_id=ROOT.name,parent_commit='1a7111ac68d5760d52b01cd060f397b1b776ed47',seed=101605,data_seed=101614,max_blocks=100000,
    train_wall_seconds=90.,monitor_interval_seconds=30,
    hypothesis='Longer trained context may improve common-tail prediction; widening or slower feedback may change the quality/time tradeoff.',
    authorization='User: Go ahead and continue; continuing explicit rapid CPU exploration and push-to-own-fork authorization. No GPU or paid provisioning.',
    recovery='Fresh v2 after environment reset. V1 console-only partial result remains separate; its checkpoint/protocol were not recovered. Same declared choices and seeds, not an independent replication.',
    primary_metric='Mean CE on identical final512 targets of16 validation windows, final checkpoint only. Lower is better.',
    useful_signal='Descriptive CE difference0.05 motivates follow-up; not a confirmation threshold or automatic promotion.',
    training='Common ordered2048-target blocks tiled into512/1024/2048 sequences. Four blocks/update for recurrent arms; one for nanoGPT. Reset each example; full BPTT. EOS does not reset within an example.',
    optimizer='AdamW betas0.9/0.95 eps1e-8 clip1. Decay0.1 for matrices except2048-row embedding tables; zero decay for vectors/biases AND nanoGPT token/position tables.',
    schedule='5% training-wall warmup with factor floor0.04; cosine to10% peak. Recurrent8192-target updates LR0.008; nanoGPT2048-target updates LR0.002.',
    timing='Training includes data/forward/CE/backward/clip/update; excludes eval/checkpoint. Separate run, setup, launcher and aggregate CPU clocks.',
    prefix_diagnostic='Replace only inputs older than final512 with next validation window prefix cyclically; suffix, targets and absolute positions unchanged. Swapped-minus-authentic CE; distribution-shifting diagnostic, not retrieval proof.',
    selection='Publish all10 trials and failures. One seed, no adaptive choices or best-checkpoint selection.',
    contrasts=['minGRU160 context512/1024/2048','nanoGPT108 context512/1024/2048','minGRU224 vs160 at512 and2048','segment64 vs32 at2048','native147 vs minGRU160 at2048'],
    caveats=['Repeatedly used validation, short CPU screen, not SOTA/scaling/GPU evidence.','Widening changes weights/compute as well as state.',
        'Chunk64 changes segment feedback and document update period.','Same targets/update within family; fewer reset sequences at longer lengths.',
        'All nanoGPT arms use the same2048-position table and width108 to stay near1M; this differs from the previous width120/512-position study.',
        'The existing optimizer exempts all2048-row embedding tables, including positions. V1 prose omitted this effect; v2 declares it explicitly.',
        'Equal training wall is not equal CPU, tokens or FLOPs. New runtime measurements do not isolate changes from earlier studies.'],
    hard_limits=dict(worker_wall_seconds=150,launcher_wall_seconds=180,cpu_seconds_per_process=700))
starts=np.random.default_rng(p['data_seed']).integers(0,p['data']['train_tokens']-2048,size=p['max_blocks'],dtype=np.int64)
p['train_stream_sha256']=hashlib.sha256(starts.tobytes()).hexdigest();p['train_stream_prefix']=starts[:16].tolist()
rng=np.random.default_rng(101615)
p['monitor_ends']=(np.arange(4)*100000+100000+rng.integers(0,50000,size=4)).tolist()
p['final_ends']=(np.arange(16)*35000+700000+rng.integers(0,10000,size=16)).tolist()
assert max(p['monitor_ends'])<min(p['final_ends'])-2048 and max(p['final_ends'])<p['data']['validation_tokens']
assert min(np.diff(sorted(p['final_ends'])))>2048
rows=[]
def add(arm,width,context,segment=32):
    prefix='nano' if arm=='nanogpt' else 'native' if arm=='native' else 'linear'
    cfg=dict(vocab_size=2048,block_size=2048,n_layer=4,n_head=4,n_embd=width,dropout=0.,bias=True) if arm=='nanogpt' else dict(vocab_size=2048,width=width,max_context=2048,segment_length=segment,document_segments=8)
    q=dict(run_id=f'{prefix}_w{width}_c{context}_s{segment}',arm=arm,width=width,context=context,segment_length=segment if arm!='nanogpt' else None,config=cfg,threads=4,
        blocks_per_update=1 if arm=='nanogpt' else 4,peak_lr=.002 if arm=='nanogpt' else .008)
    q['batch']=q['blocks_per_update']*2048//context;q['parameters']=sum(v.numel() for v in ex.build(q,p).parameters())
    q['persistent_state_fp32_bytes_per_sequence']=None if arm=='nanogpt' else 4*width*4;rows.append(q)
for n in (512,1024,2048):add('mingru_linear',160,n);add('nanogpt',108,n)
for n in (512,2048):add('mingru_linear',224,n)
add('native',147,2048);add('mingru_linear',160,2048,64)
np.random.default_rng(101627).shuffle(rows);p['runs']=rows
paths=['experiment.py','prepare.py','launch.py','../gru-batch-wall-20261016-v1/experiment.py','../gru-batch-wall-20261016-v1/vendor/nanogpt/model.py','../gru-extended-20261016-v1/models.py']
paths+=['../gru-extended-20261016-v1/vendor/representation_experiments/'+n for n in ('__init__.py','common.py','predictive_models.py')]
p['source_hashes']={n:ex.sha(ROOT/n) for n in paths};ex.save(ROOT/'PROTOCOL.json',p)
r=dict(status='FROZEN_BEFORE_TRAINING',protocol_sha256=ex.sha(ROOT/'PROTOCOL.json'),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
ex.save(ROOT/'REGISTRATION.json',r)
f=ROOT.parents[1]/'RUN_REGISTRY.json';reg=json.loads(f.read_text())
reg['runs'].append(dict(run_id=ROOT.name,status='REGISTERED_EXPLORATORY',approval=p['authorization'],protocol='auto-research/results/'+ROOT.name+'/PROTOCOL.json',
    protocol_sha256=r['protocol_sha256'],report='auto-research/reports/'+ROOT.name+'.md',recovery_of='gru-context-capacity-20261016-v1'))
ex.save(f,reg);print(json.dumps(dict(registration=r,runs=rows),indent=2))
