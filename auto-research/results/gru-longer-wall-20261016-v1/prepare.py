"""Freeze a six-recipe, longer-horizon comparison before training."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
BATCH = ROOT.parent/'gru-batch-wall-20261016-v1'
REPO = ROOT.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'PROTOCOL.json').exists()
prior = json.loads((BATCH/'PROTOCOL.json').read_text())
cells = [r for t in (1,2,4) for r in json.loads((BATCH/'benchmark'/f'threads{t}.json').read_text())['rows']]
runs = []
for arm,b,t,lr in [('mingru_linear',4,2,.004),('mingru_linear',16,4,.008),
                  ('native',4,2,.004),('native',16,4,.008),
                  ('nanogpt',4,4,.002),('nanogpt',4,4,.004)]:
    cell = next(r for r in cells if r['arm']==arm and r['batch']==b and r['threads']==t)
    runs.append(dict(run_id=f'{arm}_b{b}_t{t}_lr{int(lr*1000):03d}',arm=arm,batch=b,threads=t,peak_lr=lr,
        reference_step_wall_seconds=cell['median_step_wall_seconds']))
np.random.default_rng(101626).shuffle(runs)
sources = [ROOT/'experiment.py',ROOT/'prepare.py',ROOT/'launch.py',BATCH/'experiment.py']
sources += [BATCH/name for name in prior['source_hashes'] if name.endswith('.py') and name not in ('experiment.py','prepare.py','launch.py')]
p = {k:prior[k] for k in ('data_hashes','tokenizer_sha256','data','max_windows','monitor_starts','final_starts',
    'optimizer','stopping','final_metric','configs','parameters')}
seed, data_seed = 101604, 101613
starts = np.random.default_rng(data_seed).integers(0,p['data']['train_tokens']-512,size=p['max_windows'],dtype=np.int64)
p.update(run_id=ROOT.name,parent_commit='052d21657481c323ca2b62823105132b1ae2720a',
    parent_protocol_sha256=sha(ROOT.parent/'mingru-thread-lr-20261016-v1'/'PROTOCOL.json'),
    implementation_protocol_sha256=sha(BATCH/'PROTOCOL.json'),
    scope='Independent one-seed CPU explorer recipe comparison; no official test access',
    authorization='User: proceed with the next run; push results to own-fork draft PR JRowe47/Super-1 #1 only; no upstream submission or auto-merge',
    seed=seed,data_seed=data_seed,train_stream_sha256=hashlib.sha256(starts.tobytes()).hexdigest(),
    train_stream_prefix=starts[:16].tolist(),runs=runs,run_order_seed=101626,
    train_wall_seconds=180.,monitor_interval_seconds=30,
    schedule='5% training-wall warmup (factor floor0.04), then cosine to10% peak over180s; peak LR declared per recipe',
    training='Fresh initialization with seed101604 for all recipes; common new ordered window stream seed101613, regrouped by batch; reset512/full BPTT; sequential workers',
    timing='Training wall includes batch construction through optimizer update. Separate run wall/CPU covers initial monitoring through final evaluation and checkpoint IO, excluding setup. Setup timers start before torch import. Launcher additionally records end-to-end worker wall from before Popen through process exit.',
    contrasts=['Linear minGRU vs native GRU at matched B4/T2/LR0.004',
        'Linear minGRU vs native GRU at matched B16/T4/LR0.008',
        'nanoGPT B4/T4 LR0.004 vs0.002',
        'All six practical recipes ranked at the same180s training-wall allowance'],
    selection='Publish every declared recipe and failure; endpoint CE ranks this screen descriptively. No adaptive tuning or best-checkpoint selection.',
    expected_resources='Six sequential180s CPU training trials, max4threads; existing resources only; no GPU or new spend',
    hard_limits=dict(cpu_seconds_per_process=1200,wall_seconds_per_training_process=300,launcher_wall_seconds_per_process=330,max_windows=p['max_windows']),
    caveats=['One new seed and data stream, not a multi-seed replication; do not attribute cross-study changes solely to longer training',
        'Repeatedly viewed validation, ~1M models and a short CPU horizon; no official GPU speedrun or large-model scaling claim',
        'Two declared recipes per architecture are a small search, not tuned-best baselines; GRU recipes inherit minGRU-motivated choices',
        'GRU recipe comparisons vary batch, threads and peak LR together; only the nanoGPT pair isolates peak LR configuration',
        'Equal wall allowance does not mean equal aggregate CPU, FLOPs, tokens or optimizer steps; report all observed counts',
        'Model parameters, family-specific numerical behavior, and batch grouping differ; seeded initialization does not make distinct architectures identical',
        'Training-wall schedules depend on host speed and have a longer9s warmup; a60s checkpoint is not equivalent to the previous60s run',
        'nanoGPT uses the corrected single-CE wrapper; earlier duplicate-CE timing is not reused'],
    source_hashes={os.path.relpath(f,ROOT):sha(f) for f in sources},
    created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
(ROOT/'PROTOCOL.json').write_text(json.dumps(p,indent=2)+'\n')
(ROOT/'REGISTRATION.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_TRAINING',protocol_sha256=sha(ROOT/'PROTOCOL.json'),created_utc=p['created_utc']),indent=2)+'\n')
regpath = REPO/'auto-research/RUN_REGISTRY.json'; reg=json.loads(regpath.read_text())
assert all(r['run_id']!=ROOT.name for r in reg['runs'])
reg['runs'].append(dict(run_id=ROOT.name,status='FROZEN_BEFORE_TRAINING',protocol=str((ROOT/'PROTOCOL.json').relative_to(REPO)),
    protocol_sha256=sha(ROOT/'PROTOCOL.json'),metrics=str((ROOT/'SUMMARY.json').relative_to(REPO)),
    report=f'auto-research/reports/{ROOT.name}.md',publication_target='JRowe47/Super-1 main only'))
regpath.write_text(json.dumps(reg,indent=2)+'\n')
print('FROZEN',sha(ROOT/'PROTOCOL.json'),json.dumps(runs))
