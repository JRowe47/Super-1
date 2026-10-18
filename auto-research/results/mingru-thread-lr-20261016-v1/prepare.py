"""Declare all trials and their order before observing new training results."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'gru-batch-wall-20261016-v1'
REPO = ROOT.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT / 'PROTOCOL.json').exists()
prior = json.loads((PREVIOUS / 'PROTOCOL.json').read_text())
cells = [r for t in (1, 2, 4) for r in json.loads((PREVIOUS / 'benchmark' / f'threads{t}.json').read_text())['rows']]
runs = []
for b, t, lr in [(1,1,.002), (1,2,.002), (1,4,.002), (4,2,.002), (4,2,.004), (16,4,.002), (16,4,.008)]:
    cell = next(r for r in cells if r['arm'] == 'mingru_linear' and r['batch'] == b and r['threads'] == t)
    runs.append(dict(run_id=f'b{b}_t{t}_lr{int(lr*1000):03d}', batch=b, threads=t, peak_lr=lr,
        reference_step_wall_seconds=cell['median_step_wall_seconds']))
np.random.default_rng(101625).shuffle(runs)
sources = [ROOT/'experiment.py', ROOT/'prepare.py', ROOT/'launch.py', PREVIOUS/'experiment.py']
sources += [PREVIOUS / name for name in prior['source_hashes'] if name.endswith('.py') and name not in ('experiment.py', 'prepare.py', 'launch.py')]
p = {k: prior[k] for k in ('data_hashes', 'tokenizer_sha256', 'data', 'seed', 'data_seed', 'max_windows',
    'train_stream_sha256', 'train_stream_prefix', 'monitor_starts', 'final_starts', 'optimizer', 'timing', 'stopping', 'final_metric')}
p.update(run_id=ROOT.name, parent_commit='68f53d2f98cb52712c248014938956de2317a1d8',
    parent_protocol_sha256=sha(PREVIOUS/'PROTOCOL.json'), scope='Independent one-seed CPU explorer screen; official test untouched',
    authorization='User: continue; update existing draft PR in JRowe47/Super-1 only, never upstream without explicit go-ahead',
    configs={'mingru_linear': prior['configs']['mingru_linear']}, parameters={'mingru_linear': 996320},
    runs=runs, run_order_seed=101625, train_wall_seconds=60.,
    schedule='5% training-wall warmup (factor floor0.04), then cosine to10% peak over full60s; declared peak LR per trial',
    training='Fresh initialization with seed101603 in every trial; same ordered window stream regrouped into batches; reset512 full BPTT; sequential isolated workers',
    contrasts=['B1/T2 and B1/T4 vs fresh B1/T1 at LR0.002', 'B4/T2 LR0.004 vs LR0.002', 'B16/T4 LR0.008 vs LR0.002', 'All seven final validation losses and optimizer-update counts at matched training-wall allowance'],
    selection='Publish every declared trial, including failures; final validation CE ranks configurations descriptively; no hidden retries or replacement arms',
    expected_resources='Seven sequential60s training trials on existing CPU, max4threads, no paid resources or GPU',
    hard_limits=dict(cpu_seconds_per_process=600, wall_seconds_per_training_process=180, max_windows=100000),
    failure_policy='Persist model/optimizer/state at exception; record failure; continue remaining declared arms',
    caveats=['One seed reused from parent for controlled comparison; not independent seed replication',
        'Repeatedly viewed validation, tiny models and a short training budget; no GPU or general scaling claim',
        'Equal training-wall allowance is not equal aggregate CPU consumption; startup/evaluation/IO excluded and reported separately',
        'Two larger-batch peak LR trials are exploratory choices, not an established Adam batch-scaling law',
        'Small numerical thread differences and shared-host timing variation remain possible',
        'Common data prefix is regrouped by batch; different throughput means different exposure and optimizer-step counts'],
    source_hashes={os.path.relpath(f, ROOT): sha(f) for f in sources},
    created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
(ROOT/'PROTOCOL.json').write_text(json.dumps(p, indent=2)+'\n')
(ROOT/'REGISTRATION.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_TRAINING', protocol_sha256=sha(ROOT/'PROTOCOL.json'), created_utc=p['created_utc']), indent=2)+'\n')
registry_path = REPO / 'auto-research/RUN_REGISTRY.json'
r = json.loads(registry_path.read_text())
assert all(x['run_id'] != ROOT.name for x in r['runs'])
r['runs'].append(dict(run_id=ROOT.name, status='FROZEN_BEFORE_TRAINING', protocol=str((ROOT/'PROTOCOL.json').relative_to(REPO)),
    protocol_sha256=sha(ROOT/'PROTOCOL.json'), metrics=str((ROOT/'SUMMARY.json').relative_to(REPO)),
    report=f'auto-research/reports/{ROOT.name}.md', publication_target='JRowe47/Super-1 main only'))
registry_path.write_text(json.dumps(r, indent=2)+'\n')
print('FROZEN', sha(ROOT/'PROTOCOL.json'), json.dumps(runs))
