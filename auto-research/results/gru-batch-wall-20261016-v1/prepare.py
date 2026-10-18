"""Freeze the throughput screen and its training-selection rule before execution."""
import os,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent/'gru-extended-20261016-v1';REPO=ROOT.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'PROTOCOL.json').exists()
prior=json.loads((PARENT/'PROTOCOL.json').read_text());offsets=json.loads((PARENT/'OFFSETS.json').read_text())
configs={a:prior['configs'][a] for a in ('native','mingru_linear')}
configs['nanogpt']=dict(vocab_size=2048,block_size=512,n_layer=4,n_head=4,n_embd=120,dropout=0.,bias=True)
sources=[ROOT/'experiment.py',ROOT/'prepare.py',ROOT/'launch.py',ROOT/'vendor/nanogpt/model.py',PARENT/'models.py']+list((PARENT/'vendor').rglob('*.py'))
starts=np.random.default_rng(101612).integers(0,prior['data']['train_tokens']-512,size=100000,dtype=np.int64)
p=dict(run_id=ROOT.name,scope='Independent one-seed CPU explorer experiment; no official-test access',
    authorization='User: continue with batching/multithreading; research PRs only to JRowe47/Super-1, never upstream without explicit go-ahead',
    parent_commit='8571e1db1897b87bd9350c45b7c25f14ee94fcc2',arms=list(configs),configs=configs,
    parameters=dict(native=995925,mingru_linear=996320,nanogpt=1004880),seed=101603,data_seed=101612,
    data_hashes={f'{n}.bin':prior['data']['split_metadata']['splits'][n]['sha256'] for n in ('train','validation')},
    tokenizer_sha256=prior['data']['split_metadata']['tokenizer_sha256'],data=prior['data'],max_windows=100000,
    train_stream_sha256=hashlib.sha256(starts.tobytes()).hexdigest(),train_stream_prefix=starts[:16].tolist(),
    monitor_starts=offsets['monitor_starts'],final_starts=offsets['final_starts'],batches=[1,4,16],threads=[1,2,4],
    benchmark_warmups=2,benchmark_repeats=3,benchmark_order_seed=101623,
    benchmark='Full update incl data slicing, forward, one CE, backward, clipping and AdamW; fresh throwaway weights per cell; sequential isolated processes per thread count',
    selection_rule='For each architecture choose maximum median tokens/wall-second among nine cells, independent of validation. Also train B1/T1 reference; reuse if selected setup is identical. No hidden arms.',
    train_wall_seconds=60.,schedule='5% training-wall warmup to0.002 (factor floor0.04), then cosine to0.0002 over full time budget; same unscaled peak LR for all batches',
    optimizer='AdamW0.9/0.95 eps1e-8 wd0.1; same tied-token-table and ndim1 exclusions; clip1',
    training='Fresh seed101603 for every arm/setup; identical ordered window stream regrouped into batches; full512 BPTT with state reset per example; one process at a time',
    timing='Training wall includes batch construction through optimizer step; excludes startup, validation and checkpoint IO. Report actual full-run wall and aggregate CPU separately.',
    stopping='Stop before next update if remaining<=max(0.2s,2x largest of last10 update durations); report actual budget and any overshoot',
    final_metric='CE on same32 reused final-region windows/16384targets; monitor8 separate windows; final endpoint only',
    expected_resources='27 tiny benchmark cells then3-6 sequential60s training runs on existing CPU; max4threads; no paid resources or GPU',
    hard_limits=dict(cpu_seconds_per_process=600,wall_seconds_per_training_process=180,max_windows=100000),
    caveats=['One seed, repeatedly viewed validation, tiny models, no tuned-best-recipe or GPU claim',
             'Maximum-throughput batch may reduce optimizer steps enough to harm short-budget quality',
             'Same wall allowance/max4CPUthreads is not equal CPU consumption or FLOPs',
             'Common unscaled LR can favor one batch; no batch-specific tuning',
             'nanoGPT wrapper now reuses its returned CE; prior wrapper computed an extra discarded CE forward. Older timing includes that redundant work.'],
    source_hashes={os.path.relpath(f,ROOT):sha(f) for f in sources},created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
(ROOT/'PROTOCOL.json').write_text(json.dumps(p,indent=2)+'\n')
(ROOT/'REGISTRATION.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_BENCHMARK',protocol_sha256=sha(ROOT/'PROTOCOL.json'),created_utc=p['created_utc']),indent=2)+'\n')
registry_path=REPO/'auto-research/RUN_REGISTRY.json';r=json.loads(registry_path.read_text())
assert all(x['run_id']!=ROOT.name for x in r['runs'])
r['runs'].append(dict(run_id=ROOT.name,status='FROZEN_BEFORE_BENCHMARK',protocol=str((ROOT/'PROTOCOL.json').relative_to(REPO)),protocol_sha256=sha(ROOT/'PROTOCOL.json'),
    metrics=str((ROOT/'SUMMARY.json').relative_to(REPO)),report=f'auto-research/reports/{ROOT.name}.md',publication_target='JRowe47/Super-1 main only'))
registry_path.write_text(json.dumps(r,indent=2)+'\n');print('FROZEN',sha(ROOT/'PROTOCOL.json'))
