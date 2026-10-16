"""Reconcile all seven trials, replay final losses, retain complete checkpoints."""
import argparse, gzip, hashlib, json, math, sys, time, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'gru-batch-wall-20261016-v1'))
from experiment import build, evaluate, save, sha
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, required=True)
parser.add_argument('--archive', type=Path, required=True)
args = parser.parse_args()
p = json.loads((ROOT/'PROTOCOL.json').read_text())
for name, digest in p['source_hashes'].items(): assert sha(ROOT/name) == digest, name
for name, digest in p['data_hashes'].items(): assert sha(args.data_dir/name) == digest, name
assert sha(args.data_dir/'tokenizer.json') == p['tokenizer_sha256']
train = np.memmap(args.data_dir/'train.bin', dtype=np.uint16, mode='r')
val = np.memmap(args.data_dir/'validation.bin', dtype=np.uint16, mode='r')
starts = np.random.default_rng(p['data_seed']).integers(0, len(train)-512, size=p['max_windows'], dtype=np.int64)
assert hashlib.sha256(starts.tobytes()).hexdigest() == p['train_stream_sha256']
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
rows, failures, checks, manifest = [], [], {}, {}
for q in p['runs']:
    rid = q['run_id']; out = ROOT/'execution'/rid; cp = out/'checkpoint.pt'
    if (out/'FAILURE.json').exists():
        failure = json.loads((out/'FAILURE.json').read_text()); failures.append(failure)
        if cp.exists(): manifest[rid] = dict(path=str(cp.relative_to(ROOT)), sha256=sha(cp), bytes=cp.stat().st_size)
        continue
    r = json.loads((out/'RESULT.json').read_text())
    assert all(r[k] == v for k, v in q.items())
    assert r['protocol_sha256'] == sha(ROOT/'PROTOCOL.json') and sha(cp) == r['checkpoint_sha256']
    old = torch.load(cp, map_location='cpu', weights_only=False)
    assert old['protocol_sha256'] == r['protocol_sha256'] and old['run_id'] == rid
    c = old['state']; assert c == json.loads((out/'curves.json').read_text())
    assert len(c['train']) == r['updates'] == c['step']
    assert c['windows'] == r['updates'] * q['batch']
    assert r['training_targets'] == c['training_targets'] == c['windows'] * 512
    elapsed = 0.; cpu = 0.
    for i, step in enumerate(c['train']):
        progress = min(1., max(0., elapsed/p['train_wall_seconds']))
        factor = max(.04, progress/.05) if progress < .05 else .1+.9*.5*(1+math.cos(math.pi*(progress-.05)/.95))
        assert abs(step['lr'] - q['peak_lr'] * factor) < 1e-14
        assert step['step'] == i+1 and step['training_targets'] == (i+1)*q['batch']*512
        assert all(math.isfinite(step[k]) for k in ('ce', 'gradient_norm', 'wall', 'cpu', 'lr'))
        assert step['wall'] > 0 and step['cpu'] > 0
        elapsed += step['wall']; cpu += step['cpu']
        assert abs(step['train_wall_seconds']-elapsed) < 1e-7
        assert abs(step['train_cpu_seconds']-cpu) < 1e-7
    assert abs(elapsed-r['train_wall_seconds']) < 1e-7 and abs(cpu-r['train_cpu_seconds']) < 1e-7
    unused = p['train_wall_seconds'] - elapsed
    assert 0 <= unused <= max(.2, 2*max(x['wall'] for x in c['train'][-10:])) + 1e-6
    m = build('mingru_linear', p); m.load_state_dict(old['model'])
    assert sum(v.numel() for v in m.parameters()) == r['parameters'] == 996320
    ev = evaluate(m, 'mingru_linear', val, p['final_starts'])
    error = max(abs(a-b) for a,b in zip(ev['window_ce'], r['final']['window_ce']))
    assert error < 5e-6, (rid, error)
    assert abs(np.mean(r['final']['window_ce']) - r['final']['ce']) < 1e-10
    checks[rid] = dict(final_checkpoint_replay_max_ce_error=error, trace_time_targets_and_lr_reconciled=True,
        budget_unused_seconds=unused, finite_updates=True)
    row = {k:r[k] for k in ('run_id','batch','threads','peak_lr','parameters','updates','training_targets',
        'train_wall_seconds','train_cpu_seconds','run_wall_seconds','run_cpu_seconds','peak_rss_kib')}
    row.update(ce=r['final']['ce'], perplexity=r['final']['perplexity'], tokens_per_wall_second=r['training_targets']/elapsed,
        mean_cpu_cores=cpu/elapsed, max_gradient_norm=max(x['gradient_norm'] for x in c['train']))
    rows.append(row)
    manifest[rid] = dict(path=str(cp.relative_to(ROOT)), sha256=sha(cp), bytes=cp.stat().st_size)
    (out/'curves.json.gz').write_bytes(gzip.compress((out/'curves.json').read_bytes(), mtime=0))
    print('CHECKED', rid, row['ce'], flush=True)

by_id = {r['run_id']:r for r in rows}
pairs = [('b1_t2_lr002','b1_t1_lr002'), ('b1_t4_lr002','b1_t1_lr002'),
    ('b4_t2_lr004','b4_t2_lr002'), ('b16_t4_lr008','b16_t4_lr002')]
contrasts = []
for trial, control in pairs:
    if trial not in by_id or control not in by_id: continue
    t, c = by_id[trial], by_id[control]
    contrasts.append(dict(trial=trial, control=control, ce_change=t['ce']-c['ce'],
        perplexity_change_percent=100*(t['perplexity']/c['perplexity']-1),
        throughput_ratio=t['tokens_per_wall_second']/c['tokens_per_wall_second'],
        update_ratio=t['updates']/c['updates'], cpu_ratio=t['train_cpu_seconds']/c['train_cpu_seconds']))
previous = json.loads((ROOT.parent/'gru-batch-wall-20261016-v1'/'SUMMARY.json').read_text())
drift = []
for rid, old_rid in [('b1_t1_lr002','mingru_linear_baseline'), ('b16_t4_lr002','mingru_linear_throughput')]:
    if rid in by_id:
        r = by_id[rid]; old = next(x for x in previous['rows'] if x['run_id'] == old_rid)
        drift.append(dict(run_id=rid, previous_ce=old['ce'], fresh_ce=r['ce'], ce_change=r['ce']-old['ce'],
            throughput_ratio=r['tokens_per_wall_second']/old['tokens_per_wall_second']))
summary = dict(run_id=ROOT.name, status='COMPLETE_EXPLORATORY', seed=p['seed'], rows=sorted(rows,key=lambda x:x['ce']),
    failures=failures, contrasts=contrasts, fresh_control_vs_previous=drift, final_targets=16384,
    train_wall_budget_seconds=p['train_wall_seconds'], total_training_wall_seconds=sum(r['train_wall_seconds'] for r in rows),
    total_training_cpu_seconds=sum(r['train_cpu_seconds'] for r in rows),
    publication=dict(repository='JRowe47/Super-1', base='main', upstream_submission_permitted=False))
save(ROOT/'SUMMARY.json', summary); save(ROOT/'CHECKS.json', checks); save(ROOT/'CHECKPOINT_MANIFEST.json', manifest)
save(ROOT/'TERMINAL.json', dict(status='COMPLETE', completed_runs=len(rows), failed_runs=len(failures),
    all_declared_trials_accounted_for=len(rows)+len(failures)==len(p['runs']), checkpoint_replay_passed=True,
    training_running=False, official_test_accessed=False, new_paid_resources=False,
    completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
assert not args.archive.exists()
args.archive.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for v in manifest.values(): z.write(ROOT/v['path'], v['path'])
    z.writestr('CHECKPOINTS.json', json.dumps(manifest,indent=2)+'\n')
    z.writestr('README.txt', 'Extract under auto-research/results/mingru-thread-lr-20261016-v1/. Complete model, AdamW, RNG and training traces. Source/results are in JRowe47/Super-1. Exact data remains in gru_extended_checkpoints_20261016.zip, SHA256 c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6.\n')
with zipfile.ZipFile(args.archive) as z:
    assert z.testzip() is None
    for v in manifest.values(): assert hashlib.sha256(z.read(v['path'])).hexdigest() == v['sha256']
save(ROOT/'PRESERVATION.json', dict(archive=args.archive.name, bytes=args.archive.stat().st_size, sha256=sha(args.archive),
    status='VERIFIED', contents='Complete final model/optimizer/RNG/traces; Git-backed files and previously saved exact data not duplicated',
    exact_data_archive='gru_extended_checkpoints_20261016.zip',
    exact_data_sha256='c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6'))
regpath = ROOT.parents[2]/'auto-research/RUN_REGISTRY.json'
reg = json.loads(regpath.read_text()); entry = next(x for x in reg['runs'] if x['run_id'] == ROOT.name)
entry.update(status='COMPLETE_EXPLORATORY', completed_runs=len(rows), failed_runs=len(failures),
    summary={r['run_id']:r['ce'] for r in rows})
save(regpath, reg)
print(json.dumps(summary, indent=2))
