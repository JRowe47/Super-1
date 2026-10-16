"""Independent evidence validation; no torch, optimization, or official data."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
p = read(HERE/'PROTOCOL.json'); c = p['config']; out = HERE/'execution'
for path, digest in read(HERE/'SOURCE_MANIFEST.json').items():
    assert sha(ROOT/path) == digest, path
spec = importlib.util.spec_from_file_location('generator', HERE/'generate_data.py')
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
data = gen.generate(c)
assert read(HERE/'DATA.json') == data
for split, rows in data.items():
    assert hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest() == p['data'][split+'_sha256']
    for batch in rows:
        for doc in batch:
            state = doc[0]
            for i, token in enumerate(doc[1:], 1):
                if i % (c['commands_per_group'] + 1):
                    state = (state + token - c['modulus']) % c['modulus']
                else:
                    assert token == state
terminal = read(out/'TERMINAL.json')
assert terminal['status'] == 'COMPLETED_SYNTHETIC_EXPERIMENT'
assert terminal['failures'] == [] and terminal['sources_unchanged']
assert terminal['runtime']['intraop_threads'] == terminal['runtime']['interop_threads'] == 1
assert terminal['runtime']['device'] == 'cpu'
freeze = read(out/'CHECKPOINT_FREEZE.json')
assert len(freeze['rows']) == len(c['arms'])*len(c['seeds'])
progress = [json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()]
assert len(progress) == p['training']['updates']*len(freeze['rows'])
rows = {}
for frozen in freeze['rows']:
    seed, arm = frozen['seed'], frozen['arm']
    row = read(out/f'{seed}-{arm}.json'); rows[seed,arm] = row
    assert sha(out/frozen['checkpoint']) == frozen['checkpoint_sha256'] == row['checkpoint_sha256']
    assert row['freeze_sha256_before_confirmation'] == sha(out/'CHECKPOINT_FREEZE.json')
    assert row['confirmation'] == row['duplicate_confirmation']
    assert row['confirmation']['targets'] == c['split_batches']['confirmation']*c['batch_size']*c['readout_groups']
    assert row['confirmation']['all_targets'] == c['split_batches']['confirmation']*c['batch_size']*c['sequence_length']
    assert row['updates'] == p['training']['updates']
    assert row['target_exposures'] == p['training']['target_exposures_per_trial']
    records = [r for r in progress if r['seed'] == seed and r['arm'] == arm]
    assert [r['update'] for r in records] == list(range(1,row['updates']+1))
    assert all(math.isfinite(r['train_ce']) and math.isfinite(r['preclip_grad_norm']) for r in records)
    assert [r['targets'] for r in records] == [i*p['training']['tokens_per_update'] for i in range(1,row['updates']+1)]
    if arm in c['geometry_arms']:
        assert row['nonzero_gain_gradient_l1'] > 0 and row['learned_gain_l1'] > 0 and row['learned_phase_l1'] > 0
    if arm != 'pinned_nanogpt': assert row['identity_and_streaming_exact']
result = read(out/'RESULT.json'); gaps = []; native_gaps=[]
for seed in c['seeds']:
    assert rows[seed,'rotation']['parameters'] == rows[seed,'radial']['parameters']
    ce = {a: rows[seed,a]['confirmation']['ce_nats_per_target'] for a in c['arms']}
    saved = next(r for r in result['paired'] if r['seed'] == seed)
    assert saved['ce'] == ce
    gap = ce['rotation'] - ce['radial']; ngap=ce['rotation']-ce['native']
    assert saved['rotation_minus_radial'] == gap and saved['rotation_minus_native'] == ngap
    gaps.append(gap); native_gaps.append(ngap)
mean = statistics.mean(gaps)
half = c['t_critical_95_df2']*statistics.stdev(gaps)/math.sqrt(len(gaps))
assert result['mean_rotation_minus_radial'] == mean
assert result['paired_t_95_interval'] == [mean-half,mean+half]
passed = all(g < 0 for g in gaps) and mean <= -c['minimum_effect_nats'] and mean+half < 0 and all(g < 0 for g in native_gaps)
assert result['primary_gate_passed'] == passed
assert result['decision'] == ('SYNTHETIC_CRITERION_MET_NO_LANGUAGE_PROMOTION' if passed else 'NO_PROMOTION_INCONCLUSIVE_OR_NEGATIVE')
assert terminal['cpu_seconds'] < c['cpu_hard_seconds']
assert sum(f.stat().st_size for f in HERE.rglob('*') if f.is_file()) < c['max_artifact_bytes']
print(json.dumps({'status':'PASS','arm_seed_rows':len(rows),'updates':len(progress),'regenerated_data_exact':True,'duplicate_confirmations_exact':len(rows),'decision':result['decision']},sort_keys=True))
