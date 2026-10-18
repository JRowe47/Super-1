"""Post-run arithmetic and provenance audit; does not rerun or select measurements."""
import hashlib
import json
import math
import statistics
from pathlib import Path

H = Path(__file__).resolve().parent
C = json.loads((H / 'CONFIG.json').read_text())
O = H / 'execution'
M = json.loads((O / 'MEMORY.json').read_text())
R = json.loads((O / 'RESULT.json').read_text())
records = M['records']
expected = {(arm, seed, context) for arm in C['patterns'] for seed in C['seeds'] for context in C['memory_contexts']}
assert len(records) == len(expected)
assert {(r['arm'], r['seed'], r['context']) for r in records} == expected
for r in records:
    arm, seed, context = r['arm'], r['seed'], r['context']
    endpoint = next(row for row in R['rows'] if (row['arm'], row['seed']) == (arm, seed))
    assert r['checkpoint_sha256'] == endpoint['inference_sha256']
    assert hashlib.sha256((O / endpoint['inference_checkpoint']).read_bytes()).hexdigest() == r['checkpoint_sha256']
    pattern = C['patterns'][arm]
    matrix_bytes = C['heads'] * (C['width'] // C['heads']) ** 2 * 4
    for field, length in [('prefill_state_bytes', context), ('decode_state_bytes', context + C['decode_steps'])]:
        assert r[field] == pattern.count('A') * 2 * length * C['width'] * 4 + pattern.count('D') * matrix_bytes
    assert r['model_bytes'] == endpoint['parameter_count'] * 4
    assert r['retained_model_plus_state_bytes'] == r['model_bytes'] + r['decode_state_bytes']
    assert r['process_peak_rss_bytes'] >= r['prefill_peak_rss_bytes'] >= r['loaded_peak_rss_bytes'] >= r['startup_peak_rss_bytes']
    for field in ['process_peak_rss_bytes', 'retained_model_plus_state_bytes', 'prefill_seconds', 'decode_ms_per_token']:
        assert math.isfinite(r[field]) and r[field] > 0
        baseline = next(x for x in records if (x['arm'], x['seed'], x['context']) == ('transformer', seed, context))
        index = C['seeds'].index(seed)
        assert r[field] / baseline[field] == M['summary'][str(context)][arm]['paired_ratios'][field][index]
for context in C['memory_contexts']:
    for arm in C['patterns']:
        summary = M['summary'][str(context)][arm]
        for key, ratios in summary['paired_ratios'].items():
            assert statistics.mean(ratios) == summary['mean_ratios'][key]
primary = M['summary'][str(max(C['memory_contexts']))]['relaxed']
assert M['total_memory_pass'] == (max(primary['paired_ratios']['process_peak_rss_bytes']) <= C['memory_target_ratio'])
assert M['latency_pass'] == (max(primary['mean_ratios'][k] for k in ('prefill_seconds', 'decode_ms_per_token')) <= C['latency_max_ratio'])
assert M['combined_gate'] == (M['total_memory_pass'] and M['latency_pass'] and R['relaxed_quality_noninferiority_pass'])
receipt = {'status': 'PASS', 'isolated_measurements': len(records), 'checkpoint_hashes': True, 'exact_state_and_model_bytes': True, 'ratio_arithmetic': True, 'gate_arithmetic': True, 'scope': 'Audit of stored single measurements; no new timing samples or quality claims.'}
with (O / 'MEMORY_VERIFICATION.json').open('x') as f:
    json.dump(receipt, f, indent=2)
    f.write('\n')
print(json.dumps(receipt))
