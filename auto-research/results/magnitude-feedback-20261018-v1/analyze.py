"""Frozen descriptive analysis; all seeds retained, one primary contrast."""
import hashlib, json, math, statistics, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    begin = time.process_time()
    cfg = json.loads((ROOT/'PROTOCOL.json').read_text())
    results = [json.loads((ROOT/'main'/str(seed)/'RESULT.json').read_text()) for seed in cfg['main']['seeds']]
    receipt_paths = [ROOT/(s+'_RECEIPT.json') for s in ('VERIFY', 'PILOT', 'MAIN')]
    receipts = [json.loads(p.read_text()) for p in receipt_paths]
    valid = all(r['status'] == 'COMPLETE' for r in receipts)
    valid = valid and all(a['status'] == 'COMPLETE' for r in results for a in r['arms'].values())
    valid = valid and all(a['all_phase_final_and_shadow_states_bitwise_equal'] for r in results for a in r['replays'].values())
    for r in results:
        for arm in ('frozen_match_adaptive', 'adaptive_match_frozen'):
            valid = valid and all(m['count'] == cfg['main']['phase_steps']*4 and
                                  m['max_relative_norm_error'] <= 2e-6 for m in r['arms'][arm]['magnitude_audit'].values())
        path = ROOT/'main'/str(r['seed'])/'STATES.npz'
        valid = valid and hashlib.sha256(path.read_bytes()).hexdigest() == r['states']['sha256']
    comparisons = {}
    for label, candidate, control in (
            ('primary', 'adaptive', 'frozen_match_adaptive'),
            ('unmatched_secondary', 'adaptive', 'frozen'),
            ('reciprocal_secondary', 'adaptive_match_frozen', 'frozen')):
        rows = []
        for r in results:
            a, f = r['arms'][candidate], r['arms'][control]
            if a['primary_ce'] is None or f['primary_ce'] is None:
                continue
            rows.append({'seed': r['seed'], 'candidate_primary_ce': a['primary_ce'], 'control_primary_ce': f['primary_ce'],
                         'relative_ce_reduction_percent': 100*(f['primary_ce']-a['primary_ce'])/f['primary_ce'],
                         'old_window_delta_pp': 100*(a['guard_old_window_acc']-f['guard_old_window_acc']),
                         'final_old_delta_pp': 100*(a['final']['stable_acc']-f['final']['stable_acc'])})
        values = [r['relative_ce_reduction_percent'] for r in rows]
        effect = len(rows) == 3 and statistics.mean(values) >= 5 and min(values) >= 0
        retention = len(rows) == 3 and all(r['old_window_delta_pp'] >= -2 and r['final_old_delta_pp'] >= -2 for r in rows)
        comparisons[label] = {'candidate': candidate, 'control': control, 'per_seed': rows,
                              'mean_relative_ce_reduction_percent': statistics.mean(values) if values else None,
                              'sample_sd_percent': statistics.stdev(values) if len(values)>1 else None,
                              'effect_gate': effect, 'retention_gate': retention,
                              'combined_gate': bool(valid and effect and retention),
                              'confirmatory': label == 'primary'}
    final_means = {arm: {metric: statistics.mean(r['arms'][arm]['final'][metric] for r in results)
                         for metric in results[0]['arms'][arm]['final']} for arm in results[0]['arms']}
    cpu = sum(r['child_cpu_seconds']+r['launcher_cpu_seconds'] for r in receipts)
    out = {'run_id': cfg['run_id'], 'numerical_and_replay_valid': bool(valid), 'comparisons': comparisons,
           'final_means': final_means, 'execution_cpu_seconds': cpu,
           'within_900_cpu_seconds': cpu < 900, 'analysis_cpu_seconds': time.process_time()-begin,
           'decision': 'GATE_PASSED_SYNTHETIC_ONLY' if valid and comparisons['primary']['combined_gate'] else 'NO_PROMOTION',
           'uncertainty': 'Three paired synthetic worlds; descriptive mean, sample SD and every seed. No population significance or language-model claim.',
           'interpretation_limit': 'Matching is per hidden layer at the control own current forward state; trajectories may diverge. A shadow bank supplies magnitudes and adds compute/memory. Reciprocal contrast is secondary.'}
    (ROOT/'SUMMARY.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
