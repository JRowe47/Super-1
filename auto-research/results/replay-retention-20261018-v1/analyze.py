"""Frozen analysis of the replay factorial; one primary comparison."""
import hashlib, json, statistics, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text())


def comparison(results, candidate, control, valid, confirmatory=False):
    rows = []
    for r in results:
        a, b = r['arms'][candidate], r['arms'][control]
        if a['primary_ce'] is None or b['primary_ce'] is None:
            continue
        rows.append({'seed': r['seed'], 'candidate_primary_ce': a['primary_ce'],
                     'control_primary_ce': b['primary_ce'],
                     'relative_ce_reduction_percent': 100*(b['primary_ce']-a['primary_ce'])/b['primary_ce'],
                     'old_window_delta_pp': 100*(a['guard_old_window_acc']-b['guard_old_window_acc']),
                     'final_old_delta_pp': 100*(a['final']['stable_acc']-b['final']['stable_acc'])})
    effects = [x['relative_ce_reduction_percent'] for x in rows]
    effect = len(rows) == 3 and statistics.mean(effects) >= 5 and min(effects) >= 0
    retention = len(rows) == 3 and all(x['old_window_delta_pp'] >= -2 and x['final_old_delta_pp'] >= -2 for x in rows)
    return {'candidate': candidate, 'control': control, 'confirmatory': confirmatory,
            'per_seed': rows, 'mean_relative_ce_reduction_percent': statistics.mean(effects) if effects else None,
            'sample_sd_percent': statistics.stdev(effects) if len(effects)>1 else None,
            'effect_gate': effect, 'retention_gate': retention,
            'combined_gate': bool(valid and effect and retention)}


def main():
    begin = time.process_time()
    cfg = read('PROTOCOL.json')
    results = [read(f'main/{s}/RESULT.json') for s in cfg['main']['seeds']]
    receipts = [read(s+'_RECEIPT.json') for s in ('VERIFY', 'PILOT', 'MAIN')]
    registration = read('REGISTRATION.json')
    valid = read('VERIFICATION.json')['status'] == 'PASS' and all(r['status'] == 'COMPLETE' for r in receipts)
    valid = valid and all(a['status'] == 'COMPLETE' and a['completed_steps'] == 2048 for r in results for a in r['arms'].values())
    valid = valid and all(a['all_phase_final_and_shadow_states_bitwise_equal'] for r in results for a in r['replays'].values())
    for r in results:
        path = ROOT/'main'/str(r['seed'])/'STATES.npz'
        valid = valid and hashlib.sha256(path.read_bytes()).hexdigest() == r['states']['sha256']
        valid = valid and r['source_hashes'] == registration['sha256']
        valid = valid and r['storage_examples_per_policy'] == 512 and r['replay_examples_per_step'] == 16
    comparisons = {'primary': comparison(results, 'adaptive_mixed', 'frozen_mixed', valid, True),
                   'fifo_secondary': comparison(results, 'adaptive_fifo', 'frozen_fifo', valid)}
    within = {}
    for policy in ('backprop', 'frozen', 'adaptive'):
        rows = []
        for r in results:
            m, f = r['arms'][policy+'_mixed'], r['arms'][policy+'_fifo']
            if m['primary_ce'] is None or f['primary_ce'] is None:
                continue
            row = {'seed': r['seed'], 'primary_ce_delta_nats': m['primary_ce']-f['primary_ce'],
                   'relative_ce_reduction_percent': 100*(f['primary_ce']-m['primary_ce'])/f['primary_ce'],
                   'old_window_delta_pp': 100*(m['guard_old_window_acc']-f['guard_old_window_acc'])}
            row.update({k+'_delta_pp': 100*(m['final'][k]-f['final'][k])
                        for k in ('stable_acc', 'novel_acc', 'drift_acc', 'balanced_acc')})
            rows.append(row)
        within[policy] = {'per_seed': rows,
                          'means': {k: statistics.mean(x[k] for x in rows) for k in rows[0] if k != 'seed'} if rows else {},
                          'confirmatory': False}
    interactions = []
    for a, f in zip(within['adaptive']['per_seed'], within['frozen']['per_seed']):
        assert a['seed'] == f['seed']
        interactions.append({'seed': a['seed'], **{k: a[k]-f[k] for k in
                            ('primary_ce_delta_nats', 'old_window_delta_pp', 'stable_acc_delta_pp')}})
    final_means = {arm: {k: statistics.mean(r['arms'][arm]['final'][k] for r in results)
                         for k in results[0]['arms'][arm]['final']} for arm in results[0]['arms']}
    cpu = sum(r['child_cpu_seconds']+r['launcher_cpu_seconds'] for r in receipts)
    cpu += read('RUNTIME.json')['runtime_import_canary_cpu_seconds']
    analysis_cpu = time.process_time()-begin
    cpu += analysis_cpu
    valid = bool(valid and cpu < 900)
    primary = comparisons['primary']
    out = {'run_id': cfg['run_id'], 'numerical_and_replay_valid': valid,
           'comparisons': comparisons, 'mixed_minus_fifo': within,
           'adaptive_minus_frozen_replay_interaction': interactions,
           'final_means': final_means,
           'stale_replayed_binary_targets': {str(r['seed']): r['stale_replayed_facts'] for r in results},
           'charged_experiment_cpu_seconds': cpu, 'analysis_cpu_seconds': analysis_cpu,
           'decision': 'GATE_PASSED_SYNTHETIC_ONLY' if valid and primary['combined_gate'] else 'NO_PROMOTION',
           'uncertainty': 'Three paired worlds; every seed, descriptive paired mean and sample SD. No population significance or language-model claim.',
           'interpretation_limit': 'Equal replay storage and sample count. Mixed replay changes both retention age and data allocation: 256 recent plus256 pinned old versus512 recent. Historical labels may conflict with changed rules. Only the mixed adaptive-versus-frozen contrast is primary.'}
    (ROOT/'SUMMARY.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
