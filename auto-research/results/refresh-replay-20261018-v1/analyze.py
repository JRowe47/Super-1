"""Frozen nine-arm analysis; adaptive/frozen refreshed is the sole primary."""
import hashlib, json, resource, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ('fifo', 'mixed', 'refreshed')
POLICIES = ('backprop', 'frozen', 'adaptive')
ARMS = {p+'_'+m for p in POLICIES for m in MODES}


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


def within(results, candidate, control):
    out = {}
    for policy in POLICIES:
        rows = []
        for r in results:
            a, b = r['arms'][policy+'_'+candidate], r['arms'][policy+'_'+control]
            if a['primary_ce'] is None or b['primary_ce'] is None:
                continue
            row = {'seed': r['seed'], 'primary_ce_delta_nats': a['primary_ce']-b['primary_ce'],
                   'relative_ce_reduction_percent': 100*(b['primary_ce']-a['primary_ce'])/b['primary_ce'],
                   'old_window_delta_pp': 100*(a['guard_old_window_acc']-b['guard_old_window_acc'])}
            row.update({k+'_delta_pp': 100*(a['final'][k]-b['final'][k])
                        for k in ('stable_acc', 'novel_acc', 'drift_acc', 'balanced_acc')})
            rows.append(row)
        out[policy] = {'per_seed': rows,
                       'means': {k: statistics.mean(x[k] for x in rows) for k in rows[0] if k != 'seed'} if rows else {},
                       'confirmatory': False}
    return out


def main():
    cfg, registration = read('PROTOCOL.json'), read('REGISTRATION.json')
    for name, expected in registration['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    results = [read(f'main/{s}/RESULT.json') for s in cfg['main']['seeds']]
    pilot = read(f"pilot/{cfg['pilot']['seed']}/RESULT.json")
    receipts = [read(s+'_RECEIPT.json') for s in ('VERIFY', 'PILOT', 'MAIN')]
    valid = read('VERIFICATION.json')['status'] == 'PASS' and all(r['status'] == 'COMPLETE' and r['all_children_exited'] for r in receipts)
    for r in [pilot]+results:
        valid = valid and set(r['arms']) == set(r['replays']) == ARMS
        steps = 4*cfg['pilot' if r['pilot'] else 'main']['phase_steps']
        valid = valid and all(a['status'] == 'COMPLETE' and a['completed_steps'] == steps for a in r['arms'].values())
        valid = valid and all(a['all_phase_final_and_shadow_states_bitwise_equal'] for a in r['replays'].values())
        path = ROOT/('pilot' if r['pilot'] else 'main')/str(r['seed'])/'STATES.npz'
        valid = valid and hashlib.sha256(path.read_bytes()).hexdigest() == r['states']['sha256']
        valid = valid and r['source_hashes'] == registration['sha256']
        valid = valid and r['storage_examples_per_policy'] == 512 and r['replay_examples_per_step'] == 16
        valid = valid and r['refreshed_sampled_bank_age_max'] <= 256 and r['late_refreshed_stale_targets'] == 0
    contrasts = {a+'_minus_'+b: within(results, a, b) for a,b in
                 (('refreshed','mixed'), ('refreshed','fifo'), ('mixed','fifo'))}
    interactions = {}
    for contrast, policies in contrasts.items():
        rows = []
        for a, f in zip(policies['adaptive']['per_seed'], policies['frozen']['per_seed']):
            assert a['seed'] == f['seed']
            rows.append({'seed': a['seed'], **{k: a[k]-f[k] for k in
                         ('primary_ce_delta_nats', 'old_window_delta_pp', 'stable_acc_delta_pp')}})
        interactions[contrast] = rows
    components = {}
    for arm in sorted(ARMS):
        rows = []
        for r in results:
            p = cfg['main']['phase_steps']
            tr = r['arms'][arm]['trace']
            rows.append({'seed': r['seed'],
                         'novel_window_ce': statistics.mean(x['novel_ce'] for x in tr if p<x['t']<=p+p//2),
                         'drift_window_ce': statistics.mean(x['drift_ce'] for x in tr if 2*p<x['t']<=2*p+p//2)})
        components[arm] = {'per_seed': rows, 'means': {k:statistics.mean(x[k] for x in rows) for k in ('novel_window_ce','drift_window_ce')}}
    final_means = {arm: {k: statistics.mean(r['arms'][arm]['final'][k] for r in results)
                         for k in results[0]['arms'][arm]['final']} for arm in sorted(ARMS)}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    analysis_cpu = usage.ru_utime+usage.ru_stime
    cpu = sum(r['child_cpu_seconds']+r['launcher_cpu_seconds'] for r in receipts)
    cpu += read('RUNTIME.json')['runtime_import_canary_cpu_seconds'] + analysis_cpu
    valid = bool(valid and cpu < 900)
    comparisons = {'primary': comparison(results,'adaptive_refreshed','frozen_refreshed',valid,True),
                   'permanent_secondary': comparison(results,'adaptive_mixed','frozen_mixed',valid),
                   'fifo_secondary': comparison(results,'adaptive_fifo','frozen_fifo',valid)}
    out = {'run_id': cfg['run_id'], 'numerical_and_replay_valid': valid,
           'comparisons': comparisons, 'replay_contrasts': contrasts,
           'adaptive_minus_frozen_replay_interactions': interactions,
           'endpoint_components': components, 'final_means': final_means,
           'stale_replayed_binary_targets': {str(r['seed']): r['stale_replayed_facts'] for r in results},
           'charged_experiment_cpu_seconds': cpu, 'analysis_cpu_seconds': analysis_cpu,
           'decision': 'GATE_PASSED_SYNTHETIC_ONLY' if comparisons['primary']['combined_gate'] else 'NO_PROMOTION',
           'preservation_gate': 'Final remote checkpoint readback and evidence audit required separately before terminal completion.',
           'uncertainty': 'Three fresh paired worlds; every seed, descriptive paired mean and sample SD. No population significance or language-model claim.',
           'interpretation_limit': 'Refreshed versus permanent bank isolates one causal replacement policy at equal allocation/storage. It changes content and age together, not stale labels alone. FIFO also differs in allocation and recent horizon. Only refreshed adaptive versus frozen is primary.'}
    (ROOT/'SUMMARY.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'decision': out['decision'], 'primary': comparisons['primary'], 'cpu_seconds':cpu}, indent=2))


if __name__ == '__main__':
    main()
