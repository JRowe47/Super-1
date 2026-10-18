"""Frozen U-readout factorial analysis; U1-only versus frozen is the sole primary."""
import hashlib, json, resource, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ('backprop_fifo', 'frozen_fifo', 'u0_only_fifo', 'u1_only_fifo', 'both_u_fifo')


def read(name):
    return json.loads((ROOT / name).read_text())


def comparison(results, candidate, control, valid, confirmatory=False):
    rows = []
    for r in results:
        a, b = r['arms'][candidate], r['arms'][control]
        if a['primary_ce'] is None or b['primary_ce'] is None:
            continue
        rows.append({'seed': r['seed'], 'candidate_primary_ce': a['primary_ce'],
                     'control_primary_ce': b['primary_ce'],
                     'relative_ce_reduction_percent': 100 * (b['primary_ce'] - a['primary_ce']) / b['primary_ce'],
                     'old_window_delta_pp': 100 * (a['guard_old_window_acc'] - b['guard_old_window_acc']),
                     'final_old_delta_pp': 100 * (a['final']['stable_acc'] - b['final']['stable_acc'])})
    effects = [x['relative_ce_reduction_percent'] for x in rows]
    effect = len(rows) == 3 and statistics.mean(effects) >= 5 and min(effects) >= 0
    retention = len(rows) == 3 and all(x['old_window_delta_pp'] >= -2 and x['final_old_delta_pp'] >= -2 for x in rows)
    return {'candidate': candidate, 'control': control, 'confirmatory': confirmatory, 'per_seed': rows,
            'mean_relative_ce_reduction_percent': statistics.mean(effects) if effects else None,
            'sample_sd_percent': statistics.stdev(effects) if len(effects) > 1 else None,
            'effect_gate': effect, 'retention_gate': retention, 'combined_gate': bool(valid and effect and retention)}


def main():
    cfg, registration = read('PROTOCOL.json'), read('REGISTRATION.json')
    for name, expected in registration['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    results = [read(f'main/{s}/RESULT.json') for s in cfg['main']['seeds']]
    pilot = read(f"pilot/{cfg['pilot']['seed']}/RESULT.json")
    receipts = [read(s + '_RECEIPT.json') for s in ('VERIFY', 'PILOT', 'MAIN')]
    valid = read('VERIFICATION.json')['status'] == 'PASS' and all(r['status'] == 'COMPLETE' and r['all_children_exited'] for r in receipts)
    keys = {'backprop_fifo': [], 'frozen_fifo': [], 'u0_only_fifo': ['U0'],
            'u1_only_fifo': ['U1'], 'both_u_fifo': ['U0', 'U1']}
    for r in [pilot] + results:
        valid = valid and r['status'] == 'COMPLETE' and set(r['arms']) == set(r['replays']) == set(ARMS)
        steps = 4 * cfg['pilot' if r['pilot'] else 'main']['phase_steps']
        valid = valid and all(a['status'] == 'COMPLETE' and a['completed_steps'] == steps for a in r['arms'].values())
        valid = valid and all(a['all_phase_final_and_shadow_states_bitwise_equal'] for a in r['replays'].values())
        state = ROOT / ('pilot' if r['pilot'] else 'main') / str(r['seed']) / 'STATES.npz'
        valid = valid and hashlib.sha256(state.read_bytes()).hexdigest() == r['states']['sha256']
        valid = valid and r['source_hashes'] == registration['sha256']
        valid = valid and r['storage_examples_per_policy'] == 512 and r['replay_examples_per_step'] == 16
        for arm, a in r['arms'].items():
            active = keys[arm]
            valid = valid and a['learned_feedback_keys'] == active and not a['feedback_decay_enabled']
            valid = valid and a['feedback_learning_enabled'] == bool(active)
            valid = valid and a['learn_u0_enabled'] == ('U0' in active) and a['learn_u1_enabled'] == ('U1' in active)
            valid = valid and a['inactive_feedback_unchanged_bitwise'] and a['final_feedback_max_sigma'] <= 2.
            for family, key in (('B', 'B1'), ('V', 'V1'), ('U0', 'U0'), ('U1', 'U1')):
                if key not in active:
                    valid = valid and a['feedback_change_by_readout'][family] == 0.
    interactions, benefits = [], []
    for r in results:
        a, b, v, f = [r['arms'][name] for name in ('both_u_fifo', 'u1_only_fifo', 'u0_only_fifo', 'frozen_fifo')]
        if any(x['primary_ce'] is None for x in (a, b, v, f)):
            continue
        row = {'seed': r['seed'], 'U1_by_U0_CE_interaction_nats': (a['primary_ce'] - b['primary_ce']) - (v['primary_ce'] - f['primary_ce']),
               'old_window_accuracy_interaction_pp': 100 * ((a['guard_old_window_acc'] - b['guard_old_window_acc']) - (v['guard_old_window_acc'] - f['guard_old_window_acc']))}
        row.update({k + '_interaction_pp': 100 * ((a['final'][k] - b['final'][k]) - (v['final'][k] - f['final'][k]))
                    for k in ('stable_acc', 'novel_acc', 'drift_acc', 'balanced_acc')})
        interactions.append(row)
        gain = f['primary_ce'] - a['primary_ce']
        row = {'seed': r['seed'], 'both_U_gain_nats': gain, 'U1_only_gain_nats': f['primary_ce'] - b['primary_ce'],
               'U0_only_gain_nats': f['primary_ce'] - v['primary_ce'],
               'U1_fraction_of_both_U_gain': (f['primary_ce'] - b['primary_ce']) / gain if gain > 0 else None,
               'U0_fraction_of_both_U_gain': (f['primary_ce'] - v['primary_ce']) / gain if gain > 0 else None,
               'U1_minus_both_U_old_window_pp': 100 * (b['guard_old_window_acc'] - a['guard_old_window_acc']),
               'feedback_displacement': {arm: r['arms'][arm]['feedback_change_by_readout'] for arm in ARMS}}
        row.update({'U1_minus_both_U_' + k + '_pp': 100 * (b['final'][k] - a['final'][k])
                    for k in ('stable_acc', 'novel_acc', 'drift_acc', 'balanced_acc')})
        benefits.append(row)
    components = {}
    for arm in ARMS:
        rows = []
        for r in results:
            p = cfg['main']['phase_steps']; tr = r['arms'][arm]['trace']
            rows.append({'seed': r['seed'], 'novel_window_ce': statistics.mean(x['novel_ce'] for x in tr if p < x['t'] <= p + p // 2),
                         'drift_window_ce': statistics.mean(x['drift_ce'] for x in tr if 2 * p < x['t'] <= 2 * p + p // 2)})
        components[arm] = {'per_seed': rows, 'means': {k: statistics.mean(x[k] for x in rows) for k in ('novel_window_ce', 'drift_window_ce')}}
    final = {arm: {k: statistics.mean(r['arms'][arm]['final'][k] for r in results) for k in results[0]['arms'][arm]['final']} for arm in ARMS}
    usage = resource.getrusage(resource.RUSAGE_SELF); analysis_cpu = usage.ru_utime + usage.ru_stime
    cpu = sum(r['child_cpu_seconds'] + r['launcher_cpu_seconds'] for r in receipts) + read('RUNTIME.json')['runtime_import_canary_cpu_seconds'] + analysis_cpu
    valid = bool(valid and cpu < 900)
    pairs = {'primary': ('u1_only_fifo', 'frozen_fifo'), 'U0_vs_frozen': ('u0_only_fifo', 'frozen_fifo'),
             'both_U_vs_frozen': ('both_u_fifo', 'frozen_fifo'), 'both_U_vs_U1': ('both_u_fifo', 'u1_only_fifo'),
             'both_U_vs_U0': ('both_u_fifo', 'u0_only_fifo')}
    comparisons = {name: comparison(results, a, b, valid, name == 'primary') for name, (a, b) in pairs.items()}
    out = {'run_id': cfg['run_id'], 'numerical_and_replay_valid': valid, 'comparisons': comparisons,
           'factorial_interactions': {'per_seed': interactions, 'means': {k: statistics.mean(x[k] for x in interactions) for k in interactions[0] if k != 'seed'} if interactions else {}, 'confirmatory': False},
           'gain_decomposition': {'per_seed': benefits, 'confirmatory': False, 'interpretation': 'Descriptive readout interventions; nonlinear interactions mean gain fractions are not additive causal shares.'},
           'endpoint_components': components, 'final_means': final,
           'stale_replayed_binary_targets': {str(r['seed']): r['stale_replayed_facts'] for r in results},
           'charged_experiment_cpu_seconds': cpu, 'analysis_cpu_seconds': analysis_cpu,
           'decision': 'JOINT_GATE_PASSED_SYNTHETIC_ONLY' if comparisons['primary']['combined_gate'] else 'NO_PROMOTION',
           'decision_scope': 'Only U1-only versus frozen is confirmatory. A secondary pass cannot replace the primary or establish language/architecture promotion.',
           'preservation_gate': 'Final remote checkpoint readback and evidence audit required before terminal completion.',
           'uncertainty': 'Three fresh paired worlds; every seed, paired mean/sample SD. No population significance or transfer claim.',
           'interpretation_limit': 'Online feedback decay off; full original learning rates; same warmup, forward rule and cap. B and V fixed; U1-only still learns W0 through fixed U0. This is a feedback-readout intervention, not frozen early features. Both-U has the same W/U/B dynamics as preserved V/U-only when B is fixed; verified at steps and 12-step trajectories, not an independent main-world replication.'}
    (ROOT / 'SUMMARY.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'decision': out['decision'], 'primary': comparisons['primary'], 'cpu_seconds': cpu}, indent=2))


if __name__ == '__main__':
    main()
