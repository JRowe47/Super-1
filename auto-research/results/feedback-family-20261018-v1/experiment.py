"""Feedback-family factorial; unchanged forward rule and causal FIFO."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, platform, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import core as C

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, dict):
            for k in sorted(v):
                h.update(k.encode() + b'\0')
                visit(v[k])
        elif isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        else:
            h.update(json.dumps(v, sort_keys=True).encode())
    visit(value)
    return h.hexdigest()


def save_states(path, banks):
    arrays = {bank + '/' + k: v.detach().cpu().numpy() for bank, state in banks.items()
              for k, v in state.items()}
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def load_states(path):
    result = {}
    with np.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            bank, k = key.split('/')
            result.setdefault(bank, {})[k] = torch.from_numpy(arrays[key].copy())
    return result


def assert_frozen():
    registration = json.loads((ROOT / 'REGISTRATION.json').read_text())
    for name, expected in registration['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def finite(state):
    return all(bool(torch.isfinite(v).all()) for v in state.values())


def start_for(arm, warm):
    return C.clone(warm), {}


def probe(state, reference, x, y, step):
    """Same-state directions, training batch; exact hidden grads only diagnostic."""
    exact = C.exact_grads(state, x, y)
    active = C.local_grads(state, x, y, feedback=False)
    fixed = C.local_grads(C.with_feedback(state, reference), x, y, feedback=False)
    out = {'step': step, 'layers': {}}
    for k in ('W0', 'W1'):
        e, a, f = (g[k] + C.WD_WB * state[k] for g in (exact, active, fixed))
        out['layers'][k] = {'active_exact_cosine': C.cosine(a, e),
                            'frozen_exact_cosine': C.cosine(f, e),
                            'active_frozen_cosine': C.cosine(a, f),
                            'active_frozen_norm_ratio': float(a.norm() / f.norm().clamp_min(1e-30))}
    return out


def run_arm(start, shadow, reference, data, arm, measure=True):
    state, shadow = C.clone(start), C.clone(shadow)
    policy, replay_mode = arm.rsplit('_', 1)
    p = data['phase_steps']
    trace = [{'t': 0, **C.evaluate(state, data, False)}] if measure else []
    phases, probes = {}, []
    prequential = [{'ce_sum': 0., 'acc_sum': 0., 'n': 0} for _ in range(4)]
    status, failure, completed = 'COMPLETE', None, 0
    begin = time.process_time()
    if measure and policy in ('b_only','vu_only','learning'):
        probes.append(probe(state, reference, data['train_x'][0], data['train_y'][0], 0))
    for t, (x, y) in enumerate(zip(data['train_x'], data['train_y'])):
        if measure:
            with torch.no_grad():
                z = C.forward(state, data['current_x'][t])
                row = prequential[t // p]
                row['ce_sum'] += float(F.binary_cross_entropy_with_logits(z, data['current_y'][t]))
                row['acc_sum'] += float(((z > 0) == data['current_y'][t].bool()).float().mean())
                row['n'] += 1
        try:
            candidate, next_shadow, audit = C.policy_step(state, x, y, policy, reference, shadow)
            if not finite(candidate) or not finite(next_shadow):
                raise FloatingPointError('Nonfinite state')
        except (FloatingPointError, RuntimeError) as exc:
            status, failure = 'NUMERICAL_FAILURE', {'attempted_step': t + 1, 'error': repr(exc)}
            break
        state, shadow, completed = candidate, next_shadow, t + 1
        if measure:
            if completed % 32 == 0:
                trace.append({'t': completed, **C.evaluate(state, data, completed >= 2 * p)})
        if completed % p == 0:
            phases[str(completed)] = digest({'state': state, 'shadow': shadow})
            if measure and policy in ('b_only','vu_only','learning'):
                i = min(completed, len(data['train_x']) - 1)
                probes.append(probe(state, reference, data['train_x'][i], data['train_y'][i], completed))
    active = C.FAMILY_KEYS.get(policy, ())
    assert all(torch.equal(state[k], start[k]) for k in C.FK if k not in active)
    final_hash = digest({'state': state, 'shadow': shadow})
    result = {'arm': arm, 'status': status, 'failure': failure, 'completed_steps': completed,
              'phase_hashes': phases, 'final_state_hash': final_hash,
              'cpu_seconds': time.process_time() - begin}
    if measure:
        win = [r for r in trace if p < r['t'] <= p + p // 2 or 2*p < r['t'] <= 2*p + p//2]
        primary = [r['novel_ce'] if r['t'] < 2*p else r['drift_ce'] for r in win]
        result.update(final=C.evaluate(state, data, completed >= 2*p),
                      primary_ce=float(np.mean(primary)) if status == 'COMPLETE' else None,
                      guard_old_window_acc=float(np.mean([r['stable_acc'] for r in win])) if status == 'COMPLETE' else None,
                      trace=trace, same_state_probes=probes,
                      replay_policy=replay_mode, learning_policy=policy, prequential_phase_sums=prequential,
                      feedback_learning_enabled=bool(active), feedback_decay_enabled=False,
                      learn_b_enabled=C.FAMILY_SWITCHES.get(policy,(False,False))[0],
                      learn_vu_enabled=C.FAMILY_SWITCHES.get(policy,(False,False))[1],
                      learned_feedback_keys=list(active),
                      inactive_feedback_unchanged_bitwise=all(torch.equal(state[k],start[k]) for k in C.FK if k not in active),
                      feedback_change_by_family={name:float((C.flat(state,keys)-C.flat(start,keys)).norm()) for name,keys in [('B',C.B_KEYS),('VU',C.VU_KEYS)]},
                      feedback_change_norm=float((C.flat(state, C.FK)-C.flat(start, C.FK)).norm()),
                      final_feedback_max_sigma=max(float(torch.linalg.svdvals(v)[0]) for k, v in C.decode(state).items() if k in C.FK),
                      fact_exposures=completed*32*C.R,
                      learner_backward_calls=completed if policy == 'backprop' else 0,
                      diagnostic_backward_calls=len(probes),
                      local_calls=completed if policy != 'backprop' else 0)
    return state, shadow, result


def worker(seed, pilot=False):
    assert_frozen()
    begin = time.process_time()
    cfg = json.loads((ROOT / 'PROTOCOL.json').read_text())
    params = cfg['pilot' if pilot else 'main']
    folder = ROOT / ('pilot' if pilot else 'main') / str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        schedules = C.make_replay_data(seed, params['warmup_steps'], params['phase_steps'], params['queries_per_context'])
        data = schedules['fifo']
        data_hash = digest(schedules)
        initial = C.init(seed)
        warm = C.bounded(initial)
        warm_begin, warm_trace = time.process_time(), []
        for t, ix in enumerate(data['warm_ix']):
            warm = C.warm_step(warm, data['base_x'][ix], data['base_y'][ix])
            assert finite(warm)
            if (t + 1) % 1024 == 0:
                warm_trace.append({'t': t + 1, **C.evaluate(warm, data, False)})
                print(json.dumps({'seed': seed, 'warmup': t + 1}), flush=True)
        warm_cpu = time.process_time() - warm_begin
        reference = {k: warm[k].clone() for k in C.FK}
        banks = {'warm': C.clone(warm)}
        states_path = folder / 'STATES.npz'
        save_states(states_path, banks)
        all_results = {}
        for arm in C.REPLAY_ARMS:
            start, shadow = start_for(arm, warm)
            end, shadow, row = run_arm(start, shadow, reference, schedules[arm.rsplit('_', 1)[1]], arm)
            all_results[arm] = row
            banks[arm] = {k: end[k] for k in (*C.WK, *C.FAMILY_KEYS.get(arm.rsplit('_',1)[0],()))}
            if shadow:
                banks[arm + '_shadow'] = shadow
            save_states(states_path, banks)
            save(folder / 'PROGRESS.json', {'seed': seed, 'arms': all_results, 'data_sha256': data_hash})
            print(json.dumps({'seed': seed, 'arm': arm, 'status': row['status'], 'primary_ce': row['primary_ce']}), flush=True)
        # Reload byte-preserved starting state and independently regenerate the entire stream.
        replay_begin = time.process_time()
        restored = load_states(states_path)
        regenerated = C.make_replay_data(seed, params['warmup_steps'], params['phase_steps'], params['queries_per_context'])
        assert digest(regenerated) == data_hash
        replays = {}
        for arm in C.REPLAY_ARMS:
            start, shadow = start_for(arm, restored['warm'])
            end, shadow, row = run_arm(start, shadow, restored['warm'], regenerated[arm.rsplit('_', 1)[1]], arm, measure=False)
            original = all_results[arm]
            exact = (row['phase_hashes'] == original['phase_hashes'] and
                     row['final_state_hash'] == original['final_state_hash'] and
                     row['status'] == original['status'] and row['completed_steps'] == original['completed_steps'])
            assert exact, ('Replay mismatch', seed, arm)
            assert all(torch.equal(end[k], restored[arm][k]) for k in restored[arm])
            replays[arm] = {'all_phase_final_and_shadow_states_bitwise_equal': exact,
                            'status': row['status'], 'completed_steps': row['completed_steps']}
        r = resource.getrusage(resource.RUSAGE_SELF)
        result = {'run_id': cfg['run_id'], 'seed': seed, 'pilot': pilot, 'status': 'COMPLETE',
                  'data_sha256': data_hash, 'warm_state_sha256': digest(warm),
                  'warmup': {'steps': len(data['warm_ix']), 'cpu_seconds': warm_cpu, 'trace': warm_trace,
                             'fact_exposures': len(data['warm_ix'])*64*C.R},
                  'arms': all_results, 'replays': replays,
                  'replay_cpu_seconds': time.process_time()-replay_begin,
                  'worker_cpu_seconds': time.process_time()-begin,
                  'stale_replayed_facts': {m: int(schedules[m]['stale_replayed_facts'].sum()) for m in ('fifo',)},
                  'schedule_hashes': {m: digest(schedules[m]) for m in ('fifo',)},
                  'initial_memory_ids_sha256': digest(schedules['memory_audit']['initial_ids']),
                  'storage_examples_per_policy': 512, 'replay_examples_per_step': 16,
                  'state_layout': 'warm holds common W/F; each arm stores final W and only its learned feedback family. Full final state is warm overlaid with stored arm; inactive F reference warm.',
                  'states': {'filename': 'STATES.npz', 'bytes': states_path.stat().st_size,
                             'sha256': hashlib.sha256(states_path.read_bytes()).hexdigest()},
                  'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                  'platform': platform.platform(), 'threads': torch.get_num_threads(), 'device': 'cpu'},
                  'resources': {'ru_maxrss_kib': r.ru_maxrss, 'total_process_cpu_seconds': r.ru_utime+r.ru_stime},
                  'source_hashes': json.loads((ROOT/'REGISTRATION.json').read_text())['sha256']}
        save(folder/'RESULT.json', result)
        (folder/'PROGRESS.json').unlink()
        print(json.dumps({'seed': seed, 'status': 'COMPLETE', 'cpu_seconds': result['worker_cpu_seconds']}), flush=True)
    except BaseException as exc:
        save(folder/'FAILURE.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                                   'cpu_seconds': time.process_time()-begin})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    worker(args.seed, args.pilot)
