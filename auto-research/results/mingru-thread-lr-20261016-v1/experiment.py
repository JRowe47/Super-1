"""One-seed, matched-training-wall minGRU thread and learning-rate screen."""
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'gru-batch-wall-20261016-v1'
sys.path.insert(0, str(PREVIOUS))
from experiment import batch, build, evaluate, optimizer, runtime, save, sha, update
import numpy as np
import torch

ARM = 'mingru_linear'

def inputs(args):
    p = json.loads((ROOT / 'PROTOCOL.json').read_text())
    for name, digest in p['source_hashes'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in p['data_hashes'].items():
        assert sha(args.data_dir / name) == digest, name
    assert sha(args.data_dir / 'tokenizer.json') == p['tokenizer_sha256']
    q = next(r for r in p['runs'] if r['run_id'] == args.run_id)
    os.sched_setaffinity(0, set(range(q['threads'])))
    torch.set_num_threads(q['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    resource.setrlimit(resource.RLIMIT_CPU, (600, 610))
    train = np.memmap(args.data_dir / 'train.bin', dtype=np.uint16, mode='r')
    val = np.memmap(args.data_dir / 'validation.bin', dtype=np.uint16, mode='r')
    assert len(train) == p['data']['train_tokens']
    assert train.max() < 2048 and val.max() < 2048
    starts = np.random.default_rng(p['data_seed']).integers(0, len(train)-512, size=p['max_windows'], dtype=np.int64)
    assert hashlib.sha256(starts.tobytes()).hexdigest() == p['train_stream_sha256']
    assert max(p['monitor_starts']) + 513 < min(p['final_starts'])
    assert max(p['final_starts']) + 513 <= len(val)
    return p, q, train, val, starts

def train_run(args, p, q, train, val, starts):
    out = ROOT / 'execution' / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'RESULT.json').exists(), 'Completed run is immutable'
    assert not (out / 'FAILURE.json').exists(), 'Failed trial must not be silently replaced'
    cp = out / 'checkpoint.pt'
    m = build(ARM, p)
    assert sum(v.numel() for v in m.parameters()) == p['parameters'][ARM]
    opt = optimizer(m, q['peak_lr'])
    b = q['batch']
    state = dict(step=0, windows=0, training_targets=0, train_wall_seconds=0., train_cpu_seconds=0., train=[], monitor=[])
    if cp.exists():
        old = torch.load(cp, map_location='cpu', weights_only=False)
        assert old['protocol_sha256'] == sha(ROOT / 'PROTOCOL.json')
        assert old['run_id'] == args.run_id
        m.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        torch.set_rng_state(old['torch_rng']); state = old['state']

    def checkpoint():
        tmp = cp.with_suffix('.tmp')
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), torch_rng=torch.get_rng_state(),
            state=state, arm=ARM, config=p['configs'][ARM], run_id=args.run_id,
            protocol_sha256=sha(ROOT / 'PROTOCOL.json')), tmp)
        tmp.replace(cp)

    def monitor():
        e = evaluate(m, ARM, val, p['monitor_starts'])
        row = {k: state[k] for k in ('step', 'windows', 'training_targets', 'train_wall_seconds', 'train_cpu_seconds')}
        row.update(validation_ce=e['ce'], validation_top1=e['top1'])
        state['monitor'].append(row)
        save(out / 'curves.json', state)
        print(args.run_id, row, flush=True)

    began, cpu_began = time.perf_counter(), time.process_time()
    try:
        if not state['monitor']:
            save(out / 'START.json', dict(runtime=runtime(), protocol_sha256=sha(ROOT / 'PROTOCOL.json'),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
            monitor(); checkpoint()
        recent = [r['wall'] for r in state['train'][-10:]] or [q['reference_step_wall_seconds']]
        budget = p['train_wall_seconds']
        next_monitor = (int(state['train_wall_seconds'] // 15) + 1) * 15
        while state['windows'] + b <= len(starts):
            reserve = max(.2, 2 * max(recent[-10:]))
            if state['train_wall_seconds'] + reserve >= budget:
                break
            if time.perf_counter() - began > 180:
                raise TimeoutError('Per-run wall cap reached')
            progress = min(1., max(0., state['train_wall_seconds'] / budget))
            factor = max(.04, progress/.05) if progress < .05 else .1 + .9*.5*(1 + math.cos(math.pi*(progress-.05)/.95))
            w = state['windows']
            r = update(m, opt, ARM, train, starts[w:w+b], q['peak_lr'] * factor)
            recent.append(r['wall'])
            state['step'] += 1; state['windows'] += b; state['training_targets'] += 512*b
            state['train_wall_seconds'] += r['wall']; state['train_cpu_seconds'] += r['cpu']
            state['train'].append(dict(step=state['step'], training_targets=state['training_targets'],
                train_wall_seconds=state['train_wall_seconds'], train_cpu_seconds=state['train_cpu_seconds'], **r))
            if state['train_wall_seconds'] >= next_monitor:
                monitor(); checkpoint(); next_monitor += 15
        if state['monitor'][-1]['step'] != state['step']:
            monitor()
        checkpoint()
        final = evaluate(m, ARM, val, p['final_starts'])
        result = dict(**q, arm=ARM, parameters=p['parameters'][ARM], seed=p['seed'],
            updates=state['step'], training_targets=state['training_targets'], train_wall_seconds=state['train_wall_seconds'],
            train_cpu_seconds=state['train_cpu_seconds'], final=final, monitor=state['monitor'],
            budget_unused_seconds=budget-state['train_wall_seconds'], run_wall_seconds=time.perf_counter()-began,
            run_cpu_seconds=time.process_time()-cpu_began, checkpoint_sha256=sha(cp),
            protocol_sha256=sha(ROOT / 'PROTOCOL.json'), runtime=runtime(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(out / 'RESULT.json', result); save(out / 'curves.json', state)
        print('COMPLETE', args.run_id, result['updates'], result['train_wall_seconds'], final['ce'], flush=True)
    except Exception as error:
        checkpoint()
        save(out / 'FAILURE.json', dict(run_id=args.run_id, error_type=type(error).__name__, error=str(error),
            completed_updates=state['step'], train_wall_seconds=state['train_wall_seconds'],
            protocol_sha256=sha(ROOT / 'PROTOCOL.json'), checkpoint_sha256=sha(cp)))
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    train_run(args, *inputs(args))
