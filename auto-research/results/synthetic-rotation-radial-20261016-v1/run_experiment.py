"""Frozen synthetic experiment. Replays require a fresh SSD --output directory."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = json.loads((HERE / 'PROTOCOL.json').read_text())
C = P['config']
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
OUT = args.output.resolve()
assert os.path.ismount('/Volumes/SP AI 01_16')
assert OUT.is_relative_to(Path('/Volumes/SP AI 01_16/CodexWorkspaces'))
assert not OUT.exists(), 'Never overwrite an existing run or replay'
OUT.mkdir(parents=True)
START, CPU_START = time.monotonic(), time.process_time()
receipt = {'status': 'RUNNING', 'rows': [], 'failures': []}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_bytes():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    assert output_bytes() < C['max_artifact_bytes']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def limit(signum, frame):
    raise TimeoutError('Preregistered CPU or walltime budget exhausted')


def main():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(C['threads'])
    torch.set_num_interop_threads(C['threads'])
    torch.use_deterministic_algorithms(True)
    receipt['runtime'] = {'python': sys.version, 'executable': sys.executable,
                          'torch': torch.__version__, 'platform': platform.platform(),
                          'machine': platform.machine(), 'device': 'cpu', 'dtype': 'float32',
                          'intraop_threads': torch.get_num_threads(),
                          'interop_threads': torch.get_num_interop_threads()}
    sources = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
    assert all(sha(ROOT / k) == v for k, v in sources.items())
    registry = json.loads((ROOT / 'auto-research/RUN_REGISTRY.json').read_text())
    entry = next(r for r in registry['runs'] if r['run_id'] == P['run_id'])
    assert sha(HERE / 'PROTOCOL.json') == entry['protocol_sha256']
    save('START.json', {'protocol_sha256': entry['protocol_sha256'], 'sources': sources,
                        'preregistration_commit': __import__('subprocess').check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'argv': sys.argv, 'pid': os.getpid()})
    hist = ROOT / C['historical_source']
    package = types.ModuleType('representation_experiments')
    package.__path__ = [str(hist)]
    sys.modules[package.__name__] = package
    load('representation_experiments.common', hist / 'common.py')
    models = load('representation_experiments.predictive_models', hist / 'predictive_models.py')
    geometry = load('registered_geometry', ROOT / C['geometry_source'])
    nano = load('registered_nanogpt', ROOT / C['nanogpt_source'])
    data = {k: torch.tensor(v, dtype=torch.long) for k, v in
            json.loads((HERE / 'DATA.json').read_text()).items()}
    vocab = C['vocab_size']

    def new_model(seed, arm):
        torch.manual_seed(seed)
        if arm == 'pinned_nanogpt':
            return nano.GPT(nano.GPTConfig(vocab_size=vocab, block_size=C['sequence_length'],
                                           **C['nanogpt_config'])).cpu()
        model = models.HierarchicalDistributionStudent(vocab, C['width'], C['sequence_length'],
                                                       C['segment_length'], C['document_segments']).cpu()
        if arm != 'native':
            model = geometry.install_fast_rotation(model, arm, C['epsilon'])
        return model

    def logits_for(model, arm, x, y, reset_window=False):
        if arm == 'pinned_nanogpt':
            if reset_window:
                return torch.cat([model(a.contiguous(), b.contiguous())[0] for a, b in
                                  zip(x.split(C['chunk_length'], 1), y.split(C['chunk_length'], 1))], 1)
            return model(x, y)[0]
        state, values = None, []
        for chunk in x.split(C['chunk_length'], 1):
            if reset_window:
                state = None
            for token in chunk.unbind(1):
                logits, state, _ = model.step(token, state)
                values.append(logits)
        return torch.stack(values, 1)

    def evaluate(model, arm, split, reset_window=False):
        model.eval()
        total, targets, all_total, all_targets = 0., 0, 0., 0
        digest = hashlib.sha256()
        with torch.no_grad():
            for doc in data[split]:
                x, y = doc[:, :-1].contiguous(), doc[:, 1:].contiguous()
                logits = logits_for(model, arm, x, y, reset_window)
                losses = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction='none').reshape_as(y)
                mask = y < C['modulus']
                total += float(losses[mask].sum())
                targets += int(mask.sum())
                all_total += float(losses.sum())
                all_targets += y.numel()
                digest.update(logits.contiguous().numpy().tobytes())
        return {'ce_nats_per_target': total / targets, 'targets': targets, 'all_token_ce': all_total / all_targets, 'all_targets': all_targets, 'logits_sha256': digest.hexdigest()}

    for seed in C['seeds']:
        reference = new_model(seed, 'native').eval()
        x, y = data['selection'][0, :, :-1].contiguous(), data['selection'][0, :, 1:].contiguous()
        for arm in C['arms']:
            started = time.process_time()
            model = new_model(seed, arm)
            row = {'seed': seed, 'arm': arm, 'updates': 0, 'target_exposures': 0,
                   'parameters': sum(p.numel() for p in model.parameters())}
            receipt['rows'].append(row)
            if arm != 'pinned_nanogpt':
                with torch.no_grad():
                    assert torch.equal(model(x).logits, reference(x).logits)
                    direct = model(x)
                    streamed = logits_for(model, arm, x, y)
                    assert torch.equal(direct.logits, streamed)
                row['identity_and_streaming_exact'] = True
            if arm in C['geometry_arms']:
                probe = copy.deepcopy(model)
                probe.fast_cell.gain.data.fill_(C['probe_gain'])
                with torch.no_grad():
                    inputs = torch.linspace(-1, 1, C['batch_size']*3*C['width']).reshape(C['batch_size'], 3*C['width'])
                    prev = torch.zeros(C['batch_size'], C['width'])
                    before = probe.fast_cell.native(inputs, prev).reshape(C['batch_size'], -1, 2)
                    after = probe.fast_cell(inputs, prev).reshape_as(before)
                    norm_delta = float((before.square().sum(-1)-after.square().sum(-1)).abs().max())
                    row['probe_pair_squared_norm_max_change'] = norm_delta
                    assert norm_delta <= 1e-6 if arm == 'rotation' else norm_delta > 0
                probe_loss = F.cross_entropy(logits_for(probe, arm, x, y).reshape(-1, vocab), y.reshape(-1))
                probe_loss.backward()
                row['nonzero_gain_gradient_l1'] = float(probe.fast_cell.gain.grad.abs().sum())
                assert row['nonzero_gain_gradient_l1'] > 0
                del probe
            row['initial_selection'] = evaluate(model, arm, 'selection')
            optimizer = torch.optim.AdamW(model.parameters(), **C['optimizer'])
            model.train()
            for update, doc in enumerate(data['train'], 1):
                optimizer.zero_grad(set_to_none=True)
                x, y = doc[:, :-1].contiguous(), doc[:, 1:].contiguous()
                logits = logits_for(model, arm, x, y)
                loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), C['clip_norm']))
                optimizer.step()
                row['updates'] = update
                row['target_exposures'] += y.numel()
                with (OUT / 'progress.jsonl').open('a') as f:
                    f.write(json.dumps({'seed': seed, 'arm': arm, 'update': update,
                                        'targets': row['target_exposures'], 'train_ce': float(loss.detach()),
                                        'preclip_grad_norm': grad_norm}) + '\n')
            row['final_selection'] = evaluate(model, arm, 'selection')
            if arm in C['geometry_arms']:
                row['learned_gain_l1'] = float(model.fast_cell.gain.detach().abs().sum())
                row['learned_phase_l1'] = float(model.fast_cell.phase.detach().abs().sum())
            checkpoint = OUT / f'{seed}-{arm}.pt'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'seed': seed, 'arm': arm, 'updates': row['updates'],
                        'target_exposures': row['target_exposures']}, checkpoint)
            row['checkpoint'] = checkpoint.name
            row['checkpoint_sha256'] = sha(checkpoint)
            row['train_and_selection_cpu_seconds'] = time.process_time() - started
            assert output_bytes() < C['max_artifact_bytes']
            print(json.dumps(row), flush=True)
            del optimizer, model
        del reference
    assert len(receipt['rows']) == len(C['seeds']) * len(C['arms'])
    save('CHECKPOINT_FREEZE.json', {'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'rows': receipt['rows']})
    freeze_hash = sha(OUT / 'CHECKPOINT_FREEZE.json')
    for row in receipt['rows']:
        model = new_model(row['seed'], row['arm'])
        path = OUT / row['checkpoint']
        assert sha(path) == row['checkpoint_sha256']
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
        first = evaluate(model, row['arm'], 'confirmation')
        second = evaluate(model, row['arm'], 'confirmation')
        assert first == second
        row['freeze_sha256_before_confirmation'] = freeze_hash
        row['confirmation'] = first
        row['duplicate_confirmation'] = second
        row['reset_window_confirmation'] = evaluate(model, row['arm'], 'confirmation', True)
        save(f"{row['seed']}-{row['arm']}.json", row)
        del model
    paired = []
    for seed in C['seeds']:
        rows = {r['arm']: r for r in receipt['rows'] if r['seed'] == seed}
        assert rows['rotation']['parameters'] == rows['radial']['parameters']
        ce = {arm: r['confirmation']['ce_nats_per_target'] for arm, r in rows.items()}
        paired.append({'seed': seed, 'ce': ce,
                       'rotation_minus_radial': ce['rotation'] - ce['radial'],
                       'rotation_minus_native': ce['rotation'] - ce['native']})
    differences = [r['rotation_minus_radial'] for r in paired]
    mean = statistics.mean(differences)
    half = C['t_critical_95_df2'] * statistics.stdev(differences) / math.sqrt(len(differences))
    passed = (all(d < 0 for d in differences) and mean <= -C['minimum_effect_nats']
              and mean + half < 0 and all(r['rotation_minus_native'] < 0 for r in paired))
    result = {'paired': paired, 'mean_rotation_minus_radial': mean,
              'paired_t_95_interval': [mean-half, mean+half], 'primary_gate_passed': passed,
              'decision': 'SYNTHETIC_CRITERION_MET_NO_LANGUAGE_PROMOTION' if passed else 'NO_PROMOTION_INCONCLUSIVE_OR_NEGATIVE'}
    save('RESULT.json', result)
    receipt['result'] = result
    receipt['sources_unchanged'] = all(sha(ROOT / k) == v for k, v in sources.items())
    assert receipt['sources_unchanged']
    receipt['status'] = 'COMPLETED_SYNTHETIC_EXPERIMENT'


signal.signal(signal.SIGXCPU, limit)
signal.signal(signal.SIGALRM, limit)
resource.setrlimit(resource.RLIMIT_CPU, (C['cpu_soft_seconds'], C['cpu_hard_seconds']))
signal.alarm(C['walltime_seconds'])
try:
    main()
except BaseException:
    receipt['status'] = 'FAILED_OR_BUDGET_EXHAUSTED'
    receipt['failures'].append(traceback.format_exc())
    traceback.print_exc()
finally:
    signal.alarm(0)
    receipt['wall_seconds'] = time.monotonic() - START
    receipt['cpu_seconds'] = time.process_time()
    receipt['cpu_seconds_since_setup'] = time.process_time() - CPU_START
    receipt['peak_process_rss_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt['artifact_bytes_before_terminal'] = output_bytes()
    receipt['official_test_accessed'] = False
    receipt['external_dataset_accessed'] = False
    receipt['cloud_used'] = False
    receipt['cleanup'] = 'Synchronous CPU process only; all generated evidence retained; no resources provisioned'
    save('TERMINAL.json', receipt)
    print(json.dumps({'status': receipt['status'], 'cpu_seconds': receipt['cpu_seconds'],
                      'wall_seconds': receipt['wall_seconds']}), flush=True)
sys.exit(0 if receipt['status'] == 'COMPLETED_SYNTHETIC_EXPERIMENT' else 1)
