"""Registered synthetic mechanics only; exact references are read, never edited."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import types

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PROTOCOL = json.loads((OUT / 'PROTOCOL.json').read_text())
CONFIG = json.loads((OUT / 'HARNESS_CONFIG.json').read_text())
START = time.monotonic()
CPU_START = time.process_time()
RECEIPT = {'status': 'RUNNING', 'rows': [], 'failures': [], 'scope': PROTOCOL['kind']}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, payload):
    with (OUT / name).open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(row, name, ok, **details):
    row['checks'][name] = {'pass': bool(ok), **details}
    if not ok:
        RECEIPT['failures'].append({'seed': row['seed'], 'arm': row['arm'], 'check': name})


def timeout_handler(signum, frame):
    raise TimeoutError('Registered five-minute execution limit reached')


def main():
    assert os.path.ismount('/Volumes/SP AI 01_16')
    assert str(OUT).startswith('/Volumes/SP AI 01_16/CodexWorkspaces/')
    registry = json.loads((ROOT / 'auto-research/RUN_REGISTRY.json').read_text())
    entry = next(r for r in registry['runs'] if r['run_id'] == PROTOCOL['run_id'])
    assert sha(OUT / 'PROTOCOL.json') == entry['protocol_sha256']
    hist = ROOT / PROTOCOL['model_source']
    nano_path = ROOT / CONFIG['nanogpt_source']
    geometry_path = ROOT / CONFIG['geometry_source']
    assert sha(hist / 'predictive_models.py') == PROTOCOL['historical_model_sha256']
    assert sha(nano_path) == PROTOCOL['nanogpt_model_sha256']
    source_paths = [hist / 'predictive_models.py', hist / 'common.py', nano_path,
                    geometry_path, Path(__file__), OUT / 'HARNESS_CONFIG.json', OUT / 'PROTOCOL.json']
    RECEIPT['sources'] = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    save('SOURCE_MANIFEST.json', RECEIPT['sources'])
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(PROTOCOL['threads'])
    torch.set_num_interop_threads(PROTOCOL['threads'])
    torch.use_deterministic_algorithms(True)
    RECEIPT['runtime'] = {'python': sys.version, 'executable': sys.executable,
                          'torch': torch.__version__, 'torch_path': torch.__file__,
                          'device': 'cpu', 'dtype': 'float64',
                          'intraop_threads': torch.get_num_threads(),
                          'interop_threads': torch.get_num_interop_threads()}
    package = types.ModuleType('representation_experiments')
    package.__path__ = [str(hist)]
    sys.modules[package.__name__] = package
    load('representation_experiments.common', hist / 'common.py')
    models = load('representation_experiments.predictive_models', hist / 'predictive_models.py')
    geometry = load('registered_geometry', geometry_path)
    nano = load('registered_nanogpt', nano_path)
    tol = PROTOCOL['tolerance']['double_cpu_streaming_abs']
    length, vocab = PROTOCOL['sequence_length'], PROTOCOL['vocab_size']
    for seed in PROTOCOL['seeds']:
        torch.manual_seed(seed)
        base = models.HierarchicalDistributionStudent(vocab, PROTOCOL['width'], length,
                    PROTOCOL['segment_length'], PROTOCOL['document_segments']).double().cpu().eval()
        x = torch.randint(vocab, (PROTOCOL['batch_size'], length), device='cpu')
        target = torch.randint(vocab, x.shape, device='cpu')
        changed = x.clone()
        changed[:, CONFIG['causal_split']:] = (changed[:, CONFIG['causal_split']:] + 1) % vocab
        arms = {'native': base}
        for kind in CONFIG['geometry_queries']:
            arms[kind] = geometry.install_fast_rotation(copy.deepcopy(base), kind, CONFIG['epsilon']).double()
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in arms.items()}
        with torch.no_grad():
            initial = base(x)
        for kind, model in arms.items():
            row = {'seed': seed, 'arm': kind, 'parameters': counts[kind], 'checks': {}}
            RECEIPT['rows'].append(row)
            started = time.monotonic()
            with torch.no_grad():
                identity = model(x)
                check(row, 'exact_identity_initialization', torch.equal(identity.logits, initial.logits)
                      and torch.equal(identity.hidden_states, initial.hidden_states)
                      and all(torch.equal(v, initial.final_state[k]) if isinstance(v, torch.Tensor)
                              else v == initial.final_state[k] for k, v in identity.final_state.items()))
                if kind != 'native':
                    model.fast_cell.gain.fill_(CONFIG['nonzero_geometry_gain'])
                    check(row, 'equal_count_control', counts['oriented_cross'] == counts['alignment_dot'])
                full = model(x)
                altered = model(changed)
                check(row, 'causal_prefix', torch.equal(full.logits[:, :CONFIG['causal_split']],
                                                      altered.logits[:, :CONFIG['causal_split']]))
                calls = dict(fast=0, segment=0, document=0)
                handles = []
                for scale in calls:
                    def count_call(module, args, result, key=scale):
                        calls[key] += 1
                    handles.append(getattr(model, scale + '_cell').register_forward_hook(count_call))
                state, values, hidden = None, [], []
                for token in x.unbind(1):
                    logits, state, h = model.step(token, state)
                    values.append(logits); hidden.append(h)
                for handle in handles:
                    handle.remove()
                error = float((full.logits - torch.stack(values, 1)).abs().max())
                hidden_error = float((full.hidden_states - torch.stack(hidden, 1)).abs().max())
                state_errors = {k: float((v - state[k]).abs().max()) if isinstance(v, torch.Tensor)
                                else abs(v - state[k]) for k, v in full.final_state.items()}
                check(row, 'forward_vs_step_streaming', max([error, hidden_error] + list(state_errors.values())) <= tol,
                      logit_max_abs=error, hidden_max_abs=hidden_error, final_state_errors=state_errors,
                      tolerance_abs=tol, relative_tolerance=0)
                expected = {'fast': length, 'segment': length // PROTOCOL['segment_length'],
                            'document': length // (PROTOCOL['segment_length'] * PROTOCOL['document_segments'])}
                check(row, 'all_historical_clock_levels_exercised', calls == expected, calls=calls, expected=expected)
            optimizer = torch.optim.AdamW(model.parameters(), **CONFIG['optimizer'])
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            loss = F.cross_entropy(model(x).logits.reshape(-1, vocab), target.reshape(-1))
            loss.backward()
            check(row, 'finite_loss_and_parameter_gradients', bool(torch.isfinite(loss)) and
                  all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            grad_norms = {scale: sum(float(p.grad.abs().sum()) for p in getattr(model, scale + '_cell').parameters()
                                     if p.grad is not None) for scale in ('fast', 'segment', 'document')}
            check(row, 'nonzero_each_clock_gradients', all(v > 0 for v in grad_norms.values()), l1=grad_norms)
            if kind != 'native':
                gain_grad = float(model.fast_cell.gain.grad.abs().sum())
                check(row, 'nonzero_geometry_gain_gradients', gain_grad > 0, l1=gain_grad)
            optimizer.step()
            updates = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
            check(row, 'optimizer_update', bool(updates) and all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                  updated_parameters=updates)
            row['synthetic_loss_not_quality_metric'] = float(loss.detach())
            row['elapsed_seconds'] = time.monotonic() - started
            save(f'{seed}-{kind}.json', row)
            print(json.dumps({'seed': seed, 'arm': kind, 'checks_pass': all(c['pass'] for c in row['checks'].values()),
                              'seconds': row['elapsed_seconds']}), flush=True)
        cfg = nano.GPTConfig(block_size=length, vocab_size=vocab, n_embd=PROTOCOL['width'], **CONFIG['nanogpt_config'])
        model = nano.GPT(cfg).double().cpu().eval()
        row = {'seed': seed, 'arm': 'pinned_nanogpt', 'parameters': sum(p.numel() for p in model.parameters()),
               'parameter_matched': False, 'checks': {}, 'not_applicable': ['geometry identity/gain', 'historical clocks'],
               'streaming_definition': 'Growing-prefix inference; pinned nanoGPT exposes no recurrent step/KV-cache API'}
        RECEIPT['rows'].append(row)
        started = time.monotonic()
        with torch.no_grad():
            full, _ = model(x, target)
            altered, _ = model(changed, target)
            check(row, 'causal_prefix', torch.equal(full[:, :CONFIG['causal_split']], altered[:, :CONFIG['causal_split']]))
            prefixes = torch.cat([model(x[:, :i+1])[0] for i in range(length)], 1)
            error = float((full - prefixes).abs().max())
            check(row, 'forward_vs_prefix_streaming', error <= tol, logit_max_abs=error,
                  tolerance_abs=tol, relative_tolerance=0)
        optimizer = torch.optim.AdamW(model.parameters(), **CONFIG['optimizer'])
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        logits, loss = model(x, target)
        loss.backward()
        check(row, 'finite_loss_and_parameter_gradients', bool(torch.isfinite(loss)) and
              all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        optimizer.step()
        updates = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
        check(row, 'optimizer_update', bool(updates) and all(bool(torch.isfinite(p).all()) for p in model.parameters()),
              updated_parameters=updates)
        row['synthetic_loss_not_quality_metric'] = float(loss.detach())
        row['elapsed_seconds'] = time.monotonic() - started
        save(f'{seed}-pinned_nanogpt.json', row)
        print(json.dumps({'seed': seed, 'arm': 'pinned_nanogpt', 'checks_pass': all(c['pass'] for c in row['checks'].values()),
                          'seconds': row['elapsed_seconds']}), flush=True)
    RECEIPT['sources_unchanged'] = all(sha(ROOT / name) == digest for name, digest in RECEIPT['sources'].items())
    RECEIPT['status'] = 'PASS_SYNTHETIC_MECHANICS_ONLY' if not RECEIPT['failures'] and RECEIPT['sources_unchanged'] else 'FAILED'


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(CONFIG['execution_limit_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU, (CONFIG['execution_limit_seconds'], CONFIG['execution_limit_seconds']))
    try:
        main()
    except BaseException:
        RECEIPT['status'] = 'FAILED_EXECUTION'
        RECEIPT['exception'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        signal.alarm(0)
        RECEIPT['wall_seconds'] = time.monotonic() - START
        RECEIPT['cpu_seconds'] = time.process_time() - CPU_START
        RECEIPT['peak_process_rss_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        RECEIPT['dataset_content_read'] = False
        RECEIPT['official_test_accessed'] = False
        RECEIPT['cloud_accessed'] = False
        RECEIPT['repair_attempts'] = 0
        save('TERMINAL.json', RECEIPT)
    sys.exit(0 if RECEIPT['status'] == 'PASS_SYNTHETIC_MECHANICS_ONLY' else 1)
