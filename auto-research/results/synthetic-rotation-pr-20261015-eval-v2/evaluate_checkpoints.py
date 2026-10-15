"""Evaluation-only recovery: frozen training artifacts are never modified."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
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
PARENT = ROOT / P['recovery']['parent_path']
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
OUT = parser.parse_args().output.resolve()
assert os.path.ismount('/Volumes/SP AI 01_16')
assert OUT.is_relative_to(Path('/Volumes/SP AI 01_16/CodexWorkspaces')) and not OUT.exists()
OUT.mkdir(parents=True)
start, cpu = time.monotonic(), time.process_time()
receipt = {'status': 'RUNNING', 'rows': [], 'training_updates': 0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def limit(signum, frame):
    raise TimeoutError('Remaining shared CPU budget exhausted')


def main():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(C['threads'])
    torch.set_num_interop_threads(C['threads'])
    torch.use_deterministic_algorithms(True)
    source = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
    assert all(sha(ROOT / k) == v for k, v in source.items())
    registry = json.loads((ROOT / 'auto-research/RUN_REGISTRY.json').read_text())
    entry = next(r for r in registry['runs'] if r['run_id'] == P['run_id'])
    assert sha(HERE / 'PROTOCOL.json') == entry['protocol_sha256']
    save('START.json', {'protocol_sha256': entry['protocol_sha256'], 'sources': source,
                        'argv': sys.argv, 'pid': os.getpid(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    hist = ROOT / C['historical_source']
    package = types.ModuleType('representation_experiments')
    package.__path__ = [str(hist)]
    sys.modules[package.__name__] = package
    load('representation_experiments.common', hist / 'common.py')
    native = load('representation_experiments.predictive_models', hist / 'predictive_models.py')
    geometry = load('frozen_geometry', ROOT / C['geometry_source'])
    nano = load('frozen_nano', ROOT / C['nanogpt_source'])
    data = torch.tensor(json.loads((PARENT / 'DATA.json').read_text())['confirmation'], dtype=torch.long)
    old = json.loads((PARENT / 'execution/TERMINAL.json').read_text())
    frozen = json.loads((PARENT / 'execution/CHECKPOINT_FREEZE.json').read_text())
    receipt['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu',
                          'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()}

    def evaluate(model, arm, reset):
        total, targets, digest = 0., 0, hashlib.sha256()
        with torch.no_grad():
            for doc in data:
                x, y = doc[:, :-1].contiguous(), doc[:, 1:].contiguous()
                if arm == 'pinned_nanogpt':
                    if reset:
                        logits = torch.cat([model(a.contiguous(), b.contiguous())[0] for a, b in
                                            zip(x.split(C['chunk_length'], 1), y.split(C['chunk_length'], 1))], 1)
                    else:
                        logits = model(x, y)[0]
                else:
                    state, values = None, []
                    for chunk in x.split(C['chunk_length'], 1):
                        if reset:
                            state = None
                        for token in chunk.unbind(1):
                            out, state, _ = model.step(token, state)
                            values.append(out)
                    logits = torch.stack(values, 1)
                total += float(F.cross_entropy(logits.reshape(-1, C['vocab_size']), y.reshape(-1), reduction='sum'))
                targets += y.numel()
                digest.update(logits.contiguous().numpy().tobytes())
        return {'ce_nats_per_target': total/targets, 'targets': targets, 'logits_sha256': digest.hexdigest()}

    for row in frozen:
        seed, arm = row['seed'], row['arm']
        torch.manual_seed(seed)
        if arm == 'pinned_nanogpt':
            model = nano.GPT(nano.GPTConfig(vocab_size=C['vocab_size'], block_size=C['sequence_length'], **C['nanogpt_config']))
        else:
            model = native.HierarchicalDistributionStudent(C['vocab_size'], C['width'], C['sequence_length'], C['segment_length'], C['document_segments'])
            if arm in C['geometry_arms']:
                model = geometry.install_fast_rotation(model, arm, C['epsilon'])
        checkpoint = PARENT / 'execution' / row['checkpoint']
        assert sha(checkpoint) == row['checkpoint_sha256']
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
        model.cpu().eval()
        row['confirmation'] = evaluate(model, arm, False)
        row['duplicate_confirmation'] = evaluate(model, arm, False)
        assert row['confirmation'] == row['duplicate_confirmation']
        previous = next(r for r in old['rows'] if r['seed'] == seed and r['arm'] == arm)
        if 'confirmation' in previous:
            assert previous['confirmation'] == row['confirmation']
            row['matches_preserved_first_attempt_confirmation'] = True
        row['reset_window_confirmation'] = evaluate(model, arm, True)
        receipt['rows'].append(row)
        save(f'{seed}-{arm}.json', row)
        print(json.dumps({'seed': seed, 'arm': arm, 'confirmation': row['confirmation']}), flush=True)
    paired = []
    for seed in C['seeds']:
        rows = {r['arm']: r for r in receipt['rows'] if r['seed'] == seed}
        assert rows['oriented_cross']['parameters'] == rows['alignment_dot']['parameters']
        ce = {a:r['confirmation']['ce_nats_per_target'] for a,r in rows.items()}
        paired.append({'seed':seed, 'ce':ce,'oriented_minus_alignment':ce['oriented_cross']-ce['alignment_dot'],
                       'oriented_minus_native':ce['oriented_cross']-ce['native']})
    gaps = [r['oriented_minus_alignment'] for r in paired]
    mean = statistics.mean(gaps)
    half = C['t_critical_95_df2']*statistics.stdev(gaps)/math.sqrt(len(gaps))
    passed = all(d<0 for d in gaps) and mean <= -C['minimum_effect_nats'] and mean+half<0 and all(r['oriented_minus_native']<0 for r in paired)
    result = {'paired':paired,'mean_oriented_minus_alignment':mean,'paired_t_95_interval':[mean-half,mean+half],
              'primary_gate_passed':passed,'decision':'SYNTHETIC_CRITERION_MET_NO_LANGUAGE_PROMOTION' if passed else 'NO_PROMOTION_INCONCLUSIVE_OR_NEGATIVE'}
    save('RESULT.json',result)
    receipt['result']=result
    receipt['sources_and_checkpoints_unchanged']=all(sha(ROOT/k)==v for k,v in source.items())
    assert receipt['sources_and_checkpoints_unchanged']
    receipt['status']='COMPLETED_EVALUATION_RECOVERY_NO_RETRAINING'


signal.signal(signal.SIGXCPU,limit)
signal.signal(signal.SIGALRM,limit)
resource.setrlimit(resource.RLIMIT_CPU,(P['recovery']['cpu_soft_seconds'],P['recovery']['cpu_hard_seconds']))
signal.alarm(P['recovery']['walltime_seconds'])
try:
    main()
except BaseException:
    receipt['status']='FAILED_EVALUATION_RECOVERY'
    receipt['exception']=traceback.format_exc()
    traceback.print_exc()
finally:
    signal.alarm(0)
    receipt['cpu_seconds']=time.process_time()-cpu
    receipt['wall_seconds']=time.monotonic()-start
    receipt['peak_process_rss_bytes_macos']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt['official_test_accessed']=False
    receipt['external_dataset_accessed']=False
    receipt['cloud_used']=False
    receipt['cleanup']='Local synchronous CPU process; no background jobs or cloud resources; preserve all evidence'
    save('TERMINAL.json',receipt)
    print(json.dumps({'status':receipt['status'],'cpu_seconds':receipt['cpu_seconds']}),flush=True)
sys.exit(0 if receipt['status']=='COMPLETED_EVALUATION_RECOVERY_NO_RETRAINING' else 1)
