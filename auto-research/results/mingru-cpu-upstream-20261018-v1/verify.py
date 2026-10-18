"""Audit the imported CPU benchmark records without training or loading data.

Run from any directory. Uses only the standard library and prints a JSON receipt.
Checkpoint replay results are historical receipts, not rerun evaluations.
"""
import ast
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, label, tolerance=1e-10):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=tolerance), label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_hash(kind, data):
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).digest()


def tree_hash(files):
    """Hash the imported 100644 blobs using Git's canonical tree encoding."""
    node = {}
    for name, data in files.items():
        here = node
        parts = name.split('/')
        for part in parts[:-1]:
            here = here.setdefault(part, {})
        here[parts[-1]] = data

    def encode(items):
        chunks = []
        for name, item in sorted(items.items(), key=lambda x: x[0] + ('/' if isinstance(x[1], dict) else '')):
            directory = isinstance(item, dict)
            mode = '40000' if directory else '100644'
            digest = encode(item) if directory else git_hash('blob', item)
            chunks.append(f'{mode} {name}\0'.encode() + digest)
        return git_hash('tree', b''.join(chunks))

    return encode(node).hex()


def main():
    scope = read(ROOT / 'IMPORT.json')
    manifest = read(REPO / 'MANIFEST.sha256.json')
    imported_files = imported_bytes = syntax_files = 0
    for item in scope['source_roots']:
        path = item['path']
        names = [n for n in manifest if n == path or n.startswith(path + '/')]
        require(bool(names), f'No imported files for {path}')
        files = {}
        for name in names:
            data = (REPO / name).read_bytes()
            require(hashlib.sha256(data).hexdigest() == manifest[name], name)
            files[name[len(path) + 1:]] = data
            imported_files += 1
            imported_bytes += len(data)
            if name.endswith('.py'):
                ast.parse(data, filename=name)
                syntax_files += 1
        digest = (git_hash('blob', (REPO / path).read_bytes()).hex()
                  if item['type'] == 'blob' else tree_hash(files))
        require(digest == item['sha'], f'Changed source snapshot: {path}')
    require(imported_files == scope['imported_files'], 'Imported file count')
    require(imported_bytes == scope['imported_bytes'], 'Imported byte count')

    registry = {r['run_id']: r for r in read(REPO / 'auto-research/RUN_REGISTRY.json')['runs']}
    studies, updates = [], 0
    common_data = None
    for study in scope['studies']:
        root = REPO / 'auto-research/results' / study
        p, summary = read(root / 'PROTOCOL.json'), read(root / 'SUMMARY.json')
        terminal, checks = read(root / 'TERMINAL.json'), read(root / 'CHECKS.json')
        checkpoints = read(root / 'CHECKPOINT_MANIFEST.json')
        protocol_hash = sha(root / 'PROTOCOL.json')
        require(read(root / 'REGISTRATION.json')['protocol_sha256'] == protocol_hash, study)
        require(registry[study]['protocol_sha256'] == protocol_hash, study)
        for name, digest in p['source_hashes'].items():
            require(sha(root / name) == digest, f'{study}: {name}')
        data = (p['data_hashes'], p['tokenizer_sha256'], p['final_starts'])
        if common_data is None:
            common_data = data
        require(data == common_data, 'Data/tokenizer/final-window mismatch')
        require(terminal['status'] == 'COMPLETE' and not summary.get('failures'), study)
        require(not terminal['training_running'] and not terminal['official_test_accessed'], study)
        selection = read(root / 'SELECTION.json') if (root / 'SELECTION.json').exists() else None
        declared = selection['runs'] if selection else p['runs']
        rows = {r['run_id']: r for r in summary['rows']}
        require(set(rows) == {r['run_id'] for r in declared} == set(checks) == set(checkpoints), study)
        for q in declared:
            rid = q['run_id']
            result = read(root / 'execution' / rid / 'RESULT.json')
            trace = json.loads(gzip.decompress((root / 'execution' / rid / 'curves.json.gz').read_bytes()))
            row = rows[rid]
            require(result['protocol_sha256'] == protocol_hash and result['seed'] == p['seed'], rid)
            require(result['checkpoint_sha256'] == checkpoints[rid]['sha256'], rid)
            for key in ('batch', 'threads'):
                require(q[key] == result[key] == row[key], f'{rid}: {key}')
            require(result['parameters'] == p['parameters'][result['arm']], rid)
            require(len(trace['train']) == trace['step'] == result['updates'] == row['updates'], rid)
            require(trace['windows'] == result['updates'] * q['batch'], rid)
            targets = trace['windows'] * 512
            require(targets == trace['training_targets'] == result['training_targets'] == row['training_targets'], rid)
            if selection:
                require(result['selection_sha256'] == sha(root / 'SELECTION.json'), rid)
            wall = cpu = 0.0
            for i, step in enumerate(trace['train'], 1):
                progress = min(1.0, max(0.0, wall / p['train_wall_seconds']))
                factor = (max(.04, progress / .05) if progress < .05 else
                          .1 + .9 * .5 * (1 + math.cos(math.pi * (progress - .05) / .95)))
                close(step['lr'], q.get('peak_lr', .002) * factor, rid, 1e-14)
                require(step['step'] == i and step['training_targets'] == i * q['batch'] * 512, rid)
                require(all(math.isfinite(step[k]) for k in ('ce', 'gradient_norm', 'wall', 'cpu', 'lr')), rid)
                require(step['wall'] > 0 and step['cpu'] > 0, rid)
                wall += step['wall']
                cpu += step['cpu']
                close(step['train_wall_seconds'], wall, rid, 1e-7)
                close(step['train_cpu_seconds'], cpu, rid, 1e-7)
            for record in (trace, result, row):
                close(record['train_wall_seconds'], wall, rid, 1e-7)
                close(record['train_cpu_seconds'], cpu, rid, 1e-7)
            unused = p['train_wall_seconds'] - wall
            require(0 <= unused <= max(.2, 2 * max(x['wall'] for x in trace['train'][-10:])) + 1e-6, rid)
            final = result['final']
            require(len(final['window_ce']) == len(p['final_starts']) == 32 and final['targets'] == 16384, rid)
            close(statistics.mean(final['window_ce']), final['ce'], rid)
            close(row['ce'], final['ce'], rid)
            close(math.exp(final['ce']), final['perplexity'], rid)
            close(row['tokens_per_wall_second'], targets / wall, rid)
            require(result['monitor'] == trace['monitor'], rid)
            error = checks[rid].get('final_checkpoint_replay_max_ce_error', checks[rid].get('replay_max_ce_error'))
            require(error is not None and error < 5e-6, f'{rid}: historical replay receipt')
            updates += result['updates']
        for contrast in summary['contrasts']:
            if 'trial' in contrast:
                t, c = rows[contrast['trial']], rows[contrast['control']]
                close(contrast['ce_change'], t['ce'] - c['ce'], study)
                close(contrast['perplexity_change_percent'], 100 * math.expm1(t['ce'] - c['ce']), study)
                close(contrast['throughput_ratio'], t['tokens_per_wall_second'] / c['tokens_per_wall_second'], study)
        close(summary['total_training_cpu_seconds'], sum(r['train_cpu_seconds'] for r in rows.values()), study, 1e-7)
        studies.append({'run_id': study, 'trials': len(rows), 'status': 'PASS'})

    root = REPO / 'auto-research/results' / scope['studies'][0]
    cells = [r for threads in (1, 2, 4) for r in read(root / 'benchmark' / f'threads{threads}.json')['rows']]
    require(len(cells) == 27 and len({(r['arm'], r['batch'], r['threads']) for r in cells}) == 27, 'Benchmark cells')
    selected = read(root / 'SELECTION.json')['selected']
    for row in cells:
        median = statistics.median(r['wall'] for r in row['samples'])
        close(row['median_step_wall_seconds'], median, 'Benchmark median')
        close(row['tokens_per_wall_second'], row['batch'] * 512 / median, 'Benchmark throughput')
    for arm, chosen in selected.items():
        require(chosen == max((r for r in cells if r['arm'] == arm), key=lambda r: r['tokens_per_wall_second']), arm)

    print(json.dumps(dict(status='PASS', imported_files=imported_files, imported_bytes=imported_bytes,
        source_roots_match=True, python_sources_parsed=syntax_files, studies=studies,
        recorded_training_trials=sum(s['trials'] for s in studies), recorded_updates_checked=updates,
        throughput_cells_checked=len(cells), new_training=False, new_model_evaluation=False,
        checkpoint_replay_rerun=False, checkpoint_receipts='Historical only; external payloads not fetched',
        official_test_accessed=False), indent=2))


if __name__ == '__main__':
    main()
