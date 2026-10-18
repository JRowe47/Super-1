"""Audit copied files and saved arithmetic; never import historical runtimes."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values)


def main():
    inventory = read(HERE / 'IMPORT.json')
    parsed = Counter()
    for row in inventory['files']:
        path = ROOT / row['destination']
        require(path.resolve().is_relative_to(ROOT), 'Path escapes repository')
        require(digest(path) == row['sha256'], 'Imported hash: ' + str(path))
        require(path.stat().st_size == row['bytes'], 'Imported size: ' + str(path))
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
            parsed['python'] += 1
        elif path.suffix == '.json':
            read(path)
            parsed['json'] += 1
    lineage = HERE / 'evidence/hybrid-lineage'
    heads = {}
    for path in (lineage / 'heads').glob('*.json'):
        sha = digest(path)
        require(sha == path.name.split('-')[1].split('.')[0], 'Head content address')
        heads[sha] = read(path)
    counts = Counter()
    def audit_vectors(obj):
        if isinstance(obj, dict):
            for prefix, mean_key in [('mps_verify_per_sequence_', 'mps_verify_mean_'),
                                     ('parent_per_sequence_', 'parent_mean_')]:
                if prefix + 'first' in obj:
                    first, second = obj[prefix + 'first'], obj[prefix + 'second']
                    require(first == second, 'Saved duplicate drift')
                    require(mean(first) == obj[mean_key + 'first'], 'Saved first mean')
                    require(mean(second) == obj[mean_key + 'second'], 'Saved second mean')
                    counts['duplicated_vector_pairs'] += 1
            for value in obj.values():
                audit_vectors(value)
        elif isinstance(obj, list):
            for value in obj:
                audit_vectors(value)
    decisions = {}
    for path in (lineage / 'decisions').glob('*.json'):
        sha = digest(path)
        require(sha == path.name.split('.')[0], 'Decision content address')
        d = read(path)
        decisions[sha] = d
        parent = heads[d['base_head_record_sha256']]
        require(parent['checkpoint']['sha256'] == d['base_checkpoint_sha256'], 'Parent checkpoint linkage')
        counts['accepted_decisions' if d['accepted'] else 'rejected_decisions'] += 1
        e = d['local_verify']
        require(e['held_out_generalization_claimed'] is False, 'Unexpected generalization claim')
        audit_vectors(e)
        counts['line_candidates'] += len(e.get('alpha_rows', []))
        for kind in ('block', 'scalar'):
            for row in e.get(kind + '_rows', []):
                counts[kind + '_searches'] += 1
                counts[kind + '_candidates'] += len(row['candidates'])
                counts[kind + '_accepted'] += bool(row['accepted'])
        if 'all_completed_worker_unique_candidates_evaluated_twice' in e:
            counts['partial_candidate_sweeps'] += not e['all_completed_worker_unique_candidates_evaluated_twice']
        if d['accepted']:
            children = [h for h in heads.values() if h['decision_sha256'] == sha]
            require(len(children) == 1, 'Accepted decision head missing')
            child = children[0]
            require(child['checkpoint']['sha256'] == d['child_checkpoint']['sha256'], 'Child checkpoint linkage')
            require(child['parent_head_record_sha256'] == d['base_head_record_sha256'], 'Parent head linkage')
    last = max(heads.values(), key=lambda h: h['model_generation'])
    final = decisions[last['decision_sha256']]['selected_candidate']
    require(final['mps_verify_mean_first'] == final['mps_verify_mean_second'], 'Final duplicate')
    ab = read(HERE / 'evidence/ce-vs-kd/final-result/FINAL_RESULT.json')
    for score in ab['scores'].values():
        require(score['ce_first'] == score['ce_second'], 'A/B duplicate')
        require(mean(score['ce_first']) == score['mean_ce'], 'A/B mean')
    require(ab['promote_winner_selected_output'] is False, 'A/B no-promotion result')
    require(ab['continuation_checkpoint_choice'] == 'unchanged_parent', 'A/B parent retention')
    bench = read(next((HERE / 'evidence/benchmark/results').glob('*.json')))
    benchmarks = {}
    for task, current in bench['current']['tasks'].items():
        prior = bench['prior']['tasks'][task]
        for row in (current, prior):
            require(len(row['examples']) == row['count'], 'Benchmark example count')
            require(sum(bool(e['correct']) for e in row['examples']) / row['count'] == row['accuracy'], 'Benchmark accuracy')
        benchmarks[task] = {'count': current['count'], 'prior_accuracy': prior['accuracy'], 'current_accuracy': current['accuracy']}
    ranking = read(HERE / 'evidence/scalar-cpu-pilot/v3/RANKING.json')
    pilot = read(HERE / 'evidence/scalar-cpu-pilot/v3/RESULT.json')
    require(ranking['autograd_backward_calls'] == 1, 'Preserve diagnostic backward use')
    require(pilot['checkpoint_or_head_writes'] == 0, 'Diagnostic changed checkpoints')
    test_runs = {}
    for path in (HERE / 'evidence/hybrid-tests').glob('*.log'):
        text = path.read_text()
        total = int(re.search(r'Ran (\d+) tests', text).group(1))
        skipped = int(re.search(r'OK \(skipped=(\d+)\)', text).group(1))
        test_runs[path.name] = {'historical_tests_run': total, 'skipped': skipped, 'passed': total - skipped}
    result = {'status': 'PASS_SAVED_FILES_AND_ARITHMETIC_ONLY', 'imported_files': len(inventory['files']),
              'parsed_files': dict(parsed), 'historical_refinement_counts': dict(counts),
              'latest_recorded_model_generation': last['model_generation'],
              'latest_saved_mean_ce': final['mps_verify_mean_first'], 'historical_benchmark_canary': benchmarks,
              'historical_test_logs': test_runs, 'ce_vs_kd': {'promoted': False, 'retained': 'unchanged_parent'},
              'scalar_diagnostic_backward_calls': ranking['autograd_backward_calls'],
              'scalar_diagnostic_screens': len(pilot['screening']),
              'new_model_evaluation': False, 'external_checkpoints_verified': False}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
