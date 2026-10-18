"""Read-only archival audit; never import the solver or open external payloads."""
import ast
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    inventory = read(HERE / 'IMPORT.json')
    python_count = json_count = 0
    for entry in inventory['files']:
        path = ROOT / entry['destination']
        require(path.resolve().is_relative_to(ROOT), 'Import path escapes repository')
        require(digest(path) == entry['sha256'], 'Import hash: ' + str(path))
        require(path.stat().st_size == entry['bytes'], 'Import size: ' + str(path))
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
            python_count += 1
        elif path.suffix == '.json':
            read(path)
            json_count += 1
    manifest_path = HERE / 'provenance/frozen-release/source_manifest.v1.json'
    require(digest(manifest_path) == inventory['source_manifest_sha256'], 'Original manifest hash')
    manifest = read(manifest_path)
    frozen = HERE / 'source/hfw-w560-continuous-optimizer-20260815-v1r1'
    require(len(manifest['files']) == inventory['frozen_source_count'], 'Frozen file count')
    for name, entry in manifest['files'].items():
        require(digest(frozen / name) == entry['sha256'], 'Frozen source hash: ' + name)
        require((frozen / name).stat().st_size == entry['size_bytes'], 'Frozen source size: ' + name)
    solver_name = 'vendor/hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver/branch_search.py'
    require(digest(frozen / 'branch_search.py') == digest(frozen / solver_name)
            == digest(HERE / 'source/mps-runtime' / solver_name), 'Shared solver changed')
    ledger = HERE / 'evidence/mps-20260822/ledger'
    heads = {}
    for path in sorted((ledger / 'heads').glob('*.json')):
        sha = digest(path)
        require(sha == path.name.split('-')[1].split('.')[0], 'Head content address')
        heads[sha] = read(path)
    rows = []
    candidate_count = 0
    for path in sorted((ledger / 'decisions').glob('*.json')):
        require(digest(path) == path.name.split('.')[0], 'Decision content address')
        decision = read(path)
        evidence = decision['local_verify_evidence']
        parent = heads[decision['base_head_record_sha256']]
        require(parent['checkpoint']['sha256'] == decision['base_checkpoint_sha256'], 'Parent linkage')
        require(evidence['held_out_generalization_claimed'] is False, 'Held-out claim changed')
        require(evidence['disclosure'] == 'PERMANENT_REUSED_OPTIMIZATION_DATA_NOT_HELD_OUT', 'Data disclosure')
        parent_means = []
        for suffix in ('first', 'second'):
            values = evidence['parent_per_sequence_' + suffix]
            require(len(values) == evidence['sequences'], 'Parent sequence count')
            mean = math.fsum(values) / len(values)
            require(mean == evidence['parent_mean_' + suffix], 'Parent arithmetic')
            parent_means.append(mean)
        require(evidence['parent_per_sequence_first'] == evidence['parent_per_sequence_second'], 'Parent duplicate')
        thresholds = [x * (1 - evidence['minimum_relative_improvement']) for x in parent_means]
        candidates = evidence['candidate_results']
        require(len(candidates) == evidence['candidate_count'], 'Candidate count')
        eligible = []
        for candidate in candidates:
            means = []
            for suffix in ('first', 'second'):
                values = candidate['per_sequence_' + suffix]
                require(len(values) == evidence['sequences'], 'Candidate sequence count')
                mean = math.fsum(values) / len(values)
                require(mean == candidate['mean_' + suffix], 'Candidate arithmetic')
                means.append(mean)
            require(candidate['per_sequence_first'] == candidate['per_sequence_second'], 'Candidate duplicate')
            require(candidate['thresholds'] == thresholds, 'Threshold arithmetic')
            require(candidate['replicated_mean'] == max(means), 'Replicated mean')
            passes = all(m < t for m, t in zip(means, thresholds))
            require(passes == candidate['passes_minimum_replicated_improvement'], 'Candidate gate')
            if passes:
                eligible.append(candidate)
        selected = min(eligible, key=lambda c: (c['replicated_mean'], c['candidate_order'])) if eligible else None
        order = selected['candidate_order'] if selected else None
        require(order == evidence['selected_candidate_order'], 'Winner selection')
        require(bool(selected) == decision['accepted'], 'Promotion decision')
        if selected:
            children = [h for h in heads.values() if h['decision_sha256'] == digest(path)]
            require(len(children) == 1, 'Promoted head missing')
            child = children[0]
            require(child['checkpoint']['sha256'] == decision['child_checkpoint']['sha256'], 'Child linkage')
            require(child['parent_head_record_sha256'] == decision['base_head_record_sha256'], 'Head chain')
            require(child['selected_candidate_order'] == order, 'Head selection')
        rows.append({'group': decision['group'], 'candidates': len(candidates),
                     'accepted': decision['accepted'], 'parent_mean': parent_means[0],
                     'selected_mean': selected['replicated_mean'] if selected else None})
        candidate_count += len(candidates)
    graphs = read(HERE / 'GRAPHS.json')
    for entry in graphs['files']:
        path = ROOT / entry['destination']
        require(digest(path) == entry['sha256'], 'Graph hash')
        require(path.stat().st_size == entry['bytes'], 'Graph size')
    toy = read(HERE / 'TOY_TESTS.json')
    require(toy['exit_code'] == 0, 'Toy integrity test failure')
    print(json.dumps({'status': 'PASS_ARCHIVAL_HASHES_AND_SAVED_SCORE_ARITHMETIC',
                      'imported_files': len(inventory['files']), 'imported_bytes': sum(e['bytes'] for e in inventory['files']),
                      'frozen_manifest_files': len(manifest['files']), 'parsed_python_files': python_count,
                      'parsed_json_files': json_count, 'decisions': rows, 'heads': len(heads),
                      'candidate_evaluations': candidate_count, 'historical_graphs': len(graphs['files']),
                      'new_checkpoint_evaluation': False,
                      'new_training': False}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
