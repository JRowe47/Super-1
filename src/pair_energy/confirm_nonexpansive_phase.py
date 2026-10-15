"""Frozen, duplicate-FP32 comparison on the declared reserved byte region."""
import argparse
import hashlib
import importlib
import io
import json
import math
import statistics
from pathlib import Path

import torch
import run_overlapping_phase as training
import phase_transition_language as pilot


def verify_factory(protocol, experiment):
    """Bind a relocated factory to its sealed name and exact content hash."""
    factory = Path(experiment.__file__)
    matches = [digest for path, digest in protocol['checks']['source_hashes'].items()
               if Path(path).name == factory.name]
    assert len(matches) == 1, 'Factory identity must be uniquely sealed'
    assert pilot.shared.prior.sha(factory) == matches[0], 'Factory source drift'
    actual = experiment.checks()['source_hashes']
    sealed = protocol['checks']['source_hashes']
    def identities(hashes):
        result = {}
        for path, digest in hashes.items():
            name = Path(path).name
            assert name not in result, 'Ambiguous sealed dependency identity'
            result[name] = digest
        return result
    assert identities(actual) == identities(sealed), 'Loaded factory dependency drift'


def verify_score(score, config):
    assert set(score) == {'ce', 'bpc', 'scored_bytes'}, 'Malformed selection score'
    assert score['scored_bytes'] == config['context'] * config['validation_windows']
    assert math.isfinite(score['ce']) and score['ce'] >= 0
    assert math.isfinite(score['bpc']) and score['bpc'] == score['ce'] / math.log(2)


def verify_trial(row, config, parameters, source):
    expected_name = f"{row['seed']}-{row['arm']}-{row['lr']}.pt"
    assert Path(row['checkpoint']).name == expected_name, 'Checkpoint assignment drift'
    checkpoint = source / expected_name
    assert checkpoint.resolve() == source.resolve() / expected_name, 'Checkpoint aliasing'
    assert pilot.shared.prior.sha(checkpoint) == row['sha256']
    assert row['parameters'] == parameters[row['arm']], 'Parameter-count drift'
    assert math.isfinite(row['training_seconds']) and row['training_seconds'] > 0
    schedule = list(range(config['validation_interval'], config['steps'] + 1,
                          config['validation_interval']))
    assert schedule and [score['step'] for score in row['history']] == schedule, 'Incomplete training history'
    for score in row['history']:
        verify_score({key: value for key, value in score.items() if key != 'step'}, config)
    verify_score(row['validation'], config)
    best = min(row['history'], key=lambda score: score['ce'])
    assert row['best_step'] == best['step']
    assert row['validation'] == {key: value for key, value in best.items() if key != 'step'}, 'Selection score drift'


def verify_state(model, state):
    reference = model.state_dict()
    assert set(state) == set(reference), 'Checkpoint state-key drift'
    for name, expected in reference.items():
        value = state[name]
        assert isinstance(value, torch.Tensor)
        assert value.shape == expected.shape, 'Checkpoint state-shape drift'
        assert value.dtype == expected.dtype, 'Serialized checkpoint precision drift'
        assert torch.isfinite(value).all(), 'Nonfinite checkpoint state'


def verify_confirmation_plan(source, protocol, candidate_arm, windows):
    plan_path = source / 'CONFIRMATION_PLAN.json'
    if not plan_path.exists():
        assert 'promotion_contract' not in protocol['config'], 'Registered promotion requires a machine-readable confirmation plan'
        return
    plan = json.loads(plan_path.read_text())
    assert plan['source_protocol_sha256'] == pilot.shared.prior.sha(source / 'PROTOCOL.json')
    assert candidate_arm == plan['candidate_arm'], 'Promoted hypothesis drift'
    assert windows == plan['reserved_confirmation_windows']
    assert protocol['config']['reserved_confirmation_start'] == plan['reserved_confirmation_start']
    assert protocol['data_sha256']['valid'] == plan['validation_sha256']
    assert plan['precision'] == 'FP32' and plan['duplicate_evaluations'] == 2


def prepare_models(source, frozen, makes):
    """Preflight every frozen snapshot before consuming any reserved scores."""
    prepared = []
    for row in frozen:
        model = makes[row['arm']]().float().eval()
        checkpoint_bytes = (source / Path(row['checkpoint']).name).read_bytes()
        assert hashlib.sha256(checkpoint_bytes).hexdigest() == row['sha256'], 'Frozen snapshot drift'
        state = torch.load(io.BytesIO(checkpoint_bytes), weights_only=True, map_location='cpu')
        verify_state(model, state)
        model.load_state_dict(state)
        assert all(p.dtype == torch.float32 for p in model.parameters())
        prepared.append((row, model))
    return prepared


def inspect(source):
    protocol = json.loads((source / 'PROTOCOL.json').read_text())
    completion = json.loads((source / 'COMPLETION.json').read_text())
    candidates = json.loads((source / 'CANDIDATES.json').read_text())
    frozen = json.loads((source / 'FREEZE.json').read_text())
    config = protocol['config']
    expected = {(seed, arm, lr) for seed in config['seeds']
                for arm in config['arms'] for lr in config['lrs']}
    assert completion['official_test_read'] is False
    assert completion['candidates'] == len(expected) == len(candidates)
    assert {(r['seed'], r['arm'], r['lr']) for r in candidates} == expected
    assert len(frozen) == len(config['seeds']) * len(config['arms'])
    assert {(r['seed'], r['arm']) for r in frozen} == {(s, a) for s in config['seeds'] for a in config['arms']}
    paths = {
        pilot.__file__: protocol['source_sha256'],
        training.__file__: protocol['runner_source_sha256'],
        training.candidate.__file__: protocol['candidate_source_sha256'],
        **protocol['checks']['source_hashes'],
    }
    for path, digest in paths.items():
        assert pilot.shared.prior.sha(Path(path)) == digest, f'Source drift: {path}'
    for row in candidates:
        verify_trial(row, config, protocol['parameters'], source)
    for row in frozen:
        choices = [r for r in candidates if (r['seed'], r['arm']) == (row['seed'], row['arm'])]
        assert row == min(choices, key=lambda r: r['validation']['ce']), 'Selection must remain validation-only'
    assert config['reserved_confirmation_start'] > config['validation_start'] + config['context'] * config['validation_windows']
    return protocol, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, required=True)
    parser.add_argument('--experiment-module', choices=['run_nonexpansive_phase', 'run_input_generated_phase', 'run_relative_phase_key', 'run_relative_phase_key_scale', 'run_pair_energy_phase'],
                        default='run_nonexpansive_phase')
    parser.add_argument('--candidate-arm', default='bounded_overlapping')
    args = parser.parse_args()
    assert args.windows > 0 and not args.output.exists()
    protocol, frozen = inspect(args.source)
    config = protocol['config']
    verify_confirmation_plan(args.source, protocol, args.candidate_arm, args.windows)
    assert args.candidate_arm in config['arms']
    if 'reserved_confirmation_windows' in config:
        assert args.windows == config['reserved_confirmation_windows'], 'Confirmation extent is preregistered'
    experiment = importlib.import_module(args.experiment_module)
    verify_factory(protocol, experiment)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    pilot.C.update(config)
    pilot.C.update(validation_start=config['reserved_confirmation_start'], validation_windows=args.windows)
    makes, _, _ = experiment.builders(config)
    assert set(makes) == set(config['arms'])
    assert {arm: pilot.count(make()) for arm, make in makes.items()} == protocol['parameters']
    prepared = prepare_models(args.source, frozen, makes)
    raw = (pilot.DATA / 'valid.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol['data_sha256']['valid']
    assert config['reserved_confirmation_start'] + args.windows * config['context'] < len(raw)
    data = torch.tensor(list(raw), dtype=torch.long)
    args.output.mkdir()
    pilot.shared.write(args.output / 'PROTOCOL.json', {
        'source_protocol_sha256': pilot.shared.prior.sha(args.source / 'PROTOCOL.json'),
        'freeze_sha256': pilot.shared.prior.sha(args.source / 'FREEZE.json'),
        'confirmation_source_sha256': pilot.shared.prior.sha(Path(__file__)),
        'experiment_module': args.experiment_module, 'candidate_arm': args.candidate_arm,
        'validation_start': config['reserved_confirmation_start'],
        'scored_bytes': args.windows * config['context'], 'precision': 'FP32',
        'duplicate_evaluations': 2, 'official_test_read': False,
        'confirmation_plan_sha256': pilot.shared.prior.sha(args.source / 'CONFIRMATION_PLAN.json') if (args.source / 'CONFIRMATION_PLAN.json').exists() else None,
    })
    rows = []
    confirmation_config = dict(config, validation_windows=args.windows)
    for row, model in prepared:
        first, second = pilot.evaluate(model, data), pilot.evaluate(model, data)
        verify_score(first, confirmation_config)
        verify_score(second, confirmation_config)
        assert first == second, 'Duplicate drift forbids promotion'
        rows.append({'seed': row['seed'], 'arm': row['arm'], 'sha256': row['sha256'],
                     'validation': first, 'duplicate_drift': 0})
        pilot.shared.write(args.output / 'RESULT.json', rows)
        print(json.dumps(rows[-1]), flush=True)
    groups = [{'arm': arm, 'ce': statistics.mean(r['validation']['ce'] for r in rows if r['arm'] == arm)} for arm in config['arms']]
    for group in groups:
        group['bpc'] = group['ce'] / math.log(2)
    pairs = []
    for seed in config['seeds']:
        scores = {r['arm']: r['validation']['ce'] for r in rows if r['seed'] == seed}
        pairs.append({'seed': seed, 'candidate_minus_controls_ce': {
            arm: scores[args.candidate_arm] - score for arm, score in scores.items() if arm != args.candidate_arm}})
    pilot.shared.write(args.output / 'SUMMARY.json', {'groups': groups, 'pairs': pairs, 'duplicate_drift': 0,
        'candidate_arm': args.candidate_arm,
        'candidate_beats_all_controls_all_seeds': all(delta < 0 for pair in pairs for delta in pair['candidate_minus_controls_ce'].values()),
        'official_test_read': False, 'novelty_established': False})
    pilot.shared.write(args.output / 'COMPLETION.json', {'checkpoints': len(rows), 'duplicate_drift': 0,
        'official_test_read': False, 'cloud_resources_created': 0})


if __name__ == '__main__':
    main()
