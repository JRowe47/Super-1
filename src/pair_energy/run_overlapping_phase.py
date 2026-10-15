"""Matched local ablation of independent versus overlapping rotation banks."""
import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import phase_transition_language as pilot
import noncommuting_phase_candidate as candidate


def builders(config):
    width = config['width']
    target = pilot.count(candidate.OverlappingPhaseLM(width, True))
    d = min(range(16, 80, 2), key=lambda v: abs(pilot.count(pilot.TinyTransformer(v)) - target))
    ff = min(range(d, 4*d + 1), key=lambda v: abs(pilot.count(pilot.TinyTransformer(d, v)) - target))
    return {
        'fixed_rotation': lambda: pilot.PhaseLM(width, False),
        'geometric_rotation': lambda: pilot.PhaseLM(width, True),
        'fixed_overlapping': lambda: candidate.OverlappingPhaseLM(width, False),
        'geometric_overlapping': lambda: candidate.OverlappingPhaseLM(width, True),
        'transformer': lambda: pilot.TinyTransformer(d, ff),
    }, d, ff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    pilot.C.update(config)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    checks = candidate.checks()
    makes, d, ff = builders(config)
    assert set(config['arms']) == set(makes)
    counts = {arm: pilot.count(make()) for arm, make in makes.items()}
    assert max(counts.values()) / min(counts.values()) < 1.02
    if args.canary:
        model = makes['geometric_overlapping']()
        optimizer = torch.optim.AdamW(model.parameters())
        inputs = torch.randint(256, (config['batch'], config['context']))
        start = time.perf_counter()
        for _ in range(config['canary_steps']):
            optimizer.zero_grad()
            F.cross_entropy(model(inputs).reshape(-1, 256), inputs.flatten()).backward()
            optimizer.step()
        print(json.dumps({'checks': checks, 'parameters': counts,
                          'estimated_candidate_seconds': (time.perf_counter() - start) * config['steps'] / config['canary_steps']}))
        return
    assert not args.output.exists()
    args.output.mkdir()
    raw = {name: (pilot.DATA / (name + '.txt')).read_bytes() for name in ['train', 'valid']}
    train, valid = (torch.tensor(list(raw[name]), dtype=torch.long) for name in ['train', 'valid'])
    pilot.shared.write(args.output / 'PROTOCOL.json', {
        'config': config, 'checks': checks, 'parameters': counts,
        'transformer_width': d, 'transformer_ff_width': ff,
        'source_sha256': pilot.shared.prior.sha(Path(pilot.__file__)),
        'candidate_source_sha256': pilot.shared.prior.sha(Path(candidate.__file__)),
        'runner_source_sha256': pilot.shared.prior.sha(Path(__file__)),
        'data_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
        'contract': 'Fresh initialization, identical sampled train windows per seed, equal target-token exposure and LR tuning count; matched parameters, not matched FLOPs. Only train and valid read.',
        'hypothesis': 'Overlapping rotation planes permit cross-pair, order-sensitive transport. Fixed-overlapping control separates this from input-dependent geometry.',
        'novelty_established': False, 'official_test_read': False,
    })
    rows, frozen = [], []
    total = len(config['seeds']) * len(config['arms']) * len(config['lrs'])
    for seed in config['seeds']:
        for arm in config['arms']:
            trials = []
            for lr in config['lrs']:
                torch.manual_seed(seed)
                model = makes[arm]()
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
                generator = torch.Generator().manual_seed(seed + config['window_seed_offset'])
                best, history = float('inf'), []
                start = time.perf_counter()
                for step in range(1, config['steps'] + 1):
                    ids = torch.randint(len(train)-config['context']-1, (config['batch'],), generator=generator)
                    x = torch.stack([train[i:i+config['context']] for i in ids])
                    y = torch.stack([train[i+1:i+config['context']+1] for i in ids])
                    for group in optimizer.param_groups:
                        group['lr'] = lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / config['steps'])))
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(x).reshape(-1, 256), y.flatten())
                    assert torch.isfinite(loss)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    optimizer.step()
                    if step % config['validation_interval'] == 0:
                        score = pilot.evaluate(model, valid)
                        history.append({'step': step, **score})
                        if score['ce'] < best:
                            best, best_step, state = score['ce'], step, copy.deepcopy(model.state_dict())
                model.load_state_dict(state)
                path = args.output / f'{seed}-{arm}-{lr}.pt'
                torch.save(state, path)
                row = {'seed': seed, 'arm': arm, 'lr': lr, 'best_step': best_step,
                       'validation': pilot.evaluate(model, valid), 'history': history,
                       'checkpoint': str(path.resolve()), 'sha256': pilot.shared.prior.sha(path),
                       'parameters': pilot.count(model), 'training_seconds': time.perf_counter()-start}
                rows.append(row)
                trials.append(row)
                pilot.shared.write(args.output / 'CANDIDATES.json', rows)
                print(json.dumps({'completed': len(rows), 'total': total, **{k: row[k] for k in ['seed', 'arm', 'lr', 'validation']}}), flush=True)
            frozen.append(min(trials, key=lambda row: row['validation']['ce']))
    pilot.shared.write(args.output / 'FREEZE.json', frozen)
    summary = [{'arm': arm, 'mean_selection_ce': statistics.mean(r['validation']['ce'] for r in frozen if r['arm'] == arm)} for arm in config['arms']]
    pilot.shared.write(args.output / 'SUMMARY.json', summary)
    pilot.shared.write(args.output / 'COMPLETION.json', {'candidates': total, 'official_test_read': False, 'cloud_resources_created': 0})


if __name__ == '__main__':
    main()
