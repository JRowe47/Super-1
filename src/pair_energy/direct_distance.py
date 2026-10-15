"""Direct relational table -> compact scalar distance function.

No learned endpoint embeddings, coordinate reconstruction, cdist, or norm-based
distance. Raw exemplar inputs are counted as deployed storage. Synthetic only.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
from run import datasets


ARMS = ('dense', 'direct', 'distilled')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DistanceTable(nn.Module):
    """Independent positive off-diagonal entries, not latent coordinates."""
    def __init__(self, n, scale):
        super().__init__()
        self.n = n
        self.register_buffer('ij', torch.triu_indices(n, n, offset=1))
        self.raw = nn.Parameter(torch.randn(self.ij.shape[1]) * .05)
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self):
        upper = self.raw.new_zeros((self.n, self.n))
        upper[self.ij[0], self.ij[1]] = F.softplus(self.raw)
        return upper + upper.T


class Generator(nn.Module):
    """Pair-conditioned scalar; hidden activations are not endpoint coordinates."""
    def __init__(self, c, anchors):
        super().__init__()
        h, k = c['generator_widths']
        self.pair = nn.Sequential(nn.Linear(2*c['input_dim'], h), nn.Tanh(),
                                  nn.Linear(h, k), nn.Tanh(), nn.Linear(k, 1))
        self.log_scale = nn.Parameter(torch.tensor(math.log(c['initial_scale'])))
        self.register_buffer('anchors', anchors.clone())

    def distance(self, x, z):
        a = self.pair(torch.cat((x, z), dim=-1)).squeeze(-1)
        b = self.pair(torch.cat((z, x), dim=-1)).squeeze(-1)
        d = F.softplus((a + b) * .5)
        # Identity and symmetry are architectural constraints, not fitted labels.
        return torch.where((x == z).all(-1), torch.zeros_like(d), d)

    def forward(self, x):
        distances = torch.stack([self.distance(x, anchor.expand_as(x))
                                 for anchor in self.anchors], dim=-1)
        return -self.log_scale.exp() * distances

    def stream(self, x, chunk):
        """Only one exemplar's distances are live; no pair matrix or logits grid."""
        results = []
        for batch in x.split(chunk):
            best = batch.new_full((len(batch),), float('inf'))
            labels = torch.zeros(len(batch), dtype=torch.long)
            for i, anchor in enumerate(self.anchors):
                d = self.distance(batch, anchor.expand_as(batch))
                better = d < best
                labels = torch.where(better, i, labels)
                best = torch.minimum(best, d)
            results.append(labels)
        return torch.cat(results)


class Dense(nn.Module):
    def __init__(self, c, budget):
        super().__init__()
        width, remainder = divmod(budget-c['classes'], c['input_dim']+1+c['classes'])
        if remainder:
            raise ValueError('Config does not permit exact dense storage matching')
        self.net = nn.Sequential(nn.Linear(c['input_dim'], width), nn.Tanh(),
                                 nn.Linear(width, c['classes']))

    def forward(self, x):
        return self.net(x)


def scalars(model):
    return sum(v.numel() for v in model.state_dict().values())


def measurements(model, data):
    model.eval()
    with torch.inference_mode():
        logits = model(data[0])
        return {'ce': F.cross_entropy(logits, data[1]).item(),
                'accuracy': (logits.argmax(-1) == data[1]).float().mean().item()}


def make(arm, seed, c, anchors):
    torch.manual_seed(seed)
    generator = Generator(c, anchors)
    if arm != 'dense':
        return generator
    torch.manual_seed(seed)
    return Dense(c, scalars(generator))


def learn_table(c, seed, train, anchor_ids):
    torch.manual_seed(seed)
    model = DistanceTable(len(train[1]), c['initial_scale'])
    opt = torch.optim.Adam(model.parameters(), lr=c['teacher_lr'])
    i, j = model.ij
    same = train[1][i] == train[1][j]
    rng = torch.Generator().manual_seed(seed+20000)
    start = time.perf_counter()
    for _ in range(c['teacher_steps']):
        opt.zero_grad(set_to_none=True)
        d = model()
        near = (d[i[same], j[same]]-c['close_distance']).relu().square().mean()
        far = (c['far_distance']-d[i[~same], j[~same]]).relu().square().mean()
        a, b, e = torch.randint(len(d), (3, c['triangle_samples']), generator=rng)
        triangle = (d[a, e]-d[a, b]-d[b, e]).relu().square().mean()
        ce = F.cross_entropy(-model.log_scale.exp()*d[:, anchor_ids], train[1])
        loss = ce+c['teacher_penalty']*(near+far)*.5+c['triangle_weight']*triangle
        loss.backward()
        opt.step()
    with torch.no_grad():
        d = model().detach()
        v = torch.where(same, d[i, j]-c['close_distance'], c['far_distance']-d[i, j])
        # Exhaustive triangle audit is cheap at this training-only scale.
        tri_bad = 0
        tri_max = 0.
        for middle in range(len(d)):
            violation = d-d[:, middle, None]-d[middle, None, :]
            tri_bad += int((violation > 1e-5).sum())
            tri_max = max(tri_max, violation.max().item())
        logits = -model.log_scale.exp()*d[:, anchor_ids]
        record = {'independent_distance_parameters': model.raw.numel(),
                  'materialized_distance_bytes': d.numel()*d.element_size(),
                  'training_ce': F.cross_entropy(logits, train[1]).item(),
                  'training_accuracy': (logits.argmax(-1)==train[1]).float().mean().item(),
                  'pair_constraint_satisfied': (v <= 1e-5).float().mean().item(),
                  'triangle_violations': tri_bad, 'triangle_checks': len(d)**3,
                  'max_triangle_violation': tri_max, 'seconds': time.perf_counter()-start,
                  'warning': 'Transductive training table knows labels; not a held-out predictor.'}
    return d, record


def train_candidate(c, seed, arm, lr, train, valid, anchors, table):
    model = make(arm, seed, c, anchors)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed+30000)
    best, best_ce, best_step = copy.deepcopy(model.state_dict()), float('inf'), 0
    history = []
    start = time.perf_counter()
    for step in range(1, c['steps']+1):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train[0]), train[1])
        if arm == 'distilled':
            i, j = torch.randint(len(train[0]), (2, c['distill_pairs']), generator=rng)
            loss = loss+c['distill_weight']*F.mse_loss(model.distance(train[0][i], train[0][j]), table[i, j])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective')
        loss.backward()
        opt.step()
        if step % c['validation_interval'] == 0 or step == c['steps']:
            score = measurements(model, valid)
            history.append({'step': step, **score})
            if score['ce'] < best_ce:
                best_ce, best_step = score['ce'], step
                best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, {'seed': seed, 'arm': arm, 'lr': lr, 'best_step': best_step,
                   'validation': measurements(model, valid), 'history': history,
                   'seconds': time.perf_counter()-start}


def inference_audit(model, x, c):
    with torch.inference_mode():
        expected = model(x).argmax(-1)
        if isinstance(model, Generator):
            actual = model.stream(x, c['query_chunk'])
            assert torch.equal(expected, actual)
        # Single-query latency includes the same Python/CPU execution environment.
        query = x[:1]
        call = (lambda: model.stream(query, c['query_chunk'])) if isinstance(model, Generator) else (lambda: model(query).argmax(-1))
        for _ in range(10):
            call()
        times = []
        for _ in range(c['benchmark_repeats']):
            start = time.perf_counter_ns()
            call()
            times.append((time.perf_counter_ns()-start)/1000)
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            call()
        events = prof.events()
        # Leaf allocation/free events form the incremental CPU tensor live set.
        timeline = sorted((e.time_range.start, e.cpu_memory_usage) for e in events
                          if e.cpu_memory_usage and not e.cpu_children)
        live, peak = 0, 0
        for _, delta in timeline:
            live += delta
            peak = max(peak, live)
        return {'median_single_query_us': statistics.median(times),
                'profiled_incremental_cpu_tensor_peak_bytes': peak,
                'memory_scope': 'Profiler leaf allocation timeline, excludes weights, Python/runtime and allocator reserve; not RSS or GPU memory.',
                'stream_matches_full_predictions': True,
                'max_live_generated_distance_scalars': c['query_chunk'] if isinstance(model, Generator) else 0,
                'distance_evaluations_per_query': len(model.anchors) if isinstance(model, Generator) else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('direct_distance_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'config': c, 'source_sha256': sha(Path(__file__)),
                'data_generator_sha256': sha(Path(__file__).with_name('run.py')),
                'scope': 'Synthetic concept classification, not language modeling or SOTA.',
                'arms': ARMS, 'selection': 'Validation CE, equal two-LR search; all winners frozen before test.',
                'storage': 'FP32 parameters AND exemplar buffers counted; exact deployed tensor byte match.',
                'training': 'Full independent symmetric distance table, training labels only. Student uses CE plus sampled table regression. Unequal compute reported.',
                'inference': 'Scalar symmetric positive pair function on raw inputs, no learned endpoint coordinates. One exemplar at a time.',
                'metric_caveat': 'Triangle inequalities are training penalties, not guaranteed by the generator.',
                'interpretation': 'Tests compression of supervised relationships, not extra independent information at equal storage.',
                'torch': torch.__version__, 'device': 'cpu', 'cloud_resources_created': 0}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2))
    (out/'source.py').write_text(Path(__file__).read_text())
    records, winners, seed_data = [], [], {}
    for seed in c['seeds']:
        (train, valid, test), evidence = datasets(seed, c)
        anchor_ids = torch.stack([torch.where(train[1] == i)[0][0] for i in range(c['classes'])])
        anchors = train[0][anchor_ids]
        table, teacher = learn_table(c, seed, train, anchor_ids)
        torch.save(table, out/f'training-only-table-{seed}.pt')
        (out/f'teacher-{seed}.json').write_text(json.dumps({'metrics': teacher, 'data': evidence}, indent=2))
        seed_data[seed] = (train, valid, test, anchors, table)
        print(json.dumps({'seed': seed, 'teacher': teacher}), flush=True)
        for arm in ARMS:
            candidates = []
            for lr in c['learning_rates']:
                model, row = train_candidate(c, seed, arm, lr, train, valid, anchors, table)
                name = f'{seed}-{arm}-{lr}.pt'
                torch.save(model.state_dict(), out/name)
                row.update(checkpoint=name, sha256=sha(out/name), stored_scalars=scalars(model),
                           deployed_tensor_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),
                           checkpoint_file_bytes=(out/name).stat().st_size)
                records.append(row)
                candidates.append(row)
                (out/'CANDIDATES.json').write_text(json.dumps(records, indent=2))
                print(json.dumps({k: v for k, v in row.items() if k != 'history'}), flush=True)
            winners.append(copy.deepcopy(min(candidates, key=lambda r: r['validation']['ce'])))
    assert len(set(r['deployed_tensor_bytes'] for r in records)) == 1
    (out/'FREEZE.json').write_text(json.dumps(winners, indent=2))
    for row in winners:
        train, valid, test, anchors, table = seed_data[row['seed']]
        model = make(row['arm'], row['seed'], c, anchors)
        path = out/row['checkpoint']
        assert sha(path) == row['sha256']
        model.load_state_dict(torch.load(path, weights_only=True))
        assert measurements(model, valid) == row['validation']
        row['test'] = measurements(model, test)
        assert row['test'] == measurements(model, test)
        row['inference'] = inference_audit(model, test[0], c)
        if isinstance(model, Generator):
            with torch.inference_mode():
                distances = torch.stack([model.distance(train[0], z.expand_as(train[0])) for z in train[0]], dim=1)
                row['table_rmse'] = (distances-table).square().mean().sqrt().item()
                rng = torch.Generator().manual_seed(row['seed']+40000)
                a, b, e = torch.randint(len(test[0]), (3, c['triangle_samples']), generator=rng)
                dab = model.distance(test[0][a], test[0][b])
                dbe = model.distance(test[0][b], test[0][e])
                dae = model.distance(test[0][a], test[0][e])
                row['heldout_triangle_violation_fraction'] = ((dae-dab-dbe) > 1e-5).float().mean().item()
                assert torch.equal(dab, model.distance(test[0][b], test[0][a]))
                assert (model.distance(test[0], test[0]) == 0).all()
        assert sha(path) == row['sha256']
    (out/'RESULT.json').write_text(json.dumps(winners, indent=2))
    lines = ['# Direct-distance compression pilot', '', protocol['scope'], '',
             'Full results include all candidates, frozen selections, hashes and training-only tables. Deployments contain only compact model tensors and (for distance models) eight raw exemplars. No table is retained in the deployment.', '',
             '| Arm | Mean held-out CE | Accuracy | Deployed tensor bytes | Median query us (seed mean) |',
             '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        rows = [r for r in winners if r['arm'] == arm]
        avg = lambda f: statistics.mean(f(r) for r in rows)
        lines.append(f"| {arm} | {avg(lambda r:r['test']['ce']):.6f} | {avg(lambda r:r['test']['accuracy']):.4%} | {rows[0]['deployed_tensor_bytes']} | {avg(lambda r:r['inference']['median_single_query_us']):.2f} |")
    lines += ['', 'Training uses 256 examples, validation 128 and test 512 per seed with disjoint nuisance-style sets. Selection uses validation only. This is reused synthetic task design, not independent real-world evidence.', '',
              'The distilled generator additionally sees a large label-trained relational table. The direct generator is the identical deployed architecture trained with classification CE only. Dense matches all deployed FP32 tensor bytes including the exemplars. Training computation and inference latency are not matched.', '',
              'Distances are generated directly, but the compact generator is still parameterized by ordinary neural-network weights. Compression does not create extra independent bits. Symmetry, positivity and zero self-distance are architectural; triangle inequalities are not guaranteed. Teacher training accuracy is transductive memorization, not generalization.', '',
              'Latency is CPU single-query Python execution, not an optimized GPU prediction. Tensor-memory profiling excludes runtime/allocator overhead. No language-model or SOTA conclusion follows.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'COMPLETION.json').write_text(json.dumps({'candidates': len(records), 'winners': len(winners),
        'duplicate_test_drift': 0, 'exact_deployed_tensor_byte_match': True,
        'streaming_predictions_verified': True, 'cloud_resources_created': 0}, indent=2))


if __name__ == '__main__':
    main()
