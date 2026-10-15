"""Predeclared inference-budget comparison of direct relational compression.

Two synthetic capacity conditions and one real digits dataset. Test evaluation
occurs only after every task/seed/budget/arm checkpoint is validation-selected.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
import direct_distance as prior


def write(path, value):
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def data(task, seed, c):
    rng = torch.Generator().manual_seed(seed+10000)
    if task == 'digits':
        from sklearn.datasets import load_digits
        dataset = load_digits()
        # Fixed public pixel range, no fitted preprocessing or test statistics.
        x = torch.tensor(dataset.data, dtype=torch.float32)/16
        y = torch.tensor(dataset.target, dtype=torch.long)
        groups = {}
        keep = []
        for i, sample in enumerate(x):
            key = digest(sample)
            if key in groups:
                assert y[i] == y[groups[key]]
            else:
                groups[key] = i
                keep.append(i)
        x, y = x[keep], y[keep]
        indices = [[], [], []]
        for label in range(int(y.max())+1):
            ids = torch.where(y == label)[0]
            ids = ids[torch.randperm(len(ids), generator=rng)]
            train_n = int(len(ids)*c['digits_train_fraction'])
            valid_n = int(len(ids)*c['digits_validation_fraction'])
            for split, chunk in enumerate((ids[:train_n], ids[train_n:train_n+valid_n], ids[train_n+valid_n:])):
                indices[split].extend(chunk.tolist())
        splits = [(x[ids], y[ids]) for ids in indices]
        metadata = {'source': 'scikit-learn bundled UCI optical handwritten digits',
                    'source_arrays_sha256': digest(x), 'duplicate_inputs_removed': len(dataset.data)-len(keep),
                    'split': 'stratified per seed; exact duplicates grouped before splitting'}
    else:
        classes = int(task.removeprefix('context'))
        bits_n = int(math.log2(classes))
        assert 2**bits_n == classes
        styles = torch.tensor(list(itertools.product((-1., 1.), repeat=4)))
        parity = (styles > 0).sum(1) % 2
        odd = torch.where(parity == 1)[0]
        allowed_sets = (torch.where(parity == 0)[0], odd[:len(odd)//2], odd[len(odd)//2:])
        mixing = torch.linalg.qr(torch.randn(c['synthetic_input_dim'], bits_n+5, generator=rng)).Q
        splits = []
        for count, allowed in zip((c['synthetic_train_per_class'], c['synthetic_validation_per_class'],
                                    c['synthetic_test_per_class']), allowed_sets):
            xs, ys = [], []
            for label in range(classes):
                bits = torch.tensor([(label >> shift) & 1 for shift in range(bits_n-1, -1, -1)])
                context = torch.arange(count) % 2
                semantic = bits.expand(count, -1).clone()
                semantic[:, -1] ^= context
                style_ids = allowed[torch.arange(count)//2 % len(allowed)]
                latent = torch.cat((semantic.float()*2-1, (context.float()*2-1)[:, None], styles[style_ids]), 1)
                xs.append(torch.tanh(latent@mixing.T)+c['synthetic_noise_std']*torch.randn(count, c['synthetic_input_dim'], generator=rng))
                ys.append(torch.full((count,), label, dtype=torch.long))
            splits.append((torch.cat(xs), torch.cat(ys)))
        metadata = {'source': 'context-dependent class bits with disjoint nuisance-style combinations',
                    'style_sets': [a.tolist() for a in allowed_sets], 'classes': classes,
                    'caveat': 'Context8 extends the previous pilot task; context32 is a related capacity condition, not an independent domain.'}
    sets = [set(digest(sample) for sample in x) for x, _ in splits]
    assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i))
    metadata['splits'] = [{'examples': len(y), 'inputs_sha256': digest(x), 'labels_sha256': digest(y)} for x, y in splits]
    return splits, metadata


def architecture(c, dim, classes, h, kind):
    k = max(1, h//c['hidden_ratio'])
    if kind == 'direct':
        count = (2*dim+1)*h+(h+1)*k+(k+1)+1+dim*classes
    elif kind == 'coordinates':
        e = c['coordinate_dim']
        count = (dim+1)*h+(h+1)*k+(k+1)*e+classes*e+1
    else:
        count = (dim+1)*h+(h+1)*k+(k+1)*classes+1
    return count, k


class Student(nn.Module):
    def __init__(self, c, dim, classes, budget, arm, anchors):
        super().__init__()
        self.kind = 'direct' if arm in ('direct', 'distilled') else ('coordinates' if arm == 'coordinates_teacher' else 'dense')
        candidates = [(architecture(c, dim, classes, h, self.kind)[0], h) for h in range(1, budget+1)
                      if architecture(c, dim, classes, h, self.kind)[0] <= budget]
        if not candidates:
            raise ValueError('No architecture fits deployment cap')
        expected, h = max(candidates)
        _, k = architecture(c, dim, classes, h, self.kind)
        self.shape = [h, k]
        self.log_scale = nn.Parameter(torch.tensor(math.log(c['initial_scale'] if self.kind != 'dense' else 1.)))
        if self.kind == 'direct':
            self.pair = nn.Sequential(nn.Linear(2*dim, h), nn.Tanh(), nn.Linear(h, k), nn.Tanh(), nn.Linear(k, 1))
            self.register_buffer('anchors', anchors.clone())
        else:
            self.encoder = nn.Sequential(nn.Linear(dim, h), nn.Tanh(), nn.Linear(h, k), nn.Tanh())
            if self.kind == 'coordinates':
                self.project = nn.Linear(k, c['coordinate_dim'])
                self.prototypes = nn.Parameter(torch.randn(classes, c['coordinate_dim']))
            else:
                self.output = nn.Linear(k, classes)
        assert prior.scalars(self) == expected
        assert expected <= budget

    def coordinates(self, x):
        return F.normalize(self.project(self.encoder(x)), dim=-1)

    def distance(self, x, z):
        if self.kind == 'coordinates':
            return (self.coordinates(x)-self.coordinates(z)).square().sum(-1)
        assert self.kind == 'direct'
        a = self.pair(torch.cat((x, z), -1)).squeeze(-1)
        b = self.pair(torch.cat((z, x), -1)).squeeze(-1)
        return torch.where((x == z).all(-1), torch.zeros_like(a), F.softplus((a+b)*.5))

    def forward(self, x):
        if self.kind == 'dense':
            return self.log_scale.exp()*self.output(self.encoder(x))
        if self.kind == 'coordinates':
            distances = (self.coordinates(x)[:, None, :]-F.normalize(self.prototypes, dim=-1)[None, :, :]).square().sum(-1)
        else:
            left = x[:, None, :].expand(-1, len(self.anchors), -1)
            right = self.anchors[None, :, :].expand(len(x), -1, -1)
            distances = self.distance(left, right)
        return -self.log_scale.exp()*distances

    def stream(self, x):
        if self.kind != 'direct':
            return self(x).argmax(-1)
        best = x.new_full((len(x),), float('inf'))
        labels = torch.zeros(len(x), dtype=torch.long)
        for i, anchor in enumerate(self.anchors):
            d = self.distance(x, anchor.expand_as(x))
            labels = torch.where(d < best, i, labels)
            best = torch.minimum(best, d)
        return labels


class Auxiliary(nn.Module):
    """Training-only relational head: equal pair supervision for dense student."""
    def __init__(self, hidden, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*hidden, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, a, b):
        return F.softplus((self.net(torch.cat((a, b), -1))+self.net(torch.cat((b, a), -1))).squeeze(-1)*.5)


def evaluate(model, split):
    with torch.inference_mode():
        logits = torch.cat([model(x) for x in split[0].split(128)])
        return {'ce': F.cross_entropy(logits, split[1]).item(),
                'accuracy': (logits.argmax(-1)==split[1]).float().mean().item()}


def prepare(c, task, seed, folder):
    splits, evidence = data(task, seed, c)
    x, y = splits[0]
    classes = int(y.max())+1
    anchor_ids = torch.stack([torch.where(y == label)[0][0] for label in range(classes)])
    torch.manual_seed(seed)
    teacher = prior.DistanceTable(len(x), c['initial_scale'])
    opt = torch.optim.Adam(teacher.parameters(), lr=c['teacher_lr'])
    i, j = teacher.ij
    same = y[i] == y[j]
    rng = torch.Generator().manual_seed(seed+20000)
    start = time.perf_counter()
    for _ in range(c['teacher_steps']):
        opt.zero_grad(set_to_none=True)
        d = teacher()
        ce = F.cross_entropy(-teacher.log_scale.exp()*d[:, anchor_ids], y)
        near = (d[i[same], j[same]]-c['close_distance']).relu().square().mean()
        far = (c['far_distance']-d[i[~same], j[~same]]).relu().square().mean()
        a, b, z = torch.randint(len(x), (3, c['triangle_samples']), generator=rng)
        triangle = (d[a, z]-d[a, b]-d[b, z]).relu().square().mean()
        loss = ce+c['teacher_penalty']*(near+far)*.5+c['triangle_weight']*triangle
        loss.backward()
        opt.step()
    with torch.no_grad():
        matrix = teacher().detach()
        target_logits = (-teacher.log_scale.exp()*matrix[:, anchor_ids]).detach()
    payload = {'train': splits[0], 'valid': splits[1], 'anchors': x[anchor_ids],
               'matrix': matrix, 'target_logits': target_logits}
    torch.save(payload, folder/'training-only.pt')
    write(folder/'DATA.json', evidence)
    write(folder/'TEACHER.json', {'independent_entries': teacher.raw.numel(),
        'materialized_table_bytes': matrix.numel()*4, 'training_accuracy': (target_logits.argmax(-1)==y).float().mean().item(),
        'seconds': time.perf_counter()-start, 'sha256': prior.sha(folder/'training-only.pt'),
        'warning': 'Label-trained transductive teacher, not a held-out predictor.'})


def train(c, task, seed, budget, arm, lr, payload, folder, timing_only=False):
    x, y = payload['train']
    dim, classes = x.shape[-1], int(y.max())+1
    torch.manual_seed(seed)
    model = Student(c, dim, classes, budget, arm, payload['anchors'])
    auxiliary = Auxiliary(model.shape[-1], c['auxiliary_width']) if arm == 'dense_teacher' else None
    params = list(model.parameters())+(list(auxiliary.parameters()) if auxiliary is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    rng = torch.Generator().manual_seed(seed+30000)
    pair_rng = torch.Generator().manual_seed(seed+40000)
    best_ce, best_state, best_step = float('inf'), None, None
    history = []
    start = time.perf_counter()
    steps = c['canary_steps'] if timing_only else c['steps']
    for step in range(1, steps+1):
        factor = min(1., step/c['warmup_steps']) if step <= c['warmup_steps'] else c['minimum_lr_ratio']+(1-c['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-c['warmup_steps'])/(c['steps']-c['warmup_steps'])))
        for group in opt.param_groups:
            group['lr'] = lr*factor
        ids = torch.randint(len(x), (c['batch_size'],), generator=rng)
        opt.zero_grad(set_to_none=True)
        logits = model(x[ids])
        loss = F.cross_entropy(logits, y[ids])
        if arm.endswith('teacher') or arm == 'distilled':
            t = c['logit_distill_temperature']
            kd = F.kl_div(F.log_softmax(logits/t, -1), F.softmax(payload['target_logits'][ids]/t, -1), reduction='batchmean')*t*t
            a, b = torch.randint(len(x), (2, c['distill_pairs']), generator=pair_rng)
            if auxiliary is None:
                predicted = model.distance(x[a], x[b])
            else:
                predicted = auxiliary(model.encoder(x[a]), model.encoder(x[b]))
                predicted = torch.where(a == b, torch.zeros_like(predicted), predicted)
            loss = loss+c['logit_distill_weight']*kd+c['distill_weight']*F.mse_loss(predicted, payload['matrix'][a, b])
        assert torch.isfinite(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(params, c['gradient_clip'])
        opt.step()
        if not timing_only and (step % c['validation_interval'] == 0 or step == steps):
            score = evaluate(model, payload['valid'])
            history.append({'step': step, **score})
            if score['ce'] < best_ce:
                best_ce, best_step = score['ce'], step
                best_state = copy.deepcopy(model.state_dict())
    seconds = time.perf_counter()-start
    if timing_only:
        return {'task': task, 'arm': arm, 'steps': steps, 'seconds': seconds,
                'estimated_candidate_seconds': seconds*c['steps']/steps,
                'stored_scalars': prior.scalars(model)}
    model.load_state_dict(best_state)
    filename = f'{budget}-{arm}-{lr}.pt'
    torch.save(best_state, folder/filename)
    row = {'task': task, 'seed': seed, 'budget_scalars': budget, 'arm': arm, 'lr': lr,
           'checkpoint': str((folder/filename).resolve()), 'sha256': prior.sha(folder/filename),
           'stored_scalars': prior.scalars(model), 'deployed_tensor_bytes': prior.scalars(model)*4,
           'checkpoint_bytes': (folder/filename).stat().st_size, 'shape': model.shape,
           'training_only_auxiliary_scalars': sum(p.numel() for p in auxiliary.parameters()) if auxiliary else 0,
           'validation': evaluate(model, payload['valid']), 'best_step': best_step,
           'history': history, 'training_seconds': seconds}
    return row


def job(config, task, seed, output):
    c = config
    torch.set_num_threads(c['threads_per_worker'])
    torch.use_deterministic_algorithms(True)
    folder = Path(output)/f'{task}-{seed}'
    folder.mkdir(exist_ok=False)
    prepare(c, task, seed, folder)
    payload = torch.load(folder/'training-only.pt', weights_only=True)
    records, winners = [], []
    for budget in c['scalar_budgets']:
        for arm in c['arms']:
            candidates = []
            for lr in c['learning_rates']:
                row = train(c, task, seed, budget, arm, lr, payload, folder)
                candidates.append(row)
                records.append(row)
                write(folder/'CANDIDATES.json', records)
                print(json.dumps({k: row[k] for k in ('task', 'seed', 'budget_scalars', 'arm', 'lr', 'validation', 'training_seconds')}), flush=True)
            winners.append(min(candidates, key=lambda r: r['validation']['ce']))
    write(folder/'SELECTED.json', winners)
    return winners


def profile_call(call, repeats):
    import tempfile
    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        for _ in range(10):
            call()
        times = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            call()
            times.append((time.perf_counter_ns()-start)/1000)
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            call()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'trace.json'
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())['traceEvents']
    events = [e for e in events if e.get('name') == '[memory]' and 'Bytes' in e.get('args', {})]
    assert events
    live, peak = 0, 0
    for e in sorted(events, key=lambda e: e['ts']):
        live += e['args']['Bytes']
        peak = max(peak, live)
    return {'median_single_query_us': statistics.median(times), 'incremental_cpu_tensor_peak_bytes': peak}


def assess(c, selected, output):
    results = []
    for row in selected:
        r = copy.deepcopy(row)
        (train_split, valid, test), evidence = data(r['task'], r['seed'], c)
        anchors = torch.zeros(int(train_split[1].max())+1, train_split[0].shape[-1])
        torch.manual_seed(r['seed'])
        model = Student(c, anchors.shape[-1], len(anchors), r['budget_scalars'], r['arm'], anchors)
        path = Path(r['checkpoint'])
        assert prior.sha(path) == r['sha256']
        model.load_state_dict(torch.load(path, weights_only=True))
        assert evaluate(model, valid) == r['validation']
        r['test'] = evaluate(model, test)
        assert evaluate(model, test) == r['test']
        x = test[0][:1]
        r['inference'] = profile_call(lambda: model(x).argmax(-1), c['benchmark_repeats'])
        if model.kind == 'direct':
            with torch.inference_mode():
                reference = torch.cat([model(b).argmax(-1) for b in test[0].split(128)])
                streamed = torch.cat([model.stream(b) for b in test[0].split(128)])
                assert torch.equal(reference, streamed)
            r['streaming'] = profile_call(lambda: model.stream(x), c['benchmark_repeats'])
        results.append(r)
        write(output/'RESULT.json', results)
    write(output/'COMPLETION.json', {'selected': len(results), 'duplicate_test_drift': 0,
          'cloud_resources_created': 0, 'test_opened_after_global_freeze': True})


def canary(c, out):
    timings = []
    for task in c['tasks']:
        splits, _ = data(task, c['seeds'][0], c)
        x, y = splits[0]
        anchors = torch.stack([x[torch.where(y == i)[0][0]] for i in range(int(y.max())+1)])
        # Timing-only synthetic teacher, no canary quality-based decisions.
        payload = {'train': splits[0], 'valid': splits[1], 'anchors': anchors,
                   'matrix': torch.ones(len(x), len(x)), 'target_logits': torch.zeros(len(x), len(anchors))}
        for arm in c['arms']:
            row = train(c, task, c['seeds'][0], max(c['scalar_budgets']), arm,
                        c['learning_rates'][0], payload, out, timing_only=True)
            timings.append(row)
            print(json.dumps(row), flush=True)
    estimated = sum(r['estimated_candidate_seconds'] for r in timings)*len(c['seeds'])*len(c['scalar_budgets'])*len(c['learning_rates'])
    write(out/'CANARY.json', {'timings': timings, 'serial_training_seconds_upper_budget_estimate': estimated,
                            'caveat': 'Largest architectures only; excludes evaluation, tables and process contention. No quality scores used.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('capacity_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    torch.set_num_threads(c['threads_per_worker'])
    args.output.mkdir(exist_ok=False, parents=True)
    protocol = {'config': c, 'source_sha256': prior.sha(Path(__file__)),
                'distance_table_source_sha256': prior.sha(Path(prior.__file__)),
                'scope': 'Fixed-budget feasibility sweep, not a convergence or breakthrough guarantee.',
                'comparison': 'Common deployed FP32 storage caps; actual sizes reported, no dummy padding. Same steps, input batches and LR search; training FLOPs differ.',
                'teacher_control': 'All teacher arms receive identical class-logit KD and pair-distance targets; dense uses an auxiliary pair head discarded at deployment.',
                'selection': 'Validation CE; all models globally frozen before held-out assessment.',
                'inference_memory': 'Actual CPU tensor allocation events, plus deployed tensors; runtime/allocator/RSS excluded.',
                'criteria_note': 'Proposed engineering screening thresholds, not universal definitions of breakthrough. No test-based threshold changes.',
                'cloud_resources_created': 0}
    write(args.output/'PROTOCOL.json', protocol)
    (args.output/'source.py').write_text(Path(__file__).read_text())
    if args.canary:
        canary(c, args.output)
        return
    selected = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=c['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(job, c, task, seed, str(args.output.resolve())) for task in c['tasks'] for seed in c['seeds']]
        for future in concurrent.futures.as_completed(futures):
            selected.extend(future.result())
            write(args.output/'PROGRESS.json', {'completed_jobs': len(selected)//(len(c['arms'])*len(c['scalar_budgets'])), 'selected_models': len(selected)})
    selected.sort(key=lambda r: (r['task'], r['seed'], r['budget_scalars'], r['arm']))
    write(args.output/'FREEZE.json', selected)
    baselines = []
    for task in c['tasks']:
        for seed in c['seeds']:
            for budget in c['scalar_budgets']:
                choices = [r for r in selected if r['task'] == task and r['seed'] == seed
                           and r['budget_scalars'] == budget
                           and r['arm'] in ('dense', 'dense_teacher', 'coordinates_teacher')]
                baselines.append(min(choices, key=lambda r:r['validation']['ce']))
    write(args.output/'BASELINE_FREEZE.json', baselines)
    assess(c, selected, args.output)


if __name__ == '__main__':
    main()
