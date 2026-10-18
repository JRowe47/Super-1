"""Preserved learner parity plus an independent scalar/deque replay reference."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib, json, time
from collections import deque
from pathlib import Path
import numpy as np
import torch
import core as C
import feedback_verify as V

ROOT = Path(__file__).resolve().parent


def verify_replay(seed):
    data = C.make_replay_data(seed, 12, 32, 16)
    f, m, audit = data['fifo'], data['mixed'], data['memory_audit']
    original = C.make_data(seed, 12, 32, 16)
    def same(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        return a == b
    assert same(f, original), 'FIFO must be byte-identical to the predecessor generator'
    for key in ('base_x', 'base_y', 'warm_ix', 'current_x', 'current_y', 'query_x',
                'query_old_y', 'query_new_y', 'masks', 'world'):
        assert same(f[key], m[key]), key
    rng = np.random.default_rng(seed + 64000000)
    initial = [-(int(i)+1) for i in rng.choice(1024, 512, replace=False)]
    anchors, recent, old_fifo = initial[:256], deque(initial[256:], maxlen=256), deque(initial, maxlen=512)
    assert len(set(initial)) == 512
    records = {-(i+1): (f['base_x'][i], f['base_y'][i]) for i in range(1024)}
    for t in range(128):
        draw = rng.integers(512, size=16)
        expected = [recent[int(i)%256] for i in draw[:8]] + [anchors[int(i)%256] for i in draw[8:]]
        assert expected == m['replay_ids'][t].tolist()
        assert [old_fifo[int(i)] for i in draw] == f['replay_ids'][t].tolist()
        assert all(i < 16*t for i in expected)
        assert all(i < 0 and i in anchors for i in expected[8:])
        assert torch.equal(m['train_x'][t, :16], f['current_x'][t])
        assert torch.equal(m['train_y'][t, :16], f['current_y'][t])
        for slot, ident in enumerate(expected, 16):
            x, y = records[ident]
            assert torch.equal(m['train_x'][t, slot], x)
            assert torch.equal(m['train_y'][t, slot], y), 'Historical label changed'
        for i in range(16):
            ident = 16*t+i
            records[ident] = (f['current_x'][t, i], f['current_y'][t, i])
            recent.append(ident)
            old_fifo.append(ident)
        assert len(recent)+len(anchors) == len(old_fifo) == 512
    assert not bool(f['stale_replayed_facts'][96:].any())
    assert not bool(audit['mixed_stale_recent'][80:].any())
    assert int(audit['mixed_stale_anchor'][96:].sum()) > 0
    # No phase, label or source metadata enters the learner; only the constructed batch.
    s = C.bounded(C.init(seed))
    old_grad = torch.autograd.grad
    def forbidden(*args, **kwargs):
        raise AssertionError('Exact hidden gradient entered local replay policy')
    try:
        torch.autograd.grad = forbidden
        for schedule in (f, m):
            for policy in ('frozen', 'adaptive'):
                end, _, _ = C.policy_step(s, schedule['train_x'][65], schedule['train_y'][65], policy, shadow={})
                assert all(bool(torch.isfinite(v).all()) for v in end.values())
    finally:
        torch.autograd.grad = old_grad
    return {'seed': seed, 'steps_checked': 128, 'fifo_original_exact': True,
            'mixed_matches_independent_reference': True, 'same_current_and_queries': True,
            'causal_replay': True, 'historical_labels_unchanged': True,
            'equal_512_example_storage': True, 'fixed_32_example_batch': True,
            'late_anchor_stale_binary_targets': int(audit['mixed_stale_anchor'][96:].sum())}


def main():
    begin = time.process_time()
    V.main()  # Includes unchanged baseline equations and cap verification in FP32/FP64.
    result = json.loads((ROOT/'VERIFICATION.json').read_text())
    result['replay_intervention'] = [verify_replay(s) for s in (97987, 12817, 21817)]
    result['cpu_seconds'] = time.process_time()-begin
    result['status'] = 'PASS'
    (ROOT/'VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'replay_verification': 'PASS', 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    main()
