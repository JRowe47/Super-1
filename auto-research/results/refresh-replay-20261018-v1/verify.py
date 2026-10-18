"""Preserved learner/replay checks and independent causal refresh reference."""
import replay_verify as V
from replay_verify import np, torch, C, ROOT, deque, json, time
import replay_core as P


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return a == b


def verify_refresh(seed):
    p = 160  # Enough steps to flush pre-change labels from the whole bank.
    data = C.make_replay_data(seed, 12, p, 16)
    prior = P.make_replay_data(seed, 12, p, 16)
    f, m, r, audit = (data[k] for k in ('fifo', 'mixed', 'refreshed', 'memory_audit'))
    assert same(f, prior['fifo']) and same(m, prior['mixed'])
    for key in ('base_x', 'base_y', 'warm_ix', 'current_x', 'current_y', 'query_x',
                'query_old_y', 'query_new_y', 'masks', 'world'):
        assert same(f[key], r[key]), key
    rng, rr = np.random.default_rng(seed + 64000000), np.random.default_rng(seed + 66000000)
    initial = [-(int(i)+1) for i in rng.choice(1024, 512, replace=False)]
    bank, recent = initial[:256], deque(initial[256:], maxlen=256)
    records = {-(i+1): (f['base_x'][i], f['base_y'][i]) for i in range(1024)}
    for t in range(4*p):
        draw = rng.integers(512, size=16)
        expected = [recent[int(i)%256] for i in draw[:8]] + [bank[int(i)%256] for i in draw[8:]]
        assert expected == r['replay_ids'][t].tolist()
        assert expected[:8] == m['replay_ids'][t, :8].tolist()
        assert all(i < 16*t for i in expected), 'Future/current item replayed'
        assert torch.equal(r['train_x'][t, :16], f['current_x'][t])
        assert torch.equal(r['train_y'][t, :16], f['current_y'][t])
        for slot, ident in enumerate(expected, 16):
            x, y = records[ident]
            assert torch.equal(r['train_x'][t, slot], x)
            assert torch.equal(r['train_y'][t, slot], y), 'Historical label altered'
        if t >= 256:
            assert all(i >= 0 and 1 <= t-i//16 <= 256 for i in bank)
        ages = [t-(-1 if i<0 else i//16) for i in expected[8:]]
        assert ages == audit['refreshed_sampled_bank_age'][t].tolist()
        for i in range(16):
            ident = 16*t+i
            records[ident] = (f['current_x'][t, i], f['current_y'][t, i])
            recent.append(ident)
        selected = 16*t + int(rr.integers(16))
        before = list(bank)
        bank[t%256] = selected
        assert selected == int(audit['refreshed_replacement_ids'][t])
        assert sum(a != b for a,b in zip(before, bank)) == 1
        assert len(recent)+len(bank) == 512
    assert torch.equal(r['train_x'][0], m['train_x'][0])
    assert not bool(r['stale_replayed_facts'][2*p+256:].any())
    assert not bool(audit['refreshed_stale_recent'][2*p+16:].any())
    assert int(audit['mixed_stale_anchor'][2*p+256:].sum()) > 0
    old_grad = torch.autograd.grad
    def forbidden(*args, **kwargs):
        raise AssertionError('Exact hidden gradient entered local learner')
    try:
        torch.autograd.grad = forbidden
        for policy in ('frozen', 'adaptive'):
            end, _, _ = C.policy_step(C.bounded(C.init(seed)), r['train_x'][2*p+1],
                                      r['train_y'][2*p+1], policy, shadow={})
            assert all(bool(torch.isfinite(v).all()) for v in end.values())
    finally:
        torch.autograd.grad = old_grad
    return {'seed': seed, 'steps_checked': 4*p, 'original_controls_byte_exact': True,
            'refreshed_matches_independent_reference': True, 'causal_replay': True,
            'observed_labels_unchanged': True, 'same_current_and_queries': True,
            'one_circular_replacement_after_sampling': True, 'bank_age_after_turnover_at_most_256': True,
            'late_stale_targets_zero': True, 'equal_512_storage_and_32_batch': True}


def main():
    begin = time.process_time()
    V.main()
    result = json.loads((ROOT/'VERIFICATION.json').read_text())
    result['refresh_intervention'] = [verify_refresh(s) for s in (98997, 13817, 22817)]
    result['cpu_seconds'] = time.process_time()-begin
    result['status'] = 'PASS'
    (ROOT/'VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'refresh_verification': 'PASS', 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    main()
