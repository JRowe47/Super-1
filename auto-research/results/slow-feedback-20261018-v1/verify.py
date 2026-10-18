"""Preserved numerical checks plus independent slow-step and FIFO references."""
import feedback_verify as V
from feedback_verify import C, torch, json, time, ROOT
from collections import deque
import numpy as np


def verify_slow(seed, dtype, tol):
    gen = torch.Generator().manual_seed(seed + 78000000)
    errors, capmax = {}, 0.
    for case, gain in (('bounded', 1.), ('projection_active', 1.5)):
        state = C.bounded(C.init(seed, dtype))
        for k in C.FK:
            state[k] *= gain
        x = torch.randn(13, C.D, generator=gen, dtype=dtype)
        y = torch.randint(0, 2, (13, C.R), generator=gen).to(dtype)
        snapshot = C.clone(state)
        dense = C.decode(state)
        # Independent per-example synapse-coordinate rule, then direct decay/update.
        g = C.dense_reference(dense, x, y)
        direct = {}
        for k, w in dense.items():
            wd = C.WD_WB if k.startswith(('W','B')) else C.WD_VU
            alpha = 1. if k in C.WK else 0.1
            z = w - (C.LR * alpha) * (g[k] + wd*w)
            if k in C.FK:
                sigma = float(torch.linalg.svdvals(z)[0])
                limit = 2. * (1 - 4*torch.finfo(dtype).eps)
                z = z * min(1., limit/sigma) if sigma > 0 else z
            direct[k] = z
        slow, _, _ = C.policy_step(state, x, y, 'slow', shadow={})
        observed = C.decode(slow)
        errors[case+'_dense_step'] = max(float((observed[k]-direct[k]).abs().max()) for k in direct)
        assert errors[case+'_dense_step'] <= tol, (seed, dtype, errors)
        assert all(torch.equal(state[k], snapshot[k]) for k in state), 'Input mutation'
        assert all(not v.requires_grad and v.grad_fn is None for v in slow.values())
        capmax = max(capmax, max(float(torch.linalg.svdvals(observed[k])[0]) for k in C.FK))
        assert capmax <= 2.
        frozen, _, _ = C.P.policy_step(state, x, y, 'frozen', shadow={})
        adaptive, _, _ = C.P.policy_step(state, x, y, 'adaptive', shadow={})
        assert all(torch.equal(slow[k], frozen[k]) and torch.equal(slow[k], adaptive[k]) for k in C.WK)
        zero, one = C.feedback_step(state,x,y,0), C.feedback_step(state,x,y,1)
        assert all(torch.equal(zero[k], frozen[k]) for k in state)
        assert all(torch.equal(one[k], adaptive[k]) for k in state)
    # Multi-step endpoint equivalence against byte-preserved policies.
    s0 = C.bounded(C.init(seed, dtype)); s1 = C.clone(s0)
    f, a = C.clone(s0), C.clone(s0)
    for _ in range(12):
        x = torch.randn(7,C.D,generator=gen,dtype=dtype)
        y = torch.randint(0,2,(7,C.R),generator=gen).to(dtype)
        s0, s1 = C.feedback_step(s0,x,y,0), C.feedback_step(s1,x,y,1)
        f, _, _ = C.P.policy_step(f,x,y,'frozen',shadow={})
        a, _, _ = C.P.policy_step(a,x,y,'adaptive',shadow={})
        assert all(torch.equal(s0[k],f[k]) and torch.equal(s1[k],a[k]) for k in s0)
    old = torch.autograd.grad
    def forbidden(*args, **kwargs):
        raise AssertionError('Exact hidden gradient entered slow local learning')
    try:
        torch.autograd.grad = forbidden
        C.policy_step(s1, x, y, 'slow', shadow={})
    finally:
        torch.autograd.grad = old
    return {'seed': seed, 'dtype': str(dtype), 'tolerance': tol, 'errors': errors,
            'maximum_capped_sigma': capmax, 'forward_update_exact_at_same_state': True,
            'zero_and_one_endpoints_bitwise': True, 'endpoint_trajectories_12steps_bitwise': True,
            'local_without_autograd': True, 'input_unmodified_and_output_detached': True}


def verify_fifo(seed):
    schedules = C.make_replay_data(seed, 12, 32, 16)
    f = schedules['fifo']
    original = C.make_data(seed, 12, 32, 16)
    def same(a,b):
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        return a==b
    assert same(f,original)
    rng=np.random.default_rng(seed+64000000)
    initial=[-(int(i)+1) for i in rng.choice(1024,512,replace=False)]
    memory=deque(initial,maxlen=512)
    records={-(i+1):(f['base_x'][i],f['base_y'][i]) for i in range(1024)}
    for t in range(128):
        expected=[memory[int(i)] for i in rng.integers(512,size=16)]
        assert expected==f['replay_ids'][t].tolist()
        assert all(i<16*t for i in expected)
        assert torch.equal(f['train_x'][t,:16],f['current_x'][t])
        assert torch.equal(f['train_y'][t,:16],f['current_y'][t])
        for slot,ident in enumerate(expected,16):
            x,y=records[ident]
            assert torch.equal(f['train_x'][t,slot],x) and torch.equal(f['train_y'][t,slot],y)
        for i in range(16):
            ident=16*t+i
            records[ident]=(f['current_x'][t,i],f['current_y'][t,i]);memory.append(ident)
        assert len(memory)==512
    assert not bool(f['stale_replayed_facts'][96:].any())
    return {'seed':seed,'steps_checked':128,'original_generator_exact':True,
            'independent_queue_reference_exact':True,'historical_labels_unchanged':True,
            'causal_replay':True,'fixed_512_storage_and_32_batch':True}


def main():
    begin=time.process_time()
    V.main()
    out=json.loads((ROOT/'VERIFICATION.json').read_text())
    out['slow_feedback']=[verify_slow(s,d,tol) for s in (99907,14817,23817)
                          for d,tol in ((torch.float32,2e-6),(torch.float64,2e-12))]
    out['fifo_reference']=[verify_fifo(s) for s in (99907,14817,23817)]
    out.update(status='PASS',cpu_seconds=time.process_time()-begin)
    (ROOT/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'slow_feedback_verification':'PASS','cpu_seconds':out['cpu_seconds']}),flush=True)


if __name__=='__main__':
    main()
