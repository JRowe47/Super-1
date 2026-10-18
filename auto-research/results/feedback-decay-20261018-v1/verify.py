"""Independent dense factorial reference, endpoints, and data-free decay proof."""
import feedback_verify as V
from feedback_verify import C, torch, json, time, ROOT
from collections import deque
import numpy as np


def verify_components(seed, dtype, tol):
    gen = torch.Generator().manual_seed(seed+79000000)
    errors, capmax = {}, 0.
    for case, gain in (('bounded',1.),('projection_active',1.5)):
        state = C.bounded(C.init(seed,dtype))
        for k in C.FK:state[k] *= gain
        snapshot = C.clone(state)
        x = torch.randn(13,C.D,generator=gen,dtype=dtype)
        y = torch.randint(0,2,(13,C.R),generator=gen).to(dtype)
        dense = C.decode(state)
        gradients = C.dense_reference(dense,x,y)
        frozen,_,_ = C.P.policy_step(state,x,y,'frozen',shadow={})
        normal,_,_ = C.P.policy_step(state,x,y,'adaptive',shadow={})
        for policy,(learning,decay) in C.FEEDBACK_COMPONENTS.items():
            direct = {}
            for k,w in dense.items():
                wd = C.WD_WB if k.startswith(('W','B')) else C.WD_VU
                if k in C.WK:
                    z = w-C.LR*(gradients[k]+wd*w)
                else:
                    drive = gradients[k] if learning else torch.zeros_like(w)
                    if decay:drive = drive+wd*w
                    z = w-C.LR*drive
                    if learning or decay:
                        sigma = float(torch.linalg.svdvals(z)[0])
                        limit = 2.*(1-4*torch.finfo(dtype).eps)
                        z = z*min(1.,limit/sigma) if sigma>0 else z
                direct[k] = z
            new = C.component_step(state,x,y,learning,decay)
            observed = C.decode(new)
            errors[case+'_'+policy] = max(float((observed[k]-direct[k]).abs().max()) for k in direct)
            assert errors[case+'_'+policy] <= tol,(seed,dtype,errors)
            assert all(torch.equal(new[k],frozen[k]) for k in C.WK)
            assert all(torch.equal(state[k],snapshot[k]) for k in state)
            assert all(not v.requires_grad and v.grad_fn is None for v in new.values())
            if learning or decay:
                capmax = max(capmax,max(float(torch.linalg.svdvals(observed[k])[0]) for k in C.FK))
                assert capmax<=2.
            if policy in ('frozen','adaptive'):
                ref = frozen if policy=='frozen' else normal
                assert all(torch.equal(new[k],ref[k]) for k in new)
    initial = C.bounded(C.init(seed,dtype))
    f,a,rf,ra,d1,d2 = (C.clone(initial) for _ in range(6))
    for _ in range(12):
        x = torch.randn(7,C.D,generator=gen,dtype=dtype)
        y = torch.randint(0,2,(7,C.R),generator=gen).to(dtype)
        f = C.component_step(f,x,y,False,False)
        a = C.component_step(a,x,y,True,True)
        rf,_,_ = C.P.policy_step(rf,x,y,'frozen',shadow={})
        ra,_,_ = C.P.policy_step(ra,x,y,'adaptive',shadow={})
        assert all(torch.equal(f[k],rf[k]) and torch.equal(a[k],ra[k]) for k in f)
        d1,_,_ = C.policy_step(d1,x,y,'decay',shadow={})
        d2,_,_ = C.policy_step(d2,-x,1-y,'decay',shadow={})
        assert all(torch.equal(d1[k],d2[k]) for k in C.FK),'Decay depends on data'
    errors['decay_closed_form_12steps'] = max(float((d1[k]-initial[k]*(1-C.LR*(C.WD_WB if k.startswith('B') else C.WD_VU))**12).abs().max()) for k in C.FK)
    assert errors['decay_closed_form_12steps'] <= tol,(dtype,errors)
    old = torch.autograd.grad
    def forbidden(*args,**kwargs):raise AssertionError('Autograd entered local factorial rule')
    try:
        torch.autograd.grad = forbidden
        for policy in C.FEEDBACK_COMPONENTS:C.policy_step(initial,x,y,policy,shadow={})
    finally:torch.autograd.grad = old
    return {'seed':seed,'dtype':str(dtype),'tolerance':tol,'errors':errors,
            'maximum_capped_sigma':capmax,'same_state_forward_updates_bitwise':True,
            'original_endpoint_steps_and_12step_trajectories_bitwise':True,
            'decay_feedback_independent_of_data_bitwise':True,
            'input_unmodified_output_detached':True,'all_local_paths_without_autograd':True}


def verify_fifo(seed):
    schedules=C.make_replay_data(seed,12,32,16);f=schedules['fifo'];original=C.make_data(seed,12,32,16)
    def same(a,b):
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        return a==b
    assert same(f,original)
    rng=np.random.default_rng(seed+64000000)
    initial=[-(int(i)+1) for i in rng.choice(1024,512,replace=False)]
    memory=deque(initial,maxlen=512);records={-(i+1):(f['base_x'][i],f['base_y'][i]) for i in range(1024)}
    for t in range(128):
        expected=[memory[int(i)] for i in rng.integers(512,size=16)]
        assert expected==f['replay_ids'][t].tolist() and all(i<16*t for i in expected)
        assert torch.equal(f['train_x'][t,:16],f['current_x'][t]) and torch.equal(f['train_y'][t,:16],f['current_y'][t])
        for slot,ident in enumerate(expected,16):
            x,y=records[ident]
            assert torch.equal(f['train_x'][t,slot],x) and torch.equal(f['train_y'][t,slot],y)
        for i in range(16):
            ident=16*t+i;records[ident]=(f['current_x'][t,i],f['current_y'][t,i]);memory.append(ident)
        assert len(memory)==512
    assert not bool(f['stale_replayed_facts'][96:].any())
    return {'seed':seed,'steps_checked':128,'original_generator_and_reference_exact':True,
            'causal_replay_and_historical_labels':True,'same512_storage_and32_batch':True}


def main():
    begin=time.process_time();V.main();out=json.loads((ROOT/'VERIFICATION.json').read_text())
    out['feedback_components']=[verify_components(s,d,tol) for s in (100917,15817,24817)
                                for d,tol in ((torch.float32,2e-6),(torch.float64,2e-12))]
    out['fifo_reference']=[verify_fifo(s) for s in (100917,15817,24817)]
    out.update(status='PASS',cpu_seconds=time.process_time()-begin)
    (ROOT/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'feedback_component_verification':'PASS','cpu_seconds':out['cpu_seconds']}),flush=True)


if __name__=='__main__':main()
