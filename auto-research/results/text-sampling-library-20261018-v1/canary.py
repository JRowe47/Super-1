"""Test causality at equal shapes and preserve the earlier cross-shape exact failure."""
import time
import run
run.setup();ref=run.references()[0];m=run.model(ref);x,y=run.windows(run.data('tune'),run.torch.arange(run.C['batch']))
opt=run.torch.optim.Adam(m.router.parameters(),lr=run.C['arms']['continued']);before=run.policy.fingerprint(m.base);router_before=run.policy.fingerprint(m.router)
with run.torch.no_grad():
    a,t=m(x,positions=True);prefix=m(x[:,:32]);difference=float((prefix-a[:,:32]).abs().max())
    # Historical failed assertion: torch.equal(prefix, a[:,:32]). No tolerance relaxation.
    changed=x.clone();changed[:,32:]=(changed[:,32:]+1)%run.C['vocab'];b=m(changed)
    assert run.torch.equal(a[:,:32],b[:,:32])
    assert (t[:,:,1:]>=t[:,:,:-1]).all() and (t>=0).all() and (t<=1).all()
start=time.process_time()
for _ in range(10):
    opt.zero_grad();pred=m(x);loss=run.F.cross_entropy(pred.flatten(0,1),y.flatten());loss.backward();opt.step()
elapsed=(time.process_time()-start)/10
assert before==run.policy.fingerprint(m.base) and router_before!=run.policy.fingerprint(m.router)
assert all(p.grad is None for p in m.base.parameters())
run.save(run.H/'CANARY.json',{'status':'PASS_EQUAL_SHAPE_CAUSALITY','prior_cross_shape_exact_assertion':'FAILED_PRESERVED','cross_shape_max_abs_difference':difference,'equal_shape_suffix_intervention_exact':True,'ordered_positions':True,'frozen_base':True,'sampler_changed':True,'cpu_seconds_per_step':elapsed,'projected_training_cpu_seconds':elapsed*len(run.C['seeds'])*len(run.C['arms'])*run.C['rounds']*run.C['updates_per_round'],'scope':'Cross-length bitwise invariance is not established. Equal-shape future-token intervention verifies causality without tolerance relaxation. Existing source has archived FP64 finite-difference verification.'})
print((run.H/'CANARY.json').read_text())
