"""Execution counts, four-step parity, nonconstant-coordinate and depth gradients."""
import time
import torch
from torch.nn import functional as F
import run
run.setup();ref=run.references()[0];x,y=run.windows(run.data('tune'),torch.arange(run.C['batch']))
original=run.parent.model(ref);models={d:run.model(ref,d) for d in run.C['depths']};rows=[]
for depth,m in models.items():
    counts=[0]*run.C['layers'];handles=[]
    for i,block in enumerate(m.base.blocks):
        def hook(module,args,output,index=i):counts[index]+=1
        handles.append(block.ln1.register_forward_hook(hook))
    with torch.no_grad():logits,positions=m(x,positions=True)
    for handle in handles:handle.remove()
    assert counts==[int(i<depth) for i in range(run.C['layers'])]
    assert (positions[:,:,1:]>=positions[:,:,:-1]).all() and (positions>=0).all() and (positions<=1).all()
    changed=x.clone();changed[:,32:]=(changed[:,32:]+1)%run.C['vocab']
    with torch.no_grad():assert torch.equal(logits[:,:32],m(changed)[:,:32])
    rows.append({'depth':depth,'executed_blocks':counts})
# Full-depth execution and a complete optimizer step retain the parent semantics.
opts=[torch.optim.Adam(m.router.parameters(),lr=run.C['router_lr']) for m in (original,models[4])]
for m,opt in zip((original,models[4]),opts):
    opt.zero_grad();loss=F.cross_entropy(m(x).flatten(0,1),y.flatten());loss.backward();opt.step()
assert run.parent.policy.fingerprint(original.router)==run.parent.policy.fingerprint(models[4].router)
with torch.no_grad():assert torch.equal(original(x),models[4](x))
# Check the new three-step path using double precision, without relaxing tolerance.
m=models[3].double();p=m.router[-1].bias;loss=F.cross_entropy(m(x[:2]).flatten(0,1),y[:2].flatten());loss.backward();analytic=float(p.grad[0]);eps=1e-5
with torch.no_grad():
    initial=float(p[0]);p[0]=initial+eps;plus=float(F.cross_entropy(m(x[:2]).flatten(0,1),y[:2].flatten()));p[0]=initial-eps;minus=float(F.cross_entropy(m(x[:2]).flatten(0,1),y[:2].flatten()));p[0]=initial
numeric=(plus-minus)/(2*eps);assert abs(analytic-numeric)<1e-7 and abs(analytic)>1e-8
selector=torch.tensor([.2,-.3,.1],dtype=torch.float64,requires_grad=True);losses=torch.tensor([5.,3.,2.],dtype=torch.float64);expected=(selector.softmax(0)*losses).sum();expected.backward();analytical=selector.softmax(0)*(losses-expected.detach());assert torch.allclose(selector.grad,analytical,atol=1e-14,rtol=0)
models={d:run.model(ref,d) for d in run.C['depths']};start=time.process_time()
for _ in range(10):
    for depth,m in models.items():
        m.zero_grad();loss=F.cross_entropy(m(x).flatten(0,1),y.flatten())
        if depth>2:loss.backward()
elapsed=(time.process_time()-start)/10
run.save(run.H/'CANARY.json',{'status':'PASS','counts':rows,'four_step_optimizer_parity':True,'same_shape_causality_exact':True,'three_step_gradient':{'analytic':analytic,'numeric':numeric,'absolute_error':abs(analytic-numeric)},'categorical_expected_loss_gradient':True,'cpu_seconds_per_all_depth_batch':elapsed,'projected_training_cpu_seconds':elapsed*len(run.C['seeds'])*run.C['rounds']*run.C['updates_per_round']})
print((run.H/'CANARY.json').read_text())
