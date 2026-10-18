"""Synthetic-only fidelity and causal routing tests before corpus training."""
import copy, json, time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import build, logits, original
H = Path(__file__).resolve().parent
c = json.loads((H/'CONFIG.json').read_text())
torch.set_num_threads(c['threads']); torch.set_num_interop_threads(c['threads'])
torch.use_deterministic_algorithms(True)
torch.manual_seed(c['canary_seed'])
x = torch.randint(c['vocab'], (c['batch'], c['context'])); y = torch.roll(x, -1, 1)
report = {'tests': [], 'profiles': {}}
for shape in [(1, 7), (2, 17), (c['batch'], c['context'])]:
    model = build(c, 'fixed', c['canary_seed'])
    with torch.no_grad():
        for layer in model.base.modules():
            if isinstance(layer, original.CurveLinear):
                layer.p.normal_(0, 0.15)
    reference = copy.deepcopy(model.base)
    xx, yy = x[:shape[0], :shape[1]], y[:shape[0], :shape[1]]
    a, b = model(xx), reference(xx)
    torch.testing.assert_close(a, b, atol=c['fidelity_atol'], rtol=c['fidelity_rtol'])
    F.cross_entropy(a.reshape(-1,c['vocab']),yy.reshape(-1)).backward()
    F.cross_entropy(b.reshape(-1,c['vocab']),yy.reshape(-1)).backward()
    for p,q in zip(model.base.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=c['fidelity_atol'],rtol=c['fidelity_rtol'])
    for m in (model.base, reference):
        torch.optim.AdamW(m.parameters(), lr=c['lr'], betas=tuple(c['betas']), eps=c['eps'], weight_decay=c['weight_decay']).step()
    for p,q in zip(model.base.parameters(),reference.parameters()):
        torch.testing.assert_close(p,q,atol=c['fidelity_atol'],rtol=c['fidelity_rtol'])
    report['tests'].append({'shape':shape,'forward_gradient_optimizer_parity':True,'max_logit_error':float((a-b).detach().abs().max())})
m = build(c, 'token', c['canary_seed'])
with torch.no_grad():
    m.router[-1].weight.normal_(0, .2)
    for layer in m.base.modules():
        if isinstance(layer, original.CurveLinear): layer.p.normal_(0, .15)
a,t = m(x, positions=True)
x2=x.clone(); x2[:,c['context']//2:]=(x2[:,c['context']//2:]+1)%c['vocab']
b,t2=m(x2,positions=True)
assert torch.equal(a[:,:c['context']//2],b[:,:c['context']//2])
assert torch.equal(t[:,:c['context']//2],t2[:,:c['context']//2])
assert (t.diff(dim=-1)>0).all() and t.min()==0 and t.max()==1
assert t[:,:,1].std()>0
loss=F.cross_entropy(a.reshape(-1,c['vocab']),y.reshape(-1)); loss.backward()
assert m.router[-1].weight.grad.abs().sum()>0
# Central difference verifies the sampling coordinate gradient with a nonconstant curve.
m=m.double(); m.zero_grad()
F.cross_entropy(m(x).reshape(-1,c['vocab']),y.reshape(-1)).backward()
param=m.router[-1].bias; analytical=float(param.grad[0]); epsilon=1e-5
with torch.no_grad():
    param[0]+=epsilon; plus=F.cross_entropy(m(x).reshape(-1,c['vocab']),y.reshape(-1)).item()
    param[0]-=2*epsilon; minus=F.cross_entropy(m(x).reshape(-1,c['vocab']),y.reshape(-1)).item()
    param[0]+=epsilon
numerical=(plus-minus)/(2*epsilon)
assert abs(analytical-numerical)<1e-8
report['tests'].append({'causal_logits_and_routing':True,'ordered_positions':True,'nonzero_router_gradient':True,'gradient_analytical':analytical,'gradient_finite_difference':numerical})
for arm in c['arms']:
    m=build(c,arm,c['canary_seed']); opt=torch.optim.AdamW(m.parameters(),lr=c['lr'])
    start=time.process_time()
    for _ in range(5):
        opt.zero_grad(); a=logits(m,arm,x,y); F.cross_entropy(a.reshape(-1,c['vocab']),y.reshape(-1)).backward(); opt.step()
    report['profiles'][arm]={'parameters':sum(p.numel() for p in m.parameters()),'cpu_seconds_per_update':(time.process_time()-start)/5}
report['projected_training_cpu_seconds']=sum(v['cpu_seconds_per_update'] for v in report['profiles'].values())*c['updates']*len(c['seeds'])
with (H/'CANARY.json').open('x') as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
