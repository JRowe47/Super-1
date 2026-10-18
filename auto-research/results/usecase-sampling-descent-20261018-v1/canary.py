import json,time,torch
from pathlib import Path
from torch.nn import functional as F
from policy import build,freeze_base,fingerprint,prior
from tasks import load_data,batch,evaluation
H=Path(__file__).resolve().parent;c=json.loads((H/'CONFIG.json').read_text())
torch.set_num_threads(c['threads']);torch.set_num_interop_threads(c['threads']);torch.use_deterministic_algorithms(True)
m=build(c,c['canary_seed'],'token').double()
with torch.no_grad():
    for layer in m.base.modules():
        if isinstance(layer,prior.original.CurveLinear):layer.p.normal_(0,.15)
    m.router[-1].weight.normal_(0,.1)
params=freeze_base(m);before=fingerprint(m.base)
x=torch.randint(c['vocab'],(2,c['context']));y=torch.randint(c['vocab'],x.shape)
a,pos=m(x,positions=True);changed=x.clone();changed[:,32:]=(changed[:,32:]+1)%c['vocab']
b,pos2=m(changed,positions=True)
assert torch.equal(a[:,:32],b[:,:32]);assert torch.equal(pos[:,:32],pos2[:,:32])
assert (pos.diff(dim=-1)>0).all()
loss=F.cross_entropy(a.flatten(0,1),y.flatten());loss.backward();param=m.router[-1].bias
analytical=float(param.grad[0]);eps=1e-5
with torch.no_grad():
    param[0]+=eps;plus=F.cross_entropy(m(x).flatten(0,1),y.flatten()).item()
    param[0]-=2*eps;minus=F.cross_entropy(m(x).flatten(0,1),y.flatten()).item();param[0]+=eps
numeric=(plus-minus)/(2*eps);assert abs(analytical-numeric)<1e-8
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
assert all(p.grad is None for p in m.base.parameters())
old=[p.detach().clone() for p in params];torch.optim.Adam(params,lr=c['sampling_lr']).step()
assert any(not torch.equal(p,q) for p,q in zip(old,params));assert fingerprint(m.base)==before
# Verify all synthetic targets match task definitions and never occur as answer input.
d=load_data(c);raw=json.loads((H/'DATA.json').read_text())
for task,parts in raw.items():
    for split,rows in parts.items():
        for row in rows:
            i=row['mask'].index(True);assert sum(row['mask'])==1
            assert bytes(row['x'][:i+1]).decode()==row['prompt'];assert row['y'][i]==ord(row['answer'])
            if task=='arithmetic':ans=sum(map(int,row['prompt'].split(' mod ')[0].split('+')))%c['modulus']
            else:
                values=dict(pair.split('=') for pair in row['prompt'].split(';')[:c['retrieval_pairs']])
                key=row['prompt'].split('query=')[1].split(';')[0];ans=int(values[key])
            assert ans==int(row['answer'])
profiles={}
for mode in ['fixed','token']:
    q=build(c,c['canary_seed'],mode);params=list(q.parameters()) if mode=='fixed' else freeze_base(q)
    opt=torch.optim.AdamW(params,lr=c['base_lr']);start=time.process_time()
    for _ in range(5):
        opt.zero_grad();pred=q(x);F.cross_entropy(pred.flatten(0,1),y.flatten()).backward();opt.step()
    profiles[mode]=(time.process_time()-start)/5
result={'status':'PASS','causal_prefix_exact':True,'ordered_coordinates':True,'backbone_immutable_after_optimizer':True,'backbone_gradients_absent':True,'sampling_gradient_nonzero':True,'finite_difference':{'analytical':analytical,'numerical':numeric,'abs_error':abs(analytical-numeric)},'synthetic_labels_and_no_answer_leakage':True,'profile_cpu_seconds_per_step':profiles,'projected_cpu_seconds':len(c['seeds'])*(len(c['tasks'])*c['base_updates_per_task']*profiles['fixed']+c['rounds']*c['updates_per_round_per_task']*len(c['tasks'])*3*profiles['token'])}
with (H/'CANARY.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
