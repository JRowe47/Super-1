"""Verify recurrence, gradients, optimizer step, cached decoding and resource cost."""
import ast,copy,hashlib,importlib.util,json,time,types
from pathlib import Path
import torch
from torch.nn import functional as F
from model import build,delta_chunk,delta_sequential,tensor_bytes
H=Path(__file__).resolve().parent;C=json.loads((H/'CONFIG.json').read_text());torch.set_num_threads(1);torch.set_num_interop_threads(1)
# Load only the unmodified reference recurrence; its sibling chunk function needs einops.
source=ast.parse((H/'vendor/naive.py').read_text());function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='naive_recurrent_gated_delta_rule')
namespace={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),str(H/'vendor/naive.py'),'exec'),namespace);upstream=types.SimpleNamespace(**namespace)
torch.manual_seed(C['canary_seed']);q,k=[F.normalize(torch.randn(2,2,37,8,dtype=torch.float64),dim=-1) for _ in range(2)];v=torch.randn_like(q);g=F.logsigmoid(torch.randn(2,2,37,dtype=torch.float64));beta=torch.randn_like(g).sigmoid();state=torch.randn(2,2,8,8,dtype=torch.float64)
a=delta_chunk(q,k,v,g,beta,state,16);b=delta_sequential(q,k,v,g,beta,state)
forward_error=max(float((x-y).abs().max()) for x,y in zip(a,b));assert forward_error<C['fp64_atol']
inputs=[x.clone().requires_grad_() for x in (q,k,v,g,beta,state)];out,s=delta_chunk(*inputs[:3],*inputs[3:5],state=inputs[5],chunk_size=16);loss=out.square().mean()+s.square().mean();grads1=torch.autograd.grad(loss,inputs)
inputs2=[x.detach().clone().requires_grad_() for x in inputs];out,s=delta_sequential(*inputs2[:3],*inputs2[3:5],state=inputs2[5]);grads2=torch.autograd.grad(out.square().mean()+s.square().mean(),inputs2)
grad_error=max(float((a-b).abs().max()) for a,b in zip(grads1,grads2));assert grad_error<C['fp64_atol']
# Upstream explicitly casts to FP32, so compare at its native precision.
ref=upstream.naive_recurrent_gated_delta_rule(q.float().transpose(1,2),k.float().transpose(1,2),v.float().transpose(1,2),beta.float().transpose(1,2),g.float().transpose(1,2),scale=1,initial_state=state.float(),output_final_state=True)
own=delta_chunk(q.float(),k.float(),v.float(),g.float(),beta.float(),state.float(),16)
assert torch.allclose(own[0].transpose(1,2),ref[0],atol=C['forward_atol'],rtol=C['forward_rtol']);assert torch.allclose(own[1],ref[1],atol=C['forward_atol'],rtol=C['forward_rtol'])
records=[];profiles={}
for arm in C['patterns']:
    m=build(C,arm,C['canary_seed']);x=torch.randint(C['vocab'],(2,67));y=torch.randint(C['vocab'],x.shape)
    m.eval()
    with torch.no_grad():
        full,fullstate=m(x);cache=None;outputs=[]
        for offset in range(0,x.shape[1],13):
            logits,cache=m(x[:,offset:offset+13],cache);outputs.append(logits)
        cached=torch.cat(outputs,1);error=float((full-cached).abs().max());assert torch.allclose(full,cached,atol=C['forward_atol'],rtol=C['forward_rtol'])
        changed=x.clone();changed[:,35:]=(changed[:,35:]+1)%C['vocab'];other,_=m(changed);assert torch.equal(full[:,:35],other[:,:35])
    # Complete optimizer parity across chunk and sequential delta execution.
    m=m.double();other=copy.deepcopy(m);optimizers=[torch.optim.AdamW(z.parameters(),lr=C['lr']) for z in (m,other)]
    for z,opt,sequential in zip((m,other),optimizers,(False,True)):
        opt.zero_grad();logits,_=z(x,sequential=sequential);F.cross_entropy(logits.flatten(0,1),y.flatten()).backward();opt.step()
    optimizer_error=max(float((p-q).abs().max()) for p,q in zip(m.parameters(),other.parameters()));assert optimizer_error<C['fp64_atol']
    m=build(C,arm,C['canary_seed']);x=torch.randint(C['vocab'],(C['batch'],C['context']));y=torch.randint(C['vocab'],x.shape);opt=torch.optim.AdamW(m.parameters(),lr=C['lr']);start=time.process_time()
    for _ in range(5):
        opt.zero_grad();pred,_=m(x);F.cross_entropy(pred.flatten(0,1),y.flatten()).backward();opt.step()
    profiles[arm]=(time.process_time()-start)/5
    records.append({'arm':arm,'parameters':sum(p.numel() for p in m.parameters()),'cached_forward_max_error':error,'optimizer_max_error':optimizer_error,'state_bytes_at_batch2_length67':tensor_bytes(cache)})
receipt={'status':'PASS','fp64_recurrence_max_error':forward_error,'fp64_gradient_max_error':grad_error,'upstream_fp32_parity':True,'records':records,'cpu_seconds_per_step':profiles,'projected_training_cpu_seconds':sum(profiles.values())*C['updates']*len(C['seeds']),'source_sha256':hashlib.sha256((H/'model.py').read_bytes()).hexdigest()}
with (H/'CANARY.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
