"""Native layer identities, exact depth curves, restricted conversion and depth sampling."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import itertools,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';MODELS=ROOT/'models'
P=json.loads((ROOT/'protocol.json').read_text())['toy']
X=torch.tensor(list(itertools.product([-1.,1.],repeat=4)),dtype=torch.float64)
a,b,c,d=X.T
Y=torch.stack([a,a*b,torch.where(c<0,a,b),a*b*c*d,torch.sign(a+b+c),b*c*d],1)
TARGET=(Y+1)/2
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def basis(n,k=6):
    s=(torch.arange(n,dtype=torch.float64)+.5)/n;j=torch.arange(k,dtype=torch.float64)
    out=torch.cos(torch.pi*s[:,None]*j[None,:])*math.sqrt(2/6);out[:,0]=1/math.sqrt(6);return out
B6=basis(6);assert float((B6.T@B6-torch.eye(6)).abs().max())<1e-14
def initial(seed):
    gen=torch.Generator().manual_seed(2026101710+seed)
    return dict(A=torch.randn(16,4,generator=gen,dtype=torch.float64)/2,ai=torch.randn(16,generator=gen,dtype=torch.float64)*.1,
        W=torch.randn(6,16,16,generator=gen,dtype=torch.float64)/4,b=torch.zeros(6,16,dtype=torch.float64),
        O=torch.randn(6,16,generator=gen,dtype=torch.float64)/4,oi=torch.zeros(6,dtype=torch.float64))
class Tiny(torch.nn.Module):
    def __init__(self,state,kind='native',k=6,bridge=False):
        super().__init__();self.kind=kind;self.k=k
        for name in ['A','ai','O','oi']:setattr(self,name,torch.nn.Parameter(state[name].clone()))
        cw=torch.einsum('lj,lxy->jxy',B6,state['W']);cb=B6.T@state['b']
        self.W=torch.nn.Parameter(state['W'].clone() if kind=='native' else cw[:k].clone())
        self.b=torch.nn.Parameter(state['b'].clone() if kind=='native' else cb[:k].clone())
        rw=torch.zeros_like(cw);rb=torch.zeros_like(cb)
        if bridge:rw[k:]=cw[k:];rb[k:]=cb[k:]
        self.register_buffer('rw',rw);self.register_buffer('rb',rb)
    def weights(self,n,alpha=0.):
        if self.kind=='native' and n==6:return self.W,self.b
        if self.kind=='native':cw=torch.einsum('lj,lxy->jxy',B6,self.W);cb=B6.T@self.b
        else:cw=self.W;cb=self.b
        v=basis(n,cw.shape[0]);w=torch.einsum('ij,jxy->ixy',v,cw);b=v@cb
        if alpha:
            full=basis(n);w=w+alpha*torch.einsum('ij,jxy->ixy',full,self.rw);b=b+alpha*(full@self.rb)
        return w,b
    def forward(self,x=X,n=6,alpha=0.):
        w,b=self.weights(n,alpha);h=torch.tanh(x@self.A.T+self.ai)
        for i in range(n):h=h+torch.tanh(h@w[i].T+b[i])/n
        return h@self.O.T+self.oi
    @torch.no_grad()
    def snapshot(self,alpha=0.):
        w,b=self.weights(6,alpha)
        return {**{name:getattr(self,name).detach().clone() for name in ['A','ai','O','oi']},'W':w.detach().clone(),'b':b.detach().clone()}
@torch.no_grad()
def evaluate(model,alpha=0.):
    ref=model(n=6,alpha=alpha);rows=[]
    for n in P['test_depths']:
        logits=model(n=n,alpha=alpha);ok=(logits>0)==(TARGET>0)
        rows.append(dict(depth=n,bce=float(F.binary_cross_entropy_with_logits(logits,TARGET)),accuracy=float(ok.double().mean()),
            per_target_accuracy=ok.double().mean(0).tolist(),joint_accuracy=float(ok.all(1).double().mean()),
            logit_rms_from_depth6=float((logits-ref).square().mean().sqrt()),logits=logits.tolist()))
    return rows
def excluded_gradient(state):
    m=Tiny(state);loss=F.binary_cross_entropy_with_logits(m(),TARGET);loss.backward()
    gw=torch.einsum('lj,lxy->jxy',B6,m.W.grad);gb=B6.T@m.b.grad
    total=float(gw.square().sum()+gb.square().sum());kept=float(gw[:2].square().sum()+gb[:2].square().sum())
    return dict(bce=float(loss.detach()),residual_block_gradient_norm=math.sqrt(total),excluded_gradient_energy_fraction=1-kept/total)
def save_model(model,seed,arm,step,alpha):
    file=MODELS/f'toy_seed{seed}_{arm}_{step}.npz'
    np.savez_compressed(file,**{k:v.detach().numpy() for k,v in model.state_dict().items()},alpha=np.array(alpha),kind=np.array(model.kind),modes=np.array(model.k))
def main():
    start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    records=[];conversion=[];runtime=[];initial_checks=[]
    def record(model,seed,arm,step,alpha=0.,stages=0):
        records.append(dict(seed=seed,arm=arm,step=step,alpha=alpha,active_parameters=sum(p.numel() for p in model.parameters()),
            cumulative_executed_stages=stages,metrics=evaluate(model,alpha)))
        dump('toy_trajectories.json',records)
    for seed in P['seeds']:
        init=initial(seed);native=Tiny(init);exact=Tiny(init,'curve',6)
        err=float((native()-exact()).abs().max());assert err<1e-12;initial_checks.append(dict(seed=seed,initial_exact_curve_logit_error=err));teacher=None
        for arm,k,multi in [('native6',6,False),('curve6',6,False),('curve2',2,False),('curve2_multidepth',2,True)]:
            begin=time.time();m=Tiny(init,'native' if arm=='native6' else 'curve',k);opt=torch.optim.Adam(m.parameters(),lr=.01,foreach=False)
            stages=0;record(m,seed,arm,0)
            for step in range(1,1201):
                if step==801:opt=torch.optim.Adam(m.parameters(),lr=.01,foreach=False)
                n=[3,6,12][(step-1)%3] if multi else 6;stages+=n
                opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(m(n=n),TARGET);loss.backward();opt.step()
                if step in P['evaluations']:record(m,seed,arm,step,stages=stages)
                if arm=='native6' and step==800:
                    teacher=m.snapshot();save_model(m,seed,arm,step,0.)
            save_model(m,seed,arm,1200,0.);runtime.append(dict(seed=seed,arm=arm,seconds=time.time()-begin,stages=stages))
            score=records[-1]['metrics'][2];print('TOY',seed,arm,'BCE',round(score['bce'],5),'accuracy',score['accuracy'],flush=True)
        teacher_logits=Tiny(teacher)().detach();coefs=torch.einsum('lj,lxy->jxy',B6,teacher['W'])
        conversion.append(dict(seed=seed,source_gradient=excluded_gradient(teacher),discarded_weight_energy_fraction=float(coefs[2:].square().sum()/coefs.square().sum())))
        for arm,k,bridge in [('curve6_exact',6,False),('curve2_hard',2,False),('curve2_bridge',2,True)]:
            begin=time.time();m=Tiny(teacher,'curve',k,bridge);alpha=1. if bridge else 0.;initlogits=m(alpha=alpha).detach()
            if arm!='curve2_hard':assert float((initlogits-teacher_logits).abs().max())<1e-11
            conversion[-1][arm]=dict(initial_logit_max_error=float((initlogits-teacher_logits).abs().max()),
                initial_logit_rms_error=float((initlogits-teacher_logits).square().mean().sqrt()),gradient=excluded_gradient(m.snapshot(alpha)))
            opt=torch.optim.Adam(m.parameters(),lr=.01,foreach=False);record(m,seed,arm,0,alpha);stages=0
            for step in range(1,401):
                alpha=max(0.,1-step/200) if bridge else 0.;stages+=6
                opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(m(alpha=alpha),TARGET);loss.backward();opt.step()
                if step in [25,100,200,400]:record(m,seed,arm,step,alpha,stages)
            save_model(m,seed,arm,400,0.);runtime.append(dict(seed=seed,arm=arm,seconds=time.time()-begin,stages=stages))
            score=records[-1]['metrics'][2];print('TOY',seed,arm,'BCE',round(score['bce'],5),'accuracy',score['accuracy'],flush=True)
        dump('toy_conversion_diagnostics.json',conversion)
    dump('toy_checks.json',dict(initial_checks=initial_checks,basis_orthogonality_max_error=float((B6.T@B6-torch.eye(6)).abs().max())))
    dump('toy_environment.json',dict(seconds=time.time()-start,training_steps=30000,runs=runtime,threads=1,torch=torch.__version__))
    print('TOY COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
