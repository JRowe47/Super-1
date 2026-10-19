"""Six-hop local retrieval: explicit causal depth demand and exact operator audits."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,math,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';DEPTHS=[3,4,6,8,12,16,24,48]
def dump(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def basis(n,k=6):
    s=(torch.arange(n,dtype=torch.float64)+.5)/n;j=torch.arange(k,dtype=torch.float64)
    b=torch.cos(math.pi*s[:,None]*j[None,:])*math.sqrt(2/6);b[:,0]=1/math.sqrt(6);return b
B6=basis(6)
class Transport(torch.nn.Module):
    def __init__(self,native,kind='native',modes=6):
        super().__init__();self.kind=kind;self.modes=modes
        self.theta=torch.nn.Parameter(native.clone() if kind=='native' else (B6.T@native)[:modes].clone())
    def field(self,n):
        if self.kind=='native' and n==6:return self.theta
        c=B6.T@self.theta if self.kind=='native' else self.theta
        return basis(n,c.shape[0])@c
    def forward(self,x,n=6,intermediate=False):
        p=self.field(n);h=x;states=[]
        for i in range(n):
            h=h+(p[i,0]*h+p[i,1]*torch.roll(h,-1,dims=-1))/n
            if intermediate:states.append(h)
        return torch.stack(states,dim=0) if intermediate else h
@torch.no_grad()
def evaluate(m,x):
    eye=torch.eye(12,dtype=torch.float64);target=torch.roll(x,-6,dims=-1);targetop=torch.roll(eye,-6,dims=-1);ref=m(eye,6);rows=[]
    for n in DEPTHS:
        op=m(eye,n);prediction=x@op;direct=m(x[:8],n);assert float((direct-x[:8]@op).abs().max())<1e-8
        mse=float((op-targetop).square().sum()/12);emp=float((prediction-target).square().mean());energy=float(op.square().sum()/12);cov=float((op*targetop).sum()/12)
        rho=max(-1.,min(1.,cov/math.sqrt(energy))) if energy>0 else 0.
        theoretical_sign=1-math.acos(rho)/math.pi
        if n<6:
            assert abs(cov)<1e-12 and mse>=1-1e-12
        rows.append(dict(depth=n,expected_mse=mse,empirical_mse=emp,sign_accuracy=float(((prediction>0)==(target>0)).double().mean()),gaussian_sign_accuracy=theoretical_sign,
            expected_output_difference_from_depth6=float((op-ref).square().sum()/12),target_coefficient=cov,operator_energy=energy,operator=op.tolist(),field=m.field(n).tolist()))
    return rows
def save(m,seed,arm,step):
    np.savez_compressed(ROOT/'models'/f'transport_seed{seed}_{arm}_{step}.npz',theta=m.theta.detach().numpy(),kind=np.array(m.kind),modes=np.array(m.modes))
def main():
    start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    records=[];conversions=[];timings=[];checks=[]
    x_test=torch.from_numpy(np.random.default_rng(2026101719).normal(size=(4096,12)))
    def record(m,seed,arm,step,stages):
        rows=evaluate(m,x_test);records.append(dict(seed=seed,arm=arm,step=step,active_parameters=m.theta.numel(),cumulative_stages=stages,metrics=rows));dump('transport_trajectories.json',records)
    for seed in range(5):
        rng=np.random.default_rng(2026101720+seed);initial=torch.from_numpy(rng.normal(0,.15,(6,2)));initial[:,1]+=1
        err=float((Transport(initial)(x_test[:8])-Transport(initial,'curve',6)(x_test[:8])).abs().max());assert err<1e-12
        checks.append(dict(seed=seed,initial_full_curve_max_error=err));teacher=None
        for arm,kind,k in [('native6','native',6),('curve6','curve',6),('curve2','curve',2)]:
            t=time.time();m=Transport(initial,kind,k);opt=torch.optim.Adam(m.parameters(),lr=.08,foreach=False);data_rng=np.random.default_rng(2026101730+seed);record(m,seed,arm,0,0)
            for step in range(1,601):
                x=torch.from_numpy(data_rng.normal(size=(64,12)));targets=torch.stack([torch.roll(x,-i,dims=-1) for i in range(1,7)])
                opt.zero_grad(set_to_none=True);states=m(x,6,True);loss=(states-targets).square().mean();loss.backward();opt.step()
                if step in [200,600]:record(m,seed,arm,step,step*6)
            save(m,seed,arm,600)
            if arm=='native6':teacher=m.field(6).detach().clone()
            timings.append(dict(seed=seed,arm=arm,seconds=time.time()-t,updates=600,stages=3600));score=records[-1]['metrics'][2];print('SCRATCH',seed,arm,'MSE',score['expected_mse'],flush=True)
        teacherop=Transport(teacher)(torch.eye(12,dtype=torch.float64)).detach()
        for arm,kind,k,multi in [('native_fixed','native',6,False),('curve6_fixed','curve',6,False),('curve2_fixed','curve',2,False),('curve6_multidepth','curve',6,True),('curve2_multidepth','curve',2,True)]:
            t=time.time();m=Transport(teacher,kind,k);op=m(torch.eye(12,dtype=torch.float64)).detach()
            conversions.append(dict(seed=seed,arm=arm,initial_teacher_operator_mse=float((op-teacherop).square().sum()/12),discarded_field_energy=float((B6.T@teacher)[k:].square().sum()/teacher.square().sum()) if kind=='curve' else 0.))
            opt=torch.optim.Adam(m.parameters(),lr=.02,foreach=False);data_rng=np.random.default_rng(2026101740+seed);stages=0;record(m,seed,arm,0,stages)
            for step in range(1,401):
                n=[6,12,24][(step-1)%3] if multi else 6;stages+=n;x=torch.from_numpy(data_rng.normal(size=(64,12)));target=torch.roll(x,-6,dims=-1)
                opt.zero_grad(set_to_none=True);loss=(m(x,n)-target).square().mean();loss.backward();opt.step()
                if step in [100,400]:record(m,seed,arm,step,stages)
            save(m,seed,arm,400);timings.append(dict(seed=seed,arm=arm,seconds=time.time()-t,updates=400,stages=stages));score=records[-1]['metrics'][2];print('CONTINUE',seed,arm,'MSE6',score['expected_mse'],flush=True)
    dump('transport_trajectories.json',records)
    dump('transport_conversions.json',conversions)
    dump('transport_environment.json',dict(seconds=time.time()-start,runs=timings,training_updates=sum(r['updates'] for r in timings),threads=1,torch=torch.__version__))
    # Analytic oracle: a completely constant (zero-curvature) depth field.
    oracle=Transport(torch.tensor([[-6.,6.]]*6,dtype=torch.float64));oracle_rows=evaluate(oracle,x_test)
    eye=torch.eye(12,dtype=torch.float64);shift=torch.roll(eye,-1,dims=-1);analytic=[]
    for row in oracle_rows:
        n=row['depth'];direct=torch.linalg.matrix_power(eye+6*(shift-eye)/n,n);error=float((direct-torch.tensor(row['operator'],dtype=torch.float64)).abs().max());assert error<1e-11
        analytic.append(dict(depth=n,max_operator_formula_error=error))
    assert oracle_rows[2]['expected_mse']==0.
    dump('transport_oracle.json',oracle_rows);dump('transport_conversions.json',conversions)
    dump('transport_checks.json',dict(initial=checks,oracle=analytic,basis_max_error=float((B6.T@B6-torch.eye(6)).abs().max()),short_depth_lower_bound='For N<6, target coefficient is exactly0; iid Gaussian expected MSE >=1 and sign accuracy=0.5. Assertions passed for every evaluation.'))
    dump('transport_environment.json',dict(seconds=time.time()-start,runs=timings,training_updates=sum(r['updates'] for r in timings),threads=1,torch=torch.__version__))
    print('TRANSPORT COMPLETE',round(time.time()-start,2),flush=True)
if __name__=='__main__':main()
