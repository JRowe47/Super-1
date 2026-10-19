"""An explicit local two-real-channel smooth flow for the six-hop routing task."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,math,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
DT=torch.float64
def dump(name,value):
    p=ROOT/'results'/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def rotate(x,y,angle,kind,solver='exact'):
    c,s=(torch.cos(angle),torch.sin(angle)) if solver=='exact' else (1.,angle)
    if kind==0:return c*x-s*y,s*x+c*y
    return c*x+s*torch.roll(y,-1,-1),c*y-s*torch.roll(x,1,-1)
def discrete(x,angles,stages=False):
    y=torch.zeros_like(x);saved=[]
    for k in range(6):
        x,y=rotate(x,y,angles[2*k],0);x,y=rotate(x,y,angles[2*k+1],1)
        if stages:saved.append((x,y))
    return saved if stages else (x,y)
def pulse_integral(u):return u-math.sin(2*math.pi*u)/(2*math.pi)
def continuous(x,angles,grid,solver):
    y=torch.zeros_like(x);ops=0
    for cell in range(grid):
        lo=cell/grid;hi=(cell+1)/grid
        if solver=='midpoint':
            mid=(lo+hi)/2;j=min(11,int(12*mid));u=12*mid-j
            area=angles[j]*(24/grid)*math.sin(math.pi*u)**2
            x,y=rotate(x,y,area,j%2);ops+=1
        else:
            for j in range(12):
                a=max(lo,j/12);b=min(hi,(j+1)/12)
                if b-a<1e-14:continue
                u0=max(0.,min(1.,12*a-j));u1=max(0.,min(1.,12*b-j))
                area=angles[j]*(pulse_integral(u1)-pulse_integral(u0))
                x,y=rotate(x,y,area,j%2,'euler' if solver=='euler' else 'exact');ops+=1
    return x,y,ops
def scores(out,aux,target):
    # Identity inputs: mean(sum_i error^2) equals expected per-site MSE for iid unit Gaussians.
    err=float((out-target).square().sum()/12)
    return dict(expected_mse=err,auxiliary_energy=float(aux.square().sum()/12),singular_values=torch.linalg.svdvals(out).tolist(),max_abs_error=float((out-target).abs().max()))
def main():
    start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    eye=torch.eye(12,dtype=DT);target=torch.roll(eye,-6,-1)
    generator=torch.Generator().manual_seed(2026101721);held=torch.randn(4096,12,generator=generator,dtype=DT);heldtarget=torch.roll(held,-6,-1)
    grids=[1,6,12,17,24,31,48,96];oracleangles=torch.full((12,),math.pi/2,dtype=DT)
    oracle=[];checks=[]
    for solver in ['exact','midpoint','euler']:
        for grid in grids:
            out,aux,ops=continuous(eye,oracleangles,grid,solver);row=dict(solver=solver,grid=grid,rotation_operations=ops,**scores(out,aux,target));oracle.append(row)
            if solver=='exact':assert row['expected_mse']<1e-25 and row['auxiliary_energy']<1e-25
    # One hop on arbitrary nonzero auxiliary state is diag(S,S^-1), not just a zero-state trick.
    x=torch.randn(7,12,generator=generator,dtype=DT);y=torch.randn(7,12,generator=generator,dtype=DT)
    a,b=rotate(x,y,oracleangles[0],0);a,b=rotate(a,b,oracleangles[1],1)
    assert torch.max(torch.abs(a-torch.roll(x,-1,-1)))<1e-14
    assert torch.max(torch.abs(b-torch.roll(y,1,-1)))<1e-14
    # Show that onsite and edge generators do not commute, unlike previous scalar I/S family.
    z=torch.eye(24,dtype=DT);xx=z[:,:12];yy=z[:,12:]
    on=lambda x,y:(-y,x)
    ed=lambda x,y:(torch.roll(y,-1,-1),-torch.roll(x,1,-1))
    a,b=on(*ed(xx,yy));c,d=ed(*on(xx,yy))
    checks.append(dict(name='noncommuting_generators',commutator_frobenius=float(torch.cat([a-c,b-d],1).norm())))
    dump('transport_oracle.json',oracle)
    records=[];traces=[];times=[]
    for seed in range(5):
        gen=torch.Generator().manual_seed(1000+seed);initial=1.+.1*torch.randn(12,generator=gen,dtype=DT)
        for arm in ['full12','shared2']:
            t=time.time();p=torch.nn.Parameter(initial.clone() if arm=='full12' else initial.reshape(6,2).mean(0))
            opt=torch.optim.Adam([p],lr=.05);batchgen=torch.Generator().manual_seed(2000+seed)
            for step in range(1,401):
                inp=torch.randn(64,12,generator=batchgen,dtype=DT);angles=p if arm=='full12' else p.repeat(6)
                st=discrete(inp,angles,True)
                loss=sum(((h-torch.roll(inp,-k-1,-1)).square().mean()+a.square().mean())*.5 for k,(h,a) in enumerate(st))/6
                opt.zero_grad();loss.backward();opt.step()
                if step in [1,100,200,400]:traces.append(dict(seed=seed,arm=arm,step=step,training_loss=float(loss.detach())))
            angles=(p if arm=='full12' else p.repeat(6)).detach()
            np.savez_compressed(ROOT/'models'/f'transport_{arm}_seed{seed}.npz',parameters=p.detach().numpy(),angles=angles.numpy(),initial=initial.numpy())
            with torch.no_grad():
                reference,refaux=discrete(eye,angles)
                for grid in grids:
                    out,aux,ops=continuous(eye,angles,grid,'exact');emp,empa,_=continuous(held,angles,grid,'exact')
                    row=dict(seed=seed,arm=arm,grid=grid,rotation_operations=ops,**scores(out,aux,target),empirical_mse=float((emp-heldtarget).square().mean()),sign_accuracy=float((emp.sign()==heldtarget.sign()).double().mean()),max_drift_from_12_pulse=float((out-reference).abs().max()))
                    assert row['max_drift_from_12_pulse']<1e-12
                    records.append(row)
            times.append(dict(seed=seed,arm=arm,seconds=time.time()-t,steps=400))
            dump('transport_evaluations.json',records);dump('transport_training.json',traces);dump('transport_environment.json',dict(seconds=time.time()-start,runs=times,new_training_steps=sum(r['steps'] for r in times),threads=1,torch=torch.__version__))
            print('TOY',seed,arm,'MSE',records[-1]['expected_mse'],'drift',records[-1]['max_drift_from_12_pulse'],flush=True)
    dump('transport_checks.json',checks)
    print('TRANSPORT COMPLETE',round(time.time()-start,2),flush=True)
if __name__=='__main__':main()
