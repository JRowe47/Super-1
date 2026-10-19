"""Recover post-training metadata after a float32-only oracle audit error. No training."""
import json,math
from pathlib import Path
import numpy as np
import torch
from transport_toy import ROOT,Transport,B6,evaluate,dump
torch.set_num_threads(1)
records=json.loads((ROOT/'results/transport_trajectories.json').read_text())
x=torch.from_numpy(np.random.default_rng(2026101719).normal(size=(4096,12)))
initial_checks=[];conversions=[];runs=[];audits=[];paths=[]
for seed in range(5):
    rng=np.random.default_rng(2026101720+seed);initial=torch.from_numpy(rng.normal(0,.15,(6,2)));initial[:,1]+=1
    err=float((Transport(initial)(x[:8])-Transport(initial,'curve',6)(x[:8])).abs().max());assert err<1e-12
    initial_checks.append(dict(seed=seed,initial_full_curve_max_error=err))
    with np.load(ROOT/'models'/f'transport_seed{seed}_native6_600.npz') as z:teacher=torch.from_numpy(z['theta'].copy())
    teacherop=Transport(teacher)(torch.eye(12,dtype=torch.float64)).detach()
    for arm,kind,k,multi in [('native_fixed','native',6,False),('curve6_fixed','curve',6,False),('curve2_fixed','curve',2,False),('curve6_multidepth','curve',6,True),('curve2_multidepth','curve',2,True)]:
        m=Transport(teacher,kind,k);op=m(torch.eye(12,dtype=torch.float64)).detach()
        conversions.append(dict(seed=seed,arm=arm,initial_teacher_operator_mse=float((op-teacherop).square().sum()/12),discarded_field_energy=float((B6.T@teacher)[k:].square().sum()/teacher.square().sum()) if kind=='curve' else 0.))
    for arm in ['native6','curve6','curve2','native_fixed','curve6_fixed','curve2_fixed','curve6_multidepth','curve2_multidepth']:
        step=600 if arm in ['native6','curve6','curve2'] else 400;path=ROOT/'models'/f'transport_seed{seed}_{arm}_{step}.npz';paths.append(path)
        with np.load(path) as z:
            m=Transport(teacher,str(z['kind']),int(z['modes']))
            with torch.no_grad():m.theta.copy_(torch.from_numpy(z['theta']))
        new=evaluate(m,x)
        matches=[r for r in records if r['seed']==seed and r['arm']==arm and r['step']==step]
        if matches:rr=matches[0]
        else:
            stages=sum([6,12,24][i%3] for i in range(step)) if 'multidepth' in arm else step*6
            rr=dict(seed=seed,arm=arm,step=step,active_parameters=m.theta.numel(),cumulative_stages=stages,metrics=new,recovered_from_saved_model=True)
            records.append(rr)
        error=max(abs(a['expected_mse']-b['expected_mse']) for a,b in zip(new,rr['metrics']));assert error<1e-11
        audits.append(dict(seed=seed,arm=arm,max_expected_mse_difference=error));runs.append(dict(seed=seed,arm=arm,updates=step,stages=rr['cumulative_stages'],seconds=None))
oracle=Transport(torch.tensor([[-6.,6.]]*6,dtype=torch.float64));rows=evaluate(oracle,x);eye=torch.eye(12,dtype=torch.float64);shift=torch.roll(eye,-1,dims=-1);checks=[]
for r in rows:
    n=r['depth'];a=torch.linalg.matrix_power(eye+6*(shift-eye)/n,n);error=float((a-torch.tensor(r['operator'],dtype=torch.float64)).abs().max());assert error<1e-11
    checks.append(dict(depth=n,max_operator_formula_error=error))
assert rows[2]['expected_mse']==0.
dump('transport_trajectories.json',records);dump('transport_oracle.json',rows);dump('transport_conversions.json',conversions)
dump('transport_checks.json',dict(initial=initial_checks,oracle=checks,basis_max_error=float((B6.T@B6-torch.eye(6)).abs().max()),short_depth_lower_bound='For N<6, target coefficient is exactly0; iid Gaussian expected MSE >=1 and sign accuracy=0.5. Assertions passed for every evaluation.',saved_model_reload_audit=audits))
dump('transport_environment.json',dict(seconds=None,checkpoint_span_seconds=max(p.stat().st_mtime for p in paths)-min(p.stat().st_mtime for p in paths),runtime_note='All19000 updates and40 final models completed before an oracle audit reconstructed JSON values as float32 and failed an overly strict float64 comparison. Checks and conversions recovered from saved models without rerunning training. Original exact per-run timers were not persisted; checkpoint span is a lower bound, not wall time.',runs=runs,training_updates=sum(r['updates'] for r in runs),threads=1,torch=torch.__version__))
print(json.dumps(dict(saved_models_verified=len(audits),max_reload_error=max(r['max_expected_mse_difference'] for r in audits),oracle_mse=[dict(depth=r['depth'],mse=r['expected_mse'],sign=r['gaussian_sign_accuracy']) for r in rows]),indent=2))
